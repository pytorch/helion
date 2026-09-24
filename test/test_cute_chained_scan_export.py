from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tcgen05 import _compile
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_scan_export import requests_scan_export
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import scan_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_export(
    a: torch.Tensor,
    b: torch.Tensor,
    delta: torch.Tensor,
    coefficient: torch.Tensor,
    mode: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    side = torch.empty(
        (batch + (1 if mode == "shape" else 0),),
        dtype=torch.bfloat16 if mode == "dtype" else torch.float32,
        device=a.device,
    )
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        product = delta[bi, kk].float().clamp(min=0) * coefficient[bi].float()
        if mode == "dependent":
            product = product + row.begin * 0.01
        gc = hl.cumsum(product, dim=0)
        left = (a[bi, row, kk].float() * torch.exp(gc)[None, :]).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        gc_alias = gc
        if mode == "pre_narrow":
            gc_alias = gc.to(torch.bfloat16).float()
        last = gc_alias[reduction - 1]
        if mode == "narrow":
            last = last.to(torch.bfloat16).float()
        if mode == "missing":
            hl.store(side, [bi], last, extra_mask=row_begin == 0)
        elif mode == "extra":
            hl.store(
                side,
                [bi],
                last,
                extra_mask=(row_begin == 0) & (col_begin == 0) & (bi == 0),
            )
        elif mode == "wrong":
            hl.store(side, [bi], last, extra_mask=(row_begin == 128) & (col_begin == 0))
        elif mode == "none":
            hl.store(side, [bi], last)
        elif mode == "offset":
            hl.store(
                side, [bi + 1], last, extra_mask=(row_begin == 0) & (col_begin == 0)
            )
        elif mode == "dynamic":
            hl.store(side, [bi], gc[bi], extra_mask=(row_begin == 0) & (col_begin == 0))
        elif mode == "duplicate":
            hl.store(
                side,
                [bi],
                last,
                extra_mask=(row_begin == 0) & (col_begin == 0) & (row_begin == 0),
            )
        else:
            hl.store(side, [bi], last, extra_mask=(row_begin == 0) & (col_begin == 0))
    return out, side


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_scan_export(a, b, delta, side):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        gc = hl.cumsum(delta[bi, kk].float(), dim=0)
        left = (a[bi, row, kk].float() * torch.exp(gc)[None, :]).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(
            side,
            [bi],
            gc[reduction - 1],
            extra_mask=(row_begin == 0) & (col_begin == 0),
        )
    return out, side


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _projected_scan_export(a, b, delta, mode: hl.constexpr):
    batch, heads, rows, reduction = a.shape
    columns = b.shape[-1]
    out = torch.empty(
        (batch, heads, rows, columns), dtype=torch.float32, device=a.device
    )
    side = torch.empty(
        (heads, batch) if mode == "permuted" else (batch, heads),
        dtype=torch.float32,
        device=a.device,
    )
    for bt, ht, row, col in hl.tile(
        [batch, heads, rows, columns],
        block_size=[None if mode == "blocked" else 1, 1, None, None],
    ):
        bi: Any = bt.begin
        hi: Any = ht.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        product = delta[bi, hi, kk].float()
        if mode == "read_side":
            product = product + side[bi, hi] * 0
        elif mode == "read_primary":
            product = product + out[bi, hi, 0, 0] * 0
        gc = hl.cumsum(product, dim=0)
        left = (a[bi, hi, row, kk].float() * torch.exp(gc)[None, :]).to(a.dtype)
        out[bi, hi, row, col] = hl.dot(left, b[bi, hi, kk, col])
        last = gc[reduction - 1]
        if mode == "independent_scan":
            other = hl.cumsum(product * 2, dim=0)
            last = other[reduction - 1]
        elif mode == "invalid_index":
            last = gc[reduction]
        if mode == "permuted":
            hl.store(
                side, [hi, bi], last, extra_mask=(row_begin == 0) & (col_begin == 0)
            )
        elif mode == "or_mask":
            hl.store(
                side, [bi, hi], last, extra_mask=(row_begin == 0) | (col_begin == 0)
            )
        else:
            hl.store(
                side, [bi, hi], last, extra_mask=(row_begin == 0) & (col_begin == 0)
            )
    return out, side


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _aliased_scan_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    side = out[:, 0, 0]
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        gc = hl.cumsum(delta[bi, kk].float(), dim=0)
        left = (a[bi, row, kk].float() * torch.exp(gc)[None, :]).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(
            side,
            [bi],
            gc[reduction - 1],
            extra_mask=(row_begin == 0) & (col_begin == 0),
        )
    return out, side


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unrelated_scan_scalar(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    side = torch.empty((batch,), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        gc = hl.cumsum(delta[bi, kk].float(), dim=0)
        left = (a[bi, row, kk].float() * torch.exp(gc[reduction - 1] - gc)[None, :]).to(
            a.dtype
        )
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(
            side,
            [bi],
            delta[bi, 0].float(),
            extra_mask=(row_begin == 0) & (col_begin == 0),
        )
    return out, side


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nonzero_root_scan_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    side = torch.empty((batch,), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile(
        [0, 128, 0], [batch, rows, columns], block_size=[1, None, None]
    ):
        bi: Any = bt.begin
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        kk = hl.arange(reduction)
        gc = hl.cumsum(delta[bi, kk].float(), dim=0)
        left = (a[bi, row, kk].float() * torch.exp(gc)[None, :]).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(
            side,
            [bi],
            gc[reduction - 1],
            extra_mask=(row_begin == 0) & (col_begin == 0),
        )
    return out, side


def _args(
    dtype: torch.dtype = torch.bfloat16, kind: str = "dense"
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(625)
    a = torch.randn((2, 256, 128), generator=generator, dtype=dtype) * 0.05
    b = torch.randn((2, 128, 128), generator=generator, dtype=dtype) * 0.05
    delta = torch.rand((2, 128), generator=generator, dtype=dtype) * 0.01
    coefficient = -torch.rand((2,), generator=generator, dtype=torch.float32)
    if kind == "short":
        delta = delta[:, :127].contiguous()
    elif kind == "offset":
        offset = 16 // delta.element_size()
        backing = torch.empty(delta.numel() + offset, dtype=dtype)
        backing[offset:].copy_(delta.flatten())
        delta = backing[offset:].view(delta.shape)
    elif kind == "stride":
        delta = torch.stack((delta, delta), dim=-1)[..., 0]
    return a, b, delta, coefficient


def _config(n: int = 64, schedule: str = "tcgen05_tmem") -> helion.Config:
    return helion.Config(
        block_sizes=[128, n],
        num_warps=4,
        pid_type="flat",
        cute_chained_mma_schedule=schedule,
    )


@contextmanager
def _cpu_codegen() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=152),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _code(
    mode: str = "normal",
    dtype: torch.dtype = torch.bfloat16,
    n: int = 64,
    kind: str = "dense",
) -> str:
    with _cpu_codegen():
        return _scan_export._bind_isolated((*_args(dtype, kind), mode)).to_code(
            _config(n)
        )


def _export_branch(code: str) -> ast.If:
    matches = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If)
        and len(node.body) == 1
        and "chain_scan_0_values[127]" in ast.unparse(node.body[0])
        and ".store(" in ast.unparse(node.body[0])
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 128])
def test_scan_scalar_export_codegen(dtype: torch.dtype, n: int) -> None:
    code = _code(dtype=dtype, n=n)
    branch = _export_branch(code)
    comparisons = [
        ast.unparse(node)
        for node in ast.walk(branch.test)
        if isinstance(node, ast.Compare)
    ]
    assert comparisons == [
        "chain_origin_1 == 0",
        "chain_origin_2 == 0",
        "chain_thread == 0",
    ]
    assert "cutlass.Float32(chain_scan_0_values[127])" in ast.unparse(branch)
    assert "sync" not in ast.unparse(branch)
    assert (
        code.index("chain_epi_values")
        < code.index(ast.unparse(branch.test))
        < code.index("chain_allocator.free")
    )
    assert code.count("chain_scan_0_pointer =") == 1


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "extra",
        "wrong",
        "none",
        "offset",
        "dynamic",
        "shape",
        "dtype",
        "narrow",
        "dependent",
    ],
)
def test_scan_scalar_export_rejects_unproven_cases(mode: str) -> None:
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _code(mode)


@pytest.mark.parametrize("kind", ["offset", "stride"])
def test_scan_export_preserves_valid_input_views(kind: str) -> None:
    _export_branch(_code(kind=kind))


def test_scan_export_rejects_short_scan_leaf() -> None:
    with pytest.raises(BackendUnsupported, match="scan export"):
        _code(kind="short")


@pytest.mark.parametrize("schedule", ["coalesced", "cp_async_register_reuse_scan"])
def test_scan_export_does_not_fall_back_to_ordinary_store(schedule: str) -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), "normal"))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(_config(schedule=schedule))


def test_scan_export_requires_fresh_side_output() -> None:
    with _cpu_codegen():
        a, b, delta, coefficient = _args()
        bound = _external_scan_export._bind_isolated((a, b, delta, coefficient))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_config())


def test_tracing_normalizes_redundant_mask_clause() -> None:
    # SymBool normalization removes the repeated clause before admission.
    # The compiler still validates the complete canonical first-tile mask.
    _export_branch(_code("duplicate"))


def _projected_args() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(514)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.01
        for shape in ((2, 2, 256, 128), (2, 2, 128, 128), (2, 2, 128))
    )


def test_scan_export_permuted_equal_extent_axes() -> None:
    with _cpu_codegen():
        code = _projected_scan_export._bind_isolated(
            (*_projected_args(), "permuted")
        ).to_code(_config())
    branch = _export_branch(code)
    assert [
        ast.unparse(node)
        for node in ast.walk(branch.test)
        if isinstance(node, ast.Compare)
    ] == ["chain_origin_2 == 0", "chain_origin_3 == 0", "chain_thread == 0"]
    assert "cutlass.Int32(chain_origin_1)" in ast.unparse(branch.body[0])


@pytest.mark.parametrize(
    "mode",
    [
        "read_side",
        "read_primary",
        "independent_scan",
        "invalid_index",
        "or_mask",
    ],
)
def test_scan_export_projection_alias_and_scan_rejections(mode: str) -> None:
    with _cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        _projected_scan_export._bind_isolated((*_projected_args(), mode)).to_code(
            _config()
        )


def test_scan_export_blocked_retained_axis_rejects() -> None:
    with _cpu_codegen():
        bound = _projected_scan_export._bind_isolated((*_projected_args(), "blocked"))
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(
                helion.Config(
                    block_sizes=[2, 128, 64],
                    num_warps=4,
                    cute_chained_mma_schedule="tcgen05_tmem",
                )
            )


def test_scan_export_alias_rejects() -> None:
    with _cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        _aliased_scan_export._bind_isolated(_args()[:3]).to_code(_config())


def test_scan_export_classifier_leaves_unrelated_multioutput_alone() -> None:
    with _cpu_codegen():
        bound = _unrelated_scan_scalar._bind_isolated(_args()[:3])
        assert bound.host_function is not None
        roots = [
            graph
            for graph in bound.host_function.device_ir.graphs
            if isinstance(graph, RootGraphInfo)
        ]
        assert len(roots) == 1
        assert not requests_scan_export(tuple(roots[0].graph.nodes))


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "extra",
        "wrong",
        "none",
        "offset",
        "dynamic",
        "narrow",
        "pre_narrow",
        "shape",
        "dtype",
        "dependent",
    ],
)
def test_scan_export_default_config_fail_closed(mode: str) -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), mode))
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


def test_scan_export_pre_subscript_cast_is_recognized_not_admitted() -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), "pre_narrow"))
        assert bound.host_function is not None
        roots = [
            graph
            for graph in bound.host_function.device_ir.graphs
            if isinstance(graph, RootGraphInfo)
        ]
        assert len(roots) == 1
        assert requests_scan_export(tuple(roots[0].graph.nodes))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_config())


@pytest.mark.parametrize("mode", ["normal", "missing", "narrow", "duplicate"])
def test_scan_export_classifier_canonicalizes_identity_alias(mode: str) -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), mode))
        assert bound.host_function is not None
        roots = [
            graph
            for graph in bound.host_function.device_ir.graphs
            if isinstance(graph, RootGraphInfo)
        ]
        assert len(roots) == 1
        nodes = tuple(roots[0].graph.nodes)
        scans = {node for node in nodes if node.target is scan_ops._associative_scan}
        aliases = {
            node
            for node in nodes
            if node.target is _tracing_ops._new_var and node.args[0] in scans
        }
        # The real frontend retains gc_alias as _new_var before subscript.
        assert aliases
        assert any(
            node.target is view_ops.subscript and node.args[0] in aliases
            for node in nodes
        )
        assert requests_scan_export(tuple(roots[0].graph.nodes))


def test_scan_export_nonzero_root_rejects() -> None:
    with _cpu_codegen():
        a, b, delta, _coefficient = _args()
        bound = _nonzero_root_scan_export._bind_isolated((a, b, delta))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_config())


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("padding", [0, 4])
def test_scan_export_cache_padding_and_seed_reachability(
    cache: bool, padding: int
) -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), "normal"))
        config = _config()
        config.config.update(
            cute_chained_auxiliary_cache=cache, cute_chained_c_smem_padding=padding
        )
        _export_branch(bound.to_code(config))
        assert bound.host_function is not None
        with bound.env:
            seeds = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        assert seeds is not None
        tcgen = [
            seed
            for seed in seeds
            if seed.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
        ]
        assert tcgen
        assert any("chain_scan_0_values[127]" in bound.to_code(seed) for seed in tcgen)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scan_export_offline_compile(dtype: torch.dtype, tmp_path: Path) -> None:
    repository = str(Path(__file__).resolve().parent.parent)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
        "CUTE_DSL_DUMP_DIR": str(tmp_path),
        "CUTE_DSL_CACHE_DIR": str(tmp_path / "cute-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        "PYTHONPATH": os.pathsep.join((repository, os.environ.get("PYTHONPATH", ""))),
    }
    result = subprocess.run(
        [sys.executable, "-B", "-m", __name__, str(dtype).split(".")[-1]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scan_export_runtime_multi_tile_ownership(dtype: torch.dtype) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("TCgen05 FP16/BF16 required")
    # 48 batches x two M tiles x two N tiles = 192 distinct CTAs.
    for seed in range(5):
        generator = torch.Generator(device=DEVICE).manual_seed(4001 + seed)
        a = (
            torch.randn((48, 256, 128), device=DEVICE, dtype=dtype, generator=generator)
            * 0.05
        )
        b = (
            torch.randn((48, 128, 128), device=DEVICE, dtype=dtype, generator=generator)
            * 0.05
        )
        delta = (
            torch.rand((48, 128), device=DEVICE, dtype=dtype, generator=generator)
            * 0.01
        )
        coefficient = -torch.rand(
            (48,), device=DEVICE, dtype=torch.float32, generator=generator
        )
        args = (a, b, delta, coefficient)
        saved = tuple(value.clone() for value in args)
        product = delta.float().clamp(min=0) * coefficient[:, None]
        prefix = product.cumsum(-1)
        weighted = (a.float() * prefix.exp()[:, None, :]).to(dtype)
        expected = weighted.double() @ b.double()
        compiled = _scan_export._bind_isolated((*args, "normal")).compile_config(
            _config()
        )
        first = compiled(*args, "normal")
        copies = tuple(value.clone() for value in first)
        second = compiled(*args, "normal")
        for actual, before in zip(second, copies, strict=True):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)
        assert first[0].data_ptr() != second[0].data_ptr()
        assert first[1].data_ptr() != second[1].data_ptr()
        torch.testing.assert_close(first[0].double(), expected, atol=0.005, rtol=0.02)
        torch.testing.assert_close(first[1], prefix[:, -1], atol=2e-6, rtol=2e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*args, "normal")
        graph.replay()
        for actual, before in zip(captured, copies, strict=True):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)
        for _repeat in range(3):
            for value in captured:
                value.fill_(float("nan"))
            graph.replay()
            for actual, before in zip(captured, copies, strict=True):
                torch.testing.assert_close(actual, before, atol=0, rtol=0)
        for actual, before in zip(args, saved, strict=True):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)


if __name__ == "__main__":
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        ptx = _compile(_code(dtype=dtype), _args(dtype), "normal", entry="_scan_export")
    assert "st.global" in ptx
    assert "tcgen05.mma" in ptx
    assert not torch.cuda.is_initialized()
