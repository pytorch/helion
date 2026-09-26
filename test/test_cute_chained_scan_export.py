from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test.test_cute_chained_tcgen05 import _tcgen_compile

import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_scan_export import requests_scan_export
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import scan_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


# Scan export.


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


def _scan_export_args(
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


def _scan_export_config(n: int = 64, schedule: str = "tcgen05_tmem") -> helion.Config:
    return helion.Config(
        block_sizes=[128, n],
        num_warps=4,
        pid_type="flat",
        cute_chained_mma_schedule=schedule,
    )


@contextmanager
def _scan_export_cpu_codegen() -> Iterator[None]:
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


def _scan_export_code(
    mode: str = "normal",
    dtype: torch.dtype = torch.bfloat16,
    n: int = 64,
    kind: str = "dense",
) -> str:
    with _scan_export_cpu_codegen():
        return _scan_export._bind_isolated(
            (*_scan_export_args(dtype, kind), mode)
        ).to_code(_scan_export_config(n))


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
    code = _scan_export_code(dtype=dtype, n=n)
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
        _scan_export_code(mode)


@pytest.mark.parametrize("kind", ["offset", "stride"])
def test_scan_export_preserves_valid_input_views(kind: str) -> None:
    _export_branch(_scan_export_code(kind=kind))


def test_scan_export_rejects_short_scan_leaf() -> None:
    with pytest.raises(BackendUnsupported, match="scan export"):
        _scan_export_code(kind="short")


@pytest.mark.parametrize("schedule", ["coalesced", "cp_async_register_reuse_scan"])
def test_scan_export_does_not_fall_back_to_ordinary_store(schedule: str) -> None:
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), "normal"))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(_scan_export_config(schedule=schedule))


def test_scan_export_requires_fresh_side_output() -> None:
    with _scan_export_cpu_codegen():
        a, b, delta, coefficient = _scan_export_args()
        bound = _external_scan_export._bind_isolated((a, b, delta, coefficient))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_scan_export_config())


def test_tracing_normalizes_redundant_mask_clause() -> None:
    # SymBool normalization removes the repeated clause before admission.
    # The compiler still validates the complete canonical first-tile mask.
    _export_branch(_scan_export_code("duplicate"))


def _projected_args() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(514)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.01
        for shape in ((2, 2, 256, 128), (2, 2, 128, 128), (2, 2, 128))
    )


def test_scan_export_permuted_equal_extent_axes() -> None:
    with _scan_export_cpu_codegen():
        code = _projected_scan_export._bind_isolated(
            (*_projected_args(), "permuted")
        ).to_code(_scan_export_config())
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
    with _scan_export_cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        _projected_scan_export._bind_isolated((*_projected_args(), mode)).to_code(
            _scan_export_config()
        )


def test_scan_export_blocked_retained_axis_rejects() -> None:
    with _scan_export_cpu_codegen():
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
    with _scan_export_cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        _aliased_scan_export._bind_isolated(_scan_export_args()[:3]).to_code(
            _scan_export_config()
        )


def test_scan_export_classifier_leaves_unrelated_multioutput_alone() -> None:
    with _scan_export_cpu_codegen():
        bound = _unrelated_scan_scalar._bind_isolated(_scan_export_args()[:3])
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
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), mode))
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


def test_scan_export_pre_subscript_cast_is_recognized_not_admitted() -> None:
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), "pre_narrow"))
        assert bound.host_function is not None
        roots = [
            graph
            for graph in bound.host_function.device_ir.graphs
            if isinstance(graph, RootGraphInfo)
        ]
        assert len(roots) == 1
        assert requests_scan_export(tuple(roots[0].graph.nodes))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_scan_export_config())


@pytest.mark.parametrize("mode", ["normal", "missing", "narrow", "duplicate"])
def test_scan_export_classifier_canonicalizes_identity_alias(mode: str) -> None:
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), mode))
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
    with _scan_export_cpu_codegen():
        a, b, delta, _coefficient = _scan_export_args()
        bound = _nonzero_root_scan_export._bind_isolated((a, b, delta))
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(_scan_export_config())


@pytest.mark.parametrize("cache", [False, True])
def test_scan_export_cache_and_seed_reachability(cache: bool) -> None:
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), "normal"))
        config = _scan_export_config()
        config.config.update(cute_chained_auxiliary_cache=cache)
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
        [
            sys.executable,
            "-B",
            "-m",
            __name__,
            "scan_export",
            str(dtype).split(".")[-1],
        ],
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
            _scan_export_config()
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


# Vector export.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _vectors(a, b, delta, coefficient, mode: str):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    terminal = torch.empty((batch,), dtype=torch.float32, device=a.device)
    prefix = torch.empty(
        (reduction, batch)
        if mode == "permuted"
        else (batch, reduction + int(mode == "shape")),
        dtype=torch.bfloat16 if mode == "dtype" else torch.float32,
        device=a.device,
    )
    rawdt = torch.empty((batch, reduction), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, kk]
        product = raw.float().clamp(min=0) * coefficient[bi].float()
        if mode == "dependent":
            product = product + rb * 0.01
        scan = hl.cumsum(product, dim=0)
        left = (a[bi, row, kk].float() * torch.exp(scan)[None, :] * raw[None, :]).to(
            a.dtype
        )
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(terminal, [bi], scan[reduction - 1], extra_mask=(rb == 0) & (cb == 0))
        exported = scan
        if mode == "narrow":
            exported = exported.to(torch.bfloat16).float()
        if mode == "permuted":
            hl.store(prefix, [kk, bi], exported, extra_mask=(rb == 0) & (cb == 0))
        elif mode == "missing":
            hl.store(prefix, [bi, kk], exported, extra_mask=rb == 0)
        elif mode == "offset":
            hl.store(prefix, [bi, kk + 1], exported, extra_mask=(rb == 0) & (cb == 0))
        else:
            hl.store(prefix, [bi, kk], exported, extra_mask=(rb == 0) & (cb == 0))
        raw_export = raw
        if mode == "computed":
            raw_export = raw * 2
        hl.store(rawdt, [bi, kk], raw_export, extra_mask=(rb == 0) & (cb == 0))
    return out, terminal, prefix, rawdt


def _vector_export_args(
    dtype: torch.dtype = torch.bfloat16, k: int = 128, view: str = "dense"
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(4201)
    a = torch.randn((3, 256, k), dtype=dtype, generator=g) * 0.05
    b = torch.randn((3, k, 192), dtype=dtype, generator=g) * 0.05
    raw = torch.rand((3, k), dtype=torch.float32, generator=g)
    if view == "stride":
        raw = torch.stack((raw, raw), dim=-1)[..., 0]
    elif view == "offset":
        backing = torch.empty(raw.numel() + 4, dtype=torch.float32)
        backing[4:].copy_(raw.flatten())
        raw = backing[4:].view(raw.shape)
    elif view == "short":
        raw = raw[:, :-1].contiguous()
    elif view == "narrow":
        raw = raw.to(dtype)
    return a, b, raw, -torch.rand((3,), generator=g)


def _vector_export_code(
    mode: str = "normal",
    dtype: torch.dtype = torch.bfloat16,
    k: int = 128,
    view: str = "dense",
    n: int = 64,
) -> str:
    with _scan_export_cpu_codegen():
        return _vectors._bind_isolated(
            (*_vector_export_args(dtype, k, view), mode)
        ).to_code(_scan_export_config(n))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [32, 64, 128, 256])
def test_vector_exports_source(dtype: torch.dtype, k: int) -> None:
    code = _vector_export_code(dtype=dtype, k=k)
    assert code.count("chain_scan_0_pointer =") == 1
    assert f"chain_scan_0_values[{k - 1}]" in code
    assert "chain_export_1_index < " + str(k) in code
    assert "chain_export_2_index < " + str(k) in code
    assert "chain_scan_0_values[chain_export_1_index]" in code
    assert "chain_origin_1 == 0 and chain_origin_2 == 0" in code
    assert (
        code.index("chain_epi_values")
        < code.index("for chain_export_1_step")
        < code.index("chain_allocator.free")
    )
    ast.parse(code)


@pytest.mark.parametrize("view", ["dense", "stride", "offset"])
@pytest.mark.parametrize("mode", ["normal", "permuted"])
def test_vector_exports_valid_views(mode: str, view: str) -> None:
    _vector_export_code(mode, view=view)


@pytest.mark.parametrize(
    "mode", ["missing", "offset", "shape", "dtype", "narrow", "computed", "dependent"]
)
def test_vector_exports_reject(mode: str) -> None:
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _vector_export_code(mode)


@pytest.mark.parametrize("view", ["short", "narrow"])
def test_vector_exports_leaf_reject(view: str) -> None:
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _vector_export_code(view=view)


def test_vector_exports_padded_scan_reject() -> None:
    # A physical padded scan is not a proven full-vector logical store.
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _vector_export_code(k=48)


@pytest.mark.parametrize("mode", ["missing", "narrow", "computed"])
def test_vector_exports_no_ordinary_fallback(mode: str) -> None:
    with _scan_export_cpu_codegen():
        bound = _vectors._bind_isolated((*_vector_export_args(), mode))
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _consume(a, b, prefix, rawdt):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        kk = hl.arange(reduction)
        left = (
            a[bi, row, kk].float()
            * torch.exp(prefix[bi, kk])[None, :]
            * rawdt[bi, kk][None, :]
        ).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alias_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    prefix = torch.empty((batch, reduction), dtype=torch.float32, device=a.device)
    rawdt = prefix
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, kk]
        scan = hl.cumsum(raw, dim=0)
        out[bi, row, col] = hl.dot(
            (a[bi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, kk, col]
        )
        hl.store(prefix, [bi, kk], scan, extra_mask=(rb == 0) & (cb == 0))
        hl.store(rawdt, [bi, kk], raw, extra_mask=(rb == 0) & (cb == 0))
    return out, prefix, rawdt


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        scan = hl.cumsum(delta[bi, kk], dim=0)
        out[bi, row, col] = hl.dot(
            (a[bi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, kk, col]
        )
        hl.store(delta, [bi, kk], scan, extra_mask=(rb == 0) & (cb == 0))
    return out, delta


@pytest.mark.parametrize("kernel", [_alias_export, _external_export])
def test_vector_exports_storage_alias_reject(kernel: Any) -> None:
    with _scan_export_cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        kernel._bind_isolated(_vector_export_args()[:3]).to_code(_scan_export_config())


def _host(code: str, name: str) -> tuple[Any, list[tuple[Any, ...]]]:
    host = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    launches: list[tuple[Any, ...]] = []

    def launch(*args: Any, **kwargs: Any) -> None:
        launches.append(args)

    namespace: dict[str, Any] = {
        "torch": torch,
        "_default_cute_launcher": launch,
        "_helion_" + name: SimpleNamespace(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[host], type_ignores=[])),
            "<actual-generated-host>",
            "exec",
        ),
        namespace,
    )
    return namespace[name], launches


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_vector_exports_actual_host_fresh_returns_and_consumer(
    dtype: torch.dtype,
) -> None:
    with _scan_export_cpu_codegen():
        args = _vector_export_args(dtype)
        producer = _vectors._bind_isolated((*args, "normal"))
        host, launches = _host(producer.to_code(_scan_export_config()), "_vectors")
        retained: list[torch.Tensor] = []
        for _repeat in range(2):
            current = tuple(value.clone() for value in args)
            outputs = host(*current, "normal")
            assert tuple(value.shape for value in outputs) == (
                (3, 256, 192),
                (3,),
                (3, 128),
                (3, 128),
            )
            assert all(value.dtype is torch.float32 for value in outputs)
            storage = [
                value.untyped_storage().data_ptr()
                for value in (*current, *retained, *outputs)
            ]
            assert len(storage) == len(set(storage))
            assert all(
                launches[-1][i] is value
                for i, value in zip((4, 5, 2, 3), current, strict=True)
            )
            assert all(
                actual is value
                for actual, value in zip(launches[-1][-4:], outputs, strict=True)
            )
            consumer_args = (*current[:2], outputs[2], outputs[3])
            consumer = _consume._bind_isolated(consumer_args)
            code = consumer.to_code(_scan_export_config())
            assert "chain_scan_" not in code
            consumer_host, consumer_calls = _host(code, "_consume")
            result = consumer_host(*consumer_args)
            assert result.dtype is torch.float32
            assert consumer_calls[-1][-1] is result
            assert all(
                any(value is actual for actual in consumer_calls[-1][2:])
                for value in consumer_args
            )
            retained.extend(outputs)


def test_vector_exports_raw_normalized_preload_and_population() -> None:
    class Stop(BaseException):
        pass

    with _scan_export_cpu_codegen():
        bound = _vectors._bind_isolated((*_vector_export_args(), "normal"))
        raw = _scan_export_config()
        normalized = bound._normalized_config_copy(raw)
        expected = bound.to_code(raw)
        assert bound.to_code(normalized) == expected
        calls: list[str] = []

        def stop(source: str, **kwargs: Any) -> None:
            calls.append(source)
            assert source == expected
            raise Stop

        with (
            patch.object(PyCodeCache, "load", side_effect=stop),
            patch.object(type(bound.env.backend), "setup_compile_cache_dir"),
        ):
            for config in (raw, normalized):
                with pytest.raises(Stop):
                    bound.compile_config(config, allow_print=False)
        assert len(calls) == 2
        assert not bound._compile_cache
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            rows = generation.random_population_flat(100)
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast("Any", SimpleNamespace(config_gen=generation)), row
                )
                for row in rows
            ]
        assert len(members) == 100
        eligible = [
            member
            for member in members
            if member is not None
            and member.config.config.get("cute_chained_mma_schedule") == "tcgen05_tmem"
        ]
        assert eligible
        assert "chain_export_1_index" in bound.to_code(eligible[0].config)


class _Pointer:
    def __init__(
        self, values: torch.Tensor, writes: list[int], offset: int = 0
    ) -> None:
        self.values = values
        self.writes = writes
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.values, self.writes, self.offset + offset)

    def store(self, value: torch.Tensor) -> None:
        assert 0 <= self.offset < self.values.numel()
        self.writes[self.offset] += 1
        self.values[self.offset] = value


@pytest.mark.parametrize("k", [32, 128, 256])
@pytest.mark.parametrize("mode", ["normal", "permuted"])
def test_vector_exports_actual_store_ast_bijection_and_bits(k: int, mode: str) -> None:
    code = _vector_export_code(mode, k=k)
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion__vectors"
    )
    stores: list[ast.stmt] = [
        node
        for node in device.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("chain_export_")
    ]
    assert len(stores) == 2
    program = compile(
        ast.fix_missing_locations(ast.Module(body=stores, type_ignores=[])),
        "<actual-export-stores>",
        "exec",
    )
    # Include signed zeros, subnormals, infinities and two distinct NaN payloads.
    bits = torch.tensor(
        [0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2143289351, 1065353216],
        dtype=torch.int32,
    )
    source = bits.repeat((3 * k + 7) // 8)[: 3 * k].view(torch.float32).reshape(3, k)
    shapes = ((k, 3) if mode == "permuted" else (3, k), (3, k))
    outputs = [torch.empty(shape) for shape in shapes]
    counters = [[0] * output.numel() for output in outputs]
    env: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int, Float32=lambda value: value, range_constexpr=range
        )
    }
    for name, output, counter in zip(
        ("input_tensor_6", "input_tensor_7"), outputs, counters, strict=True
    ):
        env[name] = SimpleNamespace(
            iterator=_Pointer(output.view(-1), counter),
            layout=SimpleNamespace(stride=output.stride()),
        )
    for batch in range(3):
        env.update(
            chain_origin_0=batch,
            chain_scan_0_values=source[batch],
            chain_scan_0_input_0=source[batch],
        )
        for row in (0, 128):
            for column in (0, 64, 128):
                for thread in range(128):
                    env.update(
                        chain_origin_1=row, chain_origin_2=column, chain_thread=thread
                    )
                    exec(program, env)
    assert all(count == 1 for counts in counters for count in counts)
    expected = source.T.contiguous() if mode == "permuted" else source
    assert torch.equal(outputs[0].view(torch.int32), expected.view(torch.int32))
    assert torch.equal(outputs[1].view(torch.int32), source.view(torch.int32))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _projected_vectors(a, b, delta, kind: str):
    batch, groups, rows, reduction = a.shape
    columns = b.shape[3]
    out = torch.empty(
        (batch, groups, rows, columns), dtype=torch.float32, device=a.device
    )
    side = torch.empty((groups, reduction, batch), dtype=torch.float32, device=a.device)
    for bt, gt, row, col in hl.tile(
        [batch, groups, rows, columns], block_size=[1, 1, None, None]
    ):
        bi: Any = bt.begin
        gi: Any = gt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, gi, kk]
        scan = hl.cumsum(raw, dim=0)
        out[bi, gi, row, col] = hl.dot(
            (a[bi, gi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, gi, kk, col]
        )
        value = raw
        if kind == "scan":
            value = scan
        if kind == "computed":
            value = raw * 2
        hl.store(side, [gi, kk, bi], value, extra_mask=(rb == 0) & (cb == 0))
    return out, side


@pytest.mark.parametrize("kind", ["scan", "raw"])
def test_vector_exports_multiple_outer_axes_and_vector_middle(kind: str) -> None:
    args = (
        torch.zeros(2, 3, 256, 128, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128, 192, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128),
        kind,
    )
    with _scan_export_cpu_codegen():
        code = _projected_vectors._bind_isolated(args).to_code(_scan_export_config())
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion__projected_vectors"
    )
    loop = next(
        node
        for node in device.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_export_0_step"
    )
    # Extract the actual target pointer arithmetic, not a handwritten owner map.
    store = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    )
    assert isinstance(store.func, ast.Attribute)
    pointer = compile(
        ast.Expression(body=store.func.value), "<actual-projected-pointer>", "eval"
    )
    target = next(
        node.id
        for node in ast.walk(store.func.value)
        if isinstance(node, ast.Name) and node.id.startswith("input_tensor")
    )
    env: dict[str, Any] = {
        "cutlass": SimpleNamespace(Int32=int),
        target: SimpleNamespace(iterator=0, layout=SimpleNamespace(stride=(256, 2, 1))),
    }
    offsets = []
    for bi in range(2):
        for gi in range(3):
            for index in range(128):
                env.update(
                    chain_origin_0=bi, chain_origin_1=gi, chain_export_0_index=index
                )
                offset = eval(pointer, env)
                assert offset == gi * 256 + index * 2 + bi
                offsets.append(offset)
    assert sorted(offsets) == list(range(2 * 3 * 128))
    assert "chain_origin_2 == 0 and chain_origin_3 == 0" in ast.unparse(loop)


def test_vector_exports_raw_only_computed_family_fails_closed() -> None:
    args = (
        torch.zeros(2, 3, 256, 128, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128, 192, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128),
        "computed",
    )
    with _scan_export_cpu_codegen():
        bound = _projected_vectors._bind_isolated(args)
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


@pytest.mark.parametrize("direct", [False, True])
def test_vector_exports_m64_transport(direct: bool) -> None:
    with _scan_export_cpu_codegen():
        config = helion.Config(
            block_sizes=[64, 64],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_direct_output=direct,
        )
        source = _vectors._bind_isolated((*_vector_export_args(), "normal")).to_code(
            config
        )
    assert "Ld16x256bOp" in source
    assert "chain_export_1_index" in source
    assert "chain_export_2_index" in source


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "scan_export":
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ):
            ptx = _tcgen_compile(
                _scan_export_code(dtype=dtype),
                _scan_export_args(dtype),
                "normal",
                entry="_scan_export",
            )
        assert "st.global" in ptx
        assert "tcgen05.mma" in ptx
        assert not torch.cuda.is_initialized()
    else:
        raise AssertionError(f"Unknown test driver: {command}")
