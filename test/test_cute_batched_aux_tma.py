from __future__ import annotations

import ast
from contextlib import contextmanager
from itertools import starmap
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.aux_tensor import Tcgen05AuxTensorDescriptor
from helion._compiler.cute.aux_tensor import batched_aux_tma_layout_supported
from helion._compiler.cute.aux_tensor import batched_aux_tma_output
from helion._compiler.cute.aux_tensor import batched_aux_tma_smem_bytes
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.runtime.cute import launcher

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_aux(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batches, m, k = lhs.shape
    n = rhs.shape[2]
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        out[bi, mi, ni] = (acc * scale[bi, mi, ni] + bias[bi, mi, ni]).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rank_two_aux(
    lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.empty_like(bias)
    for mi, ni in hl.tile([m, n]):
        acc = hl.zeros([mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.addmm(acc, lhs[mi, ki], rhs[ki, ni])
        out[mi, ni] = (acc + bias[mi, ni]).to(out.dtype)
    return out


def _config(**overrides: Any) -> helion.Config:
    values: dict[str, Any] = {
        "block_sizes": [64, 256, 64],
        "num_warps": 8,
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_warp_spec_c_input_warps": 1,
        "tcgen05_aux_load_mode": "tma",
    }
    return helion.Config(**(values | overrides))


def _inputs(
    *,
    batches: int = 3,
    m: int = 128,
    n: int = 512,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((batches, m, k), (batches, k, n), *((batches, m, n),) * 3)
    )


@contextmanager
def _cpu_codegen():
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _code(args: tuple[torch.Tensor, ...], config: helion.Config) -> str:
    with _cpu_codegen():
        _batched_aux.reset()
        return _batched_aux.bind(args).to_code(config)


@pytest.mark.parametrize("order", [[0, 1, 2], [1, 2, 0]])
@pytest.mark.parametrize("tile", [[64, 256, 64], [128, 128, 64]])
def test_batched_aux_tma_codegen(order: list[int], tile: list[int]) -> None:
    code = _code(_inputs(), _config(block_sizes=tile, loop_orders=[order]))
    assert "PipelineTmaAsync.create" in code
    assert "tcgen05_aux_gmem_tile_0" in code
    assert "'c_leading_passthrough': True" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in code
    assert "tile_offset_0 = pid_" in code
    assert "tcgen05_aux_tile_m = tile_offset_1 //" in code
    assert "tcgen05_aux_tile_n = tile_offset_2 //" in code
    assert ", tile_offset_0))[None, None, 0]" in code
    assert "producer_tail" in code
    ast.parse(code)


@pytest.mark.parametrize("field,value", [("m", 127), ("n", 511), ("k", 127)])
def test_batched_aux_tma_rejects_tails(field: str, value: int) -> None:
    shape: dict[str, Any] = {field: value}
    with pytest.raises((exc.BackendUnsupported, exc.InvalidConfig)):
        _code(_inputs(**shape), _config())


def test_simt_batched_aux_stays_unstaged() -> None:
    code = _code(_inputs(), _config(tcgen05_aux_load_mode="simt"))
    assert "'c_leading_passthrough': True" not in code
    assert "tcgen05_aux_gmem_tile_0" not in code


def _runtime_kernel() -> SimpleNamespace:
    return SimpleNamespace(
        _helion_cute_wrapper_plans=[
            {
                "kind": "tcgen05_aux_tma",
                "c_idx": 0,
                "d_idx": 1,
                "c_leading_passthrough": True,
            }
        ]
    )


def test_batched_aux_layout_and_runtime_alias_guards() -> None:
    source, output = (
        torch.empty((3, 128, 256), dtype=torch.bfloat16) for _ in range(2)
    )
    kernel = _runtime_kernel()
    launcher._validate_batched_aux_tma_arguments(kernel, (source, output))
    with pytest.raises(exc.BackendUnsupported, match="overlap"):
        launcher._validate_batched_aux_tma_arguments(
            kernel, (source, source.view_as(source))
        )
    backing = torch.empty(source.numel() + 1, dtype=source.dtype)
    unaligned = backing[1:].view_as(source)
    with pytest.raises(exc.BackendUnsupported, match="aligned"):
        launcher._validate_batched_aux_tma_arguments(kernel, (unaligned, output))
    padded = torch.empty((3, 128, 272), dtype=source.dtype)[:, :, :256]
    assert batched_aux_tma_layout_supported(padded)
    launcher._validate_batched_aux_tma_arguments(kernel, (padded, output))
    assert not batched_aux_tma_layout_supported(source.transpose(1, 2))
    assert not batched_aux_tma_layout_supported(source.expand(2, *source.shape))


def test_batched_aux_guard_precedes_cached_launches() -> None:
    source = torch.empty((3, 128, 256), dtype=torch.bfloat16)
    kernel = _runtime_kernel()
    with (
        patch.object(
            launcher,
            "_cute_last_launch_cache_entry",
            side_effect=AssertionError("cache reached"),
        ),
        pytest.raises(exc.BackendUnsupported, match="overlap"),
    ):
        launcher.default_cute_launcher(kernel, (1,), source, source, block=(256,))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_batched_aux_single_batch_and_padded_stride(dtype: torch.dtype) -> None:
    args = list(_inputs(batches=1, dtype=dtype))
    for index in (2, 3, 4):
        args[index] = torch.empty((1, 128, 528), dtype=dtype)[:, :, :512]
    code = _code(tuple(args), _config())
    assert "'c_leading_passthrough': True" in code


@pytest.mark.parametrize("mode", ["dtype", "transpose", "expanded"])
def test_batched_aux_unsupported_sources(mode: str) -> None:
    args = list(_inputs())
    if mode == "dtype":
        args[2] = args[2].float()
    elif mode == "transpose":
        args[2] = torch.empty((3, 512, 128), dtype=args[2].dtype).transpose(1, 2)
    else:
        args[2] = torch.empty((1, 128, 512), dtype=args[2].dtype).expand(3, -1, -1)
    with pytest.raises((exc.BackendUnsupported, exc.InvalidConfig)):
        _code(tuple(args), _config())


def test_batched_aux_aliased_binding_keeps_runtime_guard() -> None:
    # Input fake-tensor normalization may detach alias identity. Admission is
    # therefore not the alias proof: the launch guard must remain in metadata.
    args = list(_inputs())
    args[2] = args[-1]
    code = _code(tuple(args), _config())
    assert "'c_leading_passthrough': True" in code
    assert "'d_idx':" in code


@pytest.mark.parametrize(
    "changes",
    [
        {"tcgen05_warp_spec_c_input_warps": 0},
        {"pid_type": "flat", "tcgen05_persistence_model": "non_persistent"},
    ],
)
def test_batched_aux_unsupported_schedulers(changes: dict[str, Any]) -> None:
    with pytest.raises((exc.BackendUnsupported, exc.InvalidConfig)):
        _code(_inputs(), _config(**changes))


def test_batched_aux_cluster_normalization_remains_single_cta() -> None:
    code = _code(_inputs(), _config(tcgen05_cluster_m=2, block_sizes=[128, 256, 64]))
    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in code
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" not in code


def test_batched_aux_search_keeps_independent_simt_option() -> None:
    with _cpu_codegen():
        _batched_aux.reset()
        bound = _batched_aux.bind(_inputs())
        state = bound.config_spec._cute_tcgen05_config
        assert state._batched_aux_tma_search_enabled()
        assert "tcgen05_aux_load_mode" in state.aux_load_mode_autotune_fragments()
        all_tma_seeds = [
            seed
            for seed in state.autotune_seed_configs()
            if seed.config.get("tcgen05_aux_load_mode") == "tma"
        ]
        seeds = [
            seed
            for seed in all_tma_seeds
            if not seed.config.get("tcgen05_aux_role_local_scheduler")
        ]
        assert all_tma_seeds[: len(seeds)] == seeds
        original_seeds = [
            seed
            for seed in seeds
            if seed.config.get("tcgen05_consumer_regs", 256) == 256
        ]
        low_cap_seeds = [
            seed for seed in seeds if seed.config.get("tcgen05_consumer_regs") == 128
        ]
        assert len(original_seeds) == len(low_cap_seeds) == 3
        assert seeds == original_seeds + low_cap_seeds
        for seed in all_tma_seeds:
            config = dict(seed.config)
            state._fix_aux_tma_search_config(config)
            assert config["tcgen05_aux_load_mode"] == "tma"
            config["tcgen05_aux_load_mode"] = "simt"
            state._fix_aux_tma_search_config(config)
            assert config["tcgen05_aux_load_mode"] == "simt"
            config["tcgen05_aux_load_mode"] = "tma"
            config["pid_type"] = "flat"
            state._fix_aux_tma_search_config(config)
            assert config["tcgen05_aux_load_mode"] == "simt"


def test_batched_aux_shared_memory_budget() -> None:
    small = batched_aux_tma_smem_bytes(
        bm=128, bn=256, bk=64, ab_stages=2, c_stages=2, aux_stages=2, aux_count=2
    )
    large = batched_aux_tma_smem_bytes(
        bm=128, bn=256, bk=64, ab_stages=2, c_stages=2, aux_stages=2, aux_count=16
    )
    assert small < 232448 < large
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=small - 1
        ),
    ):
        _batched_aux.reset()
        with pytest.raises(exc.BackendUnsupported, match="shared memory"):
            _batched_aux.bind(_inputs()).to_code(_config(block_sizes=[128, 256, 64]))


def test_rank_two_aux_codegen_unchanged() -> None:
    args = tuple(torch.empty((128, 128), dtype=torch.bfloat16) for _ in range(3))
    with _cpu_codegen():
        _rank_two_aux.reset()
        code = _rank_two_aux.bind(args).to_code(_config(block_sizes=[64, 128, 64]))
    assert "tcgen05_aux_tile_m = tile_offset_0 //" in code
    assert "tcgen05_aux_tile_n = tile_offset_1 //" in code
    assert "(64, 128), (tcgen05_aux_tile_m, tcgen05_aux_tile_n))" in code
    assert "'c_leading_passthrough': True" not in code


@pytest.mark.parametrize("change", ["none", "mask", "index", "second_output", "alias"])
def test_batched_aux_output_effect_proof(change: str) -> None:
    graph = torch.fx.Graph()
    source = graph.call_function(_tracing_ops._host_tensor, ("source",))
    target = graph.call_function(_tracing_ops._host_tensor, ("target",))
    source.meta["val"] = torch.empty((2, 64, 64), dtype=torch.bfloat16)
    target.meta["val"] = torch.empty_like(source.meta["val"])
    indices = [graph.placeholder(name) for name in ("b", "m", "n")]
    value = graph.call_function(memory_ops.load, (source, indices, None))
    descriptor = Tcgen05AuxTensorDescriptor(
        value, source, source.meta["val"], None, value
    )
    store_indices = indices if change != "index" else indices[::-1]
    graph.call_function(
        memory_ops.store,
        (target, store_indices, value, value if change == "mask" else None),
    )
    if change == "second_output":
        other = graph.call_function(_tracing_ops._host_tensor, ("other",))
        other.meta["val"] = torch.empty_like(source.meta["val"])
        graph.call_function(memory_ops.store, (other, indices, value, None))
    elif change == "alias":
        target.meta["val"] = source.meta["val"].view_as(source.meta["val"])
    codegen = SimpleNamespace(codegen_graphs=[SimpleNamespace(graph=graph)])
    result = batched_aux_tma_output(cast("Any", codegen), (descriptor,))
    assert (result is target.meta["val"]) is (change == "none")


@pytest.mark.parametrize(
    "batches,rows,cols,row_stride,batch_stride",
    [
        (3, 256, 384, 400, 256 * 400 + 32),
        (8192, 128, 4096, 4096, 128 * 4096),
    ],
)
def test_batched_aux_tensormap_cpu_ir(
    batches: int, rows: int, cols: int, row_stride: int, batch_stride: int
) -> None:
    import cutlass
    import cutlass.cute as cute

    from helion._compiler.cute._mlir_compat import ir

    body: list[str] = []
    call_args: list[str] = []
    launcher._append_cute_wrapper_plan(
        body,
        call_args,
        {
            "kind": "tcgen05_aux_tma",
            "c_idx": 0,
            "bm": 128,
            "bn": 128,
            "stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "kernel_args": ["atom", "tensor"],
            "c_leading_passthrough": True,
        },
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            source = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout(
                    (batches, rows, cols), stride=(batch_stride, row_stride, 1)
                ),
            )
            namespace: dict[str, Any] = {
                "cutlass": cutlass,
                "cute": cute,
                "arg0": source,
                "arg0_shape0": batches,
                "arg0_shape1": rows,
                "arg0_shape2": cols,
                "arg0_stride0": batch_stride,
                "arg0_stride1": row_stride,
                "arg0_stride2": 1,
            }
            exec("\n".join(line[4:] for line in body), namespace)
            permuted = namespace["atom_c_tma"]
            assert (
                cute.crd2idx((rows - 1, cols - 1, batches - 1), permuted.layout)
                == (batches - 1) * batch_stride + (rows - 1) * row_stride + cols - 1
            )
            tile = cute.local_tile(
                namespace["tensor"], (128, 128, 1), (0, 0, batches - 1)
            )[None, None, 0]
            assert cute.rank(tile) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TCgen05 GPU")
@pytest.mark.parametrize("tile", [[64, 256, 64], [128, 128, 64]])
@pytest.mark.parametrize("order", [[0, 1, 2], [1, 2, 0]])
@pytest.mark.parametrize("wait_mode", ["normal", "warp_leader"])
def test_batched_aux_tma_runtime(
    tile: list[int], order: list[int], wait_mode: str
) -> None:
    for seed in range(5):
        torch.manual_seed(seed)
        # 320 work tiles exceed the GB300 SM count, so producer/consumer state
        # crosses scheduler iterations as well as batch boundaries.
        args = tuple(
            torch.randn_like(tensor, device=DEVICE) * 0.1
            for tensor in _inputs(batches=5, m=256, n=4096)
        )
        lhs, rhs, bias, scale, out = args
        before = tuple(tensor.clone() for tensor in args[:-1])
        expected = (
            torch.bmm(lhs.double(), rhs.double()) * scale.double() + bias.double()
        )
        bound = _batched_aux.bind(args)
        bound.set_config(
            _config(
                block_sizes=tile,
                loop_orders=[order],
                tcgen05_sched_consumer_wait_mode=wait_mode,
            )
        )
        bound(*args)
        first = out.clone()
        bound(*args)
        assert torch.equal(out, first)
        for _ in range(3):
            out.fill_(float("nan"))
            bound(*args)
            assert torch.equal(out, first)
        torch.testing.assert_close(out.double(), expected, atol=0.002, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bound(*args)
        graph.replay()
        assert torch.equal(out, first)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out, first)
        assert all(starmap(torch.equal, zip(args[:-1], before, strict=True)))
