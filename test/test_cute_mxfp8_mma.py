"""Native MXFP8 proof, descriptor-rank, and live-input CUDA coverage."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test.test_cute_block_scaled_mma import _wrapper_plan
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.block_scaled_config import block_scaled_workspace_shape
from helion._compiler.cute.mxfp8_mma import _match_mxfp8
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language.matmul_ops import dot_scaled

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        result = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3", acc=acc
        )
        out[row, col] = result.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mxfp8_transposed(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, _ = a.shape
    n = b.shape[1]
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[:, col], sb[col, :], "e4m3", acc=acc
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partial_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    for row, col in hl.tile([a.shape[0] - 1, b.shape[0]]):
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partial_columns(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    for row, col in hl.tile([a.shape[0], b.shape[0] - 1]):
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nonzero_acc(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        acc = hl.full([row, col], 1.0, dtype=torch.float32)
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3", acc=acc
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bad_operand_row(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        out[row, col] = hl.dot_scaled(
            a[col, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bad_scale_row(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[col, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extra_output(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
    other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        result = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
        out[row, col] = result
        other[row, col] = result
    return out, other


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _multiple_roots(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
    other: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = a.shape
    n = b.shape[0]
    for row, col in hl.tile([m, n]):
        out[row, col] = hl.dot_scaled(
            a[row, :], sa[row, :], "e4m3", b[col, :].T, sb[col, :], "e4m3"
        ).to(out.dtype)
    hl.barrier()
    for row, col in hl.tile([m, n]):
        other[row, col] = out[row, col] + 1
    return out, other


def _config(bound: BoundKernel, **changes: object) -> helion.Config:
    values = bound.config_spec.default_config().config
    values.update(
        cute_scaled_mma=True,
        cute_collective_mma=False,
        cute_scaled_tile_n=128,
        cute_scaled_tile_k=128,
        cute_scaled_stages=3,
        cute_scaled_cluster_m=1,
        cute_scaled_persistent=True,
    )
    values.update(changes)
    return helion.Config.from_dict(values)


def _cpu_inputs(
    shape: tuple[int, int, int], *, typed: bool = False, transposed: bool = False
) -> tuple[torch.Tensor, ...]:
    m, n, k = shape
    a = torch.empty((m, k), dtype=torch.float8_e4m3fn)
    b = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    dtype = torch.float8_e8m0fnu if typed else torch.uint8
    return (
        a,
        b.T if transposed else b,
        torch.empty((m, k // 32), dtype=dtype),
        torch.empty((n, k // 32), dtype=dtype),
        torch.empty((m, n), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("shape", [(1, 129, 32), (73, 1, 160), (137, 273, 288)])
@pytest.mark.parametrize(
    "typed,transposed", [(False, False), (True, False), (False, True)]
)
def test_cpu_native_mxfp8_admission_and_workspace(
    shape: tuple[int, int, int], typed: bool, transposed: bool
) -> None:
    initialized = torch.cuda.is_initialized()
    inputs = _cpu_inputs(shape, typed=typed, transposed=transposed)
    with _cpu_target():
        kernel = _mxfp8_transposed if transposed else _mxfp8
        bound = kernel._bind_isolated(inputs)
        assert bound.config_spec.cute_mxfp8_mma_available
        source = bound.to_code(_config(bound))
        plan = _wrapper_plan(source)
    m, n, k = shape
    mp, np, kp = block_scaled_workspace_shape(m, n, k, 128, 128)
    assert plan["format"] == "mxfp8"
    assert (plan["m"], plan["n"], plan["k"]) == shape
    assert plan["rhs_transposed"] is transposed
    assert plan["workspace_bytes"] == (mp + np) * kp // 32
    assert not plan["lhs_byte_addresses"] and not plan["rhs_byte_addresses"]
    assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("operand", ["lhs", "rhs", "rhs_before_transpose"])
@pytest.mark.parametrize(
    "fill,admitted", [(0.0, True), (-0.0, False), (1.0, False), (float("nan"), False)]
)
def test_cpu_native_mxfp8_operand_masks(
    operand: str, fill: float, admitted: bool
) -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((73, 129, 160)))
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            root = ir.graphs[ir.root_ids[0]]
            dot = next(
                iter(root.graph.find_nodes(op="call_function", target=dot_scaled))
            )
            consumer = dot
            index = 0 if operand == "lhs" else 3
            value = consumer.args[index]
            assert isinstance(value, torch.fx.Node)
            if operand == "rhs_before_transpose":
                while value.target is _tracing_ops._mask_to:
                    value = value.args[0]
                    assert isinstance(value, torch.fx.Node)
                assert value.target is torch.ops.aten.permute.default
                consumer, index = value, 0
                value = consumer.args[index]
                assert isinstance(value, torch.fx.Node)
            # Keep any tracing-inserted masks: this also covers nested masks
            # when the current tracing pipeline already zero-fills operands.
            with root.graph.inserting_before(consumer):
                masked = root.graph.call_function(_tracing_ops._mask_to, (value, fill))
            masked.meta.update(value.meta)
            args = list(consumer.args)
            args[index] = masked
            consumer.args = tuple(args)
            match = _match_mxfp8(ir.graphs, ir)
            assert (match is not None) is admitted


@pytest.mark.parametrize(
    "change",
    [
        "nonzero_acc",
        "wrong_operand_row",
        "wrong_scale_row",
        "extra_store",
        "multi_root",
        "non_k_major",
        "partial_rows",
        "partial_columns",
    ],
)
def test_cpu_native_mxfp8_rejects_unsafe_programs(change: str) -> None:
    inputs = _cpu_inputs((128, 128, 128))
    with _cpu_target():
        if change == "nonzero_acc":
            bound = _nonzero_acc._bind_isolated(inputs)
        elif change in ("wrong_operand_row", "wrong_scale_row"):
            kernel = (
                _bad_operand_row if change == "wrong_operand_row" else _bad_scale_row
            )
            bound = kernel._bind_isolated(inputs)
        elif change in ("extra_store", "multi_root"):
            kernel = _extra_output if change == "extra_store" else _multiple_roots
            bound = kernel._bind_isolated((*inputs, torch.empty_like(inputs[-1])))
        elif change in ("partial_rows", "partial_columns"):
            kernel = _partial_rows if change == "partial_rows" else _partial_columns
            bound = kernel._bind_isolated(inputs)
        else:
            bound = _mxfp8._bind_isolated((inputs[0].T, *inputs[1:]))
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            assert _match_mxfp8(ir.graphs, ir) is None
        assert not bound.config_spec.cute_mxfp8_mma_available


def test_cpu_native_mxfp8_stages_only_under_aligned_raw_operands() -> None:
    from types import SimpleNamespace

    from helion.runtime.cute.launcher import _cute_stage_under_aligned_operands

    tensors = _cpu_inputs((73, 129, 160))
    original = torch.empty(tensors[0].numel() + 1, dtype=torch.float8_e4m3fn)[1:].view(
        tensors[0].shape
    )
    assert original.data_ptr() % 16 == 1
    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=[
            {"kind": "block_scaled_mma", "format": "mxfp8", "lhs_idx": 0, "rhs_idx": 1}
        ]
    )
    staged = _cute_stage_under_aligned_operands(kernel, (original, *tensors[1:]))
    assert staged is not None
    arguments, copy_backs = staged
    assert arguments[0].data_ptr() % 16 == 0
    assert arguments[1] is tensors[1]
    assert not copy_backs
    kernel._helion_cute_wrapper_plans[0].pop("format")
    assert _cute_stage_under_aligned_operands(kernel, (original, *tensors[1:])) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"cute_scaled_tile_k": 64},
        {
            "cute_scaled_tile_n": 256,
            "cute_scaled_tile_k": 256,
            "cute_scaled_stages": 4,
            "cute_scaled_cluster_m": 2,
        },
    ],
)
def test_cpu_native_mxfp8_rejects_unsupported_geometry(
    changes: dict[str, object],
) -> None:
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((73, 129, 160)))
        with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
            bound.to_code(_config(bound, **changes))


@pytest.mark.parametrize("m,n,k", [(1, 129, 1280), (73, 1, 160), (1, 1, 32)])
def test_cpu_singleton_operand_tma_coordinate_rank(
    m: int, n: int, k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    cpasync = pytest.importorskip("cutlass.cute.nvgpu.cpasync")
    driver = pytest.importorskip("cuda.bindings.driver")
    from helion._compiler.cute.block_scaled_runtime import Sm100BlockScaledGemm

    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    recorded = []
    original = cpasync.tma_partition

    def record(atom: Any, *args: object, **kwargs: object) -> Any:
        result = original(atom, *args, **kwargs)
        match = re.search(r'tma_gbasis = <"([^:]+):', str(atom.type))
        assert match is not None
        shape = ast.literal_eval(match[1])
        recorded.append(
            (
                len(shape) if isinstance(shape, tuple) else 1,
                cute.rank(result[1].iterator),
            )
        )
        return result

    monkeypatch.setattr(cpasync, "tma_partition", record)
    runtime = Sm100BlockScaledGemm(
        m, n, k, 128, 128, 3, 1, True, 148, operand_format="mxfp8"
    )
    specs = (
        (cutlass.Float8E4M3FN, (m, k), (k, 1)),
        (cutlass.Float8E4M3FN, (n, k), (k, 1)),
        (cutlass.Uint8, (runtime.workspace_bytes,), (1,)),
        (cutlass.BFloat16, (m, n), (n, 1)),
    )
    tensors = [
        cute.runtime.make_fake_tensor(dtype, shape, stride, assumed_align=16)
        for dtype, shape, stride in specs
    ]
    initialized = torch.cuda.is_initialized()
    cute.compile(
        runtime.mxfp8,
        *tensors,
        cutlass.Float32(1),
        driver.CUstream(0),
        no_jit_engine=True,
    )
    assert len(recorded) == 4
    assert all(descriptor == coordinates for descriptor, coordinates in recorded)
    assert torch.cuda.is_initialized() is initialized


@dataclass(frozen=True)
class _Case:
    name: str
    m: int
    n: int
    k: int
    cluster: int = 1
    bn: int = 128
    bk: int = 128
    stages: int = 3
    typed: bool = False
    transposed: bool = False
    wide_exponents: bool = False
    persistent: bool = True


CASES = (
    _Case("irregular_independent_scales", 73, 129, 160),
    _Case("singleton_m_long_k", 1, 8192, 1280),
    _Case("singleton_n", 137, 1, 288),
    _Case("both_singleton", 1, 1, 32),
    _Case("two_cta_edges", 269, 273, 288, cluster=2, bn=256),
    _Case("two_cta_bk256", 137, 273, 544, cluster=2, bn=256, bk=256, stages=2),
    _Case("typed_e8m0", 73, 129, 160, typed=True),
    _Case("rhs_transposed", 73, 129, 160, transposed=True),
    _Case("opposite_exponents", 73, 129, 160, typed=True, wide_exponents=True),
    _Case("nonpersistent", 259, 257, 288, persistent=False),
)


def _cuda_inputs(case: _Case) -> tuple[torch.Tensor, ...]:
    a = torch.empty((case.m, case.k), dtype=torch.float8_e4m3fn, device=DEVICE)
    b = torch.empty((case.n, case.k), dtype=torch.float8_e4m3fn, device=DEVICE)
    dtype = torch.float8_e8m0fnu if case.typed else torch.uint8
    return (
        a,
        b.T if case.transposed else b,
        torch.empty((case.m, case.k // 32), dtype=dtype, device=DEVICE),
        torch.empty((case.n, case.k // 32), dtype=dtype, device=DEVICE),
        torch.empty((case.m, case.n), dtype=torch.bfloat16, device=DEVICE),
    )


def _mutate(case: _Case, inputs: tuple[torch.Tensor, ...], seed: int) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    for tensor in inputs[:2]:
        tensor.copy_(
            (torch.randn(tensor.shape, generator=generator, device=DEVICE) * 0.25).to(
                tensor.dtype
            )
        )
    if case.wide_exponents:
        # Opposite per-group exponents keep products finite while stressing
        # native scale handling across a wide range without overflowing output.
        group = torch.arange(case.k // 32, device=DEVICE)
        left = torch.where(group % 2 == 0, 207, 47).to(torch.uint8)
        right = (254 - left.to(torch.int16)).to(torch.uint8)
        inputs[2].view(torch.uint8).copy_(left[None, :].expand(case.m, -1))
        inputs[3].view(torch.uint8).copy_(right[None, :].expand(case.n, -1))
    else:
        for tensor in inputs[2:4]:
            tensor.view(torch.uint8).copy_(
                torch.randint(
                    123,
                    132,
                    tensor.shape,
                    generator=generator,
                    device=DEVICE,
                    dtype=torch.uint8,
                )
            )
    inputs[-1].fill_(float("nan"))


def _reference(case: _Case, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, b, sa, sb, _ = inputs
    factors_a = torch.exp2(sa.view(torch.uint8).double() - 127).repeat_interleave(
        32, dim=1
    )
    factors_b = torch.exp2(sb.view(torch.uint8).double() - 127).repeat_interleave(
        32, dim=1
    )
    weights = b.T if case.transposed else b
    return ((a.double() * factors_a) @ (weights.double() * factors_b).T).bfloat16()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_cuda_native_mxfp8_correctness(case: _Case) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    inputs = _cuda_inputs(case)
    values = inputs
    kernel = _mxfp8_transposed if case.transposed else _mxfp8
    bound = kernel._bind_isolated(values)
    config = _config(
        bound,
        cute_scaled_cluster_m=case.cluster,
        cute_scaled_tile_n=case.bn,
        cute_scaled_tile_k=case.bk,
        cute_scaled_stages=case.stages,
        cute_scaled_persistent=case.persistent,
    )
    assert _wrapper_plan(bound.to_code(config))["format"] == "mxfp8"
    compiled = bound.compile_config(config)

    def check(current: tuple[torch.Tensor, ...]) -> None:
        torch.cuda.synchronize()
        assert torch.isfinite(current[-1]).all()
        torch.testing.assert_close(
            current[-1], _reference(case, current), atol=2e-2, rtol=2e-2
        )

    for seed in (810, 811):
        _mutate(case, inputs, seed)
        compiled(*values)
        check(inputs)
    replacements = _cuda_inputs(case)
    _mutate(case, replacements, 812)
    compiled(*replacements)
    check(replacements)


def test_cuda_native_mxfp8_graph_replay_observes_each_live_input() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    case = _Case("replay", 137, 129, 288, cluster=2)
    inputs = _cuda_inputs(case)
    _mutate(case, inputs, 820)
    values = inputs
    bound = _mxfp8._bind_isolated(values)
    compiled = bound.compile_config(_config(bound, cute_scaled_cluster_m=2))
    compiled(*values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*values)
    previous = inputs[-1].clone()
    for index in range(4):
        if index < 2:
            inputs[index].copy_((-inputs[index].float()).to(inputs[index].dtype))
        else:
            inputs[index].view(torch.uint8).add_(1)
        inputs[-1].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            inputs[-1], _reference(case, inputs), atol=2e-2, rtol=2e-2
        )
        assert not torch.equal(inputs[-1], previous)
        previous = inputs[-1].clone()


def test_cuda_native_mxfp8_under_aligned_inputs_and_replacement() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    case = _Case("under_aligned", 73, 129, 160)
    inputs = list(_cuda_inputs(case))
    _mutate(case, tuple(inputs), 830)
    for index in (0, 1):
        value = inputs[index]
        replacement = torch.empty(
            value.numel() + 1, dtype=value.dtype, device=value.device
        )[1:].view(value.shape)
        replacement.copy_(value)
        assert replacement.data_ptr() % 16 == 1
        inputs[index] = replacement
    values = tuple(inputs)
    bound = _mxfp8._bind_isolated(values)
    compiled = bound.compile_config(_config(bound))
    compiled(*values)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        values[-1], _reference(case, values), atol=2e-2, rtol=2e-2
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*values)
    _mutate(case, values, 831)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        values[-1], _reference(case, values), atol=2e-2, rtol=2e-2
    )
    aligned = _cuda_inputs(case)
    _mutate(case, aligned, 832)
    compiled(*aligned)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        aligned[-1], _reference(case, aligned), atol=2e-2, rtol=2e-2
    )
