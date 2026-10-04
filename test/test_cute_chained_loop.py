from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING

from benchmarks.cute.kda_prefill_kernels import KDA_RECURRENCE_CONFIG
from benchmarks.cute.kda_prefill_kernels import kda_chunk_recurrence_fp32
import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_loop import _disjoint_writes
from helion._compiler.cute.chained_loop import discover_chained_loop
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.device_ir import LoopCarry
from helion._compiler.device_ir import LoopInterface
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _affine_chain(left, right, initial, coefficient, steps: int, transpose: int):
    steps = hl.specialize(steps)
    transpose = hl.specialize(transpose)
    batches, _, m, k = left.shape
    n = right.shape[-1]
    history = torch.empty(
        (batches, max(steps, 1), m, n), device=left.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, 16, 16]):
        state = initial[bi.id, mi, ni].float()
        scale = coefficient[bi.id].float()
        for ti in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            product = hl.dot(
                left[bi.id, ti.id, mi, kk],
                right[bi.id, ti.id, kk, ni],
                out_dtype=torch.float32,
            )
            if transpose:
                state = state.T * scale + product
            else:
                state = state * scale + product
            history[bi.id, ti.id, mi, ni] = state
        final[bi.id, mi, ni] = state
    return history, final


def _affine_args(steps: int = 3, transpose: bool = False) -> tuple:
    return (
        torch.empty((2, max(steps, 1), 16, 16), dtype=torch.bfloat16),
        torch.empty((2, max(steps, 1), 16, 16), dtype=torch.bfloat16),
        torch.empty((2, 16, 16), dtype=torch.float32),
        torch.empty((2,), dtype=torch.float32),
        steps,
        int(transpose),
    )


def _kda_args() -> tuple:
    shapes = [(1, 2, 32, 128)] * 3 + [
        (1, 2, 2, 256),
        (1, 2, 2, 128),
        (1, 32, 2, 128),
        (1, 32, 2, 128),
        (1, 2, 128, 128),
        (1, 2, 128, 128),
        (2,),
        (2,),
    ]
    dtypes = (
        [torch.bfloat16] * 4
        + [torch.float32]
        + [torch.bfloat16] * 2
        + [torch.float32] * 2
        + [torch.int32] * 2
    )
    return (
        *(
            torch.empty(shape, dtype=dtype)
            for shape, dtype in zip(shapes, dtypes, strict=True)
        ),
        0.125,
    )


def _kda_config(schedule: str) -> helion.Config:
    grouped = schedule == "tcgen05_grouped"
    return helion.Config.from_dict(
        {
            **KDA_RECURRENCE_CONFIG.config,
            "cute_chained_mma_schedule": "tcgen05_tmem" if grouped else schedule,
            "cute_chained_group_contractions": grouped,
            "num_warps": 4,
        }
    )


def _source(bound: BoundKernel, config: helion.Config) -> str:
    source = bound.to_code(config)
    assert "chain_loop_index" in source
    assert "chain_loop_carry_" in source
    assert "chain_final_store_" in source
    tree = ast.parse(source)
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and "chain_loop_index" in node.target.id
    ]
    assert len(loops) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alloc_smem"
        for node in ast.walk(loops[0])
    )
    return source


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_affine_loop_uses_same_contraction_emitter(
    steps: int, transpose: bool, schedule: str
) -> None:
    with _cpu_codegen():
        bound = _affine_chain._bind_isolated(_affine_args(steps, transpose))
        source = _source(
            bound, helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
        )
    assert ("MmaF16BF16Op" if schedule == "coalesced" else "tcgen05.commit") in source
    assert (
        "chain_loop_carry_0_next" in source or "_next = cute.make_rmem_tensor" in source
    )


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_split_kda_loop_uses_all_four_shared_contraction_stages(schedule: str) -> None:
    with _cpu_codegen():
        bound = kda_chunk_recurrence_fp32._bind_isolated(_kda_args())
        source = _source(bound, _kda_config(schedule))
    for stage in range(4):
        assert f"chain_{stage}_mma" in source
    assert "chain_2_seed" in source
    assert "chunk_recurrence_sm100" not in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("transpose", [False, True])
def test_affine_loop_gpu_preserves_state_and_zero_trip(
    schedule: str, steps: int, transpose: bool
) -> None:
    torch.manual_seed(901)
    args = tuple(
        torch.randn_like(value, device=DEVICE) * 0.125
        if isinstance(value, torch.Tensor)
        else value
        for value in _affine_args(steps, transpose)
    )
    left, right, initial, scale, _, _ = args
    original = initial.clone()
    bound = _affine_chain._bind_isolated(args)
    config = helion.Config(cute_chained_mma_schedule=schedule, num_warps=4)
    _source(bound, config)
    compiled = bound.compile_config(config)
    actual_history, actual_final = compiled(*args)
    expected = initial.clone()
    for step in range(steps):
        if transpose:
            expected = expected.transpose(-1, -2)
        expected = expected * scale[:, None, None] + (
            left[:, step].float() @ right[:, step].float()
        )
        torch.testing.assert_close(
            actual_history[:, step], expected, rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(actual_final, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    _, repeated = compiled(*args)
    torch.testing.assert_close(repeated, actual_final, rtol=0, atol=0)


def _kda_expected(args: tuple, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    kd, qd, ak, aq, decay, values, output, initial, _, _, _, scale = args
    expected = output.clone()
    state = initial.clone()
    feature = torch.arange(128, device=values.device)
    lane = torch.arange(16, device=values.device)
    byte_offset = 2 * (lane[:, None] * 16 + (lane[None, :] ^ 8))
    pair = (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2
    for chunk in range((tokens + 15) // 16):
        valid = min(tokens - chunk * 16, 16)
        for head in range(values.size(2)):
            current = state[0, head]
            state_half = current.to(kd.dtype).float()
            first = chunk * 16
            kd_value = kd[0, head, first : first + 16, :][:, feature ^ 8].float()
            qd_value = qd[0, head, first : first + 16, :][:, feature ^ 8].float()
            ak_value = ak[0, head, first + (lane ^ 8), :].float()
            aq_value = aq[0, head, chunk, pair].float()
            projected = kd_value @ state_half.T
            residual = torch.zeros((16, 128), device=values.device, dtype=values.dtype)
            residual[:valid] = (
                values[0, first : first + valid, head].float()
                - projected[:valid].to(values.dtype).float()
            ).to(values.dtype)
            result = torch.addmm(qd_value @ state_half.T, aq_value, residual.float())
            expected[0, first : first + valid, head] = (result[:valid] * scale).to(
                output.dtype
            )
            state[0, head] = current * decay[0, head, chunk] + (
                residual.float().T @ ak_value
            )
    return expected, state


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem", "tcgen05_grouped"])
@pytest.mark.parametrize("tokens", [0, 13, 16, 29, 32])
def test_split_kda_gpu_source_arithmetic(schedule: str, tokens: int) -> None:
    torch.manual_seed(902)
    args = tuple(
        torch.randn_like(value, device=DEVICE) * 0.0625
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.to(DEVICE)
        if isinstance(value, torch.Tensor)
        else value
        for value in _kda_args()
    )
    args[9].copy_(torch.tensor([0, tokens], device=DEVICE, dtype=torch.int32))
    args[10].copy_(
        torch.tensor([0, (tokens + 15) // 16], device=DEVICE, dtype=torch.int32)
    )
    original = args[7].clone()
    expected_output, expected_final = _kda_expected(args, tokens)
    bound = kda_chunk_recurrence_fp32._bind_isolated(args)
    config = _kda_config(schedule)
    _source(bound, config)
    actual_output, actual_final = bound.compile_config(config)(*args)
    torch.testing.assert_close(actual_output, expected_output, rtol=0.02, atol=2e-4)
    torch.testing.assert_close(actual_final, expected_final, rtol=0.02, atol=2e-4)
    torch.testing.assert_close(args[7], original, rtol=0, atol=0)


def test_loop_storage_guard_rejects_external_aliases() -> None:
    source = torch.empty(128)
    output = torch.empty(64)
    assert _disjoint_writes((source, output), writes=(1,))
    assert not _disjoint_writes((source, source[16:80]), writes=(1,))
    assert not _disjoint_writes((source, output[::2]), writes=(1,))
    assert _disjoint_writes((source[:64], source[64:]), writes=(1,))


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_loop_load_eviction_hint_is_not_a_masked_value(
    masked: bool, schedule: str
) -> None:
    with _cpu_codegen():
        bound = kda_chunk_recurrence_fp32._bind_isolated(_kda_args())
        assert bound.host_function is not None
        changed = 0
        for graph in bound.host_function.device_ir.graphs:
            for node in graph.graph.nodes:
                if (
                    node.target is memory_ops.load
                    and (node.args[2] is not None) == masked
                ):
                    node.args = (*node.args[:3], "evict_last")
                    changed += 1
        assert changed
        source = _source(bound, _kda_config(schedule))
    assert "BFloat16('evict_last')" not in source
    assert 'BFloat16("evict_last")' not in source


def test_malformed_carry_interface_does_not_claim_loop() -> None:
    with _cpu_codegen():
        bound = _affine_chain._bind_isolated(_affine_args())
        assert bound.host_function is not None
        graphs = bound.host_function.device_ir.graphs
        loop = discover_chained_loop(graphs)
        assert loop is not None
        malformed = replace(
            loop.body,
            loop_interface=LoopInterface(
                len(loop.inputs), (LoopCarry(len(loop.inputs), 0),)
            ),
        )
        assert discover_chained_loop([loop.root, malformed]) is None


def test_half_product_with_fp32_accumulator_is_not_admitted() -> None:
    with _cpu_codegen():
        bound = _affine_chain._bind_isolated(_affine_args())
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            loop = discover_chained_loop(bound.host_function.device_ir.graphs)
            assert loop is not None
            contraction = next(
                node for node in loop.body.graph.nodes if node.target is dot
            )
            accumulator = loop.region.carries[0].input
            contraction.args = (*contraction.args[:2], accumulator, torch.float16)
            assert collect_contraction_region(loop.body) is not None
            assert (
                chained_matmul._classify_chained_graph(
                    bound.host_function.device_ir.graphs
                )
                is None
            )
