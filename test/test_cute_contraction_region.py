from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import LoopCarry
from helion._compiler.device_ir import LoopInterface
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language.matmul_ops import dot


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _collective_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    seed: torch.Tensor,
    scan_axis: int,
    final_reduce: hl.constexpr,
) -> torch.Tensor:
    scan_axis = hl.specialize(scan_axis)
    m, k = a.shape
    q, n = v.shape
    result = torch.empty_like(seed)
    for row, col in hl.tile([m, n], block_size=[128, 64]):
        kk, qq = hl.arange(k), hl.arange(q)
        raw = a[row, kk].float()
        prefix = hl.cumsum(delta[row, kk], dim=scan_axis)
        norm = (raw * raw).sum(1)
        left = (raw * torch.rsqrt(norm[:, None] + 1e-4) * torch.exp(prefix)).to(a.dtype)
        first = hl.dot(left, b[kk, qq], out_dtype=torch.float32)
        energy = (first * first).sum(1)
        initial = seed[row, col] + energy[:, None]
        value = hl.dot(
            first.to(v.dtype), v[qq, col], acc=initial, out_dtype=torch.float32
        )
        if final_reduce:
            value = value / ((value * value).sum(1)[:, None] + 1.0)
        result[row, col] = value
    return result


def _collective_args(device: str | torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(791)
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * scale
        for shape, dtype, scale in (
            ((128, 32), torch.bfloat16, 0.1),
            ((32, 64), torch.bfloat16, 0.1),
            ((64, 64), torch.float16, 0.1),
            ((128, 32), torch.float32, 0.001),
            ((128, 64), torch.float32, 0.1),
        )
    )


@pytest.mark.parametrize("schedule", ["cp_async_register", "tcgen05_tmem"])
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("final_reduce", [False, True])
def test_collective_chain_uses_shared_stage_emission_cpu(
    schedule: str, scan_axis: int, final_reduce: bool
) -> None:
    with _cpu_codegen():
        bound = _collective_chain._bind_isolated(
            (*_collective_args("cpu"), scan_axis, final_reduce)
        )
        config = helion.Config(cute_chained_mma_schedule=schedule)
        if schedule == "tcgen05_tmem":
            config.config["cute_chained_tmem_free"] = "last_read"
        source = bound.to_code(config)
    assert "chain_0_mma" in source and "chain_1_mma" in source
    assert "chain_collective_0" in source
    assert "shuffle_sync_down" in source
    assert "chain_1_seed" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["cp_async_register", "tcgen05_tmem"])
@pytest.mark.parametrize("scan_axis", [0, 1])
@pytest.mark.parametrize("final_reduce", [False, True])
def test_collective_chain_preserves_source_arithmetic(
    schedule: str, scan_axis: int, final_reduce: bool
) -> None:
    args = _collective_args(DEVICE)
    bound = _collective_chain.bind((*args, scan_axis, final_reduce))
    config = helion.Config(cute_chained_mma_schedule=schedule)
    source = bound.to_code(config)
    assert "chain_0_mma" in source and "chain_1_mma" in source
    compiled = bound.compile_config(config)
    actual = compiled(*args, scan_axis, final_reduce)
    a, b, v, delta, seed = args
    raw = a.float()
    norm = (raw * raw).sum(1)
    left = (
        raw * torch.rsqrt(norm[:, None] + 1e-4) * torch.exp(delta.cumsum(scan_axis))
    ).to(a.dtype)
    first = left.float() @ b.float()
    expected = (
        first.to(v.dtype).float() @ v.float() + seed + (first * first).sum(1)[:, None]
    )
    if final_reduce:
        expected = expected / ((expected * expected).sum(1)[:, None] + 1.0)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        compiled(*args, scan_axis, final_reduce), actual, atol=0, rtol=0
    )


if TYPE_CHECKING:
    from collections.abc import Callable


def _input(graph: Graph, name: str, value: torch.Tensor) -> Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _call(
    graph: Graph,
    target: Callable[..., object],
    args: tuple[Any, ...],
    value: torch.Tensor,
) -> Node:
    node = graph.call_function(target, args)
    node.meta["val"] = value
    return node


def _mixed_graph() -> RootGraphInfo:
    graph = Graph()
    a = _input(graph, "a", torch.empty((13, 17), dtype=torch.bfloat16))
    b = _input(graph, "b", torch.empty((17, 11), dtype=torch.bfloat16))
    c = _input(graph, "c", torch.empty((11, 7), dtype=torch.float16))
    seed = _input(graph, "seed", torch.empty((13, 7), dtype=torch.float32))
    first = _call(
        graph,
        dot,
        (a, b, None, torch.float32),
        torch.empty((13, 11), dtype=torch.float32),
    )
    rounded = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (first, torch.float16),
        torch.empty((13, 11), dtype=torch.float16),
    )
    second = _call(
        graph,
        dot,
        (rounded, c, seed, torch.float32),
        torch.empty_like(seed.meta["val"]),
    )
    graph.output((first, second))
    return RootGraphInfo(3, graph)


def test_preserves_mixed_precision_accumulator_and_unpadded_domains() -> None:
    info = _mixed_graph()
    before = tuple((node, node.args, dict(node.meta)) for node in info.graph.nodes)
    region = collect_contraction_region(info)
    assert region is not None
    first, second = region.contractions
    assert first.shape == (13, 11, 17)
    assert second.shape == (13, 7, 11)
    assert first.operand_dtypes == (torch.bfloat16, torch.bfloat16)
    assert second.operand_dtypes == (torch.float16, torch.float16)
    assert first.accumulator is None
    assert second.accumulator is region.live_ins[3]
    assert second.accumulator_dtype is torch.float32
    assert second.requested_out_dtype is torch.float32
    assert second.result_dtype is torch.float32
    assert region.live_outs == (first.node, second.node)
    assert second.lhs.args[0] is first.node  # The cast boundary is still a node.
    assert before == tuple(
        (node, node.args, dict(node.meta)) for node in info.graph.nodes
    )
    with pytest.raises(FrozenInstanceError):
        region.graph_id = 9  # pyrefly: ignore [read-only]


def test_recollecting_copy_never_reuses_original_nodes() -> None:
    info = _mixed_graph()
    original = collect_contraction_region(info)
    copied = collect_contraction_region(info.copy())
    assert original is not None and copied is not None
    assert not set(original.nodes).intersection(copied.nodes)
    assert all(spec.node.graph is copied.graph for spec in copied.contractions)
    assert copied.contractions[1].lhs.args[0] is copied.contractions[0].node


@pytest.mark.parametrize(
    ("position", "shape", "dtype"),
    [
        (1, (19, 11), torch.bfloat16),  # Reduction dimensions disagree.
        (4, (13, 12), torch.float32),  # Recorded result shape disagrees.
        (4, (13, 11), torch.bfloat16),  # Explicit FP32 output was lost.
        (3, (13, 8), torch.float32),  # Explicit accumulator shape disagrees.
        (3, (13, 7), torch.bfloat16),  # Invalid Helion accumulator dtype.
        (0, (1, 13, 17), torch.bfloat16),  # Batched scheduling is not described.
    ],
)
def test_malformed_dot_contracts_are_rejected(
    position: int, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    info = _mixed_graph()
    tuple(info.graph.nodes)[position].meta["val"] = torch.empty(shape, dtype=dtype)
    assert collect_contraction_region(info) is None


def test_cross_graph_operand_is_rejected() -> None:
    info = _mixed_graph()
    foreign = _input(Graph(), "foreign", torch.empty((13, 17), dtype=torch.bfloat16))
    first = next(node for node in info.graph.nodes if node.target is dot)
    first.update_arg(0, foreign)
    assert collect_contraction_region(info) is None


def test_unbacked_domains_need_no_concrete_shapes_or_guards() -> None:
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        m, n, k = (shape_env.create_unbacked_symint() for _ in range(3))
        graph = Graph()
        a = _input(graph, "a", torch.empty((m, k), dtype=torch.bfloat16))
        b = _input(graph, "b", torch.empty((k, n), dtype=torch.bfloat16))
        result = _call(
            graph, dot, (a, b, None, None), torch.empty((m, n), dtype=torch.float32)
        )
        graph.output(result)
        guards = tuple(shape_env.guards)
        region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    assert region.contractions[0].shape == (m.node.expr, n.node.expr, k.node.expr)
    assert tuple(shape_env.guards) == guards


def _loop_graph() -> ForLoopGraphInfo:
    info = _mixed_graph()
    second = tuple(info.graph.find_nodes(op="call_function", target=dot))[1]
    output = info.graph.find_nodes(op="output")[0]
    output.args = ((second,),)
    return ForLoopGraphInfo(
        info.graph_id,
        info.graph,
        node_args=[],
        block_ids=[],
        loop_interface=LoopInterface(4, (LoopCarry(3, 0),)),
    )


def test_loop_carry_binds_slots_and_preserves_captured_coefficients() -> None:
    region = collect_contraction_region(_loop_graph())
    assert region is not None and region.loop_interface is not None
    assert region.loop_interface.captures == (0, 1, 2)
    (carry,) = region.carries
    assert (carry.input_index, carry.output_index) == (3, 0)
    assert carry.input is region.contractions[1].accumulator
    assert carry.output is region.contractions[1].node


@pytest.mark.parametrize(
    "interface",
    [
        LoopInterface(3, (LoopCarry(3, 0),)),
        LoopInterface(4, (LoopCarry(-1, 0),)),
        LoopInterface(4, (LoopCarry(3, 1),)),
        LoopInterface(4, (LoopCarry(3, 0), LoopCarry(3, 0))),
        LoopInterface(4, (LoopCarry(0, 0),)),
        LoopInterface(4, ()),
    ],
)
def test_malformed_loop_ports_are_rejected(interface: LoopInterface) -> None:
    assert (
        collect_contraction_region(replace(_loop_graph(), loop_interface=interface))
        is None
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _traced_region(a, b, seed, delta):
    m, k = a.shape
    n = b.shape[1]
    first_out = torch.empty_like(seed)
    copy_out = torch.empty_like(seed)
    second_out = torch.empty_like(seed)
    for mi, ni in hl.tile([m, n], block_size=[16, 16]):
        ki = hl.arange(k)
        prefix = hl.cumsum(delta[mi, ki].float(), dim=0)
        left = a[mi, ki].float()
        magnitude = torch.sum(left * left, dim=-1)
        weighted = (left * (prefix + magnitude[:, None])).to(a.dtype)
        first = hl.dot(weighted, b[ki, ni], out_dtype=torch.float32)
        state = seed[mi, ni].float()
        for _ in hl.grid(2):
            state = hl.dot(
                a[mi, ki].to(torch.float16), b[ki, ni].to(torch.float16), acc=state
            )
            second_out[mi, ni] = state
        first_out[mi, ni] = first
        copy_out[mi, ni] = torch.relu(first)
    return first_out, copy_out, second_out


def test_real_helion_graphs_capture_reductions_scans_stores_and_carries() -> None:
    args = (
        torch.empty((31, 16), dtype=torch.bfloat16),
        torch.empty((16, 16), dtype=torch.bfloat16),
        torch.empty((31, 16), dtype=torch.float32),
        torch.empty((31, 16), dtype=torch.float32),
    )
    with _cpu_codegen():
        bound = _traced_region._bind_isolated(args)
    assert bound.host_function is not None
    regions = [
        region
        for info in bound.host_function.device_ir.graphs
        if (region := collect_contraction_region(info)) is not None
    ]
    assert len(regions) == 2
    root = next(region for region in regions if region.loop_interface is None)
    body = next(region for region in regions if region.loop_interface is not None)
    assert len(root.scans) == 1
    assert root.scans[0].meta["val"].ndim == 2
    assert len(root.reductions) == 1
    assert len(root.stores) == 2
    assert len(body.stores) == 1
    assert root.contractions[0].operand_dtypes == (torch.bfloat16, torch.bfloat16)
    assert body.contractions[0].operand_dtypes == (torch.float16, torch.float16)
    assert len(body.carries) == 1
    accumulator = body.contractions[0].accumulator
    assert accumulator is not None and accumulator.target is _tracing_ops._new_var
    assert accumulator.args[0] is body.carries[0].input
    assert all(node.target is memory_ops.store for node in (*root.stores, *body.stores))
