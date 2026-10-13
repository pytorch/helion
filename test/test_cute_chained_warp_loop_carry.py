from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_loop import advance_carries
from helion._compiler.cute.chained_warp_loop_carry import carry_topology
from helion._compiler.cute.chained_warp_loop_carry import plan_warp_loop_carry
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
from helion.language import _tracing_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST


def fixture(*, shape=(32, 32), mode="plain"):
    graph = Graph()
    rows, columns = shape
    state = _input(graph, "state", shape, torch.float32)
    left = _input(graph, "left", (rows, 32), torch.bfloat16)
    right = _input(graph, "right", (32, columns), torch.bfloat16)
    sink = _call(graph, _tracing_ops._host_tensor, ("sink",), shape, torch.bfloat16)
    incoming = _convert(graph, state, torch.bfloat16)
    if mode == "transpose_store":
        incoming = _call(
            graph, torch.ops.aten.t.default, (incoming,), shape, torch.bfloat16
        )
    store = graph.call_function(memory_ops.store, (sink, [None, None], incoming))
    seed = _call(graph, torch.ops.aten.mul.Tensor, (state, 0.75), shape, torch.float32)
    if mode == "transpose_seed":
        seed = _call(graph, torch.ops.aten.t.default, (seed,), shape, torch.float32)
    if mode == "operand":
        left = _convert(graph, state, torch.bfloat16)
    if mode in ("size", "bad_size"):
        size = graph.call_function(torch.ops.aten.sym_size.int, (state, 0))
        size.meta["val"] = rows + (mode == "bad_size")
        left = _call(
            graph, torch.ops.aten.mul.Scalar, (left, size), (rows, 32), torch.bfloat16
        )
    if mode == "alias_seed":
        host = _call(
            graph, _tracing_ops._host_tensor, ("seed_host",), shape, torch.float32
        )
        host.meta["val"] = sink.meta["val"].view(torch.float32)
        load = _call(graph, memory_ops.load, (host, [None, None]), shape, torch.float32)
        seed = _call(
            graph, torch.ops.aten.add.Tensor, (seed, load), shape, torch.float32
        )
    if mode == "escape":
        _call(graph, torch.ops.aten.neg.default, (state,), shape, torch.float32)
    result = _dot(graph, left, right, seed)
    if mode == "new_store":
        store.args = (sink, [None, None], result)
        result.append(store)
    plan = replace(_loop_plan(graph, (state,), (result,)), strategy="warp")
    assert plan.loop is not None
    root = plan.loop.root.graph
    with root.inserting_before(next(iter(root.find_nodes(op="output")))):
        final = _input(root, "final", shape, torch.float32)
        target = _call(
            root, _tracing_ops._host_tensor, ("final_target",), shape, torch.float32
        )
        final_store = root.call_function(
            memory_ops.store, (target, [None, None], final)
        )
    loop = replace(plan.loop, final_values=((final, 0),), final_stores=(final_store,))
    return replace(plan, loop=loop), state, sink


def select(plan, row_warps=2):
    with patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)):
        return plan_warp_loop_carry(plan, row_warps)


@pytest.mark.parametrize("mode", ["plain", "size"])
@pytest.mark.parametrize("shape,rows", [((32, 32), 1), ((32, 32), 2), ((64, 64), 4)])
def test_exact_carry_and_shape_only_edges_have_one_owner(mode, shape, rows):
    plan, state, _ = fixture(shape=shape, mode=mode)
    selected = select(plan, rows)
    assert selected is not None and selected.carry.input is state
    assert selected.topology == (rows, 4 // rows, 1)
    assert plan.loop is not None
    assert selected.stores == plan.loop.region.stores
    assert len(selected.final_sources) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "transpose_seed",
        "transpose_store",
        "operand",
        "escape",
        "new_store",
        "bad_size",
        "alias_seed",
    ],
)
def test_value_use_coordinate_and_mutation_conflicts_decline(mode):
    plan, _, _ = fixture(mode=mode)
    assert select(plan) is None


@pytest.mark.parametrize("change", ["dtype", "threads", "storage", "graph"])
def test_retained_plan_checks_graph_typed_owner_and_storage(change):
    plan, state, sink = fixture()
    selected = select(plan)
    assert selected is not None
    if change == "dtype":
        state.meta["val"] = state.meta["val"].to(torch.bfloat16)
    elif change == "threads":
        plan = replace(plan, threads=256)
    elif change == "storage":
        sink.meta["val"] = torch.empty_like(sink.meta["val"])
    else:
        selected.seed.args = (state, 0.5)
    with (
        patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)),
        pytest.raises(chain._UnsupportedChain, match="ownership changed"),
    ):
        selected.check(plan)


@pytest.mark.parametrize(
    "shape,threads,rows",
    [
        ((0, 32), 128, 1),
        ((32, 24), 128, 2),
        ((16, 32), 128, 2),
        ((32, 32), 128, 8),
        ((32, 32), 96, 1),
        ((32, 32), 128, 0),
    ],
)
def test_partial_atoms_or_incompatible_warp_factors_decline(shape, threads, rows):
    assert carry_topology(shape, threads, rows) is None


@pytest.mark.parametrize(
    "shape,threads,rows", [((32, 32), 128, 1), ((32, 32), 128, 2), ((64, 64), 128, 4)]
)
def test_actual_cute_owner_coordinates_cover_each_cell_once(shape, threads, rows):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    ir = pytest.importorskip("cutlass._mlir.ir")
    topology = carry_topology(shape, threads, rows)
    assert topology is not None
    owners = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
                    )
                ),
                atom_layout_mnk=topology,
            )
            for thread in range(threads):
                coords = mma.get_slice(thread).partition_C(
                    cute.make_identity_tensor(shape)
                )
                owners.extend(
                    tuple(int(x) for x in coords[index])
                    for index in range(cute.size(coords))
                )
    assert len(owners) == len(set(owners)) == shape[0] * shape[1]
    assert set(owners) == {(r, c) for r in range(shape[0]) for c in range(shape[1])}


def spec():
    return cast(
        "ConfigSpec",
        SimpleNamespace(
            backend_name="cute",
            cute_chained_loop_search_enabled=True,
            cute_chained_fragment_reuse_search_enabled=True,
        ),
    )


def test_new_topology_is_explicit_and_default_stays_column_warps():
    config: dict[str, object] = {
        "cute_chained_mma_schedule": "cp_async",
        "cute_chained_fragment_reuse": True,
        "num_warps": 4,
    }
    ConfigSpec._normalize_cute_fragment_reuse(spec(), config)
    ConfigSpec._normalize_cute_warp_row_warps(spec(), config)
    assert "cute_chained_warp_row_warps" not in config
    assert carry_topology((32, 32), 128) == (1, 4, 1)
    config["cute_chained_warp_row_warps"] = 2
    ConfigSpec._normalize_cute_warp_row_warps(spec(), config)
    assert config["cute_chained_warp_row_warps"] == 2


@pytest.mark.parametrize("value", [True, 0, 3, "2", None])
def test_topology_type_and_factor_reject(value):
    with pytest.raises(InvalidConfig):
        ConfigSpec._normalize_cute_warp_row_warps(
            spec(), {"cute_chained_warp_row_warps": value}
        )


def test_existing_tmem_fragment_reuse_admission_is_unchanged():
    config: dict[str, object] = {
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_fragment_reuse": True,
        "cute_chained_group_contractions": True,
        "cute_chained_preparation_pipeline": True,
        "cute_chained_loop_mma_rows": 64,
    }
    ConfigSpec._normalize_cute_fragment_reuse(spec(), config)
    for key in ("cute_chained_group_contractions", "cute_chained_preparation_pipeline"):
        with pytest.raises(InvalidConfig):
            ConfigSpec._normalize_cute_fragment_reuse(spec(), config | {key: False})


@pytest.mark.parametrize("trips", [0, 1, 3])
def test_initial_and_final_owner_keep_zero_trip_and_store_before_seed(trips):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.cute.chained_warp_loop_carry import WarpLoopCarryEmission
    from helion._compiler.cute.chained_warp_mma import emit_warp_mma

    plan, state, _ = fixture()
    selected = select(plan)
    assert selected is not None
    plan = replace(plan, warp_loop_carry=selected)

    class Expression:
        def __init__(self, *args):
            self.coordinate_names = set()
            self.fragments = {}
            self.memo = {}
            self.lines = []

        def value(self, node, coordinates):
            if node is selected.seed:
                self.memo[(state, coordinates)] = self.fragments[state][1]
                return f"({self.fragments[state][1]} * 0.75)"
            return f"original_initial[{coordinates[0]}, {coordinates[1]}]"

    def store(self, node, source, prefix):
        if node in selected.stores:
            return ("entering.append(tuple(chain_0_acc))",)
        return ("final.extend(chain_0_acc)",)

    with (
        patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)),
        patch.object(chain, "_Expression", Expression),
        patch.object(WarpLoopCarryEmission, "_store", store),
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(
                backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.BFloat16")
            ),
        ),
    ):
        emission = WarpLoopCarryEmission(cast("GenerateAST", SimpleNamespace()), plan)
        initial = emission.initialize()
        assert "atom_layout_mnk=(2, 2, 1)" in initial[0]
        assert (
            "original_initial[chain_carry_init_row, chain_carry_init_column]"
            in initial[-1]
        )
        with pytest.raises(chain._UnsupportedChain, match="initialized twice"):
            emission.initialize()
        seed = emission.seed_lines()
        issue = emit_warp_mma(
            "chain_0",
            "cutlass.BFloat16",
            (32, 32, 32),
            128,
            {"a": 1, "b": 1},
            seed,
            resident=emission,
        )
        assert not any("chain_0_acc =" in line for line in issue)
        assert not any("chain_0_mma =" in line for line in issue)
        assert issue.index(seed[0]) < next(
            i for i, line in enumerate(issue) if "cute.gemm(" in line
        )
        with pytest.raises(chain._UnsupportedChain, match="changed its owner"):
            emit_warp_mma(
                "chain_0",
                "cutlass.BFloat16",
                (32, 32, 32),
                128,
                {"a": 1, "b": 1},
                seed,
                resident=emission,
            )
        final = emission.final_stores()
        with pytest.raises(chain._UnsupportedChain, match="lacks completion"):
            emission.final_stores()

    original = [1.25, -2.75, 5.125, 8.375]
    namespace: dict[str, Any] = {
        "chain_0_acc": original.copy(),
        "chain_carry_coords": [(8, 16), (0, 17), (8, 0), (0, 1)],
        "cutlass": SimpleNamespace(Float32=float, range_constexpr=range),
        "cute": SimpleNamespace(size=len),
        "entering": [],
        "final": [],
    }
    expected = original.copy()
    for _ in range(trips):
        before = tuple(expected)
        exec(compile("\n".join(seed), "<bounded-original-seed>", "exec"), namespace)
        assert namespace["entering"][-1] == before
        expected = [x * 0.75 + 2 for x in expected]
        namespace["chain_0_acc"] = [x + 2 for x in namespace["chain_0_acc"]]
    exec(compile("\n".join(final), "<bounded-final-owner>", "exec"), namespace)
    assert namespace["final"] == expected
    assert len(namespace["entering"]) == trips


@pytest.mark.parametrize(
    "shape,row_warps", [((32, 32), 1), ((32, 32), 2), ((64, 64), 4)]
)
def test_all_owned_register_carries_need_no_shared_transition(shape, row_warps):
    plan, _, _ = fixture(shape=shape)
    selected = select(plan, row_warps)
    assert selected is not None
    plan = replace(plan, warp_loop_carry=selected)
    assert (
        advance_carries(
            cast("GenerateAST", None),
            plan,
            {},
            resident_carries=frozenset((selected.carry.input_index,)),
        )
        == []
    )


def test_resident_index_without_warp_owner_keeps_original_transition_join():
    plan, _, _ = fixture()
    assert plan.loop is not None
    resident = frozenset(carry.input_index for carry in plan.loop.region.carries)
    assert advance_carries(
        cast("GenerateAST", None), plan, {}, resident_carries=resident
    ) == ["cute.arch.sync_threads()"]


def test_register_transition_rechecks_actual_carry_owner():
    plan, _, _ = fixture()
    selected = select(plan)
    assert selected is not None
    changed = replace(plan, warp_loop_carry=selected, threads=64)
    with pytest.raises(chain._UnsupportedChain, match="carry ownership changed"):
        advance_carries(
            cast("GenerateAST", None),
            changed,
            {},
            resident_carries=frozenset((selected.carry.input_index,)),
        )
