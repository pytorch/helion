from __future__ import annotations

from contextlib import contextmanager
import dataclasses
from unittest.mock import patch

import pytest
import torch
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_work_order_public import _policies
from .test_cute_work_order_scalar import _ragged_sum
import helion
from helion import exc
from helion._compiler.cute import work_order
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.host_function import HostFunction
from helion._compiler.tile_dependency import TileDependency
from helion._compiler.tile_dependency import TileDependencyGraph
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import CUTE_GRID_WORK_ORDER_KEY
from helion.autotuner.config_spec import InvalidConfig
from helion.language import memory_ops


def _args():
    return (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )


@contextmanager
def _node_args(node, value):
    original = node.args
    users = {item: dict(item.users) for item in node.graph.nodes}
    try:
        node.args = value
        yield
    finally:
        node.args = original
        for item, prior in users.items():
            item.users.clear()
            item.users.update(prior)


def test_actual_tuner_coordinate_and_identity_default():
    with _cpu_codegen():
        bound = _ragged_sum._bind_isolated(_args())
        spec = bound.config_spec
        fields = spec._flat_fields()
        field = fields[CUTE_GRID_WORK_ORDER_KEY]
        assert isinstance(field, EnumFragment)
        assert field.choices == (("identity",), ("longest_first",))
        assert field.default() == ("identity",)
        assert list(fields)[-1] == CUTE_GRID_WORK_ORDER_KEY
        default = spec.default_config()
        reference = spec.autotune_reference_config()
        assert CUTE_GRID_WORK_ORDER_KEY not in default
        assert CUTE_GRID_WORK_ORDER_KEY not in reference
        with patch.object(spec, "cute_work_order_candidates", ()):
            original_fields = spec._flat_fields()
            assert tuple(original_fields) == tuple(fields)[:-1]
            assert spec.default_config() == default
            assert spec.autotune_reference_config() == reference
        selected = spec.normalized_config(
            helion.Config(
                block_sizes=[32],
                num_warps=4,
                pid_type="flat",
                cute_grid_work_order=("longest_first",),
            )
        )
        assert selected[CUTE_GRID_WORK_ORDER_KEY] == ("longest_first",)
        generation = spec.create_config_generation()
        for enabled in (False, True):
            config = spec.normalized_config(
                helion.Config.from_dict(
                    reference.config
                    | {
                        CUTE_GRID_WORK_ORDER_KEY: [
                            "longest_first" if enabled else "identity"
                        ]
                    }
                )
            )
            assert generation.unflatten(generation.flatten(config)) == config
            value = config.config.get(CUTE_GRID_WORK_ORDER_KEY, field.default())
            assert field.encode(value) == ([0.0, 1.0] if enabled else [1.0, 0.0])
            assert value not in field.pattern_neighbors(value)
        without = {
            key: value
            for key, value in fields.items()
            if key != CUTE_GRID_WORK_ORDER_KEY
        }
        with (
            patch.object(spec, "_flat_fields", return_value=without),
            pytest.raises(InvalidConfig, match="eligible scalar work axis"),
        ):
            spec.normalized_config(selected)


@pytest.mark.parametrize(
    "value",
    (True, "longest_first", [], [True], ["unknown"], ["identity", "longest_first"]),
)
def test_public_invalid_policy_rejects(value):
    with _cpu_codegen():
        bound = _ragged_sum._bind_isolated(_args())
        with pytest.raises(InvalidConfig, match="one identity/longest_first"):
            bound.config_spec.normalized_config(
                helion.Config(cute_grid_work_order=value)
            )


def test_public_unsupported_backend_and_persistent_reject():
    with _cpu_codegen():
        spec = _ragged_sum._bind_isolated(_args()).config_spec
        with patch.object(spec, "backend_name", "triton"):
            assert CUTE_GRID_WORK_ORDER_KEY not in spec._flat_fields()
            with pytest.raises(InvalidConfig, match="CuTe grid axis"):
                spec._normalize_cute_grid_work_order(
                    {CUTE_GRID_WORK_ORDER_KEY: ["identity"]}
                )
        with pytest.raises(InvalidConfig, match="original flat"):
            spec._normalize_cute_grid_work_order(
                {
                    CUTE_GRID_WORK_ORDER_KEY: ["longest_first"],
                    "pid_type": "persistent_blocked",
                }
            )


def test_original_partial_warp_launch_rejects():
    args = _args()
    with _cpu_codegen():
        bound = _ragged_sum._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)),
            pytest.raises(exc.BackendUnsupported, match="full active warps"),
        ):
            bound.to_code(
                helion.Config(
                    block_sizes=[8],
                    num_warps=4,
                    pid_type="flat",
                    cute_grid_work_order=_policies(bound, True),
                )
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "argument_name",
        "argument_host",
        "argument_binding",
        "metadata_mask",
        "metadata_index",
        "dependent_root",
        "implicit_dependency",
        "range_step",
        "effect",
    ),
)
def test_actual_admission_and_same_object_facts_reject(mutation):
    original = GenerateAST._try_lower_direct_affine_root
    hits = []

    def observe(cg, grid, body):
        plan = work_order.discover_work_order(cg, grid.block_ids[0])
        df = cg.device_function
        ir = HostFunction.current().device_ir
        metadata = next(
            node for node in plan.scalar.nodes if node.target is memory_ops.load
        )
        source = metadata.args[0]
        assert isinstance(source, Node)
        argument = df._tensor_args[source.meta["val"]]
        if mutation == "argument_name":
            change = patch.object(argument, "name", argument.name + "_foreign")
        elif mutation == "argument_host":
            change = patch.object(argument, "_host_str", "foreign_argument")
        elif mutation == "argument_binding":
            change = patch.dict(
                df._tensor_args,
                {source.meta["val"]: dataclasses.replace(argument)},
            )
        elif mutation == "metadata_mask":
            change = _node_args(metadata, (*metadata.args[:2], False, None))
        elif mutation == "metadata_index":
            change = _node_args(metadata, (metadata.args[0], [5], None, None))
        elif mutation == "dependent_root":
            edge = TileDependency(ir.root_ids[0], ir.root_ids[0], 0, frozenset(), ())
            change = patch.object(
                ir,
                "tile_dependency_graph",
                TileDependencyGraph(tuple(ir.task_families), (), (edge,)),
            )
        elif mutation == "implicit_dependency":
            change = patch.object(
                ir, "implicit_dependency_starts", frozenset(ir.root_ids)
            )
        elif mutation == "range_step":
            change = _node_args(plan.loop, (*plan.loop.args[:4], [0]))
        else:
            store = next(
                node
                for info in cg.codegen_graphs
                for node in info.graph.nodes
                if node.target is memory_ops.store
            )
            change = patch.object(store, "target", torch.ops.aten._assert_async.msg)
        with change:
            with pytest.raises(work_order.UnsupportedWorkOrder):
                plan.check(cg)
            if not mutation.startswith("argument_"):
                with pytest.raises(work_order.UnsupportedWorkOrder):
                    work_order.discover_work_order(cg, plan.axis)
        hits.append(mutation)
        return original(cg, grid, body)

    args = _args()
    with (
        _cpu_codegen(),
        patch.object(GenerateAST, "_try_lower_direct_affine_root", observe),
    ):
        bound = _ragged_sum._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)):
            bound.to_code(helion.Config(block_sizes=[32], num_warps=4, pid_type="flat"))
    assert hits == [mutation]
