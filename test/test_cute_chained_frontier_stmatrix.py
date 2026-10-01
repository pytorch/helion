from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_frontier_ownership import _native_frame
from .test_cute_chained_frontier_ownership_integration import _config
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from helion import exc
from helion._compiler.cute import chained_frontier_groups as groups
from helion._compiler.cute.chained_frontier_groups import FrontierStores
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership

KEY = "cute_chained_frontier_stmatrix"


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_final_typed_group_selection_keeps_native_binding_and_activation(dtype):
    frame, group, _operands, bindings = _native_frame(dtype)
    ownership = plan_vector_ownership((32, 32), 128, tile_columns=32)
    assert ownership is not None
    policy = FrontierStores(True)
    selected = policy.select(frame, group, bindings, ownership)
    assert len(selected) == 1 and not policy.activated
    result = next(iter(selected.values()))
    assert result.dtype == dtype and result.full_shape == (64, 32)
    member = next(
        member
        for member in bindings[0].candidate.members
        if member.logical_modes == (1, 0)
    )
    assert result.row_offset == member.row_offset
    with pytest.raises(
        exc.BackendUnsupported, match="successfully emitted native store"
    ):
        policy.validate()
    policy.activated = True
    policy.validate()


def test_missing_stale_and_ordinary_native_views_do_not_license_stmatrix():
    frame, group, _operands, bindings = _native_frame()
    ownership = plan_vector_ownership((32, 32), 128)
    assert ownership is not None
    policy = FrontierStores(True)
    assert not policy.select(frame, group, (), ownership)
    assert not policy.select(frame, replace(group, stop_event=99), bindings, ownership)
    assert not policy.select(
        frame, group, (replace(bindings[0], byte_offset=128),), ownership
    )
    with patch.object(groups, "plan_native_stmatrix_store", return_value=None):
        assert not policy.select(frame, group, bindings, ownership)
    assert not policy.activated


@pytest.mark.parametrize("invalid", ("stage_region", "graph_revision", "ready_action"))
def test_invalid_complete_frame_rejects_before_native_store_selection(invalid):
    frame, group, _operands, bindings = _native_frame()
    if invalid == "stage_region":
        stage = frame.stages[0]
        changed = replace(
            stage, a=replace(stage.a, byte_offset=stage.a.byte_offset + 128)
        )
        frame = replace(frame, stages=(changed, *frame.stages[1:]))
    elif invalid == "graph_revision":
        frame.cut.region.graph.placeholder("added_after_planning")
    else:
        assert frame.actions[-1].kind == "ready"
        frame = replace(
            frame, actions=(*frame.actions[:-1], replace(frame.actions[-1], kind="mma"))
        )
    ownership = plan_vector_ownership((32, 32), 128)
    assert ownership is not None
    policy = FrontierStores(True)
    with patch.object(
        groups,
        "plan_native_stmatrix_store",
        side_effect=AssertionError("invalid frame"),
    ):
        assert not policy.select(frame, group, bindings, ownership)
    assert not policy.activated


@pytest.mark.parametrize("value", (0, 1, None, "true", [], {}))
def test_native_store_option_requires_strict_bool(value):
    with pytest.raises(ValueError, match="bool"):
        FrontierStores(value)


def test_default_option_does_not_discover_or_change_original_source():
    kernel, args = _kda_fixture()
    config = _config(32)
    with patch.object(
        FrontierStores, "select", side_effect=AssertionError("default discovery")
    ):
        before = _source(kernel, args, config)
        config.config[KEY] = False
        assert _source(kernel, args, config) == before


def _restore_stores(source: str) -> str:
    tree = ast.parse(source)
    targets = {}
    prefixes = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name.endswith("_copy") and "StMatrix8x8x16bOp" in ast.unparse(
                node.value
            ):
                prefixes.add(name.removesuffix("_copy"))
    assert len(prefixes) == 1
    prefix = prefixes.pop()
    target = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == prefix + "_target"
    )
    assert isinstance(target.value, ast.Call)
    targets[prefix] = ast.unparse(target.value.args[0])
    group = prefix.rsplit("_output_", 1)[0]

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            lhs = node.targets[0]
            if isinstance(lhs, ast.Name) and lhs.id in {
                prefix + suffix for suffix in ("_copy", "_target", "_values")
            }:
                return None
            if (
                isinstance(lhs, ast.Subscript)
                and ast.unparse(lhs.value) == prefix + "_values"
            ):
                node.targets[0] = ast.parse(
                    f"{targets[prefix]}[{group}_row + 0, {group}_base + {group}_element]",
                    mode="eval",
                ).body
                assert isinstance(node.targets[0], ast.Subscript)
                node.targets[0].ctx = ast.Store()
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.unparse(node).startswith(f"cute.copy({prefix}_copy,"):
                return None
            return self.generic_visit(node)

    return ast.dump(Restore().visit(tree))


def test_actual_frontier_preserves_every_original_expression_and_barrier():
    kernel, args = _kda_fixture()
    config = _config(32)
    before = _source(kernel, args, config)
    config.config[KEY] = True
    after = _source(kernel, args, config)
    assert _restore_stores(after) == ast.dump(ast.parse(before))


def test_unsupported_geometry_and_failed_group_do_not_fake_store_activation():
    kernel, args = _kda_fixture()
    config = _config(0)
    config.config[KEY] = True
    with pytest.raises(
        exc.BackendUnsupported, match="successfully emitted native store"
    ):
        _source(kernel, args, config)
    config = _config(32)
    config.config[KEY] = True
    original = groups.emit_materialized_group
    attempts = []

    def reject(*args, **kwargs):
        outputs = args[3]
        if any(output.native_store is not None for output in outputs):
            attempts.append(True)
            return None
        return original(*args, **kwargs)

    with (
        patch.object(groups, "emit_materialized_group", reject),
        pytest.raises(exc.BackendUnsupported),
    ):
        _source(kernel, args, config)
    assert attempts
