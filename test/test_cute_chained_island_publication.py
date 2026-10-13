from __future__ import annotations

import ast
import copy
import operator
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_register_emission import _register_config
import helion
from helion import exc
from helion._compiler.cute import chained_island_publication as publication_module
from helion._compiler.cute import chained_preparation_actions as actions
from helion._compiler.cute import chained_preparation_storage as storage
from helion._compiler.cute.chained_matmul import _ancestors
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_register_fragments import plan_warp_fragment_map
from helion._compiler.cute.chained_tcgen05 import _layout
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _nonlinear_polynomial_sequence(x, y, initial):
    steps, rows, width = x.shape
    history = torch.empty((steps, rows, width), dtype=torch.float32, device=x.device)
    final = torch.empty_like(initial)
    for _ in hl.tile(rows, block_size=32):
        initial_rows, initial_columns = hl.arange(32), hl.arange(width)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            i, j, columns = hl.arange(32), hl.arange(32), hl.arange(width)
            base = hl.dot(
                x[step.id, i, columns],
                y[step.id, j, columns].T,
                out_dtype=torch.float32,
            )
            diagonal = (i[:, None] // 16) == (j[None, :] // 16)
            factor = torch.where(diagonal, base, 0.0).to(x.dtype)
            square = hl.dot(factor, factor, out_dtype=torch.float32)
            cube = hl.dot(square.to(x.dtype), factor, out_dtype=torch.float32)
            coefficient = torch.exp((square + cube) * 0.0625).to(x.dtype)
            left = hl.dot(coefficient, base.to(x.dtype), out_dtype=torch.float32)
            right = hl.dot(coefficient, left.to(x.dtype), out_dtype=torch.float32)
            frontier = (right + coefficient.float()).to(x.dtype)
            state = hl.dot(
                frontier, state.to(x.dtype), acc=state, out_dtype=torch.float32
            )
            history[step.id, i, columns] = state
        final[initial_rows, initial_columns] = state
    return history, final


def _capture(enabled, *, fixture=None):
    kernel, args = _kda_fixture() if fixture is None else fixture
    build = actions.build_accepted_preparation
    bind = storage.bind_preparation_storage
    bodies, bindings = [], []

    def selected(*args, **kwargs):
        result = build(*args, **kwargs, island_consumers=enabled)
        bodies.append(result)
        return result

    def bound(*args, **kwargs):
        result = bind(*args, **kwargs)
        bindings.append(result)
        return result

    with (
        patch.object(actions, "build_accepted_preparation", selected),
        patch.object(storage, "bind_preparation_storage", bound),
    ):
        source = _source(
            kernel,
            args,
            _register_config(
                3,
                cute_chained_register_islands=True,
                cute_chained_compact_preparation=True,
                **(
                    {}
                    if fixture is None
                    else {
                        "block_sizes": [],
                        "cute_chained_pointwise_cache_bytes": 0,
                        "cute_chained_group_contractions": False,
                        "cute_chained_scan_schedule": "serial",
                    }
                ),
            ),
        )
    assert len(bodies) == len(bindings) == 1
    return source, bodies[0], bindings[0]


def test_unrelated_nonlinear_multiuser_operand_has_real_native_publication():
    dtype = torch.bfloat16
    args = (
        torch.empty((1, 32, 128), dtype=dtype),
        torch.empty((1, 32, 128), dtype=dtype),
        torch.empty((32, 128), dtype=torch.float32),
    )
    source, accepted, bound = _capture(
        True, fixture=(_nonlinear_polynomial_sequence, args)
    )
    assert bound is not None and len(accepted.island_publications) == 1
    item = accepted.island_publications[0].candidate
    assert len(item.operand.users) > 1
    assert len(accepted.island_reads) >= 3
    assert any(
        node.target is torch.ops.aten.exp.default for node in _ancestors(item.operand)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exp2"
        for node in ast.walk(ast.parse(source))
    )


@pytest.fixture(scope="module")
def actual_publication():
    return _capture(True)


def test_actual_island_operand_publishes_without_materialized_exports(
    actual_publication,
):
    source, accepted, bound = actual_publication
    assert bound is not None
    assert len(accepted.island_publications) == 1
    publication = accepted.island_publications[0]
    candidate = publication.candidate
    assert publication.consumed and publication.matches()
    assert all(item.name in bound.physical.omitted for item in candidate.bound.exports)
    assert all(f"{item.name}[" not in source for item in candidate.bound.exports)
    assert f"{candidate.target}[" in source
    assert "\n".join(publication.lines).count("cute.gemm(") == len(
        candidate.bound.island.components[0].issues
    )
    assert len(accepted.island_reads) == 3
    assert len(candidate.operand.users) == 3
    assert publication.lines[-2] == candidate.bound.execution.sync
    assert publication.lines[-1] == f"{candidate.owner} = {candidate.target}"
    assert candidate.operand in dict(publication.outputs)
    assert not any(
        item.node in dict(publication.outputs) for item in candidate.bound.exports
    )


@pytest.mark.parametrize(
    "site",
    [
        "publication",
        "read",
        "body",
        "operand",
        "role",
        "shape",
        "dtype",
        "cast_args",
        "operand_metadata",
        "context",
        "fast_math",
        "config",
        "native_owner",
        "early_start",
        "late_end",
        "reader_context",
        "alias",
        "nested_args",
        "extra_export_user",
    ],
)
def test_stale_actual_publication_cannot_bind_or_consume(actual_publication, site):
    _, accepted, bound = actual_publication
    publication = accepted.island_publications[0]
    item = publication.candidate
    cleanup = None
    if site == "publication":
        target = next(
            action
            for action in accepted.actions
            if action.island_publication is publication
        )
        key, value = "island_publication", None
    elif site == "read":
        target, key, value = accepted, "island_reads", accepted.island_reads[:-1]
    elif site == "body":
        target, key, value = publication, "lines", publication.lines[:-1]
    elif site == "operand":
        target, key, value = item, "operand", item.bound.exports[0].node
    elif site == "role":
        target, key, value = item, "role", "b" if item.role == "a" else "a"
    elif site == "shape":
        target, key, value = item, "shape", (item.shape[0], item.shape[1] // 2)
    elif site == "dtype":
        target, key, value = item, "dtype", torch.float32
    elif site == "cast_args":
        assert item.operand.target is torch.ops.prims.convert_element_type.default
        target, key, value = item.operand, "args", (item.operand.args[0], torch.float32)
    elif site == "operand_metadata":
        target, key, value = (
            item.operand,
            "meta",
            {**item.operand.meta, "val": torch.empty(item.shape, dtype=torch.float32)},
        )
    elif site == "context":
        target, key, value = item.bound.execution, "sync", "unapproved_join()"
    elif site == "fast_math":
        target, key, value = item.settings, "fast_math", False
    elif site == "config":
        target, key, value = (
            item.cg.device_function.config,
            "config",
            {
                **item.cg.device_function.config.config,
                "cute_chained_pointwise_vectorize": False,
            },
        )
    elif site == "native_owner":
        target, key, value = (
            item.stage.a if item.role == "a" else item.stage.b,
            "byte_size",
            128,
        )
    elif site in ("early_start", "late_end"):
        target = item.stage.a if item.role == "a" else item.stage.b
        key, value = ("live_from", 0) if site == "early_start" else ("live_until", 0)
    elif site == "reader_context":
        target, key, value = (
            accepted.island_reads[1].action.proof.execution,
            "warp",
            "unapproved_warp",
        )
    elif site == "alias":
        target = accepted.island_reads[-1].action
        key, value = (
            "outputs",
            tuple(
                (node, "unapproved_alias" if node is item.operand else name)
                for node, name in target.outputs
            ),
        )
    elif site == "nested_args":
        target, key, value = (
            item.plan.dots[item.consumer],
            "kwargs",
            {"unapproved_nested": {"source": item.bound.exports[0].node}},
        )
    else:
        graph = item.operand.graph
        user = graph.call_function(operator.neg, (item.bound.exports[0].node,))

        def cleanup():
            graph.erase_node(user)

        target, key, value = publication, "consumed", publication.consumed
    old = getattr(target, key)
    object.__setattr__(target, key, value)
    try:
        assert not accepted.matches(
            item.plan, item.pipeline, dict(accepted.revision.shapes)
        )
        assert not bound.matches(item.plan, item.pipeline)
    finally:
        object.__setattr__(target, key, old)
        if cleanup is not None:
            cleanup()


def test_publication_is_once_only_and_checks_before_map_mutation(actual_publication):
    _, accepted, _ = actual_publication
    publication = accepted.island_publications[0]
    copied = dict(publication.candidate.boundary_owner)
    lines = list(publication.lines)
    saved = list(lines)
    with pytest.raises(_UnsupportedChain):
        publication.record(lines, copied)
    assert lines == saved and copied == publication.candidate.boundary_owner


@pytest.mark.parametrize(
    "field", ["byte_offset", "byte_size", "live_from", "live_until"]
)
def test_actual_physical_owner_cannot_be_relocated_truncated_or_released(
    actual_publication, field
):
    _, accepted, bound = actual_publication
    item = accepted.island_publications[0].candidate
    region = bound.physical.layout.region(item.owner)
    old = getattr(region, field)
    value = {
        "byte_offset": region.byte_offset + 128,
        "byte_size": 128,
        "live_from": item.bound.stop_event,
        "live_until": item.bound.stop_event,
    }[field]
    object.__setattr__(region, field, value)
    try:
        assert not bound.matches(item.plan, item.pipeline)
    finally:
        object.__setattr__(region, field, old)


def test_failure_after_real_publication_cannot_install_an_accepted_body():
    original = publication_module.IslandConsumerPublication.consume
    observed = []

    def consume(self, *args, **kwargs):
        original(self, *args, **kwargs)
        observed.append(self)
        raise _UnsupportedChain("injected original remaining producer failure")

    args = (
        torch.empty((1, 32, 128), dtype=torch.bfloat16),
        torch.empty((1, 32, 128), dtype=torch.bfloat16),
        torch.empty((32, 128)),
    )
    with (
        patch.object(publication_module.IslandConsumerPublication, "consume", consume),
        patch.object(
            storage,
            "bind_preparation_storage",
            side_effect=AssertionError("failed body reached physical binding"),
        ),
        pytest.raises(
            exc.InternalError, match="injected original remaining producer failure"
        ),
    ):
        _capture(True, fixture=(_nonlinear_polynomial_sequence, args))
    assert len(observed) == 1 and observed[0].consumed


def test_actual_cute_cells_and_complement_cover_exact_native_owner(actual_publication):
    _, accepted, bound = actual_publication
    publication = accepted.island_publications[0]
    item = publication.candidate
    mapping = plan_warp_fragment_map("c", torch.float32)
    assert mapping is not None
    axes = next(
        origin.axes
        for origin in item.bound.origins
        if origin.node is item.bound.exports[0].node
    )
    active = []
    for component in range(len(item.bound.island.components)):
        for lane in range(32):
            for row, column in mapping.source_coordinates[lane]:
                active.append(
                    (
                        axes[0][0] + component * axes[0][1] + row,
                        axes[1][0] + component * axes[1][1] + column,
                    )
                )
    assert len(active) == len(set(active))
    all_cells = {(r, c) for r in range(item.shape[0]) for c in range(item.shape[1])}
    complement = all_cells - set(active)
    zero_owners = {
        (thread + step * item.bound.execution.threads)
        for thread in range(item.bound.execution.threads)
        for step in range(
            (len(all_cells) + item.bound.execution.threads - 1)
            // item.bound.execution.threads
        )
        if thread + step * item.bound.execution.threads < len(all_cells)
    }
    assert zero_owners == set(range(len(all_cells)))
    assert set(active) | complement == all_cells and not (set(active) & complement)
    dtype = "cutlass.BFloat16" if item.dtype is torch.bfloat16 else "cutlass.Float16"
    for statement in _layout(item.target, item.shape, 1, dtype):
        assert statement in publication.lines
        assert statement in accepted.island_reads[0].lines
    region = bound.physical.layout.region(item.owner)
    assert region.byte_size == len(all_cells) * item.dtype.itemsize
    assert region.live_from <= item.bound.first_event
    assert region.live_until >= max(
        reader.action.stop for reader in accepted.island_reads
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nonfinite_and_signed_zero_original_expression_is_not_zero_filled(dtype):
    # These are original FP32 export images, not a change to the island's own
    # already-selected fast-math reduction. Both publication paths evaluate
    # the unchanged downstream expression at the original cells.
    bits = torch.tensor(
        [0, -2147483648, 1, -2147483647, 2139095040, -8388608, 2143289345, -4194303],
        dtype=torch.int32,
    )
    values = bits.view(torch.float32).repeat(32).reshape(16, 16)
    image = torch.zeros((32, 32), dtype=torch.float32)
    image[:16, :16] = values
    image[16:, 16:] = values.T
    expected = torch.exp((image + image) * 0.0625).to(dtype)
    published = torch.empty_like(expected)
    published[:16, :16] = torch.exp((values + values) * 0.0625).to(dtype)
    published[16:, 16:] = torch.exp((values.T + values.T) * 0.0625).to(dtype)
    published[:16, 16:] = torch.exp(torch.tensor(0.0, dtype=torch.float32)).to(dtype)
    published[16:, :16] = torch.exp(torch.tensor(0.0, dtype=torch.float32)).to(dtype)
    assert torch.equal(published.view(torch.int16), expected.view(torch.int16))
    assert torch.all(published[:16, 16:] == 1)


@pytest.mark.parametrize("field", ["_publication_receipts", "_publication_reads"])
def test_dropped_completed_inventory_rejects_before_seal(field):
    original = actions._PreparationRecorder.seal

    def seal(self, *args, **kwargs):
        assert self._publication_receipts and self._publication_reads
        setattr(self, field, ())
        return original(self, *args, **kwargs)

    with (
        patch.object(actions._PreparationRecorder, "seal", seal),
        pytest.raises((exc.BackendUnsupported, _UnsupportedChain)),
    ):
        _capture(True)


def _expanded_value(lines, value, exports):
    """Compare scalar lowering trees, not generated-kernel text rewrites."""
    definitions = {}
    for statement in ast.parse("\n".join(lines)).body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            definitions[statement.targets[0].id] = statement.value

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in definitions:
                return self.visit(copy.deepcopy(definitions[node.id]))
            return node

        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in exports:
                return ast.parse(exports[node.value.id], mode="eval").body
            return self.generic_visit(node)

    return ast.dump(
        Expand().visit(ast.parse(value, mode="eval").body), include_attributes=False
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_original_expression_cast_and_zero_fallback_trees(dtype):
    original = publication_module.IslandConsumerCandidate.point
    checked = []

    def point(self, coordinates, values):
        lines = original(self, coordinates, values)
        ordinary = _Expression(
            self.cg,
            self.plan,
            {
                **dict(self.inputs),
                **{item.node: item.name for item in self.bound.exports},
            },
        )
        value = ordinary.value(self.operand, coordinates)
        tree = ast.parse("\n".join(lines))
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
        body = tree.body[0].body
        store = body[-1]
        assert isinstance(store, ast.Assign) and isinstance(
            store.targets[0], ast.Subscript
        )
        normal = _expanded_value(
            ordinary.lines,
            value,
            {item.name: values[item.node] for item in self.bound.exports},
        )
        actual = _expanded_value(
            [ast.unparse(node) for node in body[:-1]], ast.unparse(store.value), {}
        )
        # The original operand's conversion is already part of `value`; the
        # original fill additionally applies exactly its output dtype cast.
        cast_name = "cutlass.BFloat16" if dtype is torch.bfloat16 else "cutlass.Float16"
        expected = ast.parse(f"{cast_name}(x)", mode="eval").body
        assert isinstance(expected, ast.Call)
        expected.args[0] = ast.parse(value, mode="eval").body
        expected_tree = _expanded_value(
            ordinary.lines,
            ast.unparse(expected),
            {item.name: values[item.node] for item in self.bound.exports},
        )
        assert actual in (normal, expected_tree)
        fallback = tree.body[0].orelse[-1]
        assert isinstance(fallback, ast.Assign)
        assert ast.unparse(fallback.value) == f"{cast_name}(0)"
        checked.append(tuple(values.values()))
        return lines

    args = (
        torch.empty((1, 32, 128), dtype=dtype),
        torch.empty((1, 32, 128), dtype=dtype),
        torch.empty((32, 128)),
    )
    with patch.object(publication_module.IslandConsumerCandidate, "point", point):
        _capture(True, fixture=(_nonlinear_polynomial_sequence, args))
    assert len(checked) == 2
    assert all(value == "cutlass.Float32(0)" for value in checked[1])


def test_default_private_selector_has_no_new_discovery():
    with patch.object(
        publication_module,
        "plan_island_consumer",
        side_effect=AssertionError("default discovered publication"),
    ):
        source, accepted, _ = _capture(False)
    assert not accepted.island_publications and not accepted.island_reads
    assert "consumer_zero" not in source
