from __future__ import annotations

import ast
import dataclasses
import operator
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import direct_affine_replay as replay_impl
from helion._compiler.cute.direct_affine_plan import DirectAffineCoefficientLayout
from helion._compiler.cute.direct_affine_plan import DirectAffineMma
from helion._compiler.cute.direct_affine_plan import DirectAffinePhaseOrder
from helion._compiler.cute.direct_affine_plan import DirectAffineStateAccessProof
from helion._compiler.cute.direct_affine_plan import DirectAffineStateIngress
from helion._compiler.cute.direct_affine_plan import resolve_direct_affine_plan
from helion._compiler.cute.direct_affine_replay import DirectAffineCoordinates
from helion._compiler.cute.direct_affine_replay import DirectAffineEffectReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineNodeReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineOrdinaryAxis
from helion._compiler.cute.direct_affine_replay import DirectAffineReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineReplayBindings
from helion._compiler.cute.direct_affine_replay import DirectAffineReplayProof
from helion._compiler.cute.direct_affine_replay import DirectAffineResolvedTemplates
from helion._compiler.cute.direct_affine_replay import DirectAffineSharedBuffer
from helion._compiler.cute.direct_affine_replay import DirectAffineStateStoreReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineStepReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineValueReplay
from helion._compiler.cute.direct_affine_replay import compose_direct_affine_replacement
from helion._compiler.cute.direct_affine_replay import instantiate_direct_affine_value
from helion._compiler.cute.direct_affine_replay import replace_direct_affine_replay
from helion._compiler.cute.direct_affine_replay import resolve_direct_affine_coordinates
from helion._compiler.cute.direct_affine_replay import resolve_direct_affine_replay
from helion._compiler.cute.direct_affine_replay import resolve_direct_affine_templates

validate_direct_affine_templates = replay_impl._validate_direct_affine_templates


def _expression(source: str) -> ast.expr:
    result = ast.parse(source, mode="eval").body
    assert isinstance(result, ast.expr)
    return result


def _statements(source: str) -> tuple[ast.stmt, ...]:
    return tuple(ast.parse(source).body)


def _ast_bytes(statements: list[ast.AST] | tuple[ast.AST, ...]) -> bytes:
    return ast.dump(
        ast.Module(body=list(statements), type_ignores=[]),
        include_attributes=True,
    ).encode()


class _FakeCodegen:
    def __init__(self) -> None:
        self._statements_by_owner_node_id: dict[
            int, list[tuple[list[ast.AST], ast.AST]]
        ] = {}
        self._nodes: dict[int, torch.fx.Node] = {}
        self._results: dict[int, object] = {}
        self.referenced_thread_block_dims = [1, 1, 1]

    def record(
        self,
        node: torch.fx.Node,
        result: object,
        body: list[ast.AST],
        *statements: ast.AST,
    ) -> None:
        self._nodes[id(node)] = node
        self._results[id(node)] = result
        self._statements_by_owner_node_id[id(node)] = [
            (body, statement) for statement in statements
        ]

    def statements_owned_by_node(
        self, node: torch.fx.Node
    ) -> tuple[tuple[list[ast.AST], ast.AST], ...]:
        return tuple(self._statements_by_owner_node_id.get(id(node), ()))

    def codegen_result_for_node(self, node: torch.fx.Node) -> tuple[bool, object]:
        if id(node) not in self._results:
            return False, None
        return True, self._results[id(node)]

    def replace_owned_statement_span(
        self,
        body: list[ast.AST],
        nodes: tuple[torch.fx.Node, ...],
        replacement: tuple[ast.AST, ...],
    ) -> bool:
        positions = {id(statement): index for index, statement in enumerate(body)}
        claimed = [
            statement
            for node in nodes
            for owner_body, statement in self.statements_owned_by_node(node)
            if owner_body is body
        ]
        if not claimed or any(id(statement) not in positions for statement in claimed):
            return False
        first = min(positions[id(statement)] for statement in claimed)
        last = max(positions[id(statement)] for statement in claimed)
        if {id(statement) for statement in body[first : last + 1]} != {
            id(statement) for statement in claimed
        }:
            return False
        body[first : last + 1] = replacement
        for node in nodes:
            self._statements_by_owner_node_id.pop(id(node), None)
        return True


@dataclasses.dataclass(frozen=True)
class _Access:
    node: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class _Fence:
    preceding: _Access | None
    following: _Access | None


def _ownership_case() -> tuple[
    Any,
    object,
    _FakeCodegen,
    list[ast.AST],
    tuple[torch.fx.Node, ...],
]:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    before = graph.call_function(operator.neg, (source,))
    producer = graph.call_function(operator.add, (source, 1))
    expression_only = graph.call_function(operator.mul, (producer, 2))
    owned = graph.call_function(operator.neg, (expression_only,))
    after = graph.call_function(operator.neg, (owned,))
    graph.output(after)

    statements = [
        *_statements("before_value = 0"),
        *_statements("producer_value = opaque(source)"),
        *_statements("owned_value = -producer_value"),
        *_statements("after_value = 1"),
    ]
    codegen = _FakeCodegen()
    codegen.record(before, _expression("before_value"), statements, statements[0])
    codegen.record(producer, _expression("producer_value"), statements, statements[1])
    codegen.record(expression_only, _expression("producer_value * 2"), statements)
    codegen.record(owned, _expression("owned_value"), statements, statements[2])
    codegen.record(after, _expression("after_value"), statements, statements[3])

    region = SimpleNamespace(
        producer_nodes=(producer,),
        owned_nodes=(expression_only, owned),
        source_interval=(producer, expression_only, owned),
        access_fences=(_Fence(_Access(before), _Access(after)),),
    )
    candidate = SimpleNamespace(region=region)
    return (
        candidate,
        object(),
        codegen,
        statements,
        (
            before,
            producer,
            expression_only,
            owned,
            after,
        ),
    )


def test_replay_resolves_exact_span_results_and_fences() -> None:
    candidate, graph_info, codegen, body, nodes = _ownership_case()
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )

    assert replay is not None
    assert replay.source_span == (1, 2)
    assert replay.statement_nodes == (nodes[1], nodes[3])
    expression_only = replay.node_replay(nodes[2])
    assert expression_only is not None
    assert expression_only.statements == ()
    assert ast.unparse(cast("ast.AST", expression_only.result)) == "producer_value * 2"


@pytest.mark.parametrize(
    "failure",
    ["gap", "missing_result", "fence", "shared_owner", "cross_container_owner"],
)
def test_replay_rejection_is_byte_for_byte_non_mutating(failure: str) -> None:
    candidate, graph_info, codegen, body, nodes = _ownership_case()
    if failure == "gap":
        body.insert(2, ast.Pass())
    elif failure == "missing_result":
        codegen._results.pop(id(nodes[2]))
    elif failure == "fence":
        body.remove(body[-1])
        body.insert(2, codegen.statements_owned_by_node(nodes[-1])[0][1])
    elif failure == "shared_owner":
        codegen._statements_by_owner_node_id[id(nodes[2])] = [
            (body, codegen.statements_owned_by_node(nodes[1])[0][1])
        ]
    else:
        codegen._statements_by_owner_node_id[-1] = [
            ([ast.Pass()], codegen.statements_owned_by_node(nodes[1])[0][1])
        ]
    before_body = _ast_bytes(body)
    before_owners = {
        owner: tuple((id(container), id(statement)) for container, statement in entries)
        for owner, entries in codegen._statements_by_owner_node_id.items()
    }
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        assert (
            resolve_direct_affine_replay(
                cast("Any", candidate), cast("Any", graph_info), codegen, body
            )
            is None
        )
    assert _ast_bytes(body) == before_body
    assert {
        owner: tuple((id(container), id(statement)) for container, statement in entries)
        for owner, entries in codegen._statements_by_owner_node_id.items()
    } == before_owners


def _plan(
    step_count: int,
    mma: DirectAffineMma,
    rows: int = 16,
    *,
    ingress: DirectAffineStateIngress = DirectAffineStateIngress.SYNC,
):
    plan = resolve_direct_affine_plan(
        step_count=step_count,
        row_extent=rows,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=mma,
        coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
        state_ingress=ingress,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
        state_access_proof=DirectAffineStateAccessProof(
            source_vector_alignment_bytes=16,
            destination_vector_alignment_bytes=16,
            source_feature_stride_one=True,
            full_vector_coverage=True,
            cp_async_supported=ingress is DirectAffineStateIngress.ASYNC,
        ),
    )
    assert plan is not None
    return plan


def _thread_index(axis: int) -> ast.expr:
    return _expression(f"cute.arch.thread_idx()[{axis}]")


def _coordinates(plan) -> DirectAffineCoordinates:
    row = DirectAffineOrdinaryAxis(
        source=_expression("source_row"),
        tile_offset=_expression("row_tile_offset"),
        local_expression=_expression(
            "cute.arch.thread_idx()[1] * 16 + ordinary_row_serial"
        ),
        extent=plan.row_extent,
        thread_axis=1,
        thread_extent=plan.row_warps,
        lane_name="ordinary_row_serial",
        lane_extent=16,
    )
    feature = DirectAffineOrdinaryAxis(
        source=_expression("source_feature"),
        tile_offset=ast.Constant(value=0),
        local_expression=_expression(
            "ordinary_feature_serial * 32 + cute.arch.thread_idx()[0]"
        ),
        extent=plan.feature_extent,
        thread_axis=0,
        thread_extent=32,
        lane_name="ordinary_feature_serial",
        lane_extent=4,
    )
    result = resolve_direct_affine_coordinates(plan, row=row, feature=feature)
    assert result is not None
    return result


def test_coordinate_resolution_accepts_permuted_complete_axes() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    coordinates = _coordinates(plan)
    assert coordinates.row.extent == 16
    assert coordinates.feature.extent == 128

    malformed = dataclasses.replace(
        coordinates.feature,
        local_expression=_expression("cute.arch.thread_idx()[0] * 4"),
    )
    assert (
        resolve_direct_affine_coordinates(plan, row=coordinates.row, feature=malformed)
        is None
    )


def test_coordinate_resolution_rejects_joint_axis_alias() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    coordinates = _coordinates(plan)
    row = dataclasses.replace(
        coordinates.row,
        local_expression=_expression("cute.arch.thread_idx()[0] + ordinary_row_serial"),
        thread_axis=0,
    )

    assert (
        resolve_direct_affine_coordinates(
            plan,
            row=row,
            feature=coordinates.feature,
        )
        is None
    )


def test_opaque_replay_preserves_renamed_permuted_formula() -> None:
    statements = _statements(
        "renamed_right = opaque(feature_coordinate, fixed_parameter)\n"
        "renamed_left = bias + renamed_right * multiplier"
    )
    replay = DirectAffineValueReplay(
        statements=statements,
        value=_expression("renamed_left"),
        bindings=DirectAffineReplayBindings(
            feature=_expression("feature_coordinate"),
        ),
    )
    before = _ast_bytes(statements)
    instantiated = instantiate_direct_affine_value(
        replay,
        feature=_expression("lane * 4 + element"),
        suffix="replayed_7",
    )
    assert instantiated is not None
    emitted, value = instantiated
    source = "\n".join(ast.unparse(statement) for statement in emitted)
    assert "opaque(lane * 4 + element, fixed_parameter)" in source
    assert "bias + renamed_right_replayed_7 * multiplier" in source
    assert ast.unparse(value) == "renamed_left_replayed_7"
    assert _ast_bytes(statements) == before


def test_opaque_replay_renaming_avoids_reserved_program_names() -> None:
    replay = DirectAffineValueReplay(
        statements=_statements("temporary = opaque(source_feature)"),
        value=_expression("temporary + temporary_replayed"),
        bindings=DirectAffineReplayBindings(
            feature=_expression("source_feature"),
        ),
    )

    instantiated = instantiate_direct_affine_value(
        replay,
        feature=_expression("direct_feature"),
        suffix="replayed",
        reserved_names=frozenset(("temporary_replayed",)),
    )

    assert instantiated is not None
    statements, value = instantiated
    assert ast.unparse(statements[0]).startswith("temporary_replayed_1 =")
    assert ast.unparse(value) == "temporary_replayed_1 + temporary_replayed"


@pytest.mark.parametrize(
    "statement",
    [
        "value = [value for value in values]",
        "match source_feature:\n    case captured:\n        value = captured",
    ],
)
def test_opaque_replay_rejects_statement_scope(statement: str) -> None:
    replay = DirectAffineValueReplay(
        statements=_statements(statement),
        value=_expression("value"),
    )

    assert instantiate_direct_affine_value(replay, suffix="replayed") is None


@pytest.mark.parametrize(
    "value",
    [
        "[source_feature for source_feature in values]",
        "(source_feature := 1) + source_feature",
    ],
)
def test_opaque_replay_rejects_scoped_expression_output(value: str) -> None:
    replay = DirectAffineValueReplay(
        statements=(),
        value=_expression(value),
        bindings=DirectAffineReplayBindings(
            feature=_expression("source_feature"),
        ),
    )

    assert (
        instantiate_direct_affine_value(
            replay,
            feature=_expression("direct_feature"),
            suffix="replayed",
        )
        is None
    )


def test_overlapping_coordinate_alias_uses_outer_substitution() -> None:
    replay = DirectAffineValueReplay(
        statements=(),
        value=_expression("row_tile + ordinary_row"),
        bindings=DirectAffineReplayBindings(
            row=_expression("row_tile + ordinary_row"),
            row_local=_expression("ordinary_row"),
        ),
    )

    instantiated = instantiate_direct_affine_value(
        replay,
        row=_expression("direct_global_row"),
        row_local=_expression("direct_local_row"),
        suffix="replayed",
    )

    assert instantiated is not None
    assert ast.unparse(instantiated[1]) == "direct_global_row"


@pytest.mark.parametrize(
    "pointer",
    [
        (
            "other.iterator + batch * state.layout.stride[0] + "
            "slot * state.layout.stride[1] + row * state.layout.stride[2] + "
            "feature * state.layout.stride[3]"
        ),
        (
            "state.iterator + batch * state.layout.stride[0] + "
            "slot * state.layout.stride[1] + row * state.layout.stride[1] + "
            "feature * state.layout.stride[3]"
        ),
        (
            "state.iterator + batch * state.layout.stride[0] + "
            "slot * state.layout.stride[1] + row * state.layout.stride[2] + "
            "feature * 2"
        ),
        (
            "state.iterator + batch * state.layout.stride[0] + "
            "slot * state.layout.stride[1] + row * state.layout.stride[2] + "
            "feature * other.layout.stride[3]"
        ),
        (
            "state.iterator + other.iterator + row * state.layout.stride[2] + "
            "feature * state.layout.stride[3]"
        ),
        (
            "state.iterator + row * state.layout.stride[2] + "
            "feature * state.layout.stride[3] + 1"
        ),
        (
            "state.iterator + row * state.layout.stride[2] + "
            "feature * state.layout.stride[3] + "
            "x * state.layout.stride[3]"
        ),
    ],
)
def test_pointer_contract_rejects_wrong_tensor_or_stride(pointer: str) -> None:
    assert not replay_impl._pointer_has_tensor_layout(
        _expression(pointer),
        argument_name="state",
        row=_expression("row"),
        feature=_expression("feature"),
        row_dimension=2,
        feature_dimension=3,
    )


def test_pointer_contract_accepts_exact_ranked_layout() -> None:
    pointer = _expression(
        "state.iterator + batch * state.layout.stride[0] + "
        "slot * state.layout.stride[1] + cutlass.Int32(row) * "
        "cutlass.Int32(state.layout.stride[2]) + cutlass.Int32(feature) * "
        "cutlass.Int32(state.layout.stride[3])"
    )

    assert replay_impl._pointer_has_tensor_layout(
        pointer,
        argument_name="state",
        row=_expression("row"),
        feature=_expression("feature"),
        row_dimension=2,
        feature_dimension=3,
    )


def _feature_reduction_loop(*extra: str) -> ast.For:
    statements = _statements(
        "for element in range(4):\n"
        "    feature = lane * 4 + element\n"
        "    value = source.iterator.load() + feature\n"
        "    reduced = _helion_lane_reduce(value, 'sum', "
        "cutlass.Float32(0), 32, 1, 0, '', 1, 1)\n"
        + "".join(f"    {line}\n" for line in extra)
        + "    (scratch.iterator + feature).store(value + reduced)"
    )
    loop = statements[0]
    assert isinstance(loop, ast.For)
    setattr(loop, replay_impl.HELION_LANE_LOOP_VAR_ATTR, "element")
    return loop


@pytest.mark.parametrize(
    "statements",
    [
        (
            "partial = source_feature * source_feature\n"
            "reduced = _helion_lane_reduce(partial, 'max', "
            "cutlass.Float32(0), 32, 1, 0, '', 1, 1)"
        ),
        (
            "partial = source_feature * source_feature\n"
            "reduced = _helion_lane_reduce(partial, 'sum', "
            "cutlass.Float32(0), 16, 1, 0, '', 1, 1)"
        ),
        (
            "partial = source_feature * source_feature\n"
            "reduced = _helion_lane_reduce(partial, 'sum', "
            "cutlass.Float32(0), 32, 2, 16, 'lane', 1, 1)"
        ),
        (
            "partial = source_row * source_row\n"
            "reduced = _helion_lane_reduce(partial, 'sum', "
            "cutlass.Float32(0), 32, 1, 0, '', 1, 1)"
        ),
    ],
)
def test_feature_reduction_rejects_unsupported_marker(statements: str) -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    assert not replay_impl._feature_reduction_markers_are_supported(
        _statements(statements),
        _coordinates(plan),
    )


def test_dependency_replay_rejects_marker_owned_by_non_reduction_node() -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    node = graph.call_function(operator.neg, (source,))
    statements = _statements(
        "partial = source_feature * source_feature\n"
        "reduced = _helion_lane_reduce(partial, 'sum', "
        "cutlass.Float32(0), 32, 1, 0, '', 1, 1)"
    )
    replay = DirectAffineReplay(
        candidate=cast(
            "Any",
            SimpleNamespace(
                region=SimpleNamespace(
                    dependency_slices=(SimpleNamespace(value=node, nodes=(node,)),)
                )
            ),
        ),
        nodes=(
            DirectAffineNodeReplay(
                node=node,
                statements=statements,
                result=_expression("reduced"),
            ),
        ),
        statement_nodes=(node, node),
        source_statements=statements,
        source_span=(0, 1),
        fence_nodes=(),
        root_statement_ids=tuple(id(statement) for statement in statements),
        root_statement_dumps=tuple(
            ast.dump(statement, include_attributes=True) for statement in statements
        ),
    )

    assert (
        replay_impl._dependency_value_replay(
            replay,
            node,
            _coordinates(_plan(3, DirectAffineMma.M16N8)),
        )
        is None
    )


def test_feature_reduction_split_is_two_pass_and_all_lane_owned() -> None:
    split = replay_impl._split_feature_reduction_loop(
        _feature_reduction_loop(),
        {
            "scratch": DirectAffineSharedBuffer(
                role="scratch",
                name="scratch",
                dtype_name="Float32",
                element_count=128,
                byte_offset=0,
            )
        },
        "lane",
        frozenset(),
    )

    assert split is not None
    source = "\n".join(ast.unparse(statement) for statement in split)
    assert source.count("cute.arch.warp_reduction_sum") == 1
    assert "_helion_lane_reduce" not in source
    assert "cute.arch.lane_idx" not in source
    assert source.count("for element in range(4)") == 2


def test_feature_reduction_split_rejects_raw_fallback_and_name_capture() -> None:
    buffer = DirectAffineSharedBuffer(
        role="scratch",
        name="scratch",
        dtype_name="Float32",
        element_count=128,
        byte_offset=0,
    )
    carried = _feature_reduction_loop("carry += value")
    assert (
        replay_impl._split_feature_reduction_loop(
            carried,
            {"scratch": buffer},
            "lane",
            frozenset(),
        )
        is None
    )
    colliding = _feature_reduction_loop()
    assert (
        replay_impl._split_feature_reduction_loop(
            colliding,
            {"scratch": buffer},
            "lane",
            frozenset(("reduced_lane_acc",)),
        )
        is None
    )


def _value_replay(label: str, *roles: str) -> DirectAffineValueReplay:
    arguments = ", ".join(f"source_{role}" for role in roles)
    bindings = DirectAffineReplayBindings(
        row=_expression("source_row") if "row" in roles else None,
        feature=_expression("source_feature") if "feature" in roles else None,
    )
    return DirectAffineValueReplay(
        statements=_statements(f"{label}_result = {label}_formula({arguments})"),
        value=_expression(f"{label}_result"),
        bindings=bindings,
    )


def _step_replay(index: int) -> DirectAffineStepReplay:
    logical_name = f"logical_value_{index}"
    output = DirectAffineEffectReplay(
        statements=_statements(
            "if cute.arch.lane_idx() % 32 == 0:\n"
            f"    output_pointer_{index}(source_row).store("
            f"cutlass.BFloat16({logical_name}))"
        ),
        logical_value=_expression(logical_name),
        bindings=DirectAffineReplayBindings(
            row=_expression("source_row"),
        ),
    )
    state = DirectAffineStateStoreReplay(
        statements=(),
        pointer=_expression(f"state_pointer_{index}(source_row, source_feature)"),
        slot=ast.Constant(value=index),
        slot_extent=_expression(f"checkpoint_extent_{index}"),
        bindings=DirectAffineReplayBindings(
            row=_expression("source_row"),
            feature=_expression("source_feature"),
        ),
    )
    return DirectAffineStepReplay(
        diagonal=_value_replay(f"diagonal_{index}", "feature"),
        prediction_vector=_value_replay(f"prediction_{index}", "feature"),
        row_input=_value_replay(f"row_input_{index}", "row"),
        update_scale=_value_replay(f"update_scale_{index}"),
        update_vector=_value_replay(f"update_{index}", "feature"),
        observation_vector=_value_replay(f"observation_{index}", "feature"),
        output_effect=output,
        state_effect=state,
        output_before_state=index % 2 == 0,
    )


def test_reduction_marker_outside_feature_materialization_is_rejected() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    marker = DirectAffineValueReplay(
        statements=_statements(
            "partial = source_feature * source_feature\n"
            "reduced = _helion_lane_reduce(partial, 'sum', "
            "cutlass.Float32(0), 32, 1, 0, '', 1, 1)"
        ),
        value=_expression("reduced"),
        bindings=DirectAffineReplayBindings(
            feature=_expression("source_feature"),
        ),
    )
    step = dataclasses.replace(_step_replay(0), row_input=marker)
    templates = DirectAffineResolvedTemplates(
        entry_state=_value_replay("entry", "row", "feature"),
        steps=(step,),
    )

    assert not replay_impl._template_reduction_placement_is_supported(
        templates,
        _coordinates(plan),
    )


def _detached_replay(
    step_count: int = 3,
    rows: int = 16,
) -> DirectAffineReplay:
    statement = ast.Pass()
    candidate = cast(
        "Any",
        SimpleNamespace(
            region=SimpleNamespace(
                step_count=step_count,
                row_extent=rows,
                feature_extent=128,
                storage_dtype=torch.bfloat16,
            )
        ),
    )
    return DirectAffineReplay(
        candidate=candidate,
        nodes=(),
        statement_nodes=(),
        source_statements=(statement,),
        source_span=(0, 0),
        fence_nodes=(),
        root_statement_ids=(id(statement),),
        root_statement_dumps=(ast.dump(statement, include_attributes=False),),
    )


def _proof(
    step_count: int,
    rows: int = 16,
    replay: DirectAffineReplay | None = None,
) -> DirectAffineReplayProof:
    return DirectAffineReplayProof(
        state_output_alias_free=True,
        state_feature_stride_one=True,
        state_vector_aligned=True,
        masks_preserved=True,
        row_tail_free=True,
        feature_tail_free=True,
        runtime_state_shape=(rows, 128),
        runtime_output_row_extents=(rows,) * step_count,
        source_replay=replay,
    )


def _resolved_templates(
    *,
    entry_state: DirectAffineValueReplay,
    steps: tuple[DirectAffineStepReplay, ...],
    async_entry_state: Any = None,
) -> DirectAffineResolvedTemplates:
    return DirectAffineResolvedTemplates(
        entry_state=entry_state,
        steps=steps,
        async_entry_state=async_entry_state,
    )


def _compose_with_templates(
    replay: DirectAffineReplay,
    plan: Any,
    coordinates: DirectAffineCoordinates,
    proof: DirectAffineReplayProof,
    templates: DirectAffineResolvedTemplates,
):
    with patch(
        "helion._compiler.cute.direct_affine_emission.resolve_direct_affine_templates",
        return_value=templates,
    ):
        return compose_direct_affine_replacement(
            replay,
            plan,
            coordinates,
            proof,
        )


@pytest.mark.parametrize(
    ("step_count", "mma"),
    [(3, DirectAffineMma.M16N8), (5, DirectAffineMma.M16N16)],
)
def test_composer_emits_generic_mma_paths(
    step_count: int, mma: DirectAffineMma
) -> None:
    plan = _plan(step_count, mma)
    replay = _detached_replay(step_count)
    coordinates = _coordinates(plan)
    source_before = _ast_bytes(replay.source_statements)
    templates = _resolved_templates(
        entry_state=_value_replay("entry", "row", "feature"),
        steps=tuple(_step_replay(index) for index in range(step_count)),
    )
    emission = _compose_with_templates(
        replay,
        plan,
        coordinates,
        _proof(step_count, replay=replay),
        templates,
    )

    assert emission is not None
    source = "\n".join(
        ast.unparse(statement) for statement in emission.replacement_statements
    )
    assert "precompute_affine_from_buffers_bf16" in source
    assert "consume_affine_steps" in source
    assert "checkpoint_affine_row_bf16" in source
    assert "store_packed_b16x8_if_valid" in source
    assert f"project_retain_affine_{mma.value}_bf16" in source
    assert f"2 * {step_count}" not in source
    compile(
        ast.Module(body=list(emission.replacement_statements), type_ignores=[]),
        "<direct-affine-replay>",
        "exec",
    )
    source_ids = {
        id(node)
        for statement in replay.source_statements
        for node in ast.walk(statement)
    }
    replacement_ids = {
        id(node)
        for statement in emission.replacement_statements
        for node in ast.walk(statement)
    }
    assert source_ids.isdisjoint(replacement_ids)
    assert _ast_bytes(replay.source_statements) == source_before


def test_composer_rejects_name_collision_from_retained_program_scope() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = dataclasses.replace(
        _detached_replay(),
        reserved_names=frozenset(("_helion_direct_affine_mma",)),
    )
    coordinates = _coordinates(plan)
    templates = _resolved_templates(
        entry_state=_value_replay("entry", "row", "feature"),
        steps=tuple(_step_replay(index) for index in range(3)),
    )

    assert (
        _compose_with_templates(
            replay,
            plan,
            coordinates,
            _proof(3, replay=replay),
            templates,
        )
        is None
    )


def _automatic_template_replay(step_count: int) -> DirectAffineReplay:
    graph = torch.fx.Graph()
    external = graph.placeholder("external")
    entry_load = graph.call_function(operator.neg, (external,))
    entry_state = graph.call_function(operator.neg, (entry_load,))
    producer_nodes: list[torch.fx.Node] = []
    owned_nodes: list[torch.fx.Node] = [entry_load, entry_state]
    dependency_slices: list[object] = []
    step_records: list[object] = []
    statement_by_node: dict[torch.fx.Node, ast.stmt] = {
        entry_load: _statements(
            "entry_loaded = (state_tensor.iterator + "
            "cutlass.Int32(source_row) * "
            "cutlass.Int32(state_tensor.layout.stride[0]) + "
            "cutlass.Int32(source_feature) * "
            "cutlass.Int32(state_tensor.layout.stride[1])).load() "
            "if slot_is_valid else cutlass.BFloat16(0)"
        )[0],
        entry_state: _statements("entry_state_value = cutlass.Float32(entry_loaded)")[
            0
        ],
    }
    result_by_node: dict[torch.fx.Node, ast.expr] = {
        entry_load: _expression("entry_loaded"),
        entry_state: _expression("entry_state_value"),
    }
    incoming = entry_state
    for index in range(step_count):
        semantic_nodes: list[torch.fx.Node] = []
        semantic_names = (
            "diagonal",
            "prediction",
            "row_input",
            "update_scale",
            "update",
            "observation",
        )
        for semantic_name in semantic_names:
            node = graph.call_function(operator.neg, (external,))
            semantic_nodes.append(node)
            producer_nodes.append(node)
            if semantic_name in {"diagonal", "prediction", "update", "observation"}:
                arguments = "source_feature"
            elif semantic_name == "row_input":
                arguments = "source_row"
            else:
                arguments = "external_scale"
            result_name = f"{semantic_name}_value_{index}"
            expression = (
                f"{arguments} + {index}"
                if semantic_name == "update_scale"
                else f"renamed_{semantic_name}_{index}({arguments})"
            )
            statement_by_node[node] = _statements(f"{result_name} = {expression}")[0]
            result_by_node[node] = _expression(result_name)
            dependency_slices.append(SimpleNamespace(value=node, nodes=(node,)))
        (
            diagonal,
            prediction,
            row_input,
            update_scale,
            update,
            observation_vector,
        ) = semantic_nodes
        state = graph.call_function(operator.add, (incoming, update))
        observation = graph.call_function(operator.neg, (state,))
        output_cast = graph.call_function(operator.neg, (observation,))
        output_effect = graph.call_function(operator.neg, (output_cast,))
        state_cast = graph.call_function(operator.neg, (state,))
        state_effect = graph.call_function(operator.add, (state_cast, update_scale))
        owned_nodes.extend(
            (state, observation, output_cast, output_effect, state_cast, state_effect)
        )
        result_by_node[state] = _expression(f"state_value_{index}")
        result_by_node[observation] = _expression(f"observation_result_{index}")
        result_by_node[output_cast] = _expression(f"output_cast_{index}")
        result_by_node[output_effect] = ast.Constant(value=None)
        result_by_node[state_cast] = _expression(f"state_cast_{index}")
        result_by_node[state_effect] = ast.Constant(value=None)
        statement_by_node[state] = _statements(
            f"state_value_{index} = recurrence_placeholder_{index}"
        )[0]
        statement_by_node[observation] = _statements(
            f"observation_result_{index} = observation_placeholder_{index}"
        )[0]
        statement_by_node[output_cast] = _statements(
            f"output_cast_{index} = cutlass.BFloat16(observation_result_{index})"
        )[0]
        statement_by_node[output_effect] = _statements(
            "if cute.arch.lane_idx() % 32 == 0:\n"
            f"    output_address_{index}(source_row).store(output_cast_{index})"
        )[0]
        statement_by_node[state_cast] = _statements(
            f"state_cast_{index} = cutlass.BFloat16(state_value_{index})"
        )[0]
        statement_by_node[state_effect] = _statements(
            f"if update_scale_value_{index} < checkpoint_extent_{index} "
            "and source_row < runtime_rows and source_feature < 128:\n"
            f"    state_address_{index}(update_scale_value_{index}, source_row, "
            f"source_feature).store(state_cast_{index})"
        )[0]
        step_records.append(
            SimpleNamespace(
                diagonal=diagonal,
                prediction_vector=prediction,
                row_input=row_input,
                update_scale=update_scale,
                update_vector=update,
                observation_vector=observation_vector,
                state=state,
                observation=observation,
                output_effect=output_effect,
                state_effect=state_effect,
                state_access=SimpleNamespace(
                    indices=(update_scale, external, external)
                ),
                owned_nodes=(
                    state,
                    observation,
                    output_cast,
                    output_effect,
                    state_cast,
                    state_effect,
                ),
            )
        )
        incoming = state
    graph.output(incoming)
    source_nodes = tuple(node for node in graph.nodes if node.op == "call_function")
    source_statements = tuple(statement_by_node[node] for node in source_nodes)
    node_replays = tuple(
        DirectAffineNodeReplay(
            node=node,
            statements=(statement_by_node[node],),
            result=result_by_node[node],
        )
        for node in source_nodes
    )
    region = SimpleNamespace(
        step_count=step_count,
        row_extent=16,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        entry_load=entry_load,
        entry_state=entry_state,
        producer_nodes=tuple(producer_nodes),
        owned_nodes=tuple(owned_nodes),
        dependency_slices=tuple(dependency_slices),
        steps=tuple(step_records),
        source_interval=source_nodes,
    )
    return DirectAffineReplay(
        candidate=cast("Any", SimpleNamespace(region=region)),
        nodes=node_replays,
        statement_nodes=source_nodes,
        source_statements=source_statements,
        source_span=(0, len(source_statements) - 1),
        fence_nodes=(),
        root_statement_ids=tuple(id(item) for item in source_statements),
        root_statement_dumps=tuple(
            ast.dump(item, include_attributes=True) for item in source_statements
        ),
    )


def test_templates_are_derived_without_application_specific_bindings() -> None:
    plan = _plan(
        3,
        DirectAffineMma.M16N8,
        ingress=DirectAffineStateIngress.ASYNC,
    )
    replay = _automatic_template_replay(3)
    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is not None
    assert "state_tensor.layout.stride[0]" in ast.unparse(
        templates.async_entry_state.row_stride
    )
    assert len(templates.steps) == 3
    assert "renamed_prediction_1" in ast.unparse(
        templates.steps[1].prediction_vector.statements[0]
    )
    assert isinstance(templates.steps[0].state_effect, DirectAffineStateStoreReplay)


def _replace_replayed_statement(
    replay: DirectAffineReplay,
    node: torch.fx.Node,
    source: str,
) -> DirectAffineReplay:
    replacement = _statements(source)
    assert replacement
    original = replay.node_replay(node)
    assert original is not None and len(original.statements) == 1
    old_statement = original.statements[0]
    nodes = tuple(
        dataclasses.replace(item, statements=replacement) if item.node is node else item
        for item in replay.nodes
    )
    source_statements = tuple(
        new_item
        for item in replay.source_statements
        for new_item in (replacement if item is old_statement else (item,))
    )
    return dataclasses.replace(
        replay,
        nodes=nodes,
        source_statements=source_statements,
    )


def test_packed_state_replay_preserves_non_slot_guard_conjuncts() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    coordinates = _coordinates(plan)

    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    state = templates.steps[0].state_effect
    assert isinstance(state, DirectAffineStateStoreReplay)
    valid = ast.unparse(state.valid)
    assert "source_row < runtime_rows" in valid
    assert "source_feature < 128" in valid
    assert "checkpoint_extent" not in valid
    validated = validate_direct_affine_templates(
        templates,
        replay,
        plan,
        coordinates,
    )
    assert validated is not None
    validated_state = validated.steps[0].state_effect
    assert isinstance(validated_state, DirectAffineStateStoreReplay)
    validated_guard = ast.unparse(validated_state.valid)
    assert "source_row < runtime_rows" in validated_guard
    assert "source_feature < 128" in validated_guard


def test_packed_state_replay_rejects_nonconjunctive_slot_guard() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "if update_scale_value_0 < checkpoint_extent_0 or semantic_guard:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is None


def test_validated_state_rejects_non_bound_feature_guard() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "if update_scale_value_0 < checkpoint_extent_0 "
        "and source_feature % 2 == 0:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_validated_state_rejects_aliased_nonuniform_feature_guard() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "feature_guard = source_feature % 2 == 0\n"
        "if update_scale_value_0 < checkpoint_extent_0 and feature_guard:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_validated_state_rejects_effectful_residual_guard() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "if update_scale_value_0 < checkpoint_extent_0 and side_effect():\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert isinstance(templates.steps[0].state_effect, DirectAffineStateStoreReplay)
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_validated_state_rejects_effectful_slot_extent() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "if update_scale_value_0 < side_effect():\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert isinstance(templates.steps[0].state_effect, DirectAffineStateStoreReplay)
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_packed_state_rejects_effectful_pointer_support() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "leading = side_effect()\n"
        "if update_scale_value_0 < checkpoint_extent_0:\n"
        "    (state_tensor.iterator + "
        "leading * state_tensor.layout.stride[0] + "
        "source_row * state_tensor.layout.stride[2] + "
        "source_feature * state_tensor.layout.stride[3]).store(state_cast_0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is None


def test_packed_state_rejects_control_flow_reaching_definition() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "feature_guard = True\n"
        "if semantic_guard:\n"
        "    feature_guard = source_feature % 2 == 0\n"
        "if update_scale_value_0 < checkpoint_extent_0 and feature_guard:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is None


def test_output_replay_rejects_effectful_residual_guard() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    output_effect = replay.candidate.region.steps[0].output_effect
    replay = _replace_replayed_statement(
        replay,
        output_effect,
        "if side_effect() and cute.arch.lane_idx() % 32 == 0:\n"
        "    output_address_0(source_row).store(output_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_validated_state_rejects_unproved_feature_bound() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "if update_scale_value_0 < checkpoint_extent_0 "
        "and source_feature < runtime_feature_extent:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(state_cast_0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
        )
        is None
    )


def test_validation_accepts_proven_size_guard_and_pre_wrap_output() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    for index, step in enumerate(replay.candidate.region.steps):
        replay = _replace_replayed_statement(
            replay,
            step.state_effect,
            f"if update_scale_value_{index} < checkpoint_extent_{index} "
            "and source_row < runtime_rows "
            "and cutlass.Int32(0 + ordinary_feature_serial * 32 + "
            "cute.arch.thread_idx()[0]) < checkpoint_pool_size_3:\n"
            f"    state_address_{index}(update_scale_value_{index}, source_row, "
            f"source_feature).store(state_cast_{index})",
        )
        replay = _replace_replayed_statement(
            replay,
            step.output_effect,
            f"output_address_{index}(source_row).store(output_cast_{index})",
        )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)
    proof = dataclasses.replace(
        _proof(3, replay=replay),
        state_feature_extent_expression=_expression("checkpoint_pool_size_3"),
        output_store_pre_wrap_proven=True,
    )

    assert templates is not None
    validated = validate_direct_affine_templates(
        templates,
        replay,
        plan,
        coordinates,
        proof,
    )
    assert validated is not None
    assert all(
        isinstance(step.state_effect, DirectAffineStateStoreReplay)
        and "ordinary_feature_serial" in ast.unparse(step.state_effect.valid)
        and "checkpoint_pool_size_3" in ast.unparse(step.state_effect.valid)
        for step in validated.steps
    )


@pytest.mark.parametrize(
    "state_source",
    [
        (
            "for index in range(1):\n"
            "    if update_scale_value_0 < checkpoint_extent_0:\n"
            "        state_address_0(update_scale_value_0, source_row, "
            "source_feature).store(state_cast_0)"
        ),
        (
            "side_effect()\n"
            "if update_scale_value_0 < checkpoint_extent_0:\n"
            "    state_address_0(update_scale_value_0, source_row, "
            "source_feature).store(state_cast_0)"
        ),
        (
            "unused = side_effect()\n"
            "if update_scale_value_0 < checkpoint_extent_0:\n"
            "    state_address_0(update_scale_value_0, source_row, "
            "source_feature).store(state_cast_0)"
        ),
    ],
)
def test_packed_state_replay_rejects_control_or_sibling_effects(
    state_source: str,
) -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(replay, state_effect, state_source)

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is None


def test_packed_state_replay_requires_store_value_from_logical_state() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    state_effect = replay.candidate.region.steps[0].state_effect
    replay = _replace_replayed_statement(
        replay,
        state_effect,
        "unused_state = state_value_0\n"
        "if update_scale_value_0 < checkpoint_extent_0:\n"
        "    state_address_0(update_scale_value_0, source_row, "
        "source_feature).store(unrelated_value)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is None


@pytest.mark.parametrize(
    "entry_source",
    [
        (
            "entry_loaded = (state_tensor.iterator + "
            "cutlass.Int32(source_row) * "
            "cutlass.Int32(state_tensor.layout.stride[0])).load() "
            "if slot_is_valid else cutlass.BFloat16(0)"
        ),
        (
            "entry_loaded = slot_is_valid and (state_tensor.iterator + "
            "cutlass.Int32(source_row) * "
            "cutlass.Int32(state_tensor.layout.stride[0]) + "
            "cutlass.Int32(source_feature) * "
            "cutlass.Int32(state_tensor.layout.stride[1])).load()"
        ),
        (
            "entry_loaded = (state_tensor.iterator + "
            "cutlass.Int32(source_row * 2) * "
            "cutlass.Int32(state_tensor.layout.stride[0]) + "
            "cutlass.Int32(source_feature) * "
            "cutlass.Int32(state_tensor.layout.stride[1])).load() "
            "if slot_is_valid else cutlass.BFloat16(0)"
        ),
    ],
)
def test_async_entry_rejects_incomplete_address_or_untracked_guard(
    entry_source: str,
) -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        entry_source,
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_entry_uses_exact_state_load_node_with_other_entry_loads() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_state,
        "unrelated = other_pointer.load()\n"
        "entry_state_value = cutlass.Float32(entry_loaded)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is not None
    assert "state_tensor.iterator" in ast.unparse(templates.async_entry_state.pointer)


def test_async_entry_rejects_sibling_effect_in_entry_state_node() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_state,
        "side_effect()\nentry_state_value = cutlass.Float32(entry_loaded)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_entry_rejects_effectful_pointer_support() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        "leading = side_effect()\n"
        "entry_loaded = (state_tensor.iterator + "
        "leading * cutlass.Int32(state_tensor.layout.stride[2]) + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if slot_is_valid else cutlass.BFloat16(0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_entry_rejects_sibling_effect_in_load_node() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        "side_effect()\n"
        "entry_loaded = (state_tensor.iterator + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if slot_is_valid else cutlass.BFloat16(0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_entry_rejects_discarded_impure_assignment() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        "unused = side_effect()\n"
        "entry_loaded = (state_tensor.iterator + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if slot_is_valid else cutlass.BFloat16(0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_validation_rejects_aliased_nonuniform_feature_guard() -> None:
    plan = _plan(
        3,
        DirectAffineMma.M16N8,
        ingress=DirectAffineStateIngress.ASYNC,
    )
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        "feature_guard = source_feature % 2 == 0\n"
        "entry_loaded = (state_tensor.iterator + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if feature_guard else cutlass.BFloat16(0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)

    assert templates is not None and templates.async_entry_state is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
            _proof(3, replay=replay),
        )
        is None
    )


@pytest.mark.parametrize(
    "prefix",
    [
        (
            "feature_guard = True\n"
            "if semantic_guard:\n"
            "    feature_guard = source_feature % 2 == 0\n"
        ),
        "feature_guard, = (True,)\n",
        "feature_guard: bool = True\n",
    ],
)
def test_async_entry_rejects_non_flat_guard_definition(prefix: str) -> None:
    plan = _plan(
        3,
        DirectAffineMma.M16N8,
        ingress=DirectAffineStateIngress.ASYNC,
    )
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        prefix + "entry_loaded = (state_tensor.iterator + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if feature_guard else cutlass.BFloat16(0)",
    )

    templates = resolve_direct_affine_templates(replay, _coordinates(plan))

    assert templates is not None
    assert templates.async_entry_state is None


def test_async_validation_accepts_distinct_local_and_global_row_bounds() -> None:
    plan = _plan(
        3,
        DirectAffineMma.M16N8,
        ingress=DirectAffineStateIngress.ASYNC,
    )
    replay = _automatic_template_replay(3)
    replay = _replace_replayed_statement(
        replay,
        replay.candidate.region.entry_load,
        "entry_loaded = (state_tensor.iterator + "
        "cutlass.Int32(source_row) * "
        "cutlass.Int32(state_tensor.layout.stride[0]) + "
        "cutlass.Int32(source_feature) * "
        "cutlass.Int32(state_tensor.layout.stride[1])).load() "
        "if row_tile_offset + cute.arch.thread_idx()[1] * 16 "
        "+ ordinary_row_serial < runtime_rows "
        "and cute.arch.thread_idx()[1] * 16 + ordinary_row_serial < 16 "
        "and source_feature < 128 else cutlass.BFloat16(0)",
    )
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)
    proof = dataclasses.replace(
        _proof(3, rows=128, replay=replay),
        state_row_extent_expression=_expression("runtime_rows"),
    )

    assert templates is not None and templates.async_entry_state is not None
    assert (
        validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
            proof,
        )
        is not None
    )


def test_composer_does_not_consume_mutated_resolved_templates() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _automatic_template_replay(3)
    coordinates = _coordinates(plan)
    templates = resolve_direct_affine_templates(replay, coordinates)
    assert templates is not None
    state = templates.steps[0].state_effect
    pointer_name = next(
        node for node in ast.walk(state.pointer) if isinstance(node, ast.Name)
    )
    pointer_name.id = "tampered_pointer"

    emission = compose_direct_affine_replacement(
        replay,
        plan,
        coordinates,
        _proof(3, replay=replay),
    )

    assert emission is not None
    assert "tampered_pointer" not in "\n".join(
        ast.unparse(statement) for statement in emission.replacement_statements
    )


def test_validator_rejects_raw_warp_coordinate() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _detached_replay()
    coordinates = _coordinates(plan)
    raw = DirectAffineResolvedTemplates(
        entry_state=DirectAffineValueReplay(
            statements=_statements("entry = cute.arch.warp_idx()"),
            value=_expression("entry"),
        ),
        steps=tuple(_step_replay(index) for index in range(3)),
    )
    with patch(
        "helion._compiler.cute.direct_affine_emission.resolve_direct_affine_templates",
        return_value=raw,
    ):
        assert (
            compose_direct_affine_replacement(
                replay,
                plan,
                coordinates,
                _proof(3, replay=replay),
            )
            is None
        )


def test_async_composer_stages_waits_and_replays_packed_state_effects() -> None:
    plan = _plan(
        3,
        DirectAffineMma.M16N8,
        ingress=DirectAffineStateIngress.ASYNC,
    )
    replay = _automatic_template_replay(3)
    coordinates = _coordinates(plan)
    proof = _proof(3, replay=replay)
    emission = compose_direct_affine_replacement(
        replay,
        plan,
        coordinates,
        proof,
    )

    assert emission is not None
    source = "\n".join(
        ast.unparse(statement) for statement in emission.replacement_statements
    )
    stage = source.index("stage_state_tile8x8_async_bf16")
    precompute = source.index("precompute_affine_from_buffers_bf16")
    wait = source.index("cute.arch.cp_async_wait_group(0)")
    barrier = source.index("cute.arch.sync_threads()", wait)
    project = source.index("project_retain_affine_m16n8_bf16")
    assert stage < precompute < wait < barrier < project
    assert "state_address_2(" in source
    assert "store_packed_b16x8_if_valid" in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_output_alias_free", False),
        ("state_feature_stride_one", False),
        ("state_vector_aligned", False),
        ("masks_preserved", False),
        ("row_tail_free", False),
        ("feature_tail_free", False),
        ("runtime_state_shape", (16, 64)),
        ("runtime_output_row_extents", (15, 16, 16)),
    ],
)
def test_composer_fails_closed_on_memory_contract(field: str, value: object) -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    replay = _detached_replay()
    coordinates = _coordinates(plan)
    root_body = [ast.Pass(), *replay.source_statements, ast.Pass()]
    before_root = _ast_bytes(root_body)
    before_source = _ast_bytes(replay.source_statements)
    proof = dataclasses.replace(_proof(3, replay=replay), **{field: value})
    templates = _resolved_templates(
        entry_state=_value_replay("entry", "row", "feature"),
        steps=tuple(_step_replay(index) for index in range(3)),
    )
    assert (
        _compose_with_templates(
            replay,
            plan,
            coordinates,
            proof,
            templates,
        )
        is None
    )
    assert _ast_bytes(root_body) == before_root
    assert _ast_bytes(replay.source_statements) == before_source


def test_replay_proof_accepts_divisible_full_runtime_row_extent() -> None:
    plan = _plan(3, DirectAffineMma.M16N8)
    proof = dataclasses.replace(
        _proof(3),
        runtime_state_shape=(128, 128),
        runtime_output_row_extents=(128, 128, 128),
    )

    assert proof.supports(plan)


def test_commit_revalidates_before_atomic_splice() -> None:
    candidate, graph_info, codegen, body, _ = _ownership_case()
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )
        assert replay is not None
        body[1].lineno = 99
        before = _ast_bytes(body)
        assert not replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            (ast.Pass(),),
        )
    assert _ast_bytes(body) == before


def test_commit_rejects_mutated_expression_only_codegen_result() -> None:
    candidate, graph_info, codegen, body, nodes = _ownership_case()
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )
        assert replay is not None
        result = codegen._results[id(nodes[2])]
        assert isinstance(result, ast.BinOp)
        result.op = ast.Sub()
        before = _ast_bytes(body)
        assert not replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            _statements("replacement = 1"),
        )
    assert _ast_bytes(body) == before


def test_commit_rejects_non_statement_and_aliased_replacements() -> None:
    candidate, graph_info, codegen, body, _ = _ownership_case()
    before = _ast_bytes(body)
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )
        assert replay is not None
        assert not replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            cast("Any", (_expression("value"),)),
        )
        assert not replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            (cast("ast.stmt", body[1]),),
        )
    assert _ast_bytes(body) == before


def test_commit_succeeds_and_rolls_back_an_injected_recorder_failure() -> None:
    candidate, graph_info, codegen, body, _ = _ownership_case()
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )
        assert replay is not None
        replacement = _statements("replacement = 1")
        assert replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            replacement,
        )
    assert ast.unparse(body[1]) == "replacement = 1"

    candidate, graph_info, codegen, body, _ = _ownership_case()
    before_body = _ast_bytes(body)
    before_owners = {
        owner: tuple(entries)
        for owner, entries in codegen._statements_by_owner_node_id.items()
    }
    with patch(
        "helion._compiler.cute.direct_affine_replay_core.revalidate_direct_affine_candidate",
        return_value=candidate,
    ):
        replay = resolve_direct_affine_replay(
            cast("Any", candidate), cast("Any", graph_info), codegen, body
        )
        assert replay is not None

        original_replace = codegen.replace_owned_statement_span

        def fail_after_mutation(*args: Any, **kwargs: Any) -> bool:
            assert original_replace(*args, **kwargs)
            codegen.referenced_thread_block_dims[0] = 99
            raise RuntimeError("injected recorder failure")

        codegen.replace_owned_statement_span = fail_after_mutation  # type: ignore[method-assign]
        assert not replace_direct_affine_replay(
            replay,
            cast("Any", graph_info),
            codegen,
            body,
            _statements("replacement = 2"),
        )
    assert _ast_bytes(body) == before_body
    assert codegen.referenced_thread_block_dims == [1, 1, 1]
    assert {
        owner: tuple(entries)
        for owner, entries in codegen._statements_by_owner_node_id.items()
    } == before_owners
