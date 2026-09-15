from __future__ import annotations

import ast
import operator
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute import direct_affine_lowering as lowering
from helion._compiler.cute import direct_affine_replay as replay_impl
from helion._compiler.cute.direct_affine_plan import DirectAffineCoefficientLayout
from helion._compiler.cute.direct_affine_plan import DirectAffineMma
from helion._compiler.cute.direct_affine_plan import DirectAffinePhaseOrder
from helion._compiler.cute.direct_affine_plan import DirectAffineStateAccessProof
from helion._compiler.cute.direct_affine_plan import DirectAffineStateIngress
from helion._compiler.cute.direct_affine_plan import resolve_direct_affine_plan
from helion._compiler.cute.direct_affine_replay import DirectAffineCoordinates
from helion._compiler.cute.direct_affine_replay import DirectAffineEffectReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineOrdinaryAxis
from helion._compiler.cute.direct_affine_replay import DirectAffineReplayBindings
from helion._compiler.cute.direct_affine_replay import DirectAffineResolvedTemplates
from helion._compiler.cute.direct_affine_replay import DirectAffineStateStoreReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineStepReplay
from helion._compiler.cute.direct_affine_replay import DirectAffineValueReplay
from helion._compiler.device_function import TensorArg


def _expression(source: str) -> ast.expr:
    result = ast.parse(source, mode="eval").body
    assert isinstance(result, ast.expr)
    return result


def _statements(source: str) -> tuple[ast.stmt, ...]:
    return tuple(ast.parse(source).body)


def _plan(
    *,
    rows: int = 64,
    ingress: DirectAffineStateIngress = DirectAffineStateIngress.SYNC,
):
    plan = resolve_direct_affine_plan(
        step_count=3,
        row_extent=rows,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=DirectAffineMma.M16N8,
        coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
        state_ingress=ingress,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
        state_access_proof=DirectAffineStateAccessProof(
            source_vector_alignment_bytes=16,
            destination_vector_alignment_bytes=128,
            source_feature_stride_one=True,
            full_vector_coverage=True,
            cp_async_supported=True,
        ),
    )
    assert plan is not None
    return plan


class _Strategy:
    def index_var(self, block_id: int) -> str:
        return {10: "ordinary_row", 20: "ordinary_feature"}[block_id]

    def offset_var(self, block_id: int) -> str:
        return {10: "row_tile", 20: "feature_tile"}[block_id]


class _RowStrategy:
    def index_var(self, block_id: int) -> str:
        return {10: "ordinary_row"}[block_id]

    def offset_var(self, block_id: int) -> str:
        return {10: "row_tile"}[block_id]


class _FeatureStrategy:
    def index_var(self, block_id: int) -> str:
        assert block_id == 20
        return "ordinary_feature"


class _BlockStrategies:
    def get_any(self, block_id: int) -> object:
        return _FeatureStrategy() if block_id == 20 else _RowStrategy()


class _CoordinateCodegen:
    def __init__(
        self,
        row: torch.fx.Node,
        feature: torch.fx.Node,
        *,
        synthetic_feature: bool = False,
    ) -> None:
        self.results = {
            row: _expression("_BLOCK_SIZE_10"),
            feature: _expression("ordinary_feature"),
        }
        self.cute_synthetic_arange_axis_sizes = {0: 32} if synthetic_feature else {}
        if synthetic_feature:
            self.device_function = SimpleNamespace(
                tile_strategy=SimpleNamespace(block_id_to_strategy=_BlockStrategies())
            )

    def codegen_result_for_node(self, node: torch.fx.Node) -> tuple[bool, object]:
        return (node in self.results), self.results.get(node)


def _coordinate_case() -> tuple[Any, Any, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    row = graph.placeholder("renamed_row")
    feature = graph.placeholder("renamed_feature")
    candidate = SimpleNamespace(
        region=SimpleNamespace(row_extent=64, feature_extent=128)
    )
    grid = SimpleNamespace(
        strategy=_Strategy(),
        hoist_parent_statements=[],
        outer_prefix=[],
        outer_suffix=[],
        lane_setup_statements=list(
            _statements(
                "ordinary_row = row_lane + "
                "cutlass.Int32(cute.arch.thread_idx()[1]) * 16 + row_tile\n"
                "ordinary_feature = cutlass.Int32(feature_lane) * 32 + "
                "cute.arch.thread_idx()[0]"
            )
        ),
        lane_loops=[("feature_lane", 4), ("row_lane", 16)],
        lane_loop_block_ids={
            "feature_lane": frozenset((20,)),
            "row_lane": frozenset((10,)),
        },
        block_thread_axes={20: 0, 10: 1},
        thread_axis_sizes={0: 32, 1: 4},
        vec_lane_wrappers={},
    )
    return candidate, grid, row, feature


def _axis_source_case(*, free_feature: bool = False) -> tuple[Any, Any, Any, Any]:
    graph = torch.fx.Graph()
    row = graph.placeholder("renamed_row")
    feature = graph.placeholder("renamed_feature")
    row.meta["tile_with_offset"] = {"block_id": 10, "offset": 0}
    if not free_feature:
        feature.meta["tile_with_offset"] = {"block_id": 20, "offset": 0}
    entry_load = graph.call_function(operator.neg, (row,))
    state_base = graph.placeholder("state")
    output_base = graph.placeholder("output")
    state_base.meta["val"] = torch.empty((4, 64, 128), dtype=torch.bfloat16)
    output_base.meta["val"] = torch.empty((4, 3, 64), dtype=torch.bfloat16)

    def access(base: torch.fx.Node, indices: tuple[object, ...]) -> Any:
        return SimpleNamespace(
            base=base,
            indices=indices,
            mask=None,
            other=None,
            kwargs=(),
        )

    entry_access = access(state_base, (0, row, feature))
    steps = tuple(
        SimpleNamespace(
            state_access=access(state_base, (index, row, feature)),
            output_access=access(output_base, (0, index, row)),
        )
        for index in range(3)
    )
    candidate = SimpleNamespace(
        region=SimpleNamespace(
            entry_access=entry_access,
            entry_load=entry_load,
            row_extent=64,
            feature_extent=128,
            steps=steps,
        )
    )
    device_function = SimpleNamespace(
        resolved_block_size=lambda block_id: {10: 64, 20: 128}[block_id]
    )
    if free_feature:
        device_function.tile_strategy = SimpleNamespace(
            block_id_to_strategy=_BlockStrategies()
        )
    codegen = SimpleNamespace(
        device_function=device_function,
        cute_synthetic_arange_axis_sizes={0: 32} if free_feature else {},
        codegen_result_for_node=lambda node: (
            True,
            _expression("ordinary_feature" if node is feature else "_BLOCK_SIZE_10"),
        ),
    )
    grid = SimpleNamespace(
        strategy=_RowStrategy() if free_feature else _Strategy(),
        block_thread_axes={20: 0, 10: 1},
    )
    env = SimpleNamespace(
        known_equal=operator.eq,
        resolve_block_id=lambda value: None,
        resolve_codegen_block_id=lambda block_id, codegen, graph: block_id,
        size_hint=int,
    )
    return candidate, codegen, grid, env


def test_axis_sources_are_derived_by_access_identity_not_names() -> None:
    candidate, codegen, grid, env = _axis_source_case()
    with patch.object(lowering.CompileEnvironment, "current", return_value=env):
        sources = lowering._resolve_axis_sources(
            cast("Any", candidate), cast("Any", codegen), cast("Any", grid)
        )

    assert sources is not None
    assert (sources.row_block_id, sources.feature_block_id) == (10, 20)
    assert not sources.feature_is_synthetic


def test_axis_sources_accept_exact_free_arange_codegen_witness() -> None:
    candidate, codegen, grid, env = _axis_source_case(free_feature=True)
    with patch.object(lowering.CompileEnvironment, "current", return_value=env):
        sources = lowering._resolve_axis_sources(
            cast("Any", candidate), cast("Any", codegen), cast("Any", grid)
        )

    assert sources is not None
    assert (sources.row_block_id, sources.feature_block_id) == (10, 20)
    assert sources.feature_is_synthetic


def test_axis_sources_reject_permuted_output_axis() -> None:
    candidate, codegen, grid, env = _axis_source_case()
    first = candidate.region.steps[0]
    feature = candidate.region.entry_access.indices[-1]
    first.output_access.indices = (*first.output_access.indices[:-1], feature)
    with patch.object(lowering.CompileEnvironment, "current", return_value=env):
        assert (
            lowering._resolve_axis_sources(
                cast("Any", candidate), cast("Any", codegen), cast("Any", grid)
            )
            is None
        )


@pytest.mark.parametrize("feature_required", [False, True])
def test_access_axes_reject_transitive_feature_in_leading_index(
    feature_required: bool,
) -> None:
    graph = torch.fx.Graph()
    row = graph.placeholder("row")
    feature = graph.placeholder("feature")
    derived_feature = graph.call_function(operator.add, (feature, 0))
    base = graph.placeholder("base")
    if feature_required:
        base.meta["val"] = torch.empty((2, 4, 16, 128), dtype=torch.bfloat16)
        indices = (0, derived_feature, row, feature)
    else:
        base.meta["val"] = torch.empty((4, 3, 16), dtype=torch.bfloat16)
        indices = (derived_feature, 0, row)
    access = SimpleNamespace(
        base=base,
        indices=indices,
        mask=None,
        other=None,
        kwargs=(),
    )

    assert not lowering._access_uses_axes(
        cast("Any", access),
        row,
        feature,
        feature_required=feature_required,
    )


def test_coordinate_adapter_accepts_renamed_permuted_complete_axes() -> None:
    candidate, grid, row, feature = _coordinate_case()
    codegen = _CoordinateCodegen(row, feature)
    result = lowering._resolve_coordinates(
        cast("Any", candidate),
        lowering._AxisSources(row, feature, 10, 20),
        _plan(),
        cast("Any", codegen),
        cast("Any", grid),
        (),
    )

    assert result is not None
    coordinates, source_cta = result
    assert source_cta == (32, 4, 1)
    assert ast.unparse(coordinates.row.source) == "ordinary_row"
    assert ast.unparse(coordinates.row.tile_offset) == "row_tile"
    assert "row_tile" not in ast.unparse(coordinates.row.local_expression)
    assert coordinates.feature.lane_name == "feature_lane"


def test_coordinate_adapter_accepts_free_arange_synthetic_feature() -> None:
    candidate, grid, row, feature = _coordinate_case()
    grid.strategy = _RowStrategy()
    grid.lane_setup_statements = [
        statement_from_string(ast.unparse(statement))
        for statement in grid.lane_setup_statements
    ]
    assert all(
        isinstance(statement, ExtendedAST) for statement in grid.lane_setup_statements
    )
    result = lowering._resolve_coordinates(
        cast("Any", candidate),
        lowering._AxisSources(row, feature, 10, 20, True),
        _plan(),
        cast("Any", _CoordinateCodegen(row, feature, synthetic_feature=True)),
        cast("Any", grid),
        (),
    )

    assert result is not None
    coordinates, source_cta = result
    assert source_cta == (32, 4, 1)
    assert ast.unparse(coordinates.feature.source) == "ordinary_feature"
    assert ast.unparse(coordinates.feature.tile_offset) == "0"


def test_coordinate_adapter_rejects_ambiguous_assignment() -> None:
    candidate, grid, row, feature = _coordinate_case()
    grid.lane_setup_statements.append(_statements("ordinary_row = row_lane")[0])

    assert (
        lowering._resolve_coordinates(
            cast("Any", candidate),
            lowering._AxisSources(row, feature, 10, 20),
            _plan(),
            cast("Any", _CoordinateCodegen(row, feature)),
            cast("Any", grid),
            (),
        )
        is None
    )


def _coordinates() -> DirectAffineCoordinates:
    return DirectAffineCoordinates(
        row=DirectAffineOrdinaryAxis(
            source=_expression("ordinary_row"),
            tile_offset=_expression("row_tile"),
            local_expression=_expression("cute.arch.thread_idx()[1] * 16 + row_lane"),
            extent=64,
            thread_axis=1,
            thread_extent=4,
            lane_name="row_lane",
            lane_extent=16,
        ),
        feature=DirectAffineOrdinaryAxis(
            source=_expression("ordinary_feature"),
            tile_offset=ast.Constant(value=0),
            local_expression=_expression(
                "cute.arch.thread_idx()[0] + feature_lane * 32"
            ),
            extent=128,
            thread_axis=0,
            thread_extent=32,
            lane_name="feature_lane",
            lane_extent=4,
        ),
    )


def _value(label: str) -> DirectAffineValueReplay:
    return DirectAffineValueReplay(
        statements=_statements(f"{label} = opaque(ordinary_feature)"),
        value=_expression(label),
        bindings=DirectAffineReplayBindings(feature=_expression("ordinary_feature")),
    )


def _templates(
    output_source: str,
) -> DirectAffineResolvedTemplates:
    value = _value("value")
    output = DirectAffineEffectReplay(
        statements=_statements(output_source),
        logical_value=_expression("logical_value"),
        bindings=DirectAffineReplayBindings(row=_expression("ordinary_row")),
    )
    state = DirectAffineStateStoreReplay(
        statements=(),
        pointer=_expression("state.iterator + ordinary_row + ordinary_feature"),
        slot=_expression("slot"),
        slot_extent=_expression("slot_extent"),
        bindings=DirectAffineReplayBindings(
            row=_expression("ordinary_row"),
            feature=_expression("ordinary_feature"),
        ),
    )
    step = DirectAffineStepReplay(
        diagonal=value,
        prediction_vector=value,
        row_input=value,
        update_scale=value,
        update_vector=value,
        observation_vector=value,
        output_effect=output,
        state_effect=state,
    )
    return DirectAffineResolvedTemplates(
        entry_state=value,
        steps=(step, step, step),
    )


def test_output_ownership_is_discharged_but_semantic_guard_is_preserved() -> None:
    replay = SimpleNamespace(nodes=())
    coordinates = _coordinates()
    templates = _templates(
        "if semantic_guard and "
        "cutlass.Int32(cute.arch.lane_idx()) % 32 == 0:\n"
        "    output(ordinary_row).store(logical_value)",
    )

    normalized = replay_impl._validate_direct_affine_templates(
        templates,
        cast("Any", replay),
        _plan(),
        coordinates,
    )

    assert normalized is not None
    source = ast.unparse(normalized.steps[0].output_effect.statements[0])
    assert source == "if semantic_guard:\n    output(ordinary_row).store(logical_value)"
    assert "lane_idx" not in source
    assert "lane_idx" in ast.unparse(templates.steps[0].output_effect.statements[0])


@pytest.mark.parametrize(
    "output_source",
    [
        "output(ordinary_row).store(logical_value)",
        (
            "if cute.arch.lane_idx() % 32 == 0:\n"
            "    scratch = logical_value\n"
            "output(ordinary_row).store(scratch)"
        ),
        (
            "if cute.arch.lane_idx() % 32 == 0:\n"
            "    output(ordinary_row).store(logical_value)\n"
            "else:\n"
            "    output(ordinary_row).store(logical_value)"
        ),
        (
            "unused = logical_value\n"
            "if cute.arch.lane_idx() % 32 == 0:\n"
            "    side_effect()\n"
            "    output(ordinary_row).store(other)"
        ),
        ("if cute.arch.lane_idx() % 32 == 0:\n    output(ordinary_row).store(other)"),
        (
            "if cute.arch.lane_idx() % 32 == 0:\n"
            "    output(fixed_row).store(logical_value)"
        ),
    ],
)
def test_output_ownership_rejects_missing_or_non_dominating_guard(
    output_source: str,
) -> None:
    replay = SimpleNamespace(nodes=())
    coordinates = _coordinates()
    templates = _templates(output_source)
    assert (
        replay_impl._validate_direct_affine_templates(
            templates,
            cast("Any", replay),
            _plan(),
            coordinates,
        )
        is None
    )


def test_prefix_proof_accepts_uniform_loads_and_rejects_lane_effects() -> None:
    grid = SimpleNamespace(
        lane_loops=[("row_lane", 16)],
        lane_setup_statements=list(_statements("ordinary_row = row_lane")),
    )
    assert lowering._prefix_is_lane_independent(
        _statements("slot = pointer.load() if valid else 0"), cast("Any", grid)
    )
    assert not lowering._prefix_is_lane_independent(
        _statements("slot = cute.arch.thread_idx()[1]"), cast("Any", grid)
    )
    assert not lowering._prefix_is_lane_independent(
        _statements("pointer.store(value)"), cast("Any", grid)
    )
    assert not lowering._prefix_is_lane_independent(
        _statements("slot = arbitrary_call()"), cast("Any", grid)
    )


def _memory_case(
    *,
    state: torch.Tensor | None = None,
    output_rows: int = 128,
    slot: object = 0,
) -> tuple[Any, Any, Any]:
    graph = torch.fx.Graph()
    state_base = graph.placeholder("state_base")
    input_base = graph.placeholder("input_base")
    output_base = graph.placeholder("output_base")
    state = (
        state if state is not None else torch.empty((4, 128, 128), dtype=torch.bfloat16)
    )
    input_tensor = torch.empty((3, 128), dtype=torch.bfloat16)
    output = torch.empty((4, 3, output_rows), dtype=torch.bfloat16)
    for node, value in (
        (state_base, state),
        (input_base, input_tensor),
        (output_base, output),
    ):
        node.meta["val"] = value
    entry_access = SimpleNamespace(indices=(slot, object(), object()))
    steps = tuple(
        SimpleNamespace(
            state_access=SimpleNamespace(indices=(slot, object(), object())),
            output_base=output_base,
        )
        for _ in range(3)
    )
    region = SimpleNamespace(
        read_bases=(state_base, input_base),
        write_bases=(state_base, output_base),
        state_base=state_base,
        output_bases=(output_base,) * 3,
        storage_dtype=torch.bfloat16,
        row_extent=64,
        feature_extent=128,
        step_count=3,
        entry_access=entry_access,
        steps=steps,
    )
    tensors = {
        state_base: state,
        input_base: input_tensor,
        output_base: output,
    }
    arguments = [
        TensorArg(node.name, tensor, node.name) for node, tensor in tensors.items()
    ]
    strides = {
        (node.name, dim): stride
        for node, tensor in tensors.items()
        for dim, stride in enumerate(tensor.stride())
    }
    device_function = SimpleNamespace(
        arguments=arguments,
        proven_tensor_size_values=dict,
        proven_tensor_stride_values=lambda: strides,
    )
    codegen = SimpleNamespace(device_function=device_function)
    env = SimpleNamespace(
        compiler_fact_specialization_facts=frozenset(("input_tensor_metadata",)),
        runtime_value_for_tensor=lambda tensor: tensor,
        tensor_input_source=lambda tensor: object(),
        config_spec=SimpleNamespace(target_device_capability=(9, 0)),
    )
    return SimpleNamespace(region=region), codegen, env


def test_memory_proof_covers_every_pair_involving_a_writer() -> None:
    candidate, codegen, env = _memory_case()
    with (
        patch.object(lowering.CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensor_has_specialized_alignment",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
            return_value=True,
        ) as disjoint,
    ):
        proof = lowering._resolve_memory_proof(
            cast("Any", candidate),
            cast("Any", codegen),
            DirectAffineStateIngress.ASYNC,
        )

    assert proof is not None
    assert disjoint.call_count == 3
    assert proof[0].supports_async_io()
    assert proof[1].runtime_state_shape == (128, 128)
    assert proof[1].runtime_output_row_extents == (128, 128, 128)


@pytest.mark.parametrize(
    ("failure", "state", "output_rows", "slot"),
    [
        (
            "state-leading-stride",
            torch.empty_strided(
                (4, 64, 128),
                (64 * 128 + 1, 128, 1),
                dtype=torch.bfloat16,
            ),
            64,
            0,
        ),
        (
            "feature-stride",
            torch.empty_strided(
                (4, 64, 128),
                (64 * 256, 256, 2),
                dtype=torch.bfloat16,
            ),
            64,
            0,
        ),
        ("output-tail", None, 63, 0),
        ("slot-overflow", None, 128, 1 << 31),
    ],
)
def test_memory_proof_rejects_stride_tail_and_slot_failures(
    failure: str,
    state: torch.Tensor | None,
    output_rows: int,
    slot: object,
) -> None:
    candidate, codegen, env = _memory_case(
        state=state,
        output_rows=output_rows,
        slot=slot,
    )
    with (
        patch.object(lowering.CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensor_has_specialized_alignment",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
            return_value=True,
        ),
    ):
        assert (
            lowering._resolve_memory_proof(
                cast("Any", candidate),
                cast("Any", codegen),
                DirectAffineStateIngress.SYNC,
            )
            is None
        ), failure


def test_memory_proof_rejects_unproved_alias() -> None:
    candidate, codegen, env = _memory_case()
    with (
        patch.object(lowering.CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensor_has_specialized_alignment",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
            side_effect=(True, False, True),
        ),
    ):
        assert (
            lowering._resolve_memory_proof(
                cast("Any", candidate),
                cast("Any", codegen),
                DirectAffineStateIngress.SYNC,
            )
            is None
        )


def _real_codegen_tensor_metadata(
    base: torch.fx.Node,
    codegen: Any,
    proven_sizes: dict[tuple[str, int], tuple[str, int]],
    proven_strides: dict[tuple[str, int], int],
    *,
    require_layout: bool,
) -> lowering._TensorMetadata | None:
    fake = base.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    argument_name = lowering._tensor_argument_name(codegen.device_function, fake)
    if argument_name is None:
        return None
    return lowering._TensorMetadata(
        fake=fake,
        runtime=fake,
        argument_name=argument_name,
        shape=tuple(int(value) for value in fake.shape),
        strides=tuple(int(value) for value in fake.stride()),
    )


def test_fixed_rank1_t3_real_codegen_reaches_direct_async_emission() -> None:
    from test.test_cute_affine_scan_heuristic import _fake_cute_context
    from test.test_cute_fixed_token_rank1_recurrence import _fixed_rank1
    from test.test_cute_fixed_token_rank1_recurrence import _inputs

    with (
        _fake_cute_context(),
        patch.object(
            lowering,
            "_tensor_metadata",
            new=_real_codegen_tensor_metadata,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensor_has_specialized_alignment",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
            return_value=True,
        ),
    ):
        bound = _fixed_rank1._bind_isolated(
            (*_inputs(3), 3, 1.0e-6, True, False, False, False)
        )
        config = next(
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_affine_scan_schedule") == "direct_m16n8_v1"
            and seed.config.get("block_sizes") == [64]
        )
        source = bound.to_triton_code(config)

    assert (
        "import helion._compiler.cute.short_affine_scan_mma "
        "as _helion_direct_affine_mma" in source
    )
    issue = source.index("stage_state_tile8x8_async_bf16")
    compute = source.index("precompute_affine_from_buffers_bf16")
    wait = source.index("cute.arch.cp_async_wait_group(0)")
    assert issue < compute < wait
    assert source.count("cute.arch.warp_reduction_sum") == 9
    assert "_helion_lane_reduce" not in source
    assert "cute.arch.lane_idx" not in source
    assert "store_packed_b16x8_if_valid" in source
    assert "block=(32, 4, 1)" in source
    assert "indices_2" not in source
    assert "indices_3" not in source


def _runtime_inputs(token_count: int) -> tuple[object, ...]:
    torch.manual_seed(token_count)
    sequences = 1
    heads = 1
    width = 128
    vectors = sequences * token_count
    shape = (vectors, heads, width)
    vector_inputs = tuple(
        (torch.randn(shape, device="cuda") * 0.1).to(torch.bfloat16) for _ in range(4)
    )
    checkpoint_pool = (
        torch.randn(
            (vectors + 2, heads, width, width),
            device="cuda",
        )
        * 0.01
    ).to(torch.bfloat16)
    checkpoint_ids = torch.arange(
        vectors,
        device="cuda",
        dtype=torch.int32,
    ).reshape(sequences, token_count)
    return (
        *vector_inputs,
        (torch.randn((vectors, heads), device="cuda") * 0.1).to(torch.bfloat16),
        torch.randn((heads,), device="cuda") * 0.1,
        torch.randn((heads * width,), device="cuda") * 0.1,
        checkpoint_pool,
        checkpoint_ids,
        torch.full((sequences,), token_count, device="cuda", dtype=torch.int32),
        torch.empty(shape, device="cuda", dtype=torch.bfloat16),
        width**-0.5,
        -5.0,
        token_count,
        1.0e-6,
        True,
        False,
        False,
        False,
    )


def _reference_fixed_rank1(args: tuple[object, ...]) -> torch.Tensor:
    (
        first_vector,
        second_vector,
        values,
        gate_source,
        update_weight,
        log_decay_rate,
        gate_bias,
        checkpoint_pool,
        checkpoint_ids,
        accepted_counts,
        result,
        projection_scale,
        decay_floor,
        token_count,
        epsilon,
        _subtract_projection,
        _beta_is_logit,
        _reverse_checkpoints,
        _shrink_sequence_axis,
    ) = args
    tensors = (
        first_vector,
        second_vector,
        values,
        gate_source,
        update_weight,
        log_decay_rate,
        gate_bias,
        checkpoint_pool,
        checkpoint_ids,
        accepted_counts,
        result,
    )
    assert all(isinstance(value, torch.Tensor) for value in tensors)
    first_vector = cast("torch.Tensor", first_vector)
    second_vector = cast("torch.Tensor", second_vector)
    values = cast("torch.Tensor", values)
    gate_source = cast("torch.Tensor", gate_source)
    update_weight = cast("torch.Tensor", update_weight)
    log_decay_rate = cast("torch.Tensor", log_decay_rate)
    gate_bias = cast("torch.Tensor", gate_bias)
    checkpoint_pool = cast("torch.Tensor", checkpoint_pool)
    checkpoint_ids = cast("torch.Tensor", checkpoint_ids)
    accepted_counts = cast("torch.Tensor", accepted_counts)
    result = cast("torch.Tensor", result)
    token_count = cast("int", token_count)
    for sequence in range(checkpoint_ids.shape[0]):
        for head in range(first_vector.shape[1]):
            accepted = torch.clamp(
                accepted_counts[sequence].long() - 1,
                min=0,
                max=token_count - 1,
            )
            initial_slot = checkpoint_ids[sequence, accepted].long()
            recurrent = checkpoint_pool[initial_slot, head].float().clone()
            for token_index in range(token_count):
                token = sequence * token_count + token_index
                query = first_vector[token, head].float()
                key = second_vector[token, head].float()
                query *= torch.rsqrt((query * query).sum() + cast("float", epsilon))
                key *= torch.rsqrt((key * key).sum() + cast("float", epsilon))
                gate = gate_source[token, head].float()
                width = first_vector.shape[-1]
                gate += gate_bias[head * width : (head + 1) * width]
                decay_parameter = torch.exp2(
                    log_decay_rate[head].float() * 1.4426950408889634
                )
                decay = torch.exp2(
                    cast("float", decay_floor)
                    * torch.sigmoid(decay_parameter * gate)
                    * 1.4426950408889634
                )
                recurrent *= decay[None, :]
                prediction = (recurrent * key[None, :]).sum(-1)
                residual = values[token, head].float() - prediction
                recurrent += (update_weight[token, head].float() * residual)[
                    :, None
                ] * key[None, :]
                result[token, head] = (
                    (recurrent * query[None, :]).sum(-1)
                    * cast("float", projection_scale)
                ).to(result.dtype)
                checkpoint_slot = checkpoint_ids[sequence, token_index].long()
                checkpoint_pool[checkpoint_slot, head] = recurrent.to(
                    checkpoint_pool.dtype
                )
    return result


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="direct affine MMA requires a Blackwell CUDA device",
)
@pytest.mark.parametrize(
    ("token_count", "schedule", "row_tile"),
    [
        (3, "direct_m16n8_v1", 64),
        (5, "direct_m16n16_v1", 128),
    ],
)
def test_fixed_rank1_direct_runtime_matches_reference(
    token_count: int,
    schedule: str,
    row_tile: int,
) -> None:
    from test.test_cute_fixed_token_rank1_recurrence import _fixed_rank1

    args = _runtime_inputs(token_count)
    reference_args = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value for value in args
    )
    expected = _reference_fixed_rank1(reference_args)
    bound = _fixed_rank1._bind_isolated(args)
    config = next(
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_affine_scan_schedule") == schedule
        and seed.config.get("block_sizes") == [row_tile]
    )
    bound.config_spec.normalize(config)
    compiled = bound.compile_config(config, allow_print=False)
    actual = compiled(*args)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.08, atol=5.0e-4)
    torch.testing.assert_close(
        cast("torch.Tensor", args[7]),
        cast("torch.Tensor", reference_args[7]),
        rtol=0.08,
        atol=5.0e-4,
    )
