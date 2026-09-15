from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING

import pytest
import torch

from helion._compiler.cute.direct_affine_candidate import (
    discover_direct_affine_candidates,
)
from helion._compiler.cute.direct_affine_candidate import (
    revalidate_direct_affine_candidate,
)
from helion._compiler.cute.direct_affine_plan import DirectAffineMma
from helion._compiler.cute.short_affine_scan import ShortAffineScanAccessKind
from helion._compiler.cute.short_affine_scan import discover_short_affine_scan_regions
from helion._compiler.device_ir import RootGraphInfo
from helion.language import memory_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from collections.abc import Callable


ROWS = 16
FEATURES = 128


@dataclasses.dataclass
class _BuiltScan:
    info: RootGraphInfo
    state_base: torch.fx.Node
    output_bases: tuple[torch.fx.Node, ...]
    entry_load: torch.fx.Node
    entry_state: torch.fx.Node
    steps: tuple[dict[str, torch.fx.Node], ...]
    live_ins: tuple[torch.fx.Node, ...]
    producer_nodes: tuple[torch.fx.Node, ...]
    interior: torch.fx.Node
    preceding_access: torch.fx.Node
    following_access: torch.fx.Node


def _placeholder(
    graph: torch.fx.Graph,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node


def _call(
    graph: torch.fx.Graph,
    target: Callable[..., object],
    args: tuple[object, ...],
    shape: tuple[int, ...] | None,
    dtype: torch.dtype = torch.float32,
    *,
    name: str | None = None,
) -> torch.fx.Node:
    # pyrefly: ignore [bad-argument-type]
    node = graph.call_function(target, args, name=name)
    node.meta["val"] = None if shape is None else torch.empty(shape, dtype=dtype)
    return node


def _build_scan(
    steps: int,
    *,
    observation_forms: tuple[str, ...] | None = None,
    storage_dtype: torch.dtype = torch.bfloat16,
    features: int = FEATURES,
    opaque_producers: bool = False,
    mask_access: str | None = None,
    entry_other: str | None = None,
    extra_effect: bool = False,
    live_out: bool = False,
    producer_live_out: bool = False,
) -> _BuiltScan:
    if observation_forms is None:
        observation_forms = ("direct",) * steps
    assert len(observation_forms) == steps

    graph = torch.fx.Graph()
    state_base = _placeholder(graph, "state_buffer", (ROWS, features), storage_dtype)
    output_bases = tuple(
        _placeholder(graph, f"output_buffer_{index}", (ROWS,), storage_dtype)
        for index in range(steps)
    )
    raw_inputs = tuple(
        tuple(
            _placeholder(
                graph,
                f"opaque_source_{step}_{slot}",
                (ROWS,) if slot == 2 else (features,),
                torch.float32,
            )
            for slot in range(5)
        )
        for step in range(steps)
    )
    access_mask = _placeholder(graph, "access_mask", (ROWS, features), torch.bool)
    extra_base = _placeholder(graph, "extra_buffer", (ROWS,), storage_dtype)

    preceding_access = _call(
        graph,
        memory_ops.load,
        (state_base, [slice(None), slice(None)], None, None),
        (ROWS, features),
        storage_dtype,
        name="before_candidate",
    )
    entry_load = _call(
        graph,
        memory_ops.load,
        (
            state_base,
            [slice(None), slice(None)],
            access_mask if mask_access == "entry" else None,
            entry_other,
        ),
        (ROWS, features),
        storage_dtype,
        name="scan_seed_load",
    )
    entry_state = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (entry_load, torch.float32),
        (ROWS, features),
        name="scan_seed_fp32",
    )
    interior = _call(
        graph,
        torch.ops.aten.neg.default,
        (raw_inputs[0][0],),
        (features,),
        name="unrelated_pure_work",
    )

    producer_nodes: list[torch.fx.Node] = []

    def coefficient(raw: torch.fx.Node, step: int, slot: int) -> torch.fx.Node:
        if not opaque_producers:
            return raw
        # Names, operators, and operand order are intentionally unrelated to
        # the recurrence roles. Discovery treats this pure producer DAG as opaque.
        args = (2.0, raw) if (step + slot) % 2 else (raw, 2.0)
        result = _call(
            graph,
            operator.mul,
            args,
            tuple(raw.meta["val"].shape),
            name=f"cipher_{4 - slot}_{steps - step}",
        )
        producer_nodes.append(result)
        return result

    incoming = entry_state
    built_steps: list[dict[str, torch.fx.Node]] = []
    live_ins: list[torch.fx.Node] = [state_base, *output_bases]
    for index, form in enumerate(observation_forms):
        diagonal, prediction_vector, row_input, update_vector, observation_vector = (
            coefficient(raw, index, slot) for slot, raw in enumerate(raw_inputs[index])
        )
        live_ins.extend(
            (
                diagonal,
                prediction_vector,
                row_input,
                update_vector,
                observation_vector,
            )
        )

        diagonal_view = _call(
            graph,
            view_ops.subscript,
            (diagonal, [None, slice(None)]),
            (1, features),
        )
        decayed = _call(
            graph,
            operator.mul,
            (diagonal_view, incoming) if index % 2 else (incoming, diagonal_view),
            (ROWS, features),
        )
        prediction_view = _call(
            graph,
            view_ops.subscript,
            (prediction_vector, [None, slice(None)]),
            (1, features),
        )
        prediction_product = _call(
            graph,
            operator.mul,
            (prediction_view, decayed) if index % 2 else (decayed, prediction_view),
            (ROWS, features),
        )
        prediction = _call(
            graph,
            torch.ops.aten.sum.dim_IntList,
            (prediction_product, [-1]),
            (ROWS,),
        )
        residual = _call(
            graph,
            operator.sub,
            (row_input, prediction),
            (ROWS,),
        )
        row_update = _call(
            graph,
            operator.mul,
            (0.5, residual) if index % 2 else (residual, 0.5),
            (ROWS,),
        )
        row_view = _call(
            graph,
            view_ops.subscript,
            (row_update, [slice(None), None]),
            (ROWS, 1),
        )
        update_view = _call(
            graph,
            view_ops.subscript,
            (update_vector, [None, slice(None)]),
            (1, features),
        )
        outer = _call(
            graph,
            operator.mul,
            (update_view, row_view) if index % 2 else (row_view, update_view),
            (ROWS, features),
        )
        state = _call(
            graph,
            operator.add,
            (outer, decayed) if index % 2 else (decayed, outer),
            (ROWS, features),
        )
        observation_view = _call(
            graph,
            view_ops.subscript,
            (observation_vector, [None, slice(None)]),
            (1, features),
        )
        if form == "direct":
            observation_product = _call(
                graph,
                operator.mul,
                (observation_view, state) if index % 2 else (state, observation_view),
                (ROWS, features),
            )
            observation = _call(
                graph,
                torch.ops.aten.sum.dim_IntList,
                (observation_product, [-1]),
                (ROWS,),
            )
        else:
            assert form == "factored"
            base_product = _call(
                graph,
                operator.mul,
                (decayed, observation_view),
                (ROWS, features),
            )
            base_sum = _call(
                graph,
                torch.ops.aten.sum.dim_IntList,
                (base_product, [-1]),
                (ROWS,),
            )
            dot_product = _call(
                graph,
                operator.mul,
                (observation_view, update_view),
                (1, features),
            )
            expanded_dot = _call(
                graph,
                torch.ops.aten.expand.default,
                (dot_product, [ROWS, features]),
                (ROWS, features),
            )
            dot_sum = _call(
                graph,
                torch.ops.aten.sum.dim_IntList,
                (expanded_dot, [-1]),
                (ROWS,),
            )
            dot_view = _call(
                graph,
                view_ops.subscript,
                (dot_sum, [slice(None), None]),
                (ROWS, 1),
            )
            scaled_product = _call(
                graph,
                operator.mul,
                (dot_view, row_view),
                (ROWS, 1),
            )
            scaled_sum = _call(
                graph,
                torch.ops.aten.sum.dim_IntList,
                (scaled_product, [-1]),
                (ROWS,),
            )
            observation = _call(
                graph,
                operator.add,
                (scaled_sum, base_sum),
                (ROWS,),
            )
        stored_observation = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (observation, storage_dtype),
            (ROWS,),
            storage_dtype,
        )
        output_store = _call(
            graph,
            memory_ops.store,
            (
                output_bases[index],
                [slice(None)],
                stored_observation,
                access_mask if mask_access == "output" else None,
            ),
            None,
        )
        if extra_effect and index == 0:
            _call(
                graph,
                memory_ops.store,
                (extra_base, [slice(None)], row_input, None),
                None,
                name="unclaimed_effect",
            )
        stored_state = _call(
            graph,
            torch.ops.prims.convert_element_type.default,
            (state, storage_dtype),
            (ROWS, features),
            storage_dtype,
        )
        state_store = _call(
            graph,
            memory_ops.store,
            (
                state_base,
                [slice(None), slice(None)],
                stored_state,
                access_mask if mask_access == "state" else None,
            ),
            None,
        )
        built_steps.append(
            {
                "diagonal": diagonal,
                "prediction_vector": prediction_vector,
                "row_input": row_input,
                "update_vector": update_vector,
                "observation_vector": observation_vector,
                "decayed": decayed,
                "state": state,
                "observation": observation,
                "output_store": output_store,
                "state_store": state_store,
            }
        )
        incoming = state

    if live_out:
        _call(
            graph,
            torch.ops.aten.neg.default,
            (incoming,),
            (ROWS, features),
            name="outside_user",
        )
    if producer_live_out:
        assert producer_nodes
        _call(
            graph,
            torch.ops.aten.neg.default,
            (producer_nodes[0],),
            (features,),
            name="outside_producer_user",
        )
    following_access = _call(
        graph,
        memory_ops.load,
        (state_base, [slice(None), slice(None)], None, None),
        (ROWS, features),
        storage_dtype,
        name="after_candidate",
    )
    graph.output(None)
    graph.lint()
    return _BuiltScan(
        info=RootGraphInfo(graph_id=17, graph=graph),
        state_base=state_base,
        output_bases=output_bases,
        entry_load=entry_load,
        entry_state=entry_state,
        steps=tuple(built_steps),
        live_ins=tuple(live_ins),
        producer_nodes=tuple(producer_nodes),
        interior=interior,
        preceding_access=preceding_access,
        following_access=following_access,
    )


def _graph_snapshot(graph: torch.fx.Graph) -> tuple[object, ...]:
    return tuple(
        (
            node,
            node.op,
            node.target,
            node.args,
            node.kwargs,
            tuple(node.users),
            tuple(node.meta),
            tuple(id(value) for value in node.meta.values()),
        )
        for node in graph.nodes
    )


@pytest.mark.parametrize("step_count", range(2, 9))
def test_discovers_t2_through_t8_without_mutating_graph(step_count: int) -> None:
    built = _build_scan(step_count)
    before = _graph_snapshot(built.info.graph)

    (region,) = discover_short_affine_scan_regions((built.info,))

    assert _graph_snapshot(built.info.graph) == before
    assert region.graph_id == 17
    assert region.step_count == step_count
    assert region.row_extent == ROWS
    assert region.feature_extent == FEATURES
    assert region.storage_dtype is torch.bfloat16


@pytest.mark.parametrize("step_count", (1, 9))
def test_discovery_rejects_steps_outside_t2_through_t8(step_count: int) -> None:
    built = _build_scan(step_count)

    assert discover_short_affine_scan_regions((built.info,)) == ()


@pytest.mark.parametrize("form", ("direct", "factored"))
def test_accepts_direct_and_factored_observations(form: str) -> None:
    built = _build_scan(2, observation_forms=(form, form))

    (region,) = discover_short_affine_scan_regions((built.info,))

    assert [step.observation for step in region.steps] == [
        step["observation"] for step in built.steps
    ]


def test_opaque_renamed_and_permuted_coefficient_producers_are_replayed() -> None:
    built = _build_scan(
        3,
        observation_forms=("factored", "direct", "factored"),
        opaque_producers=True,
    )

    (region,) = discover_short_affine_scan_regions((built.info,))

    assert region.producer_nodes == built.producer_nodes
    assert all(node.name.startswith("cipher_") for node in region.producer_nodes)
    assert {dependency.value for dependency in region.dependency_slices} == set(
        built.live_ins
    )


def test_discovery_rejects_coefficient_producer_with_external_user() -> None:
    built = _build_scan(2, opaque_producers=True, producer_live_out=True)

    assert discover_short_affine_scan_regions((built.info,)) == ()
    assert discover_direct_affine_candidates((built.info,)) == ()


def test_region_records_exact_boundary_effects_accesses_and_fences() -> None:
    built = _build_scan(2, observation_forms=("direct", "factored"))

    (region,) = discover_short_affine_scan_regions((built.info,))

    graph_nodes = tuple(built.info.graph.nodes)
    assert (
        region.source_interval
        == graph_nodes[
            graph_nodes.index(built.entry_load) : graph_nodes.index(
                built.steps[-1]["state_store"]
            )
            + 1
        ]
    )
    assert built.interior in region.source_interval
    assert built.interior not in region.owned_nodes
    assert region.live_outs == ()
    assert tuple(
        node
        for step in region.steps
        for node in (step.output_effect, step.state_effect)
    ) == tuple(
        node
        for built_step in built.steps
        for node in (built_step["output_store"], built_step["state_store"])
    )
    assert len(region.read_accesses) == 1
    assert region.read_accesses[0].kind is ShortAffineScanAccessKind.READ
    assert region.read_bases == (built.state_base,)
    assert len(region.write_accesses) == 4
    assert all(
        access.kind is ShortAffineScanAccessKind.WRITE
        for access in region.write_accesses
    )
    assert region.write_bases == (
        built.output_bases[0],
        built.state_base,
        built.output_bases[1],
    )
    state_fence = next(
        fence for fence in region.access_fences if fence.base is built.state_base
    )
    assert state_fence.preceding is not None
    assert state_fence.preceding.node is built.preceding_access
    assert state_fence.following is not None
    assert state_fence.following.node is built.following_access


@pytest.mark.parametrize("failure", ("effect", "live_out"))
def test_discovery_rejects_unsafe_boundaries(failure: str) -> None:
    built = _build_scan(
        2,
        extra_effect=failure == "effect",
        live_out=failure == "live_out",
    )

    assert discover_short_affine_scan_regions((built.info,)) == ()


@pytest.mark.parametrize("access", ("entry", "output", "state", "load_other"))
def test_candidate_rejects_masked_or_decorated_accesses(access: str) -> None:
    built = _build_scan(
        2,
        mask_access=None if access == "load_other" else access,
        entry_other="evict_last" if access == "load_other" else None,
    )

    # Discovery preserves the exact access contract for potential future users.
    assert len(discover_short_affine_scan_regions((built.info,))) == 1
    assert discover_direct_affine_candidates((built.info,)) == ()


def test_candidate_validation_rejects_access_kwargs() -> None:
    built = _build_scan(2)
    (candidate,) = discover_direct_affine_candidates((built.info,))
    entry_access = dataclasses.replace(
        candidate.region.entry_access,
        kwargs=(("unexpected", True),),
    )
    region = dataclasses.replace(
        candidate.region,
        entry_access=entry_access,
        read_accesses=(entry_access, *candidate.region.read_accesses[1:]),
    )

    with pytest.raises(ValueError, match="invalid direct affine candidate"):
        dataclasses.replace(candidate, region=region).validate()


@pytest.mark.parametrize(
    ("step_count", "expected_mma"),
    (
        (2, DirectAffineMma.M16N8),
        (3, DirectAffineMma.M16N8),
        (4, DirectAffineMma.M16N8),
        (5, DirectAffineMma.M16N16),
        (6, DirectAffineMma.M16N16),
        (7, DirectAffineMma.M16N16),
        (8, DirectAffineMma.M16N16),
    ),
)
def test_candidate_selects_smallest_algebraic_mma(
    step_count: int, expected_mma: DirectAffineMma
) -> None:
    built = _build_scan(step_count)

    (candidate,) = discover_direct_affine_candidates((built.info,))

    assert candidate.smallest_mma is expected_mma


@pytest.mark.parametrize(
    ("storage_dtype", "features"),
    ((torch.float16, FEATURES), (torch.bfloat16, 64)),
)
def test_candidate_rejects_unsupported_storage_or_feature_shape(
    storage_dtype: torch.dtype, features: int
) -> None:
    built = _build_scan(2, storage_dtype=storage_dtype, features=features)

    assert len(discover_short_affine_scan_regions((built.info,))) == 1
    assert discover_direct_affine_candidates((built.info,)) == ()


def test_revalidation_rejects_structural_fingerprint_drift() -> None:
    built = _build_scan(3)
    (candidate,) = discover_direct_affine_candidates((built.info,))
    assert revalidate_direct_affine_candidate(candidate, built.info) is not None

    decayed = built.steps[0]["decayed"]
    decayed.args = decayed.args[::-1]

    assert discover_direct_affine_candidates((built.info,))
    assert revalidate_direct_affine_candidate(candidate, built.info) is None
