from __future__ import annotations

import ast
from itertools import accumulate
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_interchanged_store_dce import _PAIRS
from test.test_cute_interchanged_store_dce import _execute as _execute_interchange
from test.test_cute_interchanged_store_dce import _program

import helion
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import ast_rename
from helion._testing import skipUnlessBackends


def _marker(owner: str | None, value: str = "partial") -> str:
    return lanes._lane_reduce_marker_expr(
        value, "sum", "cutlass.Float32(0)", 32, owner_lane=owner
    )


def _source(body: list[ast.AST]) -> str:
    return ast.unparse(ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[]))


def _body(source: str) -> list[ast.AST]:
    return list(ast.parse(source).body)


@pytest.mark.parametrize("outer,inner", [("feature", "row"), ("a", "b")])
@pytest.mark.parametrize("inner_extent", [2, 8])
def test_distinct_lane_owner_rejected_even_for_equal_extents(
    outer: str, inner: str, inner_extent: int
) -> None:
    inner_loop = lanes._create_lane_loop(
        inner,
        inner_extent,
        _body(
            f"partial = {outer} + {inner}\n"
            f"reduced = {_marker(outer)}\n"
            "sink.store(reduced)"
        ),
    )
    loop = lanes._create_lane_loop(outer, 8, [inner_loop])
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.validate_lane_reduce_owners([loop])
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.split_lane_loop_reductions([loop])


def test_marker_owner_survives_text_and_lane_cloning() -> None:
    statement = ast.parse(f"reduced = {_marker('axis')}").body[0]
    cloned = lanes._clone_stmt(statement)
    marker = lanes._is_lane_reduce_marker_assign(cloned)
    assert marker is not None and marker.owner_lane == "axis"
    loop = lanes._create_lane_loop("axis", 8, [statement])
    copied_loop = lanes._clone_lane_loop_with_body(loop, [cloned])
    lanes.validate_lane_reduce_owners([copied_loop])


def test_missing_lane_owner_cannot_reach_residual_restore() -> None:
    body = _body(f"reduced = {_marker('missing')}")
    with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
        lanes.validate_lane_reduce_owners(body)
    with pytest.raises(helion.exc.BackendUnsupported, match="no proved lane lowering"):
        lanes.restore_unprocessed_lane_reduce_markers(body)


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("copies", [1, 2])
def test_late_rename_carry_rejected_before_split_subpaths(
    guarded: bool, copies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "snapshot = carried\n"
    if copies == 2:
        text += "snapshot_2 = snapshot\n"
    source = "snapshot_2" if copies == 2 else "snapshot"
    text += f"partial = lane + 1\nreduced = {_marker('lane')}\n"
    update = f"next_value = {source} + partial"
    text += (f"if external:\n    {update}" if guarded else update) + "\n"
    text += "sink.store(reduced)"
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    body: list[ast.AST] = [loop, *_body("later.store(carried)")]
    monkeypatch.setattr(
        lanes,
        "_split_lane_loop_with_register_stash",
        lambda *_: pytest.fail("subpath selected before carry legality"),
    )
    lanes.validate_lane_reduce_owners(body)
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            body, rename_groups={"next_value": "carried", "carried": "carried"}
        )


@pytest.mark.parametrize("owner", [None, "lane"])
def test_reduced_scalar_can_update_an_existing_online_carry(owner: str | None) -> None:
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"snapshot = carried\npartial = lane + 1\nreduced = {_marker(owner)}\n"
            "next_value = snapshot + reduced\nsink.store(next_value)"
        ),
    )
    lanes.validate_lane_reduce_owners([loop])
    result = lanes.split_lane_loop_reductions(
        [loop], rename_groups={"next_value": "carried", "carried": "carried"}
    )
    code = _source(lanes.restore_unprocessed_lane_reduce_markers(result))
    assert "warp_reduction_sum" in code
    assert code.count("next_value = snapshot + reduced") == 1
    assert "_helion_lane_reduce" not in code


def test_legacy_per_lane_carry_is_not_an_owned_reduction_proof() -> None:
    def generate(owner: str | None) -> str:
        loop = lanes._create_lane_loop(
            "lane",
            8,
            _body(
                "partial = lane + 1\ncarried = carried + partial\n"
                f"reduced = {_marker(owner)}\n"
                "sink.store(reduced + carried)"
            ),
        )
        lanes.validate_lane_reduce_owners([loop])
        result = lanes.split_lane_loop_reductions([loop])
        return _source(lanes.restore_unprocessed_lane_reduce_markers(result))

    assert "reduced = partial" in generate(None)
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        generate("lane")


class _Sink:
    def __init__(self, values: dict[int, float], offset: int = 0) -> None:
        self.values = values
        self.offset = offset

    @property
    def iterator(self) -> _Sink:
        return self

    def __add__(self, offset: int) -> _Sink:
        return _Sink(self.values, self.offset + offset)

    def store(self, value: float) -> None:
        self.values[self.offset] = value


def _execute_scalars(
    body: list[ast.AST], renames: dict[str, str] | None = None
) -> tuple[dict[str, dict[int, float]], int]:
    """Execute exact small-integer scalar schedules, not CUDA collectives."""
    calls = 0

    def grouped_reduce(
        value: float,
        operation: str,
        identity: float,
        lane: int,
        lane_in_group: int,
        lane_mod_pre: int,
        *,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> float:
        nonlocal calls
        assert operation == "sum" and identity == 0
        assert lane == lane_in_group == lane_mod_pre == 0
        assert pre == group_count == 1 and group_span == 64
        calls += 1
        # All 64 inputs are identical finite integers. Any FP32 sum tree has
        # this exact result; this models the complete installed call signature.
        return value * group_span

    stores: dict[str, dict[int, float]] = {
        name: {} for name in ("out", "raw_out", "late_out")
    }
    namespace = {
        "cutlass": SimpleNamespace(
            Float32=float, Int32=int, range=range, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda extent, dtype: [0.0] * extent,
            arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0)),
        ),
        "_cute_grouped_reduce_shared_two_stage": grouped_reduce,
        **{name: _Sink(values) for name, values in stores.items()},
    }
    module = ast.parse(_source(body))
    ast_rename(module, renames or {})
    exec(
        compile(ast.fix_missing_locations(module), "<lane-carry-proof>", "exec"),
        namespace,
    )
    return stores, calls


@pytest.mark.parametrize("has_raw", [False, True])
@pytest.mark.parametrize("unduplicatable", [False, True])
def test_mixed_raw_and_renamed_carries_decline_before_stash(
    has_raw: bool, unduplicatable: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "lane + 1"
    )
    text = f"partial = {partial}\nsnapshot = late\nnext_late = snapshot + partial\n"
    if has_raw:
        text += "raw = raw + partial\n"
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text += f"reduced = {marker}\n(out.iterator + lane).store(partial + reduced)"
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    prefix, suffix = (
        _body("raw = 0.0\nlate = 0.0"),
        _body("raw_out.store(raw)\nlate_out.store(late)"),
    )
    renames = {"next_late": "late", "late": "late"}
    expected = 36.0 * (64 if unduplicatable else 1)
    # Give the reference the complete eight-lane reduction, not a raw-input
    # stand-in; both external carry updates are separately observable.
    reference_loop = lanes._create_lane_loop(
        "lane", 8, _body(text.replace(marker, repr(expected)))
    )
    reference, calls = _execute_scalars([*prefix, reference_loop, *suffix], renames)
    assert reference["raw_out"][0] == (expected if has_raw else 0.0)
    assert reference["late_out"][0] == expected
    assert calls == (8 if unduplicatable else 0)
    monkeypatch.setattr(
        lanes,
        "_split_lane_loop_with_register_stash",
        lambda *_: pytest.fail("stash selected before observable carry proof"),
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            [*prefix, loop, *suffix], rename_groups=renames
        )


def test_owned_raw_restore_would_lose_complete_reduction_numerically() -> None:
    def make(owner: str | None) -> list[ast.AST]:
        marker = lanes._lane_reduce_marker_expr(
            "partial", "sum", "cutlass.Float32(0)", 1, owner_lane=owner
        )
        return [
            *_body("carried = 0.0"),
            lanes._create_lane_loop(
                "lane",
                8,
                _body(
                    "partial = lane + 1\ncarried = carried + partial\n"
                    f"reduced = {marker}\n"
                    "(out.iterator + lane).store(reduced + carried)"
                ),
            ),
            *_body("raw_out.store(carried)"),
        ]

    legacy = lanes.split_lane_loop_reductions(make(None))
    observed, calls = _execute_scalars(legacy)
    expected = [36.0 + value for value in accumulate(range(1, 9))]
    assert observed["raw_out"][0] == 36.0 and calls == 0
    assert observed["out"][7] == 44.0 and expected[-1] == 72.0
    assert list(observed["out"].values()) != expected
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(make("lane"))


@pytest.mark.parametrize("unduplicatable", [False, True])
def test_complete_owned_split_has_numeric_and_single_execution_proof(
    unduplicatable: bool,
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "lane + 1"
    )
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"partial = {partial}\nreduced = {marker}\n"
            "(out.iterator + lane).store(partial + reduced)"
        ),
    )
    lowered = lanes.split_lane_loop_reductions([loop])
    values, calls = _execute_scalars(lowered)
    scale = 64 if unduplicatable else 1
    assert values["out"] == {
        index: float((36 + index + 1) * scale) for index in range(8)
    }
    assert calls == (8 if unduplicatable else 0)


def test_marker_dependent_online_update_uses_full_lane_sum_once() -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"snapshot = carried\npartial = lane + 1\nreduced = {marker}\n"
            "next_value = snapshot + reduced\nout.store(next_value)"
        ),
    )
    renames = {"next_value": "carried", "carried": "carried"}
    lowered = lanes.split_lane_loop_reductions(
        [*_body("carried = 6.0"), loop], rename_groups=renames
    )
    values, calls = _execute_scalars(lowered, renames)
    assert values["out"] == {0: 42.0} and calls == 0


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
def test_mixed_marker_and_partial_carry_remains_lane_varying(
    renamed: bool, indirect: bool, guarded: bool
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text = f"partial = lane + 1\nsnapshot = carried\nreduced = {marker}\n"
    if indirect:
        text += "mixed = partial + reduced\n"
    target = "next_value" if renamed else "carried"
    expression = "mixed" if indirect else "partial + reduced"
    update = f"{target} = snapshot + {expression}"
    text += (f"if flag:\n    {update}" if guarded else update) + "\n"
    text += "(out.iterator + lane).store(partial + reduced)"
    renames = {"next_value": "carried", "carried": "carried"} if renamed else {}
    prefix = _body("carried = 0.0\nflag = True")
    suffix = _body("late_out.store(carried)")
    reference = lanes._create_lane_loop("lane", 8, _body(text.replace(marker, "36.0")))
    values, calls = _execute_scalars([*prefix, reference, *suffix], renames)
    assert values["late_out"][0] == 324.0 and calls == 0
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    indices = {
        index
        for index, statement in enumerate(loop.body)
        if lanes._is_lane_reduce_marker_assign(statement) is not None
    }
    normalized = [lanes._clone_stmt(statement) for statement in loop.body]
    for statement in normalized:
        ast_rename(statement, renames)
    # The legacy marker-taint test misses the independent partial in both
    # direct and aliased expressions. Post-finalization variation must not.
    assert not lanes._has_extra_cross_lane_carry(normalized, "lane", indices)
    assert lanes._has_extra_cross_lane_carry(
        normalized, "lane", indices, finalized_markers=True
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            [*prefix, loop, *suffix], rename_groups=renames
        )


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_finalized_only_carry_is_uniform_after_reduction(
    renamed: bool, indirect: bool
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    text = f"partial = lane + 1\nsnapshot = carried\nreduced = {marker}\n"
    if indirect:
        text += "copy_reduced = reduced\n"
    target = "next_value" if renamed else "carried"
    text += f"{target} = snapshot + {'copy_reduced' if indirect else 'reduced'}\n"
    text += f"out.store({target})"
    renames = {"next_value": "carried", "carried": "carried"} if renamed else {}
    loop = lanes._create_lane_loop("lane", 8, _body(text))
    lowered = lanes.split_lane_loop_reductions(
        [*_body("carried = 6.0"), loop], rename_groups=renames
    )
    values, calls = _execute_scalars(lowered, renames)
    assert values["out"] == {0: 42.0} and calls == 0


@pytest.mark.parametrize("unduplicatable", [False, True])
@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("inside_observed", [False, True])
def test_finalized_carry_requires_a_once_per_tile_schedule(
    unduplicatable: bool, renamed: bool, inside_observed: bool
) -> None:
    partial = (
        "_cute_grouped_reduce_shared_two_stage("
        "cutlass.Float32(lane + 1), 'sum', cutlass.Float32(0), "
        "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), "
        "pre=1, group_span=64, group_count=1)"
        if unduplicatable
        else "64 * (lane + 1)"
    )
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    target = "next_value" if renamed else "carried"
    renames = {"carried": "carried", "next_value": "carried"} if renamed else {}
    consumer = target if inside_observed else "reduced"
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(
            f"partial = {partial}\nsnapshot = carried\nreduced = {marker}\n"
            f"{target} = snapshot + reduced\n"
            f"(out.iterator + lane).store(partial + {consumer})"
        ),
    )
    program = [*_body("carried = 6.0"), loop, *_body("late_out.store(carried)")]
    if unduplicatable:
        with pytest.raises(helion.exc.BackendUnsupported, match="once-per-tile"):
            lanes.split_lane_loop_reductions(program, rename_groups=renames)
    else:
        lowered = lanes.split_lane_loop_reductions(program, rename_groups=renames)
        values, calls = _execute_scalars(lowered, renames)
        assert values["late_out"] == {0: 2310.0} and calls == 0
        assert values["out"] == {
            lane: float((lane + 1) * 64 + (2310 if inside_observed else 2304))
            for lane in range(8)
        }


@pytest.mark.parametrize("compound", [False, True])
def test_finalized_carry_without_a_single_assignment_proof_declines(
    compound: bool,
) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "partial", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    update = (
        "if flag:\n    carried = carried + reduced"
        if compound
        else "carried = carried + reduced\ncarried = carried + 1"
    )
    loop = lanes._create_lane_loop(
        "lane",
        8,
        _body(f"partial = lane + 1\nreduced = {marker}\n{update}"),
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="once-per-tile"):
        lanes.split_lane_loop_reductions(
            [*_body("carried = 6.0\nflag = True"), loop, *_body("out.store(carried)")]
        )


def test_owned_marker_cannot_use_an_unproved_direct_restore() -> None:
    loop = lanes._create_lane_loop(
        "lane", 8, _body(f"partial = lane + 1\nreduced = {_marker('lane')}")
    )
    marker = lanes._is_lane_reduce_marker_assign(loop.body[1])
    assert marker is not None
    with pytest.raises(
        helion.exc.BackendUnsupported, match="complete per-lane restore"
    ):
        lanes._restore_per_lane_markers(loop, [(1, marker)])


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_ordinary_owner_proof_does_not_change_supported_split(
    guarded: bool, dependent: bool
) -> None:
    def generate(owner: str | None) -> str:
        text = f"partial = lane + 1\nreduced = {_marker(owner)}\n"
        if dependent:
            text += "other = partial + reduced\n"
            text += f"reduced_again = {_marker(owner, 'other')}\n"
        text += f"sink.store({'reduced_again' if dependent else 'reduced'})"
        body: list[ast.AST] = _body(text)
        if guarded:
            body = [
                ast.If(
                    test=ast.Name(id="flag", ctx=ast.Load()),
                    body=cast("list[ast.stmt]", body),
                    orelse=[],
                )
            ]
        loop = lanes._create_lane_loop("lane", 8, body)
        lanes.validate_lane_reduce_owners([loop])
        result = lanes.split_lane_loop_reductions([loop], uniform_names={"flag"})
        return _source(lanes.restore_unprocessed_lane_reduce_markers(result))

    assert generate(None) == generate("lane")


def test_supported_serial_interchange_preserves_owned_marker() -> None:
    def generate(owned: bool) -> str:
        loop = _program()
        if owned:
            for node in ast.walk(loop):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "_helion_lane_reduce":
                        assert len(node.args) == 8
                        node.args.extend(
                            [ast.Constant(value=1), ast.Constant(value="lane")]
                        )
        lanes.validate_lane_reduce_owners([loop])
        body = lanes.interchange_lane_outside_serial_reductions(
            [loop], proven_disjoint_tensor_pairs=_PAIRS
        )
        lanes.validate_lane_reduce_owners(body)
        body = lanes.split_lane_loop_reductions(
            body,
            uniform_names={"rows", "valid_rows", "valid_cols"},
            proven_disjoint_tensor_pairs=_PAIRS,
        )
        return _source(lanes.restore_unprocessed_lane_reduce_markers(body))

    assert generate(False) == generate(True)


@pytest.mark.parametrize(
    ("rows", "valid_rows", "valid_cols"), [(3, 3, 4), (3, 2, 3), (0, 0, 4)]
)
def test_owned_serial_interchange_has_complete_numeric_reduction(
    rows: int, valid_rows: int, valid_cols: int
) -> None:
    loop = _program()
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_helion_lane_reduce"
        ):
            node.args.extend([ast.Constant(value=1), ast.Constant(value="lane")])
    lanes.validate_lane_reduce_owners([loop])
    lowered = lanes.interchange_lane_outside_serial_reductions(
        [loop], proven_disjoint_tensor_pairs=_PAIRS
    )
    lanes.validate_lane_reduce_owners(lowered)
    actual = _execute_interchange(lowered, rows, valid_rows, valid_cols)
    values = actual["source"].values.reshape(rows, 4)
    mask = (np.arange(rows)[:, None] < valid_rows) & (
        np.arange(4)[None, :] < valid_cols
    )
    masked = np.where(mask, values, 0.0)
    product = masked * actual["weight"].values
    expected = np.where(mask, product - 2 * product.sum(-1, keepdims=True), -99.0)
    np.testing.assert_array_equal(actual["output"].values.reshape(rows, 4), expected)
    np.testing.assert_array_equal(actual["column"].values, masked.sum(0))
    assert actual["output"].stores == valid_rows * valid_cols
    assert actual["column"].stores == 4


@skipUnlessBackends(["cute"])
def test_exact_rejected_backward_config_declines_before_emission() -> None:
    x = torch.empty((4096, 4096), dtype=torch.bfloat16)
    args = (
        torch.empty_like(x),
        x,
        torch.empty(4096),
        torch.empty(4096),
        torch.empty(4096, dtype=torch.bfloat16),
        True,
    )
    kernel = helion.kernel(
        layer_norm_bwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    config = helion.Config.from_dict(
        {
            "block_sizes": [256, 256],
            "cute_cluster_n": 1,
            "cute_host_paired_sum": "off",
            "cute_lane_layouts": ["blocked", "blocked", "blocked"],
            "cute_min_blocks_per_mp": 4,
            "cute_reduction_group_rows": 4,
            "cute_reduction_reloads": ["gmem"],
            "cute_reduction_schedule": "pipelined",
            "cute_reduction_sequence": "bounded_layout",
            "cute_vector_widths": [2, 8, 1],
            "load_eviction_policies": ["streaming", "last", "last", "l2_last", "first"],
            "num_threads": [0, 32, 256],
        }
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, args)
        with pytest.raises(helion.exc.BackendUnsupported, match="different lane owner"):
            bound.to_code(config)
