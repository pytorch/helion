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


def test_strided_restore_marker_is_finalized_per_lane() -> None:
    # A device-lane-loop marker whose body kept the per-element strided
    # semantics is finalized across the thread group in place when the
    # two-pass split is refused, instead of rejecting the config.
    text = lanes._lane_reduce_marker_expr(
        "partial",
        "sum",
        "cutlass.Float32(0)",
        32,
        owner_lane="lane",
        strided_restore=True,
    )
    loop = lanes._create_lane_loop(
        "lane", 8, _body(f"partial = lane + 1\nreduced = {text}")
    )
    marker = lanes._is_lane_reduce_marker_assign(loop.body[1])
    assert marker is not None
    assert marker.owner_lane == "lane"
    assert marker.strided_restore
    restored = lanes._restore_per_lane_markers(loop, [(1, marker)])
    source = _source(list(restored.body))
    assert "_helion_lane_reduce" not in source
    assert "warp_reduction_sum" in source
    assert "reduced = " in source


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


# --- a reduction input a device loop of the lane body accumulates -----------

_TID = "cutlass.Int32(cute.arch.thread_idx()[0])"
_COLLECTIVE_END = (
    "amax = _cute_grouped_reduce_shared_two_stage(length, 'max', cutlass.Int64(0), "
    "0, 0, 0, pre=1, group_span=256, group_count=1)\n"
)
# The device function's aliases of the carried values: the jagged loop's
# output ``v_8`` is ``row_sums``, the lane body's ``v_9`` is ``mean_acc``.
_ROW_SUMS_RENAMES = {
    "v_8": "row_sums",
    "row_sums": "row_sums",
    "v_9": "mean_acc",
    "mean_acc": "mean_acc",
}


def _masked_marker(value: str, *, strided_restore: bool = True) -> str:
    return lanes._lane_reduce_marker_expr(
        value,
        "sum",
        "cutlass.Float32(0)",
        256,
        group_pre=1,
        group_span=256,
        group_lane_expr=_TID,
        group_count=1,
        owner_lane="lane",
        strided_restore=strided_restore,
    )


def _accumulated_rows_body(
    *, collective_end: bool, strided_restore: bool = True
) -> str:
    """The nested-row seed's column lane body under dynamic shapes.

    A jagged device loop accumulates ``row_sums`` under its loop-output name
    ``v_8`` (``row_sums_copy = row_sums`` at the top of its body is the
    carry's phi copy); the column mask then turns ``row_sums.sum()`` into a
    masked lane reduction of ``_mask_to``, and the result updates the
    cross-lane ``mean_acc`` carry.
    """
    end = _COLLECTIVE_END if collective_end else "amax = length\n"
    return (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mask = indices < M\n"
        "mean_acc_copy = mean_acc\n"
        "row_sums = cutlass.Float32(0.0)\n"
        f"{end}"
        "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(amax), "
        "cutlass.Int32(32)):\n"
        "    row_sums_copy = row_sums\n"
        "    partial = (x.iterator + cutlass.Int32(tile_offset_2 * M + indices)"
        " * cutlass.Int32(x.layout.stride[0])).load() if mask else "
        "cutlass.Float32(0)\n"
        "    v_8 = row_sums_copy + partial\n"
        "_mask_to = cutlass.Float32(row_sums) if mask else cutlass.Float32(0)\n"
        f"sum_2 = cutlass.Float32({_masked_marker('_mask_to', strided_restore=strided_restore)})\n"
        "v_9 = mean_acc_copy + sum_2\n"
    )


def _split(
    text: str,
    renames: dict[str, str],
    disjoint: set[frozenset[str]] | None = None,
) -> str:
    loop = lanes._create_lane_loop("lane", 2, _body(text))
    lanes.validate_lane_reduce_owners([loop])
    result = lanes.split_lane_loop_reductions(
        [loop], rename_groups=renames, proven_disjoint_tensor_pairs=disjoint
    )
    return _source(lanes.restore_unprocessed_lane_reduce_markers(result))


def _lane_passes(code: str) -> list[str]:
    """The bodies of the lane loops of ``code``, in order."""
    return [
        _source(list(node.body))
        for node in ast.parse(code).body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "lane"
    ]


def test_lane_reduction_of_an_accumulated_row_runs_after_the_loop_that_accumulates_it() -> (
    None
):
    """The accumulate pass folds ``row_sums`` only after the jagged loop ran.

    The slice producing the reduction input reads the rename groups: the loop
    writes ``v_8``, which is ``row_sums``.  The two-pass split used to fold
    the fresh zero in a first pass and run the loop in the second (the
    dynamic-shape nested-row seeds of jagged_layer_norm returned mean =
    variance = 0).
    """
    code = _split(_accumulated_rows_body(collective_end=False), _ROW_SUMS_RENAMES)
    accumulate = _lane_passes(code)[0]
    assert "sum_2_lane_acc = sum_2_lane_acc +" in accumulate
    assert accumulate.index("for tile_offset_2 in") < accumulate.index(
        "sum_2_lane_acc = sum_2_lane_acc +"
    )
    assert code.count("v_9 = mean_acc_copy + sum_2") == 1
    assert code.index("sum_2 = cutlass.Float32(sum_2_lane_acc_reduced)") < code.index(
        "v_9 = mean_acc_copy + sum_2"
    )


def test_an_accumulated_row_under_a_collective_end_is_reduced_per_lane() -> None:
    """The jagged loop's trip count is a cross-thread max, which no pass may re-run: the owned marker is restored per lane, after the loop, and an owner without a per-lane restore is declined rather than split."""
    code = _split(_accumulated_rows_body(collective_end=True), _ROW_SUMS_RENAMES)
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert body.index("sum_2 = cutlass.Float32(_mask_to_reduced)") < body.index(
        "v_9 = mean_acc_copy + sum_2"
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _accumulated_rows_body(collective_end=True, strided_restore=False),
            _ROW_SUMS_RENAMES,
        )


def test_a_carry_the_rename_groups_do_not_map_is_read_off_its_phi_copy() -> None:
    """Without the alias of ``v_8`` the slice sees the initializer alone; the loop's ``row_sums_copy = row_sums`` marks it as the carry, and the marker is restored per lane instead of being folded ahead of the loop."""
    code = _split(_accumulated_rows_body(collective_end=False), {})
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(_accumulated_rows_body(collective_end=False, strided_restore=False), {})


def test_a_loop_reading_the_reduction_input_without_carrying_it_stays_a_consumer() -> (
    None
):
    """A loop that only reads an input of the reduction (no phi copy) is no producer: the split keeps it in the consume pass."""
    text = (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mask = indices < M\n"
        "mean_acc_copy = mean_acc\n"
        "row_sums = (x.iterator + cutlass.Int32(indices) * "
        "cutlass.Int32(x.layout.stride[0])).load() if mask else cutlass.Float32(0)\n"
        "for k in range(4):\n"
        "    scaled = row_sums * k\n"
        "_mask_to = cutlass.Float32(row_sums) if mask else cutlass.Float32(0)\n"
        f"sum_2 = cutlass.Float32({_masked_marker('_mask_to')})\n"
        "v_9 = mean_acc_copy + sum_2\n"
    )
    code = _split(text, _ROW_SUMS_RENAMES)
    accumulate, consume = _lane_passes(code)
    assert "sum_2_lane_acc = sum_2_lane_acc +" in accumulate
    assert "for k in range(4)" not in accumulate
    assert "for k in range(4)" in consume


_XYZ_DISJOINT = {frozenset({"x", "y"}), frozenset({"x", "z"}), frozenset({"y", "z"})}
# The K loop's output ``v_8`` and the later rewrite ``v_11`` are both ``acc``.
_ACCUMULATED_RENAMES = {
    "v_8": "acc",
    "v_11": "acc",
    "acc": "acc",
    "v_9": "mean_acc",
    "mean_acc": "mean_acc",
}
_CONTINUING_LOOP = (
    "for tile_offset_3 in range(cutlass.Int32(0), cutlass.Int32(4), "
    "cutlass.Int32(1)):\n"
    "    acc_copy_1 = acc\n"
    "    v_11 = acc_copy_1 + (z.iterator + cutlass.Int32(tile_offset_3 * M + "
    "indices)).load()\n"
)
_JOIN_REWRITE = (
    "if flag:\n    v_11 = acc + cutlass.Float32(1.0)\nelse:\n    v_11 = acc\n"
)


def _stored_accumulator_body(*, after_marker: str = "") -> str:
    """A per-column accumulator reduced across the columns and stored.

    A K loop accumulates ``acc`` under its loop-output name ``v_8``, the
    unmasked (static-shape) reduction of ``acc`` across the column lanes
    updates the carried ``mean_acc``, and ``acc`` is stored per column, after
    ``after_marker`` rewrote it under another spelling of its group.
    """
    return (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "mean_acc_copy = mean_acc\n"
        "acc = cutlass.Float32(0.0)\n"
        "for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(4), "
        "cutlass.Int32(1)):\n"
        "    acc_copy = acc\n"
        "    partial = (x.iterator + cutlass.Int32(tile_offset_2 * M + indices))"
        ".load()\n"
        "    v_8 = acc_copy + partial\n"
        f"sum_2 = cutlass.Float32({_masked_marker('acc')})\n"
        "v_9 = mean_acc_copy + sum_2\n"
        f"{after_marker}"
        "(y.iterator + cutlass.Int32(indices)).store(cutlass.Float32(acc))\n"
    )


def test_the_initializer_of_an_accumulator_a_lane_varying_loop_rewrites_runs_in_every_pass() -> (
    None
):
    """``acc = 0`` is lane-varying through the K loop's rewrite of ``acc``.

    The loop writes ``acc`` under ``v_8``; the classification of the consume
    pass read the names as spelled, took the initializer for lane-invariant
    and emitted it once between the passes, so the consume pass's loop
    continued every lane after the first from the previous lane's final
    ``acc`` and stored that.
    """
    code = _split(_stored_accumulator_body(), _ACCUMULATED_RENAMES, _XYZ_DISJOINT)
    accumulate, consume = _lane_passes(code)
    init = "acc = cutlass.Float32(0.0)"
    assert code.count(init) == 2
    assert (
        accumulate.index(init)
        < accumulate.index("for tile_offset_2 in")
        < accumulate.index("sum_2_lane_acc = sum_2_lane_acc +")
    )
    assert (
        consume.index(init)
        < consume.index("for tile_offset_2 in")
        < consume.index(".store(")
    )
    assert code.count("v_9 = mean_acc_copy + sum_2") == 1


@pytest.mark.parametrize(
    "rewrite", [_CONTINUING_LOOP, _JOIN_REWRITE], ids=["loop", "join"]
)
def test_a_rewrite_of_the_reduction_input_after_its_marker_is_a_consumer(
    rewrite: str,
) -> None:
    """A later loop or if-join rewriting ``acc`` is no producer of the reduction.

    The producer slice is read over the statements before the marker: sliced
    over the whole body, the later rewrite of ``acc``'s group was taken into
    the accumulate pass and the fold read the value behind it.
    """
    code = _split(
        _stored_accumulator_body(after_marker=rewrite),
        _ACCUMULATED_RENAMES,
        _XYZ_DISJOINT,
    )
    accumulate, consume = _lane_passes(code)
    head = rewrite.partition("\n")[0]
    assert head not in accumulate
    assert accumulate.index("for tile_offset_2 in") < accumulate.index(
        "sum_2_lane_acc = sum_2_lane_acc +"
    )
    assert (
        consume.index("for tile_offset_2 in")
        < consume.index(head)
        < consume.index(".store(")
    )


def test_a_dependent_chains_consume_pass_restarts_an_accumulator_a_lane_varying_loop_rewrites() -> (
    None
):
    """The chained split's final pass re-runs ``acc = 0`` per lane too.

    Two chained reductions (the second's input reads the first's result) are
    followed by a loop that accumulates ``acc`` from the second result under
    ``v_8`` and a store of ``acc``: the initializer belongs to the final lane
    pass, not to the invariant tail between the passes.
    """
    text = (
        f"indices = tile_offset + {_TID} + lane * 256\n"
        "p = (x.iterator + cutlass.Int32(indices)).load()\n"
        f"m1 = cutlass.Float32({_masked_marker('p')})\n"
        "q = p * m1\n"
        f"m2 = cutlass.Float32({_masked_marker('q')})\n"
        "acc = cutlass.Float32(0.0)\n"
        "for k in range(cutlass.Int32(0), cutlass.Int32(4), cutlass.Int32(1)):\n"
        "    acc_copy = acc\n"
        "    v_8 = acc_copy + q * m2\n"
        "(y.iterator + cutlass.Int32(indices)).store(cutlass.Float32(acc))\n"
    )
    code = _split(text, {"v_8": "acc", "acc": "acc"}, _XYZ_DISJOINT)
    passes = _lane_passes(code)
    assert len(passes) == 3
    init = "acc = cutlass.Float32(0.0)"
    assert code.count(init) == 1
    consume = passes[-1]
    assert (
        consume.index(init) < consume.index("for k in range") < consume.index(".store(")
    )


_STORED_TOTAL = (
    "(tot.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot.layout.stride[0]))"
    ".store(cutlass.Float32(sum_2))\n"
)
_SCALED_ROW = (
    "(out.iterator + cutlass.Int32(indices) * cutlass.Int32(out.layout.stride[0]))"
    ".store(cutlass.Float32(_mask_to * sum_2))\n"
)
_STORED_CARRY_COPY = (
    "(tot.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot.layout.stride[0]))"
    ".store(cutlass.Float32(mean_acc_copy))\n"
)
# The tensors the restored bodies store to are not the one they load.
_RESTORED_DISJOINT = {frozenset({"x", "tot"}), frozenset({"x", "out"})}
_CARRY_COPIES = {
    0: "",
    1: "mean_acc_copy = mean_acc\n",
    2: "mean_acc_copy = mean_acc\nmean_acc_copy_0 = mean_acc_copy\n",
}


def _restored_rows_body(tail: str, *, carry_copies: int = 0) -> str:
    """``_accumulated_rows_body`` under a collective end, ``tail`` for its carry update.

    The jagged loop's collective trip count keeps the two-pass split off this
    body, so its marker is restored per lane.  ``carry_copies`` chained phi
    copies of ``mean_acc`` (``mean_acc_copy = mean_acc``, ``mean_acc_copy_0 =
    mean_acc_copy``) precede the loop for a ``tail`` that updates the carry.
    """
    body = _accumulated_rows_body(collective_end=True)
    update = "v_9 = mean_acc_copy + sum_2\n"
    copy = "mean_acc_copy = mean_acc\n"
    assert body.endswith(update) and body.count(copy) == 1
    return body[: -len(update)].replace(copy, _CARRY_COPIES[carry_copies]) + tail


@pytest.mark.parametrize("carry_copies", [1, 2])
def test_a_strided_restore_is_kept_for_a_lane_carry_update(carry_copies: int) -> None:
    """A lane's share is complete through a carry: the update off the last of the chained phi copies runs per lane and the marker stays restored (jagged_layer_norm's schedule)."""
    copy = "mean_acc_copy" if carry_copies == 1 else "mean_acc_copy_0"
    code = _split(
        _restored_rows_body(f"v_9 = {copy} + sum_2\n", carry_copies=carry_copies),
        _ROW_SUMS_RENAMES,
        _RESTORED_DISJOINT,
    )
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code and "_lane_acc" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        f"v_9 = {copy} + sum_2"
    )


def test_a_stored_strided_reduction_is_totalled_across_the_lanes() -> None:
    """A store of the result would read one lane's share: the lane loop stays whole, each lane's share is folded into a total the loop carries, and the store runs once after the loop on the total, owner-guarded."""
    code = _split(
        _restored_rows_body(_STORED_TOTAL), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT
    )
    (body,) = _lane_passes(code)
    assert "_lane_acc" not in code
    assert code.index("sum_2_lane_total = cutlass.Float32(0)") < code.index(
        "for lane in range(2):"
    )
    assert body.index("for tile_offset_2 in") < body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert "sum_2_lane_share = _mask_to_reduced" in body
    assert (
        "sum_2_lane_total = sum_2_lane_total + cutlass.Float32(sum_2_lane_share)"
        in body
    )
    assert "tot.iterator" not in body
    after = code[code.index("sum_2 = cutlass.Float32(sum_2_lane_total)") :]
    assert after.index(
        "if cutlass.Int32(cute.arch.thread_idx()[0]) % 256 < 1:"
    ) < after.index("tot.iterator")


@pytest.mark.parametrize(
    ("tail", "carry_copies"),
    [
        (_SCALED_ROW, 0),
        ("v_9 = mean_acc_copy + sum_2\n" + _STORED_TOTAL, 1),
        ("v_9 = mean_acc_copy + sum_2\n" + _STORED_CARRY_COPY, 1),
    ],
    ids=["scaled_per_lane", "carry_and_store", "second_reader_of_the_carry"],
)
def test_a_strided_reduction_without_a_whole_loop_lowering_is_declined(
    tail: str, carry_copies: int
) -> None:
    """A result consumed inside the lanes (scaled back into the per-lane values), beside a carry update, or whose carry is read elsewhere has neither a complete restore nor a lane-invariant tail: the config is declined."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _restored_rows_body(tail, carry_copies=carry_copies),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_the_restore_declines_a_stored_strided_reduction() -> None:
    loop = lanes._create_lane_loop("lane", 2, _body(_restored_rows_body(_STORED_TOTAL)))
    (index,) = [
        i
        for i, stmt in enumerate(loop.body)
        if lanes._is_lane_reduce_marker_assign(stmt) is not None
    ]
    marker = lanes._is_lane_reduce_marker_assign(loop.body[index])
    assert marker is not None and marker.strided_restore
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        lanes._restore_per_lane_markers(loop, [(index, marker)], _ROW_SUMS_RENAMES)


_MAX_CARRY_UPDATES = {
    "cute_math_max_under_casts": (
        "v_9 = cute.math.max(cutlass.Float32(mean_acc_copy), "
        "cutlass.Float32(amax_2), propagate_nan=True)\n"
    ),
    "ternary": "v_9 = mean_acc_copy if mean_acc_copy > amax_2 else amax_2\n",
    "ternary_swapped": "v_9 = amax_2 if mean_acc_copy < amax_2 else mean_acc_copy\n",
    "builtin_max": "v_9 = max(mean_acc_copy, amax_2)\n",
}
_MIN_OF_A_MAX_MARKER = {
    "ternary_min": "v_9 = mean_acc_copy if mean_acc_copy < amax_2 else amax_2\n",
    "cute_math_min": (
        "v_9 = cute.math.min(cutlass.Float32(mean_acc_copy), "
        "cutlass.Float32(amax_2), propagate_nan=True)\n"
    ),
}
_GUARDED_ATOMIC_ADD = (
    "if cute.arch.thread_idx()[0] == 0:\n"
    "    cute.arch.atomic_add((tot.iterator + cute.crd2idx((tile_offset,), "
    "tot.layout)).llvm_ptr, val=cutlass.Float32(sum_2), sem='relaxed')\n"
    "else:\n"
    "    pass\n"
)
_SUM_MARKER_LINE = f"sum_2 = cutlass.Float32({_masked_marker('_mask_to')})\n"
_STORED_AMAX = (
    "(tot2.iterator + cutlass.Int32(tile_offset) * cutlass.Int32(tot2.layout.stride[0]))"
    ".store(cutlass.Float32(amax_3))\n"
)
_TWO_TENSORS_DISJOINT = _RESTORED_DISJOINT | {
    frozenset({"x", "tot2"}),
    frozenset({"tot", "tot2"}),
}


def _max_marker(value: str) -> str:
    return lanes._lane_reduce_marker_expr(
        value,
        "max",
        "cutlass.Float32(float('-inf'))",
        256,
        group_pre=1,
        group_span=256,
        group_lane_expr=_TID,
        group_count=1,
        owner_lane="lane",
        strided_restore=True,
    )


def _restored_rows_max_body(tail: str) -> str:
    """``_restored_rows_body`` with a max marker ``amax_2`` in place of the sum, one phi copy of the carry."""
    body = _restored_rows_body("", carry_copies=1)
    assert body.count(_SUM_MARKER_LINE) == 1
    return (
        body.replace(
            _SUM_MARKER_LINE,
            f"amax_2 = cutlass.Float32({_max_marker('_mask_to')})\n",
        )
        + tail
    )


def _assert_restored_before(code: str, update: str) -> None:
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        update.strip()
    )


@pytest.mark.parametrize(
    "update", list(_MAX_CARRY_UPDATES.values()), ids=list(_MAX_CARRY_UPDATES)
)
def test_a_strided_restore_is_kept_for_a_running_max_carry(update: str) -> None:
    """The max over the lanes of the lanes' maxes is the max over the tile: a running-max carry (``cute.math.max`` under casts, the ternary, ``max()``) keeps the per-lane restore."""
    _assert_restored_before(
        _split(_restored_rows_max_body(update), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT),
        update,
    )


@pytest.mark.parametrize(
    "update", list(_MIN_OF_A_MAX_MARKER.values()), ids=list(_MIN_OF_A_MAX_MARKER)
)
def test_a_min_carry_of_a_max_reduction_is_declined(update: str) -> None:
    """A carry folding the other extremum would keep one lane's max per lane: it is no carry of this marker, and the tail touching the carry has no re-reduce either."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(_restored_rows_max_body(update), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT)


@pytest.mark.parametrize(
    "update",
    [
        "v_9 = mean_acc_copy + cutlass.Float32(sum_2)\n",
        "v_9 = cutlass.Float32(mean_acc_copy + sum_2)\n",
    ],
    ids=["cast_operand", "cast_sum"],
)
def test_a_strided_restore_is_kept_for_a_carry_update_under_a_cast(
    update: str,
) -> None:
    """A cast to the accumulator dtype around an operand or the sum is the render's spelling of the same carry update."""
    _assert_restored_before(
        _split(
            _restored_rows_body(update, carry_copies=1),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        ),
        update,
    )


def test_a_carry_update_under_a_narrowing_cast_is_declined() -> None:
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            _restored_rows_body(
                "v_9 = mean_acc_copy + cutlass.BFloat16(sum_2)\n", carry_copies=1
            ),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_a_strided_restore_is_kept_for_a_carry_update_in_an_if_join() -> None:
    """An if-join that adds the share under a lane-invariant condition and keeps the carry otherwise adds every lane's share or none: still the carry's reduction over the tile."""
    update = (
        "if cond_u:\n    v_9 = mean_acc_copy + sum_2\nelse:\n    v_9 = mean_acc_copy\n"
    )
    code = _split(
        _restored_rows_body(update, carry_copies=1),
        _ROW_SUMS_RENAMES,
        _RESTORED_DISJOINT,
    )
    (body,) = _lane_passes(code)
    assert "_lane_total" not in code
    assert body.index("_cute_grouped_reduce_shared_two_stage(_mask_to,") < body.index(
        "if cond_u:"
    )


@pytest.mark.parametrize(
    "update",
    [
        "if mask:\n    v_9 = mean_acc_copy + sum_2\nelse:\n    v_9 = mean_acc_copy\n",
        (
            "if mean_acc_copy > 0:\n    v_9 = mean_acc_copy + sum_2\n"
            "else:\n    v_9 = mean_acc_copy\n"
        ),
        (
            "if cond_u:\n    v_9 = mean_acc_copy + sum_2\n"
            "else:\n    v_9 = cutlass.Float32(0.0)\n"
        ),
    ],
    ids=["lane_varying_condition", "condition_reads_the_carry", "else_resets"],
)
def test_an_if_join_that_is_not_the_same_carry_in_every_lane_is_declined(
    update: str,
) -> None:
    """A condition varying with the lane or reading the carry, or a branch that resets the carry, would leave some lanes' shares out."""
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="no proved complete per-lane restore|loop-carried value",
    ):
        _split(
            _restored_rows_body(update, carry_copies=1),
            _ROW_SUMS_RENAMES,
            _RESTORED_DISJOINT,
        )


def test_an_atomic_add_of_a_strided_reduction_runs_once_on_the_lanes_total() -> None:
    """The emitted atomic spells its operand by keyword (``val=``); it is a lane-invariant consumer of the total, so the shares are totalled and the one guarded atomic runs after the loop."""
    code = _split(
        _restored_rows_body(_GUARDED_ATOMIC_ADD), _ROW_SUMS_RENAMES, _RESTORED_DISJOINT
    )
    (body,) = _lane_passes(code)
    assert (
        "sum_2_lane_total = sum_2_lane_total + cutlass.Float32(sum_2_lane_share)"
        in body
    )
    assert "atomic_add" not in body
    after = code[code.index("sum_2 = cutlass.Float32(sum_2_lane_total)") :]
    assert after.count("cute.arch.atomic_add(") == 1
    assert after.index("if cute.arch.thread_idx()[0] == 0:") < after.index(
        "cute.arch.atomic_add("
    )


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("cute.arch.atomic_add(p, val=v, sem='relaxed')", ("p", "v")),
        ("cute.arch.atomic_add(p, v)", ("p", "v")),
        ("cute.arch.atomic_cas(p, cmp=a, val=b, sem='relaxed')", None),
    ],
    ids=["keyword", "positional", "compare_and_swap"],
)
def test_the_atomic_operand_parser_reads_the_keyword_form(
    call: str, expected: tuple[str, str] | None
) -> None:
    node = ast.parse(call, mode="eval").body
    assert isinstance(node, ast.Call)
    parts = lanes._atomic_pointer_and_value(node)
    if expected is None:
        assert parts is None
    else:
        assert parts is not None
        assert tuple(ast.unparse(part) for part in parts) == expected


def test_a_second_marker_input_produced_between_the_markers_joins_the_prefix() -> None:
    """Two stored reductions of one accumulator under dynamic shapes render the second's masked input between the markers; it reads prefix values only, so it moves before the first marker and both shares are totalled."""
    body = _restored_rows_body(_STORED_TOTAL).replace(
        _SUM_MARKER_LINE,
        _SUM_MARKER_LINE
        + _STORED_TOTAL
        + "_mask_to2 = cutlass.Float32(row_sums) if mask else "
        "cutlass.Float32(float('-inf'))\n"
        + f"amax_3 = cutlass.Float32({_max_marker('_mask_to2')})\n",
        1,
    )
    body = body[: -len(_STORED_TOTAL)] + _STORED_AMAX
    code = _split(body, _ROW_SUMS_RENAMES, _TWO_TENSORS_DISJOINT)
    (lane_body,) = _lane_passes(code)
    assert (
        code.index("sum_2_lane_total = cutlass.Float32(0)")
        < code.index("amax_3_lane_total = cutlass.Float32(float('-inf'))")
        < code.index("for lane in range(2):")
    )
    assert lane_body.index("_mask_to2 = ") < lane_body.index(
        "_cute_grouped_reduce_shared_two_stage(_mask_to,"
    )
    assert "tot.iterator" not in lane_body and "tot2.iterator" not in lane_body
    after = code[code.index("amax_3 = cutlass.Float32(amax_3_lane_total)") :]
    assert after.count("% 256 < 1") == 2
    assert after.index("tot.iterator") < after.index("tot2.iterator")


def test_a_second_marker_input_reading_the_first_result_stays_declined() -> None:
    body = _restored_rows_body(_STORED_TOTAL).replace(
        _SUM_MARKER_LINE,
        _SUM_MARKER_LINE
        + "_mask_to2 = _mask_to * sum_2\n"
        + f"sum_3 = cutlass.Float32({_masked_marker('_mask_to2')})\n",
        1,
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="no proved complete per-lane restore"
    ):
        _split(
            body + _STORED_AMAX.replace("amax_3", "sum_3"),
            _ROW_SUMS_RENAMES,
            _TWO_TENSORS_DISJOINT,
        )
