"""The structural launch-shape fact (``helion/_compiler/cute/launch_shape.py``):
the plan-owner precedence, the evidence predicate over the SIMT candidates and
the check's record / warn / raise behavior, on hand-built inputs."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import launch_shape
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.launch_shape import LaunchShape
from helion._compiler.cute.launch_shape import LaunchShapeCheckMode
from helion._compiler.cute.launch_shape import LaunchShapeFlags
from helion._compiler.cute.launch_shape import SimtLaunchCandidates
from helion._compiler.cute.launch_shape import check_launch_shape
from helion._compiler.cute.launch_shape import plan_owned_launch
from helion._compiler.cute.launch_shape import simt_launch_candidates
from helion._compiler.cute.launch_shape import structural_launch_shape

if TYPE_CHECKING:
    from pathlib import Path

# Backend-independent (runs under every HELION_BACKEND, like a non-cute test);
# the conftest gate over test_cute_* files leaves it alone.

# (owner, the ``namespace.field`` state entries that select it, the launch) in
# the launcher's precedence order: the first row wins over every later one.
_OWNED_LAUNCHES: list[tuple[str, dict[str, object], tuple[int, int, int]]] = [
    ("topk", {"root_plans.topk_plan": SimpleNamespace(threads=64)}, (64, 1, 1)),
    (
        "register_region",
        {"root_plans.register_region_plan": SimpleNamespace(threads=96)},
        (96, 1, 1),
    ),
    (
        "direct_affine",
        {"root_plans.direct_affine_plan": SimpleNamespace(cta_shape=(128, 2, 1))},
        (128, 2, 1),
    ),
    (
        "register_chain",
        {"lanes.collective_register_chain_block_dims": (256, 1, 1)},
        (256, 1, 1),
    ),
    ("register_chain", {"lanes.owned_root_block_dims": (32, 4, 1)}, (32, 4, 1)),
    (
        "single_token_rank1",
        {"root_plans.single_token_rank1_plan": SimpleNamespace(threads=256)},
        (256, 1, 1),
    ),
    (
        "split_single_token_rank1",
        {"root_plans.split_single_token_rank1_plan": object()},
        (32, 1, 1),
    ),
    (
        "fixed_token_rank1",
        {
            "root_plans.fixed_token_rank1_plan": SimpleNamespace(
                key_split=4, value_split=2
            )
        },
        (256, 1, 1),
    ),
    ("chunk_prepare", {"root_plans.chunk_prepare_plan": object()}, (128, 1, 1)),
    ("block_scaled", {"root_plans.block_scaled_plan": object()}, (192, 1, 1)),
    (
        "chunk_recurrence",
        {"root_plans.chunk_recurrence_plan": SimpleNamespace(threads=64)},
        (64, 1, 1),
    ),
    (
        "gdn_recurrence",
        {"root_plans.gdn_recurrence_plan": SimpleNamespace(threads=128)},
        (128, 1, 1),
    ),
    (
        "warp_mma_gemm",
        {"root_plans.warp_mma_gemm_plan": SimpleNamespace(threads=256)},
        (256, 1, 1),
    ),
    (
        "attention_flash",
        {"flash.block_ids": [0, 1], "flash.threads": 256},
        (256, 1, 1),
    ),
    ("attention_flash", {"flash.bwd_block_ids": [2]}, (128, 1, 1)),
]


def _state(fields: dict[str, object] | None = None) -> CuteDeviceFunctionState:
    """A device state with no whole-root plan, overridden by the
    ``namespace.field`` entries of ``fields``."""
    state = CuteDeviceFunctionState()
    for path, value in (fields or {}).items():
        namespace, _, name = path.partition(".")
        setattr(getattr(state, namespace), name, value)
    return state


def test_simt_body_has_no_plan_owned_launch() -> None:
    assert plan_owned_launch(_state()) is None


@pytest.mark.parametrize(("owner", "fields", "dims"), _OWNED_LAUNCHES)
def test_each_plan_owns_its_launch(
    owner: str, fields: dict[str, object], dims: tuple[int, int, int]
) -> None:
    assert plan_owned_launch(_state(fields)) == LaunchShape(dims, owner)


def test_plan_owner_precedence_is_the_launcher_order() -> None:
    for index, (owner, _fields, dims) in enumerate(_OWNED_LAUNCHES):
        # Every plan from this row on is set: this row's owner wins.
        fields: dict[str, object] = {}
        for _owner, later_fields, _dims in _OWNED_LAUNCHES[index:]:
            fields.update(later_fields)
        assert plan_owned_launch(_state(fields)) == LaunchShape(dims, owner)


def test_direct_affine_launch_is_checked_against_the_block_limits() -> None:
    env = SimpleNamespace(backend=SimpleNamespace(name="cute"))
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        pytest.raises(exc.BackendUnsupported, match="thread block axis 0"),
    ):
        plan_owned_launch(
            _state(
                {
                    "root_plans.direct_affine_plan": SimpleNamespace(
                        cta_shape=(2048, 1, 1)
                    )
                }
            )
        )


def _flags(**on: bool) -> LaunchShapeFlags:
    return LaunchShapeFlags(
        **{
            field.name: on.pop(field.name, False)
            for field in dataclasses.fields(LaunchShapeFlags)
        }
    )


_BOTH = SimtLaunchCandidates(producers=(128, 1, 1), strategies=(128, 2, 1))
_PRODUCERS_ONLY = SimtLaunchCandidates(producers=(64, 2, 1), strategies=None)
_AGREEING = SimtLaunchCandidates(producers=(128, 2, 1), strategies=(128, 2, 1))
_NONE = SimtLaunchCandidates(producers=None, strategies=None)


@pytest.mark.parametrize(
    ("candidates", "flags", "expected"),
    [
        # The strategies' proven static shape is the fact.
        (_BOTH, _flags(), LaunchShape((128, 2, 1), "strategies")),
        (_BOTH, _flags(reduction=True), LaunchShape((128, 2, 1), "strategies")),
        (_BOTH, _flags(mma=True), LaunchShape((128, 2, 1), "strategies")),
        (_BOTH, _flags(pointwise_fact=True), LaunchShape((128, 2, 1), "strategies")),
        # The four facts the text recovery still owns leave the shape unknown.
        (_BOTH, _flags(multi_phase=True), None),
        (_BOTH, _flags(tcgen05=True), None),
        (_BOTH, _flags(synthetic_axes=True), None),
        (_BOTH, _flags(symbolic_axes=True), None),
        # Nested device loops: known only when the producers agree.
        (_BOTH, _flags(nested_device_loops=True), None),
        (_PRODUCERS_ONLY, _flags(nested_device_loops=True), None),
        (
            _AGREEING,
            _flags(nested_device_loops=True),
            LaunchShape((128, 2, 1), "strategies"),
        ),
        # No proven static shape: the producers are the fact for a pointwise
        # body only.
        (_PRODUCERS_ONLY, _flags(), LaunchShape((64, 2, 1), "producers")),
        (
            _PRODUCERS_ONLY,
            _flags(pointwise_fact=True),
            LaunchShape((64, 2, 1), "producers"),
        ),
        (_PRODUCERS_ONLY, _flags(reduction=True), None),
        (_PRODUCERS_ONLY, _flags(mma=True), None),
        (_NONE, _flags(), None),
    ],
)
def test_structural_launch_shape(
    candidates: SimtLaunchCandidates,
    flags: LaunchShapeFlags,
    expected: LaunchShape | None,
) -> None:
    assert structural_launch_shape(candidates, flags) == expected


def _device_function(
    body: str,
    extents: dict[str, tuple[int, int]],
    *,
    thread_block_dims: list[int] | None = None,
    symbolic_thread_axes: dict[int, str] | None = None,
) -> SimpleNamespace:
    """The parts of a ``DeviceFunction`` the check reads: the rendered
    statements, the producers' registered extents, the tile strategy's shape."""
    return SimpleNamespace(
        preamble=[],
        body=ast.parse(body).body,
        cute_state=SimpleNamespace(
            lanes=SimpleNamespace(
                launch_thread_extents=extents, launch_shape_check=None
            )
        ),
        tile_strategy=SimpleNamespace(
            symbolic_thread_axes=lambda: symbolic_thread_axes or {},
            synthetic_thread_axes=dict,
            thread_block_dims=lambda: thread_block_dims,
        ),
    )


def test_simt_launch_candidates_read_live_producers_and_the_static_shape() -> None:
    fn = _device_function(
        "indices_0 = offset_0 + tid\nindices_1 = offset_1 + tid\n",
        {"indices_0": (0, 128), "indices_1": (1, 2), "dead": (2, 64)},
        thread_block_dims=[128, 2, 1],
    )
    assert simt_launch_candidates(fn) == SimtLaunchCandidates(
        producers=(128, 2, 1), strategies=(128, 2, 1)
    )
    # No surviving producer, and a strategy shape an argument sizes, are
    # unknown candidates.
    fn = _device_function(
        "x = 1\n",
        {"indices_0": (0, 128)},
        thread_block_dims=[128, 1, 1],
        symbolic_thread_axes={0: "_BLOCK_SIZE_0"},
    )
    assert simt_launch_candidates(fn) == SimtLaunchCandidates(
        producers=None, strategies=None
    )


@contextlib.contextmanager
def _check_mode(strict: bool, log: Path | None = None):
    mode = LaunchShapeCheckMode(
        strict=strict, log_path=None if log is None else str(log)
    )
    compile_env = SimpleNamespace(settings=SimpleNamespace(ignore_warnings=[]))
    with (
        patch.object(launch_shape, "check_mode", return_value=mode),
        patch.object(CompileEnvironment, "current", return_value=compile_env),
    ):
        yield


def test_check_mode_is_read_from_the_environment_once(monkeypatch) -> None:
    monkeypatch.setenv("HELION_CUTE_LAUNCH_SHAPE_CHECK", "1")
    monkeypatch.setenv("HELION_CUTE_LAUNCH_SHAPE_LOG", "/tmp/launch_shape.jsonl")
    launch_shape.check_mode.cache_clear()
    try:
        assert launch_shape.check_mode() == LaunchShapeCheckMode(
            strict=True, log_path="/tmp/launch_shape.jsonl"
        )
        # The switches are read once: a later change is not seen...
        monkeypatch.setenv("HELION_CUTE_LAUNCH_SHAPE_CHECK", "0")
        assert launch_shape.check_mode().strict is True
        # ... and the log is a strict-mode file only.
        launch_shape.check_mode.cache_clear()
        assert launch_shape.check_mode() == LaunchShapeCheckMode(
            strict=False, log_path=None
        )
    finally:
        launch_shape.check_mode.cache_clear()


def test_check_records_an_agreeing_fact_and_logs_it(tmp_path: Path) -> None:
    fn = _device_function(
        "indices_0 = offset_0 + tid\n",
        {"indices_0": (0, 128)},
        thread_block_dims=[128, 1, 1],
    )
    log = tmp_path / "launch_shape.jsonl"
    stderr = io.StringIO()
    with _check_mode(strict=True, log=log), contextlib.redirect_stderr(stderr):
        check_launch_shape(fn, (128, 1, 1), flags=_flags(pointwise_fact=True))
    record = fn.cute_state.lanes.launch_shape_check
    assert record == {
        "launched": [128, 1, 1],
        "structural": [128, 1, 1],
        "owner": "strategies",
        "agrees": True,
        "candidates": {"producers": [128, 1, 1], "strategies": [128, 1, 1]},
        "flags": dataclasses.asdict(_flags(pointwise_fact=True)),
    }
    assert [json.loads(line) for line in log.read_text().splitlines()] == [record]
    assert stderr.getvalue() == ""


def test_check_warns_on_a_mismatch_by_default() -> None:
    fn = _device_function(
        "indices_0 = offset_0 + tid\n",
        {"indices_0": (0, 128)},
        thread_block_dims=[128, 1, 1],
    )
    stderr = io.StringIO()
    with _check_mode(strict=False), contextlib.redirect_stderr(stderr):
        check_launch_shape(fn, (64, 1, 1), flags=_flags())
    assert fn.cute_state.lanes.launch_shape_check["agrees"] is False
    assert "WARNING[LaunchShapeMismatch]" in stderr.getvalue()
    assert "(128, 1, 1)" in stderr.getvalue()
    assert "(64, 1, 1)" in stderr.getvalue()


def test_check_raises_on_a_mismatch_in_strict_mode() -> None:
    fn = _device_function(
        "indices_0 = offset_0 + tid\n",
        {"indices_0": (0, 128)},
        thread_block_dims=[128, 1, 1],
    )
    with (
        _check_mode(strict=True),
        pytest.raises(exc.LaunchShapeMismatch, match=r"\(128, 1, 1\)"),
    ):
        check_launch_shape(fn, (64, 1, 1), flags=_flags())
    # The record is kept for the failed render too.
    assert fn.cute_state.lanes.launch_shape_check["agrees"] is False


def test_check_is_silent_when_the_fact_is_unknown() -> None:
    fn = _device_function(
        "indices_0 = offset_0 + tid\n",
        {"indices_0": (0, 128)},
        thread_block_dims=[128, 1, 1],
    )
    stderr = io.StringIO()
    with _check_mode(strict=True), contextlib.redirect_stderr(stderr):
        check_launch_shape(fn, (64, 1, 1), flags=_flags(tcgen05=True))
    record = fn.cute_state.lanes.launch_shape_check
    assert record["structural"] is None
    assert record["owner"] is None
    assert record["agrees"] is None
    assert stderr.getvalue() == ""
