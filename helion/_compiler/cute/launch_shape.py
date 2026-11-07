"""The structural launch-shape fact of a CuTe kernel (check mode).

A kernel's launch block is a structural fact: the whole-root plan that owns
the body says how many threads it runs (``plan_owned_launch``), and a SIMT
body's thread extents are what its strategies and grid index producers
registered (``simt_launch_candidates``).  ``CuteBackend.launcher_keyword_args``
still recovers the SIMT launch from the rendered kernel text -- its regexes
are the source of truth -- and runs this module in CHECK MODE:
``check_launch_shape`` compares the structural candidates with the launch it
derived on every render, records the comparison on
``cute_state.launch_shape_check`` and, when the structural fact is known
(``structural_launch_shape``) and disagrees, warns
(``exc.LaunchShapeMismatch``) or, under ``HELION_CUTE_LAUNCH_SHAPE_CHECK=1``,
fails the render (the test suites run with it set, ``test/conftest.py``) and
appends every record to the file named by ``HELION_CUTE_LAUNCH_SHAPE_LOG``
(``LaunchShapeCheckMode``, read from the environment once per process).
Deleting the text recovery is the step after this check has run green across
the suites.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
from typing import TYPE_CHECKING
from typing import Sequence

from ... import exc
from .thread_budget import check_thread_block_dims

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from .device_state import CuteDeviceFunctionState

Dims = tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class LaunchShapeCheckMode:
    """The check's switches: ``strict`` (``HELION_CUTE_LAUNCH_SHAPE_CHECK=1``)
    fails a render on a disagreement instead of warning, and ``log_path``
    (``HELION_CUTE_LAUNCH_SHAPE_LOG``, honored in strict mode only) names the
    file every record is appended to."""

    strict: bool
    log_path: str | None


@functools.cache
def check_mode() -> LaunchShapeCheckMode:
    """The process's check mode, read from the environment on the first
    render (after ``test/conftest.py`` turned the strict mode on)."""
    strict = os.environ.get("HELION_CUTE_LAUNCH_SHAPE_CHECK") == "1"
    log_path = os.environ.get("HELION_CUTE_LAUNCH_SHAPE_LOG") if strict else None
    return LaunchShapeCheckMode(strict=strict, log_path=log_path or None)


@dataclasses.dataclass(frozen=True)
class LaunchShape:
    """The launch block a kernel's structural facts determine.

    ``owner`` names the fact that determines it: the whole-root plan that owns
    the body and therefore its launch (``plan_owned_launch``), or for a SIMT
    body the candidate the evidence admits (``"strategies"`` or
    ``"producers"``, ``structural_launch_shape``).
    """

    dims: Dims
    owner: str


@dataclasses.dataclass(frozen=True)
class SimtLaunchCandidates:
    """The structural launch candidates of a SIMT body.

    ``producers``: the live extents every grid index producer registered
    (``cute_state.launch_thread_extents``, ungated); ``strategies``: the
    strategies' proven static shape (``_exact_thread_block_dims``).  Either is
    ``None`` when its facts do not determine a shape.
    """

    producers: Dims | None
    strategies: Dims | None


@dataclasses.dataclass(frozen=True)
class LaunchShapeFlags:
    """What the launcher knows about the body beyond the candidates: the
    evidence that decides which candidate, if any, is the structural fact."""

    multi_phase: bool
    nested_device_loops: bool
    mma: bool
    tcgen05: bool
    reduction: bool
    synthetic_axes: bool
    symbolic_axes: bool
    pointwise_fact: bool


def plan_owned_launch(state: CuteDeviceFunctionState) -> LaunchShape | None:
    """The launch of the whole-root plan owning the body, or ``None`` for a
    SIMT body.  Plans are consulted in the launcher's precedence order."""
    if state.root_plans.topk_plan is not None:
        return LaunchShape((state.root_plans.topk_plan.threads, 1, 1), "topk")
    if state.root_plans.register_region_plan is not None:
        return LaunchShape(
            (state.root_plans.register_region_plan.threads, 1, 1), "register_region"
        )
    if state.root_plans.direct_affine_plan is not None:
        x, y, z = state.root_plans.direct_affine_plan.cta_shape
        check_thread_block_dims(
            (x, y, z), context=str(state.root_plans.direct_affine_plan.cta_shape)
        )
        return LaunchShape((x, y, z), "direct_affine")
    register_chain_block_dims = (
        state.lanes.collective_register_chain_block_dims
        or state.lanes.owned_root_block_dims
    )
    if register_chain_block_dims is not None:
        return LaunchShape(register_chain_block_dims, "register_chain")
    # The single-token rank-1 path owns the complete physical body.  The
    # original B1 schedule uses 256 threads while its batched schedule uses
    # one warp; the structural plan proves which topology was emitted.
    if state.root_plans.single_token_rank1_plan is not None:
        return LaunchShape(
            (state.root_plans.single_token_rank1_plan.threads, 1, 1),
            "single_token_rank1",
        )
    # The split-input T=1 path owns one 32-thread warp per 16 output rows.
    if state.root_plans.split_single_token_rank1_plan is not None:
        return LaunchShape((32, 1, 1), "split_single_token_rank1")
    # The fixed-token grouped rank-1 path owns one physical CTA per
    # sequence/head/value split and replaces the complete device body.
    if state.root_plans.fixed_token_rank1_plan is not None:
        plan = state.root_plans.fixed_token_rank1_plan
        return LaunchShape(
            (128 * plan.key_split // plan.value_split, 1, 1), "fixed_token_rank1"
        )
    # The exact chunk-prepare and chunk-recurrence lowerings own their physical
    # launch topology rather than the carrier's logical tile axes.
    if state.root_plans.chunk_prepare_plan is not None:
        return LaunchShape((128, 1, 1), "chunk_prepare")
    if state.root_plans.block_scaled_plan is not None:
        return LaunchShape((192, 1, 1), "block_scaled")
    if state.root_plans.chunk_recurrence_plan is not None:
        return LaunchShape(
            (state.root_plans.chunk_recurrence_plan.threads, 1, 1), "chunk_recurrence"
        )
    if state.root_plans.gdn_recurrence_plan is not None:
        return LaunchShape(
            (state.root_plans.gdn_recurrence_plan.threads, 1, 1), "gdn_recurrence"
        )
    # The register-MMA GEMM family owns the whole device body and runs
    # one CTA of ``32 * warps`` threads per output tile.
    if state.root_plans.warp_mma_gemm_plan is not None:
        return LaunchShape(
            (state.root_plans.warp_mma_gemm_plan.threads, 1, 1), "warp_mma_gemm"
        )
    # Fused tcgen05 flash-attention: 128 threads (single-warpgroup Stage-3)
    # or 256 threads (Stage-4 warp-spec producer/consumer split). The custom
    # flash codegen owns the whole device body, so the SIMT thread-axis
    # heuristics do not apply.
    if state.flash.block_ids is not None or state.flash.bwd_block_ids is not None:
        return LaunchShape((state.flash.threads, 1, 1), "attention_flash")
    return None


def live_grid_thread_extents(
    statements: Sequence[ast.AST], extents: dict[str, tuple[int, int]]
) -> dict[int, int]:
    """Per-axis thread extents of the grid index producers in ``extents``
    (name -> ``(axis, extent)``) whose assignments survive in ``statements``."""
    live: dict[int, int] = {}
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in extents:
                    axis, extent = extents[target.id]
                    live[axis] = max(live.get(axis, 1), extent)
    return live


def live_grid_thread_dims(
    statements: Sequence[ast.AST], extents: dict[str, tuple[int, int]]
) -> list[int]:
    live = live_grid_thread_extents(statements, extents)
    return [live.get(axis, 1) for axis in range(3)]


def simt_launch_candidates(device_function: DeviceFunction) -> SimtLaunchCandidates:
    """The structural launch candidates of a SIMT body (``SimtLaunchCandidates``)."""
    from ..device_function import _exact_thread_block_dims

    live = live_grid_thread_extents(
        [*device_function.preamble, *device_function.body],
        device_function.cute_state.lanes.launch_thread_extents,
    )
    return SimtLaunchCandidates(
        producers=(live.get(0, 1), live.get(1, 1), live.get(2, 1)) if live else None,
        strategies=_exact_thread_block_dims(device_function.tile_strategy),
    )


def structural_launch_shape(
    candidates: SimtLaunchCandidates, flags: LaunchShapeFlags
) -> LaunchShape | None:
    """The launch block the structural facts of a SIMT body determine, or
    ``None`` when they do not determine one.

    The strategies' proven static shape is the fact for a body without a
    tcgen05 role launch, a synthetic free-``hl.arange`` axis, an argument-sized
    axis or ``hl.barrier()`` phases (those launches are shaped by other facts
    the text recovery still owns).  With nested device loops the fact is known
    only when the grid producers' live extents agree with it: a device-loop
    axis whose index the final kernel never reads is dropped from the launch,
    which only the surviving producers show.  Without a proven static shape
    the producers' live extents are the fact for a pointwise body only (a
    reduction strategy adds threads they do not see).  The full-suite log of
    the check is the evidence for this predicate; the text recovery stays the
    source of truth until the check has run green across the suites.
    """
    if (
        flags.multi_phase
        or flags.tcgen05
        or flags.synthetic_axes
        or flags.symbolic_axes
    ):
        return None
    strategies = candidates.strategies
    producers = candidates.producers
    if flags.nested_device_loops:
        if strategies is not None and producers == strategies:
            return LaunchShape(strategies, "strategies")
        return None
    if strategies is not None:
        return LaunchShape(strategies, "strategies")
    # No proven static shape: a pointwise body (no reduction strategy, no MMA)
    # launches its grid producers' live extents.
    if producers is None or flags.reduction or flags.mma:
        return None
    return LaunchShape(producers, "producers")


def check_launch_shape(
    device_function: DeviceFunction,
    launched: Dims,
    *,
    flags: LaunchShapeFlags,
) -> None:
    """Compare the structural candidates with the text-derived ``launched``
    block of a SIMT body (CHECK MODE; see the module docstring)."""
    candidates = simt_launch_candidates(device_function)
    structural = structural_launch_shape(candidates, flags)
    record: dict[str, object] = {
        "launched": list(launched),
        "structural": None if structural is None else list(structural.dims),
        "owner": None if structural is None else structural.owner,
        "agrees": None if structural is None else structural.dims == launched,
        "candidates": {
            name: None if dims is None else list(dims)
            for name, dims in dataclasses.asdict(candidates).items()
        },
        "flags": dataclasses.asdict(flags),
    }
    device_function.cute_state.lanes.launch_shape_check = record
    mode = check_mode()
    if mode.log_path is not None:
        with open(mode.log_path, "a") as log:
            log.write(json.dumps(record) + "\n")
    if structural is not None and structural.dims != launched:
        # ``compile_environment`` imports this package's backend while it
        # initializes, so the warning channel is imported at call time.
        from ..compile_environment import warning

        mismatch = exc.LaunchShapeMismatch(structural.dims, launched)
        if mode.strict:
            raise mismatch
        warning(mismatch)
