"""Special-purpose root emitters of the CuTe backend (registry) and the
general-purpose emitter helpers they compose.

``ROOT_EMITTERS`` is the ordered tuple ``CuteBackend.root_emitters`` returns:
``GenerateAST.visit_For`` tries each entry in turn and the first that owns the
root emits it; none owning it leaves the root to the ordinary lowering.  The
order is a priority order (the computed-fragment owner is tried early when the
root asks for fragments and again last) and is pinned by tests.

Planning stays where it is today: the planners run in ``CuteBackend.pre_codegen``
(and the flash / warp-MMA detectors in ``create_loop_strategy``) and record each
family's plan in its ``cute_state`` slot; an emitter's ``plan`` reads that slot.
The codegen entry of each family is resolved on its module at call time, so
tests that patch the entry on the module keep intercepting it.

The sibling modules of this package are not root emitters but the
general-purpose helpers the emitters are written over: ``memory_access`` (the
index / address / region / mask builders of the load, store, atomic and scan
lowerings), ``lane_packets`` (the register-tile vector packet forms) and
``persistent_item_loop`` (the text layout of a warp role's work-item loop
that the flash scaffolds render).

Role: special-purpose emitter (the registry) over general-purpose emitters
(the helpers); section 3 of ``../README.md``.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING

from .... import exc
from ...emitter_protocols import RootEmitter
from ..state import ROOT_PLAN_SLOTS

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...generate_ast import GenerateAST
    from ..device_state import CuteDeviceFunctionState


def _entry(module: str, name: str) -> Callable[..., bool]:
    """The codegen entry ``name`` of ``helion._compiler.cute.<module>``, looked
    up at call time (``unittest.mock.patch`` on the module intercepts it)."""
    return getattr(importlib.import_module(f"helion._compiler.cute.{module}"), name)


def _late_validation(family: str) -> Callable[[], Exception]:
    return lambda: exc.BackendUnsupported("cute", f"{family} failed late validation")


@dataclasses.dataclass(frozen=True)
class PlannedRootEmitter(RootEmitter):
    """A family whose planner left its plan in ``cute_state.root_plans.<slot>``.

    ``failure`` builds the exception a refused root raises; ``clears_slot``
    says whether the refusal resets the slot first (most families do, so the
    ordinary lowering of a regenerated kernel is not confused by a stale
    plan).  ``passes_plan`` distinguishes entries called as ``entry(codegen,
    plan)`` from those reading the slot themselves.
    """

    name: str
    slot: str
    module: str
    entry: str
    failure: Callable[[], Exception]
    passes_plan: bool = False
    clears_slot: bool = True

    def __post_init__(self) -> None:
        if self.slot not in ROOT_PLAN_SLOTS:
            raise ValueError(f"{self.name}: {self.slot!r} is not a CuteRootPlans slot")

    def plan(self, codegen: GenerateAST) -> object | None:
        return getattr(codegen.device_function.cute_state.root_plans, self.slot)

    def emit(self, codegen: GenerateAST, plan: object) -> bool:
        entry = _entry(self.module, self.entry)
        if self.passes_plan:
            return entry(codegen, plan)
        return entry(codegen)

    def decline(self, codegen: GenerateAST, plan: object) -> None:
        if self.clears_slot:
            setattr(codegen.device_function.cute_state.root_plans, self.slot, None)
        raise self.failure()


_FLASH_BWD = "bwd"
_FLASH_GATED = "gated"
_FLASH_FWD = "fwd"


def _clear_attention_flash_state(cute_state: CuteDeviceFunctionState) -> None:
    cute_state.flash.block_ids = None
    cute_state.flash.score_plan = None
    cute_state.flash.gated_match = None
    cute_state.flash.flex_match = None
    cute_state.flash.threads = 128


@dataclasses.dataclass(frozen=True)
class AttentionFlashRootEmitter(RootEmitter):
    """The flash attention families (backward, gated, forward), detected while
    the tile strategy is built and recorded as ``attention_flash_*`` state."""

    name: str = "attention_flash"

    def plan(self, codegen: GenerateAST) -> object | None:
        cute_state = codegen.device_function.cute_state
        if cute_state.flash.bwd_block_ids is not None:
            return _FLASH_BWD
        if cute_state.flash.block_ids is None:
            return None
        if cute_state.flash.gated_match is not None:
            return _FLASH_GATED
        return _FLASH_FWD

    def emit(self, codegen: GenerateAST, plan: object) -> bool:
        if plan == _FLASH_BWD:
            return _entry("cute_flash_bwd", "codegen_attention_flash_bwd")(codegen)
        if plan == _FLASH_GATED:
            return _entry("cute_flash_gated", "codegen_gated_attention_flash")(codegen)
        return _entry("cute_flash", "codegen_attention_flash")(codegen)

    def decline(self, codegen: GenerateAST, plan: object) -> None:
        cute_state = codegen.device_function.cute_state
        if plan == _FLASH_BWD:
            cute_state.flash.bwd_block_ids = None
            cute_state.flash.bwd_match = None
            raise exc.BackendUnsupported(
                "cute", "flash attention backward failed late validation"
            )
        _clear_attention_flash_state(cute_state)
        if plan == _FLASH_GATED:
            raise exc.BackendUnsupported(
                "cute", "gated attention failed late validation"
            )
        raise exc.BackendUnsupported("cute", "flash attention failed late validation")


_FRAGMENT_SCHEDULE = "computed_fragment"


@dataclasses.dataclass(frozen=True)
class ComputedFragmentRootEmitter(RootEmitter):
    """The computed-fragment owner: it has no plan slot, decides ownership while
    emitting and declines by falling through to the next owner."""

    name: str = "computed_fragment"

    def plan(self, codegen: GenerateAST) -> object | None:
        return _FRAGMENT_SCHEDULE

    def emit(self, codegen: GenerateAST, plan: object) -> bool:
        return _entry("computed_fragment", "codegen_computed_fragment_root")(codegen)

    def decline(self, codegen: GenerateAST, plan: object) -> None:
        return None


def _requests_fragments(codegen: GenerateAST) -> bool:
    root = codegen.current_root_graph_info
    assert root is not None
    return _entry("computed_fragment", "prefer_computed_fragment_root")(
        codegen, root.graph_id
    )


@dataclasses.dataclass(frozen=True)
class GatedRootEmitter(RootEmitter):
    """Tries ``inner`` only when ``gate`` holds: the early computed-fragment
    slot, where the owner runs before the planned families when the root asks
    for fragments (it is tried again, ungated, last)."""

    name: str
    inner: RootEmitter
    gate: Callable[[GenerateAST], bool]

    def plan(self, codegen: GenerateAST) -> object | None:
        return self.inner.plan(codegen)

    def emit(self, codegen: GenerateAST, plan: object) -> bool:
        return self.inner.emit(codegen, plan)

    def decline(self, codegen: GenerateAST, plan: object) -> None:
        self.inner.decline(codegen, plan)

    def try_emit(self, codegen: GenerateAST) -> bool:
        if not self.gate(codegen):
            return False
        return self.inner.try_emit(codegen)


REGISTER_SUBBLOCKS_ROOT = PlannedRootEmitter(
    "register_subblocks",
    slot="register_subblocks_plan",
    module="cute_register_subblocks",
    entry="codegen_register_subblocks",
    failure=_late_validation("register sub-block family"),
)
RADIX_SELECTION_ROOT = PlannedRootEmitter(
    "radix_selection",
    slot="radix_selection_plan",
    module="selection.radix_selection_codegen",
    entry="codegen_radix_selection_root",
    failure=lambda: exc.InvalidConfig(
        "radix selection failed its final row/grid proof"
    ),
    passes_plan=True,
    clears_slot=False,
)
COMPUTED_FRAGMENT_ROOT = ComputedFragmentRootEmitter()
BLOCK_SCALED_ROOT = PlannedRootEmitter(
    "block_scaled",
    slot="block_scaled_plan",
    module="block_scaled_mma",
    entry="codegen_block_scaled",
    failure=_late_validation("block scaling"),
    clears_slot=False,
)
TOPK_ROOT = PlannedRootEmitter(
    "topk",
    slot="topk_plan",
    module="selection.topk_codegen",
    entry="codegen_topk_root",
    failure=_late_validation("topk"),
    passes_plan=True,
    clears_slot=False,
)
REGISTER_REGION_ROOT = PlannedRootEmitter(
    "register_region",
    slot="register_region_plan",
    module="register_region",
    entry="codegen_register_region",
    failure=_late_validation("register region"),
    passes_plan=True,
    clears_slot=False,
)
CHUNK_PREPARE_ROOT = PlannedRootEmitter(
    "chunk_prepare",
    slot="chunk_prepare_plan",
    module="chunk_prepare",
    entry="codegen_chunk_prepare",
    failure=_late_validation("chunk prepare"),
)
CHUNK_RECURRENCE_ROOT = PlannedRootEmitter(
    "chunk_recurrence",
    slot="chunk_recurrence_plan",
    module="chunk_recurrence",
    entry="codegen_chunk_recurrence",
    failure=_late_validation("chunk recurrence"),
)
GDN_RECURRENCE_ROOT = PlannedRootEmitter(
    "gdn_recurrence",
    slot="gdn_recurrence_plan",
    module="gdn_recurrence",
    entry="codegen_gdn_recurrence",
    failure=_late_validation("gdn recurrence"),
)
MMA_DAG_ROOT = PlannedRootEmitter(
    "mma_dag",
    slot="mma_dag_plan",
    module="cute_mma_dag_codegen",
    entry="codegen_mma_dag",
    failure=_late_validation("cute_mma_dag"),
)
SINGLE_TOKEN_RANK1_ROOT = PlannedRootEmitter(
    "single_token_rank1",
    slot="single_token_rank1_plan",
    module="single_token_rank1_recurrence",
    entry="codegen_single_token_rank1_recurrence",
    failure=_late_validation("single-token rank-1 recurrence"),
)
SPLIT_SINGLE_TOKEN_RANK1_ROOT = PlannedRootEmitter(
    "split_single_token_rank1",
    slot="split_single_token_rank1_plan",
    module="split_single_token_rank1_recurrence",
    entry="codegen_split_single_token_rank1_recurrence",
    failure=_late_validation("split single-token rank-1 recurrence"),
    passes_plan=True,
)
FIXED_TOKEN_RANK1_ROOT = PlannedRootEmitter(
    "fixed_token_rank1",
    slot="fixed_token_rank1_plan",
    module="fixed_token_rank1_recurrence",
    entry="codegen_fixed_token_rank1_recurrence",
    failure=_late_validation("fixed-token rank-1 recurrence"),
)
WARP_MMA_GEMM_ROOT = PlannedRootEmitter(
    "warp_mma_gemm",
    slot="warp_mma_gemm_plan",
    module="cute_warp_mma_gemm",
    entry="codegen_warp_mma_gemm",
    failure=_late_validation("warp_mma GEMM family"),
)
ATTENTION_FLASH_ROOT = AttentionFlashRootEmitter()

ROOT_EMITTERS: tuple[RootEmitter, ...] = (
    REGISTER_SUBBLOCKS_ROOT,
    RADIX_SELECTION_ROOT,
    GatedRootEmitter(
        "requested_computed_fragment", COMPUTED_FRAGMENT_ROOT, _requests_fragments
    ),
    BLOCK_SCALED_ROOT,
    TOPK_ROOT,
    REGISTER_REGION_ROOT,
    CHUNK_PREPARE_ROOT,
    CHUNK_RECURRENCE_ROOT,
    GDN_RECURRENCE_ROOT,
    MMA_DAG_ROOT,
    SINGLE_TOKEN_RANK1_ROOT,
    SPLIT_SINGLE_TOKEN_RANK1_ROOT,
    FIXED_TOKEN_RANK1_ROOT,
    WARP_MMA_GEMM_ROOT,
    ATTENTION_FLASH_ROOT,
    COMPUTED_FRAGMENT_ROOT,
)
