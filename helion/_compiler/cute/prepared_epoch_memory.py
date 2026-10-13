"""Bound original epoch boundary transports, not arbitrary source callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .chained_completed_store import _records
from .prepared_tcgen_binding import _native_source

if TYPE_CHECKING:
    from .prepared_epoch_product import DescriptorProductBinding
    from .prepared_tcgen_binding import PreparedProjectionHost


class EpochMemoryOperation(Enum):
    INPUTS = "inputs"
    IMPORT_STATE = "import_state"
    EXPORT_STATE = "export_state"
    OUTPUT_FULL = "output_full"
    OUTPUT_TAIL = "output_tail"


@dataclass(frozen=True)
class EpochMemoryAction:
    owner: PreparedProjectionHost
    operation: EpochMemoryOperation
    product: DescriptorProductBinding | None = None
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_facts", self._current())

    def _current(self) -> object:
        from . import chunk_recurrence_sm100 as original
        from .chunk_recurrence import _store_ref
        from .chunk_recurrence import _tensor_refs
        from .chunk_recurrence import validate_epoch_match

        self.owner.check()
        validate_epoch_match(self.owner.plan)
        plan = self.owner.plan
        epoch = plan.prepared_epoch
        assert epoch is not None
        operation = self.operation
        root = tuple(epoch.projection.root.graph.nodes)
        inner = tuple(epoch.projection.loop.graph.nodes)
        if operation is EpochMemoryOperation.INPUTS:
            refs = (plan.kd, plan.ak, plan.qd, plan.values, plan.g_total, plan.aq)
            loads = _tensor_refs(inner)
            nodes = tuple(
                tuple(node for node, ref in loads if _records(ref) == _records(wanted))
                for wanted in refs
            )
            if any(len(items) != 1 for items in nodes):
                raise chain._UnsupportedChain(
                    "input transaction lost original graph loads"
                )
            native = original.tma_chain_stage_load_inputs
            # Full allocated rings, not a per-use footprint or liveness discount.
            native_owners = (
                ("kd_smem", plan.input_stages, original.TILE_ELEMS),
                ("w_smem", plan.input_stages, original.TILE_ELEMS),
                ("qd_smem", plan.input_stages, original.TILE_ELEMS),
                ("v_raw_smem", plan.input_stages, original.V_TILE_ELEMS),
                ("diag_raw_smem", plan.input_stages, original.DIAG_REC_ELEMS),
                ("qk_smem", plan.input_stages, original.QK_REC_ELEMS),
            )
        elif operation is EpochMemoryOperation.IMPORT_STATE:
            refs = (plan.initial_state,)
            nodes = tuple(
                node
                for node, ref in _tensor_refs(root)
                if _records(ref) == _records(plan.initial_state)
            )
            if len(nodes) != 1:
                raise chain._UnsupportedChain("initial state lost original root load")
            nodes = (*nodes, epoch.projection.state)
            native = original.tcgen05_store_initial_state_tmem
            native_owners = original._TMEM_LAYOUT.region("state")
        elif operation is EpochMemoryOperation.EXPORT_STATE:
            refs = (plan.state,)
            nodes = tuple(
                node
                for node in root
                if (ref := _store_ref(node)) is not None
                and _records(ref) == _records(plan.state)
            )
            if len(nodes) != 1:
                raise chain._UnsupportedChain("final state lost original root store")
            nodes = (*nodes, nodes[0].args[2])
            native = original.tcgen05_store_final_state_tmem
            native_owners = original._TMEM_LAYOUT.region("state")
        elif operation in (
            EpochMemoryOperation.OUTPUT_FULL,
            EpochMemoryOperation.OUTPUT_TAIL,
        ):
            if (
                self.product is None
                or self.product.owner is not self.owner
                or self.product.cycle.kind != "output"
            ):
                raise chain._UnsupportedChain(
                    "output transport lost actual published value"
                )
            refs = (plan.output,)
            nodes = (
                self.product.source,
                self.product.destination,
                self.product.facts(),
            )
            native = (
                original.epilogue_chain_stage_store
                if operation is EpochMemoryOperation.OUTPUT_FULL
                else original.epilogue_chain_tail_store
            )
            native_owners = (
                "o_smem",
                plan.output_smem_stages,
                original.K2_O_SMEM_STAGE_SIZE,
            )
        else:
            raise chain._UnsupportedChain("unknown original boundary transport")
        if (
            operation
            not in (EpochMemoryOperation.OUTPUT_FULL, EpochMemoryOperation.OUTPUT_TAIL)
            and self.product is not None
        ):
            raise chain._UnsupportedChain("foreign result on input/state transport")
        return (
            id(self.owner),
            operation,
            id(self.product),
            nodes,
            _records(refs),
            _records(native_owners),
            _native_source(native, "_func"),
            _records((original._TMEM_LAYOUT, original.ROLES)),
        )

    def facts(self) -> object:
        if self._current() != self._facts:
            raise chain._UnsupportedChain("original boundary transport changed")
        return id(self), self._facts

    def emit(self) -> list[str]:
        self.facts()
        if self.operation is EpochMemoryOperation.INPUTS:
            return [
                "tma_chain_stage_load_inputs(tma_desc_kd, tma_desc_w, tma_desc_qd, tma_desc_v, tma_desc_diag, tma_desc_qk, kd_smem.subview(raw_stage * TILE_ELEMS), w_smem.subview(raw_stage * TILE_ELEMS), qd_smem.subview(raw_stage * TILE_ELEMS), v_raw_smem.subview(raw_stage * V_TILE_ELEMS), diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS), qk_smem.subview(raw_stage * QK_REC_ELEMS), bidy, dv_half, ws_row_start, ws_chunk, v_row_start, tma_mbar.subview(issue_mbar_slot), K2_TX_BYTES)"
            ]
        if self.operation is EpochMemoryOperation.IMPORT_STATE:
            return [
                "tcgen05_store_initial_state_tmem(tmem_raw_addr, initial_state, bidx, bidy, dv_half, warp_idx, lane)"
            ]
        if self.operation is EpochMemoryOperation.EXPORT_STATE:
            return [
                "tcgen05_store_final_state_tmem(tmem_raw_addr, KDA_TMEM_STATE_COL_OFFSET, state, bidx, bidy, dv_half, warp_idx, final_lane)"
            ]
        if self.operation is EpochMemoryOperation.OUTPUT_FULL:
            return [
                "epilogue_chain_stage_store(tma_desc_o, o_smem, sequence_start, bidy, dv_half, output_chunk_start, output_stage_base)"
            ]
        if self.operation is EpochMemoryOperation.OUTPUT_TAIL:
            return [
                "epilogue_chain_tail_store(out, o_smem, sequence_start, bidy, dv_half, output_chunk_start, seqlen, output_stage_base, lane)"
            ]
        raise chain._UnsupportedChain("unknown original boundary transport")
