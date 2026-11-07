"""The explicit paged-attention schedule's configuration contract.

The transport planner supplies graph-free facts only after proving the bound
views and online recurrence. Search, normalization and the selected planner
share this check; choosing ordinary MMA schedules keeps their existing rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import MMA_DAG_AUX_KEY
from .cute_mma_dag import MMA_DAG_AUX_REGISTER
from .cute_mma_dag import MMA_DAG_CLUSTER_KEY
from .cute_mma_dag import MMA_DAG_DERIVED_EXCHANGE
from .cute_mma_dag import MMA_DAG_DERIVED_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_ROW
from .cute_mma_dag import MMA_DAG_EXCHANGE_AUTO
from .cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from .cute_mma_dag import MMA_DAG_KEY
from .cute_mma_dag import MMA_DAG_LOADS_KEY
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_OFF
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from .cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from .cute_mma_dag import MMA_DAG_STAGES_KEY
from .cute_mma_dag import MMA_DAG_STORE_KEY
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_WARPS_KEY
from .cute_mma_dag import paged_attention_pass_warps_reason

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .cute_mma_dag import CuteMmaDagFacts


def paged_attention_required_knobs(facts: CuteMmaDagFacts) -> dict[str, object]:
    """Fixed schedule choices shared by explicit seeds and admission."""
    attention = facts.attention
    assert attention is not None
    return {
        MMA_DAG_KEY: MMA_DAG_TCGEN05,
        MMA_DAG_WARPS_KEY: attention.compute_warps,
        MMA_DAG_STAGES_KEY: attention.raw_stages,
        MMA_DAG_STORE_KEY: MMA_DAG_STORE_STAGED,
        MMA_DAG_EXCHANGE_KEY: MMA_DAG_EXCHANGE_AUTO,
        MMA_DAG_LOADS_KEY: MMA_DAG_LOADS_TMA,
        MMA_DAG_AUX_KEY: MMA_DAG_AUX_REGISTER,
        MMA_DAG_DERIVED_KEY: MMA_DAG_DERIVED_EXCHANGE,
        MMA_DAG_EPILOGUE_KEY: MMA_DAG_EPILOGUE_ROW,
        MMA_DAG_LOOKAHEAD_KEY: MMA_DAG_LOOKAHEAD_OFF,
        MMA_DAG_PERSISTENT_KEY: 0,
        MMA_DAG_CLUSTER_KEY: 1,
    }


def paged_attention_config_reason(
    facts: CuteMmaDagFacts,
    sizes: Mapping[int, object],
    config: Mapping[str, object],
    *,
    before_inert_globals: bool = False,
) -> str | None:
    """Why the exact attention schedule cannot implement this configuration."""
    attention = facts.attention
    pass_warps = config.get(MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PASS_WARPS_ATTENTION)
    reason = paged_attention_pass_warps_reason(facts, pass_warps)
    if reason is not None:
        return reason
    assert attention is not None
    for key, expected in paged_attention_required_knobs(facts).items():
        actual = config.get(key, expected)
        if type(actual) is not type(expected) or actual != expected:
            return f"the attention schedule requires {key}={expected!r}"
    if config.get("cute_pdl") and config.get("cute_pdl_spread"):
        # The generic entry wait and launch attribute cover this root, but
        # spread sizing cannot see allocations inside its imported body.
        return "the attention schedule has no PDL shared-memory padding"
    allowed_blocks = attention.min_blocks_per_mp_choices(
        scoped_registers=pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
    )
    min_blocks = config.get("cute_min_blocks_per_mp", allowed_blocks[0])
    if type(min_blocks) is not int or min_blocks not in allowed_blocks:
        return (
            f"the attention schedule requires cute_min_blocks_per_mp={allowed_blocks[0]}"
            if len(allowed_blocks) == 1
            else "the attention schedule requires cute_min_blocks_per_mp in (1, 2)"
        )
    for axis, expected in (
        (attention.query_axis, attention.query_block),
        (attention.key_axis, attention.key_width),
        (attention.dimension_axis, attention.dimension),
    ):
        if facts.block_size(axis, sizes) != expected:
            return (
                "the attention schedule requires block size "
                f"{expected} for tile axis {facts.axes[axis].block_id}"
            )
    if any(
        axis.kind == AXIS_LEAD and sizes.get(axis.block_id, 1) != 1
        for axis in facts.axes
    ):
        return "the attention schedule runs one leading item per CTA"
    # The ordinary normalizer always canonicalizes these globals for the
    # nonpersistent tcgen05 body. Early admission and viability inspect the
    # eventual values; selected planning validates the canonical config.
    if before_inert_globals:
        return None
    if config.get("pid_type", "flat") != "flat":
        return "the attention schedule requires pid_type='flat'"
    policies = config.get("load_eviction_policies", [])
    if not isinstance(policies, list) or any(policy != "" for policy in policies):
        return "the attention schedule has no load eviction policy variants"
    return None
