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
from .cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_KEY
from .cute_mma_dag import MMA_DAG_CLUSTER_KEY
from .cute_mma_dag import MMA_DAG_DERIVED_EXCHANGE
from .cute_mma_dag import MMA_DAG_DERIVED_KEY
from .cute_mma_dag import MMA_DAG_EARLY_TMEM_RETIREMENT_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from .cute_mma_dag import MMA_DAG_EPILOGUE_ROW
from .cute_mma_dag import MMA_DAG_EXCHANGE_AUTO
from .cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from .cute_mma_dag import MMA_DAG_KEY
from .cute_mma_dag import MMA_DAG_KEY_CLIP_KEY
from .cute_mma_dag import MMA_DAG_L2_CANONICAL
from .cute_mma_dag import MMA_DAG_LOADS_KEY
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from .cute_mma_dag import MMA_DAG_LOOKAHEAD_OFF
from .cute_mma_dag import MMA_DAG_PARTITION_ITEM
from .cute_mma_dag import MMA_DAG_PARTITION_KEY
from .cute_mma_dag import MMA_DAG_PARTITION_SKEW_KEY
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from .cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from .cute_mma_dag import MMA_DAG_QUERY_TMA_KEY
from .cute_mma_dag import MMA_DAG_RELEASE_LEAD_KEY
from .cute_mma_dag import MMA_DAG_SLEEP_WAITS_KEY
from .cute_mma_dag import MMA_DAG_STAGES_KEY
from .cute_mma_dag import MMA_DAG_STORE_KEY
from .cute_mma_dag import MMA_DAG_STORE_STAGED
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_V_LAG_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY
from .cute_mma_dag import MMA_DAG_WARPS_KEY
from .cute_mma_dag import paged_attention_boundary_prefetch_choices
from .cute_mma_dag import paged_attention_clipped_page_lead_choices
from .cute_mma_dag import paged_attention_early_retirement_default
from .cute_mma_dag import paged_attention_key_clip_choices
from .cute_mma_dag import paged_attention_page_order_choices
from .cute_mma_dag import paged_attention_page_order_facts
from .cute_mma_dag import paged_attention_partition_choices
from .cute_mma_dag import paged_attention_partition_skew_choices
from .cute_mma_dag import paged_attention_pass_warps_reason
from .cute_mma_dag import paged_attention_probability_choices
from .cute_mma_dag import paged_attention_register_quota_choices
from .cute_mma_dag import paged_attention_release_lead_choices
from .cute_mma_dag import paged_attention_reversed_order_partition
from .cute_mma_dag import paged_attention_v_lag_choices
from .cute_mma_dag import paged_attention_value_ring_choices
from .cute_mma_dag import paged_attention_value_ring_facts
from .cute_mma_dag import paged_attention_value_ring_fit
from .paged_attention_page_order import MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES
from .paged_attention_page_order import MMA_DAG_CLIPPED_PAGE_LEAD_DEFAULT
from .paged_attention_page_order import MMA_DAG_CLIPPED_PAGE_LEAD_KEY
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_BUCKETED
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_KEY
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_REVERSED
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_SOURCE
from .paged_attention_page_order import PAGE_ORDER_MIN_SORTED
from .paged_attention_page_order import REVERSED_ORDER_NEEDS_PAIR
from .paged_attention_registers import MMA_DAG_REGISTER_QUOTAS_KEY
from .paged_attention_release import MMA_DAG_RELEASE_LEAD_DEFAULT
from .paged_attention_shared_memory import MMA_DAG_VALUE_RING_KEY
from .paged_attention_shared_memory import PAIRED_DEFAULT_VALUE_RING

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


def paged_attention_partition_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration cannot run several units in one persistent CTA.

    A persistent CTA requests each unit's queries with the proved query map,
    and the value-metadata actor has no walk over a CTA's units.
    """
    partition = config.get(MMA_DAG_PARTITION_KEY, MMA_DAG_PARTITION_ITEM)
    if type(partition) is str and partition == MMA_DAG_PARTITION_ITEM:
        return None
    pass_warps = config.get(MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PASS_WARPS_ATTENTION)
    choices = paged_attention_partition_choices(facts, pass_warps)
    if type(partition) is not str or partition not in choices:
        if type(partition) is str and partition in paged_attention_partition_choices(
            facts, MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
        ):
            # The usual way here is a config that names the partition and
            # little else, on a kernel whose no-config default is paired.
            return (
                f"{MMA_DAG_PARTITION_KEY}={partition!r} needs the paired attention "
                f"schedule ({MMA_DAG_PASS_WARPS_KEY}="
                f"{MMA_DAG_PASS_WARPS_ATTENTION_PAIRED!r} with "
                f"{MMA_DAG_QUERY_TMA_KEY}=True), and this config selects "
                f"{MMA_DAG_PASS_WARPS_KEY}={pass_warps!r}, whose only partition is "
                f"{MMA_DAG_PARTITION_ITEM!r}. A key that a config leaves out takes "
                "that key's own default, not the kernel's default config: add the "
                "partition to a complete config of the paired schedule, such as "
                "an autotuned one"
            )
        return f"{MMA_DAG_PARTITION_KEY} must be one of {choices!r} for {pass_warps}"
    if config.get(MMA_DAG_QUERY_TMA_KEY, False) is not True:
        return f"{MMA_DAG_PARTITION_KEY}={partition!r} requires {MMA_DAG_QUERY_TMA_KEY}=True"
    if config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY, False) is not False:
        return f"{MMA_DAG_PARTITION_KEY}={partition!r} requires {MMA_DAG_VALUE_METADATA_SKIP_KEY}=False"
    return None


def paged_attention_partition_skew_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration cannot skew the ranges of its partition."""
    skew = config.get(MMA_DAG_PARTITION_SKEW_KEY, 0)
    if type(skew) is int and skew == 0:
        return None
    partition = config.get(MMA_DAG_PARTITION_KEY, MMA_DAG_PARTITION_ITEM)
    choices = paged_attention_partition_skew_choices(facts, partition)
    if type(skew) is not int or skew not in choices:
        return (
            f"{MMA_DAG_PARTITION_SKEW_KEY} must be one of {choices!r} for "
            f"{MMA_DAG_PARTITION_KEY}={partition!r}"
        )
    return None


def _schedule_transport_partition(
    config: Mapping[str, object],
) -> tuple[object, object, object]:
    """What the choices of the paired body's stream options depend on in
    ``config``: its schedule, its value-metadata transport and its
    partition, each at the option's own default where the config leaves it
    out."""
    return (
        config.get(MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PASS_WARPS_ATTENTION),
        config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY, False),
        config.get(MMA_DAG_PARTITION_KEY, MMA_DAG_PARTITION_ITEM),
    )


def _choices_reason(
    key: str,
    choices: tuple[object, ...],
    selected: tuple[object, object, object],
    where: str,
) -> str:
    """Why a config's value of the option ``key`` is refused: it is none of
    the ``choices`` the option has on ``where`` under the schedule,
    value-metadata transport and partition the config ``selected``
    (``_schedule_transport_partition``)."""
    pass_warps, value_metadata_skip, partition = selected
    return (
        f"{key} must be one of {choices!r} for "
        f"{MMA_DAG_PASS_WARPS_KEY}={pass_warps!r}, "
        f"{MMA_DAG_VALUE_METADATA_SKIP_KEY}={value_metadata_skip!r} and "
        f"{MMA_DAG_PARTITION_KEY}={partition!r} on {where}"
    )


def paged_attention_boundary_prefetch_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration's schedule cannot run its boundary prefetch.

    The role is the paired schedule's idle control warp on a grid of one
    unit per CTA. The dependent-launch policy that the form of a one-wave
    grid needs as well is resolved after this check
    (``search_surface._normalize_boundary_prefetch_policy``).
    """
    pages = config.get(MMA_DAG_BOUNDARY_PREFETCH_KEY, 0)
    if type(pages) is int and pages == 0:
        return None
    selected = _schedule_transport_partition(config)
    choices = paged_attention_boundary_prefetch_choices(facts.attention, *selected)
    if type(pages) is not int or pages not in choices:
        return _choices_reason(
            MMA_DAG_BOUNDARY_PREFETCH_KEY, choices, selected, "this kernel's grid"
        )
    return None


def paged_attention_release_lead_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration's schedule cannot release its dependents with
    the lead it names.

    The lead belongs to the paired loader's tail release. Whether the
    config's dependent-launch policy releases there is resolved after this
    check (``search_surface._normalize_release_lead_policy``).
    """
    lead = config.get(MMA_DAG_RELEASE_LEAD_KEY, MMA_DAG_RELEASE_LEAD_DEFAULT)
    if type(lead) is int and lead == MMA_DAG_RELEASE_LEAD_DEFAULT:
        return None
    pass_warps = config.get(MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PASS_WARPS_ATTENTION)
    choices = paged_attention_release_lead_choices(facts.attention, pass_warps)
    if type(lead) is not int or lead not in choices:
        return (
            f"{MMA_DAG_RELEASE_LEAD_KEY} must be one of {choices!r} for "
            f"{MMA_DAG_PASS_WARPS_KEY}={pass_warps!r}"
        )
    return None


def paged_attention_value_ring_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration's schedule cannot stage the pages of raw V per
    head it names.

    The ring is the paired body's, the value-metadata actor walks it at its
    default depth, and a deeper ring must fit the shared memory the device
    grants one CTA, with what the config's partition adds to it.
    """
    ring = config.get(MMA_DAG_VALUE_RING_KEY, PAIRED_DEFAULT_VALUE_RING)
    if type(ring) is int and ring == PAIRED_DEFAULT_VALUE_RING:
        return None
    attention = facts.attention
    selected = _schedule_transport_partition(config)
    choices = paged_attention_value_ring_choices(attention, *selected)
    if type(ring) is int and ring in choices:
        return None
    reason = _choices_reason(MMA_DAG_VALUE_RING_KEY, choices, selected, "this device")
    ringed = paged_attention_value_ring_facts(attention, *selected)
    partition = selected[2]
    if type(ring) is int and ringed is not None:
        # The schedule has the ring: say what this partition's CTA would
        # need of the device for the depth.
        fit = paged_attention_value_ring_fit(ringed, ring, partition)
        if fit is not None:
            reason += f": {fit}"
    return reason


def paged_attention_page_order_reason(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> str | None:
    """Why this configuration's schedule cannot stream a unit's pages in
    the order it names.

    The bucketed order belongs to the paired schedule's one-unit CTAs
    without the value-metadata transport, under fast math, on a kernel some
    of whose units can have enough interior pages for the sort to pay. For
    a config whose own options would run the order the reason names both
    requirements of the bind (fast math, and a unit with that many interior
    pages); for any other it names the options that exclude the order. The
    reversed order belongs to the same schedule and transport, under fast
    math, on a kernel with a sliding window whose source loop visits its
    pages oldest first, with one unit per CTA and on the persistent
    partitions of a grid of more units than SMs; its reason names the
    requirement the kernel or the config does not meet
    (``_reversed_order_reason``).
    """
    order = config.get(MMA_DAG_PAGE_ORDER_KEY, MMA_DAG_PAGE_ORDER_SOURCE)
    if type(order) is str and order == MMA_DAG_PAGE_ORDER_SOURCE:
        return None
    selected = _schedule_transport_partition(config)
    choices = paged_attention_page_order_choices(facts.attention, *selected)
    if type(order) is str and order in choices:
        return None
    if type(order) is str and order == MMA_DAG_PAGE_ORDER_REVERSED:
        return _reversed_order_reason(facts, *selected)
    if (
        type(order) is str
        and order == MMA_DAG_PAGE_ORDER_BUCKETED
        and paged_attention_page_order_facts(facts.attention, *selected) is not None
    ):
        # The schedule, transport and partition admit the order, which the
        # choices do not hold: the bind does not offer it
        # (``PagedAttentionScheduleFacts.page_order_available``).
        return (
            f"{MMA_DAG_PAGE_ORDER_KEY}={order!r} needs a kernel bound with "
            "fast_math=True, some of whose units can have "
            f"{PAGE_ORDER_MIN_SORTED} or more interior pages to order: this "
            "kernel does not offer it"
        )
    return _choices_reason(MMA_DAG_PAGE_ORDER_KEY, choices, selected, "this kernel")


def _reversed_order_reason(
    facts: CuteMmaDagFacts,
    pass_warps: object,
    value_metadata_skip: object,
    partition: object,
) -> str:
    """Which requirement of ``cute_mma_dag_page_order="reversed"`` a config
    that the choices do not admit fails: the kernel's first (the pair proof,
    fast math, a sliding window, a source loop that is not reflected), then
    the config's own (the paired schedule, a partition the walk runs on, no
    value-metadata transport)."""
    named = f"{MMA_DAG_PAGE_ORDER_KEY}={MMA_DAG_PAGE_ORDER_REVERSED!r}"
    attention = facts.attention
    missing = (
        attention.reversed_order_missing
        if attention is not None and attention.paired_items
        else REVERSED_ORDER_NEEDS_PAIR
    )
    if missing:
        return f"{named} walks a work item's pages newest first and needs {missing}"
    if pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED:
        return (
            f"{named} needs the paired attention schedule "
            f"({MMA_DAG_PASS_WARPS_KEY}={MMA_DAG_PASS_WARPS_ATTENTION_PAIRED!r}), "
            f"and this config selects {MMA_DAG_PASS_WARPS_KEY}={pass_warps!r}: "
            "no other body has the walk"
        )
    if not paged_attention_reversed_order_partition(attention, partition):
        return (
            f"{named} needs one unit per CTA "
            f"({MMA_DAG_PARTITION_KEY}={MMA_DAG_PARTITION_ITEM!r}) or a grid of "
            "more units than the device has SMs, and this config selects "
            f"{MMA_DAG_PARTITION_KEY}={partition!r} on a grid that is not "
            "known to hold more: a persistent CTA runs about one unit there, "
            "and the walk was not examined on it"
        )
    assert value_metadata_skip is True
    return (
        f"{named} needs {MMA_DAG_VALUE_METADATA_SKIP_KEY}=False: the walk was "
        "not examined with the value-metadata transport"
    )


def paged_attention_clipped_page_lead_reason(
    config: Mapping[str, object],
) -> str | None:
    """Why this configuration cannot name the lead of the clipped page it
    names: the value is none of the option's.

    Whether the config has a page for the lead to place is no question of
    validity. Only a walk in reversed order has one, and beside any other
    order a lead names nothing: normalization drops it there
    (``paged_attention_clipped_page_lead_applies``), as it drops the default.
    """
    lead = config.get(MMA_DAG_CLIPPED_PAGE_LEAD_KEY, MMA_DAG_CLIPPED_PAGE_LEAD_DEFAULT)
    if type(lead) is int and lead in MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES:
        return None
    return (
        f"{MMA_DAG_CLIPPED_PAGE_LEAD_KEY} must be one of "
        f"{MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES!r}: the pages a walk in "
        f"{MMA_DAG_PAGE_ORDER_KEY}={MMA_DAG_PAGE_ORDER_REVERSED!r} has left when "
        "it takes the work item's oldest page"
    )


def paged_attention_clipped_page_lead_applies(
    facts: CuteMmaDagFacts, config: Mapping[str, object]
) -> bool:
    """Whether a lead of the clipped page other than the default names
    something in this configuration: whether its page order, already
    validated, is the reversed walk (``paged_attention_clipped_page_lead_choices``)."""
    lead = config.get(MMA_DAG_CLIPPED_PAGE_LEAD_KEY, MMA_DAG_CLIPPED_PAGE_LEAD_DEFAULT)
    return lead in paged_attention_clipped_page_lead_choices(
        facts.attention,
        *_schedule_transport_partition(config),
        config.get(MMA_DAG_PAGE_ORDER_KEY, MMA_DAG_PAGE_ORDER_SOURCE),
    )


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
    v_lag = config.get(MMA_DAG_V_LAG_KEY, 0)
    probability_schedule = config.get(
        MMA_DAG_PROBABILITY_SCHEDULE_KEY, MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
    )
    probability_choices = paged_attention_probability_choices(facts, pass_warps)
    if (
        type(probability_schedule) is not str
        or probability_schedule not in probability_choices
    ):
        return f"{MMA_DAG_PROBABILITY_SCHEDULE_KEY} must be one of {probability_choices!r} for {pass_warps}"
    v_lags = paged_attention_v_lag_choices(facts, pass_warps)
    if type(v_lag) is not int or v_lag not in v_lags:
        return f"{MMA_DAG_V_LAG_KEY} must be one of {v_lags!r} for {pass_warps}"
    sleep_waits = config.get(MMA_DAG_SLEEP_WAITS_KEY, False)
    if type(sleep_waits) is not bool or (
        sleep_waits and pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
    ):
        return f"{MMA_DAG_SLEEP_WAITS_KEY} requires the paired schedule and a bool"
    early_tmem_retirement = config.get(
        MMA_DAG_EARLY_TMEM_RETIREMENT_KEY,
        paged_attention_early_retirement_default(pass_warps),
    )
    if type(early_tmem_retirement) is not bool or (
        early_tmem_retirement and pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
    ):
        return f"{MMA_DAG_EARLY_TMEM_RETIREMENT_KEY} requires the paired schedule and a bool"
    query_tma = config.get(MMA_DAG_QUERY_TMA_KEY, False)
    if type(query_tma) is not bool or (
        query_tma
        and (
            pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
            or not attention.query_tma_available
        )
    ):
        return f"{MMA_DAG_QUERY_TMA_KEY} requires a bool and a proved paired query map"
    value_metadata_skip = config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY, False)
    if type(value_metadata_skip) is not bool or (
        value_metadata_skip
        and (
            pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
            or not attention.paired_items
            or not attention.value_metadata_skip_available
        )
    ):
        return f"{MMA_DAG_VALUE_METADATA_SKIP_KEY} requires a bool and proved paired value metadata"
    read_overlap = config.get(MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY, False)
    if type(read_overlap) is not bool or (read_overlap and not value_metadata_skip):
        return f"{MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY} requires a bool and enabled value metadata"
    key_clips = paged_attention_key_clip_choices(
        attention, pass_warps, value_metadata_skip
    )
    key_clip = config.get(MMA_DAG_KEY_CLIP_KEY, key_clips[0])
    if type(key_clip) is not str or key_clip not in key_clips:
        return f"{MMA_DAG_KEY_CLIP_KEY} must be one of {key_clips!r} for {pass_warps}"
    partition_reason = (
        paged_attention_partition_reason(facts, config)
        or paged_attention_partition_skew_reason(facts, config)
        or paged_attention_boundary_prefetch_reason(facts, config)
        or paged_attention_release_lead_reason(facts, config)
        or paged_attention_value_ring_reason(facts, config)
        or paged_attention_page_order_reason(facts, config)
        or paged_attention_clipped_page_lead_reason(config)
    )
    if partition_reason is not None:
        return partition_reason
    quota_choices = paged_attention_register_quota_choices(attention, pass_warps)
    quotas = config.get(MMA_DAG_REGISTER_QUOTAS_KEY, quota_choices[0])
    if type(quotas) is not str or quotas not in quota_choices:
        return f"{MMA_DAG_REGISTER_QUOTAS_KEY} must be one of {quota_choices!r} for {pass_warps}"
    for key, expected in paged_attention_required_knobs(facts).items():
        actual = config.get(key, expected)
        if type(actual) is not type(expected) or actual != expected:
            return f"the attention schedule requires {key}={expected!r}"
    if config.get("cute_pdl") and config.get("cute_pdl_spread"):
        # The generic entry wait and launch attribute cover this root, but
        # spread sizing cannot see allocations inside its imported body.
        return "the attention schedule has no PDL shared-memory padding"
    allowed_blocks = attention.min_blocks_per_mp_choices(
        scoped_registers=pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_SCOPED,
        paired_items=pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED,
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
    # The normalizer pins launch ordering and keeps only the live TMA cache
    # policies. Early admission and viability inspect eventual values;
    # selected planning validates the canonical config.
    if before_inert_globals:
        return None
    if config.get("pid_type", "flat") != "flat":
        return "the attention schedule requires pid_type='flat'"
    policies = config.get("load_eviction_policies", [])
    slots = set(attention.tma_load_slots)
    if not isinstance(policies, list) or any(
        policy not in ("", *MMA_DAG_L2_CANONICAL.values())
        or (slot not in slots and policy != "")
        for slot, policy in enumerate(policies)
    ):
        return "the attention schedule accepts L2 hints only on its TMA load slots"
    return None
