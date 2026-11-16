"""The CuTe mma_dag and register sub-block families' search surface.

``enable_cute_mma_dag_search`` / ``enable_cute_register_subblocks_search``
open the families' knob fragments on a ``ConfigSpec`` from their planners'
facts; the normalizers legalize (or, under ``fix_invalid``, repair) a config
against those facts.  Called from ``search_surface.CuteSearchSurface`` at the
``normalize_recurrence_families`` phase."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import cast

from ....autotuner.config_fragment import ConfigSpecFragment
from ....autotuner.config_fragment import EnumFragment
from ....exc import InvalidConfig
from ..cute_mma_dag import AXIS_LEAD
from ..cute_mma_dag import AXIS_LOOP
from ..cute_mma_dag import AXIS_ROOT
from ..cute_mma_dag import MMA_DAG_ATTENTION_PASS_WARPS
from ..cute_mma_dag import MMA_DAG_AUX_CHOICES
from ..cute_mma_dag import MMA_DAG_AUX_KEY
from ..cute_mma_dag import MMA_DAG_AUX_RING
from ..cute_mma_dag import MMA_DAG_CLUSTER_CHOICES
from ..cute_mma_dag import MMA_DAG_CLUSTER_KEY
from ..cute_mma_dag import MMA_DAG_CLUSTER_MAX
from ..cute_mma_dag import MMA_DAG_DEFAULT_STAGES
from ..cute_mma_dag import MMA_DAG_DEFAULT_WARPS
from ..cute_mma_dag import MMA_DAG_DERIVED_CHOICES
from ..cute_mma_dag import MMA_DAG_DERIVED_KEY
from ..cute_mma_dag import MMA_DAG_DERIVED_PASS
from ..cute_mma_dag import MMA_DAG_EARLY_TMEM_RETIREMENT_KEY
from ..cute_mma_dag import MMA_DAG_EPILOGUE_CHOICES
from ..cute_mma_dag import MMA_DAG_EPILOGUE_KEY
from ..cute_mma_dag import MMA_DAG_EPILOGUE_ROW
from ..cute_mma_dag import MMA_DAG_EPILOGUE_SPLIT
from ..cute_mma_dag import MMA_DAG_EPILOGUE_STREAMED
from ..cute_mma_dag import MMA_DAG_EXCHANGE_CHOICES
from ..cute_mma_dag import MMA_DAG_EXCHANGE_KEY
from ..cute_mma_dag import MMA_DAG_KEY
from ..cute_mma_dag import MMA_DAG_KEY_CLIP_KEY
from ..cute_mma_dag import MMA_DAG_KEY_CLIP_OFF
from ..cute_mma_dag import MMA_DAG_L2_CANONICAL
from ..cute_mma_dag import MMA_DAG_L2_HINTS
from ..cute_mma_dag import MMA_DAG_LOADS_CHOICES
from ..cute_mma_dag import MMA_DAG_LOADS_KEY
from ..cute_mma_dag import MMA_DAG_LOADS_LANE
from ..cute_mma_dag import MMA_DAG_LOADS_TMA
from ..cute_mma_dag import MMA_DAG_LOOKAHEAD_CHOICES
from ..cute_mma_dag import MMA_DAG_LOOKAHEAD_KEY
from ..cute_mma_dag import MMA_DAG_LOOKAHEAD_OFF
from ..cute_mma_dag import MMA_DAG_LOOKAHEAD_ON
from ..cute_mma_dag import MMA_DAG_MODES
from ..cute_mma_dag import MMA_DAG_OFF
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_CHOICES
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_MMA
from ..cute_mma_dag import MMA_DAG_PASS_WARPS_OFF
from ..cute_mma_dag import MMA_DAG_PERSISTENT_CHOICES
from ..cute_mma_dag import MMA_DAG_PERSISTENT_KEY
from ..cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_CHOICES
from ..cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_KEY
from ..cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from ..cute_mma_dag import MMA_DAG_QUERY_TMA_KEY
from ..cute_mma_dag import MMA_DAG_SLEEP_WAITS_KEY
from ..cute_mma_dag import MMA_DAG_SPLIT_CHOICES
from ..cute_mma_dag import MMA_DAG_SPLIT_CLUSTER
from ..cute_mma_dag import MMA_DAG_SPLIT_FIXUP
from ..cute_mma_dag import MMA_DAG_SPLIT_KEY
from ..cute_mma_dag import MMA_DAG_STAGE_CHOICES
from ..cute_mma_dag import MMA_DAG_STAGES_KEY
from ..cute_mma_dag import MMA_DAG_STORE_ACCEPTED
from ..cute_mma_dag import MMA_DAG_STORE_CHOICES
from ..cute_mma_dag import MMA_DAG_STORE_KEY
from ..cute_mma_dag import MMA_DAG_TCGEN05
from ..cute_mma_dag import MMA_DAG_V_LAG_CHOICES
from ..cute_mma_dag import MMA_DAG_V_LAG_KEY
from ..cute_mma_dag import MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY
from ..cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY
from ..cute_mma_dag import MMA_DAG_WARP
from ..cute_mma_dag import MMA_DAG_WARP_CHOICES
from ..cute_mma_dag import MMA_DAG_WARPS_KEY
from ..cute_mma_dag import CuteMmaDagFacts
from ..cute_mma_dag import apply_mma_dag_cluster
from ..cute_mma_dag import epilogue_knob_admitted
from ..cute_mma_dag import fixup_admitted
from ..cute_mma_dag import mma_dag_aux_reason
from ..cute_mma_dag import mma_dag_epilogue_reason
from ..cute_mma_dag import mma_dag_lookahead_reason
from ..cute_mma_dag import mma_dag_pass_warps_reason
from ..cute_mma_dag import mma_dag_persistent_reason
from ..cute_mma_dag import mma_dag_ragged_ceilings
from ..cute_mma_dag import mma_dag_seed_configs
from ..cute_mma_dag import paged_attention_early_retirement_default
from ..cute_mma_dag import paged_attention_key_clip_choices
from ..cute_mma_dag import paged_attention_pass_warp_choices
from ..cute_register_subblocks import COLSIDE_CHOICES as RSB_COLSIDE_CHOICES
from ..cute_register_subblocks import COLSIDE_KEY as RSB_COLSIDE_KEY
from ..cute_register_subblocks import COLSIDE_RMW as RSB_COLSIDE_RMW
from ..cute_register_subblocks import KEY as RSB_KEY
from ..cute_register_subblocks import MASK_CHOICES as RSB_MASK_CHOICES
from ..cute_register_subblocks import MASK_KEY as RSB_MASK_KEY
from ..cute_register_subblocks import MASK_SELECT as RSB_MASK_SELECT
from ..cute_register_subblocks import MODE_OFF as RSB_MODE_OFF
from ..cute_register_subblocks import MODES as RSB_MODES
from ..cute_register_subblocks import PREFETCH_CHOICES as RSB_PREFETCH_CHOICES
from ..cute_register_subblocks import PREFETCH_KEY as RSB_PREFETCH_KEY
from ..cute_register_subblocks import ROLL_CHOICES as RSB_ROLL_CHOICES
from ..cute_register_subblocks import ROLL_KEY as RSB_ROLL_KEY
from ..cute_register_subblocks import ROLL_UNROLLED as RSB_ROLL_UNROLLED
from ..cute_register_subblocks import WARP_CHOICES as RSB_WARP_CHOICES
from ..cute_register_subblocks import WARPS_KEY as RSB_WARPS_KEY
from ..cute_register_subblocks import RegisterSubblockFacts
from ..cute_register_subblocks import register_subblocks_config_reason
from ..cute_register_subblocks import register_subblocks_repair
from ..cutedsl_compat import hinted_cp_async_available
from ..pdl_release import CUTE_PDL_CAPTURE_DEFAULT
from ..pdl_release import CUTE_PDL_CAPTURE_OVERLAP_KEY
from ..pdl_release import CUTE_PDL_CAPTURE_RELEASE_CHOICES
from ..pdl_release import CUTE_PDL_RELEASE_DEFAULT
from ..pdl_release import CUTE_PDL_SPREAD_KEY
from ..pdl_release import capture_policy_available
from ..phase_load_policies import project_load_policies
from ..phase_load_policies import scatter_load_policies

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ....autotuner.config_spec import ConfigSpec


# The knobs ``_normalize_cute_mma_dag`` sheds from a refused tile before the
# tile itself is repaired, in order: ``(key, value the knob is shed from
# (None: any value but its default), the knobs the probe puts at their
# defaults, the config values the shed sets)``.  Each probe re-admits the
# tile through ``mma_dag_config_reason`` with ``dropped`` at the defaults of
# that function's parameters; the first probe that admits names the knob,
# and the tile stays.  The order is policy, not derived from the refusal
# (a budget sum names no single knob): the MMA warp's barriers first (they
# overflow the shared memory the chain schedule fits, or the streamed
# loops' accumulators live in registers at this tile), the persistent
# layout (a region program's root items beside the frames), the aux rings
# (shared memory or the per-thread cap the register prefetch fits), the
# lookahead's doubled dot slot (tensor memory the program order fits), the
# split epilogue's budget or layout (and the schedule with it), the pass
# route's ring tiles last.  A later probe drops the earlier knobs too,
# whether or not they are set: a config with the MMA warp that only the
# persistent drop admits keeps the warp.  The derived probe keeps the
# epilogue and the lookahead at the config's values.
_MMA_DAG_KNOB_PROBE_DEFAULTS: dict[str, object] = {
    MMA_DAG_PASS_WARPS_KEY: MMA_DAG_PASS_WARPS_OFF,
    MMA_DAG_PERSISTENT_KEY: False,
    MMA_DAG_AUX_KEY: MMA_DAG_AUX_CHOICES[0],
    MMA_DAG_LOOKAHEAD_KEY: MMA_DAG_LOOKAHEAD_OFF,
    MMA_DAG_EPILOGUE_KEY: MMA_DAG_EPILOGUE_ROW,
    MMA_DAG_DERIVED_KEY: MMA_DAG_DERIVED_CHOICES[0],
}
_MMA_DAG_KNOB_DROPS: tuple[
    tuple[str, object | None, tuple[str, ...], dict[str, object]], ...
] = (
    (
        MMA_DAG_PASS_WARPS_KEY,
        MMA_DAG_PASS_WARPS_MMA,
        (MMA_DAG_PASS_WARPS_KEY,),
        {MMA_DAG_PASS_WARPS_KEY: MMA_DAG_PASS_WARPS_OFF},
    ),
    (
        MMA_DAG_PERSISTENT_KEY,
        None,
        (MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PERSISTENT_KEY),
        {MMA_DAG_PERSISTENT_KEY: 0},
    ),
    (
        MMA_DAG_AUX_KEY,
        MMA_DAG_AUX_RING,
        (MMA_DAG_PASS_WARPS_KEY, MMA_DAG_PERSISTENT_KEY, MMA_DAG_AUX_KEY),
        {MMA_DAG_AUX_KEY: MMA_DAG_AUX_CHOICES[0]},
    ),
    (
        MMA_DAG_LOOKAHEAD_KEY,
        MMA_DAG_LOOKAHEAD_ON,
        (
            MMA_DAG_PASS_WARPS_KEY,
            MMA_DAG_PERSISTENT_KEY,
            MMA_DAG_AUX_KEY,
            MMA_DAG_LOOKAHEAD_KEY,
        ),
        {MMA_DAG_LOOKAHEAD_KEY: MMA_DAG_LOOKAHEAD_OFF},
    ),
    (
        MMA_DAG_EPILOGUE_KEY,
        MMA_DAG_EPILOGUE_SPLIT,
        (
            MMA_DAG_PASS_WARPS_KEY,
            MMA_DAG_PERSISTENT_KEY,
            MMA_DAG_AUX_KEY,
            MMA_DAG_LOOKAHEAD_KEY,
            MMA_DAG_EPILOGUE_KEY,
        ),
        {
            MMA_DAG_EPILOGUE_KEY: MMA_DAG_EPILOGUE_ROW,
            MMA_DAG_LOOKAHEAD_KEY: MMA_DAG_LOOKAHEAD_OFF,
        },
    ),
    (
        MMA_DAG_DERIVED_KEY,
        MMA_DAG_DERIVED_PASS,
        (
            MMA_DAG_PASS_WARPS_KEY,
            MMA_DAG_PERSISTENT_KEY,
            MMA_DAG_AUX_KEY,
            MMA_DAG_DERIVED_KEY,
        ),
        {MMA_DAG_DERIVED_KEY: MMA_DAG_DERIVED_CHOICES[0]},
    ),
)


def enable_cute_register_subblocks_search(
    config_spec: ConfigSpec, facts: RegisterSubblockFacts
) -> None:
    """Expose the register sub-block family and its knobs.

    ``off`` leads the mode domain so a config that leaves the knob out
    keeps the ordinary lowering; the seed heuristic (and, on validated
    targets, the promoted default) selects ``on``.  A knob without a
    subject (no prefetchable second item, no static mask, no window
    written twice) stays off the surface and is pinned by ``normalize``.
    """
    config_spec.cute_register_subblocks_facts = facts
    config_spec.cute_register_subblocks = EnumFragment(choices=RSB_MODES)
    config_spec.cute_register_subblocks_warps = EnumFragment(choices=RSB_WARP_CHOICES)
    config_spec.cute_register_subblocks_prefetch = (
        EnumFragment(choices=RSB_PREFETCH_CHOICES) if facts.prefetch_subject else None
    )
    config_spec.cute_register_subblocks_mask = (
        EnumFragment(choices=RSB_MASK_CHOICES) if facts.has_static_mask else None
    )
    config_spec.cute_register_subblocks_colside = (
        EnumFragment(choices=RSB_COLSIDE_CHOICES) if facts.multi_write_windows else None
    )
    config_spec.cute_register_subblocks_roll = (
        EnumFragment(choices=RSB_ROLL_CHOICES) if facts.roll_subject else None
    )


def enable_cute_mma_dag_search(config_spec: ConfigSpec, facts: CuteMmaDagFacts) -> None:
    """Expose the warp-MMA recurrence mode, warp count and stage depth.

    ``off`` leads the mode domain so a config that leaves the knob out
    keeps the ordinary lowering; the seed heuristic (and, on validated
    targets, the promoted default) selects ``warp``.
    """
    config_spec.cute_mma_dag_facts = facts
    ordered_phase = config_spec.cute_mma_dag_phase_index is not None
    if ordered_phase:
        # MMA modes normalize to separate flat launches. Keep that default,
        # but the off mode can still use the ordinary persistent barrier
        # lowering and must retain its existing search choices.
        config_spec.allowed_pid_types = tuple(
            sorted(config_spec.allowed_pid_types, key=lambda pid: pid != "flat")
        )
    # A ragged root axis below its MMA atom (M = 8 under the 16-row atom)
    # is computed at the atom on the warp path, and the seeds walk that
    # tile: its search ceiling, the extent where the frontend left it,
    # admits the atom.  The default block size stays clamped to the
    # extent (``BlockSizeSpec._fragment``); no other block widens.
    block_ids = config_spec.block_sizes.valid_block_ids()
    for block_id, atom in mma_dag_ragged_ceilings(facts).items():
        if block_id in block_ids:
            config_spec.block_sizes.block_id_lookup(block_id).allow_overshoot(atom)
    if facts.attention is not None:
        query_axis = facts.axes[facts.attention.query_axis]
        if query_axis.block_id in block_ids:
            config_spec.block_sizes.block_id_lookup(
                query_axis.block_id
            ).allow_overshoot(facts.attention.query_block)
    modes = MMA_DAG_MODES
    if facts.tcgen05 is not None:
        modes = (*modes, MMA_DAG_TCGEN05)
    search = modes
    if facts.split_axis >= 0 and MMA_DAG_TCGEN05 in modes:
        # A full-slice fragment program: its row reductions over the
        # whole axis have the region renderer's lowering, while the SIMT
        # fallback's lane-restore pass declines most tilings (50 of 55
        # random "off" configs at 2048^3 and 49 of 56 at 512x1024x1024
        # failed to compile, every tensor-memory config compiled), so
        # the search samples the tensor-memory mode; "off" stays a legal
        # value (the repair fallback, explicit configs).  The warp mode
        # holds the whole axis in one CTA: it is sampled only when some
        # tile of its seed ladder runs the program (a 2048-column
        # fragment exceeds its register budget at every tiling, and a
        # random warp draw would only be repaired away).
        skipped = {MMA_DAG_OFF}
        if not mma_dag_seed_configs(facts):
            skipped.add(MMA_DAG_WARP)
        search = tuple(mode for mode in modes if mode not in skipped)
    if facts.native_s2f6_operands and facts.attention is None:
        # Native decode has no admitted MMA schedule without its proved
        # attention transport. Keep explicit-config validation and ordinary
        # lowering, but do not sample modes that always normalize to off.
        search = (MMA_DAG_OFF,)
    config_spec.cute_mma_dag = EnumFragment(
        choices=modes, search_choices=search if search != modes else None
    )
    config_spec.cute_mma_dag_warps = EnumFragment(
        choices=(
            MMA_DAG_DEFAULT_WARPS,
            *(w for w in MMA_DAG_WARP_CHOICES if w != MMA_DAG_DEFAULT_WARPS),
        )
    )
    config_spec.cute_mma_dag_stages = EnumFragment(
        choices=(
            MMA_DAG_DEFAULT_STAGES,
            *(s for s in MMA_DAG_STAGE_CHOICES if s != MMA_DAG_DEFAULT_STAGES),
        )
    )
    config_spec.cute_mma_dag_store = EnumFragment(choices=MMA_DAG_STORE_CHOICES)
    config_spec.cute_mma_dag_exchange = EnumFragment(choices=MMA_DAG_EXCHANGE_CHOICES)
    attention_choices = paged_attention_pass_warp_choices(facts)
    config_spec.cute_mma_dag_pass_warps = EnumFragment(
        choices=tuple(
            choice
            for choice in MMA_DAG_PASS_WARPS_CHOICES
            if choice not in MMA_DAG_ATTENTION_PASS_WARPS or choice in attention_choices
        )
    )
    # Facts can be re-registered on an existing spec after a proof changes.
    config_spec.cute_mma_dag_v_lag = None
    config_spec.cute_mma_dag_sleep_waits = None
    config_spec.cute_mma_dag_early_tmem_retirement = None
    config_spec.cute_mma_dag_query_tma = None
    config_spec.cute_mma_dag_probability_schedule = None
    config_spec.cute_mma_dag_value_metadata_skip = None
    config_spec.cute_mma_dag_value_metadata_read_overlap = None
    config_spec.cute_mma_dag_key_clip = None
    if facts.attention is not None and facts.attention.paired_items:
        if len(facts.attention.key_clip_choices) > 1:
            config_spec.cute_mma_dag_key_clip = EnumFragment(
                choices=facts.attention.key_clip_choices
            )
        config_spec.cute_mma_dag_v_lag = EnumFragment(choices=MMA_DAG_V_LAG_CHOICES)
        config_spec.cute_mma_dag_sleep_waits = EnumFragment(choices=(False, True))
        config_spec.cute_mma_dag_early_tmem_retirement = EnumFragment(
            choices=(False, True)
        )
        if facts.attention.reference_scores:
            config_spec.cute_mma_dag_probability_schedule = EnumFragment(
                choices=MMA_DAG_PROBABILITY_SCHEDULE_CHOICES
            )
        if facts.attention.query_tma_available:
            config_spec.cute_mma_dag_query_tma = EnumFragment(choices=(False, True))
        if facts.attention.value_metadata_skip_available:
            config_spec.cute_mma_dag_value_metadata_skip = EnumFragment(
                choices=(False, True)
            )
            config_spec.cute_mma_dag_value_metadata_read_overlap = EnumFragment(
                choices=(False, True)
            )
    config_spec.cute_mma_dag_loads = EnumFragment(choices=MMA_DAG_LOADS_CHOICES)
    if facts.aux_candidates:
        config_spec.cute_mma_dag_aux = EnumFragment(choices=MMA_DAG_AUX_CHOICES)
    config_spec.cute_mma_dag_derived = EnumFragment(choices=MMA_DAG_DERIVED_CHOICES)
    # ``streamed`` is offered only where a root block without derived
    # values stores fragments; elsewhere the knob keeps row / split.
    config_spec.cute_mma_dag_epilogue = EnumFragment(
        choices=MMA_DAG_EPILOGUE_CHOICES
        if epilogue_knob_admitted(facts)
        else tuple(
            choice
            for choice in MMA_DAG_EPILOGUE_CHOICES
            if choice != MMA_DAG_EPILOGUE_STREAMED
        )
    )
    config_spec.cute_mma_dag_lookahead = EnumFragment(choices=MMA_DAG_LOOKAHEAD_CHOICES)
    if (
        not ordered_phase
        and facts.tcgen05 is not None
        and (not facts.tcgen05.regions or facts.tcgen05.persistent_reason is None)
    ):
        # The persistent launch: the single-loop renderer's tile loop,
        # or a region program's (``cute_mma_dag_tcgen05_regions``;
        # ``persistent_reason`` names the programs it excludes).
        config_spec.cute_mma_dag_persistent = EnumFragment(
            choices=MMA_DAG_PERSISTENT_CHOICES
        )
    if (
        not ordered_phase
        and facts.tcgen05 is not None
        and facts.tcgen05.regions
        and facts.split_axis >= 0
    ):
        # The split count: a portable cluster's (<= 8), and the fixup
        # grid's wider ones where the program admits the fixup form.
        admitted = fixup_admitted(facts)
        config_spec.cute_mma_dag_cluster = EnumFragment(
            choices=tuple(
                choice
                for choice in MMA_DAG_CLUSTER_CHOICES
                if admitted or choice <= MMA_DAG_CLUSTER_MAX
            )
        )
        config_spec.cute_mma_dag_split = (
            EnumFragment(choices=MMA_DAG_SPLIT_CHOICES) if admitted else None
        )


def _normalize_cute_register_subblocks(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Validate the register sub-block knobs against the facts (tile,
    warps, register estimate) and pin the knobs the replaced body
    ignores to one identity."""
    facts = config_spec.cute_register_subblocks_facts
    assert facts is not None
    mode = config.setdefault(RSB_KEY, RSB_MODE_OFF)
    if mode not in RSB_MODES or type(mode) is not str:
        if not fix_invalid:
            raise InvalidConfig(f"{RSB_KEY} must be one of {RSB_MODES!r}, got {mode!r}")
        config[RSB_KEY] = mode = RSB_MODE_OFF
    defaults: dict[str, object] = {
        RSB_WARPS_KEY: RSB_WARP_CHOICES[0],
        RSB_PREFETCH_KEY: RSB_PREFETCH_CHOICES[0],
        RSB_MASK_KEY: RSB_MASK_SELECT,
        RSB_COLSIDE_KEY: RSB_COLSIDE_RMW,
        RSB_ROLL_KEY: RSB_ROLL_UNROLLED,
    }
    fragments = {
        RSB_WARPS_KEY: config_spec.cute_register_subblocks_warps,
        RSB_PREFETCH_KEY: config_spec.cute_register_subblocks_prefetch,
        RSB_MASK_KEY: config_spec.cute_register_subblocks_mask,
        RSB_COLSIDE_KEY: config_spec.cute_register_subblocks_colside,
        RSB_ROLL_KEY: config_spec.cute_register_subblocks_roll,
    }
    if mode == RSB_MODE_OFF:
        config.update(defaults)
        return
    for key, fragment in fragments.items():
        value = config.setdefault(key, defaults[key])
        allowed = (defaults[key],) if fragment is None else fragment.choices
        if value not in allowed or type(value) is not type(defaults[key]):
            if not fix_invalid:
                raise InvalidConfig(f"{key} must be one of {allowed!r}, got {value!r}")
            config[key] = defaults[key]
    block_sizes = config.get("block_sizes")
    tile_index = None
    bk: object = None
    if facts.tile_block_id is not None:
        tile_index = config_spec.block_sizes.block_id_to_index(facts.tile_block_id)
        assert isinstance(block_sizes, list)
        bk = block_sizes[tile_index]
    reason = register_subblocks_config_reason(
        facts,
        bk=bk,
        warps=config[RSB_WARPS_KEY],
        prefetch=config[RSB_PREFETCH_KEY],
        mask=config[RSB_MASK_KEY],
        colside=config[RSB_COLSIDE_KEY],
        roll=config[RSB_ROLL_KEY],
    )
    if reason is not None:
        if not fix_invalid:
            raise InvalidConfig(f"{RSB_KEY}={mode!r}: {reason}")
        repaired = register_subblocks_repair(
            facts,
            bk=bk,
            warps=config[RSB_WARPS_KEY],
            prefetch=config[RSB_PREFETCH_KEY],
            mask=config[RSB_MASK_KEY],
            colside=config[RSB_COLSIDE_KEY],
            roll=config[RSB_ROLL_KEY],
        )
        if repaired is None:
            config[RSB_KEY] = RSB_MODE_OFF
            config.update(defaults)
            return
        for key in defaults:
            config[key] = repaired[key]
        if tile_index is not None:
            assert isinstance(block_sizes, list)
            block_sizes[tile_index] = repaired["bk"]
    # The replaced body reads no fragment-route knob: pin them so the
    # search does not sample inert variations of one kernel.
    for key in (
        "cute_fp32_dot",
        "cute_fragment_half_mma",
        "cute_fragment_padded_rows",
        "cute_fragment_unrolled_loops",
        "cute_fragment_vector_io",
        "cute_fragment_batched_loads",
        "cute_fragment_split_k",
        "cute_fragment_slab_loads",
        "cute_fragment_threads",
        "cute_fragment_mma_rows",
        "cute_fragment_register_loads",
        "cute_fragment_producer_cache",
        "cute_fragment_warp_scan",
        "cute_fragment_scan",
        "cute_fragment_reduction",
        "cute_fragment_pure_producer_regions",
        "cute_fragment_warp_results",
        "cute_fragment_warp_producer_regions",
        "cute_fragment_published_scalars",
        "cute_fragment_register_producers",
        "cute_fragment_register_snapshots",
        "cute_fragment_private_scalar_loops",
        "cute_fragment_packet_loads",
        "cute_fragment_bounded_gather",
    ):
        config.pop(key, None)
    for key, identity in (
        ("cute_vector_widths", 1),
        ("cute_lane_layouts", "blocked"),
        ("cute_reduction_reloads", "auto"),
        ("load_eviction_policies", ""),
    ):
        value = config.get(key)
        if isinstance(value, list):
            config[key] = [identity] * len(value)
    orders = config.get("loop_orders")
    if isinstance(orders, list):
        config["loop_orders"] = [
            list(range(len(order))) if isinstance(order, list) else order
            for order in orders
        ]
    if "cute_store_policy" in config:
        config["cute_store_policy"] = "default"


def normalize_value_metadata_options(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Validate before family gates; overlap also requires metadata to be enabled."""
    for key, fragment in (
        (MMA_DAG_VALUE_METADATA_SKIP_KEY, config_spec.cute_mma_dag_value_metadata_skip),
        (
            MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY,
            config_spec.cute_mma_dag_value_metadata_read_overlap,
        ),
    ):
        if fragment is None and key not in config:
            continue
        value = config.setdefault(key, False)
        if type(value) is not bool or (fragment is None and value):
            if not fix_invalid:
                raise InvalidConfig(
                    f"{key} requires a bool and supported metadata transport, got {value!r}"
                )
            config[key] = False
        if fragment is None:
            # Source/layout specialization can remove this proof. Missing and
            # explicit defaults retain one identity, including on generic roots.
            config.pop(key, None)
    if (
        config_spec.cute_mma_dag_value_metadata_read_overlap is not None
        and config.get(MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY) is True
        and config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY) is not True
    ):
        if not fix_invalid:
            raise InvalidConfig(
                f"{MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY}=True requires "
                f"{MMA_DAG_VALUE_METADATA_SKIP_KEY}=True"
            )
        config[MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY] = False


# The paired schedule's knobs, each with the value that every body ignoring
# them keeps.
_PAIRED_KNOB_PINS: tuple[tuple[str, object], ...] = (
    (MMA_DAG_V_LAG_KEY, 0),
    (MMA_DAG_SLEEP_WAITS_KEY, False),
    (MMA_DAG_EARLY_TMEM_RETIREMENT_KEY, False),
    (MMA_DAG_QUERY_TMA_KEY, False),
    (MMA_DAG_VALUE_METADATA_SKIP_KEY, False),
    (MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY, False),
    (MMA_DAG_PROBABILITY_SCHEDULE_KEY, MMA_DAG_PROBABILITY_SCHEDULE_SERIAL),
    (MMA_DAG_KEY_CLIP_KEY, MMA_DAG_KEY_CLIP_OFF),
)


def _pin_paired_knobs(config: dict[str, object]) -> None:
    """Bodies that ignore these knobs keep one autotune identity."""
    for key, default in _PAIRED_KNOB_PINS:
        if key in config:
            config[key] = default


def drop_paired_knob_pins(
    base: dict[str, object], promoted: Mapping[str, object]
) -> None:
    """Drop ``base``'s paired knobs where ``promoted``, layered over it next,
    selects the paired schedule (``ConfigSpec.promoted_default_values``).

    The base default runs another body, which pinned them. A pin left in
    place reads as a choice of the promoted config, so a knob that config
    does not name would keep the other body's identity instead of the paired
    schedule's default: the final join of a schedule that retires its
    allocation early. Without the pins, normalization derives those
    defaults."""
    if promoted.get(MMA_DAG_PASS_WARPS_KEY) != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED:
        return
    for key, _pin in _PAIRED_KNOB_PINS:
        base.pop(key, None)


# The options that select the paired schedule, and the dependent-launch
# policy a proved kernel runs it under by default, spread padding included.
# Every rule about that policy reads these tables: what a config must hold to
# keep it (``CAPTURE_POLICY_REQUIREMENTS``), which schedule a kernel is on
# (``paired_capture_schedule``) and what a search that does not sample the
# policy keeps (``kept_capture_policy``).
PAIRED_CAPTURE_SCHEDULE = MappingProxyType(
    {
        MMA_DAG_KEY: MMA_DAG_TCGEN05,
        MMA_DAG_PASS_WARPS_KEY: MMA_DAG_PASS_WARPS_ATTENTION_PAIRED,
    }
)
CAPTURE_POLICY = MappingProxyType(
    {**CUTE_PDL_CAPTURE_DEFAULT, CUTE_PDL_SPREAD_KEY: False}
)
# What the guarded capture policy needs of a config: its own options and the
# options ``capture_overlap_eligible`` reads.
CAPTURE_POLICY_REQUIREMENTS = MappingProxyType(
    {**CAPTURE_POLICY, **PAIRED_CAPTURE_SCHEDULE}
)


def paired_capture_schedule(available: bool, config: Mapping[str, object]) -> bool:
    """Whether ``config`` selects the proved paired schedule of a kernel
    whose capture-time dependency proof exists (a single paired root without
    explicit dependent-launch markers).

    ``available`` is the kernel's side of the rule,
    ``capture_policy_available`` of a binding; the rest reads ``config``
    alone. Every rule about the guarded policy takes that one fact as an
    argument: the rules are also asked what a config does on a kernel that
    has the proof, whatever binding is at hand
    (``search_surface._held_capture_overrides``)."""
    return available and all(
        config.get(key) == value for key, value in PAIRED_CAPTURE_SCHEDULE.items()
    )


def capture_overlap_eligible(available: bool, config: Mapping[str, object]) -> bool:
    """Whether the rest of ``config`` permits capture overlap on a kernel
    whose ``capture_policy_available`` is ``available``: the proved paired
    schedule under ``cute_pdl``, without spread padding, releasing where a
    capture launch can.

    The one statement of that rule: normalization asks it, and so do code
    generation and every hook that acts on the guarded policy, such as the
    one that declines the policy under a contradicting override (in a
    search candidate and in a config no search produced), through
    ``carries_capture_overlap``."""
    return (
        paired_capture_schedule(available, config)
        and config.get("cute_pdl") is True
        and not config.get(CUTE_PDL_SPREAD_KEY)
        and config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
        in CUTE_PDL_CAPTURE_RELEASE_CHOICES
    )


def carries_capture_overlap(available: bool, config: Mapping[str, object]) -> bool:
    """Whether ``config``, normalized or a candidate under its overrides,
    launches with the guarded capture overlap on a kernel whose
    ``capture_policy_available`` is ``available``: it asks for the overlap
    and the rest of it permits that (``capture_overlap_eligible``)."""
    overlap = config.get(CUTE_PDL_CAPTURE_OVERLAP_KEY) is True
    return overlap and capture_overlap_eligible(available, config)


def carried_capture_policies() -> tuple[dict[str, object], ...]:
    """The options ``carries_capture_overlap`` reads, as every normalized
    config that launches with the guarded overlap can state them: the paired
    schedule under ``cute_pdl`` with the overlap, on each release a capture
    launch can use or none named (the automatic one), spread padding off or
    not named."""
    carried = {
        **PAIRED_CAPTURE_SCHEDULE,
        "cute_pdl": True,
        CUTE_PDL_CAPTURE_OVERLAP_KEY: True,
    }
    policies: list[dict[str, object]] = []
    for release in (None, *CUTE_PDL_CAPTURE_RELEASE_CHOICES):
        for spread in (None, False):
            policy: dict[str, object] = dict(carried)
            if release is not None:
                policy["cute_pdl_release"] = release
            if spread is not None:
                policy[CUTE_PDL_SPREAD_KEY] = spread
            policies.append(policy)
    return tuple(policies)


def kept_capture_policy(
    config_spec: ConfigSpec, overrides: Mapping[str, object]
) -> dict[str, object] | None:
    """The dependent-launch options of every candidate on the proved paired
    schedule in a search that does not sample them: the default policy with
    the options ``overrides`` name as they state them. ``None`` where
    nothing is kept: the search owns the policy, the kernel has no capture
    contract, or the overrides decline the policy or select another
    schedule.

    Eager timing cannot rank the capture policy: the guard grants its
    overlap only inside a captured graph, so the knob reads as a tie, or as
    a loss by the capture test's host time on a host-bound kernel, and a
    search that steps to ``cute_pdl=False`` returns a config without the
    overlap the default has. The policy is searched only where the
    benchmark observes it
    (``ConfigSpec.cute_pdl_capture_overlap_searched``).

    A fixed override states the option it names and nothing else. While the
    overridden options still form a guarded policy (``cute_pdl`` and the
    overlap on, no spread padding, a release a capture launch can use), the
    options the overrides leave out keep their default: under
    ``cute_pdl_release="auto"`` a search cannot step off the overlap any
    more than without the override. An override the policy cannot keep
    declines it instead (``_decline_overridden_capture_policy``), and
    nothing is kept. Which overrides those are is decided by the rule
    normalization and the decline hook apply (``capture_overlap_eligible``),
    so no policy is kept that they reject.

    Three repairs apply the answer. A flat config is repaired wherever it is
    made or decoded (``CuteConfigGenerationExtension.repair_flat_config``).
    The overrides are layered over the decoded config afterwards, so a
    candidate that an override puts on the paired schedule is repaired once
    more there (``_keep_overridden_capture_policy``). A proposal from
    outside the flat fields, which is never decoded from a flat config, is
    repaired where its search admits it
    (``_hold_proposed_capture_policy``)."""
    if config_spec.cute_pdl_capture_overlap_searched:
        return None
    # The paired schedule and its policy as the overrides state them: kept
    # while that is still a guarded policy.
    stated = {
        key: overrides.get(key, value)
        for key, value in CAPTURE_POLICY_REQUIREMENTS.items()
    }
    if not carries_capture_overlap(capture_policy_available(config_spec), stated):
        return None
    return {key: stated[key] for key in CAPTURE_POLICY}


def key_clip_choices(
    config_spec: ConfigSpec, config: Mapping[str, object]
) -> tuple[str, ...]:
    """The key clips the schedule and value transport ``config`` selects can
    run, its default first."""
    facts = config_spec.cute_mma_dag_facts
    assert facts is not None
    return paged_attention_key_clip_choices(
        facts.attention,
        config.get(MMA_DAG_PASS_WARPS_KEY),
        config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY),
    )


# What a shifted key clip needs of a config: the options
# ``paged_attention_key_clip_choices`` reads.
_KEY_CLIP_REQUIREMENTS = MappingProxyType(
    {
        MMA_DAG_PASS_WARPS_KEY: MMA_DAG_PASS_WARPS_ATTENTION_PAIRED,
        MMA_DAG_VALUE_METADATA_SKIP_KEY: False,
    }
)


def _paired_by_override_alone(
    config: Mapping[str, object], overrides: Mapping[str, object]
) -> bool:
    """Whether ``overrides``, already layered over ``config``, are what put
    it on the paired schedule, as far as the candidate still shows.

    A candidate is decoded before the overrides are layered over it, and a
    row on another body has every paired knob pinned by then
    (``_pin_paired_knobs``). So an override that names the schedule, over a
    candidate whose paired knobs all hold their pins where the overrides
    leave them alone, marks a candidate that body pinned. A paired row that
    chose every pin itself reads the same; any other paired row differs in
    a knob."""
    return any(key in overrides for key in PAIRED_CAPTURE_SCHEDULE) and all(
        config.get(name, pin) == pin
        for name, pin in _PAIRED_KNOB_PINS
        if name not in overrides
    )


def rederive_overridden_key_clip(
    config_spec: ConfigSpec,
    config: dict[str, object],
    overrides: Mapping[str, object],
) -> None:
    """Drop the key clip of a candidate where ``overrides``, already layered
    over it, select options that cannot run it, or put the candidate on the
    paired schedule with the clip another body pinned.

    The clips a config may name follow its other options: only the paired
    loader shifts a tile, and value-metadata transport keeps whole tiles.
    Strict normalization treats the two differently. A clip named together
    with value metadata is rejected, like a clip the kernel offers under no
    schedule. A clip on another body is not an error: that body pins it to
    ``off``, as it pins every paired knob (``_pin_paired_knobs``).

    A complete candidate carries a clip that fits its own options; for a
    windowed kernel that is usually ``leading``, the default no compiler
    seed spells out. An override that turns value metadata on without
    naming a clip would leave that clip behind to be rejected: under
    ``cute_mma_dag_value_metadata_skip=True`` seed transfer dropped every
    paired seed of a windowed fast_math kernel, also where the same entry
    selects another schedule, because the transport check precedes the pin.
    An override of the schedule alone needs no help here: the pin gives the
    same config with the clip or without it.

    The clip that no longer fits is dropped, so normalization derives the
    default of the overridden config, as for a config that names the
    override and no clip. A clip the overridden config still admits stays,
    also where it was the default of the options the override replaced
    (whole tiles under an override that turns value metadata off), and an
    overridden clip is never dropped. A candidate that never admitted its
    clip is left for validation to reject where the options at fault are
    its own, those the overrides do not name. Where an override names the
    option at fault, the candidate's own value is gone by the time this
    hook runs, and the option is read as the value a shifted clip needs: a
    config that names a clip of both sides beside value metadata, an error
    on its own, loses the clip under the override
    ``cute_mma_dag_value_metadata_skip=True`` like a seed whose clip the
    override displaced.

    The other direction is an override that selects the paired schedule
    over a candidate of another body: a compiler seed of the plain attention
    schedule, a random row whose own schedule slot holds another value. That
    body pinned the clip to whole tiles, which the paired schedule admits,
    so nothing is rejected, but the pin would read as the candidate's
    choice, and every seed the override moves would start without the clip
    a windowed kernel defaults to. Where the candidate shows that the
    override alone put it there (``_paired_by_override_alone``), the pinned
    clip is dropped as well and the paired default is derived. A paired
    candidate that names another paired knob keeps the whole tiles it
    chose."""
    fragment = config_spec.cute_mma_dag_key_clip
    if fragment is None or MMA_DAG_KEY_CLIP_KEY in overrides:
        return
    clip = config.get(MMA_DAG_KEY_CLIP_KEY)
    if clip is None:
        return
    choices = key_clip_choices(config_spec, config)
    if clip in choices:
        if clip != choices[0] and _paired_by_override_alone(config, overrides):
            del config[MMA_DAG_KEY_CLIP_KEY]
        return
    # The candidate as it was where an override now sits: it carried the
    # clip only if its own options satisfy what a shifted clip needs.
    carried = {
        **config,
        **{
            name: value
            for name, value in _KEY_CLIP_REQUIREMENTS.items()
            if name in overrides
        },
    }
    if clip in key_clip_choices(config_spec, carried):
        del config[MMA_DAG_KEY_CLIP_KEY]


def _normalize_cute_mma_dag(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Validate the warp-MMA dataflow knobs against the selected tiles
    and pin the knobs the replaced body ignores to one identity."""
    from ..cute_mma_dag import mma_dag_config_reason
    from ..cute_mma_dag import mma_dag_repair
    from ..paged_attention_config import paged_attention_config_reason

    facts = config_spec.cute_mma_dag_facts
    assert facts is not None
    for key, fragment in (
        (MMA_DAG_KEY, config_spec.cute_mma_dag),
        (MMA_DAG_WARPS_KEY, config_spec.cute_mma_dag_warps),
        (MMA_DAG_STAGES_KEY, config_spec.cute_mma_dag_stages),
        (MMA_DAG_STORE_KEY, config_spec.cute_mma_dag_store),
        (MMA_DAG_EXCHANGE_KEY, config_spec.cute_mma_dag_exchange),
        (MMA_DAG_PASS_WARPS_KEY, config_spec.cute_mma_dag_pass_warps),
        (MMA_DAG_LOADS_KEY, config_spec.cute_mma_dag_loads),
        (MMA_DAG_AUX_KEY, config_spec.cute_mma_dag_aux),
        (MMA_DAG_DERIVED_KEY, config_spec.cute_mma_dag_derived),
        (MMA_DAG_EPILOGUE_KEY, config_spec.cute_mma_dag_epilogue),
        (MMA_DAG_LOOKAHEAD_KEY, config_spec.cute_mma_dag_lookahead),
    ):
        if fragment is None:
            # The aux / epilogue knob has no subject here: inert, no
            # choices.
            assert key in (MMA_DAG_AUX_KEY, MMA_DAG_EPILOGUE_KEY)
            config[key] = (
                MMA_DAG_AUX_CHOICES[0]
                if key == MMA_DAG_AUX_KEY
                else MMA_DAG_EPILOGUE_ROW
            )
            continue
        value = config.setdefault(key, fragment.default())
        # The store knob accepts ``tma`` by name, off the search fragment
        # until its effect is measured.
        accepted = (
            MMA_DAG_STORE_ACCEPTED if key == MMA_DAG_STORE_KEY else fragment.choices
        )
        if value not in accepted or type(value) is bool:
            if fix_invalid:
                config[key] = fragment.default()
            else:
                raise InvalidConfig(
                    f"{key} must be one of {fragment.choices!r}, got {value!r}"
                )
    for key, fragment, default in (
        (MMA_DAG_V_LAG_KEY, config_spec.cute_mma_dag_v_lag, 0),
        (
            MMA_DAG_PROBABILITY_SCHEDULE_KEY,
            config_spec.cute_mma_dag_probability_schedule,
            MMA_DAG_PROBABILITY_SCHEDULE_SERIAL,
        ),
    ):
        if fragment is not None or key in config:
            value = config.setdefault(key, default)
            allowed = (default,) if fragment is None else fragment.choices
            if type(value) is not type(default) or value not in allowed:
                if not fix_invalid:
                    raise InvalidConfig(
                        f"{key} must be one of {allowed!r}, got {value!r}"
                    )
                config[key] = default
            if fragment is None:
                config.pop(key, None)
    clip_fragment = config_spec.cute_mma_dag_key_clip
    if clip_fragment is not None or MMA_DAG_KEY_CLIP_KEY in config:
        # A named clip must be one the paired loader runs with this config's
        # value transport (the value-metadata actor copies unshifted scale
        # words: whole tiles by default, and a named clip is rejected). A
        # body other than the paired one then pins it to off, like every
        # paired knob (``_pin_paired_knobs``).
        clips = paged_attention_key_clip_choices(
            facts.attention,
            MMA_DAG_PASS_WARPS_ATTENTION_PAIRED,
            config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY),
        )
        clip = config.setdefault(MMA_DAG_KEY_CLIP_KEY, clips[0])
        if type(clip) is not str or clip not in clips:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{MMA_DAG_KEY_CLIP_KEY} must be one of {clips!r}, got {clip!r}"
                )
            config[MMA_DAG_KEY_CLIP_KEY] = clips[0]
        if clip_fragment is None:
            config.pop(MMA_DAG_KEY_CLIP_KEY, None)
    # The paired schedule retires its allocation early unless the config says
    # otherwise; every other body has no such phase.
    retires_early = paged_attention_early_retirement_default(
        config.get(MMA_DAG_PASS_WARPS_KEY)
    )
    for key, fragment, default in (
        (MMA_DAG_SLEEP_WAITS_KEY, config_spec.cute_mma_dag_sleep_waits, False),
        (
            MMA_DAG_EARLY_TMEM_RETIREMENT_KEY,
            config_spec.cute_mma_dag_early_tmem_retirement,
            retires_early,
        ),
        (MMA_DAG_QUERY_TMA_KEY, config_spec.cute_mma_dag_query_tma, False),
    ):
        if fragment is not None or key in config:
            value = config.setdefault(key, default)
            if type(value) is not bool or (fragment is None and value):
                if not fix_invalid:
                    raise InvalidConfig(
                        f"{key} needs a bool and a supported paired transport, got {value!r}"
                    )
                # What leaving the key out gives. Without the fragment the
                # option is off, and the key is dropped below.
                config[key] = False if fragment is None else default
            if fragment is None:
                config.pop(key, None)
    persistent_fragment = config_spec.cute_mma_dag_persistent
    persistent = config.setdefault(MMA_DAG_PERSISTENT_KEY, 0)
    allowed = (0,) if persistent_fragment is None else persistent_fragment.choices
    if persistent not in allowed or type(persistent) is not int:
        if not fix_invalid:
            raise InvalidConfig(
                f"{MMA_DAG_PERSISTENT_KEY} must be one of {allowed!r}, got {persistent!r}"
            )
        config[MMA_DAG_PERSISTENT_KEY] = 0
    if config[MMA_DAG_KEY] != MMA_DAG_TCGEN05 and config[MMA_DAG_PERSISTENT_KEY]:
        if not fix_invalid:
            raise InvalidConfig(f"{MMA_DAG_PERSISTENT_KEY} needs the tcgen05 mode")
        config[MMA_DAG_PERSISTENT_KEY] = 0
    cluster_fragment = config_spec.cute_mma_dag_cluster
    cluster = config.setdefault(MMA_DAG_CLUSTER_KEY, 1)
    allowed = (1,) if cluster_fragment is None else cluster_fragment.choices
    if cluster not in allowed or type(cluster) is not int:
        if not fix_invalid:
            raise InvalidConfig(
                f"{MMA_DAG_CLUSTER_KEY} must be one of {allowed!r}, got {cluster!r}"
            )
        config[MMA_DAG_CLUSTER_KEY] = 1
    if config[MMA_DAG_KEY] != MMA_DAG_TCGEN05 and config[MMA_DAG_CLUSTER_KEY] != 1:
        if not fix_invalid:
            raise InvalidConfig(f"{MMA_DAG_CLUSTER_KEY} needs the tcgen05 mode")
        config[MMA_DAG_CLUSTER_KEY] = 1
    split_fragment = config_spec.cute_mma_dag_split
    split = config.setdefault(MMA_DAG_SPLIT_KEY, MMA_DAG_SPLIT_CLUSTER)
    split_allowed = (
        (MMA_DAG_SPLIT_CLUSTER,) if split_fragment is None else split_fragment.choices
    )
    if split not in split_allowed:
        if not fix_invalid:
            raise InvalidConfig(
                f"{MMA_DAG_SPLIT_KEY} must be one of {split_allowed!r}, got {split!r}"
            )
        config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
    if (
        config[MMA_DAG_SPLIT_KEY] == MMA_DAG_SPLIT_CLUSTER
        and cast("int", config[MMA_DAG_CLUSTER_KEY]) > MMA_DAG_CLUSTER_MAX
    ):
        # A portable thread-block cluster holds eight CTAs: a wider split
        # combines its rows through the fixup grid (offered only where
        # the program admits it; the cluster choices stop at eight
        # elsewhere, so this is the admitted case).
        if not fix_invalid:
            raise InvalidConfig(
                f"{MMA_DAG_CLUSTER_KEY}={config[MMA_DAG_CLUSTER_KEY]} needs "
                f"{MMA_DAG_SPLIT_KEY}={MMA_DAG_SPLIT_FIXUP!r} (a portable cluster "
                f"holds at most {MMA_DAG_CLUSTER_MAX} CTAs)"
            )
        config[MMA_DAG_SPLIT_KEY] = (
            MMA_DAG_SPLIT_FIXUP if split_fragment is not None else MMA_DAG_SPLIT_CLUSTER
        )
        if split_fragment is None:
            config[MMA_DAG_CLUSTER_KEY] = 1
    if config[MMA_DAG_KEY] == MMA_DAG_OFF:
        # The warp count, stage depth and data routes are inert without
        # the mode.
        config[MMA_DAG_WARPS_KEY] = MMA_DAG_DEFAULT_WARPS
        config[MMA_DAG_STAGES_KEY] = MMA_DAG_DEFAULT_STAGES
        config[MMA_DAG_STORE_KEY] = MMA_DAG_STORE_CHOICES[0]
        config[MMA_DAG_EXCHANGE_KEY] = MMA_DAG_EXCHANGE_CHOICES[0]
        config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
        _pin_paired_knobs(config)
        config[MMA_DAG_LOADS_KEY] = MMA_DAG_LOADS_CHOICES[0]
        config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
        config[MMA_DAG_DERIVED_KEY] = MMA_DAG_DERIVED_CHOICES[0]
        config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_ROW
        config[MMA_DAG_LOOKAHEAD_KEY] = MMA_DAG_LOOKAHEAD_OFF
        config[MMA_DAG_PERSISTENT_KEY] = 0
        config[MMA_DAG_CLUSTER_KEY] = 1
        config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
        return
    # The derived route has a subject only where some dot operand is a
    # pass chain, and its pass reads the chain's tiles from the ring:
    # the tensor-memory mode's ``lane`` loads leave them to the lanes.
    if not facts.pass_chains or (
        config[MMA_DAG_KEY] == MMA_DAG_TCGEN05
        and config[MMA_DAG_LOADS_KEY] != MMA_DAG_LOADS_TMA
    ):
        config[MMA_DAG_DERIVED_KEY] = MMA_DAG_DERIVED_CHOICES[0]
    derived = cast("str", config[MMA_DAG_DERIVED_KEY])
    block_sizes = config.get("block_sizes")
    assert isinstance(block_sizes, list)
    indices = {
        axis.block_id: config_spec.block_sizes.block_id_to_index(axis.block_id)
        for axis in facts.axes
        if axis.kind in (AXIS_ROOT, AXIS_LOOP, AXIS_LEAD)
        and axis.block_id in config_spec.block_sizes.valid_block_ids()
    }
    # Leading root tiles are batch coordinates: one row per CTA.
    for axis in facts.axes:
        if axis.kind == AXIS_LEAD and axis.block_id in indices:
            block_sizes[indices[axis.block_id]] = 1
    sizes_by_block: dict[int, object] = {
        block_id: block_sizes[index] for block_id, index in indices.items()
    }
    warps, stages = config[MMA_DAG_WARPS_KEY], config[MMA_DAG_STAGES_KEY]
    mode = cast("str", config[MMA_DAG_KEY])
    reason = apply_mma_dag_cluster(
        facts,
        sizes_by_block,
        config[MMA_DAG_CLUSTER_KEY],
        config[MMA_DAG_SPLIT_KEY],
    )
    if reason is not None:
        if not fix_invalid:
            raise InvalidConfig(f"{MMA_DAG_KEY}={mode!r}: {reason}")
        config[MMA_DAG_CLUSTER_KEY] = 1
    # The MMA warp (``cute_mma_dag_pass_warps='mma'``) has a subject
    # only in the tensor-memory region programs with a streamed K loop
    # whose operand tiles take a pass (a pass chain's under the pass
    # route); elsewhere the knob is inert and pinned off here, ahead of
    # the tile check (the warp's barriers count in the shared memory).
    # (``on`` is pinned below, once the tile is settled.)
    if config[MMA_DAG_PASS_WARPS_KEY] == MMA_DAG_PASS_WARPS_MMA and (
        mma_dag_pass_warps_reason(
            facts,
            cast("int", warps),
            cast("int", stages),
            mode,
            MMA_DAG_PASS_WARPS_MMA,
            derived,
        )
        is not None
    ):
        config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
    pass_warps = cast("str", config[MMA_DAG_PASS_WARPS_KEY])
    if pass_warps != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED:
        _pin_paired_knobs(config)
    # The split epilogue has a subject only in the tensor-memory region
    # programs without reductions, at eight warps; elsewhere the knob is
    # inert and pinned to the row layout.
    if (
        config[MMA_DAG_EPILOGUE_KEY] == MMA_DAG_EPILOGUE_SPLIT
        and mma_dag_epilogue_reason(facts, warps, mode) is not None
    ):
        config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_ROW
    epilogue = cast("str", config[MMA_DAG_EPILOGUE_KEY])
    # The lookahead schedule has a subject only in the region programs
    # whose one ring loop is a fresh dot and the derived values over it.
    if (
        config[MMA_DAG_LOOKAHEAD_KEY] == MMA_DAG_LOOKAHEAD_ON
        and mma_dag_lookahead_reason(facts, stages, mode) is not None
    ):
        config[MMA_DAG_LOOKAHEAD_KEY] = MMA_DAG_LOOKAHEAD_OFF
    lookahead = cast("str", config[MMA_DAG_LOOKAHEAD_KEY])
    # The aux ring has a subject only in the warp mode with a side load
    # varying with an innermost ring loop of the bound inputs; elsewhere
    # the knob is inert and pinned to the register prefetch.
    if (
        config[MMA_DAG_AUX_KEY] == MMA_DAG_AUX_RING
        and mma_dag_aux_reason(facts, mode) is not None
    ):
        config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
    aux = cast("str", config[MMA_DAG_AUX_KEY])
    # The persistent launch has a subject only in the tensor-memory mode
    # (every single-loop program; the region programs whose root loops
    # have static trips, under the program-order schedule); elsewhere
    # the knob is inert and pinned to one tile per CTA.
    if (
        config[MMA_DAG_PERSISTENT_KEY]
        and mma_dag_persistent_reason(facts, mode, lookahead) is not None
    ):
        config[MMA_DAG_PERSISTENT_KEY] = 0
    persistent = cast("int", config[MMA_DAG_PERSISTENT_KEY])
    if config[MMA_DAG_PASS_WARPS_KEY] in MMA_DAG_ATTENTION_PASS_WARPS:
        attention_reason = paged_attention_config_reason(
            facts, sizes_by_block, config, before_inert_globals=True
        )
        if attention_reason is not None:
            if not fix_invalid:
                raise InvalidConfig(attention_reason)
            config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
            pass_warps = MMA_DAG_PASS_WARPS_OFF
    reason = mma_dag_config_reason(
        facts,
        sizes_by_block,
        warps,
        stages,
        mode,
        derived,
        epilogue=epilogue,
        lookahead=lookahead,
        aux=aux,
        persistent=persistent > 0,
        pass_warps=pass_warps,
    )
    # A refused tile sheds knobs before the tile itself is repaired
    # (``_MMA_DAG_KNOB_DROPS``): the first knob whose probe admits the
    # tile is reset; the tile stays.
    knobs: dict[str, object] = {
        MMA_DAG_DERIVED_KEY: derived,
        MMA_DAG_EPILOGUE_KEY: epilogue,
        MMA_DAG_LOOKAHEAD_KEY: lookahead,
        MMA_DAG_AUX_KEY: aux,
        MMA_DAG_PERSISTENT_KEY: persistent > 0,
        MMA_DAG_PASS_WARPS_KEY: pass_warps,
    }
    for key, on_value, dropped, resets in _MMA_DAG_KNOB_DROPS:
        if reason is None:
            break
        value = config[key]
        shed = value == on_value if on_value is not None else bool(value)
        if not shed:
            continue
        probe = {
            **knobs,
            **{knob: _MMA_DAG_KNOB_PROBE_DEFAULTS[knob] for knob in dropped},
        }
        if (
            mma_dag_config_reason(
                facts,
                sizes_by_block,
                warps,
                stages,
                mode,
                cast("str", probe[MMA_DAG_DERIVED_KEY]),
                epilogue=cast("str", probe[MMA_DAG_EPILOGUE_KEY]),
                lookahead=cast("str", probe[MMA_DAG_LOOKAHEAD_KEY]),
                aux=cast("str", probe[MMA_DAG_AUX_KEY]),
                persistent=cast("bool", probe[MMA_DAG_PERSISTENT_KEY]),
                pass_warps=cast("str", probe[MMA_DAG_PASS_WARPS_KEY]),
            )
            is not None
        ):
            continue
        if not fix_invalid:
            raise InvalidConfig(f"{key}={value!r}: {reason}")
        config.update(resets)
        reason = None
    derived = cast("str", config[MMA_DAG_DERIVED_KEY])
    if reason is not None:
        if not fix_invalid:
            raise InvalidConfig(f"{MMA_DAG_KEY}={mode!r}: {reason}")
        repaired = mma_dag_repair(facts, sizes_by_block, warps, stages, mode)
        if repaired is None and mode == MMA_DAG_TCGEN05:
            # No tcgen05 geometry near the request: the warp mode runs
            # the same program.
            config[MMA_DAG_KEY] = MMA_DAG_WARP
            repaired = mma_dag_repair(facts, sizes_by_block, warps, stages)
        if repaired is None:
            config[MMA_DAG_KEY] = MMA_DAG_OFF
            _pin_paired_knobs(config)
            config[MMA_DAG_WARPS_KEY] = MMA_DAG_DEFAULT_WARPS
            config[MMA_DAG_STAGES_KEY] = MMA_DAG_DEFAULT_STAGES
            config[MMA_DAG_STORE_KEY] = MMA_DAG_STORE_CHOICES[0]
            config[MMA_DAG_EXCHANGE_KEY] = MMA_DAG_EXCHANGE_CHOICES[0]
            config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
            config[MMA_DAG_LOADS_KEY] = MMA_DAG_LOADS_CHOICES[0]
            config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
            config[MMA_DAG_DERIVED_KEY] = MMA_DAG_DERIVED_CHOICES[0]
            config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_ROW
            config[MMA_DAG_LOOKAHEAD_KEY] = MMA_DAG_LOOKAHEAD_OFF
            config[MMA_DAG_PERSISTENT_KEY] = 0
            config[MMA_DAG_CLUSTER_KEY] = 1
            config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
            return
        new_sizes, config[MMA_DAG_WARPS_KEY], config[MMA_DAG_STAGES_KEY] = repaired
        # (The repair searched under the row epilogue, in program order,
        # with the register prefetch, and -- on a region program, whose
        # persistent layout is larger -- one tile per CTA.)
        config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_ROW
        config[MMA_DAG_LOOKAHEAD_KEY] = MMA_DAG_LOOKAHEAD_OFF
        config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
        if facts.tcgen05 is not None and facts.tcgen05.regions:
            config[MMA_DAG_PERSISTENT_KEY] = 0
        for block_id, index in indices.items():
            block_sizes[index] = new_sizes[block_id]
        sizes_by_block = dict(new_sizes)
        # The repaired split axis slice names the split count (1 when
        # the repair kept the whole axis, or fell back to the warp mode);
        # beyond a portable cluster it is the fixup grid's.
        config[MMA_DAG_CLUSTER_KEY] = facts.cluster_of(new_sizes)
        if cast("int", config[MMA_DAG_CLUSTER_KEY]) > MMA_DAG_CLUSTER_MAX:
            config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_FIXUP
        if derived == MMA_DAG_DERIVED_PASS and (
            mma_dag_config_reason(
                facts,
                new_sizes,
                config[MMA_DAG_WARPS_KEY],
                config[MMA_DAG_STAGES_KEY],
                cast("str", config[MMA_DAG_KEY]),
                derived,
            )
            is not None
        ):
            # (The repair searched under the exchange route.)
            config[MMA_DAG_DERIVED_KEY] = MMA_DAG_DERIVED_CHOICES[0]
    if config[MMA_DAG_CLUSTER_KEY] == 1:
        # The whole axis in one CTA: the combine mode has no subject.
        config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
    elif (
        config[MMA_DAG_SPLIT_KEY] == MMA_DAG_SPLIT_FIXUP
        and apply_mma_dag_cluster(
            facts,
            {},
            config[MMA_DAG_CLUSTER_KEY],
            MMA_DAG_SPLIT_FIXUP,
        )
        is not None
    ):
        # The repaired split does not admit the fixup form (a program
        # with a leading axis or two root axes, or too many tiles): the
        # cluster form takes a portable count, else the whole axis.
        if cast("int", config[MMA_DAG_CLUSTER_KEY]) <= MMA_DAG_CLUSTER_MAX:
            config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
        else:
            config[MMA_DAG_CLUSTER_KEY] = 1
            config[MMA_DAG_SPLIT_KEY] = MMA_DAG_SPLIT_CLUSTER
    # Knobs without a subject are inert: no fragment read straight as an
    # A operand (only the shared tile can feed a transposed one), no loop
    # snapshot to stage, no device loop whose ring the stage depth sizes.
    # (The tensor-memory region programs give the exchange knob a second
    # subject: the fp32 A operands read from tensor memory and the
    # ``X @ X`` dots sharing one tile.)
    tcgen05_exchange = (
        config[MMA_DAG_KEY] == MMA_DAG_TCGEN05
        and facts.tcgen05 is not None
        and facts.tcgen05.tmem_operands > 0
    )
    if not facts.identity_candidates and not tcgen05_exchange:
        config[MMA_DAG_EXCHANGE_KEY] = MMA_DAG_EXCHANGE_CHOICES[0]
    if not facts.staging and not facts.element_staging:
        config[MMA_DAG_STORE_KEY] = MMA_DAG_STORE_CHOICES[0]
    if not facts.aux_candidates:
        config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
    if not facts.loops:
        config[MMA_DAG_STAGES_KEY] = MMA_DAG_DEFAULT_STAGES
    if config[MMA_DAG_PASS_WARPS_KEY] != MMA_DAG_PASS_WARPS_OFF and (
        mma_dag_pass_warps_reason(
            facts,
            cast("int", config[MMA_DAG_WARPS_KEY]),
            cast("int", config[MMA_DAG_STAGES_KEY]),
            cast("str", config[MMA_DAG_KEY]),
            cast("str", config[MMA_DAG_PASS_WARPS_KEY]),
            cast("str", config[MMA_DAG_DERIVED_KEY]),
        )
        is not None
    ):
        config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
    if config[MMA_DAG_PASS_WARPS_KEY] != MMA_DAG_PASS_WARPS_ATTENTION_PAIRED:
        _pin_paired_knobs(config)
    # The loads route has a subject in the tensor-memory region programs
    # whose derived values read tiles, and in the warp mode when a ring
    # tile's source is one a tensor map can fetch or an epilogue value
    # reads a loop-invariant tile (staged ahead of the loops).
    if config[MMA_DAG_KEY] == MMA_DAG_TCGEN05:
        loads_subject = facts.tcgen05 is not None and bool(facts.tcgen05.aux_items)
    else:
        loads_subject = facts.tma_ring or facts.fixed_derived
    if not loads_subject:
        config[MMA_DAG_LOADS_KEY] = MMA_DAG_LOADS_CHOICES[0]
    if config[MMA_DAG_KEY] == MMA_DAG_TCGEN05:
        config[MMA_DAG_AUX_KEY] = MMA_DAG_AUX_CHOICES[0]
    else:
        # The epilogue form is the tensor-memory region renderer's.
        config[MMA_DAG_EPILOGUE_KEY] = MMA_DAG_EPILOGUE_ROW
    # Inert knobs: the body is replaced, so equal kernels share one
    # autotune identity (the occupancy hint stays searchable).  The
    # persistent tcgen05 launch is the generic persistent_interleaved
    # program ids at its CTAs per SM.
    if config.get(MMA_DAG_KEY) == MMA_DAG_TCGEN05 and config[MMA_DAG_PERSISTENT_KEY]:
        config["pid_type"] = "persistent_interleaved"
        config["num_sm_multiplier"] = config[MMA_DAG_PERSISTENT_KEY]
    else:
        config[MMA_DAG_PERSISTENT_KEY] = 0
        config["pid_type"] = "flat"
        config.pop("num_sm_multiplier", None)
    policies = config.get("load_eviction_policies")
    if config[MMA_DAG_PASS_WARPS_KEY] in MMA_DAG_ATTENTION_PASS_WARPS and isinstance(
        policies, str
    ):
        config["load_eviction_policies"] = [
            policies
        ] * config_spec.load_eviction_policies.length
    if config_spec.cute_mma_dag_phase_index is not None:
        _normalize_mma_dag_phase_owned_fields(config_spec, config)
    else:
        _normalize_mma_dag_owned_fields(config_spec, config)
    if config[MMA_DAG_PASS_WARPS_KEY] in MMA_DAG_ATTENTION_PASS_WARPS:
        attention_config = config
        policies = config.get("load_eviction_policies")
        if config_spec.cute_mma_dag_phase_index is not None and isinstance(
            policies, list
        ):
            # Sibling launches retain their own policies. Only this phase's
            # projected slots belong to the attention schedule's contract.
            attention_config = {
                **config,
                "load_eviction_policies": project_load_policies(
                    policies, config_spec.cute_mma_dag_phase_eviction_map
                ),
            }
        attention_reason = paged_attention_config_reason(
            facts, sizes_by_block, attention_config
        )
        if attention_reason is not None:
            if not fix_invalid:
                raise InvalidConfig(attention_reason)
            config[MMA_DAG_PASS_WARPS_KEY] = MMA_DAG_PASS_WARPS_OFF
            # The ordinary layout has a different resource budget. A late
            # topology repair must validate that body before returning it.
            _normalize_cute_mma_dag(config_spec, config, fix_invalid=True)
            return


def _normalize_mma_dag_phase_owned_fields(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    """Canonicalize only knobs no sibling launch consumes."""
    owned = config_spec.cute_mma_dag_phase_owned_blocks
    for key, sequence in (
        ("num_threads", config_spec.num_threads),
        ("loop_orders", config_spec.loop_orders),
        ("l2_groupings", config_spec.l2_groupings),
        ("cute_vector_widths", config_spec.cute_vector_widths),
        ("cute_lane_layouts", config_spec.cute_lane_layouts),
        ("cute_reduction_reloads", config_spec.cute_reduction_reloads),
        ("reduction_loops", config_spec.reduction_loops),
        ("flatten_loops", config_spec.flatten_loops),
        ("range_unroll_factors", config_spec.range_unroll_factors),
        ("range_warp_specializes", config_spec.range_warp_specialize),
        ("range_num_stages", config_spec.range_num_stages),
        ("range_multi_buffers", config_spec.range_multi_buffers),
        ("range_flattens", config_spec.range_flattens),
        ("static_ranges", config_spec.static_ranges),
    ):
        values = config.get(key)
        if isinstance(values, list):
            config[key] = [
                item._fill_missing() if set(item.block_ids) <= owned else value
                for item, value in zip(sequence, values, strict=True)
            ]
    policies = config.get("load_eviction_policies")
    if isinstance(policies, list):
        mapping = config_spec.cute_mma_dag_phase_eviction_map
        local_policies = project_load_policies(policies, mapping)
        projected = {**config, "load_eviction_policies": local_policies}
        _normalize_mma_dag_owned_fields(config_spec, projected)
        canonical = cast("list[str]", projected["load_eviction_policies"])
        policies = list(policies)
        for global_slot in config_spec.cute_mma_dag_phase_load_slots:
            policies[global_slot] = ""
        config["load_eviction_policies"] = scatter_load_policies(
            canonical, mapping, policies
        )
        config[MMA_DAG_LOADS_KEY] = projected[MMA_DAG_LOADS_KEY]


def _normalize_mma_dag_owned_fields(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    """Pin layout knobs only for a launch whose body MMA-DAG replaces."""
    facts = config_spec.cute_mma_dag_facts
    assert facts is not None
    if "loop_orders" in config:
        config["loop_orders"] = [
            item._fill_missing() for item in config_spec.loop_orders
        ]
    if "l2_groupings" in config:
        config["l2_groupings"] = [1 for _ in config_spec.l2_groupings]
    for key, default in (
        ("num_threads", 0),
        ("cute_vector_widths", 1),
        ("cute_lane_layouts", "blocked"),
        ("cute_reduction_reloads", "auto"),
        ("reduction_loops", None),
        ("flatten_loops", False),
    ):
        value = config.get(key)
        if isinstance(value, list):
            config[key] = [default for _ in value]
    # The warp mode's staged tiles carry the L2 policy of their slot
    # (``MMA_DAG_L2_HINTS``) on their cp.async packets, one canonical
    # value per packet form; other slots are inert, except the explicit
    # attention schedule's TMA raw/scale loads handled below. Hinted packets exist only
    # where the launcher can verify their assembly
    # (``hinted_cp_async_available``): elsewhere every slot pins to "".
    policies = config.get("load_eviction_policies")
    if isinstance(policies, list):
        staged = (
            set(facts.staged_slots)
            if config[MMA_DAG_KEY] == MMA_DAG_WARP and hinted_cp_async_available()
            else set()
        )
        if (
            config.get(MMA_DAG_PASS_WARPS_KEY) in MMA_DAG_ATTENTION_PASS_WARPS
            and facts.attention is not None
        ):
            # This schedule issues TMA directly; only its original raw/scale
            # load slots consume cache hints. Phase projection stays local.
            staged = {
                slot for slot in facts.attention.tma_load_slots if slot is not None
            }
        canonical = [
            MMA_DAG_L2_CANONICAL.get(
                MMA_DAG_L2_HINTS.get(policy, "") if isinstance(policy, str) else "",
                "",
            )
            if slot in staged
            else ""
            for slot, policy in enumerate(policies)
        ]
        config["load_eviction_policies"] = canonical
        # A hinted tile keeps its cp.async packets under either loads
        # route (``plan_tma``): a ``lane`` config hinting every tile a
        # tensor map could fetch renders the ``tma`` config's kernel
        # when no fixed epilogue tile tells them apart -- one identity.
        if (
            config[MMA_DAG_KEY] == MMA_DAG_WARP
            and config.get(MMA_DAG_LOADS_KEY) == MMA_DAG_LOADS_LANE
            and facts.tma_slots
            and not facts.fixed_derived
            and all(
                slot < len(canonical) and canonical[slot] for slot in facts.tma_slots
            )
        ):
            config[MMA_DAG_LOADS_KEY] = MMA_DAG_LOADS_TMA
    for key in ("cute_collective_mma", "cute_register_chain", "cute_simt_mma"):
        if key in config:
            config[key] = False
    for key, field in config_spec._flat_fields().items():
        if (
            key.startswith(("cute_fragment_", "cute_collective_", "cute_gathered_mma_"))
            or key == "cute_proven_bounds"
        ) and key in config:
            if isinstance(field, ConfigSpecFragment):
                config[key] = field.default()
