"""The CuTe flash-attention search surface: fragments, normalization and seeds.

The forward (``cute_flash_*``), gated (``cute_flash_gate_*``) and backward
(``cute_flash_bwd_*``) families' contribution to ``ConfigSpec``: how their
knob fragments are built from the facts ``enable_cute_flash_search`` and its
siblings record on the spec, how a config is legalized against them and which
compiler seeds they plant.  Every function takes the ``ConfigSpec`` it used to
be a method of; ``search_surface.CuteSearchSurface`` calls them at the phase
positions ``ConfigSpec`` runs."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

from ....exc import InvalidConfig
from ....runtime.config import Config
from ..cute_flash import FLASH_CAUSAL_LPT_SWIZZLE_KEY
from ..cute_flash import FLASH_CORR_REGS_KEY
from ..cute_flash import FLASH_E2E_OFFSET0_KEY
from ..cute_flash import FLASH_E2E_OFFSET_KEY
from ..cute_flash import FLASH_E2E_SCHEDULE_KEY
from ..cute_flash import FLASH_EPI_STG_GMEM_KEY
from ..cute_flash import FLASH_EPI_STG_KEY
from ..cute_flash import FLASH_EPI_STG_STORE_KEY
from ..cute_flash import FLASH_EPI_TMA_KEY
from ..cute_flash import FLASH_EXP2_PACKET_KEY
from ..cute_flash import FLASH_KV_STAGE_KEY
from ..cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
from ..cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
from ..cute_flash import FLASH_MMA_INTERLEAVE_KEY
from ..cute_flash import FLASH_OTHER_REGS_KEY
from ..cute_flash import FLASH_PERSISTENT_KEY
from ..cute_flash import FLASH_PIPELINE_FAMILY_KEY
from ..cute_flash import FLASH_ROWMAX_KEY
from ..cute_flash import FLASH_SOFTMAX_LOWERING_KEY
from ..cute_flash import FLASH_SOFTMAX_REGS_KEY
from ..cute_flash import FLASH_TOPOLOGY_KEY
from ..cute_flash import ROW_MMA_FAMILY
from ..cute_flash import FlashAttentionConfig
from ..cute_flash import _flash_e2e_offset_period
from ..cute_flash import _flash_e2e_schedule_default
from ..cute_flash import _flash_masked_e2e_schedule_params
from ..cute_flash import _flash_normalize_e2e_offset
from ..cute_flash import _flash_normalize_e2e_params
from ..cute_flash import _flash_parse_e2e_schedule
from ..cute_flash import _flash_pipeline_family_flags
from ..cute_flash import _flash_resident_softmax_overrides
from ..cute_flash import flash_compound_exp2_packet_overrides
from ..cute_flash import flash_config_from_facts
from ..cute_flash import flash_effective_config_values
from ..cute_flash import flash_exp2_packet_is_compound
from ..cute_flash import flash_seeds_from_facts
from ..cute_flash_bwd_row_mma import FLASH_BWD_FAMILY_KEY
from ..cute_flash_bwd_row_mma import FLASH_BWD_ROW_FAMILY
from ..cute_flash_bwd_row_mma import FlashBwdRowFacts
from ..cute_flash_bwd_row_mma import flash_bwd_row_legalize
from ..cute_flash_bwd_row_mma import flash_bwd_row_search_grid
from ..flash_facts import FlashEnv
from ..flash_knobs import FLASH_BWD_AUX_SMEM_KEY
from ..flash_knobs import FLASH_BWD_DS_TMEM_KEY
from ..flash_knobs import FLASH_BWD_KV_STAGE_KEY
from ..flash_knobs import FLASH_BWD_PERSISTENT_KEY
from ..flash_knobs import FLASH_BWD_REGS_KEY
from ..flash_knobs import FLASH_BWD_S_FIRST_KEY
from ..flash_knobs import FLASH_BWD_STATS_MMA_KEY
from ..flash_knobs import FLASH_BWD_TABLE_STEPS_KEY
from ..flash_knobs import FLASH_BWD_TWO_CTA_KEY
from ..flash_knobs import FLASH_LEGACY_E2E_CONFIG_KEYS
from ..flash_knobs import flash_legacy_e2e_params
from ..flash_knobs import fold_flash_ws_inactive_values
from ..flash_knobs import upgrade_legacy_flash_config
from ..tcgen05_config import surface

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ....autotuner.config_fragment import ConfigSpecFragment
    from ....autotuner.config_fragment import EnumFragment
    from ....autotuner.config_spec import ConfigSpec


def _cute_flash_autotune_fragments(
    config_spec: ConfigSpec,
    topology_override: str | None = None,
    pipeline_family_override: str | None = None,
) -> dict[str, ConfigSpecFragment]:
    """Memoized ``flash_fragments_from_facts`` for this spec's flash surface.

    The facts (the knob channel included) key the memo, so a spec whose
    facts a test adjusts never sees stale fragments. The function is
    resolved through the module attribute on every miss so tests that
    ``patch.object`` it still intercept the real computation.
    """
    # Fragment values are shared: they are immutable dataclasses that all
    # callers replace rather than mutate. The mapping is copied because
    # callers merge it into their own field dicts.
    facts = config_spec.flash_facts
    key = (facts, topology_override, pipeline_family_override)
    cached = config_spec._cute_flash_fragments_cache.get(key)
    if cached is None:
        from ..cute_flash import flash_fragments_from_facts

        assert facts.num_kv is not None
        cached = flash_fragments_from_facts(
            facts,
            topology_override=topology_override,
            pipeline_family_override=pipeline_family_override,
            target_device_capability=config_spec.target_device_capability,
        )
        config_spec._cute_flash_fragments_cache[key] = cached
    return dict(cached)


def _resolve_cute_flash_config(
    config_spec: ConfigSpec, config: Mapping[str, object]
) -> FlashAttentionConfig:
    facts = config_spec.flash_facts
    assert facts.num_kv is not None
    flash_config = config
    if facts.requires_ws_overlap:
        flash_config = {**config, FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    return flash_config_from_facts(
        facts,
        flash_config,
        prefer_packed_reduce=(facts.has_kv_tile_pruning or facts.requires_ws_overlap),
        target_device_capability=config_spec.target_device_capability,
    )


def _resolve_cute_flash_pins(
    config_spec: ConfigSpec, pins: tuple[tuple[str, object], ...]
) -> FlashAttentionConfig:
    """``_resolve_cute_flash_config`` of a few pinned keys, memoized.

    Override validation resolves the same pinned family or packet for
    every config a generation produces.  The result depends only on the
    pins, this spec's flash facts (which carry the knob channel captured at
    bind time) and the target capability; ``enable_cute_flash_search``
    resets the memo.
    """
    key = (pins, config_spec.flash_facts, config_spec.target_device_capability)
    resolved = config_spec._cute_flash_pin_resolutions.get(key)
    if resolved is None:
        resolved = _resolve_cute_flash_config(config_spec, dict(pins))
        config_spec._cute_flash_pin_resolutions[key] = resolved
    return resolved


def _cute_flash_config_topology(
    config_spec: ConfigSpec, config: Mapping[str, object]
) -> str:
    """Resolve only structural selectors before validating lowering children."""
    assert config_spec._cute_flash_num_kv is not None
    if config_spec._cute_flash_requires_ws_overlap:
        return "ws_overlap"
    family = _flash_pipeline_family_flags(config.get(FLASH_PIPELINE_FAMILY_KEY))
    topology = family.topology if family is not None else config.get(FLASH_TOPOLOGY_KEY)
    if topology in ("fa4", "ws_overlap"):
        if topology == "fa4" and config_spec._cute_flash_num_kv % 2:
            return "ws_overlap"
        return cast("str", topology)
    # Register-MMA rows support odd KV counts and have their own admission.
    # The environment can select them when no explicit parent is present.
    # Resolve only structural selectors so unrelated lowering children keep
    # their established validation order.
    structural = {
        key: value
        for key, value in config.items()
        if key in (FLASH_PIPELINE_FAMILY_KEY, *FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS)
    }
    return _resolve_cute_flash_config(config_spec, structural).topology


def _legalize_cute_flash_compiler_seed(
    config_spec: ConfigSpec, seed: Config | None
) -> Config | None:
    """Project compiler-owned causal seeds onto a required TMA output path."""
    if seed is None or not config_spec._cute_flash_output_requires_tma:
        return seed
    if _resolve_cute_flash_config(config_spec, seed.config).epi_tma:
        return seed
    if not config_spec._cute_flash_is_causal:
        return None
    projected = Config.from_dict(
        {
            **seed.config,
            FLASH_EPI_TMA_KEY: True,
            FLASH_EPI_STG_KEY: False,
            FLASH_EPI_STG_STORE_KEY: "slice",
            FLASH_EPI_STG_GMEM_KEY: "stage",
        }
    )
    effective = _resolve_cute_flash_config(config_spec, projected.config)
    return projected if effective.epi_tma and not effective.epi_stg else None


def _legalize_cute_flash_compiler_seeds(
    config_spec: ConfigSpec, seeds: Sequence[Config]
) -> list[Config]:
    """Legalize and deduplicate compiler seeds while preserving rank order."""
    result: list[Config] = []
    seen: set[Config] = set()
    for seed in seeds:
        projected = _legalize_cute_flash_compiler_seed(config_spec, seed)
        if projected is not None and projected not in seen:
            result.append(projected)
            seen.add(projected)
    return result


def _normalize_cute_flash(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Normalize the flash-attention autotune surface.

    Only runs when ``cute_flash_search_enabled`` is set (the flash detector
    fired). Active keys are validated against their fragments, while legacy
    structural keys are projected into the compound pipeline-family key and
    removed before the config is used as an autotune identity.
    """
    if not config_spec.cute_flash_search_enabled:
        return
    assert config_spec._cute_flash_head_dim is not None
    assert config_spec._cute_flash_num_kv is not None
    # ``q_tile_count`` is a family-derived invariant and is projected back
    # to its canonical value below.  The historical MMA key accepted any
    # truthy object, so preserve that fixed-config compatibility before the
    # active Boolean fragment validates it.
    if FLASH_MMA_INTERLEAVE_KEY in config:
        config[FLASH_MMA_INTERLEAVE_KEY] = bool(config[FLASH_MMA_INTERLEAVE_KEY])
    causal_lpt = config.get(FLASH_CAUSAL_LPT_SWIZZLE_KEY)
    if (
        (
            config_spec._cute_flash_is_causal
            or config_spec._cute_flash_has_kv_tile_pruning
            or config_spec._cute_flash_has_list_walk
        )
        and type(causal_lpt) is int
        and config_spec._cute_flash_num_bh is not None
    ):
        # A causal launch group wider than the grid (a config carried over
        # from a larger grid) names the whole-grid order; the resolver
        # projects the explicit stateful lowering's width below.
        config[FLASH_CAUSAL_LPT_SWIZZLE_KEY] = min(
            causal_lpt, max(config_spec._cute_flash_num_bh, 1)
        )
    config.update(
        _flash_resident_softmax_overrides(config.get(FLASH_SOFTMAX_LOWERING_KEY))
    )
    block_size_targets = _cute_flash_block_size_target_list(config_spec)
    if fix_invalid:
        config["block_sizes"] = list(block_size_targets)
        config["pid_type"] = "flat"
        _normalize_cute_flash_default_sequence(config_spec, config, "l2_groupings", 1)
        _normalize_cute_flash_default_sequence(config_spec, config, "num_threads", 0)
        _normalize_cute_flash_default_sequence(
            config_spec, config, "cute_vector_widths", 1
        )
        _normalize_cute_flash_default_sequence(
            config_spec, config, "cute_lane_layouts", "blocked"
        )
        _normalize_cute_flash_default_loop_orders(config_spec, config)
        config.pop("epilogue_subtile", None)
    elif not _is_cute_flash_config_envelope(config_spec, config, block_size_targets):
        return

    upgrade_legacy_flash_config(
        config, lambda legacy: _resolve_cute_flash_config(config_spec, legacy)
    )
    requested_family = _flash_pipeline_family_flags(
        config.get(FLASH_PIPELINE_FAMILY_KEY)
    )
    if config_spec._cute_flash_requires_ws_overlap or (
        requested_family is not None and requested_family.topology == "ws_overlap"
    ):
        # Resolve the parent before expanding a compound packet: an inactive
        # child must not turn a legacy WS parent into an FA4 family.
        fold_flash_ws_inactive_values(config)
    config.update(flash_compound_exp2_packet_overrides(config_spec.flash_facts, config))
    if config_spec._cute_flash_requires_ws_overlap:
        config[FLASH_PIPELINE_FAMILY_KEY] = "ws_overlap"
        topology_override = "ws_overlap"
        pipeline_family_override = "ws_overlap"
    else:
        requested_family = _flash_pipeline_family_flags(
            config.get(FLASH_PIPELINE_FAMILY_KEY)
        )
        if requested_family is not None:
            requested_family_name = str(config[FLASH_PIPELINE_FAMILY_KEY])
            if (
                requested_family.tensor_4d_tma
                and not config_spec._cute_flash_supports_tensor_4d_tma
            ):
                effective_family = _resolve_cute_flash_pins(
                    config_spec, ((FLASH_PIPELINE_FAMILY_KEY, requested_family_name),)
                ).pipeline_family
                pipeline_family_override = effective_family
                effective_flags = _flash_pipeline_family_flags(effective_family)
                assert effective_flags is not None
                topology_override = effective_flags.topology
            else:
                pipeline_family_override = requested_family_name
                topology_override = requested_family.topology
        else:
            pipeline_family_override = None
            valid_manual_topologies = {"fa4", "ws_overlap", ROW_MMA_FAMILY}
            topology_value = config.get(FLASH_TOPOLOGY_KEY)
            topology_override = (
                topology_value if topology_value in valid_manual_topologies else None
            )

    def make_fragments(
        topology: str | None,
        family: str | None,
    ) -> dict[str, ConfigSpecFragment]:
        return _cute_flash_autotune_fragments(config_spec, topology, family)

    if (
        fix_invalid
        and config_spec._cute_flash_output_requires_tma
        and pipeline_family_override is not None
    ):
        requested_effective = _resolve_cute_flash_pins(
            config_spec,
            (
                (FLASH_PIPELINE_FAMILY_KEY, pipeline_family_override),
                (FLASH_EPI_TMA_KEY, True),
                (FLASH_EPI_STG_KEY, False),
            ),
        )
        if not requested_effective.epi_tma:
            config.pop(FLASH_PIPELINE_FAMILY_KEY, None)
            topology_override = None
            pipeline_family_override = None

    fragments = make_fragments(
        cast("str | None", topology_override), pipeline_family_override
    )
    fragment_family = _flash_pipeline_family_flags(
        fragments[FLASH_PIPELINE_FAMILY_KEY].default()
    )
    if fragment_family is not None and fragment_family.topology == "ws_overlap":
        # Fixed configs may contain old FA4 values for inactive WS fields.
        # Keep those aliases out of the search fragments; unknown values
        # still reach strict membership validation below.
        fold_flash_ws_inactive_values(config)
    e2e_offset_was_present = FLASH_E2E_OFFSET_KEY in config
    e2e_offset0_was_present = FLASH_E2E_OFFSET0_KEY in config
    e2e_offset_keys = (FLASH_E2E_OFFSET_KEY, FLASH_E2E_OFFSET0_KEY)
    explicit_e2e_offsets = {
        key: config[key] for key in e2e_offset_keys if key in config
    }
    for key, fragment in fragments.items():
        choices = cast("EnumFragment", fragment).choices
        if key in config:
            if config[key] not in choices:
                if key in e2e_offset_keys:
                    # Legacy explicit e2e frequency overrides can make offsets
                    # outside the autotune fragment valid. Validate the effective
                    # cadence after the e2e keys have been normalized below.
                    pass
                elif fix_invalid:
                    config[key] = fragment.default()
                else:
                    raise InvalidConfig(
                        f"{key} must be one of {list(choices)!r}, got {config[key]!r}"
                    )
        else:
            if key not in (*e2e_offset_keys, FLASH_PIPELINE_FAMILY_KEY):
                config[key] = fragment.default()
    if FLASH_PIPELINE_FAMILY_KEY not in config:
        config[FLASH_PIPELINE_FAMILY_KEY] = fragments[
            FLASH_PIPELINE_FAMILY_KEY
        ].default()

    effective = _resolve_cute_flash_config(config_spec, config)
    if config_spec._cute_flash_output_requires_tma and not effective.epi_tma:
        if not fix_invalid:
            raise InvalidConfig(
                f"{FLASH_EPI_TMA_KEY}=False is not legal for this output shape"
            )
        config[FLASH_EPI_TMA_KEY] = True
        config[FLASH_EPI_STG_KEY] = False
        effective = _resolve_cute_flash_config(config_spec, config)
        if not effective.epi_tma:
            repair_fragments = make_fragments(None, None)
            family_fragment = cast(
                "EnumFragment", repair_fragments[FLASH_PIPELINE_FAMILY_KEY]
            )
            for family in family_fragment.choices:
                repaired = _resolve_cute_flash_config(
                    config_spec,
                    {
                        **config,
                        FLASH_PIPELINE_FAMILY_KEY: family,
                        FLASH_EPI_TMA_KEY: True,
                        FLASH_EPI_STG_KEY: False,
                    },
                )
                if repaired.epi_tma:
                    config[FLASH_PIPELINE_FAMILY_KEY] = family
                    effective = repaired
                    fragments = repair_fragments
                    break
        if not effective.epi_tma:
            raise InvalidConfig(
                "CuTe flash output requires TMA, but no legal TMA output "
                "schedule is available"
            )
    effective_topology = effective.topology
    config.update(flash_effective_config_values(effective))
    if effective.softmax_lowering in ("resident_value_graph", "resident_stateful"):
        # The value graph owns its all-XU cadence. Legacy aliases and
        # inherited offsets must not reintroduce an inactive split schedule.
        for key in FLASH_LEGACY_E2E_CONFIG_KEYS:
            config.pop(key, None)
    elif effective_topology == "fa4":
        config.update(explicit_e2e_offsets)
        if effective.alternating_warpgroups:
            # Both softmax warpgroups of the alternating family process
            # the same rows: one emulation phase. Keep the resolver's pin
            # in the search identity so offsets differing only in the
            # second phase do not name the same program twice.
            config[FLASH_E2E_OFFSET0_KEY] = config[FLASH_E2E_OFFSET_KEY]
    e2e_schedule_default = _flash_e2e_schedule_default(
        effective_topology, config_spec._cute_flash_head_dim
    )
    exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
        str(config[FLASH_E2E_SCHEDULE_KEY]), e2e_schedule_default
    )
    exp2_impl, e2e_freq, e2e_res = flash_legacy_e2e_params(
        config.get, exp2_impl, e2e_freq, e2e_res
    )
    _impl, e2e_freq, e2e_res, _schedule = _flash_normalize_e2e_params(
        exp2_impl,
        e2e_freq,
        e2e_res,
        e2e_schedule_default,
    )
    masked_e2e_schedule = str(config.get(FLASH_MASKED_E2E_SCHEDULE_KEY, "inherit"))
    _masked_schedule, masked_e2e_freq, masked_e2e_res = (
        _flash_masked_e2e_schedule_params(
            masked_e2e_schedule,
            e2e_schedule_default,
            e2e_freq,
            e2e_res,
        )
    )
    if not (
        config_spec._cute_flash_is_causal
        or config_spec._cute_flash_has_kv_tile_pruning
        or config_spec._cute_flash_has_list_walk
    ):
        masked_e2e_freq = e2e_freq
        masked_e2e_res = e2e_res
    e2e_offset_period = _flash_e2e_offset_period(
        e2e_freq,
        e2e_res,
        masked_e2e_freq,
        masked_e2e_res,
    )
    if (
        e2e_offset_period > 0
        and effective_topology == "fa4"
        and config_spec._cute_flash_head_dim == 64
    ):
        split_default_freq = e2e_freq if e2e_res > 0 else masked_e2e_freq
        schedule_default_offset = split_default_freq // 8
    else:
        schedule_default_offset = 0
    default_offset = schedule_default_offset
    flash_env = config_spec._cute_flash_env
    assert flash_env is not None
    env_offset = flash_env.get("HELION_CUTE_FLASH_E2E_OFFSET")
    if env_offset is not None:
        default_offset = int(env_offset)
        if e2e_offset_period == 0:
            default_offset = 0
        elif default_offset < 0:
            default_offset = schedule_default_offset
        else:
            default_offset %= e2e_offset_period
    if not e2e_offset_was_present:
        config[FLASH_E2E_OFFSET_KEY] = default_offset
    default_offset0 = 0
    env_offset0 = flash_env.get("HELION_CUTE_FLASH_E2E_OFFSET0")
    if env_offset0 is not None:
        env_offset0_value = int(env_offset0)
        if e2e_offset_period == 0:
            default_offset0 = 0
        elif env_offset0_value < 0:
            default_offset0 %= e2e_offset_period
        else:
            default_offset0 = env_offset0_value % e2e_offset_period
    if not e2e_offset0_was_present:
        config[FLASH_E2E_OFFSET0_KEY] = default_offset0
    for key, default in (
        (FLASH_E2E_OFFSET_KEY, default_offset),
        (FLASH_E2E_OFFSET0_KEY, default_offset0),
    ):
        e2e_offset_value = config[key]
        if not isinstance(e2e_offset_value, int):
            if fix_invalid:
                config[key] = default
                e2e_offset_value = default
            else:
                raise InvalidConfig(
                    f"{key} must be an integer, got {e2e_offset_value!r}"
                )
        e2e_offset = e2e_offset_value
        e2e_offset_invalid = (
            e2e_offset != 0
            if e2e_offset_period == 0
            else e2e_offset < 0 or e2e_offset >= e2e_offset_period
        )
        if e2e_offset_invalid:
            if fix_invalid:
                config[key] = _flash_normalize_e2e_offset(
                    e2e_offset, default, e2e_offset_period
                )
            else:
                expected = (
                    [0] if e2e_offset_period == 0 else list(range(e2e_offset_period))
                )
                raise InvalidConfig(
                    f"{key} must be one of {expected!r} for "
                    f"{FLASH_E2E_SCHEDULE_KEY}={config[FLASH_E2E_SCHEDULE_KEY]!r}, "
                    f"got {e2e_offset!r}"
                )
    _normalize_cute_flash_register_budget(
        config_spec,
        config,
        fragments,
        effective_topology,
        fix_invalid=fix_invalid,
    )
    for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
        config.pop(key, None)


@functools.lru_cache(maxsize=4096)
def _fa4_register_budget_repair(
    softmax_choices: tuple[int, ...],
    corr_choices: tuple[int, ...],
    other_choices: tuple[int, ...],
    current: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    """The legal FA4 register split closest to ``current``, or None.

    Memoized: normalization repairs the same few over-budget splits against
    the same fragment choices for every sampled config.
    """
    best: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    for softmax_candidate in softmax_choices:
        for corr_candidate in corr_choices:
            for other_candidate in other_choices:
                candidate = (softmax_candidate, corr_candidate, other_candidate)
                if 2 * softmax_candidate + corr_candidate + other_candidate > 512:
                    continue
                score = (
                    sum(a != b for a, b in zip(candidate, current, strict=True)),
                    sum(abs(a - b) for a, b in zip(candidate, current, strict=True)),
                    -softmax_candidate,
                )
                if best is None or score < best[0]:
                    best = (score, candidate)
    return None if best is None else best[1]


def _normalize_cute_flash_register_budget(
    config_spec: ConfigSpec,
    config: dict[str, object],
    fragments: Mapping[str, ConfigSpecFragment],
    effective_topology: str,
    *,
    fix_invalid: bool,
) -> None:
    if effective_topology != "fa4":
        return
    softmax_regs = config.get(FLASH_SOFTMAX_REGS_KEY)
    corr_regs = config.get(FLASH_CORR_REGS_KEY)
    other_regs = config.get(FLASH_OTHER_REGS_KEY)
    if (
        type(softmax_regs) is not int
        or type(corr_regs) is not int
        or type(other_regs) is not int
    ):
        return
    budget = 2 * softmax_regs + corr_regs + other_regs
    if budget <= 512:
        return
    message = (
        "FA4 register budget exceeds 512: "
        f"2 * {FLASH_SOFTMAX_REGS_KEY} ({softmax_regs}) + "
        f"{FLASH_CORR_REGS_KEY} ({corr_regs}) + "
        f"{FLASH_OTHER_REGS_KEY} ({other_regs}) = {budget}"
    )
    if not fix_invalid:
        raise InvalidConfig(message)

    def _int_choices(key: str) -> tuple[int, ...]:
        fragment = cast("EnumFragment", fragments[key])
        return tuple(value for value in fragment.choices if type(value) is int)

    repaired = _fa4_register_budget_repair(
        _int_choices(FLASH_SOFTMAX_REGS_KEY),
        _int_choices(FLASH_CORR_REGS_KEY),
        _int_choices(FLASH_OTHER_REGS_KEY),
        (softmax_regs, corr_regs, other_regs),
    )
    if repaired is None:
        raise InvalidConfig(message)
    softmax_regs, corr_regs, other_regs = repaired
    config[FLASH_SOFTMAX_REGS_KEY] = softmax_regs
    config[FLASH_CORR_REGS_KEY] = corr_regs
    config[FLASH_OTHER_REGS_KEY] = other_regs


def enable_cute_flash_search(
    config_spec: ConfigSpec,
    *,
    head_dim: int,
    num_kv: int,
    num_bh: int | None = None,
    tensor_4d_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    block_size_targets: Mapping[int, int],
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    dense_lse_output: bool = False,
    standard_causal_output: bool = False,
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    output_requires_tma: bool = False,
    supports_tensor_4d_tma: bool = True,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
    score_modifier_kinds: Sequence[str] = (),
    device_sm_count: int = 0,
    head_dim_v: int | None = None,
    kv_group: int = 1,
    fast_math: bool = False,
    has_list_walk: bool = False,
    flex_has_score_region: bool = False,
    flex_has_mask_region: bool = False,
    flex_trivial_walk: bool = False,
    q_seq: int | None = None,
    kv_seq: int | None = None,
    flex_aux_smem_candidates: Sequence[int] = (),
    flex_prefetch_regs: Sequence[int] = (),
    flex_coalesce_steps: Sequence[int] = (),
    flex_hook_ops: int = 0,
    env: FlashEnv | None = None,
) -> None:
    config_spec.cute_attention_generic_fallback_enabled = False
    config_spec._cute_attention_generic_fallback_block_size_targets = {}
    config_spec.cute_flash_search_enabled = True
    config_spec._cute_flash_pin_resolutions = {}
    # The knob channel is captured once, here (bind time); ``env`` is the
    # binding's capture overlaid with ``Settings.cute_flash_env_overrides``.
    config_spec._cute_flash_env = FlashEnv.capture() if env is None else env
    # The pipeline family picks the flash body (the alternating-warpgroup
    # single tile vs the two-tile pair, the 2-CTA cluster, ...; the Q
    # tiles per CTA follow from it); the bodies' energy per call differs,
    # so the sustained clock of the final selection's windows does too.
    config_spec.final_selection_family_keys = (FLASH_PIPELINE_FAMILY_KEY,)
    config_spec._cute_flash_device_sm_count = device_sm_count
    config_spec._cute_flash_fragments_cache.clear()
    config_spec._cute_flash_head_dim = head_dim
    config_spec._cute_flash_head_dim_v = (
        None if head_dim_v is None or head_dim_v == head_dim else head_dim_v
    )
    config_spec._cute_flash_num_kv = num_kv
    config_spec._cute_flash_num_bh = num_bh
    config_spec._cute_flash_kv_group = kv_group
    config_spec._cute_flash_tensor_4d_heads = tensor_4d_heads
    config_spec._cute_flash_dtype = dtype
    config_spec._cute_flash_is_causal = is_causal
    config_spec._cute_flash_has_kv_tile_pruning = has_kv_tile_pruning
    config_spec._cute_flash_requires_ws_overlap = requires_ws_overlap
    config_spec._cute_flash_has_list_walk = has_list_walk
    config_spec._cute_flash_small_biased_candidate = small_biased_candidate
    config_spec._cute_flash_standard_dense_output = standard_dense_output
    config_spec._cute_flash_dense_lse_output = dense_lse_output
    config_spec._cute_flash_standard_causal_output = standard_causal_output
    config_spec._cute_flash_causal_resident_compatible = causal_resident_compatible
    config_spec._cute_flash_tmem_rowmax_compatible = (
        standard_dense_output or standard_causal_output
        if tmem_rowmax_compatible is None
        else tmem_rowmax_compatible
    )
    config_spec._cute_flash_output_requires_tma = output_requires_tma
    config_spec._cute_flash_supports_tensor_4d_tma = supports_tensor_4d_tma
    config_spec._cute_flash_has_row_epilogue = has_row_epilogue
    config_spec._cute_flash_plain_row_body = plain_row_body
    config_spec._cute_flash_has_score_modifiers = has_score_modifiers
    config_spec._cute_flash_score_modifier_kinds = tuple(score_modifier_kinds)
    config_spec._cute_flash_fast_math = fast_math
    config_spec._cute_flash_has_list_walk = has_list_walk
    config_spec._cute_flash_flex_has_score_region = flex_has_score_region
    config_spec._cute_flash_flex_has_mask_region = flex_has_mask_region
    config_spec._cute_flash_flex_trivial_walk = flex_trivial_walk
    config_spec._cute_flash_q_seq = q_seq
    config_spec._cute_flash_kv_seq = kv_seq
    config_spec._cute_flash_flex_aux_smem_candidates = tuple(flex_aux_smem_candidates)
    config_spec._cute_flash_flex_prefetch_regs = tuple(flex_prefetch_regs)
    config_spec._cute_flash_flex_coalesce_steps = tuple(flex_coalesce_steps)
    config_spec._cute_flash_flex_hook_ops = flex_hook_ops
    config_spec._cute_flash_block_size_targets = dict(block_size_targets)
    for block_id, target in block_size_targets.items():
        spec = config_spec.block_sizes.block_id_lookup(block_id)
        spec.pin_structurally(
            target,
            reason="the CuTe fused flash-attention forward (backend 'cute')",
        )


def enable_cute_flash_gated_search(
    config_spec: ConfigSpec,
    *,
    block_size_targets: Mapping[int, int],
    kv_block_id: int,
    kv_tile_choices: Sequence[int],
    kv_stage_choices: Sequence[int],
    kv_stage_default: int,
    gate_warpgroup_choices: Sequence[int] = (1,),
    gate_warpgroup_default: int = 1,
    kv_stage_choices_by_tile: Mapping[tuple[int, int], Sequence[int]] | None = None,
    q_block_id: int | None = None,
    q_tile_choices: Sequence[int] = (128,),
    row_warp_choices_by_tile: Mapping[tuple[int, int], Sequence[int]] | None = None,
    row_stage_choices: Mapping[tuple[int, int, int], Sequence[int]] | None = None,
    row_seed_programs: int | None = None,
) -> None:
    """Enable the gated (softmax-free) fused attention surface.

    Restricts the query block size to the gated bodies' row tiles (128 or
    64 for the fused tcgen05 body; 16 or 32 for the register-MMA row
    programs, ``row_warp_choices_by_tile``) and the KV block size to their
    tile widths (the detector fires only there) and exposes the K/V ring
    depth, the number of gate warpgroups (fused body) and the warp count
    (row programs).
    """
    config_spec.cute_flash_gated_search_enabled = True
    config_spec._cute_flash_gated_row_warp_choices_by_tile = {
        tile: tuple(warps) for tile, warps in (row_warp_choices_by_tile or {}).items()
    }
    config_spec._cute_flash_gated_row_stage_choices = {
        key: tuple(stages) for key, stages in (row_stage_choices or {}).items()
    }
    config_spec._cute_flash_gated_row_seed_programs = row_seed_programs
    config_spec._cute_flash_gated_block_size_targets = dict(block_size_targets)
    config_spec._cute_flash_gated_kv_block_id = kv_block_id
    config_spec._cute_flash_gated_q_block_id = q_block_id
    config_spec._cute_flash_gated_q_tile_choices = tuple(sorted(q_tile_choices))
    config_spec._cute_flash_gated_kv_tile_choices = tuple(sorted(kv_tile_choices))
    config_spec._cute_flash_gated_kv_stage_choices = tuple(kv_stage_choices)
    config_spec._cute_flash_gated_kv_stage_default = kv_stage_default
    config_spec._cute_flash_gated_gate_warpgroup_choices = tuple(gate_warpgroup_choices)
    config_spec._cute_flash_gated_gate_warpgroup_default = gate_warpgroup_default
    config_spec._cute_flash_gated_kv_stage_choices_by_tile = {
        tile: tuple(stages) for tile, stages in (kv_stage_choices_by_tile or {}).items()
    }
    for block_id, target in block_size_targets.items():
        spec = config_spec.block_sizes.block_id_lookup(block_id)
        if block_id == kv_block_id:
            spec.autotuner_min = min(config_spec._cute_flash_gated_kv_tile_choices)
            spec.max_size = max(config_spec._cute_flash_gated_kv_tile_choices)
            if min(config_spec._cute_flash_gated_q_tile_choices) < spec.max_size:
                # The frontend caps a KV loop bounded by the query tile
                # (``hl.tile(0, tile_q.end)``) at the query block size.  The
                # fused body walks each work item's KV range independently
                # of the tile height, so a 128-column KV tile over a 64-row
                # query tile is legal (and the efficient shape).
                spec.bounded_by_block_id = None
        elif block_id == q_block_id:
            spec.autotuner_min = min(config_spec._cute_flash_gated_q_tile_choices)
            spec.max_size = max(config_spec._cute_flash_gated_q_tile_choices)
            # The 128-row tile stays the default; 64 rows is a search choice.
            spec.default_size = target
        else:
            spec.autotuner_min = target
            spec.max_size = target


def _cute_flash_gated_tiles(config_spec: ConfigSpec) -> list[tuple[int, int]]:
    """Every fused (query rows, KV tile) shape with a ring depth that fits
    shared memory: tallest query tile and widest KV tile first."""
    by_tile = config_spec._cute_flash_gated_kv_stage_choices_by_tile
    return [
        (q_tile, kv_tile)
        for q_tile in sorted(config_spec._cute_flash_gated_q_tile_choices, reverse=True)
        for kv_tile in sorted(
            config_spec._cute_flash_gated_kv_tile_choices, reverse=True
        )
        if not by_tile or by_tile.get((q_tile, kv_tile))
    ]


def _cute_flash_gated_block_size_lists(config_spec: ConfigSpec) -> list[list[int]]:
    """The block sizes of every fused tile shape (``_cute_flash_gated_tiles`` order)."""
    base = _cute_flash_gated_block_size_target_list(config_spec)
    kv_block_id = config_spec._cute_flash_gated_kv_block_id
    assert kv_block_id is not None
    kv_index = config_spec.block_sizes.block_id_to_index(kv_block_id)
    q_block_id = config_spec._cute_flash_gated_q_block_id
    q_index = (
        config_spec.block_sizes.block_id_to_index(q_block_id)
        if q_block_id is not None
        else None
    )
    result: list[list[int]] = []
    for q_tile, kv_tile in _cute_flash_gated_tiles(config_spec):
        block_sizes = list(base)
        block_sizes[kv_index] = kv_tile
        if q_index is not None:
            block_sizes[q_index] = q_tile
        result.append(block_sizes)
    return result


def _cute_flash_gated_block_size_target_list(config_spec: ConfigSpec) -> list[int]:
    targets: list[int | None] = [None] * len(config_spec.block_sizes)
    for block_id, target in config_spec._cute_flash_gated_block_size_targets.items():
        targets[config_spec.block_sizes.block_id_to_index(block_id)] = target
    if any(target is None for target in targets):
        raise InvalidConfig("CuTe gated attention search has incomplete block sizes")
    return [target for target in targets if target is not None]


def cute_flash_gated_seed_configs(config_spec: ConfigSpec) -> list[Config]:
    if not config_spec.cute_flash_gated_search_enabled:
        return []
    from ..cute_flash_gated import gated_seed_configs
    from ..cute_flash_gated_row_mma import gated_row_seed_configs

    tiles = _cute_flash_gated_tiles(config_spec)
    block_size_lists = _cute_flash_gated_block_size_lists(config_spec)
    row_tiles = config_spec._cute_flash_gated_row_warp_choices_by_tile
    fused = [
        (sizes, tile)
        for sizes, tile in zip(block_size_lists, tiles, strict=True)
        if tile not in row_tiles
    ]
    rows = [
        (sizes, tile)
        for sizes, tile in zip(block_size_lists, tiles, strict=True)
        if tile in row_tiles
    ]
    seeds = gated_seed_configs(
        [sizes for sizes, _ in fused],
        config_spec._cute_flash_gated_kv_stage_choices,
        config_spec._cute_flash_gated_kv_stage_default,
        tiles=[tile for _, tile in fused],
        kv_stage_choices_by_tile=config_spec._cute_flash_gated_kv_stage_choices_by_tile,
    )
    seeds.extend(
        gated_row_seed_configs(
            [sizes for sizes, _ in rows],
            tiles=[tile for _, tile in rows],
            warp_choices_by_tile=row_tiles,
            stage_choices=config_spec._cute_flash_gated_row_stage_choices,
            programs=config_spec._cute_flash_gated_row_seed_programs,
        )
    )
    return seeds


def _cute_flash_gated_kv_tile(
    config_spec: ConfigSpec, config: dict[str, object]
) -> int:
    """The KV tile width of a config on the gated surface."""
    kv_block_id = config_spec._cute_flash_gated_kv_block_id
    assert kv_block_id is not None
    value = config.get("block_sizes")
    assert isinstance(value, (list, tuple))
    kv_tile = value[config_spec.block_sizes.block_id_to_index(kv_block_id)]
    assert isinstance(kv_tile, int)
    return kv_tile


def _cute_flash_gated_q_tile(config_spec: ConfigSpec, config: dict[str, object]) -> int:
    """The query tile height of a config on the gated surface."""
    q_block_id = config_spec._cute_flash_gated_q_block_id
    if q_block_id is None:
        return max(config_spec._cute_flash_gated_q_tile_choices)
    value = config.get("block_sizes")
    assert isinstance(value, (list, tuple))
    q_tile = value[config_spec.block_sizes.block_id_to_index(q_block_id)]
    assert isinstance(q_tile, int)
    return q_tile


def _normalize_cute_flash_bwd(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    """The flex backward's 2-CTA family (``cute_flash_bwd_two_cta`` = 1)
    has no persistent walk, statistics MMAs, dS^T-through-TMEM dK or
    staged aux tables: those knobs read 0 under it, so a config that
    sets them spells the same kernel as one that does not (one autotune
    identity, no inert combinations in the search). The pair's step
    table is its own (``flex_bwd_two_cta_table_steps``): a
    ``cute_flash_bwd_table_steps`` cap at or above its automatic table
    renders that table, so it reads 0 (the search's caps are the 1-CTA
    kernel's, mostly far above the pair's 64-128 steps). Where the pair
    kernel is not legal the knob renders the 1-CTA kernel, whose knobs
    all apply: the knob itself reads 0 there.

    The dense surface's register-MMA row family (``cute_flash_bwd_family``
    = "row_mma") is legalized first: its three row knobs read their
    defaults off the family, and an illegal triple on it moves to the
    nearest legal one or back to the tcgen05 family (see
    ``cute_flash_bwd_row_mma.flash_bwd_row_legalize``)."""
    if (
        config_spec.cute_flash_bwd_search_enabled
        and not config_spec._cute_flash_bwd_flex
    ):
        flash_bwd_row_legalize(config_spec._cute_flash_bwd_row_facts, config)
    if not (
        config_spec.cute_flash_bwd_search_enabled and config_spec._cute_flash_bwd_flex
    ):
        return
    # The first-step order knob reads "auto" wherever it has no effect
    # (the alias_s and pair plans, the 2-CTA family): one identity.
    if (
        not config_spec._cute_flash_bwd_s_first_allowed
        or config.get(FLASH_BWD_TWO_CTA_KEY) == 1
    ) and FLASH_BWD_S_FIRST_KEY in config:
        config[FLASH_BWD_S_FIRST_KEY] = "auto"
    # A second K / V slot serves the persistent walk alone (a one-tile
    # CTA loads once), nowhere the storage cannot hold it, and not the
    # 2-CTA family: it reads 1 there.
    if (
        not config_spec._cute_flash_bwd_kv_stage_allowed
        or config.get(FLASH_BWD_PERSISTENT_KEY) != 1
        or config.get(FLASH_BWD_TWO_CTA_KEY) == 1
    ) and FLASH_BWD_KV_STAGE_KEY in config:
        config[FLASH_BWD_KV_STAGE_KEY] = 1
    if config.get(FLASH_BWD_TWO_CTA_KEY) != 1:
        return
    if not config_spec._cute_flash_bwd_two_cta_allowed:
        config[FLASH_BWD_TWO_CTA_KEY] = 0
        return
    for key in (
        FLASH_BWD_PERSISTENT_KEY,
        FLASH_BWD_STATS_MMA_KEY,
        FLASH_BWD_DS_TMEM_KEY,
        FLASH_BWD_AUX_SMEM_KEY,
    ):
        if key in config:
            config[key] = 0
    cap = config.get(FLASH_BWD_TABLE_STEPS_KEY, 0)
    auto = config_spec._cute_flash_bwd_two_cta_table_steps
    if isinstance(cap, int) and auto > 0 and cap >= auto:
        config[FLASH_BWD_TABLE_STEPS_KEY] = 0
    # The pair's register split: the knob's generic default is the 1-CTA
    # kernel's (FBWD_REG_SPLITS[0]); under the pair it reads as the
    # pair's first seed split, so a config that pins two_cta without a
    # split renders the measured kernel (and spells one identity with
    # the seed).
    from ..cute_flash_bwd_flex import FBWD_REG_SPLITS

    if config.get(FLASH_BWD_REGS_KEY, FBWD_REG_SPLITS[0]) == FBWD_REG_SPLITS[0]:
        config[FLASH_BWD_REGS_KEY] = "88/168/88"


def pdl_searched(config_spec: ConfigSpec) -> bool:
    """Automatic PDL is searchable on sm_90+ when explicit markers do not
    already govern the kernel's launch protocol."""
    return (
        config_spec.supports_config_key("cute_pdl")
        and not config_spec.has_explicit_pdl
        and config_spec.target_device_capability is not None
        and config_spec.target_device_capability >= (9, 0)
    )


def cute_flash_bwd_seed_configs(config_spec: ConfigSpec) -> list[Config]:
    """The flex backward's 2-CTA family as an autotune seed where its
    kernel is legal: the default config with the pair under each of its
    two measured register splits (the compute warpgroups spill below 152
    registers; the MMA and load warps hold their rings at 88 or 104), so
    a cold search starts from both families instead of finding the pair
    by chance.

    The dense surface seeds its register-MMA row family on small grids
    (``cute_flash_bwd_family`` = "row_mma" over the default row knobs:
    see ``cute_flash_bwd_row_mma``) for the same reason, and, where the
    search flips ``cute_pdl``, the same seed as a programmatic dependent
    (the small-grid backward is one kernel of a chain whose launch the
    knob overlaps with the preprocess kernel ahead of it)."""
    if not config_spec.cute_flash_bwd_search_enabled:
        return []
    row_facts = config_spec._cute_flash_bwd_row_facts
    if (
        not config_spec._cute_flash_bwd_flex
        and row_facts is not None
        and flash_bwd_row_search_grid(row_facts)
    ):
        seed = config_spec.default_config()
        seed.config[FLASH_BWD_FAMILY_KEY] = FLASH_BWD_ROW_FAMILY
        config_spec.normalize(seed, _fix_invalid=True)
        if seed.config.get(FLASH_BWD_FAMILY_KEY) != FLASH_BWD_ROW_FAMILY:
            return []
        seeds = [seed]
        if config_spec.supports_config_key("cute_pdl") and pdl_searched(config_spec):
            dependent = Config.from_dict({**seed.config, "cute_pdl": True})
            config_spec.normalize(dependent, _fix_invalid=True)
            seeds.append(dependent)
        return seeds
    if not (
        config_spec._cute_flash_bwd_flex and config_spec._cute_flash_bwd_two_cta_allowed
    ):
        return []
    # Two register splits: the cold autotunes of the 32K / 128K causal
    # cells pick 88/168/88 (the dQ-reduce and load warps at 88, the
    # compute warpgroups at 168), 0.6-1.5% ahead of 104/152/104 on the
    # 32K proxies; both keep the compute warpgroups free of spills.
    seeds = []
    for regs in ("88/168/88", "104/152/104"):
        seed = config_spec.default_config()
        seed.config[FLASH_BWD_TWO_CTA_KEY] = 1
        seed.config[FLASH_BWD_REGS_KEY] = regs
        config_spec.normalize(seed, _fix_invalid=True)
        seeds.append(seed)
    return seeds


def _normalize_cute_flash_gated(
    config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Normalize the gated attention surface (see ``_normalize_cute_flash``)."""
    if not config_spec.cute_flash_gated_search_enabled:
        return
    block_size_lists = _cute_flash_gated_block_size_lists(config_spec)
    if fix_invalid:
        if config.get("block_sizes") not in block_size_lists:
            config["block_sizes"] = _cute_flash_gated_nearest_block_sizes(
                config_spec, config.get("block_sizes"), block_size_lists
            )
        config["pid_type"] = "flat"
        _normalize_cute_flash_default_sequence(config_spec, config, "l2_groupings", 1)
        _normalize_cute_flash_default_sequence(config_spec, config, "num_threads", 0)
        _normalize_cute_flash_default_sequence(
            config_spec, config, "cute_vector_widths", 1
        )
        _normalize_cute_flash_default_sequence(
            config_spec, config, "cute_lane_layouts", "blocked"
        )
        _normalize_cute_flash_default_loop_orders(config_spec, config)
        config.pop("epilogue_subtile", None)
    elif not any(
        _is_cute_flash_config_envelope(config_spec, config, block_sizes)
        for block_sizes in block_size_lists
    ):
        return
    from ..cute_flash_gated import gated_kv_stage_default

    kv_tile = _cute_flash_gated_kv_tile(config_spec, config)
    tile = (_cute_flash_gated_q_tile(config_spec, config), kv_tile)
    row_warp_choices = config_spec._cute_flash_gated_row_warp_choices_by_tile.get(tile)
    if row_warp_choices is not None:
        # A register-MMA row-program tile: its warp count and ring depth
        # are validated by the row normalizer; the gate warpgroup count
        # has no consumer there (one identity).
        _normalize_cute_flash_gated_row(
            config_spec, config, tile, row_warp_choices, fix_invalid=fix_invalid
        )
        return
    # The legal ring depths depend on the tile shape (shared memory), so
    # validate against the config's own tile.
    choices = config_spec._cute_flash_gated_kv_stage_choices_by_tile.get(
        tile, config_spec._cute_flash_gated_kv_stage_choices
    )
    stage_default = (
        config_spec._cute_flash_gated_kv_stage_default
        if config_spec._cute_flash_gated_kv_stage_default in choices
        else gated_kv_stage_default(choices)
    )
    value = config.get(FLASH_KV_STAGE_KEY)
    if value is None:
        config[FLASH_KV_STAGE_KEY] = stage_default
    elif value not in choices:
        if not fix_invalid:
            raise InvalidConfig(
                f"{FLASH_KV_STAGE_KEY} must be one of {list(choices)!r} "
                f"for block_sizes {config.get('block_sizes')!r}, got {value!r}"
            )
        config[FLASH_KV_STAGE_KEY] = stage_default
    from ..cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY
    from ..cute_flash_gated import gated_gate_warpgroup_choices
    from ..cute_flash_gated import gated_gate_warpgroup_default

    # The legal warpgroup counts depend on the tile (its chunks must split
    # evenly among the warpgroups), so validate against the config's own.
    wg_choices = tuple(
        g
        for g in gated_gate_warpgroup_choices(
            _cute_flash_gated_q_tile(config_spec, config),
            _cute_flash_gated_kv_tile(config_spec, config),
        )
        if g in config_spec._cute_flash_gated_gate_warpgroup_choices
    ) or (1,)
    wg_default = gated_gate_warpgroup_default(wg_choices)
    wgs = config.get(FLASH_GATE_WARPGROUPS_KEY)
    if wgs is None:
        config[FLASH_GATE_WARPGROUPS_KEY] = wg_default
    elif wgs not in wg_choices:
        if not fix_invalid:
            raise InvalidConfig(
                f"{FLASH_GATE_WARPGROUPS_KEY} must be one of {list(wg_choices)!r} "
                f"for block_sizes {config.get('block_sizes')!r}, got {wgs!r}"
            )
        config[FLASH_GATE_WARPGROUPS_KEY] = wg_default
    _normalize_cute_flash_gated_row(
        config_spec, config, tile, None, fix_invalid=fix_invalid
    )


def _cute_flash_gated_nearest_block_sizes(
    config_spec: ConfigSpec, value: object, block_size_lists: list[list[int]]
) -> list[int]:
    """The legal gated tile nearest a config's block sizes: the same
    query tile with the closest legal KV tile when the query tile is
    legal, else the default tile."""
    kv_block_id = config_spec._cute_flash_gated_kv_block_id
    assert kv_block_id is not None
    kv_index = config_spec.block_sizes.block_id_to_index(kv_block_id)
    if isinstance(value, (list, tuple)) and len(value) == len(block_size_lists[0]):
        kv_value = value[kv_index]
        same_q = [
            sizes
            for sizes in block_size_lists
            if all(
                a == b
                for i, (a, b) in enumerate(zip(sizes, value, strict=True))
                if i != kv_index
            )
        ]
        if same_q and isinstance(kv_value, int):
            return list(min(same_q, key=lambda sizes: abs(sizes[kv_index] - kv_value)))
    return list(block_size_lists[0])


def _normalize_cute_flash_gated_row(
    config_spec: ConfigSpec,
    config: dict[str, object],
    tile: tuple[int, int],
    warp_choices: tuple[int, ...] | None,
    *,
    fix_invalid: bool,
) -> None:
    """The row programs' knob (``cute_flash_gated_row_warps``) and their
    ring depths, which depend on the warp count; at a fused tile
    (``warp_choices`` None) the knob reads its default (one identity).
    At a row tile the gate warpgroup count reads its default."""
    from ..cute_flash_gated import FLASH_GATE_WARPGROUPS_KEY
    from ..cute_flash_gated_row_mma import GATED_ROW_WARPS_KEY
    from ..cute_flash_gated_row_mma import gated_row_kv_stage_default
    from ..cute_flash_gated_row_mma import gated_row_warps_default

    if not config_spec._cute_flash_gated_row_warp_choices_by_tile:
        config.pop(GATED_ROW_WARPS_KEY, None)
        return
    if warp_choices is not None:
        config[FLASH_GATE_WARPGROUPS_KEY] = (
            config_spec._cute_flash_gated_gate_warpgroup_default
        )
    all_warps = tuple(
        sorted(
            {
                w
                for warps in config_spec._cute_flash_gated_row_warp_choices_by_tile.values()
                for w in warps
            }
        )
    )
    if warp_choices is None:
        config[GATED_ROW_WARPS_KEY] = gated_row_warps_default(all_warps)
        return
    default_warps = gated_row_warps_default(warp_choices)
    warps = config.get(GATED_ROW_WARPS_KEY)
    if warps is None:
        warps = default_warps
    elif warps not in warp_choices:
        if not fix_invalid:
            raise InvalidConfig(
                f"{GATED_ROW_WARPS_KEY} must be one of {list(warp_choices)!r} "
                f"for block_sizes {config.get('block_sizes')!r}, got {warps!r}"
            )
        warps = default_warps
    config[GATED_ROW_WARPS_KEY] = warps
    assert isinstance(warps, int)
    stages = config_spec._cute_flash_gated_row_stage_choices.get((*tile, warps), ())
    if not stages:
        return
    stage = config.get(FLASH_KV_STAGE_KEY)
    if stage is None:
        config[FLASH_KV_STAGE_KEY] = gated_row_kv_stage_default(stages)
    elif stage not in stages:
        if not fix_invalid:
            raise InvalidConfig(
                f"{FLASH_KV_STAGE_KEY} must be one of {list(stages)!r} for "
                f"block_sizes {config.get('block_sizes')!r} at {warps} warps, "
                f"got {stage!r}"
            )
        config[FLASH_KV_STAGE_KEY] = gated_row_kv_stage_default(stages)


def enable_cute_flash_bwd_search(
    config_spec: ConfigSpec,
    *,
    block_size_targets: Mapping[int, int],
    two_cta_allowed: bool = False,
    persistent_allowed: bool = True,
    flex: bool = False,
    tile_groups: tuple[int, ...] = (0,),
    fast_math: bool = False,
    aux_smem_candidates: Sequence[int] = (),
    ds_tmem_allowed: bool = False,
    stats_mma_allowed: bool = True,
    s_first_allowed: bool = False,
    kv_stage_allowed: bool = False,
    has_mask: bool = True,
    has_score: bool = True,
    additive_bias: bool = True,
    table_step_choices: Sequence[int] = (0,),
    two_cta_table_steps: int = 0,
    row_mma_facts: FlashBwdRowFacts | None = None,
) -> None:
    """Enable the CuTe flash-attention BACKWARD surface.

    Pins the (kv_tile, q_tile) block sizes to the 128x128 envelope the
    fused backward emitter supports. ``two_cta_allowed`` opens the
    ``cute_flash_bwd_two_cta`` knob (cluster-of-2 tcgen05 family) when the
    problem shape supports 256-row cluster KV tiles;
    ``two_cta_table_steps`` is the flex pair kernel's automatic step
    table (``cute_flash_bwd_table_steps`` caps at or above it render it;
    see ``_normalize_cute_flash_bwd``). ``row_mma_facts`` (the dense
    surface) carries the plan facts of the register-MMA row family
    (``cute_flash_bwd_family`` = "row_mma"; see
    ``cute_flash_bwd_row_mma``): its knobs are offered where the facts
    admit the family and searched / seeded on small grids.
    """
    config_spec.cute_flash_bwd_search_enabled = True
    # The backward family (the tcgen05 pair vs the register-MMA rows)
    # picks the body the same way the forward's pipeline family does.
    config_spec.final_selection_family_keys = (FLASH_BWD_FAMILY_KEY,)
    config_spec._cute_flash_bwd_row_facts = row_mma_facts
    config_spec._cute_flash_bwd_block_size_targets = dict(block_size_targets)
    config_spec._cute_flash_bwd_two_cta_allowed = two_cta_allowed
    config_spec._cute_flash_bwd_two_cta_table_steps = (
        two_cta_table_steps if two_cta_allowed else 0
    )
    config_spec._cute_flash_bwd_persistent_allowed = persistent_allowed
    config_spec._cute_flash_bwd_flex = flex
    config_spec._cute_flash_bwd_tile_groups = tile_groups
    config_spec._cute_flash_bwd_fast_math = fast_math
    config_spec._cute_flash_bwd_aux_smem_candidates = tuple(aux_smem_candidates)
    config_spec._cute_flash_bwd_ds_tmem_allowed = ds_tmem_allowed
    config_spec._cute_flash_bwd_stats_mma_allowed = stats_mma_allowed
    config_spec._cute_flash_bwd_s_first_allowed = s_first_allowed
    config_spec._cute_flash_bwd_kv_stage_allowed = kv_stage_allowed
    # Which hook knobs have an effect: the mask form needs a mask region,
    # the hook group width / pin a hook at all, the fold an additive bias.
    config_spec._cute_flash_bwd_has_mask = has_mask
    config_spec._cute_flash_bwd_has_hook = has_mask or has_score
    config_spec._cute_flash_bwd_additive_bias = additive_bias
    # Steps per step-table buffer of the flex list walk (0 = automatic);
    # the detector offers the caps that chunk the problem's walk.
    config_spec._cute_flash_bwd_table_step_choices = tuple(table_step_choices)
    for block_id, target in block_size_targets.items():
        spec = config_spec.block_sizes.block_id_lookup(block_id)
        spec.pin_structurally(
            target,
            reason="the CuTe fused flash-attention backward (backend 'cute')",
        )


def enable_cute_attention_generic_fallback(
    config_spec: ConfigSpec, *, block_size_targets: Mapping[int, int] | None = None
) -> None:
    """Use generic CuTe for an attention shape with no legal flash output."""
    config_spec.cute_attention_generic_fallback_enabled = True
    config_spec._cute_attention_generic_fallback_block_size_targets = dict(
        block_size_targets or {}
    )
    # These are legality bounds for the generic attention lowering, not a
    # selected winner. They make the ordinary default a known-good initial
    # candidate while leaving all larger block sizes available to tuning.
    for (
        block_id,
        target,
    ) in config_spec._cute_attention_generic_fallback_block_size_targets.items():
        spec = config_spec.block_sizes.block_id_lookup(block_id)
        spec.autotuner_min = max(spec.autotuner_min, target)


def _pre_normalize_cute_flash_block_sizes(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    if "block_sizes" not in config:
        return
    if config_spec.cute_flash_search_enabled:
        block_size_targets = _cute_flash_block_size_target_list(config_spec)
    elif config_spec.cute_flash_gated_search_enabled:
        block_size_lists = _cute_flash_gated_block_size_lists(config_spec)
        if config["block_sizes"] in block_size_lists:
            return
        # The legal tile nearest the request (same query tile, closest
        # KV tile), so a mutation of one block size is not reset to the
        # default tile.
        block_size_targets = _cute_flash_gated_nearest_block_sizes(
            config_spec, config["block_sizes"], block_size_lists
        )
    else:
        return
    value = config["block_sizes"]
    raw_block_sizes = [*value] if isinstance(value, (list, tuple)) else [value]
    if raw_block_sizes == block_size_targets:
        return
    config["block_sizes"] = list(block_size_targets)


def _cute_flash_block_size_target_list(config_spec: ConfigSpec) -> list[int]:
    targets: list[int | None] = [None] * len(config_spec.block_sizes)
    for block_id, target in config_spec._cute_flash_block_size_targets.items():
        targets[config_spec.block_sizes.block_id_to_index(block_id)] = target
    if any(target is None for target in targets):
        raise InvalidConfig("CuTe flash attention search has incomplete block sizes")
    return [target for target in targets if target is not None]


def _normalize_cute_flash_default_sequence(
    config_spec: ConfigSpec,
    config: dict[str, object],
    key: str,
    default: object,
) -> None:
    value = config.get(key)
    if not value:
        config.pop(key, None)
        return
    if not isinstance(value, list) or any(item != default for item in value):
        config.pop(key, None)
        return
    config.pop(key, None)


def _normalize_cute_flash_default_loop_orders(
    config_spec: ConfigSpec, config: dict[str, object]
) -> None:
    value = config.get("loop_orders")
    if not value:
        config.pop("loop_orders", None)
        return
    defaults = [spec._fill_missing() for spec in config_spec.loop_orders]
    if value != defaults:
        config.pop("loop_orders", None)
        return
    config.pop("loop_orders", None)


def _is_cute_flash_config_envelope(
    config_spec: ConfigSpec, config: dict[str, object], block_size_targets: list[int]
) -> bool:
    if config.get("block_sizes") != block_size_targets:
        return False
    if config.get("pid_type", "flat") != "flat":
        return False
    if "epilogue_subtile" in config:
        return False
    for key, default in (
        ("l2_groupings", 1),
        ("num_threads", 0),
        ("cute_vector_widths", 1),
        ("cute_lane_layouts", "blocked"),
    ):
        value = config.get(key)
        if value and (
            not isinstance(value, list) or any(item != default for item in value)
        ):
            return False
    loop_orders = config.get("loop_orders")
    if loop_orders:
        defaults = [spec._fill_missing() for spec in config_spec.loop_orders]
        if loop_orders != defaults:
            return False
    return True


def cute_flash_seed_configs(config_spec: ConfigSpec) -> list[Config]:
    """The legalized flash compiler seeds of this spec's attention surface.

    One builder serves both the spec's seed union and
    ``CuteFlashAttentionHeuristic``: the two call sites used to pass
    different fact subsets (one dropped ``kv_group``, the other the MLA
    value head dim, the query length and the flex hook facts), so the
    heuristic's ranked list and the spec's list disagreed on GQA, MLA and
    flex surfaces.
    """
    flash_seeds = flash_seeds_from_facts(
        config_spec.flash_facts,
        num_sm=config_spec.num_sm,
        block_size_targets=_cute_flash_block_size_target_list(config_spec),
        target_device_capability=config_spec.target_device_capability,
    )
    return _legalize_cute_flash_compiler_seeds(config_spec, flash_seeds)


def prepare_config_generation_overrides(
    config_spec: ConfigSpec, overrides: Mapping[str, object] | None
) -> tuple[dict[str, object] | None, str | None]:
    effective_overrides = dict(overrides) if overrides else None
    family_override: str | None = None
    if config_spec.cute_flash_search_enabled and effective_overrides:
        upgrade_legacy_flash_config(
            effective_overrides,
            lambda legacy: _resolve_cute_flash_config(config_spec, legacy),
        )
        requested_packet = effective_overrides.get(FLASH_EXP2_PACKET_KEY)
        if flash_exp2_packet_is_compound(requested_packet):
            packet_requirements = flash_compound_exp2_packet_overrides(
                config_spec.flash_facts,
                {
                    key: value
                    for key, value in effective_overrides.items()
                    if key != FLASH_PIPELINE_FAMILY_KEY
                },
            )
            if not packet_requirements:
                raise InvalidConfig(
                    f"{FLASH_EXP2_PACKET_KEY}={requested_packet!r} is not "
                    "effective for this kernel"
                )
            for key, required_value in packet_requirements.items():
                if (
                    key in effective_overrides
                    and effective_overrides[key] != required_value
                ):
                    raise InvalidConfig(
                        f"{FLASH_EXP2_PACKET_KEY}={requested_packet!r} "
                        f"requires {key}={required_value!r}, got "
                        f"{effective_overrides[key]!r}"
                    )
                effective_overrides[key] = required_value

        exact_family = effective_overrides.get(FLASH_PIPELINE_FAMILY_KEY)
        family_flags = _flash_pipeline_family_flags(exact_family)
        if family_flags is not None:
            family_override = cast("str", exact_family)
            if family_flags.use_clc_scheduler or family_flags.local_tma_partition:
                if effective_overrides.get(FLASH_PERSISTENT_KEY) is False:
                    raise InvalidConfig(
                        f"{FLASH_PIPELINE_FAMILY_KEY}={family_override!r} "
                        f"requires {FLASH_PERSISTENT_KEY}=True"
                    )
                effective_overrides[FLASH_PERSISTENT_KEY] = True

        if config_spec._cute_flash_output_requires_tma:
            if effective_overrides.get(FLASH_EPI_TMA_KEY) is False:
                raise InvalidConfig(
                    f"{FLASH_EPI_TMA_KEY}=False is not legal for this output shape"
                )
            effective_overrides[FLASH_EPI_TMA_KEY] = True
    return effective_overrides, family_override


def prepare_override_normalization(
    config_spec: ConfigSpec, config: dict[str, object], overrides: Mapping[str, object]
) -> None:
    if config_spec.cute_flash_search_enabled:
        # A sampled resident mode must honor fixed parent controls too.
        # Reject incompatible samples so generation can retry another
        # lowering instead of silently overwriting an explicit override.
        requested_lowering = overrides.get(
            FLASH_SOFTMAX_LOWERING_KEY, config.get(FLASH_SOFTMAX_LOWERING_KEY)
        )
        requirements = _flash_resident_softmax_overrides(requested_lowering)
        for key, required_value in requirements.items():
            if key in overrides and overrides[key] != required_value:
                raise InvalidConfig(
                    f"{FLASH_SOFTMAX_LOWERING_KEY}={requested_lowering!r} "
                    f"requires {key}={required_value!r}, got {overrides[key]!r}"
                )
        if FLASH_SOFTMAX_LOWERING_KEY in overrides:
            config.update(requirements)
            _resolve_cute_flash_config(config_spec, config)
    family = overrides.get(FLASH_PIPELINE_FAMILY_KEY)
    family_flags = _flash_pipeline_family_flags(family)
    if config_spec.cute_flash_search_enabled and family_flags is not None:
        effective_family = _resolve_cute_flash_pins(
            config_spec, ((FLASH_PIPELINE_FAMILY_KEY, family),)
        ).pipeline_family
        if effective_family != family:
            raise InvalidConfig(
                f"cute_flash_pipeline_family={family!r} is not effective "
                f"for this kernel; it normalizes to {effective_family!r}"
            )
    if config_spec.cute_flash_search_enabled and FLASH_ROWMAX_KEY in overrides:
        requested_rowmax = overrides[FLASH_ROWMAX_KEY]
        effective_rowmax = _resolve_cute_flash_config(config_spec, config).rowmax
        if effective_rowmax != requested_rowmax:
            raise InvalidConfig(
                f"cute_flash_rowmax={requested_rowmax!r} is not effective "
                f"for this target and score layout; it normalizes to "
                f"{effective_rowmax!r}"
            )
    if config_spec.cute_flash_search_enabled and FLASH_EXP2_PACKET_KEY in overrides:
        packet_requirements = flash_compound_exp2_packet_overrides(
            config_spec.flash_facts,
            {
                key: value
                for key, value in overrides.items()
                if key != FLASH_PIPELINE_FAMILY_KEY
            },
        )
        requested_packet = overrides[FLASH_EXP2_PACKET_KEY]
        for key, required_value in packet_requirements.items():
            if key in overrides and overrides[key] != required_value:
                raise InvalidConfig(
                    f"cute_flash_exp2_packet={requested_packet!r} "
                    f"requires {key}={required_value!r}, got "
                    f"{overrides[key]!r}"
                )
        if family_flags is not None:
            effective_packet = _resolve_cute_flash_pins(
                config_spec,
                (
                    (FLASH_PIPELINE_FAMILY_KEY, family),
                    (FLASH_EXP2_PACKET_KEY, requested_packet),
                ),
            ).exp2_packet
            if effective_packet != requested_packet:
                raise InvalidConfig(
                    f"cute_flash_exp2_packet={requested_packet!r} is not "
                    f"effective with cute_flash_pipeline_family={family!r}"
                )
    if (
        config_spec.cute_flash_search_enabled
        and family_flags is not None
        and (family_flags.use_clc_scheduler or family_flags.local_tma_partition)
    ):
        if overrides.get(FLASH_PERSISTENT_KEY) is False:
            raise InvalidConfig(
                f"cute_flash_pipeline_family={family!r} requires "
                "cute_flash_persistent=True"
            )
        config[FLASH_PERSISTENT_KEY] = True
    surface.prepare_override_normalization(
        config_spec.tcgen05,
        config,
        overrides,
    )


def fingerprint_extra(config_spec: ConfigSpec) -> object | None:
    """The flash target policy identity folded into the persistent cache key."""
    if not config_spec.cute_flash_search_enabled:
        return None
    from ..flash_policy import flash_target_policy_cache_identity

    return flash_target_policy_cache_identity(
        config_spec.target_device_capability,
        head_dim=config_spec._cute_flash_head_dim,
        torch_dtype=str(config_spec._cute_flash_dtype).removeprefix("torch."),
        num_kv=config_spec._cute_flash_num_kv,
        is_causal=config_spec._cute_flash_is_causal,
    )
