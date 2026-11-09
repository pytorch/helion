"""The CuTe flash search's structural pieces the generic autotuner delegates to.

:class:`FlashCoverageDesign` is the deterministic covering design of a flash
``ConfigGeneration``'s active choices (the structural leaf catalog, the
pipeline / CLC lane catalogs and witnesses, the two prefix counts and the
coverage telemetry), computed once per generation by
:func:`design_flash_coverage`;
``CuteConfigGenerationExtension._flash_coverage_design``
(``cute/autotune/config_generation_ext.py``) memoizes it and the extension's
``flash_*`` accessors read its fields.  :class:`FlashStructuralQualification`
is the structural qualification phase
``LFBOPatternSearch._run_flash_structural_qualification`` delegates to.

Both are imported at call time (``design_flash_coverage`` by
``config_generation_ext.py``, ``FlashStructuralQualification`` by
``helion/autotuner/surrogate_pattern_search.py``) because this module imports
the autotuner modules at top level.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from typing import TYPE_CHECKING
from typing import cast

from ....autotuner.base_search import performance
from ....autotuner.config_fragment import EnumFragment
from ....autotuner.search_space_logger import canonical_config_id
from ....autotuner.surrogate_pattern_search import CUTE_FLASH_LANE_POLICY_VERSION
from ....autotuner.surrogate_pattern_search import FLASH_EXACT_SPACE_RAW_FLOOR
from ....autotuner.surrogate_pattern_search import flash_terminal_measurement_is_valid
from ....exc import InvalidConfig
from ..cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY
from ..cute_flash import FLASH_EXP2_PACKET_KEY
from ..cute_flash import FLASH_KV_STAGE_KEY
from ..cute_flash import FLASH_PIPELINE_FAMILY_FLAGS
from ..cute_flash import FLASH_PIPELINE_FAMILY_KEY
from ..cute_flash import FLASH_S_STAGE_KEY
from ..cute_flash import flash_structural_leaf_from_config
from ..flash_knobs import FLASH_AUTOTUNE_CONFIG_KEYS
from ..flash_knobs import FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
from ..flash_knobs import FLASH_AUTOTUNE_VALUE_DEPENDENCIES
from .config_generation_ext import cute_generation_extension

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ....autotuner.base_search import PopulationMember
    from ....autotuner.config_generation import ConfigGeneration
    from ....autotuner.config_generation import FlatConfig
    from ....autotuner.surrogate_pattern_search import LFBOPatternSearch
    from ....runtime.config import Config
    from ..cute_flash import FlashStructuralLeaf

_FLASH_STRUCTURAL_LEAF_GOAL_KEY = "__flash_structural_leaf__"


def _flash_log_maximin_refinements(
    legal_values: Sequence[int],
    anchor_values: Sequence[int],
) -> tuple[int, ...]:
    """Order every non-anchor legal value by positive log-space refinement."""
    legal = tuple(dict.fromkeys(value for value in legal_values if value > 0))
    anchors = tuple(
        value for value in dict.fromkeys(anchor_values) if value > 0 and value in legal
    )
    if not anchors:
        return legal

    selected = set(anchors)
    refinements: list[int] = []
    value_logs = {value: math.log(value) for value in legal}
    while len(selected) < len(legal):
        remaining = [value for value in legal if value not in selected]
        if not remaining:
            break
        refinement = max(
            remaining,
            key=lambda value: (
                min(abs(value_logs[value] - value_logs[chosen]) for chosen in selected),
                -value,
            ),
        )
        selected.add(refinement)
        refinements.append(refinement)
    return tuple(refinements)


@dataclasses.dataclass(frozen=True)
class FlashCoverageDesign:
    """A flash generation's structural covering design and its telemetry.

    The defaults are the empty design (no flash surface, or no searchable
    axis).  Accessors return copies; the design itself is never mutated.
    """

    coverage_flats: list[FlatConfig] = dataclasses.field(default_factory=list)
    active_values: list[tuple[str, object]] = dataclasses.field(default_factory=list)
    uncovered_values: list[tuple[str, object]] = dataclasses.field(default_factory=list)
    # Families whose neutral request the overrides normalize away (telemetry,
    # not a design defect).
    override_unreachable_families: list[object] = dataclasses.field(
        default_factory=list
    )
    underqualified_values: list[tuple[str, object, int]] = dataclasses.field(
        default_factory=list
    )
    leaf_catalog: list[FlashStructuralLeaf] = dataclasses.field(default_factory=list)
    pipeline_lane_catalog: dict[FlashStructuralLeaf, tuple[tuple[str, object], ...]] = (
        dataclasses.field(default_factory=dict)
    )
    pipeline_lane_witnesses: dict[tuple[FlashStructuralLeaf, str, object], Config] = (
        dataclasses.field(default_factory=dict)
    )
    clc_lane_catalog: dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]] = (
        dataclasses.field(default_factory=dict)
    )
    clc_lane_witnesses: dict[tuple[FlashStructuralLeaf, int], Config] = (
        dataclasses.field(default_factory=dict)
    )
    underqualified_leaves: list[tuple[FlashStructuralLeaf, int]] = dataclasses.field(
        default_factory=list
    )
    singleton_leaves: list[FlashStructuralLeaf] = dataclasses.field(
        default_factory=list
    )
    uncovered_interactions: list[tuple[tuple[str, ...], tuple[object, ...]]] = (
        dataclasses.field(default_factory=list)
    )
    active_interactions: list[tuple[tuple[str, ...], tuple[object, ...]]] = (
        dataclasses.field(default_factory=list)
    )
    parent_coverage_prefix_count: int = 0
    qualification_prefix_count: int = 0


def design_flash_coverage(generation: ConfigGeneration) -> FlashCoverageDesign:
    """Build a normalized covering design for the active flash choices.

    Candidate contexts are normalized before coverage is measured. Missing
    effective values receive a one- or two-axis witness, then a deterministic
    greedy reduction keeps a compact set covering every reachable choice.
    This avoids both a Cartesian product and measured-winner presets; uniform
    random candidates fill the rest of the requested population.
    """
    if not generation.config_spec.cute_flash_search_enabled:
        return FlashCoverageDesign()

    axes: dict[str, tuple[int, tuple[object, ...]]] = {}
    enum_axes: dict[str, tuple[int, EnumFragment]] = {}
    for key in FLASH_AUTOTUNE_CONFIG_KEYS:
        key_layout = generation._key_to_flat_indices.get(key)
        if key_layout is None:
            continue
        indices, is_sequence = key_layout
        if is_sequence or len(indices) != 1:
            continue
        index = indices[0]
        fragment = generation.flat_spec[index]
        if not isinstance(fragment, EnumFragment):
            continue
        enum_axes[key] = (index, fragment)
        choices = (
            fragment.coverage_choices
            if fragment.coverage_choices is not None
            else fragment.choices
            if fragment.search_choices is None
            else fragment.search_choices
        )
        structural_parent = key in (
            FLASH_PIPELINE_FAMILY_KEY,
            FLASH_EXP2_PACKET_KEY,
        )
        if (
            len(choices) <= 1 and not structural_parent
        ) or index in generation.overridden_flat_indices:
            continue
        axes[key] = (index, choices)
    clc_coverage_choices: tuple[object, ...] | None = None
    clc_layout = generation._key_to_flat_indices.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
    if clc_layout is not None and not clc_layout[1] and len(clc_layout[0]) == 1:
        clc_fragment = generation.flat_spec[clc_layout[0][0]]
        if isinstance(clc_fragment, EnumFragment):
            clc_coverage_choices = (
                clc_fragment.coverage_choices
                if clc_fragment.coverage_choices is not None
                else clc_fragment.choices
                if clc_fragment.search_choices is None
                else clc_fragment.search_choices
            )
    if not axes:
        return FlashCoverageDesign()

    base = generation._fragment_default_flat()
    raw_contexts: list[FlatConfig] = [copy.deepcopy(base)]

    def append_variant(values: Mapping[str, object]) -> None:
        flat = copy.deepcopy(base)
        for key, value in values.items():
            index = axes[key][0] if key in axes else enum_axes[key][0]
            flat[index] = value
        raw_contexts.append(flat)

    # Keep every family/protocol leaf reachable from an unconfounded point.
    # The full packet product is benchmarked by structural qualification;
    # adding it to the candidate pool here also lets the compact generation
    # zero design choose a neutral witness for each exact leaf.
    for anchor in cute_generation_extension(
        generation
    ).flash_low_confound_schedule_anchor_configs():
        raw_contexts.append(generation.flatten(anchor))

    # Family contexts let packet normalization apply its required parent and
    # child fields without embedding any arithmetic performance choices.
    for key in (FLASH_PIPELINE_FAMILY_KEY, FLASH_EXP2_PACKET_KEY):
        axis = axes.get(key)
        if axis is not None:
            for value in axis[1]:
                append_variant({key: value})

    # Declared dependency witnesses: an active value that is legal only
    # together with other knob values gets a context carrying all of them.
    for (
        dependent_key,
        dependent_value,
    ), deps in FLASH_AUTOTUNE_VALUE_DEPENDENCIES.items():
        axis = axes.get(dependent_key)
        if axis is None or dependent_value not in axis[1]:
            continue
        if any(key not in enum_axes for key in deps):
            continue
        append_variant({**deps, dependent_key: dependent_value})

    interaction_groups = tuple(
        group
        for group in FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
        if all(key in enum_axes for key in group) and any(key in axes for key in group)
    )
    for group in interaction_groups:
        for combination in itertools.product(
            *(
                axes[key][1]
                if key in axes
                else (
                    generation._override_values.get(key, enum_axes[key][1].default()),
                )
                for key in group
            )
        ):
            append_variant(dict(zip(group, combination, strict=True)))

    max_levels = max(len(choices) for _index, choices in axes.values())
    for pass_index in range(2):
        for row in range(max_levels):
            values: dict[str, object] = {}
            for axis_index, (key, (_index, choices)) in enumerate(axes.items()):
                stride = 2 * axis_index + 1
                while math.gcd(stride, len(choices)) != 1:
                    stride += 2
                offset = axis_index if pass_index == 0 else -axis_index - 1
                values[key] = choices[(row * stride + offset) % len(choices)]
            append_variant(values)

    goal_order = [
        (key, value) for key, (_index, choices) in axes.items() for value in choices
    ]
    structural_value_qualification_goal_order = [
        goal for goal in goal_order if goal[0] == FLASH_PIPELINE_FAMILY_KEY
    ]
    structural_value_qualification_goals = set(
        structural_value_qualification_goal_order
    )
    design_active_values = copy.deepcopy(goal_order)
    candidate_pool: list[
        tuple[
            Config,
            FlatConfig,
            frozenset[tuple[str | tuple[str, ...], object]],
        ]
    ] = []
    seen: set[Config] = set()
    normalized_cache: dict[
        tuple[str, bool],
        tuple[
            Config,
            FlatConfig,
            frozenset[tuple[str | tuple[str, ...], object]],
        ]
        | None,
    ] = {}

    def normalize(
        raw: FlatConfig,
        *,
        allow_clc_refinement: bool = False,
    ) -> (
        tuple[
            Config,
            FlatConfig,
            frozenset[tuple[str | tuple[str, ...], object]],
        ]
        | None
    ):
        cache_key = (repr(raw), allow_clc_refinement)
        if cache_key in normalized_cache:
            return normalized_cache[cache_key]
        try:
            config = generation.unflatten(copy.deepcopy(raw))
        except InvalidConfig:
            normalized_cache[cache_key] = None
            return None
        family = config.config.get(FLASH_PIPELINE_FAMILY_KEY)
        family_flags = FLASH_PIPELINE_FAMILY_FLAGS.get(cast("str", family))
        if (
            family_flags is not None
            and family_flags.use_clc_scheduler
            and clc_coverage_choices is not None
            and clc_layout is not None
            and clc_layout[0][0] not in generation.overridden_flat_indices
            and config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
            not in clc_coverage_choices
            and not allow_clc_refinement
        ):
            # CLC zero means an input-layout-dependent automatic value and can
            # alias an advertised divisor. It is valid for fixed configs, but
            # cannot serve as a distinct structural-search witness.
            normalized_cache[cache_key] = None
            return None
        flat = generation.flatten(config)
        leaf = flash_structural_leaf_from_config(config.config)
        covers = frozenset(
            [
                (key, value)
                for key, (_index, choices) in axes.items()
                if (value := config.config.get(key)) in choices
            ]
            + [
                (group, tuple(config.config.get(key) for key in group))
                for group in interaction_groups
            ]
            + ([(_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)] if leaf is not None else [])
        )
        result = config, flat, covers
        normalized_cache[cache_key] = result
        return result

    def add_candidate(
        normalized: (
            tuple[
                Config,
                FlatConfig,
                frozenset[tuple[str | tuple[str, ...], object]],
            ]
            | None
        ),
    ) -> None:
        if normalized is None or normalized[0] in seen:
            return
        seen.add(normalized[0])
        candidate_pool.append(normalized)

    for raw in raw_contexts:
        add_candidate(normalize(raw))

    # The surface advertises a family because its neutral request survives
    # normalization; an override can still make it unreachable (a family
    # whose only output path is the TMA store under a direct-epilogue
    # override). It stays in the uncovered telemetry but is not a defect
    # of the design.
    unreachable_families: list[object] = []
    family_axis = axes.get(FLASH_PIPELINE_FAMILY_KEY)
    if family_axis is not None:
        for value in family_axis[1]:
            flat = copy.deepcopy(base)
            flat[family_axis[0]] = value
            normalized = normalize(flat)
            if (
                normalized is None
                or normalized[0].config.get(FLASH_PIPELINE_FAMILY_KEY) != value
            ):
                unreachable_families.append(value)
    design_override_unreachable_families = unreachable_families

    interaction_goal_order: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
    seen_interactions: set[tuple[tuple[str, ...], tuple[object, ...]]] = set()
    for config, _flat, _covers in candidate_pool:
        for group in interaction_groups:
            goal = (group, tuple(config.config.get(key) for key in group))
            if goal not in seen_interactions:
                seen_interactions.add(goal)
                interaction_goal_order.append(goal)
    goals: set[tuple[str | tuple[str, ...], object]] = {
        *goal_order,
        *interaction_goal_order,
    }
    design_active_interactions = copy.deepcopy(interaction_goal_order)

    def witness_count(goal: tuple[str, object]) -> int:
        return sum(goal in covers for _config, _flat, covers in candidate_pool)

    # Pin every goal against existing contexts. Families need two distinct
    # normalized witnesses so LFBO can rank each ordinary class and keep a
    # qualified path when that class advances beyond initial ranking. Compound
    # packets need one provenance row; their measured candidates are transferred
    # from retained ordinary representatives. Conditional children such as CLC
    # controls inherit the structural parent that makes them effective, while
    # other values remain one-way probes.
    for key, requested in goal_order:
        index, _choices = axes[key]
        goal = (key, requested)
        target_count = 2 if goal in structural_value_qualification_goals else 1
        found = witness_count(goal) >= target_count
        for context in raw_contexts:
            if found:
                break
            raw = copy.deepcopy(context)
            raw[index] = requested
            normalized = normalize(raw)
            if normalized is None or normalized[0].config.get(key) != requested:
                continue
            add_candidate(normalized)
            found = witness_count(goal) >= target_count
        if found:
            continue

        # Some children need two parents (for example a staged-output policy
        # needs staged output enabled and TMA output disabled). Probe one
        # additional active axis generically rather than encoding that graph.
        for context in raw_contexts:
            if found:
                break
            for other_key, (other_index, other_choices) in axes.items():
                if other_key == key:
                    continue
                for other_value in other_choices:
                    raw = copy.deepcopy(context)
                    raw[index] = requested
                    raw[other_index] = other_value
                    normalized = normalize(raw)
                    if normalized is None or normalized[0].config.get(key) != requested:
                        continue
                    add_candidate(normalized)
                    found = witness_count(goal) >= target_count
                    if found:
                        break
                if found:
                    break

    # Give the reducer legal pairwise opportunities to combine a structural
    # qualification with an ordinary child witness. These normalized rows
    # are only a candidate pool for the covering design; the autotuner still
    # measures the reduced design plus unconstrained candidates, not this
    # pairwise product.
    witness_contexts = {
        goal: [
            copy.deepcopy(flat)
            for _config, flat, covers in candidate_pool
            if goal in covers
        ]
        for goal in goal_order
    }
    for parent_key, parent_value in structural_value_qualification_goal_order:
        parent_index, _choices = axes[parent_key]
        parent_goal = (parent_key, parent_value)
        for child_goal in goal_order:
            if child_goal[0] == parent_key:
                continue
            for context in witness_contexts[child_goal]:
                raw = copy.deepcopy(context)
                raw[parent_index] = parent_value
                normalized = normalize(raw)
                if normalized is None:
                    continue
                if parent_goal not in normalized[2] or child_goal not in normalized[2]:
                    continue
                add_candidate(normalized)
                break

    # A structural leaf is an ordinary family schedule or one exact
    # compound-packet schedule under its normalized required family. Expand
    # ordinary witnesses by one live axis when needed so every nonsingleton
    # ordinary leaf has two candidates available to qualification. Compound
    # leaves are populated later from measured ordinary representatives.
    leaf_catalog: list[FlashStructuralLeaf] = []
    seen_leaves: set[FlashStructuralLeaf] = set()
    for config, _flat, _covers in candidate_pool:
        leaf = flash_structural_leaf_from_config(config.config)
        if leaf is not None and leaf not in seen_leaves:
            seen_leaves.add(leaf)
            leaf_catalog.append(leaf)
    for leaf in leaf_catalog:
        if leaf.compound_exp2_packet is not None:
            continue
        leaf_goal = (_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)
        if witness_count(leaf_goal) >= 2:
            continue
        witnesses = [
            copy.deepcopy(flat)
            for _config, flat, covers in candidate_pool
            if leaf_goal in covers
        ]
        for context in witnesses:
            if witness_count(leaf_goal) >= 2:
                break
            for index, choices in axes.values():
                original = context[index]
                for value in choices:
                    if value == original:
                        continue
                    raw = copy.deepcopy(context)
                    raw[index] = value
                    normalized = normalize(raw)
                    if normalized is None or leaf_goal not in normalized[2]:
                        continue
                    add_candidate(normalized)
                    if witness_count(leaf_goal) >= 2:
                        break
                if witness_count(leaf_goal) >= 2:
                    break

    design_leaf_catalog = copy.deepcopy(leaf_catalog)
    pipeline_lane_catalog: dict[
        FlashStructuralLeaf, tuple[tuple[str, object], ...]
    ] = {}
    pipeline_lane_witnesses: dict[tuple[FlashStructuralLeaf, str, object], Config] = {}
    for leaf in leaf_catalog:
        if leaf.compound_exp2_packet is not None:
            pipeline_lane_catalog[leaf] = ()
            continue
        leaf_contexts = [
            flat
            for config, flat, _covers in candidate_pool
            if flash_structural_leaf_from_config(config.config) == leaf
        ]
        lanes: list[tuple[str, object]] = []
        for key in (FLASH_KV_STAGE_KEY, FLASH_S_STAGE_KEY):
            axis = enum_axes.get(key)
            if axis is None or axis[0] in generation.overridden_flat_indices:
                continue
            index, fragment = axis
            active_values = (
                fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            )
            leaf_values: list[object] = []
            for value in active_values:
                for context in leaf_contexts:
                    raw = copy.deepcopy(context)
                    raw[index] = value
                    normalized = normalize(raw)
                    if normalized is None:
                        continue
                    config = normalized[0]
                    if (
                        flash_structural_leaf_from_config(config.config) != leaf
                        or config.config.get(key) != value
                    ):
                        continue
                    leaf_values.append(value)
                    pipeline_lane_witnesses[(leaf, key, value)] = config
                    break
            if len(leaf_values) > 1:
                lanes.extend((key, value) for value in leaf_values)
            else:
                for value in leaf_values:
                    pipeline_lane_witnesses.pop((leaf, key, value))
        pipeline_lane_catalog[leaf] = tuple(lanes)
    design_pipeline_lane_catalog = copy.deepcopy(pipeline_lane_catalog)
    design_pipeline_lane_witnesses = copy.deepcopy(pipeline_lane_witnesses)
    clc_lane_catalog: dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]] = {}
    clc_lane_witnesses: dict[tuple[FlashStructuralLeaf, int], Config] = {}
    if (
        clc_layout is not None
        and not clc_layout[1]
        and len(clc_layout[0]) == 1
        and clc_layout[0][0] not in generation.overridden_flat_indices
    ):
        clc_index = clc_layout[0][0]
        clc_fragment = generation.flat_spec[clc_index]
        if isinstance(clc_fragment, EnumFragment):
            legal_values = tuple(
                cast("int", value)
                for value in clc_fragment.choices
                if type(value) is int and cast("int", value) > 0
            )
            active_values = (
                clc_fragment.choices
                if clc_fragment.search_choices is None
                else clc_fragment.search_choices
            )
            search_values = tuple(
                cast("int", value)
                for value in active_values
                if type(value) is int and cast("int", value) > 0
            )
            coverage_values = (
                clc_fragment.coverage_choices
                if clc_fragment.coverage_choices is not None
                else clc_fragment.choices
                if clc_fragment.search_choices is None
                else clc_fragment.search_choices
            )
            anchor_values = tuple(
                cast("int", value)
                for value in coverage_values
                if type(value) is int and cast("int", value) > 0
            )
            refinement_values = _flash_log_maximin_refinements(
                legal_values, anchor_values
            )
            attempted_values = (*anchor_values, *refinement_values)
            for leaf in leaf_catalog:
                if leaf.compound_exp2_packet is not None:
                    continue
                family_flags = FLASH_PIPELINE_FAMILY_FLAGS.get(leaf.pipeline_family)
                if family_flags is None or not family_flags.use_clc_scheduler:
                    continue
                leaf_contexts = [
                    flat
                    for config, flat, _covers in candidate_pool
                    if flash_structural_leaf_from_config(config.config) == leaf
                ]
                for value in attempted_values:
                    for context in leaf_contexts:
                        raw = copy.deepcopy(context)
                        raw[clc_index] = value
                        normalized = normalize(raw, allow_clc_refinement=True)
                        if normalized is None:
                            continue
                        config = normalized[0]
                        if (
                            flash_structural_leaf_from_config(config.config) != leaf
                            or config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) != value
                        ):
                            continue
                        clc_lane_witnesses[(leaf, value)] = config
                        break
                clc_lane_catalog[leaf] = {
                    "legal_values": legal_values,
                    "search_values": search_values,
                    "anchor_values": anchor_values,
                    "refinement_values": refinement_values,
                    "attempted_values": attempted_values,
                }
    design_clc_lane_catalog = copy.deepcopy(clc_lane_catalog)
    design_clc_lane_witnesses = copy.deepcopy(clc_lane_witnesses)
    leaf_goal_order = [(_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf) for leaf in leaf_catalog]
    marginal_parent_goal_order = [
        goal
        for goal in goal_order
        if goal[0] in (FLASH_PIPELINE_FAMILY_KEY, FLASH_EXP2_PACKET_KEY)
    ]
    # Exact leaves come first so a compound packet cannot stand in for its
    # ordinary parent-family schedule. Marginal packet goals remain in the
    # prefix because noncompound packets are arithmetic search choices.
    parent_goal_order = [*leaf_goal_order, *marginal_parent_goal_order]
    qualification_goal_order = [
        (_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)
        for leaf in leaf_catalog
        if leaf.compound_exp2_packet is None
        and witness_count((_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)) >= 2
    ]

    uncovered = set(goals)
    mandatory_uncovered = set(parent_goal_order)
    remaining = list(enumerate(candidate_pool))
    selected: list[FlatConfig] = []
    selected_covers: list[frozenset[tuple[str | tuple[str, ...], object]]] = []
    qualification_uncovered = {*goals, *qualification_goal_order}

    def select_multicover(
        remaining_counts: dict[tuple[str | tuple[str, ...], object], int],
    ) -> None:
        while remaining_counts and remaining:
            best_pos = max(
                range(len(remaining)),
                key=lambda pos: (
                    sum(
                        min(remaining_counts.get(goal, 0), 1)
                        for goal in remaining[pos][1][2]
                    ),
                    len(remaining[pos][1][2] & qualification_uncovered),
                    -remaining[pos][0],
                ),
            )
            _order, (_config, flat, covers) = remaining[best_pos]
            gain = [goal for goal in covers if goal in remaining_counts]
            if not gain:
                break
            remaining.pop(best_pos)
            selected.append(copy.deepcopy(flat))
            selected_covers.append(covers)
            qualification_uncovered.difference_update(covers)
            for goal in gain:
                remaining_count = remaining_counts[goal] - 1
                if remaining_count:
                    remaining_counts[goal] = remaining_count
                else:
                    remaining_counts.pop(goal)

    # Cover every exact leaf and marginal family/packet value once before a
    # second row is selected. The second phase qualifies exact nonsingleton
    # ordinary leaves; compound leaves receive measured top-K transfers.
    select_multicover(dict.fromkeys(parent_goal_order, 1))
    design_parent_coverage_prefix_count = len(selected)
    qualification_remaining: dict[tuple[str | tuple[str, ...], object], int] = {
        goal: 2 - sum(goal in covers for covers in selected_covers)
        for goal in qualification_goal_order
        if sum(goal in covers for covers in selected_covers) < 2
    }
    select_multicover(qualification_remaining)

    design_qualification_prefix_count = len(selected)
    for covers in selected_covers:
        uncovered.difference_update(covers)
        mandatory_uncovered.difference_update(covers)
    while uncovered and remaining:
        best_pos = max(
            range(len(remaining)),
            key=lambda pos: (
                len(remaining[pos][1][2] & mandatory_uncovered),
                len(remaining[pos][1][2] & uncovered),
                -remaining[pos][0],
            ),
        )
        _order, (_config, flat, covers) = remaining.pop(best_pos)
        gain = covers & uncovered
        if not gain:
            break
        selected.append(copy.deepcopy(flat))
        selected_covers.append(covers)
        uncovered.difference_update(gain)
        mandatory_uncovered.difference_update(gain)
    design_coverage_flats = selected
    design_uncovered_values = [goal for goal in goal_order if goal in uncovered]
    design_uncovered_interactions = [
        goal for goal in interaction_goal_order if goal in uncovered
    ]
    design_underqualified_values = [
        (
            key,
            value,
            sum((key, value) in covers for covers in selected_covers),
        )
        for key, value in structural_value_qualification_goal_order
        if sum((key, value) in covers for covers in selected_covers) < 2
    ]
    design_underqualified_leaves = [
        (
            leaf,
            sum(
                (_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf) in covers
                for covers in selected_covers
            ),
        )
        for leaf in leaf_catalog
        if leaf.compound_exp2_packet is None
        and witness_count((_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)) >= 2
        and sum(
            (_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf) in covers
            for covers in selected_covers
        )
        < 2
    ]
    # An ordinary leaf the witness expansion above could not give a
    # second distinct normalized config (every searchable knob of its
    # family overridden, say) is complete after one row.
    design_singleton_leaves = [
        leaf
        for leaf in leaf_catalog
        if leaf.compound_exp2_packet is None
        and witness_count((_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)) < 2
    ]
    return FlashCoverageDesign(
        coverage_flats=design_coverage_flats,
        active_values=design_active_values,
        uncovered_values=design_uncovered_values,
        override_unreachable_families=design_override_unreachable_families,
        underqualified_values=design_underqualified_values,
        leaf_catalog=design_leaf_catalog,
        pipeline_lane_catalog=design_pipeline_lane_catalog,
        pipeline_lane_witnesses=design_pipeline_lane_witnesses,
        clc_lane_catalog=design_clc_lane_catalog,
        clc_lane_witnesses=design_clc_lane_witnesses,
        underqualified_leaves=design_underqualified_leaves,
        singleton_leaves=design_singleton_leaves,
        uncovered_interactions=design_uncovered_interactions,
        active_interactions=design_active_interactions,
        parent_coverage_prefix_count=design_parent_coverage_prefix_count,
        qualification_prefix_count=design_qualification_prefix_count,
    )


class FlashStructuralQualification:
    """Qualify ordinary flash schedules, then transfer their best representatives.

    The structural phase of a flash ``LFBOPatternSearch``: measures every
    pipeline lane and CLC divisor of the leaf catalog, repairs failed
    witnesses, probes the families, transfers the compound-packet winners and
    writes the qualified population (and the ``_flash_qualified_*`` /
    ``_flash_parent_score_config_ids`` attributes ``_select_starting_paths``
    reads) back onto the search.  The lane helpers it calls stay methods of
    the search (``_flash_pipeline_lanes``, ``_flash_clc_depth_variant``, ...)
    so tests can override them.
    """

    def __init__(
        self,
        search: LFBOPatternSearch,
        visited: set[Config],
        *,
        initial_population: Sequence[PopulationMember] | None = None,
    ) -> None:
        self.search = search
        self.visited = visited
        self.initial_population = initial_population

    def run(self) -> int:
        """Run the qualification; returns the generation rounds it started."""
        search = self.search
        visited = self.visited
        initial_population = self.initial_population
        policy = getattr(search, "flash_structural_search", None)
        if (
            not search.config_spec.cute_flash_search_enabled
            or policy is None
            or search.max_generations <= 1
        ):
            return 0

        if initial_population is None:
            initial_population = search.population
        initial_population = list(initial_population)
        flash_generation = cute_generation_extension(search.config_gen)
        leaf_catalog = flash_generation.flash_structural_leaf_catalog()
        ordinary_leaves = [
            leaf for leaf in leaf_catalog if leaf.compound_exp2_packet is None
        ]
        compound_leaves = [
            leaf for leaf in leaf_catalog if leaf.compound_exp2_packet is not None
        ]
        clc_catalog = flash_generation.flash_clc_lane_catalog()
        initial_config_ids = [
            canonical_config_id(member.config) for member in initial_population
        ]
        initial_measurements = {
            id(member): {
                "attempt_perf": (
                    member.perfs[0]
                    if member.perfs and math.isfinite(member.perfs[0])
                    else None
                ),
                "selection_perf": (
                    member.perf if member.perfs and math.isfinite(member.perf) else None
                ),
                "status": member.status,
                "source_hash": search._flash_member_source_hash(member),
                "measurement_pass_index": 0,
            }
            for member in initial_population
        }
        # A small surface (the d256 pair: two families, eight searched knobs,
        # 384 raw combinations that normalize to 68 configs) must be
        # enumerated for its lanes to be provably exhausted; otherwise every
        # config is measured in the initial population, no conditional
        # candidate exists and the qualification reports the lanes incomplete.
        exact_space_raw_budget = max(
            FLASH_EXACT_SPACE_RAW_FLOOR,
            search.initial_population,
            len(initial_population),
        )
        exact_space = flash_generation.exact_effective_search_space_configs(
            exact_space_raw_budget
        )
        measured_initial_configs = {
            member.config for member in initial_population if member.perfs
        }

        def measured_space_exhausted(configs: Sequence[Config] | None) -> bool:
            return bool(configs) and all(
                config in measured_initial_configs for config in configs or ()
            )

        def space_config_count(configs: Sequence[Config] | None) -> int | None:
            return None if configs is None else len(configs)

        def hierarchical_clc_values_covered(
            leaf: FlashStructuralLeaf,
            configs: Sequence[Config] | None,
        ) -> bool:
            catalog = clc_catalog.get(leaf)
            if catalog is None:
                return True
            present_values = {
                config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                for config in configs or ()
            }
            return set(catalog["attempted_values"]).issubset(present_values)

        def hierarchical_space_exhausted(
            leaf: FlashStructuralLeaf,
            configs: Sequence[Config] | None,
        ) -> bool:
            return measured_space_exhausted(
                configs
            ) and hierarchical_clc_values_covered(leaf, configs)

        exact_space_exhausted = measured_space_exhausted(exact_space) and all(
            hierarchical_clc_values_covered(leaf, exact_space) for leaf in clc_catalog
        )
        leaf_metrics: dict[FlashStructuralLeaf, dict[str, object]] = {}
        leaf_pipeline_lanes: dict[
            FlashStructuralLeaf, tuple[tuple[str, object], ...]
        ] = {}
        lane_metrics: dict[
            tuple[FlashStructuralLeaf, tuple[str, object]], dict[str, object]
        ] = {}
        for leaf in ordinary_leaves:
            initial_members = [
                member
                for member in initial_population
                if search._flash_structural_leaf(member) == leaf
            ]
            lanes = search._flash_pipeline_lanes(leaf)
            leaf_pipeline_lanes[leaf] = lanes
            leaf_space_configs = (
                None
                if exact_space is None
                else [
                    config
                    for config in exact_space
                    if search._flash_structural_leaf_from_config(config) == leaf
                ]
            )
            lane_space_configs = {
                lane: (
                    None
                    if exact_space is None
                    else [
                        config
                        for config in exact_space
                        if search._flash_structural_leaf_from_config(config) == leaf
                        and config.config.get(lane[0]) == lane[1]
                    ]
                )
                for lane in lanes
            }
            leaf_metrics[leaf] = {
                "family": leaf.pipeline_family,
                "compound_packet": leaf.compound_exp2_packet,
                "softmax_disc": leaf.softmax_disc,
                "initial_config_ids": [
                    canonical_config_id(member.config) for member in initial_members
                ],
                "space_exhausted": hierarchical_space_exhausted(
                    leaf, leaf_space_configs
                ),
                "space_config_count": space_config_count(leaf_space_configs),
                "ordinary_search_required": not lanes
                and not hierarchical_space_exhausted(leaf, leaf_space_configs),
                "rounds": [],
                "pipeline_lanes": [
                    {
                        "key": lane[0],
                        "value": lane[1],
                        "initial_config_ids": [
                            canonical_config_id(member.config)
                            for member in initial_members
                            if search._flash_member_matches_pipeline_lane(member, lane)
                        ],
                        "rounds": [],
                        "witness_attempted": False,
                        "witness_config_id": None,
                        "witness_succeeded": False,
                        "space_exhausted": hierarchical_space_exhausted(
                            leaf, lane_space_configs[lane]
                        ),
                        "space_config_count": space_config_count(
                            lane_space_configs[lane]
                        ),
                        "conditional_required": bool(
                            policy.conditional_candidates_per_pipeline_lane
                        )
                        and not hierarchical_space_exhausted(
                            leaf, lane_space_configs[lane]
                        ),
                        "conditional_candidate_ids": [],
                        "successful_conditional_candidate_ids": [],
                        "repair_candidate_ids": [],
                        "successful_repair_candidate_ids": [],
                        "repair_parent_decisions": [],
                        # A conditional job found no unmeasured config within
                        # the neighbor budget of any measured parent in the
                        # lane: everything reachable is measured, which is
                        # what an enumerated space would have proven.
                        "conditional_exhausted": False,
                        "terminal_failure_exhausted": False,
                        "complete": False,
                    }
                    for lane in lanes
                ],
            }
            cast_lanes = leaf_metrics[leaf]["pipeline_lanes"]
            assert isinstance(cast_lanes, list)
            for lane, lane_metric in zip(lanes, cast_lanes, strict=True):
                lane_metrics[(leaf, lane)] = lane_metric
        # Selection must replay the same normalized catalog that qualification
        # measured. Re-deriving it from later children could make provenance
        # base-dependent and promote an unrecorded lane.
        search._flash_qualified_pipeline_lanes = dict(leaf_pipeline_lanes)
        qualified_population = {id(member): member for member in search.population}
        candidate_ids: set[str] = set()
        leaves_with_candidates: set[FlashStructuralLeaf] = set()
        rounds_started = 0
        generation_rounds_started = 0
        rounds_completed = 0
        budget_exhausted = False
        qualification_neighbor_limit = search._flash_qualification_neighbor_limit()

        def member_measurement_state(
            member: PopulationMember,
        ) -> dict[str, object]:
            return {
                "attempt_perf": (
                    member.perfs[0]
                    if member.perfs and math.isfinite(member.perfs[0])
                    else None
                ),
                "selection_perf": (
                    member.perf if member.perfs and math.isfinite(member.perf) else None
                ),
                "status": member.status,
                "source_hash": search._flash_member_source_hash(member),
            }

        measurement_states_by_id: dict[str, dict[str, object]] = {}
        measurement_timeline: list[dict[str, object]] = []

        def record_measurement_pass(pass_index: int) -> None:
            """Record new or changed member states after one qualification pass."""
            nonlocal measurement_states_by_id
            current: dict[str, dict[str, object]] = {}
            for member in qualified_population.values():
                if not member.perfs:
                    continue
                config_id = canonical_config_id(member.config)
                state = member_measurement_state(member)
                existing = current.get(config_id)
                if existing is not None and existing != state:
                    raise AssertionError(
                        f"conflicting measurement state for config {config_id}"
                    )
                current[config_id] = state
            if not set(measurement_states_by_id) <= set(current):
                raise AssertionError(
                    "qualified measurement state removed between passes"
                )
            updates = [
                {"config_id": config_id, **current[config_id]}
                for config_id in sorted(current)
                if measurement_states_by_id.get(config_id) != current[config_id]
            ]
            measurement_timeline.append({"pass_index": pass_index, "updates": updates})
            measurement_states_by_id = current

        record_measurement_pass(0)

        def measurement_result(member: PopulationMember) -> dict[str, object]:
            """Snapshot the measurements visible at the current pass boundary."""
            state = member_measurement_state(member)
            config_id = canonical_config_id(member.config)
            if not member.perfs:
                return {
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "unknown",
                    "source_hash": None,
                    "measurement_pass_index": None,
                }
            if measurement_states_by_id.get(config_id) != state:
                raise AssertionError(
                    f"measurement state for {config_id} was not recorded at pass "
                    f"{rounds_completed}"
                )
            return {**state, "measurement_pass_index": rounds_completed}

        def decision_member_result(member: PopulationMember) -> dict[str, object]:
            """Snapshot the measurements visible to one structural decision."""
            return {
                "config_id": canonical_config_id(member.config),
                **measurement_result(member),
            }

        def ranked_decision_results(
            members: Sequence[PopulationMember],
        ) -> list[dict[str, object]]:
            return [
                decision_member_result(member)
                for member in sorted(members, key=search._flash_member_rank_key)
            ]

        def successful_leaf_members(
            leaf: FlashStructuralLeaf,
        ) -> list[PopulationMember]:
            return sorted(
                (
                    member
                    for member in qualified_population.values()
                    if search._flash_structural_leaf(member) == leaf
                    and search._flash_member_succeeded(member)
                ),
                key=search._flash_member_rank_key,
            )

        def add_members(
            members: Sequence[PopulationMember],
            round_members: dict[int, PopulationMember],
        ) -> list[str]:
            ids: list[str] = []
            for member in members:
                qualified_population[id(member)] = member
                if member.perfs:
                    continue
                round_members[id(member)] = member
                config_id = canonical_config_id(member.config)
                candidate_ids.add(config_id)
                ids.append(config_id)
            return ids

        def novel_unbenchmarked_members(
            members: Sequence[PopulationMember],
        ) -> list[PopulationMember]:
            """Keep only configs that this phase has not already measured or queued."""
            known_configs = {member.config for member in qualified_population.values()}
            result: list[PopulationMember] = []
            for member in members:
                if member.perfs or member.config in known_configs:
                    continue
                known_configs.add(member.config)
                result.append(member)
            return result

        def reserve_pass(*, counts_toward_generation_budget: bool = True) -> bool:
            """Reserve a measurement pass before candidate generation mutates state."""
            nonlocal rounds_started, generation_rounds_started, budget_exhausted
            if budget_exhausted:
                return False
            if (
                counts_toward_generation_budget
                and generation_rounds_started >= search.max_generations - 1
            ):
                budget_exhausted = True
                return False
            if not any(True for _ in search._budgeted_range(1)):
                budget_exhausted = True
                return False
            rounds_started += 1
            generation_rounds_started += int(counts_toward_generation_budget)
            return True

        def run_pass(
            round_members: dict[int, PopulationMember],
            *,
            desc: str,
        ) -> None:
            nonlocal rounds_completed
            search.population = [*qualified_population.values()]
            unbenchmarked = [
                member for member in round_members.values() if not member.perfs
            ]
            if unbenchmarked:
                # Non-budgeted anchor work extends generation zero. Counted
                # qualification passes retain the historical generation IDs,
                # while measurement_timeline records every physical pass.
                search.set_generation(generation_rounds_started)
                search.benchmark_population(unbenchmarked, desc=desc)
                search.rebenchmark_population(
                    search.population,
                    desc=f"{desc} verifying",
                )
                for member in unbenchmarked:
                    search._append_training_sample(
                        search.config_gen.encode_config(member.flat_values),
                        member.perf,
                        member.config,
                        member.fn,
                        member=member,
                    )
                search._fit_surrogate()
                search.population.sort(key=performance)
            record_measurement_pass(rounds_completed + 1)
            rounds_completed += 1

        schedule_anchor_configs = (
            flash_generation.flash_low_confound_schedule_anchor_configs()
        )
        schedule_anchor_members: list[PopulationMember] = []
        pending_schedule_anchors: list[PopulationMember] = []
        for config in schedule_anchor_configs:
            existing = next(
                (
                    member
                    for member in qualified_population.values()
                    if member.config == config
                ),
                None,
            )
            if existing is None:
                existing = search.make_unbenchmarked(search.config_gen.flatten(config))
                if existing is None:
                    continue
                pending_schedule_anchors.append(existing)
            schedule_anchor_members.append(existing)
        schedule_anchor_pass_planned = bool(pending_schedule_anchors)
        schedule_anchor_pass_started = schedule_anchor_pass_planned and reserve_pass(
            counts_toward_generation_budget=False
        )
        if schedule_anchor_pass_started:
            schedule_anchor_round_members: dict[int, PopulationMember] = {}
            for member in pending_schedule_anchors:
                visited.add(member.config)
                anchor_ids = add_members([member], schedule_anchor_round_members)
                if anchor_ids:
                    leaf = search._flash_structural_leaf(member)
                    if leaf is not None:
                        leaves_with_candidates.add(leaf)
            run_pass(
                schedule_anchor_round_members,
                desc="Low-confound schedule anchors:",
            )
        elif pending_schedule_anchors:
            pending_ids = {id(member) for member in pending_schedule_anchors}
            schedule_anchor_members = [
                member
                for member in schedule_anchor_members
                if id(member) not in pending_ids
            ]
        schedule_anchor_results: list[dict[str, object]] = []
        for member in schedule_anchor_members:
            leaf = search._flash_structural_leaf(member)
            if leaf is None:
                continue
            schedule_anchor_results.append(
                {
                    "config_id": canonical_config_id(member.config),
                    "family": leaf.pipeline_family,
                    "compound_packet": leaf.compound_exp2_packet,
                    "softmax_disc": leaf.softmax_disc,
                    **measurement_result(member),
                }
            )
        schedule_anchor_complete = (
            (not pending_schedule_anchors or schedule_anchor_pass_started)
            and len(schedule_anchor_results) == len(schedule_anchor_configs)
            and all(
                flash_terminal_measurement_is_valid(result)
                for result in schedule_anchor_results
            )
        )

        lane_passes: dict[
            FlashStructuralLeaf,
            list[list[tuple[str, tuple[str, object] | None]]],
        ] = {}
        for leaf in ordinary_leaves:
            minimum_passes = (
                0
                if not leaf_pipeline_lanes[leaf]
                and bool(leaf_metrics[leaf]["space_exhausted"])
                else policy.qualification_rounds
            )
            lane_passes[leaf] = search._flash_lane_qualification_passes(
                leaf_pipeline_lanes[leaf],
                candidate_limit=policy.pipeline_candidates_per_leaf_per_round,
                conditional_candidates_per_lane=(
                    policy.conditional_candidates_per_pipeline_lane
                ),
                minimum_passes=minimum_passes,
                conditional_lanes=tuple(
                    lane
                    for lane in leaf_pipeline_lanes[leaf]
                    if lane_metrics[(leaf, lane)]["conditional_required"]
                ),
            )
        pipeline_pass_count = max(map(len, lane_passes.values()), default=0)
        for pass_index in range(pipeline_pass_count):
            if not reserve_pass():
                break
            round_members: dict[int, PopulationMember] = {}
            for leaf_index, leaf in enumerate(ordinary_leaves):
                jobs = (
                    lane_passes[leaf][pass_index]
                    if pass_index < len(lane_passes[leaf])
                    else []
                )
                leaf_round_ids: list[str] = []
                lane_round_ids = {lane: [] for lane in leaf_pipeline_lanes[leaf]}
                parent_decisions: list[dict[str, object]] = []
                conditional_jobs = [job for job in jobs if job[0] == "conditional"]
                conditional_limits = search._flash_lane_neighbor_limits(
                    [(job[1], 1) for job in conditional_jobs],
                    qualification_neighbor_limit,
                )
                conditional_limit_iter = iter(conditional_limits)
                neighbor_limit_by_lane: dict[tuple[str, object], int] = {}
                ordinary_neighbor_limit = 0
                for job_index, (kind, lane) in enumerate(jobs):
                    if kind == "witness":
                        assert lane is not None
                        lane_metric = lane_metrics[(leaf, lane)]
                        successful_lane_members = [
                            member
                            for member in qualified_population.values()
                            if search._flash_structural_leaf(member) == leaf
                            and search._flash_member_matches_pipeline_lane(member, lane)
                            and search._flash_member_succeeded(member)
                        ]
                        if successful_lane_members:
                            witness_candidates = sorted(
                                successful_lane_members,
                                key=search._flash_member_rank_key,
                            )
                            witness = witness_candidates[0]
                            selection_kind = "ranked_existing"
                        else:
                            witness = search._flash_pipeline_lane_witness(leaf, lane)
                            if witness is not None:
                                witness = next(
                                    (
                                        member
                                        for member in qualified_population.values()
                                        if member.config == witness.config
                                    ),
                                    witness,
                                )
                            witness_candidates = [] if witness is None else [witness]
                            selection_kind = "catalog_witness"
                        witness_decision: dict[str, object] = {
                            "job_index": job_index,
                            "kind": "witness",
                            "pipeline_lane": search._flash_pipeline_lane_metric(lane),
                            "selection_kind": selection_kind,
                            "candidate_results": [
                                decision_member_result(member)
                                for member in witness_candidates
                            ],
                            "selected_config_id": (
                                None
                                if witness is None
                                else canonical_config_id(witness.config)
                            ),
                            "generated_config_ids": [],
                        }
                        parent_decisions.append(witness_decision)
                        if witness is None:
                            continue
                        existing = next(
                            (
                                member
                                for member in qualified_population.values()
                                if member.config == witness.config
                            ),
                            None,
                        )
                        if existing is None:
                            visited.add(witness.config)
                            existing = witness
                        witness_id = canonical_config_id(existing.config)
                        lane_metric["witness_attempted"] = True
                        lane_metric["witness_config_id"] = witness_id
                        lane_round_ids[lane].append(witness_id)
                        ids = add_members([existing], round_members)
                        witness_decision["generated_config_ids"] = ids
                        leaf_round_ids.extend(ids)
                        if ids:
                            leaves_with_candidates.add(leaf)
                        continue

                    if kind == "conditional":
                        assert lane is not None
                        neighbor_limit = next(conditional_limit_iter)
                        neighbor_limit_by_lane[lane] = (
                            neighbor_limit_by_lane.get(lane, 0) + neighbor_limit
                        )
                        members = [
                            member
                            for member in qualified_population.values()
                            if search._flash_structural_leaf(member) == leaf
                            and member.perfs
                            if search._flash_member_matches_pipeline_lane(member, lane)
                        ]
                        parent_candidate_results = ranked_decision_results(members)
                        parent, added, parent_search = (
                            search._flash_conditional_parent_candidates(
                                members,
                                leaf,
                                (*search._flash_leaf_constraints(leaf), lane),
                                {
                                    member.config
                                    for member in qualified_population.values()
                                },
                                neighbor_limit,
                            )
                        )
                        novel = novel_unbenchmarked_members(added)
                        ids = add_members(novel[:1], round_members)
                        if not novel:
                            # Nothing left to measure within reach of any
                            # measured parent of the lane (see the lane
                            # metric): the lane counts as exhausted rather
                            # than leaving the qualification incomplete.
                            lane_metrics[(leaf, lane)]["conditional_exhausted"] = True
                        visited.update(member.config for member in added)
                        parent_decisions.append(
                            {
                                "job_index": job_index,
                                "kind": kind,
                                "pipeline_lane": search._flash_pipeline_lane_metric(
                                    lane
                                ),
                                "selection_kind": "ranked_parent",
                                "candidate_results": parent_candidate_results,
                                "selected_config_id": (
                                    None
                                    if parent is None
                                    else canonical_config_id(parent.config)
                                ),
                                "generated_config_ids": ids,
                                "conditional_parent_search": parent_search,
                            }
                        )
                        leaf_round_ids.extend(ids)
                        lane_round_ids[lane].extend(ids)
                        conditional_ids = lane_metrics[(leaf, lane)][
                            "conditional_candidate_ids"
                        ]
                        assert isinstance(conditional_ids, list)
                        conditional_ids.extend(ids)
                        if ids:
                            leaves_with_candidates.add(leaf)
                        continue

                    assert kind == "ordinary"
                    members = successful_leaf_members(leaf)
                    if not members:
                        parent_decisions.append(
                            {
                                "job_index": job_index,
                                "kind": kind,
                                "pipeline_lane": None,
                                "selection_kind": "ranked_parent",
                                "candidate_results": [],
                                "selected_config_id": None,
                                "generated_config_ids": [],
                            }
                        )
                        continue
                    member = members[0]
                    quota = policy.pipeline_candidates_per_leaf_per_round
                    neighbor_limit = qualification_neighbor_limit
                    ordinary_neighbor_limit += neighbor_limit
                    constraints = search._flash_leaf_constraints(leaf)

                    parent_candidate_results = ranked_decision_results(members)
                    parent_config_id = canonical_config_id(member.config)
                    search_copy = search._pruned_pattern_search_from(
                        pass_index
                        * max(1, len(ordinary_leaves))
                        * policy.pipeline_candidates_per_leaf_per_round
                        + leaf_index * policy.pipeline_candidates_per_leaf_per_round
                        + job_index,
                        member,
                        visited,
                        constraints,
                        selected_limit=quota + 1,
                        neighbor_limit=neighbor_limit,
                        required_leaf=leaf,
                        conditional_surface=True,
                        disable_early_stopping=True,
                    )
                    added = next(search_copy, ())
                    ids = add_members(
                        novel_unbenchmarked_members(added)[:quota],
                        round_members,
                    )
                    parent_decisions.append(
                        {
                            "job_index": job_index,
                            "kind": kind,
                            "pipeline_lane": search._flash_pipeline_lane_metric(lane),
                            "selection_kind": "ranked_parent",
                            "candidate_results": parent_candidate_results,
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                        }
                    )
                    leaf_round_ids.extend(ids)
                    if ids:
                        leaves_with_candidates.add(leaf)
                    if lane is not None:
                        lane_round_ids[lane].extend(ids)
                        cast_ids = lane_metrics[(leaf, lane)][
                            "conditional_candidate_ids"
                        ]
                        assert isinstance(cast_ids, list)
                        cast_ids.extend(ids)

                cast_rounds = leaf_metrics[leaf]["rounds"]
                assert isinstance(cast_rounds, list)
                cast_rounds.append(
                    {
                        "candidate_config_ids": leaf_round_ids,
                        "neighbor_generation_limit": ordinary_neighbor_limit
                        + sum(neighbor_limit_by_lane.values()),
                        "ordinary_neighbor_generation_limit": (ordinary_neighbor_limit),
                        "parent_decisions": parent_decisions,
                    }
                )
                for lane in leaf_pipeline_lanes[leaf]:
                    cast_rounds = lane_metrics[(leaf, lane)]["rounds"]
                    assert isinstance(cast_rounds, list)
                    cast_rounds.append(
                        {
                            "candidate_config_ids": lane_round_ids[lane],
                            "neighbor_generation_limit": neighbor_limit_by_lane.get(
                                lane, 0
                            ),
                        }
                    )
            run_pass(
                round_members,
                desc=f"Structural qualification {pass_index + 1}:",
            )

        def pipeline_lane_attempts(
            leaf: FlashStructuralLeaf,
            lane: tuple[str, object],
        ) -> list[PopulationMember]:
            """Return the measured attempts explicitly tracked for one lane."""
            metric = lane_metrics[(leaf, lane)]
            config_ids = [metric["witness_config_id"]]
            for key in ("conditional_candidate_ids", "repair_candidate_ids"):
                values = metric[key]
                assert isinstance(values, list)
                config_ids.extend(values)
            wanted = {
                config_id for config_id in config_ids if isinstance(config_id, str)
            }
            by_config_id = {
                canonical_config_id(member.config): member
                for member in qualified_population.values()
                if canonical_config_id(member.config) in wanted
                and search._flash_structural_leaf(member) == leaf
                and search._flash_member_matches_pipeline_lane(member, lane)
                and member.perfs
            }
            return [
                by_config_id[config_id]
                for config_id in sorted(wanted & by_config_id.keys())
            ]

        pipeline_repair_pass_count = 0
        repair_pass_ordinal = 0
        for repair_index in range(policy.qualification_failure_retries):
            repair_job_lanes: dict[FlashStructuralLeaf, list[tuple[str, object]]] = {}
            for leaf in ordinary_leaves:
                leaf_lanes = []
                for lane in leaf_pipeline_lanes[leaf]:
                    attempts = pipeline_lane_attempts(leaf, lane)
                    if not attempts or not all(
                        search._flash_member_has_retryable_failure(member)
                        for member in attempts
                    ):
                        continue
                    leaf_lanes.append(lane)
                repair_job_lanes[leaf] = leaf_lanes
            if not any(repair_job_lanes.values()):
                break
            batch_index = 0
            while any(repair_job_lanes.values()):
                current_jobs: dict[
                    FlashStructuralLeaf,
                    list[tuple[tuple[str, object], list[PopulationMember]]],
                ] = {}
                for leaf in ordinary_leaves:
                    leaf_jobs = []
                    pending_lanes = []
                    for lane in repair_job_lanes[leaf]:
                        attempts = pipeline_lane_attempts(leaf, lane)
                        if attempts and all(
                            search._flash_member_has_retryable_failure(member)
                            for member in attempts
                        ):
                            if (
                                len(leaf_jobs)
                                < policy.pipeline_candidates_per_leaf_per_round
                            ):
                                leaf_jobs.append(
                                    (
                                        lane,
                                        sorted(
                                            attempts,
                                            key=search._flash_member_rank_key,
                                        ),
                                    )
                                )
                            else:
                                pending_lanes.append(lane)
                    repair_job_lanes[leaf] = pending_lanes
                    current_jobs[leaf] = leaf_jobs
                if not any(current_jobs.values()):
                    break
                if not reserve_pass():
                    break
                pipeline_repair_pass_count += 1
                repair_pass_ordinal += 1
                round_members = {}
                for leaf_index, leaf in enumerate(ordinary_leaves):
                    jobs = current_jobs[leaf]
                    neighbor_limits = search._flash_lane_neighbor_limits(
                        [(lane, 1) for lane, _members in jobs],
                        qualification_neighbor_limit,
                    )
                    leaf_round_ids: list[str] = []
                    parent_decisions: list[dict[str, object]] = []
                    lane_round_ids = {lane: [] for lane in leaf_pipeline_lanes[leaf]}
                    lane_neighbor_limits = {
                        lane: limit
                        for (lane, _members), limit in zip(
                            jobs, neighbor_limits, strict=True
                        )
                    }
                    for job_index, ((lane, members), neighbor_limit) in enumerate(
                        zip(jobs, neighbor_limits, strict=True)
                    ):
                        parent = members[0]
                        parent_config_id = canonical_config_id(parent.config)
                        search_copy = search._pruned_pattern_search_from(
                            200_000
                            + repair_index * 10_000
                            + batch_index * max(1, len(ordinary_leaves)) * 100
                            + leaf_index * 100
                            + job_index,
                            parent,
                            visited,
                            (*search._flash_leaf_constraints(leaf), lane),
                            selected_limit=2,
                            neighbor_limit=neighbor_limit,
                            required_leaf=leaf,
                            conditional_surface=True,
                            disable_early_stopping=True,
                        )
                        added = next(search_copy, ())
                        ids = add_members(
                            novel_unbenchmarked_members(added)[:1],
                            round_members,
                        )
                        metric = lane_metrics[(leaf, lane)]
                        cast_ids = metric["repair_candidate_ids"]
                        cast_decisions = metric["repair_parent_decisions"]
                        assert isinstance(cast_ids, list)
                        assert isinstance(cast_decisions, list)
                        cast_ids.extend(ids)
                        lane_decision = {
                            "repair_index": repair_index,
                            "candidate_results": ranked_decision_results(members),
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                        }
                        cast_decisions.append(lane_decision)
                        parent_decisions.append(
                            {
                                "job_index": job_index,
                                "kind": "failure_repair",
                                "pipeline_lane": search._flash_pipeline_lane_metric(
                                    lane
                                ),
                                "selection_kind": "ranked_failed_parent",
                                **lane_decision,
                            }
                        )
                        leaf_round_ids.extend(ids)
                        lane_round_ids[lane].extend(ids)
                        if ids:
                            leaves_with_candidates.add(leaf)
                    cast_rounds = leaf_metrics[leaf]["rounds"]
                    assert isinstance(cast_rounds, list)
                    cast_rounds.append(
                        {
                            "candidate_config_ids": leaf_round_ids,
                            "neighbor_generation_limit": sum(neighbor_limits),
                            "ordinary_neighbor_generation_limit": 0,
                            "parent_decisions": parent_decisions,
                        }
                    )
                    for lane in leaf_pipeline_lanes[leaf]:
                        cast_lane_rounds = lane_metrics[(leaf, lane)]["rounds"]
                        assert isinstance(cast_lane_rounds, list)
                        cast_lane_rounds.append(
                            {
                                "candidate_config_ids": lane_round_ids[lane],
                                "neighbor_generation_limit": (
                                    lane_neighbor_limits.get(lane, 0)
                                ),
                            }
                        )
                run_pass(
                    round_members,
                    desc=(
                        f"Structural qualification failure repairs "
                        f"{repair_pass_ordinal}:"
                    ),
                )
                batch_index += 1
            if budget_exhausted:
                break

        clc_value_space_exhausted = {
            leaf: {
                value: measured_space_exhausted(
                    None
                    if exact_space is None
                    else [
                        config
                        for config in exact_space
                        if search._flash_structural_leaf_from_config(config) == leaf
                        and config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value
                    ]
                )
                for value in catalog["attempted_values"]
            }
            for leaf, catalog in clc_catalog.items()
            if leaf in ordinary_leaves
        }
        clc_metrics: dict[FlashStructuralLeaf, dict[str, object]] = {
            leaf: {
                "family": leaf.pipeline_family,
                "softmax_disc": leaf.softmax_disc,
                "space_exhausted": bool(leaf_metrics[leaf]["space_exhausted"]),
                "legal_values": list(catalog["legal_values"]),
                "search_values": list(catalog["search_values"]),
                "anchor_values": list(catalog["anchor_values"]),
                "refinement_values": list(catalog["refinement_values"]),
                "planned_values": list(catalog["attempted_values"]),
                "attempted_values": [],
                "witness_config_ids": {},
                "witness_repair_candidate_ids": {},
                "witness_repair_parent_decisions": [],
                "value_space_exhausted": {
                    str(value): exhausted
                    for value, exhausted in clc_value_space_exhausted[leaf].items()
                },
                "witness_candidate_results": [],
                "witness_selection_results": [],
                "selected_values": [],
                "selected_config_ids": [],
                "conditional_values": [],
                "conditional_neighbor_generation_limit": 0,
                "conditional_parent_decisions": [],
                "conditional_repair_candidate_ids": {},
                "conditional_repair_parent_decisions": [],
                "retained_values": [],
                "retained_config_ids": [],
                "retained_value_decisions": [],
                "retained_ranking_results": [],
                "conditional_candidate_ids": {},
                "combination_required": not bool(leaf_metrics[leaf]["space_exhausted"]),
                "depth_selection": {
                    "candidate_results": [],
                    "selected_representatives": [],
                },
                "combination_candidate_ids": [],
                "combination_depth_config_ids": [],
                "combination_divisor_values": [],
                "combination_cells": [],
                "combination_projection_complete": True,
                "successful_combination_depth_config_ids": [],
                "successful_combination_divisor_values": [],
                "combination_row_coverage_complete": True,
                "combination_column_coverage_complete": True,
                "combination_failure_statuses_allowed": True,
            }
            for leaf, catalog in clc_catalog.items()
            if leaf in ordinary_leaves
        }
        # All legal CLC decompositions share one common-context generation. A
        # per-generation slice would make coverage depend on the bounded
        # generation budget for highly composite B*H grids.
        clc_witness_pass_count = int(
            any(catalog["attempted_values"] for catalog in clc_catalog.values())
        )
        for clc_pass_index in range(clc_witness_pass_count):
            if not reserve_pass():
                break
            round_members = {}
            for leaf, catalog in clc_catalog.items():
                values = catalog["attempted_values"]
                for value in values:
                    witness = search._flash_clc_lane_witness(leaf, value)
                    if witness is None:
                        continue
                    existing = next(
                        (
                            member
                            for member in qualified_population.values()
                            if member.config == witness.config
                        ),
                        None,
                    )
                    if existing is None:
                        visited.add(witness.config)
                        existing = witness
                    cast_attempted = clc_metrics[leaf]["attempted_values"]
                    assert isinstance(cast_attempted, list)
                    cast_attempted.append(value)
                    cast_witnesses = clc_metrics[leaf]["witness_config_ids"]
                    assert isinstance(cast_witnesses, dict)
                    cast_witnesses[str(value)] = canonical_config_id(existing.config)
                    ids = add_members([existing], round_members)
                    if ids:
                        leaves_with_candidates.add(leaf)
            run_pass(
                round_members,
                desc=f"CLC divisor witnesses {clc_pass_index + 1}:",
            )

        def clc_member(
            leaf: FlashStructuralLeaf,
            value: int,
            config_id: object,
        ) -> PopulationMember | None:
            if not isinstance(config_id, str):
                return None
            members = [
                member
                for member in qualified_population.values()
                if canonical_config_id(member.config) == config_id
                and search._flash_structural_leaf(member) == leaf
                and member.config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value
            ]
            return min(members, key=search._flash_member_rank_key, default=None)

        def clc_attempt_members(
            leaf: FlashStructuralLeaf,
            value: int,
            *,
            primary_key: str,
            repair_key: str,
        ) -> list[PopulationMember]:
            """Return all measured primary and repair attempts for one divisor."""
            metric = clc_metrics[leaf]
            primary = metric[primary_key]
            assert isinstance(primary, dict)
            primary_value = primary.get(str(value))
            config_ids = (
                [primary_value]
                if isinstance(primary_value, str)
                else list(primary_value or ())
            )
            repairs = metric[repair_key]
            assert isinstance(repairs, dict)
            config_ids.extend(repairs.get(str(value), ()))
            members: dict[str, PopulationMember] = {}
            for config_id in config_ids:
                member = clc_member(leaf, value, config_id)
                if member is not None and member.perfs:
                    members[canonical_config_id(member.config)] = member
            return sorted(members.values(), key=search._flash_member_rank_key)

        def run_clc_failure_repairs(
            values_by_leaf: Mapping[FlashStructuralLeaf, Sequence[int]],
            *,
            primary_key: str,
            repair_key: str,
            decision_key: str,
            kind: str,
            desc: str,
            missing_attempt_parent_keys: tuple[str, str] | None = None,
        ) -> int:
            """Try one bounded alternate for each failed mandatory CLC obligation."""

            def repair_parents(
                leaf: FlashStructuralLeaf,
                value: int,
            ) -> list[PopulationMember]:
                attempts = clc_attempt_members(
                    leaf,
                    value,
                    primary_key=primary_key,
                    repair_key=repair_key,
                )
                if attempts:
                    if all(
                        search._flash_member_has_retryable_failure(member)
                        for member in attempts
                    ):
                        return attempts
                    return []
                if missing_attempt_parent_keys is None:
                    return []
                fallback_primary_key, fallback_repair_key = missing_attempt_parent_keys
                return [
                    member
                    for member in clc_attempt_members(
                        leaf,
                        value,
                        primary_key=fallback_primary_key,
                        repair_key=fallback_repair_key,
                    )
                    if search._flash_member_succeeded(member)
                ][:1]

            repair_pass_count = 0
            repair_pass_ordinal = 0
            for repair_index in range(policy.qualification_failure_retries):
                repair_job_values: dict[FlashStructuralLeaf, list[int]] = {}
                for leaf in clc_metrics:
                    leaf_values = []
                    for value in values_by_leaf.get(leaf, ()):
                        if not repair_parents(leaf, value):
                            continue
                        leaf_values.append(value)
                    repair_job_values[leaf] = leaf_values
                if not any(repair_job_values.values()):
                    break
                batch_index = 0
                while any(repair_job_values.values()):
                    current_jobs: dict[
                        FlashStructuralLeaf, list[tuple[int, list[PopulationMember]]]
                    ] = {}
                    for leaf in clc_metrics:
                        leaf_jobs = []
                        pending_values = []
                        for value in repair_job_values[leaf]:
                            parents = repair_parents(leaf, value)
                            if parents:
                                if (
                                    len(leaf_jobs)
                                    < policy.pipeline_candidates_per_leaf_per_round
                                ):
                                    leaf_jobs.append(
                                        (
                                            value,
                                            sorted(
                                                parents,
                                                key=search._flash_member_rank_key,
                                            ),
                                        )
                                    )
                                else:
                                    pending_values.append(value)
                        repair_job_values[leaf] = pending_values
                        current_jobs[leaf] = leaf_jobs
                    if not any(current_jobs.values()):
                        break
                    if not reserve_pass():
                        break
                    repair_pass_count += 1
                    repair_pass_ordinal += 1
                    round_members: dict[int, PopulationMember] = {}
                    for leaf_index, leaf in enumerate(clc_metrics):
                        jobs = current_jobs[leaf]
                        neighbor_limits = search._flash_lane_neighbor_limits(
                            [
                                ((FLASH_CLC_HEADS_PER_BATCH_KEY, value), 1)
                                for value, _members in jobs
                            ],
                            qualification_neighbor_limit,
                        )
                        for job_index, ((value, members), neighbor_limit) in enumerate(
                            zip(jobs, neighbor_limits, strict=True)
                        ):
                            parent = members[0]
                            parent_config_id = canonical_config_id(parent.config)
                            lane = (FLASH_CLC_HEADS_PER_BATCH_KEY, value)
                            search_copy = search._pruned_pattern_search_from(
                                300_000
                                + repair_index * 10_000
                                + batch_index * max(1, len(clc_metrics)) * 100
                                + leaf_index * 100
                                + job_index,
                                parent,
                                visited,
                                (*search._flash_leaf_constraints(leaf), lane),
                                selected_limit=2,
                                neighbor_limit=neighbor_limit,
                                required_leaf=leaf,
                                conditional_surface=True,
                                disable_early_stopping=True,
                            )
                            added = next(search_copy, ())
                            ids = add_members(
                                novel_unbenchmarked_members(added)[:1],
                                round_members,
                            )
                            metric = clc_metrics[leaf]
                            repair_ids = metric[repair_key]
                            repair_decisions = metric[decision_key]
                            assert isinstance(repair_ids, dict)
                            assert isinstance(repair_decisions, list)
                            repair_ids.setdefault(str(value), []).extend(ids)
                            repair_decisions.append(
                                {
                                    "kind": kind,
                                    "value": value,
                                    "repair_index": repair_index,
                                    "candidate_results": ranked_decision_results(
                                        members
                                    ),
                                    "selected_config_id": parent_config_id,
                                    "generated_config_ids": ids,
                                    "neighbor_generation_limit": neighbor_limit,
                                }
                            )
                            if ids:
                                leaves_with_candidates.add(leaf)
                    run_pass(
                        round_members,
                        desc=f"{desc} {repair_pass_ordinal}:",
                    )
                    batch_index += 1
                if budget_exhausted:
                    break
            return repair_pass_count

        clc_witness_repair_pass_count = run_clc_failure_repairs(
            {
                leaf: tuple(catalog["attempted_values"])
                for leaf, catalog in clc_catalog.items()
                if leaf in clc_metrics
            },
            primary_key="witness_config_ids",
            repair_key="witness_repair_candidate_ids",
            decision_key="witness_repair_parent_decisions",
            kind="witness_failure_repair",
            desc="CLC divisor witness failure repairs",
        )

        def clc_witness_member(
            leaf: FlashStructuralLeaf, value: int
        ) -> PopulationMember | None:
            attempts = clc_attempt_members(
                leaf,
                value,
                primary_key="witness_config_ids",
                repair_key="witness_repair_candidate_ids",
            )
            return next(
                (
                    member
                    for member in attempts
                    if search._flash_member_succeeded(member)
                ),
                None,
            )

        def ranked_clc_witness_values(
            leaf: FlashStructuralLeaf, values: Sequence[int]
        ) -> list[tuple[int, PopulationMember]]:
            """Rank one dedicated common-context witness per divisor."""
            ranked: list[tuple[int, PopulationMember]] = []
            for value in values:
                member = clc_witness_member(leaf, value)
                if member is not None:
                    ranked.append((value, member))
            return sorted(
                ranked,
                key=lambda item: (
                    search._flash_member_rank_key(item[1]),
                    item[0],
                ),
            )

        def tracked_clc_candidates(
            leaf: FlashStructuralLeaf, value: int
        ) -> list[PopulationMember]:
            """Rank every tracked candidate for one selected divisor."""
            candidates = {
                canonical_config_id(member.config): member
                for member in clc_attempt_members(
                    leaf,
                    value,
                    primary_key="witness_config_ids",
                    repair_key="witness_repair_candidate_ids",
                )
            }
            for member in clc_attempt_members(
                leaf,
                value,
                primary_key="conditional_candidate_ids",
                repair_key="conditional_repair_candidate_ids",
            ):
                candidates[canonical_config_id(member.config)] = member
            return sorted(candidates.values(), key=search._flash_member_rank_key)

        def clc_scoped_candidates(
            leaf: FlashStructuralLeaf, value: int
        ) -> list[PopulationMember]:
            return [
                member
                for member in tracked_clc_candidates(leaf, value)
                if search._flash_member_succeeded(member)
            ]

        clc_selected_values: dict[FlashStructuralLeaf, tuple[int, ...]] = {}
        clc_conditional_values: dict[FlashStructuralLeaf, tuple[int, ...]] = {}
        for leaf, catalog in clc_catalog.items():
            ranked = ranked_clc_witness_values(leaf, catalog["attempted_values"])
            selected = ranked
            selected_values = tuple(value for value, _member in selected)
            conditional_values = tuple(
                value
                for value in selected_values
                if not clc_value_space_exhausted[leaf][value]
            )
            clc_selected_values[leaf] = selected_values
            clc_conditional_values[leaf] = conditional_values
            witness_candidate_results: list[dict[str, object]] = []
            for value in catalog["attempted_values"]:
                members = clc_attempt_members(
                    leaf,
                    value,
                    primary_key="witness_config_ids",
                    repair_key="witness_repair_candidate_ids",
                )
                for member in members:
                    witness_candidate_results.append(
                        {"value": value, **decision_member_result(member)}
                    )
            clc_metrics[leaf]["witness_candidate_results"] = witness_candidate_results
            clc_metrics[leaf]["witness_selection_results"] = [
                {"value": value, **decision_member_result(member)}
                for value, member in ranked
            ]
            clc_metrics[leaf]["selected_values"] = list(selected_values)
            clc_metrics[leaf]["selected_config_ids"] = [
                canonical_config_id(member.config) for _value, member in selected
            ]
            clc_metrics[leaf]["conditional_values"] = list(conditional_values)

        if any(clc_conditional_values.values()) and reserve_pass():
            round_members = {}
            for leaf_index, (leaf, values) in enumerate(clc_conditional_values.items()):
                conditional_neighbor_limit = max(
                    qualification_neighbor_limit,
                    len(values),
                )
                clc_metrics[leaf]["conditional_neighbor_generation_limit"] = (
                    conditional_neighbor_limit
                )
                neighbor_limits = search._flash_lane_neighbor_limits(
                    [((FLASH_CLC_HEADS_PER_BATCH_KEY, value), 1) for value in values],
                    conditional_neighbor_limit,
                )
                for value_index, (value, neighbor_limit) in enumerate(
                    zip(values, neighbor_limits, strict=True)
                ):
                    ranked = ranked_clc_witness_values(leaf, (value,))
                    if not ranked:
                        witness_config_ids = clc_metrics[leaf]["witness_config_ids"]
                        assert isinstance(witness_config_ids, dict)
                        witness = clc_member(
                            leaf,
                            value,
                            witness_config_ids.get(str(value)),
                        )
                        cast_decisions = clc_metrics[leaf][
                            "conditional_parent_decisions"
                        ]
                        assert isinstance(cast_decisions, list)
                        cast_decisions.append(
                            {
                                "value": value,
                                "candidate_results": (
                                    []
                                    if witness is None
                                    else [decision_member_result(witness)]
                                ),
                                "selected_config_id": None,
                                "generated_config_ids": [],
                                "neighbor_generation_limit": neighbor_limit,
                            }
                        )
                        continue
                    member = ranked[0][1]
                    parent_result = decision_member_result(member)
                    parent_config_id = canonical_config_id(member.config)
                    lane = (FLASH_CLC_HEADS_PER_BATCH_KEY, value)
                    search_copy = search._pruned_pattern_search_from(
                        100_000 + leaf_index * max(1, len(values)) + value_index,
                        member,
                        visited,
                        (*search._flash_leaf_constraints(leaf), lane),
                        selected_limit=2,
                        neighbor_limit=neighbor_limit,
                        required_leaf=leaf,
                        conditional_surface=True,
                        disable_early_stopping=True,
                    )
                    added = next(search_copy, ())
                    ids = add_members(
                        novel_unbenchmarked_members(added)[:1],
                        round_members,
                    )
                    cast_ids = clc_metrics[leaf]["conditional_candidate_ids"]
                    assert isinstance(cast_ids, dict)
                    cast_ids[str(value)] = ids
                    cast_decisions = clc_metrics[leaf]["conditional_parent_decisions"]
                    assert isinstance(cast_decisions, list)
                    cast_decisions.append(
                        {
                            "value": value,
                            "candidate_results": [parent_result],
                            "selected_config_id": parent_config_id,
                            "generated_config_ids": ids,
                            "neighbor_generation_limit": neighbor_limit,
                        }
                    )
            run_pass(round_members, desc="CLC divisor conditional children:")

        clc_conditional_repair_pass_count = run_clc_failure_repairs(
            clc_conditional_values,
            primary_key="conditional_candidate_ids",
            repair_key="conditional_repair_candidate_ids",
            decision_key="conditional_repair_parent_decisions",
            kind="conditional_failure_repair",
            desc="CLC divisor conditional failure repairs",
            missing_attempt_parent_keys=(
                "witness_config_ids",
                "witness_repair_candidate_ids",
            ),
        )

        retained_clc_members: dict[
            FlashStructuralLeaf, list[tuple[int, PopulationMember]]
        ] = {}
        for leaf in clc_catalog:
            value_decisions: list[dict[str, object]] = []
            representatives: list[tuple[int, PopulationMember]] = []
            for value in clc_selected_values[leaf]:
                tracked_candidates = tracked_clc_candidates(leaf, value)
                candidates = clc_scoped_candidates(leaf, value)
                selected_member = candidates[0] if candidates else None
                value_decisions.append(
                    {
                        "value": value,
                        "candidate_results": [
                            decision_member_result(member)
                            for member in tracked_candidates
                        ],
                        "selected_config_id": (
                            None
                            if selected_member is None
                            else canonical_config_id(selected_member.config)
                        ),
                    }
                )
                if selected_member is not None:
                    representatives.append((value, selected_member))
            ranked = sorted(
                representatives,
                key=lambda item: (
                    search._flash_member_rank_key(item[1]),
                    item[0],
                ),
            )
            retained_clc_members[leaf] = ranked
            clc_metrics[leaf]["retained_value_decisions"] = value_decisions
            clc_metrics[leaf]["retained_ranking_results"] = [
                {"value": value, **decision_member_result(member)}
                for value, member in sorted(
                    representatives,
                    key=lambda item: (
                        search._flash_member_rank_key(item[1]),
                        item[0],
                    ),
                )
            ]
            clc_metrics[leaf]["retained_values"] = [value for value, _ in ranked]
            clc_metrics[leaf]["retained_config_ids"] = [
                canonical_config_id(member.config) for _value, member in ranked
            ]

        combination_members: dict[FlashStructuralLeaf, list[PopulationMember]] = {}
        clc_combination_leaves = [
            leaf
            for leaf in clc_catalog
            if bool(clc_metrics[leaf]["combination_required"])
        ]
        if clc_combination_leaves and reserve_pass():
            combination_round_members: dict[int, PopulationMember] = {}
            for leaf in clc_combination_leaves:
                divisor_members = retained_clc_members[leaf]
                depth_candidates = successful_leaf_members(leaf)
                depth_representatives = search._flash_lane_diverse_members(
                    depth_candidates,
                    leaf_pipeline_lanes[leaf],
                    policy.retained_candidates_per_leaf,
                )
                clc_metrics[leaf]["depth_selection"] = {
                    "candidate_results": [
                        decision_member_result(member) for member in depth_candidates
                    ],
                    "selected_representatives": [
                        {
                            "config_id": canonical_config_id(member.config),
                            "assigned_pipeline_lane": (
                                search._flash_pipeline_lane_metric(lane)
                            ),
                        }
                        for member, lane in depth_representatives
                    ],
                }
                depth_members = [member for member, _lane in depth_representatives]
                depth_config_ids = [
                    canonical_config_id(member.config) for member in depth_members
                ]
                divisor_values = [value for value, _member in divisor_members]
                clc_metrics[leaf]["combination_depth_config_ids"] = depth_config_ids
                clc_metrics[leaf]["combination_divisor_values"] = divisor_values
                combined: list[PopulationMember] = []
                seen_configs: set[Config] = set()
                cells: list[dict[str, object]] = []
                for depth_member, depth_config_id in zip(
                    depth_members, depth_config_ids, strict=True
                ):
                    for value, _divisor_member in divisor_members:
                        candidate = search._flash_clc_depth_variant(
                            depth_member,
                            value,
                            expected_leaf=leaf,
                        )
                        if candidate is None:
                            cells.append(
                                {
                                    "depth_config_id": depth_config_id,
                                    "divisor_value": value,
                                    "projected_config_id": None,
                                    "config_id": None,
                                    "attempt_perf": None,
                                    "selection_perf": None,
                                    "status": "projection_rejected",
                                    "source_hash": None,
                                    "measurement_pass_index": None,
                                }
                            )
                            continue
                        existing = next(
                            (
                                member
                                for member in qualified_population.values()
                                if member.config == candidate.config
                            ),
                            None,
                        )
                        combined_member = candidate if existing is None else existing
                        projected_config_id = canonical_config_id(
                            combined_member.config
                        )
                        cells.append(
                            {
                                "depth_config_id": depth_config_id,
                                "divisor_value": value,
                                "projected_config_id": projected_config_id,
                            }
                        )
                        if candidate.config in seen_configs:
                            continue
                        seen_configs.add(candidate.config)
                        if existing is None:
                            visited.add(candidate.config)
                        combined.append(combined_member)
                        add_members([combined_member], combination_round_members)
                combination_members[leaf] = combined
                cast_ids = clc_metrics[leaf]["combination_candidate_ids"]
                assert isinstance(cast_ids, list)
                cast_ids.extend(
                    canonical_config_id(member.config) for member in combined
                )
                clc_metrics[leaf]["combination_cells"] = cells
            run_pass(
                combination_round_members,
                desc="CLC depth/divisor combinations:",
            )
            for leaf in clc_combination_leaves:
                metric_cells = clc_metrics[leaf]["combination_cells"]
                assert isinstance(metric_cells, list)
                members_by_id = {
                    canonical_config_id(member.config): member
                    for member in qualified_population.values()
                }
                for cell in metric_cells:
                    assert isinstance(cell, dict)
                    projected_config_id = cell["projected_config_id"]
                    if not isinstance(projected_config_id, str):
                        continue
                    member = members_by_id[projected_config_id]
                    cell.update(decision_member_result(member))

        compound_transfer_metrics: list[dict[str, object]] = []
        compound_round_members: dict[int, PopulationMember] = {}
        compound_transfer_members: dict[str, PopulationMember] = {}
        qualified_compound_config_ids: dict[FlashStructuralLeaf, set[str]] = {}
        compound_pass_reserved = bool(compound_leaves) and reserve_pass()
        compound_states: dict[FlashStructuralLeaf, dict[str, object]] = {}
        ordinary_protocol_leaves = {
            (leaf.pipeline_family, leaf.softmax_disc): leaf for leaf in ordinary_leaves
        }
        compound_catalog_errors: list[dict[str, object]] = []

        def add_compound_transfers(
            compound_leaf: FlashStructuralLeaf,
            *,
            count: int,
            round_members: dict[int, PopulationMember],
        ) -> list[str]:
            """Project the next ranked sources into distinct compound candidates."""
            state = compound_states[compound_leaf]
            source_pool = state["source_pool"]
            seen_transfers = state["seen_transfers"]
            transfers = state["transfers"]
            source_selection = state["source_selection"]
            assert isinstance(source_pool, list)
            assert isinstance(seen_transfers, set)
            assert isinstance(transfers, list)
            assert isinstance(source_selection, dict)
            attempted_source_ids = source_selection["attempted_config_ids"]
            selected_source_ids = source_selection["selected_config_ids"]
            assert isinstance(attempted_source_ids, list)
            assert isinstance(selected_source_ids, list)
            generated_ids: list[str] = []
            source_index = state["next_source_index"]
            assert isinstance(source_index, int)
            while count > 0 and source_index < len(source_pool):
                source_member = source_pool[source_index]
                source_index += 1
                state["next_source_index"] = source_index
                source_config_id = canonical_config_id(source_member.config)
                attempted_source_ids.append(source_config_id)
                candidate = search._flash_compound_variant(
                    source_member,
                    compound_leaf.compound_exp2_packet,
                    expected_leaf=compound_leaf,
                )
                if candidate is None or candidate.config in seen_transfers:
                    continue
                seen_transfers.add(candidate.config)
                existing = next(
                    (
                        member
                        for member in qualified_population.values()
                        if member.config == candidate.config
                    ),
                    None,
                )
                if (
                    existing is not None
                    and existing.perfs
                    and search._flash_member_has_retryable_failure(existing)
                ):
                    # A later source may project to a viable candidate. Preserve
                    # this source attempt in the immutable decision record, but
                    # do not spend one of the leaf's transfer slots on a known
                    # retryable failure.
                    continue
                if existing is None:
                    visited.add(candidate.config)
                transferred = candidate if existing is None else existing
                ids = add_members([transferred], round_members)
                transferred_config_id = canonical_config_id(transferred.config)
                generated_ids.append(transferred_config_id)
                compound_transfer_members[transferred_config_id] = transferred
                selected_source_ids.append(source_config_id)
                transfers.append(
                    {
                        "source_config_id": source_config_id,
                        "source_config": copy.deepcopy(source_member.config.config),
                        "transferred_config_id": transferred_config_id,
                        "projection_overrides": {
                            FLASH_EXP2_PACKET_KEY: compound_leaf.compound_exp2_packet
                        },
                        "projected_config_id": canonical_config_id(candidate.config),
                        "projected_config": copy.deepcopy(candidate.config.config),
                        "preserved_pipeline_values": {
                            key: source_member.config.config[key]
                            for key in search._flash_pipeline_qualification_keys()
                            if key in source_member.config.config
                        },
                    }
                )
                if ids:
                    leaves_with_candidates.add(compound_leaf)
                count -= 1
            return generated_ids

        for compound_leaf in compound_leaves:
            ordinary_leaf = ordinary_protocol_leaves.get(
                (compound_leaf.pipeline_family, compound_leaf.softmax_disc)
            )
            if ordinary_leaf is None:
                catalog_error: dict[str, object] = {
                    "family": compound_leaf.pipeline_family,
                    "compound_packet": compound_leaf.compound_exp2_packet,
                    "softmax_disc": compound_leaf.softmax_disc,
                    "error": "missing_ordinary_protocol_leaf",
                    "required_parent": {
                        "family": compound_leaf.pipeline_family,
                        "compound_packet": None,
                        "softmax_disc": compound_leaf.softmax_disc,
                    },
                }
                compound_catalog_errors.append(catalog_error)
                compound_transfer_metrics.append(
                    {
                        "family": compound_leaf.pipeline_family,
                        "compound_packet": compound_leaf.compound_exp2_packet,
                        "softmax_disc": compound_leaf.softmax_disc,
                        "limit": policy.retained_candidates_per_leaf,
                        "transfer_target_count": 0,
                        "transfer_count": 0,
                        "primary_transfer_config_ids": [],
                        "backfill_rounds": [],
                        "successful_transfer_config_ids": [],
                        "qualified_transfer_config_ids": [],
                        "failure_statuses_allowed": False,
                        "source_selection": {
                            "candidate_results": [],
                            "combination_prefix_count": 0,
                            "attempted_config_ids": [],
                            "selected_config_ids": [],
                        },
                        "transfers": [],
                        "catalog_error": catalog_error["error"],
                        "complete": False,
                    }
                )
                qualified_compound_config_ids[compound_leaf] = set()
                continue
            transfers: list[dict[str, object]] = []
            source_selection: dict[str, object] = {
                "candidate_results": [],
                "combination_prefix_count": 0,
                "attempted_config_ids": [],
                "selected_config_ids": [],
            }
            combined_source_pool = sorted(
                (
                    member
                    for member in combination_members.get(ordinary_leaf, [])
                    if search._flash_member_succeeded(member)
                ),
                key=search._flash_member_rank_key,
            )
            source_configs = {member.config for member in combined_source_pool}
            source_pool = [
                *combined_source_pool,
                *(
                    member
                    for member in successful_leaf_members(ordinary_leaf)
                    if member.config not in source_configs
                ),
            ]
            source_selection["candidate_results"] = [
                decision_member_result(member) for member in source_pool
            ]
            source_selection["combination_prefix_count"] = len(combined_source_pool)
            metric: dict[str, object] = {
                "family": compound_leaf.pipeline_family,
                "compound_packet": compound_leaf.compound_exp2_packet,
                "softmax_disc": compound_leaf.softmax_disc,
                "limit": policy.retained_candidates_per_leaf,
                "transfer_target_count": 0,
                "transfer_count": 0,
                "primary_transfer_config_ids": [],
                "backfill_rounds": [],
                "successful_transfer_config_ids": [],
                "qualified_transfer_config_ids": [],
                "failure_statuses_allowed": True,
                "source_selection": source_selection,
                "transfers": transfers,
                "complete": False,
            }
            compound_states[compound_leaf] = {
                "source_pool": source_pool,
                "next_source_index": 0,
                "seen_transfers": set(),
                "transfers": transfers,
                "source_selection": source_selection,
                "metric": metric,
            }
            compound_transfer_metrics.append(metric)
            if compound_pass_reserved:
                primary_ids = add_compound_transfers(
                    compound_leaf,
                    count=policy.retained_candidates_per_leaf,
                    round_members=compound_round_members,
                )
                metric["primary_transfer_config_ids"] = primary_ids
                metric["transfer_target_count"] = len(primary_ids)
        if compound_pass_reserved:
            run_pass(compound_round_members, desc="Compound packet transfers:")

        compound_backfill_pass_count = 0
        for repair_index in range(policy.qualification_failure_retries):
            repair_needs: dict[FlashStructuralLeaf, tuple[int, list[str]]] = {}
            for compound_leaf, state in compound_states.items():
                state_metric = state["metric"]
                assert isinstance(state_metric, dict)
                target_count = state_metric["transfer_target_count"]
                state_transfers = state_metric["transfers"]
                assert isinstance(target_count, int)
                assert isinstance(state_transfers, list)
                attempted_members = [
                    compound_transfer_members[transfer["transferred_config_id"]]
                    for transfer in state_transfers
                    if isinstance(transfer, dict)
                    and isinstance(transfer.get("transferred_config_id"), str)
                ]
                successful_count = sum(
                    search._flash_member_succeeded(member)
                    for member in attempted_members
                )
                failed_members = [
                    member
                    for member in attempted_members
                    if not search._flash_member_succeeded(member)
                ]
                if not failed_members or not all(
                    search._flash_member_has_retryable_failure(member)
                    for member in failed_members
                ):
                    continue
                missing = target_count - successful_count
                if missing > 0:
                    repair_needs[compound_leaf] = (
                        missing,
                        [
                            canonical_config_id(member.config)
                            for member in failed_members
                        ],
                    )
            if not repair_needs:
                break
            compound_backfill_pass_count += 1
            if not reserve_pass():
                break
            backfill_round_members: dict[int, PopulationMember] = {}
            for compound_leaf, (missing, failed_ids) in repair_needs.items():
                state = compound_states[compound_leaf]
                state_metric = state["metric"]
                state_source_selection = state["source_selection"]
                assert isinstance(state_metric, dict)
                assert isinstance(state_source_selection, dict)
                attempted_sources = state_source_selection["attempted_config_ids"]
                assert isinstance(attempted_sources, list)
                attempted_start = len(attempted_sources)
                generated_ids = add_compound_transfers(
                    compound_leaf,
                    count=missing,
                    round_members=backfill_round_members,
                )
                backfill_rounds = state_metric["backfill_rounds"]
                assert isinstance(backfill_rounds, list)
                backfill_rounds.append(
                    {
                        "repair_index": repair_index,
                        "required_successes": missing,
                        "failed_transfer_config_ids": failed_ids,
                        "attempted_source_config_ids": attempted_sources[
                            attempted_start:
                        ],
                        "generated_config_ids": generated_ids,
                    }
                )
            run_pass(
                backfill_round_members,
                desc=f"Compound packet failure backfills {repair_index + 1}:",
            )
            if budget_exhausted:
                break

        for metric in compound_transfer_metrics:
            metric_transfer_entries = metric["transfers"]
            assert isinstance(metric_transfer_entries, list)
            for transfer in metric_transfer_entries:
                assert isinstance(transfer, dict)
                transferred_config_id = transfer["transferred_config_id"]
                assert isinstance(transferred_config_id, str)
                transferred = compound_transfer_members[transferred_config_id]
                transfer.update(measurement_result(transferred))
            metric["transfer_count"] = len(metric_transfer_entries)
            successful_transfer_ids = [
                transfer["transferred_config_id"]
                for transfer in metric_transfer_entries
                if isinstance(transfer, dict)
                and isinstance(transfer.get("transferred_config_id"), str)
                and search._flash_member_succeeded(
                    compound_transfer_members[transfer["transferred_config_id"]]
                )
            ]
            metric["successful_transfer_config_ids"] = successful_transfer_ids
            target_count = metric["transfer_target_count"]
            assert isinstance(target_count, int)
            qualified_transfer_ids = successful_transfer_ids[:target_count]
            metric["qualified_transfer_config_ids"] = qualified_transfer_ids
            failure_statuses_allowed = metric.get("catalog_error") is None and all(
                isinstance(transfer, dict)
                and flash_terminal_measurement_is_valid(transfer)
                for transfer in metric_transfer_entries
            )
            metric["failure_statuses_allowed"] = failure_statuses_allowed
            compound_leaf = next(
                leaf
                for leaf in compound_leaves
                if leaf.pipeline_family == metric["family"]
                and leaf.compound_exp2_packet == metric["compound_packet"]
                and leaf.softmax_disc == metric["softmax_disc"]
            )
            qualified_compound_config_ids[compound_leaf] = set(qualified_transfer_ids)
        search._flash_qualified_compound_config_ids = qualified_compound_config_ids

        parent_score_config_ids = {
            canonical_config_id(member.config)
            for member in qualified_population.values()
            if (leaf := search._flash_structural_leaf(member)) is not None
            and leaf.compound_exp2_packet is None
        }
        live_family_count = len({leaf.pipeline_family for leaf in ordinary_leaves})
        family_probe_required = bool(
            policy.family_probe_generations > 0
            and policy.retained_families is not None
            and live_family_count > policy.retained_families
            and not exact_space_exhausted
        )
        family_probe_paths = (
            search._flash_family_probe_paths([*qualified_population.values()])
            if family_probe_required
            else []
        )
        family_probe_path_limit = getattr(
            search, "_flash_family_probe_path_limit", len(family_probe_paths)
        )
        # The limit is the catalog's capacity: a family (or compound leaf) whose
        # measured members all failed at this shape has no probe start.
        if family_probe_required and len(family_probe_paths) > family_probe_path_limit:
            raise AssertionError(
                "family probe paths exceed the live structural catalog capacity"
            )
        family_probe_metrics: list[dict[str, object]] = []
        family_probe_generators = []
        for copy_index, (member, required_leaf, constraints, unrestricted) in enumerate(
            family_probe_paths
        ):
            start_leaf = search._flash_structural_leaf(member)
            assert start_leaf is not None
            family_probe_metrics.append(
                {
                    "family": start_leaf.pipeline_family,
                    "compound_packet": start_leaf.compound_exp2_packet,
                    "softmax_disc": start_leaf.softmax_disc,
                    "starting_config_id": canonical_config_id(member.config),
                    "unrestricted": unrestricted,
                    "rounds": [],
                }
            )
            family_probe_generators.append(
                search._pruned_pattern_search_from(
                    copy_index,
                    member,
                    visited,
                    constraints,
                    selected_limit=policy.family_probe_candidates_per_path,
                    required_leaf=required_leaf,
                    conditional_surface=required_leaf is not None,
                    disable_early_stopping=True,
                )
            )

        family_probe_generations_started = 0
        family_probe_generations_completed = 0
        for probe_generation in range(policy.family_probe_generations):
            if not family_probe_required or not reserve_pass():
                break
            family_probe_generations_started += 1
            probe_round_members: dict[int, PopulationMember] = {}
            round_members_by_path: list[list[PopulationMember]] = []
            for path_metric, generator, path in zip(
                family_probe_metrics,
                family_probe_generators,
                family_probe_paths,
                strict=True,
            ):
                added = next(generator, ())
                new_members = list(added[1:]) if added else []
                generated_ids = add_members(new_members, probe_round_members)
                required_leaf = path[1]
                if (
                    required_leaf is not None
                    and required_leaf.compound_exp2_packet is None
                ):
                    parent_score_config_ids.update(generated_ids)
                for candidate in new_members:
                    candidate_leaf = search._flash_structural_leaf(candidate)
                    if candidate_leaf is not None:
                        leaves_with_candidates.add(candidate_leaf)
                cast_rounds = path_metric["rounds"]
                assert isinstance(cast_rounds, list)
                cast_rounds.append(
                    {
                        "probe_generation": probe_generation + 1,
                        "measurement_pass_index": rounds_completed + 1,
                        "candidate_ids": generated_ids,
                        "results": [],
                    }
                )
                round_members_by_path.append(new_members)
            run_pass(
                probe_round_members,
                desc=f"Structural family probe {probe_generation + 1}:",
            )
            for path_metric, members in zip(
                family_probe_metrics, round_members_by_path, strict=True
            ):
                cast_rounds = path_metric["rounds"]
                assert isinstance(cast_rounds, list)
                round_metric = cast_rounds[-1]
                assert isinstance(round_metric, dict)
                round_metric["results"] = [
                    {
                        "config_id": canonical_config_id(member.config),
                        **measurement_result(member),
                    }
                    for member in members
                ]
                for member in members:
                    leaf = search._flash_structural_leaf(member)
                    if (
                        leaf is not None
                        and leaf.compound_exp2_packet is not None
                        and search._flash_member_succeeded(member)
                    ):
                        qualified_compound_config_ids.setdefault(leaf, set()).add(
                            canonical_config_id(member.config)
                        )
            family_probe_generations_completed += 1
        family_probe_complete = bool(
            not family_probe_required
            or (
                family_probe_generations_started == policy.family_probe_generations
                and family_probe_generations_completed
                == policy.family_probe_generations
            )
        )
        search._flash_parent_score_config_ids = parent_score_config_ids

        successful_config_ids = {
            canonical_config_id(member.config)
            for member in qualified_population.values()
            if search._flash_member_succeeded(member)
        }
        for leaf in ordinary_leaves:
            members = sorted(
                (
                    member
                    for member in qualified_population.values()
                    if search._flash_structural_leaf(member) == leaf
                ),
                key=search._flash_member_rank_key,
            )
            successful = [
                member for member in members if search._flash_member_succeeded(member)
            ]
            retained = search._flash_lane_diverse_members(
                successful,
                leaf_pipeline_lanes[leaf],
                policy.retained_candidates_per_leaf,
            )
            leaf_metrics[leaf]["qualified_results"] = [
                {
                    "config_id": canonical_config_id(member.config),
                    **measurement_result(member),
                    "pipeline_lanes": [
                        search._flash_pipeline_lane_metric(lane)
                        for lane in leaf_pipeline_lanes[leaf]
                        if search._flash_member_matches_pipeline_lane(member, lane)
                    ],
                }
                for member in members
            ]
            leaf_metrics[leaf]["retained_config_ids"] = [
                canonical_config_id(member.config) for member, _lane in retained
            ]
            cast_lanes = leaf_metrics[leaf]["pipeline_lanes"]
            assert isinstance(cast_lanes, list)
            for _lane, lane_metric in zip(
                leaf_pipeline_lanes[leaf], cast_lanes, strict=True
            ):
                cast_ids = lane_metric["conditional_candidate_ids"]
                successful_conditional_ids = lane_metric[
                    "successful_conditional_candidate_ids"
                ]
                repair_ids = lane_metric["repair_candidate_ids"]
                successful_repair_ids = lane_metric["successful_repair_candidate_ids"]
                assert isinstance(cast_ids, list)
                assert isinstance(successful_conditional_ids, list)
                assert isinstance(repair_ids, list)
                assert isinstance(successful_repair_ids, list)
                repair_parent_decisions = lane_metric["repair_parent_decisions"]
                assert isinstance(repair_parent_decisions, list)
                witness_config_id = lane_metric["witness_config_id"]
                witness_succeeded = witness_config_id in successful_config_ids
                successful_conditional_ids.extend(
                    config_id
                    for config_id in cast_ids
                    if config_id in successful_config_ids
                )
                successful_repair_ids.extend(
                    config_id
                    for config_id in repair_ids
                    if config_id in successful_config_ids
                )
                lane_metric["witness_succeeded"] = witness_succeeded
                has_success = bool(
                    witness_succeeded
                    or successful_conditional_ids
                    or successful_repair_ids
                )
                attempts = pipeline_lane_attempts(leaf, _lane)
                # The lane's conditional quota is met by enough measured
                # candidates, or by the search having nothing unmeasured left
                # to reach (then every reachable config of the lane was tried).
                conditional_exhausted = bool(lane_metric["conditional_exhausted"])
                conditional_satisfied = bool(
                    not lane_metric["conditional_required"]
                    or len(set(cast_ids))
                    >= policy.conditional_candidates_per_pipeline_lane
                    or conditional_exhausted
                )
                terminal_failure_exhausted = bool(
                    not has_success
                    and lane_metric["witness_attempted"]
                    and conditional_satisfied
                    and len(repair_ids) == policy.qualification_failure_retries
                    and len(repair_parent_decisions)
                    == policy.qualification_failure_retries
                    and len(attempts) == 1 + len(cast_ids) + len(repair_ids)
                    and all(
                        search._flash_member_has_retryable_failure(member)
                        for member in attempts
                    )
                )
                lane_metric["terminal_failure_exhausted"] = terminal_failure_exhausted
                lane_metric["complete"] = bool(
                    lane_metric["witness_attempted"]
                    and conditional_satisfied
                    and (
                        has_success
                        or terminal_failure_exhausted
                        or conditional_exhausted
                    )
                )
            leaf_metrics[leaf]["complete"] = bool(successful) and all(
                bool(metric["complete"]) for metric in cast_lanes
            )
        lane_complete = all(
            bool(metric["complete"]) for metric in lane_metrics.values()
        )
        ordinary_complete = all(
            bool(leaf_metrics[leaf]["complete"]) for leaf in ordinary_leaves
        )
        for metric in clc_metrics.values():
            planned_values = metric["planned_values"]
            attempted_values = metric["attempted_values"]
            witness_config_ids = metric["witness_config_ids"]
            witness_repair_candidate_ids = metric["witness_repair_candidate_ids"]
            conditional_values = metric["conditional_values"]
            conditional_candidate_ids = metric["conditional_candidate_ids"]
            conditional_repair_candidate_ids = metric[
                "conditional_repair_candidate_ids"
            ]
            retained_values = metric["retained_values"]
            selected_values = metric["selected_values"]
            combination_candidate_ids = metric["combination_candidate_ids"]
            combination_depth_config_ids = metric["combination_depth_config_ids"]
            combination_divisor_values = metric["combination_divisor_values"]
            combination_cells = metric["combination_cells"]
            assert isinstance(planned_values, list)
            assert isinstance(attempted_values, list)
            assert isinstance(witness_config_ids, dict)
            assert isinstance(witness_repair_candidate_ids, dict)
            assert isinstance(conditional_values, list)
            assert isinstance(conditional_candidate_ids, dict)
            assert isinstance(conditional_repair_candidate_ids, dict)
            assert isinstance(retained_values, list)
            assert isinstance(selected_values, list)
            assert isinstance(combination_candidate_ids, list)
            assert isinstance(combination_depth_config_ids, list)
            assert isinstance(combination_divisor_values, list)
            assert isinstance(combination_cells, list)
            successful_depth_ids = [
                depth_config_id
                for depth_config_id in combination_depth_config_ids
                if any(
                    isinstance(cell, dict)
                    and cell.get("depth_config_id") == depth_config_id
                    and cell.get("config_id") in successful_config_ids
                    for cell in combination_cells
                )
            ]
            successful_divisor_values = [
                value
                for value in combination_divisor_values
                if any(
                    isinstance(cell, dict)
                    and cell.get("divisor_value") == value
                    and cell.get("config_id") in successful_config_ids
                    for cell in combination_cells
                )
            ]
            projection_complete = bool(
                len(combination_cells)
                == len(combination_depth_config_ids) * len(combination_divisor_values)
                and len(
                    {
                        (cell.get("depth_config_id"), cell.get("divisor_value"))
                        for cell in combination_cells
                        if isinstance(cell, dict)
                    }
                )
                == len(combination_cells)
            )
            row_coverage_complete = successful_depth_ids == combination_depth_config_ids
            column_coverage_complete = (
                successful_divisor_values == combination_divisor_values
            )
            failure_statuses_allowed = search._flash_clc_combination_statuses_allowed(
                [cell for cell in combination_cells if isinstance(cell, dict)]
            ) and all(isinstance(cell, dict) for cell in combination_cells)
            metric["combination_projection_complete"] = projection_complete
            metric["successful_combination_depth_config_ids"] = successful_depth_ids
            metric["successful_combination_divisor_values"] = successful_divisor_values
            metric["combination_row_coverage_complete"] = row_coverage_complete
            metric["combination_column_coverage_complete"] = column_coverage_complete
            metric["combination_failure_statuses_allowed"] = failure_statuses_allowed
            metric["complete"] = bool(
                attempted_values == planned_values
                and len(selected_values) == len(planned_values)
                and set(selected_values) == set(planned_values)
                and all(
                    any(
                        config_id in successful_config_ids
                        for config_id in (
                            witness_config_ids.get(str(value)),
                            *witness_repair_candidate_ids.get(str(value), ()),
                        )
                    )
                    for value in planned_values
                )
                and all(
                    any(
                        config_id in successful_config_ids
                        for config_id in (
                            list(conditional_candidate_ids.get(str(value), ()))
                            + list(conditional_repair_candidate_ids.get(str(value), ()))
                        )
                    )
                    for value in conditional_values
                )
                and len(retained_values) == len(planned_values)
                and set(retained_values) == set(planned_values)
                and (
                    not metric["combination_required"]
                    or (
                        bool(combination_candidate_ids)
                        and len(combination_candidate_ids)
                        <= len(combination_depth_config_ids)
                        * len(combination_divisor_values)
                        and bool(combination_depth_config_ids)
                        and bool(combination_divisor_values)
                        and projection_complete
                        and row_coverage_complete
                        and column_coverage_complete
                        and failure_statuses_allowed
                    )
                )
            )
        clc_complete = all(bool(metric["complete"]) for metric in clc_metrics.values())
        for metric in compound_transfer_metrics:
            metric_transfers = metric["transfers"]
            limit = metric["limit"]
            target_count = metric["transfer_target_count"]
            successful_transfer_ids = metric["successful_transfer_config_ids"]
            qualified_transfer_ids = metric["qualified_transfer_config_ids"]
            assert isinstance(metric_transfers, list)
            assert isinstance(limit, int)
            assert isinstance(target_count, int)
            assert isinstance(successful_transfer_ids, list)
            assert isinstance(qualified_transfer_ids, list)
            metric["complete"] = bool(
                metric.get("catalog_error") is None
                and len(metric_transfers) == metric["transfer_count"]
                and 0 < target_count <= limit
                and len(metric_transfers)
                <= limit * (1 + policy.qualification_failure_retries)
                and len(successful_transfer_ids) >= target_count
                and len(set(successful_transfer_ids)) == len(successful_transfer_ids)
                and set(successful_transfer_ids) <= successful_config_ids
                and qualified_transfer_ids == successful_transfer_ids[:target_count]
                and len(qualified_transfer_ids) == target_count
                and len(set(qualified_transfer_ids)) == target_count
                and metric["failure_statuses_allowed"] is True
            )
        compound_catalog_complete = bool(
            not compound_catalog_errors
            and len(compound_transfer_metrics) == len(compound_leaves)
        )
        compound_complete = bool(
            compound_catalog_complete
            and all(bool(metric["complete"]) for metric in compound_transfer_metrics)
        )
        qualification_passes_planned = (
            int(schedule_anchor_pass_planned)
            + pipeline_pass_count
            + pipeline_repair_pass_count
            + clc_witness_pass_count
            + clc_witness_repair_pass_count
            + int(any(clc_conditional_values.values()))
            + clc_conditional_repair_pass_count
            + int(bool(clc_combination_leaves))
            + int(bool(compound_leaves))
            + compound_backfill_pass_count
            + (policy.family_probe_generations if family_probe_required else 0)
        )
        manifest_configs: dict[str, Config] = {}

        def add_manifest_config(
            config: Config,
        ) -> None:
            config_id = canonical_config_id(config)
            existing_config = manifest_configs.get(config_id)
            if existing_config is not None and existing_config != config:
                raise AssertionError(f"canonical config ID collision for {config_id}")
            manifest_configs[config_id] = config

        for config in exact_space or ():
            add_manifest_config(config)
        for member in qualified_population.values():
            add_manifest_config(member.config)

        config_manifest: dict[str, dict[str, object]] = {}
        for config_id, config in manifest_configs.items():
            config_manifest[config_id] = {
                "config": copy.deepcopy(config.config),
            }
        initial_results = []
        for member in initial_population:
            leaf = search._flash_structural_leaf(member)
            assert leaf is not None
            lanes = leaf_pipeline_lanes.get(leaf, ())
            initial_results.append(
                {
                    "config_id": canonical_config_id(member.config),
                    "family": leaf.pipeline_family,
                    "compound_packet": leaf.compound_exp2_packet,
                    "softmax_disc": leaf.softmax_disc,
                    **initial_measurements[id(member)],
                    "pipeline_lanes": [
                        search._flash_pipeline_lane_metric(lane)
                        for lane in lanes
                        if search._flash_member_matches_pipeline_lane(member, lane)
                    ],
                }
            )
        retained_family_limit = (
            live_family_count
            if policy.retained_families is None
            else min(policy.retained_families, live_family_count)
        )
        search._autotune_metrics.search_phase_metrics = {
            "phase": "cute_flash_structural_qualification_v24",
            "cute_flash_lane_policy_version": CUTE_FLASH_LANE_POLICY_VERSION,
            "completed": bool(
                rounds_completed == qualification_passes_planned
                and schedule_anchor_complete
                and lane_complete
                and ordinary_complete
                and clc_complete
                and compound_complete
                and family_probe_complete
            ),
            "initial_config_count": len(initial_config_ids),
            "initial_config_ids": initial_config_ids,
            "initial_results": initial_results,
            "schedule_anchor_design_source": (
                "live family x ordinary packet x softmax protocol from fragment defaults"
            ),
            "schedule_anchor_pass_planned": schedule_anchor_pass_planned,
            "schedule_anchor_pass_started": schedule_anchor_pass_started,
            "schedule_anchor_count": len(schedule_anchor_members),
            "schedule_anchor_complete": schedule_anchor_complete,
            "schedule_anchor_results": schedule_anchor_results,
            "measurement_timeline": measurement_timeline,
            "config_manifest": config_manifest,
            "exact_space_enumerated": exact_space is not None,
            "exact_space_exhausted": exact_space_exhausted,
            "exact_space_raw_budget": exact_space_raw_budget,
            "exact_space_config_ids": [
                canonical_config_id(config) for config in exact_space or ()
            ],
            "leaf_count": len(leaf_catalog),
            "ordinary_leaf_count": len(ordinary_leaves),
            "compound_leaf_count": len(compound_leaves),
            "leaf_results": [leaf_metrics[leaf] for leaf in ordinary_leaves],
            "pipeline_qualification_keys": list(
                search._flash_pipeline_qualification_keys()
            ),
            "qualification_rounds": policy.qualification_rounds,
            "qualification_rounds_started": rounds_started,
            "qualification_rounds_completed": rounds_completed,
            "qualification_passes_planned": qualification_passes_planned,
            "qualification_passes_started": rounds_started,
            "qualification_passes_completed": rounds_completed,
            "budget_exhausted": budget_exhausted,
            "pipeline_candidate_limit_per_leaf_per_round": (
                policy.pipeline_candidates_per_leaf_per_round
            ),
            "conditional_candidates_per_pipeline_lane": (
                policy.conditional_candidates_per_pipeline_lane
            ),
            "qualification_failure_retries": policy.qualification_failure_retries,
            "family_probe_generations": policy.family_probe_generations,
            "family_probe_generations_started": family_probe_generations_started,
            "family_probe_generations_completed": family_probe_generations_completed,
            "family_probe_candidates_per_path": (
                policy.family_probe_candidates_per_path
            ),
            "family_probe_required": family_probe_required,
            "family_probe_complete": family_probe_complete,
            "family_probe_path_limit": family_probe_path_limit,
            "family_probe_paths": family_probe_metrics,
            "neighbor_generation_limit_per_leaf_per_round": (
                qualification_neighbor_limit
            ),
            "candidate_count": len(candidate_ids),
            "leaves_with_candidates": len(leaves_with_candidates),
            "retained_candidates_per_leaf": policy.retained_candidates_per_leaf,
            "retained_family_cap": policy.retained_families,
            "retained_family_limit": retained_family_limit,
            "retained_family_slowdown_limit": (policy.retained_family_slowdown_limit),
            "clc_families": [clc_metrics[leaf] for leaf in clc_metrics],
            "compound_catalog_complete": compound_catalog_complete,
            "compound_catalog_errors": compound_catalog_errors,
            "compound_transfers": compound_transfer_metrics,
            "starting_path_limit": getattr(
                search, "_flash_promoted_path_limit", search.copies
            ),
            "maximum_path_capacity": search.copies,
            "unrestricted_path_exhausts_generation_budget": (
                policy.exhaust_unrestricted_path
            ),
        }
        search.population = [*qualified_population.values()]
        search.population.sort(key=performance)
        return generation_rounds_started
