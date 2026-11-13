"""Bounded compiler and schedule trials on structures discovered by a search."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import operator
from typing import TYPE_CHECKING

from ..exc import InvalidConfig
from .config_fragment import _immutable_fragment_key

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig


MAX_REFINEMENT_PARENTS = 4
MAX_REFINEMENT_JOINT_PRODUCT = 16
MAX_REFINEMENT_TRIALS = MAX_REFINEMENT_PARENTS * MAX_REFINEMENT_JOINT_PRODUCT


def compiler_refinement_policy(keys: tuple[str, ...]) -> dict[str, object]:
    return {
        "version": 6,
        "keys": keys,
        "max_parents": MAX_REFINEMENT_PARENTS,
        "max_new_trials": MAX_REFINEMENT_TRIALS,
        "max_joint_product": MAX_REFINEMENT_JOINT_PRODUCT,
        "ordering": "distance_key_parent",
    }


def same_refinement_value(left: object, right: object) -> bool:
    """Compare plain config values without collapsing scalar types or signed zero."""
    key = _immutable_fragment_key(left)
    return key is not None and key == _immutable_fragment_key(right)


@dataclasses.dataclass
class CompilerRefinementCandidate:
    parent: Config
    config: Config
    flat_values: FlatConfig


def compiler_refinement_candidates(
    generation: ConfigGeneration,
    parents: Sequence[Config],
    keys: tuple[str, ...],
) -> list[CompilerRefinementCandidate]:
    """Cross small compiler and schedule domains without changing other values.

    The incumbent arrives first, followed by measured performance order. Consider
    at most four distinct structures; parents differing only in refinement
    coordinates share a structure. Try the nearest candidates first and
    alternate structures at each distance so a short budget reaches each parent.
    Break distance ties by the order of the requested refinement keys.
    The trial cap covers every retained parent's bounded Cartesian neighborhood;
    the remaining search time can still stop new admissions earlier.
    Explicit overrides and restricted fragment search domains remain binding.
    Strict decoding refuses candidates needing a layout or prerequisite repair.
    Larger products retain single-coordinate exploration rather than allocating
    an unbounded Cartesian grid.
    """
    coordinates: dict[int, tuple[str, list[object]]] = {}
    for key in keys:
        layout = generation._key_to_flat_indices.get(key)
        if layout is None:
            continue
        indices, sequence = layout
        if sequence or len(indices) != 1:
            continue
        index = indices[0]
        if index in generation.overridden_flat_indices:
            continue
        choices = generation.flat_spec[index].search_values(
            limit=MAX_REFINEMENT_JOINT_PRODUCT
        )
        if choices is not None and len(choices) > 1:
            coordinates[index] = (key, choices)
    if not coordinates:
        return []

    combined = math.prod(len(choices) for _, choices in coordinates.values())
    coordinate_keys = {key for key, _ in coordinates.values()}

    structures: set[Hashable] = set()
    seen: set[Hashable] = set()
    candidates: list[tuple[int, int, int, CompilerRefinementCandidate]] = []
    for parent in parents:
        try:
            base, normalized_parent = generation.strict_config_pair(parent)
        except InvalidConfig:
            continue
        structure = _immutable_fragment_key(
            [value for i, value in enumerate(base) if i not in coordinates]
        )
        if structure is None or structure in structures:
            continue
        if len(structures) >= MAX_REFINEMENT_PARENTS:
            break
        structures.add(structure)
        parent_index = len(structures) - 1
        distance_ranks: dict[int, int] = {}
        if combined <= MAX_REFINEMENT_JOINT_PRODUCT:
            requests = (
                dict(zip(coordinates, values, strict=True))
                for values in itertools.product(
                    *(choices for _, choices in coordinates.values())
                )
            )
        else:
            requests = (
                {index: choice}
                for index, (_, choices) in coordinates.items()
                for choice in choices
            )
        # Cartesian product order depends on the parent's choice values. Keep
        # the backend's key priority even when a parent sits at the other end
        # of a choice domain; the sort is stable within each changed-key set.
        ordered_requests = sorted(
            requests,
            key=lambda replacements: tuple(
                position
                for position, index in enumerate(coordinates)
                if index in replacements
                and not same_refinement_value(base[index], replacements[index])
            ),
        )
        for replacements in ordered_requests:
            requested = copy.deepcopy(base)
            for index, choice in replacements.items():
                requested[index] = copy.deepcopy(choice)
            if same_refinement_value(requested, base):
                continue
            try:
                config = generation.strict_unflatten(requested)
                flat, config = generation.strict_config_pair(config)
            except InvalidConfig:
                continue
            if not same_refinement_value(flat, requested):
                continue
            # Fields outside the flat search surface must also survive.
            before = {
                k: v
                for k, v in normalized_parent.config.items()
                if k not in coordinate_keys
            }
            after = {k: v for k, v in config.config.items() if k not in coordinate_keys}
            identity = _immutable_fragment_key(config.config)
            if (
                not same_refinement_value(before, after)
                or identity is None
                or identity in seen
            ):
                continue
            seen.add(identity)
            distance = sum(
                not same_refinement_value(base[index], flat[index])
                for index in coordinates
            )
            rank = distance_ranks.get(distance, 0)
            distance_ranks[distance] = rank + 1
            candidates.append(
                (
                    distance,
                    rank,
                    parent_index,
                    CompilerRefinementCandidate(
                        normalized_parent,
                        config,
                        flat,
                    ),
                )
            )
    candidates.sort(key=operator.itemgetter(0, 1, 2))
    return [candidate for _, _, _, candidate in candidates]
