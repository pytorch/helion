from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
import operator
import random
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import TypeVar
from typing import cast

from .._compat import warps_to_threads
from ..exc import AutotuneError
from ..exc import InvalidConfig
from .block_id_sequence import BlockIdSequence
from .config_fragment import Category
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import ListOf
from .config_fragment import PowerOfTwoFragment
from .config_spec import shrink_block_sizes_for_numel_constraints
from helion._dist_utils import sync_seed

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .. import Config
    from .._compiler.cute.cute_flash import FlashStructuralLeaf
    from . import ConfigSpec
    from .config_priors import ValuePrior

FlatConfig = list[object]


@dataclasses.dataclass(frozen=True)
class CoordinateNeighborProjection:
    """One deterministic raw coordinate change and its normalized result."""

    flat_index: int
    key: str
    sequence_index: int | None
    from_value: object
    to_value: object
    outcome: Literal["candidate", "incumbent_alias", "candidate_alias", "invalid"]
    flat_values: FlatConfig | None
    config: Config | None


TRITON_MAX_TENSOR_NUMEL = 1048576
_FLASH_STRUCTURAL_LEAF_GOAL_KEY = "__flash_structural_leaf__"

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


def _flash_env_scoped(fn: _CallableT) -> _CallableT:
    """Run one ConfigGeneration pass under a flash env snapshot.

    Pass-level operations normalize thousands of candidate configs, and every
    normalization reads dozens of ``HELION_CUTE_FLASH_*`` env vars through
    ``resolve_flash_config``. Snapshotting at pass entry keeps those reads out
    of the hot loop while still honoring env patches made between passes
    (tests rely on ``mock.patch.dict`` taking effect per operation).
    """

    @functools.wraps(fn)
    def wrapper(self: ConfigGeneration, *args: object, **kwargs: object) -> object:
        if not self.config_spec.cute_flash_search_enabled:
            return fn(self, *args, **kwargs)
        from .._compiler.cute.cute_flash import flash_env_snapshot_scope

        with flash_env_snapshot_scope():
            return fn(self, *args, **kwargs)

    return cast("_CallableT", wrapper)


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


def _value_or(value: object, fallback: Callable[[], object]) -> object:
    """Return ``value`` unless it is ``None``, in which case call ``fallback``.

    Used by the biased sampler: a prior returns ``None`` to decline a slot, and
    the slot then falls back to the fragment's uniform ``random()``.
    """
    return fallback() if value is None else value


class ConfigGeneration:
    def __init__(
        self,
        config_spec: ConfigSpec,
        *,
        overrides: Mapping[str, object] | None = None,
        _flash_pipeline_family_override: str | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
    ) -> None:
        def _collect_spec(spec: ConfigSpecFragment) -> object:
            """
            Collect a configuration specification fragment.

            Args:
                spec: The configuration specification fragment.

            Returns:
                The default value of the fragment.
            """
            self.flat_spec.append(spec)
            return spec.default()

        super().__init__()
        self.config_spec = config_spec
        self.process_group_name = process_group_name
        self._advanced_controls_files = advanced_controls_files
        self._flash_pipeline_family_override = (
            _flash_pipeline_family_override
            if config_spec.cute_flash_search_enabled
            else None
        )
        self.flat_spec: list[ConfigSpecFragment] = []
        if self._flash_pipeline_family_override is None:
            config_spec.flat_config(
                _collect_spec,
                advanced_controls_files=advanced_controls_files,
            )
        else:
            config_spec._flat_config_with_flash_family(
                _collect_spec,
                advanced_controls_files=advanced_controls_files,
                flash_pipeline_family=self._flash_pipeline_family_override,
            )
        assert self.flat_spec, "No config values to tune"
        self._override_values = dict(overrides or {})
        self.block_size_indices: list[int] = [
            i
            for i, spec in enumerate(self.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        ]
        self.num_threads_indices: list[int] = []
        self._cute_num_thread_block_pairs: list[tuple[int, int]] = []
        self._cute_block_index_by_id: dict[int, int] = {}
        self._cute_num_thread_index_by_id: dict[int, int] = {}
        self._cute_flatten_loop_groups: list[tuple[int, list[int]]] = []
        if self.config_spec.backend_name == "cute":
            self._init_cute_num_thread_pairs()
        self.num_warps_index: int = next(
            (
                i
                for i, spec in enumerate(self.flat_spec)
                if spec.category() == Category.NUM_WARPS
            ),
            -1,
        )
        self.min_block_size: int = (
            max([spec.min_size for spec in config_spec.block_sizes])
            if config_spec.block_sizes
            else 1
        )
        # Running count of candidate configs rejected as InvalidConfig by the
        # internal generation retry loops (random_config / random_population).
        # These rejections are otherwise invisible to callers because the loops
        # silently retry; exposing the count lets the search-space logger report
        # explored-invalid alongside explored-valid.
        self.invalid_config_count: int = 0
        self._flash_coverage_cache: list[FlatConfig] | None = None
        self._flash_coverage_active_values_cache: list[tuple[str, object]] | None = None
        self._flash_coverage_uncovered_cache: list[tuple[str, object]] | None = None
        self._flash_coverage_underqualified_cache: (
            list[tuple[str, object, int]] | None
        ) = None
        self._flash_structural_leaf_catalog_cache: list[FlashStructuralLeaf] | None = (
            None
        )
        self._flash_pipeline_lane_catalog_cache: (
            dict[FlashStructuralLeaf, tuple[tuple[str, object], ...]] | None
        ) = None
        self._flash_pipeline_lane_witness_cache: (
            dict[tuple[FlashStructuralLeaf, str, object], Config] | None
        ) = None
        self._flash_clc_lane_catalog_cache: (
            dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]] | None
        ) = None
        self._flash_clc_lane_witness_cache: (
            dict[tuple[FlashStructuralLeaf, int], Config] | None
        ) = None
        self._flash_structural_underqualified_leaves_cache: (
            list[tuple[FlashStructuralLeaf, int]] | None
        ) = None
        self._flash_coverage_uncovered_interactions_cache: (
            list[tuple[tuple[str, ...], tuple[object, ...]]] | None
        ) = None
        self._flash_coverage_active_interactions_cache: (
            list[tuple[tuple[str, ...], tuple[object, ...]]] | None
        ) = None
        self._flash_parent_coverage_prefix_count_cache: int | None = None
        self._flash_qualification_prefix_count_cache: int | None = None

    def _init_cute_num_thread_pairs(self) -> None:
        """Pair each CuTe num_threads flat slot with its block_size slot."""
        try:
            block_indices, _ = self._key_to_flat_indices["block_sizes"]
            num_thread_indices, _ = self._key_to_flat_indices["num_threads"]
        except KeyError:
            return
        self.num_threads_indices = num_thread_indices
        block_index_by_id = {
            spec.block_id: block_indices[i]
            for i, spec in enumerate(self.config_spec.block_sizes)
            if i < len(block_indices)
        }
        num_thread_index_by_id = {
            spec.block_id: num_thread_indices[i]
            for i, spec in enumerate(self.config_spec.num_threads)
            if i < len(num_thread_indices)
        }
        self._cute_block_index_by_id = block_index_by_id
        self._cute_num_thread_index_by_id = num_thread_index_by_id
        self._cute_num_thread_block_pairs = [
            (num_thread_indices[i], block_index_by_id[spec.block_id])
            for i, spec in enumerate(self.config_spec.num_threads)
            if i < len(num_thread_indices) and spec.block_id in block_index_by_id
        ]
        try:
            flatten_indices, _ = self._key_to_flat_indices["flatten_loops"]
        except KeyError:
            return
        self._cute_flatten_loop_groups = [
            (
                flatten_indices[i],
                [
                    block_id
                    for block_id in spec.block_ids
                    if block_id in block_index_by_id
                    and block_id in num_thread_index_by_id
                ],
            )
            for i, spec in enumerate(self.config_spec.flatten_loops)
            if i < len(flatten_indices)
        ]

    @functools.cached_property
    def overridden_flat_indices(self) -> set[int]:
        """Return flat_spec indices that are frozen by config overrides."""
        if not self._override_values:
            return set()
        result: set[int] = set()
        for key in self._override_values:
            if key in self._key_to_flat_indices:
                indices, _ = self._key_to_flat_indices[key]
                result.update(indices)
        return result

    @functools.cached_property
    def _key_to_flat_indices(self) -> dict[str, tuple[list[int], bool]]:
        """Build mapping from config key names to (flat_spec indices, is_sequence).

        Derived from ConfigSpec.flat_key_layout().
        """
        mapping: dict[str, tuple[list[int], bool]] = {}
        idx = 0
        layout = (
            self.config_spec.flat_key_layout(
                advanced_controls_files=self._advanced_controls_files
            )
            if self._flash_pipeline_family_override is None
            else self.config_spec._flat_key_layout_with_flash_family(
                advanced_controls_files=self._advanced_controls_files,
                flash_pipeline_family=self._flash_pipeline_family_override,
            )
        )
        for key, count, is_sequence in layout:
            mapping[key] = (list(range(idx, idx + count)), is_sequence)
            idx += count
        assert idx == len(self.flat_spec), (
            f"flat_key_layout() total ({idx}) != flat_spec length ({len(self.flat_spec)})"
        )
        return mapping

    def _apply_overrides(self, config: Config) -> Config:
        if not self._override_values:
            return config
        for key, value in self._override_values.items():
            config.config[key] = copy.deepcopy(value)
        self.config_spec.prepare_override_normalization(
            config.config,
            self._override_values,
        )
        self.config_spec.normalize(config.config)
        return config

    @staticmethod
    def _largest_power_of_two_at_most(value: int) -> int:
        return 1 << (max(value, 1).bit_length() - 1)

    def _repair_cute_num_threads(self, flat_config: FlatConfig) -> None:
        """Keep CuTe launch-thread choices compatible with tuned block sizes."""
        if not self._cute_num_thread_block_pairs:
            return

        for num_threads_idx, block_size_idx in self._cute_num_thread_block_pairs:
            num_threads = flat_config[num_threads_idx]
            block_size = flat_config[block_size_idx]
            if (
                type(num_threads) is not int
                or num_threads == 0
                or type(block_size) is not int
                or block_size <= 0
            ):
                continue
            if num_threads > block_size:
                num_threads = self._largest_power_of_two_at_most(block_size)
            while num_threads > 1 and block_size % num_threads != 0:
                num_threads //= 2
            flat_config[num_threads_idx] = max(num_threads, 1)

        for flatten_idx, block_ids in self._cute_flatten_loop_groups:
            if flat_config[flatten_idx] is not True:
                continue
            group: list[tuple[int, int, bool, int]] = []
            for block_id in block_ids:
                block_size = flat_config[self._cute_block_index_by_id[block_id]]
                num_threads_idx = self._cute_num_thread_index_by_id[block_id]
                num_threads = flat_config[num_threads_idx]
                if (
                    type(block_size) is not int
                    or block_size <= 0
                    or type(num_threads) is not int
                ):
                    group = []
                    break
                resolved_threads = num_threads if num_threads > 0 else block_size
                group.append(
                    (num_threads_idx, block_size, num_threads == 0, resolved_threads)
                )
            if not group:
                continue
            thread_product = functools.reduce(
                operator.mul, (item[3] for item in group), 1
            )
            auto_positions = [i for i, item in enumerate(group) if item[2]]
            while thread_product > 1024 and auto_positions:
                largest_pos = max(auto_positions, key=lambda i: group[i][3])
                num_threads_idx, block_size, is_auto, resolved_threads = group[
                    largest_pos
                ]
                if resolved_threads <= 1:
                    auto_positions.remove(largest_pos)
                    continue
                next_threads = resolved_threads // 2
                while next_threads > 1 and block_size % next_threads != 0:
                    next_threads //= 2
                if next_threads == resolved_threads:
                    auto_positions.remove(largest_pos)
                    continue
                flat_config[num_threads_idx] = next_threads
                group[largest_pos] = (
                    num_threads_idx,
                    block_size,
                    is_auto,
                    next_threads,
                )
                thread_product = (thread_product // resolved_threads) * next_threads

        explicit_indices = [
            idx
            for idx, _ in self._cute_num_thread_block_pairs
            if type(flat_config[idx]) is int and cast("int", flat_config[idx]) > 0
        ]
        thread_product = functools.reduce(
            operator.mul,
            (cast("int", flat_config[idx]) for idx in explicit_indices),
            1,
        )
        while thread_product > 1024 and explicit_indices:
            largest_idx = max(
                explicit_indices,
                key=lambda idx: cast("int", flat_config[idx]),
            )
            largest = cast("int", flat_config[largest_idx])
            if largest <= 1:
                break
            flat_config[largest_idx] = largest // 2
            thread_product //= 2

    def flatten(self, config: Config) -> FlatConfig:
        """Inverse of unflatten: convert a Config to a FlatConfig."""
        result = self._fragment_default_flat()
        flat_fields = (
            self.config_spec._flat_fields()
            if self._flash_pipeline_family_override is None
            else self.config_spec._flat_fields_with_flash_family(
                self._flash_pipeline_family_override
            )
        )
        for key, (indices, is_sequence) in self._key_to_flat_indices.items():
            if key not in config.config:
                has_default, value = self.config_spec.flatten_missing_field_default(
                    key,
                    config.config,
                )
                if not has_default:
                    continue
            else:
                value = config.config[key]
            if is_sequence:
                assert isinstance(value, list)
                field = flat_fields[key]
                assert isinstance(field, BlockIdSequence)
                encoded_values = field._encode_flat_values(self.config_spec, value)
                for idx, encoded_value in zip(indices, encoded_values, strict=True):
                    result[idx] = copy.deepcopy(encoded_value)
            else:
                assert len(indices) == 1
                field = self.flat_spec[indices[0]]
                if isinstance(field, ListOf) and not isinstance(value, list):
                    value = [copy.deepcopy(value) for _ in range(field.length)]
                result[indices[0]] = copy.deepcopy(value)
        self._repair_cute_num_threads(result)
        return result

    def canonicalize_flat(self, flat_values: FlatConfig) -> tuple[FlatConfig, Config]:
        """Normalize a flat config and return an owned matching flat/config pair."""
        config = self.unflatten(copy.deepcopy(flat_values))
        return self.flatten(config), config

    def _flat_coordinate_identities(self) -> list[tuple[str, int | None]]:
        identities: list[tuple[str, int | None]] = [("", None) for _ in self.flat_spec]
        for key, (indices, is_sequence) in self._key_to_flat_indices.items():
            for sequence_index, flat_index in enumerate(indices):
                identities[flat_index] = (
                    key,
                    sequence_index if is_sequence else None,
                )
        return identities

    @staticmethod
    def _coordinate_catalog_value(value: object) -> object:
        """Return a JSON-safe copy of one coordinate value."""
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise AutotuneError(f"non-finite coordinate surface value {value!r}")
            return value
        if isinstance(value, (list, tuple)):
            return [ConfigGeneration._coordinate_catalog_value(item) for item in value]
        raise AutotuneError(
            f"terminal coordinate surface contains a non-JSON value: {value!r}"
        )

    def coordinate_surface_catalog(self, *, radius: int = 1) -> dict[str, object]:
        """Describe every deterministic pattern-neighbor row for this surface."""
        if type(radius) is not int or radius < 1:
            raise ValueError(f"Expected positive int radius, got {radius!r}")

        identities = self._flat_coordinate_identities()
        overridden = self.overridden_flat_indices
        coordinates: list[dict[str, object]] = []
        for flat_index, spec in enumerate(self.flat_spec):
            active_values = spec.search_values(limit=10_000)
            if active_values is None:
                raise AutotuneError(
                    "terminal coordinate surface is not finitely enumerable for "
                    f"flat index {flat_index} ({type(spec).__name__})"
                )
            base_values = (
                list(spec.choices)
                if isinstance(spec, EnumFragment)
                else list(active_values)
            )
            default = spec.default()
            if default not in base_values:
                base_values.append(default)
            key, sequence_index = identities[flat_index]
            coordinates.append(
                {
                    "flat_index": flat_index,
                    "key": key,
                    "sequence_index": sequence_index,
                    "fragment_type": type(spec).__name__,
                    "overridden": flat_index in overridden,
                    "active_values": [
                        self._coordinate_catalog_value(value) for value in active_values
                    ],
                    "neighbors_by_value": [
                        {
                            "from_value": self._coordinate_catalog_value(value),
                            "to_values": [
                                self._coordinate_catalog_value(neighbor)
                                for neighbor in spec.pattern_neighbors(value, radius)
                            ],
                        }
                        for value in base_values
                    ],
                }
            )
        return {
            "schema_version": 1,
            "radius": radius,
            "coordinates": coordinates,
        }

    def flash_terminal_coordinate_surface_catalog(
        self, *, radius: int = 1
    ) -> dict[str, object]:
        """Return the ordered coordinate surface for every reachable flash leaf."""
        if not self.config_spec.cute_flash_search_enabled:
            raise AutotuneError("terminal coordinate surface requires CuTe flash")

        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY

        leaves: list[dict[str, object]] = []
        for leaf in self.flash_structural_leaf_catalog():
            overrides = dict(self._override_values)
            overrides[FLASH_PIPELINE_FAMILY_KEY] = leaf.pipeline_family
            overrides[FLASH_SOFTMAX_DISC_KEY] = leaf.softmax_disc
            if leaf.compound_exp2_packet is not None:
                overrides[FLASH_EXP2_PACKET_KEY] = leaf.compound_exp2_packet
            leaf_generation = self.config_spec.create_config_generation(
                overrides=overrides,
                advanced_controls_files=self._advanced_controls_files,
                process_group_name=self.process_group_name,
            )
            surface = leaf_generation.coordinate_surface_catalog(radius=radius)
            leaves.append(
                {
                    "leaf": {
                        "family": leaf.pipeline_family,
                        "compound_packet": leaf.compound_exp2_packet,
                        "softmax_disc": leaf.softmax_disc,
                    },
                    "coordinates": surface["coordinates"],
                }
            )
        return {
            "schema_version": 1,
            "radius": radius,
            "leaves": leaves,
        }

    @_flash_env_scoped
    def coordinate_neighbor_projections(
        self, base: FlatConfig, *, radius: int = 1
    ) -> list[CoordinateNeighborProjection]:
        """Enumerate every normalized one-coordinate pattern neighbor.

        The returned order follows the flat ConfigSpec layout and each fragment's
        own deterministic ``pattern_neighbors`` order. Invalid and normalized
        alias requests remain in the result so callers can audit completeness.
        """
        canonical_base, base_config = self.canonicalize_flat(base)
        flat_identities = self._flat_coordinate_identities()

        seen = {base_config}
        result: list[CoordinateNeighborProjection] = []
        overridden = self.overridden_flat_indices
        for flat_index, spec in enumerate(self.flat_spec):
            if flat_index in overridden:
                continue
            key, sequence_index = flat_identities[flat_index]
            current = canonical_base[flat_index]
            for value in spec.pattern_neighbors(current, radius):
                requested = copy.deepcopy(canonical_base)
                requested[flat_index] = copy.deepcopy(value)
                try:
                    normalized_flat, config = self.canonicalize_flat(requested)
                except InvalidConfig:
                    result.append(
                        CoordinateNeighborProjection(
                            flat_index,
                            key,
                            sequence_index,
                            copy.deepcopy(current),
                            copy.deepcopy(value),
                            "invalid",
                            None,
                            None,
                        )
                    )
                    continue

                if config == base_config:
                    outcome = "incumbent_alias"
                elif config in seen:
                    outcome = "candidate_alias"
                else:
                    outcome = "candidate"
                    seen.add(config)
                result.append(
                    CoordinateNeighborProjection(
                        flat_index,
                        key,
                        sequence_index,
                        copy.deepcopy(current),
                        copy.deepcopy(value),
                        outcome,
                        normalized_flat,
                        config,
                    )
                )
        return result

    @_flash_env_scoped
    def canonicalize_coordinate_projections(
        self,
        projections: Sequence[CoordinateNeighborProjection],
        *,
        base_config: Config,
    ) -> list[CoordinateNeighborProjection]:
        """Canonicalize projections against this generation's full surface.

        A conditional generation can inject values for fields that are inactive
        on the full search surface. Recanonicalizing here keeps recorded config
        identities aligned with the configs that are actually benchmarked.
        """
        _, canonical_base = self.canonicalize_flat(self.flatten(base_config))
        seen = {canonical_base}
        result: list[CoordinateNeighborProjection] = []
        for projection in projections:
            if projection.config is None:
                result.append(
                    dataclasses.replace(
                        projection,
                        outcome="invalid",
                        flat_values=None,
                        config=None,
                    )
                )
                continue
            try:
                flat_values, config = self.canonicalize_flat(
                    self.flatten(projection.config)
                )
            except InvalidConfig:
                result.append(
                    dataclasses.replace(
                        projection,
                        outcome="invalid",
                        flat_values=None,
                        config=None,
                    )
                )
                continue
            if config == canonical_base:
                outcome = "incumbent_alias"
            elif config in seen:
                outcome = "candidate_alias"
            else:
                outcome = "candidate"
                seen.add(config)
            result.append(
                dataclasses.replace(
                    projection,
                    outcome=outcome,
                    flat_values=flat_values,
                    config=config,
                )
            )
        return result

    def unflatten(self, flat_values: FlatConfig) -> Config:
        """
        Convert a flat configuration back into a full configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            The full configuration object.
        """

        def get_next_value(spec: ConfigSpecFragment) -> object:
            i = next(count)
            assert type(self.flat_spec[i]) is type(spec)
            return flat_values[i]

        assert len(flat_values) == len(self.flat_spec)
        self._repair_cute_num_threads(flat_values)
        count: itertools.count[int] = itertools.count()
        if self._flash_pipeline_family_override is None:
            config = self.config_spec.flat_config(
                get_next_value,
                advanced_controls_files=self._advanced_controls_files,
            )
        else:
            config = self.config_spec._flat_config_with_flash_family(
                get_next_value,
                advanced_controls_files=self._advanced_controls_files,
                flash_pipeline_family=self._flash_pipeline_family_override,
            )
        assert next(count) == len(flat_values)
        config = self._apply_overrides(config)
        # Overrides may reintroduce pointer stores that break subtiled outputs
        self.config_spec.fix_epilogue_subtile_store_indexing(config.config)
        return config

    def block_numel(self, flat_config: FlatConfig) -> int:
        return functools.reduce(
            operator.mul,
            [cast("int", flat_config[i]) for i in self.block_size_indices],
            1,
        )

    def _shrink_for_numel_constraints(self, flat_config: FlatConfig) -> None:
        """Shrink block sizes in flat_config to satisfy numel constraints."""
        constraints = self.config_spec.tensor_numel_constraints
        if not constraints:
            return
        block_sizes = [cast("int", flat_config[i]) for i in self.block_size_indices]
        min_sizes = [
            max(self.flat_spec[i].get_minimum(), self.min_block_size)
            for i in self.block_size_indices
        ]
        shrink_block_sizes_for_numel_constraints(constraints, block_sizes, min_sizes)
        for idx, fi in enumerate(self.block_size_indices):
            flat_config[fi] = block_sizes[idx]

    def shrink_config(
        self, flat_config: FlatConfig, max_elements_per_thread: int
    ) -> None:
        """
        Fully random configs tend to run out of resources and tile a long time to compile.
        Here we shrink the config to a reasonable size.

        Args:
            flat_config: config to mutate in place
            max_elements_per_thread: maximum number of elements per thread
        """
        if self.num_warps_index < 0 or not self.block_size_indices:
            return
        num_threads = warps_to_threads(cast("int", flat_config[self.num_warps_index]))
        # Respect the backend's per-tile element ceiling (Triton: 2**20;
        # Pallas: None, since the real bound is VMEM bytes). Unit-test
        # callers may invoke shrink_config without an active environment;
        # default to the Triton limit in that case.
        from .._compiler.compile_environment import CompileEnvironment

        backend_limit: int | None = TRITON_MAX_TENSOR_NUMEL
        if CompileEnvironment.has_current():
            backend_limit = CompileEnvironment.current().backend.max_tensor_numel
        theoretical_max_elements = max_elements_per_thread * num_threads
        max_elements = (
            theoretical_max_elements
            if backend_limit is None
            else min(theoretical_max_elements, backend_limit)
        )
        while self.block_numel(flat_config) > max_elements:
            changes = 0
            for i in self.block_size_indices:
                val = flat_config[i]
                assert isinstance(val, int)
                threshold = max(self.flat_spec[i].get_minimum(), self.min_block_size)
                if val // 2 >= threshold:
                    flat_config[i] = val // 2
                    changes += 1
            if changes == 0:
                break
        self._shrink_for_numel_constraints(flat_config)
        self._repair_cute_num_threads(flat_config)

    def _fragment_default_flat(self) -> FlatConfig:
        """
        Retrieve the default flat configuration from raw fragment defaults.

        Returns:
            The default flat configuration values.
        """
        config = [spec.default() for spec in self.flat_spec]
        self._shrink_for_numel_constraints(config)
        self._repair_cute_num_threads(config)
        return config

    def default_flat(self) -> FlatConfig:
        """
        Retrieve the conservative autotuning reference configuration.

        Returns:
            The default flat configuration values.
        """
        return self._fragment_default_flat()

    @_flash_env_scoped
    def _flash_deterministic_coverage_flats(self) -> list[FlatConfig]:
        """Build a normalized covering design for active flash choices.

        Candidate contexts are normalized before coverage is measured. Missing
        effective values receive a one- or two-axis witness, then a deterministic
        greedy reduction keeps a compact set covering every reachable choice.
        This avoids both a Cartesian product and measured-winner presets; uniform
        random candidates fill the rest of the requested population.
        """
        if self._flash_coverage_cache is not None:
            return copy.deepcopy(self._flash_coverage_cache)
        if not self.config_spec.cute_flash_search_enabled:
            self._flash_coverage_cache = []
            self._flash_coverage_active_values_cache = []
            self._flash_coverage_uncovered_cache = []
            self._flash_coverage_underqualified_cache = []
            self._flash_structural_leaf_catalog_cache = []
            self._flash_pipeline_lane_catalog_cache = {}
            self._flash_pipeline_lane_witness_cache = {}
            self._flash_clc_lane_catalog_cache = {}
            self._flash_clc_lane_witness_cache = {}
            self._flash_structural_underqualified_leaves_cache = []
            self._flash_coverage_uncovered_interactions_cache = []
            self._flash_coverage_active_interactions_cache = []
            self._flash_parent_coverage_prefix_count_cache = 0
            self._flash_qualification_prefix_count_cache = 0
            return []
        from .._compiler.cute.cute_flash import FLASH_AUTOTUNE_CONFIG_KEYS
        from .._compiler.cute.cute_flash import FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
        from .._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY
        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_FLAGS
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import FLASH_S_STAGE_KEY
        from .._compiler.cute.cute_flash import flash_structural_leaf_from_config

        axes: dict[str, tuple[int, tuple[object, ...]]] = {}
        enum_axes: dict[str, tuple[int, EnumFragment]] = {}
        for key in FLASH_AUTOTUNE_CONFIG_KEYS:
            key_layout = self._key_to_flat_indices.get(key)
            if key_layout is None:
                continue
            indices, is_sequence = key_layout
            if is_sequence or len(indices) != 1:
                continue
            index = indices[0]
            fragment = self.flat_spec[index]
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
            ) or index in self.overridden_flat_indices:
                continue
            axes[key] = (index, choices)
        clc_coverage_choices: tuple[object, ...] | None = None
        clc_layout = self._key_to_flat_indices.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
        if clc_layout is not None and not clc_layout[1] and len(clc_layout[0]) == 1:
            clc_fragment = self.flat_spec[clc_layout[0][0]]
            if isinstance(clc_fragment, EnumFragment):
                clc_coverage_choices = (
                    clc_fragment.coverage_choices
                    if clc_fragment.coverage_choices is not None
                    else clc_fragment.choices
                    if clc_fragment.search_choices is None
                    else clc_fragment.search_choices
                )
        if not axes:
            self._flash_coverage_cache = []
            self._flash_coverage_active_values_cache = []
            self._flash_coverage_uncovered_cache = []
            self._flash_coverage_underqualified_cache = []
            self._flash_structural_leaf_catalog_cache = []
            self._flash_pipeline_lane_catalog_cache = {}
            self._flash_pipeline_lane_witness_cache = {}
            self._flash_clc_lane_catalog_cache = {}
            self._flash_clc_lane_witness_cache = {}
            self._flash_structural_underqualified_leaves_cache = []
            self._flash_coverage_uncovered_interactions_cache = []
            self._flash_coverage_active_interactions_cache = []
            self._flash_parent_coverage_prefix_count_cache = 0
            self._flash_qualification_prefix_count_cache = 0
            return []

        base = self._fragment_default_flat()
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
        for anchor in self.flash_low_confound_schedule_anchor_configs():
            raw_contexts.append(self.flatten(anchor))

        # Family contexts let packet normalization apply its required parent and
        # child fields without embedding any arithmetic performance choices.
        for key in (FLASH_PIPELINE_FAMILY_KEY, FLASH_EXP2_PACKET_KEY):
            axis = axes.get(key)
            if axis is not None:
                for value in axis[1]:
                    append_variant({key: value})

        interaction_groups = tuple(
            group
            for group in FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
            if all(key in enum_axes for key in group)
            and any(key in axes for key in group)
        )
        for group in interaction_groups:
            for combination in itertools.product(
                *(
                    axes[key][1]
                    if key in axes
                    else (self._override_values.get(key, enum_axes[key][1].default()),)
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
        self._flash_coverage_active_values_cache = copy.deepcopy(goal_order)
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
                config = self.unflatten(copy.deepcopy(raw))
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
                and clc_layout[0][0] not in self.overridden_flat_indices
                and config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                not in clc_coverage_choices
                and not allow_clc_refinement
            ):
                # CLC zero means an input-layout-dependent automatic value and can
                # alias an advertised divisor. It is valid for fixed configs, but
                # cannot serve as a distinct structural-search witness.
                normalized_cache[cache_key] = None
                return None
            flat = self.flatten(config)
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
                + (
                    [(_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf)]
                    if leaf is not None
                    else []
                )
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
        self._flash_coverage_active_interactions_cache = copy.deepcopy(
            interaction_goal_order
        )

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
                        if (
                            normalized is None
                            or normalized[0].config.get(key) != requested
                        ):
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
                    if (
                        parent_goal not in normalized[2]
                        or child_goal not in normalized[2]
                    ):
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

        self._flash_structural_leaf_catalog_cache = copy.deepcopy(leaf_catalog)
        pipeline_lane_catalog: dict[
            FlashStructuralLeaf, tuple[tuple[str, object], ...]
        ] = {}
        pipeline_lane_witnesses: dict[
            tuple[FlashStructuralLeaf, str, object], Config
        ] = {}
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
                if axis is None or axis[0] in self.overridden_flat_indices:
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
        self._flash_pipeline_lane_catalog_cache = copy.deepcopy(pipeline_lane_catalog)
        self._flash_pipeline_lane_witness_cache = copy.deepcopy(pipeline_lane_witnesses)
        clc_lane_catalog: dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]] = {}
        clc_lane_witnesses: dict[tuple[FlashStructuralLeaf, int], Config] = {}
        if (
            clc_layout is not None
            and not clc_layout[1]
            and len(clc_layout[0]) == 1
            and clc_layout[0][0] not in self.overridden_flat_indices
        ):
            clc_index = clc_layout[0][0]
            clc_fragment = self.flat_spec[clc_index]
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
                                or config.config.get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                                != value
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
        self._flash_clc_lane_catalog_cache = copy.deepcopy(clc_lane_catalog)
        self._flash_clc_lane_witness_cache = copy.deepcopy(clc_lane_witnesses)
        leaf_goal_order = [
            (_FLASH_STRUCTURAL_LEAF_GOAL_KEY, leaf) for leaf in leaf_catalog
        ]
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
        self._flash_parent_coverage_prefix_count_cache = len(selected)
        qualification_remaining: dict[tuple[str | tuple[str, ...], object], int] = {
            goal: 2 - sum(goal in covers for covers in selected_covers)
            for goal in qualification_goal_order
            if sum(goal in covers for covers in selected_covers) < 2
        }
        select_multicover(qualification_remaining)

        self._flash_qualification_prefix_count_cache = len(selected)
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
        self._flash_coverage_cache = selected
        self._flash_coverage_uncovered_cache = [
            goal for goal in goal_order if goal in uncovered
        ]
        self._flash_coverage_uncovered_interactions_cache = [
            goal for goal in interaction_goal_order if goal in uncovered
        ]
        self._flash_coverage_underqualified_cache = [
            (
                key,
                value,
                sum((key, value) in covers for covers in selected_covers),
            )
            for key, value in structural_value_qualification_goal_order
            if sum((key, value) in covers for covers in selected_covers) < 2
        ]
        self._flash_structural_underqualified_leaves_cache = [
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
        return copy.deepcopy(selected)

    def flash_structural_coverage_uncovered_values(
        self,
    ) -> list[tuple[str, object]]:
        """Return active flash values for which the design found no witness."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_uncovered_cache is not None
        return copy.deepcopy(self._flash_coverage_uncovered_cache)

    def flash_structural_coverage_active_values(self) -> list[tuple[str, object]]:
        """Return the live normalized-fragment values the design must cover."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_active_values_cache is not None
        return copy.deepcopy(self._flash_coverage_active_values_cache)

    def flash_structural_coverage_underqualified_values(
        self,
    ) -> list[tuple[str, object, int]]:
        """Return family values with fewer than two witnesses."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_underqualified_cache is not None
        return copy.deepcopy(self._flash_coverage_underqualified_cache)

    def flash_structural_leaf_catalog(self) -> list[FlashStructuralLeaf]:
        """Return exact ordinary-family and compound-packet schedule leaves."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_structural_leaf_catalog_cache is not None
        return copy.deepcopy(self._flash_structural_leaf_catalog_cache)

    def flash_pipeline_lane_catalog(
        self,
    ) -> dict[FlashStructuralLeaf, tuple[tuple[str, object], ...]]:
        """Return every reachable nonsingleton KV/S value for each exact leaf."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_pipeline_lane_catalog_cache is not None
        return copy.deepcopy(self._flash_pipeline_lane_catalog_cache)

    def flash_pipeline_lane_witnesses(
        self,
    ) -> dict[tuple[FlashStructuralLeaf, str, object], Config]:
        """Return one deterministic normalized config for every catalog lane."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_pipeline_lane_witness_cache is not None
        return copy.deepcopy(self._flash_pipeline_lane_witness_cache)

    def flash_clc_lane_catalog(
        self,
    ) -> dict[FlashStructuralLeaf, dict[str, tuple[int, ...]]]:
        """Return hierarchical legal, anchor, and refinement CLC divisors."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_clc_lane_catalog_cache is not None
        return copy.deepcopy(self._flash_clc_lane_catalog_cache)

    def flash_clc_lane_witnesses(
        self,
    ) -> dict[tuple[FlashStructuralLeaf, int], Config]:
        """Return a normalized witness for each attempted ordinary CLC divisor."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_clc_lane_witness_cache is not None
        return copy.deepcopy(self._flash_clc_lane_witness_cache)

    def flash_structural_coverage_active_leaves(self) -> list[FlashStructuralLeaf]:
        """Compatibility spelling for the exact structural leaf catalog."""
        return self.flash_structural_leaf_catalog()

    def flash_structural_coverage_underqualified_leaves(
        self,
    ) -> list[tuple[FlashStructuralLeaf, int]]:
        """Return nonsingleton ordinary leaves with fewer than two prefix rows."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_structural_underqualified_leaves_cache is not None
        return copy.deepcopy(self._flash_structural_underqualified_leaves_cache)

    def flash_structural_coverage_uncovered_interactions(
        self,
    ) -> list[tuple[tuple[str, ...], tuple[object, ...]]]:
        """Return reachable declared field interactions lacking a design row."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_uncovered_interactions_cache is not None
        return copy.deepcopy(self._flash_coverage_uncovered_interactions_cache)

    def flash_structural_coverage_active_interactions(
        self,
    ) -> list[tuple[tuple[str, ...], tuple[object, ...]]]:
        """Return reachable declared field interactions the design must cover."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_active_interactions_cache is not None
        return copy.deepcopy(self._flash_coverage_active_interactions_cache)

    def flash_structural_qualification_prefix_count(self) -> int:
        """Return rows reserved for two-witness ordinary-leaf qualification."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_qualification_prefix_count_cache is not None
        return self._flash_qualification_prefix_count_cache

    def flash_structural_parent_coverage_prefix_count(self) -> int:
        """Return leading rows that cover every live family and packet once."""
        self._flash_deterministic_coverage_flats()
        assert self._flash_parent_coverage_prefix_count_cache is not None
        return self._flash_parent_coverage_prefix_count_cache

    def flash_structural_starting_path_limit(
        self,
        *,
        minimum: int,
        retained_families: int | None,
        retained_candidates_per_leaf: int,
    ) -> int:
        """Size full-search continuation capacity from the live leaf catalog."""
        if not self.config_spec.cute_flash_search_enabled:
            return minimum
        leaves = self.flash_structural_leaf_catalog()
        ordinary_by_family: dict[str, int] = {}
        compound_count = 0
        for leaf in leaves:
            if leaf.compound_exp2_packet is None:
                ordinary_by_family[leaf.pipeline_family] = (
                    ordinary_by_family.get(leaf.pipeline_family, 0) + 1
                )
            else:
                compound_count += 1
        promoted_count = self._resolve_flash_structural_family_limit(
            retained_families, len(ordinary_by_family)
        )
        promoted_protocol_count = sum(
            sorted(ordinary_by_family.values(), reverse=True)[:promoted_count]
        )
        secondary_count = promoted_count if retained_candidates_per_leaf > 1 else 0
        # One unrestricted path, every ordinary protocol in the widest possible
        # promoted-family set, one secondary per promoted family, every compound
        # leaf. The global lane alternate occupies its family's secondary slot.
        required = 1 + promoted_protocol_count + secondary_count + compound_count
        return max(minimum, required)

    def flash_structural_family_probe_path_limit(
        self,
        retained_families: int | None,
        family_probe_generations: int,
    ) -> int:
        """Return the measured pre-promotion probe capacity for the live catalog."""
        if (
            not self.config_spec.cute_flash_search_enabled
            or family_probe_generations <= 0
        ):
            return 0
        leaves = self.flash_structural_leaf_catalog()
        ordinary_families = {
            leaf.pipeline_family for leaf in leaves if leaf.compound_exp2_packet is None
        }
        if retained_families is None or len(ordinary_families) <= retained_families:
            return 0
        compound_count = sum(leaf.compound_exp2_packet is not None for leaf in leaves)
        # One constrained path per ordinary family and compound leaf, plus one
        # unrestricted path that can discover a different structural basin.
        return len(ordinary_families) + compound_count + 1

    def flash_structural_effective_family_limit(
        self, retained_families: int | None
    ) -> int:
        """Resolve a configured family cap against the live ordinary catalog.

        ``None`` is the full-search policy: every ordinary family gets a
        conditional continuation. Explicit integer caps preserve the bounded
        policy used by custom effort profiles.
        """
        if not self.config_spec.cute_flash_search_enabled:
            return 0
        live_families = {
            leaf.pipeline_family
            for leaf in self.flash_structural_leaf_catalog()
            if leaf.compound_exp2_packet is None
        }
        return self._resolve_flash_structural_family_limit(
            retained_families, len(live_families)
        )

    @staticmethod
    def _resolve_flash_structural_family_limit(
        retained_families: int | None, live_family_count: int
    ) -> int:
        return (
            live_family_count
            if retained_families is None
            else min(retained_families, live_family_count)
        )

    def validate_flash_structural_coverage(self) -> None:
        """Reject an incomplete structural design before flash autotuning starts."""
        if not self.config_spec.cute_flash_search_enabled:
            return
        from .._compiler.cute.cute_flash import FLASH_DERIVED_CONFIG_KEYS
        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
        from .._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import flash_exp2_packet_is_compound

        structural_override_keys = {
            FLASH_PIPELINE_FAMILY_KEY,
            FLASH_PERSISTENT_KEY,
            *FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS,
            *FLASH_DERIVED_CONFIG_KEYS,
        }
        packet_override = self._override_values.get(FLASH_EXP2_PACKET_KEY)
        if self._override_values.keys() & structural_override_keys or (
            packet_override is not None
            and flash_exp2_packet_is_compound(packet_override)
        ):
            # Structural overrides can pin conditional parents and make
            # advertised children unreachable. Arithmetic overrides must not
            # suppress validation of the remaining flash search design.
            return
        self._flash_deterministic_coverage_flats()
        assert self._flash_coverage_uncovered_cache is not None
        assert self._flash_coverage_uncovered_interactions_cache is not None
        # A normalized family can legitimately have only one distinct effective
        # config. Keep two-witness shortfalls as strict-harness telemetry rather
        # than rejecting an otherwise complete ordinary search.
        problems: list[str] = []
        if self._flash_coverage_uncovered_cache:
            problems.append(
                f"uncovered values={self._flash_coverage_uncovered_cache!r}"
            )
        if self._flash_coverage_uncovered_interactions_cache:
            problems.append(
                "uncovered interactions="
                f"{self._flash_coverage_uncovered_interactions_cache!r}"
            )
        if problems:
            raise AutotuneError(
                "incomplete CuTe flash structural coverage design: "
                + "; ".join(problems)
            )

    def flash_structural_population_budget(self, population_size: int) -> int:
        """Return the deterministic-row budget for a flash population size."""
        half_population = population_size // 2
        coverage_count = len(self._flash_deterministic_coverage_flats())
        if coverage_count <= population_size:
            return max(half_population, coverage_count)
        qualification_prefix_count = self.flash_structural_qualification_prefix_count()
        if qualification_prefix_count <= half_population:
            return half_population
        return min(
            population_size,
            max(
                half_population,
                self.flash_structural_parent_coverage_prefix_count(),
            ),
        )

    @_flash_env_scoped
    def flash_deterministic_population_configs(self) -> list[Config]:
        """Return the full normalized structural-coverage design.

        Population construction always consumes its family/packet prefix, but
        may reserve later child-value rows for random exploration.
        """
        self.validate_flash_structural_coverage()
        result: list[Config] = []
        seen: set[Config] = set()
        for flat in self._flash_deterministic_coverage_flats():
            try:
                config = self.unflatten(flat)
            except InvalidConfig:
                continue
            if config in seen:
                continue
            seen.add(config)
            result.append(config)
        return result

    @_flash_env_scoped
    def flash_low_confound_schedule_anchor_configs(self) -> list[Config]:
        """Enumerate neutral anchors for every live flash schedule protocol.

        Marginal covering rows deliberately combine many axes, which makes them
        useful for reachability checks but poor optimization starting points.  A
        flash pipeline family, ordinary exp2 packet, and softmax protocol form
        the high-level schedule decision.  Enumerate that live product from the
        fragment default while leaving every other field neutral.  The anchors
        are still benchmarked and ranked; they contain no measured winner or
        sequence-length lookup.
        """
        if not self.config_spec.cute_flash_search_enabled:
            return []

        from .._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from .._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from .._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
        from .._compiler.cute.cute_flash import flash_exp2_packet_is_compound

        keys = (
            FLASH_PIPELINE_FAMILY_KEY,
            FLASH_EXP2_PACKET_KEY,
            FLASH_SOFTMAX_DISC_KEY,
        )
        axes: dict[str, tuple[int, tuple[object, ...]]] = {}
        for key in keys:
            layout = self._key_to_flat_indices.get(key)
            if layout is None or layout[1] or len(layout[0]) != 1:
                return []
            index = layout[0][0]
            fragment = self.flat_spec[index]
            if not isinstance(fragment, EnumFragment):
                return []
            choices = (
                (self._override_values[key],)
                if key in self._override_values
                else fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            )
            axes[key] = (index, tuple(choices))

        result: list[Config] = []
        seen: set[Config] = set()
        base = self._fragment_default_flat()
        for family, packet, softmax_disc in itertools.product(
            axes[FLASH_PIPELINE_FAMILY_KEY][1],
            axes[FLASH_EXP2_PACKET_KEY][1],
            axes[FLASH_SOFTMAX_DISC_KEY][1],
        ):
            if flash_exp2_packet_is_compound(packet):
                continue
            flat = copy.deepcopy(base)
            flat[axes[FLASH_PIPELINE_FAMILY_KEY][0]] = family
            flat[axes[FLASH_EXP2_PACKET_KEY][0]] = packet
            flat[axes[FLASH_SOFTMAX_DISC_KEY][0]] = softmax_disc
            try:
                config = self.unflatten(flat)
            except InvalidConfig:
                continue
            values = config.config
            if (
                values.get(FLASH_PIPELINE_FAMILY_KEY) != family
                or values.get(FLASH_EXP2_PACKET_KEY) != packet
                or values.get(FLASH_SOFTMAX_DISC_KEY) != softmax_disc
                or config in seen
            ):
                continue
            seen.add(config)
            result.append(config)
        return result

    @_flash_env_scoped
    def flash_exact_effective_search_space_configs(
        self, max_raw_configs: int
    ) -> list[Config] | None:
        """Enumerate a small CuTe-flash space after config normalization.

        The global flat surface can contain aliases that normalize to the same
        conditional schedule.  When its raw Cartesian product is bounded, this
        returns every distinct normalized config in stable product order.  A
        larger or non-enumerable space returns ``None`` without partial work.
        """
        if max_raw_configs < 1:
            raise ValueError("max_raw_configs must be positive")
        if not self.config_spec.cute_flash_search_enabled:
            return None

        value_sets: list[list[object]] = []
        overridden = self.overridden_flat_indices
        frozen_flat: FlatConfig | None = None
        if overridden:
            # Extract only the frozen slots. Normalizing the complete default
            # point can fail even when the override has legal combinations in
            # the remaining dimensions (for example a high register allocation
            # paired with a smaller non-overridden allocation).
            from ..runtime.config import Config

            frozen_flat = self.flatten(Config.from_dict(self._override_values))
        raw_size = 1
        for index, fragment in enumerate(self.flat_spec):
            if index in overridden:
                assert frozen_flat is not None
                value_sets.append([frozen_flat[index]])
                continue
            cardinality = fragment.cardinality()
            if (
                cardinality is None
                or cardinality < 1
                or raw_size > max_raw_configs // cardinality
            ):
                return None
            values = fragment.search_values(max_raw_configs)
            if values is None or len(values) != cardinality:
                return None
            raw_size *= cardinality
            value_sets.append(values)

        result: list[Config] = []
        seen: set[Config] = set()
        for values in itertools.product(*value_sets):
            try:
                config = self.unflatten(list(values))
            except InvalidConfig:
                continue
            if config in seen:
                continue
            seen.add(config)
            result.append(config)
        return result

    def seed_flat_config_pairs(
        self,
        log_func: Callable[[str], None] | None = None,
    ) -> list[tuple[FlatConfig, Config]]:
        """Return ConfigSpec-provided seeds as flat and normalized configs.

        ``ConfigSpec.compiler_seed_configs`` is compiler-owned and must
        contain configs that match the live spec structurally. Invalid seeds
        are skipped with the same transfer policy as user-provided seed configs.
        """
        result: list[tuple[FlatConfig, Config]] = []
        seen: set[Config] = set()
        for i, config in enumerate(self.config_spec.compiler_seed_configs):
            try:
                flat, normalized = self.canonicalize_flat(self.flatten(config))
            except (
                InvalidConfig,
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
            ) as e:
                if log_func is not None:
                    log_func(f"Failed to transfer compiler seed config {i + 1}: {e}")
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append((flat, normalized))
        return result

    def user_seed_flat_config_pairs(
        self,
        user_seed_configs: Sequence[Config],
        log_func: Callable[[str], None] | None = None,
    ) -> list[tuple[FlatConfig, Config]]:
        """Return user-provided seed configs as flat and normalized configs."""
        result: list[tuple[FlatConfig, Config]] = []
        seen: set[Config] = set()
        for i, config in enumerate(user_seed_configs):
            try:
                flat, normalized = self.canonicalize_flat(self.flatten(config))
            except (
                InvalidConfig,
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
            ) as e:
                if log_func is not None:
                    log_func(f"Failed to transfer autotune seed config {i + 1}: {e}")
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append((flat, normalized))
        return result

    def random_flat(self) -> FlatConfig:
        """
        Generate a random flat configuration.

        Returns:
            A random flat configuration.
        """

        with sync_seed(process_group_name=self.process_group_name):
            config = [spec.random() for spec in self.flat_spec]
            self.shrink_config(config, PowerOfTwoFragment(1, 2048, 32).random())
            self._repair_cute_num_threads(config)
            return config

    @functools.cached_property
    def _config_value_priors(self) -> dict[str, ValuePrior]:
        """Per-config-key sampling priors supplied by the active backend."""
        return dict(self.config_spec.backend.config_value_priors(self.config_spec))

    @functools.cached_property
    def _flat_index_to_key_pos(self) -> dict[int, tuple[str, int]]:
        """Map each flat_spec index to its ``(config key, position-in-key)``."""
        mapping: dict[int, tuple[str, int]] = {}
        for key, (indices, _is_sequence) in self._key_to_flat_indices.items():
            for position, flat_idx in enumerate(indices):
                mapping[flat_idx] = (key, position)
        return mapping

    def biased_random_flat(self) -> FlatConfig:
        """Random flat config biased by the backend's per-key value priors.

        Identical to :meth:`random_flat` except that, for each flat slot whose
        config key has a registered prior, the value is drawn from that prior
        (falling back to the fragment's uniform ``random()`` when the prior
        declines). Used for half of the random portion of the initial
        population; with no priors this is exactly ``random_flat``.
        """
        priors = self._config_value_priors
        if not priors:
            return self.random_flat()
        index_to_key_pos = self._flat_index_to_key_pos
        with sync_seed(process_group_name=self.process_group_name):
            config: FlatConfig = []
            for i, spec in enumerate(self.flat_spec):
                key_pos = index_to_key_pos.get(i)
                prior = priors.get(key_pos[0]) if key_pos is not None else None
                if prior is None or key_pos is None:
                    config.append(spec.random())
                elif isinstance(spec, ListOf):
                    # A list-valued key (e.g. ``indexing``) occupies one flat slot
                    # holding a list; bias each element via the inner fragment.
                    config.append(
                        [
                            _value_or(prior(spec.inner, j), spec.inner.random)
                            for j in range(spec.length)
                        ]
                    )
                else:
                    config.append(_value_or(prior(spec, key_pos[1]), spec.random))
            self.shrink_config(config, PowerOfTwoFragment(1, 2048, 32).random())
            self._repair_cute_num_threads(config)
            return config

    def random_config(self) -> Config:
        errors: dict[str, int] = {}
        for _ in range(64):
            try:
                return self.unflatten(self.random_flat())
            except InvalidConfig as e:
                msg = str(e)
                errors[msg] = errors.get(msg, 0) + 1
                self.invalid_config_count += 1
                continue
        summary = "; ".join(f"{msg} (x{n})" for msg, n in errors.items())
        raise InvalidConfig(
            f"failed to generate a valid random config after 64 attempts: {summary}"
        )

    @_flash_env_scoped
    def random_population_flat(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config] = (),
        log_func: Callable[[str], None] | None = None,
    ) -> list[FlatConfig]:
        if n <= 0:
            return [self.default_flat()]
        default_flat = self.default_flat()

        if not self.config_spec.cute_flash_search_enabled:
            result = [default_flat]
            for flat, _config in self.user_seed_flat_config_pairs(
                user_seed_configs, log_func
            ):
                if any(flat == existing for existing in result):
                    continue
                result.append(flat)
                if len(result) >= n:
                    return result[:n]
            for flat, _config in self.seed_flat_config_pairs(log_func):
                if any(flat == existing for existing in result):
                    continue
                result.append(flat)
                if len(result) >= n:
                    return result[:n]
            priors_present = bool(self._config_value_priors)
            for j in range(n - len(result)):
                result.append(
                    self.biased_random_flat()
                    if priors_present and j % 2 == 0
                    else self.random_flat()
                )
            return result

        self.validate_flash_structural_coverage()
        result: list[FlatConfig] = []
        seen: set[Config] = set()

        def append_valid(config: Config, *, required: bool = False) -> bool:
            if config in seen or (not required and len(result) >= n):
                return False
            seen.add(config)
            result.append(self.flatten(config))
            return True

        def append_if_valid(flat: FlatConfig, *, required: bool = False) -> bool:
            try:
                config = self.unflatten(flat)
            except InvalidConfig:
                return False
            return append_valid(config, required=required)

        # Flash schedules are compound categorical choices. The design first
        # reaches every exact ordinary-family or compound-packet leaf, then adds
        # a second ordinary witness before child coverage. Use half of the
        # population when qualification fits. If the complete compact design fits,
        # include all of it; a smaller population may exceed half only to preserve
        # one-witness parent coverage.
        coverage_flats = self._flash_deterministic_coverage_flats()
        qualification_prefix_count = self.flash_structural_qualification_prefix_count()
        coverage_budget = min(n, self.flash_structural_population_budget(n))
        coverage_added = 0
        reserved_prefix_count = min(qualification_prefix_count, coverage_budget)
        for flat in coverage_flats[:reserved_prefix_count]:
            try:
                config = self.unflatten(flat)
            except InvalidConfig:
                continue
            if append_valid(config, required=True):
                coverage_added += 1

        # Explicit user seeds are retained even when they make the population
        # larger than the nominal target. Compiler seeds are heuristic hints,
        # so admit only those that fit; exact structural coverage is inserted
        # first and cannot be crowded out by a large same-family seed set.
        for _flat, config in self.user_seed_flat_config_pairs(
            user_seed_configs, log_func
        ):
            append_valid(config, required=True)

        for _flat, config in self.seed_flat_config_pairs(log_func):
            append_valid(config)

        if coverage_added < coverage_budget:
            for flat in coverage_flats[reserved_prefix_count:]:
                if coverage_added >= coverage_budget:
                    break
                try:
                    config = self.unflatten(flat)
                except InvalidConfig:
                    continue
                if append_valid(config, required=True):
                    coverage_added += 1

        exact_space = self.flash_exact_effective_search_space_configs(n)
        if exact_space is not None:
            for config in exact_space:
                # Exhaustive rows cannot be crowded out by mandatory seeds.
                append_valid(config, required=True)

        append_if_valid(default_flat)

        # Fill the remainder with random configs. When the backend supplies
        # value priors, half the random fill is drawn from those priors (biased
        # toward the region good configs occupy) and half stays uniform so the
        # search keeps full coverage; with no priors every fill is uniform,
        # leaving the historical behavior unchanged.
        priors_present = bool(self._config_value_priors)
        invalid = 0
        duplicate = 0
        attempts = 0
        max_attempts = max(64, (n - len(result)) * 64)
        while exact_space is None and len(result) < n and attempts < max_attempts:
            j = attempts
            attempts += 1
            if priors_present and j % 2 == 0:
                flat = self.biased_random_flat()
            else:
                flat = self.random_flat()
            try:
                config = self.unflatten(flat)
            except InvalidConfig:
                invalid += 1
                continue
            if config in seen:
                duplicate += 1
                continue
            seen.add(config)
            result.append(self.flatten(config))
        if len(result) < n and log_func is not None:
            if exact_space is not None:
                log_func(
                    "Exhausted the exact CuTe-flash effective search space at "
                    f"{len(exact_space)} normalized configs; padding the "
                    f"{n}-row initial population with duplicates."
                )
            else:
                log_func(
                    "Generated only "
                    f"{len(result)}/{n} valid initial population configs "
                    f"after {attempts} random attempts "
                    f"({invalid} invalid, {duplicate} duplicate); "
                    "padding with duplicate valid configs."
                )
        if len(result) < n:
            if not result:
                raise InvalidConfig(
                    "failed to generate any valid initial population configs"
                )
            pad_from = [copy.deepcopy(flat) for flat in result]
            pad_idx = 0
            while len(result) < n:
                result.append(copy.deepcopy(pad_from[pad_idx % len(pad_from)]))
                pad_idx += 1
        return result

    def random_population(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config] = (),
        log_func: Callable[[str], None] | None = None,
    ) -> list[Config]:
        result: list[Config] = []
        attempts = 0
        if not self.config_spec.cute_flash_search_enabled:
            flat_population = self.random_population_flat(
                n, user_seed_configs=user_seed_configs, log_func=log_func
            )
        else:
            try:
                flat_population = self.random_population_flat(
                    n, user_seed_configs=user_seed_configs, log_func=log_func
                )
            except InvalidConfig:
                flat_population = []
        for flat in flat_population:
            try:
                result.append(self.unflatten(flat))
            except InvalidConfig:
                attempts += 1
                self.invalid_config_count += 1
        # Retry to fill the population to the requested size
        while len(result) < n and attempts < 64:
            try:
                result.append(self.unflatten(self.random_flat()))
            except InvalidConfig:
                self.invalid_config_count += 1
            attempts += 1
        return result

    def differential_mutation(
        self,
        x: FlatConfig,
        a: FlatConfig,
        b: FlatConfig,
        c: FlatConfig,
        crossover_rate: float,
    ) -> FlatConfig:
        """
        The main op in differential evolution, randomly combine `x` with `a + (b - c)`.
        """
        overridden = self.overridden_flat_indices
        result = [*x]
        mutated = False
        for i, spec in enumerate(self.flat_spec):
            if i not in overridden and random.random() < crossover_rate:
                result[i] = spec.differential_mutation(a[i], b[i], c[i])
                mutated = True
        if not mutated:
            eligible = [i for i in range(len(self.flat_spec)) if i not in overridden]
            if eligible:
                i = random.choice(eligible)
                result[i] = self.flat_spec[i].differential_mutation(a[i], b[i], c[i])
        # TODO(jansel): can this be larger? (too large and Triton compile times blow up)
        self.shrink_config(result, 8192)
        self._repair_cute_num_threads(result)
        return result

    def encode_config(self, flat_config: FlatConfig) -> list[float]:
        """
        Encode a flat configuration into a numerical vector for ML models.

        This is used by surrogate-assisted algorithms (e.g., DE-Surrogate) that need
        to represent configurations as continuous vectors for prediction models.

        Args:
            flat_config: The flat configuration values to encode.

        Returns:
            A list of floats representing the encoded configuration.
        """
        encoded: list[float] = []

        for flat_idx, spec in enumerate(self.flat_spec):
            value = flat_config[flat_idx]
            encoded_value = spec.encode(value)
            assert len(encoded_value) == spec.dim()
            encoded.extend(encoded_value)

        return encoded
