from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
import operator
import random
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import cast

from .._compat import warps_to_threads
from ..exc import AutotuneError
from ..exc import InvalidConfig
from .block_id_sequence import BlockIdSequence
from .compiler_coverage import same_value
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
    from . import ConfigSpec
    from .config_priors import ValuePrior
    from .search_surface import ConfigGenerationExtension

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
        compiler_coverage_enabled: bool = True,
        _field_view: Mapping[str, BlockIdSequence[Any] | ConfigSpecFragment]
        | None = None,
        _initial_sampling: bool = False,
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
        self._field_view = _field_view
        self._initial_sampling = _initial_sampling
        self.compiler_coverage_enabled = compiler_coverage_enabled
        self.process_group_name = process_group_name
        self._advanced_controls_files = advanced_controls_files
        self._flash_pipeline_family_override = (
            _flash_pipeline_family_override
            if config_spec.cute_flash_search_enabled
            else None
        )
        self._override_values = dict(overrides or {})
        # The backend's hooks into this generation (``search_surface.py``).
        self.extension: ConfigGenerationExtension = (
            config_spec.search_surface.config_generation_extension(self)
        )
        self.extension.narrow_search_choices()
        self.flat_spec: list[ConfigSpecFragment] = []
        if self._field_view is not None:
            config_spec._flat_config_from_fields(
                _collect_spec,
                self._field_view,
                advanced_controls_files=advanced_controls_files,
            )
        elif self._flash_pipeline_family_override is None:
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
        self.block_size_indices: list[int] = [
            i
            for i, spec in enumerate(self.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        ]
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

    @functools.cached_property
    def overridden_flat_indices(self) -> set[int]:
        """Return flat_spec indices that are frozen by config overrides."""
        result: set[int] = set()
        for key in self._override_values:
            if key in self._key_to_flat_indices:
                indices, _ = self._key_to_flat_indices[key]
                result.update(indices)
        if not self.compiler_coverage_enabled:
            for group in self.config_spec.compiler_coverage_groups:
                if group.key in self._key_to_flat_indices:
                    result.update(self._key_to_flat_indices[group.key][0])
        return result

    def _flat_fields(self) -> Mapping[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        if self._field_view is not None:
            return self._field_view
        if self._flash_pipeline_family_override is None:
            return self.config_spec._flat_fields()
        return self.config_spec._flat_fields_with_flash_family(
            self._flash_pipeline_family_override
        )

    def initial_population_view(self) -> ConfigGeneration:
        """Own a full-layout view whose random draws use only old coordinates."""
        if not self.config_spec.compiler_coverage_groups:
            return self
        self.config_spec.validate_compiler_coverage_groups()
        return ConfigGeneration(
            self.config_spec,
            overrides=copy.deepcopy(self._override_values),
            _flash_pipeline_family_override=self._flash_pipeline_family_override,
            advanced_controls_files=copy.deepcopy(self._advanced_controls_files),
            process_group_name=self.process_group_name,
            compiler_coverage_enabled=self.compiler_coverage_enabled,
            _field_view=MappingProxyType(copy.deepcopy(dict(self._flat_fields()))),
            _initial_sampling=True,
        )

    @functools.cached_property
    def _compiler_coverage_sampler(self) -> ConfigGeneration | None:
        if not self._initial_sampling and self.compiler_coverage_enabled:
            return None
        owned = {group.key for group in self.config_spec.compiler_coverage_groups}
        fields = self._flat_fields()
        if not owned.intersection(fields):
            return None
        return ConfigGeneration(
            self.config_spec,
            overrides=copy.deepcopy(
                {
                    key: value
                    for key, value in self._override_values.items()
                    if key not in owned
                }
            ),
            _flash_pipeline_family_override=self._flash_pipeline_family_override,
            advanced_controls_files=copy.deepcopy(self._advanced_controls_files),
            process_group_name=self.process_group_name,
            _field_view=MappingProxyType(
                copy.deepcopy(
                    {key: value for key, value in fields.items() if key not in owned}
                )
            ),
        )

    def _lift_projected_raw(self, flat: FlatConfig) -> FlatConfig:
        """Transfer raw slots by name without normalization, repair or RNG draws."""
        sampler = self._compiler_coverage_sampler
        assert sampler is not None
        if len(flat) != len(sampler.flat_spec):
            raise ValueError("Cached flat config does not match the projected layout")
        legacy = {
            group.key: group.legacy
            for group in self.config_spec.compiler_coverage_groups
        }
        result: FlatConfig = [None] * len(self.flat_spec)
        for key, (indices, is_sequence) in self._key_to_flat_indices.items():
            if key in legacy:
                assert len(indices) == 1 and not is_sequence
                result[indices[0]] = legacy[key]
                continue
            old_indices, old_sequence = sampler._key_to_flat_indices[key]
            if old_sequence != is_sequence or len(old_indices) != len(indices):
                raise ValueError("Compiler coverage changed old flat slots")
            for index, old_index in zip(indices, old_indices, strict=True):
                result[index] = copy.deepcopy(flat[old_index])
        return result

    def projected_cache_flat_pair(self, flat: FlatConfig) -> tuple[FlatConfig, Config]:
        sampler = self._compiler_coverage_sampler
        assert self._initial_sampling and sampler is not None
        old_flat = copy.deepcopy(flat)
        # Decode only with the old layout. Preserve the same raw slots the old
        # best-available builder retains after its normal unflatten call.
        sampler.unflatten(old_flat)
        lifted = self._lift_projected_raw(old_flat)
        return lifted, self.unflatten(lifted)

    @functools.cached_property
    def _key_to_flat_indices(self) -> dict[str, tuple[list[int], bool]]:
        """Build mapping from config key names to (flat_spec indices, is_sequence).

        Derived from ConfigSpec.flat_key_layout().
        """
        mapping: dict[str, tuple[list[int], bool]] = {}
        idx = 0
        if self._field_view is not None:
            layout = [
                (key, *field._flat_key_info())
                for key, field in self._field_view.items()
            ]
            if (
                self.config_spec._advanced_controls_file_fragment(
                    self._advanced_controls_files
                )
                is not None
            ):
                layout.append(("advanced_controls_file", 1, False))
        elif self._flash_pipeline_family_override is None:
            layout = self.config_spec.flat_key_layout(
                advanced_controls_files=self._advanced_controls_files
            )
        else:
            layout = self.config_spec._flat_key_layout_with_flash_family(
                advanced_controls_files=self._advanced_controls_files,
                flash_pipeline_family=self._flash_pipeline_family_override,
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

    def flatten(self, config: Config) -> FlatConfig:
        """Inverse of unflatten: convert a Config to a FlatConfig."""
        result = self._fragment_default_flat()
        flat_fields = self._flat_fields()
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
        self.extension.repair_flat_config(result)
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

    def coordinate_neighbor_projections(
        self,
        base: FlatConfig,
        *,
        radius: int = 1,
        limit: int | None = None,
        frozen_indices: Sequence[int] = (),
    ) -> list[CoordinateNeighborProjection]:
        """Enumerate normalized one-coordinate pattern neighbors in order.

        The returned order follows the flat ConfigSpec layout and each fragment's
        own deterministic ``pattern_neighbors`` order. Invalid and normalized
        alias requests remain in the result so callers can audit completeness.
        An optional limit bounds raw requests, including invalid and alias requests.
        Explicitly frozen coordinates are excluded before generating requests.
        """
        canonical_base, base_config = self.canonicalize_flat(base)
        flat_identities = self._flat_coordinate_identities()

        seen = {base_config}
        result: list[CoordinateNeighborProjection] = []
        overridden = self.overridden_flat_indices | set(frozen_indices)
        for flat_index, spec in enumerate(self.flat_spec):
            if flat_index in overridden:
                continue
            key, sequence_index = flat_identities[flat_index]
            current = canonical_base[flat_index]
            for value in spec.pattern_neighbors(current, radius):
                if limit is not None and len(result) >= limit:
                    return result
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

        return self._unflatten(flat_values)

    def _unflatten(
        self,
        flat_values: FlatConfig,
        *,
        fix_invalid: bool = True,
        repair: bool = True,
    ) -> Config:
        def get_next_value(spec: ConfigSpecFragment) -> object:
            i = next(count)
            assert type(self.flat_spec[i]) is type(spec)
            return flat_values[i]

        assert len(flat_values) == len(self.flat_spec)
        if repair:
            self.extension.repair_flat_config(flat_values)
        count: itertools.count[int] = itertools.count()
        if self._field_view is not None or not fix_invalid:
            config = self.config_spec._flat_config_from_fields(
                get_next_value,
                self._flat_fields(),
                advanced_controls_files=self._advanced_controls_files,
                _fix_invalid=fix_invalid,
            )
        elif self._flash_pipeline_family_override is None:
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

    def strict_config_pair(self, config: Config) -> tuple[FlatConfig, Config]:
        """Validate a complete effective coverage config before repair can hide it."""
        prepared = self._apply_overrides(copy.deepcopy(config))
        self.config_spec.normalize(prepared.config)
        for group in self.config_spec.compiler_coverage_groups:
            value = prepared.config.get(group.key, group.legacy)
            if not any(same_value(value, mode) for mode in group.domain):
                raise InvalidConfig(
                    f"Invalid compiler coverage mode {group.key}={value!r}"
                )
        flat = self.flatten(prepared)
        strict = self._unflatten(copy.deepcopy(flat), fix_invalid=False, repair=False)
        for key, value in prepared.config.items():
            if key not in strict.config or strict.config[key] != value:
                raise InvalidConfig(f"Coverage transfer changed supplied field {key!r}")
        normalized_flat, normalized = self.canonicalize_flat(flat)
        if strict != normalized:
            raise InvalidConfig("Coverage transfer requires config repair")
        self.config_spec.normalize(normalized.config)
        return normalized_flat, normalized

    def strict_unflatten(self, flat: FlatConfig) -> Config:
        """Decode an expanded warm witness without first repairing its mode."""
        return self._unflatten(copy.deepcopy(flat), fix_invalid=False, repair=False)

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
        self.extension.repair_flat_config(flat_config)

    def _fragment_default_flat(self) -> FlatConfig:
        """
        Retrieve the default flat configuration from raw fragment defaults.

        Returns:
            The default flat configuration values.
        """
        config = [spec.default() for spec in self.flat_spec]
        self._shrink_for_numel_constraints(config)
        self.extension.repair_flat_config(config)
        return config

    def default_flat(self) -> FlatConfig:
        """
        Retrieve the conservative autotuning reference configuration.

        Returns:
            The default flat configuration values.
        """
        return self._fragment_default_flat()

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
                flat, normalized = self.canonicalize_flat(
                    self.flatten(self._normalize_seed_per_loop_lists(config))
                )
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

    def _normalize_seed_per_loop_lists(self, config: Config) -> Config:
        """Fit a seed's per-loop lists to this spec, as ``ConfigSpec.normalize`` does.

        Seeds can come from a heuristic file written on another GPU or by an
        older Helion, where a per-loop list such as ``range_num_stages`` has
        fewer (often zero) entries.  ``flatten`` requires exact lengths, so
        missing entries are filled with defaults and lists that cannot fit
        raise ``InvalidConfig`` with the reason.
        """
        from ..runtime.config import Config

        values = dict(config.config)
        flat_fields = self._flat_fields()
        for key, (_indices, is_sequence) in self._key_to_flat_indices.items():
            if not is_sequence or key not in values:
                continue
            field = flat_fields[key]
            assert isinstance(field, BlockIdSequence)
            value = values[key]
            if not isinstance(value, list) or len(value) != len(field):
                # Only loop_orders holds nested lists, as in ConfigSpec.normalize.
                values[key] = field._normalize(key, value, flatten=key != "loop_orders")
        return Config.from_dict(values)

    def random_flat(self) -> FlatConfig:
        """
        Generate a random flat configuration.

        Returns:
            A random flat configuration.
        """

        sampler = self._compiler_coverage_sampler
        if sampler is not None:
            return self._lift_projected_raw(sampler.random_flat())
        with sync_seed(process_group_name=self.process_group_name):
            config = [spec.random() for spec in self.flat_spec]
            self.shrink_config(config, PowerOfTwoFragment(1, 2048, 32).random())
            self.extension.repair_flat_config(config)
            return config

    @functools.cached_property
    def _config_value_priors(self) -> dict[str, ValuePrior]:
        """Per-config-key sampling priors supplied by the active backend."""
        sampler = self._compiler_coverage_sampler
        if sampler is not None:
            return sampler._config_value_priors
        result = dict(self.config_spec.backend.config_value_priors(self.config_spec))
        if self._field_view is not None:
            for group in self.config_spec.compiler_coverage_groups:
                if group.key not in self._field_view:
                    result.pop(group.key, None)
        return result

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
        sampler = self._compiler_coverage_sampler
        if sampler is not None:
            return self._lift_projected_raw(sampler.biased_random_flat())
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
            self.extension.repair_flat_config(config)
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

    def random_population_flat(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config] = (),
        log_func: Callable[[str], None] | None = None,
    ) -> list[FlatConfig]:
        if n <= 0:
            return [self.default_flat()]
        structural = self.extension.initial_population_flats(
            n, user_seed_configs=user_seed_configs, log_func=log_func
        )
        if structural is not None:
            return structural
        result = [self.default_flat()]
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
        self.extension.repair_flat_config(result)
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
