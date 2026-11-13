"""The backend extension points of the autotuner: a backend's search surface.

A backend contributes config keys, flat fields, normalization phases and
compiler seeds to the backend-neutral ``ConfigSpec`` through one stateless
object returned by ``Backend.config_spec_extension()``.  Every method of
:class:`BackendSearchSurface` is the no-op default (Triton, Pallas and Metal
add nothing beyond their ``tunable_fragments``); the CuTe implementation lives
in ``helion/_compiler/cute/autotune/search_surface.py``.

The surface also hands every ``ConfigGeneration`` one
:class:`ConfigGenerationExtension` (``config_generation_extension``, stored
as ``generation.extension``): the per-generation hooks ``ConfigGeneration``
calls while narrowing its choices, repairing flat configs and seeding the
initial population, and the population-shaping hooks ``PatternSearch`` and
``PopulationBasedSearch`` read; the base class is the no-op default and the
CuTe implementation lives in
``helion/_compiler/cute/autotune/config_generation_ext.py``.

``ConfigSpec`` calls the hooks at fixed positions of its own pipeline, the
positions the inlined CuTe branches held before the extraction.  The flat-field
order and the ``normalize`` phase order are part of the autotune cache identity
and must not move.  Normalization runs the phases in this order::

    normalize_entry              before any generic step
    normalize_family_gates       after the generic cross_loop_pipeline gate
    normalize_knobs              after the unsupported-key check
    normalize_before_sequences   after ``provided_keys`` is captured
    normalize_after_sequences    after the block-id sequences are normalized
    reduction_loop_chunk_cap     inside the generic reduction-loop cap
    normalize_reduction_threads  after the generic reduction-loop cap
    normalize_after_defaults     after the generic defaults and the AMD MFMA step
    normalize_recurrence_families after the cross_loop_pipeline fragment
    normalize_pre_pid_type       before the Pallas / pid_type validation
    normalize_strategy           after the xcd_remap validation
    normalize_finalize           before the unknown-key check
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .block_id_sequence import BlockIdSequence
    from .config_fragment import ConfigSpecFragment
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .config_spec import ConfigSpec
    from .config_spec import ReductionLoopSpec


class BackendSearchSurface:
    """A backend's contribution to the autotuner search surface (no-op default).

    ``config_keys`` are the backend-specific config keys the backend accepts
    (searched or normalized); ``debug_keys`` are accepted by ``normalize`` but
    never searched, fingerprinted or seeded; ``implicit_default_keys`` are the
    keys ``normalize`` drops again when a config did not provide them.
    """

    config_keys: frozenset[str] = frozenset()
    debug_keys: frozenset[str] = frozenset()
    implicit_default_keys: frozenset[str] = frozenset()
    # Scalar compiler and schedule choices worth crossing with discovered layouts
    # before full-effort final selection. Empty for backends without this phase.
    compiler_refinement_keys: tuple[str, ...] = ()

    def supports_key(self, key: str) -> bool:
        """Whether this surface accepts a backend-specific config key."""
        return key in self.config_keys or key in self.debug_keys

    def explain_unavailable_key(self, key: str, config_spec: ConfigSpec) -> str | None:
        """Why the config key ``key`` is unavailable to ``config_spec``'s
        search (a human-readable reason for the search-space log), or ``None``
        when this surface has nothing to say about it."""
        return None

    def eviction_policy_search_choices(
        self, config_spec: ConfigSpec, choices: Sequence[str]
    ) -> tuple[str, ...] | None:
        """The subset of the valid load eviction policies ``choices`` worth
        searching on ``config_spec``'s target (explicit configs may still use
        any of them), or ``None`` to search them all."""
        return None

    def default_target_device_capability(self) -> tuple[int, int] | None:
        """The capability a spec built without one validates against."""
        return None

    def fields(
        self, config_spec: ConfigSpec, *, flash_pipeline_family: str | None = None
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment] | None:
        """The complete ordered flat-field table, or None for the generic table."""
        return None

    def normalize_entry(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_family_gates(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_knobs(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_before_sequences(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_after_sequences(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def reduction_loop_chunk_cap(
        self, config_spec: ConfigSpec, max_chunk: int, block_threshold: int
    ) -> int:
        """The chunk a too-large rolled reduction loop is capped to."""
        return min(max_chunk, block_threshold)

    def normalize_reduction_threads(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        register_tiles: bool,
    ) -> None:
        return

    def normalize_after_defaults(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_recurrence_families(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_pre_pid_type(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_strategy(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        return

    def normalize_finalize(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        provided_keys: set[str],
    ) -> None:
        return

    def seed_configs(self, config_spec: ConfigSpec) -> list[Config]:
        """The ordered compiler seeds of the spec's enabled families."""
        return []

    def prepare_config_generation_overrides(
        self, config_spec: ConfigSpec, overrides: Mapping[str, object] | None
    ) -> tuple[dict[str, object] | None, str | None]:
        """Complete fixed overrides; returns them with the family they pin."""
        return (dict(overrides) if overrides else None, None)

    def flatten_missing_field_default(
        self, config_spec: ConfigSpec, key: str, config: dict[str, object]
    ) -> tuple[bool, object]:
        """``(True, value)`` when a field absent from ``config`` flattens to ``value``."""
        return False, None

    def prepare_override_normalization(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        return

    def trim_reference_config(self, config_spec: ConfigSpec, config: Config) -> None:
        """Drop the optional coordinates the public default config leaves implicit."""
        return

    def trim_default_config(self, config_spec: ConfigSpec, config: Config) -> None:
        """Drop the inert knobs of a normalized promoted default."""
        return

    def prepare_flat_config(
        self, config_spec: ConfigSpec, config: dict[str, object]
    ) -> None:
        """Repair an unflattened config before its strict normalization."""
        return

    def fingerprint_extra(self, config_spec: ConfigSpec) -> object | None:
        """Extra identity folded into the persistent autotune cache key."""
        return None

    def l2_evict_last_store_policy_supported(self, config_spec: ConfigSpec) -> bool:
        """Whether the fixed L2 evict-last store policy descriptor is valid."""
        return False

    def new_spec_state(self, config_spec: ConfigSpec) -> object | None:
        """Per-spec state built in ``ConfigSpec.__init__`` (CuTe: its tcgen05
        GEMM facade, read through ``ConfigSpec.tcgen05``)."""
        return None

    def pins_thread_knobs(self, config_spec: ConfigSpec, block_id: int) -> bool:
        """Whether the search pins the per-block thread knobs of ``block_id``
        (``num_threads``, the lane layout and the vector width) to their inert
        values; CuTe does so for the tensor-core blocks of a tcgen05 search
        whose proved pointwise region owns the other blocks."""
        return False

    def eviction_policies(self) -> tuple[str, ...]:
        """The ``load_eviction_policies`` values this backend's loads accept
        (``""`` is the unhinted load every backend has)."""
        return ("",)

    def reduction_loop_limits(
        self,
        config_spec: ConfigSpec,
        block_id: int,
        threshold: int | None,
        max_chunk: int | None,
    ) -> tuple[int | None, int | None]:
        """The persistent thread threshold and rolled chunk limit of the
        reduction loop over ``block_id``, given the spec-wide defaults."""
        return threshold, max_chunk

    def explicit_reduction_chunk(
        self, config_spec: ConfigSpec, loop: ReductionLoopSpec, value: object
    ) -> object:
        """The ``reduction_loops`` entry ``value`` encodes to for ``loop``
        before flattening (``None`` is the persistent reduction); the default
        keeps the value as given."""
        return value

    def family_fragments(
        self, config_spec: ConfigSpec, family: str
    ) -> Mapping[str, ConfigSpecFragment]:
        """The searched fragments of the structural family ``family`` (the
        flash pipeline families; empty for a backend without any)."""
        return {}

    def config_generation_extension(
        self, generation: ConfigGeneration
    ) -> ConfigGenerationExtension:
        """The backend's hooks into ``generation`` (one object per generation,
        stored as ``ConfigGeneration.extension``)."""
        return ConfigGenerationExtension(generation)


class ConfigGenerationExtension:
    """A backend's contribution to one :class:`ConfigGeneration` (no-op default).

    ``ConfigGeneration`` calls the hooks at fixed positions of its own
    pipeline: ``narrow_search_choices`` before the flat field table is
    collected, ``repair_flat_config`` on every flat config it produces
    (defaults, random draws, mutations, ``flatten``),
    ``conditional_neighbor_updates`` when constructing search neighbors, and
    ``initial_population_flats`` first thing in ``random_population_flat``.
    The structural-coverage hooks size and seed the population-based searches
    (``PatternSearch``, ``PopulationBasedSearch``); the defaults describe a
    surface without a structural design.  The CuTe implementation (SIMT
    launch-thread repairs, the flash structural coverage design) lives in
    ``helion/_compiler/cute/autotune/config_generation_ext.py``.
    """

    def __init__(self, generation: ConfigGeneration) -> None:
        self.generation = generation

    def narrow_search_choices(self) -> None:
        """Narrow the generation's field view before its flat table is built."""
        return

    def repair_flat_config(self, flat_config: FlatConfig) -> None:
        """Repair backend-coupled coordinates of a flat config in place."""
        return

    def conditional_neighbor_updates(
        self, base: FlatConfig, requested: FlatConfig
    ) -> Mapping[int, object]:
        """Deterministic prerequisite flat values for a requested search transition.

        This hook only affects neighbors, not explicit config normalization or
        identity. ConfigGeneration applies the updates atomically when all
        prerequisites are searchable and no frozen coordinate would change.
        """
        return {}

    def initial_population_flats(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config],
        log_func: Callable[[str], None] | None,
    ) -> list[FlatConfig] | None:
        """A structurally designed initial population of ``n`` flat configs,
        or ``None`` for the generic seeds-then-random population."""
        return None

    def population_floor(self, n: int) -> int:
        """The smallest population that measures every structural leaf."""
        return n

    def deterministic_population_configs(self) -> list[Config]:
        """The normalized rows of the structural coverage design."""
        return []

    def structural_population_budget(self, population_size: int) -> int:
        """The deterministic-row budget of a population of ``population_size``."""
        return 0

    def structural_qualification_prefix_count(self) -> int:
        """Leading design rows reserved for structural qualification."""
        return 0

    def exact_effective_search_space_configs(
        self, max_raw_configs: int
    ) -> list[Config] | None:
        """Every distinct normalized config when the raw space is at most
        ``max_raw_configs`` points, else ``None``."""
        return None


# The surface of every backend without an extension of its own; stateless, so
# ``Backend.config_spec_extension`` hands out this one object.
EMPTY_SEARCH_SURFACE = BackendSearchSurface()
