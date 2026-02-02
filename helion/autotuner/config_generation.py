from __future__ import annotations

import copy
import functools
import itertools
import operator
import random
from typing import TYPE_CHECKING
from typing import cast

from .._compat import supports_maxnreg
from .._compat import use_tileir_tunables
from .._compat import warps_to_threads
from .config_fragment import Category
from .config_fragment import ConfigSpecFragment
from .config_fragment import PowerOfTwoFragment

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import Config
    from . import ConfigSpec

FlatConfig = list[object]


TRITON_MAX_TENSOR_NUMEL = 1048576


class ConfigGeneration:
    def __init__(
        self,
        config_spec: ConfigSpec,
        *,
        overrides: Mapping[str, object] | None = None,
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
        self.flat_spec: list[ConfigSpecFragment] = []
        config_spec.flat_config(_collect_spec)
        assert self.flat_spec, "No config values to tune"
        self._override_values = dict(overrides or {})
        self.block_size_indices: list[int] = [
            i
            for i, spec in enumerate(self.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        ]
        self.num_warps_index: int = next(
            i
            for i, spec in enumerate(self.flat_spec)
            if spec.category() == Category.NUM_WARPS
        )
        self._key_to_flat_index: dict[str, int] = self._build_key_index_mapping()
        self.num_stages_index: int = self._key_to_flat_index.get(
            "num_stages", self.num_warps_index + 1
        )
        self.indexing_index: int = self._key_to_flat_index.get(
            "indexing", self.num_stages_index + 1
        )
        self.pid_type_index: int = self._key_to_flat_index.get(
            "pid_type", self.indexing_index + 1
        )
        self.num_sm_multiplier_index: int = self._key_to_flat_index.get(
            "num_sm_multiplier", self.pid_type_index + 1
        )
        self.load_eviction_policies_index: int = self._key_to_flat_index.get(
            "load_eviction_policies", self.num_sm_multiplier_index + 1
        )
        self.min_block_size: int = (
            max([spec.min_size for spec in config_spec.block_sizes])
            if config_spec.block_sizes
            else 1
        )

    def _build_key_index_mapping(self) -> dict[str, int]:
        """
        Build a mapping from config key names to their starting flat index.

        This is used for best config available config transfer and to compute named indices
        for specific config keys. Each key maps to the index where its value(s)
        start in the flat_spec.

        The key order MUST match the order in ConfigSpec.flat_config() where
        fragments are added to flat_spec. If this order changes, update the list
        below AND the corresponding code in ConfigSpec.flat_config().

        This method also validates that known indices match expected positions
        to catch synchronization errors early.
        """
        from .config_fragment import ListOf

        pre_normalized_config = self._get_pre_normalized_config()

        # IMPORTANT: If you modify ConfigSpec.flat_config(), update this list too!
        flat_config_key_order = [
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "l2_groupings",
            "reduction_loops",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "num_warps",
            "num_stages",
            "indexing",
            "pid_type",
            "num_sm_multiplier",
            "load_eviction_policies",
            "num_ctas",
            "occupancy",
            "maxnreg",
        ]

        mapping: dict[str, int] = {}
        flat_idx = 0

        for key in flat_config_key_order:
            if key not in pre_normalized_config:
                continue
            value = pre_normalized_config[key]

            if isinstance(value, list):
                if flat_idx < len(self.flat_spec) and isinstance(
                    self.flat_spec[flat_idx], ListOf
                ):
                    mapping[key] = flat_idx
                    flat_idx += 1
                elif len(value) > 0:
                    mapping[key] = flat_idx
                    flat_idx += len(value)
            else:
                mapping[key] = flat_idx
                flat_idx += 1

        for key in self.config_spec.user_defined_tunables:
            mapping[key] = flat_idx
            flat_idx += 1

        if "num_warps" in mapping:
            assert mapping["num_warps"] == self.num_warps_index, (
                f"num_warps index mismatch: mapping says {mapping['num_warps']}, "
                f"but category-based detection found {self.num_warps_index}. "
                f"The flat_config_key_order list may be out of sync with ConfigSpec.flat_config()."
            )

        if "block_sizes" in mapping and self.block_size_indices:
            assert mapping["block_sizes"] == self.block_size_indices[0], (
                f"block_sizes index mismatch: mapping says {mapping['block_sizes']}, "
                f"but category-based detection found {self.block_size_indices[0]}. "
                f"The flat_config_key_order list may be out of sync with ConfigSpec.flat_config()."
            )

        return mapping

    def _get_pre_normalized_config(self) -> dict[str, object]:
        """
        Build a dict with the structure (not values) of the config before normalization.

        ConfigSpec.flat_config() calls normalize() which removes some keys. We need
        the pre-normalized structure to correctly map config keys to flat_spec indices.
        Only the existence of keys and length of lists matters, not actual values.
        """
        spec = self.config_spec

        pre_normalized: dict[str, object] = {
            # List fields - only length matters for index mapping
            "block_sizes": [None] * len(spec.block_sizes),
            "loop_orders": [None] * len(spec.loop_orders),
            "flatten_loops": [None] * len(spec.flatten_loops),
            "l2_groupings": [None] * len(spec.l2_groupings),
            "reduction_loops": [None] * len(spec.reduction_loops),
            "range_unroll_factors": [None] * len(spec.range_unroll_factors),
            "range_warp_specializes": [None] * len(spec.range_warp_specialize),
            "range_num_stages": [None] * len(spec.range_num_stages),
            "range_multi_buffers": [None] * len(spec.range_multi_buffers),
            "range_flattens": [None] * len(spec.range_flattens),
            "static_ranges": [None] * len(spec.static_ranges),
            # Scalar fields - just need to exist
            "num_warps": None,
            "num_stages": None,
            "indexing": None,
            "pid_type": None,
            "num_sm_multiplier": None,
            "load_eviction_policies": None,
        }

        if use_tileir_tunables():
            pre_normalized["num_ctas"] = None
            pre_normalized["occupancy"] = None

        if supports_maxnreg():
            pre_normalized["maxnreg"] = None

        return pre_normalized

    def _apply_overrides(self, config: Config) -> Config:
        if not self._override_values:
            return config
        for key, value in self._override_values.items():
            config.config[key] = copy.deepcopy(value)
        self.config_spec.normalize(config.config)
        return config

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
        count: itertools.count[int] = itertools.count()
        config = self.config_spec.flat_config(get_next_value)
        assert next(count) == len(flat_values)
        return self._apply_overrides(config)

    def block_numel(self, flat_config: FlatConfig) -> int:
        return functools.reduce(
            operator.mul,
            [cast("int", flat_config[i]) for i in self.block_size_indices],
            1,
        )

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
        num_threads = warps_to_threads(cast("int", flat_config[self.num_warps_index]))
        # Respect Triton's maximum tensor element limit
        triton_limit = TRITON_MAX_TENSOR_NUMEL
        theoretical_max_elements = max_elements_per_thread * num_threads
        max_elements = min(theoretical_max_elements, triton_limit)
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

    def default_flat(self) -> FlatConfig:
        """
        Retrieve the default flat configuration.

        Returns:
            The default flat configuration values.
        """
        return [spec.default() for spec in self.flat_spec]

    def random_flat(self) -> FlatConfig:
        """
        Generate a random flat configuration.

        Returns:
            A random flat configuration.
        """
        config = [spec.random() for spec in self.flat_spec]
        self.shrink_config(config, PowerOfTwoFragment(1, 2048, 32).random())
        return config

    def random_config(self) -> Config:
        return self.unflatten(self.random_flat())

    def random_population_flat(self, n: int) -> list[FlatConfig]:
        return [self.default_flat(), *[self.random_flat() for _ in range(n - 1)]]

    def random_population(self, n: int) -> list[Config]:
        return [*map(self.unflatten, self.random_population_flat(n))]

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
        crossover_mask = [random.random() < crossover_rate for _ in self.flat_spec]
        crossover_mask[random.randrange(len(crossover_mask))] = True
        result = [*x]
        for i, crossover in enumerate(crossover_mask):
            if crossover:
                result[i] = self.flat_spec[i].differential_mutation(a[i], b[i], c[i])
        # TODO(jansel): can this be larger? (too large and Triton compile times blow up)
        self.shrink_config(result, 8192)
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
