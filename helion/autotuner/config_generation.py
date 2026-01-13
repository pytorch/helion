from __future__ import annotations

import copy
import functools
import itertools
import operator
import random
from typing import TYPE_CHECKING
from typing import cast

from .._compat import warps_to_threads
from .config_fragment import Category
from .config_fragment import ConfigSpecFragment
from .config_fragment import ListOf
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
        self.min_block_size: int = (
            max([spec.min_size for spec in config_spec.block_sizes])
            if config_spec.block_sizes
            else 1
        )

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

    def flatten(self, config: Config) -> FlatConfig:
        """
        Convert a Config to its flat representation.

        This is the inverse of unflatten() - it extracts values from a Config
        in the same order that flat_spec expects them.

        Args:
            config: The configuration to flatten.

        Returns:
            A flat list of configuration values.
        """
        import copy

        # Normalize the config to fill in any missing defaults
        config_dict = copy.deepcopy(config.config)
        self.config_spec.normalize(config_dict)

        result: FlatConfig = []

        # Extract values in the same order as flat_config builds them
        # block_sizes
        block_sizes = config_dict.get("block_sizes", [])
        for val in block_sizes:
            result.append(val)

        # loop_orders
        loop_orders = config_dict.get("loop_orders", [])
        for order in loop_orders:
            result.append(order)

        # flatten_loops
        flatten_loops = config_dict.get("flatten_loops", [])
        for val in flatten_loops:
            result.append(val)

        # l2_groupings
        l2_groupings = config_dict.get("l2_groupings", [])
        for val in l2_groupings:
            result.append(val)

        # reduction_loops
        reduction_loops = config_dict.get("reduction_loops", [])
        for val in reduction_loops:
            result.append(val)

        # range_unroll_factors
        range_unroll_factors = config_dict.get("range_unroll_factors", [])
        for val in range_unroll_factors:
            result.append(val)

        # range_warp_specializes
        range_warp_specializes = config_dict.get("range_warp_specializes", [])
        for val in range_warp_specializes:
            result.append(val)

        # range_num_stages
        range_num_stages = config_dict.get("range_num_stages", [])
        for val in range_num_stages:
            result.append(val)

        # range_multi_buffers
        range_multi_buffers = config_dict.get("range_multi_buffers", [])
        for val in range_multi_buffers:
            result.append(val)

        # range_flattens
        range_flattens = config_dict.get("range_flattens", [])
        for val in range_flattens:
            result.append(val)

        # static_ranges
        static_ranges = config_dict.get("static_ranges", [])
        for val in static_ranges:
            result.append(val)

        # num_warps (single value)
        result.append(config_dict.get("num_warps", 4))

        # num_stages (single value)
        result.append(config_dict.get("num_stages", 1))

        # indexing (list - treated as single ListOf fragment)
        result.append(config_dict.get("indexing", []))

        # pid_type (single value)
        result.append(config_dict.get("pid_type", "flat"))

        # load_eviction_policies (list - treated as single ListOf fragment)
        result.append(config_dict.get("load_eviction_policies", []))

        # user_defined_tunables (in alphabetical order by key)
        for key in sorted(self.config_spec.user_defined_tunables.keys()):
            result.append(config_dict.get(key))

        # Expand single-element lists to match expected ListOf lengths
        # This handles cases where initial_config specifies e.g. indexing="pointer"
        # but the kernel has multiple loads/stores expecting a longer list
        for i, spec in enumerate(self.flat_spec):
            if isinstance(spec, ListOf) and spec.length > 0:
                value = result[i]
                if isinstance(value, list) and len(value) == 1 and spec.length > 1:
                    # Expand single-element list by repeating the value
                    result[i] = value * spec.length

        # Validate length matches flat_spec
        if len(result) != len(self.flat_spec):
            raise ValueError(
                f"Flattened config length {len(result)} does not match "
                f"flat_spec length {len(self.flat_spec)}"
            )

        return result
