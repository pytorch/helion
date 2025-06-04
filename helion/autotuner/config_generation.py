from __future__ import annotations

import functools
import itertools
import operator
import random
from typing import TYPE_CHECKING
from typing import cast

from helion._compat import warps_to_threads
from helion.autotuner.config_fragment import Category
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment

if TYPE_CHECKING:
    from helion import Config
    from helion.autotuner import ConfigSpec

FlatConfig = list[object]


class ConfigGeneration:
    def __init__(
        self,
        config_spec: ConfigSpec,
    ) -> None:
        def _collect_spec(spec: ConfigSpecFragment) -> object:
            """
            Collect a configuration specification fragment.

            :param spec: The configuration specification fragment.
            :type spec: ConfigSpecFragment
            :return: The default value of the fragment.
            :rtype: object
            """
            self.flat_spec.append(spec)
            return spec.default()

        super().__init__()
        self.config_spec = config_spec
        self.flat_spec: list[ConfigSpecFragment] = []
        config_spec.flat_config(_collect_spec)
        assert self.flat_spec, "No config values to tune"
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
        self.min_block_size: int = max(
            [spec.min_size for spec in config_spec.block_sizes]
        )

    def unflatten(self, flat_values: FlatConfig) -> Config:
        """
        Convert a flat configuration back into a full configuration.

        :param flat_values: The flat configuration values.
        :type flat_values: FlatConfig
        :return: The full configuration object.
        :rtype: Config
        """

        def get_next_value(spec: ConfigSpecFragment) -> object:
            i = next(count)
            assert type(self.flat_spec[i]) is type(spec)
            return flat_values[i]

        assert len(flat_values) == len(self.flat_spec)
        count: itertools.count[int] = itertools.count()
        config = self.config_spec.flat_config(get_next_value)
        assert next(count) == len(flat_values)
        return config

    def block_numel(self, flat_config: FlatConfig) -> int:
        return functools.reduce(
            operator.mul, [cast("int", flat_config[i]) for i in self.block_size_indices]
        )

    def shrink_config(
        self, flat_config: FlatConfig, max_elements_per_thread: int
    ) -> None:
        """
        Fully random configs tend to run out of resources and tile a long time to compile.
        Here we shrink the config to a reasonable size.

        :param flat_config: config to mutate in place
        :param max_elements_per_thread: maximum number of elements per thread
        """
        num_threads = warps_to_threads(cast("int", flat_config[self.num_warps_index]))
        max_elements = max_elements_per_thread * num_threads
        while self.block_numel(flat_config) > max_elements:
            changes = 0
            for i in self.block_size_indices:
                val = flat_config[i]
                assert isinstance(val, int)
                if val > self.min_block_size:
                    flat_config[i] = val // 2
                    changes += 1
            if changes == 0:
                break

    def default_flat(self) -> FlatConfig:
        """
        Retrieve the default flat configuration.

        :return: The default flat configuration values.
        :rtype: FlatConfig
        """
        return [spec.default() for spec in self.flat_spec]

    def random_flat(self) -> FlatConfig:
        """
        Generate a random flat configuration.

        :return: A random flat configuration.
        :rtype: FlatConfig
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
