from __future__ import annotations

import collections
import functools
import math
from math import inf
import re
import sys
import time
from typing import TYPE_CHECKING
from typing import NamedTuple

from torch._inductor.runtime.triton_compat import OutOfResources
from triton.testing import do_bench

from .. import exc
from ..runtime.settings import LogLevel
from .config_generation import ConfigGeneration
from .config_generation import FlatConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.settings import Settings
    from . import ConfigSpec

_expected_errors_regexp: re.Pattern[str] = re.compile(
    # pyre-ignore[6]
    r"|".join(map(re.escape, ["[CUDA]: invalid argument"]))
)


class BaseSearch:
    """
    Base class for search algorithms. This class defines the interface and utilities for all
    search algorithms.

    Attributes:
        kernel (BoundKernel): The kernel to be tuned.
        settings (Settings): The settings associated with the kernel.
        config_spec (ConfigSpec): The configuration specification for the kernel.
        args (Sequence[object]): The arguments to be passed to the kernel.
        counters (collections.Counter): A counter to track various metrics during the search.
        _benchmark_cache (dict[Config, float]): A cache to store benchmark results for configurations.
    """

    # pyre-fixme[11]: BoundKernel undefined?
    def __init__(self, kernel: BoundKernel, args: Sequence[object]) -> None:
        """
        Initialize the BaseSearch object.

        :param kernel: The kernel to be tuned.
        :type kernel: BoundKernel
        :param args: The arguments to be passed to the kernel.
        :type args: Sequence[object]
        """
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args = args
        self.counters: collections.Counter[str] = collections.Counter()

    def benchmark(self, config: Config) -> float:
        """
        Benchmark a specific configuration.

        This method compiles the kernel with the given configuration and measures its performance.

        :param config: The configuration to benchmark.
        :type config: Config
        :return: The performance of the configuration in seconds.
        :rtype: float
        """
        # TODO(jansel): early exit with fewer trials if early runs are slow
        fn = self.kernel.compile_config(config)
        self.counters["benchmark"] += 1
        self.log(lambda: f"Running benchmark for {config!r}", level=LogLevel.DEBUG)
        try:
            t0 = time.perf_counter()
            fn(*self.args)  # make sure the kernel is compiled
            t1 = time.perf_counter()
            res = do_bench(
                functools.partial(fn, *self.args),
                return_mode="median",
            )
            t2 = time.perf_counter()
            self.log(
                f"result: {res:.4f}s (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
                level=LogLevel.DEBUG,
            )
            return res
        except OutOfResources:
            self.log("Benchmarking failed: OutOfResources", level=LogLevel.DEBUG)
        except Exception as e:
            if not _expected_errors_regexp.search(str(e)):
                raise exc.TritonError(f"{type(e).__qualname__}: {e}", config) from e
            self.log(
                f"Benchmarking failed: {type(e).__name__}: {e}", level=LogLevel.DEBUG
            )
        return inf

    def parallel_benchmark(self, configs: list[Config]) -> list[tuple[Config, float]]:
        """
        Benchmark multiple configurations in parallel.

        :param configs: A list of configurations to benchmark.
        :type configs: list[Config]
        :return: A list of tuples containing configurations and their performance.
        :rtype: list[tuple[Config, float]]
        """
        # TODO(jansel): make this compile in parallel
        return [(c, self.benchmark(c)) for c in configs]

    def log(self, *msg: str | Callable[[], str], level: int = LogLevel.INFO) -> None:
        """
        Log a message at a specified log level.

        :param msg: The message(s) to log. Can be strings or callables that return strings.
        :type msg: str | Callable[[], str]
        :param level: The log level for the message.
        :type level: int
        """
        if self.settings.autotune_log_level >= level:
            sys.stderr.write(" ".join(map(_maybe_call, msg)) + "\n")

    def autotune(self) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        :return: The best configuration found during autotuning.
        :rtype: Config
        """
        start = time.perf_counter()
        best = self._autotune()
        end = time.perf_counter()
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self.counters['benchmark']} configs.\n"
            "One can hardcode the best config with and skip autotuning with:\n"
            f"    @helion.kernel(config={best!r})\n",
            level=LogLevel.SUMMARY,
        )
        return best

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


def _maybe_call(fn: Callable[[], str] | str) -> str:
    """
    Call a callable or return the string directly.

    :param fn: A callable that returns a string or a string.
    :type fn: Callable[[], str] | str
    :return: The resulting string.
    :rtype: str
    """
    if callable(fn):
        return fn()
    return fn


class PopulationMember(NamedTuple):
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perf (float): The performance of the configuration.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
    """

    perf: float
    flat_values: FlatConfig
    config: Config


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    :param member: The population member.
    :type member: PopulationMember
    :return: The performance of the member.
    :rtype: float
    """
    return member.perf


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        :param kernel: The kernel to be tuned.
        :type kernel: BoundKernel
        :param args: The arguments to be passed to the kernel.
        :type args: Sequence[object]
        """
        super().__init__(kernel, args)
        self.population: list[PopulationMember] = []
        self.config_gen: ConfigGeneration = ConfigGeneration(self.config_spec)

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        :return: The best population member.
        :rtype: PopulationMember
        """
        return min(self.population, key=performance)

    def benchmark_flat(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Benchmark a flat configuration.

        :param flat_values: The flat configuration values.
        :type flat_values: FlatConfig
        :return: A population member with the benchmark results.
        :rtype: PopulationMember
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(self.benchmark(config), flat_values, config)

    def parallel_benchmark_flat(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        :param to_check: A list of flat configurations to benchmark.
        :type to_check: list[FlatConfig]
        :return: A list of population members with the benchmark results.
        :rtype: list[PopulationMember]
        """
        configs = [*map(self.config_gen.unflatten, to_check)]
        result = []
        for flat_values, config_in, (config_out, perf) in zip(
            to_check, configs, self.parallel_benchmark(configs), strict=True
        ):
            assert config_in is config_out
            result.append(PopulationMember(perf, flat_values, config_in))
        return result

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        :return: A string summarizing the population performance.
        :rtype: str
        """
        return population_statistics(self.population)


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    :param population: The population of configurations.
    :type population: list[PopulationMember]
    :return: A string summarizing the performance of the population.
    :rtype: str
    """
    population = sorted(population, key=performance)
    if math.isinf(population[-1].perf):
        working = [x for x in population if not math.isinf(x.perf)]
        return (
            f"failed={len(population) - len(working)} "
            f"min={working[0].perf:.4f}s "
            f"mid={working[len(working) // 2].perf:.4f}s "
            f"max={working[-1].perf:.4f}s "
            f"best={population[0].config!s}"
        )
    return (
        f"min={population[0].perf:.4f}s "
        f"mid={population[len(population) // 2].perf:.4f}s "
        f"max={population[-1].perf:.4f}s "
        f"best={population[0].config!s}"
    )
