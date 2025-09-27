from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
import functools
import importlib
import os
import sys
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from helion.autotuner.benchmarker import benchmark_distributed


def formatt_large_number(num: int) -> str:
    if num >= 2**30:
        if num % (2**30) == 0:
            return f"{num / 2**30:.0f}g"
    if num >= 2**20:
        if num % (2**20) == 0:
            return f"{num / 2**20:.0f}m"
    if num >= 2**14:
        if num % (2**10) == 0:
            return f"{num / 2**10:.0f}k"
    return str(num)


def clone_symm_mem_tensor(tensor: torch.Tensor) -> torch.Tensor:
    symm_mem_tensor = symm_mem.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    assert dist.group.WORLD is not None
    symm_mem.rendezvous(symm_mem_tensor, dist.group.WORLD.group_name)
    symm_mem_tensor.copy_(tensor)
    return symm_mem_tensor


def clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.group.WORLD is not None
    symm_mem_hdl = symm_mem.rendezvous(tensor, dist.group.WORLD.group_name)
    if symm_mem_hdl:
        return clone_symm_mem_tensor(tensor)
    return tensor.clone()


def clone_inputs(args: tuple[object]) -> tuple[object]:
    return tuple([clone_tensor(t) if torch.is_tensor(t) else t for t in args])  # pyright: ignore[reportReturnType]


@dataclass
class ExperimentConfig:
    """Configuration class for distributed benchmark experiments.

    This dataclass holds the parameters needed to configure a single benchmark
    experiment, including tensor shape, data type, and backend options.

    Attributes:
        shape: Tuple defining the dimensions of tensors used in the experiment
        dtype: PyTorch data type for tensors (e.g., torch.float32, torch.bfloat16)
        backends: List of backend names to benchmark against
        device: Target device for the experiment, defaults to None (auto-detected)
    """

    shape: tuple[int]
    dtype: torch.dtype
    backends: list[str]
    device: torch.device | None = None

    def asdict(self) -> dict:
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        d.pop("backends", None)
        d.pop("device", None)

        formated_size = [formatt_large_number(num) for num in self.shape]
        d["shape"] = f"({', '.join(formated_size)})"
        return d


@dataclass
class Experiment:
    """Container for a completed benchmark experiment.

    This dataclass stores both the configuration used for an experiment
    and the timing results obtained from running different backends.

    Attributes:
        config: The ExperimentConfig used to run this experiment
        results: Dictionary mapping backend names to execution times in microseconds
    """

    config: ExperimentConfig
    results: dict[str, float]  # backend -> time in us

    def asdict(self) -> dict:
        dict1 = self.config.asdict()
        dict2 = self.results
        return {**dict1, **dict2}


class BenchmarkOperator:
    """Base class for distributed benchmark operators.

    This class provides a framework for benchmarking distributed operations
    across multiple backends. It handles distributed setup, argument parsing,
    experiment execution, and results collection/reporting.

    Each benchmark operator should inherit from this class and initiate the
    the backend_dict, ope_name and baseline attributes.

    op_name: str
        Name of the operator being benchmarked.
    baseline: str
        Name of the baseline backend to compare against.
    backend_dict: dict[str, Callable]
        Dictionary mapping backend names to callables.

    The operator subclass should also implement the following methods:
    - gen_configs(self, args: argparse.Namespace) -> list[ExperimentConfig]
        Generate ExperimentConfigs based on the command line args.
    - gen_inputs(self, config: ExperimentConfig) -> tuple
        Generate input tensors for the experiment based on the config.
    - (optional) additional_parser_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser
        Add additional command line arguments to the parser.
        Default command arguments are:
        - op, backend, dtype, save_path.
    """

    baseline: str
    op_name: str
    backend_dict: dict[str, Callable | tuple[str, str]]

    MASTER_RANK = 0

    help_str = """
Run with torchrun
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 \
benchmarks/run_distributed.py
"""

    experiments: list[Experiment]
    configs: list[ExperimentConfig]
    results_tab: defaultdict | None
    results_title: str = ""
    save_path: str = ""
    args: argparse.Namespace | None

    fn_dict: dict[str, Callable | None]

    local_rank: int
    world_size: int
    device: torch.device

    def gen_configs(self, args: argparse.Namespace) -> list[ExperimentConfig]:
        """Generate experiment configurations based on command line arguments.

        This method must be implemented by subclasses to define what experiments
        should be run based on the parsed command line arguments.

        Args:
            args: Command line arguments containing experiment parameters

        Returns:
            List of ExperimentConfig objects defining the experiments to run
        """
        raise NotImplementedError("Must overwrite config generation. ")

    def gen_inputs(self, config: ExperimentConfig) -> tuple:
        """Generate input tensors for a specific experiment configuration.

        This method must be implemented by subclasses to create the input tensors
        and arguments needed for the benchmarked operation.

        Args:
            config: ExperimentConfig containing shape, dtype, and device information

        Returns:
            Tuple of input arguments (typically tensors) for the benchmarked operation
        """
        raise NotImplementedError("Must overwrite input generation")

    def is_master_rank(self) -> bool:
        return self.local_rank == self.MASTER_RANK

    def additional_parser_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Add operation-specific command line arguments to the argument parser.

        Subclasses can override this method to add custom command line arguments
        beyond the default ones (op, backend, dtype, save_path).
        """
        return parser

    def _parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"Run benchmark for {self.__name__}. " + self.help_str
        )

        parser.add_argument(
            "--backend",
            type=str,
            nargs="+",
            choices=list(self.backend_dict.keys()),
            default=list(self.backend_dict.keys()),
            help="Backend to use for AllReduce. ",
        )

        parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")
        parser.add_argument(
            "--save-path",
            type=str,
            help="Path to save the results JSON file (optional)",
            default="",
        )

        parser = self.additional_parser_args(parser)

        self.args = parser.parse_args()
        self.args.dtype = getattr(torch, self.args.dtype)

        return self.args

    def __init__(self) -> None:
        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        except KeyError:
            print(
                "Error: LOCAL_RANK environment variable must be a valid integer. Are you running with torchrun? "
            )
            print(self.help_str)
            sys.exit(1)

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl")
        torch.manual_seed(42 + self.local_rank)

        self.experiments = []
        self.configs = []
        self.results_tab = None
        self.args = None
        self.fn_dict = {}

    def _setup(self, configs: list[ExperimentConfig] | None = None) -> None:
        assert len(self.backend_dict) > 0

        for key, value in self.backend_dict.items():
            if isinstance(value, tuple):
                module_path, func_name = value
                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    print(
                        f"Warning: Failed to import module '{module_path}': {e}. Backend {key} skipped. ",
                        file=sys.stderr,
                    )
                    self.fn_dict[key] = None
                    continue
                if not hasattr(module, func_name):
                    print(
                        f"Warning: Module '{module_path}' does not have a function named '{func_name}'. Backend {key} skipped. ",
                        file=sys.stderr,
                    )
                    kernel_func = None
                else:
                    kernel_func = getattr(module, func_name)
                self.fn_dict[key] = kernel_func
            else:
                self.fn_dict[key] = value

        if configs is None:
            self.args = self._parse_args()
            configs = self.gen_configs(self.args)

        self.configs = configs
        assert isinstance(self.args, argparse.Namespace)
        self.save_path = self.args.save_path
        self.dypte = self.args.dtype

        for config in self.configs:
            if config.device is None:
                config.device = self.device

    def __name__(self) -> str:
        return self.op_name

    def _parse_results(self, metric: str) -> defaultdict | None:
        if len(self.experiments) == 0:
            return None

        table_data = defaultdict(list)
        for experiment in self.experiments:
            baseline_time = experiment.results[self.baseline]
            min_time = float("inf")
            best_backend = self.baseline
            backends = experiment.config.backends
            for key, value in experiment.asdict().items():
                if key in backends:
                    import math

                    if not math.isnan(value) and value < min_time:
                        min_time = value
                        best_backend = key
                    if metric == "speedup":
                        table_data[key].append(baseline_time / value)
                    elif metric == "time_us":
                        table_data[key].append(value)
                    else:
                        raise NotImplementedError(
                            f"Unknown result metric: {metric}. Valid metrics are: time_us, speedup"
                        )
                else:
                    table_data[key].append(value)
            table_data["Best Backend"].append(best_backend)
        self.results_tab = table_data
        self.results_title = f"Benchmarking results for {self.op_name} on {self.world_size}x devices. ({metric})"
        return table_data

    def run(self) -> list[Experiment]:
        """Execute all configured benchmark experiments.

        Runs each experiment configuration and collect timing results.

        Returns:
            List of completed Experiment objects containing configurations and results.
        """
        if len(self.configs) == 0:
            self._setup()
        for config in self.configs:
            self.experiments.append(Experiment(config, self._run_experiment(config)))
        return self.experiments

    def save_results(self, metric: str = "speedup") -> None:
        """
        (The master rank process) Saves benchmark results to a CSV file.

        Args:
            metric: Type of metric to save ("speedup" or "time_us")
        """
        if not self.is_master_rank():
            return
        if self.save_path == "":
            return
        self.results_tab = self._parse_results(metric)
        if self.results_tab is None:
            return

        import csv

        with open(self.save_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.results_tab.keys())
            writer.writeheader()
            for i in range(len(next(iter(self.results_tab.values())))):
                row = {k: v[i] for k, v in self.results_tab.items()}
                writer.writerow(row)
        print(f"\nResults saved to {self.save_path}")

    def print_results(self, metric: str = "speedup") -> None:
        """Print benchmark results in a formatted table to stdout.

        Only the master rank process will print results to avoid duplicate output.
        Results are displayed in a GitHub-style markdown table format.

        Args:
            metric: Type of metric to display ("speedup" or "time_us")
        """
        from tabulate import tabulate

        if not self.is_master_rank():
            return

        self.results_tab = self._parse_results(metric)
        if self.results_tab is None:
            return
        print(self.results_title)
        print(
            tabulate(
                self.results_tab, headers="keys", tablefmt="github", floatfmt=".3f"
            )
        )

    def get_results(self, metric: str = "speedup") -> defaultdict | None:
        """Get benchmark results as a structured data object.

        Returns the parsed results in a defaultdict format suitable for
        programmatic access or further processing.

        Args:
            metric: Type of metric to return ("speedup" or "time_us")
        """
        self.results_tab = self._parse_results(metric)
        return self.results_tab

    def _run_experiment(self, config: ExperimentConfig) -> dict[str, float]:
        if self.baseline not in config.backends:
            backends = config.backends.append(self.baseline)
        else:
            backends = config.backends

        gloden_inp = self.gen_inputs(config)
        inputs = {backend: clone_inputs(gloden_inp) for backend in backends}  # pyright: ignore[reportOptionalIterable]

        gloden_fn = self.fn_dict[self.baseline]
        assert gloden_fn is not None

        gloden_o = gloden_fn(*gloden_inp)

        results = {}
        for backend in backends:  # pyright: ignore[reportOptionalIterable]
            fn = self.fn_dict[backend]
            if fn is None:
                results[backend] = float("nan")
                continue
            inp = inputs[backend]
            target_fn = functools.partial(fn, *inp)
            try:
                test_o = target_fn()
            except RuntimeError:
                results[backend] = float("nan")
                continue
            torch.testing.assert_close(test_o, gloden_o, atol=1e-1, rtol=1e-1)

            results[backend] = benchmark_distributed(
                target_fn, profile_ranks=[self.MASTER_RANK]
            )

        return results
