"""Benchmark 15 existing Helion TPU examples with fixed configs.

Run on a TPU: python tpu_attention_results/benchmark_examples.py --output examples.csv
Kernels and references are imported; checking and timing reuse run_example.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import partial
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING
from typing import Any

os.environ.setdefault("HELION_BACKEND", "pallas")
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_tpu import KERNEL_MAPPINGS
from examples.gdn_fwd_h import make_inputs as gdn_inputs
from examples.gdn_fwd_h import ref_gdn_fwd_h
import torch
from torch.utils._pytree import tree_leaves

import helion
from helion._testing import assert_relative_l2_close
from helion._testing import float32_matmul_precision
from helion._testing import get_test_float32_matmul_precision
from helion._testing import run_example

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Example:
    module: str
    function: str
    config_kernel: str
    inputs: Callable[[], tuple[Any, ...]]
    reference: Callable[..., Any]
    # Some kernels return auxiliary statistics as well as the reference output.
    output_index: int | None = None


def _selected_inputs(
    factory: Callable[..., list[tuple[str, tuple[Any, ...]]]], index: int
) -> tuple[Any, ...]:
    return factory(index + 1)[index][1]


def _registered_example(
    key: str,
    index: int,
    *,
    config_kernel: str | None = None,
    output_index: int | None = None,
) -> Example:
    module, function, reference, factory, _ = KERNEL_MAPPINGS[key][:5]
    assert reference is not None and factory is not None
    return Example(
        module,
        function,
        function if config_kernel is None else config_kernel,
        partial(_selected_inputs, factory, index),
        reference,
        output_index,
    )


# Entrypoints, input factories and output contracts are data; timing is shared.
CASES = {
    "attention": _registered_example("flash_attention", 1, output_index=0),
    "batch_softmax": _registered_example("batch_softmax", 1),
    "bmm": _registered_example("bmm", 1),
    "broadcast_matmul": _registered_example("broadcast_matmul", 1),
    "cross_entropy": _registered_example("cross_entropy", 0),
    "embedding": _registered_example("embedding", 1),
    "gdn_fwd_h": Example(
        "gdn_fwd_h",
        "helion_gdn_fwd_h",
        "helion_gdn_fwd_h",
        partial(gdn_inputs, 1, 8, 4096, 256, 64, 128),
        ref_gdn_fwd_h,
    ),
    "geglu": _registered_example("geglu", 1),
    "grpo_loss": _registered_example("grpo_loss", 1, config_kernel="grpo_loss_forward"),
    "layer_norm": _registered_example("layer_norm", 1, config_kernel="layer_norm_fwd"),
    "matmul_layernorm": _registered_example("matmul_layernorm", 2),
    "rms_norm": _registered_example("rms_norm", 1, output_index=0),
    "rms_norm_bwd": _registered_example("rms_norm-bwd", 1),
    "swiglu": _registered_example("swiglu", 1),
    "welford": _registered_example("welford", 1),
}

# Measured configs for the exact shapes above; this script does not run a search.
CONFIGS: dict[str, dict[str, object]] = {
    "attention": {
        "block_sizes": [2, 512, 2048],
        "loop_orders": [[0, 1]],
        "pallas_load_buffer_count": [1, 1, 1],
        "pallas_loop_type": "unroll",
        "pallas_pre_broadcast": False,
    },
    "batch_softmax": {
        "block_sizes": [8, 128],
        "loop_orders": [[0, 1]],
        "flatten_loops": [False],
    },
    "bmm": {
        "block_sizes": [1, 1024, 2048, 2048],
        "loop_orders": [[2, 1, 0]],
        "pallas_load_buffer_count": [1, 1],
        "pallas_loop_type": "unroll",
        "pallas_use_low_level_scheduler": True,
    },
    "broadcast_matmul": {
        "block_sizes": [1024, 2048, 2048],
        "loop_orders": [[1, 0]],
        "pallas_load_buffer_count": [1, 1],
        "pallas_loop_type": "unroll",
        "pallas_use_low_level_scheduler": True,
    },
    "cross_entropy": {
        "block_sizes": [128],
        "pallas_indirect_access_mode": "one_hot",
    },
    "embedding": {
        "block_sizes": [8192, 64],
        "loop_orders": [[0, 1]],
        "pallas_indirect_access_mode": "one_hot",
    },
    "gdn_fwd_h": {
        "block_sizes": [128],
        "loop_orders": [[0, 1, 2]],
        "pallas_load_buffer_count": [1, 1, 1, 1],
        "pallas_loop_type": "unroll",
        "pallas_use_low_level_scheduler": False,
        "pallas_pre_broadcast": True,
    },
    "geglu": {
        "block_sizes": [2, 8192, 128],
        "loop_orders": [[2, 0, 1]],
        "flatten_loops": [False],
    },
    "grpo_loss": {
        "block_sizes": [4, 512, 2048],
        "loop_orders": [[1, 0]],
        "pallas_load_buffer_count": [2, 1, 2, 1, 1, 2],
        "pallas_loop_type": "unroll",
        "pallas_use_low_level_scheduler": False,
    },
    "layer_norm": {"block_sizes": [128]},
    "matmul_layernorm": {
        "block_sizes": [2048, 512],
        "pallas_loop_type": "emit_pipeline",
        "pallas_emit_pipeline_group_size": 1,
        "pallas_use_low_level_scheduler": False,
        "pallas_pre_broadcast": True,
    },
    "rms_norm": {"block_sizes": [512]},
    "rms_norm_bwd": {
        "block_sizes": [8192, 128],
        "pallas_worklist_grouping": 0,
        "pallas_loop_type": "emit_pipeline",
        "pallas_emit_pipeline_group_size": 1,
        "pallas_use_low_level_scheduler": False,
        "pallas_pre_broadcast": True,
    },
    "swiglu": {
        "block_sizes": [2, 16384, 128],
        "loop_orders": [[1, 2, 0]],
        "flatten_loops": [False],
    },
    "welford": {
        "block_sizes": [4096, 512, 256],
        "pallas_load_buffer_count": [1, 1, 1],
        "pallas_loop_type": "unroll",
        "pallas_use_low_level_scheduler": True,
    },
}


def benchmark(name: str, rounds: int) -> dict[str, object]:
    torch._dynamo.reset()
    torch.manual_seed(0)
    example = CASES[name]
    module = importlib.import_module(f"examples.{example.module}")
    kernel = getattr(module, example.config_kernel)
    kernel.settings.static_shapes = True
    kernel.settings.force_autotune = False
    kernel.configs = [helion.Config.from_dict(CONFIGS[name])]
    function = getattr(module, example.function)
    inputs = example.inputs()

    def invoke(*args: object) -> torch.Tensor | tuple[torch.Tensor, ...]:
        result = function(*args)
        return result if example.output_index is None else result[example.output_index]

    expected = example.reference(*inputs)
    actual = invoke(*inputs)
    repeated = invoke(*inputs)
    for got, ref, again in zip(
        tree_leaves(actual), tree_leaves(expected), tree_leaves(repeated), strict=True
    ):
        torch.testing.assert_close(got, again, rtol=0, atol=0)
        assert_relative_l2_close(got, ref, 0.01)

    baselines = {
        "torch_tpu_eager": example.reference,
        "torch_compile": torch.compile(
            example.reference, backend="tpu", fullgraph=True, dynamic=False
        ),
    }
    # run_example also handles tuple outputs, despite its Tensor-only annotation.
    samples = [
        run_example(
            invoke,  # pyrefly: ignore[bad-argument-type]
            baselines,
            inputs,
        )
        for _ in range(rounds)
    ]
    timings = {
        provider: statistics.median(sample[provider] for sample in samples)
        for provider in samples[0]
    }
    shape = json.dumps(
        [list(arg.shape) if isinstance(arg, torch.Tensor) else arg for arg in inputs],
        separators=(",", ":"),
    )
    eager = timings["torch_tpu_eager"]
    compiled_ms = timings["torch_compile"]
    return {
        "example": name,
        "shape": shape,
        "dtype": str(
            next(
                x.dtype
                for x in inputs
                if isinstance(x, torch.Tensor) and x.is_floating_point()
            )
        ),
        "torch_tpu_eager_ms": eager,
        "torch_compile_ms": compiled_ms,
        "helion_ms": timings["helion"],
        "torch_compile_speedup": eager / compiled_ms,
        "helion_speedup": eager / timings["helion"],
        "helion_speedup_vs_best": min(eager, compiled_ms) / timings["helion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--output", type=Path, default=Path("tpu_examples.csv"))
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(8)
    with (
        float32_matmul_precision(get_test_float32_matmul_precision()),
        args.output.open("w", newline="") as handle,
    ):
        writer = None
        for name in args.kernel:
            row = benchmark(name, args.rounds)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
