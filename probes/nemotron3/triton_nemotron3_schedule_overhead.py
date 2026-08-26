"""Measure Nemotron's generated CLC schedule with empty compute roots."""

from __future__ import annotations

import argparse
import ast
import json
import linecache
from pathlib import Path
import re
from typing import Callable

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.gemma4.helion_gemma4_e4b_megakernel import _helion_resources
from probes.nemotron3.helion_nemotron3_nano_moe import FP8_MAX as _FP8_MAX
from probes.nemotron3.helion_nemotron3_nano_moe import Nemotron3NanoMoEShape
from probes.nemotron3.helion_nemotron3_nano_moe import allocate
from probes.nemotron3.helion_nemotron3_nano_moe import initialize_autotune_inputs
from probes.nemotron3.helion_nemotron3_nano_moe_megakernel import MEGAKERNELS
from probes.nemotron3.helion_nemotron3_nano_moe_megakernel import _config
from probes.nemotron3.helion_nemotron3_nano_moe_megakernel import _kernel_args


def _empty_root_source(source: str) -> str:
    source = source.replace("_source_module.FP8_MAX", repr(_FP8_MAX))
    module = ast.parse(source)
    root_pattern = re.compile(r"tile_dependency_root_[0-9]+")
    root_count = 0
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and root_pattern.fullmatch(node.name):
            node.body = [ast.Pass()]
            root_count += 1
    if root_count == 0:
        raise RuntimeError("lowering did not contain outlined compute roots")
    return ast.unparse(ast.fix_missing_locations(module)) + "\n"


def _load(
    source: str,
    path: Path,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    path.write_text(source)
    filename = str(path)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_nemotron3_schedule_overhead"}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["nemotron3_nano_moe_megakernel_source"]


def _compiler_config_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        router_block_n=32,
        router_block_k=32,
        pointwise_block=256,
        expand_block=32,
        shared_block_n=32,
        shared_up_block_k=512,
        shared_down_block_k=512,
        routed_block_n=32,
        routed_block_k=512,
        shared_up_stages=1,
        shared_down_stages=2,
        shared_down_unroll=2,
        routed_stages=1,
        worker_multiplier=4,
        workers=args.workers,
        num_warps=2,
        kernel_stages=1,
        maxnreg=None,
    )


def run(args: argparse.Namespace) -> None:
    require_idle_visible_gpu()
    shape = Nemotron3NanoMoEShape(tokens=args.tokens)
    tensors = allocate(shape)
    initialize_autotune_inputs(shape, tensors)
    kernel_args = _kernel_args(tensors, shape)
    kernel, _ = MEGAKERNELS[args.dense_routed_activation]
    bound = kernel.bind(kernel_args)
    config = _config(bound, _compiler_config_args(args))
    lowered = bound.to_triton_code(config, output_origin_lines=False)
    empty_source = _empty_root_source(lowered)
    compiled = _load(empty_source, args.output.resolve())

    compiled(*kernel_args)
    torch.cuda.synchronize()
    resources = _helion_resources(compiled)
    graph, _ = capture(lambda: compiled(*kernel_args))
    timings = benchmark_interleaved(
        {"empty_nemotron_schedule": graph.replay},
        args.repeats,
        args.batch_replays,
    )
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "tokens": args.tokens,
                "dense_routed_activation": args.dense_routed_activation,
                "resources": resources,
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--workers", type=int, default=592)
    parser.add_argument("--dense-routed-activation", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/nemotron3_empty_schedule.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
