"""Run the existing Qwen3 SM-trace harness against this worktree's CLC lowering."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import probes
import probes.common
import probes.qwen3


REFERENCE_REPO = Path(__file__).resolve().parents[3] / "helion-cross-kernel"
REFERENCE_PROBES = REFERENCE_REPO / "probes"
REFERENCE_GANTT = (
    REFERENCE_PROBES / "qwen3" / "qwen3_separate_vs_megakernel_gantt.py"
)


def _load_reference_gantt():
    probes.__path__.append(str(REFERENCE_PROBES))
    probes.qwen3.__path__.append(str(REFERENCE_PROBES / "qwen3"))
    common_spec = importlib.util.spec_from_file_location(
        "qwen3_reference_common",
        REFERENCE_PROBES / "common.py",
    )
    if common_spec is None or common_spec.loader is None:
        raise RuntimeError("cannot load reference probe helpers")
    reference_common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(reference_common)
    for name in dir(reference_common):
        if not hasattr(probes.common, name):
            setattr(probes.common, name, getattr(reference_common, name))
    module_name = "qwen3_reference_gantt"
    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_GANTT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE_GANTT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/qwen3_clc_sm_gantt.png"),
    )
    parser.add_argument("--cross-loop-workers", type=int, default=1024)
    args = parser.parse_args()

    from probes.qwen3 import helion_qwen3_granular_tile_dependency as granular

    original_probe_config = granular._probe_config

    def clc_probe_config(bound, model_args):
        model_args.cross_loop_workers = args.cross_loop_workers
        config = original_probe_config(bound, model_args)
        bound.config_spec.normalize(config.config)
        return config

    granular._probe_config = clc_probe_config
    gantt = _load_reference_gantt()
    from probes.qwen3 import benchmark_production

    benchmark_production._prepare_checkpoint_tensors = lambda _tensors, _ue8m0: None
    original_qwen_args = gantt._qwen_args

    def qwen_args():
        model_args = original_qwen_args()
        model_args.batch = 1
        model_args.helion_comparison_splits = model_args.helion_attention_splits
        return model_args

    gantt._qwen_args = qwen_args
    gantt.run(args)


if __name__ == "__main__":
    main()
