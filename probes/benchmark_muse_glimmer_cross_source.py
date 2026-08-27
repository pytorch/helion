# ruff: noqa: ANN001, ANN201, ANN202
"""Compile the authoritative Muse-Glimmer probe with helion-clc.

The model source lives in the sibling ``muse_glimmer`` workspace.  This driver
imports Helion from this ``helion-clc`` worktree first, then imports the production
shapes, exact standalone kernels, and composed megakernel from that workspace.
It checks every materialized intermediate against the PyTorch/vLLM reference,
saves the lowered Triton, and compares CUDA graphs with a 256 MiB L2 flush
before every individually timed replay.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock


HELION_ROOT = Path(__file__).resolve().parents[1]
if str(HELION_ROOT) not in sys.path:
    sys.path.insert(0, str(HELION_ROOT))

import helion
import torch


@dataclasses.dataclass(frozen=True)
class Variant:
    config_mode: str
    workers: int
    num_warps: int
    kernel_stages: int
    down_split_k: int
    maxnreg: int | None
    source_mode: str

    @property
    def name(self) -> str:
        return (
            f"{self.source_mode}_{self.config_mode}"
            f"_w{self.workers}_nw{self.num_warps}"
            f"_ks{self.kernel_stages}_down{self.down_split_k}"
            f"_r{self.maxnreg or 'auto'}"
        )


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_sources(source_root: Path, cross_root: Path):
    """Load source probes without allowing either sibling Helion to win import."""
    import probes

    cross_probes = str(cross_root / "probes")
    if cross_probes in probes.__path__:
        probes.__path__.remove(cross_probes)
    probes.__path__.insert(0, cross_probes)
    sys.modules.pop("probes.common", None)

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(1, source_text)

    common = importlib.import_module("probes.common")
    model_common = importlib.import_module("muse_glimmer_common")
    separate = importlib.import_module("helion_muse_glimmer_exact")
    megakernel = importlib.import_module("helion_muse_glimmer_megakernel")

    expected = source_root.resolve()
    for module in (model_common, separate, megakernel):
        if expected not in Path(module.__file__).resolve().parents:
            raise RuntimeError(
                f"loaded the wrong Muse-Glimmer source: {module.__file__}"
            )
    if (cross_root / "probes").resolve() not in Path(common.__file__).resolve().parents:
        raise RuntimeError(f"loaded the wrong benchmark helpers: {common.__file__}")
    return common, model_common, separate, megakernel


def _source_args(args, variant: Variant) -> SimpleNamespace:
    return SimpleNamespace(
        exact_sliding_splits=args.exact_sliding_splits,
        exact_full_splits=args.exact_full_splits,
        split_k=args.split_k,
        down_split_k=variant.down_split_k,
        worker_multiplier=args.worker_multiplier,
        cross_loop_workers=variant.workers,
        num_warps=variant.num_warps,
        kernel_stages=variant.kernel_stages,
        maxnreg=variant.maxnreg,
        qkv_block_n=args.qkv_block_n,
        gate_block_n=args.gate_block_n,
        o_block_n=args.o_block_n,
        o_block_k=args.o_block_k,
        gate_up_block_n=args.gate_up_block_n,
        down_block_n=args.down_block_n,
        projection_block_k=args.projection_block_k,
        ffn_block_k=args.ffn_block_k,
        reduce_block=args.reduce_block,
        activation_block=args.activation_block,
        attention_block=args.attention_block,
        config_mode=variant.config_mode,
        ignore_stored_config=True,
        tune_megakernel=False,
        megakernel_config_path=str(args.output.parent / "unused_configs.json"),
    )


def _exact_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        tune_exact=[],
        qkv_split_k=args.split_k,
        gate_split_k=args.split_k,
        gate_up_split_k=args.split_k,
        down_split_k=args.split_k,
        exact_sliding_splits=args.exact_sliding_splits,
        exact_full_splits=args.exact_full_splits,
    )


def _megakernel_for_source_mode(megakernel, separate, geometry, source_mode):
    if source_mode == "streaming":
        return (
            megakernel.SLIDING_MEGAKERNEL
            if geometry.use_rope
            else megakernel.FULL_MEGAKERNEL
        ), (
            megakernel.SLIDING_SOURCE
            if geometry.use_rope
            else megakernel.FULL_SOURCE
        )

    replacements = {}
    if source_mode in ("whole_input", "whole_norms", "whole_norms_activation"):
        replacements["streaming_rms_norm_offset"] = separate.rms_norm_offset
    if source_mode in ("whole_post", "whole_norms", "whole_norms_activation"):
        replacements["streaming_post_attention_state"] = (
            separate.post_attention_state
        )
    if source_mode in ("whole_final", "whole_norms", "whole_norms_activation"):
        replacements["streaming_final_recompute_residual"] = (
            separate.final_recompute_residual
        )
    if source_mode == "whole_norms_activation":
        replacements["split_aligned_silu_and_mul"] = separate.silu_and_mul
    if not replacements:
        raise ValueError(f"unknown source mode: {source_mode}")
    with mock.patch.multiple(megakernel, **replacements):
        return megakernel._build_megakernel(geometry.use_rope)


def _source_manifest(source_root: Path) -> dict[str, dict[str, str]]:
    names = (
        "muse_glimmer_common.py",
        "helion_muse_glimmer_exact.py",
        "helion_muse_glimmer_layer.py",
        "helion_muse_glimmer_megakernel.py",
        "vllm_muse_glimmer_production_layer.py",
        "muse_glimmer_b200_configs.json",
    )
    return {
        name: {"path": str(source_root / name), "sha256": _sha256(source_root / name)}
        for name in names
    }


def _save_assembly(megakernel, compiled, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = megakernel._helion_binary(compiled)
    paths = {}
    suffixes = {"ttir": ".ttir", "ttgir": ".ttgir", "llir": ".ll", "ptx": ".ptx"}
    for name, assembly in binary.asm.items():
        path = output_dir / f"{stem}{suffixes.get(name, f'.{name}')}"
        if isinstance(assembly, bytes):
            path.write_bytes(assembly)
        else:
            path.write_text(assembly)
        paths[name] = str(path)
    return paths


@torch.inference_mode()
def _run_layer(
    args: argparse.Namespace,
    layer_idx: int,
    variants: list[Variant],
    common,
    model_common,
    separate,
    megakernel,
) -> dict[str, object]:
    shape = model_common.MuseGlimmerShape(
        context=args.context,
        block_size=args.block_size,
    )
    geometry = shape.layer_geometry(layer_idx)
    tensors = model_common.allocate_layer(shape, geometry, args.seed)
    reference = model_common.layer_reference(tensors, shape, geometry)
    layer_dir = args.lowered_dir / geometry.layer_type
    layer_dir.mkdir(parents=True, exist_ok=True)
    compiled_variants: dict[str, object] = {}
    kernel_args_by_name: dict[str, tuple[object, ...]] = {}
    production_cache_by_name: dict[str, torch.Tensor] = {}
    metadata: dict[str, object] = {}
    failures: dict[str, str] = {}

    for variant in variants:
        print("COMPILE_START", geometry.layer_type, variant.name, flush=True)
        try:
            kernel, source = _megakernel_for_source_mode(
                megakernel,
                separate,
                geometry,
                variant.source_mode,
            )
            source_args = _source_args(args, variant)
            kernel_args, production_cache = megakernel._prepare_megakernel(
                tensors,
                shape,
                geometry,
                source_args,
            )
            bound = kernel.bind(kernel_args)
            config = megakernel._config(bound, source_args, geometry)
            lowered = bound.to_triton_code(config, output_origin_lines=True)
            lowered_path = layer_dir / f"{variant.name}.py"
            lowered_path.write_text(lowered)
            source_path = layer_dir / f"{variant.name}_helion.py"
            source_path.write_text(source)
            compiled = bound.compile_config(config)
            outputs = compiled(*kernel_args)
            torch.cuda.synchronize()
            correctness = megakernel._validate(
                outputs,
                reference,
                production_cache,
                tensors,
                shape,
            )
            host_function = bound.host_function
            assert host_function is not None
            dependency_graph = host_function.device_ir.tile_dependency_graph
            assert dependency_graph is not None
            entry = {
                "variant": dataclasses.asdict(variant),
                "config": dict(config),
                "root_count": len(host_function.device_ir.root_ids),
                "root_ids": host_function.device_ir.root_ids,
                "root_block_ids": host_function.device_ir.grid_block_ids,
                "dependency_graph": {
                    "edges": len(dependency_graph.edges),
                    "task_families": len(dependency_graph.task_families),
                    "accesses": len(dependency_graph.accesses),
                    "execution_scopes": len(dependency_graph.execution_scopes),
                },
                "resources": megakernel._helion_resources(compiled),
                "correctness": correctness,
                "lowered": str(lowered_path),
                "helion_source": str(source_path),
                "lowering_summary": common.lowered_triton_summary(lowered),
            }
            if args.save_assembly:
                entry["assembly"] = _save_assembly(
                    megakernel,
                    compiled,
                    args.assembly_dir / geometry.layer_type,
                    variant.name,
                )
            compiled_variants[variant.name] = compiled
            kernel_args_by_name[variant.name] = kernel_args
            production_cache_by_name[variant.name] = production_cache
            metadata[variant.name] = entry
            print(
                "COMPILE_OK",
                geometry.layer_type,
                variant.name,
                json.dumps(entry["resources"], sort_keys=True),
                flush=True,
            )
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            helion.exc.InvalidConfig,
        ) as error:
            failures[variant.name] = str(error)
            print(
                "COMPILE_REJECTED",
                geometry.layer_type,
                variant.name,
                str(error),
                flush=True,
            )

    config_path = args.config_path.resolve()
    standalone_configs = json.loads(config_path.read_text())
    baseline_tensors = model_common.allocate_layer(shape, geometry, args.seed)
    exact = separate.build_exact(
        _exact_args(args),
        baseline_tensors,
        shape,
        geometry,
        standalone_configs,
        config_path,
    )
    standalone_output = exact["launch_exact"]()
    torch.cuda.synchronize()
    separate._assert_close("standalone_exact", standalone_output, reference["output"])

    if args.smoke:
        return {
            "layer_idx": layer_idx,
            "variant": geometry.layer_type,
            "attention_context": geometry.attention_context,
            "standalone_launches": exact["launch_count"],
            "variants": metadata,
            "failures": failures,
        }

    def noop() -> None:
        return None

    graphs = {}
    graph_outputs = {}
    for name, compiled in compiled_variants.items():
        kernel_args = kernel_args_by_name[name]
        graphs[name], graph_outputs[name] = common.capture_with_reset(
            lambda compiled=compiled, kernel_args=kernel_args: compiled(*kernel_args),
            noop,
        )
    standalone_name = "standalone_helion_cudagraph"
    graphs[standalone_name], graph_outputs[standalone_name] = (
        common.capture_with_reset(exact["launch_exact"], noop)
    )
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()

    for name, output in graph_outputs.items():
        if name == standalone_name:
            separate._assert_close("standalone_graph", output, reference["output"])
        else:
            megakernel._validate(
                output,
                reference,
                production_cache_by_name[name],
                tensors,
                shape,
            )

    timings = common.benchmark_graphs_cold_l2(
        {name: (graph.replay, noop) for name, graph in graphs.items()},
        args.repeats,
        flush_mib=256,
        order_seed=args.order_seed + layer_idx,
    )
    ranking = sorted(
        (
            {
                "name": name,
                "median_us": timings[name]["median_us"],
                "resources": metadata[name]["resources"],
            }
            for name in compiled_variants
        ),
        key=lambda item: item["median_us"],
    )
    return {
        "layer_idx": layer_idx,
        "variant": geometry.layer_type,
        "attention_context": geometry.attention_context,
        "standalone_launches": exact["launch_count"],
        "variants": metadata,
        "failures": failures,
        "timings": timings,
        "ranking": ranking,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    cross_root = args.cross_root.resolve()
    common, model_common, separate, megakernel = _load_sources(
        source_root,
        cross_root,
    )
    common.require_idle_visible_gpu()

    helion_path = Path(helion.__file__).resolve()
    if HELION_ROOT.resolve() not in helion_path.parents:
        raise RuntimeError(f"expected helion-clc, loaded {helion_path}")
    if args.repeats % 2:
        raise ValueError("--repeats must be even for the Williams benchmark order")

    variants = [
        Variant(
            mode,
            workers,
            warps,
            stages,
            down_split_k,
            maxnreg or None,
            source_mode,
        )
        for (
            source_mode,
            mode,
            workers,
            warps,
            stages,
            down_split_k,
            maxnreg,
        ) in itertools.product(
            args.source_modes,
            args.config_modes,
            args.workers,
            args.num_warps,
            args.kernel_stages,
            args.down_split_k,
            args.maxnreg,
        )
    ]
    args.output = args.output.resolve()
    args.lowered_dir = args.lowered_dir.resolve()
    args.assembly_dir = args.assembly_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lowered_dir.mkdir(parents=True, exist_ok=True)

    layers = [
        _run_layer(
            args,
            layer_idx,
            variants,
            common,
            model_common,
            separate,
            megakernel,
        )
        for layer_idx in args.layers
    ]
    vllm_root = args.vllm_root.resolve()
    vllm_model_source = vllm_root / "vllm/model_executor/models/muse_glimmer.py"
    vllm_config_source = (
        vllm_root / "vllm/transformers_utils/configs/muse_glimmer.py"
    )
    result: dict[str, object] = {
        "model": "meta-models/Muse-Glimmer-30B",
        "component": "decode_layer_tp1_bf16",
        "device": torch.cuda.get_device_name(),
        "shape": dataclasses.asdict(
            model_common.MuseGlimmerShape(
                context=args.context,
                block_size=args.block_size,
            )
        ),
        "layer_mix": model_common.LAYER_COUNTS,
        "cache_state": "cold_l2",
        "l2_flush_mib": 256,
        "repeats": args.repeats,
        "order_seed": args.order_seed,
        "helion_root": str(HELION_ROOT.resolve()),
        "helion_commit": _git_revision(HELION_ROOT),
        "helion_module": str(helion_path),
        "source_root": str(source_root),
        "sources": _source_manifest(source_root),
        "cross_kernel_helpers": {
            "root": str(cross_root),
            "commit": _git_revision(cross_root),
            "common": str(Path(common.__file__).resolve()),
        },
        "official_vllm": {
            "root": str(vllm_root),
            "commit": _git_revision(vllm_root),
            "model_source": str(vllm_model_source),
            "model_source_sha256": _sha256(vllm_model_source),
            "config_source": str(vllm_config_source),
            "config_source_sha256": _sha256(vllm_config_source),
        },
        "layers": layers,
    }

    if not args.smoke and {layer["variant"] for layer in layers} == {
        "sliding",
        "full",
    }:
        weighted = {}
        timing_names = set.intersection(
            *(set(layer["timings"]) for layer in layers)
        )
        for name in sorted(timing_names):
            weighted[name] = sum(
                layer["timings"][name]["median_us"]
                * model_common.LAYER_COUNTS[layer["variant"]]
                for layer in layers
            )
        result["weighted_52_layer_sum_us"] = weighted

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    source_root = HELION_ROOT.parent / "muse_glimmer"
    cross_root = HELION_ROOT.parent / "helion-cross-kernel"
    vllm_root = HELION_ROOT.parent / "vllm"
    output_root = HELION_ROOT / "probes/muse_glimmer_cross_source_clc"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=source_root)
    parser.add_argument("--cross-root", type=Path, default=cross_root)
    parser.add_argument("--vllm-root", type=Path, default=vllm_root)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=source_root / "muse_glimmer_b200_configs.json",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--exact-sliding-splits", type=int, default=17)
    parser.add_argument("--exact-full-splits", type=int, default=64)
    parser.add_argument("--split-k", type=int, default=16)
    parser.add_argument("--down-split-k", type=int, nargs="+", default=(16,))
    parser.add_argument(
        "--config-modes",
        choices=("default", "matched", "coarse"),
        nargs="+",
        default=("coarse",),
    )
    parser.add_argument(
        "--source-modes",
        choices=(
            "streaming",
            "whole_input",
            "whole_post",
            "whole_final",
            "whole_norms",
            "whole_norms_activation",
        ),
        nargs="+",
        default=("streaming",),
    )
    parser.add_argument("--workers", type=int, nargs="+", default=(296,))
    parser.add_argument("--num-warps", type=int, nargs="+", default=(4,))
    parser.add_argument("--kernel-stages", type=int, nargs="+", default=(2,))
    parser.add_argument("--worker-multiplier", type=int, default=4)
    parser.add_argument(
        "--maxnreg",
        type=int,
        nargs="+",
        default=(0,),
        help="Register caps to sweep; zero uses the compiler default.",
    )
    parser.add_argument("--qkv-block-n", type=int, default=64)
    parser.add_argument("--gate-block-n", type=int, default=128)
    parser.add_argument("--o-block-n", type=int, default=64)
    parser.add_argument("--o-block-k", type=int, default=128)
    parser.add_argument("--gate-up-block-n", type=int, default=128)
    parser.add_argument("--down-block-n", type=int, default=128)
    parser.add_argument("--projection-block-k", type=int, default=128)
    parser.add_argument("--ffn-block-k", type=int, default=256)
    parser.add_argument("--reduce-block", type=int, default=256)
    parser.add_argument("--activation-block", type=int, default=256)
    parser.add_argument("--attention-block", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=1701)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-assembly", action="store_true")
    parser.add_argument("--lowered-dir", type=Path, default=output_root / "lowered")
    parser.add_argument("--assembly-dir", type=Path, default=output_root / "assembly")
    parser.add_argument("--output", type=Path, default=output_root / "benchmark.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
