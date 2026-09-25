"""Cold-autotune and audit selected or full-breadth KDA prefill workloads.

Each invocation owns one shape, partition, and configuration per stage. Run
different pinned configurations in separate processes. All provider timings
cover the immutable FP32-state API, including Helion's state conversions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
from itertools import accumulate
from itertools import starmap
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.config import Config
    from helion.runtime.kernel import Kernel

FLASHINFER_COMMIT = "4d75a33f19aaf48b44d5b1c5dbca33bc1eca5c58"
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Arm:
    name: str
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    details: dict[str, Any]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--library-root", type=Path, required=True)
    result.add_argument("--flashinfer-src", type=Path, required=True)
    result.add_argument(
        "--output", type=Path, required=True, help="New artifact directory"
    )
    result.add_argument(
        "--shape-set", choices=("selected", "breadth"), default="selected"
    )
    result.add_argument("--shape", type=int, required=True, help="Index in --shape-set")
    result.add_argument("--implementations", default="cake,cute,helion")
    result.add_argument(
        "--mode", choices=("autotune", "pinned", "default"), default="autotune"
    )
    result.add_argument(
        "--configs", type=Path, help="A previous result.json or configs.json"
    )
    result.add_argument(
        "--helion-schedule", choices=("staged", "fused"), default="staged"
    )
    result.add_argument(
        "--helion-fused-policy",
        choices=("native_bt16", "centered_bt32"),
        default="native_bt16",
    )
    result.add_argument("--partition", choices=("longest", "all"), default="longest")
    result.add_argument("--state-abi", choices=("bf16", "fp32"), default="bf16")
    result.add_argument(
        "--topology", choices=("auto", "serial", "origin_aux"), default="auto"
    )
    result.add_argument(
        "--native-cute-mode", choices=("auto", "engine", "decomp"), default="engine"
    )
    result.add_argument("--input-seed", type=int, default=0)
    result.add_argument("--autotune-seed", type=int, default=None)
    result.add_argument("--lower-bound", type=float, default=-5.0)
    result.add_argument("--cycles", type=int, default=30)
    result.add_argument("--repeatability-launches", type=int, default=30)
    result.add_argument("--cooldown-margin-c", type=float, default=3.0)
    result.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    result.add_argument("--reference-cache", type=Path)
    result.add_argument(
        "--cache-root",
        type=Path,
        help="Node-local compiler caches (default: output/cache)",
    )
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """Include working-tree edits; archived snapshots need not have .git."""
    paths = sorted((ROOT / "helion").rglob("*.py"))
    paths += [
        Path(__file__).resolve(),
        ROOT / "benchmarks/cute/kda_prefill_kernels.py",
        ROOT / "benchmarks/cute/kda_prefill_staged.py",
        args.library_root / "repro/breadth/compare_kda_prefill.py",
        args.library_root / "helion_kernel_library/_kernels/kda_prefill/kda_prefill.py",
        args.flashinfer_src / "flashinfer/kda_prefill.py",
        args.flashinfer_src / "flashinfer/kda_kernels/kda_chunked_bt16.py",
    ]
    if args.helion_schedule == "fused":
        paths.append(
            ROOT
            / "benchmarks/cute/"
            / (
                "kda_prefill_fused_bt32.py"
                if args.helion_fused_policy == "centered_bt32"
                else "kda_prefill_fused.py"
            )
        )
    return {str(path): file_hash(path) for path in paths}


def capture(
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        output = run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()

    def replay() -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the complete launch closure and all captured storage alive.
        graph.replay()
        return output

    replay.owners = (run, graph, stream)  # type: ignore[attr-defined]
    return replay


def build_cake(
    args: argparse.Namespace,
    compare: ModuleType,
    inputs: tuple[torch.Tensor, ...],
    cu: torch.Tensor,
    cu_cpu: torch.Tensor,
) -> Arm:
    baseline_args = compare.parser().parse_args(
        ["--impl", "flashinfer-cake", "--flashinfer-src", str(args.flashinfer_src)]
    )
    baseline_args.lower_bound = args.lower_bound
    details: dict[str, Any] = {}
    run = compare.make_call(baseline_args, inputs, details, cu, cu_cpu)
    details["config"] = {
        "backend": "cake",
        "prefill_workspace": "explicit RecurrentKDAPrefillWorkspace",
        "seq_order": None,
        "piece_persistent_veto": "Explicit workspace disables the piece-persistent route",
    }
    return Arm("cake", run, details)


def build_native_cute(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    cu: torch.Tensor,
    lengths: list[int],
) -> Arm:
    cutlass = importlib.import_module("cutlass")
    native = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    assert native.__file__ is not None
    q, k, v, gate, beta, a_log, bias, initial = inputs
    mode = None if args.native_cute_mode == "auto" else args.native_cute_mode
    fn = native.compile(
        dtype=cutlass.BFloat16,
        state_dtype=cutlass.Float32,
        gate_dtype=cutlass.BFloat16,
        safe_gate=True,
        gate_lower_bound=args.lower_bound,
        has_state_in=True,
        has_state_out=True,
        mode=mode,
    )
    chunks = [(length + 15) // 16 for length in lengths]
    cu_chunks = torch.tensor(
        [0, *accumulate(chunks)], device=q.device, dtype=torch.int32
    )
    sequence_order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    seq_order = torch.tensor(sequence_order, device=q.device, dtype=torch.int32)
    workspace_bytes = fn.workspace_size_from_total_chunks(
        len(lengths), q.shape[2], sum(chunks), q.device
    )
    workspace = (
        torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
        if workspace_bytes
        else None
    )
    output, final = torch.empty_like(v), torch.empty_like(initial)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        fn(
            q,
            k,
            v,
            gate,
            a_log,
            bias,
            beta,
            cu,
            initial,
            output,
            final,
            workspace,
            torch.cuda.current_stream(q.device).cuda_stream,
            128**-0.5,
            seq_order=seq_order,
            planned_cu_chunks=cu_chunks,
            planned_total_chunks=sum(chunks),
        )
        return output, final

    return Arm(
        "cute",
        capture(launch),
        {
            "source": str(Path(native.__file__).resolve()),
            "source_sha256": file_hash(Path(native.__file__)),
            "config": {
                "mode": mode,
                "state_dtype": "Float32",
                "gate_dtype": "BFloat16",
                "safe_gate": True,
                "sequence_order": sequence_order,
                "planned_total_chunks": sum(chunks),
                "workspace_bytes": workspace_bytes,
            },
            "numerical_contract": "Separate FP32 initial/final state; raw BF16 gates/beta; fused QK norm",
        },
    )


def load_stage_configs(
    args: argparse.Namespace, row: dict[str, Any], details: dict[str, Any]
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if args.configs:
        previous = json.loads(args.configs.read_text())
        if "implementations" in previous:
            if previous["shape"] != row["shape"]:
                raise ValueError("Pinned configuration shape does not match")
            previous = previous["implementations"]["helion"]
        if previous["partition"] != details["partition"] or (
            details["schedule"] == "staged"
            and previous["topology"] != details["topology"]
        ):
            raise ValueError(
                "Pinned configurations require the same partition and topology"
            )
        if previous.get("state_abi", "bf16") != args.state_abi:
            raise ValueError("Pinned configurations require the same state ABI")
        if previous.get("schedule", "staged") != details["schedule"]:
            raise ValueError("Pinned configurations require the same Helion schedule")
        if (
            details["schedule"] == "fused"
            and previous.get("numerical_policy", "native_bt16_bf16_rhs_v1")
            != details["numerical_policy"]
        ):
            raise ValueError("Pinned configurations require the same numerical policy")
        loaded = previous["stage_configs"]
        details["loaded_configs"] = str(args.configs.resolve())
        details["loaded_configs_sha256"] = file_hash(args.configs)
    return loaded


def make_stage_compiler(
    args: argparse.Namespace,
    row: dict[str, Any],
    details: dict[str, Any],
    loaded: dict[str, Any],
) -> Callable[[str, Kernel, tuple[object, ...], Config, str], Callable[..., object]]:
    helion = importlib.import_module("helion")

    def compile_stage(
        name: str,
        kernel: Kernel,
        values: tuple[object, ...],
        seed: Config,
        marker: str,
    ) -> Callable[..., object]:
        stage_dir = args.output / name
        stage_dir.mkdir()
        settings = kernel.settings
        # Settings precede bind(): they participate in bound-kernel identity.
        settings.autotune_effort = "full"
        settings.autotune_random_seed = args.autotune_seed
        settings.autotune_budget_seconds = None
        settings.autotune_max_generations = None
        settings.autotune_log = str(stage_dir / "autotune.log")
        settings.autotune_log_details = True
        settings.autotune_log_search_space = True
        settings.autotune_log_search_space_path = str(stage_dir / "search-space.json")
        settings.autotune_accuracy_check = True
        settings.autotune_ignore_errors = True
        settings.autotune_seed_configs = [seed]
        baseline_holder: list[Callable[..., object]] = []

        def seed_baseline(*arguments: object) -> object:
            return baseline_holder[0](*arguments)

        # The generic conservative fragment config can request a schedule the
        # whole-root lowering does not implement. Validate candidates against
        # the explicit carrier seed instead, with mutation detection and all
        # normal autotuner accuracy checks still enabled.
        settings.autotune_baseline_fn = seed_baseline
        normalized = kernel.normalize_args(*values)
        bound = kernel.bind(normalized)
        seed = bound.config_spec.normalized_config(seed)
        if marker not in bound.to_code(seed):
            raise RuntimeError(
                f"{name}: seed did not use the specialized CuTe lowering"
            )
        baseline_holder.append(bound.compile_config(seed, allow_print=False))
        started = time.monotonic()
        if args.mode == "autotune":
            print(
                f"Autotuning {name}: full effort, seed={args.autotune_seed}", flush=True
            )
            config = bound.autotune(normalized, force=True)
        elif args.mode == "pinned":
            config = bound.config_spec.normalized_config(helion.Config(**loaded[name]))
        else:
            config = seed
        source = bound.to_code(config)
        if marker not in source:
            raise RuntimeError(
                f"{name}: winner did not use the specialized CuTe lowering"
            )
        compiled = bound.compile_config(config, allow_print=False)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        compiled_hash = bound.env.backend.generated_source_hash(compiled)
        if compiled_hash != source_hash:
            raise RuntimeError(
                f"{name}: compiled source identity differs from saved source"
            )
        (stage_dir / "generated.py").write_text(source)
        details["stage_configs"][name] = dict(config.config)
        details["stages"][name] = {
            "seconds": time.monotonic() - started,
            "seed_config": dict(seed.config),
            "source_sha256": source_hash,
            "compiled_source_identity_verified": True,
            "required_source_marker": marker,
            "autotune_random_seed": args.autotune_seed,
            "full_autotune": args.mode == "autotune",
        }
        write_json(args.output / "configs.json", details)
        write_json(args.output / "progress.json", row)
        # Return the immutable compiled function, not a memoized BoundKernel.
        return compiled

    return compile_stage


def build_helion(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    offsets: list[int],
    row: dict[str, Any],
) -> Arm:
    kernels = importlib.import_module("benchmarks.cute.kda_prefill_kernels")
    staged = importlib.import_module("benchmarks.cute.kda_prefill_staged")
    q, k, v, gate, beta, a_log, bias, initial = inputs
    state = torch.empty_like(initial, dtype=torch.bfloat16)
    output, final = torch.empty_like(v), torch.empty_like(initial)
    output.zero_()
    state.copy_(initial)
    if args.partition == "all":
        metadata = staged.materialize_kda_group_metadata(
            staged.kda_group_host_metadata(offsets, (0, len(offsets) - 1)), q.device
        )
        group = staged.KdaStagedGroup(
            metadata,
            staged.allocate_kda_factor_workspace(
                q.shape[2], metadata.host.total_chunks, q.device
            ),
        )
        resources = staged.KdaStagedResources(
            group, (), staged.CuteOriginAuxDag.create(q.device)
        )
    else:
        resources = staged.create_staged_kda_resources(
            offsets, heads=q.shape[2], device=q.device
        )
    topology = args.topology
    if topology == "auto":
        topology = "origin_aux" if len(resources.auxiliary) == 1 else "serial"
    if topology == "origin_aux" and len(resources.auxiliary) > 1:
        raise ValueError("origin_aux requires at most one auxiliary range")
    groups = (resources.critical, *resources.auxiliary)
    details: dict[str, Any] = {
        "mode": args.mode,
        "schedule": "staged",
        "partition": args.partition,
        "state_abi": args.state_abi,
        "topology": topology,
        "sequence_groups": [group.metadata.host.sequence_range for group in groups],
        "workspace_bytes": sum(group.workspace.storage.numel() for group in groups),
        "stage_configs": {},
        "stages": {},
        "numerical_contract": {
            "initial_state": "FP32 converted to private BF16 in timed graph",
            "running_state": "DV2 FP32 TMEM; DV4 BF16 registers",
            "final_state": "BF16 converted to separate FP32 in timed graph",
            "factors": "BF16 kd/qd/ak/aq; FP32 g_total",
            "inverse": "FP16 operands/intermediates with FP32 accumulation",
        },
    }
    if args.state_abi == "fp32":
        details["numerical_contract"].update(
            {
                "initial_state": "Immutable FP32 input, loaded directly by recurrence",
                "running_state": "Authoritative FP32 TMEM or registers; BF16 matrix copies",
                "final_state": "Separate FP32 output written directly by recurrence",
                "residual": "BF16(V - BF16(Kd @ BF16(S).T)), zero for invalid tokens",
            }
        )
    row["implementations"]["helion"] = details
    loaded = load_stage_configs(args, row, details)
    compiled_prepare, compiled_recurrence = [], []
    prepare_arguments, recurrence_arguments = [], []

    compile_stage = make_stage_compiler(args, row, details, loaded)

    for index, group in enumerate(groups):
        metadata, workspace = group.metadata, group.workspace
        begin, end = metadata.host.sequence_range
        prepare_args = (
            q,
            k,
            gate,
            beta,
            a_log,
            bias,
            metadata.cu_seqlens,
            metadata.cu_chunks,
            metadata.chunk_to_seq,
            workspace.kd,
            workspace.qd,
            workspace.ak,
            workspace.aq,
            workspace.g_total,
            None,
            args.lower_bound * kernels.LOG2_E,
        )
        prepare = compile_stage(
            f"group{index}_prepare",
            kernels.kda_chunk_prepare,
            prepare_args,
            kernels.KDA_PREPARE_CONFIG,
            "from helion._compiler.cute.chunk_prepare_split_alias_device import",
        )
        compiled_prepare.append(prepare)
        prepare_arguments.append(prepare_args)
        # Recurrence tuning must consume real factors, not uninitialized storage.
        prepare(*prepare_args)
        state.copy_(initial)
        state_args = (
            (initial[begin:end], final[begin:end])
            if args.state_abi == "fp32"
            else (state[begin:end],)
        )
        recurrence_args = (
            workspace.kd,
            workspace.qd,
            workspace.ak,
            workspace.aq,
            workspace.g_total,
            v,
            output,
            *state_args,
            metadata.cu_seqlens,
            metadata.cu_chunks,
            kernels.DK**-0.5,
        )
        recurrence_arguments.append(recurrence_args)
        compiled_recurrence.append(
            compile_stage(
                f"group{index}_recurrence",
                kernels.kda_chunk_recurrence_fp32
                if args.state_abi == "fp32"
                else kernels.kda_chunk_recurrence,
                recurrence_args,
                kernels.KDA_RECURRENCE_CONFIG,
                "from helion._compiler.cute.chunk_recurrence_",
            )
        )

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        if args.state_abi == "fp32":

            def prepare_group(index: int) -> None:
                compiled_prepare[index](*prepare_arguments[index])

            def recurrence_group(index: int) -> None:
                compiled_recurrence[index](*recurrence_arguments[index])

            def auxiliary_branch() -> None:
                for index in range(1, len(groups)):
                    prepare_group(index)
                    recurrence_group(index)

            if topology == "serial" or not resources.auxiliary:
                for index in range(len(groups)):
                    prepare_group(index)
                    recurrence_group(index)
            else:
                resources.dag.launch(
                    lambda: prepare_group(0),
                    lambda: recurrence_group(0),
                    auxiliary_branch,
                )
            return output, final
        state.copy_(initial)
        staged.launch_staged_kda_prefill(
            resources,
            q=q,
            k=k,
            gate=gate,
            beta_logits=beta,
            a_log=a_log,
            dt_bias=bias,
            values=v,
            output=output,
            state=state,
            scale=kernels.DK**-0.5,
            gate_scale_log2=args.lower_bound * kernels.LOG2_E,
            topology=topology,
            prepare_kernels=compiled_prepare,
            recurrence_kernels=compiled_recurrence,
        )
        final.copy_(state)
        return output, final

    return Arm("helion", capture(launch), details)


def build_helion_fused(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    cu: torch.Tensor,
    row: dict[str, Any],
) -> Arm:
    helion = importlib.import_module("helion")
    if args.helion_fused_policy == "centered_bt32":
        kernels = importlib.import_module("benchmarks.cute.kda_prefill_fused_bt32")
        kernel = kernels.kda_prefill_native_math_bt32
        numerical_policy = "centered_bt32_fp32_rhs_v2"
        chunk_size = 32
    else:
        kernels = importlib.import_module("benchmarks.cute.kda_prefill_fused")
        kernel = kernels.kda_prefill_native_math
        numerical_policy = "native_bt16_bf16_rhs_v1"
        chunk_size = 16
    q, k, v, gate, beta, a_log, bias, initial = inputs
    output, final = torch.empty_like(v), torch.empty_like(initial)
    values = (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        initial,
        output,
        final,
        cu,
        128**-0.5,
        args.lower_bound * kernels.LOG2_E,
    )
    details: dict[str, Any] = {
        "mode": args.mode,
        "schedule": "fused",
        "partition": "all",
        "state_abi": "fp32",
        "topology": "serial",
        "sequence_groups": [[0, initial.shape[0]]],
        "workspace_bytes": 0,
        "numerical_policy": numerical_policy,
        "chunk_size": chunk_size,
        "stage_configs": {},
        "stages": {},
        "numerical_contract": (
            {
                "initial_state": "Immutable FP32 input loaded directly",
                "running_state": "Authoritative FP32 TMEM; BF16 matrix copies",
                "final_state": "Separate FP32 output written directly",
                "factors": "On-chip BF16 factors with centered BT32 rounding",
                "inverse": "FP16 operands/intermediates with FP32 accumulation",
                "residual": "FP32 beta times fused FP32 residual, then BF16",
            }
            if numerical_policy == "centered_bt32_fp32_rhs_v2"
            else {
                "initial_state": "Immutable FP32 input loaded directly",
                "running_state": "Authoritative FP32 TMEM; BF16 matrix copies",
                "final_state": "Separate FP32 output written directly",
                "factors": "On-chip BF16 factors with native BT16 rounding",
                "inverse": "FP16 operands/intermediates with FP32 accumulation",
                "residual": "BF16(BF16(beta) * BF16(V - BF16(Kd @ BF16(S))))",
            }
        ),
    }
    row["implementations"]["helion"] = details
    loaded = load_stage_configs(args, row, details)
    compile_stage = make_stage_compiler(args, row, details, loaded)
    compiled = compile_stage(
        "fused_prefill",
        kernel,
        values,
        helion.Config(
            block_sizes=[64],
            num_warps=4,
            num_stages=2,
            indexing="pointer",
            pid_type="flat",
        ),
        "chunk_prefill_sm100",
    )
    selected = details["stage_configs"]["fused_prefill"]
    device_schedule = selected.get("cute_chunk_prefill_schedule", "single")
    prefix_count = {"single": 0, "prefix_tail_2": 2, "prefix_tail_4": 4}[
        device_schedule
    ]
    groups = min(4, initial.shape[0]) if prefix_count else 1
    precompute_order = (
        selected.get("cute_chunk_prefill_task_order") == "longest_first_precompute"
    )
    details.update(
        {
            "device_schedule": device_schedule,
            "topology": "parallel_prefix_tail" if prefix_count else "serial",
            "sequence_groups": [
                [
                    group * initial.shape[0] // groups,
                    (group + 1) * initial.shape[0] // groups,
                ]
                for group in range(groups)
            ],
            "workspace_bytes": (
                2 * initial.numel() * initial.element_size() if prefix_count else 0
            )
            + (4 * initial.shape[0] if precompute_order else 0),
            "workspace_scope": "one launch stream/capture context",
            "device_launches": groups * (prefix_count + 1) + int(precompute_order),
        }
    )
    write_json(args.output / "configs.json", details)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        compiled(*values)
        return output, final

    return Arm("helion", capture(launch), details)


def check_arm(
    arm: Arm,
    compare: ModuleType,
    inputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, torch.Tensor],
    flush: torch.Tensor,
    launches: int,
) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    try:
        errors["reference"] = compare.audit(arm.run, expected, inputs)
    except AssertionError as error:
        errors["reference"] = {"error": str(error)}
    snapshots = tuple(value.clone() for value in inputs)
    original = tuple(value.clone() for value in arm.run())
    changed = [0, 0]
    for _ in range(launches):
        flush.add_(1)
        actual = arm.run()
        for index, (value, baseline) in enumerate(zip(actual, original, strict=True)):
            changed[index] += int(not torch.equal(value, baseline))
    immutable = all(starmap(torch.equal, zip(inputs, snapshots, strict=True)))
    errors["cold_repeatability"] = {
        "launches": launches,
        "output_changes": changed[0],
        "state_changes": changed[1],
    }
    errors["input_immutability"] = immutable
    errors["valid"] = (
        "error" not in errors["reference"] and not any(changed) and immutable
    )
    return errors


def measure(
    arms: list[Arm],
    args: argparse.Namespace,
    flush: torch.Tensor,
    utils: ModuleType,
    target_temp: float,
) -> dict[str, Any]:
    """CUDA events surround exactly one complete graph call, for every arm."""
    for arm in arms:
        for _ in range(5):
            flush.add_(1)
            arm.run()
    cooldown = utils.wait_for_cooldown(target_temp, args.cooldown_timeout_s)
    observations: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {arm.name: [] for arm in arms}
    for cycle in range(args.cycles):
        order = arms if cycle % 2 == 0 else list(reversed(arms))
        # Two arms produce ABAB/BABA; extra baselines remain balanced too.
        order = order * 2
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in order
        ]
        for arm, (start, end) in zip(order, events, strict=True):
            flush.add_(1)
            start.record()
            arm.run()
            end.record()
        events[-1][1].synchronize()
        for position, (arm, (start, end)) in enumerate(zip(order, events, strict=True)):
            elapsed = start.elapsed_time(end)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid event timing: {elapsed}")
            samples[arm.name].append(elapsed)
            observations.append(
                {
                    "cycle": cycle,
                    "position": position,
                    "implementation": arm.name,
                    "ms": elapsed,
                }
            )
    return {
        "timer": "torch.cuda.Event around one complete CUDA-graph replay",
        "ordering": "alternating forward/reverse order, twice per cycle (ABAB/BABA for two providers)",
        "cold_l2_bytes": flush.numel(),
        "state_reset": "Provider implements immutable initial state; all required copies are inside timing",
        "cooldown": cooldown,
        "samples_ms": samples,
        "median_ms": {
            name: statistics.median(values) for name, values in samples.items()
        },
        "observations": observations,
    }


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.library_root = args.library_root.resolve()
    args.flashinfer_src = args.flashinfer_src.resolve()
    implementations = args.implementations.split(",")
    if (
        not implementations
        or set(implementations) - {"cake", "cute", "helion"}
        or len(set(implementations)) != len(implementations)
    ):
        raise ValueError(
            "--implementations must be a comma-separated subset of cake,cute,helion"
        )
    if args.helion_schedule == "fused" and (
        args.partition != "all"
        or args.state_abi != "fp32"
        or args.topology == "origin_aux"
    ):
        raise ValueError(
            "Fused schedule requires --partition all --state-abi fp32 and serial topology"
        )
    if (args.mode == "pinned") != (args.configs is not None):
        raise ValueError("--mode pinned and --configs must be specified together")
    if args.cycles < 2 or args.cycles % 2 or args.repeatability_launches < 1:
        raise ValueError(
            "Even --cycles >= 2 and positive --repeatability-launches required"
        )
    if "artifacts" not in args.output.parts:
        raise ValueError("--output must be under an artifacts directory")
    args.output.mkdir(parents=True, exist_ok=False)
    cache_root = (args.cache_root or args.output / "cache").resolve()
    args.autotune_seed = (
        secrets.randbits(31) if args.autotune_seed is None else args.autotune_seed
    )
    os.environ.update(
        {
            "HELION_BACKEND": "cute",
            "HELION_SKIP_CACHE": "1",
            "HELION_AUTOTUNE_RANDOM_SEED": str(args.autotune_seed),
            "HELION_AUTOTUNE_EFFORT": "full",
            "HELION_BENCHMARK_CUDAGRAPH": "1",
            "HELION_CACHE_DIR": str(cache_root / "helion"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "CUTE_DSL_CACHE_DIR": str(cache_root / "cute"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
            "FLA_FLASH_KDA": "0",
            "FLA_TILELANG": "0",
        }
    )
    # These providers are imported only after selecting the frozen source trees.
    sys.path[:0] = [str(ROOT), str(args.library_root), str(args.flashinfer_src)]
    # Some environments install an unrelated regular package named benchmarks,
    # which otherwise takes precedence over this repository's namespace package.
    benchmark_namespace = ModuleType("benchmarks")
    benchmark_namespace.__path__ = [str(ROOT / "benchmarks")]
    benchmark_namespace.__package__ = "benchmarks"
    sys.modules["benchmarks"] = benchmark_namespace
    compare = importlib.import_module("repro.breadth.compare_kda_prefill")
    utils = importlib.import_module("benchmarks.cute.kda_benchmark_utils")
    commit = subprocess.check_output(
        ["git", "-C", str(args.flashinfer_src), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != FLASHINFER_COMMIT:
        raise ValueError(
            f"FlashInfer must be v0.7.0 ({FLASHINFER_COMMIT}), got {commit}"
        )
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.flashinfer_src),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"FlashInfer source has tracked modifications: {dirty}")
    shapes = compare.SHAPES if args.shape_set == "breadth" else compare.SELECTED_SHAPES
    if not 0 <= args.shape < len(shapes):
        raise ValueError(
            f"--shape must be between 0 and {len(shapes) - 1} for {args.shape_set}"
        )
    shape = shapes[args.shape]
    lengths = compare.make_sequence_lengths(
        shape["batch"], shape["seq_len"], shape["layout"]
    )
    offsets = [0, *accumulate(lengths)]
    source_before = source_fingerprint(args)
    write_json(args.output / "source-manifest.json", source_before)
    startup_temp = utils._gpu_field("temperature.gpu")
    if startup_temp is None:
        raise RuntimeError("Cannot read startup GPU temperature")
    row: dict[str, Any] = {
        "shape_index": args.shape,
        "shape_set": args.shape_set,
        "shape": shape,
        "sequence_lengths": lengths,
        "total_tokens": sum(lengths),
        "input_seed": args.input_seed,
        "flashinfer_commit": commit,
        "gpu": utils.gpu_info(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "autotune_seed": args.autotune_seed,
        "autotuner": os.environ.get("HELION_AUTOTUNER", "default"),
        "mode": args.mode,
        "cold_full_autotune": args.mode == "autotune" and "helion" in implementations,
        "cache_bypassed": True,
        "cache_root": str(cache_root),
        "startup_temp_c": startup_temp,
        "implementations": {},
        "status": "RUNNING",
    }
    write_json(args.output / "progress.json", row)
    try:
        with torch.no_grad():
            inputs = compare.make_inputs(**shape, seed=args.input_seed)
            row["logical_io_bytes"] = sum(
                value.numel() * value.element_size() for value in inputs
            ) + sum(
                value.numel() * value.element_size() for value in (inputs[2], inputs[7])
            )
            cu_cpu = torch.tensor(offsets, dtype=torch.int64)
            cu = cu_cpu.cuda()
            arms = []
            for name in implementations:
                if name == "cake":
                    arm = build_cake(args, compare, inputs, cu, cu_cpu)
                elif name == "cute":
                    arm = build_native_cute(args, inputs, cu, lengths)
                else:
                    arm = (
                        build_helion_fused(args, inputs, cu, row)
                        if args.helion_schedule == "fused"
                        else build_helion(args, inputs, offsets, row)
                    )
                arms.append(arm)
                row["implementations"][name] = arm.details
                write_json(args.output / "progress.json", row)
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "shape": shape,
                        "seed": args.input_seed,
                        "lower_bound": args.lower_bound,
                        "reference_source": source_before[
                            str(
                                args.library_root
                                / "helion_kernel_library/_kernels/kda_prefill/kda_prefill.py"
                            )
                        ],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            reference_path = (
                args.reference_cache / f"{cache_key}.pt"
                if args.reference_cache
                else None
            )
            started = time.monotonic()
            if reference_path is not None and reference_path.exists():
                expected = torch.load(
                    reference_path, weights_only=True, map_location="cuda"
                )
                row["reference_cache_hit"] = True
            else:
                expected = compare.reference(inputs, args.lower_bound, cu_seqlens=cu)
                if reference_path is not None:
                    reference_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(expected, reference_path)
                row["reference_cache_hit"] = False
            torch.cuda.synchronize()
            row["reference_seconds"] = time.monotonic() - started
            row["reference"] = "Breadth FP64 sequential reference (untimed)"
            flush = torch.empty(
                2 * utils.l2_cache_bytes(torch.cuda.get_device_properties(0)),
                dtype=torch.uint8,
                device="cuda",
            )
            valid = []
            for arm in arms:
                arm.details["correctness_before"] = check_arm(
                    arm,
                    compare,
                    (*inputs, cu),
                    expected,
                    flush,
                    args.repeatability_launches,
                )
                if arm.details["correctness_before"]["valid"]:
                    valid.append(arm)
                write_json(args.output / "progress.json", row)
            if valid:
                row["timing"] = measure(
                    valid, args, flush, utils, startup_temp + args.cooldown_margin_c
                )
                row["timing"]["logical_io_gb_s"] = {
                    name: row["logical_io_bytes"] / (milliseconds * 1e6)
                    for name, milliseconds in row["timing"]["median_ms"].items()
                }
                row["timing"]["logical_io_note"] = (
                    "One read per input and one write per output; excludes "
                    "intermediate workspace traffic and repeated reads"
                )
            for arm in valid:
                arm.details["correctness_after"] = check_arm(
                    arm,
                    compare,
                    (*inputs, cu),
                    expected,
                    flush,
                    args.repeatability_launches,
                )
            valid_names = [
                arm.name for arm in valid if arm.details["correctness_after"]["valid"]
            ]
            row["valid_implementations"] = valid_names
            if "helion" in valid_names:
                medians = row["timing"]["median_ms"]
                baselines = [name for name in valid_names if name != "helion"]
                if baselines:
                    best = min(baselines, key=medians.__getitem__)
                    row["best_baseline"] = best
                    row["ratio"] = medians[best] / medians["helion"]
            source_after = source_fingerprint(args)
            if source_before != source_after:
                raise RuntimeError("Source files changed during this run")
            row["status"] = "PASS" if len(valid_names) == len(arms) else "FAIL"
    except Exception:
        row["status"] = "ERROR"
        row["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output / "result.json", row)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "status",
                        "shape_index",
                        "ratio",
                        "valid_implementations",
                    )
                    if key in row
                }
            ),
            flush=True,
        )
    if row["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
