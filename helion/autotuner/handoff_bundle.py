"""Package a confirmed autotune result for native source optimization."""

from __future__ import annotations

import ast
import dataclasses
import datetime
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

from .._hardware import get_hardware_info
from ..language.constexpr import ConstExpr
from .accuracy import is_fp8_dtype
from .benchmark_provider import _materialize_multi_shape_config
from .benchmark_provider import _MultiShapeAutotuneArgs
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .handoff_agent import HandoffAgentResult
from .handoff_agent import run_agent_rounds
from .handoff_evaluation import HandoffEvaluation
from .handoff_evaluation import _alias_signature
from .handoff_evaluation import _artifact_path
from .handoff_evaluation import _device_context
from .handoff_evaluation import evaluate_handoff
from .handoff_inputs import _clone_inputs
from .handoff_inputs import _save_inputs
from .handoff_prompt import build_handoff_prompt

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.settings import Settings
    from .base_cache import AutotuneCacheBase
    from .base_search import BaseSearch
    from .handoff import HandoffCandidate
    from .handoff import HandoffMeasurement
    from .handoff import HandoffPoint
    from .logger import AutotuningLogger


@dataclasses.dataclass(frozen=True)
class HandoffBundle:
    """An editable native-source workspace, independent of its original kernel.

    Construct with an existing directory to reopen a saved bundle. Agent adapters
    receive this object and return complete replacement files keyed by their
    relative source paths; they need no Helion configuration or BoundKernel.
    """

    directory: Path

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.directory / "manifest.json").read_text())

    @property
    def prompt(self) -> str:
        return (self.directory / "prompt.md").read_text()

    @property
    def sources(self) -> dict[str, str]:
        directory = self.directory.resolve()
        return {
            case["source"]: _artifact_path(directory, case["source"]).read_text()
            for case in self.manifest["cases"]
        }

    def evaluate(
        self,
        *,
        repetitions: int = 5,
        timeout: float = 120,
        deadline: float | None = None,
    ) -> HandoffEvaluation:
        """Validate and measure the current native sources, saving the result."""
        return evaluate_handoff(
            self.directory, repetitions=repetitions, timeout=timeout, deadline=deadline
        )

    def run_agent(
        self,
        agent: Callable[[HandoffBundle], Mapping[str, str]],
        *,
        repetitions: int = 5,
        timeout: float = 120,
    ) -> HandoffEvaluation:
        """Apply one agent proposal and evaluate it; round selection is separate."""
        paths = set(self.sources)
        replacements = agent(self)
        if not replacements or not replacements.keys() <= paths:
            raise ValueError(
                "Agent must replace one or more existing native source files"
            )
        if any(not isinstance(source, str) for source in replacements.values()):
            raise TypeError("Agent replacements must contain complete source strings")
        directory = self.directory.resolve()
        writes = [
            (_artifact_path(directory, path), source)
            for path, source in replacements.items()
        ]
        for path, source in writes:
            path.write_text(source)
        return self.evaluate(repetitions=repetitions, timeout=timeout)

    def run_agent_rounds(
        self,
        agent: Callable[[HandoffBundle], Mapping[str, str]] | None = None,
        *,
        budget_seconds: float,
        repetitions: int = 5,
        timeout: float = 120,
        log: AutotuningLogger | None = None,
    ) -> HandoffAgentResult:
        """Run bounded proposals, retaining only validated, noise-separated gains.

        Set a wall-clock budget. By default, one continuous agent CLI session
        submits candidates and receives feedback until it exits or time runs out.
        An explicit synchronous callback must honor
        the monotonic deadline exposed in its prompt.
        Sources, feedback, and decisions are saved under ``agent_runs/``.
        """
        return run_agent_rounds(
            self,
            agent,
            budget_seconds=budget_seconds,
            repetitions=repetitions,
            timeout=timeout,
            log=log,
        )


def _input_metadata(args: Sequence[object]) -> object:
    storage_groups: dict[object, int] = {}

    def describe(value: object) -> object:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()._cdata
            group = storage_groups.setdefault(storage, len(storage_groups))
            return {
                "type": "Tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "storage_offset": value.storage_offset(),
                "storage_group": group,
                "requires_grad": value.requires_grad,
            }
        if isinstance(value, float) and not math.isfinite(value):
            return {"type": "float", "value": str(value)}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"type": type(value).__name__, "value": value}
        if isinstance(value, (torch.dtype, torch.device)):
            return {"type": type(value).__name__, "value": str(value)}
        raise TypeError(f"Unsupported handoff input type: {type(value).__name__}")

    return tree_map(describe, args)


def _finite_json(value: object) -> object:
    """Represent rejected/nonfinite measurements as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    return value


def _autotune_evidence(point: HandoffPoint) -> dict[str, Any]:
    def observation(value: HandoffCandidate | HandoffMeasurement) -> dict[str, Any]:
        result = dataclasses.asdict(value)
        result["config"] = dict(result["config"])
        return result

    return cast(
        "dict[str, Any]",
        _finite_json(
            {
                "reason": point.reason,
                "config": dict(point.config),
                "progress": dataclasses.asdict(point.progress),
                "finalists": [observation(value) for value in point.finalists],
                "measurements": [observation(value) for value in point.measurements],
            }
        ),
    )


def _tolerances(
    settings: Settings,
    output: object,
    inputs: Sequence[object],
    post_args: Sequence[object],
) -> tuple[float, float]:
    # Match autotuning's strict FP8 default; unchanged input dtypes should not
    # weaken the checks on an FP8 output.
    outputs, _ = tree_flatten(output)
    before, _ = tree_flatten(inputs)
    after, _ = tree_flatten(post_args)
    dtypes = {value.dtype for value in outputs if isinstance(value, torch.Tensor)}
    for original, result in zip(before, after, strict=True):
        if isinstance(original, torch.Tensor) and isinstance(result, torch.Tensor):
            lhs = (
                original.view(torch.uint8) if is_fp8_dtype(original.dtype) else original
            )
            rhs = result.view(torch.uint8) if is_fp8_dtype(result.dtype) else result
            if not torch.equal(lhs, rhs):
                dtypes.add(result.dtype)
    atol, rtol = settings.autotune_baseline_atol, settings.autotune_baseline_rtol
    if atol is None and rtol is None and dtypes and all(map(is_fp8_dtype, dtypes)):
        return 0.0, 0.0
    return atol if atol is not None else 0.01, rtol if rtol is not None else 0.01


def _dependencies(source: str) -> list[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots - sys.stdlib_module_names)


_EVALUATE_SCRIPT = '''"""Validate and benchmark this bundle's current native sources."""
import argparse
import dataclasses
import json
from pathlib import Path

from helion.autotuner import evaluate_handoff

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    result = evaluate_handoff(
        Path(__file__).resolve().parent,
        repetitions=args.repetitions,
        timeout=args.timeout,
    )
    print(json.dumps(dataclasses.asdict(result), indent=2, allow_nan=False))
    raise SystemExit(0 if result.ok else 1)
'''


def build_handoff(
    autotuner: BaseSearch | AutotuneCacheBase,
    point: HandoffPoint,
    directory: str | Path,
    *,
    reference_fn: Callable[..., object] | None = None,
    repetitions: int = 5,
    timeout: float = 120,
    timing: str | None = None,
) -> HandoffBundle:
    """Export a selected kernel, freeze its workload, and verify a native baseline.

    ``directory`` must be new. Each bound shape gets a standalone module with
    its original entrypoint. References come from ``reference_fn``, then the
    configured autotune baseline, or otherwise the confirmed original kernel.
    The confirmed original defines input mutations and output aliases; custom
    references supply output values. Evaluation needs Helion, but the
    exported kernels and agent source replacements do not.

    CUDA bundles default to GPU-only graph timing; ``timing`` can explicitly
    select a native policy. Autotuning retains its own timing policy.
    Custom accuracy/arbitrary benchmark callbacks and distributed workloads are
    unsupported. Unsupported standalone export paths raise the backend's error.
    The resulting bundle contains no BoundKernel or search object.
    """
    # Runtime imports avoid the kernel -> autotuner package import cycle.
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import OutputCodeOptions
    from ..runtime.kernel import _CachedBoundKernel
    from .base_cache import AutotuneCacheBase

    search = (
        autotuner.autotuner if isinstance(autotuner, AutotuneCacheBase) else autotuner
    )
    multi = search.args if isinstance(search.args, _MultiShapeAutotuneArgs) else None
    cases = multi.cases if multi else ((search.kernel, tuple(search.args)),)
    if timing is not None and timing not in ("cuda_graph", "cuda_event", "wall_clock"):
        raise ValueError(f"Unsupported handoff timing mode: {timing}")
    # Explicit handoff can retain the disk runner passed to a search even after
    # frontend access materialized its delegate. Export the ordinary binding.
    cases = tuple(
        (
            bound._materialize(tuple(args))
            if isinstance(bound, _CachedBoundKernel)
            else bound,
            args,
        )
        for bound, args in cases
    )
    for bound, _ in cases:
        if not isinstance(bound, BoundKernel):
            raise TypeError(
                "build_handoff requires a BoundKernel with standalone export"
            )
        if bound.env.process_group_name is not None:
            raise NotImplementedError("Distributed handoff evaluation is not supported")
        if bound.settings.autotune_baseline_accuracy_check_fn is not None:
            raise NotImplementedError(
                "Handoff does not support custom accuracy callbacks"
            )
        if bound.settings.autotune_benchmark_fn is not None:
            raise NotImplementedError(
                "Handoff does not support custom benchmark callbacks"
            )

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = (
        _materialize_multi_shape_config(search.config_spec, point.config)
        if multi
        else point.config
    )
    case_metadata = []
    for index, (bound, args) in enumerate(cases):
        assert isinstance(bound, BoundKernel)
        default_timing = (
            "cuda_graph"
            if bound.env.device.type == "cuda" and torch.version.hip is None
            else "wall_clock"
            if bound.env.backend.get_do_bench() is do_bench_generic
            or bound.env.device.type not in ("cuda", "xpu")
            else "cuda_event"
        )
        case_timing = timing or default_timing
        case_dir = directory / f"case_{index}"
        case_dir.mkdir()
        source = bound.to_code(
            config, options=OutputCodeOptions(allow_helion_deps=False)
        )
        (case_dir / "kernel.py").write_text(source)
        (case_dir / "original.py").write_text(source)
        inputs = tuple(
            tree_map(
                lambda value: value.value if isinstance(value, ConstExpr) else value,
                args,
                is_leaf=lambda value: isinstance(value, ConstExpr),
            )
        )
        inputs = _clone_inputs(inputs)
        _save_inputs(inputs, case_dir / "inputs.pt")
        reference = reference_fn or bound.settings.autotune_baseline_fn
        reference_kind = (
            "reference_fn" if reference_fn is not None else "autotune_baseline_fn"
        )
        post_args = _clone_inputs(inputs)
        with _device_context(inputs), torch.no_grad():
            output = bound.compile_config(config)(*post_args)
            synchronize_device()
            alias_signature = _alias_signature((output, post_args))
            if reference is None:
                reference_kind = "confirmed_original"
            else:
                reference_args = _clone_inputs(inputs)
                output = reference(*reference_args)
                synchronize_device()
        _save_inputs((output, post_args), case_dir / "reference.pt")
        atol, rtol = _tolerances(bound.settings, output, inputs, post_args)
        case_metadata.append(
            {
                "source": f"case_{index}/kernel.py",
                "entrypoint": bound.kernel.name,
                "backend": bound.env.backend_name,
                "hardware": (
                    dataclasses.asdict(get_hardware_info(bound.env.device))
                    if bound.env.device.type in ("cuda", "xpu", "tpu")
                    else {
                        "device_kind": bound.env.device.type,
                        "runtime_version": torch.__version__,
                    }
                ),
                "input_metadata": _input_metadata(inputs),
                "dependencies": _dependencies(source),
                "static_shapes": bound.settings.static_shapes,
                "reference_kind": reference_kind,
                "alias_signature": alias_signature,
                "timing": case_timing,
                "atol": atol,
                "rtol": rtol,
                "scale_atol": bound.settings.autotune_baseline_atol is None,
                "inputs": f"case_{index}/inputs.pt",
                "reference": f"case_{index}/reference.pt",
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": {
            "aggregation": multi.aggregation if multi else "geomean",
            "unit": point.objective_unit,
            "reference_latencies": point.reference_latencies,
        },
        "cases": case_metadata,
        "autotune": _autotune_evidence(point),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (directory / "evaluate.py").write_text(_EVALUATE_SCRIPT)
    bundle = HandoffBundle(directory)
    baseline = bundle.evaluate(repetitions=repetitions, timeout=timeout)
    if not baseline.ok:
        raise RuntimeError(
            f"Standalone handoff baseline failed: {baseline.directory}/evaluation.json"
        )
    manifest["baseline_evaluation"] = dataclasses.asdict(baseline)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (directory / "prompt.md").write_text(build_handoff_prompt(manifest))
    return bundle
