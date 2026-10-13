"""Bootstrap ordinary partners, then run a cold full epoch for every SSD role."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute.mamba2_ssd_inputs import RAW_NAMES
from benchmarks.cute.mamba2_ssd_inputs import configure
from benchmarks.cute.mamba2_ssd_inputs import stages

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from benchmarks.cute.mamba2_ssd_objective import StageResult

    from helion.autotuner.benchmark_objective import BenchmarkRequest
    from helion.autotuner.objective_timing import ObjectiveTiming


def _raw_guard(bundle: dict[str, Any]) -> Callable[[], None]:
    from benchmarks.cute.mamba2_ssd_bootstrap import ReadonlyArguments

    guard = ReadonlyArguments(tuple(bundle["inputs"][n] for n in RAW_NAMES))

    def validate() -> None:
        guard.validate(tuple(bundle["inputs"][n] for n in RAW_NAMES))

    return validate


def compile_defaults(
    case: dict[str, Any],
    bundle: dict[str, Any],
    output: Path,
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    from benchmarks.cute.mamba2_ssd_bootstrap import bootstrap_partner
    from benchmarks.cute.mamba2_ssd_objective import check_result
    from benchmarks.cute.mamba2_ssd_objective import compare
    from benchmarks.cute.mamba2_ssd_objective import stage_arguments
    import torch
    from torch.utils._pytree import tree_flatten

    unchanged = _raw_guard(bundle)
    unchanged()
    raw = bundle["inputs"]
    values = []
    members = []
    for i, name in enumerate(stages(case)):
        unchanged()
        args = stage_arguments(raw, case, i, values)
        kernel = bundle["objective_binding"]["kernels"][name]
        configure(kernel)
        bound = kernel.bind(args)
        reference = bundle["objective_binding"]["references"][i]

        def validate(private: tuple[Any, ...], result: StageResult) -> None:
            used = {
                x.untyped_storage()._cdata
                for x in tree_flatten(private)[0]
                if isinstance(x, torch.Tensor)
            }
            check_result(result, case, i, used, raw["x"].device)  # noqa: B023 - synchronous callback, never retained
            compare(result, reference(*private))  # noqa: B023 - synchronous callback, never retained

        member = bootstrap_partner(
            bound,
            args,
            validate_result=validate,
            synchronize=synchronize,
            output=output / f"bootstrap-{name}",
        )
        unchanged()
        members.append(member)
        values.append(member.result)
    # Every original role has actually run; this assembled tuple must satisfy
    # the original full oracle before it can provide any search partners.
    compare(values[-1], bundle["reference"]())
    unchanged()
    return {
        "case": case,
        "bundle": bundle,
        "bounds": [m.bound for m in members],
        "args": [m.arguments for m in members],
        "requests": [m.request for m in members],
        "representatives": values,
        "bootstrap_members": members,
        "raw_guard": unchanged,
    }


def run_sweep(
    binding: dict[str, Any],
    *,
    provenance: dict[str, Any],
    timing: ObjectiveTiming,
    output: Path,
    synchronize: Callable[[], None],
    run_epoch: Callable[..., BenchmarkRequest] | None = None,
) -> dict[str, Any]:
    from benchmarks.cute.mamba2_ssd_bootstrap import ReadonlyArguments
    from benchmarks.cute.mamba2_ssd_objective import MambaObjective
    from benchmarks.cute.mamba2_ssd_objective import check_result
    from benchmarks.cute.mamba2_ssd_objective import compare
    from benchmarks.cute.mamba2_ssd_objective import stage_arguments

    from helion.autotuner.kernel_args import _clone_args

    if run_epoch is None:
        from benchmarks.cute.mamba2_ssd_epoch import run_epoch
    case = binding["case"]
    bundle = binding["bundle"]
    unchanged = binding.get("raw_guard") or _raw_guard(bundle)
    unchanged()
    order = stages(case)
    current = list(binding["requests"])
    bounds = list(binding["bounds"])
    owned: list[Any] = [binding]
    records = []
    for selected, name in enumerate(order):
        unchanged()
        # Refresh actual representative data from selected earlier partners.
        # Every timed member still executes all one/four stages independently.
        if selected:
            raw = dict(bundle["inputs"])
            raw.update(
                zip(
                    RAW_NAMES,
                    _clone_args(tuple(raw[n] for n in RAW_NAMES), None),
                    strict=True,
                )
            )
            values = []
            used = {raw[n].untyped_storage()._cdata for n in RAW_NAMES}
            arguments = []
            for i, request in enumerate(current):
                args = stage_arguments(raw, case, i, values)
                readonly = ReadonlyArguments(args)
                result = request.compiled(*args)
                synchronize()
                readonly.validate(args)
                unchanged()
                check_result(result, case, i, used, raw["x"].device)
                compare(result, bundle["objective_binding"]["references"][i](*args))
                synchronize()
                readonly.validate(args)
                unchanged()
                values.append(result)
                arguments.append(args)
            compare(values[-1], bundle["reference"]())
            unchanged()
            call_args = arguments[selected]
            bounds[selected] = bundle["objective_binding"]["kernels"][name].bind(
                call_args
            )
            owned.append((values, arguments))
        else:
            call_args = binding["args"][selected]
        objective = MambaObjective(
            bundle,
            case=case,
            selected=selected,
            partners={i: r for i, r in enumerate(current) if i != selected},
            provenance=provenance,
            timing_policy=timing.cache_record(),
        )
        # Admission is repeated on the actual current complete tuple before
        # every full epoch; bootstrap receipts cannot replace this oracle.
        check = objective.bind(current[selected], _clone_args(objective.raw_args, None))
        with ExitStack() as resources:
            resources.callback(check.close)
            check.reset()
            checked_output = check.run()
            synchronize()
            check.validate(checked_output)
        unchanged()
        records.append(objective.cache_record())
        current[selected] = run_epoch(
            bounds[selected],
            call_args,
            objective=objective,
            timing=timing,
            output=output / f"epoch{selected + 1}-{name}",
        )
        unchanged()
    final_objective = MambaObjective(
        bundle,
        case=case,
        selected=len(order) - 1,
        partners={i: r for i, r in enumerate(current) if i != len(order) - 1},
        provenance=provenance,
        timing_policy=timing.cache_record(),
    )
    work = final_objective.bind(
        current[-1], _clone_args(final_objective.raw_args, None)
    )
    with ExitStack() as resources:
        resources.callback(work.close)
        work.reset()
        result = work.run()
        synchronize()
        work.validate(result)
    unchanged()
    return {
        "requests": tuple(current),
        "bounds": tuple(bounds),
        "objective": final_objective,
        "representatives": owned,
        "epochs": records,
    }
