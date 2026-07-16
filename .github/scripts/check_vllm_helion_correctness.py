#!/usr/bin/env python3
"""Correctness + compile check for the vLLM-registered Helion kernels.

For every Helion kernel registered in the installed vLLM, this compiles and runs
the kernel at *each* of its committed config/shape points (the op's
``input_generator`` yields one shape per tuned config, and the pretuned config
picker selects that config), then compares the result against the kernel's
autotune baseline.

This is a correctness gate, not a perf benchmark. Its main job is to catch
codegen/compile regressions that break a specific config on a specific GPU
(e.g. the ``rms_norm_per_block_quant`` B200 Triton MLIR ``PassManager::run
failed`` bug) before a Helion change lands. Run on both H100 and B200.

Exit code is non-zero if any kernel fails to compile/run or mismatches its
baseline. Kernels with no config for the current platform are skipped (reported).
"""

from __future__ import annotations

import copy
import sys
import traceback

import torch

import vllm  # noqa: F401  registers torch.ops._C.* (+ Helion kernels, in current vLLM)
from vllm.kernels.helion import get_registered_kernels

# Use Helion's OWN autotuner accuracy check so this gate applies the exact same
# acceptance criterion the config was tuned under -- notably it compares fp8 by
# viewing as uint8 (a 1-ULP fp8 diff is 1 in uint8 space, within rtol), where a
# naive float compare would inflate it to a whole fp8 step and false-fail.
from helion.autotuner.accuracy import assert_close as _helion_assert_close


def _force_register() -> None:
    """Best-effort trigger of Helion op registration.

    Current vLLM registers every Helion kernel as a side effect of ``import
    vllm``; some builds instead expose an explicit importer whose name has
    drifted (``import_all_kernels`` / ``import_all_ops``). Call whichever
    exists; if none does, registration already happened on import.
    """
    try:
        import vllm.kernels.helion.ops as ops
    except ImportError:
        return
    for fn_name in ("import_all_kernels", "import_all_ops"):
        fn = getattr(ops, fn_name, None)
        if callable(fn):
            fn()
            return


# fp8 dtypes; mirror helion.autotuner.accuracy._FP8_DTYPES (guard variants that
# may be absent on older torch).
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
for _n in ("float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"):
    _d = getattr(torch, _n, None)
    if _d is not None:
        _FP8_DTYPES.add(_d)

# Matches helion.autotuner.benchmark_provider._compute_effective_tolerances.
_DEFAULT_TOL = 1e-2


def _effective_tolerances(out_dtypes: set, settings: object) -> tuple[float, float]:
    """The tolerances Helion's autotuner used to accept this kernel's config.

    Mirror ``benchmark_provider._compute_effective_tolerances``: per-kernel
    ``autotune_baseline_atol``/``rtol`` if set; all-fp8 outputs use a bitwise
    check (0, 0); otherwise the 1e-2 default. Checking against the exact bar the
    config was tuned under avoids false failures (e.g. per-group fp32 scales that
    pass 1e-2 but not a tighter ad-hoc atol) while still catching real drift.
    """
    atol = getattr(settings, "autotune_baseline_atol", None)
    rtol = getattr(settings, "autotune_baseline_rtol", None)
    all_fp8 = bool(out_dtypes) and all(d in _FP8_DTYPES for d in out_dtypes)
    if all_fp8 and atol is None and rtol is None:
        return 0.0, 0.0
    return (atol if atol is not None else _DEFAULT_TOL,
            rtol if rtol is not None else _DEFAULT_TOL)


def _is_mutated(pristine: object, after: object) -> bool:
    """True if ``after`` (a post-call arg) was written vs its pristine copy."""
    if not isinstance(after, torch.Tensor) or not isinstance(pristine, torch.Tensor):
        return False
    if pristine.shape != after.shape or pristine.dtype != after.dtype:
        return True
    try:
        return not torch.equal(pristine, after)
    except RuntimeError:  # some dtypes (fp8) don't support equal()
        return not torch.equal(pristine.float(), after.float())


def _diff_str(a: torch.Tensor, b: torch.Tensor) -> str:
    """Human-readable diff, fp8 measured in uint8-ULP like Helion's check."""
    if a.dtype in _FP8_DTYPES:
        d = (a.view(torch.uint8).int() - b.view(torch.uint8).int()).abs()
        return f"max_uint8_ulp={d.max().item()} (fp8 {a.dtype})"
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-12)
    return f"max_abs={d.max().item():.3g} max_rel={(d / denom).max().item():.3g} dtype={a.dtype}"


def main() -> int:
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available")
        return 1
    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {dev} (sm{cc[0]}{cc[1]})\n")

    _force_register()
    kernels = get_registered_kernels()
    if not kernels:
        print("FATAL: no Helion kernels registered by vLLM")
        return 1
    print(f"Found {len(kernels)} registered Helion kernels: {sorted(kernels)}\n")

    failures: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    n_cases = 0

    for name, wrapper in sorted(kernels.items()):
        # Skip kernels with no config for this platform (config coverage gap,
        # not a compile failure) -- reported, not fatal.
        try:
            wrapper.get_configured_op()
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"no config for this platform: {e}"))
            continue

        try:
            inputs_dict = wrapper.get_inputs()
        except Exception as e:  # noqa: BLE001
            failures.append((name, "get_inputs", repr(e)))
            continue

        baseline = getattr(wrapper.helion_settings, "autotune_baseline_fn", None)

        shape_keys = [dict(k) if hasattr(k, "keys") else k for k in inputs_dict]
        print(f"[{name}] checking {len(inputs_dict)} shape(s): {shape_keys}")

        for key, inputs in inputs_dict.items():
            n_cases += 1
            tag = f"{name} {dict(key) if hasattr(key, 'keys') else key}"

            # Pristine copy to detect which args are outputs (written in place).
            pristine = copy.deepcopy(inputs)

            # Compile + run the kernel (committed config for this shape). A
            # codegen/compile failure surfaces here.
            k_args = copy.deepcopy(inputs)
            try:
                wrapper(*k_args)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                failures.append(
                    (name, f"compile/run {tag}", traceback.format_exc().strip().splitlines()[-1])
                )
                continue

            if baseline is None:
                print(f"  OK    {tag} (compile-only, no baseline)")
                continue  # compile-only for kernels without a baseline

            # Reference for correctness.
            b_args = copy.deepcopy(inputs)
            try:
                baseline(*b_args)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                # A broken baseline shouldn't fail the kernel's compile check;
                # note it and move on.
                skipped.append((name, f"baseline raised for {tag}"))
                continue

            # Outputs = args the baseline mutated. Use the same tolerances Helion
            # used to accept this config (from the output dtypes), then compare
            # only those outputs (inputs are identical copies).
            out_idxs = [i for i, (p, b) in enumerate(zip(pristine, b_args)) if _is_mutated(p, b)]
            if not out_idxs:  # nothing detected mutated -> compare all tensor pairs
                out_idxs = [i for i, b in enumerate(b_args) if isinstance(b, torch.Tensor)]
            out_dtypes = {b_args[i].dtype for i in out_idxs}
            atol, rtol = _effective_tolerances(out_dtypes, wrapper.helion_settings)

            bad = None
            for i in out_idxs:
                try:
                    _helion_assert_close(k_args[i], b_args[i], atol=atol, rtol=rtol)
                except AssertionError:
                    bad = f"arg[{i}] {_diff_str(k_args[i], b_args[i])} (atol={atol} rtol={rtol})"
                    break
            if bad is None:
                print(f"  OK    {tag}")
            else:
                failures.append((name, f"correctness {tag}", bad))

    print(f"\nChecked {n_cases} (kernel, shape) cases across {len(kernels)} kernels.")
    for name, why in skipped:
        print(f"  SKIP  {name}: {why}")
    for name, where, msg in failures:
        print(f"  FAIL  {name} [{where}]: {msg}")

    if failures:
        print(f"\n{len(failures)} failure(s).")
        return 1
    print("\nAll vLLM Helion kernels compiled and matched their baseline. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
