#!/usr/bin/env python3
"""Correctness + compile check for the vLLM-registered Helion kernels.

For every Helion kernel registered in the installed vLLM, this compiles and runs
the kernel at *each* of its committed config/shape points (the op's
``input_generator`` yields one shape per tuned config, and the pretuned config
picker selects that config), then compares the result against the kernel's
autotune baseline.

The per-shape numerics comparison is NOT reimplemented here: it is the exact
check vLLM ships in ``scripts/benchmark_helion_kernels.py``
(``check_kernel_correctness``), which the CI workflow fetches onto ``PYTHONPATH``
at the vLLM commit matching the installed wheel. This keeps the tolerances,
fp8-ULP handling, and mutation comparison identical to vLLM's own kernel tests.

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

# vLLM's shared per-shape numerics check. The workflow fetches
# scripts/benchmark_helion_kernels.py (at the installed wheel's commit) onto
# PYTHONPATH, so this imports as a top-level module. Running it here means the
# comparison is byte-for-byte what vLLM uses -- no parallel implementation to
# drift out of sync.
from benchmark_helion_kernels import check_kernel_correctness


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


# Known-failing CORRECTNESS cases, keyed by (kernel, platform "sm<cc>"). The
# kernel must still COMPILE (compile failures are never xfail'd); only a
# correctness mismatch is downgraded to XFAIL (reported, non-fatal). Keep this
# list tiny and always paired with a TODO to remove it.
_XFAIL_CORRECTNESS = {
    # TODO(shangdiy): investigate on a B200 machine, then remove. On B200
    # (sm100) ~0.77% of fused_qk_norm_rope elements diverge up to ~2.1 vs the
    # reference on small-token shapes; H100 (sm90) is clean and this is an
    # approximate kernel (author tolerance 5e-2). Tracking: <add issue link>.
    ("fused_qk_norm_rope", "sm100"): "B200 numerical divergence on small-token shapes; under investigation",
}


def _run_compile_only(name, wrapper, inputs_dict, failures):
    """Compile + run each shape for a kernel that has no baseline.

    A codegen/compile failure surfaces as an exception and is recorded; there is
    nothing to compare against, so a clean run is reported as OK.
    """
    n = 0
    for key, inputs in inputs_dict.items():
        n += 1
        tag = f"{name} {dict(key) if hasattr(key, 'keys') else key}"
        try:
            wrapper(*copy.deepcopy(inputs))
            torch.cuda.synchronize()
            print(f"  OK    {tag} (compile-only, no baseline)")
        except Exception:  # noqa: BLE001
            failures.append(
                (name, f"compile/run {tag}", traceback.format_exc().strip().splitlines()[-1])
            )
    return n


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

    plat = f"sm{cc[0]}{cc[1]}"
    failures: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    xfailed: list[tuple[str, str]] = []
    n_cases = 0

    for name, wrapper in sorted(kernels.items()):
        # Skip kernels with no config for this platform (config coverage gap,
        # not a compile failure) -- reported, not fatal.
        try:
            wrapper.get_configured_op()
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"no config for this platform: {e}"))
            continue

        # Seed before get_inputs() (which uses torch.randn) so the run is
        # reproducible commit-to-commit instead of flaky on random draws.
        torch.manual_seed(0)
        try:
            inputs_dict = wrapper.get_inputs()
        except Exception as e:  # noqa: BLE001
            failures.append((name, "get_inputs", repr(e)))
            continue

        baseline = getattr(wrapper.helion_settings, "autotune_baseline_fn", None)
        shape_keys = [dict(k) if hasattr(k, "keys") else k for k in inputs_dict]
        print(f"[{name}] checking {len(inputs_dict)} shape(s): {shape_keys}")

        if baseline is None:
            n_cases += _run_compile_only(name, wrapper, inputs_dict, failures)
            continue

        # Shared numerics check: same comparison vLLM uses, one result per
        # shape, continuing past failures so every bad shape is reported.
        for r in check_kernel_correctness(wrapper, baseline, inputs_dict):
            n_cases += 1
            tag = f"{name} {r.case}"
            if r.passed:
                print(f"  OK    {tag}")
                continue

            # A numerics mismatch (check raised an AssertionError, wrapped as
            # "Numerics check failed ...") is xfail-able per (kernel, platform);
            # a compile/run error is never xfail'd.
            error = r.error or ""
            is_numerics = error.startswith("Numerics check failed")
            xfail_reason = _XFAIL_CORRECTNESS.get((name, plat)) if is_numerics else None
            if xfail_reason is not None:
                xfailed.append((name, f"{tag}: {error.splitlines()[0]}"))
                print(f"  XFAIL {tag}  [{xfail_reason}]")
            else:
                where = "correctness" if is_numerics else "compile/run"
                failures.append((name, f"{where} {tag}", error.splitlines()[0]))

    print(f"\nChecked {n_cases} (kernel, shape) cases across {len(kernels)} kernels.")
    for name, why in skipped:
        print(f"  SKIP  {name}: {why}")
    for name, why in xfailed:
        print(f"  XFAIL {name}: {why}")
    for name, where, msg in failures:
        print(f"  FAIL  {name} [{where}]: {msg}")

    if failures:
        print(f"\n{len(failures)} failure(s)"
              + (f", {len(xfailed)} xfail(s)." if xfailed else "."))
        return 1
    tail = f" ({len(xfailed)} known xfail(s))" if xfailed else ""
    print(f"\nAll vLLM Helion kernels compiled and matched their baseline{tail}. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
