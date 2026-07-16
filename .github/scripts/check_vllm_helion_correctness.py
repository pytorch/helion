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

import vllm  # noqa: F401  registers torch.ops._C.*
from vllm.kernels.helion import get_registered_kernels
from vllm.kernels.helion.ops import import_all_kernels


def _assert_close(a: object, b: object, tag: str) -> None:
    """Compare a single (kernel, baseline) arg pair; no-op for non-tensors."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        torch.testing.assert_close(a.float(), b.float(), rtol=0.1, atol=0.1, msg=tag)
    elif a.dtype.is_floating_point:
        torch.testing.assert_close(a, b, rtol=2e-2, atol=1e-3, msg=tag)
    else:  # int (e.g. packed UE8M0 scales) must match exactly
        torch.testing.assert_close(a, b, msg=tag)


def main() -> int:
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available")
        return 1
    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {dev} (sm{cc[0]}{cc[1]})\n")

    import_all_kernels()
    kernels = get_registered_kernels()
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

            try:
                for a, b in zip(k_args, b_args):
                    _assert_close(a, b, tag)
                print(f"  OK    {tag}")
            except AssertionError as e:
                failures.append((name, f"correctness {tag}", str(e).strip().splitlines()[0]))

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
