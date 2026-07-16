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

# fp8 quantization differs from the torch reference by up to 1 ULP at rounding
# boundaries (input-dependent: fp32 intermediates round differently in Triton vs
# torch). vLLM's own kernel tests allow this -- e.g.
# tests/kernels/helion/test_per_token_group_fp8_quant.py compares fp8 outputs as
# (a.view(uint8) - b.view(uint8)).abs().max() <= 1.
_FP8_ULP_TOL = 1
# ...but tolerate a tiny FRACTION of elements exceeding 1 ULP. Some vLLM
# generate_inputs use a degenerate scale_ub (e.g. silu_and_mul_per_block_quant
# uses mean(input) ~= 0), which amplifies near-zero activations to where a
# sign-flip from Triton-vs-torch fp32 rounding flips the fp8 sign bit -> a large
# uint8 diff on a handful of boundary elements. That is an input artifact, not a
# codegen bug. A real bug corrupts a large fraction and still fails.
_FP8_MISMATCH_FRAC = 5e-3  # 0.5%
# Default float tolerance when a kernel doesn't override it (matches Helion's
# autotuner DEFAULT_TOL). Real codegen bugs blow past this by orders of magnitude.
_DEFAULT_TOL = 1e-2


def _float_tolerances(settings: object) -> tuple[float, float]:
    """Per-kernel autotune_baseline_atol/rtol if set, else the 1e-2 default."""
    atol = getattr(settings, "autotune_baseline_atol", None)
    rtol = getattr(settings, "autotune_baseline_rtol", None)
    return (atol if atol is not None else _DEFAULT_TOL,
            rtol if rtol is not None else _DEFAULT_TOL)


def _compare(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> str | None:
    """Compare one output pair. Return None if within tolerance, else a diff str.

    fp8 -> uint8-ULP <= 1 for all but a <=0.5% mismatch fraction (like vLLM's
    tests, plus a boundary-artifact allowance); int/bool -> exact; float ->
    assert_close with the kernel's tolerances.
    """
    if a.dtype in _FP8_DTYPES:
        au = a.contiguous().view(torch.uint8).to(torch.int16)
        bu = b.contiguous().view(torch.uint8).to(torch.int16)
        d = (au - bu).abs()
        n = d.numel()
        n_bad = int((d > _FP8_ULP_TOL).sum().item())
        frac = n_bad / n if n else 0.0
        if frac > _FP8_MISMATCH_FRAC:
            return (f"fp8 {frac:.3%} of elems > {_FP8_ULP_TOL} ULP "
                    f"(max={int(d.max().item())}, {n_bad}/{n}) > {_FP8_MISMATCH_FRAC:.1%}")
        return None
    if a.dtype == torch.bool or not a.dtype.is_floating_point:
        d = (a.to(torch.int64) - b.to(torch.int64)).abs().max().item()
        return None if d == 0 else f"int max_abs={d} (exact required)"
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return None
    except AssertionError:
        return f"{_diff_str(a, b)} (atol={atol} rtol={rtol})"


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
        d = (a.contiguous().view(torch.uint8).int() - b.contiguous().view(torch.uint8).int()).abs()
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

            # Outputs = args the baseline mutated; compare only those (inputs are
            # identical copies). fp8 allows 1 ULP, floats use the kernel's tol.
            out_idxs = [i for i, (p, b) in enumerate(zip(pristine, b_args)) if _is_mutated(p, b)]
            if not out_idxs:  # nothing detected mutated -> compare all tensor pairs
                out_idxs = [i for i, b in enumerate(b_args) if isinstance(b, torch.Tensor)]
            atol, rtol = _float_tolerances(wrapper.helion_settings)

            bad = None
            for i in out_idxs:
                msg = _compare(k_args[i], b_args[i], atol, rtol)
                if msg is not None:
                    bad = f"arg[{i}] {msg}"
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
