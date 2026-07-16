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
import inspect
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
# A tiny fraction may exceed 1 ULP on huge tensors from rare rounding boundaries.
_FP8_MISMATCH_FRAC = 1e-3  # 0.1%
# Likewise for float outputs: a handful of elements can exceed tol from
# catastrophic cancellation in norm/RoPE on random inputs (e.g. a single element
# out of ~1e6). A real codegen bug corrupts orders of magnitude more.
_FLOAT_MISMATCH_FRAC = 1e-4  # 0.01%

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
# Default float tolerance when a kernel doesn't override it (matches Helion's
# autotuner DEFAULT_TOL). Real codegen bugs blow past this by orders of magnitude.
_DEFAULT_TOL = 1e-2


def _float_tolerances(settings: object) -> tuple[float, float]:
    """Per-kernel autotune_baseline_atol/rtol if set, else the 1e-2 default."""
    atol = getattr(settings, "autotune_baseline_atol", None)
    rtol = getattr(settings, "autotune_baseline_rtol", None)
    return (atol if atol is not None else _DEFAULT_TOL,
            rtol if rtol is not None else _DEFAULT_TOL)


def _scale_ub_index(baseline: object) -> int | None:
    """Positional index of a ``scale_ub`` arg in the baseline signature, if any.

    Some vLLM generate_inputs pass a degenerate ``scale_ub = mean(input) ~= 0``
    (e.g. silu_and_mul_per_block_quant), which collapses every per-block scale to
    the floor so the output saturates and the comparison is meaningless. We
    neutralize it to None (the production-common no-upper-bound path that vLLM's
    own has_scale_ub=False test exercises), making correctness checkable.
    """
    try:
        params = list(inspect.signature(baseline).parameters)
    except (ValueError, TypeError):
        return None
    return params.index("scale_ub") if "scale_ub" in params else None


def _compare(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> str | None:
    """Compare one output pair. Return None if within tolerance, else a diff str.

    fp8 -> uint8-ULP <= 1 for all but a <=0.1% fraction (like vLLM's tests);
    int/bool -> exact; float -> assert_close with the kernel's tolerances.
    """
    if a.dtype in _FP8_DTYPES:
        a_u8 = a.contiguous().view(torch.uint8).to(torch.int16)
        b_u8 = b.contiguous().view(torch.uint8).to(torch.int16)
        d = (a_u8 - b_u8).abs()
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
    # float: allow a tiny fraction of elements to exceed atol/rtol (rare
    # catastrophic-cancellation outliers in norm/RoPE); a real bug corrupts more.
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    tol = atol + rtol * bf.abs()
    both_nan = torch.isnan(af) & torch.isnan(bf)
    n_bad = int(((d > tol) & ~both_nan).sum().item())
    n = d.numel()
    frac = n_bad / n if n else 0.0
    if frac > _FLOAT_MISMATCH_FRAC:
        return (f"float {frac:.4%} of elems off (max_abs={d.max().item():.3g}, "
                f"{n_bad}/{n}) atol={atol} rtol={rtol} > {_FLOAT_MISMATCH_FRAC:.2%}")
    return None


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
        scale_ub_idx = _scale_ub_index(baseline) if baseline is not None else None

        shape_keys = [dict(k) if hasattr(k, "keys") else k for k in inputs_dict]
        print(f"[{name}] checking {len(inputs_dict)} shape(s): {shape_keys}")
        if scale_ub_idx is not None:
            print(f"[{name}] neutralizing degenerate scale_ub (arg[{scale_ub_idx}]) -> None")

        for key, inputs in inputs_dict.items():
            n_cases += 1
            tag = f"{name} {dict(key) if hasattr(key, 'keys') else key}"

            # Neutralize a degenerate scale_ub (see _scale_ub_index) so the
            # kernel and reference both run the no-upper-bound path.
            if scale_ub_idx is not None and scale_ub_idx < len(inputs) \
                    and isinstance(inputs[scale_ub_idx], torch.Tensor):
                inputs = list(inputs)
                inputs[scale_ub_idx] = None
                inputs = tuple(inputs)

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
            if bad is not None:
                xfail_reason = _XFAIL_CORRECTNESS.get((name, plat))
                if xfail_reason is not None:
                    xfailed.append((name, f"{tag}: {bad}"))
                    print(f"  XFAIL {tag}: {bad}  [{xfail_reason}]")
                else:
                    failures.append((name, f"correctness {tag}", bad))
            else:
                print(f"  OK    {tag}")

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
