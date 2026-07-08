"""Shared test / benchmark / accuracy for the linear-attention example variants.

Each example builds a `LinearAttentionExampleHarness` naming its kernel variant,
how to make inputs, and how to call FLA, then calls run_test / run_benchmark /
run_accuracy here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable
from typing import cast
import warnings

import torch
from triton.testing import do_bench

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_engine import recurrent_step
from .linear_attention_utils import ACC_BWD_TOL
from .linear_attention_utils import ACC_FWD_TOL
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import head_to_time_first as _htf
from .linear_attention_utils import naive_recurrent_reference
from .linear_attention_utils import rel_error as _rel_error
from helion._testing import DEVICE

# Test/benchmark config
DTYPE = torch.bfloat16
TEST_SHAPE = (2, 4, 128, 32, 32)
TEST_C = 32
BENCH_CONFIGS = [(1, 32, 2048, 128, 128), (1, 32, 4096, 128, 128)]
BENCH_C = 64


@dataclass
class Inputs:
    """Variant inputs: q/k/v/scale always, plus the decay/correction extras."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scale: float
    g: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    gate: torch.Tensor | None = None


@dataclass
class LinearAttentionExampleHarness:
    """Test, benchmark, and accuracy harness for one linear-attention variant."""

    name: str
    title: str
    variant: LinearAttentionVariant
    make_inputs: Callable[..., Inputs]
    fla_fwd: Callable[[Inputs, float], torch.Tensor] | None = None
    check_recurrent: bool = True
    grad_tensors: tuple[str, ...] = ("q", "k", "v")
    dtype: torch.dtype = DTYPE

    def helion_fwd(self, i: Inputs, C: int) -> torch.Tensor:
        fwd = self.variant.get_fwd_kernel()
        return fwd(i.q, i.k, i.v, i.g, i.beta, C=C, scale=i.scale)

    def helion_fb(self, i: Inputs, grad_out: torch.Tensor, C: int) -> None:
        self.helion_fwd(i, C).backward(grad_out)

    def fla_fb(self, i: Inputs, go_t: torch.Tensor, scale: float) -> None:
        assert self.fla_fwd is not None
        self.fla_fwd(i, scale).backward(go_t)

    def reference(self, i: Inputs) -> torch.Tensor:
        assert i.g is not None
        return naive_recurrent_reference(
            i.q, i.k, i.v, i.g.float(), beta=i.beta, q_scale=i.scale
        )

    def chunked_reference(self, i: Inputs, C: int) -> torch.Tensor:
        assert i.g is not None
        return chunked_linear_attn_reference(
            i.q * i.scale, i.k, i.v, i.g, beta=i.beta, C=C
        )

    # test / benchmark / accuracy: the module-level API run_linattn.py imports.
    def test(self) -> None:
        run_test(self, TEST_SHAPE, TEST_C)

    def benchmark(
        self, configs: list | None = None
    ) -> list[tuple[str, float, float, float, float]]:
        return run_benchmark(
            self, configs if configs is not None else BENCH_CONFIGS, BENCH_C
        )

    def accuracy(self, configs: list | None = None) -> list[tuple[str, str]]:
        return run_accuracy(
            self, configs if configs is not None else BENCH_CONFIGS, BENCH_C
        )


def _grad_leaves(
    harness: LinearAttentionExampleHarness, inputs: Inputs
) -> tuple[Inputs, list]:
    """Copy of inputs with grad_tensors swapped for fresh requires_grad copies."""
    out = dataclasses.replace(inputs)
    leaves = []
    for name in harness.grad_tensors:
        leaf = getattr(out, name).detach().clone().requires_grad_(True)
        setattr(out, name, leaf)
        leaves.append(leaf)
    return out, leaves


def _fla_inputs(inputs: Inputs) -> Inputs:
    """Inputs in FLA's time-first layout: transpose every tensor, keep scalars."""
    out = dataclasses.replace(inputs)
    for f in dataclasses.fields(out):
        val = getattr(out, f.name)
        if isinstance(val, torch.Tensor):
            setattr(out, f.name, _htf(val))
    return out


def _recurrent_error(
    harness: LinearAttentionExampleHarness, inputs: Inputs, C: int
) -> float:
    """Rel error of the chunked output vs the step-by-step recurrent_step loop."""
    q, k, v, g, scale = inputs.q, inputs.k, inputs.v, inputs.g, inputs.scale
    B, H, T, D = q.shape
    DV = v.shape[-1]

    o_chunked = harness.helion_fwd(inputs, C)

    state = q.new_zeros(B, H, D, DV, dtype=torch.float32)
    o_steps = []
    for t in range(T):
        gt = g[:, :, t : t + 1] if g is not None else q.new_zeros(B, H, 1)
        beta_t = inputs.beta[:, :, t : t + 1] if inputs.beta is not None else None
        o_t, state = recurrent_step(
            q[:, :, t : t + 1] * scale,
            k[:, :, t : t + 1],
            v[:, :, t : t + 1],
            state,
            alpha=torch.exp(gt),
            beta_val=beta_t,
        )
        o_steps.append(o_t)
    o_recurrent = torch.cat(o_steps, dim=2)
    return _rel_error(o_chunked, o_recurrent)


def run_test(
    harness: LinearAttentionExampleHarness,
    test_shape: tuple[int, int, int, int, int],
    C: int,
) -> None:
    """Forward + backward correctness vs reference and FLA."""
    torch.manual_seed(42)
    B, H, T, D, DV = test_shape
    inputs = harness.make_inputs(B, H, T, D, DV, dtype=harness.dtype, device=DEVICE)
    scale = inputs.scale

    # === Forward: vs naive recurrent reference ===
    out = harness.helion_fwd(inputs, C)
    ref = harness.reference(inputs)
    fwd_err = _rel_error(out, ref)
    assert fwd_err < ACC_FWD_TOL, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # === Forward: vs FLA (fla_fwd is None when fla is not installed) ===
    if harness.fla_fwd is None:
        warnings.warn("fla not installed, skipping FLA comparisons", stacklevel=1)
        has_fla = False
    else:
        # fla_fwd returns time-first; transpose back to compare (untimed).
        o_fla = harness.fla_fwd(_fla_inputs(inputs), scale).transpose(1, 2).contiguous()
        fla_err = _rel_error(out, o_fla)
        print(
            f"  fwd vs FLA:       {fla_err:.4e}"
            f" {'PASS' if fla_err < ACC_FWD_TOL else 'FAIL'}"
        )
        has_fla = True

    # === Backward: Helion grads vs chunked reference ===
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=harness.dtype)
    h_inputs, h_leaves = _grad_leaves(harness, inputs)
    harness.helion_fb(h_inputs, grad_out, C)
    r_inputs, r_leaves = _grad_leaves(harness, inputs)
    harness.chunked_reference(r_inputs, C).backward(grad_out)
    for name, hl, rl in zip(harness.grad_tensors, h_leaves, r_leaves, strict=True):
        err = _rel_error(hl.grad, rl.grad)
        assert err < ACC_BWD_TOL, f"Backward d{name} error: {err}"
        print(f"  bwd d{name} vs ref: {err:.4e} PASS")

    # === Backward: Helion grads vs FLA (dq asserted, dk/dv info) ===
    if has_fla:
        f_inputs, f_leaves = _grad_leaves(harness, _fla_inputs(inputs))
        harness.fla_fb(f_inputs, _htf(grad_out), scale)
        for name, hl, fl in zip(harness.grad_tensors, h_leaves, f_leaves, strict=True):
            err = _rel_error(hl.grad, fl.grad.transpose(1, 2).contiguous())
            gate = (
                f" {'PASS' if err < ACC_BWD_TOL else 'FAIL'}"
                if name == "q"
                else " (info)"
            )
            print(f"  bwd d{name} vs FLA:  {err:.4e}{gate}")

    # === Recurrent step: chunked vs step-by-step recurrent_step ===
    if harness.check_recurrent:
        rec_err = _recurrent_error(harness, inputs, C)
        assert rec_err < ACC_BWD_TOL, f"Recurrent vs chunked error: {rec_err}"
        print(f"  recurrent step:   {rec_err:.4e} PASS")

    print("All tests passed.")


def _time_config(
    harness: LinearAttentionExampleHarness,
    shape: tuple[int, int, int, int, int],
    C: int,
) -> tuple[float, float, float, float]:
    """Time helion/FLA forward and fwd+bwd for one shape."""
    bi, hi, ti, di, dvi = shape
    inputs = harness.make_inputs(
        bi, hi, ti, di, dvi, dtype=harness.dtype, device=DEVICE, requires_grad=True
    )
    scale = inputs.scale
    grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=harness.dtype)
    fla_inputs = _fla_inputs(inputs)
    go_t = _htf(grad_out)
    h_grads = [getattr(inputs, n) for n in harness.grad_tensors]
    fla_grads = [getattr(fla_inputs, n) for n in harness.grad_tensors]

    fwd_ms = do_bench(lambda: harness.helion_fwd(inputs, C))
    fla_fwd_ms = do_bench(
        lambda: harness.fla_fwd(fla_inputs, scale)  # type: ignore[misc]
    )
    fb_ms = do_bench(
        lambda: harness.helion_fb(inputs, grad_out, C), grad_to_none=h_grads
    )
    fla_fb_ms = do_bench(
        lambda: harness.fla_fb(fla_inputs, go_t, scale),
        grad_to_none=fla_grads,
    )
    return (
        cast("float", fwd_ms),
        cast("float", fla_fwd_ms),
        cast("float", fb_ms),
        cast("float", fla_fb_ms),
    )


def run_benchmark(
    harness: LinearAttentionExampleHarness,
    configs: list,
    C: int,
) -> list[tuple[str, float, float, float, float]]:
    """Benchmark forward and fwd+bwd, comparing against FLA.

    Returns one (config, helion_fwd_ms, fla_fwd_ms, helion_fb_ms, fla_fb_ms) row
    per config; empty when fla is unavailable.
    """
    rows: list[tuple[str, float, float, float, float]] = []
    if harness.fla_fwd is None:
        # also None when the variant has no comparable FLA op
        warnings.warn("fla not installed, skipping benchmark", stacklevel=1)
        return rows

    print(
        f"{'Config':<24} {'Helion fwd':>10} {'FLA fwd':>10}"
        f" {'Helion f+b':>12} {'FLA f+b':>12}"
    )
    print("-" * 72)

    for shape in configs:
        fwd_ms, fla_fwd_ms, fb_ms, fla_fb_ms = _time_config(harness, shape, C)
        cfg = f"({','.join(str(x) for x in shape)})"
        print(
            f"{cfg:<24} {fwd_ms:>10.3f} {fla_fwd_ms:>10.3f}"
            f" {fb_ms:>12.3f} {fla_fb_ms:>12.3f}"
        )
        rows.append((cfg, fwd_ms, fla_fwd_ms, fb_ms, fla_fb_ms))

    return rows


def run_accuracy(
    harness: LinearAttentionExampleHarness,
    configs: list,
    C: int,
) -> list[tuple[str, str]]:
    """Per-config (fwd, bwd) verdicts vs the fp32 PyTorch reference.

    Each of fwd and bwd is one of: ``ok`` (matches within tolerance), ``FAIL``
    (ran but over tolerance), ``HEL-ERR`` (the Helion kernel errored), ``REF-ERR``
    (the reference errored, e.g. its autograd graph OOMs). Forward compares
    against the naive recurrent reference; backward compares autograd gradients
    against the chunked reference.
    """
    verdicts: list[tuple[str, str]] = []
    for bi, hi, ti, di, dvi in configs:
        inputs = harness.make_inputs(
            bi, hi, ti, di, dvi, dtype=harness.dtype, device=DEVICE
        )

        try:
            out = harness.helion_fwd(inputs, C)
        except Exception:
            torch.cuda.empty_cache()
            fwd = "HEL-ERR"
        else:
            try:
                ref = harness.reference(inputs)
            except Exception:
                torch.cuda.empty_cache()
                fwd = "REF-ERR"
            else:
                fwd = "ok" if _rel_error(out, ref) < ACC_FWD_TOL else "FAIL"

        grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=harness.dtype)
        try:
            h_inputs, h_leaves = _grad_leaves(harness, inputs)
            harness.helion_fwd(h_inputs, C).backward(grad_out)
        except Exception:
            torch.cuda.empty_cache()
            bwd = "HEL-ERR"
        else:
            try:
                r_inputs, r_leaves = _grad_leaves(harness, inputs)
                harness.chunked_reference(r_inputs, C).backward(grad_out)
            except Exception:
                torch.cuda.empty_cache()
                bwd = "REF-ERR"
            else:
                bwd = (
                    "ok"
                    if all(
                        _rel_error(h.grad, r.grad) < ACC_BWD_TOL
                        for h, r in zip(h_leaves, r_leaves, strict=True)
                    )
                    else "FAIL"
                )
        verdicts.append((fwd, bwd))
    return verdicts
