#!/usr/bin/env python3
"""Random variants of the examples of a change, compiled with helion.inductor
and checked against eager: does the change still compile, still give the right
numbers, and fire (or decline) where it should, beyond the one shape of the
example? (The skill's Generality and Numerics rules.)

The examples file is pr_examples.py's, with PyTorch functions only: a
helion.Kernel would be its own reference. For a Helion compiler change, sweep
a PyTorch program that reaches the changed code through helion.inductor. An
example that takes part declares its variants: named dims shared between its
arguments, each drawn from a pool (POOL has no 0: list 0 in a dim's own pool
for empty tensors), optionally a predicate on the generated Helion source
saying whether the change fired, and casts_down=True if the function casts
to float32 itself (its float64 reference is then no more precise than eager):

    variants=dict(
        dims={"S": None, "H": [1, 8], "D": [64, 96, 128, 256]},  # None: POOL
        args=[("S", "H", "D"), ("D",), {"shape": ("S",), "dtype": "int", "high": "D"}],
    ),
    fired=lambda code: "128:256" in code,

A tuple is a floating-point argument of that shape: every variant draws one
dtype for them (float32, bfloat16, float16, float64) and a layout per argument
(contiguous, transposed storage, a slice with an offset, channels-last for 4-D,
an expanded size-1 dim). An int argument holds values below its `high` dim,
and a bool one random bools. The function must be deterministic and must not
write into its arguments: the sweep compares what it returns, and a sample of
random numbers differs between the implementations.

Each variant compiles with stock Inductor (cuda_backend "triton") and with
helion.inductor (pr_examples.helion_options(): the example's backend, autotune
effort none), and runs eager, stock and helion.inductor on every input
distribution below against the graph run in float64:

- ulp: the mean error in units of the output dtype's spacing at the
  reference, floored at eps times the output's median nonzero magnitude, so
  that an element near zero in an output of larger values counts its error
  on the output's scale (an output that is zero throughout in the reference
  takes the inputs' mean magnitude). A cancellation's errors run to many
  ulp for eager and helion.inductor alike, which the rule's standard errors
  absorb;
- nonfinite: the share of outputs that are inf or NaN where the reference,
  rounded to the output dtype, is finite;
- special: the share of outputs that are not that inf or NaN where the rounded
  reference is inf or NaN;
- gross: the share of float outputs off by a quarter of their value or more
  (of the output's median magnitude, for a smaller value): wrong elements,
  which a mean over many right ones hides (a [1000, 257] bf16 output wrong
  at x = 256 averages under 1 ulp);
- mismatch: for integer and bool outputs, the share that differ.

Rule per distribution: helion's ulp <= eager's + 1 (plus three standard
errors of eager's mean, which only matter for errors of many ulp), helion's
gross share <= eager's, helion's nonfinite share <= stock's (eager's where
stock failed), helion's special and mismatch shares <= eager's, each share
within three standard errors of its bound and none where the bound has none.
A reference no more precise than eager (a float64 variant, or a function
that casts to float32 itself and declares casts_down) leaves eager's float
error at 0: the ulp and gross bounds are then the larger of eager's and
stock's, and where stock failed the ulp goes unchecked and the gross share
keeps eager's bound. The reference runs on copies of the inputs. A compile
or runtime error of helion.inductor, an output of another structure, type,
dtype or shape than eager's, or a NaN or inf metric, is a failure (the same
error on the first two distributions, a compile error, is recorded for every
distribution). A comparison whose eager or stock side failed is reported as
unchecked, and a draw the function rejects in float64 (an odd dim it halves)
as invalid. A Python float output is measured as a float64. An error outside
the compiled calls (a CUDA fault a kernel left) stops the sweep with the
records so far, written after each variant. A complex output is measured as
its real and imaginary parts.

Variant i of an example draws its dims, dtype, layouts and inputs from
(--seed, the example's name, i) alone, so --names and --index rerun it on
its own.

    PYTHONPATH=<tree> python sweep_variants.py EXAMPLES.py out.json \
        [--variants 12] [--seed 0] [--names a,b] [--index i]

It exits 1 when a variant fails, or when no example is selected or one has
no variant that compared anything.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import random
import re
import sys
import traceback
from typing import Any
from typing import Callable
import zlib

from pr_examples import helion_options
from pr_examples import helion_sources
from pr_examples import load_examples
import torch
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code
from torch.utils._pytree import tree_flatten

import helion

DEV = "cuda"
POOL = [1, 2, 3, 7, 16, 17, 31, 32, 33, 64, 100, 127, 128, 255, 256, 257, 1000, 1024]
FLOATS = [torch.float32, torch.bfloat16, torch.float16, torch.float64]
MAX_NUMEL = 1 << 22


# Input distributions: name -> values of x's shape, in float64, for `dtype`,
# from x = randn values and the variant's generator.
Dist = Callable[[torch.Tensor, torch.dtype, torch.Generator], torch.Tensor]


def _rand(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype)


def _outliers(
    x: torch.Tensor, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    """Outlier channels: about 2% of the last dim's channels scaled up, as in
    a transformer's activations."""
    if not x.numel() or not x.dim():
        return x
    big = min(1e3, torch.finfo(dtype).max ** 0.25)
    channels = _rand(x[(0,) * (x.dim() - 1)], gen) < 0.02
    return torch.where(channels, x * big, x)


def _log_uniform(
    x: torch.Tensor, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    e = min(10.0, math.log10(torch.finfo(dtype).max) / 4)
    return torch.sign(x) * 10 ** ((_rand(x, gen) * 2 - 1) * e)


def _cancel(x: torch.Tensor, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    """Mixed signs that sum to about zero along the last dim: no order cancels
    them exactly."""
    return x - x.mean(-1, keepdim=True) if x.dim() else x


def _signed_zeros(
    x: torch.Tensor, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    u = _rand(x, gen)
    return torch.where(u < 0.15, 0.0, torch.where(u < 0.3, -0.0, x))


def _nonfinite(
    x: torch.Tensor, dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    u = _rand(x, gen)
    x = torch.where(u < 0.002, math.inf, x)
    x = torch.where((u >= 0.002) & (u < 0.004), -math.inf, x)
    return torch.where((u >= 0.004) & (u < 0.006), math.nan, x)


def _constant_last_dim(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Constant along the last dim: one value per index of the others."""
    return (x[..., :1] * scale).expand_as(x).clone() if x.dim() else x * scale


DISTRIBUTIONS: dict[str, Dist] = {
    "normal": lambda x, d, g: x,
    "uniform": lambda x, d, g: _rand(x, g) * 2 - 1,
    "offset": lambda x, d, g: 1000 + x * 0.01,
    "outliers": _outliers,
    "log_uniform": _log_uniform,
    "constant_last_dim": lambda x, d, g: _constant_last_dim(x, 100.0),
    "tiny": lambda x, d, g: x * torch.finfo(d).tiny * 16,
    "cancel": _cancel,
    "signed_zeros": _signed_zeros,
    "nonfinite": _nonfinite,
    "large_last_dim": lambda x, d, g: _constant_last_dim(
        x.abs() + 0.1, torch.finfo(d).max / 64
    ),
}


def draw_variant(spec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    for _ in range(100):
        dims = {k: rng.choice(pool or POOL) for k, pool in spec["dims"].items()}
        shapes = [
            [dims[d] for d in (arg["shape"] if isinstance(arg, dict) else arg)]
            for arg in spec["args"]
        ]
        if all(math.prod(s) <= MAX_NUMEL for s in shapes):
            break
    else:
        raise SystemExit(
            f"no draw of {spec['dims']} keeps every argument within {MAX_NUMEL} elements"
        )
    layouts = []
    for shape in shapes:
        # Only the layouts make_arg() can give a tensor of this rank.
        options = ["contiguous", "contiguous"]
        if shape:
            options.append("sliced")
        if len(shape) >= 2:
            options.append("transposed")
        if len(shape) == 4:
            options.append("channels_last")
        if shape and shape[0] > 1:
            options.append("expanded")
        layouts.append(rng.choice(options))
    return {
        "dims": dims,
        "dtype": str(rng.choice(FLOATS)).removeprefix("torch."),
        "layouts": layouts,
    }


class InvalidDraw(ValueError):
    """Inputs the variant's dims cannot make (an index into an empty dim)."""


def make_arg(
    arg: tuple[str, ...] | dict[str, Any],
    dims: dict[str, int],
    dtype: torch.dtype,
    layout: str,
    dist: str,
    gen: torch.Generator,
) -> torch.Tensor:
    if isinstance(arg, dict):
        shape = [dims[d] for d in arg["shape"]]
        if arg["dtype"] == "int" and not math.prod(shape):
            values = torch.zeros(shape, dtype=torch.int64, device=DEV)
        elif arg["dtype"] == "int" and not dims[arg["high"]]:
            raise InvalidDraw(f"indices below an empty dim {arg['high']}")
        elif arg["dtype"] == "int":
            values = torch.randint(
                0, dims[arg["high"]], shape, device=DEV, generator=gen
            )
        else:
            values = torch.rand(shape, device=DEV, generator=gen) < 0.5
        out_dtype = values.dtype
    else:
        shape = [dims[d] for d in arg]
        base = torch.randn(shape, device=DEV, dtype=torch.float64, generator=gen)
        values = DISTRIBUTIONS[dist](base, dtype, gen)
        out_dtype = dtype
    if layout == "transposed" and len(shape) >= 2:
        t = torch.empty(shape[::-1], device=DEV, dtype=out_dtype).permute(
            *range(len(shape) - 1, -1, -1)
        )
    elif layout == "sliced" and shape:
        t = torch.empty([*shape[:-1], shape[-1] + 3], device=DEV, dtype=out_dtype)[
            ..., 1 : shape[-1] + 1
        ]
    elif layout == "channels_last" and len(shape) == 4:
        t = torch.empty(shape, device=DEV, dtype=out_dtype).to(
            memory_format=torch.channels_last
        )
    elif layout == "expanded" and shape and shape[0] > 1:
        return values[:1].to(out_dtype).expand(shape)
    else:
        t = torch.empty(shape, device=DEV, dtype=out_dtype)
    return t.copy_(values)


def mean_abs(values: torch.Tensor) -> float:
    """The mean magnitude of the finite entries of `values`, without overflow
    near float64's maximum."""
    v = values.double().flatten()
    v = v[torch.isfinite(v)].abs()
    top = float(v.max()) if v.numel() else 0.0
    return float((v / top).mean()) * top if top > 0 else 0.0


def ulp_floor_scale(ref: torch.Tensor, input_scale: float) -> float:
    """The magnitude whose eps floors the ulp spacing (module docstring)."""
    v = ref[torch.isfinite(ref)].abs()
    v = v[v > 0]
    return float(v.median()) if v.numel() else input_scale


def measure(
    outs: object, refs: object, input_scale: float, like: object
) -> dict[str, float]:
    """The metrics of `outs` against the float64 `refs` (module docstring).
    `like`, eager's outputs, gives each output's type and dtype."""
    out_leaves, out_spec = tree_flatten(outs)
    ref_leaves, ref_spec = tree_flatten(refs)
    like_leaves, like_spec = tree_flatten(like)
    if not out_spec == ref_spec == like_spec:
        raise ValueError(f"outputs {out_spec} differ in structure from {like_spec}")
    for y, e in zip(out_leaves, like_leaves, strict=True):
        if type(y) is not type(e) or (
            isinstance(y, torch.Tensor) and y.dtype != e.dtype
        ):
            raise ValueError(
                f"output {y!r:.40} of another type or dtype than {e!r:.40}"
            )
    # Sums of the ulp errors and their squares, and the count of gross ones,
    # as Python numbers: the leaves may live on different devices.
    total = total_sq = 0.0
    count = gross = 0
    nonfinite = finite_count = special_bad = special_count = mismatch = ints = 0
    for y, ref in zip(out_leaves, ref_leaves, strict=True):
        if y is None:
            continue
        if isinstance(y, float):
            # A Python float is a double: measured as a 1-element float64.
            y = torch.tensor([y], dtype=torch.float64)
            ref = torch.tensor([float(ref)], dtype=torch.float64)
        elif not isinstance(y, torch.Tensor):
            mismatch += int(y != ref)
            ints += 1
            continue
        if y.shape != ref.shape or y.is_floating_point() != ref.is_floating_point():
            raise ValueError(
                f"output {y.dtype}{list(y.shape)} for {ref.dtype}{list(ref.shape)}"
            )
        if y.is_complex():
            # Its real and imaginary parts, as floats.
            y, ref = torch.view_as_real(y), torch.view_as_real(ref)
        if not y.is_floating_point():
            mismatch += int((y != ref).sum())
            ints += y.numel()
            continue
        info = torch.finfo(y.dtype)
        ref = ref.double()
        # The reference rounded to the output dtype: NaN stays NaN, and a
        # value past the dtype's range becomes inf.
        expected = ref.to(y.dtype).double()
        finite = torch.isfinite(expected)
        yd = y.double()
        nan = torch.isnan(expected)
        same = (nan & torch.isnan(yd)) | (~nan & (yd == expected))
        special_bad += int((~finite & ~same).sum())
        special_count += int((~finite).sum())
        nonfinite += int((finite & ~torch.isfinite(yd)).sum())
        finite_count += int(finite.sum())
        both = finite & torch.isfinite(yd)
        if both.any():
            # The spacing of the output dtype at the reference, eps *
            # 2**(exponent - 1) for |ref| = mantissa * 2**exponent (frexp: no
            # overflow at float64's maximum), at least the subnormal spacing.
            _, exponent = torch.frexp(expected.abs().clamp_min(info.tiny))
            spacing = torch.ldexp(torch.full_like(expected, info.eps), exponent - 1)
            scale = ulp_floor_scale(ref, input_scale)
            floor = info.eps * scale
            errors = ((yd - ref).abs() / spacing.clamp_min(floor))[both]
            total += float(errors.sum())
            total_sq += float((errors * errors).sum())
            count += errors.numel()
            err = (yd - ref).abs()
            off = (err > 0) & (err >= 0.25 * ref.abs().clamp_min(scale))
            gross += int(off[both].sum())
    mean = total / count if count else 0.0
    variance = max(total_sq / count - mean * mean, 0.0) if count else 0.0
    return {
        "ulp": mean,
        # The standard error of that mean, for comparing two noisy means.
        "ulp_se": math.sqrt(variance / count) if count else 0.0,
        "gross": gross / count if count else 0.0,
        "count": count,
        "nonfinite": nonfinite / finite_count if finite_count else 0.0,
        "special": special_bad / special_count if special_count else 0.0,
        "mismatch": mismatch / ints if ints else 0.0,
        # The counts the shares are of, for their standard errors.
        "leaves": sum(y is not None for y in out_leaves),
        "nonfinite_count": finite_count,
        "special_count": special_count,
        "mismatch_count": ints,
    }


def compiled(
    fn: Callable[..., Any], impl: str, example: dict[str, Any]
) -> Callable[..., Any]:
    torch._dynamo.reset()
    options = {"cuda_backend": "triton"} if impl == "stock" else helion_options(example)
    return torch.compile(fn, fullgraph=True, dynamic=False, options=options)


def run_variant(
    example: dict[str, Any], variant: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Every implementation on every distribution's inputs, one implementation
    at a time (each compile after a dynamo reset), eager first: its outputs
    give the types and dtypes the others must match. Each implementation gets
    inputs of its own, drawn again from the seed. Only what the function
    returns is compared: it must not write into its arguments or draw random
    numbers (module docstring)."""
    fn, spec = example["fn"], example["variants"]
    dtype = getattr(torch, variant["dtype"])
    # A function that casts to float32 itself computes its float64 reference
    # in float32 there: no more precise than eager (verdicts()).
    casts_down = bool(example.get("casts_down"))
    record: dict[str, Any] = {**variant, "casts_down": casts_down, "impls": {}}

    def make_inputs(dist: str) -> list[torch.Tensor]:
        gen = torch.Generator(device=DEV).manual_seed(seed)
        return [
            make_arg(a, variant["dims"], dtype, layout, dist, gen)
            for a, layout in zip(spec["args"], variant["layouts"], strict=True)
        ]

    inputs, refs, scales = {}, {}, {}
    for dist in DISTRIBUTIONS:
        try:
            inputs[dist] = make_inputs(dist)
        except InvalidDraw:
            record["invalid"] = traceback.format_exc()[-1500:]
            return record
        try:
            refs[dist] = fn(
                # Copies: a float64 or integer input would otherwise be the
                # very tensor the backends get, and a reference that is (a
                # view of) an input would change with a wrong in-place write.
                *[
                    a.double().clone() if a.is_floating_point() else a.clone()
                    for a in inputs[dist]
                ]
            )
        except Exception as e:
            # Out of memory or a CUDA fault is no rejection: a crash (main()).
            if isinstance(e, (torch.cuda.OutOfMemoryError, torch.AcceleratorError)):
                raise
            # The function rejects this draw (an odd dim it halves, an index
            # into an empty dim): no reference, nothing to compare.
            record["invalid"] = traceback.format_exc()[-1500:]
            return record
        scales[dist] = max(
            (mean_abs(a) for a in inputs[dist] if a.is_floating_point()), default=0.0
        )
    eager_outs: dict[str, object] = {}
    with inductor_config.patch(example.get("inductor_config", {})):
        for impl in ("eager", "stock", "helion"):
            f = fn if impl == "eager" else compiled(fn, impl, example)
            results = record["impls"][impl] = {}
            ran = False
            first_error = None
            for dist in inputs:
                args = make_inputs(dist)
                if impl != "eager" and dist not in eager_outs:
                    results[dist] = {"error": "eager failed: no dtypes to match"}
                    continue
                try:
                    if impl == "helion" and "fired" not in record:
                        out, codes = run_and_get_code(f, *args)
                        sources = "\n".join(helion_sources("\n".join(codes)))
                        fired = example.get("fired")
                        # None: compiled, with no predicate to say whether the
                        # change fired.
                        record["fired"] = bool(fired(sources)) if fired else None
                    else:
                        out = f(*args)
                    torch.cuda.synchronize()
                except Exception:
                    error = {"error": traceback.format_exc()[-1500:]}
                    results[dist] = error
                    line = exception_line(error["error"])
                    if impl != "eager" and not ran and first_error == line:
                        # The same error on two distributions before any
                        # run: a compile error, which the others would only
                        # compile again to repeat. (Eager's, and an error of
                        # one distribution's values, stay per distribution.)
                        for other in inputs:
                            results.setdefault(other, error)
                        break
                    first_error = first_error or line
                    continue
                ran = True
                if impl == "eager":
                    eager_outs[dist] = out
                try:
                    results[dist] = measure(
                        out, refs[dist], scales[dist], eager_outs[dist]
                    )
                except Exception:
                    if impl == "eager":
                        # The script cannot judge eager's own outputs against
                        # the reference: the draw is invalid, not unchecked.
                        record["invalid"] = traceback.format_exc()[-1500:]
                        return record
                    # Another structure, shape, type or dtype than eager's.
                    results[dist] = {"error": traceback.format_exc()[-1500:]}
    return record


def share_exceeds(
    helion: dict[str, Any], side: dict[str, Any], key: str, count: int
) -> str:
    """The failure of helion's share `key`, of `count` elements, against
    `side`'s, or "": within three standard errors of `side`'s, and none where
    `side` has none. Which elements a rounding tips over a boundary (an
    overflow, a comparison of rounded values) differs between two correct
    kernels; that any does where the bound has none does not."""
    bound = side[key]
    noise = 3 * math.sqrt(bound / max(count, 1))
    if helion[key] <= bound + noise:
        return ""
    return f"{key} {helion[key]:.2e} > {bound:.2e} + {noise:.2e}"


def verdicts(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The rules' failures, and the comparisons that could not run."""
    fails, unchecked = [], []
    for dist in DISTRIBUTIONS:
        eager, stock, helion = (
            record["impls"].get(impl, {}).get(dist, {"error": "not run"})
            for impl in ("eager", "stock", "helion")
        )
        if "error" in eager:
            unchecked.append(f"{dist}: eager failed")
            continue
        if not eager["leaves"]:
            # A function that returns nothing (a write into its arguments,
            # which the sweep does not compare) has nothing to check.
            unchecked.append(f"{dist}: no outputs")
            continue
        if "error" in helion:
            fails.append(f"{dist}: error")
            continue
        for key in ("special", "mismatch"):
            fail = share_exceeds(helion, eager, key, helion[f"{key}_count"])
            if fail:
                fails.append(f"{dist}: {fail} (eager)")
        # A reference no more precise than eager (a float64 variant, or a
        # function that casts to float32 itself) leaves eager's float error
        # at 0: bound such a variant's float error by stock's too.
        exact = eager["count"] > 0 and (
            record["dtype"] == "float64" or record["casts_down"]
        )
        # Where stock failed, such a variant keeps the gross check against
        # eager's alone, which a reordering does not trip, and skips the ulp.
        ulp_checked = not (exact and "error" in stock)
        bounded_by = [eager, stock] if exact and ulp_checked else [eager]
        if not ulp_checked:
            unchecked.append(f"{dist}: ulp (stock failed)")
        # Stock's share, as which input overflows depends on rounding; eager's
        # where stock failed, so that an output of NaN still fails.
        side, label = (eager, "eager") if "error" in stock else (stock, "stock")
        fail = share_exceeds(helion, side, "nonfinite", helion["nonfinite_count"])
        if fail:
            fails.append(f"{dist}: {fail} ({label})")
        if not all(math.isfinite(v) for v in helion.values()):
            fails.append(f"{dist}: non-finite metric {helion}")
            continue
        if ulp_checked:
            # Within 1 ulp of the bound, plus three standard errors of the
            # bound's mean: an output whose errors run to hundreds of ulp (a
            # cancellation) has means that differ by that much between two
            # equally good summation orders. Not helion's own standard error,
            # which a few wrong elements would inflate to cover themselves.
            side = max(bounded_by, key=operator.itemgetter("ulp"))
            noise = 3 * side["ulp_se"]
            if not helion["ulp"] <= side["ulp"] + 1 + noise:
                fails.append(
                    f"{dist}: ulp {helion['ulp']:.2f} > {side['ulp']:.2f} + 1 + {noise:.2f}"
                )
        # None where the bound has none: one element off by a quarter is a
        # wrong one (a tail tile's, a gather's), which a correct kernel never
        # gives.
        side = max(bounded_by, key=operator.itemgetter("gross"))
        fail = share_exceeds(helion, side, "gross", helion["count"])
        if fail:
            fails.append(f"{dist}: {fail}")
    return fails, list(dict.fromkeys(unchecked))


def exception_line(error: str) -> str:
    """The last "Name: message" line of a traceback."""
    lines = re.findall(r"^[A-Za-z_][\w.]*: .+$", error, re.MULTILINE)
    return lines[-1] if lines else error.strip().splitlines()[-1]


def variant_seed(seed: int, name: str, index: int) -> int:
    """The seed of variant `index` of example `name`: the same in any run."""
    return zlib.crc32(f"{seed}:{name}:{index}".encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("examples")
    parser.add_argument("out")
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--names", default=None)
    parser.add_argument("--index", type=int, default=None, help="run only this variant")
    args = parser.parse_args()
    examples = load_examples(args.examples)
    names = (
        args.names.split(",")
        if args.names
        else [n for n, e in examples.items() if "variants" in e]
    )
    indices = [args.index] if args.index is not None else range(args.variants)
    if not names:
        raise SystemExit("no example with variants to sweep")
    for name in names:
        if isinstance(examples[name]["fn"], helion.Kernel):
            raise SystemExit(f"{name}: a helion.Kernel is its own reference")
        if "variants" not in examples[name]:
            raise SystemExit(f"{name}: no variants to draw")
        spec = examples[name]["variants"]
        for arg in spec["args"]:
            shape = arg["shape"] if isinstance(arg, dict) else arg
            high = [arg["high"]] if isinstance(arg, dict) and "high" in arg else []
            known = [*shape, *high]
            if any(d not in spec["dims"] for d in known):
                raise SystemExit(f"{name}: argument {arg} names a dim not in dims")
            if isinstance(arg, dict) and arg.get("dtype") not in ("int", "bool"):
                raise SystemExit(f"{name}: argument {arg}: dtype int or bool")
            if isinstance(arg, dict) and arg["dtype"] == "int" and "high" not in arg:
                raise SystemExit(f"{name}: int argument {arg} without high")
    records: list[dict[str, Any]] = []
    for name in names:
        for i in indices:
            seed = variant_seed(args.seed, name, i)
            variant = draw_variant(examples[name]["variants"], random.Random(seed))
            try:
                ran = run_variant(examples[name], variant, seed)
            except Exception:
                # An error outside the compiled calls, as a CUDA fault that a
                # kernel left behind, which breaks every later variant too:
                # recorded as a failure, and the sweep stops with what it has.
                crash = traceback.format_exc()[-1500:]
                records.append(
                    {
                        "name": name,
                        "index": i,
                        **variant,
                        "impls": {},
                        "fails": [f"crash: {exception_line(crash)}"],
                        "unchecked": [],
                    }
                )
                print(f"{name}[{i}] CRASH {exception_line(crash)}", flush=True)
                break
            record = {"name": name, "index": i, **ran}
            if "invalid" in record:
                record["fails"], record["unchecked"] = [], ["invalid draw"]
            else:
                record["fails"], record["unchecked"] = verdicts(record)
            records.append(record)
            # The exception line of each helion.inductor error ("Name: message"),
            # not the source context Helion prints after it.
            errors = {
                exception_line(r["error"])
                for r in record["impls"].get("helion", {}).values()
                if "error" in r
            }
            status = "FAIL " + "; ".join(record["fails"]) if record["fails"] else "ok"
            eager_results = record["impls"].get("eager", {}).values()
            helion_results = record["impls"].get("helion", {}).values()
            if eager_results and all("error" in r for r in eager_results):
                status = "eager failed everywhere"
            elif (
                len(errors) == 1
                and helion_results
                and all("error" in r for r in helion_results)
            ):
                status = f"FAIL everywhere: {errors.pop()}"
            if record["unchecked"]:
                status += f" (unchecked: {'; '.join(record['unchecked'])})"
            print(
                f"{name}[{i}] {variant['dims']} {variant['dtype']} {variant['layouts']} "
                f"fired={record.get('fired')}: {status}",
                flush=True,
            )
            # Written after each variant, so that a crash keeps the records.
            with open(args.out, "w") as f:
                json.dump(records, f, indent=1)
        else:
            continue
        break
    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)
    failed = sum(bool(r["fails"]) for r in records)
    fired = sum(r.get("fired") is True for r in records)
    declined = sum(r.get("fired") is False for r in records)
    unchecked = sum("fired" in r and r["fired"] is None for r in records)
    invalid = sum("invalid" in r for r in records)
    eager_failed = sum(
        "invalid" not in r
        and "fired" not in r
        and bool(r["impls"])
        and all("error" in e for e in r["impls"].get("eager", {}).values())
        for r in records
    )
    crashed = sum(not r["impls"] and "invalid" not in r for r in records)
    print(
        f"{len(records)} variants: {failed} failing, {fired} fired, "
        f"{declined} declined, {unchecked} with no fired predicate, "
        f"{invalid} invalid draws, {eager_failed} where eager failed, "
        f"{crashed} crashed, "
        f"{len(records) - fired - declined - unchecked - invalid - eager_failed - crashed}"
        " not compiled"
    )
    # An example none of whose variants compared anything (every draw
    # invalid, eager failing, nothing returned) fails too: no evidence.

    def checked(r: dict[str, Any]) -> bool:
        compared = [
            d for d in DISTRIBUTIONS if f"{d}: no outputs" not in r["unchecked"]
        ]
        return (
            "invalid" not in r
            and bool(r["impls"])
            and any(
                "error" not in r["impls"]["eager"].get(d, {"error": ""})
                for d in compared
            )
        )

    nothing = [
        n for n in names if not any(checked(r) for r in records if r["name"] == n)
    ]
    for name in nothing:
        print(f"{name}: no variant compared anything")
    sys.exit(1 if failed or nothing else 0)


if __name__ == "__main__":
    main()
