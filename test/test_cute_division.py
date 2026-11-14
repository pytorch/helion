"""How the CuTe backend divides, on CPU renders.

A 32-bit or 16-bit float ``/`` is the Triton backend's ``div.full.f32``: the
generated code keeps ``/`` and the launcher compiles the kernel with
libNVVM's ``-prec-div=0``.  ``Settings.cute_exact_division`` compiles it with
the IEEE division instead, and a quotient its node truncates or floors is
the IEEE one in every mode.  ``test_cute_division_cuda.py`` checks the
results on a GPU.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import IEEE_DIVISION_LAUNCH
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl
from helion.runtime.cute.compile_cache import FULL_RANGE_DIVISION_COMPILE_OPTION
from helion.runtime.cute.compile_cache import IEEE_DIVISION_COMPILE_OPTION
from helion.runtime.cute.compile_cache import compile_with_options
from helion.runtime.cute.compile_cache import merge_default_compile_options

pytestmark = pytest.mark.cpu_only

IEEE_QUOTIENT = "rounding=cute.math.RoundingMode.NEAREST_EVEN"


def _divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


def _reciprocal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.reciprocal(y[tile])
    return out


def _truncated(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="trunc")
    return out


def _truncated_by_a_scalar(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[0], rounding_mode="trunc")
    return out


def _divide_by_a_scalar(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[0]
    return out


def _floored_by_a_scalar(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] // y[0]
    return out


def _truncated_in_place(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        quotient = x[tile]
        quotient.div_(y[tile], rounding_mode="trunc")
        out[tile] = quotient
    return out


def _truncated_by_a_host_scalar(x: torch.Tensor, divisor: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], divisor, rounding_mode="trunc")
    return out


def _divided_in_place(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        quotient = x[tile]
        quotient.div_(y[tile])
        out[tile] = quotient
    return out


def _row_normalize(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] / x[tile, :].sum(dim=1, keepdim=True)
    return out


def _code(
    fn: object,
    dtype: torch.dtype,
    *,
    divisor_dtype: torch.dtype | None = None,
    config: helion.Config | None = None,
    **settings: Any,
) -> str:
    shape = (64, 128) if fn is _row_normalize else (4096,)
    x = torch.empty(shape, dtype=dtype)
    y: object = torch.empty(shape, dtype=divisor_dtype or dtype)
    if fn is _truncated_by_a_host_scalar:
        y = 3.0
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[exc.FastMathUnused],
        **settings,
    )
    with cpu_target():
        bound = cpu_bind(kernel, (x, y))
        return bound.to_code(config or bound.config_spec.default_config())


def _launch(code: str) -> str:
    (line,) = [line for line in code.splitlines() if "_launcher(_helion_" in line]
    return line


@pytest.mark.parametrize(
    ("fn", "dtype"),
    [
        (_divide, torch.float32),
        (_divide, torch.bfloat16),
        (_divide, torch.float64),
        (_reciprocal, torch.float32),
        (_row_normalize, torch.float32),
        (_row_normalize, torch.bfloat16),
    ],
)
def test_a_division_is_spelled_as_written(fn: object, dtype: torch.dtype) -> None:
    """The kernel's ``/`` stays ``/``: what it compiles to is the launcher's
    choice, so the launch names no option by default and the IEEE division
    under ``cute_exact_division``."""
    code = _code(fn, dtype)
    assert " / " in code
    assert "cute.math.div(" not in code
    assert "cute_compile_options" not in _launch(code)
    exact = _code(fn, dtype, cute_exact_division=True)
    assert _launch(exact).endswith(IEEE_DIVISION_LAUNCH + ")")
    assert "cute.math.div(" not in exact


@pytest.mark.parametrize(
    ("dtype", "truncated"),
    [
        (torch.float32, "v_1 = cute.math.trunc(v_0)"),
        # Eager divides two 16-bit tensors in their dtype: the quotient is
        # rounded to it before it is truncated.
        (torch.bfloat16, "v_1 = cutlass.Float32(cutlass.BFloat16(v_0))"),
    ],
    ids=["fp32", "bf16"],
)
def test_a_truncated_quotient_is_the_ieee_one(
    dtype: torch.dtype, truncated: str
) -> None:
    """``div.full`` gives 33 / 33 just below 1, which truncates to 0: a
    quotient its node rounds to an integer is ``div.rn``, spelled as the
    intrinsic the launcher's division option does not change.  Under
    ``fast_math`` too, whose approximate division is for plain quotients,
    and under ``cute_exact_division``, with or without ``fast_math``."""
    quotient = (
        "v_0 = cute.math.div(cutlass.Float32(load), cutlass.Float32(load_1), "
        f"{IEEE_QUOTIENT})"
    )
    for settings in (
        {},
        {"fast_math": True},
        {"cute_exact_division": True},
        {"cute_exact_division": True, "fast_math": True},
    ):
        code = _code(_truncated, dtype, **settings)
        assert quotient in code and truncated in code, code
        assert "approx=True" not in code


def test_a_quotient_computed_in_fp32_is_truncated_in_fp32() -> None:
    """A bf16 ``x.div_(y, rounding_mode="trunc")`` by an fp32 ``y`` divides
    in fp32, and eager truncates an fp32 product for a host scalar divisor:
    neither quotient is rounded to bf16 before it is truncated (2.99985 is
    3 in bf16)."""
    truncated_as_divided = re.compile(
        r"(v_\d+) = cute\.math\.div\(.*NEAREST_EVEN\)\n\s+v_\d+ = cute\.math\.trunc\(\1\)"
    )
    for code in (
        _code(_truncated_in_place, torch.bfloat16, divisor_dtype=torch.float32),
        _code(_truncated_by_a_host_scalar, torch.bfloat16),
    ):
        assert truncated_as_divided.search(code), code


def test_fast_math_keeps_a_rounded_quotient_by_an_invariant_divisor() -> None:
    """``fast_math`` multiplies by the reciprocal of a divisor that does not
    change across a lane loop, and 41 * RN(1 / 41) truncates to 0: the
    intrinsic is not a ``/`` it rewrites, under ``cute_exact_division``
    either (where the quotient used to be that ``/``)."""
    lanes = helion.Config(block_sizes=[1024], num_threads=[128])
    for fn, dtype in (
        (_truncated_by_a_scalar, torch.float32),
        (_truncated_by_a_scalar, torch.bfloat16),
        (_floored_by_a_scalar, torch.float32),
    ):
        for exact in (False, True):
            code = _code(
                fn, dtype, config=lanes, fast_math=True, cute_exact_division=exact
            )
            assert IEEE_QUOTIENT in code, code
            assert "_helion_inv_div" not in code and "approx=True" not in code
    # A plain 16-bit quotient by the same divisor is a ``/`` it rewrites.
    assert "_helion_inv_div" in _code(
        _divide_by_a_scalar, torch.bfloat16, config=lanes, fast_math=True
    )


def test_a_float64_truncated_quotient_keeps_its_division() -> None:
    # A float64 ``/`` is ``div.rn.f64`` under either division option.
    code = _code(_truncated, torch.float64)
    assert IEEE_QUOTIENT not in code and "v_0 = load / load_1" in code


def test_fast_math_divides_approximately_in_fp32() -> None:
    code = _code(_divide, torch.float32, fast_math=True)
    assert "cute.math.div(load, load_1, approx=True, ftz=True)" in code
    # A 16-bit quotient is not an fp32 scalar: it keeps ``/``.
    assert "approx=True" not in _code(_divide, torch.bfloat16, fast_math=True)
    # ``x.div_(y)`` keeps the dtype of ``x``, but a float64 ``y`` promotes the
    # division: its operands are float64, which the intrinsic does not take.
    assert "approx=True" in _code(_divided_in_place, torch.float32, fast_math=True)
    mixed = _code(
        _divided_in_place, torch.float32, divisor_dtype=torch.float64, fast_math=True
    )
    assert "approx=True" not in mixed and "= v_0 / load_1" in mixed, mixed


def test_the_exact_division_knob_needs_the_setting() -> None:
    """``cute_exact_div_form`` picks between two IEEE forms, so it is
    searched only under ``cute_exact_division``; a recorded config that
    names it still loads without the setting, and the knob is dropped."""
    x = torch.empty(64, 128, dtype=torch.float32)
    for exact in (False, True):
        kernel = helion.kernel(
            _row_normalize,
            backend="cute",
            static_shapes=True,
            cute_exact_division=exact,
            ignore_warnings=[exc.FastMathUnused],
        )
        with cpu_target():
            bound = cpu_bind(kernel, (x, x))
            spec = bound.config_spec
            assert spec.cute_exact_division_sites is exact
            recorded = spec.default_config().config | {"cute_exact_div_form": "redo"}
            normalized = spec.normalized_config(helion.Config(**recorded))
            assert ("cute_exact_div_form" in normalized.config) is exact
            code = bound.to_code(normalized)
        assert ("_cute_exact_div_redo_f32(" in code) is exact


def test_the_launcher_compiles_with_full_range_division_by_default() -> None:
    merged = merge_default_compile_options(None)
    assert merged.split() == ["--enable-tvm-ffi", FULL_RANGE_DIVISION_COMPILE_OPTION]
    assert merge_default_compile_options(merged) == merged
    # A kernel that chose the IEEE division keeps it, next to its options.
    ptxas = "--ptxas-options='--override-directive-values --maxrregcount=48'"
    chosen = f"{IEEE_DIVISION_COMPILE_OPTION} {ptxas}"
    merged = merge_default_compile_options(chosen)
    assert merged == f"{chosen} --enable-tvm-ffi"
    assert FULL_RANGE_DIVISION_COMPILE_OPTION not in merged


class _RecordingCompile:
    """``cute.compile``: records how it was called."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("string", args, kwargs))
        return "compiled"

    def __getitem__(self, options: object) -> Any:
        def compile_(*args: object) -> str:
            self.calls.append(("options", options, args))
            return "compiled"

        return compile_


def test_the_division_option_reaches_the_compiler_as_a_raw_pass_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DSL's option string cannot name a libNVVM option: the launcher
    passes the DSL's option objects, the division option as a raw pass
    option next to the public classes for the options a kernel launches
    with."""
    from cutlass.base_dsl import compiler
    import cutlass.cute as cute

    compile_ = _RecordingCompile()
    monkeypatch.setattr(cute, "compile", compile_)
    ptxas = "--override-directive-values --maxrregcount=48"
    launched = f"--opt-level 2 --ptxas-options='{ptxas}' --generate-line-info"
    for chosen, division in (
        ("", FULL_RANGE_DIVISION_COMPILE_OPTION),
        (IEEE_DIVISION_COMPILE_OPTION + " ", IEEE_DIVISION_COMPILE_OPTION),
    ):
        compile_.calls.clear()
        options = merge_default_compile_options(chosen + launched)
        assert compile_with_options("jit", ("a", "b"), options) == "compiled"
        ((kind, parsed, args),) = compile_.calls
        assert (kind, args) == ("options", ("jit", "a", "b"))
        # The DSL accepts them as the options of one compile.
        compiler.CompileOptions(tuple(parsed))  # type: ignore[arg-type]
        assert {type(option): option.value for option in parsed} == {  # type: ignore[attr-defined]
            cute.OptLevel: 2,
            cute.PtxasOptions: ptxas,
            cute.GenerateLineInfo: True,
            cute.EnableTVMFFI: True,
            compiler.ExtraCompilerOpts: division,
        }
    # Another option, or two division options, is an error rather than a
    # kernel compiled with a division nobody chose.
    for bad in (
        "--keep-ptx",
        "--opt-level",
        f"{FULL_RANGE_DIVISION_COMPILE_OPTION} {IEEE_DIVISION_COMPILE_OPTION}",
        "nvvm-options=-ftz=1",
    ):
        with pytest.raises(ValueError, match="option"):
            compile_with_options("jit", ("a",), merge_default_compile_options(bad))
    # A caller that skipped the merge still gets the launcher's defaults.
    compile_.calls.clear()
    compile_with_options("jit", ("a",), "--generate-line-info")
    ((_, parsed, _),) = compile_.calls
    assert {type(option): option.value for option in parsed} == {  # type: ignore[attr-defined]
        cute.GenerateLineInfo: True,
        cute.EnableTVMFFI: True,
        compiler.ExtraCompilerOpts: FULL_RANGE_DIVISION_COMPILE_OPTION,
    }
    # A launcher built without options (a test double) compiles as written.
    compile_.calls.clear()
    compile_with_options("jit", ("a",), None)
    assert compile_.calls == [("string", ("jit", "a"), {})]


def _recorded(compile_: _RecordingCompile, call: Any, *args: object) -> list[object]:
    """What ``call(*args)`` gave ``cute.compile``, in a comparable form."""
    compile_.calls.clear()
    call(*args)
    ((kind, parsed, compiled),) = compile_.calls
    options = sorted((type(option).__name__, option.value) for option in parsed)  # type: ignore[attr-defined]
    return [kind, compiled, options]


def test_the_standalone_launcher_compiles_as_the_launcher_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Helion-free standalone launcher carries its own copy of the option
    handling (the exporter splices the module's text into the exported
    file): both launchers give ``cute.compile`` the same options, the default
    division included, and refuse the same ones."""
    import cutlass.cute as cute

    from helion.runtime.cute import standalone_launcher

    compile_ = _RecordingCompile()
    monkeypatch.setattr(cute, "compile", compile_)

    def launcher(options: str | None) -> object:
        merged = merge_default_compile_options(options)
        return compile_with_options("jit", ("a", "b"), merged)

    def standalone(options: str | None) -> object:
        return standalone_launcher._standalone_compile("jit", ("a", "b"), options)

    ptxas = "--ptxas-options='--override-directive-values --maxrregcount=48'"
    for options in (
        None,
        IEEE_DIVISION_COMPILE_OPTION,
        f"--opt-level 2 {ptxas} --generate-line-info",
        f"{IEEE_DIVISION_COMPILE_OPTION} {ptxas}",
    ):
        expected = _recorded(compile_, launcher, options)
        assert _recorded(compile_, standalone, options) == expected, options
    for bad in (
        "--keep-ptx",
        "--opt-level",
        f"{FULL_RANGE_DIVISION_COMPILE_OPTION} {IEEE_DIVISION_COMPILE_OPTION}",
        "nvvm-options=-ftz=1",
        "nvvm-options=-prec-div=2",
    ):
        for compile_under in (launcher, standalone):
            with pytest.raises(ValueError, match="option"):
                compile_under(bad)
