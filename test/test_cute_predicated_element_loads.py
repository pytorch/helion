"""GPU-free coverage of ``cute_predicated_element_loads``: the pass that
loads a masked element with one predicated instruction and gives it the
constant of the select it feeds
(``helion/_compiler/cute/predicated_element_loads.py``), the gathered
coordinate bounds the load's own mask states (``gathered_bounds.py``), the
knob's offer, its coverage witnesses and the N-D gather tile seeds."""

from __future__ import annotations

import ast
import textwrap
from types import SimpleNamespace

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.element_policy_loads import _assembly
from helion._compiler.cute.gathered_bounds import _coordinate_lanes
from helion._compiler.cute.gathered_bounds import mask_bounded_gather_positions
from helion._compiler.cute.predicated_element_loads import predicate_element_loads
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.guarded_lane_kernels import padded_sum
from helion._testing.guarded_lane_kernels import window_max
from helion._testing.predicated_load_kernels import gathered_or_fill
from helion._testing.predicated_load_kernels import gathered_under_row_mask
from helion._testing.predicated_load_kernels import interior_or_fill_in_place
from helion._testing.predicated_load_kernels import padded_window_max
from helion._testing.predicated_load_kernels import strided_pair_sum
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_KNOB = "cute_predicated_element_loads"
_LOAD = "_cute_element_fill_load_if"
_NEG_INF_BF16 = 0xFF80


def _predicate(
    source: str,
    hints: dict[str, tuple[str, str] | None] | None = None,
    *,
    x: torch.dtype = torch.bfloat16,
    y: torch.dtype = torch.float32,
) -> str:
    """The body of ``def kernel(x, y, m, n, i, s, q): <source>`` after the
    pass, ``x`` and ``y`` being its global tensors, of the given dtypes."""
    definition = ast.parse(
        "def kernel(x, y, m, n, i, s, q):\n"
        + textwrap.indent(textwrap.dedent(source).strip("\n"), "    ")
    ).body[0]
    assert isinstance(definition, ast.FunctionDef)
    body = predicate_element_loads(definition, {"x": x, "y": y}, hints or {})
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _same(source: str, *, x: torch.dtype = torch.float32) -> None:
    """The pass leaves ``source`` as it is (``x`` an fp32 tensor unless
    given)."""
    assert _predicate(source, x=x) == ast.unparse(ast.parse(textwrap.dedent(source)))


# ---------------------------------------------------------------------------
# The pass on its own
# ---------------------------------------------------------------------------


def test_masked_load_becomes_one_predicated_load() -> None:
    """``load if mask else T(0)`` loads predicated and leaves zero bits; the
    mask may be any expression, and a load in a nested block is taken
    too."""
    assert _predicate(
        """
        a = (x.iterator + i * s).load() if m and n < 4 else cutlass.Float32(0)
        if q:
            b = (y.iterator + cutlass.Int32(i) * cutlass.Int32(y.layout.stride[0])).load() if m else cutlass.Int16(0)
        """,
        x=torch.float32,
        y=torch.int16,
    ) == ast.unparse(
        ast.parse(
            textwrap.dedent(
                f"""
                a = {_LOAD}(x.iterator + i * s, m and n < 4, cutlass.Float32, 0)
                if q:
                    b = {_LOAD}(y.iterator + cutlass.Int32(i) * cutlass.Int32(y.layout.stride[0]), m, cutlass.Int16, 0)
                """
            )
        )
    )


def test_select_under_the_load_mask_gives_the_fill() -> None:
    """The window tap: a widening cast, then a select under the load's mask
    against ``-inf``.  The load leaves bf16 ``-inf`` and the select goes."""
    assert _predicate(
        """
        fill = cutlass.Float32(float('-inf'))
        load = (x.iterator + i).load() if m else cutlass.BFloat16(0)
        wide = cutlass.Float32(load)
        tap = cutlass.Float32(wide) if m else cutlass.Float32(fill)
        best = cute.arch.fmax(q, tap, nan=True)
        """
    ) == ast.unparse(
        ast.parse(
            textwrap.dedent(
                f"""
                fill = cutlass.Float32(float('-inf'))
                load = {_LOAD}(x.iterator + i, m, cutlass.BFloat16, {_NEG_INF_BF16})
                wide = cutlass.Float32(load)
                tap = cutlass.Float32(wide)
                best = cute.arch.fmax(q, tap, nan=True)
                """
            )
        )
    )


# One case per element type and per way a constant is spelled: the fill is
# the element type's bits of the constant.
@pytest.mark.parametrize(
    ("dtype", "select", "bits"),
    [
        ("BFloat16", "cutlass.Float32(load) if m else cutlass.Float32(0.5)", 0x3F00),
        ("BFloat16", "cutlass.Float64(load) if m else -0.0", 0x8000),
        ("Float16", "cutlass.Float32(load) if m else float('inf')", 0x7C00),
        ("Float32", "cutlass.Float32(load) if m else cutlass.Float32(2.5)", 0x40200000),
        ("Float32", "load if m else 1", 0x3F800000),
        ("Float64", "load if m else -2.0", 0xC000000000000000),
        ("Int16", "cutlass.Int32(load) if m else -1", 0xFFFF),
        ("Int32", "cutlass.Int32(load) if m else cutlass.Int32(-7)", 0xFFFFFFF9),
        ("Int64", "load if m else 5", 5),
    ],
)
def test_fill_is_the_constant_in_the_element_type(
    dtype: str, select: str, bits: int
) -> None:
    code = _predicate(
        f"""
        load = (x.iterator + i).load() if m else cutlass.{dtype}(0)
        tap = {select}
        """,
        x=getattr(torch, dtype.lower()),
    )
    kept = select.split(" if ")[0]
    assert code == ast.unparse(
        ast.parse(
            f"load = {_LOAD}(x.iterator + i, m, cutlass.{dtype}, {bits})\ntap = {kept}"
        )
    )


# Each decline of the fill: the load is still predicated, with zero bits,
# and the select stays.  One case per condition of the proof.
@pytest.mark.parametrize(
    "tail",
    [
        # The constant is not a value of the element type.
        "tap = cutlass.Float32(load) if m else cutlass.Float32(0.1)",
        "tap = cutlass.Float32(load) if m else 3.0e38",
        "tap = cutlass.Int32(load) if m else 70000",
        # A NaN.
        "tap = cutlass.Float32(load) if m else float('nan')",
        # Not a constant.
        "tap = cutlass.Float32(load) if m else q",
        "tap = cutlass.Float32(load) if m else cutlass.Float32(q)",
        # Another mask, a stronger one, a weaker one.
        "tap = cutlass.Float32(load) if n else cutlass.Float32(1.0)",
        "tap = cutlass.Float32(load) if m and n else cutlass.Float32(1.0)",
        # The select computes with the value.
        "tap = cutlass.Float32(load) + q if m else cutlass.Float32(1.0)",
        "tap = cutlass.Float32(1.0) if m else cutlass.Float32(load)",
        # A second reader of the load, of its cast.
        "tap = cutlass.Float32(load) if m else cutlass.Float32(1.0)\nother = load",
        (
            "wide = cutlass.Float32(load)\n"
            "tap = wide if m else cutlass.Float32(1.0)\nother = wide + 1"
        ),
        # The select in another statement list than the load.
        "if q:\n    tap = cutlass.Float32(load) if m else cutlass.Float32(1.0)",
        # A cast that does not hold every value of its operand.
        "tap = cutlass.Float16(load) if m else cutlass.Float16(1.0)",
        "tap = cutlass.Int32(load) if m else 1",
        "wide = cutlass.Float32(load)\ntap = cutlass.BFloat16(wide) if m else 1.0",
        # The cast's name is bound twice; so is the load's.
        (
            "wide = cutlass.Float32(load)\nwide = wide\n"
            "tap = wide if m else cutlass.Float32(1.0)"
        ),
        (
            "if q:\n    load = cutlass.BFloat16(2.0)\n"
            "tap = cutlass.Float32(load) if m else cutlass.Float32(1.0)"
        ),
        # A constant's name is bound twice, or is a parameter the body
        # binds once (its first value is the caller's); a bool is no number.
        ("c = 1.0\nif q:\n    c = 2.0\ntap = cutlass.Float32(load) if m else c"),
        "tap = cutlass.Float32(load) if m else s\ns = 1.0",
        "tap = cutlass.Float32(load) if m else True",
        # No select at all.
        "tap = cutlass.Float32(load) + q",
    ],
)
def test_select_that_is_not_proven_keeps_its_place(tail: str) -> None:
    source = (
        "load = (x.iterator + i).load() if m else cutlass.BFloat16(0)\n" + tail + "\n"
    )
    expected = f"load = {_LOAD}(x.iterator + i, m, cutlass.BFloat16, 0)\n" + tail + "\n"
    assert _predicate(source) == ast.unparse(ast.parse(expected))


def test_constant_of_another_type_than_the_value_keeps_the_select() -> None:
    """A select takes the wider of its operands' types, so its value alone
    has the select's type only beside a bare literal of its kind or a
    constant cast to the value's own type; and the constant must be a value
    of the element type that the casts behind the load give back."""
    for dtype, select in (
        ("BFloat16", "load if m else cutlass.Float32(1.0)"),
        ("Int16", "cutlass.Int32(load) if m else cutlass.Int64(2)"),
        ("Int32", "load if m else 2.0"),
        # An integer cast of a constant that is no integer: not known to
        # be one value.
        ("Int32", "load if m else cutlass.Int32(2.5)"),
        ("Int32", "load if m else cutlass.Int32(float('inf'))"),
        # An integer the element type does not hold, and a cast to a
        # narrower integer, which does not hold every element.
        ("Int16", "cutlass.Int32(load) if m else 70000"),
        ("Int32", "cutlass.Int16(load) if m else 1"),
    ):
        source = f"load = (x.iterator + i).load() if m else cutlass.{dtype}(0)\n"
        expected = f"load = {_LOAD}(x.iterator + i, m, cutlass.{dtype}, 0)\n"
        assert _predicate(
            source + f"tap = {select}\n", x=getattr(torch, dtype.lower())
        ) == ast.unparse(ast.parse(expected + f"tap = {select}\n"))


def test_half_constant_that_rounds_two_ways_keeps_the_select() -> None:
    """A constant cast to bf16 or fp16 that lies within an fp32 rounding of
    a tie of the half type is one value rounded once (the DSL's cast of a
    Python float) and another rounded through fp32 (a device conversion,
    eager PyTorch): the fill is not known, so the select stays.  A constant
    on which both agree is folded: an exact tie, a value that is no tie."""
    cases = [
        # 1 + 2**-8 + 2**-40: 1 + 2**-7 once, 1 through fp32.
        ("BFloat16", "cutlass.BFloat16(1.0039062500009095)", None),
        ("BFloat16", "cutlass.BFloat16(cutlass.BFloat16(1.0039062500009095))", None),
        # 1 + 2**-11 + 2**-40: 1 + 2**-10 once, 1 through fp32.
        ("Float16", "cutlass.Float16(1.0004882812509095)", None),
        # The exact tie goes to even both ways; 0.1 is no tie.
        ("BFloat16", "cutlass.BFloat16(1.00390625)", 0x3F80),
        ("Float16", "cutlass.Float16(1.00048828125)", 0x3C00),
        ("BFloat16", "cutlass.BFloat16(0.1)", 0x3DCD),
        # Past the type's range both ways: its infinity.
        ("Float16", "cutlass.Float16(1e9)", 0x7C00),
    ]
    for dtype, constant, bits in cases:
        code = _predicate(
            f"""
            load = (x.iterator + i).load() if m else cutlass.{dtype}(0)
            tap = load if m else {constant}
            """,
            x=getattr(torch, dtype.lower()),
        )
        if bits is None:
            assert f"load = {_LOAD}(x.iterator + i, m, cutlass.{dtype}, 0)\n" in code
            assert f"tap = load if m else {constant}" in code
        else:
            assert f"cutlass.{dtype}, {bits})\ntap = load" in code + "\n", code


def test_nan_constant_node_keeps_the_select() -> None:
    """A NaN spelled as a constant node, as generated code can hold one."""
    definition = ast.parse(
        "def kernel(x, m, i):\n"
        "    load = (x.iterator + i).load() if m else cutlass.Float32(0)\n"
        "    tap = load if m else 1.0\n"
    ).body[0]
    assert isinstance(definition, ast.FunctionDef)
    select = definition.body[1]
    assert isinstance(select, ast.Assign) and isinstance(select.value, ast.IfExp)
    select.value.orelse = ast.Constant(value=float("nan"))
    code = ast.unparse(
        ast.Module(
            body=predicate_element_loads(definition, {"x": torch.float32}, {}),
            type_ignores=[],
        )
    )
    assert f"load = {_LOAD}(x.iterator + i, m, cutlass.Float32, 0)" in code
    assert "tap = load if m else " in code


def test_mask_bound_between_the_load_and_the_select_keeps_the_select() -> None:
    """The select reads another value of the mask than the load did: a
    rebinding between them, also one nested in a loop whose target carries
    no Store context."""
    rebound = """
        load = (x.iterator + i).load() if m else cutlass.BFloat16(0)
        m = n
        tap = cutlass.Float32(load) if m else cutlass.Float32(1.0)
        """
    assert f"{_LOAD}(x.iterator + i, m, cutlass.BFloat16, 0)" in _predicate(rebound)
    assert "if m else cutlass.Float32(1.0)" in _predicate(rebound)
    looped = """
        load = (x.iterator + i).load() if n < k else cutlass.BFloat16(0)
        for k in range(2):
            pass
        tap = cutlass.Float32(load) if n < k else cutlass.Float32(1.0)
        """
    definition = ast.parse(
        "def kernel(x, n, i):\n" + textwrap.indent(textwrap.dedent(looped), "    ")
    ).body[0]
    assert isinstance(definition, ast.FunctionDef)
    for node in ast.walk(definition):
        if isinstance(node, ast.For):
            assert isinstance(node.target, ast.Name)
            node.target.ctx = ast.Load()
    code = ast.unparse(
        ast.Module(
            body=predicate_element_loads(definition, {"x": torch.bfloat16}, {}),
            type_ignores=[],
        )
    )
    assert f"{_LOAD}(x.iterator + i, n < k, cutlass.BFloat16, 0)" in code
    assert "if n < k else cutlass.Float32(1.0)" in code


def test_mask_that_reads_memory_keeps_the_select() -> None:
    """Two evaluations of a test that calls or subscripts may differ (a
    store between them); a name holding such a value is one value."""
    for test in ("flags[i]", "probe(n)", "y.shape[0] > i"):
        code = _predicate(
            f"""
            load = (x.iterator + i).load() if {test} else cutlass.BFloat16(0)
            tap = cutlass.Float32(load) if {test} else cutlass.Float32(1.0)
            """
        )
        assert f"{_LOAD}(x.iterator + i, {test}, cutlass.BFloat16, 0)" in code
        assert f"if {test} else cutlass.Float32(1.0)" in code
    code = _predicate(
        """
        kept = flags[i] != cutlass.Uint8(0)
        load = (x.iterator + i).load() if kept else cutlass.BFloat16(0)
        (x.iterator + i).store(cutlass.BFloat16(q))
        tap = cutlass.Float32(load) if kept else cutlass.Float32(1.0)
        """
    )
    assert f"{_LOAD}(x.iterator + i, kept, cutlass.BFloat16, 16256)" in code
    assert "tap = cutlass.Float32(load)\n" in code + "\n"


# Each decline of the predicated form: the statement stays as it was.
@pytest.mark.parametrize(
    "source",
    [
        # An address that reads memory, calls a helper or selects.
        "a = (x.iterator + table[i]).load() if m else cutlass.Float32(0)",
        "a = (x.iterator + offset(i)).load() if m else cutlass.Float32(0)",
        "a = (x.iterator + (i if m else 0)).load() if m else cutlass.Float32(0)",
        # Not an element of a tensor parameter: a named pointer, a tensor
        # that is no parameter, two iterators.
        "a = p.load() if m else cutlass.Float32(0)",
        "a = (z.iterator + i).load() if m else cutlass.Float32(0)",
        "a = (x.iterator + y.iterator).load() if m else cutlass.Float32(0)",
        # A parameter the body rebinds.
        "x = y\na = (x.iterator + i).load() if m else cutlass.Float32(0)",
        # A masked-off value that is not the zero of an element type the
        # helper loads: a byte, a bool compare, another constant, a name.
        "a = (x.iterator + i).load() if m else cutlass.Uint8(0)",
        (
            "a = (x.iterator + i).load() != cutlass.Uint8(0) if m "
            "else cutlass.Boolean(0)"
        ),
        "a = (x.iterator + i).load() if m else cutlass.Float32(1)",
        "a = (x.iterator + i).load() if m else q",
        # A load with a cache hint of its own, a packet load.
        (
            "a = cute.arch.load(x.iterator + i, cutlass.Float32, "
            "level1_eviction_priority='evict_last') if m else cutlass.Float32(0)"
        ),
        "a = cute.arch.load(x.iterator + i, vt) if m else cutlass.Float32(0)",
        # Not the whole right-hand side of an assignment to a name.
        "a = cutlass.Float32((x.iterator + i).load() if m else cutlass.BFloat16(0))",
        "a = q + ((x.iterator + i).load() if m else cutlass.Float32(0))",
        "a[0] = (x.iterator + i).load() if m else cutlass.Float32(0)",
        # A test the DSL decides while tracing: no mask in the traced kernel.
        "a = (x.iterator + i).load() if True else cutlass.Float32(0)",
        "on = True\na = (x.iterator + i).load() if on else cutlass.Float32(0)",
        (
            "on = True\n"
            "a = (x.iterator + i).load() if on and not False else cutlass.Float32(0)"
        ),
        (
            "a = (x.iterator + i).load() if cutlass.const_expr(s > 1) "
            "else cutlass.Float32(0)"
        ),
        # A load call that takes arguments.
        "a = (x.iterator + i).load(q) if m else cutlass.Float32(0)",
        "a = (x.iterator + i).load(order=q) if m else cutlass.Float32(0)",
        # A plain load.
        "a = (x.iterator + i).load()",
    ],
)
def test_statement_is_left_alone(source: str) -> None:
    _same(source)


def test_masked_off_value_of_another_type_than_the_tensor_is_left_alone() -> None:
    """The helper reads an element of ``T``'s width, so ``T`` must be the
    tensor's dtype: a 4-byte zero on a bf16 tensor, a 2-byte one on an fp32
    tensor, another 2-byte float, and a pointer that is no global tensor
    parameter stay conditionals."""
    for dtype, tensor in (
        ("Float32", torch.bfloat16),
        ("BFloat16", torch.float32),
        ("Float16", torch.bfloat16),
        ("Int32", torch.float32),
    ):
        _same(f"a = (x.iterator + i).load() if m else cutlass.{dtype}(0)", x=tensor)
    _same("a = (q.iterator + i).load() if m else cutlass.Float32(0)")


def test_unsigned_and_byte_elements_are_left_alone() -> None:
    """The predicated form is for 2-, 4- and 8-byte floats and signed
    integers: an unsigned integer of any width, a byte and a bool keep
    their conditional on a tensor of that very dtype, where nothing but the
    element type declines them."""
    for dtype, tensor in (
        ("Uint16", torch.uint16),
        ("Uint32", torch.uint32),
        ("Uint64", torch.uint64),
        ("Uint8", torch.uint8),
        ("Int8", torch.int8),
        ("Boolean", torch.bool),
    ):
        _same(f"a = (x.iterator + i).load() if m else cutlass.{dtype}(0)", x=tensor)


def test_name_bound_twice_is_a_mask() -> None:
    """A name with two bindings is not known to be a trace-time value,
    whether or not the last of them is a literal: its load is a masked load
    like any other.  So is a load under a parameter that the body binds
    once to a literal: its first value is the caller's."""
    for bindings in (
        "on = True\nif q:\n    on = m\n",
        "if q:\n    on = m\nelse:\n    on = True\n",
    ):
        code = _predicate(
            bindings + "a = (x.iterator + i).load() if on else cutlass.Float32(0)",
            x=torch.float32,
        )
        assert f"a = {_LOAD}(x.iterator + i, on, cutlass.Float32, 0)" in code
    code = _predicate(
        """
        for k in range(n):
            a = (x.iterator + i).load() if q else cutlass.Float32(0)
            q = True
        """,
        x=torch.float32,
    )
    assert f"a = {_LOAD}(x.iterator + i, q, cutlass.Float32, 0)" in code


def test_predicated_load_carries_the_tensor_hint() -> None:
    """A tensor whose element loads take a cache policy
    (``element_load_hints``) keeps it on its predicated loads that take a
    fill; a hint recorded for another element type, or none, adds nothing.
    A hinted load whose fill stays zero is left as it is, for
    ``hint_element_loads``, which writes the same statement without the
    knob."""
    source = """
        a = (x.iterator + i).load() if m else cutlass.BFloat16(0)
        b = (y.iterator + i).load() if m else cutlass.Float32(0)
        t = cutlass.Float32(a) if m else cutlass.Float32(float('-inf'))
        """
    code = _predicate(
        source,
        {"x": ("cutlass.BFloat16", "last"), "y": ("cutlass.BFloat16", "l2_last")},
    )
    assert (
        f"a = {_LOAD}(x.iterator + i, m, cutlass.BFloat16, {_NEG_INF_BF16}, 'last')"
        in code
    )
    assert f"b = {_LOAD}(y.iterator + i, m, cutlass.Float32, 0)\n" in code
    assert "'last'" not in _predicate(source, {"x": None})
    hinted = _predicate(source, {"y": ("cutlass.Float32", "l2_last")})
    assert "b = (y.iterator + i).load() if m else cutlass.Float32(0)\n" in hinted
    assert (
        f"a = {_LOAD}(x.iterator + i, m, cutlass.BFloat16, {_NEG_INF_BF16})" in hinted
    )


def test_fill_assembly() -> None:
    """``mov fill; @p ld.global``: the fill in hex, a zero fill and a cache
    policy as the hinted predicated loads spell them."""
    assert _assembly("", 16, predicated=True, fill=_NEG_INF_BF16) == (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0xff80; "
        "@p ld.global.b16 $0, [$1]; }"
    )
    assert _assembly("", 64, predicated=True, fill=1 << 63) == (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b64 $0, 0x8000000000000000; "
        "@p ld.global.b64 $0, [$1]; }"
    )
    assert _assembly("last", 32, predicated=True, fill=0) == _assembly(
        "last", 32, predicated=True
    )
    assert "mov.b32 $0, 0;" in _assembly("last", 32, predicated=True)
    assert "ld.global.L1::evict_last.b16" in _assembly(
        "last", 16, predicated=True, fill=1
    )


# ---------------------------------------------------------------------------
# Generated kernels
# ---------------------------------------------------------------------------


def _code(kernel: object, args: tuple[object, ...], **config: object) -> str:
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        return bound.to_code(
            helion.Config.from_dict({**spec.default_config().config, **config})
        )


def _window(shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(shape, dtype=torch.bfloat16)
    out = torch.zeros(
        [shape[0], (shape[1] - 1) // 2 + 1, (shape[2] - 1) // 2 + 1],
        dtype=torch.bfloat16,
    )
    return x, out


# One element per thread; the last tiles of the planes and the columns are
# partial.
_ELEMENTS = {"block_sizes": [2, 1, 16], "num_threads": [2, 1, 16]}


def test_padded_window_taps_load_predicated_with_their_padding() -> None:
    """Under the store's guard every tap of the padded window is one
    predicated load that leaves ``-inf``, at an even and at an odd image
    size (where the last column's coordinate can pass the image and only
    the program's mask keeps it inside): no conditional load, no select."""
    for shape in ((5, 12, 20), (5, 13, 13)):
        off = _code(padded_window_max, _window(shape), **_ELEMENTS)
        on = _code(
            padded_window_max,
            _window(shape),
            **_ELEMENTS,
            cute_guarded_lanes=True,
            cute_fmnmx_extrema=True,
            **{_KNOB: True},
        )
        assert _LOAD not in off and off.count(".load() if ") == 9
        assert on.count(f"{_LOAD}(x.iterator + ") == 9
        assert on.count(f", cutlass.BFloat16, {_NEG_INF_BF16})") == 9
        assert ".load()" not in on
        assert "else cutlass.Float32(scalar_tensor" not in on
        assert on.count("cute.arch.fmax(") == 8


def test_tile_masks_beside_the_window_mask_keep_the_select() -> None:
    """Without ``cute_guarded_lanes`` a tap's test holds the tile masks
    beside the window's mask: the tap loads predicated, with zero bits, and
    its select stays.  The bound the window's own mask states is gone from
    the test all the same."""
    args = _window((5, 13, 13))
    off = _code(padded_window_max, args, **_ELEMENTS)
    on = _code(padded_window_max, args, **_ELEMENTS, **{_KNOB: True})
    assert off.count(".load() if ") == 9 and off.count(" < 13") > 9
    assert on.count(f"{_LOAD}(x.iterator + ") == 9
    assert on.count(", cutlass.BFloat16, 0)") == 9
    assert on.count("else cutlass.Float32(scalar_tensor") == 9
    for line in on.splitlines():
        if f"{_LOAD}(" in line:
            test = line.split("x.layout.stride[2]), ")[1]
            assert " < 13" not in test and "mask_0 and mask_2" in test


def test_vector_lanes_of_a_partial_tile_keep_their_selects() -> None:
    """A vector loop stores one packet per thread behind its lanes; where
    its last tile is partial (24 columns in tiles of 16) each lane's tap
    keeps the lane mask beside the window's: predicated, zero bits, the
    select stays."""
    on = _code(
        padded_window_max,
        _window((5, 12, 48)),
        block_sizes=[1, 1, 16],
        num_threads=[1, 1, 2],
        cute_vector_widths=[1, 1, 8],
        cute_guarded_lanes=True,
        **{_KNOB: True},
    )
    assert "_cute_store_u16_vec(" in on
    assert on.count(f"{_LOAD}(x.iterator + ") == 9
    assert on.count(" and mask_2, cutlass.BFloat16, 0)") == 9
    assert on.count("else cutlass.Float32(scalar_tensor") == 9


def test_same_dtype_select_gives_the_fill() -> None:
    """A gather through a loaded index, selected against a constant in the
    tensor's own type: the fill is that type's bits (fp32 2.5, int32 -1, the
    bf16 nearest 0.1, which is what the select stored).  An int64 index
    stays bounded by the mask's own ``>= 0`` and ``< columns``."""
    for dtype, fill, bits in (
        (torch.float32, 2.5, 0x40200000),
        (torch.int32, -1, 0xFFFFFFFF),
        (torch.bfloat16, 0.1, 0x3DCD),
    ):
        for index_dtype in (torch.int32, torch.int64):
            args = (
                torch.zeros(6, 37, dtype=dtype),
                torch.zeros(21, dtype=index_dtype),
                fill,
            )
            on = _code(
                gathered_or_fill,
                args,
                block_sizes=[2, 16],
                num_threads=[2, 16],
                cute_guarded_lanes=True,
                **{_KNOB: True},
            )
            (line,) = [text for text in on.splitlines() if f"{_LOAD}(" in text]
            assert line.rstrip().endswith(f", {bits})"), line
            assert " < 37, " not in line
            assert " if " not in on.split(line)[1].split("\n")[1]


def test_bound_the_mask_does_not_state_is_kept() -> None:
    """The mask states no upper bound for the coordinate (it bounds another
    value, or bounds the coordinate past the dim): the ``index < size``
    term stays in the load's test and the select stays."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def loose(
        x: torch.Tensor, index: torch.Tensor, limit: hl.constexpr
    ) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = (col >= 0) & (col < limit)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def other(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = (col + 1 >= 0) & (col + 1 < columns)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def either(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = (col < columns) | (col < 3)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def unsigned(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = col < columns
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    x = torch.zeros(6, 37)
    config = {
        "block_sizes": [2, 16],
        "num_threads": [2, 16],
        "cute_guarded_lanes": True,
        _KNOB: True,
    }
    cases = [
        # A bound past the dim.
        (loose, (x, torch.zeros(21, dtype=torch.int32), 38), True),
        (loose, (x, torch.zeros(21, dtype=torch.int32), 37), False),
        # A comparison against a float: only integer limits are read.
        (loose, (x, torch.zeros(21, dtype=torch.int32), 36.5), True),
        # A comparison of another value than the coordinate.
        (other, (x, torch.zeros(21, dtype=torch.int32)), True),
        # A disjunction.
        (either, (x, torch.zeros(21, dtype=torch.int32)), True),
        # An int64 coordinate the mask does not keep at or above an int32:
        # the address reads its low half.
        (unsigned, (x, torch.zeros(21, dtype=torch.int64)), True),
        (unsigned, (x, torch.zeros(21, dtype=torch.int32)), False),
    ]
    for kernel, args, kept in cases:
        code = _code(kernel, args, **config)
        (line,) = [text for text in code.splitlines() if f"{_LOAD}(" in text]
        assert (" < 37, cutlass.Float32, 0)" in line) is kept, (kernel.name, line)
        assert ("else cutlass.Float32(scalar_tensor" in code) is kept


def test_closed_bound_counts_as_the_next_one() -> None:
    """``col <= limit`` states ``col < limit + 1`` and ``col > low`` states
    ``col >= low + 1``: the last column as the limit drops the load's own
    bound (for an int64 index too, which ``col > -1`` keeps above an
    int32), the dim itself as the limit admits ``col == columns`` and keeps
    it.  A limit the index's type does not hold states nothing: the
    comparison reads it in that type, as another number."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def closed(
        x: torch.Tensor, index: torch.Tensor, low: hl.constexpr, limit: hl.constexpr
    ) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = (col > low) & (col <= limit)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    x = torch.zeros(6, 37)
    config = {
        "block_sizes": [2, 16],
        "num_threads": [2, 16],
        "cute_guarded_lanes": True,
        _KNOB: True,
    }
    int32_min = -(1 << 31)
    cases = [
        (torch.int32, -1, 36, False),
        (torch.int32, -1, 37, True),
        (torch.int64, -1, 36, False),
        (torch.int64, -1, 37, True),
        # ``col > INT32_MIN - 1`` keeps an int64 index an int32; one less
        # does not.
        (torch.int64, int32_min - 1, 36, False),
        (torch.int64, int32_min - 2, 36, True),
        # An int32 reads ``col <= INT32_MIN - 2`` as ``col <= INT32_MAX - 1``
        # and an int64 ``col > 2**63`` as ``col > INT64_MIN``: almost every
        # index passes, so neither is the bound it looks like.
        (torch.int32, -1, int32_min - 2, True),
        (torch.int64, 1 << 63, 36, True),
    ]
    for index_dtype, low, limit, kept in cases:
        index = torch.zeros(21, dtype=index_dtype)
        code = _code(closed, (x, index, low, limit), **config)
        (line,) = [text for text in code.splitlines() if f"{_LOAD}(" in text]
        assert (" < 37, cutlass.Float32, 0)" in line) is kept, (low, limit, line)
        assert ("else cutlass.Float32(scalar_tensor" in code) is kept


def test_bound_of_a_dim_that_is_not_static_is_kept() -> None:
    """With dynamic shapes the size of the gathered dim is a symbol: a
    literal bound in the mask is not compared with it (which would record a
    guard on the example's size), so the load keeps its own bound and the
    select stays.  The loads are predicated all the same."""

    @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
    def dynamic(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows = x.size(0)
        out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, index.size(0)]):
            col = index[tile_c]
            inside = (col >= 0) & (col < 37)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r, tile_c] = torch.where(inside[None, :], value, 2.5)
        return out

    with mock_cuda_unavailable():
        bound = cpu_bind(
            dynamic, (torch.zeros(6, 37), torch.zeros(21, dtype=torch.int32))
        )
        spec = bound.config_spec
        assert _KNOB in spec._flat_fields()
        guards = len(bound.env.shape_env.guards)
        code = bound.to_code(
            helion.Config.from_dict(
                {
                    **spec.default_config().config,
                    "block_sizes": [2, 16],
                    "num_threads": [2, 16],
                    "cute_guarded_lanes": True,
                    _KNOB: True,
                }
            )
        )
        assert len(bound.env.shape_env.guards) == guards
    (line,) = [text for text in code.splitlines() if f"{_LOAD}(" in text]
    assert line.rstrip().endswith(", cutlass.Float32, 0)")
    assert "cutlass.Int32(col) < " in line
    assert "else cutlass.Float32(scalar_tensor" in code


def _mask_bounded(kernel: object, args: tuple[object, ...]) -> list[frozenset[int]]:
    """``mask_bounded_gather_positions`` for each masked load of ``kernel``
    bound to ``args``, on its device graph."""
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
    found = []
    with bound.env, bound.host_function:
        for info in bound.host_function.device_ir.graphs:
            for node in info.graph.find_nodes(op="call_function", target=hl.load):
                if len(node.args) < 3 or node.args[2] is None:
                    continue
                tensor = node.args[0].meta["val"]
                subscript = [
                    entry.meta["val"] if isinstance(entry, torch.fx.Node) else entry
                    for entry in node.args[1]
                ]
                found.append(
                    mask_bounded_gather_positions(
                        SimpleNamespace(fx_node=node),  # type: ignore[arg-type]
                        tensor,
                        subscript,
                    )
                )
    return found


def test_int64_range_past_an_int32_keeps_the_bound() -> None:
    """The address reads an int64 coordinate as an int32, so the range its
    mask states must be an int32's: ``0 <= col < columns`` over a dim past
    that range (one column expanded to ``2**31 + 8``) keeps the load's
    bound, where a dim of 37 drops it.  An int32 coordinate keeps it there
    too (it does not hold that limit, so its comparison is against another
    number) and drops it at 37.  A coordinate of any other integer type (an
    int16 index) keeps it: only int32 and int64 coordinates are read."""
    wide = torch.zeros(6, 1).expand(6, 2**31 + 8)
    for x, dtype, proven in (
        (wide, torch.int64, frozenset()),
        (torch.zeros(6, 37), torch.int64, frozenset({1})),
        (wide, torch.int32, frozenset()),
        (torch.zeros(6, 37), torch.int32, frozenset({1})),
        (torch.zeros(6, 37), torch.int16, frozenset()),
    ):
        index = torch.zeros(21, dtype=dtype)
        assert _mask_bounded(gathered_or_fill, (x, index, 2.5)) == [proven]


def test_mask_term_counts_only_on_the_coordinate_axis() -> None:
    """The mask's ``col < columns`` states the load's bound only for the
    lane the load reads: the comparison on the coordinate's own axis.
    Helion accepts a mask over another tile of the same extent
    (``ok[:, None]`` for a coordinate along the last axis), whose term is
    about another lane, also when it is one operand of an ``&`` beside a
    term of the right tile."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def aligned(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        n, columns = x.shape
        out = torch.empty([n, n], dtype=x.dtype, device=x.device)
        for tile_a, tile_b in hl.tile([n, n]):
            col = index[tile_b]
            ok = (col >= 0) & (col < columns)
            value = hl.load(x, [tile_a, col], extra_mask=ok[None, :])
            out[tile_a, tile_b] = torch.where(ok[None, :], value, -1.0)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def mixed(x: torch.Tensor, index: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        n, columns = x.shape
        out = torch.empty([n, n], dtype=x.dtype, device=x.device)
        for tile_a, tile_b in hl.tile([n, n]):
            col = index[tile_b]
            ok = (col >= 0) & (col < columns)
            mask = ok[:, None] & (keep[tile_a] > 0)[:, None]
            value = hl.load(x, [tile_a, col], extra_mask=mask)
            out[tile_a, tile_b] = torch.where(mask, value, -1.0)
        return out

    x = torch.zeros(16, 37)
    index = torch.zeros(16, dtype=torch.int32)
    assert _mask_bounded(aligned, (x, index)) == [frozenset({1})]
    assert _mask_bounded(gathered_under_row_mask, (x, index)) == [frozenset()]
    assert _mask_bounded(mixed, (x, index, index)) == [frozenset()]
    # The generated code keeps the gathered column's bound with and without
    # the knob (the lowering exchanges such a mask between threads through
    # a staged image, whose load tests its own index).
    config = {"block_sizes": [8, 8], "num_threads": [8, 8]}
    for knobs in ({}, {"cute_guarded_lanes": True, _KNOB: True}):
        code = _code(gathered_under_row_mask, (x, index), **config, **knobs)
        assert "fragment_tensor_index < 37" in code


def test_coordinate_must_run_over_one_axis_of_the_value() -> None:
    """The subscript's axis is known by its tile.  A coordinate over a tile
    that is two axes of the load's value (``x[col[:, None], col[None,
    :]]``) keeps its bound wherever the mask's term sits: on the last axis,
    or on the first, where it is about the other subscript's lane.  So does
    a coordinate with a dim that is no tile."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def squared(
        x: torch.Tensor, index: torch.Tensor, on_rows: hl.constexpr
    ) -> torch.Tensor:
        n = index.size(0)
        columns = x.size(1)
        out = torch.empty([n], dtype=torch.float32, device=x.device)
        for tile_c in hl.tile(n):
            col = index[tile_c]
            inside = (col >= 0) & (col < columns)
            mask = inside[:, None] if on_rows else inside[None, :]
            value = hl.load(x, [col[:, None], col[None, :]], extra_mask=mask)
            out[tile_c] = value.sum(-1)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def tiled(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows], dtype=torch.float32, device=x.device)
        for tile_r in hl.tile(rows):
            for tile_c in hl.tile(index.size(0)):
                col = index[tile_c]
                inside = (col >= 0) & (col < columns)
                value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
                out[tile_r] += value.sum(-1)
        return out

    x = torch.zeros(37, 37)
    index = torch.zeros(21, dtype=torch.int32)
    assert _mask_bounded(tiled, (x, index)) == [frozenset({1})]
    assert _mask_bounded(squared, (x, index, False)) == [frozenset()]
    assert _mask_bounded(squared, (x, index, True)) == [frozenset()]
    # A dim that is neither a unit dim nor a tile: no kernel of these tests
    # has one (a slice is a tile of its own), so the node is made by hand,
    # beside a value with one axis that is no tile either.
    with mock_cuda_unavailable():
        bound = cpu_bind(tiled, (x, index))
    graph = torch.fx.Graph()
    with bound.env, bound.host_function:
        for shape, lanes in (((3,), None), ((1, 1), {})):
            node = graph.placeholder("coordinate")
            node.meta["val"] = torch.zeros(shape, dtype=torch.int32)
            assert _coordinate_lanes(node, [0, None]) == lanes


def test_masked_off_zero_that_is_added_stays_zero() -> None:
    """A masked gather whose zero is added, not selected: predicated with
    zero bits.  So is a tile-masked element of a partial tile without
    ``cute_guarded_lanes`` (with it, the guard drops that mask and the
    element is a plain load)."""
    args = (torch.zeros(6, 37, dtype=torch.bfloat16),)
    config = {"block_sizes": [2, 16], "num_threads": [2, 16]}
    on = _code(strided_pair_sum, args, **config, **{_KNOB: True})
    assert on.count(f"{_LOAD}(x.iterator + ") == 2
    assert on.count(", cutlass.BFloat16, 0)") == 2
    guarded = _code(
        strided_pair_sum, args, **config, cute_guarded_lanes=True, **{_KNOB: True}
    )
    assert guarded.count(f"{_LOAD}(x.iterator + ") == 1
    assert guarded.count(".load()") == 1


def test_hinted_load_with_a_zero_fill_is_left_to_the_hint_pass() -> None:
    """A hinted masked gather whose zero is added takes no fill, so it is
    the hinted predicated load ``hint_element_loads`` writes, with and
    without the knob, and not a second spelling of it.  The knob still
    drops the bound the gather's mask states, which the masks of the tiles
    do not imply here (``index * 2`` over the whole row passes it), so the
    two configs are two sources, one comparison apart."""
    args = (torch.zeros(6, 37, dtype=torch.bfloat16),)
    config = {
        "block_sizes": [2, 16],
        "num_threads": [2, 16],
        "cute_guarded_lanes": True,
        "cute_narrow_load_hints": True,
        "load_eviction_policies": ["last", "last"],
    }
    bounds = []
    for knob in (False, True):
        code = _code(strided_pair_sum, args, **config, **{_KNOB: knob})
        assert code.count("_cute_element_policy_load_if(x.iterator + ") == 1
        assert _LOAD not in code
        bounds.append(code.count(" < 37"))
    assert bounds[0] == bounds[1] + 1


def test_stencil_over_its_own_tiles_keeps_the_select() -> None:
    """A gather over the tensor's own tiles, stored back in place, keeps
    lane terms beside the mask: the load is predicated with zero bits and
    the select stays."""
    x = torch.zeros(3, 9, 21, dtype=torch.bfloat16)
    on = _code(
        interior_or_fill_in_place,
        (x, x.clone()),
        block_sizes=[2, 2, 16],
        num_threads=[2, 2, 16],
        cute_guarded_lanes=True,
        **{_KNOB: True},
    )
    (line,) = [text for text in on.splitlines() if f"{_LOAD}(x.iterator + " in text]
    assert line.rstrip().endswith(", cutlass.BFloat16, 0)")
    assert "else cutlass.Float32(scalar_tensor)" in on


def test_narrow_hint_rides_on_the_predicated_load() -> None:
    """With ``cute_narrow_load_hints`` the taps' policy is on the predicated
    load itself; the pass that places hints finds no masked load left."""
    on = _code(
        padded_window_max,
        _window((5, 12, 20)),
        **_ELEMENTS,
        cute_guarded_lanes=True,
        cute_narrow_load_hints=True,
        load_eviction_policies=["last"] * 9,
        **{_KNOB: True},
    )
    assert on.count(f", cutlass.BFloat16, {_NEG_INF_BF16}, 'last')") == 9
    assert "_cute_element_policy_load" not in on


# ---------------------------------------------------------------------------
# Search surface, coverage and seeds
# ---------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unmasked_strided(x: torch.Tensor) -> torch.Tensor:
    """Every other column of each row: a scaled gather along the last dim
    with no ``extra_mask``."""
    rows, columns = x.shape
    out = torch.empty([rows, columns // 2], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns // 2]):
        out[tile_r, tile_c] = hl.load(x, [tile_r, tile_c.index * 2])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_and_gathered(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Each element plus a masked gather through a loaded index: a direct
    load of the row beside the gather."""
    rows, columns = x.shape
    out = torch.empty([rows, columns], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns]):
        col = index[tile_c]
        inside = (col >= 0) & (col < columns)
        value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
        out[tile_r, tile_c] = x[tile_r, tile_c] + value
    return out


def test_knob_is_offered_on_masked_gathers_along_the_last_dim_only() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def masked_row_sums(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows], dtype=torch.float32, device=x.device)
        for tile_r in hl.tile(rows):
            col = index
            inside = (col >= 0) & (col < columns)
            value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
            out[tile_r] = value.to(torch.float32).sum(-1)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def shifted(x: torch.Tensor) -> torch.Tensor:
        rows, columns = x.shape
        out = torch.empty([rows, columns], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([rows, columns]):
            col = tile_c.index + 3
            inside = col < columns
            out[tile_r, tile_c] = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def unfold(x: torch.Tensor, window: hl.constexpr) -> torch.Tensor:
        seq, channels = x.shape
        flat = x.view(-1)
        n = flat.size(0)
        out = torch.empty([n, window], dtype=x.dtype, device=x.device)
        for tile_n, tile_k in hl.tile([n, window]):
            i = tile_n.index
            k = tile_k.index
            inside = (i // channels)[:, None] + k[None, :] < seq
            offset = i[:, None] + channels * k[None, :]
            out[tile_n, tile_k] = hl.load(flat, [offset], extra_mask=inside)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def flat_rows(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty([4, 32], dtype=x.dtype, device=x.device)
        for tile_r, tile_c in hl.tile([4, 32]):
            row = 13 * tile_r.index.reshape([tile_r, 1])
            inside = (tile_c.index < 13)[None, :]
            out[tile_r, tile_c] = hl.load(x, [row + tile_c.index], extra_mask=inside)
        return out

    with mock_cuda_unavailable():
        offered = [
            cpu_bind(padded_window_max, _window((5, 12, 20))),
            # A masked flat gather that adds a 1-D tile index to a row of
            # higher rank (helion.inductor's flat subscripts), as offered as
            # before match_flat_gather_lane() broadcast such terms.
            cpu_bind(flat_rows, (torch.zeros(60, dtype=torch.bfloat16),)),
            cpu_bind(
                gathered_or_fill,
                (torch.zeros(6, 37), torch.zeros(21, dtype=torch.int32), 2.5),
            ),
            cpu_bind(strided_pair_sum, (torch.zeros(6, 37, dtype=torch.bfloat16),)),
            # A flattened gather along a tile of 8-byte elements, which no
            # vector lane loads in packets.
            cpu_bind(unfold, (torch.zeros(12, 16, dtype=torch.float64), 3)),
        ]
        declined = [
            # A masked load of a shifted tile: packets, like the tile's own.
            cpu_bind(shifted, (torch.zeros(6, 37, dtype=torch.bfloat16),)),
            # A masked flattened gather that is contiguous along a tile: a
            # vector lane there loads it in predicated packets.
            cpu_bind(unfold, (torch.zeros(12, 16, dtype=torch.bfloat16), 3)),
            # A window without a mask of its own.
            cpu_bind(window_max, (torch.zeros(18, 14, 16, dtype=torch.bfloat16),)),
            # A scaled gather along the last dim with no ``extra_mask``: its
            # elements load unconditionally, so there is nothing to predicate.
            cpu_bind(_unmasked_strided, (torch.zeros(6, 38, dtype=torch.bfloat16),)),
            # A masked load along a tile: a padded packet.
            cpu_bind(padded_sum, (torch.zeros(18, 14, 16, dtype=torch.bfloat16),)),
            # A reduction.
            cpu_bind(
                masked_row_sums,
                (torch.zeros(6, 37), torch.zeros(21, dtype=torch.int32)),
            ),
        ]
        for bound in offered:
            spec = bound.config_spec
            assert _KNOB in spec._flat_fields()
            config = spec.default_config().config
            assert _KNOB not in config or config[_KNOB] is False
            on = helion.Config.from_dict(config | {_KNOB: True})
            spec.normalize(on)
            assert on.config[_KNOB] is True
        for bound in declined:
            spec = bound.config_spec
            assert _KNOB not in spec._flat_fields()
            with pytest.raises(exc.InvalidConfig, match=f"{_KNOB} requires"):
                spec.normalize(spec.default_config().config | {_KNOB: True})
            config = spec.default_config().config | {_KNOB: False}
            spec.normalize(config)
            assert _KNOB not in config


def test_search_space_is_the_same_at_every_shape() -> None:
    """The fingerprint follows the kernel's structure: equal at an even and
    an odd image, a tall and a wide one, one row and many."""
    with mock_cuda_unavailable():
        prints = {
            repr(
                cpu_bind(
                    padded_window_max, _window(shape)
                ).config_spec.structural_fingerprint()
            )
            for shape in ((5, 12, 20), (5, 13, 13), (16, 112, 112), (64, 7, 9))
        }
    assert len(prints) == 1


def _seeds(bound: object) -> list[dict[str, object]]:
    generation = bound.config_spec.create_config_generation()  # type: ignore[attr-defined]
    return [dict(seed.config) for _flat, seed in generation.seed_flat_config_pairs()]


def test_gather_tile_seeds_put_a_row_run_on_thread_x() -> None:
    """The window's seeds: one element per thread, the stride-1 axis on
    thread x with the smallest power of two that covers the row (at most
    64) and with half of it, the other threads on the planes, the three
    knobs on; every seed renders."""
    with mock_cuda_unavailable():
        bound = cpu_bind(padded_window_max, _window((64, 112, 112)))
        seeds = [seed for seed in _seeds(bound) if seed.get(_KNOB) is True]
        assert [(seed["block_sizes"], seed["num_threads"]) for seed in seeds] == [
            ([4, 1, 64], [4, 0, 64]),
            ([8, 1, 32], [8, 0, 32]),
        ]
        for seed in seeds:
            assert seed["loop_orders"] == [[2, 1, 0]]
            assert seed["cute_vector_widths"] == [1, 1, 1]
            assert seed["cute_guarded_lanes"] is True
            assert seed["cute_fmnmx_extrema"] is True
        code = bound.to_code(helion.Config.from_dict(seeds[0]))
        assert "block=(64, 4, 1)" in code
        assert code.count(f", cutlass.BFloat16, {_NEG_INF_BF16})") == 9
        # A short row: 20 outputs fit 32 threads.
        short = cpu_bind(padded_window_max, _window((64, 40, 40)))
        assert [
            seed["block_sizes"] for seed in _seeds(short) if seed.get(_KNOB) is True
        ] == [[8, 1, 32], [16, 1, 16]]


def test_gather_tile_seeds_do_not_follow_the_packet_family() -> None:
    """A thread of a gather tile takes one element, so nothing has to
    divide the row: the seeds are planted at an odd output width (57, 55,
    49 columns) and at a width of four, where no tile of the packet family,
    whose packets must divide the extent, takes packets; each renders with
    the taps predicated.  At the odd widths the packet family flattens the
    tile instead (the kernel reads tile.index, whose flatten choice it
    keeps): those seeds, without the knob, are flattened."""
    with mock_cuda_unavailable():
        for shape, tiles, flattened in (
            ((64, 113, 113), [[4, 1, 64], [8, 1, 32]], True),
            ((64, 112, 110), [[4, 1, 64], [8, 1, 32]], True),
            ((64, 112, 98), [[4, 1, 64], [8, 1, 32]], True),
            ((64, 7, 7), [[64, 1, 4], [64, 1, 2]], False),
        ):
            bound = cpu_bind(padded_window_max, _window(shape))
            packet_seeds = [seed for seed in _seeds(bound) if not seed.get(_KNOB)]
            assert bool(packet_seeds) == flattened, shape
            assert all(seed["flatten_loops"] == [True] for seed in packet_seeds)
            seeds = [seed for seed in _seeds(bound) if seed.get(_KNOB)]
            assert [seed["block_sizes"] for seed in seeds] == tiles, shape
            for seed in seeds:
                assert seed[_KNOB] is True
                assert seed["num_threads"] == [
                    seed["block_sizes"][0],
                    0,
                    seed["block_sizes"][2],
                ]
            code = bound.to_code(helion.Config.from_dict(seeds[0]))
            assert code.count(f", cutlass.BFloat16, {_NEG_INF_BF16})") == 9


def test_gather_tile_seeds_stay_off_packet_loads() -> None:
    """A kernel whose loads along the stride-1 axis take packets gets no
    gather tile: a masked gather through a loaded index beside a direct
    load of the row.  Nor does a kernel of one tile (the seed puts a run of
    the row on thread x and the other threads on another axis), nor a
    kernel that is not offered the knob, also when all its loads gather
    along the stride-1 axis.  One load that runs along that axis twice,
    through the tile itself and across it (``x[tile_b, tile_b]`` over a
    square), is a direct load here, although it takes no packets."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def strided_row(x: torch.Tensor) -> torch.Tensor:
        (columns,) = x.shape
        out = torch.empty([columns], dtype=x.dtype, device=x.device)
        for tile in hl.tile(columns):
            col = tile.index * 2
            out[tile] = hl.load(x, [col], extra_mask=col < columns)
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def square_and_gathered(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        n, columns = z.shape
        out = torch.empty([n, n], dtype=torch.float32, device=x.device)
        for tile_a, tile_b in hl.tile([n, n]):
            col = tile_b.index * 2
            far = hl.load(z, [tile_a, col], extra_mask=(col < columns)[None, :])
            out[tile_a, tile_b] = x[tile_b, tile_b].to(torch.float32) + far.to(
                torch.float32
            )
        return out

    with mock_cuda_unavailable():
        packets = cpu_bind(
            _direct_and_gathered,
            (torch.zeros(64, 256), torch.zeros(256, dtype=torch.int32)),
        )
        assert _KNOB in packets.config_spec._flat_fields()
        assert not [seed for seed in _seeds(packets) if seed.get(_KNOB) is True]
        square = cpu_bind(
            square_and_gathered,
            (
                torch.zeros(32, 32, dtype=torch.bfloat16),
                torch.zeros(32, 37, dtype=torch.bfloat16),
            ),
        )
        assert _KNOB in square.config_spec._flat_fields()
        assert not [seed for seed in _seeds(square) if seed.get(_KNOB) is True]
        row = cpu_bind(strided_row, (torch.zeros(4096, dtype=torch.bfloat16),))
        assert _KNOB in row.config_spec._flat_fields()
        assert not [seed for seed in _seeds(row) if seed.get(_KNOB) is True]
        for kernel, shape in (
            (window_max, (18, 14, 16)),
            (_unmasked_strided, (64, 256)),
        ):
            plain = cpu_bind(kernel, (torch.zeros(shape, dtype=torch.bfloat16),))
            assert _KNOB not in plain.config_spec._flat_fields()
            # The seeds as the heuristic plants them: normalization would
            # drop a knob that is not offered and keep the seed's tiles.
            planted = plain.config_spec.compiler_seed_configs
            assert planted and not [seed for seed in planted if _KNOB in seed.config]


def _knob_group(bound: object) -> object:
    (group,) = [
        group
        for group in bound.config_spec.compiler_coverage_groups  # type: ignore[attr-defined]
        if group.key == _KNOB
    ]
    assert group.domain == (False, True) and group.legacy is False
    return group


def test_gather_tile_seeds_cover_the_knob_without_witnesses() -> None:
    """Where the gather tile seeds carry the knob, the group owns it and
    takes no witness: the kernel's other seeds walk whole rows per thread,
    where a witness would time the knob at tiles that cannot win.  At an
    odd width too, where the gather tiles are the only seeds."""
    with mock_cuda_unavailable():
        for shape in ((64, 112, 112), (64, 113, 113)):
            bound = cpu_bind(padded_window_max, _window(shape))
            assert [seed for seed in _seeds(bound) if seed.get(_KNOB) is True]
            assert _knob_group(bound).witnesses == ()  # type: ignore[attr-defined]


def test_witnesses_ride_on_seeds_of_a_kernel_without_gather_tiles() -> None:
    """A kernel offered the knob that gets no gather tile (a direct load of
    the row beside its masked gather): witnesses on its first seeds, each of
    which changes its carrier's code (every seed of an offered kernel has
    masked element loads)."""

    with mock_cuda_unavailable():
        bound = cpu_bind(
            _direct_and_gathered,
            (torch.zeros(64, 256), torch.zeros(256, dtype=torch.int32)),
        )
        assert not [seed for seed in _seeds(bound) if seed.get(_KNOB) is True]
        group = _knob_group(bound)
        assert 1 <= len(group.witnesses) <= 4  # type: ignore[attr-defined]
        for witness in group.witnesses:  # type: ignore[attr-defined]
            carrier = dict(witness.carrier.config)
            assert witness.value is True and carrier.get(_KNOB) is not True
            off = bound.to_code(helion.Config.from_dict(carrier))
            on = bound.to_code(helion.Config.from_dict(carrier | {_KNOB: True}))
            assert _LOAD not in off and _LOAD in on
