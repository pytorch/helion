from __future__ import annotations

import math
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import sympy
from torch.utils._sympy.functions import FloatTrueDiv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import ToFloat

from helion._compiler.cute.printer import HelionCutePrinter
from helion._compiler.device_function import DeviceFunction
from helion._compiler.triton.printer import HelionTritonPrinter
from helion._testing import TestCase
from helion._testing import onlyBackends

pytestmark = pytest.mark.cpu_only


@onlyBackends(["triton"])
class TestHelionTritonPrinter(TestCase):
    """Tests for the HelionTritonPrinter class."""

    def test_print_ToFloat(self):
        """Test that ToFloat expressions are printed correctly."""
        printer = HelionTritonPrinter()

        # A symbolic integer becomes a floating expression.
        x = sympy.Symbol("x", integer=True)
        value = eval(printer.doprint(ToFloat(x)), {"x": 7})
        self.assertEqual(value, 7.0)
        self.assertIs(type(value), float)

        # Complex expression
        y = sympy.Symbol("y", integer=True)
        result = printer.doprint(ToFloat(x + y))
        self.assertNotIn("ToFloat", result)
        self.assertIn("0.0", result)

        # Concrete integer: ToFloat(5) simplifies to 5.0
        result = printer.doprint(ToFloat(sympy.Integer(5)))
        self.assertNotIn("ToFloat", result)
        self.assertEqual(float(result), 5.0)

    def test_print_Float(self):
        """Test that Float expressions are printed as raw literals."""
        printer = HelionTritonPrinter()

        # Non-symbolic floats should print as raw numeric literals (not tl.full)
        for val in [math.pi, 0.0, -2.5]:
            result = printer.doprint(sympy.Float(val))
            self.assertNotIn("tl.full", result)
            self.assertAlmostEqual(float(result), val)

    def test_print_FloorDiv_constexpr(self):
        """Test that FloorDiv with constexpr LHS prints as // operator.

        This is required for TMA tensor descriptors which need compile-time
        constant block shapes. When the LHS is a constexpr argument (like a
        block size), we must emit `lhs // rhs` instead of
        `triton_helpers.div_floor_integer(lhs, rhs)` so Triton can evaluate
        the expression at compile time.
        """
        printer = HelionTritonPrinter()

        # LHS is constexpr -> use //
        mock_df = Mock(_constexpr_args={"_BLOCK_SIZE_0": None})
        with patch.object(DeviceFunction, "current", return_value=mock_df):
            block_size = sympy.Symbol("_BLOCK_SIZE_0", integer=True)
            expr = FloorDiv(block_size, 2)
            result = printer.doprint(expr)
            self.assertEqual(result, "_BLOCK_SIZE_0 // 2")

        # LHS is NOT constexpr -> fallback to triton_helpers
        mock_df = Mock(_constexpr_args={})
        with patch.object(DeviceFunction, "current", return_value=mock_df):
            x = sympy.Symbol("x", integer=True)
            expr = FloorDiv(x, 2)
            result = printer.doprint(expr)
            self.assertIn("div_floor_integer", result)


@onlyBackends(["triton"])
@pytest.mark.parametrize("printer_type", [HelionTritonPrinter, HelionCutePrinter])
def test_float_conversion_remains_grouped_in_parent_arithmetic(printer_type):
    x, y = sympy.symbols("x y", integer=True)
    expressions = (
        FloatTrueDiv(ToFloat(x), ToFloat(y)),
        ToFloat(x + y) / 2,
        3 - ToFloat(x + y),
        ToFloat(x + y) ** 2,
    )
    # Both scalar printers share ToFloat; division, subtraction and power put
    # the conversion in different precedence contexts. Test emitted semantics.
    for expression in expressions:
        code = printer_type().doprint(expression)
        actual = eval(code, {"x": 7, "y": 2})
        expected = float(expression.subs({x: 7, y: 2}))
        assert actual == expected, (expression, code, actual, expected)
