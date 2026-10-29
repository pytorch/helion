"""Render a spec tree as Python source, using torch-spyre's own printer.

Plan 5.1: ``SpyreKernel.codegen_kernel`` already prints exactly such a literal
via ``_codegen_op_spec_list`` (torch_spyre/_inductor/spyre_kernel.py:1646), and
``tests/op_specs/capture.py`` reuses the same printer. "Reuse that printer; do
not write a second one." -- so this module is only the ``sympy_str`` closure and
an ``IndentedBuffer``, mirroring capture.py:291-303.

Reusing it is also what makes the G2 golden diff meaningful: emitter output and
captured reference go through one printer, so a diff is a real difference in the
specs rather than in two formatters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def spec_source(specs: Sequence[object]) -> str:
    """Python source for a spec list, as torch-spyre itself would print it."""
    from torch._inductor.utils import IndentedBuffer
    from torch_spyre._inductor.op_spec import IndirectAccess
    from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list

    def sympy_str(expr: object) -> str:
        # Mirrors the local closure in SpyreKernel.codegen_kernel, which cannot
        # be imported; capture.py:291-296 keeps the same copy.
        if isinstance(expr, IndirectAccess):
            return f"IndirectAccess('{expr.args[0]}')"
        return "sympify('" + str(expr) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(list(specs), buf, sympy_str)
    buf.writeline("]")
    return buf.getvalue()
