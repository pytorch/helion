from __future__ import annotations

import pytest

from helion._compiler.inline_asm_constraints import input_constraint_codes
from helion._compiler.inline_asm_constraints import output_constraint_codes

pytestmark = pytest.mark.cpu_only


def test_output_constraint_codes() -> None:
    assert output_constraint_codes("=h,h,h") == ["h"]
    assert output_constraint_codes("=&r, =v ,v,0,~{memory}") == ["r", "v"]
    assert output_constraint_codes("v,v") == []


def test_input_constraint_codes() -> None:
    """Inputs follow the outputs in operand order; a matching constraint
    takes its output's register class; clobbers bind nothing; GCC's
    read-write ``+`` is not LLVM syntax and is rejected."""
    assert input_constraint_codes("=h,h,h") == ["h", "h"]
    assert input_constraint_codes("=&r,r,0") == ["r", "r"]
    assert input_constraint_codes("=h,0,h") == ["h", "h"]
    assert input_constraint_codes("=r,=h,h,r,~{memory}") == ["h", "r"]
    assert input_constraint_codes("=f, f, f") == ["f", "f"]
    for constraints in ("+r,r", "=r,+h,h"):
        with pytest.raises(ValueError, match="not an LLVM inline asm"):
            input_constraint_codes(constraints)


def test_input_constraint_codes_allow_plus() -> None:
    """``allow_plus`` reads an input's ``+`` as LLVM does: a plain input."""
    assert input_constraint_codes("=v,+v", allow_plus=True) == ["v"]
    assert input_constraint_codes("=v,+h,0", allow_plus=True) == ["h", "v"]
