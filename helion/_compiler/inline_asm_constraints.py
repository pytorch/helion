"""Backend-neutral parsing of LLVM inline asm constraint strings."""

from __future__ import annotations


def output_constraint_codes(constraints: str) -> list[str]:
    """The constraint code of every asm output operand, in operand order."""
    items = [item.strip() for item in constraints.split(",") if item.strip()]
    return [item.lstrip("=&") for item in items if item.startswith("=")]


def input_constraint_codes(constraints: str, *, allow_plus: bool = False) -> list[str]:
    """The constraint code of every asm input operand, in operand order.

    An LLVM constraint string lists the outputs (``=`` prefixed, ``&`` for an
    early clobber) before the inputs, which bind to the call's operands in
    order; a matching constraint (``0``) ties an input to that output's
    register class; ``~{...}`` clobbers bind nothing.  GCC's read-write ``+``
    prefix is not LLVM IR syntax: LLVM parses ``+r`` as a plain input and
    Clang spells a read-write operand as ``=r`` plus a trailing matching
    input, so a ``+`` is rejected instead of mis-bound.  ``allow_plus`` reads
    an input's ``+`` as LLVM does, as a plain input of that class.
    """
    items = [item.strip() for item in constraints.split(",") if item.strip()]
    if allow_plus:
        items = [item.lstrip("+") for item in items]
    elif any(item.startswith("+") for item in items):
        raise ValueError(
            f"'+' is not an LLVM inline asm constraint prefix: {constraints!r}; "
            "spell a read-write operand as an '=' output plus a matching input"
        )
    outputs = output_constraint_codes(constraints)
    codes = []
    for item in items[len(outputs) :]:
        if item.startswith("~"):
            continue
        if item.isdigit() and int(item) < len(outputs):
            item = outputs[int(item)]
        codes.append(item)
    return codes
