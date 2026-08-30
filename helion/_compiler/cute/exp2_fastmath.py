# pyrefly: ignore-errors
"""Apply ``fastmath=True`` to every ``cute.math.exp2`` call in the kernel
body when the ``fast_math`` SETTING is enabled.

This is deliberately a setting (``helion.Settings.fast_math`` /
``HELION_FAST_MATH=1``) and not an autotuner config knob: it changes
numerics, and tuned configs must never change numerics.

The default exp2 lowering guards the denormal output range: each call
costs an extra FSETP plus two predicated FMULs per element and lengthens
live ranges (the input must survive until the fixup).  ``ex2.approx.ftz``
skips the fixup; exp results below the normal f32 range flush to zero.
For exp-of-shifted-value reductions (softmax and friends) those elements
contribute nothing to the sum and round to zero in a 16-bit output anyway
— measured +1.7% on register-resident softmax rows on B200.
"""

from __future__ import annotations

import ast


def apply_exp2_fastmath(body: list[ast.stmt]) -> list[ast.stmt]:
    for top in body:
        for node in ast.walk(top):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exp2"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "math"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "cute"
            ):
                continue
            if any(kw.arg == "fastmath" for kw in node.keywords):
                continue
            node.keywords.append(
                ast.keyword(arg="fastmath", value=ast.Constant(value=True))
            )
            ast.fix_missing_locations(node)
    return body
