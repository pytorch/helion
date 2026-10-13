from __future__ import annotations

import ast

import pytest

from .test_cute_computed_sweep_cache import _body
from .test_cute_computed_sweep_cache import _declarations
from .test_cute_computed_sweep_cache import _run


@pytest.mark.parametrize("name", ["exp2", "log"])
@pytest.mark.parametrize("policy", ["", ", fastmath=False", ", fastmath=True"])
def test_identical_explicit_math_policy_caches_original_result(
    name: str, policy: str
) -> None:
    expression = f"cute.math.{name}(cutlass.Float32(raw){policy})"
    source = _run(_body(first=expression))
    assert len(_declarations(source)) == 1
    assert source.count(expression) == 1
    loops = [node for node in ast.parse(source).body if isinstance(node, ast.For)]
    assert ast.unparse(loops[0].body[1]) == f"current = {expression}"
    assert ast.unparse(loops[0].body[-1]) == "total = total + current"
    assert "current = _computed_cache_" in ast.unparse(loops[1])
    assert (
        ast.unparse(loops[1].body[-1])
        == "(out.iterator + synthetic_lane_7).store(current + total)"
    )


@pytest.mark.parametrize("name", ["exp2", "log"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("", ", fastmath=False"),
        (", fastmath=False", ""),
        ("", ", fastmath=True"),
        (", fastmath=True", ""),
        (", fastmath=False", ", fastmath=True"),
        (", fastmath=True", ", fastmath=False"),
    ],
)
def test_math_policy_is_part_of_exact_value_identity(
    name: str, first: str, second: str
) -> None:
    source = _run(
        _body(
            first=f"cute.math.{name}(cutlass.Float32(raw){first})",
            second=f"cute.math.{name}(cutlass.Float32(raw){second})",
        )
    )
    assert not _declarations(source)


@pytest.mark.parametrize("name", ["exp2", "log"])
@pytest.mark.parametrize(
    "policy",
    [
        ", fastmath=0",
        ", fastmath=1",
        ", fastmath=None",
        ", fastmath='True'",
        ", fastmath=flag",
        ", fastmath=True, fastmath=True",
        ", fastmath=True, unknown=False",
        ", unknown=True",
        ", **options",
        ", True",
    ],
)
def test_unknown_or_nonliteral_math_policy_declines(name: str, policy: str) -> None:
    assert not _declarations(
        _run(_body(first=f"cute.math.{name}(cutlass.Float32(raw){policy})"))
    )
