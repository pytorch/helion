from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


def transform(code: str, *, enabled: bool = True, disjoint: bool = False) -> str:
    body = ast.parse(code).body
    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={3: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))} if disjoint else set(),
        preserve_scalar_invariants=enabled,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


def sweeps(
    expr: str = "(x.iterator + head).load()",
    *,
    pre: str = "",
    inner: str = "",
    between: str = "",
    tail: str = "",
) -> str:
    return f"""
{pre}
for synthetic_lane_3 in cutlass.range_constexpr(4):
    {inner or "pass"}
    first = {expr}
    a = cute.math.exp2(cutlass.Float32(first))
{between}
for synthetic_lane_3 in cutlass.range_constexpr(4):
    {inner or "pass"}
    second = {expr}
    b = cute.math.exp2(cutlass.Float32(second))
{tail}
"""


def test_repeated_readonly_scalar_load_keeps_original_positions() -> None:
    code = sweeps()
    assert "_fuse_cache_" in transform(code, enabled=False)
    assert transform(code) == ast.unparse(ast.parse(code))


@pytest.mark.parametrize(
    "expr",
    [
        "(x.iterator + head).load() if active else cutlass.Float32(0)",
        "(x.iterator + index).load()",
    ],
)
def test_complete_invariant_mask_and_reaching_address(expr: str) -> None:
    code = sweeps(expr, pre="index = head + 1")
    assert "_fuse_cache_" in transform(code, enabled=False)
    assert transform(code) == ast.unparse(ast.parse(code))


@pytest.mark.parametrize(
    "code",
    [
        sweeps("(x.iterator + synthetic_lane_3).load()"),
        sweeps(
            "(x.iterator + head).load() if synthetic_lane_3 < 2 else cutlass.Float32(0)"
        ),
        sweeps("(x.iterator + index).load()", inner="index = synthetic_lane_3 + 1"),
        sweeps(
            "(x.iterator + head).load() if active else cutlass.Float32(0)",
            inner="active = synthetic_lane_3 < 2",
        ),
        sweeps(
            "(x.iterator + index).load()", pre="index = head", inner="index = index + 1"
        ),
        sweeps(
            "(x.iterator + index).load()", pre="index = head", between="index = other"
        ),
        sweeps(
            "(x.iterator + head).load() if active else cutlass.Float32(0)",
            between="active = other",
        ),
        sweeps(tail="(x.iterator + head).store(replacement)"),
        sweeps(tail="(out.iterator + head).store(replacement)"),
        sweeps(tail="alias = x.iterator\n(alias + head).store(replacement)"),
        sweeps(between="unknown(x)"),
        sweeps(tail="x.layout = replacement"),
        sweeps(tail="alias = x\nalias[0] = replacement"),
        sweeps("(x.iterator + index).load()", pre="for j in range(2):\n    index = j"),
        sweeps(
            "(x.iterator + head).load() if active else cutlass.Float32(synthetic_lane_3)"
        ),
    ],
)
def test_uncertain_or_mutable_cases_retain_original_fuser(code: str) -> None:
    assert transform(code) == transform(code, enabled=False)


def test_whole_kernel_write_requires_actual_disjointness() -> None:
    code = sweeps(tail="(out.iterator + head).store(replacement)")
    assert "_fuse_cache_" in transform(code)
    assert transform(code, disjoint=True) == ast.unparse(ast.parse(code))


def test_scalar_bypass_does_not_remove_varying_cache() -> None:
    code = """
for synthetic_lane_3 in cutlass.range_constexpr(4):
    scalar = (x.iterator + head).load()
    element = (x.iterator + synthetic_lane_3).load()
for synthetic_lane_3 in cutlass.range_constexpr(4):
    scalar2 = (x.iterator + head).load()
    element2 = (x.iterator + synthetic_lane_3).load()
"""
    text = transform(code)
    assert text.count("(x.iterator + head).load()") == 2
    assert text.count("(x.iterator + synthetic_lane_3).load()") == 1
    assert "element2 = _fuse_cache_" in text


def test_existing_scalar_licm_can_hoist_complete_gate_chain() -> None:
    from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
    from helion._compiler.cute.hoist_lane_invariant_reductions import (
        hoist_lane_invariant_reductions,
    )

    code = """
for lane_0 in cutlass.range_constexpr(4):
    for synthetic_lane_3 in cutlass.range_constexpr(4):
        load = (x.iterator + head).load()
        gate = cute.math.exp2(cutlass.Float32(load))
        use = gate * synthetic_lane_3 + lane_0
    for synthetic_lane_3 in cutlass.range_constexpr(4):
        load = (x.iterator + head).load()
        gate = cute.math.exp2(cutlass.Float32(load))
        use2 = gate + synthetic_lane_3 + lane_0
"""
    body = ast.parse(transform(code)).body
    setattr(body[0], HELION_LANE_LOOP_VAR_ATTR, "lane_0")
    result = hoist_lane_invariant_reductions(
        body,
        tensor_names={"x"},
        tensor_dtypes={"x": "cutlass.Float32"},
        proven_disjoint_tensor_pairs=set(),
        rename_groups={},
    )
    text = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert text.count("cute.math.exp2") == 1
    assert text.index("gate =") < text.index("for lane_0")
