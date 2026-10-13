from __future__ import annotations

import ast
from collections import Counter
import re
from textwrap import indent
from unittest.mock import patch

import pytest

from helion._compiler.cute import fuse_two_pass_loads as fuser


def _sweeps(middle: str = "", address: str = "base + synthetic_lane_7") -> str:
    return f"""
for synthetic_lane_7 in range(8):
    address = {address}
    first = (x.iterator + address).load() if predicate else fallback
    dead = first * scale + dead_input
{middle}
for synthetic_lane_7 in range(8):
    address = {address}
    second = (x.iterator + address).load() if predicate else fallback
    dead = second * scale + dead_input
    (out.iterator + synthetic_lane_7).store(second)
"""


def _run(
    source: str, *, eager: bool, mode: str = "register", disjoint: bool = True
) -> tuple[str, list[str], list[bool], Counter[str]]:
    owner = fuser._CuteFuseTwoPassLoads(
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: mode},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))} if disjoint else set(),
        preserve_scalar_invariants=True,
    )
    keys: list[str] = []
    admissions: list[bool] = []
    freezes: Counter[str] = Counter()
    original_key = fuser._canonical_load_text
    original_freeze = owner._freeze_binding
    original_admission = owner._readonly_scalar_is_invariant
    original_demand = fuser._binding_demand

    def identity_text(text: str) -> str:
        # Allocation ordinals are internal. Compare the actual lexical token
        # identity, including the recursively bound induction-domain identity.
        identities: dict[str, str] = {}
        for (statement, name), token in owner._binding_names.items():
            name = re.sub(r"\b_fuser_binding_\d+_*\b", lambda m: identities[m[0]], name)
            identities[token] = repr((statement.lineno, statement.col_offset, name))
        return re.sub(r"\b_fuser_binding_\d+_*\b", lambda m: identities[m[0]], text)

    def key(
        node: ast.AST,
        aliases: dict[str, str],
        definitions: dict[str, ast.expr] | None = None,
    ) -> str:
        result = original_key(node, aliases, definitions)
        keys.append(identity_text(result))
        return result

    def freeze(
        stmt: ast.stmt, value: ast.expr, previous: dict[str, ast.expr]
    ) -> ast.expr:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            freezes[stmt.targets[0].id] += 1
        return original_freeze(stmt, value, previous)

    def admission(
        stmt: ast.Assign, kind: str, definitions: dict[str, ast.expr]
    ) -> bool:
        value = original_admission(stmt, kind, definitions)
        admissions.append(value)
        return value

    with (
        patch.object(fuser, "_canonical_load_text", key),
        patch.object(owner, "_freeze_binding", freeze),
        patch.object(owner, "_readonly_scalar_is_invariant", admission),
        patch.object(
            fuser,
            "_binding_demand",
            side_effect=lambda body, types: (
                None if eager else original_demand(body, types)
            ),
        ),
    ):
        body = owner.transform(ast.parse(source).body)
    return (
        ast.unparse(ast.Module(body=body, type_ignores=[])),
        keys,
        admissions,
        freezes,
    )


def _compare(
    source: str, *, mode: str = "register", disjoint: bool = True
) -> tuple[str, Counter[str], Counter[str]]:
    old = _run(source, eager=True, mode=mode, disjoint=disjoint)
    new = _run(source, eager=False, mode=mode, disjoint=disjoint)
    assert old[:3] == new[:3]
    return new[0], old[3], new[3]


@pytest.mark.parametrize(
    "middle",
    [
        "",
        "base = different",
        "base = base + 1",
        "base += 1",
        "if active:\n    base = different",
        "for base in range(2):\n    pass",
        "(x.iterator + base).store(0)",
        "alias = x\n(alias.iterator + base).store(0)",
        "unknown(x)",
        "predicate = other_predicate",
        "fallback = other_fallback",
    ],
)
def test_rebindings_masks_mutations_and_opaque_identities(middle: str) -> None:
    source = "base = original\npredicate = True\nfallback = 0\n" + _sweeps(middle)
    _compare(source)


@pytest.mark.parametrize("mode", ["auto", "gmem", "register", "smem", "computed"])
@pytest.mark.parametrize("disjoint", [False, True])
def test_existing_admission_and_fallback_modes(mode: str, disjoint: bool) -> None:
    _compare(
        "base = original\npredicate = True\nfallback = 0\n" + _sweeps(),
        mode=mode,
        disjoint=disjoint,
    )


@pytest.mark.parametrize("kind", ["nested", "branch", "capture", "cycle"])
def test_definition_time_binding_and_induction_domains(kind: str) -> None:
    prefix = "base = outer\npredicate = True\nfallback = 0\n"
    body = prefix + _sweeps()
    if kind == "nested":
        body = "for inner in range(2):\n" + indent(
            prefix.replace("base = outer", "base = outer + inner") + _sweeps(), "    "
        )
    elif kind == "branch":
        body = (
            prefix
            + "if active:\n"
            + indent(_sweeps(), "    ")
            + "else:\n"
            + indent("base = other\n" + _sweeps(), "    ")
        )
    elif kind == "capture":
        body = (
            prefix
            + "saved = base\nbase = other\n"
            + _sweeps(address="saved + synthetic_lane_7")
        )
    elif kind == "cycle":
        body = prefix + "saved = base\nbase = saved\nsaved = base\n" + _sweeps()
    _compare("for outer in range(3):\n" + indent(body, "    "))


@pytest.mark.parametrize("terms", [60, 125, 126, 127, 128, 129, 170])
def test_original_expanded_node_budget(terms: int) -> None:
    expression = " + ".join(["small"] * terms)
    source = (
        "small = original + 1\n"
        f"unused = {expression}\n"
        f"base = {expression}\n"
        "predicate = base < limit\nfallback = 0\n" + _sweeps()
    )
    _compare(source)


def test_all_versions_feed_demand_but_only_original_freezer_proves_values() -> None:
    body = ast.parse(
        "left = first\nright = second\naddress = left\n"
        "if active:\n    address = right\n"
        "first = second\nsecond = first\n"
        "value = (x.iterator + address).load()"
    ).body
    needed = fuser._binding_demand(body, {"x": "cutlass.Float32"})
    assert {"address", "left", "right", "first", "second", "x"} <= needed


def test_irrelevant_computation_is_not_frozen() -> None:
    dead = "\n".join(
        f"temporary{i} = (temporary{i - 1} + scale) * multiplier"
        if i
        else "temporary0 = seed"
        for i in range(80)
    )
    source = dead + "\nbase = original\npredicate = True\nfallback = 0\n" + _sweeps()
    _, old, new = _compare(source)
    assert sum(old[name] for name in old if name.startswith("temporary")) >= 80
    assert not any(new[name] for name in old if name.startswith("temporary"))
    assert old["base"] == new["base"] > 0


@pytest.mark.parametrize("mode", ["register", "smem"])
def test_vector_rewrites_preserve_later_memory_derived_keys(mode: str) -> None:
    source = """
base = input_offset
for synthetic_lane_7 in range(8):
    packet0 = cute.arch.load(x.iterator + base + synthetic_lane_7 * 4, ir.VectorType.get([4], cutlass.Float32.mlir_type))
    first = packet0[0]
    value = (x.iterator + first).load()
for synthetic_lane_7 in range(8):
    packet1 = cute.arch.load(x.iterator + base + synthetic_lane_7 * 4, ir.VectorType.get([4], cutlass.Float32.mlir_type))
    second = packet1[0]
    value = (x.iterator + second).load()
for synthetic_lane_7 in range(8):
    packet2 = cute.arch.load(x.iterator + base + synthetic_lane_7 * 4, ir.VectorType.get([4], cutlass.Float32.mlir_type))
    third = packet2[0]
    value = (x.iterator + third).load()
"""
    output, _, _ = _compare(source, mode=mode)
    assert "packet1 = cute.arch.load" not in output


@pytest.mark.parametrize(
    "middle", ["", "base = next_offset", "(x.iterator + 0).store(1)"]
)
def test_original_readonly_load_expansion_remains_demanded(middle: str) -> None:
    source = (
        "base = (x.iterator + 0).load()\npredicate = True\nfallback = 0\n"
        + _sweeps(middle)
    )
    _, old, new = _compare(source)
    assert old["base"] == new["base"]
    assert (old["base"] == 0) == (".store(" in middle)
