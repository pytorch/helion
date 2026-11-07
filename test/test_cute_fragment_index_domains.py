from __future__ import annotations

import ast
import itertools
import re

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import fragment_domain
from helion._compiler.cute.fragment_context import Fragment
from helion._compiler.cute.fragment_context import FragmentContext
from helion._compiler.cute.fragment_domain import OwnershipTerm
from helion._compiler.cute.fragment_domain import ownership_terms
from helion._compiler.cute.fragment_domain import project_contracted_ownership
from helion._compiler.cute.fragment_domain import render_ownership
from helion._compiler.cute.fragment_dot import _dot_output_domain
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_fragment_index_domains import dot_then_reduce
from helion._testing.cute_fragment_index_domains import indexed_gemm
from helion._testing.cute_fragment_index_domains import reshaped_dot_then_max

pytestmark = pytest.mark.cpu_only


def test_indexed_storage_retains_the_global_tile_bound() -> None:
    bound = cpu_bind(
        indexed_gemm,
        (
            torch.empty((2 * 17 + 32) * 16, dtype=torch.bfloat16),
            torch.empty((16, 16), dtype=torch.bfloat16),
            torch.empty((2 * 17 + 32) * 16, dtype=torch.bfloat16),
            17,
        ),
    )
    source = bound.to_code(helion.Config(block_sizes=[32]))
    assert "fragment_tensor_index" in source
    # The physical flat capacity is larger than the logical row extent.
    # This exact global bound must survive arithmetic/broadcast index recipes.
    assert re.search(r"tile_offset_1 \+ [^\n]+ < 17", source)


@pytest.mark.parametrize("contraction", [0, 1, 8])
def test_dot_domain_projects_only_free_coordinate_conjuncts(contraction: int) -> None:
    lhs = Fragment(
        (1, 8, contraction),
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (
            f"({c[1]} < 7) and ({c[2]} < 0)",
            # A nested singleton broadcast can already have folded a
            # contracted empty-domain condition to a coordinate-free false.
            "0 < 0" if contraction == 1 else "True",
        ),
    )
    rhs = Fragment(
        (4, contraction, 16),
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (f"{c[0]} < 3 and {c[1]} >= 0 and {c[2]} < 9",),
    )
    acc = Fragment(
        (8, 16),
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (f"{c[0]} < 6 and {c[1]} < 5",),
    )
    domain = _dot_output_domain(lhs, rhs, acc, ("batch", "row", "col"), "k")
    assert domain is not None
    # Empty/singleton K and a broadcast batch do not veto the output. The
    # accumulator has stricter free-axis ownership than either padded input.
    for batch in range(4):
        for row in range(8):
            for col in range(16):
                masks = domain((str(batch), str(row), str(col)))
                assert all(
                    eval(render_ownership(mask), {"__builtins__": {}}, {})
                    for mask in masks
                ) == (batch < 3 and row < 6 and col < 5)


@pytest.mark.parametrize("empty_operand", ["lhs", "rhs"])
def test_dot_domain_preserves_empty_singleton_batch(empty_operand: str) -> None:
    lhs = Fragment(
        (1, 8, 8),
        torch.float32,
        lambda _: "0",
        logical_domain=(lambda c: (f"{c[0]} < 0",)) if empty_operand == "lhs" else None,
    )
    rhs = Fragment(
        (1, 8, 16),
        torch.float32,
        lambda _: "0",
        logical_domain=(lambda c: (f"{c[0]} < 0",)) if empty_operand == "rhs" else None,
    )
    domain = _dot_output_domain(lhs, rhs, None, ("batch", "row", "col"), "k")
    assert domain is not None
    # Both operand roles retain the batch-axis provenance before their
    # physical singleton is broadcast. K is nonempty and cannot justify it.
    for row in range(8):
        for col in range(16):
            assert not all(
                eval(render_ownership(mask), {"__builtins__": {}}, {})
                for mask in domain(("0", str(row), str(col)))
            )


@pytest.mark.parametrize("role", ["lhs", "rhs"])
@pytest.mark.parametrize("empty_axis", ["batch", "k"])
@pytest.mark.parametrize("chained", [False, True])
def test_broadcast_and_dot_keep_empty_axis_provenance(
    role, empty_axis, chained
) -> None:
    # Eight CPU cases distinguish each operand role, free versus contracted
    # singleton emptiness, and a second dot that consumes the first domain.
    lhs = Fragment((1, 8, 1), torch.float32, lambda _: "0")
    rhs = Fragment((1, 1, 8), torch.float32, lambda _: "0")
    source = lhs if role == "lhs" else rhs
    axis = 0 if empty_axis == "batch" else (2 if role == "lhs" else 1)
    source = Fragment(
        source.shape,
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (f"{c[axis]} < 0",),
    )
    # This is the production expand/clone domain recipe, composed twice.
    expanded = Fragment(
        source.shape,
        source.dtype,
        source.element,
        logical_domain=source.broadcast_domain,
    )
    cloned = Fragment(
        source.shape,
        source.dtype,
        source.element,
        logical_domain=expanded.broadcast_domain,
    )
    if role == "lhs":
        lhs = cloned
    else:
        rhs = cloned
    domain = _dot_output_domain(lhs, rhs, None, ("batch", "row", "col"), "k")
    assert domain is not None
    if chained:
        result = Fragment(
            (1, 8, 8), torch.float32, lambda _: "0", logical_domain=domain
        )
        other = Fragment((1, 8, 8), torch.float32, lambda _: "0")
        domain = _dot_output_domain(
            result if role == "lhs" else other,
            other if role == "lhs" else result,
            None,
            ("batch", "row", "col"),
            "k",
        )
        assert domain is not None
    for row in range(8):
        for col in range(8):
            masks = domain(("0", str(row), str(col)))
            assert all(
                eval(render_ownership(mask), {"__builtins__": {}}, {}) for mask in masks
            ) == (empty_axis == "k")
            assert all(
                "__helion_ownership_axis_" not in render_ownership(mask)
                for mask in masks
            )
            for mask in masks:
                for name in mask.coordinates:
                    assert isinstance(name, str)
                    assert not name.startswith("__helion_ownership_axis_")


def test_equal_predicates_keep_distinct_coordinate_origins() -> None:
    batch = OwnershipTerm("0 < 0", frozenset({"batch"}))
    contracted = OwnershipTerm("0 < 0", frozenset({"k"}))
    terms = ownership_terms((batch, contracted, batch))
    assert terms == (batch, contracted)
    assert len(set(terms)) == 2


def test_dot_domain_preserves_empty_singleton_accumulator() -> None:
    lhs = Fragment((8, 0), torch.float32, lambda _: "0")
    rhs = Fragment((0, 16), torch.float32, lambda _: "0")
    acc = Fragment(
        (1, 16),
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (f"{c[0]} < 0",),
    )
    domain = _dot_output_domain(lhs, rhs, acc, ("row", "col"), "k")
    assert domain is not None
    # K=0 does not veto an output, but broadcasting an accumulator whose
    # logical singleton is empty must retain its coordinate-free false mask.
    for row in range(8):
        for col in range(16):
            assert not all(
                eval(render_ownership(mask), {"__builtins__": {}}, {})
                for mask in domain((str(row), str(col)))
            )


@pytest.mark.parametrize(
    "kind,axis,rows,columns,half_mma",
    [
        ("max", 0, 7, 16, True),
        ("min", 1, 8, 9, True),
        ("prod", 0, 7, 9, True),
        ("max", 0, 7, 16, False),
    ],
)
def test_dot_reduction_keeps_logical_extent(
    kind, axis, rows, columns, half_mma
) -> None:
    bound = cpu_bind(
        dot_then_reduce,
        (
            torch.empty((2, rows, 8), dtype=torch.bfloat16),
            torch.empty((2, 8, columns), dtype=torch.bfloat16),
            kind,
            axis,
        ),
    )
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config())
        | {
            "pid_type": "flat",
            "cute_fragment_half_mma": half_mma,
        }
    )
    source = bound.to_code(config)
    assert ("mma_bf16(" in source) == half_mma
    # Reduction masks use the logical free extent, not just the MMA capacity.
    extent = rows if axis == 0 else columns
    reduction_loops = (
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_reduce_index")
    )
    assert any(
        isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Lt) for op in node.ops)
        and any(
            isinstance(c, ast.Constant) and c.value == extent for c in node.comparators
        )
        for loop in reduction_loops
        for node in ast.walk(loop)
    )


def test_broadcast_axis_alias_cannot_replace_a_real_variable(monkeypatch) -> None:
    monkeypatch.setattr(fragment_domain, "_symbol_ids", iter(range(8)))
    source = Fragment(
        (1,),
        torch.float32,
        lambda _: "0",
        logical_domain=lambda c: (f"{c[0]} < __helion_ownership_axis_0",),
    )
    masks = source.broadcast_domain(("row",))
    assert any("__helion_ownership_axis_0" in mask.coordinates for mask in masks)
    assert all(":=" not in render_ownership(mask) for mask in masks)
    for bound in (0, 1):
        assert all(
            eval(
                render_ownership(mask),
                {"__builtins__": {}},
                {"__helion_ownership_axis_0": bound},
            )
            for mask in masks
        ) == (bound == 1)


@pytest.mark.parametrize(
    "clauses",
    [
        ("(row * 8 + k) % 64 < 56",),
        ("row * 8 + (7 - k) < 56",),
        ("k > row", "2 * k > row + 3"),
    ],
)
def test_mixed_dot_domain_has_one_simultaneous_k_witness(clauses) -> None:
    terms = ownership_terms(("0 <= row", "row < 8", "0 <= k", "k < 8", *clauses))
    projected = project_contracted_ownership(terms, {"row": 8, "k": 8}, "k")
    for row in range(8):
        expected = any(
            all(
                eval(clause, {"__builtins__": {}}, {"row": row, "k": k})
                for clause in clauses
            )
            for k in range(8)
        )
        assert (
            all(
                eval(render_ownership(term), {"__builtins__": {}}, {"row": row})
                for term in projected
            )
            == expected
        )
        assert all("k" not in term.coordinates for term in projected)


@pytest.mark.parametrize(
    "clauses",
    [
        ("k < row", "k > row"),
        ("(row + k) % 3 < 2",),
        ("row * k < 16",),
        ("row * 1073741824 + k < 16",),
        ("row + k < 8", "k < limit"),
        ("row * 8 + k < 56 + cute.arch.thread_idx()[0]",),
    ],
)
def test_mixed_dot_domain_refuses_unproved_projection(clauses) -> None:
    terms = ownership_terms(("0 <= row", "row < 8", "0 <= k", "k < 8", *clauses))
    with pytest.raises(exc.InvalidConfig, match="mixed fragment ownership"):
        project_contracted_ownership(terms, {"row": 8, "k": 8}, "k")


def test_empty_k_does_not_turn_a_mixed_domain_into_an_output_veto() -> None:
    for size, k_bound in ((0, 0), (8, 0)):
        terms = ownership_terms(("row < 7", f"k < {k_bound}", "row * 8 + k < 0"))
        projected = project_contracted_ownership(terms, {"row": 8, "k": size}, "k")
        for row in range(8):
            assert all(
                eval(render_ownership(term), {"__builtins__": {}}, {"row": row})
                for term in projected
            ) == (row < 7)


@pytest.mark.parametrize("half_mma", [False, True])
def test_reshaped_dot_projects_the_flat_logical_bound(half_mma) -> None:
    bound = cpu_bind(
        reshaped_dot_then_max,
        (
            torch.empty((2, 56), dtype=torch.bfloat16),
            torch.empty((8, 16), dtype=torch.bfloat16),
        ),
    )
    source = bound.to_code(
        helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"pid_type": "flat", "cute_fragment_half_mma": half_mma}
        )
    )
    loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_reduce_index")
    ]
    assert any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(value, ast.Constant) and value.value == 56
            for value in node.comparators
        )
        for loop in loops
        for node in ast.walk(loop)
    )
    if not half_mma:
        contraction = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("fragment_k")
        )
        assert any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("fragment_coordinate")
                for target in node.targets
            )
            for node in ast.walk(contraction)
        )


@pytest.mark.parametrize("bound", ["56", "limit"])
def test_dot_ownership_is_required_only_by_its_consumer(bound: str) -> None:
    reads = []

    def input_domain(coords):
        reads.append(coords)
        return (f"{coords[0]} * 8 + {coords[1]} < {bound}",)

    lhs = Fragment((8, 8), torch.float32, lambda _: "0", logical_domain=input_domain)
    rhs = Fragment((8, 16), torch.float32, lambda _: "0")
    domain = _dot_output_domain(lhs, rhs, None, ("row", "column"), "k")
    assert domain is not None
    # Creating a dot result for a direct store needs no ownership proof.
    # An ownership-consuming reduction still rejects an opaque mixed bound.
    assert not reads
    if bound == "limit":
        with pytest.raises(exc.InvalidConfig, match="not bounded affine"):
            domain(("first_row", "first_column"))
        return
    first = domain(("first_row", "first_column"))
    second = domain(("second_row", "second_column"))
    assert len(reads) == 1
    for masks, name, column in (
        (first, "first_row", "first_column"),
        (second, "second_row", "second_column"),
    ):
        for row in range(8):
            assert all(
                eval(
                    render_ownership(mask), {"__builtins__": {}}, {name: row, column: 0}
                )
                for mask in masks
            ) == (row < 7)


@pytest.mark.parametrize("expression", ["a or b", "a if choose else b"])
def test_ownership_rendering_preserves_each_predicate(expression: str) -> None:
    context = object.__new__(FragmentContext)
    rendered = context.predicate(ownership_terms((expression, "c")))
    for values in itertools.product((False, True), repeat=4):
        names = dict(zip(("a", "b", "c", "choose"), values, strict=True))
        expected = eval(expression, {"__builtins__": {}}, names) and names["c"]
        assert eval(rendered, {"__builtins__": {}}, names) == expected
