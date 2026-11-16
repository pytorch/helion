from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.lane_loops.markers import _LaneReduceMarker
from helion._compiler.cute.lane_loops.split import _lane_split_reorders_aliasing_memory
from helion._compiler.cute.persistent_branch_vec import _BranchLocalPersistentVectorizer
from helion._compiler.cute.persistent_branch_vec import _definition_snapshots
from helion._compiler.cute.persistent_branch_vec import _is_affine_in_lane
from helion._compiler.cute.persistent_branch_vec import (
    _memory_address_definition_snapshots,
)
from helion._compiler.cute.persistent_branch_vec import _memory_definition_snapshots

pytestmark = pytest.mark.cpu_only


def _repeated_expression(count: int) -> list[ast.stmt]:
    return ast.parse(
        "value_0 = seed\n"
        + "\n".join(
            f"value_{index} = (value_{index - 1} * 3) ^ value_{index - 1}"
            for index in range(1, count + 1)
        )
        + "\npass"
    ).body


def _hidden_lane_chain(lane: str, steps: int = 12) -> str:
    """Statements defining ``hidden`` as ``-lane`` through a doubling chain of
    ``steps`` (12 is too large to freeze): ``x.iterator + lane + hidden`` is
    ``x[0]`` in every lane."""
    return (
        f"term_0 = {lane}\n"
        + "\n".join(
            f"term_{index} = term_{index - 1} + term_{index - 1}"
            for index in range(1, steps + 1)
        )
        + f"\nhidden = (term_{steps} - term_{steps}) - {lane}\n"
    )


def test_small_assignment_time_definitions_preserve_values() -> None:
    statements = _repeated_expression(4)
    snapshots = _definition_snapshots(statements)
    assert snapshots is not None
    values = {"seed": 17}
    for statement, definitions in zip(statements, snapshots, strict=True):
        for name, expression in definitions.items():
            assert sum(1 for node in ast.walk(expression)) <= 512
            code = compile(ast.Expression(body=expression), "<frozen>", "eval")
            assert eval(code, {}, values) == values[name]
        exec(
            compile(
                ast.Module(body=[statement], type_ignores=[]), "<original>", "exec"
            ),
            {},
            values,
        )


def test_oversized_definition_declines_without_modifying_source() -> None:
    statements = _repeated_expression(80)
    before = ast.dump(ast.Module(body=statements, type_ignores=[]))
    assert _definition_snapshots(statements) is None
    assert ast.dump(ast.Module(body=statements, type_ignores=[])) == before


def test_cached_dependencies_invalidate_aliases_on_reassignment() -> None:
    statements = ast.parse("alias = source\naddress = alias + 4\nsource = 0\npass").body
    snapshots = _definition_snapshots(statements)
    assert snapshots is not None
    assert ast.unparse(snapshots[-2]["alias"]) == "source"
    assert "alias" not in snapshots[-1]
    assert "address" not in snapshots[-1]
    assert ast.literal_eval(snapshots[-1]["source"]) == 0


def test_memory_snapshot_closure_keeps_nested_masks_and_alias_mutations() -> None:
    body = ast.parse(
        "alias = source\n"
        "address = source[0] + lane\n"
        "active = address < limit\n"
        "unused = other * 2\n"
        "if active:\n    value = (x.iterator + address).load()\n"
        "alias[0] = 9\n"
        "if active:\n    (out.iterator + address).store(value)\n"
    ).body
    full = _definition_snapshots(body)
    selected = _memory_definition_snapshots(body)
    assert full is not None and selected is not None
    assert "address" in selected[4] and "active" in selected[4]
    assert "address" not in selected[-1] and "active" not in selected[-1]
    for all_definitions, chosen in zip(full, selected, strict=True):
        assert "unused" not in chosen
        for name, expression in chosen.items():
            assert ast.dump(expression) == ast.dump(all_definitions[name])


def test_oversized_lane_dependency_cannot_justify_load_or_store_motion() -> None:
    body = ast.parse(
        _hidden_lane_chain("lane")
        + "value = (x.iterator + lane + hidden).load()\n"
        + "(x.iterator + lane + hidden).store(value)\n"
    ).body
    # Every iteration reads/writes x[0]. Forgetting `hidden` would incorrectly
    # make the pointer's visible `lane` term appear to prove independence.
    assert _definition_snapshots(body) is None
    load = ast.parse(
        "marker(0, 0, 0, 0, x.iterator + lane + hidden, True)", mode="eval"
    ).body
    store = ast.parse(
        "marker(0, 0, 0, x.iterator + lane + hidden, value, True)", mode="eval"
    ).body
    assert isinstance(load, ast.Call) and isinstance(store, ast.Call)
    vectorizer = _BranchLocalPersistentVectorizer(body, set(), {})
    # The addresses' snapshots decline: ``hidden`` reaches both.
    snapshots = _memory_address_definition_snapshots(body)
    assert snapshots is None
    assert vectorizer._has_aliasing_write_in_loop(
        body, load, len(body) - 2, "lane", 4, snapshots
    )
    assert vectorizer._has_other_aliasing_access_in_loop(
        body, store, len(body) - 1, "lane", 4, snapshots
    )
    body.extend(ast.parse("reduced = value").body)
    marker = _LaneReduceMarker(
        "reduced", "value", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert _lane_split_reorders_aliasing_memory(
        list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
    )


def test_disjoint_memory_does_not_require_expanding_unrelated_arithmetic() -> None:
    body = ast.parse("value = (x.iterator + lane).load()").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "result = value + value_80\n"
            "(out.iterator + lane).store(result)\n"
            "reduced = result\n"
        ).body
    )
    marker = _LaneReduceMarker(
        "reduced", "result", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert _definition_snapshots(body) is None
    assert not _lane_split_reorders_aliasing_memory(
        list(body),
        [(len(body) - 1, marker)],
        {frozenset(("x", "out"))},
        "lane",
        4,
        {},
    )
    # Without the allocation proof the same source must remain conservative.
    assert _lane_split_reorders_aliasing_memory(
        list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
    )


def test_packed_word_load_without_a_write_has_nothing_to_reorder() -> None:
    # The FP4 decode helpers load a byte-packed word whose element count the
    # access proof cannot see.  With no write in the lane body there is still
    # nothing for the split to reorder; with one, the split stays refused.
    marker = _LaneReduceMarker(
        "reduced", "value", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    load = (
        "w = cute.arch.load(x.iterator + lane * 8, cutlass.Uint64)\n"
        "value = cutlass.Float32(w)\n"
        "reduced = value\n"
    )
    for source, reorders in (
        (load, False),
        (load + "(x.iterator + lane).store(reduced)\n", True),
        ("(x.iterator + lane).store(cutlass.Float32(0))\n" + load, True),
    ):
        body = ast.parse(source).body
        index = next(
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "reduced"
        )
        assert (
            _lane_split_reorders_aliasing_memory(
                body, [(index, marker)], set(), "lane", 2, {}
            )
            is reorders
        )


@pytest.mark.parametrize(
    "store_offset, reorders", [("lane", False), ("lane + 1", True)]
)
def test_address_proof_ignores_unrelated_large_values(
    store_offset: str, reorders: bool
) -> None:
    body = ast.parse("pointer = x.iterator + lane\nvalue = pointer.load()").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "result = value + value_80\n"
            f"(x.iterator + {store_offset}).store(result)\n"
            "reduced = result\n"
        ).body
    )
    before = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert _definition_snapshots(body) is None
    snapshots = _memory_address_definition_snapshots(body)
    assert snapshots is not None
    assert ast.unparse(snapshots[1]["pointer"]) == "x.iterator + lane"
    assert all("value_80" not in definitions for definitions in snapshots)
    marker = _LaneReduceMarker(
        "reduced", "result", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert (
        _lane_split_reorders_aliasing_memory(
            list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
        )
        is reorders
    )
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == before


@pytest.mark.parametrize("overwrite", ["", "hidden = 0\n"])
def test_address_slice_keeps_every_reaching_hidden_lane_definition(
    overwrite: str,
) -> None:
    body = ast.parse(
        _hidden_lane_chain("lane")
        + "value = (x.iterator + lane + hidden).load()\n"
        + overwrite
        + "(x.iterator + lane + hidden).store(value)\n"
    ).body
    assert _memory_address_definition_snapshots(body) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "alias[0] = 9",
        "alias.field = 9",
        "source += 1",
        "source: int = 9",
        "for source in range(2):\n    pass",
        "del source",
    ],
)
def test_address_slice_retains_all_write_invalidation(mutation: str) -> None:
    body = ast.parse(
        "alias = source\n"
        "address = source[0] + lane\n"
        "unused = other * 2\n" + mutation + "\npass\n"
    ).body
    full = _definition_snapshots(body)
    selected = _definition_snapshots(body, required_names={"address"})
    assert full is not None and selected is not None
    assert "address" in selected[-2]
    assert "address" not in selected[-1]
    for all_definitions, selected_definitions in zip(full, selected, strict=True):
        for name, expression in selected_definitions.items():
            assert ast.dump(expression) == ast.dump(all_definitions[name])
        assert "unused" not in selected_definitions


def test_address_slice_does_not_expand_store_value_in_vector_marker() -> None:
    body = ast.parse("address = x.iterator + synthetic_lane_0\n").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "value = _helion_persistent_branch_vec_load("
            "0, 4, 'cutlass.Float32', 0.0, address, True)\n"
            "_helion_persistent_branch_vec_store("
            "0, 4, 'cutlass.Float32', address, value + value_80, True)\n"
        ).body
    )
    assert _definition_snapshots(body) is None
    snapshots = _memory_address_definition_snapshots(body)
    assert snapshots is not None
    assert ast.unparse(snapshots[-1]["address"]) == "x.iterator + synthetic_lane_0"
    assert "value_80" not in snapshots[-1]


def _packet_sweep(load: str, store: str, *, chain: bool = True) -> list[ast.stmt]:
    """A row sweep: ``load``, then (with ``chain``) a long pure chain that
    reaches no address, as a word0 Philox draw does, then ``store``."""
    body = ast.parse(f"{load}\n").body
    body.extend(_repeated_expression(80 if chain else 1))
    body.extend(
        ast.parse(f"result = value + value_{80 if chain else 1}\n{store}\n").body
    )
    assert (_definition_snapshots(body) is None) is chain
    return body


@pytest.mark.parametrize(
    ("store_offset", "aliases"),
    [("synthetic_lane_1", False), ("synthetic_lane_1 + 1", True)],
)
def test_packet_aliasing_checks_ignore_unrelated_large_values(
    store_offset: str, aliases: bool
) -> None:
    # The hoisted load and the sunk store move across lanes: only another
    # lane's element in the same tensor may decline them, never the chain.
    load = (
        "value = _helion_persistent_branch_vec_load("
        "1, 4, 'cutlass.Float32', '', x.iterator + synthetic_lane_1, x0)"
    )
    store = (
        "_helion_persistent_branch_vec_store("
        f"1, 4, 'cutlass.Float32', x.iterator + {store_offset}, result, None)"
    )
    body = _packet_sweep(load, store)
    load_marker = body[0].value
    store_marker = body[-1].value
    assert isinstance(load_marker, ast.Call) and isinstance(store_marker, ast.Call)
    vectorizer = _BranchLocalPersistentVectorizer(body, set(), {})
    snapshots = _memory_address_definition_snapshots(body)
    assert (
        vectorizer._has_aliasing_write_in_loop(
            body, load_marker, 0, "synthetic_lane_1", 4, snapshots
        )
        is aliases
    )
    assert (
        vectorizer._has_other_aliasing_access_in_loop(
            body, store_marker, len(body) - 1, "synthetic_lane_1", 4, snapshots
        )
        is aliases
    )


@pytest.mark.parametrize("steps", [12, 2])
def test_oversized_lane_definition_reaching_marker_addresses_declines(
    steps: int,
) -> None:
    """``hidden`` cancels the lane out of both markers' addresses (every lane
    reads and writes ``x[0]``): neither the hoisted load nor the sunk store
    may take the visible ``synthetic_lane_1`` term for an injective lane
    mapping.  Too large to freeze (12 steps), the definition declines the
    addresses' snapshots; frozen (2 steps), the address proofs see it."""
    body = ast.parse(
        _hidden_lane_chain("synthetic_lane_1", steps)
        + "value = _helion_persistent_branch_vec_load("
        "1, 4, 'cutlass.Float32', '', x.iterator + synthetic_lane_1 + hidden, x0)\n"
        "_helion_persistent_branch_vec_store("
        "1, 4, 'cutlass.Float32', x.iterator + synthetic_lane_1 + hidden, value, None)\n"
    ).body
    load_marker = body[-2].value
    store_marker = body[-1].value
    assert isinstance(load_marker, ast.Call) and isinstance(store_marker, ast.Call)
    vectorizer = _BranchLocalPersistentVectorizer(body, set(), {})
    snapshots = _memory_address_definition_snapshots(body)
    assert (snapshots is None) is (steps == 12)
    assert vectorizer._has_aliasing_write_in_loop(
        body, load_marker, len(body) - 2, "synthetic_lane_1", 4, snapshots
    )
    assert vectorizer._has_other_aliasing_access_in_loop(
        body, store_marker, len(body) - 1, "synthetic_lane_1", 4, snapshots
    )


@pytest.mark.parametrize(
    "load",
    [
        "value = _cute_load_l2_evict_last_8b()",
        # A tensor among the arguments of a form the parser cannot read (its
        # vector type is a bare name) does not name what the load reads.
        "value = _cute_load_l2_evict_last_8b(y.iterator + synthetic_lane_1 * 2, vt)",
    ],
)
def test_a_hinted_load_the_parser_cannot_read_may_alias_any_store(load: str) -> None:
    # Its pointer unknown, an L2-hinted load must decline the sunk store even
    # beside proven-disjoint tensors; it used to count as no load at all.
    store = (
        "_helion_persistent_branch_vec_store("
        "1, 4, 'cutlass.BFloat16', out.iterator + synthetic_lane_1, result, None)"
    )
    body = _packet_sweep(load, store)
    store_marker = body[-1].value
    assert isinstance(store_marker, ast.Call)
    vectorizer = _BranchLocalPersistentVectorizer(body, {frozenset(("y", "out"))}, {})
    assert vectorizer._has_other_aliasing_access_in_loop(
        body,
        store_marker,
        len(body) - 1,
        "synthetic_lane_1",
        4,
        _memory_address_definition_snapshots(body),
    )


@pytest.mark.parametrize(
    "pointer, expected",
    [
        ("x.iterator + lane", (True, True)),
        ("x.iterator + cutlass.Int32(lane)", (True, True)),
        # The lane in a call's receiver or keyword makes it lane dependent.
        ("x.iterator + cutlass.Int32(v[lane]).bitcast(cutlass.Int32)", (True, False)),
        ("x.iterator + f(n=lane)", (True, False)),
        ("x.iterator + cutlass.Int32(v[0]).bitcast(cutlass.Int32)", (False, True)),
    ],
)
def test_lane_dependence_reaches_into_a_calls_receiver_and_keywords(
    pointer: str, expected: tuple[bool, bool]
) -> None:
    assert _is_affine_in_lane(ast.parse(pointer, mode="eval").body, "lane") == expected


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize(
    "load",
    [
        "value = (y.iterator + synthetic_lane_1 * 2).load()",
        (
            "value = _cute_load_l2_evict_last_8b(y.iterator + synthetic_lane_1 * 2,"
            " ir.VectorType.get([4], cutlass.Uint16.mlir_type))"
        ),
    ],
)
@pytest.mark.parametrize("disjoint", [False, True])
def test_packet_store_checks_every_load_of_a_possibly_aliasing_tensor(
    load: str, disjoint: bool, chain: bool
) -> None:
    # A load of another tensor the store may alias (y could be out's storage)
    # declines the sunk store, the L2-hinted packet form too; a proven-disjoint
    # pair needs no address proof, with or without the chain beside it.
    store = (
        "_helion_persistent_branch_vec_store("
        "1, 4, 'cutlass.BFloat16', out.iterator + synthetic_lane_1, result, None)"
    )
    body = _packet_sweep(load, store, chain=chain)
    store_marker = body[-1].value
    assert isinstance(store_marker, ast.Call)
    pairs = {frozenset(("y", "out"))} if disjoint else set()
    vectorizer = _BranchLocalPersistentVectorizer(body, pairs, {})
    assert (
        vectorizer._has_other_aliasing_access_in_loop(
            body,
            store_marker,
            len(body) - 1,
            "synthetic_lane_1",
            4,
            _memory_address_definition_snapshots(body),
        )
        is not disjoint
    )
