from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from helion._compiler.cute.coordinate_sdk_effects import coordinate_sdk_effects
from helion._compiler.cute.coordinate_sdk_effects import readonly_slice_uses
from helion._compiler.cute.coordinate_subtree_sharing import share_coordinate_subtree

EXPR = "((q // 8 * 8 + q % 8) // 4 % 16)"
SOURCE = f"""
t = cutlass.Int32(cute.arch.thread_idx()[0])
for step in cutlass.range_constexpr(4):
    q = t + step * 32
    a = {EXPR} + 1
    b = {EXPR} + 2
    store(a + b)
"""


def _share(source=SOURCE, *, sdk=None, name="shared_coordinate", excluded=frozenset()):
    body = ast.parse(source).body if isinstance(source, str) else source
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    result, proof = share_coordinate_subtree(
        body,
        thread_block_dims=(32, 1, 1),
        block_grid_dims=(3, 1, 1),
        excluded_names=excluded,
        protected_names=frozenset({"argument"}),
        fresh_name=lambda prefix: name,
        sdk_effects=sdk,
    )
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == original
    if proof is not None:
        proof.check_before(body)
        proof.check_inverse(result)
    return body, result, proof


class _Int32:
    """Mutable wrappers expose accidental result aliasing in transformed Python."""

    def __init__(self, value):
        self.value = value.value if isinstance(value, _Int32) else value

    def __add__(self, other):
        return _Int32(self.value + _Int32(other).value)

    __radd__ = __add__

    def __mul__(self, other):
        return _Int32(self.value * _Int32(other).value)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return _Int32(self.value // _Int32(other).value)

    def __mod__(self, other):
        return _Int32(self.value % _Int32(other).value)


def _execute(body, thread):
    events = []

    def mutate(value):
        value.value = 700

    namespace = {
        "cutlass": SimpleNamespace(Int32=_Int32, range_constexpr=range),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(thread_idx=lambda: (thread, 0, 0))
        ),
        "store": lambda value: events.append(value.value),
        "mutate": mutate,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<sharing>", "exec"), namespace
    )
    return events


def test_runtime_values_fresh_wrappers_and_complete_inverse():
    before, after, proof = _share()
    assert proof is not None and proof.occurrences == 2 and proof.nodes_saved > 0
    for thread in range(32):
        assert _execute(before, thread) == _execute(after, thread)


@pytest.mark.parametrize(
    "statement",
    [
        "mutate(t)",
        "mutate([t])",
        "alias = t",
        "t.value = 9",
        "t.value += 9",
        "t = cutlass.Int32(9)",
        "for t in cutlass.range_constexpr(2):\n    pass",
        "del t",
        "import math",
        "def nested():\n    return t",
    ],
)
def test_dependency_escape_rebinding_and_unsupported_control_decline(statement):
    source = SOURCE + "\n" + statement
    before, after, proof = _share(source)
    assert proof is None and before is after


def test_unknown_receiver_gets_fresh_expression_wrapper():
    source = SOURCE.replace(
        f"    b = {EXPR} + 2", f"    mutate({EXPR})\n    b = {EXPR} + 2"
    )
    before, after, proof = _share(source)
    # Smaller safe descendants may still be shared, but the escaping wrapper is not.
    if proof is not None:
        assert ast.dump(ast.parse(proof.expression, mode="eval").body) != ast.dump(
            ast.parse(EXPR, mode="eval").body
        )
    for thread in range(32):
        assert _execute(before, thread) == _execute(after, thread)


@pytest.mark.parametrize(
    "replacement",
    [
        "q - 1",
        "q / 2",
        "q // 0",
        "q // -1",
        "q % 0",
        "q * (1 << 31)",
        "float(q)",
        "q + unknown",
        "q & 15",
        "q ** 2",
    ],
)
def test_unknown_operators_divisors_and_types_decline(replacement):
    before, after, proof = _share(SOURCE.replace(EXPR, f"({replacement})"))
    assert proof is None and before is after


@pytest.mark.parametrize("name", ["t", "q", "step", "argument", "store", "cutlass"])
def test_fresh_name_cannot_capture_any_binding_or_free_name(name):
    with pytest.raises(ValueError, match="collision"):
        _share(name=name)


def test_rename_authority_excludes_dependencies():
    assert _share(excluded=frozenset({"t"}))[2] is None
    assert _share(excluded=frozenset({"q"}))[2] is None


def test_first_use_must_be_unconditional_pure_assignment():
    source = SOURCE.replace(
        f"    a = {EXPR} + 1", f"    a = {EXPR} + 1 if opaque else 0"
    )
    assert _share(source)[2] is None


def test_constexpr_does_not_become_runtime_int32():
    source = SOURCE.replace("cutlass.Int32(cute.arch.thread_idx()[0])", "4")
    assert _share(source)[2] is None


@pytest.mark.parametrize("change", ["binding", "use", "unrelated", "operator"])
def test_complete_receipt_rejects_every_body_change(change):
    _, after, proof = _share()
    assert proof is not None
    loop = after[1]
    assert isinstance(loop, ast.For)
    binding, use = loop.body[1:3]
    assert isinstance(binding, ast.Assign) and isinstance(use, ast.Assign)
    target = binding.targets[0]
    assert isinstance(target, ast.Name)
    assert isinstance(binding.value, ast.BinOp)
    assert isinstance(use.value, ast.BinOp)
    assert isinstance(use.value.left, ast.Name)
    if change == "binding":
        target.id = "different"
    elif change == "use":
        use.value.left.id = "different"
    elif change == "unrelated":
        after.append(ast.Pass())
    else:
        binding.value.op = ast.Mult()
    with pytest.raises(ValueError):
        proof.check_inverse(after)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda tree: setattr(tree[1], "body", tuple(tree[1].body)),
        lambda tree: setattr(
            tree[1], "_fields", ("target", "iter", "orelse", "type_comment")
        ),
        lambda tree: setattr(
            tree[1].body[0], "targets", tuple(tree[1].body[0].targets)
        ),
    ],
)
def test_noncanonical_native_ast_schema_declines(mutate):
    body = ast.parse(SOURCE).body
    mutate(body)
    assert _share(body)[2] is None


SLICE = """
mma = cute.make_tiled_mma(atom)
thr = mma.get_slice(t)
a = thr.partition_A(tensor)
b = thr.partition_B(tensor)
c = thr.partition_C(tensor)
"""


def test_sdk_receiver_contract_preserves_retained_numeric_root():
    sdk = coordinate_sdk_effects()
    assert sdk is not None
    source = SOURCE.replace("for step", SLICE + "for step")
    body = ast.parse(source).body
    uses = readonly_slice_uses(body)
    assert len(uses) == 1
    assert _share(source)[2] is None
    assert _share(source, sdk=sdk)[2] is not None


@pytest.mark.parametrize(
    "extra",
    [
        "escape(thr)",
        "alias = thr",
        "saved = thr.thr_idx",
        "saved = thr._thr_idx",
        "thr.unknown(tensor)",
        "thr.__new_from_mlir_values__(values)",
        "thr.partition_A = different",
        "thr.partition_A += different",
    ],
)
def test_retained_receiver_escape_or_mutation_declines(extra):
    sdk = coordinate_sdk_effects()
    assert sdk is not None
    source = SOURCE.replace("for step", SLICE + extra + "\nfor step")
    assert _share(source, sdk=sdk)[2] is None


def test_unknown_factory_and_pre_slice_escape_decline():
    sdk = coordinate_sdk_effects()
    assert sdk is not None
    for changed in (
        SLICE.replace("cute.make_tiled_mma", "unknown"),
        SLICE.replace("thr =", "escape(mma)\nthr ="),
    ):
        assert (
            _share(SOURCE.replace("for step", changed + "for step"), sdk=sdk)[2] is None
        )


def test_sdk_callable_mutation_revokes_existing_contract(monkeypatch):
    from cutlass.cute import atom

    sdk = coordinate_sdk_effects()
    assert sdk is not None
    monkeypatch.setattr(atom.ThrMma, "partition_A", lambda self, tensor: tensor)
    assert coordinate_sdk_effects() is None
    with pytest.raises(ValueError, match="contract changed"):
        sdk.check()


@pytest.mark.parametrize(
    "target", ["sink[t]", "sink[(0, t)]", "sink[:t]", "sink[t:]", "sink[::t]"]
)
@pytest.mark.parametrize("operator", ["=", "+="])
def test_unknown_store_target_cannot_mutate_coordinate_root(target, operator):
    source = SOURCE.replace("for step", f"{target} {operator} 0\nfor step")
    assert _share(source)[2] is None


def test_fresh_arithmetic_target_does_not_escape_coordinate_root():
    source = SOURCE.replace("for step", "sink[t + 1] = 0\nfor step")
    assert _share(source)[2] is not None


@pytest.mark.parametrize(
    "statement",
    [
        "for sink[t] in items:\n    pass",
        "with manager as sink[t]:\n    pass",
        "from pkg import *",
        "from pkg import t",
        "from pkg import value as t",
    ],
)
def test_other_binding_and_store_target_boundaries(statement):
    assert _share(SOURCE + statement + "\n")[2] is None


def test_imported_unused_alias_is_still_a_protected_binding():
    source = SOURCE.replace(
        f"    b = {EXPR} + 2",
        f"    from pkg import value as shared_coordinate\n    b = {EXPR} + 2",
    )
    with pytest.raises(ValueError, match="collision"):
        _share(source)


@pytest.mark.parametrize("name", ["mma", "thr"])
def test_imported_receiver_rebinding_invalidates_sdk_exemption(name):
    sdk = coordinate_sdk_effects()
    assert sdk is not None
    changed = (
        SLICE.replace("thr =", "from pkg import replacement as mma\nthr =")
        if name == "mma"
        else SLICE.replace("\na =", "\nfrom pkg import replacement as thr\na =")
    )
    assert _share(SOURCE.replace("for step", changed + "for step"), sdk=sdk)[2] is None


@pytest.mark.parametrize("edge", ["transform_leaf", "promote", "signed"])
def test_actual_sdk_lookup_edge_or_type_field_is_pinned(monkeypatch, edge):
    from cutlass.base_dsl import typing as numeric
    from cutlass.cute import core

    sdk = coordinate_sdk_effects()
    assert sdk is not None
    if edge == "transform_leaf":
        monkeypatch.setattr(core, "transform_leaf", lambda function, value: value)
    elif edge == "promote":
        monkeypatch.setattr(numeric, "_binary_op_type_promote", lambda *args: args)
    else:
        monkeypatch.setattr(numeric.Int32, "signed", False)
    assert coordinate_sdk_effects() is None
    with pytest.raises(ValueError, match="contract changed"):
        sdk.check()
