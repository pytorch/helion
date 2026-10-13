from __future__ import annotations

import ast
import copy
import copyreg
import sys
from unittest.mock import patch

import pytest

from helion._compiler.ast_deepcopy import _ordinary_ast_protocols
from helion._compiler.ast_deepcopy import deepcopy_ast


def _graph(value: object) -> tuple[object, list[object]]:
    """Describe values and object sharing independently of their addresses."""
    objects: list[object] = []
    indices: dict[int, int] = {}

    def ref(item: object) -> object:
        if type(item) in (type(None), bool, int, float, complex, str, bytes):
            return (type(item).__name__, item)
        if id(item) not in indices:
            indices[id(item)] = len(objects)
            objects.append(item)
        return ("ref", indices[id(item)])

    root = ref(value)
    rows: list[object] = []
    cursor = 0
    while cursor < len(objects):
        item = objects[cursor]
        if isinstance(item, ast.AST):
            rows.append((type(item).__name__, ref(item.__dict__)))
        elif isinstance(item, dict):
            rows.append(("dict", [(ref(k), ref(v)) for k, v in item.items()]))
        elif isinstance(item, (list, tuple)):
            rows.append((type(item).__name__, [ref(v) for v in item]))
        else:
            raise AssertionError(type(item))
        cursor += 1
    return root, rows


@pytest.mark.parametrize("cycle", [False, True])
@pytest.mark.parametrize("expose_state", [False, True])
def test_ast_deepcopy_matches_stdlib_sharing_and_metadata(
    cycle: bool, expose_state: bool
) -> None:
    tree = ast.parse("y = x + x\nz = x + 1")
    shared = tree.body[0].value.left
    tree.body[1].value.left = shared
    tree.metadata = {"tuple": (shared, shared), "bytes": b"data", "value": 2j}
    tree.shared_metadata = tree.metadata
    if cycle:
        shared.parent = tree
        tree.metadata["self"] = tree.metadata
    if expose_state:
        tree.metadata["state"] = shared.__dict__
        shared.state = shared.__dict__
    before = _graph(tree)
    expected = copy.deepcopy(tree)
    actual = deepcopy_ast(tree)
    assert _graph(actual) == _graph(expected)
    assert _graph(tree) == before
    assert actual is not tree
    actual.body[0].value.left.id = "different"
    assert shared.id == "x"


def test_ast_deepcopy_deep_graph() -> None:
    depth = sys.getrecursionlimit() * 2
    leaf = ast.Name(id="x", ctx=ast.Load())
    root = leaf
    for _ in range(depth):
        root = ast.UnaryOp(op=ast.USub(), operand=root)
    before = _graph(root)
    limit = sys.getrecursionlimit()
    with pytest.raises(RecursionError):
        copy.deepcopy(root)
    copied = deepcopy_ast(root)
    assert _graph(copied) == before
    assert _graph(root) == before
    assert sys.getrecursionlimit() == limit


@pytest.mark.parametrize("kind", ["subclass", "metadata", "container"])
def test_ast_deepcopy_custom_protocol_falls_back_once(kind: str) -> None:
    calls: list[str] = []

    class CustomNode(ast.AST):
        def __deepcopy__(self, memo: dict[int, object]) -> ast.AST:
            calls.append("node")
            return ast.Constant(value=7)

    class Metadata:
        def __deepcopy__(self, memo: dict[int, object]) -> str:
            calls.append("metadata")
            return "copied"

    class Container(list):
        def __deepcopy__(self, memo: dict[int, object]) -> list[int]:
            calls.append("container")
            return [9]

    root = ast.parse("x + 1")
    root.metadata = {
        "subclass": CustomNode(),
        "metadata": Metadata(),
        "container": Container(),
    }[kind]
    expected = copy.deepcopy(root)
    expected_calls = list(calls)
    calls.clear()
    actual = deepcopy_ast(root)
    assert _graph(actual) == _graph(expected)
    assert calls == expected_calls


def test_ast_deepcopy_registered_reducer_falls_back() -> None:
    calls: list[str] = []

    def reduce_name(node: ast.Name) -> tuple[object, tuple[()]]:
        calls.append(node.id)
        return ast.Pass, ()

    tree = ast.parse("x + y")
    with patch.dict(copyreg.dispatch_table, {ast.Name: reduce_name}):
        expected = copy.deepcopy(tree)
        expected_calls = list(calls)
        calls.clear()
        actual = deepcopy_ast(tree)
    assert _graph(actual) == _graph(expected)
    assert calls == expected_calls


def test_ast_deepcopy_instance_protocol_falls_back() -> None:
    tree = ast.parse("x + 1")
    calls: list[str] = []

    def clone(memo: dict[int, object]) -> ast.AST:
        calls.append("instance")
        return ast.Pass()

    tree.__deepcopy__ = clone
    expected = copy.deepcopy(tree)
    actual = deepcopy_ast(tree)
    assert _graph(actual) == _graph(expected)
    assert calls == ["instance", "instance"]


def test_ast_deepcopy_set_metadata() -> None:
    tree = ast.parse("x + 1")
    tree.metadata = {"set": {1, 2}, "frozen": frozenset({3, 4})}
    actual = deepcopy_ast(tree)
    expected = copy.deepcopy(tree)
    assert ast.dump(actual, include_attributes=True) == ast.dump(
        expected, include_attributes=True
    )
    assert actual.metadata == expected.metadata
    assert actual.metadata["set"] is not tree.metadata["set"]


@pytest.mark.parametrize("container", [set, frozenset])
def test_ast_deepcopy_container_reducer_cannot_observe_shallow_shell(
    container: type,
) -> None:
    records = []
    for cloner in (copy.deepcopy, deepcopy_ast):
        leaf = ast.Name(id="x", ctx=ast.Load())
        leaf.metadata = ["original"]
        root = ast.Module(body=[leaf], type_ignores=[])
        root.tags = container([leaf])
        observations = []

        def construct(node, observations=observations, leaf=leaf):
            observations.append(node.metadata is leaf.metadata)
            node.metadata.append("constructor mutation")
            return container([node])

        def reducer(value):
            return construct, (next(iter(value)),)

        with patch.dict(copyreg.dispatch_table, {container: reducer}):
            result = cloner(root)
        assert leaf.metadata == ["original"]
        records.append((observations, result.body[0].metadata))
    assert records[0] == records[1] == ([False], ["original", "constructor mutation"])


@pytest.mark.parametrize(
    "protocol", ["__init__", "__hash__", "__eq__", "__getattribute__"]
)
def test_ast_deepcopy_requires_native_protocol_descriptors(protocol: str) -> None:
    def custom(*args, **kwargs):
        raise AssertionError("admission must not call custom protocols")

    with patch.object(ast.Name, protocol, custom):
        assert not _ordinary_ast_protocols(ast.Name)
    assert _ordinary_ast_protocols(ast.Name)


def test_ast_deepcopy_native_constructor_order_preserved() -> None:
    root = ast.parse("a + b")
    records = []
    original = ast.Name.__init__
    for cloner in (copy.deepcopy, deepcopy_ast):
        calls = []

        def constructor(self, *args, calls=calls, **kwargs):
            original(self, *args, **kwargs)
            calls.append(self)
            self.copy_order = len(calls)

        with patch.object(ast.Name, "__init__", constructor):
            result = cloner(root)
        records.append(_graph(result))
    assert records[0] == records[1]


def test_ast_deepcopy_native_hash_cannot_mutate_original() -> None:
    records = []
    for cloner in (copy.deepcopy, deepcopy_ast):
        leaf = ast.Name(id="x", ctx=ast.Load())
        leaf.metadata = ["original"]
        root = ast.Module(body=[leaf], type_ignores=[])
        root.mapping = {leaf: 1}

        def hash_node(self):
            self.metadata.append("hashed")
            return object.__hash__(self)

        with patch.object(ast.Name, "__hash__", hash_node):
            result = cloner(root)
        assert leaf.metadata == ["original"]
        records.append(_graph(result))
    assert records[0] == records[1]


def test_ast_deepcopy_admission_does_not_invoke_descriptors() -> None:
    records = []
    for cloner in (copy.deepcopy, deepcopy_ast):
        root = ast.Name(id="x", ctx=ast.Load())
        calls = []

        class Descriptor:
            def __get__(self, instance, owner, calls=calls):
                calls.append("class" if instance is None else "instance")
                return lambda memo: ast.Pass()

        with patch.object(ast.Name, "__deepcopy__", Descriptor(), create=True):
            result = cloner(root)
        records.append((calls, ast.dump(result)))
    assert records[0] == records[1] == (["instance"], "Pass()")


def test_ast_deepcopy_does_not_iterate_custom_state_before_fallback() -> None:
    class State(dict):
        def __iter__(self):
            raise AssertionError("stdlib reconstruction does not iterate state")

    node = ast.Name(id="x", ctx=ast.Load())
    node.__dict__ = State(node.__dict__)
    assert _graph(deepcopy_ast(node)) == _graph(copy.deepcopy(node))


def test_ast_deepcopy_does_not_hash_custom_state_keys_before_fallback() -> None:
    class Key:
        def __init__(self):
            self.mutations = []

        def __hash__(self):
            self.mutations.append("hash")
            return 123

    records = []
    for cloner in (copy.deepcopy, deepcopy_ast):
        key = Key()
        node = ast.Name(id="x", ctx=ast.Load())
        node.__dict__[key] = 1
        key.mutations.clear()
        result = cloner(node)
        copied_key = next(k for k in result.__dict__ if isinstance(k, Key))
        records.append((key.mutations, copied_key.mutations))
    assert records[0] == records[1]
    assert records[0][0] == []


def test_ast_deepcopy_custom_metaclass_has_no_extra_hash_calls() -> None:
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("metaclass hash")
            return type.__hash__(cls)

    class Metadata(metaclass=Meta):
        pass

    node = ast.Name(id="x", ctx=ast.Load())
    node.metadata = Metadata()
    calls.clear()
    copy.deepcopy(node)
    expected = list(calls)
    calls.clear()
    deepcopy_ast(node)
    assert calls == expected


def test_ast_deepcopy_dynamic_copy_protocol_falls_back() -> None:
    node = ast.Name(id="original", ctx=ast.Load())

    def dynamic(self, name):
        if name == "__deepcopy__":
            return lambda memo: ast.Name(id="custom", ctx=ast.Load())
        raise AttributeError(name)

    with patch.object(ast.Name, "__getattr__", dynamic, create=True):
        expected = copy.deepcopy(node)
        actual = deepcopy_ast(node)
    assert _graph(actual) == _graph(expected)
    assert actual.id == "custom"


@pytest.mark.parametrize(
    "kind", [ast.Name, list, dict, tuple, str, int, set, frozenset]
)
def test_ast_deepcopy_custom_deep_dispatch_falls_back(kind: type) -> None:
    values = {
        ast.Name: ast.Name(id="x", ctx=ast.Load()),
        list: [ast.Name(id="x", ctx=ast.Load())],
        dict: {"x": ast.Name(id="x", ctx=ast.Load())},
        tuple: (ast.Name(id="x", ctx=ast.Load()),),
        str: "x",
        int: 123,
        set: {1},
        frozenset: frozenset({1}),
    }
    calls = []

    def handler(value, memo):
        calls.append(kind)
        return ast.Pass()

    with patch.dict(vars(copy)["_deepcopy_dispatch"], {kind: handler}):
        expected = copy.deepcopy(values[kind])
        expected_calls = list(calls)
        calls.clear()
        actual = deepcopy_ast(values[kind])
    assert _graph(actual) == _graph(expected)
    assert calls == expected_calls


def test_ast_deepcopy_does_not_use_shallow_dispatch() -> None:
    node = ast.Name(id="x", ctx=ast.Load())

    def forbidden(value):
        raise AssertionError("deepcopy must not consult shallow dispatch")

    with patch.dict(vars(copy)["_copy_dispatch"], {ast.Name: forbidden}):
        assert _graph(deepcopy_ast(node)) == _graph(copy.deepcopy(node))


def test_ast_deepcopy_rebound_reducer_registry_falls_back() -> None:
    node = ast.Name(id="x", ctx=ast.Load())
    calls = []

    def reducer(value):
        calls.append(value.id)
        return ast.Pass, ()

    with patch.dict(vars(copy), {"dispatch_table": {ast.Name: reducer}}):
        expected = copy.deepcopy(node)
        actual = deepcopy_ast(node)
    assert _graph(actual) == _graph(expected)
    assert calls == ["x", "x"]


def test_ast_deepcopy_allows_unrelated_identity_metaclass_registry_keys() -> None:
    class Meta(type):
        pass

    class Extension(metaclass=Meta):
        pass

    def reducer(value):
        raise AssertionError("unrelated reducer must not be invoked")

    root = ast.Name(id="x", ctx=ast.Load())
    for _ in range(sys.getrecursionlimit()):
        root = ast.UnaryOp(op=ast.USub(), operand=root)
    with patch.dict(vars(copy)["dispatch_table"], {Extension: reducer}):
        with pytest.raises(RecursionError):
            copy.deepcopy(root)
        assert _graph(deepcopy_ast(root)) == _graph(root)
