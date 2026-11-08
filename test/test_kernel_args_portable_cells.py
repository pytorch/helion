from __future__ import annotations

import dataclasses
from itertools import islice
import marshal
import pickle

import pytest
import torch
from torch.utils._pytree import register_pytree_node

from helion.autotuner import kernel_args
from helion.autotuner.accuracy import assert_close
from helion.autotuner.accuracy import relative_l2_disagreement
from helion.autotuner.kernel_args import PortableFunction
from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.kernel_args import make_portable_args
from helion.autotuner.kernel_args import resolve_portable_args

pytestmark = pytest.mark.cpu_only


def _cell(callback, name):
    return dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))[
        name
    ]


def _independent_counter():
    state = 0

    def increment():
        nonlocal state
        state += 1
        return state

    return increment


def _shared_default_callback(callback):
    def outer(first=callback, second=callback, *, keyword=callback):
        return callback if first is second is keyword is callback else None

    return outer


def test_shared_callback_defaults_have_bounded_traversal_and_roundtrip():
    tensor = torch.arange(3, dtype=torch.float32)

    def leaf():
        tensor.add_(1)
        return tensor

    depth = 20
    callback = leaf
    for _ in range(depth):
        callback = _shared_default_callback(callback)

    # Bound a failing walk by object count, not elapsed time. Walking every
    # reference path through these shared defaults would expand exponentially.
    operands = list(islice(kernel_args._input_tree_leaves(callback), depth + 2))
    assert len(operands) == 1 and operands[0] is tensor

    # Check sharing on a small graph before serializing the deep graph, so a
    # serializer regression fails without constructing an exponential payload.
    small = make_portable_args((_shared_default_callback(leaf), leaf))
    assert small is not None
    outer, inner = small
    assert outer.defaults[0] is outer.defaults[1] is outer.kwdefaults["keyword"]
    assert outer.defaults[0] is outer.closure[0].contents is inner

    portable = make_portable_args((callback, callback, leaf))
    assert portable is not None and portable[0] is portable[1]
    cursor = portable[0]
    records = {id(cursor)}
    for _ in range(depth):
        child = cursor.defaults[0]
        assert child is cursor.defaults[1] is cursor.kwdefaults["keyword"]
        assert child is cursor.closure[0].contents
        records.add(id(child))
        cursor = child
    assert cursor is portable[2] and len(records) == depth + 1
    operands = list(kernel_args._input_tree_leaves(portable))
    assert len(operands) == 1 and operands[0] is tensor

    transported = pickle.loads(pickle.dumps(portable))
    rebuilt = resolve_portable_args(transported)
    assert rebuilt[0] is rebuilt[1]
    assert_close(callback, transported[0], atol=0, rtol=0)
    assert_close(callback, rebuilt[0], atol=0, rtol=0)
    private = kernel_args._clone_tensor_leaves(rebuilt)
    assert private[0] is private[1]
    for args in (rebuilt, private):
        cursor = args[0]
        for _ in range(depth):
            child = cursor()
            assert child is cursor.__defaults__[0] is cursor.__defaults__[1]
            assert child is cursor.__kwdefaults__["keyword"]
            cursor = child
        assert cursor is args[2]
        torch.testing.assert_close(cursor(), torch.arange(3, dtype=torch.float32) + 1)
        assert len(list(kernel_args._input_tree_leaves(args))) == 1
    assert rebuilt[2]() is not private[2]()
    torch.testing.assert_close(tensor, torch.arange(3, dtype=torch.float32))


@pytest.mark.parametrize("boundary", ["resolve", "pickle", "worker"])
def test_shared_cells_cross_portable_and_private_worker_boundaries(boundary, tmp_path):
    x = torch.arange(4)
    state = 0

    def read():
        return x, state

    def update():
        nonlocal state
        state += 1
        return state

    def default_update(callback=update):
        return callback()

    unrelated = _independent_counter()
    assert _cell(read, "state") is _cell(update, "state")
    assert _cell(read, "state") is not _cell(unrelated, "state")
    assert _cell(read, "state").cell_contents is _cell(unrelated, "state").cell_contents
    portable = make_portable_args((x, read, update, default_update, unrelated))
    assert portable is not None
    path = str(tmp_path / "args.pt")
    try:
        if boundary == "worker":
            torch.save(portable, path)
            rebuilt = kernel_args.load_trusted_kernel_args(path)
        else:
            if boundary == "pickle":
                portable = pickle.loads(pickle.dumps(portable))
            rebuilt = resolve_portable_args(portable)
        tensor, reader, updater, default_updater, independent = rebuilt
        assert tensor is reader()[0]
        assert _cell(reader, "state") is _cell(updater, "state")
        assert _cell(reader, "state") is not _cell(independent, "state")
        assert updater() == 1
        assert reader()[1] == 1
        assert default_updater() == 2
        assert reader()[1] == 2
        assert read()[1] == 0 and _cell(unrelated, "state").cell_contents == 0

        if boundary == "worker":
            first = kernel_args.load_private_kernel_args(path, 0)
            second = kernel_args.load_private_kernel_args(path, 1)
        else:
            first = _clone_args(rebuilt, None, idx_to_clone=(0,))
            second = _clone_args(rebuilt, None, idx_to_clone=(0,))
        assert first[0] is first[1]()[0]
        assert second[0] is second[1]()[0]
        assert first[0].data_ptr() != tensor.data_ptr()
        assert first[0].data_ptr() != second[0].data_ptr()
        assert first[2]() == 3
        assert first[1]()[1] == 3
        assert first[3]() == 4
        assert first[1]()[1] == 4
        assert second[1]()[1] == reader()[1] == 2
        assert first[4] is independent
        assert _cell(independent, "state").cell_contents == 0
        first[0].add_(5)
        torch.testing.assert_close(tensor, torch.arange(4))
        torch.testing.assert_close(second[0], torch.arange(4))
        torch.testing.assert_close(x, torch.arange(4))
    finally:
        kernel_args.load_private_kernel_args.cache_clear()
        kernel_args.load_trusted_kernel_args.cache_clear()


def test_shared_cells_preserve_picklable_container_cycles_in_worker_copies(tmp_path):
    x = torch.arange(3)
    state = [x]
    state.append(state)

    def read():
        return state

    def replace():
        nonlocal state
        state = [state[0] + 1]
        state.append(state)

    portable = make_portable_args((x, read, replace))
    assert portable is not None
    path = str(tmp_path / "cyclic.pt")
    try:
        torch.save(portable, path)
        original = kernel_args.load_trusted_kernel_args(path)
        first = kernel_args.load_private_kernel_args(path, 0)
        second = kernel_args.load_private_kernel_args(path, 1)
        assert first[1]()[0] is first[0]
        assert first[1]()[1] is first[1]()
        assert _cell(first[1], "state") is _cell(first[2], "state")
        first[2]()
        assert first[1]()[1] is first[1]()
        torch.testing.assert_close(first[1]()[0], torch.arange(3) + 1)
        for untouched in (original, second):
            assert untouched[1]()[1] is untouched[1]()
            torch.testing.assert_close(untouched[1]()[0], torch.arange(3))
        torch.testing.assert_close(x, torch.arange(3))
    finally:
        kernel_args.load_private_kernel_args.cache_clear()
        kernel_args.load_trusted_kernel_args.cache_clear()


@pytest.mark.parametrize("cyclic_container", [False, True])
@pytest.mark.parametrize("same_inner_function", [False, True])
def test_nested_callback_accuracy_preserves_shared_capture_occurrences(
    cyclic_container, same_inner_function
):
    tensor = torch.arange(3, dtype=torch.float32)
    capture = [tensor] if cyclic_container else tensor
    if cyclic_container:
        capture.append(capture)

    def read_a():
        return capture

    def read_b():
        return capture

    if same_inner_function:
        read_b = read_a

    def callback():
        return read_a(), read_b()

    assert _cell(read_a, "capture") is _cell(read_b, "capture")
    portable_args = make_portable_args((callback,))
    assert portable_args is not None
    (portable,) = portable_args
    assert_close(callback, portable, atol=0, rtol=0)
    assert relative_l2_disagreement(callback, portable, 0.0) is None

    (transported,) = pickle.loads(pickle.dumps(portable_args))
    (rebuilt,) = resolve_portable_args((transported,))
    inner_a = _cell(rebuilt, "read_a").cell_contents
    inner_b = _cell(rebuilt, "read_b").cell_contents
    assert (inner_a is inner_b) == same_inner_function
    assert _cell(inner_a, "capture") is _cell(inner_b, "capture")
    assert_close(callback, transported, atol=0, rtol=0)
    assert_close(callback, rebuilt, atol=0, rtol=0)
    rebuilt_capture = _cell(inner_a, "capture").cell_contents
    if cyclic_container:
        assert rebuilt_capture[1] is rebuilt_capture
        rebuilt_capture = rebuilt_capture[0]
    rebuilt_capture.add_(1)
    for changed in (transported, rebuilt):
        with pytest.raises(AssertionError, match="Tensor-likes are not"):
            assert_close(callback, changed, atol=0, rtol=0)
        assert relative_l2_disagreement(callback, changed, 0.0) is not None
    torch.testing.assert_close(tensor, torch.arange(3, dtype=torch.float32))


def test_legacy_portable_closure_value_payload_remains_readable():
    x = torch.arange(3)

    def callback(scale=2, *, shift=3):
        return x * scale + shift

    legacy = PortableFunction(
        module=callback.__module__,
        qualname=callback.__qualname__,
        name=callback.__name__,
        code=marshal.dumps(callback.__code__),
        defaults=callback.__defaults__,
        kwdefaults=callback.__kwdefaults__,
        closure=(x,),
    )
    (rebuilt,) = resolve_portable_args(pickle.loads(pickle.dumps((legacy,))))
    torch.testing.assert_close(rebuilt(), x * 2 + 3)
    assert rebuilt.__qualname__ == callback.__qualname__


@pytest.mark.parametrize("kind", ["recursive", "empty"])
def test_unsupported_closure_graphs_keep_in_process_fallback(kind):
    if kind == "recursive":

        def callback(value):
            return callback(value - 1) if value else 0

    else:
        value = 1

        def callback():
            return value

        del value
    assert make_portable_args((callback,)) is None


def _module_default_callback(value=None):
    return value


@dataclasses.dataclass
class _CallbackRecord:
    callback: object
    metadata: object


@pytest.mark.parametrize("kind", ["list", "tuple", "dict", "record"])
def test_captured_container_callbacks_preserve_identity_and_parent_defaults(
    kind, monkeypatch, tmp_path
):
    tensor = torch.arange(3, dtype=torch.float32)
    monkeypatch.setattr(_module_default_callback, "__defaults__", (tensor,))
    leaf = _module_default_callback
    metadata = object()
    if kind == "list":
        container = [leaf]
    elif kind == "tuple":
        container = (leaf,)
    elif kind == "dict":
        container = {"callback": leaf}
    else:
        container = _CallbackRecord(leaf, metadata)

    def callback(first=leaf, shared=container, *, keyword=container):
        return first, container, shared, keyword

    args = (callback, leaf, container)
    portable = make_portable_args(args)
    assert portable is not None
    assert_close(args, portable, atol=0, rtol=0)
    if kind == "record":
        assert portable[2].metadata is metadata
    path = str(tmp_path / "mixed.pt")
    torch.save(portable, path)
    payload = pickle.dumps(portable)
    # A worker may have different module-level defaults. Captured references
    # must use the serialized parent's state, not pickle's function-by-name.
    monkeypatch.setattr(
        _module_default_callback, "__defaults__", (torch.full((3,), 99.0),)
    )
    try:
        rebuilt = resolve_portable_args(pickle.loads(payload))
        trusted = kernel_args.load_trusted_kernel_args(path)
        private = kernel_args.load_private_kernel_args(path, 0)
        for actual in (rebuilt, trusted, private):
            outer, inner, shared = actual
            direct, captured, default, keyword = outer()
            assert captured is default is keyword is shared
            nested = (
                shared.callback
                if kind == "record"
                else shared["callback"]
                if kind == "dict"
                else shared[0]
            )
            assert direct is inner is nested
            torch.testing.assert_close(inner(), tensor)
            assert len(list(kernel_args._input_tree_leaves(outer))) == 1 + (
                kind == "record"
            )
        private[1]().add_(5)
        torch.testing.assert_close(trusted[1](), tensor)
        torch.testing.assert_close(rebuilt[1](), tensor)
        torch.testing.assert_close(tensor, torch.arange(3, dtype=torch.float32))
    finally:
        kernel_args.load_private_kernel_args.cache_clear()
        kernel_args.load_trusted_kernel_args.cache_clear()


@pytest.mark.parametrize("kind", ["list", "dict", "tuple_list"])
def test_captured_callback_container_cycles_keep_shared_topology(kind):
    tensor = torch.arange(3)

    def leaf():
        return tensor

    if kind == "list":
        container = [leaf]
        container.append(container)
    elif kind == "dict":
        container = {"callback": leaf}
        container["cycle"] = container
    else:
        link = []
        container = (leaf, link)
        link.append(container)

    def callback():
        return leaf, container

    portable = make_portable_args((container, callback, leaf))
    assert portable is not None
    rebuilt = resolve_portable_args(pickle.loads(pickle.dumps(portable)))
    for actual in (rebuilt, kernel_args._clone_tensor_leaves(rebuilt)):
        shared, outer, inner = actual
        assert outer()[0] is inner and outer()[1] is shared
        if kind == "list":
            assert shared[0] is inner and shared[1] is shared
        elif kind == "dict":
            assert shared["callback"] is inner and shared["cycle"] is shared
        else:
            assert shared[0] is inner and shared[1][0] is shared
        torch.testing.assert_close(inner(), tensor)


@pytest.mark.parametrize("container_first", [False, True])
def test_recursive_callback_through_container_keeps_fallback(container_first):
    container = []

    def callback():
        return container

    container.append(callback)
    args = (container, callback) if container_first else (callback, container)
    assert make_portable_args(args) is None


def test_callback_conversion_reuses_registered_flatten_and_metadata():
    @dataclasses.dataclass
    class Registered:
        callback: object
        metadata: object

    calls = []

    def flatten(value):
        calls.append(value)
        return [value.callback], value.metadata

    register_pytree_node(
        Registered, flatten, lambda values, metadata: Registered(values[0], metadata)
    )
    tensor = torch.arange(2)
    metadata = object()

    def leaf():
        return tensor

    shared = Registered(leaf, metadata)

    def outer():
        return shared

    portable = make_portable_args((outer, shared, shared, leaf))
    assert portable is not None
    assert calls == [shared]
    assert portable[1] is portable[2]
    assert portable[1].metadata is metadata
    calls.clear()
    rebuilt = resolve_portable_args(portable)
    assert calls == [portable[1]]
    assert rebuilt[0]() is rebuilt[1] is rebuilt[2]
    assert rebuilt[1].callback is rebuilt[3]
    assert rebuilt[1].metadata is metadata
    torch.testing.assert_close(rebuilt[3](), tensor)
