from __future__ import annotations

import pytest
import sympy
import torch

import helion
from helion import exc
from helion._compiler.utils import host_subscript_geometry
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _assert_traced_geometry(bound, args, expected, *, layout=False):
    replacements = {}
    for fake, actual in zip(bound.fake_args, args, strict=True):
        if isinstance(fake, torch.Tensor):
            symbolic_values = (*fake.shape, fake.storage_offset())
            actual_values = (*actual.shape, actual.storage_offset())
            if layout:
                symbolic_values += fake.stride()
                actual_values += actual.stride()
        else:
            symbolic_values, actual_values = (fake,), (actual,)
        for symbolic, value in zip(symbolic_values, actual_values, strict=True):
            if isinstance(symbolic, torch.SymInt):
                replacements[symbolic.node.expr] = sympy.Integer(value)
    traced = bound.host_function.local_types["created"].fake_value
    metadata = tuple(traced.shape)
    expected_metadata = tuple(expected.shape)
    if layout:
        metadata += (*traced.stride(), traced.storage_offset())
        expected_metadata += (*expected.stride(), expected.storage_offset())
    for value, actual in zip(metadata, expected_metadata, strict=True):
        if isinstance(value, torch.SymInt):
            value = value.node.expr.xreplace(replacements)
        assert int(value) == actual


@helion.kernel(static_shapes=False, config=helion.Config(block_sizes=[4, 8]))
def _slice_copy(x: torch.Tensor, kind: hl.constexpr) -> torch.Tensor:
    if kind == "prefix":
        created = x[:, :4]
    elif kind == "tail":
        created = x[:, 4:]
    elif kind == "negative":
        created = x[:, -4:]
    else:
        created = x[:, 1:4:2]
    out = torch.empty_like(created)
    for rows, columns in hl.tile(created.shape):
        out[rows, columns] = created[rows, columns]
    return out


@pytest.mark.parametrize(
    "kind,slice_",
    [
        ("prefix", slice(None, 4)),
        ("tail", slice(4, None)),
        ("negative", slice(-4, None)),
        ("stepped", slice(1, 4, 2)),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_host_slice_bounds_match_runtime_geometry(kind, slice_, reverse):
    kernel = helion.kernel(
        _slice_copy.fn,
        static_shapes=False,
        config=helion.Config(block_sizes=[4, 8]),
    )
    widths = [3, 5, 0, 1, 4, 7]
    if reverse:
        widths.reverse()
    widths.append(widths[0])
    with cpu_render_target():
        for width in widths:
            x = torch.empty(4, width)
            bound = kernel.bind((x, kind))
            config = bound.config_spec.default_config()
            launches = []
            host = load_host_function(
                bound,
                config,
                lambda *args, records=launches, **kwargs: records.append(args),
            )
            out = host(x, kind)
            expected = x[:, slice_]
            assert out.shape == expected.shape
            assert launches
            # Compare the compiler's actual tile extent against the view the
            # generated wrapper produced, including reused dynamic bindings.
            _assert_traced_geometry(bound, (x, kind), expected)


@helion.kernel(static_shapes=False, config=helion.Config(block_sizes=[4, 8]))
def _bounded_copy(x: torch.Tensor, start: int, stop: int, step: int) -> torch.Tensor:
    created = x[:, start:stop:step]
    out = torch.empty_like(created)
    for rows, columns in hl.tile(created.shape):
        out[rows, columns] = created[rows, columns]
    return out


def test_host_symbolic_slice_bounds_and_steps():
    kernel = helion.kernel(
        _bounded_copy.fn,
        static_shapes=False,
        config=helion.Config(block_sizes=[4, 8]),
    )
    x = torch.empty_strided((4, 7), (32, 2))
    with cpu_render_target():
        for start, stop, step in ((1, 4, 1), (-4, 20, 2), (20, 30, 1), (-20, -2, 3)):
            args = (x, start, stop, step)
            bound = kernel.bind(args)
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            expected = x[:, start:stop:step]
            assert host(*args).shape == expected.shape
            _assert_traced_geometry(bound, args, expected, layout=True)


@helion.kernel(static_shapes=False, config=helion.Config(block_sizes=[1, 4, 8]))
def _composed_copy(x: torch.Tensor) -> torch.Tensor:
    created = x[1, None, ..., -4:]
    out = torch.empty_like(created)
    for tile in hl.tile(created.shape):
        out[tile] = created[tile]
    return out


def test_host_basic_index_rank_and_offset():
    kernel = helion.kernel(
        _composed_copy.fn,
        static_shapes=False,
        config=helion.Config(block_sizes=[1, 4, 8]),
    )
    with cpu_render_target():
        for width in (3, 7, 0, 1, 5):
            # Fixed noncanonical input strides make the view's full metadata
            # independently replayable even at size zero or one.
            x = torch.empty_strided((3, 2, width), (64, 16, 2))
            bound = kernel.bind((x,))
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            expected = x[1, None, ..., -4:]
            assert host(x).shape == expected.shape
            _assert_traced_geometry(bound, (x,), expected, layout=True)


def test_host_slice_keeps_python_index_conversion():
    class Index:
        calls = 0

        def __index__(self):
            self.calls += 1
            return 3

    x = torch.empty(7)
    index = Index()
    custom = slice(None, index)
    assert x[custom].shape == (3,)
    assert index.calls == 1
    assert host_subscript_geometry(x, custom) is None
    assert index.calls == 1
    # Python accepts and saturates an arbitrarily large literal step. The
    # metadata repair must leave that conversion to PyTorch.
    huge = slice(None, None, 10**100)
    x[huge]
    assert host_subscript_geometry(x, huge) is None


@pytest.mark.parametrize("as_list", [False, True])
def test_host_repeated_ellipsis_is_rejected_before_torch_indexing(as_list):
    @helion.kernel(static_shapes=False, config=helion.Config(block_sizes=[4, 8]))
    def copy(x: torch.Tensor, as_list: hl.constexpr) -> torch.Tensor:
        if as_list:
            index = [..., ..., slice(None)]
        else:
            index = (..., ..., slice(None))
        created = x[index]
        out = torch.empty_like(created)
        for rows, columns in hl.tile(created.shape):
            out[rows, columns] = created[rows, columns]
        return out

    with cpu_render_target():
        x = torch.empty_strided((4, 3), (32, 2))
        # Repeated ellipses can pass an invalid dimension to PyTorch's slice
        # implementation, making even FakeTensor inference order-dependent.
        with pytest.raises(exc.TypeInferenceError, match="at most one ellipsis"):
            copy.bind((x, as_list))
        index = [..., ..., slice(None)] if as_list else (..., ..., slice(None))
        assert host_subscript_geometry(x, index) is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,slice_", [("prefix", slice(None, 4)), ("tail", slice(4, None))]
)
def test_host_slice_native_crosses_clamp(kind, slice_):
    kernel = helion.kernel(
        _slice_copy.fn,
        static_shapes=False,
        config=helion.Config(block_sizes=[4, 8]),
    )
    for width in (3, 5, 0, 1, 4, 7, 3):
        x = torch.randn(4, width, device=DEVICE)
        torch.testing.assert_close(kernel(x, kind), x[:, slice_])


@pytest.mark.cuda
def test_host_symbolic_slice_native_changes_bounds_and_step():
    kernel = helion.kernel(
        _bounded_copy.fn,
        static_shapes=False,
        config=helion.Config(block_sizes=[4, 8]),
    )
    x = torch.randn(4, 7, device=DEVICE)
    for start, stop, step in (
        (1, 4, 1),
        (-4, 20, 2),
        (20, 30, 1),
        (-20, -2, 3),
        (1, 4, 1),
    ):
        torch.testing.assert_close(kernel(x, start, stop, step), x[:, start:stop:step])
