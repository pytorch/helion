"""Bound kernels keyed by argument overlap on Triton and Gluon.

Their barrier walks order accesses through tensor arguments that overlap at
runtime as accesses of one storage (``Backend.orders_overlapping_inputs``),
so the cache key partitions tensor arguments by overlap: a call passing a
view of another argument's elements gets its own binding.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._compat import gluon_api_available
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfFn
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.triton_codegen import gluon_cpu_codegen
import helion.language as hl
from helion.runtime.kernel import _input_tensor_aliases
from helion.runtime.kernel import _make_prepared_arg_guard
from helion.runtime.settings import _get_backend

pytestmark = pytest.mark.cpu_only


def _triton(test: object) -> object:
    for skip in (
        skipUnlessBackends(["triton"]),
        skipIfTileIR("TileIR runs no barrier walk"),
        skipIfRefEager("ref-eager mode generates no code"),
    ):
        test = skip(test)
    return test


def _store_then_load_argument(scratch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(scratch)
    for tile_m in hl.tile(scratch.size(0)):
        scratch[tile_m, :] = scratch[tile_m, :] * 2.0
        half = hl.arange(64)
        out[tile_m, half] = x[tile_m, half + 64]
        out[tile_m, half + 64] = x[tile_m, half]
    return out


def _call_args(
    scratch: torch.Tensor, other: torch.Tensor, aliased: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    s = scratch.clone()
    return (s, s.view_as(s)) if aliased else (s, other)


@_triton
def test_the_partition_keys_runtime_overlap() -> None:
    @dataclasses.dataclass
    class Inputs:
        first: torch.Tensor
        second: torch.Tensor

    x = torch.randn(4, 8)
    y = torch.nn.Parameter(torch.randn(4, 8))
    cases = (
        ((x, y, 1e-5, None), None),
        # Disjoint slices of one storage key as distinct tensors, row or
        # column slices alike.
        ((x[:2], x[2:]), None),
        (tuple(x.split(2, dim=-1)), None),
        ((x[:, ::2], x[:, 1::2]), None),
        ((x[:, :4].T, x[:, 4:].T), None),
        # (overlap class, tensor class) per tensor.
        ((x, x.view_as(x)), ((0, 0), (0, 1))),
        ((x, y.T, x[1:]), ((0, 0), (1, 1), (0, 2))),
        ((x, [y, x.view_as(x)]), ((0, 0), (1, 1), (0, 2))),
        # Overlap only by identity keys as the identity partition.
        ((x, 1e-5, x, y), (0, 0, 1)),
        ((Inputs(x, y), x), (0, 1, 0)),
    )
    for args, expected in cases:
        assert _input_tensor_aliases(args, by_storage=True) == expected, args

    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Dynamo keys tensor identity alone.
        assert _input_tensor_aliases((x.T, y.T), by_storage=True) is None
        assert _input_tensor_aliases((x, y, x), by_storage=True) == (0, 1, 0)
        return x + y

    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, x.view_as(x)), x + x)


@_triton
def test_the_prepared_guard_checks_overlap_relations() -> None:
    kernel = helion.kernel(_store_then_load_argument, static_shapes=True)
    x = torch.empty(64, 128)
    distinct = _make_prepared_arg_guard(kernel, (x, torch.empty(64, 128)))
    viewed = _make_prepared_arg_guard(kernel, (x, x.view_as(x)))
    same = _make_prepared_arg_guard(kernel, (x, x))

    scratch = torch.empty(64, 128)
    other = torch.empty(64, 128)
    halves = torch.empty(128, 128)
    columns = torch.empty(64, 256)
    assert distinct((scratch, other))
    assert distinct((halves[:64], halves[64:]))
    distinct_columns = _make_prepared_arg_guard(
        kernel, (torch.empty(64, 256)[:, :128], torch.empty(64, 256)[:, :128])
    )
    assert distinct_columns((columns[:, :128], columns[:, 128:]))
    # Two empty parameters at one offset overlap nothing, but one passed
    # twice keys apart.
    empty = torch.nn.Parameter(torch.empty(0, 128))
    empties = _make_prepared_arg_guard(
        kernel, (empty, torch.nn.Parameter(empty.view_as(empty)))
    )
    assert not empties((empty, empty))
    assert not distinct((scratch, scratch.view_as(scratch)))
    assert viewed((scratch, scratch.view_as(scratch)))
    assert not viewed((scratch, other))
    assert not viewed((scratch, scratch))
    assert same((scratch, scratch))
    assert not same((scratch, scratch.view_as(scratch)))
    assert not same((scratch, other))


@_triton
@pytest.mark.parametrize("static_shapes", [True, False])
def test_a_view_of_another_argument_binds_apart_and_orders_its_load(
    static_shapes: bool,
) -> None:
    # ``(scratch, scratch.view_as(scratch))`` gets a binding whose walk orders
    # the loads of ``x`` after the store to ``scratch``, as does one tensor
    # passed twice; two distinct tensors get none and share their binding
    # with two disjoint halves of one storage, row or column halves, under
    # static shapes.
    kernel = helion.kernel(_store_then_load_argument, static_shapes=static_shapes)
    scratch = torch.randn(64, 128)
    other = torch.randn(64, 128)
    halves = torch.randn(128, 128)
    columns = torch.randn(64, 256)
    config = helion.Config(block_sizes=[1], num_warps=4)
    with cpu_render_target():
        distinct = kernel.bind(_call_args(scratch, other, aliased=False))
        aliased = kernel.bind(_call_args(scratch, other, aliased=True))
        twice = kernel.bind((scratch, scratch))
        assert len({distinct, aliased, twice}) == 3
        assert "tl.debug_barrier()" not in distinct.to_code(config)
        assert "tl.debug_barrier()" in aliased.to_code(config)
        assert "tl.debug_barrier()" in twice.to_code(config)
        if static_shapes:
            assert kernel.bind((halves[:64], halves[64:])) is distinct
            assert kernel.bind((columns[:, :128], columns[:, 128:])) is kernel.bind(
                (torch.randn(64, 256)[:, :128], torch.randn(64, 256)[:, :128])
            )


def _store_then_load_with_a_constant(
    scratch: torch.Tensor, x: torch.Tensor, constant: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(scratch)
    for tile_m in hl.tile(scratch.size(0)):
        scratch[tile_m, :] = scratch[tile_m, :] * 2.0
        out[tile_m, :] = x[tile_m, :]
    return out


@_triton
def test_a_constexpr_holding_a_tensor_binds() -> None:
    # The key skips the constexpr; its fake holds the tensor unwrapped.
    kernel = helion.kernel(_store_then_load_with_a_constant, static_shapes=True)
    scratch = torch.randn(64, 128)
    constant = hl.constexpr((torch.empty(1),))
    config = helion.Config(block_sizes=[1], num_warps=4)
    with cpu_render_target():
        bound = kernel.bind((scratch, scratch.view_as(scratch), constant))
        assert "tl.debug_barrier()" in bound.to_code(config)


class _Source(NamedTuple):
    x: torch.Tensor


def _store_then_load_a_field(scratch: torch.Tensor, source: _Source) -> torch.Tensor:
    out = torch.empty_like(scratch)
    for tile_m in hl.tile(scratch.size(0)):
        scratch[tile_m, :] = scratch[tile_m, :] * 2.0
        out[tile_m, :] = source.x[tile_m, :]
    return out


@_triton
def test_a_view_in_a_named_tuple_binds() -> None:
    # A named tuple's field overlaps the first argument.
    kernel = helion.kernel(_store_then_load_a_field, static_shapes=True)
    scratch = torch.randn(64, 128)
    config = helion.Config(block_sizes=[1], num_warps=4)
    with cpu_render_target():
        bound = kernel.bind((scratch, _Source(scratch.view_as(scratch))))
        assert "tl.debug_barrier()" in bound.to_code(config)


@skipIfFn(
    lambda: _get_backend() != "triton" or not gluon_api_available(),
    "the gluon backend runs in the triton lane on Triton >= 3.8",
)
@skipIfRefEager("ref-eager mode generates no code")
def test_gluon_binds_a_view_of_another_argument_apart() -> None:
    kernel = helion.kernel(
        _store_then_load_argument, backend="gluon", static_shapes=True
    )
    scratch = torch.randn(64, 128)
    config = helion.Config(block_sizes=[1], num_warps=4)
    with gluon_cpu_codegen():
        aliased = kernel.bind(_call_args(scratch, scratch, aliased=True))
        distinct = kernel.bind(_call_args(scratch, torch.randn(64, 128), False))
        assert aliased is not distinct
        aliased_code = aliased.to_triton_code(config)
        distinct_code = distinct.to_triton_code(config)
    assert aliased_code.count("barrier()") > distinct_code.count("barrier()")


@_triton
def test_a_backend_change_keys_by_the_new_backend_after_reset() -> None:
    # A kernel built for a backend whose walks see no overlap, switched to
    # Triton: reset() drops the old bindings and keys by overlap again.
    kernel = helion.kernel(
        _store_then_load_argument, backend="metal", static_shapes=True
    )
    kernel.settings.backend = "triton"
    kernel.reset()
    scratch = torch.randn(64, 128)
    config = helion.Config(block_sizes=[1], num_warps=4)
    with cpu_render_target():
        aliased = kernel.bind(_call_args(scratch, scratch, aliased=True))
        assert aliased is not kernel.bind(_call_args(scratch, scratch, aliased=False))
        assert "tl.debug_barrier()" in aliased.to_code(config)


@_triton
def test_fake_arguments_bind_by_identity() -> None:
    # Fake tensors have no addresses: a view of another fake argument keys
    # and compiles as a distinct tensor, apart from eager views.
    kernel = helion.kernel(_store_then_load_argument, static_shapes=True)
    with FakeTensorMode():
        fake = torch.empty(64, 128)
        fake_view = fake.view_as(fake)
    assert _input_tensor_aliases((fake, fake_view), by_storage=True) is None
    scratch = torch.randn(64, 128)
    with cpu_render_target():
        assert kernel.bind((fake, fake_view)) is not kernel.bind(
            _call_args(scratch, scratch, aliased=True)
        )
        config = helion.Config(block_sizes=[1], num_warps=4)
        assert "tl.debug_barrier()" in kernel.bind((fake, fake)).to_code(config)


@_triton
@pytest.mark.cuda
def test_eager_calls_keep_the_two_bindings_apart() -> None:
    # The fast key, then the prepared call, in either order.
    kernel = helion.kernel(
        _store_then_load_argument,
        config=helion.Config(block_sizes=[1], num_warps=4),
    )
    scratch = torch.randn(64, 128, device=DEVICE)
    other = torch.randn(64, 128, device=DEVICE)
    for aliased in (False, False, True, True, False):
        args = _call_args(scratch, other, aliased)
        read = scratch * 2.0 if aliased else other
        torch.testing.assert_close(
            kernel(*args), torch.cat([read[:, 64:], read[:, :64]], dim=-1)
        )
        assert kernel._prepared_call is not None
        assert kernel._prepared_call.bound is kernel.bind(args)
    assert kernel.bind(_call_args(scratch, other, True)) is not kernel.bind(
        _call_args(scratch, other, False)
    )
