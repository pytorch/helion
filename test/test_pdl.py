from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import guard_or_true

import helion
from helion._compiler import compile_environment
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.distributed_kernels import fake_group as fake_group
from helion._testing.triton_codegen import cpu_codegen as cpu_codegen
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_code_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.prefetch_code()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_exit_trigger(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_after_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.pdl_wait()
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_on_device(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.pdl_wait()
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_under_if(x: torch.Tensor, wait: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    if wait:
        hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_in_host_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for _ in range(1):
        hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_between_roots(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        # y[0] keeps the roots apart (no task of the second continues one of
        # the first).
        out[tile] = y[tile] * 2 + y[0]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_between_loops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2
    return y, out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_two_triggers_between_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_between_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = y[tile] * 2
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_code_between_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.prefetch_code()
    for tile in hl.tile(x.size(0)):
        y[tile] = y[tile] * 2
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_repointed_in_branch(
    w: torch.Tensor, x: torch.Tensor, z: torch.Tensor, flag: bool
) -> torch.Tensor:
    if flag:
        w.set_(x)
    else:
        w.set_(z)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_written_repointed_in_branch(
    w: torch.Tensor, y: torch.Tensor, z: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    if flag:
        y.set_(w)
    else:
        y.set_(z)
    for tile in hl.tile(y.size(0)):
        y[tile] = w[tile] + 1
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_repointed_in_comprehension(
    x: torch.Tensor, y: torch.Tensor, flag: bool
) -> torch.Tensor:
    v = x.view(-1)
    [x.set_(y) for _ in (0,) if flag]
    hl.pdl_wait(independent=x)
    for tile in hl.tile(v.size(0)):
        v[tile] = y[tile] + 1
    return v


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_repointed_in_bool_op(
    x: torch.Tensor, y: torch.Tensor, flag: bool
) -> torch.Tensor:
    v = x.view(-1)
    flag and x.set_(y) is not None
    hl.pdl_wait(independent=x)
    for tile in hl.tile(v.size(0)):
        v[tile] = y[tile] + 1
    return v


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_list_then_mutated(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    named = [x]
    hl.pdl_wait(independent=named)
    named[0] = y
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_restrided_view_write(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    # Offset 0 of the storage: x's memory when both are views of one tensor.
    z = y.as_strided(x.size(), x.stride(), 0)
    for tile in hl.tile(x.size(0)):
        z[tile] = x[tile] + 1
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_restrided_write(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    y.as_strided_(x.size(), x.stride(), 0)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    return y


def _restrided_like(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.as_strided(like.size(), like.stride(), 0)


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_restrided_in_helper(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    z = _restrided_like(y, x)
    for tile in hl.tile(x.size(0)):
        z[tile] = x[tile] + 1
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_restrided_by_op(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    z = torch.ops.aten.as_strided.default(y, x.size(), x.stride(), 0)
    for tile in hl.tile(x.size(0)):
        z[tile] = x[tile] + 1
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_restrided_by_keyword(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    z = torch.as_strided(input=y, size=x.size(), stride=x.stride(), storage_offset=0)
    for tile in hl.tile(x.size(0)):
        z[tile] = x[tile] + 1
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_set_to_computed(x: torch.Tensor) -> torch.Tensor:
    # Host code computes x's new memory before the launch.
    x.set_(x * 2)
    hl.pdl_wait(independent=x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_repointed_in_chained_compare(
    x: torch.Tensor, y: torch.Tensor, flag: int
) -> torch.Tensor:
    v = x.view(-1)
    _ = 0 < flag < x.set_(y).dim() + 2
    hl.pdl_wait(independent=x)
    for tile in hl.tile(v.size(0)):
        v[tile] = y[tile] + 1
    return v


def _grown(y: torch.Tensor, big: torch.Tensor) -> torch.Tensor:
    v = y.view(-1)
    v.resize_as_(big)
    return v


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_resized_in_helper(
    x: torch.Tensor, y: torch.Tensor, big: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    # The helper grows a view of y, which it is not given, over x.
    v = _grown(y, big)
    for tile in hl.tile(v.size(0)):
        v[tile] = big[tile] + 1
    return v


def _grown_by_out(y: torch.Tensor, big: torch.Tensor) -> torch.Tensor:
    v = y.view(-1)
    torch.neg(big, out=v)
    return v


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_resized_by_out_in_helper(
    x: torch.Tensor, y: torch.Tensor, big: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    # The out= resize happens below the op, on a view the call is not given.
    v = _grown_by_out(y, big)
    for tile in hl.tile(v.size(0)):
        v[tile] = big[tile] + 1
    return v


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_view_of_moved(
    w: torch.Tensor, z: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    y = w.view(-1)
    if flag:
        y.set_(z)
    # Taken after the move, v is w's memory where it does not happen.
    v = y.view(-1)
    for tile in hl.tile(v.size(0)):
        v[tile] = 1.0
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_written_after_move(
    w: torch.Tensor, z: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    y = w.view(-1)
    if flag:
        y.set_(z)
    y.add_(1)
    out = torch.empty_like(w)
    for tile in hl.tile(w.size(0)):
        out[tile] = w[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_resized_as(
    x: torch.Tensor, y: torch.Tensor, big: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    # In place over the storage both share: y then covers x.
    y.resize_as_(big)
    for tile in hl.tile(y.size(0)):
        y[tile] = big[tile] + 1
    return y


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_resized_by_out(
    x: torch.Tensor, y: torch.Tensor, big: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    torch.neg(big, out=y)
    for tile in hl.tile(y.size(0)):
        y[tile] = big[tile] + 1
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_filtered(x: torch.Tensor, count: int) -> torch.Tensor:
    hl.pdl_wait(independent=[x for _ in (0,) if count > 0])
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_filtered_unpacked(x: torch.Tensor, count: int) -> torch.Tensor:
    hl.pdl_wait(*[x for _ in (0,) if count > 0])
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_temporary(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=(out,))
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_after_host_loop(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for _ in range(2):
        pass
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] * 2 + y[0]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_written(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.pdl_wait(independent=(x, y))
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    return y


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_host_write(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    x.view(-1).add_(1)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_host_augassign(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    y = x.view(-1)
    y += 1
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_host_write_repointed(
    x: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    # The write lands in x's memory before set_ points the view elsewhere.
    y = x.view(-1)
    y.zero_()
    y.set_(z)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_list_augassign(xs: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty_like(xs[0])
    hl.pdl_wait(independent=xs[0])
    xs[0] += 1
    for tile in hl.tile(out.size(0)):
        out[tile] = xs[0][tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_list_alias_augassign(
    xs: list[torch.Tensor],
) -> torch.Tensor:
    out = torch.empty_like(xs[0])
    a = xs[0]
    hl.pdl_wait(independent=a)
    xs[0] += 1
    for tile in hl.tile(out.size(0)):
        out[tile] = a[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_host_join_write(
    x: torch.Tensor, y: torch.Tensor, flag: bool
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    if flag:
        target = x
    else:
        target = y
    target.add_(1)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_host_setitem(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    x[0] = 3.0
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_view(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x.T)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_local_view(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    columns = x.T
    hl.pdl_wait(independent=columns)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_twice(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_snippet_write(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
        hl.inline_triton(
            "tl.store({0} + {1}, {2})",
            args=(x, tile.index, out[tile]),
            output_like=None,
        )
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_stack_write(
    x: torch.Tensor, dev_ptrs: torch.Tensor, example: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    hl.pdl_wait(independent=x)
    for i in hl.grid(x.size(0)):
        out[i] = x[i] + 1
        tensors = hl.stacktensor_like(example, dev_ptrs[:])
        tensors[i] = x[None, i]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_beside(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.pdl_wait(x)
    y.zero_()
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    return y


_wait = hl.pdl_wait


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_by_alias(x: torch.Tensor) -> torch.Tensor:
    _wait(independent=x)
    x.add_(1)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_by_local(x: torch.Tensor) -> torch.Tensor:
    wait = hl.pdl_wait
    wait(independent=x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_joined(
    x: torch.Tensor, y: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    if flag:
        target = x
    else:
        target = y
    for tile in hl.tile(x.size(0)):
        target[tile] = x[tile] + 1
    return target


def _same(t: torch.Tensor) -> torch.Tensor:
    return t


def _half(n: int) -> int:
    return n // 2


def _move_and_back(w: torch.Tensor, v: torch.Tensor, flag: bool) -> None:
    # Through ``.data``, which dispatches no op: no capture sees the moves.
    original = w.view(w.shape)
    w.data = v
    if guard_or_true(flag):
        w.data = original


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_moved_and_back(
    w: torch.Tensor, x: torch.Tensor, flag: bool
) -> torch.Tensor:
    # The helper's move and its undo leave w's storage as it was.
    _move_and_back(w, x, flag)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_view_moved_and_back(
    w: torch.Tensor, x: torch.Tensor, flag: bool
) -> torch.Tensor:
    # Through the view's base, the helper may move w itself.
    _view_base_moved_and_back(w.view(-1), x, flag)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_joined_moved_and_back(
    w: torch.Tensor, z: torch.Tensor, x: torch.Tensor, select: bool, flag: bool
) -> torch.Tensor:
    # The helper may move w itself, whichever tensor the join is.
    target = w if select else z
    _move_and_back(target, x, flag)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_moved_view_moved_and_back(
    w: torch.Tensor, z: torch.Tensor, x: torch.Tensor, flag: bool
) -> torch.Tensor:
    # Moved off w, the view still has w as its base.
    v = w.view(-1)
    v.set_(z)
    _view_base_moved_and_back(v, x, flag)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


def _view_base_moved_and_back(v: torch.Tensor, x: torch.Tensor, flag: bool) -> None:
    base = v._base
    assert base is not None
    _move_and_back(base, x, flag)


def _repoint(t: torch.Tensor, z: torch.Tensor) -> None:
    t.data = z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_written_then_repointed(
    x: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    y = x.view(-1)
    for tile in hl.tile(y.size(0)):
        y[tile] = y[tile] + 1
    # Runs after the launch, which wrote x through y.
    _repoint(y, z)
    return z


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_repointed_after_launch(
    x: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    alias = x.view(-1)
    for tile in hl.tile(alias.size(0)):
        alias[tile] = alias[tile] + 1
    # Runs after the launch, which wrote x's memory through the view.
    x.set_(z)
    return z


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_set_only(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=x)
    # set_ moves the view, writing no element of x.
    y = x.view(-1)
    y.set_(z)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_moved_before_launch(
    x: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    y = x.view(-1)
    # Before the launch, in a helper that might skip it.
    _repoint(y, z)
    for tile in hl.tile(y.size(0)):
        y[tile] = x[tile] + 1
    return y


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_set_to_own_join(
    w: torch.Tensor, z: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    if flag:
        target = w
    else:
        target = z
    # w launches as w or z, which the kernel writes; moving w drops what the
    # join may be.
    w.set_(target)
    for tile in hl.tile(w.size(0)):
        z[tile] = w[tile] + 1
    return z


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_join_copy(x: torch.Tensor, flag: bool) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    if flag:
        selected = x.T
    else:
        selected = x
    # A contiguous x's contiguous() is x itself.
    target = selected.contiguous()
    for tile in hl.tile(target.size(0)):
        target[tile, :] = target[tile, :] + 1
    return target


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_join_of_copy(
    x: torch.Tensor, y: torch.Tensor, flag: bool, flag2: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    selected = x.T if flag else x
    copy = selected.contiguous()
    target = copy if flag2 else y
    for tile in hl.tile(target.size(0)):
        target[tile, :] = target[tile, :] + 1
    return target


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_set_onto_copy(
    w: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # x.contiguous() is x itself for a contiguous x: w may alias x.
    w.set_(x.contiguous())
    hl.pdl_wait(independent=w)
    out = torch.empty_like(w)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = w[tile, :] + 1
        x[tile, :] = out[tile, :]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def pdl_wait_independent_dynamic_reshape(x: torch.Tensor) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    # A view of a contiguous x, a copy of another: either way unknown here.
    flat = x.reshape(-1)
    for tile in hl.tile(flat.size(0)):
        flat[tile] = flat[tile] + 1
    return flat


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_after_host_only_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for _ in range(2):
        pass
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_starred(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(*(w,))
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * w[tile]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_after_host_helper(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros_like(x)
    hl.pdl_wait(independent=w)
    # An opaque host call after the allocation leaves ``out`` its own memory.
    half = _half(x.size(0))
    for tile in hl.tile(half):
        out[tile] = x[tile] * w[tile]
    return out


def _shifted(t: torch.Tensor) -> torch.Tensor:
    return torch.cond(t.sum() > 0, lambda u: u + 1, lambda u: u - 1, (t,))


def _inc(u: torch.Tensor) -> torch.Tensor:
    return u + 1


def _dec(u: torch.Tensor) -> torch.Tensor:
    return u - 1


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_beside_host_cond(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    y = torch.cond(x.sum() > 0, _inc, _dec, (x,))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] * w[tile]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_helper_cond(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    hl.pdl_wait(independent=w)
    # The helper's condition is memory its opaque call made: unknown here.
    y = _shifted(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] * w[tile]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_host_cond(x: torch.Tensor) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    y = _shifted(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_opaque_join(
    x: torch.Tensor, y: torch.Tensor, flag: bool
) -> torch.Tensor:
    hl.pdl_wait(independent=x)
    if flag:
        target = x
    else:
        target = y
    target = _same(target)
    for tile in hl.tile(x.size(0)):
        target[tile] = x[tile] + 1
    return target


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_independent_list(xs: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty_like(xs[0])
    hl.pdl_wait(independent=xs)
    for tile in hl.tile(out.size(0)):
        out[tile] = xs[0][tile] + xs[1][tile]
    return out


def _kernel_statements(code: str, kernel: str) -> list[str]:
    body = code[code.index("def _helion_") : code.index(f"\ndef {kernel}(")]
    return [
        line.strip()
        for line in body.splitlines()[1:]
        if line.strip() and not line.strip().startswith("#")
    ]


@helion.kernel(autotune_effort="none", static_shapes=True)
def pdl_independent_peer_write_kernel(
    symm: torch.Tensor, x: torch.Tensor, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    peers = torch.ops.symm_mem.get_remote_tensors(symm, group_name)
    out = torch.empty_like(x)
    hl.pdl_wait(independent=symm)
    for tile in hl.tile(x.size(0)):
        out[tile] = symm[tile] + x[tile]
        peers[1][tile] = x[tile]
    return out


def _move_onto(w: torch.Tensor, y: torch.Tensor) -> None:
    w.data = y


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_moved_between_grid_loops(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait(independent=y)
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] + 1
    # Host code between grid loops runs before the launch: w launches as y.
    for _ in range(1):
        _move_onto(w, y)
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        w[tile] = x[tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def pdl_wait_independent_overlap_moved_and_back(
    w: torch.Tensor, v: torch.Tensor, x: torch.Tensor, flag: bool
) -> torch.Tensor:
    # The helper is given only v, which overlaps w at run time.
    _move_and_back(v, x, flag)
    hl.pdl_wait(independent=w)
    for tile in hl.tile(x.size(0)):
        x[tile] = w[tile] + 1
    return x


@helion.kernel(autotune_effort="none", static_shapes=True)
def pdl_independent_unresolved_peer_write_kernel(
    symm: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    flag: bool,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    # The views of a joined tensor have no resolved owner; rank 0's own view
    # is symm itself when flag is set.
    target = symm if flag else z
    peers = torch.ops.symm_mem.get_remote_tensors(target, group_name)
    out = torch.empty_like(x)
    hl.pdl_wait(independent=symm)
    for tile in hl.tile(x.size(0)):
        out[tile] = symm[tile] + x[tile]
        peers[0][tile] = x[tile]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def pdl_independent_joined_peer_write_kernel(
    symm: torch.Tensor, x: torch.Tensor, flag: bool, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    # The op directly: a helper would be an opaque host call, after which any
    # write to a joined target is rejected without naming its memory.
    peers = torch.ops.symm_mem.get_remote_tensors(symm, group_name)
    out = torch.empty_like(x)
    hl.pdl_wait(independent=symm)
    # A host branch may pick another rank's view of symm as the target.
    if flag:
        target = peers[1]
    else:
        target = x
    for tile in hl.tile(x.size(0)):
        out[tile] = symm[tile] + x[tile]
        target[tile] = out[tile]
    return out


_X = torch.arange(256, dtype=torch.float32)
_OVERLAPPED = torch.zeros(512)
_HALVES = tuple(torch.zeros(512).split(256))
_NOT_ARGUMENT = "not a tensor argument"
_HOST = "the kernel's host code may write x"
_KERNEL_WRITES = "the kernel may write x"


# (kernel, arguments, the reason the error gives)
_REJECTIONS: list[tuple[helion.Kernel, tuple[object, ...], str]] = [
    (pdl_wait_independent_temporary, (_X,), _NOT_ARGUMENT),
    (
        pdl_wait_independent_moved_between_grid_loops,
        (_X, torch.zeros(256), torch.zeros(256)),
        "the kernel may write y",
    ),
    # A move in either arm of a host branch: w may launch as x, or y as w.
    (
        pdl_wait_independent_repointed_in_branch,
        (_X.clone(), _X.clone(), _X.clone(), False),
        "the kernel may write w",
    ),
    (
        pdl_wait_independent_written_repointed_in_branch,
        (_X.clone(), _X.clone(), _X.clone(), False),
        "the kernel may write w",
    ),
    # A move in a filtered comprehension or a boolean operation may not run
    # either. (Type propagation rejects ``match``.)
    # (``set_`` with a tensor reaches the capture in its storage form, a
    # re-stride, so the moved x is unknown memory.)
    (
        pdl_wait_independent_repointed_in_comprehension,
        (_X.clone(), _X.clone(), False),
        "host code left unknown which memory x is",
    ),
    (
        pdl_wait_independent_repointed_in_bool_op,
        (_X.clone(), _X.clone(), False),
        "the kernel may write x",
    ),
    # A view host code re-strides may reach past its argument: the two halves
    # of one tensor, the second written at the storage's start.
    (
        pdl_wait_independent_restrided_view_write,
        _HALVES,
        "host code left unknown which memory the kernel writes",
    ),
    (
        pdl_wait_independent_restrided_write,
        _HALVES,
        "host code left unknown which memory the kernel writes",
    ),
    # Also in a host helper, through the op, or by keyword.
    *(
        (kernel, _HALVES, "host code left unknown which memory the kernel writes")
        for kernel in (
            pdl_wait_independent_restrided_in_helper,
            pdl_wait_independent_restrided_by_op,
            pdl_wait_independent_restrided_by_keyword,
        )
    ),
    # Any move before the launch may not happen: in a helper, or in a later
    # comparison of a chain.
    (
        pdl_wait_independent_moved_before_launch,
        (_X.clone(), torch.empty_like(_X)),
        "the kernel may write x",
    ),
    (
        pdl_wait_independent_repointed_in_chained_compare,
        (_X.clone(), _X.clone(), 0),
        "the kernel may write x",
    ),
    # Host code computes the memory a named tensor moves onto.
    (pdl_wait_independent_set_to_computed, (_X.clone(),), _HOST),
    # In-place resizes may reach past the argument as well, also inside a
    # helper on a tensor the call is not given.
    *(
        (kernel, (*_HALVES, torch.zeros(512)), "host code left unknown which memory")
        for kernel in (
            pdl_wait_independent_resized_as,
            pdl_wait_independent_resized_by_out,
            pdl_wait_independent_resized_in_helper,
            pdl_wait_independent_resized_by_out_in_helper,
        )
    ),
    # A write through a tensor taken from one whose move may not happen, or
    # through that tensor itself, may reach the memory it was moved off.
    (
        pdl_wait_independent_view_of_moved,
        (_X.clone(), _X.clone(), False),
        "the kernel may write w",
    ),
    (
        pdl_wait_independent_written_after_move,
        (_X.clone(), _X.clone(), False),
        "the kernel's host code may write w",
    ),
    # An opaque helper may move w onto x and leave it there on another path.
    (
        pdl_wait_independent_moved_and_back,
        (_X.clone(), _X.clone(), False),
        "an opaque host call before the launch may move w",
    ),
    (
        pdl_wait_independent_view_moved_and_back,
        (_X.clone(), _X.clone(), False),
        "an opaque host call before the launch may move w",
    ),
    (
        pdl_wait_independent_moved_view_moved_and_back,
        (_X.clone(), _X.clone(), _X.clone(), False),
        "an opaque host call before the launch may move w",
    ),
    (
        pdl_wait_independent_joined_moved_and_back,
        (_X.clone(), _X.clone(), _X.clone(), True, False),
        "an opaque host call before the launch may move w",
    ),
    # A global alias records host writes; a local one cannot be found.
    (pdl_wait_independent_by_alias, (_X.clone(),), "host code may write x"),
    (pdl_wait_independent_by_local, (_X,), "through a global or a module"),
    # A filtered comprehension may name none of its elements, also through *.
    (pdl_wait_independent_filtered, (_X, 0), "not known exactly"),
    (pdl_wait_independent_filtered_unpacked, (_X, 0), "not known exactly"),
    (pdl_wait_independent_written, (_X, torch.empty_like(_X)), "may write y"),
    # An in-place op, augmented assignment or subscript store of the host
    # code (through a view too, or a target a host branch picks): it runs
    # as the kernel's predecessor on the stream.
    (pdl_wait_independent_host_write, (_X,), _HOST),
    (pdl_wait_independent_host_augassign, (_X,), _HOST),
    # Also when a later set_ points the written view elsewhere, and through
    # an augmented assignment of a list's tensor.
    (pdl_wait_independent_host_write_repointed, (_X, torch.empty_like(_X)), _HOST),
    (
        pdl_wait_independent_list_augassign,
        ([_X.clone()],),
        r"host code may write xs\[0\]",
    ),
    (
        pdl_wait_independent_list_alias_augassign,
        ([_X.clone()],),
        r"host code may write xs\[0\]",
    ),
    # A kernel write whose target host code moves elsewhere after the
    # launch (the view, or the independent tensor itself).
    (
        pdl_wait_independent_written_then_repointed,
        (_X.clone(), torch.empty_like(_X)),
        _KERNEL_WRITES,
    ),
    (
        pdl_wait_independent_repointed_after_launch,
        (_X.clone(), torch.empty_like(_X)),
        _KERNEL_WRITES,
    ),
    (
        pdl_wait_independent_host_join_write,
        (_X, torch.empty_like(_X), False),
        _HOST,
    ),
    (pdl_wait_independent_host_setitem, (_X,), _HOST),
    # A view of an argument is no argument, also bound to a local.
    (pdl_wait_independent_view, (_X.view(16, 16),), _NOT_ARGUMENT),
    (pdl_wait_independent_local_view, (_X.view(16, 16),), _NOT_ARGUMENT),
    (pdl_wait_independent_twice, (_X,), "one hl.pdl_wait"),
    # Writes the kernel cannot bound: a snippet given ``x``, a store through
    # device pointers.
    (pdl_wait_independent_snippet_write, (_X,), _KERNEL_WRITES),
    (
        pdl_wait_independent_stack_write,
        (_X, torch.zeros([2], dtype=torch.uint64), _X),
        "device pointers",
    ),
    # A host branch may pick it as the target, also behind an opaque host
    # call.
    (
        pdl_wait_independent_joined,
        (_X, torch.empty_like(_X), False),
        _KERNEL_WRITES,
    ),
    (
        pdl_wait_independent_opaque_join,
        (_X, torch.empty_like(_X), False),
        "host code left unknown which memory the kernel writes",
    ),
    # It may launch as a tensor a host branch picked (here one the kernel
    # writes), whose record a move drops.
    (
        pdl_wait_independent_set_to_own_join,
        (_X.clone(), torch.empty_like(_X), False),
        "host code left unknown which memory w is",
    ),
    # A conditional copy may return its input: its memory is unknown, also
    # as one branch of a later join, and whatever layout the first call had.
    (
        pdl_wait_independent_join_copy,
        (_X.view(16, 16), False),
        "host code left unknown which memory the kernel writes",
    ),
    (
        pdl_wait_independent_join_of_copy,
        (_X.view(16, 16), torch.empty(16, 16), False, False),
        "host code left unknown which memory the kernel writes",
    ),
    (
        pdl_wait_independent_dynamic_reshape,
        (_X.view(16, 16).T,),
        "host code left unknown which memory the kernel writes",
    ),
    (
        pdl_wait_independent_dynamic_reshape,
        (_X.view(16, 16),),
        "the kernel may write x",
    ),
    (
        pdl_wait_independent_set_onto_copy,
        (torch.empty(16, 16), _X.view(16, 16).T),
        "host code left unknown which memory w is",
    ),
    (
        pdl_wait_independent_helper_cond,
        (_X, _X.clone()),
        "host code left unknown which memory the kernel's host code writes",
    ),
    # A higher-order op may mutate what it is given.
    (pdl_wait_independent_host_cond, (_X,), _HOST),
]


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfRefEager("PDL arguments are checked at compile time")
@pytest.mark.parametrize(
    ("kernel", "args", "reason"),
    [pytest.param(*case, id=case[0].name) for case in _REJECTIONS],
)
def test_independent_rejects(
    cpu_codegen: None, kernel: helion.Kernel, args: tuple[object, ...], reason: str
) -> None:
    """``independent`` names tensor arguments (also in a list argument) the
    kernel does not and cannot write, on its only wait."""
    with pytest.raises(helion.exc.PdlIndependentArgument, match=reason):
        kernel.bind(args)


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfRefEager("PDL arguments are checked at compile time")
def test_independent_names_what_the_call_bound(cpu_codegen: None) -> None:
    """``independent`` holds a list's tensors as the call bound them: host
    code storing another tensor into the list afterwards names nothing."""
    x, y = _X.clone(), _X.clone()
    bound = pdl_wait_independent_list_then_mutated.bind((x, y))
    (named,) = bound.env.pdl_independent
    assert named is bound.fake_args[0]


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfRefEager("PDL arguments are checked at compile time")
def test_independent_rejects_peer_writes(cpu_codegen: None, fake_group: str) -> None:
    """A write through another rank's view of a tensor
    ``hl.pdl_wait(independent=...)`` names writes that tensor's allocation,
    also when a host branch picks that view as the target."""
    symm = torch.zeros(256)
    x = torch.randn(256)
    for kernel, args in (
        (pdl_independent_peer_write_kernel, (symm, x, fake_group)),
        (pdl_independent_joined_peer_write_kernel, (symm, x, False, fake_group)),
    ):
        kernel.reset()
        with pytest.raises(
            helion.exc.PdlIndependentArgument, match="the kernel may write symm"
        ):
            kernel.bind(args)
    # A peer view with no resolved owner may be symm itself.
    pdl_independent_unresolved_peer_write_kernel.reset()
    with pytest.raises(
        helion.exc.PdlIndependentArgument,
        match="host code left unknown which memory the kernel writes",
    ):
        pdl_independent_unresolved_peer_write_kernel.bind(
            (symm, torch.zeros(256), x, False, fake_group)
        )


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR keys no binding by argument overlap")
@skipIfRefEager("PDL arguments are checked at compile time")
def test_independent_rejects_an_overlapping_written_argument(
    cpu_codegen: None,
) -> None:
    """A written argument, or one an opaque host call is given, that overlaps
    an independent one is its memory (the binding keys arguments by overlap);
    column slices of one tensor do not overlap."""
    with pytest.raises(helion.exc.PdlIndependentArgument, match="may write x"):
        pdl_wait_independent_beside.bind((_X, _X))
    with pytest.raises(
        helion.exc.PdlIndependentArgument,
        match="an opaque host call before the launch may move w",
    ):
        pdl_wait_independent_overlap_moved_and_back.bind(
            (_OVERLAPPED[:256], _OVERLAPPED[128:384], _X.clone(), False)
        )
    columns = torch.zeros(256, 2)
    bound = pdl_wait_independent_beside.bind((columns[:, 0], columns[:, 1]))
    assert list(bound.env.pdl_independent.values()) == ["x"]


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfRefEager("PDL arguments are checked at compile time")
def test_only_independent_captures_host_writes(cpu_codegen: None) -> None:
    """Host calls run under the host-write capture only in a kernel that
    passes ``hl.pdl_wait`` an argument, the one reader of what it records."""
    named = pdl_wait_independent_beside.bind((_X, torch.empty_like(_X)))
    assert named.env.captures_host_writes
    assert not pdl_trigger_after_host_only_loop.bind((_X,)).env.captures_host_writes


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfRefEager("PDL arguments are checked at compile time")
def test_host_cond_runs_eagerly_after_the_binding(cpu_codegen: None) -> None:
    """Binding host code with a higher-order op under the host-write capture
    leaves the op's eager calls working."""
    pdl_wait_independent_beside_host_cond.reset()
    pdl_wait_independent_beside_host_cond.bind((_X, torch.ones_like(_X)))
    torch.testing.assert_close(torch.cond(_X.sum() > 0, _inc, _dec, (_X,)), _X + 1)


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR emits no PDL ops")
@skipIfRefEager("PDL placement is checked at compile time")
def test_trigger_after_a_host_only_loop_runs_at_entry(cpu_codegen: None) -> None:
    """A host loop is no root: a trigger before the first grid loop runs at
    entry, as the pdl_launch_dependents docstring says."""
    bound = pdl_trigger_after_host_only_loop.bind((_X,))
    assert bound.env.pdl_entry == ["launch_dependents"]
    assert bound.env.pdl_exit == []


@onlyBackends(["triton"])
class TestPdl(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_entry_ops_run_in_source_order(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(pdl_add, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        self.assertIn("launch_pdl=True", code)
        statements = _kernel_statements(code, "pdl_add")
        self.assertEqual(
            statements[:2],
            [
                "tl.extra.cuda.gdc_launch_dependents()",
                "tl.extra.cuda.gdc_wait()",
            ],
        )
        self.assertEqual(code.count("gdc_"), 2)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_after_loops_runs_at_exit(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(pdl_exit_trigger, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        statements = _kernel_statements(code, "pdl_exit_trigger")
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_roots_runs_before_the_later_root(self) -> None:
        """A persistent megakernel triggers once per program before the second
        root's tasks (after its grid wait), not at exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            pdl_trigger_between_roots,
            (x,),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
        )

        torch.testing.assert_close(out, (x + 1) * 2 + 1)
        statements = _kernel_statements(code, "pdl_trigger_between_roots")
        trigger = statements.index("tl.extra.cuda.gdc_launch_dependents()")
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        loops = [
            i for i, s in enumerate(statements) if s.startswith("for virtual_pid in")
        ]
        self.assertEqual(len(loops), 2)
        self.assertTrue(loops[0] < trigger < loops[1], statements)
        self.assertEqual(code.count("gdc_launch_dependents()"), 1)
        # A host loop is not a root: the trigger still precedes the second.
        code, out = code_and_output(
            pdl_trigger_after_host_loop,
            (x,),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
        )
        torch.testing.assert_close(out, (x + 1) * 2 + 1)
        statements = _kernel_statements(code, "pdl_trigger_after_host_loop")
        trigger = statements.index("tl.extra.cuda.gdc_launch_dependents()")
        loops = [
            i for i, s in enumerate(statements) if s.startswith("for virtual_pid in")
        ]
        self.assertTrue(loops[0] < trigger < loops[1], statements)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_roots_runs_at_exit_without_a_static_schedule(
        self,
    ) -> None:
        """Packet dispatch and the barrier pipeline place no trigger between
        roots: it runs once, at exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        for pipeline in ("dynamic", "barrier"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    pdl_trigger_between_roots,
                    (x,),
                    block_sizes=[64, 64],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                )
                torch.testing.assert_close(out, (x + 1) * 2 + 1)
                statements = _kernel_statements(code, "pdl_trigger_between_roots")
                self.assertEqual(
                    statements[-1], "tl.extra.cuda.gdc_launch_dependents()"
                )
                self.assertEqual(code.count("gdc_launch_dependents()"), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_independent_loops_runs_at_exit(self) -> None:
        """Without a megakernel to place it in, a trigger between loops runs at
        exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, (y, out) = code_and_output(
            pdl_trigger_between_loops, (x,), block_sizes=[64, 64]
        )

        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(out, x * 2)
        statements = _kernel_statements(code, "pdl_trigger_between_loops")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")
        self.assertEqual(code.count("gdc_launch_dependents()"), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    def test_ops_do_nothing_before_sm90(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            hl.pdl_launch_dependents()
            hl.pdl_wait()
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        with mock.patch.object(
            compile_environment, "target_device_capability", lambda device: (8, 0)
        ):
            code, out = code_and_output(add_one, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        self.assertNotIn("gdc_", code)
        self.assertNotIn("launch_pdl", code)

    @skipIfRefEager("PDL placement is checked at compile time")
    def test_misplaced_ops_raise(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        for kernel, args in (
            (pdl_wait_after_loop, (x,)),
            (pdl_wait_on_device, (x,)),
            (pdl_wait_under_if, (x, True)),
            (pdl_wait_in_host_loop, (x,)),
            (pdl_two_triggers_between_loops, (x,)),
            # Between two loops (a megakernel's) they raise too.
            (pdl_wait_between_loops, (x,)),
            (prefetch_code_between_loops, (x,)),
        ):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaises(helion.exc.PdlPlacement),
            ):
                code_and_output(kernel, args, block_sizes=[64])

    @skipIfRefEager("PDL arguments are checked at compile time")
    def test_independent_binds_through_unpacking(self) -> None:
        """``independent`` is checked as the call binds it, also through ``*``
        unpacking."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        w = torch.full_like(x, 2.0)
        bound = pdl_wait_independent_starred.bind((x, w))
        self.assertEqual(list(bound.env.pdl_independent.values()), ["w"])
        _, out = code_and_output(pdl_wait_independent_starred, (x, w), block_sizes=[64])
        torch.testing.assert_close(out, x * w)

    @skipIfRefEager("PDL arguments are checked at compile time")
    def test_independent_takes_kernel_arguments(self) -> None:
        """``independent`` takes tensor arguments (also in a list argument) the
        kernel does not write; test_independent_rejects has the rest."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        # A set_ alone writes nothing.
        _, out = code_and_output(
            pdl_wait_independent_set_only, (x, torch.empty_like(x)), block_sizes=[64]
        )
        torch.testing.assert_close(out, x + 1)
        _, out = code_and_output(pdl_wait_independent_list, ([x, x],), block_sizes=[64])
        torch.testing.assert_close(out, 2 * x)
        # Inference tensors keep no version counter; the check does not need one.
        # (A new length binds the kernel again, under inference mode.)
        with torch.inference_mode():
            y = torch.arange(128, device=DEVICE, dtype=torch.float32)
            _, out = code_and_output(
                pdl_wait_independent_list, ([y, y],), block_sizes=[64]
            )
        torch.testing.assert_close(out, 2 * y)
        # A host higher-order op on another tensor (host code that compiled
        # before the host-write capture still compiles).
        w = torch.full_like(x, 2.0)
        _, out = code_and_output(
            pdl_wait_independent_beside_host_cond, (x, w), block_sizes=[64]
        )
        torch.testing.assert_close(out, (x + 1) * w)
        # Written memory allocated before an opaque host helper call.
        _, out = code_and_output(
            pdl_wait_independent_after_host_helper, (x, w), block_sizes=[64]
        )
        torch.testing.assert_close(out, torch.cat([x[:128] * 2, x[128:] * 0]))
        # Positional, and beside an argument with its own memory that the
        # kernel and its host code write.
        _, out = code_and_output(
            pdl_wait_independent_beside, (x, torch.empty_like(x)), block_sizes=[64]
        )
        torch.testing.assert_close(out, x + 1)


class TestPrefetchCodeBackends(RefEagerTestBase, TestCase):
    def test_prefetch_code_runs_on_every_backend(self) -> None:
        # Only Triton and Gluon on sm_90+ emit the prefetch; elsewhere the call
        # compiles to nothing.
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(prefetch_code_add, (x,), block_sizes=[64])
        torch.testing.assert_close(out, x + 1)
        if _get_backend() not in ("triton", "gluon"):
            # The source comments quote hl.prefetch_code(); check the emitted call.
            self.assertNotIn("helion_prefetch_code.prefetch_code()", code)
