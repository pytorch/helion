"""Block sizes and extents a host helper over tunables computes, inline.

``hl.tile([n], block_size=[helion.next_power_of_2(foo)])`` passes a value no
host variable names.  The tile takes the value the helper gives under the
config being compiled, as a static block size like a literal one, on each
axis of a tile; a loop extent written the same way is that value too, and so
is the helper's value in device code (a device loop's block size or extent, a
scale).  Python's min and max over tunables on the host take the argument
every config picks, or else the config's value, instead of comparing the
tunables at their hints.  Code generation only, apart from a GPU run of
cross-loop split-K; test_register_tunable.py runs one too.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEagerMark
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessGluon
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton", "cute"]),
    # Every test binds a kernel and reads its config spec or generated code.
    skipIfRefEagerMark(
        "inspects config specs and generated code; ref mode has neither"
    ),
]


@helion.kernel(autotune_effort="none")
def scale_by_power_of_2_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[helion.next_power_of_2(foo)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_cdiv_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[helion.cdiv(foo, 4)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_power_of_2_above_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[helion.next_power_of_2(foo + 1)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_min_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[min(foo, 128)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_max_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[max(foo, 128)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_2d_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, helion.next_power_of_2(foo)]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_helper_extent(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile(helion.next_power_of_2(foo)):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_inner_tile_by_tunable(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=helion.next_power_of_2(foo)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * foo
    return out


@helion.kernel(autotune_effort="none")
def scale_by_device_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(foo + 1)
    return out


@helion.kernel(autotune_effort="none")
def inner_extent_device_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(0, helion.next_power_of_2(foo)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def branch_on_device_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        if helion.next_power_of_2(foo + 1) > 64:
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
        else:
            out[tile_m, tile_n] = x[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def mask_tile_at_tunable_index(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile(n, block_size=1):
        keep = (tile_n.id == foo - 1) & (x[tile_n] > 0)
        out[tile_n] = torch.where(keep, x[tile_n], 0.0)
    return out


@helion.kernel(autotune_effort="none")
def flip_tile_at_tunable_index(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile(n, block_size=1):
        keep = (x[tile_n] > 0) != (tile_n.id == foo - 1)
        out[tile_n] = torch.where(keep, x[tile_n], 0.0)
    return out


@helion.kernel(autotune_effort="none")
def sum_device_helper_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, helion.next_power_of_2(foo)], dtype=torch.float32)
        out[tile_m] = acc.sum(1) + x[tile_m, 0]
    return out


@helion.kernel(autotune_effort="none")
def sum_twice_device_helper_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, helion.next_power_of_2(foo) * 2], dtype=torch.float32)
        out[tile_m] = acc.sum(1) + x[tile_m, 0]
    return out


@helion.kernel(autotune_effort="none")
def fill_device_helper_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, helion.next_power_of_2(foo)], dtype=torch.float32)
        out[tile_m, : helion.next_power_of_2(foo)] = acc + 1
    return out


@helion.kernel(autotune_effort="none")
def fill_host_and_device_helper_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    width = helion.next_power_of_2(foo)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, helion.next_power_of_2(foo)], dtype=torch.float32)
        out[tile_m, :width] = acc + 1
    return out


@helion.kernel(autotune_effort="none")
def fill_device_cdiv_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    out = torch.empty([m, (foo + 1) // 2], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, helion.cdiv(foo, 2)], dtype=torch.float32)
        out[tile_m, :] = acc + hl.full([tile_m, (foo + 1) // 2], 1.0, torch.float32)
    return out


@helion.kernel(autotune_effort="none")
def fill_size_named_helper_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    out = torch.empty(
        [m, helion.next_power_of_2(foo)], dtype=torch.float32, device=x.device
    )
    for tile_m in hl.tile(m):
        out[tile_m, :] = hl.zeros([tile_m, out.size(1)], dtype=torch.float32) + 1
    return out


@helion.kernel(autotune_effort="none")
def fill_tunable_width_from_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(1, 256))
    out = torch.empty([m, foo], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m, :] = hl.zeros([tile_m, foo], dtype=torch.float32) + 1
    return out


def _two_unless_one(value: int) -> int:
    return 2 if value == 1 else 3


def _two_unless_17(value: int) -> int:
    return 2 if value == 17 else 3


@helion.kernel(autotune_effort="none")
def scale_unless_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", IntegerFragment(1, 8))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _two_unless_one(foo)
    return out


@helion.kernel(autotune_effort="none")
def scale_unless_17(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", IntegerFragment(2, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _two_unless_17(foo)
    return out


@helion.kernel(autotune_effort="none")
def scale_unless_17_of_an_enum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", EnumFragment((17, 32), search_choices=(32,)))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _two_unless_17(foo)
    return out


def _at_least_256(value: int) -> int:
    return max(value, 256)


@helion.kernel(autotune_effort="none")
def scale_by_floor_over_four_tunables(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    a = hl.register_tunable("a", IntegerFragment(64, 127))
    b = hl.register_tunable("b", IntegerFragment(64, 127))
    c = hl.register_tunable("c", IntegerFragment(64, 127))
    d = hl.register_tunable("d", IntegerFragment(64, 127))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _at_least_256(a + b + c + d)
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_max_with_a_default(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", IntegerFragment(1, 8))
    scale = max([foo, 2], default=0)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


def _capped(value: int) -> int:
    return min(value, 128)


@helion.kernel(autotune_effort="none")
def scale_by_capped_device_cdiv(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _capped(helion.cdiv(foo, 128))
    return out


@helion.kernel(autotune_effort="none")
def scale_by_capped_device_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _capped(helion.next_power_of_2(foo))
    return out


@helion.kernel(autotune_effort="none")
def scale_by_capped_tunable(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _capped(foo)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_tunable_asserted_equal_to_a_size(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    assert n == foo
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * foo
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_helper_of_a_helper_and_size(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(
            helion.next_power_of_2(foo) + n
        )
    return out


@helion.kernel(autotune_effort="none")
def scale_by_keyword_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(n=foo + 1)
    return out


def _next_power_of_2(value: int) -> int:
    return helion.next_power_of_2(value)


@helion.kernel(autotune_effort="none")
def scale_by_wrapped_helper(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * _next_power_of_2(foo + 1)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_device_helper_of_size(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(foo + n)
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo, 100)
    high = max(foo, 100)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low + high
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_max_of_sequences(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min([foo, 100])
    high = max((foo, 100))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low + high
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_host_min_of_size(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    scale = min(foo, n)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


@helion.kernel(autotune_effort="none")
def scale_by_device_helper_of_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(min(foo, 100))
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_of_a_square_and_zero(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    scale = min(foo * foo, 0)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


@helion.kernel(autotune_effort="none")
def sum_host_min_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo, 32)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, hl.arange(low)].sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def sum_host_min_width_every_config_agrees_on(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo, 8)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, low], dtype=x.dtype)
        out[tile_m] = (x[tile_m, hl.arange(low)] + acc).sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def sum_device_min_width_every_config_agrees_on(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, hl.arange(min(foo, 8))].sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_max_every_config_agrees_on(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo, 5000)
    high = max(foo, 8)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low + high
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_max_of_a_tunable_from_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", IntegerFragment(1, 8))
    scale = max(foo, 2)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_max_beside_a_float_or_a_default(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo, 8.0)
    high = max([foo, 8], default=0)
    one = min([foo], default=0)
    none = min([], default=foo)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low + high + one + none
    return out


@helion.kernel(autotune_effort="none")
def scale_by_tile_over_a_host_min_every_config_agrees_on(
    x: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile(min(foo, 8)):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_tile_over_a_host_min_of_a_size(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 2048))
    for tile_n in hl.tile(min(foo, n)):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_device_min_of_a_list(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * min([foo, 100])
    return out


@helion.kernel(autotune_effort="none")
def scale_by_device_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * min(foo, 100)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_helper_of_a_host_max_and_size(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = max(foo, 128)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * helion.next_power_of_2(low + n)
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_of_a_float(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    scale = min(foo, 100.0)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_with_a_default(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    scale = min([foo, 100], default=0)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out


def _render(kernel: helion.Kernel, x: torch.Tensor, foo: int) -> str:
    """``kernel``'s code for ``x`` under its default config with ``foo``."""
    with cpu_render_target():
        bound = cpu_bind(kernel, (x,))
        config = bound.config_spec.default_config().config | {"foo": foo}
        return bound.to_code(helion.Config(**config))


def _literal_block_sizes(code: str) -> dict[int, int]:
    """Each ``_BLOCK_SIZE_<i>`` the code defines as a literal: a Triton
    ``tl.constexpr`` or a CuTe host constant."""
    return {
        int(block): int(value)
        for block, value in re.findall(
            r"_BLOCK_SIZE_(\d+) = (?:tl\.constexpr\()?(\d+)\b", code
        )
    }


@pytest.mark.parametrize(
    ("kernel", "shape", "block_sizes"),
    [
        (scale_by_power_of_2_tile, (1000,), {0: 64}),
        (scale_by_cdiv_tile, (1000,), {0: 16}),
        # A helper over arithmetic on the tunable.
        (scale_by_power_of_2_above_tile, (1000,), {0: 128}),
        # CuTe sizes its thread block by both axes: the helper's had stayed
        # symbolic there while device code printed its value (refused).
        (scale_by_2d_tile, (64, 1000), {0: 16, 1: 64}),
    ],
)
def test_block_takes_the_helper_value_under_the_config(
    kernel: helion.Kernel, shape: tuple[int, ...], block_sizes: dict[int, int]
) -> None:
    # foo=64 is not the default (16): a block that ignored the config shows.
    code = _render(kernel, torch.empty(shape), foo=64)
    assert _literal_block_sizes(code) == block_sizes, code


# Python's min and max compared foo with 128 at its hint, 8192, a test the
# configs below 128 fail (refused).  Where the configs pick different
# arguments, they are the symbolic min or max device code computes, which the
# host evaluates under the config, as it does ``block_size=foo * 2``.
@pytest.mark.parametrize(
    ("kernel", "block"),
    [(scale_by_min_tile, "min(128, foo)"), (scale_by_max_tile, "max(128, foo)")],
)
def test_block_takes_the_host_min_or_max_under_the_config(
    kernel: helion.Kernel, block: str
) -> None:
    code = _render(kernel, torch.empty(1000), foo=64)
    assert f"_BLOCK_SIZE_0 = {block}" in code, code
    assert "8192" not in code


@helion.kernel(autotune_effort="none")
def scale_by_cdiv_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for i in hl.grid(helion.cdiv(foo, 2)):
        out[i] = x[i] * 2.0
    return out


def test_grid_takes_the_helper_value_under_the_config() -> None:
    code = _render(scale_by_cdiv_grid, torch.empty(1000), foo=64)
    # The launch covers the 32 programs ``helion.cdiv(foo, 2)`` gives.
    assert re.search(r"_launcher\(\w+, \(\(?32\b", code), code


@helion.kernel(autotune_effort="none")
def row_sum_by_tunable_tile(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m, block_size=foo):
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    ("kernel", "args", "heuristic"),
    [
        # ``block_size=[None, None, k_block]``: a named helper over a tunable.
        (
            import_example("matmul_split_k").matmul_split_k,
            (
                torch.empty(512, 4096, dtype=torch.float16),
                torch.empty(4096, 512, dtype=torch.float16),
            ),
            "triton_b200_formula_matmul",
        ),
        (row_sum_by_tunable_tile, (torch.empty(4096, 1024),), "triton_reduction"),
    ],
    ids=["split_k", "tunable_tile"],
)
@skipIfTileIR("names the Triton backend's seed heuristics; TileIR registers its own")
def test_seed_heuristics_read_a_tunable_block_size(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], heuristic: str
) -> None:
    # The seed heuristics read fixed block sizes outside code generation,
    # with no current host function: a block size a host variable names
    # stays symbolic for them to resolve.
    with cpu_render_target():
        spec = cpu_bind(kernel, args).config_spec
    assert heuristic in spec.autotuner_heuristics
    assert spec.compiler_seed_configs


@helion.kernel(
    static_shapes=False, config=helion.Config(indexing="tensor_descriptor", foo=16)
)
def scale_by_tunable_descriptor_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(8, 64))
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, foo * 2]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@skipUnlessBackends(["triton"])
def test_descriptor_bind_reads_a_tunable_block_size() -> None:
    # Binding caps the descriptors' extents by the config's block sizes,
    # outside code generation too.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        bound = scale_by_tunable_descriptor_tile.bind((torch.empty(256, 256),))
    assert bound.env.tensor_descriptor_layout_guards


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_named_mixed_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = helion.next_power_of_2(foo) * helion.next_power_of_2(n)
    for tile_n in hl.tile(n, block_size=block):
        out[tile_n] = x[tile_n] * block
    return out


def test_a_named_block_size_over_an_inline_helper_keeps_its_variable() -> None:
    # ``block`` names the product, but neither power of two alone: the host's
    # block size and the device's scale take the variable, rather than foo's
    # value times n's power of two, which nothing names (KeyError).
    code = _render(scale_by_named_mixed_tile, torch.empty(1000), foo=64)
    assert re.search(r"_BLOCK_SIZE_0 = block\b", code), code
    assert re.search(r"def _helion_\w+\([^)]*\bblock\b", code), code


@helion.kernel(autotune_effort="none")
def scale_by_scaled_helper_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = 2 * helion.next_power_of_2(foo)
    for tile_n in hl.tile([n], block_size=[block]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_rebound_helper_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = helion.next_power_of_2(foo)
    block = block * 2
    for tile_n in hl.tile([n], block_size=[block]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@pytest.mark.parametrize(
    "kernel", [scale_by_scaled_helper_tile, scale_by_rebound_helper_tile]
)
def test_a_variable_over_an_inline_helper_takes_the_config_value(
    kernel: helion.Kernel,
) -> None:
    # No variable names the helper's result alone, so the block takes 2 * 64
    # as a literal under foo=64, where origin/main passed ``block`` in.
    code = _render(kernel, torch.empty(1000), foo=64)
    assert _literal_block_sizes(code) == {0: 128}, code


def test_bind_time_facts_leave_a_helper_block_unfixed() -> None:
    # Bind computes some facts once for every config (``_immovable_extent``,
    # read by the affine ranges of cross-loop dependencies and by trip
    # counts): a block the config sizes has no width they may assume, while
    # a block a host variable names keeps the symbolic block it had.
    from helion._compiler.device_ir_analysis import _immovable_extent

    with cpu_render_target():
        helper = cpu_bind(scale_by_power_of_2_tile, (torch.empty(1000),))
        named = cpu_bind(row_sum_by_tunable_tile, (torch.empty(4096, 1024),))
    (block,) = helper.env.block_sizes
    assert block.block_size_source.varies_with_config
    with helper.env:
        assert _immovable_extent(helper.env, helper.config_spec, block.block_id) is None
    assert not named.env.block_sizes[0].block_size_source.varies_with_config


def test_extent_takes_the_helper_value_under_the_config() -> None:
    code = _render(scale_helper_extent, torch.empty(1000), foo=64)
    # The tile mask bounds the 1000 elements at the 64 the loop covers.
    assert re.search(r"< 64\b", code), code


# The type pass ran these helpers on the tunable's hint (8192), and every
# config computed with it.
def test_device_loop_block_and_scale_take_the_config_value() -> None:
    code = _render(scale_inner_tile_by_tunable, torch.empty(64, 1000), foo=64)
    assert _literal_block_sizes(code)[1] == 64, code
    assert "8192" not in code


def test_device_helper_value_takes_the_config_value() -> None:
    code = _render(scale_by_device_helper, torch.empty(64, 1000), foo=64)
    # next_power_of_2(64 + 1), not next_power_of_2(8192 + 1).
    assert re.search(r"symnode_\d+ = 128\b", code), code
    assert "16384" not in code


def test_device_loop_extent_takes_the_config_value() -> None:
    code = _render(inner_extent_device_helper, torch.empty(64, 1000), foo=64)
    assert re.search(r"symnode_\d+ = 64\b", code), code
    assert "8192" not in code


@pytest.mark.parametrize(("foo", "taken"), [(16, "False"), (64, "True")])
def test_device_branch_on_a_helper_value_follows_the_config(
    foo: int, taken: str
) -> None:
    code = _render(branch_on_device_helper, torch.empty(64, 1000), foo=foo)
    assert f"if {taken}:" in code, code


# A scalar comparison with the tunable combined with a tensor (``(tile.id ==
# foo - 1) & mask``, ``mask != (tile.id == foo - 1)``): the tensor op
# converted the SymBool with bool(), a test at the hint that refused the
# kernel. The comparison stays in the code, and the mask combines with it
# (CuTe spells != as operator.ne).
@pytest.mark.parametrize(
    ("kernel", "combine"),
    [
        (mask_tile_at_tunable_index, r"v_\d+ & eq\b"),
        (flip_tile_at_tunable_index, r"v_\d+ != eq\b|operator\.ne\(v_\d+, eq\)"),
    ],
)
def test_device_comparison_with_the_tunable_masks_a_tensor(
    kernel: helion.Kernel, combine: str
) -> None:
    code = _render(kernel, torch.empty(1000), foo=64)
    assert "eq = tile_id == -1 + foo" in code, code
    assert re.search(combine, code), code
    assert "foo = 64" in code, code


# A device cdiv over config values is the arithmetic it stands for, so its
# shape is the same arithmetic written out (as a symbol of its own, it was
# not, and the shapes did not match).
def test_device_cdiv_is_the_arithmetic_written_out() -> None:
    code = _render(fill_device_cdiv_width, torch.empty(64, 1000), foo=64)
    # Triton shapes the tile by the width, CuTe masks its lanes by it.
    assert re.search(r"tl\.full\(\[_BLOCK_SIZE_0, 32\]|indices_1 < 32\b", code), code
    assert "out_size_1" not in code


# A value the host names only by a tensor's size (``out.size(1)``) is its
# value under the config in device code, a constant a shape can take: the
# size argument is a runtime value (``tl.full([.., out_size_1])`` does not
# compile).
def test_a_helper_value_named_by_a_size_is_a_constant_shape() -> None:
    code = _render(fill_size_named_helper_width, torch.empty(64, 1000), foo=64)
    assert re.search(r"symnode_\d+ = 64\b|indices_1 < 64\b", code), code
    assert "out_size_1" not in code


# ``min(value, 128)`` in a function defined outside the kernel compares its
# argument at the hint.  Over cdiv(foo, 128), at most 2 for every value foo
# takes, the comparison tells no configs apart and the kernel keeps the
# division (it multiplied by 128: the cdiv was a symbol at the hint 8192).
def test_comparison_every_config_agrees_on_keeps_the_value() -> None:
    code = _render(scale_by_capped_device_cdiv, torch.empty(64, 1000), foo=64)
    assert "128.0" not in code, code
    assert re.search(
        r"div_floor_integer\(127 \+ foo, 128\)|\(127 \+ foo\) // 128", code
    ), code


# Over next_power_of_2(foo) or foo itself the comparison picks 128 for some
# configs and not for others: every config multiplied by 128.
@pytest.mark.parametrize(
    "kernel", [scale_by_capped_device_helper, scale_by_capped_tunable]
)
def test_comparison_that_tells_configs_apart_is_refused(kernel: helion.Kernel) -> None:
    with pytest.raises(helion.exc.TunableSpecialized, match="'foo'"):
        _render(kernel, torch.empty(64, 1000), foo=64)


# A comparison some value of the tunable fails is refused, and the refusal
# names the test and where it was made: 1, below the least size Helion
# assumes (2), or 17, between the ends and powers of two a sample would take.
@pytest.mark.parametrize(
    ("kernel", "test"),
    [(scale_unless_one, "Ne(foo, 1)"), (scale_unless_17, "Ne(foo, 17)")],
)
def test_comparison_one_value_fails_is_refused(
    kernel: helion.Kernel, test: str
) -> None:
    with pytest.raises(
        helion.exc.TunableSpecialized,
        match=rf"by the test {re.escape(test)} \(.*in _two_unless",
    ):
        _render(kernel, torch.empty(64, 1000), foo=4)


# A choice the search leaves out is still a config: 17 takes the helper's other
# branch, though 32, the one value searched, agrees with itself.
def test_comparison_an_unsearched_enum_choice_fails_is_refused() -> None:
    with pytest.raises(
        helion.exc.TunableSpecialized,
        match=r"by the test .*foo.*17.* \(.*in _two_unless",
    ):
        _render(scale_unless_17_of_an_enum, torch.empty(64, 1000), foo=32)


# The tunables' ranges decide ``max(a + b + c + d, 256)`` (the sum is 256 to
# 508): no config is tried one by one, where trying each would take 64**4.
def test_comparison_the_ranges_decide_tries_no_config() -> None:
    with (
        patch.object(
            CompileEnvironment, "_holds_for_each_config", side_effect=AssertionError
        ),
        cpu_render_target(),
    ):
        bound = cpu_bind(scale_by_floor_over_four_tunables, (torch.empty(64, 1000),))
        code = bound.to_code(bound.config_spec.default_config())
    assert "= a + b + c + d\n" in code, code


# A builtin the kernel calls compares in Helion's call of it, not in a frame
# of its own, and the test is the kernel's: ``max([foo, 2], default=0)``
# tests foo >= 2, which foo=1 fails (it multiplied by foo, 1 at foo=1).
def test_comparison_in_a_builtin_the_kernel_calls_is_refused() -> None:
    with pytest.raises(
        helion.exc.TunableSpecialized,
        match=r"by the test foo >= 2 \(in a builtin the kernel calls\)",
    ):
        _render(scale_by_host_max_with_a_default, torch.empty(64, 1000), foo=4)


# torch's shape checks of a dim a tunable sizes take the dim to be at least 2
# (``Ne(foo, 1)``), as Helion takes every symbolic size, even where the
# tunable starts at 1.
def test_size_checks_of_a_dim_a_tunable_sizes_from_1_are_kept() -> None:
    code = _render(fill_tunable_width_from_one, torch.empty(64, 1000), foo=1)
    # Triton shapes the tile by the width, CuTe masks its lanes by it.
    assert re.search(r"tl\.full\(\[_BLOCK_SIZE_0, foo\]|indices_1 < foo\b", code), code


# ``assert n == foo`` on the host unifies the tunable with the size, which
# still computes with the config's value (it was refused).
def test_tunable_asserted_equal_to_a_size_renders() -> None:
    code = _render(scale_by_tunable_asserted_equal_to_a_size, torch.empty(64, 64), 64)
    assert "8192" not in code, code


def test_device_allocation_takes_the_config_value() -> None:
    code = _render(fill_device_helper_width, torch.empty(64, 1000), foo=64)
    # The allocation and the slice cover the 64 columns the config gives.
    assert re.search(r"indices_1 < 64\b", code), code
    assert "8192" not in code


# The device call shares the host's value, which the kernel takes as the
# argument ``width``: a compile-time constant, as a device allocation's
# shape must be (Triton refused a plain one).
def test_host_and_device_helper_value_is_a_constexpr_argument() -> None:
    code = _render(fill_host_and_device_helper_width, torch.empty(64, 1000), foo=64)
    assert re.search(r"\bwidth: (tl\.constexpr|cutlass\.Constexpr)", code), code
    assert "8192" not in code


@pytest.mark.parametrize(
    "kernel",
    [sum_device_helper_width, sum_twice_device_helper_width, sum_host_min_width],
)
def test_reduction_over_a_helper_width_is_refused(kernel: helion.Kernel) -> None:
    with pytest.raises(helion.exc.ReductionOnNonTile, match="a size the config sets"):
        _render(kernel, torch.empty(64, 1000), foo=64)


# A min every config agrees on (8, where foo takes 16 to 256) is that
# argument, a literal width the reduction takes.  The host's compiled before
# too (Python's min left a test every config passes); the device's reduced
# over Min(8, foo) (refused).
@pytest.mark.parametrize(
    "kernel",
    [
        sum_host_min_width_every_config_agrees_on,
        sum_device_min_width_every_config_agrees_on,
    ],
)
def test_reduction_over_a_min_every_config_agrees_on_compiles(
    kernel: helion.Kernel,
) -> None:
    code = _render(kernel, torch.empty(64, 1000), foo=64)
    assert not {"low", "foo"} & set(_kernel_parameters(code)), code


# Each ran the helper on the hints, and every use of foo printed 8192: the
# value a helper computes from the tunable in the first one, the keyword
# argument and the wrapper, whose calls the type pass does not record.
@pytest.mark.parametrize(
    "kernel",
    [
        scale_by_helper_of_a_helper_and_size,
        scale_by_helper_of_a_host_max_and_size,
        scale_by_keyword_helper,
        scale_by_wrapped_helper,
    ],
)
def test_helper_over_a_tunable_traced_at_the_hint_is_refused(
    kernel: helion.Kernel,
) -> None:
    with pytest.raises(helion.exc.TunableSpecialized, match="'foo'"):
        _render(kernel, torch.empty(64, 1000), foo=64)


def test_device_helper_over_a_tunable_and_a_dynamic_size_is_refused() -> None:
    with pytest.raises(helion.exc.TunableSpecialized, match="'foo'"):
        _render(scale_by_device_helper_of_size, torch.empty(64, 1000), foo=64)


def _kernel_parameters(code: str) -> list[str]:
    """The device kernel's parameter names, without annotations."""
    match = re.search(r"def _helion_\w+\(([^)]*)\)", code)
    assert match, code
    return [parameter.split(":")[0] for parameter in match.group(1).split(", ")]


# Python's min and max compared foo with 100 at its hint, 8192, a test some
# configs fail (refused).  A single list or tuple argument compared the same
# way.
@pytest.mark.parametrize(
    "kernel", [scale_by_host_min_max, scale_by_host_min_max_of_sequences]
)
def test_host_min_and_max_pass_their_values(kernel: helion.Kernel) -> None:
    code = _render(kernel, torch.empty(64, 1000), foo=64)
    assert _kernel_parameters(code) == ["x", "out", "low", "high"], code


# Over a dynamic size the comparison at the hints picked n, a test configs
# fail (refused).
def test_host_min_beside_a_dynamic_size_passes_its_value() -> None:
    code = _render(scale_by_host_min_of_size, torch.empty(64, 1000), foo=64)
    assert "scale" in _kernel_parameters(code), code


def test_device_helper_over_a_min_takes_the_config_value() -> None:
    code = _render(scale_by_device_helper_of_min, torch.empty(64, 1000), foo=64)
    # next_power_of_2(min(64, 100)), not next_power_of_2(100).
    assert re.search(r"symnode_\d+ = 64\b", code), code


# The symbolic min of foo * foo (which no config value expression reads)
# and 0 simplifies to the int 0.
def test_host_min_that_simplifies_to_an_int_renders() -> None:
    code = _render(scale_by_host_min_of_a_square_and_zero, torch.empty(64, 1000), 64)
    assert "scale" not in _kernel_parameters(code), code


# Beside a float, or with default=, Python's min still compares foo with 100
# at its hint, a test some configs fail: refused.
@pytest.mark.parametrize(
    "kernel", [scale_by_host_min_of_a_float, scale_by_host_min_with_a_default]
)
def test_host_min_compared_at_the_hint_is_refused(kernel: helion.Kernel) -> None:
    with pytest.raises(helion.exc.TunableSpecialized, match="'foo'"):
        _render(kernel, torch.empty(64, 1000), foo=64)


# Comparing foo with 8, a test every config passes, or comparing nothing
# (``min([foo], default=0)``, ``min([], default=foo)``) keeps the value:
# 8.0 and foo.
def test_host_min_max_beside_a_float_or_a_default_every_config_agrees_on() -> None:
    code = _render(
        scale_by_host_min_max_beside_a_float_or_a_default, torch.empty(64, 1000), 64
    )
    assert _kernel_parameters(code) == ["x", "out", "foo"], code
    assert "8.0" in code, code


# Every config picks foo over 5000 and over 8, so both are foo; the
# comparison at the hint, 8192, picked 5000 (refused).
def test_host_min_max_every_config_agrees_on_is_that_argument() -> None:
    code = _render(
        scale_by_host_min_max_every_config_agrees_on, torch.empty(64, 1000), 64
    )
    assert _kernel_parameters(code) == ["x", "out", "foo"], code


# With foo from 1 to 8, max(foo, 2) is 2 at foo=1 and foo above: the host
# passes Max(2, foo).  The comparison at the hint tested foo >= 2, which
# foo=1 fails (refused).
def test_host_max_the_configs_disagree_on_passes_its_value() -> None:
    code = _render(scale_by_host_max_of_a_tunable_from_one, torch.empty(64, 1000), 4)
    assert _kernel_parameters(code) == ["x", "out", "scale"], code


# A tile over min(foo, 8) has the extent 8, as every config gives it.  Over a
# min the configs disagree on, its block sizes reach the extent the hints
# give, min(8192, 1000), not the 8192 an unknown size takes.
@pytest.mark.parametrize(
    ("kernel", "max_size"),
    [
        (scale_by_tile_over_a_host_min_every_config_agrees_on, 8),
        (scale_by_tile_over_a_host_min_of_a_size, 1024),
    ],
)
def test_tile_over_a_host_min_searches_up_to_its_extent(
    kernel: helion.Kernel, max_size: int
) -> None:
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(1000),))
    assert bound.config_spec.block_sizes[0].max_size == max_size


def _without_source_lines(code: str, name: str) -> str:
    """``code`` without its source comments, the kernel named ``K``."""
    return "\n".join(
        line.replace(name, "K")
        for line in code.splitlines()
        if not line.lstrip().startswith("# src[")
    )


# In device code ``min([foo, 100])`` compares like ``min(foo, 100)``, as on
# the host (it failed to trace).
def test_device_min_of_a_list_compares_its_items() -> None:
    of_a_list = _render(scale_by_device_min_of_a_list, torch.empty(64, 1000), 64)
    separate = _render(scale_by_device_min, torch.empty(64, 1000), 64)
    assert _without_source_lines(
        of_a_list, "scale_by_device_min_of_a_list"
    ) == _without_source_lines(separate, "scale_by_device_min")


@helion.kernel(autotune_effort="none")
def fill_host_and_device_min_width(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(64, 256))
    width = min(foo, 128)
    out = torch.empty([m, width], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        ones = hl.full([tile_m, width], 1.0, dtype=torch.float32)
        out[tile_m, :] = ones + hl.full([tile_m, min(foo, 128)], 2.0, torch.float32)
    return out


@helion.kernel(autotune_effort="none")
def fill_host_and_device_min_width_every_config_agrees_on(
    x: torch.Tensor,
) -> torch.Tensor:
    m, n = x.size()
    foo = hl.register_tunable("foo", PowerOfTwoFragment(128, 256))
    width = min(foo, 128)
    out = torch.empty([m, width], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        ones = hl.full([tile_m, width], 1.0, dtype=torch.float32)
        out[tile_m, :] = ones + hl.full([tile_m, min(foo, 128)], 2.0, torch.float32)
    return out


# The host's ``min(foo, 128)`` is what device code computes, so the two
# shapes match: ``Min(128, foo)`` where foo takes 64 to 256, and 128 where it
# takes 128 or 256.
@pytest.mark.parametrize(
    "kernel",
    [
        fill_host_and_device_min_width,
        fill_host_and_device_min_width_every_config_agrees_on,
    ],
)
def test_host_and_device_min_compare_equal(kernel: helion.Kernel) -> None:
    code = _render(kernel, torch.empty(64, 1000), foo=128)
    assert "8192" not in code, code


@helion.kernel(autotune_effort="none")
def scale_by_host_min_max_of_one_element(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min([foo])
    high = max((foo,))
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low + high
    return out


@helion.kernel(autotune_effort="none")
def scale_by_host_min_of_a_number(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    low = min(foo)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * low
    return out


# A one-element list or tuple is its element, which Python's min compares
# with nothing, and ``min`` of a single number fails to compile, as Python's
# does at run time.
def test_host_min_max_of_one_element_is_the_element() -> None:
    code = _render(scale_by_host_min_max_of_one_element, torch.empty(64, 1000), 64)
    assert _kernel_parameters(code) == ["x", "out", "foo"], code


def test_host_min_of_a_number_is_refused() -> None:
    with pytest.raises(helion.exc.TorchOpTracingError, match="not iterable"):
        _render(scale_by_host_min_of_a_number, torch.empty(64, 1000), foo=64)


# The SM count of the modelled CPU target, which sets the workers' stride.
_CPU_TARGET_SMS = 148


def _worker_tasks(code: str) -> str:
    """The task count of the first root a static plan's workers stride over."""
    return re.findall(
        rf"tl\.range\(tl\.program_id\(0\)[^,]*, (\d+), {_CPU_TARGET_SMS}\)", code
    )[0]


def _assert_no_config_value_pinned(bound: helion.BoundKernel) -> None:
    """No symbol a config sets got replaced by an integer in the binding."""
    env = bound.env
    assert not set(env.shape_env.replacements) & env.config_value_expressions.keys()


def _split_k_partials_then_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # examples/split_k_barrier.py without its barrier: the cross-loop
    # pipeline orders the sum after the partials it reads.
    m, k = x.size()
    _, n = y.size()
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 8, 4))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    partials = torch.empty([split_k, m, n], device=x.device, dtype=torch.float32)
    out = torch.empty([m, n], device=x.device, dtype=torch.float32)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k], block_size=[None, None, block_k]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k_inner in hl.tile(tile_k.begin, tile_k.end):
            acc = hl.dot(x[tile_m, tile_k_inner], y[tile_k_inner, tile_n], acc=acc)
        partials[tile_k.id, tile_m, tile_n] = acc
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(partials[:, tile_m, tile_n], dim=0)
    return out


def _split_k_config(split_k: int, pipeline: str) -> helion.Config:
    return helion.Config(
        block_sizes=[16, 64, 256, 16, 64],
        num_warps=4,
        num_stages=3,
        split_k=split_k,
        pid_type="persistent_blocked",
        cross_loop_pipeline=pipeline,
    )


# The static plan read the named helper's block_k with int(), which pinned
# its symbol to the fallback hint 8192 in the bound kernel's shape
# environment: every config, static or dynamic, then planned the partials
# root for two K tiles while the host passed the config's block_k, and the
# sum read partials nobody wrote.  One bound kernel compiles each config.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
@pytest.mark.parametrize("splits", [(2, 4, 8), (8, 4, 2)], ids=["up", "down"])
def test_cross_loop_plan_takes_each_configs_named_helper_block(
    splits: tuple[int, ...],
) -> None:
    x = torch.empty(16, 16384, dtype=torch.bfloat16)
    y = torch.empty(16384, 512, dtype=torch.bfloat16)
    kernel = helion.kernel(_split_k_partials_then_sum, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        bound = cpu_bind(kernel, (x, y))
        for split_k, pipeline in [(s, "static") for s in splits] + [
            (splits[0], "dynamic")
        ]:
            code = bound.to_triton_code(_split_k_config(split_k, pipeline))
            # The sum's root barrier waits for every task of the partials
            # root: one 16x64 output tile of 8 per K tile.
            assert re.findall(r"tl\.cast\((\d+), tl\.uint32\)", code)[0] == str(
                8 * split_k
            ), (split_k, pipeline, code)
            # So is the module constant an outlined root reads block_k from.
            assert _literal_block_sizes(code)[2] == 16384 // split_k, code
    _assert_no_config_value_pinned(bound)


def _plan_over_an_input_size_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=y.size(0)):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_derived_size_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=y.size(0) // 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_merged_size_block(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    assert z.size(0) == y.size(0)
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=y.size(0) // 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


# An input size torch merged into another (``assert z.size(0) == y.size(0)``)
# keys the binding through the size that stands for it.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
def test_cross_loop_plan_keys_a_merged_input_size() -> None:
    kernel = helion.kernel(_plan_over_a_merged_size_block, static_shapes=False)
    config = helion.Config(
        block_sizes=[64], pid_type="persistent_blocked", cross_loop_pipeline="static"
    )
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        args = (torch.empty(256), torch.empty(128), torch.empty(128))
        bound = kernel.bind(args)
        code = bound.to_triton_code(config)
        assert _worker_tasks(code) == "4", code
        assert kernel.bind(args) is bound
        smaller = (torch.empty(256), torch.empty(64), torch.empty(64))
        assert kernel.bind(smaller) is not bound


# The static plan counts the tasks of a block an input size gives by the bound
# size, so the binding keys each input size the block reads (from_config_int).
# main's int() keyed a bare size through a replacement but only guarded a
# derived one, so the derived case is the regression check.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
@pytest.mark.parametrize(
    ("fn", "size"),
    [(_plan_over_an_input_size_block, 64), (_plan_over_a_derived_size_block, 128)],
    ids=["bare", "derived"],
)
def test_cross_loop_plan_keys_the_binding_on_an_input_size_block(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], size: int
) -> None:
    kernel = helion.kernel(fn, static_shapes=False)
    config = helion.Config(
        block_sizes=[64], pid_type="persistent_blocked", cross_loop_pipeline="static"
    )
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        bound = kernel.bind((torch.empty(256), torch.empty(size)))
        code = bound.to_triton_code(config)
        assert _worker_tasks(code) == "4", code
        assert kernel.bind((torch.empty(256), torch.empty(size))) is bound
        assert kernel.bind((torch.empty(256), torch.empty(size // 2))) is not bound


def _plan_over_a_scalar_block(x: torch.Tensor, bs: int) -> torch.Tensor:
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=bs * 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


# A module constant: the plan takes its value at compile time.
_PLAN_GLOBAL_BLOCK = 32


def _plan_over_a_global_block(x: torch.Tensor) -> torch.Tensor:
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=_PLAN_GLOBAL_BLOCK * 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_scalar_times_tunable_block(x: torch.Tensor, bs: int) -> torch.Tensor:
    factor = hl.register_tunable("factor", PowerOfTwoFragment(1, 4, 2))
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=bs * factor):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_specialized_scalar_block(x: torch.Tensor, bs: int) -> torch.Tensor:
    bs = hl.specialize(bs)
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=bs * 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_specialized_derived_block(x: torch.Tensor, bs: int) -> torch.Tensor:
    bs = hl.specialize(bs // 2)
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=bs * 2):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


def _plan_over_a_specialized_times_tunable_block(
    x: torch.Tensor, bs: int
) -> torch.Tensor:
    bs = hl.specialize(bs // 2)
    factor = hl.register_tunable("factor", PowerOfTwoFragment(1, 4, 2))
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=bs * factor):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


# hl.specialize and a tunable fix a block between them: 32 * factor.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
@pytest.mark.parametrize(("factor", "tasks"), [(2, "4"), (4, "2")])
def test_cross_loop_plan_takes_a_specialized_times_tunable_block(
    factor: int, tasks: str
) -> None:
    kernel = helion.kernel(
        _plan_over_a_specialized_times_tunable_block, static_shapes=True
    )
    config = helion.Config(
        block_sizes=[64],
        pid_type="persistent_blocked",
        cross_loop_pipeline="static",
        factor=factor,
    )
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        code = kernel.bind((torch.empty(256), 64)).to_triton_code(config)
    assert _worker_tasks(code) == tasks, code


def _plan_over_a_global_times_tunable_block(x: torch.Tensor) -> torch.Tensor:
    factor = hl.register_tunable("factor", PowerOfTwoFragment(1, 4, 2))
    mid = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(256, block_size=_PLAN_GLOBAL_BLOCK * factor):
        mid[tile] = x[tile] + 1
    for tile in hl.tile(256):
        out[tile] = mid[tile] * 2
    return out


# A module constant times a tunable: 32 * factor.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
@pytest.mark.parametrize(("factor", "tasks"), [(2, "4"), (4, "2")])
def test_cross_loop_plan_takes_a_global_times_tunable_block(
    factor: int, tasks: str
) -> None:
    kernel = helion.kernel(_plan_over_a_global_times_tunable_block, static_shapes=True)
    config = helion.Config(
        block_sizes=[64],
        pid_type="persistent_blocked",
        cross_loop_pipeline="static",
        factor=factor,
    )
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        code = kernel.bind((torch.empty(256),)).to_triton_code(config)
    assert _worker_tasks(code) == tasks, code


# Split-K's block_k over a dynamic K has no value to plan by: the plan says
# so and how to give it one.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
def test_cross_loop_plan_refuses_a_tunable_beside_a_dynamic_size() -> None:
    kernel = helion.kernel(_split_k_partials_then_sum, static_shapes=False)
    x = torch.empty(16, 16384, dtype=torch.bfloat16)
    y = torch.empty(16384, 512, dtype=torch.bfloat16)
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        bound = kernel.bind((x, y))
        with pytest.raises(helion.exc.InvalidConfig, match="static_shapes=True"):
            bound.to_triton_code(_split_k_config(4, "static"))


# A bound kernel does not key a scalar argument's value, so a plan over a
# block it gives would serve every value with the first one's task count: the
# plan refuses it, also times a tunable, and hl.specialize keys the value and
# plans it.  A module constant's block plans its value.
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
def test_cross_loop_plan_takes_a_scalar_arguments_block_only_specialized() -> None:
    config = helion.Config(
        block_sizes=[64], pid_type="persistent_blocked", cross_loop_pipeline="static"
    )
    with cpu_render_target(fake_device_sm_count=_CPU_TARGET_SMS):
        kernel = helion.kernel(_plan_over_a_scalar_block, static_shapes=True)
        bound = kernel.bind((torch.empty(256), 32))
        with pytest.raises(helion.exc.InvalidConfig, match="hl.specialize"):
            bound.to_triton_code(config)
        kernel = helion.kernel(_plan_over_a_scalar_times_tunable_block)
        bound = kernel.bind((torch.empty(256), 32))
        with pytest.raises(helion.exc.InvalidConfig, match="scalar argument"):
            bound.to_triton_code(helion.Config(**config, factor=2))
        for fn, args in [
            (_plan_over_a_specialized_scalar_block, (32,)),
            (_plan_over_a_specialized_derived_block, (64,)),
            (_plan_over_a_global_block, ()),
        ]:
            kernel = helion.kernel(fn, static_shapes=True)
            bound = kernel.bind((torch.empty(256), *args))
            code = bound.to_triton_code(config)
            if args:
                assert kernel.bind((torch.empty(256), args[0] // 2)) is not bound
            assert "_BLOCK_SIZE_0 = tl.constexpr(64)" in code, code
            assert _worker_tasks(code) == "4", code


@helion.kernel(autotune_effort="none")
def _rows_over_a_helper_block(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    split = hl.register_tunable("split", PowerOfTwoFragment(1, 8, 4))
    block = helion.next_power_of_2(helion.cdiv(n, split))
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


# The range pipeline clamp counts a device loop's iterations by the config's
# block (4096 / 512 = 8 at split 8), not the 8192 fallback hint, which made
# every config one stage and pinned the tunable.
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR does not pipeline ranges")
@pytest.mark.parametrize(
    ("split", "unroll", "num_warps", "stages"),
    [(8, 2, 1, 3), (8, 4, 1, 2), (2, 2, 1, 1), (8, 2, 4, 1)],
)
def test_range_clamp_counts_a_helper_blocks_iterations(
    split: int, unroll: int, num_warps: int, stages: int
) -> None:
    with cpu_render_target():
        bound = cpu_bind(_rows_over_a_helper_block, (torch.empty(64, 4096),))
        code = bound.to_triton_code(
            helion.Config(
                block_sizes=[16],
                num_warps=num_warps,
                split=split,
                range_num_stages=[0, 3],
                range_unroll_factors=[0, unroll],
            )
        )
    assert f"loop_unroll_factor={unroll}, num_stages={stages})" in code, code
    _assert_no_config_value_pinned(bound)


@helion.kernel(autotune_effort="none")
def _rows_over_a_scalar_block(x: torch.Tensor, bs: int) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=bs * 2):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


# A block over a scalar argument has no size the clamp may count by: one
# warp has no hazard and keeps the stages asked for, and more warps keep one.
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR does not pipeline ranges")
@pytest.mark.parametrize(("num_warps", "stages"), [(1, 3), (4, 1)])
def test_range_clamp_without_a_blocks_size(num_warps: int, stages: int) -> None:
    with cpu_render_target():
        bound = cpu_bind(_rows_over_a_scalar_block, (torch.empty(64, 4096), 256))
        code = bound.to_triton_code(
            helion.Config(
                block_sizes=[16],
                num_warps=num_warps,
                range_num_stages=[0, 3],
                range_unroll_factors=[0, 2],
            )
        )
    assert f"loop_unroll_factor=2, num_stages={stages})" in code, code


def _rows_over_a_derived_size_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(4096, block_size=y.size(0) // 2):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


# With more than one warp the clamp keeps one stage whatever the block, so it
# reads no block size and keys no input size it would read.
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR does not pipeline ranges")
def test_multi_warp_range_clamp_keys_no_size() -> None:
    kernel = helion.kernel(_rows_over_a_derived_size_block, static_shapes=False)
    config = helion.Config(
        block_sizes=[16],
        num_warps=4,
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 2],
    )
    with cpu_render_target():
        bound = kernel.bind((torch.empty(64, 4096), torch.empty(1024)))
        code = bound.to_triton_code(config)
        assert kernel.bind((torch.empty(64, 4096), torch.empty(512))) is bound
    assert "loop_unroll_factor=2, num_stages=1)" in code, code
    assert not bound.env.keyed_size_equalities


# On one warp the clamp counts by the bound size's block, a heuristic, so it
# keys no size: another size shares the bound kernel.
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR does not pipeline ranges")
def test_one_warp_range_clamp_keys_no_size() -> None:
    kernel = helion.kernel(_rows_over_a_derived_size_block, static_shapes=False)
    config = helion.Config(
        block_sizes=[16],
        num_warps=1,
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 2],
    )
    with cpu_render_target():
        bound = kernel.bind((torch.empty(64, 4096), torch.empty(1024)))
        code = bound.to_triton_code(config)
        assert kernel.bind((torch.empty(64, 4096), torch.empty(512))) is bound
    assert "loop_unroll_factor=2, num_stages=3)" in code, code
    assert not bound.env.keyed_size_equalities


@helion.kernel(autotune_effort="none")
def _swizzle_gather_over_a_helper_block(
    scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    (n,) = out.shape
    split = hl.register_tunable("split", PowerOfTwoFragment(1, 8, 4))
    block = helion.next_power_of_2(helion.cdiv(n, split))
    for tile in hl.tile(n, block_size=block):
        out[tile] = scale[(tile.index // 4) * 512 + tile.index % 4]
    return out


# The gather contiguity proof evaluates program 0's tile at the config's
# block; reading it with int() pinned the tunable to 8192, and later configs
# passed that block.
@skipUnlessBackends(["triton"])
@skipIfTileIR("hint is emitted by the Triton pointer indexing strategy")
def test_contiguity_proof_reads_a_helper_block_unpinned() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            _swizzle_gather_over_a_helper_block,
            (torch.empty(8192), torch.empty(64)),
        )
        for split in (8, 2):
            code = bound.to_triton_code(
                helion.Config(indexing=["pointer", "pointer"], split=split)
            )
            assert "tl.max_contiguous(" in code and ", [4])" in code, code
            assert "_BLOCK_SIZE_0 = block" in code, code
    _assert_no_config_value_pinned(bound)


def _run_split_k_configs(backend: str) -> None:
    x = torch.randn(16, 4096, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(4096, 128, device=DEVICE, dtype=torch.bfloat16)
    expected = x.float() @ y.float()
    kernel = helion.kernel(
        _split_k_partials_then_sum, static_shapes=True, backend=backend
    )
    bound = kernel.bind((x, y))
    for split_k, pipeline in [(2, "static"), (8, "static"), (4, "dynamic")]:
        out = bound.compile_config(_split_k_config(split_k, pipeline))(x, y)
        torch.testing.assert_close(
            out,
            expected,
            rtol=1e-3,
            atol=1e-2,
            msg=lambda message, case=(split_k, pipeline): f"{case}: {message}",
        )


@pytest.mark.cuda
@skipIfNotCUDA()
@skipUnlessBackends(["triton"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
def test_cross_loop_split_k_over_a_tunable_split_count() -> None:
    _run_split_k_configs("triton")


@pytest.mark.cuda
@skipIfNotCUDA()
@skipUnlessBackends(["triton"])
@skipUnlessGluon()
def test_gluon_cross_loop_split_k_over_a_tunable_split_count() -> None:
    _run_split_k_configs("gluon")
