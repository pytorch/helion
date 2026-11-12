"""Block sizes and extents a host helper over tunables computes, inline.

``hl.tile([n], block_size=[helion.next_power_of_2(foo)])`` passes a value no
host variable names.  The tile takes the value the helper gives under the
config being compiled, as a static block size like a literal one, on each
axis of a tile; a loop extent written the same way is that value too, and so
is the helper's value in device code (a device loop's block size or extent, a
scale).  Python's min and max over tunables on the host take the argument
every config picks, or else the config's value, instead of comparing the
tunables at their hints.  Code generation only; test_register_tunable.py runs
one.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton", "cute"]),
    # Every test binds a kernel and reads its config spec or generated code.
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="inspects config specs and generated code; ref mode has neither",
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
