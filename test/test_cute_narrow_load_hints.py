"""The ``cute_narrow_load_hints`` knob: cache policies on 2-byte element loads.

A bf16, fp16 or int16 element load (a gather, a lane on a strided dim)
carried no ``load_eviction_policies`` value at all: its ``cute.arch.load``
lowers to ``nvvm.load.ext``, which rejects cache modifiers.  Under the knob
the element loads of such a tensor take the first policy its load slots
name, through the exact-policy helpers of ``element_policy_loads.py``, a
masked one as one predicated load; ``hint_element_loads`` places them after
every other pass, like the L2 policies of ``cute_scalar_load_hints``.  The
knob is off by default and the search does not set it: the hinted 2-byte
load measured slower than the plain one on the kernel it was built for (a
3x3 max pool over bf16 gathers).  The GPU numerics are at the end of the
module.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.cache_policy_loads import CachePolicy
from helion._compiler.cute.cache_policy_loads import element_load_hint
from helion._compiler.cute.target_facts import TargetFacts
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_element_load_hints import adversarial_pairs
from helion._testing.cute_element_load_hints import assert_same_bits
from helion._testing.cute_element_load_hints import bound_components
from helion._testing.cute_element_load_hints import component_add
from helion._testing.cute_element_load_hints import element_load_helper_ir
from helion._testing.cute_element_load_hints import lines_without_loads
from helion._testing.cute_element_load_hints import load_lines
from helion._testing.cute_element_load_hints import policy_config
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

# Source checks, in the CPU lane: CUDA hidden and native compilation
# forbidden.  The GPU numerics at the end of the module are marked ``cuda``.
pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render(TargetFacts())]

_POLICIES = ("first", "last", "streaming", "l2_last", "l1_l2_first", "l1_l2_last")
_KNOB = {"cute_narrow_load_hints": True}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _relu_max_pool(x: torch.Tensor) -> torch.Tensor:
    # ReLU and a padded 3x3 max pool of stride 2: nine masked element
    # gathers of one tensor per output.
    planes, height, width = x.shape
    out_h = (height + 2 - 3) // 2 + 1
    out_w = (width + 2 - 3) // 2 + 1
    out = torch.empty([planes, out_h, out_w], dtype=x.dtype, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        top = tile_h.index * 2 - 1
        left = tile_w.index * 2 - 1
        best = torch.full_like(out[tile_p, tile_h, tile_w], float("-inf")).to(
            torch.float32
        )
        for offset in hl.static_range(9):
            row = top + offset // 3
            col = left + offset % 3
            inside = ((row >= 0) & (row < height))[None, :, None] & (
                (col >= 0) & (col < width)
            )[None, None, :]
            value = torch.relu(
                hl.load(x, [tile_p, row[:, None], col[None, :]], extra_mask=inside).to(
                    torch.float32
                )
            )
            best = torch.maximum(best, torch.where(inside, value, float("-inf")))
        out[tile_p, tile_h, tile_w] = best.to(out.dtype)
    return out


@pytest.mark.parametrize("policy", _POLICIES)
def test_masked_two_byte_elements_take_one_predicated_load(policy: str) -> None:
    bound = bound_components(torch.bfloat16, 1000)
    off = bound.to_code(policy_config(bound, [policy, ""]))
    # Off, and under the knob of the wider elements alone: no hint, as
    # always.
    assert "_policy_load" not in off
    for line in load_lines(off):
        assert line.endswith(".load() if mask_0 else cutlass.BFloat16(0)")
    assert off == bound.to_code(
        policy_config(bound, [policy, ""], cute_scalar_load_hints=True)
    )
    # On: every element load of the tensor carries the policy its first
    # slot names, each under its mask in one call.
    on = bound.to_code(policy_config(bound, [policy, ""], **_KNOB))
    first, second = load_lines(on)
    for line in (first, second):
        assert "= _cute_element_policy_load_if(pairs.iterator + " in line
        assert line.endswith(f", mask_0, cutlass.BFloat16, {policy!r})")
        assert " if mask_0 else " not in line
    # Nothing else changes: the store, the arithmetic and the launch.
    assert lines_without_loads(on) == lines_without_loads(off)


@pytest.mark.parametrize("policy", ["last", "l1_l2_last"])
def test_unmasked_elements_take_the_unpredicated_load(policy: str) -> None:
    bound = bound_components(torch.int16, 1024, component_add)
    off = bound.to_code(policy_config(bound, [policy, ""]))
    on = bound.to_code(policy_config(bound, [policy, ""], **_KNOB))
    strided, contiguous = load_lines(on)
    assert strided.startswith("load = _cute_element_policy_load(pairs.iterator + ")
    assert strided.endswith(f", cutlass.Int16, {policy!r})")
    # The other tensor's slot names nothing: its load is the plain one.
    assert contiguous == load_lines(off)[1]
    assert "_cute_element_policy_load_if" not in on
    assert lines_without_loads(on) == lines_without_loads(off)


def test_a_hint_keeps_no_element_loads_from_merging() -> None:
    # The two unmasked components are one 4-byte packet; the hint is placed
    # after the merge, so no element load is left to take it.
    bound = bound_components(torch.bfloat16, 1024)
    plain = bound.to_code(policy_config(bound, ["", ""]))
    assert "_helion_literal_vec" in plain
    assert bound.to_code(policy_config(bound, ["last", "l2_last"], **_KNOB)) == plain


@pytest.mark.parametrize("dtype", [torch.int8, torch.bool, torch.uint8, torch.uint16])
def test_byte_and_uint16_elements_take_no_hint(dtype: torch.dtype) -> None:
    bound = bound_components(dtype, 1000)
    policies = ["last", "l2_last"]
    off = bound.to_code(policy_config(bound, policies))
    assert bound.to_code(policy_config(bound, policies, **_KNOB)) == off
    assert "_policy_load" not in off


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _spelled_then_slot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ``x``: an element load that spells out its own policy (it owns no load
    # slot), lowered ahead of one through a slot.  ``y``: a slot of its own.
    n = x.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        spelled = hl.load(x, [tile, 0], eviction_policy="evict_first")
        out[tile] = spelled + x[tile, 1] + y[tile, 0]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _slot_then_spelled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The same loads with the slot's lowered first.
    n = x.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        slot = x[tile, 1]
        spelled = hl.load(x, [tile, 0], eviction_policy="evict_first")
        out[tile] = slot + spelled + y[tile, 0]
    return out


@pytest.mark.parametrize("kernel", [_spelled_then_slot, _slot_then_spelled])
@pytest.mark.parametrize("policy", ["l2_last", ""])
def test_a_tensor_with_a_spelled_out_policy_takes_no_hint(
    kernel: Any, policy: str
) -> None:
    # A 2-byte load that spells out its policy is a plain load after the
    # passes, like the tensor's other loads: the pass could neither leave it
    # its own policy nor keep a slot's from it, so the tensor takes none,
    # whichever load is lowered first.  The other tensor takes its slot's.
    arguments = (
        torch.empty((1000, 2), dtype=torch.bfloat16),
        torch.empty((1000, 2), dtype=torch.bfloat16),
    )
    bound = cpu_bind(kernel, arguments)
    assert bound.config_spec.load_eviction_policies.length == 2
    off = bound.to_code(policy_config(bound, [policy, policy]))
    on = bound.to_code(policy_config(bound, [policy, policy], **_KNOB))
    assert "_policy_load" not in off
    assert "(x.iterator" in on and "_policy_load_if(x.iterator" not in on
    assert "_policy_load(x.iterator" not in on
    if policy:
        assert on.count("_cute_element_policy_load_if(y.iterator") == 1
        # Nothing but that load and its helper's import differs.
        plain = set(off.splitlines())
        changed = [line for line in on.splitlines() if line not in plain]
        assert len(changed) == 2
        assert all(
            "y.iterator" in line or "_policy_load_if as" in line for line in changed
        )
    else:
        assert on == off


def test_wider_elements_keep_their_forms_under_the_narrow_knob() -> None:
    # The knob names the 2-byte elements only: a 4-byte element keeps the
    # conditional around its L1 hint and drops the L2 policy, as without it.
    bound = bound_components(torch.float32, 1000)
    off = bound.to_code(policy_config(bound, ["last", "l2_last"]))
    assert "_cute_scalar_policy_load(" in off
    assert bound.to_code(policy_config(bound, ["last", "l2_last"], **_KNOB)) == off


def test_the_hint_a_slot_gives_a_two_byte_tensor() -> None:
    for dtype in (torch.bfloat16, torch.float16, torch.int16):
        for policy in _POLICIES:
            hint = CachePolicy.tunable(policy)
            assert (
                element_load_hint(dtype, hint, l2_elements=False, narrow_elements=True)
                == policy
            )
            assert (
                element_load_hint(dtype, hint, l2_elements=True, narrow_elements=False)
                is None
            )
        for policy in ("nc_l2_first", ""):
            assert (
                element_load_hint(
                    dtype,
                    CachePolicy.tunable(policy),
                    l2_elements=True,
                    narrow_elements=True,
                )
                is None
            )
    for dtype in (torch.int8, torch.uint8, torch.bool, torch.uint16):
        assert (
            element_load_hint(
                dtype,
                CachePolicy.tunable("last"),
                l2_elements=True,
                narrow_elements=True,
            )
            is None
        )


def test_the_search_leaves_the_knob_alone() -> None:
    bound = bound_components(torch.bfloat16, 1000)
    spec = bound.config_spec
    with bound.env:
        generation = ConfigGeneration(spec)
        saved = policy_config(bound, ["l1_l2_last", "last"])
        decoded = generation.unflatten(generation.flatten(saved))
        assert "cute_narrow_load_hints" not in decoded.config
        assert "cute_narrow_load_hints" not in spec._flat_fields()
    # The search times the plain loads of a 2-byte tensor, whatever its
    # slots name.
    assert bound.to_code(decoded) == bound.to_code(saved)
    base = spec.default_config().config
    with pytest.raises(
        exc.InvalidConfig, match="cute_narrow_load_hints must be a boolean"
    ):
        spec.normalize(helion.Config.from_dict(base | {"cute_narrow_load_hints": 1}))
    for value in (False, True):
        normalized = spec.normalized_config(
            helion.Config.from_dict(base | {"cute_narrow_load_hints": value})
        )
        assert normalized.config["cute_narrow_load_hints"] is value
    assert (
        "cute_narrow_load_hints"
        not in spec.normalized_config(helion.Config.from_dict(base)).config
    )


def test_two_byte_load_assembly() -> None:
    pytest.importorskip("cutlass")
    from helion._compiler.cute.element_policy_loads import _assembly

    assert (
        _assembly("streaming", 16, predicated=False) == "{ ld.global.cs.b16 $0, [$1]; }"
    )
    assert _assembly("l1_l2_last", 16, predicated=False) == (
        "{ .reg .b64 pol; createpolicy.fractional.L2::evict_last.b64 pol, 1.0; "
        "ld.global.L1::evict_last.L2::cache_hint.b16 $0, [$1], pol; }"
    )
    # A predicated load zeroes its 32-bit register and loads under ``@p``.
    assert _assembly("last", 16, predicated=True) == (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; "
        "@p ld.global.L1::evict_last.b16 $0, [$1]; }"
    )


def test_helpers_build_valid_ir_for_two_byte_elements() -> None:
    pytest.importorskip("cutlass")
    text, loads = element_load_helper_ir(("BFloat16", "Float16", "Int16"))
    assert text.count("llvm.inline_asm has_side_effects") == loads
    assert text.count("~{memory}") == loads
    # Bits only: no numeric conversion of the loaded value.
    assert "sitofp" not in text and "uitofp" not in text and "fptrunc" not in text
    # The elements load into a 32-bit register.
    assert text.count(".b16 $0, [$1]") == loads
    assert text.count("@p ld.global") == loads // 2
    assert "mov.b16" not in text


# -- GPU numerics --------------------------------------------------------------


# dtype x size x layout x policy.  Every value of each axis occurs, not
# every pair: each of the three 2-byte element types runs with a partial,
# masked last tile (1000) and without one (1024) on two of the three view
# layouts, and each of the six policies runs once.
_GPU_CASES = [
    (torch.bfloat16, 1000, "contiguous", "l2_last"),
    (torch.bfloat16, 1024, "offset", "last"),
    (torch.float16, 1000, "offset", "l1_l2_first"),
    (torch.float16, 1024, "permuted", "streaming"),
    (torch.int16, 1000, "permuted", "l1_l2_last"),
    (torch.int16, 1024, "contiguous", "first"),
]


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
@pytest.mark.parametrize("dtype,n,layout,policy", _GPU_CASES)
def test_hinted_two_byte_loads_match_eager(
    dtype: torch.dtype, n: int, layout: str, policy: str
) -> None:
    pairs = adversarial_pairs(dtype, n, layout)
    x = adversarial_pairs(dtype, n, "contiguous")[:, 1].contiguous()
    bound = component_add.bind((pairs, x))
    spec = bound.config_spec
    config = spec.normalized_config(
        helion.Config.from_dict(
            spec.default_config().config
            | {"block_sizes": [256], "load_eviction_policies": [policy, policy]}
            | _KNOB
        )
    )
    assert "_cute_element_policy_load" in bound.to_code(config)
    assert_same_bits(bound.compile_config(config)(pairs, x), pairs[:, 0] + x)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
def test_hinted_padded_pool_gathers_match_the_plain_kernel() -> None:
    # Nine masked gathers per output; the window reaches one element past
    # every border of the 13 x 17 planes, so each tap is masked somewhere.
    generator = torch.Generator().manual_seed(3)
    x = torch.randn((6, 13, 17), generator=generator)
    x[0, 0, 0] = float("nan")
    x[1, 12, 16] = float("inf")
    x = x.to(torch.bfloat16).to(DEVICE)
    bound = _relu_max_pool.bind((x,))
    spec = bound.config_spec
    base = spec.default_config().config | {"block_sizes": [2, 4, 8]}
    slots = spec.load_eviction_policies.length
    hinted = spec.normalized_config(
        helion.Config.from_dict(
            base
            | {"load_eviction_policies": ["", "l1_l2_last"] + [""] * (slots - 2)}
            | _KNOB
        )
    )
    assert bound.to_code(hinted).count("_cute_element_policy_load_if(x.iterator") == 9
    actual = bound.compile_config(hinted)(x)
    plain = bound.compile_config(spec.normalized_config(helion.Config.from_dict(base)))(
        x
    )
    assert_same_bits(actual, plain)
    padded = torch.full((6, 15, 19), float("-inf"), device=DEVICE)
    padded[:, 1:14, 1:18] = torch.relu(x.float())
    windows = padded.unfold(1, 3, 2).unfold(2, 3, 2).reshape(6, 7, 9, 9)
    expected = windows[..., 0]
    for tap in range(1, 9):
        expected = torch.maximum(expected, windows[..., tap])
    assert_same_bits(actual, expected.to(torch.bfloat16))
