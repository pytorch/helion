"""The ``cute_scalar_load_hints`` knob: L2 cache policies on element loads,
one predicated load for a hinted element under a mask, and the search-side
rule that sets the load-hint form knobs.

``load_eviction_policies`` reached the packet loads only: an element load (a
gather, a lane on a strided dim) took ``first`` / ``last`` / ``streaming`` at
4 and 8 bytes, as inline PTX inside the conditional of its mask (a branch),
and dropped the L2 policies.  Under the knob the element loads of a 4- or
8-byte tensor take the first ``l2_last`` / ``l1_l2_first`` / ``l1_l2_last``
its load slots name, and a hinted element load under a mask becomes one
predicated load (``_cute_element_policy_load_if``).  The hints are placed
after every other pass (``hint_element_loads``), on the loads that are still
plain element loads, so no pass that merges element loads into packets sees
a hinted form.  The knob is off by default, so a saved config renders the
code it was tuned with; the search sets it, with ``cute_wide_packet_hints``,
on the configs it decodes (``open_load_hint_forms``).  The GPU numerics are
at the end of the module.
"""

from __future__ import annotations

import ast
from typing import Any

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.search_surface import open_load_hint_forms
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.cache_policy_loads import CachePolicy
from helion._compiler.cute.cache_policy_loads import element_load_hint
from helion._compiler.cute.scalar_policy_loads import hint_element_loads
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
from helion._testing.unfold_kernels import sliding_windows
from helion._testing.unfold_kernels import sliding_windows_reference
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

# Source checks, in the CPU lane: CUDA hidden and native compilation
# forbidden.  The GPU numerics at the end of the module are marked ``cuda``.
pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render(TargetFacts())]

_L2_POLICIES = ("l2_last", "l1_l2_first", "l1_l2_last")
_L1_POLICIES = ("first", "last", "streaming")
_KNOB = {"cute_scalar_load_hints": True}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_pair_sum(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # Masked element gathers through a loaded index:
    # ``out[p, j] = x[p, idx[j]] + x[p, idx[j] + 1]``, each tap only where it
    # is inside the row.
    planes, n = x.shape
    (m,) = idx.shape
    out = torch.empty([planes, m], dtype=x.dtype, device=x.device)
    for tile_p, tile_j in hl.tile([planes, m]):
        col = idx[tile_j]
        inside = ((col >= 0) & (col < n))[None, :]
        first = hl.load(x, [tile_p, col], extra_mask=inside)
        col1 = col + 1
        inside1 = ((col1 >= 0) & (col1 < n))[None, :]
        out[tile_p, tile_j] = first + hl.load(x, [tile_p, col1], extra_mask=inside1)
    return out


# -- what the knob renders -------------------------------------------------


# One fp32 element of the window gather per thread, the three slots a lane
# loop: the staged store (``cute_staged_store``) with element loads left.
_STAGED_WINDOWS = {
    "block_sizes": [64, 4],
    "num_threads": [64, 1],
    "cute_staged_store": True,
}


def _staged_windows_config(bound: Any, **extra: object) -> helion.Config:
    spec = bound.config_spec
    slots = spec.load_eviction_policies.length
    return spec.normalized_config(
        helion.Config.from_dict(
            _STAGED_WINDOWS | {"load_eviction_policies": ["l1_l2_last"] * slots} | extra
        )
    )


def test_a_staged_kernels_element_loads_take_their_hints() -> None:
    # ``hint_element_loads`` runs after ``stage_cta_stores``: the staged
    # store's proof runs the body it is given and takes plain element loads
    # only, so a hinted form in front of it would leave the kernel with its
    # direct stores.  A search sets the hint knob on every config that names
    # a policy; the staged kernel must stage with it too.
    bound = cpu_bind(
        helion.kernel(sliding_windows.fn, backend="cute", static_shapes=True),
        (torch.empty(2, 8, 16), 3, 1),
    )
    staged = bound.to_code(_staged_windows_config(bound))
    both = bound.to_code(_staged_windows_config(bound, **_KNOB))
    assert "_cta_stage_base" in staged and "_policy_load" not in staged
    assert both.count("_cute_element_policy_load_if(flat.iterator + ") == 1
    # The same image, barrier and flush; only the gather's load differs.
    assert both.count("_cta_stage") == staged.count("_cta_stage") > 0
    assert both.count("cute.arch.sync_threads()") == 1
    assert "(out.iterator + " not in both.replace("cute.arch.store(out.iterator + ", "")


@pytest.mark.parametrize("policy", _L2_POLICIES)
def test_a_masked_element_takes_its_l2_policy_in_one_predicated_load(
    policy: str,
) -> None:
    bound = bound_components(torch.float32, 1000)
    off = bound.to_code(policy_config(bound, [policy, ""]))
    on = bound.to_code(policy_config(bound, [policy, ""], **_KNOB))
    # Off: the L2 policy is dropped, as it always was.
    assert "_policy_load" not in off
    for line in load_lines(off):
        assert line.endswith(".load() if mask_0 else cutlass.Float32(0)")
    # On: every element load of the tensor carries the policy its first
    # slot names, each under its mask in one call.
    first, second = load_lines(on)
    for line in (first, second):
        assert "= _cute_element_policy_load_if(pairs.iterator + " in line
        assert line.endswith(f", mask_0, cutlass.Float32, {policy!r})")
        assert " if mask_0 else " not in line
    # Nothing else changes: the store, the arithmetic and the launch.
    assert lines_without_loads(on) == lines_without_loads(off)


def test_the_first_slot_of_the_tensor_that_names_an_l2_policy_decides() -> None:
    bound = bound_components(torch.float32, 1000)
    late = bound.to_code(policy_config(bound, ["", "l1_l2_last"], **_KNOB))
    assert late.count(", mask_0, cutlass.Float32, 'l1_l2_last')") == 2
    both = bound.to_code(policy_config(bound, ["l1_l2_first", "l2_last"], **_KNOB))
    assert both.count(", mask_0, cutlass.Float32, 'l1_l2_first')") == 2
    assert "'l2_last'" not in both


def test_an_l1_priority_under_a_mask_becomes_the_predicated_load() -> None:
    # The forms every config renders: an L1 priority on a 4-byte element is
    # the unpredicated helper inside the mask's conditional.  Under the knob
    # that load is predicated; the tensor's other load, still plain, takes
    # the L2 policy the second slot names.
    bound = bound_components(torch.float32, 1000)
    off = bound.to_code(policy_config(bound, ["last", "l2_last"]))
    first, second = load_lines(off)
    assert first.startswith("load = _cute_scalar_policy_load(pairs.iterator + ")
    assert first.endswith(
        ", cutlass.Float32, 'last') if mask_0 else cutlass.Float32(0)"
    )
    assert second.endswith(".load() if mask_0 else cutlass.Float32(0)")
    on = bound.to_code(policy_config(bound, ["last", "l2_last"], **_KNOB))
    first, second = load_lines(on)
    assert first.startswith("load = _cute_element_policy_load_if(pairs.iterator + ")
    assert first.endswith(", mask_0, cutlass.Float32, 'last')")
    assert second.endswith(", mask_0, cutlass.Float32, 'l2_last')")
    assert lines_without_loads(on) == lines_without_loads(off)


def test_unmasked_elements_take_the_unpredicated_load_of_their_own_tensor() -> None:
    bound = bound_components(torch.float32, 1024, component_add)
    off = bound.to_code(policy_config(bound, ["l1_l2_last", ""]))
    on = bound.to_code(policy_config(bound, ["l1_l2_last", ""], **_KNOB))
    strided, contiguous = load_lines(on)
    assert strided.startswith("load = _cute_element_policy_load(pairs.iterator + ")
    assert strided.endswith(", cutlass.Float32, 'l1_l2_last')")
    # The other tensor's slot names nothing: its load is the plain one.
    assert contiguous == load_lines(off)[1]
    assert contiguous.startswith("load_1 = (x.iterator + ")
    assert "_cute_element_policy_load_if" not in on
    assert lines_without_loads(on) == lines_without_loads(off)


@pytest.mark.parametrize("policies", [["l2_last", ""], ["l2_last", "l1_l2_last"]])
def test_a_hint_keeps_no_element_loads_from_merging(policies: list[str]) -> None:
    # The two unmasked components are one 8-byte packet (the literal-offset
    # merge reads plain element loads).  The hint is placed after that pass,
    # so the packet forms as it does without a policy and no element load
    # is left to hint.
    bound = bound_components(torch.float32, 1024)
    plain = bound.to_code(policy_config(bound, ["", ""]))
    assert "_helion_literal_vec" in plain
    assert bound.to_code(policy_config(bound, policies, **_KNOB)) == plain


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.int16, torch.int8, torch.bool, torch.uint8, torch.uint16],
)
def test_narrower_elements_take_no_hint(dtype: torch.dtype) -> None:
    bound = bound_components(dtype, 1000)
    policies = ["last", "l2_last"]
    off = bound.to_code(policy_config(bound, policies))
    assert bound.to_code(policy_config(bound, policies, **_KNOB)) == off
    assert "_policy_load" not in off


def test_the_read_only_packet_policy_stays_a_packet_form() -> None:
    bound = bound_components(torch.float32, 1000)
    plain = bound.to_code(policy_config(bound, ["", ""], **_KNOB))
    nc = bound.to_code(policy_config(bound, ["nc_l2_first", "nc_l2_first"], **_KNOB))
    assert nc == plain


def _element_hint(dtype: torch.dtype, policy: str) -> str | None:
    return element_load_hint(
        dtype, CachePolicy.tunable(policy), l2_elements=True, narrow_elements=False
    )


def test_the_hint_a_slot_gives_its_tensors_element_loads() -> None:
    for policy in _L2_POLICIES:
        for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
            assert _element_hint(dtype, policy) == policy
        # Narrower elements, bools and complex pairs take none.
        for dtype in (torch.bfloat16, torch.int16, torch.int8, torch.bool):
            assert _element_hint(dtype, policy) is None
        assert _element_hint(torch.complex64, policy) is None
    # The hints with a placeholder of their own, the packet-only policy and
    # no policy record nothing.
    for policy in (*_L1_POLICIES, "nc_l2_first", ""):
        assert _element_hint(torch.float32, policy) is None


# -- the pass on its own ---------------------------------------------------

_HINTS = {"x": ("cutlass.Float32", "l2_last"), "w": ("cutlass.Int64", "l1_l2_last")}


def _hint(body: str, parameters: str = "x, w, y") -> str:
    definition = ast.parse(f"def kernel({parameters}):\n    {body}").body[0]
    assert isinstance(definition, ast.FunctionDef)
    return "\n".join(
        ast.unparse(statement) for statement in hint_element_loads(definition, _HINTS)
    )


def test_plain_loads_of_a_hinted_tensor_take_its_policy() -> None:
    assert _hint("a = (x.iterator + i * s).load()") == (
        "a = _cute_element_policy_load(x.iterator + i * s, cutlass.Float32, 'l2_last')"
    )
    assert _hint("a = (w.iterator + i - 1).load() + x.iterator.load()") == (
        "a = _cute_element_policy_load(w.iterator + i - 1, cutlass.Int64, 'l1_l2_last')"
        " + _cute_element_policy_load(x.iterator, cutlass.Float32, 'l2_last')"
    )
    # A load in an ``if`` statement is under a branch as it is.
    assert _hint("if m:\n        a = (x.iterator + i).load()") == (
        "if m:\n    a = _cute_element_policy_load(x.iterator + i, cutlass.Float32, 'l2_last')"
    )
    for untouched in (
        # Another tensor, a store, a packet and a hinted placeholder.
        "a = (y.iterator + i).load()",
        "(x.iterator + i).store(a)",
        "a = cute.arch.load(x.iterator + i, ir.VectorType.get([4], cutlass.Float32.mlir_type))",
        "a = cute.arch.load(x.iterator + i, cutlass.Float32, cop='cs')",
        # Pointers that are not the tensor's iterator plus an offset: a named
        # or selected pointer, a cast one, an offset naming another iterator.
        "a = p.load()",
        "a = (x.iterator + i if m else x.iterator + 0).load()",
        "a = cute.make_ptr(cutlass.Uint8, x.iterator.toint()).load()",
        "a = (x.iterator + (y.iterator - y.iterator)).load()",
        "a = (x.iterator * 2).load()",
    ):
        assert _hint(untouched) == untouched
    # Only a parameter the body never rebinds names its tensor.
    plain = "a = (x.iterator + i).load()"
    assert _hint(plain, parameters="w, y") == plain
    rebound = f"x = other\n    {plain}"
    assert _hint(rebound) == rebound.replace("\n    ", "\n")
    looped = f"for x in items:\n        {plain}"
    assert _hint(looped) == looped.replace("\n        ", "\n    ")


def test_only_the_masked_zero_form_is_predicated() -> None:
    pointer = "x.iterator + cutlass.Int32(i) * cutlass.Int32(x.layout.stride[1])"
    assert (
        _hint(f"a = ({pointer}).load() if m and n < 4 else cutlass.Float32(0)")
        == f"a = _cute_element_policy_load_if({pointer}, m and n < 4, cutlass.Float32, 'l2_last')"
    )
    # The lowered placeholder of an L1 priority or the streaming operator,
    # of any tensor: its type and policy are in the call.
    lowered = "_cute_scalar_policy_load(y.iterator + i, cutlass.Float64, 'streaming')"
    assert _hint(f"a = {lowered} if m else cutlass.Float64(0)") == (
        "a = _cute_element_policy_load_if(y.iterator + i, m, cutlass.Float64, 'streaming')"
    )
    # Any other alternative is no masked load: a nonzero fill, a zero of
    # another type, a second load.  A lowered load keeps its conditional, a
    # plain load inside a conditional stays plain (a hinted one would be
    # inline PTX under a branch).
    for orelse in ("cutlass.Float32(1)", "cutlass.Int32(0)", "q.load()", "0"):
        plain = f"a = (x.iterator + i).load() if m else {orelse}"
        assert _hint(plain) == plain
        kept = f"a = {lowered} if m else {orelse}"
        assert _hint(kept) == kept
    for nested in (
        "a = cutlass.Float32(0) if m else (x.iterator + i).load()",
        "a = (x.iterator + i).load() + b if m else cutlass.Float32(0)",
        # A masked load in a branch of another conditional.
        "a = b if n else (x.iterator + i).load() if m else cutlass.Float32(0)",
        f"a = ({lowered} if m else cutlass.Float64(0)) if n else b",
    ):
        assert _hint(nested) == nested
    # The condition is outside the conditional's branches.
    assert _hint("a = b if (x.iterator + i).load() > 0 else c") == (
        "a = b if _cute_element_policy_load(x.iterator + i, cutlass.Float32, 'l2_last')"
        " > 0 else c"
    )
    # So is the mask of a predicated load: a hinted tensor's load in it takes
    # its hint too, a plain load's mask and a lowered load's alike.
    mask = "(w.iterator + j).load() > 0"
    hinted_mask = (
        "_cute_element_policy_load(w.iterator + j, cutlass.Int64, 'l1_l2_last') > 0"
    )
    assert _hint(f"a = (x.iterator + i).load() if {mask} else cutlass.Float32(0)") == (
        f"a = _cute_element_policy_load_if(x.iterator + i, {hinted_mask}, "
        "cutlass.Float32, 'l2_last')"
    )
    assert _hint(f"a = {lowered} if {mask} else cutlass.Float64(0)") == (
        f"a = _cute_element_policy_load_if(y.iterator + i, {hinted_mask}, "
        "cutlass.Float64, 'streaming')"
    )


def test_only_a_plain_address_is_evaluated_outside_its_mask() -> None:
    # The predicated load evaluates its address for masked-off elements
    # too: an address that reads memory (a load, an element of a tensor: the
    # form a gather through a loaded index takes), calls a helper or selects
    # may only run where the mask holds.
    for offset in (
        "cutlass.Int32(q.load()) * s",
        "table[i]",
        "table[i, j]",
        "x.layout.stride[i]",
        "helper(i)",
        "(i if m else j)",
        "cutlass.Int32(i, signed=True)",
        "cute.arch.thread_idx(0)[0]",
        "cute.arch.thread_idx()[i]",
    ):
        plain = f"a = (x.iterator + {offset}).load() if m else cutlass.Float32(0)"
        assert _hint(plain) == plain
        lowered = (
            f"a = _cute_scalar_policy_load(x.iterator + {offset}, cutlass.Float32, 'last') "
            "if m else cutlass.Float32(0)"
        )
        assert _hint(lowered) == lowered
    # Names, integer casts, constant subscripts of layout attributes and the
    # launch coordinates are plain arithmetic.
    for offset in (
        "-i",
        "cutlass.Int32(cute.arch.thread_idx()[0] + i) * cutlass.Int32(x.layout.stride[1])",
        "cutlass.Int64(i) * w.shape[0]",
        "cute.arch.block_idx()[1] * 4",
    ):
        assert _hint(
            f"a = (x.iterator + {offset}).load() if m else cutlass.Float32(0)"
        ) == (
            f"a = _cute_element_policy_load_if(x.iterator + {offset}, m, "
            "cutlass.Float32, 'l2_last')"
        )


def test_a_gather_through_a_loaded_index_is_predicated_on_its_name() -> None:
    # The loaded index is assigned to a name above the gather, where its own
    # mask holds; the gather's address reads that name.
    arguments = (torch.empty((8, 100)), torch.empty(77, dtype=torch.int64))
    bound = cpu_bind(_gather_pair_sum, arguments)
    spec = bound.config_spec
    code = bound.to_code(
        spec.normalized_config(
            helion.Config.from_dict(
                spec.default_config().config
                | {
                    "block_sizes": [4, 32],
                    "load_eviction_policies": ["l2_last"]
                    * spec.load_eviction_policies.length,
                }
                | _KNOB
            )
        )
    )
    gathers = [line for line in code.splitlines() if "(x.iterator" in line]
    assert len(gathers) == 2
    for line in gathers:
        assert "= _cute_element_policy_load_if(x.iterator + " in line
        assert line.endswith(", cutlass.Float32, 'l2_last')")
        assert ".load()" not in line and "idx" not in line
    # The index itself is an 8-byte element with the same policy.
    (index,) = (line for line in code.splitlines() if "(idx.iterator" in line)
    assert "col = _cute_element_policy_load_if(idx.iterator + " in index
    assert index.endswith(", cutlass.Int64, 'l2_last')")


def test_element_load_assembly() -> None:
    pytest.importorskip("cutlass")
    from helion._compiler.cute.element_policy_loads import _assembly

    assert (
        _assembly("streaming", 32, predicated=False) == "{ ld.global.cs.b32 $0, [$1]; }"
    )
    # An L2 family creates its descriptor; a predicated load zeroes its
    # register and loads under ``@p``.
    assert _assembly("l1_l2_last", 32, predicated=False) == (
        "{ .reg .b64 pol; createpolicy.fractional.L2::evict_last.b64 pol, 1.0; "
        "ld.global.L1::evict_last.L2::cache_hint.b32 $0, [$1], pol; }"
    )
    assert _assembly("last", 32, predicated=True) == (
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; "
        "@p ld.global.L1::evict_last.b32 $0, [$1]; }"
    )
    assert _assembly("l2_last", 64, predicated=True) == (
        "{ .reg .pred p; .reg .b64 pol; "
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0; "
        "setp.ne.b32 p, $2, 0; mov.b64 $0, 0; "
        "@p ld.global.L2::cache_hint.b64 $0, [$1], pol; }"
    )


def test_helpers_build_valid_ir_for_every_width_and_form() -> None:
    pytest.importorskip("cutlass")
    text, loads = element_load_helper_ir(("Float32", "Int32", "Float64", "Int64"))
    assert text.count("llvm.inline_asm has_side_effects") == loads
    assert text.count("~{memory}") == loads
    # Bits only: no numeric conversion of the loaded value.
    assert "sitofp" not in text and "uitofp" not in text and "fptrunc" not in text
    # Half the loads are predicated, half carry an L2 descriptor.
    assert text.count("@p ld.global") == loads // 2
    assert text.count("createpolicy.fractional.L2::evict_last") == loads // 2
    assert text.count(".b32 $0, [$1]") == text.count(".b64 $0, [$1]") == loads // 2


# -- the search side ---------------------------------------------------------

_BOTH = {"cute_wide_packet_hints": True, "cute_scalar_load_hints": True}


@pytest.mark.parametrize(
    "policies,expected",
    [
        (["", ""], {}),
        (["last", ""], {"cute_scalar_load_hints": True}),
        (["nc_l2_first", "streaming"], {"cute_scalar_load_hints": True}),
        (["nc_l2_first", ""], {}),
        (["", "l2_last"], _BOTH),
        ("l1_l2_first", _BOTH),
        ("", {}),
    ],
)
def test_decoded_configs_take_the_forms_their_policies_need(
    policies: object, expected: dict[str, bool]
) -> None:
    config: dict[str, object] = {"load_eviction_policies": policies}
    open_load_hint_forms(config)
    assert config == {"load_eviction_policies": policies, **expected}


def test_search_configs_open_the_hint_forms() -> None:
    bound = bound_components(torch.float32, 1000)
    spec = bound.config_spec
    with bound.env:
        generation = ConfigGeneration(spec)
        saved = policy_config(bound, ["l1_l2_last", "last"])
        # A saved config normalizes to itself: no knob appears.
        assert not set(_BOTH) & set(saved.config)
        # Through the search's flat layout it takes both forms.
        flat = generation.flatten(saved)
        decoded = generation.unflatten(flat)
        assert decoded.config["cute_wide_packet_hints"] is True
        assert decoded.config["cute_scalar_load_hints"] is True
        # The flat layout holds neither knob, and the round trip is stable.
        assert not set(_BOTH) & set(spec._flat_fields())
        assert generation.flatten(decoded) == flat
        # A config without a policy decodes without either.
        plain = generation.unflatten(generation.flatten(policy_config(bound, ["", ""])))
        assert not set(_BOTH) & set(plain.config)
    # The search times the hinted element loads; the saved config keeps the
    # code it was tuned with.
    searched = bound.to_code(decoded)
    assert searched.count("_cute_element_policy_load_if(") == 2
    assert "_cute_element_policy_load" not in bound.to_code(saved)


def test_overridden_policies_decide_the_forms() -> None:
    # ``autotune_config_overrides`` replaces the policies a config was
    # decoded with.  The forms follow the policies it runs with, whatever
    # the flat config held, so every candidate renders them the same way.
    bound = bound_components(torch.float32, 1000)
    spec = bound.config_spec
    hinted = ["l2_last", "l2_last"]
    with bound.env:
        generation = ConfigGeneration(spec)
        flats = [
            generation.flatten(policy_config(bound, policies))
            for policies in (["l1_l2_last", "last"], ["nc_l2_first", ""], ["", ""])
        ]
        fixed = ConfigGeneration(spec, overrides={"load_eviction_policies": hinted})
        candidates = [fixed.unflatten(list(flat)) for flat in flats]
        for decoded in candidates:
            assert decoded.config["load_eviction_policies"] == hinted
            assert {key: decoded.config.get(key) for key in _BOTH} == _BOTH
        # Overridden to no policy, a config carries no form.
        plain = ConfigGeneration(spec, overrides={"load_eviction_policies": ["", ""]})
        for flat in flats:
            assert not set(_BOTH) & set(plain.unflatten(list(flat)).config)
        # A form the overrides name themselves stays as they set it.
        chosen = ConfigGeneration(
            spec,
            overrides={
                "load_eviction_policies": hinted,
                "cute_scalar_load_hints": False,
            },
        )
        for flat in flats:
            decoded = chosen.unflatten(list(flat))
            assert decoded.config["cute_scalar_load_hints"] is False
            assert decoded.config["cute_wide_packet_hints"] is True
    (code,) = {bound.to_code(decoded) for decoded in candidates}
    assert code.count("_cute_element_policy_load_if(") == 2


@pytest.mark.parametrize("key", sorted(_BOTH))
def test_the_knobs_are_booleans(key: str) -> None:
    bound = bound_components(torch.float32, 1000)
    spec = bound.config_spec
    base = spec.default_config().config
    with pytest.raises(exc.InvalidConfig, match=f"{key} must be a boolean"):
        spec.normalize(helion.Config.from_dict(base | {key: 1}))
    for value in (False, True):
        normalized = spec.normalized_config(
            helion.Config.from_dict(base | {key: value})
        )
        assert normalized.config[key] is value
    assert key not in spec.normalized_config(helion.Config.from_dict(base)).config


def _simplified(bound: Any, config: helion.Config) -> tuple[helion.Config, int]:
    # The selected config as the search simplifies it, and the renders that
    # took: the result must render the code of ``config``.
    code = bound.to_code(config)
    asked: list[helion.Config] = []

    def same_source(candidate: helion.Config) -> bool:
        asked.append(candidate)
        return bound.to_code(candidate) == code

    result = CuteBackend().simplified_selected_config(
        bound.config_spec, config, same_source
    )
    assert bound.to_code(result) == code
    return result, len(asked)


def test_selected_config_drops_the_knobs_its_code_does_not_need() -> None:
    bound = bound_components(torch.float32, 1000)
    # An element load carries its L2 policy through the scalar knob alone:
    # the wide-packet knob goes, the scalar one and the policy stay.
    live, _renders = _simplified(bound, policy_config(bound, ["l2_last", ""], **_BOTH))
    assert live.config["load_eviction_policies"] == ["l2_last", ""]
    assert live.config["cute_scalar_load_hints"] is True
    assert "cute_wide_packet_hints" not in live.config
    # A policy no load form carries (the read-only packet policy on element
    # loads) is dead, and with it both knobs: one render answers.
    dead, renders = _simplified(
        bound, policy_config(bound, ["nc_l2_first", "nc_l2_first"], **_BOTH)
    )
    assert dead.config["load_eviction_policies"] == ["", ""]
    assert not set(_BOTH) & set(dead.config)
    assert renders == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _spelled_hint_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # An fp32 element whose hint the kernel spells out (no load slot) beside
    # a bf16 element, the only load slot.
    n = x.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        a = hl.load(x, [tile, 0], eviction_policy="evict_last")
        out[tile] = a + y[tile, 0].to(torch.float32)
    return out


def test_selected_config_drops_a_dead_policy_beside_a_live_knob() -> None:
    # The scalar knob predicates the masked element the kernel hints itself,
    # so it stays, while the slot's ``l2_last`` reaches no load (a 2-byte
    # element takes none under this knob): clearing every policy with the
    # knobs dropped is another source, clearing them with the knob kept is
    # not, and the selection names no policy.
    bound = cpu_bind(
        _spelled_hint_add,
        (torch.empty((1000, 2)), torch.empty((1000, 2), dtype=torch.bfloat16)),
    )
    assert bound.config_spec.load_eviction_policies.length == 1
    config = policy_config(bound, ["l2_last"], **_KNOB)
    code = bound.to_code(config)
    assert "_cute_element_policy_load_if(x.iterator" in code
    assert code != bound.to_code(policy_config(bound, [""]))
    selected, _renders = _simplified(bound, config)
    assert selected.config["load_eviction_policies"] == [""]
    assert selected.config["cute_scalar_load_hints"] is True


# -- GPU numerics --------------------------------------------------------------


# dtype x size x layout x policy.  Every value of each axis occurs, not
# every pair: each element type the helpers load (4 and 8 bytes, float and
# integer) runs twice, on two sizes (1000 and 1001 end in a partial, masked
# tile; 1024 does not), two of the three view layouts and two policies, and
# the list holds each L2 policy twice and (the predicated form of the older
# hints) an L1 priority and the streaming operator once.
_GPU_CASES = [
    (torch.float32, 1000, "permuted", "l2_last"),
    (torch.float32, 1024, "offset", "l1_l2_first"),
    (torch.int32, 1000, "offset", "l1_l2_last"),
    (torch.int32, 1024, "contiguous", "l2_last"),
    (torch.int64, 1000, "contiguous", "l1_l2_first"),
    (torch.int64, 1001, "permuted", "last"),
    (torch.float64, 1001, "offset", "streaming"),
    (torch.float64, 1024, "permuted", "l1_l2_last"),
]


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
@pytest.mark.parametrize("dtype,n,layout,policy", _GPU_CASES)
def test_hinted_element_loads_match_eager(
    dtype: torch.dtype, n: int, layout: str, policy: str
) -> None:
    pairs = adversarial_pairs(dtype, n, layout)
    x = adversarial_pairs(dtype, n, "contiguous")[:, 1].contiguous()
    assert pairs.shape == (n, 2)
    bound = component_add.bind((pairs, x))
    spec = bound.config_spec
    config = spec.normalized_config(
        helion.Config.from_dict(
            spec.default_config().config
            | {"block_sizes": [256], "load_eviction_policies": [policy, policy]}
            | _KNOB
        )
    )
    code = bound.to_code(config)
    assert "_cute_element_policy_load" in code
    assert_same_bits(bound.compile_config(config)(pairs, x), pairs[:, 0] + x)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
def test_predicated_gathers_read_nothing_outside_their_mask() -> None:
    # Index entries far outside the row (and negative ones) are masked off:
    # the predicated load evaluates their addresses, a gigabyte and more
    # past the tensor, and must not read them.
    planes, n, m = 5, 100, 77
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((planes, n), generator=generator).to(DEVICE)
    idx = torch.randint(-3, n + 3, (m,), generator=generator)
    idx[::7] = 10**9
    idx[3::11] = -(10**9)
    idx = idx.to(DEVICE)
    bound = _gather_pair_sum.bind((x, idx))
    spec = bound.config_spec
    base = spec.default_config().config | {"block_sizes": [4, 32]}
    slots = spec.load_eviction_policies.length
    hinted = spec.normalized_config(
        helion.Config.from_dict(
            base | {"load_eviction_policies": ["l1_l2_last"] * slots} | _KNOB
        )
    )
    assert bound.to_code(hinted).count("_cute_element_policy_load_if(x.iterator") == 2
    expected = torch.zeros((planes, m), device=DEVICE)
    for tap in range(2):
        col = idx + tap
        inside = (col >= 0) & (col < n)
        value = x[:, col.clamp(0, n - 1)]
        expected += torch.where(inside[None, :], value, torch.zeros_like(value))
    actual = bound.compile_config(hinted)(x, idx)
    plain = bound.compile_config(spec.normalized_config(helion.Config.from_dict(base)))(
        x, idx
    )
    assert_same_bits(actual, plain)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0), reason="L2 cache policy loads need sm_90+")
def test_a_staged_kernel_with_hinted_gathers_matches_eager() -> None:
    # The staged store and the element hints together, as a search renders
    # them: 1968 elements in tiles of 64 (thirty staged CTAs and a partial
    # one with direct stores), the padded slots masked off.
    batch, seq, channels, window, pad = 3, 41, 16, 3, 1
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((batch, seq, channels), generator=generator).to(DEVICE)
    bound = helion.kernel(sliding_windows.fn, backend="cute", static_shapes=True).bind(
        (x, window, pad)
    )
    config = _staged_windows_config(bound, **_KNOB)
    code = bound.to_code(config)
    assert "_cta_stage_base" in code and "_cute_element_policy_load_if(" in code
    actual = bound.compile_config(config)(x, window, pad)
    expected = sliding_windows_reference(x.view(-1), seq, channels, window, pad)
    assert_same_bits(actual, expected)
