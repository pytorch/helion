from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_postscale import POSTSCALE_ATTENTION_CASES
from helion._testing.cute_paged_postscale import postscale_attention_fixture

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(
    form="per_key",
    rows=4,
    key=-6,
    value=-6,
    reference=False,
    base2=False,
    reverse=False,
    lse=True,
):
    kernel, args = postscale_attention_fixture(
        form,
        rows,
        lse,
        key=key,
        value=value,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return bound, program


@pytest.mark.parametrize(
    "form,rows,key,value,reference,base2,reverse,schedule,occupancy,fast,lse",
    POSTSCALE_ATTENTION_CASES,
)
def test_postdecode_power_of_two_source(
    form, rows, key, value, reference, base2, reverse, schedule, occupancy, fast, lse
):
    bound, program = _bound(form, rows, key, value, reference, base2, reverse, lse)
    plan = match_paged_attention(program)
    assert plan is not None
    assert (
        plan.key.post_scale_log2,
        plan.value.post_scale_log2,
        plan.output_scale_log2,
    ) == (key, value, -value)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(rows * 8)],
        cute_mma_dag_pass_warps=schedule,
        cute_min_blocks_per_mp=occupancy,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert ("paged_attention_paired_body(" in source) == (
        schedule == "attention_paired"
    )
    for name, exponent in (
        ("key_post_scale_log2", key),
        ("value_post_scale_log2", value),
        ("output_scale_log2", -value),
    ):
        assert (f"{name}={exponent}" in source) == (exponent != 0)


@pytest.mark.parametrize(
    "mutation",
    ["factor", "range", "before_cast", "dtype", "output_factor", "two_factors"],
)
def test_reject_different_postdecode_rounding(mutation):
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    post = plan.key.value
    assert post.target is torch.ops.aten.mul.Tensor
    decoded, factor = post.args
    if mutation in ("factor", "range"):
        post.args = (decoded, 0.03 if mutation == "factor" else 2.0**-23)
    elif mutation == "before_cast":
        product = decoded.args[0]
        post.args = (product, factor)
    elif mutation == "dtype":
        post.meta["val"] = post.meta["val"].to(torch.float32)
    else:
        reciprocal = next(
            node
            for node in program.stores[0].value.graph.nodes
            if node.target is torch.ops.aten.reciprocal.default
        )
        (identity,) = reciprocal.users
        (scaled,) = identity.users
        assert scaled.target is torch.ops.aten.mul.Tensor
        if mutation == "output_factor":
            scaled.args = (identity, 3.0)
        else:
            identity.args = (reciprocal, 2.0)
    assert match_paged_attention(program) is None


def test_guarded_postscale_lattice_including_subnormals():
    codes = torch.arange(65536, dtype=torch.int32)
    raw = (codes % 256).to(torch.uint8).view(torch.int8).float()
    scales = codes // 256
    decoded = (raw * torch.exp2(scales.float() - 127)).bfloat16()
    # All dependent lattice endpoints share tensors; this does not compile kernels.
    for exponent in range(-22, 1):
        safe = scales <= 247
        source = (decoded * 2.0**exponent).bfloat16()
        folded = torch.ldexp(raw, scales - 127 + exponent).bfloat16()
        assert torch.equal(
            source[safe].view(torch.int16), folded[safe].view(torch.int16)
        )
        underflow = scales <= -14 - exponent
        expected_sign = torch.where(raw < 0, -32768, 0).to(torch.int16)
        assert torch.equal(
            source[underflow].view(torch.int16), expected_sign[underflow]
        )
    # Moving the postscale before the first BF16 rounding is not legal globally.
    source = (decoded * 2.0**-6).bfloat16()
    folded = torch.ldexp(raw, scales - 133).bfloat16()
    witness = (raw == -128) & (scales == 248)
    assert source[witness].isneginf().all()
    assert folded[witness].isfinite().all()


def test_commuted_postscale_and_independent_output_factor():
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    for operand in (plan.key, plan.value):
        operand.value.args = tuple(reversed(operand.value.args))
    reciprocal = next(
        node
        for node in program.stores[0].value.graph.nodes
        if node.target is torch.ops.aten.reciprocal.default
    )
    (identity,) = reciprocal.users
    (scaled,) = identity.users
    scaled.args = tuple(reversed(scaled.args))
    plan = match_paged_attention(program)
    assert plan is not None
    assert (plan.key.post_scale_log2, plan.value.post_scale_log2) == (-6, -6)
    assert plan.output_scale_log2 == 6
    # The compiler must preserve an uncompensated source as written, rather
    # than infer an output factor from the representation of V.
    scaled.args = (1.0, identity)
    plan = match_paged_attention(program)
    assert plan is not None and plan.output_scale_log2 == 0


@pytest.mark.parametrize(
    "dtype,exponent,accepted",
    [
        (torch.float16, 15, True),
        (torch.float16, 16, False),
        (torch.float16, 22, False),
        (torch.bfloat16, 22, True),
        (torch.float32, 22, True),
        (torch.float64, 22, True),
    ],
)
def test_output_factor_preserves_scalar_constructor_dtype(dtype, exponent, accepted):
    _, original = _bound()
    program = copy_paged_attention_program(original)
    graph = program.stores[0].value.graph
    reciprocal = next(
        node for node in graph.nodes if node.target is torch.ops.aten.reciprocal.default
    )
    (identity,) = reciprocal.users
    (scaled,) = identity.users
    with graph.inserting_before(scaled):
        factor = graph.call_function(
            torch.ops.aten.scalar_tensor.default,
            (2.0**exponent,),
            {"dtype": dtype},
        )
    factor.meta["val"] = torch.scalar_tensor(2.0**exponent, dtype=dtype)
    scaled.args = (identity, factor)
    plan = match_paged_attention(program)
    if accepted:
        assert plan is not None and plan.output_scale_log2 == exponent
    else:
        assert plan is None


@pytest.mark.parametrize(
    "dtype,literal,expected",
    [
        (torch.float16, 2.0**-22, -22),
        (torch.bfloat16, 2.0**-14, -14),
        (torch.float32, 2.0**-13, -13),
        (torch.float64, 2.0**-6, -6),
        (torch.float16, 2.0**-25, None),
        (torch.float16, 2.0**16, None),
        (torch.float32, 2.0**-150, None),
        (torch.float32, float("inf"), None),
        (torch.float64, 2.0**-23, None),
        (torch.float32, 0.0, None),
        (torch.float32, -0.0, None),
        (torch.float32, float("nan"), None),
    ],
)
def test_postdecode_factor_preserves_typed_extremes(dtype, literal, expected):
    # Dependent boundaries, not a dtype/exponent product. All supported
    # constructor dtypes represent [-22,0] exactly. Literals that would round
    # to zero/Inf in their constructor are already outside that interval and
    # must decline, along with the finite -23 boundary, zero signs, and NaN.
    _, original = _bound()
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None
    post = plan.value.value
    decoded, _ = post.args
    graph = post.graph
    with graph.inserting_before(post):
        factor = graph.call_function(
            torch.ops.aten.scalar_tensor.default, (literal,), {"dtype": dtype}
        )
    factor.meta["val"] = torch.scalar_tensor(literal, dtype=dtype)
    post.args = (decoded, factor)
    actual = match_paged_attention(program)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None and actual.value.post_scale_log2 == expected


def test_packed_gap_scale_bits_preserve_independent_lane_rounding(monkeypatch):
    from helion._compiler.cute import _paged_attention_runtime as runtime

    # Execute the actual scale-bit preparation with CPU int32 tensors. This
    # preserves signed shifts/wrap while avoiding GPU compilation in this lane.
    monkeypatch.setattr(
        runtime,
        "cutlass",
        SimpleNamespace(
            Int32=lambda value: torch.as_tensor(value, dtype=torch.int32),
            max=lambda value, bound: value.clamp_min(bound),
            min=lambda value, bound: value.clamp_max(bound),
        ),
    )
    prepare = inspect.unwrap(runtime._postscale_pair_bits)
    codes = torch.arange(248 * 256, dtype=torch.int32)
    first_scale = codes // 256
    second_scale = (first_scale * 73 + 29) % 248
    scales = torch.stack((first_scale, second_scale), dim=1)
    first_raw = (codes % 256).to(torch.uint8).view(torch.int8)
    second_raw = ((codes * 37 + 11) % 256).to(torch.uint8).view(torch.int8)
    raw = torch.stack((first_raw, second_raw), dim=1).float()
    packed_scales = first_scale | (second_scale << 8)
    decoded = torch.ldexp(raw, scales - 127).bfloat16()
    # Every source byte/scale pair occurs in each lane for every dependent
    # postscale. Affine high-lane permutations expose cross-byte borrows.
    for exponent in range(-22, -6):
        native, multipliers = prepare(packed_scales, exponent)
        # The selector's unused upper bits may be arbitrary and signed.
        for high_bits in (0xA5A50000, 0xFFFF0000):
            dirty = packed_scales | torch.tensor(high_bits, dtype=torch.int64).int()
            dirty_native, dirty_multipliers = prepare(dirty, exponent)
            assert torch.equal(dirty_native, native)
            assert torch.equal(dirty_multipliers, multipliers)
        native_scales = torch.stack((native & 255, (native >> 8) & 255), dim=1)
        factor_bits = torch.stack(
            (multipliers & 65535, (multipliers >> 16) & 65535), dim=1
        ).to(torch.int16)
        factors = factor_bits.view(torch.bfloat16)
        assert ((factor_bits.int() & 0x7F80) != 0).all()
        # Expected values keep the original BF16 cast before the postmultiply.
        expected = decoded * 2.0**exponent
        converted = torch.ldexp(raw / 64, native_scales - 127).bfloat16()
        actual = converted * factors
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_native_scale_word_lower_bound_preserves_byte_independence(monkeypatch):
    from helion._compiler.cute import _paged_attention_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "cutlass",
        SimpleNamespace(
            Uint32=lambda value: torch.as_tensor(value, dtype=torch.int64) & 0xFFFFFFFF,
            const_expr=bool,
        ),
    )
    at_least = inspect.unwrap(runtime._scale_word_at_least)
    pairs = torch.arange(65536, dtype=torch.int64)
    first, second = pairs & 255, pairs >> 8
    # Every adjacent byte pair, plus a lone veto in each position, exercises
    # borrow propagation and high-bit bytes without compiling a kernel.
    repeated = first | (second << 8) | (first << 16) | (second << 24)
    byte_values = torch.arange(256, dtype=torch.int64)
    for minimum in range(1, 17):
        assert torch.equal(
            at_least(repeated, minimum), (first >= minimum) & (second >= minimum)
        )
        for background in (0, minimum - 1, minimum, 127, 128, 247, 248, 255):
            for position in range(4):
                shift = position * 8
                word = (background * 0x01010101 & ~(255 << shift)) | (
                    byte_values << shift
                )
                expected = (byte_values >= minimum) & (background >= minimum)
                assert torch.equal(at_least(word, minimum), expected)


def test_owned_scale_group_word_ignores_neighbor_groups(monkeypatch):
    from helion._compiler.cute import _paged_attention_runtime as runtime

    def select(first, second, selector):
        result = torch.zeros_like(first)
        for output_byte in range(4):
            nibble = (selector >> (4 * output_byte)) & 15
            source = first if (nibble & 7) < 4 else second
            byte = (source >> (8 * (nibble & 3))) & 255
            if nibble & 8:
                byte = torch.where(byte & 128 != 0, 255, 0)
            result |= byte << (8 * output_byte)
        return result

    monkeypatch.setattr(
        runtime, "cutlass", SimpleNamespace(Int32=lambda x: torch.as_tensor(x).int())
    )
    monkeypatch.setattr(runtime, "_select_scale_pair", select)
    prepare = inspect.unwrap(runtime._select_scale_group_word)
    shifts = torch.arange(4) * 8
    for position in range(16):
        values = (torch.arange(256)[:, None] * 37 + torch.arange(16) * 53) & 255
        values[:, position] = torch.arange(256)
        words = (values.reshape(256, 4, 4) << shifts).sum(-1).int()
        for group in range(4):
            actual = prepare(tuple(words[:, index] for index in range(4)), group)
            expected = (values[:, group::4] << shifts).sum(-1).int()
            assert torch.equal(actual, expected)
