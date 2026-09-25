from __future__ import annotations

import re
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _args
from .test_cute_chained_pointwise import _code
from .test_cute_chained_pointwise import _config
from .test_cute_chained_pointwise import _pointwise_dot
from .test_cute_chained_pointwise_fp32 import _mixed_args
import helion
from helion import exc
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_pointwise_unroll"
LOOP = re.compile(
    r"(for chain_\d+_[ab]_pointwise_step in cutlass.range\(\d+, unroll=)2(\):)"
)


def _config_unroll(factor: object = 1, enabled: bool = True) -> helion.Config:
    return helion.Config.from_dict(_config(enabled).config | {KEY: factor})


def _unrolled_code(args: tuple, factor: object = 1, enabled: bool = True) -> str:
    with patch(
        "test.test_cute_chained_pointwise._config",
        return_value=_config_unroll(factor, enabled),
    ):
        return _code(args, enabled)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded", "broadcast"]
)
def test_only_existing_vector_loop_digit_changes(dtype: torch.dtype, kind: str) -> None:
    args = _mixed_args("cpu", kind, dtype)
    old = _code(args)
    assert _unrolled_code(args) == old
    candidate = _unrolled_code(args, 2)
    restored, count = LOOP.subn(r"\g<1>1\2", candidate)
    assert count >= 1
    assert restored == old  # Complete source, including host, comments and fallback.
    assert "cutlass.Float32, num_bits_per_copy=128" in candidate or kind == "stride"


@pytest.mark.parametrize(
    "value", [True, False, 0, 3, 16, -1, 1.0, 2.0, 4.0, 8.0, "2", None]
)
def test_invalid_values_rejected(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 1, 2, 4 or 8"):
        _unrolled_code(_args("cpu", "dense"), value)


def test_disabled_vectorization_canonicalizes_to_default() -> None:
    args = _args("cpu", "dense")
    assert _unrolled_code(args, 2, False) == _code(args, False)


def test_no_exact_vector_leaf_rejects_factor_two() -> None:
    a, b, scale, bias, transpose = _args("cpu", "stride")
    # Leave scalar coefficient inputs intact so the structural discovery is a
    # superset; neither dense operand satisfies the actual coordinate proof.
    a = torch.empty((*a.shape[:-1], a.shape[-1] * 2), dtype=a.dtype)[..., ::2]
    args = (a, b, scale, bias, transpose)
    assert "_pointwise_copy =" not in _unrolled_code(args)
    with pytest.raises(exc.BackendUnsupported, match="admitted vector staging loop"):
        _unrolled_code(args, 2)


def test_activation_is_private_and_single_trip_is_inactive() -> None:
    first, second = PointwiseUnroll(2), PointwiseUnroll(2)
    assert first.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported):
        first.validate()
    assert first.loop_factor(2) == 2
    first.validate()
    with pytest.raises(exc.BackendUnsupported):
        second.validate()
    PointwiseUnroll(1).validate()


def test_direct_and_dot_derived_operands_do_not_advertise_unroll() -> None:
    from .test_cute_chained_tcgen05_config import _config_chain
    from .test_cute_chained_tcgen05_config import _inputs

    with patch_cute_mma_support():
        bound = _config_chain._bind_isolated((*_inputs(), None, False))
    spec = bound.config_spec
    assert not spec.cute_chained_pointwise_unroll_search_enabled
    assert KEY not in spec._flat_fields()
    assert all(KEY not in seed.config for seed in spec.compiler_seed_configs)
    with pytest.raises(exc.InvalidConfig, match="computed TCgen05 vector operands"):
        spec.normalized_config(_config_unroll(2))
    fixed = _config_unroll(2)
    spec.normalize(fixed, _fix_invalid=True)
    assert KEY not in fixed.config


@pytest.mark.parametrize("schedule", ["coalesced", "cp_async_register"])
def test_other_schedules_canonicalize_factor_two(schedule: str) -> None:
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_args("cpu", "dense"))
    config = _config_unroll(2)
    config.config["cute_chained_mma_schedule"] = schedule
    normalized = bound.config_spec.normalized_config(config)
    assert normalized.config[KEY] == 1
    assert normalized.config["cute_chained_pointwise_vectorize"] is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_original_operand_dtypes_keep_exact_source(dtype: torch.dtype) -> None:
    args = _args("cpu", "dense", dtype)
    candidate = _unrolled_code(args, 2)
    restored, count = LOOP.subn(r"\g<1>1\2", candidate)
    assert count == 2
    assert restored == _code(args)


@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "broadcast"]
)
def test_larger_factors_change_only_the_existing_loop_digit(
    factor: int, dtype: torch.dtype, kind: str
) -> None:
    args = _mixed_args("cpu", kind, dtype)
    original = _unrolled_code(args, 1)
    candidate = _unrolled_code(args, factor)
    pattern = re.compile(
        rf"(for chain_\d+_[ab]_pointwise_step in cutlass.range\(\d+, unroll=){factor}(\):)"
    )
    restored, count = pattern.subn(r"\g<1>1\2", candidate)
    assert count >= 1 and restored == original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["tail", "padded"])
def test_shorter_real_staging_loop_accepts_four_but_rejects_eight(
    dtype: torch.dtype, kind: str
) -> None:
    args = _mixed_args("cpu", kind, dtype)
    source = _unrolled_code(args, 4)
    assert "cutlass.range(4, unroll=4)" in source
    assert "< 49" in source  # Keep all scalar fallback/tail predicates.
    with pytest.raises(exc.BackendUnsupported, match="whole number of unroll groups"):
        _unrolled_code(args, 8)


@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("trips", [2, 3, 4, 6, 8, 12, 16])
def test_larger_factor_requires_complete_groups(factor: int, trips: int) -> None:
    tracker = PointwiseUnroll(factor)
    if trips < factor or trips % factor:
        with pytest.raises(
            exc.BackendUnsupported, match="whole number of unroll groups"
        ):
            tracker.loop_factor(trips)
        assert not tracker.activated
    else:
        assert tracker.loop_factor(trips) == factor
        tracker.validate()


@pytest.mark.parametrize("factor", [4, 8])
def test_larger_factor_cannot_hide_a_short_loop_after_activation(factor: int) -> None:
    tracker = PointwiseUnroll(factor)
    assert tracker.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported, match="admitted vector staging loop"):
        tracker.validate()
    assert tracker.loop_factor(16) == factor
    with pytest.raises(exc.BackendUnsupported, match="whole number of unroll groups"):
        tracker.loop_factor(2)


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("trips", [1, 2, 3, 5, 8, 16])
def test_existing_factor_activation_behavior_unchanged(factor: int, trips: int) -> None:
    tracker = PointwiseUnroll(factor)
    assert tracker.loop_factor(trips) == (1 if trips == 1 else factor)
    assert tracker.activated == (trips >= 2)


@pytest.mark.parametrize("factor", [4, 8])
def test_larger_inactive_configs_preserve_canonical_default(factor: int) -> None:
    args = _args("cpu", "dense")
    assert _unrolled_code(args, factor, False) == _code(args, False)
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(args)
    for schedule in ("coalesced", "cp_async_register"):
        config = _config_unroll(factor)
        config.config["cute_chained_mma_schedule"] = schedule
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.config[KEY] == 1
        assert normalized.config["cute_chained_pointwise_vectorize"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "offset"])
def test_pointwise_unroll_runtime(dtype: torch.dtype, kind: str) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(_config_unroll(2))
        raw = b.transpose(-2, -1) if transpose else b
        right = (raw * (scale.exp() * bias[:, None])[:, :, None] + 1.0).to(dtype)
        left = (a.float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual, repeated = run(*args), run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "offset"])
def test_larger_pointwise_unroll_runtime(
    factor: int, dtype: torch.dtype, kind: str
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(
                _config_unroll(factor)
            )
        raw = b.transpose(-2, -1) if transpose else b
        right = (raw * (scale.exp() * bias[:, None])[:, :, None] + 1.0).to(dtype)
        left = (a.float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual, repeated = run(*args), run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)
