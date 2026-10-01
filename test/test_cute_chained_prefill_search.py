from __future__ import annotations

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_scratch_layout import _allocations
from .test_cute_chunk_prefill import _inputs
import helion
from helion import exc

_GROUP_OVERRIDES = {
    "cute_chained_group_contractions": True,
    "cute_chained_mma_schedule": "tcgen05_tmem",
}


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_explicit_grouped_family_exposes_real_value_tiles_and_search_knobs(
    chunk_size: int,
) -> None:
    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    grouped = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=_GROUP_OVERRIDES,
    )
    with _cpu_codegen():
        args = _inputs(heads=8, device=torch.device("cpu"))
        legacy = original._bind_isolated(args)
        bound = grouped._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_group_search_enabled
        assert spec.cute_chunk_prefill_task_order is None
        assert spec.cute_chunk_prefill_schedule is None
        assert (spec.block_sizes[0].min_size, spec.block_sizes[0].max_size) == (64, 128)
        fields = spec._flat_fields()
        assert set(_GROUP_OVERRIDES) <= fields.keys()
        assert "cute_chunk_prefill_task_order" not in fields
        assert "cute_chunk_prefill_schedule" not in fields
        config = spec.normalized_config(
            helion.Config(block_sizes=[128], num_warps=4, **_GROUP_OVERRIDES)
        )
        generation = spec.create_config_generation(overrides=_GROUP_OVERRIDES)
        flat = generation.flatten(config)
        restored = generation.unflatten(flat)
        assert restored["block_sizes"] == [128]
        assert all(restored[key] == value for key, value in _GROUP_OVERRIDES.items())
        assert generation.flatten(restored) == flat
        source = bound.to_code(restored)
        assert "chain_c_workspace" in source
        assert "chain_0_mma" in source
        assert "stride=(128, 1)" in source
        assert legacy.config_spec.block_sizes[0].max_size == 64
        assert set(legacy.config_spec._flat_fields()) == {
            "block_sizes",
            "cute_chunk_prefill_task_order",
            "cute_chunk_prefill_schedule",
            "cute_state_transfer_max_bits",
        } | ({"cute_state_transfer_transport"} if chunk_size == 32 else set())
        legacy_source = legacy.to_code(helion.Config(block_sizes=[64], num_warps=4))
        assert "chain_0_mma" not in legacy_source


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"cute_chained_mma_schedule": "tcgen05_tmem"},
        {
            "cute_chained_group_contractions": False,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        },
    ],
)
def test_partial_or_disabled_opt_in_keeps_legacy_search(
    overrides: dict[str, object],
) -> None:
    kernel = helion.kernel(
        kda_prefill_native_math.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=overrides,
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
    assert bound.config_spec.cute_chunk_prefill_task_order is not None
    assert bound.config_spec.block_sizes[0].max_size == 64


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cute_chained_group_contractions": True}, "requires a resident TCgen05"),
        (
            {
                "cute_chained_group_contractions": True,
                "cute_chained_mma_schedule": "cp_async",
            },
            "requires a resident TCgen05",
        ),
        (
            {
                "cute_chained_group_contractions": 1,
                "cute_chained_mma_schedule": "tcgen05_tmem",
            },
            "must be bool",
        ),
    ],
)
def test_invalid_explicit_grouping_is_not_silently_repaired(
    overrides: dict[str, object], message: str
) -> None:
    kernel = helion.kernel(
        kda_prefill_native_math.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=overrides,
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        with pytest.raises(exc.InvalidConfig, match=message):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(
                    {"block_sizes": [64], "num_warps": 4, **overrides}
                )
            )


@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("repair", [False, True])
def test_xor_cannot_be_ignored_by_unopted_legacy_prefill(
    chunk_size: int, repair: bool
) -> None:
    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    config = helion.Config(
        block_sizes=[64], num_warps=4, cute_chained_scratch_layout="xor"
    )
    with _cpu_codegen():
        bound = original._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        with pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"):
            bound.config_spec.normalize(config, _fix_invalid=repair)
        with pytest.raises(exc.InvalidConfig, match="explicit shared prefill family"):
            bound.to_code(config)


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_shared_prefill_xor_preserves_dv128_workspace_envelope(
    chunk_size: int,
) -> None:
    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    grouped = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=_GROUP_OVERRIDES,
    )
    with _cpu_codegen():
        bound = grouped._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        config = helion.Config(block_sizes=[128], num_warps=4, **_GROUP_OVERRIDES)
        row_major = bound.to_code(config)
        xor = bound.to_code(
            helion.Config.from_dict(
                config.config | {"cute_chained_scratch_layout": "xor"}
            )
        )
    assert _allocations(row_major) == _allocations(xor)
    assert "chain_c_workspace" in xor
    assert "make_swizzle(5, 0, 7)" in xor
    expected = "make_swizzle(5, 0, 5)" if chunk_size == 32 else "make_swizzle(4, 0, 4)"
    assert expected in xor
