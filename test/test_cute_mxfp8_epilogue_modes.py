"""Native epilogue configuration, descriptor and resource admission proofs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test.test_cute_mxfp8_mma import _cpu_inputs
from test.test_cute_mxfp8_mma import _cpu_target
from test.test_cute_mxfp8_mma import _mxfp8
from test.test_cute_mxfp8_mma import _wrapper_plan

import helion
from helion._compiler.cute.block_scaled_config import MXFP8_EPILOGUE_MODES
from helion._compiler.cute.block_scaled_config import mxfp8_shared_memory_bytes
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
import helion.language as hl
from helion.runtime.cute.launcher import _cute_stage_under_aligned_operands
from helion.runtime.cute.launcher import _cute_staged_plan_operands


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_full_slice_matmul_tiling=True,
)
def _complete_packed_contraction(
    left: torch.Tensor,
    right: torch.Tensor,
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    rows, channels = left.shape
    columns = right.shape[0]
    for row, column in hl.tile([rows, columns]):
        groups = hl.arange(channels // 32)
        left_words = hl.load(packed_left, [row, groups // 4])
        right_words = hl.load(packed_right, [column, groups // 4])
        shift = ((groups % 4) * 8)[None, :]
        left_scale = ((left_words >> shift) & 255).to(torch.uint8)
        right_scale = ((right_words >> shift) & 255).to(torch.uint8)
        output[row, column] = hl.dot_scaled(
            left[row, :],
            left_scale,
            "e4m3",
            right[column, :].T,
            right_scale,
            "e4m3",
        ).to(output.dtype)
    return output


def _inputs(m: int = 8, n: int = 264, *, output: torch.Tensor | None = None):
    k = 256
    return (
        torch.empty((m, k), dtype=torch.float8_e4m3fn),
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.empty_strided((m, 2), (1, (m + 3) // 4 * 4), dtype=torch.int32),
        torch.empty_strided((n, 2), (1, (n + 3) // 4 * 4), dtype=torch.int32),
        torch.empty((m, n), dtype=torch.bfloat16) if output is None else output,
    )


def _config(bound, **changes: object) -> helion.Config:
    return helion.Config.from_dict(
        bound.config_spec.default_config().config
        | {
            "cute_scaled_mma": True,
            "cute_collective_mma": False,
            "cute_scaled_fused_scales": True,
            "cute_scaled_tile_m": 128,
            "cute_scaled_tile_n": 224,
            "cute_scaled_tile_k": 128,
            "cute_scaled_stages": 2,
            "cute_scaled_cluster_m": 2,
            "cute_scaled_swap_ab": False,
            "cute_scaled_compact_m": False,
        }
        | changes
    )


@pytest.fixture(autouse=True)
def _no_cuda():
    with _cpu_target(), _forbid_cuda_work():
        yield


@pytest.mark.parametrize("mode", MXFP8_EPILOGUE_MODES)
def test_cpu_epilogue_modes_survive_config_transfer(mode: str) -> None:
    bound = _complete_packed_contraction._bind_isolated(_inputs())
    assert bound.config_spec.cute_mxfp8_tma_output_available
    config = _config(bound, cute_scaled_epilogue_mode=mode)
    normalized = bound.config_spec.normalized_config(config)
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        restored = generation.unflatten(generation.flatten(config))
    assert restored == normalized
    assert restored["cute_scaled_epilogue_mode"] == mode
    plan = _wrapper_plan(bound.to_code(config))
    actual = (
        plan.get("accumulator_stages", 1),
        plan.get("prefetch_output_registers", False),
        plan.get("tma_output", False),
    )
    assert actual == MXFP8_EPILOGUE_MODES[mode]
    if mode == "default":
        assert not {
            "accumulator_stages",
            "prefetch_output_registers",
            "tma_output",
        }.intersection(plan)


@pytest.mark.parametrize(
    "changes",
    [
        {"cute_scaled_swap_ab": True},
        {"cute_scaled_tile_m": 64},
        {"cute_scaled_tile_n": 16},
        {"cute_scaled_tile_n": 256},
        {"cute_scaled_fused_scales": False},
        {"cute_scaled_epilogue_mode": False},
        {"cute_scaled_epilogue_mode": "unknown"},
    ],
)
def test_cpu_epilogue_invalid_profiles_reject(changes: dict[str, object]) -> None:
    bound = _complete_packed_contraction._bind_isolated(_inputs())
    config = _config(bound, cute_scaled_epilogue_mode="tma_prefetch_2")
    config.config.update(changes)
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(config)


def test_cpu_epilogue_is_inactive_for_the_existing_scalar_fallback() -> None:
    bound = _complete_packed_contraction._bind_isolated(_inputs())
    config = _config(
        bound,
        cute_scaled_mma=False,
        cute_scaled_epilogue_mode="tma_prefetch_2",
    )
    normalized = bound.config_spec.normalized_config(config)
    assert normalized["cute_scaled_epilogue_mode"] == "default"


def test_cpu_unpacked_family_keeps_only_the_default_epilogue_field() -> None:
    bound = _mxfp8._bind_isolated(_cpu_inputs((8, 264, 256)))
    assert bound.config_spec.cute_mxfp8_mma_available
    assert not bound.config_spec.cute_mxfp8_tma_output_available
    with bound.env:
        fragment = bound.config_spec._flat_fields()["cute_scaled_epilogue_mode"]
    assert fragment.choices == ("default",)


@pytest.mark.parametrize(
    "output",
    [
        torch.empty((8, 264), dtype=torch.float32),
        torch.empty((264, 8), dtype=torch.bfloat16).T,
        torch.empty_strided((8, 264), (265, 1), dtype=torch.bfloat16),
        torch.empty((1, 264), dtype=torch.bfloat16),
        torch.empty((8, 1), dtype=torch.bfloat16),
    ],
)
def test_cpu_epilogue_descriptor_metadata_reject(output: torch.Tensor) -> None:
    m, n = output.shape
    bound = _complete_packed_contraction._bind_isolated(_inputs(m, n, output=output))
    assert bound.config_spec.cute_mxfp8_mma_available
    assert not bound.config_spec.cute_mxfp8_tma_output_available
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config(bound, cute_scaled_epilogue_mode="tma_staged_2"))
    assert _wrapper_plan(bound.to_code(_config(bound)))["format"] == "mxfp8"


@pytest.mark.parametrize("mode", tuple(MXFP8_EPILOGUE_MODES)[1:])
def test_cpu_epilogue_invalid_fixup_preserves_geometry(mode: str) -> None:
    values = {
        "cute_scaled_mma": True,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_n": 256,
        "cute_scaled_tile_k": 128,
        "cute_scaled_epilogue_mode": mode,
    }
    normalize_block_scaled_config(
        values,
        available=True,
        capability=(10, 3),
        fix_invalid=True,
        mxfp8=True,
        fused_scales_available=True,
        tma_output_available=True,
    )
    assert values["cute_scaled_epilogue_mode"] == "default"
    assert values["cute_scaled_tile_n"] == 256


@pytest.mark.parametrize(
    "sf_tma,sf_inplace", [(False, False), (True, False), (True, True)]
)
def test_cpu_epilogue_shared_storage_counts_both_stages(
    sf_tma: bool, sf_inplace: bool
) -> None:
    base = {"compact_m": False, "sf_tma": sf_tma, "sf_inplace": sf_inplace}
    default = mxfp8_shared_memory_bytes(1024, 128, 224, 128, 6, 2, 2, **base)
    assert default == (221184 if sf_tma and not sf_inplace else 211968)
    for mode in tuple(MXFP8_EPILOGUE_MODES)[1:]:
        assert (
            mxfp8_shared_memory_bytes(
                1024, 128, 224, 128, 6, 2, 2, epilogue_mode=mode, **base
            )
            == default
        )
        # N32 uses two 128x32 BF16 stages, adding8192 bytes to the default.
        assert (
            mxfp8_shared_memory_bytes(
                8, 128, 32, 128, 2, 1, 2, epilogue_mode=mode, **base
            )
            - mxfp8_shared_memory_bytes(8, 128, 32, 128, 2, 1, 2, **base)
            == 8192
        )


def test_cpu_tma_output_under_alignment_preserves_stride_values_and_padding() -> None:
    backing = torch.full((8 * 272 + 8,), 63, dtype=torch.bfloat16)
    output = torch.as_strided(backing, (8, 264), (272, 1), storage_offset=1)
    output.copy_(torch.arange(8 * 264).reshape(8, 264))
    original = backing.clone()
    plan = {
        "kind": "block_scaled_mma",
        "format": "mxfp8",
        "tma_output": True,
        "out_idx": 0,
    }
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    assert _cute_staged_plan_operands(kernel) == ((0, True),)
    staged = _cute_stage_under_aligned_operands(kernel, (output,))
    assert staged is not None
    arguments, copy_backs = staged
    copy = arguments[0]
    assert copy.data_ptr() % 16 == 0
    assert copy.stride() == output.stride()
    assert copy.storage_offset() == 0
    assert torch.equal(copy, output)
    assert copy_backs == [(output, copy)]
    copy.add_(1)
    copy_backs[0][0].copy_(copy_backs[0][1])
    live = torch.zeros_like(backing, dtype=torch.bool)
    torch.as_strided(live, output.shape, output.stride(), storage_offset=1).fill_(True)
    assert torch.equal(backing[~live], original[~live])
    assert torch.equal(output, copy)
    plan["tma_output"] = False
    assert _cute_stage_under_aligned_operands(kernel, (output,)) is None
