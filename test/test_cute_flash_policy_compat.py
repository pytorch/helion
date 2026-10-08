from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from helion.autotuner.config_fragment import EnumFragment


def _config_spec(
    num_kv: int,
    *,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = True,
    requires_ws_overlap: bool = False,
) -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=is_causal,
        requires_ws_overlap=requires_ws_overlap,
        standard_dense_output=not is_causal,
        standard_causal_output=is_causal,
    )
    return spec


def _fixed_config(values: dict[str, object]) -> helion.Config:
    return helion.Config.from_dict({"block_sizes": [1, 128, 128], **values})


_LEGACY_WS_VALUES = (
    *(
        (cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY, value)
        for value in ("xu", "16/4", "8/2", "16/6", "16/8")
    ),
    *(
        (cute_flash.FLASH_EXP2_PACKET_KEY, value)
        for value in (
            "4x1",
            "4x2",
            "8x1",
            "8x2",
            "deg2_16x6",
            "hybrid_deg1_16x8",
            "deg1_16x8",
            "deg1_8x2_corr10",
            "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
        )
    ),
    (cute_flash.FLASH_WAIT_HINT_KEY, 0),
)


@pytest.mark.parametrize("num_kv", (4, 4096))
@pytest.mark.parametrize(("key", "value"), _LEGACY_WS_VALUES)
def test_fixed_ws_aliases_have_one_canonical_config(
    num_kv: int, key: str, value: object
) -> None:
    spec = _config_spec(num_kv)
    canonical = _fixed_config({cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"})
    legacy = _fixed_config(
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", key: value}
    )
    spec.normalize(canonical)
    spec.normalize(legacy)
    assert legacy == canonical
    spec.normalize(legacy)
    assert legacy == canonical


@pytest.mark.parametrize("parent", ("legacy", "required", "implicit-odd"))
@pytest.mark.parametrize(
    ("head_dim", "dtype", "is_causal", "packet"),
    (
        (64, torch.float16, True, "deg2_16x6"),
        (64, torch.float16, False, "deg1_16x8"),
        (64, torch.bfloat16, True, "hybrid_deg1_16x8"),
        (128, torch.bfloat16, False, "deg2_16x6"),
        (
            128,
            torch.bfloat16,
            True,
            "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
        ),
    ),
)
def test_inactive_compound_packet_does_not_change_ws_parent(
    parent: str, head_dim: int, dtype: torch.dtype, is_causal: bool, packet: str
) -> None:
    spec = _config_spec(
        3 if parent == "implicit-odd" else 4,
        head_dim=head_dim,
        dtype=dtype,
        is_causal=is_causal,
        requires_ws_overlap=parent == "required",
    )
    parent_values: dict[str, object] = (
        {cute_flash.FLASH_TOPOLOGY_KEY: "ws_overlap"} if parent == "legacy" else {}
    )
    canonical = _fixed_config(parent_values)
    legacy = _fixed_config(
        {
            **parent_values,
            cute_flash.FLASH_EXP2_PACKET_KEY: packet,
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
            cute_flash.FLASH_WAIT_HINT_KEY: 0,
        }
    )
    spec.normalize(canonical)
    spec.normalize(legacy)
    assert canonical.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
    assert legacy == canonical


@pytest.mark.parametrize("family", ("ws_overlap", "fa4"))
@pytest.mark.parametrize("num_kv", (4, 4096))
@pytest.mark.parametrize(
    ("key", "value"),
    (
        (cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY, "32/16"),
        (cute_flash.FLASH_EXP2_PACKET_KEY, "unknown_packet"),
        (cute_flash.FLASH_WAIT_HINT_KEY, 1),
    ),
)
def test_unknown_flash_values_still_fail_membership_validation(
    family: str, num_kv: int, key: str, value: object
) -> None:
    spec = _config_spec(num_kv)
    config = _fixed_config({cute_flash.FLASH_PIPELINE_FAMILY_KEY: family, key: value})
    with pytest.raises(InvalidConfig, match=rf"{key} must be one of"):
        spec.normalize(config)


@pytest.mark.parametrize("num_kv", (4, 4096))
def test_ws_alias_does_not_bypass_active_fa4_packet_validation(num_kv: int) -> None:
    spec = _config_spec(num_kv, head_dim=128)
    config = _fixed_config(
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EXP2_PACKET_KEY: "4x1",
        }
    )
    with pytest.raises(InvalidConfig, match="cute_flash_exp2_packet must be one of"):
        spec.normalize(config)


@pytest.mark.parametrize(
    "parent_key", (cute_flash.FLASH_PIPELINE_FAMILY_KEY, cute_flash.FLASH_TOPOLOGY_KEY)
)
@pytest.mark.parametrize("packet", ("4x1", "deg2_16x6", "hybrid_deg1_16x8"))
def test_explicit_ws_packet_override_must_be_effective(
    parent_key: str, packet: str
) -> None:
    spec = _config_spec(4)
    with pytest.raises(
        InvalidConfig, match="requires cute_flash_pipeline_family|is not effective with"
    ):
        generation = spec.create_config_generation(
            overrides={
                parent_key: "ws_overlap",
                cute_flash.FLASH_EXP2_PACKET_KEY: packet,
            }
        )
        generation.unflatten(generation.default_flat())


@pytest.mark.parametrize("num_kv", (4, 4096))
def test_active_fa4_degree2_packet_remains_effective(num_kv: int) -> None:
    spec = _config_spec(num_kv)
    generation = spec.create_config_generation(
        overrides={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EXP2_PACKET_KEY: "deg2_16x6",
        }
    )
    config = generation.unflatten(generation.default_flat())
    assert config.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
    assert config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == "deg2_16x6"
    assert config.config[cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY] == "16/6"
    assert config.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is True
    spec.normalize(config)
    assert config.config[cute_flash.FLASH_EXP2_PACKET_KEY] == "deg2_16x6"


@pytest.mark.parametrize("num_kv", (4, 4096))
@pytest.mark.parametrize(
    ("key", "value"),
    (
        (cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY, "16/6"),
        (cute_flash.FLASH_EXP2_PACKET_KEY, "deg2_16x6"),
        (cute_flash.FLASH_WAIT_HINT_KEY, 0),
    ),
)
def test_fixed_config_compatibility_does_not_expand_ws_fragment_choices(
    num_kv: int, key: str, value: object
) -> None:
    spec = _config_spec(num_kv)
    fragments = spec._cute_flash_autotune_fragments("ws_overlap", "ws_overlap")
    assert value not in cast("EnumFragment", fragments[key]).choices
    config = _fixed_config(
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", key: value}
    )
    spec.normalize(config)
    assert config.config[key] != value
