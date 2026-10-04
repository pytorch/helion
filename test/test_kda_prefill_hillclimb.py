from __future__ import annotations

import argparse
import json

from benchmarks.cute import kda_prefill_hillclimb as hillclimb
from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch


@pytest.mark.parametrize("previous_lowering", [None, "legacy", "shared"])
@pytest.mark.parametrize("requested_lowering", ["legacy", "shared"])
def test_pinned_configs_preserve_requested_lowering(
    tmp_path, previous_lowering, requested_lowering
) -> None:
    previous = {
        "partition": "all",
        "state_abi": "fp32",
        "schedule": "fused",
        "numerical_policy": "centered_bt32_fp32_rhs_v2",
        "stage_configs": {"fused_prefill": {"block_sizes": [64]}},
    }
    if previous_lowering is not None:
        previous["requested_lowering"] = previous_lowering
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps(previous))
    args = argparse.Namespace(configs=configs, state_abi="fp32")
    details = {
        "partition": "all",
        "schedule": "fused",
        "numerical_policy": previous["numerical_policy"],
        "requested_lowering": requested_lowering,
    }
    if (previous_lowering or "legacy") != requested_lowering:
        with pytest.raises(ValueError, match="same requested lowering"):
            hillclimb.load_stage_configs(args, {}, details)
        assert "loaded_configs" not in details
    else:
        assert (
            hillclimb.load_stage_configs(args, {}, details) == previous["stage_configs"]
        )
        assert details["loaded_configs"] == str(configs.resolve())


@pytest.mark.parametrize(
    ("source", "marker", "expected"),
    [
        ("chunk_prefill_sm100()", "chunk_prefill_sm100", ("chunk_prefill_sm100",)),
        (
            "chain_loop_index = 0; chain_0_mma = 1",
            "chunk_prefill_sm100",
            ("chain_loop_index", "chain_0_mma"),
        ),
        (
            "chain_loop_index = 0; chain_0_mma = 1",
            ("chain_loop_index", "chain_0_mma"),
            ("chain_loop_index", "chain_0_mma"),
        ),
        ("prepare_stage()", "prepare_stage", ("prepare_stage",)),
        (
            "chain_loop_index = 0; chain_0_warp_mma = 1",
            "chunk_prefill_sm100",
            ("chain_loop_index", "chain_0_warp_mma"),
        ),
        (
            "chain_loop_index = 0; chain_0_warp_mma = 1",
            ("chain_loop_index", "chain_0_mma"),
            ("chain_loop_index", "chain_0_warp_mma"),
        ),
    ],
)
def test_stage_source_markers_accept_only_complete_routes(
    source: str, marker: str | tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert hillclimb.stage_source_markers(source, marker) == expected


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        ("chain_loop_index = 0", "chunk_prefill_sm100"),
        ("chain_0_mma = 1", "chunk_prefill_sm100"),
        ("chunk_prefill_sm100()", ("chain_loop_index", "chain_0_mma")),
        ("ordinary_fallback()", "prepare_stage"),
        ("chain_0_warp_mma = 1", "chunk_prefill_sm100"),
        ("chain_0_warp_mma = 1", ("chain_loop_index", "chain_0_mma")),
    ],
)
def test_stage_source_markers_reject_fallbacks(
    source: str, marker: str | tuple[str, ...]
) -> None:
    with pytest.raises(RuntimeError, match="lowering markers"):
        hillclimb.stage_source_markers(source, marker)


@pytest.mark.parametrize("missing", hillclimb.PREPARED_PREFILL_MARKERS)
def test_prepared_source_requires_every_bound_program(missing: str) -> None:
    source = " ".join(
        marker for marker in hillclimb.PREPARED_PREFILL_MARKERS if marker != missing
    )
    with pytest.raises(RuntimeError, match="lowering markers"):
        hillclimb.stage_source_markers(source, hillclimb.PREPARED_PREFILL_MARKERS)


def test_prepared_source_preserves_physical_schedule_metadata() -> None:
    source = " ".join(hillclimb.PREPARED_PREFILL_MARKERS)
    assert (
        hillclimb.stage_source_markers(source, "chunk_prefill_sm100")
        == hillclimb.PREPARED_PREFILL_MARKERS
    )
    selected = {
        "cute_chunk_prefill_schedule": "single",
        "cute_chunk_prefill_task_order": "longest_first_precompute",
    }
    prepared = hillclimb.fused_schedule_details(
        selected, {"lowering_path": "shared_prepared_prefill"}, 7, 4096
    )
    legacy = hillclimb.fused_schedule_details(
        selected, {"lowering_path": "legacy_prefill"}, 7, 4096
    )
    assert prepared == {**legacy, "lowering_path": "shared_prepared_prefill"}


def test_shared_metadata_does_not_claim_ignored_legacy_launches() -> None:
    selected = {
        "cute_chained_group_contractions": True,
        "cute_chunk_prefill_schedule": "prefix_tail_4",
        "cute_chunk_prefill_task_order": "longest_first_precompute",
    }
    details = hillclimb.fused_schedule_details(
        selected, {"lowering_path": "shared_contraction_loop"}, 7, 4096
    )
    assert details["device_launches"] == 1
    assert details["workspace_bytes"] == 0
    assert "SMEM" in details["workspace_scope"]
    assert details["sequence_groups"] == [[0, 7]]
    assert details["topology"] == "serial"
    assert details["legacy_schedule_fields_ignored"] == {
        key: value for key, value in selected.items() if key.startswith("cute_chunk_")
    }


@pytest.mark.parametrize(
    ("schedule", "launches", "workspace"),
    [
        ("single", 2, 28),
        ("prefix_tail_2", 13, 8220),
        ("prefix_tail_4", 21, 8220),
    ],
)
def test_legacy_metadata_preserves_existing_schedule_contract(
    schedule: str, launches: int, workspace: int
) -> None:
    details = hillclimb.fused_schedule_details(
        {
            "cute_chunk_prefill_schedule": schedule,
            "cute_chunk_prefill_task_order": "longest_first_precompute",
        },
        {"lowering_path": "legacy_prefill"},
        7,
        4096,
    )
    assert details["device_launches"] == launches
    assert details["workspace_bytes"] == workspace
    assert details["lowering_path"] == "legacy_prefill"


@pytest.mark.parametrize("policy", ["native_bt16", "centered_bt32"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_shared_fused_build_clones_kernel_before_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path, policy: str, block_size: int
) -> None:
    args = hillclimb.parser().parse_args(
        [
            "--library-root",
            str(tmp_path),
            "--flashinfer-src",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--shape",
            "0",
            "--mode",
            "default",
            "--state-abi",
            "fp32",
            "--helion-schedule",
            "fused",
            "--helion-fused-policy",
            policy,
            "--helion-fused-lowering",
            "shared",
            "--helion-fused-block-size",
            str(block_size),
        ]
    )
    original = (
        kda_prefill_native_math_bt32
        if policy == "centered_bt32"
        else kda_prefill_native_math
    )
    original_overrides = original.settings.autotune_config_overrides
    observed = []

    def fake_factory(args, row, details, loaded):
        def compile_stage(name, kernel, values, seed, marker):
            assert kernel is not original and kernel.fn is original.fn
            assert kernel.settings is not original.settings
            assert kernel.settings.autotune_config_overrides == {
                "cute_chained_group_contractions": True,
                "cute_chained_mma_schedule": "tcgen05_tmem",
            }
            assert seed.config["block_sizes"] == [block_size]
            assert seed.config["cute_chained_group_contractions"] is True
            assert marker == ("chain_loop_index", "chain_0_mma")
            details["stage_configs"][name] = dict(seed.config)
            details["stages"][name] = {"lowering_path": "shared_contraction_loop"}
            observed.append(kernel)
            return lambda *args: None

        return compile_stage

    monkeypatch.setattr(hillclimb, "make_stage_compiler", fake_factory)
    monkeypatch.setattr(hillclimb, "capture", lambda function: function)
    q = torch.ones((1, 1, 1, 128), dtype=torch.bfloat16)
    inputs = (
        q,
        q.clone(),
        q.clone(),
        q.clone(),
        q[..., 0].clone(),
        torch.zeros(1),
        torch.zeros((1, 128)),
        torch.ones((1, 1, 128, 128)),
    )
    arm = hillclimb.build_helion_fused(
        args, inputs, torch.tensor([0, 1]), {"implementations": {}}
    )
    assert len(observed) == 1
    assert original.settings.autotune_config_overrides is original_overrides
    assert arm.details["device_launches"] == 1
    assert arm.details["workspace_bytes"] == 0
    assert "FP32 SMEM" in arm.details["numerical_contract"]["running_state"]
    assert "TMEM" not in arm.details["numerical_contract"]["running_state"]
