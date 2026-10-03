from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import NumThreadsFragment
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.autotuner.external import UserConfigSpec
from helion.autotuner.llm.configs import describe_fragment
from helion.autotuner.llm.feedback import format_config_for_prompt
from helion.autotuner.llm.prompting import build_initial_prompt
from helion.autotuner.llm.prompting import build_initial_search_guidance
from helion.autotuner.llm.prompting import build_refinement_prompt
from helion.autotuner.llm.prompting import build_system_prompt
from helion.autotuner.llm.workload import compute_workload_hints
from helion.runtime.config import Config
from helion.runtime.settings import Settings


def _config_spec(backend: str) -> ConfigSpec:
    return ConfigSpec(
        backend=CuteBackend() if backend == "cute" else TritonBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=148,
    )


def _generic_config_spec(backend: str) -> ConfigSpec:
    spec = _config_spec(backend)
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
    if backend == "cute":
        spec.num_threads.append(NumThreadsSpec(block_id=0, size_hint=256))
    return spec


def _flash_config_spec() -> ConfigSpec:
    spec = _config_spec("cute")
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=128,
        num_kv=12288,
        num_bh=8,
        tensor_4d_heads=8,
        block_size_targets={0: 1, 1: 128, 2: 128},
        standard_dense_output=True,
    )
    return spec


def _refinement_prompt(backend: str, *, failed_count: int = 0) -> str:
    return build_refinement_prompt(
        backend=backend,
        configs_per_round=15,
        compile_timeout_s=60,
        failed_count=failed_count,
        total_count=15,
        search_state="Round 2; 15 candidates measured",
        anchor_configs="Anchor 1",
        results="0.5 ms",
        top_patterns="Successful family",
        failed_patterns="Failed family",
    )


def _assert_no_unavailable_controls(prompt: str) -> None:
    for name in (
        "num_warps",
        "num_stages",
        "pid_type",
        "tensor_descriptor",
        "maxnreg",
        "load_eviction_policies",
    ):
        assert name not in prompt


@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_system_prompt_identifies_selected_backend(backend: str) -> None:
    prompt = build_system_prompt(backend=backend)
    assert backend in prompt.lower()
    assert "backend" in prompt.lower()
    assert '"configs"' in prompt
    assert "minified JSON" in prompt
    assert "LFBO" not in prompt
    assert (
        "analyze the kernel source, input tensors, GPU hardware, and config space "
        "to infer likely optimization traits from the code itself and target "
        "hardware; if unsure, stay closer to default."
    ) in prompt
    for name in ("block_sizes", "pid_type", "indexing", "l2_groupings"):
        assert name in prompt
    if backend == "cute":
        assert "Helion/Triton" not in prompt
        assert "num_threads" in prompt
        for name in ("num_warps", "num_stages", "maxnreg"):
            assert name not in prompt


@pytest.mark.parametrize("failed_count", [0, 10])
def test_flash_guidance_uses_available_knobs_and_fixed_tiles(
    failed_count: int,
) -> None:
    spec = _flash_config_spec()
    initial = build_initial_search_guidance(
        flat_fields=spec._flat_fields(),
        backend=spec.backend_name,
        configs_per_round=15,
        compile_timeout_s=60,
    )
    refinement = _refinement_prompt(spec.backend_name, failed_count=failed_count)
    for prompt in (initial, refinement):
        _assert_no_unavailable_controls(prompt)
        assert "LFBO" not in prompt
        assert "Start from compatible compiler seeds" not in prompt
    assert "where the displayed bounds permit it" in initial
    assert "fixed fields and fixed coordinates unchanged" in initial
    for name in ("cute_flash_", "num_threads", "block_sizes", "CuTe controls exposed"):
        assert name not in refinement
    if failed_count:
        assert "Recent rounds had many failures" in refinement
        assert "Back off aggressive settings first" in refinement
        assert "About two thirds" not in refinement
    else:
        assert "About two thirds" in refinement
        assert "Recent rounds had many failures" not in refinement


def test_initial_flash_prompt_preserves_defaults_and_compiler_seeds() -> None:
    spec = _flash_config_spec()
    default = spec.autotune_reference_config()
    seed = Config.from_dict({**default, "cute_flash_e2e_schedule": "16/4"})
    spec.compiler_seed_configs = [seed]
    spec.autotuner_heuristics = ["cute_flash_attention"]
    default_text = format_config_for_prompt(default)
    seed_text = format_config_for_prompt(seed)
    kernel = SimpleNamespace(
        config_spec=spec,
        settings=Settings(backend="cute"),
        env=SimpleNamespace(device=torch.device("cpu")),
    )
    args = tuple(
        torch.empty(
            (1, 8, 12288, 128), device=torch.device("meta"), dtype=torch.float16
        )
        for _ in range(3)
    )
    with (
        patch(
            "helion.autotuner.llm.workload._gpu_hardware_lines",
            return_value=["Device: test GPU", "Compute units (SMs): 148"],
        ),
        patch(
            "helion.autotuner.llm.prompting.detect_workload_traits",
            return_value=frozenset({"matmul", "reduction", "attention_reduction"}),
        ),
    ):
        prompt = build_initial_prompt(
            kernel=kernel,
            args=args,
            config_spec=spec,
            configs_per_round=15,
            compile_timeout_s=60,
        )
    assert "cute" in prompt.lower()
    assert "backend" in prompt.lower()
    assert default_text in prompt
    assert seed_text in prompt
    assert "cute_flash_attention" in prompt
    assert "[1, 8, 12288, 128]" in prompt
    assert "LFBO" not in prompt
    _assert_no_unavailable_controls(prompt)
    assert format_config_for_prompt(spec.autotune_reference_config()) == default_text
    assert format_config_for_prompt(spec.compiler_seed_configs[0]) == seed_text


@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_generic_guidance_preserves_applicable_tuning_controls(backend: str) -> None:
    spec = _generic_config_spec(backend)
    initial = build_initial_search_guidance(
        flat_fields=spec._flat_fields(),
        backend=spec.backend_name,
        configs_per_round=15,
        compile_timeout_s=60,
    )
    assert "block_sizes" in initial
    assert (
        "40% near-default safe, 40% balanced throughput, and 20% aggressive" in initial
    )
    assert "If the kernel structure is unclear, stay closer to default" in initial
    assert "exceed 6 only when several coupled changes are needed" in initial
    assert "Vary block_sizes coherently across dimensions" in initial
    assert "Prefer edits with attributable effects" in _refinement_prompt(backend)
    if backend == "cute":
        assert "num_threads" in initial
        for prompt in (initial, _refinement_prompt(backend)):
            assert "LFBO" not in prompt
            _assert_no_unavailable_controls(prompt)
    else:
        assert "num_warps" in initial
        assert "num_stages" in initial


def test_cute_tcgen05_preserves_shared_scheduling_and_indexing_guidance() -> None:
    spec = _config_spec("cute")
    spec.cute_tcgen05_search_enabled = True
    spec.indexing.length = 2
    fields = spec._flat_fields()
    assert {"pid_type", "indexing", "l2_groupings"} <= fields.keys()
    initial = build_initial_search_guidance(
        flat_fields=fields,
        backend=spec.backend_name,
        configs_per_round=15,
        compile_timeout_s=60,
    )
    assert "If tensor_descriptor is available, treat it as a separate family" in initial
    assert (
        "Include both flat and persistent scheduling families when plausible" in initial
    )
    args = tuple(
        torch.empty((4096, 4096), device=torch.device("meta")) for _ in range(2)
    )
    with patch("helion.autotuner.llm.workload.num_compute_units", return_value=148):
        hints = compute_workload_hints(
            args,
            flat_fields=fields,
            backend=spec.backend_name,
            workload_traits=frozenset({"matmul"}),
        )
    assert "Matmul-like: [4096x4096] @ [4096x4096]" in hints
    assert "pid_type" in hints
    assert "l2_groupings" in hints
    assert "num_stages" not in hints


def test_schema_describes_cute_threads_auto_and_numeric_choices() -> None:
    description = describe_fragment(NumThreadsFragment(128))
    assert "0" in description
    assert "128" in description
    assert "NumThreadsFragment()" not in description


def test_external_guidance_does_not_invent_block_sizes() -> None:
    spec = UserConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=148,
        user_defined_tunables={"tile": IntegerFragment(32, 128, 64)},
    )
    prompts = (
        build_initial_search_guidance(
            flat_fields=spec._flat_fields(),
            backend=spec.backend_name,
            configs_per_round=4,
            compile_timeout_s=60,
        ),
        _refinement_prompt(spec.backend_name),
    )
    for prompt in prompts:
        assert "block_sizes" not in prompt
        assert "num_warps" not in prompt


@pytest.mark.parametrize("flash", [False, True])
def test_cute_workload_hints_exclude_unavailable_controls(flash: bool) -> None:
    spec = _flash_config_spec() if flash else _generic_config_spec("cute")
    args = tuple(
        torch.empty(
            (1, 8, 12288, 128), device=torch.device("meta"), dtype=torch.float16
        )
        for _ in range(3)
    )
    hints = compute_workload_hints(
        args,
        flat_fields=spec._flat_fields(),
        backend=spec.backend_name,
        workload_traits=frozenset({"matmul", "reduction", "attention_reduction"}),
    )
    _assert_no_unavailable_controls(hints)
    assert "Total input data" in hints
    assert "attention_reduction" in hints
    assert "use scheduling or indexing changes to create diversity" in hints
    assert "Include at least one persistent scheduling family when available" in hints


@pytest.mark.parametrize(
    "traits",
    [
        (),
        ("attention_reduction",),
        ("reduction",),
        ("matmul",),
        ("matmul", "reduction"),
        ("attention_reduction", "matmul", "reduction"),
    ],
)
def test_cute_workload_hints_follow_traits(traits: tuple[str, ...]) -> None:
    args = tuple(torch.empty((128, 128), device=torch.device("meta")) for _ in range(3))
    hints = compute_workload_hints(
        args,
        flat_fields=_generic_config_spec("cute")._flat_fields(),
        backend="cute",
        workload_traits=frozenset(traits),
    )
    assert ("attention/reduction-style" in hints) == ("attention_reduction" in traits)
    assert ("Matmul-like: [128x128] @ [128x128]" in hints) == ("matmul" in traits)
    assert ("streams each input row once" in hints) == (
        "reduction" in traits
        and not {"matmul", "attention_reduction"}.intersection(traits)
    )
    _assert_no_unavailable_controls(hints)


def _reduction_config_spec(backend: str, extent: int) -> ConfigSpec:
    spec = _config_spec(backend)
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=extent))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=0, size_hint=extent))
    spec.reduction_block_ids.add(0)
    if backend == "cute":
        spec.num_threads.append(NumThreadsSpec(block_id=0, size_hint=min(extent, 256)))
    return spec


def test_cute_reduction_prompts_support_reduction_loop_spec() -> None:
    spec = _reduction_config_spec("cute", 65536)
    prompt = build_initial_search_guidance(
        flat_fields=spec._flat_fields(),
        backend=spec.backend_name,
        configs_per_round=15,
        compile_timeout_s=60,
    )
    assert "num_threads" in prompt
    _assert_no_unavailable_controls(prompt)
