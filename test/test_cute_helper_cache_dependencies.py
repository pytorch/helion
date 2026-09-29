"""External compiled helper edits must invalidate the correct disk artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_resident_reductions_codegen import _kernel_and_arguments

from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
from helion.runtime.cute import launcher
from helion.runtime.cute import source_dependencies


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = set(source_dependencies._COMMON_DEPENDENCIES)
    for paths in source_dependencies._WRAPPER_DEPENDENCIES.values():
        files.update(paths)
    for relative in files:
        filename = tmp_path / relative
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(f"# {relative}\ndef device_helper():\n    return 1\n")
    monkeypatch.setattr(source_dependencies, "_PACKAGE_ROOT", tmp_path)
    return tmp_path


def _kernel(kind: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        _helion_cute_source_hash="unchanged generated wrapper and plan",
        _helion_cute_wrapper_plans=[] if kind is None else [{"kind": kind}],
    )


def _key(kernel: SimpleNamespace) -> str | None:
    plans = tuple(repr(plan) for plan in kernel._helion_cute_wrapper_plans)
    return launcher._cute_disk_cache_key(
        kernel, (), (192, 1, 1), plans, None, "--enable-tvm-ffi", 148
    )


@pytest.mark.parametrize(
    "kind,edited,unrelated",
    [
        (
            "gathered_mma_tma",
            "_compiler/cute/gathered_mma_runtime.py",
            "block_scaled_mma",
        ),
        (
            "gathered_mma_tma",
            "_compiler/cute/gathered_mma_runtime.py",
            "tcgen05_ab_tma",
        ),
        (
            "chunk_recurrence_sm100",
            "_compiler/cute/chunk_recurrence_sm100.py",
            "chunk_recurrence_warp_dv4",
        ),
        (
            "chunk_recurrence_warp_dv4",
            "_compiler/cute/kda_device_primitives.py",
            "chunk_recurrence_sm100",
        ),
    ],
)
def test_transitive_helper_changes_only_its_family(
    source_tree: Path, kind: str, edited: str, unrelated: str
) -> None:
    affected, unaffected = _kernel(kind), _kernel(unrelated)
    before = _key(affected), _key(unaffected)
    (source_tree / edited).write_text("def helper():\n    return 9\n")
    assert _key(affected) != before[0]
    assert _key(unaffected) == before[1]


@pytest.mark.parametrize(
    "edited", ("_compiler/cute/_flash_runtime.py", "_compiler/cute/_mlir_compat.py")
)
def test_grouped_mailbox_helpers_invalidate_both_conversion_families(
    source_tree: Path, edited: str
) -> None:
    rna = _kernel("tcgen05_grouped_rna")
    tma_rn = _kernel("tcgen05_grouped_tma_rn")
    unrelated = _kernel("chunk_recurrence_sm100")
    before = _key(rna), _key(tma_rn), _key(unrelated)
    path = source_tree / edited
    path.write_text("# changed shared mailbox lowering\n")
    assert _key(rna) != before[0]
    assert _key(tma_rn) != before[1]
    assert _key(unrelated) == before[2]
    path.unlink()
    assert _key(rna) is None and _key(tma_rn) is None
    assert _key(unrelated) == before[2]


def test_wrapper_generator_edit_invalidates_all_kinds(source_tree: Path) -> None:
    kernels = [_kernel(None), *map(_kernel, source_dependencies._WRAPPER_DEPENDENCIES)]
    before = list(map(_key, kernels))
    (source_tree / "runtime/cute/launcher.py").write_text("# new wrapper generation\n")
    assert all(old != _key(kernel) for old, kernel in zip(before, kernels, strict=True))


@pytest.mark.parametrize("schedule", ["scalar", "resident", "pipelined"])
@skipUnlessBackends(["cute"])
def test_ordinary_resident_codegen_tracks_helper_edits_and_missing_source(
    source_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    kernel, args = _kernel_and_arguments("layer_norm", torch.bfloat16)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = _cpu_bind(kernel, args)
        host = bound.host_function
        assert host is not None
        seeds = CuteResidentReductionHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        assert seeds
        seed = (
            type(seeds[0]).from_dict(
                seeds[0].config
                | {"cute_reduction_schedule": "scalar", "cute_reduction_group_rows": 1}
            )
            if schedule == "scalar"
            else next(
                value for value in seeds if value["cute_reduction_schedule"] == schedule
            )
        )
        source = bound.to_code(seed)
        assert ("_cute_resident_sums" in source) == (schedule != "scalar")
        module = PyCodeCache.load(source)
        backend = bound.env.backend
        backend.annotate_compiled_module(module, source, kernel.name)
        caller = getattr(module, kernel.name)
        (compiled_kernel,) = backend._compiled_kernels(caller, kernel.name)
        loaded = _load_compiled_fn(_serialize_compiled_fn(caller))
        try:
            (reloaded_kernel,) = backend._compiled_kernels(loaded, kernel.name)
            expected_kinds = () if schedule == "scalar" else ("resident_reduction",)
            assert getattr(compiled_kernel, "_helion_cute_helper_kinds", ()) == (
                expected_kinds
            )
            assert getattr(reloaded_kernel, "_helion_cute_helper_kinds", ()) == (
                expected_kinds
            )
        finally:
            _unload_compiled_fn(loaded)

    def key() -> str | None:
        return launcher._cute_disk_cache_key(
            compiled_kernel,
            (),
            (max(seed.num_threads), 1, 1),
            (),
            None,
            "--enable-tvm-ffi",
            148,
        )

    plain = _kernel(None)
    before, plain_before = key(), _key(plain)
    assert before is not None
    helper = source_tree / "_compiler/cute/resident_reduction_runtime.py"
    helper.write_text("# changed ordinary resident device helper\n")
    assert (key() != before) == (schedule != "scalar")
    assert _key(plain) == plain_before
    helper.unlink()
    assert (key() is None) == (schedule != "scalar")
    assert _key(plain) == plain_before


def test_direct_helper_cached_launcher_does_not_rehash_on_relaunch(
    source_tree: Path,
) -> None:
    kernel = _kernel(None)
    kernel._helion_cute_helper_kinds = ("resident_reduction",)
    original = source_dependencies.wrapper_source_dependencies
    with (
        patch.object(launcher, "_create_cute_wrapper", return_value=object()),
        patch.object(
            launcher, "wrapper_source_dependencies", wraps=original
        ) as fingerprint,
    ):
        first = launcher._get_compiled_cute_launcher(kernel, (), (256, 1, 1))
        second = launcher._get_compiled_cute_launcher(kernel, (), (256, 1, 1))
    assert first is second
    assert fingerprint.call_count == 1
    assert first._cache_key is not None


@pytest.mark.parametrize(
    "edited", source_dependencies._WRAPPER_DEPENDENCIES["paired_host_sum"]
)
def test_external_paired_sum_kernel_covers_all_compiled_helper_sources(
    source_tree: Path, edited: str
) -> None:
    first = _kernel(None)
    source_dependencies.set_helper_source_hash(first, "paired_host_sum")
    before = _key(first)
    assert before is not None
    (source_tree / edited).write_text("# changed paired sum implementation\n")
    second = _kernel(None)
    source_dependencies.set_helper_source_hash(second, "paired_host_sum")
    assert _key(second) != before
    assert _key(first) == before  # The already-loaded object remains unchanged.


def test_missing_paired_helper_source_disables_disk_reuse(source_tree: Path) -> None:
    (source_tree / "_compiler/cute/resident_reduction_runtime.py").unlink()
    kernel = _kernel(None)
    source_dependencies.set_helper_source_hash(kernel, "paired_host_sum")
    assert _key(kernel) is None


@pytest.mark.parametrize(
    "edited", source_dependencies._WRAPPER_DEPENDENCIES["single_host_sum"]
)
def test_external_single_sum_kernel_covers_all_compiled_helper_sources(
    source_tree: Path, edited: str
) -> None:
    single, paired = _kernel(None), _kernel(None)
    source_dependencies.set_helper_source_hash(single, "single_host_sum")
    source_dependencies.set_helper_source_hash(paired, "paired_host_sum")
    before = _key(single), _key(paired)
    assert all(value is not None for value in before)
    (source_tree / edited).write_text("# changed single sum dependency\n")
    changed_single, changed_paired = _kernel(None), _kernel(None)
    source_dependencies.set_helper_source_hash(changed_single, "single_host_sum")
    source_dependencies.set_helper_source_hash(changed_paired, "paired_host_sum")
    assert _key(changed_single) != before[0]
    assert (_key(changed_paired) != before[1]) == (
        edited in source_dependencies._WRAPPER_DEPENDENCIES["paired_host_sum"]
    )
    assert _key(single) == before[0]  # Already loaded objects retain their source.


def test_missing_single_helper_source_disables_only_single_disk_reuse(
    source_tree: Path,
) -> None:
    (source_tree / "_compiler/cute/single_sum_runtime.py").unlink()
    single, paired = _kernel(None), _kernel(None)
    source_dependencies.set_helper_source_hash(single, "single_host_sum")
    source_dependencies.set_helper_source_hash(paired, "paired_host_sum")
    assert _key(single) is None
    assert _key(paired) is not None


@pytest.mark.parametrize(
    "edited",
    [
        "_compiler/cute/block_scaled_config.py",
        "_compiler/cute/block_scaled_prepare.py",
        "_compiler/cute/block_scaled_runtime.py",
    ],
)
def test_helper_only_edit_changes_key_without_wrapper_or_plan_change(
    source_tree: Path, edited: str
) -> None:
    scaled = _kernel("block_scaled_mma")
    gathered = _kernel("gathered_mma_tma")
    plain = _kernel(None)
    before = tuple(_key(kernel) for kernel in (scaled, gathered, plain))
    assert all(key is not None for key in before)
    (source_tree / edited).write_text("def device_helper():\n    return 2\n")
    after = tuple(_key(kernel) for kernel in (scaled, gathered, plain))
    assert before[0] != after[0]
    assert before[1:] == after[1:]


def test_missing_required_source_disables_disk_reuse(source_tree: Path) -> None:
    unrelated = _kernel("gathered_mma_tma")
    before = _key(unrelated)
    (source_tree / "_compiler/cute/block_scaled_prepare.py").unlink()
    assert _key(_kernel("block_scaled_mma")) is None
    assert _key(unrelated) == before


def test_each_dependency_is_read_once_for_multiple_plans(source_tree: Path) -> None:
    kinds = ["block_scaled_mma", "gathered_mma_tma", "block_scaled_mma"]
    reads: list[Path] = []
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    with patch.object(Path, "read_bytes", read):
        result = source_dependencies.wrapper_source_dependencies(kinds)
    assert result is not None
    assert len(reads) == len(set(reads)) == len(result)
    assert source_tree / "_compiler/cute/block_scaled_prepare.py" in reads


def test_cached_launcher_does_not_rehash_helpers_on_relaunch(source_tree: Path) -> None:
    kernel = _kernel("block_scaled_mma")
    original = source_dependencies.wrapper_source_dependencies
    with (
        patch.object(launcher, "_create_cute_wrapper", return_value=object()),
        patch.object(
            launcher, "wrapper_source_dependencies", wraps=original
        ) as fingerprint,
    ):
        first = launcher._get_compiled_cute_launcher(kernel, (), (192, 1, 1))
        second = launcher._get_compiled_cute_launcher(kernel, (), (192, 1, 1))
    assert first is second
    assert fingerprint.call_count == 1


def test_source_checkout_path_does_not_enter_key(
    source_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _key(_kernel("block_scaled_mma"))
    other = tmp_path / "second-checkout"
    for relative in (
        *source_dependencies._COMMON_DEPENDENCIES,
        *source_dependencies._WRAPPER_DEPENDENCIES["block_scaled_mma"],
    ):
        filename = other / relative
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes((source_tree / relative).read_bytes())
    monkeypatch.setattr(source_dependencies, "_PACKAGE_ROOT", other)
    assert _key(_kernel("block_scaled_mma")) == before
