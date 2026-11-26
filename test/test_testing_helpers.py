"""Tests of the shared test helpers in ``helion._testing``."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
from pathlib import Path
import runpy
import shutil
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion import _compat
from helion import _hardware
from helion._compiler.cute.autotune.search_surface import CuteSearchSurface
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.target_facts import B200
from helion._testing import cpu_render_target
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import pairwise
from helion._testing import patch_target_facts
from helion._testing import sharding
from helion._testing.cute_codegen import CpuRefMode
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target
from helion._testing.sharding import MIN_WEIGHT
from helion._testing.sharding import assign_shards
from helion._testing.sharding import load_durations
from helion._testing.sharding import shard_loads
from helion.autotuner import config_spec
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.search_surface import BackendSearchSurface
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@onlyBackends(["cute"])
def test_direct_cute_config_uses_the_scoped_target() -> None:
    probe = _compat.target_device_capability
    sm_probe = config_spec.device_num_sm
    with (
        patch.object(_compat, "target_device_capability", return_value=(9, 0)),
        patch.object(config_spec, "device_num_sm", return_value=120),
    ):
        with cpu_render_target(capability=(10, 3), num_sm=148):
            outer = ConfigSpec(backend=CuteBackend())
            assert (outer.target_device_capability, outer.num_sm) == ((10, 3), 148)
            with patch_target_facts(capability=None, num_sm=17):
                inner = ConfigSpec(backend=CuteBackend())
                assert (inner.target_device_capability, inner.num_sm) == (None, 17)
            restored = ConfigSpec(backend=CuteBackend())
            assert (restored.target_device_capability, restored.num_sm) == (
                (10, 3),
                148,
            )
        external = ConfigSpec(backend=CuteBackend())
        assert (external.target_device_capability, external.num_sm) == ((9, 0), 120)
    assert _compat.target_device_capability is probe
    assert config_spec.device_num_sm is sm_probe


@onlyBackends(["cute"])
def test_direct_config_uses_its_device_for_an_implicit_capability() -> None:
    b200 = torch.device("cuda:1")
    with (
        patch.object(
            _compat,
            "target_device_capability",
            side_effect=lambda device=None: (10, 0) if device == b200 else (9, 0),
        ) as probe,
        patch.object(
            config_spec,
            "device_num_sm",
            side_effect=lambda device: 148 if device == b200 else 120,
        ),
    ):
        backend = CuteBackend()
        current = ConfigSpec(backend=backend)
        requested = ConfigSpec(backend=backend, device=b200)
        assert (current.target_device_capability, current.num_sm) == ((9, 0), 120)
        assert (requested.target_device_capability, requested.num_sm) == ((10, 0), 148)
        probe.reset_mock()
        unknown = ConfigSpec(
            backend=backend, device=b200, target_device_capability=None
        )
        explicit = ConfigSpec(
            backend=backend, device=b200, target_device_capability=(8, 0)
        )
        assert (unknown.target_device_capability, unknown.num_sm) == (None, 148)
        assert explicit.target_device_capability == (8, 0)
        probe.assert_not_called()


@onlyBackends(["cute"])
def test_unavailable_cuda_skips_cold_and_warm_indexed_capability_caches() -> None:
    # A fresh instance of the actual indexed cache leaves other tests' state
    # alone and makes the cold-to-warm transition deterministic.
    cached = functools.cache(_compat._target_device_capability.__wrapped__)
    device = torch.device("cuda:0")
    backend = CuteBackend()
    with (
        patch.object(_compat, "_target_device_capability", cached),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ) as hardware,
    ):
        for start_size in (0, 1):
            before = cached.cache_info()
            assert before.currsize == start_size
            with mock_cuda_unavailable():
                direct = ConfigSpec(backend=backend, device=device, num_sm=148)
                assert direct.target_device_capability is None
            assert cached.cache_info() == before
            assert _compat.target_device_capability(device) == (10, 0)
        hardware.assert_called_once_with(0)

        # Explicit test models are above the raw cache, so an outer model
        # remains effective inside the CUDA-denied renderer.
        with patch_target_facts(capability=(9, 0)), mock_cuda_unavailable():
            assert ConfigSpec(
                backend=backend, device=device
            ).target_device_capability == (9, 0)


def test_indexed_capability_retries_after_cuda_becomes_visible() -> None:
    cached = functools.cache(_compat._target_device_capability.__wrapped__)
    device = torch.device("cuda:0")
    with (
        patch.object(_compat, "_target_device_capability", cached),
        patch.object(torch.cuda, "is_available", return_value=False) as visible,
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)) as read,
    ):
        assert _compat.target_device_capability(device) is None
        assert cached.cache_info().currsize == 0
        read.assert_not_called()

        visible.return_value = True
        assert _compat.target_device_capability(device) == (10, 0)
        visible.reset_mock()
        assert _compat.target_device_capability(device) == (10, 0)
        visible.assert_not_called()
        read.assert_called_once_with(0)
        assert cached.cache_info().hits == 1


@onlyBackends(["cute"])
def test_direct_config_handles_only_an_unavailable_cuda_ordinal() -> None:
    # Exercise Torch's actual get_device_capability -> get_device_properties
    # validation with one visible ordinal; only device discovery is simulated.
    cached = functools.cache(_compat._target_device_capability.__wrapped__)
    backend = CuteBackend()
    device = torch.device("cuda:1")
    with (
        mock_cuda_unavailable(),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(_compat, "_target_device_capability", cached),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(config_spec, "get_num_xcd", return_value=1),
        patch.object(torch.cuda, "_lazy_init"),
        patch.object(torch.cuda, "device_count", return_value=1) as count,
        patch.object(
            torch.cuda,
            "_get_device_properties",
            create=True,
            return_value=SimpleNamespace(major=10, minor=0),
        ) as properties,
    ):

        def capability() -> tuple[int, int] | None:
            return ConfigSpec(
                backend=backend, device=device, num_sm=1
            ).target_device_capability

        assert capability() is None
        assert cached.cache_info().currsize == 0
        properties.assert_not_called()
        with patch_target_facts(capability=(9, 0)):
            assert capability() == (9, 0)
        assert ConfigSpec(
            backend=backend, device=device, num_sm=1, target_device_capability=(8, 0)
        ).target_device_capability == (8, 0)

        count.return_value = 2
        assert capability() == (10, 0)
        before = count.call_count
        assert capability() == (10, 0)
        assert count.call_count == before
        properties.assert_called_once_with(1)

        for available, error in (
            (1, AssertionError("unrelated assertion")),
            (2, AssertionError("Invalid device id")),
            (1, RuntimeError("driver failed")),
        ):
            count.return_value = available
            with (
                patch.object(
                    _compat,
                    "target_device_capability",
                    side_effect=error,
                ),
                pytest.raises(type(error), match=str(error)),
            ):
                capability()


@onlyBackends(["cute"])
def test_direct_config_keeps_legacy_capability_hook() -> None:
    class PluginSurface(BackendSearchSurface):
        def default_target_device_capability(self) -> tuple[int, int]:
            return (7, 0)

    class SpecializedSurface(CuteSearchSurface):
        def default_target_device_capability(self) -> tuple[int, int]:
            return (8, 0)

    backend = CuteBackend()
    with patch_target_facts(capability=(10, 0), num_sm=148):
        for surface, expected in (
            (PluginSurface(), (7, 0)),
            (SpecializedSurface(), (8, 0)),
        ):
            with patch.object(backend, "config_spec_extension", return_value=surface):
                assert ConfigSpec(backend=backend).target_device_capability == expected
                assert (
                    ConfigSpec(
                        backend=backend, device=torch.device("cuda:1")
                    ).target_device_capability
                    == expected
                )
        with patch.object(
            CuteSearchSurface, "default_target_device_capability", lambda self: (8, 9)
        ):
            assert ConfigSpec(
                backend=backend, device=torch.device("cuda:1")
            ).target_device_capability == (8, 9)


@onlyBackends(["triton", "cute"])
def test_cpu_render_target_constructs_cache_without_host_cuda_runtime(monkeypatch):
    @helion.kernel()
    def pointwise(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + 1
        return out

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    args = (torch.empty(8),)
    with cpu_render_target():
        assert not torch.cuda.is_available()
        bound = cpu_bind(pointwise, args)
        with (
            patch(
                "helion.autotuner.local_cache.extract_device",
                return_value=torch.device("cuda:0"),
            ),
            patch(
                "helion.autotuner.local_cache.get_device_name",
                return_value=B200.hardware_name,
            ),
            patch("torch.cuda.is_available", return_value=True),
        ):
            cache = LocalAutotuneCache(BaseSearch(bound, args))
        assert cache.key.runtime_name == B200.runtime_version
        assert cache.key.hardware == B200.hardware_name
    assert torch.version.cuda is None and torch.version.hip is None


def test_patch_target_runtime_versions_restore_nested_models(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "host-cuda")
    monkeypatch.setattr(torch.version, "hip", "host-hip")
    cuda = dataclasses.replace(B200.hardware_info(), runtime_version="modeled-cuda")
    hip = dataclasses.replace(
        cuda,
        device_kind="rocm",
        hardware_name="AMD Instinct MI350X",
        compute_capability="gfx950",
        runtime_version="modeled-hip",
    )
    with patch_target_facts(hardware=cuda, is_hip=False):
        assert (torch.version.cuda, torch.version.hip) == ("modeled-cuda", None)
        with (
            pytest.raises(RuntimeError, match="leave nested model"),
            patch_target_facts(hardware=hip, is_hip=True),
        ):
            assert (torch.version.cuda, torch.version.hip) == (None, "modeled-hip")
            assert _hardware.get_hardware_info() is hip
            assert _compat.is_hip()
            raise RuntimeError("leave nested model")
        assert (torch.version.cuda, torch.version.hip) == ("modeled-cuda", None)
        assert _hardware.get_hardware_info() is cuda
        assert not _compat.is_hip()
    assert (torch.version.cuda, torch.version.hip) == ("host-cuda", "host-hip")


def test_tcgen05_target_models_runtime_and_restores_nested_contexts(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "host-hip")
    expected = (B200.runtime_version, None)
    with tcgen05_target():
        assert (torch.version.cuda, torch.version.hip) == expected
        with pytest.raises(RuntimeError, match="leave tcgen05 model"), tcgen05_target():
            assert (torch.version.cuda, torch.version.hip) == expected
            raise RuntimeError("leave tcgen05 model")
        assert (torch.version.cuda, torch.version.hip) == expected
    assert (torch.version.cuda, torch.version.hip) == (None, "host-hip")


@pytest.mark.parametrize("device_kind,is_hip", [("cuda", True), ("rocm", False)])
def test_patch_target_runtime_versions_reject_inconsistent_models(device_kind, is_hip):
    outer = dataclasses.replace(B200.hardware_info(), runtime_version="outer-cuda")
    inconsistent = dataclasses.replace(outer, device_kind=device_kind)
    with patch_target_facts(hardware=outer, is_hip=False):
        with (
            pytest.raises(
                AssertionError, match="is_hip must match hardware.device_kind"
            ),
            patch_target_facts(hardware=inconsistent, is_hip=is_hip),
        ):
            pass
        assert _hardware.get_hardware_info() is outer
        assert not _compat.is_hip()
        assert (torch.version.cuda, torch.version.hip) == ("outer-cuda", None)


def test_patch_target_runtime_versions_leave_unspecified_models_untouched(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "host-hip")
    hardware = B200.hardware_info()

    def uncalled_probe(*args):
        raise AssertionError("an opaque hardware probe must not be evaluated")

    with patch_target_facts(hardware=uncalled_probe, is_hip=False):
        assert (torch.version.cuda, torch.version.hip) == (None, "host-hip")
    with patch_target_facts(hardware=hardware):
        assert (torch.version.cuda, torch.version.hip) == (None, "host-hip")


def test_import_path_supports_dataclass_annotations_and_cached_defaults(tmp_path):
    path = tmp_path / "helper_dataclass.py"
    name = "helion._testing.helper_dataclass"
    path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations
            from dataclasses import dataclass, fields
            from typing import ClassVar

            @dataclass(slots=True)
            class Defaults:
                marker: ClassVar[int] = 7
                scale: float = -0.0

            default = Defaults()
            def call(value=default):
                return value
            """
        )
    )
    try:
        module = import_path(path)
        assert [field.name for field in module.fields(module.Defaults)] == ["scale"]
        assert module.call() is module.default
        assert module.call().scale.hex() == (-0.0).hex()
        assert import_path(path) is module
        assert sys.modules[name] is module
    finally:
        sys.modules.pop(name, None)


def test_import_path_failed_execution_does_not_publish_partial_module(tmp_path):
    path = tmp_path / "helper_failed_import.py"
    name = "helion._testing.helper_failed_import"
    path.write_text(
        "import sys\n"
        "published_during_execution = sys.modules[__name__]\n"
        "raise RuntimeError('failed during execution')\n"
    )
    try:
        with pytest.raises(RuntimeError, match="failed during execution"):
            import_path(path)
        assert name not in sys.modules
        path.write_text("value = 2\n")
        assert import_path(path).value == 2
    finally:
        sys.modules.pop(name, None)


def _assert_all_pairs_covered(rows: list[tuple[object, ...]], **axes: tuple) -> None:
    names = list(axes)
    for i, j in itertools.combinations(range(len(names)), 2):
        seen = {(row[i], row[j]) for row in rows}
        assert seen == set(itertools.product(axes[names[i]], axes[names[j]]))


def test_pairwise_covers_every_pair_of_eight_flags() -> None:
    flags = {f"flag{i}": (False, True) for i in range(8)}
    rows = pairwise(**flags)
    _assert_all_pairs_covered(rows, **flags)
    assert len(rows) == 9  # a full product would be 256 rows


def test_pairwise_mixes_axis_sizes_and_keeps_axis_order() -> None:
    axes = {
        "dtype": ("f16", "bf16", "f32"),
        "lanes": (1, 2, 4),
        "largest": (True, False),
    }
    rows = pairwise(**axes)
    _assert_all_pairs_covered(rows, **axes)
    assert len(rows) == 9
    assert all(
        row[0] in axes["dtype"]
        and row[1] in axes["lanes"]
        and row[2] in axes["largest"]
        for row in rows
    )


def test_pairwise_is_deterministic() -> None:
    axes = {"a": (1, 2, 3), "b": ("x", "y"), "c": (0.5, 1.5)}
    assert pairwise(**axes) == pairwise(**axes)


def test_pairwise_single_axis_lists_every_value() -> None:
    assert pairwise(width=(16, 32, 64)) == [(16,), (32,), (64,)]


def test_pairwise_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError, match="no values"):
        pairwise(a=(1, 2), b=())


def test_assign_shards_balances_recorded_durations() -> None:
    durations = {"a": 10.0, "b": 9.0, "c": 5.0, "d": 4.0, "e": 1.0, "f": 1.0}
    assignment = assign_shards(durations, 2, durations)
    assert sorted(assignment) == sorted(durations)
    assert assignment["a"] != assignment["b"]  # the two heaviest stay apart
    assert shard_loads(assignment, 2, durations) == [15.0, 15.0]


def test_assign_shards_is_deterministic_in_module_order() -> None:
    durations = {f"m{i}": float(i % 7 + 1) for i in range(40)}
    modules = list(durations)
    forward = assign_shards(modules, 3, durations)
    backward = assign_shards(reversed(modules), 3, durations)
    assert forward == backward
    assert set(forward.values()) == {0, 1, 2}


def test_assign_shards_weighs_unrecorded_modules_as_the_median() -> None:
    durations = {"heavy": 100.0, "light1": 1.0, "light2": 1.0}
    assignment = assign_shards(
        ["heavy", "light1", "light2", "new1", "new2"], 2, durations
    )
    # The new modules weigh 1.0 (the median), so everything else fits opposite
    # the heavy module.
    assert {m for m, s in assignment.items() if s == assignment["heavy"]} == {"heavy"}


def test_assign_shards_single_shard_and_invalid_count() -> None:
    assert assign_shards(["a", "b"], 1, {}) == {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="positive"):
        assign_shards(["a"], 0, {})


def test_assign_shards_counts_zero_second_modules_at_the_floor() -> None:
    durations = {"heavy": 30.0, "mid": 20.0, **{f"skipped{i}": 0.0 for i in range(20)}}
    assignment = assign_shards(durations, 2, durations)
    # Each skipped module weighs MIN_WEIGHT, so they fill the lighter shard up
    # to the heavier one and then alternate, instead of all tying onto one.
    assert shard_loads(assignment, 2, durations) == [
        30.0 + 5 * MIN_WEIGHT,
        20.0 + 15 * MIN_WEIGHT,
    ]
    assert assignment["heavy"] != assignment["mid"]


def test_recorded_shard_durations_name_existing_test_modules() -> None:
    durations = load_durations()
    assert durations, "test/shard_durations.json is missing or empty"
    root = Path(__file__).resolve().parents[1]
    missing = [m for m in durations if not (root / m).is_file()]
    assert not missing, (
        f"run scripts/shard_durations.py (without reports it drops them): {missing}"
    )
    assert all(m.startswith("test/") and m.endswith(".py") for m in durations)
    assert all(seconds >= 0 for seconds in durations.values())


@pytest.mark.parametrize("explicit_output", [False, True])
def test_shard_duration_script_uses_its_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_output: bool,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "test").mkdir()
    script = checkout / "scripts" / "shard_durations.py"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts" / script.name, script
    )
    for name in ("test_kept.py", "test_small.py", "test_measured_zero.py"):
        (checkout / "test" / name).touch()
    (checkout / "test" / "test_unreported.py").touch()
    (checkout / "test" / "test_legacy_zero.py").touch()

    # The helper is imported from another checkout. Redirect its duration path
    # to a sentinel so the regression cannot overwrite that checkout's real file.
    foreign_output = tmp_path / "installed-durations.json"
    foreign_contents = '{"test/test_unreported.py": 99.0}\n'
    foreign_output.write_text(foreign_contents)
    monkeypatch.setattr(sharding, "DURATIONS_FILE", foreign_output)
    # Keep even a mistaken cwd-relative output path inside the temporary tree.
    monkeypatch.chdir(tmp_path)

    local_output = checkout / "test" / "shard_durations.json"
    local_contents = (
        '{"test/test_kept.py": 1.0, "test/test_deleted.py": 2.0, '
        '"test/test_unreported.py": 3.0, "test/test_legacy_zero.py": 0.0}\n'
    )
    local_output.write_text(local_contents)
    output = tmp_path / "explicit.json" if explicit_output else local_output
    if explicit_output:
        output.write_text('{"test/test_unreported.py": 4.0}\n')
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="test.test_kept.TestKept" time="3.5"/>'
        '<testcase classname="test.test_small.TestSmall" time="0.004"/>'
        '<testcase classname="test.test_measured_zero.TestZero" time="0.0"/>'
        "</testsuite>"
    )
    args = [str(report)]
    if explicit_output:
        args.extend(["--output", str(output)])

    namespace = runpy.run_path(str(script))
    assert namespace["read_junit"](report, checkout) == {
        "test/test_kept.py": 3.5,
        "test/test_small.py": 0.004,
        "test/test_measured_zero.py": 0.0,
    }
    assert namespace["main"](args) == 0
    assert json.loads(output.read_text()) == {
        "test/test_kept.py": 3.5,
        "test/test_small.py": 0.1,
        "test/test_measured_zero.py": 0.0,
        "test/test_unreported.py": 4.0 if explicit_output else 3.0,
        **({} if explicit_output else {"test/test_legacy_zero.py": 0.0}),
    }
    assert foreign_output.read_text() == foreign_contents
    if explicit_output:
        assert local_output.read_text() == local_contents


def test_cpu_ref_mode_matmuls_follow_torch() -> None:
    """``CpuRefMode`` multiplies in FP32 where CPU ``torch.mm`` has no
    ``out_dtype`` (``hl.dot``'s reference path) and, as ``torch.addmm``,
    ignores the bias under a zero ``beta``, NaN included."""
    lhs = torch.randn(3, 5, dtype=torch.bfloat16)
    rhs = torch.randn(5, 2, dtype=torch.bfloat16)
    bias = torch.randn(3, 2)
    nan = torch.full((3, 2), float("nan"))
    product = lhs.float() @ rhs.float()
    scaled = 2 * product + bias
    with CpuRefMode():
        dot = torch.mm(lhs, rhs, out_dtype=torch.float32)
        plain = torch.mm(lhs, rhs)
        ignored = torch.addmm(nan, lhs, rhs, beta=0)
        added = torch.addmm(bias, lhs, rhs, alpha=2)
    assert torch.equal(dot, product)
    assert plain.dtype is torch.bfloat16
    assert torch.equal(ignored, product)
    assert torch.equal(added, scaled)
