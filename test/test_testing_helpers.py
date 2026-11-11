"""Tests of the shared test helpers in ``helion._testing``."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import pytest
import torch

import helion
from helion import _compat
from helion import _hardware
from helion._compiler.cute.target_facts import B200
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import pairwise
from helion._testing import patch_target_facts
from helion._testing import sharding
from helion._testing.cute_codegen import cpu_bind
from helion._testing.sharding import MIN_WEIGHT
from helion._testing.sharding import assign_shards
from helion._testing.sharding import load_durations
from helion._testing.sharding import shard_loads
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.local_cache import LocalAutotuneCache
import helion.language as hl

pytestmark = pytest.mark.cpu_only


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
