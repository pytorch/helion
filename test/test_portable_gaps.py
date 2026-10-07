"""Tests for the portable-suite gap machinery in ``test.backends``."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Callable
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pytest

from test.backends import GAP_STASH_KEY
from test.backends import PortableGap
from test.backends import active_gaps
from test.backends import apply_gap
from test.backends import backend_gap
from test.backends import matching_gap
from test.backends import validate_gaps
from test.conftest import pytest_collection_modifyitems

from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
from helion.runtime.ref_mode import RefMode


class _FakeItem:
    def __init__(
        self,
        cls: type[unittest.TestCase],
        name: str,
        path: Path,
        module: types.ModuleType | None = None,
    ) -> None:
        self.cls = cls
        self.name = name
        self.instance = cls(name)
        self.module = module or types.ModuleType(cls.__module__)
        setattr(self.module, cls.__name__, cls)
        self.obj = getattr(self.instance, name)
        self.path = path
        self.nodeid = f"test/portable/{path.name}::{cls.__name__}::{name}"
        self.stash = pytest.Stash()


class _SyntheticBackendTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gaps: list[PortableGap] = []
        registry = mock.patch.dict(
            "test.backends._GAPS", {"_unit": self.gaps}, clear=True
        )
        registry.start()
        self.addCleanup(registry.stop)

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.portable_dir = Path(temporary_directory.name)
        self.test_path = self.portable_dir / "test_sample.py"
        self.test_path.write_text("# synthetic portable test\n")
        portable_dir = mock.patch("test.backends._PORTABLE_DIR", self.portable_dir)
        portable_dir.start()
        self.addCleanup(portable_dir.stop)

        class SampleCase(unittest.TestCase):
            def test_example(self) -> None:
                pass

            def test_other(self) -> None:
                pass

        self.sample_case = SampleCase
        self.module = types.ModuleType(SampleCase.__module__)
        self.module.SampleCase = SampleCase
        self.config = types.SimpleNamespace(
            option=types.SimpleNamespace(runxfail=False)
        )

    def declare(
        self,
        inner_key: str = "SampleCase::test_example",
        *,
        condition: bool | Callable[[], bool] = True,
        skip: bool = False,
        reason: str = "synthetic gap",
    ) -> None:
        condition_fn = condition if callable(condition) else lambda: condition
        self.gaps.append(
            backend_gap(
                "test_sample",
                inner_key,
                reason=reason,
                condition=condition_fn,
                skip=skip,
            )
        )


@onlyBackends(["triton"])
class TestMatchingGap(_SyntheticBackendTestCase):
    def matching(self, class_and_method: str) -> PortableGap | None:
        validate_gaps({"test_sample": self.module})
        gaps = active_gaps("_unit")
        return matching_gap(gaps, "test_sample", class_and_method)

    def test_exact_gap_beats_earlier_whole_file_gap(self) -> None:
        self.declare("", reason="whole")
        self.declare(reason="exact")
        self.assertEqual(
            self.matching("SampleCase::test_example").reason, "[_unit] exact"
        )

    def test_exact_gap_beats_later_whole_file_gap(self) -> None:
        self.declare(reason="exact")
        self.declare("", reason="whole")
        self.assertEqual(
            self.matching("SampleCase::test_example").reason, "[_unit] exact"
        )

    def test_false_condition_is_ignored(self) -> None:
        self.declare(condition=False)
        self.assertIsNone(self.matching("SampleCase::test_example"))

    def test_false_exact_gap_does_not_mask_whole_file_gap(self) -> None:
        self.declare(condition=False)
        self.declare("")
        self.assertEqual(
            self.matching("SampleCase::test_example").reason,
            "[_unit] synthetic gap",
        )

    def test_whole_file_gap_matches_other_method(self) -> None:
        self.declare("")
        self.assertEqual(
            self.matching("SampleCase::test_other").reason,
            "[_unit] synthetic gap",
        )


@onlyBackends(["triton"])
class TestApplyGap(_SyntheticBackendTestCase):
    def test_expected_failure_uses_class_local_unbound_wrapper(self) -> None:
        calls: list[unittest.TestCase] = []

        class SharedBase(unittest.TestCase):
            def test_shared(self) -> None:
                calls.append(self)

        class GappedCase(SharedBase):
            pass

        class OtherCase(SharedBase):
            pass

        original = SharedBase.test_shared
        self.declare("", reason="shared gap")
        item = _FakeItem(GappedCase, "test_shared", self.test_path)
        apply_gap(item, self.gaps[0])

        self.assertIsNot(vars(GappedCase)["test_shared"], original)
        self.assertTrue(
            getattr(GappedCase.test_shared, "__unittest_expecting_failure__", False)
        )
        self.assertIs(OtherCase.test_shared, original)
        self.assertFalse(getattr(original, "__unittest_expecting_failure__", False))

        second_instance = GappedCase("test_shared")
        second_instance.test_shared()
        self.assertEqual(calls, [second_instance])
        self.assertIs(item.obj.__self__, item.instance)


@onlyBackends(["triton"])
class TestCollectionHook(_SyntheticBackendTestCase):
    def make_item(self) -> tuple[_FakeItem, type[unittest.TestCase]]:
        return (
            _FakeItem(self.sample_case, "test_example", self.test_path, self.module),
            self.sample_case,
        )

    def test_eager_mode_applies_no_gap(self) -> None:
        self.declare()
        item, cls = self.make_item()
        with mock.patch("test.conftest._get_ref_mode", return_value=RefMode.EAGER):
            pytest_collection_modifyitems(self.config, [item])
        self.assertFalse(
            getattr(vars(cls)["test_example"], "__unittest_expecting_failure__", False)
        )

    def test_compiled_mode_applies_gap(self) -> None:
        self.declare()
        item, cls = self.make_item()
        with (
            mock.patch("test.conftest._get_ref_mode", return_value=RefMode.OFF),
            mock.patch("test.conftest._get_backend", return_value="_unit"),
        ):
            pytest_collection_modifyitems(self.config, [item])
        self.assertTrue(
            getattr(vars(cls)["test_example"], "__unittest_expecting_failure__", False)
        )
        self.assertEqual(item.stash[GAP_STASH_KEY].reason, "[_unit] synthetic gap")

    def test_item_outside_portable_is_ignored(self) -> None:
        self.declare()
        item, cls = self.make_item()
        item.path = self.portable_dir.parent / "test_other.py"
        pytest_collection_modifyitems(self.config, [item])
        self.assertFalse(
            getattr(vars(cls)["test_example"], "__unittest_expecting_failure__", False)
        )

    def test_plain_pytest_function_is_rejected(self) -> None:
        item = types.SimpleNamespace(
            cls=None,
            nodeid="test/portable/test_sample.py::test_example",
            path=self.test_path,
        )
        with self.assertRaisesRegex(
            pytest.UsageError, "must be a unittest.TestCase method"
        ):
            pytest_collection_modifyitems(self.config, [item])

    def test_runxfail_still_validates_registry(self) -> None:
        self.declare("SampleCase::test_missing")
        item, _ = self.make_item()
        self.config.option.runxfail = True
        with self.assertRaises(pytest.UsageError):
            pytest_collection_modifyitems(self.config, [item])


@onlyBackends(["triton"])
class TestGapValidation(_SyntheticBackendTestCase):
    def test_valid_gap(self) -> None:
        condition = mock.Mock(return_value=True)
        self.declare(condition=condition)
        validate_gaps({"test_sample": self.module})
        active_gaps("_unit")
        condition.assert_called_once_with()

    def test_inherited_collected_test_is_valid(self) -> None:
        class BaseCase(unittest.TestCase):
            def test_inherited(self) -> None:
                pass

        class DerivedCase(BaseCase):
            pass

        module = types.ModuleType(DerivedCase.__module__)
        module.DerivedCase = DerivedCase
        self.declare("DerivedCase::test_inherited")
        validate_gaps({"test_sample": module})

    def test_imported_collected_test_is_valid(self) -> None:
        module = types.ModuleType("portable_module")
        module.SampleCase = self.sample_case
        self.declare()
        validate_gaps({"test_sample": module})

    def test_unknown_inner_key_raises(self) -> None:
        self.declare("SampleCase::test_missing")
        with self.assertRaises(pytest.UsageError):
            validate_gaps({"test_sample": self.module})

    def test_uncollected_file_key_is_not_checked(self) -> None:
        self.declare("SampleCase::test_missing")
        validate_gaps({})

    def test_missing_subpath_raises(self) -> None:
        self.gaps.append(backend_gap("test_missing", "", reason="x"))
        with self.assertRaises(pytest.UsageError):
            validate_gaps({})

    def test_duplicate_applicable_gap_raises(self) -> None:
        self.declare(reason="first")
        self.declare(reason="second")
        with self.assertRaises(pytest.UsageError):
            active_gaps("_unit")

    def test_duplicate_with_inapplicable_first_is_valid(self) -> None:
        self.declare(condition=False, reason="inactive")
        self.declare(reason="active")
        active_gaps("_unit")

    def test_inactive_backend_condition_is_not_evaluated(self) -> None:
        condition = mock.Mock(side_effect=AssertionError("must remain lazy"))
        other_gaps: list[PortableGap] = []
        with mock.patch.dict(
            "test.backends._GAPS", {"_unit": self.gaps, "_other": other_gaps}
        ):
            other_gaps.append(
                backend_gap(
                    "test_sample",
                    "SampleCase::test_example",
                    reason="other backend",
                    condition=condition,
                )
            )
            active_gaps("_unit")
        condition.assert_not_called()


@skipUnlessBackends(["triton"])
def test_gap_behavior_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    (tmp_path / "conftest.py").write_text(
        """
from pathlib import Path

import test.backends as gaps
import test.conftest as root
from test.backends import backend_gap
from test.conftest import pytest_collection_modifyitems
from test.conftest import pytest_runtest_makereport
from helion.runtime.ref_mode import RefMode

gaps._PORTABLE_DIR = Path(__file__).parent
gaps._GAPS.clear()
gaps._GAPS["_unit"] = []
for name in ("test_fails", "test_passes", "test_subtest", "test_skips"):
    gaps._GAPS["_unit"].append(
        backend_gap("test_probe", f"TestProbe::{name}", reason=f"reason for {name}")
    )
gaps._GAPS["_unit"].append(
    backend_gap("test_probe", "TestProbe::test_unsafe", reason="unsafe", skip=True)
)
root._get_backend = lambda: "_unit"
root._get_ref_mode = lambda: RefMode.OFF
"""
    )
    (tmp_path / "test_probe.py").write_text(
        """
import unittest

class TestProbe(unittest.TestCase):
    def test_fails(self):
        self.fail("expected")

    def test_passes(self):
        pass

    def test_subtest(self):
        with self.subTest(value=1):
            self.fail("expected subtest")

    def test_skips(self):
        self.skipTest("runtime prerequisite missing")

    def test_unsafe(self):
        self.fail("unsafe body ran")
"""
    )
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root), env.get("PYTHONPATH")) if part
    )
    env.pop("PYTEST_ADDOPTS", None)

    def run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--color=no", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    result = run_pytest("-q", "-rxXs", "--junitxml=report.xml")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 2 skipped, 2 xfailed" in result.stdout
    assert "runtime prerequisite missing" in result.stdout
    assert "reason for test_fails" in result.stdout
    assert "reason for test_passes" in result.stdout
    assert "reason for test_subtest" in result.stdout

    cases = {
        case.attrib["name"]: case
        for case in ET.parse(tmp_path / "report.xml").iter("testcase")
    }
    assert cases["test_passes"].find("failure") is not None
    assert cases["test_passes"].find("skipped") is None
    skipped = cases["test_skips"].find("skipped")
    assert skipped is not None
    assert skipped.attrib["message"] == "runtime prerequisite missing"

    selected = run_pytest("-q", "-rxXs", "test_probe.py::TestProbe::test_fails")
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert "1 xfailed" in selected.stdout

    runxfail = run_pytest("-q", "--runxfail", "test_probe.py::TestProbe::test_fails")
    assert runxfail.returncode == 1, runxfail.stdout + runxfail.stderr
    assert "AssertionError: expected" in runxfail.stdout
    assert "_pytest.outcomes.XFailed" not in runxfail.stdout

    unsafe = run_pytest("-q", "--runxfail", "test_probe.py::TestProbe::test_unsafe")
    assert unsafe.returncode == 0, unsafe.stdout + unsafe.stderr
    assert "1 skipped" in unsafe.stdout
    assert "unsafe body ran" not in unsafe.stdout
