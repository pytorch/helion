"""Tests for the portable-suite gap machinery in ``test.backends``."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pytest

from test.backends import GAP_STASH_KEY
from test.backends import PortableGap
from test.backends import apply_gap
from test.backends import matching_gap
from test.backends import validate_gaps
from test.backends import xfail_test
from test.conftest import pytest_collection_modifyitems
from test.conftest import pytest_runtest_makereport

from helion.runtime.ref_mode import RefMode


class _FakeItem:
    def __init__(self, cls: type[unittest.TestCase], name: str, path: Path) -> None:
        self.cls = cls
        self.name = name
        self.instance = cls(name)
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

    def declare(
        self,
        inner_key: str = "SampleCase::test_example",
        *,
        condition: bool = True,
        skip: bool = False,
        reason: str = "synthetic gap",
    ) -> None:
        xfail_test(
            "_unit",
            "test_sample",
            inner_key,
            reason=reason,
            condition=condition,
            skip=skip,
        )


class TestXfailTestAPI(unittest.TestCase):
    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(KeyError):
            xfail_test("_not_registered", "test_sample", "", reason="x")


class TestMatchingGap(_SyntheticBackendTestCase):
    def test_exact_gap_beats_earlier_whole_file_gap(self) -> None:
        self.declare("", reason="whole")
        self.declare(reason="exact")
        self.assertIs(
            matching_gap(self.gaps, "test_sample", "SampleCase::test_example"),
            self.gaps[1],
        )

    def test_exact_gap_beats_later_whole_file_gap(self) -> None:
        self.declare(reason="exact")
        self.declare("", reason="whole")
        self.assertIs(
            matching_gap(self.gaps, "test_sample", "SampleCase::test_example"),
            self.gaps[0],
        )

    def test_false_condition_is_ignored(self) -> None:
        self.declare(condition=False)
        self.assertIsNone(
            matching_gap(self.gaps, "test_sample", "SampleCase::test_example")
        )

    def test_false_exact_gap_does_not_mask_whole_file_gap(self) -> None:
        self.declare(condition=False)
        self.declare("")
        self.assertIs(
            matching_gap(self.gaps, "test_sample", "SampleCase::test_example"),
            self.gaps[1],
        )

    def test_whole_file_gap_matches_other_method(self) -> None:
        self.declare("")
        self.assertIs(
            matching_gap(self.gaps, "test_sample", "SampleCase::test_other"),
            self.gaps[0],
        )


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

    def test_skip_gap_uses_class_local_copy(self) -> None:
        class SampleCase(unittest.TestCase):
            def test_example(self) -> None:
                pass

        original = SampleCase.test_example
        self.declare(skip=True, reason="unsafe")
        item = _FakeItem(SampleCase, "test_example", self.test_path)
        apply_gap(item, self.gaps[0])

        self.assertIsNot(SampleCase.test_example, original)
        self.assertTrue(getattr(SampleCase.test_example, "__unittest_skip__", False))
        self.assertIn(
            "unsafe", getattr(SampleCase.test_example, "__unittest_skip_why__", "")
        )
        self.assertNotIn(GAP_STASH_KEY, item.stash)


class TestCollectionHook(_SyntheticBackendTestCase):
    def make_item(self) -> tuple[_FakeItem, type[unittest.TestCase]]:
        class SampleCase(unittest.TestCase):
            def test_example(self) -> None:
                pass

        return _FakeItem(SampleCase, "test_example", self.test_path), SampleCase

    def test_eager_mode_applies_no_gap(self) -> None:
        self.declare()
        item, cls = self.make_item()
        with mock.patch("test.conftest._get_ref_mode", return_value=RefMode.EAGER):
            pytest_collection_modifyitems([item])
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
            pytest_collection_modifyitems([item])
        self.assertTrue(
            getattr(vars(cls)["test_example"], "__unittest_expecting_failure__", False)
        )
        self.assertIs(item.stash[GAP_STASH_KEY], self.gaps[0])

    def test_item_outside_portable_is_ignored(self) -> None:
        self.declare()
        item, cls = self.make_item()
        item.path = self.portable_dir.parent / "test_other.py"
        pytest_collection_modifyitems([item])
        self.assertFalse(
            getattr(vars(cls)["test_example"], "__unittest_expecting_failure__", False)
        )


class TestReportHook(_SyntheticBackendTestCase):
    def make_gapped_item(self) -> _FakeItem:
        class SampleCase(unittest.TestCase):
            def test_example(self) -> None:
                pass

        self.declare(reason="visible reason")
        item = _FakeItem(SampleCase, "test_example", self.test_path)
        apply_gap(item, self.gaps[0])
        return item

    def run_hook(self, item: _FakeItem, report):
        hook = pytest_runtest_makereport(item)
        next(hook)
        with self.assertRaises(StopIteration) as stopped:
            hook.send(report)
        return stopped.exception.value

    def test_reason_replaces_existing_xfail_reason(self) -> None:
        report = types.SimpleNamespace(
            when="call", skipped=True, failed=False, wasxfail=""
        )
        result = self.run_hook(self.make_gapped_item(), report)
        self.assertIs(result, report)
        self.assertEqual(report.wasxfail, "[_unit] visible reason")

    def test_self_skip_is_not_converted_to_xfail(self) -> None:
        report = types.SimpleNamespace(when="call", skipped=True, failed=False)
        result = self.run_hook(self.make_gapped_item(), report)
        self.assertIs(result, report)
        self.assertFalse(hasattr(report, "wasxfail"))

    def test_unexpected_success_keeps_failure_and_adds_section(self) -> None:
        report = types.SimpleNamespace(
            when="call",
            skipped=False,
            failed=True,
            longrepr="Failed: Unexpected success",
            sections=[],
        )
        result = self.run_hook(self.make_gapped_item(), report)
        self.assertIs(result, report)
        self.assertEqual(report.sections, [("Portable gap", "[_unit] visible reason")])
        self.assertFalse(hasattr(report, "wasxfail"))


class TestGapValidation(_SyntheticBackendTestCase):
    def test_valid_gap(self) -> None:
        self.declare()
        validate_gaps({"test_sample": {"SampleCase::test_example"}})

    def test_inherited_collected_test_is_valid(self) -> None:
        self.declare("DerivedCase::test_inherited")
        validate_gaps({"test_sample": {"DerivedCase::test_inherited"}})

    def test_unknown_inner_key_raises(self) -> None:
        self.declare("SampleCase::test_missing")
        with self.assertRaises(ValueError):
            validate_gaps({"test_sample": {"SampleCase::test_example"}})

    def test_uncollected_file_key_is_not_checked(self) -> None:
        self.declare("SampleCase::test_missing")
        validate_gaps({})

    def test_missing_subpath_raises(self) -> None:
        xfail_test("_unit", "test_missing", "", reason="x")
        with self.assertRaises(FileNotFoundError):
            validate_gaps({})

    def test_duplicate_applicable_gap_raises(self) -> None:
        self.declare(reason="first")
        self.declare(reason="second")
        with self.assertRaises(ValueError):
            validate_gaps({"test_sample": {"SampleCase::test_example"}})

    def test_duplicate_with_inapplicable_first_is_valid(self) -> None:
        self.declare(condition=False, reason="inactive")
        self.declare(reason="active")
        validate_gaps({"test_sample": {"SampleCase::test_example"}})


def test_gap_behavior_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    (tmp_path / "conftest.py").write_text(
        """
from pathlib import Path

import test.backends as gaps
import test.conftest as root
from test.backends import xfail_test
from test.conftest import pytest_collection_modifyitems
from test.conftest import pytest_runtest_makereport
from helion.runtime.ref_mode import RefMode

gaps._PORTABLE_DIR = Path(__file__).parent
gaps._GAPS.clear()
gaps._GAPS["_unit"] = []
for name in ("test_fails", "test_passes", "test_subtest", "test_skips"):
    xfail_test("_unit", "test_probe", f"TestProbe::{name}", reason=f"reason for {name}")
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
"""
    )
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rxXrs",
            "--junitxml=report.xml",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 skipped, 2 xfailed" in result.stdout
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
