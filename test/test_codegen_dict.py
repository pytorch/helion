from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import cast
import unittest
from unittest import mock

from helion._compiler import backend_registry
from helion.language._decorators import APIFunc
from helion.language._decorators import CodegenDict
from helion.language._decorators import api
from helion.language._decorators import codegen


def _stub_common(state: object) -> str:
    return "common"


def _stub_triton(state: object) -> str:
    return "triton"


def _stub_pallas(state: object) -> str:
    return "pallas"


class TestCodegenDict(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("helion._compiler.backend_registry.repair_backend_codegen")
        self.repair_backend_codegen = patcher.start()
        self.addCleanup(patcher.stop)

    def test_getitem_exact_match(self) -> None:
        d = CodegenDict({"triton": _stub_triton, "common": _stub_common})
        self.assertIs(d["triton"], _stub_triton)
        self.assertIs(d["common"], _stub_common)
        self.repair_backend_codegen.assert_called_once_with("triton")

    def test_getitem_falls_back_to_common(self) -> None:
        d = CodegenDict({"common": _stub_common})
        self.assertIs(d["pallas"], _stub_common)
        self.repair_backend_codegen.assert_called_once_with("pallas")

    def test_getitem_repairs_before_falling_back_to_common(self) -> None:
        d = CodegenDict({"common": _stub_common})

        def repair(key: str) -> None:
            d[key] = _stub_triton

        self.repair_backend_codegen.side_effect = repair
        self.assertIs(d["triton"], _stub_triton)
        self.repair_backend_codegen.assert_called_once_with("triton")

    def test_getitem_raises_when_empty(self) -> None:
        d: CodegenDict = CodegenDict()
        with self.assertRaises(KeyError):
            d["triton"]

    def test_getitem_raises_when_no_common(self) -> None:
        d = CodegenDict({"triton": _stub_triton})
        with self.assertRaises(KeyError):
            d["pallas"]

    def test_getitem_common_raises_when_not_set(self) -> None:
        d: CodegenDict = CodegenDict()
        with self.assertRaises(KeyError):
            d["common"]
        self.repair_backend_codegen.assert_not_called()

    def test_get_falls_back_to_common(self) -> None:
        d = CodegenDict({"common": _stub_common})
        self.assertIs(d.get("triton"), _stub_common)

    def test_get_returns_exact_match(self) -> None:
        d = CodegenDict({"triton": _stub_triton, "common": _stub_common})
        self.assertIs(d.get("triton"), _stub_triton)

    def test_get_returns_none_when_empty(self) -> None:
        d: CodegenDict = CodegenDict()
        self.assertIsNone(d.get("triton"))

    def test_get_returns_default_when_no_common(self) -> None:
        d = CodegenDict({"triton": _stub_triton})
        self.assertIs(d.get("pallas", _stub_pallas), _stub_pallas)

    def test_get_returns_none_for_missing_common(self) -> None:
        d: CodegenDict = CodegenDict()
        self.assertIsNone(d.get("common"))

    def test_prefers_backend_over_common(self) -> None:
        d = CodegenDict({"triton": _stub_triton, "common": _stub_common})
        self.assertIs(d["triton"], _stub_triton)
        self.assertIs(d.get("triton"), _stub_triton)

    def test_setitem_and_lookup(self) -> None:
        d: CodegenDict = CodegenDict()
        d["common"] = _stub_common
        d["triton"] = _stub_triton
        self.assertIs(d["triton"], _stub_triton)
        self.assertIs(d["pallas"], _stub_common)

    def test_codegen_rejects_duplicate_registration(self) -> None:
        @api()
        def op() -> None:
            return None

        codegen(op, "triton")(_stub_triton)
        with self.assertRaisesRegex(
            AssertionError, "codegen already registered for backend 'triton'"
        ):
            codegen(op, "triton")(_stub_common)


class TestCodegenRepairConcurrency(unittest.TestCase):
    def test_exact_lookup_waits_for_active_repair(self) -> None:
        @api()
        def op() -> None:
            return None

        def original(state: object) -> str:
            return "triton"

        def replacement(state: object) -> str:
            return "replacement"

        codegen(op, "triton")(original)
        codegen_impls = cast("APIFunc", op)._codegen

        registration_published = threading.Event()
        release_repair = threading.Event()
        second_waiting = threading.Event()

        class TrackingRLock:
            def __init__(self) -> None:
                self.lock = threading.RLock()

            def __enter__(self) -> None:
                if registration_published.is_set():
                    second_waiting.set()
                self.lock.acquire()

            def __exit__(self, *args: object) -> None:
                self.lock.release()

        def reload_backend(codegen_name: str) -> None:
            self.assertEqual(codegen_name, "triton")
            codegen(op, codegen_name)(replacement)
            self.assertIs(dict.__getitem__(codegen_impls, codegen_name), replacement)
            registration_published.set()
            self.assertTrue(release_repair.wait(timeout=5))

        with (
            mock.patch.object(backend_registry, "_REPAIRED_CODEGEN_NAMES", frozenset()),
            mock.patch.object(
                backend_registry, "_CODEGEN_REPAIR_LOCK", TrackingRLock()
            ),
            mock.patch.object(
                backend_registry,
                "_reload_backend_codegen",
                side_effect=reload_backend,
            ) as reload_mock,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first_future = pool.submit(codegen_impls.__getitem__, "triton")
            self.assertTrue(registration_published.wait(timeout=5))
            second_future = pool.submit(codegen_impls.__getitem__, "triton")
            self.assertTrue(second_waiting.wait(timeout=5))
            try:
                self.assertFalse(second_future.done())
            finally:
                release_repair.set()
            self.assertIs(first_future.result(timeout=5), replacement)
            self.assertIs(second_future.result(timeout=5), replacement)

        reload_mock.assert_called_once_with("triton")


if __name__ == "__main__":
    unittest.main()
