from __future__ import annotations

import unittest

from helion.language._decorators import CodegenDict


def _stub_common(state: object) -> str:
    return "common"


def _stub_triton(state: object) -> str:
    return "triton"


def _stub_pallas(state: object) -> str:
    return "pallas"


class TestCodegenDict(unittest.TestCase):
    def test_getitem_exact_match(self) -> None:
        d = CodegenDict({"triton": _stub_triton, "common": _stub_common})
        self.assertIs(d["triton"], _stub_triton)
        self.assertIs(d["common"], _stub_common)

    def test_getitem_falls_back_to_common(self) -> None:
        d = CodegenDict({"common": _stub_common})
        self.assertIs(d["pallas"], _stub_common)

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


if __name__ == "__main__":
    unittest.main()
