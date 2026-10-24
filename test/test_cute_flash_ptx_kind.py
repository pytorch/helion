"""The flash PTX emitters' ``tcgen05.mma`` kind follows the operand dtype (CPU).

``_flash_gemm_ptx`` used to spell ``kind::f16`` at every MMA issue; the kind
is now ``kind::f16`` / ``kind::tf32`` / ``kind::f8f6f4`` from the operand
type (``mma_kind_for_dtype``), the TMEM-sourced A operand's element width is
a parameter, and the fp8 CUTLASS type names resolve. fp16 / bf16 renders are
unchanged: the callers pass no kind and the defaults spell ``kind::f16``.
"""

from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass

from helion._compiler.cute import _flash_gemm_ptx as flash_ptx
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def test_mma_kind_for_dtype() -> None:
    assert flash_ptx.mma_kind_for_dtype(cutlass.Float16) == "f16"
    assert flash_ptx.mma_kind_for_dtype(cutlass.BFloat16) == "f16"
    assert flash_ptx.mma_kind_for_dtype(cutlass.TFloat32) == "tf32"
    assert flash_ptx.mma_kind_for_dtype(cutlass.Float8E4M3FN) == "f8f6f4"
    assert flash_ptx.mma_kind_for_dtype(cutlass.Float8E5M2) == "f8f6f4"
    with pytest.raises(TypeError):
        flash_ptx.mma_kind_for_dtype(cutlass.Int8)
    assert flash_ptx._mma_kind_qualifier("f16") == "kind::f16"
    assert flash_ptx._mma_kind_qualifier("f8f6f4") == "kind::f8f6f4"
    with pytest.raises(ValueError, match="kind"):
        flash_ptx._mma_kind_qualifier("f8")


def test_fp8_instruction_descriptor() -> None:
    """The fp8 operand formats resolve (the CUTLASS type names are the
    ``Float8*`` ones) and encode in the instruction descriptor's A/B fields."""
    assert flash_ptx.to_UMMA_format(cutlass.Float8E4M3FN) == (
        flash_ptx.MXF8F6F4Format.E4M3
    )
    assert flash_ptx.to_UMMA_format(cutlass.Float8E5M2) == (
        flash_ptx.MXF8F6F4Format.E5M2
    )
    k_major = flash_ptx.Major.K
    e4m3 = flash_ptx.make_instr_desc(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float32,
        128,
        128,
        k_major,
        k_major,
    )
    e5m2 = flash_ptx.make_instr_desc(
        cutlass.Float8E5M2,
        cutlass.Float8E5M2,
        cutlass.Float32,
        128,
        128,
        k_major,
        k_major,
    )
    # a_fmt at bits 7..9, b_fmt at bits 10..12.
    assert e5m2 - e4m3 == (1 << 7) | (1 << 10)
    f16 = flash_ptx.make_instr_desc(
        cutlass.Float16, cutlass.Float16, cutlass.Float32, 128, 128, k_major, k_major
    )
    # E4M3 and F16 share format code 0: the ``kind::`` qualifier tells them
    # apart, which is why every issue site reads it from the operand dtype.
    assert e4m3 == f16


def test_every_mma_issue_reads_the_kind_qualifier() -> None:
    source = inspect.getsource(flash_ptx)
    issues = re.findall(r"tcgen05\.mma\.\{cta_group_qualifier\}\.(\S+) ", source)
    # Every issue site spells the kind through the qualifier, and no literal
    # ``kind::f16`` remains outside the qualifier table.
    assert issues
    assert set(issues) == {"{mma_kind_qualifier}"}
    assert "kind::f16" not in source.replace('"kind::f16"', "")
    assert set(flash_ptx.MMA_KINDS) == {"f16", "tf32", "f8f6f4"}
