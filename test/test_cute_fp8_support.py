from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._compiler.cute.mma_support import _probe_tcgen05_f8f6f4
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion.runtime import _torch_dtype_to_cutlass

cutlass = pytest.importorskip("cutlass")


def test_dtype_mapping_fp8() -> None:
    assert _torch_dtype_to_cutlass(torch.float8_e4m3fn) is cutlass.Float8E4M3FN
    assert _torch_dtype_to_cutlass(torch.float8_e5m2) is cutlass.Float8E5M2


def test_dtype_mapping_existing_unchanged() -> None:
    assert _torch_dtype_to_cutlass(torch.float16) is cutlass.Float16
    assert _torch_dtype_to_cutlass(torch.bfloat16) is cutlass.BFloat16
    assert _torch_dtype_to_cutlass(torch.float32) is cutlass.Float32


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="probe requires CUDA",
)
def test_tcgen05_f8f6f4_probe_on_b200() -> None:
    cap = torch.cuda.get_device_capability(0)
    ok, err = _probe_tcgen05_f8f6f4()
    if cap >= (10, 0):
        assert ok, f"expected fp8/f6/f4 MMA on capability {cap}, got error: {err}"
        assert err is None
    else:
        # Older arches may or may not support the atom depending on cutlass-dsl
        # backend; either way the probe must not raise.
        assert isinstance(ok, bool)
        if not ok:
            assert isinstance(err, str) and err


def test_capability_dataclass_no_cuda_fields() -> None:
    # When CUDA is unavailable, all probe fields are False and error fields populated.
    support = CuteMmaSupport(
        device_name=None,
        capability=None,
        cutlass_arch=None,
        universal=False,
        warp_f16bf16=False,
        warpgroup_f16bf16=False,
        tcgen05_f16bf16=False,
        tcgen05_f8f6f4=False,
        warp_error="x",
        warpgroup_error="x",
        tcgen05_error="x",
        tcgen05_f8f6f4_error="x",
    )
    assert support.tcgen05_f8f6f4 is False
    assert support.tcgen05_f8f6f4_error == "x"
    # supported_impls should not list tcgen05 if nothing under it is supported.
    assert "tcgen05" not in support.supported_impls


def test_supported_impls_lists_tcgen05_when_only_fp8_supported() -> None:
    # If only fp8 tcgen05 is available (not f16/bf16), tcgen05 should still appear.
    support = CuteMmaSupport(
        device_name="X",
        capability=(10, 0),
        cutlass_arch="SM_100",
        universal=True,
        warp_f16bf16=False,
        warpgroup_f16bf16=False,
        tcgen05_f16bf16=False,
        tcgen05_f8f6f4=True,
    )
    assert "tcgen05" in support.supported_impls


def test_probe_does_not_raise_when_cutlass_missing() -> None:
    # If cutlass.cute.nvgpu.tcgen05 cannot be imported, the probe must return
    # (False, error_str) rather than propagating.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cutlass.cute.nvgpu") or name == "cutlass":
            raise ImportError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        ok, err = _probe_tcgen05_f8f6f4()

    assert ok is False
    assert isinstance(err, str) and "ImportError" in err


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="aggregate probe requires CUDA",
)
def test_get_cute_mma_support_populates_fp8_field() -> None:
    support = get_cute_mma_support()
    cap = torch.cuda.get_device_capability(0)
    # Field is always present (no AttributeError) on supported builds.
    assert hasattr(support, "tcgen05_f8f6f4")
    assert hasattr(support, "tcgen05_f8f6f4_error")
    if cap >= (10, 0) and support.cutlass_arch is not None:
        assert support.tcgen05_f8f6f4 is True
