from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion import exc
from helion.runtime.cute import launcher


def _plan(source: torch.Tensor, output: torch.Tensor) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": "tcgen05_aux_direct",
        "c_idx": 0,
        "d_idx": 1,
    }
    for prefix, value in (("c", source), ("d", output)):
        result[f"{prefix}_shape"] = tuple(value.shape)
        result[f"{prefix}_stride"] = tuple(value.stride())
        result[f"{prefix}_dtype"] = str(value.dtype)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_direct_aux_reads_allow_positive_stride_and_natural_alignment(
    dtype: torch.dtype, stride: int
) -> None:
    source = torch.empty(129 * stride, dtype=dtype)[1::stride][:128]
    output = torch.empty((3, 128, 256), dtype=torch.bfloat16)
    plan = _plan(source, output)
    launcher._validate_direct_aux_arguments(plan, (source, output))
    # New allocations and tensor identities do not invalidate proven layout.
    new_source = torch.empty_like(source, memory_format=torch.preserve_format)
    if stride == 1:
        launcher._validate_direct_aux_arguments(plan, (new_source, output.clone()))


def test_direct_aux_reads_allow_disjoint_shared_storage() -> None:
    backing = torch.empty(512, dtype=torch.bfloat16)
    source = backing[:128]
    output = backing[128:].view(3, 128)
    launcher._validate_direct_aux_arguments(_plan(source, output), (source, output))


def test_direct_aux_reads_reject_overlap_with_different_dtype() -> None:
    backing = torch.empty(1024, dtype=torch.uint8)
    source = backing[:512].view(torch.float32)
    output = backing[256:].view(torch.bfloat16).view(3, 128)
    with pytest.raises(exc.BackendUnsupported, match="overlap"):
        launcher._validate_direct_aux_arguments(_plan(source, output), (source, output))


@pytest.mark.parametrize("changed", ["shape", "stride", "dtype", "negative_view"])
def test_direct_aux_runtime_layout_must_match_proof(changed: str) -> None:
    source = torch.empty(128, dtype=torch.float32)
    output = torch.empty((3, 128, 256), dtype=torch.bfloat16)
    plan = _plan(source, output)
    if changed == "shape":
        source = source[:64]
    elif changed == "stride":
        source = torch.empty(256, dtype=source.dtype)[::2]
    elif changed == "dtype":
        source = source.to(torch.bfloat16)
    else:
        source = torch._neg_view(source)
    with pytest.raises(exc.BackendUnsupported, match="proven shape"):
        launcher._validate_direct_aux_arguments(plan, (source, output))


def test_direct_aux_guard_precedes_fast_and_last_launch_caches() -> None:
    backing = torch.empty(1024, dtype=torch.bfloat16)
    source = backing[:128]
    output = backing[64:448].view(3, 128)
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[_plan(source, output)])
    with (
        patch.object(
            launcher._CuteFastRelaunch,
            "try_launch",
            side_effect=AssertionError("cache"),
        ),
        patch.object(
            launcher,
            "_cute_last_launch_cache_entry",
            side_effect=AssertionError("cache"),
        ),
        pytest.raises(exc.BackendUnsupported, match="overlap"),
    ):
        launcher.default_cute_launcher(kernel, (1,), source, output, block=(256,))


def test_direct_aux_second_call_rechecks_new_alias() -> None:
    source = torch.empty(128, dtype=torch.bfloat16)
    output = torch.empty((3, 128), dtype=torch.bfloat16)
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[_plan(source, output)])
    launcher._validate_batched_aux_tma_arguments(kernel, (source, output))
    with pytest.raises(exc.BackendUnsupported, match="overlap"):
        launcher._validate_batched_aux_tma_arguments(kernel, (output[0], output))
