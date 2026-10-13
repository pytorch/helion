from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from helion import _compat


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_cpu_codegen_feature_caches_follow_context(warm, nested, raises):
    initialized = torch.cuda.is_initialized()
    probes = (_compat._supports_maxnreg, _compat._supports_tensor_descriptor)
    for probe in probes:
        probe.cache_clear()
    try:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
            ),
        ):
            if warm:
                assert all(probe() for probe in probes)

            def exercise():
                with _cpu_codegen():
                    assert not any(probe() for probe in probes)
                    with pytest.raises(AssertionError, match="CUDA forbidden"):
                        torch.cuda._lazy_init()
                    if nested:
                        with _cpu_codegen():
                            assert not any(probe() for probe in probes)
                        assert not any(probe() for probe in probes)
                    if raises:
                        raise RuntimeError("body failure")

            if raises:
                with pytest.raises(RuntimeError, match="body failure"):
                    exercise()
            else:
                exercise()
            assert all(probe() for probe in probes)
            assert (
                _compat._supports_maxnreg,
                _compat._supports_tensor_descriptor,
            ) == probes
    finally:
        for probe in probes:
            probe.cache_clear()
    assert torch.cuda.is_initialized() == initialized
