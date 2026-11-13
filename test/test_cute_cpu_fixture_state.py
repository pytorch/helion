from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import inspect
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test import test_cute_epilogue_fanout as fanout

from helion._testing.cute_codegen import cpu_target

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("cached_arch", [None, "sm_90a"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_offline_target_preserves_sdk_architecture_without_probing(
    monkeypatch, cached_arch: str | None, body_fails: bool
) -> None:
    from cutlass._mlir import ir
    from cutlass.base_dsl.env_manager import EnvironmentVarManager
    from cutlass.cutlass_dsl import CuTeDSL

    dsl = CuTeDSL._get_dsl()
    environment = dsl.envar
    monkeypatch.setitem(vars(environment), "_arch", cached_arch)
    before = vars(environment).copy()
    descriptor = inspect.getattr_static(EnvironmentVarManager, "arch")
    expected = (
        pytest.raises(RuntimeError, match="context exit")
        if body_fails
        else nullcontext()
    )
    with (
        ir.Context(),
        ir.Location.unknown(),
        patch(
            "cutlass.base_dsl.env_manager.detect_gpu_arch",
            side_effect=AssertionError("device probe forbidden"),
        ) as probe,
    ):
        module = ir.Module.create()
        original_key = dsl.get_module_hash(module, "offline_target_probe")
        with expected, cpu_target(forbid_native=False):
            assert environment.arch == "sm_100a"
            assert vars(environment)["_arch"] == "sm_100a"
            target_key = dsl.get_module_hash(module, "offline_target_probe")
            assert target_key != original_key
            with cpu_target(forbid_native=False):
                assert environment.arch == "sm_100a"
                assert dsl.get_module_hash(module, "offline_target_probe") == target_key
            assert environment.arch == "sm_100a"
            if body_fails:
                raise RuntimeError("context exit")
        assert dsl.get_module_hash(module, "offline_target_probe") == original_key
        probe.assert_not_called()
    assert vars(environment) == before
    assert inspect.getattr_static(EnvironmentVarManager, "arch") is descriptor


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
def test_cpu_metadata_fixture_preserves_entry_state_and_threads(
    initialized: bool, body_fails: bool
) -> None:
    threads = torch.get_num_threads()
    fixture = contextmanager(cast("Any", fanout.cuda_trace).__wrapped__)
    with patch.object(torch.cuda, "is_initialized", return_value=initialized):
        try:
            with fixture():
                assert torch.get_num_threads() == 1
                assert torch.cuda.is_initialized() is initialized
                with pytest.raises(AssertionError, match="GPU forbidden"):
                    torch.cuda._lazy_init()
                if body_fails:
                    raise RuntimeError("held body failure")
        except RuntimeError as error:
            assert body_fails and str(error) == "held body failure"
        else:
            assert not body_fails
        assert torch.get_num_threads() == threads
        assert torch.cuda.is_initialized() is initialized


def test_cpu_metadata_fixture_rejects_a_new_initialization() -> None:
    threads = torch.get_num_threads()
    state = [False]
    fixture = contextmanager(cast("Any", fanout.cuda_trace).__wrapped__)
    with (
        patch.object(torch.cuda, "is_initialized", side_effect=lambda: state[0]),
        pytest.raises(AssertionError),
        fixture(),
    ):
        state[0] = True
    assert torch.get_num_threads() == threads
