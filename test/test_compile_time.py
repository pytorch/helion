from __future__ import annotations

import time

import pytest

import helion._compile_time as compile_time

pytestmark = pytest.mark.cpu_only


def test_enable_after_import_activates_measurement(monkeypatch) -> None:
    compile_time.reset()
    monkeypatch.setattr(compile_time, "_enabled", False)

    with compile_time.measure("Kernel.bind"):
        time.sleep(0.001)
    assert compile_time.get_total_time() == 0.0

    try:
        compile_time.enable()
        with compile_time.measure("Kernel.bind"):
            time.sleep(0.001)
        assert compile_time.get_total_time() > 0.0
    finally:
        compile_time.reset()
