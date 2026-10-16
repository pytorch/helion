from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from helion._compat import _flush_cache_state
from helion._compat import active_cache_clearer
from helion._compat import make_cache_clearer
from helion._compat import safe_clear_cache


class _FakeDriver:
    """Test double for a triton driver with the L2-flush primitives."""

    def __init__(self) -> None:
        self.device: object = "gpu0"
        self.allocated: list[object] = []
        self.flushed: list[object] = []

    def get_current_device(self) -> object:
        return self.device

    def get_empty_cache_for_benchmark(self) -> object:
        buffer = object()
        self.allocated.append(buffer)
        return buffer

    def clear_cache(self, cache: object) -> None:
        self.flushed.append(cache)


class TestMakeCacheClearer(unittest.TestCase):
    def setUp(self) -> None:
        # Module-level (driver, device, buffer) slot shared with
        # safe_clear_cache(); isolate each test from prior allocations and
        # clear the fake driver/buffer again when the tests finish.
        _flush_cache_state.clear()
        self.addCleanup(_flush_cache_state.clear)

    def test_driver_without_primitives_yields_none(self) -> None:
        self.assertIsNone(make_cache_clearer(SimpleNamespace()))
        self.assertIsNone(
            make_cache_clearer(SimpleNamespace(clear_cache=lambda cache: None))
        )

    def test_clearer_allocates_eagerly_and_reuses_one_buffer(self) -> None:
        driver = _FakeDriver()
        clearer = make_cache_clearer(driver)
        self.assertIsNotNone(clearer)
        # The buffer is allocated at binding time, before any timed loop.
        self.assertEqual(len(driver.allocated), 1)
        assert clearer is not None
        clearer()
        clearer()
        self.assertEqual(driver.flushed.count(driver.allocated[0]), 2)
        self.assertEqual(len(driver.allocated), 1)


class TestSafeClearCache(unittest.TestCase):
    def setUp(self) -> None:
        _flush_cache_state.clear()
        self.addCleanup(_flush_cache_state.clear)

    def test_buffer_is_reallocated_on_device_change(self) -> None:
        driver = _FakeDriver()
        clearer = make_cache_clearer(driver)
        assert clearer is not None
        first = driver.allocated[0]
        driver.device = "gpu1"
        clearer()
        self.assertEqual(len(driver.allocated), 2)
        self.assertIsNot(driver.allocated[1], first)
        self.assertIs(driver.flushed[-1], driver.allocated[1])

    def test_noop_when_driver_resolution_raises(self) -> None:
        # triton's driver.active is a LazyProxy: resolution happens on the
        # first attribute access, so the capability probes can raise
        # RuntimeError (zero or several active drivers). hasattr only
        # suppresses AttributeError, so that must be caught at the gate.
        class _UnresolvableDriver:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError("no active driver")

        runtime = ModuleType("triton.runtime")
        runtime.driver = SimpleNamespace(  # type: ignore[attr-defined]
            active=_UnresolvableDriver()
        )
        triton_module = ModuleType("triton")
        triton_module.runtime = runtime  # type: ignore[attr-defined]
        with patch.dict(
            sys.modules, {"triton": triton_module, "triton.runtime": runtime}
        ):
            self.assertIsNone(active_cache_clearer())
            safe_clear_cache()  # must not raise

    def test_noop_when_driver_lacks_primitives(self) -> None:
        # A resolvable driver without the flush primitives (common for
        # out-of-tree Triton drivers on non-CUDA accelerators) must be a
        # silent no-op.
        runtime = ModuleType("triton.runtime")
        runtime.driver = SimpleNamespace(  # type: ignore[attr-defined]
            active=SimpleNamespace(get_empty_cache_for_benchmark=lambda: object())
        )
        triton_module = ModuleType("triton")
        triton_module.runtime = runtime  # type: ignore[attr-defined]
        with patch.dict(
            sys.modules, {"triton": triton_module, "triton.runtime": runtime}
        ):
            safe_clear_cache()  # must not raise

    def test_flushes_through_the_shared_buffer(self) -> None:
        driver = _FakeDriver()
        runtime = ModuleType("triton.runtime")
        runtime.driver = SimpleNamespace(active=driver)  # type: ignore[attr-defined]
        triton_module = ModuleType("triton")
        triton_module.runtime = runtime  # type: ignore[attr-defined]
        with patch.dict(
            sys.modules, {"triton": triton_module, "triton.runtime": runtime}
        ):
            safe_clear_cache()
            safe_clear_cache()
        # One shared buffer for both flushes (and shared with any
        # _make_l2_cache_clearer() user on the same driver/device).
        self.assertEqual(len(driver.allocated), 1)
        self.assertEqual(driver.flushed, [driver.allocated[0]] * 2)


if __name__ == "__main__":
    unittest.main()
