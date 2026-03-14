"""
Tests for FileAutotuneCache (issue #164 — caching layer for autotuned configs).

Coverage:
  * cache hit returns stored config
  * cache miss (no file) returns None
  * cache miss on key change (different shapes -> different key)
  * version mismatch treated as cache miss
  * custom key template overrides default
  * atomic write: tmp file is renamed, not left behind
  * corrupt/empty file handled gracefully
  * version_stamp changes when Helion/Triton/CUDA version stub changes
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner.file_cache import FileAutotuneCache
from helion.autotuner.file_cache import _FORMAT_VERSION
from helion.autotuner.file_cache import build_version_stamp
import helion.language as hl


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@onlyBackends(["triton"])
class TestFileAutotuneCache(TestCase):
    """Unit tests for FileAutotuneCache."""

    def _make_autotuner(
        self,
        env: dict[str, str],
        *,
        shapes: tuple[int, ...] = (8,),
    ) -> tuple[FileAutotuneCache, helion.Config]:
        """Create a FileAutotuneCache and return (cache, default_config).

        The kernel is defined and bound *inside* the env patch so that
        ``Settings()`` reads the correct ``HELION_AUTOTUNE_CACHE*`` values.
        """
        args = (
            torch.randn(list(shapes), device=DEVICE),
            torch.randn(list(shapes), device=DEVICE),
        )

        with patch.dict(os.environ, env, clear=False):
            @helion.kernel(autotune_baseline_fn=operator.add, autotune_log_level=0)
            def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add.bind(args)
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                cache = bound.settings.autotuner_fn(bound, args)

        assert isinstance(cache, FileAutotuneCache), (
            f"Expected FileAutotuneCache, got {type(cache).__name__}"
        )
        return cache, bound.config_spec.default_config()

    # ------------------------------------------------------------------
    # Basic roundtrip
    # ------------------------------------------------------------------

    def test_cache_miss_when_file_absent(self) -> None:
        """get() returns None when the cache file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "missing.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, _ = self._make_autotuner(env)
            self.assertIsNone(cache.get())

    def test_cache_hit_after_put(self) -> None:
        """put() writes a config; get() returns it on the next call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, config = self._make_autotuner(env)
            cache.put(config)
            self.assertEqual(cache.get(), config)

    def test_file_is_valid_json(self) -> None:
        """Cache file is valid JSON and has the expected top-level structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, config = self._make_autotuner(env)
            cache.put(config)
            data = json.loads(cache_path.read_text())
            self.assertEqual(data["format_version"], _FORMAT_VERSION)
            self.assertIn("entries", data)
            self.assertIn("helion_version", data)
            self.assertIn("triton_version", data)
            self.assertIn("cuda_version", data)

    # ------------------------------------------------------------------
    # Key differentiation
    # ------------------------------------------------------------------

    def test_cache_miss_on_different_shape(self) -> None:
        """A config stored for shape (8,) is not returned for shape (16,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache_small, config = self._make_autotuner(env, shapes=(8,))
            cache_small.put(config)

            cache_large, _ = self._make_autotuner(env, shapes=(16,))
            # The default key includes specialization_key_hash which encodes shape
            self.assertIsNone(cache_large.get())

    def test_custom_key_template(self) -> None:
        """A custom key template is used when autotune_cache_key is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            # Omit specialization_key_hash so shape (8,) and (16,) share a key
            custom_key = "{kernel_name}-{hardware}-{backend}"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
                "HELION_AUTOTUNE_CACHE_KEY": custom_key,
            }
            cache_small, config = self._make_autotuner(env, shapes=(8,))
            cache_small.put(config)

            cache_large, _ = self._make_autotuner(env, shapes=(16,))
            # Same kernel + hardware + backend -> same key -> cache hit
            self.assertEqual(cache_large.get(), config)

    def test_invalid_key_template_raises(self) -> None:
        """An unknown field in the template raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
                "HELION_AUTOTUNE_CACHE_KEY": "{kernel_name}-{nonexistent_field}",
            }
            with self.assertRaisesRegex(ValueError, "nonexistent_field"):
                self._make_autotuner(env)

    # ------------------------------------------------------------------
    # Version stamp / version mismatch
    # ------------------------------------------------------------------

    def test_version_stamp_is_stable(self) -> None:
        """build_version_stamp() is deterministic within a single run."""
        stamp1 = build_version_stamp()
        stamp2 = build_version_stamp()
        self.assertEqual(stamp1, stamp2)

    def test_version_mismatch_causes_cache_miss(self) -> None:
        """
        If the version_stamp changes between put() and get() calls, the old
        entry is no longer visible (different cache key, so get() returns None).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, config = self._make_autotuner(env)
            cache.put(config)
            # Simulate a version change by patching the stamp used during key build
            with patch(
                "helion.autotuner.file_cache.build_version_stamp",
                return_value="00000000000000xx",
            ):
                cache2, _ = self._make_autotuner(env)
            self.assertIsNone(cache2.get())

    def test_format_version_mismatch_returns_empty(self) -> None:
        """
        If the on-disk format_version doesn't match _FORMAT_VERSION,
        the file is treated as empty (cache miss, no exception).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            # Write a file with the wrong format version
            cache_path.write_text(
                json.dumps(
                    {
                        "format_version": _FORMAT_VERSION + 99,
                        "entries": {"some-key": {"config": {}, "key_fields": {}}},
                    }
                )
            )
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, _ = self._make_autotuner(env)
            self.assertIsNone(cache.get())

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def test_corrupt_file_handled_gracefully(self) -> None:
        """A corrupted JSON file causes a warning-level log but not an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text("not valid json {{{")
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, _ = self._make_autotuner(env)
            self.assertIsNone(cache.get())  # No exception

    def test_empty_file_handled_gracefully(self) -> None:
        """An empty cache file does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            cache_path.write_text("")
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, _ = self._make_autotuner(env)
            self.assertIsNone(cache.get())

    def test_atomic_write_no_tmp_leftover(self) -> None:
        """After put(), no .tmp file should remain next to the cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, config = self._make_autotuner(env)
            cache.put(config)
            tmp = cache_path.with_suffix(".tmp")
            self.assertFalse(tmp.exists(), f"Leftover tmp file: {tmp}")

    def test_multiple_kernels_share_file(self) -> None:
        """Two different caches (simulating two kernels) coexist in one file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "shared.json"
            env_a = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
                "HELION_AUTOTUNE_CACHE_KEY": "{kernel_name}-{specialization_key_hash}-A",
            }
            env_b = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
                "HELION_AUTOTUNE_CACHE_KEY": "{kernel_name}-{specialization_key_hash}-B",
            }
            cache_a, config = self._make_autotuner(env_a, shapes=(8,))
            cache_b, _ = self._make_autotuner(env_b, shapes=(8,))

            cache_a.put(config)
            cache_b.put(config)

            data = json.loads(cache_path.read_text())
            # Both entries should be present
            self.assertEqual(len(data["entries"]), 2)
            # Each cache can still read its own entry
            self.assertEqual(cache_a.get(), config)
            self.assertEqual(cache_b.get(), config)

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def test_get_cache_info_message_contains_path(self) -> None:
        """_get_cache_info_message() should mention the cache file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "info.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, _ = self._make_autotuner(env)
            msg = cache._get_cache_info_message()
            self.assertIn(str(cache_path), msg)

    def test_list_cache_entries_after_put(self) -> None:
        """_list_cache_entries() returns one entry per put()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "list.json"
            env = {
                "HELION_AUTOTUNE_CACHE": "FileAutotuneCache",
                "HELION_AUTOTUNE_CACHE_PATH": str(cache_path),
            }
            cache, config = self._make_autotuner(env)
            self.assertEqual(len(cache._list_cache_entries()), 0)
            cache.put(config)
            self.assertEqual(len(cache._list_cache_entries()), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
