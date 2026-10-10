from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from filelock import FileLock

import helion
from helion._testing import TestCase
from helion.runtime import generated_code_cache as cache


class TestGeneratedCodeEviction(TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "HELION_CACHE_DIR": root,
                    "HELION_SKIP_CACHE": "0",
                    "HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": "2048",
                },
            )
        )
        self.root = Path(root) / "generated_code"

    def test_sources_and_bindings_share_the_byte_budget(self) -> None:
        config = helion.Config(block_sizes=[32])
        source = "def cached():\n    return 1\n"
        source_key = "a" * 64
        cache.save_generated_code(source_key, source)
        cache.save_compiled_kernel("binding", config, config, source_key)
        self.assertEqual(cache.load_generated_code(source_key), source)
        self.assertEqual(
            cache.load_compiled_kernel("binding"), (config, config, source)
        )
        self.assertLessEqual(
            sum(path.stat().st_size for path in self.root.rglob("*.json")), 2048
        )
        binding = self.root / "bindings" / "binding.json"
        os.utime(binding, ns=(1, 1))
        size = (self.root / f"{source_key}.json").stat().st_size
        with patch.dict(
            "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": str(2 * size)}
        ):
            cache.save_generated_code("second", source)
        self.assertFalse(binding.exists())
        self.assertIsNone(cache.load_compiled_kernel("binding"))
        self.assertEqual(cache.load_generated_code(source_key), source)
        self.assertLessEqual(
            sum(path.stat().st_size for path in self.root.rglob("*.json")), 2 * size
        )

    def test_capacity_evicts_oldest_entries_and_skips_oversized_writes(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("old", source)
        old = self.root / "old.json"
        size = old.stat().st_size
        os.utime(old, ns=(1, 1))
        with patch.dict(
            "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": str(2 * size)}
        ):
            cache.save_generated_code("second", source)
            cache.save_generated_code("third", source)
            self.assertIsNone(cache.load_generated_code("old"))
            self.assertEqual(cache.load_generated_code("second"), source)
            self.assertEqual(cache.load_generated_code("third"), source)
            cache.save_generated_code("oversized", source * (2 * size))
            self.assertIsNone(cache.load_generated_code("oversized"))
            self.assertEqual(cache.load_generated_code("third"), source)
            self.assertLessEqual(
                sum(path.stat().st_size for path in self.root.rglob("*.json")),
                2 * size,
            )
        with patch.dict(
            "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": "0"}
        ):
            cache.save_generated_code("disabled", source)
            self.assertIsNone(cache.load_generated_code("disabled"))

    def test_write_failures_preserve_complete_existing_entries(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("existing", source)
        with (
            self.assertLogs(cache.log, level="WARNING"),
            patch.object(cache.os, "replace", side_effect=OSError("read-only cache")),
        ):
            cache.save_generated_code("existing", "replacement")
        self.assertEqual(cache.load_generated_code("existing"), source)
        with (
            FileLock(self.root / ".lock"),
            self.assertLogs(cache.log, level="WARNING"),
        ):
            cache.save_generated_code("locked", source)
        self.assertIsNone(cache.load_generated_code("locked"))
        with (
            patch.dict(
                "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": "-1"}
            ),
            self.assertLogs(cache.log, level="WARNING"),
        ):
            cache.save_generated_code("invalid", source)
        self.assertIsNone(cache.load_generated_code("invalid"))
        self.assertEqual(
            {path.name for path in self.root.iterdir()} - {".lock"}, {"existing.json"}
        )


if __name__ == "__main__":
    from helion._testing import main

    main()
