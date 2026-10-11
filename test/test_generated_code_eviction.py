from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
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
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        # Tests pin the budget; the parsed one is restored for the next test.
        self.parse_limit = cache._cache_limit
        self._test_stack.enter_context(self._limit(2048))
        self.root = Path(root) / "generated_code"

    def _limit(self, limit: int):
        return patch.object(cache, "_cache_limit", lambda: limit)

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
        with self._limit(2 * size):
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
        with self._limit(2 * size):
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
        with self._limit(0):
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
            patch.object(
                cache, "FileLock", lambda path, timeout: FileLock(path, timeout=0.1)
            ),
            self.assertLogs(cache.log, level="DEBUG") as logs,
        ):
            cache.save_generated_code("locked", source)
        self.assertIsNone(cache.load_generated_code("locked"))
        # A lock held by another writer is routine, not a warning.
        self.assertEqual([record.levelname for record in logs.records], ["DEBUG"])
        self.assertIn("lock timeout", logs.output[0])
        self.assertEqual(
            {path.name for path in self.root.iterdir()} - {".lock", ".size"},
            {"existing.json"},
        )

    def test_identical_entry_is_not_rewritten(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("entry", source)
        entry = self.root / "entry.json"
        os.utime(entry, ns=(1, 1))
        with (
            patch.object(cache.os, "replace", side_effect=AssertionError("rewrite")),
            patch.object(cache, "FileLock", side_effect=AssertionError("locked")),
        ):
            cache.save_generated_code("entry", source)
        self.assertGreater(entry.stat().st_mtime_ns, 1)
        self.assertEqual(cache.load_generated_code("entry"), source)

    def test_budget_is_parsed_once_and_malformed_values_warn_once(self) -> None:
        parse_limit = self.parse_limit
        self.addCleanup(parse_limit.cache_clear)
        parse_limit.cache_clear()
        with patch.dict(
            "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": "4096"}
        ):
            self.assertEqual(parse_limit(), 4096)
        for value in ("-1", "many"):
            parse_limit.cache_clear()
            with (
                patch.dict(
                    "os.environ", {"HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES": value}
                ),
                self.assertLogs(cache.log, level="WARNING") as logs,
            ):
                for _ in range(2):
                    self.assertEqual(parse_limit(), cache._DEFAULT_CACHE_LIMIT)
            self.assertEqual(len(logs.output), 1)

    def _json_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*.json"))

    def _indexed_bytes(self) -> int:
        return int((self.root / ".size").read_text().split()[0])

    def test_writes_scan_only_when_budget_may_be_exceeded(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("first", source)
        size = (self.root / "first.json").stat().st_size
        with (
            self._limit(3 * size),
            patch.object(
                cache, "_evict_entries", side_effect=cache._evict_entries
            ) as evict,
        ):
            cache.save_generated_code("second", source)
            cache.save_generated_code("second", source)
            cache.save_generated_code("third", source)
            self.assertEqual(evict.call_count, 0)
            self.assertEqual(self._indexed_bytes(), 3 * size)
            cache.save_generated_code("fourth", source)
            self.assertEqual(evict.call_count, 1)
        self.assertEqual(self._indexed_bytes(), self._json_bytes())
        self.assertLessEqual(self._json_bytes(), 3 * size)

    def test_stale_size_index_self_heals(self) -> None:
        source = "def cached():\n    return 1\n"
        for name in ("first", "second", "third"):
            cache.save_generated_code(name, source)
        size = (self.root / "first.json").stat().st_size
        index = self.root / ".size"
        with self._limit(3 * size):
            # A missing or corrupt index is rebuilt from the directory.
            for number, contents in enumerate((None, "corrupt", "-5 0")):
                if contents is None:
                    index.unlink()
                else:
                    index.write_text(contents)
                cache.save_generated_code(f"rebuilt{number}", source)
                self.assertEqual(self._indexed_bytes(), 3 * size)
                self.assertEqual(self._json_bytes(), 3 * size)
            # Manual removal leaves an overestimate, corrected by the next scan.
            next(self.root.glob("*.json")).unlink()
            cache.save_generated_code("fourth", source)
            self.assertEqual(self._indexed_bytes(), 3 * size)
            self.assertEqual(cache.load_generated_code("fourth"), source)
            # A failed publication also leaves an overestimate.
            with (
                self.assertLogs(cache.log, level="WARNING"),
                patch.object(cache.os, "replace", side_effect=OSError("read-only")),
            ):
                cache.save_generated_code("failed", source)
            self.assertGreater(self._indexed_bytes(), self._json_bytes())
            cache.save_generated_code("fifth", source)
            self.assertEqual(self._indexed_bytes(), self._json_bytes())
            self.assertLessEqual(self._json_bytes(), 3 * size)
            # Files added outside the writers are found by the periodic rescan.
            index.write_text("0 1")
            with patch.object(cache, "_SIZE_INDEX_RESCAN_WRITES", 2):
                cache.save_generated_code("sixth", source)
                self.assertGreater(self._json_bytes(), 3 * size)
                cache.save_generated_code("seventh", source)
            self.assertEqual(self._indexed_bytes(), self._json_bytes())
            self.assertLessEqual(self._json_bytes(), 3 * size)

    def test_concurrent_writers_keep_index_and_budget(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("probe", source)
        size = (self.root / "probe.json").stat().st_size
        with (
            self._limit(5 * size),
            patch.object(cache, "FileLock", lambda path, timeout: FileLock(path)),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            list(
                pool.map(
                    lambda index: cache.save_generated_code(f"entry{index}", source),
                    range(32),
                )
            )
        self.assertEqual(self._indexed_bytes(), self._json_bytes())
        self.assertLessEqual(self._json_bytes(), 5 * size)

    def test_hits_make_eviction_least_recently_used(self) -> None:
        config = helion.Config(block_sizes=[32])
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("a" * 64, source)
        cache.save_compiled_kernel("binding", config, config, "a" * 64)
        cache.save_generated_code("unused", source)
        binding = self.root / "bindings" / "binding.json"
        for path, time in (
            (self.root / f"{'a' * 64}.json", 1),
            (binding, 2),
            (self.root / "unused.json", 3),
        ):
            os.utime(path, ns=(time, time))
        # The binding hit also refreshes the source it references.
        self.assertEqual(
            cache.load_compiled_kernel("binding"), (config, config, source)
        )
        with self._limit(self._json_bytes()):
            cache.save_generated_code("new", source)
        self.assertIsNone(cache.load_generated_code("unused"))
        self.assertEqual(
            cache.load_compiled_kernel("binding"), (config, config, source)
        )
        self.assertEqual(cache.load_generated_code("new"), source)

    def test_hit_on_read_only_entry_is_still_a_hit(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("entry", source)
        with patch.object(cache.os, "utime", side_effect=PermissionError("read-only")):
            self.assertEqual(cache.load_generated_code("entry"), source)

    def test_settings_key_uses_explicit_denylist(self) -> None:
        settings = helion.Settings()
        key = cache._compilation_settings_key(settings)
        self.assertIsNot(key, cache._UNCACHEABLE)
        for name, value in (
            ("autotune_random_seed", settings.autotune_random_seed + 1),
            ("autotune_effort", "none"),
            ("generated_code_cache", not settings.generated_code_cache),
        ):
            self.assertEqual(
                cache._compilation_settings_key(settings.copy(**{name: value})), key
            )
        self.assertNotEqual(
            cache._compilation_settings_key(
                settings.copy(static_shapes=not settings.static_shapes)
            ),
            key,
        )

        @dataclasses.dataclass
        class HiddenSettings:
            static_shapes: bool = True
            hidden: int = dataclasses.field(default=0, repr=False)

        # A field hidden from repr can still affect codegen.
        self.assertNotEqual(
            cache._compilation_settings_key(HiddenSettings()),
            cache._compilation_settings_key(HiddenSettings(hidden=1)),
        )


if __name__ == "__main__":
    from helion._testing import main

    main()
