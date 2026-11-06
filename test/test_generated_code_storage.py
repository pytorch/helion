from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from helion._testing import TestCase
from helion.runtime import generated_code_cache as cache


class TestGeneratedCodeStorage(TestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(directory) / "generated_code"
        self._test_stack.enter_context(
            patch.dict(
                os.environ,
                {"HELION_CACHE_DIR": directory, "HELION_SKIP_CACHE": "0"},
            )
        )

    @unittest.skipIf(os.name == "nt", "POSIX file permissions")
    def test_entry_permissions_follow_umask(self) -> None:
        # Set the process-wide umask in a subprocess, never in a pytest worker
        # that may be running other threads. Test all three entry directories.
        script = r"""
import json
import os
from pathlib import Path
import sys
from helion.runtime import generated_code_cache as cache
os.umask(int(sys.argv[1], 8))
root = Path(os.environ["HELION_CACHE_DIR"]) / "generated_code"
for subdir in ("", "bindings", "schemas"):
    cache._save_entry(root / subdir / f"entry{sys.argv[1]}.json", {"value": 1})
print(json.dumps([(path.stat().st_mode & 0o777) for path in root.rglob(f"entry{sys.argv[1]}.json")]))
"""
        for mask, expected in (("022", 0o644), ("077", 0o600)):
            with self.subTest(umask=mask):
                process = subprocess.run(
                    [sys.executable, "-c", script, mask],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                self.assertEqual(json.loads(process.stdout), [expected] * 3)

    @unittest.skipUnless(
        os.name == "posix" and os.geteuid() == 0,
        "requires root to switch to a second UID",
    )
    def test_root_warmed_cache_is_readable_by_another_uid(self) -> None:
        # A container image may warm as root and serve as an unprivileged UID.
        # Keep this independent of the checkout's directory permissions: import
        # first, then drop privileges and read the public cache APIs.
        script = r"""
import os
from pathlib import Path
from unittest.mock import patch
import helion
from helion.runtime import generated_code_cache as cache
os.umask(0o022)
source = "def cached():\n    return 1\n"
config = helion.Config(block_sizes=[32])
schema = {"guards": [], "compiler_seed_facts": []}
cache.save_generated_code("a" * 64, source)
cache.save_compiled_kernel("binding", config, config, "a" * 64)
cache.save_binding_schema("schema", schema)
root = Path(os.environ["HELION_CACHE_DIR"]) / "generated_code"
for path in (root, root / "bindings", root / "schemas"):
    path.chmod(0o555)
os.setgroups([])
os.setgid(65534)
os.setuid(65534)
with patch.object(cache, "FileLock", side_effect=AssertionError("read took lock")):
    assert cache.load_generated_code("a" * 64) == source
    assert cache.load_compiled_kernel("binding") == (config, config, source)
    assert cache.load_binding_schema("schema") == schema
"""
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            Path(directory).chmod(0o755)
            env = dict(os.environ, HELION_CACHE_DIR=directory)
            subprocess.run(
                [sys.executable, "-c", script], env=env, check=True, timeout=30
            )

    def test_interrupted_write_is_reclaimed_by_rescan(self) -> None:
        source = "def cached():\n    return 1\n"
        cache.save_generated_code("existing", source)
        # os._exit models SIGKILL after the complete temp write: Python's
        # finally cleanup does not run. The reservation must count these bytes.
        script = r"""
import os
from unittest.mock import patch
from helion.runtime import generated_code_cache as cache
with patch.object(cache.os, "replace", side_effect=lambda *args: os._exit(0)):
    cache.save_generated_code("interrupted", "unfinished publication")
raise AssertionError("publication hook was not reached")
"""
        subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
        temporary = list(self.root.rglob("*.json.tmp"))
        self.assertEqual(len(temporary), 1)
        self.assertFalse((self.root / "interrupted.json").exists())
        indexed = int((self.root / ".size").read_text().split()[0])
        self.assertEqual(
            indexed,
            sum(path.stat().st_size for path in self.root.rglob("*.json*")),
        )
        # Reclaim under the existing writers' lock at the next periodic scan.
        with patch.object(cache, "_SIZE_INDEX_RESCAN_WRITES", 0):
            cache.save_generated_code("next", source)
        self.assertFalse(temporary[0].exists())
        self.assertEqual(cache.load_generated_code("existing"), source)
        self.assertEqual(cache.load_generated_code("next"), source)
        self.assertEqual(
            int((self.root / ".size").read_text().split()[0]),
            sum(path.stat().st_size for path in self.root.rglob("*.json")),
        )

    def test_rescan_cleans_temporary_files_in_all_entry_directories(self) -> None:
        for subdir in ("", "bindings", "schemas"):
            path = self.root / subdir / ".abandoned.json.tmp"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"abandoned")
        unrelated = self.root / "user-file.tmp"
        unrelated.write_bytes(b"keep")
        cache.save_generated_code("new", "source")
        self.assertFalse(list(self.root.rglob("*.json.tmp")))
        self.assertEqual(unrelated.read_bytes(), b"keep")
        self.assertEqual(cache.load_generated_code("new"), "source")


if __name__ == "__main__":
    from helion._testing import main

    main()
