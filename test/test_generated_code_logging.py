from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import PyCodeCache
from helion.runtime.settings import _warn_unsupported_code_cache_backend


@helion.kernel(backend="triton")
def _logging_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="triton")
def _logging_divide(x: torch.Tensor, shape: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / shape[0]
    return out


class TestGeneratedCodeLogging(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _warn_unsupported_code_cache_backend.cache_clear()
        self.addCleanup(_warn_unsupported_code_cache_backend.cache_clear)

    def test_environment_backend_fallback_warns_once_per_backend(self) -> None:
        with (
            patch.dict("os.environ", {"HELION_CACHE_LEVEL": "code"}),
            self.assertLogs("helion.runtime.settings", level="WARNING") as logs,
        ):
            for backend in ("cute", "cute", "pallas", "pallas"):
                settings = helion.Settings(backend=backend)
                self.assertEqual(settings.cache_level, "config")
        self.assertEqual(len(logs.records), 2)
        self.assertIn("backend='cute'", logs.output[0])
        self.assertIn("backend='pallas'", logs.output[1])

    def test_effective_cache_level_is_logged(self) -> None:
        with (
            patch.dict("os.environ", {"HELION_CACHE_LEVEL": "code"}),
            self.assertLogs("helion.runtime.settings", level="INFO") as logs,
        ):
            helion.Settings(backend="triton")
            helion.Settings(backend="triton", cache_level="config")
            helion.Settings(backend="cute")
        messages = [record.getMessage() for record in logs.records]
        self.assertIn("Using cache_level='code' for backend='triton'", messages)
        self.assertIn("Using cache_level='config' for backend='triton'", messages)
        self.assertIn("Using cache_level='config' for backend='cute'", messages)


class TestGeneratedCodeFallbackLogging(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        cache._warn_cache_unavailable_once.cache_clear()
        self.addCleanup(cache._warn_cache_unavailable_once.cache_clear)
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.cache_root = Path(root)
        self._test_stack.enter_context(
            patch.dict(
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(
                patch.object(cache, name, return_value=f"{name}-v1")
            )
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )

    def test_ordinary_cold_miss_has_no_warning(self) -> None:
        config = helion.Config(block_sizes=[32])
        kernel = helion.kernel(backend="triton", cache_level="code", config=config)(
            _logging_add.fn
        )
        x = torch.ones(16)

        def load_source(source: str, *, extra: str = "") -> SimpleNamespace:
            path = self.cache_root / "logging_add.py"
            path.write_text(source)
            return SimpleNamespace(__file__=str(path), _logging_add=lambda x: x + 1)

        with (
            self.assertNoLogs("helion.runtime.generated_code_cache", level="WARNING"),
            patch.object(PyCodeCache, "load", side_effect=load_source),
        ):
            bound = kernel.bind((x,))
            bound.set_config(config)
            torch.testing.assert_close(bound(x), x + 1)
        self.assertTrue(list((self.cache_root / "generated_code").glob("*.json")))

    def test_descriptor_fallback_warns_once_across_bindings(self) -> None:
        config = helion.Config(block_sizes=[32], indexing="tensor_descriptor")
        x = torch.ones(16)
        with self.assertLogs(
            "helion.runtime.generated_code_cache", level="WARNING"
        ) as logs:
            for _iteration in range(2):
                kernel = helion.kernel(
                    backend="triton", cache_level="code", config=config
                )(_logging_add.fn)
                bound = kernel.bind((x,))
                # Exercise the real set_config eligibility path without a
                # hardware-dependent tensor-descriptor compilation.
                with patch.object(
                    bound, "compile_config", return_value=lambda x: x + 1
                ):
                    bound.set_config(config)
                    bound.set_config(config)
                    torch.testing.assert_close(bound(x), x + 1)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("tensor-descriptor", logs.output[0])
        self.assertFalse(list((self.cache_root / "generated_code").glob("*.json")))

    def test_unsupported_constexpr_warns_once_and_writes_no_entries(self) -> None:
        kernel = helion.kernel(backend="triton", cache_level="code")(_logging_divide.fn)
        args = (torch.ones(16), hl.constexpr(torch.Size([16])))
        with self.assertLogs(
            "helion.runtime.generated_code_cache", level="WARNING"
        ) as logs:
            self.assertIsNone(cache.binding_input_key(kernel, args))
            self.assertIsNone(cache.binding_input_key(kernel, args))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("unsupported value", logs.output[0])
        self.assertFalse((self.cache_root / "generated_code").exists())
