from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="triton")
def _policy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


class TestGeneratedCodePolicy(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.cache_root = Path(root)
        self._test_stack.enter_context(
            patch.dict(
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        self._test_stack.enter_context(
            patch.object(cache, "helion_key", return_value="helion-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "torch_key_wrapper", return_value="torch-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "triton_key_wrapper", return_value="triton-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.sources: list[str] = []
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load_source)
        )
        self.x = torch.ones(16)
        self.config = helion.Config(block_sizes=[32])

    def _load_source(self, source: str, *, extra: str = "") -> SimpleNamespace:
        self.sources.append(source)
        path = self.cache_root / f"policy_{len(self.sources)}.py"
        path.write_text(source)
        return SimpleNamespace(__file__=str(path), _policy_kernel=lambda x: x + 1)

    def _kernel(self, **settings: object) -> Kernel:
        return helion.kernel(backend="triton", generated_code_cache=True, **settings)(
            _policy_kernel.fn
        )

    def test_adaptive_tuning_ignores_stale_binding_manifest(self) -> None:
        kernel = self._kernel(autotune_effort="quick")
        bound = kernel.bind((self.x,))
        bound.set_config(self.config)
        signature = kernel._base_specialization_key((self.x,))
        self.assertIsNone(cache.compiled_kernel_cache_key(kernel, (self.x,), signature))
        # Model a manifest written before adaptive tuning was excluded. The
        # artifact must not pin the old winner when tuning chooses another one.
        with patch.object(cache, "default_autotuner_fn", object()):
            stale_key = cache.compiled_kernel_cache_key(kernel, (self.x,), signature)
        assert stale_key is not None
        source_key = next((self.cache_root / "generated_code").glob("*.json")).stem
        normalized = bound._normalized_config_copy(self.config)
        cache.save_compiled_kernel(stale_key, self.config, normalized, source_key)
        kernel.reset()
        with (
            patch(
                "helion.runtime.cached_kernel.load_binding_schema",
                side_effect=AssertionError("adaptive tuning consulted a manifest"),
            ),
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
        ):
            fresh = kernel.bind((self.x,))
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIsNone(fresh._user_provided_config())
        changed_winner = helion.Config(block_sizes=[64])
        # Supply the changed tuning result without GPU benchmarking. The public
        # autotune path must ask the backend rather than reuse the old manifest.
        with patch.object(
            fresh.env.backend, "autotune", return_value=changed_winner
        ) as tune:
            self.assertEqual(fresh.autotune((self.x,), force=False), changed_winner)
        tune.assert_called_once()
        torch.testing.assert_close(fresh(self.x), self.x + 1)
        self.assertNotEqual(self.sources[0], self.sources[-1])

    def test_explicit_config_reuses_frontend_artifact(self) -> None:
        self._kernel(config=self.config).bind((self.x,)).autotune(
            (self.x,), force=False
        )
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            fresh = self._kernel(config=self.config).bind((self.x,))
            self.assertEqual(fresh.autotune((self.x,), force=False), self.config)
        self.assertEqual(self.sources[0], self.sources[1])
        torch.testing.assert_close(fresh(self.x), self.x + 1)

    def test_force_autotune_bypasses_existing_frontend_artifact(self) -> None:
        self._kernel(config=self.config).bind((self.x,)).set_config(self.config)
        forced = self._kernel(config=self.config, force_autotune=True)
        signature = forced._base_specialization_key((self.x,))
        self.assertIsNone(cache.compiled_kernel_cache_key(forced, (self.x,), signature))
        with (
            patch(
                "helion.runtime.cached_kernel.load_binding_schema",
                side_effect=AssertionError("force autotune consulted a manifest"),
            ),
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
        ):
            fresh = forced.bind((self.x,))
        self.assertEqual(compile_fn.call_count, 1)
        self.assertIsNone(fresh._user_provided_config())
