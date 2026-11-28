"""Tests for canonical runtime embedding in autotuner AOT output."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import onlyOn
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfCute
from helion._testing import skipIfNotCUDA
from helion.autotuner.aot_cache import _config_uses_tensor_descriptor
from helion.autotuner.aot_compile import generate_standalone_file
import helion.language as hl


def _synthetic_code(kernel_name: str = "add") -> str:
    return f"""from helion.runtime import default_launcher as _default_launcher
import torch

def _helion_{kernel_name}():
    pass

def {kernel_name}(x):
    helion.runtime.set_triton_allocator()
    helion.runtime.get_num_sm(x)
    helion.runtime.get_num_xcd(x)
    return _default_launcher
"""


# The AOT file embeds the Triton launcher (default_launcher shim); CuTe AOT
# output keeps its helion launcher import (test_cute_structural_aot*.py cover it).
_TRITON_RUNTIME_EMBEDDING = (
    "checks the embedded Triton launcher; CuTe AOT keeps its helion launcher"
)


@onlyBackends(["triton", "cute"])
@skipIfCute(_TRITON_RUNTIME_EMBEDDING)
class TestAOTRuntimeEmbedding(TestCase):
    def test_invalid_runtime_contract_is_rejected(self) -> None:
        cases = (
            (
                "alias",
                "add",
                _synthetic_code().replace("as _default_launcher", "as launch"),
                "unsupported Helion import",
            ),
            (
                "helper",
                "add",
                _synthetic_code().replace("get_num_xcd", "unknown_helper"),
                "unsupported Helion runtime helper",
            ),
            ("collision", "helion", _synthetic_code("helion"), "collides"),
        )
        for name, kernel_name, code, error in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as tmp,
                self.assertRaisesRegex(ValueError, error),
            ):
                generate_standalone_file(
                    kernel_name,
                    [code],
                    "",
                    Path(tmp),
                )


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
)
def _aot_add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[64, 64], prefetch_indexing="tensor_descriptor"),
)
def _aot_prefetch_copy(w: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(w)
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_k, tile_n in hl.tile(w.size()):
        out[tile_k, tile_n] = w[tile_k, tile_n]
    return out


@onlyBackends(["triton", "cute"])
@skipIfCute(_TRITON_RUNTIME_EMBEDDING)
@onlyOn(["cuda", "xpu"])
class TestAOTStandaloneRuntime(RefEagerTestDisabled, TestCase):
    def test_multiple_configs_share_and_execute_canonical_runtime(self) -> None:
        heuristic = "def key_add(*args):\n    return 0\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_standalone_file(
                "add",
                [_synthetic_code(), _synthetic_code()],
                heuristic,
                Path(tmp),
            )
            source = path.read_text()
            self.assertEqual(source.count("def _make_helion_runtime"), 1)
            self.assertEqual(source.count("def default_launcher("), 1)
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)

            name = "_helion_aot_runtime_test"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {name: module}):
                spec.loader.exec_module(module)
                self.assertIs(
                    module.add(torch.device(DEVICE)), module._default_launcher
                )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_box_prefetch_inlines_tensor_maps(self) -> None:
        """A box-form prefetch's tensor maps are inlined (no helion import),
        and its config counts as descriptor indexing for the dispatch's
        alignment predicates."""
        w = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        bound = _aot_prefetch_copy.bind((w, w.view(16, 64, 256)))
        self.assertTrue(_config_uses_tensor_descriptor(_aot_prefetch_copy.configs[0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_standalone_file(
                "_aot_prefetch_copy", [bound.to_triton_code()], "", Path(tmp)
            )
            source = path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertEqual(source.count("def _make_helion_tensor_maps"), 1)
            self.assertIn("cp.async.bulk.prefetch.tensor", source)
            name = "_helion_aot_prefetch_test"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            # Not registered in sys.modules: the inlined maps need no module.
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            out = module._aot_prefetch_copy(w, w.view(16, 64, 256))
            torch.testing.assert_close(out, w, atol=0, rtol=0)

    def test_generated_aot_reuses_canonical_runtime(self) -> None:
        x = torch.randn(64, device=DEVICE)
        bound = _aot_add_one.bind((x,))
        code = bound.to_triton_code()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_standalone_file(
                "_aot_add_one",
                [code],
                "",
                Path(tmp),
            )
            source = path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)

            name = "_helion_aot_prepared_test"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
                first = module._aot_add_one(x)
                torch.testing.assert_close(first, x + 1)
                jit_fn = module._helion__aot_add_one_c0

                other = torch.randn_like(x)
                with patch.object(jit_fn, "run", wraps=jit_fn.run) as jit_run:
                    second = module._aot_add_one(other)
                torch.testing.assert_close(second, other + 1)
                jit_run.assert_called_once()
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
