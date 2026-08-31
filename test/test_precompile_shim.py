from __future__ import annotations

from collections import OrderedDict
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

from packaging.version import Version

from helion._utils import triton_is_available
from helion.runtime.config import Config
from helion.runtime.precompile_shim import make_precompiler


@unittest.skipUnless(triton_is_available(), "requires Triton")
class TestPrecompileShim(unittest.TestCase):
    def test_matches_triton_runtime_cache_key_and_argument_packing(self) -> None:
        kernel_cache: dict[object, object] = {}
        kernel_key_cache = object()
        target = object()
        backend = Mock()
        specialization = [("*fp32", "D")]
        bound_args = OrderedDict((("x", object()),))
        binder_options = SimpleNamespace(binder=True)
        packed_options = SimpleNamespace(packed=True)
        binder = Mock(return_value=(bound_args, specialization, binder_options))
        compiled_kernel = Mock()
        fn = Mock()
        fn.debug = False
        fn.device_caches = {
            0: (kernel_cache, kernel_key_cache, target, backend, binder)
        }
        fn._pack_args.return_value = (
            packed_options,
            {"x": "*fp32"},
            {},
            {(0,): "tt.divisibility"},
        )
        fn.ASTSource.return_value = "source"
        fn.compile.return_value = compiled_kernel
        kernel_module = importlib.import_module("helion.runtime.kernel")

        with (
            patch.object(kernel_module, "_find_device", return_value=0),
            patch(
                "helion.runtime.precompile_shim.get_triton_version",
                return_value=Version("3.7.0"),
            ),
            patch(
                "triton.runtime.jit.compute_cache_key",
                return_value="runtime-cache-key",
            ) as compute_cache_key,
        ):
            precompile = make_precompiler(fn, Config(), Mock())(object())
            self.assertTrue(precompile(in_child_process=False))

        compute_cache_key.assert_called_once_with(
            kernel_key_cache, specialization, binder_options
        )
        fn._pack_args.assert_called_once_with(
            backend, ANY, bound_args, specialization, binder_options
        )
        fn.ASTSource.assert_called_once_with(
            fn,
            {"x": "*fp32"},
            {},
            {(0,): "tt.divisibility"},
        )
        fn.compile.assert_called_once_with(
            "source", target=target, options={"packed": True}
        )
        self.assertIs(kernel_cache["runtime-cache-key"], compiled_kernel)
        compiled_kernel._init_handles.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
