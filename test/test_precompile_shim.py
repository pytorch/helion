from __future__ import annotations

from collections import OrderedDict
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

from packaging.version import Version
import torch

from helion._argument_device import _find_argument_device
from helion._utils import triton_is_available
from helion.runtime.config import Config
from helion.runtime.precompile_shim import _device_probe_values
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

    def test_device_discovery_unwraps_tensor_descriptors(self) -> None:
        # A launch whose tensors all travel as tensor descriptors (every
        # indexing choice ``tensor_descriptor``, no bare tensor argument) used
        # to abort the whole autotune with NoTensorArgs in the fork precompile.
        from triton.tools.tensor_descriptor import TensorDescriptor

        base = torch.zeros(32, 64)
        descriptor = TensorDescriptor.from_tensor(base, [16, 16])
        probed = _device_probe_values([descriptor, 148])
        self.assertIs(probed[0], base)
        self.assertEqual(probed[1], 148)
        self.assertEqual(_find_argument_device(probed), base.device)

    def test_precompile_keys_the_launch_resident_fact(self) -> None:
        """The autotuner's precompile compiles (and keys) the binary the launch
        ships: the prefetch pass gets the launch's resident cohort (verify5 C0-a:
        it was missing, so a resident launch's precompile built another binary)."""
        from helion.runtime.triton import ttgir_weight_prefetch as twp

        kernel_cache: dict[object, object] = {}
        fn = Mock()
        fn.debug = False
        fn.device_caches = {
            0: (
                kernel_cache,
                object(),
                object(),
                Mock(),
                Mock(return_value=(OrderedDict(), [], SimpleNamespace())),
            )
        }
        fn._pack_args.return_value = (SimpleNamespace(), {}, {}, {})
        fn.compile.return_value = Mock()
        seen: list[tuple[object, str]] = []
        real_context = importlib.import_module(
            "helion.runtime.precompile_shim"
        ).ttgir_weight_prefetch_context

        def context(*a: object, **k: object) -> object:
            ctx = real_context(*a, **k)

            class _Spy:
                def __enter__(self) -> None:
                    ctx.__enter__()
                    facts = twp._current_facts()
                    seen.append((k.get("resident_ctas"), facts.key() if facts else ""))

                def __exit__(self, *exc: object) -> None:
                    ctx.__exit__(*exc)

            return _Spy()

        resident_calls: list[tuple[object, ...]] = []

        def resident(args: object, required: object, ring: object) -> int:
            resident_calls.append((required, ring))
            return 2

        kernel_module = importlib.import_module("helion.runtime.kernel")
        with (
            patch.object(kernel_module, "_find_device", return_value=0),
            patch(
                "helion.runtime.precompile_shim.get_triton_version",
                return_value=Version("3.7.0"),
            ),
            patch("triton.runtime.jit.compute_cache_key", return_value="k"),
            patch(
                "helion.runtime.precompile_shim.prefetch_alias_classes",
                return_value=(),
            ),
            patch("helion.runtime.precompile_shim.ring_resident_ctas", resident),
            patch(
                "helion.runtime.precompile_shim.ttgir_weight_prefetch_context",
                context,
            ),
        ):
            precompile = make_precompiler(fn, Config(), Mock())(
                object(),
                _ttgir_weight_prefetch=True,
                _ttgir_weight_ring=True,
                _minimum_resident_programs=296,
            )
            self.assertTrue(precompile(in_child_process=False))
        self.assertEqual(resident_calls, [(296, True)])
        # both the key computation and the compile run with the launch's fact,
        # keyed exactly as the launcher keys it
        self.assertEqual(seen, [(2, "-resident2"), (2, "-resident2")])
        self.assertEqual(twp._Facts(ring=True, resident_ctas=2).key(), "-resident2")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ring_resident_ctas_matches_the_launch(self) -> None:
        from triton.tools.tensor_descriptor import TensorDescriptor

        from helion.runtime.triton.launcher import ring_resident_ctas

        sms = torch.cuda.get_device_properties(0).multi_processor_count
        t = torch.zeros(32, 64, device="cuda")
        self.assertEqual(ring_resident_ctas((t, 3), 2 * sms, True), 2)
        # descriptor-only launches find the device through the base tensor
        d = TensorDescriptor.from_tensor(t, [16, 16])
        self.assertEqual(ring_resident_ctas((d,), 2 * sms, True), 2)
        self.assertEqual(ring_resident_ctas((t,), 2 * sms, False), 0)
        self.assertEqual(ring_resident_ctas((t,), 0, True), 0)


if __name__ == "__main__":
    unittest.main()
