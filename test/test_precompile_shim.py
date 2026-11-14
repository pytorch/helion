from __future__ import annotations

from collections import OrderedDict
import importlib
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

from packaging.version import Version
import pytest
import torch

from helion._argument_device import _find_argument_device
from helion._utils import triton_is_available
from helion.autotuner.logger import SUPPRESSED_TRITON_CODE_MSG
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.precompile_future import PrecompileContext
from helion.autotuner.precompile_future import PrecompileFuture
from helion.runtime.config import Config
from helion.runtime.precompile_shim import _device_probe_values
from helion.runtime.precompile_shim import make_precompiler
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


def test_precompile_warning_formats_message_and_preserves_exception(capsys) -> None:
    decorator = "@helion.kernel(config=Config(block_sizes=[32]))"
    settings = Settings(autotune_precompile="fork")
    kernel = Mock()
    kernel.format_kernel_decorator.return_value = decorator
    context = PrecompileContext(settings, AutotuningLogger(settings), kernel, (), 1)
    with (
        patch("helion.autotuner.precompile_future.maybe_dump_triton_failure"),
        patch("helion.autotuner.precompile_future.log_generated_triton_code_debug"),
        pytest.raises(RuntimeError, match="precompile failed"),
    ):
        PrecompileFuture.create(
            context,
            Config(),
            Mock(side_effect=RuntimeError("precompile failed")),
            (),
            "unused-result",
            None,
        )
    output = capsys.readouterr().err
    assert (
        f"Helion autotuner precompile error for {decorator}. "
        f"{SUPPRESSED_TRITON_CODE_MSG}\n"
    ) in output
    assert "%s" not in output
    assert "Traceback (most recent call last):" in output
    assert "RuntimeError: precompile failed" in output


@pytest.mark.parametrize(
    "error,ignore,level,visible",
    [
        (
            torch.cuda.OutOfMemoryError("allocation failed"),
            False,
            logging.WARNING,
            False,
        ),
        (torch.cuda.OutOfMemoryError("allocation failed"), False, logging.DEBUG, True),
        (RuntimeError("PassManager::run failed"), False, logging.WARNING, True),
        (RuntimeError("unexpected compiler bug"), True, logging.DEBUG, False),
        (torch.cuda.OutOfMemoryError("allocation failed"), True, logging.DEBUG, False),
    ],
)
def test_precompile_diagnostics_follow_error_policy(
    capsys, error, ignore, level, visible
) -> None:
    settings = Settings(
        autotune_precompile="fork",
        autotune_ignore_errors=ignore,
        autotune_log_level=level,
    )
    kernel = Mock()
    kernel.format_kernel_decorator.return_value = "@helion.kernel()"
    context = PrecompileContext(settings, AutotuningLogger(settings), kernel, (), 1)
    with (
        patch("helion.autotuner.precompile_future.maybe_dump_triton_failure"),
        patch("helion.autotuner.precompile_future.log_generated_triton_code_debug"),
    ):
        future = PrecompileFuture.create(
            context, Config(), Mock(side_effect=error), (), "unused-result", None
        )
    assert future.ok is False and future.process is None
    output = capsys.readouterr().err
    assert ("Helion autotuner precompile error" in output) is visible
    assert ("Traceback (most recent call last):" in output) is visible
    if visible:
        assert str(error) in output


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


if __name__ == "__main__":
    unittest.main()
