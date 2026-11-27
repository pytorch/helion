from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch._inductor import config as inductor_config

import helion
from helion._compiler.output_code_utils import dependency_free_runtime_source
from helion._compiler.triton.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRocm
from helion._utils import next_power_of_2
from helion.autotuner import BooleanFragment
from helion.autotuner import PowerOfTwoFragment
from helion.experimental import ArtifactCase
from helion.experimental import export_artifact
from helion.experimental import load_artifact
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache

pytestmark = pytest.mark.cpu_only

_LIVE_BIAS = torch.ones(16)


def _artifact_add(x: torch.Tensor, y: torch.Tensor, scale: float = 2.0):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile] * scale
    return out


def _artifact_maximum(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.maximum(x[tile], y[tile])
    return out


def _artifact_prefetch(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        hl.prefetch(y, [tile_m, tile_n])
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


def _artifact_live_global(x: torch.Tensor):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + _LIVE_BIAS[tile]
    return out


def _artifact_registered_op(x: torch.Tensor):
    out = torch.ops.helion_artifact_test.clone(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = out[tile] + 1
    return out


def _artifact_indirect_registered_op(x: torch.Tensor):
    out = torch.functional.torch.ops.helion_artifact_test.clone(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = out[tile] + 1
    return out


def _kernel(fn, config=None):
    return helion.kernel(
        backend="triton",
        config=config or helion.Config(block_sizes=[32]),
        static_shapes=False,
        fast_math=False,
        autotune_effort="none",
    )(fn)


# The complete kernel body below comes from the pinned public vLLM source.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# https://github.com/vllm-project/vllm/blob/d88feb483b9c4cd8f507ebbe88f0d785a15dfb35/vllm/kernels/helion/ops/scaled_mm.py
def scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    b_scales: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    a_scales = a_scales.reshape(-1, 1) if a_scales.dim() <= 1 else a_scales
    b_scales = b_scales.reshape(-1, 1) if b_scales.dim() <= 1 else b_scales

    assert a_scales.dtype == b_scales.dtype and a_scales.is_floating_point()
    assert a_scales.shape[1] == 1 and (a_scales.shape[0] == 1 or a_scales.shape[0] == M)
    assert b_scales.shape[1] == 1 and (b_scales.shape[0] == 1 or b_scales.shape[0] == N)
    hl.specialize(b_scales.shape[1])

    out_dtype = out.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block_size = helion.next_power_of_2(helion.cdiv(K, split_k))
    if split_k > 1:
        out.zero_()

    swap_ab = hl.register_tunable("swap_ab", BooleanFragment())

    for tile_m, tile_n, outer_k in hl.tile(
        [M, N, K], block_size=[None, None, k_block_size]
    ):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        acc_t = acc.t()
        for tile_k in hl.tile(outer_k.begin, outer_k.end):
            if swap_ab:
                a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
                b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])
                acc_t = hl.dot(
                    b_blk,
                    a_blk,
                    acc=acc_t,
                    out_dtype=acc_dtype,
                )
            else:
                acc = hl.dot(
                    a[tile_m, tile_k],
                    b[tile_k, tile_n],
                    acc=acc,
                    out_dtype=acc_dtype,
                )

        if swap_ab:
            acc = acc_t.t().to(torch.float32)
        else:
            acc = acc.to(torch.float32)

        if a_scales.shape[0] == M:
            a_scales_blk = a_scales[tile_m, :]
        else:
            a_scales_blk = a_scales[0, 0].expand(tile_m.block_size, 1)
        acc = a_scales_blk * acc

        if b_scales.shape[0] == N:
            b_scales_blk = b_scales[tile_n, :]
        else:
            b_scales_blk = b_scales[0, 0].expand(tile_n.block_size, 1)
        acc = b_scales_blk.T * acc

        out_blk = acc.to(out_dtype)

        if bias is not None:
            if outer_k.begin == 0:
                out_blk += bias[tile_n]

        if split_k == 1:
            out[tile_m, tile_n] = out_blk
        else:
            hl.atomic_add(out, [tile_m, tile_n], out_blk)


def _scaled_mm_baseline(out, a, b, a_scales, b_scales, bias=None):
    result = torch.mm(a.to(torch.float32), b.to(torch.float32))
    result = a_scales * result
    result = b_scales.T * result
    result = result.to(out.dtype)
    if bias is not None:
        result = result + bias
    out.copy_(result)


def _scaled_mm_kernel():
    # Retain the real wrapper's warning and autotune-baseline settings.
    return helion.kernel(
        config=helion.Config(block_sizes=[16, 32, 32], split_k=1, swap_ab=False),
        settings=helion.Settings(
            backend="triton",
            static_shapes=False,
            autotune_effort="none",
            autotune_baseline_fn=_scaled_mm_baseline,
            autotune_baseline_atol=1e-1,
            autotune_baseline_rtol=1e-1,
            ignore_warnings=[
                helion.exc.TensorOperationInWrapper,
                helion.exc.ProcessGroupNameNotFound,
            ],
        ),
    )(scaled_mm)


def _scaled_mm_inputs(device="cpu"):
    a = torch.randint(-3, 4, (16, 64), device=device, dtype=torch.int8)
    b = torch.randint(-3, 4, (64, 64), device=device, dtype=torch.int8).t()
    a_scales = torch.full((16, 1), 0.5, device=device)
    b_scales = torch.full((64, 1), 0.25, device=device)
    out = torch.empty((16, 64), device=device, dtype=torch.bfloat16)
    return out, a, b, a_scales, b_scales


@onlyBackends(["triton"])
class TestKernelArtifact(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        self.root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        self.x = torch.ones(16)
        self.y = torch.zeros(16)
        self.config = helion.Config(block_sizes=[32])

    def _export(self, fn=_artifact_add, *, name="program.py", cases=None):
        return export_artifact(
            _kernel(fn),
            cases or [ArtifactCase((self.x, self.y), self.config)],
            self.root / name,
        )

    @staticmethod
    def _source(path):
        return path.read_text().partition("\n")[2]

    def test_export_load_preserves_defaults_and_explicit_cases_without_compilation(
        self,
    ):
        path = self._export(
            cases=[
                ArtifactCase((self.x, self.y), self.config),
                ArtifactCase((self.x, self.y), self.config, {"scale": 3.0}),
            ]
        )
        self.assertIn("@triton.jit", self._source(path))
        with (
            patch.object(
                KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
            ),
            patch.object(
                BoundKernel, "to_code", side_effect=AssertionError("codegen ran")
            ),
            patch.object(
                PyCodeCache, "load", side_effect=AssertionError("CPU source executed")
            ) as loader,
        ):
            frozen = load_artifact(path)
            # Matching CPU calls reach the documented accelerator restriction,
            # demonstrating signature acceptance without pretending to run Triton.
            for args, kwargs in (
                ((torch.zeros_like(self.x), torch.ones_like(self.y)), {}),
                ((), {"x": self.x, "y": self.y, "scale": 3.0}),
            ):
                with self.assertRaisesRegex(ValueError, "accelerator tensors"):
                    frozen(*args, **kwargs)
            loader.assert_not_called()

    def test_unknown_input_contract_rejected_before_loading_source(self):
        frozen = load_artifact(self._export())
        calls = {
            "shape": ((torch.ones(17), self.y), {}),
            "dtype": ((self.x.double(), self.y), {}),
            "stride": ((torch.ones(32)[::2], self.y), {}),
            "offset": ((torch.ones(17)[1:], self.y), {}),
            "object_alias": ((self.x, self.x), {}),
            "storage_alias": ((self.x, self.x.view_as(self.x)), {}),
            "scalar": ((self.x, self.y), {"scale": 4.0}),
            "requires_grad": ((self.x.detach().requires_grad_(), self.y), {}),
            "missing_argument": ((self.x,), {}),
        }
        with patch.object(
            PyCodeCache, "load", side_effect=AssertionError("unsupported call executed")
        ) as loader:
            for name, (args, kwargs) in calls.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    frozen(*args, **kwargs)
            loader.assert_not_called()

    def test_unsupported_scalars_are_rejected_at_export_and_call(self):
        unsupported = (
            (float("nan"), "NaN scalar"),
            (b"2", "Unsupported artifact input type bytes"),
        )
        frozen = load_artifact(self._export())
        for value, message in unsupported:
            with self.subTest(value=value):
                with (
                    patch.object(helion.Kernel, "bind") as bind,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._export(
                        name="unsupported.py",
                        cases=[ArtifactCase((self.x, self.y, value), self.config)],
                    )
                bind.assert_not_called()
                self.assertFalse((self.root / "unsupported.py").exists())
                with (
                    patch.object(PyCodeCache, "load") as loader,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    frozen(self.x, self.y, value)
                loader.assert_not_called()

    def test_integrity_failure_precedes_source_loading(self):
        path = self._export()
        original = path.read_text()
        for old, new in (
            ("tl.store(", "tl.load("),
            ('"entrypoint":"_artifact_add"', '"entrypoint":"other"'),
        ):
            corrupted = original.replace(old, new, 1)
            self.assertNotEqual(original, corrupted)
            path.write_text(corrupted)
            with (
                patch.object(PyCodeCache, "load") as loader,
                self.assertRaisesRegex(ValueError, "integrity"),
            ):
                load_artifact(path)
            loader.assert_not_called()

    def test_triton_version_comes_from_imported_module(self):
        triton = types.SimpleNamespace(__version__="3.7.1+git00000001")
        imports = types.SimpleNamespace(import_module=Mock(return_value=triton))
        with (
            patch("helion.experimental.artifact.importlib", imports),
            patch.object(PyCodeCache, "load") as loader,
        ):
            path = self._export()
            load_artifact(path)
            imports.import_module.assert_called_with("triton")
            triton.__version__ = "3.7.1+git00000002"
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                load_artifact(path)
            loader.assert_not_called()

    def test_runtime_mismatch_and_changed_factory_defaults_are_rejected(self):
        path = self._export()
        frozen = load_artifact(path)
        with (
            patch.object(torch, "__version__", "different-artifact-runtime"),
            patch.object(PyCodeCache, "load") as loader,
        ):
            with self.assertRaisesRegex(ValueError, "runtime"):
                load_artifact(path)
            loader.assert_not_called()
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(
                torch.float64 if original_dtype != torch.float64 else torch.float32
            )
            with patch.object(PyCodeCache, "load") as loader:
                with self.assertRaisesRegex(ValueError, "runtime"):
                    load_artifact(path)
                with self.assertRaisesRegex(ValueError, "defaults"):
                    frozen(self.x, self.y)
                loader.assert_not_called()
        finally:
            torch.set_default_dtype(original_dtype)

    def test_live_original_module_tensor_cannot_be_exported(self):
        with self.assertRaises(ValueError):
            self._export(
                _artifact_live_global,
                cases=[ArtifactCase((self.x,), self.config)],
            )
        self.assertFalse((self.root / "program.py").exists())

    def test_registered_host_operator_cannot_be_exported(self):
        library = torch.library.Library("helion_artifact_test", "FRAGMENT")
        self.addCleanup(library._destroy)
        library.define("clone(Tensor x) -> Tensor")
        library.impl("clone", torch.clone, "CompositeExplicitAutograd")
        for kernel in (_artifact_registered_op, _artifact_indirect_registered_op):
            with (
                self.subTest(kernel=kernel.__name__),
                self.assertRaisesRegex(ValueError, "Torch operator registry"),
            ):
                self._export(kernel, cases=[ArtifactCase((self.x,), self.config)])
        self.assertFalse((self.root / "program.py").exists())

    def test_compiler_configuration_changes_require_a_new_explicit_export(self):
        with inductor_config.patch(use_fast_math=True):
            fast_path = self._export(_artifact_maximum, name="fast.py")
        fast_bytes = fast_path.read_bytes()
        with inductor_config.patch(use_fast_math=False):
            strict_path = self._export(_artifact_maximum, name="strict.py")
            with (
                patch.object(
                    KernelCompiler,
                    "compile",
                    side_effect=AssertionError("frontend ran"),
                ),
                patch.object(
                    BoundKernel, "to_code", side_effect=AssertionError("codegen ran")
                ),
                patch.object(
                    PyCodeCache,
                    "load",
                    side_effect=AssertionError("CPU source executed"),
                ),
                patch.dict(_artifact_maximum.__globals__, {"_artifact_maximum": None}),
            ):
                old_selected = load_artifact(fast_path)
                new_selected = load_artifact(strict_path)
                for frozen in (old_selected, new_selected):
                    with self.assertRaisesRegex(ValueError, "accelerator tensors"):
                        frozen(self.x, self.y)
        fast_source = self._source(fast_path)
        strict_source = self._source(strict_path)
        self.assertIn("tl.maximum(", fast_source)
        self.assertNotEqual(
            hashlib.sha256(fast_source.encode()).digest(),
            hashlib.sha256(strict_source.encode()).digest(),
        )
        self.assertEqual(fast_bytes, fast_path.read_bytes())

    def test_exact_vllm_kernel_and_wrapper_export(self):
        path = export_artifact(
            _scaled_mm_kernel(),
            [
                ArtifactCase(
                    _scaled_mm_inputs(),
                    helion.Config(block_sizes=[16, 32, 32], split_k=1, swap_ab=False),
                )
            ],
            self.root / "scaled_mm.py",
        )
        source = self._source(path)
        self.assertIn("tl.dot(", source)
        self.assertIn("def integer_full(", source)
        self.assertNotIn("from helion.runtime.triton import helpers", source)
        self.assertNotIn("_source_module", source)
        with (
            patch.object(
                KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
            ),
            patch.object(
                BoundKernel, "to_code", side_effect=AssertionError("codegen ran")
            ),
            patch.object(
                PyCodeCache, "load", side_effect=AssertionError("CPU source executed")
            ) as loader,
        ):
            frozen = load_artifact(path)
            with self.assertRaisesRegex(ValueError, "accelerator tensors"):
                frozen(*_scaled_mm_inputs())
            loader.assert_not_called()

    def test_actual_scaled_mm_host_wrapper_reaches_launcher(self) -> None:
        args = _scaled_mm_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = export_artifact(
                _scaled_mm_kernel(),
                [
                    ArtifactCase(
                        args,
                        helion.Config(
                            block_sizes=[16, 32, 32], split_k=1, swap_ab=False
                        ),
                    )
                ],
                Path(directory) / "scaled_mm.py",
            )
            tree = ast.parse(path.read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_scaled_mm_c0"
        )
        namespace = {"torch": torch, "types": types}
        exec(
            dependency_free_runtime_source(
                TritonBackend().dependency_free_launcher_info
            ),
            namespace,
        )
        # Exercise actual emitted host code on CPU. Only the GPU kernel/launcher
        # and Triton's integer helper are substituted; this is not a GPU result.
        namespace["triton"] = types.SimpleNamespace(next_power_of_2=next_power_of_2)
        namespace["_helion_scaled_mm_c0"] = object()
        exec(ast.unparse(wrapper), namespace)
        launch = Mock()
        namespace["_scaled_mm_c0"](*args, _launcher=launch)
        launch.assert_called_once()
        assert launch.call_args.args[7] == 64


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
@pytest.mark.cuda
@onlyBackends(["triton"])
class TestKernelArtifactCUDA(RefEagerTestDisabled, TestCase):
    @skipIfRocm("prefetch lowers to PTX")
    @skipIfCudaCapabilityLessThan((9, 0), reason="tensor prefetch needs TMA")
    def test_descriptor_prefetch_is_rejected_before_codegen(self):
        x = torch.zeros((96, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.zeros_like(x)
        config = helion.Config(
            block_sizes=[16, 64],
            indexing="pointer",
            prefetch_indexing="tensor_descriptor",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(BoundKernel, "to_code") as to_code,
        ):
            destination = Path(directory) / "prefetch.py"
            with self.assertRaisesRegex(
                ValueError, "Tensor-descriptor indexing and prefetch"
            ):
                export_artifact(
                    _kernel(_artifact_prefetch, config),
                    [ArtifactCase((x, y), config)],
                    destination,
                )
            to_code.assert_not_called()
            self.assertFalse(destination.exists())

    def test_scaled_mm_with_embedded_native_helpers(self):
        args = _scaled_mm_inputs("cuda")
        out, a, b, a_scales, b_scales = args
        expected = ((a.float() @ b.float()) * a_scales * b_scales.T).to(out.dtype)
        with tempfile.TemporaryDirectory() as directory:
            path = export_artifact(
                _scaled_mm_kernel(),
                [
                    ArtifactCase(
                        args,
                        helion.Config(
                            block_sizes=[16, 32, 32], split_k=1, swap_ab=False
                        ),
                    )
                ],
                Path(directory) / "scaled_mm.py",
            )
            with (
                patch.object(
                    KernelCompiler,
                    "compile",
                    side_effect=AssertionError("frontend ran"),
                ),
                patch.object(
                    BoundKernel, "to_code", side_effect=AssertionError("codegen ran")
                ),
            ):
                load_artifact(path)(*args)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    @pytest.mark.slow
    def test_fresh_process_execution_without_original_kernel_module(self):
        root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        code = """
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock
from unittest.mock import patch
import torch
from torch._inductor import config as inductor_config
import helion
from helion.experimental import ArtifactCase, export_artifact, load_artifact
from helion.runtime.kernel import BoundKernel, KernelCompiler

phase, fixture_path, destination = sys.argv[1:]
root = Path(destination)
if phase == "export":
    spec = importlib.util.spec_from_file_location("artifact_export_fixture", fixture_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    x = torch.ones(16, device="cuda")
    y = torch.zeros(16, device="cuda")
    config = helion.Config(block_sizes=[32])
    export_artifact(module._kernel(module._artifact_add), [ArtifactCase((x, y), config), ArtifactCase((x, y), config, {"scale": 3.0})], root / "add.py")
    export_artifact(module._scaled_mm_kernel(), [ArtifactCase(module._scaled_mm_inputs("cuda"), helion.Config(block_sizes=[16, 32, 32], split_k=1, swap_ab=False))], root / "scaled_mm.py")
    for enabled, name in ((True, "fast"), (False, "strict")):
        with inductor_config.patch(use_fast_math=enabled):
            export_artifact(module._kernel(module._artifact_maximum), [ArtifactCase((x, y), config)], root / (name + ".py"))
else:
    assert "artifact_export_fixture" not in sys.modules
    sys.modules["artifact_export_fixture"] = None
    with patch.object(KernelCompiler, "compile", side_effect=AssertionError("artifact load ran frontend")), patch.object(BoundKernel, "to_code", side_effect=AssertionError("artifact load ran codegen")):
        add = load_artifact(root / "add.py")
        mm = load_artifact(root / "scaled_mm.py")
        with inductor_config.patch(use_fast_math=False):
            fast = load_artifact(root / "fast.py")
            strict = load_artifact(root / "strict.py")
            nan_x = torch.ones(16, device="cuda")
            nan_y = torch.full((16,), 2.0, device="cuda")
            nan_x[0] = float("nan")
            nan_y[1] = float("nan")
            torch.testing.assert_close(fast(nan_x, nan_y), torch.fmax(nan_x, nan_y), equal_nan=True)
            torch.testing.assert_close(strict(nan_x, nan_y), torch.maximum(nan_x, nan_y), equal_nan=True)
        with inductor_config.patch(use_fast_math=True):
            torch.testing.assert_close(strict(nan_x, nan_y), torch.maximum(nan_x, nan_y), equal_nan=True)
        for seed in (0, 1):
            torch.manual_seed(seed)
            x = torch.randn(16, device="cuda")
            y = torch.randn(16, device="cuda")
            torch.testing.assert_close(add(x, y), x + y * 2.0)
            torch.testing.assert_close(add(y=y, x=x, scale=3.0), x + y * 3.0)
            a = torch.randint(-3, 4, (16, 64), device="cuda", dtype=torch.int8)
            b = torch.randint(-3, 4, (64, 64), device="cuda", dtype=torch.int8).t()
            a_scales = torch.full((16, 1), 0.5, device="cuda")
            b_scales = torch.full((64, 1), 0.25, device="cuda")
            out = torch.empty((16, 64), device="cuda", dtype=torch.bfloat16)
            expected = ((a.float() @ b.float()) * a_scales * b_scales.T).to(out.dtype)
            mm(out, a, b, a_scales, b_scales)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        for args in ((x.double(), y), (torch.ones(17, device="cuda"), y), (torch.ones(32, device="cuda")[::2], y), (torch.ones(17, device="cuda")[1:], y), (x, x), (x, x.view_as(x))):
            try:
                add(*args)
            except ValueError:
                pass
            else:
                raise AssertionError("unsupported input signature was accepted")
    assert sys.modules["artifact_export_fixture"] is None
"""
        for phase in ("export", "load"):
            with self.subTest(phase=phase):
                process = subprocess.run(
                    [sys.executable, "-c", code, phase, __file__, str(root)],
                    env={**os.environ, "HELION_BACKEND": "triton", "ENABLE_TILE": "0"},
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
