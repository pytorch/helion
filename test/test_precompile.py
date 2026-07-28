from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessPallas
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[64, 64], pid_type="persistent_blocked")
)
def add_persistent(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Persistent grid -> the launch computes its grid from ``get_num_sm``."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[64, 64], indexing="tensor_descriptor"))
def add_tma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """TMA (tensor-descriptor) indexing -> the launch calls
    ``set_triton_allocator`` for the device-side descriptor allocator."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[32, 32]), static_shapes=False)
def add_dynamic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """static_shapes=False -> the standalone must run at any shape: it inlines the
    real host wrapper, which derives the grid + output shape from the runtime
    input rather than baking the precompiled sample shape."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _run_standalone_without_helion(
    path: str, entrypoint: str, shape: tuple[int, int]
) -> str:
    """Import ``path`` and run ``entrypoint`` in a subprocess where importing
    ``helion`` is blocked, proving the standalone has zero Helion dependency."""
    m, n = shape
    script = textwrap.dedent(f"""
        import sys
        # Poison the helion package so any leftover dependency fails loudly.
        sys.modules["helion"] = None
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("_standalone", {path!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        torch.manual_seed(0)
        x = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
        y = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
        out = mod.{entrypoint}(x, y)
        torch.testing.assert_close(out, x + y)
        print("STANDALONE_OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if "STANDALONE_OK" not in result.stdout:
        raise AssertionError(
            f"standalone run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


@onlyBackends(["triton"])
@skipIfRefEager("precompile compiles real Triton code; not meaningful in ref-eager")
class TestPrecompile(TestCase):
    def test_source_has_no_helion_dependency(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_precompiled.py"
            result = helion.precompile(
                helion.PrecompilationInput(
                    kernel=add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            source = out_path.read_text()

        self.assertEqual(result.entrypoint_name, "add")
        # No *runtime* dependency on helion: no imports and no qualified
        # helion.runtime.* calls. (Inlined launcher docstrings may still mention
        # helion for provenance; test_standalone_runs_without_helion is the
        # authoritative proof that importing helion is never needed.)
        self.assertNotIn("import helion", source)
        self.assertNotIn("from helion", source)
        self.assertNotIn("helion.runtime.get_num_sm(", source)
        self.assertNotIn("helion.runtime.get_num_xcd(", source)
        self.assertNotIn("helion.runtime.set_triton_allocator(", source)
        # The dependency-free launcher is inlined.
        self.assertIn("def default_launcher(", source)
        self.assertIn("_default_launcher = default_launcher", source)
        self.assertIn("def add(", source)

    def test_standalone_runs_without_helion(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_precompiled.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            _run_standalone_without_helion(str(out_path), "add", (128, 128))

    def test_custom_entrypoint_name(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_precompiled.py"
            result = helion.precompile(
                helion.PrecompilationInput(
                    kernel=add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                    entrypoint_name="fused_add",
                )
            )
            source = out_path.read_text()

        self.assertEqual(result.entrypoint_name, "fused_add")
        self.assertIn("def fused_add(", source)
        self.assertNotIn("def add(", source)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "fused.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                    entrypoint_name="fused_add",
                )
            )
            _run_standalone_without_helion(str(out_path), "fused_add", (64, 64))

    def test_persistent_kernel_inlines_get_num_sm(self) -> None:
        """A persistent kernel calls ``helion.runtime.get_num_sm`` to size its
        grid; the standalone must inline it (no qualified reference remains) and
        still run without helion."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_persistent.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=add_persistent,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertNotIn("helion.runtime.get_num_sm(", source)
            # Assert the rewritten bare call in the generated-kernel section, not
            # the whole file: the inlined launcher also *defines* ``get_num_sm``,
            # so a whole-file assertIn would pass even without the call rewrite.
            gen = source.split("# --- generated kernel ---", 1)[1]
            self.assertIn("get_num_sm(", gen)
            _run_standalone_without_helion(str(out_path), "add_persistent", (128, 128))

    def test_tma_kernel_inlines_set_triton_allocator(self) -> None:
        """A tensor-descriptor (TMA) kernel calls
        ``helion.runtime.set_triton_allocator``; the standalone must inline it and
        still run without helion."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_tma.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=add_tma,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertNotIn("helion.runtime.set_triton_allocator(", source)
            # Assert the rewritten bare call in the generated-kernel section, not
            # the whole file: the inlined launcher also *defines*
            # ``set_triton_allocator``, so a whole-file assertIn would be vacuous.
            gen = source.split("# --- generated kernel ---", 1)[1]
            self.assertIn("set_triton_allocator(", gen)
            _run_standalone_without_helion(str(out_path), "add_tma", (128, 128))

    def test_dynamic_shapes_run_at_other_shapes(self) -> None:
        """A static_shapes=False kernel precompiled at one sample shape runs
        correctly at OTHER shapes: the standalone inlines the real host wrapper,
        which derives the grid + output shape from the runtime input rather than
        freezing the precompiled sample shape."""
        x = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "add_dynamic.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=add_dynamic,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            # Shapes other than the (128, 128) sample it was precompiled with.
            for shape in ((128, 128), (256, 64), (64, 256)):
                _run_standalone_without_helion(str(out_path), "add_dynamic", shape)


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[8]), static_shapes=False, backend="pallas"
)
def pallas_rows_times_two(x: torch.Tensor) -> torch.Tensor:
    """static_shapes=False Pallas (torch-tensor) kernel: the standalone must run at
    any leading dim -- it inlines the real host wrapper, which derives the grid +
    output shape from the runtime input rather than the precompiled sample shape."""
    n, _d = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile, :] = x[tile, :] * 2.0
    return out


def _import_standalone(path: str, name: str) -> object:
    """Import a generated standalone module.

    The module is registered in ``sys.modules`` before ``exec_module`` so that
    any inlined ``@dataclass`` (e.g. in the Pallas launcher) can resolve
    ``cls.__module__`` during class creation -- exactly what the normal import
    machinery does for a real ``import``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


@skipUnlessPallas("precompile Pallas test requires the Pallas backend / TPU")
@skipIfRefEager("precompile compiles real kernels; not meaningful in ref-eager")
class TestPrecompilePallas(TestCase):
    def test_pallas_standalone_runs(self) -> None:
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pallas_add_standalone.py"
            res = helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                )
            )
            source = out_path.read_text()
            self.assertEqual(res.entrypoint_name, "pallas_add")
            # No runtime dependency on helion; the dependency-free Pallas launcher
            # is inlined instead.
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertIn("def default_pallas_launcher(", source)
            self.assertIn("_default_pallas_launcher = default_pallas_launcher", source)
            self.assertIn("def pallas_add(", source)
            # Import the standalone (no helion needed at import) and run on TPU.
            name = "pallas_add_standalone_test"
            mod = _import_standalone(str(out_path), name)
            try:
                out = mod.pallas_add(x, y)
                torch.testing.assert_close(out, x + y)
            finally:
                sys.modules.pop(name, None)

    def test_pallas_dynamic_shapes_run_at_other_shapes(self) -> None:
        """A static_shapes=False Pallas (torch-tensor) kernel precompiled at one
        sample shape runs correctly at OTHER shapes: the standalone inlines the
        real host wrapper, which derives the grid + output shape from the runtime
        input rather than freezing the precompiled sample shape."""
        d = 128
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pallas_rt2.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_rows_times_two,
                    sample_inputs=(
                        torch.zeros(512, d, device=DEVICE, dtype=torch.float32),
                    ),
                    output_path=str(out_path),
                )
            )
            name = "pallas_rt2_test"
            mod = _import_standalone(str(out_path), name)
            try:
                # Shapes other than the T=512 sample it was precompiled with.
                for t in (512, 128, 256):
                    x = torch.randn(t, d, device=DEVICE, dtype=torch.float32)
                    out = mod.pallas_rows_times_two(x)
                    self.assertEqual(tuple(out.shape), (t, d))
                    torch.testing.assert_close(out, x * 2.0)
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
