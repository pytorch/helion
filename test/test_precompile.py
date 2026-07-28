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


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_cast_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 in, f32 out -- the ``.to(float32)`` casts emit ``lax.*`` in the
    device body, so the standalone must import ``jax.lax``."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile].to(torch.float32) + y[tile].to(torch.float32)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add_sub(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two outputs -- exercises the multi-output launcher return / unpack."""
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        a[tile] = x[tile] + y[tile]
        b[tile] = x[tile] - y[tile]
    return a, b


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Uses dot_general -- a Pallas feature the jax_fn path gates out."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += x[tile_m, tile_k] @ y[tile_k, tile_n]
        out[tile_m, tile_n] = acc
    return out


_JAX_FN_FILL = -1e30  # module scalar -> host wrapper lifts it to a (1,) const tensor


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_masked_row_sum(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """Minimal repro of launch args beyond the user inputs. ``hl.specialize(k)``
    passes the reduction-dim size as a plain int (a non-tensor launch arg), and
    the module scalar ``_JAX_FN_FILL`` is lifted to a ``(1,)`` constant tensor --
    both created by the host wrapper, not user inputs. jax_fn precompile must bake
    both into the standalone rather than reject them."""
    t, k = x.shape
    k = hl.specialize(k)
    out = torch.empty([t], dtype=torch.float32, device=x.device)
    for tile_t in hl.tile(t):
        row = x[tile_t, :]
        keep = row > thr[tile_t][:, None]
        masked = torch.where(keep, row, _JAX_FN_FILL)
        z = torch.zeros_like(masked)
        masked = torch.where(masked > 0.0, masked, z)
        out[tile_t] = torch.sum(masked, dim=-1)
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[8]), static_shapes=False, backend="pallas"
)
def pallas_dynamic_rows(x: torch.Tensor) -> torch.Tensor:
    """static_shapes=False: the device kernel and the jax_fn standalone must run at
    any leading (row) dim. The output leading dim equals the input's, so a wrongly
    baked (sample) output shape is obvious."""
    n, _d = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile, :] = x[tile, :] * 2.0
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[128]), static_shapes=False, backend="pallas"
)
def pallas_dynamic_row_sum(x: torch.Tensor) -> torch.Tensor:
    """static_shapes=False *reduction*: a per-row reduction makes the host wrapper
    materialize the row count as a scalar launch arg that the tile mask reads
    (``indices < t``) -- on top of the grid and output shape. jax_fn precompile
    must derive that scalar from the runtime input too; baking the sample row count
    masks off / over-runs rows at every other shape, so the result is wrong even
    where the output shape happens to look right. (T, k) -> (T,)."""
    t, _k = x.shape
    out = torch.empty([t], dtype=torch.float32, device=x.device)
    for tile in hl.tile(t):
        row = x[tile, :]
        out[tile] = torch.sum(
            torch.exp(row - torch.amax(row, dim=-1, keepdim=True)), dim=-1
        )
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

    def test_pallas_jax_fn_standalone_runs(self) -> None:
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pallas_add_jax.py"
            res = helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertEqual(res.entrypoint_name, "pallas_add")
            # jax_fn standalone is jax-only: no helion AND no torch.
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertNotIn("import torch", source)
            self.assertIn("import jax", source)
            # Reuses the real pl.kernel compile core via the inlined shared
            # _pallas_jax_call (not a hand-rolled pl.pallas_call).
            self.assertIn("def _pallas_jax_call(", source)
            self.assertIn("pl.kernel(", source)
            self.assertIn("_pallas_jax_call(", source)
            self.assertIn("def pallas_add(", source)
            # Run the standalone on JAX arrays (on TPU) and compare.
            xj = jnp.asarray(x.detach().float().cpu().numpy())
            yj = jnp.asarray(y.detach().float().cpu().numpy())
            name = "pallas_add_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                out = mod.pallas_add(xj, yj)
                expected = (x + y).cpu().numpy()
                np.testing.assert_allclose(
                    np.asarray(out), expected, rtol=1e-5, atol=1e-5
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_cast_imports_lax(self) -> None:
        """A dtype-casting kernel emits ``lax.*`` in the device body; the
        standalone must import ``jax.lax`` (else it crashes with NameError).
        Regression test for the jax_fn path dropping the generated imports."""
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pallas_cast_jax.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_cast_add,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertNotIn("import torch", source)
            self.assertIn("import jax.lax as lax", source)
            xj = jnp.asarray(x.float().cpu().numpy()).astype(jnp.bfloat16)
            yj = jnp.asarray(y.float().cpu().numpy()).astype(jnp.bfloat16)
            name = "pallas_cast_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                out = mod.pallas_cast_add(xj, yj)
                expected = (x.float() + y.float()).cpu().numpy()
                np.testing.assert_allclose(
                    np.asarray(out), expected, rtol=1e-3, atol=1e-3
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_multi_output(self) -> None:
        """A two-output kernel: exercises the tuple launcher return (the host
        wrapper unpacks ``a, b = _launcher(...)``) and multi-out pallas_call."""
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pallas_add_sub_jax.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_add_sub,
                    sample_inputs=(x, y),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("import torch", source)
            xj = jnp.asarray(x.cpu().numpy())
            yj = jnp.asarray(y.cpu().numpy())
            name = "pallas_add_sub_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                a, b = mod.pallas_add_sub(xj, yj)
                np.testing.assert_allclose(
                    np.asarray(a), (x + y).cpu().numpy(), rtol=1e-5, atol=1e-5
                )
                np.testing.assert_allclose(
                    np.asarray(b), (x - y).cpu().numpy(), rtol=1e-5, atol=1e-5
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_const_and_nontensor_args(self) -> None:
        """jax_fn precompile of a kernel whose host wrapper passes launch args
        beyond the user inputs: a lifted module-scalar constant tensor and a
        non-tensor specialization int. Both are baked into the standalone, whose
        entrypoint still takes only the user tensor inputs, and the result matches
        eager."""
        import jax.numpy as jnp
        import numpy as np

        t_dim, k_dim = 32, 16
        x = torch.randn([t_dim, k_dim], device=DEVICE, dtype=torch.float32)
        thr = torch.zeros([t_dim], device=DEVICE, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "masked_row_sum_jax.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_masked_row_sum,
                    sample_inputs=(x, thr),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("from helion", source)
            self.assertNotIn("import torch", source)
            # The host-wrapper constants (lifted (1,) tensor + specialization int)
            # are baked into the standalone, not taken as entrypoint inputs.
            self.assertIn("constants baked in from the original host wrapper", source)
            self.assertIn("jnp.array(", source)
            xj = jnp.asarray(x.detach().cpu().numpy())
            thrj = jnp.asarray(thr.detach().cpu().numpy())
            name = "masked_row_sum_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                out = mod.pallas_masked_row_sum(xj, thrj)
                m = torch.where(x > thr[:, None], x, torch.full_like(x, _JAX_FN_FILL))
                m = torch.where(m > 0.0, m, torch.zeros_like(m))
                expected = m.sum(dim=-1).cpu().numpy()
                np.testing.assert_allclose(
                    np.asarray(out), expected, rtol=1e-4, atol=1e-4
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_dynamic_shapes(self) -> None:
        """A static_shapes=False kernel precompiled at ONE sample shape must run
        correctly at OTHER shapes: the jax_fn standalone derives the grid and
        output shapes from the runtime input instead of baking the sample (else it
        returns the sample shape for every input)."""
        import jax.numpy as jnp
        import numpy as np

        d = 128
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "dynamic_rows_jax.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_dynamic_rows,
                    sample_inputs=(
                        torch.zeros(512, d, device=DEVICE, dtype=torch.float32),
                    ),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("import torch", source)
            # Grid + output shape are derived from the runtime input, not baked.
            self.assertIn("inputs[0].shape[0]", source)
            name = "dynamic_rows_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                # Run at shapes other than the T=512 sample it was precompiled with.
                for t in (512, 128, 256):
                    xj = jnp.arange(t * d, dtype=jnp.float32).reshape(t, d)
                    out = mod.pallas_dynamic_rows(xj)
                    self.assertEqual(tuple(out.shape), (t, d))
                    np.testing.assert_allclose(
                        np.asarray(out), np.asarray(xj) * 2.0, rtol=1e-6, atol=1e-6
                    )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_dynamic_shapes_reduction(self) -> None:
        """A static_shapes=False *reduction* precompiled at ONE shape must run
        correctly at OTHER shapes. Beyond the grid + output shape, a reduction adds
        a scalar row-count launch arg (the tile mask ``indices < t``); the jax_fn
        standalone must derive that scalar from the runtime input too. Baking the
        sample row count leaves the mask wrong at every other shape -- so this
        checks values, not just the output shape."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        k = 64
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "dynamic_row_sum_jax.py"
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_dynamic_row_sum,
                    sample_inputs=(
                        torch.zeros(128, k, device=DEVICE, dtype=torch.float32),
                    ),
                    output_path=str(out_path),
                    jax_fn=True,
                )
            )
            source = out_path.read_text()
            self.assertNotIn("import helion", source)
            self.assertNotIn("import torch", source)
            # The scalar row count is derived from the input, not baked to 128.
            self.assertIn("jnp.array([inputs[0].shape[0]]", source)
            self.assertNotIn("jnp.array([128]", source)
            name = "dynamic_row_sum_jax_test"
            mod = _import_standalone(str(out_path), name)
            try:
                # Run at shapes other than the T=128 sample (all multiples of the
                # block); a baked row count would give wrong values here.
                for t in (128, 256, 384):
                    xj = jax.random.normal(jax.random.PRNGKey(t), (t, k), jnp.float32)
                    out = mod.pallas_dynamic_row_sum(xj)
                    ref = jnp.sum(jnp.exp(xj - jnp.max(xj, -1, keepdims=True)), -1)
                    self.assertEqual(tuple(out.shape), (t,))
                    np.testing.assert_allclose(
                        np.asarray(out), np.asarray(ref), rtol=1e-2, atol=1e-2
                    )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_unsupported_feature_raises(self) -> None:
        """A dot_general (matmul) kernel is gated: jax_fn precompile must raise a
        clear ``NotImplementedError`` rather than emit a silently-wrong standalone."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaises(NotImplementedError),
        ):
            helion.precompile(
                helion.PrecompilationInput(
                    kernel=pallas_matmul,
                    sample_inputs=(x, y),
                    output_path=str(Path(tmp) / "nope.py"),
                    jax_fn=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
