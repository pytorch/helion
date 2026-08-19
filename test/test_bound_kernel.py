"""Tests for :meth:`helion.runtime.kernel.BoundKernel.to_code` with
:class:`helion.OutputCodeOptions` -- i.e. emitting dependency-free ("standalone")
output code that runs with no ``helion`` runtime dependency (``torch`` + the
backend DSL only, or ``jax`` alone for ``jax_fn=True``)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from typing import Any
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessPallas
import helion.language as hl

_FREE = helion.OutputCodeOptions(allow_helion_deps=False)
_JAX = helion.OutputCodeOptions(allow_helion_deps=False, jax_fn=True)


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
    """static_shapes=False -> the standalone must run at any shape: the emitted
    host wrapper derives the grid + output shape from the runtime input rather than
    baking the sample shape."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[64, 64], pid_type="persistent_blocked")
)
def get_num_sm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A kernel deliberately named like a launcher helper. The launcher helpers
    live in the private ``_make_helion_runtime`` scope, so this name cannot
    collide with them (the persistent grid still calls the shim's get_num_sm)."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _write(tmp: str, code: str) -> str:
    path = Path(tmp) / "generated.py"
    path.write_text(code)
    return str(path)


def _run_add_no_helion(code: str, entrypoint: str, shape: tuple[int, int]) -> None:
    """Run an **add** standalone in a subprocess where importing ``helion`` is
    blocked, proving zero Helion runtime dependency and the right result. Specific
    to add kernels -- it hardcodes the ``x + y`` reference."""
    m, n = shape
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, code)
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
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        if "STANDALONE_OK" not in result.stdout:
            raise AssertionError(
                f"standalone run failed:\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


def _import_code(code: str, name: str, tmp: str) -> Any:
    """Import generated standalone source as a module (registered in
    ``sys.modules`` before ``exec_module`` so any inlined ``@dataclass`` can
    resolve ``cls.__module__``)."""
    import importlib.util

    path = _write(tmp, code)
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


@onlyBackends(["triton"])
@skipIfRefEager("to_code compiles real Triton code; not meaningful in ref-eager")
class TestToCodeTriton(TestCase):
    def test_default_still_has_helion_deps(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        # options=None (default) keeps the original to_code behavior, which still
        # depends on helion at runtime (here: the launcher import).
        code = add.bind((x, y)).to_code()
        self.assertIn("from helion.runtime import default_launcher", code)

    def test_allow_helion_deps_false_has_no_helion_import(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        code = add.bind((x, y)).to_code(options=_FREE)
        # The helion *package* is never imported. The body DOES contain
        # ``helion.runtime.*`` references, but they resolve against the inlined
        # local ``helion`` shim (see the shim banner), not the real package.
        self.assertNotIn("import helion", code)
        self.assertNotIn("from helion", code)
        self.assertIn("def default_launcher(", code)
        self.assertIn(
            "helion = types.SimpleNamespace(runtime=_make_helion_runtime())", code
        )
        self.assertIn("_default_launcher = helion.runtime.default_launcher", code)
        self.assertIn("def add(", code)

    def test_standalone_runs_without_helion(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add.bind((x, y)).to_code(options=_FREE)
        _run_add_no_helion(code, "add", (128, 128))

    def test_to_triton_code_alias_unchanged(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        bound = add.bind((x, y))
        # Back-compat: to_triton_code is an alias for to_code's default behavior.
        self.assertEqual(bound.to_triton_code(), bound.to_code())

    def test_persistent_kernel_get_num_sm_via_shim(self) -> None:
        """A persistent kernel calls ``helion.runtime.get_num_sm`` to size its
        grid. The body is emitted verbatim and the call is satisfied by the inlined
        ``helion.runtime`` shim, so it runs with no real helion."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_persistent.bind((x, y)).to_code(options=_FREE)
        self.assertNotIn("import helion", code)
        self.assertIn("helion.runtime.get_num_sm(", code)  # verbatim body call
        self.assertIn("get_num_sm=get_num_sm", code)  # shim re-export
        _run_add_no_helion(code, "add_persistent", (128, 128))

    def test_tma_kernel_set_triton_allocator_via_shim(self) -> None:
        """A tensor-descriptor (TMA) kernel calls
        ``helion.runtime.set_triton_allocator``; the shim satisfies it."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_tma.bind((x, y)).to_code(options=_FREE)
        self.assertNotIn("import helion", code)
        self.assertIn("helion.runtime.set_triton_allocator(", code)  # verbatim call
        self.assertIn("set_triton_allocator=set_triton_allocator", code)
        _run_add_no_helion(code, "add_tma", (128, 128))

    def test_dynamic_shapes_run_at_other_shapes(self) -> None:
        """A static_shapes=False kernel's standalone runs correctly at shapes other
        than the one it was bound with (grid + output shape derived from input)."""
        x = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_dynamic.bind((x, y)).to_code(options=_FREE)
        for shape in ((128, 128), (256, 64), (64, 256)):
            _run_add_no_helion(code, "add_dynamic", shape)

    def test_kernel_named_like_runtime_helper(self) -> None:
        """A kernel literally named ``get_num_sm`` exports safely: the launcher
        helpers are isolated in a private scope, so the kernel name can't shadow
        them (this would previously have produced two module-level defs)."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = get_num_sm.bind((x, y)).to_code(options=_FREE)
        _run_add_no_helion(code, "get_num_sm", (128, 128))


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
    """static_shapes=False Pallas (torch-tensor): the standalone must run at any
    leading dim (grid + output shape derived from the runtime input)."""
    n, _d = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile, :] = x[tile, :] * 2.0
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_cast_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 in, f32 out -- the ``.to(float32)`` casts emit ``lax.*`` in the device
    body, so the jax_fn module must import ``jax.lax``."""
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
    """Launch args beyond the user inputs: ``hl.specialize(k)`` passes the
    reduction-dim size as a non-tensor int, and the module scalar ``_JAX_FN_FILL``
    is lifted to a ``(1,)`` constant tensor. jax_fn export must bake both in."""
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
    """static_shapes=False: the jax_fn module must run at any leading (row) dim."""
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
    materialize the row count as a scalar launch arg (the tile mask ``indices <
    t``); jax_fn export must derive that scalar from the runtime input too, else
    the mask is wrong at other shapes. (T, k) -> (T,)."""
    t, _k = x.shape
    out = torch.empty([t], dtype=torch.float32, device=x.device)
    for tile in hl.tile(t):
        row = x[tile, :]
        out[tile] = torch.sum(
            torch.exp(row - torch.amax(row, dim=-1, keepdim=True)), dim=-1
        )
    return out


def _pallas_to_code(
    kernel: Any, args: tuple[object, ...], options: helion.OutputCodeOptions
) -> str:
    """``to_code`` with an explicit config. These kernels are never run/autotuned in
    the test, so ``to_code(config=None)`` has no implicit config to resolve; use the
    kernel's own config when it declares one, else the backend default."""
    bound = kernel.bind(args)
    configs = bound.kernel.configs
    config = configs[0] if len(configs) == 1 else bound.config_spec.default_config()
    return bound.to_code(config, options=options)


@skipUnlessPallas("Pallas to_code test requires the Pallas backend / TPU")
@skipIfRefEager("to_code compiles real kernels; not meaningful in ref-eager")
class TestToCodePallas(TestCase):
    def test_pallas_torch_standalone_runs(self) -> None:
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_add, (x, y), _FREE)
        # The helion package is never imported; the dependency-free Pallas launcher
        # is inlined into a local ``helion.runtime`` shim instead.
        self.assertNotIn("import helion", code)
        self.assertNotIn("from helion", code)
        self.assertIn("def default_pallas_launcher(", code)
        self.assertIn(
            "_default_pallas_launcher = helion.runtime.default_pallas_launcher", code
        )
        self.assertIn("def pallas_add(", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_add_standalone_test"
            mod = _import_code(code, name, tmp)
            try:
                torch.testing.assert_close(mod.pallas_add(x, y), x + y)
            finally:
                sys.modules.pop(name, None)

    def test_pallas_torch_dynamic_shapes(self) -> None:
        """A static_shapes=False Pallas (torch) standalone runs at other shapes."""
        d = 128
        x0 = torch.zeros(512, d, device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_rows_times_two, (x0,), _FREE)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_rt2_test"
            mod = _import_code(code, name, tmp)
            try:
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
        code = _pallas_to_code(pallas_add, (x, y), _JAX)
        # jax_fn module is jax-only: no helion AND no torch.
        self.assertNotIn("import helion", code)
        self.assertNotIn("from helion", code)
        self.assertNotIn("import torch", code)
        self.assertIn("import jax", code)
        # Reuses the real pl.kernel compile core via the inlined _pallas_jax_call.
        self.assertIn("def _pallas_jax_call(", code)
        self.assertIn("pl.kernel(", code)
        self.assertIn("def pallas_add(", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_add_jax_test"
            mod = _import_code(code, name, tmp)
            try:
                xj = jnp.asarray(x.detach().float().cpu().numpy())
                yj = jnp.asarray(y.detach().float().cpu().numpy())
                out = mod.pallas_add(xj, yj)
                np.testing.assert_allclose(
                    np.asarray(out), (x + y).cpu().numpy(), rtol=1e-5, atol=1e-5
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_cast_imports_lax(self) -> None:
        """A dtype-casting kernel emits ``lax.*`` in the device body; the jax_fn
        module must import ``jax.lax`` (else NameError)."""
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = _pallas_to_code(pallas_cast_add, (x, y), _JAX)
        self.assertNotIn("import helion", code)
        self.assertNotIn("import torch", code)
        self.assertIn("import jax.lax as lax", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_cast_jax_test"
            mod = _import_code(code, name, tmp)
            try:
                xj = jnp.asarray(x.float().cpu().numpy()).astype(jnp.bfloat16)
                yj = jnp.asarray(y.float().cpu().numpy()).astype(jnp.bfloat16)
                out = mod.pallas_cast_add(xj, yj)
                expected = (x.float() + y.float()).cpu().numpy()
                np.testing.assert_allclose(
                    np.asarray(out), expected, rtol=1e-3, atol=1e-3
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_multi_output(self) -> None:
        """Two-output kernel: exercises the tuple launcher return / multi-out."""
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_add_sub, (x, y), _JAX)
        self.assertNotIn("import helion", code)
        self.assertNotIn("import torch", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_add_sub_jax_test"
            mod = _import_code(code, name, tmp)
            try:
                xj = jnp.asarray(x.cpu().numpy())
                yj = jnp.asarray(y.cpu().numpy())
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
        """A kernel whose host wrapper passes launch args beyond the user inputs (a
        lifted module-scalar constant tensor + a non-tensor specialization int):
        both are baked into the jax_fn module, whose entrypoint still takes only the
        user tensor inputs."""
        import jax.numpy as jnp
        import numpy as np

        t_dim, k_dim = 32, 16
        x = torch.randn([t_dim, k_dim], device=DEVICE, dtype=torch.float32)
        thr = torch.zeros([t_dim], device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_masked_row_sum, (x, thr), _JAX)
        self.assertNotIn("import helion", code)
        self.assertNotIn("import torch", code)
        self.assertIn("constants baked in from the original host wrapper", code)
        self.assertIn("jnp.array(", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "masked_row_sum_jax_test"
            mod = _import_code(code, name, tmp)
            try:
                xj = jnp.asarray(x.detach().cpu().numpy())
                thrj = jnp.asarray(thr.detach().cpu().numpy())
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
        """A static_shapes=False kernel's jax_fn module derives the grid + output
        shapes from the runtime input, so it runs at shapes other than the sample."""
        import jax.numpy as jnp
        import numpy as np

        d = 128
        x0 = torch.zeros(512, d, device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_dynamic_rows, (x0,), _JAX)
        self.assertNotIn("import helion", code)
        self.assertNotIn("import torch", code)
        self.assertIn("inputs[0].shape[0]", code)  # derived, not baked
        with tempfile.TemporaryDirectory() as tmp:
            name = "dynamic_rows_jax_test"
            mod = _import_code(code, name, tmp)
            try:
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
        """A static_shapes=False *reduction* jax_fn module must derive the scalar
        row-count launch arg from the runtime input too -- checked by running at
        shapes other than the sample and comparing values (a baked row count would
        be wrong even where the output shape looks right)."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        k = 64
        x0 = torch.zeros(128, k, device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_dynamic_row_sum, (x0,), _JAX)
        self.assertNotIn("import helion", code)
        self.assertNotIn("import torch", code)
        self.assertIn("jnp.array([inputs[0].shape[0]]", code)  # derived, not baked
        self.assertNotIn("jnp.array([128]", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "dynamic_row_sum_jax_test"
            mod = _import_code(code, name, tmp)
            try:
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

    def test_jax_fn_with_helion_deps_imports_launcher(self) -> None:
        """``jax_fn`` is orthogonal to ``allow_helion_deps``: with deps allowed the
        entrypoint still operates on ``jax.Array``s, but imports the launch core from
        helion instead of inlining the jax-only slice."""
        import jax.numpy as jnp
        import numpy as np

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(
            pallas_add,
            (x, y),
            helion.OutputCodeOptions(jax_fn=True, allow_helion_deps=True),
        )
        # Launch core imported from helion, not inlined; still jax-array (no torch).
        self.assertIn(
            "from helion.runtime.pallas.launcher import _pallas_jax_call", code
        )
        self.assertNotIn("def _pallas_jax_call(", code)
        self.assertNotIn("import torch", code)
        self.assertIn("def pallas_add(", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_add_jax_deps_test"
            mod = _import_code(code, name, tmp)
            try:
                xj = jnp.asarray(x.detach().float().cpu().numpy())
                yj = jnp.asarray(y.detach().float().cpu().numpy())
                out = mod.pallas_add(xj, yj)
                np.testing.assert_allclose(
                    np.asarray(out), (x + y).cpu().numpy(), rtol=1e-5, atol=1e-5
                )
            finally:
                sys.modules.pop(name, None)

    def test_pallas_jax_fn_unsupported_feature_raises(self) -> None:
        """A dot_general (matmul) kernel is gated: jax_fn export must raise a clear
        ``NotImplementedError`` rather than emit a silently-wrong module."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            _pallas_to_code(pallas_matmul, (x, y), _JAX)


if __name__ == "__main__":
    unittest.main()
