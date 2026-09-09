"""Tests for launching Pallas kernels while ``jax_enable_x64`` is set."""

from __future__ import annotations

import unittest

jax_export = None
pipeline = None
HAS_PL_KERNEL = False

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if HAS_JAX:
    HAS_PL_KERNEL = callable(getattr(pl, "kernel", None))
    try:
        from jax import export as jax_export
    except ImportError:
        jax_export = None  # type: ignore[assignment]
    try:
        from jax._src.pallas.mosaic import pipeline
    except ImportError:
        pipeline = None  # type: ignore[assignment]

    from helion.runtime.pallas.launcher import _ensure_cpu_tpu_info
    from helion.runtime.pallas.launcher import _pallas_kernel_scratch_kwarg
    from helion.runtime.pallas.launcher import _x64_disabled_scope
    from helion.runtime.pallas.launcher import _x64_scoped_jit_fn

ROWS = 30
WINDOW = 8


@unittest.skipUnless(HAS_JAX, "requires jax")
class TestPallasX64Launch(unittest.TestCase):
    def setUp(self) -> None:
        self._x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        _ensure_cpu_tpu_info()
        # Older JAX releases query this helper directly during pipeline tracing.
        self._tpu_gen = getattr(pipeline, "_get_tpu_generation", None)
        if self._tpu_gen is not None:
            pipeline._get_tpu_generation = lambda: 6

    def tearDown(self) -> None:
        if self._tpu_gen is not None:
            pipeline._get_tpu_generation = self._tpu_gen
        jax.config.update("jax_enable_x64", self._x64)

    def test_literals_canonicalize_to_int32_inside_the_scope(self) -> None:
        wrapped = _x64_scoped_jit_fn(lambda: jnp.asarray(0).dtype)
        self.assertEqual(wrapped(), jnp.int32)
        self.assertEqual(jnp.asarray(0).dtype, jnp.int64)

    def test_works_with_x64_already_off(self) -> None:
        jax.config.update("jax_enable_x64", False)
        wrapped = _x64_scoped_jit_fn(lambda: jnp.asarray(0).dtype)
        self.assertEqual(wrapped(), jnp.int32)

    def test_scope_is_available_on_this_jax(self) -> None:
        with _x64_disabled_scope():
            self.assertEqual(jnp.asarray(0).dtype, jnp.int32)

    def test_rejects_64bit_integer_inputs(self) -> None:
        wrapped = _x64_scoped_jit_fn(lambda ids: ids)
        with self.assertRaisesRegex(RuntimeError, "64-bit"):
            wrapped(jnp.arange(4, dtype=jnp.int64))

    def test_dynamic_grid_pipeline_traces(self) -> None:
        out = jax.eval_shape(_x64_scoped_jit_fn(_dynamic_grid_kernel), *_kernel_args())
        self.assertEqual(out.shape, (ROWS, 128))

    def test_scope_holds_under_jit(self) -> None:
        wrapped = _x64_scoped_jit_fn(_dynamic_grid_kernel)
        out = jax.eval_shape(jax.jit(wrapped), *_kernel_args())
        self.assertEqual(out.shape, (ROWS, 128))

    @unittest.skipUnless(HAS_PL_KERNEL, "requires pl.kernel")
    @unittest.skipIf(jax_export is None, "requires jax.export")
    def test_nested_pl_kernel_dynamic_grid_exports_for_tpu(self) -> None:
        wrapped = jax.jit(_x64_scoped_jit_fn(_nested_dynamic_grid_kernel))
        exported = jax_export.export(wrapped, platforms=("tpu",))(
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((ROWS, 128), jnp.float32),
        )
        self.assertIn("stablehlo.custom_call", exported.mlir_module())
        self.assertTrue(jax.config.jax_enable_x64)


def _kernel_args() -> tuple[jax.Array, jax.Array]:
    return jnp.array([3], jnp.int32), jnp.zeros((ROWS, 128), jnp.float32)


def _dynamic_grid_kernel(num_work: jax.Array, x: jax.Array) -> jax.Array:
    def body(x_ref: object, o_ref: object) -> None:
        o_ref[...] = x_ref[...] + 1.0

    spec = pl.BlockSpec(
        (pl.BoundedSlice(WINDOW), 128),
        lambda i: (pl.ds(i * WINDOW, WINDOW), 0),
    )

    def kernel(nw_ref: object, x_hbm: object, o_hbm: object) -> None:
        pltpu.emit_pipeline(
            body, grid=(nw_ref[0],), in_specs=[spec], out_specs=[spec]
        )(x_hbm, o_hbm)

    return pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(num_work, x)


def _nested_dynamic_grid_kernel(num_work: jax.Array, x: jax.Array) -> jax.Array:
    """Mirror compact-worklist's explicit ANY-to-SMEM grid-bound path."""
    num_work_buffer = jnp.reshape(num_work, (1,))
    spec = pl.BlockSpec(
        (pl.BoundedSlice(WINDOW), 128),
        lambda i: (pl.ds(i * WINDOW, WINDOW), 0),
    )

    def body(x_ref: object, o_ref: object) -> None:
        o_ref[...] = x_ref[...] + 1.0

    def kernel(
        num_work_hbm: object,
        x_hbm: object,
        o_hbm: object,
        num_work_smem: object,
    ) -> None:
        pltpu.sync_copy(num_work_hbm, num_work_smem)
        pltpu.emit_pipeline(
            body,
            grid=(num_work_smem[0],),
            in_specs=[spec],
            out_specs=[spec],
        )(x_hbm, o_hbm)

    scratch_kwarg = _pallas_kernel_scratch_kwarg(pl)
    call = pl.kernel(
        kernel,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        mesh=pltpu.create_tensorcore_mesh("x64_test", num_cores=1),
        **{scratch_kwarg: [pltpu.SMEM((1,), jnp.int32)]},
    )
    return call(num_work_buffer, x)
