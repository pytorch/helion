//! Minimal Rust launcher (Chunk E experiment).
//!
//! Exposes ``CompiledLauncher`` — a Python type whose ``__call__``
//! slot dispatches a Triton kernel launch directly into
//! ``compiled_kernel.run`` (the C launcher Triton emits), bypassing
//! the Python ``default_launcher`` frame.
//!
//! Compared to a full Python-side fast launcher, this minimal version
//! trades correctness coverage for simplicity:
//!
//! - No multi-spec cache. The compiled kernel captured on first
//!   ``prime()`` is reused for ALL subsequent calls. Caller is
//!   responsible for only using stable specs (same alignment, same
//!   shape, same dtype).
//! - No knob/hook re-reads. If the user enables
//!   ``knobs.runtime.debug`` or attaches a profiler hook AFTER
//!   priming, we don't observe it.
//! - No ``used_global_vals`` check. Mutating a tracked global between
//!   calls would silently use the stale binary.
//! - No multi-device guard.
//!
//! Purpose: measure the per-call savings of replacing the Python
//! ``default_launcher`` frame with a Rust ``__call__``. The Phase 2
//! launcher's correctness guards aren't in this commit; a production
//! Chunk E would port them.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyclass(
    subclass,
    name = "CompiledLauncher",
    module = "helion._native._launcher"
)]
struct CompiledLauncher {
    // All captured at prime() time and held as strong refs.
    compiled_run: Option<Py<PyAny>>,            // compiled_kernel.run callable
    triton_function: Option<Py<PyAny>>,         // compiled_kernel.function
    packed_metadata: Option<Py<PyAny>>,         // compiled_kernel.packed_metadata
    kernel_launch_metadata: Option<Py<PyAny>>,  // compiled_kernel.launch_metadata
    get_current_stream: Option<Py<PyAny>>,      // driver.active.get_current_stream
    device: Option<Py<PyAny>>,                  // device id from driver
    primed: bool,
}

#[pymethods]
impl CompiledLauncher {
    #[new]
    fn new() -> Self {
        Self {
            compiled_run: None,
            triton_function: None,
            packed_metadata: None,
            kernel_launch_metadata: None,
            get_current_stream: None,
            device: None,
            primed: false,
        }
    }

    /// Run Triton's warmup compile once and capture the
    /// ``CompiledKernel`` references we will keep dispatching
    /// against.
    #[pyo3(signature = (triton_kernel, grid, args, num_warps, num_stages))]
    fn prime<'py>(
        &mut self,
        py: Python<'py>,
        triton_kernel: &Bound<'py, PyAny>,
        grid: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        num_warps: i32,
        num_stages: i32,
    ) -> PyResult<()> {
        // Build warmup kwargs: {grid, warmup=True, num_warps,
        // num_stages, launch_cooperative_grid=False}.
        let warmup_kwargs = PyDict::new(py);
        warmup_kwargs.set_item("grid", grid)?;
        warmup_kwargs.set_item("warmup", true)?;
        warmup_kwargs.set_item("num_warps", num_warps)?;
        warmup_kwargs.set_item("num_stages", num_stages)?;
        warmup_kwargs.set_item("launch_cooperative_grid", false)?;

        // triton_kernel.run(*args, **warmup_kwargs) -> CompiledKernel
        let run_method = triton_kernel.getattr("run")?;
        let compiled = run_method.call(args, Some(&warmup_kwargs))?;
        if compiled.is_none() {
            return Err(PyRuntimeError::new_err("Triton warmup returned None"));
        }

        // Access ``compiled.run`` FIRST so ``_init_handles()`` fires
        // before we read function/packed_metadata.
        let run_fn = compiled.getattr("run")?;
        let function = compiled.getattr("function")?;
        let packed = compiled.getattr("packed_metadata")?;
        let lm = compiled.getattr("launch_metadata")?;

        // driver.active.{get_current_device(), get_current_stream}
        let driver_mod = py.import("triton.runtime.driver")?;
        let driver = driver_mod.getattr("driver")?;
        let active = driver.getattr("active")?;
        let device = active.getattr("get_current_device")?.call0()?;
        let gcs = active.getattr("get_current_stream")?;

        // Publish all state; ``primed`` flag last so a concurrent
        // reader either sees the un-primed state or a fully-published
        // primed one.
        self.compiled_run = Some(run_fn.unbind());
        self.triton_function = Some(function.unbind());
        self.packed_metadata = Some(packed.unbind());
        self.kernel_launch_metadata = Some(lm.unbind());
        self.get_current_stream = Some(gcs.unbind());
        self.device = Some(device.unbind());
        self.primed = true;
        Ok(())
    }

    /// ``__call__`` dispatch: args is ``(triton_kernel, grid,
    /// *kernel_args)``; kwargs (``num_warps`` / ``num_stages`` / etc.)
    /// are ignored — the values were baked at prime time.
    ///
    /// Builds the underlying call:
    /// ``compiled_run(grid_0, grid_1, grid_2, stream, function,
    ///                packed_metadata, None, None, None,
    ///                *kernel_args)``
    #[pyo3(signature = (*args, **_kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        _kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if !self.primed {
            return Err(PyRuntimeError::new_err("CompiledLauncher not primed"));
        }
        let n_args = args.len();
        if n_args < 2 {
            return Err(PyTypeError::new_err(
                "launcher requires at least (triton_kernel, grid, *kernel_args)",
            ));
        }
        // args[0] is triton_kernel (unused — we have cached state)
        // args[1] is grid (tuple of ints)
        let grid = args.get_item(1)?;
        let grid_tuple = grid.cast::<PyTuple>()?;
        let grid_n = grid_tuple.len();
        let grid_0: i64 = if grid_n >= 1 { grid_tuple.get_item(0)?.extract()? } else { 1 };
        let grid_1: i64 = if grid_n >= 2 { grid_tuple.get_item(1)?.extract()? } else { 1 };
        let grid_2: i64 = if grid_n >= 3 { grid_tuple.get_item(2)?.extract()? } else { 1 };

        // Compute the CUDA stream for the captured device.
        let device = self.device.as_ref().unwrap().bind(py);
        let gcs = self.get_current_stream.as_ref().unwrap().bind(py);
        let stream = gcs.call1((device,))?;

        // Build (grid_0, grid_1, grid_2, stream, function,
        //        packed_metadata, None, None, None, *kernel_args)
        let n_kernel_args = n_args - 2;
        let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(9 + n_kernel_args);
        out.push(grid_0.into_pyobject(py)?.into_any());
        out.push(grid_1.into_pyobject(py)?.into_any());
        out.push(grid_2.into_pyobject(py)?.into_any());
        out.push(stream);
        out.push(self.triton_function.as_ref().unwrap().bind(py).clone());
        out.push(self.packed_metadata.as_ref().unwrap().bind(py).clone());
        out.push(py.None().into_bound(py));
        out.push(py.None().into_bound(py));
        out.push(py.None().into_bound(py));
        for i in 0..n_kernel_args {
            out.push(args.get_item(2 + i)?);
        }
        let out_tuple = PyTuple::new(py, &out)?;
        let compiled_run = self.compiled_run.as_ref().unwrap().bind(py);
        let result = compiled_run.call1(&out_tuple)?;
        Ok(result.unbind())
    }
}

#[pymodule]
fn _launcher(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledLauncher>()?;
    Ok(())
}
