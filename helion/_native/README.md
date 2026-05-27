# helion._native — optional Rust extension

This directory holds an OPTIONAL compiled extension (PyO3-based,
written in Rust) that accelerates Helion's hot paths. It is not built
by the default Helion install (which uses `hatchling` and ships a
pure-Python wheel).

## What's here

- `Cargo.toml` — Cargo manifest. Targets Python's `abi3-py310` so the
  built `.so` is binary-compatible with any CPython ≥ 3.10.
- `src/lib.rs` — minimal Rust launcher. Exposes `CompiledLauncher`
  (a `#[pyclass]` with a `__call__` method) that dispatches a Triton
  kernel directly into `compiled_kernel.run`, bypassing both the
  Python `default_launcher` frame and Triton's `JITFunction.run`
  pipeline.
- `__init__.py` — Python shim that imports the compiled extension
  if present and exposes `AVAILABLE` + per-symbol fallback sentinels.

The shim also reserves a `tensor_key = None` slot. That slot is
unused by default — an earlier C experiment shipped a `tensor_key`
accelerator, but per-call savings turned out to be within run-to-run
noise for typical workloads (only ~110 ns saved per `tensor_key`
call, ~220 ns end-to-end for a 2-tensor kernel — too small to
distinguish from variance). The slot remains as an extension point
in case a future accelerator can do meaningfully better.

## When to build

Skip the build unless you're chasing per-call launch overhead on
small kernels. The Python fallback works on every supported platform.

## How to build manually

From the repo root, with the active Python env (the build only needs
`cargo` + `rustc`; PyO3's `abi3-py310` feature means no Python headers
are required at link time):

```sh
cd helion/_native && cargo build --release && cd ../..
python -c "
import sysconfig, shutil
ext = sysconfig.get_config_var('EXT_SUFFIX')
src = 'helion/_native/target/release/lib_launcher.so'
dst = f'helion/_native/_launcher{ext}'
shutil.copy(src, dst)
print('Installed', dst)
"
```

On macOS replace `lib_launcher.so` with `lib_launcher.dylib`; on
Windows the build emits `_launcher.dll` directly.

## How to verify

```sh
python -c "from helion import _native; print('AVAILABLE:', _native.AVAILABLE, 'CompiledLauncher:', _native.CompiledLauncher)"
```

When built: `AVAILABLE: True CompiledLauncher: <class 'helion._native._launcher.CompiledLauncher'>`.
When absent: `AVAILABLE: False CompiledLauncher: None`.

## Why Rust (not C)

A previous draft of this stack used a hand-written C extension. The
Rust version is functionally identical and benchmarks the same on
the hot path:

- Both go through the same CPython entry points for tensor metadata
  reads and Triton kernel dispatch. The Python C API is the
  bottleneck, and PyO3 generates code that compiles down to the same
  CPython calls a hand-written C extension would issue.
- PyO3's macro-generated trampolines add negligible overhead (<100 ns
  per `__call__` invocation by our measurements).

Rust buys us:

- Memory safety on error paths — PyO3's `PyResult` / `Bound<'_, T>`
  types prevent the kinds of reference-leak bugs the manual
  `Py_XDECREF` ladder in C is prone to.
- A standard build system (`cargo`) instead of an ad-hoc `gcc`
  one-liner.
- An ergonomic API: the launcher source is roughly half the length
  of the equivalent C and reads closer to the Python it replaces.
