# helion._C — optional native extensions

This directory holds OPTIONAL compiled extensions that accelerate
Helion's hot paths. They are not built by the default Helion install
(which uses `hatchling` and ships a pure-Python wheel).

## What's here

- `_native.c` — single-file C extension. Currently exports
  `tensor_key(tensor, static_indices) -> tuple | None`, the
  static-shapes specialization key built per-call by `Kernel.bind`.
- `_launcher.c` — minimal C launcher. Exposes
  `CompiledLauncher` (a Python type with a `tp_call` slot) that
  dispatches a Triton kernel directly into `compiled_kernel.run`,
  bypassing both the Python `default_launcher` frame and Triton's
  `JITFunction.run` pipeline.
- `__init__.py` — Python shim that imports the compiled extensions if
  present and exposes `AVAILABLE` + per-symbol fallback sentinels.

## When to build

Skip the build unless you're chasing per-call launch overhead on
small kernels. The Python fallback produces identical keys and works
on every supported platform.

## How to build manually

From the repo root, with the active Python env:

```sh
python -c "
import subprocess, sysconfig
inc = sysconfig.get_path('include')
ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
for stem in ('_native', '_launcher'):
    out = f'helion/_C/{stem}{ext_suffix}'
    subprocess.check_call([
        'gcc', '-shared', '-fPIC',
        '-O3',
        f'-I{inc}',
        f'helion/_C/{stem}.c',
        '-o', out,
    ])
    print('Built', out)
"
```

If the build succeeds, `import helion._C` will pick up the extension
and `_C.AVAILABLE` will become `True`. If it fails or you don't run
the command, `_C.AVAILABLE` stays `False` and the Python path handles
everything.

## What it does NOT do

- It does not provide a C launcher — that's a much larger piece
  (see `agent_space/c_extension_chunking_plan.md` Chunk E).
- It does not touch PyTorch's C++ ABI. It uses only the Python C-API
  (`PyObject_GetAttrString`, `PyObject_CallMethod`, `PyTuple_*`), so
  it builds against any PyTorch.

## How to verify

```sh
python -c "from helion import _C; print('AVAILABLE:', _C.AVAILABLE, 'tensor_key:', _C.tensor_key)"
```

When built: prints `AVAILABLE: True tensor_key: <built-in function tensor_key>`.
When absent: prints `AVAILABLE: False tensor_key: None`.
