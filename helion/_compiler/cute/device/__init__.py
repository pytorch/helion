"""Device helpers: Python the CuTe DSL traces into generated kernels.

Every module here is imported by generated code (or by the runtime modules
generated code imports) and traced inline, so it follows the device-module
rules ``test/test_cute_device_helper_names.py`` checks:

* importable without CUDA -- it depends on ``cutlass`` only, never on the
  compiler, the launcher or torch;
* every exported helper is one top-level binding (a ``def`` or a
  ``name = ptx_op(...)`` assignment) so the standalone exporter
  (``helion._compiler.cute.standalone``) can copy it by name;
* a module defining a ``@cute.struct`` carries no ``from __future__ import
  annotations`` (it would stringify the struct's field types);
* helper names are part of the rendered source and of the disk-cache key: a
  rename is a render change, a body edit is picked up through the module's
  path in ``helion.runtime.cute.source_dependencies``.

Role: device helper (``../README.md``, "Device helpers and byte-pinned paths").
"""
