// Helion C extension: bind cache + CompiledLauncher.
//
// Two primary consumers:
//
//   1. ``helion._C.tensor_key(tensor)`` — drop-in C replacement for the
//      hot ``_tensor_key`` Python function called once per tensor arg per
//      ``Kernel.__call__`` from inside ``_base_specialization_key``.
//      Returns the same shape of Python tuple, but avoids a fistful of
//      Python frames per invocation. Plumbed in behind ``HELION_C_BIND``
//      (default on when the .so is present).
//
//   2. ``helion._C.CompiledLauncher`` — a Python type whose instances
//      hold ``(compiled_run, function, packed_metadata, hooks, device,
//      stream getter, binder, run_kwargs)`` and expose a single
//      ``launch(grid_0, grid_1, grid_2, *args)`` Python method that
//      issues the kernel through Triton's own C launcher with as little
//      Python work as possible. Used by :class:`helion.runtime._FastLauncher`
//      to dispatch the hot path without the per-call Python frame.
//
// The extension is loaded lazily; if any part of import fails (e.g. on
// MTIA / debug builds), the pure-Python ``_FastLauncher`` continues to
// work as a fallback.

// Limited API is convenient but we intentionally don't enable it because
// CPython 3.10's limited API doesn't expose tuple internals we'd want
// for the bind cache fast path. Use the regular CPython API.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

// -----------------------------------------------------------------------------
// Bind cache helpers
// -----------------------------------------------------------------------------
//
// ``tensor_key(tensor)`` mirrors the static-shape branch of
// ``helion.runtime.kernel._tensor_key`` exactly:
//   ``(dtype, hashable_dims(size), hashable_dims(stride), static_indices)``.
//
// ``hashable_dims`` for non-SymInt ints is just the int itself (we leave
// the SymInt branch in Python because it touches torch.SymInt internals
// that aren't trivially reachable from C). Likewise the bucketed
// (non-static_shapes) branch and the int64 indexing probe stay in
// Python — they are taken on a far colder path.

// Cached references resolved on first use. We hold strong refs that
// outlive the module. Lifetime matches the interpreter — there is no
// module finalizer (``m_free``) registered, so these are intentionally
// leaked at process shutdown.
static PyObject *g_torch_Tensor = nullptr;
static PyObject *g_empty_frozenset = nullptr;
static PyObject *g_static_indices_attr = nullptr;
static PyObject *g_frozenset_type = nullptr;

static int resolve_torch_tensor() {
  if (g_torch_Tensor != nullptr) {
    return 0;
  }
  PyObject *torch_mod = PyImport_ImportModule("torch");
  if (torch_mod == nullptr) {
    return -1;
  }
  PyObject *tensor_cls = PyObject_GetAttrString(torch_mod, "Tensor");
  Py_DECREF(torch_mod);
  if (tensor_cls == nullptr) {
    return -1;
  }
  g_torch_Tensor = tensor_cls; // keep strong ref
  return 0;
}

static int ensure_static_cache() {
  if (g_empty_frozenset == nullptr) {
    g_empty_frozenset = PyFrozenSet_New(nullptr);
    if (g_empty_frozenset == nullptr) {
      return -1;
    }
  }
  if (g_static_indices_attr == nullptr) {
    g_static_indices_attr = PyUnicode_InternFromString("_dynamo_static_indices");
    if (g_static_indices_attr == nullptr) {
      return -1;
    }
  }
  if (g_frozenset_type == nullptr) {
    g_frozenset_type = reinterpret_cast<PyObject *>(&PyFrozenSet_Type);
    Py_INCREF(g_frozenset_type);
  }
  return 0;
}

// Build the Python tuple ``hashable_dims(dims)``. For non-SymInt
// integral dims this is just a tuple of those ints. If a SymInt is
// encountered we bail out to ``nullptr`` and the caller falls back to
// the Python implementation (which knows how to hash the SymInt).
static PyObject *hashable_dims_from_sizes(const int64_t *dims, Py_ssize_t n) {
  PyObject *out = PyTuple_New(n);
  if (out == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *v = PyLong_FromLongLong(static_cast<long long>(dims[i]));
    if (v == nullptr) {
      Py_DECREF(out);
      return nullptr;
    }
    PyTuple_SET_ITEM(out, i, v); // steals ref
  }
  return out;
}

// Public ``helion._C.tensor_key(tensor) -> tuple | None``.
//
// Always uses the static_shapes branch — that's the only one that
// matters for the hot path (``examples/add.py``). The dynamic-shape
// and bucketed branches stay in Python and are taken on far colder
// paths via the ``_tensor_key`` Python fallback. Returns ``None`` to
// tell the caller to take the Python path (e.g. SymInt sizes, or any
// input that isn't a ``torch.Tensor``).
static PyObject *helion_c_tensor_key(PyObject *self, PyObject *const *args,
                                     Py_ssize_t nargs, PyObject *kwnames) {
  if (resolve_torch_tensor() < 0) {
    return nullptr;
  }
  if (ensure_static_cache() < 0) {
    return nullptr;
  }
  if (nargs != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "tensor_key() requires exactly one positional argument");
    return nullptr;
  }
  PyObject *tensor = args[0];
  // Quick type check via fast path. ``PyObject_IsInstance`` is cheap and
  // accepts subclasses (e.g. torch.nn.Parameter, FakeTensor).
  int is_tensor = PyObject_IsInstance(tensor, g_torch_Tensor);
  if (is_tensor < 0) {
    return nullptr;
  }
  if (is_tensor == 0) {
    Py_RETURN_NONE; // signal caller to take the Python path
  }

  // Pull dtype as an opaque object (compared via ``is`` / ``==``).
  PyObject *dtype = PyObject_GetAttrString(tensor, "dtype");
  if (dtype == nullptr) {
    return nullptr;
  }

  // Sizes + strides as Python tuples of ints. For most tensors these
  // are 1–4 element tuples; we go through the public ``.size()`` /
  // ``.stride()`` methods to avoid touching at::Tensor internals.
  PyObject *size_seq = PyObject_CallMethod(tensor, "size", nullptr);
  if (size_seq == nullptr) {
    Py_DECREF(dtype);
    return nullptr;
  }
  PyObject *stride_seq = PyObject_CallMethod(tensor, "stride", nullptr);
  if (stride_seq == nullptr) {
    Py_DECREF(dtype);
    Py_DECREF(size_seq);
    return nullptr;
  }

  // Detect SymInts cheaply: ``size()`` returns ``torch.Size`` (a tuple
  // subclass) — fast-path the all-int case via PyLong_Check on each
  // element. Any non-PyLong → return None and let Python handle it.
  PyObject *size_fast = PySequence_Fast(size_seq, "size() must be a sequence");
  if (size_fast == nullptr) {
    Py_DECREF(dtype);
    Py_DECREF(size_seq);
    Py_DECREF(stride_seq);
    return nullptr;
  }
  Py_ssize_t ndim = PySequence_Fast_GET_SIZE(size_fast);
  PyObject **size_items = PySequence_Fast_ITEMS(size_fast);
  int64_t stack_sizes[8];
  int64_t *sizes_buf = ndim <= 8 ? stack_sizes : new int64_t[ndim];
  for (Py_ssize_t i = 0; i < ndim; ++i) {
    if (!PyLong_CheckExact(size_items[i])) {
      // SymInt or some other unusual object — bail to Python.
      if (sizes_buf != stack_sizes) delete[] sizes_buf;
      Py_DECREF(size_fast);
      Py_DECREF(dtype);
      Py_DECREF(size_seq);
      Py_DECREF(stride_seq);
      Py_RETURN_NONE;
    }
    sizes_buf[i] = PyLong_AsLongLong(size_items[i]);
    if (sizes_buf[i] == -1 && PyErr_Occurred()) {
      if (sizes_buf != stack_sizes) delete[] sizes_buf;
      Py_DECREF(size_fast);
      Py_DECREF(dtype);
      Py_DECREF(size_seq);
      Py_DECREF(stride_seq);
      return nullptr;
    }
  }
  Py_DECREF(size_fast);

  PyObject *stride_fast =
      PySequence_Fast(stride_seq, "stride() must be a sequence");
  if (stride_fast == nullptr) {
    if (sizes_buf != stack_sizes) delete[] sizes_buf;
    Py_DECREF(dtype);
    Py_DECREF(size_seq);
    Py_DECREF(stride_seq);
    return nullptr;
  }
  Py_ssize_t sndim = PySequence_Fast_GET_SIZE(stride_fast);
  PyObject **stride_items = PySequence_Fast_ITEMS(stride_fast);
  int64_t stack_strides[8];
  int64_t *strides_buf =
      sndim <= 8 ? stack_strides : new int64_t[sndim];
  for (Py_ssize_t i = 0; i < sndim; ++i) {
    if (!PyLong_CheckExact(stride_items[i])) {
      if (sizes_buf != stack_sizes) delete[] sizes_buf;
      if (strides_buf != stack_strides) delete[] strides_buf;
      Py_DECREF(stride_fast);
      Py_DECREF(dtype);
      Py_DECREF(size_seq);
      Py_DECREF(stride_seq);
      Py_RETURN_NONE;
    }
    strides_buf[i] = PyLong_AsLongLong(stride_items[i]);
    if (strides_buf[i] == -1 && PyErr_Occurred()) {
      if (sizes_buf != stack_sizes) delete[] sizes_buf;
      if (strides_buf != stack_strides) delete[] strides_buf;
      Py_DECREF(stride_fast);
      Py_DECREF(dtype);
      Py_DECREF(size_seq);
      Py_DECREF(stride_seq);
      return nullptr;
    }
  }
  Py_DECREF(stride_fast);
  Py_DECREF(size_seq);
  Py_DECREF(stride_seq);

  PyObject *sizes_tuple = hashable_dims_from_sizes(sizes_buf, ndim);
  PyObject *strides_tuple = hashable_dims_from_sizes(strides_buf, sndim);
  if (sizes_buf != stack_sizes) delete[] sizes_buf;
  if (strides_buf != stack_strides) delete[] strides_buf;
  if (sizes_tuple == nullptr || strides_tuple == nullptr) {
    Py_DECREF(dtype);
    Py_XDECREF(sizes_tuple);
    Py_XDECREF(strides_tuple);
    return nullptr;
  }

  // static_indices: ``frozenset(getattr(t, '_dynamo_static_indices', ()))``.
  // The vast majority of tensors don't have that attr; check via
  // GetAttr-without-error for speed (PyObject_GetAttr would raise).
  PyObject *static_indices_obj = nullptr;
  if (PyObject_HasAttr(tensor, g_static_indices_attr)) {
    PyObject *raw = PyObject_GetAttr(tensor, g_static_indices_attr);
    if (raw == nullptr) {
      Py_DECREF(dtype);
      Py_DECREF(sizes_tuple);
      Py_DECREF(strides_tuple);
      return nullptr;
    }
    static_indices_obj = PyFrozenSet_New(raw);
    Py_DECREF(raw);
    if (static_indices_obj == nullptr) {
      Py_DECREF(dtype);
      Py_DECREF(sizes_tuple);
      Py_DECREF(strides_tuple);
      return nullptr;
    }
  } else {
    Py_INCREF(g_empty_frozenset);
    static_indices_obj = g_empty_frozenset;
  }

  PyObject *out = PyTuple_New(4);
  if (out == nullptr) {
    Py_DECREF(dtype);
    Py_DECREF(sizes_tuple);
    Py_DECREF(strides_tuple);
    Py_DECREF(static_indices_obj);
    return nullptr;
  }
  PyTuple_SET_ITEM(out, 0, dtype);
  PyTuple_SET_ITEM(out, 1, sizes_tuple);
  PyTuple_SET_ITEM(out, 2, strides_tuple);
  PyTuple_SET_ITEM(out, 3, static_indices_obj);
  (void)kwnames;
  (void)self;
  return out;
}

// -----------------------------------------------------------------------------
// CompiledLauncher
// -----------------------------------------------------------------------------
//
// Holds all the per-config state captured from Triton at warmup time
// (CompiledKernel, packed metadata, hooks, ...). Its ``launch(grid_0,
// grid_1, grid_2, *args)`` method dispatches into Triton's own C
// launcher (``compiled_kernel.run``) with no Python wrapper frames
// between us and it.

typedef struct {
  PyObject_HEAD
  PyObject *compiled_run;        // bound method: compiled_kernel.run
  PyObject *triton_function;     // compiled_kernel.function
  PyObject *packed_metadata;     // compiled_kernel.packed_metadata
  PyObject *kernel_launch_metadata; // compiled_kernel.launch_metadata (callable)
  PyObject *launch_enter_hook;   // triton.knobs.runtime.launch_enter_hook (object)
  PyObject *launch_exit_hook;    // triton.knobs.runtime.launch_exit_hook (object)
  PyObject *get_current_stream;  // active_driver.get_current_stream (bound method)
  PyObject *device;              // device handle
  PyObject *binder;              // device_cache[4] (or None if skip)
  PyObject *run_kwargs;          // pre-built dict {num_warps, num_stages, ...}
  int bind_skip_safe;            // 0 / 1
} CompiledLauncherObject;

static int CompiledLauncher_init(CompiledLauncherObject *self, PyObject *args,
                                 PyObject *kwds) {
  static const char *kwlist[] = {"compiled_run",
                                 "triton_function",
                                 "packed_metadata",
                                 "kernel_launch_metadata",
                                 "launch_enter_hook",
                                 "launch_exit_hook",
                                 "get_current_stream",
                                 "device",
                                 "binder",
                                 "run_kwargs",
                                 "bind_skip_safe",
                                 nullptr};
  PyObject *compiled_run, *triton_function, *packed_metadata;
  PyObject *kernel_launch_metadata, *launch_enter_hook, *launch_exit_hook;
  PyObject *get_current_stream, *device, *binder, *run_kwargs;
  int bind_skip_safe;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "OOOOOOOOOOp",
          // Cast away const for the legacy keyword-list signature.
          const_cast<char **>(kwlist), &compiled_run, &triton_function,
          &packed_metadata, &kernel_launch_metadata, &launch_enter_hook,
          &launch_exit_hook, &get_current_stream, &device, &binder,
          &run_kwargs, &bind_skip_safe)) {
    return -1;
  }
  Py_INCREF(compiled_run);
  Py_INCREF(triton_function);
  Py_INCREF(packed_metadata);
  Py_INCREF(kernel_launch_metadata);
  Py_INCREF(launch_enter_hook);
  Py_INCREF(launch_exit_hook);
  Py_INCREF(get_current_stream);
  Py_INCREF(device);
  Py_INCREF(binder);
  Py_INCREF(run_kwargs);
  self->compiled_run = compiled_run;
  self->triton_function = triton_function;
  self->packed_metadata = packed_metadata;
  self->kernel_launch_metadata = kernel_launch_metadata;
  self->launch_enter_hook = launch_enter_hook;
  self->launch_exit_hook = launch_exit_hook;
  self->get_current_stream = get_current_stream;
  self->device = device;
  self->binder = binder;
  self->run_kwargs = run_kwargs;
  self->bind_skip_safe = bind_skip_safe ? 1 : 0;
  return 0;
}

static void CompiledLauncher_dealloc(CompiledLauncherObject *self) {
  Py_XDECREF(self->compiled_run);
  Py_XDECREF(self->triton_function);
  Py_XDECREF(self->packed_metadata);
  Py_XDECREF(self->kernel_launch_metadata);
  Py_XDECREF(self->launch_enter_hook);
  Py_XDECREF(self->launch_exit_hook);
  Py_XDECREF(self->get_current_stream);
  Py_XDECREF(self->device);
  Py_XDECREF(self->binder);
  Py_XDECREF(self->run_kwargs);
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// Determine whether either hook has any active subscribers. Newer
// Triton wraps these in a ``HookChain`` whose ``.calls`` attribute is a
// list (empty means no subscribers); older Triton just uses ``None`` /
// callable. Matches the Python ``_FastLauncher`` check.
static int hooks_inactive(PyObject *enter_hook, PyObject *exit_hook) {
  // Returns 1 if both hooks are inactive (so we can skip launch_metadata).
  // Returns 0 if either has work to do.
  static PyObject *calls_attr = nullptr;
  if (calls_attr == nullptr) {
    calls_attr = PyUnicode_InternFromString("calls");
    if (calls_attr == nullptr) {
      // Defer to the slow path; can't decide.
      PyErr_Clear();
      return 0;
    }
  }
  for (int i = 0; i < 2; ++i) {
    PyObject *hook = (i == 0) ? enter_hook : exit_hook;
    if (hook == Py_None) {
      continue;
    }
    if (!PyObject_HasAttr(hook, calls_attr)) {
      return 0; // Treat as active (older Triton style).
    }
    PyObject *calls = PyObject_GetAttr(hook, calls_attr);
    if (calls == nullptr) {
      PyErr_Clear();
      return 0;
    }
    int empty = 0;
    if (PyList_Check(calls)) {
      empty = (PyList_GET_SIZE(calls) == 0);
    } else {
      Py_ssize_t sz = PyObject_Size(calls);
      empty = (sz == 0);
      if (sz < 0) PyErr_Clear();
    }
    Py_DECREF(calls);
    if (!empty) {
      return 0;
    }
  }
  return 1;
}

// Shared core of the CompiledLauncher dispatch. Both the explicit
// ``.launch(...)`` method and the ``tp_call`` slot funnel through here.
// ``grid_0/1/2`` are the unpacked grid dims; ``kernel_args``/``kernel_nargs``
// is the kernel argument array (already grid-stripped).
static PyObject *CompiledLauncher_dispatch(CompiledLauncherObject *self,
                                           PyObject *grid_0, PyObject *grid_1,
                                           PyObject *grid_2,
                                           PyObject *const *kernel_args,
                                           Py_ssize_t kernel_nargs) {
  // Resolve binder result: either pass kernel_args through, or call
  // the binder to canonicalize. Storage for the bound-args tuple lives
  // for the duration of this call.
  PyObject *bound_tuple_owner = nullptr;
  PyObject *const *final_kernel_args = kernel_args;
  Py_ssize_t final_kernel_nargs = kernel_nargs;

  if (!self->bind_skip_safe && self->binder != Py_None) {
    PyObject *binder_args = PyTuple_New(kernel_nargs);
    if (binder_args == nullptr) {
      return nullptr;
    }
    for (Py_ssize_t i = 0; i < kernel_nargs; ++i) {
      Py_INCREF(kernel_args[i]);
      PyTuple_SET_ITEM(binder_args, i, kernel_args[i]);
    }
    PyObject *binder_result =
        PyObject_Call(self->binder, binder_args, self->run_kwargs);
    Py_DECREF(binder_args);
    if (binder_result == nullptr) {
      return nullptr;
    }
    if (!PyTuple_Check(binder_result) || PyTuple_GET_SIZE(binder_result) < 1) {
      Py_DECREF(binder_result);
      PyErr_SetString(PyExc_RuntimeError, "binder returned unexpected shape");
      return nullptr;
    }
    PyObject *bound_args_dict = PyTuple_GET_ITEM(binder_result, 0);
    PyObject *values_iter = PyObject_CallMethod(bound_args_dict, "values", nullptr);
    Py_DECREF(binder_result);
    if (values_iter == nullptr) {
      return nullptr;
    }
    // tuple(values()) → store and use its PySequence_Fast_ITEMS.
    bound_tuple_owner = PySequence_Tuple(values_iter);
    Py_DECREF(values_iter);
    if (bound_tuple_owner == nullptr) {
      return nullptr;
    }
    final_kernel_args =
        reinterpret_cast<PyObject *const *>(PySequence_Fast_ITEMS(bound_tuple_owner));
    final_kernel_nargs = PyTuple_GET_SIZE(bound_tuple_owner);
  }

  // stream = self._get_current_stream(self._device)
  PyObject *stream = PyObject_CallOneArg(self->get_current_stream, self->device);
  if (stream == nullptr) {
    Py_XDECREF(bound_tuple_owner);
    return nullptr;
  }

  // launch_metadata: only compute if hooks are active.
  PyObject *launch_metadata;
  if (hooks_inactive(self->launch_enter_hook, self->launch_exit_hook)) {
    Py_INCREF(Py_None);
    launch_metadata = Py_None;
  } else {
    // grid_tuple = (grid_0, grid_1, grid_2)
    PyObject *grid_tuple = PyTuple_Pack(3, grid_0, grid_1, grid_2);
    if (grid_tuple == nullptr) {
      Py_DECREF(stream);
      Py_XDECREF(bound_tuple_owner);
      return nullptr;
    }
    // metadata_args = (grid_tuple, stream, *kernel_args)
    Py_ssize_t md_nargs = 2 + final_kernel_nargs;
    PyObject *md_call_args = PyTuple_New(md_nargs);
    if (md_call_args == nullptr) {
      Py_DECREF(grid_tuple);
      Py_DECREF(stream);
      Py_XDECREF(bound_tuple_owner);
      return nullptr;
    }
    PyTuple_SET_ITEM(md_call_args, 0, grid_tuple);
    Py_INCREF(stream);
    PyTuple_SET_ITEM(md_call_args, 1, stream);
    for (Py_ssize_t i = 0; i < final_kernel_nargs; ++i) {
      Py_INCREF(final_kernel_args[i]);
      PyTuple_SET_ITEM(md_call_args, 2 + i, final_kernel_args[i]);
    }
    launch_metadata = PyObject_Call(self->kernel_launch_metadata,
                                    md_call_args, nullptr);
    Py_DECREF(md_call_args);
    if (launch_metadata == nullptr) {
      Py_DECREF(stream);
      Py_XDECREF(bound_tuple_owner);
      return nullptr;
    }
  }

  // compiled_run(grid_0, grid_1, grid_2, stream, function,
  //              packed_metadata, launch_metadata,
  //              enter_hook, exit_hook, *kernel_args)
  Py_ssize_t total_args = 9 + final_kernel_nargs;
  PyObject *call_args = PyTuple_New(total_args);
  if (call_args == nullptr) {
    Py_DECREF(stream);
    Py_DECREF(launch_metadata);
    Py_XDECREF(bound_tuple_owner);
    return nullptr;
  }
  Py_INCREF(grid_0); PyTuple_SET_ITEM(call_args, 0, grid_0);
  Py_INCREF(grid_1); PyTuple_SET_ITEM(call_args, 1, grid_1);
  Py_INCREF(grid_2); PyTuple_SET_ITEM(call_args, 2, grid_2);
  // stream (steal)
  PyTuple_SET_ITEM(call_args, 3, stream);
  Py_INCREF(self->triton_function);
  PyTuple_SET_ITEM(call_args, 4, self->triton_function);
  Py_INCREF(self->packed_metadata);
  PyTuple_SET_ITEM(call_args, 5, self->packed_metadata);
  // launch_metadata (steal)
  PyTuple_SET_ITEM(call_args, 6, launch_metadata);
  Py_INCREF(self->launch_enter_hook);
  PyTuple_SET_ITEM(call_args, 7, self->launch_enter_hook);
  Py_INCREF(self->launch_exit_hook);
  PyTuple_SET_ITEM(call_args, 8, self->launch_exit_hook);
  for (Py_ssize_t i = 0; i < final_kernel_nargs; ++i) {
    Py_INCREF(final_kernel_args[i]);
    PyTuple_SET_ITEM(call_args, 9 + i, final_kernel_args[i]);
  }

  PyObject *result = PyObject_Call(self->compiled_run, call_args, nullptr);
  Py_DECREF(call_args);
  Py_XDECREF(bound_tuple_owner);
  return result;
}

// CompiledLauncher.launch(grid_0, grid_1, grid_2, *args) → object
//
// METH_FASTCALL: args is a contiguous PyObject* array. The first three
// are grid coordinates (raw Python ints); the rest pass through to the
// binder (if active) and the Triton C launcher.
static PyObject *CompiledLauncher_launch(CompiledLauncherObject *self,
                                         PyObject *const *args,
                                         Py_ssize_t nargs) {
  if (nargs < 3) {
    PyErr_SetString(
        PyExc_TypeError,
        "CompiledLauncher.launch requires at least grid_0, grid_1, grid_2");
    return nullptr;
  }
  return CompiledLauncher_dispatch(self, args[0], args[1], args[2],
                                   args + 3, nargs - 3);
}

// CompiledLauncher.__call__(triton_kernel, grid_tuple, *args, **kwargs)
//
// Implements ``tp_call`` so a ``CompiledLauncher`` instance can be
// installed directly as a generated wrapper's ``_launcher=`` default —
// eliminating the ``_FastLauncher.__call__`` Python frame on the hot
// path. The wrapper's call site looks like::
//
//     _launcher(_helion_add, ((N + B - 1) // B,), x, y, out,
//               num_warps=2, num_stages=5)
//
// The triton_kernel positional arg is already baked into ``self->compiled_run``,
// so we ignore it. The grid is a tuple like ``(grid_0,)`` (or up to 3 elems);
// the kwargs are already baked into ``self->run_kwargs`` so we ignore them too.
static PyObject *CompiledLauncher_call(CompiledLauncherObject *self,
                                       PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  if (nargs < 2) {
    PyErr_SetString(PyExc_TypeError,
                    "CompiledLauncher() requires at least (triton_kernel, grid)");
    return nullptr;
  }
  // args[0] is the triton_kernel function — already baked in, ignored.
  PyObject *grid = PyTuple_GET_ITEM(args, 1);
  if (!PyTuple_Check(grid)) {
    PyErr_SetString(PyExc_TypeError,
                    "CompiledLauncher() grid argument must be a tuple");
    return nullptr;
  }
  Py_ssize_t grid_len = PyTuple_GET_SIZE(grid);
  PyObject *grid_0 = (grid_len > 0) ? PyTuple_GET_ITEM(grid, 0) : nullptr;
  PyObject *grid_1 = (grid_len > 1) ? PyTuple_GET_ITEM(grid, 1) : nullptr;
  PyObject *grid_2 = (grid_len > 2) ? PyTuple_GET_ITEM(grid, 2) : nullptr;
  static PyObject *g_one = nullptr;
  if (g_one == nullptr) {
    g_one = PyLong_FromLong(1);
    if (g_one == nullptr) return nullptr;
  }
  if (grid_0 == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    "CompiledLauncher() grid tuple must have ≥ 1 element");
    return nullptr;
  }
  if (grid_1 == nullptr) grid_1 = g_one;
  if (grid_2 == nullptr) grid_2 = g_one;

  // Kernel args: positional args[2:] of the call tuple. We need a
  // pointer to the contiguous storage; PyTuple_GET_ITEM(args, i) gives
  // us indexed access, and the underlying storage is contiguous, so we
  // can hand its pointer (offset by 2) to the dispatch helper directly.
  PyObject *const *kernel_args = nullptr;
  Py_ssize_t kernel_nargs = nargs - 2;
  if (kernel_nargs > 0) {
    kernel_args = &PyTuple_GET_ITEM(args, 2);
  }
  return CompiledLauncher_dispatch(self, grid_0, grid_1, grid_2, kernel_args,
                                   kernel_nargs);
}

static PyMethodDef CompiledLauncher_methods[] = {
    {"launch", reinterpret_cast<PyCFunction>(CompiledLauncher_launch),
     METH_FASTCALL,
     "Issue the kernel via Triton's C launcher. Args: (grid_0, grid_1, grid_2, *kernel_args)."},
    {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject CompiledLauncher_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "helion._C_ext.CompiledLauncher",              // tp_name
    sizeof(CompiledLauncherObject),                // tp_basicsize
    0,                                             // tp_itemsize
    reinterpret_cast<destructor>(CompiledLauncher_dealloc), // tp_dealloc
    0,                                             // tp_vectorcall_offset
    nullptr,                                       // tp_getattr
    nullptr,                                       // tp_setattr
    nullptr,                                       // tp_as_async
    nullptr,                                       // tp_repr
    nullptr,                                       // tp_as_number
    nullptr,                                       // tp_as_sequence
    nullptr,                                       // tp_as_mapping
    nullptr,                                       // tp_hash
    reinterpret_cast<ternaryfunc>(CompiledLauncher_call), // tp_call
    nullptr,                                       // tp_str
    nullptr,                                       // tp_getattro
    nullptr,                                       // tp_setattro
    nullptr,                                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                            // tp_flags
    "Helion C launcher (config + compiled kernel state).", // tp_doc
    nullptr,                                       // tp_traverse
    nullptr,                                       // tp_clear
    nullptr,                                       // tp_richcompare
    0,                                             // tp_weaklistoffset
    nullptr,                                       // tp_iter
    nullptr,                                       // tp_iternext
    CompiledLauncher_methods,                      // tp_methods
    nullptr,                                       // tp_members
    nullptr,                                       // tp_getset
    nullptr,                                       // tp_base
    nullptr,                                       // tp_dict
    nullptr,                                       // tp_descr_get
    nullptr,                                       // tp_descr_set
    0,                                             // tp_dictoffset
    reinterpret_cast<initproc>(CompiledLauncher_init), // tp_init
    nullptr,                                       // tp_alloc
    PyType_GenericNew,                             // tp_new
};

static PyMethodDef HelionCMethods[] = {
    {"tensor_key",
     reinterpret_cast<PyCFunction>(helion_c_tensor_key),
     METH_FASTCALL | METH_KEYWORDS,
     "Return ``(dtype, sizes_tuple, strides_tuple, static_indices_frozenset)`` "
     "for a tensor, or ``None`` if the caller should fall back to the Python "
     "implementation (e.g. SymInt sizes)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef helion_c_module = {
    PyModuleDef_HEAD_INIT,
    "helion._C_ext",
    "Helion's compiled-kernel hot path: bind cache and CompiledLauncher.",
    -1,
    HelionCMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // anonymous namespace

PyMODINIT_FUNC PyInit__C_ext(void) {
  PyObject *module = PyModule_Create(&helion_c_module);
  if (module == nullptr) {
    return nullptr;
  }
  if (PyType_Ready(&CompiledLauncher_Type) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  Py_INCREF(&CompiledLauncher_Type);
  if (PyModule_AddObject(module, "CompiledLauncher",
                         reinterpret_cast<PyObject *>(&CompiledLauncher_Type)) <
      0) {
    Py_DECREF(&CompiledLauncher_Type);
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
