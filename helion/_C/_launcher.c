/* Minimal C launcher (Chunk E experiment).
 *
 * A Python type whose ``tp_call`` slot dispatches a Triton kernel
 * launch directly into ``compiled_kernel.run`` (the C launcher
 * Triton emits), bypassing the Python ``default_launcher`` frame.
 *
 * Compared to a full Python-side fast launcher, this minimal version
 * trades correctness coverage for simplicity:
 *
 *   - No multi-spec cache. The compiled kernel captured on first
 *     ``prime()`` call is reused for ALL subsequent calls. Caller
 *     is responsible for only using stable specs (same alignment,
 *     same shape, etc.).
 *   - No knob/hook re-reads. If the user enables
 *     ``knobs.runtime.debug`` or attaches a profiler hook, we don't
 *     observe it. Caller is responsible for pre-priming with the
 *     final config.
 *   - No ``used_global_vals`` check. Mutating a tracked global
 *     between calls would silently use the stale binary.
 *   - No multi-device guard.
 *
 * Purpose: measure the per-call savings of replacing the Python
 * ``default_launcher`` frame with a C tp_call. The Phase 2 multi-spec
 * launcher's correctness guards aren't on this branch yet; a real
 * Chunk E would port them.
 *
 * Usage from Python:
 *
 *   import helion._C
 *   launcher = helion._C.CompiledLauncher()
 *   launcher.prime(triton_kernel, grid, args, num_warps=4, num_stages=2)
 *   # Then call as a plain function:
 *   launcher(triton_kernel, grid, *args, num_warps=4, num_stages=2)
 *
 * Or install it as the wrapper's ``_launcher`` kwdefault via Python.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>


typedef struct {
    PyObject_HEAD
    /* All captured at prime() time and held as strong refs. */
    PyObject *compiled_run;          /* compiled_kernel.run callable */
    PyObject *triton_function;       /* compiled_kernel.function */
    PyObject *packed_metadata;       /* compiled_kernel.packed_metadata */
    PyObject *kernel_launch_metadata;/* compiled_kernel.launch_metadata callable */
    PyObject *get_current_stream;    /* driver.active.get_current_stream */
    PyObject *device;                /* device id from driver */
    int primed;
} LauncherObject;


static void Launcher_dealloc(LauncherObject *self) {
    Py_XDECREF(self->compiled_run);
    Py_XDECREF(self->triton_function);
    Py_XDECREF(self->packed_metadata);
    Py_XDECREF(self->kernel_launch_metadata);
    Py_XDECREF(self->get_current_stream);
    Py_XDECREF(self->device);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *Launcher_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    LauncherObject *self = (LauncherObject *)type->tp_alloc(type, 0);
    if (self == NULL) return NULL;
    self->compiled_run = NULL;
    self->triton_function = NULL;
    self->packed_metadata = NULL;
    self->kernel_launch_metadata = NULL;
    self->get_current_stream = NULL;
    self->device = NULL;
    self->primed = 0;
    return (PyObject *)self;
}


/* prime(triton_kernel, grid, args, num_warps, num_stages) — runs
 * Triton's warmup compile once to capture the CompiledKernel and
 * driver references. ``args`` is a Python tuple of the kernel's
 * positional args. */
static PyObject *Launcher_prime(LauncherObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {
        "triton_kernel", "grid", "args", "num_warps", "num_stages", NULL
    };
    PyObject *triton_kernel;
    PyObject *grid;
    PyObject *kargs;
    int num_warps, num_stages;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOii", kwlist,
            &triton_kernel, &grid, &kargs, &num_warps, &num_stages)) {
        return NULL;
    }

    /* Build warmup kwargs dict: {grid, warmup=True, num_warps, num_stages,
     * launch_cooperative_grid=False} */
    PyObject *warmup_kwargs = PyDict_New();
    if (warmup_kwargs == NULL) return NULL;
    if (PyDict_SetItemString(warmup_kwargs, "grid", grid) < 0) goto fail;
    Py_INCREF(Py_True);
    if (PyDict_SetItemString(warmup_kwargs, "warmup", Py_True) < 0) {
        Py_DECREF(Py_True);
        goto fail;
    }
    Py_DECREF(Py_True);
    PyObject *nw = PyLong_FromLong(num_warps);
    if (nw == NULL) goto fail;
    PyDict_SetItemString(warmup_kwargs, "num_warps", nw);
    Py_DECREF(nw);
    PyObject *ns = PyLong_FromLong(num_stages);
    if (ns == NULL) goto fail;
    PyDict_SetItemString(warmup_kwargs, "num_stages", ns);
    Py_DECREF(ns);
    PyDict_SetItemString(warmup_kwargs, "launch_cooperative_grid", Py_False);

    /* Call triton_kernel.run(*args, **warmup_kwargs) */
    PyObject *run_method = PyObject_GetAttrString(triton_kernel, "run");
    if (run_method == NULL) goto fail;
    PyObject *compiled = PyObject_Call(run_method, kargs, warmup_kwargs);
    Py_DECREF(run_method);
    Py_DECREF(warmup_kwargs);
    warmup_kwargs = NULL;
    if (compiled == NULL || compiled == Py_None) {
        Py_XDECREF(compiled);
        PyErr_SetString(PyExc_RuntimeError, "Triton warmup returned None");
        return NULL;
    }

    /* Extract compiled_kernel.run FIRST so it triggers _init_handles */
    PyObject *run_fn = PyObject_GetAttrString(compiled, "run");
    if (run_fn == NULL) {
        Py_DECREF(compiled);
        return NULL;
    }
    PyObject *function = PyObject_GetAttrString(compiled, "function");
    PyObject *packed = PyObject_GetAttrString(compiled, "packed_metadata");
    PyObject *lm = PyObject_GetAttrString(compiled, "launch_metadata");
    Py_DECREF(compiled);
    if (function == NULL || packed == NULL || lm == NULL) {
        Py_DECREF(run_fn);
        Py_XDECREF(function);
        Py_XDECREF(packed);
        Py_XDECREF(lm);
        return NULL;
    }

    /* Get driver.active.get_current_stream and device. */
    PyObject *driver_mod = PyImport_ImportModule("triton.runtime.driver");
    if (driver_mod == NULL) goto state_fail;
    PyObject *driver = PyObject_GetAttrString(driver_mod, "driver");
    Py_DECREF(driver_mod);
    if (driver == NULL) goto state_fail;
    PyObject *active = PyObject_GetAttrString(driver, "active");
    Py_DECREF(driver);
    if (active == NULL) goto state_fail;
    PyObject *get_dev = PyObject_GetAttrString(active, "get_current_device");
    if (get_dev == NULL) { Py_DECREF(active); goto state_fail; }
    PyObject *device = PyObject_CallNoArgs(get_dev);
    Py_DECREF(get_dev);
    PyObject *gcs = PyObject_GetAttrString(active, "get_current_stream");
    Py_DECREF(active);
    if (device == NULL || gcs == NULL) {
        Py_XDECREF(device);
        Py_XDECREF(gcs);
        goto state_fail;
    }

    /* Publish all state, last is `primed`. */
    Py_XDECREF(self->compiled_run); self->compiled_run = run_fn;
    Py_XDECREF(self->triton_function); self->triton_function = function;
    Py_XDECREF(self->packed_metadata); self->packed_metadata = packed;
    Py_XDECREF(self->kernel_launch_metadata); self->kernel_launch_metadata = lm;
    Py_XDECREF(self->get_current_stream); self->get_current_stream = gcs;
    Py_XDECREF(self->device); self->device = device;
    self->primed = 1;
    Py_RETURN_NONE;

state_fail:
    Py_DECREF(run_fn);
    Py_DECREF(function);
    Py_DECREF(packed);
    Py_DECREF(lm);
    return NULL;
fail:
    Py_XDECREF(warmup_kwargs);
    return NULL;
}


/* tp_call: args = (triton_kernel, grid, *kernel_args).
 * kwargs may include num_warps/num_stages etc — we ignore them
 * (the values were baked at prime time).
 *
 * Builds: compiled_run(grid_0, grid_1, grid_2, stream,
 *                     function, packed_metadata,
 *                     None (launch_metadata),
 *                     None (enter_hook),
 *                     None (exit_hook),
 *                     *kernel_args)
 */
static PyObject *Launcher_call(LauncherObject *self, PyObject *args, PyObject *kwargs) {
    if (!self->primed) {
        PyErr_SetString(PyExc_RuntimeError, "CompiledLauncher not primed");
        return NULL;
    }

    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError,
            "launcher requires at least (triton_kernel, grid, *kernel_args)");
        return NULL;
    }
    /* args[0] is triton_kernel (unused — we have cached state)
     * args[1] is grid (tuple of ints) */
    PyObject *grid = PyTuple_GET_ITEM(args, 1);
    Py_ssize_t grid_n = PyTuple_GET_SIZE(grid);

    long grid_0 = 1, grid_1 = 1, grid_2 = 1;
    if (grid_n >= 1) grid_0 = PyLong_AsLong(PyTuple_GET_ITEM(grid, 0));
    if (grid_n >= 2) grid_1 = PyLong_AsLong(PyTuple_GET_ITEM(grid, 1));
    if (grid_n >= 3) grid_2 = PyLong_AsLong(PyTuple_GET_ITEM(grid, 2));

    PyObject *stream = PyObject_CallOneArg(self->get_current_stream, self->device);
    if (stream == NULL) return NULL;

    /* Build new args tuple: (grid_0, grid_1, grid_2, stream, function,
     * packed_metadata, None, None, None, *kernel_args) */
    Py_ssize_t n_kernel_args = nargs - 2;
    Py_ssize_t out_n = 9 + n_kernel_args;
    PyObject *out = PyTuple_New(out_n);
    if (out == NULL) { Py_DECREF(stream); return NULL; }

    PyTuple_SET_ITEM(out, 0, PyLong_FromLong(grid_0));
    PyTuple_SET_ITEM(out, 1, PyLong_FromLong(grid_1));
    PyTuple_SET_ITEM(out, 2, PyLong_FromLong(grid_2));
    PyTuple_SET_ITEM(out, 3, stream);
    Py_INCREF(self->triton_function);
    PyTuple_SET_ITEM(out, 4, self->triton_function);
    Py_INCREF(self->packed_metadata);
    PyTuple_SET_ITEM(out, 5, self->packed_metadata);
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(out, 6, Py_None);
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(out, 7, Py_None);
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(out, 8, Py_None);
    for (Py_ssize_t i = 0; i < n_kernel_args; ++i) {
        PyObject *ka = PyTuple_GET_ITEM(args, 2 + i);
        Py_INCREF(ka);
        PyTuple_SET_ITEM(out, 9 + i, ka);
    }

    PyObject *result = PyObject_Call(self->compiled_run, out, NULL);
    Py_DECREF(out);
    return result;
}


static PyMethodDef Launcher_methods[] = {
    {"prime", (PyCFunction)(void(*)(void))Launcher_prime,
     METH_VARARGS | METH_KEYWORDS, "Compile-and-capture the Triton kernel state."},
    {NULL, NULL, 0, NULL}
};


static PyTypeObject LauncherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "helion._C._launcher.CompiledLauncher",
    .tp_doc = "Minimal C launcher (Chunk E experiment).",
    .tp_basicsize = sizeof(LauncherObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Launcher_new,
    .tp_dealloc = (destructor)Launcher_dealloc,
    .tp_call = (ternaryfunc)Launcher_call,
    .tp_methods = Launcher_methods,
};


static struct PyModuleDef launcher_module = {
    PyModuleDef_HEAD_INIT,
    "_launcher",
    "Minimal C launcher type for Helion.",
    -1,
    NULL,
};


PyMODINIT_FUNC PyInit__launcher(void) {
    PyObject *m = PyModule_Create(&launcher_module);
    if (m == NULL) return NULL;
    if (PyType_Ready(&LauncherType) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&LauncherType);
    if (PyModule_AddObject(m, "CompiledLauncher", (PyObject *)&LauncherType) < 0) {
        Py_DECREF(&LauncherType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
