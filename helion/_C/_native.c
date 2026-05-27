/* Native accelerators for Helion's hot paths.
 *
 * Currently exports only:
 *   tensor_key(tensor, static_indices) -> tuple | None
 *     Returns (dtype, sizes_tuple, strides_tuple, static_indices) for
 *     the static-shapes specialization-key build inside Kernel.bind.
 *     Returns Python's ``None`` (not raising) for any input it can't
 *     handle, signaling the Python caller to fall back.
 *
 * Build (until a build-system hook lands):
 *   See helion/_C/README.md for the manual compile command. The
 *   extension is OPTIONAL: helion._C imports cleanly without it, and
 *   the Python fallback path produces identical keys.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>


static PyObject *
tensor_key(PyObject *self, PyObject *args)
{
    PyObject *tensor;
    PyObject *static_indices;
    if (!PyArg_ParseTuple(args, "OO", &tensor, &static_indices)) {
        return NULL;
    }

    /* dtype = tensor.dtype */
    PyObject *dtype = PyObject_GetAttrString(tensor, "dtype");
    if (dtype == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    /* sizes = tensor.size() */
    PyObject *sizes_obj = PyObject_CallMethod(tensor, "size", NULL);
    if (sizes_obj == NULL) {
        Py_DECREF(dtype);
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    /* strides = tensor.stride() */
    PyObject *strides_obj = PyObject_CallMethod(tensor, "stride", NULL);
    if (strides_obj == NULL) {
        Py_DECREF(dtype);
        Py_DECREF(sizes_obj);
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    /* Both sizes() and stride() return torch.Size (a tuple subclass)
     * for non-SymInt tensors. Each element must be a Python int — if
     * any are SymInts, we bail out to let the Python path build the
     * (env_id, expr) key it needs.
     *
     * Detect SymInt via the standard Python int check: torch.SymInt is
     * NOT a subclass of int, so PyLong_Check is False for them. The
     * tensor's sizes/strides tuples may contain SymInts in dynamic
     * shapes mode; we want only the static (all-ints) fast path here. */
    PyObject *sizes_tuple = NULL;
    PyObject *strides_tuple = NULL;

    if (!PyTuple_Check(sizes_obj) || !PyTuple_Check(strides_obj)) {
        /* Unexpected — torch.Size IS a tuple subclass. Fall back. */
        goto fallback;
    }

    Py_ssize_t n_sizes = PyTuple_GET_SIZE(sizes_obj);
    for (Py_ssize_t i = 0; i < n_sizes; ++i) {
        PyObject *item = PyTuple_GET_ITEM(sizes_obj, i);
        if (!PyLong_CheckExact(item)) {
            goto fallback;
        }
    }
    Py_ssize_t n_strides = PyTuple_GET_SIZE(strides_obj);
    for (Py_ssize_t i = 0; i < n_strides; ++i) {
        PyObject *item = PyTuple_GET_ITEM(strides_obj, i);
        if (!PyLong_CheckExact(item)) {
            goto fallback;
        }
    }

    /* All ints — convert the torch.Size subclasses to plain tuples so
     * the key is independent of torch's internal types. */
    sizes_tuple = PyTuple_GetSlice(sizes_obj, 0, n_sizes);
    if (sizes_tuple == NULL) goto fallback;
    strides_tuple = PyTuple_GetSlice(strides_obj, 0, n_strides);
    if (strides_tuple == NULL) goto fallback;

    /* Build the 4-tuple (dtype, sizes, strides, static_indices). */
    PyObject *result = PyTuple_New(4);
    if (result == NULL) goto fallback;

    /* Steal references into the tuple. */
    PyTuple_SET_ITEM(result, 0, dtype);          /* steals */
    PyTuple_SET_ITEM(result, 1, sizes_tuple);    /* steals */
    PyTuple_SET_ITEM(result, 2, strides_tuple);  /* steals */
    Py_INCREF(static_indices);
    PyTuple_SET_ITEM(result, 3, static_indices); /* steals */

    Py_DECREF(sizes_obj);
    Py_DECREF(strides_obj);
    return result;

fallback:
    Py_DECREF(dtype);
    Py_DECREF(sizes_obj);
    Py_DECREF(strides_obj);
    Py_XDECREF(sizes_tuple);
    Py_XDECREF(strides_tuple);
    PyErr_Clear();
    Py_RETURN_NONE;
}


static PyMethodDef NativeMethods[] = {
    {"tensor_key", tensor_key, METH_VARARGS,
     "Build the static-shapes specialization key for a tensor, "
     "or return None if the input is unsupported."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef nativemodule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "helion._C native accelerators",
    -1,
    NativeMethods
};


PyMODINIT_FUNC
PyInit__native(void)
{
    return PyModule_Create(&nativemodule);
}
