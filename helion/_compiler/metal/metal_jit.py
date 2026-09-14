"""@metal_jit — decorator that JIT-compiles a Python function to an MSL kernel.

Helion's codegen produces a Python function decorated with ``@metal_jit``.
The function body is Python-syntax Metal — valid Python AST that maps to
MSL C++.  Expressions come from Inductor's MetalOverrides
(``metal.precise.sin``, ``c10.metal.max``, ``static_cast<float>``),
pointer ops from PointerIndexingStrategy (``tl.load``, ``tl.store``),
and thread indices from MetalBackend (``tgid[0]``, ``tid[0]``).

The decorator infers all metadata at runtime from the actual call arguments:
- **Tensor dtypes** → from ``arg.dtype`` (mapped via ``DTYPE_TO_METAL``)
- **Scalar args** → detected as 0-dim tensors (``arg.ndim == 0``)
- **Block sizes** → read from module-level globals (``_BLOCK_SIZE_0 = 256``)

When the launcher calls ``metal_kernel(*args)``, the decorator:
1. Parses the function source to recover the Python AST
2. Infers the MSL kernel signature from the actual tensor args
3. Reads block sizes from the function's module globals
4. Translates the AST body to MSL C++ via the walker
5. Compiles via ``torch.mps.compile_shader``
6. Returns ``(compiled_lib, kernel_name)``
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ... import exc
from .msl_ast_walker import EmitState
from .msl_ast_walker import _emit_stmts
from .msl_reduction import REDUCTION_INCLUDES
from .msl_reduction import REDUCTION_NAMESPACE
from .msl_reduction import REDUCTION_PREAMBLE
from .reduction import SIMD_GROUP_VAR
from .reduction import TG_BUFFER_GLOBAL_PREFIX

if TYPE_CHECKING:
    from collections.abc import Callable


def metal_jit(fn: Callable[..., object]) -> _MetalKernel:
    """Decorator that JIT-compiles a Python function to an MSL Metal kernel."""
    return _MetalKernel(fn)


class _MetalKernel:
    """Wrapped Metal kernel that translates Python AST to MSL on first call."""

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        self._name = fn.__name__
        self.msl_source: str | None = None
        self.required_threads_per_threadgroup: tuple[int, int, int] | None = None
        self._compiled: dict[object, tuple[object, str]] = {}

    def _signature_key(self, args: tuple[object, ...]) -> tuple[object, ...]:
        """Everything the generated MSL depends on, besides the (fixed) body.

        The body AST and module globals are constant for a given generated
        module, so only the argument shapes/dtypes, the values of non-tensor
        (constexpr) arguments, and the launcher-supplied threadgroup size can
        change the emitted source.
        """
        arg_keys = tuple(
            (arg.dtype, arg.ndim) if isinstance(arg, torch.Tensor) else (type(arg), arg)
            for arg in args
        )
        return (self.required_threads_per_threadgroup, arg_keys)

    def __call__(self, *args: object) -> tuple[object, str]:
        """Return (compiled_lib, kernel_name) for the launcher.

        Args are the kernel arguments (tensors and scalars) — used to
        infer dtypes for the MSL kernel signature.

        Results are memoized: re-deriving the MSL means re-parsing the source
        AST and re-running ``torch.mps.compile_shader``'s header embedding, a
        few milliseconds that would otherwise be paid on every launch.
        """
        key = self._signature_key(args)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached

        # Parse the function source to get the AST
        source = inspect.getsource(self._fn)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        fn_def = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

        # Generate MSL from the AST body + actual args + module globals
        param_names = [a.arg for a in fn_def.args.args]
        self.msl_source = _generate_msl(
            kernel_name=self._name,
            body_stmts=fn_def.body,
            param_names=param_names,
            args=args,
            fn_globals=self._fn.__globals__,
            required_threads_per_threadgroup=self.required_threads_per_threadgroup,
        )

        # Compile MSL to a Metal shader library
        lib = torch.mps.compile_shader(self.msl_source)  # type: ignore[attr-defined]
        result = (lib, self._name)
        self._compiled[key] = result
        return result


def _generate_msl(
    kernel_name: str,
    body_stmts: list[ast.stmt],
    param_names: list[str],
    args: tuple[object, ...],
    fn_globals: dict[str, object],
    required_threads_per_threadgroup: tuple[int, int, int] | None = None,
) -> str:
    """Generate complete MSL source from Python AST body and actual args.

    Infers Metal dtypes and scalar vs tensor distinction directly from
    the args.  Reads block sizes from fn_globals.
    """
    from torch._inductor.codegen.mps import DTYPE_TO_METAL

    msl_parts: list[str] = [
        "#include <metal_stdlib>",
        "#include <c10/metal/utils.h>",
        "#include <c10/metal/special_math.h>",
        "using namespace metal;",
    ]
    threadgroup_buffers = _threadgroup_buffers(fn_globals, body_stmts)
    uses_reductions = _uses_reductions(body_stmts)
    if uses_reductions:
        msl_parts.extend(REDUCTION_INCLUDES)
        msl_parts.extend(("", REDUCTION_PREAMBLE))
    if _uses_mpp_markers(body_stmts):
        _raise_if_mpp_unsupported_device()
        msl_parts.extend(
            [
                "#if !__has_include(<metal_tensor>)",
                '#error "Helion Metal MPP lowering requires <metal_tensor>"',
                "#endif",
                (
                    "#if !__has_include("
                    "<MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>)"
                ),
                (
                    '#error "Helion Metal MPP lowering requires '
                    '<MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>"'
                ),
                "#endif",
                "#include <metal_tensor>",
                "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>",
                "using namespace mpp::tensor_ops;",
            ]
        )
    msl_parts.append("")

    params: list[str] = []
    scalar_preamble: list[str] = []
    buf_idx = 0
    for name, arg in zip(param_names, args, strict=True):
        if not isinstance(arg, torch.Tensor):
            # A host-side constexpr the launcher passes positionally (e.g. a
            # rolled reduction's ``_REDUCTION_BLOCK_*``).  ``default_metal_launcher``
            # only binds tensors as buffers, so bake the value into the shader.
            # ``_MetalKernel._signature_key`` includes these values, so a
            # different one recompiles rather than reusing a stale shader.
            scalar_preamble.append(f"    {_constexpr_decl(name, arg)}")
            continue
        if arg.dtype not in DTYPE_TO_METAL:
            raise ValueError(f"Unsupported Metal dtype: {arg.dtype}")
        metal_dtype = DTYPE_TO_METAL[arg.dtype]
        is_scalar = arg.ndim == 0
        if is_scalar:
            buf_param = f"_buf_{name}"
            params.append(
                f"device const {metal_dtype}* {buf_param} [[buffer({buf_idx})]]"
            )
            scalar_preamble.append(f"    {metal_dtype} {name} = {buf_param}[0];")
        else:
            params.append(f"device {metal_dtype}* {name} [[buffer({buf_idx})]]")
        buf_idx += 1

    params.extend(
        (
            "uint3 tgid [[threadgroup_position_in_grid]]",
            "uint3 tid [[thread_position_in_threadgroup]]",
        )
    )
    if threadgroup_buffers:
        # Cross-SIMD-group reductions index their slice of the shared scratch
        # buffer by SIMD group; see metal/reduction.py::_group_base_expr.
        params.append(f"uint {SIMD_GROUP_VAR} [[simdgroup_index_in_threadgroup]]")

    sig = ",\n    ".join(params)
    required_threads_attr = ""
    if required_threads_per_threadgroup is not None:
        tx, ty, tz = required_threads_per_threadgroup
        required_threads_attr = (
            f"[[required_threads_per_threadgroup({tx}, {ty}, {tz})]] "
        )
    msl_parts.append(
        f"{required_threads_attr}kernel void {kernel_name}(\n    {sig}\n) {{"
    )
    msl_parts.extend(scalar_preamble)

    # Declare _BLOCK_SIZE_* constants from module globals
    block_sizes = {
        name: int(val)
        for name, val in fn_globals.items()
        if name.startswith("_BLOCK_SIZE_") and isinstance(val, int)
    }
    for name in sorted(block_sizes):
        msl_parts.append(f"    constexpr int {name} = {block_sizes[name]};")

    # ``threadgroup`` variables are only legal at kernel scope, but a reduction
    # can be emitted from inside a loop body, so their declarations are hoisted
    # here from the generated ``_METAL_TG_BUF_*`` module globals.
    for buf_name, buf_dtype, buf_slots in threadgroup_buffers:
        msl_parts.append(f"    threadgroup {buf_dtype} {buf_name}[{buf_slots}];")

    state = EmitState(
        declared={*block_sizes, *(name for name, _, _ in threadgroup_buffers)}
    )

    _emit_stmts(body_stmts, msl_parts, indent=4, state=state)

    msl_parts.append("}")
    return "\n".join(msl_parts)


def _constexpr_decl(name: str, value: object) -> str:
    """MSL declaration for a non-tensor kernel argument."""
    if isinstance(value, bool):
        return f"constexpr bool {name} = {'true' if value else 'false'};"
    if isinstance(value, int):
        return f"constexpr int {name} = {value};"
    if isinstance(value, float):
        return f"constexpr float {name} = {value!r};"
    raise exc.BackendUnsupported("metal", f"kernel argument type: {type(value)}")


def _uses_mpp_markers(body_stmts: list[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id.startswith("_metal_mpp_"):
            return True
    return False


def _uses_reductions(body_stmts: list[ast.stmt]) -> bool:
    """Return True if the body calls into the ``helion_red`` MSL namespace.

    Every reduction the Metal backend emits is routed through that namespace
    (see ``metal/msl_reduction.py``), so this one check decides whether the
    kernel needs the reduction preamble and the SIMD-group index parameter.
    """
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id == REDUCTION_NAMESPACE:
            return True
    return False


def _threadgroup_buffers(
    fn_globals: dict[str, object],
    body_stmts: list[ast.stmt],
) -> list[tuple[str, str, int]]:
    """Collect ``(name, metal_dtype, slots)`` scratch declarations to hoist.

    Written by ``metal/reduction.py::alloc_threadgroup_buffer`` as generated
    module-level globals, the same transport ``_BLOCK_SIZE_*`` uses.  Only
    buffers this kernel body actually references are declared, so a global left
    behind by discarded codegen cannot consume threadgroup memory here.
    """
    referenced = {
        node.id
        for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[]))
        if isinstance(node, ast.Name)
    }
    buffers = []
    for name in sorted(fn_globals):
        if not name.startswith(TG_BUFFER_GLOBAL_PREFIX):
            continue
        buf_name, buf_dtype, buf_slots = fn_globals[name]  # type: ignore[misc]
        if buf_name not in referenced:
            continue
        buffers.append((str(buf_name), str(buf_dtype), int(buf_slots)))  # type: ignore[arg-type]
    return buffers


def _raise_if_mpp_unsupported_device() -> None:
    device_name = torch._C._mps_get_name()  # type: ignore[attr-defined]
    if "paravirtual" in device_name.lower():
        raise exc.BackendUnsupported(
            "metal",
            "MPP matmul lowering is not supported on Apple paravirtual MPS "
            "devices; use a physical Apple GPU such as a self-hosted M-series "
            "runner",
        )
