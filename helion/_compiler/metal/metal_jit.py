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

from .msl_ast_walker import _emit_stmts

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

    def __call__(self, *args: object) -> tuple[object, str]:
        """Return (compiled_lib, kernel_name) for the launcher.

        Args are the kernel arguments (tensors and scalars) — used to
        infer dtypes for the MSL kernel signature.
        """
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
        )

        # Compile MSL to a Metal shader library
        lib = torch.mps.compile_shader(self.msl_source)  # type: ignore[attr-defined]
        return lib, self._name


def _generate_msl(
    kernel_name: str,
    body_stmts: list[ast.stmt],
    param_names: list[str],
    args: tuple[object, ...],
    fn_globals: dict[str, object],
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
        "",
    ]

    params: list[str] = []
    scalar_preamble: list[str] = []
    for buf_idx, (name, arg) in enumerate(zip(param_names, args, strict=True)):
        assert isinstance(arg, torch.Tensor), f"Expected tensor, got {type(arg)}"
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

    params.extend(
        (
            "uint3 tgid [[threadgroup_position_in_grid]]",
            "uint3 tid [[thread_position_in_threadgroup]]",
        )
    )

    sig = ", ".join(params)
    msl_parts.append(f"kernel void {kernel_name}({sig}) {{")
    msl_parts.extend(scalar_preamble)

    # Declare _BLOCK_SIZE_* constants from module globals
    block_sizes = {
        name: int(val)
        for name, val in fn_globals.items()
        if name.startswith("_BLOCK_SIZE_") and isinstance(val, int)
    }
    for name in sorted(block_sizes):
        msl_parts.append(f"    constexpr int {name} = {block_sizes[name]};")

    declared: set[str] = set(block_sizes)
    _emit_stmts(body_stmts, msl_parts, indent=4, declared=declared)

    msl_parts.append("}")
    return "\n".join(msl_parts)
