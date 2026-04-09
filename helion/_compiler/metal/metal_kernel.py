"""@metal_jit — decorator that translates Python AST to MSL at runtime.

Helion's codegen produces a Python function decorated with
``@metal_jit(args=[...], block_sizes=[...])``.  The function body
contains "fake Python" — valid Python AST that represents Metal kernel logic
(tl.load, tl.store, tgid/tid thread indices, etc.).

When the decorated function is called (with no arguments, by the launcher),
the decorator:
1. Extracts the function's source code and parses it to AST
2. Translates the AST body to MSL C++ source
3. Returns (msl_source, kernel_name) for the launcher to compile and dispatch
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def metal_jit(
    args: list[dict[str, object]],
    block_sizes: list[int],
) -> Callable[..., _MetalKernel]:
    """Decorator factory for Metal Python kernels.

    Args:
        args: List of arg metadata dicts with 'kind' ('tensor'/'symbol'),
              'name', and 'metal_dtype' (for tensors).
        block_sizes: Block size values for _BLOCK_SIZE_* constants.
    """

    def decorator(fn: Callable[..., object]) -> _MetalKernel:
        return _MetalKernel(fn, args, block_sizes)

    return decorator


class _MetalKernel:
    """Wrapped Metal kernel that translates Python AST to MSL on first call."""

    def __init__(
        self,
        fn: Callable[..., object],
        args_metadata: list[dict[str, object]],
        block_sizes: list[int],
    ) -> None:
        self._fn = fn
        self._name = fn.__name__
        self._args_metadata = args_metadata
        self._block_sizes = block_sizes

    def __call__(self) -> tuple[object, str]:
        """Return (compiled_lib, kernel_name) for the launcher."""
        import torch

        # Parse the function source to get the AST body
        source = inspect.getsource(self._fn)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        fn_def = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

        # Generate MSL from the AST body
        msl_source = _generate_msl(
            kernel_name=self._name,
            body_stmts=fn_def.body,
            args_metadata=self._args_metadata,
            block_sizes=self._block_sizes,
        )

        # Compile MSL to a Metal shader library
        lib = torch.mps.compile_shader(msl_source)  # type: ignore[attr-defined]
        return lib, self._name


def _generate_msl(
    kernel_name: str,
    body_stmts: list[ast.stmt],
    args_metadata: list[dict[str, object]],
    block_sizes: list[int],
) -> str:
    """Generate complete MSL source from Python AST body and metadata."""
    import operator

    from .msl_ast_walker import _collect_block_size_names
    from .msl_ast_walker import _emit_stmts

    msl_parts: list[str] = [
        "#include <metal_stdlib>",
        "#include <c10/metal/utils.h>",
        "#include <c10/metal/special_math.h>",
        "using namespace metal;",
        "",
    ]

    params: list[str] = []
    scalar_preamble: list[str] = []
    buf_idx = 0
    for arg_info in args_metadata:
        if arg_info["kind"] == "tensor":
            metal_dtype = arg_info["metal_dtype"]
            name = arg_info["name"]
            params.append(f"device {metal_dtype}* {name} [[buffer({buf_idx})]]")
            buf_idx += 1
        elif arg_info["kind"] == "symbol":
            name = arg_info["name"]
            buf_param = f"_buf_{name}"
            params.append(f"device const float* {buf_param} [[buffer({buf_idx})]]")
            buf_idx += 1
            scalar_preamble.append(f"    float {name} = {buf_param}[0];")

    params.extend(
        (
            "uint3 tgid [[threadgroup_position_in_grid]]",
            "uint3 tid [[thread_position_in_threadgroup]]",
        )
    )

    sig = ", ".join(params)
    msl_parts.append(f"kernel void {kernel_name}({sig}) {{")
    msl_parts.extend(scalar_preamble)

    # Declare _BLOCK_SIZE_* constants referenced by the AST body.
    block_size_map = _collect_block_size_names(body_stmts)
    for name, idx in sorted(block_size_map.items(), key=operator.itemgetter(1)):
        val = block_sizes[idx] if idx < len(block_sizes) else 1
        msl_parts.append(f"    constexpr int {name} = {val};")

    declared: set[str] = set(block_size_map)
    _emit_stmts(body_stmts, msl_parts, indent=4, declared=declared)

    msl_parts.append("}")
    return "\n".join(msl_parts)
