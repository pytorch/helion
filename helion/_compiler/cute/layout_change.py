"""Layout change operation for the CuTe backend.

Provides ``_cute_layout_change``, an API function inserted into the FX graph
by the layout propagation pass when a producer and consumer disagree on
thread-value layout.  The CuTe codegen lowers it to a shared-memory
round-trip that rearranges data between layouts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction

if TYPE_CHECKING:
    import ast

    from ..backend import Backend
    from ..inductor_lowering import CodegenState
    from .layout import ThreadLayout

log = logging.getLogger(__name__)


@_decorators.api()
def _cute_layout_change(tensor: torch.Tensor) -> torch.Tensor:
    """Placeholder op inserted into the FX graph for layout rearrangement.

    At codegen time this is lowered to a shared-memory round-trip
    (or a register shuffle when possible).
    """
    raise AssertionError("this should never be called directly")


@_decorators.register_fake(_cute_layout_change)
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor)


@_decorators.codegen(_cute_layout_change, "cute")
def _(state: CodegenState) -> ast.AST:
    """Generate a shared-memory round-trip to change thread-value layout.

    The source and destination layouts are stored in the FX node's metadata:
      - node.meta["cute_layout_change_src"]  — ThreadLayout of the producer
      - node.meta["cute_layout_constraint"].output_layout — ThreadLayout for the
        consumer-facing output

    Each thread writes its value to shared memory at the flat offset dictated
    by the *source* layout's strides, then (after a barrier) reads back from
    the offset dictated by the *destination* layout's strides.  This
    implements the data permutation between the two thread-value mappings.
    """
    assert state.fx_node is not None
    input_ast = state.ast_arg(0)
    input_val = state.proxy_arg(0)
    assert isinstance(input_val, torch.Tensor)

    from .layout_propagation import META_KEY

    src_layout: ThreadLayout = state.fx_node.meta["cute_layout_change_src"]
    dst_layout = state.fx_node.meta[META_KEY].output_layout
    assert dst_layout is not None

    log.debug(
        "layout change %s: %s -> %s",
        state.fx_node.name,
        src_layout.tag.value,
        dst_layout.tag.value,
    )

    # If layouts are actually compatible, just pass through
    if src_layout.is_compatible(dst_layout):
        return input_ast

    env = CompileEnvironment.current()
    backend = env.backend
    fn = state.device_function
    dtype_str = backend.dtype_str(input_val.dtype)

    # Compute shared memory size — must cover the full tile
    tile_numel = src_layout.tile_numel()

    # Allocate shared memory buffer
    smem_ptr_var = fn.new_var("layout_change_smem_ptr", dce=True)
    smem_var = fn.new_var("layout_change_smem", dce=True)
    state.add_statement(
        f"{smem_ptr_var} = cute.arch.alloc_smem({dtype_str}, {tile_numel})"
    )
    state.add_statement(
        f"{smem_var} = cute.make_tensor({smem_ptr_var}, ({tile_numel},))"
    )

    # Assign input to a named variable using an AST placeholder so the
    # input expression is properly emitted (not repr'd).
    input_name = fn.new_var("layout_change_input", dce=True)
    state.add_statement(
        statement_from_string(f"{input_name} = {{_inp}}", _inp=input_ast)
    )

    # Compute flat smem offsets using layout strides
    thread_expr = _thread_id_expr(backend)
    src_offset = _flat_offset_expr(thread_expr, src_layout)
    dst_offset = _flat_offset_expr(thread_expr, dst_layout)

    # Write: each thread stores its value at the source-layout offset
    state.add_statement(f"{smem_var}[{src_offset}] = {input_name}")

    # Synchronize all threads
    state.add_statement("cute.arch.sync_threads()")

    # Read: each thread loads from the destination-layout offset
    result_var = fn.new_var("layout_change_result", dce=True)
    state.add_statement(f"{result_var} = {smem_var}[{dst_offset}]")

    return expr_from_string(result_var)


def _thread_id_expr(backend: Backend) -> str:
    """Build the linear thread index from the kernel's physical thread block dims.

    Uses the actual launched thread-block axes (not the layout's 1D thread_shape)
    so that all (x, y, z) threads get unique shared-memory offsets in ND
    kernels.
    """
    dims = DeviceFunction.current().tile_strategy.thread_block_dims()
    axis_sizes = {axis: dim for axis, dim in enumerate(dims) if dim > 1}
    return backend.thread_linear_index_expr(axis_sizes)  # type: ignore[return-value]


def _flat_offset_expr(thread_expr: str, layout: ThreadLayout) -> str:
    """Build the flat shared-memory offset expression from a thread ID and layout.

    Uses the layout's ``thread_stride`` to map the linear thread index to a tile
    element position.  Only 1-D thread shapes are supported (all current
    factories produce these).  Multi-value layouts (``num_values > 1``) are not
    yet handled — only the base thread offset is emitted, which is correct when
    each thread holds exactly one scalar.
    """
    assert len(layout.thread_shape) == 1, (
        "multi-dim thread_shape not yet supported in layout change codegen"
    )
    thread_stride = layout.thread_stride[0]
    if thread_stride == 1:
        return thread_expr
    return f"({thread_expr}) * {thread_stride}"
