"""Stage TensorMap-unaligned GEMM operands through padded host copies.

The tcgen05 pipeline loads A and B through TMA only when both operands pass
the TensorMap proof (``_tcgen05_tma_tensor_is_aligned``: a 16-byte base, one
unit stride, every other stride a 16-byte multiple). An operand whose rows
are odd-sized fails it -- bf16 rows of ``K = 4097`` are 8194 bytes -- and
``_emit_mma_pipeline`` then keeps the scalar shared-memory producers for both
operands: one load warp copying elements one at a time, which ran a 4097^3
bf16 GEMM in 22 ms on a B200 (cuBLAS 0.87 ms, the Triton backend 0.5 ms, the
TMA pipeline at 4096^3 0.12 ms).

This source-level pass (``Backend.customize_ast``, before type propagation)
rewrites such kernels to re-lay the offending operand out on the host::

    x = torch.constant_pad_nd(x, [0, 63])[..., :4097]

a row-major copy whose row stride is a 128-byte multiple and whose logical
shape is unchanged (the slice bounds the view, so the pad columns are never
read). The copy is a fresh allocation at storage offset 0, which the TMA
proof accepts (``_fresh_tensors`` lists ``torch.constant_pad_nd`` as a fresh
factory). It costs one read and one write of the operand per call, against
a GEMM that reads the operand once per output tile column.

An argument is staged when it (1) is a tensor of rank 2 or 3 with static
strides and at least ``STAGING_MIN_ELEMENTS`` elements, (2) fails the
TensorMap stride rule (the compile environment's fake arguments sit at
storage offset 0, so a misaligned base is not visible here and keeps the
scalar producers), (3) is read as a matmul operand -- the bare name or its
tile subscript inside a device loop (``x[tile_m, tile_k]``, every index a
plain name) passed to ``torch.addmm`` / ``baddbmm`` / ``matmul`` / ``mm`` /
``bmm``, ``hl.dot`` or ``@`` -- and (4) is mentioned nowhere else except
through such device-loop reads and its logical metadata (``size``,
``shape``, ``dtype``, ...): a store, a rebinding, an alias, a view, a host
subscript (``x[0]`` is a view of the copy), a layout query, a ``return`` or
any other call would observe the copy as a different tensor or write into
it instead of the caller's (``_uses_are_benign``). Dynamic shapes are left
alone; ``cute_stage_unaligned_operands=False`` disables the pass.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _global_value
import helion.language as hl

if TYPE_CHECKING:
    from ..host_function import HostFunction

# Operands below this many elements keep the scalar producers: the copy's
# launch is a visible fraction of a latency-bound GEMM, and the small-shape
# lowerings (row-resident contractions, materialized fission, the paired
# fan-out epilogue) match the kernel's host prelude as written.
STAGING_MIN_ELEMENTS = 1 << 18
# TensorMap bases and outer strides must be 16-byte multiples; a 128-byte
# row stride additionally keeps every 64-element bf16 TMA box row inside one
# 128-byte line (4097^3 bf16 on the pre-existing two-CTA edge family:
# 0.315 ms at 128 vs 0.333 ms at a 32-byte stride) and satisfies the
# packet-alignment facts.
STAGED_ALIGNMENT_BYTES = 128
_TMA_ALIGNMENT_BYTES = 16

# (function, positional operand indices, keyword operand names): the two
# MMA operands only, never the accumulator (``addmm``'s ``input``).
_MATMUL_OPERANDS: tuple[tuple[object, tuple[int, ...], tuple[str, ...]], ...] = (
    (torch.addmm, (1, 2), ("mat1", "mat2")),
    (torch.baddbmm, (1, 2), ("batch1", "batch2")),
    (torch.matmul, (0, 1), ("input", "other")),
    (torch.mm, (0, 1), ("input", "mat2")),
    (torch.bmm, (0, 1), ("input", "mat2")),
    (hl.dot, (0, 1), ("mat1", "mat2")),
)
# Tensor attributes the padded copy answers like the original: its logical
# shape and dtype. Layout queries (``stride``, ``is_contiguous``,
# ``storage_offset``, ``data_ptr``) and ``data`` are not among them.
_METADATA_ATTRIBUTES = frozenset(
    {
        "size",
        "shape",
        "dtype",
        "device",
        "ndim",
        "dim",
        "numel",
        "element_size",
        "is_cuda",
    }
)


def _needs_staging(tensor: torch.Tensor) -> int | None:
    """Pad (in elements of the last axis) when ``tensor`` fails the TMA rule.

    Returns None for tensors the TensorMap proof accepts, whose strides are
    symbolic, that are not rank 2 or 3, that hold fewer than
    ``STAGING_MIN_ELEMENTS`` elements, or that broadcast an axis (a stride-0
    axis: the copy would materialize every repeat). Axes of extent one carry
    no addressable stride and are ignored.
    """
    # GEMM operands: a matrix or a batch of matrices. Rank-4 attention
    # operands go through the flash matcher, which has its own layout rules.
    if tensor.ndim not in (2, 3):
        return None
    strides = tensor.stride()
    sizes = tensor.size()
    offset = tensor.storage_offset()
    if not all(type(value) is int for value in (*strides, *sizes, offset)):
        return None
    if math.prod(sizes) < STAGING_MIN_ELEMENTS:
        return None
    element_size = tensor.element_size()
    unit_strides = 0
    aligned = offset * element_size % _TMA_ALIGNMENT_BYTES == 0
    for size, stride in zip(sizes, strides, strict=True):
        if size == 1:
            continue
        if stride == 0:
            return None
        if stride == 1:
            unit_strides += 1
        elif stride * element_size % _TMA_ALIGNMENT_BYTES:
            aligned = False
    if aligned and unit_strides == 1:
        return None
    align_elements = STAGED_ALIGNMENT_BYTES // element_size
    # A whole alignment block when the extent is already a multiple: the
    # operand then fails the rule through its layout (a misaligned base or a
    # non-unit inner stride), and a zero pad would clone that layout.
    return (-sizes[-1]) % align_elements or align_elements


def _plain_tile_subscript(node: ast.Subscript) -> bool:
    """``x[tile_m, tile_k]``: every index a plain name. Offset arithmetic,
    slices and ``hl.load`` forms belong to the grouped, row-union, attention
    and full-slice lowerings, which keep their own operand protocols."""
    indices = (
        list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    )
    return all(isinstance(index, ast.Name) for index in indices)


def _matmul_operands(
    call: ast.Call, host: HostFunction, local_names: set[str]
) -> list[ast.AST]:
    """The MMA operand expressions of a matmul-family call (else empty)."""
    callee = _global_value(call.func, host, local_names)
    for function, positions, keyword_names in _MATMUL_OPERANDS:
        if callee is function:
            return [
                *(
                    call.args[position]
                    for position in positions
                    if position < len(call.args)
                ),
                *(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg in keyword_names
                ),
            ]
    return []


def _tile_read_aliases(module: ast.Module) -> dict[str, str]:
    """Names bound exactly once to a plain tile read of another name
    (``x_tile = x[tile_m, tile_k]``), mapped to that name."""
    aliases: dict[str, str | None] = {}
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        target = node.targets[0].id
        root = None
        if (
            isinstance(node.value, ast.Subscript)
            and _plain_tile_subscript(node.value)
            and isinstance(node.value.value, ast.Name)
        ):
            root = node.value.value.id
        aliases[target] = None if target in aliases else root
    return {name: root for name, root in aliases.items() if root is not None}


def _root_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """The argument a matmul operand reads: a name, its plain tile subscript,
    or a name bound once to such a subscript."""
    if isinstance(node, ast.Subscript) and _plain_tile_subscript(node):
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return aliases.get(node.id, node.id)


def _matmul_operand_names(host: HostFunction, local_names: set[str]) -> set[str]:
    """Argument names read as a matmul operand somewhere in the kernel."""
    module = ast.Module(body=host.body, type_ignores=[])
    aliases = _tile_read_aliases(module)
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            operands: list[ast.AST] = [node.left, node.right]
        elif isinstance(node, ast.Call):
            operands = _matmul_operands(node, host, local_names)
        else:
            continue
        for operand in operands:
            root = _root_name(operand, aliases)
            if root is not None:
                names.add(root)
    return names


def _device_loop_nodes(
    module: ast.Module, host: HostFunction, local_names: set[str]
) -> set[int]:
    """ids of the nodes inside ``for ... in hl.tile(...)`` / ``hl.grid(...)``
    loops: the code that runs on the device, where a tile subscript is a
    load and not a view."""
    inside: set[int] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and _global_value(node.iter.func, host, local_names) in (hl.tile, hl.grid)
        ):
            for statement in node.body:
                inside.update(id(child) for child in ast.walk(statement))
    return inside


def _uses_are_benign(host: HostFunction, name: str, local_names: set[str]) -> bool:
    """Whether every mention of the argument ``name`` reads it in a form the
    padded copy answers like the original.

    Allowed: a plain tile subscript read inside a device loop, a metadata
    attribute (``_METADATA_ATTRIBUTES``), and the bare name as an MMA
    operand (``@`` or a ``_MATMUL_OPERANDS`` call). Everything else -- a
    store or augmented assignment, a rebinding, an alias of any form
    (``s = a.detach()``, tuple unpacking, an annotated assignment), a view,
    a host subscript (``row = a[0]`` is a view of the copy), a layout
    query, a ``return``, or the name passed to any other call -- keeps the
    argument unstaged: the copy would then be observed as a different
    tensor, or a write into it would never reach the caller.
    """
    module = ast.Module(body=host.body, type_ignores=[])
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    device_nodes = _device_loop_nodes(module, host, local_names)
    for node in ast.walk(module):
        if not (isinstance(node, ast.Name) and node.id == name):
            continue
        if not isinstance(node.ctx, ast.Load):
            return False
        parent = parents.get(id(node))
        if isinstance(parent, ast.Subscript) and parent.value is node:
            if (
                isinstance(parent.ctx, ast.Load)
                and id(parent) in device_nodes
                and _plain_tile_subscript(parent)
            ):
                continue
            return False
        if (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr in _METADATA_ATTRIBUTES
        ):
            continue
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.MatMult):
            continue
        call = parent
        if isinstance(parent, ast.keyword):
            call = parents.get(id(parent))
        if isinstance(call, ast.Call) and any(
            operand is node for operand in _matmul_operands(call, host, local_names)
        ):
            continue
        return False
    return True


def stage_unaligned_matmul_operands(host: HostFunction) -> bool:
    """Pad TensorMap-unaligned matmul operands on the host; True when rewritten."""
    env = CompileEnvironment.current()
    if not env.settings.cute_stage_unaligned_operands or not env.settings.static_shapes:
        return False
    tensors = {
        name: value
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    pads = {
        name: pad
        for name, tensor in tensors.items()
        if (pad := _needs_staging(tensor)) is not None
    }
    if not pads:
        return False
    local_names = _bound_names(host)
    operands = _matmul_operand_names(host, local_names)
    staged = [
        name
        for name in pads
        if name in operands and _uses_are_benign(host, name, local_names)
    ]
    if not staged:
        return False
    torch_alias = next(
        (
            name
            for name, value in host.fn.__globals__.items()
            if value is torch and name.isidentifier() and name not in local_names
        ),
        None,
    )
    if torch_alias is None:
        return False
    statements = []
    for name in staged:
        statement = statement_from_string(
            f"{name} = {torch_alias}.constant_pad_nd({name}, [0, {pads[name]}])"
            f"[..., :{tensors[name].size(-1)}]"
        )
        for node in ast.walk(statement):
            if isinstance(node, ExtendedAST):
                node._compiler_inserted = True
        statements.append(statement)
    host.body[:0] = statements
    return True
