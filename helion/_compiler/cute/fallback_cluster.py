"""Preferred + fallback cluster launch of a 4-CTA pair kernel
(``tcgen05_fallback_cluster``).

GB300 co-schedules 36 4-CTA clusters, 144 of its 152 SMs.  Launched with a
fallback cluster (``cudaLaunchAttributePreferredClusterDimension``) the GPU
forms the preferred (2, 2, 1) cluster where it fits and the (2, 1, 1) pair on
the SMs left over.  The kernel's body is outlined into a ``@cute.jit``
function whose ``cutlass.Constexpr`` flag the statements that depend on the
physical cluster select on (``fallback_cluster_select``), and the kernel calls
it once per cluster shape (4096x50264x2048 bf16 on 256x256 pairs)::

    @cute.jit
    def _helion_mm_body(a, b, out, tma_atom_a, tma_tensor_a, ..., tcgen05_fallback_cluster: cutlass.Constexpr):
        tcgen05_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(((2, 1, 1) if tcgen05_fallback_cluster else (2, 2, 1))), ...)
        tcgen05_cta_rank_in_cluster = (...block_idx_in_cluster()) & cutlass.Int32((1 if tcgen05_fallback_cluster else 3))
        tcgen05_ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(..., cutlass.Int32((1 if tcgen05_fallback_cluster else 2)))
        smem_a = cutlass.utils.SmemAllocator().allocate_array(cutlass.BFloat16, cute.cosize(sA_layout.outer), byte_alignment=128)
        ...

    @cute.kernel
    def _helion_mm(a, b, out, tma_atom_a, tma_tensor_a, ..., tma_atom_a_fb, tma_tensor_a_fb, ...):
        if cute.experimental.is_preferred_cluster((2, 2, 1)):
            _helion_mm_body(a, b, out, tma_atom_a, tma_tensor_a, ..., False)
        else:
            _helion_mm_body(a, b, out, tma_atom_a_fb, tma_tensor_a_fb, ..., True)

Both bodies walk the preferred schedule: a pair runs the N lane
(``block_idx.y``; under CLC the cancelled unit's) of the 2x2 slot it occupies,
loads its own A (the fallback TMA atoms carry no multicast across the lanes)
and its MMA alone releases its ring's stages.  A CTA runs one body, so the two
share their shared memory (``smem_merge_branch_allocs=True``); the DSL
overlays ``SmemAllocator`` allocations only, so the outlined body's
``cute.arch.alloc_smem`` calls become ``allocate_array`` calls (two static
copies of a 6-stage 256x256 ring would not fit: libNVVM fails).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ... import exc
from ..ast_extension import create
from ..ast_extension import create_arg
from ..ast_extension import create_arguments
from ..ast_extension import expr_from_string
from .tcgen05_prologue import masked_cluster_rank_expr

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


def enable_fallback_cluster(
    fn: DeviceFunction, fallback_cluster_shape: tuple[int, int, int]
) -> None:
    """Record the fallback cluster and reserve the body's Constexpr flag."""
    state = fn.cute_state.tcgen05
    if state.fallback_cluster_var is None:
        state.fallback_cluster_var = fn.new_var("tcgen05_fallback_cluster")
    state.fallback_cluster_shape = fallback_cluster_shape


def fallback_cluster_select(
    fn: DeviceFunction, preferred: object, fallback: object
) -> str:
    """``preferred``, or ``(fallback if <flag> else preferred)`` in a kernel
    with a fallback cluster: a Python conditional on the outlined body's
    ``cutlass.Constexpr`` flag, resolved when the DSL traces each body."""
    flag = fn.cute_state.tcgen05.fallback_cluster_var
    if flag is None or preferred == fallback:
        return f"{preferred}"
    return f"({fallback} if {flag} else {preferred})"


def physical_cluster_rank_expr(
    fn: DeviceFunction, rank_expr: str, cluster_size: int, fallback_cluster_size: int
) -> str:
    """``rank_expr`` (the CTA's rank in its cluster) masked to the physical
    cluster: ``& 3`` in a 4-CTA cluster, ``& 1`` in a fallback pair."""
    if fn.cute_state.tcgen05.fallback_cluster_var is None:
        return masked_cluster_rank_expr(rank_expr, cluster_size)
    mask = fallback_cluster_select(fn, cluster_size - 1, fallback_cluster_size - 1)
    return f"({rank_expr}) & cutlass.Int32({mask})"


def _unwrap_static_int32(node: ast.expr) -> ast.expr:
    """``cutlass.Int32(<int>)`` -> ``<int>``: the allocator sizes a static
    layout, and an ``Int32`` is a dynamic value."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Int32"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and type(node.args[0].value) is int
    ):
        return node.args[0]
    return node


class _MergeableSmemAllocations(ast.NodeTransformer):
    """``cute.arch.alloc_smem(T, n[, alignment])`` ->
    ``cutlass.utils.SmemAllocator().allocate_array(T, n[, byte_alignment=])``."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "alloc_smem"
            and ast.unparse(func.value) == "cute.arch"
        ):
            return node
        args = list(node.args)
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if len(args) == 3:
            keywords.setdefault("alignment", args.pop())
        if len(args) != 2 or set(keywords) - {"alignment"}:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_fallback_cluster shares the two cluster bodies' shared "
                f"memory, which needs every allocation in allocator form: {ast.unparse(node)}",
            )
        call = expr_from_string(
            "cutlass.utils.SmemAllocator().allocate_array({dtype}, {count})",
            dtype=args[0],
            count=_unwrap_static_int32(args[1]),
        )
        assert isinstance(call, ast.Call)
        if "alignment" in keywords:
            call.keywords.append(
                create(ast.keyword, arg="byte_alignment", value=keywords["alignment"])
            )
        return ast.copy_location(call, node)


def outline_fallback_cluster_bodies(
    fn: DeviceFunction, fn_def: ast.FunctionDef, prefix: list[ast.stmt]
) -> None:
    """Outline the kernel's body into ``<kernel>_body`` (appended to
    ``prefix``) and make the kernel call it once per cluster shape: the
    preferred cluster with its own TMA atoms and the flag ``False``, the
    fallback with the fallback atoms (``fallback_cluster_twin_args``) and
    ``True``."""
    state = fn.cute_state.tcgen05
    if state.fallback_cluster_shape is None:
        return
    flag = state.fallback_cluster_var
    preferred = state.cluster_shape
    assert flag is not None and preferred is not None
    twins = state.fallback_cluster_twin_args
    fallback_args = set(twins.values())
    params = [arg for arg in fn_def.args.args if arg.arg not in fallback_args]
    if fn_def.args.vararg or fn_def.args.kwarg or fn_def.args.kwonlyargs:
        raise exc.BackendUnsupported(
            "cute", "tcgen05_fallback_cluster outlines a positional kernel signature"
        )
    body_args = create_arguments(
        [
            *(
                create_arg(
                    arg.arg,
                    None if arg.annotation is None else ast.unparse(arg.annotation),
                )
                for arg in params
            ),
            create_arg(flag, "cutlass.Constexpr"),
        ]
    )
    body_name = f"{fn_def.name}_body"
    allocations = _MergeableSmemAllocations()
    body_def = create(
        ast.FunctionDef,
        name=body_name,
        args=body_args,
        body=[allocations.visit(stmt) for stmt in fn_def.body],
        decorator_list=[expr_from_string("cute.jit")],
        type_params=[],
    )

    def call(names: list[str], fallback: bool) -> ast.stmt:
        return create(
            ast.Expr,
            value=expr_from_string(
                f"{body_name}({', '.join(names)}, {fallback})",
            ),
        )

    preferred_names = [arg.arg for arg in params]
    fn_def.body = [
        create(
            ast.If,
            test=expr_from_string(
                f"cute.experimental.is_preferred_cluster({preferred!r})"
            ),
            body=[call(preferred_names, False)],
            orelse=[call([twins.get(name, name) for name in preferred_names], True)],
        )
    ]
    prefix.append(body_def)
