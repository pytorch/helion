from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import CapturedEpilogueExpr
from torch._inductor.ir import ExternalTritonTemplateBuffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..ast_read_writes import ast_rename
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import get_needed_imports
from ...language import memory_ops as helion_memory_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._inductor.ir import IRNode, MultiOutput
    from torch._inductor.scheduler import BaseSchedulerNode

    from ..inductor_lowering import CodegenState
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


class _CodeExpr(str):
    """A str whose repr() returns itself, for embedding variable names in generated code.

    When generating a kernel call like ``kernel(x, (a, b))``, container args are
    rebuilt via pytree into e.g. ``(_CodeExpr("a"), _CodeExpr("b"))``.  Python's
    built-in ``repr()`` on that tuple then produces ``(a, b)`` instead of
    ``('a', 'b')``, giving us correct code for free.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


class HelionTemplateBuffer(ExternalTritonTemplateBuffer):
    """Helion's concrete ``ExternalTritonTemplateBuffer`` implementation.

    Combines the Inductor IR node with Helion's kernel codegen logic by
    implementing ``_build_partial_render()`` — no separate backend object needed.

    Lifecycle
    ---------
    1. ``lower_helion_kernel`` calls ``HelionTemplateBuffer.create``
       which builds the IR node and returns ``(buf, outputs)``.
    2. The caller sets ``buf.fusable_outputs`` after computing output-info
       from ``create``.
    3. Inductor's scheduler reads ``fusable_outputs`` / ``all_inputs`` /
       ``all_output_names`` to plan epilogue/prologue fusion.
    4. ``_make_kernel_render`` (inherited) creates an ``ExternalTritonTemplateKernel``
       whose ``store_output()`` / ``load_input()`` capture fusion data, then
       ``_build_partial_render()`` generates the Triton AST with fused expressions.
    5. ``call_kernel`` (on the kernel handle) emits the call using the cached
       kernel source — no further involvement of this class.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs: "Sequence[IRNode]",
        *,
        kernel: "Kernel",
        bound_kernel: "BoundKernel",
        constant_args: dict[str, object],
        autotune_args: tuple[object, ...] | None = None,
        mutated_inputs: "Optional[Iterable[IRNode]]" = None,
        allowed_prologue_inps: "Optional[OrderedSet[str]]" = None,
    ) -> None:
        self._kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args = constant_args
        self._autotune_args = autotune_args

        super().__init__(
            layout=layout,
            inputs=inputs,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )

    def _build_partial_render(self, kernel: object) -> object:
        """Build PartialRender from captured data after store_output()/load_input().

        Called by render() in ExternalTritonTemplateBuffer._make_kernel_render after
        all fusion data has been captured.  Drives AST generation via the
        _codegen_epilogue_fusion / _codegen_prologue_fusion callbacks, computes
        imports and call args, and returns PartialRender with epilogue hooks.
        Sets ``kernel._precomputed_ks_meta = (imports, call_args, call_preamble)``.
        """
        # 1. Always autotune before AST generation so the unfused kernel is
        #    measured/cached before any fusion patterns are woven in.
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        # 2. Set captured data on self for use by _codegen_*_fusion callbacks.
        self._captured_epilogue_data: dict[str, CapturedEpilogueExpr] = {
            cap.kernel_output_param: cap for cap in kernel._captured_epilogue  # type: ignore[union-attr]
        }
        # Set of param names with active prologues (for _codegen_prologue_fusion eligibility check).
        self._prologue_fused_params: set[str] = {
            name for name in kernel.named_input_nodes  # type: ignore[union-attr]
            if f"_LOAD_INPUT_{name}" in kernel.render_hooks  # type: ignore[union-attr]
        }
        # Bridge info for prologue hooks: populated by _codegen_prologue_fusion,
        # read by load_input hook closures.
        self._prologue_bridge_info: dict[str, dict] = {}

        # 3. Build extra_params list: (param_name, buf_name) pairs appended after
        #    the regular kernel params in the call signature.
        extra_params: list[tuple[str, str]] = []
        for buf_name, param_name in kernel._epi_extra_inputs.items():  # type: ignore[union-attr]
            extra_params.append((param_name, buf_name))
        for src_buf, param_name in kernel._pro_extra_inputs.items():  # type: ignore[union-attr]
            extra_params.append((param_name, src_buf))

        # 4. Handle store_target redirects: when an epilogue stores to a different
        #    buffer than the kernel output, rename the output param and mark the
        #    original buffer as removed — unless the buffer has external consumers,
        #    in which case use two-store mode (emit both the original and epilogue stores).
        self._epilogue_renames: dict[str, str] = {}
        self._epilogue_two_store_params: dict[str, str] = {}

        # Compute the fused-node names so we can check buffer removability.
        scheduler = V.graph.scheduler
        fused_node_names: OrderedSet[str] | None = None
        if scheduler is not None:
            all_store_names: OrderedSet[str] = OrderedSet()
            all_store_names.add(self.get_name())
            for name in self._multi_output_children:
                all_store_names.add(name)
            for cap in kernel._captured_epilogue:  # type: ignore[union-attr]
                if cap.store_target:
                    all_store_names.add(cap.store_target)
            fused_node_names = OrderedSet(
                scheduler.name_to_buf[n].defining_op_name()
                for n in all_store_names
                if n in scheduler.name_to_buf
            )

        for cap in kernel._captured_epilogue:  # type: ignore[union-attr]
            if cap.store_target is not None and cap.store_target != cap.kernel_output_buf:
                redirect_param = f"_epi_out_{len(extra_params)}"
                extra_params.append((redirect_param, cap.store_target))
                # Check if the kernel output buffer can be safely removed.
                can_remove = (
                    scheduler is not None
                    and fused_node_names is not None
                    and scheduler.can_buffer_be_removed_through_fusion(
                        cap.kernel_output_buf, fused_node_names
                    )
                )
                if can_remove:
                    self._epilogue_renames[cap.kernel_output_param] = redirect_param
                    self.removed_buffers.add(cap.kernel_output_buf)
                else:
                    # Another consumer exists: keep original store and emit
                    # a second store for the epilogue target.
                    self._epilogue_two_store_params[cap.kernel_output_param] = redirect_param

        # 4b. Pre-register store_target redirects in kernel.args.output_buffers
        #     so that Inductor's V.ops.store can find the correct parameter name
        #     when the epilogue stores through the standard codegen path.
        for cap in kernel._captured_epilogue:  # type: ignore[union-attr]
            if cap.store_target and cap.store_target != cap.kernel_output_buf:
                for param, buf in extra_params:
                    if buf == cap.store_target:
                        kernel.args.output_buffers[cap.store_target] = param  # type: ignore[union-attr]

        # 5. Generate Triton AST with store/load transform callbacks active.
        #    _codegen_epilogue_fusion emits tile alias definitions and a placeholder.
        #    _codegen_prologue_fusion captures indexing context and returns a placeholder.
        extra_param_names = [p for p, _ in extra_params]
        root = self._generate_triton_ast(extra_params=extra_param_names if extra_params else None)
        if root is None:
            kernel._precomputed_ks_meta = ([], [])  # type: ignore[union-attr]
            return PartialRender("", {})

        # 6. Apply epilogue output-param redirects (rename kernel_output_param →
        #    redirect_param) directly in the device function.
        if self._epilogue_renames:
            for node in ast.walk(root):
                if isinstance(node, ast.FunctionDef) and "_helion_" in node.name:
                    ast_rename(node, self._epilogue_renames)
                    break

        # 7. Serialize to source (contains _helion_epi_i placeholder identifiers).
        source = self._ast_to_source(root)

        # 8. Compute call args and preamble.
        #    Import scanning is deferred to finalize_kernel_source() so that
        #    finalized hook content (preamble code, fused expressions) is included.
        call_order, constant_repr = self._call_order_and_constant_repr()
        prologue_primary_sources: dict[str, str | None] = dict(
            kernel._prologue_primary_sources  # type: ignore[union-attr]
        )
        call_preamble, call_args = self._build_call_args(
            call_order, constant_repr, prologue_primary_sources, extra_params
        )

        # 9. Store precomputed metadata for finalize_kernel_source().
        #    Imports are computed later in finalize_kernel_source() after hook resolution.
        kernel._precomputed_ks_meta = (call_args, call_preamble)  # type: ignore[union-attr]

        # 10. Return PartialRender — epilogue hooks finalized by finalize_kernel_source().
        return PartialRender(source, kernel.render_hooks)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                   #
    # ------------------------------------------------------------------ #

    def _call_order_and_constant_repr(self) -> tuple[list[str], dict[str, str]]:
        """Compute the kernel call order and pre-repr'd non-tensor args.

        ``call_order`` lists every parameter name in signature order.
        ``constant_repr`` maps non-tensor param names to their ``repr()``-ready
        strings (scalars, defaults, and rebuilt container args) so the inherited
        ``call_kernel`` can emit them without calling back into this class.
        """
        # Both tensor inputs AND constant args must be combined before
        # _rebuild_container_args so it can pop 'param.0', 'param.1' etc.
        all_args: dict[str, object] = {
            n: _CodeExpr(inp.get_name())  # type: ignore[union-attr]
            for n, inp in self._named_inputs.items()
        }
        for n, v in self._constant_args.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        tensor_flat_params = frozenset(self._named_inputs.keys())
        sig = self._kernel.signature.parameters
        order: list[str] = []
        const_repr: dict[str, str] = {}
        for n, p in sig.items():
            if n in all_args:
                order.append(n)
                if n not in tensor_flat_params:
                    const_repr[n] = repr(all_args[n])
            elif p.default is not p.empty:
                order.append(n)
                const_repr[n] = repr(p.default)
        return order, const_repr

    # ------------------------------------------------------------------ #
    # Private Helion-specific helpers                                    #
    # ------------------------------------------------------------------ #

    def _generate_triton_ast(
        self,
        extra_params: list[str] | None = None,
    ) -> ast.Module | None:
        """Generate and rename the Triton kernel AST.

        Activates ``store_transform`` / ``load_transform`` callbacks when
        active fusion specs are present so that ``hl.store`` / ``hl.load``
        sites inline fused expressions directly.
        """
        if not self._bound_kernel:
            return None

        cfg = self._bound_kernel._config
        assert cfg is not None, "Config should be set after ensure_config_exists"
        host_fn = self._kernel.name
        inner_fn = f"_helion_{host_fn}"
        inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            assert host_function is not None, "BoundKernel must have a host_function"
            root = generate_ast(
                host_function,
                cfg,
                emit_repro_caller=False,
                store_transform=self._codegen_epilogue_fusion
                if getattr(self, "_captured_epilogue_data", None)
                else None,
                load_transform=self._codegen_prologue_fusion
                if getattr(self, "_prologue_fused_params", None)
                else None,
                extra_params=extra_params,
            )

        assert isinstance(root, ast.Module)

        # Collect module-level variable names for uniquification
        # (e.g. constexpr assignments like ``_BLOCK_SIZE_0 = tl.constexpr(32)``).
        module_level_vars: dict[str, str] = {}
        for node in root.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level_vars[target.id] = (
                            f"{target.id}_{Placeholder.KERNEL_NAME}"
                        )

        # Rename functions, module-level vars, and all references to them.
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                elif node.name == inner_fn:
                    node.name = inner_fn_placeholder
            elif isinstance(node, ast.Name):
                if node.id == inner_fn:
                    node.id = inner_fn_placeholder
                elif node.id in module_level_vars:
                    node.id = module_level_vars[node.id]

        return root  # pyrefly: ignore[bad-return]

    def _ast_to_source(self, root: ast.Module) -> str:
        return get_needed_imports(root) + unparse(
            root, output_origin_lines=self._bound_kernel.settings.output_origin_lines
        )

    def _codegen_epilogue_fusion(
        self,
        state: "CodegenState",
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST | None:
        """Emit per-epilogue index definitions + ``_STORE_OUTPUT_{i}`` placeholder.

        Returns None to suppress the original tl.store (single-store mode),
        or the kernel value name to keep the original store (two-store mode).
        """
        param_name = state.device_function.tensor_arg(tensor).name
        cap = self._captured_epilogue_data.get(param_name)
        if cap is None:
            return value

        epi_idx = cap.epi_idx
        kernel_val_name = f"_kernel_val_{epi_idx}"

        # 1. Assign original value to unique temp variable, upcasting to float32
        #    when output dtype is float16/bfloat16.
        output_dtype = V.graph.get_dtype(cap.kernel_output_buf) if cap.kernel_output_buf else None
        if output_dtype in (torch.float16, torch.bfloat16):
            value_str = ast.unparse(value)
            state.add_statement(
                ast.parse(
                    f"{kernel_val_name} = ({value_str}).to(tl.float32)",
                    mode="exec",
                ).body[0]
            )
        else:
            state.add_statement(
                ast.Assign(
                    targets=[ast.Name(id=kernel_val_name, ctx=ast.Store())],
                    value=value,
                    lineno=0,
                )
            )

        # 2. Emit per-dimension index definitions with per-epilogue unique names.
        #    These match the names set on range tree entries by store_output()
        #    (x_epi{i}_{d}) and keep referenced variables alive through DCE.
        #    Using x_ prefix ensures get_block_shape recognizes them as XBLOCK.
        indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        for d, dim_str in enumerate(indexing.dim_index_exprs):
            state.add_statement(
                ast.parse(
                    f"x_epi{epi_idx}_{d} = {dim_str}", mode="exec"
                ).body[0]
            )

        # 3. Emit per-epilogue mask alias (unique name avoids cross-epilogue collision).
        mask_str = ast.unparse(indexing.mask_expr) if indexing.has_mask() else "None"
        state.add_statement(
            ast.parse(
                f"_tile_mask_{epi_idx} = {mask_str}", mode="exec"
            ).body[0]
        )

        # 4. Emit single placeholder statement.
        state.add_statement(
            ast.Expr(value=ast.Name(id=f"_STORE_OUTPUT_{epi_idx}", ctx=ast.Load()))
        )

        # 5. Handle store_target redirects.
        two_store_redirect = self._epilogue_two_store_params.get(param_name)
        output_ptr = self._epilogue_renames.get(param_name, param_name)

        if two_store_redirect is not None:
            # Two-store mode: keep original store (return kernel val).
            # Protect both the original and redirect params from DCE.
            state.device_function.protected_arg_names.add(output_ptr)
            state.device_function.protected_arg_names.add(two_store_redirect)
            return ast.Name(id=kernel_val_name, ctx=ast.Load())

        # Single-store: suppress original tl.store (return None sentinel).
        # Protect the output parameter (potentially renamed) from DCE.
        state.device_function.protected_arg_names.add(output_ptr)
        return None

    def _codegen_prologue_fusion(
        self,
        state: "CodegenState",
        tensor: torch.Tensor,
        value: ast.AST,
        indexing: "SubscriptIndexing",
    ) -> ast.AST:
        """Emit bridge variables + single ``_LOAD_INPUT_{param_name}`` placeholder.

        Bridge variable definitions are emitted as AST statements (to keep
        referenced variables like ``indices_0`` alive through DCE).  The
        ``_LOAD_INPUT_{param_name}`` placeholder is expanded at finalize time
        by the hook closure into preamble + result assignment.
        """
        param_name = state.device_function.tensor_arg(tensor).name
        if param_name not in self._prologue_fused_params:
            return value

        # Compute bridge: linearized offset + mask from SubscriptIndexing
        offset_str = ast.unparse(indexing.index_expr)
        mask_str = ast.unparse(indexing.mask_expr) if indexing.has_mask() else None

        # Emit bridge variable definitions as AST statements (prevents DCE of
        # referenced variables like indices_0, indices_1).
        state.add_statement(
            ast.parse(
                f"_pro_{param_name}_xindex = {offset_str}", mode="exec"
            ).body[0]
        )
        xmask_rhs = mask_str if mask_str else "True"
        state.add_statement(
            ast.parse(
                f"_pro_{param_name}_xmask = {xmask_rhs}", mode="exec"
            ).body[0]
        )

        # Record bridge info for hook closure (so it doesn't need to re-emit them).
        self._prologue_bridge_info[param_name] = {
            "offset_str": offset_str,
            "mask_str": mask_str if mask_str else "True",
        }

        # Emit single placeholder statement (preamble + result assignment).
        state.add_statement(
            ast.Expr(value=ast.Name(id=f"_LOAD_INPUT_{param_name}", ctx=ast.Load()))
        )

        # Protect param from DCE
        state.device_function.protected_arg_names.add(param_name)

        # Return variable reference (hook will assign fused value to this name).
        return ast.Name(id=f"_pro_{param_name}_result", ctx=ast.Load())


def _flatten_return_ast(
    ast_node: ast.expr | None,
    structured: object,
) -> list[ast.expr | None]:
    """Get the per-leaf AST nodes in DFS order matching build_multi_outputs traversal.

    Walks ``structured`` in the same order as ``build_multi_outputs`` to produce
    a flat list mapping ``leaf_idx`` → the corresponding AST node from the
    kernel's return statement.  Used to extract kernel parameter names
    (``ast.Name`` nodes) and detect symbolic (non-constant) non-tensor returns.
    """
    result: list[ast.expr | None] = []

    def walk(node: ast.expr | None, out: object) -> None:
        if isinstance(out, (list, tuple)):
            elts = node.elts if isinstance(node, (ast.Tuple, ast.List)) else None
            for i, item in enumerate(out):
                walk(elts[i] if elts is not None and i < len(elts) else None, item)
        else:
            result.append(node)  # leaf (tensor or non-tensor)

    walk(ast_node, structured)
    return result


@register_lowering(helion_kernel_wrapper_mutation, type_promotion_kind=None)
def lower_helion_kernel(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
) -> tuple[TensorBox, ...]:
    """Lower a Helion kernel HOP to a ``HelionTemplateBuffer``.

    Calls ``HelionTemplateBuffer.create`` to build the Inductor IR
    node and multi-output structure, then sets ``buf.fusable_outputs`` so
    Inductor's scheduler can plan epilogue/prologue fusion.
    """
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox → buffer / ReinterpretView.
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = HelionTemplateBuffer.realize_template_input(tb)

    # Build fake tensors for kernel binding (sympy exprs → concrete ints).
    def as_int(x: object, default: int) -> int:
        return int(x) if isinstance(x, (int, sympy.Integer)) else default

    all_args: dict[str, object] = {**constant_args}
    for n, r in realized.items():
        all_args[n] = torch.empty_strided(
            [as_int(s, 64) for s in r.get_size()],
            [as_int(s, 1) for s in r.get_stride()],
            dtype=r.get_dtype(),
            device=r.get_device(),
        )
    _rebuild_container_args(all_args)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))

    # Derive output structure from the bound kernel using inductor-time layouts.
    flat_leaves, tree_spec, return_ast = _get_flat_output(bound.host_function)

    if not flat_leaves:
        # No outputs — create still creates the buffer for mutations.
        buf, _ = HelionTemplateBuffer.create(
            realized_inputs=realized,
            structured_outputs=None,
            mutated_input_names=mutated_inputs_list or [],
            direct_aliases={},
            kernel=kernel,
            bound_kernel=bound,
            constant_args=constant_args,
            autotune_args=tuple(fake_tensors),
        )
        buf.fusable_outputs = {}
        return ()

    # Reconstruct structured output and create MultiOutput nodes.
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Flatten return_ast to index by leaf_idx (same traversal as build_multi_outputs).
    flat_ast = _flatten_return_ast(return_ast, structured)

    has_symbolic_returns = any(
        not isinstance(leaf, torch.Tensor)
        and not isinstance(flat_ast[i] if i < len(flat_ast) else None, ast.Constant)
        for i, leaf in enumerate(flat_leaves)
    )

    # Collect the set of tensor proxy ids that are directly stored by the Triton
    # device function.  Only these can be targets for Inductor epilogue fusion.
    # Using id() is safe here because the proxy objects live for the duration of
    # this function call.
    stored_proxy_ids: set[int] = set()
    for _graph_info in bound.host_function.device_ir.graphs:
        for _node in _graph_info.graph.nodes:
            if _node.op == "call_function" and _node.target is helion_memory_ops.store:
                if _node.args:
                    # node.args[0] is an fx.Node; its FakeTensor is in meta["val"]
                    _tensor_node = _node.args[0]
                    _fake_val = _tensor_node.meta.get("val") if hasattr(_tensor_node, "meta") else None
                    if _fake_val is not None:
                        stored_proxy_ids.add(id(_fake_val))

    # {mo_name: (kernel_param_name | None, proxy_id | None)}
    output_fusion_meta: dict[str, tuple[str | None, int | None]] = {}

    def on_tensor_leaf(
        mo_name: str,
        mo: "MultiOutput",
        _indices: list[tuple[type, int]],
        leaf_idx: int,
    ) -> None:
        ast_node = flat_ast[leaf_idx] if leaf_idx < len(flat_ast) else None
        leaf_proxy = flat_leaves[leaf_idx] if leaf_idx < len(flat_leaves) else None
        output_fusion_meta[mo_name] = (
            ast_node.id if isinstance(ast_node, ast.Name) else None,
            id(leaf_proxy) if leaf_proxy is not None else None,
        )

    buf, result = HelionTemplateBuffer.create(
        realized_inputs=realized,
        structured_outputs=structured,
        mutated_input_names=mutated_inputs_list or [],
        direct_aliases={
            i: realized[name]
            for i, name in cast(
                "dict[int, str]", output_spec.get("direct_aliases", {})
            ).items()
            if name in realized
        },
        on_tensor_leaf=on_tensor_leaf,
        kernel=kernel,
        bound_kernel=bound,
        constant_args=constant_args,
        autotune_args=tuple(fake_tensors),
    )

    # Compute fusable_outputs: param is known + no symbolic returns + the return
    # value is a tensor directly stored by the Triton device function.
    # Using the stored-proxy check (rather than an input-shape heuristic) correctly
    # handles matmul-like kernels whose output shape differs from all inputs, while
    # still excluding reduction outputs (e.g. out.sum(dim=1)) that are computed
    # outside the Triton kernel and therefore cannot be epilogue-fused.
    seen_params: set[str] = set()
    fusable_outputs: dict[str, str] = {}
    for mo_name, (param, proxy_id) in output_fusion_meta.items():
        if (
            param is not None
            and not has_symbolic_returns
            and proxy_id in stored_proxy_ids
            and param not in seen_params  # one epilogue per store site
        ):
            fusable_outputs[mo_name] = param
            seen_params.add(param)

    buf.fusable_outputs = fusable_outputs

    return result


@register_lowering(helion_kernel_wrapper_functional, type_promotion_kind=None)
def lower_helion_kernel_functional(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[TensorBox, ...], dict[str, TensorBox]]:
    from torch._inductor.lowering import clone

    cloned = {
        n: clone(tb) if n in tensors_to_clone and isinstance(tb, TensorBox) else tb
        for n, tb in tensor_args.items()
    }
    outputs = lower_helion_kernel(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned,
        output_spec=output_spec,
    )
    return (outputs, {n: cloned[n] for n in tensors_to_clone if n in cloned})
