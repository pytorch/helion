from __future__ import annotations

import ast
from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import clone
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import (
    ExternalTritonTemplateKernel,  # pyrefly: ignore[missing-module-attribute]
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from ...language import memory_ops as helion_memory_ops
from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import get_needed_import_lines

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from typing import Any
    from typing import Iterable

    from torch._inductor.ir import IRNode
    from torch._inductor.ir import MultiOutput

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


class HelionTemplateBuffer(TemplateBuffer):
    """Inductor template buffer for Helion kernel."""

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        *,
        kernel: Kernel,
        bound_kernel: BoundKernel,
        constant_args: dict[str, object],
        autotune_args: tuple[object, ...] | None = None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
        named_inputs: dict[str, IRNode] | None = None,
    ) -> None:
        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args

        tb_self = self  # capture for closure

        def _make_kernel_render(
            out_node: TemplateBuffer, hint_override: object = None
        ) -> tuple[object, Callable[[], PartialRender]]:
            kernel = ExternalTritonTemplateKernel(out_node)

            def render() -> PartialRender:
                return tb_self._render_with_hooks(kernel)

            return kernel, render

        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=_make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
            named_inputs=named_inputs,  # pyrefly: ignore[unexpected-keyword]
        )

    def _render_with_hooks(self, kernel: Any) -> PartialRender:  # noqa: ANN401
        """Set up fusion hooks, generate AST, and return a PartialRender.

        Called as the ``render()`` function from the standard
        ``codegen_template_body`` path.  ``_setup_fusion_hooks()``
        populates fusion metadata on the kernel (epilogue indices,
        prologue variables, source buffers, etc.); this method reads
        that metadata and wires up store/load transform callbacks
        before generating the Triton AST.
        """
        # 1. Always autotune before AST generation.
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        # 2. Set up fusion hooks (requires V.kernel context).
        kernel._setup_fusion_hooks()

        # 3. Read pre-computed fusion metadata from kernel.
        self._epilogue_idx_by_param = kernel._epilogue_idx_by_param
        self._epilogue_keep_store = kernel._epilogue_keep_store
        self._prologue_vars = kernel._prologue_vars
        self._prologue_fused_params = set(kernel._prologue_vars.keys())
        self._prologue_has_source = {
            param_name
            for param_name in self._prologue_fused_params
            if kernel._prologue_source_buffers.get(param_name) is not None
        }
        self._prologue_emitted: set[str] = set()
        prologue_source_buffers = dict(kernel._prologue_source_buffers)

        # 4. Build extra_params list from extra inputs and store targets.
        extra_params = [(param, buf) for buf, param in kernel._extra_inputs.items()]
        for buf, param in kernel._extra_store_targets.items():
            if (param, buf) not in extra_params:
                extra_params.append((param, buf))

        # 5. Generate Triton AST with store/load transform callbacks active.
        root = self._generate_triton_ast(extra_params=[p for p, _ in extra_params])
        if root is None:
            return PartialRender("", kernel.render_hooks)

        # 6. Compute call args and preamble, store on kernel.
        call_order, constant_repr = self._call_order_and_constant_repr()
        kernel._call_preamble, kernel._call_args = self._build_call_args(
            call_order, constant_repr, prologue_source_buffers, extra_params
        )

        # 7. Store imports on kernel for emit_kernel_override, return
        # PartialRender with just the kernel body (no import lines).
        kernel._kernel_imports = get_needed_import_lines(root)
        source = unparse(
            root,
            output_origin_lines=self._bound_kernel.settings.output_origin_lines,
        )
        return PartialRender(source, kernel.render_hooks)

    # ------------------------------------------------------------------ #
    # TemplateBuffer overrides for multi-output layout                   #
    # ------------------------------------------------------------------ #

    def should_allocate(self) -> bool:
        return False

    def get_size(self) -> Sequence[sympy.Expr]:
        children = self._multi_output_children  # pyrefly: ignore[missing-attribute]
        if children:
            first_child = next(iter(children.values()))
            return first_child.get_size()
        return []

    def get_outputs(self) -> list[Buffer]:
        return [self, *self.mutation_outputs]

    def set_current_node(self, node: object) -> AbstractContextManager[None]:
        return nullcontext()

    def _build_call_args(
        self,
        call_order: list[str],
        constant_repr: dict[str, str],
        prologue_source_buffers: dict[str, str | None],
        extra_params: list[tuple[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Compute ``(call_preamble, call_args)`` for the kernel invocation."""
        preamble: list[str] = []

        def resolve_param(param_name: str) -> str | None:
            named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
            node = named_inputs.get(param_name)
            if node is None:
                return constant_repr.get(param_name)

            source_buf = prologue_source_buffers.get(param_name)

            if isinstance(node, ReinterpretView):
                base = source_buf if source_buf is not None else node.data.get_name()
                name = f"reinterp_{len(preamble)}"
                preamble.append(
                    f"{name} = reinterpret_tensor("
                    f"{base}, {tuple(node.get_size())}, {tuple(node.get_stride())}, {node.layout.offset})"
                )
                return name

            if source_buf is not None:
                return source_buf
            if param_name in prologue_source_buffers:
                return None  # source-less prologue, fully inlined
            return node.get_name()  # type: ignore[union-attr]

        call_args: list[str] = [
            resolved
            for param in call_order
            if (resolved := resolve_param(param)) is not None
        ]
        call_args.extend(buf_name for _, buf_name in extra_params)
        return preamble, call_args

    @classmethod
    def create(
        cls,
        realized_inputs: dict[str, IRNode],
        structured_outputs: object,
        mutated_input_names: list[str],
        direct_aliases: dict[int, IRNode],
        *,
        on_tensor_leaf: Callable[[str, Any, list[tuple[type, int]], int], None]
        | None = None,
        on_non_tensor_leaf: Callable[[int], None] | None = None,
        **buffer_kwargs: Any,  # noqa: ANN401
    ) -> tuple[HelionTemplateBuffer, tuple[TensorBox, ...]]:
        """Build a HelionTemplateBuffer and return ``(buf, outputs)``."""
        inputs = list(realized_inputs.values())
        dev = inputs[0].get_device() if inputs else torch.device("cuda")

        mutated_nodes = [
            realized_inputs[n] for n in mutated_input_names if n in realized_inputs
        ]
        mutated_inp_names = {n.get_name() for n in mutated_nodes}
        # Exclude container-flattened inputs (names with dots like "tensors.0")
        # from prologue fusion — the parameter remapping doesn't handle them.
        container_inp_names = {
            inp.get_name()  # type: ignore[union-attr]
            for param_name, inp in realized_inputs.items()
            if "." in param_name
        }
        buf = cls(
            layout=MultiOutputLayout(device=dev),  # pyrefly: ignore[bad-argument-type]
            inputs=inputs,
            mutated_inputs=mutated_nodes or None,
            allowed_prologue_inps=OrderedSet(
                inp.get_name()
                for inp in inputs  # type: ignore[union-attr]
                if inp.get_name() not in mutated_inp_names
                and inp.get_name() not in container_inp_names
            ),
            named_inputs=realized_inputs,
            **buffer_kwargs,
        )
        for inp in mutated_nodes:
            V.graph.never_reuse_buffers.add(inp.get_name())

        flat, _ = (
            pytree.tree_flatten(structured_outputs)
            if structured_outputs is not None
            else ([], None)
        )
        if not any(isinstance(leaf, torch.Tensor) for leaf in flat):
            return buf, ()

        result = (
            TemplateBuffer.build_multi_outputs(  # pyrefly: ignore[missing-attribute]
                buf,
                structured_outputs,
                direct_alias_at_leaf=direct_aliases,
                on_tensor_leaf=on_tensor_leaf,
                on_non_tensor_leaf=on_non_tensor_leaf,
            )
        )
        return buf, result

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
        named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
        all_args: dict[str, object] = {
            n: _CodeExpr(inp.get_name())  # type: ignore[union-attr]
            for n, inp in named_inputs.items()
        }
        for n, v in self._constant_args_dict.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        tensor_flat_params = frozenset(named_inputs.keys())
        sig = self._helion_kernel.signature.parameters
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
        host_fn = self._helion_kernel.name
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
                if self._epilogue_idx_by_param
                else None,
                load_transform=self._codegen_prologue_fusion
                if self._prologue_fused_params
                else None,
                extra_params=extra_params,
            )

        # Collect module-level variable names for uniquification
        # (e.g. constexpr assignments like ``_BLOCK_SIZE_0 = tl.constexpr(32)``).
        module_level_vars: dict[str, str] = {
            target.id: f"{target.id}_{Placeholder.KERNEL_NAME}"
            for node in root.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

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

        return root

    def _codegen_epilogue_fusion(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
        codegen_store: Callable[[ast.expr], ast.AST],
    ) -> ast.expr:
        """Emit per-epilogue index definitions + ``<STORE_OUTPUT_{i}>`` placeholder.

        For non-epilogue params, delegates to ``codegen_store`` unchanged.
        For epilogue params, emits the placeholder and either:
        - Two-store mode: also emits the original store via ``codegen_store``
        - Single-store mode: returns the placeholder name (original store suppressed)
        """
        param_name = state.device_function.tensor_arg(tensor).name
        epilogue_idx = self._epilogue_idx_by_param.get(param_name)
        if epilogue_idx is None:
            return codegen_store(value)

        kernel_val_name = f"_kernel_val_{epilogue_idx}"

        # 1. Assign original value to unique temp variable, upcasting to float32
        #    when output dtype is float16/bfloat16.
        if tensor.dtype in (torch.float16, torch.bfloat16):
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
        #    dim_index_exprs gives us the individual index expressions *before*
        #    stride multiplication (e.g. "(indices_0)[:, None]"), which we assign
        #    to x_epilogue{i}_{d} variables.  These names match what Inductor's
        #    store_output() sets on range tree entries.  The x_ prefix ensures
        #    get_block_shape recognizes them as XBLOCK-sized.
        indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        for d, dim_str in enumerate(indexing.dim_index_exprs):
            state.add_statement(
                ast.parse(
                    f"x_epilogue{epilogue_idx}_{d} = {dim_str}", mode="exec"
                ).body[0]
            )

        # 3. Emit per-epilogue mask alias (unique name avoids cross-epilogue collision).
        mask_str = ast.unparse(indexing.mask_expr) if indexing.has_mask() else "None"
        state.add_statement(
            ast.parse(f"_tile_mask_{epilogue_idx} = {mask_str}", mode="exec").body[0]
        )

        # 4. Emit single placeholder statement.
        state.add_statement(
            ast.Expr(
                value=ast.Name(id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load())
            )
        )

        # 5. Decision: keep or suppress the original tl.store.
        #    - Two-store mode (keep): the output buffer is consumed both by the
        #      fused epilogue AND by other downstream users, so the original
        #      store must be preserved.  Emit the store via codegen_store with
        #      the temp variable as the value.
        #    - Single-store mode (suppress): Inductor's epilogue handles the
        #      store entirely via the <STORE_OUTPUT_{i}> placeholder, so the
        #      original tl.store is not needed.  Return the placeholder name —
        #      run_node sees an ast.Name with 0 users and silently drops it,
        #      while the placeholder was already emitted via add_statement.
        if param_name in self._epilogue_keep_store:
            return codegen_store(ast.Name(id=kernel_val_name, ctx=ast.Load()))

        return ast.Name(id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load())

    def _codegen_prologue_fusion(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        value: ast.expr,
        indexing: SubscriptIndexing,
    ) -> ast.expr:
        """Emit prologue variables + single ``<LOAD_INPUT_{param_name}>`` placeholder.

        Prologue variable definitions are emitted as AST statements (to keep
        referenced variables like ``indices_0`` alive through DCE).  The
        ``<LOAD_INPUT_{param_name}>`` placeholder is expanded at finalize time
        by the hook closure into preamble + result assignment.
        """
        param_name = state.device_function.tensor_arg(tensor).name
        if param_name not in self._prologue_fused_params:
            return value

        # Read prologue variable names from kernel (set by _setup_prologue_hook).
        prologue_vars = self._prologue_vars[param_name]
        result_name = prologue_vars["result"]

        # For multi-output kernels the same input may be loaded at multiple
        # store sites.  The prologue hook is only registered once per input,
        # so only emit the placeholder + variable definitions on the first
        # encounter; subsequent references just reuse the result variable.
        if param_name not in self._prologue_emitted:
            self._prologue_emitted.add(param_name)

            xindex_name = prologue_vars["xindex"]
            xmask_name = prologue_vars["xmask"]

            # Compute linearized offset + mask from SubscriptIndexing
            offset_str = ast.unparse(indexing.index_expr)
            mask_str = (
                ast.unparse(indexing.mask_expr) if indexing.has_mask() else "True"
            )

            # Emit prologue variable definitions as AST statements (prevents DCE of
            # referenced variables like indices_0, indices_1).
            state.add_statement(
                ast.parse(f"{xindex_name} = {offset_str}", mode="exec").body[0]
            )
            state.add_statement(
                ast.parse(f"{xmask_name} = {mask_str}", mode="exec").body[0]
            )

            # Emit single placeholder statement (preamble + result assignment).
            state.add_statement(
                ast.Expr(
                    value=ast.Name(id=f"<LOAD_INPUT_{param_name}>", ctx=ast.Load())
                )
            )

            # Exempt param from DCE only if it has a source buffer — the
            # tensor pointer appears only in finalize_hook string substitutions.
            # Sourceless prologues (e.g. ones_like) are fully inlined by the
            # hook, so the param should be DCE'd away and also removed from
            # the host function signature.
            if param_name in self._prologue_has_source:
                state.device_function.placeholder_args.add(param_name)
            else:
                state.device_function.sourceless_prologue_params.add(param_name)

        # Return variable reference (hook will assign fused value to this name).
        return ast.Name(id=result_name, ctx=ast.Load())


def _flatten_return_ast(
    ast_node: ast.expr | None,
    structured: object,
) -> list[ast.expr | None]:
    """Get the per-leaf AST nodes in DFS order matching build_multi_outputs traversal.

    Walks ``structured`` in the same order as ``build_multi_outputs`` to produce
    a flat list mapping ``leaf_idx`` to the corresponding AST node from the
    kernel's return statement.  Used to extract kernel parameter names
    (``ast.Name`` nodes) and detect symbolic (non-constant) non-tensor returns.
    """
    result: list[ast.expr | None] = []

    def walk(node: ast.expr | None, out: object) -> None:
        if isinstance(out, (list, tuple)):
            elts = node.elts if isinstance(node, (ast.Tuple, ast.List)) else None
            for i, item in enumerate(out):
                walk(elts[i] if elts is not None else None, item)
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
    """Lower a Helion kernel HOP to a ``HelionTemplateBuffer``."""
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox to buffer / ReinterpretView
    _realize = (
        TemplateBuffer.realize_template_input  # pyrefly: ignore[missing-attribute]
    )
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = _realize(tb)

    # Build fake tensors for kernel binding (sympy exprs to concrete ints)
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

    # Derive output structure from bound kernel using inductor-time input layouts.
    # This gives correct strides even when inductor changes input memory layouts.
    host_function = bound.host_function
    assert host_function is not None
    flat_leaves, tree_spec, return_ast = _get_flat_output(host_function)

    if not flat_leaves:
        # No outputs — create still creates the buffer for mutations.
        buf, _ = HelionTemplateBuffer.create(
            realized_inputs=realized,
            structured_outputs=None,
            mutated_input_names=mutated_inputs_list,
            direct_aliases={},
            kernel=kernel,
            bound_kernel=bound,
            constant_args=constant_args,
            autotune_args=tuple(fake_tensors),
        )
        buf.epilogue_fusable_outputs = {}  # pyrefly: ignore[missing-attribute]
        return ()

    # Reconstruct structured output and create MultiOutput nodes.
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Flatten return_ast to index by leaf_idx (same traversal as build_multi_outputs).
    flat_ast = _flatten_return_ast(return_ast, structured)

    has_symbolic_returns = any(
        not isinstance(leaf, torch.Tensor) and not isinstance(flat_ast[i], ast.Constant)
        for i, leaf in enumerate(flat_leaves)
    )

    # Collect the set of tensor proxy ids that are directly stored by the Triton
    # device function.  Only these can be targets for Inductor epilogue fusion.
    # Using id() is safe here because the proxy objects live for the duration of
    # this function call.
    stored_proxy_ids = {
        id(n.args[0].meta["val"])
        for g in host_function.device_ir.graphs
        for n in g.graph.nodes
        if n.op == "call_function" and n.target is helion_memory_ops.store
    }

    # {mo_name: (kernel_param_name | None, proxy_id | None)}
    output_fusion_meta: dict[str, tuple[str | None, int | None]] = {}

    def on_tensor_leaf(
        mo_name: str,
        mo: MultiOutput,
        _indices: list[tuple[type, int]],
        leaf_idx: int,
    ) -> None:
        ast_node = flat_ast[leaf_idx]
        output_fusion_meta[mo_name] = (
            ast_node.id if isinstance(ast_node, ast.Name) else None,
            id(flat_leaves[leaf_idx]),
        )

    buf, result = HelionTemplateBuffer.create(
        realized_inputs=realized,
        structured_outputs=structured,
        mutated_input_names=mutated_inputs_list,
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

    # Compute epilogue_fusable_outputs: param is known + no symbolic returns + the return
    # value is a tensor directly stored by the Triton device function.
    # Using the stored-proxy check (rather than an input-shape heuristic) correctly
    # handles matmul-like kernels whose output shape differs from all inputs, while
    # still excluding reduction outputs (e.g. out.sum(dim=1)) that are computed
    # outside the Triton kernel and therefore cannot be epilogue-fused.
    seen_params: set[str] = set()
    epilogue_fusable_outputs: dict[str, str] = {}
    for mo_name, (param, proxy_id) in output_fusion_meta.items():
        if (
            param is not None
            and not has_symbolic_returns
            and proxy_id in stored_proxy_ids
            and param not in seen_params  # one epilogue per store site
        ):
            epilogue_fusable_outputs[mo_name] = param
            seen_params.add(param)

    efo = epilogue_fusable_outputs
    buf.epilogue_fusable_outputs = efo  # pyrefly: ignore[missing-attribute]
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
