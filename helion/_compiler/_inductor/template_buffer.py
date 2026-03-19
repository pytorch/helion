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

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..output_header import get_needed_import_lines

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from typing import Any
    from typing import Iterable

    from torch._inductor.ir import IRNode

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
        """Generate AST and return a PartialRender.

        Called as the ``render()`` function from the standard
        ``codegen_template_body`` path.
        """
        # 1. Always autotune before AST generation.
        if self._autotune_args:
            self._bound_kernel.ensure_config_exists(self._autotune_args)

        # 2. Generate Triton AST.
        root = self._generate_triton_ast()
        if root is None:
            return PartialRender("", kernel.render_hooks)

        # 3. Compute call args and preamble, store on kernel.
        call_order, constant_repr = self._call_order_and_constant_repr()
        kernel._call_preamble, kernel._call_args = self._build_call_args(
            call_order, constant_repr
        )

        # 4. Store imports on kernel for emit_kernel_override, return
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
    ) -> tuple[list[str], list[str]]:
        """Compute ``(call_preamble, call_args)`` for the kernel invocation."""
        preamble: list[str] = []

        def resolve_param(param_name: str) -> str | None:
            named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
            node = named_inputs.get(param_name)
            if node is None:
                return constant_repr.get(param_name)

            if isinstance(node, ReinterpretView):
                base = node.data.get_name()
                name = f"reinterp_{len(preamble)}"
                preamble.append(
                    f"{name} = reinterpret_tensor("
                    f"{base}, {tuple(node.get_size())}, {tuple(node.get_stride())}, {node.layout.offset})"
                )
                return name

            return node.get_name()  # type: ignore[union-attr]

        call_args: list[str] = [
            resolved
            for param in call_order
            if (resolved := resolve_param(param)) is not None
        ]
        return preamble, call_args

    @classmethod
    def create(
        cls,
        realized_inputs: dict[str, IRNode],
        structured_outputs: object,
        mutated_input_names: list[str],
        direct_aliases: dict[int, IRNode],
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
        all_args: dict[str, object] = {
            n: _CodeExpr(inp.get_name())  # type: ignore[union-attr]
            for n, inp in self._named_inputs.items()  # pyrefly: ignore[missing-attribute]
        }
        for n, v in self._constant_args_dict.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
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

    def _generate_triton_ast(self) -> ast.Module | None:
        """Generate and rename the Triton kernel AST."""
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
        kernel=kernel,
        bound_kernel=bound,
        constant_args=constant_args,
        autotune_args=tuple(fake_tensors),
    )

    buf.epilogue_fusable_outputs = {}  # pyrefly: ignore[missing-attribute]

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
