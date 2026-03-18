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


# Mapping from Inductor reduction type to Triton reduction function.
_TRITON_REDUCE_FN: dict[str, str] = {
    "sum": "tl.sum",
    "max": "tl.max",
    "min": "tl.min",
}

# Identity values for loop-reduction accumulators.
_REDUCE_IDENTITY: dict[str, str] = {
    "sum": "0.0",
    "max": "float('-inf')",
    "min": "float('inf')",
}


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


class _TritonTransformTracer:
    """Replay a Reduction's inner_fn and record pointwise ops as Triton code.

    Used to extract chained pointwise transformations (e.g. relu, abs) from a
    reduction's inner_fn so they can be applied to the template's computed value
    before reducing.

    The tracer intercepts ops.load() to return the input variable name and
    records all subsequent ops as Triton code strings.
    """

    def __init__(self, input_var: str) -> None:
        self.input_var = input_var
        self.lines: list[str] = []
        self._counter = 0

    def _tmp(self) -> str:
        self._counter += 1
        return f"_transform_tmp_{self._counter}"

    # --- Intercepted ops ---

    def load(self, name: object, index: object) -> str:
        return self.input_var

    def to_dtype(
        self, x: str, dtype: object, src_dtype: object = None
    ) -> str:
        import torch

        dtype_map = {
            torch.float32: "tl.float32",
            torch.float16: "tl.float16",
            torch.bfloat16: "tl.bfloat16",
            torch.int32: "tl.int32",
            torch.int64: "tl.int64",
            torch.bool: "tl.int1",
        }
        triton_dtype = dtype_map.get(dtype, "tl.float32")
        t = self._tmp()
        self.lines.append(f"{t} = ({x}).to({triton_dtype})")
        return t

    def constant(self, value: object, dtype: object) -> str:
        return repr(value)

    def relu(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl.where({x} > 0, {x}, 0)")
        return t

    def abs(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.abs({x})")
        return t

    def neg(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = -{x}")
        return t

    def exp(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.exp({x})")
        return t

    def log(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.log({x})")
        return t

    def sqrt(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.sqrt({x})")
        return t

    def rsqrt(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.rsqrt({x})")
        return t

    def sigmoid(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl.sigmoid({x})")
        return t

    def square(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} * {x}")
        return t

    def mul(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} * {y}")
        return t

    def add(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} + {y}")
        return t

    def sub(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} - {y}")
        return t

    def truediv(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} / {y}")
        return t

    def floordiv(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = {x} // {y}")
        return t

    def where(self, cond: str, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl.where({cond}, {x}, {y})")
        return t

    def minimum(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl.minimum({x}, {y})")
        return t

    def maximum(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl.maximum({x}, {y})")
        return t

    def reciprocal(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = 1.0 / {x}")
        return t

    def tanh(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.tanh({x})")
        return t

    def cos(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.cos({x})")
        return t

    def sin(self, x: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = tl_math.sin({x})")
        return t

    def ge(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} >= {y})")
        return t

    def gt(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} > {y})")
        return t

    def le(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} <= {y})")
        return t

    def lt(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} < {y})")
        return t

    def eq(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} == {y})")
        return t

    def ne(self, x: str, y: str) -> str:
        t = self._tmp()
        self.lines.append(f"{t} = ({x} != {y})")
        return t

    def __getattr__(self, name: str) -> object:
        # Fallback for unhandled ops — generate a generic function call
        def handler(*args: object) -> str:
            t = self._tmp()
            args_str = ", ".join(str(a) for a in args)
            self.lines.append(f"{t} = tl.{name}({args_str})")
            return t

        return handler


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
            out_node: object, hint_override: object = None
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
        self._epilogue_reduction_info: dict[
            int, tuple[str, list[object], list[object], object, str | None]
        ] = getattr(kernel, "_epilogue_reduction_info", {})
        # Phase 2 loop reduction: prefix/suffix AST statements to inject
        # around the for-loop that contains the store site.
        self._reduction_loop_prefix: list[ast.AST] = []
        self._reduction_loop_suffix: list[ast.AST] = []
        self._reduction_loop_block_id: int | None = None
        self._reduction_loop_dim_size: int | None = None
        self._handled_reduction_epilogues: set[int] = set()
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

        # 5b. Communicate which reduction epilogues were handled by the template.
        kernel._handled_reduction_epilogues = self._handled_reduction_epilogues

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

        # Phase 2 loop reduction: inject accumulator init before the for-loop
        # and final reduce + store after the for-loop.
        if self._reduction_loop_prefix or self._reduction_loop_suffix:
            self._inject_reduction_loop_stmts(root)

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

    def _inject_reduction_loop_stmts(self, root: ast.Module) -> None:
        """Post-process the AST to convert a grid dim to a loop for Phase 2.

        For Phase 2 reduction epilogue, the reduction dim's grid blocks need to
        be converted to a loop so that partial sums are accumulated correctly.

        This rewrites the inner (device) function to:
        1. Remove pid/offset/indices for the reduction dim from flat body
        2. Wrap the body in a for-loop that iterates over the reduction dim
        3. Add accumulator init before the loop and finalize after
        4. Also rewrites the host function's grid size
        """
        if self._reduction_loop_block_id is None:
            return

        block_id = self._reduction_loop_block_id
        dim_size = self._reduction_loop_dim_size
        offset_var = f"offset_{block_id}"
        pid_var = f"pid_{block_id}"
        block_size_var = f"_BLOCK_SIZE_{block_id}"

        # Find the inner (device) function — first FunctionDef in module
        func_defs = [
            node for node in root.body if isinstance(node, ast.FunctionDef)
        ]
        if not func_defs:
            return
        inner_fn = func_defs[0]

        # Partition the body into:
        # - grid_setup: statements before the reduction dim's offset assignment
        # - kernel_body: statements after (loads, compute, stores)
        grid_setup: list[ast.AST] = []
        kernel_body: list[ast.AST] = []
        found_reduction = False

        for stmt in inner_fn.body:
            stmt_str = ast.unparse(stmt)
            if not found_reduction:
                # Skip the pid_N assignment for the reduction dim
                if isinstance(stmt, ast.Assign) and stmt_str.startswith(f"{pid_var} ="):
                    continue
                # Skip the offset_N assignment for the reduction dim
                if isinstance(stmt, ast.Assign) and stmt_str.startswith(f"{offset_var} ="):
                    found_reduction = True
                    continue
                grid_setup.append(stmt)
            else:
                kernel_body.append(stmt)

        if not found_reduction:
            return

        # Build the loop: for _red_loop_{block_id} in range(0, dim_size, BS):
        loop_var = f"_red_loop_{block_id}"

        # Replace pid_N references in offset_N assignment with loop variable
        # offset_N = pid_N * _BLOCK_SIZE_N -> offset_N = _red_loop_N
        offset_assign = ast.parse(
            f"{offset_var} = {loop_var}", mode="exec"
        ).body[0]

        # Build loop body: offset assignment + indices + rest of kernel body
        loop_body = [offset_assign, *kernel_body]

        # Build the for-loop
        for_loop = ast.parse(
            f"for {loop_var} in range(0, {dim_size}, {block_size_var}):\n    pass",
            mode="exec",
        ).body[0]
        for_loop.body = loop_body

        # Rebuild the function body
        inner_fn.body = (
            grid_setup
            + self._reduction_loop_prefix
            + [for_loop]
            + self._reduction_loop_suffix
        )

        # Now fix the host function's grid size.
        # Find the _launcher call and remove the reduction dim's cdiv from the grid.
        host_fn = func_defs[1] if len(func_defs) > 1 else None
        if host_fn is not None:
            self._fix_host_grid_for_reduction_loop(host_fn, block_id)

    def _fix_host_grid_for_reduction_loop(
        self, host_fn: ast.FunctionDef, block_id: int
    ) -> None:
        """Fix the host function's grid size for Phase 2 reduction loop.

        Removes the reduction dim's cdiv(...) from the grid size computation.
        The grid is typically a tuple like ``(cdiv(M, BS0) * cdiv(N, BS1),)``.
        We need to remove the ``cdiv(N, BS1)`` factor.
        """
        block_size_var = f"_BLOCK_SIZE_{block_id}"

        for node in ast.walk(host_fn):
            # Find the _launcher(...) call
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_launcher" or "_launcher" in node.func.id:
                    # The grid is typically the second argument (a tuple)
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Tuple):
                        grid_tuple = node.args[1]
                        if grid_tuple.elts:
                            grid_expr = grid_tuple.elts[0]
                            # The grid expr is typically BinOp(cdiv1, Mult, cdiv2)
                            new_grid = self._remove_cdiv_factor(
                                grid_expr, block_size_var
                            )
                            if new_grid is not None:
                                grid_tuple.elts[0] = new_grid

    @staticmethod
    def _remove_cdiv_factor(
        expr: ast.expr, block_size_var: str
    ) -> ast.expr | None:
        """Remove a cdiv(..., block_size_var) factor from a multiply expression.

        E.g., ``cdiv(32, BS0) * cdiv(64, BS1)`` with block_size_var='_BLOCK_SIZE_1'
        returns ``cdiv(32, BS0)``.
        """
        expr_str = ast.unparse(expr)
        if block_size_var not in expr_str:
            return None

        # For a BinOp(left, Mult, right), check which side has the block_size_var
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
            left_str = ast.unparse(expr.left)
            right_str = ast.unparse(expr.right)
            if block_size_var in right_str and block_size_var not in left_str:
                return expr.left
            if block_size_var in left_str and block_size_var not in right_str:
                return expr.right
            # Both sides reference it — nested multiply
            left_result = HelionTemplateBuffer._remove_cdiv_factor(
                expr.left, block_size_var
            )
            if left_result is not None:
                expr.left = left_result
                return expr
            right_result = HelionTemplateBuffer._remove_cdiv_factor(
                expr.right, block_size_var
            )
            if right_result is not None:
                expr.right = right_result
                return expr

        # Single cdiv(..., block_size_var) — replace with 1
        if "cdiv" in expr_str and block_size_var in expr_str:
            return ast.Constant(value=1)

        return None

    def _codegen_epilogue_fusion(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
    ) -> ast.expr | None:
        """Emit per-epilogue index definitions + ``<STORE_OUTPUT_{i}>`` placeholder.

        Returns None to suppress the original tl.store (single-store mode),
        or the kernel value name to keep the original store (two-store mode).
        """
        param_name = state.device_function.tensor_arg(tensor).name
        epilogue_idx = self._epilogue_idx_by_param.get(param_name)
        if epilogue_idx is None:
            return value

        # Check if this is a reduction epilogue
        red_info = self._epilogue_reduction_info.get(epilogue_idx)
        if red_info is not None:
            _, pw_ranges, red_ranges, _, _ = red_info
            if (len(red_ranges) == 1 and len(pw_ranges) >= 1) or (
                len(red_ranges) >= 1 and len(pw_ranges) == 0
            ):
                result = self._codegen_reduction_epilogue(
                    state, tensor, subscript, value, extra_mask, epilogue_idx, red_info
                )
                # result == value means the reduction couldn't be fused
                # (e.g. full reduction with non-persistent tiles).
                if result is not value:
                    self._handled_reduction_epilogues.add(epilogue_idx)
                    return result

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
        #      store must be preserved.  Return the temp variable so it is used
        #      as the store value (instead of the original expression).
        #    - Single-store mode (suppress): Inductor's epilogue handles the
        #      store entirely via the <STORE_OUTPUT_{i}> placeholder, so the
        #      original tl.store is redundant.  Return None to suppress it.
        if param_name in self._epilogue_keep_store:
            return ast.Name(id=kernel_val_name, ctx=ast.Load())

        return None

    def _codegen_reduction_epilogue(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
        epilogue_idx: int,
        red_info: tuple[str, list[object], list[object], object, str | None],
    ) -> ast.expr | None:
        """Emit reduction epilogue: reduce the kernel value and store the result.

        Phase 1 (persistent): tile covers full reduction dim. The template
        performs the reduction inline using tl.sum/tl.max/etc.

        Phase 2 (loop): tile does NOT cover full reduction dim. The template
        accumulates partial reductions across loop iterations, then stores.

        For chained reductions (inner_fn is not None), the inner_fn's pointwise
        ops are replayed before reducing using _TritonTransformTracer.
        """
        red_type, pw_ranges, red_ranges, inner_fn, template_buf = red_info
        is_full_reduction = len(pw_ranges) == 0

        if is_full_reduction:
            return self._codegen_full_reduction_epilogue(
                state, tensor, subscript, value, extra_mask,
                epilogue_idx, red_info,
            )

        # Find the reduction axis in the template output tensor.
        reduction_axis = self._find_reduction_axis(
            tensor, subscript, pw_ranges, red_ranges
        )

        # Check if this is Phase 1 (persistent) or Phase 2 (loop).
        # Phase 1: the reduction dim is covered by a full slice (':') or
        #          the tile block_size == dim_size.
        red_subscript = subscript[reduction_axis]
        is_persistent = isinstance(red_subscript, slice)

        if not is_persistent and isinstance(red_subscript, torch.SymInt):
            # Check if block_size == dim_size
            block_id = self._get_block_id_for_subscript(state, red_subscript)
            if block_id is not None:
                cfg = state.config
                dim_size = tensor.shape[reduction_axis]
                if block_id < len(cfg.block_sizes):
                    block_size = cfg.block_sizes[block_id]
                    if isinstance(block_size, int) and isinstance(dim_size, int):
                        is_persistent = block_size >= dim_size

        if not is_persistent:
            # Phase 2: loop reduction — accumulate across tile iterations.
            return self._codegen_loop_reduction_epilogue(
                state, tensor, subscript, value, extra_mask,
                epilogue_idx, red_info, reduction_axis,
            )

        # Phase 1: persistent reduction
        kernel_val_name = f"_kernel_val_{epilogue_idx}"
        reduced_val_name = f"_reduced_val_{epilogue_idx}"

        # Map reduction type to Triton function
        assert red_type in _TRITON_REDUCE_FN, f"unsupported reduction type: {red_type}"
        triton_reduce_fn = _TRITON_REDUCE_FN[red_type]

        # 1. Assign original value to unique temp, upcasting to float32.
        value_str = ast.unparse(value)
        state.add_statement(
            ast.parse(
                f"{kernel_val_name} = ({value_str}).to(tl.float32)",
                mode="exec",
            ).body[0]
        )

        # 2. For chained reductions, replay inner_fn to apply pointwise ops
        #    (e.g. relu) before reducing.
        reduce_input = kernel_val_name
        if inner_fn is not None:
            transform_lines, result_var = self._trace_inner_fn_transform(
                inner_fn, kernel_val_name, pw_ranges, red_ranges
            )
            for line in transform_lines:
                state.add_statement(ast.parse(line, mode="exec").body[0])
            reduce_input = result_var

        # 3. Emit the reduction: _reduced_val_N = tl.sum(reduce_input, axis=red_axis)
        state.add_statement(
            ast.parse(
                f"{reduced_val_name} = {triton_reduce_fn}({reduce_input}, axis={reduction_axis})",
                mode="exec",
            ).body[0]
        )

        # 4. Emit 1D index definitions for non-reduction dims only.
        #    Use raw index variables (without ND broadcast expansion) since the
        #    output is now reduced (fewer dims than the template output).
        pw_dim_idx = 0
        for d, k in enumerate(subscript):
            if d == reduction_axis:
                continue
            # Get the raw index variable for this dimension's block
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    idx_var = state.codegen.index_var(block_id)
                    state.add_statement(
                        ast.parse(
                            f"x_epilogue{epilogue_idx}_{pw_dim_idx} = {idx_var}",
                            mode="exec",
                        ).body[0]
                    )
                    pw_dim_idx += 1

        # 5. Emit mask for reduced shape (1D, non-reduction dims only).
        mask_parts = []
        for d, k in enumerate(subscript):
            if d == reduction_axis:
                continue
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    mask = state.codegen.mask_var(block_id)
                    if mask:
                        mask_parts.append(mask)

        if mask_parts:
            mask_str = " & ".join(mask_parts)
            state.add_statement(
                ast.parse(
                    f"_tile_mask_{epilogue_idx} = {mask_str}",
                    mode="exec",
                ).body[0]
            )
        else:
            state.add_statement(
                ast.parse(
                    f"_tile_mask_{epilogue_idx} = None",
                    mode="exec",
                ).body[0]
            )

        # 6. Emit placeholder.
        state.add_statement(
            ast.Expr(
                value=ast.Name(
                    id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load()
                )
            )
        )

        # 7. Always suppress the original tl.store for reduction epilogues.
        return None

    def _codegen_full_reduction_epilogue(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
        epilogue_idx: int,
        red_info: tuple[str, list[object], list[object], object, str | None],
    ) -> ast.expr | None:
        """Emit full reduction epilogue (e.g. out.sum()).

        All dims are reduced → scalar output. Only supported in persistent
        mode where every tile covers its entire dimension. Emits nested
        reductions from the last axis to the first, e.g.
        ``tl.sum(tl.sum(val, axis=1), axis=0)``.
        """
        red_type, pw_ranges, red_ranges, inner_fn, template_buf = red_info
        n_dims = len(subscript)

        # Verify all dims are persistent (tile covers full dim).
        for d in range(n_dims):
            k = subscript[d]
            if isinstance(k, slice):
                continue  # full slice → persistent
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    cfg = state.config
                    dim_size = tensor.shape[d]
                    if block_id < len(cfg.block_sizes):
                        block_size = cfg.block_sizes[block_id]
                        if isinstance(block_size, int) and isinstance(dim_size, int):
                            if block_size >= dim_size:
                                continue
                # Tile doesn't cover this dim → can't do full reduction
                # (would need atomics). Fall through to non-reduction path.
                return value

        kernel_val_name = f"_kernel_val_{epilogue_idx}"
        reduced_val_name = f"_reduced_val_{epilogue_idx}"

        assert red_type in _TRITON_REDUCE_FN, f"unsupported reduction type: {red_type}"
        triton_reduce_fn = _TRITON_REDUCE_FN[red_type]

        # 1. Assign original value to unique temp, upcasting to float32.
        value_str = ast.unparse(value)
        state.add_statement(
            ast.parse(
                f"{kernel_val_name} = ({value_str}).to(tl.float32)",
                mode="exec",
            ).body[0]
        )

        # 2. For chained reductions, replay inner_fn pointwise ops.
        reduce_input = kernel_val_name
        if inner_fn is not None:
            transform_lines, result_var = self._trace_inner_fn_transform(
                inner_fn, kernel_val_name, pw_ranges, red_ranges
            )
            for line in transform_lines:
                state.add_statement(ast.parse(line, mode="exec").body[0])
            reduce_input = result_var

        # 3. Emit nested reductions from last axis to first.
        #    e.g. for 2D: tl.sum(tl.sum(val, axis=1), axis=0)
        #    Each reduction removes the last axis, so axes 0..k remain
        #    intact after removing axis k+1..n-1.
        cur_input = reduce_input
        for axis in reversed(range(n_dims)):
            if axis == 0:
                # Final reduction → assign to reduced_val
                state.add_statement(
                    ast.parse(
                        f"{reduced_val_name} = {triton_reduce_fn}({cur_input}, axis=0)",
                        mode="exec",
                    ).body[0]
                )
            else:
                tmp_name = f"_reduce_tmp_{epilogue_idx}_{axis}"
                state.add_statement(
                    ast.parse(
                        f"{tmp_name} = {triton_reduce_fn}({cur_input}, axis={axis})",
                        mode="exec",
                    ).body[0]
                )
                cur_input = tmp_name

        # 4. No index vars for 0-dim output.
        # 5. No mask for 0-dim output.
        state.add_statement(
            ast.parse(
                f"_tile_mask_{epilogue_idx} = None",
                mode="exec",
            ).body[0]
        )

        # 6. Emit placeholder.
        state.add_statement(
            ast.Expr(
                value=ast.Name(
                    id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load()
                )
            )
        )

        # 7. Suppress original tl.store.
        return None

    def _codegen_loop_reduction_epilogue(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
        epilogue_idx: int,
        red_info: tuple[str, list[object], list[object], object, str | None],
        reduction_axis: int,
    ) -> ast.expr | None:
        """Phase 2 loop reduction: accumulate partial reductions across tile iterations.

        When the tile block_size < dim_size for the reduction dimension, we:
        1. Initialize an accumulator before the loop starts (saved as prefix)
        2. At each store site: accumulate partial tl.sum into the accumulator
        3. After the loop ends: emit indices, mask, and store placeholder (saved as suffix)

        The prefix/suffix lists are stored on ``self`` and applied by
        ``_generate_triton_ast`` which does an AST post-processing pass to
        move them outside the for-loop.
        """
        red_type, pw_ranges, red_ranges, inner_fn, template_buf = red_info
        acc_name = f"_acc_{epilogue_idx}"
        reduced_val_name = f"_reduced_val_{epilogue_idx}"
        kernel_val_name = f"_kernel_val_{epilogue_idx}"

        # Map reduction type to Triton function and identity value
        assert red_type in _TRITON_REDUCE_FN, f"unsupported reduction type: {red_type}"
        triton_reduce_fn = _TRITON_REDUCE_FN[red_type]
        assert red_type in _REDUCE_IDENTITY, f"unsupported reduction type: {red_type}"
        identity_val = _REDUCE_IDENTITY[red_type]

        # Identify the reduction dim's block_id for grid restructuring
        red_k = subscript[reduction_axis]
        if isinstance(red_k, torch.SymInt):
            red_block_id = self._get_block_id_for_subscript(state, red_k)
            if red_block_id is not None:
                self._reduction_loop_block_id = red_block_id
                self._reduction_loop_dim_size = int(tensor.shape[reduction_axis])

        # Get the non-reduction block size for accumulator shape
        non_red_block_sizes = []
        for d, k in enumerate(subscript):
            if d == reduction_axis:
                continue
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    bs_var = state.device_function.block_size_var(block_id)
                    non_red_block_sizes.append(bs_var or "1")
            elif isinstance(k, slice):
                non_red_block_sizes.append(str(tensor.shape[d]))

        acc_shape = ", ".join(non_red_block_sizes) if non_red_block_sizes else "1"

        # 1. Save accumulator init to prefix (injected before the for-loop).
        self._reduction_loop_prefix.append(
            ast.parse(
                f"{acc_name} = tl.full([{acc_shape}], {identity_val}, tl.float32)",
                mode="exec",
            ).body[0]
        )

        # 2. Emit partial reduction and accumulation at the store site (in-loop).
        value_str = ast.unparse(value)
        state.add_statement(
            ast.parse(
                f"{kernel_val_name} = ({value_str}).to(tl.float32)",
                mode="exec",
            ).body[0]
        )
        if red_type == "sum":
            state.add_statement(
                ast.parse(
                    f"{acc_name} = {acc_name} + {triton_reduce_fn}({kernel_val_name}, axis={reduction_axis})",
                    mode="exec",
                ).body[0]
            )
        else:
            # For max/min, combine old accumulator with new partial result
            state.add_statement(
                ast.parse(
                    f"{acc_name} = {triton_reduce_fn}("
                    f"tl.cat([{acc_name}[None, :], "
                    f"{triton_reduce_fn}({kernel_val_name}, axis={reduction_axis})[None, :]], 0), axis=0)",
                    mode="exec",
                ).body[0]
            )

        # 3. Save suffix stmts (injected after the for-loop).
        self._reduction_loop_suffix.append(
            ast.parse(
                f"{reduced_val_name} = {acc_name}",
                mode="exec",
            ).body[0]
        )

        # Emit 1D index definitions for non-reduction dims
        pw_dim_idx = 0
        for d, k in enumerate(subscript):
            if d == reduction_axis:
                continue
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    idx_var = state.codegen.index_var(block_id)
                    self._reduction_loop_suffix.append(
                        ast.parse(
                            f"x_epilogue{epilogue_idx}_{pw_dim_idx} = {idx_var}",
                            mode="exec",
                        ).body[0]
                    )
                    pw_dim_idx += 1

        # Emit mask
        mask_parts = []
        for d, k in enumerate(subscript):
            if d == reduction_axis:
                continue
            if isinstance(k, torch.SymInt):
                block_id = self._get_block_id_for_subscript(state, k)
                if block_id is not None:
                    mask = state.codegen.mask_var(block_id)
                    if mask:
                        mask_parts.append(mask)

        mask_str = " & ".join(mask_parts) if mask_parts else "None"
        self._reduction_loop_suffix.append(
            ast.parse(
                f"_tile_mask_{epilogue_idx} = {mask_str}",
                mode="exec",
            ).body[0]
        )

        # Store placeholder
        self._reduction_loop_suffix.append(
            ast.Expr(
                value=ast.Name(
                    id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load()
                )
            )
        )

        # 4. Suppress original tl.store
        return None

    @staticmethod
    def _get_block_id_for_subscript(
        state: CodegenState,
        k: torch.SymInt,
    ) -> int | None:
        """Get the block_id for a SymInt subscript element."""
        from ..host_function import HostFunction
        from ..indexing_strategy import BlockSizeOrigin

        symbol = k._sympy_()
        if isinstance(symbol, sympy.Symbol):
            origin = HostFunction.current().expr_to_origin.get(symbol)
            if origin and isinstance(origin.origin, BlockSizeOrigin):
                return origin.origin.block_id
        return None

    @staticmethod
    def _find_reduction_axis(
        tensor: torch.Tensor,
        subscript: list[object],
        pw_ranges: list[object],
        red_ranges: list[object],
    ) -> int:
        """Map from reduction node's (pw_ranges, red_ranges) to tensor axis.

        For sum(dim=1) on [M, N]: pw_ranges=[M], red_ranges=[N].
        For sum(dim=0) on [M, N]: pw_ranges=[N], red_ranges=[M].

        We match dimension sizes from the tensor shape against pw_ranges
        and red_ranges to identify which axis is being reduced.

        Returns the axis index in the subscript list (template output dims).
        """
        n_dims = len(subscript)
        if len(red_ranges) != 1 or n_dims != len(pw_ranges) + 1:
            # Fallback for multi-axis or unexpected layouts
            return n_dims - 1

        red_size = int(red_ranges[0])
        pw_sizes = [int(s) for s in pw_ranges]

        # Build a list of candidate axes where tensor dim matches red_size
        # and remove dims already accounted for by pw_ranges.
        pw_remaining = list(pw_sizes)
        candidates = []
        for axis in range(n_dims):
            dim_size = int(tensor.shape[axis])
            if dim_size == red_size:
                candidates.append(axis)

        # Try to find an axis whose size matches red_size and is NOT needed
        # to satisfy pw_ranges.  Remove pw_ranges matches greedily from
        # non-candidate axes first.
        pw_unmatched = list(pw_sizes)
        for axis in range(n_dims):
            dim_size = int(tensor.shape[axis])
            if axis not in candidates and dim_size in pw_unmatched:
                pw_unmatched.remove(dim_size)

        # Now remove from candidates any that are needed for remaining pw
        for axis in candidates:
            dim_size = int(tensor.shape[axis])
            if dim_size in pw_unmatched:
                pw_unmatched.remove(dim_size)
            else:
                # This candidate is not needed by pw_ranges → it's the reduction axis
                return axis

        # If all candidates were consumed by pw_ranges, fall back to last candidate
        if candidates:
            return candidates[-1]
        return n_dims - 1

    @staticmethod
    def _trace_inner_fn_transform(
        inner_fn: object,
        input_var: str,
        pw_ranges: list[object],
        red_ranges: list[object],
    ) -> tuple[list[str], str]:
        """Replay a reduction's inner_fn to extract pointwise transform ops.

        Uses _TritonTransformTracer to call inner_fn under a custom ops handler
        that intercepts ops.load() → returns input_var, and records all other
        ops as Triton code strings.

        Returns (lines, result_var) where lines are Triton code statements and
        result_var is the variable name holding the transformed value.
        """
        import sympy
        from torch._inductor.virtualized import V

        tracer = _TritonTransformTracer(input_var)
        dummy_index = [sympy.Integer(0)] * len(pw_ranges)
        dummy_rindex = [sympy.Integer(0)] * len(red_ranges)

        with V.set_ops_handler(tracer):
            result = inner_fn(dummy_index, dummy_rindex)

        return tracer.lines, str(result)

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

    def _supports_reduction_epilogue(node: object) -> bool:
        """Check if a specific reduction epilogue can be fused.

        Supports single-axis reductions (e.g. sum(dim=1)) and full
        reductions (e.g. sum()) with simple reduction types (sum, max,
        min). Full reductions require persistent mode (all tiles cover
        the entire tensor) since the loop path would need atomics.
        argmax/argmin, welford reductions, and fused nodes are not
        supported.
        """
        inner = getattr(node, "node", None)
        if inner is None:
            # FusedSchedulerNode (e.g. mean = sum + div): extract the
            # reduction sub-node for capability checking. Only accept
            # if there is exactly 1 reduction sub-node with a supported
            # type and all other sub-nodes are non-reduction (pointwise).
            snodes = getattr(node, "snodes", None)
            if snodes is not None:
                red_snodes = [
                    sn for sn in snodes
                    if hasattr(sn, "is_reduction") and sn.is_reduction()
                ]
                if len(red_snodes) == 1:
                    inner_node = getattr(red_snodes[0], "node", None)
                    if inner_node is not None:
                        inner = inner_node
                        node = red_snodes[0]
            if inner is None:
                return False
        red_type = inner.get_reduction_type()
        if red_type not in ("sum", "max", "min"):
            return False
        pw_ranges, red_ranges = node.get_ranges()  # type: ignore[union-attr]
        if len(red_ranges) >= 1 and len(pw_ranges) == 0:
            # Full reduction (e.g. sum()): only supported when all
            # block_sizes cover the entire tensor (persistent mode).
            # Check the config's block sizes against the reduction
            # ranges (which have one entry per tensor dim).
            bound.ensure_config_exists(tuple(fake_tensors))
            cfg = bound._config
            if cfg is None:
                return False
            for i, dim_size in enumerate(red_ranges):
                if i >= len(cfg.block_sizes):
                    return False
                bs = cfg.block_sizes[i]
                if not (isinstance(bs, int) and isinstance(dim_size, (int, sympy.Integer)) and bs >= int(dim_size)):
                    return False
            return True
        # Single-axis reduction with at least 1 pointwise dim
        return len(red_ranges) == 1 and len(pw_ranges) >= 1

    buf.supports_reduction_epilogue = _supports_reduction_epilogue  # pyrefly: ignore[missing-attribute]
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
