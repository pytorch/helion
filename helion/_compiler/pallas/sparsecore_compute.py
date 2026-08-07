"""SparseCore lane mapping on top of Helion's existing FX lowerings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir

from ..ast_extension import expr_from_string
from ..helper_function import CodegenInterface
from ..inductor_lowering import GraphInterpreter
from ..inductor_lowering import ReductionLowering
from .sc_base import _CAST_STORE_DTYPES
from .sc_base import SC_LANES
from .sparsecore_plan import IndirectLoadPlan

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .sparsecore_program import SparseCoreProgram


class _LocalCodegen(CodegenInterface):
    """Keep shared-lowering temporaries inside the SC item function."""

    def __init__(self, owner: SparseCoreCompute, codegen: GenerateAST) -> None:
        super().__init__(codegen.device_function)
        self.owner = owner
        self.codegen = codegen

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "v") -> ast.Name:
        del dce
        if isinstance(expr, ast.Name):
            return expr
        name = self.owner.new_var(prefix)
        self.owner.lines.append(f"{self.owner.indent}{name} = {ast.unparse(expr)}")
        result = expr_from_string(name)
        assert isinstance(result, ast.Name)
        return result

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is None:
            return
        source = stmt if isinstance(stmt, str) else ast.unparse(stmt)
        self.owner.lines.extend(
            f"{self.owner.indent}{line}" for line in source.splitlines()
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self.codegen, name)


@dataclass(frozen=True)
class LaneChunk:
    start: int
    size: int


def chunk_schedule(value_size: int) -> list[LaneChunk]:
    if value_size < 1:
        raise NotImplementedError("SparseCore value is empty")
    if value_size % SC_LANES:
        raise NotImplementedError(
            f"SparseCore value size {value_size} must be a multiple of {SC_LANES}"
        )
    return [LaneChunk(start, SC_LANES) for start in range(0, value_size, SC_LANES)]


def _value_size(value: torch.Tensor) -> int:
    return math.prod(int(dim) for dim in value.shape[1:]) if value.ndim > 1 else 1


@dataclass(frozen=True)
class _Reduction:
    node: torch.fx.Node
    source: torch.fx.Node
    kind: str
    input_count: int
    input_size: int
    output_size: int

    @property
    def scalar(self) -> bool:
        return self.output_size == 1


def _reductions(program: SparseCoreProgram) -> list[_Reduction]:
    result: list[_Reduction] = []
    for node in program.graph.nodes:
        lowering = node.meta.get("lowering")
        if not isinstance(lowering, ReductionLowering):
            continue
        source: object = node.args[0] if node.args else None
        if isinstance(source, (list, tuple)) and len(source) == 1:
            source = source[0]
        if not isinstance(source, torch.fx.Node):
            raise NotImplementedError("SparseCore reduction needs one FX input")
        source_value = source.meta.get("val")
        output_value = node.meta.get("val")
        if not isinstance(source_value, torch.Tensor) or not isinstance(
            output_value, torch.Tensor
        ):
            raise NotImplementedError("SparseCore reduction values must be tensors")
        reduction = lowering.buffer.data
        assert isinstance(reduction, ir.Reduction)
        ranges = reduction.reduction_ranges
        if len(ranges) != 1 or not isinstance(ranges[0], (int, sympy.Integer)):
            raise NotImplementedError("SparseCore reduction needs one static axis")
        input_count = int(ranges[0])
        input_size = _value_size(source_value)
        output_size = _value_size(output_value)
        if input_size != input_count * output_size:
            raise NotImplementedError(
                "SparseCore reduction must use the first flattened value dimension"
            )
        if lowering.reduction_type not in ("sum", "max"):
            raise NotImplementedError(
                f"SparseCore reduction {lowering.reduction_type!r} is not implemented"
            )
        result.append(
            _Reduction(
                node,
                source,
                lowering.reduction_type,
                input_count,
                input_size,
                output_size,
            )
        )
    return result


class _ChunkInterpreter(GraphInterpreter):
    """Evaluate the existing FX graph for one logical lane chunk."""

    def __init__(
        self,
        owner: SparseCoreCompute,
        chunk: LaneChunk,
        reduction_values: dict[torch.fx.Node, ast.AST],
        *,
        entry: int | None = None,
    ) -> None:
        super().__init__(owner.program.graph, owner.local_codegen)
        self.owner = owner
        self.chunk = chunk
        self.entry = entry
        self.env.update(reduction_values)

    def run_until(self, target: torch.fx.Node) -> ast.AST:
        self._evaluate(target)
        return self.to_ast(target)

    def _evaluate(self, node: torch.fx.Node) -> None:
        if node in self.env:
            return
        from ...language import memory_ops
        from ...language._tracing_ops import _get_symnode
        from ...language._tracing_ops import _host_tensor

        target = node.target
        if target not in (memory_ops.load, _host_tensor, _get_symnode):
            if target is torch.ops.aten.stack.default or isinstance(
                node.meta.get("lowering"), ReductionLowering
            ):
                pass
            else:
                for dependency in node.all_input_nodes:
                    self._evaluate(dependency)
        self.env[node] = self.run_node(node)

    def _stack(self, node: torch.fx.Node) -> ast.AST:
        values = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(values, (list, tuple)) or dim != 1:
            raise NotImplementedError("SparseCore stack is implemented only on dim=1")
        sizes = []
        for value in values:
            if not isinstance(value, torch.fx.Node):
                raise NotImplementedError("SparseCore stack values must be FX tensors")
            fake = value.meta.get("val")
            if not isinstance(fake, torch.Tensor):
                raise NotImplementedError("SparseCore stack value is not a tensor")
            sizes.append(_value_size(fake))
        if len(set(sizes)) != 1:
            raise NotImplementedError("SparseCore stack inputs need equal value sizes")
        value_size = sizes[0]
        group, local = divmod(self.chunk.start, value_size)
        if group >= len(values) or local + self.chunk.size > value_size:
            raise NotImplementedError("SparseCore lane chunk crosses a stack boundary")
        selected = values[group]
        assert isinstance(selected, torch.fx.Node)
        child = _ChunkInterpreter(
            self.owner,
            LaneChunk(local, self.chunk.size),
            {
                key: value
                for key, value in self.env.items()
                if isinstance(value, ast.AST)
            },
            entry=self.entry,
        )
        return child.run_until(selected)

    def run_node(self, n: torch.fx.Node) -> object:
        from ...language import memory_ops
        from ...language._tracing_ops import _get_symnode
        from ...language._tracing_ops import _host_tensor
        from ...language._tracing_ops import _inductor_lowering_extra
        from ...language._tracing_ops import _mask_to

        target = n.target
        lowering = n.meta.get("lowering")
        if isinstance(lowering, ReductionLowering):
            return self.owner.vector_reduction_expr(n, self.chunk, self.env)
        if target in (_host_tensor, _get_symnode):
            return expr_from_string("0")
        if target is memory_ops.load:
            return self.owner.load_expr(n, self.chunk, self.entry)
        if target is memory_ops.store:
            return None
        if target is _mask_to:
            return self.to_ast(n.args[0])
        if target is _inductor_lowering_extra:
            args = n.args[0]
            if not isinstance(args, (list, tuple)) or len(args) != 1:
                raise NotImplementedError("SparseCore inductor extra has arity != 1")
            return self.to_ast(args[0])
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.alias.default,
        ):
            source = n.args[0]
            source_value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            value = n.meta.get("val")
            if (
                isinstance(source_value, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and _value_size(source_value) != _value_size(value)
            ):
                raise NotImplementedError(
                    "SparseCore view cannot change the per-item value size"
                )
            return self.to_ast(source)
        if target is torch.ops.aten.stack.default:
            return self._stack(n)
        if target is operator.getitem:
            if n.args[1] != 0:
                raise NotImplementedError(
                    "SparseCore tuple access is implemented only for element 0"
                )
            return self.to_ast(n.args[0])
        if target is torch.ops.prims.convert_element_type.default:
            value = n.meta.get("val")
            if isinstance(value, torch.Tensor) and value.dtype in (
                torch.int8,
                torch.bool,
            ):
                if any(user.target is not memory_ops.store for user in n.users):
                    raise NotImplementedError(
                        "SparseCore staged output casts must feed a store directly"
                    )
                # Output storage performs these casts.
                return self.to_ast(n.args[0])

        custom = self._custom_op(n)
        if custom is not None:
            return custom
        if lowering is None:
            raise NotImplementedError(
                f"SparseCore has no compute lowering for {target}"
            )
        return super().run_node(n)

    def _custom_op(self, node: torch.fx.Node) -> ast.AST | None:
        target = node.target
        if target is torch.ops.aten.round.default:
            source = ast.unparse(self.to_ast(node.args[0]))
            offset = f"jnp.where({source} >= 0, 8388608.0, -8388608.0)"
            return expr_from_string(
                f"jnp.where(jnp.abs({source}) >= 8388608.0, {source}, "
                f"({source} + {offset}) - {offset})"
            )

        from ...language._gelu_tanh_approx import GELU_TANH_APPROX_KAPPA
        from ...language._gelu_tanh_approx import GELU_TANH_APPROX_LAMBDA
        from ...language._gelu_tanh_approx import _gelu_tanh_approx

        if target is _gelu_tanh_approx:
            source_name = self.cg.lift(
                self.to_ast(node.args[0]), dce=True, prefix="gelu"
            )
            source = source_name.id
            exponent = (
                f"{source} * ({2 * GELU_TANH_APPROX_KAPPA!r} + "
                f"{2 * GELU_TANH_APPROX_LAMBDA!r} * {source} * {source})"
            )
            return expr_from_string(
                f"({source} - {source} / (1.0 + jnp.exp({exponent})))"
            )
        if target is torch.ops.aten.sigmoid.default:
            source = ast.unparse(self.to_ast(node.args[0]))
            return expr_from_string(f"(1.0 / (1.0 + jnp.exp(-({source}))))")
        if target is torch.ops.aten.tanh.default:
            source = ast.unparse(self.to_ast(node.args[0]))
            return expr_from_string(f"(1.0 - 2.0 / (1.0 + jnp.exp(2.0 * ({source}))))")
        return None


class SparseCoreCompute:
    """Render store values using shared Helion compute lowerings."""

    def __init__(
        self,
        codegen: GenerateAST,
        program: SparseCoreProgram,
        buffers: dict[torch.fx.Node, str],
        output_buffers: dict[torch.fx.Node, str],
    ) -> None:
        self.codegen = codegen
        self.local_codegen = _LocalCodegen(self, codegen)
        self.program = program
        self.buffers = buffers
        self.output_buffers = output_buffers
        self.reductions = _reductions(program)
        self.lines: list[str] = []
        self.indent = ""
        self.counter = 0

    def new_var(self, prefix: str) -> str:
        self.counter += 1
        return f"_sc_{prefix}{self.counter}"

    def _buffer_ref(self, buffer: str, item: str, start: int, size: int) -> str:
        return f"{buffer}[_sc_q, {item}, pl.ds({start}, {size})]"

    def load_expr(
        self,
        node: torch.fx.Node,
        chunk: LaneChunk,
        entry: int | None,
    ) -> ast.AST:
        plan = self.program.plan_by_node[node]
        buffer = self.buffers[node]
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError("SparseCore load has no tensor value")
        value_size = _value_size(value)

        entries = plan.stream.elements_per_item if plan.stream is not None else 1
        if isinstance(plan, IndirectLoadPlan):
            entry_size = value_size // entries
            if entry is None and entries != 1:
                raise NotImplementedError(
                    "multi-entry SparseCore gather must be consumed by a reduction"
                )
            item = "_sc_item" if entries == 1 else f"_sc_item * {entries} + {entry}"
            start = chunk.start % entry_size
            ref = self._buffer_ref(buffer, item, start, chunk.size)
        elif entry is not None and value_size == entries:
            return expr_from_string(
                f"{buffer}[_sc_q, _sc_item, pl.ds(0, {SC_LANES})][{entry}]"
            )
        else:
            start = chunk.start % value_size
            ref = self._buffer_ref(buffer, "_sc_item", start, chunk.size)

        return expr_from_string(ref)

    def vector_reduction_expr(
        self,
        node: torch.fx.Node,
        chunk: LaneChunk,
        reduction_values: dict[torch.fx.Node, object],
    ) -> ast.AST:
        info = next(
            reduction for reduction in self.reductions if reduction.node is node
        )
        if info.scalar:
            value = reduction_values.get(node)
            if isinstance(value, ast.AST):
                return value
            raise AssertionError("SparseCore scalar reduction was not computed")
        expressions = []
        inherited = {
            key: value
            for key, value in reduction_values.items()
            if isinstance(value, ast.AST)
        }
        for entry in range(info.input_count):
            interpreter = _ChunkInterpreter(self, chunk, inherited, entry=entry)
            expressions.append(ast.unparse(interpreter.run_until(info.source)))
        if info.kind == "sum":
            return expr_from_string("(" + " + ".join(expressions) + ")")
        result = expressions[0]
        for expression in expressions[1:]:
            result = f"jnp.maximum({result}, {expression})"
        return expr_from_string(result)

    def emit_scalar_reductions(
        self,
        reduction_values: dict[torch.fx.Node, ast.AST],
        active_nodes: set[torch.fx.Node],
    ) -> None:
        for info in self.reductions:
            if not info.scalar or info.node not in active_nodes:
                continue
            acc = self.new_var("acc")
            init = "jnp.zeros" if info.kind == "sum" else "jnp.full"
            init_args = "" if info.kind == "sum" else "-jnp.inf, "
            self.lines.append(
                f"{self.indent}{acc} = {init}(({SC_LANES},), "
                f"{init_args}dtype=jnp.float32)"
            )
            for chunk in chunk_schedule(info.input_size):
                interpreter = _ChunkInterpreter(self, chunk, reduction_values)
                expression = ast.unparse(interpreter.run_until(info.source))
                if info.kind == "sum":
                    self.lines.append(f"{self.indent}{acc} = {acc} + {expression}")
                else:
                    self.lines.append(
                        f"{self.indent}{acc} = jnp.maximum({acc}, {expression})"
                    )
            result = self.new_var("red")
            aggregate = "jnp.sum" if info.kind == "sum" else "jnp.max"
            self.lines.append(
                f"{self.indent}{result} = jnp.full(({SC_LANES},), {aggregate}({acc}))"
            )
            reduction_values[info.node] = expr_from_string(result)

    def _value(
        self,
        node: torch.fx.Node,
        chunk: LaneChunk,
        reduction_values: dict[torch.fx.Node, ast.AST],
    ) -> str:
        return ast.unparse(
            _ChunkInterpreter(self, chunk, reduction_values).run_until(node)
        )

    def emit_store(
        self,
        store_node: torch.fx.Node,
        reduction_values: dict[torch.fx.Node, ast.AST],
    ) -> None:
        plan = self.program.plan_by_node[store_node]
        value_node = plan.access.value_node
        if value_node is None:
            raise AssertionError("SparseCore store has no value")
        buffer = self.output_buffers[store_node]
        value = value_node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError("SparseCore store value is not a tensor")
        value_size = _value_size(value)
        if value_size == 1:
            chunk = LaneChunk(0, SC_LANES)
            expression = self._value(value_node, chunk, reduction_values)
            if plan.layout.logical_dtype in _CAST_STORE_DTYPES:
                expression = f"({expression}).astype(jnp.int32)"
            self.lines.append(f"{self.indent}{buffer}[_sc_item] = {expression}")
            return
        cast_output = plan.layout.logical_dtype in _CAST_STORE_DTYPES
        for chunk in chunk_schedule(value_size):
            dst = f"{buffer}[_sc_item, pl.ds({chunk.start}, {chunk.size})]"
            expression = self._value(value_node, chunk, reduction_values)
            if cast_output:
                self.lines.append(
                    f"{self.indent}{dst} = ({expression}).astype(jnp.int32)"
                )
            else:
                self.lines.append(f"{self.indent}{dst} = {expression}")

    def emit_body(self, indent: str, *, store_nodes: set[torch.fx.Node]) -> list[str]:
        self.lines = []
        self.indent = indent
        reduction_values: dict[torch.fx.Node, ast.AST] = {}
        active_nodes: set[torch.fx.Node] = set()

        def visit(node: torch.fx.Node) -> None:
            if node in active_nodes:
                return
            active_nodes.add(node)
            for parent in node.all_input_nodes:
                visit(parent)

        for store_node in store_nodes:
            value_node = self.program.plan_by_node[store_node].access.value_node
            if value_node is not None:
                visit(value_node)
        self.emit_scalar_reductions(reduction_values, active_nodes)
        for plan in self.program.stores:
            if plan.access.node in store_nodes:
                self.emit_store(plan.access.node, reduction_values)
        return self.lines
