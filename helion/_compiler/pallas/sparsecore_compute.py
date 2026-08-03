"""SparseCore lane mapping on top of Helion's existing FX lowerings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor import ir

from ..ast_extension import expr_from_string
from ..helper_function import CodegenInterface
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from .access import AccessKind
from .sc_base import _CAST_STORE_DTYPES
from .sc_base import SC_LANES
from .sparsecore_access import CachedLoadAccess
from .sparsecore_access import IndirectLoadAccess

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from .sparsecore_program import SparseCoreProgram


def _custom_ops() -> frozenset[object]:
    from ...language._gelu_tanh_approx import _gelu_tanh_approx

    return frozenset(
        {
            _gelu_tanh_approx,
            torch.ops.aten.round.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.tanh.default,
        }
    )


def supports_sparsecore_compute_node(node: torch.fx.Node) -> bool:
    """Whether SC can emit this node."""
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _inductor_lowering_extra
    from ...language._tracing_ops import _mask_to

    if node.op in ("placeholder", "output"):
        return True
    if node.op != "call_function":
        return False
    if node.target in {
        _host_tensor,
        _get_symnode,
        _inductor_lowering_extra,
        _mask_to,
        operator.getitem,
        torch.ops.aten.stack.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.alias.default,
    }:
        return True
    return (
        isinstance(node.meta.get("lowering"), (PointwiseLowering, ReductionLowering))
        or node.target in _custom_ops()
    )


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
    unique_start: int


def chunk_schedule(value_size: int, dtype: torch.dtype) -> list[LaneChunk]:
    if value_size < 1:
        raise NotImplementedError("SparseCore value is empty")
    if dtype is not torch.bfloat16:
        if value_size % SC_LANES:
            raise NotImplementedError(
                f"SparseCore value size {value_size} must be a multiple of {SC_LANES}"
            )
        return [
            LaneChunk(start, SC_LANES, start)
            for start in range(0, value_size, SC_LANES)
        ]
    if value_size < 2 * SC_LANES or value_size % SC_LANES:
        raise NotImplementedError(
            f"SparseCore bf16 value size {value_size} must be a multiple of "
            f"{SC_LANES} and at least {2 * SC_LANES}"
        )
    starts = list(range(0, max(value_size - 31, 1), 32))
    if starts[-1] != value_size - 32:
        starts.append(value_size - 32)
    chunks: list[LaneChunk] = []
    covered = 0
    for start in starts:
        chunks.append(LaneChunk(start, 32, max(covered, start)))
        covered = max(covered, start + 32)
    return chunks


def _value_size(value: torch.Tensor) -> int:
    return math.prod(int(dim) for dim in value.shape[1:]) if value.ndim > 1 else 1


def packed_bf16_segments(
    program: SparseCoreProgram,
    node: torch.fx.Node,
    memo: dict[torch.fx.Node, tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    """Find packed BF16 boundaries inherited by a value."""
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    access = program.access_by_node.get(node)
    if (
        access is not None
        and access.site.kind is AccessKind.LOAD
        and not isinstance(access, CachedLoadAccess)
        and access.layout.logical_dtype is torch.bfloat16
    ):
        result = (access.layout.value_size,)
    else:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and _value_size(value) == 1:
            result = ()
        elif node.target is torch.ops.aten.stack.default:
            values = node.args[0]
            if not isinstance(values, (list, tuple)):
                raise NotImplementedError("SparseCore stack has no value sequence")
            result = tuple(
                segment
                for child in values
                if isinstance(child, torch.fx.Node)
                for segment in packed_bf16_segments(program, child, memo)
            )
        else:
            candidates = [
                segments
                for parent in node.all_input_nodes
                if (segments := packed_bf16_segments(program, parent, memo))
            ]
            if not candidates:
                result = ()
            elif all(candidate == candidates[0] for candidate in candidates[1:]):
                result = candidates[0]
            else:
                raise NotImplementedError(
                    "SparseCore pointwise operands have incompatible packed BF16 layouts"
                )
    memo[node] = result
    return result


def classify_cached_loads(
    program: SparseCoreProgram,
) -> tuple[frozenset[torch.fx.Node], frozenset[torch.fx.Node]]:
    """Cached loads used by packed and ordinary values."""
    packed: set[torch.fx.Node] = set()
    ordinary: set[torch.fx.Node] = set()
    memo: dict[torch.fx.Node, tuple[int, ...]] = {}

    for access in program.stores:
        root = access.site.value_node
        if root is None:
            continue
        result = packed if packed_bf16_segments(program, root, memo) else ordinary
        pending = [root]
        seen: set[torch.fx.Node] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            if isinstance(program.access_by_node.get(node), CachedLoadAccess):
                result.add(node)
            pending.extend(node.all_input_nodes)
    return frozenset(packed), frozenset(ordinary)


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


class _ChunkInterpreter:
    """Evaluate the existing FX graph for one logical lane chunk."""

    def __init__(
        self,
        owner: SparseCoreCompute,
        chunk: LaneChunk,
        part: str,
        reduction_values: dict[torch.fx.Node, ast.AST],
        *,
        entry: int | None = None,
    ) -> None:
        self.owner = owner
        self.cg = owner.local_codegen
        self.chunk = chunk
        self.part = part
        self.entry = entry
        self.env: dict[torch.fx.Node, object] = dict(reduction_values)

    def to_ast(self, value: object) -> ast.AST:
        if isinstance(value, torch.fx.Node):
            value = self.env[value]
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(repr(value))
        raise TypeError(f"SparseCore value has no expression: {value!r}")

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
        self.env[node] = self._lower(node)

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
            LaneChunk(
                local,
                self.chunk.size,
                max(0, self.chunk.unique_start - group * value_size),
            ),
            self.part,
            {
                key: value
                for key, value in self.env.items()
                if isinstance(value, ast.AST)
            },
            entry=self.entry,
        )
        return child.run_until(selected)

    def _lower(self, node: torch.fx.Node) -> object:
        from ...language import memory_ops
        from ...language._tracing_ops import _get_symnode
        from ...language._tracing_ops import _host_tensor
        from ...language._tracing_ops import _inductor_lowering_extra
        from ...language._tracing_ops import _mask_to

        target = node.target
        lowering = node.meta.get("lowering")
        if isinstance(lowering, ReductionLowering):
            return self.owner.vector_reduction_expr(
                node, self.chunk, self.part, self.env
            )
        if target in (_host_tensor, _get_symnode):
            return expr_from_string("0")
        if target is memory_ops.load:
            return self.owner.load_expr(node, self.chunk, self.part, self.entry)
        if target in (memory_ops.store, self.owner.atomic_add_target):
            return None
        if target is _mask_to:
            return self.to_ast(node.args[0])
        if target is _inductor_lowering_extra:
            args = node.args[0]
            if not isinstance(args, (list, tuple)) or len(args) != 1:
                raise NotImplementedError("SparseCore inductor extra has arity != 1")
            return self.to_ast(args[0])
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.alias.default,
        ):
            source = node.args[0]
            source_value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            value = node.meta.get("val")
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
            return self._stack(node)
        if target is operator.getitem:
            if node.args[1] != 0:
                raise NotImplementedError(
                    "SparseCore tuple access is implemented only for element 0"
                )
            return self.to_ast(node.args[0])
        if target is torch.ops.prims.convert_element_type.default:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and value.dtype in (
                torch.int8,
                torch.bool,
                torch.bfloat16,
            ):
                if any(user.target is not memory_ops.store for user in node.users):
                    raise NotImplementedError(
                        "SparseCore staged output casts must feed a store directly"
                    )
                # Output storage performs these casts.
                return self.to_ast(node.args[0])

        custom = self._custom_op(node)
        if custom is not None:
            return custom
        if isinstance(lowering, PointwiseLowering):
            return lowering.codegen(cast("LoweringContext", self), node)
        if lowering is not None:
            return lowering.codegen(cast("LoweringContext", self), node)
        raise NotImplementedError(f"SparseCore has no compute lowering for {target}")

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
        from ...language.atomic_ops import atomic_add

        self.codegen = codegen
        self.local_codegen = _LocalCodegen(self, codegen)
        self.program = program
        self.buffers = buffers
        self.output_buffers = output_buffers
        self.atomic_add_target = atomic_add
        self.reductions = _reductions(program)
        self.lines: list[str] = []
        self.indent = ""
        self.counter = 0

    def new_var(self, prefix: str) -> str:
        self.counter += 1
        return f"_sc_{prefix}{self.counter}"

    def value_chunks(
        self, node: torch.fx.Node, value_size: int, dtype: torch.dtype
    ) -> tuple[list[LaneChunk], bool]:
        segments = packed_bf16_segments(self.program, node)
        if not segments:
            return chunk_schedule(value_size, dtype), False
        if sum(segments) != value_size:
            raise NotImplementedError(
                "SparseCore packed BF16 layout does not cover the logical value"
            )
        result: list[LaneChunk] = []
        offset = 0
        for segment in segments:
            result.extend(
                LaneChunk(
                    offset + chunk.start,
                    chunk.size,
                    offset + chunk.unique_start,
                )
                for chunk in chunk_schedule(segment, torch.bfloat16)
            )
            offset += segment
        return result, True

    def _buffer_ref(self, buffer: str, item: str, start: int, size: int) -> str:
        return f"{buffer}[_sc_q, {item}, pl.ds({start}, {size})]"

    def load_expr(
        self,
        node: torch.fx.Node,
        chunk: LaneChunk,
        part: str,
        entry: int | None,
    ) -> ast.AST:
        access = self.program.access_by_node[node]
        buffer = self.buffers[node]
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError("SparseCore load has no tensor value")
        value_size = _value_size(value)

        if isinstance(access, CachedLoadAccess):
            return self._cached_expr(access.site.tensor, buffer, chunk, part)

        entries = access.stream.elements_per_item if access.stream is not None else 1
        if isinstance(access, IndirectLoadAccess):
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

        if value.dtype is not torch.bfloat16:
            return expr_from_string(ref)
        even = self.new_var("e")
        odd = self.new_var("o")
        self.lines.append(
            f"{self.indent}{even}, {odd} = plsc.unpack({ref}, "
            "format=plsc.PackFormat.INTERLEAVED)"
        )
        return expr_from_string(even if part == "even" else odd)

    def _cached_expr(
        self,
        tensor: torch.Tensor,
        buffer: str,
        chunk: LaneChunk,
        part: str,
    ) -> ast.AST:
        value_size = int(tensor.shape[-1]) if tensor.ndim else 1
        start = chunk.start % value_size
        if tensor.ndim == 1:
            prefix = ""
        elif tensor.ndim == 2:
            group = (chunk.start // value_size) % int(tensor.shape[0])
            prefix = f"{group}, "
        else:
            raise NotImplementedError("SparseCore cached inputs support rank 1 or 2")
        if part in ("even", "odd") and chunk.size == 32:
            half = value_size // 2
            start = start // 2 + (half if part == "odd" else 0)
        return expr_from_string(f"{buffer}[{prefix}pl.ds({start}, {SC_LANES})]")

    def vector_reduction_expr(
        self,
        node: torch.fx.Node,
        chunk: LaneChunk,
        part: str,
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
            interpreter = _ChunkInterpreter(self, chunk, part, inherited, entry=entry)
            expressions.append(ast.unparse(interpreter.run_until(info.source)))
        if info.kind == "sum":
            return expr_from_string("(" + " + ".join(expressions) + ")")
        result = expressions[0]
        for expression in expressions[1:]:
            result = f"jnp.maximum({result}, {expression})"
        return expr_from_string(result)

    def _part_mask(self, chunk: LaneChunk, part: str) -> str | None:
        if chunk.size != 32 or chunk.unique_start <= chunk.start:
            return None
        parity = 0 if part == "even" else 1
        return f"({chunk.start + parity} + 2 * _sc_lane) >= {chunk.unique_start}"

    def emit_scalar_reductions(
        self,
        reduction_values: dict[torch.fx.Node, ast.AST],
        active_nodes: set[torch.fx.Node],
    ) -> None:
        for info in self.reductions:
            if not info.scalar or info.node not in active_nodes:
                continue
            source_value = info.source.meta.get("val")
            if not isinstance(source_value, torch.Tensor):
                raise NotImplementedError("SparseCore reduction source is not a tensor")
            chunks, packed_bf16 = self.value_chunks(
                info.source, info.input_size, source_value.dtype
            )
            neutral = "0.0" if info.kind == "sum" else "-jnp.inf"
            acc = self.new_var("acc")
            init = "jnp.zeros" if info.kind == "sum" else "jnp.full"
            init_args = "" if info.kind == "sum" else "-jnp.inf, "
            self.lines.append(
                f"{self.indent}{acc} = {init}(({SC_LANES},), "
                f"{init_args}dtype=jnp.float32)"
            )
            for chunk in chunks:
                parts: Iterable[str] = ("even", "odd") if packed_bf16 else ("even",)
                for part in parts:
                    interpreter = _ChunkInterpreter(self, chunk, part, reduction_values)
                    expression = ast.unparse(interpreter.run_until(info.source))
                    mask = self._part_mask(chunk, part)
                    if mask is not None:
                        expression = f"jnp.where({mask}, {expression}, {neutral})"
                    combine = "+" if info.kind == "sum" else "jnp.maximum"
                    if combine == "+":
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
        part: str,
        reduction_values: dict[torch.fx.Node, ast.AST],
    ) -> str:
        return ast.unparse(
            _ChunkInterpreter(self, chunk, part, reduction_values).run_until(node)
        )

    def emit_store(
        self,
        store_node: torch.fx.Node,
        reduction_values: dict[torch.fx.Node, ast.AST],
    ) -> None:
        access = self.program.access_by_node[store_node]
        value_node = access.site.value_node
        if value_node is None:
            raise AssertionError("SparseCore store has no value")
        buffer = self.output_buffers[store_node]
        value = value_node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError("SparseCore store value is not a tensor")
        value_size = _value_size(value)
        if value_size == 1:
            chunk = LaneChunk(0, SC_LANES, 0)
            expression = self._value(value_node, chunk, "even", reduction_values)
            if access.layout.logical_dtype in _CAST_STORE_DTYPES:
                expression = f"({expression}).astype(jnp.int32)"
            self.lines.append(f"{self.indent}{buffer}[_sc_item] = {expression}")
            return
        cast_output = access.layout.logical_dtype in _CAST_STORE_DTYPES
        chunks, _ = self.value_chunks(value_node, value_size, value.dtype)
        for chunk in chunks:
            dst = f"{buffer}[_sc_item, pl.ds({chunk.start}, {chunk.size})]"
            even = self._value(value_node, chunk, "even", reduction_values)
            if value.dtype is torch.bfloat16:
                odd = self._value(value_node, chunk, "odd", reduction_values)
                self.lines.append(
                    f"{self.indent}{dst} = plsc.pack({even}, {odd}, "
                    "format=plsc.PackFormat.INTERLEAVED, "
                    "preferred_element_type=jnp.bfloat16)"
                )
            elif cast_output:
                self.lines.append(f"{self.indent}{dst} = ({even}).astype(jnp.int32)")
            else:
                self.lines.append(f"{self.indent}{dst} = {even}")

    def emit_body(self, indent: str, *, store_nodes: set[torch.fx.Node]) -> list[str]:
        self.lines = []
        self.indent = indent
        if any(
            access.layout.logical_dtype is torch.bfloat16
            for access in self.program.accesses
        ):
            self.lines.append(f"{indent}_sc_lane = lax.iota(jnp.int32, {SC_LANES})")
        reduction_values: dict[torch.fx.Node, ast.AST] = {}
        active_nodes: set[torch.fx.Node] = set()

        def visit(node: torch.fx.Node) -> None:
            if node in active_nodes:
                return
            active_nodes.add(node)
            for parent in node.all_input_nodes:
                visit(parent)

        for store_node in store_nodes:
            value_node = self.program.access_by_node[store_node].site.value_node
            if value_node is not None:
                visit(value_node)
        self.emit_scalar_reductions(reduction_values, active_nodes)
        for access in self.program.stores:
            if access.site.node in store_nodes:
                self.emit_store(access.site.node, reduction_values)
        return self.lines
