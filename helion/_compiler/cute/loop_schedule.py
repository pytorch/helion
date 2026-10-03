"""Graph-planned vector transfers and raw-load schedules for tile loops.

Transfer legality is decided from typed coordinates and dependencies before
scalar emission. Emission consumes that plan directly; no scalar AST is
searched, copied, or rewritten to recover loads, stores, or loop bounds.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import memory_ops
from ...language import tile_ops
from ...language.memory_ops import _cute_combined_mask
from ...language.memory_ops import _cute_index_exprs
from ...language.memory_ops import _cute_scalar_load_expr
from ...language.memory_ops import _cute_scalar_pointer_expr
from ...language.memory_ops import _cute_tensor_dim_size_expr
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import CodegenState

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...runtime.config import Config
    from ..device_ir import ForLoopGraphInfo
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import ThreadTileLayout
    from .fragment import RegisterBuffer
    from .loop_state import TileLoopEmitter


@dataclasses.dataclass(frozen=True)
class TileTransfer:
    node: Node
    tensor: torch.Tensor
    axis_dim: int | None

    @property
    def is_store(self) -> bool:
        return self.node.target is memory_ops.store


@dataclasses.dataclass(frozen=True)
class LoopMemoryPlan:
    transfers: tuple[TileTransfer, ...]
    depth: int
    prefetch: bool
    steps: int | None


def plan_memory(
    info: ForLoopGraphInfo, call: Node, layout: ThreadTileLayout, config: Config
) -> LoopMemoryPlan | None:
    from .loop_state import _index_axis

    if config.config.get("cute_loop_vectorize") is not True:
        return None
    # Ordinary memory lowering owns non-default cache/eviction policies.
    if any(config.load_eviction_policies):
        return None
    transfers = []
    for node in info.graph.nodes:
        if node.target not in (memory_ops.load, memory_ops.store):
            continue
        source, indices = node.args[:2]
        mask = node.args[3 if node.target is memory_ops.store else 2]
        if node.target is memory_ops.load and node.args[3] is not None:
            return None
        if mask is not None or not isinstance(source, Node):
            return None
        tensor = source.meta.get("val")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
        ):
            return None
        if not isinstance(indices, (tuple, list)):
            return None
        # Direct typed coordinates have no dependency on a carried value or a
        # loaded index. Gather/conditional addresses retain ordinary lowering.
        axes = [_index_axis(index) for index in indices]
        if any(
            type(index) is not int and axis is None
            for index, axis in zip(indices, axes, strict=True)
        ):
            return None
        dims = [
            dim
            for dim, (index, axis) in enumerate(zip(indices, axes, strict=True))
            if axis == layout.block_id
            and isinstance(index, Node)
            and index.target is not tile_ops.tile_begin
        ]
        if len(dims) > 1:
            return None
        transfers.append(TileTransfer(node, tensor, dims[0] if dims else None))
    mode = config.config.get("cute_loop_load_schedule", "current")
    assert isinstance(mode, str)
    begins, ends = call.args[1:3]
    assert isinstance(begins, (tuple, list)) and isinstance(ends, (tuple, list))
    begin, end = begins[0], ends[0]
    steps = end if begin == 0 and type(end) is int and 0 < end < 2**31 else None
    depth = int(mode[-1]) if mode != "current" and steps is not None else 1
    prefetch = depth > 1 and mode.startswith("prefetch")
    payload = sum(
        transfer.tensor.element_size()
        * (1 if transfer.axis_dim is None else layout.fragment_elements)
        for transfer in transfers
        if not transfer.is_store
    )
    if prefetch and payload * depth > 128:
        depth, prefetch = 1, False
    return LoopMemoryPlan(tuple(transfers), depth, prefetch, steps)


@contextlib.contextmanager
def _scope(emitter: TileLoopEmitter, statement: ast.For | ast.If) -> Iterator[None]:
    ast.fix_missing_locations(statement)
    emitter.cg.add_statement(statement)
    with emitter.cg.set_statements(cast("list[ast.AST]", statement.body)):
        yield


def _transfer(
    emitter: TileLoopEmitter, transfer: TileTransfer, buffer: RegisterBuffer
) -> None:
    if transfer.axis_dim is None:
        _transfer_group(emitter, transfer, buffer)
    else:
        with emitter.groups():
            _transfer_group(emitter, transfer, buffer)


def _transfer_group(
    emitter: TileLoopEmitter, transfer: TileTransfer, buffer: RegisterBuffer
) -> None:
    """Emit one typed transfer with the ordinary index/mask semantics."""
    cg, layout = emitter.cg, emitter.layout
    graph = emitter.graph(emitter.plan.loop)
    # Only shape metadata of loop inputs can occur in admitted addresses.
    for placeholder in emitter.plan.loop.graph.find_nodes(op="placeholder"):
        graph.env[placeholder] = expr_from_string("0")
    node = transfer.node
    tensor = transfer.tensor
    indices = node.args[1]
    assert isinstance(indices, (list, tuple))
    proxy_indices = [
        index.meta["val"] if isinstance(index, Node) else index for index in indices
    ]
    ast_indices = [
        graph.emit(index) if isinstance(index, Node) else index for index in indices
    ]
    state = CodegenState(
        cg,
        node,
        graph.env,
        [tensor, proxy_indices],
        [None, ast_indices],
    )
    name = emitter.df.tensor_arg(tensor).name
    dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    coordinates = _cute_index_exprs(
        state,
        proxy_indices,
        ast_indices,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    mask = _cute_combined_mask(state, proxy_indices, None, tensor=tensor)
    slot = "0" if transfer.axis_dim is None else layout.fragment_index

    def scalar() -> None:
        if transfer.is_store:
            statement = statement_from_string(
                f"{_cute_scalar_pointer_expr(name, coordinates)}.store({buffer.name}[{slot}])"
            )
            if mask is not None:
                statement = statement_from_string(
                    f"if {mask}:\n    {{store}}", store=statement
                )
        else:
            value = _cute_scalar_load_expr(name, coordinates, tensor.dtype)
            if mask is not None:
                value = f"({value} if {mask} else {dtype}(0))"
            statement = statement_from_string(f"{buffer.name}[{slot}] = {value}")
        cg.add_statement(statement)

    def fallback() -> None:
        if transfer.axis_dim is None:
            scalar()
        else:
            with emitter.vector_elements():
                scalar()

    dim = transfer.axis_dim
    # Match the ordinary vector transport's 128-bit copy-atom limit.
    if (
        dim is None
        or tensor.stride(dim) != 1
        or layout.elements * tensor.element_size() > 16
    ):
        fallback()
        return
    base_coordinates = list(coordinates)
    base_coordinates[dim] = layout.base_index_var
    pointer = emitter.df.new_var("loop_pointer", dce=False)
    vector = emitter.df.new_var("loop_vector", dce=False)
    atom = emitter.df.new_var("loop_copy", dce=False)
    alignment = layout.elements * tensor.element_size()
    cg.add_statement(f"{pointer} = {_cute_scalar_pointer_expr(name, base_coordinates)}")
    cg.add_statement(
        f"{atom} = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy={alignment * 8})"
    )
    bounds = []
    for axis, coordinate in enumerate(base_coordinates):
        size = _cute_tensor_dim_size_expr(state, tensor, axis)
        extent = layout.elements if axis == dim else 1
        bounds.extend((f"({coordinate} >= 0)", f"({coordinate} + {extent} <= {size})"))
    bounds.extend(
        (
            f"({name}.layout.stride[{dim}] == 1)",
            f"({pointer}.toint() % {alignment} == 0)",
        )
    )
    if mask is not None:
        # Physical tensor bounds alone do not imply logical tile validity:
        # explicit subranges and surplus threads can have stricter masks.
        # Evaluate the ordinary predicate in the assigned element context.
        valid = emitter.df.new_var("loop_all_valid", dce=False)
        cg.add_statement(f"{valid} = cutlass.Boolean(True)")
        with emitter.vector_elements():
            cg.add_statement(f"{valid} = {valid} & ({mask})")
        bounds.append(valid)
    branch = create(
        ast.If, test=expr_from_string(" & ".join(bounds)), body=[], orelse=[]
    )
    with _scope(emitter, branch):
        cg.add_statement(
            f"{vector} = cute.make_tensor({pointer}.align({alignment}), cute.make_layout(({layout.elements},), stride=(1,)))"
        )
        local = buffer.name
        if layout.group_extent > 1:
            local = emitter.df.new_var("loop_group_buffer", dce=False)
            cg.add_statement(
                f"{local} = cute.make_tensor({buffer.name}.iterator + {layout.group_var} * {layout.elements}, cute.make_layout(({layout.elements},), stride=(1,)))"
            )
        source, target = (local, vector) if transfer.is_store else (vector, local)
        cg.add_statement(f"cute.copy({atom}, {source}, {target})")
    with cg.set_statements(cast("list[ast.AST]", branch.orelse)):
        fallback()


def emit_memory_loop(
    emitter: TileLoopEmitter, loop: DeviceLoopState, plan: LoopMemoryPlan
) -> None:
    cg, df = emitter.cg, emitter.df
    loads, stores = {}, {}
    for transfer in plan.transfers:
        buffer = emitter.allocate(
            "loop_store" if transfer.is_store else "loop_load",
            transfer.tensor.dtype,
            1 if transfer.axis_dim is None else emitter.layout.fragment_elements,
        )
        (stores if transfer.is_store else loads)[transfer.node] = buffer

    def fetch(buffers: dict[Node, RegisterBuffer]) -> None:
        for transfer in plan.transfers:
            if not transfer.is_store:
                _transfer(emitter, transfer, buffers[transfer.node])

    def consume() -> None:
        emitter.update(loads, stores)
        for transfer in plan.transfers:
            if transfer.is_store:
                _transfer(emitter, transfer, stores[transfer.node])

    if plan.depth == 1:
        with cg.add_device_loop(loop):
            fetch(loads)
            consume()
        return

    assert plan.steps is not None
    steps, depth = plan.steps, plan.depth
    block = emitter.plan.loop.block_ids[0]
    index = loop.strategy.index_var(block)
    offset = loop.strategy.offset_var(block)

    def set_step(expression: str) -> None:
        cg.add_statement(f"{index} = {expression}")
        if offset != index:
            cg.add_statement(f"{offset} = {index}")
        if mask := loop.strategy.mask_var(block):
            cg.add_statement(f"{mask} = ({index} >= 0) & ({index} < {steps})")

    slots = (
        [
            {
                node: emitter.allocate(
                    "loop_prefetch", value.type.dtype, value.type.elements
                )
                for node, value in loads.items()
            }
            for _ in range(depth)
        ]
        if plan.prefetch
        else []
    )

    scheduled: list[ast.AST] = []
    with cg.set_statements(scheduled), cg.bind_device_loop(loop):
        if plan.prefetch:
            for slot in range(min(steps, depth)):
                set_step(f"cutlass.Int32({slot})")
                fetch(slots[slot])
        group = df.new_var("loop_group", dce=False)
        group_base = df.new_var("loop_group_base", dce=False)
        outer = create(
            ast.For,
            target=ast.Name(id=group, ctx=ast.Store()),
            iter=expr_from_string(
                f"range(cutlass.Int32(0), cutlass.Int32({(steps + depth - 1) // depth}))"
            ),
            body=[],
            orelse=[],
            type_comment=None,
        )
        with _scope(emitter, outer):
            cg.add_statement(f"{group_base} = {group} * cutlass.Int32({depth})")
            for slot in range(depth):
                current = f"{group_base} + cutlass.Int32({slot})"
                guard = create(
                    ast.If,
                    test=expr_from_string(
                        f"{group_base} < cutlass.Int32({steps - slot})"
                    ),
                    body=[],
                    orelse=[],
                )
                with (
                    _scope(emitter, guard)
                    if steps % depth
                    else contextlib.nullcontext()
                ):
                    set_step(current)
                    if plan.prefetch:
                        for node, value in loads.items():
                            if value.type.elements == 1:
                                cg.add_statement(
                                    f"{value.name}[0] = {slots[slot][node].name}[0]"
                                )
                            else:
                                with emitter.elements():
                                    emitter.write(
                                        value,
                                        slots[slot][node].element(
                                            emitter.layout.fragment_index
                                        ),
                                    )
                        future_guard = create(
                            ast.If,
                            test=expr_from_string(
                                f"{group_base} < cutlass.Int32({steps - depth - slot})"
                            ),
                            body=[],
                            orelse=[],
                        )
                        with _scope(emitter, future_guard):
                            set_step(f"{group_base} + cutlass.Int32({slot + depth})")
                            fetch(slots[slot])
                        set_step(current)
                    else:
                        fetch(loads)
                    consume()
    cg.statements_stack[-1].extend(loop.outer_prefix)
    cg.statements_stack[-1].extend(scheduled)
    cg.statements_stack[-1].extend(loop.outer_suffix)
