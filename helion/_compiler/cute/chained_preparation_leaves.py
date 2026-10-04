"""Typed rectangular input images at preparation-DAG publication points."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .chained_matmul import _ancestors
from .chained_matmul import _Expression
from .chained_matmul import _host_shape
from .chained_matmul import _shape
from .chained_matmul import _UnsupportedChain
from .chained_pointwise_residency import _finite_consumers
from .chained_preparation_frame import PreparationAction
from .chained_preparation_frame import PreparationBuffer
from .chained_rectangular_leaf import prove_rectangular_leaf
from .chained_tcgen05 import _layout
from .warp_specialized_plan import SharedBufferRegion
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationFrame
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_rectangular_leaf import RectangularLeafPlan


@dataclass(frozen=True)
class PreparationLeaf:
    node: Node
    name: str
    proof: RectangularLeafPlan
    first_event: int
    last_event: int
    read_events: tuple[int, ...]
    wrapper: dict[str, object]

    @property
    def byte_size(self) -> int:
        return math.prod(self.proof.tile_shape) * self.proof.element_bytes

    def view(self, byte_pointer: str, byte_offset: int) -> list[str]:
        return [
            f"{self.name}_ptr = cute.recast_ptr({byte_pointer} + {byte_offset}, dtype={self.proof.dtype})",
            *_layout(self.name, self.proof.tile_shape, 1, self.proof.dtype),
        ]

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        execution: ChainedExecution,
        barrier: str,
        phase: str,
    ) -> list[str]:
        """Publish the exact masked raw value; both branches advance one phase."""
        from .chained_leaf_schedule import emit_leaf_transfer

        return emit_leaf_transfer(
            self, cg, plan, boundaries, execution, barrier, phase
        ).lines()


def preparation_leaf_candidates(
    cg: GenerateAST, plan: ChainedMatmulPlan, frame: PreparationFrame
) -> tuple[PreparationLeaf, ...]:
    """Find exact raw images consumed at the existing preparation actions.

    Every load keeps its own original logical shape, dtype, indices and mask.
    This is a candidate list, not an allocation or permission to bind a leaf.
    Only complete native tiles with finite logical consumers are supported.
    """
    from ...language import _tracing_ops
    from ...language import memory_ops

    assert plan.loop is not None and plan.region is not None
    materialized: set[Node] = set()
    events: dict[Node, list[int]] = {}
    for action in frame.actions:
        if action.kind in ("mma", "ready"):
            if action.kind == "mma":
                materialized.update(action.nodes)
            continue
        roots = (
            tuple(source for node in action.nodes for source in node.all_input_nodes)
            if action.kind == "fill"
            else action.nodes
        )
        pending, visited = list(roots), set()
        while pending:
            node = pending.pop()
            if node in visited or node in materialized:
                continue
            visited.add(node)
            if (
                node.target is memory_ops.load
                and cast("Node", node.args[0]).target is _tracing_ops._host_tensor
            ):
                events.setdefault(node, []).append(action.event)
            pending.extend(node.all_input_nodes)
        if action.kind in ("collective", "cache"):
            materialized.update(action.nodes)
    boundaries = {*plan.dots, *plan.region.scans, *plan.region.reductions}
    if plan.pointwise_cache is not None:
        boundaries.update(entry.node for entry in plan.pointwise_cache.entries)
    result = []
    for node in plan.region.nodes:
        if node not in events:
            continue
        dtype, shape = node.meta["val"].dtype, _shape(node)
        if (
            dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or len(shape) != 2
            or shape[0] % 8
            or shape[1] * dtype.itemsize % 32
            or any(size > 256 for size in shape)
            or not _finite_consumers(
                node, boundaries, within=frozenset(frame.cut.preparation)
            )
        ):
            continue
        # A read-dependent mask/index cannot be made into a uniform descriptor
        # origin. The syntax proof also rejects any remaining dynamic access.
        if any(
            ancestor.target is memory_ops.load
            for source in node.all_input_nodes
            for ancestor in _ancestors(source)
        ):
            continue
        name = f"chain_leaf_{len(result)}"
        row, column = f"{name}_row", f"{name}_column"
        expression = _Expression(cg, plan, {})
        expression.coordinate_names.update((row, column))
        try:
            expression.value(node, (row, column))
            loaded = [item for item in expression.loaded_inputs if item[0] is node]
            if len(loaded) != 1:
                continue
            _, _, indices, _ = loaded[0]
            mask = expression._load_mask(node, (row, column))
        except _UnsupportedChain:
            continue
        uniform = {
            *expression.origins.values(),
            "chain_loop_begin",
            "chain_loop_end",
        }
        # Integer symbolic host parameters can be used directly in the loop,
        # without a lexical capture. The device-function argument map owns
        # their names and types; arbitrary expression-local names remain out.
        uniform.update(
            argument.name
            for scalar, argument in cg.device_function._expr_args.items()
            if sympy.ask(sympy.Q.integer(scalar)) is True
        )
        uniform.update(
            name
            for captured, name in plan.loop.captures().items()
            if isinstance(captured.meta.get("val"), (int, torch.SymInt))
            or isinstance(captured.meta.get("val"), torch.Tensor)
            and captured.meta["val"].ndim == 0
            and captured.meta["val"].dtype in (torch.int32, torch.int64)
        )
        source = cast("Node", node.args[0])
        tensor = source.meta["val"]
        proof = prove_rectangular_leaf(
            indices,
            expression.definitions,
            row=row,
            column=column,
            uniform_names=uniform,
            tile_shape=(shape[0], shape[1]),
            shape=_host_shape(tensor),
            strides=tuple(tensor.stride()),
            dtype=dtype,
            mask=mask,
        )
        if proof is None:
            continue
        source_name = cg.device_function.tensor_arg(
            tensor, prefer_name=cast("str", source.args[0])
        ).name
        _, write_names = cg.device_function.get_tensor_read_write_names()
        result.append(
            PreparationLeaf(
                node,
                name,
                proof,
                min(events[node]),
                max(events[node]),
                tuple(sorted(set(events[node]))),
                {
                    "kind": "chained_rectangular_leaf_tma",
                    "source_name": source_name,
                    "write_names": sorted(write_names),
                    "shape": proof.shape,
                    "strides": proof.strides,
                    "dtype": str(dtype).removeprefix("torch."),
                    "rows": proof.view_shape[0],
                    "columns": proof.view_shape[1],
                    "tile": proof.tile_shape,
                    "kernel_args": [f"{name}_atom", f"{name}_tensor"],
                },
            )
        )
    return tuple(result)


def insert_preparation_leaf(
    frame: PreparationFrame,
    leaf: PreparationLeaf,
    *,
    capacity: int,
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    append: bool = False,
) -> PreparationFrame | None:
    """Insert one typed raw image and rebind all offsets within a fixed quota.

    The load's completion/publication is a separate event before its first
    expression read. Its storage can be reused only after the last consuming
    action publishes. Original arithmetic, frontier order and cut stay intact.
    """
    event = leaf.first_event
    if (
        not 0 <= event <= leaf.last_event < frame.actions[-1].event
        or not leaf.read_events
        or leaf.read_events[0] != event
        or leaf.read_events[-1] != leaf.last_event
        or leaf.name in {buffer.name for buffer in frame.buffers}
        or not append
        and any(action.kind == "leaf" for action in frame.actions)
    ):
        return None
    # At an old exclusive end equal to the insertion event, storage is already
    # free for the new load. Starts at that event occur after its publication.
    regions = tuple(
        replace(
            region,
            live_from=region.live_from + int(region.live_from >= event),
            live_until=region.live_until + int(region.live_until > event),
        )
        for region in frame.layout.regions
    )
    reservations = tuple(
        SharedBufferRegion(
            binding.candidate.name,
            binding.byte_offset,
            binding.candidate.byte_size,
            binding.candidate.live_from + int(binding.candidate.live_from >= event),
            binding.candidate.live_until + int(binding.candidate.live_until > event),
            128,
        )
        for binding in prepared_groups
    )
    leaf_region = SharedBufferRegion(
        leaf.name, 0, leaf.byte_size, event, leaf.last_event + 2, 128
    )
    # Keep the already selected placements. Repacking a short-lived leaf first
    # can needlessly displace a long-lived collective occupying the same cut.
    # Additional retained images may grow the frame within a caller-owned
    # quota. Append-only placement keeps every existing physical view intact;
    # the final pipeline planner still charges the complete extended slab.
    offset = frame.layout.allocated_bytes if append else 0
    for region in sorted(
        (
            item
            for item in (*regions, *reservations)
            if item.overlaps_lifetime(leaf_region)
        ),
        key=lambda item: item.byte_offset,
    ):
        if offset + leaf.byte_size <= region.byte_offset:
            break
        offset = max(offset, region.byte_end)
    size = max(frame.layout.allocated_bytes, offset + leaf.byte_size)
    if size > capacity:
        return None
    layout = SharedMemoryLayoutPlan(
        (*regions, replace(leaf_region, byte_offset=offset)), size
    )
    actions = tuple(
        replace(
            action,
            event=action.event + int(action.event >= event),
            reads=(*action.reads, leaf.name)
            if action.event in leaf.read_events
            else action.reads,
        )
        for action in frame.actions
    )
    actions = (
        *actions[:event],
        PreparationAction("leaf", event, (leaf.node,), (), None, (), (leaf.name,)),
        *actions[event:],
    )
    return replace(
        frame,
        layout=layout,
        buffers=(
            *frame.buffers,
            PreparationBuffer(
                leaf.name,
                "leaf",
                leaf.node,
                leaf.node.meta["val"].dtype,
                leaf.proof.tile_shape,
            ),
        ),
        actions=actions,
        stages=tuple(
            replace(stage, a=layout.region(stage.a.name), b=layout.region(stage.b.name))
            for stage in frame.stages
        ),
        peak_live_bytes=max(
            sum(
                region.byte_size
                for region in layout.regions
                if region.live_from <= action.event < region.live_until
            )
            for action in actions
        ),
    )
