"""Same-owner warp seeds and temporary raw leases in packed result storage.

Selection owns graph/lifetime facts only. The original stage emitter supplies
actual per-role copy programs, completed fragments and all synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from . import chained_matmul as chain
from .chained_loop_tmem_bridges import _ancestors
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_workspace import _reuse_requests
from .chained_workspace import plan_contraction_workspace
from .physical_use_frontier import PhysicalUseFrontier
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _RegisterBridge
    from .chained_matmul import _ScanInput
    from .chained_root_warp_stage import WarpOperandEmission
    from .warp_specialized_plan import SharedMemoryLayoutPlan


MODE = "packed_register"


def _raw_load(plan: ChainedMatmulPlan, operand: Node) -> Node | None:
    """One original host read, including its pure coordinate/mask recipe."""
    node = operand
    while node.target in chain._VIEWS:
        node = cast("Node", node.args[0])
    if node.target is not memory_ops.load:
        return None
    host = node.args[0]
    if not isinstance(host, Node) or host.target is not _tracing_ops._host_tensor:
        return None
    assert plan.region is not None
    forbidden = {*plan.dots, *plan.region.scans, *plan.region.reductions}
    ancestors = _ancestors(operand)
    if ancestors & forbidden or any(
        not chain._supported(item)
        or item.target is memory_ops.store
        or item.target is memory_ops.load
        and item is not node
        for item in ancestors
    ):
        return None
    value = host.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    storage = value.untyped_storage()
    for store in (*plan.region.stores, plan.store):
        target = cast("Node", store.args[0]).meta.get("val")
        if not isinstance(target, torch.Tensor) or target.untyped_storage() is storage:
            return None
    return node


def _revision(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    # Preserve the no-alias proof as well as the original graph/shape facts.
    return (
        chain._register_bridge_revision(plan),
        tuple(
            (node, node.meta["val"].untyped_storage()._cdata)
            for node in plan.dots[0].graph.nodes
            if node.target is _tracing_ops._host_tensor
            and isinstance(node.meta.get("val"), torch.Tensor)
        ),
    )


@dataclass(frozen=True)
class WarpRegisterPlan:
    source: int
    destination: int
    seed: Node
    lease_stage: int
    lease_role: str
    reuse: tuple[tuple[int, str], ...]
    workspace: SharedMemoryLayoutPlan
    revision: tuple[object, ...]

    def check(self, plan: ChainedMatmulPlan) -> None:
        if self.revision != _revision(plan):
            raise chain._UnsupportedChain("warp register graph changed")

    def seed_lines(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        scans: list[_ScanInput],
    ) -> list[str]:
        from .chained_fragment_epilogue import bind_fragment_expression

        self.check(plan)
        prefix = f"chain_{self.destination}"
        coordinates = (f"{prefix}_seed_row", f"{prefix}_seed_col")
        expression = bind_fragment_expression(
            cg,
            plan,
            boundaries,
            scans,
            plan.dots[self.source],
            self.seed,
            coordinates,
            coordinates,
            f"chain_{self.source}_acc[{prefix}_seed_index]",
            "cutlass.Float32",
            operand_domain=False,
        )
        expression.check()
        rows, columns = plan.shapes[self.destination][:2]
        return [
            f"{prefix}_seed_coords = {prefix}_thr.partition_C(cute.make_identity_tensor(({rows}, {columns})))",
            f"for {prefix}_seed_index in cutlass.range_constexpr(cute.size({prefix}_acc)):",
            f"    {coordinates[0]}, {coordinates[1]} = {prefix}_seed_coords[{prefix}_seed_index]",
            chain._indent(expression.expression.lines),
            f"    {prefix}_acc[{prefix}_seed_index] = cutlass.Float32({expression.value})",
        ]


def plan_warp_register_transport(plan: ChainedMatmulPlan) -> WarpRegisterPlan | None:
    from .chained_warp_result_transport import complete_warp_tiles

    region = plan.region
    if (
        region is None
        or plan.strategy != "warp"
        or plan.loop is not None
        or plan.scans
        or region.scans
        or region.reductions
        or plan.pointwise_cache is not None
        or plan.root_vector_cache is not None
        or plan.root_exports is not None
        or plan.scan_exports
        or plan.root_native is not None
        or plan.row_registers is not None
        or plan.row_parallel is not None
        or plan.completed_image_bytes
        or not complete_warp_tiles(plan)
    ):
        return None
    ancestors = {node: _ancestors(node) for node in region.nodes}
    frontier = PhysicalUseFrontier(tuple(plan.dots[0].graph.nodes))
    for destination, consumer in enumerate(plan.dots):
        seed = consumer.args[2]
        if not isinstance(seed, Node) or seed.meta["val"].dtype != torch.float32:
            continue
        for source in range(destination):
            producer = plan.dots[source]
            if (
                plan.shapes[source][:2] != plan.shapes[destination][:2]
                or producer.meta["val"].dtype != torch.float32
                or producer not in ancestors[seed]
                or any(
                    dot in ancestors[seed] for dot in plan.dots if dot is not producer
                )
                or not frontier.exclusive_use(producer, seed, consumer)
                or _coordinate_path(
                    seed, ("row", "col"), producer, ("row", "col"), ancestors
                )
                is None
            ):
                continue
            # A raw input needed before the seed consumer can occupy result
            # bytes until its original full-reader join. No value is recomputed.
            for lease_stage in range(source + 1, destination):
                for index, role in enumerate(("a", "b")):
                    operand = cast("Node", plan.dots[lease_stage].args[index])
                    if (
                        operand in plan.dots[source].args[:2]
                        or _raw_load(plan, operand) is None
                    ):
                        continue
                    rows, columns, reduction = plan.shapes[lease_stage]
                    height = rows if role == "a" else columns
                    # This bounds either orientation of the original padded
                    # shared view; the actual role receipt checks its cosize.
                    raw_bytes = 2 * (height * reduction + 8 * max(height, reduction))
                    original = plan_contraction_workspace(plan, row_padding=4)
                    requests = tuple(
                        SharedBufferRequest(
                            r.name, r.byte_size, r.alignment, r.live_from, r.live_until
                        )
                        for r in original.regions
                        if r.name != f"chain_{source}_c"
                    )
                    workspace = _reuse_requests(
                        (
                            *requests,
                            SharedBufferRequest(
                                "chain_raw_lease",
                                raw_bytes,
                                128,
                                2 * source,
                                2 * lease_stage + 1,
                            ),
                        )
                    )
                    # Lease selection must fit in existing packed result bytes.
                    if workspace.allocated_bytes > original.allocated_bytes:
                        continue
                    reuse = tuple(
                        (stage, side)
                        for stage in range(1, len(plan.dots))
                        for operand_index, side in enumerate(("a", "b"))
                        if not (
                            side == role and stage in (lease_stage, lease_stage + 1)
                        )
                        and plan.dots[stage].args[operand_index]
                        is plan.dots[stage - 1].args[operand_index]
                        and _raw_load(
                            plan, cast("Node", plan.dots[stage].args[operand_index])
                        )
                        is not None
                        and plan.operand_dtype(stage) == plan.operand_dtype(stage - 1)
                        and (plan.shapes[stage][operand_index], plan.shapes[stage][2])
                        == (
                            plan.shapes[stage - 1][operand_index],
                            plan.shapes[stage - 1][2],
                        )
                    )
                    frontier.check()
                    return WarpRegisterPlan(
                        source,
                        destination,
                        seed,
                        lease_stage,
                        role,
                        reuse,
                        workspace,
                        _revision(plan),
                    )
    frontier.check()
    return None


def _replace_once(lines: list[str], old: tuple[str, ...], new: Sequence[str]) -> None:
    positions = [
        i
        for i in range(len(lines) - len(old) + 1)
        if tuple(lines[i : i + len(old)]) == old
    ]
    if not old or len(positions) != 1:
        raise chain._UnsupportedChain("original warp operand program changed")
    start = positions[0]
    lines[start : start + len(old)] = new


def relocate_operands(
    selected: WarpRegisterPlan,
    plan: ChainedMatmulPlan,
    lines: list[str],
    operands: dict[int, WarpOperandEmission],
    stages: dict[int, tuple[str, ...]],
    bridges: dict[int, _RegisterBridge],
) -> tuple[list[str], dict[int, tuple[str, ...]]]:
    """Move the actual original role once; retain original waits and barriers."""
    selected.check(plan)
    if tuple(operands) != tuple(range(len(plan.dots))) or any(
        not item.matches()
        or len(item.role_lines) != 2
        or tuple(line for role in item.role_lines for line in role) != item.lines
        or item.native_destinations
        for item in operands.values()
    ):
        raise chain._UnsupportedChain("warp operand receipt changed")
    result = list(lines)
    actual = {stage: list(program) for stage, program in stages.items()}
    index = int(selected.lease_role == "b")
    original = operands[selected.lease_stage]
    layout = original.layouts[index]
    program = original.role_lines[index]
    prefix = f"chain_{selected.lease_stage}_{selected.lease_role}"
    if (
        (
            selected.lease_stage in bridges
            and bridges[selected.lease_stage].role == selected.lease_role
        )
        or layout.dtype not in (torch.float16, torch.bfloat16)
        or not original.asynchronous
        or not operands[selected.source].asynchronous
        or not program
        or program[0] != f"{prefix}_ptr = chain_{selected.lease_role}_workspace"
        or stages[selected.source][-1] != "cute.arch.sync_threads()"
        or stages[selected.lease_stage][-1] != "cute.arch.sync_threads()"
    ):
        raise chain._UnsupportedChain("early warp operand lacks original copy/read cut")
    lease = selected.workspace.region("chain_raw_lease")
    elements = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(layout.shape, layout.stride, strict=True)
    )
    if 2 * elements > lease.byte_size or lease.byte_offset % 128:
        raise chain._UnsupportedChain("early warp operand exceeds aligned lease")
    dtype = "cutlass.BFloat16" if layout.dtype == torch.bfloat16 else "cutlass.Float16"
    relocated = (
        f"{prefix}_ptr = cute.recast_ptr(chain_c_workspace + {lease.byte_offset // 4}, dtype={dtype})",
        *program[1:],
    )
    _replace_once(actual[selected.lease_stage], program, ())
    # Insert after the source stage's original operands and before its wait.
    anchor = operands[selected.source].lines
    _replace_once(actual[selected.source], anchor, (*anchor, *relocated))
    # Identical raw source + actual same shared layout; no intervening writer
    # exists between adjacent uses, and the prior stage ends in the full join.
    for stage, role in selected.reuse:
        index = int(role == "b")
        current, previous = operands[stage], operands[stage - 1]
        if stage in bridges and bridges[stage].role == role:
            continue
        before, after = previous.layouts[index], current.layouts[index]
        if before != after or (
            stage - 1 in bridges and bridges[stage - 1].role == role
        ):
            raise chain._UnsupportedChain("reused warp leaf changed layout or owner")
        program = current.role_lines[index]
        prefix = f"chain_{stage}_{role}"
        if (
            len(program) < 2
            or program[0] != f"{prefix}_ptr = chain_{role}_workspace"
            or not program[1].startswith(f"{prefix} = cute.make_tensor({prefix}_ptr,")
        ):
            raise chain._UnsupportedChain(
                "reused leaf lacks original shared allocation"
            )
        _replace_once(actual[stage], program, program[:2])
    for stage, original_stage in stages.items():
        _replace_once(result, original_stage, actual[stage])
    selected.check(plan)
    return result, {stage: tuple(program) for stage, program in actual.items()}
