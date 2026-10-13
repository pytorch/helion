"""Same-attempt authority for a streamed, exclusive root C-to-A bridge.

The root scheduler still owns all input publication, waits and stage order.
This record does not discover a bridge or authorize arbitrary TMEM aliasing.
Its accepted bridge is the original coordinate/exclusive-use proof; the actual
expression is lowered once, at the original after-wait bridge location.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .chained_tmem_snapshots import OrderedSnapshotAccess
from .chained_tmem_snapshots import PackedSnapshotPanels
from .chained_tmem_snapshots import emit_streamed_snapshot

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def _facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    from .chained_pipeline_storage import _freeze

    return (
        chain._register_bridge_revision(plan),
        _freeze(plan.axes),
        tuple(
            (
                node,
                _freeze(node.args),
                _freeze(node.kwargs),
                (
                    _freeze(value.shape),
                    _freeze(value.stride()),
                    _freeze(value.storage_offset()),
                    value.dtype,
                    value.device,
                )
                if isinstance(value := node.meta.get("val"), torch.Tensor)
                else _freeze(value),
                node.meta.get("lowering"),
            )
            for node in plan.dots[0].graph.nodes
        ),
    )


def _scan_facts(scans: tuple[chain._ScanInput, ...]) -> object:
    from .chained_pipeline_storage import _freeze

    return _freeze(tuple(vars(scan) for scan in scans))


def _transport_facts(
    bridge: chain._RegisterBridge, panels: PackedSnapshotPanels
) -> tuple[object, ...]:
    """Copy immutable facts, rather than retaining mutable descriptor aliases."""
    return (
        bridge.role,
        bridge.lines,
        bridge.stage,
        bridge.source,
        bridge.operand,
        bridge.revision,
        bridge.epilogue_facts(),
        panels.shape,
        panels.source_offset,
        panels.destination_offset,
        panels.allocation_columns,
        None if panels.ordered is None else tuple(vars(panels.ordered).items()),
    )


@dataclass(frozen=True)
class RootSnapshot:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    bridge: chain._RegisterBridge
    panels: PackedSnapshotPanels
    facts: tuple[object, ...]
    scans: object
    boundaries: tuple[tuple[Node, str], ...]
    options: object
    _selection: tuple[object, ...] = field(repr=False)

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        scans: tuple[chain._ScanInput, ...],
    ) -> bool:
        from .chained_pipeline_storage import _freeze

        return (
            self.codegen is cg
            and self.plan is plan
            and self._selection
            == (
                self.codegen,
                self.plan,
                _transport_facts(self.bridge, self.panels),
                self.facts,
                self.scans,
                self.boundaries,
                self.options,
            )
            and self.facts == _facts(plan)
            and self.scans == _scan_facts(scans)
            # The scheduler's synthetic source boundary is overridden by the
            # exact fragment binding. No other new/changed image is accepted.
            and self.boundaries
            == tuple(
                (node, name)
                for node, name in boundaries.items()
                if node is not self.bridge.source
            )
            and self.options == _freeze(cg.device_function.config.config)
        )

    def render(
        self,
        expression: chain._Expression,
        coords: tuple[str, str],
        prefix: str,
        previous: str,
        dtype: str,
        value: str,
    ) -> list[str]:
        source = self.bridge.source
        expected = f"{previous}_values[{prefix}_index]"
        if (
            not self.matches(
                expression.cg,
                expression.plan,
                expression.boundaries,
                tuple(expression.scan_inputs),
            )
            or expression.fragments.get(source) != (coords, expected)
            or expression.memo.get((source, coords)) != expected
            or any(node is source and key != coords for node, key in expression.memo)
        ):
            raise chain._UnsupportedChain("streamed root snapshot expression changed")
        return emit_streamed_snapshot(
            self.panels,
            prefix,
            previous,
            dtype,
            coords,
            expression.lines,
            value,
            execution=ChainedExecution(128),
        )


def plan_root_snapshot(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bridge: chain._RegisterBridge,
    boundaries: dict[Node, str],
    scans: tuple[chain._ScanInput, ...],
) -> RootSnapshot:
    from .chained_pipeline_storage import _freeze

    shape = plan.shapes[0][:2]
    if (
        plan.loop is not None
        or plan.threads != 128
        or len(plan.dots) != 2
        or bridge.stage != 1
        or bridge.role != "a"
        or bridge.source is not plan.dots[0]
        or bridge.operand is not plan.dots[1].args[0]
        or bridge.revision != chain._register_bridge_revision(plan)
        or chain._shape(bridge.source) != shape
        or chain._shape(bridge.operand) != shape
        or bridge.source.meta["val"].dtype != torch.float32
        or plan.operand_dtype(1) not in (torch.bfloat16, torch.float16)
        or shape != (plan.shapes[1][0], plan.shapes[1][2])
    ):
        raise chain._UnsupportedChain("unsupported streamed root snapshot")
    columns = max(n for _, n, _ in plan.shapes)
    allocation = max(32, 2 ** ((columns * 2 - 1).bit_length()))
    try:
        panels = PackedSnapshotPanels(
            shape, 0, 0, allocation, OrderedSnapshotAccess(shape, 0, 0, allocation)
        )
    except ValueError as error:
        raise chain._UnsupportedChain(str(error)) from error
    facts = _facts(plan)
    scan_facts = _scan_facts(scans)
    bindings = tuple(boundaries.items())
    options = _freeze(cg.device_function.config.config)
    witness = (
        cg,
        plan,
        _transport_facts(bridge, panels),
        facts,
        scan_facts,
        bindings,
        options,
    )
    return RootSnapshot(
        cg, plan, bridge, panels, facts, scan_facts, bindings, options, witness
    )
