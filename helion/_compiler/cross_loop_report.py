"""Readable reports of a megakernel's cross-loop schedule, on the
``helion.megakernel`` logger (``HELION_LOGS=megakernel``): why each root barrier
exists, and the roots, dependencies and their synchronization, one table per
compiled kernel."""

from __future__ import annotations

import collections
import logging
import os
from typing import TYPE_CHECKING
import weakref

from .host_function import HostFunction

if TYPE_CHECKING:
    from .cross_loop_scheduler import StaticPipelinePlan
    from .device_ir import DeviceIR
    from .tile_dependency import TileDependencyGraph

log = logging.getLogger("helion.megakernel")
_SOURCE_WIDTH = 72
# Reports already logged per traced kernel: one generated again (e.g. compiled
# after its code was printed) logs each once.
_logged: weakref.WeakKeyDictionary[HostFunction, set[str]] = weakref.WeakKeyDictionary()


def _log_once(level: int, text: str) -> None:
    logged = _logged.setdefault(HostFunction.current(), set())
    if text not in logged:
        logged.add(text)
        log.log(level, "%s", text)


class RootNames:
    """Source names of a kernel's roots and accesses."""

    def __init__(self, device_ir: DeviceIR, graph: TileDependencyGraph) -> None:
        self.device_ir = device_ir
        self.accesses = {access.access_id: access for access in graph.accesses}

    def root(self, root: int) -> str:
        """``root <n> [<file>:<first>-<last>]``: the lines of the root's graph and
        of the loops and branches inside it, in the file of its first located
        node."""
        from .device_ir import control_flow_child_graph_ids  # (imports this module)

        locations = []
        pending = [self.device_ir.root_ids[root]]
        while pending:
            for node in self.device_ir.graphs[pending.pop()].graph.nodes:
                if (location := node.meta.get("location")) is not None:
                    locations.append(location)
                pending.extend(control_flow_child_graph_ids(node))
        if not locations:
            return f"root {root}"
        filename = locations[0].filename
        lines = [loc.lineno for loc in locations if loc.filename == filename]
        first, last = min(lines), max(lines)
        span = f"{first}" if last == first else f"{first}-{last}"
        return f"root {root} [{os.path.basename(filename)}:{span}]"

    def access(self, access_id: int) -> str:
        """``<tensor> <stored|loaded> at <line>: <source>``."""
        access = self.accesses[access_id]
        verb = "stored" if access.kind == "store" else "loaded"
        mask = " (masked)" if access.has_explicit_mask else ""
        text = f"{access.tensor_name or 'a tensor'} {verb}{mask}"
        nodes = list(self.device_ir.graphs[access.graph_id].graph.nodes)
        if 0 <= access.graph_node_index < len(nodes) and (
            location := nodes[access.graph_node_index].meta.get("location")
        ):
            source = (location.line or "").strip()
            if len(source) > _SOURCE_WIDTH:
                source = source[: _SOURCE_WIDTH - 3] + "..."
            text += f" at line {location.lineno}: {source}"
        return text


def log_root_barriers(
    plan: StaticPipelinePlan,
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    kernel: str,
) -> None:
    """INFO: each pair of roots a root barrier orders, the dependencies only it
    orders and why no readiness counter does."""
    if not plan.barrier_reasons or not log.isEnabledFor(logging.INFO):
        return
    names = RootNames(device_ir, graph)
    dependencies = {
        dependency.dependency_id: (edge, dependency)
        for edge in graph.edges
        for dependency in edge.access_dependencies
    }
    # (producer root, consumer root) -> producer access -> consumer access -> reasons
    pairs: dict[tuple[int, int], dict[int, dict[int, set[str]]]] = {}
    for (dependency_id, _producer, _consumer), reason in plan.barrier_reasons.items():
        edge, dependency = dependencies[dependency_id]
        stores = pairs.setdefault((edge.producer_root, edge.consumer_root), {})
        loads = stores.setdefault(dependency.producer_access_id, {})
        loads.setdefault(dependency.consumer_access_id, set()).add(reason)
    lines = []
    for (producer, consumer), stores in sorted(pairs.items()):
        through = (producer, consumer) not in plan.root_barrier_edges
        lines.append(
            f"{names.root(consumer)} waits for every task of {names.root(producer)}"
            + (" (through the barriers between them)" if through else "")
        )
        for store, loads in sorted(stores.items()):
            lines.append(f"  {names.access(store)}")
            for load, reasons in sorted(loads.items()):
                lines.append(f"    {names.access(load)}")
                lines += [f"      no readiness counter: {r}" for r in sorted(reasons)]
    header = f"megakernel {kernel} ({plan.dispatch_mode} dispatch) root barriers:"
    _log_once(logging.INFO, "\n".join([header, *lines]))


def log_schedule(
    plan: StaticPipelinePlan,
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    kernel: str,
) -> None:
    """INFO: the kernel's roots, its dependencies and what orders each."""
    if not log.isEnabledFor(logging.INFO):
        return
    names = RootNames(device_ir, graph)
    workers = plan.worker_count
    lines = [
        (
            f"megakernel {kernel}: {len(plan.execution_orders)} roots on {workers} "
            f"workers, {plan.dispatch_mode} dispatch"
        )
    ]
    for root, domain in enumerate(plan.root_domains):
        tags = [
            tag
            for tag, roots in (
                ("trailing", plan.trailing_roots),
                ("continuation", plan.continuation_roots),
                ("every rank waits for it", plan.done_roots),
            )
            if root in roots
        ]
        lines.append(
            f"  {names.root(root)}: {domain.size_expr} tasks"
            + "".join(f", {tag}" for tag in tags)
        )
    orders: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    for kind, counters in (
        ("counter", plan.readiness_counters),
        ("peer counter", plan.peer_counters),
    ):
        for counter in counters:
            arrivals = counter.uniform_arrival_count()
            for consumer in counter.consumers:
                for producer in counter.producers:
                    orders[producer.producer_root, consumer.consumer_root].add(
                        f"{kind} of {counter.readiness_key_domain.size_expr} keys"
                        + (
                            ""
                            if arrivals is None
                            else f" ({arrivals} arrival{'s' * (arrivals != 1)} a key)"
                        )
                        + ("" if consumer.consumer_site_id is None else " in its loop")
                    )
    for pair in plan.root_barrier_edges:
        orders[pair].add("root barrier")
    for pair in plan.peer_edges:
        orders[pair].add("whole-root peer wait")
    for edge in graph.edges:
        pair = edge.producer_root, edge.consumer_root
        transports = {graph.transport(d) for d in edge.access_dependencies}
        how = orders.get(pair, set()) | (transports & {"cluster", "inband", "tagged"})
        lines.append(
            f"  root {edge.producer_root} -> root {edge.consumer_root}"
            f" ({', '.join(sorted(edge.tensor_names)) or 'a tensor'}): "
            + ("; ".join(sorted(how)) or "ordered by the root barriers")
        )
    _log_once(logging.INFO, "\n".join(lines))
