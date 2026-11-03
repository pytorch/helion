"""``cute_fragment_bounded_gather``: the complete root proof of a logical-coordinate gather.

The knob's coverage is the ``BOUNDED_GATHER`` row of
``cute.autotune.fragment_knobs``; the row and the dependent warp-scan and
register-snapshot projections call this proof through the module so a test
can observe it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cute_fragment_common import fragment_root_regions

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

KEY = "cute_fragment_bounded_gather"


def bounded_gather_roots(
    env: CompileEnvironment, device_ir: DeviceIR, *, allow_unbound: bool = False
) -> frozenset[int]:
    """The complete root proof, shared by explicit dependent capabilities.

    Discovery may use registered live alias facts before binding snapshots them.
    Code generation and other callers keep requiring immutable bound facts.
    """
    from ...bounded_gather import prove_gather
    from ...computed_fragment import computed_fragment_supported
    from ...selection.topk import match_topk_root

    roots = set()
    host = device_ir.host_function
    assert host is not None
    with host:
        for root, graphs in fragment_root_regions(device_ir):
            if (
                match_topk_root(
                    graphs,
                    noncanonical_block_ids=device_ir.noncanonical_task_origin_block_ids,
                )
                is not None
            ):
                continue
            if not any(
                prove_gather(env, node, graphs=graphs, allow_unbound=allow_unbound)
                is not None
                for info in graphs
                for node in info.graph.nodes
            ):
                continue
            if computed_fragment_supported(
                env, graphs, bounded_gather_owned=True, allow_unbound=allow_unbound
            ):
                roots.add(root)
    return frozenset(roots)
