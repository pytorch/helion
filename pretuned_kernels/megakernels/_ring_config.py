"""The config fields of a gluon megakernel that follow from its source and the
tensors it streams through tensor descriptors (the TMA ring)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from collections.abc import Collection

    from helion.runtime.kernel import BoundKernel


def ring_config_fields(
    bound: BoundKernel,
    loads: Collection[str],
    stores: Collection[str] = (),
    warp_specialize: Literal["roots", "k_loops"] = "roots",
) -> dict[str, list[object]]:
    """``indexing`` and ``range_warp_specializes`` for ``bound``.

    ``indexing``: tensor descriptors for the loads of the tensors named in
    ``loads`` and the stores of those in ``stores`` (host argument names),
    pointers for every other memory op (``config_spec.memory_op_facts``).

    ``range_warp_specializes``, with ``warp_specialize``:

    * ``"roots"``: the roots whose dots read a descriptor-loaded operand
      (their persistent task loops split into load, MMA and epilogue
      partitions);
    * ``"k_loops"``: the single-dimension loop nested first in each
      three-dimensional root (GPT-OSS: an MMA partition for each projection /
      expert K loop, so a task's epilogue overlaps the next task's MMAs, and
      for the verify layer's attention key loop).
    """
    spec = bound.config_spec
    facts = spec.memory_op_facts
    count = len(spec.default_config().config["indexing"])
    ring = {
        fact.indexing_index
        for fact in facts
        if fact.tensor_name in (loads if fact.kind == "load" else stores)
    }
    groups = [tuple(group.block_ids) for group in spec.range_warp_specialize]
    if warp_specialize == "roots":
        roots = {
            spec.kernel_grid_fact.group_for_graph(fact.graph_id)
            for fact in facts
            if fact.matmul_operand is not None and fact.tensor_name in loads
        }
        chosen = [group in roots for group in groups]
    else:
        chosen = [
            len(group) == 1 and i > 0 and len(groups[i - 1]) == 3
            for i, group in enumerate(groups)
        ]
    return {
        "indexing": [
            "tensor_descriptor" if index in ring else "pointer"
            for index in range(count)
        ],
        "range_warp_specializes": [True if c else None for c in chosen],
    }
