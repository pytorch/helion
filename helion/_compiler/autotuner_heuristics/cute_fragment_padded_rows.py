"""``cute_fragment_padded_rows``: bank-conflict-free row strides for staged tiles.

The CTA-local fragment lowering stages every loaded or computed tile densely
in shared memory.  A tile whose row is a multiple of 128 bytes (a ``[16, 128]``
fp32 slab, a ``[64, 128]`` bf16 slab) puts every row's first element in bank
0, so a column walk -- the transposed read of a ``k.T`` operand, the lane
fragments of a tensor-core ``mma`` -- serializes up to 32 ways.  With the knob
on, such rows are padded by four 32-bit words so a column walk touches 32
distinct banks.  The price is a few percent of shared memory per tile, which
can cost a CTA per SM at the occupancy boundary; the autotuner decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from torch.fx import Node

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_padded_rows"
# A dense row this many bytes long (one full sweep of the 32 banks) starts
# every row in the same bank.
BANK_SWEEP_BYTES = 128


def conflict_prone_row(shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    """Whether a dense tile of *shape* starts every row in the same bank."""
    if len(shape) < 2:
        return False
    row_bytes = shape[-1] * dtype.itemsize
    return row_bytes >= BANK_SWEEP_BYTES and row_bytes % BANK_SWEEP_BYTES == 0


def _has_conflict_prone_tile(env: CompileEnvironment, graphs: list[GraphInfo]) -> bool:
    def static_shape(node: Node) -> tuple[int, ...] | None:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        return tuple(env.size_hint(size) for size in value.shape)

    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function":
                continue
            shape = static_shape(node)
            if shape is not None and conflict_prone_row(shape, node.meta["val"].dtype):
                return True
    return False


class CuteFragmentPaddedRowsHeuristic(AutotunerHeuristic):
    """Register the fragment roots that stage a conflict-prone tile."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_padded_rows_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
                and _has_conflict_prone_tile(env, graphs)
            )
        # Shape/stride assumptions of an admitted root must survive rebinding.
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_padded_rows_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_padded_rows_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_padded_rows_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_padded_rows_search_enabled
    spec.cute_fragment_padded_rows_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_padded_rows_search_enabled = previous_enabled
