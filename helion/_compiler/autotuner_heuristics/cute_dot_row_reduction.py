"""Seed: the row-reduction axis of a dot's result stays in-thread.

An online softmax contracts ``q @ k^T`` over head_dim, then reduces the
result over the key tile (``amax`` / ``sum`` along N) and contracts ``p @ v``
over that same key tile.  The generic CuTe default spreads the key tile over
threads, so every row statistic becomes a block-wide grouped reduction
through shared memory with two barriers per element: 194 ms for
examples/fp8_attention.py at (1, 4, 256, 64) on B200 against 2.5 ms with the
key tile in-thread, and at (2, 8, 1024, 64) the default exceeds the
autotuner's benchmark timeout, ending the search in NoConfigFound.  Keeping
that axis as per-thread lanes (``num_threads = 1``) accumulates the row
statistics and the ``p @ v`` running sum in registers; the search found this
first and gained 77x from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...runtime.config import Config
from .common import REDUCTION_TARGET_NAMES
from .common import op_name_parts
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _row_reduced_dot_n_blocks(env: CompileEnvironment, device_ir: DeviceIR) -> set[int]:
    """Blocks that are both a dot's N axis and a sequential device loop whose
    body reduces a value over that axis (its last dimension)."""
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import ReductionLoopGraphInfo

    spec = env.config_spec
    candidates = {
        env.canonical_block_id(fact.n_block_id)
        for fact in spec.matmul_facts
        if fact.n_block_id is not None
    }
    if not candidates:
        return set()
    result: set[int] = set()
    for graph in device_ir.graphs:
        if (
            not isinstance(graph, ForLoopGraphInfo)
            or isinstance(graph, ReductionLoopGraphInfo)
            or len(graph.block_ids) != 1
        ):
            continue
        block_id = env.canonical_block_id(graph.block_ids[0])
        if block_id not in candidates:
            continue
        for node in graph.graph.nodes:
            if node.op != "call_function" or not (
                op_name_parts(node.target) & REDUCTION_TARGET_NAMES
            ):
                continue
            source = node.args[0] if node.args else None
            value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            if isinstance(dims, int):
                dims = [dims]
            if not isinstance(dims, (list, tuple)) or not any(
                isinstance(dim, int) and dim % value.ndim == value.ndim - 1
                for dim in dims
            ):
                continue
            reduced = env.get_block_id(value.shape[-1])
            if reduced is not None and env.canonical_block_id(reduced) == block_id:
                result.add(block_id)
                break
    return result


class CuteDotRowReductionHeuristic(AutotunerHeuristic):
    """``num_threads = 1`` on the dot N axis a row reduction consumes.

    The seed is also the autotune-off default; a later promoted heuristic
    (the flash family, a collective GEMM) still overrides it, since the
    loader takes the last promotion.
    """

    name = "cute_dot_row_reduction"
    backend = "cute"
    promote_seed_to_default = True

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls.get_seed_config(env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        if spec.cute_flash_search_enabled or not spec.supports_config_key(
            "num_threads"
        ):
            return None
        host = device_ir.host_function
        if host is None:
            return None
        with host:
            blocks = _row_reduced_dot_n_blocks(env, device_ir)
        slots = {
            item.block_id
            for item in spec.num_threads
            if env.canonical_block_id(item.block_id) in blocks
        }
        if not slots:
            return None
        return Config.from_dict(
            {"num_threads": _seq_config_list(spec.num_threads, dict.fromkeys(slots, 1))}
        )
