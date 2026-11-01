"""The Metal autotuner's seeds for row grids, derived from structure.

A row grid whose rows run tile passes over one extent
(``sibling_row_loop_blocks``) is seeded with every pass as ``T`` threads by
``V`` contiguous lanes:

- ``V`` is one 16-byte load per thread of the widest element a pass reduction
  records (8 elements of fp16/bf16, 4 of fp32), CuTe's tile-vec vector width;
- ``T`` is one SIMD group, whose reductions are barrier-free ``simd_*``
  collectives, or four.

Reaching that layout from the default config takes a larger pass block and a
pass thread count other than 0, 1 or the fragment's maximum in one move,
which the search's neighbours never make.

A pass reduced after its loop needs one element per thread
(``MetalBlockReductionStrategy._needs_loop_carried_accumulator``), so it is
seeded at ``V = 1`` across the whole thread budget instead, its row block at
the grid floor.  Otherwise the row block fills a threadgroup of four SIMD
groups (CuTe's ``rows_per_cta``), never below its grid floor.  Rows go on
threads when ``rows * T`` fits the budget, else serially on one thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..autotuner_heuristics.registry import AutotunerHeuristic
from ..cute.loop_nesting import sibling_row_loop_blocks
from ..inductor_lowering import ReductionLowering
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .backend import MetalBackend

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

_VECTOR_BYTES = 16
_SIMD_GROUPS_PER_ROW = (1, 4)
# Threadgroups of one SIMD group measured 6-10% slower than four rows of them
# on 12672-wide fp16 rows.
_SIMD_GROUPS_PER_THREADGROUP = 4


def _reduces_pass_after_loop(
    device_ir: DeviceIR, pass_block_ids: tuple[int, ...]
) -> bool:
    """Whether a reduction over a pass block sits outside that block's loop."""
    # ``device_ir`` imports ``helion.Config``, which is not yet defined when
    # ``runtime.kernel`` imports the heuristic registry and with it this module.
    from ..device_ir import ForLoopGraphInfo

    for info in device_ir.graphs:
        loop_blocks = info.block_ids if isinstance(info, ForLoopGraphInfo) else []
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and lowering.block_index in pass_block_ids
                and lowering.block_index not in loop_blocks
            ):
                return True
    return False


def _pass_load_itemsize(
    env: CompileEnvironment, pass_block_ids: tuple[int, ...]
) -> int:
    """The largest ``input_load_itemsize`` (each reduction's narrowest feeding
    load) over the pass blocks' reductions, 4 bytes if none is recorded."""
    fact = env.config_spec.reduction_kernel_fact
    sizes = [
        reduction.input_load_itemsize
        for reduction in (fact.reductions if fact is not None else ())
        if reduction.block_id in pass_block_ids and reduction.input_load_itemsize
    ]
    return max(sizes, default=4)


class MetalRowReductionHeuristic(AutotunerHeuristic):
    name = "metal_row_reduction"
    backend = "metal"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        spec = env.config_spec
        if spec.matmul_facts or spec.reduction_loops:
            return []
        plan = sibling_row_loop_blocks(env, device_ir, device_ir.graphs)
        if plan is None:
            return []
        row_block_id, pass_block_ids = plan
        blocks = {item.block_id: item for item in spec.block_sizes}
        if set(blocks) != {row_block_id, *pass_block_ids}:
            return []
        backend = env.backend
        assert isinstance(backend, MetalBackend)
        # The threadgroup is the product of its thread axes, so the reduction
        # axes' reservation divides what the row and pass blocks may use.
        reserved = backend._reserved_reduction_threads(spec.default_config())
        budget = MAX_THREADS_PER_THREADGROUP // reserved
        row_fragment = blocks[row_block_id]._fragment(spec)
        floor = row_fragment.low
        pass_fragments = [blocks[b]._fragment(spec) for b in pass_block_ids]
        pass_high = min(f.high for f in pass_fragments)
        if _reduces_pass_after_loop(device_ir, pass_block_ids):
            # Codegen combines the block after its loop only under a grid plan
            # that fixes its thread axis (``exited_block_thread_axis``), which
            # a reduction dimension or threaded passes of distinct block ids
            # deny.
            if reserved > 1 or len(pass_block_ids) > 1:
                return []
            shapes = [(floor, min(budget // floor, pass_high), 1)]
        else:
            lanes = _VECTOR_BYTES // _pass_load_itemsize(env, pass_block_ids)
            group = min(_SIMD_GROUPS_PER_THREADGROUP * SIMD_WIDTH, budget)
            shapes = [
                (min(max(floor, group // threads), row_fragment.high), threads, lanes)
                for threads in (g * SIMD_WIDTH for g in _SIMD_GROUPS_PER_ROW)
            ]
        seeds = []
        for rows, threads, lanes in shapes:
            width = min(threads * lanes, pass_high)
            # Drop a shape whose clamped width gives a thread fewer than
            # min(V, 2) lanes: it is a narrower layout with idle threads.
            if (
                threads > budget
                or width < threads * min(lanes, 2)
                or not all(f.low <= width <= f.high for f in pass_fragments)
            ):
                continue
            row_threads = 0 if rows * threads <= budget else 1
            block_sizes = {row_block_id: rows, **dict.fromkeys(pass_block_ids, width)}
            num_threads = {
                row_block_id: row_threads,
                **dict.fromkeys(pass_block_ids, threads),
            }
            seeds.append(
                Config(
                    block_sizes=[block_sizes[i.block_id] for i in spec.block_sizes],
                    num_threads=[num_threads[i.block_id] for i in spec.num_threads],
                )
            )
        return seeds
