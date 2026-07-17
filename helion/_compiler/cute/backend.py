"""CuteBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
from itertools import starmap
import operator
import os
import re
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

import sympy
import torch

from ... import exc
from ..backend import Backend
from ..backend import _active_loop_block_ids
from ..backend import _attention_flash_gate_enabled
from ..backend import _attention_flash_supported
from ..backend import _attention_softmax_pattern_head_dim
from ..backend import _kernel_specialized_mma_impl
from ..backend import _largest_divisor_at_most
from ..backend import _loop_contains_matmul
from ..backend import log
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_priors import ValuePrior
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ..device_function import Argument
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..tile_dispatch import TileStrategyDispatch
    from ..tile_strategy import TileStrategy
    from .attention_plan import AttentionScorePlan

    InductorOpOverrides = OpsHandler[Any]


def _detect_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
) -> bool:
    """Check if a device loop contains a matmul with MMA-compatible dtypes.

    Returns True only when the loop contains a compatible addmm/dot AND
    the grid has at least 2 block IDs (M and N), so the MMA pipeline
    can map them to tile offsets.  Three-level loops (grid[M] +
    device_loop[N] + device_loop[K]) are NOT supported yet.
    """
    from ...language._decorators import is_api_func
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction
    from .cute_mma import can_codegen_cute_mma_aten
    from .cute_mma import can_codegen_cute_mma_dot

    # MMA lowering currently relies on a single grid state that carries
    # both the M and N axes. Nested grid loops like grid[M] + grid[N] do
    # not satisfy that requirement because GenerateAST.current_grid_state
    # only tracks the innermost grid.
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    if len(device_ir.grid_block_ids[0]) != 2:
        return False
    root_grid_ids = set(device_ir.grid_block_ids[0])
    # CuTe MMA fragment partitioning is currently keyed to physical threads.
    # When an M/N tile is partially serialized into lane loops, the same
    # fragment would be reused for multiple logical lanes and produce
    # incorrect results. A pure K reduction loop is different: it does not
    # contribute MMA fragment coordinates, so we can still enable mma_mode
    # there to suppress synthetic lane loops around the K body.
    if any(
        block_id in root_grid_ids and threads > 0 and threads < block_size
        for block_id, block_size, threads in zip(
            block_ids,
            block_sizes,
            num_threads_config,
            strict=False,
        )
    ):
        return False
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        if graph_info.block_ids != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            # Only addmm/baddbmm trigger MMA mode — mm/bmm don't have
            # a built-in accumulator so their result is needed per iteration.
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                return True
    return False


def _detect_specialized_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    config: Config,
) -> bool:
    from ...language._decorators import is_api_func
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .cute_mma import _choose_mma_impl
    from .cute_mma import _mma_active_n_threads
    from .cute_mma import _tcgen05_root_m_threads
    from .cute_mma import can_codegen_cute_mma_aten
    from .cute_mma import can_codegen_cute_mma_dot

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 2:
        return False
    if len(block_ids) != 1 or any(block_id in root_grid_ids for block_id in block_ids):
        return False

    env = CompileEnvironment.current()
    root_block_sizes: list[int] = []
    root_thread_counts: list[int] = []
    root_thread_auto: list[bool] = []
    for block_id in root_grid_ids:
        block_size = env.block_sizes[block_id].from_config(config)
        if not isinstance(block_size, int):
            return False
        root_block_sizes.append(block_size)
        threads = env.config_spec.num_threads.config_get(
            config.num_threads, block_id, 0
        )
        resolved_threads = threads if threads > 0 else block_size
        root_thread_counts.append(resolved_threads)
        root_thread_auto.append(threads == 0)

    if functools.reduce(operator.mul, root_thread_counts, 1) > 1024:
        for idx in sorted(
            (i for i, is_auto in enumerate(root_thread_auto) if is_auto),
            reverse=True,
        ):
            other_threads = functools.reduce(
                operator.mul,
                (
                    root_thread_counts[j]
                    for j in range(len(root_thread_counts))
                    if j != idx
                ),
                1,
            )
            if other_threads <= 0:
                continue
            thread_budget = max(1024 // other_threads, 1)
            next_threads = _largest_divisor_at_most(
                root_block_sizes[idx], thread_budget
            )
            root_thread_counts[idx] = next_threads
            if functools.reduce(operator.mul, root_thread_counts, 1) <= 1024:
                break

    (bk,) = block_sizes
    if not isinstance(bk, int):
        return False
    bm, bn = root_block_sizes
    root_m_threads, root_n_threads = root_thread_counts

    def root_threads_support_impl(mma_impl: str) -> bool:
        if mma_impl == "tcgen05":
            mma_n_threads = _mma_active_n_threads("tcgen05")
            min_root_m_threads = _tcgen05_root_m_threads(bm, bn)
            if (
                root_m_threads < min_root_m_threads
                or root_m_threads > bm
                or bm % root_m_threads != 0
            ):
                return False
            if root_n_threads < mma_n_threads or root_n_threads > bn:
                return False
            if bn % root_n_threads != 0:
                return False
            return root_m_threads * root_n_threads <= 1024
        if mma_impl == "warp":
            return root_m_threads == bm and root_n_threads == bn
        return False

    for graph_info in fn.codegen.codegen_graphs:
        if getattr(graph_info, "block_ids", None) != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                lhs_node = node.args[1]
                if not isinstance(lhs_node, torch.fx.Node):
                    continue
                lhs_val = lhs_node.meta.get("val")
                if not isinstance(lhs_val, torch.Tensor):
                    continue
                mma_impl = _choose_mma_impl(
                    lhs_val.dtype, bm=bm, bn=bn, bk=bk, config=config
                )
                if mma_impl != "universal" and root_threads_support_impl(mma_impl):
                    return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                lhs_node = node.args[0]
                if not isinstance(lhs_node, torch.fx.Node):
                    continue
                lhs_val = lhs_node.meta.get("val")
                if not isinstance(lhs_val, torch.Tensor):
                    continue
                mma_impl = _choose_mma_impl(
                    lhs_val.dtype, bm=bm, bn=bn, bk=bk, config=config
                )
                if mma_impl != "universal" and root_threads_support_impl(mma_impl):
                    return True
    return False


def _attention_loop_shape(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> tuple[int, int, AttentionScorePlan] | None:
    """Recognize the flash-attention dataflow in a tile_n device loop.

    Matches BY SHAPE (not exclusivity), mirroring the spike sequence: a
    ``ForLoopGraphInfo`` over the key tile whose body holds a QK matmul
    (``aten.bmm.dtype``) feeding the online-softmax chain
    (``amax -> maximum -> sub -> exp2 -> sum``) and a ``baddbmm.default``
    (P@V) whose acc is the loop-carried output.

    Returns ``(bm, bn, score_plan)`` derived from the operand shapes -- bm =
    tile_m (query rows), bn = tile_n (key tile), score_plan = tile-local
    QK-score transforms plus head_dim -- or None when the loop is not the
    attention pattern or the validated config envelope is not met.

    This is deliberately strict so single-GEMM detection
    (``_detect_specialized_mma_loop``) is 100% untouched.
    """
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    # Attention binds to a 2-axis root grid (tile_b, tile_m) plus one inner
    # tile_n device loop.
    if len(device_ir.grid_block_ids) != 1:
        return None
    root_grid_ids = device_ir.grid_block_ids[0]
    if len(root_grid_ids) != 2:
        return None
    if len(block_ids) != 1 or any(bid in root_grid_ids for bid in block_ids):
        return None

    graph_info = None
    for candidate in fn.codegen.codegen_graphs:
        if isinstance(candidate, ForLoopGraphInfo) and candidate.block_ids == block_ids:
            graph_info = candidate
            break
    if graph_info is None:
        return None

    pattern = _attention_softmax_pattern_head_dim(
        graph_info.graph,
        kv_block_id=block_ids[0],
    )
    if pattern is None:
        return None

    env = CompileEnvironment.current()

    # The flash kernel walks full 128-row/128-col tiles, so BOTH the query
    # (tile_m) and key/value (tile_n) sequence lengths must be exact multiples of
    # 128. ``known_multiple`` is conservative under dynamic shapes (False unless
    # PROVABLY divisible), so a runtime-variable seq keeps the scalar path. This
    # mirrors the codegen's seq%128 gate at detection time, where the device
    # function arguments are not yet populated -- so the detector never sets
    # ``mma_mode`` / forces ``block=(128,1,1)`` on a config codegen would reject.
    if not env.block_sizes[root_grid_ids[1]].known_multiple(128):
        return None
    if not env.block_sizes[block_ids[0]].known_multiple(128):
        return None
    # The flash codegen derives num_kv from the QUERY length, so it only handles
    # SQUARE self-attention (query seq == key/value seq). Cross-attention
    # (q seq != kv seq, e.g. the 1024-query/512-kv test_attention_block_pointer
    # shape) would iterate the wrong number of KV tiles -> keep the scalar path.
    # Require statically-equal concrete sizes: a dynamic (SymInt) seq is rejected
    # conservatively (size hints can collide for distinct unbacked symbols).
    q_seq = env.block_sizes[root_grid_ids[1]].size
    kv_seq = env.block_sizes[block_ids[0]].size
    if not (isinstance(q_seq, int) and isinstance(kv_seq, int) and q_seq == kv_seq):
        return None
    # The flash kernel emits its OWN (batch*head, query-tile) program mapping; it
    # is only valid for the default flat pid with no L2 reordering. Persistent /
    # interleaved pids and l2_grouping remap program ids, so the emitted tile
    # indexing would address the wrong (batch, head, query-tile) -> reject.
    if config.pid_type != "flat":
        return None
    if any(grouping != 1 for grouping in config.l2_groupings):
        return None
    if any(order != [*range(len(order))] for order in config.loop_orders):
        return None
    if any(thread_count != 0 for thread_count in config.num_threads):
        return None
    cute_vector_widths = config.config.get("cute_vector_widths", [])
    if isinstance(cute_vector_widths, list) and any(
        width != 1 for width in cute_vector_widths
    ):
        return None

    # bm/bn from the config block sizes for tile_m (root grid axis 1) and tile_n
    # (the device loop block), NOT from the grid sizes.
    bm = env.block_sizes[root_grid_ids[1]].from_config(config)
    (bn,) = (env.block_sizes[bid].from_config(config) for bid in block_ids)
    if not isinstance(bm, int) or not isinstance(bn, int):
        return None
    # tile_b must be one (batch, head) per CTA -- the flash kernel maps the
    # b*h flatten one row per program. A tile_b block > 1 would pack multiple
    # (batch, head) rows per CTA and miscompute. Enforce what the docstrings
    # claim (previously unchecked).
    tile_b = env.block_sizes[root_grid_ids[0]].from_config(config)
    if not isinstance(tile_b, int) or tile_b != 1:
        return None
    return bm, bn, pattern.score_plan


def _detect_attention_mma_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    config: Config,
) -> bool:
    """True when the tile_n loop is a fused tcgen05 flash-attention body.

    Gated behind ``HELION_CUTE_FLASH`` and the validated config envelope
    (tile_b block = 1, mma_tiler 128x128, head_dim in {64, 128}, fp16/bf16) so the
    default scalar-fallback path is unchanged while the path is incomplete.
    """
    if not _attention_flash_gate_enabled() or not _attention_flash_supported():
        return False
    shape = _attention_loop_shape(fn, block_ids, config=config)
    if shape is None:
        return False
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    root_block_ids = device_ir.grid_block_ids[0]
    bm, bn, score_plan = shape
    from .cute_flash import flash_attention_graph_lse_plan_valid

    if not flash_attention_graph_lse_plan_valid(
        fn,
        root_block_ids=root_block_ids,
        kv_block_id=block_ids[0],
        score_plan=score_plan,
    ):
        return False
    bk = score_plan.head_dim
    # Validated envelope mirrors the spike: 128x128 mma_tiler, head_dim 64/128.
    if bm == 128 and bn == 128 and bk in (64, 128):
        fn.cute_state.attention_flash_score_plan = score_plan
        return True
    return False


def _is_mma_candidate_loop(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
    grid_ids: set[int],
) -> bool:
    if not any(bid not in grid_ids for bid in block_ids):
        return False
    resolved_threads: list[int] = [
        num_threads
        if num_threads > 0
        else int(block_size)
        if isinstance(block_size, int)
        else 0
        for block_size, num_threads in zip(block_sizes, num_threads_config, strict=True)
    ]
    return _detect_mma_loop(
        fn,
        block_ids,
        block_sizes=block_sizes,
        num_threads_config=resolved_threads,
    )


def _loop_may_use_mma(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    from ...language._decorators import is_api_func
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction
    from .cute_mma import can_codegen_cute_mma_aten
    from .cute_mma import can_codegen_cute_mma_dot

    device_ir = HostFunction.current().device_ir
    graph_by_id = {
        graph_info.graph_id: graph_info
        for graph_info in fn.codegen.codegen_graphs
        if hasattr(graph_info, "graph")
    }

    def graph_contains_mma(graph: object) -> bool:
        if not isinstance(graph, torch.fx.Graph):
            return False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ) and can_codegen_cute_mma_aten(node, with_acc=True):
                return True
            if (
                callable(node.target)
                and is_api_func(node.target)
                and getattr(node.target, "__name__", "") == "dot"
                and can_codegen_cute_mma_dot(node)
            ):
                return True
            if is_api_func(node.target) and getattr(node.target, "__name__", "") in {
                "_for_loop",
                "_for_loop_step",
            }:
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int):
                    nested = graph_by_id.get(graph_id)
                    if nested is not None and graph_contains_mma(nested.graph):
                        return True
        return False

    def graph_matches_loop(graph_info: object) -> bool:
        if getattr(graph_info, "block_ids", None) == block_ids:
            return True
        if not isinstance(graph_info, RootGraphInfo):
            return False
        phase_index = graph_info.phase_index
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    for graph_info in fn.codegen.codegen_graphs:
        if not graph_matches_loop(graph_info):
            continue
        if graph_contains_mma(getattr(graph_info, "graph", None)):
            return True
    return False


def _kernel_has_specialized_mma_path(
    fn: DeviceFunction,
    *,
    config: Config,
) -> bool:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    grid_ids = {
        bid for ids in HostFunction.current().device_ir.grid_block_ids for bid in ids
    }
    seen_block_ids: set[tuple[int, ...]] = set()
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = tuple(graph_info.block_ids)
        if block_ids in seen_block_ids or not any(
            bid not in grid_ids for bid in block_ids
        ):
            continue
        seen_block_ids.add(block_ids)
        block_sizes: list[int | torch.SymInt] = []
        for bid in block_ids:
            size = env.block_sizes[bid].from_config(config)
            if not isinstance(size, (int, torch.SymInt)):
                break
            block_sizes.append(size)
        else:
            if _detect_specialized_mma_loop(
                fn,
                list(block_ids),
                block_sizes=block_sizes,
                config=config,
            ):
                return True
    return False


def _loop_contains_atomic(
    fn: DeviceFunction,
    block_ids: list[int],
) -> bool:
    from ...language._decorators import is_api_func
    from ...language.atomic_ops import ATOMIC_OPS as atomic_targets
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    graph_by_id = {
        graph_info.graph_id: graph_info
        for graph_info in fn.codegen.codegen_graphs
        if hasattr(graph_info, "graph")
    }

    def graph_contains_atomic(graph: object) -> bool:
        if not isinstance(graph, torch.fx.Graph):
            return False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in atomic_targets:
                return True
            if is_api_func(node.target) and getattr(node.target, "__name__", "") in {
                "_for_loop",
                "_for_loop_step",
            }:
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int):
                    nested = graph_by_id.get(graph_id)
                    if nested is not None and graph_contains_atomic(nested.graph):
                        return True
        return False

    def graph_matches_loop(graph_info: object) -> bool:
        if getattr(graph_info, "block_ids", None) == block_ids:
            return True
        if not isinstance(graph_info, RootGraphInfo):
            return False
        phase_index = graph_info.phase_index
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    for graph_info in fn.codegen.codegen_graphs:
        if not graph_matches_loop(graph_info):
            continue
        if graph_contains_atomic(getattr(graph_info, "graph", None)):
            return True
    return False


def _graph_used_block_ids(
    fn: DeviceFunction,
    block_ids: list[int],
) -> set[int]:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    candidate_block_ids = set(block_ids)
    used: set[int] = set()

    def visit_value(value: object) -> None:
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                visit_value(dim)
            return
        if isinstance(value, torch.SymInt):
            block_id = env.get_block_id(value)
            if block_id is not None and block_id in candidate_block_ids:
                used.add(block_id)
            raw_expr = getattr(getattr(value, "node", None), "_expr", None)
            if isinstance(raw_expr, sympy.Expr):
                visit_value(raw_expr)
            return
        if isinstance(value, sympy.Expr):
            for symbol in value.free_symbols:
                block_id = env.get_block_id(symbol)
                if block_id is not None and block_id in candidate_block_ids:
                    used.add(block_id)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit_value(key)
                visit_value(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit_value(item)

    def is_tensor_like_value(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(
                is_tensor_like_value(key) or is_tensor_like_value(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(is_tensor_like_value(item) for item in value)
        return False

    for block_id in candidate_block_ids:
        block_info = env.block_sizes[block_id]
        if block_info.reduction:
            used.add(block_id)

    def graph_matches_loop(graph_info: object) -> bool:
        if getattr(graph_info, "block_ids", None) == block_ids:
            return True
        if not isinstance(graph_info, RootGraphInfo):
            return False
        try:
            phase_index = device_ir.root_ids.index(graph_info.graph_id)
        except ValueError:
            return False
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    for graph_info in fn.codegen.codegen_graphs:
        if not graph_matches_loop(graph_info):
            continue
        graph = getattr(graph_info, "graph", None)
        if graph is None:
            continue
        for node in graph.nodes:
            value = node.meta.get("val")
            if is_tensor_like_value(value):
                visit_value(value)
                for arg in node.args:
                    if is_tensor_like_value(arg):
                        visit_value(arg)
                for arg in node.kwargs.values():
                    if is_tensor_like_value(arg):
                        visit_value(arg)
    return used


# Leave broad headroom below the G1 sweep's 3600s subprocess timeout: budget
# checks happen between inline CuTe compile/benchmark units, then the selected
# config still has to compile, pass correctness, and run the final benchmark.
_CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS = 600


class CuteBackend(Backend):
    """CuTe DSL (CUTLASS Python DSL) code generation backend."""

    @property
    def name(self) -> str:
        return "cute"

    def validate_environment(self) -> None:
        from .cutedsl_compat import check_cute_backend_requirements

        check_cute_backend_requirements()

    def config_value_priors(self, config_spec: ConfigSpec) -> dict[str, ValuePrior]:
        """Bias the random half of the initial population toward the config
        family that performs well on Blackwell tcgen05 kernels.

        This encodes, as a distribution, what the backend's former hardcoded
        per-shape seed configs all converged on: a 2-CTA, TMA-fed,
        role-local-monolithic, static-persistent, tvm-ffi launch with deep AB
        staging and a 4-warp epilogue. Keys a given kernel does not expose
        (e.g. the ``tcgen05_*`` keys on a pointwise/reduction kernel) are
        ignored, and values a fragment cannot represent are dropped, so the
        priors are safe for any cute kernel -- a non-matmul kernel just picks up
        the generic biases (TMA indexing, 8 warps) on whichever keys it has.
        """
        from ...autotuner.config_priors import weighted_choice
        from .strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
        from .strategies import TCGEN05_STRATEGY_CONFIG_KEY
        from .strategies import Tcgen05PersistenceModel
        from .strategies import Tcgen05Strategy
        from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE

        priors: dict[str, ValuePrior] = {
            # Generic knobs shared by every cute kernel.
            "num_warps": weighted_choice({8: 4.0, 4: 2.0, 16: 1.0}),
            "num_stages": weighted_choice({4: 3.0, 3: 2.0, 2: 1.0}),
            "indexing": weighted_choice(
                {"tensor_descriptor": 4.0, "pointer": 1.0, "block_ptr": 1.0}
            ),
            "pid_type": weighted_choice(
                {
                    TCGEN05_TWO_CTA_SEED_PID_TYPE: 3.0,
                    "flat": 1.0,
                    "persistent_blocked": 1.0,
                }
            ),
            # tcgen05 / 2-CTA matmul knobs (absent on non-matmul kernels).
            "tcgen05_cluster_m": weighted_choice({2: 3.0, 1: 1.0}),
            "tcgen05_ab_stages": weighted_choice(
                {3: 3.0, 4: 2.0, 5: 1.0, 6: 1.0, 2: 1.0}
            ),
            "tcgen05_acc_stages": weighted_choice({2: 4.0, 1: 1.0}),
            "tcgen05_c_stages": weighted_choice({2: 3.0, 4: 2.0, 1: 1.0}),
            "tcgen05_num_epi_warps": weighted_choice({4: 3.0, 2: 1.0, 8: 1.0}),
            TCGEN05_STRATEGY_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value: 3.0,
                    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value: 1.0,
                }
            ),
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: weighted_choice(
                {
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value: 3.0,
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value: 1.0,
                }
            ),
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: weighted_choice({True: 3.0, False: 1.0}),
        }
        if config_spec is not None and config_spec.cute_flash_search_enabled:
            priors.update(self._cute_flash_config_value_priors(config_spec))
        return priors

    @staticmethod
    def _cute_flash_config_value_priors(
        config_spec: ConfigSpec,
    ) -> dict[str, ValuePrior]:
        from ...autotuner.config_priors import weighted_choice
        from .cute_flash import flash_attention_value_prior_weights

        return {
            key: weighted_choice(weights)
            for key, weights in flash_attention_value_prior_weights(
                config_spec._cute_flash_head_dim or 0,
                config_spec._cute_flash_num_kv,
                is_causal=config_spec._cute_flash_is_causal,
                has_kv_tile_pruning=config_spec._cute_flash_has_kv_tile_pruning,
                requires_ws_overlap=config_spec._cute_flash_requires_ws_overlap,
            ).items()
        }

    def customize_ast(self, hf: HostFunction) -> None:
        """CuTe-specific AST rewrites that rewrite high-level patterns into
        equivalent forms that compile to materially faster code.

        Currently:
          * ``rewrite_online_to_3pass`` rewrites the online two-pass
            softmax pattern into the 3-pass form (max-only, then
            sum-only, then consume).  The 3-pass form's two reductions
            are independent and compile to a more efficient layout on
            the CuTe backend.
        """
        from .online_to_3pass import rewrite_online_to_3pass

        rewrite_online_to_3pass(hf)

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from .layout_propagation import plan_layouts
        from .view_subtile import annotate_view_subtiles

        annotate_view_subtiles(graphs, config)
        plan_layouts(graphs, config, tile_strategy)

    def supports_config_key(self, key: str) -> bool:
        if (
            key == "num_threads"
            or key == "cute_vector_widths"
            or key.startswith(("tcgen05_", "cute_flash_"))
        ):
            return True
        return super().supports_config_key(key)

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        if (
            inductor_dtype := CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        ) is not None:
            return inductor_dtype
        if dtype is torch.float4_e2m1fn_x2:
            # PyTorch's shell dtype stores two E2M1 values in one byte.  CuTe
            # does not support scalar dereference for its 4-bit type yet, so
            # SIMT scalar loads treat the tensor as raw byte storage.
            return "cutlass.Uint8"
        if dtype is torch.uint64:
            return "cutlass.Int64"

        raise ValueError(f"Unsupported dtype for Cute backend: {dtype}")

    def acc_type(self, dtype: torch.dtype) -> str:
        if dtype in (torch.float16, torch.bfloat16):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    def supports_precompile(self) -> bool:
        # The CuTe DSL does not expose a Triton-style precompile entry point;
        # the autotuner has to compile + benchmark each config inline.
        return False

    def setup_compile_cache_dir(self, device_index: int) -> None:
        if "CUTE_DSL_CACHE_DIR" not in os.environ:
            from ...autotuner.local_cache import helion_cute_cache_dir

            cute_dir = helion_cute_cache_dir(device_index)
            os.environ["CUTE_DSL_CACHE_DIR"] = cute_dir
            log.debug("Set CUTE_DSL_CACHE_DIR=%s", cute_dir)

    def make_ephemeral_cache(
        self,
    ) -> contextlib.AbstractContextManager[None] | None:
        if self.keep_compile_cache_requested():
            return None
        return self._ephemeral_cute_cache()

    @contextlib.contextmanager
    def _ephemeral_cute_cache(self) -> Generator[None, None, None]:
        """Redirect the CuTe DSL on-disk cache to a temporary dir during
        autotuning so candidate compilations don't pollute the real cache.

        The winning config's artifact is re-persisted from memory into the
        real cache afterward (see :meth:`finalize_ephemeral_cache`).
        """
        saved = os.environ.get("CUTE_DSL_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="helion_cute_autotune_") as ephemeral:
            os.environ["CUTE_DSL_CACHE_DIR"] = ephemeral
            log.debug("Ephemeral CuTe cache: %s", ephemeral)
            try:
                yield
            finally:
                if saved is not None:
                    os.environ["CUTE_DSL_CACHE_DIR"] = saved
                else:
                    os.environ.pop("CUTE_DSL_CACHE_DIR", None)

    def finalize_ephemeral_cache(
        self, bound_kernel: BoundKernel[Any], config: Config
    ) -> None:
        """Persist the winning config's compiled artifact into the real cache.

        Candidate artifacts died with the ephemeral dir, but the winner's
        launcher still holds the compiled module in memory and the disk-cache
        key excludes ``CUTE_DSL_CACHE_DIR``, so re-persisting from memory
        writes the exact artifact a later process will look up.  Launchers and
        compile-cache entries are kept so the winner launches without
        recompiling.
        """
        from ...runtime.config import Config

        compiled_fn = bound_kernel._compile_cache.get(config)
        if compiled_fn is None:
            # The autotuner may return a minimized config (default values
            # stripped); the compiled entry is keyed by the full config.
            default = bound_kernel.config_spec.default_config()
            # pyrefly: ignore [bad-argument-type]
            full_config = Config(**(default.config | config.config))
            compiled_fn = bound_kernel._compile_cache.get(full_config)
        if compiled_fn is None:
            return
        cute_kernel = compiled_fn.__globals__.get(  # type: ignore[attr-defined]
            f"_helion_{bound_kernel.kernel.name}"
        )
        launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", None)
        if not launchers:
            return
        device_index = (
            bound_kernel.env.device.index
            if bound_kernel.env.device.index is not None
            else 0
        )
        # The ephemeral context restored CUTE_DSL_CACHE_DIR on exit; this sets
        # the real per-device dir when the user did not provide one.
        self.setup_compile_cache_dir(device_index)
        for launcher in launchers.values():
            launcher.persist_compiled()

    def compiled_cache_key(
        self, bound_kernel: BoundKernel[Any], compiled_fn: object
    ) -> str | None:
        cute_kernel = compiled_fn.__globals__.get(  # type: ignore[attr-defined]
            f"_helion_{bound_kernel.kernel.name}"
        )
        if cute_kernel is None:
            return None
        launchers = getattr(cute_kernel, "_helion_cute_compiled_launchers", None)
        if not launchers:
            return None
        for launcher in launchers.values():
            key = getattr(launcher, "_cache_key", None)
            if key is not None:
                return key
        return None

    def annotate_compiled_module(
        self, module: object, source: str, kernel_name: str
    ) -> None:
        cute_kernel = getattr(module, f"_helion_{kernel_name}", None)
        if cute_kernel is None:
            return
        with contextlib.suppress(AttributeError, TypeError):
            cute_kernel._helion_cute_source_hash = hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest()

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # Exceptions raised from inside the cute/cutlass DSL during compile or
        # launch are expected when an invalid config is tried; treat them as
        # benign so the autotuner moves on. Anything else (AssertionError,
        # NameError/TypeError/AttributeError from generated code, ...) is
        # almost certainly a real Helion bug and is surfaced at warn level so
        # it gets noticed without terminating tuning.
        cls = type(err)
        module = getattr(cls, "__module__", "") or ""
        if module.startswith(("cutlass", "cute")):
            return "debug"
        if isinstance(err, Exception):
            return "warn"
        return None

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        # The default Triton do_bench uses CUDA events that mis-time the CuTe
        # path on Blackwell - launches show up as ~5ms when the kernel is
        # actually 250ms+. Use synchronized wall-clock timing instead so
        # autotune scores reflect real performance.
        from ...autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        # Same rationale as get_do_bench: the default interleaved bench uses
        # CUDA events that mis-time the CuTe path. Use the synchronized
        # wall-clock fallback so the autotuner's interleaved compare path
        # produces real timings.
        from ...autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def autotune(
        self,
        bound_kernel: BoundKernel[Any],
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        original_budget = bound_kernel.settings.autotune_budget_seconds
        # "full" should mean the search algorithms run to completion unless the
        # caller explicitly provides a budget. Keep the defensive default budget
        # only for cheaper efforts where bounded local iteration is expected.
        if (
            bound_kernel.settings.autotune_budget_seconds is None
            and bound_kernel.settings.autotune_effort != "full"
        ):
            bound_kernel.settings.autotune_budget_seconds = (
                _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS
            )
        try:
            return super().autotune(bound_kernel, args, force=force, **kwargs)
        finally:
            bound_kernel.settings.autotune_budget_seconds = original_budget

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        return "cutlass.Constexpr"

    def inline_constexpr(self, name: str, value: str) -> str:
        return f"{name} = {value}"

    @property
    def default_launcher_name(self) -> str:
        return "_default_cute_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "operator": "import operator",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "cutlass": "import cutlass",
            "cute": "import cutlass.cute as cute",
            "ir": "from cutlass._mlir import ir",
            "mlir_math": "from cutlass._mlir.dialects import math as mlir_math",
            "_default_cute_launcher": "from helion.runtime import default_cute_launcher as _default_cute_launcher",
            "_next_power_of_2": "from helion._utils import next_power_of_2 as _next_power_of_2",
            "_cute_argreduce_index": "from helion._compiler.cute.reduce_helpers import _cute_argreduce_index",
            "_helion_tcgen05_pipeline": (
                "from helion._compiler.cute import tcgen05_pipeline "
                "as _helion_tcgen05_pipeline"
            ),
            "_cute_gelu_erf_exact_f32x2": (
                "from helion._compiler.cute.epilogue_helpers import "
                "gelu_erf_exact_f32x2 as _cute_gelu_erf_exact_f32x2"
            ),
            "_cute_grouped_reduce_shared_tree": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_tree",
            "_cute_grouped_reduce_shared_two_stage": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_shared_two_stage",
            "_cute_grouped_reduce_warp": "from helion._compiler.cute.reduce_helpers import _cute_grouped_reduce_warp",
            "_cute_pre_vec_fold": "from helion._compiler.cute.reduce_helpers import _cute_pre_vec_fold",
            "_cute_store_shared_remote_x4": "from helion._compiler.cute.cluster_helpers import store_shared_remote_x4 as _cute_store_shared_remote_x4",
            "_cute_issue_clc_query_nomulticast": "from helion._compiler.cute.clc_helpers import issue_clc_query_nomulticast as _cute_issue_clc_query_nomulticast",
            "_cute_inline_asm_elementwise": "from helion._compiler.cute.inline_asm_helpers import inline_asm_elementwise as _cute_inline_asm_elementwise",
            "_cute_fp8e4m3fn_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_to_float32 as _cute_fp8e4m3fn_to_float32",
            "_cute_fp8e4m3fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import fp8e4m3fn_x2_to_float32 as _cute_fp8e4m3fn_x2_to_float32",
            "_cute_float4_e2m1fn_x2_to_float32": "from helion._compiler.cute.quantized_helpers import float4_e2m1fn_x2_to_float32 as _cute_float4_e2m1fn_x2_to_float32",
            "_cute_grid_barrier": "from helion._compiler.cute.grid_barrier import grid_barrier as _cute_grid_barrier",
            "_cute_atomic_max_float32": "from helion._compiler.cute.atomic_helpers import atomic_max_float32 as _cute_atomic_max_float32",
            "_cute_atomic_min_float32": "from helion._compiler.cute.atomic_helpers import atomic_min_float32 as _cute_atomic_min_float32",
        }

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"{index_dtype}(cute.arch.block_idx()[{dim}])"

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLArg
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        class HelionCuteDSLOpOverrides(CuteDSLOpOverrides):
            @staticmethod
            def where(
                condition: CuteDSLArg,
                a: CuteDSLArg,
                b: CuteDSLArg,
            ) -> CuteDSLArg:
                tensor_arg = (
                    HelionCuteDSLOpOverrides._get_cse_var(a)
                    or HelionCuteDSLOpOverrides._get_cse_var(b)
                    or HelionCuteDSLOpOverrides._get_cse_var(condition)
                )
                if tensor_arg is not None:
                    return CuteDSLOpOverrides.where(condition, a, b)
                return f"(({a}) if ({condition}) else ({b}))"

        return HelionCuteDSLOpOverrides()

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"{dtype_str}({expr_str})"

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        from ..device_function import DeviceFunction
        from ..device_function import NoCurrentFunction

        if isinstance(x, ast.Name):
            try:
                df = DeviceFunction.current()
            except NoCurrentFunction:
                pass
            else:
                if (
                    df.cute_state.get_tcgen05_store_value(df.variable_aliases(x.id))
                    is not None
                ):
                    return x
        return super().cast_ast(x, target_dtype)

    def grid_barrier_stmt(self, sem_arg: str) -> str | None:
        # ``sem_arg`` is a TensorArg that arrives as a ``cute.Tensor``; its
        # ``.iterator`` is the underlying ``cute.Pointer`` to the semaphore.
        return f"_cute_grid_barrier({sem_arg}.iterator)"

    def lane_index_expr(
        self, offset_var: str, elements_per_thread: int, *, axis: int
    ) -> str:
        from ..compile_environment import CompileEnvironment

        index_dtype = CompileEnvironment.current().index_type()
        return (
            f"{offset_var} + {index_dtype}(cute.arch.thread_idx()[{axis}])"
            f" * {elements_per_thread}"
        )

    def lane_offset_expr(self, lane_var: str) -> str:
        return f"cutlass.Int32({lane_var})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from ..device_function import cute_texpr

        return cute_texpr(expr)

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(f"cutlass.Int32({begin})")
        range_args.append(f"cutlass.Int32({end})")
        if step is not None and step != "1":
            range_args.append(f"cutlass.Int32({step})")
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return (
            f"{offsets_var} = ({lid}) * ({block_size_var})"
            f" + {dtype}(cute.arch.thread_idx()[{axis}])"
        )

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        if axis >= 3 and block_size_var != "1":
            raise exc.BackendUnsupported(self.name, f"thread axis {axis}")
        if block_size_var == "1":
            return offset_var
        return f"{offset_var} + {dtype}(cute.arch.thread_idx()[{axis}])"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return self.grid_index_expr(offset_var, block_size_var, dtype, axis=axis)

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def max_reduction_threads(self) -> int | None:
        return 1024

    def max_reduction_loop(self) -> int | None:
        from ..reduction_strategy import cute_looped_reduction_block_size

        max_threads = self.max_reduction_threads()
        if max_threads is None:
            return None
        return cute_looped_reduction_block_size(2**31 - 1, max_threads)

    def adjust_reduction_thread_count(
        self, requested: int, existing_strategies: list[TileStrategy]
    ) -> int:
        from ..reduction_strategy import ReductionStrategy
        from .thread_budget import MAX_THREADS_PER_BLOCK

        if requested <= 1:
            return requested
        other_threads = 1
        for strategy in existing_strategies:
            if isinstance(strategy, ReductionStrategy):
                count = strategy._reduction_thread_count()
                if count > 0:
                    other_threads *= count
            else:
                for size in strategy.thread_block_sizes():
                    if size > 1:
                        other_threads *= size
        while other_threads * requested > MAX_THREADS_PER_BLOCK and requested > 1:
            requested //= 2
        return requested

    def create_synthetic_reduction_lanes(
        self,
        thread_count: int,
        size_hint: int,
    ) -> int | None:
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        if thread_count <= 0:
            return None
        padded_size = next_power_of_2(max(1, size_hint))
        if padded_size > thread_count:
            return padded_size // thread_count
        return None

    def reduction_axis_first(self) -> bool:
        return True

    def thread_in_tile_mask_expr(
        self, block_size_var: str, *, axis: int = 0
    ) -> str | None:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) < ({block_size_var})"

    def force_tile_mask(self) -> bool:
        return True

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        # One element per thread: tile-shaped temporaries are scalars.
        return f"{self.dtype_str(dtype)}({value_expr})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return expr

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return expr

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"({true_val}) if ({mask}) else ({false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        from ..compile_environment import CompileEnvironment

        index_type = CompileEnvironment.current().index_type()
        lhs = self.cast_expr(a, index_type)
        rhs = self.cast_expr(b, index_type)
        return f"({lhs}) if ({lhs}) < ({rhs}) else ({rhs})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return "cutlass.Int32(0)"

    def next_power_of_2_host_expr(self, expr: str) -> str:
        return f"_next_power_of_2({expr})"

    def reduction_combine_expr(
        self,
        reduction_type: str,
        acc: str,
        val: str,
        dtype: torch.dtype,
    ) -> str:
        # Use Python ternary instead of cute.where for max/min because
        # these operate on scalar registers, not tensors.
        #
        # Cast the incoming value to the accumulator dtype first.  The
        # accumulator is promoted to the computation dtype (fp32 for
        # half-precision inputs), but the per-iteration reduction input keeps
        # the tensor's storage dtype (e.g. bf16 for a masked half load).  The
        # CUTLASS DSL strictly type-checks the two branches of a Python ternary
        # ("Then and else blocks of ifexp return different types"), so a bare
        # ``acc if acc > val else val`` with mixed fp32/bf16 operands fails to
        # compile.  The cast is a no-op when ``val`` already matches.
        val = self.cast_expr(val, self.dtype_str(dtype))
        if reduction_type == "sum":
            return f"({acc} + {val})"
        if reduction_type == "max":
            return f"({acc}) if ({acc}) > ({val}) else ({val})"
        if reduction_type == "min":
            return f"({acc}) if ({acc}) < ({val}) else ({val})"
        if reduction_type == "prod":
            return f"({acc} * {val})"
        raise exc.BackendUnsupported(self.name, f"reduction combine {reduction_type!r}")

    def _threads_for_block_size_var(self, block_size_var: str | None) -> int:
        # threads_in_group must be a Python int literal for CuTe DSL.
        from ..reduction_strategy import ReductionStrategy
        from ..tile_strategy import BlockSizeTileStrategy

        threads = 32
        strategies = self._get_strategies()
        if block_size_var is not None:
            for strategy in strategies:
                if not isinstance(strategy, ReductionStrategy):
                    continue
                strategy_bs_var = strategy.block_size_var(strategy.block_index)
                if strategy_bs_var != block_size_var:
                    continue
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc

            # Block reductions are keyed by a tile block-size var rather than a
            # ReductionStrategy var. Recover the tile width from the owning strategy.
            for strategy in strategies:
                if not isinstance(strategy, BlockSizeTileStrategy):
                    continue
                for idx, block_id in enumerate(strategy.block_ids):
                    strategy_bs_var = strategy.block_size_var(block_id)
                    if strategy_bs_var != block_size_var:
                        continue
                    block_size = strategy.block_size
                    if isinstance(block_size, list) and idx < len(block_size):
                        block_size = block_size[idx]
                    if isinstance(block_size, int) and block_size > 0:
                        return min(block_size, 32)
            return threads

        for strategy in strategies:
            if isinstance(strategy, ReductionStrategy):
                tc = strategy._reduction_thread_count()
                if tc > 0:
                    return tc
        return threads

    def reduction_threads_hint(self, block_size_var: str | None = None) -> int | None:
        return self._threads_for_block_size_var(block_size_var)

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        threads = (
            threads_in_group
            if threads_in_group is not None
            else self._threads_for_block_size_var(block_size_var)
        )
        tg = f", threads_in_group={threads}"
        if reduction_type == "sum":
            return f"cute.arch.warp_reduction_sum({input_name}{tg})"
        if reduction_type == "max":
            return f"cute.arch.warp_reduction_max({input_name}{tg})"
        if reduction_type == "min":
            return (
                f"cute.arch.warp_reduction("
                f"{input_name}, lambda a, b: (a if a < b else b){tg})"
            )
        if reduction_type == "prod":
            return f"cute.arch.warp_reduction({input_name}, lambda a, b: (a * b){tg})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def thread_linear_index_expr(self, axis_sizes: dict[int, int]) -> str | None:
        from ..compile_environment import CompileEnvironment

        index_dtype = CompileEnvironment.current().index_dtype
        index_type = self.index_type_str(index_dtype)
        if not axis_sizes:
            return self.cast_expr("0", index_type)
        stride = 1
        terms: list[str] = []
        for axis, size in sorted(axis_sizes.items()):
            term = self.cast_expr(f"cute.arch.thread_idx()[{axis}]", index_type)
            if stride != 1:
                term = f"({term}) * {self.cast_expr(repr(stride), index_type)}"
            terms.append(term)
            stride *= size
        return " + ".join(terms)

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        if index_dtype is None:
            raise exc.BackendUnsupported(self.name, "missing index_dtype for argreduce")
        value_reduction = "min" if reduction_type == "argmin" else "max"
        reduced_value = self.reduction_expr(
            input_name,
            value_reduction,
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        index_dtype_str = self.index_type_str(index_dtype)
        max_index = self.cast_expr(repr(torch.iinfo(index_dtype).max), index_dtype_str)
        candidate_index = f"({index_value}) if (({input_name}) == ({reduced_value})) else ({max_index})"
        reduced_index = self.reduction_expr(
            candidate_index,
            "min",
            dim,
            block_size_var=block_size_var,
            threads_in_group=threads_in_group,
        )
        return self.cast_expr(reduced_index, self.dtype_str(output_dtype))

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        if reduction_type == "argmin":
            better = (
                f"(({value}) < ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        else:
            better = (
                f"(({value}) > ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        return [
            (
                f"{acc}, {acc_index} = "
                f"(({value}), ({index})) if ({better}) else (({acc}), ({acc_index}))"
            )
        ]

    def _get_strategies(self) -> list[TileStrategy]:
        """Get the current device function's strategies."""
        from ..device_function import DeviceFunction

        try:
            return DeviceFunction.current().tile_strategy.strategies
        except Exception:
            return []

    def launcher_keyword_args(self, config: Config, *, has_barrier: bool) -> list[str]:
        from ..device_function import DeviceFunction
        from ..host_function import HostFunction
        from .thread_budget import MAX_THREADS_PER_BLOCK

        device_function = DeviceFunction.current()
        codegen = device_function.codegen
        tile_strategy = device_function.tile_strategy
        final_kernel_text = "\n".join(
            ast.unparse(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )
        final_thread_axes = {
            int(axis_text)
            for axis_text in re.findall(
                r"cute\.arch\.thread_idx\(\)\[(\d+)\]",
                final_kernel_text,
            )
        }

        def launcher_args_with_compile_options(block_arg: str) -> list[str]:
            launcher_args = [block_arg]
            compile_options: list[str] = []
            if config.get(TCGEN05_CUBIN_LINEINFO_CONFIG_KEY) is True:
                compile_options.append("--generate-line-info")
            # ``--enable-tvm-ffi`` is emitted in codegen only when the
            # autotune flag is True so the generated code reflects which
            # configs deliberately requested FFI. The runtime
            # (``_get_compiled_cute_launcher``) unconditionally merges
            # the flag in for the generic launcher, so configs with this
            # flag False still execute with FFI enabled — that drift is
            # intentional for now and noted here for future cleanups.
            if config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True:
                if (
                    _kernel_specialized_mma_impl(
                        device_function,
                        config=device_function.config,
                    )
                    != "tcgen05"
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY}=True requires "
                        "tcgen05 CuTe lowering",
                    )
                compile_options.append("--enable-tvm-ffi")
            if compile_options:
                launcher_args.append(
                    f"cute_compile_options={' '.join(compile_options)!r}"
                )
            return launcher_args

        # Fused tcgen05 flash-attention: 128 threads (single-warpgroup Stage-3)
        # or 256 threads (Stage-4 warp-spec producer/consumer split). The custom
        # flash codegen owns the whole device body, so the SIMT thread-axis
        # heuristics below do not apply.
        if device_function.cute_state.attention_flash_block_ids is not None:
            flash_threads = device_function.cute_state.attention_flash_threads
            return launcher_args_with_compile_options(f"block=({flash_threads}, 1, 1)")

        block_size_values = {
            name: int(value)
            for name, value in re.findall(
                r"^(_BLOCK_SIZE_\d+) = (\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        }
        # Accept both the historical ``offset_<n>`` prefix (non-CuTe backends)
        # and the post-rename ``tile_offset_<n>`` prefix (CuTe backend, see the
        # CuTe DSL preprocessor counter-collision note in
        # ``tile_strategy.py``). The launch-dim recovery walks the generated
        # source to pair Helion's per-axis offsets with their thread axes; if
        # the regex misses, we fall back to ``[1, 1, 1]`` and any kernel that
        # depends on the recovery launches with too-small dims.
        offset_block_sizes = dict(
            re.findall(
                r"^\s*(?:tile_)?offset_(\d+) = .* \* (_BLOCK_SIZE_\d+)$",
                final_kernel_text,
                flags=re.MULTILINE,
            )
        )
        offset_thread_dims = [1, 1, 1]
        # When a lane loop is active for an axis the generated index expression
        # has the form
        #   ``indices_<n> = tile_offset_<n> + Int32(thread_idx()[<axis>]) * <epT> + Int32(lane_<n>)``
        # where ``<epT>`` is ``elements_per_thread`` for that axis and the
        # outer ``for lane_<n> in range(<epT>):`` covers the residual. In
        # that case the launch-time thread extent for the axis is
        # ``block_size / <epT>``, not ``block_size``. Without dividing by
        # ``<epT>`` here the launch dim ends up at ``block_size`` while the
        # generated tile arithmetic only spans ``block_size / <epT>`` threads,
        # which means ``thread_idx[axis] >= block_size / <epT>`` writes past
        # the tile and triggers ``cudaErrorIllegalAddress`` mid-search. The
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx`` is closed
        # before the ``* <epT>`` multiplier, so the closing ``)`` between
        # ``[axis]`` and ``*`` is part of the line we have to skip over.
        #
        # The launch dim along ``axis`` has to serve **every** indices line
        # that uses that axis. If two lines on the same axis emit different
        # multipliers (e.g. one block_id with ``epT=1`` and another with
        # ``epT=2``), the line with the larger ``thread_idx[axis]`` range
        # is the binding one — so we compute ``block_size // epT`` *per
        # line* and take the ``max`` across lines, rather than combining
        # multipliers across lines and dividing once. ``re.findall`` over
        # the optional-multiplier alternation cannot be made to populate
        # the multiplier group reliably (the optional ``(?:...)?`` form
        # prefers the empty match), so we scan the lines once in Python
        # to keep the per-line ``(block_size, epT)`` pair intact.
        # ``indices_line_re`` anchors on the
        # ``cutlass.Int32(cute.arch.thread_idx()[<axis>])`` form the CuTe
        # backend emits via ``lane_index_expr`` (see backend.py:2666).
        # The trailing ``\)`` after ``\]`` is the close of the
        # ``cutlass.Int32(...)`` wrapper around ``thread_idx``; if a
        # future codegen path drops the wrapper, the regex will
        # silently fail to match and the launch dim under-dimensions
        # without a signal. ``indices_line_assert_re`` below detects
        # the "wrapper-dropped" form so we can fail loudly instead.
        indices_line_re = re.compile(
            r"^\s*indices_\d+ = (?:tile_)?offset_(\d+) \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)\[(\d+)\]\)"
            r"(?:\s*\*\s*(?:cutlass\.Int32\()?(\d+))?",
            flags=re.MULTILINE,
        )
        # Loose form: any ``indices_<n>`` line containing ``thread_idx``
        # under any wrapping. Used only for the wrapper-invariant
        # assertion below — never consulted for launch-dim values.
        indices_line_loose_re = re.compile(
            r"^\s*indices_\d+ = (?:tile_)?offset_\d+ \+ "
            r"[^\n]*?cute\.arch\.thread_idx\(\)",
            flags=re.MULTILINE,
        )
        matched_lines = 0
        for line_match in indices_line_re.finditer(final_kernel_text):
            matched_lines += 1
            offset_id = line_match.group(1)
            axis_text = line_match.group(2)
            multiplier_text = line_match.group(3)
            axis = int(axis_text)
            if not (0 <= axis < len(offset_thread_dims)):
                continue
            block_name = offset_block_sizes.get(offset_id)
            block_size = block_size_values.get(block_name or "")
            if block_size is None and block_name is not None:
                try:
                    config_index = int(block_name.removeprefix("_BLOCK_SIZE_"))
                except ValueError:
                    config_index = -1
                if 0 <= config_index < len(config.block_sizes):
                    config_block_size = config.block_sizes[config_index]
                    if isinstance(config_block_size, int):
                        block_size = config_block_size
            if block_size is None:
                continue
            elements_per_thread = int(multiplier_text) if multiplier_text else 1
            if elements_per_thread <= 0:
                # ``lane_index_expr`` only emits ``* <n>`` for ``n >= 1``;
                # a ``* 0`` multiplier would mean the index expression is
                # already invalid (every thread maps to offset 0), so
                # silently falling back to ``block_size`` here would
                # mask the very class of bug this recovery exists to
                # catch. Surface it loudly.
                raise AssertionError(
                    f"launch-dim recovery: non-positive "
                    f"elements_per_thread={elements_per_thread} for axis="
                    f"{axis}"
                )
            if block_size % elements_per_thread != 0:
                # The strategy invariant ``_thread_extent_for_axis``
                # rejects non-divisible ``block_size / nt`` at
                # construction, so the regex should never see a line
                # whose ``epT`` does not divide ``block_size`` evenly.
                # If a future codegen regression breaks that invariant,
                # surface it loudly here rather than silently
                # under-dimensioning the launch.
                raise AssertionError(
                    f"launch-dim recovery: block_size={block_size} not "
                    f"divisible by elements_per_thread={elements_per_thread} "
                    f"for axis={axis}"
                )
            line_extent = block_size // elements_per_thread
            offset_thread_dims[axis] = max(offset_thread_dims[axis], line_extent)
        # Wrapper-invariant assertion: when there is at least one
        # ``indices_<n> = ... thread_idx() ...`` line and we have
        # ``_BLOCK_SIZE_<n>`` constants to consume, the strict regex
        # above must have matched at least one of them. A loose match
        # without a strict match means the codegen emitted
        # ``thread_idx[axis]`` outside the ``cutlass.Int32(...)``
        # wrapper the strict regex anchors on; in that case the
        # launch dim would silently fall back to ``[1, 1, 1]`` and
        # under-dimension every kernel that uses a thread axis.
        if (
            offset_block_sizes
            and indices_line_loose_re.search(final_kernel_text)
            and matched_lines == 0
        ):
            raise AssertionError(
                "launch-dim recovery: indices_<n> lines reference "
                "cute.arch.thread_idx() but none matched the "
                "cutlass.Int32(...)-wrapped form. The strict regex in "
                "_launcher_block_arg is anchored on the wrapper emitted "
                "by lane_index_expr; if the codegen producer was changed "
                "to drop the wrapper, update the regex to match the new "
                "form."
            )
        dims = tuple(codegen.max_thread_block_dims)
        root_live_dims = tuple(codegen.root_thread_block_dims)
        referenced_dims = tuple(codegen.referenced_thread_block_dims)
        static_dims = tile_strategy.thread_block_dims()
        dim_exprs = tile_strategy.thread_block_dim_exprs()
        static_threads = functools.reduce(operator.mul, static_dims, 1)
        dynamic_threads = functools.reduce(operator.mul, dims, 1)
        has_nested_device_loops = any(
            getattr(graph_info, "block_ids", None) is not None
            for graph_info in codegen.codegen_graphs
        )
        root_grid_dims = [1, 1, 1]
        device_ir = HostFunction.current().device_ir
        for block_ids in device_ir.grid_block_ids:
            strategy = tile_strategy.block_id_to_strategy.get(tuple(block_ids))
            if strategy is None:
                continue
            for axis, size in enumerate(strategy.thread_block_sizes()):
                if axis < len(root_grid_dims):
                    root_grid_dims[axis] = max(root_grid_dims[axis], size)
        root_static_dims = tuple(root_grid_dims)
        root_static_threads = functools.reduce(operator.mul, root_static_dims, 1)
        specialized_root_tcgen05 = (
            _kernel_specialized_mma_impl(device_function, config=device_function.config)
            == "tcgen05"
            and root_static_dims != (1, 1, 1)
            and root_static_threads <= MAX_THREADS_PER_BLOCK
        )
        tcgen05_compact_dims = (
            device_function.cute_state.block_shape if specialized_root_tcgen05 else None
        )
        if referenced_dims != (1, 1, 1):
            dims = referenced_dims
        elif has_nested_device_loops:
            dims = tuple(codegen.max_thread_block_dims)
        if functools.reduce(operator.mul, dims, 1) > MAX_THREADS_PER_BLOCK:
            if (
                root_static_dims != (1, 1, 1)
                and root_static_threads <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_static_dims
            elif static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
        recorded_dims = tuple(
            max(
                codegen.max_thread_block_dims[axis],
                root_live_dims[axis],
                referenced_dims[axis],
            )
            for axis in range(3)
        )
        if tcgen05_compact_dims is not None:
            dims = tcgen05_compact_dims
        elif specialized_root_tcgen05:
            dims = root_static_dims
        elif final_thread_axes and (
            referenced_dims != (1, 1, 1) or has_nested_device_loops
        ):
            dims = tuple(
                max(size, root_static_dims[axis], recorded_dims[axis])
                if axis in final_thread_axes
                else size
                for axis, size in enumerate(dims)
            )
            dims = tuple(
                size if axis in final_thread_axes else 1
                for axis, size in enumerate(dims)
            )
        else:
            dims = tuple(
                min(size, recorded_dims[axis]) for axis, size in enumerate(dims)
            )
        current_threads = functools.reduce(operator.mul, dims, 1)
        if current_threads > MAX_THREADS_PER_BLOCK:
            if static_dims != (1, 1, 1) and static_threads <= MAX_THREADS_PER_BLOCK:
                dims = static_dims
            elif (
                root_live_dims != (1, 1, 1)
                and functools.reduce(operator.mul, root_live_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = root_live_dims
            elif (
                referenced_dims != (1, 1, 1)
                and functools.reduce(operator.mul, referenced_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = referenced_dims
        if (
            dims != (1, 1, 1)
            and static_dims != (1, 1, 1)
            and not has_nested_device_loops
            and static_threads < dynamic_threads
            # Synthetic free-``hl.arange`` thread axes are real launch lanes that
            # the strategy's ``static_dims`` does not know about, so do not fall
            # back to ``static_dims`` (which would drop them) when they are live.
            and not codegen.cute_synthetic_arange_axis_sizes
        ):
            dims = static_dims
        if (
            dim_exprs is not None
            and dim_exprs != ("1", "1", "1")
            # ``dim_exprs`` is the strategy's static per-axis launch shape; it
            # has no entry for synthetic free-``hl.arange`` axes, so adopting it
            # wholesale would shrink those axes back to 1. Skip it when they are
            # live (the synthetic extents are already folded into ``dims``).
            and not codegen.cute_synthetic_arange_axis_sizes
        ):
            if all(expr.isdigit() for expr in dim_exprs):
                expr_dims = tuple(int(expr) for expr in dim_exprs)
                if functools.reduce(
                    operator.mul, expr_dims, 1
                ) <= MAX_THREADS_PER_BLOCK and all(
                    expr_dim <= current_dim
                    for expr_dim, current_dim in zip(expr_dims, dims, strict=True)
                ):
                    dims = expr_dims
            elif dims == (1, 1, 1):
                return launcher_args_with_compile_options(
                    f"block=({dim_exprs[0]}, {dim_exprs[1]}, {dim_exprs[2]})"
                )
        if offset_thread_dims != [1, 1, 1]:
            candidate_dims = tuple(
                starmap(max, zip(dims, offset_thread_dims, strict=True))
            )
            if (
                functools.reduce(operator.mul, candidate_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = candidate_dims
        if dims == (1, 1, 1):
            dynamic_dims = tuple(codegen.max_thread_block_dims)
            if (
                dynamic_dims != (1, 1, 1)
                and functools.reduce(operator.mul, dynamic_dims, 1)
                <= MAX_THREADS_PER_BLOCK
            ):
                dims = dynamic_dims
            else:
                dims = DeviceFunction.current().tile_strategy.thread_block_dims()
        # Detect the silent-truncation case: codegen has already emitted
        # thread_idx[axis] references that assume a certain per-axis
        # extent (recorded in ``referenced_thread_block_dims``), but the
        # chosen launch ``dims`` for that axis is smaller. This happens
        # when the joint requested thread count exceeds 1024 and the
        # earlier fallback paths in this function dropped axes to (1, 1, 1)
        # to fit the budget. Under-dimensioning here would leave
        # cross-thread reductions (group_span, warp_reduce) operating
        # against nonexistent lanes, silently producing wrong results
        # (e.g. softmax with M*N>1024 returning max_err on the order of
        # tens). Surface this as ``BackendUnsupported`` so the autotuner
        # skips the config and falls back to a viable one.
        #
        # Skip this check when the joint referenced thread count fits
        # within ``MAX_THREADS_PER_BLOCK``: in that case any per-axis
        # mismatch comes from a strategy that intentionally launches
        # fewer threads than the codegen "references" (e.g. an outer-
        # tile axis with a single logical lane that still appears in a
        # warp-reduction call), and the CuTe runtime degenerates the
        # reduction to the live lanes without losing data.
        #
        # Skip this check for tcgen05-specialized matmul kernels: those
        # use a custom role-warp launch shape that intentionally differs
        # from the SIMT thread-axis counts (the latter being how many
        # threads the user-visible per-element loops would expect). The
        # tcgen05 code paths know which lanes are alive on their own.
        # Also skip when the kernel has any matmul / MMA call (checks for
        # cute.gemm calls): those paths cooperate within a warp through
        # CUTLASS MMA intrinsics that don't depend on the SIMT axis layout,
        # so the strategy can intentionally launch fewer threads on a
        # reduction axis (e.g. K) than the codegen "references" through the
        # strategy's per-block thread count.
        from .thread_budget import check_thread_limit

        def _emits_cute_gemm(stmt: ast.AST) -> bool:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "gemm"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "cute"
                ):
                    return True
            return False

        kernel_has_mma = any(
            _emits_cute_gemm(stmt)
            for stmt in [*device_function.preamble, *device_function.body]
        )
        if (
            tcgen05_compact_dims is None
            and not specialized_root_tcgen05
            and not kernel_has_mma
        ):
            referenced_threads = functools.reduce(
                operator.mul, codegen.referenced_thread_block_dims, 1
            )
            if referenced_threads > MAX_THREADS_PER_BLOCK:
                for axis, ref_size in enumerate(codegen.referenced_thread_block_dims):
                    if ref_size > 1 and axis < len(dims) and dims[axis] < ref_size:
                        raise exc.BackendUnsupported(
                            self.name,
                            (
                                f"launch dims {tuple(dims)} under-dimension"
                                f" referenced_thread_block_dims="
                                f"{tuple(codegen.referenced_thread_block_dims)}"
                                f" (axis {axis}: launched={dims[axis]} <"
                                f" referenced={ref_size}). Codegen would access"
                                f" nonexistent threads — joint requested"
                                f" thread count {referenced_threads} >"
                                f" {MAX_THREADS_PER_BLOCK}."
                            ),
                        )

        check_thread_limit(dims[0] * dims[1] * dims[2], context=str(tuple(dims)))
        return launcher_args_with_compile_options(
            f"block=({dims[0]}, {dims[1]}, {dims[2]})"
        )

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
    ) -> list[str]:
        if not tensor_host_args:
            raise exc.BackendUnsupported(self.name, "kernel launch without tensor args")
        out = [*args]
        if has_rng_ops:
            out.append("_rng_seed_buffer")
        out.extend(self.launcher_keyword_args(config, has_barrier=has_barrier))
        return out

    def create_loop_strategy(
        self, fn: DeviceFunction, block_ids: list[int], config: Config
    ) -> TileStrategy:
        from ..compile_environment import CompileEnvironment
        from ..device_ir import ForLoopGraphInfo
        from ..device_ir import ReductionLoopGraphInfo
        from ..host_function import HostFunction
        from ..tile_strategy import CuteFlattenedTileStrategy
        from ..tile_strategy import CuteNDTileStrategy

        env = CompileEnvironment.current()
        device_ir = HostFunction.current().device_ir
        block_size_infos = [env.block_sizes[i] for i in block_ids]
        flattened = block_size_infos[0].is_flattened(config)
        loop_order = env.config_spec.loop_orders.config_get(
            config.loop_orders, block_ids[0]
        ) or [*range(len(block_ids))]
        l2_grouping = env.config_spec.l2_groupings.config_get(
            config.l2_groupings, block_ids[0], 1
        )
        has_device_loops = any(
            isinstance(graph, ForLoopGraphInfo)
            and not isinstance(graph, ReductionLoopGraphInfo)
            for graph in fn.codegen.codegen_graphs
        )
        has_dynamic_shape = any(env.block_sizes[i].size is None for i in block_ids)
        grid_ids = {bid for ids in device_ir.grid_block_ids for bid in ids}
        num_threads_config = [
            int(env.config_spec.num_threads.config_get(config.num_threads, block_id, 0))
            for block_id in block_ids
        ]
        # Compute the total thread count across all block dimensions
        # (grid + device loops) to check against the hardware limit.
        # When it would exceed 1024, default device-loop (non-grid)
        # dimensions to 1 thread to avoid budget overflow.
        from .thread_budget import MAX_THREADS_PER_BLOCK

        def _shrink_auto_thread_counts(
            nd_block_size: Sequence[object], thread_limit: int
        ) -> int:
            int_positions: list[int] = []
            int_block_sizes: dict[int, int] = {}
            for i, block_size in enumerate(nd_block_size):
                if isinstance(block_size, int):
                    int_positions.append(i)
                    int_block_sizes[i] = block_size
            resolved_threads = [
                num_threads_config[i]
                if num_threads_config[i] > 0
                else int_block_sizes[i]
                for i in int_positions
            ]
            auto_positions = {
                pos
                for pos, block_idx in enumerate(int_positions)
                if num_threads_config[block_idx] == 0
            }
            static_threads = functools.reduce(operator.mul, resolved_threads, 1)
            while static_threads > thread_limit and auto_positions:
                shrink_idx = max(
                    (pos for pos in auto_positions if resolved_threads[pos] > 1),
                    key=lambda pos: resolved_threads[pos],
                    default=None,
                )
                if shrink_idx is None:
                    break
                block_idx = int_positions[shrink_idx]
                block_size = int_block_sizes[block_idx]
                next_threads = _largest_divisor_at_most(
                    block_size, resolved_threads[shrink_idx] - 1
                )
                if next_threads == resolved_threads[shrink_idx]:
                    break
                resolved_threads[shrink_idx] = next_threads
                num_threads_config[block_idx] = next_threads
                static_threads = functools.reduce(operator.mul, resolved_threads, 1)
            return static_threads

        active_loop_block_ids = _active_loop_block_ids(fn)
        all_block_infos = [env.block_sizes[i] for i in sorted(active_loop_block_ids)]
        total_threads = 1
        for info in all_block_infos:
            if info.reduction:
                continue
            bs = info.from_config(config)
            if isinstance(bs, int):
                nt = int(
                    env.config_spec.num_threads.config_get(
                        config.num_threads, info.block_id, 0
                    )
                )
                total_threads *= nt if nt > 0 else bs
        if total_threads > MAX_THREADS_PER_BLOCK:
            for i, block_id in enumerate(block_ids):
                if num_threads_config[i] == 0 and block_id not in grid_ids:
                    num_threads_config[i] = 1
        if (
            has_device_loops
            or has_dynamic_shape
            or len(device_ir.grid_block_ids) != 1
            or (len(block_ids) > 1 and not flattened)
        ):
            known_equal = getattr(env, "known_equal", None)

            def sizes_known_equal(
                lhs: int | torch.SymInt,
                rhs: int | torch.SymInt,
            ) -> bool:
                if known_equal is not None:
                    return known_equal(lhs, rhs)
                return bool(lhs == rhs)

            nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
            original_num_threads_config = list(num_threads_config)
            mma_candidate = _is_mma_candidate_loop(
                fn,
                block_ids,
                block_sizes=nd_block_size,
                num_threads_config=original_num_threads_config,
                grid_ids=grid_ids,
            )
            should_filter_inactive_block_ids = len(block_ids) > 1
            inactive_block_ids: set[int] = set()
            if should_filter_inactive_block_ids:
                used_block_ids = _graph_used_block_ids(fn, block_ids)
                if not used_block_ids:
                    used_block_ids = set(block_ids)
                for block_id in tuple(used_block_ids):
                    block_size = env.block_sizes[block_id].size
                    if block_size is None or not isinstance(
                        block_size, (int, torch.SymInt)
                    ):
                        continue
                    for other_block_id in block_ids:
                        if other_block_id == block_id:
                            continue
                        other_size = env.block_sizes[other_block_id].size
                        if other_size is None or not isinstance(
                            other_size, (int, torch.SymInt)
                        ):
                            continue
                        if sizes_known_equal(block_size, other_size):
                            used_block_ids.add(other_block_id)
                inactive_block_ids = set(block_ids) - used_block_ids
                for i, block_id in enumerate(block_ids):
                    if block_id in inactive_block_ids:
                        num_threads_config[i] = 1
            is_device_loop = any(bid not in grid_ids for bid in block_ids)
            reduction_axis_reserve = (
                1
                if any(info.reduction for info in env.block_sizes)
                and self.reduction_axis_first()
                else 0
            )

            def uses_thread_axis_for(
                block_id: int, block_size: object, num_threads: int
            ) -> bool:
                if block_id in inactive_block_ids:
                    return False
                if num_threads > 0:
                    return num_threads > 1
                return not (isinstance(block_size, int) and block_size == 1)

            consumed_grid_axes = 0
            if is_device_loop:
                for grid_block_id in grid_ids:
                    grid_info = env.block_sizes[grid_block_id]
                    grid_block_size = grid_info.from_config(config)
                    grid_threads = int(
                        env.config_spec.num_threads.config_get(
                            config.num_threads,
                            grid_block_id,
                            0,
                        )
                    )
                    if uses_thread_axis_for(
                        grid_block_id,
                        grid_block_size,
                        grid_threads,
                    ):
                        consumed_grid_axes += 1
            available_axes = max(0, 3 - reduction_axis_reserve - consumed_grid_axes)

            def current_strategy_axes() -> int:
                return sum(
                    int(uses_thread_axis_for(block_id, block_size, threads))
                    for block_id, block_size, threads in zip(
                        block_ids,
                        nd_block_size,
                        num_threads_config,
                        strict=True,
                    )
                )

            while current_strategy_axes() > available_axes:
                candidates = [
                    i
                    for i, (block_id, block_size, threads) in enumerate(
                        zip(block_ids, nd_block_size, num_threads_config, strict=True)
                    )
                    if threads == 0
                    and uses_thread_axis_for(block_id, block_size, threads)
                    and (not is_device_loop or block_id not in grid_ids)
                ]
                if not candidates:
                    break
                num_threads_config[candidates[-1]] = 1
            thread_limit = MAX_THREADS_PER_BLOCK
            if len(block_ids) > 1 and _loop_contains_matmul(fn, block_ids):
                forced_mma_impl = os.environ.get("HELION_CUTE_MMA_IMPL", "auto")
                if (
                    mma_candidate
                    or _kernel_has_specialized_mma_path(fn, config=config)
                    or (
                        _loop_may_use_mma(fn, block_ids)
                        and not _loop_contains_atomic(fn, block_ids)
                        and forced_mma_impl.strip().lower() != "auto"
                    )
                ):
                    thread_limit = MAX_THREADS_PER_BLOCK
                else:
                    # Matmul-heavy CuTe kernels with no viable MMA path, and
                    # especially atomic-accumulating split-K loops, can be
                    # register/smem limited well before the 1024-thread hard
                    # cap. Keep those auto-threaded ND tiles within 256
                    # threads and let lane loops cover the rest.
                    thread_limit = min(thread_limit, 256)
            if should_filter_inactive_block_ids and mma_candidate:
                inactive_block_ids.clear()
                num_threads_config = [
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                    for block_id in block_ids
                ]
            # OPTION B (matmul-contraction synthetic-lane fix): when a reduction
            # block is the *contraction* (K) axis of a matmul lowered through the
            # scalar fallback, it must keep enough real hardware threads to cover
            # its full extent so the cross-warp shared-memory reduction sums the
            # whole K.  Otherwise the budget below would hand the free tile axes
            # the threads and leave K split into ``threads x synthetic-lane`` -
            # the reduction then sums only the thread lanes, never the synthetic
            # lanes, so each contracted dot product covers only a fraction of K.
            # Reserve the K block's full thread extent up front by shrinking the
            # thread limit available to the free tile axes; the contraction axis
            # then claims that budget when its reduction strategy is created and
            # the synthetic lane is pushed onto the free tile axes instead.
            reserved_contraction_threads = self._cute_matmul_contraction_thread_reserve(
                fn, block_ids
            )
            if reserved_contraction_threads > 1:
                # Budget left for the free tile axes after the contraction axis
                # has claimed its full thread extent.
                free_budget = max(1, thread_limit // reserved_contraction_threads)
                # The free axes are built across separate ``create_loop_strategy``
                # calls but their thread counts multiply, so split the budget so
                # the product over every free axis stays within ``free_budget``.
                free_axes = self._cute_free_auto_thread_axis_count(fn, config)
                per_axis_limit = free_budget
                while per_axis_limit > 1 and per_axis_limit**free_axes > free_budget:
                    per_axis_limit //= 2
                thread_limit = max(1, per_axis_limit)
            static_threads = _shrink_auto_thread_counts(nd_block_size, thread_limit)
            from .thread_budget import check_thread_limit

            # Detect MMA-compatible K-loops: device loops containing
            # addmm/mm with float16/bfloat16 operands.
            mma_mode = False
            if is_device_loop:
                mma_mode = _detect_specialized_mma_loop(
                    fn,
                    block_ids,
                    block_sizes=nd_block_size,
                    config=config,
                )
                if not mma_mode and _detect_attention_mma_loop(
                    fn,
                    block_ids,
                    config=config,
                ):
                    # Fused tcgen05 flash-attention: suppress synthetic lane
                    # loops (mma_mode) and run a single 128-thread warpgroup
                    # like the spike. The dedicated flash codegen emits the
                    # whole device body; the FX-graph statement walk is bypassed.
                    mma_mode = True
                    fn.cute_state.attention_flash_block_ids = list(block_ids)
            elif (
                len(device_ir.grid_block_ids) == 1
                and block_ids == device_ir.grid_block_ids[0]
            ):
                specialized_mma_impl = _kernel_specialized_mma_impl(fn, config=config)
                if specialized_mma_impl == "tcgen05" and len(nd_block_size) == 2:
                    from .cute_mma import _tcgen05_root_m_threads

                    root_m_threads = (
                        _tcgen05_root_m_threads(
                            int(nd_block_size[0]), int(nd_block_size[1])
                        )
                        if num_threads_config[0] == 0
                        and isinstance(nd_block_size[0], int)
                        and isinstance(nd_block_size[1], int)
                        else num_threads_config[0]
                    )
                    root_n_threads = (
                        min(int(nd_block_size[1]), 8)
                        if num_threads_config[1] == 0
                        and isinstance(nd_block_size[1], int)
                        else num_threads_config[1]
                    )
                    num_threads_config[0] = root_m_threads
                    num_threads_config[1] = root_n_threads
                    static_threads = root_m_threads * root_n_threads

            check_thread_limit(static_threads, context=str(tuple(nd_block_size)))
            return CuteNDTileStrategy(
                fn,
                block_ids,
                block_size=nd_block_size,
                loop_order=loop_order,
                l2_grouping=l2_grouping,
                num_threads=num_threads_config,
                mma_mode=mma_mode,
                inactive_block_ids=inactive_block_ids,
            )
        nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
        block_size = functools.reduce(operator.mul, nd_block_size)  # pyrefly: ignore[incompatible-overload-residual]
        # Resolve per-axis thread counts then flatten to a single total
        all_auto = all(nt <= 0 for nt in num_threads_config)
        flat_num_threads = functools.reduce(
            operator.mul,
            (
                nt if nt > 0 else (int(bs) if isinstance(bs, int) else 0)
                for nt, bs in zip(num_threads_config, nd_block_size, strict=True)
            ),
            1,
        )
        if (
            isinstance(block_size, int)
            and flat_num_threads > MAX_THREADS_PER_BLOCK
            and all_auto
        ):
            # Auto thread budget exceeds the 1024-per-CTA cap: fall back to a
            # lane loop (each thread owns block_size // 1024 elements).
            flat_num_threads = MAX_THREADS_PER_BLOCK
        if isinstance(block_size, int) and flat_num_threads > 0:
            from .thread_budget import check_thread_limit

            check_thread_limit(flat_num_threads, context=str(block_size))
        return CuteFlattenedTileStrategy(
            fn,
            block_ids,
            block_size=block_size,
            loop_order=loop_order,
            num_threads=flat_num_threads,
        )
