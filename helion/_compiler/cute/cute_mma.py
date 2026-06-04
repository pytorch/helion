"""CuTe MMA (tensor core) codegen for matmul operations.

Generates cute.gemm calls using MmaUniversalOp for warp-level MMA.
Follows the reduction strategy pattern: initialization in outer_prefix,
per-K-tile MMA in the loop body, fragment→scalar conversion in outer_suffix.

The MMA always accumulates in float32 for precision.  Input data (float16
or bfloat16) is cast to float32 during the register load.  After the
K-loop the fragment is written to shared memory via partition_C and each
thread reads back its own scalar element, re-entering the normal
scalar-per-thread model so epilogue ops (bias, activation, cast) work.

Features:
- Works through both aten lowering (addmm/mm) and hl.dot API paths
- Shared memory staging for A and B operands with sync_threads
- Multi-warp tiling via atom_layout_mnk for larger tile sizes
- Masking for non-divisible tile boundaries
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
import textwrap
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import Node

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from ..tile_strategy import DeviceLoopState
from .aux_tensor import discover_tcgen05_aux_tensor_descriptors
from .cutedsl_compat import emit_pipeline_advance
from .device_state import CuteDeviceFunctionState
from .device_state import CuteTcgen05MatmulPlan
from .device_state import CuteTcgen05StoreValue
from .layout import MatmulExecutionKind
from .layout import MatmulExecutionPlan
from .matmul_utils import analyze_direct_grouped_n_loads
from .mma_support import get_cute_mma_support
from .strategies import TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
from .strategies import Tcgen05PersistenceModel
from .strategies import is_pure_matmul_role_lifecycle_config
from .strategies import l2_swizzle_size_from_config
from .strategies import layout_overrides_from_config
from .strategies import smem_swizzle_min_major_mode_bytes
from .strategies import tcgen05_default_epilogue_tile_expr
from .strategies import tcgen05_explicit_epilogue_tile_expr
from .strategies import tcgen05_smem_layout_expr
from .strategies import warp_spec_from_config
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from .tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_DEFAULT
from .tcgen05_constants import TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE
from .tcgen05_constants import TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TARGET1_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET2_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET3_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET4_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET5_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET6_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET7_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET9_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET9_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET9_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET10_TVM_FFI_AB_STAGES
from .tcgen05_constants import TCGEN05_TARGET10_TVM_FFI_BLOCK_K
from .tcgen05_constants import TCGEN05_TARGET10_TVM_FFI_C_STAGES
from .tcgen05_constants import TCGEN05_TARGET10_TVM_FFI_SHAPE
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES
from .tcgen05_lifecycle import Tcgen05LifecycleContext
from .tcgen05_pure_matmul import Tcgen05PureClcSchedulerObject
from .tcgen05_pure_matmul import Tcgen05PureMatmulObjectModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ..aten_lowering import LoweringContext
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from .strategies import Tcgen05WarpSpec


_TRACE_THROUGH_TARGETS = {
    torch.ops.prims.convert_element_type.default,
    # NOTE: permute is NOT included because the MMA pipeline reads
    # raw tensor data — tracing through permute would bypass the
    # data shuffle.  Permuted operands fall back to scalar codegen.
}

# Register reallocation budget for tcgen05 warp-specialized kernels.
# Producer warps (TMA loads, scheduler) only do address arithmetic and
# barrier ops, so they can give back registers; consumer warps (MMA
# exec, epilogue) need the extra budget for register-resident
# accumulators and TMEM↔RMEM staging. Values match Quack's sm100
# reference (`gemm_sm100.py`). The consumer-side ceiling is autotune-
# searchable via ``TCGEN05_CONSUMER_REGS_CONFIG_KEY`` (cycle 15 H2);
# the default 256 lives in ``tcgen05_constants.py`` so the codegen
# call site reads the per-config value and emission stays byte-
# identical at the default.
_TCGEN05_PRODUCER_REGS = 120
# 128x32 bf16 gives the validated 64 Ki-bit D-store TMA box and x32 TMEM
# drain for the current Target1 CtaGroup.TWO diagnostic path.
_TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE = (
    TCGEN05_TWO_CTA_BLOCK_M // 2,
    32,
    32,
)
# Cluster-leader (cta_rank == 0) form. Used only when V-leader semantics
# degenerate to cluster-leader -- i.e. cluster_size == V (no V-non-leader
# CTAs), today's cluster_m=2 cluster_n=1 use_2cta=True path. Preserves the
# cluster_n=1 leader predicate form.
_TCGEN05_CLUSTER_LEADER_PREDICATE = (
    "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == cutlass.Int32(0)"
)
# V-leader form: ``cta_rank % V == 0``. Required for cluster_m=2 cluster_n=2
# use_2cta=True (V=2) where V-leaders are ranks {0, 2} but the cluster-leader
# is only {0}. The V-non-leaders {1, 3} hold their V-pair's TMEM allocation
# and *do not* commit on the AB consumer-release barrier — only V-leaders
# {0, 2} do. See cute_plan.md §6.12.3 for the full diagnosis: cycle 26's hang
# at cluster_n=2 was caused by Helion using the cluster-leader form here,
# which only fires from rank 0 and races ranks {2, 3}.
_TCGEN05_V_LEADER_PREDICATE = (
    "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
    "% cutlass.Int32(2) == cutlass.Int32(0)"
)

# Named-barrier ids reserved by Helion's tcgen05 codegen. Kept as module
# constants so the codegen sites read symbolically instead of hardcoding
# magic numbers, and so the next free id is obvious if a third role-local
# barrier is added.
_TCGEN05_TMEM_ALLOC_BARRIER_ID = 1
_TCGEN05_EPILOG_SYNC_BARRIER_ID = 2


@dataclass(frozen=True)
class _Tcgen05LayoutPlan:
    """Generated CuTe variable names for the tcgen05 layout setup.

    Pure name container: every field is the textual identifier of a value
    materialized in the kernel prefix. Compile-time integer constants
    (stage counts, arrive counts, barrier ids) are not stored here; they
    live as Python ints alongside ``CuteTcgen05MatmulPlan`` and get
    inlined at the codegen call site.
    """

    exec_active: str
    smem_a_layout: str
    smem_b_layout: str
    c_layout: str
    epi_tile: str
    tmem_load_atom: str
    acc_tmem_cols: str
    tmem_holding_buf: str
    tmem_dealloc_mbar_ptr: str
    tmem_alloc_barrier: str
    tmem_allocator: str
    acc_pipeline_barriers: str
    acc_pipeline_producer_group: str
    acc_pipeline_consumer_group: str
    acc_pipeline: str
    acc_producer_state: str
    acc_consumer_state: str
    epilogue_rest_mode: str


@dataclass(frozen=True)
class _Tcgen05AuxPerDescriptorRingNames:
    """Generated CuTe variable names for one auxiliary-tensor SMEM ring.

    One instance per :class:`Tcgen05AuxTensorDescriptor` registered on
    the matmul plan when the productive-body gate fires
    (``c_input_warp_count > 0`` AND non-empty
    ``aux_tensor_descriptors``). Each ring has its own
    ``make_smem_layout_epi``-based SMEM layout, ``alloc_smem`` ptr,
    and ``make_tensor`` view; the producer body in
    ``program_id._build_c_input_warp_role_local_while`` indexes the
    ring by descriptor position, and the consumer-side flip
    in ``memory_ops._aux_subtile_load_source`` reads from the same
    SMEM tensor by looking up each
    ``_AuxStepRecord.load_node`` in the matmul plan's
    descriptor list to find its ring position.
    """

    smem_layout: str
    smem_ptr: str
    smem: str
    tma_atom: str | None
    tma_tensor: str | None


@dataclass(frozen=True)
class _Tcgen05AuxPipelinePlan:
    """Generated CuTe variable names for the C-input warp's
    auxiliary-tensor SMEM-ring pipeline (``cute_plan.md`` §7.5.3.2).

    Pure name container, mirroring ``_Tcgen05SchedPipelinePlan``.
    Each field is the textual identifier of a value materialized in
    the kernel prefix when ``_emit_tcgen05_aux_pipeline_setup``
    runs. The pipeline is allocated only when the productive-body
    gate fires (``c_input_warp_count > 0`` AND non-empty
    ``aux_tensor_descriptors``).

    ``rings`` carries the per-descriptor SMEM ring names in
    descriptor-list order — the same order
    ``CuteTcgen05MatmulPlan.aux_tensor_descriptors`` exposes them.
    The producer body indexes by position; the consumer side
    looks up by ``load_node`` identity (see
    ``memory_ops._codegen_cute_store_tcgen05_tile``'s
    ``aux_ring_index_by_step``) so multi-step chains within one
    store map cleanly onto the descriptor order.

    Consumer cooperative group: **``epi_warp_count`` (per-warp,
    NOT per-thread)**. The consumer-side flip in
    ``memory_ops._aux_subtile_load_source`` gates
    ``consumer_release(c_pipeline_aux)`` on ``elect_one()``
    (matching the sched-pipeline pattern from
    ``_build_role_local_while_with_scheduler``), so the consumer
    arrive count is per-warp. Setting per-thread would hang the
    handshake waiting for 31 missing per-warp arrivals per
    stage.
    """

    barriers: str
    producer_group: str
    consumer_group: str
    pipeline: str
    producer_state: str
    # The consumer-side flip in
    # ``memory_ops._aux_subtile_load_source`` issues
    # ``c_pipeline_aux.consumer_wait`` / ``consumer_release`` keyed
    # off this state per subtile of the per-output-tile aux region.
    consumer_state: str
    rings: tuple[_Tcgen05AuxPerDescriptorRingNames, ...]
    use_tma_load: bool
    stage_count: int
    # ``epi_tile_var`` is the matmul-plan ``epi_tile`` variable
    # name. The producer body in
    # ``program_id._build_c_input_warp_role_local_while`` uses it
    # to compute the per-subtile GMEM slice that gets cooperative-
    # copied into a SMEM ring stage. Each stage holds one
    # ``epi_tile`` worth of aux data so per-subtile staging keeps
    # the SMEM footprint small (one stage = one subtile). The
    # producer-body codegen reads ``(bm, bn)`` directly from the
    # matmul plan so no block-shape fields are plumbed through
    # this dataclass.
    epi_tile_var: str


@dataclass(frozen=True)
class _Tcgen05SchedPipelinePlan:
    """Generated CuTe variable names for the scheduler-broadcast pipeline.

    Pure name container, mirroring ``_Tcgen05LayoutPlan``: each field
    is the textual identifier of a value materialized in the kernel
    prefix when ``_emit_sched_pipeline_setup`` runs.

    The ``clc_*`` fields are populated only under
    ``Tcgen05PersistenceModel.CLC_PERSISTENT`` (G2-H, cute_plan.md);
    empty strings on the static path. They name SMEM storage
    + an mbarrier that the scheduler-warp loop body uses to issue
    ``nvvm.clusterlaunchcontrol_try_cancel`` and read back the next
    cluster's CTA id (or a "canceled" sentinel) per persistent-loop
    iteration.
    """

    barriers: str
    producer_group: str
    consumer_group: str
    pipeline: str
    producer_state: str
    consumer_state: str
    # CLC SMEM/mbarrier handles. Only emitted/used on the CLC path.
    clc_response_smem_ptr: str = ""
    clc_response_tensor: str = ""
    clc_mbar_smem_ptr: str = ""
    clc_mbar_tensor: str = ""
    clc_mbar_phase: str = ""


class _ConfigLike(Protocol):
    def get(self, key: str, default: object = ..., /) -> object: ...


def _iter_node_inputs(arg: object) -> list[Node]:
    nodes: list[Node] = []
    if isinstance(arg, Node):
        nodes.append(arg)
    elif isinstance(arg, (list, tuple)):
        for item in arg:
            nodes.extend(_iter_node_inputs(item))
    elif isinstance(arg, dict):
        for item in arg.values():
            nodes.extend(_iter_node_inputs(item))
    return nodes


def _collect_node_dependencies(node: Node) -> set[Node]:
    required: set[Node] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in required:
            continue
        required.add(current)
        for arg in current.args:
            stack.extend(_iter_node_inputs(arg))
        for arg in current.kwargs.values():
            stack.extend(_iter_node_inputs(arg))
    return required


def _collective_load_dependency_nodes(
    load_node: Node,
    collective_dependency_nodes: set[Node],
    terminal_load_nodes: set[Node],
) -> tuple[Node, ...]:
    """Return exclusive FX dependency nodes for a tcgen05 collective load."""
    dependencies = _collect_node_dependencies(load_node)
    exclusive_nodes = set(collective_dependency_nodes)
    terminal_load_nodes = terminal_load_nodes & exclusive_nodes
    changed = True
    while changed:
        changed = False
        for dependency in tuple(exclusive_nodes):
            if dependency in terminal_load_nodes:
                continue
            if any(user not in exclusive_nodes for user in dependency.users):
                exclusive_nodes.remove(dependency)
                changed = True
    return tuple(
        sorted(dependencies & exclusive_nodes, key=lambda dependency: dependency.name)
    )


def _register_collective_handled_loads(
    cute_state: CuteDeviceFunctionState, *load_nodes: Node
) -> None:
    # This is called both by the early lane-loop-suppression probe and by MMA
    # emission. Registration is set-based, so repeating it is idempotent and
    # keeps both call sites local to the decision they support.
    collective_dependency_nodes: set[Node] = set()
    for load_node in load_nodes:
        collective_dependency_nodes.update(_collect_node_dependencies(load_node))
    terminal_load_nodes = set(load_nodes)
    for load_node in load_nodes:
        dependency_nodes = _collective_load_dependency_nodes(
            load_node, collective_dependency_nodes, terminal_load_nodes
        )
        cute_state.register_collective_handled_load(
            load_node.name,
            dependency_nodes=dependency_nodes,
        )


def _mma_loop_is_exclusive(node: Node) -> bool:
    """Require the loop body to contain only the candidate MMA dataflow."""
    required = _collect_node_dependencies(node)
    for graph_node in node.graph.nodes:
        if graph_node in required or graph_node.op in {
            "placeholder",
            "output",
            "get_attr",
        }:
            continue
        if graph_node.op == "call_function":
            return False
    return True


def _trace_to_load(node: Node) -> Node | None:
    """Trace through casts/permutes to the underlying load node."""
    from ...language import memory_ops

    cur = node
    while cur.op == "call_function" and cur.target is not memory_ops.load:
        if cur.target not in _TRACE_THROUGH_TARGETS:
            return None
        input_nodes = [a for a in cur.args if isinstance(a, Node)]
        if len(input_nodes) != 1:
            return None
        cur = input_nodes[0]

    if cur.op != "call_function" or cur.target is not memory_ops.load:
        return None
    return cur


def _trace_to_load_tensor(node: Node) -> tuple[Node, str, torch.Tensor] | None:
    """Trace through casts/permutes to find the underlying load tensor.

    Only traces through data-preserving ops (type casts, permute).
    Does NOT trace through arithmetic (add, mul, etc.) because the MMA
    pipeline reads raw tensor data and those ops would be skipped.
    """
    load_node = _trace_to_load(node)
    if load_node is None:
        return None
    tensor_node = load_node.args[0]
    if not isinstance(tensor_node, Node):
        return None
    fake = tensor_node.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    return load_node, tensor_node.name, fake


def _has_mma_operands(lhs_node: Node, rhs_node: Node) -> bool:
    """Check if lhs/rhs come from loads with MMA-compatible dtypes."""
    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return False
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
    supported = {torch.float16, torch.bfloat16, torch.float32}
    return (
        lhs_fake.dtype in supported
        and rhs_fake.dtype in supported
        and lhs_fake.dtype == rhs_fake.dtype
        and lhs_fake.ndim == 2
        and rhs_fake.ndim == 2
    )


def is_mma_compatible_aten(node: Node, with_acc: bool) -> bool:
    """Check if an aten addmm/mm node can use MMA."""
    args = node.args
    if with_acc:
        if len(args) < 3:
            return False
        acc_node = args[0]
        lhs_node, rhs_node = args[1], args[2]
        if isinstance(acc_node, Node):
            acc_val = acc_node.meta.get("val")
            if isinstance(acc_val, torch.Tensor) and acc_val.ndim != 2:
                return False
    else:
        if len(args) < 2:
            return False
        lhs_node, rhs_node = args[0], args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return False
    return _has_mma_operands(lhs_node, rhs_node)


def is_mma_compatible_dot(node: Node) -> bool:
    """Check if an hl.dot FX node can use MMA."""
    # dot args: (lhs, rhs, acc_or_None, out_dtype_or_None)
    if len(node.args) < 2:
        return False
    acc_node = node.args[2] if len(node.args) > 2 else None
    lhs_node, rhs_node = node.args[0], node.args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return False
    if isinstance(acc_node, Node):
        acc_val = acc_node.meta.get("val")
        if isinstance(acc_val, torch.Tensor) and acc_val.ndim != 2:
            return False
    return _has_mma_operands(lhs_node, rhs_node)


def can_codegen_cute_mma_dot(node: Node) -> bool:
    """Return True when hl.dot both supports MMA and matches MMA dtype semantics."""
    if not is_mma_compatible_dot(node):
        return False
    if not _mma_result_can_be_deferred(node) or not _mma_loop_is_exclusive(node):
        return False

    lhs_node = node.args[0]
    rhs_node = node.args[1]
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return False

    if not _needs_f32_accumulator(lhs_val.dtype, rhs_val.dtype):
        return True

    acc_dtype: torch.dtype | None = None
    if len(node.args) > 2 and isinstance(node.args[2], Node):
        acc_val = node.args[2].meta.get("val")
        if isinstance(acc_val, torch.Tensor):
            acc_dtype = acc_val.dtype

    out_dtype = node.args[3] if len(node.args) > 3 else None
    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        return False

    return out_dtype in (None, torch.float32) and acc_dtype in (
        None,
        torch.float32,
    )


def can_codegen_cute_mma_aten(node: Node, with_acc: bool) -> bool:
    return (
        is_mma_compatible_aten(node, with_acc)
        and _mma_result_can_be_deferred(node)
        and _mma_loop_is_exclusive(node)
    )


def _graph_signature(graph: torch.fx.Graph) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    for node in graph.nodes:
        target = node.op
        if node.op == "call_function":
            target = getattr(node.target, "__name__", str(node.target))
        signature.append((node.op, target))
    return tuple(signature)


def _graph_tensor_output_count(graph: torch.fx.Graph) -> int:
    output_nodes = list(graph.find_nodes(op="output"))
    if not output_nodes:
        return 0
    (output_node,) = output_nodes
    outputs: set[Node] = set()
    for node in _iter_node_inputs(output_node.args):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            outputs.add(node)
    return len(outputs)


def _trace_acc_init_node(node: Node) -> Node | None:
    from ...language import _tracing_ops
    from ..device_ir import NodeArgsGraphInfo
    from ..host_function import HostFunction

    current = node
    seen: set[Node] = set()
    while current not in seen:
        seen.add(current)
        if current.op == "placeholder":
            current_placeholders = list(current.graph.find_nodes(op="placeholder"))
            current_signature = _graph_signature(current.graph)
            for graph_info in HostFunction.current().device_ir.graphs:
                if current.graph is graph_info.graph and isinstance(
                    graph_info, NodeArgsGraphInfo
                ):
                    if _graph_tensor_output_count(current.graph) > 1:
                        return current
                    current = graph_info.placeholder_to_outer_arg(current)
                    break
                if not isinstance(graph_info, NodeArgsGraphInfo):
                    continue
                if _graph_signature(graph_info.graph) != current_signature:
                    continue
                if _graph_tensor_output_count(graph_info.graph) > 1:
                    return current
                for placeholder, outer_node in zip(
                    current_placeholders,
                    graph_info.node_args,
                    strict=True,
                ):
                    if placeholder is current:
                        current = outer_node
                        break
                else:
                    continue
                break
            else:
                return current
            continue
        if current.op != "call_function":
            return current
        if current.target is _tracing_ops._new_var:
            (arg,) = current.args
            if not isinstance(arg, Node):
                return None
            current = arg
            continue
        if current.target is _tracing_ops._phi:
            lhs = current.args[0]
            if not isinstance(lhs, Node):
                return None
            current = lhs
            continue
        return current
    return None


def _is_zero_init_acc_node(node: Node) -> bool:
    from ...language import creation_ops

    init_node = _trace_acc_init_node(node)
    if init_node is None or init_node.op != "call_function":
        return False
    if init_node.target is creation_ops.full:
        value = init_node.args[1]
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
    return False


def _physical_mma_coord_expr(
    cg: GenerateAST,
    block_id: int,
) -> str:
    """Return the physical thread coordinate for an MMA output axis."""
    grid_state = cg.current_grid_state
    if grid_state is None:
        return "cutlass.Int32(0)"
    thread_axis = grid_state.block_thread_axes.get(block_id)
    if thread_axis is None:
        return "cutlass.Int32(0)"
    return f"cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])"


def _local_mma_coord_expr(
    cg: GenerateAST,
    block_id: int,
) -> str:
    """Return the current block-local coordinate for an MMA output axis.

    Same as ``_physical_mma_coord_expr`` plus a lane offset when the grid
    strategy has registered a per-block lane var (the lane-loop fast path
    serializes `elements_per_thread` consecutive elements per physical
    thread).
    """
    coord = _physical_mma_coord_expr(cg, block_id)
    grid_state = cg.current_grid_state
    if grid_state is None or grid_state.block_thread_axes.get(block_id) is None:
        return coord

    strategy = grid_state.strategy
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not isinstance(lane_vars, dict) or block_id not in lane_vars:
        return coord

    elements_per_thread_fn = getattr(strategy, "_elements_per_thread_for_block", None)
    if not callable(elements_per_thread_fn):
        return coord
    elements_per_thread = elements_per_thread_fn(block_id)
    lane_var = lane_vars[block_id]
    if elements_per_thread == 1:
        return f"{coord} + cutlass.Int32({lane_var})"
    return f"{coord} * cutlass.Int32({elements_per_thread}) + cutlass.Int32({lane_var})"


def _grid_thread_extent(cg: GenerateAST, block_id: int) -> int:
    grid_state = cg.current_grid_state
    if grid_state is None:
        return 1
    thread_axis = grid_state.block_thread_axes.get(block_id)
    if thread_axis is None:
        return 1
    return grid_state.thread_axis_sizes.get(thread_axis, 1)


@dataclass(frozen=True)
class _MmaRoleCoordinatePlan:
    """Logical MMA/role coordinates for the current CUDA launch topology."""

    mma_m_coord: str
    mma_n_coord: str
    mma_m_thread_extent: int
    mma_active_n_threads: int

    def mma_tidx_expr(self) -> str:
        return (
            f"{self.mma_m_coord} + "
            f"({self.mma_n_coord}) * cutlass.Int32({self.mma_m_thread_extent})"
        )

    def mma_active_expr(self) -> str:
        return f"({self.mma_n_coord}) < cutlass.Int32({self.mma_active_n_threads})"


def _mma_epi_tidx_expr(*, lane_idx: str, warp_idx: str, epi_active: str) -> str:
    return (
        f"{lane_idx} + {warp_idx} * cutlass.Int32(32) "
        f"if {epi_active} else cutlass.Int32(0)"
    )


def _block_axis_mma_role_coordinate_plan(
    cg: GenerateAST,
    *,
    m_block_id: int,
    n_block_id: int,
    mma_m_thread_extent: int,
    mma_active_n_threads: int,
) -> _MmaRoleCoordinatePlan:
    """Return the current block-axis-backed MMA role-coordinate plan."""
    return _MmaRoleCoordinatePlan(
        mma_m_coord=_physical_mma_coord_expr(cg, m_block_id),
        mma_n_coord=_physical_mma_coord_expr(cg, n_block_id),
        mma_m_thread_extent=mma_m_thread_extent,
        mma_active_n_threads=mma_active_n_threads,
    )


def _flat_mma_role_coordinate_plan(
    *,
    lane_idx: str,
    warp_idx: str,
    mma_active_n_threads: int,
) -> _MmaRoleCoordinatePlan:
    """Return logical MMA coordinates derived from a flat 8-warp launch."""
    return _MmaRoleCoordinatePlan(
        mma_m_coord=lane_idx,
        mma_n_coord=warp_idx,
        mma_m_thread_extent=32,
        mma_active_n_threads=mma_active_n_threads,
    )


def _grid_cta_thread_count(cg: GenerateAST) -> int:
    grid_state = cg.current_grid_state
    if grid_state is None:
        return 1
    cta_threads = 1
    for size in grid_state.thread_axis_sizes.values():
        cta_threads *= size
    return cta_threads


def _get_mma_k_loop_info(
    cg: GenerateAST,
    env: CompileEnvironment,
    lhs_fake: torch.Tensor,
    rhs_fake: torch.Tensor,
    fx_node: Node | None = None,
) -> tuple[DeviceLoopState, int, str, int] | None:
    """Return the active reduction loop for the operands' shared K dimension."""
    if fx_node is not None:
        from ..device_ir import ForLoopGraphInfo

        graph_k_block_ids = [
            graph_info.block_ids
            for graph_info in cg.codegen_graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.graph is fx_node.graph
        ]
        if len(graph_k_block_ids) == 1:
            active_graph_block_ids = [
                block_id
                for block_id in graph_k_block_ids[0]
                if any(
                    isinstance(loop_state, DeviceLoopState)
                    for loop_state in cg.active_device_loops.get(block_id, ())
                )
            ]
            if len(active_graph_block_ids) == 1:
                k_block_id = active_graph_block_ids[0]
                loops = cg.active_device_loops.get(k_block_id)
                assert loops is not None
                device_loop = next(
                    (
                        loop_state
                        for loop_state in reversed(loops)
                        if isinstance(loop_state, DeviceLoopState)
                    ),
                    None,
                )
                if device_loop is not None:
                    block_size = env.block_sizes[k_block_id].from_config(
                        cg.device_function.config
                    )
                    if isinstance(block_size, int):
                        return (
                            device_loop,
                            k_block_id,
                            device_loop.strategy.offset_var(k_block_id),
                            block_size,
                        )

    lhs_k_block_id = env.resolve_block_id(lhs_fake.shape[1])
    rhs_k_block_id = env.resolve_block_id(rhs_fake.shape[0])
    candidate_block_ids: set[int] = set()
    if (
        lhs_k_block_id is not None
        and rhs_k_block_id is not None
        and lhs_k_block_id == rhs_k_block_id
    ):
        candidate_block_ids.add(lhs_k_block_id)
    else:
        for block_id, loops in cg.active_device_loops.items():
            if not any(isinstance(loop_state, DeviceLoopState) for loop_state in loops):
                continue
            size = env.block_sizes[block_id].size
            if not isinstance(size, int | torch.SymInt):
                continue
            if env.known_equal(size, lhs_fake.shape[1]) and env.known_equal(
                size, rhs_fake.shape[0]
            ):
                candidate_block_ids.add(block_id)

    if len(candidate_block_ids) != 1:
        return None

    (k_block_id,) = tuple(candidate_block_ids)
    loops = cg.active_device_loops.get(k_block_id)
    assert loops is not None

    device_loop = next(
        (
            loop_state
            for loop_state in reversed(loops)
            if isinstance(loop_state, DeviceLoopState)
        ),
        None,
    )
    if device_loop is None:
        return None

    block_size = env.block_sizes[k_block_id].from_config(cg.device_function.config)
    if not isinstance(block_size, int):
        return None

    return (
        device_loop,
        k_block_id,
        device_loop.strategy.offset_var(k_block_id),
        block_size,
    )


def _device_loop_begin_expr(device_loop: DeviceLoopState) -> str:
    loop_iter = device_loop.for_node.iter
    if not isinstance(loop_iter, ast.Call) or not loop_iter.args:
        return "cutlass.Int32(0)"
    if len(loop_iter.args) == 1:
        return "cutlass.Int32(0)"
    return ast.unparse(loop_iter.args[0])


def _has_non_root_lane_loops(
    cg: GenerateAST, *, allowed_loop_states: tuple[DeviceLoopState, ...] = ()
) -> bool:
    seen: set[int] = set()
    allowed_ids = {id(loop_state) for loop_state in allowed_loop_states}
    for loops in cg.active_device_loops.values():
        for loop_state in loops:
            key = id(loop_state)
            if key in seen:
                continue
            seen.add(key)
            if loop_state is cg.current_grid_state or key in allowed_ids:
                continue
            strategy = getattr(loop_state, "strategy", None)
            lane_vars = getattr(strategy, "_lane_var_by_block", None)
            if lane_vars:
                return True
    return False


def prepare_cute_collective_lane_loop_suppression(
    cg: GenerateAST, graph: torch.fx.Graph
) -> None:
    from ..compile_environment import CompileEnvironment

    grid_state = cg.current_grid_state
    if grid_state is None:
        return

    env = CompileEnvironment.current()
    if env.backend_name != "cute":
        return

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is torch.ops.aten.addmm.default:
            with_acc = True
            lhs_node = node.args[1]
            rhs_node = node.args[2]
            if not can_codegen_cute_mma_aten(node, with_acc):
                continue
        elif node.target is torch.ops.aten.mm.default:
            with_acc = False
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            if not can_codegen_cute_mma_aten(node, with_acc):
                continue
        elif can_codegen_cute_mma_dot(node):
            lhs_node = node.args[0]
            rhs_node = node.args[1]
        else:
            continue

        if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
            continue

        lhs_info = _trace_to_load_tensor(lhs_node)
        rhs_info = _trace_to_load_tensor(rhs_node)
        if lhs_info is None or rhs_info is None:
            continue
        lhs_load, _, lhs_fake = lhs_info
        rhs_load, _, rhs_fake = rhs_info
        if lhs_fake.ndim != 2 or rhs_fake.ndim != 2:
            continue

        if not (
            isinstance(lhs_fake.shape[0], int)
            and isinstance(rhs_fake.shape[1], int)
            and isinstance(lhs_fake.shape[1], int)
        ):
            continue
        bm = bn = bk = None
        candidate_block_ids = [*grid_state.block_ids]
        if (
            k_loop_info := _get_mma_k_loop_info(
                cg, env, lhs_fake, rhs_fake, fx_node=node
            )
        ) is not None:
            _, k_block_id, _, k_block_size = k_loop_info
            candidate_block_ids.append(k_block_id)
            bk = int(k_block_size)
        for bid in dict.fromkeys(candidate_block_ids):
            size = env.block_sizes[bid].size
            bs = cg.device_function.resolved_block_size(bid)
            if not isinstance(bs, int):
                continue
            if isinstance(size, (int, torch.SymInt)):
                if bm is None and env.known_equal(size, lhs_fake.shape[0]):
                    bm = int(bs)
                elif bn is None and env.known_equal(size, rhs_fake.shape[1]):
                    bn = int(bs)
                elif bk is None and env.known_equal(size, lhs_fake.shape[1]):
                    bk = int(bs)
        if bm is None or bn is None or bk is None:
            continue
        if (
            _choose_mma_impl(
                lhs_fake.dtype, bm=bm, bn=bn, bk=bk, config=cg.device_function.config
            )
            != "tcgen05"
        ):
            continue
        if (
            len(lhs_load.users) != 1
            or len(rhs_load.users) != 1
            or next(iter(lhs_load.users)) is not node
            or next(iter(rhs_load.users)) is not node
        ):
            continue

        # Mirror the real codegen bailout in ``_emit_mma_pipeline`` (it returns
        # ``None`` and falls back to the scalar matmul path when non-root lane
        # loops are active, see the ``_has_non_root_lane_loops`` guard there).
        # If we predicted the collective tcgen05 path here but codegen actually
        # takes the scalar fallback, requesting root lane-loop suppression would
        # drop the synthetic-lane index/mask definitions for the grid axis and
        # produce a ``NameError`` at runtime. Only register the loads / request
        # suppression when the collective path will truly be taken.
        if _has_non_root_lane_loops(cg):
            continue

        cute_state = cg.device_function.cute_state
        _register_collective_handled_loads(cute_state, lhs_load, rhs_load)
        if grid_state.has_lane_loops():
            cute_state.request_root_lane_loop_suppression()


def _mma_result_can_be_deferred(node: Node) -> bool:
    """Return True when the node value is only consumed after the K loop finishes."""
    return all(user.op == "output" for user in node.users)


@dataclass(frozen=True)
class _PerKiterTmaArgs:
    """Variable names + flags threaded into the per-K-iter TMA builders.

    All ``str`` fields name a Python identifier in the generated code.
    Only valid when the tcgen05 TMA path is active, so every name is
    guaranteed bound at the call site.
    """

    tma_pipeline: str
    tma_producer_state: str
    tma_consumer_state: str
    tma_producer_try_token: str
    tma_consumer_try_token: str
    tma_barrier_ptr: str
    tma_full_tile: str
    tma_next_full_tile: str
    tma_next_consumer_tile: str
    tma_warp: str
    tma_atom_a: str
    tma_atom_b: str
    tma_gA: str
    tma_gB: str
    tma_sA: str
    tma_sB: str
    tma_k_tile: str
    tma_a_mcast_mask: str
    tma_b_mcast_mask: str
    ab_stage_count: int
    is_two_cta: bool
    use_tma_b_mcast_mask: bool
    use_tma_a: bool
    use_tma_b: bool
    skip_producer_acquire: bool
    skip_producer_advance: bool
    skip_consumer_wait: bool
    exec_active: str
    scalar_load_a: ast.stmt
    scalar_load_b: ast.stmt
    # ``cluster_n`` is only consulted when ``is_two_cta=True`` to pick the
    # V-leader vs cluster-leader form for the AB consumer-release predicate
    # (cute_plan.md §6.12.7). Default 1 preserves byte-identity for the
    # validated cluster_m=2 cluster_n=1 path.
    cluster_n: int = 1
    # Static-full one-CTA pipelined TMA loops can drop the per-K runtime
    # full-tile branch and scalar fallback. Non-pipelined/asymmetric or two-CTA
    # TMA paths must keep the guarded fallback path.
    static_full_tiles: bool = False


def _kloop_tma_copy_a_src(args: _PerKiterTmaArgs, *, k_offset: str) -> str:
    """Per-K-iter TMA copy source for A; ``""`` when A is not TMA-loaded.

    A only multicasts in 2-CTA mode (asymmetric vs. B, which can also
    multicast across cluster CTAs).
    """
    if not args.use_tma_a:
        return ""
    mcast = f", mcast_mask={args.tma_a_mcast_mask}" if args.is_two_cta else ""
    return (
        f"    cute.copy({args.tma_atom_a}, "
        f"{args.tma_gA}[None, {k_offset}], "
        f"{args.tma_sA}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _kloop_tma_copy_b_src(args: _PerKiterTmaArgs, *, k_offset: str) -> str:
    """Per-K-iter TMA copy source for B; ``""`` when B is not TMA-loaded.

    Callers pass a mask whenever the B TMA atom is multicast. The guarded
    clustered CtaGroup.ONE bridge diagnostic uses a self-only mask so each CTA
    duplicates local-B loads while satisfying CuTe's multicast-atom contract.
    """
    if not args.use_tma_b:
        return ""
    mcast = f", mcast_mask={args.tma_b_mcast_mask}" if args.use_tma_b_mcast_mask else ""
    return (
        f"    cute.copy({args.tma_atom_b}, "
        f"{args.tma_gB}[None, {k_offset}], "
        f"{args.tma_sB}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _tcgen05_two_cta_owner_predicate(
    exec_active: str,
    *,
    is_two_cta: bool,
    gate_exec_warp: bool,
    cluster_n: int = 1,
) -> str | None:
    """Owner-of-the-V-pair predicate for AB consumer release / MMA issuance.

    At ``is_two_cta=True`` V=2; the predicate must fire only on the V-leader
    of each V-pair so the AB consumer-release barrier and MMA issuance are
    *not* duplicated by the V-non-leader CTA. With ``cluster_n=1`` the only
    V-pair has V-leader rank 0 (cluster-leader), so the cheaper ``rank == 0``
    spelling keeps the validated cluster_n=1 predicate shape. With
    ``cluster_n=2`` there are two V-pairs (V-leaders {0, 2}) so the predicate
    must use ``rank % 2 == 0``; rank 2 must commit on its own V-pair's empty
    barrier or the cluster races (cycle-26 hang root cause; see cute_plan.md
    §6.12.3).
    """
    predicate_terms = []
    if gate_exec_warp:
        predicate_terms.append(exec_active)
    if is_two_cta:
        if cluster_n > 1:
            predicate_terms.append(_TCGEN05_V_LEADER_PREDICATE)
        else:
            predicate_terms.append(_TCGEN05_CLUSTER_LEADER_PREDICATE)
    if not predicate_terms:
        return None
    return " and ".join(predicate_terms)


def _tcgen05_emit_optional_gate(src: str, predicate: str | None, *, indent: str) -> str:
    if predicate is None:
        return textwrap.indent(src, indent)
    return f"{indent}if {predicate}:\n{textwrap.indent(src, indent + '    ')}"


def _build_kloop_pipeline_producer_if(
    args: _PerKiterTmaArgs, *, gate_tma_warp: bool = True
) -> ast.stmt:
    """Per-K-iter TMA producer ``if`` for the pipelined branch.

    The pipelined branch is only entered when both A and B are TMA-
    loaded (``tcgen05_use_tma_pipeline = use_tma_a and use_tma_b``), so
    both ``cute.copy`` emissions must be present; assert that invariant
    rather than silently dropping a side.
    """
    assert args.use_tma_a and args.use_tma_b, (
        "pipelined branch requires both A and B to be TMA-loaded"
    )
    assert not (args.static_full_tiles and args.is_two_cta), (
        "static-full fast path is only valid for one-CTA pipelined TMA loops"
    )
    k_offset = f"{args.tma_k_tile} + cutlass.Int32({args.ab_stage_count})"
    predicate_terms = []
    if not args.static_full_tiles:
        predicate_terms.append(args.tma_full_tile)
    if gate_tma_warp:
        predicate_terms.append(args.tma_warp)
    predicate_terms.append(args.tma_next_full_tile)
    copy_src = _kloop_tma_copy_a_src(args, k_offset=k_offset) + _kloop_tma_copy_b_src(
        args, k_offset=k_offset
    )
    # CtaGroup.TWO uses CTA-rank-specific TMA partitions, so both CTAs issue
    # these copies; PipelineTmaUmma gates the full-barrier tx setup internally.
    src = f"if {' and '.join(predicate_terms)}:\n"
    producer_advance_src = (
        emit_pipeline_advance(args.tma_producer_state, indent="    ")
        if not args.skip_producer_advance
        else ""
    )
    if not args.skip_producer_acquire:
        src += (
            f"    {args.tma_producer_try_token} = "
            f"{args.tma_pipeline}.producer_try_acquire({args.tma_producer_state})\n"
            f"    {args.tma_pipeline}.producer_acquire("
            f"{args.tma_producer_state}, {args.tma_producer_try_token})\n"
        )
    src += (
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + copy_src
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})\n"
        + producer_advance_src
    )
    return statement_from_string(src)


def _build_kloop_pipeline_consumer_if(
    args: _PerKiterTmaArgs,
    *,
    gate_exec_warp: bool = True,
    include_scalar_fallback: bool = True,
    use_existing_try_token: bool = False,
    sync_before_scalar_fallback: bool = False,
) -> ast.stmt:
    """Per-K-iter TMA consumer / scalar-fallback ``if`` for the pipelined branch."""
    if args.static_full_tiles:
        assert not args.is_two_cta, (
            "static-full fast path is only valid for one-CTA pipelined TMA loops"
        )
        assert gate_exec_warp, "static-full fast path requires an exec-warp gate"
        assert not include_scalar_fallback, (
            "static-full fast path has no scalar fallback branch"
        )
        assert not sync_before_scalar_fallback, (
            "static-full fast path has no scalar fallback presync"
        )
    if args.skip_consumer_wait:
        consumer_src = "pass"
    else:
        consumer_src = ""
        if not use_existing_try_token:
            consumer_src = (
                f"{args.tma_consumer_try_token} = "
                f"{args.tma_pipeline}.consumer_try_wait({args.tma_consumer_state})\n"
            )
        consumer_src += (
            f"{args.tma_pipeline}.consumer_wait("
            f"{args.tma_consumer_state}, {args.tma_consumer_try_token})"
        )
    full_tile_src = _tcgen05_emit_optional_gate(
        consumer_src,
        _tcgen05_two_cta_owner_predicate(
            args.exec_active,
            is_two_cta=args.is_two_cta,
            gate_exec_warp=gate_exec_warp,
            cluster_n=args.cluster_n,
        ),
        indent="" if args.static_full_tiles else "    ",
    )
    if args.static_full_tiles:
        return statement_from_string(full_tile_src)
    fallback_src = ""
    if include_scalar_fallback:
        scalar_load_a_src = textwrap.indent(ast.unparse(args.scalar_load_a), "    ")
        scalar_load_b_src = textwrap.indent(ast.unparse(args.scalar_load_b), "    ")
        leading_sync_src = (
            "    cute.arch.sync_threads()\n" if sync_before_scalar_fallback else ""
        )
        fallback_src = (
            "\nelse:\n"
            f"{leading_sync_src}"
            f"{scalar_load_a_src}\n"
            f"{scalar_load_b_src}\n"
            "    cute.arch.sync_threads()"
        )
    src = f"if {args.tma_full_tile}:\n{full_tile_src}{fallback_src}"
    return statement_from_string(src)


def _build_kloop_pipeline_consumer_prefetch_stmts(
    args: _PerKiterTmaArgs,
    *,
    gate_exec_warp: bool = True,
) -> list[ast.stmt]:
    """Peek the next AB full barrier after advancing the consumer state."""
    assert args.is_two_cta, "AB consumer prefetch is validated for CtaGroup.TWO"
    predicate = args.tma_next_consumer_tile
    owner_predicate = _tcgen05_two_cta_owner_predicate(
        args.exec_active,
        is_two_cta=args.is_two_cta,
        gate_exec_warp=gate_exec_warp,
        cluster_n=args.cluster_n,
    )
    if owner_predicate is not None:
        predicate = f"{predicate} and {owner_predicate}"
    return [
        statement_from_string(f"{args.tma_consumer_try_token} = cutlass.Boolean(1)"),
        statement_from_string(
            f"if {predicate}:\n"
            f"    {args.tma_consumer_try_token} = "
            f"{args.tma_pipeline}.consumer_try_wait({args.tma_consumer_state})"
        ),
    ]


def _build_kloop_pipeline_release_if(
    args: _PerKiterTmaArgs,
    *,
    gate_exec_warp: bool = True,
    include_scalar_fallback: bool = True,
) -> ast.stmt:
    """Per-K-iter consumer release ``if`` for the pipelined branch.

    Producer-state advance lives in the producer block (one per
    commit), so only the consumer-state advance is emitted here. In
    CtaGroup.TWO the empty-barrier release is leader-owned, matching the
    PipelineTmaUmma multicast-mask semantics, while both CTA exec warps still
    advance their local consumer state. Peer CTAs participate via the
    multicast mask; separate peer arrivals over-count the empty barrier.
    """
    if args.static_full_tiles:
        assert not args.is_two_cta, (
            "static-full fast path is only valid for one-CTA pipelined TMA loops"
        )
        assert gate_exec_warp, "static-full fast path requires an exec-warp gate"
        assert not include_scalar_fallback, (
            "static-full fast path has no scalar fallback branch"
        )
    release_src = f"{args.tma_pipeline}.consumer_release({args.tma_consumer_state})"
    release_gate = _tcgen05_two_cta_owner_predicate(
        args.exec_active,
        is_two_cta=args.is_two_cta,
        gate_exec_warp=gate_exec_warp,
        cluster_n=args.cluster_n,
    )
    advance_src = emit_pipeline_advance(args.tma_consumer_state)
    indent = "" if args.static_full_tiles else "    "
    if args.is_two_cta:
        # With gate_exec_warp=False the caller is already inside the
        # role-local exec loop, so every iteration can advance local state.
        advance_gate = args.exec_active if gate_exec_warp else None
        full_tile_src = (
            _tcgen05_emit_optional_gate(release_src, release_gate, indent=indent)
            + "\n"
            + _tcgen05_emit_optional_gate(advance_src, advance_gate, indent=indent)
        )
    else:
        full_tile_src = _tcgen05_emit_optional_gate(
            release_src + "\n" + advance_src, release_gate, indent=indent
        )
    if args.static_full_tiles:
        return statement_from_string(full_tile_src)
    fallback_src = (
        "\nelse:\n    cute.arch.sync_threads()" if include_scalar_fallback else ""
    )
    src = f"if {args.tma_full_tile}:\n{full_tile_src}{fallback_src}"
    return statement_from_string(src)


def _build_tcgen05_mma_accumulate_reset_stmt(
    exec_active: str,
    *,
    tiled_mma: str,
    gate_exec_warp: bool = True,
    is_two_cta: bool = False,
    cluster_n: int = 1,
) -> ast.stmt:
    reset_src = f"{tiled_mma}.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)"
    predicate = _tcgen05_two_cta_owner_predicate(
        exec_active,
        is_two_cta=is_two_cta,
        gate_exec_warp=gate_exec_warp,
        cluster_n=cluster_n,
    )
    if predicate is None:
        return statement_from_string(reset_src)
    return statement_from_string(f"if {predicate}:\n    {reset_src}")


def _build_tcgen05_mma_issue_stmt(
    *,
    exec_active: str,
    tiled_mma: str,
    acc_frag: str,
    tcgen05_frag_a: str,
    tcgen05_frag_b: str,
    mma_stage: str,
    gate_exec_warp: bool = True,
    is_two_cta: bool = False,
    cluster_n: int = 1,
) -> ast.stmt:
    issue_src = (
        f"for _tcgen05_kblk_idx in range(cute.size({tcgen05_frag_a}, mode=[2])):\n"
        f"    cute.gemm(\n"
        f"        {tiled_mma},\n"
        f"        {acc_frag},\n"
        f"        [{tcgen05_frag_a}[None, None, cutlass.Int32(_tcgen05_kblk_idx), {mma_stage}]],\n"
        f"        [{tcgen05_frag_b}[None, None, cutlass.Int32(_tcgen05_kblk_idx), {mma_stage}]],\n"
        f"        {acc_frag},\n"
        "    )\n"
        f"    {tiled_mma}.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)"
    )
    predicate = _tcgen05_two_cta_owner_predicate(
        exec_active,
        is_two_cta=is_two_cta,
        gate_exec_warp=gate_exec_warp,
        cluster_n=cluster_n,
    )
    if predicate is not None:
        issue_src = f"if {predicate}:\n{textwrap.indent(issue_src, '    ')}"
    return statement_from_string(issue_src)


def _build_kloop_non_pipeline_producer_if(
    args: _PerKiterTmaArgs, *, gate_tma_warp: bool = True
) -> ast.stmt:
    """Per-K-iter TMA producer ``if`` for the non-pipelined branch.

    Single AB stage alive at a time: no try-token, no stage-count
    offset on the cute.copy, and no ``advance`` here (the release block
    advances both producer and consumer state).
    """
    assert not args.static_full_tiles, (
        "static-full fast path is only valid for pipelined all-TMA K loops"
    )
    predicate_terms = [args.tma_full_tile]
    if gate_tma_warp:
        predicate_terms.append(args.tma_warp)
    copy_src = _kloop_tma_copy_a_src(
        args, k_offset=args.tma_k_tile
    ) + _kloop_tma_copy_b_src(args, k_offset=args.tma_k_tile)
    src = f"if {' and '.join(predicate_terms)}:\n"
    if not args.skip_producer_acquire:
        src += f"    {args.tma_pipeline}.producer_acquire({args.tma_producer_state})\n"
    src += (
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + copy_src
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})"
    )
    return statement_from_string(src)


def _build_kloop_non_pipeline_consumer_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter consumer / scalar-fallback ``if`` for the non-pipelined branch.

    Interleaves scalar fallback loads for any operand NOT TMA-loaded
    into the full-tile branch (e.g. A-TMA + B-scalar still loads B
    here on full tiles).
    """
    assert not args.static_full_tiles, (
        "static-full fast path is only valid for pipelined all-TMA K loops"
    )
    scalar_load_a_src = ast.unparse(args.scalar_load_a)
    scalar_load_b_src = ast.unparse(args.scalar_load_b)
    scalar_load_a_tma_src = scalar_load_a_src + "\n" if not args.use_tma_a else ""
    scalar_load_b_tma_src = scalar_load_b_src + "\n" if not args.use_tma_b else ""
    full_body = (
        f"{scalar_load_a_tma_src}"
        f"{scalar_load_b_tma_src}"
        f"if {args.exec_active}:\n"
        "    cute.arch.sync_warp()\n"
        + (
            "    pass\n"
            if args.skip_consumer_wait
            else (
                f"    {args.tma_pipeline}.consumer_wait("
                f"{args.tma_consumer_state}, {args.tma_consumer_try_token})\n"
            )
        )
        + "cute.arch.sync_threads()"
    )
    fallback_body = (
        f"{scalar_load_a_src}\n{scalar_load_b_src}\ncute.arch.sync_threads()"
    )
    src = (
        f"if {args.tma_full_tile}:\n"
        f"{textwrap.indent(full_body, '    ')}\n"
        "else:\n"
        f"{textwrap.indent(fallback_body, '    ')}"
    )
    return statement_from_string(src)


def _build_kloop_non_pipeline_release_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter consumer release ``if`` for the non-pipelined branch.

    CTA-wide ``sync_threads()`` runs first so every warp sees the
    consumer wait completed; single-stage means both producer and
    consumer state normally advance here. The producer advance is omitted
    only by the guarded invalid-output bridge diagnostic.
    """
    assert not args.static_full_tiles, (
        "static-full fast path is only valid for pipelined all-TMA K loops"
    )
    producer_advance_src = (
        emit_pipeline_advance(args.tma_producer_state, indent="") + "\n"
        if not args.skip_producer_advance
        else ""
    )
    full_body = (
        "cute.arch.sync_threads()\n"
        f"if {args.exec_active}:\n"
        "    cute.arch.sync_warp()\n"
        f"    {args.tma_pipeline}.consumer_release({args.tma_consumer_state})\n"
        + producer_advance_src
        + emit_pipeline_advance(args.tma_consumer_state, indent="")
    )
    src = (
        f"if {args.tma_full_tile}:\n"
        f"{textwrap.indent(full_body, '    ')}\n"
        "else:\n"
        "    cute.arch.sync_threads()"
    )
    return statement_from_string(src)


@dataclass(frozen=True)
class _InitialPrefetchTmaArgs:
    """Variable names threaded into the initial-prefetch TMA builder.

    The initial prefetch warms stages ``0..ab_stage_count-1`` of the AB
    pipeline at the start of each tile. Only valid on the tcgen05 TMA
    path, which requires both A and B to be TMA-loaded, so both
    ``cute.copy`` emissions are always present.

    All ``str`` fields name a Python identifier or expression in the
    generated code; every name is guaranteed bound at the call site.
    """

    tma_pipeline: str
    tma_producer_state: str
    tma_barrier_ptr: str
    tma_warp: str
    tma_atom_a: str
    tma_atom_b: str
    tma_gA: str
    tma_gB: str
    tma_sA: str
    tma_sB: str
    tma_a_mcast_mask: str
    tma_b_mcast_mask: str
    is_two_cta: bool
    use_tma_b_mcast_mask: bool
    skip_producer_acquire: bool
    skip_producer_advance: bool


def _initial_prefetch_copy_a_src(
    args: _InitialPrefetchTmaArgs, *, k_offset: str
) -> str:
    """Initial-prefetch TMA copy source for A.

    A only multicasts in 2-CTA mode (asymmetric vs. B, which can also
    multicast across cluster CTAs); matches the asymmetry pinned by
    ``test_mcast_mask_asymmetry_between_a_and_b`` for the per-K-iter
    builders.
    """
    mcast = f", mcast_mask={args.tma_a_mcast_mask}" if args.is_two_cta else ""
    return (
        f"    cute.copy({args.tma_atom_a}, "
        f"{args.tma_gA}[None, {k_offset}], "
        f"{args.tma_sA}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _initial_prefetch_copy_b_src(
    args: _InitialPrefetchTmaArgs, *, k_offset: str
) -> str:
    """Initial-prefetch TMA copy source for B.

    Callers pass a mask whenever the B TMA atom is multicast. The guarded
    clustered CtaGroup.ONE bridge diagnostic uses a self-only mask so each CTA
    duplicates local-B loads while satisfying CuTe's multicast-atom contract.
    """
    mcast = f", mcast_mask={args.tma_b_mcast_mask}" if args.use_tma_b_mcast_mask else ""
    return (
        f"    cute.copy({args.tma_atom_b}, "
        f"{args.tma_gB}[None, {k_offset}], "
        f"{args.tma_sB}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _build_initial_prefetch_if(
    args: _InitialPrefetchTmaArgs,
    *,
    full_tile_gates: list[str],
    k_offset: str,
    skip_producer_acquire: bool | None = None,
) -> ast.stmt:
    """Initial-prefetch ``if`` block for stage ``k_offset``.

    The predicate is ``<full_tile_gates joined with ' and '> and
    {args.tma_warp}``: stage-0 callers pass
    ``[tma_initial_full_tile]``; stage-(N-1) callers (only when
    ``ab_stage_count > 1``) extend with ``tma_initial_next_full_tile``.
    The body performs optional ``producer_acquire``, then
    ``get_barrier / copy A / copy B / producer_commit`` and optional
    producer-state ``advance``. The optional edges are omitted only by
    guarded invalid-output bridge diagnostics. Caller passes a literal
    ``cutlass.Int32(stage_idx)`` for ``k_offset``.
    """
    predicate = " and ".join([*full_tile_gates, args.tma_warp])
    if skip_producer_acquire is None:
        skip_producer_acquire = args.skip_producer_acquire
    producer_advance_src = (
        emit_pipeline_advance(args.tma_producer_state, indent="    ")
        if not args.skip_producer_advance
        else ""
    )
    copy_src = _initial_prefetch_copy_a_src(
        args, k_offset=k_offset
    ) + _initial_prefetch_copy_b_src(args, k_offset=k_offset)
    src = f"if {predicate}:\n"
    if not skip_producer_acquire:
        src += f"    {args.tma_pipeline}.producer_acquire({args.tma_producer_state})\n"
    src += (
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + copy_src
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})\n"
        + producer_advance_src
    )
    return statement_from_string(src)


def _is_persistent_pid_config(config: Mapping[str, object]) -> bool:
    pid_type = config.get("pid_type", "flat")
    return isinstance(pid_type, str) and pid_type.startswith("persistent")


def _clone_k_loop_with_body(
    device_loop: DeviceLoopState,
    body: list[ast.stmt],
    *,
    iter_expr: ast.expr | None = None,
) -> ast.For:
    """Clone the active K-loop header with a replacement body."""
    # Do not reuse the original target / iter AST nodes: the original loop
    # remains the shared consumer loop, while this clone becomes the
    # TMA-load producer loop. Round-tripping the simple generated
    # ``for offset_k in range(...)`` header avoids shared mutable AST nodes.
    parsed_loop = ast.parse(
        f"for {ast.unparse(device_loop.for_node.target)} in ():\n    pass"
    ).body[0]
    assert isinstance(parsed_loop, ast.For)
    if iter_expr is None:
        iter_expr = cast(
            "ast.expr", expr_from_string(ast.unparse(device_loop.for_node.iter))
        )
    return ast.copy_location(
        ast.For(
            target=parsed_loop.target,
            iter=iter_expr,
            body=body,
            orelse=[],
            type_comment=device_loop.for_node.type_comment,
        ),
        device_loop.for_node,
    )


def _tcgen05_k_loop_nounroll_iter_expr(device_loop: DeviceLoopState) -> ast.expr:
    """Preserve the active K-loop bounds while adding CuTe no-unroll metadata."""
    iter_expr = cast(
        "ast.expr", expr_from_string(ast.unparse(device_loop.for_node.iter))
    )
    assert isinstance(iter_expr, ast.Call)
    assert isinstance(iter_expr.func, ast.Name)
    assert iter_expr.func.id == "range"
    assert not any(
        keyword.arg in ("unroll", "unroll_full") for keyword in iter_expr.keywords
    )
    iter_expr.func = cast("ast.expr", expr_from_string("cutlass.range"))
    iter_expr.keywords.append(ast.keyword(arg="unroll", value=ast.Constant(value=1)))
    return iter_expr


def _wrap_stmt_in_if(stmt: ast.stmt, predicate_src: str) -> ast.If:
    return ast.copy_location(
        ast.If(
            test=cast("ast.expr", expr_from_string(predicate_src)),
            body=[stmt],
            orelse=[],
        ),
        stmt,
    )


def _trace_mma_to_single_store_value_and_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
    *,
    extra_trace_through: frozenset[object] = frozenset(),
) -> tuple[Node, torch.dtype] | None:
    """Forward-trace ``mma_node`` to exactly one consuming store.

    Walks ``_phi`` / ``_new_var`` / ``convert_element_type`` /
    ``operator.getitem`` chains and crosses subgraph boundaries via the
    inner ``output`` node and the matching outer-graph
    ``operator.getitem(_for_loop_node, idx)``. Returns ``None`` when the
    trace cannot pin exactly one store target — multiple stores even with
    the same dtype, an op the trace cannot safely follow, no reachable
    store, or ``mma_node.graph`` not present in ``graphs``. The caller
    must treat ``None`` as "fall back; the cross-site assertion catches
    real mismatches".

    ``graphs`` must be the live codegen graph list the
    ``GraphInterpreter`` is walking so ``mma_node.graph`` matches a
    ``GraphInfo.graph`` by identity — that's the disambiguator across
    structurally-identical subgraphs (e.g., a kernel with two distinct
    matmuls would otherwise collide on ``_graph_signature``).

    ``extra_trace_through`` lets the caller broaden the set of ops the
    walker will follow past — e.g. ``aten.relu.default`` for the
    relu-store gate. Default behaviour is unchanged.
    """
    import operator

    from ...language import _tracing_ops
    from ...language import memory_ops

    graph_id_of: dict[torch.fx.Graph, int] = {}
    for_loop_calls_by_graph_id: dict[int, list[Node]] = {}
    for graph_info in graphs:
        graph_id_of[graph_info.graph] = graph_info.graph_id
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if not _tracing_ops.is_for_loop_target(node.target):
                continue
            graph_id_arg = node.args[0] if node.args else None
            if not isinstance(graph_id_arg, int):
                continue
            for_loop_calls_by_graph_id.setdefault(graph_id_arg, []).append(node)

    if mma_node.graph not in graph_id_of:
        return None

    store_target: tuple[Node, torch.dtype] | None = None
    store_count = 0
    visited: set[Node] = set()
    stack: list[Node] = [mma_node]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for user in cur.users:
            if user.op == "output":
                graph_id = graph_id_of.get(cur.graph)
                if graph_id is None:
                    return None
                output_args = user.args[0] if user.args else None
                if not isinstance(output_args, (list, tuple)):
                    return None
                out_indices = [i for i, arg in enumerate(output_args) if arg is cur]
                if not out_indices:
                    return None
                for outer_call in for_loop_calls_by_graph_id.get(graph_id, []):
                    for outer_user in outer_call.users:
                        if (
                            outer_user.op == "call_function"
                            and outer_user.target is operator.getitem
                            and len(outer_user.args) >= 2
                            and outer_user.args[1] in out_indices
                            and outer_user not in visited
                        ):
                            stack.append(outer_user)
                continue
            if user.op != "call_function":
                continue
            target = user.target
            if target is memory_ops.store:
                tensor_node = user.args[0] if user.args else None
                if not isinstance(tensor_node, Node):
                    return None
                fake = tensor_node.meta.get("val")
                if not isinstance(fake, torch.Tensor):
                    return None
                store_count += 1
                if store_count > 1:
                    return None
                store_target = (cur, fake.dtype)
                continue
            if store_count > 0:
                # A value that already reached its single store cannot also feed
                # additional pointwise/use chains under this identity-store trace.
                return None
            if (
                target is _tracing_ops._phi
                or target is _tracing_ops._new_var
                or target is operator.getitem
                or target in _TRACE_THROUGH_TARGETS
                or target in extra_trace_through
            ):
                stack.append(user)
                continue
            return None

    return store_target if store_count == 1 else None


def _trace_mma_to_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
) -> torch.dtype | None:
    """Forward-trace ``mma_node`` to a unique reachable store dtype.

    Unlike the pure CLC identity-store gate, this general dtype-inference
    helper allows same-dtype fan-out because normal tcgen05 epilogue planning
    only needs a single target dtype.
    """
    import operator

    from ...language import _tracing_ops
    from ...language import memory_ops

    graph_id_of: dict[torch.fx.Graph, int] = {}
    for_loop_calls_by_graph_id: dict[int, list[Node]] = {}
    for graph_info in graphs:
        graph_id_of[graph_info.graph] = graph_info.graph_id
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if not _tracing_ops.is_for_loop_target(node.target):
                continue
            graph_id_arg = node.args[0] if node.args else None
            if not isinstance(graph_id_arg, int):
                continue
            for_loop_calls_by_graph_id.setdefault(graph_id_arg, []).append(node)

    if mma_node.graph not in graph_id_of:
        return None

    discovered: set[torch.dtype] = set()
    visited: set[Node] = set()
    stack: list[Node] = [mma_node]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for user in cur.users:
            if user.op == "output":
                graph_id = graph_id_of.get(cur.graph)
                if graph_id is None:
                    return None
                output_args = user.args[0] if user.args else None
                if not isinstance(output_args, (list, tuple)):
                    return None
                out_indices = [i for i, arg in enumerate(output_args) if arg is cur]
                if not out_indices:
                    return None
                for outer_call in for_loop_calls_by_graph_id.get(graph_id, []):
                    for outer_user in outer_call.users:
                        if (
                            outer_user.op == "call_function"
                            and outer_user.target is operator.getitem
                            and len(outer_user.args) >= 2
                            and outer_user.args[1] in out_indices
                            and outer_user not in visited
                        ):
                            stack.append(outer_user)
                continue
            if user.op != "call_function":
                continue
            target = user.target
            if target is memory_ops.store:
                tensor_node = user.args[0] if user.args else None
                if not isinstance(tensor_node, Node):
                    return None
                fake = tensor_node.meta.get("val")
                if not isinstance(fake, torch.Tensor):
                    return None
                discovered.add(fake.dtype)
                if len(discovered) > 1:
                    return None
                continue
            if (
                target is _tracing_ops._phi
                or target is _tracing_ops._new_var
                or target is operator.getitem
                or target in _TRACE_THROUGH_TARGETS
            ):
                stack.append(user)
                continue
            return None

    if len(discovered) == 1:
        return next(iter(discovered))
    return None


def _is_rank1_bias_load_of_dtype(node: Node, expected_dtype: torch.dtype) -> bool:
    """``helion.language.load`` of a rank-1 ``expected_dtype`` GMEM tensor.

    Shared by the T2 (``acc + bias[n]``), T6 (``relu(acc + bias[n])``),
    and T10 (fp16/bf16 2048³ bias) direct-entry codegen walkers. The
    bias dtype is pinned to the caller-supplied operand dtype so a
    fp16 bias against a bf16-operand matmul (or vice versa) is rejected
    at the codegen walker boundary instead of only at the runtime
    validator.
    """
    from ...language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.load:
        return False
    host_arg = node.args[0] if node.args else None
    if not isinstance(host_arg, Node):
        return False
    host_val = host_arg.meta.get("val")
    if not isinstance(host_val, torch.Tensor):
        return False
    return host_val.ndim == 1 and host_val.dtype == expected_dtype


def _trace_mma_to_single_cast_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
    *,
    extra_trace_through: frozenset[object] = frozenset(),
    validate_cast_input: Callable[[Node, set[Node], dict], bool],
) -> torch.dtype | None:
    """Common skeleton for the direct-entry envelope walkers.

    Traces ``mma_node`` to its single store via
    ``_trace_mma_to_single_store_value_and_dtype``, requires a
    ``convert_element_type.default`` cast as the store value, then defers
    to ``validate_cast_input`` to verify the epilogue chain reaches the
    MMA carrier. The closing dtype-arg check pins the cast's target dtype
    to the recorded store dtype.
    """
    from .cute_fx_walk import build_inner_outputs_index_from_graphs

    traced = _trace_mma_to_single_store_value_and_dtype(
        mma_node, graphs, extra_trace_through=extra_trace_through
    )
    if traced is None:
        return None
    store_value, output_dtype = traced
    if (
        store_value.op != "call_function"
        or store_value.target is not torch.ops.prims.convert_element_type.default
        or store_value.kwargs
    ):
        return None
    cast_input = store_value.args[0] if store_value.args else None
    if not isinstance(cast_input, Node):
        return None
    if not validate_cast_input(
        cast_input, {mma_node}, build_inner_outputs_index_from_graphs(graphs)
    ):
        return None
    dtype_arg = store_value.args[1] if len(store_value.args) > 1 else None
    if dtype_arg != output_dtype:
        return None
    return output_dtype


def _trace_mma_to_single_identity_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
) -> torch.dtype | None:
    """Return the store dtype only for exactly one identity cast store."""
    from .cute_fx_walk import walk_carrier_to_tcgen05_matmul

    def _validate(cast_input: Node, mma_set: set[Node], idx: dict) -> bool:
        return walk_carrier_to_tcgen05_matmul(cast_input, mma_set, idx) is not None

    return _trace_mma_to_single_cast_store_dtype(
        mma_node, graphs, validate_cast_input=_validate
    )


def _trace_mma_to_single_relu_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
) -> torch.dtype | None:
    """Return the store dtype only for exactly one ``relu`` + cast store.

    The chain accepted here is ``mma -> relu -> convert_element_type ->
    store``: exactly one ``aten.relu.default`` between the MMA carrier and
    the cast that feeds the store. Mirrors
    ``_trace_mma_to_single_identity_store_dtype`` for the relu epilogue so
    the TVM-FFI direct entry can admit Target 4 without broadening the
    general-purpose identity-store gate.
    """
    from .cute_fx_walk import walk_carrier_to_tcgen05_matmul

    def _validate(cast_input: Node, mma_set: set[Node], idx: dict) -> bool:
        if (
            cast_input.op != "call_function"
            or cast_input.target is not torch.ops.aten.relu.default
            or cast_input.kwargs
            or len(cast_input.args) != 1
        ):
            return False
        relu_input = cast_input.args[0]
        if not isinstance(relu_input, Node):
            return False
        return walk_carrier_to_tcgen05_matmul(relu_input, mma_set, idx) is not None

    return _trace_mma_to_single_cast_store_dtype(
        mma_node,
        graphs,
        extra_trace_through=frozenset({torch.ops.aten.relu.default}),
        validate_cast_input=_validate,
    )


def _validate_bias_add_chain(
    add_node: Node,
    mma_set: set[Node],
    inner_outputs_index: dict,
    *,
    expected_bias_dtype: torch.dtype,
) -> bool:
    """``aten.add.Tensor(carrier, rank-1 ``expected_bias_dtype`` bias_load)`` (or commuted)."""
    from .cute_fx_walk import walk_carrier_to_tcgen05_matmul

    if (
        add_node.op != "call_function"
        or add_node.target is not torch.ops.aten.add.Tensor
        or add_node.kwargs
        or len(add_node.args) != 2
    ):
        return False
    add_lhs, add_rhs = add_node.args
    if not isinstance(add_lhs, Node) or not isinstance(add_rhs, Node):
        return False
    # Either side of the add can be the carrier (commutative); the other
    # must be a rank-1 bias load whose dtype matches the operand dtype.
    carrier_first = walk_carrier_to_tcgen05_matmul(
        add_lhs, mma_set, inner_outputs_index
    ) is not None and _is_rank1_bias_load_of_dtype(add_rhs, expected_bias_dtype)
    carrier_second = walk_carrier_to_tcgen05_matmul(
        add_rhs, mma_set, inner_outputs_index
    ) is not None and _is_rank1_bias_load_of_dtype(add_lhs, expected_bias_dtype)
    return carrier_first or carrier_second


def _trace_mma_to_single_bias_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
    *,
    expected_bias_dtype: torch.dtype,
) -> torch.dtype | None:
    """Return the store dtype only for exactly one ``acc + bias[n]`` + cast store.

    The chain accepted here is ``mma -> aten.add.Tensor(carrier,
    bias_load) -> convert_element_type -> store``: exactly one
    ``aten.add.Tensor`` between the MMA carrier and the cast that feeds
    the store, where one of the add's operands is the MMA carrier and the
    other is a rank-1 ``helion.language.load`` against a GMEM tensor with
    shape ``(N,)`` and dtype ``expected_bias_dtype``.

    Mutually exclusive with the identity and relu walkers: identity
    rejects any binary op in the chain, and relu requires a single-arg
    ``aten.relu.default`` rather than the two-arg add. A T6
    (``acc + bias[n] -> relu -> convert -> store``) chain is rejected
    because the cast feeds off a relu rather than the add.
    """

    def _validate(cast_input: Node, mma_set: set[Node], idx: dict) -> bool:
        return _validate_bias_add_chain(
            cast_input, mma_set, idx, expected_bias_dtype=expected_bias_dtype
        )

    return _trace_mma_to_single_cast_store_dtype(
        mma_node,
        graphs,
        extra_trace_through=frozenset({torch.ops.aten.add.Tensor}),
        validate_cast_input=_validate,
    )


def _trace_mma_to_single_bias_relu_store_dtype(
    mma_node: Node,
    graphs: list[GraphInfo],
    *,
    expected_bias_dtype: torch.dtype,
) -> torch.dtype | None:
    """Return the store dtype only for one ``relu(acc + bias[n])`` + cast store.

    Same as :func:`_trace_mma_to_single_bias_store_dtype` but with an
    intervening ``aten.relu.default`` between the add and the cast.
    """

    def _validate(cast_input: Node, mma_set: set[Node], idx: dict) -> bool:
        # T6 requires a relu directly feeding the cast (whereas T2 has the
        # bias add feed the cast directly without an intervening relu).
        if (
            cast_input.op != "call_function"
            or cast_input.target is not torch.ops.aten.relu.default
            or cast_input.kwargs
            or len(cast_input.args) != 1
        ):
            return False
        relu_input = cast_input.args[0]
        if not isinstance(relu_input, Node):
            return False
        return _validate_bias_add_chain(
            relu_input, mma_set, idx, expected_bias_dtype=expected_bias_dtype
        )

    return _trace_mma_to_single_cast_store_dtype(
        mma_node,
        graphs,
        extra_trace_through=frozenset(
            {torch.ops.aten.add.Tensor, torch.ops.aten.relu.default}
        ),
        validate_cast_input=_validate,
    )


def _emit_mma_pipeline(
    cg: GenerateAST,
    lhs_node: Node,
    rhs_node: Node,
    acc_expr: ast.AST | None = None,
    fx_node: Node | None = None,
) -> ast.AST | None:
    """Core MMA codegen shared by both aten and hl.dot paths.

    Emits outer_prefix (MMA setup + acc init), loop body (smem staging +
    gemm), and outer_suffix (fragment → per-thread scalar via smem).

    Returns a per-thread scalar expression, or None on failure.
    """
    from ..compile_environment import CompileEnvironment

    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return None
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
    if lhs_fake.ndim != 2 or rhs_fake.ndim != 2:
        return None

    # Universal-MMA / tcgen05 MMA kernels rely on runtime tensor layouts
    # for SMEM-load guards and TMA descriptors; baking literal shapes
    # silently miscompiles those paths.  Mirror the flag set in
    # ``_emit_cute_matmul`` so the host-side launcher disables the bake.
    cg.cute_uses_matmul = True

    df = cg.device_function
    lhs_arg = df.tensor_arg(lhs_fake)
    rhs_arg = df.tensor_arg(rhs_fake)
    lhs_arg_name = lhs_arg.name
    rhs_arg_name = rhs_arg.name

    input_dtype = lhs_fake.dtype
    _dtype_map = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
    }
    input_dtype_str = _dtype_map[input_dtype]
    acc_dtype_str = "cutlass.Float32"
    # The kernel-side `tcgen05_epi_tile` (built in
    # `_make_tcgen05_layout_plan_setup`) and the store-side
    # `tcgen05_store_epi_tile` (built in `_codegen_cute_store_tcgen05_tile`)
    # must agree on the `elem_ty_d` / `elem_ty_c` passed to
    # `compute_epilogue_tile_shape` — `tile_n` differs between the
    # with-source bf16/fp16 `n_perf=64` branch and the fp32-with-source
    # `n_perf=32` branch, and a mismatch silently corrupts SMEM staging.
    # Forward-trace the matmul fx_node to its consuming store target so
    # both sides see the same dtype. When the trace fails (no fx_node,
    # multi-store fan-out, opaque op) or returns a dtype outside the
    # known matmul output family, we fall back to the input dtype here;
    # the store-side equality check on the registered
    # `CuteTcgen05StoreValue` is the loud-failure backstop and
    # surfaces `BackendUnsupported` if the runtime D-tensor dtype
    # disagrees with the matmul plan's assumption.
    epi_elem_dtype: torch.dtype | None = None
    if fx_node is not None:
        traced = _trace_mma_to_store_dtype(fx_node, cg.codegen_graphs)
        if traced in _dtype_map:
            epi_elem_dtype = traced
    epi_elem_dtype_str = (
        _dtype_map[epi_elem_dtype] if epi_elem_dtype is not None else input_dtype_str
    )
    tcgen05_use_tma = (
        input_dtype in (torch.float16, torch.bfloat16)
        and lhs_fake.is_contiguous()
        and rhs_fake.is_contiguous()
    )
    tcgen05_use_tma_a = tcgen05_use_tma
    tcgen05_use_tma_b = tcgen05_use_tma
    tcgen05_use_tma = tcgen05_use_tma_a or tcgen05_use_tma_b
    tcgen05_use_tma_pipeline = tcgen05_use_tma_a and tcgen05_use_tma_b
    tcgen05_requested_pure_matmul_role_lifecycle = is_pure_matmul_role_lifecycle_config(
        df.config
    )

    k_total_size = int(lhs_fake.shape[1])

    env = CompileEnvironment.current()

    k_loop_info = _get_mma_k_loop_info(cg, env, lhs_fake, rhs_fake, fx_node=fx_node)
    if k_loop_info is None:
        return None
    device_loop, _, k_offset_var, bk = k_loop_info
    if _has_non_root_lane_loops(cg):
        return None
    k_loop_begin_expr = _device_loop_begin_expr(device_loop)

    # Get M, N offsets and block sizes from grid state
    m_offset_var: str | None = None
    n_offset_var: str | None = None
    m_block_id: int | None = None
    n_block_id: int | None = None
    bm: int | None = None
    bn: int | None = None
    grid_state = cg.current_grid_state
    if grid_state is not None:
        if len(grid_state.block_ids) == 2:
            m_block_id, n_block_id = grid_state.block_ids
            m_offset_var = grid_state.strategy.offset_var(m_block_id)
            n_offset_var = grid_state.strategy.offset_var(n_block_id)
            m_bs = df.resolved_block_size(m_block_id)
            n_bs = df.resolved_block_size(n_block_id)
            bm = int(m_bs) if isinstance(m_bs, int) else None
            bn = int(n_bs) if isinstance(n_bs, int) else None
        else:
            for bid in grid_state.block_ids:
                offset = grid_state.strategy.offset_var(bid)
                bs_info = env.block_sizes[bid]
                size = bs_info.size
                bs = bs_info.from_config(df.config)
                if isinstance(size, (int, torch.SymInt)):
                    if m_offset_var is None and env.known_equal(
                        size, lhs_fake.shape[0]
                    ):
                        m_offset_var = offset
                        m_block_id = bid
                        bm = int(bs) if isinstance(bs, int) else None
                    elif n_offset_var is None and env.known_equal(
                        size, rhs_fake.shape[1]
                    ):
                        n_offset_var = offset
                        n_block_id = bid
                        bn = int(bs) if isinstance(bs, int) else None

    if (
        bm is None
        or bn is None
        or m_offset_var is None
        or n_offset_var is None
        or m_block_id is None
        or n_block_id is None
    ):
        return None
    # tcgen05 epilogues are emitted by `_codegen_cute_store_tcgen05_tile` in
    # `helion/language/memory_ops.py`. Static-full flat kernels and validated
    # role-local persistent kernels use the SMEM-staged TMA-store epilogue;
    # partial/unsupported fallbacks keep the direct TMEM->register->GMEM SIMT
    # path.

    m_index_var = cg.index_var(m_block_id)
    n_index_var = cg.index_var(n_block_id)
    # Use thread_idx directly for local indices within the tile.
    # indices_0 - offset_0 SHOULD equal thread_idx[0], but the CuTe DSL
    # compiler may not simplify the subtraction, leading to illegal memory
    # accesses when partition shapes depend on dynamic values.
    assert grid_state is not None
    m_local = _local_mma_coord_expr(cg, m_block_id)
    n_local = _local_mma_coord_expr(cg, n_block_id)
    # ``m_physical`` / ``n_physical`` strip the lane-var offset so SMEM-load
    # guards select the same hardware thread across every iteration of an
    # outer ``for lane_<n> in range(elements_per_thread):`` loop. The
    # universal-MMA SMEM load below would otherwise gate on ``n_local == 0``
    # which is only true on the lane=0 iteration when ``n`` has a lane var,
    # leaving ``sA`` stale from the previous lane iteration and producing
    # wrong-output for the entire post-lane-0 accumulator.
    m_physical = _physical_mma_coord_expr(cg, m_block_id)
    n_physical = _physical_mma_coord_expr(cg, n_block_id)
    m_global = f"cutlass.Int32({m_index_var})"
    n_global = f"cutlass.Int32({n_index_var})"
    m_size = int(lhs_fake.shape[0])
    n_size = int(rhs_fake.shape[1])

    tcgen05_cluster_m = _tcgen05_cluster_m(df.config)
    tcgen05_large_bn_proof = _tcgen05_large_bn_proof_enabled(df.config)
    if tcgen05_large_bn_proof and (
        not _tcgen05_large_bn_proof_shape(
            bm=bm,
            bn=bn,
            bk=bk,
            tcgen05_cluster_m=tcgen05_cluster_m,
        )
        or (m_size, n_size, k_total_size) != TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE
        or cast("_ConfigLike", df.config).get("pid_type", "flat")
        != TCGEN05_LARGE_BN_PROOF_PID_TYPE
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY}=True requires the guarded "
            "G4 proof envelope "
            f"M={TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE[0]},"
            f"N={TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE[1]},"
            f"K={TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE[2]},"
            f"bm={TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES[0]},"
            f"bn={TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES[1]},"
            f"bk={TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES[2]},"
            f"tcgen05_cluster_m={TCGEN05_LARGE_BN_PROOF_CLUSTER_M},"
            f"pid_type={TCGEN05_LARGE_BN_PROOF_PID_TYPE!r}",
        )

    mma_impl = _choose_mma_impl(input_dtype, bm=bm, bn=bn, bk=bk, config=df.config)
    zero_acc_expr = acc_expr is not None and _is_zero_acc_expr(acc_expr)
    if (
        not zero_acc_expr
        and acc_expr is not None
        and fx_node is not None
        and fx_node.target is torch.ops.aten.addmm.default
    ):
        acc_node = fx_node.args[0] if fx_node.args else None
        if isinstance(acc_node, Node) and _is_zero_init_acc_node(acc_node):
            zero_acc_expr = True
    if acc_expr is not None and mma_impl != "universal" and not zero_acc_expr:
        mma_impl = "universal"
    if mma_impl != "universal" and zero_acc_expr:
        acc_expr = None
    tcgen05_requested_flat_role_coordinates = bool(
        df.config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False)
    )
    tcgen05_requested_direct_entry_plan = bool(
        df.config.get(TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY, False)
    )
    tcgen05_requested_pure_clc_scheduler_object = bool(
        df.config.get(TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY, False)
    )
    tcgen05_requested_pure_dynamic_scheduler_object = bool(
        df.config.get(TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY, False)
    )
    if tcgen05_requested_flat_role_coordinates and mma_impl != "tcgen05":
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY}=True requires "
            "tcgen05 MMA codegen",
        )
    if tcgen05_requested_pure_clc_scheduler_object and mma_impl != "tcgen05":
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY}=True requires "
            "tcgen05 MMA codegen",
        )
    # Cycle-16 H3 Option B (staged): the pure dynamic-persistent scheduler
    # object lives at the same dispatch layer as pure-CLC. We surface the
    # config key + autotune knob now so cycle 17 can land productive
    # codegen without re-doing the search / validator plumbing; in this
    # cycle every selection of the knob still errors out with
    # BackendUnsupported (the error is consistent whether the knob is
    # selected by autotune or by an explicit user Config). The mma_impl
    # gate matches the CLC predecessor.
    if tcgen05_requested_pure_dynamic_scheduler_object and mma_impl != "tcgen05":
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY}=True "
            "requires tcgen05 MMA codegen",
        )
    if tcgen05_requested_direct_entry_plan and mma_impl != "tcgen05":
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY}=True requires tcgen05 MMA codegen",
        )
    tcgen05_pid_is_persistent = _is_persistent_pid_config(df.config)
    tcgen05_requested_two_cta = _tcgen05_use_2cta_instrs(
        bm=bm, cluster_m=tcgen05_cluster_m
    )
    tcgen05_cluster_n_requested = _tcgen05_cluster_n(df.config)
    tcgen05_static_output_tiles = m_size % bm == 0 and n_size % bn == 0
    tcgen05_static_full_tiles = tcgen05_static_output_tiles and k_total_size % bk == 0
    tcgen05_has_k_tail = k_total_size > bk and k_total_size % bk != 0
    tcgen05_k_tail_only = tcgen05_static_output_tiles and tcgen05_has_k_tail
    tcgen05_double_edge_output = m_size % bm != 0 and n_size % bn != 0
    tcgen05_double_edge_tma = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 2
        and tcgen05_cluster_n_requested == 1
        and tcgen05_requested_two_cta
        and tcgen05_double_edge_output
        and (k_total_size % bk == 0 or tcgen05_has_k_tail)
    )
    if (
        mma_impl == "tcgen05"
        and tcgen05_double_edge_output
        and not tcgen05_double_edge_tma
        and (tcgen05_pid_is_persistent or tcgen05_cluster_m != 1)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 SIMT edge epilogue double-edge output tiles are currently "
            "validated only for flat tcgen05_cluster_m=1 kernels or "
            "role-local CtaGroup.TWO kernels where K is divisible by block_k "
            "or K > block_k with a tail; K <= block_k with a partial K tile "
            "is still unsupported. "
            "Choose a block_m or block_n that divides the static output extent, "
            "choose a supported block_k for the static K extent, or use a "
            "non-tcgen05 fallback for this persistent/clustered config.",
        )
    tcgen05_mixed_tma_scalar_fallback = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and not tcgen05_static_full_tiles
        and not tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 1
    )
    tcgen05_edge_scalar_fallback_needs_inter_smem_a = (
        tcgen05_mixed_tma_scalar_fallback
        and bk >= 128
        and (m_size % bm != 0 or n_size % bn != 0)
    )
    tcgen05_sync_before_scalar_fallback = (
        tcgen05_mixed_tma_scalar_fallback and k_total_size % bk != 0
    )
    tcgen05_preserve_tma_for_two_cta_k_tail = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 2
        and tcgen05_cluster_n_requested == 1
        and tcgen05_requested_two_cta
        and tcgen05_k_tail_only
    )
    tcgen05_m_edge_only = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 2
        and tcgen05_cluster_n_requested == 1
        and tcgen05_requested_two_cta
        and m_size % bm != 0
        and n_size % bn == 0
        and k_total_size % bk == 0
    )
    tcgen05_n_edge_only = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 2
        and tcgen05_cluster_n_requested == 1
        and tcgen05_requested_two_cta
        and m_size % bm == 0
        and n_size % bn != 0
        and k_total_size % bk == 0
    )
    if (
        mma_impl == "tcgen05"
        and not tcgen05_static_full_tiles
        and not tcgen05_mixed_tma_scalar_fallback
        and not tcgen05_preserve_tma_for_two_cta_k_tail
        and not tcgen05_m_edge_only
        and not tcgen05_n_edge_only
        and not tcgen05_double_edge_tma
    ):
        # Mixed TMA full K tiles + scalar fallback tails are currently
        # validated only for flat one-CTA kernels. Persistent CtaGroup.TWO
        # K-tail-only kernels keep TMA enabled for
        # tcgen05_role_local_k_tail_tma. Output-edge kernels keep TMA
        # enabled for role-local AB production plus the predicated SIMT
        # epilogue. Other persistent or clustered partial-output kernels keep
        # the shared scalar path.
        tcgen05_use_tma_a = False
        tcgen05_use_tma_b = False
        tcgen05_use_tma = False
        tcgen05_use_tma_pipeline = False
    # cluster_m == 2 has two valid shapes:
    # - bm=128, CtaGroup.ONE, where each clustered CTA owns a different
    #   128-row output tile. This legacy clustered shape remains guarded.
    # - bm=256, CtaGroup.TWO, where two CTAs cooperate on one 256-row output
    #   tile. That shape is valid even when the logical M tile count is 1, so
    #   do not apply the old "at least cluster_m M tiles" demotion to it.
    #
    # Still demote unsupported or unguarded shapes before emitting cluster code.
    # The autotune search is separately narrowed to the validated subset; this
    # catches explicit user configs that bypass autotune. For non-tcgen05 MMA
    # implementations the tcgen05 cluster knob is irrelevant, so normalize it
    # away instead of rejecting an otherwise valid fallback config.
    if mma_impl != "tcgen05":
        tcgen05_cluster_m = 1
    elif tcgen05_cluster_m > 1:
        if not tcgen05_pid_is_persistent:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_cluster_m > 1 is currently supported only for "
                "guarded persistent tcgen05 codegen while G3 validates "
                "CtaGroup.TWO runtime ownership. Use tcgen05_cluster_m=1 "
                "or a persistent pid_type.",
            )
        if bm < 128 or (
            not tcgen05_requested_two_cta and m_size // bm < tcgen05_cluster_m
        ):
            tcgen05_cluster_m = 1
    assert tcgen05_cluster_m == 1 or bm >= 128
    tcgen05_is_two_cta = tcgen05_requested_two_cta and tcgen05_cluster_m > 1
    # ``tcgen05_cluster_n`` is the multicast factor along the cluster's N
    # axis. cluster_n=2 builds the canonical Quack-best 4-CTA cluster
    # ``(cluster_m=2, cluster_n=2, 1)`` and only runs under
    # ``use_2cta=True`` (V=2). At V=2 cluster_n=2 the V-leader gate
    # (cute_plan.md §6.12.7) and ``mcast_size`` arrive count
    # (``num_mcast_ctas_a + num_mcast_ctas_b - 1 = 2 + 1 - 1 = 2``) replace
    # the cluster_n=1 cluster-leader / arrive_count=1 spelling. Outside
    # this validated pairing demote to cluster_n=1 instead of throwing —
    # explicit user configs hit the BackendUnsupported gate, and autotune
    # stays narrowed to the validated subset.
    if tcgen05_cluster_n_requested > 1 and not (
        mma_impl == "tcgen05" and tcgen05_is_two_cta and tcgen05_cluster_m == 2
    ):
        if mma_impl == "tcgen05" and tcgen05_cluster_n_requested == 2:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_cluster_n=2 requires tcgen05_cluster_m=2 with "
                "use_2cta=True (bm=256). See cute_plan.md §6.12 for the "
                "validated 4-CTA cluster envelope.",
            )
        tcgen05_cluster_n = 1
    else:
        tcgen05_cluster_n = tcgen05_cluster_n_requested
    tcgen05_role_local_k_tail_tma = (
        tcgen05_preserve_tma_for_two_cta_k_tail
        and tcgen05_is_two_cta
        and tcgen05_cluster_n == 1
    )
    # Mirror the K-tail role-local guard so later cluster demotion or
    # cluster_n enablement cannot silently admit unvalidated edge ownership.
    tcgen05_role_local_m_edge_tma = (
        tcgen05_m_edge_only and tcgen05_is_two_cta and tcgen05_cluster_n == 1
    )
    tcgen05_role_local_n_edge_tma = (
        tcgen05_n_edge_only and tcgen05_is_two_cta and tcgen05_cluster_n == 1
    )
    tcgen05_role_local_double_edge_tma = (
        tcgen05_double_edge_tma and tcgen05_is_two_cta and tcgen05_cluster_n == 1
    )
    tcgen05_role_local_uses_k_tail_tma = tcgen05_role_local_k_tail_tma or (
        tcgen05_role_local_double_edge_tma and tcgen05_has_k_tail
    )
    tcgen05_diagnose_cluster_m2_one_cta_role_local = bool(
        df.config.get(TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY, False)
    )
    tcgen05_cluster_m2_one_cta_role_local_bridge = (
        tcgen05_diagnose_cluster_m2_one_cta_role_local
        and mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_static_full_tiles
        and tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 2
        and not tcgen05_is_two_cta
        and bm == 128
        and bn == 256
        and bk == 128
    )
    if (
        tcgen05_diagnose_cluster_m2_one_cta_role_local
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY}=True requires "
            "static-full persistent tcgen05 TMA codegen for the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_role_local_codegen_allowed = (
        tcgen05_cluster_m == 1
        or tcgen05_is_two_cta
        or tcgen05_cluster_m2_one_cta_role_local_bridge
    )
    # The exact CtaGroup.ONE bridge duplicates A/B TMA production locally and
    # remains runtime-guarded. Do not apply the CtaGroup.TWO deferred cluster
    # pipeline protocol to that diagnostic shape.
    tcgen05_use_cluster_deferred_pipelines = (
        tcgen05_cluster_m > 1 and not tcgen05_cluster_m2_one_cta_role_local_bridge
    )
    # The clustered CtaGroup.ONE bridge flag is compile-proof only: ProgramID
    # still emits the host guard because this runtime path is not validated.
    tcgen05_use_role_local_tma_producer = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and (
            tcgen05_static_full_tiles
            or tcgen05_role_local_k_tail_tma
            or tcgen05_role_local_m_edge_tma
            or tcgen05_role_local_n_edge_tma
            or tcgen05_role_local_double_edge_tma
        )
        and tcgen05_role_local_codegen_allowed
        and tcgen05_pid_is_persistent
    )
    tcgen05_use_pure_matmul_role_lifecycle = (
        tcgen05_requested_pure_matmul_role_lifecycle
        and mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and tcgen05_static_full_tiles
        and not tcgen05_pid_is_persistent
        and tcgen05_cluster_m == 1
        and tcgen05_cluster_n == 1
        and acc_expr is None
    )
    if (
        tcgen05_requested_pure_matmul_role_lifecycle
        and not tcgen05_use_pure_matmul_role_lifecycle
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires static-full "
            "non-persistent tcgen05 TMA pure matmul with cluster_m=1, "
            "cluster_n=1, and an identity/zero accumulator epilogue",
        )
    tcgen05_use_separate_tma_producer = (
        tcgen05_use_role_local_tma_producer or tcgen05_use_pure_matmul_role_lifecycle
    )
    # Keep a distinct name so future MMA-exec gating changes are localized.
    tcgen05_use_role_local_mma_exec = tcgen05_use_role_local_tma_producer
    tcgen05_use_separate_mma_exec = (
        tcgen05_use_role_local_mma_exec or tcgen05_use_pure_matmul_role_lifecycle
    )
    tcgen05_pipeline_state_ns = (
        "_helion_tcgen05_pipeline"
        if tcgen05_use_pure_matmul_role_lifecycle
        else "cutlass.pipeline"
    )
    tcgen05_static_full_tma_fast_path = (
        tcgen05_static_full_tiles
        and tcgen05_use_tma_pipeline
        and not tcgen05_is_two_cta
        and not tcgen05_use_role_local_mma_exec
    )
    tcgen05_acc_producer_mode = df.config.get(
        TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_MODE_NORMAL,
    )
    diagnose_skip_umma_issue = (
        tcgen05_acc_producer_mode == TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA
    )
    if diagnose_skip_umma_issue and mma_impl != "tcgen05":
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY}="
            f"{TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA!r} requires tcgen05 MMA codegen",
        )
    tcgen05_acc_producer_advance_mode = df.config.get(
        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL,
    )
    diagnose_skip_acc_producer_advance = (
        tcgen05_acc_producer_advance_mode == TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP
    )
    if (
        diagnose_skip_acc_producer_advance
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY}="
            f"{TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP!r} requires the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_ab_producer_acquire_mode = df.config.get(
        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL,
    )
    diagnose_skip_ab_producer_acquire = (
        tcgen05_ab_producer_acquire_mode == TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP
    )
    if (
        diagnose_skip_ab_producer_acquire
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY}="
            f"{TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP!r} requires the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_ab_initial_producer_acquire_mode = df.config.get(
        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
    )
    diagnose_skip_initial_ab_producer_acquire = (
        tcgen05_ab_initial_producer_acquire_mode
        == TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST
    )
    if (
        diagnose_skip_initial_ab_producer_acquire
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY}="
            f"{TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST!r} requires "
            "the guarded cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_ab_producer_advance_mode = df.config.get(
        TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL,
    )
    diagnose_skip_ab_producer_advance = (
        tcgen05_ab_producer_advance_mode == TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP
    )
    if (
        diagnose_skip_ab_producer_advance
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY}="
            f"{TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP!r} requires the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_ab_consumer_wait_mode = df.config.get(
        TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL,
    )
    diagnose_skip_ab_consumer_wait = (
        tcgen05_ab_consumer_wait_mode == TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP
    )
    if (
        diagnose_skip_ab_consumer_wait
        and not tcgen05_cluster_m2_one_cta_role_local_bridge
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY}="
            f"{TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP!r} requires the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    tcgen05_ab_consumer_phase_mode = df.config.get(
        TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
        TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL,
    )
    diagnose_ab_consumer_phase1 = (
        tcgen05_ab_consumer_phase_mode == TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1
    )
    if diagnose_ab_consumer_phase1 and not tcgen05_cluster_m2_one_cta_role_local_bridge:
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY}="
            f"{TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1!r} requires the guarded "
            "cluster_m=2, CtaGroup.ONE, 128x256x128 bridge shape",
        )
    # Static-full CtaGroup.TWO keeps a prefetched AB consumer token live
    # across the accumulator acquire and each K-loop issue. CtaGroup.ONE
    # keeps the older adjacent try-wait/wait sequence.
    tcgen05_use_role_local_ab_consumer_prefetch = (
        tcgen05_use_role_local_mma_exec
        and tcgen05_is_two_cta
        and tcgen05_use_tma_pipeline
    )
    # Keep a distinct name so future epi-role gating changes are localized.
    tcgen05_use_role_local_epi = (
        tcgen05_use_role_local_tma_producer or tcgen05_use_pure_matmul_role_lifecycle
    )
    # This is the kernel-wide contract ProgramID consumes. Today the TMA
    # producer flag is the master predicate for all three role-local loops.
    tcgen05_use_role_local_persistent_body = tcgen05_use_role_local_tma_producer
    tcgen05_ab_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_ab_stages", _tcgen05_ab_stage_count(df.config.num_stages)
    )
    tcgen05_output_edge_tma_store_fits_smem = (
        tcgen05_ab_stage_count_value <= TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES
    )
    tcgen05_use_output_edge_tma_store_for_full_tiles = (
        tcgen05_role_local_m_edge_tma
        or tcgen05_role_local_n_edge_tma
        or tcgen05_role_local_double_edge_tma
    ) and tcgen05_output_edge_tma_store_fits_smem
    # Flat kernels process one output tile per CTA, so the c_pipeline stage is
    # just the subtile index. Persistent kernels use a role-local tile counter
    # to rotate c_pipeline stages across work tiles. Static-full CtaGroup.TWO
    # uses the same SMEM-staged TMA-store epilogue: each CTA's epilogue warp 0
    # stores its partitioned C tile. Output-edge role-local kernels can use the
    # same TMA-store path for interior full tiles while retaining the predicated
    # SIMT fallback for fringe tiles, but only when the AB-stage count leaves
    # enough SMEM budget for the extra epilogue tile.
    tcgen05_use_tma_store_epilogue = (
        mma_impl == "tcgen05"
        and tcgen05_use_tma_pipeline
        and (
            tcgen05_static_full_tiles
            or tcgen05_role_local_k_tail_tma
            or tcgen05_use_output_edge_tma_store_for_full_tiles
        )
        and tcgen05_role_local_codegen_allowed
        and (not _is_persistent_pid_config(df.config) or tcgen05_use_role_local_epi)
    )

    def tcgen05_tma_store_full_tiles_only_for(
        partial_output_tma_store: bool,
    ) -> bool:
        return (
            tcgen05_use_tma_store_epilogue
            and tcgen05_use_output_edge_tma_store_for_full_tiles
            and not tcgen05_static_output_tiles
            and not partial_output_tma_store
        )

    tcgen05_partial_output_tma_store = False
    tcgen05_tma_store_full_tiles_only = tcgen05_tma_store_full_tiles_only_for(
        tcgen05_partial_output_tma_store
    )
    if tcgen05_large_bn_proof and (
        mma_impl != "tcgen05"
        or tcgen05_is_two_cta
        or not tcgen05_use_tma_pipeline
        or not tcgen05_static_full_tiles
        or not tcgen05_use_tma_store_epilogue
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY}=True requires final tcgen05 "
            "CtaGroup.ONE TMA-load and TMA-store lowering for the G4 proof",
        )
    tcgen05_collective_handles_operand_loads = (
        mma_impl == "tcgen05"
        and fx_node is not None
        and cg.current_grid_state is not None
        and len(lhs_load.users) == 1
        and len(rhs_load.users) == 1
        and next(iter(lhs_load.users)) is fx_node
        and next(iter(rhs_load.users)) is fx_node
    )
    if tcgen05_collective_handles_operand_loads:
        cute_state = df.cute_state
        _register_collective_handled_loads(cute_state, lhs_load, rhs_load)
        grid_state = cg.current_grid_state
        assert grid_state is not None
        if grid_state.has_lane_loops():
            cute_state.request_root_lane_loop_suppression()

    # Variable names
    tiled_mma = df.new_var("tiled_mma")
    thr_mma = df.new_var("thr_mma")
    acc_frag = df.new_var("acc_frag")
    acc_frag_base = df.new_var("acc_frag_base")
    tcgen05_exec_acc_frag_base = df.new_var("tcgen05_exec_acc_frag_base")
    tcgen05_exec_acc_tmem_ptr = df.new_var("tcgen05_exec_acc_tmem_ptr")
    tcgen05_epi_acc_tmem_ptr = df.new_var("tcgen05_epi_acc_tmem_ptr")
    tcgen05_epi_acc_frag_base = df.new_var("tcgen05_epi_acc_frag_base")
    tcgen05_plan = _new_tcgen05_layout_plan(df) if mma_impl == "tcgen05" else None
    tcgen05_cluster_layout_vmnk = df.new_var("tcgen05_cluster_layout_vmnk")

    # === outer_prefix: MMA setup + shared memory alloc + accumulator init ===
    prefix = device_loop.outer_prefix
    suffix = device_loop.outer_suffix
    # This call corresponds to one MMA FX-node lowering. The later
    # register_tcgen05_kloop_owned_stmts slice starts here, so pre-existing
    # K-loop prelude remains outside the cleanup region and later FX-node code
    # remains unowned. Future role-lifecycle cleanup must pass that exact slice.
    # Operand-load scaffolding emitted before this snapshot is owned separately
    # by the FX-node statement-owner hook in GenerateAST.add_statement.
    tcgen05_kloop_stmt_start = len(device_loop.inner_statements)
    # Statements appended to ``prefix`` that reference per-tile coordinates
    # (m_offset_var, n_offset_var, advancing pipeline state). When the
    # persistent kernel splits the device-loop prefix, these stay inside the
    # work-tile loop while everything else hoists out. See
    # ``DeviceFunction.cute_state.register_tcgen05_per_tile_stmts`` and
    # ``ProgramID._split_tcgen05_invariant_setup``.
    per_tile_stmts: list[ast.AST] = []
    # Statements that conceptually belong to the TMA-load warp's role
    # block (see ``Tcgen05PersistentProgramIDs._collect_tcgen05_role_blocks``).
    # Statements get tagged only when the role-local producer path is active:
    # - The initial TMA tile/partition setup and prefetch cycle at the
    #   start of each tile. These are top-level statements added via
    #   ``_emit_per_tile(..., tma_load=True)``.
    # - The per-K-iter producer loop emitted as a top-level sibling of
    #   the shared consumer K-loop for persistent tcgen05 TMA-pipeline
    #   kernels. The role-local-while partitioner extracts that whole
    #   loop into the TMA-load warp's persistent loop.
    tma_load_role_stmts: list[ast.AST] = []
    # Statements conceptually owned by the MMA-exec warp. These are
    # extracted only with the same narrow static-full persistent tcgen05
    # predicate as the role-local TMA producer path.
    mma_exec_role_stmts: list[ast.AST] = []

    def _emit_per_tile(
        text: str, *, tma_load: bool = False, mma_exec: bool = False
    ) -> ast.stmt:
        """Append a per-tile statement to ``prefix`` and tag it for the
        persistent-loop splitter. Returns the AST node so callers can
        chain (e.g. when constructing ``If`` bodies). When ``tma_load``
        is true the statement is ALSO tagged for the role-block
        partitioner so it lands in the TMA-load warp's role block; when
        ``mma_exec`` is true it lands in the MMA-exec warp's role block.
        """
        stmt = statement_from_string(text)
        prefix.append(stmt)
        per_tile_stmts.append(stmt)
        if tma_load:
            tma_load_role_stmts.append(stmt)
        if mma_exec:
            mma_exec_role_stmts.append(stmt)
        return stmt

    tcgen05_tmem_setup_emitted = False

    def _emit_tcgen05_tmem_setup() -> None:
        nonlocal tcgen05_tmem_setup_emitted

        assert not tcgen05_tmem_setup_emitted
        assert tcgen05_plan is not None
        assert tcgen05_mma_owner_active is not None
        assert epi_active is not None
        tcgen05_tmem_setup_emitted = True

        if tcgen05_use_cluster_deferred_pipelines:
            # Keep the two-CTA cluster rendezvous after the AB/acc pipeline
            # objects exist and before any role allocates or retrieves TMEM.
            prefix.append(
                statement_from_string(
                    "cutlass.pipeline.pipeline_init_arrive("
                    f"cluster_shape_mn={tcgen05_cluster_layout_vmnk}, "
                    "is_relaxed=True)"
                )
            )
            prefix.append(
                statement_from_string(
                    "cutlass.pipeline.pipeline_init_wait("
                    f"cluster_shape_mn={tcgen05_cluster_layout_vmnk})"
                )
            )
        prefix.append(
            statement_from_string(
                f"if {epi_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.allocate({tcgen05_plan.acc_tmem_cols})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_exec_acc_tmem_ptr} = cute.make_ptr("
                f"{acc_dtype_str}, 0, cute.AddressSpace.tmem, assumed_align=16)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_epi_acc_tmem_ptr} = cute.make_ptr("
                f"{acc_dtype_str}, 0, cute.AddressSpace.tmem, assumed_align=16)"
            )
        )
        # ``acc_frag`` is reassigned per-tile below to a stage-indexed
        # slice; an extra ``acc_frag = acc_frag_base`` here would land
        # in the hoisted setup with a different CuTe type and break the
        # persistent ``while`` ("acc_frag is structured different after
        # this while").
        prefix.append(
            statement_from_string(
                f"if {tcgen05_plan.exec_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.wait_for_alloc()\n"
                f"    {tcgen05_exec_acc_tmem_ptr} = "
                f"{tcgen05_plan.tmem_allocator}.retrieve_ptr({acc_dtype_str})"
            )
        )
        prefix.append(
            statement_from_string(
                f"if {epi_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.wait_for_alloc()\n"
                f"    {tcgen05_epi_acc_tmem_ptr} = "
                f"{tcgen05_plan.tmem_allocator}.retrieve_ptr({acc_dtype_str})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_exec_acc_frag_base} = cute.make_tensor("
                f"{tcgen05_exec_acc_tmem_ptr}, {acc_frag_base}.layout)"
            )
        )
        # ``acc_frag`` indexes ``tcgen05_exec_acc_frag_base`` by the current
        # ``acc_producer_state.index`` stage. The K-loop suffix advances
        # that producer state once per UMMA fence, so under the persistent
        # path each tile sees a different index. Mark per-tile so the
        # alias is recomputed inside the work-tile loop.
        _emit_per_tile(
            f"{acc_frag} = "
            f"{tcgen05_exec_acc_frag_base}[None, None, None, "
            f"{tcgen05_plan.acc_producer_state}.index]",
            mma_exec=tcgen05_use_role_local_mma_exec,
        )
        if tcgen05_use_role_local_ab_consumer_prefetch:
            ab_consumer_prefetch_owner_predicate = _tcgen05_two_cta_owner_predicate(
                tcgen05_plan.exec_active,
                is_two_cta=tcgen05_is_two_cta,
                gate_exec_warp=False,
                cluster_n=tcgen05_cluster_n,
            )
            assert ab_consumer_prefetch_owner_predicate is not None
            _emit_per_tile(
                f"{tma_consumer_try_token} = cutlass.Boolean(1)",
                mma_exec=tcgen05_use_role_local_mma_exec,
            )
            _emit_per_tile(
                f"if {ab_consumer_prefetch_owner_predicate}:\n"
                f"    {tma_consumer_try_token} = "
                f"{tma_pipeline}.consumer_try_wait({tma_consumer_state})",
                mma_exec=tcgen05_use_role_local_mma_exec,
            )
        prefix.append(
            statement_from_string(
                f"{tcgen05_epi_acc_frag_base} = cute.make_tensor("
                f"{tcgen05_epi_acc_tmem_ptr}, {acc_frag_base}.layout)"
            )
        )
        # Initial producer_acquire for stage 0 of the acc pipeline. The
        # ``acc_producer_state`` advances once per UMMA fence inside the
        # K-loop, so per tile we want to start by acquiring whatever stage
        # the persistent loop currently points at. Tag as per-tile so this
        # acquire stays in the work-tile body when the persistent loop
        # splitter runs.
        _emit_per_tile(
            f"if {tcgen05_mma_owner_active}:\n"
            f"    {tcgen05_plan.acc_pipeline}.producer_acquire("
            f"{tcgen05_plan.acc_producer_state})",
            mma_exec=tcgen05_use_role_local_mma_exec,
        )
        reset_accumulate_stmt = _build_tcgen05_mma_accumulate_reset_stmt(
            tcgen05_plan.exec_active,
            tiled_mma=tiled_mma,
            is_two_cta=tcgen05_is_two_cta,
            cluster_n=tcgen05_cluster_n,
        )
        prefix.append(reset_accumulate_stmt)
        per_tile_stmts.append(reset_accumulate_stmt)
        if tcgen05_use_role_local_mma_exec:
            mma_exec_role_stmts.append(reset_accumulate_stmt)

    mma_participant_linear: str | None = None
    mma_slice_linear: str | None = None
    mma_copy_linear: str | None = None
    mma_active: str | None = None
    tma_warp: str | None = None
    warp_idx: str | None = None
    lane_idx: str | None = None
    epi_active: str | None = None
    epi_tidx: str | None = None
    mma_phys_n = _mma_active_n_threads(mma_impl)
    mma_physical_m_threads = _grid_thread_extent(cg, m_block_id)
    tcgen05_cta_thread_count = _grid_cta_thread_count(cg)
    if mma_impl == "tcgen05" and tcgen05_cluster_m * tcgen05_cluster_n > 1:
        df.cute_state.cluster_shape = (tcgen05_cluster_m, tcgen05_cluster_n, 1)
    tcgen05_acc_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_acc_stages", _tcgen05_acc_stage_count(bn)
    )
    # PipelineTmaUmma empty barriers are released by the leader CTA with the
    # pipeline's multicast mask. Peer CTAs still advance local consumer state,
    # but they must not add a second empty-barrier arrival: doing so lets the
    # TMA producer reuse an AB stage before the leader's ordered release when
    # the K loop has more than two tiles.
    #
    # ``mcast_size`` formula matches Quack's
    # ``num_mcast_ctas_a + num_mcast_ctas_b - 1``
    # (gemm_sm100.py:1839-1840). For Helion's V=2 path,
    # ``num_mcast_ctas_a = cluster_layout_vmnk.shape[2] = cluster_n`` and
    # ``num_mcast_ctas_b = cluster_layout_vmnk.shape[1] = cluster_m / V``.
    #
    # Concrete table (cute_plan.md §6.12.7 step 3):
    #   - cluster_m=2 cluster_n=1 V=2: 1 + 1 - 1 = 1 (today's value)
    #   - cluster_m=2 cluster_n=2 V=2: 2 + 1 - 1 = 2 (the cluster_n=2 fix)
    #   - cluster_m=1 cluster_n=1 V=1 (no use_2cta): 1 (collapses to single
    #     CTA; no multicast)
    if tcgen05_is_two_cta and tcgen05_cluster_n > 1:
        # V=2 absorbs cluster_m into the cluster_layout_vmnk V dim; the
        # post-V CTAs along M (= cluster_m // V) carry the B multicast,
        # while cluster_n CTAs along N carry the A multicast.
        v_for_mcast = 2 if tcgen05_is_two_cta else 1
        num_mcast_ctas_a = tcgen05_cluster_n
        num_mcast_ctas_b = max(1, tcgen05_cluster_m // v_for_mcast)
        tcgen05_ab_consumer_arrive_count_value = num_mcast_ctas_a + num_mcast_ctas_b - 1
    else:
        tcgen05_ab_consumer_arrive_count_value = 1
    tcgen05_c_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_c_stages", _tcgen05_c_stage_count(bn)
    )
    tcgen05_defer_pipeline_sync_arg = (
        ", defer_sync=True" if tcgen05_use_cluster_deferred_pipelines else ""
    )
    tcgen05_matmul_plan: CuteTcgen05MatmulPlan | None = None
    tcgen05_mma_owner_active: str | None = None
    # Initialized inside the ``mma_impl == "tcgen05"`` branch so the
    # non-tcgen05 path doesn't pay for warp-spec / barrier-count work
    # it never uses; consumed only at later tcgen05-gated emission
    # sites that share this `if mma_impl == "tcgen05":` predicate.
    tcgen05_epi_warp_count_value = 0
    tcgen05_tmem_barrier_thread_count_value = 0
    tcgen05_acc_consumer_arrive_count_value = 0
    # Layout-override values are read by separate later
    # ``if mma_impl == "tcgen05":`` blocks (the function has multiple
    # gated emission sites). Initialize to ``None`` outside the
    # branch so pyrefly's flow analysis sees a definition along every
    # control path; the values are pulled from the active config
    # below when the branch is entered.
    tcgen05_smem_swizzle_a: int | None = None
    tcgen05_smem_swizzle_b: int | None = None
    tcgen05_explicit_epi_tile_m: int | None = None
    tcgen05_explicit_epi_tile_n: int | None = None
    tcgen05_explicit_d_store_box_n: int | None = None
    tcgen05_use_pure_clc_scheduler_object = False
    tcgen05_use_flat_role_coordinates = False
    if mma_impl == "tcgen05":
        # Use ``warp_spec.ab_load_warps`` so the strategy data model
        # stays the source of truth for warp role IDs; ``epi_warps``
        # flows the same way via ``_tcgen05_epi_warp_count`` below.
        tcgen05_warp_spec = warp_spec_from_config(df.config)
        # Pull swizzle overrides off the config and validate the
        # bytes-per-row contract before constructing the matmul plan
        # so a bad config raises ``BackendUnsupported`` here rather
        # than silently inducing a CuTe ``ValueError`` at atom-build
        # time. ``layout_overrides_from_config`` returns ``None`` for
        # absent keys, which preserves the no-override byte-identity
        # path.
        _tcgen05_layout_overrides = layout_overrides_from_config(df.config)
        tcgen05_smem_swizzle_a = _tcgen05_layout_overrides.smem_swizzle_a
        tcgen05_smem_swizzle_b = _tcgen05_layout_overrides.smem_swizzle_b
        tcgen05_explicit_epi_tile_m = _tcgen05_layout_overrides.epi_tile_m
        tcgen05_explicit_epi_tile_n = _tcgen05_layout_overrides.epi_tile_n
        tcgen05_explicit_d_store_box_n = _tcgen05_layout_overrides.d_store_box_n
        if tcgen05_edge_scalar_fallback_needs_inter_smem_a:
            # The mixed TMA/scalar edge path writes logical (_row, _col)
            # coordinates into the tcgen05 A SMEM view. With bk=128,
            # CuTe's default SW128 A atom is correct for TMA-filled full
            # tiles but corrupts scalar-filled output-edge tiles. Force
            # the explicit INTER atom so fallback writes and the wrapper
            # descriptor agree on the same layout.
            if tcgen05_smem_swizzle_a not in (None, 0):
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 output-edge scalar fallback with bk >= 128 "
                    "requires smem_swizzle_a=0 (INTER); nonzero "
                    "A swizzle overrides are not validated for this path.",
                )
            tcgen05_smem_swizzle_a = 0
        if tcgen05_smem_swizzle_a is not None:
            _validate_tcgen05_smem_swizzle_override(
                operand="a",
                swizzle_bytes=tcgen05_smem_swizzle_a,
                bm=bm,
                bn=bn,
                bk=bk,
                input_dtype=input_dtype,
            )
        if tcgen05_smem_swizzle_b is not None:
            _validate_tcgen05_smem_swizzle_override(
                operand="b",
                swizzle_bytes=tcgen05_smem_swizzle_b,
                bm=bm,
                bn=bn,
                bk=bk,
                input_dtype=input_dtype,
            )
        tcgen05_epi_warp_count_value = _tcgen05_epi_warp_count(
            tcgen05_warp_spec, cta_thread_count=tcgen05_cta_thread_count
        )
        tcgen05_tmem_barrier_thread_count_value = _tcgen05_tmem_barrier_thread_count(
            tcgen05_epi_warp_count_value
        )
        # Each CtaGroup.TWO CTA has its own epilogue warps consuming
        # the distributed accumulator slot, so the acc empty barrier
        # expects each CTA's epi warp leaders. The CtaGroup.ONE
        # clustered fallback remains on the single-CTA count until
        # it has separate runtime coverage.
        tcgen05_acc_consumer_arrive_count_value = tcgen05_epi_warp_count_value * (
            2 if tcgen05_is_two_cta else 1
        )
        # Scheduler-warp pipeline depth. The default remains one stage because
        # consumer warps advance their own register-state independently; with
        # one shared mailbox, a multi-stage producer could overwrite metadata
        # while a slower consumer still reads the previous tile. The optional
        # two-stage diagnostic is paired with a staged SMEM mailbox in
        # ``program_id.py`` so the scheduler can publish the next work tile
        # without overwriting a slower consumer's current tile.
        tcgen05_sched_stage_count_value = (
            cast("int", df.config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, 1))
            if tcgen05_warp_spec.scheduler_warps > 0
            else 0
        )
        # Persistence model from the active config. Default
        # ``static_persistent`` keeps the existing path; G2-H
        # ``clc_persistent`` (cute_plan.md, see plan: G2-H CLC)
        # selects CLC issuance in the scheduler-warp role.
        # Validation in ``ConfigSpec.normalize`` has already ensured
        # the value is consistent with the chosen strategy + arch
        # (rejecting CLC under MONOLITHIC or on arch < 100), so a
        # missing field falls back to the static default rather
        # than raising.
        tcgen05_persistence_model_str = df.config.get(
            "tcgen05_persistence_model",
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        assert isinstance(tcgen05_persistence_model_str, str), (
            "tcgen05_persistence_model must be a string (the "
            "Tcgen05PersistenceModel enum's .value); got "
            f"{type(tcgen05_persistence_model_str).__name__}"
        )
        # ``tcgen05_l2_swizzle_size``: L2 tile-scheduler grouping factor
        # (Quack ``max_swizzle_size`` equivalent). Default 1 keeps the
        # cycle 41 byte-identity path; concrete values flow into the
        # ``cutlass.utils.PersistentTileSchedulerParams(swizzle_size=...)``
        # kwarg at every prelude site in ``program_id.py``. The value
        # is validated upstream by ``ConfigSpec.normalize`` against
        # ``TCGEN05_LEGAL_L2_SWIZZLE_SIZES`` so it is always a positive
        # integer here.
        tcgen05_l2_swizzle_size_value = l2_swizzle_size_from_config(df.config)
        # Both production callers of ``_emit_mma_pipeline`` propagate
        # an FX node into this codepath (``codegen_cute_mma_dot``
        # passes ``state.fx_node``; the aten-style site passes the
        # call node directly), so the tcgen05 branch requires an
        # ``fx_node``. The default-``None`` signature is a leftover
        # that the once-future-tcgen05 callers may set; assert it
        # here so a future caller that forgets to thread the FX node
        # fails loudly rather than silently producing an empty
        # ``aux_tensor_descriptors`` tuple and breaking the
        # productive-body codegen (which sizes the SMEM ring by
        # descriptor count).
        assert fx_node is not None, (
            "tcgen05 MMA codegen requires a non-None fx_node so the "
            "aux-tensor walker can identify downstream stores"
        )
        # Register the matmul fx_node in
        # ``cute_state.matmul_fx_nodes`` before the aux-tensor
        # walker runs. The walker's analyzer reads the same set to
        # know where to stop, so the fx_node must be present
        # *before* the walk. The registered-graph invariant: the
        # matmul fx_node lives inside a K-loop body subgraph (one
        # of the registered codegen graphs), so the cute_fx_walk
        # carrier walker reaches it via ``_phi.args[1]`` (body
        # branch), never ``_phi.args[0]`` (init value, e.g.
        # ``hl.zeros``). This holds structurally because
        # ``_emit_mma_pipeline`` only emits the MMA into the loop
        # body — there is no path where the matmul appears as the
        # phi's init value. The walker's ``_phi.args[1]``-only
        # descent depends on this invariant; pinning it here
        # prevents a future ``_emit_mma_pipeline`` refactor from
        # registering an off-graph node and silently breaking the
        # walker's fast-path. For non-residual kernels — and every
        # kernel until the productive C-input body lands — the
        # walker returns an empty tuple and the plan field defaults
        # to ``()``, preserving byte identity for every existing
        # config.
        assert any(fx_node.graph is gi.graph for gi in cg.codegen_graphs), (
            "matmul fx_node graph must be a registered codegen graph"
        )
        df.cute_state.matmul_fx_nodes.add(fx_node)
        aux_tensor_descriptors_value = discover_tcgen05_aux_tensor_descriptors(
            cg, fx_node
        )
        c_input_aux_tensor_descriptors_value = tuple(
            d for d in aux_tensor_descriptors_value if d.broadcast_axis is None
        )
        aux_tma_productive_body_gate_open = (
            tcgen05_warp_spec.c_input_warps > 0
            and bool(c_input_aux_tensor_descriptors_value)
            and len({d.store_value_node for d in c_input_aux_tensor_descriptors_value})
            <= 1
        )
        # CuTe's TMA descriptor bounds checks correctly suppress partial M/N
        # output stores for the admitted aux-TMA output-edge family, so keep
        # those edge tiles on the same TMA-store path as full tiles. Kernels
        # without a productive aux-TMA body keep the original predicated
        # full-tile/edge split.
        tcgen05_partial_output_tma_store = (
            tcgen05_use_tma_store_epilogue
            and tcgen05_use_output_edge_tma_store_for_full_tiles
            and not tcgen05_static_output_tiles
            and df.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
            == TCGEN05_AUX_LOAD_MODE_TMA
            and aux_tma_productive_body_gate_open
        )
        tcgen05_tma_store_full_tiles_only = tcgen05_tma_store_full_tiles_only_for(
            tcgen05_partial_output_tma_store
        )
        explicit_epi_tile_requested = any(
            value is not None
            for value in (
                tcgen05_explicit_epi_tile_m,
                tcgen05_explicit_epi_tile_n,
                tcgen05_explicit_d_store_box_n,
            )
        )
        tcgen05_use_flat_role_coordinates = tcgen05_requested_flat_role_coordinates
        # T2's rowvec ``acc + bias[n]`` epilogue surfaces a single rank-1
        # (broadcast_axis=1) aux descriptor. The aux pipeline keeps the
        # bias on the SIMT load path (no TMA), so the explicit
        # epilogue-tile family stays validated for the T2 envelope: the
        # store side still uses the same TMA-store + epi-tile shape as
        # T1/T3/T4/T5. Exact-shape rank-2 aux tensors (broadcast_axis=
        # None) and any other broadcast shape remain rejected here.
        aux_descriptors_compatible_with_explicit_epi_tile = all(
            d.broadcast_axis == 1 for d in aux_tensor_descriptors_value
        )
        # T10 (cycle 42) admits fp16 operands at the validated 2048³
        # bias-store shape. The explicit-epi-tile dtype gate stays
        # bf16-only for T1-T9; only T10's shape+bias-store unlocks fp16
        # here. Compute the bias-store fingerprint early so the dtype gate
        # below can require the bias chain — without it a fp16 identity
        # or relu kernel at 2048³ would otherwise slip through this gate.
        _t10_bias_store_present = (
            _trace_mma_to_single_bias_store_dtype(
                fx_node,
                cg.codegen_graphs,
                expected_bias_dtype=input_dtype,
            )
            is not None
        )
        explicit_epi_tile_t10_fp16_ok = (
            input_dtype == torch.float16
            and epi_elem_dtype_str == "cutlass.Float16"
            and (m_size, n_size, k_total_size) == TCGEN05_TARGET10_TVM_FFI_SHAPE
            and _t10_bias_store_present
        )
        explicit_epi_tile_bf16_ok = (
            input_dtype == torch.bfloat16 and epi_elem_dtype_str == "cutlass.BFloat16"
        )
        if explicit_epi_tile_requested:
            if not (
                tcgen05_static_full_tiles
                and tcgen05_is_two_cta
                and bm == TCGEN05_TWO_CTA_BLOCK_M
                and bn == TCGEN05_TWO_CTA_BLOCK_N
                and (explicit_epi_tile_bf16_ok or explicit_epi_tile_t10_fp16_ok)
                and aux_descriptors_compatible_with_explicit_epi_tile
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "explicit tcgen05 epilogue tile overrides are validated only "
                    "for static-full bf16 pure matmul CtaGroup.TWO kernels "
                    "(rank-1 rowvec aux tensors admitted for the T2 bias "
                    "envelope; fp16 admitted only for the T10 2048³ bias "
                    "envelope)",
                )
            if (
                tcgen05_explicit_epi_tile_m,
                tcgen05_explicit_epi_tile_n,
                tcgen05_explicit_d_store_box_n,
            ) != _TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE:
                raise exc.BackendUnsupported(
                    "cute",
                    "explicit tcgen05 epilogue tile is currently validated only "
                    "for epi_tile="
                    f"{_TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE[:2]} "
                    "and d_store_box_n="
                    f"{_TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE[2]}",
                )
        if tcgen05_use_flat_role_coordinates:
            if not (
                explicit_epi_tile_requested
                and (
                    tcgen05_explicit_epi_tile_m,
                    tcgen05_explicit_epi_tile_n,
                    tcgen05_explicit_d_store_box_n,
                )
                == _TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE
                and tcgen05_static_full_tiles
                and tcgen05_is_two_cta
                and tcgen05_cluster_n == 1
                and bm == TCGEN05_TWO_CTA_BLOCK_M
                and bn == TCGEN05_TWO_CTA_BLOCK_N
                # T1 validated at bk=64, T4 validated at bk=128. Both share
                # the bm=bn=256, cluster_m=2, cluster_n=1 envelope and use
                # the same flat-role launch shape. T10 (cycle 42) reuses
                # the same flat-role shape at fp16 operand dtype, pinned
                # to the 2048³ bias envelope below.
                and bk in (64, 128)
                and (explicit_epi_tile_bf16_ok or explicit_epi_tile_t10_fp16_ok)
                and aux_descriptors_compatible_with_explicit_epi_tile
                and tcgen05_use_tma_store_epilogue
                and tcgen05_warp_spec.scheduler_warps == 0
                and tcgen05_warp_spec.c_input_warps == 0
                and tcgen05_warp_spec.ab_load_warps == 1
                and tcgen05_warp_spec.epi_warps == 4
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    f"{TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY}=True requires "
                    "the guarded static-full bf16 pure matmul CtaGroup.TWO "
                    "256x256 bk in {64,128} explicit-epilogue-tile path "
                    "(fp16 admitted only for the T10 2048³ bias envelope)",
                )
        tcgen05_use_pure_clc_scheduler_object = (
            tcgen05_requested_pure_clc_scheduler_object
        )
        tcgen05_use_pure_dynamic_scheduler_object = (
            tcgen05_requested_pure_dynamic_scheduler_object
        )
        # Cycle-16 H3 Option B (staged): a request for the
        # ``DYNAMIC_PERSISTENT`` scheduler object is mutually exclusive
        # with the ``PURE_CLC`` scheduler object — they are alternate
        # persistence-model lowerings of the same dispatch slot. Reject
        # the simultaneous request loudly so a user / autotune
        # exploration cannot reach an ambiguous state. The autotune
        # surface gates these as separate ``BooleanFragment`` knobs;
        # the validator narrowing below ensures at most one is True
        # in normal search.
        if (
            tcgen05_use_pure_clc_scheduler_object
            and tcgen05_use_pure_dynamic_scheduler_object
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY}=True and "
                f"{TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY}=True "
                "are mutually exclusive scheduler-object selections "
                "(CLC vs dynamic-persistent persistence models cannot "
                "share the same lowering slot)",
            )
        tcgen05_use_direct_entry_plan = tcgen05_requested_direct_entry_plan
        if tcgen05_use_pure_clc_scheduler_object or tcgen05_use_direct_entry_plan:
            identity_store_dtype = (
                _trace_mma_to_single_identity_store_dtype(fx_node, cg.codegen_graphs)
                if fx_node is not None
                else None
            )
            # P2 (cycle-2 review): the pure-CLC scheduler-object diagnostic
            # is still narrow-gated to the validated T1 identity-store
            # envelope. The TVM-FFI direct entry additionally admits the
            # T4 (8192x1024x1024) relu-epilogue, T5 (1024x8192x1024)
            # identity-store, T3 (2048x4096x2048) identity-store, and
            # T2 (4096x2048x2048) rowvec bias-store envelopes at bk=128;
            # the relu (T4) and bias (T2) fusions are handled inside the
            # kernel body, so all the direct entry needs is to confirm
            # the store shape/dtype matches. The two surfaces are gated
            # separately so ``pure_clc_scheduler_object=True`` +
            # T2/T3/T4/T5 cannot silently admit a non-T1 launch through
            # an unvalidated CLC scheduler.
            relu_store_dtype = (
                _trace_mma_to_single_relu_store_dtype(fx_node, cg.codegen_graphs)
                if fx_node is not None
                else None
            )
            # Pin the bias-load dtype to the operand dtype so a kernel
            # with mismatched operand/bias dtypes (e.g. bf16 operands +
            # fp16 bias smuggled in) is rejected at the codegen walker
            # rather than only at the runtime validator.
            bias_store_dtype = (
                _trace_mma_to_single_bias_store_dtype(
                    fx_node,
                    cg.codegen_graphs,
                    expected_bias_dtype=input_dtype,
                )
                if fx_node is not None
                else None
            )
            bias_relu_store_dtype = (
                _trace_mma_to_single_bias_relu_store_dtype(
                    fx_node,
                    cg.codegen_graphs,
                    expected_bias_dtype=input_dtype,
                )
                if fx_node is not None
                else None
            )
            # A1 (cycle-2 second-pass review): each envelope predicate
            # pins ``bk`` to its validated block_k so a T1-shape config
            # with ``bk=128`` (or T3/T4/T5-shape with ``bk=64``) is
            # rejected here at codegen instead of only at the runtime
            # validator.
            target1_envelope_ok = (
                (m_size, n_size, k_total_size) == (1024, 4096, 1024)
                and bk == TCGEN05_TARGET1_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                in ((3, 2), (6, 4))
                and acc_expr is None
                and identity_store_dtype == input_dtype
            )
            target4_envelope_ok = (
                (m_size, n_size, k_total_size) == (8192, 1024, 1024)
                and bk == TCGEN05_TARGET4_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET4_TVM_FFI_AB_STAGES, TCGEN05_TARGET4_TVM_FFI_C_STAGES)
                and acc_expr is None
                and relu_store_dtype == input_dtype
            )
            # T5 (1024x8192x1024) uses identity store (no relu epilogue),
            # matching T1's epilogue shape but at the T4 bk=128 stage
            # tuple. The predicate is pinned to T5's shape so a T1-shape
            # config with bk=128 is still rejected (handled by the T1
            # predicate's bk pin), and a T4-shape config with identity
            # store would also fall through here.
            target5_envelope_ok = (
                (m_size, n_size, k_total_size) == (1024, 8192, 1024)
                and bk == TCGEN05_TARGET5_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET5_TVM_FFI_AB_STAGES, TCGEN05_TARGET5_TVM_FFI_C_STAGES)
                and acc_expr is None
                and identity_store_dtype == input_dtype
            )
            # T3 (2048x4096x2048) uses identity store at the same bk=128
            # (ab=3, c=2) stage tuple as T4/T5. The K=2048 problem yields
            # ``k_tile_count = 16`` (double T4/T5's 8); this fits the
            # ``TCGEN05_TWO_CTA_MAX_K_TILES`` cap and the (3, 2) stage
            # tuple. The predicate is pinned to T3's shape so a T1/T5
            # host function does not get a T3 admission and vice versa.
            target3_envelope_ok = (
                (m_size, n_size, k_total_size) == (2048, 4096, 2048)
                and bk == TCGEN05_TARGET3_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET3_TVM_FFI_AB_STAGES, TCGEN05_TARGET3_TVM_FFI_C_STAGES)
                and acc_expr is None
                and identity_store_dtype == input_dtype
            )
            # T2 (4096x2048x2048) uses a rank-1 trailing-axis (rowvec)
            # ``acc + bias[n]`` epilogue at the same bk=128 (ab=3, c=2)
            # stage tuple as T3/T4/T5. The bias-store walker (above)
            # confirms exactly one ``aten.add.Tensor(carrier,
            # bias_load)`` between the MMA carrier and the
            # convert-and-store node and that ``bias_load`` is a rank-1
            # GMEM load. The predicate is pinned to T2's shape so T6
            # (8192x2048x2048 + bias_relu) does not silently get a T2
            # admission, and so a T2-shape config with bk=64 (or a
            # T1/T3/T4/T5 host function at the T2 shape) is rejected.
            target2_envelope_ok = (
                (m_size, n_size, k_total_size) == (4096, 2048, 2048)
                and bk == TCGEN05_TARGET2_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET2_TVM_FFI_AB_STAGES, TCGEN05_TARGET2_TVM_FFI_C_STAGES)
                and acc_expr is None
                and bias_store_dtype == input_dtype
            )
            # T6 (8192x2048x2048) composes T2's rank-1 trailing-axis
            # (rowvec) ``acc + bias[n]`` with T4's relu activation:
            # ``relu(acc + bias[n])``. The bias-relu walker (above)
            # confirms exactly one ``aten.add.Tensor(carrier,
            # bias_load)`` followed by exactly one
            # ``aten.relu.default`` between the MMA carrier and the
            # convert-and-store node, and that ``bias_load`` is a
            # rank-1 bf16 GMEM load. The predicate is pinned to T6's
            # shape so a T2 (4096x2048x2048 + bias) host function does
            # not silently get a T6 admission, and so a T6-shape
            # config with bk=64 (or a T1/T3/T4/T5 host function at the
            # T6 shape) is rejected.
            target6_envelope_ok = (
                (m_size, n_size, k_total_size) == (8192, 2048, 2048)
                and bk == TCGEN05_TARGET6_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET6_TVM_FFI_AB_STAGES, TCGEN05_TARGET6_TVM_FFI_C_STAGES)
                and acc_expr is None
                and bias_relu_store_dtype == input_dtype
            )
            # T7 (2048x8192x2048) uses identity store at the same
            # bk=128 (ab=3, c=2) stage tuple as T3/T4/T5/T6. The
            # K=2048 problem yields ``k_tile_count = 16`` (same as
            # T2/T3/T6); this fits the ``TCGEN05_TWO_CTA_MAX_K_TILES``
            # cap. T7 is structurally similar to T5 (1024x8192x1024)
            # with M and K doubled (M_tiles*N_tiles = 8*32 = 256
            # work clusters, matching T6's 256). The identity-store
            # gate is shared with the T1/T3/T5 seeds; the shape gate
            # below pins it to T7 so a T1/T3/T5 host function does not
            # get a T7 admission and vice versa.
            target7_envelope_ok = (
                (m_size, n_size, k_total_size) == (2048, 8192, 2048)
                and bk == TCGEN05_TARGET7_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET7_TVM_FFI_AB_STAGES, TCGEN05_TARGET7_TVM_FFI_C_STAGES)
                and acc_expr is None
                and identity_store_dtype == input_dtype
            )
            # T9 (2048x2048x2048) uses identity store at the same
            # bk=128 (ab=3, c=2) stage tuple as T3/T4/T5/T6/T7. The
            # K=2048 problem yields ``k_tile_count = 16`` (same as
            # T2/T3/T6/T7); this fits the ``TCGEN05_TWO_CTA_MAX_K_TILES``
            # cap. T9 is the square 2048³ identity-store envelope
            # (M_tiles*N_tiles = 8*8 = 64 work clusters). The
            # identity-store gate is shared with the T1/T3/T5/T7 seeds;
            # the shape gate below pins it to T9 so a T1/T3/T5/T7 host
            # function does not get a T9 admission and vice versa.
            target9_envelope_ok = (
                (m_size, n_size, k_total_size) == (2048, 2048, 2048)
                and bk == TCGEN05_TARGET9_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (TCGEN05_TARGET9_TVM_FFI_AB_STAGES, TCGEN05_TARGET9_TVM_FFI_C_STAGES)
                and acc_expr is None
                and identity_store_dtype == input_dtype
            )
            # T10 (2048x2048x2048) uses a rank-1 trailing-axis (rowvec)
            # ``acc + bias[n]`` epilogue at the same bk=128 (ab=3, c=2)
            # stage tuple as T2 but at the square 2048³ shape (T9's
            # shape, T2's bias add). T10 is the only direct-entry
            # envelope that admits fp16 operands — every other
            # envelope is pinned to bf16 via the upstream
            # explicit-epi-tile / flat-role-coordinates gates and via
            # the per-target ``input_dtype in (bf16, fp16)`` check
            # below. The bias-store walker confirms exactly one
            # ``aten.add.Tensor(carrier, bias_load)`` in the chain;
            # the cycle-42 walker is dtype-pinned to the operand
            # dtype so an fp16-bias kernel against a bf16-operand
            # matmul (or vice versa) is rejected at the host detector
            # before it can produce a T10 seed.
            target10_envelope_ok = (
                (m_size, n_size, k_total_size) == TCGEN05_TARGET10_TVM_FFI_SHAPE
                and bk == TCGEN05_TARGET10_TVM_FFI_BLOCK_K
                and (tcgen05_ab_stage_count_value, tcgen05_c_stage_count_value)
                == (
                    TCGEN05_TARGET10_TVM_FFI_AB_STAGES,
                    TCGEN05_TARGET10_TVM_FFI_C_STAGES,
                )
                and acc_expr is None
                and bias_store_dtype == input_dtype
                and input_dtype in (torch.bfloat16, torch.float16)
            )
            common_ok = (
                tcgen05_use_flat_role_coordinates
                and TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY in df.config
                and df.config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
                and tcgen05_acc_stage_count_value == 2
            )
            if tcgen05_use_pure_clc_scheduler_object:
                pure_clc_ok = common_ok and target1_envelope_ok
                if not pure_clc_ok:
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY}=True "
                        "requires the validated T1 identity-store "
                        "TVM-FFI flat-role seed (pure-CLC scheduler "
                        "object is not validated for T2 bias-store, "
                        "T3 identity-store, T4 relu-store, T5 "
                        "identity-store, T6 bias-relu-store, T7 "
                        "identity-store, T9 identity-store, or T10 "
                        "fp16/bf16 bias-store)",
                    )
            if tcgen05_use_direct_entry_plan:
                direct_entry_ok = common_ok and (
                    target1_envelope_ok
                    or target4_envelope_ok
                    or target5_envelope_ok
                    or target3_envelope_ok
                    or target2_envelope_ok
                    or target6_envelope_ok
                    or target7_envelope_ok
                    or target9_envelope_ok
                    or target10_envelope_ok
                )
                if not direct_entry_ok:
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_DIRECT_ENTRY_PLAN_CONFIG_KEY}=True "
                        "requires the validated T1 identity-store, T2 "
                        "bias-store, T3 identity-store, T4 relu-store, "
                        "T5 identity-store, T6 bias-relu-store, T7 "
                        "identity-store, T9 identity-store, or T10 "
                        "fp16/bf16 bias-store TVM-FFI flat-role seed",
                    )
        # Cycle-16 H3 Option B (staged): the productive
        # ``DYNAMIC_PERSISTENT`` codegen path is staged for cycle 17.
        # Any request for the pure-dynamic scheduler object reaches the
        # dispatch slot but bails out here with a clear error so the
        # autotuner / explicit-config caller sees a deterministic
        # rejection. Until the productive emission lands, this raise
        # is the entire user-visible behavior of the knob. The
        # rejection happens AFTER the pure-CLC envelope checks so the
        # mutually-exclusive guard above is the source of truth for
        # "you cannot select both"; this raise is the source of truth
        # for "the dynamic-persistent codegen has not landed yet".
        if tcgen05_use_pure_dynamic_scheduler_object:
            raise exc.BackendUnsupported(
                "cute",
                f"{TCGEN05_PURE_DYNAMIC_SCHEDULER_OBJECT_CONFIG_KEY}=True "
                "is cycle-16 infrastructure only; the productive "
                "DYNAMIC_PERSISTENT codegen (atomic-counter "
                "tile_count_semaphore work-tile loop) is staged for "
                "cycle 17 (see cute_plan.md §6 Target 8 cycle-16 H3 "
                "Option B landing)",
            )
        tcgen05_scheduler_warp_count_for_plan = (
            1
            if tcgen05_use_pure_clc_scheduler_object
            else tcgen05_warp_spec.scheduler_warps
        )
        tcgen05_sched_stage_count_for_plan = (
            max(1, tcgen05_sched_stage_count_value)
            if tcgen05_use_pure_clc_scheduler_object
            else tcgen05_sched_stage_count_value
        )
        tcgen05_persistence_model_for_plan = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
            if tcgen05_use_pure_clc_scheduler_object
            else tcgen05_persistence_model_str
        )
        tcgen05_matmul_plan = CuteTcgen05MatmulPlan(
            bm=bm,
            bn=bn,
            bk=bk,
            k_tile_count=(k_total_size + bk - 1) // bk,
            cluster_m=tcgen05_cluster_m,
            is_two_cta=tcgen05_is_two_cta,
            uses_role_local_persistent_body=tcgen05_use_role_local_persistent_body,
            uses_cluster_m2_one_cta_role_local_bridge=(
                tcgen05_cluster_m2_one_cta_role_local_bridge
            ),
            cta_thread_count=tcgen05_cta_thread_count,
            physical_m_threads=mma_physical_m_threads,
            acc_stage_count=tcgen05_acc_stage_count_value,
            ab_stage_count=tcgen05_ab_stage_count_value,
            c_stage_count=tcgen05_c_stage_count_value,
            epi_warp_count=tcgen05_epi_warp_count_value,
            ab_load_warp_count=tcgen05_warp_spec.ab_load_warps,
            scheduler_warp_count=tcgen05_scheduler_warp_count_for_plan,
            sched_stage_count=tcgen05_sched_stage_count_for_plan,
            # ``c_input_warp_count`` plumbs the warp-spec slot
            # through the matmul plan (``cute_plan.md`` §7.5.3.2).
            # Validator restricts the value to ``{0, 1}`` under
            # WITH_SCHEDULER and ``{0}`` under MONOLITHIC; codegen
            # body for the C-input warp is inert today.
            c_input_warp_count=tcgen05_warp_spec.c_input_warps,
            persistence_model=tcgen05_persistence_model_for_plan,
            cluster_n=tcgen05_cluster_n,
            l2_swizzle_size=tcgen05_l2_swizzle_size_value,
            tma_store_full_tiles_only=tcgen05_tma_store_full_tiles_only,
            aux_tensor_descriptors=aux_tensor_descriptors_value,
            flat_role_launch_warp_count=8
            if tcgen05_use_flat_role_coordinates
            else None,
        )
        assert tcgen05_plan is not None
        tcgen05_mma_owner_active = _tcgen05_two_cta_owner_predicate(
            tcgen05_plan.exec_active,
            is_two_cta=tcgen05_is_two_cta,
            gate_exec_warp=True,
            cluster_n=tcgen05_cluster_n,
        )
        candidate_block_shape = tcgen05_matmul_plan.block_shape
        df.cute_state.register_tcgen05_matmul_plan(tcgen05_matmul_plan)
        if (
            candidate_block_shape[0]
            * candidate_block_shape[1]
            * candidate_block_shape[2]
            > 1024
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"tcgen05 launch block shape {candidate_block_shape} exceeds 1024 threads",
            )
        # SMEM-budget rejection: ``tcgen05_ab_stages=3`` +
        # productive C-input warp
        # (``tcgen05_warp_spec_c_input_warps=1`` AND non-empty
        # ``aux_tensor_descriptors`` AND single-store fan-out
        # gate open) is over the 232 KB B200 SMEM cap at every
        # validated tcgen05 tile shape (canonical
        # ``(bm=bn=256, bk=128, cluster_m=2)`` measured 263 KB
        # used vs 232 KB cap; reducing the aux ring to
        # ``num_stages=1`` only drops it to 246 KB, still 14 KB
        # over). Reject loudly at MMA-codegen time so:
        #   - explicit user configs fail with a clear message
        #     instead of an opaque ``ptxas: uses too much
        #     shared data`` deep inside the cute_dsl invocation;
        #   - the autotune search-time fixup can demote
        #     ``tcgen05_ab_stages=3`` candidates that would
        #     otherwise trigger this raise mid-tuning (see
        #     ``_fix_tcgen05_ab_stages_three_search_config``).
        # The predicate mirrors the productive-body aux-pipeline
        # allocation gate at ``_emit_mma_pipeline`` below
        # (``has_c_input_warp AND aux_tensor_descriptors AND
        # aux_single_store_value``). When the multi-store
        # fan-out gate closes the productive body, the aux SMEM
        # ring + ``c_pipeline_aux`` are NOT allocated and the
        # kernel falls back to GMEM-aux reads with no extra
        # SMEM cost, so the rejection must NOT fire — fan-out
        # ``ab=3 + c_input=1`` paths are legal and pinned by
        # ``test_aux_pipeline_ab_stages_3_with_c_input_fanout_not_rejected``.
        c_input_aux_tensor_descriptors = (
            tcgen05_matmul_plan.c_input_aux_tensor_descriptors
        )
        aux_single_store_value = (
            len({d.store_value_node for d in c_input_aux_tensor_descriptors}) <= 1
        )
        if (
            tcgen05_matmul_plan.has_c_input_warp
            and c_input_aux_tensor_descriptors
            and aux_single_store_value
            and tcgen05_matmul_plan.ab_stage_count >= 3
        ):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 ``tcgen05_ab_stages=3`` is incompatible "
                "with the productive C-input warp "
                "(``tcgen05_warp_spec_c_input_warps=1`` + "
                "non-empty aux tensors from the epilogue chain): "
                "the aux SMEM ring + AB pipeline together "
                "overshoot the 232 KB B200 SMEM cap at every "
                "validated tile shape (the canonical "
                "``(bm=bn=256, bk=128, cluster_m=2)`` shape uses "
                "263 KB vs 232 KB cap). Drop to "
                "``tcgen05_ab_stages=2`` for residual epilogues "
                "with ``tcgen05_warp_spec_c_input_warps=1``, or "
                "drop ``tcgen05_warp_spec_c_input_warps=0`` to "
                "keep ``tcgen05_ab_stages=3``. See "
                "``cute_plan.md`` §1.3 / §7.5.3.2.",
            )
        df.cute_state.block_shape = candidate_block_shape
    if mma_impl == "universal":
        prefix.extend(
            _make_tiled_mma_setup(
                mma_impl,
                tiled_mma,
                thr_mma,
                f"{m_local} + ({n_local}) * cutlass.Int32({bm})",
                input_dtype_str,
                acc_dtype_str,
                bm,
                bn,
                tcgen05_cluster_m=tcgen05_cluster_m,
            )
        )
    else:
        mma_participant_linear = df.new_var("mma_tidx")
        mma_slice_linear = df.new_var("mma_slice_tidx")
        mma_copy_linear = df.new_var("mma_copy_tidx")
        mma_active = df.new_var("mma_active")
        tma_warp = df.new_var("tcgen05_tma_warp")
        warp_idx = df.new_var("tcgen05_warp_idx")
        lane_idx = df.new_var("tcgen05_lane_idx")
        epi_active = df.new_var("tcgen05_epi_active")
        epi_tidx = df.new_var("tcgen05_epi_tidx")
        prefix.append(
            statement_from_string(
                f"{warp_idx} = cute.arch.make_warp_uniform(cute.arch.warp_idx())"
            )
        )
        prefix.append(statement_from_string(f"{lane_idx} = cute.arch.lane_idx()"))
        if tcgen05_use_flat_role_coordinates:
            mma_role_coordinates = _flat_mma_role_coordinate_plan(
                lane_idx=lane_idx,
                warp_idx=warp_idx,
                mma_active_n_threads=mma_phys_n,
            )
        else:
            mma_role_coordinates = _block_axis_mma_role_coordinate_plan(
                cg,
                m_block_id=m_block_id,
                n_block_id=n_block_id,
                mma_m_thread_extent=mma_physical_m_threads,
                mma_active_n_threads=mma_phys_n,
            )
        prefix.append(
            statement_from_string(
                f"{mma_participant_linear} = {mma_role_coordinates.mma_tidx_expr()}"
            )
        )
        prefix.append(
            statement_from_string(
                f"{mma_copy_linear} = "
                + (
                    mma_participant_linear
                    if tcgen05_collective_handles_operand_loads
                    else f"{m_local} + ({n_local}) * cutlass.Int32({bm})"
                )
            )
        )
        prefix.append(
            statement_from_string(
                f"{mma_active} = {mma_role_coordinates.mma_active_expr()}"
            )
        )
        if mma_impl == "tcgen05":
            assert tcgen05_plan is not None
            assert tcgen05_matmul_plan is not None
            # The current lowering has a single A/B load warp at
            # `tma_warp_id`, so `tma_warp` doubles as the A/B-load-active
            # predicate. When role-local persistent loops land and split
            # those roles, this is the place to add a separate
            # `tcgen05_ab_load_active` predicate.
            prefix.append(
                statement_from_string(
                    f"{tma_warp} = {warp_idx} == cutlass.Int32({tcgen05_matmul_plan.tma_warp_id})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tcgen05_plan.exec_active} = "
                    f"{warp_idx} == cutlass.Int32({tcgen05_matmul_plan.exec_warp_id})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{epi_active} = "
                    f"{warp_idx} < cutlass.Int32({tcgen05_matmul_plan.epi_warp_count})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{epi_tidx} = "
                    f"{_mma_epi_tidx_expr(lane_idx=lane_idx, warp_idx=warp_idx, epi_active=epi_active)}"
                )
            )
            # Register reallocation: consumer warps (exec MMA + epilogue
            # warps) request a larger per-thread register budget; the
            # producer warps (TMA load, A/B load) drop to the producer
            # budget. The "not consumer" form is just a more compact
            # spelling of "tma_warp or ab_load_active"; both are
            # equivalent now that the launched CTA has no idle padding
            # warps. Matches Quack's sm100 split. The setmaxregister
            # calls are warp-uniform and must precede the first pipeline
            # op of each role; placing them with the role-gate invariants
            # keeps them out of the per-tile work loop.
            #
            # ``setmaxregister`` is *warpgroup-uniform* (every warp in
            # the same 4-warp warpgroup must call the same value) on
            # sm_100a. Under ``ROLE_LOCAL_MONOLITHIC`` the 6 launched
            # warps are 4 epi + 1 exec + 1 tma_load; the consumer
            # predicate ``exec_active or epi_active`` covers warps
            # 0-4 (warpgroup 0 + warp 4 of warpgroup 1) and the
            # producer covers warp 5. Warpgroup 1 is partially
            # populated (warps 4 + 5; warps 6-7 absent), and the
            # mixed setmaxregister within warpgroup 1 happens to be
            # tolerated by hardware because the CTA shape has only
            # warps 4 and 5 (no warps 6/7 to disagree with).
            #
            # Under ``ROLE_LOCAL_WITH_SCHEDULER`` the launched CTA
            # has 7 warps (4 epi + 1 exec + 1 tma_load + 1 sched).
            # If we kept the MONOLITHIC consumer predicate the
            # warpgroup-1 warps would split as
            # exec=increase / tma=decrease / sched=decrease, which
            # is a real warpgroup-uniformity violation that triggers
            # ``CUDA_ERROR_LAUNCH_FAILED`` at launch on sm_100a.
            # Match Quack's pattern: only the 4 epi warps are
            # consumers; exec joins the producer warpgroup (lower
            # register budget) so warpgroup 1 is uniformly
            # decrease.
            if (
                tcgen05_matmul_plan.has_scheduler_warp
                or tcgen05_use_flat_role_coordinates
            ):
                consumer_predicate = epi_active
            else:
                consumer_predicate = f"{tcgen05_plan.exec_active} or {epi_active}"
            # Cycle 15 H2 (cute_plan.md §6 Target 8): the consumer-warp
            # register ceiling is config-driven. Default 256 preserves
            # cycle-14 byte-identity; lower values cap ``ptxas``'s
            # per-thread register allocation and force a spill rather
            # than reserving at the natural 255-reg peak. The autotune
            # gate ``consumer_regs_autotune_fragments`` admits the knob
            # only for the T8 wide-N CLC + aux-TMA seed family (same
            # gate as ``aux_stages``), so T1-T7 stay byte-identical at
            # the 256 default.
            consumer_regs_value = _tcgen05_consumer_regs_from_config(df.config)
            prefix.append(
                statement_from_string(
                    f"if not ({consumer_predicate}):\n"
                    f"    cute.arch.setmaxregister_decrease("
                    f"{_TCGEN05_PRODUCER_REGS})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"if {consumer_predicate}:\n"
                    f"    cute.arch.setmaxregister_increase("
                    f"{consumer_regs_value})"
                )
            )
            prefix.append(
                statement_from_string(
                    # tcgen05 tiled_mma slicing is CTA-scoped, not per-thread.
                    # Quack/CUTLASS use the CTA's MMA tile coordinate here.
                    # On Helion's 1-CTA path that is always 0, but clustered
                    # widened kernels need the cluster-local CTA rank so each
                    # CTA takes the right MMA slice.
                    f"{mma_slice_linear} = "
                    + (
                        "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
                        f"% cutlass.Int32({tcgen05_cluster_m})"
                        if tcgen05_cluster_m > 1
                        else "cutlass.Int32(0)"
                    )
                )
            )
            prefix.extend(
                _make_tiled_mma_setup(
                    mma_impl,
                    tiled_mma,
                    thr_mma,
                    mma_slice_linear,
                    input_dtype_str,
                    acc_dtype_str,
                    bm,
                    bn,
                    tcgen05_cluster_m=tcgen05_cluster_m,
                )
            )
        else:
            prefix.append(
                statement_from_string(f"{tma_warp} = {warp_idx} == cutlass.Int32(0)")
            )
            prefix.extend(
                _make_tiled_mma_setup(
                    mma_impl,
                    tiled_mma,
                    thr_mma,
                    mma_participant_linear,
                    input_dtype_str,
                    acc_dtype_str,
                    bm,
                    bn,
                    tcgen05_cluster_m=tcgen05_cluster_m,
                )
            )
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        assert tcgen05_mma_owner_active is not None
        prefix.append(
            statement_from_string(
                f"{tcgen05_cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout(({tcgen05_cluster_m}, {tcgen05_cluster_n}, 1)), "
                f"({tiled_mma}.thr_id.shape,))"
            )
        )
        prefix.extend(
            _make_tcgen05_layout_plan_setup(
                tcgen05_plan,
                tiled_mma,
                bm=bm,
                bn=bn,
                bk=bk,
                ab_stage_count=tcgen05_ab_stage_count_value,
                is_two_cta=tcgen05_is_two_cta,
                input_dtype_str=input_dtype_str,
                acc_dtype_str=acc_dtype_str,
                epi_elem_dtype_str=epi_elem_dtype_str,
                smem_swizzle_a=tcgen05_smem_swizzle_a,
                smem_swizzle_b=tcgen05_smem_swizzle_b,
                explicit_epi_tile_m=tcgen05_explicit_epi_tile_m,
                explicit_epi_tile_n=tcgen05_explicit_epi_tile_n,
            )
        )
        prefix.append(
            statement_from_string(
                f"{acc_frag_base} = {tiled_mma}.make_fragment_C("
                f"cute.append({tiled_mma}.partition_shape_C(({bm}, {bn})), "
                f"{tcgen05_acc_stage_count_value}))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_tmem_cols} = cutlass.utils.get_num_tmem_alloc_cols("
                f"{acc_frag_base}, arch='sm_100')"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_holding_buf} = cute.arch.alloc_smem(cutlass.Int32, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_dealloc_mbar_ptr} = cute.arch.alloc_smem(cutlass.Int64, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_alloc_barrier} = cutlass.pipeline.NamedBarrier("
                f"barrier_id={_TCGEN05_TMEM_ALLOC_BARRIER_ID}, "
                f"num_threads={tcgen05_tmem_barrier_thread_count_value})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator} = cutlass.utils.TmemAllocator("
                f"{tcgen05_plan.tmem_holding_buf}, "
                f"barrier_for_retrieve={tcgen05_plan.tmem_alloc_barrier}, "
                f"allocator_warp_id=0, is_two_cta={tcgen05_is_two_cta!s}, "
                f"two_cta_tmem_dealloc_mbar_ptr={tcgen05_plan.tmem_dealloc_mbar_ptr})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_barriers} = cute.arch.alloc_smem("
                f"cutlass.Int64, cutlass.Int32({tcgen05_acc_stage_count_value * 2}))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_producer_group} = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_consumer_group} = "
                f"cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_acc_consumer_arrive_count_value}))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline} = cutlass.pipeline.PipelineUmmaAsync.create("
                f"num_stages={tcgen05_acc_stage_count_value}, "
                f"producer_group={tcgen05_plan.acc_pipeline_producer_group}, "
                f"consumer_group={tcgen05_plan.acc_pipeline_consumer_group}, "
                f"barrier_storage={tcgen05_plan.acc_pipeline_barriers}, "
                f"cta_layout_vmnk={tcgen05_cluster_layout_vmnk}"
                f"{tcgen05_defer_pipeline_sync_arg})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_producer_state} = {tcgen05_pipeline_state_ns}.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, {tcgen05_acc_stage_count_value})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_consumer_state} = {tcgen05_pipeline_state_ns}.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, {tcgen05_acc_stage_count_value})"
            )
        )
        # ``ROLE_LOCAL_WITH_SCHEDULER`` allocates a scheduler-broadcast
        # ``PipelineAsync`` here. The plan and emission helpers live in
        # this file (``_new_tcgen05_sched_pipeline_plan`` /
        # ``_emit_sched_pipeline_setup``); the variable names are
        # registered on ``DeviceFunction`` so ``program_id.py`` can
        # emit consumer-side ``consumer_wait`` / ``consumer_release``
        # against the same plan. The ``MONOLITHIC`` byte-identity
        # path is preserved because this branch is gated on
        # ``has_scheduler_warp`` and emits no ``df.new_var`` when
        # scheduler_warps == 0.
        assert tcgen05_matmul_plan is not None
        if tcgen05_matmul_plan.has_scheduler_warp:
            tcgen05_sched_plan = _new_tcgen05_sched_pipeline_plan(
                df, use_clc=tcgen05_matmul_plan.is_clc_persistent
            )
            df.cute_state.register_tcgen05_sched_pipeline_plan(tcgen05_sched_plan)
            if tcgen05_use_pure_clc_scheduler_object:
                df.cute_state.register_tcgen05_pure_clc_scheduler_object(
                    Tcgen05PureClcSchedulerObject(
                        sched_plan=tcgen05_sched_plan,
                        scheduler_warp_id=tcgen05_matmul_plan.scheduler_warp_id,
                    )
                )
            # WITH_SCHEDULER's scheduler-warp topology: every CTA in
            # the cluster runs its own scheduler warp, publishing to
            # its own SMEM mailbox. Both CTAs converge on the same
            # cluster-level virtual_pid because the consumer's
            # ``virtual_pid = work_tile_smem[0] // cluster_m + ...``
            # formula collapses the per-CTA ``cta_id_in_cluster``
            # offset that ``StaticPersistentTileScheduler.create``
            # bakes into ``tile_idx[0]``. So each CTA's scheduler
            # publishes locally and each CTA's consumers release
            # locally — no peer-CTA broadcast needed. The
            # ``consumer_arrive_count`` is therefore *per-CTA*
            # (no ``× cluster_size`` multiplier), and
            # ``consumer_mask_to_leader=False`` keeps releases on
            # the local empty barrier. Quack's
            # ``make_sched_pipeline`` uses a different topology —
            # single cluster-leader scheduler with peer-CTA
            # broadcast — and consequently sets the cluster-wide
            # arrive count and ``consumer_mask=Int32(0)``. Picking
            # the wrong pair for the active topology starves
            # non-leader CTAs of empty-barrier arrivals and hangs
            # the kernel.
            # CLC overlays a different sched_pipeline topology than
            # the static path: leader-only producer + cluster-routed
            # empty mbar (mirrors Quack's ``make_sched_pipeline`` for
            # ``cluster_size > 1``). For cluster_size == 1 the two
            # topologies degenerate to the same per-CTA shape.
            # ``cluster_size`` is the full cluster envelope
            # (``cluster_m * cluster_n``) so the cluster-wide arrive
            # count under cluster_n=2 spans the full 4-CTA cluster
            # AND the deferred-init protocol participates in the same
            # cluster-wide barrier init as the AB / acc pipelines.
            # For cluster_n=2 under ``ROLE_LOCAL_WITH_SCHEDULER`` the
            # per-CTA-local scheduler topology is preserved (each CTA
            # in the 4-CTA cluster runs its own scheduler that
            # publishes locally and consumers release locally — see
            # the ``consumer_mask_to_leader=False`` branch below);
            # only the deferred-init participation needs the full
            # cluster envelope so every CTA in the cluster contributes
            # its arrival to the cluster-wide ``pipeline_init``
            # barrier.
            tcgen05_sched_cluster_size = tcgen05_cluster_m * tcgen05_cluster_n
            # Consumer arrive count excludes the scheduler warp (the
            # producer of this pipeline). The C-input warp's
            # participation depends on whether the productive-body
            # gate fires (``has_c_input_warp AND
            # aux_tensor_descriptors``):
            #
            # - Gate fires: the C-input role-local while emitted by
            #   ``program_id._build_c_input_warp_role_local_while``
            #   consumer-waits on the sched_pipeline every iteration
            #   (``cute_plan.md`` §7.5.3.2 cycle 1: empty body, but
            #   the wait/release runs to receive the per-tile work
            #   coords for cycles 2/3). Include the C-input warp in
            #   the arrive count.
            #
            # - Gate does not fire (``c_input_warps=1`` without an
            #   aux residual, or ``c_input_warps=0``): the C-input
            #   warp body is fully inert and never calls
            #   ``consumer_arrive`` on the sched_pipeline. Subtract
            #   ``c_input_warp_count`` so ``producer_commit`` is not
            #   blocked on a missing arrival.
            #
            # With ``c_input_warps=0`` the subtraction is a no-op
            # and the byte-identity path is preserved exactly.
            # Mirror the multi-store fan-out gate from
            # ``_build_c_input_warp_role_local_while`` /
            # the aux pipeline allocation below: the C-input
            # warp only participates as a sched consumer when
            # the productive body actually fires.
            _aux_single_store_value = (
                len(
                    {
                        d.store_value_node
                        for d in tcgen05_matmul_plan.c_input_aux_tensor_descriptors
                    }
                )
                <= 1
            )
            c_input_is_sched_consumer = (
                tcgen05_matmul_plan.has_c_input_warp
                and bool(tcgen05_matmul_plan.c_input_aux_tensor_descriptors)
                and _aux_single_store_value
            )
            tcgen05_sched_consumer_role_count = (
                tcgen05_matmul_plan.role_warp_count
                - tcgen05_matmul_plan.scheduler_warp_count
                - (
                    0
                    if c_input_is_sched_consumer
                    else tcgen05_matmul_plan.c_input_warp_count
                )
            )
            if tcgen05_matmul_plan.is_clc_persistent and tcgen05_sched_cluster_size > 1:
                tcgen05_sched_consumer_arrive_count = (
                    tcgen05_sched_consumer_role_count * tcgen05_sched_cluster_size
                )
                tcgen05_sched_consumer_mask_to_leader = True
            else:
                tcgen05_sched_consumer_arrive_count = tcgen05_sched_consumer_role_count
                tcgen05_sched_consumer_mask_to_leader = False
            prefix.extend(
                _emit_sched_pipeline_setup(
                    tcgen05_sched_plan,
                    sched_stage_count=tcgen05_matmul_plan.sched_stage_count,
                    consumer_arrive_count=tcgen05_sched_consumer_arrive_count,
                    cluster_size=tcgen05_sched_cluster_size,
                    defer_sync=tcgen05_use_cluster_deferred_pipelines,
                    consumer_mask_to_leader=tcgen05_sched_consumer_mask_to_leader,
                    # One leader thread (lane 0 of the scheduler
                    # warp) arrives on the full barrier per stage
                    # via ``producer_commit``.
                    producer_arrive_count=1,
                )
            )
            # G2-H (cute_plan.md): allocate the CLC response
            # buffer + mbarrier on the CLC path. The mbarrier-init
            # call inside the scheduler-warp body (gated on
            # lane 0 of the scheduler warp) follows in
            # ``program_id._build_scheduler_warp_role_local_while_clc``.
            # The SMEM allocations themselves are warp-uniform and
            # safe to emit at the kernel prefix.
            if tcgen05_matmul_plan.is_clc_persistent:
                prefix.extend(_emit_clc_smem_setup(tcgen05_sched_plan))
            # C-input warp aux SMEM ring + ``c_pipeline_aux``
            # ``PipelineAsync`` (``cute_plan.md`` §7.5.3.2 cycle 2
            # of the producer-body split). Fires only when the
            # productive-body gate is open: ``c_input_warp_count > 0``
            # AND a non-empty exact-shape ``c_input_aux_tensor_descriptors``
            # tuple. Broadcast row-vector aux loads intentionally stay on the
            # direct per-thread path; staging them as 2-D rings burns a full
            # epilogue tile of SMEM for a one-dimensional input. The
            # role-local while builder in
            # ``program_id._build_c_input_warp_role_local_while``
            # emits the per-descriptor producer body that issues
            # ``producer_acquire`` → cooperative
            # ``cute.copy(GMEM, SMEM_ring[stage])`` →
            # ``producer_commit`` against the same plan; the
            # consumer-side splice in
            # ``memory_ops._aux_subtile_load_source`` reads from
            # the SMEM ring under ``consumer_wait`` /
            # ``consumer_release`` gating. Gate-closed configs
            # (``c_input_warps=0`` or no aux residual) skip this
            # allocation entirely and preserve byte identity.
            # Multi-store fan-out safety gate: the productive body
            # only fires when every aux descriptor for this matmul
            # comes from a single store_value_node. With fan-out
            # (one matmul → multiple stores with different aux
            # operands), the producer would fire
            # ``producer_commit`` on every ring per subtile while
            # each store's per-store-codegen consumer covers only
            # a subset of rings — leaving the unmatched rings
            # uncommitted and deadlocking the producer once a CTA
            # wraps the pipeline depth. The ``store_value_node``
            # field on ``Tcgen05AuxTensorDescriptor`` is the
            # discriminator; the descriptor walker dedups by it
            # already, so single-store fan-out into multiple
            # writes of the same value gives one ``store_value_node``
            # in the descriptor set (and the GMEM fallback path
            # remains byte-identical to the pre-cycle-2b shape).
            c_input_aux_tensor_descriptors = (
                tcgen05_matmul_plan.c_input_aux_tensor_descriptors
            )
            all_aux_tensor_descriptors = tcgen05_matmul_plan.aux_tensor_descriptors
            tcgen05_aux_tma_requested = (
                df.config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
            )
            aux_store_value_nodes = {
                desc.store_value_node for desc in c_input_aux_tensor_descriptors
            }
            aux_single_store_value = len(aux_store_value_nodes) <= 1
            aux_productive_body_gate_open = (
                tcgen05_matmul_plan.has_c_input_warp
                and c_input_aux_tensor_descriptors
                and aux_single_store_value
            )
            if (
                tcgen05_aux_tma_requested
                and all_aux_tensor_descriptors
                and not aux_productive_body_gate_open
            ):
                if not tcgen05_matmul_plan.has_c_input_warp:
                    reason = (
                        "requires the productive C-input warp "
                        "(``tcgen05_warp_spec_c_input_warps=1``)"
                    )
                elif not c_input_aux_tensor_descriptors:
                    reason = (
                        "requires at least one exact-shape rank-2 auxiliary "
                        "tensor; broadcast-only auxiliary tensors are not "
                        "staged by the aux TMA path"
                    )
                else:
                    reason = (
                        "requires single-store fan-out for the staged aux descriptors"
                    )
                raise exc.BackendUnsupported(
                    "cute",
                    f"{TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}="
                    f"{TCGEN05_AUX_LOAD_MODE_TMA!r} {reason}",
                )
            if aux_productive_body_gate_open:
                aux_descriptor_dtype_strs = tuple(
                    env.backend.dtype_str(desc.host_tensor_val.dtype)
                    for desc in c_input_aux_tensor_descriptors
                )
                # With dynamic M/N output shapes, a TMA-store epilogue is only
                # enabled for the output-edge family, which routes partial tiles
                # through either the full/edge split or the bounds-checked
                # partial-output TMA-store path.
                tma_store_handles_partial_tiles = tcgen05_use_tma_store_epilogue
                aux_tma_needs_edge_routing = (
                    tcgen05_aux_tma_requested
                    and not tcgen05_static_output_tiles
                    and not tma_store_handles_partial_tiles
                )
                if aux_tma_needs_edge_routing:
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}="
                        f"{TCGEN05_AUX_LOAD_MODE_TMA!r} with partial output "
                        "tiles requires either a partial-output TMA-store "
                        "epilogue or the full-tile/edge fallback split used "
                        "by output-edge stores",
                    )
                if tcgen05_aux_tma_requested and any(
                    dtype_str != epi_elem_dtype_str
                    for dtype_str in aux_descriptor_dtype_strs
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        f"{TCGEN05_AUX_LOAD_MODE_CONFIG_KEY}="
                        f"{TCGEN05_AUX_LOAD_MODE_TMA!r} requires auxiliary "
                        "tensor dtype to match the epilogue/output dtype",
                    )
                tcgen05_aux_use_tma_load = tcgen05_aux_tma_requested
                tcgen05_aux_stage_count = _tcgen05_aux_pipeline_stage_count_from_config(
                    df.config
                )
                tcgen05_aux_plan = _new_tcgen05_aux_pipeline_plan(
                    df,
                    num_rings=len(c_input_aux_tensor_descriptors),
                    epi_tile_var=tcgen05_plan.epi_tile,
                    use_tma_load=tcgen05_aux_use_tma_load,
                    stage_count=tcgen05_aux_stage_count,
                )
                if tcgen05_aux_use_tma_load:
                    for desc, ring, aux_dtype_str in zip(
                        c_input_aux_tensor_descriptors,
                        tcgen05_aux_plan.rings,
                        aux_descriptor_dtype_strs,
                        strict=True,
                    ):
                        tma_atom = ring.tma_atom
                        tma_tensor = ring.tma_tensor
                        assert tma_atom is not None
                        assert tma_tensor is not None
                        aux_tensor_name = df.tensor_arg(desc.host_tensor_val).name
                        df.placeholder_args.add(aux_tensor_name)
                        df.wrapper_only_params.extend([tma_atom, tma_tensor])
                        cg.cute_wrapper_plans.append(
                            {
                                "kind": "tcgen05_aux_tma",
                                "c_name": aux_tensor_name,
                                "bm": bm,
                                "bn": bn,
                                "stage_count": tcgen05_aux_stage_count,
                                "input_dtype": aux_dtype_str,
                                "kernel_args": [tma_atom, tma_tensor],
                            }
                        )
                df.cute_state.register_tcgen05_aux_pipeline_plan(tcgen05_aux_plan)
                prefix.extend(
                    _emit_tcgen05_aux_pipeline_setup(
                        tcgen05_aux_plan,
                        descriptor_dtype_strs=aux_descriptor_dtype_strs,
                        # Each SMEM ring stage holds one ``epi_tile``
                        # worth of aux data (one subtile of the
                        # per-output-tile aux region). The producer
                        # body in
                        # ``program_id._build_c_input_warp_role_local_while``
                        # loops over the matmul's subtile axis once
                        # per output tile and cooperative-copies one
                        # epi-tile into each ring stage; the
                        # consumer's per-subtile loop waits, reads
                        # the active stage with the existing
                        # ``partition_C → flat_divide(epi_tile) →
                        # partition_D`` pipeline, then lane-0
                        # releases. Per-subtile staging reduces the
                        # epilogue SMEM footprint vs whole-tile
                        # staging, but the AB ring at ``bk=128`` plus
                        # the aux/D-store rings still overshoots the
                        # 232 KB B200 cap at ``cluster_m=2 +
                        # tcgen05_ab_stages=3`` (cycle 48 measured
                        # 263 KB used at bk=128; bk=64 fits).
                        tile_shape_expr=tcgen05_plan.epi_tile,
                        # Single C-input warp = 32 lanes (validator
                        # pins ``c_input_warp_count`` to ``{0, 1}``
                        # under WITH_SCHEDULER); all 32 lanes
                        # participate in the producer-side
                        # cooperative copy.
                        c_input_warp_thread_count=(
                            tcgen05_matmul_plan.c_input_warp_count * 32
                        ),
                        # Per-warp consumer arrive count for the
                        # lane-0-gated release — see the emitter
                        # docstring.
                        epi_warp_count=tcgen05_matmul_plan.epi_warp_count,
                        defer_sync=tcgen05_use_cluster_deferred_pipelines,
                    )
                )
        if not tcgen05_use_tma:
            _emit_tcgen05_tmem_setup()
    else:
        prefix.append(
            statement_from_string(
                f"{acc_frag} = cute.make_rmem_tensor("
                f"{tiled_mma}.partition_shape_C(({bm}, {bn})), {acc_dtype_str})"
            )
        )
    # Allocate shared memory for A and B tiles (reused across K iterations)
    # Keep these allocations in the device-loop prefix. Lane-loop MMA relies on
    # per-iteration shared-memory state; hoisting them outside the lane loops
    # regresses the existing lane-loop coverage.
    smem_a_ptr = df.new_var("smem_a")
    smem_b_ptr = df.new_var("smem_b")
    smem_a = df.new_var("sA")
    smem_b = df.new_var("sB")
    smem_a_mma = df.new_var("sA_mma")
    smem_b_mma = df.new_var("sB_mma")
    tma_smem_a_layout = df.new_var("sA_tma_layout")
    tma_smem_b_layout = df.new_var("sB_tma_layout")
    tma_thr_mma = df.new_var("tma_thr_mma")
    gmem_a_tma = df.new_var("gA_tma")
    gmem_b_tma = df.new_var("gB_tma")
    gmem_a_tma_part = df.new_var("gA_tma_part")
    gmem_b_tma_part = df.new_var("gB_tma_part")
    tma_atom_a = df.new_var("tma_atom_a")
    tma_atom_b = df.new_var("tma_atom_b")
    tma_store_atom = (
        df.new_var("tcgen05_tma_store_atom") if tcgen05_use_tma_store_epilogue else ""
    )
    tma_tensor_a = df.new_var("tma_tensor_a")
    tma_tensor_b = df.new_var("tma_tensor_b")
    tma_store_tensor = (
        df.new_var("tcgen05_tma_store_tensor") if tcgen05_use_tma_store_epilogue else ""
    )
    tma_cta_layout = df.new_var("tma_cta_layout")
    tma_a_cta_layout = df.new_var("tma_a_cta_layout")
    tma_b_cta_layout = df.new_var("tma_b_cta_layout")
    tma_a_cta_coord = df.new_var("tma_a_cta_coord")
    tma_b_cta_coord = df.new_var("tma_b_cta_coord")
    tma_gA = df.new_var("tma_gA")
    tma_sA = df.new_var("tma_sA")
    tma_gB = df.new_var("tma_gB")
    tma_sB = df.new_var("tma_sB")
    tma_initial_full_tile = df.new_var("tcgen05_tma_initial_full_tile")
    tma_initial_next_full_tile = df.new_var("tcgen05_tma_initial_next_full_tile")
    tma_full_tile = df.new_var("tcgen05_tma_full_tile")
    tma_next_full_tile = df.new_var("tcgen05_tma_next_full_tile")
    tma_next_consumer_tile = df.new_var("tcgen05_tma_next_consumer_tile")

    def _tcgen05_tma_output_tile_predicate() -> str:
        if tcgen05_role_local_double_edge_tma:
            # The role-local double-edge path lets TMA handle both partial AB
            # stripes while the SIMT epilogue predicates aux loads and stores.
            return (
                f"{m_offset_var} < cutlass.Int32({m_size}) "
                f"and {n_offset_var} < cutlass.Int32({n_size}) "
            )
        if tcgen05_role_local_m_edge_tma:
            # The role-local M-edge path lets TMA handle the partial A stripe
            # while the SIMT epilogue predicates aux loads and D stores.
            return (
                f"{m_offset_var} < cutlass.Int32({m_size}) "
                f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
            )
        if tcgen05_role_local_n_edge_tma:
            # The role-local N-edge path lets TMA handle the partial B stripe
            # while the SIMT epilogue predicates aux loads and D stores.
            return (
                f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                f"and {n_offset_var} < cutlass.Int32({n_size}) "
            )
        return (
            f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
            f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
        )

    def _tcgen05_tma_k_tile_predicate(
        *, k_tile_start_expr: str, full_tile_end_expr: str
    ) -> str:
        if tcgen05_role_local_uses_k_tail_tma:
            return f"{k_tile_start_expr} < cutlass.Int32({k_total_size})"
        return f"{full_tile_end_expr} <= cutlass.Int32({k_total_size})"

    def _tcgen05_tma_tile_predicate(
        *, k_tile_start_expr: str, full_tile_end_expr: str
    ) -> str:
        return (
            _tcgen05_tma_output_tile_predicate()
            + "and "
            + _tcgen05_tma_k_tile_predicate(
                k_tile_start_expr=k_tile_start_expr,
                full_tile_end_expr=full_tile_end_expr,
            )
        )

    tma_k_tile = df.new_var("tcgen05_tma_k_tile")
    tma_barrier_ptr = df.new_var("tcgen05_tma_barrier")
    tma_producer_try_token = df.new_var("tcgen05_ab_producer_try_token")
    tma_consumer_try_token = df.new_var("tcgen05_ab_consumer_try_token")
    tma_cta_rank_in_cluster = df.new_var("tcgen05_cta_rank_in_cluster")
    tma_block_in_cluster_coord_vmnk = df.new_var("tcgen05_block_in_cluster_coord_vmnk")
    tma_a_mcast_mask = df.new_var("tcgen05_a_mcast_mask")
    tma_b_mcast_mask = df.new_var("tcgen05_b_mcast_mask")
    tcgen05_use_tma_b_mcast_mask = False
    tma_pipeline_mbars = df.new_var("tcgen05_ab_pipeline_mbars")
    tma_pipeline_producer_group = df.new_var("tcgen05_ab_pipeline_producer_group")
    tma_pipeline_consumer_group = df.new_var("tcgen05_ab_pipeline_consumer_group")
    tma_pipeline_tx_count = df.new_var("tcgen05_ab_pipeline_tx_count")
    tma_pipeline = df.new_var("tcgen05_ab_pipeline")
    tma_producer_state = df.new_var("tcgen05_ab_producer_state")
    tma_consumer_state = df.new_var("tcgen05_ab_consumer_state")
    tma_store_role_tile_counter = (
        df.new_var("tcgen05_tma_store_role_tile")
        if tcgen05_use_tma_store_epilogue and tcgen05_use_role_local_tma_producer
        else ""
    )
    tcgen05_frag_a = df.new_var("tcgen05_tCrA")
    tcgen05_frag_b = df.new_var("tcgen05_tCrB")
    mma_stage = df.new_var("mma_stage")
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        # ``tcgen05_matmul_plan`` is initialized in the same
        # ``mma_impl == "tcgen05"`` branch upstream; the assert
        # narrows the type for pyrefly so the ``is_clc_persistent``
        # property access below doesn't trip a missing-attribute
        # check on the ``Optional[CuteTcgen05MatmulPlan]`` annotation.
        assert tcgen05_matmul_plan is not None
        if tcgen05_use_tma:
            # Applied for every tcgen05 TMA path even though only the role-local
            # path strictly needs it: TMA wrapper plans consume the original
            # tensor arguments on the host even when device DCE sees no scalar
            # fallback references to those tensors.
            df.placeholder_args.update((lhs_arg_name, rhs_arg_name))
            df.wrapper_only_params.extend(
                [tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b]
            )
            ab_tma_plan: dict[str, object] = {
                "kind": "tcgen05_ab_tma",
                "lhs_name": lhs_arg_name,
                "rhs_name": rhs_arg_name,
                "bm": bm,
                "bn": bn,
                "bk": bk,
                "cluster_m": tcgen05_cluster_m,
                "cluster_n": tcgen05_cluster_n,
                "ab_stage_count": tcgen05_ab_stage_count_value,
                "input_dtype": input_dtype_str,
                "acc_dtype": acc_dtype_str,
                # B1 (cycle-3 review): plumb the validated problem
                # shape onto the AB plan so the direct-entry plan can
                # carry an envelope identity that the runtime
                # validator dispatches on. T4 and T5 share ``bk=128``,
                # so the bk-keyed shape-set alone cannot distinguish
                # a T4 plan from a T5 plan when both are admitted.
                "m_size": m_size,
                "n_size": n_size,
                "k_total_size": k_total_size,
                "kernel_args": [tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b],
            }
            # ``smem_swizzle_*`` overrides are recorded only when codegen
            # selected an explicit SMEM atom kind (either from a user
            # override or the scalar-edge fallback workaround). Keeping
            # the keys absent on the default path preserves the legacy
            # wrapper-plan literal. The wrapper-side
            # ``_append_cute_wrapper_plan`` reads ``plan.get(..., None)``
            # and emits the override-aware SMEM atom expression only
            # when an explicit value is present.
            if tcgen05_smem_swizzle_a is not None:
                ab_tma_plan["smem_swizzle_a"] = tcgen05_smem_swizzle_a
            if tcgen05_smem_swizzle_b is not None:
                ab_tma_plan["smem_swizzle_b"] = tcgen05_smem_swizzle_b
            # G2-H (cute_plan.md): CLC kernels need PDL enabled at
            # the host launch so ``nvvm.clusterlaunchcontrol_try_cancel``
            # returns valid responses (without PDL the very first
            # ``cute.arch.clc_response`` returns ``valid=0``).
            # Threaded through the wrapper plan rather than a
            # side-channel attribute on the kernel object so the
            # launch flag's provenance is the same as the cluster
            # shape and other plan-level launch metadata.
            # ``use_pdl`` only added to the dict when True so the
            # static-path kernels' wrapper-plan literals stay
            # byte-identical to the pre-G2-H golden.
            if tcgen05_matmul_plan.is_clc_persistent:
                ab_tma_plan["use_pdl"] = True
            cg.cute_wrapper_plans.append(ab_tma_plan)
        prefix.append(
            statement_from_string(
                f"{smem_a_ptr} = cute.arch.alloc_smem("
                f"{input_dtype_str}, cute.cosize({tcgen05_plan.smem_a_layout}.outer), alignment=128)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_a} = cute.make_tensor("
                f"cute.recast_ptr({smem_a_ptr}, {tcgen05_plan.smem_a_layout}.inner, dtype={input_dtype_str}), "
                f"{tcgen05_plan.smem_a_layout}.outer)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b_ptr} = cute.arch.alloc_smem("
                f"{input_dtype_str}, cute.cosize({tcgen05_plan.smem_b_layout}.outer), alignment=128)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b} = cute.make_tensor("
                f"cute.recast_ptr({smem_b_ptr}, {tcgen05_plan.smem_b_layout}.inner, dtype={input_dtype_str}), "
                f"{tcgen05_plan.smem_b_layout}.outer)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_frag_a} = {tiled_mma}.make_fragment_A({smem_a})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_frag_b} = {tiled_mma}.make_fragment_B({smem_b})"
            )
        )
        if tcgen05_use_tma:
            if tcgen05_is_two_cta:
                assert mma_slice_linear is not None
                tma_thr_mma_slice = mma_slice_linear
            else:
                tma_thr_mma_slice = "cutlass.Int32(0)"
            prefix.append(
                statement_from_string(
                    f"{tma_smem_a_layout} = cute.slice_({tcgen05_plan.smem_a_layout}, (None, None, None, 0))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_smem_b_layout} = cute.slice_({tcgen05_plan.smem_b_layout}, (None, None, None, 0))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_thr_mma} = {tiled_mma}.get_slice({tma_thr_mma_slice})"
                )
            )
            # gA, gB depend on per-tile (m_offset_var, n_offset_var). Their
            # downstream partitions and tma_partition outputs all inherit
            # that per-tile dependency, so all of these stay inside the
            # work-tile body when the persistent loop splitter runs.
            _emit_per_tile(
                f"{gmem_a_tma} = cute.local_tile("
                f"{tma_tensor_a}, ({bm}, {bk}), "
                f"({m_offset_var} // cutlass.Int32({bm}), None))",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            _emit_per_tile(
                f"{gmem_b_tma} = cute.local_tile("
                f"{tma_tensor_b}, ({bn}, {bk}), "
                f"({n_offset_var} // cutlass.Int32({bn}), None))",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            _emit_per_tile(
                f"{gmem_a_tma_part} = {tma_thr_mma}.partition_A({gmem_a_tma})",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            _emit_per_tile(
                f"{gmem_b_tma_part} = {tma_thr_mma}.partition_B({gmem_b_tma})",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            # The guarded clustered CtaGroup.ONE bridge keeps one TMA producer
            # transaction per CTA and duplicates B locally. The B atom is still
            # a multicast TMA atom for cluster_m=2/CtaGroup.ONE, so feed it a
            # self-only mask instead of the normal all-M peers mask.
            tcgen05_use_tma_b_peer_mcast = (
                tcgen05_cluster_m > 1 or tcgen05_is_two_cta
            ) and not tcgen05_cluster_m2_one_cta_role_local_bridge
            tcgen05_use_tma_b_self_mcast = tcgen05_cluster_m2_one_cta_role_local_bridge
            tcgen05_use_tma_b_mcast_mask = (
                tcgen05_use_tma_b_peer_mcast or tcgen05_use_tma_b_self_mcast
            )
            if tcgen05_is_two_cta:
                prefix.append(
                    statement_from_string(
                        f"{tma_a_cta_layout} = cute.make_layout("
                        f"cute.slice_({tcgen05_cluster_layout_vmnk}, "
                        "(0, 0, None, 0)).shape)"
                    )
                )
                prefix.append(
                    statement_from_string(
                        f"{tma_b_cta_layout} = cute.make_layout("
                        f"cute.slice_({tcgen05_cluster_layout_vmnk}, "
                        "(0, None, 0, 0)).shape)"
                    )
                )
            else:
                prefix.append(
                    statement_from_string(f"{tma_cta_layout} = cute.make_layout(1)")
                )
            if tcgen05_cluster_m > 1 or tcgen05_is_two_cta:
                prefix.append(
                    statement_from_string(
                        f"{tma_cta_rank_in_cluster} = cute.arch.make_warp_uniform("
                        "cute.arch.block_idx_in_cluster())"
                    )
                )
                prefix.append(
                    statement_from_string(
                        f"{tma_block_in_cluster_coord_vmnk} = "
                        f"{tcgen05_cluster_layout_vmnk}.get_flat_coord({tma_cta_rank_in_cluster})"
                    )
                )
                if tcgen05_is_two_cta:
                    prefix.append(
                        statement_from_string(
                            f"{tma_a_cta_coord} = {tma_block_in_cluster_coord_vmnk}[2]"
                        )
                    )
                    prefix.append(
                        statement_from_string(
                            f"{tma_b_cta_coord} = {tma_block_in_cluster_coord_vmnk}[1]"
                        )
                    )
                if tcgen05_is_two_cta:
                    prefix.append(
                        statement_from_string(
                            f"{tma_a_mcast_mask} = cute.nvgpu.cpasync.create_tma_multicast_mask("
                            f"{tcgen05_cluster_layout_vmnk}, {tma_block_in_cluster_coord_vmnk}, "
                            "mcast_mode=2)"
                        )
                    )
                if tcgen05_use_tma_b_mcast_mask:
                    tma_b_mcast_mode = 2 if tcgen05_use_tma_b_self_mcast else 1
                    prefix.append(
                        statement_from_string(
                            f"{tma_b_mcast_mask} = cute.nvgpu.cpasync.create_tma_multicast_mask("
                            f"{tcgen05_cluster_layout_vmnk}, {tma_block_in_cluster_coord_vmnk}, "
                            f"mcast_mode={tma_b_mcast_mode})"
                        )
                    )
            # tma_partition consumes the per-tile gA_part / gB_part, so the
            # resulting (tma_sA, tma_gA) / (tma_sB, tma_gB) are also per-tile.
            tma_a_cta_coord_expr = tma_a_cta_coord if tcgen05_is_two_cta else "0"
            tma_b_cta_coord_expr = tma_b_cta_coord if tcgen05_is_two_cta else "0"
            tma_a_cta_layout_expr = (
                tma_a_cta_layout if tcgen05_is_two_cta else tma_cta_layout
            )
            tma_b_cta_layout_expr = (
                tma_b_cta_layout if tcgen05_is_two_cta else tma_cta_layout
            )
            _emit_per_tile(
                f"{tma_sA}, {tma_gA} = cute.nvgpu.cpasync.tma_partition("
                f"{tma_atom_a}, {tma_a_cta_coord_expr}, {tma_a_cta_layout_expr}, "
                f"cute.group_modes({smem_a}, 0, cute.rank({smem_a}) - 1), "
                f"cute.group_modes({gmem_a_tma_part}, 0, "
                f"cute.rank({gmem_a_tma_part}) - 1))",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            _emit_per_tile(
                f"{tma_sB}, {tma_gB} = cute.nvgpu.cpasync.tma_partition("
                f"{tma_atom_b}, {tma_b_cta_coord_expr}, {tma_b_cta_layout_expr}, "
                f"cute.group_modes({smem_b}, 0, cute.rank({smem_b}) - 1), "
                f"cute.group_modes({gmem_b_tma_part}, 0, "
                f"cute.rank({gmem_b_tma_part}) - 1))",
                tma_load=tcgen05_use_role_local_tma_producer,
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_mbars} = cute.arch.alloc_smem("
                    f"cutlass.Int64, cutlass.Int32({tcgen05_ab_stage_count_value}))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, 1)"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_consumer_group} = cutlass.pipeline.CooperativeGroup("
                    f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_ab_consumer_arrive_count_value}))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_tx_count} = "
                    f"({'cute.size_in_bytes(' + input_dtype_str + ', ' + tma_smem_a_layout + ')' if tcgen05_use_tma_a else '0'} + "
                    f"{'cute.size_in_bytes(' + input_dtype_str + ', ' + tma_smem_b_layout + ')' if tcgen05_use_tma_b else '0'})"
                    + (
                        f" * cute.size({tiled_mma}.thr_id.shape)"
                        if tcgen05_is_two_cta
                        else ""
                    )
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline} = cutlass.pipeline.PipelineTmaUmma.create("
                    f"num_stages={tcgen05_ab_stage_count_value}, "
                    f"producer_group={tma_pipeline_producer_group}, "
                    f"consumer_group={tma_pipeline_consumer_group}, "
                    f"tx_count={tma_pipeline_tx_count}, "
                    f"barrier_storage={tma_pipeline_mbars}, "
                    f"cta_layout_vmnk={tcgen05_cluster_layout_vmnk}"
                    f"{tcgen05_defer_pipeline_sync_arg})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_producer_state} = {tcgen05_pipeline_state_ns}.make_pipeline_state("
                    f"cutlass.pipeline.PipelineUserType.Producer, {tcgen05_ab_stage_count_value})"
                )
            )
            tma_consumer_state_init = (
                # Diagnostic phase override intentionally uses the upstream
                # raw state constructor; it does not participate in the Helion
                # wrapper ownership experiment.
                f"cutlass.pipeline.PipelineState({tcgen05_ab_stage_count_value}, "
                "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(1))"
                if diagnose_ab_consumer_phase1
                else (
                    f"{tcgen05_pipeline_state_ns}.make_pipeline_state("
                    "cutlass.pipeline.PipelineUserType.Consumer, "
                    f"{tcgen05_ab_stage_count_value})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_consumer_state} = {tma_consumer_state_init}"
                )
            )
            _emit_tcgen05_tmem_setup()
            if tcgen05_use_tma_pipeline:
                # Initial TMA prefetch warms stages 0..ab_stage_count-1 of the
                # AB pipeline at the START of each tile. Both the boolean
                # full-tile predicates and the TMA copies reference per-tile
                # gA/gB tensors and m_offset/n_offset, so they must stay in
                # the work-tile body.
                #
                # In the role-local producer path, the TMA-load warp needs
                # its own per-tile tensor partitions and full-tile predicates
                # because it no longer runs the shared work-tile loop. Tag
                # those prerequisites together with the prefetch IFs so the
                # partitioner extracts one self-contained TMA-load role body.
                assert tma_warp is not None
                prefetch_args = _InitialPrefetchTmaArgs(
                    tma_pipeline=tma_pipeline,
                    tma_producer_state=tma_producer_state,
                    tma_barrier_ptr=tma_barrier_ptr,
                    tma_warp=tma_warp,
                    tma_atom_a=tma_atom_a,
                    tma_atom_b=tma_atom_b,
                    tma_gA=tma_gA,
                    tma_gB=tma_gB,
                    tma_sA=tma_sA,
                    tma_sB=tma_sB,
                    tma_a_mcast_mask=tma_a_mcast_mask,
                    tma_b_mcast_mask=tma_b_mcast_mask,
                    is_two_cta=tcgen05_is_two_cta,
                    use_tma_b_mcast_mask=tcgen05_use_tma_b_mcast_mask,
                    skip_producer_acquire=diagnose_skip_ab_producer_acquire,
                    skip_producer_advance=diagnose_skip_ab_producer_advance,
                )
                _emit_per_tile(
                    f"{tma_initial_full_tile} = "
                    + _tcgen05_tma_tile_predicate(
                        k_tile_start_expr="cutlass.Int32(0)",
                        full_tile_end_expr=f"cutlass.Int32({bk})",
                    ),
                    tma_load=tcgen05_use_role_local_tma_producer,
                )
                stage0_prefetch = _build_initial_prefetch_if(
                    prefetch_args,
                    full_tile_gates=[tma_initial_full_tile],
                    k_offset="cutlass.Int32(0)",
                    skip_producer_acquire=(
                        diagnose_skip_ab_producer_acquire
                        or diagnose_skip_initial_ab_producer_acquire
                    ),
                )
                prefix.append(stage0_prefetch)
                per_tile_stmts.append(stage0_prefetch)
                if tcgen05_use_role_local_tma_producer:
                    tma_load_role_stmts.append(stage0_prefetch)
                if tcgen05_ab_stage_count_value > 1:
                    # Warm every stage 1..ab_stage_count-1; each gated by
                    # an ``i+1``-k_tile fits-in-K predicate. The old
                    # two-call pattern only covered stages 0 and N-1
                    # (sufficient for ab=2 where they're the same set);
                    # ab>=3 leaves intermediate stages unarmed and the
                    # consumer ``consumer_wait`` deadlocks on stage 1
                    # phase 0. See cute_plan.md §6.9.1.
                    _emit_per_tile(
                        f"{tma_initial_next_full_tile} = "
                        + _tcgen05_tma_tile_predicate(
                            k_tile_start_expr=f"cutlass.Int32({bk * (tcgen05_ab_stage_count_value - 1)})",
                            full_tile_end_expr=f"cutlass.Int32({bk * tcgen05_ab_stage_count_value})",
                        ),
                        tma_load=tcgen05_use_role_local_tma_producer,
                    )
                    for stage_idx in range(1, tcgen05_ab_stage_count_value):
                        if stage_idx == tcgen05_ab_stage_count_value - 1:
                            stage_gates = [
                                tma_initial_full_tile,
                                tma_initial_next_full_tile,
                            ]
                        else:
                            stage_gate_var = df.new_var(
                                f"tcgen05_tma_initial_stage_{stage_idx}_full_tile"
                            )
                            _emit_per_tile(
                                f"{stage_gate_var} = "
                                + _tcgen05_tma_tile_predicate(
                                    k_tile_start_expr=f"cutlass.Int32({bk * stage_idx})",
                                    full_tile_end_expr=f"cutlass.Int32({bk * (stage_idx + 1)})",
                                ),
                                tma_load=tcgen05_use_role_local_tma_producer,
                            )
                            stage_gates = [tma_initial_full_tile, stage_gate_var]
                        stage_prefetch = _build_initial_prefetch_if(
                            prefetch_args,
                            full_tile_gates=stage_gates,
                            k_offset=f"cutlass.Int32({stage_idx})",
                        )
                        prefix.append(stage_prefetch)
                        per_tile_stmts.append(stage_prefetch)
                        if tcgen05_use_role_local_tma_producer:
                            tma_load_role_stmts.append(stage_prefetch)
    else:
        prefix.append(
            statement_from_string(
                f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bm * bk})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_a} = cute.make_tensor("
                f"{smem_a_ptr}, cute.make_layout(({bm}, {bk}), stride=({bk}, 1)))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bn * bk})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b} = cute.make_tensor("
                f"{smem_b_ptr}, cute.make_layout(({bn}, {bk}), stride=({bk}, 1)))"
            )
        )
    # === loop body: global → smem → register → gemm ===
    rA = df.new_var("rA")
    rB = df.new_var("rB")
    tAsA = df.new_var("tAsA")
    tBsB = df.new_var("tBsB")
    # Built once below in the tcgen05+TMA branch; reused by the
    # release block emitted later in the same branch.
    tma_kloop_args: _PerKiterTmaArgs | None = None

    # --- Global → Shared memory with masking ---
    # Each thread loads elements into shared memory using scalar indexing
    # with bounds checking for non-divisible tile boundaries.
    if acc_expr is None and mma_impl == "universal":
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == {k_loop_begin_expr}:\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
            )
        )
    elif acc_expr is not None and mma_impl == "universal":
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == {k_loop_begin_expr}:\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}({{acc}})",
                acc=acc_expr,
            )
        )
    elif acc_expr is None:
        assert mma_active is not None
        if mma_impl == "warp":
            cg.add_statement(
                statement_from_string(
                    f"if {mma_active} and {k_offset_var} == {k_loop_begin_expr}:\n"
                    f"    for _mma_i in range(cute.size({acc_frag})):\n"
                    f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
                )
            )
    else:
        raise AssertionError("non-universal MMA with acc_expr should fall back")
    if mma_impl == "universal":
        # Guards select the hardware thread that loads each row/column of
        # the A/B SMEM cache. Use the *physical* thread coord (not the
        # lane-aware local coord) so the same hardware threads load on
        # every iteration of an outer ``for lane_<n> in range(epT):`` loop
        # when ``elements_per_thread > 1``. ``n_local == 0`` only matches
        # ``(thread_y, lane) == (0, 0)`` — fine for the no-lane case but
        # leaves sA stale on ``lane > 0`` iterations because the guard
        # never fires. ``n_physical == 0`` matches ``thread_y == 0`` on
        # every lane iteration so sA is re-populated for the current K
        # tile. The store target is still ``sA[m_local, _k]`` so different
        # lane iterations naturally write to different rows of sA / sB if
        # the m-axis has its own lane var.
        #
        # Local invariant: because the K loop is nested INSIDE the
        # lane loop in the current scheduler, skipping the A/B load
        # on lane>0 iterations would reuse the previous K tile's sA
        # (overwritten by the previous K iteration) — incorrect.
        # Deferred hoist would require K/lane interchange. See
        # cute_plan.md for the deferred-restructure paths.
        cg.add_statement(
            statement_from_string(
                f"if {n_physical} == cutlass.Int32(0):\n"
                f"    for _k in range({bk}):\n"
                f"        _gk = {k_offset_var} + cutlass.Int32(_k)\n"
                f"        {smem_a}[{m_local}, cutlass.Int32(_k)] = ("
                f"{lhs_arg_name}[{m_global}, _gk] "
                f"if {m_global} < cutlass.Int32({m_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"if {m_physical} == cutlass.Int32(0):\n"
                f"    for _k in range({bk}):\n"
                f"        _gk = {k_offset_var} + cutlass.Int32(_k)\n"
                f"        {smem_b}[{n_local}, cutlass.Int32(_k)] = ("
                f"{rhs_arg_name}[_gk, {n_global}] "
                f"if {n_global} < cutlass.Int32({n_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )
        cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    else:
        active_threads = bm * mma_phys_n
        assert (
            mma_active is not None
            and mma_participant_linear is not None
            and mma_copy_linear is not None
        )
        load_thread_count = (
            mma_physical_m_threads * mma_phys_n
            if mma_impl == "tcgen05" and tcgen05_collective_handles_operand_loads
            else active_threads
        )
        load_guard = mma_active
        mma_stage_stmt: ast.stmt | None = None
        smem_a_mma_stmt: ast.stmt | None = None
        smem_b_mma_stmt: ast.stmt | None = None
        tma_full_tile_predicate_src: str | None = None
        tma_full_tile_stmt: ast.stmt | None = None
        if mma_impl == "tcgen05":
            assert tcgen05_plan is not None
            # The smem cache for A/B is laid out as (..., ab_stage_count); we
            # index into the current stage every K-loop iteration.
            #
            # When the role-local persistent kernel uses the TMA pipeline,
            # ``tma_consumer_state.index`` is the canonical stage index: it
            # advances exactly once per K-loop iteration via ``consumer_release``
            # + ``advance`` and carries its value across virtual tiles. Computing
            # ``mma_stage`` from ``k_offset // bk`` resets to zero at each tile
            # while the pipeline state stays where it was at the end of the
            # prior tile -- the two diverge across persistent tile boundaries.
            #
            # Non-role-local edge fallback is different: scalar fallback loader
            # warps also need the stage, but only the exec warp advances its
            # thread-local AB consumer state on full TMA K tiles. Compute the
            # stage from the K tile index there so loader and exec warps agree
            # when a later partial K tile falls back to scalar SMEM fills.
            #
            # For the non-TMA tcgen05 path there is no pipeline state to track
            # and ``ab_stage_count`` is always 1, so the modular form is a
            # constant zero anyway.
            if tcgen05_use_tma and tcgen05_use_separate_mma_exec:
                mma_stage_stmt = statement_from_string(
                    f"{mma_stage} = {tma_consumer_state}.index"
                )
            else:
                mma_stage_stmt = statement_from_string(
                    f"{mma_stage} = "
                    f"({k_offset_var} // cutlass.Int32({bk})) "
                    f"% cutlass.Int32({tcgen05_ab_stage_count_value})"
                )
            smem_a_mma_stmt = statement_from_string(
                f"{smem_a_mma} = {smem_a}[(None, 0, 0, {mma_stage})]"
            )
            smem_b_mma_stmt = statement_from_string(
                f"{smem_b_mma} = {smem_b}[(None, 0, 0, {mma_stage})]"
            )
            if not tcgen05_use_separate_mma_exec:
                cg.add_statement(mma_stage_stmt)
                cg.add_statement(smem_a_mma_stmt)
                cg.add_statement(smem_b_mma_stmt)
            if tcgen05_use_tma:
                tma_k_tile_stmt = statement_from_string(
                    f"{tma_k_tile} = {k_offset_var} // cutlass.Int32({bk})"
                )
                tma_full_tile_predicate_src = _tcgen05_tma_tile_predicate(
                    k_tile_start_expr=k_offset_var,
                    full_tile_end_expr=f"{k_offset_var} + cutlass.Int32({bk})",
                )
                tma_full_tile_stmt = statement_from_string(
                    f"{tma_full_tile} = " + tma_full_tile_predicate_src
                )
                if not tcgen05_use_separate_mma_exec:
                    cg.add_statement(tma_k_tile_stmt)
                    if not tcgen05_static_full_tma_fast_path:
                        cg.add_statement(tma_full_tile_stmt)
        smem_a_store = f"{smem_a}[_row, _col]"
        smem_b_store = f"{smem_b}[_row, _col]"
        if mma_impl == "tcgen05":
            smem_a_store = f"{smem_a_mma}[((_row, _col),)]"
            smem_b_store = f"{smem_b_mma}[((_row, _col),)]"
        scalar_load_a = statement_from_string(
            f"if {load_guard}:\n"
            f"    for _load_i in range(({bm * bk} + {load_thread_count} - 1) // {load_thread_count}):\n"
            f"        _flat = {mma_copy_linear} + cutlass.Int32(_load_i) * cutlass.Int32({load_thread_count})\n"
            f"        if _flat < cutlass.Int32({bm * bk}):\n"
            f"            _row = _flat // cutlass.Int32({bk})\n"
            f"            _col = _flat % cutlass.Int32({bk})\n"
            f"            _gm = {m_offset_var} + _row\n"
            f"            _gk = {k_offset_var} + _col\n"
            f"            {smem_a_store} = ("
            f"{lhs_arg_name}[_gm, _gk] "
            f"if _gm < cutlass.Int32({m_size}) "
            f"and _gk < cutlass.Int32({k_total_size}) "
            f"else {input_dtype_str}(0.0))"
        )
        scalar_load_b = statement_from_string(
            f"if {load_guard}:\n"
            f"    for _load_i in range(({bn * bk} + {load_thread_count} - 1) // {load_thread_count}):\n"
            f"        _flat = {mma_copy_linear} + cutlass.Int32(_load_i) * cutlass.Int32({load_thread_count})\n"
            f"        if _flat < cutlass.Int32({bn * bk}):\n"
            f"            _row = _flat // cutlass.Int32({bk})\n"
            f"            _col = _flat % cutlass.Int32({bk})\n"
            f"            _gn = {n_offset_var} + _row\n"
            f"            _gk = {k_offset_var} + _col\n"
            f"            {smem_b_store} = ("
            f"{rhs_arg_name}[_gk, _gn] "
            f"if _gn < cutlass.Int32({n_size}) "
            f"and _gk < cutlass.Int32({k_total_size}) "
            f"else {input_dtype_str}(0.0))"
        )
        if mma_impl == "tcgen05" and tcgen05_use_tma:
            assert tcgen05_plan is not None
            assert tma_warp is not None
            # The validated explicit two-CTA store-box family uses bk=64 to
            # match Quack's four logical 2CTA MMA K blocks. Python ``range``
            # lets the outer K-loop duplicate those issue sites; a CuTe range
            # with no-unroll metadata keeps that guarded family in one loop
            # body while normal loops keep their existing range lowering.
            tcgen05_use_nounroll_k_loop = (
                any(
                    value is not None
                    for value in (
                        tcgen05_explicit_epi_tile_m,
                        tcgen05_explicit_epi_tile_n,
                        tcgen05_explicit_d_store_box_n,
                    )
                )
                and tcgen05_static_full_tiles
                and tcgen05_is_two_cta
                and bk == 64
            )
            tma_kloop_args = _PerKiterTmaArgs(
                tma_pipeline=tma_pipeline,
                tma_producer_state=tma_producer_state,
                tma_consumer_state=tma_consumer_state,
                tma_producer_try_token=tma_producer_try_token,
                tma_consumer_try_token=tma_consumer_try_token,
                tma_barrier_ptr=tma_barrier_ptr,
                tma_full_tile=tma_full_tile,
                tma_next_full_tile=tma_next_full_tile,
                tma_next_consumer_tile=tma_next_consumer_tile,
                tma_warp=tma_warp,
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_gA=tma_gA,
                tma_gB=tma_gB,
                tma_sA=tma_sA,
                tma_sB=tma_sB,
                tma_k_tile=tma_k_tile,
                tma_a_mcast_mask=tma_a_mcast_mask,
                tma_b_mcast_mask=tma_b_mcast_mask,
                ab_stage_count=tcgen05_ab_stage_count_value,
                is_two_cta=tcgen05_is_two_cta,
                use_tma_b_mcast_mask=tcgen05_use_tma_b_mcast_mask,
                use_tma_a=tcgen05_use_tma_a,
                use_tma_b=tcgen05_use_tma_b,
                skip_producer_acquire=diagnose_skip_ab_producer_acquire,
                skip_producer_advance=diagnose_skip_ab_producer_advance,
                skip_consumer_wait=diagnose_skip_ab_consumer_wait,
                exec_active=tcgen05_plan.exec_active,
                scalar_load_a=scalar_load_a,
                scalar_load_b=scalar_load_b,
                cluster_n=tcgen05_cluster_n,
                static_full_tiles=tcgen05_static_full_tma_fast_path,
            )
            if tcgen05_use_tma_pipeline:
                if tcgen05_use_separate_tma_producer:
                    producer_loop_body = [
                        statement_from_string(
                            f"{tma_k_tile} = {k_offset_var} // cutlass.Int32({bk})"
                        )
                    ]
                    if not tcgen05_static_full_tma_fast_path:
                        assert tma_full_tile_predicate_src is not None
                        producer_loop_body.append(
                            statement_from_string(
                                f"{tma_full_tile} = " + tma_full_tile_predicate_src
                            )
                        )
                    producer_loop_body.extend(
                        [
                            statement_from_string(
                                f"{tma_next_full_tile} = "
                                + _tcgen05_tma_tile_predicate(
                                    k_tile_start_expr=f"{k_offset_var} + cutlass.Int32({bk * tcgen05_ab_stage_count_value})",
                                    full_tile_end_expr=f"{k_offset_var} + cutlass.Int32({bk * (tcgen05_ab_stage_count_value + 1)})",
                                )
                            ),
                            statement_from_string(
                                f"{tma_producer_try_token} = cutlass.Boolean(0)"
                            ),
                            _build_kloop_pipeline_producer_if(
                                tma_kloop_args, gate_tma_warp=False
                            ),
                        ]
                    )
                    producer_loop = _clone_k_loop_with_body(
                        device_loop,
                        producer_loop_body,
                        iter_expr=(
                            _tcgen05_k_loop_nounroll_iter_expr(device_loop)
                            if tcgen05_use_nounroll_k_loop
                            else None
                        ),
                    )
                    producer_stmt: ast.stmt = producer_loop
                    if tcgen05_use_pure_matmul_role_lifecycle:
                        # Pure lifecycle emits role bodies directly instead of
                        # relying on the generic role-local partitioner.
                        producer_stmt = _wrap_stmt_in_if(producer_loop, tma_warp)
                    prefix.append(producer_stmt)
                    per_tile_stmts.append(producer_stmt)
                    if tcgen05_use_role_local_tma_producer:
                        tma_load_role_stmts.append(producer_loop)
                if tcgen05_use_separate_mma_exec:
                    assert mma_stage_stmt is not None
                    assert smem_a_mma_stmt is not None
                    assert smem_b_mma_stmt is not None
                    exec_loop_body: list[ast.stmt] = [
                        mma_stage_stmt,
                        smem_a_mma_stmt,
                        smem_b_mma_stmt,
                    ]
                    if not tcgen05_static_full_tma_fast_path:
                        assert tma_full_tile_stmt is not None
                        exec_loop_body.append(tma_full_tile_stmt)
                    if tcgen05_use_role_local_ab_consumer_prefetch:
                        exec_loop_body.append(
                            statement_from_string(
                                f"{tma_next_consumer_tile} = "
                                + _tcgen05_tma_k_tile_predicate(
                                    k_tile_start_expr=f"{k_offset_var} + cutlass.Int32({bk})",
                                    full_tile_end_expr=f"{k_offset_var} + cutlass.Int32({bk * 2})",
                                )
                            )
                        )
                    else:
                        exec_loop_body.append(
                            statement_from_string(
                                f"{tma_consumer_try_token} = cutlass.Boolean(0)"
                            )
                        )
                    exec_loop_body.append(
                        _build_kloop_pipeline_consumer_if(
                            tma_kloop_args,
                            # Static-full pipeline builders require their
                            # internal exec gate to keep emitting a single
                            # statement. The outer wrapper below makes the
                            # cloned loop's role ownership explicit.
                            gate_exec_warp=tcgen05_static_full_tma_fast_path,
                            include_scalar_fallback=False,
                            use_existing_try_token=tcgen05_use_role_local_ab_consumer_prefetch,
                        )
                    )
                    if not diagnose_skip_umma_issue:
                        # The AB pipeline's ``consumer_wait`` is a
                        # transaction-count ``mbarrier_try_wait`` that already
                        # orders the TMA shared stores before the UMMA load,
                        # so an extra ``fence_view_async_shared()`` is
                        # redundant on this pipelined path. See cute_plan.md
                        # §6.9.2 for the cycle's bench/NCU write-up.
                        exec_loop_body.append(
                            _build_tcgen05_mma_issue_stmt(
                                exec_active=tcgen05_plan.exec_active,
                                tiled_mma=tiled_mma,
                                acc_frag=acc_frag,
                                tcgen05_frag_a=tcgen05_frag_a,
                                tcgen05_frag_b=tcgen05_frag_b,
                                mma_stage=mma_stage,
                                # See the consumer wait comment above: pure
                                # lifecycle has an outer exec-active role
                                # wrapper plus the static-full builder gate.
                                gate_exec_warp=tcgen05_static_full_tma_fast_path,
                                is_two_cta=tcgen05_is_two_cta,
                                cluster_n=tcgen05_cluster_n,
                            )
                        )
                    exec_loop_body.append(
                        _build_kloop_pipeline_release_if(
                            tma_kloop_args,
                            # See the consumer wait comment above.
                            gate_exec_warp=tcgen05_static_full_tma_fast_path,
                            include_scalar_fallback=False,
                        )
                    )
                    if tcgen05_use_role_local_ab_consumer_prefetch:
                        exec_loop_body.extend(
                            _build_kloop_pipeline_consumer_prefetch_stmts(
                                tma_kloop_args,
                                gate_exec_warp=False,
                            )
                        )
                    exec_loop = _clone_k_loop_with_body(
                        device_loop,
                        exec_loop_body,
                        iter_expr=(
                            _tcgen05_k_loop_nounroll_iter_expr(device_loop)
                            if tcgen05_use_nounroll_k_loop
                            else None
                        ),
                    )
                    exec_stmt: ast.stmt = exec_loop
                    if tcgen05_use_pure_matmul_role_lifecycle:
                        exec_stmt = _wrap_stmt_in_if(
                            exec_loop, tcgen05_plan.exec_active
                        )
                    prefix.append(exec_stmt)
                    per_tile_stmts.append(exec_stmt)
                    if tcgen05_use_role_local_mma_exec:
                        mma_exec_role_stmts.append(exec_loop)
                else:
                    cg.add_statement(
                        statement_from_string(
                            f"{tma_next_full_tile} = "
                            f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                            f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
                            f"and {k_offset_var} + cutlass.Int32({bk * (tcgen05_ab_stage_count_value + 1)}) <= cutlass.Int32({k_total_size})"
                        )
                    )
                    cg.add_statement(
                        statement_from_string(
                            f"{tma_producer_try_token} = cutlass.Boolean(0)"
                        )
                    )
                    cg.add_statement(
                        statement_from_string(
                            f"{tma_consumer_try_token} = cutlass.Boolean(0)"
                        )
                    )
                    if not tcgen05_use_separate_tma_producer:
                        # Legacy inline path: keep producer and consumer
                        # adjacent inside the shared K-loop. Persistent
                        # role-local mode emits the producer as a top-level
                        # sibling loop above so the TMA-load role can extract
                        # it wholesale.
                        pipeline_producer_stmt = _build_kloop_pipeline_producer_if(
                            tma_kloop_args
                        )
                        cg.add_statement(pipeline_producer_stmt)
                    cg.add_statement(
                        _build_kloop_pipeline_consumer_if(
                            tma_kloop_args,
                            include_scalar_fallback=(
                                not tcgen05_static_full_tma_fast_path
                            ),
                            sync_before_scalar_fallback=(
                                tcgen05_sync_before_scalar_fallback
                                and not tcgen05_static_full_tma_fast_path
                            ),
                        )
                    )
            else:
                non_pipeline_producer_stmt = _build_kloop_non_pipeline_producer_if(
                    tma_kloop_args
                )
                cg.add_statement(non_pipeline_producer_stmt)
                cg.add_statement(_build_kloop_non_pipeline_consumer_if(tma_kloop_args))
        else:
            cg.add_statement(scalar_load_a)
            cg.add_statement(scalar_load_b)
            cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    # --- Shared → Register with f16→f32 cast ---
    if mma_impl == "universal":
        cg.add_statement(
            statement_from_string(f"{tAsA} = {thr_mma}.partition_A({smem_a})")
        )
        cg.add_statement(
            statement_from_string(f"{tBsB} = {thr_mma}.partition_B({smem_b})")
        )
        cg.add_statement(
            statement_from_string(
                f"{rA} = cute.make_fragment_like({tAsA}, {acc_dtype_str})"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"{rB} = cute.make_fragment_like({tBsB}, {acc_dtype_str})"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"for _mma_i in range(cute.size({rA})):\n"
                f"    {rA}[_mma_i] = {acc_dtype_str}({tAsA}[_mma_i])"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"for _mma_i in range(cute.size({rB})):\n"
                f"    {rB}[_mma_i] = {acc_dtype_str}({tBsB}[_mma_i])"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})"
            )
        )
    else:
        assert mma_active is not None
        if mma_impl == "warp":
            cg.add_statement(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tAsA} = {thr_mma}.partition_A({smem_a})\n"
                    f"    {tBsB} = {thr_mma}.partition_B({smem_b})\n"
                    f"    {rA} = cute.make_fragment_like({tAsA}, {input_dtype_str})\n"
                    f"    {rB} = cute.make_fragment_like({tBsB}, {input_dtype_str})\n"
                    f"    for _mma_i in range(cute.size({rA})):\n"
                    f"        {rA}[_mma_i] = {tAsA}[_mma_i]\n"
                    f"    for _mma_i in range(cute.size({rB})):\n"
                    f"        {rB}[_mma_i] = {tBsB}[_mma_i]\n"
                    f"    cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})"
                )
            )
        else:
            assert tcgen05_plan is not None
            if not tcgen05_use_separate_mma_exec:
                if not diagnose_skip_umma_issue:
                    # No async-shared fence: the pipelined AB consumer_wait
                    # is a transaction-count ``mbarrier_try_wait`` and the
                    # non-pipelined branch follows ``consumer_wait`` with a
                    # CTA-wide ``sync_threads()`` (see
                    # ``_build_kloop_non_pipeline_consumer_if``); both
                    # already order the TMA shared stores before the UMMA
                    # load. Mirrors the role-local path above.
                    cg.add_statement(
                        _build_tcgen05_mma_issue_stmt(
                            exec_active=tcgen05_plan.exec_active,
                            tiled_mma=tiled_mma,
                            acc_frag=acc_frag,
                            tcgen05_frag_a=tcgen05_frag_a,
                            tcgen05_frag_b=tcgen05_frag_b,
                            mma_stage=mma_stage,
                            is_two_cta=tcgen05_is_two_cta,
                            cluster_n=tcgen05_cluster_n,
                        )
                    )
                if tcgen05_use_tma:
                    assert tma_kloop_args is not None
                    if tcgen05_use_tma_pipeline:
                        cg.add_statement(
                            _build_kloop_pipeline_release_if(
                                tma_kloop_args,
                                include_scalar_fallback=(
                                    not tcgen05_static_full_tma_fast_path
                                ),
                            )
                        )
                    else:
                        cg.add_statement(
                            _build_kloop_non_pipeline_release_if(tma_kloop_args)
                        )
                else:
                    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    # === outer_suffix: convert fragment → per-thread scalar ===
    # Allocate smem_c in outer_prefix so all smem is allocated at the same
    # scope level (CuTe DSL assigns static smem offsets per scope). Only the
    # `universal` and `warp` MMA paths still need the staged smem_c buffer;
    # tcgen05 epilogues are handled by `_codegen_cute_store_tcgen05_tile`
    # and skip the older generic smem_c allocation.
    smem_c_ptr = df.new_var("smem_c")
    smem_c = df.new_var("smem_c_t")
    tCsC = df.new_var("tCsC")
    result_var = df.new_var("mma_result")

    tile_numel = bm * bn
    if mma_impl != "tcgen05":
        prefix.append(
            statement_from_string(
                f"{smem_c_ptr} = cute.arch.alloc_smem({acc_dtype_str}, {tile_numel}, alignment=128)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_c} = cute.make_tensor("
                f"{smem_c_ptr}, cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))"
            )
        )
    if mma_impl == "universal":
        suffix.append(
            statement_from_string(f"{tCsC} = {thr_mma}.partition_C({smem_c})")
        )
        suffix.append(
            statement_from_string(
                f"for _mma_i in range(cute.size({tCsC})):\n"
                f"    {tCsC}[_mma_i] = {acc_frag}[_mma_i]"
            )
        )
    else:
        assert mma_active is not None
        if mma_impl == "warp":
            suffix.append(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tCsC} = {thr_mma}.partition_C({smem_c})\n"
                    f"    for _mma_i in range(cute.size({tCsC})):\n"
                    f"        {tCsC}[_mma_i] = {acc_frag}[_mma_i]"
                )
            )
            suffix.append(statement_from_string("cute.arch.sync_threads()"))
        else:
            assert tcgen05_plan is not None
            assert tcgen05_mma_owner_active is not None
            assert epi_active is not None
            assert epi_tidx is not None
            # The K-loop suffix's `acc_pipeline.producer_commit` +
            # `acc_producer_state.advance()` must run ONCE PER OUTPUT TILE.
            # In the persistent path, the splitter walks top-level
            # statements and only marks them per-tile if they read or
            # write a name that's already per-tile. These suffix
            # statements only mutate ``acc_producer_state`` via a method
            # call (no AST-visible write) and reference no per-tile
            # name directly, so without explicit tagging they get hoisted
            # out of the work-tile loop -- which means the SECOND tile
            # never commits its accumulator and the consumer-side
            # ``consumer_wait`` deadlocks (or the data is silently wrong
            # if no deadlock fires). Tag them per-tile via
            # ``_emit_per_tile_suffix`` so they stay inside the work-tile
            # loop.
            suffix_stmt = statement_from_string(
                f"if {tcgen05_mma_owner_active}:\n"
                f"    {tcgen05_plan.acc_pipeline}.producer_commit({tcgen05_plan.acc_producer_state})"
            )
            suffix.append(suffix_stmt)
            per_tile_stmts.append(suffix_stmt)
            if tcgen05_use_role_local_mma_exec:
                mma_exec_role_stmts.append(suffix_stmt)
            # Bridge-only invalid-output diagnostic: preserve producer_commit
            # while removing only the acc producer PipelineState advance edge.
            if not diagnose_skip_acc_producer_advance:
                advance_stmt = statement_from_string(
                    emit_pipeline_advance(tcgen05_plan.acc_producer_state)
                )
                suffix.append(advance_stmt)
                per_tile_stmts.append(advance_stmt)
                if tcgen05_use_role_local_mma_exec:
                    mma_exec_role_stmts.append(advance_stmt)
            # The tcgen05 epilogue + allocator teardown is emitted by
            # `_codegen_cute_store_tcgen05_tile` when the kernel stores
            # `out[tile_m, tile_n] = result`. Static-full flat and validated
            # role-local persistent kernels, including CtaGroup.TWO, take the
            # TMA-store path; partial/unsupported fallbacks keep SIMT.
            sync_stmt = statement_from_string("cute.arch.sync_threads()")
            suffix.append(sync_stmt)
            per_tile_stmts.append(sync_stmt)

    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        assert epi_tidx is not None
        assert epi_active is not None
        assert tma_warp is not None
        assert warp_idx is not None
        tcgen05_lifecycle_context = Tcgen05LifecycleContext(
            exec_active=tcgen05_plan.exec_active,
            epi_active=epi_active,
            tma_warp=tma_warp,
            tma_pipeline=tma_pipeline,
            tma_producer_state=tma_producer_state,
            acc_pipeline=tcgen05_plan.acc_pipeline,
            acc_producer_state=tcgen05_plan.acc_producer_state,
            acc_consumer_state=tcgen05_plan.acc_consumer_state,
            tmem_alloc_barrier=tcgen05_plan.tmem_alloc_barrier,
            tmem_allocator=tcgen05_plan.tmem_allocator,
            tmem_holding_buf=tcgen05_plan.tmem_holding_buf,
            tmem_dealloc_mbar_ptr=tcgen05_plan.tmem_dealloc_mbar_ptr,
            epi_acc_tmem_ptr=tcgen05_epi_acc_tmem_ptr,
            acc_tmem_cols=tcgen05_plan.acc_tmem_cols,
            is_two_cta=tcgen05_is_two_cta,
            use_tma=tcgen05_use_tma,
            ab_stage_count=tcgen05_ab_stage_count_value,
            acc_stage_count=tcgen05_acc_stage_count_value,
            skip_ab_producer_advance=diagnose_skip_ab_producer_advance,
        )
        tcgen05_pure_matmul_object = (
            Tcgen05PureMatmulObjectModel(
                lifecycle_context=tcgen05_lifecycle_context,
                cleanup_loop=device_loop,
            )
            if tcgen05_use_pure_matmul_role_lifecycle
            else None
        )
        df.cute_state.register_tcgen05_store_value(
            result_var,
            CuteTcgen05StoreValue(
                lifecycle_context=tcgen05_lifecycle_context,
                pure_matmul_object=tcgen05_pure_matmul_object,
                bm=bm,
                bn=bn,
                bk=bk,
                thr_mma=thr_mma,
                epi_warp_count=tcgen05_epi_warp_count_value,
                epi_acc_frag_base=tcgen05_epi_acc_frag_base,
                epi_tidx=epi_tidx,
                warp_idx=warp_idx,
                epi_tile=tcgen05_plan.epi_tile,
                c_stage_count=tcgen05_c_stage_count_value,
                epilog_sync_barrier_id=_TCGEN05_EPILOG_SYNC_BARRIER_ID,
                tmem_load_atom=tcgen05_plan.tmem_load_atom,
                epilogue_rest_mode=tcgen05_plan.epilogue_rest_mode,
                tma_store_atom=tma_store_atom,
                tma_store_tensor=tma_store_tensor,
                role_local_tile_counter=tma_store_role_tile_counter,
                use_role_local_epi=tcgen05_use_role_local_epi,
                use_tma_store_epilogue=tcgen05_use_tma_store_epilogue,
                tma_store_full_tiles_only=tcgen05_tma_store_full_tiles_only,
                partial_output_tma_store=tcgen05_partial_output_tma_store,
                # Mirror the value passed to `_make_tcgen05_layout_plan_setup`
                # above; the store path compares this against `target_dtype`
                # to enforce the kernel/store equality contract on
                # `compute_epilogue_tile_shape`'s dtype kwargs.
                epi_elem_dtype_str=epi_elem_dtype_str,
                explicit_epi_tile_m=tcgen05_explicit_epi_tile_m,
                explicit_epi_tile_n=tcgen05_explicit_epi_tile_n,
                explicit_d_store_box_n=tcgen05_explicit_d_store_box_n,
            ),
        )
        if tcgen05_pure_matmul_object is not None:
            tcgen05_pure_matmul_object.register_pending_store(df.cute_state)
        if fx_node is not None:
            # Map the matmul fx_node -> result_var so the G3.1.1 fused
            # epilogue splice path can reuse the existing
            # `CuteTcgen05StoreValue` registration via a backward FX
            # walk from the user's store value through a whitelisted
            # unary chain to this matmul fx_node.
            df.cute_state.matmul_fx_node_result_vars[fx_node] = result_var
    else:
        # Each thread reads its own (m, n) element from shared memory.
        suffix.append(
            statement_from_string(f"{result_var} = {smem_c}[{m_local}, {n_local}]")
        )

    # Register per-tile statements with the persistent-loop splitter so
    # everything else hoists out of the work-tile loop. The splitter also
    # auto-detects PID-decomposition statements via ``virtual_pid_var``
    # name lookup, so callers don't need to plumb registration through
    # ``_decompose_virtual_pid``. No-op when the kernel uses a
    # non-persistent ``pid_type`` (the splitter is only invoked from
    # ``_setup_tcgen05_persistent_kernel``).
    if per_tile_stmts:
        df.cute_state.register_tcgen05_per_tile_stmts(per_tile_stmts)
    if mma_impl == "tcgen05":
        df.cute_state.register_tcgen05_kloop_owned_stmts(
            device_loop, device_loop.inner_statements[tcgen05_kloop_stmt_start:]
        )
    # Register role-block statements with the persistent role partitioner
    # (see ``Tcgen05PersistentProgramIDs._collect_tcgen05_role_blocks``).
    # Two registration shapes land here:
    # - Top-level prefix statements (the initial TMA prefetch IFs) --
    #   these are ALSO registered as per-tile via ``_emit_per_tile``,
    #   which is what keeps them inside the work-tile body so the
    #   partitioner can see them at top level.
    # - Nested statements emitted inside the K-loop body via
    #   ``cg.add_statement(...)`` -- these are NOT per-tile-registered;
    #   the K-loop itself rides into the work-tile body via per-tile
    #   name propagation, and the partitioner recurses one level into
    #   it to wrap these tagged children. The current static-full
    #   role-local path emits producer and exec K-loops as top-level
    #   sibling loops, so nested tags mainly serve the legacy inline path.
    #   Revisit this traversal if the legacy inline path is removed.
    # The partitioner asserts at run time that every registered tag was
    # visited, so a misregistered top-level stmt fails loudly rather
    # than silently dropping its role gate.
    if tma_load_role_stmts:
        df.cute_state.register_tcgen05_tma_load_role_stmts(tma_load_role_stmts)
    if mma_exec_role_stmts:
        df.cute_state.register_tcgen05_mma_exec_role_stmts(mma_exec_role_stmts)

    return expr_from_string(result_var)


def _mma_active_n_threads(mma_impl: str) -> int:
    if mma_impl in ("warp", "tcgen05"):
        return 2
    return 0


def _tcgen05_root_m_threads(bm: int, bn: int) -> int:
    # Wide tcgen05 tiles use a compact physical M thread map plus root-lane
    # serialization. Narrow N=8 tiles keep the original full-M thread family.
    if bn <= 8:
        return bm
    return min(32, bm)


def _tcgen05_tmem_barrier_thread_count(epi_warp_count: int) -> int:
    return 32 * (epi_warp_count + 1)


def _tcgen05_c_stage_count(bn: int) -> int:
    # Match the SM100 GEMM helper: narrow epilogues use a deeper TMA-store
    # ring buffer, while wider-N tiles fall back to two stages.
    return 4 if bn <= 16 else 2


def _tcgen05_ab_stage_count(num_stages: int) -> int:
    return max(1, min(int(num_stages), 2))


def _tcgen05_acc_stage_count(bn: int) -> int:
    # Match Quack/CUTLASS SM100 staging for the current non-blockscaled path:
    # keep two accumulator stages for all currently supported N<=256 tiles.
    return 2 if bn <= 256 else 1


def _tcgen05_config_int(config: object, key: str, default: int) -> int:
    value = cast("_ConfigLike", config).get(key, default)
    if not isinstance(value, int):
        return default
    return value


def _tcgen05_cluster_m(config: object) -> int:
    return max(1, min(2, _tcgen05_config_int(config, "tcgen05_cluster_m", 1)))


def _tcgen05_cluster_n(config: object) -> int:
    """Read the validated ``tcgen05_cluster_n`` knob (default 1).

    cluster_n=2 only runs under the canonical Quack-best 4-CTA cluster
    (``cluster_m=2 use_2cta=True``); the ``cluster_n>1`` capability gate
    in ``_codegen_cute_mma`` rejects unsupported pairings so this helper
    returns the *requested* value and lets the caller demote.
    """
    return max(1, min(2, _tcgen05_config_int(config, "tcgen05_cluster_n", 1)))


def _tcgen05_large_bn_proof_enabled(config: object | None) -> bool:
    return config is not None and (
        cast("_ConfigLike", config).get(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, False)
        is True
    )


def _tcgen05_large_bn_proof_shape(
    *, bm: int, bn: int, bk: int, tcgen05_cluster_m: int
) -> bool:
    return (
        (bm, bn, bk) == TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
        and tcgen05_cluster_m == TCGEN05_LARGE_BN_PROOF_CLUSTER_M
    )


def _tcgen05_use_2cta_instrs(*, bm: int, cluster_m: int) -> bool:
    # Match Quack/CUTLASS SM100: clustered kernels are not automatically the
    # tcgen05 "CTA pair" instruction family. The special 2-CTA instructions only
    # apply to the 256-wide M tiler. Our current legal Helion family still uses
    # CTA-local M=128 tiles, even when the cluster shape is (2, 1, 1).
    return cluster_m == 2 and bm == TCGEN05_TWO_CTA_BLOCK_M


def _tcgen05_epi_warp_count(
    warp_spec: Tcgen05WarpSpec, *, cta_thread_count: int
) -> int:
    """Pick the epilogue warp count for a tcgen05 matmul kernel.

    Returns at most ``cta_thread_count // 32`` warps, capped by
    ``warp_spec.epi_warps`` (the strategy data model's source of
    truth, sourced from the ``tcgen05_num_epi_warps`` autotune
    knob, default 4). The other roles (one MMA exec warp + one A/B
    load warp) are added on top of this in
    ``CuteTcgen05MatmulPlan.role_warp_count``.

    Today the only correct value for the SIMT-store epilogue is 4: the
    CUTLASS ``epilogue_tmem_copy_and_partition`` helper uses
    ``tmem_warp_shape_mn = (4, 1)`` for every supported tcgen05 path,
    which hard-codes a 4-warp t2r partition; the hardware ``tcgen05.ld``
    is per-warp so the partition is uncoverable by fewer warps. Both
    the autotune search and ``Config.normalize()`` validation are
    pinned to ``(4,)`` via
    ``ConfigSpec.narrow_tcgen05_autotune_to_validated_configs``;
    ``_codegen_cute_store_tcgen05_tile`` raises ``BackendUnsupported``
    if a value other than 4 still slips through. The 1 / 2 branches
    will only become meaningful when item 2's multi-warp epilogue
    (c_pipeline SMEM ring + TMA bulk store) lands and lets the t2r
    side keep its 4-warp partition independent of how many warps drive
    the GMEM store. See ``cute_plan.md`` Section 2.
    """
    cta_warp_count = max(1, cta_thread_count // 32)
    return min(cta_warp_count, max(1, warp_spec.epi_warps))


def _mma_impl_matches_problem_shape(
    mma_impl: str,
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
    tcgen05_cluster_m: int = 1,
    tcgen05_large_bn_proof: bool = False,
) -> bool:
    if mma_impl == "universal":
        return True
    if input_dtype not in (torch.float16, torch.bfloat16) or bn < 8 or bn % 8 != 0:
        return False
    if bn > 256 and (
        mma_impl != "tcgen05"
        or not tcgen05_large_bn_proof
        or not _tcgen05_large_bn_proof_shape(
            bm=bm,
            bn=bn,
            bk=bk,
            tcgen05_cluster_m=tcgen05_cluster_m,
        )
    ):
        return False
    if mma_impl == "warp":
        # Warp MMA atom is fixed-K (16 elements per BF16/FP16 instruction).
        return bk == 16 and bm >= 16 and bm % 16 == 0 and bn == 8
    if mma_impl == "tcgen05":
        # tcgen05 mma instruction K is 16 elements for BF16/FP16, but the
        # tile's K can be any positive multiple of that (the inner cute.gemm
        # loop just runs more instructions per K iteration). Larger tile_k
        # roughly halves the per-K-iter overhead per doubling. Production
        # remains capped at block_n=256 to keep AB SMEM staging budget sane;
        # the explicit G4 proof key admits only the smallest 512-N candidate.
        if bk < 16 or bk > 256 or bk % 16 != 0:
            return False
        if bm in (64, 128):
            return True
        return bm == TCGEN05_TWO_CTA_BLOCK_M and tcgen05_cluster_m == 2
    return False


def _is_zero_acc_expr(acc_expr: ast.AST) -> bool:
    if isinstance(acc_expr, ast.Constant):
        return acc_expr.value in (0, 0.0)
    if isinstance(acc_expr, ast.Call):
        if len(acc_expr.args) != 1 or acc_expr.keywords:
            return False
        if not _is_zero_acc_expr(acc_expr.args[0]):
            return False
        if isinstance(acc_expr.func, ast.Attribute):
            return acc_expr.func.attr in {"Float16", "Float32", "BFloat16"}
        if isinstance(acc_expr.func, ast.Name):
            return acc_expr.func.id in {"float", "int"}
    return False


def _is_zero_acc_node(node: Node | None) -> bool:
    if node is None:
        return False
    if node.op != "call_function":
        return False

    if node.target in _TRACE_THROUGH_TARGETS:
        src = node.args[0] if node.args else None
        return isinstance(src, Node) and _is_zero_acc_node(src)

    target_name = getattr(node.target, "__name__", "")
    if target_name in {"clone", "detach"}:
        src = node.args[0] if node.args else None
        return isinstance(src, Node) and _is_zero_acc_node(src)
    if target_name == "zeros":
        return True
    if target_name != "full":
        return False

    value = None
    if len(node.args) > 1:
        value = node.args[1]
    elif "value" in node.kwargs:
        value = node.kwargs["value"]
    return value in (0, 0.0)


def _choose_mma_impl(
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
    config: object | None = None,
) -> str:
    tcgen05_cluster_m = 1
    if config is not None:
        tcgen05_cluster_m = _tcgen05_cluster_m(config)
    tcgen05_large_bn_proof = _tcgen05_large_bn_proof_enabled(config)
    env_choice = os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower()
    support = get_cute_mma_support()
    if env_choice != "auto":
        if env_choice not in support.supported_impls:
            raise exc.BackendUnsupported(
                "cute",
                (
                    f"Requested HELION_CUTE_MMA_IMPL={env_choice!r} is not supported "
                    f"on this machine. Supported: {support.supported_impls}"
                ),
            )
        if _mma_impl_matches_problem_shape(
            env_choice,
            input_dtype,
            bm=bm,
            bn=bn,
            bk=bk,
            tcgen05_cluster_m=tcgen05_cluster_m,
            tcgen05_large_bn_proof=tcgen05_large_bn_proof,
        ):
            return env_choice
        return "universal"
    if _mma_impl_matches_problem_shape(
        "tcgen05",
        input_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        tcgen05_cluster_m=tcgen05_cluster_m,
        tcgen05_large_bn_proof=tcgen05_large_bn_proof,
    ):
        if support.tcgen05_f16bf16:
            return "tcgen05"
    if _mma_impl_matches_problem_shape("warp", input_dtype, bm=bm, bn=bn, bk=bk):
        if support.warp_f16bf16:
            return "warp"
    return "universal"


def _make_tiled_mma_setup(
    mma_impl: str,
    tiled_mma: str,
    thr_mma: str,
    mma_thread_linear: str,
    input_dtype_str: str,
    acc_dtype_str: str,
    bm: int,
    bn: int,
    *,
    tcgen05_cluster_m: int = 1,
) -> list[ast.AST]:
    if mma_impl == "warp":
        tiled_mma_expr = (
            "cute.make_tiled_mma("
            "cute.make_mma_atom("
            f"cute.nvgpu.warp.MmaF16BF16Op({input_dtype_str}, {acc_dtype_str}, (16, 8, 16))"
            f"), atom_layout_mnk=({bm // 16}, 1, 1))"
        )
    elif mma_impl == "tcgen05":
        tiled_mma_expr = _tcgen05_tiled_mma_expr(
            input_dtype_str,
            acc_dtype_str,
            bm,
            bn,
            tcgen05_cluster_m=tcgen05_cluster_m,
        )
    else:
        assert mma_thread_linear
        return [
            statement_from_string(
                f"{tiled_mma} = cute.make_tiled_mma("
                f"cute.nvgpu.MmaUniversalOp(abacc_dtype={acc_dtype_str}), "
                f"atom_layout_mnk=({bm}, {bn}, 1))"
            ),
            statement_from_string(
                f"{thr_mma} = {tiled_mma}.get_slice({mma_thread_linear})"
            ),
        ]
    return [
        statement_from_string(f"{tiled_mma} = {tiled_mma_expr}"),
        statement_from_string(
            f"{thr_mma} = {tiled_mma}.get_slice({mma_thread_linear})"
        ),
    ]


def _tcgen05_tiled_mma_expr(
    input_dtype_str: str,
    acc_dtype_str: str,
    bm: int,
    bn: int,
    *,
    tcgen05_cluster_m: int = 1,
) -> str:
    cta_group_expr = "cute.nvgpu.tcgen05.CtaGroup.ONE"
    if _tcgen05_use_2cta_instrs(bm=bm, cluster_m=tcgen05_cluster_m):
        cta_group_expr = "cute.nvgpu.tcgen05.CtaGroup.TWO"
    return (
        "cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
        f"{input_dtype_str}, "
        f"{input_dtype_str}, "
        "cute.nvgpu.OperandMajorMode.K, "
        "cute.nvgpu.OperandMajorMode.MN, "
        f"{acc_dtype_str}, "
        f"{cta_group_expr}, "
        f"({bm}, {bn}), "
        "cute.nvgpu.tcgen05.OperandSource.SMEM)"
    )


def _new_tcgen05_layout_plan(df: DeviceFunction) -> _Tcgen05LayoutPlan:
    return _Tcgen05LayoutPlan(
        exec_active=df.new_var("tcgen05_exec_active"),
        smem_a_layout=df.new_var("sA_layout"),
        smem_b_layout=df.new_var("sB_layout"),
        c_layout=df.new_var("tcgen05_c_layout"),
        epi_tile=df.new_var("tcgen05_epi_tile"),
        tmem_load_atom=df.new_var("tcgen05_tmem_load_atom"),
        acc_tmem_cols=df.new_var("tcgen05_acc_tmem_cols"),
        tmem_holding_buf=df.new_var("tcgen05_tmem_holding_buf"),
        tmem_dealloc_mbar_ptr=df.new_var("tcgen05_tmem_dealloc_mbar_ptr"),
        tmem_alloc_barrier=df.new_var("tcgen05_tmem_alloc_barrier"),
        tmem_allocator=df.new_var("tcgen05_tmem_allocator"),
        acc_pipeline_barriers=df.new_var("tcgen05_acc_pipeline_barriers"),
        acc_pipeline_producer_group=df.new_var("tcgen05_acc_pipeline_producer_group"),
        acc_pipeline_consumer_group=df.new_var("tcgen05_acc_pipeline_consumer_group"),
        acc_pipeline=df.new_var("tcgen05_acc_pipeline"),
        acc_producer_state=df.new_var("tcgen05_acc_producer_state"),
        acc_consumer_state=df.new_var("tcgen05_acc_consumer_state"),
        epilogue_rest_mode=df.new_var("tcgen05_epilogue_rest_mode"),
    )


def _make_tcgen05_layout_plan_setup(
    plan: _Tcgen05LayoutPlan,
    tiled_mma: str,
    *,
    bm: int,
    bn: int,
    bk: int,
    ab_stage_count: int,
    is_two_cta: bool,
    input_dtype_str: str,
    acc_dtype_str: str,
    epi_elem_dtype_str: str | None = None,
    smem_swizzle_a: int | None = None,
    smem_swizzle_b: int | None = None,
    explicit_epi_tile_m: int | None = None,
    explicit_epi_tile_n: int | None = None,
) -> list[ast.AST]:
    # `compute_epilogue_tile_shape` must receive `elem_ty_d` and `elem_ty_c`
    # equal to the eventual D-output dtype so the helper takes the
    # with-source branch (e.g. bf16 → `tile_n=64`) rather than the
    # `disable_source=True` branch (`tile_n=32`); the matmul-plan epi_tile
    # built here, the store-side epi_tile in
    # `_codegen_cute_store_tcgen05_tile`, and the wrapper-side TMA atom in
    # `helion/runtime/__init__.py` must all see the same `tile_n`. When
    # `epi_elem_dtype_str` is omitted the input dtype is used as a fallback;
    # the store-side equality check on the registered `CuteTcgen05StoreValue`
    # is the loud-failure backstop for any mismatch.
    if epi_elem_dtype_str is None:
        epi_elem_dtype_str = input_dtype_str
    if (explicit_epi_tile_m is None) != (explicit_epi_tile_n is None):
        raise exc.BackendUnsupported(
            "cute",
            "explicit tcgen05 epilogue tile requires both tile dimensions",
        )
    epi_tile_expr = (
        tcgen05_explicit_epilogue_tile_expr(explicit_epi_tile_m, explicit_epi_tile_n)
        if explicit_epi_tile_m is not None and explicit_epi_tile_n is not None
        else tcgen05_default_epilogue_tile_expr(
            bm, bn, epi_elem_dtype_str, c_layout=plan.c_layout
        )
    )
    return [
        statement_from_string(
            f"{plan.smem_a_layout} = "
            f"{tcgen05_smem_layout_expr(tiled_mma=tiled_mma, bm=bm, bn=bn, bk=bk, dtype_str=input_dtype_str, num_stages=ab_stage_count, operand='a', swizzle_override=smem_swizzle_a)}"
        ),
        statement_from_string(
            f"{plan.smem_b_layout} = "
            f"{tcgen05_smem_layout_expr(tiled_mma=tiled_mma, bm=bm, bn=bn, bk=bk, dtype_str=input_dtype_str, num_stages=ab_stage_count, operand='b', swizzle_override=smem_swizzle_b)}"
        ),
        statement_from_string(
            f"{plan.c_layout} = cutlass.utils.layout.LayoutEnum.ROW_MAJOR"
        ),
        statement_from_string(f"{plan.epi_tile} = {epi_tile_expr}"),
        statement_from_string(
            f"{plan.tmem_load_atom} = cutlass.utils.blackwell_helpers.get_tmem_load_op("
            f"({bm}, {bn}, {bk}), {plan.c_layout}, "
            f"{acc_dtype_str}, {acc_dtype_str}, {plan.epi_tile}, {is_two_cta!s})"
        ),
        statement_from_string(
            f"{plan.epilogue_rest_mode} = cute.make_layout(1, stride=0)"
        ),
    ]


def _new_tcgen05_sched_pipeline_plan(
    df: DeviceFunction,
    *,
    use_clc: bool = False,
) -> _Tcgen05SchedPipelinePlan:
    """Allocate variable names for the scheduler-broadcast pipeline.

    The ``tcgen05_sched_pipeline_*`` prefix family is shared with
    the existing cluster_m=2 ONE-CTA bridge emission in
    ``program_id._build_tcgen05_persistent_layout``: ``df.new_var``
    appends an incrementing suffix so the two emissions cannot
    actually collide, but a future cycle that consolidates the two
    paths should drive both call sites through this allocator.

    ``use_clc=True`` additionally allocates the CLC response buffer
    + mbarrier variable names (G2-H, cute_plan.md). Static path
    leaves them empty so consumers can detect via simple string
    truthiness.
    """
    clc_response_smem_ptr = ""
    clc_response_tensor = ""
    clc_mbar_smem_ptr = ""
    clc_mbar_tensor = ""
    clc_mbar_phase = ""
    if use_clc:
        clc_response_smem_ptr = df.new_var("tcgen05_clc_response_smem_ptr")
        clc_response_tensor = df.new_var("tcgen05_clc_response_tensor")
        clc_mbar_smem_ptr = df.new_var("tcgen05_clc_mbar_smem_ptr")
        clc_mbar_tensor = df.new_var("tcgen05_clc_mbar_tensor")
        clc_mbar_phase = df.new_var("tcgen05_clc_mbar_phase")
    return _Tcgen05SchedPipelinePlan(
        barriers=df.new_var("tcgen05_sched_pipeline_mbars"),
        producer_group=df.new_var("tcgen05_sched_pipeline_producer_group"),
        consumer_group=df.new_var("tcgen05_sched_pipeline_consumer_group"),
        pipeline=df.new_var("tcgen05_sched_pipeline"),
        producer_state=df.new_var("tcgen05_sched_pipeline_producer_state"),
        consumer_state=df.new_var("tcgen05_sched_pipeline_consumer_state"),
        clc_response_smem_ptr=clc_response_smem_ptr,
        clc_response_tensor=clc_response_tensor,
        clc_mbar_smem_ptr=clc_mbar_smem_ptr,
        clc_mbar_tensor=clc_mbar_tensor,
        clc_mbar_phase=clc_mbar_phase,
    )


def _emit_clc_smem_setup(plan: _Tcgen05SchedPipelinePlan) -> list[ast.AST]:
    """Emit the SMEM allocation + CLC mbarrier-init for the CLC path.

    Mirrors Quack's ``TileScheduler._init_clc_mbarrier``: allocate a
    SMEM tile sized for the CLC response (4 Int32 = 16 bytes) and a
    one-arrival mbarrier that the scheduler warp uses with
    ``nvvm.clusterlaunchcontrol_try_cancel``. Quack packs the
    mbarrier next to the response in a single SMEM tile keyed by
    pipeline index; Helion's CLC path uses the simpler one-stage
    layout (the broadcast pipeline already serializes the consumer
    handoff), so a single 4-Int32 response buffer + a 2-Int32
    mbarrier (one Int64 cell) is enough.

    The mbarrier itself is initialized with ``mbarrier_init(addr,
    1)`` — only the single CLC issuer (lane 0 of the scheduler warp)
    arrives — and a phase counter is materialized so the wait
    flips between 0/1 each iteration. ``mbarrier_init_fence`` +
    ``sync_warp`` follow Quack's pattern; both are warp-uniform and
    happen before the persistent loop begins.

    Caller wires this into the prefix immediately after the
    ``_emit_sched_pipeline_setup`` block so the alloc / init pair
    sits where the existing pipeline init lives.
    """
    assert plan.clc_response_smem_ptr, (
        "CLC SMEM setup requires plan.clc_response_smem_ptr; was "
        "_new_tcgen05_sched_pipeline_plan called with use_clc=True?"
    )
    return [
        # CLC response buffer: 4 Int32 (bidx, bidy, bidz, valid).
        # ``cute.arch.clc_response`` reads the 16-byte block back
        # into 4 register values.
        statement_from_string(
            f"{plan.clc_response_smem_ptr} = cute.arch.alloc_smem("
            f"cutlass.Int32, cutlass.Int32(4), alignment=16)"
        ),
        statement_from_string(
            f"{plan.clc_response_tensor} = cute.make_tensor("
            f"{plan.clc_response_smem_ptr}, cute.make_layout((4,), stride=(1,)))"
        ),
        # CLC mbarrier: one Int64 cell. ``mbarrier_init`` arms it
        # with arrival count 1 (only the CLC issuer arrives via
        # ``mbarrier_arrive_and_expect_tx``).
        statement_from_string(
            f"{plan.clc_mbar_smem_ptr} = cute.arch.alloc_smem("
            f"cutlass.Int64, cutlass.Int32(1), alignment=8)"
        ),
        statement_from_string(
            f"{plan.clc_mbar_tensor} = cute.make_tensor("
            f"{plan.clc_mbar_smem_ptr}, cute.make_layout((1,), stride=(1,)))"
        ),
        # Phase counter for the CLC mbarrier wait. The scheduler-warp
        # body flips this each iteration; initialized to 0 so the
        # first wait pairs with the first arrival's phase.
        statement_from_string(f"{plan.clc_mbar_phase} = cutlass.Int32(0)"),
    ]


def _emit_sched_pipeline_setup(
    plan: _Tcgen05SchedPipelinePlan,
    *,
    sched_stage_count: int,
    consumer_arrive_count: int,
    cluster_size: int,
    defer_sync: bool,
    producer_arrive_count: int,
    consumer_mask_to_leader: bool = True,
) -> list[ast.AST]:
    """Emit the prefix statements that construct the sched pipeline.

    Mirrors Quack's ``make_sched_pipeline`` in
    ``quack/quack/gemm_sm100.py``. The cluster_m=2 ONE-CTA bridge
    diagnostic in
    ``program_id.Tcgen05PersistentProgramIDs._build_tcgen05_persistent_layout``
    already inlines an equivalent emission for a different role
    topology (peer-CTA work-tile publish via
    ``_cute_store_shared_remote_x4``); G2-C should consider
    consolidating that path onto this helper rather than carrying
    two parallel emitters.

    Parameters:

    - ``consumer_arrive_count``: caller-supplied total number of
      consumer arrivals per stage on the empty barrier. With
      ``consumer_mask_to_leader=True`` (Quack pattern) every CTA's
      consumer release routes to the leader CTA's empty barrier so
      this is the cluster-wide total
      (``warps_per_cta * cluster_size``). With
      ``consumer_mask_to_leader=False`` releases stay local so this
      is the per-CTA count (``warps_per_cta``).
    - ``cluster_size``: cluster-multicast factor. ``> 1`` lets
      ``defer_sync`` participate in cluster-wide barrier init.
    - ``consumer_mask_to_leader``: ``True`` emits
      ``consumer_mask=cutlass.Int32(0)`` so every consumer release
      arrives on the leader CTA's empty barrier (matches Quack's
      single-cluster-leader scheduler topology where only the
      leader runs the producer side and broadcasts via peer-CTA
      writes). ``False`` omits the mask so each CTA's empty
      barrier collects its own consumers' arrivals (matches the
      "every CTA runs its own scheduler that publishes to its own
      consumers" topology used by the WITH_SCHEDULER strategy).
      Picking the wrong topology for the actual scheduler
      placement causes a clean-on-cluster_m=1 / hang-on-cluster_m>1
      regression because the asymmetric arrival counts mismatch.
      Ignored when ``cluster_size <= 1`` (no mask is emitted in
      either case).
    - ``defer_sync``: emits ``defer_sync=True`` so the pipeline
      participates in the cluster-wide deferred-init protocol
      coordinated via ``pipeline_init_arrive`` /
      ``pipeline_init_wait``. The caller threads
      ``tcgen05_use_cluster_deferred_pipelines`` (see
      ``cute_mma._codegen_cute_mma``) the same way the AB / acc
      pipelines do via ``tcgen05_defer_pipeline_sync_arg``;
      forgetting this on a clustered call site risks barrier-init
      ordering hangs.

    ``num_stages`` and ``make_pipeline_state`` count arguments are
    bare ints (matching the existing AB / acc / c pipeline
    emissions), but the SMEM mbar size and the consumer-arrive
    count are wrapped in ``cutlass.Int32(...)`` literals (also
    matching the existing emissions and the established pattern in
    ``program_id.py``'s scheduler emission). No named compile-time
    constants are materialized — same convention as
    ``_make_tcgen05_layout_plan_setup``.
    """
    extra_args = ""
    if cluster_size > 1 and consumer_mask_to_leader:
        extra_args += ", consumer_mask=cutlass.Int32(0)"
    if defer_sync:
        extra_args += ", defer_sync=True"
    return [
        statement_from_string(
            f"{plan.barriers} = cute.arch.alloc_smem("
            f"cutlass.Int64, cutlass.Int32({sched_stage_count * 2}))"
        ),
        statement_from_string(
            f"{plan.producer_group} = "
            "cutlass.pipeline.CooperativeGroup("
            f"cutlass.pipeline.Agent.Thread, {producer_arrive_count})"
        ),
        statement_from_string(
            f"{plan.consumer_group} = "
            "cutlass.pipeline.CooperativeGroup("
            f"cutlass.pipeline.Agent.Thread, cutlass.Int32({consumer_arrive_count}))"
        ),
        statement_from_string(
            f"{plan.pipeline} = cutlass.pipeline.PipelineAsync.create("
            f"num_stages={sched_stage_count}, "
            f"producer_group={plan.producer_group}, "
            f"consumer_group={plan.consumer_group}, "
            f"barrier_storage={plan.barriers}"
            f"{extra_args})"
        ),
        statement_from_string(
            f"{plan.producer_state} = cutlass.pipeline.make_pipeline_state("
            f"cutlass.pipeline.PipelineUserType.Producer, {sched_stage_count})"
        ),
        statement_from_string(
            f"{plan.consumer_state} = cutlass.pipeline.make_pipeline_state("
            f"cutlass.pipeline.PipelineUserType.Consumer, {sched_stage_count})"
        ),
    ]


def _tcgen05_aux_pipeline_stage_count_from_config(config: object) -> int:
    """Return the aux-pipeline stage count for ``config``, defaulting to
    ``TCGEN05_AUX_STAGE_COUNT_DEFAULT`` when the knob is absent.

    Bad values raise via the downstream
    ``_new_tcgen05_aux_pipeline_plan`` assert; the validator gate
    (``_validate_int_enum_config`` in ``tcgen05_config.py``) is the
    single source of truth that rejects out-of-range values before
    they reach codegen, so any value seen here that fails the
    downstream assert is a programmer bug, not user input.
    """
    return _tcgen05_config_int(
        config, TCGEN05_AUX_STAGES_CONFIG_KEY, TCGEN05_AUX_STAGE_COUNT_DEFAULT
    )


def _tcgen05_consumer_regs_from_config(config: object) -> int:
    """Return the consumer-warp ``setmaxregister_increase`` ceiling for
    ``config``, defaulting to ``TCGEN05_CONSUMER_REGS_DEFAULT`` (256)
    when the knob is absent.

    Cycle 15 H2 (``cute_plan.md`` §6 Target 8). The default preserves
    cycle-14 byte-identical emission; lower values force ``ptxas`` to
    cap the consumer-warp per-thread register count. The validator
    (``_validate_int_enum_config`` in ``tcgen05_config.py``) is the
    single source of truth that rejects out-of-range values before
    they reach codegen; values seen here that are outside
    ``TCGEN05_CONSUMER_REGS_CHOICES`` are programmer bugs, not user
    input, so this helper falls back to the default rather than
    asserting (matching the ``_tcgen05_aux_pipeline_stage_count_from_config``
    pattern). The default is included in ``CHOICES`` so the
    default-with-knob configuration emits the same code as the
    default-without-knob configuration.
    """
    value = _tcgen05_config_int(
        config, TCGEN05_CONSUMER_REGS_CONFIG_KEY, TCGEN05_CONSUMER_REGS_DEFAULT
    )
    if value not in TCGEN05_CONSUMER_REGS_CHOICES:
        return TCGEN05_CONSUMER_REGS_DEFAULT
    return value


def _new_tcgen05_aux_pipeline_plan(
    df: DeviceFunction,
    *,
    num_rings: int,
    epi_tile_var: str,
    use_tma_load: bool,
    stage_count: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
) -> _Tcgen05AuxPipelinePlan:
    """Allocate variable names for the C-input warp's aux-tensor
    SMEM-ring pipeline (``cute_plan.md`` §7.5.3.2).

    ``num_rings`` is the number of aux-tensor descriptors registered
    on the matmul plan; the producer body indexes the rings by
    descriptor position, and the consumer-side flip in
    ``memory_ops._aux_subtile_load_source`` consumes the same
    ordering.

    ``epi_tile_var`` is the matmul-plan ``epi_tile`` variable name;
    the producer body uses it to subdivide the per-output-tile aux
    GMEM region into epi-tile-sized chunks. The ``(bm, bn)`` block
    shape is read directly from the matmul plan by the
    producer-body codegen, so no block-shape fields are plumbed
    through the aux pipeline plan.

    ``stage_count`` controls the depth of the SMEM ring. Cycle 10
    makes this config-driven so the T8 wide-N CLC + aux-TMA seed
    family can sample ``{2, 3}`` while every other path keeps the
    pre-cycle-10 default of 2 (preserving T1-T7 byte-identity).
    """
    assert num_rings >= 1, (
        "aux pipeline plan requires at least one descriptor; the gate "
        "in ``_codegen_cute_mma`` should not call this with an empty "
        "descriptor tuple"
    )
    assert stage_count in TCGEN05_AUX_STAGE_COUNT_CHOICES, (
        f"aux pipeline stage_count={stage_count!r} not in "
        f"TCGEN05_AUX_STAGE_COUNT_CHOICES={TCGEN05_AUX_STAGE_COUNT_CHOICES!r}"
    )
    rings = tuple(
        _Tcgen05AuxPerDescriptorRingNames(
            smem_layout=df.new_var(f"tcgen05_aux_smem_layout_{idx}"),
            smem_ptr=df.new_var(f"tcgen05_aux_smem_ptr_{idx}"),
            smem=df.new_var(f"tcgen05_aux_smem_{idx}"),
            tma_atom=(
                df.new_var(f"tcgen05_aux_tma_atom_{idx}") if use_tma_load else None
            ),
            tma_tensor=(
                df.new_var(f"tcgen05_aux_tma_tensor_{idx}") if use_tma_load else None
            ),
        )
        for idx in range(num_rings)
    )
    return _Tcgen05AuxPipelinePlan(
        barriers=df.new_var("tcgen05_aux_pipeline_mbars"),
        producer_group=df.new_var("tcgen05_aux_pipeline_producer_group"),
        consumer_group=df.new_var("tcgen05_aux_pipeline_consumer_group"),
        pipeline=df.new_var("tcgen05_aux_pipeline"),
        producer_state=df.new_var("tcgen05_aux_pipeline_producer_state"),
        consumer_state=df.new_var("tcgen05_aux_pipeline_consumer_state"),
        rings=rings,
        use_tma_load=use_tma_load,
        stage_count=stage_count,
        epi_tile_var=epi_tile_var,
    )


def _emit_tcgen05_aux_pipeline_setup(
    plan: _Tcgen05AuxPipelinePlan,
    *,
    descriptor_dtype_strs: tuple[str, ...],
    tile_shape_expr: str,
    c_input_warp_thread_count: int,
    epi_warp_count: int,
    defer_sync: bool,
) -> list[ast.AST]:
    """Emit the prefix statements that construct the aux SMEM rings
    + ``c_pipeline_aux`` ``PipelineAsync`` for cycle 2 of the
    producer-body split (``cute_plan.md`` §7.5.3.2).

    Per descriptor, allocates a SMEM ring sized by
    ``make_smem_layout_epi(<aux_dtype>, ROW_MAJOR, epi_tile,
    plan.stage_count)`` — each ring stage holds ONE subtile
    (``epi_tile``-shaped slice) of the per-output-tile aux region,
    not the full ``(bm, bn)`` tile. Per-subtile staging keeps the
    SMEM ring footprint at one ``epi_tile`` chunk per stage rather
    than one ``(bm, bn)`` chunk, which is essential to fit
    cluster_m=2 + ``tcgen05_ab_stages=3`` in the 228 KB B200 SMEM
    cap. The producer issues one cooperative
    ``cute.copy(GMEM_aux_subtile, SMEM_aux_ring[stage])`` *per
    subtile* (looping over the per-output-tile subtile axis),
    framed by ``producer_acquire`` / ``producer_commit`` / state
    advance; the consumer issues one ``consumer_wait`` / Quack-
    style ``tiled_copy_s2r`` ``cute.copy(SMEM_ring[stage], rmem)``
    / lane-0 ``consumer_release`` / state advance per subtile.
    ``tile_shape_expr`` is the ``epi_tile`` variable name so the
    SMEM ring sizing matches the producer-side subtile copy
    extent. ``plan.stage_count`` controls the depth — cycle 10
    makes it config-driven so the T8 wide-N CLC + aux-TMA seed
    family can sample ``{2, 3}``.

    Pipeline parameters:

    - SIMT aux loads use ``producer_arrive_count =
      c_input_warp_thread_count`` (32 for the single C-input warp).
      The producer body issues ``producer_acquire`` /
      ``producer_commit`` per-thread, so the cooperative-group
      arrive count is per-thread. TMA aux loads use a
      ``PipelineTmaAsync`` producer group matching the CUTLASS TMA
      pipeline convention.
    - ``consumer_arrive_count = epi_warp_count`` (per-warp, NOT
      per-thread). The consumer-side flip in
      ``memory_ops._aux_subtile_load_source`` gates
      ``consumer_release(c_pipeline_aux)`` on ``elect_one()``
      (matching the sched-pipeline pattern from
      ``_build_role_local_while_with_scheduler``). Setting
      per-thread would hang the handshake waiting for 31
      missing per-warp arrivals per stage.
    - ``defer_sync`` mirrors the AB / acc / sched pipelines'
      cluster-deferred-init participation so the
      ``pipeline_init_arrive`` / ``pipeline_init_wait`` rendezvous
      spans every pipeline.
    """
    extra_args = ", defer_sync=True" if defer_sync else ""
    stage_count = plan.stage_count
    lines: list[ast.AST] = []
    for ring, dtype_str in zip(plan.rings, descriptor_dtype_strs, strict=True):
        lines.extend(
            [
                statement_from_string(
                    f"{ring.smem_layout} = "
                    f"cutlass.utils.blackwell_helpers.make_smem_layout_epi("
                    f"{dtype_str}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                    f"{tile_shape_expr}, {stage_count})"
                ),
                statement_from_string(
                    f"{ring.smem_ptr} = cute.arch.alloc_smem("
                    f"{dtype_str}, cute.cosize({ring.smem_layout}.outer), "
                    "alignment=1024)"
                ),
                statement_from_string(
                    f"{ring.smem} = cute.make_tensor("
                    f"cute.recast_ptr({ring.smem_ptr}, "
                    f"{ring.smem_layout}.inner, dtype={dtype_str}), "
                    f"{ring.smem_layout}.outer)"
                ),
            ]
        )
    lines.append(
        statement_from_string(
            f"{plan.barriers} = cute.arch.alloc_smem("
            f"cutlass.Int64, cutlass.Int32({stage_count * 2}))"
        )
    )
    if plan.use_tma_load:
        tx_terms = [
            f"cute.size_in_bytes({dtype_str}, cute.slice_({ring.smem_layout}.outer, "
            "(None, None, 0)))"
            for ring, dtype_str in zip(plan.rings, descriptor_dtype_strs, strict=True)
        ]
        tx_count = " + ".join(tx_terms)
        lines.extend(
            [
                statement_from_string(
                    f"{plan.producer_group} = "
                    "cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread)"
                ),
                statement_from_string(
                    f"{plan.consumer_group} = "
                    "cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, "
                    f"cutlass.Int32({epi_warp_count}))"
                ),
                statement_from_string(
                    f"{plan.pipeline} = cutlass.pipeline.PipelineTmaAsync.create("
                    f"num_stages={stage_count}, "
                    f"producer_group={plan.producer_group}, "
                    f"consumer_group={plan.consumer_group}, "
                    f"tx_count={tx_count}, "
                    f"barrier_storage={plan.barriers}"
                    f"{extra_args})"
                ),
            ]
        )
    else:
        lines.extend(
            [
                statement_from_string(
                    f"{plan.producer_group} = "
                    "cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, "
                    f"cutlass.Int32({c_input_warp_thread_count}))"
                ),
                statement_from_string(
                    f"{plan.consumer_group} = "
                    "cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, "
                    f"cutlass.Int32({epi_warp_count}))"
                ),
                statement_from_string(
                    f"{plan.pipeline} = cutlass.pipeline.PipelineAsync.create("
                    f"num_stages={stage_count}, "
                    f"producer_group={plan.producer_group}, "
                    f"consumer_group={plan.consumer_group}, "
                    f"barrier_storage={plan.barriers}"
                    f"{extra_args})"
                ),
            ]
        )
    lines.extend(
        [
            statement_from_string(
                f"{plan.producer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, {stage_count})"
            ),
            statement_from_string(
                f"{plan.consumer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, {stage_count})"
            ),
        ]
    )
    return lines


def _tcgen05_epilogue_dest_expr(plan: _Tcgen05LayoutPlan, tensor: str) -> str:
    planned_layout = tensor + ".layout"
    for _ in range(3):
        planned_layout = f"cute.append({planned_layout}, {plan.epilogue_rest_mode})"
    return f"cute.make_tensor({tensor}.iterator, {planned_layout})"


def _validate_tcgen05_smem_swizzle_override(
    *,
    operand: str,
    swizzle_bytes: int,
    bm: int,
    bn: int,
    bk: int,
    input_dtype: torch.dtype,
) -> None:
    """Reject illegal ``smem_swizzle_a/b`` overrides at codegen time.

    CuTe's ``make_smem_layout_atom`` requires the major-mode bytes-per-row
    to be a multiple of the swizzle pattern's contiguous bytes. For
    Helion's tcgen05 lowering:

    - A is K-major: major-mode = K dimension; bytes-per-row =
      ``bk * dtype_width_bits / 8``.
    - B is MN-major: major-mode = N dimension; bytes-per-row =
      ``bn * dtype_width_bits / 8``.

    This helper computes the active bytes-per-row from the live tile
    shape + dtype and rejects swizzle overrides that violate the atom
    contract with a structured ``BackendUnsupported`` error so the
    autotune surface drops the bad config rather than crashing inside
    CuTe at runtime.

    Caller has already verified ``swizzle_bytes`` is a member of
    ``TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES`` (the data-model validator
    in ``strategies.validate_tcgen05_strategy_invariants`` does that).
    Here we layer the *contract* check (does the active tile fit the
    requested swizzle?) on top of the *value* check.
    """
    assert swizzle_bytes in TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES, (
        f"_validate_tcgen05_smem_swizzle_override: invalid swizzle byte "
        f"{swizzle_bytes!r}; expected one of {TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r}"
    )
    # ``torch.dtype.itemsize`` is bytes; the major-mode size in bytes is
    # the major-mode tile extent times the dtype width in bytes. (We
    # could equivalently express this in bits to mirror CuTe's
    # ``num_contiguous_bits`` constants — bytes is more readable and
    # the comparison is exact since dtype widths divide 8 for every
    # MMA-supported dtype.)
    dtype_bytes = input_dtype.itemsize
    if operand == "a":
        major_mode_extent = bk
        major_mode_axis = "K"
    else:
        assert operand == "b", f"unexpected operand {operand!r}"
        major_mode_extent = bn
        major_mode_axis = "N"
    major_mode_bytes = major_mode_extent * dtype_bytes
    min_required = smem_swizzle_min_major_mode_bytes(swizzle_bytes)
    if major_mode_bytes % min_required != 0:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 smem_swizzle_{operand}={swizzle_bytes} requires "
            f"the {major_mode_axis}-axis bytes-per-row to be a multiple "
            f"of {min_required} (CuTe SmemLayoutAtom contract); active "
            f"tile shape (bm={bm}, bn={bn}, bk={bk}) with dtype "
            f"{input_dtype!s} ({dtype_bytes}B) yields "
            f"{major_mode_axis}-axis bytes={major_mode_bytes}",
        )


# ---- Aten lowering entry point (addmm/mm/bmm/baddbmm) ----


def codegen_cute_mma(
    ctx: LoweringContext,
    node: Node,
    with_acc: bool,
) -> ast.AST | None:
    """Generate MMA code for an aten addmm/mm node.  Returns None to fall back."""
    from ..generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        return None
    if ctx.cg.current_grid_state is None:
        return None
    if not can_codegen_cute_mma_aten(node, with_acc):
        return None

    if with_acc:
        acc_node = node.args[0]
        assert isinstance(acc_node, Node)
        acc_expr = (
            None if _is_zero_init_acc_node(acc_node) else ctx.to_ast(ctx.env[acc_node])
        )
        lhs_node, rhs_node = node.args[1], node.args[2]
    else:
        acc_expr = None
        lhs_node, rhs_node = node.args[0], node.args[1]
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    return _emit_mma_pipeline(
        ctx.cg,
        lhs_node,
        rhs_node,
        acc_expr=acc_expr,
        fx_node=node,
    )


def codegen_cute_mma_direct_mm(
    ctx: LoweringContext,
    node: Node,
    *,
    serial_k_extent: int | None,
) -> ast.AST | None:
    from ..generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        return None
    plan = getattr(ctx, "cute_matmul_plan", None)
    if not isinstance(plan, MatmulExecutionPlan):
        return None
    if plan.kind is not MatmulExecutionKind.DIRECT_GROUPED_N:
        return None
    if serial_k_extent is None or serial_k_extent <= 0:
        return None
    if node.target is not torch.ops.aten.mm.default:
        return None

    lhs_node = node.args[0]
    rhs_node = node.args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return None
    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return None
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if (
        lhs_fake.ndim != 2
        or rhs_fake.ndim != 2
        or not isinstance(lhs_val, torch.Tensor)
        or not isinstance(rhs_val, torch.Tensor)
        or lhs_val.ndim != 2
        or rhs_val.ndim != 2
    ):
        return None
    if lhs_fake.dtype not in (torch.float16, torch.bfloat16):
        return None
    load_plan = analyze_direct_grouped_n_loads(
        lhs_load,
        rhs_load,
        k_extent=serial_k_extent,
        n_extent=int(rhs_val.shape[1]),
    )
    if load_plan is None:
        return None

    mma_impl = _choose_mma_impl(
        lhs_fake.dtype,
        bm=plan.bm,
        bn=plan.bn,
        bk=plan.bk,
        config=ctx.cg.device_function.config,
    )
    # The grouped-N direct path only emits warp MMA. Auto-selection prefers
    # tcgen05 for plan.bm in (64, 128), but tcgen05 isn't implemented here, so
    # transparently fall back to warp on tcgen05-capable machines as long as
    # the user didn't explicitly request a different implementation.
    if (
        mma_impl == "tcgen05"
        and os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower() == "auto"
        and _mma_impl_matches_problem_shape(
            "warp", lhs_fake.dtype, bm=plan.bm, bn=plan.bn, bk=plan.bk
        )
        and get_cute_mma_support().warp_f16bf16
    ):
        mma_impl = "warp"
    if mma_impl != "warp":
        return None

    cg = ctx.cg
    grid_state = cg.current_grid_state
    if grid_state is None:
        return None
    prefix = grid_state.outer_prefix
    scalar_axis = grid_state.block_thread_axes.get(plan.scalar_block_id)
    if scalar_axis is None:
        return None
    scalar_strategy = cg.device_function.tile_strategy.block_id_to_strategy.get(
        (plan.scalar_block_id,)
    )
    lane_var = getattr(scalar_strategy, "_synthetic_cute_lane_var", None)
    if plan.lane_extent > 1 and not isinstance(lane_var, str):
        return None

    m_index_var = grid_state.strategy.index_var(plan.m_block_id)
    m_local = _local_mma_coord_expr(cg, plan.m_block_id)
    m_tile_origin = f"cutlass.Int32({m_index_var}) - ({m_local})"
    scalar_thread = f"cutlass.Int32(cute.arch.thread_idx()[{scalar_axis}])"
    lane_group_base = (
        "cutlass.Int32(0)"
        if not isinstance(lane_var, str)
        else f"cutlass.Int32({lane_var}) * cutlass.Int32({plan.groups_per_lane})"
    )
    tile_group = f"({scalar_thread}) // cutlass.Int32({plan.bn})"
    tile_n_local = f"({scalar_thread}) % cutlass.Int32({plan.bn})"
    mma_active = f"({tile_n_local}) < cutlass.Int32({_mma_active_n_threads(mma_impl)})"
    mma_thread_linear = f"{m_local} + ({tile_n_local}) * cutlass.Int32({plan.bm})"
    m_size = int(lhs_fake.shape[0])
    n_size = int(rhs_val.shape[1])
    k_size = serial_k_extent

    df = cg.device_function
    input_dtype_str = (
        "cutlass.Float16" if lhs_fake.dtype is torch.float16 else "cutlass.BFloat16"
    )
    acc_dtype_str = "cutlass.Float32"
    lhs_arg_name = df.tensor_arg(lhs_fake).name
    rhs_arg_name = df.tensor_arg(rhs_fake).name

    tiled_mma = df.new_var("direct_tiled_mma")
    thr_mma = df.new_var("direct_thr_mma")
    acc_frag = df.new_var("direct_acc_frag")
    smem_a_ptr = df.new_var("direct_smem_a")
    smem_a = df.new_var("direct_sA")
    smem_b_ptr = df.new_var("direct_smem_b")
    smem_b = df.new_var("direct_sB")
    smem_c_ptr = df.new_var("direct_smem_c")
    smem_c = df.new_var("direct_sC")
    tAsA = df.new_var("direct_tAsA")
    tBsB = df.new_var("direct_tBsB")
    tCsC = df.new_var("direct_tCsC")
    rA = df.new_var("direct_rA")
    rB = df.new_var("direct_rB")
    k_offset_var = df.new_var("direct_k_offset")
    result_var = df.new_var("direct_mma_result")

    for stmt in _make_tiled_mma_setup(
        mma_impl,
        tiled_mma,
        thr_mma,
        mma_thread_linear,
        input_dtype_str,
        acc_dtype_str,
        plan.bm,
        plan.bn,
    ):
        prefix.append(stmt)
    prefix.append(
        statement_from_string(
            f"{acc_frag} = cute.make_rmem_tensor("
            f"{tiled_mma}.partition_shape_C(({plan.bm}, {plan.bn})), {acc_dtype_str})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {plan.bm * plan.bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_a} = cute.make_tensor("
            f"{smem_a_ptr}, cute.make_layout(({plan.bm}, {plan.bk}), stride=({plan.bk}, 1)))"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b_ptr} = cute.arch.alloc_smem({input_dtype_str}, {plan.bn * plan.bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b} = cute.make_tensor("
            f"{smem_b_ptr}, "
            f"cute.make_layout(({plan.bn}, {plan.bk}), stride=({plan.bk}, 1)))"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c_ptr} = cute.arch.alloc_smem({acc_dtype_str}, {plan.bm * plan.bn}, alignment=128)"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c} = cute.make_tensor("
            f"{smem_c_ptr}, "
            f"cute.make_layout(({plan.bm}, {plan.bn}), stride=({plan.bn}, 1)))"
        )
    )
    cg.add_statement(statement_from_string(f"{result_var} = {acc_dtype_str}(0.0)"))
    cg.add_statement(
        statement_from_string(
            f"if {mma_active}:\n"
            f"    for _mma_i in range(cute.size({acc_frag})):\n"
            f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"for {k_offset_var} in range(0, {k_size}, {plan.bk}):\n"
            f"    if {mma_active} and ({tile_group}) == cutlass.Int32(0):\n"
            f"        for _load_i in range(({plan.bm * plan.bk} + {plan.bm * 2} - 1) // {plan.bm * 2}):\n"
            f"            _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({plan.bm * 2})\n"
            f"            if _flat < cutlass.Int32({plan.bm * plan.bk}):\n"
            f"                _row = _flat // cutlass.Int32({plan.bk})\n"
            f"                _col = _flat % cutlass.Int32({plan.bk})\n"
            f"                _gm = {m_tile_origin} + _row\n"
            f"                _gk = cutlass.Int32({load_plan.lhs_k_offset}) + cutlass.Int32({k_offset_var}) + _col\n"
            f"                {smem_a}[_row, _col] = ("
            f"{lhs_arg_name}[_gm, _gk] "
            f"if _gm < cutlass.Int32({m_size}) and _gk < cutlass.Int32({load_plan.lhs_k_offset + k_size}) "
            f"else {input_dtype_str}(0.0))\n"
            f"    cute.arch.sync_threads()\n"
            f"    for _n_group in range({plan.groups_per_lane}):\n"
            f"        if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"            for _load_i in range(({plan.bn * plan.bk} + {plan.bm * 2} - 1) // {plan.bm * 2}):\n"
            f"                _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({plan.bm * 2})\n"
            f"                if _flat < cutlass.Int32({plan.bn * plan.bk}):\n"
            f"                    _row = _flat // cutlass.Int32({plan.bk})\n"
            f"                    _col = _flat % cutlass.Int32({plan.bk})\n"
            f"                    _gn = cutlass.Int32({load_plan.rhs_n_offset}) + ({lane_group_base} + cutlass.Int32(_n_group)) * cutlass.Int32({plan.bn}) + _row\n"
            f"                    _gk = cutlass.Int32({load_plan.rhs_k_offset}) + cutlass.Int32({k_offset_var}) + _col\n"
            f"                    {smem_b}[_row, _col] = ("
            f"{rhs_arg_name}[_gk, _gn] "
            f"if _gn < cutlass.Int32({load_plan.rhs_n_offset + n_size}) and _gk < cutlass.Int32({load_plan.rhs_k_offset + k_size}) "
            f"else {input_dtype_str}(0.0))\n"
            f"        cute.arch.sync_threads()\n"
            f"        if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"            {tAsA} = {thr_mma}.partition_A({smem_a})\n"
            f"            {tBsB} = {thr_mma}.partition_B({smem_b})\n"
            f"            {rA} = cute.make_fragment_like({tAsA}, {input_dtype_str})\n"
            f"            {rB} = cute.make_fragment_like({tBsB}, {input_dtype_str})\n"
            f"            for _mma_i in range(cute.size({rA})):\n"
            f"                {rA}[_mma_i] = {tAsA}[_mma_i]\n"
            f"            for _mma_i in range(cute.size({rB})):\n"
            f"                {rB}[_mma_i] = {tBsB}[_mma_i]\n"
            f"            cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})\n"
            f"        cute.arch.sync_threads()"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"for _n_group in range({plan.groups_per_lane}):\n"
            f"    if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"        {tCsC} = {thr_mma}.partition_C({smem_c})\n"
            f"        for _mma_i in range(cute.size({tCsC})):\n"
            f"            {tCsC}[_mma_i] = {acc_frag}[_mma_i]\n"
            f"    cute.arch.sync_threads()\n"
            f"    if ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"        {result_var} = {smem_c}[{m_local}, {tile_n_local}]\n"
            f"    cute.arch.sync_threads()"
        )
    )
    return expr_from_string(result_var)


# ---- hl.dot entry point ----


def codegen_cute_mma_dot(state: CodegenState) -> object | None:
    """Generate MMA code for an hl.dot node.  Returns None to fall back."""
    from ..generate_ast import GenerateAST

    if not isinstance(state.codegen, GenerateAST):
        return None
    if state.codegen.current_grid_state is None:
        return None
    if state.fx_node is None:
        return None
    if not can_codegen_cute_mma_dot(state.fx_node):
        return None

    lhs_node = state.fx_node.args[0]
    rhs_node = state.fx_node.args[1]
    acc_expr = None
    if len(state.fx_node.args) > 2:
        acc_node = state.fx_node.args[2]
        if isinstance(acc_node, Node) and _is_zero_init_acc_node(acc_node):
            acc_expr = None
        else:
            acc_ast = state.ast_arg(2)
            if not (isinstance(acc_ast, ast.Constant) and acc_ast.value is None):
                acc_expr = acc_ast
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    result = _emit_mma_pipeline(
        state.codegen,
        lhs_node,
        rhs_node,
        acc_expr=acc_expr,
        fx_node=state.fx_node,
    )
    if result is None:
        if is_pure_matmul_role_lifecycle_config(state.device_function.config):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05_strategy='pure_matmul_role_lifecycle' requires hl.dot "
                "to lower through the tcgen05 K-loop path",
            )
        return None

    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    if isinstance(acc_proxy, FakeTensor) and acc_proxy.dtype != torch.float32:
        return cast_ast(result, acc_proxy.dtype)

    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None
    if isinstance(out_dtype_proxy, torch.dtype) and out_dtype_proxy != torch.float32:
        return cast_ast(result, out_dtype_proxy)

    return result
