"""Fused tcgen05 flash-attention forward codegen for the Helion CuTe backend.

This is a dedicated codegen entry, parallel to ``codegen_cute_mma`` in
``cute_mma.py``, that emits a numerically verified tcgen05 FA forward sequence.
Rather than threading the fused QK->softmax->PV dataflow through the single-GEMM
tcgen05 pipeline (which is welded to a once-per-tile commit + store-welded
epilogue), this path replaces the whole device body with a specialized flash
kernel when the flash detector fires and the ``HELION_CUTE_FLASH`` dev gate is
on.

Gate/scope (see ``backend._detect_attention_mma_loop``):
  * tile_b block size = 1 (one (batch, head) per CTA), tile_m = 128, tile_n = 128
  * head_dim in {64, 128}, fp16, mma_tiler 128x128 (the only legal one)
  * single-SM (CtaGroup.ONE), single 128-thread warpgroup. Stage 3 adds a
    multi-stage K/V TMA ring (``kv_stage`` deep, prologue + prefetch-ahead) so
    the K/V load latency overlaps the current tile's MMA + softmax. Still NO warp
    specialization / FA3 ping-pong.

The whole sequence (TMEM lifecycle, the two chained MMAs, P-over-S, online
softmax, alpha-rescale) is mirrored here statement-for-statement from the
verified standalone sequence.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
import dataclasses
import itertools
import math
import operator
import os
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast

import torch

from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...runtime.config import Config
from ..device_function import TensorArg
from .attention_plan import ALIBI_BIAS_KIND
from .attention_plan import CAUSAL_MASK_KIND
from .attention_plan import DOCUMENT_MASK_KIND
from .attention_plan import PREFIX_LM_MASK_KIND
from .attention_plan import RELATIVE_BIAS_KIND
from .attention_plan import SLIDING_WINDOW_MASK_KIND
from .attention_plan import SOFTCAP_KIND
from .attention_plan import TENSOR_BIAS_KIND
from .attention_plan import AttentionScorePlan

_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence


class FlashGraphOutputPlan(NamedTuple):
    """Host tensor names and static shape proven safe for flash replacement."""

    q_name: str
    k_name: str
    v_name: str
    o_name: str
    lse_name: str | None
    lse_log_base: str | None
    bias_names: tuple[str, ...]
    alibi_names: tuple[str, ...]
    document_names: tuple[str, ...]
    batch: int
    seq: int
    head_dim: int
    dtype: torch.dtype


def _flash_supported_io_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


def _flash_io_dtype_str(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "cutlass.Float16"
    if dtype is torch.bfloat16:
        return "cutlass.BFloat16"
    raise AssertionError(f"unsupported flash dtype: {dtype}")


def _small_biased_attention_score_plan_supported(
    score_plan: AttentionScorePlan,
    *,
    seq: int,
    head_dim: int,
    io_dtype: torch.dtype,
    has_lse: bool,
    bias_arg_count: int,
) -> bool:
    if has_lse:
        return False
    if seq != 128 or head_dim != 64 or not _flash_supported_io_dtype(io_dtype):
        return False
    if score_plan.modifier_kinds != (TENSOR_BIAS_KIND,):
        return False
    if len(score_plan.tensor_biases) != 1 or bias_arg_count != 1:
        return False
    if not math.isclose(
        score_plan.qk_scale_log2,
        math.log2(math.e) / math.sqrt(head_dim),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return False
    return math.isclose(
        score_plan.tensor_biases[0].scale_log2,
        math.log2(math.e),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _flash_store_host_tensor_name(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops
    from ...language._tracing_ops import _host_tensor

    if node.op != "call_function" or node.target is not memory_ops.store:
        return None
    tensor = node.args[0]
    if (
        isinstance(tensor, torch.fx.Node)
        and tensor.op == "call_function"
        and tensor.target is _host_tensor
        and tensor.args
        and isinstance(tensor.args[0], str)
    ):
        return tensor.args[0]
    return None


def _flash_load_host_tensor_name(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops
    from ...language._tracing_ops import _host_tensor

    if node.op != "call_function" or node.target is not memory_ops.load:
        return None
    tensor = node.args[0] if node.args else None
    if (
        isinstance(tensor, torch.fx.Node)
        and tensor.op == "call_function"
        and tensor.target is _host_tensor
        and tensor.args
        and isinstance(tensor.args[0], str)
    ):
        return tensor.args[0]
    return None


def _flash_source_load_node(node: torch.fx.Node) -> torch.fx.Node | None:
    from ...language import memory_ops

    if node.op == "call_function" and node.target is memory_ops.load:
        return node
    if node.op != "call_function" or not node.args:
        return None
    if node.target in (
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
    ) and isinstance(node.args[0], torch.fx.Node):
        return _flash_source_load_node(node.args[0])
    return None


def _flash_k_source_load_node(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or not node.args:
        return None
    if node.target is torch.ops.aten.permute.default:
        if len(node.args) < 2 or node.args[1] != [0, 2, 1]:
            return None
    elif node.target is torch.ops.aten.transpose.int:
        if len(node.args) < 3 or node.args[1] != 1 or node.args[2] != 2:
            return None
    else:
        return None
    if not isinstance(node.args[0], torch.fx.Node):
        return None
    return _flash_source_load_node(node.args[0])


def _flash_source_host_tensor_name(node: torch.fx.Node) -> str | None:
    load_node = _flash_source_load_node(node)
    if load_node is not None:
        return _flash_load_host_tensor_name(load_node)
    return None


class FlashRootLoopInfo(NamedTuple):
    q_name: str
    index_pair: tuple[torch.fx.Node, torch.fx.Node]


def _flash_is_block_symnode(node: torch.fx.Node, block_id: int) -> bool:
    from ...language._tracing_ops import _get_symnode

    return (
        node.op == "call_function"
        and node.target is _get_symnode
        and len(node.args) >= 1
        and node.args[0] == f"block_size_{block_id}"
    )


def _flash_is_inner_batch_index(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.sym_size.int
        and len(node.args) >= 2
        and isinstance(node.args[1], int)
        and node.args[1] == 0
    )


def _flash_store_value_lse_log_base(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.store:
        return None
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        return None
    if len(node.args) < 3 or not isinstance(node.args[2], torch.fx.Node):
        return None
    value = node.args[2]
    if value.op != "call_function" or value.target is not torch.ops.aten.add.Tensor:
        return None
    if value.kwargs.get("alpha", 1) != 1:
        return None
    if len(value.args) < 2:
        return None
    lhs, rhs = value.args[:2]
    log_targets = (torch.ops.aten.log.default, torch.ops.aten.log2.default)
    if isinstance(lhs, torch.fx.Node) and lhs.target in log_targets:
        log2_node = lhs
        max_node = rhs
    elif isinstance(rhs, torch.fx.Node) and rhs.target in log_targets:
        log2_node = rhs
        max_node = lhs
    else:
        return None
    if not isinstance(max_node, torch.fx.Node):
        return None
    if not log2_node.args or not isinstance(log2_node.args[0], torch.fx.Node):
        return None
    sum_node = log2_node.args[0]
    max_loop = _flash_phi_loop_result(max_node, 0)
    sum_loop = _flash_phi_loop_result(sum_node, 1)
    if max_loop is None or max_loop is not sum_loop:
        return None
    return "log2" if log2_node.target is torch.ops.aten.log2.default else "log"


def _flash_lse_log_base_for_scale(lse_scale: float) -> str | None:
    if math.isclose(lse_scale, 1.0, rel_tol=1e-6, abs_tol=1e-7):
        return "log2"
    if math.isclose(lse_scale, math.log(2.0), rel_tol=1e-6, abs_tol=1e-7):
        return "log"
    return None


def _flash_root_loop_info(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
) -> FlashRootLoopInfo | None:
    from ...language import memory_ops
    from ...language._tracing_ops import _for_loop

    root_loop_info: FlashRootLoopInfo | None = None
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not _for_loop:
                continue
            if len(node.args) < 4 or not isinstance(node.args[3], (list, tuple)):
                return None
            loop_inputs = node.args[3]
            if not loop_inputs or not isinstance(loop_inputs[0], torch.fx.Node):
                return None
            q_tile = loop_inputs[0]
            if q_tile.op != "call_function" or q_tile.target is not memory_ops.load:
                return None
            if (
                len(q_tile.args) < 4
                or q_tile.args[2] is not None
                or q_tile.args[3] is not None
            ):
                return None
            q_name = _flash_load_host_tensor_name(q_tile)
            if q_name is None:
                return None
            indices = q_tile.args[1] if len(q_tile.args) > 1 else None
            if not isinstance(indices, (list, tuple)) or len(indices) != 3:
                return None
            if not isinstance(indices[0], torch.fx.Node) or not isinstance(
                indices[1], torch.fx.Node
            ):
                return None
            if not _is_full_slice(indices[2]):
                return None
            if root_block_ids is not None:
                if len(root_block_ids) != 2:
                    return None
                if not _flash_is_block_symnode(
                    indices[0], int(root_block_ids[0])
                ) or not _flash_is_block_symnode(indices[1], int(root_block_ids[1])):
                    return None
            current_pair = (indices[0], indices[1])
            current_info = FlashRootLoopInfo(q_name, current_pair)
            if root_loop_info is None:
                root_loop_info = current_info
            elif (
                root_loop_info.q_name != q_name
                or root_loop_info.index_pair[0] is not current_pair[0]
                or root_loop_info.index_pair[1] is not current_pair[1]
            ):
                return None
    return root_loop_info


def _flash_attention_operand_names(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
) -> tuple[str, str, str] | None:
    root_info = _flash_root_loop_info(graphs, root_block_ids=root_block_ids)
    if root_info is None:
        return None

    k_names: set[str] = set()
    v_names: set[str] = set()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is torch.ops.aten.bmm.dtype:
                if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
                    return None
                k_load = _flash_k_source_load_node(node.args[1])
                if k_load is None or not _flash_kv_load_has_canonical_indices(
                    k_load, kv_block_id
                ):
                    return None
                k_name = _flash_load_host_tensor_name(k_load)
                if k_name is None:
                    return None
                k_names.add(k_name)
            elif node.target is torch.ops.aten.baddbmm.default:
                if len(node.args) < 3 or not isinstance(node.args[2], torch.fx.Node):
                    return None
                v_load = _flash_source_load_node(node.args[2])
                if v_load is None or not _flash_kv_load_has_canonical_indices(
                    v_load, kv_block_id
                ):
                    return None
                v_name = _flash_load_host_tensor_name(v_load)
                if v_name is None:
                    return None
                v_names.add(v_name)
    if len(k_names) != 1 or len(v_names) != 1:
        return None
    k_name = next(iter(k_names))
    v_name = next(iter(v_names))
    if len({root_info.q_name, k_name, v_name}) != 3:
        return None
    return root_info.q_name, k_name, v_name


def _is_full_slice(value: object) -> bool:
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _flash_store_has_canonical_indices(
    node: torch.fx.Node,
    rank: int,
    index_pair: tuple[torch.fx.Node, torch.fx.Node],
) -> bool:
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != rank:
        return False
    if indices[0] is not index_pair[0] or indices[1] is not index_pair[1]:
        return False
    if len(node.args) > 3 and node.args[3] is not None:
        return False
    return rank == 2 or _is_full_slice(indices[2])


def _flash_kv_load_has_canonical_indices(
    node: torch.fx.Node,
    kv_block_id: int | None,
) -> bool:
    from ...language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.load:
        return False
    if len(node.args) < 4 or node.args[2] is not None or node.args[3] is not None:
        return False
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return False
    if not isinstance(indices[0], torch.fx.Node) or not isinstance(
        indices[1], torch.fx.Node
    ):
        return False
    if not _flash_is_inner_batch_index(indices[0]):
        return False
    if kv_block_id is not None and not _flash_is_block_symnode(
        indices[1], int(kv_block_id)
    ):
        return False
    return _is_full_slice(indices[2])


def _flash_store_value_is_output(node: torch.fx.Node) -> bool:
    from ...language import memory_ops
    from ...language import view_ops

    if node.op != "call_function" or node.target is not memory_ops.store:
        return False
    if len(node.args) < 3 or not isinstance(node.args[2], torch.fx.Node):
        return False
    value = node.args[2]
    if (
        value.op == "call_function"
        and value.target is torch.ops.prims.convert_element_type.default
        and value.args
        and isinstance(value.args[0], torch.fx.Node)
    ):
        value = value.args[0]
    if value.op != "call_function" or value.target is not torch.ops.aten.div.Tensor:
        return False
    if len(value.args) < 2:
        return False
    acc_node, sum_view = value.args[:2]
    if not isinstance(acc_node, torch.fx.Node) or not isinstance(
        sum_view, torch.fx.Node
    ):
        return False
    acc_loop = _flash_phi_loop_result(acc_node, 2)
    if acc_loop is None:
        return False
    if sum_view.op != "call_function" or sum_view.target is not view_ops.subscript:
        return False
    if not sum_view.args or not isinstance(sum_view.args[0], torch.fx.Node):
        return False
    sum_loop = _flash_phi_loop_result(sum_view.args[0], 1)
    if sum_loop is not acc_loop:
        return False
    indices = sum_view.args[1] if len(sum_view.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return False
    return (
        _is_full_slice(indices[0]) and _is_full_slice(indices[1]) and indices[2] is None
    )


def _flash_phi_loop_result(node: torch.fx.Node, index: int) -> torch.fx.Node | None:
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _phi

    if node.op != "call_function" or node.target is not _phi:
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
        return None
    getitem = node.args[1]
    if getitem.op != "call_function" or getitem.target is not operator.getitem:
        return None
    if len(getitem.args) < 2 or getitem.args[1] != index:
        return None
    loop = getitem.args[0]
    if not isinstance(loop, torch.fx.Node):
        return None
    if loop.op != "call_function" or loop.target is not _for_loop:
        return None
    return loop


def _flash_host_tensor_stores(
    graphs: Iterable[GraphInfo],
) -> list[tuple[torch.fx.Node, str]]:
    stores: list[tuple[torch.fx.Node, str]] = []
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            host_tensor_name = _flash_store_host_tensor_name(node)
            if host_tensor_name is not None:
                stores.append((node, host_tensor_name))
    return stores


def _flash_graph_host_tensors(graphs: Iterable[GraphInfo]) -> dict[str, torch.Tensor]:
    from ...language._tracing_ops import _host_tensor

    host_tensors: dict[str, torch.Tensor] = {}
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not _host_tensor:
                continue
            if not node.args or not isinstance(node.args[0], str):
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                host_tensors.setdefault(node.args[0], value)
    return host_tensors


def _flash_graph_output_plan_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan | None = None,
) -> FlashGraphOutputPlan | None:
    graphs = tuple(graphs)
    host_tensors = _flash_graph_host_tensors(graphs)
    flash_3d_tensors = {
        name: value
        for name, value in host_tensors.items()
        if value.ndim == 3
        and _flash_supported_io_dtype(value.dtype)
        and value.is_contiguous()
    }
    operand_names = _flash_attention_operand_names(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
    )
    if operand_names is None:
        return None
    q_name, k_name, v_name = operand_names
    if not all(name in flash_3d_tensors for name in operand_names):
        return None
    bias_names = (
        tuple(
            modifier.tensor_name
            for modifier in score_plan.tensor_biases
            if modifier.tensor_name is not None
        )
        if score_plan is not None
        else ()
    )
    if len(bias_names) != len(set(bias_names)):
        return None
    alibi_names = (
        tuple(
            modifier.tensor_name
            for modifier in score_plan.alibi_biases
            if modifier.tensor_name is not None
        )
        if score_plan is not None
        else ()
    )
    if len(alibi_names) != len(set(alibi_names)):
        return None
    document_names = (
        tuple(
            modifier.tensor_name
            for modifier in score_plan.document_masks
            if modifier.tensor_name is not None
        )
        if score_plan is not None
        else ()
    )
    if len(document_names) != len(set(document_names)):
        return None

    root_info = _flash_root_loop_info(graphs, root_block_ids=root_block_ids)
    if root_info is None:
        return None
    index_pair = root_info.index_pair
    store_nodes = _flash_host_tensor_stores(graphs)
    if not store_nodes:
        return None

    o_store_names: list[str] = []
    lse_store_names: list[str] = []
    lse_log_bases: list[str] = []
    for node, name in store_nodes:
        value = host_tensors.get(name)
        if value is None:
            return None
        if (
            value.ndim == 3
            and _flash_supported_io_dtype(value.dtype)
            and value.is_contiguous()
        ):
            if not _flash_store_has_canonical_indices(node, 3, index_pair):
                return None
            if not _flash_store_value_is_output(node):
                return None
            o_store_names.append(name)
            continue
        if value.ndim == 2 and value.dtype == torch.float32 and value.is_contiguous():
            if not _flash_store_has_canonical_indices(node, 2, index_pair):
                return None
            lse_log_base = _flash_store_value_lse_log_base(node)
            if lse_log_base is None:
                return None
            lse_store_names.append(name)
            lse_log_bases.append(lse_log_base)
            continue
        return None

    if len(o_store_names) != 1 or len(set(o_store_names)) != 1:
        return None
    if len(lse_store_names) > 1 or len(set(lse_store_names)) != len(lse_store_names):
        return None
    if len(set(lse_log_bases)) != len(lse_log_bases):
        return None

    o_name = o_store_names[0]
    o_value = flash_3d_tensors.get(o_name)
    if o_value is None or o_name in operand_names:
        return None
    io_dtype = o_value.dtype
    batch = int(o_value.shape[0])
    seq = int(o_value.shape[1])
    head_dim = int(o_value.shape[2])
    if batch <= 0 or seq <= 0:
        return None
    if head_dim not in (64, 128):
        return None
    if seq % 128 != 0:
        return None

    if set(flash_3d_tensors) != {q_name, k_name, v_name, o_name, *bias_names}:
        return None
    for name in operand_names:
        value = flash_3d_tensors[name]
        if (
            int(value.shape[0]) != batch
            or int(value.shape[1]) != seq
            or int(value.shape[2]) != head_dim
            or value.dtype != io_dtype
        ):
            return None
    for name in bias_names:
        value = flash_3d_tensors.get(name)
        if (
            value is None
            or int(value.shape[0]) != batch
            or int(value.shape[1]) != seq
            or int(value.shape[2]) != seq
            or value.dtype != io_dtype
        ):
            return None
    alibi_modifiers = score_plan.alibi_biases if score_plan is not None else ()
    for modifier in alibi_modifiers:
        if modifier.tensor_name is None:
            return None
        value = host_tensors.get(modifier.tensor_name)
        if (
            value is None
            or value.ndim != 1
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or int(value.shape[0]) <= 0
        ):
            return None
        alibi_count = int(value.shape[0])
        if modifier.index_mode == "identity":
            if modifier.index_divisor is not None or alibi_count != batch:
                return None
        elif modifier.index_mode == "mod":
            if modifier.index_divisor != alibi_count or batch % alibi_count != 0:
                return None
        else:
            return None
    document_modifiers = score_plan.document_masks if score_plan is not None else ()
    for modifier in document_modifiers:
        if modifier.tensor_name is None:
            return None
        value = host_tensors.get(modifier.tensor_name)
        if (
            value is None
            or value.ndim != 2
            or value.dtype not in (torch.int32, torch.int64)
            or not value.is_contiguous()
            or int(value.shape[0]) <= 0
            or int(value.shape[1]) != seq
        ):
            return None
        document_batch = int(value.shape[0])
        if modifier.index_mode == "identity":
            if modifier.index_divisor is not None or document_batch != batch:
                return None
        elif modifier.index_mode == "floordiv":
            if batch % document_batch != 0:
                return None
            if modifier.index_divisor != batch // document_batch:
                return None
        else:
            return None

    lse_name = lse_store_names[0] if lse_store_names else None
    lse_log_base = lse_log_bases[0] if lse_log_bases else None
    if lse_log_base is not None and score_plan is not None:
        if lse_log_base != _flash_lse_log_base_for_scale(score_plan.lse_scale):
            return None
    lse_candidate_names = {
        name
        for name, value in host_tensors.items()
        if value.ndim == 2
        and value.dtype == torch.float32
        and value.is_contiguous()
        and int(value.shape[0]) == batch
        and int(value.shape[1]) == seq
    }
    if lse_name is None:
        if lse_candidate_names:
            return None
    elif lse_candidate_names != {lse_name}:
        return None
    allowed_host_names = {
        q_name,
        k_name,
        v_name,
        o_name,
        *bias_names,
        *alibi_names,
        *document_names,
    }
    if lse_name is not None:
        allowed_host_names.add(lse_name)
    if set(host_tensors) != allowed_host_names:
        return None
    return FlashGraphOutputPlan(
        q_name,
        k_name,
        v_name,
        o_name,
        lse_name,
        lse_log_base,
        bias_names,
        alibi_names,
        document_names,
        batch,
        seq,
        head_dim,
        io_dtype,
    )


def flash_attention_graph_lse_plan_valid_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan | None = None,
) -> bool:
    return (
        _flash_graph_output_plan_from_graphs(
            graphs,
            root_block_ids=root_block_ids,
            kv_block_id=kv_block_id,
            score_plan=score_plan,
        )
        is not None
    )


def flash_attention_graph_small_biased_candidate_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> bool:
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    if graph_plan is None:
        return False
    return _small_biased_attention_score_plan_supported(
        score_plan,
        seq=graph_plan.seq,
        head_dim=graph_plan.head_dim,
        io_dtype=graph_plan.dtype,
        has_lse=graph_plan.lse_name is not None,
        bias_arg_count=len(graph_plan.bias_names),
    )


def flash_attention_graph_lse_plan_valid(
    df: DeviceFunction,
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan | None = None,
) -> bool:
    """Detector-time LSE/aux-output gate using only FX graph metadata.

    ``DeviceFunction.arguments`` is still incomplete when the flash detector
    runs, so the detector cannot call ``flash_attention_tensor_plan`` directly.
    This mirrors its output-tensor disambiguation from ``_host_tensor`` metadata:
    exactly one canonical fp16/bf16 ``(B, S, D)`` output store is required, and a lone
    fp32 ``(B, S)`` tensor is accepted only when the graph stores canonical
    ``m_i + log2(l_i)`` to it. Any unrelated output keeps the scalar path so
    ``mma_mode`` is not set on a body codegen will later reject.
    """
    return flash_attention_graph_lse_plan_valid_from_graphs(
        df.codegen.codegen_graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )


def _flash_kv_stage(head_dim: int) -> int:
    """Number of K/V TMA ring stages (Stage 3 multi-stage pipelining).

    Each K/V ring slot is ``128 * head_dim * 2`` bytes; head_dim 64 -> 16 KB,
    head_dim 128 -> 32 KB. Default kv_stage=2 (K+V = 4 slots) keeps two K/V tile
    loads in flight to overlap the load latency with the softmax + MMA compute,
    while leaving SMEM headroom for >=1 CTA/SM. Overridable via
    ``HELION_CUTE_FLASH_KV_STAGE`` for tuning.

    Stage 4 (s_stage==2 warp spec) needs the QK look-ahead one tile further than
    the V consumption, so K/V must be at least kv_stage=3 deep to keep K(k+1)
    resident while V(k) is still in flight. Bumped accordingly when warp spec is
    on (still overridable).
    """
    override = os.environ.get("HELION_CUTE_FLASH_KV_STAGE")
    if override is not None:
        # Clamp to >=1: a 0/negative stage count would build zero-byte SMEM
        # rings and empty pipelines (a confusing compile hang).
        return max(1, int(override))
    if _flash_s_stage() == 2:
        return 3
    # head_dim 128 slots are 2x the bytes; keep 2 to stay within SMEM/occupancy.
    return 2


def _flash_persistent() -> bool:
    """Whether to emit a static-persistent scheduler (grid capped at num_SMs,
    each CTA strides over a flat tile-id range).

    Default ON: amortizes per-CTA setup/teardown (TMEM alloc, mbar init,
    descriptor prefetch, launch/exit handshake) across many work tiles and hides
    the per-tile pipeline fill/drain (attacks the 43.8% CTA-barrier stall profile
    at 1 CTA/SM); validated +1.5-4.5% real SDPA with no regression. Set
    ``HELION_CUTE_FLASH_PERSISTENT=0`` (or "false"/"off") to fall back to the flat
    one-tile-per-CTA grid.
    """
    override = os.environ.get("HELION_CUTE_FLASH_PERSISTENT")
    if override is None:
        return True
    return override.lower() not in ("0", "", "false", "off")


def _flash_s_stage() -> int:
    """Number of S (QK accumulator) TMEM buffers (Stage 4 warp specialization).

    ``s_stage == 2`` enables the 2-warpgroup producer/consumer split with a
    DOUBLE-BUFFERED S accumulator (S0 @ TMEM col 0, S1 @ col 128, O @ col 256):
    warp 0 (producer) issues QK(k+1) into S[(k+1)%2] WHILE the consumer
    warpgroup runs softmax(k) on S[k%2], so the QK MMA overlaps the softmax
    (fills the ``wait`` bubble; see STAGE4 in STAGE2_PROGRESS.md). ``s_stage == 1``
    (default) is the single-warpgroup Stage-3 path. Overridable via
    ``HELION_CUTE_FLASH_S_STAGE``.
    """
    override = os.environ.get("HELION_CUTE_FLASH_S_STAGE")
    if override is not None:
        return 2 if int(override) >= 2 else 1
    # Default ON: the warp-spec double-buffered-S overlap (the Stage-4 win).
    return 2


# Producer/consumer register budgets for the Stage-4 (s_stage==2) warp-spec
# split. The consumer warpgroup carries the softmax/rescale/epilogue state; the
# producer warp + the 3 empty warps release theirs (raising occupancy).
_FLASH_NUM_REGS_PRODUCER = 96
_FLASH_NUM_REGS_CONSUMER = 184


@dataclasses.dataclass(frozen=True)
class FlashAttentionConfig:
    """Resolved topology config for the CuTe flash-attention codegen.

    Stage A1 captures the current env-derived knobs verbatim. Later stages add
    fields (warp-role map, exp2 impl, tmem plan, ...), each defaulting to the
    current kernel's behavior, so FA4 becomes one value of this config.
    """

    s_stage: int
    kv_stage: int
    persistent: bool
    persistent_ctas_per_sm: int
    recompute_tile_coords: bool
    num_regs_producer: int
    num_regs_consumer: int

    # Stage B/C topology fields (consumed by the fa4 emitter)
    topology: str = "ws_overlap"
    num_softmax_warpgroups: int = 1
    num_correction_warps: int = 0
    num_mma_warps: int = 0
    num_load_warps: int = 0
    num_epilogue_warps: int = 0
    q_tile_count: int = 1
    acc_stage: int = 1
    epi_stage: int = 1
    # exp2 defaults to the FMA/XU pipe-split (split f8/r2): measured +2.9pp hd64
    # (65.5->68.4%) / +1.2pp hd128 of SDPA at correctness identical to the
    # hardware-XU path (max_abs_err 2.44e-4 == fp16 rounding floor). The split
    # shortens the t2r->exp2->r2t consumer critical chain by offloading ~1-in-8
    # exp2 pairs onto the FMA pipe. ``HELION_CUTE_FLASH_EXP2_IMPL=xu`` reverts.
    exp2_impl: str = "split"
    e2e_freq: int = 8
    e2e_res: int = 2
    e2e_schedule: str = "8/2"
    # Causal diagonal/masked tiles can use a different exp2 schedule from the
    # fully unmasked tiles. ``inherit`` preserves the main e2e schedule.
    masked_e2e_schedule: str = "inherit"
    masked_e2e_freq: int = 8
    masked_e2e_res: int = 2
    e2e_offset: int = 0
    e2e_offset0: int = 0
    tmem_plan: str = "separate"
    tmem_s_to_p_offset: int = 0
    mma_interleave: bool = False
    # fa4 Stage 2b: issue the MMA-warp QK/PV via the FA4 ``gemm_ptx_partial`` (one
    # inline-asm region with literal-immediate descriptors) instead of cute.gemm.
    # Fits the MMA warp at 48 regs (cute.gemm spills ~116 STL/133 LDL) AND folds the
    # pfor2 wait inside the PV issue stream. Default ON for the fa4 topology.
    mma_ptx: bool = True
    # fa4 Step 2: chunked-t2r ("disc") softmax body. The whole-row ("sp") body keeps
    # a 128-f32 row resident across the exp2/convert + the deferred row-sum, which
    # overruns the 200-reg grant and spills the row (the LDL.LU.64 [R1+..] reloads).
    # The chunked-t2r body loads/exp/converts/stores/sums ONE 32-elem column chunk
    # at a time so peak live = one fragment, matching the zero-spill structure
    # used by the verified FA4-style prototype. Default ON for fa4;
    # HELION_CUTE_FLASH_SOFTMAX_DISC=0 reverts to the whole-row body for A/B.
    softmax_disc: bool = True
    # fa4 Step 3 (L1 lever): software-pipeline depth of the disc PASS2 chunk loop.
    # 1 = the current serial-per-chunk zero-spill body (default, byte-identical);
    # >=2 = prefetch chunk ci+depth's t2r before chunk ci's exp2 burst so the next
    # chunk's TMEM-read latency overlaps the XU exp2 (attacks long_scoreboard). The
    # prefetch is PINNED by a side-effecting inline-asm scheduling barrier so ptxas
    # cannot sink it adjacent to its consumer (the plain prefetch was measured neutral
    # in the spike/campaign). HELION_CUTE_FLASH_DISC_PIPE selects the depth.
    disc_pipe_depth: int = 1
    # FA4 staged-P handshake. When enabled, softmax releases the first 3/4 of P
    # early on pfor and releases the final chunk on pfor2, allowing the MMA warp
    # to start PV before the whole P tile has been stored. Upstream FA4 disables
    # this for dense non-causal hd64, where the extra split barrier lost to the
    # simpler full-P release.
    split_p_arrive: bool = True
    # TMEM P-store repetition for the FA4 softmax P path. Upstream FA4 exposes
    # this as a useful hd64 lever: 16 preserves the original 4-way staged-P
    # chunks, while 32 halves the r2t chunk count and is searched on dense hd64.
    p_store_repetition: int = 16
    # TMEM S-load repetition for the FA4 softmax path. FA4 exposes this as
    # FA4_S_LOAD_REP; 32 is the upstream default, while 16 is useful to keep
    # searchable for chunked softmax codegen variants.
    s_load_repetition: int = 32
    # QK PTX descriptor mode. ``False`` matches FA4's default: declare one Q
    # descriptor set in the MMA warp and mutate it by a Q-stage stride per MMA.
    # ``True`` declares one descriptor set per Q stage and issues MMAs without
    # per-call descriptor mutation, matching the FA4 static-Q descriptor lever.
    precompute_qk_desc: bool = False
    # Initial TMA prologue order for the FA4 load warp. 0 preserves Helion's
    # original Q0/K/Q1/V order; 4 keeps the measured K/Q0/Q1/V variant for
    # long-shape seeds. The remaining values are manual/autotune experiments.
    first_load_order: int = 0
    # KV traversal order for FA4. Upstream FA4 walks dense non-causal KV blocks
    # from the end toward the beginning; Helion's original path was ascending.
    kv_order: str = "ascending"
    # fa4 Step 3 (Lever A): epilogue TMA-store. The committed fa4 corr epilogue does a
    # per-thread STG.E.128 store of the rescaled-fp16 O straight to gmem (coord->linear
    # address division per thread = the epilogue IMAD~20/MOV~34). With this ON, the corr
    # warp instead stages the rescaled-fp16 O to smem (REUSES the sQ region -- Q is fully
    # consumed by the epilogue, so no extra smem -- critical at hd128 where the kv_stage=2
    # smem is near the 232 KB cap) then a TMA bulk store (cp.async.bulk.tensor S2G)
    # computes the address ONCE via the descriptor. Default ON for hd128 fa4 and OFF
    # for hd64; HELION_CUTE_FLASH_EPI_TMA overrides. fa4-only.
    epi_tma: bool = False
    # FA4's non-TMA-O epilogue still uses a dedicated epilogue warp: correction
    # writes sO, then that warp vector-stores sO to gmem. This is distinct from
    # Helion's older direct correction-warp STG path.
    epi_stg: bool = False
    # STG epilogue store shape. ``slice`` keeps one 16B row-slice live at a
    # time; ``whole`` matches FA4's whole-tile smem->register copy before gmem
    # stores and is exposed for hd64 STG tuning.
    epi_stg_store: str = "slice"
    # STG epilogue gmem tile shape. ``stage`` builds one 128-row tile per stage;
    # ``pair`` builds FA4's 256-row pair tile once and slices the stage mode.
    epi_stg_gmem: str = "stage"
    # fa4 softmax: FlashAttention-4's ``rescale_threshold`` (alpha-pin). When the
    # running row-max grows by less than this many log2-units on a steady tile
    # (``scale_log2 * (old_max - new_max) >= -rescale_threshold``), FA4 KEEPS the old
    # max and pins ``alpha = 1.0`` instead of advancing the max + rescaling O. The
    # correction warp's ``vote_ballot(alpha < 1.0)`` is then false for the whole warp
    # -> the O-rescale (t2r O / mul / r2t O / fence) is SKIPPED. That t2r->r2t->fence on
    # the O accumulator sits ON the correction->PV critical path (the next PV waits on
    # ``pfor``), so skipping it deletes a dependency edge -> measured -3.8% hd64 /
    # -5.3% hd128 wall-clock, correctness IMPROVED (1.22e-4). FA4 ships this at 8.0 for
    # fp16/bf16 and 0.0 (disabled = always-rescale = our prior behavior) for fp8.
    # 0.0 emits the prior body byte-identically. Dtype-gated default in
    # ``resolve_flash_config``; HELION_CUTE_FLASH_RESCALE_THRESHOLD overrides. The
    # autotuner can refine the threshold for fa4 and ws_overlap shapes.
    rescale_threshold: float = 0.0
    # Experimental FA4_SKIP_RESCALE_STATS lever. The resolver currently clamps
    # this off because dropping per-KV alpha handoffs is only correct if every
    # later tile is known to stay on the pinned exponent base.
    skip_rescale_stats: bool = False
    # TMEM O-rescale chunk width. hd64 defaults to 32 cols because 64 cols pushes
    # some FA4 attention shapes over the ptxas register target, but manual
    # experiments can opt into 64 cols to test reduced loop/address overhead.
    rescale_chunk_cols: int = 0
    # FA4 setmaxregister budgets for softmax and correction warpgroups. Softmax
    # regs are part of the characterized autotune surface; correction regs stay
    # manual until we have a validated multi-value envelope.
    softmax_regs: int = 200
    corr_regs: int = 64
    # Register budget for load/epilogue/other warpgroup roles. This is separate
    # from softmax/correction grants because FA4's non-TMA dense path prefers 32,
    # while Helion's TMA-O dense path has measured best around 48.
    other_regs: int = 48
    # Final O correction-epilogue chunk width for the FA4 sO staging paths.
    # FA4's dense hd64 default is 8, while earlier Helion TMA-O work used 16.
    corr_tile_size: int = 16
    # ws_overlap packed row reductions. The default follows
    # HELION_CUTE_FLASH_PACKED_REDUCE and the autotuner can enable it per shape.
    packed_reduce: bool = False
    # Small contiguous [BH, 128, 64] biased-attention SIMT path. Kept as a config
    # knob so benchmarks/autotune can opt back to the general flash emitter.
    small_biased: bool = True
    # Causal FA4 launch-order swizzle. 0 keeps the shape-derived L2 heuristic;
    # positive values force the batch/head interleave group size.
    causal_lpt_swizzle: int = 0
    # Causal FA4 KV traversal order. ``ascending`` preserves Helion's original
    # left-to-right stream; ``descending`` matches upstream FA4's diagonal-first
    # stream and is searchable because it is shape-sensitive.
    causal_kv_order: str = "ascending"
    # Warp-role assignment inside the last FA4 warpgroup. ``helion`` preserves
    # the original transcription; ``fa4`` matches upstream's load/epilogue warp
    # ordering.
    role_map: str = "helion"
    # Dense non-causal FA4 can use SM100 CtaGroup.TWO: two CTAs cooperate on one
    # logical M tile and each CTA owns one rank-local half. This is gated because
    # the raw-barrier FA4 transcription needs separate cluster-aware handshakes.
    use_2cta_instrs: bool = False
    # Dense non-causal hdim64 FA4 can also cluster adjacent independent CTAs
    # without using CtaGroup.TWO. Each CTA still runs a normal 128-row MMA tile,
    # but cluster-rank 0/1 take neighboring Q-tile pairs to improve co-scheduling
    # and K/V cache reuse while preserving the proven single-CTA barrier graph.
    use_cga2_local_cta: bool = False
    # Dense non-causal hdim64 FA4 uses Blackwell Cluster Launch Control to turn
    # the full problem grid into a dynamic-persistent schedule. This matches the
    # tuned FA4 scheduler path and requires a PDL launch.
    use_clc_scheduler: bool = False
    # Optional CLC launch decomposition. 0 keeps Helion's historical flattened
    # batch/head CLC axis; a positive divisor of num_bh launches CLC as
    # (m_pairs, heads_per_batch, batch_count), matching FA4's dense MHA geometry.
    clc_heads_per_batch: int = 0
    # Flash CLC launch-control options. FA4 forward uses a CLC async pipeline but
    # does not pass PDL at launch; keep PDL searchable for A/B.
    clc_use_pdl: bool = False
    clc_stages: int = 1
    # Build TMA gmem partitions inside the FA4 load/epilogue roles from
    # per-work-item local tiles. This matches FA4's structure and avoids keeping
    # whole-problem TMA partition tensors live through the common setup.
    local_tma_partition: bool = False
    # Treat dense flattened ``(z*h, s, d)`` views as 4D TMA tensor maps
    # ``(s, d, h, z)`` / ``(d, s, h, z)`` when the original contiguous 4D input
    # shape is recoverable. This matches FA4's dense MHA tensor-map rank.
    tensor_4d_tma: bool = False
    # Causal descending KV can run the short masked diagonal prefix separately
    # from the hot unmasked suffix, removing a per-KV branch from most tiles.
    causal_loop_split: bool = False


def _flash_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var; treat '1'/'true'/'on' (case-insensitive) as True."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "on")


def _flash_e2e_schedule_default(topology: str, head_dim: int) -> str:
    """Return the default exp2 schedule name for a flash shape."""
    return "16/4" if topology == "fa4" and head_dim == 64 else "8/2"


def _flash_causal_hd64_seed_num_kv_supported(num_kv: int | None) -> bool:
    return isinstance(num_kv, int) and num_kv >= 32 and (num_kv & (num_kv - 1)) == 0


_FLASH_CAUSAL_HD64_LONG_AUTOTUNE_MIN_KV = 4096


def _flash_causal_hd64_seed_params(num_kv: int) -> tuple[int, int, int, int]:
    """Return best-known causal hd64 FA4 knobs as (offset, disc, regs, lpt)."""
    if num_kv <= 64:
        e2e_offset = 0
        disc_pipe = 2
        softmax_regs = 200
        lpt_swizzle = 8
    elif num_kv <= 128:
        e2e_offset = 0
        disc_pipe = 2
        softmax_regs = 192
        lpt_swizzle = 8
    elif num_kv <= 256:
        e2e_offset = 1
        disc_pipe = 4
        softmax_regs = 184
        lpt_swizzle = 4
    else:
        e2e_offset = 9
        disc_pipe = 4
        softmax_regs = 184
        lpt_swizzle = 1
    return e2e_offset, disc_pipe, softmax_regs, lpt_swizzle


def _flash_causal_hd64_seed_masked_schedule(num_kv: int) -> str:
    return "inherit" if num_kv <= 64 else "16/4"


def _flash_causal_hd64_seed_role_map(num_kv: int) -> str:
    if num_kv == 128:
        return "fa4"
    return "helion"


def _flash_causal_hd64_seed_epi_tma(num_kv: int) -> bool:
    return num_kv <= 128


def _flash_causal_hd64_seed_rescale_chunk_cols(num_kv: int) -> int:
    return 16 if num_kv <= 256 else 32


def _flash_causal_hd64_seed_offset0(num_kv: int) -> int:
    if num_kv <= 64:
        return 1
    if num_kv <= 128:
        return 0
    if num_kv <= 256:
        return 11
    return 3


def _flash_causal_hd64_kv_order_weights(num_kv: int) -> dict[str, float]:
    return {"descending": 4.0, "ascending": 1.0}


def _flash_causal_hd64_seed_kv_order(num_kv: int) -> str:
    weights = _flash_causal_hd64_kv_order_weights(num_kv)
    return "descending" if weights["descending"] > weights["ascending"] else "ascending"


def _flash_parse_e2e_schedule(schedule: str, fallback: str) -> tuple[str, int, int]:
    """Map an exp2 schedule string to ``(exp2_impl, e2e_freq, e2e_res)``."""

    def _parse(candidate: str) -> tuple[str, int, int] | None:
        candidate = candidate.strip().lower()
        if candidate == "xu":
            return "xu", 8, 0
        if "/" not in candidate:
            return None
        freq_str, res_str = candidate.split("/", 1)
        try:
            freq = int(freq_str)
            res = int(res_str)
        except ValueError:
            return None
        if freq <= 0 or res < 0 or res >= freq:
            return None
        return "split", freq, res

    parsed = _parse(schedule)
    if parsed is not None:
        return parsed
    parsed = _parse(fallback)
    if parsed is not None:
        return parsed
    return "split", 8, 2


def _flash_e2e_schedule_name(exp2_impl: str, e2e_freq: int, e2e_res: int) -> str:
    if exp2_impl != "split" or e2e_res == 0:
        return "xu"
    return f"{e2e_freq}/{e2e_res}"


def _flash_masked_e2e_schedule_params(
    schedule: str,
    fallback_schedule: str,
    base_e2e_freq: int,
    base_e2e_res: int,
) -> tuple[str, int, int]:
    """Return canonical masked-tile schedule fields.

    ``inherit`` is the default and reuses the unmasked schedule. Other accepted
    values are intentionally the small characterized schedule set.
    """
    candidate = schedule.strip().lower()
    if candidate in ("", "inherit"):
        return "inherit", base_e2e_freq, base_e2e_res
    if candidate not in ("xu", "16/4", "8/2"):
        return "inherit", base_e2e_freq, base_e2e_res
    exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
        candidate, fallback_schedule
    )
    _exp2_impl, e2e_freq, e2e_res, e2e_schedule = _flash_normalize_e2e_params(
        exp2_impl, e2e_freq, e2e_res, fallback_schedule
    )
    return e2e_schedule, e2e_freq, e2e_res


def _flash_e2e_offset_period(
    e2e_freq: int,
    e2e_res: int,
    masked_e2e_freq: int,
    masked_e2e_res: int,
) -> int:
    """Return the modulo period needed by all active split-exp2 pass2 paths."""
    period = 0
    if e2e_res > 0:
        period = max(period, e2e_freq)
    if masked_e2e_res > 0:
        period = max(period, masked_e2e_freq)
    return period


def _flash_normalize_e2e_offset(value: int, default: int, period: int) -> int:
    if period <= 0:
        return 0
    if value < 0:
        return default % period
    return value % period


def _flash_normalize_e2e_params(
    exp2_impl: str, e2e_freq: int, e2e_res: int, fallback_schedule: str
) -> tuple[str, int, int, str]:
    """Validate legacy exp2 overrides and return canonical schedule fields."""
    fallback_impl, fallback_freq, fallback_res = _flash_parse_e2e_schedule(
        fallback_schedule, "8/2"
    )
    if exp2_impl != "split":
        # The FA4 disc path consumes only freq/res, so canonicalize any hardware-XU
        # override to an all-XU cadence instead of merely changing the label.
        return "xu", max(e2e_freq, 1), 0, "xu"
    if e2e_freq <= 0 or e2e_res <= 0 or e2e_res >= e2e_freq:
        return (
            fallback_impl,
            fallback_freq,
            fallback_res,
            _flash_e2e_schedule_name(fallback_impl, fallback_freq, fallback_res),
        )
    return (
        "split",
        e2e_freq,
        e2e_res,
        _flash_e2e_schedule_name("split", e2e_freq, e2e_res),
    )


def resolve_flash_config(
    head_dim: int,
    num_kv: int,
    config: Mapping[str, object] | None = None,
    dtype: torch.dtype = torch.float16,
    *,
    is_causal: bool = False,
    prefer_packed_reduce: bool = False,
) -> FlashAttentionConfig:
    """Resolve the flash-attention topology config from shape, env vars and config.

    When *config* is supplied and the autotuner has enabled the flash search
    surface (``cute_flash_search_enabled``), the autotunable knobs (see
    ``FLASH_CONFIG_KEYS`` / ``flash_autotune_fragments``) are read from the
    config Mapping; any knob the config does not carry falls back to the env-var
    resolution below, so behavior is byte-identical when the key is absent (the
    default-off gating path) or unset.
    """

    def _cfg(key: str) -> object | None:
        if config is None:
            return None
        return config.get(key)

    s_stage = _flash_s_stage()
    kv_stage = _flash_kv_stage(head_dim)
    if num_kv < 2:
        s_stage = 1
        kv_stage = 1
    # Config overrides for the warp-spec / KV-ring depths (autotune knobs).
    s_stage_cfg = _cfg(FLASH_S_STAGE_KEY)
    if s_stage_cfg is not None and num_kv >= 2:
        s_stage = int(s_stage_cfg)  # type: ignore[arg-type]
    kv_stage_cfg = _cfg(FLASH_KV_STAGE_KEY)
    if kv_stage_cfg is not None:
        kv_stage = int(kv_stage_cfg)  # type: ignore[arg-type]
    persistent = _flash_persistent() and s_stage == 2
    persistent_cfg = _cfg(FLASH_PERSISTENT_KEY)
    if persistent_cfg is not None:
        persistent = bool(persistent_cfg) and s_stage == 2
    if is_causal:
        persistent = False
    # Stage B/C topology fields. fa4 is the default for shapes whose sequence
    # length is a multiple of 256 (the fa4 body handles two 128-row Q tiles per
    # work item); otherwise fall back to ws_overlap.
    topology_default = "fa4" if num_kv % 2 == 0 else "ws_overlap"
    topology = os.environ.get("HELION_CUTE_FLASH_TOPOLOGY", topology_default)
    # Config override: the autotuner picks the topology. Stale/ineligible fa4
    # values can appear for best-config cache transfer, but are clamped below.
    topology_cfg = _cfg(FLASH_TOPOLOGY_KEY)
    if topology_cfg is not None:
        topology = str(topology_cfg)
    if topology not in ("ws_overlap", "fa4"):
        topology = "ws_overlap"
    if topology == "fa4" and num_kv % 2 != 0:
        topology = "ws_overlap"
    long_dense_hd64_fa4 = (
        topology == "fa4" and not is_causal and head_dim == 64 and num_kv >= 64
    )
    causal_hd64_fa4 = topology == "fa4" and is_causal and head_dim == 64
    causal_hd64_seeded_fa4 = (
        causal_hd64_fa4 and _flash_causal_hd64_seed_num_kv_supported(num_kv)
    )
    (
        causal_e2e_offset_default,
        causal_disc_pipe_default,
        causal_softmax_regs_default,
        causal_lpt_swizzle_default,
    ) = _flash_causal_hd64_seed_params(num_kv)
    # The fa4 topology persistence (Stage 2a) follows the HELION_CUTE_FLASH_PERSISTENT
    # env knob (default ON) WITHOUT the ws-only s_stage==2 gate: the fa4 device body
    # carries the raw-mbarrier phases (s_full/pfor/pfor2/o_full) across work-items in
    # its own strided loop, so it does not depend on the cutlass-pipeline s_stage path.
    if topology == "fa4":
        if persistent_cfg is not None:
            persistent = bool(persistent_cfg) and not is_causal
        else:
            persistent = _flash_persistent() and not is_causal
        if (
            (is_causal or long_dense_hd64_fa4)
            and kv_stage_cfg is None
            and os.environ.get("HELION_CUTE_FLASH_KV_STAGE") is None
        ):
            kv_stage = (
                _flash_dense_hd64_seed_params(num_kv)[0] if long_dense_hd64_fa4 else 2
            )
        # fa4 is 2-Q-tile (sQ holds q_stage=2 tiles) and uses one aliased K/V
        # shared-memory ring, matching FA4. That keeps kv_stage=3 within the
        # sm_100a shared-memory budget even for head_dim 128.
        if kv_stage < 2:
            # The aliased stream is ordered K0, V0, K1, V1, ... . A one-slot ring
            # cannot hold V_i while prefetching K_{i+1}, so clamp user/env
            # overrides to the minimum safe ring depth.
            kv_stage = 2
    persistent_ctas_per_sm_default = (
        _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv)
        if long_dense_hd64_fa4 and persistent
        else 1
    )
    persistent_ctas_per_sm = int(
        os.environ.get(
            "HELION_CUTE_FLASH_PERSISTENT_CTAS_PER_SM",
            str(persistent_ctas_per_sm_default),
        )
    )
    persistent_ctas_per_sm_cfg = _cfg(FLASH_PERSISTENT_CTAS_PER_SM_KEY)
    if persistent_ctas_per_sm_cfg is not None:
        persistent_ctas_per_sm = int(persistent_ctas_per_sm_cfg)  # type: ignore[arg-type]
    if not persistent:
        persistent_ctas_per_sm = 1
    else:
        persistent_ctas_per_sm = min(max(persistent_ctas_per_sm, 1), 4)
    recompute_tile_coords_default = (
        _flash_dense_hd64_seed_recompute_tile_coords(num_kv)
        if long_dense_hd64_fa4 and persistent
        else False
    )
    recompute_tile_coords = _flash_bool_env(
        "HELION_CUTE_FLASH_RECOMPUTE_TILE_COORDS",
        recompute_tile_coords_default,
    )
    recompute_tile_coords_cfg = _cfg(FLASH_RECOMPUTE_TILE_COORDS_KEY)
    if recompute_tile_coords_cfg is not None:
        recompute_tile_coords = bool(recompute_tile_coords_cfg)
    num_softmax_warpgroups = int(
        os.environ.get("HELION_CUTE_FLASH_NUM_SOFTMAX_WG", "1")
    )
    num_correction_warps = int(
        os.environ.get("HELION_CUTE_FLASH_NUM_CORRECTION_WARPS", "0")
    )
    num_mma_warps = int(os.environ.get("HELION_CUTE_FLASH_NUM_MMA_WARPS", "0"))
    num_load_warps = int(os.environ.get("HELION_CUTE_FLASH_NUM_LOAD_WARPS", "0"))
    num_epilogue_warps = int(os.environ.get("HELION_CUTE_FLASH_NUM_EPI_WARPS", "0"))
    q_tile_count = int(os.environ.get("HELION_CUTE_FLASH_Q_TILE_COUNT", "1"))
    q_tile_count_cfg = _cfg(FLASH_Q_TILE_COUNT_KEY)
    if q_tile_count_cfg is not None:
        q_tile_count = int(q_tile_count_cfg)  # type: ignore[arg-type]
    acc_stage = int(os.environ.get("HELION_CUTE_FLASH_ACC_STAGE", "1"))
    epi_stage = int(os.environ.get("HELION_CUTE_FLASH_EPI_STAGE", "1"))
    # Exp2 pipe-split schedule. The autotuner sees this as one paired schedule
    # knob so it never searches meaningless combinations like exp2_impl="xu"
    # with independent e2e cadence values, or regressive res/freq pairings.
    e2e_schedule_default = (
        _flash_dense_hd64_seed_e2e_schedule(num_kv)
        if long_dense_hd64_fa4
        else "8/2"
        if causal_hd64_seeded_fa4
        else _flash_e2e_schedule_default(topology, head_dim)
    )
    e2e_schedule = os.environ.get(
        "HELION_CUTE_FLASH_E2E_SCHEDULE", e2e_schedule_default
    )
    exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
        e2e_schedule, e2e_schedule_default
    )
    # Legacy env overrides remain accepted for scripts. Config overrides are
    # applied afterward so autotuned/fixed config values still win over env.
    exp2_impl = os.environ.get("HELION_CUTE_FLASH_EXP2_IMPL", exp2_impl)
    e2e_freq = int(os.environ.get("HELION_CUTE_FLASH_E2E_FREQ", str(e2e_freq)))
    e2e_res = int(os.environ.get("HELION_CUTE_FLASH_E2E_RES", str(e2e_res)))
    exp2_impl, e2e_freq, e2e_res, e2e_schedule = _flash_normalize_e2e_params(
        exp2_impl, e2e_freq, e2e_res, e2e_schedule_default
    )
    e2e_schedule_cfg = _cfg(FLASH_E2E_SCHEDULE_KEY)
    if e2e_schedule_cfg is not None:
        exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
            str(e2e_schedule_cfg), e2e_schedule_default
        )
    exp2_impl_cfg = _cfg(FLASH_EXP2_IMPL_KEY)
    if exp2_impl_cfg is not None:
        exp2_impl = str(exp2_impl_cfg)
    e2e_freq_cfg = _cfg(FLASH_E2E_FREQ_KEY)
    if e2e_freq_cfg is not None:
        e2e_freq = int(e2e_freq_cfg)  # type: ignore[arg-type]
    e2e_res_cfg = _cfg(FLASH_E2E_RES_KEY)
    if e2e_res_cfg is not None:
        e2e_res = int(e2e_res_cfg)  # type: ignore[arg-type]
    exp2_impl, e2e_freq, e2e_res, e2e_schedule = _flash_normalize_e2e_params(
        exp2_impl, e2e_freq, e2e_res, e2e_schedule_default
    )
    masked_e2e_schedule_default = (
        _flash_causal_hd64_seed_masked_schedule(num_kv)
        if causal_hd64_seeded_fa4
        else "inherit"
    )
    masked_e2e_schedule = os.environ.get(
        "HELION_CUTE_FLASH_MASKED_E2E_SCHEDULE", masked_e2e_schedule_default
    )
    masked_e2e_schedule_cfg = _cfg(FLASH_MASKED_E2E_SCHEDULE_KEY)
    if masked_e2e_schedule_cfg is not None:
        masked_e2e_schedule = str(masked_e2e_schedule_cfg)
    masked_e2e_schedule, masked_e2e_freq, masked_e2e_res = (
        _flash_masked_e2e_schedule_params(
            masked_e2e_schedule,
            e2e_schedule_default,
            e2e_freq,
            e2e_res,
        )
    )
    if not is_causal or topology != "fa4":
        masked_e2e_schedule = "inherit"
        masked_e2e_freq = e2e_freq
        masked_e2e_res = e2e_res
    e2e_offset_period = _flash_e2e_offset_period(
        e2e_freq,
        e2e_res,
        masked_e2e_freq,
        masked_e2e_res,
    )
    if long_dense_hd64_fa4 and e2e_offset_period > 0:
        e2e_offset_default = (
            _flash_dense_hd64_seed_params(num_kv)[1] % e2e_offset_period
        )
    elif causal_hd64_seeded_fa4 and e2e_offset_period > 0:
        e2e_offset_default = causal_e2e_offset_default % e2e_offset_period
    else:
        split_default_freq = e2e_freq if e2e_res > 0 else masked_e2e_freq
        e2e_offset_default = (
            split_default_freq // 8
            if topology == "fa4" and head_dim == 64 and e2e_offset_period > 0
            else 0
        )
    e2e_offset = int(
        os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET", str(e2e_offset_default))
    )
    e2e_offset_cfg = _cfg(FLASH_E2E_OFFSET_KEY)
    if e2e_offset_cfg is not None:
        e2e_offset = int(e2e_offset_cfg)  # type: ignore[arg-type]
    e2e_offset = _flash_normalize_e2e_offset(
        e2e_offset, e2e_offset_default, e2e_offset_period
    )
    e2e_offset0_default = (
        _flash_dense_hd64_seed_params(num_kv)[2]
        if long_dense_hd64_fa4 and e2e_offset_period > 0
        else _flash_causal_hd64_seed_offset0(num_kv)
        if causal_hd64_seeded_fa4 and e2e_offset_period > 0
        else 0
    )
    e2e_offset0 = int(
        os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET0", str(e2e_offset0_default))
    )
    e2e_offset0_cfg = _cfg(FLASH_E2E_OFFSET0_KEY)
    if e2e_offset0_cfg is not None:
        e2e_offset0 = int(e2e_offset0_cfg)  # type: ignore[arg-type]
    e2e_offset0 = _flash_normalize_e2e_offset(
        e2e_offset0, e2e_offset0_default, e2e_offset_period
    )
    tmem_plan = os.environ.get("HELION_CUTE_FLASH_TMEM_PLAN", "separate")
    tmem_s_to_p_offset = int(
        os.environ.get("HELION_CUTE_FLASH_TMEM_S_TO_P_OFFSET", "0")
    )
    mma_interleave = _flash_bool_env("HELION_CUTE_FLASH_MMA_INTERLEAVE", False)
    mma_interleave_cfg = _cfg(FLASH_MMA_INTERLEAVE_KEY)
    if mma_interleave_cfg is not None:
        mma_interleave = bool(mma_interleave_cfg)
    # fa4 Stage 2b: PTX-path MMA warp (default ON for fa4). HELION_CUTE_FLASH_MMA_PTX=0
    # reverts to the cute.gemm path (the Stage-1/2a body, for A/B comparison).
    mma_ptx = _flash_bool_env("HELION_CUTE_FLASH_MMA_PTX", True)
    # fa4 Step 2: chunked-t2r ("disc") softmax body (default ON for fa4).
    # HELION_CUTE_FLASH_SOFTMAX_DISC=0 reverts to the whole-row body for A/B.
    softmax_disc_default = (
        _flash_dense_hd64_seed_softmax_disc(num_kv) if long_dense_hd64_fa4 else True
    )
    softmax_disc = _flash_bool_env(
        "HELION_CUTE_FLASH_SOFTMAX_DISC", softmax_disc_default
    )
    softmax_disc_cfg = _cfg(FLASH_SOFTMAX_DISC_KEY)
    if softmax_disc_cfg is not None:
        softmax_disc = bool(softmax_disc_cfg)
    if topology != "fa4":
        softmax_disc = True
    # fa4 Step 3 (L1): disc PASS2 software-pipeline depth. The best measured
    # depth is shape-sensitive: dense hd64 benefits from prefetching all four
    # PASS2 chunks in the prologue, while causal hd64 and hd128 only tolerate one
    # extra prefetched fragment before pressure dominates. Env/config overrides
    # still win.
    if topology == "fa4" and softmax_disc:
        disc_pipe_default = (
            3
            if long_dense_hd64_fa4
            else causal_disc_pipe_default
            if causal_hd64_seeded_fa4
            else 2
            if is_causal or head_dim == 128
            else 4
        )
    else:
        disc_pipe_default = 1
    disc_pipe_depth = int(
        os.environ.get("HELION_CUTE_FLASH_DISC_PIPE", str(disc_pipe_default))
    )
    disc_pipe_depth_cfg = _cfg(FLASH_DISC_PIPE_KEY)
    if disc_pipe_depth_cfg is not None:
        disc_pipe_depth = int(disc_pipe_depth_cfg)  # type: ignore[arg-type]
    split_p_arrive_default = (
        _flash_dense_hd64_seed_split_p_arrive(num_kv)
        if long_dense_hd64_fa4
        else not (topology == "fa4" and not is_causal and head_dim <= 64)
    )
    split_p_arrive = _flash_bool_env(
        "HELION_CUTE_FLASH_SPLIT_P_ARRIVE", split_p_arrive_default
    )
    split_p_arrive_cfg = _cfg(FLASH_SPLIT_P_ARRIVE_KEY)
    if split_p_arrive_cfg is not None:
        split_p_arrive = bool(split_p_arrive_cfg)
    if topology != "fa4" or not mma_ptx:
        split_p_arrive = True
    p_store_repetition_default = (
        _flash_dense_hd64_seed_p_store_rep(num_kv) if long_dense_hd64_fa4 else 16
    )
    p_store_repetition = int(
        os.environ.get("HELION_CUTE_FLASH_P_STORE_REP", str(p_store_repetition_default))
    )
    p_store_repetition_cfg = _cfg(FLASH_P_STORE_REP_KEY)
    if p_store_repetition_cfg is not None:
        p_store_repetition = int(p_store_repetition_cfg)  # type: ignore[arg-type]
    if p_store_repetition not in (16, 32) or topology != "fa4":
        p_store_repetition = 16
    s_load_repetition_default = (
        _flash_dense_hd64_seed_s_load_rep(num_kv) if long_dense_hd64_fa4 else 32
    )
    s_load_repetition = int(
        os.environ.get("HELION_CUTE_FLASH_S_LOAD_REP", str(s_load_repetition_default))
    )
    s_load_repetition_cfg = _cfg(FLASH_S_LOAD_REP_KEY)
    if s_load_repetition_cfg is not None:
        s_load_repetition = int(s_load_repetition_cfg)  # type: ignore[arg-type]
    if (
        s_load_repetition not in (16, 32)
        or topology != "fa4"
        or p_store_repetition != 16
    ):
        s_load_repetition = 32
    precompute_qk_desc_default = (
        _flash_dense_hd64_seed_precompute_qk_desc(num_kv)
        if long_dense_hd64_fa4
        else False
    )
    precompute_qk_desc = _flash_bool_env(
        "HELION_CUTE_FLASH_PRECOMPUTE_QK_DESC", precompute_qk_desc_default
    )
    precompute_qk_desc_cfg = _cfg(FLASH_PRECOMPUTE_QK_DESC_KEY)
    if precompute_qk_desc_cfg is not None:
        precompute_qk_desc = bool(precompute_qk_desc_cfg)
    if topology != "fa4" or not mma_ptx:
        precompute_qk_desc = False
    first_load_order_default = (
        _flash_dense_hd64_seed_first_load_order(num_kv) if long_dense_hd64_fa4 else 0
    )
    first_load_order = int(
        os.environ.get(
            "HELION_CUTE_FLASH_FIRST_LOAD_ORDER", str(first_load_order_default)
        )
    )
    first_load_order_cfg = _cfg(FLASH_FIRST_LOAD_ORDER_KEY)
    if first_load_order_cfg is not None:
        first_load_order = int(first_load_order_cfg)  # type: ignore[arg-type]
    if topology != "fa4" or first_load_order not in (0, 1, 2, 3, 4):
        first_load_order = 0
    kv_order_default = (
        _flash_dense_hd64_seed_kv_order(num_kv) if long_dense_hd64_fa4 else "ascending"
    )
    kv_order = os.environ.get("HELION_CUTE_FLASH_KV_ORDER", kv_order_default)
    kv_order_cfg = _cfg(FLASH_KV_ORDER_KEY)
    if kv_order_cfg is not None:
        kv_order = str(kv_order_cfg)
    if topology != "fa4" or kv_order not in ("ascending", "descending"):
        kv_order = "ascending"
    # fa4 Step 3 (Lever A): epilogue TMA-store. With a dedicated epilogue warp,
    # this is a measured win for hd128 and for the very-long dense hd64 family;
    # shorter hd64 families keep their characterized seed choices. fa4-only.
    epi_tma_default = (
        _flash_dense_hd64_seed_params(num_kv)[4]
        if long_dense_hd64_fa4
        else _flash_causal_hd64_seed_epi_tma(num_kv)
        if causal_hd64_seeded_fa4
        else topology == "fa4" and head_dim == 128
    )
    epi_tma = _flash_bool_env("HELION_CUTE_FLASH_EPI_TMA", epi_tma_default)
    epi_tma_cfg = _cfg(FLASH_EPI_TMA_KEY)
    if epi_tma_cfg is not None:
        epi_tma = bool(epi_tma_cfg)
    epi_tma = epi_tma and topology == "fa4"
    epi_stg_default = (
        _flash_dense_hd64_seed_epi_stg(num_kv) if long_dense_hd64_fa4 else False
    )
    epi_stg = _flash_bool_env("HELION_CUTE_FLASH_EPI_STG", epi_stg_default)
    epi_stg_cfg = _cfg(FLASH_EPI_STG_KEY)
    if epi_stg_cfg is not None:
        epi_stg = bool(epi_stg_cfg)
    epi_stg = epi_stg and topology == "fa4" and not epi_tma
    epi_stg_store = os.environ.get("HELION_CUTE_FLASH_EPI_STG_STORE", "slice")
    epi_stg_store_cfg = _cfg(FLASH_EPI_STG_STORE_KEY)
    if epi_stg_store_cfg is not None:
        epi_stg_store = str(epi_stg_store_cfg)
    if epi_stg_store not in ("slice", "whole") or not epi_stg:
        epi_stg_store = "slice"
    epi_stg_gmem = os.environ.get("HELION_CUTE_FLASH_EPI_STG_GMEM", "stage")
    epi_stg_gmem_cfg = _cfg(FLASH_EPI_STG_GMEM_KEY)
    if epi_stg_gmem_cfg is not None:
        epi_stg_gmem = str(epi_stg_gmem_cfg)
    if epi_stg_gmem not in ("stage", "pair") or not epi_stg:
        epi_stg_gmem = "stage"
    # fa4 softmax rescale_threshold (alpha-pin), dtype-gated: FA4 ships 8.0 for
    # fp16/bf16 and 0.0 (disabled = always-rescale = prior behavior) for fp8. The
    # default is keyed off the input dtype (real check, not try/except) so an fp8
    # path is correct-by-default; env/config overrides still win.
    dtype_rescale_threshold_default = (
        8.0 if dtype in (torch.float16, torch.bfloat16) else 0.0
    )
    rescale_threshold_default = (
        _flash_dense_hd64_seed_rescale_threshold(num_kv)
        if long_dense_hd64_fa4 and dtype in (torch.float16, torch.bfloat16)
        else dtype_rescale_threshold_default
    )
    rescale_threshold = float(
        os.environ.get("HELION_CUTE_FLASH_RESCALE_THRESHOLD", rescale_threshold_default)
    )
    rescale_threshold_cfg = _cfg(FLASH_RESCALE_THRESHOLD_KEY)
    if rescale_threshold_cfg is not None:
        rescale_threshold = float(rescale_threshold_cfg)  # type: ignore[arg-type]
    if not math.isfinite(rescale_threshold):
        rescale_threshold = rescale_threshold_default
    skip_rescale_stats_default = (
        _flash_dense_hd64_seed_skip_rescale_stats(num_kv)
        if long_dense_hd64_fa4
        else False
    )
    skip_rescale_stats = _flash_bool_env(
        "HELION_CUTE_FLASH_SKIP_RESCALE_STATS", skip_rescale_stats_default
    )
    skip_rescale_stats_cfg = _cfg(FLASH_SKIP_RESCALE_STATS_KEY)
    if skip_rescale_stats_cfg is not None:
        skip_rescale_stats = bool(skip_rescale_stats_cfg)
    if skip_rescale_stats:
        # Without per-KV alpha handoffs, a later tile whose max jumps beyond the
        # pinned threshold would leave accumulated O on the old exponent base.
        # Keep the knob in the schema for cached/manual config compatibility but
        # compile the correct rescale path until a safe dynamic fallback exists.
        skip_rescale_stats = False
    rescale_chunk_cols = (
        _flash_dense_hd64_seed_params(num_kv)[6]
        if long_dense_hd64_fa4
        else _flash_causal_hd64_seed_rescale_chunk_cols(num_kv)
        if causal_hd64_seeded_fa4
        else 32
        if head_dim == 64
        else 16
    )
    softmax_regs = (
        _flash_dense_hd64_seed_softmax_regs(num_kv) if long_dense_hd64_fa4 else 200
    )
    corr_regs = _flash_dense_hd64_seed_corr_regs(num_kv) if long_dense_hd64_fa4 else 64
    if topology == "fa4":
        rescale_chunk_default = rescale_chunk_cols
        rescale_chunk_cols = int(
            os.environ.get(
                "HELION_CUTE_FLASH_RESCALE_CHUNK_COLS", rescale_chunk_default
            )
        )
        rescale_chunk_cols_cfg = _cfg(FLASH_RESCALE_CHUNK_COLS_KEY)
        if rescale_chunk_cols_cfg is not None:
            rescale_chunk_cols = int(rescale_chunk_cols_cfg)  # type: ignore[arg-type]
        if (
            rescale_chunk_cols not in (8, 16, 32, 64)
            or head_dim % rescale_chunk_cols != 0
        ):
            rescale_chunk_cols = rescale_chunk_default
        softmax_regs = int(
            os.environ.get(
                "HELION_CUTE_FLASH_SOFTMAX_REGS",
                str(
                    causal_softmax_regs_default
                    if causal_hd64_seeded_fa4
                    else softmax_regs
                ),
            )
        )
        softmax_regs_cfg = _cfg(FLASH_SOFTMAX_REGS_KEY)
        if softmax_regs_cfg is not None:
            softmax_regs = int(softmax_regs_cfg)  # type: ignore[arg-type]
        if softmax_regs % 8 != 0 or softmax_regs <= 0:
            softmax_regs = 200
        corr_regs_default = corr_regs
        corr_regs = int(
            os.environ.get("HELION_CUTE_FLASH_CORR_REGS", str(corr_regs_default))
        )
        corr_regs_cfg = _cfg(FLASH_CORR_REGS_KEY)
        if corr_regs_cfg is not None:
            corr_regs = int(corr_regs_cfg)  # type: ignore[arg-type]
        if corr_regs % 8 != 0 or corr_regs <= 0:
            corr_regs = corr_regs_default
        other_regs_default = (
            _flash_dense_hd64_seed_other_regs(num_kv) if long_dense_hd64_fa4 else 48
        )
        other_regs = int(
            os.environ.get("HELION_CUTE_FLASH_OTHER_REGS", str(other_regs_default))
        )
        other_regs_cfg = _cfg(FLASH_OTHER_REGS_KEY)
        if other_regs_cfg is not None:
            other_regs = int(other_regs_cfg)  # type: ignore[arg-type]
        if other_regs % 8 != 0 or other_regs < 24:
            other_regs = other_regs_default
        corr_tile_size_default = (
            _flash_dense_hd64_seed_corr_tile_size(num_kv)
            if long_dense_hd64_fa4
            else 8
            if not is_causal and head_dim <= 64
            else 16
        )
        corr_tile_size = int(
            os.environ.get(
                "HELION_CUTE_FLASH_CORR_TILE_SIZE",
                str(corr_tile_size_default),
            )
        )
        corr_tile_size_cfg = _cfg(FLASH_CORR_TILE_SIZE_KEY)
        if corr_tile_size_cfg is not None:
            corr_tile_size = int(corr_tile_size_cfg)  # type: ignore[arg-type]
        if corr_tile_size not in (8, 16, 32) or head_dim % corr_tile_size != 0:
            corr_tile_size = corr_tile_size_default
    else:
        other_regs = 48
        corr_tile_size = 16
    packed_reduce = _flash_bool_env(
        "HELION_CUTE_FLASH_PACKED_REDUCE",
        prefer_packed_reduce or long_dense_hd64_fa4 or causal_hd64_seeded_fa4,
    )
    packed_reduce_cfg = _cfg(FLASH_PACKED_REDUCE_KEY)
    if packed_reduce_cfg is not None:
        packed_reduce = bool(packed_reduce_cfg)
    small_biased = True
    small_biased_cfg = _cfg(FLASH_SMALL_BIASED_KEY)
    if small_biased_cfg is not None:
        small_biased = bool(small_biased_cfg)
    causal_lpt_swizzle = int(
        os.environ.get(
            "HELION_CUTE_FLASH_CAUSAL_LPT_SWIZZLE",
            str(causal_lpt_swizzle_default if causal_hd64_seeded_fa4 else 0),
        )
    )
    causal_lpt_swizzle_cfg = _cfg(FLASH_CAUSAL_LPT_SWIZZLE_KEY)
    if causal_lpt_swizzle_cfg is not None:
        causal_lpt_swizzle = int(causal_lpt_swizzle_cfg)  # type: ignore[arg-type]
    if not is_causal:
        causal_lpt_swizzle = 0
    causal_kv_order_default = (
        _flash_causal_hd64_seed_kv_order(num_kv)
        if causal_hd64_seeded_fa4
        else "ascending"
    )
    causal_kv_order = os.environ.get(
        "HELION_CUTE_FLASH_CAUSAL_KV_ORDER", causal_kv_order_default
    )
    causal_kv_order_cfg = _cfg(FLASH_CAUSAL_KV_ORDER_KEY)
    if causal_kv_order_cfg is not None:
        causal_kv_order = str(causal_kv_order_cfg)
    if (
        not is_causal
        or topology != "fa4"
        or causal_kv_order not in ("ascending", "descending")
    ):
        causal_kv_order = "ascending"
    role_map_default = (
        _flash_causal_hd64_seed_role_map(num_kv)
        if causal_hd64_seeded_fa4
        else _flash_dense_hd64_seed_role_map(num_kv)
        if long_dense_hd64_fa4
        else "helion"
    )
    role_map = os.environ.get("HELION_CUTE_FLASH_ROLE_MAP", role_map_default)
    role_map_cfg = _cfg(FLASH_ROLE_MAP_KEY)
    if role_map_cfg is not None:
        role_map = str(role_map_cfg)
    if role_map not in ("helion", "fa4"):
        role_map = "helion"
    if topology != "fa4":
        role_map = "helion"
    use_2cta_instrs = _flash_bool_env("HELION_CUTE_FLASH_USE_2CTA", False)
    use_2cta_cfg = _cfg(FLASH_USE_2CTA_KEY)
    if use_2cta_cfg is not None:
        use_2cta_instrs = bool(use_2cta_cfg)
    if topology != "fa4" or is_causal or num_kv % 4 != 0 or head_dim != 128:
        use_2cta_instrs = False
    if use_2cta_instrs:
        precompute_qk_desc = False
    use_cga2_local_default = (
        _flash_dense_hd64_seed_cga2_local(num_kv) if long_dense_hd64_fa4 else False
    )
    use_cga2_local_cta = _flash_bool_env(
        "HELION_CUTE_FLASH_CGA2_LOCAL", use_cga2_local_default
    )
    use_cga2_local_cfg = _cfg(FLASH_CGA2_LOCAL_KEY)
    use_cga2_local_overridden = (
        "HELION_CUTE_FLASH_CGA2_LOCAL" in os.environ or use_cga2_local_cfg is not None
    )
    if use_cga2_local_cfg is not None:
        use_cga2_local_cta = bool(use_cga2_local_cfg)
    if (
        topology != "fa4"
        or is_causal
        or num_kv % 4 != 0
        or head_dim != 64
        or use_2cta_instrs
    ):
        use_cga2_local_cta = False
    use_clc_scheduler_default = (
        _flash_dense_hd64_seed_clc(num_kv) if long_dense_hd64_fa4 else False
    )
    use_clc_scheduler = _flash_bool_env(
        "HELION_CUTE_FLASH_CLC", use_clc_scheduler_default
    )
    use_clc_scheduler_cfg = _cfg(FLASH_CLC_KEY)
    if use_clc_scheduler_cfg is not None:
        use_clc_scheduler = bool(use_clc_scheduler_cfg)
    if use_clc_scheduler and use_cga2_local_cta and not use_cga2_local_overridden:
        use_cga2_local_cta = False
    if (
        topology != "fa4"
        or not persistent
        or is_causal
        or head_dim != 64
        or num_kv % 2 != 0
        or use_2cta_instrs
        or use_cga2_local_cta
    ):
        use_clc_scheduler = False
    clc_heads_per_batch_default = (
        _flash_dense_hd64_seed_clc_heads_per_batch(num_kv) if long_dense_hd64_fa4 else 0
    )
    clc_heads_per_batch = int(
        os.environ.get("HELION_CUTE_FLASH_CLC_HEADS", str(clc_heads_per_batch_default))
    )
    clc_heads_per_batch_cfg = _cfg(FLASH_CLC_HEADS_PER_BATCH_KEY)
    if clc_heads_per_batch_cfg is not None:
        clc_heads_per_batch = int(clc_heads_per_batch_cfg)  # type: ignore[arg-type]
    if clc_heads_per_batch < 0 or not use_clc_scheduler:
        clc_heads_per_batch = 0
    clc_use_pdl = _flash_bool_env("HELION_CUTE_FLASH_CLC_PDL", False)
    clc_use_pdl_cfg = _cfg(FLASH_CLC_PDL_KEY)
    if clc_use_pdl_cfg is not None:
        clc_use_pdl = bool(clc_use_pdl_cfg)
    if not use_clc_scheduler:
        clc_use_pdl = False
    clc_stages = int(os.environ.get("HELION_CUTE_FLASH_CLC_STAGES", "1"))
    clc_stages_cfg = _cfg(FLASH_CLC_STAGES_KEY)
    if clc_stages_cfg is not None:
        clc_stages = int(clc_stages_cfg)  # type: ignore[arg-type]
    if not use_clc_scheduler or clc_stages not in (1, 2, 3):
        clc_stages = 1
    if (
        topology != "fa4"
        or not persistent
        or use_clc_scheduler
        or use_2cta_instrs
        or use_cga2_local_cta
    ):
        recompute_tile_coords = False
    local_tma_partition_default = (
        _flash_dense_hd64_seed_local_tma_partition(num_kv)
        if long_dense_hd64_fa4
        else False
    )
    local_tma_partition = _flash_bool_env(
        "HELION_CUTE_FLASH_LOCAL_TMA", local_tma_partition_default
    )
    local_tma_partition_cfg = _cfg(FLASH_LOCAL_TMA_PARTITION_KEY)
    if local_tma_partition_cfg is not None:
        local_tma_partition = bool(local_tma_partition_cfg)
    local_tma_partition = local_tma_partition and topology == "fa4"
    if use_cga2_local_cta:
        local_tma_partition = False
    tensor_4d_tma_default = (
        _flash_dense_hd64_seed_tensor_4d_tma(num_kv) if long_dense_hd64_fa4 else False
    )
    tensor_4d_tma = _flash_bool_env(
        "HELION_CUTE_FLASH_TENSOR_4D", tensor_4d_tma_default
    )
    tensor_4d_tma_cfg = _cfg(FLASH_TENSOR_4D_TMA_KEY)
    if tensor_4d_tma_cfg is not None:
        tensor_4d_tma = bool(tensor_4d_tma_cfg)
    tensor_4d_tma = tensor_4d_tma and topology == "fa4"
    causal_loop_split_default = (
        causal_hd64_seeded_fa4 and causal_kv_order == "descending"
    )
    causal_loop_split = _flash_bool_env(
        "HELION_CUTE_FLASH_CAUSAL_LOOP_SPLIT", causal_loop_split_default
    )
    causal_loop_split_cfg = _cfg(FLASH_CAUSAL_LOOP_SPLIT_KEY)
    if causal_loop_split_cfg is not None:
        causal_loop_split = bool(causal_loop_split_cfg)
    if not is_causal or topology != "fa4" or causal_kv_order != "descending":
        causal_loop_split = False
    return FlashAttentionConfig(
        s_stage=s_stage,
        kv_stage=kv_stage,
        persistent=persistent,
        persistent_ctas_per_sm=persistent_ctas_per_sm,
        recompute_tile_coords=recompute_tile_coords,
        num_regs_producer=_FLASH_NUM_REGS_PRODUCER,
        num_regs_consumer=_FLASH_NUM_REGS_CONSUMER,
        topology=topology,
        num_softmax_warpgroups=num_softmax_warpgroups,
        num_correction_warps=num_correction_warps,
        num_mma_warps=num_mma_warps,
        num_load_warps=num_load_warps,
        num_epilogue_warps=num_epilogue_warps,
        q_tile_count=q_tile_count,
        acc_stage=acc_stage,
        epi_stage=epi_stage,
        exp2_impl=exp2_impl,
        e2e_freq=e2e_freq,
        e2e_res=e2e_res,
        e2e_schedule=e2e_schedule,
        masked_e2e_schedule=masked_e2e_schedule,
        masked_e2e_freq=masked_e2e_freq,
        masked_e2e_res=masked_e2e_res,
        e2e_offset=e2e_offset,
        e2e_offset0=e2e_offset0,
        tmem_plan=tmem_plan,
        tmem_s_to_p_offset=tmem_s_to_p_offset,
        mma_interleave=mma_interleave,
        mma_ptx=mma_ptx,
        softmax_disc=softmax_disc,
        disc_pipe_depth=disc_pipe_depth,
        split_p_arrive=split_p_arrive,
        p_store_repetition=p_store_repetition,
        s_load_repetition=s_load_repetition,
        precompute_qk_desc=precompute_qk_desc,
        first_load_order=first_load_order,
        kv_order=kv_order,
        epi_tma=epi_tma,
        epi_stg=epi_stg,
        epi_stg_store=epi_stg_store,
        epi_stg_gmem=epi_stg_gmem,
        rescale_threshold=rescale_threshold,
        skip_rescale_stats=skip_rescale_stats,
        rescale_chunk_cols=rescale_chunk_cols,
        softmax_regs=softmax_regs,
        corr_regs=corr_regs,
        other_regs=other_regs,
        corr_tile_size=corr_tile_size,
        packed_reduce=packed_reduce,
        small_biased=small_biased,
        causal_lpt_swizzle=causal_lpt_swizzle,
        causal_kv_order=causal_kv_order,
        role_map=role_map,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        clc_use_pdl=clc_use_pdl,
        clc_stages=clc_stages,
        local_tma_partition=local_tma_partition,
        tensor_4d_tma=tensor_4d_tma,
        causal_loop_split=causal_loop_split,
    )


# ---------------------------------------------------------------------------
# Autotune surface for the flash-attention config (Tasks #25 + #28).
#
# Mirrors the ``Tcgen05WarpSpec`` pattern in ``strategies.py`` /
# ``tcgen05_config.py``: each autotunable field gets its own config key so the
# autotuner can permute them independently, a ``FLASH_CONFIG_KEYS`` tuple
# aggregates them, and ``flash_config_from_config`` reconstructs the dataclass
# from a config Mapping (falling back to env/shape resolution for every key the
# config does not carry).
#
# Gating: these keys are ONLY inserted into the search surface when
# ``ConfigSpec.cute_flash_search_enabled`` is True (set when the flash detector
# fires; default False). With the flag off the keys never appear, so the config
# Mapping never carries them and ``resolve_flash_config`` behaves byte-identical
# to the pre-autotune env-only path.
# ---------------------------------------------------------------------------
FLASH_S_STAGE_KEY = "cute_flash_s_stage"
FLASH_KV_STAGE_KEY = "cute_flash_kv_stage"
FLASH_PERSISTENT_KEY = "cute_flash_persistent"
FLASH_PERSISTENT_CTAS_PER_SM_KEY = "cute_flash_persistent_ctas_per_sm"
FLASH_RECOMPUTE_TILE_COORDS_KEY = "cute_flash_recompute_tile_coords"
FLASH_E2E_SCHEDULE_KEY = "cute_flash_e2e_schedule"
FLASH_MASKED_E2E_SCHEDULE_KEY = "cute_flash_masked_e2e_schedule"
FLASH_E2E_OFFSET_KEY = "cute_flash_e2e_offset"
FLASH_E2E_OFFSET0_KEY = "cute_flash_e2e_offset0"
# Legacy fixed-config override keys. The autotuner uses the paired schedule key above.
FLASH_EXP2_IMPL_KEY = "cute_flash_exp2_impl"
FLASH_E2E_FREQ_KEY = "cute_flash_e2e_freq"
FLASH_E2E_RES_KEY = "cute_flash_e2e_res"
FLASH_MMA_INTERLEAVE_KEY = "cute_flash_mma_interleave"
FLASH_Q_TILE_COUNT_KEY = "cute_flash_q_tile_count"
# fa4 win (commit 38ff4d1a): the topology selector + the two fa4 perf levers.
FLASH_TOPOLOGY_KEY = "cute_flash_topology"
FLASH_SOFTMAX_DISC_KEY = "cute_flash_softmax_disc"
FLASH_DISC_PIPE_KEY = "cute_flash_disc_pipe"
FLASH_SPLIT_P_ARRIVE_KEY = "cute_flash_split_p_arrive"
FLASH_P_STORE_REP_KEY = "cute_flash_p_store_rep"
FLASH_S_LOAD_REP_KEY = "cute_flash_s_load_rep"
FLASH_PRECOMPUTE_QK_DESC_KEY = "cute_flash_precompute_qk_desc"
FLASH_FIRST_LOAD_ORDER_KEY = "cute_flash_first_load_order"
FLASH_KV_ORDER_KEY = "cute_flash_kv_order"
FLASH_EPI_TMA_KEY = "cute_flash_epi_tma"
FLASH_EPI_STG_KEY = "cute_flash_epi_stg"
FLASH_EPI_STG_STORE_KEY = "cute_flash_epi_stg_store"
FLASH_EPI_STG_GMEM_KEY = "cute_flash_epi_stg_gmem"
FLASH_RESCALE_THRESHOLD_KEY = "cute_flash_rescale_threshold"
FLASH_SKIP_RESCALE_STATS_KEY = "cute_flash_skip_rescale_stats"
FLASH_RESCALE_CHUNK_COLS_KEY = "cute_flash_rescale_chunk_cols"
FLASH_SOFTMAX_REGS_KEY = "cute_flash_softmax_regs"
FLASH_CORR_REGS_KEY = "cute_flash_corr_regs"
FLASH_OTHER_REGS_KEY = "cute_flash_other_regs"
FLASH_CORR_TILE_SIZE_KEY = "cute_flash_corr_tile_size"
FLASH_PACKED_REDUCE_KEY = "cute_flash_packed_reduce"
FLASH_SMALL_BIASED_KEY = "cute_flash_small_biased"
FLASH_CAUSAL_LPT_SWIZZLE_KEY = "cute_flash_causal_lpt_swizzle"
FLASH_CAUSAL_KV_ORDER_KEY = "cute_flash_causal_kv_order"
FLASH_ROLE_MAP_KEY = "cute_flash_role_map"
FLASH_USE_2CTA_KEY = "cute_flash_use_2cta"
FLASH_CGA2_LOCAL_KEY = "cute_flash_cga2_local"
FLASH_CLC_KEY = "cute_flash_clc"
FLASH_CLC_HEADS_PER_BATCH_KEY = "cute_flash_clc_heads_per_batch"
FLASH_CLC_PDL_KEY = "cute_flash_clc_pdl"
FLASH_CLC_STAGES_KEY = "cute_flash_clc_stages"
FLASH_LOCAL_TMA_PARTITION_KEY = "cute_flash_local_tma_partition"
FLASH_TENSOR_4D_TMA_KEY = "cute_flash_tensor_4d_tma"
FLASH_CAUSAL_LOOP_SPLIT_KEY = "cute_flash_causal_loop_split"

FLASH_AUTOTUNE_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_S_STAGE_KEY,
    FLASH_KV_STAGE_KEY,
    FLASH_PERSISTENT_KEY,
    FLASH_PERSISTENT_CTAS_PER_SM_KEY,
    FLASH_RECOMPUTE_TILE_COORDS_KEY,
    FLASH_E2E_SCHEDULE_KEY,
    FLASH_MASKED_E2E_SCHEDULE_KEY,
    FLASH_E2E_OFFSET_KEY,
    FLASH_E2E_OFFSET0_KEY,
    FLASH_TOPOLOGY_KEY,
    FLASH_SOFTMAX_DISC_KEY,
    FLASH_DISC_PIPE_KEY,
    FLASH_SPLIT_P_ARRIVE_KEY,
    FLASH_P_STORE_REP_KEY,
    FLASH_S_LOAD_REP_KEY,
    FLASH_PRECOMPUTE_QK_DESC_KEY,
    FLASH_FIRST_LOAD_ORDER_KEY,
    FLASH_KV_ORDER_KEY,
    FLASH_EPI_TMA_KEY,
    FLASH_EPI_STG_KEY,
    FLASH_EPI_STG_STORE_KEY,
    FLASH_EPI_STG_GMEM_KEY,
    FLASH_RESCALE_THRESHOLD_KEY,
    FLASH_SKIP_RESCALE_STATS_KEY,
    FLASH_RESCALE_CHUNK_COLS_KEY,
    FLASH_SOFTMAX_REGS_KEY,
    FLASH_CORR_REGS_KEY,
    FLASH_OTHER_REGS_KEY,
    FLASH_CORR_TILE_SIZE_KEY,
    FLASH_PACKED_REDUCE_KEY,
    FLASH_SMALL_BIASED_KEY,
    FLASH_CAUSAL_LPT_SWIZZLE_KEY,
    FLASH_CAUSAL_KV_ORDER_KEY,
    FLASH_ROLE_MAP_KEY,
    FLASH_USE_2CTA_KEY,
    FLASH_CGA2_LOCAL_KEY,
    FLASH_CLC_KEY,
    FLASH_CLC_HEADS_PER_BATCH_KEY,
    FLASH_CLC_PDL_KEY,
    FLASH_CLC_STAGES_KEY,
    FLASH_LOCAL_TMA_PARTITION_KEY,
    FLASH_TENSOR_4D_TMA_KEY,
    FLASH_CAUSAL_LOOP_SPLIT_KEY,
)

FLASH_LEGACY_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_EXP2_IMPL_KEY,
    FLASH_E2E_FREQ_KEY,
    FLASH_E2E_RES_KEY,
    FLASH_MMA_INTERLEAVE_KEY,
    FLASH_Q_TILE_COUNT_KEY,
)

FLASH_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_AUTOTUNE_CONFIG_KEYS + FLASH_LEGACY_CONFIG_KEYS
)


def _flash_choices_with_default(default: _T, choices: Iterable[_T]) -> tuple[_T, ...]:
    return (default, *(choice for choice in choices if choice != default))


_FLASH_SEED_BLOCK_SIZE_TARGETS = (1, 128, 128)
_FLASH_DENSE_HD64_MID_MIN_KV = 16
_FLASH_DENSE_HD64_LONG_MIN_KV = 64
_FLASH_DENSE_HD64_VERY_LONG_MIN_KV = 256


def _flash_dense_hd64_seed_params(
    num_kv: int,
) -> tuple[int, int, int, int, bool, float, int, bool]:
    """Return dense hd64 FA4 seed knobs.

    The families track the configs found by full no-seed search for the
    4K/8K/16K/32K dense sweep without baking in an exact shape-only config.
    """
    if num_kv < _FLASH_DENSE_HD64_LONG_MIN_KV:
        return 3, 2, 2, 4, True, 8.0, 16, False
    if num_kv < 128:
        return 2, 2, 2, 3, True, 8.0, 16, True
    if num_kv < _FLASH_DENSE_HD64_VERY_LONG_MIN_KV:
        return 2, 3, 2, 3, False, 8.0, 16, True
    if num_kv < 512:
        if num_kv == _FLASH_DENSE_HD64_VERY_LONG_MIN_KV:
            return 2, 0, 1, 1, True, 8.0, 8, True
        return 2, 0, 3, 1, True, 32.0, 8, True
    if num_kv < 1024:
        return 2, 0, 0, 3, False, 8.0, 8, True
    if num_kv == 1024:
        return 2, 0, 0, 1, True, 8.0, 8, True
    if num_kv < 2048:
        return 2, 0, 0, 1, False, 8.0, 8, True
    if num_kv == 2048:
        return 3, 0, 0, 1, False, 8.0, 8, True
    return 3, 0, 0, 1, False, 8.0, 8, True


def _flash_dense_hd64_seed_rescale_threshold(num_kv: int) -> float:
    if num_kv == 2048:
        return 32.0
    return _flash_dense_hd64_seed_params(num_kv)[5]


def _flash_dense_hd64_seed_e2e_schedule(num_kv: int) -> str:
    if num_kv == 1024 or num_kv >= 2048:
        return "16/4"
    return "8/2"


def _flash_dense_hd64_seed_p_store_rep(num_kv: int) -> int:
    return 16


def _flash_dense_hd64_seed_s_load_rep(num_kv: int) -> int:
    return 32


def _flash_dense_hd64_seed_split_p_arrive(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_precompute_qk_desc(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_first_load_order(num_kv: int) -> int:
    if num_kv < _FLASH_DENSE_HD64_VERY_LONG_MIN_KV:
        return 0
    if num_kv < 512:
        return 0
    if num_kv == 512:
        return 0
    if num_kv < 1024:
        return 0
    if num_kv == 1024:
        return 0
    if num_kv < 2048:
        return 1
    return 4


def _flash_dense_hd64_seed_kv_order(num_kv: int) -> str:
    if num_kv in (_FLASH_DENSE_HD64_VERY_LONG_MIN_KV, 512, 1024) or num_kv >= 2048:
        return "descending"
    return "ascending"


def _flash_dense_hd64_seed_epi_stg(num_kv: int) -> bool:
    return num_kv >= 512


def _flash_dense_hd64_seed_corr_tile_size(num_kv: int) -> int:
    if num_kv > 2048:
        return 8
    return 8 if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV else 16


def _flash_dense_hd64_seed_skip_rescale_stats(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_clc(num_kv: int) -> bool:
    return num_kv == 1024


def _flash_dense_hd64_seed_clc_heads_per_batch(num_kv: int) -> int:
    return 32 if _flash_dense_hd64_seed_clc(num_kv) else 0


def _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv: int) -> int:
    return 1


def _flash_dense_hd64_seed_recompute_tile_coords(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_softmax_disc(num_kv: int) -> bool:
    return num_kv < _FLASH_DENSE_HD64_VERY_LONG_MIN_KV


def _flash_dense_hd64_seed_role_map(num_kv: int) -> str:
    return "helion"


def _flash_dense_hd64_seed_cga2_local(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_softmax_regs(num_kv: int) -> int:
    if num_kv == 2048:
        return 192
    return 200 if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV else 184


def _flash_dense_hd64_seed_corr_regs(num_kv: int) -> int:
    if num_kv >= 2048:
        return 80
    if num_kv == _FLASH_DENSE_HD64_VERY_LONG_MIN_KV:
        return 64
    return 72 if num_kv > _FLASH_DENSE_HD64_VERY_LONG_MIN_KV else 64


def _flash_dense_hd64_seed_other_regs(num_kv: int) -> int:
    if num_kv >= 2048:
        return 32
    if _FLASH_DENSE_HD64_VERY_LONG_MIN_KV < num_kv <= 512:
        return 32
    return 40 if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV else 48


def _flash_dense_hd64_seed_local_tma_partition(num_kv: int) -> bool:
    return False


def _flash_dense_hd64_seed_tensor_4d_tma(num_kv: int) -> bool:
    return False


def flash_attention_value_prior_weights(
    head_dim: int,
    num_kv: int | None,
    *,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
) -> dict[str, dict[object, float]]:
    """Return per-key sampling weights for flash-attention shape families.

    These weights bias random search toward measured-good FA4 families without
    injecting a complete fixed config. Exact fragment validation happens when
    the backend converts the weights to ``ValuePrior`` objects.
    """
    if head_dim != 64 or num_kv is None:
        return {}
    if requires_ws_overlap or num_kv % 2 != 0:
        return {}

    if not is_causal and not has_kv_tile_pruning:
        if num_kv < _FLASH_DENSE_HD64_MID_MIN_KV:
            return {}
        (
            kv_stage,
            e2e_offset,
            e2e_offset0,
            disc_pipe,
            epi_tma,
            rescale_threshold,
            rescale_chunk_cols,
            packed_reduce,
        ) = _flash_dense_hd64_seed_params(num_kv)
        rescale_threshold = _flash_dense_hd64_seed_rescale_threshold(num_kv)
        p_store_rep = _flash_dense_hd64_seed_p_store_rep(num_kv)
        s_load_rep = _flash_dense_hd64_seed_s_load_rep(num_kv)
        split_p_arrive = _flash_dense_hd64_seed_split_p_arrive(num_kv)
        precompute_qk_desc = _flash_dense_hd64_seed_precompute_qk_desc(num_kv)
        first_load_order = _flash_dense_hd64_seed_first_load_order(num_kv)
        kv_order = _flash_dense_hd64_seed_kv_order(num_kv)
        epi_stg = _flash_dense_hd64_seed_epi_stg(num_kv) and not epi_tma
        corr_tile_size = _flash_dense_hd64_seed_corr_tile_size(num_kv)
        softmax_disc = _flash_dense_hd64_seed_softmax_disc(num_kv)
        softmax_regs = _flash_dense_hd64_seed_softmax_regs(num_kv)
        corr_regs = _flash_dense_hd64_seed_corr_regs(num_kv)
        other_regs = _flash_dense_hd64_seed_other_regs(num_kv)
        e2e_schedule = _flash_dense_hd64_seed_e2e_schedule(num_kv)
        softmax_regs_alternates = (
            (176, 184, 192, 200)
            if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
            else (184, 200)
        )
        skip_rescale_stats = _flash_dense_hd64_seed_skip_rescale_stats(num_kv)
        clc = _flash_dense_hd64_seed_clc(num_kv)
        clc_heads_per_batch = _flash_dense_hd64_seed_clc_heads_per_batch(num_kv)
        persistent_ctas_per_sm = _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv)
        recompute_tile_coords = _flash_dense_hd64_seed_recompute_tile_coords(num_kv)
        local_tma_partition = _flash_dense_hd64_seed_local_tma_partition(num_kv)
        very_long_dense_hd64_fa4 = num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
        priors = cast(
            "dict[str, dict[object, float]]",
            {
                FLASH_S_STAGE_KEY: {2: 1.0},
                FLASH_TOPOLOGY_KEY: {"fa4": 4.0, "ws_overlap": 1.0},
                FLASH_PERSISTENT_KEY: {True: 1.0},
                FLASH_PERSISTENT_CTAS_PER_SM_KEY: {
                    persistent_ctas_per_sm: 4.0,
                    **{
                        ctas_per_sm: 2.0
                        for ctas_per_sm in (1, 2, 3, 4)
                        if ctas_per_sm != persistent_ctas_per_sm
                    },
                },
                FLASH_RECOMPUTE_TILE_COORDS_KEY: {
                    recompute_tile_coords: 4.0,
                    not recompute_tile_coords: 1.0,
                },
                FLASH_E2E_SCHEDULE_KEY: {
                    e2e_schedule: 4.0,
                    **{
                        schedule: 2.0
                        for schedule in ("8/2", "16/4")
                        if schedule != e2e_schedule
                    },
                },
                FLASH_SOFTMAX_DISC_KEY: {
                    softmax_disc: 5.0,
                    not softmax_disc: 1.0,
                },
                FLASH_DISC_PIPE_KEY: {
                    disc_pipe: 4.0,
                    **{depth: 2.0 for depth in (2, 3, 4) if depth != disc_pipe},
                },
                FLASH_SPLIT_P_ARRIVE_KEY: {
                    split_p_arrive: 4.0,
                    not split_p_arrive: 2.0,
                },
                FLASH_P_STORE_REP_KEY: {
                    p_store_rep: 4.0,
                    **{rep: 2.0 for rep in (16, 32) if rep != p_store_rep},
                },
                FLASH_S_LOAD_REP_KEY: {
                    s_load_rep: 4.0,
                    **{rep: 2.0 for rep in (16, 32) if rep != s_load_rep},
                },
                FLASH_PRECOMPUTE_QK_DESC_KEY: {
                    precompute_qk_desc: 4.0,
                    not precompute_qk_desc: 1.0,
                },
                FLASH_FIRST_LOAD_ORDER_KEY: {
                    first_load_order: 4.0,
                    **{
                        order: 1.0
                        for order in (0, 1, 2, 3, 4)
                        if order != first_load_order
                    },
                },
                FLASH_KV_ORDER_KEY: {
                    kv_order: 4.0,
                    **{
                        order: 1.0
                        for order in ("ascending", "descending")
                        if order != kv_order
                    },
                },
                FLASH_RESCALE_THRESHOLD_KEY: (
                    {
                        rescale_threshold: 4.0,
                        **{
                            threshold: 3.0
                            for threshold in (8.0, 16.0, 32.0)
                            if threshold != rescale_threshold
                        },
                    }
                    if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
                    else {8.0: 1.0}
                ),
                FLASH_RESCALE_CHUNK_COLS_KEY: {
                    rescale_chunk_cols: 4.0,
                    **{
                        cols: 2.0
                        for cols in (
                            (8, 16, 32)
                            if num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
                            else (16, 32)
                        )
                        if cols != rescale_chunk_cols
                    },
                },
                FLASH_CORR_REGS_KEY: {
                    corr_regs: 4.0,
                    **{regs: 1.0 for regs in (64, 72, 80, 88) if regs != corr_regs},
                },
                FLASH_OTHER_REGS_KEY: {
                    other_regs: 4.0,
                    **{
                        regs: 2.0
                        for regs in (32, 40, 48, 56, 64, 80)
                        if regs != other_regs
                    },
                },
                FLASH_CORR_TILE_SIZE_KEY: {
                    corr_tile_size: 4.0,
                    **{tile: 2.0 for tile in (8, 16, 32) if tile != corr_tile_size},
                },
                FLASH_EPI_TMA_KEY: {epi_tma: 4.0, not epi_tma: 1.0},
                FLASH_EPI_STG_KEY: {epi_stg: 4.0, not epi_stg: 1.0},
                FLASH_EPI_STG_STORE_KEY: {"slice": 3.0, "whole": 2.0},
                FLASH_EPI_STG_GMEM_KEY: {"stage": 3.0, "pair": 2.0},
                FLASH_SKIP_RESCALE_STATS_KEY: {
                    skip_rescale_stats: 4.0,
                    not skip_rescale_stats: 1.0,
                },
                FLASH_SOFTMAX_REGS_KEY: {
                    softmax_regs: 4.0,
                    **{
                        regs: 2.0
                        for regs in softmax_regs_alternates
                        if regs != softmax_regs
                    },
                },
                FLASH_KV_STAGE_KEY: {
                    kv_stage: 4.0,
                    **{stage: 2.0 for stage in (2, 3) if stage != kv_stage},
                },
                FLASH_PACKED_REDUCE_KEY: {packed_reduce: 4.0, not packed_reduce: 1.0},
                FLASH_CGA2_LOCAL_KEY: (
                    {False: 2.0, True: 2.0} if num_kv % 4 == 0 else {False: 1.0}
                ),
                FLASH_CLC_KEY: {clc: 4.0, not clc: 1.0},
                FLASH_CLC_HEADS_PER_BATCH_KEY: {
                    clc_heads_per_batch: 4.0,
                    **{heads: 2.0 for heads in (0, 32) if heads != clc_heads_per_batch},
                },
                FLASH_CLC_PDL_KEY: {False: 4.0, True: 1.0},
                FLASH_CLC_STAGES_KEY: {1: 4.0, 2: 1.0, 3: 1.0},
                FLASH_LOCAL_TMA_PARTITION_KEY: {
                    local_tma_partition: 4.0,
                    not local_tma_partition: 1.0,
                },
                FLASH_TENSOR_4D_TMA_KEY: {
                    _flash_dense_hd64_seed_tensor_4d_tma(num_kv): 4.0,
                    (not _flash_dense_hd64_seed_tensor_4d_tma(num_kv)): 2.0,
                },
                FLASH_E2E_OFFSET_KEY: {
                    e2e_offset: 4.0,
                    **{offset: 2.0 for offset in (0, 1, 2, 3) if offset != e2e_offset},
                },
                FLASH_E2E_OFFSET0_KEY: {
                    e2e_offset0: 4.0,
                    **{offset: 2.0 for offset in (0, 1, 2, 3) if offset != e2e_offset0},
                },
            },
        )
        if very_long_dense_hd64_fa4:
            priors.update(
                {
                    FLASH_EPI_STG_KEY: {epi_stg: 1.0},
                    FLASH_EPI_STG_STORE_KEY: {"slice": 1.0},
                    FLASH_EPI_STG_GMEM_KEY: {"stage": 1.0},
                    FLASH_PACKED_REDUCE_KEY: {packed_reduce: 1.0},
                    FLASH_CGA2_LOCAL_KEY: {
                        _flash_dense_hd64_seed_cga2_local(num_kv): 1.0
                    },
                    FLASH_SKIP_RESCALE_STATS_KEY: {False: 1.0},
                }
            )
        return priors

    if (
        is_causal
        and not has_kv_tile_pruning
        and _flash_causal_hd64_seed_num_kv_supported(num_kv)
    ):
        e2e_offset, disc_pipe, softmax_regs, lpt_swizzle = (
            _flash_causal_hd64_seed_params(num_kv)
        )
        masked_schedule = _flash_causal_hd64_seed_masked_schedule(num_kv)
        role_map = _flash_causal_hd64_seed_role_map(num_kv)
        epi_tma = _flash_causal_hd64_seed_epi_tma(num_kv)
        rescale_chunk_cols = _flash_causal_hd64_seed_rescale_chunk_cols(num_kv)
        e2e_offset0 = _flash_causal_hd64_seed_offset0(num_kv)
        secondary_offsets = {
            offset: 2.0 for offset in (0, 1, 2, 3, 4, 7, 9, 11) if offset != e2e_offset
        }
        causal_offset_weights = {
            e2e_offset: 4.0,
            **secondary_offsets,
            **{
                offset: 1.0
                for offset in range(16)
                if offset != e2e_offset and offset not in secondary_offsets
            },
        }
        causal_lpt_weights = {
            lpt_swizzle: 4.0,
            **{
                swizzle: 1.0
                for swizzle in (0, 1, 2, 4, 8, 16)
                if swizzle != lpt_swizzle
            },
        }
        causal_disc_weights = {
            disc_pipe: 4.0,
            **{depth: 2.0 for depth in (2, 3, 4) if depth != disc_pipe},
        }
        causal_softmax_weights = {
            softmax_regs: 4.0,
            **{regs: 2.0 for regs in (184, 200) if regs != softmax_regs},
            **({192: 1.0} if softmax_regs != 192 else {}),
        }
        masked_schedule_weights = {
            masked_schedule: 4.0,
            **{
                schedule: 2.0
                for schedule in ("inherit", "xu", "16/4", "8/2")
                if schedule != masked_schedule
            },
        }
        role_map_weights = {
            role_map: 4.0,
            **{role: 1.0 for role in ("helion", "fa4") if role != role_map},
        }
        epi_tma_weights = {epi_tma: 4.0, not epi_tma: 1.0}
        rescale_chunk_weights = {
            rescale_chunk_cols: 4.0,
            **{cols: 2.0 for cols in (16, 32) if cols != rescale_chunk_cols},
        }
        return cast(
            "dict[str, dict[object, float]]",
            {
                FLASH_TOPOLOGY_KEY: {"fa4": 4.0, "ws_overlap": 1.0},
                FLASH_PERSISTENT_KEY: {False: 1.0},
                FLASH_KV_STAGE_KEY: {
                    2: 4.0,
                    3: 2.0,
                    4: 1.0,
                    6: 2.0,
                    8: 2.0,
                    10: 2.0,
                },
                FLASH_E2E_SCHEDULE_KEY: {"8/2": 4.0, "16/4": 2.0, "xu": 1.0},
                FLASH_MASKED_E2E_SCHEDULE_KEY: masked_schedule_weights,
                FLASH_SOFTMAX_DISC_KEY: {True: 1.0},
                FLASH_PACKED_REDUCE_KEY: {True: 1.0},
                FLASH_EPI_TMA_KEY: epi_tma_weights,
                FLASH_RESCALE_CHUNK_COLS_KEY: rescale_chunk_weights,
                FLASH_ROLE_MAP_KEY: role_map_weights,
                FLASH_CAUSAL_LOOP_SPLIT_KEY: {True: 4.0, False: 1.0},
                FLASH_CAUSAL_LPT_SWIZZLE_KEY: causal_lpt_weights,
                FLASH_DISC_PIPE_KEY: causal_disc_weights,
                FLASH_SOFTMAX_REGS_KEY: causal_softmax_weights,
                FLASH_E2E_OFFSET_KEY: causal_offset_weights,
                FLASH_E2E_OFFSET0_KEY: {
                    e2e_offset0: 4.0,
                    **{
                        offset: 2.0
                        for offset in (0, 1, 2, 3, 4, 8, 11)
                        if offset != e2e_offset0
                    },
                },
                FLASH_CAUSAL_KV_ORDER_KEY: _flash_causal_hd64_kv_order_weights(num_kv),
            },
        )

    return {}


def _flash_seed_set(
    seed: dict[str, object],
    fragments: Mapping[str, ConfigSpecFragment],
    key: str,
    value: object,
) -> bool:
    fragment = fragments[key]
    assert isinstance(fragment, EnumFragment)
    if value not in fragment.choices:
        return False
    seed[key] = value
    return True


def _flash_seed_set_all(
    seed: dict[str, object],
    fragments: Mapping[str, ConfigSpecFragment],
    values: Mapping[str, object],
) -> bool:
    for key, value in values.items():
        if not _flash_seed_set(seed, fragments, key, value):
            return False
    return True


def _flash_seed_fragments(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    topology_override: str | None,
) -> dict[str, ConfigSpecFragment]:
    return flash_autotune_fragments(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        topology_override=topology_override,
    )


def _flash_seed_block_sizes(block_size_targets: Sequence[int]) -> list[int] | None:
    targets = tuple(block_size_targets)
    if targets != _FLASH_SEED_BLOCK_SIZE_TARGETS:
        return None
    return [*targets]


def _flash_default_seed_config(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    block_size_targets: Sequence[int],
) -> Config | None:
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None

    topology = (
        "ws_overlap" if requires_ws_overlap else "fa4" if num_kv % 2 == 0 else None
    )
    fragments = _flash_seed_fragments(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        topology_override=topology,
    )
    seed: dict[str, object] = {"block_sizes": block_sizes}

    if small_biased_candidate:
        _flash_seed_set(seed, fragments, FLASH_SMALL_BIASED_KEY, True)
    if topology is not None and not _flash_seed_set(
        seed, fragments, FLASH_TOPOLOGY_KEY, topology
    ):
        return None

    dense_hd64_fa4 = (
        topology == "fa4"
        and not is_causal
        and head_dim == 64
        and num_kv >= _FLASH_DENSE_HD64_MID_MIN_KV
    )
    if dense_hd64_fa4:
        (
            kv_stage,
            e2e_offset,
            e2e_offset0,
            disc_pipe,
            epi_tma,
            rescale_threshold,
            rescale_chunk_cols,
            packed_reduce,
        ) = _flash_dense_hd64_seed_params(num_kv)
        rescale_threshold = _flash_dense_hd64_seed_rescale_threshold(num_kv)
        family_values = {
            FLASH_S_STAGE_KEY: 2,
            FLASH_KV_STAGE_KEY: kv_stage,
            FLASH_PERSISTENT_KEY: True,
            FLASH_PERSISTENT_CTAS_PER_SM_KEY: (
                _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv)
            ),
            FLASH_RECOMPUTE_TILE_COORDS_KEY: (
                _flash_dense_hd64_seed_recompute_tile_coords(num_kv)
            ),
            FLASH_E2E_SCHEDULE_KEY: _flash_dense_hd64_seed_e2e_schedule(num_kv),
            FLASH_E2E_OFFSET_KEY: e2e_offset,
            FLASH_E2E_OFFSET0_KEY: e2e_offset0,
            FLASH_SOFTMAX_DISC_KEY: _flash_dense_hd64_seed_softmax_disc(num_kv),
            FLASH_DISC_PIPE_KEY: disc_pipe,
            FLASH_SPLIT_P_ARRIVE_KEY: _flash_dense_hd64_seed_split_p_arrive(num_kv),
            FLASH_P_STORE_REP_KEY: _flash_dense_hd64_seed_p_store_rep(num_kv),
            FLASH_S_LOAD_REP_KEY: _flash_dense_hd64_seed_s_load_rep(num_kv),
            FLASH_PRECOMPUTE_QK_DESC_KEY: _flash_dense_hd64_seed_precompute_qk_desc(
                num_kv
            ),
            FLASH_FIRST_LOAD_ORDER_KEY: _flash_dense_hd64_seed_first_load_order(num_kv),
            FLASH_KV_ORDER_KEY: _flash_dense_hd64_seed_kv_order(num_kv),
            FLASH_SOFTMAX_REGS_KEY: _flash_dense_hd64_seed_softmax_regs(num_kv),
            FLASH_CORR_REGS_KEY: _flash_dense_hd64_seed_corr_regs(num_kv),
            FLASH_OTHER_REGS_KEY: _flash_dense_hd64_seed_other_regs(num_kv),
            FLASH_EPI_TMA_KEY: epi_tma,
            FLASH_EPI_STG_KEY: _flash_dense_hd64_seed_epi_stg(num_kv) and not epi_tma,
            FLASH_EPI_STG_STORE_KEY: "slice",
            FLASH_EPI_STG_GMEM_KEY: "stage",
            FLASH_CORR_TILE_SIZE_KEY: _flash_dense_hd64_seed_corr_tile_size(num_kv),
            FLASH_RESCALE_THRESHOLD_KEY: rescale_threshold,
            FLASH_SKIP_RESCALE_STATS_KEY: _flash_dense_hd64_seed_skip_rescale_stats(
                num_kv
            ),
            FLASH_RESCALE_CHUNK_COLS_KEY: rescale_chunk_cols,
            FLASH_PACKED_REDUCE_KEY: packed_reduce,
            FLASH_CGA2_LOCAL_KEY: _flash_dense_hd64_seed_cga2_local(num_kv),
            FLASH_CLC_KEY: _flash_dense_hd64_seed_clc(num_kv),
            FLASH_CLC_HEADS_PER_BATCH_KEY: _flash_dense_hd64_seed_clc_heads_per_batch(
                num_kv
            ),
            FLASH_CLC_PDL_KEY: False,
            FLASH_CLC_STAGES_KEY: 1,
            FLASH_LOCAL_TMA_PARTITION_KEY: _flash_dense_hd64_seed_local_tma_partition(
                num_kv
            ),
            FLASH_TENSOR_4D_TMA_KEY: _flash_dense_hd64_seed_tensor_4d_tma(num_kv),
        }
        role_map = _flash_dense_hd64_seed_role_map(num_kv)
        if role_map != "helion":
            family_values[FLASH_ROLE_MAP_KEY] = role_map
        if not _flash_seed_set_all(seed, fragments, family_values):
            return None

    if is_causal:
        if head_dim == 64:
            _flash_seed_set(seed, fragments, FLASH_PACKED_REDUCE_KEY, True)
        elif head_dim == 128:
            _flash_seed_set(seed, fragments, FLASH_KV_STAGE_KEY, 2)
    if has_kv_tile_pruning:
        _flash_seed_set(seed, fragments, FLASH_PACKED_REDUCE_KEY, True)
    return Config.from_dict(seed)


def _flash_dense_sp_seed_config(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    block_size_targets: Sequence[int],
) -> Config | None:
    # This is a manual experiment seed for the resident softmax variant. It is
    # intentionally not emitted by flash_attention_seed_configs() until that
    # family is faster and stable enough for automatic autotune exploration.
    if (
        is_causal
        or has_kv_tile_pruning
        or requires_ws_overlap
        or small_biased_candidate
        or head_dim != 64
        or num_kv < _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
    ):
        return None
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None

    fragments = _flash_seed_fragments(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        topology_override="fa4",
    )
    (
        kv_stage,
        e2e_offset,
        e2e_offset0,
        _disc_pipe,
        epi_tma,
        rescale_threshold,
        rescale_chunk_cols,
        packed_reduce,
    ) = _flash_dense_hd64_seed_params(num_kv)
    seed: dict[str, object] = {"block_sizes": block_sizes}
    family_values: dict[str, object] = {
        FLASH_TOPOLOGY_KEY: "fa4",
        FLASH_S_STAGE_KEY: 2,
        FLASH_KV_STAGE_KEY: kv_stage,
        FLASH_PERSISTENT_KEY: True,
        FLASH_PERSISTENT_CTAS_PER_SM_KEY: (
            _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv)
        ),
        FLASH_RECOMPUTE_TILE_COORDS_KEY: (
            _flash_dense_hd64_seed_recompute_tile_coords(num_kv)
        ),
        FLASH_E2E_SCHEDULE_KEY: "8/2",
        FLASH_E2E_OFFSET_KEY: e2e_offset,
        FLASH_E2E_OFFSET0_KEY: e2e_offset0,
        FLASH_SOFTMAX_DISC_KEY: False,
        FLASH_DISC_PIPE_KEY: 1,
        FLASH_SPLIT_P_ARRIVE_KEY: False,
        FLASH_P_STORE_REP_KEY: 32,
        FLASH_S_LOAD_REP_KEY: 32,
        FLASH_PRECOMPUTE_QK_DESC_KEY: True,
        FLASH_FIRST_LOAD_ORDER_KEY: 1,
        FLASH_KV_ORDER_KEY: "descending",
        FLASH_SOFTMAX_REGS_KEY: 200,
        FLASH_CORR_REGS_KEY: 80,
        FLASH_OTHER_REGS_KEY: 32,
        FLASH_EPI_TMA_KEY: epi_tma,
        FLASH_EPI_STG_KEY: _flash_dense_hd64_seed_epi_stg(num_kv) and not epi_tma,
        FLASH_EPI_STG_STORE_KEY: "slice",
        FLASH_EPI_STG_GMEM_KEY: "stage",
        FLASH_CORR_TILE_SIZE_KEY: _flash_dense_hd64_seed_corr_tile_size(num_kv),
        FLASH_RESCALE_THRESHOLD_KEY: rescale_threshold,
        FLASH_SKIP_RESCALE_STATS_KEY: _flash_dense_hd64_seed_skip_rescale_stats(num_kv),
        FLASH_RESCALE_CHUNK_COLS_KEY: rescale_chunk_cols,
        FLASH_PACKED_REDUCE_KEY: packed_reduce,
        FLASH_CGA2_LOCAL_KEY: False,
        FLASH_CLC_KEY: False,
        FLASH_CLC_PDL_KEY: False,
        FLASH_CLC_STAGES_KEY: 1,
        FLASH_LOCAL_TMA_PARTITION_KEY: True,
        FLASH_TENSOR_4D_TMA_KEY: True,
    }
    if not _flash_seed_set_all(seed, fragments, family_values):
        return None
    return Config.from_dict(seed)


def _flash_causal_lpt_seed_config(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    block_size_targets: Sequence[int],
) -> Config | None:
    if (
        not is_causal
        or head_dim != 64
        or requires_ws_overlap
        or not _flash_causal_hd64_seed_num_kv_supported(num_kv)
    ):
        return None
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None

    fragments = _flash_seed_fragments(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        topology_override="fa4",
    )
    e2e_offset, disc_pipe, softmax_regs, lpt_swizzle = _flash_causal_hd64_seed_params(
        num_kv
    )
    offset0 = _flash_causal_hd64_seed_offset0(num_kv)
    seed: dict[str, object] = {"block_sizes": block_sizes}
    family_values: dict[str, object] = {
        FLASH_TOPOLOGY_KEY: "fa4",
        FLASH_S_STAGE_KEY: 2,
        FLASH_KV_STAGE_KEY: 2,
        FLASH_PERSISTENT_KEY: False,
        FLASH_E2E_SCHEDULE_KEY: "8/2",
        FLASH_MASKED_E2E_SCHEDULE_KEY: _flash_causal_hd64_seed_masked_schedule(num_kv),
        FLASH_E2E_OFFSET_KEY: e2e_offset,
        FLASH_E2E_OFFSET0_KEY: offset0,
        FLASH_DISC_PIPE_KEY: disc_pipe,
        FLASH_EPI_TMA_KEY: _flash_causal_hd64_seed_epi_tma(num_kv),
        FLASH_RESCALE_CHUNK_COLS_KEY: _flash_causal_hd64_seed_rescale_chunk_cols(
            num_kv
        ),
        FLASH_ROLE_MAP_KEY: _flash_causal_hd64_seed_role_map(num_kv),
        FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_PACKED_REDUCE_KEY: True,
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: lpt_swizzle,
        FLASH_CAUSAL_KV_ORDER_KEY: _flash_causal_hd64_seed_kv_order(num_kv),
        FLASH_SOFTMAX_REGS_KEY: softmax_regs,
    }
    if not _flash_seed_set_all(seed, fragments, family_values):
        return None
    return Config.from_dict(seed)


def _flash_causal_split_seed_config(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    block_size_targets: Sequence[int],
) -> Config | None:
    if (
        not is_causal
        or head_dim != 64
        or requires_ws_overlap
        or not _flash_causal_hd64_seed_num_kv_supported(num_kv)
    ):
        return None
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None

    fragments = _flash_seed_fragments(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        topology_override="fa4",
    )
    (
        e2e_offset,
        _disc_pipe,
        softmax_regs,
        lpt_swizzle,
    ) = _flash_causal_hd64_seed_params(num_kv)
    split_masked_schedule = "16/4"
    split_lpt_swizzle = 8 if num_kv <= 128 else lpt_swizzle
    split_disc_pipe = 2 if num_kv <= 128 else 4
    split_role_map = "fa4" if num_kv <= 128 else "helion"
    split_offset0 = 0 if num_kv <= 128 else _flash_causal_hd64_seed_offset0(num_kv)
    seed: dict[str, object] = {"block_sizes": block_sizes}
    family_values: dict[str, object] = {
        FLASH_TOPOLOGY_KEY: "fa4",
        FLASH_S_STAGE_KEY: 2,
        FLASH_KV_STAGE_KEY: 2,
        FLASH_PERSISTENT_KEY: False,
        FLASH_E2E_SCHEDULE_KEY: "8/2",
        FLASH_MASKED_E2E_SCHEDULE_KEY: split_masked_schedule,
        FLASH_E2E_OFFSET_KEY: e2e_offset,
        FLASH_E2E_OFFSET0_KEY: split_offset0,
        FLASH_DISC_PIPE_KEY: split_disc_pipe,
        FLASH_EPI_TMA_KEY: _flash_causal_hd64_seed_epi_tma(num_kv),
        FLASH_RESCALE_CHUNK_COLS_KEY: _flash_causal_hd64_seed_rescale_chunk_cols(
            num_kv
        ),
        FLASH_ROLE_MAP_KEY: split_role_map,
        FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_PACKED_REDUCE_KEY: True,
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: split_lpt_swizzle,
        FLASH_CAUSAL_KV_ORDER_KEY: _flash_causal_hd64_seed_kv_order(num_kv),
        FLASH_SOFTMAX_REGS_KEY: softmax_regs,
    }
    if not _flash_seed_set_all(seed, fragments, family_values):
        return None
    return Config.from_dict(seed)


def flash_attention_seed_config(
    head_dim: int,
    num_kv: int | None,
    *,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
    seed_kind: str = "default",
) -> Config | None:
    """Return a fragment-valid seed config for the detected flash surface.

    The seed policy is shape-family based rather than exact sequence-length
    based. ``block_size_targets`` comes from the flash detector's search-surface
    facts; this helper only emits a seed for the fused 1x128x128 envelope.
    """
    if num_kv is None:
        return None
    if seed_kind == "default":
        return _flash_default_seed_config(
            head_dim,
            num_kv,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            block_size_targets=block_size_targets,
        )
    if seed_kind == "causal_lpt":
        return _flash_causal_lpt_seed_config(
            head_dim,
            num_kv,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            block_size_targets=block_size_targets,
        )
    if seed_kind == "dense_sp":
        return _flash_dense_sp_seed_config(
            head_dim,
            num_kv,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            block_size_targets=block_size_targets,
        )
    if seed_kind == "causal_split":
        return _flash_causal_split_seed_config(
            head_dim,
            num_kv,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            block_size_targets=block_size_targets,
        )
    raise AssertionError(f"unknown flash attention seed kind: {seed_kind!r}")


def flash_attention_seed_configs(
    head_dim: int,
    num_kv: int | None,
    *,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
) -> tuple[Config, ...]:
    seeds: list[Config] = []
    default_seed = flash_attention_seed_config(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        block_size_targets=block_size_targets,
    )
    if default_seed is not None:
        seeds.append(default_seed)
    causal_lpt_seed = flash_attention_seed_config(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        block_size_targets=block_size_targets,
        seed_kind="causal_lpt",
    )
    if causal_lpt_seed is not None:
        seeds.append(causal_lpt_seed)
    causal_split_seed = flash_attention_seed_config(
        head_dim,
        num_kv,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        block_size_targets=block_size_targets,
        seed_kind="causal_split",
    )
    if causal_split_seed is not None:
        seeds.append(causal_split_seed)
    return tuple(seeds)


def flash_autotune_fragments(
    head_dim: int,
    num_kv: int,
    *,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    topology_override: str | None = None,
) -> dict[str, ConfigSpecFragment]:
    """Return ``{config_key: ConfigSpecFragment}`` for the flash autotune surface.

    The defaults dataclass (env/shape resolution) supplies the FIRST element of
    each ``EnumFragment`` so the seed/default config is byte-identical to the
    current kernel. Only the known-safe, meaningful knobs are exposed:

      * ``s_stage`` — warp-spec double-buffered-S overlap (1 vs 2).
      * ``kv_stage`` — K/V TMA ring depth. Most flash kernels keep the validated
        2/3-stage envelope; dense hd64 FA4 additionally accepts sparse deeper
        manual/cache-transfer values while actively searching only the stable
        2/3-stage subset.
      * ``persistent`` — static-persistent scheduler on/off.
      * ``e2e_schedule`` — paired exp2-pipe-split schedule choices. This
        replaces the old independent exp2/freq/res search, which produced
        meaningless duplicate configs and heavily regressive pairings. Long
        dense hd64 FA4 shapes search only the validated ``16/4`` and ``8/2``
        schedules; the wider schedule set remains available as manual
        overrides.
      * ``e2e_offset`` — stage-1 residue phase offset for the split exp2
        schedule, exposed across the full phase range required by any selectable
        split schedule so the autotuner can desynchronize the two softmax
        warpgroups without relying on a sparse hand-picked subset. Long dense
        hd64 FA4 shapes use a seed-local offset set to avoid repeatedly
        benchmarking very slow high-phase candidates.
      * ``e2e_offset0`` — stage-0 residue phase offset. This defaults to 0 to
        preserve the FA4-aligned program order, but gives the autotuner the same
        phase freedom for the first softmax warpgroup when a shape benefits from
        different XU/FMA pipe staggering.
      * ``topology`` — the device-body topology selector. ``ws_overlap`` (the
        fallback) plus ``fa4`` (the commit-38ff4d1a win, ~77% SDPA) WHEN the
        shape meets the fa4 2-Q-tile envelope (``seq % 256 == 0`` i.e.
        ``num_kv`` even; head_dim is already pinned to {64, 128} by the
        detector, and ``resolve_flash_config`` clamps FA4's aliased K/V ring to
        at least 2 stages). Causal even-KV shapes also use FA4, but with the
        persistent scheduler disabled; odd-KV shapes accept stale cached
        ``fa4`` configs and resolve them back to ``ws_overlap`` before codegen.
      * ``disc_pipe`` / ``epi_tma`` / ``rescale_threshold`` / ``packed_reduce`` —
        fa4/ws perf levers for PASS2 software-pipelining, epilogue TMA-store,
        alpha-pin O-rescale skipping, rescale chunk width, and packed ws row
        reductions. ``disc_pipe`` offers the small validated depth set
        {1, 2, 3, 4}; ``epi_tma`` and ``packed_reduce`` remain booleans;
        ``rescale_threshold`` uses a narrow enum around FA4's 8.0 fp16/bf16
        constant; ``rescale_chunk_cols`` searches the characterized
        16/32-column set. Very long dense hd64 FA4 shapes search the measured
        8.0 and 12.0 thresholds; manual overrides remain available. Larger
        64-column chunks remain available as manual overrides but are not
        searched because they can exceed ptxas register-allocation
        limits on FA4 shapes. ``resolve_flash_config`` gates ``epi_tma`` on
        topology, so stale values are harmless choices for the ws_overlap
        default.
      * ``causal_lpt_swizzle`` — causal-only launch-order/cache-locality swizzle.
        ``0`` preserves the shape-derived L2 heuristic; positive values force a
        compact set of batch/head interleave group sizes.
      * ``causal_kv_order`` — causal-only KV traversal order. The original
        Helion stream walks left-to-right; upstream FA4 walks diagonal-first and
        then backward. Both orders remain accepted for manual/cache-transfer
        configs; causal hd64 actively searches the diagonal-first descending
        stream so split-loop candidates are meaningful rather than clamped away.
      * ``small_biased`` — enables the small contiguous biased-attention SIMT
        emitter for its validated [BH, 128, 64] envelope. Non-candidate shapes
        carry the key as a fixed default so cached configs round-trip without
        widening unrelated searches.
      * ``softmax_regs`` / ``corr_regs`` — FA4 softmax/correction warpgroup
        register grants. These stay in narrow characterized sets so promoted
        seeds survive flatten/unflatten without exposing the rest of the
        register-budget surface.

    The remaining register budgets (``num_regs_producer``/``num_regs_consumer``)
    and warp-count fields (``num_*_warps``, ``num_softmax_warpgroups``,
    ``acc_stage``, ``epi_stage``, ``tmem_plan``, ``tmem_s_to_p_offset``) are kept
    FIXED here: they have no validated multi-value envelope today, so exposing
    them would let the autotuner pick combinations the emitter cannot legally
    build. They are sourced from ``resolve_flash_config`` (env/shape) as single
    values and can be promoted to fragments once characterized.
    """
    env_topology = os.environ.get("HELION_CUTE_FLASH_TOPOLOGY")
    valid_topology_override = (
        topology_override if topology_override in ("fa4", "ws_overlap") else None
    )
    topology_config = "ws_overlap" if requires_ws_overlap else valid_topology_override
    if topology_config is None and env_topology is not None:
        topology_config = (
            env_topology if env_topology in ("fa4", "ws_overlap") else None
        )
    defaults = resolve_flash_config(
        head_dim,
        num_kv,
        (
            {FLASH_TOPOLOGY_KEY: topology_config}
            if topology_config is not None
            else None
        ),
        is_causal=is_causal,
        prefer_packed_reduce=has_kv_tile_pruning or requires_ws_overlap,
    )
    dense_hd64_fa4 = not is_causal and head_dim == 64 and defaults.topology == "fa4"
    very_long_dense_hd64_fa4 = (
        dense_hd64_fa4 and num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
    )
    causal_hd64_seeded_fa4 = (
        is_causal
        and head_dim == 64
        and _flash_causal_hd64_seed_num_kv_supported(num_kv)
        and defaults.topology == "fa4"
    )
    long_causal_hd64_seeded_fa4 = (
        causal_hd64_seeded_fa4 and num_kv >= _FLASH_CAUSAL_HD64_LONG_AUTOTUNE_MIN_KV
    )
    (
        causal_e2e_offset_search,
        causal_disc_pipe_search,
        causal_softmax_regs_search,
        causal_lpt_swizzle_search,
    ) = _flash_causal_hd64_seed_params(num_kv)
    if causal_softmax_regs_search == 0:
        causal_softmax_regs_search = defaults.softmax_regs
    if causal_lpt_swizzle_search == 0:
        causal_lpt_swizzle_search = defaults.causal_lpt_swizzle
    # ``s_stage`` is forced to 1 for num_kv < 2 (single KV tile); do not offer 2
    # as a choice there (the resolver would override it anyway, but keep the
    # surface honest). ``persistent`` requires s_stage == 2 for ws_overlap and is
    # disabled for causal FA4 to match upstream's nonpersistent causal scheduler.
    # Keep true in the enum as a cache-compatibility value for best-config
    # transfer from dense runs.
    if num_kv < 2:
        s_stage_choices: tuple[int, ...] = (1,)
        s_stage_search_choices: tuple[int, ...] | None = None
    elif dense_hd64_fa4 or causal_hd64_seeded_fa4:
        s_stage_choices = _flash_choices_with_default(defaults.s_stage, (1, 2))
        s_stage_search_choices = _flash_choices_with_default(defaults.s_stage, (2,))
    else:
        # Seed with the resolved default first (byte-identity), then the other.
        s_stage_choices = (defaults.s_stage, 1 if defaults.s_stage == 2 else 2)
        s_stage_search_choices = None
    if dense_hd64_fa4:
        kv_stage_choices = _flash_choices_with_default(
            defaults.kv_stage, (2, 3, 4, 6, 8, 10)
        )
        kv_stage_search_choices: tuple[int, ...] | None = (
            _flash_choices_with_default(defaults.kv_stage, (2, 3))
            if very_long_dense_hd64_fa4
            else (
                2,
                3,
                4,
                6,
                8,
            )
        )
    elif causal_hd64_seeded_fa4:
        kv_stage_choices = _flash_choices_with_default(
            defaults.kv_stage, (2, 3, 4, 6, 8, 10)
        )
        kv_stage_search_choices = (
            (2,)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(defaults.kv_stage, (10, 8, 6, 4, 3, 2))
        )
    else:
        kv_stage_choices = (defaults.kv_stage, 3 if defaults.kv_stage == 2 else 2)
        kv_stage_search_choices = None

    if dense_hd64_fa4:
        persistent_choices = _flash_choices_with_default(
            defaults.persistent, (True, False)
        )
        persistent_search_choices: tuple[bool, ...] | None = (
            _flash_choices_with_default(defaults.persistent, (True,))
        )
        persistent_ctas_per_sm_choices = _flash_choices_with_default(
            defaults.persistent_ctas_per_sm, (1, 2, 3, 4)
        )
        persistent_ctas_per_sm_search_choices: tuple[int, ...] | None = (
            (defaults.persistent_ctas_per_sm,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_persistent_ctas_per_sm(num_kv), (1, 2, 3, 4)
            )
        )
        recompute_tile_coords_choices = _flash_choices_with_default(
            defaults.recompute_tile_coords, (False, True)
        )
        recompute_tile_coords_search_choices: tuple[bool, ...] | None = (
            defaults.recompute_tile_coords,
        )
        e2e_schedule_default = _flash_e2e_schedule_default(defaults.topology, head_dim)
        e2e_schedule_choices = _flash_choices_with_default(
            defaults.e2e_schedule, ("16/4", "8/2", "16/2", "xu")
        )
        e2e_schedule_search_choices: tuple[str, ...] | None = (
            _flash_choices_with_default(defaults.e2e_schedule, ("8/2", "16/4"))
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(defaults.e2e_schedule, ("16/4", "8/2"))
        )
    elif causal_hd64_seeded_fa4:
        persistent_choices = _flash_choices_with_default(defaults.persistent, (False,))
        persistent_search_choices = _flash_choices_with_default(
            defaults.persistent, (False,)
        )
        persistent_ctas_per_sm_choices = (defaults.persistent_ctas_per_sm,)
        persistent_ctas_per_sm_search_choices = None
        recompute_tile_coords_choices = (defaults.recompute_tile_coords,)
        recompute_tile_coords_search_choices = None
        e2e_schedule_default = _flash_e2e_schedule_default(defaults.topology, head_dim)
        e2e_schedule_choices = _flash_choices_with_default(
            defaults.e2e_schedule, ("16/4", "8/2", "16/2", "xu")
        )
        e2e_schedule_search_choices = (
            ("8/2",)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                defaults.e2e_schedule, ("xu", "16/4", "8/2")
            )
        )
    else:
        persistent_choices = (
            defaults.persistent,
            not defaults.persistent,
        )
        persistent_search_choices = None
        persistent_ctas_per_sm_choices = (defaults.persistent_ctas_per_sm,)
        persistent_ctas_per_sm_search_choices = None
        recompute_tile_coords_choices = (defaults.recompute_tile_coords,)
        recompute_tile_coords_search_choices = None
        e2e_schedule_search_set = ("16/4", "8/2", "16/2", "xu")
        e2e_schedule_default = (
            defaults.e2e_schedule
            if defaults.e2e_schedule in e2e_schedule_search_set
            else e2e_schedule_search_set[0]
        )
        e2e_schedule_choices = (
            e2e_schedule_default,
            *(
                schedule
                for schedule in e2e_schedule_search_set
                if schedule != e2e_schedule_default
            ),
        )
        e2e_schedule_search_choices = None
    if is_causal:
        masked_e2e_schedule_choices = _flash_choices_with_default(
            defaults.masked_e2e_schedule, ("inherit", "xu", "16/4", "8/2")
        )
        masked_e2e_schedule_search_choices: tuple[str, ...] | None = (
            (_flash_causal_hd64_seed_masked_schedule(num_kv),)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                _flash_causal_hd64_seed_masked_schedule(num_kv),
                ("inherit", "xu", "16/4", "8/2"),
            )
            if causal_hd64_seeded_fa4
            else None
        )
    else:
        masked_e2e_schedule_choices = ("inherit",)
        masked_e2e_schedule_search_choices = None
    # Topology: default to fa4 only when the shape meets the fa4 2-Q-tile envelope.
    # The fa4 device body processes a PAIR of adjacent 128-row Q-tiles per CTA, so
    # it requires ``seq % 256 == 0``. The codegen already gates seq % 128 == 0 (so
    # seq == num_kv * 128); ``seq % 256 == 0`` is therefore exactly ``num_kv`` even.
    # Ineligible shapes still include fa4 as a cache-compatibility enum value, but
    # resolve_flash_config() clamps it to ws_overlap before codegen. The FIRST
    # choice is the env/shape-resolved default. With no env override, eligible
    # shapes default to fa4 so the default config exercises the faster topology;
    # invalid env values are sanitized by resolve_flash_config().
    fa4_eligible = num_kv % 2 == 0
    topology_search_choices: tuple[str, ...] | None = None
    if requires_ws_overlap:
        topology_choices = ("ws_overlap",)
        topology_search_choices = None
    elif dense_hd64_fa4 or causal_hd64_seeded_fa4:
        topology_choices = _flash_choices_with_default(
            defaults.topology, ("fa4", "ws_overlap")
        )
        topology_search_defaults = ("fa4", "ws_overlap")
        topology_search_choices = (
            ("fa4",)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                defaults.topology, topology_search_defaults
            )
        )
    elif fa4_eligible:
        # Seed with the resolved default first, then the other topology.
        topology_choices: tuple[str, ...] = (
            defaults.topology,
            "fa4" if defaults.topology != "fa4" else "ws_overlap",
        )
    else:
        # Ineligible shapes still accept stale cached fa4 configs so best-available
        # autotune handoff can transfer dense runs to causal/odd-KV runs. The
        # resolver clamps fa4 back to ws_overlap before codegen, so this is only
        # cache compatibility, not a second device topology.
        topology_choices = ("ws_overlap", "fa4")
    if dense_hd64_fa4:
        softmax_disc_choices = _flash_choices_with_default(
            defaults.softmax_disc, (True, False)
        )
        softmax_disc_search_choices: tuple[bool, ...] | None = (
            (defaults.softmax_disc,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(defaults.softmax_disc, (True, False))
        )
    else:
        softmax_disc_choices = (defaults.softmax_disc,)
        softmax_disc_search_choices = (
            (defaults.softmax_disc,) if causal_hd64_seeded_fa4 else None
        )
    # fa4 disc PASS2 software-pipeline depth lever. The first choice is the
    # shape-resolved default; the remaining validated depths are accepted for
    # autotune/cache transfer. They are consumed only by the fa4 emitter.
    if dense_hd64_fa4 or causal_hd64_seeded_fa4:
        disc_pipe_choices = _flash_choices_with_default(
            defaults.disc_pipe_depth,
            (1, 2, 3, 4),
        )
        disc_pipe_search_choices: tuple[int, ...] | None = (
            (causal_disc_pipe_search,)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(causal_disc_pipe_search, (2, 3, 4))
            if causal_hd64_seeded_fa4
            else (defaults.disc_pipe_depth,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(defaults.disc_pipe_depth, (2, 3, 4))
        )
    else:
        disc_pipe_choices = (
            defaults.disc_pipe_depth,
            *(depth for depth in (1, 2, 3, 4) if depth != defaults.disc_pipe_depth),
        )
        disc_pipe_search_choices = None
    split_p_arrive_choices = _flash_choices_with_default(
        defaults.split_p_arrive, (False, True)
    )
    split_p_arrive_search_choices: tuple[bool, ...] | None = (
        _flash_choices_with_default(defaults.split_p_arrive, (False, True))
        if very_long_dense_hd64_fa4 and num_kv >= 2048
        else (defaults.split_p_arrive,)
        if very_long_dense_hd64_fa4
        else _flash_choices_with_default(defaults.split_p_arrive, (False, True))
        if dense_hd64_fa4
        else (defaults.split_p_arrive,)
    )
    if dense_hd64_fa4:
        p_store_rep_choices = _flash_choices_with_default(
            defaults.p_store_repetition, (16, 32)
        )
        p_store_rep_search_choices: tuple[int, ...] | None = (
            (defaults.p_store_repetition,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_p_store_rep(num_kv), (16, 32)
            )
        )
        s_load_rep_choices = _flash_choices_with_default(
            defaults.s_load_repetition, (16, 32)
        )
        s_load_rep_search_choices: tuple[int, ...] | None = (
            (defaults.s_load_repetition,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_s_load_rep(num_kv), (16, 32)
            )
        )
        precompute_qk_desc_choices = _flash_choices_with_default(
            defaults.precompute_qk_desc, (False, True)
        )
        precompute_qk_desc_search_choices: tuple[bool, ...] | None = (
            (defaults.precompute_qk_desc,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_precompute_qk_desc(num_kv), (False, True)
            )
        )
        first_load_order_choices = _flash_choices_with_default(
            defaults.first_load_order, (0, 1, 2, 3, 4)
        )
        first_load_order_search_choices: tuple[int, ...] | None = (
            (defaults.first_load_order,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_first_load_order(num_kv), (0, 1, 2, 3, 4)
            )
        )
        kv_order_choices = _flash_choices_with_default(
            defaults.kv_order, ("ascending", "descending")
        )
        kv_order_search_choices: tuple[str, ...] | None = (
            (defaults.kv_order,)
            if very_long_dense_hd64_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_kv_order(num_kv), ("ascending", "descending")
            )
        )
    else:
        p_store_rep_choices = (defaults.p_store_repetition,)
        p_store_rep_search_choices = None
        s_load_rep_choices = (defaults.s_load_repetition,)
        s_load_rep_search_choices = None
        precompute_qk_desc_choices = (defaults.precompute_qk_desc,)
        precompute_qk_desc_search_choices = None
        first_load_order_choices = (defaults.first_load_order,)
        first_load_order_search_choices = None
        kv_order_choices = (defaults.kv_order,)
        kv_order_search_choices = None
    max_e2e_freq = 0
    for schedule in e2e_schedule_choices:
        _impl, freq, res = _flash_parse_e2e_schedule(schedule, e2e_schedule_default)
        if res > 0:
            max_e2e_freq = max(max_e2e_freq, freq)
    if max_e2e_freq > 0:
        offset_search_values: Iterable[int]
        offset_search_values = range(max_e2e_freq)
        if dense_hd64_fa4:
            e2e_offset_default = defaults.e2e_offset
            e2e_offset_search_choices: tuple[int, ...] | None = (
                (defaults.e2e_offset,)
                if very_long_dense_hd64_fa4
                else _flash_choices_with_default(defaults.e2e_offset, range(16))
            )
        elif causal_hd64_seeded_fa4:
            e2e_offset_default = defaults.e2e_offset
            e2e_offset_search_choices = (
                (causal_e2e_offset_search,)
                if long_causal_hd64_seeded_fa4
                else _flash_choices_with_default(
                    causal_e2e_offset_search,
                    range(16),
                )
            )
        else:
            e2e_offset_default = defaults.e2e_offset
            e2e_offset_search_choices = None
        e2e_offset_choices = (
            e2e_offset_default,
            *(
                offset
                for offset in offset_search_values
                if offset != e2e_offset_default
            ),
        )
        offset0_search_values: Iterable[int] = range(max_e2e_freq)
        if dense_hd64_fa4:
            e2e_offset0_default = defaults.e2e_offset0
            e2e_offset0_search_choices: tuple[int, ...] | None = (
                (defaults.e2e_offset0,)
                if very_long_dense_hd64_fa4
                else _flash_choices_with_default(defaults.e2e_offset0, range(16))
            )
        elif causal_hd64_seeded_fa4:
            e2e_offset0_default = defaults.e2e_offset0
            e2e_offset0_search_choices = (
                (_flash_causal_hd64_seed_offset0(num_kv),)
                if long_causal_hd64_seeded_fa4
                else _flash_choices_with_default(
                    _flash_causal_hd64_seed_offset0(num_kv), range(16)
                )
            )
        else:
            e2e_offset0_default = defaults.e2e_offset0
            e2e_offset0_search_choices = None
        e2e_offset0_choices = (
            e2e_offset0_default,
            *(
                offset
                for offset in offset0_search_values
                if offset != e2e_offset0_default
            ),
        )
    else:
        e2e_offset_choices = (defaults.e2e_offset,)
        e2e_offset0_choices = (defaults.e2e_offset0,)
        e2e_offset_search_choices = None
        e2e_offset0_search_choices = None
    if dense_hd64_fa4 or causal_hd64_seeded_fa4:
        rescale_threshold_choices = _flash_choices_with_default(
            defaults.rescale_threshold, (0.0, 4.0, 8.0, 12.0, 16.0, 32.0)
        )
        rescale_threshold_search_defaults = (
            (32.0, 16.0, 8.0)
            if very_long_dense_hd64_fa4
            else (8.0, 12.0, 32.0)
            if dense_hd64_fa4 and num_kv >= _FLASH_DENSE_HD64_VERY_LONG_MIN_KV
            else (8.0,)
        )
        rescale_threshold_search_choices: tuple[float, ...] | None = (
            _flash_choices_with_default(
                defaults.rescale_threshold, rescale_threshold_search_defaults
            )
        )
        rescale_chunk_cols_manual_values = (
            (8, 16, 32, 64) if very_long_dense_hd64_fa4 else (16, 32, 64)
        )
        rescale_chunk_cols_choices = _flash_choices_with_default(
            defaults.rescale_chunk_cols,
            (cols for cols in rescale_chunk_cols_manual_values if head_dim % cols == 0),
        )
        rescale_chunk_cols_search_choices: tuple[int, ...] | None = (
            (defaults.rescale_chunk_cols,)
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                _flash_causal_hd64_seed_rescale_chunk_cols(num_kv)
                if causal_hd64_seeded_fa4
                else defaults.rescale_chunk_cols,
                (16, 32),
            )
        )
        skip_rescale_stats_choices = _flash_choices_with_default(
            defaults.skip_rescale_stats, (False, True)
        )
        skip_rescale_stats_search_choices: tuple[bool, ...] | None = (
            (False,)
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                _flash_dense_hd64_seed_skip_rescale_stats(num_kv)
                if dense_hd64_fa4
                else defaults.skip_rescale_stats,
                (False, True),
            )
        )
    else:
        rescale_threshold_choices = (
            defaults.rescale_threshold,
            *(
                threshold
                for threshold in (0.0, 4.0, 8.0, 12.0, 16.0, 32.0)
                if threshold != defaults.rescale_threshold
            ),
        )
        rescale_chunk_cols_choices = _flash_choices_with_default(
            defaults.rescale_chunk_cols,
            (cols for cols in (16, 32, 64) if head_dim % cols == 0),
        )
        rescale_chunk_cols_search_choices = _flash_choices_with_default(
            defaults.rescale_chunk_cols,
            (cols for cols in (16, 32) if head_dim % cols == 0),
        )
        rescale_threshold_search_choices = None
        skip_rescale_stats_choices = (defaults.skip_rescale_stats,)
        skip_rescale_stats_search_choices = None
    if dense_hd64_fa4:
        softmax_regs_choices = _flash_choices_with_default(
            defaults.softmax_regs, (176, 184, 192, 200)
        )
        softmax_regs_search_choices: tuple[int, ...] | None = (
            _flash_choices_with_default(defaults.softmax_regs, (200, 192))
            if very_long_dense_hd64_fa4
            else None
        )
    elif causal_hd64_seeded_fa4:
        softmax_regs_choices = _flash_choices_with_default(
            defaults.softmax_regs, (176, 184, 192, 200)
        )
        softmax_regs_search_choices = (
            (causal_softmax_regs_search,)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                causal_softmax_regs_search, (184, 192, 200)
            )
        )
    else:
        softmax_regs_choices = (
            defaults.softmax_regs,
            *(regs for regs in (176, 184, 192, 200) if regs != defaults.softmax_regs),
        )
        softmax_regs_search_choices = None
    corr_regs_manual_choices = (64, 72, 80, 88)
    if dense_hd64_fa4 or causal_hd64_seeded_fa4:
        corr_regs_choices = _flash_choices_with_default(
            defaults.corr_regs, corr_regs_manual_choices
        )
        corr_regs_search_choices: tuple[int, ...] | None = _flash_choices_with_default(
            defaults.corr_regs,
            (defaults.corr_regs,) if very_long_dense_hd64_fa4 else (64,),
        )
    else:
        corr_regs_choices = (
            defaults.corr_regs,
            *(regs for regs in corr_regs_manual_choices if regs != defaults.corr_regs),
        )
        corr_regs_search_choices = (
            defaults.corr_regs,
            *(regs for regs in (64, 88) if regs != defaults.corr_regs),
        )
    other_regs_manual_choices = (32, 40, 48, 56, 64, 80)
    other_regs_choices = _flash_choices_with_default(
        defaults.other_regs, other_regs_manual_choices
    )
    other_regs_search_choices: tuple[int, ...] | None = (
        (defaults.other_regs,)
        if very_long_dense_hd64_fa4
        else _flash_choices_with_default(defaults.other_regs, other_regs_manual_choices)
        if dense_hd64_fa4
        else (defaults.other_regs,)
        if causal_hd64_seeded_fa4
        else None
    )
    corr_tile_candidates = tuple(tile for tile in (8, 16, 32) if head_dim % tile == 0)
    corr_tile_choices = _flash_choices_with_default(
        defaults.corr_tile_size, corr_tile_candidates
    )
    if dense_hd64_fa4:
        corr_tile_search_choices: tuple[int, ...] | None = _flash_choices_with_default(
            defaults.corr_tile_size
            if very_long_dense_hd64_fa4
            else _flash_dense_hd64_seed_corr_tile_size(num_kv),
            (8, 16) if very_long_dense_hd64_fa4 else corr_tile_candidates,
        )
    elif causal_hd64_seeded_fa4:
        corr_tile_search_choices = _flash_choices_with_default(
            defaults.corr_tile_size, (16, 8, 32)
        )
    else:
        corr_tile_search_choices = None
    if dense_hd64_fa4 or causal_hd64_seeded_fa4:
        packed_reduce_choices = _flash_choices_with_default(
            defaults.packed_reduce, (True, False)
        )
        packed_reduce_search_defaults = (
            (defaults.packed_reduce,)
            if very_long_dense_hd64_fa4
            else (True,)
            if causal_hd64_seeded_fa4 or num_kv >= 64
            else (False,)
        )
        packed_reduce_search_choices: tuple[bool, ...] | None = (
            _flash_choices_with_default(
                defaults.packed_reduce, packed_reduce_search_defaults
            )
        )
    else:
        packed_reduce_choices = (defaults.packed_reduce, not defaults.packed_reduce)
        packed_reduce_search_choices = None
    if is_causal:
        causal_lpt_swizzle_choices = (
            defaults.causal_lpt_swizzle,
            *(
                swizzle
                for swizzle in (0, 1, 2, 4, 8, 16, 32, 64)
                if swizzle != defaults.causal_lpt_swizzle
            ),
        )
        causal_lpt_swizzle_search_choices: tuple[int, ...] | None = (
            (causal_lpt_swizzle_search,)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                causal_lpt_swizzle_search, (0, 1, 2, 4, 8, 16)
            )
            if causal_hd64_seeded_fa4
            else None
        )
        causal_kv_order_choices = _flash_choices_with_default(
            defaults.causal_kv_order, ("ascending", "descending")
        )
        causal_kv_order_search_choices: tuple[str, ...] | None = (
            (_flash_causal_hd64_seed_kv_order(num_kv),)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                _flash_causal_hd64_seed_kv_order(num_kv),
                ("ascending", "descending"),
            )
            if causal_hd64_seeded_fa4
            else None
        )
    else:
        causal_lpt_swizzle_choices = (0,)
        causal_lpt_swizzle_search_choices = None
        causal_kv_order_choices = ("ascending",)
        causal_kv_order_search_choices = None
    role_map_choices = _flash_choices_with_default(defaults.role_map, ("helion", "fa4"))
    role_map_search_choices: tuple[str, ...] | None = (
        (_flash_causal_hd64_seed_role_map(num_kv),)
        if long_causal_hd64_seeded_fa4
        else _flash_choices_with_default(
            _flash_causal_hd64_seed_role_map(num_kv), ("helion", "fa4")
        )
        if causal_hd64_seeded_fa4
        else _flash_choices_with_default(defaults.role_map, ("helion", "fa4"))
        if very_long_dense_hd64_fa4
        else (defaults.role_map,)
    )
    dense_hd128_fa4 = (
        topology_choices[0] == "fa4"
        and not is_causal
        and head_dim == 128
        and num_kv % 4 == 0
    )
    use_2cta_eligible = dense_hd128_fa4
    use_2cta_choices = (
        _flash_choices_with_default(defaults.use_2cta_instrs, (False, True))
        if use_2cta_eligible
        else (False,)
    )
    use_2cta_search_choices: tuple[bool, ...] | None = (
        use_2cta_choices if use_2cta_eligible else (False,)
    )
    use_cga2_local_eligible = (
        dense_hd64_fa4 and not is_causal and head_dim == 64 and num_kv % 4 == 0
    )
    use_cga2_local_choices = (
        _flash_choices_with_default(defaults.use_cga2_local_cta, (False, True))
        if use_cga2_local_eligible
        else (False,)
    )
    use_cga2_local_search_choices: tuple[bool, ...] | None = (
        # The very-long hd64 target already saturates with one CTA per tile, so
        # two-local-CTA variants remain manual.
        (defaults.use_cga2_local_cta,)
        if very_long_dense_hd64_fa4
        else use_cga2_local_choices
        if use_cga2_local_eligible
        else (False,)
    )
    use_clc_eligible = (
        dense_hd64_fa4
        and not is_causal
        and head_dim == 64
        and num_kv % 2 == 0
        and defaults.persistent
    )
    use_clc_choices = (
        _flash_choices_with_default(defaults.use_clc_scheduler, (True, False))
        if use_clc_eligible
        else (False,)
    )
    use_clc_search_choices: tuple[bool, ...] | None = (
        (defaults.use_clc_scheduler,)
        if use_clc_eligible and very_long_dense_hd64_fa4
        else use_clc_choices
        if use_clc_eligible
        else (False,)
    )
    clc_heads_choices = _flash_choices_with_default(
        defaults.clc_heads_per_batch, (0, 1, 2, 4, 8, 16, 32, 64)
    )
    clc_heads_search_choices: tuple[int, ...] | None = (
        (defaults.clc_heads_per_batch,)
        if use_clc_eligible and very_long_dense_hd64_fa4
        else (0, 32, 16, 64)
        if use_clc_eligible
        else (0,)
    )
    clc_pdl_choices = (
        _flash_choices_with_default(defaults.clc_use_pdl, (False, True))
        if use_clc_eligible
        else (False,)
    )
    clc_pdl_search_choices: tuple[bool, ...] | None = (
        (defaults.clc_use_pdl,)
        if use_clc_eligible and very_long_dense_hd64_fa4
        else clc_pdl_choices
        if use_clc_eligible
        else (False,)
    )
    clc_stages_choices = (
        _flash_choices_with_default(defaults.clc_stages, (1, 2, 3))
        if use_clc_eligible
        else (1,)
    )
    clc_stages_search_choices: tuple[int, ...] | None = (
        (defaults.clc_stages,)
        if use_clc_eligible and very_long_dense_hd64_fa4
        else clc_stages_choices
        if use_clc_eligible
        else (1,)
    )
    local_tma_eligible = dense_hd64_fa4 and not is_causal and head_dim == 64
    local_tma_choices = (
        _flash_choices_with_default(defaults.local_tma_partition, (False, True))
        if local_tma_eligible
        else (False,)
    )
    local_tma_search_choices: tuple[bool, ...] | None = (
        (defaults.local_tma_partition,)
        if very_long_dense_hd64_fa4
        else local_tma_choices
        if local_tma_eligible
        else (False,)
    )
    tensor_4d_tma_choices = (
        _flash_choices_with_default(defaults.tensor_4d_tma, (False, True))
        if local_tma_eligible
        else (False,)
    )
    tensor_4d_tma_search_choices: tuple[bool, ...] | None = (
        (defaults.tensor_4d_tma,)
        if very_long_dense_hd64_fa4
        else tensor_4d_tma_choices
        if local_tma_eligible
        else (False,)
    )
    causal_loop_split_choices = (
        defaults.causal_loop_split,
        not defaults.causal_loop_split,
    )
    causal_loop_split_search_choices: tuple[bool, ...] | None = (
        (True,)
        if long_causal_hd64_seeded_fa4
        else (True, False)
        if causal_hd64_seeded_fa4
        else (defaults.causal_loop_split,)
    )
    small_biased_choices = _flash_choices_with_default(
        defaults.small_biased, (True, False)
    )
    small_biased_search_choices: tuple[bool, ...] | None = (
        small_biased_choices if small_biased_candidate else (defaults.small_biased,)
    )
    # fa4 epilogue TMA-store lever. Include TMA-O in search for FA4 hd64 shape
    # families; value priors keep the characterized default favored without
    # making it fixed.
    if dense_hd64_fa4 or causal_hd64_seeded_fa4:
        epi_tma_choices = _flash_choices_with_default(defaults.epi_tma, (False, True))
        epi_tma_search_defaults = (False, True)
        epi_tma_search_choices: tuple[bool, ...] | None = (
            (_flash_causal_hd64_seed_epi_tma(num_kv),)
            if long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(
                defaults.epi_tma
                if very_long_dense_hd64_fa4
                else _flash_causal_hd64_seed_epi_tma(num_kv)
                if causal_hd64_seeded_fa4
                else defaults.epi_tma,
                (defaults.epi_tma,)
                if very_long_dense_hd64_fa4
                else epi_tma_search_defaults,
            )
        )
        epi_stg_choices = _flash_choices_with_default(defaults.epi_stg, (False, True))
        epi_stg_search_choices: tuple[bool, ...] | None = _flash_choices_with_default(
            defaults.epi_stg
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else _flash_dense_hd64_seed_epi_stg(num_kv)
            if dense_hd64_fa4
            else defaults.epi_stg,
            (defaults.epi_stg,)
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else epi_tma_search_defaults,
        )
        epi_stg_store_choices = _flash_choices_with_default(
            defaults.epi_stg_store, ("slice", "whole")
        )
        epi_stg_store_search_choices: tuple[str, ...] | None = (
            (defaults.epi_stg_store,)
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(defaults.epi_stg_store, ("slice", "whole"))
        )
        epi_stg_gmem_choices = _flash_choices_with_default(
            defaults.epi_stg_gmem, ("stage", "pair")
        )
        epi_stg_gmem_search_choices: tuple[str, ...] | None = (
            (defaults.epi_stg_gmem,)
            if very_long_dense_hd64_fa4 or long_causal_hd64_seeded_fa4
            else _flash_choices_with_default(defaults.epi_stg_gmem, ("stage", "pair"))
        )
    else:
        epi_tma_choices = (defaults.epi_tma, not defaults.epi_tma)
        epi_tma_search_choices = None
        epi_stg_choices = (defaults.epi_stg,)
        epi_stg_search_choices = None
        epi_stg_store_choices = (defaults.epi_stg_store,)
        epi_stg_store_search_choices = None
        epi_stg_gmem_choices = (defaults.epi_stg_gmem,)
        epi_stg_gmem_search_choices = None
    return {
        FLASH_S_STAGE_KEY: EnumFragment(s_stage_choices, s_stage_search_choices),
        FLASH_KV_STAGE_KEY: EnumFragment(kv_stage_choices, kv_stage_search_choices),
        FLASH_PERSISTENT_KEY: EnumFragment(
            persistent_choices, persistent_search_choices
        ),
        FLASH_PERSISTENT_CTAS_PER_SM_KEY: EnumFragment(
            persistent_ctas_per_sm_choices,
            persistent_ctas_per_sm_search_choices,
        ),
        FLASH_RECOMPUTE_TILE_COORDS_KEY: EnumFragment(
            recompute_tile_coords_choices,
            recompute_tile_coords_search_choices,
        ),
        FLASH_E2E_SCHEDULE_KEY: EnumFragment(
            e2e_schedule_choices, e2e_schedule_search_choices
        ),
        FLASH_MASKED_E2E_SCHEDULE_KEY: EnumFragment(
            masked_e2e_schedule_choices, masked_e2e_schedule_search_choices
        ),
        FLASH_E2E_OFFSET_KEY: EnumFragment(
            e2e_offset_choices, e2e_offset_search_choices
        ),
        FLASH_E2E_OFFSET0_KEY: EnumFragment(
            e2e_offset0_choices, e2e_offset0_search_choices
        ),
        FLASH_TOPOLOGY_KEY: EnumFragment(topology_choices, topology_search_choices),
        FLASH_SOFTMAX_DISC_KEY: EnumFragment(
            softmax_disc_choices, softmax_disc_search_choices
        ),
        FLASH_DISC_PIPE_KEY: EnumFragment(disc_pipe_choices, disc_pipe_search_choices),
        FLASH_SPLIT_P_ARRIVE_KEY: EnumFragment(
            split_p_arrive_choices, split_p_arrive_search_choices
        ),
        FLASH_P_STORE_REP_KEY: EnumFragment(
            p_store_rep_choices, p_store_rep_search_choices
        ),
        FLASH_S_LOAD_REP_KEY: EnumFragment(
            s_load_rep_choices, s_load_rep_search_choices
        ),
        FLASH_PRECOMPUTE_QK_DESC_KEY: EnumFragment(
            precompute_qk_desc_choices, precompute_qk_desc_search_choices
        ),
        FLASH_FIRST_LOAD_ORDER_KEY: EnumFragment(
            first_load_order_choices, first_load_order_search_choices
        ),
        FLASH_KV_ORDER_KEY: EnumFragment(kv_order_choices, kv_order_search_choices),
        FLASH_EPI_TMA_KEY: EnumFragment(epi_tma_choices, epi_tma_search_choices),
        FLASH_EPI_STG_KEY: EnumFragment(epi_stg_choices, epi_stg_search_choices),
        FLASH_EPI_STG_STORE_KEY: EnumFragment(
            epi_stg_store_choices, epi_stg_store_search_choices
        ),
        FLASH_EPI_STG_GMEM_KEY: EnumFragment(
            epi_stg_gmem_choices, epi_stg_gmem_search_choices
        ),
        FLASH_RESCALE_THRESHOLD_KEY: EnumFragment(
            rescale_threshold_choices, rescale_threshold_search_choices
        ),
        FLASH_SKIP_RESCALE_STATS_KEY: EnumFragment(
            skip_rescale_stats_choices, skip_rescale_stats_search_choices
        ),
        FLASH_RESCALE_CHUNK_COLS_KEY: EnumFragment(
            rescale_chunk_cols_choices, rescale_chunk_cols_search_choices
        ),
        FLASH_SOFTMAX_REGS_KEY: EnumFragment(
            softmax_regs_choices, softmax_regs_search_choices
        ),
        FLASH_CORR_REGS_KEY: EnumFragment(corr_regs_choices, corr_regs_search_choices),
        FLASH_OTHER_REGS_KEY: EnumFragment(
            other_regs_choices, other_regs_search_choices
        ),
        FLASH_CORR_TILE_SIZE_KEY: EnumFragment(
            corr_tile_choices, corr_tile_search_choices
        ),
        FLASH_PACKED_REDUCE_KEY: EnumFragment(
            packed_reduce_choices, packed_reduce_search_choices
        ),
        FLASH_SMALL_BIASED_KEY: EnumFragment(
            small_biased_choices, small_biased_search_choices
        ),
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: EnumFragment(
            causal_lpt_swizzle_choices, causal_lpt_swizzle_search_choices
        ),
        FLASH_CAUSAL_KV_ORDER_KEY: EnumFragment(
            causal_kv_order_choices, causal_kv_order_search_choices
        ),
        FLASH_ROLE_MAP_KEY: EnumFragment(role_map_choices, role_map_search_choices),
        FLASH_USE_2CTA_KEY: EnumFragment(use_2cta_choices, use_2cta_search_choices),
        FLASH_CGA2_LOCAL_KEY: EnumFragment(
            use_cga2_local_choices, use_cga2_local_search_choices
        ),
        FLASH_CLC_KEY: EnumFragment(use_clc_choices, use_clc_search_choices),
        FLASH_CLC_HEADS_PER_BATCH_KEY: EnumFragment(
            clc_heads_choices, clc_heads_search_choices
        ),
        FLASH_CLC_PDL_KEY: EnumFragment(clc_pdl_choices, clc_pdl_search_choices),
        FLASH_CLC_STAGES_KEY: EnumFragment(
            clc_stages_choices, clc_stages_search_choices
        ),
        FLASH_LOCAL_TMA_PARTITION_KEY: EnumFragment(
            local_tma_choices, local_tma_search_choices
        ),
        FLASH_TENSOR_4D_TMA_KEY: EnumFragment(
            tensor_4d_tma_choices, tensor_4d_tma_search_choices
        ),
        FLASH_CAUSAL_LOOP_SPLIT_KEY: EnumFragment(
            causal_loop_split_choices, causal_loop_split_search_choices
        ),
    }


def flash_config_from_config(
    config: Mapping[str, object],
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype = torch.float16,
    *,
    is_causal: bool = False,
) -> FlashAttentionConfig:
    """Reconstruct ``FlashAttentionConfig`` from a (normalized) config Mapping.

    Thin wrapper over ``resolve_flash_config`` that prefers values present in
    *config* (the autotuner-chosen knobs in ``FLASH_CONFIG_KEYS``) and falls
    back to the env/shape resolution for every key the config does not carry.
    """
    return resolve_flash_config(
        head_dim,
        num_kv,
        config,
        dtype=dtype,
        is_causal=is_causal,
    )


if TYPE_CHECKING:
    from ...autotuner.config_fragment import ConfigSpecFragment
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


# The SharedStorage struct (Gotcha G: stringified annotations break
# ``@cute.struct``) and the inline-traced rescale helper live in
# ``_flash_runtime`` (a real module compiled WITHOUT ``from __future__ import
# annotations``); the generated module imports them. The remaining cute / utils
# / pipeline symbols are imported under flash-local aliases.
_FLASH_PREAMBLE_IMPORTS = """\
import cutlass.utils as cutlass_utils_flash
import cutlass.pipeline as cutlass_pipeline_flash
from cutlass.cute.nvgpu import cpasync as cute_cpasync_flash
from cutlass.cute.nvgpu import tcgen05 as cute_tcgen05_flash
import cutlass.utils.blackwell_helpers as sm100_utils_flash
import helion._compiler.cute._flash_runtime as _helion_flash_rt
import helion._compiler.cute._flash_gemm_ptx as _helion_flash_ptx
"""


def emit_flash_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module flash imports."""
    if getattr(cg, "_helion_flash_module_emitted", False):
        return
    cg._helion_flash_module_emitted = True  # type: ignore[attr-defined]
    for line_stmt in ast.parse(_FLASH_PREAMBLE_IMPORTS).body:
        cg.module_statements.append(line_stmt)


def _small_biased_attention_gmem_expr(
    tensor_name: str,
    dim0: str,
    dim1: str,
    dim2: str,
    *,
    strides: tuple[int, int, int] | None = None,
) -> str:
    stride0, stride1, stride2 = strides or (None, None, None)
    stride0_expr = (
        f"cutlass.Int32({stride0})"
        if stride0 is not None
        else f"cutlass.Int32({tensor_name}.layout.stride[0])"
    )
    stride1_expr = (
        f"cutlass.Int32({stride1})"
        if stride1 is not None
        else f"cutlass.Int32({tensor_name}.layout.stride[1])"
    )
    stride2_expr = (
        f"cutlass.Int32({stride2})"
        if stride2 is not None
        else f"cutlass.Int32({tensor_name}.layout.stride[2])"
    )
    return (
        f"({tensor_name}.iterator + {dim0} * {stride0_expr} "
        f"+ {dim1} * {stride1_expr} "
        f"+ {dim2} * {stride2_expr})"
    )


def _small_biased_qkv_strides(seq: int, head_dim: int) -> tuple[int, int, int]:
    return (seq * head_dim, head_dim, 1)


def _small_biased_bias_strides(seq: int) -> tuple[int, int, int]:
    return (seq * seq, seq, 1)


def _small_biased_attention_score_pair_expr(
    q_name: str,
    k_name: str,
    bias_name: str,
    *,
    seq: int,
    head_dim: int,
) -> str:
    qkv_strides = _small_biased_qkv_strides(seq, head_dim)
    bias_strides = _small_biased_bias_strides(seq)
    q_load = _small_biased_attention_gmem_expr(
        q_name,
        "small_bh",
        "small_row",
        "small_d",
        strides=qkv_strides,
    )
    k0_load = _small_biased_attention_gmem_expr(
        k_name,
        "small_bh",
        "small_col0",
        "small_d",
        strides=qkv_strides,
    )
    k1_load = _small_biased_attention_gmem_expr(
        k_name,
        "small_bh",
        "small_col1",
        "small_d",
        strides=qkv_strides,
    )
    bias0_load = _small_biased_attention_gmem_expr(
        bias_name,
        "small_bh",
        "small_row",
        "small_col0",
        strides=bias_strides,
    )
    bias1_load = _small_biased_attention_gmem_expr(
        bias_name,
        "small_bh",
        "small_row",
        "small_col1",
        strides=bias_strides,
    )
    return f"""small_dot0 = cutlass.Float32(0.0)
small_dot1 = cutlass.Float32(0.0)
for small_d in cutlass.range({head_dim}, unroll=2):
    small_q = {q_load}.load()
    small_k0 = {k0_load}.load()
    small_k1 = {k1_load}.load()
    small_dot0 += cutlass.Float32(small_q) * cutlass.Float32(small_k0)
    small_dot1 += cutlass.Float32(small_q) * cutlass.Float32(small_k1)
small_score0 = small_dot0 * small_qk_scale + cutlass.Float32({bias0_load}.load())
small_score1 = small_dot1 * small_qk_scale + cutlass.Float32({bias1_load}.load())"""


def _small_biased_attention_reduce_block(
    op: str,
    reduce_tensor: str,
    value: str,
) -> str:
    assert op in {"max", "sum"}
    reduction = "max" if op == "max" else "sum"
    combine = (
        "small_partial0 if small_partial0 > small_partial1 else small_partial1"
        if op == "max"
        else "small_partial0 + small_partial1"
    )
    return f"""small_warp_{op} = cute.arch.warp_reduction_{reduction}(
    {value}, threads_in_group=32)
if small_lane == cutlass.Int32(0):
    {reduce_tensor}[small_warp] = small_warp_{op}
cute.arch.sync_threads()
small_partial0 = {reduce_tensor}[cutlass.Int32(0)]
small_partial1 = {reduce_tensor}[cutlass.Int32(1)]
small_block_{op} = {combine}"""


def _small_biased_attention_simt_supported(
    score_plan: AttentionScorePlan,
    *,
    seq: int,
    head_dim: int,
    io_dtype: torch.dtype,
    lse_arg: TensorArg | None,
    bias_args: Sequence[TensorArg],
) -> bool:
    return _small_biased_attention_score_plan_supported(
        score_plan,
        seq=seq,
        head_dim=head_dim,
        io_dtype=io_dtype,
        has_lse=lse_arg is not None,
        bias_arg_count=len(bias_args),
    )


def emit_small_biased_attention_simt_device_body(
    *,
    q_name: str,
    k_name: str,
    v_name: str,
    o_name: str,
    bias_name: str,
    seq: int,
    head_dim: int,
    io_dtype: str,
) -> list[ast.AST]:
    score_body = _small_biased_attention_score_pair_expr(
        q_name,
        k_name,
        bias_name,
        seq=seq,
        head_dim=head_dim,
    )
    max_reduce = _small_biased_attention_reduce_block(
        "max",
        "small_max_reduce",
        "small_thread_max",
    )
    sum_reduce = _small_biased_attention_reduce_block(
        "sum",
        "small_sum_reduce",
        "small_p0 + small_p1",
    )
    v_load = _small_biased_attention_gmem_expr(
        v_name,
        "small_bh",
        "small_kv",
        "small_col",
        strides=_small_biased_qkv_strides(seq, head_dim),
    )
    o_store = _small_biased_attention_gmem_expr(
        o_name,
        "small_bh",
        "small_row",
        "small_col",
        strides=_small_biased_qkv_strides(seq, head_dim),
    )
    qk_scale = 1.0 / math.sqrt(head_dim)
    body = f"""
small_tid, _, _ = cute.arch.thread_idx()
small_col = cutlass.Int32(small_tid)
small_col0 = small_col * cutlass.Int32(2)
small_col1 = small_col0 + cutlass.Int32(1)
small_warp = small_col // cutlass.Int32(32)
small_lane = small_col - small_warp * cutlass.Int32(32)
small_row = cutlass.Int32(cute.arch.block_idx()[0])
small_bh = cutlass.Int32(cute.arch.block_idx()[1])
small_qk_scale = cutlass.Float32({qk_scale!r})
small_log2e = cutlass.Float32(1.4426950408889634)
small_prob_ptr = cute.arch.alloc_smem(cutlass.Float32, {seq}, alignment=128)
small_max_reduce_ptr = cute.arch.alloc_smem(cutlass.Float32, {seq}, alignment=128)
small_sum_reduce_ptr = cute.arch.alloc_smem(cutlass.Float32, {seq}, alignment=128)
small_prob = cute.make_tensor(
    small_prob_ptr, cute.make_layout(({seq},), stride=(1,)))
small_max_reduce = cute.make_tensor(
    small_max_reduce_ptr, cute.make_layout(({seq},), stride=(1,)))
small_sum_reduce = cute.make_tensor(
    small_sum_reduce_ptr, cute.make_layout(({seq},), stride=(1,)))
{score_body}
small_prob[small_col0] = small_score0
small_prob[small_col1] = small_score1
small_thread_max = small_score0 if small_score0 > small_score1 else small_score1
{max_reduce}
small_row_max = small_block_max
small_score0 = small_prob[small_col0]
small_score1 = small_prob[small_col1]
small_has_finite_score = small_row_max > cutlass.Float32(-cutlass.Float32.inf)
small_row_max_safe = (
    small_row_max if small_has_finite_score else cutlass.Float32(0.0))
small_p0 = cute.math.exp2(
    (small_score0 - small_row_max_safe) * small_log2e, fastmath=True)
small_p1 = cute.math.exp2(
    (small_score1 - small_row_max_safe) * small_log2e, fastmath=True)
small_prob[small_col0] = small_p0
small_prob[small_col1] = small_p1
{sum_reduce}
small_row_sum = small_block_sum
small_row_sum_safe = (
    small_row_sum
    if small_row_sum > cutlass.Float32(0.0)
    else cutlass.Float32(1.0)
)
small_inv_sum = cutlass.Float32(1.0) / small_row_sum_safe
small_out_acc = cutlass.Float32(0.0)
for small_kv in cutlass.range({seq}, unroll=16):
    small_pv_p = small_prob[small_kv]
    small_v = {v_load}.load()
    small_out_acc += small_pv_p * cutlass.Float32(small_v)
{o_store}.store({io_dtype}(small_out_acc * small_inv_sum))
"""
    return list(ast.parse(textwrap.dedent(body)).body)


def _flash_score_transform_block(
    score_plan: AttentionScorePlan,
    *,
    indent: str,
    score_tensor: str,
    coord_tensor: str,
    bh_expr: str,
    m_tile_expr: str,
    kv_tile_expr: str,
    causal_condition: str | None = None,
    chunk_expr: str | None = None,
    io_dtype: str | None = None,
) -> str:
    lines: list[str] = []
    for modifier in score_plan.modifiers:
        if modifier.kind == TENSOR_BIAS_KIND:
            if chunk_expr is None or io_dtype is None:
                lines.append(
                    f"{indent}_helion_flash_rt.add_score_bias_t2r("
                    f"{score_tensor}, {coord_tensor}, _flash_mBias, {bh_expr}, "
                    f"{m_tile_expr}, {kv_tile_expr}, _flash_score_bias_scale)"
                )
            else:
                lines.append(
                    f"{indent}_helion_flash_rt.add_score_bias_t2r_contiguous("
                    f"{score_tensor}, {coord_tensor}, _flash_mBias, {bh_expr}, "
                    f"{m_tile_expr}, {kv_tile_expr}, _flash_score_bias_scale, "
                    f"{io_dtype})"
                )
        elif modifier.kind == RELATIVE_BIAS_KIND:
            bias_scale = modifier.scale_log2 / score_plan.qk_scale_log2
            lines.append(
                f"{indent}_helion_flash_rt.add_relative_bias_t2r("
                f"{score_tensor}, {coord_tensor}, {m_tile_expr}, {kv_tile_expr}, "
                f"cutlass.Float32({bias_scale!r}))"
            )
        elif modifier.kind == ALIBI_BIAS_KIND:
            bias_scale = modifier.scale_log2 / score_plan.qk_scale_log2
            lines.append(
                f"{indent}_helion_flash_rt.add_alibi_bias_t2r("
                f"{score_tensor}, {coord_tensor}, _flash_mAlibi, {bh_expr}, "
                f"_flash_num_alibi, {m_tile_expr}, {kv_tile_expr}, "
                f"cutlass.Float32({bias_scale!r}))"
            )
        elif modifier.kind == CAUSAL_MASK_KIND:
            if chunk_expr is None:
                mask_line = (
                    f"_helion_flash_rt.causal_mask_t2r("
                    f"{score_tensor}, {coord_tensor}, {m_tile_expr}, {kv_tile_expr})"
                )
            else:
                mask_line = (
                    f"_helion_flash_rt.causal_mask_t2r_chunk("
                    f"{score_tensor}, {coord_tensor}, {m_tile_expr}, {kv_tile_expr}, "
                    f"{chunk_expr})"
                )
            if causal_condition is None:
                lines.append(f"{indent}{mask_line}")
            else:
                lines.extend(
                    (
                        f"{indent}if {causal_condition}:",
                        f"{indent}    {mask_line}",
                    )
                )
        elif modifier.kind == SLIDING_WINDOW_MASK_KIND:
            assert modifier.window_size is not None
            lines.append(
                f"{indent}_helion_flash_rt.sliding_window_mask_t2r("
                f"{score_tensor}, {coord_tensor}, {m_tile_expr}, {kv_tile_expr}, "
                f"cutlass.Int32({modifier.window_size}))"
            )
        elif modifier.kind == PREFIX_LM_MASK_KIND:
            assert modifier.prefix_length is not None
            lines.append(
                f"{indent}_helion_flash_rt.prefix_lm_mask_t2r("
                f"{score_tensor}, {coord_tensor}, {m_tile_expr}, {kv_tile_expr}, "
                f"cutlass.Int32({modifier.prefix_length}))"
            )
        elif modifier.kind == DOCUMENT_MASK_KIND:
            lines.append(
                f"{indent}_helion_flash_rt.document_mask_t2r("
                f"{score_tensor}, {coord_tensor}, _flash_mDoc, {bh_expr}, "
                f"_flash_doc_heads_per_batch, {m_tile_expr}, {kv_tile_expr})"
            )
        elif modifier.kind == SOFTCAP_KIND:
            assert modifier.value_log2 is not None
            lines.append(
                f"{indent}_helion_flash_rt.softcap_t2r("
                f"{score_tensor}, _flash_scale_log2, "
                f"cutlass.Float32({modifier.value_log2!r}))"
            )
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def _flash_kv_tile_active_condition(
    score_plan: AttentionScorePlan,
    *,
    bh_expr: str,
    m_tile_expr: str,
    kv_tile_expr: str,
) -> str | None:
    conditions: list[str] = []
    for modifier in score_plan.modifiers:
        if modifier.kind == CAUSAL_MASK_KIND:
            conditions.append(f"{kv_tile_expr} <= {m_tile_expr}")
        elif modifier.kind == SLIDING_WINDOW_MASK_KIND:
            assert modifier.window_size is not None
            back_tiles = (modifier.window_size + 127) // 128
            conditions.append(
                f"({kv_tile_expr} <= {m_tile_expr})"
                f" & ({kv_tile_expr} + cutlass.Int32({back_tiles}) >= {m_tile_expr})"
            )
        elif modifier.kind == PREFIX_LM_MASK_KIND:
            assert modifier.prefix_length is not None
            conditions.append(
                f"({kv_tile_expr} <= {m_tile_expr})"
                f" | (({kv_tile_expr}) * cutlass.Int32(128)"
                f" < cutlass.Int32({modifier.prefix_length}))"
            )
        elif modifier.kind == DOCUMENT_MASK_KIND:
            conditions.append(
                "_helion_flash_rt.document_tile_maybe_active("
                f"_flash_mDoc, {bh_expr}, _flash_doc_heads_per_batch, "
                f"{m_tile_expr}, {kv_tile_expr})"
            )
    if not conditions:
        return None
    return " & ".join(f"({condition})" for condition in conditions)


def _flash_contiguous_kv_tile_range(
    score_plan: AttentionScorePlan,
    num_kv: int,
) -> tuple[str, str, str] | None:
    """Return inclusive first/last KV tile expressions when the mask is a range."""
    mask_modifiers = tuple(
        modifier
        for modifier in score_plan.modifiers
        if modifier.kind
        in {
            CAUSAL_MASK_KIND,
            SLIDING_WINDOW_MASK_KIND,
            PREFIX_LM_MASK_KIND,
            DOCUMENT_MASK_KIND,
        }
    )
    if len(mask_modifiers) != 1:
        return None
    modifier = mask_modifiers[0]
    if modifier.kind == CAUSAL_MASK_KIND:
        return "", "cutlass.Int32(0)", "flash_m_tile"
    if modifier.kind == SLIDING_WINDOW_MASK_KIND:
        assert modifier.window_size is not None
        back_tiles = (modifier.window_size + 127) // 128
        return (
            (
                "        flash_first_kv = "
                f"cutlass.max(cutlass.Int32(0), flash_m_tile - cutlass.Int32({back_tiles}))\n"
                "        flash_last_kv = flash_m_tile\n"
            ),
            "flash_first_kv",
            "flash_last_kv",
        )
    if modifier.kind == PREFIX_LM_MASK_KIND:
        assert modifier.prefix_length is not None
        prefix_tiles = (modifier.prefix_length + 127) // 128
        if prefix_tiles <= 1:
            return "", "cutlass.Int32(0)", "flash_m_tile"
        return (
            "",
            "cutlass.Int32(0)",
            (
                "cutlass.min("
                f"cutlass.Int32({num_kv - 1}), "
                f"cutlass.max(flash_m_tile, cutlass.Int32({prefix_tiles - 1})))"
            ),
        )
    if modifier.kind == DOCUMENT_MASK_KIND:
        prelude = (
            "        flash_doc_bh = flash_bh // _flash_doc_heads_per_batch\n"
            "        flash_doc_q_bits = _helion_flash_rt._document_tile_bits_warp(\n"
            "            _flash_mDoc, flash_doc_bh, flash_m_tile * cutlass.Int32(128))\n"
            f"        flash_doc_first_kv = cutlass.Int32({num_kv})\n"
            "        flash_doc_last_kv = flash_m_tile\n"
            f"        for flash_doc_n in cutlass.range({num_kv}, unroll=1):\n"
            "            flash_doc_k_bits = _helion_flash_rt._document_tile_bits_warp(\n"
            "                _flash_mDoc, flash_doc_bh, flash_doc_n * cutlass.Int32(128))\n"
            "            flash_doc_active = (flash_doc_n <= flash_m_tile) & ((flash_doc_q_bits & flash_doc_k_bits) != 0)\n"
            "            flash_doc_first_kv = cutlass.Int32(cutlass.select_(\n"
            f"                flash_doc_active & (flash_doc_first_kv == cutlass.Int32({num_kv})),\n"
            "                flash_doc_n,\n"
            "                flash_doc_first_kv,\n"
            "            ))\n"
            "            flash_doc_last_kv = cutlass.Int32(cutlass.select_(\n"
            "                flash_doc_active,\n"
            "                flash_doc_n,\n"
            "                flash_doc_last_kv,\n"
            "            ))\n"
            "        flash_doc_first_kv = cutlass.Int32(cutlass.select_(\n"
            f"            flash_doc_first_kv == cutlass.Int32({num_kv}),\n"
            "            flash_m_tile,\n"
            "            flash_doc_first_kv,\n"
            "        ))\n"
        )
        return prelude, "flash_doc_first_kv", "flash_doc_last_kv"
    return None


def _flash_fa4_runtime_disc_score_plan_supported(
    score_plan: AttentionScorePlan,
) -> bool:
    """Whether the hand-written FA4 disc runtime helpers cover this transform."""
    return all(modifier.kind == CAUSAL_MASK_KIND for modifier in score_plan.modifiers)


def _flash_kv_iteration(
    hd: int,
    kv_stage: int,
    *,
    prefetch: bool,
    io_dtype: str,
    score_plan: AttentionScorePlan,
    rescale_threshold: float,
    packed_reduce: bool,
    exp2_impl: str,
    e2e_freq: int,
    e2e_res: int,
) -> str:
    """One KV-loop iteration body (QK MMA -> softmax -> O rescale -> PV MMA).

    ``prefetch=True`` additionally issues the K/V TMA load for the tile
    ``kv_stage`` ahead (runtime index ``flash_kv + kv_stage``, always in range in
    the prefetch phase). ``prefetch=False`` (drain phase) issues nothing. The
    flag is resolved at codegen time so each emitted phase loop has a statically
    uniform producer-advance count (see ``emit_flash_device_body``).
    """
    k_prefetch = (
        """
        flash_k_next = flash_kv + cutlass.Int32({kv_stage})
        flash_k_empty = flash_k_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, flash_k_next], tKsK[None, flash_k_empty.index],
                  tma_bar_ptr=flash_k_empty.barrier)"""
        if prefetch
        else ""
    )
    v_prefetch = (
        """
        flash_v_next = flash_kv + cutlass.Int32({kv_stage})
        flash_v_empty = flash_v_prod.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, flash_v_next], tVsV[None, flash_v_empty.index],
                  tma_bar_ptr=flash_v_empty.barrier)"""
        if prefetch
        else ""
    )
    score_transform = _flash_score_transform_block(
        score_plan,
        indent="    ",
        score_tensor="tLDrS",
        coord_tensor="tLDcS",
        bh_expr="flash_bh",
        m_tile_expr="flash_m_tile",
        kv_tile_expr="flash_kv",
        causal_condition="flash_kv >= flash_m_tile",
    )
    if rescale_threshold > 0.0:
        alpha_pre = f"""    flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
    flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
    if (flash_kv != 0) & (flash_acc_log >= -{rescale_threshold}):
        flash_row_max = flash_old_row_max
        flash_row_max_safe = flash_old_row_max
        flash_alpha = cutlass.Float32(1.0)
    flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"""
        alpha_post = ""
        rescale_block = f"""    if flash_kv != 0:
        flash_o_full = flash_mma_o_cons.wait_and_advance()
        if cute.arch.vote_ballot_sync(flash_alpha < 1.0) != 0:
            _helion_flash_rt.rescale_o_tmem(tOtO, flash_alpha, tidx, {hd})
            cute.arch.fence_view_async_tmem_store()
        flash_o_full.release()"""
    else:
        alpha_pre = (
            "    flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"
        )
        alpha_post = """    flash_alpha = cute.math.exp2(
        _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""
        rescale_block = f"""    if flash_kv != 0:
        flash_o_full = flash_mma_o_cons.wait_and_advance()
        _helion_flash_rt.rescale_o_tmem(tOtO, flash_alpha, tidx, {hd})
        cute.arch.fence_view_async_tmem_store()
        flash_o_full.release()"""
    if packed_reduce:
        row_max_reduce = (
            "flash_row_max = _helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
        )
        p_sum_reduce = "flash_p_sum = _helion_flash_rt.fadd_reduce_packed(tLDrS)"
    else:
        row_max_reduce = (
            "flash_row_max = tLDrS.load().reduce("
            "cute.ReductionOp.MAX, flash_row_max, 0)"
        )
        p_sum_reduce = "flash_p_sum = tLDrS.load().reduce(cute.ReductionOp.ADD, 0.0, 0)"
    if exp2_impl == "split":
        exp2_block = f"""    _helion_flash_rt.exp2_split_inplace(
        tLDrS, _flash_scale_log2, flash_minus_max_scale, {e2e_freq}, {e2e_res})"""
    else:
        exp2_block = """    flash_n = cute.size(tLDrS)
    for flash_j in cutlass.range_constexpr(flash_n):
        tLDrS[flash_j] = cute.math.exp2(
            tLDrS[flash_j] * _flash_scale_log2 + flash_minus_max_scale, fastmath=True)"""
    body = f"""
    if warp_idx == 0:
        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS, tSrQ[None, None, flash_kp, flash_q_full.index],
                      tSrK[None, None, flash_kp, flash_k_full.index], tStS)
        flash_s_handle.commit()
        flash_k_full.release(){k_prefetch}

    flash_s_full = flash_mma_s_cons.wait_and_advance()
    tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
    cute.copy(flash_tiled_ld, tLDtS, tLDrS)
    cute.arch.fence_view_async_tmem_load(){score_transform}

    flash_old_row_max = flash_row_max
    {row_max_reduce}
    flash_row_max_safe = flash_row_max
    if flash_row_max == -cutlass.Float32.inf:
        flash_row_max_safe = cutlass.Float32(0.0)
{alpha_pre}

{exp2_block}
    {p_sum_reduce}

    tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
    tSTrS_e = cute.make_tensor(
        cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
    tSTrS_e.store(tLDrS.load().to({io_dtype}))

{alpha_post}
    flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum

    cute.copy(flash_tiled_st, tSTrS, tSTtS)
    cute.arch.fence_view_async_tmem_store()

{rescale_block}
    cute.arch.barrier()
    flash_s_full.release()

    if warp_idx == 0:
        flash_v_full = flash_v_cons.wait_and_advance()
        flash_o_handle = flash_mma_o_prod.acquire_and_advance()
        flash_first_acc = flash_kv != 0
        flash_nk2 = cute.size(tOrP, mode=[2])
        for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
            flash_acc_flag = flash_first_acc | (flash_kp != 0)
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_acc_flag)
            cute.gemm(_flash_pv_mma, tOtO, tOrP[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO)
        flash_o_handle.commit()
        flash_v_full.release(){v_prefetch}
    cute.arch.barrier()"""
    return body.format(hd=hd, kv_stage=kv_stage)


def _flash_kv_loop_body(
    *,
    hd: int,
    kv_stage: int,
    prefetch_iters: int,
    drain_iters: int,
    io_dtype: str,
    score_plan: AttentionScorePlan,
    rescale_threshold: float,
    packed_reduce: bool,
    exp2_impl: str,
    e2e_freq: int,
    e2e_res: int,
) -> str:
    """Emit the PREFETCH-phase + DRAIN-phase KV loops as one indented block.

    Both phases share the same per-iteration body (``_flash_kv_iteration``); the
    prefetch phase additionally issues the look-ahead K/V loads. Each loop has a
    compile-time-constant trip count so the pipeline producer/consumer advance
    counts stay statically balanced.
    """
    prefetch_iter = _flash_kv_iteration(
        hd,
        kv_stage,
        prefetch=True,
        io_dtype=io_dtype,
        score_plan=score_plan,
        rescale_threshold=rescale_threshold,
        packed_reduce=packed_reduce,
        exp2_impl=exp2_impl,
        e2e_freq=e2e_freq,
        e2e_res=e2e_res,
    )
    drain_iter = _flash_kv_iteration(
        hd,
        kv_stage,
        prefetch=False,
        io_dtype=io_dtype,
        score_plan=score_plan,
        rescale_threshold=rescale_threshold,
        packed_reduce=packed_reduce,
        exp2_impl=exp2_impl,
        e2e_freq=e2e_freq,
        e2e_res=e2e_res,
    )
    pieces: list[str] = []
    if prefetch_iters > 0:
        # flash_kv is the loop var 0..prefetch_iters-1 directly.
        pieces.append(
            f"for flash_kv in cutlass.range({prefetch_iters}, unroll=1):{prefetch_iter}"
        )
    if drain_iters > 0:
        # The drain loop continues the tile index from prefetch_iters so the
        # ``flash_kv != 0`` first-tile test stays correct. The body lines are
        # already 4-space indented (loop-body level); prepend the index assign.
        assign = f"\n    flash_kv = flash_kv_d + cutlass.Int32({prefetch_iters})"
        pieces.append(
            f"for flash_kv_d in cutlass.range({drain_iters}, unroll=1):"
            + assign
            + drain_iter
        )
    return "\n".join(pieces)


def _flash_lse_store(
    row_expr: str,
    bh_expr: str,
    indent: str,
    *,
    lse_scale: float,
) -> str:
    value = "flash_row_max * _flash_scale_log2 + cute.math.log2(flash_row_sum)"
    if not math.isclose(lse_scale, 1.0, rel_tol=1e-6, abs_tol=1e-7):
        value = f"({value}) * cutlass.Float32({lse_scale!r})"
    return f"\n{indent}_flash_mLSE[{row_expr}, {bh_expr}] = {value}"


def _flash_ws_guard(src: str, condition: str, indent: str = "            ") -> str:
    src = textwrap.dedent(src).strip("\n")
    return f"""
{indent}if {condition}:
{textwrap.indent(src, indent + "    ")}"""


def _flash_guard(src: str, condition: str | None, indent: str = "            ") -> str:
    if condition is None:
        return src
    return _flash_ws_guard(src, condition, indent=indent)


def _flash_ws_qk_ahead(
    kv_stage: int,
    *,
    kpf: bool,
    qk_condition: str | None,
    kpf_condition: str | None,
) -> str:
    """One ahead-QK(kv+1) into S[(kv+1)%2] + (optional) K look-ahead. The S-buffer
    parity is a runtime index (flash_qk_idx) toggled per ahead-QK."""
    k_pf = (
        f"""
            flash_k_empty = flash_k_prod.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, flash_kv + cutlass.Int32({kv_stage + 1})],
                      tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)"""
        if kpf
        else ""
    )
    if k_pf:
        k_pf = _flash_guard(k_pf, kpf_condition)
    qk_body = """
            flash_k_full = flash_k_cons.wait_and_advance()
            flash_s_handle = flash_mma_s_prod.acquire_and_advance()
            if (flash_qk_idx % 2) == 0:
                for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                    _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                    cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q_full.index],
                              tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
            else:
                for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                    _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                    cute.gemm(_flash_qk_mma, tStS1, tSrQ[None, None, flash_kp, flash_q_full.index],
                              tSrK[None, None, flash_kp, flash_k_full.index], tStS1)
            flash_s_handle.commit()
            flash_k_full.release()
            flash_qk_idx = (flash_qk_idx + 1) % 2"""
    return _flash_guard(qk_body, qk_condition) + k_pf


def _flash_ws_pv(
    kv_stage: int,
    *,
    vpf: bool,
    pv_condition: str | None,
    vpf_condition: str | None,
) -> str:
    """One PV(kv) reading P[kv%2] + (optional) V look-ahead (issued AFTER PV frees
    V -- Bug L). P-fragment parity is the runtime flash_p_idx."""
    v_pf = (
        f"""
            flash_v_empty = flash_v_prod.acquire_and_advance()
            cute.copy(_flash_tma_v, tVgV[None, flash_kv + cutlass.Int32({kv_stage})],
                      tVsV[None, flash_v_empty.index], tma_bar_ptr=flash_v_empty.barrier)"""
        if vpf
        else ""
    )
    if v_pf:
        v_pf = _flash_guard(v_pf, vpf_condition)
    pv_body = """
            flash_p_full = flash_p_ready_cons.wait_and_advance()
            flash_v_full = flash_v_cons.wait_and_advance()
            flash_o_handle = flash_mma_o_prod.acquire_and_advance()
            flash_first_acc = flash_o_started
            if (flash_p_idx % 2) == 0:
                for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
                    _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_first_acc | (flash_kp != 0))
                    cute.gemm(_flash_pv_mma, tOtO, tOrP0[None, None, flash_kp, 0],
                              tOrV[None, None, flash_kp, flash_v_full.index], tOtO)
            else:
                for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
                    _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_first_acc | (flash_kp != 0))
                    cute.gemm(_flash_pv_mma, tOtO, tOrP1[None, None, flash_kp, 0],
                              tOrV[None, None, flash_kp, flash_v_full.index], tOtO)
            flash_o_handle.commit()
            flash_v_full.release()
            flash_p_full.release()
            flash_p_idx = (flash_p_idx + 1) % 2
            flash_o_started = cutlass.Boolean(True)"""
    return _flash_guard(pv_body, pv_condition) + v_pf


def _flash_persistent_tile_prelude(indent: str) -> str:
    """Per-tile decode + gmem slices, emitted at the top of each role's strided
    persistent loop (replaces the once-in-setup decode of the flat path). The
    mapping (`% _flash_num_bh`, `// _flash_num_bh`) is byte-identical to the flat
    path's setup decode so the gmem-view math is unchanged."""
    body = """flash_bh = flash_tile_id % _flash_num_bh
flash_m_tile = flash_tile_id // _flash_num_bh
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]
tVgV = tVgV_dkl[None, 0, None, flash_bh]"""
    return textwrap.indent(body, indent)


def _flash_persistent_wrap(
    role_guard: str, inner: str, persistent: bool, head: str = ""
) -> str:
    """Wrap a role body (the dedented ``if warp_idx ...:`` block's INNER source,
    8-space indented) for the persistent scheduler.

    Non-persistent: emit ``\\n{role_guard}\\n{head}\\n{inner}`` (head omitted when
    empty -- byte-identical to the historical 2-arg call).
    Persistent: open the role guard, optionally emit ``head`` (a 4-space line that
    runs ONCE per thread -- e.g. a per-role setmaxnreg, kept OUT of the per-tile
    while loop), init a flat tile id from block_idx, and run a strided ``while
    flash_tile_id < _flash_total_tiles`` loop. Each iteration first re-decodes (bh,
    m_tile) + re-slices the per-tile gmem views, then runs the per-tile body (the
    original inner, re-indented 4 spaces deeper), then a CTA barrier (Risk 3: stop
    the next tile's producer Q/K load from overwriting sQ/sK / TMEM-O the current
    tile's consumer epilogue still reads), then strides the tile id by grid_dim.
    TMEM/SMEM/mbarriers/pipelines are created ONCE in setup (outside this loop) and
    reused; the mbarrier phase carries over across tiles (Risk 1). Dense tiles
    advance a compile-time-constant count. Range-pruned sparse tiles compute one
    identical per-tile KV interval in producer and consumer before advancing the
    S/P/O barriers, so phases still remain paired across the carried loop.

    ``head`` exists so each role's setmaxnreg can sit at the HEAD of its OWN guard
    (the fa4 pattern) rather than in a unified ``if/elif`` dispatch over warp_idx
    -- the latter trips ptxas C7600 (register-allocation failure)."""
    if not persistent:
        # Flat: ``inner`` is 8-space indented (it lived directly under the guard).
        # Re-indent ``head`` (written at 4-space, the persistent guard-body level)
        # to 8-space so it matches ``inner`` -- a mixed 4/8 indent under the guard
        # is an IndentationError.
        flat_head = (
            textwrap.indent(textwrap.dedent(head), "        ") + "\n" if head else ""
        )
        return f"\n{role_guard}\n{flat_head}{inner}"
    head_block = f"{head}\n" if head else ""
    # ``inner`` is 8-space indented (it lived directly under ``if warp_idx ...:``).
    # Under persistence the role guard's body is ``flash_tile_id = ...`` +
    # ``while ...:`` at 4 spaces, so the while body sits at 8 spaces -- exactly
    # ``inner``'s existing indent. The per-tile prelude is emitted at the same
    # 8-space level so it precedes ``inner`` inside the loop.
    prelude = _flash_persistent_tile_prelude("        ")
    return f"""
{role_guard}
{head_block}    flash_tile_id = cutlass.Int32(cute.arch.block_idx()[0])
    while flash_tile_id < _flash_total_tiles:
{prelude}
{inner}
        cute.arch.barrier()
        flash_tile_id = flash_tile_id + cutlass.Int32(cute.arch.grid_dim()[0])"""


def _flash_ws_producer_body(
    num_kv: int,
    kv_stage: int,
    hd: int,
    *,
    score_plan: AttentionScorePlan,
) -> str:
    """Producer (warp 0) body for the Stage-4 double-buffered-S overlap.

    PHASE-SPLIT runtime loops (NOT a full unroll -- it makes the CuTe MLIR compile
    take minutes by bloating the IR, see STAGE3). Dense attention keeps each phase
    body fixed so producer/consumer advance counts are statically uniform. Single
    range masks walk the active contiguous KV span directly; other dynamic masks
    wrap the fixed phase bodies in guards to skip inactive KV tiles.

    Warp 0 stays ONE KV tile ahead on QK: iteration kv issues QK(kv+1) into
    S[(kv+1)%2] BEFORE PV(kv), overlapping the consumer's softmax(kv). K is freed
    early (by QK) -> its look-ahead is right after the QK; V is freed late (by PV)
    -> its look-ahead is after PV (Bug L). The phases partition iters [0, num_kv)
    by which look-aheads are still in range:
      ahead-QK(kv+1) fires for kv in [0, num_kv-1);
      K look-ahead (tile kv+1+kv_stage) for kv in [0, num_kv-1-kv_stage);
      V look-ahead (tile kv+kv_stage)   for kv in [0, num_kv-kv_stage).
    """
    contiguous_range = _flash_contiguous_kv_tile_range(score_plan, num_kv)
    if contiguous_range is None:
        range_prelude = ""
        first_kv_expr = ""
        last_kv_expr = ""
    else:
        range_prelude, first_kv_expr, last_kv_expr = contiguous_range
        active_count_expr = f"{last_kv_expr} - {first_kv_expr} + cutlass.Int32(1)"
        prologue_pieces: list[str] = []
        for pf in range(min(kv_stage, num_kv)):
            piece = f"""
        if cutlass.Int32({pf}) < flash_active_count:
            flash_k_empty = flash_k_prod.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, {first_kv_expr} + cutlass.Int32({pf})],
                      tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)
            flash_v_empty = flash_v_prod.acquire_and_advance()
            cute.copy(_flash_tma_v, tVgV[None, {first_kv_expr} + cutlass.Int32({pf})],
                      tVsV[None, flash_v_empty.index], tma_bar_ptr=flash_v_empty.barrier)"""
            prologue_pieces.append(piece)
        prologue_pf = "".join(prologue_pieces)
        qk0_pf = ""
        if kv_stage < num_kv:
            qk0_pf = f"""
        if cutlass.Int32({kv_stage}) < flash_active_count:
            flash_k_empty = flash_k_prod.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, {first_kv_expr} + cutlass.Int32({kv_stage})],
                      tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)"""
        qk_next = _flash_ws_qk_ahead(
            kv_stage,
            kpf=True,
            qk_condition=("flash_active_kv + cutlass.Int32(1) < flash_active_count"),
            kpf_condition=(
                f"flash_active_kv + cutlass.Int32({kv_stage + 1}) < flash_active_count"
            ),
        )
        pv_current = _flash_ws_pv(
            kv_stage,
            vpf=True,
            pv_condition=None,
            vpf_condition=(
                f"flash_active_kv + cutlass.Int32({kv_stage}) < flash_active_count"
            ),
        )
        loop_body = f"""
        for flash_active_kv in cutlass.range(flash_active_count, unroll=1):
            flash_kv = {first_kv_expr} + flash_active_kv{qk_next}{pv_current}"""
        return f"""{range_prelude}        flash_active_count = {active_count_expr}
        flash_q_empty = flash_q_prod.acquire_and_advance()
        cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile], tQsQ[None, flash_q_empty.index],
                  tma_bar_ptr=flash_q_empty.barrier)
        flash_q_full = flash_q_cons.wait_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        flash_p_idx = cutlass.Int32(0)   # P-fragment parity (kv % 2)
        flash_qk_idx = cutlass.Int32(0)  # S target parity among active QK tiles
        flash_o_started = cutlass.Boolean(False){prologue_pf}
        # Prologue QK(first active KV) -> S0.
        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q_full.index],
                      tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
        flash_s_handle.commit()
        flash_k_full.release()
        flash_qk_idx = (flash_qk_idx + 1) % 2{qk0_pf}
        {loop_body}
        flash_q_full.release()"""
    # Phase boundaries (exclusive upper iter index for each look-ahead).
    b_kpf = max(num_kv - 1 - kv_stage, 0)
    b_vpf = max(num_kv - kv_stage, 0)
    b_qk = num_kv - 1
    # Sorted distinct boundaries -> contiguous phases over [0, num_kv).
    bounds = sorted({0, b_kpf, b_vpf, b_qk, num_kv})
    phases = []

    def active(kv_tile_expr: str) -> str | None:
        if contiguous_range is not None:
            return (
                f"({kv_tile_expr} >= {first_kv_expr})"
                f" & ({kv_tile_expr} <= {last_kv_expr})"
            )
        return _flash_kv_tile_active_condition(
            score_plan,
            bh_expr="flash_bh",
            m_tile_expr="flash_m_tile",
            kv_tile_expr=kv_tile_expr,
        )

    for lo, hi in itertools.pairwise(bounds):
        if hi <= lo:
            continue
        has_qk = lo < b_qk
        has_kpf = lo < b_kpf
        has_vpf = lo < b_vpf
        qk_src = (
            _flash_ws_qk_ahead(
                kv_stage,
                kpf=has_kpf,
                qk_condition=active("flash_kv + cutlass.Int32(1)"),
                kpf_condition=active(f"flash_kv + cutlass.Int32({kv_stage + 1})"),
            )
            if has_qk
            else ""
        )
        pv_src = _flash_ws_pv(
            kv_stage,
            vpf=has_vpf,
            pv_condition=active("flash_kv"),
            vpf_condition=active(f"flash_kv + cutlass.Int32({kv_stage})"),
        )
        phases.append(
            f"""
        for flash_phi in cutlass.range({hi - lo}, unroll=1):
            flash_kv = flash_phi + cutlass.Int32({lo}){qk_src}{pv_src}"""
        )
    loop_body = "".join(phases)

    prologue_pieces: list[str] = []
    for pf in range(min(kv_stage, num_kv)):
        piece = f"""
        flash_k_empty = flash_k_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, {pf}], tKsK[None, flash_k_empty.index],
                  tma_bar_ptr=flash_k_empty.barrier)
        flash_v_empty = flash_v_prod.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, {pf}], tVsV[None, flash_v_empty.index],
                  tma_bar_ptr=flash_v_empty.barrier)"""
        piece = _flash_guard(
            piece,
            active(f"cutlass.Int32({pf})"),
            indent="        ",
        )
        prologue_pieces.append(piece)
    prologue_pf = "".join(prologue_pieces)
    qk0_pf = ""
    if kv_stage < num_kv:
        qk0_pf = f"""
        flash_k_empty = flash_k_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, {kv_stage}], tKsK[None, flash_k_empty.index],
                  tma_bar_ptr=flash_k_empty.barrier)"""
        qk0_pf = _flash_guard(
            qk0_pf,
            active(f"cutlass.Int32({kv_stage})"),
            indent="        ",
        )
    qk0_body = """
        # Prologue QK(0) -> S0.
        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q_full.index],
                      tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
        flash_s_handle.commit()
        flash_k_full.release()
        flash_qk_idx = (flash_qk_idx + 1) % 2"""
    qk0_body = _flash_guard(
        qk0_body,
        active("cutlass.Int32(0)"),
        indent="        ",
    )
    # Inner producer body (8-space indented, i.e. directly under ``if warp_idx ==
    # 0:``). The caller (`emit_flash_ws_device_body`) wraps it in the role guard
    # and -- when persistent -- a strided per-tile while loop.
    return f"""{range_prelude}        flash_q_empty = flash_q_prod.acquire_and_advance()
        cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile], tQsQ[None, flash_q_empty.index],
                  tma_bar_ptr=flash_q_empty.barrier)
        flash_q_full = flash_q_cons.wait_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        flash_p_idx = cutlass.Int32(0)   # P-fragment parity (kv % 2)
        flash_qk_idx = cutlass.Int32(0)  # S target parity among active QK tiles
        flash_o_started = cutlass.Boolean(False){prologue_pf}{qk0_body}{qk0_pf}
        {loop_body}
        flash_q_full.release()"""


def _flash_ws_consumer_body(
    hd: int,
    num_kv: int,
    cfg: FlashAttentionConfig,
    *,
    io_dtype: str,
    score_plan: AttentionScorePlan,
) -> str:
    """Consumer warpgroup (warps 4-7) body for the Stage-4 overlap: per-tile
    online softmax on S[k%2], P-over-S write, O alpha-rescale, p_ready signal;
    then the O/l divide + cast + store epilogue. Runtime loop over num_kv (the
    stage index toggles each tile; no prefetch -> no dynamic pipeline branch).

    Returns the inner body (8-space indented, i.e. directly under ``if warp_idx
    >= 4:``); the caller wraps it in the role guard and -- when persistent -- a
    strided per-tile while loop. The per-tile online-softmax state (row_max,
    row_sum, s_idx) re-inits at the TOP of this body, so each tile starts clean
    (Risk 2: row_max/row_sum reset per tile; S/O TMEM accumulators are overwritten
    by the first-KV ``ACCUMULATE=False`` MMA, needing no explicit clear).

    ``cfg.exp2_impl == "split"`` (the default) replaces the per-element exp2
    loop with a call to ``_helion_flash_rt.exp2_split_inplace`` which routes a
    fraction (~1-in-e2e_freq with e2e_res resolution) of exp2 ops through the FA4
    degree-3 minimax polynomial on the FMA/ALU pipe, shortening the
    t2r->exp2->r2t dependency chain (measured +2.9pp hd64 of SDPA). Setting
    ``HELION_CUTE_FLASH_EXP2_IMPL=xu`` selects the original hardware-XU loop.
    """
    if cfg.exp2_impl == "split":
        exp2_block = f"""            _helion_flash_rt.exp2_split_inplace(
                tLDrS, _flash_scale_log2, flash_minus_max_scale, {cfg.e2e_freq}, {cfg.e2e_res})"""
    else:
        exp2_block = """            flash_n = cute.size(tLDrS)
            for flash_j in cutlass.range_constexpr(flash_n):
                tLDrS[flash_j] = cute.math.exp2(
                    tLDrS[flash_j] * _flash_scale_log2 + flash_minus_max_scale, fastmath=True)"""
    # FA4-faithful multi-accumulator packed reductions. The naive
    # ``tLDrS.load().reduce(MAX/ADD)`` lowers to a SINGLE-accumulator serial
    # FADD/FMNMX chain over the 128-wide row; the packed helpers fold with 4
    # independent packed-f32x2 accumulators (FA4's fadd_reduce/fmax_reduce
    # arch>=100). This is default-on only for the characterized long/causal FA4
    # seeds and KV-pruned sparse ws_overlap masks; other shapes keep the serial
    # reduction unless ``HELION_CUTE_FLASH_PACKED_REDUCE`` or
    # ``cute_flash_packed_reduce`` opts in.
    if cfg.packed_reduce:
        row_max_reduce = (
            "flash_row_max = _helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
        )
        p_sum_reduce = "flash_p_sum = _helion_flash_rt.fadd_reduce_packed(tLDrS)"
    else:
        row_max_reduce = "flash_row_max = tLDrS.load().reduce(cute.ReductionOp.MAX, flash_row_max, 0)"
        p_sum_reduce = "flash_p_sum = tLDrS.load().reduce(cute.ReductionOp.ADD, 0.0, 0)"
    score_transform = _flash_score_transform_block(
        score_plan,
        indent="            ",
        score_tensor="tLDrS",
        coord_tensor="tLDcS",
        bh_expr="flash_bh",
        m_tile_expr="flash_m_tile",
        kv_tile_expr="flash_kv",
        causal_condition="flash_kv == flash_m_tile",
    )
    active_condition = _flash_kv_tile_active_condition(
        score_plan,
        bh_expr="flash_bh",
        m_tile_expr="flash_m_tile",
        kv_tile_expr="flash_kv",
    )
    contiguous_range = _flash_contiguous_kv_tile_range(score_plan, num_kv)
    first_kv_expr = ""
    last_kv_expr = ""
    if contiguous_range is not None:
        range_prelude, first_kv_expr, last_kv_expr = contiguous_range
        active_condition = (
            f"(flash_kv >= {first_kv_expr}) & (flash_kv <= {last_kv_expr})"
        )
    else:
        range_prelude = ""
    has_previous_o = "flash_o_started"
    if cfg.rescale_threshold > 0.0:
        alpha_pre = f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if ({has_previous_o}) & (flash_acc_log >= -{cfg.rescale_threshold}):
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"""
        alpha_post = ""
        rescale_block = f"""            if {has_previous_o}:
                flash_o_full = flash_mma_o_cons.wait_and_advance()
                if cute.arch.vote_ballot_sync(flash_alpha < 1.0) != 0:
                    _helion_flash_rt.rescale_o_tmem(
                        tOtO, flash_alpha, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                    cute.arch.fence_view_async_tmem_store()
                flash_o_full.release()"""
    else:
        alpha_pre = (
            "            flash_minus_max_scale ="
            " (0.0 - flash_row_max_safe) * _flash_scale_log2"
        )
        alpha_post = """            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""
        rescale_block = f"""            if {has_previous_o}:
                flash_o_full = flash_mma_o_cons.wait_and_advance()
                _helion_flash_rt.rescale_o_tmem(
                    tOtO, flash_alpha, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                cute.arch.fence_view_async_tmem_store()
                flash_o_full.release()"""
    consumer_iter = f"""
            flash_s_full = flash_mma_s_cons.wait_and_advance()
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            if (flash_s_idx % 2) == 0:
                cute.copy(flash_tiled_ld0, tLDtS0, tLDrS)
            else:
                cute.copy(flash_tiled_ld1, tLDtS1, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}

            flash_old_row_max = flash_row_max
            {row_max_reduce}
            flash_row_max_safe = flash_row_max
            if flash_row_max == -cutlass.Float32.inf:
                flash_row_max_safe = cutlass.Float32(0.0)
{alpha_pre}

{exp2_block}
            {p_sum_reduce}

            tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
            tSTrS_e = cute.make_tensor(
                cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrS_e.store(tLDrS.load().to({io_dtype}))

{alpha_post}
            flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum

            if (flash_s_idx % 2) == 0:
                cute.copy(flash_tiled_st0, tSTrS, tSTtS0)
            else:
                cute.copy(flash_tiled_st1, tSTrS, tSTtS1)
            cute.arch.fence_view_async_tmem_store()

{rescale_block}
            flash_p_handle = flash_p_ready_prod.acquire_and_advance()
            flash_p_handle.commit()
            flash_s_full.release()
            flash_o_started = cutlass.Boolean(True)
            flash_s_idx = (flash_s_idx + 1) % 2"""
    if contiguous_range is not None:
        loop_header = (
            f"        for flash_active_kv in cutlass.range({last_kv_expr}"
            f" - {first_kv_expr} + cutlass.Int32(1), unroll=1):\n"
            f"            flash_kv = {first_kv_expr} + flash_active_kv"
        )
    else:
        consumer_iter = _flash_guard(consumer_iter, active_condition)
        loop_header = (
            "        for flash_kv in cutlass.range(_flash_num_kv_tiles, unroll=1):"
        )

    return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
        flash_s_idx = cutlass.Int32(0)
        flash_o_started = cutlass.Boolean(False)
{range_prelude}{loop_header}{consumer_iter}

        flash_o_full = flash_mma_o_cons.wait_and_advance()
        flash_inv_sum = cutlass.Float32(1.0) / flash_row_sum
        gO = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))
        tOgO_mma = flash_pvt.partition_C(gO)[None, None, None, flash_m_tile, 0, flash_bh]
        flash_epi_tiler = ((cute.size(tOtO, mode=[0, 0]), cute.size(tOtO, mode=[0, 1])),)
        tOtO_epi = cute.zipped_divide(tOtO, flash_epi_tiler)
        gO_epi = cute.zipped_divide(tOgO_mma, flash_epi_tiler)
        flash_o_ld_atom = cute.make_copy_atom(
            cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition.x64), cutlass.Float32)
        flash_tiled_o_ld = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom, tOtO_epi[None, 0])
        flash_thr_o_ld = flash_tiled_o_ld.get_slice(flash_local_tidx)
        tDtO = flash_thr_o_ld.partition_S(tOtO_epi)
        tDgO = flash_thr_o_ld.partition_D(gO_epi)
        flash_reg = cute.make_rmem_tensor(tDgO[None, None, 0].shape, cutlass.Float32)
        flash_rego = cute.make_rmem_tensor(tDgO[None, None, 0].shape, {io_dtype})
        for flash_i in cutlass.range(cute.size(tDtO, mode=[2])):
            cute.copy(flash_tiled_o_ld, tDtO[None, None, flash_i], flash_reg)
            flash_reg.store(flash_reg.load() * flash_inv_sum)
            flash_rego.store(flash_reg.load().to({io_dtype}))
            cute.autovec_copy(flash_rego, tDgO[None, None, flash_i])
        flash_o_full.release()"""


def emit_flash_ws_device_body(
    df: DeviceFunction,
    *,
    head_dim: int,
    num_kv: int,
    cfg: FlashAttentionConfig,
    has_lse: bool,
    io_dtype: str,
    score_plan: AttentionScorePlan,
) -> list[ast.stmt]:
    """Stage-4 device body: 2-warpgroup producer/consumer split with a
    double-buffered S TMEM accumulator (S0 @ col 0, S1 @ col 128, O @ col 256).

    Launch is ``block=(256, 1, 1)``: warp 0 = producer (Q/K/V TMA ring + both
    tcgen05 MMAs), warps 1-3 empty, warps 4-7 = consumer warpgroup (softmax +
    O rescale + epilogue). The producer runs QK(k+1) while the consumer runs
    softmax(k) -> the QK MMA overlaps the softmax (fills the `wait` bubble).
    """
    if score_plan.is_causal:
        assert not cfg.persistent
    kv_stage = cfg.kv_stage
    hd = head_dim
    persistent = cfg.persistent
    ws_lse_store = (
        _flash_lse_store(
            "flash_m_tile * 128 + flash_local_tidx",
            "flash_bh",
            "        ",
            lse_scale=score_plan.lse_scale,
        )
        if has_lse
        else ""
    )
    # Non-persistent (default): decode the one-tile-per-CTA mapping once in setup
    # (byte-identical to the historical body). Persistent: the decode + per-tile
    # gmem slices move INSIDE each role's strided while loop (see below); setup
    # only computes the tile-independent state.
    setup_pid = (
        ""
        if persistent
        else """
flash_pid = cutlass.Int32(cute.arch.block_idx()[0])
flash_bh = flash_pid % _flash_num_bh
flash_m_tile = flash_pid // _flash_num_bh"""
    )
    # The per-tile gmem slices depend on flash_bh; in the persistent path they
    # are re-sliced at the top of each role's per-tile loop instead.
    setup_gmem_slice = (
        ""
        if persistent
        else """
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]
tVgV = tVgV_dkl[None, 0, None, flash_bh]"""
    )
    setup = f"""
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()){setup_pid}
flash_local_tidx = tidx % 128

_flash_storage_cls = _helion_flash_rt.flash_shared_storage({hd}, {kv_stage}, 2, {io_dtype})
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)
sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner)

# TMEM allocator: warp 0 is the sole allocator; all 256 threads wait_for_alloc
# (NamedBarrier id=1, 256 threads) and the final sync(barrier_id=1) before free.
flash_tmem_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=1, num_threads=256)
flash_tmem = cutlass_utils_flash.TmemAllocator(
    storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_bar)
flash_tmem.allocate(512)
if warp_idx == 0:
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_q)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_k)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_v)

flash_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2]))
flash_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2]))
flash_v_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2]))
flash_q_prod, flash_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr()).make_participants()
flash_k_prod, flash_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr()).make_participants()
flash_v_prod, flash_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_v_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr()).make_participants()
# mma_s: QK UMMA (warp 0) -> softmax (consumer WG, 128 threads). 2 stages = the
# 2 S buffers, so QK(k+1) overlaps softmax(k).
flash_mma_s_prod, flash_mma_s_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_s_mbar_ptr.data_ptr()).make_participants()
flash_mma_o_prod, flash_mma_o_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_o_mbar_ptr.data_ptr()).make_participants()
# p_ready: consumer WG (128 threads) -> warp 0. PipelineAsync has NO elect-one,
# so all 32 threads of warp 0 arrive on release() -> the consumer group MUST be a
# FULL WARP (32), not 1 (Bug K: a 1-thread group is over-arrived -> illegal
# barrier arrive, launch fail 719).
flash_p_ready_prod, flash_p_ready_cons = cutlass_pipeline_flash.PipelineAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 32),
    barrier_storage=storage.p_ready_mbar_ptr.data_ptr()).make_participants()

flash_qkt = _flash_qk_mma.get_slice(0)
flash_pvt = _flash_pv_mma.get_slice(0)
tSrQ = flash_qkt.make_fragment_A(sQ)
tSrK = flash_qkt.make_fragment_B(sK)
tOrV = flash_pvt.make_fragment_B(sV)
flash_qk_acc_shape = flash_qkt.partition_shape_C((128, 128))
tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
flash_pv_acc_shape = flash_pvt.partition_shape_C((128, {hd}))
tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)

gQ = cute.flat_divide(_flash_mQt, cute.select((128, 128, {hd}), mode=[0, 2]))
gK = cute.flat_divide(_flash_mKt, cute.select((128, 128, {hd}), mode=[1, 2]))
gV = cute.flat_divide(_flash_mVt, cute.select((128, {hd}, 128), mode=[1, 2]))
tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tOgV = flash_pvt.partition_B(gV)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3)){setup_gmem_slice}

flash_tmem.wait_for_alloc()
flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
flash_s_layout = tStS.layout
# Double-buffered S: S0 @ col 0, S1 @ col 128, O @ col 256.
tStS0 = cute.make_tensor(flash_tmem_ptr, flash_s_layout)
tStS1 = cute.make_tensor(flash_tmem_ptr + 128, flash_s_layout)
tOtO = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)
# P operand fragments (OperandSource.TMEM). make_fragment_A is built ONCE at col
# 0; the per-stage fragment is the FRAGMENT iterator offset by
# (acc_width/io_width) * column (fp16 units). make_fragment_A does NOT honor a
# column offset baked into its input tensor's iterator -> offset the fragment.
tP = cute.make_tensor(flash_tmem_ptr, _flash_ptl.outer)
tOrP0 = flash_pvt.make_fragment_A(tP)
tOrP1 = cute.make_tensor(
    tOrP0.iterator + (cutlass.Float32.width // {io_dtype}.width) * 128, tOrP0.layout)

cS = cute.make_identity_tensor((128, 128))
tScS = flash_qkt.partition_C(cS)
# Per-stage t2r/r2t tiled copies built over the ALREADY-OFFSET S tensor (the
# lane->column map is baked at make_tmem_copy time; partition_S over a different
# column of a col-0-built copy targets the wrong columns).
flash_ld_atom = cute.make_copy_atom(
    cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
tLDtS0 = flash_thr_ld0.partition_S(tStS0)
tLDtS1 = flash_thr_ld1.partition_S(tStS1)
tLDcS = flash_thr_ld0.partition_D(tScS)

flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width
flash_P_layout = cute.composition(flash_s_layout, cute.make_layout((128, flash_tilePlikeFP32)))
tStS_P0 = cute.make_tensor(flash_tmem_ptr, flash_P_layout)
tStS_P1 = cute.make_tensor(flash_tmem_ptr + 128, flash_P_layout)
flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
flash_st_atom = cute.make_copy_atom(
    cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
flash_tiled_st0 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P0)
flash_tiled_st1 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P1)
flash_thr_st0 = flash_tiled_st0.get_slice(flash_local_tidx)
flash_thr_st1 = flash_tiled_st1.get_slice(flash_local_tidx)
tSTtS0 = flash_thr_st0.partition_D(tStS_P0)
tSTtS1 = flash_thr_st1.partition_D(tStS_P1)
tSTcS = flash_thr_st0.partition_S(tScS_P)
"""
    # No acc pipeline in the 2-WG path: the consumer's final epilogue read is
    # ordered by the last PV's mma_o, and the 256-thread sync(barrier_id=1) at
    # teardown guarantees all reads precede warp 0's TMEM free (matches the
    # validated standalone spike, which has no acc pipeline).
    setmaxreg = f"""
if warp_idx >= 4:
    cute.arch.setmaxregister_increase({cfg.num_regs_consumer})
else:
    cute.arch.setmaxregister_decrease({cfg.num_regs_producer})
"""
    # Producer/consumer emitters now return only their INNER body (8-space
    # indented, i.e. directly under the role guard). ``_flash_persistent_wrap``
    # adds the role guard and -- when persistent -- the strided per-tile while
    # loop. Both roles read flash_tile_id from the SAME block_idx/grid_dim, so
    # they walk identical lock-step tile sequences (the cross-role mma_s/mma_o/
    # p_ready pipelines stay matched).
    producer = _flash_persistent_wrap(
        "if warp_idx == 0:",
        _flash_ws_producer_body(
            num_kv,
            kv_stage,
            hd,
            score_plan=score_plan,
        ),
        persistent,
    )
    consumer = _flash_persistent_wrap(
        "if warp_idx >= 4:",
        _flash_ws_consumer_body(
            hd,
            num_kv,
            cfg,
            io_dtype=io_dtype,
            score_plan=score_plan,
        )
        + ws_lse_store,
        persistent,
    )
    # Teardown: warp 0 (TMEM allocator) relinquishes, all 256 threads converge on
    # a plain CTA barrier (matches the validated spike -- a pipeline.sync over
    # barrier_id=1 collides with the TMEM-allocator NamedBarrier id=1 and trips
    # synccheck "Missing init"), then warp 0 frees TMEM.
    teardown = """
if warp_idx == 0:
    flash_tmem.relinquish_alloc_permit()
cute.arch.barrier()
if warp_idx == 0:
    flash_tmem.free(flash_tmem_ptr)
"""
    src = setmaxreg + producer + "\n" + consumer + "\n" + teardown
    src = setup + "\n" + src
    return list(ast.parse(src).body)


def _flash_fa4_tile_prelude(
    indent: str,
    prelude: str = "full",
    *,
    use_2cta_instrs: bool = False,
    use_cga2_local_cta: bool = False,
    tensor_4d_heads: int = 0,
) -> str:
    """Per-work-item tile ids + gmem re-slice for the fa4 persistent scheduler.

    Emitted at the top of each role's strided ``while`` loop. Each work-item is
    a Q-tile PAIR. Match FA4's static persistent scheduler order: Q-block pair
    is the fastest dimension, with the collapsed batch/head index outside it.
    The persistent wrapper carries ``flash_m_pair`` / ``flash_bh`` across loop
    iterations, so this prelude derives only the current m tile ids and re-slices
    the gmem-source TMA views (the smem-dst partitions tQsQ/tKsK/tVsV are
    work-item-independent and stay in setup).

    ``prelude`` selects the ROLE-MINIMAL subset (Step 2 hoist): only the load warp
    actually consumes the per-work-item gmem TMA re-slices (tQgQ/tKgK/tVgV) and the
    m_tile ids; the correction warp uses only the (bh, m_pair, m_tile) decode (its
    epilogue re-derives its own gmem store views); the mma/softmax/empty/epi warps
    consume NOTHING from the prelude. Emitting only the consumed subset per role
    drops the dead bh-decode + TMA-partition IMAD/MOV recompute out of the
    register-pressure-critical softmax/mma loops (and the corr loop):

      * ``full``   -- load warp: bh decode + m_tiles + gmem TMA re-slices.
      * ``decode`` -- correction warp: bh + m_pair + m_tile ids only (no gmem).
      * ``none``   -- mma/softmax/empty/epi: nothing.
    """
    if prelude == "none":
        return ""
    if use_2cta_instrs:
        decode = """flash_q_mma_tile0 = flash_m_pair * 2
flash_q_mma_tile1 = flash_q_mma_tile0 + 1
flash_m_tile0 = flash_q_mma_tile0 * 2 + flash_mma_tile_coord_v
flash_m_tile1 = flash_q_mma_tile1 * 2 + flash_mma_tile_coord_v"""
    elif use_cga2_local_cta:
        decode = """flash_local_m_pair = flash_m_pair * 2 + flash_cga2_local_rank
flash_m_tile0 = flash_local_m_pair * 2
flash_m_tile1 = flash_m_tile0 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    else:
        decode = """flash_m_tile0 = flash_m_pair * 2
flash_m_tile1 = flash_m_tile0 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    if tensor_4d_heads > 0:
        decode += f"""
flash_head = flash_bh % {tensor_4d_heads}
flash_batch = flash_bh // {tensor_4d_heads}"""
    if prelude == "decode":
        return textwrap.indent(decode, indent)
    if tensor_4d_heads > 0:
        body = (
            decode
            + """
tQgQ = tQgQ_qdl[None, None, 0, flash_head, flash_batch]
tKgK = tKgK_kdl[None, None, 0, flash_head, flash_batch]
tVgV = tVgV_dkl[None, 0, None, flash_head, flash_batch]"""
        )
        return textwrap.indent(body, indent)
    body = (
        decode
        + """
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]
tVgV = tVgV_dkl[None, 0, None, flash_bh]"""
    )
    return textwrap.indent(body, indent)


def _flash_power2_decode_exprs(value_expr: str, divisor: int) -> tuple[str, str] | None:
    if divisor <= 1 or divisor & (divisor - 1):
        return None
    mask = divisor - 1
    shift = divisor.bit_length() - 1
    return f"({value_expr} & cutlass.Int32({mask}))", f"({value_expr} >> {shift})"


def _flash_fa4_load_prologue_for_order(
    first_load_order: int,
    load_q0: str,
    load_k0: str,
    load_q1: str,
    load_v0: str,
) -> str:
    if first_load_order == 1:
        return f"{load_k0}\n{load_v0}\n{load_q0}\n{load_q1}"
    if first_load_order == 2:
        return f"{load_q0}\n{load_q1}\n{load_k0}\n{load_v0}"
    if first_load_order == 3:
        return f"{load_k0}\n{load_q0}\n{load_v0}\n{load_q1}"
    if first_load_order == 4:
        return f"{load_k0}\n{load_q0}\n{load_q1}\n{load_v0}"
    return f"{load_q0}\n{load_k0}\n{load_q1}\n{load_v0}"


def _flash_fa4_wrap(
    role_guard: str,
    head: str,
    inner: str,
    persistent: bool,
    prelude: str = "full",
    tail: str = "",
    total_tiles: int | None = None,
    num_m_pairs: int | None = None,
    use_2cta_instrs: bool = False,
    use_cga2_local_cta: bool = False,
    use_clc_scheduler: bool = False,
    clc_heads_per_batch: int | None = None,
    tensor_4d_heads: int = 0,
    recompute_tile_coords: bool = False,
) -> str:
    """Wrap an fa4 role body for the (non-)persistent scheduler.

    ``role_guard`` is the ``if warp_idx ...:`` line. ``head`` is the role's
    setmaxnreg + cross-work-item phase declarations (4-space indented; lives
    DIRECTLY under the guard so the phases persist across work-items). ``inner`` is
    the per-work-item body written at 8-space base indent. ``prelude`` selects the
    role-minimal per-work-item decode subset (see ``_flash_fa4_tile_prelude``).
    ``tail`` is optional 4-space-indented role code emitted once after the
    per-work-item body/loop.

    Non-persistent: ``{guard}\\n{head}\\n{inner}`` (the head's phases init once, the
    inner runs once -- byte-identical to the Stage-1 flat body).

    Persistent: after ``head`` open a strided ``while flash_tile_id <
    _flash_total_tiles`` loop; roles that need tile coordinates carry
    ``(m_pair, bh)`` state and advance it by ``grid_dim`` each iteration, avoiding
    repeated per-work-item div/mod decode. The raw-mbarrier phases in ``head``
    carry across iterations (every per-tile acquire/arrive count is a
    compile-time constant, so the phases advance deterministically)."""
    if not persistent:
        # Flat: head + inner sit directly under the guard at 4-space. ``inner`` is
        # written at 8-space (the persistent while-loop body level), so dedent it 4.
        inner_flat = textwrap.indent(textwrap.dedent(inner), "    ") if inner else ""
        tail_block = f"\n{tail}" if tail else ""
        return f"\n{role_guard}\n{head}\n{inner_flat}{tail_block}"
    # The while loop sits at 4-space (under the guard), so its body is at 8-space.
    # ``inner`` is already written at 8-space (the loop-body level), so it slots in
    # directly; the prelude is emitted at the same 8-space level.
    # The flash detector specializes shapes, so FA4 callers pass literal tile counts
    # here to avoid dynamic scheduler div/rem in the generated PTX.
    total_tiles_expr = "_flash_total_tiles" if total_tiles is None else str(total_tiles)
    prelude_src = _flash_fa4_tile_prelude(
        "        ",
        prelude,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        tensor_4d_heads=tensor_4d_heads,
    )
    prelude_block = f"{prelude_src}\n" if prelude_src else ""
    if use_clc_scheduler:
        assert num_m_pairs is not None
        assert clc_heads_per_batch is not None and clc_heads_per_batch > 0
        clc_m_pair_expr = "cutlass.Int32(flash_clc_work.tile_idx[0])"
        advance = """        flash_clc_pipeline.consumer_wait(flash_clc_consumer_state)
        flash_clc_response_ptr = (
            flash_clc_response_base
            + flash_clc_consumer_state.index * cutlass.Int32(4))
        flash_clc_work = flash_clc_hw.work_tile_info_from_clc_response(
            flash_clc_response_ptr)
        flash_clc_pipeline.consumer_release(flash_clc_consumer_state)
        flash_clc_consumer_state.advance()"""
        if prelude == "none":
            return f"""
{role_guard}
{head}
    flash_clc_work = flash_clc_hw.initial_work_tile_info()
    while flash_clc_work.is_valid_tile:
{inner}
{advance}
{tail}"""
        return f"""
{role_guard}
{head}
    flash_clc_work = flash_clc_hw.initial_work_tile_info()
    while flash_clc_work.is_valid_tile:
        flash_m_pair = {clc_m_pair_expr}
        flash_bh = (
            cutlass.Int32(flash_clc_work.tile_idx[2]) * {clc_heads_per_batch}
            + cutlass.Int32(flash_clc_work.tile_idx[1]))
{prelude_block}{inner}
{advance}
{tail}"""
    use_cluster_scheduler = use_2cta_instrs or use_cga2_local_cta
    tile_id_expr = (
        "cutlass.Int32(cute.arch.cluster_idx()[0])"
        if use_cluster_scheduler
        else "cutlass.Int32(cute.arch.block_idx()[0])"
    )
    grid_dim_expr = (
        "cutlass.Int32(cute.arch.cluster_dim()[0])"
        if use_cluster_scheduler
        else "cutlass.Int32(cute.arch.grid_dim()[0])"
    )
    if num_m_pairs is None:
        num_m_pairs_setup = (
            f"    flash_num_m_pairs = {total_tiles_expr} // _flash_num_bh\n"
        )
        num_m_pairs_expr = "flash_num_m_pairs"
    else:
        num_m_pairs_setup = ""
        num_m_pairs_expr = str(num_m_pairs)
    power2_decode = (
        _flash_power2_decode_exprs("flash_tile_id", num_m_pairs)
        if num_m_pairs is not None
        else None
    )
    power2_grid_decode = (
        _flash_power2_decode_exprs("flash_grid_dim", num_m_pairs)
        if num_m_pairs is not None
        else None
    )
    # The counted loop was slightly faster for the measured 32K dense FA4 seed,
    # while longer dense seeds regressed or were neutral. Gate on both total work
    # and num_m_pairs (seq // 256 for 1-CTA FA4) so small batch/head long-sequence
    # cases do not slip through solely because their collapsed tile count is low.
    counted_loop_total_tile_limit = 8192
    counted_loop_m_pair_limit = 512
    use_counted_no_prelude = (
        prelude == "none"
        and total_tiles is not None
        and num_m_pairs is not None
        and total_tiles <= counted_loop_total_tile_limit
        and num_m_pairs <= counted_loop_m_pair_limit
    )
    if use_counted_no_prelude:
        counted_loop_body = inner
        # Preserve the old no-prelude ``flash_tile_id`` semantics for generated
        # role code that observes it, but avoid a measured dead update otherwise.
        if "flash_tile_id" in inner or "flash_tile_id" in tail:
            counted_loop_body += (
                "\n        flash_tile_id = flash_tile_id + flash_grid_dim"
            )
        elif not counted_loop_body.strip():
            counted_loop_body = "        pass"
        return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
    flash_grid_dim = {grid_dim_expr}
    flash_tile_count = cutlass.Int32(0)
    if flash_tile_id < {total_tiles_expr}:
        flash_tile_count = (
            ({total_tiles_expr} - flash_tile_id + flash_grid_dim - cutlass.Int32(1))
            // flash_grid_dim)
    for flash_tile_iter in cutlass.range(flash_tile_count, unroll=1):
{counted_loop_body}
{tail}"""
    if prelude == "none":
        return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
    while flash_tile_id < {total_tiles_expr}:
{inner}
        flash_tile_id = flash_tile_id + {grid_dim_expr}
{tail}"""
    if recompute_tile_coords:
        if power2_decode is not None:
            m_pair_expr, bh_expr = power2_decode
            return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
    flash_grid_dim = {grid_dim_expr}
    while flash_tile_id < {total_tiles_expr}:
        flash_m_pair = {m_pair_expr}
        flash_bh = {bh_expr}
{prelude_block}{inner}
        flash_tile_id = flash_tile_id + flash_grid_dim
{tail}"""
        return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
{num_m_pairs_setup}    flash_grid_dim = {grid_dim_expr}
    while flash_tile_id < {total_tiles_expr}:
        flash_m_pair = flash_tile_id % {num_m_pairs_expr}
        flash_bh = flash_tile_id // {num_m_pairs_expr}
{prelude_block}{inner}
        flash_tile_id = flash_tile_id + flash_grid_dim
{tail}"""
    if power2_decode is not None and power2_grid_decode is not None:
        m_pair_expr, bh_expr = power2_decode
        grid_m_pairs_delta_expr, grid_bh_delta_expr = power2_grid_decode
        return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
    flash_grid_dim = {grid_dim_expr}
    flash_grid_bh_delta = {grid_bh_delta_expr}
    flash_grid_m_pairs_delta = {grid_m_pairs_delta_expr}
    flash_m_pair = {m_pair_expr}
    flash_bh = {bh_expr}
    while flash_tile_id < {total_tiles_expr}:
{prelude_block}{inner}
        flash_tile_id = flash_tile_id + flash_grid_dim
        flash_m_pair = flash_m_pair + flash_grid_m_pairs_delta
        flash_bh = flash_bh + flash_grid_bh_delta
        if flash_m_pair >= {num_m_pairs_expr}:
            flash_m_pair = flash_m_pair - {num_m_pairs_expr}
            flash_bh = flash_bh + 1
{tail}"""
    return f"""
{role_guard}
{head}
    flash_tile_id = {tile_id_expr}
{num_m_pairs_setup}    flash_grid_dim = {grid_dim_expr}
    flash_grid_bh_delta = flash_grid_dim // {num_m_pairs_expr}
    flash_grid_m_pairs_delta = flash_grid_dim - flash_grid_bh_delta * {num_m_pairs_expr}
    flash_m_pair = flash_tile_id % {num_m_pairs_expr}
    flash_bh = flash_tile_id // {num_m_pairs_expr}
    while flash_tile_id < {total_tiles_expr}:
{prelude_block}{inner}
        flash_tile_id = flash_tile_id + flash_grid_dim
        flash_m_pair = flash_m_pair + flash_grid_m_pairs_delta
        flash_bh = flash_bh + flash_grid_bh_delta
        if flash_m_pair >= {num_m_pairs_expr}:
            flash_m_pair = flash_m_pair - {num_m_pairs_expr}
            flash_bh = flash_bh + 1
{tail}"""


def emit_flash_fa4_device_body(
    df: DeviceFunction,
    *,
    head_dim: int,
    num_kv: int,
    num_bh: int,
    total_tiles: int,
    cfg: FlashAttentionConfig,
    has_lse: bool,
    io_dtype: str,
    score_plan: AttentionScorePlan,
    tensor_4d_batch: int = 0,
    tensor_4d_heads: int = 0,
) -> list[ast.stmt]:
    """FA4-topology device body: faithful transcription of the validated 16-warp /
    512-thread spike kernel (sp single-pass softmax body), adapted for Helion's
    flat-divided 3D tensors.

    Stage 2a adds the persistent scheduler (``cfg.persistent``): the grid is capped
    at num_SMs and each CTA strides over its work-items (Q-tile pairs), carrying the
    raw-mbarrier phases (s_full/pfor/pfor2/o_full/s_corr)
    states across work-items -- the spike's ``kernel_persistent`` structure.

    Warp roles (one CTA processes a *pair* of adjacent 128-row Q-tiles): softmax0
    = warps 0-3 (Q-tile 0), softmax1 = warps 4-7 (Q-tile 1), correction = warps
    8-11, MMA = warp 12, load/epilogue = warps 13/14 (order is searchable),
    empty = warp 15. The two softmax warpgroups each run the single-pass
    resident-row softmax_step over the SAME K/V stream, staggered solely by the
    MMA committing ``s_full+0`` before ``s_full+1`` each iteration.

    TMEM: S0 @ col 0, S1 @ col 128, O0 @ col 256, O1 @ col 384. P-over-S aliases
    each S buffer (staged-P r2t split 3/4 + 1/4 on pfor / pfor2). Raw mbarriers
    (s_full / pfor / pfor2 / o_full) carry the MMA<->softmax<->correction
    handshakes; s0_corr / s1_corr are raw full/empty two-stage mbarriers.
    """
    hd = head_dim
    kv_stage = cfg.kv_stage
    q_stage = 2
    s_corr_stage = 2
    assert total_tiles % num_bh == 0
    if cfg.skip_rescale_stats:
        cfg = dataclasses.replace(cfg, skip_rescale_stats=False)
    is_causal = score_plan.is_causal
    if is_causal:
        assert not cfg.persistent
    causal_desc_kv = is_causal and cfg.causal_kv_order == "descending"
    desc_kv = causal_desc_kv or (not is_causal and cfg.kv_order == "descending")
    num_m_pairs = total_tiles // num_bh
    persistent = cfg.persistent
    use_tensor_4d_tma = (
        cfg.tensor_4d_tma
        and tensor_4d_batch > 0
        and tensor_4d_heads > 0
        and tensor_4d_batch * tensor_4d_heads == num_bh
    )
    if not use_tensor_4d_tma:
        tensor_4d_heads = 0
    use_2cta_instrs = cfg.use_2cta_instrs
    use_cga2_local_cta = cfg.use_cga2_local_cta
    use_clc_scheduler = cfg.use_clc_scheduler
    use_local_tma_partition = (
        cfg.local_tma_partition
        and persistent
        and not use_2cta_instrs
        and not use_cga2_local_cta
    )
    if use_clc_scheduler:
        assert persistent
        assert not is_causal
        assert not use_2cta_instrs
        assert not use_cga2_local_cta
    clc_heads_per_batch = (
        cfg.clc_heads_per_batch
        if use_clc_scheduler and cfg.clc_heads_per_batch > 0
        else num_bh
    )
    if num_bh % clc_heads_per_batch != 0:
        clc_heads_per_batch = num_bh
    clc_batch_count = num_bh // clc_heads_per_batch
    split_p_arrive = cfg.split_p_arrive
    sp_whole_row_sum = (
        _FLASH_DENSE_HD64_VERY_LONG_MIN_KV <= num_kv <= 2048
        and not is_causal
        and hd == 64
        and io_dtype == "cutlass.Float16"
    )
    cta_group_size = 2 if use_2cta_instrs else 1
    mma_m = 256 if use_2cta_instrs else 128
    pfor2_count = 2 * 128 if use_2cta_instrs else 128
    pfor_count = (
        pfor2_count
        if cfg.skip_rescale_stats
        else 4 * 128
        if use_2cta_instrs
        else 2 * 128
    )
    pfor_peer_arg = (
        ", cutlass.Int32(0), flash_mma_tile_coord_v" if use_2cta_instrs else ""
    )
    commit_group_arg = (
        ", flash_tcgen05_mcast_mask, cute_tcgen05_flash.CtaGroup.TWO"
        if use_2cta_instrs
        else ""
    )
    gemm_cta_group_arg = f", cta_group={cta_group_size}" if use_2cta_instrs else ""
    # CLC still receives flash_cta_layout_vmnk in PipelineClcFetchAsync below; the
    # ordinary Q/KV TMA pipelines only need the clustered layout for real 2-CTA.
    use_cta_layout_pipeline = use_2cta_instrs
    q_tma_cluster_arg = (
        ", cta_layout_vmnk=flash_cta_layout_vmnk" if use_cta_layout_pipeline else ""
    )
    kv_tma_cluster_arg = (
        ", cta_layout_vmnk=flash_cta_layout_vmnk" if use_cta_layout_pipeline else ""
    )
    q_tma_byte_scale = f" * {cta_group_size}" if use_2cta_instrs else ""
    kv_tma_byte_scale = f" * {cta_group_size}" if use_2cta_instrs else ""
    cluster_init_arrive = (
        "\ncutlass_pipeline_flash.pipeline_init_arrive("
        "cluster_shape_mn=flash_cta_layout_vmnk, is_relaxed=True)"
        if use_cta_layout_pipeline
        else ""
    )
    cluster_init_wait = (
        "\ncutlass_pipeline_flash.pipeline_init_wait("
        "cluster_shape_mn=flash_cta_layout_vmnk)"
        if use_cta_layout_pipeline
        else ""
    )
    tmem_dealloc_init = (
        ""
        if use_2cta_instrs
        else "    cute.arch.mbarrier_init(flash_tmem_dealloc_ptr, 12 * 32)\n"
    )
    kv_loop_bound = "flash_num_active_kv" if is_causal else "_flash_num_kv_tiles"
    kv_loop_bound_minus_1 = f"{kv_loop_bound} - 1"
    epi_smem = cfg.epi_tma or cfg.epi_stg
    storage_extra_args = f", {epi_smem!s}, {use_clc_scheduler!s}, {cfg.clc_stages}"
    prefetch_epi_tma = (
        "\n    cute_cpasync_flash.prefetch_descriptor(_flash_tma_o)"
        if cfg.epi_tma
        else ""
    )
    prefetch_descriptors = f"""
if warp_idx == 0:
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_q)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_k)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_v){prefetch_epi_tma}
"""
    clc_setup = (
        f"""
flash_clc_params = cutlass_utils_flash.ClcDynamicPersistentTileSchedulerParams(
    problem_shape_ntile_mnl=({num_m_pairs}, {clc_heads_per_batch}, {clc_batch_count}),
    cluster_shape_mnk=(1, 1, 1))
flash_clc_hw = cutlass_utils_flash.ClcDynamicPersistentTileScheduler.create(
    flash_clc_params,
    cute.arch.block_idx(),
    cute.arch.grid_dim(),
    storage.clc_response.data_ptr())
flash_clc_response_base = storage.clc_response.data_ptr()
flash_clc_pipeline = cutlass_pipeline_flash.PipelineClcFetchAsync.create(
    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
    num_stages={cfg.clc_stages},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(
        cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(
        cutlass_pipeline_flash.Agent.Thread, 512),
    tx_count=16,
    cta_layout_vmnk=flash_cta_layout_vmnk)
flash_clc_consumer_state = cutlass_pipeline_flash.make_pipeline_state(
    cutlass_pipeline_flash.PipelineUserType.Consumer, {cfg.clc_stages})
"""
        if use_clc_scheduler
        else ""
    )
    # P-over-S aliases S1 at fp16 granularity: (acc_width/io_width) * TMEM_S1.
    p1_off = (32 // 16) * 128  # = 256 fp16 units (acc Float32 -> io Float16)
    # In-place P-over-S layout: P0 aliases S0@0, P1 aliases S1@128. The staged-P STORE
    # targets are tStS0.iterator / tStS1.iterator and the PV-read P addresses are the S
    # addresses directly (flash_s0_addr / flash_s1_addr). The P0->P1 step is 256 fp16
    # units (= 128 f32-cols).
    p1_step = p1_off
    p0_store_iter = "tStS0.iterator"
    p1_store_iter = "tStS1.iterator"
    pv_p0 = "flash_s0_addr"
    pv_p1 = "flash_s1_addr"
    tmem_local_views = """
tStS0 = tStS0_full
tStS1 = tStS1_full
tOtO0 = tOtO0_full
tOtO1 = tOtO1_full
"""
    load_warp = 13 if cfg.role_map == "fa4" else 14
    epi_warp = 14 if cfg.role_map == "fa4" else 13
    # FUSED softmax body (split path): one frag-by-frag pass does scale-subtract +
    # exp2(pipe-split) + f32->f16 convert holding only one 32-elem fragment's
    # transients at a time, then a deferred packed row-sum (spike sp/fa4ip ordering).
    # This collapses the prior 3-pass [exp2_split_inplace (full-row scale-subtract THEN
    # full-row exp2) + fadd_reduce_packed (a 3rd full-row sum pass) + frag-convert]
    # sequence whose coexisting live set (full f32 row + exp2-emulation temps +
    # reduction accumulators + fp16 buffer) overran the 200-reg grant -> cut softmax
    # spills 137->87 (cuobjdump). tSTrS/tSTrS_e are built up front so the fused pass
    # writes the staged-P fp16 chunks directly; flash_p_sum is the returned row-sum.
    if cfg.exp2_impl == "split":
        softmax_exp_block = f"""            tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
            tSTrS_e = cute.make_tensor(
                cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
            _helion_flash_rt.exp2_split_inplace(
                tLDrS, _flash_scale_log2, flash_minus_max_scale,
                {cfg.e2e_freq}, {cfg.e2e_res})
            flash_p_sum = _helion_flash_rt.fadd_reduce_packed(tLDrS)
            flash_cvt_src = cute.logical_divide(tLDrS, cute.make_layout(32))
            flash_cvt_dst = cute.logical_divide(tSTrS_e, cute.make_layout(32))
            for flash_cf in cutlass.range_constexpr(cute.size(tLDrS) // 32):
                flash_cvt_dst[None, flash_cf].store(
                    flash_cvt_src[None, flash_cf].load().to({io_dtype}))"""
    else:
        # Non-split fallback: keep the prior [inline per-element exp2 + separate
        # packed row-sum + frag-by-frag convert] structure (no exp2 pipe-split).
        softmax_exp_block = f"""            flash_n = cute.size(tLDrS)
            for flash_j in cutlass.range_constexpr(flash_n):
                tLDrS[flash_j] = cute.math.exp2(
                    tLDrS[flash_j] * _flash_scale_log2 + flash_minus_max_scale, fastmath=True)
            flash_p_sum = _helion_flash_rt.fadd_reduce_packed(tLDrS)
            tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
            tSTrS_e = cute.make_tensor(
                cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
            flash_cvt_src = cute.logical_divide(tLDrS, cute.make_layout(32))
            flash_cvt_dst = cute.logical_divide(tSTrS_e, cute.make_layout(32))
            for flash_cf in cutlass.range_constexpr(cute.size(tLDrS) // 32):
                flash_cvt_dst[None, flash_cf].store(
                    flash_cvt_src[None, flash_cf].load().to({io_dtype}))"""
    # Non-persistent: decode the one-work-item-per-CTA mapping once in setup. The
    # gmem slices depend on flash_bh and are computed in setup too. Persistent: the
    # decode + per-work-item gmem slices move INSIDE each role's strided while loop
    # (see _flash_fa4_wrap); setup computes only work-item-independent state.
    lpt_swizzle = 1
    if is_causal:
        if cfg.causal_lpt_swizzle > 0:
            lpt_swizzle = cfg.causal_lpt_swizzle
        else:
            size_one_bh = num_kv * 128 * (head_dim + head_dim) * 2
            size_l2 = 50 * 1024 * 1024
            if size_one_bh <= size_l2:
                lpt_swizzle = 1 << int(math.log2(size_l2 // size_one_bh))
        lpt_swizzle = max(1, min(lpt_swizzle, num_bh))
    lpt_group_tiles = lpt_swizzle * num_m_pairs
    lpt_quotient = num_bh // lpt_swizzle
    lpt_remainder = num_bh % lpt_swizzle
    lpt_residual = max(lpt_remainder, 1)
    if lpt_remainder == 0:
        causal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.block_idx()[0])
flash_lpt_group = flash_pid // {lpt_group_tiles}
flash_lpt_mod = flash_pid % {lpt_group_tiles}
flash_m_pair_raw = flash_lpt_mod // {lpt_swizzle}
flash_bh = flash_lpt_group * {lpt_swizzle} + (flash_lpt_mod % {lpt_swizzle})
flash_m_pair = {num_m_pairs - 1} - flash_m_pair_raw
flash_m_tile0 = flash_m_pair * 2
flash_m_tile1 = flash_m_pair * 2 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    else:
        causal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.block_idx()[0])
flash_lpt_group = flash_pid // {lpt_group_tiles}
flash_lpt_mod = flash_pid % {lpt_group_tiles}
flash_m_pair_raw = cutlass.Int32(0)
flash_bh = cutlass.Int32(0)
if flash_lpt_group < {lpt_quotient}:
    flash_m_pair_raw = flash_lpt_mod // {lpt_swizzle}
    flash_bh = flash_lpt_group * {lpt_swizzle} + (flash_lpt_mod % {lpt_swizzle})
else:
    flash_m_pair_raw = flash_lpt_mod // {lpt_residual}
    flash_bh = flash_lpt_group * {lpt_swizzle} + (flash_lpt_mod % {lpt_residual})
flash_m_pair = {num_m_pairs - 1} - flash_m_pair_raw
flash_m_tile0 = flash_m_pair * 2
flash_m_tile1 = flash_m_pair * 2 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    if cfg.use_2cta_instrs:
        noncausal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.cluster_idx()[0])
flash_m_pair = flash_pid % {num_m_pairs}
flash_bh = flash_pid // {num_m_pairs}
flash_q_mma_tile0 = flash_m_pair * 2
flash_q_mma_tile1 = flash_q_mma_tile0 + 1
flash_m_tile0 = flash_q_mma_tile0 * 2 + flash_mma_tile_coord_v
flash_m_tile1 = flash_q_mma_tile1 * 2 + flash_mma_tile_coord_v"""
    elif use_cga2_local_cta:
        noncausal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.cluster_idx()[0])
flash_cluster_m_pair = flash_pid % {num_m_pairs}
flash_bh = flash_pid // {num_m_pairs}
flash_m_pair = flash_cluster_m_pair * 2 + flash_cga2_local_rank
flash_m_tile0 = flash_m_pair * 2
flash_m_tile1 = flash_m_tile0 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    else:
        noncausal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.block_idx()[0])
flash_m_pair = flash_pid % {num_m_pairs}
flash_bh = flash_pid // {num_m_pairs}
flash_m_tile0 = flash_m_pair * 2
flash_m_tile1 = flash_m_tile0 + 1
flash_q_mma_tile0 = flash_m_tile0
flash_q_mma_tile1 = flash_m_tile1"""
    setup_pid = (
        "" if persistent else (causal_setup_pid if is_causal else noncausal_setup_pid)
    )
    active_kv_setup = (
        """
flash_num_active_kv = flash_m_tile1 + cutlass.Int32(1)"""
        if is_causal
        else ""
    )
    if persistent:
        setup_gmem_slice = ""
    elif use_tensor_4d_tma:
        setup_gmem_slice = f"""
flash_head = flash_bh % {tensor_4d_heads}
flash_batch = flash_bh // {tensor_4d_heads}
tQgQ = tQgQ_qdl[None, None, 0, flash_head, flash_batch]
tKgK = tKgK_kdl[None, None, 0, flash_head, flash_batch]
tVgV = tVgV_dkl[None, 0, None, flash_head, flash_batch]"""
    else:
        setup_gmem_slice = """
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]
tVgV = tVgV_dkl[None, 0, None, flash_bh]"""
    cta_group_setup = (
        """
flash_mma_tile_coord_v = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster())
flash_cga2_local_rank = cutlass.Int32(0)
flash_is_leader_cta = flash_mma_tile_coord_v == cutlass.Int32(0)
flash_cta_layout_vmnk = cute.tiled_divide(
    cute.make_layout((2, 1, 1)), (_flash_qk_mma.thr_id.shape,))
flash_tcgen05_mcast_mask = (
    cutlass_pipeline_flash.PipelineUmmaAsync._compute_tmem_sync_mask(
        flash_cta_layout_vmnk))
"""
        if use_2cta_instrs
        else """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster())
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = None
flash_tcgen05_mcast_mask = None
"""
        if use_cga2_local_cta
        else """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cutlass.Int32(0)
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = cute.tiled_divide(
    cute.make_layout((1, 1, 1)), (_flash_qk_mma.thr_id.shape,))
flash_tcgen05_mcast_mask = None
"""
        if use_clc_scheduler
        else """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cutlass.Int32(0)
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = None
flash_tcgen05_mcast_mask = None
"""
    )
    mixed_p_store = cfg.p_store_repetition == 32 and split_p_arrive
    p_store_repetition = 16 if mixed_p_store else cfg.p_store_repetition
    p_store_mixed_setup = (
        """
flash_st32_atom = cute.make_copy_atom(
    cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
flash_tiled_st32_0 = cute_tcgen05_flash.make_tmem_copy(flash_st32_atom, tStS0_P)
flash_tiled_st32_1 = cute_tcgen05_flash.make_tmem_copy(flash_st32_atom, tStS1_P)
flash_thr_st32_0 = flash_tiled_st32_0.get_slice(flash_local_tidx)
flash_thr_st32_1 = flash_tiled_st32_1.get_slice(flash_local_tidx)
tST32tS0 = flash_thr_st32_0.partition_D(tStS0_P)
tST32tS1 = flash_thr_st32_1.partition_D(tStS1_P)
tST32cS = flash_thr_st32_0.partition_S(tScS_P)
flash_P_STORE32_CHUNKS = cute.size(tST32tS0, mode=[2])"""
        if mixed_p_store
        else ""
    )
    setup_tma_partitions = (
        ""
        if use_local_tma_partition
        else f"""
gQ = cute.flat_divide(_flash_mQt, cute.select(({mma_m}, 128, {hd}), mode=[0, 2]))
gK = cute.flat_divide(_flash_mKt, cute.select(({mma_m}, 128, {hd}), mode=[1, 2]))
gV = cute.flat_divide(_flash_mVt, cute.select(({mma_m}, {hd}, 128), mode=[1, 2]))
tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tOgV = flash_pvt.partition_B(gV)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3)){setup_gmem_slice}"""
    )
    setup = f"""
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
{cta_group_setup}{setup_pid}{active_kv_setup}
flash_local_tidx = tidx % 128{prefetch_descriptors}

_flash_storage_cls = _helion_flash_rt.flash_fa4_shared_storage(
    {hd}, {kv_stage}, {q_stage}, {s_corr_stage}, {io_dtype}{storage_extra_args})
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)
sV = cute.make_tensor(cute.recast_ptr(sK.iterator, _flash_vsl.inner), _flash_vsl.outer)
flash_scale_t = storage.sScale.get_tensor(
    cute.make_layout(({s_corr_stage}, {q_stage}, 128)))

# Raw mbarrier init -> fence -> CTA sync, before the pipelines.
flash_s_full_ptr = storage.s_full_mbar.data_ptr()
flash_pfor_ptr = storage.pfor_mbar.data_ptr()
flash_pfor2_ptr = storage.pfor2_mbar.data_ptr()
flash_o_full_ptr = storage.o_full_mbar.data_ptr()
flash_corr_epi_full_ptr = storage.corr_epi_mbar_ptr.data_ptr()
flash_corr_epi_empty_ptr = flash_corr_epi_full_ptr + 2
flash_s0_corr_full_ptr = storage.s0_corr_mbar_ptr.data_ptr()
flash_s0_corr_empty_ptr = flash_s0_corr_full_ptr + {s_corr_stage}
flash_s1_corr_full_ptr = storage.s1_corr_mbar_ptr.data_ptr()
flash_s1_corr_empty_ptr = flash_s1_corr_full_ptr + {s_corr_stage}
flash_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr()
if tidx == 0:
{tmem_dealloc_init.rstrip()}
    for flash_st in cutlass.range_constexpr(2):
        cute.arch.mbarrier_init(flash_s_full_ptr + flash_st, 1)
        cute.arch.mbarrier_init(flash_o_full_ptr + flash_st, 1)
        cute.arch.mbarrier_init(flash_pfor_ptr + flash_st, {pfor_count})
        cute.arch.mbarrier_init(flash_pfor2_ptr + flash_st, {pfor2_count})
        cute.arch.mbarrier_init(flash_corr_epi_full_ptr + flash_st, 128)
        cute.arch.mbarrier_init(flash_corr_epi_empty_ptr + flash_st, 1)
    for flash_st in cutlass.range_constexpr({s_corr_stage}):
        cute.arch.mbarrier_init(flash_s0_corr_full_ptr + flash_st, 128)
        cute.arch.mbarrier_init(flash_s0_corr_empty_ptr + flash_st, 128)
        cute.arch.mbarrier_init(flash_s1_corr_full_ptr + flash_st, 128)
        cute.arch.mbarrier_init(flash_s1_corr_empty_ptr + flash_st, 128)
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()

# TMEM allocator: warp 12 (MMA warp) is the sole allocator. Match FA4's narrower
# retrieve barrier: only MMA, softmax, and correction warps wait for the TMEM
# pointer, so load/epilogue can start without a 512-thread allocation sync.
flash_tmem_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=1, num_threads=512)
flash_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(
    barrier_id=2, num_threads=13 * 32)
flash_tmem = cutlass_utils_flash.TmemAllocator(
    storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_user_bar,
    allocator_warp_id=12, is_two_cta={use_2cta_instrs!s},
    two_cta_tmem_dealloc_mbar_ptr=flash_tmem_dealloc_ptr)
flash_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2])){q_tma_byte_scale}
flash_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2])){kv_tma_byte_scale}
flash_q_prod, flash_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={q_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr(){q_tma_cluster_arg}).make_participants()
flash_kv_prod, flash_kv_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.kv_mbar_ptr.data_ptr(){kv_tma_cluster_arg}).make_participants(){cluster_init_arrive}{cluster_init_wait}{clc_setup}

flash_qkt = _flash_qk_mma.get_slice(flash_mma_tile_coord_v)
flash_pvt = _flash_pv_mma.get_slice(flash_mma_tile_coord_v)
tSrQ = flash_qkt.make_fragment_A(sQ)
tSrK = flash_qkt.make_fragment_B(sK)
tOrV = flash_pvt.make_fragment_B(sV)
flash_qk_acc_shape = flash_qkt.partition_shape_C(({mma_m}, 128))
tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
flash_pv_acc_shape = flash_pvt.partition_shape_C(({mma_m}, {hd}))
tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)
{setup_tma_partitions}
    """

    tmem_base_setup = f"""    flash_tmem.wait_for_alloc()
    flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
    tStS0_full = cute.make_tensor(flash_tmem_ptr, tStS.layout)
    tStS1_full = cute.make_tensor(flash_tmem_ptr + 128, tStS.layout)
    tOtO0_full = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)
    tOtO1_full = cute.make_tensor(flash_tmem_ptr + {256 + hd}, tOtO.layout)
{textwrap.indent(tmem_local_views.strip(), "    ")}
"""
    tmem_mma_setup = (
        tmem_base_setup
        + f"""    tP = cute.make_tensor(tStS0.iterator, _flash_ptl.outer)
    tOrP0 = flash_pvt.make_fragment_A(tP)
    tOrP1 = cute.make_tensor(tOrP0.iterator + {p1_step}, tOrP0.layout)
"""
    )
    tmem_softmax_setup = (
        tmem_base_setup
        + f"""    cS = cute.make_identity_tensor((128, 128))
    tScS = flash_qkt.partition_C(cS)
    flash_ld_atom = cute.make_copy_atom(
        cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition({cfg.s_load_repetition})), cutlass.Float32)
    flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
    flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
    flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
    flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
    tLDtS0 = flash_thr_ld0.partition_S(tStS0)
    tLDtS1 = flash_thr_ld1.partition_S(tStS1)
    tLDcS = flash_thr_ld0.partition_D(tScS)
    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
    flash_tiled_score_st0 = cute_tcgen05_flash.make_tmem_copy(flash_score_st_atom, tStS0)
    flash_tiled_score_st1 = cute_tcgen05_flash.make_tmem_copy(flash_score_st_atom, tStS1)
    flash_thr_score_st0 = flash_tiled_score_st0.get_slice(flash_local_tidx)
    flash_thr_score_st1 = flash_tiled_score_st1.get_slice(flash_local_tidx)
    tScoreSTtS0 = flash_thr_score_st0.partition_D(tStS0)
    tScoreSTtS1 = flash_thr_score_st1.partition_D(tStS1)

    # Staged-P store atom repetition is autotuned. Rep16 preserves the original
    # 4-chunk FA4 granularity; Rep32 halves the P r2t chunk count on hd64.
    flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width
    flash_P_layout = cute.composition(
        tStS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
    tStS0_P = cute.make_tensor({p0_store_iter}, flash_P_layout)
    tStS1_P = cute.make_tensor({p1_store_iter}, flash_P_layout)
    flash_tScS_P_layout = cute.composition(
        tScS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
    tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
    flash_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({p_store_repetition})), cutlass.Float32)
    flash_tiled_st0 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS0_P)
    flash_tiled_st1 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS1_P)
    flash_thr_st0 = flash_tiled_st0.get_slice(flash_local_tidx)
    flash_thr_st1 = flash_tiled_st1.get_slice(flash_local_tidx)
    tSTtS0 = flash_thr_st0.partition_D(tStS0_P)
    tSTtS1 = flash_thr_st1.partition_D(tStS1_P)
    tSTcS = flash_thr_st0.partition_S(tScS_P)
    flash_P_STORE_CHUNKS = cute.size(tSTtS0, mode=[2])
    # Chunked-t2r ("disc") softmax body: the load partition's chunk mode (= 4
    # chunks of 32 f32 cols); load chunk ci aliases the same 32 kv-cols that
    # P-store chunk ci overwrites in place for Rep16. Rep32 halves the P-store
    # chunks, so compute the split barrier in LD chunks and round up.
    flash_LD_CHUNKS = cute.size(tLDtS0, mode=[1])
    flash_LD_CHUNKS_PER_P_STORE = flash_LD_CHUNKS // flash_P_STORE_CHUNKS
    flash_PV_SPLIT_LD_CHUNKS = flash_LD_CHUNKS * 3 // 4
    flash_P_STORE_SPLIT = (
        flash_PV_SPLIT_LD_CHUNKS + flash_LD_CHUNKS_PER_P_STORE - 1
    ) // flash_LD_CHUNKS_PER_P_STORE
{textwrap.indent(p_store_mixed_setup.strip(), "    ") if p_store_mixed_setup else ""}
"""
    )
    default_stage_local_softmax_setup = (
        hd == 64
        and num_kv >= 1024
        and not is_causal
        and not has_lse
        and not score_plan.modifiers
    )
    stage_local_softmax_setup = (
        _flash_bool_env(
            "HELION_CUTE_FLASH_STAGE_LOCAL_SOFTMAX_SETUP",
            default_stage_local_softmax_setup,
        )
        and not mixed_p_store
    )

    def _tmem_softmax_setup_stage(stage: str) -> str:
        ptr_expr = "flash_tmem_ptr" if stage == "0" else "flash_tmem_ptr + 128"
        p_store_iter = p0_store_iter if stage == "0" else p1_store_iter
        coord_setup = (
            "    tLDcS = flash_thr_ld0.partition_D(tScS)\n"
            if stage == "0"
            else """    tStS0_coord = cute.make_tensor(flash_tmem_ptr, tStS.layout)
    flash_tiled_ld_coord = cute_tcgen05_flash.make_tmem_copy(
        flash_ld_atom, tStS0_coord)
    flash_thr_ld_coord = flash_tiled_ld_coord.get_slice(flash_local_tidx)
    tLDcS = flash_thr_ld_coord.partition_D(tScS)
"""
        )
        return f"""    flash_tmem.wait_for_alloc()
    flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
    tStS{stage} = cute.make_tensor({ptr_expr}, tStS.layout)
    cS = cute.make_identity_tensor((128, 128))
    tScS = flash_qkt.partition_C(cS)
    flash_ld_atom = cute.make_copy_atom(
        cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition({cfg.s_load_repetition})), cutlass.Float32)
    flash_tiled_ld{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_ld_atom, tStS{stage})
    flash_thr_ld{stage} = flash_tiled_ld{stage}.get_slice(flash_local_tidx)
    tLDtS{stage} = flash_thr_ld{stage}.partition_S(tStS{stage})
{coord_setup.rstrip()}
    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
    flash_tiled_score_st{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_score_st_atom, tStS{stage})
    flash_thr_score_st{stage} = flash_tiled_score_st{stage}.get_slice(
        flash_local_tidx)
    tScoreSTtS{stage} = flash_thr_score_st{stage}.partition_D(tStS{stage})

    # Staged-P store atom repetition is autotuned. Rep16 preserves the original
    # 4-chunk FA4 granularity; Rep32 halves the P r2t chunk count on hd64.
    flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width
    flash_P_layout = cute.composition(
        tStS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
    tStS{stage}_P = cute.make_tensor({p_store_iter}, flash_P_layout)
    flash_tScS_P_layout = cute.composition(
        tScS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
    tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
    flash_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({p_store_repetition})), cutlass.Float32)
    flash_tiled_st{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_st_atom, tStS{stage}_P)
    flash_thr_st{stage} = flash_tiled_st{stage}.get_slice(flash_local_tidx)
    tSTtS{stage} = flash_thr_st{stage}.partition_D(tStS{stage}_P)
    tSTcS = flash_thr_st{stage}.partition_S(tScS_P)
    flash_P_STORE_CHUNKS = cute.size(tSTtS{stage}, mode=[2])
    # Chunked-t2r ("disc") softmax body: the load partition's chunk mode (= 4
    # chunks of 32 f32 cols); load chunk ci aliases the same 32 kv-cols that
    # P-store chunk ci overwrites in place for Rep16. Rep32 halves the P-store
    # chunks, so compute the split barrier in LD chunks and round up.
    flash_LD_CHUNKS = cute.size(tLDtS{stage}, mode=[1])
    flash_LD_CHUNKS_PER_P_STORE = flash_LD_CHUNKS // flash_P_STORE_CHUNKS
    flash_PV_SPLIT_LD_CHUNKS = flash_LD_CHUNKS * 3 // 4
    flash_P_STORE_SPLIT = (
        flash_PV_SPLIT_LD_CHUNKS + flash_LD_CHUNKS_PER_P_STORE - 1
    ) // flash_LD_CHUNKS_PER_P_STORE
"""

    # Epilogue-warp O store paths use a dedicated O smem staging buffer. sO is
    # 2-staged (_flash_osl stage 2) so both adjacent Q-tiles' rescaled-fp16 O
    # coexist; stage ``s`` is sliced per Q-tile in the corr epilogue.
    #
    # The t2r->rescale->cast->r2s pipeline is a FAITHFUL port of the FA4
    # correction_epilogue (flash_fwd_sm100.py): the rescaled-fp16 O fragment is
    # written into sO via a SMEM-STORE atom (get_smem_store_op) matched to the
    # TMEM-LOAD tiled copy (get_tmem_load_op) so the smem layout the TMA descriptor
    # reads matches the per-thread r2s write. (The committed STG.E.128 path reused a
    # bare Ld32x32 Rep16 t2r + partition_D-over-gmem, whose thread->element map is
    # the TMEM coord map -- valid for a per-thread gmem STG but NOT for an r2s into a
    # row-major sO that the epilogue warp re-reads.) tOsO/tOcO use the PV MMA
    # C-partition. The TMA-O variant additionally builds its TMA partition over
    # the TMA-adjusted _flash_mOt, while the STG variant builds a FA4-style
    # universal-copy tiled store over the regular _flash_mOt view.
    epi_smem_setup = (
        """
sO = storage.sO.get_tensor(_flash_osl.outer, swizzle=_flash_osl.inner)
"""
        if epi_smem
        else ""
    )
    corr_epi_smem_setup = (
        f"""
    # FA4 correction_epilogue tiling. Build the t2r (TMEM->reg) + r2s
    # (reg->smem) tiled copies after this role has retrieved the TMEM pointer.
    flash_o_corr_tile = {cfg.corr_tile_size}
    flash_o_layout_enum = cutlass.utils.layout.LayoutEnum.ROW_MAJOR
    flash_o_epi_subtile = (128, flash_o_corr_tile)
    flash_o_tmem_atom = sm100_utils_flash.get_tmem_load_op(
        (128, {hd}), flash_o_layout_enum, {io_dtype}, cutlass.Float32,
        flash_o_epi_subtile, use_2cta_instrs=False)
    flash_o_cO = cute.make_identity_tensor((128, {hd}))
    tOcO_corr = flash_pvt.partition_C(flash_o_cO)
    tOtO0_corr_i = cute.logical_divide(
        tOtO0, cute.make_layout((128, flash_o_corr_tile)))
    tOtO1_corr_i = cute.logical_divide(
        tOtO1, cute.make_layout((128, flash_o_corr_tile)))
    tOcO_corr_i = cute.logical_divide(
        tOcO_corr, cute.make_layout((128, flash_o_corr_tile)))
    flash_o_tiled_t2r = cute_tcgen05_flash.make_tmem_copy(
        flash_o_tmem_atom, tOtO0_corr_i[(None, None), 0])
    flash_o_smem_atom = sm100_utils_flash.get_smem_store_op(
        flash_o_layout_enum, {io_dtype}, cutlass.Float32, flash_o_tiled_t2r)
    flash_o_tiled_r2s = cute.make_tiled_copy_D(
        flash_o_smem_atom, flash_o_tiled_t2r)
    flash_o_thr_t2r = flash_o_tiled_t2r.get_slice(flash_local_tidx)
    tOcO_corr_t2r = flash_o_thr_t2r.partition_D(
        tOcO_corr_i[(None, None), None])
    tOtO0_corr_t2r = flash_o_thr_t2r.partition_S(
        tOtO0_corr_i[(None, None), None])
    tOtO1_corr_t2r = flash_o_thr_t2r.partition_S(
        tOtO1_corr_i[(None, None), None])
    tOsO0_corr = flash_pvt.partition_C(sO[None, None, 0])
    tOsO1_corr = flash_pvt.partition_C(sO[None, None, 1])
    tOsO0_corr_i = cute.logical_divide(
        tOsO0_corr, cute.make_layout((128, flash_o_corr_tile)))
    tOsO1_corr_i = cute.logical_divide(
        tOsO1_corr, cute.make_layout((128, flash_o_corr_tile)))
    tOsO0_corr_r2s = _helion_flash_rt.partition_D_position_independent(
        flash_o_thr_t2r, tOsO0_corr_i[(None, None), None])
    tOsO1_corr_r2s = _helion_flash_rt.partition_D_position_independent(
        flash_o_thr_t2r, tOsO1_corr_i[(None, None), None])
    flash_o_corr_chunks = {hd} // flash_o_corr_tile
"""
        if epi_smem
        else ""
    )
    epi_tma_setup = (
        f"""
gO_tma = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))
tOgO_tma_mma = flash_pvt.partition_C(gO_tma)
tOsO_tma, tOgO_tma = cute_cpasync_flash.tma_partition(
    _flash_tma_o, 0, cute.make_layout(1),
    cute.group_modes(sO, 0, 2), cute.group_modes(tOgO_tma_mma, 0, 3))
"""
        if cfg.epi_tma
        else ""
    )
    # Moving O-TMA partition setup into the epilogue role helps the measured
    # output-only dense noncausal hd64 32K/65K shapes, but 131K measured
    # neutral/slower. Also use it for the 2048+ KV-tile family where measured
    # TMA-O wins.
    local_epi_tma_setup = (
        cfg.epi_tma
        and not is_causal
        and hd == 64
        and not has_lse
        and not score_plan.modifiers
        and (num_kv < 1024 or num_kv >= 2048)
    )
    epi_stg_setup = (
        f"""
flash_o_store_bits = 128
flash_o_async_copy_elems = flash_o_store_bits // {io_dtype}.width
flash_o_tO_shape_dim_1 = {hd} // flash_o_async_copy_elems
flash_o_tO_layout = cute.make_ordered_layout(
    (32 // flash_o_tO_shape_dim_1, flash_o_tO_shape_dim_1), order=(1, 0))
flash_o_vO_layout = cute.make_layout((1, flash_o_async_copy_elems))
flash_o_gmem_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), {io_dtype}, num_bits_per_copy=flash_o_store_bits)
flash_o_gmem_tiled_copy = cute.make_tiled_copy_tv(
    flash_o_gmem_atom, flash_o_tO_layout, flash_o_vO_layout)
flash_epi_tidx = tidx % 32
flash_o_gmem_thr = flash_o_gmem_tiled_copy.get_slice(flash_epi_tidx)
tOsO_epi0 = flash_o_gmem_thr.partition_S(sO[None, None, 0])
tOsO_epi1 = flash_o_gmem_thr.partition_S(sO[None, None, 1])
"""
        if cfg.epi_stg
        else ""
    )

    # Per-role setmaxnreg placed at the HEAD of each role guard (NOT a unified
    # if/elif dispatch -- that triggers the C7600). Softmax/correction/other
    # register budgets follow the FA4 setmaxnreg split. Each role is assembled as
    # (guard, head, inner) and wrapped by ``_flash_fa4_wrap``: ``head`` (setmaxnreg
    # + cross-work-item phase decls) lives directly under the guard; ``inner`` (the
    # per-work-item body, 8-space indented) runs once (flat) or in a strided
    # while loop (persistent). The raw-mbarrier phases carry across work-items.
    # EMPTY (15) / EPI: setmaxnreg only, no work -- but under persistence they
    # MUST hit the per-work-item CTA barrier in lockstep, so they are wrapped too
    # (their inner is empty; the wrap supplies the prelude + barrier + stride).
    if use_clc_scheduler:
        empty_block = f"""
if warp_idx == 15:
    cute.arch.setmaxregister_decrease({cfg.other_regs})
    flash_clc_producer_state = cutlass_pipeline_flash.make_pipeline_state(
        cutlass_pipeline_flash.PipelineUserType.Producer, {cfg.clc_stages})
    flash_clc_work = flash_clc_hw.initial_work_tile_info()
    while flash_clc_work.is_valid_tile:
        flash_clc_pipeline.producer_acquire(flash_clc_producer_state)
        flash_clc_mbar = flash_clc_pipeline.producer_get_barrier(
            flash_clc_producer_state)
        flash_clc_response_ptr = (
            flash_clc_response_base
            + flash_clc_producer_state.index * cutlass.Int32(4))
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(flash_clc_mbar, flash_clc_response_ptr)
        flash_clc_producer_state.advance()
        flash_clc_pipeline.consumer_wait(flash_clc_consumer_state)
        flash_clc_response_ptr = (
            flash_clc_response_base
            + flash_clc_consumer_state.index * cutlass.Int32(4))
        flash_clc_work = flash_clc_hw.work_tile_info_from_clc_response(
            flash_clc_response_ptr)
        flash_clc_pipeline.consumer_release(flash_clc_consumer_state)
        flash_clc_consumer_state.advance()
    flash_clc_pipeline.producer_tail(flash_clc_producer_state)
"""
    else:
        empty_block = _flash_fa4_wrap(
            "if warp_idx == 15:",
            f"    cute.arch.setmaxregister_decrease({cfg.other_regs})",
            "",
            persistent,
            prelude="none",
            total_tiles=total_tiles,
            num_m_pairs=num_m_pairs,
            use_2cta_instrs=use_2cta_instrs,
            use_cga2_local_cta=use_cga2_local_cta,
            clc_heads_per_batch=clc_heads_per_batch,
            tensor_4d_heads=tensor_4d_heads,
            recompute_tile_coords=cfg.recompute_tile_coords,
        )
    if use_tensor_4d_tma:
        epi_tma_gmem0 = "tOgO_tma[None, flash_m_tile0, 0, flash_head, flash_batch]"
        epi_tma_gmem1 = "tOgO_tma[None, flash_m_tile1, 0, flash_head, flash_batch]"
        epi_stg_mO_cur = "_flash_mOt[None, None, flash_head, flash_batch]"
        corr_gmem_o_index = (
            "flash_pvt.partition_C(flash_gO_corr)"
            "[None, None, None, {mtile}, 0, flash_head, flash_batch]"
        )
    else:
        epi_tma_gmem0 = "tOgO_tma[None, flash_m_tile0, 0, flash_bh]"
        epi_tma_gmem1 = "tOgO_tma[None, flash_m_tile1, 0, flash_bh]"
        epi_stg_mO_cur = "_flash_mOt[None, None, flash_bh]"
        corr_gmem_o_index = (
            "flash_pvt.partition_C(flash_gO_corr)"
            "[None, None, None, {mtile}, 0, flash_bh]"
        )
    if cfg.epi_tma:
        epi_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
{textwrap.indent(epi_tma_setup.strip(), "    ") if local_epi_tma_setup else ""}
    flash_corr_epi_full_phase = cutlass.Int32(0)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"""
        epi_inner = f"""        _helion_flash_rt.mbar_spin_wait(
            flash_corr_epi_full_ptr + 0, flash_corr_epi_full_phase)
        with cute.arch.elect_one():
            cute.copy(_flash_tma_o, tOsO_tma[None, 0], {epi_tma_gmem0})
            cute.arch.cp_async_bulk_commit_group()
        _helion_flash_rt.mbar_spin_wait(
            flash_corr_epi_full_ptr + 1, flash_corr_epi_full_phase)
        with cute.arch.elect_one():
            cute.copy(_flash_tma_o, tOsO_tma[None, 1], {epi_tma_gmem1})
            cute.arch.cp_async_bulk_commit_group()
        with cute.arch.elect_one():
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)
        flash_corr_epi_full_phase ^= 1"""
        epi_prelude = "decode"
    elif cfg.epi_stg:
        epi_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
    flash_corr_epi_full_phase = cutlass.Int32(0)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"""
        epi_stg_store_fn = (
            "fa4_store_o_smem_to_gmem_whole"
            if cfg.epi_stg_store == "whole"
            else "fa4_store_o_smem_to_gmem"
        )
        if cfg.epi_stg_gmem == "pair":
            epi_stg_gmem_setup = f"""
        flash_gO_pair = cute.local_tile(
            flash_mO_cur, ({mma_m * q_stage}, {hd}), (flash_m_pair, 0))
        flash_gO_pair = _helion_flash_rt.select(
            cute.flat_divide(flash_gO_pair, ({mma_m},)), mode=[0, 2, 1])"""
            epi_stg_gmem_expr = "flash_gO_pair[None, None, {stage}]"
        else:
            epi_stg_gmem_setup = ""
            epi_stg_gmem_expr = (
                "cute.local_tile(flash_mO_cur, (128, {hd}), (flash_m_tile{stage}, 0))"
            )

        def _epi_stg_inner(stage: str) -> str:
            return f"""        _helion_flash_rt.mbar_spin_wait(
            flash_corr_epi_full_ptr + {stage}, flash_corr_epi_full_phase)
        tOgO_stg{stage} = {epi_stg_gmem_expr.format(stage=stage, hd=hd)}
        _helion_flash_rt.{epi_stg_store_fn}(
            flash_o_gmem_tiled_copy, flash_o_gmem_thr,
            tOsO_epi{stage}, tOgO_stg{stage}, {io_dtype})
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + {stage})"""

        epi_inner = (
            f"""        flash_mO_cur = {epi_stg_mO_cur}
{epi_stg_gmem_setup}
"""
            + _epi_stg_inner("0")
            + "\n"
            + _epi_stg_inner("1")
            + "\n        flash_corr_epi_full_phase ^= 1"
        )
        epi_prelude = "decode"
    else:
        epi_head = f"    cute.arch.setmaxregister_decrease({cfg.other_regs})"
        epi_inner = ""
        epi_prelude = "none"
    epi_block = _flash_fa4_wrap(
        f"if warp_idx == {epi_warp}:",
        epi_head,
        epi_inner,
        persistent,
        prelude=epi_prelude,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )

    load_head = f"    cute.arch.setmaxregister_decrease({cfg.other_regs})"
    if desc_kv:
        load_first_kv = f"{kv_loop_bound} - cutlass.Int32(1)"
        load_next_kv = f"{kv_loop_bound} - cutlass.Int32(2) - flash_kv_load_iter"
        load_loop_var = "flash_kv_load_iter"
    else:
        load_first_kv = "0"
        load_next_kv = "flash_kv_load_iter + cutlass.Int32(1)"
        load_loop_var = "flash_kv_load_iter"
    if use_local_tma_partition:
        if use_tensor_4d_tma:
            load_mq_cur = "_flash_mQt[None, None, flash_head, flash_batch]"
            load_mk_cur = "_flash_mKt[None, None, flash_head, flash_batch]"
            load_mv_cur = "_flash_mVt[None, None, flash_head, flash_batch]"
        else:
            load_mq_cur = "_flash_mQt[None, None, flash_bh]"
            load_mk_cur = "_flash_mKt[None, None, flash_bh]"
            load_mv_cur = "_flash_mVt[None, None, flash_bh]"
        local_load_tma_block = f"""        flash_mQ_cur = {load_mq_cur}
        flash_mK_cur = {load_mk_cur}
        flash_mV_cur = {load_mv_cur}
        flash_gQ_pair = cute.local_tile(
            flash_mQ_cur, ({mma_m * q_stage}, {hd}), (flash_m_pair, 0))
        flash_gQ_pair = _helion_flash_rt.select(
            cute.flat_divide(flash_gQ_pair, ({mma_m},)), mode=[0, 2, 1])
        flash_gK = cute.local_tile(flash_mK_cur, (128, {hd}), (None, 0))
        flash_gV = cute.local_tile(flash_mV_cur, ({hd}, 128), (0, None))
        tSgQ = flash_qkt.partition_A(flash_gQ_pair)
        tSgK = flash_qkt.partition_B(flash_gK)
        tOgV = flash_pvt.partition_B(flash_gV)
        tQsQ, tQgQ = cute_cpasync_flash.tma_partition(
            _flash_tma_q, 0, cute.make_layout(1),
            cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
        tKsK, tKgK = cute_cpasync_flash.tma_partition(
            _flash_tma_k, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
        tVsV, tVgV = cute_cpasync_flash.tma_partition(
            _flash_tma_v, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))
"""
        load_prelude_mode = "decode"
    else:
        local_load_tma_block = ""
        load_prelude_mode = "full"
    load_q0_src = "0" if use_local_tma_partition else "flash_q_mma_tile0"
    load_q1_src = "1" if use_local_tma_partition else "flash_q_mma_tile1"
    load_q0 = f"""        flash_qe0 = flash_q_prod.acquire_and_advance()
        cute.copy(_flash_tma_q, tQgQ[None, {load_q0_src}], tQsQ[None, flash_qe0.index],
                  tma_bar_ptr=flash_qe0.barrier)"""
    load_q1 = f"""        flash_qe1 = flash_q_prod.acquire_and_advance()
        cute.copy(_flash_tma_q, tQgQ[None, {load_q1_src}], tQsQ[None, flash_qe1.index],
                  tma_bar_ptr=flash_qe1.barrier)"""
    load_k0 = f"""        flash_kve = flash_kv_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, {load_first_kv}], tKsK[None, flash_kve.index],
                  tma_bar_ptr=flash_kve.barrier)"""
    load_v0 = f"""        flash_kve = flash_kv_prod.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, {load_first_kv}], tVsV[None, flash_kve.index],
                  tma_bar_ptr=flash_kve.barrier)"""
    load_prologue = _flash_fa4_load_prologue_for_order(
        cfg.first_load_order, load_q0, load_k0, load_q1, load_v0
    )
    load_inner = f"""{local_load_tma_block}{load_prologue}
        for {load_loop_var} in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
            flash_kv_next = {load_next_kv}
            flash_kve = flash_kv_prod.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, flash_kv_next], tKsK[None, flash_kve.index],
                      tma_bar_ptr=flash_kve.barrier)
            flash_kve = flash_kv_prod.acquire_and_advance()
            cute.copy(_flash_tma_v, tVgV[None, flash_kv_next], tVsV[None, flash_kve.index],
                      tma_bar_ptr=flash_kve.barrier)"""
    load_block = _flash_fa4_wrap(
        f"if warp_idx == {load_warp}:",
        load_head,
        load_inner,
        persistent,
        prelude=load_prelude_mode,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )

    # MMA warp (12): prologue QK0(0)/QK1(0); steady i=0..N-2 interleave PV(i)
    # (staged on pfor/pfor2) with QK(i+1) (commit s_full); epilogue PV(N-1) commit
    # o_full. The PV-before-QK program order protects the in-place P-over-S (S_full
    # commit-only, no empty-S acquire). flash_pfor_phase is the head (carries across
    # work-items); the inner flips it once at the end of each work-item so the
    # carried parity matches the spike kernel_persistent.
    #
    # cfg.mma_ptx (Stage 2b, default ON): issue every QK/PV via
    # _helion_flash_ptx.gemm_ptx_partial -- one inline-asm region with literal
    # descriptors. This fits the MMA warp at 48 regs (cute.gemm spills ~116/133)
    # AND folds the pfor2 wait INSIDE the PV issue stream (mbar_ptr=pfor2), so the
    # tensor-core stream is never broken by a Python-level spin-wait between the 3/4
    # and 1/4 PV K-chunks. The S/O TMEM column addresses are loop-invariant (TMEM is
    # fixed for the whole kernel) so they are hoisted into the head.
    if cfg.mma_ptx:
        q_stage_stride = mma_m * hd // 8
        if cfg.precompute_qk_desc:
            qk_desc_head = """
    flash_q_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(
        sQ, _helion_flash_ptx.Major.K)
    _helion_flash_ptx.declare_ptx_smem_desc(
        _helion_flash_ptx.make_smem_desc_start_addr(
            sQ[None, None, None, 0].iterator),
        flash_q_smem_base, tSrQ[None, None, None, 0].layout,
        "helion_flash_q0_smem_desc")
    _helion_flash_ptx.declare_ptx_smem_desc(
        _helion_flash_ptx.make_smem_desc_start_addr(
            sQ[None, None, None, 1].iterator),
        flash_q_smem_base, tSrQ[None, None, None, 0].layout,
        "helion_flash_q1_smem_desc")
    _helion_flash_ptx.declare_ptx_idesc(
        _flash_qk_mma.op, "helion_flash_qk_mma_idesc")"""

            def _qk_gemm(stage: str, k_state: str) -> str:
                return f"""            _helion_flash_ptx.gemm_ptx_precomputed_qk_static(
                _flash_qk_mma.op, flash_s{stage}_addr,
                tSrK[None, None, None, {k_state}.index],
                sK[None, None, None, {k_state}.index],
                "helion_flash_q{stage}_smem_desc",
                "helion_flash_qk_mma_idesc",
                zero_init=True{gemm_cta_group_arg})"""

        else:
            qk_desc_head = f"""
    flash_q_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(
        sQ, _helion_flash_ptx.Major.K)
    _helion_flash_ptx.declare_ptx_smem_desc(
        _helion_flash_ptx.make_smem_desc_start_addr(
            sQ[None, None, None, {q_stage - 1}].iterator),
        flash_q_smem_base, tSrQ[None, None, None, 0].layout,
        "helion_flash_q_smem_desc")
    flash_k_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(
        sK, _helion_flash_ptx.Major.K)
    _helion_flash_ptx.declare_ptx_idesc(
        _flash_qk_mma.op, "helion_flash_qk_mma_idesc")"""

            def _qk_gemm(stage: str, k_state: str) -> str:
                smem_offset = -q_stage_stride if stage == "0" else q_stage_stride
                return f"""            _helion_flash_ptx.gemm_ptx_precomputed_qk(
                flash_s{stage}_addr,
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sK[None, None, None, {k_state}.index].iterator),
                flash_k_smem_base,
                tSrK[None, None, None, 0].layout,
                "helion_flash_q_smem_desc",
                "helion_flash_qk_mma_idesc",
                smem_offset={smem_offset},
                zero_init=True{gemm_cta_group_arg})"""

        pv_desc_head = """
    flash_v_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(
        sV, _helion_flash_ptx.Major.MN)
    _helion_flash_ptx.declare_ptx_idesc(
        _flash_pv_mma.op, "helion_flash_pv_mma_idesc")"""
        # flash_s0_addr/flash_s1_addr are the TMEM addresses of the PV-gemm P operand
        # source (passed as tA_addr; for a TS gemm tCrA.iterator.toint() returns 0). With
        # the in-place P-over-S layout P0==S0@0 / P1==S1@128 so they equal the S addresses.
        mma_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
    flash_tmem.allocate(512)
{tmem_mma_setup.rstrip()}
    flash_pfor_phase = cutlass.Int32(0)
    flash_s0_addr = tStS0.iterator.toint()
    flash_s1_addr = tStS1.iterator.toint()
    flash_o0_addr = tOtO0.iterator.toint()
    flash_o1_addr = tOtO1.iterator.toint(){qk_desc_head}{pv_desc_head}"""
        pv0_split_wait_arg = (
            "\n                mbar_ptr=flash_pfor2_ptr + 0, mbar_phase=flash_pfor_phase,"
            if split_p_arrive
            else ""
        )
        pv1_split_wait_arg = (
            "\n                mbar_ptr=flash_pfor2_ptr + 1, mbar_phase=flash_pfor_phase,"
            if split_p_arrive
            else ""
        )

        # STEADY-loop body (in-place P): PV(i) issued BEFORE QK(i+1) -- the PV-before-QK
        # program order is what protects the in-place P-over-S; the pfor2 wait is folded
        # inside the PV gemm.
        mma_steady_body = f"""
            # stage 0: PV0(i) (pfor2 wait folded inside the gemm) then QK0(i+1).
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase)
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o0_addr, {pv_p0},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP0[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv0_split_wait_arg}
                zero_init=flash_o_zero0{gemm_cta_group_arg})
            flash_o_zero0 = cutlass.Boolean(False)
            flash_k_full = flash_kv_cons.wait_and_advance()
{_qk_gemm("0", "flash_k_full")}
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
            # stage 1: PV1(i) (pfor2 folded) then QK1(i+1).
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase)
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o1_addr, {pv_p1},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP1[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv1_split_wait_arg}
                zero_init=flash_o_zero1{gemm_cta_group_arg})
            flash_o_zero1 = cutlass.Boolean(False)
            flash_v_full.release()
{_qk_gemm("1", "flash_k_full")}
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(flash_s_full_ptr + 1{commit_group_arg})
            flash_k_full.release()"""
        mma_inner = f"""        flash_q0_full = flash_q_cons.wait_and_advance()
        flash_q1_full = flash_q_cons.wait_and_advance()

        # PROLOGUE: QK0(0)->S0, QK1(0)->S1 against K0; then release K0.
        flash_k0_full = flash_kv_cons.wait_and_advance()
{textwrap.indent(_qk_gemm("0", "flash_k0_full").lstrip(), "        ")}
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
{textwrap.indent(_qk_gemm("1", "flash_k0_full").lstrip(), "        ")}
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_s_full_ptr + 1{commit_group_arg})
        flash_k0_full.release()

        # STEADY: i = 0..N-2. PV(i) for both stages interleaved with QK(i+1). o_zero
        # is the per-tile zero-init Boolean (True first tile, False after).
        flash_o_zero0 = cutlass.Boolean(True)
        flash_o_zero1 = cutlass.Boolean(True)
        for flash_i in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
            flash_v_full = flash_kv_cons.wait_and_advance(){mma_steady_body}
            flash_pfor_phase ^= 1

        flash_q0_full.release()
        flash_q1_full.release()

        # EPILOGUE: PV(N-1) for both stages; commit O_full.
        flash_v_full = flash_kv_cons.wait_and_advance()
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase)
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
            flash_o0_addr, {pv_p0},
            _helion_flash_ptx.make_smem_desc_start_addr(
                sV[None, None, None, flash_v_full.index].iterator),
            flash_v_smem_base,
            tOrP0[None, None, None, 0].layout,
            tOrV[None, None, None, 0].layout,
            "helion_flash_pv_mma_idesc",{pv0_split_wait_arg}
            zero_init=flash_o_zero0{gemm_cta_group_arg})
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_o_full_ptr + 0{commit_group_arg})
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase)
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
            flash_o1_addr, {pv_p1},
            _helion_flash_ptx.make_smem_desc_start_addr(
                sV[None, None, None, flash_v_full.index].iterator),
            flash_v_smem_base,
            tOrP1[None, None, None, 0].layout,
            tOrV[None, None, None, 0].layout,
            "helion_flash_pv_mma_idesc",{pv1_split_wait_arg}
            zero_init=flash_o_zero1{gemm_cta_group_arg})
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_o_full_ptr + 1{commit_group_arg})
        flash_v_full.release()
        # The 2 epilogue PV waits did NOT flip pfor_phase; flip once so the carried
        # parity matches the next work-item's correction pre-arrive (spike L1973-1977).
        flash_pfor_phase ^= 1"""
    else:
        mma_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
    flash_tmem.allocate(512)
{tmem_mma_setup.rstrip()}
    flash_nk = cute.size(tSrQ, mode=[2])
    flash_nk2 = cute.size(tOrP0, mode=[2])
    flash_pv_split = flash_nk2 // 4 * 3
    flash_pfor_phase = cutlass.Int32(0)"""
        mma_inner = f"""        flash_q0_full = flash_q_cons.wait_and_advance()
        flash_q1_full = flash_q_cons.wait_and_advance()

        # PROLOGUE: QK0(0)->S0, QK1(0)->S1 against K0; then release K0.
        flash_k0_full = flash_kv_cons.wait_and_advance()
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q0_full.index],
                      tSrK[None, None, flash_kp, flash_k0_full.index], tStS0)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS1, tSrQ[None, None, flash_kp, flash_q1_full.index],
                      tSrK[None, None, flash_kp, flash_k0_full.index], tStS1)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_s_full_ptr + 1{commit_group_arg})
        flash_k0_full.release()

        # STEADY: i = 0..N-2. PV(i) for both stages interleaved with QK(i+1).
        flash_O_acc0 = False
        flash_O_acc1 = False
        for flash_i in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
            flash_v_full = flash_kv_cons.wait_and_advance()
            # stage 0: PV0(i) then QK0(i+1). STAGED-P: first 96 kv on pfor.
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase)
            for flash_kp in cutlass.range_constexpr(flash_pv_split):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc0 | (flash_kp != 0))
                cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
            _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 0, flash_pfor_phase)
            for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
                cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
            flash_O_acc0 = True
            flash_k_full = flash_kv_cons.wait_and_advance()
            for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q0_full.index],
                          tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
            # stage 1: PV1(i) then QK1(i+1). STAGED-P: first 96 kv on pfor.
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase)
            for flash_kp in cutlass.range_constexpr(flash_pv_split):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc1 | (flash_kp != 0))
                cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
            _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 1, flash_pfor_phase)
            for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
                cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
            flash_O_acc1 = True
            flash_v_full.release()
            for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                cute.gemm(_flash_qk_mma, tStS1, tSrQ[None, None, flash_kp, flash_q1_full.index],
                          tSrK[None, None, flash_kp, flash_k_full.index], tStS1)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(flash_s_full_ptr + 1{commit_group_arg})
            flash_k_full.release()
            flash_pfor_phase ^= 1

        flash_q0_full.release()
        flash_q1_full.release()

        # EPILOGUE: PV(N-1) for both stages; commit O_full.
        flash_v_full = flash_kv_cons.wait_and_advance()
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase)
        for flash_kp in cutlass.range_constexpr(flash_pv_split):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc0 | (flash_kp != 0))
            cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
        _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 0, flash_pfor_phase)
        for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
            cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_o_full_ptr + 0{commit_group_arg})
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase)
        for flash_kp in cutlass.range_constexpr(flash_pv_split):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc1 | (flash_kp != 0))
            cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
        _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 1, flash_pfor_phase)
        for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
            cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_o_full_ptr + 1{commit_group_arg})
        flash_v_full.release()
        # The 2 epilogue PV waits did NOT flip pfor_phase; flip once so the carried
        # parity matches the next work-item's correction pre-arrive (spike L1973-1977).
        flash_pfor_phase ^= 1"""
    mma_tmem_teardown = (
        """    flash_tmem.relinquish_alloc_permit()
    flash_tmem_user_bar.arrive_and_wait()
    flash_tmem.free(flash_tmem_ptr)"""
        if use_2cta_instrs
        else """    flash_tmem.relinquish_alloc_permit()
    cute.arch.mbarrier_wait(flash_tmem_dealloc_ptr, 0)
    flash_tmem.free(flash_tmem_ptr)"""
    )
    mma_block = _flash_fa4_wrap(
        "if (warp_idx == 12) & flash_is_leader_cta:"
        if use_2cta_instrs
        else "if warp_idx == 12:",
        mma_head,
        mma_inner,
        persistent,
        prelude="none",
        tail=mma_tmem_teardown,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )

    # SOFTMAX0/1 (warps 0-3 / 4-7): single-pass resident-row body. The
    # flash_s_full_phase carries across work-items (head); row_max/row_sum re-init
    # per work-item (inner). The frag-by-frag f32->f16 convert (32-wide) caps the
    # register peak (a full-row tLDrS.load() materialises a 128-wide TensorSSA on
    # top of the resident row -> spills past the 200 grant).
    # Step 3 L1: route PASS2 to the software-pipelined helper when disc_pipe_depth>=2
    # (prefetch chunk ci+depth's t2r before chunk ci's exp2 burst); depth==1 keeps the
    # byte-identical serial helper as the default.
    sload16_paired = cfg.s_load_repetition == 16 and cfg.p_store_repetition == 16
    if mixed_p_store:
        _disc_pass2_name = "fa4_disc_exp_convert_store_rep32_split"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_rep32_split_causal"
    elif sload16_paired:
        _disc_pass2_name = "fa4_disc_exp_convert_store_sload16_pair_pipe"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_sload16_pair_pipe_causal"
    elif cfg.p_store_repetition == 32 and cfg.disc_pipe_depth >= 2:
        _disc_pass2_name = "fa4_disc_exp_convert_store_rep32_pipe"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_rep32_pipe_causal"
    elif cfg.p_store_repetition == 32:
        _disc_pass2_name = "fa4_disc_exp_convert_store_rep32"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_rep32_causal"
    elif cfg.disc_pipe_depth >= 2:
        _disc_pass2_name = "fa4_disc_exp_convert_store_pipe"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_pipe_causal"
    else:
        _disc_pass2_name = "fa4_disc_exp_convert_store"
        _disc_pass2_causal_name = "fa4_disc_exp_convert_store_causal"
    softmax_loop_var = "flash_kv_iter" if desc_kv else "flash_kv"
    softmax_actual_kv = (
        f"\n            flash_kv = {kv_loop_bound} - cutlass.Int32(1) - flash_kv_iter"
        if desc_kv
        else ""
    )
    softmax_not_first = "flash_kv_iter != 0" if desc_kv else "flash_kv != 0"

    # FA4 rescale_threshold (alpha-pin) softmax block. When the running max grows by
    # less than the threshold (scale_log2*(old-new) >= -thresh), keep the OLD max and
    # pin alpha=1.0 so the correction warp's vote_ballot(alpha<1.0) is false for the
    # whole warp -> the O-rescale (t2r/mul/r2t/fence) on the correction->PV critical
    # path is SKIPPED. ``flash_minus_max_scale`` is computed AFTER the pin so
    # PASS2's exp2 consumes the (possibly kept-old) max. The threshold is a static
    # codegen constant (literal), so the compare is const-folded; threshold==0.0
    # (fp8) emits the prior always-rescale block byte-identically.
    # Stage-parameterized, so one definition covers both softmax warpgroups (0/1).
    def _disc_alpha_block_for(not_first: str) -> str:
        if cfg.rescale_threshold > 0.0:
            pin_condition = (
                f"({not_first}) & (flash_acc_log >= -{cfg.rescale_threshold})"
            )
            return f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {pin_condition}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"""

        return """            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2
            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""

    _disc_alpha_block = _disc_alpha_block_for(softmax_not_first)
    # The whole-row (non-disc) body computes flash_minus_max_scale BEFORE the exp2
    # PASS (which consumes it) and flash_alpha AFTER, so the alpha-pin reorders to a
    # PRE-exp block (decide alpha + pin the max) so the kept-old max feeds the exp
    # PASS via flash_minus_max_scale, and an empty POST-exp piece (no second alpha
    # compute). threshold==0.0 keeps the prior pre/post split byte-identically.
    if cfg.rescale_threshold > 0.0:
        sp_pin_condition = (
            f"({softmax_not_first}) & (flash_acc_log >= -{cfg.rescale_threshold})"
        )
        _sp_alpha_pre = f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {sp_pin_condition}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"""
        _sp_alpha_post = ""
    else:
        _sp_alpha_pre = (
            "            flash_minus_max_scale ="
            " (0.0 - flash_row_max_safe) * _flash_scale_log2"
        )
        _sp_alpha_post = """            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""

    def _softmax_inner(
        stage: str,
        ld: str,
        ldt: str,
        st: str,
        stt: str,
        score_st: str,
        score_stt: str,
    ) -> str:
        def _format_disc_pass2(name: str, *, causal: bool) -> str:
            e2e_freq = cfg.masked_e2e_freq if causal else cfg.e2e_freq
            e2e_res = cfg.masked_e2e_res if causal else cfg.e2e_res
            e2e_offset = (
                0
                if e2e_res == 0
                else str(cfg.e2e_offset0)
                if stage == "0"
                else str(cfg.e2e_offset)
            )
            if mixed_p_store:
                args = [
                    ld,
                    ldt,
                    "tLDcS",
                    f"flash_tiled_st32_{stage}",
                    f"tST32tS{stage}",
                    "tST32cS",
                    st,
                    stt,
                    "tSTcS",
                    "_flash_scale_log2",
                    "flash_minus_max_scale",
                    str(e2e_freq),
                    str(e2e_res),
                    str(e2e_offset),
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}",
                    "flash_LD_CHUNKS",
                ]
            else:
                args = [
                    ld,
                    ldt,
                    "tLDcS",
                    st,
                    stt,
                    "tSTcS",
                    "_flash_scale_log2",
                    "flash_minus_max_scale",
                    str(e2e_freq),
                    str(e2e_res),
                    str(e2e_offset),
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}" if split_p_arrive else "None",
                    "flash_P_STORE_SPLIT",
                    "flash_P_STORE_CHUNKS",
                ]
            if (cfg.disc_pipe_depth >= 2 or sload16_paired) and not mixed_p_store:
                args.append(str(cfg.disc_pipe_depth))
            if causal:
                args.extend([f"flash_m_tile{stage}", "flash_kv"])
            args.append(io_dtype)
            if use_2cta_instrs:
                args.extend(["cutlass.Int32(0)", "flash_mma_tile_coord_v"])
            return f"_helion_flash_rt.{name}(" + ", ".join(args) + ")"

        pass2_call = _format_disc_pass2(_disc_pass2_name, causal=False)
        pass2_causal_call = _format_disc_pass2(_disc_pass2_causal_name, causal=True)
        zero_pass2_call = (
            "_helion_flash_rt.fa4_disc_zero_store("
            + ", ".join(
                [
                    st,
                    stt,
                    "tSTcS",
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}" if split_p_arrive else "None",
                    "flash_P_STORE_SPLIT",
                    "flash_P_STORE_CHUNKS",
                    *(["cutlass.Int32(0)"] if use_2cta_instrs else []),
                ]
            )
            + ")"
        )
        if has_lse:
            lse_store = _flash_lse_store(
                f"flash_m_tile{stage} * 128 + flash_local_tidx",
                "flash_bh",
                "        ",
                lse_scale=score_plan.lse_scale,
            )
        else:
            lse_store = ""
        corr_empty_ptr = f"flash_s{stage}_corr_empty_ptr"
        corr_prod_index = "0" if not is_causal else "flash_s_corr_prod_index"
        corr_prod_advance = (
            "flash_s_corr_prod_phase ^= 1"
            if not is_causal
            else textwrap.dedent(
                """\
                flash_s_corr_prod_index ^= 1
                if flash_s_corr_prod_index == 0:
                    flash_s_corr_prod_phase ^= 1
                """
            ).strip()
        )

        def _corr_publish_alpha(indent: str) -> str:
            if cfg.skip_rescale_stats:
                return ""
            publish_alpha_store = f"{indent}flash_scale_t[{corr_prod_index}, {stage}, flash_local_tidx] = flash_alpha\n"
            publish_alpha_advance = corr_prod_advance.replace("\n", f"\n{indent}")
            return f"""{indent}_helion_flash_rt.mbar_spin_wait(
{indent}    {corr_empty_ptr} + {corr_prod_index}, flash_s_corr_prod_phase)
{publish_alpha_store.rstrip()}
{indent}cute.arch.barrier_arrive(
{indent}    barrier_id={3 + int(stage) * 4} + warp_idx % 4, number_of_threads=64)
{indent}{publish_alpha_advance}"""

        corr_publish_alpha = _corr_publish_alpha("                ")
        corr_rowsum_advance = corr_prod_advance.replace("\n", "\n        ")
        corr_publish_rowsum = f"""        _helion_flash_rt.mbar_spin_wait(
            {corr_empty_ptr} + {corr_prod_index}, flash_s_corr_prod_phase)
        flash_scale_t[{corr_prod_index}, {stage}, flash_local_tidx] = flash_row_sum
        cute.arch.barrier_arrive(
            barrier_id={3 + int(stage) * 4} + warp_idx % 4, number_of_threads=64)
{lse_store}
        {corr_rowsum_advance}"""
        # The correction warp consumes per-KV alpha handoffs during the loop and
        # one final row-sum handoff after the loop.
        final_corr_publish_rowsum = corr_publish_rowsum
        if cfg.softmax_disc:
            if not _flash_fa4_runtime_disc_score_plan_supported(score_plan):
                score_transform = _flash_score_transform_block(
                    score_plan,
                    indent="                ",
                    score_tensor="flash_disc_frg",
                    coord_tensor="tLDcS[None, flash_ci, None, None]",
                    bh_expr="flash_bh",
                    m_tile_expr=f"flash_m_tile{stage}",
                    kv_tile_expr="flash_kv",
                    chunk_expr="flash_ci",
                    io_dtype=io_dtype,
                )
                rowmax_block = f"""            flash_ld_shape = tLDcS[None, 0, None, None].shape
            for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):
                flash_disc_frg = cute.make_rmem_tensor(flash_ld_shape, cutlass.Float32)
                cute.copy({ld}, {ldt}[None, flash_ci, None, None], flash_disc_frg){score_transform}
                flash_row_max = _helion_flash_rt._fmax_reduce_chunk(flash_disc_frg, flash_row_max)
                cute.copy({score_st}, flash_disc_frg, {score_stt}[None, flash_ci, None, None])
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_view_async_tmem_store()"""
                pass2_block = f"            flash_p_sum = {pass2_call}"
                alpha_publish = (
                    ""
                    if cfg.skip_rescale_stats
                    else f"""            if {softmax_not_first}:
{corr_publish_alpha}"""
                )
                return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
        for {softmax_loop_var} in cutlass.range({kv_loop_bound}, unroll=1):{softmax_actual_kv}
            _helion_flash_rt.mbar_spin_wait(flash_s_full_ptr + {stage}, flash_s_full_phase)
            flash_s_full_phase ^= 1
            flash_old_row_max = flash_row_max
{rowmax_block}
            flash_row_max_safe = flash_row_max
            if flash_row_max == -cutlass.Float32.inf:
                flash_row_max_safe = cutlass.Float32(0.0)
{_disc_alpha_block}
{alpha_publish}
{pass2_block}
            flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum
{final_corr_publish_rowsum}"""
            rowmax_dense_call = (
                f"_helion_flash_rt.fa4_disc_rowmax("
                f"{ld}, {ldt}, tLDcS, flash_row_max, flash_LD_CHUNKS)"
            )
            direct_dense_rowmax = f"            flash_row_max = {rowmax_dense_call}"
            direct_dense_pass2 = f"            flash_p_sum = {pass2_call}"

            def _format_disc_loop(
                loop_var: str,
                loop_bound: str,
                actual_kv: str,
                not_first: str,
                loop_rowmax_block: str,
                loop_pass2_block: str,
                *,
                publish_rowsum: bool = True,
                zero_first_tile: bool = False,
            ) -> str:
                rowsum_publish = final_corr_publish_rowsum if publish_rowsum else ""
                alpha_publish = (
                    ""
                    if cfg.skip_rescale_stats
                    else f"""            if {not_first}:
{corr_publish_alpha}"""
                )
                if zero_first_tile:
                    loop_rowmax_block = f"""            if {not_first}:
{textwrap.indent(loop_rowmax_block, "    ")}"""
                    loop_pass2_block = f"""            if {not_first}:
{textwrap.indent(loop_pass2_block, "    ")}
            else:
                flash_p_sum = {zero_pass2_call}"""
                return f"""        for {loop_var} in cutlass.range({loop_bound}, unroll=1):{actual_kv}
            _helion_flash_rt.mbar_spin_wait(flash_s_full_ptr + {stage}, flash_s_full_phase)
            flash_s_full_phase ^= 1
            flash_old_row_max = flash_row_max
{loop_rowmax_block}
            flash_row_max_safe = flash_row_max
            if flash_row_max == -cutlass.Float32.inf:
                flash_row_max_safe = cutlass.Float32(0.0)
{_disc_alpha_block_for(not_first)}
{alpha_publish}
            flash_p_sum = cutlass.Float32(0.0)
{loop_pass2_block}
            flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum
{rowsum_publish}"""

            if is_causal:
                rowmax_causal_call = (
                    f"_helion_flash_rt.fa4_disc_rowmax_causal("
                    f"{ld}, {ldt}, tLDcS, flash_row_max, flash_LD_CHUNKS, "
                    f"flash_m_tile{stage}, flash_kv)"
                )
                direct_causal_rowmax = (
                    f"            flash_row_max = {rowmax_causal_call}"
                )
                direct_causal_pass2 = f"            flash_p_sum = {pass2_causal_call}"
                if causal_desc_kv and cfg.causal_loop_split:
                    masked_loop = _format_disc_loop(
                        "flash_kv_mask_iter",
                        f"{kv_loop_bound} - flash_m_tile{stage}",
                        (
                            f"\n            flash_kv = {kv_loop_bound} - "
                            "cutlass.Int32(1) - flash_kv_mask_iter"
                        ),
                        "flash_kv_mask_iter != 0",
                        direct_causal_rowmax,
                        direct_causal_pass2,
                        publish_rowsum=False,
                        zero_first_tile=stage == "0",
                    )
                    unmasked_loop = _format_disc_loop(
                        "flash_kv_unmask_iter",
                        f"flash_m_tile{stage}",
                        (
                            f"\n            flash_kv = flash_m_tile{stage} - "
                            "cutlass.Int32(1) - flash_kv_unmask_iter"
                        ),
                        "flash_kv_unmask_iter >= cutlass.Int32(0)",
                        direct_dense_rowmax,
                        direct_dense_pass2,
                    )
                    return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{masked_loop}
{unmasked_loop}"""
                rowmax_block = f"""            if flash_kv >= flash_m_tile{stage}:
                flash_row_max = {rowmax_causal_call}
            else:
                flash_row_max = {rowmax_dense_call}"""
                pass2_block = f"""            if flash_kv >= flash_m_tile{stage}:
                flash_p_sum = {pass2_causal_call}
            else:
                flash_p_sum = {pass2_call}"""
            else:
                rowmax_block = direct_dense_rowmax
                pass2_block = direct_dense_pass2
            # CHUNKED-t2r ("disc") body: the spike's PROVEN zero-spill structure.
            # PASS1 t2r's each 32-elem chunk to find the row-max (never the full
            # row resident); alpha + the s_corr handoff are computed between the
            # passes (spike disc ordering); PASS2 t2r's each chunk again, does
            # scale-subtract + exp2(split) + fp16 convert + r2t-store + row-sum
            # fold + the staged-P pfor/pfor2 arrives, freeing each chunk before the
            # next. Peak live = ONE 32-elem fragment -> fits the 200-grant zero-spill.
            loop = _format_disc_loop(
                softmax_loop_var,
                kv_loop_bound,
                softmax_actual_kv,
                softmax_not_first,
                rowmax_block,
                pass2_block,
            )
            return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{loop}"""
        score_transform = _flash_score_transform_block(
            score_plan,
            indent="            ",
            score_tensor="tLDrS",
            coord_tensor="tLDcS",
            bh_expr="flash_bh",
            m_tile_expr=f"flash_m_tile{stage}",
            kv_tile_expr="flash_kv",
        )
        if split_p_arrive:
            p_store_block = f"""            for flash_ci in cutlass.range_constexpr(flash_P_STORE_SPLIT):
                cute.copy({st}, tSTrS[None, None, flash_ci], {
                stt
            }[None, None, flash_ci])
            cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(
                flash_pfor_ptr + {stage}{pfor_peer_arg})
            for flash_ci in cutlass.range_constexpr(flash_P_STORE_SPLIT, flash_P_STORE_CHUNKS):
                cute.copy({st}, tSTrS[None, None, flash_ci], {
                stt
            }[None, None, flash_ci])
            cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(
                flash_pfor2_ptr + {stage}{pfor_peer_arg})"""
        else:
            p_store_block = f"""            for flash_ci in cutlass.range_constexpr(flash_P_STORE_CHUNKS):
                cute.copy({st}, tSTrS[None, None, flash_ci], {
                stt
            }[None, None, flash_ci])
            cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(
                flash_pfor_ptr + {stage}{pfor_peer_arg})"""
        if cfg.exp2_impl == "split":
            if mixed_p_store:
                sp_pass2_name = (
                    "fa4_sp_exp_convert_store_rep32_split_whole_rowsum"
                    if sp_whole_row_sum
                    else "fa4_sp_exp_convert_store_rep32_split"
                )
                sp_pass2_args = [
                    "tLDrS",
                    f"flash_tiled_st32_{stage}",
                    f"tST32tS{stage}",
                    "tST32cS",
                    st,
                    stt,
                    "tSTcS",
                    "_flash_scale_log2",
                    "flash_minus_max_scale",
                    str(cfg.e2e_freq),
                    str(cfg.e2e_res),
                    str(cfg.e2e_offset0 if stage == "0" else cfg.e2e_offset),
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}",
                    io_dtype,
                ]
            else:
                sp_pass2_name = (
                    "fa4_sp_exp_convert_store_whole_rowsum"
                    if sp_whole_row_sum
                    else "fa4_sp_exp_convert_store"
                )
                sp_pass2_args = [
                    "tLDrS",
                    st,
                    stt,
                    "tSTcS",
                    "_flash_scale_log2",
                    "flash_minus_max_scale",
                    str(cfg.e2e_freq),
                    str(cfg.e2e_res),
                    str(cfg.e2e_offset0 if stage == "0" else cfg.e2e_offset),
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}" if split_p_arrive else "None",
                    "flash_P_STORE_SPLIT",
                    "flash_P_STORE_CHUNKS",
                    io_dtype,
                ]
            if use_2cta_instrs:
                sp_pass2_args.extend(["cutlass.Int32(0)", "flash_mma_tile_coord_v"])
            sp_exp_block = (
                f"            flash_p_sum = _helion_flash_rt.{sp_pass2_name}("
                + ", ".join(sp_pass2_args)
                + ")"
            )
            sp_p_store_block = ""
            sp_corr_publish_alpha = ""
        else:
            sp_exp_block = softmax_exp_block
            sp_p_store_block = p_store_block
            sp_corr_publish_alpha = ""
        if not sp_corr_publish_alpha and not cfg.skip_rescale_stats:
            sp_corr_publish_alpha = f"""            if {softmax_not_first}:
{corr_publish_alpha}"""
        return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
        for {softmax_loop_var} in cutlass.range({kv_loop_bound}, unroll=1):{
            softmax_actual_kv
        }
            _helion_flash_rt.mbar_spin_wait(flash_s_full_ptr + {
            stage
        }, flash_s_full_phase)
            flash_s_full_phase ^= 1
            flash_old_row_max = flash_row_max
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            cute.copy({ld}, {ldt}, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}
            flash_row_max = _helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)
            flash_row_max_safe = flash_row_max
            if flash_row_max == -cutlass.Float32.inf:
                flash_row_max_safe = cutlass.Float32(0.0)
{_sp_alpha_pre}
{sp_exp_block}
{_sp_alpha_post}
{sp_corr_publish_alpha}
            flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum
{sp_p_store_block}
{final_corr_publish_rowsum}"""

    softmax0_setup = (
        _tmem_softmax_setup_stage("0")
        if stage_local_softmax_setup
        else tmem_softmax_setup
    )
    softmax1_setup = (
        _tmem_softmax_setup_stage("1")
        if stage_local_softmax_setup
        else tmem_softmax_setup
    )
    softmax0_head = f"""    cute.arch.setmaxregister_increase({cfg.softmax_regs})
{softmax0_setup.rstrip()}
    flash_s_full_phase = cutlass.Int32(0)
    flash_s_corr_prod_index = cutlass.Int32(0)
    flash_s_corr_prod_phase = cutlass.Int32(0)"""
    softmax1_head = f"""    cute.arch.setmaxregister_increase({cfg.softmax_regs})
{softmax1_setup.rstrip()}
    flash_s_full_phase = cutlass.Int32(0)
    flash_s_corr_prod_index = cutlass.Int32(0)
    flash_s_corr_prod_phase = cutlass.Int32(0)"""
    softmax_needs_tile_decode = (
        has_lse
        or is_causal
        or any(
            modifier.kind
            in {
                TENSOR_BIAS_KIND,
                RELATIVE_BIAS_KIND,
                ALIBI_BIAS_KIND,
            }
            for modifier in score_plan.modifiers
        )
    )
    softmax_prelude = "decode" if softmax_needs_tile_decode else "none"
    tmem_dealloc_arrive = (
        "    flash_tmem_user_bar.arrive()"
        if use_2cta_instrs
        else "    cute.arch.mbarrier_arrive(flash_tmem_dealloc_ptr)"
    )
    softmax0_block = _flash_fa4_wrap(
        "if warp_idx < 4:",
        softmax0_head,
        _softmax_inner(
            "0",
            "flash_tiled_ld0",
            "tLDtS0",
            "flash_tiled_st0",
            "tSTtS0",
            "flash_tiled_score_st0",
            "tScoreSTtS0",
        ),
        persistent,
        prelude=softmax_prelude,
        tail=tmem_dealloc_arrive,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )
    softmax1_block = _flash_fa4_wrap(
        "if (warp_idx >= 4) & (warp_idx < 8):",
        softmax1_head,
        _softmax_inner(
            "1",
            "flash_tiled_ld1",
            "tLDtS1",
            "flash_tiled_st1",
            "tSTtS1",
            "flash_tiled_score_st1",
            "tScoreSTtS1",
        ),
        persistent,
        prelude=softmax_prelude,
        tail=tmem_dealloc_arrive,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )

    # CORRECTION (8-11): pre-arrive pfor; steady rescale O[k] by alpha (conditional
    # warp-uniform vote); final epilogue divide-by-rowsum + store. flash_o_full_phase
    # carries across work-items (head); the inner waits o_full on that phase and
    # flips it once per work-item (spike L2043/2065/2069/2072).
    corr_pfor_prearrive = (
        ""
        if cfg.skip_rescale_stats
        else f"""
    _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0{pfor_peer_arg})
    _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1{pfor_peer_arg})"""
    )
    corr_head = f"""    cute.arch.setmaxregister_decrease({cfg.corr_regs})
{tmem_base_setup.rstrip()}
{corr_epi_smem_setup.rstrip()}
    flash_o_full_phase = cutlass.Int32(0)
    flash_s_corr_cons_index = cutlass.Int32(0)
    flash_s_corr_cons_phase = cutlass.Int32(0)
    flash_corr_epi_empty_phase = cutlass.Int32(0){corr_pfor_prearrive}
    cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)
    cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 1)
    cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)
    cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 1)"""
    if not epi_smem:
        corr_head += f"""
    flash_gO_corr = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))
    flash_epi_tiler0 = ((cute.size(tOtO0, mode=[0, 0]), cute.size(tOtO0, mode=[0, 1])),)
    flash_epi_tiler1 = ((cute.size(tOtO1, mode=[0, 0]), cute.size(tOtO1, mode=[0, 1])),)
    tOtO_epi0 = cute.zipped_divide(tOtO0, flash_epi_tiler0)
    tOtO_epi1 = cute.zipped_divide(tOtO1, flash_epi_tiler1)
    flash_o_ld_atom0 = cute.make_copy_atom(
        cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
    flash_o_ld_atom1 = cute.make_copy_atom(
        cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
    flash_tiled_o_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom0, tOtO_epi0[None, 0])
    flash_tiled_o_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom1, tOtO_epi1[None, 0])
    flash_thr_o_ld0 = flash_tiled_o_ld0.get_slice(flash_local_tidx)
    flash_thr_o_ld1 = flash_tiled_o_ld1.get_slice(flash_local_tidx)
    tDtO0 = flash_thr_o_ld0.partition_S(tOtO_epi0)
    tDtO1 = flash_thr_o_ld1.partition_S(tOtO_epi1)"""

    def _corr_epi(stage: str, mtile: str) -> str:
        corr_cons_index = "0" if not is_causal else "flash_s_corr_cons_index"
        scale_expr = f"flash_scale_t[{corr_cons_index}, {stage}, flash_local_tidx]"
        if not epi_smem:
            # Committed path: per-thread t2r (Ld32x32 Rep16) -> rescale -> cast ->
            # STG.E.128 straight to gmem (coord->linear address division per thread =
            # the epilogue IMAD/MOV overhead).
            return f"""        cute.arch.barrier(
            barrier_id={3 + int(stage) * 4} + warp_idx % 4, number_of_threads=64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
        cute.arch.mbarrier_arrive(flash_s{stage}_corr_empty_ptr + {corr_cons_index})
        _helion_flash_rt.mbar_spin_wait(flash_o_full_ptr + {stage}, flash_o_full_phase)
        tOgO_mma{stage} = {corr_gmem_o_index.format(mtile=mtile)}
        gO_epi{stage} = cute.zipped_divide(tOgO_mma{stage}, flash_epi_tiler{stage})
        tDgO{stage} = flash_thr_o_ld{stage}.partition_D(gO_epi{stage})
        for flash_i in cutlass.range(cute.size(tDtO{stage}, mode=[2])):
            flash_reg{stage} = cute.make_rmem_tensor(tDgO{stage}[None, None, 0].shape, cutlass.Float32)
            flash_rego{stage} = cute.make_rmem_tensor(tDgO{stage}[None, None, 0].shape, {io_dtype})
            cute.copy(flash_tiled_o_ld{stage}, tDtO{stage}[None, None, flash_i], flash_reg{stage})
            _helion_flash_rt._scale_fragment_packed_f32x2(
                flash_reg{stage}, flash_inv_sum{stage})
            flash_rego{stage}.store(flash_reg{stage}.load().to({io_dtype}))
            cute.autovec_copy(flash_rego{stage}, tDgO{stage}[None, None, flash_i])"""
        # FAITHFUL FA4 correction_epilogue port. Per corr-chunk: t2r 16 O cols ->
        # packed mul by inv_sum -> cast fp16 -> r2s into the sO smem buffer (stage
        # {stage} of the 2-staged _flash_osl) via the SMEM-STORE atom matched to
        # the t2r tiled copy. A dedicated epilogue warp then drains sO either by
        # TMA-O or by vector STG.
        return f"""        cute.arch.barrier(
            barrier_id={3 + int(stage) * 4} + warp_idx % 4, number_of_threads=64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
        cute.arch.mbarrier_arrive(flash_s{stage}_corr_empty_ptr + {corr_cons_index})
        _helion_flash_rt.mbar_spin_wait(flash_o_full_ptr + {stage}, flash_o_full_phase)
        _helion_flash_rt.mbar_spin_wait(
            flash_corr_epi_empty_ptr + {stage}, flash_corr_epi_empty_phase)
        for flash_i in cutlass.range_constexpr(flash_o_corr_chunks):
            flash_oreg{stage} = cute.make_rmem_tensor(tOcO_corr_t2r[None, 0, 0, flash_i].shape, cutlass.Float32)
            cute.copy(flash_o_tiled_t2r, tOtO{stage}_corr_t2r[None, 0, 0, flash_i], flash_oreg{stage})
            flash_oreg{stage}.store(flash_oreg{stage}.load() * flash_inv_sum{stage})
            _helion_flash_rt.cvt_copy(flash_o_tiled_r2s, flash_oreg{stage}, tOsO{stage}_corr_r2s[None, 0, 0, flash_i])
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr + {stage})"""

    corr_epi_empty_toggle = (
        "        flash_corr_epi_empty_phase ^= 1" if epi_smem else ""
    )
    corr_cons_index = "0" if not is_causal else "flash_s_corr_cons_index"
    corr_empty0_early = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s0_corr_empty_ptr + {corr_cons_index})"
        if not is_causal
        else ""
    )
    corr_empty1_early = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s1_corr_empty_ptr + {corr_cons_index})"
        if not is_causal
        else ""
    )
    corr_empty0_late = (
        ""
        if not is_causal
        else "            cute.arch.mbarrier_arrive("
        "flash_s0_corr_empty_ptr + flash_s_corr_cons_index)"
    )
    corr_empty1_late = (
        ""
        if not is_causal
        else "            cute.arch.mbarrier_arrive("
        "flash_s1_corr_empty_ptr + flash_s_corr_cons_index)"
    )
    corr_stage0 = f"""            cute.arch.barrier(
                barrier_id=3 + warp_idx % 4, number_of_threads=64)
            flash_a0 = flash_scale_t[{corr_cons_index}, 0, flash_local_tidx]
{corr_empty0_early}
            flash_need_rescale0 = cute.arch.vote_ballot_sync(flash_a0 < 1.0) != 0
            if flash_need_rescale0:
                _helion_flash_rt.rescale_o_tmem(
                    tOtO0, flash_a0, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0{pfor_peer_arg})
{corr_empty0_late}"""
    corr_stage1 = f"""            cute.arch.barrier(
                barrier_id=7 + warp_idx % 4, number_of_threads=64)
            flash_a1 = flash_scale_t[{corr_cons_index}, 1, flash_local_tidx]
{corr_empty1_early}
            flash_need_rescale1 = cute.arch.vote_ballot_sync(flash_a1 < 1.0) != 0
            if flash_need_rescale1:
                _helion_flash_rt.rescale_o_tmem(
                    tOtO1, flash_a1, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1{pfor_peer_arg})
{corr_empty1_late}"""
    if cfg.skip_rescale_stats:
        corr_inner = f"""        # Final: divide by row_sum, cast, store (waits MMA's last-tile O_full).
{_corr_epi("0", "flash_m_tile0")}
{_corr_epi("1", "flash_m_tile1")}
{corr_epi_empty_toggle}
        flash_s_corr_cons_index ^= 1
        if flash_s_corr_cons_index == 0:
            flash_s_corr_cons_phase ^= 1
        flash_o_full_phase ^= 1"""
    else:
        corr_inner = f"""        for flash_kv in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
{corr_stage0}
{corr_stage1}
            flash_s_corr_cons_index ^= 1
            if flash_s_corr_cons_index == 0:
                flash_s_corr_cons_phase ^= 1
        # Final: divide by row_sum, cast, store (waits MMA's last-tile O_full).
{_corr_epi("0", "flash_m_tile0")}
        _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0{pfor_peer_arg})
{_corr_epi("1", "flash_m_tile1")}
        _helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1{pfor_peer_arg})
{corr_epi_empty_toggle}
        flash_s_corr_cons_index ^= 1
        if flash_s_corr_cons_index == 0:
            flash_s_corr_cons_phase ^= 1
        flash_o_full_phase ^= 1"""
    corr_block = _flash_fa4_wrap(
        "if (warp_idx >= 8) & (warp_idx < 12):",
        corr_head,
        corr_inner,
        persistent,
        prelude="decode",
        tail=tmem_dealloc_arrive,
        total_tiles=total_tiles,
        num_m_pairs=num_m_pairs,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        clc_heads_per_batch=clc_heads_per_batch,
        tensor_4d_heads=tensor_4d_heads,
        recompute_tile_coords=cfg.recompute_tile_coords,
    )

    src = (
        setup
        + epi_smem_setup
        + ("" if local_epi_tma_setup else epi_tma_setup)
        + epi_stg_setup
        + empty_block
        + load_block
        + mma_block
        + epi_block
        + softmax0_block
        + softmax1_block
        + corr_block
    )
    return list(ast.parse(src).body)


def emit_flash_device_body(
    df: DeviceFunction,
    *,
    head_dim: int,
    num_kv: int,
    cfg: FlashAttentionConfig,
    has_lse: bool,
    io_dtype: str,
    score_plan: AttentionScorePlan,
) -> list[ast.stmt]:
    """Build the spike kernel device body, adapted for Helion's 3D tensors.

    Helion passes the reshaped 3D views ``(batch*heads, seqlen, head_dim)``; the
    grid is a flat 1D launch where ``block_idx % num_bh`` selects the (batch,
    head) slice and ``block_idx // num_bh`` selects the query 128-row tile. The
    host wrapper (``kind == "helion_flash"``) reorders the buffers to the
    reference (S, D, B)/(D, S, B) layouts and builds the two tiled_mma
    (``_flash_qk_mma`` from SMEM, ``_flash_pv_mma`` with OperandSource.TMEM), the
    three TMA atoms (``_flash_tma_q/_flash_mQt`` ...), the SMEM layouts
    (``_flash_qsl`` ...), and the scalars (``_flash_scale_log2``,
    ``_flash_num_kv_tiles``, ``_flash_num_bh``), passing them all as
    wrapper-only kernel params.

    With ``kv_stage > 1`` K/V live in a multi-stage TMA ring (Stage 3): warp 0
    prefetches the first ``kv_stage`` K/V tiles in a prologue, then issues each
    future tile's load as soon as its ring slot is released, so the K/V load
    latency overlaps the current tile's MMA + softmax, reducing the dominant
    ``long_scoreboard`` stall.
    """
    hd = head_dim
    kv_stage = cfg.kv_stage
    flash_prologue = min(kv_stage, num_kv)
    # Phase split: PREFETCH iters each issue the K/V load {kv_stage} tiles ahead;
    # DRAIN iters (the last {flash_prologue}) issue nothing (ring already filled).
    prefetch_iters = max(num_kv - kv_stage, 0)
    drain_iters = num_kv - prefetch_iters
    loop_body = _flash_kv_loop_body(
        hd=hd,
        kv_stage=kv_stage,
        prefetch_iters=prefetch_iters,
        drain_iters=drain_iters,
        io_dtype=io_dtype,
        score_plan=score_plan,
        rescale_threshold=cfg.rescale_threshold,
        packed_reduce=cfg.packed_reduce,
        exp2_impl=cfg.exp2_impl,
        e2e_freq=cfg.e2e_freq,
        e2e_res=cfg.e2e_res,
    )
    lse_store = (
        _flash_lse_store(
            "flash_m_tile * 128 + tidx",
            "flash_bh",
            "",
            lse_scale=score_plan.lse_scale,
        )
        if has_lse
        else ""
    )
    src = f"""
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
flash_pid = cutlass.Int32(cute.arch.block_idx()[0])
flash_bh = flash_pid % _flash_num_bh
flash_m_tile = flash_pid // _flash_num_bh

_flash_storage_cls = _helion_flash_rt.flash_shared_storage({hd}, {kv_stage}, dtype={io_dtype})
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)
sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner)

flash_tmem_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=1, num_threads=128)
flash_tmem = cutlass_utils_flash.TmemAllocator(
    storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_bar)
flash_tmem.allocate(512)
if warp_idx == 0:
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_q)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_k)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_v)

# Per-tile TMA byte counts: select the 3 spatial (MMA, MMA_K, MMA_D) modes,
# dropping the trailing stage mode, so the tx_count is ONE ring slot regardless
# of kv_stage (matches the reference fmha.py k_copy_size).
flash_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2]))
flash_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2]))
flash_v_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2]))
flash_q_prod, flash_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    tx_count=flash_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr()).make_participants()
flash_k_prod, flash_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr()).make_participants()
flash_v_prod, flash_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_v_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr()).make_participants()
flash_mma_s_prod, flash_mma_s_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_s_mbar_ptr.data_ptr()).make_participants()
flash_mma_o_prod, flash_mma_o_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_o_mbar_ptr.data_ptr()).make_participants()
flash_acc_prod, flash_acc_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

flash_qkt = _flash_qk_mma.get_slice(0)
flash_pvt = _flash_pv_mma.get_slice(0)
tSrQ = flash_qkt.make_fragment_A(sQ)
tSrK = flash_qkt.make_fragment_B(sK)
tOrV = flash_pvt.make_fragment_B(sV)
flash_qk_acc_shape = flash_qkt.partition_shape_C((128, 128))
tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
flash_pv_acc_shape = flash_pvt.partition_shape_C((128, {hd}))
tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)

# Per-CTA gmem views: flat_divide tiles only the (s, d) modes (reference
# pattern), leaving the (batch, head) as a trailing loop mode. The batch coord
# (flash_bh) is applied AFTER tma_partition; the query 128-row tile index
# (flash_m_tile) indexes the resulting per-CTA tensor.
gQ = cute.flat_divide(_flash_mQt, cute.select((128, 128, {hd}), mode=[0, 2]))
gK = cute.flat_divide(_flash_mKt, cute.select((128, 128, {hd}), mode=[1, 2]))
gV = cute.flat_divide(_flash_mVt, cute.select((128, {hd}, 128), mode=[1, 2]))
tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tOgV = flash_pvt.partition_B(gV)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
# K/V smem are multi-stage rings: the staged smem layout is (MMA, MMA_K, MMA_D,
# PIPE) -- group the 3 spatial modes (0..3), leaving the stage (PIPE) as the
# trailing indexable mode (tKsK[None, idx]). Matches the reference fmha.py.
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]
tVgV = tVgV_dkl[None, 0, None, flash_bh]

flash_tmem.wait_for_alloc()
flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
tStS = cute.make_tensor(flash_tmem_ptr, tStS.layout)
tOtO = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)
tP = cute.make_tensor(tStS.iterator, _flash_ptl.outer)
tOrP = flash_pvt.make_fragment_A(tP)

cS = cute.make_identity_tensor((128, 128))
tScS = flash_qkt.partition_C(cS)
flash_ld_atom = cute.make_copy_atom(
    cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
flash_tiled_ld = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS)
flash_thr_ld = flash_tiled_ld.get_slice(tidx)
tLDtS = flash_thr_ld.partition_S(tStS)
tLDcS = flash_thr_ld.partition_D(tScS)

flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width
flash_tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
tStS_P = cute.make_tensor(tStS.iterator, flash_tStS_P_layout)
flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, flash_tilePlikeFP32)))
tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
flash_st_atom = cute.make_copy_atom(
    cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
flash_tiled_st = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P)
flash_thr_st = flash_tiled_st.get_slice(tidx)
tSTtS = flash_thr_st.partition_D(tStS_P)
tSTcS = flash_thr_st.partition_S(tScS_P)

flash_acc_empty = flash_acc_prod.acquire_and_advance()

if warp_idx == 0:
    flash_q_empty = flash_q_prod.acquire_and_advance()
    cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile], tQsQ[None, flash_q_empty.index],
              tma_bar_ptr=flash_q_empty.barrier)
flash_q_full = flash_q_cons.wait_and_advance()

flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
flash_row_sum = cutlass.Float32(0.0)

# Stage 3 multi-stage TMA ring: warp 0 prefetches the first {flash_prologue} K/V
# tiles into the ring (prologue) so their loads overlap the first iterations'
# compute. The KV loop is split into a PREFETCH phase ({prefetch_iters} iters,
# each issues the load for the tile {kv_stage} ahead) and a DRAIN phase
# (remaining iters, no issue). num_kv and kv_stage are compile-time constants, so
# each phase's producer-advance count is statically uniform -- the pipeline state
# stays balanced (a data-dependent prefetch branch makes the CuTe pipeline
# fixed-point analysis diverge / hang the JIT).
if warp_idx == 0:
    for flash_pf in cutlass.range_constexpr({flash_prologue}):
        flash_k_empty = flash_k_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, flash_pf], tKsK[None, flash_k_empty.index],
                  tma_bar_ptr=flash_k_empty.barrier)
        flash_v_empty = flash_v_prod.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, flash_pf], tVsV[None, flash_v_empty.index],
                  tma_bar_ptr=flash_v_empty.barrier)

{loop_body}

if warp_idx == 0:
    flash_acc_empty.commit()
flash_q_full.release()

flash_tmem.relinquish_alloc_permit()
flash_acc_full = flash_acc_cons.wait_and_advance()
flash_o_full = flash_mma_o_cons.wait_and_advance()

flash_inv_sum = cutlass.Float32(1.0) / flash_row_sum
gO = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))
tOgO_mma = flash_pvt.partition_C(gO)[None, None, None, flash_m_tile, 0, flash_bh]
flash_epi_tiler = ((cute.size(tOtO, mode=[0, 0]), cute.size(tOtO, mode=[0, 1])),)
tOtO_epi = cute.zipped_divide(tOtO, flash_epi_tiler)
gO_epi = cute.zipped_divide(tOgO_mma, flash_epi_tiler)
flash_o_ld_atom = cute.make_copy_atom(
    cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition.x64), cutlass.Float32)
flash_tiled_o_ld = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom, tOtO_epi[None, 0])
flash_thr_o_ld = flash_tiled_o_ld.get_slice(tidx)
tDtO = flash_thr_o_ld.partition_S(tOtO_epi)
tDgO = flash_thr_o_ld.partition_D(gO_epi)
flash_reg = cute.make_rmem_tensor(tDgO[None, None, 0].shape, cutlass.Float32)
flash_rego = cute.make_rmem_tensor(tDgO[None, None, 0].shape, {io_dtype})
for flash_i in cutlass.range(cute.size(tDtO, mode=[2])):
    cute.copy(flash_tiled_o_ld, tDtO[None, None, flash_i], flash_reg)
    flash_reg.store(flash_reg.load() * flash_inv_sum)
    flash_rego.store(flash_reg.load().to({io_dtype}))
    cute.autovec_copy(flash_rego, tDgO[None, None, flash_i])
{lse_store}
flash_o_full.release()
flash_acc_full.release()

cutlass_pipeline_flash.sync(barrier_id=1)
flash_tmem.free(flash_tmem_ptr)
"""
    return list(ast.parse(src).body)


# Kernel param order MUST match the ``call_args`` order appended by the
# ``helion_flash`` wrapper plan in ``runtime._append_cute_wrapper_plan``.
_FLASH_KERNEL_PARAMS = [
    "_flash_qk_mma",
    "_flash_pv_mma",
    "_flash_tma_q",
    "_flash_mQt",
    "_flash_tma_k",
    "_flash_mKt",
    "_flash_tma_v",
    "_flash_mVt",
    "_flash_mOt",
    "_flash_qsl",
    "_flash_ksl",
    "_flash_vsl",
    "_flash_ptl",
    "_flash_scale_log2",
    "_flash_num_kv_tiles",
]

_FLASH_DYNAMIC_TILE_KERNEL_PARAMS = [
    "_flash_num_bh",
    "_flash_total_tiles",
]

# Lever A (epilogue TMA-store): the extra wrapper-built kernel params -- the O TMA
# STORE atom and its smem layout. Only appended when cfg.epi_tma (fa4-only). The
# device epilogue stages rescaled-fp16 O into the sQ smem region via _flash_osl,
# then cute.copy(_flash_tma_o, sO, gO) does the cp.async.bulk.tensor S2G.
_FLASH_EPI_TMA_KERNEL_PARAMS = [
    "_flash_tma_o",
    "_flash_osl",
]

_FLASH_EPI_STG_KERNEL_PARAMS = [
    "_flash_osl",
]

_FLASH_LSE_KERNEL_PARAMS = [
    "_flash_mLSE",
]

_FLASH_BIAS_KERNEL_PARAMS = [
    "_flash_mBias",
    "_flash_score_bias_scale",
]

_FLASH_ALIBI_KERNEL_PARAMS = [
    "_flash_mAlibi",
    "_flash_num_alibi",
]

_FLASH_DOCUMENT_KERNEL_PARAMS = [
    "_flash_mDoc",
    "_flash_doc_heads_per_batch",
]


class FlashTensorPlan(NamedTuple):
    """Resolved tensor operands + static shape for the flash-attention path."""

    q_arg: TensorArg
    k_arg: TensorArg
    v_arg: TensorArg
    o_arg: TensorArg
    lse_arg: TensorArg | None
    lse_log_base: str | None
    bias_args: tuple[TensorArg, ...]
    alibi_args: tuple[TensorArg, ...]
    document_args: tuple[TensorArg, ...]
    batch: int
    seq: int
    head_dim: int
    dtype: torch.dtype
    tensor_4d_batch: int
    tensor_4d_heads: int


def _flash_4d_base_batch_heads(
    arg: TensorArg,
    *,
    batch: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[int, int] | None:
    """Return ``(z, h)`` when a flattened 3D flash view came from ``[z,h,s,d]``."""
    value = arg.fake_value
    base = value._base
    if base is None:
        return None
    if (
        base.ndim != 4
        or base.dtype != dtype
        or not base.is_contiguous()
        or int(base.shape[2]) != seq
        or int(base.shape[3]) != head_dim
        or int(value.storage_offset()) != 0
        or int(base.storage_offset()) != 0
    ):
        return None
    tensor_4d_batch = int(base.shape[0])
    tensor_4d_heads = int(base.shape[1])
    if tensor_4d_batch <= 0 or tensor_4d_heads <= 0:
        return None
    if tensor_4d_batch * tensor_4d_heads != batch:
        return None
    return tensor_4d_batch, tensor_4d_heads


def _flash_current_block_ids(
    df: DeviceFunction,
) -> tuple[Sequence[int], int] | None:
    block_ids = df.cute_state.attention_flash_block_ids
    if block_ids is None or len(block_ids) != 1:
        return None
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 2:
        return None
    return device_ir.grid_block_ids[0], block_ids[0]


def flash_attention_tensor_plan(df: DeviceFunction) -> FlashTensorPlan | None:
    """Resolve tensor operands for the fused flash-attention codegen.

    Returns the three contiguous fp16/bf16 ``(B, S, D)`` input views (q_view,
    k_view, v_view), the graph-proven canonical output store, an optional
    graph-proven canonical fp32 ``(B, S)`` LSE output, and the static
    batch/seq/head_dim.

    The detector performs the same LSE/aux-output disambiguation using
    ``flash_attention_graph_lse_plan_valid`` because ``DeviceFunction.arguments``
    is still incomplete at detector time.
    """
    block_ids = _flash_current_block_ids(df)
    if block_ids is None:
        return None
    root_block_ids, kv_block_id = block_ids
    score_plan = df.cute_state.attention_flash_score_plan
    graph_plan = _flash_graph_output_plan_from_graphs(
        df.codegen.codegen_graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    if graph_plan is None:
        return None

    tensor_args = [a for a in df.arguments if isinstance(a, TensorArg)]
    tensor_args_by_name = {a.name: a for a in tensor_args}
    q_arg = tensor_args_by_name.get(graph_plan.q_name)
    k_arg = tensor_args_by_name.get(graph_plan.k_name)
    v_arg = tensor_args_by_name.get(graph_plan.v_name)
    if q_arg is None or k_arg is None or v_arg is None:
        return None
    o_arg = tensor_args_by_name.get(graph_plan.o_name)
    if o_arg is None:
        return None
    lse_arg = (
        tensor_args_by_name.get(graph_plan.lse_name)
        if graph_plan.lse_name is not None
        else None
    )
    if graph_plan.lse_name is not None and lse_arg is None:
        return None
    resolved_bias_args: list[TensorArg] = []
    for name in graph_plan.bias_names:
        bias_arg = tensor_args_by_name.get(name)
        if bias_arg is None:
            return None
        resolved_bias_args.append(bias_arg)
    resolved_alibi_args: list[TensorArg] = []
    for name in graph_plan.alibi_names:
        alibi_arg = tensor_args_by_name.get(name)
        if alibi_arg is None:
            return None
        resolved_alibi_args.append(alibi_arg)
    resolved_document_args: list[TensorArg] = []
    for name in graph_plan.document_names:
        document_arg = tensor_args_by_name.get(name)
        if document_arg is None:
            return None
        resolved_document_args.append(document_arg)

    for arg in (q_arg, k_arg, v_arg, o_arg):
        if (
            arg.fake_value.ndim != 3
            or arg.fake_value.dtype != graph_plan.dtype
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[0]) != graph_plan.batch
            or int(arg.fake_value.shape[1]) != graph_plan.seq
            or int(arg.fake_value.shape[2]) != graph_plan.head_dim
        ):
            return None
    for arg in resolved_bias_args:
        if (
            arg.fake_value.ndim != 3
            or arg.fake_value.dtype != graph_plan.dtype
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[0]) != graph_plan.batch
            or int(arg.fake_value.shape[1]) != graph_plan.seq
            or int(arg.fake_value.shape[2]) != graph_plan.seq
        ):
            return None
    for arg in resolved_alibi_args:
        if (
            arg.fake_value.ndim != 1
            or arg.fake_value.dtype != torch.float32
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[0]) <= 0
        ):
            return None
    for arg in resolved_document_args:
        if (
            arg.fake_value.ndim != 2
            or arg.fake_value.dtype not in (torch.int32, torch.int64)
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[0]) <= 0
            or int(arg.fake_value.shape[1]) != graph_plan.seq
        ):
            return None
    if lse_arg is not None:
        if (
            lse_arg.fake_value.ndim != 2
            or lse_arg.fake_value.dtype != torch.float32
            or not lse_arg.fake_value.is_contiguous()
            or int(lse_arg.fake_value.shape[0]) != graph_plan.batch
            or int(lse_arg.fake_value.shape[1]) != graph_plan.seq
        ):
            return None
    tensor_4d_batch = 0
    tensor_4d_heads = 0
    if (
        graph_plan.dtype is torch.float16
        and graph_plan.head_dim == 64
        and not resolved_bias_args
        and not resolved_alibi_args
        and not resolved_document_args
    ):
        q_base_shape = _flash_4d_base_batch_heads(
            q_arg,
            batch=graph_plan.batch,
            seq=graph_plan.seq,
            head_dim=graph_plan.head_dim,
            dtype=graph_plan.dtype,
        )
        if q_base_shape is not None:
            k_base_shape = _flash_4d_base_batch_heads(
                k_arg,
                batch=graph_plan.batch,
                seq=graph_plan.seq,
                head_dim=graph_plan.head_dim,
                dtype=graph_plan.dtype,
            )
            v_base_shape = _flash_4d_base_batch_heads(
                v_arg,
                batch=graph_plan.batch,
                seq=graph_plan.seq,
                head_dim=graph_plan.head_dim,
                dtype=graph_plan.dtype,
            )
            if k_base_shape == q_base_shape and v_base_shape == q_base_shape:
                tensor_4d_batch, tensor_4d_heads = q_base_shape
    return FlashTensorPlan(
        q_arg,
        k_arg,
        v_arg,
        o_arg,
        lse_arg,
        graph_plan.lse_log_base,
        tuple(resolved_bias_args),
        tuple(resolved_alibi_args),
        tuple(resolved_document_args),
        graph_plan.batch,
        graph_plan.seq,
        graph_plan.head_dim,
        graph_plan.dtype,
        tensor_4d_batch,
        tensor_4d_heads,
    )


def codegen_attention_flash(cg: GenerateAST) -> bool:
    """Replace the device body with the fused tcgen05 flash-attention kernel.

    Called from ``generate_ast.visit_For`` after the FX body walk completes and
    the flash detector has set ``attention_flash_block_ids``. Returns True when
    the flash kernel was emitted (and the FX-derived body discarded), False when
    the shape/layout is outside the validated envelope (so the caller keeps the
    scalar fallback body). Because the detector routes through the same
    ``flash_attention_tensor_plan`` gate, a False return here is a defensive
    backstop rather than an expected path.
    """
    df = cg.device_function
    if df.cute_state.attention_flash_block_ids is None:
        return False

    plan = flash_attention_tensor_plan(df)
    if plan is None:
        return False
    q_arg, k_arg, v_arg, o_arg = plan.q_arg, plan.k_arg, plan.v_arg, plan.o_arg
    lse_arg = plan.lse_arg
    bias_args = plan.bias_args
    alibi_args = plan.alibi_args
    document_args = plan.document_args
    batch, seq, head_dim, io_dtype = plan.batch, plan.seq, plan.head_dim, plan.dtype
    io_dtype_str = _flash_io_dtype_str(io_dtype)
    score_plan = df.cute_state.attention_flash_score_plan
    if score_plan is None or score_plan.head_dim != head_dim:
        return False
    if not score_plan.has_lowering():
        return False
    if lse_arg is not None and plan.lse_log_base != _flash_lse_log_base_for_scale(
        score_plan.lse_scale
    ):
        return False
    is_causal = score_plan.is_causal
    tensor_biases = score_plan.tensor_biases
    if len(tensor_biases) != len(bias_args):
        return False
    alibi_biases = score_plan.alibi_biases
    if len(alibi_biases) != len(alibi_args):
        return False
    document_masks = score_plan.document_masks
    if len(document_masks) != len(document_args):
        return False

    scale_log2 = score_plan.qk_scale_log2
    num_kv = (seq + 127) // 128
    flash_config: Mapping[str, object] | None = df.config
    if score_plan.requires_ws_overlap:
        flash_config = {**df.config, FLASH_TOPOLOGY_KEY: "ws_overlap"}
    cfg = resolve_flash_config(
        head_dim,
        num_kv,
        flash_config,
        dtype=io_dtype,
        is_causal=is_causal,
        prefer_packed_reduce=bool(score_plan.modifiers),
    )
    clc_heads_per_batch = cfg.clc_heads_per_batch
    if (
        cfg.use_clc_scheduler
        and (clc_heads_per_batch <= 0 or clc_heads_per_batch == batch)
        and plan.tensor_4d_batch > 0
        and plan.tensor_4d_heads > 0
        and plan.tensor_4d_batch * plan.tensor_4d_heads == batch
    ):
        clc_heads_per_batch = plan.tensor_4d_heads
    if cfg.use_clc_scheduler and (
        clc_heads_per_batch <= 0 or batch % clc_heads_per_batch != 0
    ):
        clc_heads_per_batch = batch
    if clc_heads_per_batch != cfg.clc_heads_per_batch:
        cfg = dataclasses.replace(cfg, clc_heads_per_batch=clc_heads_per_batch)
    use_tensor_4d_tma = (
        cfg.tensor_4d_tma
        and not is_causal
        and head_dim == 64
        and io_dtype is torch.float16
        and plan.tensor_4d_batch > 0
        and plan.tensor_4d_heads > 0
        and not score_plan.modifiers
    )
    if cfg.small_biased and _small_biased_attention_simt_supported(
        score_plan,
        seq=seq,
        head_dim=head_dim,
        io_dtype=io_dtype,
        lse_arg=lse_arg,
        bias_args=bias_args,
    ):
        bias_arg = bias_args[0]
        cg.cute_wrapper_plans.append(
            {
                "kind": "helion_small_biased_attention",
                "batch": batch,
                "seq": seq,
            }
        )
        df.placeholder_args.update(
            (q_arg.name, k_arg.name, v_arg.name, o_arg.name, bias_arg.name)
        )
        df.cute_state.attention_flash_threads = 64
        df.body = emit_small_biased_attention_simt_device_body(
            q_name=q_arg.name,
            k_name=k_arg.name,
            v_name=v_arg.name,
            o_name=o_arg.name,
            bias_name=bias_arg.name,
            seq=seq,
            head_dim=head_dim,
            io_dtype=io_dtype_str,
        )
        df.preamble = []
        return True

    # num_bh = batch (the collapsed batch*head dim); num_m_tiles = seq // 128.
    # The fa4 topology processes a PAIR of adjacent 128-row Q-tiles per CTA, so
    # its tile space is seq // 256 (requires seq % 256 == 0).
    if cfg.topology == "fa4":
        fa4_tile_rows = 512 if cfg.use_2cta_instrs or cfg.use_cga2_local_cta else 256
        if seq % fa4_tile_rows != 0:
            return False
        total_tiles = batch * (seq // fa4_tile_rows)
    else:
        total_tiles = batch * (seq // 128)

    emit_flash_module_statements(cg)

    # Register the host-side wrapper plan + wrapper-only kernel params.
    wrapper_plan: dict[str, object] = {
        "kind": "helion_flash",
        # ``*_name`` keys are rewritten to positional ``*_idx`` by
        # ``generate_ast.resolve_cute_plan_arg_positions`` against the
        # device function's sorted-arg ordering.
        "q_name": q_arg.name,
        "k_name": k_arg.name,
        "v_name": v_arg.name,
        "o_name": o_arg.name,
        "seq": seq,
        "head_dim": head_dim,
        "dtype": io_dtype_str,
        "batch": batch,
        "scale_log2": scale_log2,
        "kv_stage": cfg.kv_stage,
        "s_stage": cfg.s_stage,
        "persistent": cfg.persistent,
        "persistent_ctas_per_sm": cfg.persistent_ctas_per_sm,
        "total_tiles": total_tiles,
        "topology": cfg.topology,
        # The fa4 topology stages 2 Q-tiles per CTA -> the Q smem layout must
        # be 2-deep (ws_overlap stages a single Q-tile -> 1).
        "q_stage": 2 if cfg.topology == "fa4" else 1,
        # Lever A: build the O TMA-store atom host-side and pass it to the corr
        # epilogue (fa4-only; the env gate already forced topology == "fa4").
        "epi_tma": cfg.epi_tma,
        "epi_stg": cfg.epi_stg,
        "use_2cta_instrs": cfg.use_2cta_instrs,
        "use_cga2_local_cta": cfg.use_cga2_local_cta,
        "use_clc_scheduler": cfg.use_clc_scheduler,
        "clc_heads_per_batch": cfg.clc_heads_per_batch,
        "local_tma_partition": cfg.local_tma_partition,
    }
    if use_tensor_4d_tma:
        wrapper_plan["tensor_4d_batch"] = plan.tensor_4d_batch
        wrapper_plan["tensor_4d_heads"] = plan.tensor_4d_heads
    if cfg.use_clc_scheduler and cfg.clc_use_pdl:
        wrapper_plan["use_pdl"] = True
    if lse_arg is not None:
        wrapper_plan["lse_name"] = lse_arg.name
    if tensor_biases:
        bias_arg = bias_args[0]
        bias_modifier = tensor_biases[0]
        wrapper_plan["bias_name"] = bias_arg.name
        wrapper_plan["score_bias_scale"] = bias_modifier.scale_log2 / scale_log2
    if alibi_biases:
        wrapper_plan["alibi_name"] = alibi_args[0].name
        wrapper_plan["alibi_count"] = int(alibi_args[0].fake_value.shape[0])
    if document_masks:
        wrapper_plan["document_name"] = document_args[0].name
        document_batch = int(document_args[0].fake_value.shape[0])
        wrapper_plan["document_batch"] = document_batch
        wrapper_plan["document_heads_per_batch"] = batch // document_batch
    cg.cute_wrapper_plans.append(wrapper_plan)
    if cfg.use_2cta_instrs or cfg.use_cga2_local_cta:
        df.cute_state.cluster_shape = (2, 1, 1)
    df.wrapper_only_params.extend(_FLASH_KERNEL_PARAMS)
    if cfg.topology != "fa4":
        df.wrapper_only_params.extend(_FLASH_DYNAMIC_TILE_KERNEL_PARAMS)
    if lse_arg is not None:
        df.wrapper_only_params.extend(_FLASH_LSE_KERNEL_PARAMS)
    if tensor_biases:
        df.wrapper_only_params.extend(_FLASH_BIAS_KERNEL_PARAMS)
    if alibi_biases:
        df.wrapper_only_params.extend(_FLASH_ALIBI_KERNEL_PARAMS)
    if document_masks:
        df.wrapper_only_params.extend(_FLASH_DOCUMENT_KERNEL_PARAMS)
    if cfg.epi_tma:
        df.wrapper_only_params.extend(_FLASH_EPI_TMA_KERNEL_PARAMS)
    elif cfg.epi_stg:
        df.wrapper_only_params.extend(_FLASH_EPI_STG_KERNEL_PARAMS)
    # Keep all four tensors alive on the host even though the device body now
    # references them only through the wrapper-built TMA tensors.
    df.placeholder_args.update((q_arg.name, k_arg.name, v_arg.name, o_arg.name))
    if lse_arg is not None:
        df.placeholder_args.add(lse_arg.name)
    for bias_arg in bias_args:
        df.placeholder_args.add(bias_arg.name)
    for alibi_arg in alibi_args:
        df.placeholder_args.add(alibi_arg.name)
    for document_arg in document_args:
        df.placeholder_args.add(document_arg.name)
    cg.cute_uses_matmul = True

    if cfg.topology == "ws_overlap":
        df.cute_state.attention_flash_threads = 256 if cfg.s_stage == 2 else 128
        if cfg.s_stage == 2:
            df.body = list(
                emit_flash_ws_device_body(
                    df,
                    head_dim=head_dim,
                    num_kv=num_kv,
                    cfg=cfg,
                    has_lse=lse_arg is not None,
                    io_dtype=io_dtype_str,
                    score_plan=score_plan,
                )
            )
        else:
            df.body = list(
                emit_flash_device_body(
                    df,
                    head_dim=head_dim,
                    num_kv=num_kv,
                    cfg=cfg,
                    has_lse=lse_arg is not None,
                    io_dtype=io_dtype_str,
                    score_plan=score_plan,
                )
            )
    elif cfg.topology == "fa4":
        df.cute_state.attention_flash_threads = 512
        df.body = list(
            emit_flash_fa4_device_body(
                df,
                head_dim=head_dim,
                num_kv=num_kv,
                num_bh=batch,
                total_tiles=total_tiles,
                cfg=cfg,
                has_lse=lse_arg is not None,
                io_dtype=io_dtype_str,
                score_plan=score_plan,
                tensor_4d_batch=plan.tensor_4d_batch if use_tensor_4d_tma else 0,
                tensor_4d_heads=plan.tensor_4d_heads if use_tensor_4d_tma else 0,
            )
        )
    else:
        raise ValueError(f"unknown flash attention topology: {cfg.topology!r}")

    df.preamble = []
    return True
