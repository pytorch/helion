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
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..device_function import TensorArg
from .attention_plan import ALIBI_BIAS_KIND
from .attention_plan import CAUSAL_MASK_KIND
from .attention_plan import DENSE_SCORE_KIND
from .attention_plan import DOCUMENT_MASK_KIND
from .attention_plan import PREFIX_LM_MASK_KIND
from .attention_plan import RELATIVE_BIAS_KIND
from .attention_plan import SLIDING_WINDOW_MASK_KIND
from .attention_plan import SOFTCAP_KIND
from .attention_plan import TENSOR_BIAS_KIND
from .attention_plan import AttentionScorePlan
from .causal_range import CausalRangeProof
from .causal_range import IntegerInterval
from .causal_range import TileLayout
from .causal_range import prove_descending_causal_prefix_unmasked
from .flash_policy import get_flash_target_policy
from .flash_schedule import FlashScheduleSpec
from .flash_schedule import FlashStatReleaseMapping
from .flash_schedule import build_fa4_schedule
from .flash_schedule import max_fa4_kv_depth
from .flash_schedule import verify_flash_schedule
from .flash_tuning import FlashCausalSeedTemplate
from .flash_tuning import FlashPackedExp2Mode
from .flash_tuning import FlashSoftmaxLowering

_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .flash_tuning import FlashCausalTuningPolicy
    from .flash_tuning import FlashDenseTuningPolicy


class FlashGraphOutputPlan(NamedTuple):
    """Host tensor names and static shape proven safe for flash replacement."""

    q_name: str
    k_name: str
    v_name: str
    o_name: str
    output_epilogue: str
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
    return _flash_store_value_output_epilogue(node) is not None


_FLASH_OUTPUT_EPILOGUE_IDENTITY = "identity"
_FLASH_OUTPUT_EPILOGUE_RELU = "relu"


def _flash_output_relu_enabled(output_epilogue: str) -> bool:
    if output_epilogue == _FLASH_OUTPUT_EPILOGUE_IDENTITY:
        return False
    if output_epilogue == _FLASH_OUTPUT_EPILOGUE_RELU:
        return True
    raise AssertionError(f"unsupported flash output epilogue: {output_epilogue!r}")


def _flash_store_value_output_epilogue(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops
    from ...language import view_ops

    if node.op != "call_function" or node.target is not memory_ops.store:
        return None
    if len(node.args) < 3 or not isinstance(node.args[2], torch.fx.Node):
        return None
    value = node.args[2]
    if (
        value.op == "call_function"
        and value.target is torch.ops.prims.convert_element_type.default
    ):
        output = node.args[0]
        if (
            value.kwargs
            or len(value.args) != 2
            or not isinstance(value.args[0], torch.fx.Node)
            or not isinstance(output, torch.fx.Node)
            or value.args[1] != getattr(output.meta.get("val"), "dtype", None)
        ):
            return None
        value = value.args[0]
    output_epilogue = _FLASH_OUTPUT_EPILOGUE_IDENTITY
    if value.op == "call_function" and value.target is torch.ops.aten.relu.default:
        if (
            value.kwargs
            or len(value.args) != 1
            or not isinstance(value.args[0], torch.fx.Node)
        ):
            return None
        output_epilogue = _FLASH_OUTPUT_EPILOGUE_RELU
        value = value.args[0]
    if value.op != "call_function" or value.target is not torch.ops.aten.div.Tensor:
        return None
    if len(value.args) < 2:
        return None
    acc_node, sum_view = value.args[:2]
    if not isinstance(acc_node, torch.fx.Node) or not isinstance(
        sum_view, torch.fx.Node
    ):
        return None
    acc_loop = _flash_phi_loop_result(acc_node, 2)
    if acc_loop is None:
        return None
    if sum_view.op != "call_function" or sum_view.target is not view_ops.subscript:
        return None
    if not sum_view.args or not isinstance(sum_view.args[0], torch.fx.Node):
        return None
    sum_loop = _flash_phi_loop_result(sum_view.args[0], 1)
    if sum_loop is not acc_loop:
        return None
    indices = sum_view.args[1] if len(sum_view.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    if not (
        _is_full_slice(indices[0]) and _is_full_slice(indices[1]) and indices[2] is None
    ):
        return None
    return output_epilogue


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
    o_store_epilogues: list[str] = []
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
            output_epilogue = _flash_store_value_output_epilogue(node)
            if output_epilogue is None:
                return None
            o_store_names.append(name)
            o_store_epilogues.append(output_epilogue)
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
    if len(o_store_epilogues) != 1:
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
    output_epilogue = o_store_epilogues[0]
    if output_epilogue == _FLASH_OUTPUT_EPILOGUE_RELU and (
        io_dtype is not torch.bfloat16
        or lse_name is not None
        or bias_names
        or alibi_names
        or document_names
        or score_plan is None
        or not (
            _standard_dense_score_plan_supported(score_plan)
            or _standard_causal_score_plan_supported(score_plan)
        )
    ):
        return None
    return FlashGraphOutputPlan(
        q_name,
        k_name,
        v_name,
        o_name,
        output_epilogue,
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


def _flash_4d_value_base_batch_heads(
    value: torch.Tensor,
    *,
    batch: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[int, int] | None:
    """Return ``(z, h)`` when a flattened flash input has a BHSD base."""
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


def _flash_values_support_tensor_4d_tma(
    q_value: torch.Tensor,
    k_value: torch.Tensor,
    v_value: torch.Tensor,
    *,
    score_plan: AttentionScorePlan,
    batch: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
) -> bool:
    """Mirror the emitter's proof that Q/K/V can use BHSD TMA descriptors."""
    if score_plan.modifiers or dtype is not torch.float16 or head_dim != 64:
        return False
    return (
        _flash_values_tensor_4d_batch_heads(
            q_value,
            k_value,
            v_value,
            batch=batch,
            seq=seq,
            head_dim=head_dim,
            dtype=dtype,
        )
        is not None
    )


def _flash_values_tensor_4d_batch_heads(
    q_value: torch.Tensor,
    k_value: torch.Tensor,
    v_value: torch.Tensor,
    *,
    batch: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[int, int] | None:
    """Return common BHSD base geometry independently of TMA eligibility."""
    q_base_shape = _flash_4d_value_base_batch_heads(
        q_value,
        batch=batch,
        seq=seq,
        head_dim=head_dim,
        dtype=dtype,
    )
    if q_base_shape is None:
        return None
    if not all(
        _flash_4d_value_base_batch_heads(
            value,
            batch=batch,
            seq=seq,
            head_dim=head_dim,
            dtype=dtype,
        )
        == q_base_shape
        for value in (k_value, v_value)
    ):
        return None
    return q_base_shape


def flash_attention_graph_tensor_4d_batch_heads_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> tuple[int, int] | None:
    """Return common input BHSD geometry without imposing 4D-TMA gates."""
    graphs = tuple(graphs)
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    if graph_plan is None:
        return None
    host_tensors = _flash_graph_host_tensors(graphs)
    values = tuple(
        host_tensors.get(name)
        for name in (graph_plan.q_name, graph_plan.k_name, graph_plan.v_name)
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        return None
    q_value, k_value, v_value = cast("tuple[torch.Tensor, ...]", values)
    return _flash_values_tensor_4d_batch_heads(
        q_value,
        k_value,
        v_value,
        batch=graph_plan.batch,
        seq=graph_plan.seq,
        head_dim=graph_plan.head_dim,
        dtype=graph_plan.dtype,
    )


def flash_attention_graph_supports_tensor_4d_tma_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> bool:
    """Return whether this graph can emit real 4D Q/K/V TMA descriptors."""
    graphs = tuple(graphs)
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    if graph_plan is None:
        return False
    host_tensors = _flash_graph_host_tensors(graphs)
    values = tuple(
        host_tensors.get(name)
        for name in (graph_plan.q_name, graph_plan.k_name, graph_plan.v_name)
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        return False
    q_value, k_value, v_value = cast("tuple[torch.Tensor, ...]", values)
    return _flash_values_support_tensor_4d_tma(
        q_value,
        k_value,
        v_value,
        score_plan=score_plan,
        batch=graph_plan.batch,
        seq=graph_plan.seq,
        head_dim=graph_plan.head_dim,
        dtype=graph_plan.dtype,
    )


def flash_attention_graph_lse_plan_valid_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan | None = None,
) -> bool:
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    if graph_plan is None:
        return False
    # LSE uses a direct layout with signed Int32 element coordinates. Unlike O,
    # it has no TMA store path to carry a wider address.
    return graph_plan.lse_name is None or graph_plan.batch * graph_plan.seq <= 1 << 31


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


def _standard_dense_score_plan_supported(score_plan: AttentionScorePlan) -> bool:
    return not score_plan.modifiers and math.isclose(
        score_plan.qk_scale_log2,
        math.log2(math.e) / math.sqrt(score_plan.head_dim),
        rel_tol=1e-8,
        abs_tol=1e-8,
    )


def _standard_causal_score_plan_supported(score_plan: AttentionScorePlan) -> bool:
    return score_plan.modifier_kinds == (CAUSAL_MASK_KIND,) and math.isclose(
        score_plan.qk_scale_log2,
        math.log2(math.e) / math.sqrt(score_plan.head_dim),
        rel_tol=1e-8,
        abs_tol=1e-8,
    )


def _flash_output_requires_tma(num_bh: int, seq: int, head_dim: int) -> bool:
    # Direct/STG output layouts use signed Int32 element coordinates. The
    # largest valid coordinate is numel - 1, so exactly 2**31 elements is safe.
    return num_bh * seq * head_dim > 1 << 31


def flash_attention_graph_standard_dense_output_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> bool:
    """Return whether the graph has canonical dense output-only semantics."""
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    return bool(
        graph_plan is not None
        and graph_plan.lse_name is None
        and not graph_plan.bias_names
        and not graph_plan.alibi_names
        and not graph_plan.document_names
        and _standard_dense_score_plan_supported(score_plan)
    )


def flash_attention_graph_standard_causal_output_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> bool:
    """Return whether the graph has canonical causal output-only semantics."""
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    return bool(
        graph_plan is not None
        and graph_plan.lse_name is None
        and not graph_plan.bias_names
        and not graph_plan.alibi_names
        and not graph_plan.document_names
        and _standard_causal_score_plan_supported(score_plan)
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


def _flash_deep_1cta_kv_stage_cap(head_dim: int) -> int:
    """Largest legal per-ring K/V depth while retaining local S2."""
    if head_dim == 64:
        return 4
    if head_dim == 128:
        return 2
    return 0


def _flash_aliased_kv_stage_cap(head_dim: int, *, stage_output: bool) -> int:
    """Largest legal aliased K/V depth for the requested output storage."""
    return max_fa4_kv_depth(
        FlashScheduleSpec(
            head_dim=head_dim,
            kv_depth=2,
            stage_output=stage_output,
        )
    )


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


class FlashPipelineFamilyFlags(NamedTuple):
    topology: str
    separate_kv_rings: bool = False
    causal_two_cta: bool = False
    use_2cta_instrs: bool = False
    use_cga2_local_cta: bool = False
    use_clc_scheduler: bool = False
    local_tma_partition: bool = False
    tensor_4d_tma: bool = False


FLASH_PIPELINE_FAMILY_FLAGS: dict[str, FlashPipelineFamilyFlags] = {
    "ws_overlap": FlashPipelineFamilyFlags("ws_overlap"),
    "fa4": FlashPipelineFamilyFlags("fa4"),
    "fa4_deep_1cta": FlashPipelineFamilyFlags("fa4", separate_kv_rings=True),
    "fa4_2cta_causal": FlashPipelineFamilyFlags(
        "fa4", causal_two_cta=True, use_2cta_instrs=True
    ),
    "fa4_tma_4d": FlashPipelineFamilyFlags("fa4", tensor_4d_tma=True),
    "fa4_local_tma": FlashPipelineFamilyFlags("fa4", local_tma_partition=True),
    "fa4_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4", local_tma_partition=True, tensor_4d_tma=True
    ),
    "fa4_cga2_local": FlashPipelineFamilyFlags("fa4", use_cga2_local_cta=True),
    "fa4_cga2_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_cga2_local_cta=True, tensor_4d_tma=True
    ),
    "fa4_2cta": FlashPipelineFamilyFlags("fa4", use_2cta_instrs=True),
    "fa4_2cta_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_2cta_instrs=True, tensor_4d_tma=True
    ),
    "fa4_clc": FlashPipelineFamilyFlags("fa4", use_clc_scheduler=True),
    "fa4_clc_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_clc_scheduler=True, tensor_4d_tma=True
    ),
    "fa4_clc_local_tma": FlashPipelineFamilyFlags(
        "fa4", use_clc_scheduler=True, local_tma_partition=True
    ),
    "fa4_clc_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4",
        use_clc_scheduler=True,
        local_tma_partition=True,
        tensor_4d_tma=True,
    ),
}
FLASH_PIPELINE_FAMILIES = tuple(FLASH_PIPELINE_FAMILY_FLAGS)
FLASH_AUTOTUNE_PIPELINE_FAMILIES = FLASH_PIPELINE_FAMILIES


def _flash_pipeline_family_flags(
    family: object,
) -> FlashPipelineFamilyFlags | None:
    return FLASH_PIPELINE_FAMILY_FLAGS.get(family) if isinstance(family, str) else None


def _flash_pipeline_family_from_flags(
    *,
    topology: str,
    separate_kv_rings: bool,
    causal_two_cta: bool,
    use_2cta_instrs: bool,
    use_cga2_local_cta: bool,
    use_clc_scheduler: bool,
    local_tma_partition: bool,
    tensor_4d_tma: bool,
) -> str:
    if topology != "fa4":
        return "ws_overlap"
    if separate_kv_rings:
        base = "fa4_deep_1cta"
        local_tma_partition = False
        tensor_4d_tma = False
    elif use_2cta_instrs:
        base = "fa4_2cta_causal" if causal_two_cta else "fa4_2cta"
        local_tma_partition = False
    elif use_cga2_local_cta:
        base = "fa4_cga2_local"
        local_tma_partition = False
    elif use_clc_scheduler:
        base = "fa4_clc"
    else:
        base = "fa4"
    if local_tma_partition:
        base += "_local_tma"
    if tensor_4d_tma:
        base += "_tma_4d" if not local_tma_partition else "_4d"
    assert base in FLASH_PIPELINE_FAMILY_FLAGS
    return base


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

    # Source-level scheduling choices must be explicit config dimensions.  In
    # particular, do not select these from sequence-length thresholds in the
    # emitter: doing so makes one nominal config mean different programs for
    # neighboring shapes and prevents the autotuner from measuring both.
    persistent_loop: str = "while"
    sp_row_sum: str = "fragment"
    softmax_setup: str = "shared"
    epi_tma_setup: str = "shared"

    # Stage B/C topology fields (consumed by the fa4 emitter)
    topology: str = "ws_overlap"
    pipeline_family: str = "ws_overlap"
    num_softmax_warpgroups: int = 1
    num_correction_warps: int = 0
    num_mma_warps: int = 0
    num_load_warps: int = 0
    num_epilogue_warps: int = 0
    # Query-stage count is structural: FA4's barrier/TMEM graph owns two local
    # query slots, while ws_overlap owns one. Legacy fixed configs are accepted
    # but canonicalized to this family-derived value.
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
    # Packed exp2 instruction scheduling within one score fragment. The first
    # component is the packed-pair window; the second is the polynomial batch.
    exp2_packet: str = "1x1"
    tmem_plan: str = "separate"
    tmem_s_to_p_offset: int = 0
    # True alternates PV/QK by query slot; False groups the two PV issues before
    # the two following QK issues. Both preserve P-before-QK TMEM alias edges.
    mma_interleave: bool = True
    # Explicit PTX mbarrier try-wait hint. Zero is a distinct policy, not an
    # omitted operand.
    wait_hint: int = 10_000_000
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
    # original Q0/K/Q1/V order; the other values expose legal load orderings.
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
    # Statistics handoff between the softmax and correction roles. ``single``
    # uses one slot with a named ready barrier and an mbarrier reuse
    # acknowledgement; ``single_final`` relies on the MMA chain for steady-state
    # reuse and acknowledges only the terminal row sum, once per work item;
    # ``ring2`` uses the two-slot mbarrier ring.
    stat_transport: str = "ring2"
    # Experimental FA4_SKIP_RESCALE_STATS lever. The resolver currently clamps
    # this off because dropping per-KV alpha handoffs is only correct if every
    # later tile is known to stay on the pinned exponent base.
    skip_rescale_stats: bool = False
    # TMEM O-rescale chunk width. hd64 defaults to 32 cols because 64 cols pushes
    # some FA4 attention shapes over the ptxas register target, but manual
    # Manual configurations can opt into 64 columns.
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
    # Causal FA4 launch-order swizzle. Legacy 0 canonicalizes to the no-swizzle
    # group size 1; larger values set the batch/head interleave explicitly.
    causal_lpt_swizzle: int = 0
    # Causal FA4 KV traversal order. ``ascending`` preserves Helion's original
    # left-to-right stream; ``descending`` matches upstream FA4's diagonal-first
    # stream and is searchable because workload characteristics can favor it.
    causal_kv_order: str = "ascending"
    # Warp-role assignment inside the last FA4 warpgroup. ``helion`` preserves
    # the original transcription; ``fa4`` matches upstream's load/epilogue warp
    # ordering.
    role_map: str = "helion"
    # Chain mutually exclusive warp roles through one if/elif dispatch. This is
    # source-visible and must be captured by saved autotune configurations.
    role_chain: bool = False
    # One-CTA pipeline variant: K and V have distinct shared-memory rings and
    # independent producer/consumer barriers. The score topology stays fixed at
    # FA4's two local query stages.
    separate_kv_rings: bool = False
    # Causal CtaGroup.TWO uses a cluster-uniform K/V trip count while retaining
    # rank-local query coordinates for masking and output ownership.
    causal_two_cta: bool = False
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


def _flash_causal_hd64_num_kv_supported(num_kv: int | None) -> bool:
    """Return whether paired-query causal FA4 can cover this KV extent."""
    return isinstance(num_kv, int) and num_kv >= 2 and num_kv % 2 == 0


def _flash_causal_degree2_num_kv_supported(num_kv: int | None) -> bool:
    """Return whether the causal degree-2 packet can cover this KV extent."""
    return _flash_causal_hd64_num_kv_supported(num_kv)


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


_FLASH_EXP2_PACKET_PARAMS: dict[str, tuple[int, int]] = {
    "1x1": (1, 1),
    "4x1": (4, 1),
    "4x2": (4, 2),
    "8x1": (8, 1),
    "8x2": (8, 2),
}
_FLASH_DEG2_EXP2_PACKET = "deg2_16x6"
_FLASH_HYBRID_EXP2_PACKET = "hybrid_deg1_16x8"
_FLASH_DEG1_EXP2_PACKET = "deg1_16x8"
_FLASH_DEG1_SHORT_CORR10_EXP2_PACKET = "deg1_8x2_corr10"
_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET = (
    "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
)
_FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS = frozenset(
    (_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET,)
)
_FLASH_DEG1_EXP2_PACKETS = frozenset(
    (_FLASH_DEG1_EXP2_PACKET, _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET)
)
_FLASH_DEG1_EXP2_OFFSET = 0
_FLASH_DEG1_EXP2_OFFSET0 = 10
_FLASH_MANUAL_EXP2_PACKET_PARAMS: dict[str, tuple[int, int]] = {
    _FLASH_DEG2_EXP2_PACKET: (8, 3),
    _FLASH_HYBRID_EXP2_PACKET: (8, 4),
    _FLASH_DEG1_EXP2_PACKET: (8, 4),
    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET: (8, 2),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET: (
        8,
        3,
    ),
}
_FLASH_MANUAL_EXP2_PACKET_SCHEDULES: dict[str, tuple[int, int]] = {
    _FLASH_DEG2_EXP2_PACKET: (16, 6),
    _FLASH_HYBRID_EXP2_PACKET: (16, 8),
    _FLASH_DEG1_EXP2_PACKET: (16, 8),
    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET: (8, 2),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET: (
        16,
        6,
    ),
}


def flash_exp2_packet_is_compound(packet: object) -> bool:
    """Return whether a packet owns structural schedule requirements."""
    return packet in _FLASH_MANUAL_EXP2_PACKET_PARAMS


def _flash_dense_hd64_2cta_num_kv_supported(num_kv: int) -> bool:
    """Return whether ``num_kv`` is in the aligned dense hd64 2CTA envelope."""
    return num_kv >= 4 and num_kv % 4 == 0


def _flash_dense_hd128_degree2_num_kv_supported(num_kv: int) -> bool:
    """Return whether dense D128 can use the paired-CTA degree-2 schedule."""
    return num_kv >= 4 and num_kv % 4 == 0


def _flash_paired_num_kv_supported(num_kv: int, *, minimum: int) -> bool:
    """Return whether a two-query-tile schedule covers an aligned KV extent."""
    return num_kv >= minimum and num_kv % 2 == 0


def _flash_causal_hd128_resident_num_kv_supported(num_kv: int) -> bool:
    """Return whether BF16 D128 can use the resident-score schedule."""
    return _flash_paired_num_kv_supported(num_kv, minimum=2)


def _flash_causal_hd64_hybrid_num_kv_supported(num_kv: int) -> bool:
    """Return whether D64 can use the causal hybrid schedule."""
    return _flash_paired_num_kv_supported(num_kv, minimum=2)


def _flash_causal_hd64_hybrid_family_eligible(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    is_causal: bool,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_causal_output: bool,
) -> bool:
    """Return whether the hybrid packet is structurally legal."""
    return (
        standard_causal_output
        and is_causal
        and _flash_supported_io_dtype(dtype)
        and head_dim == 64
        and _flash_causal_hd64_hybrid_num_kv_supported(num_kv)
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
    )


def _flash_compound_exp2_packet_overrides(
    head_dim: int,
    num_kv: int,
    config: Mapping[str, object],
    *,
    dtype: torch.dtype,
    is_causal: bool,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
) -> dict[str, object]:
    """Canonicalize only the structural requirements of a polynomial packet.

    Packet cadence and barrier prerequisites are part of the algorithm.  Phase
    offsets, register budgets, wait policy, pipeline depth, and other
    performance choices deliberately remain untouched so autotuning must
    measure them rather than receiving an embedded winner preset.
    """
    packet = config.get(FLASH_EXP2_PACKET_KEY)
    common: dict[str, object] = {
        FLASH_Q_TILE_COUNT_KEY: 2,
        FLASH_P_STORE_REP_KEY: 16,
        FLASH_S_LOAD_REP_KEY: 32,
    }
    if packet not in _FLASH_MANUAL_EXP2_PACKET_PARAMS:
        return {}

    def with_required_family(
        family: str, overrides: Mapping[str, object]
    ) -> dict[str, object]:
        requested_family = config.get(FLASH_PIPELINE_FAMILY_KEY)
        if requested_family is not None and requested_family != family:
            # The family is the parent schedule choice. A packet that requires a
            # different family is inactive under a pinned/hierarchical family,
            # rather than silently changing the requested search space.
            return {FLASH_EXP2_PACKET_KEY: "1x1"}
        return {**overrides, FLASH_PIPELINE_FAMILY_KEY: family}

    if (
        packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
        and standard_causal_output
        and is_causal
        and dtype is torch.bfloat16
        and head_dim == 128
        and _flash_causal_hd128_resident_num_kv_supported(num_kv)
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
    ):
        return with_required_family(
            "fa4",
            {
                **common,
                FLASH_E2E_SCHEDULE_KEY: "16/6",
                FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
                FLASH_SOFTMAX_DISC_KEY: True,
                FLASH_DISC_PIPE_KEY: 2,
                FLASH_SPLIT_P_ARRIVE_KEY: True,
                FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            },
        )
    if (
        packet == _FLASH_DEG2_EXP2_PACKET
        and standard_dense_output
        and not is_causal
        and dtype is torch.bfloat16
        and head_dim == 128
        and _flash_dense_hd128_degree2_num_kv_supported(num_kv)
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
    ):
        return with_required_family(
            "fa4_2cta",
            {
                **common,
                FLASH_E2E_SCHEDULE_KEY: "16/6",
                FLASH_SOFTMAX_DISC_KEY: True,
            },
        )
    if (
        packet == _FLASH_HYBRID_EXP2_PACKET
        and _flash_causal_hd64_hybrid_family_eligible(
            head_dim,
            num_kv,
            dtype=dtype,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            standard_causal_output=standard_causal_output,
        )
    ):
        requested_disc_pipe = config.get(FLASH_DISC_PIPE_KEY)
        disc_pipe = (
            requested_disc_pipe
            if isinstance(requested_disc_pipe, int) and requested_disc_pipe >= 2
            else 2
        )
        return with_required_family(
            "fa4",
            {
                **common,
                FLASH_E2E_SCHEDULE_KEY: "16/8",
                FLASH_MASKED_E2E_SCHEDULE_KEY: "16/8",
                FLASH_SOFTMAX_DISC_KEY: True,
                FLASH_DISC_PIPE_KEY: disc_pipe,
                FLASH_SPLIT_P_ARRIVE_KEY: True,
                FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
            },
        )
    if (
        dtype is torch.float16
        and head_dim == 64
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
    ):
        if (
            standard_dense_output
            and not is_causal
            and _flash_dense_hd64_2cta_num_kv_supported(num_kv)
            and packet
            in (
                _FLASH_DEG2_EXP2_PACKET,
                _FLASH_DEG1_EXP2_PACKET,
                _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET,
            )
        ):
            cadence = _FLASH_MANUAL_EXP2_PACKET_SCHEDULES[str(packet)]
            return with_required_family(
                "fa4_2cta",
                {
                    **common,
                    FLASH_E2E_SCHEDULE_KEY: f"{cadence[0]}/{cadence[1]}",
                    FLASH_SOFTMAX_DISC_KEY: False,
                },
            )
        if (
            packet == _FLASH_DEG2_EXP2_PACKET
            and standard_causal_output
            and is_causal
            and _flash_causal_degree2_num_kv_supported(num_kv)
        ):
            requested_disc_pipe = config.get(FLASH_DISC_PIPE_KEY)
            disc_pipe = (
                requested_disc_pipe
                if isinstance(requested_disc_pipe, int) and requested_disc_pipe >= 2
                else 2
            )
            return with_required_family(
                "fa4",
                {
                    **common,
                    FLASH_E2E_SCHEDULE_KEY: "16/6",
                    FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
                    FLASH_SOFTMAX_DISC_KEY: True,
                    FLASH_DISC_PIPE_KEY: disc_pipe,
                    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
                },
            )
    return {}


class _FlashDiscExp2CodegenParams(NamedTuple):
    e2e_freq: int
    e2e_res: int
    pair_batch: int
    emu_batch: int
    degree2: bool
    degree1_unmasked: bool


def _flash_exp2_packet_params(packet: str) -> tuple[int, int]:
    """Return the static packed-pair and emulation batches for ``packet``."""
    return _FLASH_EXP2_PACKET_PARAMS.get(
        packet,
        _FLASH_MANUAL_EXP2_PACKET_PARAMS.get(packet, _FLASH_EXP2_PACKET_PARAMS["1x1"]),
    )


def _flash_disc_exp2_codegen_params(
    packet: str, e2e_freq: int, e2e_res: int
) -> _FlashDiscExp2CodegenParams:
    """Return cadence, packet batches, and polynomial route for disc PASS2."""
    pair_batch, emu_batch = _flash_exp2_packet_params(packet)
    manual_schedule = _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.get(packet)
    if manual_schedule is not None:
        return _FlashDiscExp2CodegenParams(
            *manual_schedule,
            pair_batch,
            emu_batch,
            packet not in _FLASH_DEG1_EXP2_PACKETS,
            packet == _FLASH_HYBRID_EXP2_PACKET or packet in _FLASH_DEG1_EXP2_PACKETS,
        )
    return _FlashDiscExp2CodegenParams(
        e2e_freq, e2e_res, pair_batch, emu_batch, False, False
    )


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
    num_bh: int | None = None,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    supports_tensor_4d_tma: bool = True,
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

    packet_config = config
    env_exp2_packet = os.environ.get("HELION_CUTE_FLASH_EXP2_PACKET")
    if env_exp2_packet is not None and (
        packet_config is None or FLASH_EXP2_PACKET_KEY not in packet_config
    ):
        packet_config = {
            **({} if packet_config is None else packet_config),
            FLASH_EXP2_PACKET_KEY: env_exp2_packet,
        }
    compound_packet_overrides = (
        _flash_compound_exp2_packet_overrides(
            head_dim,
            num_kv,
            packet_config,
            dtype=dtype,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            standard_dense_output=standard_dense_output,
            standard_causal_output=standard_causal_output,
        )
        if packet_config is not None
        else {}
    )

    def _cfg(key: str) -> object | None:
        if key in compound_packet_overrides:
            return compound_packet_overrides[key]
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
    if kv_stage_cfg is not None and num_kv >= 2:
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
    legacy_structural_config = any(
        _cfg(key) is not None for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
    )
    pipeline_family_cfg = _cfg(FLASH_PIPELINE_FAMILY_KEY)
    pipeline_family_env = os.environ.get("HELION_CUTE_FLASH_PIPELINE_FAMILY")
    requested_family_flags = (
        _flash_pipeline_family_flags(pipeline_family_cfg)
        if pipeline_family_cfg is not None
        else None
        if legacy_structural_config
        else _flash_pipeline_family_flags(pipeline_family_env)
    )
    if requested_family_flags is not None:
        topology = requested_family_flags.topology
    else:
        topology = os.environ.get("HELION_CUTE_FLASH_TOPOLOGY", topology_default)
        # Legacy fixed configs and environment overrides remain accepted. The
        # compound family above takes precedence when explicitly selected.
        topology_cfg = _cfg(FLASH_TOPOLOGY_KEY)
        if topology_cfg is not None:
            topology = str(topology_cfg)
    if topology not in ("ws_overlap", "fa4"):
        topology = "ws_overlap"
    if topology == "fa4" and num_kv % 2 != 0:
        topology = "ws_overlap"
    dense_hd64_fa4 = topology == "fa4" and not is_causal and head_dim == 64
    causal_hd64_fa4 = topology == "fa4" and is_causal and head_dim == 64
    causal_e2e_offset_default = 2
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
            (is_causal or dense_hd64_fa4)
            and kv_stage_cfg is None
            and os.environ.get("HELION_CUTE_FLASH_KV_STAGE") is None
        ):
            kv_stage = 3 if dense_hd64_fa4 else 2
        # fa4 is 2-Q-tile (sQ holds q_stage=2 tiles) and uses one aliased K/V
        # shared-memory ring, matching FA4. That keeps kv_stage=3 within the
        # sm_100a shared-memory budget even for head_dim 128.
        if kv_stage < 2:
            # The aliased stream is ordered K0, V0, K1, V1, ... . A one-slot ring
            # cannot hold V_i while prefetching K_{i+1}, so clamp user/env
            # overrides to the minimum safe ring depth.
            kv_stage = 2
    elif num_kv >= 2:
        # WS stores K and V in distinct shared-memory rings. Bound inherited
        # family-agnostic requests by the storage-derived per-ring capacity;
        # deeper FA4 aliased-ring values otherwise fail at launch.
        ws_kv_stage_cap = _flash_deep_1cta_kv_stage_cap(head_dim)
        if ws_kv_stage_cap > 0:
            kv_stage = min(max(kv_stage, 2), ws_kv_stage_cap)
    persistent_ctas_per_sm_default = 1
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
    recompute_tile_coords_default = False
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
    # The FA4 emitter, TMEM layout, and barrier graph all own exactly two local
    # query slots. Preserve the legacy input but canonicalize it to the selected
    # structural family before it contributes to config identity.
    q_tile_count = 2 if topology == "fa4" else 1
    mma_interleave = _flash_bool_env("HELION_CUTE_FLASH_MMA_INTERLEAVE", True)
    mma_interleave_cfg = _cfg(FLASH_MMA_INTERLEAVE_KEY)
    if mma_interleave_cfg is not None:
        mma_interleave = bool(mma_interleave_cfg)
    if topology != "fa4":
        mma_interleave = False
    wait_hint_default = 10_000_000
    wait_hint = int(
        os.environ.get("HELION_CUTE_FLASH_WAIT_HINT", str(wait_hint_default))
    )
    wait_hint_cfg = _cfg(FLASH_WAIT_HINT_KEY)
    if wait_hint_cfg is not None:
        wait_hint = int(wait_hint_cfg)  # type: ignore[arg-type]
    if wait_hint not in (0, wait_hint_default) or topology != "fa4":
        wait_hint = wait_hint_default
    acc_stage = int(os.environ.get("HELION_CUTE_FLASH_ACC_STAGE", "1"))
    epi_stage = int(os.environ.get("HELION_CUTE_FLASH_EPI_STAGE", "1"))
    # Exp2 pipe-split schedule. The autotuner sees this as one paired schedule
    # knob so it never searches meaningless combinations like exp2_impl="xu"
    # with independent e2e cadence values, or regressive res/freq pairings.
    e2e_schedule_default = _flash_e2e_schedule_default(topology, head_dim)
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
    masked_e2e_schedule_default = "inherit"
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
    e2e_offset0_default = 0
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
    # fa4 Stage 2b: PTX-path MMA warp (default ON for fa4). HELION_CUTE_FLASH_MMA_PTX=0
    # reverts to the cute.gemm path (the Stage-1/2a body, for A/B comparison).
    mma_ptx = _flash_bool_env("HELION_CUTE_FLASH_MMA_PTX", True)
    if topology == "fa4" and not mma_ptx:
        # The CuTe-GEMM fallback has one established interleaved issue order.
        # Canonicalize the inactive arithmetic child before source identity.
        mma_interleave = True
    # fa4 Step 2: chunked-t2r ("disc") softmax body (default ON for fa4).
    # HELION_CUTE_FLASH_SOFTMAX_DISC=0 reverts to the whole-row body for A/B.
    softmax_disc_default = True
    softmax_disc = _flash_bool_env(
        "HELION_CUTE_FLASH_SOFTMAX_DISC", softmax_disc_default
    )
    softmax_disc_cfg = _cfg(FLASH_SOFTMAX_DISC_KEY)
    if softmax_disc_cfg is not None:
        softmax_disc = bool(softmax_disc_cfg)
    if topology != "fa4":
        softmax_disc = True
    elif is_causal:
        # Causal correction has no acknowledged single-slot statistics handoff.
        # The whole-row/ring2 protocol can lap after a sufficiently long KV
        # traversal, so causal FA4 uses the chunked pipeline unconditionally.
        softmax_disc = True
    # fa4 Step 3 (L1): disc PASS2 software-pipeline depth. Keep one neutral
    # default; the autotuner measures the legal depths for each workload.
    if topology == "fa4" and softmax_disc:
        disc_pipe_default = 2
    else:
        disc_pipe_default = 1
    disc_pipe_depth = int(
        os.environ.get("HELION_CUTE_FLASH_DISC_PIPE", str(disc_pipe_default))
    )
    disc_pipe_depth_cfg = _cfg(FLASH_DISC_PIPE_KEY)
    if disc_pipe_depth_cfg is not None:
        disc_pipe_depth = int(disc_pipe_depth_cfg)  # type: ignore[arg-type]
    split_p_arrive_default = not (
        topology == "fa4" and not is_causal and head_dim <= 64
    )
    split_p_arrive = _flash_bool_env(
        "HELION_CUTE_FLASH_SPLIT_P_ARRIVE", split_p_arrive_default
    )
    split_p_arrive_cfg = _cfg(FLASH_SPLIT_P_ARRIVE_KEY)
    if split_p_arrive_cfg is not None:
        split_p_arrive = bool(split_p_arrive_cfg)
    if topology != "fa4" or not mma_ptx:
        split_p_arrive = True
    p_store_repetition_default = 16
    p_store_repetition = int(
        os.environ.get("HELION_CUTE_FLASH_P_STORE_REP", str(p_store_repetition_default))
    )
    p_store_repetition_cfg = _cfg(FLASH_P_STORE_REP_KEY)
    if p_store_repetition_cfg is not None:
        p_store_repetition = int(p_store_repetition_cfg)  # type: ignore[arg-type]
    if p_store_repetition not in (16, 32) or topology != "fa4":
        p_store_repetition = 16
    if is_causal and split_p_arrive and p_store_repetition == 32:
        # Long-sequence stress found the causal mixed Rep32/Rep16 publication
        # helper can fail to complete. Keep Rep32 available with one P arrival,
        # but use the established Rep16 helper when causal publication is split.
        p_store_repetition = 16
    s_load_repetition_default = 32
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
    precompute_qk_desc_default = False
    precompute_qk_desc = _flash_bool_env(
        "HELION_CUTE_FLASH_PRECOMPUTE_QK_DESC", precompute_qk_desc_default
    )
    precompute_qk_desc_cfg = _cfg(FLASH_PRECOMPUTE_QK_DESC_KEY)
    if precompute_qk_desc_cfg is not None:
        precompute_qk_desc = bool(precompute_qk_desc_cfg)
    if topology != "fa4" or not mma_ptx:
        precompute_qk_desc = False
    first_load_order_default = 0
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
    kv_order_default = "ascending"
    kv_order = os.environ.get("HELION_CUTE_FLASH_KV_ORDER", kv_order_default)
    kv_order_cfg = _cfg(FLASH_KV_ORDER_KEY)
    if kv_order_cfg is not None:
        kv_order = str(kv_order_cfg)
    if topology != "fa4" or is_causal or kv_order not in ("ascending", "descending"):
        kv_order = "ascending"
    # fa4 Step 3 (Lever A): epilogue TMA-store. D128 uses the resource-safe TMA
    # default; D64 searches both output paths. fa4-only.
    epi_tma_default = topology == "fa4" and head_dim == 128
    epi_tma = _flash_bool_env("HELION_CUTE_FLASH_EPI_TMA", epi_tma_default)
    epi_tma_cfg = _cfg(FLASH_EPI_TMA_KEY)
    if epi_tma_cfg is not None:
        epi_tma = bool(epi_tma_cfg)
    epi_tma = epi_tma and topology == "fa4"
    epi_stg_default = False
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
    rescale_threshold_default = dtype_rescale_threshold_default
    rescale_threshold = float(
        os.environ.get("HELION_CUTE_FLASH_RESCALE_THRESHOLD", rescale_threshold_default)
    )
    rescale_threshold_cfg = _cfg(FLASH_RESCALE_THRESHOLD_KEY)
    if rescale_threshold_cfg is not None:
        rescale_threshold = float(rescale_threshold_cfg)  # type: ignore[arg-type]
    if not math.isfinite(rescale_threshold):
        rescale_threshold = rescale_threshold_default
    if rescale_threshold >= math.log2(torch.finfo(dtype).max):
        # A pinned score can be up to exp2(threshold) before the fp32
        # probabilities are cast to the I/O dtype. Reject stale/manual values
        # that can overflow that cast; the shape default is always safe.
        rescale_threshold = rescale_threshold_default
    skip_rescale_stats_default = False
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
    rescale_chunk_cols = 32 if head_dim == 64 else 16
    softmax_regs = 200
    corr_regs = 64
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
        valid_rescale_chunks = (8, 16, 32, 64) if head_dim == 64 else (8, 16, 32)
        if rescale_chunk_cols not in valid_rescale_chunks:
            rescale_chunk_cols = rescale_chunk_default
        softmax_regs = int(
            os.environ.get(
                "HELION_CUTE_FLASH_SOFTMAX_REGS",
                str(softmax_regs),
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
        other_regs_default = 48
        other_regs = int(
            os.environ.get("HELION_CUTE_FLASH_OTHER_REGS", str(other_regs_default))
        )
        other_regs_cfg = _cfg(FLASH_OTHER_REGS_KEY)
        if other_regs_cfg is not None:
            other_regs = int(other_regs_cfg)  # type: ignore[arg-type]
        if other_regs % 8 != 0 or other_regs < 24:
            other_regs = other_regs_default
        corr_tile_size_default = 8 if not is_causal and head_dim <= 64 else 16
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
        prefer_packed_reduce or dense_hd64_fa4 or causal_hd64_fa4,
    )
    packed_reduce_cfg = _cfg(FLASH_PACKED_REDUCE_KEY)
    if packed_reduce_cfg is not None:
        packed_reduce = bool(packed_reduce_cfg)
    small_biased = True
    small_biased_cfg = _cfg(FLASH_SMALL_BIASED_KEY)
    if small_biased_cfg is not None:
        small_biased = bool(small_biased_cfg)
    if not small_biased_candidate:
        small_biased = True
    causal_lpt_swizzle = int(
        os.environ.get(
            "HELION_CUTE_FLASH_CAUSAL_LPT_SWIZZLE",
            "0",
        )
    )
    causal_lpt_swizzle_cfg = _cfg(FLASH_CAUSAL_LPT_SWIZZLE_KEY)
    if causal_lpt_swizzle_cfg is not None:
        causal_lpt_swizzle = int(causal_lpt_swizzle_cfg)  # type: ignore[arg-type]
    if not is_causal:
        causal_lpt_swizzle = 0
    else:
        # Wider same-length CTA waves expose an unresolved, nondeterministic
        # long-running causal barrier race. Only the serial head ordering is
        # safe, so canonicalize both fresh and cached configs to it.
        causal_lpt_swizzle = 1
        if num_bh is not None:
            causal_lpt_swizzle = min(causal_lpt_swizzle, max(num_bh, 1))
    causal_kv_order_default = "ascending"
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
    causal_loop_split_default = False
    causal_loop_split = _flash_bool_env(
        "HELION_CUTE_FLASH_CAUSAL_LOOP_SPLIT", causal_loop_split_default
    )
    causal_loop_split_cfg = _cfg(FLASH_CAUSAL_LOOP_SPLIT_KEY)
    if causal_loop_split_cfg is not None:
        causal_loop_split = bool(causal_loop_split_cfg)
    if not is_causal or topology != "fa4" or causal_kv_order != "descending":
        causal_loop_split = False
    role_map_default = "helion"
    role_map = os.environ.get("HELION_CUTE_FLASH_ROLE_MAP", role_map_default)
    role_map_cfg = _cfg(FLASH_ROLE_MAP_KEY)
    if role_map_cfg is not None:
        role_map = str(role_map_cfg)
    if role_map not in ("helion", "fa4"):
        role_map = "helion"
    if topology != "fa4":
        role_map = "helion"
    separate_kv_rings = bool(
        requested_family_flags is not None
        and requested_family_flags.separate_kv_rings
        and topology == "fa4"
        and dtype in (torch.float16, torch.bfloat16)
        and _flash_deep_1cta_kv_stage_cap(head_dim) > 0
    )
    requested_causal_two_cta = bool(
        requested_family_flags is not None and requested_family_flags.causal_two_cta
    )
    causal_two_cta = bool(
        requested_causal_two_cta
        and is_causal
        and topology == "fa4"
        and head_dim == 64
        and dtype is torch.float16
        and num_kv % 4 == 0
    )
    if requested_family_flags is not None:
        use_2cta_instrs = requested_family_flags.use_2cta_instrs
    else:
        use_2cta_instrs = _flash_bool_env("HELION_CUTE_FLASH_USE_2CTA", False)
        use_2cta_cfg = _cfg(FLASH_USE_2CTA_KEY)
        if use_2cta_cfg is not None:
            use_2cta_instrs = bool(use_2cta_cfg)
    dense_hd64_2cta = (
        head_dim == 64
        and _flash_dense_hd64_2cta_num_kv_supported(num_kv)
        and _flash_supported_io_dtype(dtype)
    )
    if (
        topology != "fa4"
        or (is_causal and not causal_two_cta)
        or (requested_causal_two_cta and not causal_two_cta)
        or num_kv % 4 != 0
        or (head_dim != 128 and not dense_hd64_2cta and not causal_two_cta)
        or (separate_kv_rings and not causal_two_cta)
    ):
        use_2cta_instrs = False
        causal_two_cta = False
    if use_2cta_instrs:
        if head_dim == 128:
            precompute_qk_desc = False
        if epi_stg_gmem == "pair":
            # Pair destinations span both CTA rows; without a rank slice the two
            # CTAs alias the same output half.
            epi_stg_gmem = "stage"
    if causal_two_cta:
        # The causal cluster decoder currently uses contiguous cluster work.
        causal_lpt_swizzle = 1
    use_cga2_local_default = False
    use_cga2_local_cfg = _cfg(FLASH_CGA2_LOCAL_KEY)
    if requested_family_flags is not None:
        use_cga2_local_cta = requested_family_flags.use_cga2_local_cta
        use_cga2_local_overridden = True
    else:
        use_cga2_local_cta = _flash_bool_env(
            "HELION_CUTE_FLASH_CGA2_LOCAL", use_cga2_local_default
        )
        use_cga2_local_overridden = (
            "HELION_CUTE_FLASH_CGA2_LOCAL" in os.environ
            or use_cga2_local_cfg is not None
        )
        if use_cga2_local_cfg is not None:
            use_cga2_local_cta = bool(use_cga2_local_cfg)
    if (
        topology != "fa4"
        or is_causal
        or num_kv % 4 != 0
        or head_dim != 64
        or use_2cta_instrs
        or separate_kv_rings
    ):
        use_cga2_local_cta = False
    if use_cga2_local_cta and persistent and epi_stg_gmem == "pair":
        # CGA2-local still launches a two-CTA cluster. Pair destinations make
        # both ranks alias one output half when the persistent grid reuses work.
        epi_stg_gmem = "stage"
    use_clc_scheduler_default = False
    if requested_family_flags is not None:
        use_clc_scheduler = requested_family_flags.use_clc_scheduler
    else:
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
        or separate_kv_rings
    ):
        use_clc_scheduler = False
    if use_clc_scheduler and rescale_chunk_cols == 64:
        # The 64-column chunk expands the CLC softmax body beyond what NVVM can
        # compile on Blackwell. Preserve legacy configs via the safe D64 form.
        rescale_chunk_cols = 32
    clc_heads_per_batch_default = 0
    clc_heads_per_batch = int(
        os.environ.get("HELION_CUTE_FLASH_CLC_HEADS", str(clc_heads_per_batch_default))
    )
    clc_heads_per_batch_cfg = _cfg(FLASH_CLC_HEADS_PER_BATCH_KEY)
    if clc_heads_per_batch_cfg is not None:
        clc_heads_per_batch = int(clc_heads_per_batch_cfg)  # type: ignore[arg-type]
    if (
        clc_heads_per_batch < 0
        or not use_clc_scheduler
        or (
            num_bh is not None
            and clc_heads_per_batch > 0
            and num_bh % clc_heads_per_batch != 0
        )
    ):
        clc_heads_per_batch = 0
    clc_use_pdl = _flash_bool_env("HELION_CUTE_FLASH_CLC_PDL", False)
    clc_use_pdl_cfg = _cfg(FLASH_CLC_PDL_KEY)
    if clc_use_pdl_cfg is not None:
        clc_use_pdl = bool(clc_use_pdl_cfg)
    if not use_clc_scheduler:
        clc_use_pdl = False
    clc_stages = int(
        os.environ.get(
            "HELION_CUTE_FLASH_CLC_STAGES", "2" if use_clc_scheduler else "1"
        )
    )
    clc_stages_cfg = _cfg(FLASH_CLC_STAGES_KEY)
    if clc_stages_cfg is not None:
        clc_stages = int(clc_stages_cfg)  # type: ignore[arg-type]
    if not use_clc_scheduler:
        clc_stages = 1
    elif clc_stages not in (2, 3):
        # A one-stage CLC response pipeline can deadlock when several kernels
        # contend for the GPU. Two stages is performance-neutral and keeps the
        # producer from reusing the only response slot while roles retire it.
        clc_stages = 2
    if use_clc_scheduler:
        # CLC owns the grid residency policy; this field is consumed only by
        # the static-persistent launcher and must not create distinct aliases.
        persistent_ctas_per_sm = 1
    if (
        topology != "fa4"
        or not persistent
        or use_clc_scheduler
        or use_2cta_instrs
        or use_cga2_local_cta
    ):
        recompute_tile_coords = False
    local_tma_partition_default = False
    if requested_family_flags is not None:
        local_tma_partition = requested_family_flags.local_tma_partition
    else:
        local_tma_partition = _flash_bool_env(
            "HELION_CUTE_FLASH_LOCAL_TMA", local_tma_partition_default
        )
        local_tma_partition_cfg = _cfg(FLASH_LOCAL_TMA_PARTITION_KEY)
        if local_tma_partition_cfg is not None:
            local_tma_partition = bool(local_tma_partition_cfg)
    local_tma_partition = (
        local_tma_partition
        and topology == "fa4"
        and persistent
        and not use_2cta_instrs
        and not use_cga2_local_cta
        and not separate_kv_rings
    )
    tensor_4d_tma_default = False
    if requested_family_flags is not None:
        tensor_4d_tma = requested_family_flags.tensor_4d_tma
    else:
        tensor_4d_tma = _flash_bool_env(
            "HELION_CUTE_FLASH_TENSOR_4D", tensor_4d_tma_default
        )
        tensor_4d_tma_cfg = _cfg(FLASH_TENSOR_4D_TMA_KEY)
        if tensor_4d_tma_cfg is not None:
            tensor_4d_tma = bool(tensor_4d_tma_cfg)
    tensor_4d_tma = (
        tensor_4d_tma
        and supports_tensor_4d_tma
        and topology == "fa4"
        and not is_causal
        and head_dim == 64
        and dtype is torch.float16
        and not separate_kv_rings
    )
    if separate_kv_rings:
        # The one-CTA family isolates K/V depth. The causal cluster family may
        # compose the already-validated TMA epilogue.
        use_2cta_instrs = causal_two_cta
        use_cga2_local_cta = False
        use_clc_scheduler = False
        clc_heads_per_batch = 0
        clc_use_pdl = False
        clc_stages = 1
        local_tma_partition = False
        tensor_4d_tma = False
        recompute_tile_coords = False
        if not causal_two_cta:
            epi_tma = False
            epi_stg = False
            epi_stg_store = "slice"
            epi_stg_gmem = "stage"
        kv_stage = min(max(kv_stage, 2), _flash_deep_1cta_kv_stage_cap(head_dim))
    dense_degree1_eligible = (
        standard_dense_output
        and _flash_dense_hd64_2cta_num_kv_supported(num_kv)
        and use_2cta_instrs
        and not is_causal
        and head_dim == 64
        and dtype is torch.float16
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
    )
    causal_hd64_hybrid_eligible = _flash_causal_hd64_hybrid_family_eligible(
        head_dim,
        num_kv,
        dtype=dtype,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_causal_output=standard_causal_output,
    )
    exp2_packet_default = (
        "8x2"
        if use_2cta_instrs
        and not is_causal
        and head_dim == 64
        and _flash_supported_io_dtype(dtype)
        else "1x1"
    )
    exp2_packet = os.environ.get("HELION_CUTE_FLASH_EXP2_PACKET", exp2_packet_default)
    exp2_packet_cfg = _cfg(FLASH_EXP2_PACKET_KEY)
    if exp2_packet_cfg is not None:
        exp2_packet = str(exp2_packet_cfg)
    manual_exp2_common = (
        topology == "fa4"
        and (
            (head_dim == 64 and dtype is torch.float16)
            or (head_dim == 128 and dtype is torch.bfloat16)
            or causal_hd64_hybrid_eligible
        )
        and exp2_impl == "split"
        and q_tile_count == 2
        and not use_cga2_local_cta
        and not separate_kv_rings
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
        and p_store_repetition == 16
        and s_load_repetition == 32
    )
    manual_exp2_eligible = manual_exp2_common and (
        (
            standard_causal_output
            and is_causal
            and head_dim == 64
            and dtype is torch.float16
            and not use_2cta_instrs
            and softmax_disc
            and disc_pipe_depth >= 2
            and exp2_packet != _FLASH_HYBRID_EXP2_PACKET
            and exp2_packet not in _FLASH_DEG1_EXP2_PACKETS
            and exp2_packet not in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
        )
        or (
            causal_hd64_hybrid_eligible
            and not use_2cta_instrs
            and softmax_disc
            and disc_pipe_depth >= 2
            and split_p_arrive
            and causal_kv_order == "descending"
            and causal_loop_split
            and exp2_packet == _FLASH_HYBRID_EXP2_PACKET
        )
        or (
            not is_causal
            and head_dim == 64
            and dtype is torch.float16
            and use_2cta_instrs
            and not softmax_disc
            and dense_degree1_eligible
            and exp2_packet in (*_FLASH_DEG1_EXP2_PACKETS, _FLASH_DEG2_EXP2_PACKET)
        )
        or (
            standard_dense_output
            and not is_causal
            and use_2cta_instrs
            and softmax_disc
            and head_dim == 128
            and dtype is torch.bfloat16
            and _flash_dense_hd128_degree2_num_kv_supported(num_kv)
            and exp2_packet == _FLASH_DEG2_EXP2_PACKET
        )
        or (
            standard_causal_output
            and is_causal
            and head_dim == 128
            and dtype is torch.bfloat16
            and _flash_causal_hd128_resident_num_kv_supported(num_kv)
            and not use_2cta_instrs
            and softmax_disc
            and disc_pipe_depth == 2
            and split_p_arrive
            and causal_kv_order == "descending"
            and causal_loop_split
            and not use_clc_scheduler
            and not local_tma_partition
            and not tensor_4d_tma
            and exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
        )
    )
    if exp2_packet in _FLASH_MANUAL_EXP2_PACKET_PARAMS:
        if not manual_exp2_eligible:
            exp2_packet = exp2_packet_default
    elif exp2_packet not in _FLASH_EXP2_PACKET_PARAMS:
        exp2_packet = exp2_packet_default
    if (
        topology != "fa4"
        or exp2_impl != "split"
        or head_dim not in (64, 128)
        or (
            head_dim == 128
            and exp2_packet
            not in (
                _FLASH_DEG2_EXP2_PACKET,
                *_FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS,
            )
        )
    ):
        exp2_packet = "1x1"
    manual_exp2_schedule = _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.get(exp2_packet)
    if manual_exp2_schedule is not None:
        # Manual polynomial packets are measured compound schedules. Canonicalize
        # the cadence they replace so equivalent requests have one config identity;
        # the phase offsets remain active tuning dimensions.
        exp2_impl = "split"
        e2e_freq, e2e_res = manual_exp2_schedule
        e2e_schedule = f"{e2e_freq}/{e2e_res}"
        if is_causal:
            masked_e2e_freq = e2e_freq
            masked_e2e_res = e2e_res
            masked_e2e_schedule = e2e_schedule
        else:
            masked_e2e_freq = e2e_freq
            masked_e2e_res = e2e_res
            masked_e2e_schedule = "inherit"
        e2e_offset_period = e2e_freq
        e2e_offset_default = (
            _FLASH_DEG1_EXP2_OFFSET
            if exp2_packet in _FLASH_DEG1_EXP2_PACKETS
            else causal_e2e_offset_default % e2e_offset_period
        )
        e2e_offset = int(
            os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET", str(e2e_offset_default))
        )
        if e2e_offset_cfg is not None:
            e2e_offset = int(e2e_offset_cfg)  # type: ignore[arg-type]
        e2e_offset = _flash_normalize_e2e_offset(
            e2e_offset, e2e_offset_default, e2e_offset_period
        )
        e2e_offset0_default = (
            _FLASH_DEG1_EXP2_OFFSET0 if exp2_packet in _FLASH_DEG1_EXP2_PACKETS else 0
        )
        e2e_offset0 = int(
            os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET0", str(e2e_offset0_default))
        )
        if e2e_offset0_cfg is not None:
            e2e_offset0 = int(e2e_offset0_cfg)  # type: ignore[arg-type]
        e2e_offset0 = _flash_normalize_e2e_offset(
            e2e_offset0, e2e_offset0_default, e2e_offset_period
        )
    if is_causal and (not causal_loop_split or not standard_causal_output):
        # A runtime causal loop can alternate masked and unmasked tiles. Its
        # shared PASS2 pipeline has one static cadence, so distinct cadences are
        # safe only when codegen proves separate loop regions.
        masked_e2e_schedule = "inherit"
        masked_e2e_freq = e2e_freq
        masked_e2e_res = e2e_res
        e2e_offset_period = _flash_e2e_offset_period(
            e2e_freq,
            e2e_res,
            masked_e2e_freq,
            masked_e2e_res,
        )
        e2e_offset = _flash_normalize_e2e_offset(
            e2e_offset, e2e_offset_default, e2e_offset_period
        )
        e2e_offset0 = _flash_normalize_e2e_offset(
            e2e_offset0, e2e_offset0_default, e2e_offset_period
        )
    if p_store_repetition == 32 and split_p_arrive:
        # The mixed Rep32/Rep16 publication helper owns a fixed PASS2 pipeline
        # and does not consume the generic pipeline-depth argument.
        disc_pipe_depth = 1
    stat_transport_eligible = topology == "fa4" and not is_causal and head_dim == 64
    legacy_stat_handoff = _flash_bool_env("HELION_CUTE_FLASH_FA4_STAT_HANDOFF", True)
    stat_transport_default = (
        "single" if stat_transport_eligible and legacy_stat_handoff else "ring2"
    )
    stat_transport = os.environ.get(
        "HELION_CUTE_FLASH_STAT_TRANSPORT", stat_transport_default
    )
    stat_transport_cfg = _cfg(FLASH_STAT_TRANSPORT_KEY)
    if stat_transport_cfg is not None:
        stat_transport = str(stat_transport_cfg)
    if stat_transport not in ("ring2", "single", "single_final"):
        stat_transport = stat_transport_default
    if not stat_transport_eligible:
        stat_transport = "ring2"
    single_final_stat_eligible = (
        stat_transport_eligible
        and exp2_impl == "split"
        and standard_dense_output
        and (not persistent or use_2cta_instrs)
        and not softmax_disc
        and rescale_threshold > 0.0
        and exp2_packet != _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET
    )
    if stat_transport == "single_final" and not single_final_stat_eligible:
        stat_transport = "single"
    if (
        stat_transport_eligible
        and not softmax_disc
        and stat_transport == "ring2"
        and mma_ptx
    ):
        # Whole-row PTX softmax can let the two-slot stats ring lap its consumer,
        # including without split-P publication. Use the established single-slot
        # acknowledgement for both split and hardware-XU exp2. The CuTe-GEMM
        # fallback retains its ring2 handoff.
        stat_transport = "single"
    if topology == "fa4" and not softmax_disc and stat_transport == "ring2":
        # The two-slot statistics handoff can lap a K/V pipeline deeper than two
        # stages after enough KV iterations, leaving correction and
        # softmax waiting on opposite barrier phases. Single-slot handoff has an
        # explicit acknowledgement and can use deeper rings; ring2 whole-row
        # schedules must retain the proven two-stage depth.
        kv_stage = 2
    if topology == "fa4":
        # FA4 has a fixed two-query score pipeline. The generic WS s-stage and
        # row-reduction choices do not participate in its emitted source.
        s_stage = 2
        packed_reduce = True
        if not softmax_disc:
            disc_pipe_depth = 1
    else:
        # Canonicalize FA4-only children so WS candidates that emit the same
        # source also have the same normalized config identity.
        disc_pipe_depth = 1
        e2e_offset = 0
        e2e_offset0 = 0
        causal_lpt_swizzle = 0
    if topology == "fa4" and not separate_kv_rings and head_dim in (64, 128):
        aliased_kv_stage_cap = _flash_aliased_kv_stage_cap(
            head_dim,
            stage_output=epi_tma or epi_stg,
        )
        kv_stage = min(max(kv_stage, 2), aliased_kv_stage_cap)
    pipeline_family = _flash_pipeline_family_from_flags(
        topology=topology,
        separate_kv_rings=separate_kv_rings,
        causal_two_cta=causal_two_cta,
        use_2cta_instrs=use_2cta_instrs,
        use_cga2_local_cta=use_cga2_local_cta,
        use_clc_scheduler=use_clc_scheduler,
        local_tma_partition=local_tma_partition,
        tensor_4d_tma=tensor_4d_tma,
    )
    role_chain_default = _flash_role_chain_default(
        hd=head_dim, is_causal=is_causal, role_map=role_map
    ) or (use_2cta_instrs and not is_causal and head_dim == 64)
    role_chain = _flash_bool_env("HELION_CUTE_FLASH_ROLE_CHAIN", role_chain_default)
    role_chain_cfg = _cfg(FLASH_ROLE_CHAIN_KEY)
    if role_chain_cfg is not None:
        role_chain = bool(role_chain_cfg)
    if (
        topology != "fa4"
        or is_causal
        or head_dim != 64
        or not _flash_supported_io_dtype(dtype)
        or use_cga2_local_cta
        or use_clc_scheduler
    ):
        role_chain = False

    persistent_loop = os.environ.get("HELION_CUTE_FLASH_PERSISTENT_LOOP", "while")
    persistent_loop_cfg = _cfg(FLASH_PERSISTENT_LOOP_KEY)
    if persistent_loop_cfg is not None:
        persistent_loop = str(persistent_loop_cfg)
    if persistent_loop not in ("while", "counted") or not (
        topology == "fa4" and persistent and not use_clc_scheduler
    ):
        persistent_loop = "while"

    sp_row_sum = os.environ.get("HELION_CUTE_FLASH_SP_ROW_SUM", "fragment")
    sp_row_sum_cfg = _cfg(FLASH_SP_ROW_SUM_KEY)
    if sp_row_sum_cfg is not None:
        sp_row_sum = str(sp_row_sum_cfg)
    sp_row_sum_eligible = (
        topology == "fa4"
        and not softmax_disc
        and not is_causal
        and head_dim == 64
        and dtype is torch.float16
    )
    if sp_row_sum not in ("fragment", "whole") or not sp_row_sum_eligible:
        sp_row_sum = "fragment"

    softmax_setup = (
        "stage_local"
        if _flash_bool_env("HELION_CUTE_FLASH_STAGE_LOCAL_SOFTMAX_SETUP", False)
        else "shared"
    )
    softmax_setup_cfg = _cfg(FLASH_SOFTMAX_SETUP_KEY)
    if softmax_setup_cfg is not None:
        softmax_setup = str(softmax_setup_cfg)
    softmax_setup_eligible = (
        topology == "fa4"
        and not is_causal
        and head_dim == 64
        and dtype is torch.float16
        and p_store_repetition == 16
    )
    if softmax_setup not in ("shared", "stage_local") or not softmax_setup_eligible:
        softmax_setup = "shared"
    if (
        softmax_setup == "stage_local"
        and use_cga2_local_cta
        and persistent
        and softmax_disc
        and stat_transport == "single"
        and rescale_threshold <= 0.0
    ):
        # This long-lived CGA2-local combination has wedged a persistent kernel.
        # Keep the arithmetic policy intact and use the established shared
        # descriptor setup.
        softmax_setup = "shared"

    epi_tma_setup = os.environ.get("HELION_CUTE_FLASH_EPI_TMA_SETUP", "shared")
    epi_tma_setup_cfg = _cfg(FLASH_EPI_TMA_SETUP_KEY)
    if epi_tma_setup_cfg is not None:
        epi_tma_setup = str(epi_tma_setup_cfg)
    epi_tma_setup_eligible = (
        topology == "fa4"
        and epi_tma
        and not is_causal
        and head_dim == 64
        and standard_dense_output
    )
    if epi_tma_setup not in ("shared", "role_local") or not epi_tma_setup_eligible:
        epi_tma_setup = "shared"

    if exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS and (
        pipeline_family != "fa4" or not causal_loop_split
    ):
        exp2_packet = "1x1"
    return FlashAttentionConfig(
        s_stage=s_stage,
        kv_stage=kv_stage,
        persistent=persistent,
        persistent_ctas_per_sm=persistent_ctas_per_sm,
        recompute_tile_coords=recompute_tile_coords,
        num_regs_producer=_FLASH_NUM_REGS_PRODUCER,
        num_regs_consumer=_FLASH_NUM_REGS_CONSUMER,
        persistent_loop=persistent_loop,
        sp_row_sum=sp_row_sum,
        softmax_setup=softmax_setup,
        epi_tma_setup=epi_tma_setup,
        topology=topology,
        pipeline_family=pipeline_family,
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
        exp2_packet=exp2_packet,
        tmem_plan=tmem_plan,
        tmem_s_to_p_offset=tmem_s_to_p_offset,
        mma_interleave=mma_interleave,
        wait_hint=wait_hint,
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
        stat_transport=stat_transport,
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
        role_chain=role_chain,
        separate_kv_rings=separate_kv_rings,
        causal_two_cta=causal_two_cta,
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
# Autotune surface for the flash-attention config.
#
# Mirrors the ``Tcgen05WarpSpec`` pattern in ``strategies.py`` /
# ``tcgen05_config.py``. Independent arithmetic fields retain individual keys;
# structural scheduling is represented by one compound pipeline-family key.
# ``FLASH_CONFIG_KEYS`` aggregates the active and legacy inputs, and
# ``flash_config_from_config`` reconstructs the effective dataclass.
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
FLASH_STAT_TRANSPORT_KEY = "cute_flash_stat_transport"
FLASH_WAIT_HINT_KEY = "cute_flash_wait_hint"
FLASH_EXP2_PACKET_KEY = "cute_flash_exp2_packet"
# Compound schedule family. Legacy topology/cluster/TMA keys normalize into it.
FLASH_PIPELINE_FAMILY_KEY = "cute_flash_pipeline_family"
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
FLASH_ROLE_CHAIN_KEY = "cute_flash_role_chain"
FLASH_USE_2CTA_KEY = "cute_flash_use_2cta"
FLASH_CGA2_LOCAL_KEY = "cute_flash_cga2_local"
FLASH_CLC_KEY = "cute_flash_clc"
FLASH_CLC_HEADS_PER_BATCH_KEY = "cute_flash_clc_heads_per_batch"
FLASH_CLC_PDL_KEY = "cute_flash_clc_pdl"
FLASH_CLC_STAGES_KEY = "cute_flash_clc_stages"
FLASH_LOCAL_TMA_PARTITION_KEY = "cute_flash_local_tma_partition"
FLASH_TENSOR_4D_TMA_KEY = "cute_flash_tensor_4d_tma"
FLASH_CAUSAL_LOOP_SPLIT_KEY = "cute_flash_causal_loop_split"
FLASH_PERSISTENT_LOOP_KEY = "cute_flash_persistent_loop"
FLASH_SP_ROW_SUM_KEY = "cute_flash_sp_row_sum"
FLASH_SOFTMAX_SETUP_KEY = "cute_flash_softmax_setup"
FLASH_EPI_TMA_SETUP_KEY = "cute_flash_epi_tma_setup"


class FlashStructuralLeaf(NamedTuple):
    """Exact structural schedule ranked by hierarchical flash tuning.

    Ordinary exp2 packets are arithmetic children of a pipeline family. A
    compound packet owns additional scheduling requirements, so each such
    packet is a distinct leaf under its normalized parent family.  The
    whole-row and chunked softmax implementations use different barrier and
    statistics-transport protocols, so they must also be qualified as
    separate leaves instead of competing as an ordinary Boolean child.
    """

    pipeline_family: str
    compound_exp2_packet: str | None
    softmax_disc: bool = True


def flash_structural_leaf_from_config(
    config: Mapping[str, object],
) -> FlashStructuralLeaf | None:
    """Return the exact normalized structural leaf represented by ``config``."""
    family = config.get(FLASH_PIPELINE_FAMILY_KEY)
    if not isinstance(family, str) or family not in FLASH_PIPELINE_FAMILY_FLAGS:
        return None
    packet = config.get(FLASH_EXP2_PACKET_KEY)
    return FlashStructuralLeaf(
        family,
        cast("str", packet) if flash_exp2_packet_is_compound(packet) else None,
        bool(config.get(FLASH_SOFTMAX_DISC_KEY, True)),
    )


# These fields form one staged-output schedule. Marginal coverage is insufficient:
# all effective combinations emit different producer/consumer code.
FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        FLASH_EPI_TMA_KEY,
        FLASH_EPI_STG_KEY,
        FLASH_EPI_STG_STORE_KEY,
        FLASH_EPI_STG_GMEM_KEY,
    ),
)

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
    FLASH_EXP2_PACKET_KEY,
    FLASH_MMA_INTERLEAVE_KEY,
    FLASH_WAIT_HINT_KEY,
    FLASH_STAT_TRANSPORT_KEY,
    FLASH_PIPELINE_FAMILY_KEY,
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
    FLASH_ROLE_CHAIN_KEY,
    FLASH_CLC_HEADS_PER_BATCH_KEY,
    FLASH_CLC_PDL_KEY,
    FLASH_CLC_STAGES_KEY,
    FLASH_CAUSAL_LOOP_SPLIT_KEY,
    FLASH_PERSISTENT_LOOP_KEY,
    FLASH_SP_ROW_SUM_KEY,
    FLASH_SOFTMAX_SETUP_KEY,
    FLASH_EPI_TMA_SETUP_KEY,
)

FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_TOPOLOGY_KEY,
    FLASH_USE_2CTA_KEY,
    FLASH_CGA2_LOCAL_KEY,
    FLASH_CLC_KEY,
    FLASH_LOCAL_TMA_PARTITION_KEY,
    FLASH_TENSOR_4D_TMA_KEY,
)

FLASH_LEGACY_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_EXP2_IMPL_KEY,
    FLASH_E2E_FREQ_KEY,
    FLASH_E2E_RES_KEY,
    *FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS,
)

# Accepted as an input and retained in normalized configs, but not sampled:
# changing the local query-slot count requires selecting another structural
# family rather than mutating an arithmetic child.
FLASH_DERIVED_CONFIG_KEYS: tuple[str, ...] = (FLASH_Q_TILE_COUNT_KEY,)

FLASH_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_AUTOTUNE_CONFIG_KEYS + FLASH_LEGACY_CONFIG_KEYS + FLASH_DERIVED_CONFIG_KEYS
)


def flash_effective_config_values(
    config: FlashAttentionConfig,
) -> dict[str, object]:
    """Project a resolved flash config onto the active autotune schema."""
    return {
        FLASH_S_STAGE_KEY: config.s_stage,
        FLASH_KV_STAGE_KEY: config.kv_stage,
        FLASH_PERSISTENT_KEY: config.persistent,
        FLASH_PERSISTENT_CTAS_PER_SM_KEY: config.persistent_ctas_per_sm,
        FLASH_RECOMPUTE_TILE_COORDS_KEY: config.recompute_tile_coords,
        FLASH_E2E_SCHEDULE_KEY: config.e2e_schedule,
        FLASH_MASKED_E2E_SCHEDULE_KEY: config.masked_e2e_schedule,
        FLASH_E2E_OFFSET_KEY: config.e2e_offset,
        FLASH_E2E_OFFSET0_KEY: config.e2e_offset0,
        FLASH_EXP2_PACKET_KEY: config.exp2_packet,
        FLASH_MMA_INTERLEAVE_KEY: config.mma_interleave,
        FLASH_Q_TILE_COUNT_KEY: config.q_tile_count,
        FLASH_WAIT_HINT_KEY: config.wait_hint,
        FLASH_STAT_TRANSPORT_KEY: config.stat_transport,
        FLASH_PIPELINE_FAMILY_KEY: config.pipeline_family,
        FLASH_SOFTMAX_DISC_KEY: config.softmax_disc,
        FLASH_DISC_PIPE_KEY: config.disc_pipe_depth,
        FLASH_SPLIT_P_ARRIVE_KEY: config.split_p_arrive,
        FLASH_P_STORE_REP_KEY: config.p_store_repetition,
        FLASH_S_LOAD_REP_KEY: config.s_load_repetition,
        FLASH_PRECOMPUTE_QK_DESC_KEY: config.precompute_qk_desc,
        FLASH_FIRST_LOAD_ORDER_KEY: config.first_load_order,
        FLASH_KV_ORDER_KEY: config.kv_order,
        FLASH_EPI_TMA_KEY: config.epi_tma,
        FLASH_EPI_STG_KEY: config.epi_stg,
        FLASH_EPI_STG_STORE_KEY: config.epi_stg_store,
        FLASH_EPI_STG_GMEM_KEY: config.epi_stg_gmem,
        FLASH_RESCALE_THRESHOLD_KEY: config.rescale_threshold,
        FLASH_SKIP_RESCALE_STATS_KEY: config.skip_rescale_stats,
        FLASH_RESCALE_CHUNK_COLS_KEY: config.rescale_chunk_cols,
        FLASH_SOFTMAX_REGS_KEY: config.softmax_regs,
        FLASH_CORR_REGS_KEY: config.corr_regs,
        FLASH_OTHER_REGS_KEY: config.other_regs,
        FLASH_CORR_TILE_SIZE_KEY: config.corr_tile_size,
        FLASH_PACKED_REDUCE_KEY: config.packed_reduce,
        FLASH_SMALL_BIASED_KEY: config.small_biased,
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: config.causal_lpt_swizzle,
        FLASH_CAUSAL_KV_ORDER_KEY: config.causal_kv_order,
        FLASH_ROLE_MAP_KEY: config.role_map,
        FLASH_ROLE_CHAIN_KEY: config.role_chain,
        FLASH_CLC_HEADS_PER_BATCH_KEY: config.clc_heads_per_batch,
        FLASH_CLC_PDL_KEY: config.clc_use_pdl,
        FLASH_CLC_STAGES_KEY: config.clc_stages,
        FLASH_CAUSAL_LOOP_SPLIT_KEY: config.causal_loop_split,
        FLASH_PERSISTENT_LOOP_KEY: config.persistent_loop,
        FLASH_SP_ROW_SUM_KEY: config.sp_row_sum,
        FLASH_SOFTMAX_SETUP_KEY: config.softmax_setup,
        FLASH_EPI_TMA_SETUP_KEY: config.epi_tma_setup,
    }


def _flash_choices_with_default(default: _T, choices: Iterable[_T]) -> tuple[_T, ...]:
    ordered = [default]
    for choice in choices:
        if choice not in ordered:
            ordered.append(choice)
    return tuple(ordered)


def _flash_clc_heads_per_batch_candidates(num_bh: int | None) -> tuple[int, ...]:
    """Return non-aliasing CLC decompositions for the detected work grid."""
    if num_bh is None or num_bh <= 0:
        return (1, 2, 4, 8, 16, 32, 64)

    divisors: set[int] = set()
    for lower in range(1, math.isqrt(num_bh) + 1):
        if num_bh % lower == 0:
            divisors.add(lower)
            divisors.add(num_bh // lower)
    return tuple(sorted(divisors))


def _flash_clc_heads_per_batch_coverage_candidates(
    num_bh: int | None,
    *,
    tensor_4d_heads: int | None = None,
    effective_default: int | None = None,
    limit: int = 8,
) -> tuple[int, ...]:
    """Bound deterministic CLC coverage while preserving input geometry."""
    candidates = _flash_clc_heads_per_batch_candidates(num_bh)
    if effective_default is not None and effective_default not in candidates:
        raise ValueError("CLC effective default must be a legal decomposition")
    if len(candidates) <= limit:
        return candidates

    mandatory = {candidates[0], candidates[-1]}
    if effective_default is not None:
        mandatory.add(effective_default)
    if (
        num_bh is not None
        and tensor_4d_heads is not None
        and tensor_4d_heads > 0
        and num_bh % tensor_4d_heads == 0
    ):
        mandatory.add(tensor_4d_heads)
        mandatory.add(num_bh // tensor_4d_heads)
    if len(mandatory) > limit:
        raise ValueError("CLC coverage limit cannot hold the mandatory decompositions")

    selected = set(mandatory)
    candidate_logs = {candidate: math.log(candidate) for candidate in candidates}
    while len(selected) < limit:
        remaining = [candidate for candidate in candidates if candidate not in selected]
        if not remaining:
            break
        selected.add(
            max(
                remaining,
                key=lambda candidate: (
                    min(
                        abs(candidate_logs[candidate] - candidate_logs[chosen])
                        for chosen in selected
                    ),
                    -candidate,
                ),
            )
        )
    return tuple(candidate for candidate in candidates if candidate in selected)


def _flash_causal_lpt_candidates() -> tuple[int, ...]:
    """Return causal launch-order groups in the stress-tested safe envelope."""
    return (1,)


_FLASH_SEED_BLOCK_SIZE_TARGETS = (1, 128, 128)
_FLASH_RESCALE_THRESHOLD_VALUES = (0.0, 4.0, 8.0, 12.0, 16.0, 32.0)


def _flash_safe_rescale_threshold_values(dtype: torch.dtype) -> tuple[float, ...]:
    """Return thresholds whose largest pinned probability fits in ``dtype``."""
    max_log2 = math.log2(torch.finfo(dtype).max)
    return tuple(
        threshold
        for threshold in _FLASH_RESCALE_THRESHOLD_VALUES
        if threshold < max_log2
    )


def _flash_role_chain_default(*, hd: int, is_causal: bool, role_map: str) -> bool:
    """Make dense role chaining follow the autotuned role-map choice."""
    return not is_causal and hd == 64 and role_map == "fa4"


def _flash_seed_block_sizes(
    block_size_targets: Sequence[int],
) -> list[int] | None:
    targets = tuple(block_size_targets)
    if targets != _FLASH_SEED_BLOCK_SIZE_TARGETS:
        return None
    return [*targets]


def _flash_seed_values(
    head_dim: int,
    num_kv: int,
    *,
    num_bh: int | None,
    tensor_4d_heads: int | None,
    dtype: torch.dtype,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    supports_tensor_4d_tma: bool,
    pipeline_family_override: str | None = None,
) -> dict[str, object]:
    fragments = flash_autotune_fragments(
        head_dim,
        num_kv,
        num_bh=num_bh,
        tensor_4d_heads=tensor_4d_heads,
        dtype=dtype,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        pipeline_family_override=pipeline_family_override,
    )
    return {key: fragment.default() for key, fragment in fragments.items()}


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


def _flash_config_with_values(
    block_sizes: Sequence[int],
    values: Mapping[str, object],
) -> Config:
    return Config.from_dict({"block_sizes": list(block_sizes), **values})


def _flash_causal_degree2_template_values(
    *,
    e2e_offset: int,
    e2e_offset0: int,
    role_map: str,
    epi_tma: bool,
) -> dict[str, object]:
    """Return explicit values for the shared causal degree-2 seed template."""
    return {
        FLASH_PIPELINE_FAMILY_KEY: "fa4",
        FLASH_E2E_SCHEDULE_KEY: "16/6",
        FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
        FLASH_E2E_OFFSET_KEY: e2e_offset,
        FLASH_E2E_OFFSET0_KEY: e2e_offset0,
        FLASH_EXP2_PACKET_KEY: _FLASH_DEG2_EXP2_PACKET,
        FLASH_WAIT_HINT_KEY: 0,
        FLASH_DISC_PIPE_KEY: 3,
        FLASH_EPI_TMA_KEY: epi_tma,
        FLASH_RESCALE_CHUNK_COLS_KEY: 16,
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: 1,
        FLASH_ROLE_MAP_KEY: role_map,
    }


def _flash_dense_tuning_overrides(
    policy: FlashDenseTuningPolicy,
) -> dict[str, object]:
    values: dict[str, object] = {
        FLASH_PIPELINE_FAMILY_KEY: policy.pipeline_family,
        FLASH_KV_STAGE_KEY: policy.kv_stage,
        FLASH_PERSISTENT_KEY: policy.persistent,
        FLASH_E2E_SCHEDULE_KEY: policy.e2e_schedule,
        FLASH_E2E_OFFSET_KEY: policy.e2e_offset,
        FLASH_E2E_OFFSET0_KEY: policy.e2e_offset0,
        FLASH_EXP2_PACKET_KEY: policy.exp2_packet,
        FLASH_STAT_TRANSPORT_KEY: policy.stat_transport,
        FLASH_RESCALE_THRESHOLD_KEY: policy.rescale_threshold,
    }
    optional_values = {
        FLASH_EPI_TMA_KEY: policy.epi_tma,
        FLASH_KV_ORDER_KEY: policy.kv_order,
        FLASH_PRECOMPUTE_QK_DESC_KEY: policy.precompute_qk_desc,
        FLASH_RESCALE_CHUNK_COLS_KEY: policy.rescale_chunk_cols,
        FLASH_FIRST_LOAD_ORDER_KEY: policy.first_load_order,
        FLASH_CORR_TILE_SIZE_KEY: policy.corr_tile_size,
        FLASH_ROLE_MAP_KEY: policy.role_map,
        FLASH_SOFTMAX_REGS_KEY: policy.softmax_regs,
        FLASH_SPLIT_P_ARRIVE_KEY: policy.split_p_arrive,
        FLASH_SOFTMAX_DISC_KEY: policy.softmax_disc,
        FLASH_DISC_PIPE_KEY: policy.disc_pipe_depth,
        FLASH_SP_ROW_SUM_KEY: policy.sp_row_sum,
    }
    values.update(
        {key: value for key, value in optional_values.items() if value is not None}
    )
    if policy.corr_regs is not None:
        values[FLASH_CORR_REGS_KEY] = policy.corr_regs
    if policy.other_regs is not None:
        values[FLASH_OTHER_REGS_KEY] = policy.other_regs
    return values


def _flash_causal_tuning_overrides(
    policy: FlashCausalTuningPolicy,
) -> dict[str, object]:
    if policy.seed_template is not FlashCausalSeedTemplate.DEGREE2_V1:
        raise AssertionError(
            f"unsupported causal seed template: {policy.seed_template!r}"
        )
    overrides = {
        **_flash_causal_degree2_template_values(
            e2e_offset=policy.e2e_offset,
            e2e_offset0=policy.e2e_offset0,
            role_map=policy.role_map,
            epi_tma=policy.epi_tma,
        ),
        FLASH_KV_STAGE_KEY: policy.kv_stage,
        FLASH_CAUSAL_LOOP_SPLIT_KEY: policy.causal_loop_split,
        FLASH_CAUSAL_KV_ORDER_KEY: policy.causal_kv_order,
    }
    if policy.softmax_regs is not None:
        overrides[FLASH_SOFTMAX_REGS_KEY] = policy.softmax_regs
    if policy.first_load_order is not None:
        overrides[FLASH_FIRST_LOAD_ORDER_KEY] = policy.first_load_order
    return overrides


def _flash_config_matches_tuning_values(
    cfg: FlashAttentionConfig,
    expected: Mapping[str, object],
) -> bool:
    actual = flash_effective_config_values(cfg)
    return all(actual.get(key) == value for key, value in expected.items())


def _flash_dense_target_seed_matches(
    cfg: FlashAttentionConfig,
    policy: FlashDenseTuningPolicy | None,
) -> bool:
    """Return whether ``cfg`` is the validated target-promoted dense seed."""
    if policy is None:
        return False
    return _flash_config_matches_tuning_values(
        cfg,
        {
            **_flash_dense_tuning_overrides(policy),
            FLASH_Q_TILE_COUNT_KEY: 2,
        },
    )


def _flash_resident_softmax_config(
    requested: FlashAttentionConfig,
) -> FlashAttentionConfig:
    """Return the effective config for the resident causal softmax lowering."""
    return dataclasses.replace(
        requested,
        softmax_disc=False,
        disc_pipe_depth=1,
        exp2_impl="xu",
        e2e_freq=8,
        e2e_res=0,
        e2e_schedule="xu",
        masked_e2e_schedule="xu",
        masked_e2e_freq=8,
        masked_e2e_res=0,
        e2e_offset=0,
        e2e_offset0=0,
        exp2_packet="1x1",
        stat_transport="single",
    )


def _flash_causal_resident_native_seed_matches(
    cfg: FlashAttentionConfig,
    policy: FlashCausalTuningPolicy | None,
) -> bool:
    """Return whether ``cfg`` is the validated causal resident seed shape."""
    return policy is not None and _flash_config_matches_tuning_values(
        cfg,
        {
            **_flash_causal_tuning_overrides(policy),
            FLASH_Q_TILE_COUNT_KEY: 2,
        },
    )


def _flash_validated_target_seed(
    *,
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    num_bh: int | None,
    is_causal: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    supports_tensor_4d_tma: bool,
    values: Mapping[str, object],
    expected: Mapping[str, object],
) -> Config:
    """Build a target seed and reject policies normalized by config resolution."""
    seed = Config.from_dict(dict(values))
    resolved = resolve_flash_config(
        head_dim,
        num_kv,
        seed.config,
        dtype=dtype,
        num_bh=num_bh,
        is_causal=is_causal,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
    )
    actual = flash_effective_config_values(resolved)
    mismatches = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "flash target tuning policy does not round-trip through config "
            f"resolution: {mismatches!r}"
        )
    return seed


def _flash_target_seed_config(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    num_bh: int | None,
    tensor_4d_heads: int | None,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    target_device_capability: tuple[int, int] | None,
    supports_tensor_4d_tma: bool,
    block_size_targets: Sequence[int],
) -> Config | None:
    target_policy = get_flash_target_policy(target_device_capability)
    tuning_policy = target_policy.tuning_for_torch(
        head_dim, str(dtype).removeprefix("torch.")
    )
    if (
        tuning_policy is None
        or has_kv_tile_pruning
        or requires_ws_overlap
        or small_biased_candidate
    ):
        return None

    if is_causal:
        causal_policy = tuning_policy.causal_policy(num_kv)
        if not standard_causal_output or causal_policy is None:
            return None
        block_sizes = _flash_seed_block_sizes(block_size_targets)
        if block_sizes is None:
            return None
        causal_overrides = _flash_causal_tuning_overrides(causal_policy)
        pipeline_family = cast("str", causal_overrides[FLASH_PIPELINE_FAMILY_KEY])
        fragments = flash_autotune_fragments(
            head_dim,
            num_kv,
            num_bh=num_bh,
            tensor_4d_heads=tensor_4d_heads,
            dtype=dtype,
            is_causal=True,
            has_kv_tile_pruning=False,
            requires_ws_overlap=False,
            small_biased_candidate=False,
            standard_causal_output=True,
            target_device_capability=target_device_capability,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
            pipeline_family_override=pipeline_family,
        )
        values = {key: fragment.default() for key, fragment in fragments.items()}
        if not _flash_seed_set_all(values, fragments, causal_overrides):
            return None
        expected = {
            **causal_overrides,
            FLASH_Q_TILE_COUNT_KEY: 2,
        }
        return _flash_validated_target_seed(
            head_dim=head_dim,
            num_kv=num_kv,
            dtype=dtype,
            num_bh=num_bh,
            is_causal=True,
            standard_dense_output=False,
            standard_causal_output=True,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
            values={"block_sizes": block_sizes, **values, FLASH_Q_TILE_COUNT_KEY: 2},
            expected=expected,
        )

    dense_policy = tuning_policy.dense_policy(num_kv)
    if not standard_dense_output or dense_policy is None:
        return None
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None
    fragments = flash_autotune_fragments(
        head_dim,
        num_kv,
        num_bh=num_bh,
        tensor_4d_heads=tensor_4d_heads,
        dtype=dtype,
        is_causal=False,
        has_kv_tile_pruning=False,
        requires_ws_overlap=False,
        small_biased_candidate=False,
        standard_dense_output=True,
        target_device_capability=target_device_capability,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        pipeline_family_override=dense_policy.pipeline_family,
    )
    values = {key: fragment.default() for key, fragment in fragments.items()}
    overrides = _flash_dense_tuning_overrides(dense_policy)
    if not _flash_seed_set_all(values, fragments, overrides):
        return None
    expected = {
        **overrides,
        FLASH_Q_TILE_COUNT_KEY: 2,
    }
    return _flash_validated_target_seed(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        num_bh=num_bh,
        is_causal=False,
        standard_dense_output=True,
        standard_causal_output=False,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        values={"block_sizes": block_sizes, **values, FLASH_Q_TILE_COUNT_KEY: 2},
        expected=expected,
    )


def flash_attention_seed_config(
    head_dim: int,
    num_kv: int | None,
    *,
    num_bh: int | None = None,
    tensor_4d_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    supports_tensor_4d_tma: bool = True,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
    seed_kind: str = "default",
) -> Config | None:
    """Return a legal measured starting point for the flash search.

    Target-specific seeds may depend on length, but they only guide generation
    zero. The legality-driven search surface remains independent of a winner
    table, and every seed is benchmarked like any other candidate.
    """

    if num_kv is None:
        return None
    if seed_kind == "default":
        target_seed = _flash_target_seed_config(
            head_dim,
            num_kv,
            dtype=dtype,
            num_bh=num_bh,
            tensor_4d_heads=tensor_4d_heads,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            standard_dense_output=standard_dense_output,
            standard_causal_output=standard_causal_output,
            target_device_capability=target_device_capability,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
            block_size_targets=block_size_targets,
        )
        if target_seed is not None:
            return target_seed
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return None
    values = _flash_seed_values(
        head_dim,
        num_kv,
        num_bh=num_bh,
        tensor_4d_heads=tensor_4d_heads,
        dtype=dtype,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
    )
    if seed_kind == "default":
        return _flash_config_with_values(block_sizes, values)
    if seed_kind == "causal_lpt":
        if not is_causal or values[FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap":
            return None
        values[FLASH_CAUSAL_LPT_SWIZZLE_KEY] = 1
    if seed_kind == "causal_split":
        if not is_causal or values[FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap":
            return None
        values[FLASH_CAUSAL_KV_ORDER_KEY] = "descending"
        values[FLASH_CAUSAL_LOOP_SPLIT_KEY] = True
    if seed_kind == "dense_sp":
        if (
            is_causal
            or head_dim != 64
            or values[FLASH_PIPELINE_FAMILY_KEY] == "ws_overlap"
        ):
            return None
        values[FLASH_SOFTMAX_DISC_KEY] = False
        if dtype is torch.float16:
            values[FLASH_SP_ROW_SUM_KEY] = "whole"
    elif seed_kind not in ("causal_lpt", "causal_split"):
        raise AssertionError(f"unknown flash attention seed kind: {seed_kind!r}")
    return _flash_config_with_values(block_sizes, values)


def flash_attention_seed_configs(
    head_dim: int,
    num_kv: int | None,
    *,
    num_bh: int | None = None,
    tensor_4d_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    supports_tensor_4d_tma: bool = True,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
) -> tuple[Config, ...]:
    """Return measured starting points plus generic structural coverage.

    A target policy may contribute the first candidate. The remaining seeds
    cover every legal family and compound packet from fragment defaults. Every
    candidate is subsequently measured; none is selected without benchmarking.
    """

    if num_kv is None:
        return ()
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return ()

    common = {
        "num_bh": num_bh,
        "tensor_4d_heads": tensor_4d_heads,
        "dtype": dtype,
        "is_causal": is_causal,
        "has_kv_tile_pruning": has_kv_tile_pruning,
        "requires_ws_overlap": requires_ws_overlap,
        "small_biased_candidate": small_biased_candidate,
        "standard_dense_output": standard_dense_output,
        "standard_causal_output": standard_causal_output,
        "supports_tensor_4d_tma": supports_tensor_4d_tma,
    }
    fragments = flash_autotune_fragments(head_dim, num_kv, **common)
    base_values = {key: fragment.default() for key, fragment in fragments.items()}
    seeds: list[Config] = []
    target_seed = flash_attention_seed_config(
        head_dim,
        num_kv,
        target_device_capability=target_device_capability,
        block_size_targets=block_size_targets,
        **common,
    )
    if target_seed is not None:
        seeds.append(target_seed)
    base_seed = _flash_config_with_values(block_sizes, base_values)
    if base_seed not in seeds:
        seeds.append(base_seed)

    family_fragment = fragments[FLASH_PIPELINE_FAMILY_KEY]
    assert isinstance(family_fragment, EnumFragment)
    families = (
        family_fragment.choices
        if family_fragment.search_choices is None
        else family_fragment.search_choices
    )
    for family in families:
        assert isinstance(family, str)
        values = _flash_seed_values(
            head_dim,
            num_kv,
            pipeline_family_override=family,
            **common,
        )
        values[FLASH_PIPELINE_FAMILY_KEY] = family
        seeds.append(_flash_config_with_values(block_sizes, values))

    packet_fragment = fragments[FLASH_EXP2_PACKET_KEY]
    assert isinstance(packet_fragment, EnumFragment)
    packets = (
        packet_fragment.choices
        if packet_fragment.search_choices is None
        else packet_fragment.search_choices
    )
    for packet in packets:
        if packet == base_values[FLASH_EXP2_PACKET_KEY]:
            continue
        values = dict(base_values)
        values[FLASH_EXP2_PACKET_KEY] = packet
        if packet in _FLASH_MANUAL_EXP2_PACKET_PARAMS:
            # A compound packet seed owns its required parent family. Normal
            # search configs keep an explicit family, where a conflicting child
            # packet instead canonicalizes to the neutral packet.
            values.pop(FLASH_PIPELINE_FAMILY_KEY, None)
        values.update(
            _flash_compound_exp2_packet_overrides(
                head_dim,
                num_kv,
                values,
                dtype=dtype,
                is_causal=is_causal,
                has_kv_tile_pruning=has_kv_tile_pruning,
                requires_ws_overlap=requires_ws_overlap,
                small_biased_candidate=small_biased_candidate,
                standard_dense_output=standard_dense_output,
                standard_causal_output=standard_causal_output,
            )
        )
        seeds.append(_flash_config_with_values(block_sizes, values))

    # Seed each emitter-level policy once so quick tuning cannot miss a source
    # variant merely because it is conditionally active with another field.
    source_variants: tuple[tuple[str, object, Mapping[str, object]], ...] = (
        (FLASH_PERSISTENT_LOOP_KEY, "counted", {}),
        (
            FLASH_SP_ROW_SUM_KEY,
            "whole",
            {FLASH_SOFTMAX_DISC_KEY: False},
        ),
        (FLASH_SOFTMAX_SETUP_KEY, "stage_local", {}),
        (
            FLASH_EPI_TMA_SETUP_KEY,
            "role_local",
            {FLASH_EPI_TMA_KEY: True},
        ),
    )
    for key, value, dependencies in source_variants:
        fragment = fragments[key]
        assert isinstance(fragment, EnumFragment)
        active = (
            fragment.choices
            if fragment.search_choices is None
            else fragment.search_choices
        )
        if value not in active:
            continue
        values = {**base_values, **dependencies, key: value}
        seeds.append(_flash_config_with_values(block_sizes, values))

    unique: list[Config] = []
    for seed in seeds:
        if seed not in unique:
            unique.append(seed)
    return tuple(unique)


def _flash_legal_autotune_pipeline_families(
    head_dim: int,
    num_kv: int,
    *,
    num_bh: int | None,
    dtype: torch.dtype,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    output_requires_tma: bool,
    supports_tensor_4d_tma: bool,
    requested_family: str | None,
) -> tuple[str, ...]:
    """Enumerate families whose requested structure survives normalization."""
    candidates = (
        ("ws_overlap",)
        if requires_ws_overlap
        else (requested_family,)
        if requested_family is not None
        else FLASH_AUTOTUNE_PIPELINE_FAMILIES
    )
    result: list[str] = []
    for family in candidates:
        if family == "fa4_deep_1cta" and is_causal and requested_family is None:
            # The separate-ring causal prototype is available for explicit
            # experiments, but its phase graph has not passed long-sequence
            # stress or synccheck. Keep it out of unattended searches.
            continue
        if (has_kv_tile_pruning or small_biased_candidate) and family not in (
            "fa4",
            "ws_overlap",
        ):
            continue
        requested: dict[str, object] = {FLASH_PIPELINE_FAMILY_KEY: family}
        if output_requires_tma:
            requested[FLASH_EPI_TMA_KEY] = True
        effective = resolve_flash_config(
            head_dim,
            num_kv,
            requested,
            dtype=dtype,
            num_bh=num_bh,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            standard_dense_output=standard_dense_output,
            standard_causal_output=standard_causal_output,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
        )
        if effective.pipeline_family != family:
            continue
        if output_requires_tma and not effective.epi_tma:
            continue
        result.append(family)
    return tuple(result)


def flash_autotune_fragments(
    head_dim: int,
    num_kv: int,
    *,
    num_bh: int | None = None,
    tensor_4d_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    output_requires_tma: bool = False,
    supports_tensor_4d_tma: bool = True,
    topology_override: str | None = None,
    pipeline_family_override: str | None = None,
) -> dict[str, ConfigSpecFragment]:
    """Build a legality-driven, length-invariant flash search surface.

    Sequence length participates only in structural proofs such as even-pair or
    four-tile cluster alignment.  Within one legality class, defaults, active
    choices, and their ordering are identical for every length.
    """

    valid_topology = (
        topology_override if topology_override in ("fa4", "ws_overlap") else None
    )
    valid_family = (
        pipeline_family_override
        if pipeline_family_override in FLASH_PIPELINE_FAMILY_FLAGS
        else None
    )
    valid_family_flags = _flash_pipeline_family_flags(valid_family)
    if requires_ws_overlap:
        defaults_config: Mapping[str, object] | None = {
            FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"
        }
    elif valid_family is not None:
        defaults_config = {FLASH_PIPELINE_FAMILY_KEY: valid_family}
    elif valid_topology is not None:
        defaults_config = {FLASH_TOPOLOGY_KEY: valid_topology}
    else:
        defaults_config = None
    defaults = resolve_flash_config(
        head_dim,
        num_kv,
        defaults_config,
        dtype=dtype,
        num_bh=num_bh,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        prefer_packed_reduce=has_kv_tile_pruning or requires_ws_overlap,
    )
    paired = num_kv >= 2 and num_kv % 2 == 0
    cluster_aligned = num_kv >= 4 and num_kv % 4 == 0
    fa4_eligible = paired and not requires_ws_overlap
    requested_search_topology = (
        valid_family_flags.topology
        if valid_family_flags is not None
        else valid_topology
    )
    fa4_search_eligible = fa4_eligible and requested_search_topology != "ws_overlap"
    d64_fa4 = head_dim == 64 and fa4_search_eligible
    dense_d64_fa4 = d64_fa4 and not is_causal
    causal_d64_fa4 = d64_fa4 and is_causal
    standard_dense_d64 = (
        dense_d64_fa4
        and standard_dense_output
        and not has_kv_tile_pruning
        and not small_biased_candidate
    )
    standard_causal_d64 = (
        causal_d64_fa4
        and standard_causal_output
        and not has_kv_tile_pruning
        and not small_biased_candidate
    )
    bf16_d128_fa4 = head_dim == 128 and dtype is torch.bfloat16 and fa4_search_eligible
    bf16_d128_compound_packet_eligible = (
        bf16_d128_fa4 and not has_kv_tile_pruning and not small_biased_candidate
    )

    def enum(
        default: _T,
        choices: Iterable[_T],
        search_choices: Iterable[_T] | None = None,
        coverage_choices: Iterable[_T] | None = None,
    ) -> EnumFragment:
        all_choices = _flash_choices_with_default(default, choices)
        active = (
            None
            if search_choices is None
            else _flash_choices_with_default(default, search_choices)
        )
        coverage = (
            None
            if coverage_choices is None
            else _flash_choices_with_default(default, coverage_choices)
        )
        return EnumFragment(all_choices, active, coverage)

    if num_kv < 2:
        s_stage = enum(1, (1,), (1,))
    elif (
        fa4_search_eligible
        and valid_family != "ws_overlap"
        and (valid_family is not None or output_requires_tma)
    ):
        s_stage = enum(defaults.s_stage, (1, 2), (2,))
    else:
        s_stage = enum(defaults.s_stage, (1, 2))

    ws_only_kv_pipeline = not fa4_search_eligible
    if num_kv < 2:
        kv_stage = enum(1, (1,), (1,))
    elif ws_only_kv_pipeline:
        cap = _flash_deep_1cta_kv_stage_cap(head_dim)
        kv_stage_search = tuple(range(2, cap + 1)) if cap >= 2 else (2, 3)
        kv_stage_values = (
            (2, 3, 4, 6, 8, 10)
            if head_dim == 64
            else tuple(dict.fromkeys((2, 3, *kv_stage_search)))
        )
        kv_stage = enum(defaults.kv_stage, kv_stage_values, kv_stage_search)
    elif defaults.separate_kv_rings:
        cap = _flash_deep_1cta_kv_stage_cap(head_dim)
        kv_stage_values = tuple(range(2, cap + 1))
        kv_stage = enum(defaults.kv_stage, kv_stage_values, kv_stage_values)
    elif fa4_search_eligible and head_dim in (64, 128):
        value_cap = _flash_aliased_kv_stage_cap(head_dim, stage_output=False)
        search_cap = _flash_aliased_kv_stage_cap(
            head_dim,
            stage_output=output_requires_tma,
        )
        kv_stage_values = tuple(range(2, value_cap + 1))
        kv_stage_search = tuple(range(2, search_cap + 1))
        kv_stage = enum(defaults.kv_stage, kv_stage_values, kv_stage_search)
    else:
        kv_stage = enum(defaults.kv_stage, (2, 3))

    fixed_family_requires_persistence = valid_family_flags is not None and (
        valid_family_flags.use_clc_scheduler or valid_family_flags.local_tma_partition
    )
    if (
        valid_family_flags is not None
        and valid_family_flags.use_clc_scheduler
        and not requires_ws_overlap
    ):
        persistent = enum(True, (True, False), (True,))
        persistent_ctas = enum(1, (1,), (1,))
        recompute = enum(False, (False, True), (False,))
    elif fa4_search_eligible and not is_causal:
        persistent = enum(defaults.persistent, (True, False), (True, False))
        persistent_ctas = enum(
            defaults.persistent_ctas_per_sm, (1, 2, 3, 4), (1, 2, 3, 4)
        )
        recompute = enum(defaults.recompute_tile_coords, (False, True), (False, True))
    elif is_causal or num_kv < 2:
        persistent = enum(defaults.persistent, (False, True), (False,))
        persistent_ctas = enum(defaults.persistent_ctas_per_sm, (1,), (1,))
        recompute = enum(defaults.recompute_tile_coords, (False,), (False,))
    else:
        persistent = enum(defaults.persistent, (False, True))
        persistent_ctas = enum(defaults.persistent_ctas_per_sm, (1,), (1,))
        recompute = enum(defaults.recompute_tile_coords, (False,), (False,))
    if fixed_family_requires_persistence:
        persistent = enum(True, (True, False), (True,))
    if valid_family_flags is not None and (
        valid_family_flags.use_clc_scheduler
        or valid_family_flags.use_2cta_instrs
        or valid_family_flags.use_cga2_local_cta
        or valid_family_flags.separate_kv_rings
    ):
        recompute = enum(False, (False, True), (False,))

    manual_schedule_names = tuple(
        dict.fromkeys(
            f"{freq}/{res}"
            for freq, res in _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.values()
        )
    )
    schedule_values = ("16/4", "8/2", "16/2", "xu", *manual_schedule_names)
    schedule_search = (
        schedule_values if fa4_search_eligible else (defaults.e2e_schedule,)
    )
    manual_schedule = _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.get(defaults.exp2_packet)
    if manual_schedule is not None:
        manual_name = f"{manual_schedule[0]}/{manual_schedule[1]}"
        e2e_schedule = enum(manual_name, (manual_name,), (manual_name,))
    else:
        e2e_schedule = enum(defaults.e2e_schedule, schedule_values, schedule_search)

    if is_causal:
        masked_values = (
            ("inherit", "xu", "16/4", "8/2", *manual_schedule_names)
            if fa4_search_eligible
            else ("inherit", "xu", "16/4", "8/2")
        )
        if (
            fa4_search_eligible
            and manual_schedule is not None
            and standard_causal_output
        ):
            masked_name = f"{manual_schedule[0]}/{manual_schedule[1]}"
            masked_schedule = enum(masked_name, (masked_name,), (masked_name,))
        else:
            # Legacy fixed configs may name a distinct cadence even when the
            # resolver canonicalizes it away; keep it valid but out of search.
            masked_search = (
                ("inherit", "xu", "16/4", "8/2")
                if fa4_search_eligible and standard_causal_output
                else ("inherit",)
            )
            masked_schedule = enum(
                defaults.masked_e2e_schedule,
                masked_values,
                masked_search,
            )
    else:
        masked_schedule = enum("inherit", ("inherit",), ("inherit",))

    offset_values = tuple(range(16))
    if fa4_search_eligible:
        e2e_offset = enum(defaults.e2e_offset, offset_values, offset_values)
        e2e_offset0 = enum(defaults.e2e_offset0, offset_values, offset_values)
    else:
        e2e_offset = enum(defaults.e2e_offset, offset_values, (defaults.e2e_offset,))
        e2e_offset0 = enum(defaults.e2e_offset0, offset_values, (defaults.e2e_offset0,))

    if fa4_search_eligible:
        softmax_disc = enum(
            defaults.softmax_disc,
            (True, False),
            (True,) if is_causal else (True, False),
        )
    else:
        softmax_disc = enum(
            defaults.softmax_disc, (True, False), (defaults.softmax_disc,)
        )

    if fa4_search_eligible:
        disc_pipe = enum(defaults.disc_pipe_depth, (1, 2, 3, 4), (1, 2, 3, 4))
        split_p = enum(defaults.split_p_arrive, (False, True), (False, True))
        p_store = enum(defaults.p_store_repetition, (16, 32), (16, 32))
        s_load = enum(defaults.s_load_repetition, (16, 32), (16, 32))
        precompute = enum(defaults.precompute_qk_desc, (False, True), (False, True))
        first_load = enum(defaults.first_load_order, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4))
        kv_order = enum(
            defaults.kv_order,
            ("ascending", "descending"),
            ("ascending",) if is_causal else ("ascending", "descending"),
        )
    else:
        disc_pipe = enum(
            defaults.disc_pipe_depth,
            (1, 2, 3, 4),
            (defaults.disc_pipe_depth,),
        )
        split_p = enum(
            defaults.split_p_arrive, (False, True), (defaults.split_p_arrive,)
        )
        p_store = enum(
            defaults.p_store_repetition, (16, 32), (defaults.p_store_repetition,)
        )
        s_load = enum(
            defaults.s_load_repetition, (16, 32), (defaults.s_load_repetition,)
        )
        precompute = enum(
            defaults.precompute_qk_desc, (False, True), (defaults.precompute_qk_desc,)
        )
        first_load = enum(
            defaults.first_load_order,
            (0, 1, 2, 3, 4),
            (defaults.first_load_order,),
        )
        kv_order = enum(
            defaults.kv_order,
            ("ascending", "descending"),
            (defaults.kv_order,),
        )

    threshold_search = _flash_safe_rescale_threshold_values(dtype)
    rescale_threshold = enum(
        defaults.rescale_threshold,
        _FLASH_RESCALE_THRESHOLD_VALUES,
        threshold_search if fa4_search_eligible else (defaults.rescale_threshold,),
    )
    rescale_chunk_search = (8, 16, 32, 64) if head_dim == 64 else (8, 16, 32)
    if valid_family_flags is not None and valid_family_flags.use_clc_scheduler:
        rescale_chunk_search = (8, 16, 32)
    rescale_chunk = enum(
        defaults.rescale_chunk_cols,
        (8, 16, 32, 64),
        rescale_chunk_search if fa4_search_eligible else (defaults.rescale_chunk_cols,),
    )
    softmax_regs = enum(
        defaults.softmax_regs,
        (176, 184, 192, 200),
        (176, 184, 192, 200) if fa4_search_eligible else (defaults.softmax_regs,),
    )
    corr_regs = enum(
        defaults.corr_regs,
        (64, 72, 80, 88),
        (64, 72, 80, 88) if fa4_search_eligible else (defaults.corr_regs,),
    )
    other_regs = enum(
        defaults.other_regs,
        (24, 32, 40, 48, 56, 64, 80),
        (24, 32, 40, 48, 56, 64, 80) if fa4_search_eligible else (defaults.other_regs,),
    )
    corr_tile = enum(
        defaults.corr_tile_size,
        (8, 16, 32),
        (8, 16, 32) if fa4_search_eligible else (defaults.corr_tile_size,),
    )
    packed_reduce = enum(
        defaults.packed_reduce,
        (False, True),
        (True,)
        if output_requires_tma or requested_search_topology == "fa4"
        else (False, True),
    )

    if is_causal:
        causal_lpt_search = _flash_causal_lpt_candidates()
        causal_lpt = enum(
            defaults.causal_lpt_swizzle,
            (0, *causal_lpt_search),
            causal_lpt_search
            if fa4_search_eligible
            else (defaults.causal_lpt_swizzle,),
        )
        causal_order = enum(
            defaults.causal_kv_order,
            ("ascending", "descending"),
            ("ascending", "descending")
            if fa4_search_eligible
            else (defaults.causal_kv_order,),
        )
        causal_split = enum(
            defaults.causal_loop_split,
            (False, True),
            (False, True) if fa4_search_eligible else (defaults.causal_loop_split,),
        )
    else:
        causal_lpt = enum(0, (0,), (0,))
        causal_order = enum("ascending", ("ascending",), ("ascending",))
        causal_split = enum(False, (False,), (False,))
    role_map = enum(
        defaults.role_map,
        ("helion", "fa4"),
        ("helion", "fa4") if fa4_search_eligible else (defaults.role_map,),
    )

    active_families = _flash_legal_autotune_pipeline_families(
        head_dim,
        num_kv,
        num_bh=num_bh,
        dtype=dtype,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        output_requires_tma=output_requires_tma,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        requested_family=valid_family,
    )
    if valid_family is None and valid_topology is not None:
        active_families = tuple(
            family
            for family in active_families
            if cast(
                "FlashPipelineFamilyFlags", _flash_pipeline_family_flags(family)
            ).topology
            == valid_topology
        )
    if not active_families:
        raise InvalidConfig(
            f"CuTe flash pipeline family {valid_family!r} is not legal for "
            f"head_dim={head_dim}, num_kv={num_kv}, causal={is_causal}"
        )
    family_default = (
        defaults.pipeline_family
        if defaults.pipeline_family in active_families
        else active_families[0]
    )
    pipeline_family = enum(
        family_default,
        FLASH_AUTOTUNE_PIPELINE_FAMILIES,
        active_families,
    )
    role_chain_supported = any(
        flags is not None
        and flags.topology == "fa4"
        and not flags.use_cga2_local_cta
        and not flags.use_clc_scheduler
        and (not flags.use_2cta_instrs or (head_dim == 64 and not is_causal))
        for flags in (
            _flash_pipeline_family_flags(family) for family in active_families
        )
    )
    role_chain = enum(
        defaults.role_chain,
        (False, True),
        (False, True)
        if role_chain_supported and head_dim == 64 and not is_causal
        else (defaults.role_chain,),
    )

    clc_candidates = _flash_clc_heads_per_batch_candidates(num_bh)
    use_clc = any(
        cast_flags is not None and cast_flags.use_clc_scheduler
        for cast_flags in (
            _flash_pipeline_family_flags(family) for family in active_families
        )
    )
    clc_default = (
        defaults.clc_heads_per_batch
        if defaults.clc_heads_per_batch in clc_candidates
        else clc_candidates[0]
    )
    clc_coverage_candidates = _flash_clc_heads_per_batch_coverage_candidates(
        num_bh,
        tensor_4d_heads=tensor_4d_heads,
        effective_default=clc_default if use_clc else None,
    )
    clc_heads = enum(
        clc_default if use_clc else 0,
        (0, *clc_candidates),
        clc_candidates if use_clc else (0,),
        clc_coverage_candidates if use_clc else (0,),
    )
    clc_pdl = enum(
        defaults.clc_use_pdl,
        (False, True),
        (False, True) if use_clc else (False,),
    )
    clc_stages = enum(
        defaults.clc_stages,
        (1, 2, 3),
        (2, 3) if use_clc else (1,),
    )

    fixed_family_disables_epilogue = (
        valid_family_flags is not None and valid_family_flags.separate_kv_rings
    )
    fixed_family_disables_paired_epilogue = valid_family_flags is not None and (
        valid_family_flags.use_2cta_instrs or fixed_family_disables_epilogue
    )
    if output_requires_tma:
        epi_tma = enum(True, (True, False), (True,))
        epi_stg = enum(False, (False, True), (False,))
        epi_stg_store = enum("slice", ("slice", "whole"), ("slice",))
        epi_stg_gmem = enum("stage", ("stage", "pair"), ("stage",))
    elif fixed_family_disables_epilogue:
        epi_tma = enum(False, (False, True), (False,))
        epi_stg = enum(False, (False, True), (False,))
        epi_stg_store = enum("slice", ("slice", "whole"), ("slice",))
        epi_stg_gmem = enum("stage", ("stage", "pair"), ("stage",))
    elif fixed_family_disables_paired_epilogue:
        epi_tma = enum(defaults.epi_tma, (False, True), (False, True))
        epi_stg = enum(defaults.epi_stg, (False, True), (False, True))
        epi_stg_store = enum(
            defaults.epi_stg_store,
            ("slice", "whole"),
            ("slice", "whole"),
        )
        epi_stg_gmem = enum("stage", ("stage", "pair"), ("stage",))
    elif fa4_search_eligible:
        epi_tma = enum(defaults.epi_tma, (False, True), (False, True))
        epi_stg = enum(defaults.epi_stg, (False, True), (False, True))
        epi_stg_store = enum(
            defaults.epi_stg_store,
            ("slice", "whole"),
            ("slice", "whole"),
        )
        epi_stg_gmem = enum(
            defaults.epi_stg_gmem,
            ("stage", "pair"),
            ("stage", "pair"),
        )
    else:
        epi_tma = enum(False, (False, True), (False,))
        epi_stg = enum(defaults.epi_stg, (False, True), (defaults.epi_stg,))
        epi_stg_store = enum(
            defaults.epi_stg_store,
            ("slice", "whole"),
            (defaults.epi_stg_store,),
        )
        epi_stg_gmem = enum(
            defaults.epi_stg_gmem,
            ("stage", "pair"),
            (defaults.epi_stg_gmem,),
        )

    if defaults.topology == "fa4":
        wait_hint = enum(defaults.wait_hint, (10_000_000, 0), (10_000_000, 0))
        mma_interleave = enum(defaults.mma_interleave, (True, False), (True, False))
    else:
        wait_hint = enum(10_000_000, (10_000_000,), (10_000_000,))
        mma_interleave = enum(False, (False,), (False,))

    packet_values: list[str] = ["1x1"]
    packet_search: list[str] = ["1x1"]
    if d64_fa4:
        packet_values = list(_FLASH_EXP2_PACKET_PARAMS)
        packet_search = list(_FLASH_EXP2_PACKET_PARAMS)
        if dtype is torch.float16 and standard_dense_d64 and cluster_aligned:
            packet_values.extend(
                (
                    _FLASH_DEG2_EXP2_PACKET,
                    _FLASH_DEG1_EXP2_PACKET,
                    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET,
                )
            )
            packet_search.extend(
                (
                    _FLASH_DEG2_EXP2_PACKET,
                    _FLASH_DEG1_EXP2_PACKET,
                    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET,
                )
            )
        elif dtype is torch.float16 and standard_causal_d64:
            packet_values.append(_FLASH_DEG2_EXP2_PACKET)
            packet_search.append(_FLASH_DEG2_EXP2_PACKET)
            packet_values.append(_FLASH_HYBRID_EXP2_PACKET)
            packet_search.append(_FLASH_HYBRID_EXP2_PACKET)
        elif dtype is torch.bfloat16 and standard_causal_d64:
            packet_values.append(_FLASH_HYBRID_EXP2_PACKET)
            packet_search.append(_FLASH_HYBRID_EXP2_PACKET)
    elif (
        bf16_d128_compound_packet_eligible
        and not is_causal
        and standard_dense_output
        and cluster_aligned
    ):
        packet_values.append(_FLASH_DEG2_EXP2_PACKET)
        packet_search.append(_FLASH_DEG2_EXP2_PACKET)
    elif bf16_d128_compound_packet_eligible and is_causal and standard_causal_output:
        resident_packet = (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET
        )
        packet_values.append(resident_packet)
        packet_search.append(resident_packet)
    if valid_family is not None:
        packet_search = [
            packet
            for packet in packet_search
            if resolve_flash_config(
                head_dim,
                num_kv,
                {
                    FLASH_PIPELINE_FAMILY_KEY: valid_family,
                    FLASH_EXP2_PACKET_KEY: packet,
                },
                dtype=dtype,
                num_bh=num_bh,
                is_causal=is_causal,
                has_kv_tile_pruning=has_kv_tile_pruning,
                requires_ws_overlap=requires_ws_overlap,
                small_biased_candidate=small_biased_candidate,
                standard_dense_output=standard_dense_output,
                standard_causal_output=standard_causal_output,
                supports_tensor_4d_tma=supports_tensor_4d_tma,
            ).exp2_packet
            == packet
        ]
    if defaults.exp2_packet in _FLASH_MANUAL_EXP2_PACKET_PARAMS:
        exp2_packet = enum(
            defaults.exp2_packet,
            (defaults.exp2_packet,),
            (defaults.exp2_packet,),
        )
    else:
        exp2_packet = enum(defaults.exp2_packet, packet_values, packet_search)

    if dense_d64_fa4:
        transports = (
            ("ring2", "single", "single_final")
            if standard_dense_output
            else ("ring2", "single")
        )
        if fixed_family_requires_persistence:
            # These families require persistence, while single_final needs a
            # nonpersistent grid or the two-CTA instruction path.
            transports = ("ring2", "single")
        stat_transport = enum(
            defaults.stat_transport,
            transports,
            transports,
        )
    else:
        stat_transport = enum("ring2", ("ring2",), ("ring2",))

    small_biased = enum(
        defaults.small_biased,
        (True, False),
        (True, False) if small_biased_candidate else (defaults.small_biased,),
    )
    skip_rescale = enum(False, (False, True), (False,))

    persistent_loop_eligible = (
        fa4_search_eligible and not is_causal and not defaults.use_clc_scheduler
    )
    persistent_loop = enum(
        defaults.persistent_loop,
        ("while", "counted"),
        ("while", "counted") if persistent_loop_eligible else ("while",),
    )
    sp_row_sum = enum(
        defaults.sp_row_sum,
        ("fragment", "whole"),
        ("fragment", "whole")
        if dense_d64_fa4 and dtype is torch.float16
        else ("fragment",),
    )
    softmax_setup = enum(
        defaults.softmax_setup,
        ("shared", "stage_local"),
        ("shared", "stage_local")
        if dense_d64_fa4 and dtype is torch.float16
        else ("shared",),
    )
    epi_tma_setup = enum(
        defaults.epi_tma_setup,
        ("shared", "role_local"),
        ("shared", "role_local")
        if standard_dense_d64 and not fixed_family_disables_epilogue
        else ("shared",),
    )

    fragments: dict[str, ConfigSpecFragment] = {
        FLASH_S_STAGE_KEY: s_stage,
        FLASH_KV_STAGE_KEY: kv_stage,
        FLASH_PERSISTENT_KEY: persistent,
        FLASH_PERSISTENT_CTAS_PER_SM_KEY: persistent_ctas,
        FLASH_RECOMPUTE_TILE_COORDS_KEY: recompute,
        FLASH_E2E_SCHEDULE_KEY: e2e_schedule,
        FLASH_MASKED_E2E_SCHEDULE_KEY: masked_schedule,
        FLASH_E2E_OFFSET_KEY: e2e_offset,
        FLASH_E2E_OFFSET0_KEY: e2e_offset0,
        FLASH_EXP2_PACKET_KEY: exp2_packet,
        FLASH_MMA_INTERLEAVE_KEY: mma_interleave,
        FLASH_WAIT_HINT_KEY: wait_hint,
        FLASH_STAT_TRANSPORT_KEY: stat_transport,
        FLASH_PIPELINE_FAMILY_KEY: pipeline_family,
        FLASH_SOFTMAX_DISC_KEY: softmax_disc,
        FLASH_DISC_PIPE_KEY: disc_pipe,
        FLASH_SPLIT_P_ARRIVE_KEY: split_p,
        FLASH_P_STORE_REP_KEY: p_store,
        FLASH_S_LOAD_REP_KEY: s_load,
        FLASH_PRECOMPUTE_QK_DESC_KEY: precompute,
        FLASH_FIRST_LOAD_ORDER_KEY: first_load,
        FLASH_KV_ORDER_KEY: kv_order,
        FLASH_EPI_TMA_KEY: epi_tma,
        FLASH_EPI_STG_KEY: epi_stg,
        FLASH_EPI_STG_STORE_KEY: epi_stg_store,
        FLASH_EPI_STG_GMEM_KEY: epi_stg_gmem,
        FLASH_RESCALE_THRESHOLD_KEY: rescale_threshold,
        FLASH_SKIP_RESCALE_STATS_KEY: skip_rescale,
        FLASH_RESCALE_CHUNK_COLS_KEY: rescale_chunk,
        FLASH_SOFTMAX_REGS_KEY: softmax_regs,
        FLASH_CORR_REGS_KEY: corr_regs,
        FLASH_OTHER_REGS_KEY: other_regs,
        FLASH_CORR_TILE_SIZE_KEY: corr_tile,
        FLASH_PACKED_REDUCE_KEY: packed_reduce,
        FLASH_SMALL_BIASED_KEY: small_biased,
        FLASH_CAUSAL_LPT_SWIZZLE_KEY: causal_lpt,
        FLASH_CAUSAL_KV_ORDER_KEY: causal_order,
        FLASH_ROLE_MAP_KEY: role_map,
        FLASH_ROLE_CHAIN_KEY: role_chain,
        FLASH_CLC_HEADS_PER_BATCH_KEY: clc_heads,
        FLASH_CLC_PDL_KEY: clc_pdl,
        FLASH_CLC_STAGES_KEY: clc_stages,
        FLASH_CAUSAL_LOOP_SPLIT_KEY: causal_split,
        FLASH_PERSISTENT_LOOP_KEY: persistent_loop,
        FLASH_SP_ROW_SUM_KEY: sp_row_sum,
        FLASH_SOFTMAX_SETUP_KEY: softmax_setup,
        FLASH_EPI_TMA_SETUP_KEY: epi_tma_setup,
    }
    target_tuning_policy = get_flash_target_policy(
        target_device_capability
    ).tuning_for_torch(head_dim, str(dtype).removeprefix("torch."))
    policy_values: dict[str, object] = {}
    if target_tuning_policy is not None:
        if not is_causal and standard_dense_output:
            dense_policy = target_tuning_policy.dense_policy(num_kv)
            if dense_policy is not None:
                policy_values = _flash_dense_tuning_overrides(dense_policy)
        elif is_causal and standard_causal_output:
            causal_policy = target_tuning_policy.causal_policy(num_kv)
            if causal_policy is not None:
                policy_values = _flash_causal_tuning_overrides(causal_policy)
    for key, value in policy_values.items():
        if key not in fragments:
            continue
        fragment = cast("EnumFragment", fragments[key])
        if value not in fragment.choices:
            search_choices = fragment.search_choices
            if search_choices is None:
                # Target-only seed values remain legal without entering the
                # generic search surface.
                search_choices = fragment.choices
            fragments[key] = EnumFragment(
                (*fragment.choices, value),
                search_choices,
                fragment.coverage_choices,
            )
    return fragments


def flash_config_from_config(
    config: Mapping[str, object],
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype = torch.float16,
    *,
    num_bh: int | None = None,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    supports_tensor_4d_tma: bool = True,
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
        num_bh=num_bh,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
    )


if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


# The SharedStorage struct (Gotcha G: stringified annotations break
# ``@cute.struct``) and the inline-traced rescale helper live in
# ``_flash_runtime`` (a real module compiled WITHOUT ``from __future__ import
# annotations``); the generated module imports them. The remaining cute / utils
# / pipeline symbols are imported under flash-local aliases.
_FLASH_RUNTIME_ABI = 3

# This literal is part of generated source and therefore the CuTe disk-cache
# key. Bump it whenever an imported flash runtime helper changes semantics.
_FLASH_PREAMBLE_IMPORTS = f"""\
import cutlass.utils as cutlass_utils_flash
import cutlass.pipeline as cutlass_pipeline_flash
from cutlass.cute.nvgpu import cpasync as cute_cpasync_flash
from cutlass.cute.nvgpu import tcgen05 as cute_tcgen05_flash
import cutlass.utils.blackwell_helpers as sm100_utils_flash
import helion._compiler.cute._flash_runtime as _helion_flash_rt
import helion._compiler.cute._flash_gemm_ptx as _helion_flash_ptx
_helion_flash_runtime_abi = {_FLASH_RUNTIME_ABI}
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


def _flash_fa4_descending_causal_split_proof(
    *,
    sequence_extent: int,
    num_query_tiles: int,
    num_kv_tiles: int,
    score_plan: AttentionScorePlan,
) -> CausalRangeProof:
    """Prove the FA4 descending split's mask-free runtime loop."""
    if num_query_tiles != num_kv_tiles:
        return CausalRangeProof(False, "query/KV tile-count mismatch")
    if sequence_extent != num_kv_tiles * 128:
        return CausalRangeProof(False, "partial or uncovered sequence tail")
    tile_layout = TileLayout(extent=sequence_extent, stride=128, width=128)
    return prove_descending_causal_prefix_unmasked(
        query_tiles=IntegerInterval(0, num_query_tiles),
        query_layout=tile_layout,
        kv_layout=tile_layout,
        has_additional_modifiers=(score_plan.modifier_kinds != (CAUSAL_MASK_KIND,)),
        has_kv_tile_pruning=score_plan.has_kv_tile_pruning,
    )


def _flash_fa4_causal_split_equal_iteration_proof(
    *,
    split_range_proof: CausalRangeProof,
    query_slots_per_cta: int,
) -> CausalRangeProof:
    """Prove that every split causal query slot executes the same KV count."""
    if not split_range_proof.proven:
        return CausalRangeProof(False, split_range_proof.reason)
    if query_slots_per_cta != 2:
        return CausalRangeProof(False, "FA4 resident path requires two query slots")
    # Each resident query slot shares ``flash_num_active_kv``.  Its descending
    # split executes ``flash_num_active_kv - m_tile`` masked iterations followed
    # by ``m_tile`` unmasked iterations, so both slots execute the same active
    # KV count while traversing different mask-free suffix lengths.
    return CausalRangeProof(
        True, "resident query slots share the same active KV iteration count"
    )


@dataclasses.dataclass(frozen=True)
class _FlashSoftmaxLoopSegment:
    """Runtime loop domain and phase facts for one online-softmax segment."""

    loop_var: str
    loop_bound: str
    kv_expr: str | None = None
    continues_previous_segment: bool = False

    @property
    def not_first_condition(self) -> str:
        """Return the loop-local expression proving this is not the first KV tile."""
        if self.continues_previous_segment:
            return f"{self.loop_var} >= cutlass.Int32(0)"
        return f"{self.loop_var} != 0"


_FLASH_ONLINE_STATISTICS_UPDATE = (
    "            flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum"
)


@dataclasses.dataclass(frozen=True)
class _FlashOnlineSoftmaxIteration:
    """Ordered source blocks for one statically specialized softmax iteration.

    The final row-sum publication is deliberately not part of this plan because
    it occurs after the complete loop (or masked/unmasked loop pair).
    """

    load_and_reduce: str
    alpha_pre_probability: str
    alpha_publish_pre_probability: str
    probability_update: str
    pre_max_update: str = ""
    alpha_post_probability: str = ""
    alpha_publish_post_probability: str = ""
    statistics_acquire: str = ""
    statistics_update: str = _FLASH_ONLINE_STATISTICS_UPDATE
    post_statistics: str = ""


def _flash_causal_split_softmax_segments(
    kv_loop_bound: str,
    stage: str,
    *,
    split_range_proof: CausalRangeProof,
) -> tuple[_FlashSoftmaxLoopSegment, _FlashSoftmaxLoopSegment]:
    """Return the proof-backed masked prefix and unmasked causal suffix."""
    if not split_range_proof.proven:
        raise AssertionError(
            "causal softmax split requires a proven complete masked/unmasked range"
        )
    return (
        _FlashSoftmaxLoopSegment(
            loop_var="flash_kv_mask_iter",
            loop_bound=f"{kv_loop_bound} - flash_m_tile{stage}",
            kv_expr=(f"{kv_loop_bound} - cutlass.Int32(1) - flash_kv_mask_iter"),
        ),
        _FlashSoftmaxLoopSegment(
            loop_var="flash_kv_unmask_iter",
            loop_bound=f"flash_m_tile{stage}",
            kv_expr=(f"flash_m_tile{stage} - cutlass.Int32(1) - flash_kv_unmask_iter"),
            continues_previous_segment=True,
        ),
    )


def _format_fa4_online_softmax_loop(
    segment: _FlashSoftmaxLoopSegment,
    iteration: _FlashOnlineSoftmaxIteration,
    *,
    stage: str,
    wait_hint: int,
) -> str:
    """Render the common FA4 wait/max/alpha/P/row-sum iteration order."""
    kv_assignment = (
        f"\n            flash_kv = {segment.kv_expr}" if segment.kv_expr else ""
    )
    return f"""        for {segment.loop_var} in cutlass.range({segment.loop_bound}, unroll=1):{kv_assignment}
            _helion_flash_rt.mbar_spin_wait(
                flash_s_full_ptr + {stage}, flash_s_full_phase, {wait_hint})
            flash_s_full_phase ^= 1
{iteration.pre_max_update}
            flash_old_row_max = flash_row_max
{iteration.load_and_reduce}
            flash_row_max_safe = flash_row_max
            if flash_row_max == -cutlass.Float32.inf:
                flash_row_max_safe = cutlass.Float32(0.0)
{iteration.alpha_pre_probability}
{iteration.alpha_publish_pre_probability}
{iteration.probability_update}
{iteration.alpha_post_probability}
{iteration.alpha_publish_post_probability}
{iteration.statistics_acquire}
{iteration.statistics_update}
{iteration.post_statistics}"""


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
    output_epilogue: str = _FLASH_OUTPUT_EPILOGUE_IDENTITY,
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
    relu_output = _flash_output_relu_enabled(output_epilogue)
    relu_store = (
        "\n            _helion_flash_rt.relu_fragment_inplace(flash_reg)"
        if relu_output
        else ""
    )
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
    # arch>=100). The choice is represented in the config and measured instead
    # of being selected from a sequence-length threshold.
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
            flash_reg.store(flash_reg.load() * flash_inv_sum){relu_store}
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
    output_epilogue: str = _FLASH_OUTPUT_EPILOGUE_IDENTITY,
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
            output_epilogue=output_epilogue,
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
    persistent_loop: str = "while",
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
        clc_m_pair_expr = (
            f"cutlass.Int32({num_m_pairs} - 1) - "
            "cutlass.Int32(flash_clc_work.tile_idx[0])"
        )
        advance = """        flash_clc_pipeline.consumer_wait(flash_clc_consumer_state)
        flash_clc_response_ptr = (
            flash_clc_response_base
            + flash_clc_consumer_state.index * cutlass.Int32(4)).align(16)
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
    use_counted_no_prelude = (
        persistent_loop == "counted"
        and prelude == "none"
        and total_tiles is not None
        and num_m_pairs is not None
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
    sequence_extent: int,
    num_bh: int,
    total_tiles: int,
    cfg: FlashAttentionConfig,
    has_lse: bool,
    io_dtype: str,
    score_plan: AttentionScorePlan,
    tensor_4d_batch: int = 0,
    tensor_4d_heads: int = 0,
    target_device_capability: tuple[int, int] | None = None,
    output_epilogue: str = _FLASH_OUTPUT_EPILOGUE_IDENTITY,
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
    relu_output = _flash_output_relu_enabled(output_epilogue)
    direct_relu_store = (
        "\n            _helion_flash_rt.relu_fragment_inplace(flash_reg{stage})"
        if relu_output
        else ""
    )
    staged_relu_keyword = ", relu_output=True" if relu_output else ""
    kv_stage = cfg.kv_stage
    q_stage = cfg.q_tile_count
    assert q_stage == 2
    s_corr_stage = 2
    assert total_tiles % num_bh == 0
    if cfg.skip_rescale_stats:
        cfg = dataclasses.replace(cfg, skip_rescale_stats=False)
    is_causal = score_plan.is_causal
    if is_causal:
        assert not cfg.persistent
    target_policy = get_flash_target_policy(target_device_capability)
    hardware_capabilities = target_policy.hardware
    tuning_policy = target_policy.tuning_for_cute(hd, io_dtype)
    tmem_row_reduce_min_kv = (
        tuning_policy.tmem_row_reduce_min_kv if tuning_policy is not None else None
    )
    use_tmem_row_reduce = (
        hardware_capabilities.supports_tmem_row_reduce
        and tmem_row_reduce_min_kv is not None
        and num_kv >= tmem_row_reduce_min_kv
        and cfg.s_load_repetition == 32
        and score_plan.modifier_kinds in ((DENSE_SCORE_KIND,), (CAUSAL_MASK_KIND,))
    )
    causal_desc_kv = is_causal and cfg.causal_kv_order == "descending"
    desc_kv = causal_desc_kv or (not is_causal and cfg.kv_order == "descending")
    num_m_pairs = total_tiles // num_bh
    causal_split_proof = _flash_fa4_descending_causal_split_proof(
        sequence_extent=sequence_extent,
        num_query_tiles=num_m_pairs * (4 if cfg.causal_two_cta else 2),
        num_kv_tiles=num_kv,
        score_plan=score_plan,
    )
    causal_split_equal_iteration_proof = _flash_fa4_causal_split_equal_iteration_proof(
        split_range_proof=causal_split_proof,
        query_slots_per_cta=q_stage,
    )
    dense_tuning = (
        tuning_policy.dense_policy(num_kv) if tuning_policy is not None else None
    )
    probability_log2_shift = (
        dense_tuning.probability_log2_shift if dense_tuning is not None else 0
    )
    dense_seed_matches = _flash_dense_target_seed_matches(cfg, dense_tuning)
    dense_softmax_lowering = (
        dense_tuning.softmax_lowering
        if dense_tuning is not None
        else FlashSoftmaxLowering.STANDARD
    )
    dense_packed_exp2_mode = (
        dense_tuning.packed_exp2_mode
        if dense_tuning is not None
        else FlashPackedExp2Mode.DISABLED
    )
    dense_target_lowering_applies = (
        dense_tuning is not None
        and dense_seed_matches
        and not is_causal
        and not has_lse
        and cfg.use_2cta_instrs
        and not cfg.separate_kv_rings
        and not cfg.softmax_disc
        and cfg.p_store_repetition == 16
        and cfg.s_load_repetition == 32
        and score_plan.modifier_kinds == (DENSE_SCORE_KIND,)
    )
    dense_resident_value_graph_candidate = (
        dense_target_lowering_applies
        and cfg.split_p_arrive
        and cfg.rescale_threshold > 0.0
        and dense_softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
    )
    causal_tuning = (
        tuning_policy.causal_policy(num_kv) if tuning_policy is not None else None
    )
    causal_seed_matches = _flash_causal_resident_native_seed_matches(cfg, causal_tuning)
    use_causal_resident_native = (
        causal_tuning is not None
        and causal_seed_matches
        and use_tmem_row_reduce
        and is_causal
        and not has_lse
        and not cfg.use_2cta_instrs
        and not cfg.separate_kv_rings
        and cfg.causal_loop_split
        and causal_desc_kv
        and causal_split_proof.proven
        and causal_split_equal_iteration_proof.proven
        and cfg.p_store_repetition == 16
        and cfg.s_load_repetition == 32
        and cfg.split_p_arrive
        and cfg.rescale_threshold > 0.0
        and score_plan.modifier_kinds == (CAUSAL_MASK_KIND,)
    )
    use_causal_resident_value_graph = False
    use_causal_stateful_softmax = False
    if use_causal_resident_native:
        assert causal_tuning is not None
        resident_softmax_family = causal_tuning.softmax_lowering
        if resident_softmax_family is FlashSoftmaxLowering.STANDARD:
            pass
        elif resident_softmax_family is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH:
            use_causal_resident_value_graph = True
        elif resident_softmax_family is FlashSoftmaxLowering.STATEFUL:
            use_causal_stateful_softmax = True
        else:
            raise AssertionError(
                "unsupported resident softmax lowering family: "
                f"{resident_softmax_family!r}"
            )
    if use_causal_resident_native:
        # Use the architecture-selected resident softmax lowering only for the
        # validated rank-0 schedule. Other manual/autotuned configs remain on
        # their resolved standard lowering.
        cfg = _flash_resident_softmax_config(cfg)
    stat_release_mapping = (
        FlashStatReleaseMapping.SAME_SLOT
        if use_causal_stateful_softmax
        else FlashStatReleaseMapping.CROSS_SLOT
    )
    use_whole_row_tmem_reduce = (
        use_tmem_row_reduce and not is_causal and not cfg.softmax_disc
    )
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
    use_packed_f16x2_xu = (
        dense_packed_exp2_mode is FlashPackedExp2Mode.ALL_XU
        and hardware_capabilities.supports_packed_f16x2_exp2
        and dense_target_lowering_applies
        and cfg.exp2_impl == "split"
        and cfg.exp2_packet in _FLASH_DEG1_EXP2_PACKETS
    )
    effective_probability_log2_shift = (
        probability_log2_shift
        if use_packed_f16x2_xu or dense_resident_value_graph_candidate
        else 0
    )
    use_cga2_local_cta = cfg.use_cga2_local_cta
    use_clc_scheduler = cfg.use_clc_scheduler
    separate_kv_rings = cfg.separate_kv_rings
    fa4_stat_handoff = cfg.stat_transport in ("single", "single_final")
    # Match FA4's one-slot statistics pipeline: the softmax role acquires slot
    # ownership after publishing P, while correction releases the slot selected
    # by the architecture policy. Unsupported schedules retain the conservative
    # handoff below.
    fa4_stat_pipeline = (
        fa4_stat_handoff
        and (
            (not is_causal and cfg.exp2_impl == "split")
            or (use_causal_resident_native and cfg.exp2_impl == "xu")
        )
        and not cfg.softmax_disc
        and cfg.rescale_threshold > 0.0
        and (
            cfg.exp2_packet != _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET
            or dense_resident_value_graph_candidate
        )
    )
    # The MMA warp's PV -> next-QK order makes each alpha slot safe to reuse
    # without a per-iteration empty acknowledgement. Keep a single terminal
    # handoff per work item before softmax overwrites alpha with row sum.
    final_only_stat_pipeline = (
        fa4_stat_pipeline and cfg.stat_transport == "single_final"
    )
    acknowledged_stat_pipeline = fa4_stat_pipeline and not final_only_stat_pipeline
    if dense_resident_value_graph_candidate:
        assert acknowledged_stat_pipeline
    if use_causal_stateful_softmax:
        assert acknowledged_stat_pipeline
    verified_shared_memory_bytes: int | None = None
    if separate_kv_rings:
        assert not use_cga2_local_cta
        assert not use_clc_scheduler
        assert not cfg.tensor_4d_tma
        if not use_2cta_instrs:
            assert not cfg.epi_tma
            assert not cfg.epi_stg
        verified_schedule = verify_flash_schedule(
            build_fa4_schedule(
                FlashScheduleSpec(
                    head_dim=head_dim,
                    kv_depth=kv_stage,
                    query_slots_per_cta=q_stage,
                    cta_count=2 if use_2cta_instrs else 1,
                    separate_kv=True,
                    causal=is_causal,
                    multicast_kv=use_2cta_instrs,
                    cooperative_mma=use_2cta_instrs,
                    persistent=persistent,
                    kv_iterations=num_kv if persistent else None,
                    stage_output=cfg.epi_tma or cfg.epi_stg,
                    split_p_arrive=cfg.split_p_arrive,
                    stat_depth=1 if fa4_stat_handoff else 2,
                    pipelined_stat_handoff=acknowledged_stat_pipeline,
                    final_only_stat_handoff=final_only_stat_pipeline,
                )
            )
        )
        kv_stage = verified_schedule.spec.kv_depth
        separate_kv_rings = verified_schedule.spec.separate_kv
        verified_shared_memory_bytes = verified_schedule.schedule.shared_memory_bytes
    elif use_2cta_instrs:
        verified_schedule = verify_flash_schedule(
            build_fa4_schedule(
                FlashScheduleSpec(
                    head_dim=head_dim,
                    kv_depth=kv_stage,
                    query_slots_per_cta=q_stage,
                    cta_count=2,
                    causal=is_causal,
                    multicast_kv=True,
                    cooperative_mma=True,
                    persistent=persistent,
                    kv_iterations=num_kv if persistent else None,
                    stage_output=cfg.epi_tma or cfg.epi_stg,
                    split_p_arrive=cfg.split_p_arrive,
                    stat_depth=1 if fa4_stat_handoff else 2,
                    pipelined_stat_handoff=acknowledged_stat_pipeline,
                    final_only_stat_handoff=final_only_stat_pipeline,
                )
            )
        )
        kv_stage = verified_schedule.spec.kv_depth
        verified_shared_memory_bytes = verified_schedule.schedule.shared_memory_bytes
    elif use_causal_resident_native:
        assert causal_split_equal_iteration_proof.proven
        verified_schedule = verify_flash_schedule(
            build_fa4_schedule(
                FlashScheduleSpec(
                    head_dim=head_dim,
                    kv_depth=kv_stage,
                    query_slots_per_cta=q_stage,
                    causal=True,
                    persistent=False,
                    stage_output=cfg.epi_tma or cfg.epi_stg,
                    split_p_arrive=cfg.split_p_arrive,
                    stat_depth=1,
                    pipelined_stat_handoff=True,
                    stat_release_mapping=stat_release_mapping,
                    query_slots_have_equal_kv_iterations=(
                        causal_split_equal_iteration_proof.proven
                    ),
                )
            )
        )
        verified_shared_memory_bytes = verified_schedule.schedule.shared_memory_bytes
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
    exp2_codegen = _flash_disc_exp2_codegen_params(
        cfg.exp2_packet, cfg.e2e_freq, cfg.e2e_res
    )
    if use_packed_f16x2_xu:
        exp2_codegen = exp2_codegen._replace(e2e_res=0)
    if exp2_codegen.degree2:
        hd128_bf16_dense = (
            hd == 128
            and io_dtype == "cutlass.BFloat16"
            and not is_causal
            and use_2cta_instrs
        )
        hd128_bf16_causal_resident = (
            hd == 128
            and io_dtype == "cutlass.BFloat16"
            and is_causal
            and cfg.exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
        )
        hd64_bf16_causal_hybrid = (
            hd == 64
            and io_dtype == "cutlass.BFloat16"
            and is_causal
            and _flash_causal_hd64_hybrid_num_kv_supported(num_kv)
            and cfg.exp2_packet == _FLASH_HYBRID_EXP2_PACKET
            and not has_lse
            and _standard_causal_score_plan_supported(score_plan)
            and causal_split_proof.proven
            and causal_desc_kv
            and cfg.causal_loop_split
            and cfg.softmax_disc
            and cfg.disc_pipe_depth >= 2
            and cfg.split_p_arrive
            and not use_2cta_instrs
        )
        assert (
            (hd == 64 and io_dtype == "cutlass.Float16")
            or hd128_bf16_dense
            or hd128_bf16_causal_resident
            or hd64_bf16_causal_hybrid
        )
        assert cfg.q_tile_count == 2
        assert not use_cga2_local_cta
        assert not separate_kv_rings
        assert cfg.p_store_repetition == 16
        assert cfg.s_load_repetition == 32
        if is_causal:
            assert not use_2cta_instrs
            assert cfg.softmax_disc
            assert cfg.disc_pipe_depth >= 2
        else:
            assert use_2cta_instrs
            assert not cfg.softmax_disc or hd128_bf16_dense
    sp_whole_row_sum = use_causal_resident_native or cfg.sp_row_sum == "whole"
    cta_group_size = 2 if use_2cta_instrs else 1
    mma_m = 256 if use_2cta_instrs else 128
    hd64_2cta = use_2cta_instrs and hd == 64
    pfor2_count = 2 * 128 if use_2cta_instrs else 128
    pfor_count = (
        pfor2_count
        if cfg.skip_rescale_stats
        else 4 * 128
        if use_2cta_instrs
        else 2 * 128
    )
    pfor_self_cta_rank = "None" if hd64_2cta else "flash_mma_tile_coord_v"
    pfor_peer_arg = (
        f", cutlass.Int32(0), {pfor_self_cta_rank}" if use_2cta_instrs else ""
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
    kv_loop_bound = "flash_num_active_kv" if is_causal else "_flash_num_kv_tiles"
    kv_loop_bound_minus_1 = f"{kv_loop_bound} - 1"
    epi_smem = cfg.epi_tma or cfg.epi_stg
    role_chain = cfg.role_chain
    storage_extra_args = f", {epi_smem!s}, {use_clc_scheduler!s}, {cfg.clc_stages}"
    if separate_kv_rings:
        storage_extra_args += ", True"
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
    if cfg.causal_two_cta:
        causal_setup_pid = f"""
flash_pid = cutlass.Int32(cute.arch.cluster_idx()[0])
flash_m_pair_raw = flash_pid % {num_m_pairs}
flash_bh = flash_pid // {num_m_pairs}
flash_m_pair = {num_m_pairs - 1} - flash_m_pair_raw
flash_q_mma_tile0 = flash_m_pair * 2
flash_q_mma_tile1 = flash_q_mma_tile0 + 1
flash_m_tile0 = flash_q_mma_tile0 * 2 + flash_mma_tile_coord_v
flash_m_tile1 = flash_q_mma_tile1 * 2 + flash_mma_tile_coord_v"""
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
    if cfg.causal_two_cta:
        active_kv_setup = """
flash_num_active_kv = (
    (flash_q_mma_tile1 + cutlass.Int32(1)) * cutlass.Int32(2))"""
    elif is_causal:
        active_kv_setup = """
flash_num_active_kv = flash_m_tile1 + cutlass.Int32(1)"""
    else:
        active_kv_setup = ""
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
    if use_2cta_instrs:
        cta_group_setup = """
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
    elif use_cga2_local_cta:
        cta_group_setup = """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster())
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = None
flash_tcgen05_mcast_mask = None
"""
    elif use_clc_scheduler:
        cta_group_setup = """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cutlass.Int32(0)
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = cute.tiled_divide(
    cute.make_layout((1, 1, 1)), (_flash_qk_mma.thr_id.shape,))
flash_tcgen05_mcast_mask = None
"""
    else:
        cta_group_setup = """
flash_mma_tile_coord_v = cutlass.Int32(0)
flash_cga2_local_rank = cutlass.Int32(0)
flash_is_leader_cta = cutlass.Boolean(True)
flash_cta_layout_vmnk = None
flash_tcgen05_mcast_mask = None
"""
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
    scale_layout = (
        f"cute.make_layout(({q_stage} * 128))"
        if fa4_stat_handoff
        else f"cute.make_layout(({s_corr_stage}, {q_stage}, 128))"
    )

    def _scale_slot_expr(index: str, stage: str) -> str:
        if fa4_stat_handoff:
            return f"flash_scale_t[{stage} * 128 + flash_local_tidx]"
        return f"flash_scale_t[{index}, {stage}, flash_local_tidx]"

    smem_kv_setup = (
        "sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)\n"
        "sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner)"
        if separate_kv_rings
        else "sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)\n"
        "sV = cute.make_tensor(cute.recast_ptr(sK.iterator, _flash_vsl.inner), "
        "_flash_vsl.outer)"
    )
    if separate_kv_rings:
        kv_pipeline_setup = f"""flash_v_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2])){kv_tma_byte_scale}
flash_k_prod, flash_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr(){kv_tma_cluster_arg},
    defer_sync=True).make_participants()
flash_v_prod, flash_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_v_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr(){kv_tma_cluster_arg}).make_participants(){cluster_init_arrive}{cluster_init_wait}"""
    else:
        kv_pipeline_setup = f"""flash_kv_prod, flash_kv_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.kv_mbar_ptr.data_ptr(){kv_tma_cluster_arg}).make_participants(){cluster_init_arrive}{cluster_init_wait}{clc_setup}"""

    storage_size_assert = (
        f"\nassert _flash_storage_cls.size_in_bytes() == {verified_shared_memory_bytes}"
        if verified_shared_memory_bytes is not None
        else ""
    )
    setup = f"""
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
{cta_group_setup}{setup_pid}{active_kv_setup}
flash_local_tidx = tidx % 128{prefetch_descriptors}

_flash_storage_cls = _helion_flash_rt.flash_fa4_shared_storage(
    {hd}, {kv_stage}, {q_stage}, {s_corr_stage}, {io_dtype}{storage_extra_args}){storage_size_assert}
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
{smem_kv_setup}
flash_scale_t = storage.sScale.get_tensor({scale_layout})

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
{kv_pipeline_setup}

flash_qkt = _flash_qk_mma.get_slice(flash_mma_tile_coord_v)
flash_pvt = _flash_pv_mma.get_slice(flash_mma_tile_coord_v)
{setup_tma_partitions}
    """

    tmem_fragment_setup = f"""    flash_qk_acc_shape = flash_qkt.partition_shape_C(({mma_m}, 128))
    tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
    flash_pv_acc_shape = flash_pvt.partition_shape_C(({mma_m}, {hd}))
    tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)
"""
    tmem_base_setup = f"""{tmem_fragment_setup}    _helion_flash_rt.named_barrier_wait_unaligned(
        2, 13 * 32)
    flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
    tStS0_full = cute.make_tensor(flash_tmem_ptr, tStS.layout)
    tStS1_full = cute.make_tensor(flash_tmem_ptr + 128, tStS.layout)
    tOtO0_full = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)
    tOtO1_full = cute.make_tensor(flash_tmem_ptr + {256 + hd}, tOtO.layout)
{textwrap.indent(tmem_local_views.strip(), "    ")}
"""
    tmem_mma_setup = (
        tmem_base_setup
        + f"""    tSrQ = flash_qkt.make_fragment_A(sQ)
    tSrK = flash_qkt.make_fragment_B(sK)
    tOrV = flash_pvt.make_fragment_B(sV)
    tP = cute.make_tensor(tStS0.iterator, _flash_ptl.outer)
    tOrP0 = flash_pvt.make_fragment_A(tP)
    tOrP1 = cute.make_tensor(tOrP0.iterator + {p1_step}, tOrP0.layout)
"""
    )
    score_store_needed = (
        cfg.softmax_disc
        and not _flash_fa4_runtime_disc_score_plan_supported(score_plan)
    )
    score_store_setup = ""
    if score_store_needed:
        score_store_setup = """    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
    flash_tiled_score_st0 = cute_tcgen05_flash.make_tmem_copy(flash_score_st_atom, tStS0)
    flash_tiled_score_st1 = cute_tcgen05_flash.make_tmem_copy(flash_score_st_atom, tStS1)
    flash_thr_score_st0 = flash_tiled_score_st0.get_slice(flash_local_tidx)
    flash_thr_score_st1 = flash_tiled_score_st1.get_slice(flash_local_tidx)
    tScoreSTtS0 = flash_thr_score_st0.partition_D(tStS0)
    tScoreSTtS1 = flash_thr_score_st1.partition_D(tStS1)
"""
    flash_ld_op = "LdRed32x32bOp" if use_whole_row_tmem_reduce else "Ld32x32bOp"
    disc_ldred_setup = (
        f"""    flash_ldred_atom = cute.make_copy_atom(
        cute_tcgen05_flash.LdRed32x32bOp(cute_tcgen05_flash.Repetition({cfg.s_load_repetition})), cutlass.Float32)
    flash_tiled_ldred0 = cute_tcgen05_flash.make_tmem_copy(flash_ldred_atom, tStS0)
    flash_tiled_ldred1 = cute_tcgen05_flash.make_tmem_copy(flash_ldred_atom, tStS1)
    flash_thr_ldred0 = flash_tiled_ldred0.get_slice(flash_local_tidx)
    flash_thr_ldred1 = flash_tiled_ldred1.get_slice(flash_local_tidx)
    tLDRedtS0 = flash_thr_ldred0.partition_S(tStS0)
    tLDRedtS1 = flash_thr_ldred1.partition_S(tStS1)
"""
        if use_tmem_row_reduce and (cfg.softmax_disc or use_causal_resident_native)
        else ""
    )
    tmem_softmax_setup = (
        tmem_base_setup
        + f"""    cS = cute.make_identity_tensor((128, 128))
    tScS = flash_qkt.partition_C(cS)
    flash_ld_atom = cute.make_copy_atom(
        cute_tcgen05_flash.{flash_ld_op}(cute_tcgen05_flash.Repetition({
            cfg.s_load_repetition
        })), cutlass.Float32)
    flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
    flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
    flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
    flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
    tLDtS0 = flash_thr_ld0.partition_S(tStS0)
    tLDtS1 = flash_thr_ld1.partition_S(tStS1)
    tLDcS = flash_thr_ld0.partition_D(tScS)
{disc_ldred_setup.rstrip()}
{score_store_setup.rstrip()}

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
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({
            p_store_repetition
        })), cutlass.Float32)
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
    stage_local_softmax_setup = (
        use_causal_stateful_softmax or cfg.softmax_setup == "stage_local"
    ) and not mixed_p_store

    def _tmem_softmax_setup_stage(stage: str) -> str:
        ptr_expr = "flash_tmem_ptr" if stage == "0" else "flash_tmem_ptr + 128"
        p_store_iter = p0_store_iter if stage == "0" else p1_store_iter
        stage_score_store_setup = ""
        if score_store_needed:
            stage_score_store_setup = f"""    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
    flash_tiled_score_st{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_score_st_atom, tStS{stage})
    flash_thr_score_st{stage} = flash_tiled_score_st{stage}.get_slice(
        flash_local_tidx)
    tScoreSTtS{stage} = flash_thr_score_st{stage}.partition_D(tStS{stage})
"""
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
        stage_ldred_setup = (
            f"""    flash_ldred_atom = cute.make_copy_atom(
        cute_tcgen05_flash.LdRed32x32bOp(cute_tcgen05_flash.Repetition({cfg.s_load_repetition})), cutlass.Float32)
    flash_tiled_ldred{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_ldred_atom, tStS{stage})
    flash_thr_ldred{stage} = flash_tiled_ldred{stage}.get_slice(flash_local_tidx)
    tLDRedtS{stage} = flash_thr_ldred{stage}.partition_S(tStS{stage})
"""
            if use_tmem_row_reduce and (cfg.softmax_disc or use_causal_resident_native)
            else ""
        )
        return f"""    _helion_flash_rt.named_barrier_wait_unaligned(
        2, 13 * 32)
    flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
    flash_qk_acc_shape = flash_qkt.partition_shape_C(({mma_m}, 128))
    tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
    tStS{stage} = cute.make_tensor({ptr_expr}, tStS.layout)
    cS = cute.make_identity_tensor((128, 128))
    tScS = flash_qkt.partition_C(cS)
    flash_ld_atom = cute.make_copy_atom(
        cute_tcgen05_flash.{flash_ld_op}(cute_tcgen05_flash.Repetition({
            cfg.s_load_repetition
        })), cutlass.Float32)
    flash_tiled_ld{stage} = cute_tcgen05_flash.make_tmem_copy(
        flash_ld_atom, tStS{stage})
    flash_thr_ld{stage} = flash_tiled_ld{stage}.get_slice(flash_local_tidx)
    tLDtS{stage} = flash_thr_ld{stage}.partition_S(tStS{stage})
{coord_setup.rstrip()}
{stage_ldred_setup.rstrip()}
{stage_score_store_setup.rstrip()}

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
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({
            p_store_repetition
        })), cutlass.Float32)
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
        ({mma_m}, {hd}), flash_o_layout_enum, {io_dtype}, cutlass.Float32,
        flash_o_epi_subtile, use_2cta_instrs={use_2cta_instrs!s})
    flash_o_cO = cute.make_identity_tensor(({mma_m}, {hd}))
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
    tOsO0_corr = flash_pvt.get_slice(0).partition_C(sO[None, None, 0])
    tOsO1_corr = flash_pvt.get_slice(0).partition_C(sO[None, None, 1])
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
    scoped_corr_epi_smem = epi_smem and _flash_bool_env(
        "HELION_CUTE_FLASH_SCOPED_CORR_EPI", True
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
    local_epi_tma_setup = cfg.epi_tma_setup == "role_local"
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
    role_next = "elif" if role_chain else "if"
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
            + flash_clc_producer_state.index * cutlass.Int32(4)).align(16)
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(flash_clc_mbar, flash_clc_response_ptr)
        flash_clc_producer_state.advance()
        flash_clc_pipeline.consumer_wait(flash_clc_consumer_state)
        flash_clc_response_ptr = (
            flash_clc_response_base
            + flash_clc_consumer_state.index * cutlass.Int32(4)).align(16)
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
            persistent_loop=cfg.persistent_loop,
            prelude="none",
            total_tiles=total_tiles,
            num_m_pairs=num_m_pairs,
            use_2cta_instrs=use_2cta_instrs,
            use_cga2_local_cta=use_cga2_local_cta,
            clc_heads_per_batch=clc_heads_per_batch,
            tensor_4d_heads=tensor_4d_heads,
            recompute_tile_coords=cfg.recompute_tile_coords,
        )
    # The rank-sliced PV partitions add the CTA rank themselves. Start their
    # output coordinates at the rank-zero tile to avoid applying the rank twice.
    output_m_tile0 = "flash_q_mma_tile0 * 2" if use_2cta_instrs else "flash_m_tile0"
    output_m_tile1 = "flash_q_mma_tile1 * 2" if use_2cta_instrs else "flash_m_tile1"
    if use_tensor_4d_tma:
        epi_tma_gmem0 = f"tOgO_tma[None, {output_m_tile0}, 0, flash_head, flash_batch]"
        epi_tma_gmem1 = f"tOgO_tma[None, {output_m_tile1}, 0, flash_head, flash_batch]"
        epi_stg_mO_cur = "_flash_mOt[None, None, flash_head, flash_batch]"
        corr_gmem_o_index = (
            "flash_pvt.partition_C(flash_gO_corr)"
            "[None, None, None, {mtile}, 0, flash_head, flash_batch]"
        )
    else:
        epi_tma_gmem0 = f"tOgO_tma[None, {output_m_tile0}, 0, flash_bh]"
        epi_tma_gmem1 = f"tOgO_tma[None, {output_m_tile1}, 0, flash_bh]"
        epi_stg_mO_cur = "_flash_mOt[None, None, flash_bh]"
        corr_gmem_o_index = (
            "flash_pvt.partition_C(flash_gO_corr)"
            "[None, None, None, {mtile}, 0, flash_bh]"
        )

    def _epi_wait_corr_full(stage: str) -> str:
        return (
            "_helion_flash_rt.mbar_spin_wait(\n"
            f"            flash_corr_epi_full_ptr + {stage}, "
            f"flash_corr_epi_full_phase, {cfg.wait_hint})"
        )

    def _epi_release_corr_empty(stage: str) -> str:
        return (
            "with cute.arch.elect_one():\n"
            f"    cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + {stage})"
        )

    epi_empty_prearrive = """
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
        cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"""

    if cfg.epi_tma:
        epi_tma_release_corr_empty = """        with cute.arch.elect_one():
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"""
        epi_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
{textwrap.indent(epi_tma_setup.strip(), "    ") if local_epi_tma_setup else ""}
    flash_corr_epi_full_phase = cutlass.Int32(0){epi_empty_prearrive}"""
        epi_inner = f"""        {_epi_wait_corr_full("0")}
        with cute.arch.elect_one():
            cute.copy(_flash_tma_o, tOsO_tma[None, 0], {epi_tma_gmem0})
            cute.arch.cp_async_bulk_commit_group()
        {_epi_wait_corr_full("1")}
        with cute.arch.elect_one():
            cute.copy(_flash_tma_o, tOsO_tma[None, 1], {epi_tma_gmem1})
            cute.arch.cp_async_bulk_commit_group()
{epi_tma_release_corr_empty}
        flash_corr_epi_full_phase ^= 1"""
        epi_prelude = "decode"
    elif cfg.epi_stg:
        epi_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
{textwrap.indent(epi_stg_setup.strip(), "    ") if cfg.epi_stg else ""}
    flash_corr_epi_full_phase = cutlass.Int32(0){epi_empty_prearrive}"""
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
            return f"""        {_epi_wait_corr_full(stage)}
        tOgO_stg{stage} = {epi_stg_gmem_expr.format(stage=stage, hd=hd)}
        _helion_flash_rt.{epi_stg_store_fn}(
            flash_o_gmem_tiled_copy, flash_o_gmem_thr,
            tOsO_epi{stage}, tOgO_stg{stage}, {io_dtype})
{textwrap.indent(_epi_release_corr_empty(stage), "        ")}"""

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
        f"{role_next} warp_idx == {epi_warp}:",
        epi_head,
        epi_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
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
    load_k_prod = "flash_k_prod" if separate_kv_rings else "flash_kv_prod"
    load_v_prod = "flash_v_prod" if separate_kv_rings else "flash_kv_prod"
    load_k0 = f"""        flash_kve = {load_k_prod}.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, {load_first_kv}], tKsK[None, flash_kve.index],
                  tma_bar_ptr=flash_kve.barrier)"""
    load_v0 = f"""        flash_kve = {load_v_prod}.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, {load_first_kv}], tVsV[None, flash_kve.index],
                  tma_bar_ptr=flash_kve.barrier)"""
    load_prologue = _flash_fa4_load_prologue_for_order(
        cfg.first_load_order, load_q0, load_k0, load_q1, load_v0
    )
    load_inner = f"""{local_load_tma_block}{load_prologue}
        for {load_loop_var} in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
            flash_kv_next = {load_next_kv}
            flash_kve = {load_k_prod}.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, flash_kv_next], tKsK[None, flash_kve.index],
                      tma_bar_ptr=flash_kve.barrier)
            flash_kve = {load_v_prod}.acquire_and_advance()
            cute.copy(_flash_tma_v, tVgV[None, flash_kv_next], tVsV[None, flash_kve.index],
                      tma_bar_ptr=flash_kve.barrier)"""
    load_tail = (
        "    flash_k_prod.tail()\n    flash_v_prod.tail()\n    flash_q_prod.tail()"
        if separate_kv_rings
        else "    flash_kv_prod.tail()\n    flash_q_prod.tail()"
    )
    load_block = _flash_fa4_wrap(
        f"{role_next} warp_idx == {load_warp}:",
        load_head,
        load_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
        prelude=load_prelude_mode,
        tail=load_tail,
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
    def _mma_wait_p_ready(stage: str) -> str:
        return (
            "_helion_flash_rt.mbar_spin_wait("
            f"flash_pfor_ptr + {stage}, flash_pfor_phase, {cfg.wait_hint})"
        )

    def _mma_commit_s_ready(stage: str) -> str:
        return (
            "with cute.arch.elect_one():\n"
            f"    cute_tcgen05_flash.commit(flash_s_full_ptr + {stage}{commit_group_arg})"
        )

    def _mma_commit_o_ready(stage: str) -> str:
        return (
            "with cute.arch.elect_one():\n"
            f"    cute_tcgen05_flash.commit(flash_o_full_ptr + {stage}{commit_group_arg})"
        )

    mma_k_cons = "flash_k_cons" if separate_kv_rings else "flash_kv_cons"
    mma_v_cons = "flash_v_cons" if separate_kv_rings else "flash_kv_cons"
    if cfg.mma_ptx:
        # Each CTA owns 128 Q rows even when a CtaGroup.TWO MMA spans 256 rows.
        # SMEM descriptors use 16-byte units, so derive the stage stride from
        # the per-CTA tile rather than the cluster-wide MMA M extent.
        q_stage_stride = (mma_m // cta_group_size) * hd // 8
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
        pv_wait_hint_arg = f", wait_hint={cfg.wait_hint}"

        # STEADY-loop body (in-place P): PV(i) issued BEFORE QK(i+1) -- the PV-before-QK
        # program order is what protects the in-place P-over-S; the pfor2 wait is folded
        # inside the PV gemm.
        if cfg.mma_interleave:
            mma_steady_body = f"""
            # stage 0: PV0(i) (pfor2 wait folded inside the gemm) then QK0(i+1).
            {_mma_wait_p_ready("0")}
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o0_addr, {pv_p0},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP0[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv0_split_wait_arg}
                zero_init=flash_o_zero0{gemm_cta_group_arg}{pv_wait_hint_arg})
            flash_o_zero0 = cutlass.Boolean(False)
            flash_k_full = {mma_k_cons}.wait_and_advance()
{_qk_gemm("0", "flash_k_full")}
{textwrap.indent(_mma_commit_s_ready("0"), "            ")}
            # stage 1: PV1(i) (pfor2 folded) then QK1(i+1).
            {_mma_wait_p_ready("1")}
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o1_addr, {pv_p1},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP1[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv1_split_wait_arg}
                zero_init=flash_o_zero1{gemm_cta_group_arg}{pv_wait_hint_arg})
            flash_o_zero1 = cutlass.Boolean(False)
            flash_v_full.release()
{_qk_gemm("1", "flash_k_full")}
{textwrap.indent(_mma_commit_s_ready("1"), "            ")}
            flash_k_full.release()"""
        else:
            mma_steady_body = f"""
            # grouped issue: finish both PV stages before starting either next QK.
            {_mma_wait_p_ready("0")}
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o0_addr, {pv_p0},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP0[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv0_split_wait_arg}
                zero_init=flash_o_zero0{gemm_cta_group_arg}{pv_wait_hint_arg})
            flash_o_zero0 = cutlass.Boolean(False)
            {_mma_wait_p_ready("1")}
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
                flash_o1_addr, {pv_p1},
                _helion_flash_ptx.make_smem_desc_start_addr(
                    sV[None, None, None, flash_v_full.index].iterator),
                flash_v_smem_base,
                tOrP1[None, None, None, 0].layout,
                tOrV[None, None, None, 0].layout,
                "helion_flash_pv_mma_idesc",{pv1_split_wait_arg}
                zero_init=flash_o_zero1{gemm_cta_group_arg}{pv_wait_hint_arg})
            flash_o_zero1 = cutlass.Boolean(False)
            flash_v_full.release()
            flash_k_full = {mma_k_cons}.wait_and_advance()
{_qk_gemm("0", "flash_k_full")}
{textwrap.indent(_mma_commit_s_ready("0"), "            ")}
{_qk_gemm("1", "flash_k_full")}
{textwrap.indent(_mma_commit_s_ready("1"), "            ")}
            flash_k_full.release()"""
        mma_inner = f"""        flash_q0_full = flash_q_cons.wait_and_advance()
        flash_q1_full = flash_q_cons.wait_and_advance()

        # PROLOGUE: QK0(0)->S0, QK1(0)->S1 against K0; then release K0.
        flash_k0_full = {mma_k_cons}.wait_and_advance()
{textwrap.indent(_qk_gemm("0", "flash_k0_full").lstrip(), "        ")}
{textwrap.indent(_mma_commit_s_ready("0"), "        ")}
{textwrap.indent(_qk_gemm("1", "flash_k0_full").lstrip(), "        ")}
{textwrap.indent(_mma_commit_s_ready("1"), "        ")}
        flash_k0_full.release()

        # STEADY: i = 0..N-2. PV(i) for both stages interleaved with QK(i+1). o_zero
        # is the per-tile zero-init Boolean (True first tile, False after).
        flash_o_zero0 = cutlass.Boolean(True)
        flash_o_zero1 = cutlass.Boolean(True)
        for flash_i in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
            flash_v_full = {mma_v_cons}.wait_and_advance(){mma_steady_body}
            flash_pfor_phase ^= 1

        flash_q0_full.release()
        flash_q1_full.release()

        # EPILOGUE: PV(N-1) for both stages; commit O_full.
        flash_v_full = {mma_v_cons}.wait_and_advance()
        {_mma_wait_p_ready("0")}
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
            flash_o0_addr, {pv_p0},
            _helion_flash_ptx.make_smem_desc_start_addr(
                sV[None, None, None, flash_v_full.index].iterator),
            flash_v_smem_base,
            tOrP0[None, None, None, 0].layout,
            tOrV[None, None, None, 0].layout,
            "helion_flash_pv_mma_idesc",{pv0_split_wait_arg}
            zero_init=flash_o_zero0{gemm_cta_group_arg}{pv_wait_hint_arg})
{textwrap.indent(_mma_commit_o_ready("0"), "        ")}
        {_mma_wait_p_ready("1")}
        _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(
            flash_o1_addr, {pv_p1},
            _helion_flash_ptx.make_smem_desc_start_addr(
                sV[None, None, None, flash_v_full.index].iterator),
            flash_v_smem_base,
            tOrP1[None, None, None, 0].layout,
            tOrV[None, None, None, 0].layout,
            "helion_flash_pv_mma_idesc",{pv1_split_wait_arg}
            zero_init=flash_o_zero1{gemm_cta_group_arg}{pv_wait_hint_arg})
{textwrap.indent(_mma_commit_o_ready("1"), "        ")}
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
        flash_k0_full = {mma_k_cons}.wait_and_advance()
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
            flash_v_full = {mma_v_cons}.wait_and_advance()
            # stage 0: PV0(i) then QK0(i+1). STAGED-P: first 96 kv on pfor.
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase, {cfg.wait_hint})
            for flash_kp in cutlass.range_constexpr(flash_pv_split):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc0 | (flash_kp != 0))
                cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
            _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 0, flash_pfor_phase, {cfg.wait_hint})
            for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
                cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
            flash_O_acc0 = True
            flash_k_full = {mma_k_cons}.wait_and_advance()
            for flash_kp in cutlass.range(flash_nk, unroll_full=True):
                _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
                cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q0_full.index],
                          tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
            # stage 1: PV1(i) then QK1(i+1). STAGED-P: first 96 kv on pfor.
            _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase, {cfg.wait_hint})
            for flash_kp in cutlass.range_constexpr(flash_pv_split):
                _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc1 | (flash_kp != 0))
                cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                          tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
            _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 1, flash_pfor_phase, {cfg.wait_hint})
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
        flash_v_full = {mma_v_cons}.wait_and_advance()
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 0, flash_pfor_phase, {cfg.wait_hint})
        for flash_kp in cutlass.range_constexpr(flash_pv_split):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc0 | (flash_kp != 0))
            cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
        _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 0, flash_pfor_phase, {cfg.wait_hint})
        for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
            cute.gemm(_flash_pv_mma, tOtO0, tOrP0[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO0)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_o_full_ptr + 0{commit_group_arg})
        _helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase, {cfg.wait_hint})
        for flash_kp in cutlass.range_constexpr(flash_pv_split):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc1 | (flash_kp != 0))
            cute.gemm(_flash_pv_mma, tOtO1, tOrP1[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full.index], tOtO1)
        _helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + 1, flash_pfor_phase, {cfg.wait_hint})
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
    mma_tmem_teardown = """    flash_tmem.relinquish_alloc_permit()
    _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
    flash_tmem.free(flash_tmem_ptr)"""
    mma_role_inner = mma_inner
    if use_2cta_instrs:
        # Both allocator warps must allocate and free the paired TMEM block. Only
        # the leader CTA issues the CtaGroup.TWO MMA instructions.
        mma_role_inner = f"""        if flash_is_leader_cta:
{textwrap.indent(mma_inner, "    ")}"""
    mma_block = _flash_fa4_wrap(
        f"{role_next} warp_idx == 12:",
        mma_head,
        mma_role_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
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
    causal_hd128_resident = (
        cfg.exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
        and hd == 128
        and io_dtype == "cutlass.BFloat16"
        and is_causal
        and cfg.pipeline_family == "fa4"
        and causal_desc_kv
        and cfg.causal_loop_split
        and causal_split_proof.proven
        and cfg.softmax_disc
        and cfg.disc_pipe_depth == 2
        and cfg.s_load_repetition == 32
        and cfg.p_store_repetition == 16
        and split_p_arrive
        and not mixed_p_store
        and not use_2cta_instrs
        and _flash_fa4_runtime_disc_score_plan_supported(score_plan)
    )
    causal_hd128_resident_early_acquire = (
        causal_hd128_resident
        and cfg.exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS
    )
    softmax_segment = _FlashSoftmaxLoopSegment(
        loop_var="flash_kv_iter" if desc_kv else "flash_kv",
        loop_bound=kv_loop_bound,
        kv_expr=(
            f"{kv_loop_bound} - cutlass.Int32(1) - flash_kv_iter" if desc_kv else None
        ),
    )
    softmax_not_first = softmax_segment.not_first_condition
    defer_causal_minus_scale = (
        is_causal and _flash_fa4_runtime_disc_score_plan_supported(score_plan)
    )

    # FA4 rescale_threshold (alpha-pin) softmax block. When the running max grows by
    # less than the threshold (scale_log2*(old-new) >= -thresh), keep the OLD max and
    # pin alpha=1.0 so the correction warp's vote_ballot(alpha<1.0) is false for the
    # whole warp -> the O-rescale (t2r/mul/r2t/fence) on the correction->PV critical
    # path is SKIPPED. For causal chunked softmax, the private
    # ``flash_minus_max_scale`` computation follows alpha publication, shortening
    # the correction handoff while still consuming the possibly pinned max before
    # PASS2. Other paths retain their measured schedule; threshold==0.0 (fp8)
    # emits the prior block byte-identically.
    # Stage-parameterized, so one definition covers both softmax warpgroups (0/1).
    def _disc_alpha_blocks_for(
        not_first: str, *, known_not_first: bool = False
    ) -> tuple[str, str]:
        if cfg.rescale_threshold > 0.0:
            pin_condition = (
                f"flash_acc_log >= -{cfg.rescale_threshold}"
                if known_not_first
                else f"({not_first}) & (flash_acc_log >= -{cfg.rescale_threshold})"
            )
            if defer_causal_minus_scale:
                return (
                    f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {pin_condition}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)""",
                    "            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2",
                )
            return (
                f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {pin_condition}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2""",
                "",
            )

        return (
            """            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2
            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)""",
            "",
        )

    _disc_alpha_block, _disc_minus_scale_block = _disc_alpha_blocks_for(
        softmax_not_first
    )
    # The whole-row (non-disc) body computes flash_minus_max_scale BEFORE the exp2
    # PASS (which consumes it) and flash_alpha AFTER, so the alpha-pin reorders to a
    # PRE-exp block (decide alpha + pin the max) so the kept-old max feeds the exp
    # PASS via flash_minus_max_scale, and an empty POST-exp piece (no second alpha
    # compute). threshold==0.0 keeps the prior pre/post split byte-identically.
    if effective_probability_log2_shift:
        # Keep P, the running denominator, and the output accumulator in the same
        # power-of-two scaled domain.  The online-softmax recurrence preserves that
        # common scale across iterations and the final O / l normalization cancels
        # it.  Folding the offset into the existing scalar bias extends the packed
        # f16x2 exp2 tail range without a per-element scale-back instruction.
        sp_minus_max_scale = (
            f"(cutlass.Float32({float(effective_probability_log2_shift)!r})"
            " - flash_row_max_safe * _flash_scale_log2)"
        )
    else:
        sp_minus_max_scale = "(0.0 - flash_row_max_safe) * _flash_scale_log2"
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
            flash_minus_max_scale = {sp_minus_max_scale}"""
        _sp_alpha_post = ""
    else:
        _sp_alpha_pre = f"            flash_minus_max_scale = {sp_minus_max_scale}"
        _sp_alpha_post = """            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""

    def _softmax_wait_s_ready(stage: str) -> str:
        return (
            "_helion_flash_rt.mbar_spin_wait("
            f"flash_s_full_ptr + {stage}, flash_s_full_phase, {cfg.wait_hint})"
        )

    def _softmax_inner(
        stage: str,
        ld: str,
        ldt: str,
        st: str,
        stt: str,
        score_st: str,
        score_stt: str,
    ) -> str:
        score_m_tile_expr = (
            f"flash_q_mma_tile{stage} * cutlass.Int32(2)"
            if use_2cta_instrs
            else f"flash_m_tile{stage}"
        )

        def _format_disc_pass2(name: str, *, causal: bool) -> str:
            e2e_freq = cfg.masked_e2e_freq if causal else cfg.e2e_freq
            e2e_res = cfg.masked_e2e_res if causal else cfg.e2e_res
            disc_exp2_codegen = _flash_disc_exp2_codegen_params(
                cfg.exp2_packet, e2e_freq, e2e_res
            )
            e2e_freq = disc_exp2_codegen.e2e_freq
            e2e_res = disc_exp2_codegen.e2e_res
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
                mask_m_tile = (
                    f"flash_q_mma_tile{stage} * cutlass.Int32(2)"
                    if use_2cta_instrs
                    else f"flash_m_tile{stage}"
                )
                args.extend([mask_m_tile, "flash_kv"])
            args.append(io_dtype)
            if use_2cta_instrs:
                args.extend(["cutlass.Int32(0)", "flash_mma_tile_coord_v"])
            if disc_exp2_codegen.pair_batch != 1:
                args.extend(
                    [
                        f"pair_batch={disc_exp2_codegen.pair_batch}",
                        f"emu_batch={disc_exp2_codegen.emu_batch}",
                    ]
                )
            if disc_exp2_codegen.degree1_unmasked and not causal:
                args.append("degree1=True")
            elif disc_exp2_codegen.degree2:
                args.append("degree2=True")
            return f"_helion_flash_rt.{name}(" + ", ".join(args) + ")"

        pass2_call = _format_disc_pass2(_disc_pass2_name, causal=False)
        pass2_causal_call = _format_disc_pass2(_disc_pass2_causal_name, causal=True)

        def _format_disc_resident_pass2() -> str:
            disc_exp2_codegen = _flash_disc_exp2_codegen_params(
                cfg.exp2_packet, cfg.e2e_freq, cfg.e2e_res
            )
            assert not disc_exp2_codegen.degree1_unmasked
            e2e_offset = str(cfg.e2e_offset0 if stage == "0" else cfg.e2e_offset)
            resident_layout = (0, 1, 3)
            args = [
                *(f"flash_res_frg{chunk}" for chunk in resident_layout),
                ld,
                ldt,
                "tLDcS",
                st,
                stt,
                "tSTcS",
                "_flash_scale_log2",
                "flash_minus_max_scale",
                str(disc_exp2_codegen.e2e_freq),
                str(disc_exp2_codegen.e2e_res),
                e2e_offset,
                f"flash_pfor_ptr + {stage}",
                f"flash_pfor2_ptr + {stage}",
                io_dtype,
            ]
            if disc_exp2_codegen.pair_batch != 1:
                args.extend(
                    [
                        f"pair_batch={disc_exp2_codegen.pair_batch}",
                        f"emu_batch={disc_exp2_codegen.emu_batch}",
                    ]
                )
            if disc_exp2_codegen.degree2:
                args.append("degree2=True")
            return (
                "_helion_flash_rt.fa4_disc_exp_convert_store_resident3_013_prefetch2("
                + ", ".join(args)
                + ")"
            )

        resident_pass2_call = (
            _format_disc_resident_pass2() if causal_hd128_resident else ""
        )
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
        corr_prod_index = "0" if fa4_stat_handoff else "flash_s_corr_prod_index"
        corr_prod_advance = (
            ""
            if fa4_stat_pipeline
            else "flash_s_corr_prod_phase ^= 1"
            if fa4_stat_handoff
            else textwrap.dedent(
                """\
                flash_s_corr_prod_index ^= 1
                if flash_s_corr_prod_index == 0:
                    flash_s_corr_prod_phase ^= 1
                """
            ).strip()
        )

        def _corr_acquire_alpha(indent: str) -> str:
            return (
                ""
                if fa4_stat_pipeline
                else f"""{indent}_helion_flash_rt.mbar_spin_wait(
{indent}    {corr_empty_ptr} + {corr_prod_index}, flash_s_corr_prod_phase, {cfg.wait_hint})
"""
            )

        def _corr_publish_alpha(indent: str, *, acquire: bool = True) -> str:
            if cfg.skip_rescale_stats:
                return ""
            publish_alpha_store = (
                f"{indent}{_scale_slot_expr(corr_prod_index, stage)} = flash_alpha\n"
            )
            publish_alpha_advance = corr_prod_advance.replace("\n", f"\n{indent}")
            producer_acquire = _corr_acquire_alpha(indent) if acquire else ""
            return f"""{producer_acquire}{publish_alpha_store.rstrip()}
{indent}_helion_flash_rt.named_barrier_arrive_unaligned(
{indent}    {3 + int(stage) * 4} + warp_idx % 4, 64)
{indent}{publish_alpha_advance}"""

        corr_publish_alpha = _corr_publish_alpha("                ")
        corr_publish_dummy = f"""                _helion_flash_rt.named_barrier_arrive_unaligned(
                    {3 + int(stage) * 4} + warp_idx % 4, 64)"""
        corr_rowsum_advance = corr_prod_advance.replace("\n", "\n        ")
        if final_only_stat_pipeline:
            final_only_phase_advance = (
                "        flash_s_corr_prod_phase ^= 1\n" if persistent else ""
            )
            rowsum_producer_acquire = f"""        _helion_flash_rt.mbar_spin_wait(
            {corr_empty_ptr} + 0, flash_s_corr_prod_phase, {cfg.wait_hint})
{final_only_phase_advance}"""
        elif acknowledged_stat_pipeline:
            rowsum_producer_acquire = ""
        else:
            rowsum_producer_acquire = f"""        _helion_flash_rt.mbar_spin_wait(
            {corr_empty_ptr} + {corr_prod_index}, flash_s_corr_prod_phase, {cfg.wait_hint})
"""
        softmax_row_sum_expr = (
            "flash_softmax.row_sum[0]"
            if use_causal_stateful_softmax
            else "flash_row_sum"
        )
        corr_publish_rowsum = f"""{rowsum_producer_acquire}        {_scale_slot_expr(corr_prod_index, stage)} = {softmax_row_sum_expr}
        _helion_flash_rt.named_barrier_arrive_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
{lse_store}
        {corr_rowsum_advance}"""
        # The correction warp consumes per-KV alpha handoffs during the loop and
        # one final row-sum handoff after the loop.
        if cfg.softmax_disc:
            if not _flash_fa4_runtime_disc_score_plan_supported(score_plan):
                score_transform = _flash_score_transform_block(
                    score_plan,
                    indent="                ",
                    score_tensor="flash_disc_frg",
                    coord_tensor="tLDcS[None, flash_ci, None, None]",
                    bh_expr="flash_bh",
                    m_tile_expr=score_m_tile_expr,
                    kv_tile_expr="flash_kv",
                    chunk_expr="flash_ci",
                    io_dtype=io_dtype,
                )
                rowmax_block = f"""            flash_ld_shape = tLDcS[None, 0, None, None].shape
            for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):
                flash_disc_frg = cute.make_rmem_tensor(flash_ld_shape, cutlass.Float32)
                cute.copy({ld}, {ldt}[None, flash_ci, None, None], flash_disc_frg){score_transform}
                flash_row_max = _helion_flash_rt._fmax_reduce_chunk_balanced(flash_disc_frg, flash_row_max)
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
                iteration = _FlashOnlineSoftmaxIteration(
                    load_and_reduce=rowmax_block,
                    alpha_pre_probability=_disc_alpha_block,
                    alpha_publish_pre_probability=alpha_publish,
                    probability_update=f"""{_disc_minus_scale_block}
{pass2_block}""",
                )
                loop = _format_fa4_online_softmax_loop(
                    softmax_segment,
                    iteration,
                    stage=stage,
                    wait_hint=cfg.wait_hint,
                )
                return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{loop}
{corr_publish_rowsum}"""
            rowmax_dense_call = (
                f"_helion_flash_rt.disc_rowmax_ldred("
                f"flash_tiled_ldred{stage}, tLDRedtS{stage}, tLDcS, "
                "flash_row_max, flash_LD_CHUNKS)"
                if use_tmem_row_reduce
                else f"_helion_flash_rt.fa4_disc_rowmax_balanced("
                f"{ld}, {ldt}, tLDcS, flash_row_max, flash_LD_CHUNKS)"
            )
            direct_dense_rowmax = f"            flash_row_max = {rowmax_dense_call}"
            direct_dense_pass2 = f"            flash_p_sum = {pass2_call}"
            if causal_hd128_resident:
                resident_layout = (0, 1, 3)
                rowmax_lines = [
                    "            flash_res_ld_shape = tLDcS[None, 0, None, None].shape"
                ]
                for chunk in range(4):
                    fragment = (
                        f"flash_res_frg{chunk}"
                        if chunk in resident_layout
                        else "flash_res_rowmax_tmp"
                    )
                    rowmax_lines.extend(
                        [
                            (
                                f"            {fragment} = cute.make_rmem_tensor("
                                "flash_res_ld_shape, cutlass.Float32)"
                            ),
                            (
                                f"            cute.copy({ld}, {ldt}[None, {chunk}, "
                                f"None, None], {fragment})"
                            ),
                            (
                                "            flash_row_max = "
                                "_helion_flash_rt._fmax_reduce_chunk_balanced("
                                f"{fragment}, flash_row_max)"
                            ),
                        ]
                    )
                rowmax_lines.append(
                    "            cute.arch.fence_view_async_tmem_load()"
                )
                unmasked_dense_rowmax = "\n".join(rowmax_lines)
                unmasked_dense_pass2 = (
                    f"            flash_p_sum = {resident_pass2_call}"
                )
            else:
                unmasked_dense_rowmax = direct_dense_rowmax
                unmasked_dense_pass2 = direct_dense_pass2

            def _format_disc_loop(
                segment: _FlashSoftmaxLoopSegment,
                loop_rowmax_block: str,
                loop_pass2_block: str,
                *,
                zero_first_tile: bool = False,
                early_alpha_acquire: bool = False,
            ) -> str:
                assert segment.continues_previous_segment or not early_alpha_acquire
                if cfg.skip_rescale_stats:
                    alpha_publish = ""
                elif segment.continues_previous_segment:
                    alpha_publish = _corr_publish_alpha(
                        "            ", acquire=not early_alpha_acquire
                    )
                else:
                    alpha_publish = f"""            if {segment.not_first_condition}:
{corr_publish_alpha}"""
                early_acquire = (
                    _corr_acquire_alpha("            ")
                    if early_alpha_acquire and not cfg.skip_rescale_stats
                    else ""
                )
                if zero_first_tile:
                    loop_rowmax_block = f"""            if {segment.not_first_condition}:
{textwrap.indent(loop_rowmax_block, "    ")}"""
                    loop_pass2_block = f"""            if {segment.not_first_condition}:
{textwrap.indent(loop_pass2_block, "    ")}
            else:
                flash_p_sum = {zero_pass2_call}"""
                alpha_block, minus_scale_block = _disc_alpha_blocks_for(
                    segment.not_first_condition,
                    known_not_first=segment.continues_previous_segment,
                )
                iteration = _FlashOnlineSoftmaxIteration(
                    load_and_reduce=loop_rowmax_block,
                    alpha_pre_probability=alpha_block,
                    alpha_publish_pre_probability=alpha_publish,
                    probability_update=f"""{minus_scale_block}
            flash_p_sum = cutlass.Float32(0.0)
{loop_pass2_block}""",
                    pre_max_update=early_acquire.rstrip(),
                )
                return _format_fa4_online_softmax_loop(
                    segment,
                    iteration,
                    stage=stage,
                    wait_hint=cfg.wait_hint,
                )

            if is_causal:
                mask_m_tile = score_m_tile_expr
                rowmax_causal_call = (
                    f"_helion_flash_rt.fa4_disc_rowmax_causal_balanced("
                    f"{ld}, {ldt}, tLDcS, flash_row_max, flash_LD_CHUNKS, "
                    f"{mask_m_tile}, flash_kv)"
                )
                direct_causal_rowmax = (
                    f"            flash_row_max = {rowmax_causal_call}"
                )
                direct_causal_pass2 = f"            flash_p_sum = {pass2_causal_call}"
                if (
                    causal_desc_kv
                    and cfg.causal_loop_split
                    and causal_split_proof.proven
                ):
                    masked_segment, unmasked_segment = (
                        _flash_causal_split_softmax_segments(
                            kv_loop_bound,
                            stage,
                            split_range_proof=causal_split_proof,
                        )
                    )
                    masked_loop = _format_disc_loop(
                        masked_segment,
                        direct_causal_rowmax,
                        direct_causal_pass2,
                        zero_first_tile=stage == "0",
                    )
                    # The proven split always executes a nonempty masked prefix
                    # before entering its unmasked suffix.
                    unmasked_loop = _format_disc_loop(
                        unmasked_segment,
                        unmasked_dense_rowmax,
                        unmasked_dense_pass2,
                        early_alpha_acquire=causal_hd128_resident_early_acquire,
                    )
                    return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{masked_loop}
{unmasked_loop}
{corr_publish_rowsum}"""
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
                softmax_segment,
                rowmax_block,
                pass2_block,
                zero_first_tile=is_causal and causal_desc_kv and stage == "0",
            )
            return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{loop}
{corr_publish_rowsum}"""
        score_transform = _flash_score_transform_block(
            score_plan,
            indent="            ",
            score_tensor="tLDrS",
            coord_tensor="tLDcS",
            bh_expr="flash_bh",
            m_tile_expr=score_m_tile_expr,
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
        use_resident_value_graph = (
            use_causal_resident_value_graph or dense_resident_value_graph_candidate
        )
        sp_corr_publish_alpha = ""
        if use_resident_value_graph:
            assert split_p_arrive
            assert not mixed_p_store
            resident_pfor_args = (
                f""",
                pfor_peer_cta_rank=cutlass.Int32(0),
                pfor_self_cta_rank={pfor_self_cta_rank}"""
                if dense_resident_value_graph_candidate
                else ""
            )
            sp_exp_block = f"""            flash_row_sum = _helion_flash_rt.resident_softmax_value_graph(
                tLDrS, {st}, {stt}, tSTcS, _flash_scale_log2,
                flash_minus_max_scale, flash_pfor_ptr + {stage},
                flash_pfor2_ptr + {stage}, flash_P_STORE_SPLIT,
                {corr_empty_ptr} + 0,
                flash_s_corr_prod_phase, flash_row_sum * flash_alpha,
                {cfg.wait_hint}{resident_pfor_args})"""
            sp_p_store_block = ""
        elif cfg.exp2_impl == "split" or use_causal_resident_native:
            sp_e2e_offset = (
                "0"
                if exp2_codegen.e2e_res == 0
                else str(cfg.e2e_offset0 if stage == "0" else cfg.e2e_offset)
            )
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
                    str(exp2_codegen.e2e_freq),
                    str(exp2_codegen.e2e_res),
                    sp_e2e_offset,
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
                    str(exp2_codegen.e2e_freq),
                    str(exp2_codegen.e2e_res),
                    sp_e2e_offset,
                    f"flash_pfor_ptr + {stage}",
                    f"flash_pfor2_ptr + {stage}" if split_p_arrive else "None",
                    "flash_P_STORE_SPLIT",
                    "flash_P_STORE_CHUNKS",
                    io_dtype,
                ]
            if use_2cta_instrs:
                sp_pass2_args.extend(["cutlass.Int32(0)", pfor_self_cta_rank])
                if hd64_2cta and not mixed_p_store:
                    sp_pass2_args.append("early_split_publish=True")
            if exp2_codegen.pair_batch != 1:
                sp_pass2_args.extend(
                    [
                        f"pair_batch={exp2_codegen.pair_batch}",
                        f"emu_batch={exp2_codegen.emu_batch}",
                    ]
                )
            if exp2_codegen.degree1_unmasked:
                sp_pass2_args.append("degree1=True")
            elif exp2_codegen.degree2:
                sp_pass2_args.append("degree2=True")
            if use_packed_f16x2_xu:
                assert not mixed_p_store
                sp_pass2_args.append("f16x2_xu=True")
            sp_exp_block = (
                f"            flash_p_sum = _helion_flash_rt.{sp_pass2_name}("
                + ", ".join(sp_pass2_args)
                + ")"
            )
            sp_p_store_block = ""
        else:
            sp_exp_block = softmax_exp_block
            sp_p_store_block = p_store_block
        if not cfg.skip_rescale_stats:
            if acknowledged_stat_pipeline:
                sp_corr_publish_alpha = f"""            if {softmax_not_first}:
{corr_publish_alpha}
            else:
{corr_publish_dummy}"""
            else:
                sp_corr_publish_alpha = f"""            if {softmax_not_first}:
{corr_publish_alpha}"""
        fa4_publish_alpha_before_exp = fa4_stat_handoff and (
            cfg.rescale_threshold > 0.0 or fa4_stat_pipeline
        )
        sp_alpha_publish_pre = (
            sp_corr_publish_alpha if fa4_publish_alpha_before_exp else ""
        )
        sp_alpha_publish_post = (
            "" if fa4_publish_alpha_before_exp else sp_corr_publish_alpha
        )
        fa4_entry_stat_acquire = (
            f"""        _helion_flash_rt.mbar_spin_wait(
            {corr_empty_ptr} + 0, flash_s_corr_prod_phase, {cfg.wait_hint})
        flash_s_corr_prod_phase ^= 1
"""
            if acknowledged_stat_pipeline
            else ""
        )
        fa4_post_p_stat_acquire = (
            f"""            _helion_flash_rt.mbar_spin_wait(
                {corr_empty_ptr} + 0, flash_s_corr_prod_phase, {cfg.wait_hint})
            flash_s_corr_prod_phase ^= 1
"""
            if acknowledged_stat_pipeline
            else ""
        )
        if use_whole_row_tmem_reduce:
            assert not score_transform
        sp_score_load = (
            f"""            tLDrS_red = cute.make_rmem_tensor(
                ((1, 1), *tLDrS.shape[1:]), cutlass.Float32)
            cute.copy({ld}, {ldt}, (tLDrS, tLDrS_red))
            cute.arch.fence_view_async_tmem_load()
            flash_hw_row_max = cutlass.Float32(-cutlass.Float32.inf)
            for flash_red_i in cutlass.range_constexpr(cute.size(tLDrS_red.shape)):
                flash_hw_row_max = cute.arch.fmax(
                    flash_hw_row_max, tLDrS_red[flash_red_i])"""
            if use_whole_row_tmem_reduce
            else f"""            cute.copy({ld}, {ldt}, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}"""
        )
        sp_rowmax = (
            "            flash_row_max = cute.arch.fmax("
            "flash_row_max, flash_hw_row_max)"
            if use_whole_row_tmem_reduce
            else "            flash_row_max = "
            "_helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
        )
        if use_causal_resident_native:
            assert acknowledged_stat_pipeline
            assert cfg.exp2_impl == "xu"

            def _format_resident_causal_loop(
                segment: _FlashSoftmaxLoopSegment,
                score_load: str,
                rowmax_update: str,
            ) -> str:
                pin_condition = (
                    f"flash_acc_log >= -{cfg.rescale_threshold}"
                    if segment.continues_previous_segment
                    else f"({segment.not_first_condition}) & "
                    f"(flash_acc_log >= -{cfg.rescale_threshold})"
                )
                alpha_pre = f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {pin_condition}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"""
                if cfg.skip_rescale_stats:
                    alpha_publish = ""
                elif segment.continues_previous_segment:
                    alpha_publish = _corr_publish_alpha("            ")
                else:
                    alpha_publish = f"""            if {segment.not_first_condition}:
{_corr_publish_alpha("                ")}
            else:
                _helion_flash_rt.named_barrier_arrive_unaligned(
                    {3 + int(stage) * 4} + warp_idx % 4, 64)"""
                post_p_stat_acquire = (
                    ""
                    if use_causal_resident_value_graph
                    else fa4_post_p_stat_acquire.rstrip()
                )
                statistics_update = (
                    "            flash_s_corr_prod_phase ^= 1"
                    if use_causal_resident_value_graph
                    else _FLASH_ONLINE_STATISTICS_UPDATE
                )
                iteration = _FlashOnlineSoftmaxIteration(
                    load_and_reduce=f"""            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
{score_load}
{rowmax_update}""",
                    alpha_pre_probability=alpha_pre,
                    alpha_publish_pre_probability=alpha_publish,
                    probability_update=sp_exp_block,
                    statistics_acquire=post_p_stat_acquire,
                    statistics_update=statistics_update,
                )
                return _format_fa4_online_softmax_loop(
                    segment,
                    iteration,
                    stage=stage,
                    wait_hint=cfg.wait_hint,
                )

            masked_score_load = f"""            cute.copy({ld}, {ldt}, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}"""
            masked_rowmax = (
                "            flash_row_max = "
                "_helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
            )
            unmasked_score_load = f"""            tLDrS_red = cute.make_rmem_tensor(
                ((1, 1), *tLDrS.shape[1:]), cutlass.Float32)
            cute.copy(
                flash_tiled_ldred{stage}, tLDRedtS{stage}, (tLDrS, tLDrS_red))
            cute.arch.fence_view_async_tmem_load()
            flash_hw_row_max = cutlass.Float32(-cutlass.Float32.inf)
            for flash_red_i in cutlass.range_constexpr(cute.size(tLDrS_red.shape)):
                flash_hw_row_max = cute.arch.fmax(
                    flash_hw_row_max, tLDrS_red[flash_red_i])"""
            unmasked_rowmax = (
                "            flash_row_max = "
                "cute.arch.fmax(flash_row_max, flash_hw_row_max)"
            )
            if use_causal_stateful_softmax:
                assert not use_causal_resident_value_graph

                def _format_stateful_softmax_step(
                    score_load: str,
                    rowmax_call: str,
                    *,
                    indent: str,
                    is_first: bool,
                ) -> str:
                    alpha_publish = (
                        f"""{indent}_helion_flash_rt.named_barrier_arrive_unaligned(
{indent}    {3 + int(stage) * 4} + warp_idx % 4, 64)"""
                        if is_first
                        else _corr_publish_alpha(indent)
                    )
                    row_sum_first_arg = ", True" if is_first else ""
                    score_load_at_indent = textwrap.indent(
                        textwrap.dedent(score_load), indent
                    )
                    return f"""{indent}{_softmax_wait_s_ready(stage)}
{indent}flash_s_full_phase ^= 1
{indent}tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
{score_load_at_indent}
{indent}flash_row_max_safe, flash_alpha = flash_softmax.{rowmax_call}
{alpha_publish}
{indent}flash_softmax.scale_subtract_rowmax(tLDrS, flash_row_max_safe)
{indent}flash_tSrP_f32 = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
{indent}flash_tSrP = cute.make_tensor(
{indent}    cute.recast_ptr(flash_tSrP_f32.iterator, dtype={io_dtype}),
{indent}    tLDrS.layout)
{indent}flash_softmax.apply_exp2_convert(tLDrS, flash_tSrP)
{indent}for flash_ci in cutlass.range_constexpr(flash_P_STORE_SPLIT):
{indent}    cute.copy({st}, flash_tSrP_f32[None, None, flash_ci],
{indent}              {stt}[None, None, flash_ci])
{indent}cute.arch.fence_view_async_tmem_store()
{indent}_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + {stage})
{indent}for flash_ci in cutlass.range_constexpr(
{indent}        flash_P_STORE_SPLIT, flash_P_STORE_CHUNKS):
{indent}    cute.copy({st}, flash_tSrP_f32[None, None, flash_ci],
{indent}              {stt}[None, None, flash_ci])
{indent}cute.arch.fence_view_async_tmem_store()
{indent}_helion_flash_rt.mbarrier_arrive(flash_pfor2_ptr + {stage})
{indent}_helion_flash_rt.mbar_spin_wait(
{indent}    {corr_empty_ptr} + 0, flash_s_corr_prod_phase, {cfg.wait_hint})
{indent}flash_softmax.update_row_sum(
{indent}    tLDrS.load(), flash_alpha{row_sum_first_arg})
{indent}flash_s_corr_prod_phase ^= 1"""

                first_step = _format_stateful_softmax_step(
                    masked_score_load,
                    "update_row_max_masked(tLDrS.load(), True)",
                    indent="        ",
                    is_first=True,
                )
                masked_tail_step = _format_stateful_softmax_step(
                    masked_score_load,
                    "update_row_max_masked(tLDrS.load(), False)",
                    indent="            ",
                    is_first=False,
                )
                unmasked_step = _format_stateful_softmax_step(
                    unmasked_score_load,
                    "update_row_max_precomputed(flash_hw_row_max, False)",
                    indent="            ",
                    is_first=False,
                )
                return f"""{fa4_entry_stat_acquire}        flash_softmax = _helion_flash_rt.ResidentSoftmaxState.create(
            _flash_scale_log2, rescale_threshold={cfg.rescale_threshold})
        flash_kv = {kv_loop_bound} - cutlass.Int32(1)
{first_step}
        for flash_kv_mask_iter in cutlass.range(
                {kv_loop_bound} - flash_m_tile{stage} - cutlass.Int32(1), unroll=1):
            flash_kv = {kv_loop_bound} - cutlass.Int32(2) - flash_kv_mask_iter
{masked_tail_step}
        for flash_kv_unmask_iter in cutlass.range(flash_m_tile{stage}, unroll=1):
            flash_kv = flash_m_tile{stage} - cutlass.Int32(1) - flash_kv_unmask_iter
{unmasked_step}
{corr_publish_rowsum}"""
            masked_segment, unmasked_segment = _flash_causal_split_softmax_segments(
                kv_loop_bound,
                stage,
                split_range_proof=causal_split_proof,
            )
            masked_loop = _format_resident_causal_loop(
                masked_segment,
                masked_score_load,
                masked_rowmax,
            )
            unmasked_loop = _format_resident_causal_loop(
                unmasked_segment,
                unmasked_score_load,
                unmasked_rowmax,
            )
            return f"""{fa4_entry_stat_acquire}        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{masked_loop}
{unmasked_loop}
{corr_publish_rowsum}"""
        post_p_stat_acquire = (
            ""
            if dense_resident_value_graph_candidate
            else fa4_post_p_stat_acquire.rstrip()
        )
        statistics_update = (
            "            flash_s_corr_prod_phase ^= 1"
            if dense_resident_value_graph_candidate
            else _FLASH_ONLINE_STATISTICS_UPDATE
        )
        iteration = _FlashOnlineSoftmaxIteration(
            load_and_reduce=f"""            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
{sp_score_load}
{sp_rowmax}""",
            alpha_pre_probability=_sp_alpha_pre,
            alpha_publish_pre_probability=sp_alpha_publish_pre,
            probability_update=sp_exp_block,
            alpha_post_probability=_sp_alpha_post,
            alpha_publish_post_probability=sp_alpha_publish_post,
            statistics_acquire=post_p_stat_acquire,
            statistics_update=statistics_update,
            post_statistics=sp_p_store_block,
        )
        loop = _format_fa4_online_softmax_loop(
            softmax_segment,
            iteration,
            stage=stage,
            wait_hint=cfg.wait_hint,
        )
        return f"""{
            fa4_entry_stat_acquire
        }        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
{loop}
{corr_publish_rowsum}"""

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
    softmax_corr_prod_state = (
        "\n    flash_s_corr_prod_phase = cutlass.Int32(1)"
        if acknowledged_stat_pipeline
        else "\n    flash_s_corr_prod_phase = cutlass.Int32(0)"
    )
    if not fa4_stat_handoff:
        softmax_corr_prod_state = (
            "\n    flash_s_corr_prod_index = cutlass.Int32(0)" + softmax_corr_prod_state
        )
    softmax0_head = f"""    cute.arch.setmaxregister_increase({cfg.softmax_regs})
{softmax0_setup.rstrip()}
    flash_s_full_phase = cutlass.Int32(0){softmax_corr_prod_state}"""
    softmax1_head = f"""    cute.arch.setmaxregister_increase({cfg.softmax_regs})
{softmax1_setup.rstrip()}
    flash_s_full_phase = cutlass.Int32(0){softmax_corr_prod_state}"""
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
        "    _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)"
    )

    def _softmax_tail(stage: str) -> str:
        if not acknowledged_stat_pipeline:
            return tmem_dealloc_arrive
        return f"""    _helion_flash_rt.mbar_spin_wait(
        flash_s{stage}_corr_empty_ptr + 0, flash_s_corr_prod_phase, {cfg.wait_hint})
    _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)"""

    softmax0_inner = _softmax_inner(
        "0",
        "flash_tiled_ld0",
        "tLDtS0",
        "flash_tiled_st0",
        "tSTtS0",
        "flash_tiled_score_st0",
        "tScoreSTtS0",
    )
    softmax1_inner = _softmax_inner(
        "1",
        "flash_tiled_ld1",
        "tLDtS1",
        "flash_tiled_st1",
        "tSTtS1",
        "flash_tiled_score_st1",
        "tScoreSTtS1",
    )
    softmax0_block = _flash_fa4_wrap(
        f"{role_next} warp_idx < 4:",
        softmax0_head,
        softmax0_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
        prelude=softmax_prelude,
        tail=_softmax_tail("0"),
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
        f"{role_next} (warp_idx >= 4) & (warp_idx < 8):",
        softmax1_head,
        softmax1_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
        prelude=softmax_prelude,
        tail=_softmax_tail("1"),
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
    def _corr_release_p_ready(stage: str) -> str:
        return (
            f"_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + {stage}{pfor_peer_arg})"
        )

    def _corr_wait_o_ready(stage: str) -> str:
        return (
            "_helion_flash_rt.mbar_spin_wait("
            f"flash_o_full_ptr + {stage}, flash_o_full_phase, {cfg.wait_hint})"
        )

    def _corr_wait_epi_empty(stage: str) -> str:
        return (
            "_helion_flash_rt.mbar_spin_wait(\n"
            f"            flash_corr_epi_empty_ptr + {stage}, "
            f"flash_corr_epi_empty_phase, {cfg.wait_hint})"
        )

    def _corr_commit_epi_full(stage: str) -> str:
        return f"cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr + {stage})"

    corr_pfor_prearrive = (
        ""
        if cfg.skip_rescale_stats
        else f"""
    {_corr_release_p_ready("0")}
    {_corr_release_p_ready("1")}"""
    )
    corr_stat_empty_init = (
        ""
        if fa4_stat_pipeline
        else """
    cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)
    cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"""
        if fa4_stat_handoff
        else """
    cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)
    cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 1)
    cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)
    cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 1)"""
    )
    corr_cons_state = (
        "\n    flash_s_corr_cons_index = cutlass.Int32(0)"
        if not fa4_stat_handoff
        else ""
    )
    corr_epi_empty_phase_head = (
        "" if not epi_smem else "\n    flash_corr_epi_empty_phase = cutlass.Int32(0)"
    )
    corr_head = f"""    cute.arch.setmaxregister_decrease({cfg.corr_regs})
{tmem_base_setup.rstrip()}
{"" if scoped_corr_epi_smem else corr_epi_smem_setup.rstrip()}
    flash_o_full_phase = cutlass.Int32(0){corr_cons_state}{corr_epi_empty_phase_head}{corr_pfor_prearrive}{corr_stat_empty_init}"""
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

    split_corr_epi_handoff = scoped_corr_epi_smem and _flash_bool_env(
        "HELION_CUTE_FLASH_SPLIT_CORR_EPILOGUE_HANDOFF", True
    )
    scoped_corr_epi_fn = (
        "fa4_correction_epilogue_to_smem_scoped_2cta"
        if use_2cta_instrs
        else "fa4_correction_epilogue_to_smem_scoped"
    )
    scoped_corr_epi_handoff_fn = (
        "fa4_correction_epilogue_handoff_to_smem_scoped_2cta"
        if use_2cta_instrs
        else "fa4_correction_epilogue_handoff_to_smem_scoped"
    )

    def _corr_epi(stage: str, mtile: str) -> str:
        corr_cons_index = "0" if fa4_stat_handoff else "flash_s_corr_cons_index"
        scale_expr = _scale_slot_expr(corr_cons_index, stage)
        stat_empty_arrive = ""
        if not final_only_stat_pipeline:
            stat_empty_arrive = (
                f"        cute.arch.mbarrier_arrive("
                f"flash_s{stage}_corr_empty_ptr + {corr_cons_index})\n"
            )
        if not epi_smem:
            # Committed path: per-thread t2r (Ld32x32 Rep16) -> rescale -> cast ->
            # STG.E.128 straight to gmem (coord->linear address division per thread =
            # the epilogue IMAD/MOV overhead).
            return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        {_corr_wait_o_ready(stage)}
        tOgO_mma{stage} = {corr_gmem_o_index.format(mtile=mtile)}
        gO_epi{stage} = cute.zipped_divide(tOgO_mma{stage}, flash_epi_tiler{stage})
        tDgO{stage} = flash_thr_o_ld{stage}.partition_D(gO_epi{stage})
        for flash_i in cutlass.range(cute.size(tDtO{stage}, mode=[2])):
            flash_reg{stage} = cute.make_rmem_tensor(tDgO{stage}[None, None, 0].shape, cutlass.Float32)
            flash_rego{stage} = cute.make_rmem_tensor(tDgO{stage}[None, None, 0].shape, {io_dtype})
            cute.copy(flash_tiled_o_ld{stage}, tDtO{stage}[None, None, flash_i], flash_reg{stage})
            _helion_flash_rt._scale_fragment_packed_f32x2(
                flash_reg{stage}, flash_inv_sum{stage}){direct_relu_store.format(stage=stage)}
            flash_rego{stage}.store(flash_reg{stage}.load().to({io_dtype}))
            cute.autovec_copy(flash_rego{stage}, tDgO{stage}[None, None, flash_i])"""
        # FAITHFUL FA4 correction_epilogue port. Per corr-chunk: t2r 16 O cols ->
        # packed mul by inv_sum -> cast fp16 -> r2s into the sO smem buffer (stage
        # {stage} of the 2-staged _flash_osl) via the SMEM-STORE atom matched to
        # the t2r tiled copy. A dedicated epilogue warp then drains sO either by
        # TMA-O or by vector STG.
        if scoped_corr_epi_smem:
            if split_corr_epi_handoff:
                return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        {_corr_wait_o_ready(stage)}
        {_corr_wait_epi_empty(stage)}
        _helion_flash_rt.{scoped_corr_epi_fn}(
            flash_pvt, tOtO{stage}, sO[None, None, {stage}], flash_local_tidx,
            flash_inv_sum{stage}, {hd}, {cfg.corr_tile_size}, {io_dtype}{staged_relu_keyword})
        cute.arch.fence_view_async_shared()
        {_corr_commit_epi_full(stage)}
"""
            return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        _helion_flash_rt.{scoped_corr_epi_handoff_fn}(
            flash_o_full_ptr + {stage}, flash_o_full_phase,
            flash_corr_epi_empty_ptr + {stage}, flash_corr_epi_empty_phase,
            flash_corr_epi_full_ptr + {stage},
            flash_pvt, tOtO{stage}, sO[None, None, {stage}], flash_local_tidx,
            flash_inv_sum{stage}, {hd}, {cfg.corr_tile_size}, {io_dtype},
            {cfg.wait_hint}{staged_relu_keyword})
"""
        return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        _helion_flash_rt.fa4_correction_epilogue_handoff_to_smem(
            flash_o_full_ptr + {stage}, flash_o_full_phase,
            flash_corr_epi_empty_ptr + {stage}, flash_corr_epi_empty_phase,
            flash_corr_epi_full_ptr + {stage},
            flash_o_tiled_t2r, flash_o_tiled_r2s,
            tOtO{stage}_corr_t2r, tOsO{stage}_corr_r2s, tOcO_corr_t2r,
            flash_inv_sum{stage}, flash_o_corr_chunks, {cfg.wait_hint}{staged_relu_keyword})
"""

    corr_epi_empty_toggle = (
        "        flash_corr_epi_empty_phase ^= 1" if epi_smem else ""
    )
    corr_cons_index = "0" if fa4_stat_handoff else "flash_s_corr_cons_index"
    corr_empty0_early = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s0_corr_empty_ptr + {corr_cons_index})"
        if not is_causal and not fa4_stat_pipeline
        else ""
    )
    corr_empty1_early = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s1_corr_empty_ptr + {corr_cons_index})"
        if not is_causal and not fa4_stat_pipeline
        else ""
    )
    corr_empty0_late = (
        ""
        if not is_causal or fa4_stat_pipeline
        else "            cute.arch.mbarrier_arrive("
        "flash_s0_corr_empty_ptr + flash_s_corr_cons_index)"
    )
    corr_empty1_late = (
        ""
        if not is_causal or fa4_stat_pipeline
        else "            cute.arch.mbarrier_arrive("
        "flash_s1_corr_empty_ptr + flash_s_corr_cons_index)"
    )
    corr_cross_release0 = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s{'0' if stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT else '1'}_corr_empty_ptr + 0)"
        if acknowledged_stat_pipeline
        else ""
    )
    corr_cross_release1 = (
        "            cute.arch.mbarrier_arrive("
        f"flash_s{'1' if stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT else '0'}_corr_empty_ptr + 0)"
        if acknowledged_stat_pipeline
        else ""
    )

    corr_stage0 = f"""            _helion_flash_rt.named_barrier_wait_unaligned(
                3 + warp_idx % 4, 64)
            flash_a0 = {_scale_slot_expr(corr_cons_index, "0")}
{corr_empty0_early}
            flash_need_rescale0 = cute.arch.vote_ballot_sync(flash_a0 < 1.0) != 0
            if flash_need_rescale0:
                _helion_flash_rt.rescale_o_tmem(
                    tOtO0, flash_a0, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                cute.arch.fence_view_async_tmem_store()
            {_corr_release_p_ready("0")}
{corr_empty0_late}
{corr_cross_release0}"""
    corr_stage1 = f"""            _helion_flash_rt.named_barrier_wait_unaligned(
                7 + warp_idx % 4, 64)
            flash_a1 = {_scale_slot_expr(corr_cons_index, "1")}
{corr_empty1_early}
            flash_need_rescale1 = cute.arch.vote_ballot_sync(flash_a1 < 1.0) != 0
            if flash_need_rescale1:
                _helion_flash_rt.rescale_o_tmem(
                    tOtO1, flash_a1, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})
                cute.arch.fence_view_async_tmem_store()
            {_corr_release_p_ready("1")}
{corr_empty1_late}
{corr_cross_release1}"""
    corr_cons_advance = (
        "        flash_s_corr_cons_index ^= 1\n" if not fa4_stat_handoff else ""
    )
    corr_stat_dummy = (
        """        _helion_flash_rt.named_barrier_wait_unaligned(
            3 + warp_idx % 4, 64)
        cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)
        _helion_flash_rt.named_barrier_wait_unaligned(
            7 + warp_idx % 4, 64)"""
        + (
            "\n        cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"
            if stat_release_mapping is FlashStatReleaseMapping.SAME_SLOT
            else ""
        )
        + "\n"
        if acknowledged_stat_pipeline
        else ""
    )
    corr_stat_release_held = (
        "        cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)\n"
        if (
            acknowledged_stat_pipeline
            and stat_release_mapping is FlashStatReleaseMapping.CROSS_SLOT
        )
        else ""
    )
    corr_final_stat_release = (
        """        cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)
        cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)
"""
        if final_only_stat_pipeline
        else ""
    )
    # flash_pvt's direct-gmem partition already includes the CTA rank. Start it
    # from the rank-zero base tile so the follower does not apply its rank twice.
    corr_output_m_tile0 = output_m_tile0 if not epi_smem else "flash_m_tile0"
    corr_output_m_tile1 = output_m_tile1 if not epi_smem else "flash_m_tile1"
    corr_steady_stages = (
        f"{corr_stage1}\n{corr_stage0}"
        if (
            cfg.exp2_packet == _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET
            and not acknowledged_stat_pipeline
        )
        else f"{corr_stage0}\n{corr_stage1}"
    )
    if cfg.skip_rescale_stats:
        corr_inner = f"""        # Final: divide by row_sum, cast, store (waits MMA's last-tile O_full).
{_corr_epi("0", corr_output_m_tile0)}
{_corr_epi("1", corr_output_m_tile1)}
{corr_epi_empty_toggle}
{corr_cons_advance.rstrip()}
        flash_o_full_phase ^= 1"""
    else:
        corr_inner = f"""{corr_stat_dummy}        for flash_kv in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
{corr_steady_stages}
{textwrap.indent(corr_cons_advance.rstrip(), "    ")}
{corr_stat_release_held.rstrip()}
{corr_final_stat_release.rstrip()}
        # Final: divide by row_sum, cast, store (waits MMA's last-tile O_full).
{_corr_epi("0", corr_output_m_tile0)}
        {_corr_release_p_ready("0")}
{_corr_epi("1", corr_output_m_tile1)}
        {_corr_release_p_ready("1")}
{corr_epi_empty_toggle}
{corr_cons_advance.rstrip()}
        flash_o_full_phase ^= 1"""
    corr_prelude = (
        "none"
        if epi_smem and not is_causal and not has_lse and not score_plan.modifiers
        else "decode"
    )
    corr_block = _flash_fa4_wrap(
        f"{role_next} (warp_idx >= 8) & (warp_idx < 12):",
        corr_head,
        corr_inner,
        persistent,
        persistent_loop=cfg.persistent_loop,
        prelude=corr_prelude,
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
    output_epilogue: str = _FLASH_OUTPUT_EPILOGUE_IDENTITY,
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
    relu_output = _flash_output_relu_enabled(output_epilogue)
    relu_store = (
        "\n    _helion_flash_rt.relu_fragment_inplace(flash_reg)" if relu_output else ""
    )
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
    flash_reg.store(flash_reg.load() * flash_inv_sum){relu_store}
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
    output_epilogue: str
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

    required_names = (
        graph_plan.q_name,
        graph_plan.k_name,
        graph_plan.v_name,
        graph_plan.o_name,
        *graph_plan.bias_names,
        *graph_plan.alibi_names,
        *graph_plan.document_names,
    )
    if graph_plan.lse_name is not None:
        required_names = (*required_names, graph_plan.lse_name)
    host_tensors = _flash_graph_host_tensors(df.codegen.codegen_graphs)
    if any(name not in host_tensors for name in required_names):
        return None
    for name in dict.fromkeys(required_names):
        df.tensor_arg(host_tensors[name], prefer_name=name)

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
    if score_plan is not None and _flash_values_support_tensor_4d_tma(
        q_arg.fake_value,
        k_arg.fake_value,
        v_arg.fake_value,
        score_plan=score_plan,
        batch=graph_plan.batch,
        seq=graph_plan.seq,
        head_dim=graph_plan.head_dim,
        dtype=graph_plan.dtype,
    ):
        q_base_shape = _flash_4d_value_base_batch_heads(
            q_arg.fake_value,
            batch=graph_plan.batch,
            seq=graph_plan.seq,
            head_dim=graph_plan.head_dim,
            dtype=graph_plan.dtype,
        )
        assert q_base_shape is not None
        tensor_4d_batch, tensor_4d_heads = q_base_shape
    return FlashTensorPlan(
        q_arg,
        k_arg,
        v_arg,
        o_arg,
        graph_plan.output_epilogue,
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

    Called from ``generate_ast.visit_For`` after the flash detector has set
    ``attention_flash_block_ids``. Returns True when the flash kernel was emitted
    and the FX-derived scalar body can be skipped, False when the shape/layout is
    outside the validated envelope. The caller treats False as a late validation
    failure and raises ``BackendUnsupported`` rather than emitting the scalar
    body. Because the detector routes through the same
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
    output_epilogue = plan.output_epilogue
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
    if lse_arg is not None and batch * seq > 1 << 31:
        return False

    scale_log2 = score_plan.qk_scale_log2
    num_kv = (seq + 127) // 128
    small_biased_candidate = _small_biased_attention_score_plan_supported(
        score_plan,
        seq=seq,
        head_dim=head_dim,
        io_dtype=io_dtype,
        has_lse=lse_arg is not None,
        bias_arg_count=len(bias_args),
    )
    supports_tensor_4d_tma = _flash_values_support_tensor_4d_tma(
        plan.q_arg.fake_value,
        plan.k_arg.fake_value,
        plan.v_arg.fake_value,
        score_plan=score_plan,
        batch=batch,
        seq=seq,
        head_dim=head_dim,
        dtype=io_dtype,
    )
    flash_config: Mapping[str, object] | None = df.config
    if score_plan.requires_ws_overlap:
        flash_config = {**df.config, FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    cfg = resolve_flash_config(
        head_dim,
        num_kv,
        flash_config,
        dtype=io_dtype,
        num_bh=batch,
        is_causal=is_causal,
        has_kv_tile_pruning=score_plan.has_kv_tile_pruning,
        requires_ws_overlap=score_plan.requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=(
            not is_causal
            and lse_arg is None
            and not bias_args
            and not alibi_args
            and not document_args
            and _standard_dense_score_plan_supported(score_plan)
        ),
        standard_causal_output=(
            is_causal
            and lse_arg is None
            and not bias_args
            and not alibi_args
            and not document_args
            and _standard_causal_score_plan_supported(score_plan)
        ),
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        prefer_packed_reduce=bool(score_plan.modifiers),
    )
    if _flash_output_requires_tma(batch, seq, head_dim) and not cfg.epi_tma:
        return False
    clc_heads_per_batch = cfg.clc_heads_per_batch
    if (
        cfg.use_clc_scheduler
        and clc_heads_per_batch <= 0
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
        fa4_tile_rows = 128 * cfg.q_tile_count
        if cfg.use_2cta_instrs or cfg.use_cga2_local_cta:
            fa4_tile_rows *= 2
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
        "q_stage": cfg.q_tile_count,
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
                    output_epilogue=output_epilogue,
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
                    output_epilogue=output_epilogue,
                )
            )
    elif cfg.topology == "fa4":
        from ..compile_environment import CompileEnvironment

        df.cute_state.attention_flash_threads = 512
        df.body = list(
            emit_flash_fa4_device_body(
                df,
                head_dim=head_dim,
                num_kv=num_kv,
                sequence_extent=seq,
                num_bh=batch,
                total_tiles=total_tiles,
                cfg=cfg,
                has_lse=lse_arg is not None,
                io_dtype=io_dtype_str,
                score_plan=score_plan,
                tensor_4d_batch=plan.tensor_4d_batch if use_tensor_4d_tma else 0,
                tensor_4d_heads=plan.tensor_4d_heads if use_tensor_4d_tma else 0,
                target_device_capability=(
                    CompileEnvironment.current().config_spec.target_device_capability
                ),
                output_epilogue=output_epilogue,
            )
        )
    else:
        raise ValueError(f"unknown flash attention topology: {cfg.topology!r}")

    df.preamble = []
    return True
