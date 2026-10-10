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
import functools
from functools import partial
import itertools
import math
import operator
import os
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast
from typing import overload

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
from .cute_flash_alt import emit_flash_fa4_alt_device_body
from .cute_flash_alt import fa4_alt_supported
from .cute_flash_row_mma import ROW_MMA_DEFAULT_TILE_M
from .cute_flash_row_mma import ROW_MMA_DEFAULT_WARPS
from .cute_flash_row_mma import ROW_MMA_FAMILY
from .cute_flash_row_mma import ROW_MMA_PLAN_KIND
from .cute_flash_row_mma import ROW_MMA_TILE_M_CHOICES
from .cute_flash_row_mma import ROW_MMA_WARP_CHOICES
from .cute_flash_row_mma import emit_flash_row_mma_device_body
from .cute_flash_row_mma import emit_flash_row_mma_module_statements
from .cute_flash_row_mma import row_mma_aux_smem_bytes
from .cute_flash_row_mma import row_mma_search_grid
from .cute_flash_row_mma import row_mma_shape_supported
from .cute_flash_row_mma import row_mma_supported
from .flash_policy import get_flash_target_policy
from .flash_row_epilogue import FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM
from .flash_row_epilogue import FlashRowEpilogueProgram
from .flash_row_epilogue import emit_row_epilogue
from .flash_row_epilogue import match_flash_row_epilogue
from .flash_schedule import FlashScheduleSpec
from .flash_schedule import FlashStatReleaseMapping
from .flash_schedule import build_fa4_schedule
from .flash_schedule import max_fa4_kv_depth
from .flash_schedule import verify_flash_schedule
from .flash_tuning import FLASH_FLOAT16_MAX_LOG2
from .flash_tuning import FlashCausalSeedTemplate
from .flash_tuning import FlashPackedExp2Mode
from .flash_tuning import FlashSoftmaxLowering

_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections.abc import Callable
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
    row_epilogue: FlashRowEpilogueProgram | None = None


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


def _flash_lse_tensor_geometry(value: torch.Tensor) -> bool:
    """True for an fp32 ``[bh, seq]`` or ``[bh, seq, 1]`` contiguous lse buffer.

    The trailing unit axis is the same storage as the rank-2 form; the emitter
    addresses ``_flash_mLSE[row, bh]`` through a layout built from the plan's
    ``(seq, batch)`` constants, so both ranks share one store path.
    """
    if value.dtype != torch.float32 or not value.is_contiguous():
        return False
    if value.ndim == 2:
        return True
    return value.ndim == 3 and int(value.shape[2]) == 1


def _flash_unwrap_trailing_unit_axis(value: torch.fx.Node) -> torch.fx.Node:
    """Strip ``x[:, :, None]`` / ``x.unsqueeze(-1)`` off a stored rank-2 value."""
    from ...language import view_ops

    while value.op == "call_function":
        source = value.args[0] if value.args else None
        if not isinstance(source, torch.fx.Node):
            break
        if value.target is view_ops.subscript and len(value.args) == 2:
            indices = value.args[1]
            if (
                isinstance(indices, (list, tuple))
                and len(indices) == 3
                and _is_full_slice(indices[0])
                and _is_full_slice(indices[1])
                and indices[2] is None
            ):
                value = source
                continue
        elif (
            value.target is torch.ops.aten.unsqueeze.default
            and len(value.args) == 2
            and value.args[1] in (-1, 2)
        ):
            value = source
            continue
        break
    return value


def _flash_store_value_lse_log_base(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.store:
        return None
    indices = node.args[1] if len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)) or len(indices) not in (2, 3):
        return None
    if len(indices) == 3 and not _is_full_slice(indices[2]):
        return None
    if len(node.args) < 3 or not isinstance(node.args[2], torch.fx.Node):
        return None
    value = node.args[2]
    if len(indices) == 3:
        value = _flash_unwrap_trailing_unit_axis(value)
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
    if output_epilogue == FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM:
        return False
    raise AssertionError(f"unsupported flash output epilogue: {output_epilogue!r}")


@dataclasses.dataclass(frozen=True)
class FlashRowEpilogueEmit:
    """Codegen inputs of a fused row epilogue: the program plus kernel params."""

    program: FlashRowEpilogueProgram
    aux_params: tuple[str, ...]
    aux_dtypes: tuple[str, ...]
    scalar_names: tuple[str, ...]


def _flash_epi_aux_param(index: int) -> str:
    return f"_flash_mEpiAux{index}"


def _flash_aux_smem_staging(
    cfg: FlashAttentionConfig,
    row_epilogue: FlashRowEpilogueEmit | None,
    io_dtype: str,
    *,
    tensor_4d: bool,
) -> bool:
    """Whether the fused row epilogue reads its aux rows from the ``sO`` tile.

    On the FA4 softmax-warpgroup route with a staged output the epilogue warp
    TMA-loads the (single, output-dtype) aux tile into the free ``sO`` stage at
    the top of each work item, so the row program reads aux rows from shared
    memory after the KV loop instead of paying L2 latency per chunk on the
    critical path (and before the loop for the aux-only passes).
    """
    if cfg.topology == "fa4":
        staged = cfg.row_epilogue_warps == "softmax" and (cfg.epi_tma or cfg.epi_stg)
    else:
        # ws_overlap: warp 0 issues the aux TMA with the prologue loads and the
        # consumer warpgroup reads the rows from the single sO stage.
        staged = cfg.epi_stg and cfg.q_tile_m == 128
    return (
        row_epilogue is not None
        and staged
        and len(row_epilogue.aux_params) == 1
        and row_epilogue.aux_dtypes[0] == io_dtype
        and not tensor_4d
        and _flash_bool_env("HELION_CUTE_FLASH_AUX_SMEM", True)
    )


def _flash_epilogue_dtype_str(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "cutlass.Float32"
    return _flash_io_dtype_str(dtype)


def _flash_dtype_bytes(dtype_str: str) -> int:
    """Element size of a ``cutlass.*`` dtype name emitted by the flash codegen."""
    return 4 if dtype_str == "cutlass.Float32" else 2


def _flash_row_epilogue_gmem_views(
    spec: FlashRowEpilogueEmit, *, hd: int, indent: str, suffix: str, mma_index: str
) -> list[str]:
    """Flat-divide every aux tensor like O and select this CTA's MMA C tile.

    ``mma_index`` is the O tile expression with ``{tensor}`` in place of the
    flat-divided O view, so aux rows follow exactly the rows the O store owns.
    """
    lines: list[str] = []
    for index, param in enumerate(spec.aux_params):
        tile = f"_ep_gaux{index}{suffix}"
        view = f"cute.flat_divide({param}, cute.select((128, {hd}, 128), mode=[0, 1]))"
        lines.extend(
            (
                f"{indent}{tile} = {view}",
                f"{indent}{tile}_mma = {mma_index.format(tensor=tile)}",
            )
        )
    return lines


def _flash_row_epilogue_direct_source(
    spec: FlashRowEpilogueEmit,
    *,
    hd: int,
    io_dtype: str,
    indent: str,
    suffix: str,
    mma_index: str,
    epi_tiler: str,
    thr_ld: str,
    tiled_ld: str,
    tDtO: str,
    tDgO: str,
    inv_sum: str,
    ld_repetition: int,
    split: bool = False,
) -> tuple[str, str]:
    """Program epilogue for the per-thread t2r -> registers -> STG store routes.

    ``ld_repetition`` is the Ld32x32b repetition of ``tiled_ld``: each thread's
    chunk holds that many consecutive columns of its row.  Returns
    ``(prologue, epilogue)``; with ``split`` the aux views and the passes that
    do not read O go to the prologue (run before the main loop).
    """
    setup = _flash_row_epilogue_gmem_views(
        spec, hd=hd, indent=indent, suffix=suffix, mma_index=mma_index
    )
    prefix = f"_ep{suffix}"
    aux_allocs: list[str] = []
    aux_loads: list[list[str]] = []
    aux_elems: list[str] = []
    for index, dtype in enumerate(spec.aux_dtypes):
        tile = f"_ep_gaux{index}{suffix}"
        part = f"{prefix}_tdgaux{index}"
        setup.append(
            f"{indent}{part} = {thr_ld}.partition_D(cute.zipped_divide({tile}_mma, {epi_tiler}))"
        )
        aux_allocs.append(
            f"{{a}} = cute.make_rmem_tensor({part}[None, 0, 0].shape, {dtype})"
        )
        aux_loads.append([f"cute.autovec_copy({part}[None, {{i}}, 0], {{a}})"])
        aux_elems.append("cutlass.Float32({a}[{j}])")
    prologue, epilogue = emit_row_epilogue(
        spec.program,
        chunks=hd // ld_repetition,
        elem_var=f"{prefix}_j",
        chunk_width=ld_repetition,
        o_alloc=f"{{o}} = cute.make_rmem_tensor({tDgO}[None, 0, 0].shape, cutlass.Float32)",
        o_load=[
            f"cute.copy({tiled_ld}, {tDtO}[None, {{i}}, 0], {{o}})",
            f"_helion_flash_rt._scale_fragment_packed_f32x2({{o}}, {inv_sum})",
        ],
        o_elem="{o}[{j}]",
        aux_allocs=aux_allocs,
        aux_loads=aux_loads,
        aux_elems=aux_elems,
        out_alloc=f"{{out}} = cute.make_rmem_tensor({tDgO}[None, 0, 0].shape, {io_dtype})",
        store_elem=f"{{out}}[{{j}}] = {io_dtype}({{value}})",
        store=[f"cute.autovec_copy({{out}}, {tDgO}[None, {{i}}, 0])"],
        scalar_names=spec.scalar_names,
        indent=indent,
        prefix=prefix,
        split=split,
    )
    return _flash_row_epilogue_join(setup, prologue, epilogue, split=split)


def _flash_row_epilogue_join(
    setup: list[str], prologue: str, epilogue: str, *, split: bool
) -> tuple[str, str]:
    setup_src = "\n".join(setup)
    if split:
        return "\n".join(part for part in (setup_src, prologue) if part), epilogue
    return "", "\n".join(part for part in (setup_src, prologue, epilogue) if part)


def _flash_row_epilogue_smem_source(
    spec: FlashRowEpilogueEmit,
    *,
    hd: int,
    io_dtype: str,
    indent: str,
    suffix: str,
    mma_index: str,
    corr_tile_size: int,
    use_2cta_instrs: bool,
    tOtO: str,
    sO: str,
    tidx: str,
    inv_sum: str,
    split: bool = False,
) -> tuple[str, str]:
    """Program epilogue for the FA4 correction -> staged-smem store routes."""
    setup, prologue, epilogue = _flash_row_epilogue_smem_source_parts(
        spec,
        hd=hd,
        io_dtype=io_dtype,
        indent=indent,
        suffix=suffix,
        mma_index=mma_index,
        corr_tile_size=corr_tile_size,
        use_2cta_instrs=use_2cta_instrs,
        tOtO=tOtO,
        sO=sO,
        tidx=tidx,
        inv_sum=inv_sum,
        split=split,
    )
    return _flash_row_epilogue_join(setup, prologue, epilogue, split=split)


def _flash_row_epilogue_smem_source_parts(
    spec: FlashRowEpilogueEmit,
    *,
    hd: int,
    io_dtype: str,
    indent: str,
    suffix: str,
    mma_index: str,
    corr_tile_size: int,
    use_2cta_instrs: bool,
    tOtO: str,
    sO: str,
    tidx: str,
    inv_sum: str,
    split: bool = False,
    aux_prefetch: bool = False,
    aux_from_smem: bool = False,
) -> tuple[list[str], str, str]:
    """``(setup lines, prologue, epilogue)`` of the staged-smem row epilogue.

    ``setup`` holds only copy-view algebra (no memory traffic) and, with
    ``aux_prefetch``, an L2 prefetch of every aux row chunk this thread will
    read, so a caller can run it long before the aux loads (e.g. ahead of the
    KV loop) and keep the loads themselves next to their consumers.
    """
    setup = (
        []
        if aux_from_smem
        else _flash_row_epilogue_gmem_views(
            spec, hd=hd, indent=indent, suffix=suffix, mma_index=mma_index
        )
    )
    prefix = f"_ep{suffix}"
    setup.append(
        f"{indent}{prefix}_t2r, {prefix}_r2s, {prefix}_tOtO, {prefix}_tOsO, {prefix}_tOcO = "
        f"_helion_flash_rt.fa4_correction_epilogue_partitions(flash_pvt, {tOtO}, {sO}, "
        f"{tidx}, {hd}, {corr_tile_size}, {io_dtype}, {use_2cta_instrs!s})"
    )
    aux_allocs: list[str] = []
    aux_loads: list[list[str]] = []
    aux_elems: list[str] = []
    for index, dtype in enumerate(spec.aux_dtypes):
        if aux_from_smem:
            # The aux tile sits in this thread's own sO chunks (the TMA load
            # used the output layout); every chunk is read before the store
            # pass overwrites it with the output chunk of the same thread.
            aux_allocs.append(
                f"{{a}} = cute.make_rmem_tensor({prefix}_tOcO[None, 0, 0, 0].shape, {dtype})"
            )
            aux_loads.append(
                [f"cute.autovec_copy({prefix}_tOsO[None, 0, 0, {{i}}], {{a}})"]
            )
            aux_elems.append("cutlass.Float32({a}[{j}])")
            continue
        part = f"{prefix}_tgaux{index}"
        setup.append(
            f"{indent}{part} = _helion_flash_rt.fa4_correction_epilogue_gmem_partition("
            f"{prefix}_t2r, {tidx}, _ep_gaux{index}{suffix}_mma, {corr_tile_size})"
        )
        if aux_prefetch:
            setup.extend(
                f"{indent}_helion_flash_rt.prefetch_global_l2({part}[None, 0, 0, {chunk}].iterator)"
                for chunk in range(hd // corr_tile_size)
            )
        aux_allocs.append(
            f"{{a}} = cute.make_rmem_tensor({part}[None, 0, 0, 0].shape, {dtype})"
        )
        aux_loads.append([f"cute.autovec_copy({part}[None, 0, 0, {{i}}], {{a}})"])
        aux_elems.append("cutlass.Float32({a}[{j}])")
    prologue, epilogue = emit_row_epilogue(
        spec.program,
        chunks=hd // corr_tile_size,
        elem_var=f"{prefix}_j",
        chunk_width=corr_tile_size,
        o_alloc=f"{{o}} = cute.make_rmem_tensor({prefix}_tOcO[None, 0, 0, 0].shape, cutlass.Float32)",
        o_load=[
            f"cute.copy({prefix}_t2r, {prefix}_tOtO[None, 0, 0, {{i}}], {{o}})",
            f"_helion_flash_rt._scale_fragment_packed_f32x2({{o}}, {inv_sum})",
        ],
        o_elem="{o}[{j}]",
        aux_allocs=aux_allocs,
        aux_loads=aux_loads,
        aux_elems=aux_elems,
        out_alloc=f"{{out}} = cute.make_rmem_tensor({prefix}_tOcO[None, 0, 0, 0].shape, cutlass.Float32)",
        store_elem="{out}[{j}] = {value}",
        store=[
            f"_helion_flash_rt.cvt_copy({prefix}_r2s, {{out}}, {prefix}_tOsO[None, 0, 0, {{i}}])"
        ],
        scalar_names=spec.scalar_names,
        indent=indent,
        prefix=prefix,
        split=split,
        aux_prefetch=1 if aux_from_smem else None,
    )
    return setup, prologue, epilogue


def _flash_store_value_output_epilogue(node: torch.fx.Node) -> str | None:
    from ...language import memory_ops

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
    if not _flash_normalized_accumulator_node(value):
        return None
    return output_epilogue


def _flash_normalized_accumulator_node(value: torch.fx.Node) -> bool:
    """True for ``acc_phi / l_i_phi[:, :, None]``: the flash-normalized output."""
    from ...language import view_ops

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
    return bool(
        _is_full_slice(indices[0]) and _is_full_slice(indices[1]) and indices[2] is None
    )


def _flash_row_epilogue_load_name(
    node: torch.fx.Node,
    *,
    index_pair: tuple[torch.fx.Node, torch.fx.Node],
    host_tensors: Mapping[str, torch.Tensor],
    shape: tuple[int, int, int],
) -> str | None:
    """Host tensor name of a ``[tile_b, tile_m, :]`` load with the output geometry."""
    from ...language import memory_ops

    if node.op != "call_function" or node.target is not memory_ops.load:
        return None
    if len(node.args) < 2:
        return None
    if len(node.args) > 2 and node.args[2] is not None:
        return None
    if len(node.args) > 3 and node.args[3] is not None:
        return None
    name = _flash_load_host_tensor_name(node)
    if name is None:
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        return None
    if indices[0] is not index_pair[0] or indices[1] is not index_pair[1]:
        return None
    if not _is_full_slice(indices[2]):
        return None
    value = host_tensors.get(name)
    if (
        value is None
        or value.ndim != 3
        or not value.is_contiguous()
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or tuple(int(size) for size in value.shape) != shape
    ):
        return None
    return name


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
    row_programs: list[FlashRowEpilogueProgram] = []
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
                output_shape = tuple(int(size) for size in value.shape)
                program = match_flash_row_epilogue(
                    node,
                    head_dim=output_shape[2],
                    is_o_norm=_flash_normalized_accumulator_node,
                    is_row_load=functools.partial(
                        _flash_row_epilogue_load_name,
                        index_pair=index_pair,
                        host_tensors=host_tensors,
                        shape=output_shape,
                    ),
                )
                if program is None:
                    return None
                row_programs.append(program)
                output_epilogue = FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM
            o_store_names.append(name)
            o_store_epilogues.append(output_epilogue)
            continue
        if _flash_lse_tensor_geometry(value):
            if not _flash_store_has_canonical_indices(node, value.ndim, index_pair):
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

    row_epilogue = row_programs[0] if row_programs else None
    aux_names = row_epilogue.aux_names if row_epilogue is not None else ()
    core_names = {q_name, k_name, v_name, o_name, *bias_names}
    if not core_names <= set(flash_3d_tensors):
        return None
    if set(flash_3d_tensors) - core_names - set(aux_names):
        return None
    for name in aux_names:
        aux_value = host_tensors.get(name)
        if (
            aux_value is None
            or aux_value.ndim != 3
            or not aux_value.is_contiguous()
            or int(aux_value.shape[0]) != batch
            or int(aux_value.shape[1]) != seq
            or int(aux_value.shape[2]) != head_dim
        ):
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
        if _flash_lse_tensor_geometry(value)
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
        *aux_names,
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
        row_epilogue,
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


def flash_attention_graph_row_epilogue_from_graphs(
    graphs: Iterable[GraphInfo],
    *,
    root_block_ids: Sequence[int] | None = None,
    kv_block_id: int | None = None,
    score_plan: AttentionScorePlan,
) -> bool:
    """Return whether the output store is a fused row-epilogue program."""
    graph_plan = _flash_graph_output_plan_from_graphs(
        graphs,
        root_block_ids=root_block_ids,
        kv_block_id=kv_block_id,
        score_plan=score_plan,
    )
    return (
        graph_plan is not None
        and graph_plan.output_epilogue == FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM
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


_FLASH_ENV_PREFIX = "HELION_CUTE_FLASH_"


class _FlashEnvScope(NamedTuple):
    values: dict[str, str]
    fingerprint: tuple[tuple[str, str], ...]


_flash_env_scope_state: _FlashEnvScope | None = None


def _flash_env_fingerprint_read() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith(_FLASH_ENV_PREFIX)
        )
    )


def flash_env_fingerprint() -> tuple[tuple[str, str], ...]:
    """Canonical value of every ``HELION_CUTE_FLASH_*`` env var.

    ``resolve_flash_config`` (and everything built on it, notably
    ``flash_autotune_fragments``) reads only env vars with this prefix, so this
    tuple is a complete environment cache key for their results.
    """
    scope = _flash_env_scope_state
    if scope is not None:
        return scope.fingerprint
    return _flash_env_fingerprint_read()


class _FlashEnvSnapshotScope:
    """Reentrant scope that freezes the flash env for one design pass.

    The autotuner's structural-coverage design resolves millions of configs per
    pass; each resolution reads ~60 env vars. Within a scope every read comes
    from one snapshot taken at entry, so env patches made *before* an operation
    (mock.patch.dict / monkeypatch in tests) still take effect while the
    per-resolve ``os.environ`` traffic disappears. Never process-lifetime: the
    snapshot dies with the outermost scope.
    """

    def __init__(self) -> None:
        self._owner = False

    def __enter__(self) -> None:
        global _flash_env_scope_state
        if _flash_env_scope_state is None:
            fingerprint = _flash_env_fingerprint_read()
            _flash_env_scope_state = _FlashEnvScope(dict(fingerprint), fingerprint)
            self._owner = True

    def __exit__(self, *args: object) -> None:
        global _flash_env_scope_state
        if self._owner:
            _flash_env_scope_state = None
            self._owner = False


def flash_env_snapshot_scope() -> _FlashEnvSnapshotScope:
    return _FlashEnvSnapshotScope()


@overload
def _flash_env_get(name: str) -> str | None: ...


@overload
def _flash_env_get(name: str, default: str | _T) -> str | _T: ...


def _flash_env_get(name: str, default: object = None) -> object:
    scope = _flash_env_scope_state
    if scope is None:
        return os.environ.get(name, default)  # pyright: ignore[reportArgumentType]
    value = scope.values.get(name)
    return default if value is None else value


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
    override = _flash_env_get("HELION_CUTE_FLASH_KV_STAGE")
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


def _flash_aliased_kv_stage_cap(
    head_dim: int,
    *,
    stage_output: bool,
    kv_tile_n: int = 128,
    use_2cta_instrs: bool = False,
) -> int:
    """Largest legal aliased K/V depth for the requested output storage.

    A wider KV tile makes each staging slot proportionally larger, so the depth
    that fits in shared memory drops with it.
    """
    return max_fa4_kv_depth(
        FlashScheduleSpec(
            head_dim=head_dim,
            kv_depth=2,
            stage_output=stage_output,
            kv_tile_n=kv_tile_n,
            cta_count=2 if use_2cta_instrs else 1,
            cooperative_mma=use_2cta_instrs,
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
    override = _flash_env_get("HELION_CUTE_FLASH_PERSISTENT")
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
    override = _flash_env_get("HELION_CUTE_FLASH_S_STAGE")
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
    # One query tile per work item, two softmax warpgroups on alternating KV
    # steps, separate QK/PV issue warps (see ``cute_flash_alt``).
    alternating_warpgroups: bool = False


FLASH_PIPELINE_FAMILY_FLAGS: dict[str, FlashPipelineFamilyFlags] = {
    "ws_overlap": FlashPipelineFamilyFlags("ws_overlap"),
    "fa4": FlashPipelineFamilyFlags("fa4"),
    # Register-MMA row programs (mma.sync, cp.async, no TMA/TMEM): 8- or
    # 16-row query tiles whose warps split the keys; the latency-bound
    # structure for grids too small to fill the device.
    ROW_MMA_FAMILY: FlashPipelineFamilyFlags(ROW_MMA_FAMILY),
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
    # New families go last: autotune seeds and coverage rows index the family
    # list by position.
    "fa4_alt": FlashPipelineFamilyFlags("fa4", alternating_warpgroups=True),
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
    alternating_warpgroups: bool = False,
) -> str:
    if topology == ROW_MMA_FAMILY:
        return ROW_MMA_FAMILY
    if topology != "fa4":
        return "ws_overlap"
    if alternating_warpgroups:
        return "fa4_alt"
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
    rowmax: str = "software"
    # ``auto`` preserves existing target-policy lowerings. Explicit choices
    # let the autotuner compare whole-row implementations on any legal shape.
    softmax_lowering: str = "auto"
    row_sum_schedule: str = "post_acquire"
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
    # Per-chunk staged-P release on top of ``split_p_arrive`` (dense fa4 bodies
    # with the chunked softmax, Rep16 P stores and the PTX MMA stream): softmax
    # arrives once per 32-column P chunk (pfor, pforc0, pforc1, pfor2) and the PV
    # issue stream waits before each K-chunk quarter, so the first three quarters
    # of PV run under the exp pass and QK(i+1) follows the last chunk by one
    # quarter of PV instead of the 3/4-split's late start. Same stores, same MMA
    # order: bitwise identical to the split release.
    p_chunk_arrive: bool = False
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
    # Warps that evaluate a fused row epilogue on the FA4 topology: the
    # correction warpgroup (FA4's layout: 64 registers, one warp per SMSP,
    # both Q tiles in sequence, and every rescale of the next tile waits behind
    # it) or the softmax warpgroup that owns the rows (200 registers, both Q
    # tiles in parallel, overlapping the next tile's first QK). ws_overlap has a
    # single consumer warpgroup and ignores the choice.
    row_epilogue_warps: str = "correction"
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
    # ``fa4_alt``: one 128-row query tile per persistent work item, softmax
    # warpgroups A/B on alternating KV steps with their own score/probability
    # buffers, a QK-issue warp and a PV-issue warp, the output staged through
    # the finished item's dead Q stage. Dense hd128 plain rows only.
    alternating_warpgroups: bool = False
    # Causal descending KV can run the short masked diagonal prefix separately
    # from the hot unmasked suffix, removing a per-KV branch from most tiles.
    causal_loop_split: bool = False
    # KV tile width for the FA4 score/probability tile. 128 is the historical
    # fixed value. A wider tile amortizes the per-KV-tile softmax correction --
    # which sits on the critical path -- over more columns; FA4's tuned sm_103
    # plans all use 160 and gain ~9.6% from it. Legality: a multiple of 32 that
    # keeps two S buffers plus two O buffers inside 512 TMEM columns, and (until
    # the masked tail tile lands) one that divides the sequence.
    kv_tile_n: int = 128
    # Query-tile height of the two-warpgroup ws_overlap body. 64 uses the
    # tcgen05 M=64 shape: twice the CTAs for a given problem, half the softmax
    # and epilogue work per CTA (two rows per consumer thread, quad-shared).
    # Legal only for the flat staged dense ws_overlap body.
    q_tile_m: int = 128
    # One-pass softmax for a 64-row tile whose sequence is exactly two KV
    # tiles: both score tiles are read as one 256-wide row (one max, one exp2
    # sweep handing off one P tile at a time, both PV MMAs accumulate with no
    # rescale between them), instead of two online-softmax rounds with their
    # handshakes.
    ws_one_pass: bool = False
    # ``row_mma`` family: warps per CTA (each owns an equal share of the keys)
    # and query rows per CTA (8 or 16: one or two octets sharing the warps'
    # K/V shared-memory slices).  Pinned to the defaults for every other
    # family so configs that differ only here stay one autotune identity.
    row_warps: int = ROW_MMA_DEFAULT_WARPS
    row_tile_m: int = ROW_MMA_DEFAULT_TILE_M


# TMEM is 512 columns. The FA4 score pipeline holds ``s_stage`` score buffers of
# ``kv_tile_n`` columns plus two ``head_dim``-wide output accumulators.
FLASH_TMEM_COLUMNS = 512
# Validated KV tile widths. 192 exactly saturates TMEM at head-dim 64 with two
# score buffers (2*192 + 2*64 == 512) and produces NaN, so the budget check
# below demands headroom rather than a fit; it is also slower than 160 on every
# part measured, so nothing is lost by leaving it out of the choice set.
FLASH_KV_TILE_N_CHOICES = (128, 160)


def _flash_kv_tile_n_fits_tmem(kv_tile_n: int, head_dim: int, s_stage: int) -> bool:
    """Return whether the score and output accumulators leave TMEM headroom."""
    return s_stage * kv_tile_n + 2 * head_dim < FLASH_TMEM_COLUMNS


def _flash_kv_tile_n_supported(
    kv_tile_n: int,
    *,
    head_dim: int,
    num_kv: int,
    topology: str,
    is_causal: bool,
    s_stage: int,
    desc_kv: bool = True,
    softmax_disc: bool = False,
) -> bool:
    """Return whether a non-default KV tile width is legal for this shape.

    ``kv_tile_n`` must be a tcgen05-legal MMA N and must leave the score and
    output accumulators inside TMEM. A width that does not divide the sequence
    is allowed: the trailing partial tile is masked, which needs the descending
    KV order that puts it first. Causal keeps the default width because its
    diagonal masking is written against 128-column tiles.
    """
    if kv_tile_n == 128:
        return True
    if kv_tile_n not in FLASH_KV_TILE_N_CHOICES or kv_tile_n % 32:
        return False
    if is_causal or topology != "fa4":
        return False
    if not _flash_kv_tile_n_fits_tmem(kv_tile_n, head_dim, s_stage):
        return False
    sequence_extent = num_kv * 128
    if sequence_extent % kv_tile_n == 0:
        return True
    # The trailing partial tile has to be masked, and only the whole-row softmax
    # body does that; the chunked ("disc") body would consume the out-of-range
    # columns as real zero scores. It also needs the descending KV order that
    # visits the partial tile first.
    return desc_kv and not softmax_disc


def _flash_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var; treat '1'/'true'/'on' (case-insensitive) as True."""
    val = _flash_env_get(name)
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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    supports_tensor_4d_tma: bool = True,
    prefer_packed_reduce: bool = False,
    plain_row_body: bool = True,
    has_row_epilogue: bool = False,
    has_score_modifiers: bool = False,
    row_mma_aligned: bool = True,
    row_mma_aux_dtypes: Sequence[str] = (),
    target_device_capability: tuple[int, int] | None = None,
) -> FlashAttentionConfig:
    """Resolve the flash-attention topology config from shape, env vars and config.

    When *config* is supplied and the autotuner has enabled the flash search
    surface (``cute_flash_search_enabled``), the autotunable knobs (see
    ``FLASH_CONFIG_KEYS`` / ``flash_autotune_fragments``) are read from the
    config Mapping; any knob the config does not carry falls back to the env-var
    resolution below. New lowering choices have explicit conservative defaults;
    normalized configs record the selected implementation.

    *plain_row_body* is False when the score plan has modifiers or a fused row
    epilogue; those rows exist only in the 128-row body, so the 64-row query
    tile is canonicalized away instead of aliasing it. *has_row_epilogue*
    keeps fused row epilogues on the chunked FA4 softmax body.
    *has_score_modifiers* is True when the score plan carries a modifier
    (mask, bias, softcap, ...); ``plain_row_body`` alone cannot tell that
    apart from a fused row epilogue, which the ``row_mma`` row programs do
    run. *row_mma_aligned* is False when a q/k/v/o/lse (or aux) base is not
    proven 16-byte aligned; the ``row_mma`` row programs move 16-byte packets
    from those bases, so such a request resolves to the tcgen05 default
    instead. *row_mma_aux_dtypes* names the fused row epilogue's aux rows
    (known at codegen only); a request whose aux staging does not fit the
    row programs' shared-memory budget resolves to the tcgen05 default too.
    """

    if tmem_rowmax_compatible is None:
        tmem_rowmax_compatible = standard_dense_output or standard_causal_output

    softmax_lowering = (
        config.get(FLASH_SOFTMAX_LOWERING_KEY, "auto") if config is not None else "auto"
    )
    if softmax_lowering not in (
        "auto",
        "standard",
        "resident_value_graph",
        "resident_stateful",
    ):
        raise ValueError(f"invalid flash softmax lowering: {softmax_lowering!r}")
    row_sum_schedule = (
        config.get(FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire")
        if config is not None
        else "post_acquire"
    )
    if row_sum_schedule not in ("post_acquire", "pre_acquire"):
        raise ValueError(f"invalid flash row-sum schedule: {row_sum_schedule!r}")
    if softmax_lowering != "resident_stateful":
        row_sum_schedule = "post_acquire"
    resident_overrides = _flash_resident_softmax_overrides(
        softmax_lowering, is_causal=is_causal
    )
    packet_config = config
    if resident_overrides:
        packet_config = {**({} if config is None else config), **resident_overrides}
    env_exp2_packet = _flash_env_get("HELION_CUTE_FLASH_EXP2_PACKET")
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
        if key in resident_overrides:
            return resident_overrides[key]
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
    pipeline_family_env = _flash_env_get("HELION_CUTE_FLASH_PIPELINE_FAMILY")
    requested_family_flags = (
        _flash_pipeline_family_flags(pipeline_family_cfg)
        if pipeline_family_cfg is not None
        else None
        if legacy_structural_config
        else _flash_pipeline_family_flags(pipeline_family_env)
    )
    alt_requested = bool(
        requested_family_flags is not None
        and requested_family_flags.alternating_warpgroups
    )
    if requested_family_flags is not None:
        topology = requested_family_flags.topology
    else:
        topology = _flash_env_get("HELION_CUTE_FLASH_TOPOLOGY", topology_default)
        # Legacy fixed configs and environment overrides remain accepted. The
        # compound family above takes precedence when explicitly selected.
        topology_cfg = _cfg(FLASH_TOPOLOGY_KEY)
        if topology_cfg is not None:
            topology = str(topology_cfg)
    if topology not in ("ws_overlap", "fa4", ROW_MMA_FAMILY):
        topology = "ws_overlap"
    if (
        topology == ROW_MMA_FAMILY or alt_requested or has_row_epilogue
    ) and softmax_lowering in (
        "resident_value_graph",
        "resident_stateful",
    ):
        raise InvalidConfig(
            "resident softmax lowerings require ordinary FA4 without a row epilogue"
        )
    if topology == ROW_MMA_FAMILY:
        if row_mma_aligned and row_mma_supported(
            head_dim=head_dim,
            dtype=dtype,
            is_causal=is_causal,
            has_kv_tile_pruning=has_kv_tile_pruning,
            requires_ws_overlap=requires_ws_overlap,
            small_biased_candidate=small_biased_candidate,
            plain_row_body=plain_row_body,
            has_row_epilogue=has_row_epilogue,
            has_score_modifiers=has_score_modifiers,
        ):
            row_mma_config = _resolve_row_mma_config(
                head_dim,
                num_kv,
                _cfg,
                dtype=dtype,
                num_bh=num_bh,
                is_causal=is_causal,
                has_kv_tile_pruning=has_kv_tile_pruning,
                requires_ws_overlap=requires_ws_overlap,
                small_biased_candidate=small_biased_candidate,
                standard_dense_output=standard_dense_output,
                standard_causal_output=standard_causal_output,
                supports_tensor_4d_tma=supports_tensor_4d_tma,
                prefer_packed_reduce=prefer_packed_reduce,
                plain_row_body=plain_row_body,
                has_row_epilogue=has_row_epilogue,
                has_score_modifiers=has_score_modifiers,
                aux_dtypes=row_mma_aux_dtypes,
            )
            if row_mma_config is not None:
                return row_mma_config
        topology = topology_default
    if topology == "fa4" and num_kv % 2 != 0:
        topology = "ws_overlap"
    if softmax_lowering == "resident_value_graph" and not (
        _flash_dense_resident_workload_supported(
            fa4=topology == "fa4" and not has_row_epilogue,
            dtype=dtype,
            standard_dense_output=standard_dense_output,
        )
        and not is_causal
    ):
        raise InvalidConfig(
            "cute_flash_softmax_lowering='resident_value_graph' requires "
            "FP16/BF16 dense output on an FA4 pipeline"
        )
    explicit_stateful = softmax_lowering == "resident_stateful"
    if explicit_stateful and not _flash_causal_stateful_workload_supported(
        fa4=topology == "fa4" and not has_row_epilogue,
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        is_causal=is_causal,
        standard_causal_output=standard_causal_output,
        causal_resident_compatible=causal_resident_compatible,
    ):
        raise InvalidConfig(
            "cute_flash_softmax_lowering='resident_stateful' requires "
            "D64/D128 FP16/BF16 causal output with proven complete paired coverage"
        )
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
            and _flash_env_get("HELION_CUTE_FLASH_KV_STAGE") is None
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
        _flash_env_get(
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
        _flash_env_get("HELION_CUTE_FLASH_NUM_SOFTMAX_WG", "1")
    )
    num_correction_warps = int(
        _flash_env_get("HELION_CUTE_FLASH_NUM_CORRECTION_WARPS", "0")
    )
    num_mma_warps = int(_flash_env_get("HELION_CUTE_FLASH_NUM_MMA_WARPS", "0"))
    num_load_warps = int(_flash_env_get("HELION_CUTE_FLASH_NUM_LOAD_WARPS", "0"))
    num_epilogue_warps = int(_flash_env_get("HELION_CUTE_FLASH_NUM_EPI_WARPS", "0"))
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
        _flash_env_get("HELION_CUTE_FLASH_WAIT_HINT", str(wait_hint_default))
    )
    wait_hint_cfg = _cfg(FLASH_WAIT_HINT_KEY)
    if wait_hint_cfg is not None:
        wait_hint = int(wait_hint_cfg)  # type: ignore[arg-type]
    if wait_hint not in (0, wait_hint_default) or topology != "fa4":
        wait_hint = wait_hint_default
    acc_stage = int(_flash_env_get("HELION_CUTE_FLASH_ACC_STAGE", "1"))
    epi_stage = int(_flash_env_get("HELION_CUTE_FLASH_EPI_STAGE", "1"))
    # Exp2 pipe-split schedule. The autotuner sees this as one paired schedule
    # knob so it never searches meaningless combinations like exp2_impl="xu"
    # with independent e2e cadence values, or regressive res/freq pairings.
    e2e_schedule_default = _flash_e2e_schedule_default(topology, head_dim)
    e2e_schedule = _flash_env_get(
        "HELION_CUTE_FLASH_E2E_SCHEDULE", e2e_schedule_default
    )
    exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
        e2e_schedule, e2e_schedule_default
    )
    # Legacy env overrides remain accepted for scripts. Config overrides are
    # applied afterward so autotuned/fixed config values still win over env.
    exp2_impl = _flash_env_get("HELION_CUTE_FLASH_EXP2_IMPL", exp2_impl)
    e2e_freq = int(_flash_env_get("HELION_CUTE_FLASH_E2E_FREQ", str(e2e_freq)))
    e2e_res = int(_flash_env_get("HELION_CUTE_FLASH_E2E_RES", str(e2e_res)))
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
    masked_e2e_schedule = _flash_env_get(
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
        _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET", str(e2e_offset_default))
    )
    e2e_offset_cfg = _cfg(FLASH_E2E_OFFSET_KEY)
    if e2e_offset_cfg is not None:
        e2e_offset = int(e2e_offset_cfg)  # type: ignore[arg-type]
    e2e_offset = _flash_normalize_e2e_offset(
        e2e_offset, e2e_offset_default, e2e_offset_period
    )
    e2e_offset0_default = 0
    e2e_offset0 = int(
        _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET0", str(e2e_offset0_default))
    )
    e2e_offset0_cfg = _cfg(FLASH_E2E_OFFSET0_KEY)
    if e2e_offset0_cfg is not None:
        e2e_offset0 = int(e2e_offset0_cfg)  # type: ignore[arg-type]
    e2e_offset0 = _flash_normalize_e2e_offset(
        e2e_offset0, e2e_offset0_default, e2e_offset_period
    )
    tmem_plan = _flash_env_get("HELION_CUTE_FLASH_TMEM_PLAN", "separate")
    tmem_s_to_p_offset = int(
        _flash_env_get("HELION_CUTE_FLASH_TMEM_S_TO_P_OFFSET", "0")
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
    elif is_causal and not explicit_stateful:
        # Generic causal correction has no acknowledged single-slot statistics
        # handoff. The explicit stateful graph supplies its own SAME_SLOT
        # acknowledgments; the ordinary graph retains its chunked pipeline.
        softmax_disc = True
    elif has_row_epilogue:
        # A fused row epilogue holds a warpgroup at the end of every work item
        # (the program walks O out of TMEM after the last PV).  With the
        # whole-row body that stall lets the two-slot statistics handoff lap
        # its consumer: XSA at 2x16x2048x128 deadlocks about once per thousand
        # launches on both epilogue routes (also on the tree before this
        # change).  The chunked pipeline ran 35k launches without one.
        softmax_disc = True
    # fa4 Step 3 (L1): disc PASS2 software-pipeline depth. Keep one neutral
    # default; the autotuner measures the legal depths for each workload.
    if topology == "fa4" and softmax_disc:
        disc_pipe_default = 2
    else:
        disc_pipe_default = 1
    disc_pipe_depth = int(
        _flash_env_get("HELION_CUTE_FLASH_DISC_PIPE", str(disc_pipe_default))
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
        _flash_env_get("HELION_CUTE_FLASH_P_STORE_REP", str(p_store_repetition_default))
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
        _flash_env_get("HELION_CUTE_FLASH_S_LOAD_REP", str(s_load_repetition_default))
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
    p_chunk_arrive = _flash_bool_env("HELION_CUTE_FLASH_P_CHUNK_ARRIVE", False)
    p_chunk_arrive_cfg = _cfg(FLASH_P_CHUNK_ARRIVE_KEY)
    if p_chunk_arrive_cfg is not None:
        p_chunk_arrive = bool(p_chunk_arrive_cfg)
    if (
        topology != "fa4"
        or is_causal
        or not mma_ptx
        or not softmax_disc
        or not split_p_arrive
        or p_store_repetition != 16
        or s_load_repetition != 32
    ):
        # The per-chunk release is written for the dense chunked pass with Rep16
        # P stores and the PTX PV stream (the waits live inside that stream).
        p_chunk_arrive = False
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
        _flash_env_get(
            "HELION_CUTE_FLASH_FIRST_LOAD_ORDER", str(first_load_order_default)
        )
    )
    first_load_order_cfg = _cfg(FLASH_FIRST_LOAD_ORDER_KEY)
    if first_load_order_cfg is not None:
        first_load_order = int(first_load_order_cfg)  # type: ignore[arg-type]
    if topology != "fa4" or first_load_order not in FLASH_FIRST_LOAD_ORDER_CHOICES:
        first_load_order = 0
    kv_order_default = "ascending"
    kv_order = _flash_env_get("HELION_CUTE_FLASH_KV_ORDER", kv_order_default)
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
    # The ws_overlap warpgroup path defaults to the staged (smem + coalesced
    # vector store) O epilogue: its direct per-thread STG.128 store writes one
    # 128-byte row per lane, so every warp-level store touches 32 distinct
    # lines. The single-warpgroup (s_stage == 1) body keeps the direct store.
    epi_stg_default = topology == "ws_overlap" and s_stage == 2
    epi_stg = _flash_bool_env("HELION_CUTE_FLASH_EPI_STG", epi_stg_default)
    epi_stg_cfg = _cfg(FLASH_EPI_STG_KEY)
    if epi_stg_cfg is not None:
        epi_stg = bool(epi_stg_cfg)
    epi_stg = epi_stg and not epi_tma and (topology == "fa4" or s_stage == 2)
    # ws_overlap drains sO with the 128-thread consumer warpgroup; loading the
    # whole per-thread slice before storing hides the smem load latency that
    # the per-row-slice loop exposes eight times.
    epi_stg_store_default = "whole" if topology == "ws_overlap" else "slice"
    epi_stg_store = _flash_env_get(
        "HELION_CUTE_FLASH_EPI_STG_STORE", epi_stg_store_default
    )
    epi_stg_store_cfg = _cfg(FLASH_EPI_STG_STORE_KEY)
    if epi_stg_store_cfg is not None:
        epi_stg_store = str(epi_stg_store_cfg)
    if epi_stg_store not in ("slice", "whole") or not epi_stg:
        epi_stg_store = "slice"
    epi_stg_gmem = _flash_env_get("HELION_CUTE_FLASH_EPI_STG_GMEM", "stage")
    epi_stg_gmem_cfg = _cfg(FLASH_EPI_STG_GMEM_KEY)
    if epi_stg_gmem_cfg is not None:
        epi_stg_gmem = str(epi_stg_gmem_cfg)
    if epi_stg_gmem not in ("stage", "pair") or not epi_stg or topology != "fa4":
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
        _flash_env_get("HELION_CUTE_FLASH_RESCALE_THRESHOLD", rescale_threshold_default)
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
            _flash_env_get(
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
            _flash_env_get(
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
            _flash_env_get("HELION_CUTE_FLASH_CORR_REGS", str(corr_regs_default))
        )
        corr_regs_cfg = _cfg(FLASH_CORR_REGS_KEY)
        if corr_regs_cfg is not None:
            corr_regs = int(corr_regs_cfg)  # type: ignore[arg-type]
        if corr_regs % 8 != 0 or corr_regs <= 0:
            corr_regs = corr_regs_default
        other_regs_default = 48
        other_regs = int(
            _flash_env_get("HELION_CUTE_FLASH_OTHER_REGS", str(other_regs_default))
        )
        other_regs_cfg = _cfg(FLASH_OTHER_REGS_KEY)
        if other_regs_cfg is not None:
            other_regs = int(other_regs_cfg)  # type: ignore[arg-type]
        if other_regs % 8 != 0 or other_regs < 24:
            other_regs = other_regs_default
        # The 2 * softmax + corr + other <= 512 register-file budget is
        # enforced (and repaired) by ``ConfigSpec.normalize``; env overrides
        # for experiments are passed through as before.
        corr_tile_size_default = 8 if not is_causal and head_dim <= 64 else 16
        corr_tile_size = int(
            _flash_env_get(
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
        # ws_overlap staged epilogue chunk width. The consumer warpgroup has
        # registers for a whole 64-column O chunk, so one TMEM load per pass
        # replaces four serialized ones (the fused row epilogue walks O twice).
        corr_tile_size_default = 64 if head_dim % 64 == 0 else 16
        corr_tile_size = int(
            _flash_env_get(
                "HELION_CUTE_FLASH_CORR_TILE_SIZE",
                str(corr_tile_size_default),
            )
        )
        corr_tile_size_cfg = _cfg(FLASH_CORR_TILE_SIZE_KEY)
        if corr_tile_size_cfg is not None:
            corr_tile_size = int(corr_tile_size_cfg)  # type: ignore[arg-type]
        if corr_tile_size not in (16, 32, 64) or head_dim % corr_tile_size != 0:
            corr_tile_size = corr_tile_size_default
        if not epi_stg:
            # Only the staged epilogue reads the chunk width (the single-stage
            # and direct-store bodies never do): canonicalize the dead value
            # instead of aliasing one source three ways.
            corr_tile_size = corr_tile_size_default
    row_epilogue_warps_default = "softmax" if topology == "fa4" else "correction"
    row_epilogue_warps = str(
        _flash_env_get(
            "HELION_CUTE_FLASH_ROW_EPILOGUE_WARPS", row_epilogue_warps_default
        )
    )
    row_epilogue_warps_cfg = _cfg(FLASH_ROW_EPILOGUE_WARPS_KEY)
    if row_epilogue_warps_cfg is not None:
        row_epilogue_warps = str(row_epilogue_warps_cfg)
    if row_epilogue_warps not in ("correction", "softmax") or topology != "fa4":
        row_epilogue_warps = "correction"
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
        _flash_env_get(
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
        # Keep historical protocols in their tested serial order. The explicit
        # SAME_SLOT graph owns CTA-local credits and disjoint output pairs.
        causal_lpt_swizzle = (
            max(1, min(causal_lpt_swizzle, 64))
            if explicit_stateful
            else _flash_inactive_lpt_domain(is_causal=is_causal, topology=topology)[0]
        )
        if num_bh is not None:
            causal_lpt_swizzle = min(causal_lpt_swizzle, max(num_bh, 1))
    causal_kv_order_default = "ascending"
    causal_kv_order = _flash_env_get(
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
    role_map = _flash_env_get("HELION_CUTE_FLASH_ROLE_MAP", role_map_default)
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
            _flash_env_get("HELION_CUTE_FLASH_CGA2_LOCAL") is not None
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
        _flash_env_get("HELION_CUTE_FLASH_CLC_HEADS", str(clc_heads_per_batch_default))
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
        _flash_env_get(
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
    exp2_packet = _flash_env_get("HELION_CUTE_FLASH_EXP2_PACKET", exp2_packet_default)
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
            # The degree-2 pass is held to the bf16 output's rounding floor
            # (max abs error unchanged vs the exact pass); an fp32 LSE output
            # exposes the row sum (LSE error 1e-4 -> 2.5e-3 measured), so plain
            # row bodies with an LSE store keep the exact pass.
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
                *(() if is_causal else tuple(_FLASH_EXP2_PACKET_PARAMS)),
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
            _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET", str(e2e_offset_default))
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
            _flash_env_get("HELION_CUTE_FLASH_E2E_OFFSET0", str(e2e_offset0_default))
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
    # Statistics have 128 FP32 rows per query slot, independent of the head
    # width and 16-bit I/O dtype. Preserve the existing D64 default and ring2
    # repair separately from the general dense transport capability.
    stat_transport_eligible = topology == "fa4" and (not is_causal or explicit_stateful)
    legacy_stat_transport_eligible = (
        stat_transport_eligible and not is_causal and head_dim == 64
    )
    legacy_stat_handoff = _flash_bool_env("HELION_CUTE_FLASH_FA4_STAT_HANDOFF", True)
    stat_transport_default = (
        "single" if legacy_stat_transport_eligible and legacy_stat_handoff else "ring2"
    )
    stat_transport = _flash_env_get(
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
        legacy_stat_transport_eligible
        and exp2_impl == "split"
        and standard_dense_output
        and (not persistent or use_2cta_instrs)
        and not softmax_disc
        and rescale_threshold > 0.0
        and exp2_packet != _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET
    )
    if stat_transport == "single_final" and not single_final_stat_eligible:
        stat_transport = "single" if legacy_stat_transport_eligible else "ring2"
    if (
        legacy_stat_transport_eligible
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
    kv_tile_n = int(_flash_env_get("HELION_CUTE_FLASH_KV_TILE_N", "128") or "128")
    kv_tile_n_cfg = _cfg(FLASH_KV_TILE_N_KEY)
    if kv_tile_n_cfg is not None:
        kv_tile_n = int(cast("int", kv_tile_n_cfg))
    if not _flash_kv_tile_n_supported(
        kv_tile_n,
        head_dim=head_dim,
        num_kv=num_kv,
        topology=topology,
        is_causal=is_causal,
        s_stage=s_stage,
        desc_kv=kv_order == "descending",
        softmax_disc=softmax_disc,
    ):
        kv_tile_n = 128
    if kv_tile_n != 128:
        # Four 32-column P chunks per KV tile: one arrival each.
        p_chunk_arrive = False
    # 64-row query tiles: only the flat, staged, dense two-warpgroup ws_overlap
    # body has the two-rows-per-thread softmax/epilogue math, and only for
    # plain rows (score modifiers and fused row epilogues index rows by
    # 128-row tiles, one row per thread).
    q_tile_m = int(_flash_env_get("HELION_CUTE_FLASH_Q_TILE_M", "128"))
    q_tile_m_cfg = _cfg(FLASH_Q_TILE_M_KEY)
    if q_tile_m_cfg is not None:
        q_tile_m = int(q_tile_m_cfg)  # type: ignore[arg-type]
    if q_tile_m != 64 or not _flash_q_tile_m64_supported(
        topology=topology,
        s_stage=s_stage,
        persistent=persistent,
        epi_stg=epi_stg,
        head_dim=head_dim,
        kv_tile_n=kv_tile_n,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        small_biased_candidate=small_biased_candidate,
        plain_row_body=plain_row_body,
    ):
        q_tile_m = 128
    ws_one_pass = _flash_bool_env("HELION_CUTE_FLASH_WS_ONE_PASS", False)
    ws_one_pass_cfg = _cfg(FLASH_WS_ONE_PASS_KEY)
    if ws_one_pass_cfg is not None:
        ws_one_pass = bool(ws_one_pass_cfg)
    if not _flash_ws_one_pass_supported(
        q_tile_m=q_tile_m, num_kv=num_kv, kv_stage=kv_stage
    ):
        ws_one_pass = False

    if topology == "fa4" and not separate_kv_rings and head_dim in (64, 128):
        # The staging depth has to be capped for the *configured* tile width:
        # Cooperative MMA partitions each K/V slot across its two CTAs. A
        # wider tile still fits fewer stages, and exceeding the budget fails
        # with cudaErrorInvalidValue rather than falling back.
        aliased_kv_stage_cap = _flash_aliased_kv_stage_cap(
            head_dim,
            stage_output=epi_tma or epi_stg,
            kv_tile_n=kv_tile_n,
            use_2cta_instrs=use_2cta_instrs,
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

    persistent_loop = _flash_env_get("HELION_CUTE_FLASH_PERSISTENT_LOOP", "while")
    persistent_loop_cfg = _cfg(FLASH_PERSISTENT_LOOP_KEY)
    if persistent_loop_cfg is not None:
        persistent_loop = str(persistent_loop_cfg)
    if persistent_loop not in ("while", "counted") or not (
        topology == "fa4" and persistent and not use_clc_scheduler
    ):
        persistent_loop = "while"

    sp_row_sum = _flash_env_get("HELION_CUTE_FLASH_SP_ROW_SUM", "fragment")
    sp_row_sum_cfg = _cfg(FLASH_SP_ROW_SUM_KEY)
    if sp_row_sum_cfg is not None:
        sp_row_sum = str(sp_row_sum_cfg)
    sp_row_sum_eligible = (
        topology == "fa4"
        and not softmax_disc
        and ((not is_causal and _flash_supported_io_dtype(dtype)) or explicit_stateful)
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

    epi_tma_setup = _flash_env_get("HELION_CUTE_FLASH_EPI_TMA_SETUP", "shared")
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

    # The alternating-warpgroup family (``fa4_alt``) owns its structure: the
    # fa4 body's structural knobs below have no meaning there and are pinned so
    # that configs differing only in them normalize to one effective config.
    # Its only output path is the TMA store through the finished item's Q
    # stage, so a request without the TMA epilogue is not this family.
    alternating_warpgroups = (
        alt_requested
        and topology == "fa4"
        and epi_tma
        and _flash_supported_io_dtype(dtype)
        and not has_kv_tile_pruning
        and not small_biased_candidate
        and not use_2cta_instrs
        and not use_cga2_local_cta
        and not use_clc_scheduler
        and not local_tma_partition
        and not tensor_4d_tma
        and not separate_kv_rings
        and fa4_alt_supported(
            head_dim=head_dim,
            num_kv=num_kv,
            is_causal=is_causal,
            plain_row_body=plain_row_body,
            has_row_epilogue=has_row_epilogue,
            kv_tile_n=kv_tile_n,
        )
    )
    if alternating_warpgroups:
        softmax_lowering = "auto"
        row_sum_schedule = "post_acquire"
        pipeline_family = "fa4_alt"
        persistent = True
        persistent_ctas_per_sm = 1
        persistent_loop = "while"
        recompute_tile_coords = False
        q_tile_count = 1
        q_tile_m = 128
        # K ring depth (the V ring is two deep); 3 fills shared memory exactly.
        kv_stage = min(max(kv_stage, 2), 3)
        epi_stg = False
        epi_stg_store = "slice"
        epi_stg_gmem = "stage"
        epi_tma_setup = "shared"
        softmax_setup = "shared"
        sp_row_sum = "fragment"
        softmax_disc = True
        disc_pipe_depth = 1
        split_p_arrive = True
        p_chunk_arrive = False
        p_store_repetition = 16
        s_load_repetition = 32
        precompute_qk_desc = False
        first_load_order = 0
        kv_order = "ascending"
        stat_transport = "ring2"
        skip_rescale_stats = False
        mma_interleave = True
        mma_ptx = True
        role_map = "helion"
        role_chain = False
        masked_e2e_schedule = "inherit"
        masked_e2e_freq = e2e_freq
        masked_e2e_res = e2e_res
        # both warpgroups process the same rows: one emulation phase
        e2e_offset0 = e2e_offset
        # 2 * 192 + 64 + 64 fills the register file; the four single warps
        # spill below 64 and the softmax row does not fit above 192.
        softmax_regs = 192
        corr_regs = 64
        other_regs = 64
    if exp2_packet in _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS and (
        pipeline_family != "fa4" or not causal_loop_split
    ):
        exp2_packet = "1x1"
    rowmax = _cfg(FLASH_ROWMAX_KEY)
    if rowmax is None:
        rowmax = "software"
    if rowmax not in ("software", "tmem"):
        raise ValueError(f"invalid flash row-max lowering: {rowmax!r}")
    if (
        not _flash_tmem_rowmax_supported(
            target_device_capability=target_device_capability,
            fa4=topology == "fa4",
            score_compatible=tmem_rowmax_compatible,
        )
        or s_load_repetition != 32
        or alternating_warpgroups
    ):
        rowmax = "software"
    if explicit_stateful and (
        pipeline_family != "fa4"
        or requested_family_flags not in (None, FLASH_PIPELINE_FAMILY_FLAGS["fa4"])
        or persistent
        or q_tile_count != 2
        or not mma_ptx
        or kv_tile_n != 128
    ):
        raise InvalidConfig(
            "resident_stateful requires the ordinary nonpersistent two-slot "
            "FA4 PTX pipeline with KV128"
        )
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
        rowmax=cast("str", rowmax),
        softmax_lowering=cast("str", softmax_lowering),
        row_sum_schedule=cast("str", row_sum_schedule),
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
        p_chunk_arrive=p_chunk_arrive,
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
        row_epilogue_warps=row_epilogue_warps,
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
        kv_tile_n=kv_tile_n,
        q_tile_m=q_tile_m,
        ws_one_pass=ws_one_pass,
        alternating_warpgroups=alternating_warpgroups,
    )


def _resolve_row_mma_config(
    head_dim: int,
    num_kv: int,
    cfg: Callable[[str], object | None],
    *,
    dtype: torch.dtype,
    num_bh: int | None,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    supports_tensor_4d_tma: bool,
    prefer_packed_reduce: bool,
    plain_row_body: bool,
    has_row_epilogue: bool,
    has_score_modifiers: bool,
    aux_dtypes: Sequence[str] = (),
) -> FlashAttentionConfig | None:
    """Resolve the ``row_mma`` family: its two knobs over a fixed base.

    Every tcgen05 knob is dead for the row programs, so they take the values
    of the plain ``ws_overlap`` resolution of the same problem (independent of
    the requested config), which makes every row_mma config that differs only
    in dead knobs the same autotune identity.  The sequence is ``128 * num_kv``
    (the flash plan requires ``seq % 128 == 0``); a warp count the key range
    cannot be split over, or whose staging of the fused row epilogue's aux
    rows (``aux_dtypes``, known at codegen only) does not fit next to the K/V
    stages, falls back to the default knobs.  None declines the family when
    even those cannot be planned.
    """
    base = resolve_flash_config(
        head_dim,
        num_kv,
        {FLASH_TOPOLOGY_KEY: "ws_overlap"},
        dtype=dtype,
        num_bh=num_bh,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        prefer_packed_reduce=prefer_packed_reduce,
        plain_row_body=plain_row_body,
        has_row_epilogue=has_row_epilogue,
        has_score_modifiers=has_score_modifiers,
    )
    seq = 128 * num_kv

    def plannable(row_warps: int, row_tile_m: int) -> bool:
        return row_mma_shape_supported(
            seq=seq,
            head_dim=head_dim,
            row_warps=row_warps,
            row_tile_m=row_tile_m,
            aux_bytes=row_mma_aux_smem_bytes(
                head_dim=head_dim,
                row_tile_m=row_tile_m,
                row_warps=row_warps,
                aux_dtypes=aux_dtypes,
            ),
        )

    row_warps = ROW_MMA_DEFAULT_WARPS
    row_warps_cfg = cfg(FLASH_ROW_WARPS_KEY)
    if row_warps_cfg is not None and int(row_warps_cfg) in ROW_MMA_WARP_CHOICES:  # type: ignore[call-overload]
        row_warps = int(row_warps_cfg)  # type: ignore[call-overload]
    row_tile_m = ROW_MMA_DEFAULT_TILE_M
    row_tile_m_cfg = cfg(FLASH_ROW_TILE_M_KEY)
    if row_tile_m_cfg is not None and int(row_tile_m_cfg) in ROW_MMA_TILE_M_CHOICES:  # type: ignore[call-overload]
        row_tile_m = int(row_tile_m_cfg)  # type: ignore[call-overload]
    if not plannable(row_warps, row_tile_m):
        row_warps = ROW_MMA_DEFAULT_WARPS
        row_tile_m = ROW_MMA_DEFAULT_TILE_M
        if not plannable(row_warps, row_tile_m):
            return None
    return dataclasses.replace(
        base,
        topology=ROW_MMA_FAMILY,
        pipeline_family=ROW_MMA_FAMILY,
        # One CTA per row tile, never a persistent grid.
        persistent=False,
        row_warps=row_warps,
        row_tile_m=row_tile_m,
    )


def _flash_ws_one_pass_supported(*, q_tile_m: int, num_kv: int, kv_stage: int) -> bool:
    """Whether the one-pass softmax is legal: the 64-row body (flat, staged,
    dense, plain rows) over exactly two KV tiles, both issued by the prologue
    (a ring of at least two stages)."""
    return q_tile_m == 64 and num_kv == 2 and kv_stage >= 2


def _flash_q_tile_m64_supported(
    *,
    topology: str,
    s_stage: int,
    persistent: bool,
    epi_stg: bool,
    head_dim: int,
    kv_tile_n: int,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    small_biased_candidate: bool,
    plain_row_body: bool,
) -> bool:
    """Whether the 64-row query tile is legal for this schedule and body."""
    return (
        plain_row_body
        and topology == "ws_overlap"
        and s_stage == 2
        and not persistent
        and epi_stg
        and head_dim in (64, 128)
        and kv_tile_n == 128
        and not is_causal
        and not has_kv_tile_pruning
        and not small_biased_candidate
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
FLASH_P_CHUNK_ARRIVE_KEY = "cute_flash_p_chunk_arrive"
FLASH_P_STORE_REP_KEY = "cute_flash_p_store_rep"
FLASH_S_LOAD_REP_KEY = "cute_flash_s_load_rep"
FLASH_PRECOMPUTE_QK_DESC_KEY = "cute_flash_precompute_qk_desc"
FLASH_FIRST_LOAD_ORDER_KEY = "cute_flash_first_load_order"
# 0..4 permute the first work item's Q0/K0/Q1/V0 TMA issue order; 5 is the
# staged order: K0 and Q0 first, then (first work item only) wait for the MMA
# warp's signal that they landed before issuing Q1 and V0, so the kernel-start
# HBM burst carries only the bytes the first score tile needs.
FLASH_FIRST_LOAD_ORDER_CHOICES = (0, 1, 2, 3, 4, 5)
FLASH_FIRST_LOAD_ORDER_STAGED = 5
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
_FLASH_SOFTMAX_REGS_VALUES = (152, 160, 168, 176, 184, 192, 200)
_FLASH_CORR_REGS_VALUES = (64, 72, 80, 88, 96, 104, 112, 120, 128)
_FLASH_ROLE_REGISTER_BUDGET = 512
FLASH_OTHER_REGS_KEY = "cute_flash_other_regs"
FLASH_CORR_TILE_SIZE_KEY = "cute_flash_corr_tile_size"
FLASH_ROW_EPILOGUE_WARPS_KEY = "cute_flash_row_epilogue_warps"
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
FLASH_SOFTMAX_LOWERING_KEY = "cute_flash_softmax_lowering"
FLASH_ROW_SUM_SCHEDULE_KEY = "cute_flash_row_sum_schedule"
FLASH_SOFTMAX_SETUP_KEY = "cute_flash_softmax_setup"
FLASH_KV_TILE_N_KEY = "cute_flash_kv_tile_n"
FLASH_Q_TILE_M_KEY = "cute_flash_q_tile_m"
FLASH_WS_ONE_PASS_KEY = "cute_flash_ws_one_pass"
FLASH_ROW_WARPS_KEY = "cute_flash_row_warps"
FLASH_ROW_TILE_M_KEY = "cute_flash_row_tile_m"
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

# Active values whose legality needs other knobs set at the same time.  The
# structural coverage design adds one witness context per entry (the
# dependencies plus the value) so the value is reachable by construction
# instead of by a lucky combination of covering rows: the 64-row query tile
# and its one-pass softmax exist only on the flat, staged, two-stage
# ``ws_overlap`` body.
FLASH_AUTOTUNE_VALUE_DEPENDENCIES: dict[tuple[str, object], dict[str, object]] = {
    (FLASH_Q_TILE_M_KEY, 64): {
        FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
        FLASH_PERSISTENT_KEY: False,
        FLASH_EPI_STG_KEY: True,
        FLASH_KV_STAGE_KEY: 2,
    },
    (FLASH_WS_ONE_PASS_KEY, True): {
        FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
        FLASH_PERSISTENT_KEY: False,
        FLASH_EPI_STG_KEY: True,
        FLASH_KV_STAGE_KEY: 2,
        FLASH_Q_TILE_M_KEY: 64,
    },
}
FLASH_ROWMAX_KEY = "cute_flash_rowmax"


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
    FLASH_ROW_EPILOGUE_WARPS_KEY,
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
    FLASH_ROWMAX_KEY,
    FLASH_SOFTMAX_LOWERING_KEY,
    FLASH_ROW_SUM_SCHEDULE_KEY,
    FLASH_SOFTMAX_SETUP_KEY,
    FLASH_EPI_TMA_SETUP_KEY,
    FLASH_KV_TILE_N_KEY,
    FLASH_Q_TILE_M_KEY,
    FLASH_WS_ONE_PASS_KEY,
    FLASH_ROW_WARPS_KEY,
    FLASH_ROW_TILE_M_KEY,
    # New axes go last: the structural coverage design derives its row pattern
    # from the axis positions, and an inserted axis would reshuffle every later
    # axis's rows (the 64-row query tile's witness depends on them).
    FLASH_P_CHUNK_ARRIVE_KEY,
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
        FLASH_P_CHUNK_ARRIVE_KEY: config.p_chunk_arrive,
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
        FLASH_ROW_EPILOGUE_WARPS_KEY: config.row_epilogue_warps,
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
        FLASH_ROWMAX_KEY: config.rowmax,
        FLASH_SOFTMAX_LOWERING_KEY: config.softmax_lowering,
        FLASH_ROW_SUM_SCHEDULE_KEY: config.row_sum_schedule,
        FLASH_SOFTMAX_SETUP_KEY: config.softmax_setup,
        FLASH_EPI_TMA_SETUP_KEY: config.epi_tma_setup,
        FLASH_KV_TILE_N_KEY: config.kv_tile_n,
        FLASH_Q_TILE_M_KEY: config.q_tile_m,
        FLASH_WS_ONE_PASS_KEY: config.ws_one_pass,
        FLASH_ROW_WARPS_KEY: config.row_warps,
        FLASH_ROW_TILE_M_KEY: config.row_tile_m,
    }


def _flash_dense_resident_workload_supported(
    *, fa4: bool, dtype: torch.dtype, standard_dense_output: bool
) -> bool:
    # The helper keeps FP32 scores and packs P using the 16-bit I/O dtype.
    # Head width, KV count and persistence do not alter its score-row layout
    # or the surrounding single-slot protocol.
    return fa4 and _flash_supported_io_dtype(dtype) and standard_dense_output


def _flash_causal_stateful_workload_supported(
    *,
    fa4: bool,
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    is_causal: bool,
    standard_causal_output: bool,
    causal_resident_compatible: bool,
) -> bool:
    # The state object converts through the destination element type. Its
    # masked/unmasked loop and SAME_SLOT graph require complete query pairs;
    # the matcher supplies that proof, independently of any tuning policy.
    return (
        fa4
        and head_dim in (64, 128)
        and _flash_supported_io_dtype(dtype)
        and is_causal
        and standard_causal_output
        and causal_resident_compatible
        and num_kv >= 2
        and num_kv % 2 == 0
    )


def _flash_inactive_lpt_domain(*, is_causal: bool, topology: str) -> tuple[int, ...]:
    """Canonical inactive width first, followed by accepted historical aliases."""
    if not is_causal:
        return (0,)
    return (1, 0) if topology == "fa4" else (0, 1)


def _flash_resident_softmax_overrides(
    lowering: object, *, is_causal: bool | None = None, topology: str | None = None
) -> dict[str, object]:
    """Canonical controls owned by the explicit whole-row lowerings."""
    # Only the explicit stateful implementation exposes private row-sum
    # scheduling. Other modes retain their historical order; keeping this in
    # the ownership map also removes the inactive coordinate from proposals.
    values: dict[str, object] = (
        {}
        if lowering == "resident_stateful"
        else {FLASH_ROW_SUM_SCHEDULE_KEY: "post_acquire"}
    )
    # The inactive mapping differs between causal and dense kernels. Derive it
    # from the actual workload, never from an observed normalized parent.
    if (
        lowering != "resident_stateful"
        and is_causal is not None
        and topology is not None
    ):
        values[FLASH_CAUSAL_LPT_SWIZZLE_KEY] = _flash_inactive_lpt_domain(
            is_causal=is_causal, topology=topology
        )[0]
    if lowering not in ("resident_value_graph", "resident_stateful"):
        return values
    values.update(
        {
            FLASH_SOFTMAX_DISC_KEY: False,
            FLASH_DISC_PIPE_KEY: 1,
            FLASH_STAT_TRANSPORT_KEY: "single",
            FLASH_SPLIT_P_ARRIVE_KEY: True,
            FLASH_P_CHUNK_ARRIVE_KEY: False,
            # Rep32 stores halve the chunk count and cannot implement this helper's
            # split. Rep32 loads retain the full-row register layout it consumes.
            FLASH_P_STORE_REP_KEY: 16,
            FLASH_S_LOAD_REP_KEY: 32,
            FLASH_SP_ROW_SUM_KEY: "whole",
            FLASH_E2E_SCHEDULE_KEY: "xu",
            FLASH_MASKED_E2E_SCHEDULE_KEY: "inherit",
            FLASH_E2E_OFFSET_KEY: 0,
            FLASH_E2E_OFFSET0_KEY: 0,
            FLASH_EXP2_PACKET_KEY: "1x1",
            FLASH_EXP2_IMPL_KEY: "xu",
            FLASH_E2E_FREQ_KEY: 8,
            FLASH_E2E_RES_KEY: 0,
        }
    )
    if lowering == "resident_stateful":
        values.update(
            {
                FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
                FLASH_KV_TILE_N_KEY: 128,
            }
        )
    return values


def flash_tmem_rowmax_score_plan_supported(score_plan: AttentionScorePlan) -> bool:
    """TMEM scores can be reduced before the optional causal register mask."""
    return score_plan.modifier_kinds in ((DENSE_SCORE_KIND,), (CAUSAL_MASK_KIND,))


def _flash_tmem_rowmax_supported(
    *,
    target_device_capability: tuple[int, int] | None,
    fa4: bool,
    score_compatible: bool,
) -> bool:
    """Current LDRED layout is proved for FA4 dense/causal score plans.

    Hardware reduction reads FP32 scores, independent of input dtype and head
    dimension. Other score transforms require a separate post-transform proof.
    The 32-repetition load requirement is normalized with the actual config.
    """
    return (
        fa4
        and score_compatible
        and get_flash_target_policy(
            target_device_capability
        ).hardware.supports_tmem_row_reduce
    )


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


def _flash_stateful_lpt_candidates(num_bh: int | None) -> tuple[int, ...]:
    """Logarithmic effective widths for the explicit CTA-local decoder."""
    if num_bh is None or num_bh < 1:
        return (1,)
    return tuple(sorted({min(num_bh, width) for width in (1, 2, 4, 8, 16, 32, 64)}))


# A seed working-set budget, matching the upstream FA4 scheduling heuristic.
# This is a tuning policy, not a claim about the physical device's L2 capacity.
_FLASH_LPT_SEED_KV_BUDGET_BYTES = 50 * 1024 * 1024


def _flash_stateful_lpt_seed_widths(
    head_dim: int,
    num_kv: int,
    *,
    num_bh: int | None,
    num_sm: int | None,
    dtype: torch.dtype,
) -> tuple[int, ...]:
    """Balance KV locality, query-pair wave size and the legal head endpoint."""
    if num_bh is None or num_bh < 1 or num_sm is None or num_sm < 1:
        return (1,)
    kv_bytes_per_head = num_kv * 128 * (2 * head_dim) * (torch.finfo(dtype).bits // 8)
    query_pairs = max(1, num_kv // 2)
    cache_heads = max(1, _FLASH_LPT_SEED_KV_BUDGET_BYTES // kv_bytes_per_head)
    wave_heads = max(1, num_sm // query_pairs)
    cache_width = 1 << (cache_heads.bit_length() - 1)
    wave_width = 1 << (wave_heads.bit_length() - 1)
    endpoint = min(num_bh, 64)
    return tuple(
        sorted(
            {
                1,
                min(endpoint, cache_width, wave_width),
                min(endpoint, cache_width),
                endpoint,
            }
        )
    )


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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    supports_tensor_4d_tma: bool,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
    pipeline_family_override: str | None = None,
    target_device_capability: tuple[int, int] | None = None,
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
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        has_row_epilogue=has_row_epilogue,
        plain_row_body=plain_row_body,
        has_score_modifiers=has_score_modifiers,
        pipeline_family_override=pipeline_family_override,
        target_device_capability=target_device_capability,
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
    values = dict(values)
    # Adding an optional lowering must retain every existing raw seed.
    if values.get(FLASH_SOFTMAX_LOWERING_KEY) == "auto":
        values.pop(FLASH_SOFTMAX_LOWERING_KEY)
    if values.get(FLASH_ROW_SUM_SCHEDULE_KEY) == "post_acquire":
        values.pop(FLASH_ROW_SUM_SCHEDULE_KEY)
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
        FLASH_KV_TILE_N_KEY: policy.kv_tile_n,
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


def _flash_target_seed_templates(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype,
    num_bh: int | None,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    standard_dense_output: bool,
    standard_causal_output: bool,
    target_device_capability: tuple[int, int] | None,
    supports_tensor_4d_tma: bool,
    allowed_families: Sequence[str],
    output_requires_tma: bool = False,
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
) -> tuple[dict[str, object], ...]:
    """Project every target template onto the current implementation's legality.

    Historical policy lengths select emitter lowerings elsewhere; they neither
    select nor rank compiler seeds. A template contributes only when every
    explicit coordinate survives resolution, including its parent family and
    compound packet. This preserves optimized templates without leaking child
    values into incompatible families or silently changing their algorithm.
    """
    tuning = get_flash_target_policy(target_device_capability).tuning_for_torch(
        head_dim, str(dtype).removeprefix("torch.")
    )
    if (
        tuning is None
        or has_kv_tile_pruning
        or requires_ws_overlap
        or small_biased_candidate
    ):
        return ()
    if is_causal and standard_causal_output:
        templates = [
            _flash_causal_tuning_overrides(policy) for policy in tuning.causal_policies
        ]
    elif not is_causal and standard_dense_output:
        templates = [
            _flash_dense_tuning_overrides(policy) for policy in tuning.dense_policies
        ]
    else:
        return ()
    legal: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}
    for template in templates:
        expected = {**template, FLASH_Q_TILE_COUNT_KEY: 2}
        if expected[FLASH_PIPELINE_FAMILY_KEY] not in allowed_families:
            continue
        resolved = resolve_flash_config(
            head_dim,
            num_kv,
            expected,
            dtype=dtype,
            num_bh=num_bh,
            is_causal=is_causal,
            standard_dense_output=standard_dense_output,
            standard_causal_output=standard_causal_output,
            tmem_rowmax_compatible=tmem_rowmax_compatible,
            causal_resident_compatible=causal_resident_compatible,
            target_device_capability=target_device_capability,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
            has_row_epilogue=has_row_epilogue,
            plain_row_body=plain_row_body,
            has_score_modifiers=has_score_modifiers,
        )
        if not _flash_config_matches_tuning_values(resolved, expected):
            continue
        if output_requires_tma and not resolved.epi_tma:
            continue
        if 2 * resolved.softmax_regs + resolved.corr_regs + resolved.other_regs > 512:
            continue
        # Config content, never registry order, historical lengths or timings,
        # determines ordering. Identical templates contribute exactly once.
        identity = tuple((key, repr(value)) for key, value in sorted(expected.items()))
        legal[identity] = template
    return tuple(legal[identity] for identity in sorted(legal))


def _flash_config_matches_tuning_values(
    cfg: FlashAttentionConfig,
    expected: Mapping[str, object],
) -> bool:
    actual = flash_effective_config_values(cfg)
    return all(actual.get(key) == value for key, value in expected.items())


def _flash_fitted_probability_log2_shift(shift: int, rescale_threshold: float) -> int:
    """Return the largest legal probability shift at ``rescale_threshold``.

    The specialized dense bodies store probabilities pre-scaled by
    ``2**shift``, and alpha-pinning lets a stored probability reach
    ``2**(shift + rescale_threshold)``, which has to stay inside fp16. Shrink
    the shift to fit instead of abandoning the lowering: a shift of 0 is the
    original unshifted body, so every threshold stays expressible.
    """
    headroom = int(math.floor(FLASH_FLOAT16_MAX_LOG2 - rescale_threshold))
    return max(0, min(shift, headroom))


# FA4 pipeline families whose barrier graph the specialized dense softmax
# bodies can run under. The bodies themselves are CTA-count agnostic -- the
# causal resident lowering already runs on a one-CTA pipeline -- so the only
# requirement is the plain FA4 topology without a separate K/V ring, CGA2
# pairing or the CLC scheduler.
_FLASH_DENSE_LOWERING_FAMILIES = frozenset(
    {"fa4", "fa4_2cta", "fa4_tma_4d", "fa4_2cta_tma_4d"}
)


def _flash_dense_lowering_schedule_supported(
    cfg: FlashAttentionConfig,
    policy: FlashDenseTuningPolicy | None,
) -> bool:
    """Return whether ``cfg``'s schedule can host the policy's dense lowering.

    The specialized dense bodies (resident value graph, packed f16x2 exp2)
    replace the whole per-KV softmax value graph, so they need the surrounding
    FA4 schedule to supply exactly the barrier protocol, TMEM repetitions and
    statistics handoff they were written against. Only those structural
    requirements belong here.

    In particular this must NOT require ``cfg`` to equal the promoted seed
    field for field. Doing that turns every neighbour of the seed into a
    silent fallback to the standard body, which measured ~14% slower on
    ``2x32x32768x64`` fp16 -- a cliff the autotuner cannot climb out of, so the
    search converges back onto the seed no matter how long it runs.
    """
    if policy is None:
        return False
    lowering = policy.softmax_lowering
    packed = policy.packed_exp2_mode
    if (
        lowering is FlashSoftmaxLowering.STANDARD
        and packed is FlashPackedExp2Mode.DISABLED
    ):
        return False
    if not (
        cfg.pipeline_family in _FLASH_DENSE_LOWERING_FAMILIES
        and not cfg.persistent
        and cfg.q_tile_count == 2
        and cfg.split_p_arrive
        and cfg.exp2_impl == "split"
        # Both bodies are whole-row value graphs; the chunked ("disc") body
        # runs a different barrier protocol.
        and not cfg.softmax_disc
        and cfg.sp_row_sum == "whole"
        and cfg.e2e_schedule != "xu"
        # Both bodies pin alpha, so the rescale-skip lever must be armed.
        and cfg.rescale_threshold > 0.0
    ):
        return False
    if lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH:
        # The resident value graph acknowledges the statistics slot per KV
        # tile, which is the ``single`` transport's protocol.
        return cfg.stat_transport == "single"
    return cfg.stat_transport in ("single", "single_final") and (
        cfg.exp2_packet in _FLASH_DEG1_EXP2_PACKETS
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


def _flash_causal_resident_schedule_supported(
    cfg: FlashAttentionConfig,
    policy: FlashCausalTuningPolicy | None,
) -> bool:
    """Return whether ``cfg``'s schedule can host the causal resident lowering.

    Same reasoning as the dense gate: list the structural requirements the
    lowering actually has, not a byte-exact comparison against the promoted
    seed. The exact match cost ~20% for every neighbour of the causal seed on
    GB300 (1367 -> ~1090 TFLOP/s on 2x32x262144x64 fp16, the same number for
    every single-field perturbation), which is a cliff the autotuner cannot
    climb out of.
    """
    if policy is None:
        return False
    return (
        cfg.pipeline_family == "fa4"
        and not cfg.persistent
        and cfg.q_tile_count == 2
        # The degree-2 causal body is written against this exp2 packet and its
        # paired masked/unmasked cadence.
        and cfg.exp2_packet == _FLASH_DEG2_EXP2_PACKET
        and cfg.e2e_schedule == "16/6"
        and cfg.masked_e2e_schedule == "16/6"
        and cfg.split_p_arrive
        and cfg.rescale_threshold > 0.0
        and cfg.p_store_repetition == 16
        and cfg.s_load_repetition == 32
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
    tmem_rowmax_compatible: bool,
    causal_resident_compatible: bool = False,
    supports_tensor_4d_tma: bool,
    target_device_capability: tuple[int, int] | None,
    values: Mapping[str, object],
    expected: Mapping[str, object],
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
) -> Config:
    """Build a target seed and reject policies normalized by config resolution.

    The tuning tables are measured on the plain row body.  A fused row epilogue
    or a score modifier pins some of their children (the whole-row-only
    statistics transport and row-sum form, the partial-tile KV width) exactly
    as normalization does, so for those bodies the seed carries the resolved
    values instead of being rejected: it then equals its normalized form.
    """
    seed = Config.from_dict(dict(values))
    if seed.config.get(FLASH_SOFTMAX_LOWERING_KEY) == "auto":
        seed.config.pop(FLASH_SOFTMAX_LOWERING_KEY)
    if seed.config.get(FLASH_ROW_SUM_SCHEDULE_KEY) == "post_acquire":
        seed.config.pop(FLASH_ROW_SUM_SCHEDULE_KEY)
    resolved = resolve_flash_config(
        head_dim,
        num_kv,
        seed.config,
        dtype=dtype,
        num_bh=num_bh,
        is_causal=is_causal,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        plain_row_body=plain_row_body,
        has_row_epilogue=has_row_epilogue,
        has_score_modifiers=has_score_modifiers,
        target_device_capability=target_device_capability,
    )
    actual = flash_effective_config_values(resolved)
    mismatches = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if not mismatches:
        return seed
    if plain_row_body and not has_row_epilogue:
        raise ValueError(
            "flash target tuning policy does not round-trip through config "
            f"resolution: {mismatches!r}"
        )
    return Config.from_dict(
        {
            **seed.config,
            **{key: value for key, value in actual.items() if key in seed.config},
        }
    )


def _flash_target_seed_configs(
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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    target_device_capability: tuple[int, int] | None,
    supports_tensor_4d_tma: bool,
    block_size_targets: Sequence[int],
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
) -> tuple[Config, ...]:
    if tmem_rowmax_compatible is None:
        tmem_rowmax_compatible = standard_dense_output or standard_causal_output
    block_sizes = _flash_seed_block_sizes(block_size_targets)
    if block_sizes is None:
        return ()
    common = {
        "dtype": dtype,
        "num_bh": num_bh,
        "is_causal": is_causal,
        "has_kv_tile_pruning": has_kv_tile_pruning,
        "requires_ws_overlap": requires_ws_overlap,
        "small_biased_candidate": small_biased_candidate,
        "standard_dense_output": standard_dense_output,
        "standard_causal_output": standard_causal_output,
        "supports_tensor_4d_tma": supports_tensor_4d_tma,
        "has_row_epilogue": has_row_epilogue,
        "plain_row_body": plain_row_body,
        "has_score_modifiers": has_score_modifiers,
    }
    families = _flash_legal_autotune_pipeline_families(
        head_dim, num_kv, output_requires_tma=False, requested_family=None, **common
    )
    templates = _flash_target_seed_templates(
        head_dim,
        num_kv,
        target_device_capability=target_device_capability,
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        allowed_families=families,
        **common,
    )
    seeds: list[Config] = []
    for template in templates:
        fragments = flash_autotune_fragments(
            head_dim,
            num_kv,
            tensor_4d_heads=tensor_4d_heads,
            tmem_rowmax_compatible=tmem_rowmax_compatible,
            causal_resident_compatible=causal_resident_compatible,
            target_device_capability=target_device_capability,
            pipeline_family_override=cast("str", template[FLASH_PIPELINE_FAMILY_KEY]),
            **common,
        )
        values = {key: fragment.default() for key, fragment in fragments.items()}
        if not _flash_seed_set_all(values, fragments, template):
            raise AssertionError("legal flash target template is absent from fragments")
        # Preserve the row-reduction implementation of former target seeds.
        # Instruction legality is independent of the template's historical length.
        rowmax = (
            "tmem"
            if _flash_tmem_rowmax_supported(
                target_device_capability=target_device_capability,
                fa4=True,
                score_compatible=tmem_rowmax_compatible,
            )
            and values[FLASH_S_LOAD_REP_KEY] == 32
            else "software"
        )
        values[FLASH_ROWMAX_KEY] = rowmax
        seeds.append(
            _flash_validated_target_seed(
                head_dim=head_dim,
                num_kv=num_kv,
                dtype=dtype,
                num_bh=num_bh,
                is_causal=is_causal,
                standard_dense_output=standard_dense_output,
                standard_causal_output=standard_causal_output,
                tmem_rowmax_compatible=tmem_rowmax_compatible,
                causal_resident_compatible=causal_resident_compatible,
                supports_tensor_4d_tma=supports_tensor_4d_tma,
                target_device_capability=target_device_capability,
                has_row_epilogue=has_row_epilogue,
                plain_row_body=plain_row_body,
                has_score_modifiers=has_score_modifiers,
                values={
                    "block_sizes": block_sizes,
                    **values,
                    FLASH_Q_TILE_COUNT_KEY: 2,
                },
                expected={
                    **template,
                    FLASH_Q_TILE_COUNT_KEY: 2,
                    FLASH_ROWMAX_KEY: rowmax,
                },
            )
        )
    return tuple(seeds)


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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    supports_tensor_4d_tma: bool = True,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
    seed_kind: str = "default",
) -> Config | None:
    """Return the first legal template in deterministic order, or a generic seed.

    The default is also the first plural seed. Historical lengths do not select
    templates; only implementation legality can change the candidate set.
    """

    if num_kv is None:
        return None
    if seed_kind == "default":
        target_seeds = _flash_target_seed_configs(
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
            tmem_rowmax_compatible=tmem_rowmax_compatible,
            causal_resident_compatible=causal_resident_compatible,
            target_device_capability=target_device_capability,
            supports_tensor_4d_tma=supports_tensor_4d_tma,
            block_size_targets=block_size_targets,
            has_row_epilogue=has_row_epilogue,
            plain_row_body=plain_row_body,
            has_score_modifiers=has_score_modifiers,
        )
        if target_seeds:
            return target_seeds[0]
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
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        has_row_epilogue=has_row_epilogue,
        plain_row_body=plain_row_body,
        has_score_modifiers=has_score_modifiers,
        target_device_capability=target_device_capability,
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


def _flash_stateful_joint_seed_configs(
    seeds: Sequence[Config],
    fragments: Mapping[str, ConfigSpecFragment],
    resolve_seed: Callable[[Mapping[str, object]], FlashAttentionConfig],
) -> tuple[Config, ...]:
    """Append legal depth/register endpoints to the existing stateful templates."""
    parents = [
        seed
        for seed in seeds
        if seed.config.get(FLASH_PIPELINE_FAMILY_KEY) == "fa4"
        and seed.config.get(FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        and seed.config.get(FLASH_ROWMAX_KEY) == "tmem"
    ]
    depths = cast("EnumFragment", fragments[FLASH_KV_STAGE_KEY])._active_choices()
    registers = cast(
        "EnumFragment", fragments[FLASH_SOFTMAX_REGS_KEY]
    )._active_choices()
    if not parents or not registers:
        return ()
    resolved_parents = [(seed, resolve_seed(seed.config)) for seed in parents]
    seen = {
        _flash_config_with_values(
            cast("Sequence[int]", seed.config["block_sizes"]),
            flash_effective_config_values(resolved),
        )
        for seed, resolved in resolved_parents
    }
    result: list[Config] = []
    for seed, parent in resolved_parents:
        values = flash_effective_config_values(parent)
        low_registers = min(cast("Sequence[int]", registers))
        register_endpoints = (parent.softmax_regs,)
        if low_registers < parent.softmax_regs:
            register_endpoints += (low_registers,)
        for depth in sorted(
            (
                value
                for value in cast("Sequence[int]", depths)
                if parent.kv_stage < value <= 2 * parent.kv_stage
            ),
            reverse=True,
        ):
            deeper = {**values, FLASH_KV_STAGE_KEY: depth}
            # The family domain can exceed this parent's output-storage cap.
            # Try the next depth when normalization clamps or changes its graph.
            if not _flash_config_matches_tuning_values(resolve_seed(deeper), deeper):
                continue
            for softmax_regs in register_endpoints:
                if softmax_regs not in registers:
                    continue
                candidate_values = {**deeper, FLASH_SOFTMAX_REGS_KEY: softmax_regs}
                if 2 * softmax_regs + parent.corr_regs + parent.other_regs > 512:
                    continue
                if not _flash_config_matches_tuning_values(
                    resolve_seed(candidate_values), candidate_values
                ):
                    continue
                candidate = _flash_config_with_values(
                    cast("Sequence[int]", seed.config["block_sizes"]), candidate_values
                )
                if candidate not in seen:
                    seen.add(candidate)
                    result.append(candidate)
            break
    return tuple(result)


def flash_attention_seed_configs(
    head_dim: int,
    num_kv: int | None,
    *,
    num_bh: int | None = None,
    num_sm: int | None = None,
    tensor_4d_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    has_kv_tile_pruning: bool = False,
    requires_ws_overlap: bool = False,
    small_biased_candidate: bool = False,
    standard_dense_output: bool = False,
    standard_causal_output: bool = False,
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    supports_tensor_4d_tma: bool = True,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
    block_size_targets: Sequence[int] = _FLASH_SEED_BLOCK_SIZE_TARGETS,
    device_sm_count: int = 0,
) -> tuple[Config, ...]:
    """Return measured starting points plus generic structural coverage.

    All legal target templates precede generic family and compound-packet
    coverage. Template order depends only on config content, and every candidate
    is subsequently measured; none is selected without benchmarking.

    ``device_sm_count`` (0 when unknown) lets a problem whose 128-row tiles
    cannot fill the device seed the 64-row tile of the flat ws_overlap grid.
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
        "tmem_rowmax_compatible": tmem_rowmax_compatible,
        "causal_resident_compatible": causal_resident_compatible,
        "supports_tensor_4d_tma": supports_tensor_4d_tma,
        "has_row_epilogue": has_row_epilogue,
        "plain_row_body": plain_row_body,
        "has_score_modifiers": has_score_modifiers,
        "target_device_capability": target_device_capability,
    }
    fragments = flash_autotune_fragments(head_dim, num_kv, **common)
    base_values = {key: fragment.default() for key, fragment in fragments.items()}
    seeds: list[Config] = []
    target_seeds = _flash_target_seed_configs(
        head_dim,
        num_kv,
        block_size_targets=block_size_targets,
        **common,
    )
    seeds.extend(target_seeds)
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
        # The flat ws_overlap grid is a distinct source variant (prologue TMA
        # loads issued before the TMEM barrier, TMEM freed under the output
        # drain) from the persistent ws_overlap loop the family seed carries.
        # It is seeded with the staged O epilogue the measured flat winners
        # use; the fragment default follows the surface's default family,
        # which is the direct store wherever that family is fa4.
        (
            FLASH_PERSISTENT_KEY,
            False,
            {FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", FLASH_EPI_STG_KEY: True},
        ),
        (FLASH_ROWMAX_KEY, "tmem", {FLASH_S_LOAD_REP_KEY: 32}),
        (
            FLASH_STAT_TRANSPORT_KEY,
            "single",
            {FLASH_SOFTMAX_DISC_KEY: False, FLASH_EXP2_PACKET_KEY: "1x1"},
        ),
        (
            FLASH_STAT_TRANSPORT_KEY,
            "single",
            {
                FLASH_SOFTMAX_DISC_KEY: False,
                FLASH_EXP2_PACKET_KEY: "1x1",
                FLASH_E2E_SCHEDULE_KEY: "xu",
                FLASH_E2E_OFFSET_KEY: 0,
                FLASH_E2E_OFFSET0_KEY: 0,
            },
        ),
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
        (
            FLASH_SOFTMAX_LOWERING_KEY,
            "resident_value_graph",
            {
                key: value
                for key, value in _flash_resident_softmax_overrides(
                    "resident_value_graph"
                ).items()
                if key in FLASH_AUTOTUNE_CONFIG_KEYS
            },
        ),
        (
            FLASH_SOFTMAX_LOWERING_KEY,
            "resident_stateful",
            {
                **_flash_resident_softmax_overrides("resident_stateful"),
                FLASH_PIPELINE_FAMILY_KEY: "fa4",
            },
        ),
        (FLASH_SOFTMAX_LOWERING_KEY, "standard", {}),
    )
    if has_row_epilogue:
        # The fa4 family seeds carry the softmax-warpgroup row epilogue (the
        # fa4 default); seed the correction-warp evaluation once as well.
        source_variants += (
            (
                FLASH_ROW_EPILOGUE_WARPS_KEY,
                "correction",
                {FLASH_PIPELINE_FAMILY_KEY: "fa4"},
            ),
        )
    if device_sm_count > 0 and num_bh is not None and num_bh * num_kv < device_sm_count:
        # A grid of fewer 128-row tiles than SMs is bound by per-CTA fixed
        # latency: seed the 64-row tile of the flat ws_overlap grid (twice the
        # CTAs, half the softmax per CTA) with the one-pass softmax where the
        # sequence is exactly two KV tiles. The variant loop below adds it only
        # where the 64-row tile is a searchable choice.
        small_grid: dict[str, object] = {
            FLASH_PERSISTENT_KEY: False,
            FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            # The 64-row tile is legal only with the staged O epilogue (a
            # direct-store default would canonicalize the tile back to 128
            # rows and the one-pass softmax off); the two-stage K/V ring
            # measured best for it.
            FLASH_EPI_STG_KEY: True,
            FLASH_KV_STAGE_KEY: 2,
        }
        one_pass_fragment = fragments[FLASH_WS_ONE_PASS_KEY]
        assert isinstance(one_pass_fragment, EnumFragment)
        if True in (
            one_pass_fragment.choices
            if one_pass_fragment.search_choices is None
            else one_pass_fragment.search_choices
        ):
            small_grid[FLASH_WS_ONE_PASS_KEY] = True
        source_variants += ((FLASH_Q_TILE_M_KEY, 64, small_grid),)
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

    # These lowerings interact with row reduction and pipeline depth. Marginal
    # coverage of each field does not give their combination a starting point.
    # Retain the normal family defaults, including its register allocation, and
    # let tuning measure both the ordinary ring and one deeper prefetch ring.
    lowering_fragment = fragments[FLASH_SOFTMAX_LOWERING_KEY]
    assert isinstance(lowering_fragment, EnumFragment)
    for family, lowering in (
        ("fa4_2cta", "resident_value_graph"),
        ("fa4", "resident_stateful"),
    ):
        if (
            family not in families
            or lowering not in lowering_fragment._active_choices()
        ):
            continue
        joint_fragments = flash_autotune_fragments(
            head_dim, num_kv, pipeline_family_override=family, **common
        )
        rowmax_fragment = joint_fragments[FLASH_ROWMAX_KEY]
        assert isinstance(rowmax_fragment, EnumFragment)
        if "tmem" not in rowmax_fragment._active_choices():
            continue
        values = {
            **{key: fragment.default() for key, fragment in joint_fragments.items()},
            **{
                key: value
                for key, value in _flash_resident_softmax_overrides(lowering).items()
                if key in FLASH_AUTOTUNE_CONFIG_KEYS
            },
            FLASH_PIPELINE_FAMILY_KEY: family,
            FLASH_SOFTMAX_LOWERING_KEY: lowering,
            FLASH_ROWMAX_KEY: "tmem",
        }
        depth = cast("int", values[FLASH_KV_STAGE_KEY])
        depths = [depth]
        if lowering == "resident_value_graph":
            depth_fragment = joint_fragments[FLASH_KV_STAGE_KEY]
            assert isinstance(depth_fragment, EnumFragment)
            deeper = max(
                candidate
                for candidate in depth_fragment._active_choices()
                if isinstance(candidate, int) and candidate <= 2 * depth
            )
            if deeper > depth:
                depths.append(deeper)
        for depth in depths:
            values[FLASH_KV_STAGE_KEY] = depth
            resolved = resolve_flash_config(
                head_dim,
                num_kv,
                values,
                dtype=dtype,
                num_bh=num_bh,
                is_causal=is_causal,
                has_kv_tile_pruning=has_kv_tile_pruning,
                requires_ws_overlap=requires_ws_overlap,
                small_biased_candidate=small_biased_candidate,
                standard_dense_output=standard_dense_output,
                standard_causal_output=standard_causal_output,
                tmem_rowmax_compatible=tmem_rowmax_compatible,
                causal_resident_compatible=causal_resident_compatible,
                supports_tensor_4d_tma=supports_tensor_4d_tma,
                target_device_capability=target_device_capability,
                has_row_epilogue=has_row_epilogue,
                plain_row_body=plain_row_body,
                has_score_modifiers=has_score_modifiers,
            )
            if (
                resolved.pipeline_family == family
                and resolved.softmax_lowering == lowering
                and resolved.rowmax == "tmem"
                and resolved.kv_stage == depth
            ):
                seeds.append(_flash_config_with_values(block_sizes, values))

    unique: list[Config] = []
    for seed in seeds:
        if seed not in unique:
            unique.append(seed)
    # Preserve every normal post-acquire seed, and give the conditionally
    # active scheduling alternative the same general legal starting points.
    # No shape-specific configuration or measured winner is injected.
    for seed in tuple(unique):
        if seed.config.get(FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful":
            values = {**seed.config, FLASH_ROW_SUM_SCHEDULE_KEY: "pre_acquire"}
            candidate = Config.from_dict(values)
            if candidate not in unique:
                unique.append(candidate)
    # Retain that entire prefix. Seed the same normal stateful templates at a
    # small geometry-derived set of mappings, including both row-sum orders.
    stateful_seeds = tuple(
        seed
        for seed in unique
        if seed.config.get(FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        and seed.config.get(FLASH_CAUSAL_LPT_SWIZZLE_KEY) == 1
    )
    if stateful_seeds:
        for width in _flash_stateful_lpt_seed_widths(
            head_dim, num_kv, num_bh=num_bh, num_sm=num_sm, dtype=dtype
        ):
            if width == 1:
                continue
            for seed in stateful_seeds:
                candidate = Config.from_dict(
                    {**seed.config, FLASH_CAUSAL_LPT_SWIZZLE_KEY: width}
                )
                if candidate not in unique:
                    unique.append(candidate)
    # Preserve the complete old prefix, including row-sum and LPT coverage.
    if any(
        seed.config.get(FLASH_PIPELINE_FAMILY_KEY) == "fa4"
        and seed.config.get(FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        and seed.config.get(FLASH_ROWMAX_KEY) == "tmem"
        for seed in unique
    ):
        joint_fragments = flash_autotune_fragments(
            head_dim, num_kv, pipeline_family_override="fa4", **common
        )

        # Head count is fragment-only; all resolver context is already shared.
        resolve_seed = partial(
            resolve_flash_config,
            head_dim,
            num_kv,
            **{key: value for key, value in common.items() if key != "tensor_4d_heads"},
        )

        unique.extend(
            _flash_stateful_joint_seed_configs(unique, joint_fragments, resolve_seed)
        )
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
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
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
        if (
            family == ROW_MMA_FAMILY
            and requested_family is None
            and not row_mma_search_grid(num_bh=num_bh, num_kv=num_kv)
        ):
            # The row programs stream every key once per 8-row tile, so
            # they win only where the 128-row tiles cannot fill the
            # device. Unattended searches skip them on larger grids (no
            # seed, coverage row or fragment); explicit configs stay legal.
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
            plain_row_body=plain_row_body,
            has_row_epilogue=has_row_epilogue,
            has_score_modifiers=has_score_modifiers,
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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    target_device_capability: tuple[int, int] | None = None,
    output_requires_tma: bool = False,
    supports_tensor_4d_tma: bool = True,
    has_row_epilogue: bool = False,
    plain_row_body: bool = True,
    has_score_modifiers: bool = False,
    topology_override: str | None = None,
    pipeline_family_override: str | None = None,
) -> dict[str, ConfigSpecFragment]:
    """Build a legality-driven, length-invariant flash search surface.

    Sequence length participates only in structural proofs such as even-pair or
    four-tile cluster alignment.  Within one legality class, defaults, active
    choices, and their ordering are identical for every length.
    """

    if tmem_rowmax_compatible is None:
        tmem_rowmax_compatible = standard_dense_output or standard_causal_output

    valid_topology = (
        topology_override
        if topology_override in ("fa4", "ws_overlap", ROW_MMA_FAMILY)
        else None
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
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        prefer_packed_reduce=has_kv_tile_pruning or requires_ws_overlap,
        plain_row_body=plain_row_body,
        has_row_epilogue=has_row_epilogue,
        has_score_modifiers=has_score_modifiers,
        target_device_capability=target_device_capability,
    )
    paired = num_kv >= 2 and num_kv % 2 == 0
    cluster_aligned = num_kv >= 4 and num_kv % 4 == 0
    fa4_eligible = paired and not requires_ws_overlap
    # The ws_overlap warpgroup path measures its staged O epilogue knobs.
    ws_staged_search = defaults.topology == "ws_overlap" and defaults.s_stage == 2
    requested_search_topology = (
        valid_family_flags.topology
        if valid_family_flags is not None
        else valid_topology
    )
    fa4_search_eligible = fa4_eligible and requested_search_topology in (None, "fa4")
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
        has_row_epilogue=has_row_epilogue,
        plain_row_body=plain_row_body,
        has_score_modifiers=has_score_modifiers,
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
        # A mixed-family fragment must represent the deepest legal cooperative
        # ring. Normalization applies each selected family's own resource cap.
        # Pinning a local-MMA family keeps its original per-CTA storage domain.
        use_2cta_kv_layout = any(
            FLASH_PIPELINE_FAMILY_FLAGS[family].use_2cta_instrs
            for family in active_families
        )
        value_cap = _flash_aliased_kv_stage_cap(
            head_dim,
            stage_output=False,
            use_2cta_instrs=use_2cta_kv_layout,
        )
        search_cap = _flash_aliased_kv_stage_cap(
            head_dim,
            stage_output=output_requires_tma,
            use_2cta_instrs=use_2cta_kv_layout,
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
        # Causal plans and fused row epilogues resolve to the chunked body.
        softmax_disc = enum(
            defaults.softmax_disc,
            (True, False),
            (True,) if is_causal or has_row_epilogue else (True, False),
        )
    else:
        softmax_disc = enum(
            defaults.softmax_disc, (True, False), (defaults.softmax_disc,)
        )

    if fa4_search_eligible:
        disc_pipe = enum(defaults.disc_pipe_depth, (1, 2, 3, 4), (1, 2, 3, 4))
        split_p = enum(defaults.split_p_arrive, (False, True), (False, True))
        p_chunk = enum(
            defaults.p_chunk_arrive,
            (False, True),
            (False,) if is_causal else (False, True),
        )
        p_store = enum(defaults.p_store_repetition, (16, 32), (16, 32))
        s_load = enum(defaults.s_load_repetition, (16, 32), (16, 32))
        precompute = enum(defaults.precompute_qk_desc, (False, True), (False, True))
        first_load = enum(
            defaults.first_load_order,
            FLASH_FIRST_LOAD_ORDER_CHOICES,
            FLASH_FIRST_LOAD_ORDER_CHOICES,
        )
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
        p_chunk = enum(
            defaults.p_chunk_arrive, (False, True), (defaults.p_chunk_arrive,)
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
            FLASH_FIRST_LOAD_ORDER_CHOICES,
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
    # setmaxnreg budgets: 8 softmax + 4 correction + 4 other warps share the
    # 64K-register file, so 2 * softmax + corr + other <= 512 (checked when the
    # config is resolved). Heavier correction epilogues (fused row programs)
    # trade softmax registers for correction registers.
    softmax_regs = enum(
        defaults.softmax_regs,
        _FLASH_SOFTMAX_REGS_VALUES,
        _FLASH_SOFTMAX_REGS_VALUES if fa4_search_eligible else (defaults.softmax_regs,),
    )
    corr_regs = enum(
        defaults.corr_regs,
        _FLASH_CORR_REGS_VALUES,
        _FLASH_CORR_REGS_VALUES if fa4_search_eligible else (defaults.corr_regs,),
    )
    other_regs = enum(
        defaults.other_regs,
        (24, 32, 40, 48, 56, 64, 80),
        (24, 32, 40, 48, 56, 64, 80) if fa4_search_eligible else (defaults.other_regs,),
    )
    # ws_overlap keeps its 64-column epilogue chunk fixed (one TMEM load per
    # pass): searching it would only inflate the small ws-only raw product with
    # source-identical direct-store aliases.
    corr_tile = enum(
        defaults.corr_tile_size,
        (8, 16, 32, 64),
        (8, 16, 32) if fa4_search_eligible else (defaults.corr_tile_size,),
    )
    # Which FA4 warps evaluate a fused row epilogue. Without a row program the
    # value is source-neutral, so it is only searched when one exists.
    row_epilogue_warps = enum(
        defaults.row_epilogue_warps,
        ("correction", "softmax"),
        ("correction", "softmax")
        if fa4_search_eligible and has_row_epilogue
        else (defaults.row_epilogue_warps,),
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
        # The ws_overlap warpgroup path measures both O epilogues (staged
        # coalesced store vs direct per-thread STG).
        epi_stg = enum(
            defaults.epi_stg,
            (False, True),
            (False, True) if ws_staged_search else (defaults.epi_stg,),
        )
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
    if d64_fa4 or (head_dim == 128 and fa4_search_eligible and not is_causal):
        # The plain packets only batch the scale/exp2 instruction emission of
        # the chunked softmax pass (same arithmetic per element). Dense hd128
        # searches them like hd64; the causal hd128 surface keeps its resident
        # packet envelope.
        packet_values = list(_FLASH_EXP2_PACKET_PARAMS)
        packet_search = list(_FLASH_EXP2_PACKET_PARAMS)
    if d64_fa4:
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
                tmem_rowmax_compatible=tmem_rowmax_compatible,
                causal_resident_compatible=causal_resident_compatible,
                supports_tensor_4d_tma=supports_tensor_4d_tma,
                has_row_epilogue=has_row_epilogue,
                plain_row_body=plain_row_body,
                has_score_modifiers=has_score_modifiers,
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

    stateful_search_eligible = _flash_causal_stateful_workload_supported(
        fa4=fa4_search_eligible and not has_row_epilogue,
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        is_causal=is_causal,
        standard_causal_output=standard_causal_output,
        causal_resident_compatible=causal_resident_compatible,
    ) and pipeline_family_override in (None, "fa4")
    if stateful_search_eligible:
        causal_lpt = enum(
            defaults.causal_lpt_swizzle,
            range(65),
            _flash_stateful_lpt_candidates(num_bh),
        )
    if fa4_search_eligible and not is_causal:
        transports = (
            ("ring2", "single", "single_final")
            if dense_d64_fa4 and standard_dense_output
            else ("ring2", "single")
        )
        if fixed_family_requires_persistence or has_row_epilogue:
            # These families require persistence, while single_final needs a
            # nonpersistent grid or the two-CTA instruction path; it also needs
            # the whole-row body, which a fused row epilogue never gets.
            transports = ("ring2", "single")
        stat_transport = enum(
            defaults.stat_transport,
            transports,
            transports,
        )
    elif stateful_search_eligible:
        # Single transport is a child of the explicit stateful graph; it does
        # not enable an unproven generic causal single-slot implementation.
        stat_transport = enum("ring2", ("ring2", "single"), ("ring2",))
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
    # ``whole`` needs the whole-row body, which a fused row epilogue never gets.
    sp_row_sum = enum(
        defaults.sp_row_sum,
        ("fragment", "whole"),
        ("fragment", "whole")
        if fa4_search_eligible
        and not is_causal
        and _flash_supported_io_dtype(dtype)
        and not has_row_epilogue
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

    # A wider KV tile amortizes the per-tile softmax correction over more
    # columns. Now that the trailing partial tile is masked, a width no longer
    # has to divide the sequence, so which widths are legal depends only on the
    # workload class -- head dim, dtype, causality, topology and the TMEM
    # budget -- and the search surface stays length-independent.
    # A fused row epilogue keeps the chunked body, which cannot mask a trailing
    # partial tile; the only width that is legal for every length is 128.
    kv_tile_n_search = tuple(
        width
        for width in FLASH_KV_TILE_N_CHOICES
        if (width == 128 or not has_row_epilogue)
        and _flash_kv_tile_n_supported(
            width,
            head_dim=head_dim,
            num_kv=num_kv,
            topology=defaults.topology,
            is_causal=is_causal,
            s_stage=defaults.s_stage,
        )
    )
    kv_tile_n = enum(
        defaults.kv_tile_n, FLASH_KV_TILE_N_CHOICES, kv_tile_n_search or (128,)
    )
    # The 64-row query tile doubles the CTA count of the flat staged dense
    # ws_overlap body. It is sampled from the shared flat surface, so it is
    # offered in the paired legality classes (whose product is already large)
    # and left out of the small ws-only odd classes, which the autotuner
    # enumerates exactly and where every alias would cost a measurement.
    # A pinned ws_overlap surface (family/topology override, required ws) keeps
    # the persistent default, from which the 64-row tile cannot round-trip;
    # the shared surface is where it is sampled and the ws family keeps it.
    # Score modifiers and fused row epilogues have the 128-row body only, so
    # their surfaces never offer 64 (it would alias 128 in tuned configs).
    q_tile_m_searchable = (
        paired
        and plain_row_body
        and head_dim in (64, 128)
        and not is_causal
        and not has_kv_tile_pruning
        and not small_biased_candidate
        and not requires_ws_overlap
        and not output_requires_tma
        and valid_family is None
        and valid_topology is None
    )
    q_tile_m = enum(
        defaults.q_tile_m, (128, 64), (128, 64) if q_tile_m_searchable else (128,)
    )
    # The one-pass softmax of the 64-row tile is sampled where that tile is,
    # on surfaces whose sequence is exactly two KV tiles.
    ws_one_pass_searchable = q_tile_m_searchable and _flash_ws_one_pass_supported(
        q_tile_m=64, num_kv=num_kv, kv_stage=2
    )
    ws_one_pass = enum(
        defaults.ws_one_pass,
        (False, True),
        (False, True) if ws_one_pass_searchable else (defaults.ws_one_pass,),
    )
    # The row-program knobs are searched only when the family itself is in
    # play (its structure is independent of the sequence length, so the
    # surface stays length-invariant); otherwise they stay at their defaults.
    row_mma_searchable = ROW_MMA_FAMILY in active_families and valid_family in (
        None,
        ROW_MMA_FAMILY,
    )
    row_warps = enum(
        defaults.row_warps,
        ROW_MMA_WARP_CHOICES,
        ROW_MMA_WARP_CHOICES if row_mma_searchable else (defaults.row_warps,),
    )
    row_tile_m = enum(
        defaults.row_tile_m,
        ROW_MMA_TILE_M_CHOICES,
        ROW_MMA_TILE_M_CHOICES if row_mma_searchable else (defaults.row_tile_m,),
    )

    fragments: dict[str, ConfigSpecFragment] = {
        FLASH_KV_TILE_N_KEY: kv_tile_n,
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
        FLASH_ROW_EPILOGUE_WARPS_KEY: row_epilogue_warps,
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
        FLASH_SOFTMAX_LOWERING_KEY: enum(
            "auto",
            ("auto", "standard", "resident_value_graph", "resident_stateful"),
            ("auto", "standard", "resident_value_graph")
            if _flash_dense_resident_workload_supported(
                fa4=fa4_search_eligible and not has_row_epilogue,
                dtype=dtype,
                standard_dense_output=standard_dense_output and not is_causal,
            )
            else ("auto", "standard", "resident_stateful")
            if stateful_search_eligible
            else ("auto",),
        ),
        FLASH_ROWMAX_KEY: enum(
            "software",
            ("software", "tmem"),
            ("software", "tmem")
            if _flash_tmem_rowmax_supported(
                target_device_capability=target_device_capability,
                fa4=fa4_search_eligible,
                score_compatible=tmem_rowmax_compatible,
            )
            else ("software",),
        ),
        FLASH_ROW_SUM_SCHEDULE_KEY: enum(
            "post_acquire",
            ("post_acquire", "pre_acquire"),
            ("post_acquire", "pre_acquire")
            if stateful_search_eligible
            else ("post_acquire",),
        ),
        FLASH_SOFTMAX_SETUP_KEY: softmax_setup,
        FLASH_EPI_TMA_SETUP_KEY: epi_tma_setup,
        FLASH_Q_TILE_M_KEY: q_tile_m,
        FLASH_WS_ONE_PASS_KEY: ws_one_pass,
        FLASH_ROW_WARPS_KEY: row_warps,
        FLASH_ROW_TILE_M_KEY: row_tile_m,
        FLASH_P_CHUNK_ARRIVE_KEY: p_chunk,
    }
    if valid_family == "fa4_alt":
        for key, value in (
            (FLASH_SOFTMAX_LOWERING_KEY, "auto"),
            (FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire"),
            (FLASH_ROWMAX_KEY, "software"),
            (FLASH_SP_ROW_SUM_KEY, "fragment"),
        ):
            fragment = fragments[key]
            assert isinstance(fragment, EnumFragment)
            fragments[key] = EnumFragment(fragment.choices, (value,))
    if valid_family == ROW_MMA_FAMILY:
        # Every other knob is dead for the row programs: pin its search to the
        # default so the family's surface is exactly its two knobs.
        for key, fragment in list(fragments.items()):
            if key in (
                FLASH_PIPELINE_FAMILY_KEY,
                FLASH_ROW_WARPS_KEY,
                FLASH_ROW_TILE_M_KEY,
            ):
                continue
            assert isinstance(fragment, EnumFragment)
            fragments[key] = EnumFragment(fragment.choices, (fragment.default(),))
    templates = _flash_target_seed_templates(
        head_dim,
        num_kv,
        dtype=dtype,
        num_bh=num_bh,
        is_causal=is_causal,
        has_kv_tile_pruning=has_kv_tile_pruning,
        requires_ws_overlap=requires_ws_overlap,
        small_biased_candidate=small_biased_candidate,
        standard_dense_output=standard_dense_output,
        standard_causal_output=standard_causal_output,
        target_device_capability=target_device_capability,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        allowed_families=active_families,
        has_row_epilogue=has_row_epilogue,
        plain_row_body=plain_row_body,
        has_score_modifiers=has_score_modifiers,
        output_requires_tma=output_requires_tma,
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
    )
    for template in templates:
        for key, value in template.items():
            fragment = cast("EnumFragment", fragments[key])
            if value not in fragment.choices:
                search_choices = fragment.search_choices
                if search_choices is None:
                    # Seed compatibility values do not widen the generic
                    # coordinate search or structural coverage surface.
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
    tmem_rowmax_compatible: bool | None = None,
    causal_resident_compatible: bool = False,
    supports_tensor_4d_tma: bool = True,
    plain_row_body: bool = True,
    has_row_epilogue: bool = False,
    has_score_modifiers: bool = False,
    target_device_capability: tuple[int, int] | None = None,
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
        tmem_rowmax_compatible=tmem_rowmax_compatible,
        causal_resident_compatible=causal_resident_compatible,
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        plain_row_body=plain_row_body,
        has_row_epilogue=has_row_epilogue,
        has_score_modifiers=has_score_modifiers,
        target_device_capability=target_device_capability,
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
_FLASH_RUNTIME_ABI = 7

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

# Only modules with an alternating-warpgroup body import its runtime, so every
# other flash render (and its disk-cache key) is unchanged by the family.
_FLASH_ALT_PREAMBLE_IMPORTS = """\
import helion._compiler.cute._flash_alt_runtime as _helion_flash_alt_rt
"""


def emit_flash_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module flash imports."""
    if getattr(cg, "_helion_flash_module_emitted", False):
        return
    cg._helion_flash_module_emitted = True  # type: ignore[attr-defined]
    for line_stmt in ast.parse(_FLASH_PREAMBLE_IMPORTS).body:
        cg.module_statements.append(line_stmt)


def emit_flash_alt_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the alternating family's runtime."""
    if getattr(cg, "_helion_flash_alt_module_emitted", False):
        return
    cg._helion_flash_alt_module_emitted = True  # type: ignore[attr-defined]
    for line_stmt in ast.parse(_FLASH_ALT_PREAMBLE_IMPORTS).body:
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
    probability_publication: str = ""
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
{iteration.probability_publication}
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
    row_max: str = "flash_row_max",
    row_sum: str = "flash_row_sum",
) -> str:
    value = f"{row_max} * _flash_scale_log2 + cute.math.log2({row_sum})"
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


def _flash_ws_active_range_prelude(score_plan: AttentionScorePlan, num_kv: int) -> str:
    """Contiguous-mask KV range prelude + ``flash_active_count`` (8-space
    indented, newline-terminated); empty for dense / non-range masks."""
    contiguous_range = _flash_contiguous_kv_tile_range(score_plan, num_kv)
    if contiguous_range is None:
        return ""
    range_prelude, first_kv_expr, last_kv_expr = contiguous_range
    return (
        f"{range_prelude}        flash_active_count = "
        f"{last_kv_expr} - {first_kv_expr} + cutlass.Int32(1)\n"
    )


def _flash_ws_prologue_loads(
    num_kv: int,
    kv_stage: int,
    *,
    score_plan: AttentionScorePlan,
    tensors: str = "kv",
) -> str:
    """Warp-0 source issuing the Q TMA plus the first ``min(kv_stage, num_kv)``
    K/V TMA pairs (8-space indented, starts with a newline).

    ``tensors`` selects the operands: ``"kv"`` (Q, then K/V per stage, the
    persistent producer's order), ``"k"`` (K tile 0 first -- the largest
    operand of the first MMA -- then Q, then the remaining K tiles) or
    ``"v"`` (the V tiles).  The flat one-tile grid emits the ``"k"`` block
    right after the Q/K pipelines exist and the ``"v"`` block once the V
    pipeline does, both BEFORE the TMEM allocation barrier, so the loads are
    in flight while the CTA finishes its prologue; the persistent path keeps
    the ``"kv"`` block at the head of the producer role.  Contiguous-range
    masks expect ``flash_active_count`` to be defined (see
    ``_flash_ws_active_range_prelude``).
    """
    assert tensors in ("kv", "k", "v")
    contiguous_range = _flash_contiguous_kv_tile_range(score_plan, num_kv)
    q_issue = """
        flash_q_empty = flash_q_prod.acquire_and_advance()
        cute.copy(_flash_tma_q, tQgQ[None, flash_m_tile], tQsQ[None, flash_q_empty.index],
                  tma_bar_ptr=flash_q_empty.barrier)"""
    pieces: list[str] = []
    if contiguous_range is not None:
        _, first_kv_expr, _ = contiguous_range
        for pf in range(min(kv_stage, num_kv)):
            k_piece = f"""
            flash_k_empty = flash_k_prod.acquire_and_advance()
            cute.copy(_flash_tma_k, tKgK[None, {first_kv_expr} + cutlass.Int32({pf})],
                      tKsK[None, flash_k_empty.index], tma_bar_ptr=flash_k_empty.barrier)"""
            v_piece = f"""
            flash_v_empty = flash_v_prod.acquire_and_advance()
            cute.copy(_flash_tma_v, tVgV[None, {first_kv_expr} + cutlass.Int32({pf})],
                      tVsV[None, flash_v_empty.index], tma_bar_ptr=flash_v_empty.barrier)"""
            body = {"kv": k_piece + v_piece, "k": k_piece, "v": v_piece}[tensors]
            pieces.append(
                f"""
        if cutlass.Int32({pf}) < flash_active_count:{body}"""
            )
    else:
        for pf in range(min(kv_stage, num_kv)):
            k_piece = f"""
        flash_k_empty = flash_k_prod.acquire_and_advance()
        cute.copy(_flash_tma_k, tKgK[None, {pf}], tKsK[None, flash_k_empty.index],
                  tma_bar_ptr=flash_k_empty.barrier)"""
            v_piece = f"""
        flash_v_empty = flash_v_prod.acquire_and_advance()
        cute.copy(_flash_tma_v, tVgV[None, {pf}], tVsV[None, flash_v_empty.index],
                  tma_bar_ptr=flash_v_empty.barrier)"""
            piece = {"kv": k_piece + v_piece, "k": k_piece, "v": v_piece}[tensors]
            pieces.append(
                _flash_guard(
                    piece,
                    _flash_kv_tile_active_condition(
                        score_plan,
                        bh_expr="flash_bh",
                        m_tile_expr="flash_m_tile",
                        kv_tile_expr=f"cutlass.Int32({pf})",
                    ),
                    indent="        ",
                )
            )
    if tensors == "v":
        return "".join(pieces)
    if tensors == "k" and pieces:
        # K tile 0 is the first MMA's largest operand: it leaves first.
        return pieces[0] + q_issue + "".join(pieces[1:])
    return q_issue + "".join(pieces)


def _flash_ws_early_prologue_loads(
    num_kv: int,
    kv_stage: int,
    *,
    score_plan: AttentionScorePlan,
    tensors: str = "kv",
) -> str:
    """Setup-time (4-space indented) warp-0 block that starts the Q/K/V TMA
    loads of the flat one-tile grid before the TMEM allocation barrier (see
    ``_flash_ws_prologue_loads`` for ``tensors``).

    Warp 0 owns every mbarrier init (``PipelineTmaUmma.create`` initializes
    from warp 0's elected lane), so after ``mbarrier_init_fence`` plus a warp
    sync it may arrive on the freshly initialized full barriers before the
    CTA-wide barrier publishes them to the other warps.

    Contiguous-range plans guard each issue with ``flash_active_count``,
    which ``_flash_ws_early_active_range`` defines once, for every warp,
    ahead of the first block.
    """
    body = _flash_ws_prologue_loads(
        num_kv, kv_stage, score_plan=score_plan, tensors=tensors
    ).lstrip("\n")
    return f"""
if warp_idx == 0:
    cute.arch.sync_warp()
{textwrap.indent(textwrap.dedent(body), "    ")}
"""


def _flash_ws_early_active_range(score_plan: AttentionScorePlan, num_kv: int) -> str:
    """Setup-time (4-space indented, newline-terminated) contiguous-range
    prelude of the flat one-tile grid: the active KV tile range and
    ``flash_active_count`` computed once, by every warp, before the two
    warp-0 issue blocks that guard their loads with it; empty for dense and
    non-range masks."""
    return textwrap.dedent(_flash_ws_active_range_prelude(score_plan, num_kv))


# The otherwise idle warp 1 owns the ws_overlap TMEM allocation: it allocates
# (and hands the permit back) right after the shared-memory setup, while warp
# 0 initializes the mbarriers and issues the first TMA loads, and it frees TMEM
# at the end.  Named barrier 1 is the allocation barrier every thread joins;
# named barrier 3 is the flat staged body's teardown barrier (warp 1 plus the
# consumer warpgroup), which also serves as the consumer warpgroup's sync
# between staging O to smem and draining it.
_FLASH_WS_TMEM_WARP = 1
_FLASH_WS_TEARDOWN_BARRIER_ID = 3
_FLASH_WS_TEARDOWN_THREADS = 32 + 128


def _flash_ws_one_pass_producer_body() -> str:
    """Producer (warp 0) inner body for the one-pass softmax of a 64-row tile
    over exactly two KV tiles (8-space indented; the prologue issued Q and
    both K/V tiles): QK(0) -> S0 and QK(1) -> S1 back to back, one P handoff,
    then PV(0) and PV(1) accumulate into O with a single mma_o commit.
    """
    return """
        flash_q_full = flash_q_cons.wait_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        # QK(0) -> S0 and QK(1) -> S1: the consumer reads both as one row.
        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q_full.index],
                      tSrK[None, None, flash_kp, flash_k_full.index], tStS0)
        flash_s_handle.commit()
        flash_k_full.release()
        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS1, tSrQ[None, None, flash_kp, flash_q_full.index],
                      tSrK[None, None, flash_kp, flash_k_full.index], tStS1)
        flash_s_handle.commit()
        flash_k_full.release()
        flash_q_full.release()
        # Both V tiles and the O accumulator are waited for ahead of the P
        # handoffs (they are long ready), so after each P tile lands only the
        # PV MMAs remain: PV(0) runs under the consumer's second exp2 sweep and
        # PV(1) accumulates into O with no rescale between them.
        flash_o_handle = flash_mma_o_prod.acquire_and_advance()
        flash_v_full0 = flash_v_cons.wait_and_advance()
        flash_v_full1 = flash_v_cons.wait_and_advance()
        flash_p_full = flash_p_ready_cons.wait_and_advance()
        for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_pv_mma, tOtO, tOrP0[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full0.index], tOtO)
        flash_v_full0.release()
        flash_p_full.release()
        flash_p_full = flash_p_ready_cons.wait_and_advance()
        for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
            _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, cutlass.Boolean(True))
            cute.gemm(_flash_pv_mma, tOtO, tOrP1[None, None, flash_kp, 0],
                      tOrV[None, None, flash_kp, flash_v_full1.index], tOtO)
        flash_o_handle.commit()
        flash_v_full1.release()
        flash_p_full.release()"""


def _flash_ws_producer_body(
    num_kv: int,
    kv_stage: int,
    hd: int,
    *,
    score_plan: AttentionScorePlan,
    early_issue: bool = False,
    one_pass: bool = False,
) -> str:
    """Producer (warp 0) body for the Stage-4 double-buffered-S overlap.

    With ``early_issue`` the Q TMA and the prologue K/V loads were already
    issued in the common setup (flat grid); the body then starts by waiting
    for Q.

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
    if one_pass:
        assert early_issue and num_kv == 2 and kv_stage >= 2
        return _flash_ws_one_pass_producer_body()
    contiguous_range = _flash_contiguous_kv_tile_range(score_plan, num_kv)
    prologue_loads = (
        ""
        if early_issue
        else _flash_ws_prologue_loads(num_kv, kv_stage, score_plan=score_plan)
    )
    if contiguous_range is None:
        range_prelude = ""
        first_kv_expr = ""
        last_kv_expr = ""
    else:
        range_prelude, first_kv_expr, last_kv_expr = contiguous_range
        active_count_expr = f"{last_kv_expr} - {first_kv_expr} + cutlass.Int32(1)"
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
        return f"""{range_prelude}        flash_active_count = {active_count_expr}{prologue_loads}
        flash_q_full = flash_q_cons.wait_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        flash_p_idx = cutlass.Int32(0)   # P-fragment parity (kv % 2)
        flash_qk_idx = cutlass.Int32(0)  # S target parity among active QK tiles
        flash_o_started = cutlass.Boolean(False)
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
    return f"""{range_prelude}{prologue_loads.lstrip(chr(10))}
        flash_q_full = flash_q_cons.wait_and_advance()
        flash_nk = cute.size(tSrQ, mode=[2])
        flash_nk2 = cute.size(tOrP0, mode=[2])
        flash_p_idx = cutlass.Int32(0)   # P-fragment parity (kv % 2)
        flash_qk_idx = cutlass.Int32(0)  # S target parity among active QK tiles
        flash_o_started = cutlass.Boolean(False){qk0_body}{qk0_pf}
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
    row_epilogue: FlashRowEpilogueEmit | None = None,
    staged_store: bool = False,
    early_teardown: bool = False,
    lse_store: str = "",
    two_rows: bool = False,
    aux_smem_staging: bool = False,
    one_pass: bool = False,
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
    relu_keyword = ", relu_output=True" if relu_output else ""
    o_mma_index = (
        "flash_pvt.partition_C({tensor})[None, None, None, flash_m_tile, 0, flash_bh]"
    )
    ws_o_setup = ""
    ws_o_prologue = ""
    if staged_store:
        # Staged epilogue: t2r -> 1/l scale -> cast -> swizzled sO (the FA4
        # correction-epilogue helper), release the O accumulator, sync the
        # consumer warpgroup, then drain sO with 16-byte row-contiguous vector
        # stores. The direct per-thread STG.128 path below writes one 128-byte
        # row per lane, so every warp-level store touches 32 distinct lines
        # (~1000 LSU wavefronts for a 128x64 bf16 tile, 0.46 us per CTA);
        # the staged drain writes 512 contiguous bytes per warp instruction.
        if row_epilogue is not None:
            # Three-way split: the copy-view setup plus an L2 prefetch of the
            # aux rows runs ahead of the KV loop; the aux loads and the passes
            # that do not read O run before the wait for the last PV (under
            # the MMA); only the O passes follow it.
            setup_lines, ws_o_prologue, ws_o_stage = (
                _flash_row_epilogue_smem_source_parts(
                    row_epilogue,
                    hd=hd,
                    io_dtype=io_dtype,
                    indent="        ",
                    suffix="",
                    mma_index=o_mma_index,
                    corr_tile_size=cfg.corr_tile_size,
                    use_2cta_instrs=False,
                    tOtO="tOtO",
                    sO="sO",
                    tidx="flash_local_tidx",
                    inv_sum="flash_inv_sum",
                    split=True,
                    aux_prefetch=not aux_smem_staging,
                    aux_from_smem=aux_smem_staging,
                )
            )
            ws_o_setup = "\n".join(setup_lines) + "\n"
            if aux_smem_staging:
                # The aux tile was TMA-loaded into sO with the prologue loads.
                ws_o_prologue = (
                    "        _helion_flash_rt.mbar_spin_wait(flash_aux_full_ptr, "
                    f"flash_aux_full_phase, {cfg.wait_hint})\n" + ws_o_prologue
                )
                ws_o_stage += "\n        flash_aux_full_phase ^= 1"
            ws_o_prologue += "\n"
        elif two_rows:
            ws_o_stage = f"""        _helion_flash_rt.ws_m64_epilogue_to_smem(
            flash_pvt, tOtO, sO, flash_local_tidx, flash_inv_sum_a, flash_inv_sum_b,
            {hd}, {cfg.corr_tile_size}, {io_dtype}{relu_keyword})"""
        else:
            ws_o_stage = f"""        _helion_flash_rt.fa4_correction_epilogue_to_smem_scoped(
            flash_pvt, tOtO, sO, flash_local_tidx, flash_inv_sum,
            {hd}, {cfg.corr_tile_size}, {io_dtype}{relu_keyword})"""
        # The flat grid folds the warpgroup sync into the teardown barrier it
        # shares with warp 1, which frees TMEM while the stores drain; the
        # persistent loop keeps its end-of-tile CTA barrier and only needs the
        # warpgroup here.
        stage_sync = (
            f"cute.arch.barrier(barrier_id={_FLASH_WS_TEARDOWN_BARRIER_ID},"
            f" number_of_threads={_FLASH_WS_TEARDOWN_THREADS})"
            if early_teardown
            else "cute.arch.barrier(barrier_id=2, number_of_threads=128)"
        )
        drain_fn = (
            "fa4_store_o_smem_to_gmem_whole"
            if cfg.epi_stg_store == "whole"
            else "fa4_store_o_smem_to_gmem"
        )
        ws_o_store = f"""{ws_o_stage}
        cute.arch.fence_view_async_tmem_load()
        flash_o_full.release()
        {stage_sync}
        flash_gO_stg = cute.local_tile(
            _flash_mOt[None, None, flash_bh], ({64 if two_rows else 128}, {hd}), (flash_m_tile, 0))
        _helion_flash_rt.{drain_fn}(
            flash_o_gmem_tiled_copy, flash_o_gmem_thr, tOsO_epi, flash_gO_stg, {io_dtype})"""
    elif row_epilogue is not None:
        ws_o_store = "\n".join(
            _flash_row_epilogue_direct_source(
                row_epilogue,
                hd=hd,
                io_dtype=io_dtype,
                indent="        ",
                suffix="",
                mma_index=o_mma_index,
                epi_tiler="flash_epi_tiler",
                thr_ld="flash_thr_o_ld",
                tiled_ld="flash_tiled_o_ld",
                tDtO="tDtO",
                tDgO="tDgO",
                inv_sum="flash_inv_sum",
                ld_repetition=64,
            )
        )
    else:
        ws_o_store = f"""        flash_reg = cute.make_rmem_tensor(tDgO[None, None, 0].shape, cutlass.Float32)
        flash_rego = cute.make_rmem_tensor(tDgO[None, None, 0].shape, {io_dtype})
        for flash_i in cutlass.range(cute.size(tDtO, mode=[2])):
            cute.copy(flash_tiled_o_ld, tDtO[None, None, flash_i], flash_reg)
            flash_reg.store(flash_reg.load() * flash_inv_sum){relu_store}
            flash_rego.store(flash_reg.load().to({io_dtype}))
            cute.autovec_copy(flash_rego, tDgO[None, None, flash_i])"""
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
    state_init = """        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
        flash_s_idx = cutlass.Int32(0)
        flash_o_started = cutlass.Boolean(False)"""
    inv_sum_init = "        flash_inv_sum = cutlass.Float32(1.0) / flash_row_sum"
    if two_rows:
        # 64-row tiles: each consumer thread owns two rows (r and r+8) of the
        # 16x256b fragment, alternating in column pairs, and shares each row
        # with the other three lanes of its quad. Every row statistic is kept
        # twice and finished with quad shuffles.
        consumer_iter = _flash_ws_two_row_consumer_iter(
            hd=hd,
            cfg=cfg,
            io_dtype=io_dtype,
            score_transform=score_transform,
        )
        state_init = """        flash_row_max_a = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_max_b = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum_a = cutlass.Float32(0.0)
        flash_row_sum_b = cutlass.Float32(0.0)
        flash_s_idx = cutlass.Int32(0)
        flash_o_started = cutlass.Boolean(False)"""
        inv_sum_init = """        flash_inv_sum_a = cutlass.Float32(1.0) / flash_row_sum_a
        flash_inv_sum_b = cutlass.Float32(1.0) / flash_row_sum_b"""
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
    if one_pass:
        assert two_rows and staged_store and num_kv == 2
        assert contiguous_range is None and not score_plan.modifiers
        state_init = ""
        range_prelude = ""
        loop_header = ""
        consumer_iter = _flash_ws_one_pass_consumer_block(cfg, io_dtype=io_dtype)

    # The LSE only needs the final row statistics: store it right after the
    # KV loop (under the last PV) instead of after the O drain.
    if staged_store:
        return f"""{ws_o_setup}{state_init}
{range_prelude}{loop_header}{consumer_iter}{lse_store}
{ws_o_prologue}
        flash_o_full = flash_mma_o_cons.wait_and_advance()
{inv_sum_init}
{ws_o_store}"""
    assert not two_rows
    return f"""        flash_row_max = cutlass.Float32(-cutlass.Float32.inf)
        flash_row_sum = cutlass.Float32(0.0)
        flash_s_idx = cutlass.Int32(0)
        flash_o_started = cutlass.Boolean(False)
{range_prelude}{loop_header}{consumer_iter}{lse_store}

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
{ws_o_store}
        flash_o_full.release()"""


def _flash_ws_one_pass_consumer_block(
    cfg: FlashAttentionConfig, *, io_dtype: str
) -> str:
    """The consumer's whole softmax for the one-pass form of a 64-row tile
    over two KV tiles (8-space indented, replaces the per-tile loop).

    Both score tiles are loaded first, so the row maximum spans the full
    256-wide row; the exp2 sweep then finishes and hands off one P tile at a
    time (PV(0) runs under the second tile's exp2, and the first tile's
    registers die before the second's conversion); the row sums add the two
    halves before the quad shuffle; O accumulates both PV MMAs without a
    rescale.
    """

    def reduce_max(tensor: str, init: str) -> str:
        if cfg.packed_reduce:
            return f"_helion_flash_rt.fmax_reduce_packed({tensor}, {init})"
        return f"{tensor}.load().reduce(cute.ReductionOp.MAX, {init}, 0)"

    def reduce_sum(tensor: str) -> str:
        if cfg.packed_reduce:
            return f"_helion_flash_rt.fadd_reduce_packed({tensor})"
        return f"{tensor}.load().reduce(cute.ReductionOp.ADD, 0.0, 0)"

    def exp2_rows(tensor: str, minus_max_scale: str) -> str:
        if cfg.exp2_impl == "split":
            return f"""        _helion_flash_rt.exp2_split_inplace(
            {tensor}, _flash_scale_log2, {minus_max_scale}, {cfg.e2e_freq}, {cfg.e2e_res})"""
        return f"""        for flash_j in cutlass.range_constexpr(cute.size({tensor})):
            {tensor}[flash_j] = cute.math.exp2(
                {tensor}[flash_j] * _flash_scale_log2 + {minus_max_scale}, fastmath=True)"""

    neg_inf = "cutlass.Float32(-cutlass.Float32.inf)"
    return f"""
        # One-pass softmax over both score tiles (S0 | S1 is one 256-wide row per
        # query row): a single max, one exp2 sweep, one P handoff, no rescale.
        flash_s_full0 = flash_mma_s_cons.wait_and_advance()
        tLDrS0 = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
        cute.copy(flash_tiled_ld0, tLDtS0, tLDrS0)
        flash_s_full1 = flash_mma_s_cons.wait_and_advance()
        tLDrS1 = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
        cute.copy(flash_tiled_ld1, tLDtS1, tLDrS1)
        cute.arch.fence_view_async_tmem_load()
        tLDrS0_a, tLDrS0_b = _helion_flash_rt.ws_m64_row_views(tLDrS0)
        tLDrS1_a, tLDrS1_b = _helion_flash_rt.ws_m64_row_views(tLDrS1)
        flash_row_max_a = _helion_flash_rt.quad_max({reduce_max("tLDrS1_a", reduce_max("tLDrS0_a", neg_inf))})
        flash_row_max_b = _helion_flash_rt.quad_max({reduce_max("tLDrS1_b", reduce_max("tLDrS0_b", neg_inf))})
        flash_row_max_safe_a = flash_row_max_a
        if flash_row_max_a == -cutlass.Float32.inf:
            flash_row_max_safe_a = cutlass.Float32(0.0)
        flash_row_max_safe_b = flash_row_max_b
        if flash_row_max_b == -cutlass.Float32.inf:
            flash_row_max_safe_b = cutlass.Float32(0.0)
        flash_minus_max_scale_a = (0.0 - flash_row_max_safe_a) * _flash_scale_log2
        flash_minus_max_scale_b = (0.0 - flash_row_max_safe_b) * _flash_scale_log2
{exp2_rows("tLDrS0_a", "flash_minus_max_scale_a")}
{exp2_rows("tLDrS0_b", "flash_minus_max_scale_b")}
        flash_p_sum0_a = {reduce_sum("tLDrS0_a")}
        flash_p_sum0_b = {reduce_sum("tLDrS0_b")}
        tSTrS0 = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
        tSTrS0_e = cute.make_tensor(
            cute.recast_ptr(tSTrS0.iterator, dtype={io_dtype}), tLDrS0.layout)
        tSTrS0_e.store(tLDrS0.load().to({io_dtype}))
        cute.copy(flash_tiled_st0, tSTrS0, tSTtS0)
        cute.arch.fence_view_async_tmem_store()
        flash_p_handle = flash_p_ready_prod.acquire_and_advance()
        flash_p_handle.commit()
        flash_s_full0.release()
{exp2_rows("tLDrS1_a", "flash_minus_max_scale_a")}
{exp2_rows("tLDrS1_b", "flash_minus_max_scale_b")}
        flash_row_sum_a = _helion_flash_rt.quad_sum(flash_p_sum0_a + {reduce_sum("tLDrS1_a")})
        flash_row_sum_b = _helion_flash_rt.quad_sum(flash_p_sum0_b + {reduce_sum("tLDrS1_b")})
        tSTrS1 = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
        tSTrS1_e = cute.make_tensor(
            cute.recast_ptr(tSTrS1.iterator, dtype={io_dtype}), tLDrS1.layout)
        tSTrS1_e.store(tLDrS1.load().to({io_dtype}))
        cute.copy(flash_tiled_st1, tSTrS1, tSTtS1)
        cute.arch.fence_view_async_tmem_store()
        flash_p_handle = flash_p_ready_prod.acquire_and_advance()
        flash_p_handle.commit()
        flash_s_full1.release()"""


def _flash_ws_two_row_consumer_iter(
    *,
    hd: int,
    cfg: FlashAttentionConfig,
    io_dtype: str,
    score_transform: str,
) -> str:
    """One KV iteration of the consumer body for 64-row query tiles.

    The 16x256b score fragment holds two rows per thread (``_a`` = row r,
    ``_b`` = row r+8, alternating in column pairs); ``ws_m64_row_views`` splits
    it, every reduction finishes with the quad shuffles of ``quad_max`` /
    ``quad_sum`` (the four lanes sharing a row then hold identical statistics),
    and the O rescale scales the two rows with their own alphas.
    """

    def reduce_max(tensor: str, init: str) -> str:
        if cfg.packed_reduce:
            return f"_helion_flash_rt.fmax_reduce_packed({tensor}, {init})"
        return f"{tensor}.load().reduce(cute.ReductionOp.MAX, {init}, 0)"

    def reduce_sum(tensor: str) -> str:
        if cfg.packed_reduce:
            return f"_helion_flash_rt.fadd_reduce_packed({tensor})"
        return f"{tensor}.load().reduce(cute.ReductionOp.ADD, 0.0, 0)"

    def exp2_rows(tensor: str, minus_max_scale: str) -> str:
        if cfg.exp2_impl == "split":
            return f"""            _helion_flash_rt.exp2_split_inplace(
                {tensor}, _flash_scale_log2, {minus_max_scale}, {cfg.e2e_freq}, {cfg.e2e_res})"""
        return f"""            for flash_j in cutlass.range_constexpr(cute.size({tensor})):
                {tensor}[flash_j] = cute.math.exp2(
                    {tensor}[flash_j] * _flash_scale_log2 + {minus_max_scale}, fastmath=True)"""

    def alpha_row(row: str) -> str:
        if cfg.rescale_threshold > 0.0:
            return f"""            flash_acc_log_{row} = _flash_scale_log2 * (flash_old_row_max_{row} - flash_row_max_safe_{row})
            flash_alpha_{row} = cute.math.exp2(flash_acc_log_{row}, fastmath=True)
            if (flash_o_started) & (flash_acc_log_{row} >= -{cfg.rescale_threshold}):
                flash_row_max_{row} = flash_old_row_max_{row}
                flash_row_max_safe_{row} = flash_old_row_max_{row}
                flash_alpha_{row} = cutlass.Float32(1.0)
            flash_minus_max_scale_{row} = (0.0 - flash_row_max_safe_{row}) * _flash_scale_log2"""
        return f"""            flash_alpha_{row} = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max_{row} - flash_row_max_safe_{row}), fastmath=True)
            flash_minus_max_scale_{row} = (0.0 - flash_row_max_safe_{row}) * _flash_scale_log2"""

    rescale_call = f"""_helion_flash_rt.rescale_o_tmem_m64(
                    tOtO, flash_alpha_a, flash_alpha_b, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})"""
    if cfg.rescale_threshold > 0.0:
        rescale_block = f"""            if flash_o_started:
                flash_o_full = flash_mma_o_cons.wait_and_advance()
                if cute.arch.vote_ballot_sync((flash_alpha_a < 1.0) | (flash_alpha_b < 1.0)) != 0:
                    {rescale_call}
                    cute.arch.fence_view_async_tmem_store()
                flash_o_full.release()"""
    else:
        rescale_block = f"""            if flash_o_started:
                flash_o_full = flash_mma_o_cons.wait_and_advance()
                {rescale_call}
                cute.arch.fence_view_async_tmem_store()
                flash_o_full.release()"""
    return f"""
            flash_s_full = flash_mma_s_cons.wait_and_advance()
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            if (flash_s_idx % 2) == 0:
                cute.copy(flash_tiled_ld0, tLDtS0, tLDrS)
            else:
                cute.copy(flash_tiled_ld1, tLDtS1, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}
            tLDrS_a, tLDrS_b = _helion_flash_rt.ws_m64_row_views(tLDrS)

            flash_old_row_max_a = flash_row_max_a
            flash_old_row_max_b = flash_row_max_b
            flash_row_max_a = _helion_flash_rt.quad_max({reduce_max("tLDrS_a", "flash_row_max_a")})
            flash_row_max_b = _helion_flash_rt.quad_max({reduce_max("tLDrS_b", "flash_row_max_b")})
            flash_row_max_safe_a = flash_row_max_a
            if flash_row_max_a == -cutlass.Float32.inf:
                flash_row_max_safe_a = cutlass.Float32(0.0)
            flash_row_max_safe_b = flash_row_max_b
            if flash_row_max_b == -cutlass.Float32.inf:
                flash_row_max_safe_b = cutlass.Float32(0.0)
{alpha_row("a")}
{alpha_row("b")}

{exp2_rows("tLDrS_a", "flash_minus_max_scale_a")}
{exp2_rows("tLDrS_b", "flash_minus_max_scale_b")}
            flash_p_sum_a = _helion_flash_rt.quad_sum({reduce_sum("tLDrS_a")})
            flash_p_sum_b = _helion_flash_rt.quad_sum({reduce_sum("tLDrS_b")})

            tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
            tSTrS_e = cute.make_tensor(
                cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrS_e.store(tLDrS.load().to({io_dtype}))

            flash_row_sum_a = flash_row_sum_a * flash_alpha_a + flash_p_sum_a
            flash_row_sum_b = flash_row_sum_b * flash_alpha_b + flash_p_sum_b

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
    row_epilogue: FlashRowEpilogueEmit | None = None,
) -> list[ast.stmt]:
    """Stage-4 device body: 2-warpgroup producer/consumer split with a
    double-buffered S TMEM accumulator (S0 @ col 0, S1 @ col 128, O @ col 256).

    Launch is ``block=(256, 1, 1)``: warp 0 = producer (Q/K/V TMA ring + both
    tcgen05 MMAs), warp 1 = TMEM owner (allocates at entry, frees at the end),
    warps 2-3 empty, warps 4-7 = consumer warpgroup (softmax + O rescale +
    epilogue). The producer runs QK(k+1) while the consumer runs softmax(k) ->
    the QK MMA overlaps the softmax (fills the `wait` bubble).
    """
    if score_plan.is_causal:
        assert not cfg.persistent
    kv_stage = cfg.kv_stage
    hd = head_dim
    persistent = cfg.persistent
    staged_store = cfg.epi_stg
    early_teardown = staged_store and not persistent
    # Staged O epilogue: a dedicated swizzled ``sO`` tile (``_flash_osl``, one
    # stage) plus a 128-thread 16-byte universal copy that drains it row-major
    # to gmem (each warp instruction writes 512 contiguous bytes).
    aux_smem_staging = _flash_aux_smem_staging(
        cfg, row_epilogue, io_dtype, tensor_4d=False
    )
    setup_smem_o = (
        "\nsO_stages = storage.sO.get_tensor(_flash_osl.outer, swizzle=_flash_osl.inner)"
        "\nsO = sO_stages[None, None, 0]"
        "\nflash_aux_full_ptr = storage.aux_full_mbar.data_ptr()"
        if aux_smem_staging
        else "\nsO = storage.sO.get_tensor(_flash_osl.outer, swizzle=_flash_osl.inner)[None, None, 0]"
        if staged_store
        else ""
    )
    # The fused row epilogue's aux tile: warp 0 TMA-loads it into sO right
    # after the prologue Q/K/V loads (flat) or at the head of every producer
    # tile (persistent; the per-tile CTA barrier has retired the previous
    # tile's drain); the consumer waits on ``aux_full`` before reading it.
    setup_aux_tma = ""
    aux_issue_lines: list[str] = []
    setup_aux_mbar = (
        "\nif tidx == 0:\n    cute.arch.mbarrier_init(flash_aux_full_ptr, 1)"
        if aux_smem_staging
        else ""
    )
    if aux_smem_staging:
        aux_tx_bytes = 128 * hd * _flash_dtype_bytes(io_dtype)
        setup_aux_tma = f"""
gA_aux_tma = cute.flat_divide(_flash_mEpiAux0t, cute.select((128, {hd}, 128), mode=[0, 1]))
tAgA_aux_mma = flash_pvt.partition_C(gA_aux_tma)
tAsA_aux_tma, tAgA_aux_tma = cute_cpasync_flash.tma_partition(
    _flash_tma_aux0, 0, cute.make_layout(1),
    cute.group_modes(sO_stages, 0, 2), cute.group_modes(tAgA_aux_mma, 0, 3))"""
        aux_issue_lines = [
            "with cute.arch.elect_one():",
            f"    cute.arch.mbarrier_arrive_and_expect_tx(flash_aux_full_ptr, {aux_tx_bytes})",
            "cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, flash_m_tile, 0, flash_bh],",
            "          tAsA_aux_tma[None, 0], tma_bar_ptr=flash_aux_full_ptr)",
        ]
    setup_o_store = (
        f"""
flash_o_store_elems = 128 // {io_dtype}.width
flash_o_tO_dim1 = {hd} // flash_o_store_elems
flash_o_tO_layout = cute.make_ordered_layout(
    (128 // flash_o_tO_dim1, flash_o_tO_dim1), order=(1, 0))
flash_o_vO_layout = cute.make_layout((1, flash_o_store_elems))
flash_o_gmem_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), {io_dtype}, num_bits_per_copy=128)
flash_o_gmem_tiled_copy = cute.make_tiled_copy_tv(
    flash_o_gmem_atom, flash_o_tO_layout, flash_o_vO_layout)
flash_o_gmem_thr = flash_o_gmem_tiled_copy.get_slice(flash_local_tidx)
tOsO_epi = flash_o_gmem_thr.partition_S(sO)"""
        if staged_store
        else ""
    )
    # Flat grid: warp 0 starts the K/Q TMA loads as soon as their pipelines
    # exist and the V loads once the V pipeline does, all BEFORE the TMEM
    # allocation barrier, so the first MMA's operands are in flight while the
    # CTA finishes its prologue. The persistent loop issues them at the head
    # of each producer tile.
    setup_active_range = (
        "" if persistent else _flash_ws_early_active_range(score_plan, num_kv)
    )
    setup_k_loads = (
        ""
        if persistent
        else _flash_ws_early_prologue_loads(
            num_kv, kv_stage, score_plan=score_plan, tensors="k"
        )
    )
    setup_v_loads = (
        ""
        if persistent
        else _flash_ws_early_prologue_loads(
            num_kv, kv_stage, score_plan=score_plan, tensors="v"
        )
    )
    if aux_issue_lines and not persistent:
        setup_v_loads += "\n" + "\n".join(f"    {line}" for line in aux_issue_lines)
    q_rows = cfg.q_tile_m
    two_rows = q_rows == 64
    assert not two_rows or (staged_store and not persistent)
    one_pass = cfg.ws_one_pass
    dense_tiles = (
        _flash_contiguous_kv_tile_range(score_plan, num_kv) is None
        and _flash_kv_tile_active_condition(
            score_plan,
            bh_expr="flash_bh",
            m_tile_expr="flash_m_tile",
            kv_tile_expr="cutlass.Int32(0)",
        )
        is None
    )
    if one_pass:
        assert two_rows and num_kv == 2 and kv_stage >= 2 and dense_tiles
        assert not score_plan.modifiers and row_epilogue is None
    if not has_lse:
        ws_lse_store = ""
    elif two_rows:
        # Two rows per consumer thread (r and r+8 of its warp's 16); the four
        # lanes of a quad hold identical statistics and store the same value.
        ws_lse_store = _flash_lse_store(
            "flash_m_tile * 64 + cutlass.Int32(tLDcS[0][0])",
            "flash_bh",
            "        ",
            lse_scale=score_plan.lse_scale,
            row_max="flash_row_max_a",
            row_sum="flash_row_sum_a",
        ) + _flash_lse_store(
            "flash_m_tile * 64 + cutlass.Int32(tLDcS[2][0])",
            "flash_bh",
            "        ",
            lse_scale=score_plan.lse_scale,
            row_max="flash_row_max_b",
            row_sum="flash_row_sum_b",
        )
    else:
        ws_lse_store = _flash_lse_store(
            "flash_m_tile * 128 + flash_local_tidx",
            "flash_bh",
            "        ",
            lse_scale=score_plan.lse_scale,
        )
    # TMEM copy atoms over the score/probability tiles. M=128 accumulators read
    # one row per thread (32x32b); M=64 accumulators keep 16 rows per lane
    # quadrant, so 16x256b loads and 16x128b stores hand each thread two rows.
    if two_rows:
        ld_atom = "cute_tcgen05_flash.Ld16x256bOp(cute_tcgen05_flash.Repetition(16))"
        st_atom = "cute_tcgen05_flash.St16x128bOp(cute_tcgen05_flash.Repetition(16))"
    else:
        ld_atom = "cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32))"
        st_atom = "cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(32))"
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
    setup_qk_gmem_slice = (
        ""
        if persistent
        else """
tQgQ = tQgQ_qdl[None, None, 0, flash_bh]
tKgK = tKgK_kdl[None, None, 0, flash_bh]"""
    )
    setup_v_gmem_slice = (
        ""
        if persistent
        else """
tVgV = tVgV_dkl[None, 0, None, flash_bh]"""
    )
    setup = f"""
tidx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()){setup_pid}
flash_local_tidx = tidx % 128
# Start the TMA descriptor fetches first: they overlap the smem/mbarrier setup
# instead of sitting in front of the first Q/K/V load.
if warp_idx == 0:
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_q)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_k)
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_v)

_flash_storage_cls = _helion_flash_rt.flash_shared_storage(
    {hd}, {kv_stage}, 2, {io_dtype}, o_stage={1 if staged_store else 0}, q_rows={q_rows})
smem = cutlass_utils_flash.SmemAllocator()
storage = smem.allocate(_flash_storage_cls)
sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)
sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)
sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner){setup_smem_o}

# TMEM allocator: the idle warp {_FLASH_WS_TMEM_WARP} allocates (and hands the permit back,
# so co-resident CTAs can allocate) after the shared setup. The allocation is
# placed in program order behind warp 0's issue blocks because a tcgen05.alloc
# running next to the mbarrier inits measured slower; nothing orders the two
# (warp 1 skips those blocks), so the placement is a heuristic, not a
# guarantee. All 256 threads wait_for_alloc (NamedBarrier id=1, 256 threads)
# before retrieving the address. That barrier is the ONE CTA-wide sync of the
# prologue: every pipeline below is created with ``defer_sync`` (warp 0
# initializes the mbarriers; a fence publishes them to the async proxy)
# instead of a fence + __syncthreads per pipeline. The Q/K TMA pipelines and
# partitions come first so the flat grid issues the first MMA's operands
# before the V pipeline, the TMEM allocation and the MMA/softmax handshake
# barriers even exist; the V loads follow as soon as their pipeline does.
flash_tmem_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=1, num_threads=256)
flash_tmem = cutlass_utils_flash.TmemAllocator(
    storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_bar,
    allocator_warp_id={_FLASH_WS_TMEM_WARP})

flash_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2]))
flash_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2]))
flash_q_prod, flash_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants()
flash_k_prod, flash_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants()
cute.arch.mbarrier_init_fence()

flash_qkt = _flash_qk_mma.get_slice(0)
tSrQ = flash_qkt.make_fragment_A(sQ)
tSrK = flash_qkt.make_fragment_B(sK)
flash_qk_acc_shape = flash_qkt.partition_shape_C(({q_rows}, 128))
tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
gQ = cute.flat_divide(_flash_mQt, cute.select(({q_rows}, 128, {hd}), mode=[0, 2]))
gK = cute.flat_divide(_flash_mKt, cute.select((128, 128, {hd}), mode=[1, 2]))
tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3)){setup_qk_gmem_slice}
{setup_active_range}{setup_k_loads}
flash_v_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2]))
flash_v_prod, flash_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(
    num_stages={kv_stage},
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    tx_count=flash_v_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr(),
    defer_sync=True).make_participants(){setup_aux_mbar}
cute.arch.mbarrier_init_fence()

flash_pvt = _flash_pv_mma.get_slice(0)
tOrV = flash_pvt.make_fragment_B(sV)
flash_pv_acc_shape = flash_pvt.partition_shape_C(({q_rows}, {hd}))
tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)
gV = cute.flat_divide(_flash_mVt, cute.select((128, {hd}, 128), mode=[1, 2]))
tOgV = flash_pvt.partition_B(gV)
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3)){setup_v_gmem_slice}{setup_aux_tma}
{setup_v_loads}
flash_tmem.allocate(512)
if warp_idx == {_FLASH_WS_TMEM_WARP}:
    flash_tmem.relinquish_alloc_permit()
# mma_s: QK UMMA (warp 0) -> softmax (consumer WG, 128 threads). 2 stages = the
# 2 S buffers, so QK(k+1) overlaps softmax(k).
flash_mma_s_prod, flash_mma_s_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_s_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
flash_mma_o_prod, flash_mma_o_cons = cutlass_pipeline_flash.PipelineUmmaAsync.create(
    num_stages=1,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    barrier_storage=storage.mma_o_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
# p_ready: consumer WG (128 threads) -> warp 0. PipelineAsync has NO elect-one,
# so all 32 threads of warp 0 arrive on release() -> the consumer group MUST be a
# FULL WARP (32), not 1 (Bug K: a 1-thread group is over-arrived -> illegal
# barrier arrive, launch fail 719).
flash_p_ready_prod, flash_p_ready_cons = cutlass_pipeline_flash.PipelineAsync.create(
    num_stages=2,
    producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 128),
    consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, 32),
    barrier_storage=storage.p_ready_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
cute.arch.mbarrier_init_fence()
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

cS = cute.make_identity_tensor(({q_rows}, 128))
tScS = flash_qkt.partition_C(cS)
# Per-stage t2r/r2t tiled copies built over the ALREADY-OFFSET S tensor (the
# lane->column map is baked at make_tmem_copy time; partition_S over a different
# column of a col-0-built copy targets the wrong columns).
flash_ld_atom = cute.make_copy_atom({ld_atom}, cutlass.Float32)
flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
tLDtS0 = flash_thr_ld0.partition_S(tStS0)
tLDtS1 = flash_thr_ld1.partition_S(tStS1)
tLDcS = flash_thr_ld0.partition_D(tScS)

flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width
flash_P_layout = cute.composition(flash_s_layout, cute.make_layout(({q_rows}, flash_tilePlikeFP32)))
tStS_P0 = cute.make_tensor(flash_tmem_ptr, flash_P_layout)
tStS_P1 = cute.make_tensor(flash_tmem_ptr + 128, flash_P_layout)
flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout(({q_rows}, flash_tilePlikeFP32)))
tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
flash_st_atom = cute.make_copy_atom({st_atom}, cutlass.Float32)
flash_tiled_st0 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P0)
flash_tiled_st1 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P1)
flash_thr_st0 = flash_tiled_st0.get_slice(flash_local_tidx)
flash_thr_st1 = flash_tiled_st1.get_slice(flash_local_tidx)
tSTtS0 = flash_thr_st0.partition_D(tStS_P0)
tSTtS1 = flash_thr_st1.partition_D(tStS_P1)
tSTcS = flash_thr_st0.partition_S(tScS_P){setup_o_store}
"""
    # No acc pipeline in the 2-WG path: the consumer's final epilogue read is
    # ordered by the last PV's mma_o, and the teardown barrier the consumer
    # joins after that read guarantees all reads precede warp 1's TMEM free
    # (matches the validated standalone spike, which has no acc pipeline).
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
    producer_body = _flash_ws_producer_body(
        num_kv,
        kv_stage,
        hd,
        score_plan=score_plan,
        early_issue=not persistent,
        one_pass=one_pass,
    )
    if aux_issue_lines and persistent:
        producer_body = (
            "\n".join(f"        {line}" for line in aux_issue_lines)
            + "\n"
            + producer_body
        )
    producer = _flash_persistent_wrap("if warp_idx == 0:", producer_body, persistent)
    consumer = _flash_persistent_wrap(
        "if warp_idx >= 4:",
        _flash_ws_consumer_body(
            hd,
            num_kv,
            cfg,
            io_dtype=io_dtype,
            score_plan=score_plan,
            output_epilogue=output_epilogue,
            row_epilogue=row_epilogue,
            staged_store=staged_store,
            early_teardown=early_teardown,
            lse_store=ws_lse_store,
            two_rows=two_rows,
            aux_smem_staging=aux_smem_staging,
            one_pass=one_pass,
        ),
        persistent,
        head="    flash_aux_full_phase = cutlass.Int32(0)" if aux_smem_staging else "",
    )
    if early_teardown:
        # Flat staged grid: the consumer warpgroup joins the teardown barrier
        # right after its last TMEM read (before the gmem drain) and warp 1
        # frees TMEM while the output stores are still in flight; warp 0 and
        # the two idle warps simply exit after their roles.
        teardown = f"""
if warp_idx == {_FLASH_WS_TMEM_WARP}:
    cute.arch.barrier(barrier_id={_FLASH_WS_TEARDOWN_BARRIER_ID}, number_of_threads={_FLASH_WS_TEARDOWN_THREADS})
    flash_tmem.free(flash_tmem_ptr)
"""
        src = setmaxreg + teardown + producer + "\n" + consumer
    else:
        # Teardown: all 256 threads converge on a plain CTA barrier (matches
        # the validated spike -- a pipeline.sync over barrier_id=1 collides
        # with the TMEM-allocator NamedBarrier id=1 and trips synccheck
        # "Missing init"), then the TMEM owner frees TMEM.
        teardown = f"""
cute.arch.barrier()
if warp_idx == {_FLASH_WS_TMEM_WARP}:
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
    staged_wait: str = "",
) -> str:
    if first_load_order == FLASH_FIRST_LOAD_ORDER_STAGED:
        return f"{load_k0}\n{load_q0}\n{staged_wait}{load_q1}\n{load_v0}"
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
    row_epilogue: FlashRowEpilogueEmit | None = None,
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
    use_tmem_row_reduce = (
        cfg.rowmax == "tmem"
        and hardware_capabilities.supports_tmem_row_reduce
        and cfg.s_load_repetition == 32
        and flash_tmem_rowmax_score_plan_supported(score_plan)
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
    # KV tile width for the score/probability tile. The resolver already
    # clamped an illegal request back to the historical 128. A width that does
    # not divide the sequence leaves a partial trailing tile whose out-of-range
    # columns must be masked to -inf before the row max.
    kv_n = cfg.kv_tile_n
    # ``num_kv`` counts default-width tiles and is the shape key the tuning
    # policy is indexed by; the actual trip count follows the configured width.
    kv_tile_count = -(-sequence_extent // kv_n)
    kv_tail_cols = sequence_extent - (kv_tile_count - 1) * kv_n
    assert 0 < kv_tail_cols <= kv_n
    has_kv_tail = kv_tail_cols != kv_n
    dense_tuning = (
        tuning_policy.dense_policy(num_kv)
        if cfg.softmax_lowering == "auto" and tuning_policy is not None
        else None
    )
    probability_log2_shift = (
        dense_tuning.probability_log2_shift if dense_tuning is not None else 0
    )
    dense_lowering_schedule_ok = _flash_dense_lowering_schedule_supported(
        cfg, dense_tuning
    )
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
        cfg.softmax_lowering == "auto"
        and dense_tuning is not None
        and dense_lowering_schedule_ok
        and not is_causal
        and not has_lse
        # Both bodies read the score tile through the whole-row TMEM reduction.
        and use_tmem_row_reduce
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
    explicit_resident_value_graph = cfg.softmax_lowering == "resident_value_graph"
    if explicit_resident_value_graph:
        # The full-row load/P-store views and acknowledged statistics graph are
        # shared by all FA4 families, including their persistent wrappers. The
        # helper changes only the per-KV value graph inside that protocol.
        assert not is_causal and not has_lse
        assert io_dtype in ("cutlass.Float16", "cutlass.BFloat16")
        assert score_plan.modifier_kinds == (DENSE_SCORE_KIND,)
        assert cfg.q_tile_count == 2 and not cfg.softmax_disc
        assert cfg.p_store_repetition == 16 and cfg.s_load_repetition == 32
        assert cfg.split_p_arrive and cfg.stat_transport == "single"
        assert cfg.sp_row_sum == "whole" and cfg.exp2_impl == "xu"
        dense_resident_value_graph_candidate = True
    causal_tuning = (
        tuning_policy.causal_policy(num_kv)
        if cfg.softmax_lowering == "auto" and tuning_policy is not None
        else None
    )
    causal_schedule_ok = _flash_causal_resident_schedule_supported(cfg, causal_tuning)
    use_causal_resident_native = (
        cfg.softmax_lowering == "auto"
        and causal_tuning is not None
        and causal_schedule_ok
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
    explicit_stateful = cfg.softmax_lowering == "resident_stateful"
    if explicit_stateful:
        # Recheck the actual emitted tile/mask proof; detector metadata alone
        # must not authorize an unmasked load in an incompatible source graph.
        assert is_causal and not has_lse
        assert io_dtype in ("cutlass.Float16", "cutlass.BFloat16")
        assert score_plan.modifier_kinds == (CAUSAL_MASK_KIND,)
        assert cfg.pipeline_family == "fa4" and not cfg.persistent
        assert not cfg.use_2cta_instrs and not cfg.separate_kv_rings
        assert not cfg.use_cga2_local_cta and not cfg.use_clc_scheduler
        assert cfg.q_tile_count == 2 and cfg.kv_tile_n == 128 and cfg.mma_ptx
        assert causal_split_proof.proven and causal_split_equal_iteration_proof.proven
        assert cfg.causal_loop_split and causal_desc_kv
        assert cfg.p_store_repetition == 16 and cfg.s_load_repetition == 32
        assert cfg.split_p_arrive and not cfg.softmax_disc
        assert cfg.stat_transport == "single" and cfg.sp_row_sum == "whole"
        assert cfg.exp2_impl == "xu" and cfg.e2e_offset == cfg.e2e_offset0 == 0
        use_causal_resident_native = True
        use_causal_stateful_softmax = True
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
        0
        if explicit_resident_value_graph
        else _flash_fitted_probability_log2_shift(
            probability_log2_shift, cfg.rescale_threshold
        )
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
        and (not is_causal or (use_causal_resident_native and cfg.exp2_impl == "xu"))
        and not cfg.softmax_disc
        and (
            cfg.rescale_threshold > 0.0
            or explicit_resident_value_graph
            or explicit_stateful
        )
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
                    p_chunk_arrive=cfg.p_chunk_arrive,
                    stat_depth=1 if fa4_stat_handoff else 2,
                    pipelined_stat_handoff=acknowledged_stat_pipeline,
                    final_only_stat_handoff=final_only_stat_pipeline,
                    kv_tile_n=kv_n,
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
                    p_chunk_arrive=cfg.p_chunk_arrive,
                    stat_depth=1 if fa4_stat_handoff else 2,
                    pipelined_stat_handoff=acknowledged_stat_pipeline,
                    final_only_stat_handoff=final_only_stat_pipeline,
                    kv_tile_n=kv_n,
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
                    p_chunk_arrive=cfg.p_chunk_arrive,
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
    # A fused row epilogue evaluated by the softmax warpgroups (one per Q tile,
    # 200 registers each, both tiles in parallel) instead of the correction
    # warpgroup (64 registers, tiles in sequence, and the next tile's rescales
    # queue behind it).
    softmax_row_epilogue = (
        row_epilogue is not None and cfg.row_epilogue_warps == "softmax"
    )
    aux_smem_staging = _flash_aux_smem_staging(
        cfg, row_epilogue, io_dtype, tensor_4d=use_tensor_4d_tma
    )
    role_chain = cfg.role_chain
    storage_extra_args = f", {epi_smem!s}, {use_clc_scheduler!s}, {cfg.clc_stages}"
    storage_extra_args += f", {separate_kv_rings!s}, {kv_n}, {cta_group_size}"
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
gQ = cute.flat_divide(_flash_mQt, cute.select(({mma_m}, {kv_n}, {hd}), mode=[0, 2]))
gK = cute.flat_divide(_flash_mKt, cute.select(({mma_m}, {kv_n}, {hd}), mode=[1, 2]))
gV = cute.flat_divide(_flash_mVt, cute.select(({mma_m}, {hd}, {kv_n}), mode=[1, 2]))
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
    # A warp publishes/consumes 32 adjacent rows. Keep rows contiguous so the
    # ring and query slots do not multiply the lane stride and alias SMEM banks.
    scale_layout = (
        f"cute.make_layout(({q_stage} * 128))"
        if fa4_stat_handoff
        else (
            f"cute.make_layout(({s_corr_stage}, {q_stage}, 128), "
            f"stride=({q_stage * 128}, 128, 1))"
        )
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
    # The aux tile's TMA barrier exists only when the softmax route stages it.
    aux_full_ptr_setup = (
        "flash_aux_full_ptr = storage.aux_full_mbar.data_ptr()\n"
        if aux_smem_staging
        else ""
    )
    aux_full_init = (
        "        cute.arch.mbarrier_init(flash_aux_full_ptr + flash_st, 1)\n"
        if aux_smem_staging
        else ""
    )
    staged_first_load_setup = cfg.first_load_order == FLASH_FIRST_LOAD_ORDER_STAGED
    prologue_ptr_setup = (
        "\nflash_prologue_ptr = storage.prologue_mbar.data_ptr()"
        if staged_first_load_setup
        else ""
    )
    prologue_init = (
        "\n    cute.arch.mbarrier_init(flash_prologue_ptr, 1)"
        if staged_first_load_setup
        else ""
    )
    # Per-chunk staged-P release: the middle chunks' barriers (two per Q slot).
    pforc_ptr_setup = (
        "\nflash_pforc_ptr = storage.pforc_mbar.data_ptr()"
        if cfg.p_chunk_arrive
        else ""
    )
    pforc_init = (
        f"\n        cute.arch.mbarrier_init(flash_pforc_ptr + 2 * flash_st, {pfor2_count})"
        f"\n        cute.arch.mbarrier_init(flash_pforc_ptr + 2 * flash_st + 1, {pfor2_count})"
        if cfg.p_chunk_arrive
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
flash_pfor2_ptr = storage.pfor2_mbar.data_ptr(){pforc_ptr_setup}
flash_o_full_ptr = storage.o_full_mbar.data_ptr()
flash_corr_epi_full_ptr = storage.corr_epi_mbar_ptr.data_ptr()
flash_corr_epi_empty_ptr = flash_corr_epi_full_ptr + 2
{aux_full_ptr_setup}flash_s0_corr_full_ptr = storage.s0_corr_mbar_ptr.data_ptr()
flash_s0_corr_empty_ptr = flash_s0_corr_full_ptr + {s_corr_stage}
flash_s1_corr_full_ptr = storage.s1_corr_mbar_ptr.data_ptr()
flash_s1_corr_empty_ptr = flash_s1_corr_full_ptr + {s_corr_stage}
flash_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr(){prologue_ptr_setup}
if tidx == 0:{prologue_init}
    for flash_st in cutlass.range_constexpr(2):
        cute.arch.mbarrier_init(flash_s_full_ptr + flash_st, 1)
        cute.arch.mbarrier_init(flash_o_full_ptr + flash_st, 1)
        cute.arch.mbarrier_init(flash_pfor_ptr + flash_st, {pfor_count})
        cute.arch.mbarrier_init(flash_pfor2_ptr + flash_st, {pfor2_count}){pforc_init}
        cute.arch.mbarrier_init(flash_corr_epi_full_ptr + flash_st, 128)
        cute.arch.mbarrier_init(flash_corr_epi_empty_ptr + flash_st, 1)
{aux_full_init}    for flash_st in cutlass.range_constexpr({s_corr_stage}):
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

    tmem_fragment_setup = f"""    flash_qk_acc_shape = flash_qkt.partition_shape_C(({mma_m}, {kv_n}))
    tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
    flash_pv_acc_shape = flash_pvt.partition_shape_C(({mma_m}, {hd}))
    tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)
"""
    tmem_base_setup = f"""{tmem_fragment_setup}    _helion_flash_rt.named_barrier_wait_unaligned(
        2, 13 * 32)
    flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)
    tStS0_full = cute.make_tensor(flash_tmem_ptr, tStS.layout)
    tStS1_full = cute.make_tensor(flash_tmem_ptr + {kv_n}, tStS.layout)
    tOtO0_full = cute.make_tensor(flash_tmem_ptr + {2 * kv_n}, tOtO.layout)
    tOtO1_full = cute.make_tensor(flash_tmem_ptr + {2 * kv_n + hd}, tOtO.layout)
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
        # The modified scores are written back chunk by chunk with the load
        # partition's chunk index, so the store atom must repeat the same
        # number of columns as the load atom (Rep16 -> eight 16-column chunks).
        score_store_setup = f"""    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({
            cfg.s_load_repetition
        })), cutlass.Float32)
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
        + f"""    cS = cute.make_identity_tensor((128, {kv_n}))
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
    flash_tilePlikeFP32 = {kv_n} // cutlass.Float32.width * {io_dtype}.width
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
        ptr_expr = "flash_tmem_ptr" if stage == "0" else f"flash_tmem_ptr + {kv_n}"
        p_store_iter = p0_store_iter if stage == "0" else p1_store_iter
        stage_score_store_setup = ""
        if score_store_needed:
            stage_score_store_setup = f"""    flash_score_st_atom = cute.make_copy_atom(
        cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition({
                cfg.s_load_repetition
            })), cutlass.Float32)
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
    flash_qk_acc_shape = flash_qkt.partition_shape_C(({mma_m}, {kv_n}))
    tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)
    tStS{stage} = cute.make_tensor({ptr_expr}, tStS.layout)
    cS = cute.make_identity_tensor((128, {kv_n}))
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
    flash_tilePlikeFP32 = {kv_n} // cutlass.Float32.width * {io_dtype}.width
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

    # The staged aux tile: the epilogue warp TMA-loads the row program's aux
    # rows for this work item into both sO stages as soon as the previous
    # work item's output drained (the top of its per-work-item body), and the
    # softmax warpgroups wait on ``aux_full`` before reading them.
    epi_aux_setup = ""
    epi_aux_issue = ""
    if aux_smem_staging:
        aux_tx_bytes = 128 * hd * _flash_dtype_bytes(io_dtype)
        # The first work item's aux loads follow its first QK so the start-up
        # Q/K(0) TMA traffic is not slowed by rows that are needed only at that
        # tile's epilogue (2 us on the XSA 2x16x2048x128 winner); later work
        # items issue theirs under the mainloop. ``s_full[0]`` completes once
        # per KV step, driven by roles the epilogue warp does not pace, so a
        # plain parity-0 wait would alias if that barrier completed twice
        # before the probe and could never return for a work item with two KV
        # steps. The wait is therefore bounded (about 256 us, two orders of
        # magnitude above the first QK's completion): it only paces the loads.
        epi_aux_setup = f"""
    cute_cpasync_flash.prefetch_descriptor(_flash_tma_aux0)
    gA_aux_tma = cute.flat_divide(_flash_mEpiAux0t, cute.select((128, {hd}, 128), mode=[0, 1]))
    tAgA_aux_mma = flash_pvt.partition_C(gA_aux_tma)
    tAsA_aux_tma, tAgA_aux_tma = cute_cpasync_flash.tma_partition(
        _flash_tma_aux0, 0, cute.make_layout(1),
        cute.group_modes(sO, 0, 2), cute.group_modes(tAgA_aux_mma, 0, 3))
    _helion_flash_rt.mbar_spin_wait_bounded(flash_s_full_ptr + 0, cutlass.Int32(0), 256, 1000)"""
        # One lane posts the transaction bytes; the TMA copies are issued at
        # warp scope like every other flash TMA load (the DSL elects the
        # issuing lane itself).
        epi_aux_issue = f"""        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(flash_aux_full_ptr + 0, {aux_tx_bytes})
            cute.arch.mbarrier_arrive_and_expect_tx(flash_aux_full_ptr + 1, {aux_tx_bytes})
        cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, {output_m_tile0}, 0, flash_bh],
                  tAsA_aux_tma[None, 0], tma_bar_ptr=flash_aux_full_ptr + 0)
        cute.copy(_flash_tma_aux0, tAgA_aux_tma[None, {output_m_tile1}, 0, flash_bh],
                  tAsA_aux_tma[None, 1], tma_bar_ptr=flash_aux_full_ptr + 1)
"""
    if cfg.epi_tma:
        epi_tma_release_corr_empty = """        with cute.arch.elect_one():
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"""
        epi_head = f"""    cute.arch.setmaxregister_decrease({cfg.other_regs})
{textwrap.indent(epi_tma_setup.strip(), "    ") if local_epi_tma_setup else ""}{epi_aux_setup}
    flash_corr_epi_full_phase = cutlass.Int32(0){epi_empty_prearrive}"""
        epi_inner = f"""{epi_aux_issue}        {_epi_wait_corr_full("0")}
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
{textwrap.indent(epi_stg_setup.strip(), "    ") if cfg.epi_stg else ""}{epi_aux_setup}
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
            f"""{epi_aux_issue}        flash_mO_cur = {epi_stg_mO_cur}
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
        flash_gK = cute.local_tile(flash_mK_cur, ({kv_n}, {hd}), (None, 0))
        flash_gV = cute.local_tile(flash_mV_cur, ({hd}, {kv_n}), (0, None))
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
    staged_first_load = cfg.first_load_order == FLASH_FIRST_LOAD_ORDER_STAGED
    if staged_first_load:
        # First work item only: hold Q1/V0 (and the K/V ring behind them) until
        # the MMA warp has seen K0 and Q0 land, so the kernel-start HBM burst is
        # only the bytes the first score tile needs.
        load_head += "\n    flash_load_first = cutlass.Boolean(True)"
        staged_wait = f"""        if flash_load_first:
            _helion_flash_rt.mbar_spin_wait(
                flash_prologue_ptr, cutlass.Int32(0), {cfg.wait_hint})
        flash_load_first = cutlass.Boolean(False)
"""
    else:
        staged_wait = ""
    load_prologue = _flash_fa4_load_prologue_for_order(
        cfg.first_load_order, load_q0, load_k0, load_q1, load_v0, staged_wait
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
    mma_first_head = (
        "\n    flash_mma_first = cutlass.Boolean(True)" if staged_first_load else ""
    )
    if staged_first_load:
        # First work item: Q0 and K0 landed -> release the load warp's staged
        # Q1/V0 issue in this CTA and, for the two-CTA pair, in the peer.
        # The barrier counts one arrival per work item: one elected lane arrives.
        mma_prologue_signal = (
            "\n        if flash_mma_first:"
            "\n            with cute.arch.elect_one():"
            "\n                cute.arch.mbarrier_arrive(flash_prologue_ptr)"
        )
        if use_2cta_instrs:
            mma_prologue_signal += (
                "\n                _helion_flash_rt.mbarrier_arrive("
                "flash_prologue_ptr, cutlass.Int32(1), flash_mma_tile_coord_v)"
            )
        mma_prologue_signal += "\n        flash_mma_first = cutlass.Boolean(False)"
    else:
        mma_prologue_signal = ""
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
    flash_pfor_phase = cutlass.Int32(0){mma_first_head}
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
        if cfg.p_chunk_arrive:
            # One wait per K-chunk quarter: the middle quarters on this slot's
            # pforc pair, the last quarter on pfor2 (pfor gates the first, above).
            pv0_split_wait_arg += "\n                mbar_mid_ptrs=(flash_pforc_ptr + 0, flash_pforc_ptr + 1),"
            pv1_split_wait_arg += "\n                mbar_mid_ptrs=(flash_pforc_ptr + 2, flash_pforc_ptr + 3),"
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

        # PROLOGUE: QK0(0)->S0 as soon as Q0 and K0 landed, then QK1(0)->S1 once
        # Q1 landed too; then release K0. Waiting for Q1 only before QK1 lets the
        # first score tile start while the second Q tile is still in flight.
        flash_k0_full = {mma_k_cons}.wait_and_advance(){mma_prologue_signal}
{textwrap.indent(_qk_gemm("0", "flash_k0_full").lstrip(), "        ")}
{textwrap.indent(_mma_commit_s_ready("0"), "        ")}
        flash_q1_full = flash_q_cons.wait_and_advance()
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
    flash_pfor_phase = cutlass.Int32(0){mma_first_head}"""
        mma_inner = f"""        flash_q0_full = flash_q_cons.wait_and_advance()

        # PROLOGUE: QK0(0)->S0 as soon as Q0 and K0 landed, then QK1(0)->S1 once
        # Q1 landed too; then release K0.
        flash_k0_full = {mma_k_cons}.wait_and_advance(){mma_prologue_signal}
        for flash_kp in cutlass.range(flash_nk, unroll_full=True):
            _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
            cute.gemm(_flash_qk_mma, tStS0, tSrQ[None, None, flash_kp, flash_q0_full.index],
                      tSrK[None, None, flash_kp, flash_k0_full.index], tStS0)
        with cute.arch.elect_one():
            cute_tcgen05_flash.commit(flash_s_full_ptr + 0{commit_group_arg})
        flash_q1_full = flash_q_cons.wait_and_advance()
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
    if cfg.rescale_threshold > 0.0 or explicit_resident_value_graph:

        def _sp_alpha_pre_for(not_first: str) -> str:
            pin = f"({not_first}) & (flash_acc_log >= -{cfg.rescale_threshold})"
            return f"""            flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)
            flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)
            if {pin}:
                flash_row_max = flash_old_row_max
                flash_row_max_safe = flash_old_row_max
                flash_alpha = cutlass.Float32(1.0)
            flash_minus_max_scale = {sp_minus_max_scale}"""

        _sp_alpha_pre = _sp_alpha_pre_for(softmax_not_first)
        _sp_alpha_post = ""
    else:
        _sp_alpha_pre_for = None  # type: ignore[assignment]
        _sp_alpha_pre = f"            flash_minus_max_scale = {sp_minus_max_scale}"
        _sp_alpha_post = """            flash_alpha = cute.math.exp2(
                _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe), fastmath=True)"""

    def _row_epilogue_pieces(stage: str) -> tuple[str, str]:
        """(prologue, epilogue) program source for one Q-tile stage.

        Aux rows are addressed through ``flash_pvt.partition_C`` like the
        direct-store route, so they take the MMA-pair tile index even when the
        staged route indexes sO by the per-CTA tile.  The prologue (aux views
        plus passes that never read O) runs at the top of the per-work-item
        body, hidden behind the KV loop.
        """
        assert row_epilogue is not None
        aux_mtile = output_m_tile0 if stage == "0" else output_m_tile1
        aux_mma_index = corr_gmem_o_index.replace("flash_gO_corr", "{tensor}").format(
            tensor="{tensor}", mtile=aux_mtile
        )
        if not epi_smem:
            return _flash_row_epilogue_direct_source(
                row_epilogue,
                hd=hd,
                io_dtype=io_dtype,
                indent="        ",
                suffix=stage,
                mma_index=aux_mma_index,
                epi_tiler=f"flash_epi_tiler{stage}",
                thr_ld=f"flash_thr_o_ld{stage}",
                tiled_ld=f"flash_tiled_o_ld{stage}",
                tDtO=f"tDtO{stage}",
                tDgO=f"tDgO{stage}",
                inv_sum=f"flash_inv_sum{stage}",
                ld_repetition=16,
                split=True,
            )
        if aux_smem_staging:
            # The aux tile lands in sO by TMA after the KV loop starts, so every
            # pass (including the aux-only ones) runs after the aux_full wait;
            # only the copy-view algebra sits at the top of the work item.
            setup, prologue, epilogue = _flash_row_epilogue_smem_source_parts(
                row_epilogue,
                hd=hd,
                io_dtype=io_dtype,
                indent="        ",
                suffix=stage,
                mma_index=aux_mma_index,
                corr_tile_size=cfg.corr_tile_size,
                use_2cta_instrs=use_2cta_instrs,
                tOtO=f"tOtO{stage}",
                sO=f"sO[None, None, {stage}]",
                tidx="flash_local_tidx",
                inv_sum=f"flash_inv_sum{stage}",
                split=True,
                aux_from_smem=True,
            )
            row_hoisted[stage] = prologue
            return "\n".join(setup), epilogue
        return _flash_row_epilogue_smem_source(
            row_epilogue,
            hd=hd,
            io_dtype=io_dtype,
            indent="        ",
            suffix=stage,
            mma_index=aux_mma_index,
            corr_tile_size=cfg.corr_tile_size,
            use_2cta_instrs=use_2cta_instrs,
            tOtO=f"tOtO{stage}",
            sO=f"sO[None, None, {stage}]",
            tidx="flash_local_tidx",
            inv_sum=f"flash_inv_sum{stage}",
            split=True,
        )

    # Aux-only passes of the smem-staged route (run after the aux tile landed,
    # under the wait for the last PV); empty on the other routes.
    row_hoisted: dict[str, str] = {"0": "", "1": ""}
    row_pieces: dict[str, tuple[str, str]] = (
        {stage: _row_epilogue_pieces(stage) for stage in ("0", "1")}
        if row_epilogue is not None
        else {}
    )
    row_prologue = "\n".join(pieces[0] for pieces in row_pieces.values() if pieces[0])
    if row_prologue:
        row_prologue += "\n"

    def _softmax_row_epilogue(stage: str, row_sum_expr: str) -> str:
        """O passes of the fused row epilogue on the softmax warpgroup of ``stage``.

        Waits for the MMA's final PV (``o_full``) like the correction epilogue
        did, evaluates the program straight out of TMEM, and hands the staged
        sO tile to the epilogue warp.  The next tile's P store of this same
        warpgroup orders every TMEM read here before the MMA overwrites O.
        """
        wait_hint = cfg.wait_hint
        program_src = row_pieces[stage][1]
        lines = [
            f"        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({row_sum_expr})",
        ]
        if aux_smem_staging:
            # The aux tile landed long ago; its passes overlap the last PV.
            lines.append(
                f"        _helion_flash_rt.mbar_spin_wait(flash_aux_full_ptr + {stage}, flash_sm_aux_full_phase, {wait_hint})"
            )
            if row_hoisted[stage]:
                lines.append(row_hoisted[stage])
        lines.append(
            f"        _helion_flash_rt.mbar_spin_wait(flash_o_full_ptr + {stage}, flash_sm_o_full_phase, {wait_hint})"
        )
        if epi_smem:
            lines.append(
                f"        _helion_flash_rt.mbar_spin_wait(flash_corr_epi_empty_ptr + {stage}, flash_sm_epi_empty_phase, {wait_hint})"
            )
        else:
            mtile = output_m_tile0 if stage == "0" else output_m_tile1
            lines.extend(
                (
                    f"        tOgO_mma{stage} = {corr_gmem_o_index.format(mtile=mtile)}",
                    f"        gO_epi{stage} = cute.zipped_divide(tOgO_mma{stage}, flash_epi_tiler{stage})",
                    f"        tDgO{stage} = flash_thr_o_ld{stage}.partition_D(gO_epi{stage})",
                )
            )
        lines.extend((program_src, "        cute.arch.fence_view_async_tmem_load()"))
        if epi_smem:
            lines.extend(
                (
                    "        cute.arch.fence_view_async_shared()",
                    f"        cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr + {stage})",
                )
            )
        lines.append("        flash_sm_o_full_phase ^= 1")
        if epi_smem:
            lines.append("        flash_sm_epi_empty_phase ^= 1")
        if aux_smem_staging:
            lines.append("        flash_sm_aux_full_phase ^= 1")
        return "\n" + "\n".join(lines)

    def _softmax_row_epilogue_head(stage: str) -> str:
        """Cross-tile phases (and direct-store copy views) of the softmax epilogue."""
        if not softmax_row_epilogue:
            return ""
        head = "\n    flash_sm_o_full_phase = cutlass.Int32(0)"
        if aux_smem_staging:
            head += "\n    flash_sm_aux_full_phase = cutlass.Int32(0)"
        if epi_smem:
            return head + "\n    flash_sm_epi_empty_phase = cutlass.Int32(0)"
        return (
            head
            + f"""
    flash_gO_corr = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))
    flash_epi_tiler{stage} = ((cute.size(tOtO{stage}, mode=[0, 0]), cute.size(tOtO{stage}, mode=[0, 1])),)
    tOtO_epi{stage} = cute.zipped_divide(tOtO{stage}, flash_epi_tiler{stage})
    flash_o_ld_atom{stage} = cute.make_copy_atom(
        cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
    flash_tiled_o_ld{stage} = cute_tcgen05_flash.make_tmem_copy(flash_o_ld_atom{stage}, tOtO_epi{stage}[None, 0])
    flash_thr_o_ld{stage} = flash_tiled_o_ld{stage}.get_slice(flash_local_tidx)
    tDtO{stage} = flash_thr_o_ld{stage}.partition_S(tOtO_epi{stage})"""
        )

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
            if cfg.p_chunk_arrive and not causal:
                assert split_p_arrive and not mixed_p_store
                args.append(f"pforc_ptr_stage=flash_pforc_ptr + {2 * int(stage)}")
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
            # The row sum is the one publish without a score dependency: it can
            # follow the last alpha before the correction warp has consumed
            # that alpha (it may be parked on the other warpgroup's barrier),
            # and two ``bar.arrive`` from one warp complete the 64-thread named
            # barrier by themselves, stranding the correction warp's later
            # ``bar.sync`` (an intermittent deadlock on the last KV step). Also
            # wait for the last alpha's slot to be released: in the two-slot
            # ring that release has parity ``phase ^ index``. The single-slot
            # conservative handoff already waits for the previous publish.
            rowsum_producer_acquire = f"""        _helion_flash_rt.mbar_spin_wait(
            {corr_empty_ptr} + {corr_prod_index}, flash_s_corr_prod_phase, {cfg.wait_hint})
"""
        if not fa4_stat_handoff and not cfg.skip_rescale_stats:
            # Ring2 protects each data slot, but both slots reuse one named
            # notification barrier. Unlike the next alpha, the final row sum
            # has no intervening QK/PV dependency that guarantees correction
            # consumed the previous notification. Wait for that alpha's empty
            # credit before arriving again. The producer has already advanced:
            # its previous slot is index^1 and that slot's next phase is
            # phase^index. This observes the credit without advancing the ring.
            rowsum_producer_acquire = (
                f"""        if {kv_loop_bound} > 1:
            _helion_flash_rt.mbar_spin_wait(
                {corr_empty_ptr} + (flash_s_corr_prod_index ^ 1),
                flash_s_corr_prod_phase ^ flash_s_corr_prod_index, {cfg.wait_hint})
"""
                + rowsum_producer_acquire
            )
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
                    rowmax_lines.append(
                        f"            {fragment} = cute.make_rmem_tensor("
                        "flash_res_ld_shape, cutlass.Float32)"
                    )
                    if use_tmem_row_reduce:
                        rowmax_lines.extend(
                            f"""            flash_res_red = cute.make_rmem_tensor(
                ((1, 1), *{fragment}.shape[1:]), cutlass.Float32)
            cute.copy(flash_tiled_ldred{stage},
                tLDRedtS{stage}[None, {chunk}, None, None],
                ({fragment}, flash_res_red))
            for flash_red_i in cutlass.range_constexpr(cute.size(flash_res_red.shape)):
                flash_row_max = cute.arch.fmax(flash_row_max, flash_res_red[flash_red_i])""".splitlines()
                        )
                    else:
                        rowmax_lines.extend(
                            f"""            cute.copy({ld}, {ldt}[None, {chunk}, None, None], {fragment})
            flash_row_max = _helion_flash_rt._fmax_reduce_chunk_balanced({fragment}, flash_row_max)""".splitlines()
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
                if dense_resident_value_graph_candidate and use_2cta_instrs
                else ""
            )
            # Keep the old FP16 call text and default lowering unchanged.
            if explicit_resident_value_graph and io_dtype == "cutlass.BFloat16":
                resident_pfor_args += ", io_dtype=cutlass.BFloat16"
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

        def _sp_corr_publish_alpha_for(not_first: str) -> str:
            if cfg.skip_rescale_stats:
                return ""
            if acknowledged_stat_pipeline:
                return f"""            if {not_first}:
{corr_publish_alpha}
            else:
{corr_publish_dummy}"""
            return f"""            if {not_first}:
{corr_publish_alpha}"""

        if not cfg.skip_rescale_stats:
            sp_corr_publish_alpha = _sp_corr_publish_alpha_for(softmax_not_first)
        fa4_publish_alpha_before_exp = fa4_stat_handoff and (
            cfg.rescale_threshold > 0.0 or fa4_stat_pipeline
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
        plain_score_load = f"""            cute.copy({ld}, {ldt}, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}"""
        tmem_ld = f"flash_tiled_ldred{stage}" if is_causal else ld
        tmem_ldt = f"tLDRedtS{stage}" if is_causal else ldt
        tmem_score_load = f"""            tLDrS_red = cute.make_rmem_tensor(
                ((1, 1), *tLDrS.shape[1:]), cutlass.Float32)
            cute.copy({tmem_ld}, {tmem_ldt}, (tLDrS, tLDrS_red))
            cute.arch.fence_view_async_tmem_load()
            flash_hw_row_max = cutlass.Float32(-cutlass.Float32.inf)
            for flash_red_i in cutlass.range_constexpr(cute.size(tLDrS_red.shape)):
                flash_hw_row_max = cute.arch.fmax(
                    flash_hw_row_max, tLDrS_red[flash_red_i])"""
        software_rowmax = (
            "            flash_row_max = "
            "_helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
        )
        hardware_rowmax = (
            "            flash_row_max = cute.arch.fmax("
            "flash_row_max, flash_hw_row_max)"
        )
        if use_whole_row_tmem_reduce:
            assert not score_transform
            sp_score_load = tmem_score_load
            sp_rowmax = hardware_rowmax
        else:
            sp_score_load = plain_score_load
            sp_rowmax = software_rowmax
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

            masked_score_load = plain_score_load
            masked_rowmax = software_rowmax
            unmasked_score_load = tmem_score_load
            unmasked_rowmax = hardware_rowmax
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
                    row_sum_update = f"""{indent}flash_softmax.update_row_sum(
{indent}    tLDrS.load(), flash_alpha{row_sum_first_arg})"""
                    stat_acquire = f"""{indent}_helion_flash_rt.mbar_spin_wait(
{indent}    {corr_empty_ptr} + 0, flash_s_corr_prod_phase, {cfg.wait_hint})"""
                    post_p_statistics = (
                        f"{row_sum_update}\n{stat_acquire}"
                        if explicit_stateful and cfg.row_sum_schedule == "pre_acquire"
                        else f"{stat_acquire}\n{row_sum_update}"
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
{post_p_statistics}
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
                # A plain causal load includes the diagonal mask. The proven
                # suffix needs neither that mask nor LDRED: software row-max is
                # an independent implementation choice on the raw full row.
                stateful_unmasked_load = (
                    unmasked_score_load
                    if use_tmem_row_reduce
                    else f"""            cute.copy({ld}, {ldt}, tLDrS)
            cute.arch.fence_view_async_tmem_load()"""
                )
                stateful_unmasked_rowmax = (
                    "update_row_max_precomputed(flash_hw_row_max, False)"
                    if use_tmem_row_reduce
                    else "update_row_max_masked(tLDrS.load(), False)"
                )
                unmasked_step = _format_stateful_softmax_step(
                    stateful_unmasked_load,
                    stateful_unmasked_rowmax,
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
        # A KV tile width that does not divide the sequence leaves one partial
        # trailing tile. Its out-of-range columns arrive as zeros (TMA
        # zero-fills past the tensor extent), and a zero score is not a
        # no-op: exp2(0 - max) contributes to the row sum. Mask them to -inf.
        # The hardware TMEM row-reduce folds the max into the load, so the
        # masked tile must use the plain load and reduce in software -- the
        # same pair the causal diagonal already uses. Descending KV order puts
        # the partial tile first, so it peels off as a one-iteration prefix.
        split_segments: tuple[tuple[_FlashSoftmaxLoopSegment, str, str], ...] | None = (
            None
        )
        if has_kv_tail:
            assert desc_kv
            # Reuse the ordinary load. When it carries the hardware row
            # reduction the max it produces covers the out-of-range columns, so
            # it is discarded and the row max is recomputed in software from
            # the masked fragment.
            tail_score_load = (
                f"{sp_score_load}\n"
                "            _helion_flash_rt.mask_r2p_sm100_rank1(\n"
                f"                tLDrS, cutlass.Int32({kv_tail_cols}))"
            )
            tail_rowmax = (
                "            flash_row_max = "
                "_helion_flash_rt.fmax_reduce_packed(tLDrS, flash_row_max)"
            )
            tail_segment = _FlashSoftmaxLoopSegment(
                loop_var="flash_kv_tail_iter",
                loop_bound="cutlass.Int32(1)",
                kv_expr=f"{kv_loop_bound} - cutlass.Int32(1)",
            )
            body_segment = _FlashSoftmaxLoopSegment(
                loop_var="flash_kv_iter",
                loop_bound=f"{kv_loop_bound} - cutlass.Int32(1)",
                kv_expr=f"{kv_loop_bound} - cutlass.Int32(2) - flash_kv_iter",
                continues_previous_segment=True,
            )
            split_segments = (
                (tail_segment, tail_score_load, tail_rowmax),
                (body_segment, sp_score_load, sp_rowmax),
            )

        def _dense_iteration(
            score_load: str,
            rowmax: str,
            segment: _FlashSoftmaxLoopSegment | None = None,
        ) -> _FlashOnlineSoftmaxIteration:
            # A split loop has a different "is this the first KV tile" proof per
            # segment, and both the alpha pin and the alpha publish depend on it.
            not_first = None if segment is None else segment.not_first_condition
            alpha_pre = (
                _sp_alpha_pre
                if not_first is None or _sp_alpha_pre_for is None
                else _sp_alpha_pre_for(not_first)
            )
            publish = (
                sp_corr_publish_alpha
                if not_first is None
                else _sp_corr_publish_alpha_for(not_first)
            )
            return _FlashOnlineSoftmaxIteration(
                load_and_reduce=(
                    "            tLDrS = cute.make_rmem_tensor("
                    "tLDcS.shape, cutlass.Float32)\n"
                    f"{score_load}\n{rowmax}"
                ),
                alpha_pre_probability=alpha_pre,
                alpha_publish_pre_probability=(
                    publish if fa4_publish_alpha_before_exp else ""
                ),
                probability_update=sp_exp_block,
                # Generic XU computes P in registers. Its stores, TMEM fence,
                # and P-ready arrivals must precede the empty-slot acquire:
                # correction may need PV to complete before granting credit.
                # Split helpers already publish P inside probability_update.
                probability_publication=(
                    sp_p_store_block if acknowledged_stat_pipeline else ""
                ),
                alpha_post_probability=_sp_alpha_post,
                alpha_publish_post_probability=(
                    "" if fa4_publish_alpha_before_exp else publish
                ),
                statistics_acquire=post_p_stat_acquire,
                statistics_update=statistics_update,
                post_statistics=(
                    "" if acknowledged_stat_pipeline else sp_p_store_block
                ),
            )

        if split_segments is not None:
            loop = "\n".join(
                _format_fa4_online_softmax_loop(
                    segment,
                    _dense_iteration(score_load, rowmax, segment),
                    stage=stage,
                    wait_hint=cfg.wait_hint,
                )
                for segment, score_load, rowmax in split_segments
            )
        else:
            loop = _format_fa4_online_softmax_loop(
                softmax_segment,
                _dense_iteration(sp_score_load, sp_rowmax),
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
    flash_s_full_phase = cutlass.Int32(0){softmax_corr_prod_state}{_softmax_row_epilogue_head("0")}"""
    softmax1_head = f"""    cute.arch.setmaxregister_increase({cfg.softmax_regs})
{softmax1_setup.rstrip()}
    flash_s_full_phase = cutlass.Int32(0){softmax_corr_prod_state}{_softmax_row_epilogue_head("1")}"""
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
    softmax_prelude = (
        "decode" if softmax_needs_tile_decode or softmax_row_epilogue else "none"
    )
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
    if softmax_row_epilogue:
        # The aux views and the passes that never read O run before the KV
        # loop (hidden behind it); the O passes wait for the last PV and, on
        # the staged routes, for the epilogue warp to drain sO.
        softmax_row_sum_expr = (
            "flash_softmax.row_sum[0]"
            if use_causal_stateful_softmax
            else "flash_row_sum"
        )
        softmax0_inner = (
            row_pieces["0"][0]
            + "\n"
            + softmax0_inner
            + _softmax_row_epilogue("0", softmax_row_sum_expr)
        )
        softmax1_inner = (
            row_pieces["1"][0]
            + "\n"
            + softmax1_inner
            + _softmax_row_epilogue("1", softmax_row_sum_expr)
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
        if softmax_row_epilogue:
            # The softmax warpgroup evaluates the program; the correction warps
            # only retire the final row-sum handoff slot.  They still wait for
            # the last PV before pre-arriving the next tile's ``pfor``: that
            # arrival must not be counted toward the phase the softmax
            # warpgroup's final P store is still completing.
            return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
{stat_empty_arrive}\
        {_corr_wait_o_ready(stage)}"""
        if row_epilogue is not None:
            program_src = row_pieces[stage][1]
            if not epi_smem:
                return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        {_corr_wait_o_ready(stage)}
        tOgO_mma{stage} = {corr_gmem_o_index.format(mtile=mtile)}
        gO_epi{stage} = cute.zipped_divide(tOgO_mma{stage}, flash_epi_tiler{stage})
        tDgO{stage} = flash_thr_o_ld{stage}.partition_D(gO_epi{stage})
{program_src}"""
            return f"""        _helion_flash_rt.named_barrier_wait_unaligned(
            {3 + int(stage) * 4} + warp_idx % 4, 64)
        flash_inv_sum{stage} = _helion_flash_rt.rcp_approx_ftz({scale_expr})
{stat_empty_arrive}\
        {_corr_wait_o_ready(stage)}
        {_corr_wait_epi_empty(stage)}
{program_src}
        cute.arch.fence_view_async_shared()
        {_corr_commit_epi_full(stage)}
"""
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
    corr_row_prologue = "" if softmax_row_epilogue else row_prologue
    if cfg.skip_rescale_stats:
        corr_inner = f"""{corr_row_prologue}        # Final: divide by row_sum, cast, store (waits MMA's last-tile O_full).
{_corr_epi("0", corr_output_m_tile0)}
{_corr_epi("1", corr_output_m_tile1)}
{corr_epi_empty_toggle}
{corr_cons_advance.rstrip()}
        flash_o_full_phase ^= 1"""
    else:
        corr_inner = f"""{corr_row_prologue}{corr_stat_dummy}        for flash_kv in cutlass.range({kv_loop_bound_minus_1}, unroll=1):
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
    # A fused row epilogue addresses aux rows from the correction warps, so they
    # need the per-work-item tile decode even on the staged-smem store routes.
    corr_prelude = (
        "none"
        if epi_smem
        and not is_causal
        and not has_lse
        and not score_plan.modifiers
        and (row_epilogue is None or softmax_row_epilogue)
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
    row_epilogue: FlashRowEpilogueEmit | None = None,
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
    if row_epilogue is not None:
        o_store = "\n".join(
            _flash_row_epilogue_direct_source(
                row_epilogue,
                hd=hd,
                io_dtype=io_dtype,
                indent="",
                suffix="",
                mma_index="flash_pvt.partition_C({tensor})[None, None, None, flash_m_tile, 0, flash_bh]",
                epi_tiler="flash_epi_tiler",
                thr_ld="flash_thr_o_ld",
                tiled_ld="flash_tiled_o_ld",
                tDtO="tDtO",
                tDgO="tDgO",
                inv_sum="flash_inv_sum",
                ld_repetition=64,
            )
        )
    else:
        o_store = f"""flash_reg = cute.make_rmem_tensor(tDgO[None, None, 0].shape, cutlass.Float32)
flash_rego = cute.make_rmem_tensor(tDgO[None, None, 0].shape, {io_dtype})
for flash_i in cutlass.range(cute.size(tDtO, mode=[2])):
    cute.copy(flash_tiled_o_ld, tDtO[None, None, flash_i], flash_reg)
    flash_reg.store(flash_reg.load() * flash_inv_sum){relu_store}
    flash_rego.store(flash_reg.load().to({io_dtype}))
    cute.autovec_copy(flash_rego, tDgO[None, None, flash_i])"""
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
{o_store}
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
# TMA load atom and TMA-adjusted view of the single aux tensor a fused row
# epilogue stages through the ``sO`` tile (see ``_flash_aux_smem_staging``).
_FLASH_EPI_AUX_TMA_KERNEL_PARAMS = [
    "_flash_tma_aux0",
    "_flash_mEpiAux0t",
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
    row_epilogue: FlashRowEpilogueProgram | None = None
    epi_aux_args: tuple[TensorArg, ...] = ()
    epi_scalar_names: tuple[str, ...] = ()


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
    from ..host_function import HostFunction

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

    row_epilogue = graph_plan.row_epilogue
    aux_names = row_epilogue.aux_names if row_epilogue is not None else ()
    required_names = (
        graph_plan.q_name,
        graph_plan.k_name,
        graph_plan.v_name,
        graph_plan.o_name,
        *graph_plan.bias_names,
        *graph_plan.alibi_names,
        *graph_plan.document_names,
        *aux_names,
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
    resolved_aux_args: list[TensorArg] = []
    for name in aux_names:
        aux_arg = tensor_args_by_name.get(name)
        if aux_arg is None:
            return None
        if (
            aux_arg.fake_value.ndim != 3
            or aux_arg.fake_value.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            or not aux_arg.fake_value.is_contiguous()
            or int(aux_arg.fake_value.shape[0]) != graph_plan.batch
            or int(aux_arg.fake_value.shape[1]) != graph_plan.seq
            or int(aux_arg.fake_value.shape[2]) != graph_plan.head_dim
        ):
            return None
        resolved_aux_args.append(aux_arg)
    epi_scalar_names: list[str] = []
    if row_epilogue is not None:
        expr_to_origin = HostFunction.current().expr_to_origin
        for expr in row_epilogue.scalar_exprs:
            if any(symbol not in expr_to_origin for symbol in expr.free_symbols):
                return None
            epi_scalar_names.append(f"cutlass.Float32({df.sympy_expr(expr)})")

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
            not _flash_lse_tensor_geometry(lse_arg.fake_value)
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
        row_epilogue,
        tuple(resolved_aux_args),
        tuple(epi_scalar_names),
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
    from ..compile_environment import CompileEnvironment

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
    row_emit: FlashRowEpilogueEmit | None = None
    if plan.row_epilogue is not None:
        row_emit = FlashRowEpilogueEmit(
            plan.row_epilogue,
            tuple(
                _flash_epi_aux_param(index) for index in range(len(plan.epi_aux_args))
            ),
            tuple(
                _flash_epilogue_dtype_str(aux_arg.fake_value.dtype)
                for aux_arg in plan.epi_aux_args
            ),
            plan.epi_scalar_names,
        )
    small_biased_candidate = (
        plan.row_epilogue is None
        and _small_biased_attention_score_plan_supported(
            score_plan,
            seq=seq,
            head_dim=head_dim,
            io_dtype=io_dtype,
            has_lse=lse_arg is not None,
            bias_arg_count=len(bias_args),
        )
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
    # The row programs address q/k/v/o/lse with 16-byte cp.async and
    # st.global.v4 packets from the tensor bases. The bound kernel is keyed
    # on each input's pointer residue, so a base proven only 8-byte aligned
    # (a contiguous view four bf16 elements into a buffer, say) resolves to
    # the tcgen05 families for that binding alone.
    from .memory_ops import cute_tensor_base_is_aligned

    env = CompileEnvironment.current()
    row_mma_aligned = all(
        cute_tensor_base_is_aligned(env, arg.fake_value, 16)
        for arg in (q_arg, k_arg, v_arg, o_arg, lse_arg, *plan.epi_aux_args)
        if arg is not None
    )
    row_mma_aux_dtypes = () if row_emit is None else row_emit.aux_dtypes
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
        tmem_rowmax_compatible=flash_tmem_rowmax_score_plan_supported(score_plan),
        causal_resident_compatible=seq % 256 == 0,
        prefer_packed_reduce=bool(score_plan.modifiers),
        plain_row_body=row_emit is None and not score_plan.modifiers,
        has_row_epilogue=row_emit is not None,
        has_score_modifiers=bool(score_plan.modifiers),
        row_mma_aligned=row_mma_aligned,
        row_mma_aux_dtypes=row_mma_aux_dtypes,
        target_device_capability=(
            CompileEnvironment.current().config_spec.target_device_capability
        ),
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
    if (
        cfg.small_biased
        and row_emit is None
        and _small_biased_attention_simt_supported(
            score_plan,
            seq=seq,
            head_dim=head_dim,
            io_dtype=io_dtype,
            lse_arg=lse_arg,
            bias_args=bias_args,
        )
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

    if cfg.topology == ROW_MMA_FAMILY:
        # A fused row epilogue addresses its aux rows as plain kernel tensor
        # parameters (the same (B, S, D) geometry as O) and stages them through
        # shared memory, which the staging plan must leave room for.
        row_mma_epilogue: FlashRowEpilogueEmit | None = None
        if row_emit is not None:
            row_mma_epilogue = FlashRowEpilogueEmit(
                row_emit.program,
                tuple(aux_arg.name for aux_arg in plan.epi_aux_args),
                row_emit.aux_dtypes,
                row_emit.scalar_names,
            )
        row_mma_aux_bytes = row_mma_aux_smem_bytes(
            head_dim=head_dim,
            row_tile_m=cfg.row_tile_m,
            row_warps=cfg.row_warps,
            aux_dtypes=row_mma_aux_dtypes,
        )
        if not row_mma_shape_supported(
            seq=seq,
            head_dim=head_dim,
            row_warps=cfg.row_warps,
            row_tile_m=cfg.row_tile_m,
            aux_bytes=row_mma_aux_bytes,
        ):
            return False
        emit_flash_row_mma_module_statements(cg)
        cg.cute_wrapper_plans.append(
            {
                "kind": ROW_MMA_PLAN_KIND,
                "batch": batch,
                "seq": seq,
                "head_dim": head_dim,
                "row_warps": cfg.row_warps,
                "row_tile_m": cfg.row_tile_m,
                "total_tiles": batch * (seq // cfg.row_tile_m),
                # Resolved to launcher argument positions so the wrapper
                # can refuse a base its schema proves under-aligned.
                "q_name": q_arg.name,
                "k_name": k_arg.name,
                "v_name": v_arg.name,
                "o_name": o_arg.name,
                **({} if lse_arg is None else {"lse_name": lse_arg.name}),
                **(
                    {
                        "epi_aux_count": len(plan.epi_aux_args),
                        **{
                            f"epi_aux{index}_name": aux_arg.name
                            for index, aux_arg in enumerate(plan.epi_aux_args)
                        },
                    }
                    if plan.epi_aux_args
                    else {}
                ),
            }
        )
        df.placeholder_args.update((q_arg.name, k_arg.name, v_arg.name, o_arg.name))
        if lse_arg is not None:
            df.placeholder_args.add(lse_arg.name)
        df.placeholder_args.update(aux_arg.name for aux_arg in plan.epi_aux_args)
        df.cute_state.attention_flash_threads = 32 * cfg.row_warps
        df.body = emit_flash_row_mma_device_body(
            q_name=q_arg.name,
            k_name=k_arg.name,
            v_name=v_arg.name,
            o_name=o_arg.name,
            lse_name=None if lse_arg is None else lse_arg.name,
            num_bh=batch,
            seq=seq,
            head_dim=head_dim,
            io_dtype=io_dtype_str,
            scale_log2=scale_log2,
            lse_scale=score_plan.lse_scale,
            row_warps=cfg.row_warps,
            row_tile_m=cfg.row_tile_m,
            relu_output=_flash_output_relu_enabled(output_epilogue),
            row_epilogue=row_mma_epilogue,
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
        total_tiles = batch * (seq // cfg.q_tile_m)

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
        "kv_tile_n": cfg.kv_tile_n,
        "s_stage": cfg.s_stage,
        "persistent": cfg.persistent,
        "persistent_ctas_per_sm": cfg.persistent_ctas_per_sm,
        "total_tiles": total_tiles,
        "topology": cfg.topology,
        # The fa4 topology stages 2 Q-tiles per CTA -> the Q smem layout must
        # be 2-deep (ws_overlap stages a single Q-tile -> 1).
        "q_stage": 2 if cfg.alternating_warpgroups else cfg.q_tile_count,
        "q_tile_m": cfg.q_tile_m,
        "ws_one_pass": cfg.ws_one_pass,
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
    aux_smem_staging = _flash_aux_smem_staging(
        cfg,
        row_emit,
        io_dtype_str,
        tensor_4d=(
            cfg.tensor_4d_tma
            and plan.tensor_4d_batch > 0
            and plan.tensor_4d_heads > 0
            and plan.tensor_4d_batch * plan.tensor_4d_heads == batch
        ),
    )
    if row_emit is not None:
        wrapper_plan["epi_aux_count"] = len(plan.epi_aux_args)
        for index, aux_arg in enumerate(plan.epi_aux_args):
            wrapper_plan[f"epi_aux{index}_name"] = aux_arg.name
        wrapper_plan["epi_aux_tma"] = aux_smem_staging
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
    if row_emit is not None:
        df.wrapper_only_params.extend(row_emit.aux_params)
        if aux_smem_staging:
            df.wrapper_only_params.extend(_FLASH_EPI_AUX_TMA_KERNEL_PARAMS)
        df.placeholder_args.update(aux_arg.name for aux_arg in plan.epi_aux_args)
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
                    row_epilogue=row_emit,
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
                    row_epilogue=row_emit,
                )
            )
    elif cfg.topology == "fa4" and cfg.alternating_warpgroups:
        emit_flash_alt_module_statements(cg)
        df.cute_state.attention_flash_threads = 512
        alt_exp2 = _flash_disc_exp2_codegen_params(
            cfg.exp2_packet, cfg.e2e_freq, cfg.e2e_res
        )
        df.body = list(
            emit_flash_fa4_alt_device_body(
                head_dim=head_dim,
                num_kv=num_kv,
                sequence_extent=seq,
                total_tiles=total_tiles,
                cfg=cfg,
                has_lse=lse_arg is not None,
                io_dtype=io_dtype_str,
                lse_scale=score_plan.lse_scale,
                exp2_pair_batch=alt_exp2.pair_batch,
                exp2_emu_batch=alt_exp2.emu_batch,
                exp2_degree2=alt_exp2.degree2,
                exp2_degree1=alt_exp2.degree1_unmasked,
                e2e_freq=alt_exp2.e2e_freq,
                e2e_res=alt_exp2.e2e_res,
            )
        )
    elif cfg.topology == "fa4":
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
                row_epilogue=row_emit,
            )
        )
    else:
        raise ValueError(f"unknown flash attention topology: {cfg.topology!r}")

    df.preamble = []
    return True
