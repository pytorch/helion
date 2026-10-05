"""Tests for the whole-root tcgen05 lowering of the gdn chunk recurrence."""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect
import math
import operator
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute.gdn_recurrence_geometry import GDN_ADMITTED_BLOCK_V
from helion._compiler.cute.gdn_recurrence_geometry import GDN_MMA_M
from helion._compiler.cute.gdn_recurrence_geometry import gdn_acc_load_cols
from helion._compiler.cute.gdn_recurrence_geometry import gdn_candidate_block_sizes
from helion._compiler.cute.gdn_recurrence_geometry import gdn_cta_warps
from helion._compiler.cute.gdn_recurrence_geometry import gdn_epilogue_warp_choices
from helion._compiler.cute.gdn_recurrence_geometry import gdn_half_lanes
from helion._compiler.cute.gdn_recurrence_geometry import gdn_max_stages
from helion._compiler.cute.gdn_recurrence_geometry import gdn_mma_m_choices
from helion._compiler.cute.gdn_recurrence_geometry import gdn_prefetch_tokens
from helion._compiler.cute.gdn_recurrence_geometry import gdn_shape_admitted
from helion._compiler.cute.gdn_recurrence_geometry import gdn_smem_bytes
from helion._compiler.cute.gdn_recurrence_geometry import gdn_stage_choices
from helion._compiler.cute.gdn_recurrence_geometry import gdn_stage_smem_bytes
from helion._compiler.cute.gdn_recurrence_geometry import gdn_state_replicas
from helion._compiler.cute.gdn_recurrence_geometry import gdn_store_warps
from helion._compiler.cute.gdn_recurrence_geometry import gdn_tmem_layout
from helion._compiler.cute.gdn_recurrence_geometry import gdn_token_group_choices
from helion._compiler.cute.gdn_recurrence_geometry import gdn_token_splits
from helion._compiler.cute.gdn_recurrence_geometry import gdn_tokens_per_split
from helion._compiler.cute.gdn_recurrence_geometry import gdn_update_tile_bytes
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY
from helion.autotuner.config_spec import CUTE_GDN_RECURRENCE_MMA_M_KEY
from helion.autotuner.config_spec import CUTE_GDN_RECURRENCE_STAGES_KEY
from helion.autotuner.config_spec import CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

_CAPABILITY = (10, 0)
# The launcher argument checks require CUDA tensors; fake tensors on this
# device exercise them without a GPU.
_FAKE_CUDA = torch.device("cuda")
_PLAN_KIND = "'kind': 'gdn_recurrence_sm100'"
_SIMT_FALLBACK = "_cute_grouped_reduce_shared_two_stage"


def _example_module() -> object:
    return import_path(EXAMPLES_DIR / "gdn_fwd_h.py")


def _fake_inputs(
    *,
    batch: int = 2,
    heads: int = 8,
    seqlen: int = 1024,
    dhead: int = 64,
    dstate: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    gate_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    with FakeTensorMode():
        k = torch.empty(batch, seqlen, heads, dhead, dtype=dtype, device=DEVICE)
        w = torch.empty(batch, seqlen, heads, dhead, dtype=dtype, device=DEVICE)
        u = torch.empty(batch, seqlen, heads, dstate, dtype=dtype, device=DEVICE)
        g = torch.empty(batch, seqlen, heads, dtype=gate_dtype, device=DEVICE)
    return k, w, u, g


@helion.kernel()
def _gdn_fwd_h_with_head_end(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """``examples/gdn_fwd_h.py`` with ``tile_h.end`` as the head coordinate.

    With a head tile of one row this reads head + 1.  It resolves to the same
    block id as ``tile_h.id``, so only the origin check can tell them apart.
    """

    batch, seqlen, nheads, dhead = k.shape
    dhead = hl.specialize(dhead)
    chunk_size = hl.specialize(chunk_size)
    dstate = u.shape[-1]

    acc_dtype = torch.float32
    dtype = k.dtype

    nchunks = (seqlen + chunk_size - 1) // chunk_size
    h = torch.empty(batch, nchunks, nheads, dhead, dstate, dtype=dtype, device=k.device)
    block_v = hl.register_block_size(dstate)

    for tile_b, tile_h, tile_v in hl.tile(
        [batch, nheads, dstate], block_size=[1, 1, block_v]
    ):
        i_b = tile_b.id
        i_h = tile_h.end
        b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)
        for t_i in hl.tile(seqlen, block_size=chunk_size):
            h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(dtype)
            b_w = w[i_b, t_i, i_h, :]
            c_h = b_h.to(dtype)
            b_v = hl.dot(b_w, c_h, out_dtype=acc_dtype)
            p_v = u[i_b, t_i, i_h, tile_v].to(acc_dtype)
            b_v = p_v - b_v
            m_t = t_i.index < seqlen
            t_i_last = min(t_i.begin + chunk_size, seqlen) - 1
            b_g_last = g[i_b, t_i_last, i_h].to(acc_dtype)
            b_g = g[i_b, t_i, i_h].to(acc_dtype)
            b_v *= torch.where(m_t, torch.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = torch.exp(b_g_last)
            b_h *= b_g_last
            b_v = b_v.to(dtype)
            p_k = k[i_b, t_i, i_h, :]
            b_h = hl.dot(p_k.T, b_v, acc=b_h)
    return h


@contextlib.contextmanager
def _sm100_environment(
    capability: tuple[int, int] = _CAPABILITY, num_sm: int = 148
) -> Iterator[None]:
    with (
        patch(
            "helion.runtime.kernel.target_device_capability",
            return_value=capability,
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=capability,
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch(
            "helion.language.loops._supports_warp_specialize",
            return_value=capability >= (10, 0),
        ),
        patch(
            "helion._compat._supports_tensor_descriptor",
            return_value=capability >= (9, 0),
        ),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.runtime.get_num_sm", return_value=num_sm),
    ):
        yield


def _bind(
    *,
    chunk: int = 128,
    shape: dict[str, int] | None = None,
    capability: tuple[int, int] = _CAPABILITY,
    mutate: Callable[[object], None] | None = None,
    kernel: object | None = None,
    inputs: tuple[torch.Tensor, ...] | None = None,
) -> object:
    if kernel is None:
        kernel = cast("object", _example_module()).helion_gdn_fwd_h  # type: ignore[attr-defined]
    if inputs is None:
        inputs = _fake_inputs(**cast("dict[str, Any]", shape or {}))
    bound = kernel._bind_isolated((*inputs, chunk))  # type: ignore[attr-defined]
    if mutate is not None:
        mutate(bound.host_function.device_ir)
    bound.config_spec.target_device_capability = capability
    return bound


def _config(bound: object, block_v: int, **overrides: object) -> helion.Config:
    config = bound.config_spec.default_config()  # type: ignore[attr-defined]
    config.config["block_sizes"] = [block_v]
    config.config.update(overrides)
    return config


def _code(
    *,
    chunk: int = 128,
    block_v: int = 128,
    shape: dict[str, int] | None = None,
    capability: tuple[int, int] = _CAPABILITY,
    mutate: Callable[[object], None] | None = None,
    overrides: dict[str, object] | None = None,
    kernel: object | None = None,
    inputs: tuple[torch.Tensor, ...] | None = None,
) -> str:
    with _sm100_environment(capability):
        bound = _bind(
            chunk=chunk,
            shape=shape,
            capability=capability,
            mutate=mutate,
            kernel=kernel,
            inputs=inputs,
        )
        return bound.to_triton_code(  # type: ignore[attr-defined]
            _config(bound, block_v, **(overrides or {}))
        )


def _captured_plan(
    *,
    chunk: int = 128,
    block_v: int = 128,
    shape: dict[str, int] | None = None,
    capability: tuple[int, int] = _CAPABILITY,
    mutate: Callable[[object], None] | None = None,
    kernel: object | None = None,
    inputs: tuple[torch.Tensor, ...] | None = None,
) -> object:
    from helion._compiler.cute import gdn_recurrence

    captured: list[object] = []
    original = gdn_recurrence._plan_gdn_recurrence

    def capture(graphs: object, tile_strategy: object) -> object:
        result = original(graphs, tile_strategy)  # type: ignore[arg-type]
        captured.append(result)
        return result

    with (
        patch.object(gdn_recurrence, "_plan_gdn_recurrence", side_effect=capture),
        contextlib.suppress(exc.BackendUnsupported),
    ):
        _code(
            chunk=chunk,
            block_v=block_v,
            shape=shape,
            capability=capability,
            mutate=mutate,
            kernel=kernel,
            inputs=inputs,
        )
    assert len(captured) == 1
    return captured[0]


def test_gdn_recurrence_codegen_emits_sm100_plan() -> None:
    code = _code()
    assert "from helion._compiler.cute.gdn_recurrence_sm100 import" in code
    assert _PLAN_KIND in code
    assert "'block_v': 128" in code
    assert "'epilogue_warps': 8" in code
    assert "'stages': 3" in code
    assert "'threads': 288" in code
    assert "'tmem_cols': 512" in code
    assert "'state_col': 0" in code
    assert "'state_image_col': 64" in code
    assert "'acc_col': 96" in code
    assert "'update_image_col': 224" in code
    assert "'num_chunks': 8" in code
    assert "'token_groups': 2" in code
    assert "'mma_m': 128" in code
    assert "'device_abi': 5" in code
    assert "block=(288, 1, 1)" in code
    assert _SIMT_FALLBACK not in code
    assert "cute.gemm(" not in code


def test_gdn_recurrence_knobs_reach_the_plan() -> None:
    code = _code(
        overrides={
            CUTE_GDN_RECURRENCE_STAGES_KEY: 2,
            CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: 16,
            CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 4,
        }
    )
    assert _PLAN_KIND in code
    assert "'stages': 2" in code
    assert "'epilogue_warps': 16" in code
    assert "'token_groups': 4" in code
    assert "'threads': 544" in code
    assert "block=(544, 1, 1)" in code


@pytest.mark.parametrize("block_v", (16, 32, 64))
def test_gdn_recurrence_smaller_dstate_tiles_pad_the_mma(block_v: int) -> None:
    code = _code(block_v=block_v)
    assert _PLAN_KIND in code
    assert f"'block_v': {block_v}" in code
    # The narrower tiles replicate the state across the TMEM quadrants: the
    # eight epilogue warps, four h store warps and the TMA warp.  A config
    # that leaves the MMA height out keeps the full 128-row tile.
    assert "'threads': 416" in code
    assert "'mma_m': 128" in code


@pytest.mark.parametrize(
    ("block_v", "threads", "smem_bytes"),
    (
        # The 64-row tile fills the 64-row MMA: no replica, no store warps, no
        # shared-memory update image.
        (64, 288, gdn_smem_bytes(128, 64, 64, 3, 64)),
        # Two and four replicas keep the store warps and a half-size image.
        (32, 416, gdn_smem_bytes(128, 64, 32, 3, 64)),
        (16, 416, gdn_smem_bytes(128, 64, 16, 3, 64)),
    ),
)
def test_gdn_recurrence_mma_m64_reaches_the_plan(
    block_v: int, threads: int, smem_bytes: int
) -> None:
    code = _code(block_v=block_v, overrides={CUTE_GDN_RECURRENCE_MMA_M_KEY: 64})
    assert _PLAN_KIND in code
    assert "'mma_m': 64" in code
    assert f"'threads': {threads}" in code
    assert f"block=({threads}, 1, 1)" in code
    assert f"'smem_bytes': {smem_bytes}" in code
    assert smem_bytes < gdn_smem_bytes(128, 64, block_v, 3, GDN_MMA_M)


def test_gdn_recurrence_mma_m64_needs_a_tile_of_at_most_64_rows() -> None:
    """The 128-row tile has no 64-row MMA: normalization rejects an explicit
    request up front (and the launcher re-proves the pair at run time)."""

    with pytest.raises(exc.InvalidConfig, match="dstate tile and epilogue warps"):
        _code(block_v=128, overrides={CUTE_GDN_RECURRENCE_MMA_M_KEY: 64})


@pytest.mark.parametrize(("epilogue_warps", "threads"), ((4, 288), (8, 416), (16, 672)))
def test_gdn_recurrence_thread_count_follows_the_epilogue_warps(
    epilogue_warps: int, threads: int
) -> None:
    code = _code(
        chunk=64,
        block_v=32,
        shape={"dhead": 32, "dstate": 32, "seqlen": 256, "heads": 4, "batch": 1},
        overrides={
            CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: epilogue_warps,
            # The seeded default picks the 64-row MMA on a real device; the
            # shared-memory figure below is the full-height one.
            CUTE_GDN_RECURRENCE_MMA_M_KEY: GDN_MMA_M,
        },
    )
    assert _PLAN_KIND in code
    assert f"'threads': {threads}" in code
    assert f"block=({threads}, 1, 1)" in code
    assert f"'smem_bytes': {gdn_smem_bytes(64, 32, 32, 3, GDN_MMA_M)}" in code


@pytest.mark.parametrize(
    ("chunk", "shape"),
    (
        (64, {"dhead": 32, "dstate": 32, "seqlen": 256, "heads": 4, "batch": 1}),
        (64, {"dhead": 64, "dstate": 64, "seqlen": 512, "heads": 8, "batch": 1}),
        (64, {"dhead": 16, "dstate": 32, "seqlen": 512, "heads": 4}),
        (256, {"dhead": 64, "dstate": 128, "seqlen": 4096, "heads": 3, "batch": 1}),
    ),
)
def test_gdn_recurrence_admits_the_study_siblings(
    chunk: int, shape: dict[str, int]
) -> None:
    block_v = gdn_candidate_block_sizes(shape["dstate"])[0]
    plan = _captured_plan(chunk=chunk, block_v=block_v, shape=shape)
    assert plan is not None
    assert plan.block_v == block_v  # type: ignore[attr-defined]
    assert plan.geometry.dhead == shape["dhead"]  # type: ignore[attr-defined]
    assert plan.geometry.chunk == chunk  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("chunk", "block_v", "shape"),
    (
        (128, 128, {"dhead": 128}),
        (128, 128, {"dhead": 8}),
        (8, 128, {"seqlen": 512}),
        (512, 128, {"seqlen": 1024}),
        (96, 128, {"seqlen": 960}),
        # dstate must keep the u tensor map's head stride 16-byte aligned.
        (128, 64, {"dstate": 100}),
        (64, 32, {"dstate": 36, "dhead": 32, "seqlen": 256}),
    ),
)
def test_gdn_recurrence_unadmitted_shapes_fall_back(
    chunk: int, block_v: int, shape: dict[str, int]
) -> None:
    assert _captured_plan(chunk=chunk, block_v=block_v, shape=shape) is None
    assert _PLAN_KIND not in _code(chunk=chunk, block_v=block_v, shape=shape)


def test_gdn_recurrence_rejects_sm80() -> None:
    assert _captured_plan(capability=(8, 0)) is None


@pytest.mark.parametrize(
    ("dtype", "gate_dtype"),
    ((torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16)),
)
def test_gdn_recurrence_rejects_other_dtypes(
    dtype: torch.dtype, gate_dtype: torch.dtype
) -> None:
    inputs = _fake_inputs(dtype=dtype, gate_dtype=gate_dtype)
    assert _captured_plan(inputs=inputs) is None
    assert _PLAN_KIND not in _code(inputs=inputs)
    with _sm100_environment():
        bound = _bind(inputs=inputs)
    assert bound.config_spec.cute_gdn_recurrence_stages is None  # type: ignore[attr-defined]


def test_gdn_recurrence_rejects_tile_end_coordinates() -> None:
    assert _captured_plan(kernel=_gdn_fwd_h_with_head_end) is None
    assert _PLAN_KIND not in _code(kernel=_gdn_fwd_h_with_head_end)
    with _sm100_environment():
        bound = _bind(kernel=_gdn_fwd_h_with_head_end)
    assert bound.config_spec.cute_gdn_recurrence_stages is None  # type: ignore[attr-defined]


def test_gdn_recurrence_ragged_seqlen_and_dstate_tail() -> None:
    code = _code(shape={"seqlen": 1000, "dstate": 192, "heads": 3, "batch": 1})
    assert _PLAN_KIND in code
    assert "'num_chunks': 8" in code
    assert "'seqlen': 1000" in code
    assert "'dstate': 192" in code
    assert "'block_v': 128" in code


def _loop_nodes(device_ir: object) -> list[torch.fx.Node]:
    from helion._compiler.device_ir import ForLoopGraphInfo

    loop = next(
        graph
        for graph in device_ir.graphs  # type: ignore[attr-defined]
        if isinstance(graph, ForLoopGraphInfo)
    )
    return list(loop.graph.nodes)


def _root_nodes(device_ir: object) -> list[torch.fx.Node]:
    from helion._compiler.device_ir import RootGraphInfo

    root = next(
        graph
        for graph in device_ir.graphs  # type: ignore[attr-defined]
        if isinstance(graph, RootGraphInfo)
    )
    return list(root.graph.nodes)


def _node(nodes: list[torch.fx.Node], target: object, index: int = 0) -> torch.fx.Node:
    return [
        node for node in nodes if node.op == "call_function" and node.target is target
    ][index]


def _arg_node(node: torch.fx.Node, index: int) -> torch.fx.Node:
    return cast("torch.fx.Node", node.args[index])


def _reverse_residual(device_ir: object) -> None:
    sub = _node(_loop_nodes(device_ir), torch.ops.aten.sub.Tensor)
    sub.args = (sub.args[1], sub.args[0])


def _reverse_gate_difference(device_ir: object) -> None:
    sub = _node(_loop_nodes(device_ir), torch.ops.aten.sub.Tensor, 1)
    sub.args = (sub.args[1], sub.args[0])


def _change_gate_default(device_ir: object) -> None:
    zero = _node(_loop_nodes(device_ir), torch.ops.aten.scalar_tensor.default)
    zero.args = (1,)


def _drop_gate_mask(device_ir: object) -> None:
    from helion.language.view_ops import subscript

    nodes = _loop_nodes(device_ir)
    where = _node(nodes, torch.ops.aten.where.self)
    bcast = _node(nodes, subscript)
    bcast.args = (where.args[1], bcast.args[1])


def _rescale_with_chunk_gate(device_ir: object) -> None:
    from helion.language.memory_ops import load

    nodes = _loop_nodes(device_ir)
    g_chunk = _node(nodes, load, 3)
    rescale_exp = _node(nodes, torch.ops.aten.exp.default, 1)
    rescale_exp.args = (g_chunk,)


def _drop_rescale(device_ir: object) -> None:
    from helion.language.matmul_ops import dot

    update = _node(_loop_nodes(device_ir), dot, 1)
    state_phi = _arg_node(_arg_node(update, 2), 0)
    update.args = (update.args[0], update.args[1], state_phi, update.args[3])


def _swap_projection_operands(device_ir: object) -> None:
    from helion.language.matmul_ops import dot

    projection = _node(_loop_nodes(device_ir), dot)
    projection.args = (projection.args[1], projection.args[0], *projection.args[2:])


def _identity_permute(device_ir: object) -> None:
    permute = _node(_loop_nodes(device_ir), torch.ops.aten.permute.default)
    permute.args = (permute.args[0], [0, 1])


def _project_in_fp16(device_ir: object) -> None:
    from helion.language.matmul_ops import dot

    projection = _node(_loop_nodes(device_ir), dot)
    convert = _arg_node(projection, 1)
    convert.args = (convert.args[0], torch.float16)


def _store_unrounded_state(device_ir: object) -> None:
    from helion.language._tracing_ops import _new_var
    from helion.language.memory_ops import store

    nodes = _loop_nodes(device_ir)
    stored = _node(nodes, store)
    stored.args = (stored.args[0], stored.args[1], _node(nodes, _new_var), None)


def _store_with_chunk_offset(device_ir: object) -> None:
    from helion.language.memory_ops import store
    from helion.language.tile_ops import tile_id

    nodes = _loop_nodes(device_ir)
    stored = _node(nodes, store)
    chunk_sym = _node(nodes, tile_id).args[0]
    indices = list(cast("list[object]", stored.args[1]))
    indices[1] = chunk_sym
    stored.args = (
        stored.args[0],
        cast("torch.fx.node.Argument", indices),
        *stored.args[2:],
    )


def _load_k_with_head_as_batch(device_ir: object) -> None:
    from helion.language.memory_ops import load

    k_load = _node(_loop_nodes(device_ir), load, 4)
    indices = list(cast("list[object]", k_load.args[1]))
    indices[0] = indices[2]
    k_load.args = (
        k_load.args[0],
        cast("torch.fx.node.Argument", indices),
        *k_load.args[2:],
    )


def _mask_the_gate_load(device_ir: object) -> None:
    from helion.language.memory_ops import load

    nodes = _loop_nodes(device_ir)
    g_load = _node(nodes, load, 3)
    mask = _node(nodes, torch.ops.aten.lt.Scalar)
    g_load.args = (g_load.args[0], g_load.args[1], mask, g_load.args[3])


def _shift_the_last_gate_row(device_ir: object) -> None:
    # g_last must be the gate of the last valid token of the chunk (t - 1).
    last = _node(_loop_nodes(device_ir), operator.sub)
    last.args = (last.args[0], 2)


def _halve_loop_extent(device_ir: object) -> None:
    from helion.language._tracing_ops import _for_loop

    loop_call = _node(_root_nodes(device_ir), _for_loop)
    loop_call.args = (loop_call.args[0], loop_call.args[1], [512], loop_call.args[3])


def _nonzero_initial_state(device_ir: object) -> None:
    from helion.language.creation_ops import full

    full_node = _node(_root_nodes(device_ir), full)
    full_node.args = (full_node.args[0], 1.0, *full_node.args[2:])


def _permute_root_task_axes(device_ir: object) -> None:
    axes = list(device_ir.task_families[0].axes)  # type: ignore[attr-defined]
    axes[0], axes[1] = axes[1], axes[0]
    device_ir.task_families[0] = dataclasses.replace(  # type: ignore[attr-defined]
        device_ir.task_families[0],  # type: ignore[attr-defined]
        axes=tuple(axes),
    )


def _shorten_root_task_extent(device_ir: object) -> None:
    axes = list(device_ir.task_families[0].axes)  # type: ignore[attr-defined]
    extent = axes[2].extent
    assert extent is not None and not isinstance(extent, str)
    axes[2] = dataclasses.replace(axes[2], extent=extent - 1)
    device_ir.task_families[0] = dataclasses.replace(  # type: ignore[attr-defined]
        device_ir.task_families[0],  # type: ignore[attr-defined]
        axes=tuple(axes),
    )


@pytest.mark.parametrize(
    "mutate",
    (
        _reverse_residual,
        _reverse_gate_difference,
        _change_gate_default,
        _drop_gate_mask,
        _rescale_with_chunk_gate,
        _drop_rescale,
        _swap_projection_operands,
        _identity_permute,
        _project_in_fp16,
        _store_unrounded_state,
        _store_with_chunk_offset,
        _load_k_with_head_as_batch,
        _mask_the_gate_load,
        _shift_the_last_gate_row,
        _halve_loop_extent,
        _nonzero_initial_state,
        _permute_root_task_axes,
        _shorten_root_task_extent,
    ),
)
def test_gdn_recurrence_rejects_semantic_mutations(
    mutate: Callable[[object], None],
) -> None:
    assert _captured_plan(mutate=mutate) is None


def test_gdn_recurrence_search_knobs_and_seeds() -> None:
    with _sm100_environment():
        bound = _bind()
    spec = bound.config_spec  # type: ignore[attr-defined]
    # The ring-depth domain is the union over the admitted tiles: the 128-row
    # tile fits three stages, the 64-row tile (whose replicated update image
    # takes shared memory) three as well, the smaller tiles four.
    assert spec.cute_gdn_recurrence_stages.choices == (3, 2, 1, 4)
    assert spec.cute_gdn_recurrence_epilogue_warps.choices == (8, 4, 16)
    # The MMA height: the 64-row tcgen05 tile for tiles of at most 64 rows,
    # the full tile for every tile; the full tile leads the domain so a config
    # that leaves the knob out keeps it.
    assert spec.cute_gdn_recurrence_mma_m.choices == (128, 64)
    assert spec.cute_gdn_recurrence_mma_m_choices_by_tile == {
        (tile, warps): ((64, 128) if tile <= 64 else (128,))
        for tile in (128, 64, 32, 16)
        for warps in (8, 4, 16)
    }
    # Ring depths per (tile, M): the 64-row tile under the full MMA keeps
    # its replicated update image in shared memory (three stages), under
    # M = 64 it has no replica and fits four like the smaller tiles.
    assert spec.cute_gdn_recurrence_stage_choices_by_tile == {
        (128, 128): (3, 2, 1),
        (64, 64): (3, 4, 2, 1),
        (64, 128): (3, 2, 1),
        (32, 64): (3, 4, 2, 1),
        (32, 128): (3, 4, 2, 1),
        (16, 64): (3, 4, 2, 1),
        (16, 128): (3, 4, 2, 1),
    }
    # Token groups: the 128-row tile's two warps per quadrant split a chunk
    # in two; four groups need four computing warps per replica set.  Under
    # M = 64 a 64-row tile has one replica (its four warps share one slice)
    # and a 32-row tile two.
    assert spec.cute_gdn_recurrence_token_groups.choices == (2, 1, 4)
    by_tile = spec.cute_gdn_recurrence_token_group_choices_by_tile
    assert by_tile[(128, 8, 128)] == (2, 1)
    assert by_tile[(128, 4, 128)] == (1,)
    assert by_tile[(128, 16, 128)] == (2, 1, 4)
    assert by_tile[(64, 4, 128)] == (2, 1)
    assert by_tile[(32, 4, 128)] == (2, 1, 4)
    assert by_tile[(64, 4, 64)] == (1,)
    assert by_tile[(64, 8, 64)] == (2, 1)
    assert by_tile[(32, 4, 64)] == (2, 1)
    assert by_tile[(32, 8, 64)] == (2, 1, 4)
    assert (128, 8, 64) not in by_tile
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if CUTE_GDN_RECURRENCE_STAGES_KEY in seed.config
    ]
    assert seeds[0] == {
        "block_sizes": [128],
        CUTE_GDN_RECURRENCE_STAGES_KEY: 3,
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: 8,
        CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 2,
        CUTE_GDN_RECURRENCE_MMA_M_KEY: 128,
    }
    # Every seed carries its tile's preferred MMA height (M = 64 for the
    # 64-row tile) and the preferred pipelining its tile, warps and M admit.
    assert all(
        seed[CUTE_GDN_RECURRENCE_MMA_M_KEY]
        == (64 if seed["block_sizes"] == [64] else 128)
        for seed in seeds
    )
    assert all(
        seed[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY]
        == by_tile[
            (
                seed["block_sizes"][0],
                seed[CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY],
                seed[CUTE_GDN_RECURRENCE_MMA_M_KEY],
            )
        ][0]
        for seed in seeds
    )
    assert {seed[CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY] for seed in seeds} == {
        4,
        8,
        16,
    }
    assert [seed["block_sizes"] for seed in seeds][:6] == [[128]] * 6
    assert {tuple(seed["block_sizes"]) for seed in seeds} == {(128,), (64,)}
    # The 64-row tile's seeds take M = 64's preferred depths (three, then four).
    assert [
        seed[CUTE_GDN_RECURRENCE_STAGES_KEY]
        for seed in seeds
        if seed["block_sizes"] == [64]
    ] == [3, 4] * 3
    # Every seed is legal as written: normalization neither repairs a depth
    # nor folds two seeds into one.
    assert len(seeds) == 12
    normalized = [spec.normalized_config(seed).config for seed in seeds]
    assert [
        {key: config[key] for key in seed}
        for seed, config in zip(seeds, normalized, strict=True)
    ] == seeds
    assert len({str(sorted(config.items())) for config in normalized}) == 12
    dimensions = {
        dimension.name: dimension.values
        for dimension in spec.iter_search_dimensions()
        if dimension.name
        in (
            CUTE_GDN_RECURRENCE_STAGES_KEY,
            CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY,
            CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY,
            CUTE_GDN_RECURRENCE_MMA_M_KEY,
        )
    }
    assert dimensions == {
        CUTE_GDN_RECURRENCE_STAGES_KEY: [3, 2, 1, 4],
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: [8, 4, 16],
        CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: [2, 1, 4],
        CUTE_GDN_RECURRENCE_MMA_M_KEY: [128, 64],
    }
    # The MMA height follows the selected tile and warps: a config that
    # leaves it out keeps the full tile, an explicit M = 64 on the 128-row
    # tile is rejected (or repaired), and the ring depth then follows the
    # (tile, M) pair: four stages fit the 64-row tile only under M = 64.
    m64 = {CUTE_GDN_RECURRENCE_MMA_M_KEY: 64}
    assert (
        spec.normalized_config(helion.Config(block_sizes=[64])).config[
            CUTE_GDN_RECURRENCE_MMA_M_KEY
        ]
        == 128
    )
    with pytest.raises(exc.InvalidConfig, match="dstate tile and epilogue warps"):
        spec.normalized_config(_config(bound, 128, **m64))
    repaired = _config(bound, 128, **m64)
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config[CUTE_GDN_RECURRENCE_MMA_M_KEY] == 128
    assert (
        spec.normalized_config(
            _config(bound, 64, **m64, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 4})
        ).config[CUTE_GDN_RECURRENCE_STAGES_KEY]
        == 4
    )
    with pytest.raises(exc.InvalidConfig, match="must be one of .* dstate tile"):
        spec.normalized_config(
            _config(bound, 64, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 4})
        )
    # A depth left to the default follows the (tile, M) pair as well.
    assert (
        spec.normalized_config(helion.Config(block_sizes=[64], **m64)).config[
            CUTE_GDN_RECURRENCE_STAGES_KEY
        ]
        == 3
    )
    # Token groups follow the (tile, warps, M) triple: two groups need two
    # computing warps per replica set, which the unreplicated 64-row tile
    # only has with eight warps.
    with pytest.raises(exc.InvalidConfig, match="dstate tile and epilogue warps"):
        spec.normalized_config(
            _config(
                bound,
                64,
                **m64,
                **{
                    CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: 4,
                    CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 2,
                },
            )
        )
    # Token groups follow the selected tile and warp count the same way: a
    # config that leaves them out gets the pair's first legal count, an
    # explicit count the pair cannot split is rejected (or repaired).
    four_warps = {CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: 4}
    assert (
        spec.normalized_config(helion.Config(block_sizes=[128], **four_warps)).config[
            CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY
        ]
        == 1
    )
    assert (
        spec.normalized_config(
            _config(
                bound, 32, **four_warps, **{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 4}
            )
        ).config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY]
        == 4
    )
    with pytest.raises(exc.InvalidConfig, match="dstate tile and epilogue warps"):
        spec.normalized_config(
            _config(
                bound, 128, **four_warps, **{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 2}
            )
        )
    repaired = _config(
        bound, 128, **four_warps, **{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 4}
    )
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] == 1
    # Four stages are legal for the 32- and 16-row tiles only.
    assert (
        spec.normalized_config(
            _config(bound, 32, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 4})
        ).config[CUTE_GDN_RECURRENCE_STAGES_KEY]
        == 4
    )
    with pytest.raises(exc.InvalidConfig, match="must be one of .* dstate tile"):
        spec.normalized_config(
            _config(bound, 128, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 4})
        )
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        spec.normalized_config(
            _config(bound, 128, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 5})
        )
    repaired = _config(bound, 128, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 4})
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config[CUTE_GDN_RECURRENCE_STAGES_KEY] == 3
    spec.cute_gdn_recurrence_stages = None
    spec.cute_gdn_recurrence_epilogue_warps = None
    spec.cute_gdn_recurrence_token_groups = None
    spec.cute_gdn_recurrence_mma_m = None
    with pytest.raises(exc.InvalidConfig, match="available only for matched"):
        spec.normalized_config(
            _config(bound, 128, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 3})
        )
    repaired = _config(bound, 128, **{CUTE_GDN_RECURRENCE_STAGES_KEY: 3})
    spec.normalize(repaired, _fix_invalid=True)
    assert CUTE_GDN_RECURRENCE_STAGES_KEY not in repaired.config


def test_gdn_recurrence_small_dstate_seeds_its_own_tile() -> None:
    with _sm100_environment():
        bound = _bind(
            chunk=64, shape={"dhead": 32, "dstate": 32, "seqlen": 256, "heads": 4}
        )
    spec = bound.config_spec  # type: ignore[attr-defined]
    assert spec.cute_gdn_recurrence_stages.choices == (3, 4, 2, 1)
    assert spec.cute_gdn_recurrence_epilogue_warps.choices == (8, 4, 16)
    assert spec.cute_gdn_recurrence_token_groups.choices == (2, 1, 4)
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if CUTE_GDN_RECURRENCE_STAGES_KEY in seed.config
    ]
    # A tile of at most 32 rows spreads one column slice over every
    # sub-partition, so its first seed uses four epilogue warps (the shape's
    # tuned best under the full MMA); every seed takes the 64-row MMA.
    assert seeds[0] == {
        "block_sizes": [32],
        CUTE_GDN_RECURRENCE_STAGES_KEY: 3,
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: 4,
        CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 2,
        CUTE_GDN_RECURRENCE_MMA_M_KEY: 64,
    }
    assert {seed[CUTE_GDN_RECURRENCE_MMA_M_KEY] for seed in seeds} == {64}
    assert [seed[CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY] for seed in seeds[:6]] == [
        4,
        4,
        8,
        8,
        16,
        16,
    ]
    assert {tuple(seed["block_sizes"]) for seed in seeds} == {(32,), (16,)}
    # The preference is a seed-order heuristic only: a config that omits the
    # knob still gets the search domain's default.
    with _sm100_environment():
        code = bound.to_triton_code(helion.Config(block_sizes=[32]))  # type: ignore[attr-defined]
    assert "'epilogue_warps': 8" in code
    assert "'token_groups': 2" in code
    assert "'mma_m': 128" in code


@pytest.mark.parametrize(
    ("chunk", "expected_stage_choices"),
    (
        (
            128,
            {
                (64, 64): (3, 4, 2, 1),
                (64, 128): (3, 2, 1),
                (32, 64): (3, 4, 2, 1),
                (32, 128): (3, 4, 2, 1),
                (16, 64): (3, 4, 2, 1),
                (16, 128): (3, 4, 2, 1),
                (128, 128): (3, 2, 1),
            },
        ),
        (
            256,
            {
                (64, 64): (2, 1),
                (64, 128): (1,),
                (32, 64): (2, 1),
                (32, 128): (1,),
                (16, 64): (2, 1),
                (16, 128): (2, 1),
                (128, 128): (1,),
            },
        ),
    ),
)
def test_gdn_recurrence_every_searched_tile_compiles(
    chunk: int, expected_stage_choices: dict[tuple[int, int], tuple[int, ...]]
) -> None:
    """A 96-wide state: the block search reaches the 128-row tile above it.

    Every tile the search can select has a per-tile depth entry, so a
    normalized config either lowers to the schedule or is rejected up front;
    codegen never raises.
    """

    with _sm100_environment():
        bound = _bind(chunk=chunk, shape={"dstate": 96, "seqlen": 512})
    spec = bound.config_spec  # type: ignore[attr-defined]
    assert spec.block_sizes[0].max_size == 128
    assert spec.cute_gdn_recurrence_stage_choices_by_tile == expected_stage_choices
    union = spec.cute_gdn_recurrence_stages.choices
    assert set(union) == {
        stages for choices in expected_stage_choices.values() for stages in choices
    }
    # The seeds stay on the candidate tiles, which are no wider than dstate.
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if CUTE_GDN_RECURRENCE_STAGES_KEY in seed.config
    ]
    assert {tuple(seed["block_sizes"]) for seed in seeds} == {(64,), (32,)}
    for (block_v, mma_m), legal in expected_stage_choices.items():
        for stages in union:
            for warps in spec.cute_gdn_recurrence_epilogue_warps.choices:
                legal_groups = spec.cute_gdn_recurrence_token_group_choices_by_tile[
                    (block_v, warps, mma_m)
                ]
                overrides = {
                    CUTE_GDN_RECURRENCE_STAGES_KEY: stages,
                    CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: warps,
                    CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY: 4,
                    CUTE_GDN_RECURRENCE_MMA_M_KEY: mma_m,
                }
                # The autotuner path repairs the depth and the token groups
                # to the selected tile, warps and M.
                repaired = _config(bound, block_v, **overrides)
                spec.normalize(repaired, _fix_invalid=True)
                assert repaired.config["block_sizes"] == [block_v]
                assert repaired.config[CUTE_GDN_RECURRENCE_MMA_M_KEY] == mma_m
                assert repaired.config[CUTE_GDN_RECURRENCE_STAGES_KEY] == (
                    stages if stages in legal else legal[0]
                )
                groups = 4 if 4 in legal_groups else legal_groups[0]
                assert repaired.config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] == groups
                with _sm100_environment():
                    code = bound.to_triton_code(repaired)  # type: ignore[attr-defined]
                assert _PLAN_KIND in code
                assert f"'token_groups': {groups}" in code
                assert f"'mma_m': {mma_m}" in code
                # An explicit depth the tile cannot hold (or a group count its
                # warps cannot split) is rejected up front.
                explicit = _config(bound, block_v, **overrides)
                if stages in legal and 4 in legal_groups:
                    normalized = spec.normalized_config(explicit).config
                    assert normalized[CUTE_GDN_RECURRENCE_STAGES_KEY] == stages
                else:
                    with pytest.raises(
                        exc.InvalidConfig, match="for the selected dstate tile"
                    ):
                        spec.normalized_config(explicit)
    # A config that leaves the depth to the default follows the tile (and
    # the full MMA it defaults to).
    partial = helion.Config(block_sizes=[128])
    assert (
        spec.normalized_config(partial).config[CUTE_GDN_RECURRENCE_STAGES_KEY]
        == expected_stage_choices[(128, 128)][0]
    )
    with _sm100_environment():
        assert _PLAN_KIND in bound.to_triton_code(partial)  # type: ignore[attr-defined]


def test_gdn_recurrence_declines_tiles_outside_the_schedule() -> None:
    """A 200-wide state: the block search reaches a 256-row tile.

    The schedule has no 256-row tile, so the depth check is inert for it and
    the kernel takes the SIMT lowering instead of raising.
    """

    shape = {"dstate": 200, "seqlen": 512}
    with _sm100_environment():
        bound = _bind(chunk=128, shape=shape)
    spec = bound.config_spec  # type: ignore[attr-defined]
    assert spec.block_sizes[0].max_size == 256
    assert {
        tile for tile, _mma_m in spec.cute_gdn_recurrence_stage_choices_by_tile
    } == {128, 64, 32, 16}
    for stages in spec.cute_gdn_recurrence_stages.choices:
        config = _config(bound, 256, **{CUTE_GDN_RECURRENCE_STAGES_KEY: stages})
        normalized = spec.normalized_config(config).config
        assert normalized["block_sizes"] == [256]
        assert normalized[CUTE_GDN_RECURRENCE_STAGES_KEY] == stages
        with _sm100_environment():
            code = bound.to_triton_code(config)  # type: ignore[attr-defined]
        assert _PLAN_KIND not in code
    assert _captured_plan(chunk=128, block_v=256, shape=shape) is None


def test_gdn_recurrence_unmatched_kernel_has_no_knobs() -> None:
    with _sm100_environment():
        bound = _bind(chunk=128, shape={"dhead": 128})
    spec = bound.config_spec  # type: ignore[attr-defined]
    assert spec.cute_gdn_recurrence_stages is None
    assert spec.cute_gdn_recurrence_epilogue_warps is None
    assert spec.cute_gdn_recurrence_token_groups is None
    assert spec.cute_gdn_recurrence_mma_m is None
    assert not any(
        CUTE_GDN_RECURRENCE_STAGES_KEY in seed.config
        for seed in spec.compiler_seed_configs
    )


def test_gdn_recurrence_raises_the_dstate_tile_floor() -> None:
    """A matched kernel never samples a dstate tile below 16 rows.

    On SM100 the dot minimum is ``(1, 1, 16)`` and leaves the dstate tile's
    spec at one row, while no lowering of this kernel compiles a narrower
    tile; the CPU environment's dot minimum of 16 would hide that, so the
    real SM100 value is used here.
    """

    sm100_dot_minimum = patch("helion._compat._min_dot_size", return_value=(1, 1, 16))
    with _sm100_environment(), sm100_dot_minimum:
        bound = _bind(
            chunk=64, shape={"dhead": 32, "dstate": 32, "seqlen": 256, "heads": 4}
        )
        narrow = _bind(
            chunk=16, shape={"dhead": 16, "dstate": 8, "seqlen": 64, "heads": 2}
        )
        unmatched = _bind(chunk=128, shape={"dhead": 128})
    spec = bound.config_spec  # type: ignore[attr-defined]
    block_spec = spec.block_sizes.block_id_lookup(
        spec.cute_gdn_recurrence_value_block_id
    )
    assert (block_spec.min_size, block_spec.max_size) == (GDN_ADMITTED_BLOCK_V[-1], 32)
    # A state narrower than the smallest tile gets that tile (rows past
    # dstate are padding); an unmatched kernel keeps the dot minimum's floor.
    narrow_spec = narrow.config_spec.block_sizes[0]  # type: ignore[attr-defined]
    assert (narrow_spec.min_size, narrow_spec.max_size) == (16, 16)
    assert unmatched.config_spec.block_sizes[0].min_size == 1  # type: ignore[attr-defined]
    random.seed(20260930)
    tiles = {
        spec.normalized_config(
            spec.flat_config(lambda fragment: fragment.random())
        ).config["block_sizes"][0]
        for _ in range(200)
    }
    assert tiles == {16, 32}
    # An explicit tile below the floor is clamped to it, not compiled.
    assert spec.normalized_config(helion.Config(block_sizes=[8])).config[
        "block_sizes"
    ] == [16]


def test_gdn_recurrence_geometry_helpers() -> None:
    layout = gdn_tmem_layout(64, 128)
    assert layout is not None
    assert (
        layout.state_col,
        layout.state_image_col,
        layout.acc_col,
        layout.update_image_col,
        layout.alloc_cols,
    ) == (0, 64, 96, 224, 512)
    assert gdn_tmem_layout(64, 256).alloc_cols == 512  # type: ignore[union-attr]
    assert gdn_tmem_layout(16, 64).alloc_cols == 256  # type: ignore[union-attr]
    assert gdn_tmem_layout(128, 256) is None
    assert gdn_stage_choices(128, 64, 128, 128) == (3, 2, 1)
    assert gdn_stage_choices(256, 64, 128, 128) == (1,)
    assert gdn_stage_choices(64, 64, 128, 128) == (3, 4, 2, 1)
    # Replicated tiles keep the bf16 update image in shared memory; under
    # M = 64 the 64-row tile has no replica and the smaller tiles half the
    # image, so the ring can go deeper.
    assert gdn_stage_choices(128, 64, 64, 128) == (3, 2, 1)
    assert gdn_stage_choices(256, 64, 64, 128) == (1,)
    assert gdn_stage_choices(128, 64, 64, 64) == (3, 4, 2, 1)
    assert gdn_stage_choices(256, 64, 64, 64) == (2, 1)
    assert gdn_stage_choices(256, 64, 32, 128) == (1,)
    assert gdn_stage_choices(256, 64, 32, 64) == (2, 1)
    assert gdn_epilogue_warp_choices(128, 64) == (8, 4, 16)
    assert gdn_epilogue_warp_choices(64, 64) == (8, 4, 16)
    assert gdn_epilogue_warp_choices(16, 16) == (8, 4, 16)
    # The 64-row MMA needs a tile of at most 64 rows and a warp's state slice
    # in whole eight-column groups (the 16-lane data path's granularity).
    assert gdn_mma_m_choices(128, 64, 128, 8) == (128,)
    assert gdn_mma_m_choices(128, 64, 64, 8) == (64, 128)
    assert gdn_mma_m_choices(64, 32, 32, 4) == (64, 128)
    assert gdn_mma_m_choices(64, 16, 32, 8) == (64, 128)
    assert gdn_mma_m_choices(64, 16, 32, 16) == (128,)
    assert gdn_mma_m_choices(16, 16, 16, 4) == (64, 128)
    assert gdn_half_lanes(64) and not gdn_half_lanes(128)
    assert [gdn_state_replicas(block_v, 128) for block_v in (16, 32, 64, 128)] == [
        4,
        4,
        2,
        1,
    ]
    assert [gdn_state_replicas(block_v, 64) for block_v in (16, 32, 64)] == [4, 2, 1]
    assert gdn_update_tile_bytes(64, 32, 128) == 128 * 64 * 2
    assert gdn_update_tile_bytes(256, 64, 128) == 128 * 256 * 2
    assert gdn_update_tile_bytes(128, 128, 128) == 0
    assert gdn_update_tile_bytes(64, 32, 64) == 64 * 64 * 2
    assert gdn_update_tile_bytes(128, 64, 64) == 0
    assert gdn_smem_bytes(64, 32, 32, 3, 128) == (
        3 * gdn_stage_smem_bytes(64, 32, 32) + 3 * 4 + 128 * 64 * 2 + 1024
    )
    assert gdn_smem_bytes(64, 32, 32, 3, 64) == (
        3 * gdn_stage_smem_bytes(64, 32, 32) + 3 * 4 + 64 * 64 * 2 + 1024
    )
    # Every replica's warps take a slice of the chunk's tokens (at least a
    # bf16 pair each); extra warps only keep the state and h.
    assert gdn_token_splits(64, 32, 4, 128) == 4
    assert gdn_tokens_per_split(64, 32, 4, 128) == 16
    assert gdn_token_splits(64, 32, 16, 128) == 16
    assert gdn_tokens_per_split(64, 32, 16, 128) == 4
    assert gdn_token_splits(16, 16, 16, 128) == 8
    assert gdn_tokens_per_split(16, 16, 16, 128) == 2
    assert gdn_token_splits(128, 128, 8, 128) == 2
    assert gdn_tokens_per_split(128, 128, 8, 128) == 64
    assert gdn_token_splits(64, 64, 4, 128) == 2
    # Under M = 64 a slice is at least one eight-column repetition of the
    # 16-lane data path, and fewer replicas share the tokens.
    assert gdn_token_splits(128, 32, 8, 64) == 4
    assert gdn_tokens_per_split(128, 32, 8, 64) == 32
    assert gdn_token_splits(64, 32, 4, 64) == 2
    assert gdn_tokens_per_split(64, 32, 4, 64) == 32
    assert gdn_token_splits(16, 32, 8, 64) == 2
    assert gdn_tokens_per_split(16, 16, 16, 64) == 8
    assert gdn_token_splits(128, 64, 8, 64) == 2
    assert gdn_tokens_per_split(128, 64, 8, 64) == 64
    assert gdn_acc_load_cols(128, 128, 8, 128) == 64
    assert gdn_acc_load_cols(256, 128, 4, 128) == 64
    assert gdn_acc_load_cols(64, 64, 16, 128) == 8
    assert gdn_acc_load_cols(16, 16, 16, 128) == 2
    assert gdn_acc_load_cols(128, 32, 8, 64) == 32
    assert gdn_acc_load_cols(256, 64, 4, 64) == 64
    # Sixteen epilogue warps run under the 96/80-register cap: half-size
    # accumulator blocks and half the prefetched tokens.  Under M = 64 a
    # thread holds two rows of half the lane columns, so twice the lane
    # columns fill the same registers.
    assert gdn_acc_load_cols(256, 128, 16, 128) == 32
    assert gdn_acc_load_cols(256, 64, 16, 128) == 32
    assert gdn_acc_load_cols(256, 64, 8, 128) == 64
    assert gdn_prefetch_tokens(128, 32, 8, 128) == 16
    assert gdn_prefetch_tokens(256, 128, 16, 128) == 8
    assert gdn_prefetch_tokens(64, 32, 16, 128) == 4
    assert gdn_prefetch_tokens(16, 16, 16, 128) == 2
    assert gdn_prefetch_tokens(128, 32, 8, 64) == 32
    assert gdn_prefetch_tokens(256, 64, 16, 64) == 16
    assert gdn_prefetch_tokens(16, 32, 8, 64) == 8
    # Token groups: whole K steps, whole computing warps, every warp computing.
    assert gdn_token_group_choices(128, 128, 8, 128) == (2, 1)
    assert gdn_token_group_choices(128, 128, 4, 128) == (1,)
    assert gdn_token_group_choices(128, 128, 16, 128) == (2, 1, 4)
    assert gdn_token_group_choices(64, 32, 4, 128) == (2, 1, 4)
    assert gdn_token_group_choices(64, 64, 8, 128) == (2, 1, 4)
    assert gdn_token_group_choices(32, 32, 4, 128) == (2, 1)
    assert gdn_token_group_choices(16, 16, 16, 128) == (1,)
    assert gdn_token_group_choices(256, 128, 8, 128) == (2, 1)
    assert gdn_token_group_choices(128, 64, 8, 64) == (2, 1)
    assert gdn_token_group_choices(128, 64, 4, 64) == (1,)
    assert gdn_token_group_choices(128, 32, 8, 64) == (2, 1, 4)
    assert gdn_token_group_choices(64, 32, 4, 64) == (2, 1)
    assert gdn_token_group_choices(16, 32, 8, 64) == (1,)
    # Epilogue warps, four h store warps for a replicated tile, the TMA warp.
    assert [gdn_cta_warps(32, warps, 128) for warps in (4, 8, 16)] == [9, 13, 21]
    assert [gdn_cta_warps(128, warps, 128) for warps in (4, 8, 16)] == [5, 9, 17]
    assert [gdn_cta_warps(64, warps, 64) for warps in (4, 8, 16)] == [5, 9, 17]
    assert [gdn_cta_warps(32, warps, 64) for warps in (4, 8, 16)] == [9, 13, 21]
    assert gdn_store_warps(64, 128) == 4
    assert gdn_store_warps(128, 128) == 0
    assert gdn_store_warps(64, 64) == 0
    assert gdn_store_warps(16, 64) == 4
    assert gdn_candidate_block_sizes(128) == (128, 64, 32, 16)
    assert gdn_candidate_block_sizes(256) == (128, 64, 32, 16)
    assert gdn_candidate_block_sizes(96) == (64, 32, 16)
    assert gdn_candidate_block_sizes(32) == (32, 16)
    assert gdn_candidate_block_sizes(8) == (16,)
    assert gdn_smem_bytes(128, 64, 128, 3, 128) <= 227 * 1024
    assert gdn_smem_bytes(128, 64, 128, 4, 128) > 227 * 1024
    assert gdn_shape_admitted(dhead=64, chunk=128, dstate=128, block_v=128)
    assert gdn_shape_admitted(dhead=64, chunk=256, dstate=256, block_v=128)
    assert gdn_shape_admitted(dhead=32, chunk=64, dstate=32, block_v=32)
    assert gdn_shape_admitted(dhead=16, chunk=64, dstate=32, block_v=32)
    assert gdn_shape_admitted(dhead=64, chunk=128, dstate=192, block_v=128)
    assert gdn_shape_admitted(dhead=16, chunk=16, dstate=16, block_v=16)
    # The smallest tile may exceed dstate; TMA zero-fills the rows past it.
    assert gdn_shape_admitted(dhead=16, chunk=16, dstate=8, block_v=16)
    assert not gdn_shape_admitted(dhead=64, chunk=128, dstate=100, block_v=64)
    assert not gdn_shape_admitted(dhead=32, chunk=64, dstate=36, block_v=32)
    assert not gdn_shape_admitted(dhead=128, chunk=128, dstate=128, block_v=128)
    assert not gdn_shape_admitted(dhead=64, chunk=96, dstate=128, block_v=128)
    assert not gdn_shape_admitted(dhead=64, chunk=8, dstate=128, block_v=128)
    assert not gdn_shape_admitted(dhead=64, chunk=128, dstate=128, block_v=256)


def _wrapper_plan(**overrides: object) -> dict[str, object]:
    layout = gdn_tmem_layout(64, 128)
    assert layout is not None
    plan: dict[str, object] = {
        "kind": "gdn_recurrence_sm100",
        "k_idx": 0,
        "w_idx": 1,
        "u_idx": 2,
        "g_idx": 3,
        "h_idx": 4,
        "batch": 2,
        "heads": 8,
        "seqlen": 1024,
        "chunk": 128,
        "dhead": 64,
        "dstate": 128,
        "num_chunks": 8,
        "block_v": 128,
        "epilogue_warps": 8,
        "stages": 3,
        "token_groups": 2,
        "mma_m": 128,
        "threads": 288,
        "tmem_cols": layout.alloc_cols,
        "smem_bytes": gdn_smem_bytes(128, 64, 128, 3, 128),
        "state_col": layout.state_col,
        "state_image_col": layout.state_image_col,
        "acc_col": layout.acc_col,
        "update_image_col": layout.update_image_col,
        "device_abi": 5,
    }
    plan.update(overrides)
    return plan


def _tensor_schema(dtype: torch.dtype, shape: tuple[int, ...]) -> tuple[object, ...]:
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return ("tensor", str(dtype), len(shape), shape, tuple(reversed(strides)))


def _helion_imports(relative: str) -> set[str]:
    """Package-relative paths of the Helion modules ``relative`` imports.

    Follows the relative ``from . import`` statements transitively (type-only
    imports excluded), so a new Helion import in the schedule shows up here.
    """

    root = Path(cast("str", helion.__file__)).resolve().parent
    pending = [relative]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        package = Path(current).parent
        statements = list(ast.parse((root / current).read_text()).body)
        while statements:
            statement = statements.pop()
            if isinstance(statement, ast.If):
                test = statement.test
                if not (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"):
                    statements.extend(statement.body)
                statements.extend(statement.orelse)
                continue
            if not isinstance(statement, ast.ImportFrom) or not statement.level:
                continue
            base = package
            for _ in range(statement.level - 1):
                base = base.parent
            module = base.joinpath(*(statement.module or "").split("."))
            candidates = (
                [module.with_suffix(".py")]
                if statement.module
                else [module / f"{alias.name}.py" for alias in statement.names]
            )
            for candidate in candidates:
                if (root / candidate).is_file():
                    pending.append(candidate.as_posix())
    return seen


def test_gdn_recurrence_schedule_sources_fingerprint_the_disk_cache() -> None:
    """Editing the schedule must change the on-disk compile cache key.

    The generated wrapper only imports the schedule, so the launcher's key has
    to hash the schedule sources themselves or a schedule fix would reload
    obsolete machine code from an earlier build.  The expected set follows
    the schedule's imports, so a new Helion import needs a table entry too.
    """

    from helion.runtime.cute import source_dependencies

    dependencies = source_dependencies._WRAPPER_DEPENDENCIES["gdn_recurrence_sm100"]
    imported = _helion_imports("_compiler/cute/gdn_recurrence_sm100.py")
    assert imported >= {
        "_compiler/cute/gdn_recurrence_sm100.py",
        "_compiler/cute/gdn_recurrence_geometry.py",
        "_compiler/cute/affine_recurrence_primitives.py",
    }
    assert set(dependencies) | set(source_dependencies._COMMON_DEPENDENCIES) >= imported
    assert set(dependencies) >= imported - set(source_dependencies._COMMON_DEPENDENCIES)
    fingerprints = source_dependencies.wrapper_source_dependencies(
        ("gdn_recurrence_sm100",)
    )
    assert fingerprints is not None
    assert {relative for relative, _digest in fingerprints} >= set(dependencies)


def test_gdn_recurrence_wrapper_calls_external_host_schedule() -> None:
    from helion.runtime.cute.launcher import _create_cute_wrapper

    def dummy_kernel(*_args: object) -> None:
        pass

    dummy_kernel._helion_cute_wrapper_plans = [_wrapper_plan()]  # type: ignore[attr-defined]
    schema = (
        _tensor_schema(torch.bfloat16, (2, 1024, 8, 64)),
        _tensor_schema(torch.bfloat16, (2, 1024, 8, 64)),
        _tensor_schema(torch.bfloat16, (2, 1024, 8, 128)),
        _tensor_schema(torch.float32, (2, 1024, 8)),
        _tensor_schema(torch.bfloat16, (2, 8, 8, 64, 128)),
        ("scalar_constexpr", "int", 128, 128),
    )
    wrapper = _create_cute_wrapper(dummy_kernel, schema, (288, 1, 1))
    source = inspect.getsource(cast("Callable[..., object]", wrapper))
    assert "_helion_cute_kernel_tag = 'gdn_recurrence_sm100'" in source
    assert (
        "_helion_gdn_recurrence_host(arg0, arg1, arg2, arg3, arg4, stream, "
        "128, 128, 64, 8, 3, 2, 128, 0, 64, 96, 224, 512)"
    ) in source
    assert "_kernel(" not in source


@pytest.mark.parametrize(
    "overrides",
    (
        {"threads": 416},
        {"device_abi": 2},
        {"block_v": 256},
        {"stages": 4},
        {"epilogue_warps": 12},
        {"token_groups": 4},
        {"token_groups": 3},
        {"mma_m": 64},
        {"mma_m": 32},
        {"num_chunks": 7},
        {"acc_col": 128},
        {"dhead": 128},
        {"chunk": 96},
        {"dstate": 100},
        {"smem_bytes": 4096},
    ),
)
def test_gdn_recurrence_wrapper_rejects_invalid_abi(
    overrides: dict[str, object],
) -> None:
    from helion.runtime.cute.launcher import _validate_gdn_recurrence_plan

    _validate_gdn_recurrence_plan(_wrapper_plan())
    # The 64-row MMA on the 64-row tile: no store warps, no update image.
    _validate_gdn_recurrence_plan(
        _wrapper_plan(
            block_v=64,
            mma_m=64,
            threads=288,
            smem_bytes=gdn_smem_bytes(128, 64, 64, 3, 64),
        )
    )
    with pytest.raises(exc.BackendUnsupported, match="gdn recurrence"):
        _validate_gdn_recurrence_plan(_wrapper_plan(**overrides))


def _fake_pointer_patches(
    args: tuple[torch.Tensor, ...],
) -> tuple[AbstractContextManager[object], AbstractContextManager[object]]:
    storage_bases = {
        storage._cdata: 0x10_0000 + index * 0x100_0000
        for index, storage in enumerate({tensor.untyped_storage() for tensor in args})
    }

    def storage_data_ptr(storage: object) -> int:
        return storage_bases[storage._cdata]  # type: ignore[attr-defined]

    def tensor_data_ptr(tensor: FakeTensor) -> int:
        return int(
            storage_bases[tensor.untyped_storage()._cdata]
            + tensor.storage_offset() * tensor.element_size()
        )

    return (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
    )


def test_gdn_recurrence_runtime_args_must_match_the_compiled_geometry() -> None:
    from helion.runtime.cute.launcher import _validate_gdn_recurrence_launch_args

    plan = _wrapper_plan()
    with FakeTensorMode():
        k = torch.empty(2, 1024, 8, 64, dtype=torch.bfloat16, device=_FAKE_CUDA)
        w = torch.empty_like(k)
        u = torch.empty(2, 1024, 8, 128, dtype=torch.bfloat16, device=_FAKE_CUDA)
        g = torch.empty(2, 1024, 8, dtype=torch.float32, device=_FAKE_CUDA)
        h = torch.empty(2, 8, 8, 64, 128, dtype=torch.bfloat16, device=_FAKE_CUDA)
        wrong_seqlen = torch.empty(
            2, 512, 8, 64, dtype=torch.bfloat16, device=_FAKE_CUDA
        )
        fp16 = torch.empty(2, 1024, 8, 64, dtype=torch.float16, device=_FAKE_CUDA)
        shared = torch.empty(
            2 * 1024 * 8 * 128, dtype=torch.bfloat16, device=_FAKE_CUDA
        )
        misaligned = torch.empty(
            2 * 1024 * 8 * 64 + 8, dtype=torch.bfloat16, device=_FAKE_CUDA
        )
    tensors = (k, w, u, g, h, wrong_seqlen, fp16, shared, misaligned)
    storage_patch, tensor_patch = _fake_pointer_patches(tensors)
    with storage_patch, tensor_patch:
        _validate_gdn_recurrence_launch_args(plan, (k, w, u, g, h, 128))
        with pytest.raises(exc.BackendUnsupported, match="k_idx"):
            _validate_gdn_recurrence_launch_args(plan, (wrong_seqlen, w, u, g, h, 128))
        with pytest.raises(exc.BackendUnsupported, match="w_idx"):
            _validate_gdn_recurrence_launch_args(plan, (k, fp16, u, g, h, 128))
        with pytest.raises(exc.BackendUnsupported, match="u_idx"):
            _validate_gdn_recurrence_launch_args(
                plan, (k, w, u.transpose(1, 2), g, h, 128)
            )
        with pytest.raises(exc.BackendUnsupported, match="must not alias"):
            _validate_gdn_recurrence_launch_args(
                plan,
                (
                    k,
                    w,
                    shared.view(2, 1024, 8, 128),
                    g,
                    shared[: 2 * 8 * 8 * 64 * 128].view(2, 8, 8, 64, 128),
                    128,
                ),
            )
        with pytest.raises(exc.BackendUnsupported, match="misaligned"):
            _validate_gdn_recurrence_launch_args(
                plan,
                (misaligned[1 : 1 + k.numel()].view(2, 1024, 8, 64), w, u, g, h, 128),
            )


def _require_sm100_runtime() -> torch.device:
    from helion._compat import requires_cuda_version

    if not torch.cuda.is_available() or not requires_cuda_version("13"):
        pytest.skip("gdn recurrence runtime test requires CUDA >= 13")
    pytest.importorskip("cutlass.cute")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 10:
        pytest.skip("gdn recurrence runtime test requires SM100")
    return device


def _runtime_inputs(
    device: torch.device,
    *,
    batch: int,
    heads: int,
    seqlen: int,
    chunk: int,
    dhead: int,
    dstate: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, dtype=torch.float32, device=device
        )

    k = torch.nn.functional.rms_norm(randn(batch, seqlen, heads, dhead), (dhead,))
    padded = -(-seqlen // chunk) * chunk
    w = randn(batch, padded // chunk, chunk, heads, dhead)
    wu, _, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
    w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
    w = w.permute(0, 1, 3, 2, 4).reshape(batch, padded, heads, dhead)[:, :seqlen]
    u = torch.nn.functional.rms_norm(randn(batch, seqlen, heads, dstate), (dstate,))
    g = torch.cumsum(
        0.5
        * math.log(1 / dhead)
        * torch.rand(batch, seqlen, heads, generator=generator, device=device),
        dim=1,
    )
    return (
        k.to(torch.bfloat16),
        w.contiguous().to(torch.bfloat16),
        u.to(torch.bfloat16),
        g,
        chunk,
    )


def _torch_reference(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk: int
) -> torch.Tensor:
    """fp32 torch chunk recurrence for any ``dstate`` and a ragged ``seqlen``.

    ``ref_gdn_fwd_h`` in the example needs ``seqlen`` divisible by ``chunk``
    and ``dstate`` a multiple of ``dhead``.
    """

    batch, seqlen, heads, dhead = k.shape
    dstate = u.shape[-1]
    num_chunks = -(-seqlen // chunk)
    h = torch.empty(
        batch, num_chunks, heads, dhead, dstate, dtype=k.dtype, device=k.device
    )
    state = torch.zeros(
        batch, heads, dhead, dstate, dtype=torch.float32, device=k.device
    )
    for index in range(num_chunks):
        begin, end = index * chunk, min((index + 1) * chunk, seqlen)
        h[:, index] = state.to(k.dtype)
        projection = torch.einsum(
            "bthk,bhkv->bthv", w[:, begin:end].float(), state.to(k.dtype).float()
        )
        g_last = g[:, end - 1]
        gate = torch.exp(g_last[:, None, :] - g[:, begin:end])
        update = ((u[:, begin:end].float() - projection) * gate[..., None]).to(k.dtype)
        state = state * torch.exp(g_last)[:, :, None, None] + torch.einsum(
            "bthk,bthv->bhkv", k[:, begin:end].float(), update.float()
        )
    return h


def _tensor_core_config(
    block_v: int,
    epilogue_warps: int,
    stages: int,
    token_groups: int | None = None,
    mma_m: int | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "block_sizes": [block_v],
        CUTE_GDN_RECURRENCE_STAGES_KEY: stages,
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY: epilogue_warps,
    }
    if token_groups is not None:
        config[CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY] = token_groups
    if mma_m is not None:
        config[CUTE_GDN_RECURRENCE_MMA_M_KEY] = mma_m
    return config


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "shape"),
    (
        # The study workload and its siblings (chunk 128 / dhead 64 / dstate 128,
        # chunk 64 / dhead 64 / dstate 64, chunk 64 / dhead 32 / dstate 32).
        (
            128,
            8,
            3,
            {
                "batch": 2,
                "heads": 3,
                "seqlen": 512,
                "chunk": 128,
                "dhead": 64,
                "dstate": 128,
            },
        ),
        (
            64,
            8,
            3,
            {
                "batch": 1,
                "heads": 8,
                "seqlen": 512,
                "chunk": 64,
                "dhead": 64,
                "dstate": 64,
            },
        ),
        (
            32,
            8,
            3,
            {
                "batch": 1,
                "heads": 4,
                "seqlen": 256,
                "chunk": 64,
                "dhead": 32,
                "dstate": 32,
            },
        ),
        # The example's own test shape (dhead 16) and the remaining knob values.
        (
            32,
            4,
            2,
            {
                "batch": 2,
                "heads": 4,
                "seqlen": 512,
                "chunk": 64,
                "dhead": 16,
                "dstate": 32,
            },
        ),
        (
            128,
            16,
            2,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 512,
                "chunk": 64,
                "dhead": 64,
                "dstate": 128,
            },
        ),
        (
            128,
            4,
            1,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 384,
                "chunk": 128,
                "dhead": 64,
                "dstate": 256,
            },
        ),
        (
            128,
            8,
            1,
            {
                "batch": 1,
                "heads": 1,
                "seqlen": 512,
                "chunk": 256,
                "dhead": 64,
                "dstate": 128,
            },
        ),
        # Padded dstate tiles (64-row tile of a 128 state) and a dstate tail.
        (
            64,
            8,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 256,
                "chunk": 128,
                "dhead": 64,
                "dstate": 128,
            },
        ),
        (
            128,
            8,
            2,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 256,
                "chunk": 128,
                "dhead": 64,
                "dstate": 192,
            },
        ),
        (
            16,
            4,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 256,
                "chunk": 32,
                "dhead": 16,
                "dstate": 48,
            },
        ),
        # 16-token chunks fill only half of the gate producer warp.
        (
            32,
            8,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 224,
                "chunk": 16,
                "dhead": 16,
                "dstate": 32,
            },
        ),
        (
            16,
            16,
            4,
            {
                "batch": 1,
                "heads": 1,
                "seqlen": 224,
                "chunk": 16,
                "dhead": 16,
                "dstate": 16,
            },
        ),
        # Replicated tiles: four replicas splitting 64 tokens four ways (the
        # study's shape 0 with its tuned config), two replicas with a single
        # 64-token slice per warp, a 16-token chunk where only eight of the
        # sixteen warps get a token pair and h splits down to one column, and
        # the four-token slices written as two-word stores.
        (
            32,
            4,
            3,
            {
                "batch": 1,
                "heads": 4,
                "seqlen": 256,
                "chunk": 64,
                "dhead": 32,
                "dstate": 32,
            },
        ),
        (
            64,
            4,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 256,
                "chunk": 128,
                "dhead": 64,
                "dstate": 64,
            },
        ),
        (
            16,
            16,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 96,
                "chunk": 16,
                "dhead": 16,
                "dstate": 16,
            },
        ),
        (
            16,
            4,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 96,
                "chunk": 16,
                "dhead": 16,
                "dstate": 16,
            },
        ),
    ),
)
def test_gdn_recurrence_matches_reference_cuda(
    block_v: int, epilogue_warps: int, stages: int, shape: dict[str, int]
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(device, seed=20260929, **shape)
    expected = module.ref_gdn_fwd_h(*args)  # type: ignore[attr-defined]
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages),
    )
    assert _PLAN_KIND in code
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "token_groups", "shape"),
    (
        # Every pipelining depth of the three campaign shapes, plus the full
        # 128-row tile split four ways over sixteen warps.
        (32, 4, 1, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (32, 4, 2, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (32, 4, 4, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (32, 8, 4, {"seqlen": 512, "chunk": 64, "dhead": 64, "dstate": 64}),
        (64, 8, 2, {"seqlen": 512, "chunk": 64, "dhead": 64, "dstate": 64}),
        (32, 8, 4, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        (128, 8, 2, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        (128, 16, 4, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        (64, 4, 2, {"seqlen": 384, "chunk": 128, "dhead": 64, "dstate": 64}),
    ),
)
def test_gdn_recurrence_token_groups_match_reference_cuda(
    block_v: int, epilogue_warps: int, token_groups: int, shape: dict[str, int]
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(device, seed=20260930, batch=1, heads=2, **shape)
    expected = module.ref_gdn_fwd_h(*args)  # type: ignore[attr-defined]
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, 3, token_groups),
    )
    assert _PLAN_KIND in code
    assert f"'token_groups': {token_groups}" in code
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "token_groups", "shape"),
    (
        # The three campaign shapes on the 64-row MMA: two replicas of a
        # 32-row tile, four of a 16-row tile, none for the 64-row tile (its
        # update image lives in TMEM and its epilogue warps store h).
        (32, 8, 3, 2, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (32, 4, 4, 2, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (16, 4, 3, 2, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (16, 16, 3, 1, {"seqlen": 256, "chunk": 64, "dhead": 32, "dstate": 32}),
        (64, 8, 4, 2, {"seqlen": 512, "chunk": 64, "dhead": 64, "dstate": 64}),
        (64, 4, 3, 1, {"seqlen": 512, "chunk": 64, "dhead": 64, "dstate": 64}),
        (64, 16, 3, 4, {"seqlen": 512, "chunk": 64, "dhead": 64, "dstate": 64}),
        (32, 8, 3, 4, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        (64, 8, 3, 2, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        (32, 16, 3, 4, {"seqlen": 512, "chunk": 128, "dhead": 64, "dstate": 128}),
        # dhead 16: one eight-column state group per warp with eight warps,
        # two with four.
        (32, 8, 2, 2, {"seqlen": 512, "chunk": 64, "dhead": 16, "dstate": 32}),
        (32, 4, 2, 2, {"seqlen": 512, "chunk": 64, "dhead": 16, "dstate": 32}),
        # 16-token chunks: one eight-token slice per computing warp, the
        # other warps only keep the state.
        (32, 8, 3, 1, {"seqlen": 224, "chunk": 16, "dhead": 16, "dstate": 32}),
        (16, 8, 4, 1, {"seqlen": 224, "chunk": 16, "dhead": 16, "dstate": 16}),
        # Padded and partial tiles: an 8-wide state under a 16-row tile, a
        # 48-wide state in 16-row tiles, a 200-wide state with an 8-row tail.
        (16, 8, 3, 1, {"seqlen": 200, "chunk": 16, "dhead": 16, "dstate": 8}),
        (16, 4, 3, 1, {"seqlen": 256, "chunk": 32, "dhead": 16, "dstate": 48}),
        (64, 8, 3, 2, {"seqlen": 256, "chunk": 128, "dhead": 64, "dstate": 200}),
        # Ragged and single-chunk sequences.
        (64, 8, 3, 2, {"seqlen": 300, "chunk": 128, "dhead": 64, "dstate": 64}),
        (32, 4, 3, 1, {"seqlen": 64, "chunk": 64, "dhead": 32, "dstate": 32}),
        (64, 8, 3, 2, {"seqlen": 100, "chunk": 128, "dhead": 64, "dstate": 64}),
        # 256-token chunks: the deepest ring the 64-row tile fits, a single
        # 256-token slice per warp (a rolled sixteen-step update group), the
        # register-capped sixteen warps and a four-replica 16-row tile.
        (64, 8, 2, 2, {"seqlen": 1024, "chunk": 256, "dhead": 64, "dstate": 128}),
        (64, 4, 1, 1, {"seqlen": 768, "chunk": 256, "dhead": 64, "dstate": 128}),
        (64, 16, 1, 1, {"seqlen": 768, "chunk": 256, "dhead": 64, "dstate": 128}),
        (16, 16, 2, 4, {"seqlen": 768, "chunk": 256, "dhead": 64, "dstate": 16}),
    ),
)
def test_gdn_recurrence_mma_m64_matches_reference_cuda(
    block_v: int,
    epilogue_warps: int,
    stages: int,
    token_groups: int,
    shape: dict[str, int],
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(device, seed=20260937, batch=1, heads=2, **shape)
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages, token_groups, 64),
    )
    assert _PLAN_KIND in code
    assert "'mma_m': 64" in code
    expected = _torch_reference(*args)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


def test_gdn_recurrence_ragged_seqlen_matches_simt_cuda() -> None:
    """A ragged last chunk is never computed; only its ``h`` (the state
    entering it) is stored, and the chunk count must still cover it."""

    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(
        device,
        batch=1,
        heads=2,
        seqlen=448,
        chunk=128,
        dhead=64,
        dstate=256,
        seed=20260930,
    )
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(128, 8, 2),
    )
    assert _PLAN_KIND in code
    # A 256-row dstate tile is not admitted, so this is the independent SIMT
    # lowering of the same kernel (the block-size spec clamps tiles below 16).
    simt_code, simt = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        block_sizes=[256],
    )
    assert _PLAN_KIND not in simt_code
    expected = _torch_reference(*args)
    torch.testing.assert_close(simt.float(), expected.float(), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(result.float(), simt.float(), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "shape"),
    (
        # A single chunk: nothing is computed, so the ring, the store warps
        # and the state waits run zero times and only ``h[:, 0] = 0`` is
        # stored; a ragged single chunk does the same.
        (32, 4, 3, {"seqlen": 64, "chunk": 64, "dhead": 32, "dstate": 32}),
        (64, 8, 3, {"seqlen": 100, "chunk": 128, "dhead": 64, "dstate": 64}),
        # Fewer computed chunks than ring stages: the prologue commits empty
        # cp.async groups past the last chunk.
        (32, 4, 3, {"seqlen": 128, "chunk": 64, "dhead": 32, "dstate": 32}),
        (128, 8, 3, {"seqlen": 384, "chunk": 128, "dhead": 64, "dstate": 128}),
    ),
)
def test_gdn_recurrence_few_chunks_cuda(
    block_v: int, epilogue_warps: int, stages: int, shape: dict[str, int]
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(device, batch=1, heads=2, seed=20260934, **shape)
    assert -(-shape["seqlen"] // shape["chunk"]) <= stages
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages),
    )
    assert _PLAN_KIND in code
    expected = _torch_reference(*args)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "shape"),
    (
        # The deepest ring each geometry admits: one more stage would not fit.
        (64, 8, 3, {"seqlen": 1024, "chunk": 256, "dhead": 16, "dstate": 64}),
        (32, 8, 3, {"seqlen": 1024, "chunk": 256, "dhead": 32, "dstate": 32}),
        (16, 8, 2, {"seqlen": 1024, "chunk": 256, "dhead": 64, "dstate": 16}),
    ),
)
def test_gdn_recurrence_tightest_smem_budget_cuda(
    block_v: int, epilogue_warps: int, stages: int, shape: dict[str, int]
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    assert gdn_max_stages(shape["chunk"], shape["dhead"], block_v, GDN_MMA_M) == stages
    assert (
        gdn_smem_bytes(shape["chunk"], shape["dhead"], block_v, stages, GDN_MMA_M)
        <= 227 * 1024
    )
    assert gdn_smem_bytes(
        shape["chunk"], shape["dhead"], block_v, stages + 1, GDN_MMA_M
    ) > (227 * 1024)
    args = _runtime_inputs(device, batch=1, heads=2, seed=20260935, **shape)
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages),
    )
    assert _PLAN_KIND in code
    expected = _torch_reference(*args)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


# The widest token slices (256 and 128 tokens per warp) of the example's own
# chunk-256 shape, on the full 128-row tile.
# (tile, epilogue warps, stages, token groups, MMA M): the widest slices of
# the example's main() shape, and the sixteen-warp CTAs whose fifth warp per
# sub-partition caps the kernel at 96 (tile 128) or 80 (tile 64, with the
# store warps) registers, at the group counts that used to spill; then the
# 64-row MMA's widest slices (a whole 256-token chunk per warp) and its
# register-capped CTAs.
_WIDE_SLICE_CASES = (
    (128, 4, 1, 1, 128),
    (128, 8, 1, 2, 128),
    (128, 16, 1, 1, 128),
    (128, 16, 1, 4, 128),
    (64, 16, 1, 1, 128),
    (64, 4, 1, 1, 64),
    (64, 16, 1, 1, 64),
    (32, 16, 1, 1, 64),
)


def _compile_wide_slice_with_sass(
    block_v: int, epilogue_warps: int, stages: int, token_groups: int, mma_m: int
) -> None:
    """Subprocess body of the spill test: one compile with the SASS kept."""

    device = torch.device("cuda", torch.cuda.current_device())
    module = _example_module()
    args = _runtime_inputs(
        device,
        batch=1,
        heads=2,
        seqlen=1024,
        chunk=256,
        dhead=64,
        dstate=128,
        seed=20260936,
    )
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages, token_groups, mma_m),
    )
    assert _PLAN_KIND in code
    assert f"'token_groups': {token_groups}" in code
    assert f"'mma_m': {mma_m}" in code
    expected = module.ref_gdn_fwd_h(*args)  # type: ignore[attr-defined]
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "token_groups", "mma_m"),
    _WIDE_SLICE_CASES,
)
def test_gdn_recurrence_wide_token_slices_do_not_spill_cuda(
    block_v: int,
    epilogue_warps: int,
    stages: int,
    token_groups: int,
    mma_m: int,
    tmp_path: Path,
) -> None:
    """A warp packs its token slice one accumulator block at a time, so even
    the widest slices stay within the register budget: no local-memory
    spills (``LDL``/``STL``) in the SASS."""

    _require_sm100_runtime()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            __name__,
            "sass",
            str(block_v),
            str(epilogue_warps),
            str(stages),
            str(token_groups),
            str(mma_m),
        ],
        cwd=Path(cast("str", helion.__file__)).resolve().parents[1],
        env={
            **os.environ,
            "HELION_BACKEND": "cute",
            "CUTE_DSL_KEEP": "sass",
            "CUTE_DSL_DUMP_DIR": str(tmp_path),
            "CUTE_DSL_NO_CACHE": "1",
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    sass_files = sorted(tmp_path.glob("*.sass"))
    assert len(sass_files) == 1, sass_files
    spills = [
        line
        for line in sass_files[0].read_text().splitlines()
        if re.search(r"\b(LDL|STL)\b", line)
    ]
    assert not spills, spills[:8]


@pytest.mark.parametrize(
    ("block_v", "epilogue_warps", "stages", "shape"),
    (
        # dstate 8 is narrower than the 16-row tile and 200 tokens leave a
        # ragged half-filled 16-token chunk.
        (
            16,
            8,
            3,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 200,
                "chunk": 16,
                "dhead": 16,
                "dstate": 8,
            },
        ),
        # The 128-row tile the block search reaches above a 96-wide state, at
        # the deepest chunk, where its only legal ring depth is a single stage.
        (
            128,
            8,
            1,
            {
                "batch": 1,
                "heads": 2,
                "seqlen": 640,
                "chunk": 256,
                "dhead": 64,
                "dstate": 96,
            },
        ),
        # A 128-row tile over an 8-wide state with a single head and one row
        # past the first chunk: the last u row of the tensor is the widest
        # overhang the L2 prefetch could reach past.
        (
            128,
            4,
            1,
            {
                "batch": 1,
                "heads": 1,
                "seqlen": 65,
                "chunk": 64,
                "dhead": 32,
                "dstate": 8,
            },
        ),
    ),
)
def test_gdn_recurrence_tile_wider_than_dstate_cuda(
    block_v: int, epilogue_warps: int, stages: int, shape: dict[str, int]
) -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    args = _runtime_inputs(device, seed=20260932, **shape)
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        **_tensor_core_config(block_v, epilogue_warps, stages),
    )
    assert _PLAN_KIND in code
    # The example reference needs dstate to be a multiple of dhead.
    expected = _torch_reference(*args)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


def test_gdn_recurrence_unaligned_dstate_falls_back_cuda() -> None:
    device = _require_sm100_runtime()
    module = _example_module()
    # dstate 36 leaves the u head stride at 72 bytes: no TMA tensor map, so the
    # kernel must keep the SIMT lowering and stay correct.
    args = _runtime_inputs(
        device,
        batch=1,
        heads=2,
        seqlen=256,
        chunk=64,
        dhead=32,
        dstate=36,
        seed=20260931,
    )
    code, result = code_and_output(
        module.helion_gdn_fwd_h,  # type: ignore[attr-defined]
        args,
        block_sizes=[32],
    )
    assert _PLAN_KIND not in code
    expected = _torch_reference(*args)
    torch.testing.assert_close(result.float(), expected.float(), atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    assert sys.argv[1] == "sass"
    _compile_wide_slice_with_sass(*(int(value) for value in sys.argv[2:7]))
