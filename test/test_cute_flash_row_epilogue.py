"""Fused row-local output epilogues on the CuTe flash-attention path."""

from __future__ import annotations

import ast
import math

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_row_epilogue
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _xsa_like(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Attention whose output row is projected off the normalized value row."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        o = acc / l_i[:, :, None]
        v_self = v_view[tile_b, tile_m, :].to(torch.float32)
        v_norm = torch.sqrt(torch.sum(v_self * v_self, dim=-1, keepdim=True))
        vn = v_self / torch.clamp(v_norm, min=eps)
        proj = torch.sum(o * vn, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o - proj * vn).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _unsupported_epilogue(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """A cumulative sum over head_dim is not a row-local program."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        o = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = torch.cumsum(o, dim=-1).to(out.dtype)
    return out.view(q_in.size())


def _ref_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    vn = torch.nn.functional.normalize(v.float(), dim=-1, eps=eps)
    return (y - (y * vn).sum(dim=-1, keepdim=True) * vn).to(q.dtype)


def _meta_args(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(*shape, dtype=torch.bfloat16, device="meta")  # @ignore-device-lint
        for _ in range(3)
    )


def _graph_plan(
    bound: helion.runtime.kernel.BoundKernel,
) -> cute_flash.FlashGraphOutputPlan:
    with bound.env:
        device_ir = bound.host_function.device_ir
        root_block_ids = device_ir.grid_block_ids[0]
        loop = next(
            graph
            for graph in device_ir.graphs
            if type(graph).__name__ == "ForLoopGraphInfo"
        )
        plan = cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=root_block_ids,
            kv_block_id=loop.block_ids[0],
            score_plan=None,
        )
    assert plan is not None
    return plan


def test_row_program_detected_for_xsa_like_kernel() -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    assert bound.config_spec.cute_flash_search_enabled
    plan = _graph_plan(bound)
    assert plan.output_epilogue == flash_row_epilogue.FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM
    program = plan.row_epilogue
    assert program is not None
    assert program.aux_names == ("v_view",)
    assert len(program.scalar_exprs) == 1  # eps
    assert program.passes == 3
    reductions = [op for op in program.ops if op.op == "reduce"]
    assert [op.reduce_pass for op in reductions] == [0, 1]
    # The value-norm pass never reads the accumulator, so it can run before the loop.
    assert flash_row_epilogue.hoistable_passes(program) == 1
    assert program.op(program.output).op == "sub"


def test_row_program_rejects_non_row_local_epilogue() -> None:
    bound = _unsupported_epilogue.bind(_meta_args((2, 8, 1024, 64)))
    assert not bound.config_spec.cute_flash_search_enabled


@pytest.mark.parametrize(
    "config",
    (
        {"cute_flash_pipeline_family": "fa4"},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_stg": True},
        {"cute_flash_pipeline_family": "fa4_2cta", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "ws_overlap"},
        {"cute_flash_pipeline_family": "ws_overlap", "cute_flash_s_stage": 1},
    ),
)
def test_row_program_codegen_every_route(config: dict[str, object]) -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    full = {"block_sizes": [1, 128, 128], "cute_flash_persistent": False, **config}
    code = bound.to_code(bound._normalized_config_copy(helion.Config(**full)))
    ast.parse(code)
    assert "_flash_mEpiAux0" in code
    assert "_ep0_v" in code or "_ep_v" in code
    # Chunk indices are literal so the DSL unrolls every fragment access.
    assert "range_constexpr(cute.size" not in code
    if config["cute_flash_pipeline_family"].startswith("fa4"):
        # The value-norm pass is hoisted ahead of the correction warps' KV loop.
        correction = code.split("(warp_idx >= 8) & (warp_idx < 12)", 1)[1]
        prologue = correction.split("for flash_kv in cutlass.range", 1)[0]
        assert "cute.math.sqrt" in prologue


def test_row_epilogue_emitter_hoists_and_unrolls() -> None:
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    program = _graph_plan(bound).row_epilogue
    assert program is not None
    prologue, epilogue = flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=4,
        chunk_width=16,
        elem_var="j",
        o_alloc="{o} = alloc_o()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc_a()"],
        aux_loads=[["load_a({i}, {a})"]],
        aux_elems=["cutlass.Float32({a}[{j}])"],
        out_alloc="{out} = alloc_out()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=["eps"],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
    )
    assert "cute.math.sqrt" in prologue and "load_o" not in prologue
    assert "load_a(0, e_a0_0)" in prologue and "load_a(1, e_a0_1)" in prologue
    assert "load_o(0, e_o0)" in epilogue and "store_out(3, e_out1)" in epilogue
    assert epilogue.count("for j in cutlass.range_constexpr(16):") == 8
    assert "e_v4_0 + e_v4_1" in prologue  # split reduction accumulators are folded


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "config",
    (
        {"cute_flash_pipeline_family": "fa4"},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "ws_overlap"},
    ),
)
def test_row_program_runtime_matches_reference(config: dict[str, object]) -> None:
    torch.manual_seed(0)
    outputs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        q = torch.randn(2, 8, 1024, 64, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        bound = _xsa_like.bind((q, k, v, 1e-6))
        full = {"block_sizes": [1, 128, 128], "cute_flash_persistent": False, **config}
        fn = bound.compile_config(bound._normalized_config_copy(helion.Config(**full)))
        out = fn(q, k, v, 1e-6)
        torch.testing.assert_close(
            out.float(), _ref_xsa(q, k, v, 1e-6).float(), rtol=1e-2, atol=5e-2
        )
        outputs.append(out)
    assert not torch.equal(outputs[0], outputs[1])
