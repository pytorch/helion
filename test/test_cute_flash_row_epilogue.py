"""Fused row-local output epilogues on the CuTe flash-attention path."""

from __future__ import annotations

import ast
import math
import operator
import types
from typing import Callable

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_row_epilogue
from helion._compiler.cute.flash_row_epilogue import ROW
from helion._compiler.cute.flash_row_epilogue import VEC
from helion._compiler.cute.flash_row_epilogue import FlashRowEpilogueProgram
from helion._compiler.cute.flash_row_epilogue import RowEpilogueOp
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


@helion.kernel(backend="cute", static_shapes=True)
def _gated_output(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """The output is a single-use product: ``o * g``."""
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
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = (o * g).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _two_products_added(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Both operands of the output add are single-use products."""
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
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = (o * g + o * (g * g)).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _projection_residual(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """A product feeds a reduction in one pass and the store pass's sub."""
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
        g = v_view[tile_b, tile_m, :].to(torch.float32)
        proj = torch.sum(o * g, dim=-1, keepdim=True)
        out[tile_b, tile_m, :] = (o * g - proj * g).to(out.dtype)
    return out.view(q_in.size())


def _ref_gated(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    return (y * v.float()).to(q.dtype)


def _ref_two_products(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    g = v.float()
    return (y * g + y * (g * g)).to(q.dtype)


def _ref_projection_residual(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v).float()
    g = v.float()
    return (y * g - (y * g).sum(dim=-1, keepdim=True) * g).to(q.dtype)


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
        # The program runs on the softmax warpgroups (``cute_flash_row_epilogue_warps``
        # defaults to ``softmax``): the value-norm pass sits ahead of the softmax
        # KV loop when the aux rows come from gmem, and right after the
        # ``aux_full`` wait when the aux tile is staged through sO.
        softmax0 = code.split("if warp_idx < 4:", 1)[1].split("\n    if ", 1)[0]
        correction = code.split("(warp_idx >= 8) & (warp_idx < 12)", 1)[1]
        assert "cute.math.sqrt" in softmax0 and "cute.math.sqrt" not in correction
        staged = config.get("cute_flash_epi_tma") or config.get("cute_flash_epi_stg")
        anchor = (
            "mbar_spin_wait(flash_aux_full_ptr + 0"
            if staged
            else "for flash_kv in cutlass.range"
        )
        assert (softmax0.index("cute.math.sqrt") > softmax0.index(anchor)) is bool(
            staged
        )


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
    # Element pairs per step; the two chunk-parity accumulators and both lanes
    # are folded once the pass closes.
    assert epilogue.count("for j in cutlass.range_constexpr(0, 16, 2):") == 8
    assert "e_v4_0_p0 + e_v4_0_p1 + e_v4_1_p0 + e_v4_1_p1" in prologue
    assert "cute.arch.fma_packed_f32x2(" in epilogue
    scalar_prologue, scalar_epilogue = flash_row_epilogue.emit_row_epilogue(
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
        packed=False,
    )
    # The scalar form walks the same pairs with the same accumulators and folds
    # the same products (one scalar fma per lane of a packed fma).
    assert scalar_epilogue.count("for j in cutlass.range_constexpr(0, 16, 2):") == 8
    assert "e_v4_0_p0 + e_v4_0_p1 + e_v4_1_p0 + e_v4_1_p1" in scalar_prologue
    assert "_packed_f32x2(" not in scalar_epilogue
    assert scalar_epilogue.count("cute.math.fma(") == 2 * epilogue.count(
        "cute.arch.fma_packed_f32x2("
    )
    assert scalar_prologue.count("cute.math.fma(") == 2 * prologue.count(
        "cute.arch.fma_packed_f32x2("
    )


# --------------------------------------------------------------------------
# Emitter semantics: run the emitted source on plain Python floats
# --------------------------------------------------------------------------

_HEAD_DIM = 16
_CHUNK = 8


def _program(
    ops: list[RowEpilogueOp], output: str, aux_count: int
) -> FlashRowEpilogueProgram:
    scheduled = flash_row_epilogue._schedule(ops)
    return FlashRowEpilogueProgram(
        tuple(scheduled),
        output,
        tuple(f"aux{i}" for i in range(aux_count)),
        (torch.bfloat16,) * aux_count,
        (),
        flash_row_epilogue._lookup(scheduled, output).avail,
    )


def _emit(program: FlashRowEpilogueProgram, packed: bool) -> tuple[str, str]:
    return flash_row_epilogue.emit_row_epilogue(
        program,
        chunks=_HEAD_DIM // _CHUNK,
        chunk_width=_CHUNK,
        elem_var="j",
        o_alloc="{o} = alloc()",
        o_load=["load_o({i}, {o})", "scale({o})"],
        o_elem="{o}[{j}]",
        aux_allocs=["{a} = alloc()"] * len(program.aux_names),
        aux_loads=[
            [f"load_aux({k}, {{i}}, {{a}})"] for k in range(len(program.aux_names))
        ],
        aux_elems=["cutlass.Float32({a}[{j}])"] * len(program.aux_names),
        out_alloc="{out} = alloc()",
        store_elem="{out}[{j}] = {value}",
        store=["store_out({i}, {out})"],
        scalar_names=[],
        indent="",
        prefix="e",
        aux_prefetch=1,
        split=True,
        packed=packed,
    )


class _Float32:
    inf = math.inf

    def __call__(self, value: object) -> float:
        return float(value)  # pyrefly: ignore [bad-argument-type]


def _run_emitted(
    program: FlashRowEpilogueProgram,
    o_row: list[float],
    aux_rows: list[list[float]],
    packed: bool,
) -> list[float]:
    """Execute the emitted passes with DSL stand-ins on Python floats."""
    result = [math.nan] * _HEAD_DIM

    def load_o(i: int, o: list[float]) -> None:
        o[:] = o_row[i * _CHUNK : (i + 1) * _CHUNK]

    def load_aux(k: int, i: int, a: list[float]) -> None:
        a[:] = aux_rows[k][i * _CHUNK : (i + 1) * _CHUNK]

    def store_out(i: int, out: list[float]) -> None:
        result[i * _CHUNK : (i + 1) * _CHUNK] = out

    def packed_op(
        fn: Callable[[float, float], float],
    ) -> Callable[..., tuple[float, float]]:
        return lambda x, y: (fn(x[0], y[0]), fn(x[1], y[1]))

    namespace: dict[str, object] = {
        "alloc": lambda: [math.nan] * _CHUNK,
        "load_o": load_o,
        "scale": lambda o: None,
        "load_aux": load_aux,
        "store_out": store_out,
        "cutlass": types.SimpleNamespace(
            Float32=_Float32(),
            BFloat16=_Float32(),
            Float16=_Float32(),
            range_constexpr=range,
            select_=lambda c, a, b: a if c else b,
        ),
        "cute": types.SimpleNamespace(
            math=types.SimpleNamespace(
                fma=lambda a, b, c: a * b + c,
                max=lambda a, b, propagate_nan=True: max(a, b),
                min=lambda a, b, propagate_nan=True: min(a, b),
                sqrt=math.sqrt,
                rsqrt=lambda x: 1.0 / math.sqrt(x),
                exp=math.exp,
                exp2=lambda x, fastmath=False: 2.0**x,
                log=math.log,
                log2=math.log2,
                tanh=math.tanh,
                abs=abs,
                rcp=lambda x, approx=False, ftz=False: 1.0 / x,
            ),
            arch=types.SimpleNamespace(
                fma_packed_f32x2=lambda x, y, c: (
                    x[0] * y[0] + c[0],
                    x[1] * y[1] + c[1],
                ),
                add_packed_f32x2=packed_op(operator.add),
                sub_packed_f32x2=packed_op(operator.sub),
                mul_packed_f32x2=packed_op(operator.mul),
            ),
        ),
    }
    prologue, epilogue = _emit(program, packed)
    exec(prologue + "\n" + epilogue, namespace)
    return result


_O = RowEpilogueOp("o", VEC, "o_norm")
_A0 = RowEpilogueOp("a0", VEC, "aux", (), (0,))
_A1 = RowEpilogueOp("a1", VEC, "aux", (), (1,))


def _mul(name: str, a: str, b: str) -> RowEpilogueOp:
    return RowEpilogueOp(name, VEC, "mul", (a, b))


def _reduce_sum(name: str, source: str) -> RowEpilogueOp:
    return RowEpilogueOp(name, ROW, "reduce", (source,), ("sum",))


_EMITTER_CASES: dict[
    str,
    tuple[
        list[RowEpilogueOp],
        str,
        int,
        Callable[[list[float], list[list[float]]], list[float]],
        int,
    ],
] = {
    # The output itself is a single-use product: nothing to fold into.
    "output_is_mul": (
        [_O, _A0, _mul("m", "o", "a0")],
        "m",
        1,
        lambda o, aux: [x * a for x, a in zip(o, aux[0], strict=True)],
        0,
    ),
    # Both add operands are single-use products: one folds, the other stays.
    "two_muls_into_add": (
        [
            _O,
            _A0,
            _A1,
            _mul("m0", "o", "a0"),
            _mul("m1", "o", "a1"),
            RowEpilogueOp("s", VEC, "add", ("m0", "m1")),
        ],
        "s",
        2,
        lambda o, aux: [
            x * a + x * b for x, a, b in zip(o, aux[0], aux[1], strict=True)
        ],
        1,
    ),
    "two_muls_into_sub": (
        [
            _O,
            _A0,
            _A1,
            _mul("m0", "o", "a0"),
            _mul("m1", "a0", "a1"),
            RowEpilogueOp("s", VEC, "sub", ("m0", "m1")),
        ],
        "s",
        2,
        lambda o, aux: [
            x * a - a * b for x, a, b in zip(o, aux[0], aux[1], strict=True)
        ],
        1,
    ),
    # A product feeds the first pass's reduction and the store pass's sub,
    # whose other operand is a product as well.
    "mul_into_sum_and_output": (
        [
            _O,
            _A0,
            _mul("m", "o", "a0"),
            _reduce_sum("r", "m"),
            _mul("pa", "r", "a0"),
            RowEpilogueOp("s", VEC, "sub", ("m", "pa")),
        ],
        "s",
        1,
        lambda o, aux: [
            x * a - sum(y * b for y, b in zip(o, aux[0], strict=True)) * a
            for x, a in zip(o, aux[0], strict=True)
        ],
        2,
    ),
    # XSA's shape: the projection folds into the sum, the residual into the sub.
    "projection_off_the_aux_row": (
        [
            _O,
            _A0,
            _mul("m", "o", "a0"),
            _reduce_sum("r", "m"),
            _mul("pa", "r", "a0"),
            RowEpilogueOp("s", VEC, "sub", ("o", "pa")),
        ],
        "s",
        1,
        lambda o, aux: [
            x - sum(y * b for y, b in zip(o, aux[0], strict=True)) * a
            for x, a in zip(o, aux[0], strict=True)
        ],
        2,
    ),
}


@pytest.mark.parametrize("case", sorted(_EMITTER_CASES))
@pytest.mark.parametrize("packed", (True, False))
def test_row_epilogue_emitter_folds_one_product_per_consumer(
    case: str, packed: bool
) -> None:
    ops, output, aux_count, reference, fused_per_step = _EMITTER_CASES[case]
    program = _program(ops, output, aux_count)
    torch.manual_seed(0)
    o_row = torch.randn(_HEAD_DIM, dtype=torch.float64).tolist()
    aux_rows = [
        torch.randn(_HEAD_DIM, dtype=torch.float64).tolist() for _ in range(aux_count)
    ]
    # Every name the emitted source reads is defined (the packed emitter used to
    # skip a folded product's definition while its consumer still read it, or
    # to look up the consumer of the output) and the arithmetic is right.
    result = _run_emitted(program, o_row, aux_rows, packed)
    expected = reference(o_row, aux_rows)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)
    prologue, epilogue = _emit(program, packed)
    code = prologue + "\n" + epilogue
    # Products fold exactly where the program has a consumer for them: the same
    # set in both instruction forms (a packed fma covers two lanes).
    fma_lines = code.count(
        "cute.arch.fma_packed_f32x2(" if packed else "cute.math.fma("
    )
    steps = _HEAD_DIM // _CHUNK
    assert fma_lines == fused_per_step * steps * (1 if packed else 2)
    if packed:
        assert "cute.math.fma(" not in code
    else:
        assert "_packed_f32x2(" not in code
    if case.startswith("two_muls"):
        # The unfolded product is still materialized.
        assert ("mul_packed_f32x2(" in code) is packed
        assert ("e_m0_p0 = (e_o_p0 * e_a0_p0)" in code) is not packed
    if case == "output_is_mul":
        assert "fma" not in code


def test_row_epilogue_emitter_forms_agree_bitwise_on_the_xsa_program() -> None:
    # Same pairs, same accumulators, same folds: the two instruction forms
    # differ only in the instruction spelling.
    bound = _xsa_like.bind((*_meta_args((2, 8, 1024, 64)), 1e-6))
    program = _graph_plan(bound).row_epilogue
    assert program is not None
    packed_prologue, packed_epilogue = flash_row_epilogue.emit_row_epilogue(
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
        packed=True,
    )
    scalar_prologue, scalar_epilogue = flash_row_epilogue.emit_row_epilogue(
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
        packed=False,
    )

    # Outside the arithmetic lines the two emissions are identical.
    def arithmetic_free(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if "_packed_f32x2(" not in line
            and "cute.math.fma(" not in line
            and not any(op in line for op in (" + ", " - ", " * "))
        ]

    assert arithmetic_free(packed_prologue) == arithmetic_free(scalar_prologue)
    assert arithmetic_free(packed_epilogue) == arithmetic_free(scalar_epilogue)


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "kernel, reference",
    (
        pytest.param(_gated_output, _ref_gated, id="output_is_mul"),
        pytest.param(_two_products_added, _ref_two_products, id="two_muls_into_add"),
        pytest.param(
            _projection_residual, _ref_projection_residual, id="mul_into_sum_and_sub"
        ),
    ),
)
def test_row_programs_with_folded_products_match_reference(
    kernel: helion.Kernel, reference: Callable[..., torch.Tensor]
) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bound = kernel.bind((q, k, v))
    assert bound.config_spec.cute_flash_search_enabled
    config = bound._normalized_config_copy(
        helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=True,
            cute_flash_epi_tma=True,
        )
    )
    code = bound.to_code(config)
    assert "cute.arch.fma_packed_f32x2(" in code or kernel is _gated_output
    out = bound.compile_config(config)(q, k, v)
    torch.testing.assert_close(
        out.float(), reference(q, k, v).float(), rtol=2e-2, atol=2e-2
    )


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "config",
    (
        {"cute_flash_pipeline_family": "fa4"},
        {"cute_flash_pipeline_family": "fa4", "cute_flash_epi_tma": True},
        {"cute_flash_pipeline_family": "ws_overlap"},
        # The register-MMA row programs evaluate the program in their combine
        # epilogue (here behind a double-buffered chunk loop over 256 keys per
        # warp); explicit configs are legal on any grid.
        {"cute_flash_pipeline_family": "row_mma"},
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
