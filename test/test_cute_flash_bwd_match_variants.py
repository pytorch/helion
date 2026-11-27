"""Store roles and dot operands the fused flash-attention backward matcher accepts.

The matcher used to accept a dK store of any scaled accumulator, while the
emitted epilogue scales dK only, and dV and dK dots that take any second
operand, while the emitted body computes dV from dO and dK from Q. The kernel
here is ``examples/attention.py``'s backward with those parts switched by
compile-time arguments; every test binds it on the CPU and reads the match
(the example's own runs are in test_cute_flash_bwd_match.py).
"""

from __future__ import annotations

from typing import Any

import pytest
import sympy
import torch

import helion
from helion._compiler.cute.cute_flash_bwd import match_attention_bwd
from helion._testing.cute_aux import cpu_codegen
from helion._testing.cute_aux import flash_env
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _variant(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
    scale_dv: hl.constexpr,
    swap_stores: hl.constexpr,
    operands: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The example's body with a scaled dV store (and unscaled dK), with the
    dK accumulator stored into ``dv`` and the dV accumulator into ``dk``, or
    with the second operands of its dV, dK and dQ dots as ``operands`` says:
    "dv", "dk" or "dq" gives that dot another tile (Q for dO, dO for Q, V for
    K), "copies" gives the dV and dK dots plain-named copies of the example's
    tiles and "reloads" loads of the same rows of their own."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(k.size(0), block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        start_m = tile_n.begin // n_dim * m_dim
        for tile_m in hl.tile(start_m, start_m + m_dim, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - lse[tile_m][None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            if operands == "copies":
                dv_rhs = do_i
                dk_rhs = q_i
            elif operands == "reloads":
                dv_rhs = do[tile_m, :]
                dk_rhs = q[tile_m, :]
            else:
                dv_rhs = q_i if operands == "dv" else do_i
                dk_rhs = do_i if operands == "dk" else q_i
            dq_rhs = v_j if operands == "dq" else k_j
            dv_acc = hl.dot(p_t.to(v.dtype), dv_rhs, acc=dv_acc)
            ds_t = (p_t * (dp_t - delta[tile_m][None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, dq_rhs, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * 0.5)
            dk_acc = hl.dot(ds_t, dk_rhs, acc=dk_acc)
        if swap_stores:
            dv[tile_n, :] = (dk_acc * sm_scale).to(v.dtype)
            dk[tile_n, :] = dv_acc.to(k.dtype)
        elif scale_dv:
            dv[tile_n, :] = (dv_acc * sm_scale).to(v.dtype)
            dk[tile_n, :] = dk_acc.to(k.dtype)
        else:
            dv[tile_n, :] = dv_acc.to(v.dtype)
            dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dq, dk, dv


def _bind_and_match(
    scale_dv: bool, swap_stores: bool, operands: str | None
) -> tuple[Any, Any]:
    """A CPU bind of the variant and what the backward matcher makes of it."""
    q, k, v, o, do = (
        torch.empty(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(5)
    )
    lse, delta = (torch.empty(1, 2, 256, dtype=torch.float32) for _ in range(2))
    args = (q, k, v, o, lse, do, delta, 0.125, scale_dv, swap_stores, operands)
    with flash_env(), cpu_codegen():
        bound = _variant._bind_isolated(args)
        with bound.env:
            return bound, match_attention_bwd(bound.host_function.device_ir)


@pytest.mark.parametrize(
    ("scale_dv", "swap_stores", "roles"),
    [(False, False, ("dk", "dv")), (True, False, None), (False, True, ("dv", "dk"))],
)
def test_store_roles_follow_the_accumulators(
    scale_dv: bool, swap_stores: bool, roles: tuple[str, str] | None
) -> None:
    """Roles come from which accumulator a store reads, not from which store is
    scaled: the emitted epilogue scales dK only, so a scaled dV is refused."""
    _, match = _bind_and_match(scale_dv, swap_stores, None)
    if roles is None:
        assert match is None
        return
    assert match is not None
    assert (match.dk_name, match.dv_name) == roles
    assert match.qk_scale_sym == 1.0
    assert match.out_scale_sym == 0.5
    assert isinstance(match.dk_scale_sym, sympy.Expr)


@pytest.mark.parametrize(
    ("operands", "fused"),
    [
        (None, True),
        ("copies", True),
        ("dv", False),
        ("dk", False),
        ("dq", False),
        ("reloads", False),
    ],
    ids=["example", "copies", "dv_takes_q", "dk_takes_do", "dq_takes_v", "reloads"],
)
def test_dots_take_the_tiles_the_fused_body_reads(
    operands: str | None, fused: bool
) -> None:
    """The fused body computes dV from the tile the dP dot transposes (dO), dK
    from the tile the score dot transposes (Q) and dQ from the K tile, so a
    kernel whose dot takes another tile second is not that body's: with Q for
    dO it used to be fused, and returned the example's dV. A plain-named copy
    of a tile is the tile; another load of its rows is another node, which the
    matcher does not prove equal. On a CPU bind, a declined kernel keeps the
    search surface closed and renders the general path at the fused body's
    block sizes."""
    bound, match = _bind_and_match(False, False, operands)
    with flash_env(), cpu_codegen():
        searched = bound.config_spec.cute_flash_bwd_search_enabled
        code = bound.to_code(helion.Config(block_sizes=[128, 128]))
    assert (match is not None, searched) == (fused, fused)
    assert ("'kind': 'helion_flash_bwd'" in code) is fused
    if match is not None:
        assert (match.q_name, match.do_name) == ("q", "do")
        assert (match.k_name, match.v_name) == ("k", "v")
