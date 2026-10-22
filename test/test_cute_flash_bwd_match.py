"""Scale forms accepted by the fused flash-attention backward matcher.

``examples/attention.py`` pre-scales K by ``sm_scale * log2(e)`` (so the
softmax recompute has no exp2-domain multiply), rescales dQ by the constant
``ln(2)`` and dK by ``sm_scale``. The matcher used to require one host scalar
for both dK and dQ, so that kernel fell back to the generic SIMT lowering
(seconds per call). It also accepted a dK store of any scaled accumulator,
while the emitted epilogue scales dK only.
"""

from __future__ import annotations

import math

from examples.attention import attention_backward
import pytest
import sympy
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute.cute_flash_bwd import match_attention_bwd

RCP_LN2 = 1.4426950408889634


def _inputs(
    shape: tuple[int, int, int, int],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """(kernel args with K pre-scaled, float64 reference dq/dk/dv)."""
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(4)
    )
    sm_scale = 1.0 / math.sqrt(shape[-1])
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    scores = q64 @ k64.transpose(-1, -2) * sm_scale
    out = scores.softmax(-1) @ v64
    out.backward(do.double())
    o = out.detach().to(q.dtype)
    lse = (torch.logsumexp(scores.detach(), -1) * RCP_LN2).float()
    delta = (o.float() * do.float()).sum(-1)
    args = (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)
    grads = tuple(t.grad for t in (q64, k64, v64))
    return args, grads  # pyrefly: ignore [bad-return]


@onlyBackends(["cute"])
@skipIfRefEager("checks the generated CuTe kernel kind")
@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 256, 128)])
def test_example_backward_takes_the_flash_path(shape: tuple[int, ...]) -> None:
    kernel = helion.kernel(
        attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )
    args, expected = _inputs(shape)  # pyrefly: ignore [bad-argument-type]
    code, grads = code_and_output(kernel, args, block_sizes=[128, 128])
    assert "'kind': 'helion_flash_bwd'" in code
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The example's body with a scaled dV store (and unscaled dK) or with the
    dK accumulator stored into ``dv`` and the dV accumulator into ``dk``."""
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
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - delta[tile_m][None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * 0.5)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
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


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("scale_dv", "swap_stores", "roles"),
    [(False, False, ("dk", "dv")), (True, False, None), (False, True, ("dv", "dk"))],
)
def test_store_roles_follow_the_accumulators(
    scale_dv: bool, swap_stores: bool, roles: tuple[str, str] | None
) -> None:
    """Roles come from which accumulator a store reads, not from which store is
    scaled: the emitted epilogue scales dK only, so a scaled dV is refused."""
    args, _ = _inputs((1, 2, 256, 64))
    bound = _variant.bind((*args, scale_dv, swap_stores))
    with bound.env:
        match = match_attention_bwd(bound.host_function.device_ir)
    if roles is None:
        assert match is None
        return
    assert match is not None
    assert (match.dk_name, match.dv_name) == roles
    assert match.qk_scale_sym == 1.0
    assert match.out_scale_sym == 0.5
    assert isinstance(match.dk_scale_sym, sympy.Expr)
