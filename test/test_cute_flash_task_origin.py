"""The fused flash bodies number the root grid's work items from 0 in steps
of 1, so a kernel whose root grid starts elsewhere stays on the general path
(``match_common.root_grid_origin_is_canonical``): ``examples/attention.py``'s
forward and backward here, against copies whose grid skips the first
batch-head; the gated bodies in test_cute_flash_gated_codegen.py."""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._compiler.cute.analysis.match_common import root_grid_origin_is_canonical
from helion._testing import import_example
from helion._testing.cute_aux import cpu_codegen
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")

_EXAMPLE = import_example("attention")


@helion.kernel(backend="cute", static_shapes=True)
def _attention_from_one(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``examples/attention.py``'s ``attention`` over batch-heads 1 to n - 1."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504
    for tile_b, tile_m in hl.tile([1, 0], [q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m, :] = (m_i + torch.log2(l_i))[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.reshape(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def _attention_backward_from_second(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``examples/attention.py``'s ``attention_backward`` over the K/V rows
    of the second batch-head on."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_n_rows = k.size(0)
    LN2: float = 0.6931471824645996
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(n_dim, total_n_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        batch_idx = tile_n.begin // n_dim
        start_m = batch_idx * m_dim
        end_m = start_m + m_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * LN2)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dq.reshape(q_in.size()), dk.reshape(k_in.size()), dv.reshape(v_in.size())


def _renders(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    configs: tuple[helion.Config | None, ...],
) -> tuple[list[str], bool]:
    """The renders at ``configs`` (the default config for None) of one CPU
    bind, and whether the bind opened a flash search surface."""
    kernel.reset()
    with cpu_codegen():
        bound = kernel.bind(args)
        spec = bound.config_spec
        searched = spec.cute_flash_search_enabled or spec.cute_flash_bwd_search_enabled
        codes = [bound.to_code(config or spec.default_config()) for config in configs]
        return codes, searched


def test_a_forward_grid_that_does_not_start_at_zero_takes_the_general_path() -> None:
    """Over batch-heads 1 to 7 the fused body would compute 0 to 6: that
    bind keeps the flash search closed, and both its default config and the
    fused body's block sizes render the general path, while the example's
    grid from 0 renders the fused body at both."""
    args = tuple(torch.randn(2, 4, 2048, 64, dtype=torch.bfloat16) for _ in range(3))
    example = helion.kernel(_EXAMPLE.attention.fn, backend="cute", static_shapes=True)
    configs = (None, helion.Config(block_sizes=[1, 128, 128]))
    codes, searched = _renders(example, args, configs)
    assert searched and all("'kind': 'helion_flash'" in code for code in codes)
    codes, searched = _renders(_attention_from_one, args, configs)
    assert not searched
    assert not any("'kind': 'helion_flash" in code for code in codes)


def test_a_backward_grid_that_does_not_start_at_zero_takes_the_general_path() -> None:
    """The backward's K/V tiles from the second batch-head on: the fused body
    would start at tile 0."""
    q, k, v, o, do = (
        torch.randn(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(5)
    )
    lse, delta = (torch.randn(1, 2, 256, dtype=torch.float32) for _ in range(2))
    args = (q, k, v, o, lse, do, delta, 0.125)
    config = helion.Config(block_sizes=[128, 128])
    example = helion.kernel(
        _EXAMPLE.attention_backward.fn, backend="cute", static_shapes=True
    )
    (code,), searched = _renders(example, args, (config,))
    assert searched and "'kind': 'helion_flash_bwd'" in code
    (code,), searched = _renders(_attention_backward_from_second, args, (config,))
    assert not searched and "'kind': 'helion_flash_bwd'" not in code


@helion.kernel(backend="cute", static_shapes=True)
def _copy_from_one(x: torch.Tensor) -> torch.Tensor:
    """Copies ``x`` from row 1 on, the range unpacked from a tuple."""
    out = torch.zeros_like(x)
    for tile in hl.tile(*([1], [x.size(0)])):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _copy_all(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile([0], [x.size(0)]):
        out[tile] = x[tile]
    return out


def test_a_root_range_unpacked_from_a_tuple_has_no_canonical_origin() -> None:
    """``hl.tile(*ranges)`` does not spell out its begin, which the origin
    rule reads from the call (the fused bodies then take ``hl.tile(*([1,
    0], ...))`` for a grid from 0); a literal 0 begin is canonical."""
    x = torch.empty(64, 8)
    for kernel, args, canonical in (
        (_copy_from_one, (x,), False),
        (_copy_all, (x,), True),
    ):
        kernel.reset()
        with cpu_codegen():
            device_ir = kernel.bind(args).host_function.device_ir
        assert root_grid_origin_is_canonical(device_ir) is canonical
