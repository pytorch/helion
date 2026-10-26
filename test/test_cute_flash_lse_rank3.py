"""The flash-attention path admits a rank-3 ``[B*H, S, 1]`` fp32 lse store.

``examples/attention.py`` stores lse with a trailing unit axis
(``lse[tile_b, tile_m, :] = (m_i + log2(l_i))[:, :, None]``, a workaround for
pytorch/helion#2842).  The detector used to accept only a rank-2 lse, which
kept the canonical attention example on the scalar fallback.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _attention_rank3_lse(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m, :] = (m_i + torch.log2(l_i))[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.reshape(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def _attention_rank3_not_lse(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stored rank-3 value is not ``m_i + log2(l_i)``: no flash admission."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    stat = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
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
        acc = acc / l_i[:, :, None]
        stat[tile_b, tile_m, :] = (m_i * 2.0)[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), stat.reshape(q_in.size()[:-1])


def _meta_args(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(*shape, dtype=torch.bfloat16, device="meta")  # @ignore-device-lint
        for _ in range(3)
    )


def _graph_plan(
    bound: BoundKernel,
) -> cute_flash.FlashGraphOutputPlan | None:
    with bound.env:
        device_ir = bound.host_function.device_ir
        root_block_ids = device_ir.grid_block_ids[0]
        loop = next(
            graph
            for graph in device_ir.graphs
            if type(graph).__name__ == "ForLoopGraphInfo"
        )
        return cute_flash._flash_graph_output_plan_from_graphs(
            device_ir.graphs,
            root_block_ids=root_block_ids,
            kv_block_id=loop.block_ids[0],
            score_plan=None,
        )


def _bind_as_if_sm100(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
) -> BoundKernel:
    """Bind with the tcgen05 support probe forced on (no GPU needed)."""
    with (
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        return kernel.bind(args)


def test_rank3_lse_store_is_the_flash_lse_output() -> None:
    bound = _bind_as_if_sm100(_attention_rank3_lse, _meta_args((2, 8, 1024, 64)))
    assert bound.config_spec.cute_flash_search_enabled
    plan = _graph_plan(bound)
    assert plan is not None
    assert plan.lse_name == "lse"
    assert plan.lse_log_base == "log2"
    assert plan.output_epilogue == "identity"


def test_rank3_store_of_a_non_lse_value_declines() -> None:
    bound = _bind_as_if_sm100(_attention_rank3_not_lse, _meta_args((2, 8, 1024, 64)))
    assert not bound.config_spec.cute_flash_search_enabled
    assert _graph_plan(bound) is None


def test_canonical_attention_example_reaches_the_flash_path() -> None:
    from examples.attention import attention

    kernel = helion.kernel(
        attention.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = _bind_as_if_sm100(kernel, _meta_args((2, 8, 1024, 64)))
    assert bound.config_spec.cute_flash_search_enabled
    plan = _graph_plan(bound)
    assert plan is not None
    assert plan.lse_name == "lse"


@onlyBackends(["cute"])
@pytest.mark.parametrize("family", ("ws_overlap", "fa4"))
def test_rank3_lse_runtime_matches_reference(family: str) -> None:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(2, 8, 512, 64, dtype=torch.float16, device=DEVICE) for _ in range(3)
    )
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
    expected_lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    expected_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    code, (out, lse) = code_and_output(
        _attention_rank3_lse,
        (q, k, v),
        block_sizes=[1, 128, 128],
        cute_flash_pipeline_family=family,
        cute_flash_persistent=False,
    )
    assert "_flash_mLSE" in code
    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
