"""Under-aligned operands of the fused flash kernels.

A TMA descriptor encodes its global address in 16-byte units, and the CuTe
DSL builds one over a pointer whose assumed alignment is only 8 bytes without
complaint. An 8-byte-aligned q, k or v base (a contiguous view four bf16
elements into a buffer) therefore made the tcgen05 flash bodies read every
tile 8 bytes early: the output and the LSE were those of the shifted data,
with no fault; the fused backward read q, k, v and dO the same way.
``default_cute_launcher`` now stages such operands of the flash plan kinds
through 16-byte-aligned copies and copies the outputs back.
"""

from __future__ import annotations

import math
import types

from benchmarks.cute.attnbwd_kernels import attention_bwd
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.cute import launcher as cute_launcher

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash

FAMILY = cute_flash.FLASH_PIPELINE_FAMILY_KEY


def attention_into(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    out_in: torch.Tensor,
    lse_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``examples.attention.attention`` writing into caller-provided buffers."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = out_in.reshape([-1, m_dim, head_dim])
    lse = lse_in.reshape([-1, m_dim, 1])
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
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


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 host math: PyTorch's default SDPA backend shares the defect."""
    q64, k64, v64 = (t.detach().cpu().double() for t in (q, k, v))
    scores = torch.matmul(q64, k64.transpose(-1, -2)) / math.sqrt(q.size(-1))
    out = torch.matmul(torch.softmax(scores, dim=-1), v64)
    lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    return out.to(q.dtype).to(q.device), lse.float().to(q.device)


def _backward_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """float64 autograd: (dq, dk, dv), base-2 LSE and delta = rowsum(o * do)."""
    q64, k64, v64, do64 = (t.detach().cpu().double() for t in (q, k, v, do))
    for leaf in (q64, k64, v64):
        leaf.requires_grad_(True)
    scores = torch.matmul(q64, k64.transpose(-1, -2)) / math.sqrt(q.size(-1))
    out = torch.matmul(torch.softmax(scores, dim=-1), v64)
    out.backward(do64)
    lse = torch.logsumexp(scores.detach(), dim=-1) * math.log2(math.e)
    delta = (out.detach() * do64).sum(-1)
    grads = tuple(
        leaf.grad.to(q.device) for leaf in (q64, k64, v64) if leaf.grad is not None
    )
    return grads, lse.float().to(q.device), delta.float().to(q.device)


def _offset_view(
    shape: tuple[int, ...], dtype: torch.dtype, offset: int, seed: int
) -> torch.Tensor:
    """A contiguous view ``offset`` elements into a fresh buffer."""
    numel = math.prod(shape)
    torch.manual_seed(seed)
    buffer = torch.randn(numel + 64, dtype=dtype, device=DEVICE)
    view = buffer[offset : offset + numel].view(shape)
    assert view.is_contiguous()
    assert view.data_ptr() % 16 == (offset * view.element_size()) % 16
    # The buffer outlives the view through the view's storage.
    return view


# Family -> (shape, config). The 2-CTA family needs a KV count that is a
# multiple of four, so it runs twice the sequence.
_FAMILIES: dict[str, tuple[tuple[int, int, int, int], dict[str, object]]] = {
    "ws_overlap": ((1, 4, 256, 64), {FAMILY: "ws_overlap"}),
    "fa4": ((1, 4, 256, 64), {FAMILY: "fa4"}),
    "fa4_2cta": (
        (1, 4, 512, 64),
        {
            FAMILY: "fa4_2cta",
            cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
            cute_flash.FLASH_PERSISTENT_KEY: False,
        },
    ),
}


def _fake_kernel(*plans: dict[str, object]) -> object:
    return types.SimpleNamespace(_helion_cute_wrapper_plans=list(plans))


def test_staging_clones_under_aligned_operands_and_copies_outputs_back() -> None:
    kernel = _fake_kernel(
        {
            "kind": "helion_flash",
            "q_idx": 0,
            "k_idx": 1,
            "v_idx": 2,
            "o_idx": 3,
            "lse_idx": 4,
            "epi_aux_count": 1,
            "epi_aux0_idx": 5,
        }
    )
    assert cute_launcher._cute_staged_plan_operands(kernel) == (
        (0, False),
        (1, False),
        (2, False),
        (5, False),
        (3, True),
        (4, True),
    )
    shape = (2, 8, 16)
    numel = math.prod(shape)
    torch.manual_seed(0)
    q = torch.randn(numel + 8, dtype=torch.bfloat16)[4 : 4 + numel].view(shape)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    o = torch.zeros(numel + 8, dtype=torch.bfloat16)[4 : 4 + numel].view(shape)
    lse = torch.zeros(numel + 4, dtype=torch.float32)[2 : 2 + numel].view(shape)
    aux = torch.randn(shape, dtype=torch.bfloat16)
    assert q.data_ptr() % 16 == 8
    assert o.data_ptr() % 16 == 8
    assert lse.data_ptr() % 16 == 8
    args: tuple[object, ...] = (q, k, v, o, lse, aux, 3)
    staging = cute_launcher._cute_stage_under_aligned_operands(kernel, args)
    assert staging is not None
    staged, copy_backs = staging
    assert staged[1] is k and staged[2] is v and staged[5] is aux
    assert staged[6] == 3
    for index in (0, 3, 4):
        source, copy = args[index], staged[index]
        assert isinstance(source, torch.Tensor)
        assert isinstance(copy, torch.Tensor)
        assert copy is not source
        assert copy.data_ptr() % 16 == 0
        assert copy.stride() == source.stride()
        torch.testing.assert_close(copy, source)
    assert [(d is o, c is staged[3]) for d, c in copy_backs[:1]] == [(True, True)]
    assert [(d is lse, c is staged[4]) for d, c in copy_backs[1:]] == [(True, True)]
    # Inputs are not copied back.
    assert len(copy_backs) == 2


def test_staging_passes_aligned_and_foreign_operands_through() -> None:
    shape = (4, 16)
    aligned = tuple(torch.randn(shape, dtype=torch.bfloat16) for _ in range(5))
    flash = _fake_kernel(
        {"kind": "helion_flash", "q_idx": 0, "k_idx": 1, "v_idx": 2, "o_idx": 3}
    )
    assert cute_launcher._cute_stage_under_aligned_operands(flash, aligned) is None
    numel = math.prod(shape)
    under = torch.randn(numel + 8, dtype=torch.bfloat16)[4 : 4 + numel].view(shape)
    other = _fake_kernel({"kind": "cute_tcgen05_gemm", "a_idx": 0, "b_idx": 1})
    assert (
        cute_launcher._cute_stage_under_aligned_operands(other, (under, *aligned[1:]))
        is None
    )
    # An argument index that is not a tensor is skipped, not staged.
    scalars = _fake_kernel({"kind": "helion_flash_row_mma", "q_idx": 0, "o_idx": 1})
    assert cute_launcher._cute_stage_under_aligned_operands(scalars, (7, 8)) is None


def test_default_launcher_stages_under_aligned_operands_around_the_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launch sees 16-byte-aligned copies; the caller's buffers get the results."""
    kernel = _fake_kernel(
        {
            "kind": "helion_flash",
            "q_idx": 0,
            "k_idx": 1,
            "v_idx": 2,
            "o_idx": 3,
            "lse_idx": 4,
        }
    )
    shape = (2, 8, 16)
    numel = math.prod(shape)
    torch.manual_seed(0)
    q = torch.randn(numel + 8, dtype=torch.bfloat16)[4 : 4 + numel].view(shape)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    o = torch.zeros(numel + 8, dtype=torch.bfloat16)[4 : 4 + numel].view(shape)
    lse = torch.zeros(numel + 4, dtype=torch.float32)[2 : 2 + numel].view(shape)
    launches: list[tuple[object, ...]] = []

    def launch(
        cute_kernel: object,
        args_tuple: tuple[object, ...],
        grid_xyz: tuple[int, int, int],
        block_xyz: tuple[int, int, int],
        cute_compile_options: str | None,
    ) -> object:
        launches.append(
            (cute_kernel, args_tuple, grid_xyz, block_xyz, cute_compile_options)
        )
        # Stand in for the kernel: write into the buffers the launch was handed.
        q_, k_, v_, out, stats = (
            arg for arg in args_tuple[:5] if isinstance(arg, torch.Tensor)
        )
        out.copy_(q_ + k_ + v_)
        stats.fill_(1.0)
        return "launched"

    monkeypatch.setattr(cute_launcher, "_launch_cute_marshalled", launch)
    result = cute_launcher.default_cute_launcher(
        kernel, (3, 2), q, k, v, o, lse, 5, block=(128,)
    )
    assert result == "launched"
    ((seen_kernel, seen_args, grid_xyz, block_xyz, options),) = launches
    assert seen_kernel is kernel
    assert (grid_xyz, block_xyz, options) == ((3, 2, 1), (128, 1, 1), None)
    assert seen_args[1] is k and seen_args[2] is v and seen_args[5] == 5
    for index, source in ((0, q), (3, o), (4, lse)):
        copy = seen_args[index]
        assert isinstance(copy, torch.Tensor)
        assert copy is not source and copy.data_ptr() % 16 == 0
    torch.testing.assert_close(seen_args[0], q)
    # The results were copied back into the caller's under-aligned buffers.
    assert o.data_ptr() % 16 == 8 and lse.data_ptr() % 16 == 8
    torch.testing.assert_close(o, q + k + v)
    assert bool((lse == 1.0).all())
    # Aligned operands launch as themselves.
    launches.clear()
    aligned: tuple[torch.Tensor, ...] = (
        *(torch.zeros(shape, dtype=torch.bfloat16) for _ in range(4)),
        torch.zeros(shape),
    )
    cute_launcher.default_cute_launcher(kernel, (1,), *aligned)
    assert all(a is b for a, b in zip(launches[0][1], aligned, strict=True))
    # An empty grid launches nothing and stages nothing.
    launches.clear()
    assert cute_launcher.default_cute_launcher(kernel, (0, 1), q, k, v, o, lse) is None
    assert launches == []


def _run(
    kernel: helion.Kernel,
    family: str,
    tensors: dict[str, torch.Tensor],
) -> tuple[str, torch.Tensor, torch.Tensor]:
    _shape, config = _FAMILIES[family]
    code, (out, lse) = code_and_output(
        kernel,
        (tensors["q"], tensors["k"], tensors["v"], tensors["o"], tensors["lse"]),
        block_sizes=[1, 128, 128],
        **config,
    )
    assert "'kind': 'helion_flash'" in code
    assert out.data_ptr() == tensors["o"].data_ptr()
    assert lse.data_ptr() == tensors["lse"].data_ptr()
    return code, out, lse


@onlyBackends(["cute"])
@pytest.mark.parametrize("operand", ["q", "k", "v", "o", "lse"])
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_under_aligned_operand_matches_the_reference(family: str, operand: str) -> None:
    """Offsets of 0, 8 and 16 bytes (and 32 for the fp32 LSE) per operand.

    Four bf16 (two fp32) elements leave an 8-byte-aligned base: before the
    staging the tcgen05 bodies read q, k and v from 8 bytes before it, and
    the 128-bit output stores faulted. Eight elements restore 16 bytes.
    """
    shape, _config = _FAMILIES[family]
    kernel = helion.kernel(
        attention_into, backend="cute", static_shapes=True, autotune_effort="none"
    )
    dtype = torch.float32 if operand == "lse" else torch.bfloat16
    offsets = (0, 2, 4, 8) if operand == "lse" else (0, 4, 8)
    for offset in offsets:
        tensors = {
            "q": _offset_view(shape, torch.bfloat16, 0, seed=1),
            "k": _offset_view(shape, torch.bfloat16, 0, seed=2),
            "v": _offset_view(shape, torch.bfloat16, 0, seed=3),
            "o": _offset_view(shape, torch.bfloat16, 0, seed=4),
            "lse": _offset_view(shape[:-1], torch.float32, 0, seed=5),
        }
        tensors[operand] = _offset_view(
            shape[:-1] if operand == "lse" else shape, dtype, offset, seed=6
        )
        expected_out, expected_lse = _reference(
            tensors["q"], tensors["k"], tensors["v"]
        )
        code, out, lse = _run(kernel, family, tensors)
        if family == "fa4_2cta":
            assert "'use_2cta_instrs': True" in code
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize("family", ["ws_overlap", "fa4"])
def test_example_attention_under_aligned_q_and_k_match_the_reference(
    family: str,
) -> None:
    """The reported case: the example kernel with an 8-byte-aligned q or k."""
    from examples.attention import attention

    kernel = helion.kernel(
        attention.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    shape, config = _FAMILIES[family]
    base = {
        "q": _offset_view(shape, torch.bfloat16, 0, seed=1),
        "k": _offset_view(shape, torch.bfloat16, 0, seed=2),
        "v": _offset_view(shape, torch.bfloat16, 0, seed=3),
    }
    for operand in ("q", "k"):
        tensors = {**base, operand: _offset_view(shape, torch.bfloat16, 4, seed=6)}
        expected_out, expected_lse = _reference(
            tensors["q"], tensors["k"], tensors["v"]
        )
        code, (out, lse) = code_and_output(
            kernel,
            (tensors["q"], tensors["k"], tensors["v"]),
            block_sizes=[1, 128, 128],
            **config,
        )
        assert "'kind': 'helion_flash'" in code
        torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)


@onlyBackends(["cute"])
@pytest.mark.parametrize("operand", ["q", "k", "v", "do"])
def test_backward_under_aligned_operand_matches_the_reference(operand: str) -> None:
    """The fused backward (``helion_flash_bwd``) TMA-loads q, k, v and dO too."""
    shape = (4, 256, 64)
    kernel = helion.kernel(
        attention_bwd.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    for offset in (0, 4, 8):
        tensors = {
            name: _offset_view(shape, torch.bfloat16, 0, seed=seed)
            for seed, name in enumerate(("q", "k", "v", "do"), start=1)
        }
        tensors[operand] = _offset_view(shape, torch.bfloat16, offset, seed=6)
        (dq_ref, dk_ref, dv_ref), lse, delta = _backward_reference(
            tensors["q"], tensors["k"], tensors["v"], tensors["do"]
        )
        code, (dq, dk, dv) = code_and_output(
            kernel,
            (
                tensors["q"],
                tensors["k"],
                tensors["v"],
                lse,
                tensors["do"],
                delta,
                1.0 / math.sqrt(shape[-1]),
            ),
        )
        assert "'kind': 'helion_flash_bwd'" in code
        torch.testing.assert_close(dq, dq_ref.to(dq.dtype), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(dk, dk_ref.to(dk.dtype), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(dv, dv_ref.to(dv.dtype), atol=2e-2, rtol=2e-2)
