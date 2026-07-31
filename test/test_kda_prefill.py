from __future__ import annotations

import inspect
import math

from examples.linear.kda_prefill import chunk_kda
import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends


def _torch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = (
            q.float() / torch.sqrt((q.float().square()).sum(-1, keepdim=True) + 1e-6)
        ).to(q.dtype)
        k = (
            k.float() / torch.sqrt((k.float().square()).sum(-1, keepdim=True) + 1e-6)
        ).to(k.dtype)

    if A_log is not None:
        raw = g.float()
        if dt_bias is not None:
            raw = raw + dt_bias.reshape(g.shape[2], g.shape[3])[None, None]
        a = torch.exp(A_log.reshape(-1).float())[None, None, :, None]
        if lower_bound is None:
            gate = -a * torch.nn.functional.softplus(raw)
        else:
            gate = lower_bound * torch.sigmoid(a * raw)
    else:
        gate = g.float()

    output = torch.empty_like(v)
    h_chunks: list[torch.Tensor] = []
    if cu_seqlens is None:
        sequences = [
            (batch, batch * q.shape[1], (batch + 1) * q.shape[1])
            for batch in range(q.shape[0])
        ]
    else:
        offsets = cu_seqlens.cpu().tolist()
        sequences = [
            (sequence, offsets[sequence], offsets[sequence + 1])
            for sequence in range(len(offsets) - 1)
        ]

    for sequence, begin, end in sequences:
        state_index = int(initial_state_indices[sequence].item())
        state = initial_state[state_index].float()
        sequence_chunks: list[torch.Tensor] = []
        for flat_token in range(begin, end):
            if (flat_token - begin) % 64 == 0:
                sequence_chunks.append(state.to(q.dtype))
            if cu_seqlens is None:
                batch = sequence
                token = flat_token - begin
            else:
                batch = 0
                token = flat_token
            q_t = q[batch, token].float()
            k_t = k[batch, token].float()
            v_t = v[batch, token].float()
            beta_t = beta[batch, token].float()
            state = state * torch.exp(gate[batch, token])[:, None, :]
            prediction = (state * k_t[:, None, :]).sum(-1)
            residual = (v_t - prediction) * beta_t[:, None]
            state = state + residual[:, :, None] * k_t[:, None, :]
            output[batch, token] = (
                (state * (q_t * scale)[:, None, :]).sum(-1).to(output.dtype)
            )
        initial_state[state_index] = state.to(initial_state.dtype)
        h_chunks.extend(sequence_chunks)

    h = torch.stack(h_chunks).unsqueeze(0)
    if cu_seqlens is None:
        chunks_per_batch = math.ceil(q.shape[1] / 64)
        h = h.reshape(q.shape[0], chunks_per_batch, *h.shape[2:])
    return output, h


@onlyBackends(["triton"])
class TestKdaPrefill(RefEagerTestDisabled, TestCase):
    def test_public_signature(self) -> None:
        signature = inspect.signature(chunk_kda)
        expected_names = [
            "q",
            "k",
            "v",
            "g",
            "beta",
            "scale",
            "initial_state",
            "initial_state_indices",
            "use_qk_l2norm_in_kernel",
            "cu_seqlens",
            "A_log",
            "dt_bias",
            "lower_bound",
            "output_intermediate_states",
            "kwargs",
        ]
        self.assertEqual(list(signature.parameters), expected_names)
        self.assertEqual(
            [parameter.kind for parameter in signature.parameters.values()],
            [inspect.Parameter.POSITIONAL_OR_KEYWORD] * 14
            + [inspect.Parameter.VAR_KEYWORD],
        )
        self.assertEqual(
            [parameter.default for parameter in signature.parameters.values()],
            [inspect.Parameter.empty] * 5
            + [None, None, None, False, None, None, None, None, False]
            + [inspect.Parameter.empty],
        )

    def test_fixed_partial_chunk_and_state_pool(self) -> None:
        torch.manual_seed(123)
        B, T, H, K, V = 2, 17, 2, 32, 32
        q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
        g = torch.randn_like(q) * 0.2
        beta = torch.rand(B, T, H, device=DEVICE)
        a_log = torch.full([H], -2.0, device=DEVICE)
        dt_bias = torch.zeros(H * K, device=DEVICE)
        indices = torch.tensor([3, 1], device=DEVICE, dtype=torch.int32)
        initial = torch.randn(5, H, V, K, device=DEVICE) * 0.01

        reference_state = initial.clone()
        actual_state = initial.clone()
        expected, expected_h = _torch_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            reference_state,
            indices,
            use_qk_l2norm_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )
        output_buffer = v.clone()
        actual, actual_h = chunk_kda(
            q,
            k,
            output_buffer,
            g,
            beta,
            initial_state=actual_state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            output_intermediate_states=True,
        )

        self.assertEqual(actual.data_ptr(), output_buffer.data_ptr())
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_h, expected_h, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_state, reference_state, atol=2e-2, rtol=2e-2)
        self.assertTrue(torch.equal(actual_state[0], initial[0]))
        self.assertTrue(torch.equal(actual_state[2], initial[2]))
        self.assertTrue(torch.equal(actual_state[4], initial[4]))

    def test_varlen_safe_gate(self) -> None:
        torch.manual_seed(456)
        lengths = [1, 15, 17]
        T, H, K, V = sum(lengths), 2, 32, 32
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=DEVICE,
            dtype=torch.int32,
        )
        q = torch.randn(1, T, H, K, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, T, H, V, device=DEVICE, dtype=torch.bfloat16) * 0.1
        g = torch.randn_like(q) * 0.2
        beta = torch.rand(1, T, H, device=DEVICE)
        a_log = torch.full([H], -2.0, device=DEVICE)
        indices = torch.tensor([4, 1, 3], device=DEVICE, dtype=torch.int32)
        initial = torch.randn(6, H, V, K, device=DEVICE) * 0.01

        reference_state = initial.clone()
        actual_state = initial.clone()
        expected, expected_h = _torch_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            reference_state,
            indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            lower_bound=-0.01,
        )
        actual, actual_h = chunk_kda(
            q,
            k,
            v.clone(),
            g,
            beta,
            initial_state=actual_state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            lower_bound=-0.01,
            output_intermediate_states=True,
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_h, expected_h, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_state, reference_state, atol=2e-2, rtol=2e-2)

    def test_fp16_preactivated_gate(self) -> None:
        torch.manual_seed(789)
        B, T, H, K, V = 1, 17, 1, 32, 32
        q = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, device=DEVICE), dim=-1
        ).half()
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, device=DEVICE), dim=-1
        ).half()
        v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.float16) * 0.1
        g = -torch.rand(B, T, H, K, device=DEVICE) * 0.01
        beta = torch.rand(B, T, H, device=DEVICE)
        indices = torch.tensor([1], device=DEVICE, dtype=torch.int32)
        initial = torch.randn(3, H, V, K, device=DEVICE, dtype=torch.bfloat16) * 0.01

        reference_state = initial.clone()
        actual_state = initial.clone()
        expected, expected_h = _torch_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            reference_state,
            indices,
        )
        actual, actual_h = chunk_kda(
            q,
            k,
            v.clone(),
            g,
            beta,
            initial_state=actual_state,
            initial_state_indices=indices,
            output_intermediate_states=True,
        )

        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual_h.dtype, torch.float16)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_h, expected_h, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_state, reference_state, atol=2e-2, rtol=2e-2)
