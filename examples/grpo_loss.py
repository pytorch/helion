"""
Helion GRPO Loss Implementation
===============================

This example demonstrates a Helion kernel implementation of Group Relative Policy Optimization (GRPO) loss.
GRPO is a reinforcement learning algorithm used for training language models with human feedback.

The implementation includes:
1. Forward pass computing GRPO loss with clipping and KL regularization
2. Backward pass for gradient computation
3. Support for completion masking and temperature scaling
4. Comparison with PyTorch reference implementation
"""

# %%
# Imports
# -------

from __future__ import annotations

import time
from typing import Callable
from typing import cast

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

# %%
# Helper Functions
# ----------------


def extract_selected_logits_pytorch(
    logits: torch.Tensor, completion_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    # Gather only the needed elements; avoid full-tensor cast and huge index grids
    sel = logits.gather(dim=2, index=completion_ids.unsqueeze(-1)).squeeze(-1)
    return sel.to(torch.float32) / temperature


def get_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities for given logits and input IDs.

    Args:
        logits: Logits tensor of shape [B, L+1, V]
        input_ids: Input token IDs of shape [B, L]

    Returns:
        Log probabilities of shape [B, L]
    """
    per_token_logps = []
    for logits_row, input_ids_row in zip(
        logits, input_ids[:, -logits.size(1) :], strict=True
    ):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)
        ).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def torch_grpo_loss(
    logits: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None,
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    PyTorch reference implementation of GRPO loss.

    Args:
        logits: Logits tensor of shape [B, L+1, V]
        old_logp: Old log probabilities of shape [B, L] or None
        ref_logp: Reference log probabilities of shape [B, L] or None
        completion_ids: Completion token IDs of shape [B, L]
        advantages: Advantages of shape [B]
        completion_mask: Completion mask of shape [B, L] or None
        temperature: Temperature scaling factor
        beta: KL regularization weight
        eps_low: Lower clipping bound
        eps_high: Upper clipping bound

    Returns:
        Tuple of (loss, kl_loss, is_clipped)
    """
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True

    logits = logits[:, :-1]  # Remove last token
    per_token_logps = get_log_probs(logits / temperature, completion_ids)
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()

    coef_1 = torch.exp(per_token_logps - old_logp)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    if completion_mask is not None:
        per_token_loss = per_token_loss * completion_mask

    per_token_kl = None
    if beta != 0.0 and ref_per_token_logps is not None:
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    return per_token_loss, per_token_kl, is_clipped


# %%
# Helion GRPO Loss Kernels
# ------------------------


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def grpo_loss_forward(
    logits: torch.Tensor,  # [B, L+1, V] input logits
    selected_logits: torch.Tensor,  # [B, L] pre-computed selected logits
    old_logp: torch.Tensor | None,  # [B, L] old log probabilities
    ref_logp: torch.Tensor | None,  # [B, L] reference log probabilities
    advantages: torch.Tensor,  # [B] advantages
    completion_mask: torch.Tensor | None,  # [B, L] completion mask
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helion kernel for GRPO loss forward pass.

    Args:
        logits: Logits tensor of shape [B, L+1, V]
        selected_logits: Pre-computed selected logits of shape [B, L]
        old_logp: Old log probabilities of shape [B, L] or None
        ref_logp: Reference log probabilities of shape [B, L] or None
        advantages: Advantages of shape [B]
        completion_mask: Completion mask of shape [B, L] or None
        temperature: Temperature scaling factor
        beta: KL regularization weight
        eps_low: Lower clipping bound
        eps_high: Upper clipping bound

    Returns:
        Tuple of (loss, kl_loss, is_clipped, lse)
    """
    B, L_ADD_1, V = logits.shape
    L = L_ADD_1 - 1

    logits = logits[:, :-1, :]  # [B, L, V]

    loss = torch.zeros([B, L], dtype=torch.float32, device=logits.device)
    is_clipped = torch.zeros([B, L], dtype=torch.float32, device=logits.device)
    kl_loss = torch.zeros([B, L], dtype=torch.float32, device=logits.device)
    lse = torch.zeros([B, L], dtype=torch.float32, device=logits.device)

    for tile_b, tile_l in hl.tile([B, L]):
        max_logits = hl.full([tile_b, tile_l], float("-inf"), dtype=torch.float32)
        sum_exp = hl.zeros([tile_b, tile_l], dtype=torch.float32)

        for tile_v in hl.tile(V):
            logits_tile = logits[tile_b, tile_l, tile_v].to(torch.float32) / temperature
            new_m_i = torch.maximum(max_logits, torch.amax(logits_tile, dim=-1))
            alpha = torch.exp(max_logits - new_m_i)
            sum_exp = sum_exp * alpha + torch.sum(
                torch.exp(logits_tile - new_m_i[:, :, None]), dim=-1
            )
            max_logits = new_m_i

        log_sum_exp = max_logits + torch.log(sum_exp)  # [tile_b, tile_l]
        lse[tile_b, tile_l] = log_sum_exp

        logp = selected_logits[tile_b, tile_l] - log_sum_exp

        if old_logp is None:
            old_logp_val = logp
        else:
            old_logp_val = old_logp[tile_b, tile_l]

        coef_1 = torch.exp(logp - old_logp_val)
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)

        advantage = advantages[tile_b]

        per_token_loss1 = coef_1 * advantage[:, None]
        per_token_loss2 = coef_2 * advantage[:, None]
        per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)

        if completion_mask is not None:
            per_token_loss *= completion_mask[tile_b, tile_l]

        if beta != 0.0 and ref_logp is not None:
            ref_logp_val = ref_logp[tile_b, tile_l]
            kl = torch.exp(ref_logp_val - logp) - (ref_logp_val - logp) - 1
            if completion_mask is not None:
                kl *= completion_mask[tile_b, tile_l]
            per_token_loss += beta * kl
            kl_loss[tile_b, tile_l] = kl

        loss[tile_b, tile_l] = per_token_loss
        is_clipped[tile_b, tile_l] = (per_token_loss1 < per_token_loss2).float()

    return loss, kl_loss, is_clipped, lse


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def grpo_loss_backward(
    grad_output: torch.Tensor,  # [B, L] gradient from downstream
    logits: torch.Tensor,  # [B, L+1, V] original logits
    selected_logits: torch.Tensor,  # [B, L] pre-computed selected logits
    completion_ids: torch.Tensor,  # [B, L] completion token IDs (needed for gradients)
    old_logp: torch.Tensor | None,  # [B, L] old log probabilities
    ref_logp: torch.Tensor | None,  # [B, L] reference log probabilities
    advantages: torch.Tensor,  # [B] advantages
    completion_mask: torch.Tensor | None,  # [B, L] completion mask
    lse: torch.Tensor,  # [B, L] stored log-sum-exp values
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> torch.Tensor:
    """
    Helion kernel for GRPO loss backward pass.

    Args:
        grad_output: Gradient from downstream layers [B, L]
        logits: Original logits tensor [B, L+1, V]
        selected_logits: Pre-computed selected logits [B, L]
        completion_ids: Completion token IDs [B, L] (needed for gradients)
        old_logp: Old log probabilities [B, L] or None
        ref_logp: Reference log probabilities [B, L] or None
        advantages: Advantages [B]
        completion_mask: Completion mask [B, L] or None
        lse: Stored log-sum-exp values [B, L]
        temperature: Temperature scaling factor
        beta: KL regularization weight
        eps_low: Lower clipping bound
        eps_high: Upper clipping bound

    Returns:
        Gradient with respect to logits [B, L+1, V]
    """
    B, L_ADD_1, V = logits.shape
    L = L_ADD_1 - 1

    logits_fwd = logits[:, :-1, :]  # [B, L, V]

    grad_logits = torch.zeros_like(logits)

    for tile_b, tile_l in hl.tile([B, L]):
        completion_id = completion_ids[tile_b, tile_l]

        log_sum_exp = lse[tile_b, tile_l]

        logp = selected_logits[tile_b, tile_l] - log_sum_exp

        if old_logp is None:
            old_logp_val = logp
        else:
            old_logp_val = old_logp[tile_b, tile_l]

        coef_1 = torch.exp(logp - old_logp_val)
        coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)

        advantage = advantages[tile_b]

        per_token_loss1 = coef_1 * advantage[:, None]
        per_token_loss2 = coef_2 * advantage[:, None]

        mask = (per_token_loss2 >= per_token_loss1).float()

        dlogp = -per_token_loss1 * mask

        if beta != 0.0 and ref_logp is not None:
            ref_logp_val = ref_logp[tile_b, tile_l]
            dlogp += beta * (1 - torch.exp(ref_logp_val - logp))

        dlogp = dlogp * grad_output[tile_b, tile_l] / temperature

        if completion_mask is not None:
            mask_val = completion_mask[tile_b, tile_l]
            dlogp *= mask_val

        for tile_v in hl.tile(V):
            logits_tile = (
                logits_fwd[tile_b, tile_l, tile_v].to(torch.float32) / temperature
            )
            probs = torch.exp(logits_tile - log_sum_exp[:, :, None])

            v_indices = tile_v.index
            sel = v_indices[None, None, :] == completion_id[:, :, None]

            grad_logits_tile = torch.where(
                sel,
                dlogp[:, :, None] * (1 - probs),
                -dlogp[:, :, None] * probs,
            )
            grad_logits[tile_b, tile_l, tile_v] = grad_logits_tile

    grad_logits[:, -1, :] = 0

    return grad_logits


# %%
# GRPO Loss Function Class
# ------------------------


class GrpoLossFunction(torch.autograd.Function):
    """Custom autograd function for GRPO loss with forward and backward passes."""

    @staticmethod
    def forward(
        ctx: object,
        logits: torch.Tensor,
        old_logp: torch.Tensor | None,
        ref_logp: torch.Tensor | None,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor | None,
        temperature: float,
        beta: float,
        eps_low: float,
        eps_high: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of GRPO loss."""
        selected_logits = extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )

        loss, kl_loss, is_clipped, lse = grpo_loss_forward(
            logits,
            selected_logits,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        ctx.save_for_backward(  # type: ignore[attr-defined]
            logits,
            selected_logits,
            completion_ids,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            lse,
        )
        ctx.temperature = temperature  # type: ignore[attr-defined]
        ctx.beta = beta  # type: ignore[attr-defined]
        ctx.eps_low = eps_low  # type: ignore[attr-defined]
        ctx.eps_high = eps_high  # type: ignore[attr-defined]

        return loss, kl_loss, is_clipped

    @staticmethod
    def backward(
        ctx: object,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward pass of GRPO loss."""
        # Unpack incoming gradients (we only need the first one for 'loss')
        grad_loss = grad_outputs[0]

        (
            logits,
            selected_logits,
            completion_ids,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            lse,
        ) = ctx.saved_tensors  # type: ignore[attr-defined]

        grad_logits = grpo_loss_backward(
            grad_loss,
            logits,
            selected_logits,
            completion_ids,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            lse,
            ctx.temperature,  # type: ignore[attr-defined]
            ctx.beta,  # type: ignore[attr-defined]
            ctx.eps_low,  # type: ignore[attr-defined]
            ctx.eps_high,  # type: ignore[attr-defined]
        )

        return (
            grad_logits,  # d(logits)
            None,  # d(old_logp)
            None,  # d(ref_logp)
            None,  # d(completion_ids)
            None,  # d(advantages)
            None,  # d(completion_mask)
            None,  # d(temperature)
            None,  # d(beta)
            None,  # d(eps_low)
            None,  # d(eps_high)
        )


def helion_grpo_loss(
    logits: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    temperature: float = 0.9,
    beta: float = 0.04,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helion implementation of GRPO loss.

    Args:
        logits: Logits tensor of shape [B, L+1, V]
        old_logp: Old log probabilities of shape [B, L] or None
        ref_logp: Reference log probabilities of shape [B, L] or None
        completion_ids: Completion token IDs of shape [B, L]
        advantages: Advantages of shape [B]
        completion_mask: Completion mask of shape [B, L] or None
        temperature: Temperature scaling factor
        beta: KL regularization weight
        eps_low: Lower clipping bound
        eps_high: Upper clipping bound

    Returns:
        Tuple of (loss, kl_loss, is_clipped)
    """
    result = cast(
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
        GrpoLossFunction.apply(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        ),
    )
    loss, kl_loss, is_clipped = result
    return loss, kl_loss, is_clipped


# %%
# Verification and Testing
# ------------------------


def compare_tensors(
    tensor1: torch.Tensor | None, tensor2: torch.Tensor | None, name: str = ""
) -> None:
    """Compare two tensors and print statistics."""
    if tensor1 is None or tensor2 is None:
        return
    if any([tensor1.dtype == torch.float32, tensor2.dtype == torch.float32]):
        tensor1, tensor2 = tensor1.float(), tensor2.float()
    diff = (tensor1 - tensor2).abs()
    diff = diff / (torch.max(tensor1.abs(), tensor2.abs()) + 1e-5)
    print(f"Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")


def test_grpo_loss(
    B: int = 8,
    L: int = 1024,
    V: int = 12800,
    temperature: float = 0.9,
    beta: float = 0.2,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
) -> None:
    """Test GRPO loss implementation against PyTorch reference."""
    print(f"Testing GRPO Loss: B={B}, L={L}, V={V}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    logits1 = torch.randn(
        B, L + 1, V, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    logits2 = logits1.clone().detach().requires_grad_(True)
    logits_ref = logits1.detach().clone().float().requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, L), dtype=torch.int64, device=DEVICE)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)
    ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)

    print("\n=== Forward Pass Test ===")

    loss_ref, kl_ref, clipped_ref = torch_grpo_loss(
        logits_ref,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
    )

    loss_helion, kl_helion, clipped_helion = helion_grpo_loss(
        logits2,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
    )

    compare_tensors(loss_helion, loss_ref, "Loss")
    compare_tensors(kl_helion, kl_ref, "KL Loss")
    compare_tensors(clipped_helion, clipped_ref, "Is Clipped")

    print("\n=== Backward Pass Test ===")

    grad_output = torch.randn_like(loss_ref)

    loss_ref.backward(grad_output, retain_graph=True)
    grad_ref = logits_ref.grad.clone() if logits_ref.grad is not None else None

    logits_ref.grad = None

    loss_helion.backward(grad_output, retain_graph=True)
    grad_helion = logits2.grad.clone() if logits2.grad is not None else None

    compare_tensors(grad_helion, grad_ref, "Gradient")

    print("\n=== Test Complete ===")


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_timing(run_fn: Callable[[], None], iters: int, warmup: int) -> float:
    times = []
    for _ in range(warmup):
        run_fn()
        _cuda_sync()
    for _ in range(iters):
        t0 = time.perf_counter()
        run_fn()
        _cuda_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 == 1 else 0.5 * (times[mid - 1] + times[mid])


def benchmark_grpo_loss(
    B: int = 8,
    L: int = 1024,
    V: int = 12800,
    temperature: float = 0.9,
    beta: float = 0.2,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
    iters: int = 50,
    warmup: int = 10,
) -> None:
    print(
        f"Benchmarking GRPO Loss: B={B}, L={L}, V={V}  (iters={iters}, warmup={warmup})"
    )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    logits_ref = torch.randn(
        B, L + 1, V, device=DEVICE, dtype=torch.float32, requires_grad=True
    )
    logits_hel = logits_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)

    completion_ids = torch.randint(0, V - 1, (B, L), dtype=torch.int64, device=DEVICE)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)
    ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)

    grad_out = torch.randn(B, L, device=DEVICE, dtype=torch.float32)

    def run_torch_fwd() -> None:
        torch_grpo_loss(
            logits_ref,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

    def run_torch_bwd() -> None:
        logits_ref.grad = None
        loss_ref, _, _ = torch_grpo_loss(
            logits_ref,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )
        loss_ref.backward(grad_out, retain_graph=False)

    def run_helion_fwd() -> None:
        helion_grpo_loss(
            logits_hel,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

    def run_helion_bwd() -> None:
        logits_hel.grad = None
        loss_hel, _, _ = helion_grpo_loss(
            logits_hel,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )
        loss_hel.backward(grad_out, retain_graph=False)

    torch_fwd_ms = _measure_timing(run_torch_fwd, iters, warmup)
    torch_bwd_ms = _measure_timing(run_torch_bwd, iters, warmup)
    hel_fwd_ms = _measure_timing(run_helion_fwd, iters, warmup)
    hel_bwd_ms = _measure_timing(run_helion_bwd, iters, warmup)

    def speedup(a: float, b: float) -> float:
        return a / b if b > 0 else float("inf")

    print("\n=== Timing (median ms) ===")
    print(f"PyTorch  Forward: {torch_fwd_ms:.3f} ms")
    print(f"PyTorch  Backward: {torch_bwd_ms:.3f} ms")
    print(
        f"Helion   Forward: {hel_fwd_ms:.3f} ms  (x{speedup(torch_fwd_ms, hel_fwd_ms):.2f} vs Torch)"
    )
    print(
        f"Helion   Backward: {hel_bwd_ms:.3f} ms  (x{speedup(torch_bwd_ms, hel_bwd_ms):.2f} vs Torch)"
    )

    tokens = B * L
    print("\n=== Throughput ===")
    print(f"PyTorch  Fwd tokens/s: {tokens / (torch_fwd_ms / 1000.0):.1f}")
    print(f"PyTorch  Bwd tokens/s: {tokens / (torch_bwd_ms / 1000.0):.1f}")
    print(f"Helion   Fwd tokens/s: {tokens / (hel_fwd_ms / 1000.0):.1f}")
    print(f"Helion   Bwd tokens/s: {tokens / (hel_bwd_ms / 1000.0):.1f}")


# %%
# Main Function
# -------------


def main() -> None:
    """Main entry point for GRPO loss testing."""
    print("Helion GRPO Loss Implementation")
    print("=" * 50)

    test_configs = [
        {"B": 8, "L": 2048, "V": 64000},
        # {"B": 4, "L": 2048, "V": 128000},
        # {"B": 8, "L": 4096, "V": 100000},
    ]

    for config in test_configs:
        test_grpo_loss(**config)
        print()

    benchmark_grpo_loss(
        B=8,
        L=2048,
        V=64000,
        temperature=0.9,
        beta=0.2,
        eps_low=0.2,
        eps_high=0.4,
        iters=50,
        warmup=10,
    )


if __name__ == "__main__":
    main()
