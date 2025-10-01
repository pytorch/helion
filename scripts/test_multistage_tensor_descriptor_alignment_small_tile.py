#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

import helion
import helion.language as hl
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE


BLOCK_SIZE_BT = 4  # __BLOCK_SIZE_BT__
BLOCK_SIZE_V = 256  # __BLOCK_SIZE_V__


def _build_kernel(block_size_bt: int, block_size_v: int):
    @helion.kernel(
        config=helion.Config(
            block_sizes=[block_size_bt, block_size_v],
            indexing="tensor_descriptor",
            num_stages=4,
            num_warps=4,
            pid_type="flat",
            range_flattens=[None, False],
            range_multi_buffers=[None, False],
            range_num_stages=[0, 4],
            range_unroll_factors=[0, 0],
            range_warp_specializes=[],
        ),
        static_shapes=True,
    )
    def jsd_forward_kernel(
        _input: torch.Tensor,
        target: torch.Tensor,
        shift_labels: torch.Tensor | None = None,
        beta: float = 0.5,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BT, V = _input.shape
        assert target.shape == _input.shape, (
            f"Shape mismatch: {target.shape} != {_input.shape}"
        )
        block_size_n = hl.register_block_size(V)
        block_size_m = hl.register_block_size(BT)

        # Create output tensor for accumulating loss
        loss = torch.zeros([BT], dtype=torch.float32, device=_input.device)
        dX = torch.empty_like(loss)

        one_minus_beta = 1 - beta

        # Count non-ignored elements
        n_non_ignore = float(BT)
        if shift_labels is not None:
            n_non_ignore = float((shift_labels != ignore_index).sum().item())
            if n_non_ignore == 0:
                return torch.zeros(
                    [], dtype=_input.dtype, device=_input.device
                ), torch.zeros_like(_input)

        # Process each sequence position
        for tile_bt in hl.tile(BT, block_size=block_size_m):
            # Check for label masking
            if shift_labels is not None:
                if shift_labels[tile_bt] == ignore_index:
                    for tile_X in hl.tile(V):
                        dX[tile_bt, tile_X] = 0.0
                    continue
            intermediate_loss = hl.zeros([tile_bt, block_size_n], dtype=torch.float32)
            intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=_input.dtype)
            for tile_v in hl.tile(V, block_size=block_size_n):
                # Load log probabilities and convert to float32
                X = _input[tile_bt, tile_v]
                Y = target[tile_bt, tile_v]

                if beta == 0.0:  # Forward KL: KL(P || Q)
                    Y_max = torch.amax(Y, dim=0)
                    Y_shift = Y - Y_max
                    Y_prob = torch.exp(Y_shift) * torch.exp(
                        Y_max
                    )  # Compensate for the shift
                    intermediate_loss += Y_prob * (Y - X)
                    intermediate_dX += -Y_prob
                elif beta == 1.0:  # Reverse KL: KL(Q || P)
                    X_max = torch.amax(X, dim=0)
                    X_shift = X - X_max
                    X_prob = torch.exp(X_shift) * torch.exp(
                        X_max
                    )  # Compensate for the shift
                    intermediate_loss += X_prob * (X - Y)
                    intermediate_dX += intermediate_loss + X_prob
                else:  # General JSD: beta*KL(P||M) + (1-beta)*KL(Q||M)
                    Q = torch.exp(X)  # = exp(X)
                    P = torch.exp(Y)  # = exp(Y)

                    beta_P = beta * P
                    one_minus_beta_Q = one_minus_beta * Q
                    M = beta_P + one_minus_beta_Q
                    log_M = torch.log(
                        M
                    )
                    x_minus_log_m = X - log_M
                    kl_q_m = one_minus_beta_Q * x_minus_log_m
        
                    intermediate_loss += beta_P * (Y - log_M) + kl_q_m
                    intermediate_dX += kl_q_m

            # Accumulate over vocabulary dimension
            scale = 1.0 / n_non_ignore
            loss[tile_bt] = torch.sum(intermediate_loss * scale, dim=1)
            dX[tile_bt] = torch.sum(intermediate_dX * scale, dim=1)

        # Normalize by number of non-ignored elements, run it on host to match liger_kernel
        final_loss = torch.sum(
            loss
        )
        return final_loss, dX

    return jsd_forward_kernel


def run(block_size_bt: int, block_size_v: int) -> None:
    if not supports_tensor_descriptor():
        raise SystemExit("Tensor descriptor support is required to run this example.")

    kernel = _build_kernel(block_size_bt, block_size_v)

    vocab = 512
    batch = 512
    log_q = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)
    log_p = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)

    kernel(log_q, log_p)
    print(f"Ran kernel with block_sizes=[{block_size_bt}, {block_size_v}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JSD forward kernel with custom block sizes.")
    parser.add_argument(
        "--block-size-bt",
        type=int,
        default=BLOCK_SIZE_BT,
        help="Block size for the batch dimension.",
    )
    parser.add_argument(
        "--block-size-v",
        type=int,
        default=BLOCK_SIZE_V,
        help="Block size for the vocabulary dimension.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.block_size_bt, args.block_size_v)
