"""Multi-shape autotuning entry point for KDA prefill helpers."""

from __future__ import annotations

import argparse
from itertools import pairwise

from examples.linear.kda_prefill import _chunk_output
from examples.linear.kda_prefill import _chunk_state
from examples.linear.kda_prefill import _chunk_state_varlen
from examples.linear.kda_prefill import _intra_matrices_wide
from examples.linear.kda_prefill import _intra_matrices_wide_forward
from examples.linear.kda_prefill import _intra_solve
from examples.linear.kda_prefill import _intra_solve_recompute
from examples.linear.kda_prefill import _intra_solve_recompute_newton
from examples.linear.kda_prefill import _recompute_u
from examples.linear.kda_prefill import _recompute_w_kg
from examples.linear.kda_prefill import prepare_chunk_indices
from examples.linear.kda_prefill import prepare_chunk_offsets
import torch


def _matrix_args(
    sequence_length: int,
    varlen: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    bool,
]:
    batch, heads, key_dim = 1, 16, 128
    q = torch.nn.functional.normalize(
        torch.randn(
            batch,
            sequence_length,
            heads,
            key_dim,
            device="cuda",
        ),
        dim=-1,
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn_like(q, dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    g_step = (
        -torch.rand(
            batch,
            sequence_length,
            heads,
            key_dim,
            device="cuda",
        )
        * 0.1
    )
    beta = torch.rand(batch, sequence_length, heads, device="cuda") * 0.1
    if varlen:
        if sequence_length == 512:
            lengths = [129, 383]
        elif sequence_length == 8192:
            lengths = [513, 1023, 2049, 4607]
        else:
            first = sequence_length // 3 + 1
            lengths = [first, sequence_length - first]
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        chunk_indices = prepare_chunk_indices(cu_seqlens)
    else:
        lengths = [sequence_length]
        cu_seqlens = torch.empty(0, device="cuda", dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device="cuda", dtype=torch.long)
    g = torch.empty_like(g_step)
    sequence_begin = 0
    for length in lengths:
        for chunk_begin in range(sequence_begin, sequence_begin + length, 64):
            chunk_end = min(chunk_begin + 64, sequence_begin + length)
            g[:, chunk_begin:chunk_end] = (
                torch.cumsum(
                    g_step[:, chunk_begin:chunk_end],
                    dim=1,
                )
                * 1.4426950216293335
            )
        sequence_begin += length
    return q, k, g, beta, cu_seqlens, chunk_indices, key_dim**-0.5, varlen


def _kernel_args(
    kernel_name: str,
    sequence_length: int,
    varlen: bool,
    newton_schulz: bool,
) -> tuple[object, ...]:
    matrix_args = _matrix_args(sequence_length, varlen)
    preinvert_diagonal = not newton_schulz
    if kernel_name == "matrix":
        return (*matrix_args, preinvert_diagonal, newton_schulz)

    q, k, g, beta, cu_seqlens, chunk_indices, _, is_varlen = matrix_args
    v = torch.randn_like(q)
    matrix_kernel = (
        _intra_matrices_wide_forward if preinvert_diagonal else _intra_matrices_wide
    )
    aqk, akk = matrix_kernel(
        *matrix_args,
        preinvert_diagonal,
        newton_schulz,
    )
    qg = (q.float() * (q.size(-1) ** -0.5) * torch.exp2(g)).to(q.dtype)
    wk = (k.float() * beta[..., None] * torch.exp2(g)).to(k.dtype)
    kg = torch.empty_like(k)
    if is_varlen:
        sequence_offsets = cu_seqlens.tolist()
    else:
        sequence_offsets = [0, q.size(1)]
    chunk_decays = []
    for sequence_begin, sequence_end in pairwise(sequence_offsets):
        for chunk_begin in range(sequence_begin, sequence_end, 64):
            chunk_end = min(chunk_begin + 64, sequence_end)
            chunk_decays.append(torch.exp2(g[0, chunk_end - 1]))
            kg[:, chunk_begin:chunk_end] = (
                k[:, chunk_begin:chunk_end].float()
                * torch.exp2(
                    g[:, chunk_end - 1 : chunk_end] - g[:, chunk_begin:chunk_end]
                )
            ).to(k.dtype)
    if kernel_name == "fused":
        return (
            akk,
            wk,
            v,
            beta,
            cu_seqlens,
            chunk_indices,
            is_varlen,
            newton_schulz,
            preinvert_diagonal,
        )
    solve_args = (akk, k, cu_seqlens, chunk_indices, is_varlen)
    if kernel_name == "solve":
        return solve_args

    inverse = _intra_solve(*solve_args)
    if kernel_name == "u":
        return v, beta, inverse, cu_seqlens, chunk_indices, is_varlen
    if kernel_name == "w":
        return k, g, beta, inverse, cu_seqlens, chunk_indices, is_varlen
    if kernel_name in {"state", "output"}:
        solve_kernel = (
            _intra_solve_recompute_newton if newton_schulz else _intra_solve_recompute
        )
        w, u = solve_kernel(
            akk,
            wk,
            v,
            beta,
            cu_seqlens,
            chunk_indices,
            is_varlen,
            newton_schulz,
            preinvert_diagonal,
        )
        if is_varlen:
            num_sequences = cu_seqlens.size(0) - 1
            chunk_offsets = prepare_chunk_offsets(cu_seqlens)
        else:
            num_sequences = q.size(0)
            chunk_offsets = torch.empty(0, device="cuda", dtype=torch.long)
        initial_state = torch.zeros(
            num_sequences,
            q.size(2),
            v.size(3),
            k.size(3),
            device="cuda",
            dtype=torch.float32,
        )
        initial_state_indices = torch.arange(
            num_sequences,
            device="cuda",
            dtype=torch.int32,
        )
        state_args = (
            kg,
            w,
            u,
            torch.stack(chunk_decays),
            initial_state,
            initial_state_indices,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            is_varlen,
        )
        if kernel_name == "state":
            return state_args
        state_kernel = _chunk_state_varlen if is_varlen else _chunk_state
        h, v_new = state_kernel(*state_args)
        return (
            qg,
            v_new,
            aqk,
            h,
            v.clone(),
            cu_seqlens,
            chunk_indices,
            is_varlen,
        )
    raise AssertionError(f"unsupported kernel: {kernel_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[512, 8192])
    parser.add_argument("--cache-tag", default="kda-prefill-matrix-gb200-v1")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--newton-schulz", action="store_true")
    parser.add_argument(
        "--kernel",
        choices=["matrix", "solve", "fused", "u", "w", "state", "output"],
        default="matrix",
    )
    args = parser.parse_args()

    fused_kernel = (
        _intra_solve_recompute_newton if args.newton_schulz else _intra_solve_recompute
    )
    state_kernel = _chunk_state_varlen if args.varlen else _chunk_state
    matrix_kernel = (
        _intra_matrices_wide if args.newton_schulz else _intra_matrices_wide_forward
    )
    kernels = {
        "matrix": matrix_kernel,
        "solve": _intra_solve,
        "fused": fused_kernel,
        "u": _recompute_u,
        "w": _recompute_w_kg,
        "state": state_kernel,
        "output": _chunk_output,
    }
    winner = kernels[args.kernel].autotune_multi(
        [
            _kernel_args(
                args.kernel,
                length,
                args.varlen,
                args.newton_schulz,
            )
            for length in args.sequence_lengths
        ],
        aggregation="geomean",
        relative_to=None,
        cache_tag=args.cache_tag,
        force=True,
    )
    print(f"Multi-shape winner: {winner}")


if __name__ == "__main__":
    main()
