"""Pretuned target-aware causal jagged HSTU attention for Pallas TPU.

The packed input tensors have shape ``[total_tokens, heads, head_dim]`` and
``seq_offsets`` partitions the token dimension into sequences.  For a sequence
ending at ``end`` with ``num_targets`` targets, history tokens use an ordinary
causal mask.  A target query attends to every history key and to its own target
key, but never to another target key.

The forward computes ``silu(Q @ K.T) @ V`` without an additional score or
output scale, matching the RankFM kernel this recipe was tuned for. Backward is
split into a dQ kernel and a fused dK/dV kernel so every output tile has a single
writer. The fixed bfloat16 streaming configuration covers the included shapes
with per-sequence lengths bounded by 16384.

Run the sharded benchmark with ``HELION_BACKEND=pallas``::

    HELION_BACKEND=pallas HELION_AOT_MODE=evaluate python -m \
      pretuned_kernels.jagged_hstu_attention.jagged_hstu_attention \
      --case s3072_b5 --kernel all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

NUM_HEADS = 4
HEAD_DIM = 128

HSTU_CONFIG = helion.Config(
    block_sizes=[512, 512],
    pallas_loop_type="emit_pipeline",
    pallas_worklist_grouping=2,
)


def _target_causal_mask(
    length: int,
    num_targets: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    query = positions[:, None]
    key = positions[None, :]
    history_end = length - num_targets
    return ((query >= key) & (key < history_end)) | (query == key)


def reference_jagged_hstu_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """Float32 PyTorch reference for the target-aware forward pass."""
    out = torch.zeros_like(v)
    for seq_idx in range(seq_offsets.size(0) - 1):
        start = int(seq_offsets[seq_idx].item())
        end = int(seq_offsets[seq_idx + 1].item())
        target_count = int(num_targets[seq_idx].item())
        length = end - start

        q_seq = q[start:end].transpose(0, 1).float()
        k_seq = k[start:end].transpose(0, 1).float()
        v_seq = v[start:end].transpose(0, 1).float()
        scores = F.silu(torch.bmm(q_seq, k_seq.transpose(-2, -1)))
        mask = _target_causal_mask(length, target_count, q.device)
        scores = torch.where(mask[None, :, :], scores, 0.0)
        out[start:end] = torch.bmm(scores, v_seq).transpose(0, 1).to(out.dtype)
    return out


def reference_jagged_hstu_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float32 analytical reference returning ``(dQ, dK, dV)``."""
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for seq_idx in range(seq_offsets.size(0) - 1):
        start = int(seq_offsets[seq_idx].item())
        end = int(seq_offsets[seq_idx + 1].item())
        target_count = int(num_targets[seq_idx].item())
        length = end - start

        q_seq = q[start:end].transpose(0, 1).float()
        k_seq = k[start:end].transpose(0, 1).float()
        v_seq = v[start:end].transpose(0, 1).float()
        do_seq = grad_out[start:end].transpose(0, 1).float()

        scores = torch.bmm(q_seq, k_seq.transpose(-2, -1))
        probability = F.silu(scores)
        d_probability = torch.bmm(do_seq, v_seq.transpose(-2, -1))
        sigmoid = torch.sigmoid(scores)
        d_scores = d_probability * sigmoid * (1.0 + scores * (1.0 - sigmoid))
        mask = _target_causal_mask(length, target_count, q.device)
        probability = torch.where(mask[None, :, :], probability, 0.0)
        d_scores = torch.where(mask[None, :, :], d_scores, 0.0)

        dq[start:end] = torch.bmm(d_scores, k_seq).transpose(0, 1).to(dq.dtype)
        dk[start:end] = (
            torch.bmm(d_scores.transpose(-2, -1), q_seq).transpose(0, 1).to(dk.dtype)
        )
        dv[start:end] = (
            torch.bmm(probability.transpose(-2, -1), do_seq)
            .transpose(0, 1)
            .to(dv.dtype)
        )

    return dq, dk, dv


@helion.aot_kernel(
    backend="pallas",
    config=HSTU_CONFIG,
    static_shapes=True,
)
def jagged_hstu_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """Target-aware causal jagged HSTU forward pass."""
    heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    out = torch.empty_like(v)

    for seq_idx in hl.grid(seq_offsets.size(0) - 1):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]

        for tile_q in hl.tile(start, end):
            q_block = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([heads, tile_q, head_dim], dtype=torch.float32)

            for tile_kv in hl.tile(start, min(end, tile_q.end)):
                k_block = k[tile_kv, :, :].transpose(0, 1)
                v_block = v[tile_kv, :, :].transpose(0, 1)
                scores = F.silu(torch.bmm(q_block, k_block.transpose(-2, -1)))

                query = tile_q.index.unsqueeze(1)
                key = tile_kv.index.unsqueeze(0)
                keep = ((query >= key) & (key < history_end)) | (query == key)
                scores = torch.where(keep[None, :, :], scores, 0.0)
                acc = acc + torch.bmm(scores.to(v.dtype), v_block)

            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


@helion.aot_kernel(
    backend="pallas",
    config=HSTU_CONFIG,
    static_shapes=True,
)
def jagged_hstu_attention_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """Compute dQ for target-aware causal jagged HSTU attention."""
    heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    dq = torch.empty_like(q)

    for seq_idx in hl.grid(seq_offsets.size(0) - 1):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]

        for tile_q in hl.tile(start, end):
            q_block = q[tile_q, :, :].transpose(0, 1)
            do_block = grad_out[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([heads, tile_q, head_dim], dtype=torch.float32)

            for tile_kv in hl.tile(start, min(end, tile_q.end)):
                k_block = k[tile_kv, :, :].transpose(0, 1)
                v_block = v[tile_kv, :, :].transpose(0, 1)
                scores = torch.bmm(q_block, k_block.transpose(-2, -1))
                d_probability = torch.bmm(do_block, v_block.transpose(-2, -1))
                sigmoid = torch.sigmoid(scores)
                d_scores = d_probability * sigmoid * (1.0 + scores * (1.0 - sigmoid))

                query = tile_q.index.unsqueeze(1)
                key = tile_kv.index.unsqueeze(0)
                keep = ((query >= key) & (key < history_end)) | (query == key)
                d_scores = torch.where(keep[None, :, :], d_scores, 0.0)
                acc = acc + torch.bmm(d_scores.to(k.dtype), k_block)

            dq[tile_q, :, :] = acc.transpose(0, 1).to(dq.dtype)

    return dq


@helion.aot_kernel(
    backend="pallas",
    config=HSTU_CONFIG,
    static_shapes=True,
)
def jagged_hstu_attention_bwd_dk_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dK and dV for target-aware causal jagged HSTU attention."""
    heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    for seq_idx in hl.grid(seq_offsets.size(0) - 1):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]

        for tile_kv in hl.tile(start, end):
            k_block = k[tile_kv, :, :].transpose(0, 1)
            v_block = v[tile_kv, :, :].transpose(0, 1)
            acc_dk = hl.zeros([heads, tile_kv, head_dim], dtype=torch.float32)
            acc_dv = hl.zeros([heads, tile_kv, head_dim], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_block = q[tile_q, :, :].transpose(0, 1)
                do_block = grad_out[tile_q, :, :].transpose(0, 1)
                scores_t = torch.bmm(k_block, q_block.transpose(-2, -1))
                d_probability_t = torch.bmm(v_block, do_block.transpose(-2, -1))
                sigmoid = torch.sigmoid(scores_t)
                probability_t = scores_t * sigmoid
                d_scores_t = (
                    d_probability_t * sigmoid * (1.0 + scores_t * (1.0 - sigmoid))
                )

                key = tile_kv.index.unsqueeze(1)
                query = tile_q.index.unsqueeze(0)
                keep_t = ((query >= key) & (key < history_end)) | (query == key)
                probability_t = torch.where(keep_t[None, :, :], probability_t, 0.0)
                d_scores_t = torch.where(keep_t[None, :, :], d_scores_t, 0.0)
                acc_dk = acc_dk + torch.bmm(d_scores_t.to(q.dtype), q_block)
                acc_dv = acc_dv + torch.bmm(probability_t.to(grad_out.dtype), do_block)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)
            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dk, dv


def jagged_hstu_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the two backward kernels and return ``(dQ, dK, dV)``."""
    dq = jagged_hstu_attention_bwd_dq(q, k, v, grad_out, seq_offsets, num_targets)
    dk, dv = jagged_hstu_attention_bwd_dk_dv(
        q, k, v, grad_out, seq_offsets, num_targets
    )
    return dq, dk, dv


_DATA_AXIS = "data"


@dataclass(frozen=True)
class ShapeCase:
    """One device-local static signature from the RankFM benchmark matrix."""

    name: str
    local_batch_size: int
    declared_max_seq_len: int
    local_tokens: int


SHAPE_CASES = {
    case.name: case
    for case in (
        ShapeCase("s4096_b65", 65, 4096, 245760),
        ShapeCase("s3072_b5", 5, 4096, 18432),
        ShapeCase("s16384_b3", 3, 16384, 32768),
    )
}


def _uniform_distribution(
    case: ShapeCase, num_devices: int
) -> tuple[np.ndarray, np.ndarray]:
    remaining_tokens = case.local_tokens - case.declared_max_seq_len
    remaining_sequences = case.local_batch_size - 1
    base_length, remainder = divmod(remaining_tokens, remaining_sequences)
    if base_length <= 0 or base_length > case.declared_max_seq_len:
        raise ValueError(f"{case.name} cannot be represented by bounded lengths")
    local_lengths = np.full(case.local_batch_size, base_length, dtype=np.int32)
    local_lengths[:remainder] += 1
    local_lengths[-1] = case.declared_max_seq_len
    lengths = np.broadcast_to(
        local_lengths, (num_devices, case.local_batch_size)
    ).copy()
    targets = np.full_like(lengths, 10)
    return lengths, targets


def _bounded_jagged_distribution(
    case: ShapeCase, num_devices: int
) -> tuple[np.ndarray, np.ndarray]:
    length_shards = []
    target_shards = []
    for device_idx in range(num_devices):
        rng = np.random.default_rng(20260813 + device_idx)
        variable_count = case.local_batch_size - 1
        remaining_tokens = case.local_tokens - case.declared_max_seq_len
        if (
            variable_count < 1
            or remaining_tokens < variable_count
            or remaining_tokens > variable_count * case.declared_max_seq_len
        ):
            raise ValueError(f"{case.name} cannot form a bounded jagged distribution")

        variable_lengths = np.empty(variable_count, dtype=np.int32)
        for index in range(variable_count):
            remaining_count = variable_count - index - 1
            minimum = max(
                1,
                remaining_tokens - remaining_count * case.declared_max_seq_len,
            )
            maximum = min(
                case.declared_max_seq_len,
                remaining_tokens - remaining_count,
            )
            length = int(rng.integers(minimum, maximum + 1))
            variable_lengths[index] = length
            remaining_tokens -= length

        lengths = np.concatenate(
            (
                variable_lengths,
                np.asarray([case.declared_max_seq_len], dtype=np.int32),
            )
        )
        rng.shuffle(lengths)
        targets = rng.integers(1, 11, size=case.local_batch_size, dtype=np.int32)
        targets = np.minimum(targets, lengths).astype(np.int32)
        length_shards.append(lengths)
        target_shards.append(targets)
    return np.stack(length_shards), np.stack(target_shards)


def make_distribution(
    case: ShapeCase,
    num_devices: int,
    distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    if distribution == "uniform":
        result = _uniform_distribution(case, num_devices)
    elif distribution == "jagged":
        result = _bounded_jagged_distribution(case, num_devices)
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    lengths, targets = result
    expected_shape = (num_devices, case.local_batch_size)
    if lengths.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError(f"distribution shape does not match {expected_shape}")
    if not np.all(lengths.sum(axis=1) == case.local_tokens):
        raise ValueError("each device must own exactly local_tokens physical rows")
    if np.any(lengths > case.declared_max_seq_len):
        raise ValueError("a sequence exceeds the pretuned maximum length")
    if np.any(targets < 0) or np.any(targets > lengths):
        raise ValueError("num_targets must be within each sequence length")
    return lengths, targets


def _run_case(
    case: ShapeCase,
    *,
    distribution: str,
    kernel_names: tuple[str, ...],
    warm_repetitions: int,
) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices())
    if not devices.size or any(device.platform != "tpu" for device in devices.flat):
        raise RuntimeError("the benchmark requires a TPU-only JAX device set")

    lengths, targets = make_distribution(case, devices.size, distribution)
    mesh = Mesh(devices, (_DATA_AXIS,))
    value_sharding = NamedSharding(mesh, P(_DATA_AXIS, None, None))
    metadata_sharding = NamedSharding(mesh, P(_DATA_AXIS))

    def sharded_constant(value: float) -> object:
        local_shape = (case.local_tokens, NUM_HEADS, HEAD_DIM)
        local_arrays = []
        for device in devices.flat:
            with jax.default_device(device):
                local_arrays.append(jnp.full(local_shape, value, jnp.bfloat16))
        global_shape = (case.local_tokens * devices.size, NUM_HEADS, HEAD_DIM)
        return jax.make_array_from_single_device_arrays(
            global_shape, value_sharding, local_arrays
        )

    q = sharded_constant(0.015625)
    k = sharded_constant(0.03125)
    v = sharded_constant(0.0625)
    grad_out = sharded_constant(0.0078125)
    lengths_array = jax.device_put(jnp.asarray(lengths.reshape(-1)), metadata_sharding)
    targets_array = jax.device_put(jnp.asarray(targets.reshape(-1)), metadata_sharding)

    value_spec = P(_DATA_AXIS, None, None)
    metadata_spec = P(_DATA_AXIS)

    def wrap_local_call(
        local_call: Callable[..., object],
        input_specs: tuple[object, ...],
        output_specs: object,
    ) -> Callable[..., object]:
        return jax.jit(
            jax.shard_map(
                local_call,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=output_specs,
                check_vma=False,
            )
        )

    def offsets_from_lengths(local_lengths: object) -> object:
        return jnp.concatenate(
            (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(local_lengths))
        )

    def local_fwd(
        local_q: object,
        local_k: object,
        local_v: object,
        local_lengths: object,
        local_targets: object,
    ) -> object:
        offsets = offsets_from_lengths(local_lengths)
        return jagged_hstu_attention_fwd.jax_fn(
            local_q, local_k, local_v, offsets, local_targets
        )

    def local_dq(
        local_q: object,
        local_k: object,
        local_v: object,
        local_grad_out: object,
        local_lengths: object,
        local_targets: object,
    ) -> object:
        offsets = offsets_from_lengths(local_lengths)
        return jagged_hstu_attention_bwd_dq.jax_fn(
            local_q, local_k, local_v, local_grad_out, offsets, local_targets
        )

    def local_dk_dv(
        local_q: object,
        local_k: object,
        local_v: object,
        local_grad_out: object,
        local_lengths: object,
        local_targets: object,
    ) -> object:
        offsets = offsets_from_lengths(local_lengths)
        return jagged_hstu_attention_bwd_dk_dv.jax_fn(
            local_q, local_k, local_v, local_grad_out, offsets, local_targets
        )

    value_inputs = {
        "fwd": (
            wrap_local_call(
                local_fwd,
                (value_spec, value_spec, value_spec, metadata_spec, metadata_spec),
                value_spec,
            ),
            (q, k, v, lengths_array, targets_array),
        ),
        "dq": (
            wrap_local_call(
                local_dq,
                (
                    value_spec,
                    value_spec,
                    value_spec,
                    value_spec,
                    metadata_spec,
                    metadata_spec,
                ),
                value_spec,
            ),
            (q, k, v, grad_out, lengths_array, targets_array),
        ),
        "dk_dv": (
            wrap_local_call(
                local_dk_dv,
                (
                    value_spec,
                    value_spec,
                    value_spec,
                    value_spec,
                    metadata_spec,
                    metadata_spec,
                ),
                (value_spec, value_spec),
            ),
            (q, k, v, grad_out, lengths_array, targets_array),
        ),
    }

    print(
        f"case={case.name} dtype=bf16 distribution={distribution} "
        f"devices={devices.size} b_local={case.local_batch_size} "
        f"max_seq={case.declared_max_seq_len} l_local={case.local_tokens} "
        f"actual_max={int(lengths.max())}"
    )
    for kernel_name in kernel_names:
        run, inputs = value_inputs[kernel_name]
        start = time.perf_counter()
        output = jax.block_until_ready(run(*inputs))
        first_seconds = time.perf_counter() - start
        warm_times = []
        for _ in range(warm_repetitions):
            start = time.perf_counter()
            output = jax.block_until_ready(run(*inputs))
            warm_times.append(time.perf_counter() - start)
        checksum = sum(
            float(jax.device_get(jnp.sum(leaf.astype(jnp.float32))))
            for leaf in jax.tree_util.tree_leaves(output)
        )
        warm_ms = 0.0 if not warm_times else 1000.0 * float(np.median(warm_times))
        print(
            f"  {kernel_name}: first={first_seconds:.3f}s "
            f"warm_median={warm_ms:.3f}ms checksum={checksum:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(*SHAPE_CASES, "all"),
        default="s3072_b5",
    )
    parser.add_argument(
        "--distribution",
        choices=("uniform", "jagged"),
        default="jagged",
    )
    parser.add_argument(
        "--kernel",
        choices=("fwd", "dq", "dk_dv", "all"),
        default="all",
    )
    parser.add_argument("--warm-repetitions", type=int, default=2)
    args = parser.parse_args()

    cases = (
        tuple(SHAPE_CASES.values()) if args.case == "all" else (SHAPE_CASES[args.case],)
    )
    kernel_names = ("fwd", "dq", "dk_dv") if args.kernel == "all" else (args.kernel,)
    config = HSTU_CONFIG.config
    print(
        "config="
        f"block_sizes={config['block_sizes']} "
        f"pallas_loop_type={config['pallas_loop_type']} "
        f"pallas_worklist_grouping={config['pallas_worklist_grouping']}"
    )
    for case in cases:
        _run_case(
            case,
            distribution=args.distribution,
            kernel_names=kernel_names,
            warm_repetitions=args.warm_repetitions,
        )


if __name__ == "__main__":
    main()
