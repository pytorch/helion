# Dist-MoE NVFP4 expert projections

`nvfp4_projections.py` contains the two Helion kernels tuned for a balanced
T1024, two-rank expert-parallel forward on GB300. W13 multiplies each expert's
packed inputs and interleaved gate/up weights, rounds to BF16, and applies
clamped SwiGLU. W2 multiplies the requantized intermediate and writes BF16 to
the two destination tensors only where the route belongs to that expert row.
These kernels are separate from the routing benchmark in `bench.py`.

The source comes from `meta-pytorch/dist_moe`, development commit
`bcbf4b482f63128d3af8a348b22eb8fea79a891e`:
`dist_moe/kernels/helion_nvfp4_g8.py` and the two-line scale-address helper in
`dist_moe/kernels/helion_blockscaled.py`. The routing kernel in
`layout_kernel.py` is adapted from `dist_moe/kernels/helion_routing_chunks.py`
at the same commit. The Dist-MoE license is in [LICENSE.dist-moe](LICENSE.dist-moe).

## Render W2 on the public compiler

From a checkout with the CuTe development dependencies installed:

```sh
CUDA_VISIBLE_DEVICES="" TRITON_INTERPRET=1 HELION_BACKEND=cute python - <<'PYTHON'
from benchmarks.cute.dist_moe.nvfp4_projections import (
    _nvfp4_batched_peer_rows as kernel, projection_inputs,
)
from helion._compiler.cute.target_facts import GB300
from helion._testing import cpu_render_target, default_cute_mma_support

args = projection_inputs("w2", "cpu")
with cpu_render_target(GB300, num_sm=152, matmul_num_sm=152,
                       mma_support=default_cute_mma_support(), fake_device_sm_count=152,
                       reduction_smem_budget_bytes=232448):
    bound = kernel.bind(kernel.normalize_args(*args))
    print(bound.to_code(kernel.configs[0]))
PYTHON
```

W2 uses generic lowering here. W13 is supplied as a compiler workload: the
public backend declines its native block-scaled contraction; generic lowering
with 16×16 output tiles cannot reshape the gate/up layout consumed by
`hl.split`. Use the historical compiler and config below to reproduce its
measured code.

The fixture preserves the measured tensor shapes, strides, and W13 constants
(`alpha=1.702`, `limit=7.0`). It uses deterministic synthetic values and
separate local tensors for W2's two destinations. It does not reproduce peer
communication, the routing workload, or full-forward timing.

## Historical configurations

Each projection had its own 48-proposal search. Both ranks selected the
same W13 configuration and the same W2 configuration. The complete settings
below are for the measured development compiler
`2c718b26f53c3831b18b3e9074a6cbfaca963575`. The public compiler lacks
`cute_scaled_accumulator_stages`, `cute_scaled_gated_lane_pairs`,
`cute_scaled_reuse_row_scale`, and `cute_scaled_warp_route`. No public
performance result is implied by rendering W2. Using the full settings with
the historical compiler produced these SM103a cubins byte-for-byte for both
ranks of the recorded full forward (SHA-256):

- W13: `853199ddf80a4e796abfe9e33b2b33080ef9f657a7fddb515259be3b590e8b66`
- W2: `30aa144a4c6235bf6e0a759c9d6b5a712dd52856c8062a8870112b37b1568455`

```python
shared = dict(
    block_sizes=[16, 16, 16],
    cute_collective_compute="warp",
    cute_collective_copy="scalar",
    cute_collective_epilogue="scalar",
    cute_collective_mma=False,
    cute_collective_native_seeded=False,
    cute_collective_operand_packets=False,
    cute_collective_recipe="scalar",
    cute_collective_stages=1,
    cute_collective_tmem_a=False,
    cute_collective_tmem_seed=False,
    cute_compiler_pipeline="default",
    cute_gathered_mma_n=256,
    cute_gathered_mma_stages=3,
    cute_lane_layouts=["blocked"] * 4,
    cute_pdl=False,
    cute_proven_bounds=False,
    cute_register_chain=False,
    cute_scaled_cluster_m=2,
    cute_scaled_epilogue_n=64,
    cute_scaled_mma=True,
    cute_scaled_persistent="shared",
    cute_scaled_prepare="fused",
    cute_scaled_prepare_rows=32,
    cute_scaled_tile_k=256,
    cute_store_policy="default",
    cute_vector_widths=[1] * 4,
    loop_orders=[[0, 1, 2]],
)
w13 = dict(
    **shared,
    cute_scaled_accumulator_stages=2,
    cute_scaled_gated_lane_pairs=True,
    cute_scaled_reuse_row_scale=True,
    cute_scaled_stages=6,
    cute_scaled_tile_n=128,
    load_eviction_policies=[""] * 24,
)
w2 = dict(
    **shared,
    cute_scaled_stages=5,
    cute_scaled_tile_n=256,
    cute_scaled_warp_route=True,
    load_eviction_policies=[""] * 27,
)
```
