# Vanilla linear attention: fused vs split forward

H100, bf16, `HELION_AUTOTUNE_EFFORT=quick`. `%FLA = fla_ms / helion_ms * 100`
(>100% = Helion faster). FLA transposed to native layout outside the timer.

## Forward

| shape | path | helion ms | fla ms | %FLA |
|---|---|---|---|---|
| B1_T8192_H96_D128 | fused | 0.442 | 0.727 | **164.4%** |
|  | split | 0.757 | 0.726 | 95.9% |
| B2_T16384_H16_D128 | fused | 0.334 | 0.610 | **182.6%** |
|  | split | 0.462 | 0.608 | 131.7% |
| B4_T2048_H16_D128 | fused | 0.113 | 0.148 | **130.1%** |
|  | split | 0.156 | 0.146 | 93.3% |
| B4_T4096_H64_D128 | fused | 0.724 | 0.953 | **131.6%** |
|  | split | 0.908 | 0.951 | 104.8% |
| B8_T2048_H32_D256 | fused | 1.640 | 1.337 | 81.6% |
|  | split | 1.380 | 1.337 | **96.9%** |
| B8_T1024_H8_D64 | fused | 0.078 | 0.145 | **186.0%** |
|  | split | 0.146 | 0.142 | 97.5% |

## Forward + backward

| shape | path | helion ms | fla ms | %FLA |
|---|---|---|---|---|
| B1_T8192_H96_D128 | fused | 1.583 | 3.370 | **212.9%** |
|  | split | 1.902 | 3.363 | 176.8% |
| B2_T16384_H16_D128 | fused | 1.152 | 2.592 | **225.0%** |
|  | split | 1.270 | 2.587 | 203.8% |
| B4_T2048_H16_D128 | fused | 0.649 | 1.485 | **228.8%** |
|  | split | 0.901 | 0.945 | 104.9% |
| B4_T4096_H64_D128 | fused | 2.355 | 4.525 | 192.2% |
|  | split | 2.533 | 4.526 | **178.7%** |
| B8_T2048_H32_D256 | fused | 4.842 | 8.989 | 185.7% |
|  | split | 4.585 | 8.998 | **196.3%** |
| B8_T1024_H8_D64 | fused | 0.748 | 1.225 | **163.8%** |
|  | split | 0.783 | 1.189 | 151.9% |

## Per-program working set (symbolic)

C = chunk size (64), D = key dim, DV = value dim. `tile_d` / `tile_dv` =
autotuner-chosen D-block / DV-block.

### fused (`chunk_fwd_fused`): grid `[BH, DV]` (D untiled), serial over N

| tensor | shape | dtype | bytes | lifetime |
|---|---|---|---|---|
| `h_acc` | `[D, tile_dv]` | fp32 | 4·D·tile_dv | **across chunks** |
| `q_i`, `k_i` | `[C, D]` | bf16 | 2·C·D | per chunk |
| `v_i` | `[C, tile_dv]` | bf16 | 2·C·tile_dv | per chunk |
| `o_cross` | `[C, tile_dv]` | fp32 | 4·C·tile_dv | per chunk |
| `attn` | `[C, C]` | fp32 | 4·C² | per chunk |
| `causal` | `[C, C]` | fp32 | 4·C² | per chunk |
| `o_intra` | `[C, tile_dv]` | fp32 | 4·C·tile_dv | per chunk |

### split pass 1 (`chunk_fwd_h`): grid `[BH, D, DV]` (D tiled to tile_d), serial over N

| tensor | shape | dtype | bytes | lifetime |
|---|---|---|---|---|
| `h_acc` | `[tile_d, tile_dv]` | fp32 | 4·tile_d·tile_dv | across chunks |
| `k_i` | `[C, tile_d]` | bf16 | 2·C·tile_d | per chunk |
| `v_i` | `[C, tile_dv]` | bf16 | 2·C·tile_dv | per chunk |

### split pass 2 (`chunk_fwd_o`): grid `[BHN, DV]` (N parallel), inner loop over tile_d

| tensor | shape | dtype | bytes | lifetime |
|---|---|---|---|---|
| `o_cross` | `[1, C, tile_dv]` | fp32 | 4·C·tile_dv | across D-tiles |
| `attn` | `[1, C, C]` | fp32 | 4·C² | across D-tiles |
| `causal` | `[C, C]` | fp32 | 4·C² | per D-tile |
| `qt`, `kt` | `[1, C, tile_d]` | bf16 | 2·C·tile_d | per D-tile |
| `ht` | `[1, tile_d, tile_dv]` | bf16 | 2·tile_d·tile_dv | per D-tile |
| `o_intra` | `[1, C, tile_dv]` | fp32 | 4·C·tile_dv | per D-tile |

### Why fused is slower at D=256

Fused forward at D=256 is 81.6% of FLA vs split's 96.9%, the only shape where
fused loses. Three tensors scale with D:

| tensor | bytes |
|---|---|
| `h_acc` | 4·D·tile_dv |
| `q_i`, `k_i` | 2·C·D |

At D=256 these overflow the register file and **spill to local memory**.

The table below is from ncu, profiling the forward only. Each row is one GPU
kernel: fused launches 1 kernel, split launches 2 (`chunk_fwd_h` then
`chunk_fwd_o`). `grid` is the number of parallel programs. `local stores` is the
number of spilled writes: 0 means everything fit in registers (no spill).

| shape | path | kernel | grid | regs/thread | local stores |
|---|---|---|---|---|---|
| B1_T8192_H96_D128 | fused | chunk_fwd_fused | 96 | 230 | 0 |
|  | split | chunk_fwd_h | 384 | 78 | 0 |
|  | split | chunk_fwd_o | 12288 | 173 | 0 |
| B2_T16384_H16_D128 | fused | chunk_fwd_fused | 1056 | 184 | 0 |
|  | split | chunk_fwd_h | 4224 | 128 | 0 |
|  | split | chunk_fwd_o | 8192 | 124 | 0 |
| B4_T2048_H16_D128 | fused | chunk_fwd_fused | 128 | 232 | 0 |
|  | split | chunk_fwd_h | 64 | 218 | 0 |
|  | split | chunk_fwd_o | 4096 | 126 | 0 |
| B4_T4096_H64_D128 | fused | chunk_fwd_fused | 256 | 236 | 0 |
|  | split | chunk_fwd_h | 256 | 128 | 0 |
|  | split | chunk_fwd_o | 16384 | 173 | 0 |
| **B8_T2048_H32_D256** | **fused** | **chunk_fwd_fused** | **1024** | **255 (cap)** | **2.1M (spills)** |
|  | split | chunk_fwd_h | 512 | 214 | 0 |
|  | split | chunk_fwd_o | 16384 | 110 | 0 |
| B8_T1024_H8_D64 | fused | chunk_fwd_fused | 256 | 92 | 0 |
|  | split | chunk_fwd_h | 256 | 54 | 0 |
|  | split | chunk_fwd_o | 1024 | 122 | 0 |

Only fused D=256 spills: it hits the 255-register hardware cap and still needs
more, so the overflow goes to local memory. Every other fused shape, and every split kernel, is under the cap with zero local traffic, because split tiles D at the cost of the `h_all` HBM round-trip.

## Reproduce

ms / %FLA tables.

Fused:
```
cd /home/dev/linatt-helion/helion && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick .venv/bin/python -m examples.linear_attn.bench
```
Split:
```
cd /home/dev/linatt-helion/helion && LINATT_NO_FUSED_FWD=1 CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick .venv/bin/python -m examples.linear_attn.bench
```

Spill table.

Fused:
```
cd /home/dev/linatt-helion/helion && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick ncu --kernel-name regex:chunk_fwd --launch-count 6 --metrics launch__grid_size,launch__registers_per_thread,smsp__sass_inst_executed_op_local_st.sum .venv/bin/python -m examples.linear_attn.one_shape 8 2048 32 256
```
Split:
```
cd /home/dev/linatt-helion/helion && LINATT_NO_FUSED_FWD=1 CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick ncu --kernel-name regex:chunk_fwd --launch-count 6 --metrics launch__grid_size,launch__registers_per_thread,smsp__sass_inst_executed_op_local_st.sum .venv/bin/python -m examples.linear_attn.one_shape 8 2048 32 256
```
