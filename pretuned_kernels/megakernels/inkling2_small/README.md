# Inkling-2-Small verify megakernels (TP4, B200)

Helion megakernels (Gluon backend, `cross_loop_pipeline="dynamic"`) for the
target-verify forward of Inkling-2-Small speculative decoding: batch 1, a
verify window of T = 8 rows, TP4 attention and EP4 experts on 4 x B200, served
in vLLM. Each layer runs as one launch, or two (attention, then MoE) where no
one-launch kernel exists.

| Module | Layers | What it runs |
|---|---|---|
| `layer_local_nvfp4.py` | local NVFP4 | attention half + router + routed experts in one launch |
| `layer_block_nvfp4.py` | block-boundary NVFP4 (every 6th layer) | the same, plus the new attn_res block's statistics |
| `layer_local_mxfp8.py` | local MXFP8 (layers 40-47) | attention half + MXFP8 router + routed experts |
| `attn_global.py` | global (full attention) | attention half, plus the layer's MLP input |
| `attn_block.py` | block-boundary MXFP8 | attention half |
| `attn_local.py` | local, when the one-launch layer kernels are off | attention half |
| `moe_nvfp4.py` | NVFP4, after an attention kernel | router + EP routed experts (four-over-six NVFP4) |
| `moe_mxfp8.py` | MXFP8, after an attention kernel | router + EP routed experts (MXFP8) |
| `_fp4.py` | | four-over-six NVFP4 quantization, capped SwiGLU, round-to-nearest divides |
| `_config.py` | | the served configs, plus per-call tensor-descriptor indexing and warp specialization |

Each kernel's docstring lists its roots. The attn_res mixes and RMSNorms
between sublayers are folded into the kernel that produces their input.

The NVFP4 MoE halves have the R1 (W13 + SwiGLU), Z (four-over-six quantization
of h) and R2 (W2) roots that claim groups interleave: Z needs every R1 tile of
its expert, for the per-token amax over the whole intermediate.

## Serving

`vllm_helion_verify.py` is a copy of the vLLM-side driver,
`vllm/models/inkling2/helion_verify.py`. `vllm_model.patch` adds its hook to
`vllm/models/inkling2/model.py`. Both apply on the Inkling 2 vLLM branch at
`0dacf2b345` ("[Bugfix] Inkling 2: misc fixes"). The driver's defaults load
the modules above:

```
INK_HELION=1 PYTHONHASHSEED=0 PYTHONPATH=<this Helion checkout> \
  vllm serve <Inkling-2-Small> --tensor-parallel-size 4 \
  <DSpark speculative config, 7 tokens> \
  --num-gpu-blocks-override 32000 --max-model-len 16384
```

The KV cache limit keeps the kernels' cache views indexable in int32.

## Results

All numbers below were measured on this branch. On 4 x B200 (GSM8K, 128
prompts, batch 1, DSpark with 7 speculative tokens), the stock vLLM Inkling 2
path ran at 773.7 tok/s and these kernels ran at 825.7 tok/s (1.067x), with
GSM8K at 120/128.

A chain of local NVFP4 layers measured about 94.9 us per layer.

There is no `correctness_check()` here: the kernels need the model's weights
and vLLM's cache layouts. They were checked end to end through vLLM (the GSM8K
accuracy above). With `INK_HELION_CHECK=1`, the driver's first eager forward
also runs the stock layer loop and logs each layer's difference.
