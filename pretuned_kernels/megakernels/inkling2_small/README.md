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
| `layer_local_mxfp8.py` | local MXFP8 | attention half + MXFP8 router + routed experts |
| `attn_global.py` | global (full attention) | attention half, plus the layer's MLP input |
| `attn_block.py` | block-boundary MXFP8 (all block-boundary layers with `INK_HELION_LAYERS=0`) | attention half |
| `attn_local.py` | local, with the one-launch layer kernels off (`INK_HELION_LAYERS=0`) | attention half |
| `moe_nvfp4.py` | NVFP4, after an attention kernel | router + EP routed experts (four-over-six NVFP4) |
| `moe_mxfp8.py` | MXFP8, after an attention kernel | router + EP routed experts (MXFP8) |
| `_fp4.py` | | four-over-six NVFP4 quantization, capped SwiGLU, round-to-nearest divides |
| `_common.py` | | the kernels' shared helpers (attention: approximate rsqrt / sigmoid, L2 clip, verify conv; MoE: route selection and weights, scale tiles, MXFP8 blocks) and weight views |
| `_config.py` | | the served configs, plus per-call tensor-descriptor indexing and warp specialization (`../_ring_config.py`) |
| `_synthetic.py` | | the driver's kernel builds without vLLM: its `_Runner` and forward over a synthetic model (uninitialized weights), each launch key's first launch recorded instead of run |

Each kernel's docstring lists its roots. The attn_res mixes and RMSNorms
between sublayers are folded into the kernel that produces their input.

The NVFP4 MoE halves have R1 (W13 + SwiGLU), Z (four-over-six quantization of
h) and R2 (W2) roots. Z is one task per (256-column chunk, distinct local
expert), claimed like every root's tasks; it needs every R1 tile of its expert,
for the per-token amax over the whole intermediate.

## Serving

`vllm_helion_verify.py` is the vLLM-side driver,
`vllm/models/inkling2/helion_verify.py`. `vllm_model.patch` adds its hook to
`vllm/models/inkling2/model.py`. Both apply on the Inkling 2 vLLM branch at
`0dacf2b345` ("[Bugfix] Inkling 2: misc fixes"). The driver loads the modules
above:

```
INK_HELION=1 PYTHONPATH=<this Helion checkout> \
  vllm serve <Inkling-2-Small> --tensor-parallel-size 4 \
  <DSpark speculative config, 7 tokens> \
  --num-gpu-blocks-override 32000 --max-model-len 16384
```

The KV cache limit keeps the kernels' cache views indexable in int32 (`applies()`
asserts it).

## Results

On 4 x B200 (GSM8K, 128 prompts, batch 1, DSpark with 7 speculative tokens),
the stock vLLM Inkling 2 path ran at 773.7 tok/s and these kernels at
825.7 tok/s (1.067x). With the kernels, 120 of the 128 answers were correct:
a sanity check of the served output, with no stock-path score to compare it
with.

There is no `correctness_check()` here: the reference is vLLM's own Inkling 2
path, so the kernels were run end to end through vLLM (the GSM8K run above).
`test/test_verify_megakernels_codegen.py` generates the code of every kernel
build of the driver on CPU (`_synthetic.py`).
