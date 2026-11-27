# GLM-5.3 TP4 megakernels (Gluon)

Helion megakernels for GLM-5.3 (`GlmMoeDsa`, NVFP4) speculative decoding on
4 x GB300: batch 1, EAGLE 5/1/6 (a verify window of 6 tokens), served in
SGLang. They replace the target's verify step layer by layer and
the MTP draft's input and LM head. Each kernel's docstring lists its roots.

| Module | Replaces | Runs in |
|---|---|---|
| `glm5_3_attn_front_tp4.py` | the MLA prologue: q_a / kv_a, their norms, q_b, the W_UK absorption, RoPE, the FP8 query and the window's MLA cache rows | every layer; the draft's attention |
| `glm5_3_indexer_tp4.py` | the DSA indexer's k / q preparation, its cache store and the paged FP8 logits (SGLang's top-k reads them) | the 21 layers with their own indexer; the draft extend |
| `glm5_3_attn_out_tp4.py` | the sparse MLA over the top-k slots, W_UV, o_proj, the all-reduce, the residual and the post-attention norm | every layer; the draft's attention |
| `glm5_3_moe_tp4.py` | the MoE (router, top-8, NVFP4 routed and BF16 shared experts), the all-reduce, the residual and the next layer's input norm | the 75 MoE layers |
| `glm5_3_draft_eh_tp4.py` | the draft's embedding, enorm / hnorm, the eh projection (column parallel) and its input norm | the MTP draft |
| `glm5_3_lm_head_tp4.py` | the draft's LM head GEMM, logits all-gather and top-1 | the MTP draft |
| `_setup.py` | | synthetic weights and inputs, torch references |

The 3 dense layers, the draft's BF16 MoE and the target's LM head stay
SGLang's. Every kernel sums its split-K partials and the ranks' partials in a
fixed order, with no float atomics: the outputs are run-to-run deterministic.

```
NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. python -m torch.distributed.run \
    --standalone --nproc-per-node=4 \
    pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_moe_tp4.py [--check]
```

`--check` compares every rank with the module's torch reference (FP64 for the
LM head) at each served window, and a second launch bit for bit; without it,
`main()` then times the kernel with cold weights (`LAYERS` copies back to
back in one CUDA graph; the draft eh and LM head against the stock chain).

## Numbers

SGLang with `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` on both arms,
4 x GB300, us per decode step (weighted by steps) of two interleaved same-node
pairs (pair 1 / pair 2):

| | stock | Helion | speedup |
|---|---|---|---|
| text | 9105 / 9061 | 8527 / 8525 | 1.068x / 1.063x |
| input 2k | 9381 / 9438 | 8695 / 8691 | 1.079x / 1.086x |
| input 8k | 9289 / 9341 | 8652 / 8668 | 1.074x / 1.078x |
| GSM8K decode | 9001 / 9032 | 8431 / 8424 | 1.068x / 1.072x |

GSM8K (200 questions, 5 shots): 181 stock, 184 Helion; mean accept length
3.983 stock, 3.990 Helion. Without the flag: 1.058x text, 1.085x input 2k,
1.080x input 8k (one pair).

Per launch (4 ranks, cold weights, median of 3 rounds, us): MoE 42.6;
attention front 18.4 (T=6) / 18.9 (T=1); attention out 26.6 (T=6, context
8192) / 24.9 (T=1); indexer 13.4; draft eh 19.8 (6 rows) / 20.0 (1 row); LM
head 77.4 (one row, greedy only).
