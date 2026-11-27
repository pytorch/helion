# GLM-5.3 Flash TP4 megakernels (Gluon)

Helion megakernels for GLM-5.3 Flash (`Glm5NextForConditionalGeneration`,
NVFP4: 34 KDA and 11 MLA + DSA layers, 42 MoE and 3 dense FFNs, mHC residual
streams) speculative decoding on 4 x GB300: batch 1, EAGLE 5/1/6 (a verify
window of 6 tokens), served in SGLang. They replace the target's verify step
layer by layer, the mHC boundaries included (the last layer's MoE and exit
stay SGLang's), and the MTP draft's layer and LM head. Each kernel's docstring
lists its roots.

| Module | Replaces | Runs in |
|---|---|---|
| `kda_tp4.py` | the KDA layer (projections, conv, gated delta rule, gated norm, o_proj), the all-reduce and the attention -> FFN mHC boundary | the 34 KDA layers |
| `mla_frontidx_tp4.py` | the MLA prologue (q_a / kv_a, norms, q_b, W_UK, FP8 queries, cache rows) and the pooled DSA indexer's keys, cache writes and logits (SGLang's top-k reads them) | the 11 MLA layers |
| `mla_out_tp4.py` | the sparse MLA over the top-k, W_UV, o_proj, the all-reduce and the attention -> FFN mHC boundary | the 11 MLA layers |
| `moe_tp4.py` | the MoE (router, top-8, NVFP4 routed and shared experts), the all-reduce and the FFN -> next attention mHC boundary | 41 of the 42 MoE layers (the last layer's stays SGLang's) |
| `dense_tp4.py` | the NVFP4 dense MLP, the all-reduce and the FFN -> next attention mHC boundary | layers 0-2 |
| `mla_front_tp4.py` | the MLA prologue without the indexer | the MTP draft's attention |
| `mtp_eh_tp4.py` | the draft's embedding, enorm / hnorm, eh projection and input norm | the MTP draft |
| `mtp_attn_out_tp4.py` | the draft's sparse MLA, W_UV, o_proj, the all-reduce, residual and post-attention norm | the MTP draft |
| `mtp_moe_tp4.py` | the draft's BF16 MoE, the all-reduce, residual and final norm | the MTP draft |
| `mtp_lm_head_tp4.py` | the draft's LM head GEMM, logits all-gather and top-1 | the MTP draft |
| `_mhc.py` | | the mHC boundary's device helpers (the tails of the KDA, MLA-out, MoE and dense kernels) and host operands |
| `_common.py` | | the MoE kernels' routing and NVFP4 SwiGLU device helpers |
| `_setup.py` | | synthetic weights and inputs, torch references (with `glm5_3_tp4/_setup.py`), the kernels' compile helper |

The target's embedding and LM head stay SGLang's, and so does the DSA top-k
between `mla_frontidx_tp4` and `mla_out_tp4`. A kernel that hands its mHC comb
to another kernel passes it as logits (the `comb_out_raw` / `comb_in_raw`
modes): the consumer runs the Sinkhorn iterations off its critical path. Every
kernel sums its split-K partials and the ranks' partials in a fixed order, with
no float atomics: the outputs are run-to-run deterministic.

```
NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. python -m torch.distributed.run \
    --standalone --nproc-per-node=4 \
    pretuned_kernels/megakernels/glm5_3_flash_tp4/moe_tp4.py [--check]
```

`--check` compares every rank with the module's torch reference on synthetic
weights (the MoE at windows routed to the served 23 / 31 / 41 distinct
experts), back-to-back launches and a repeat with NaN scratch bit for bit;
without it, `main()` then times the kernel with cold weights.

## Numbers

SGLang with `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` on both arms,
4 x GB300, us per decode step of two interleaved same-node pairs (pair 1 /
pair 2):

| | stock | Helion | speedup |
|---|---|---|---|
| text | 6500 / 6489 | 5422 / 5378 | 1.199x / 1.207x |
| input 2k | 6588 / 6594 | 5514 / 5384 | 1.195x / 1.225x |
| input 8k | 6633 / 6622 | 5387 / 5399 | 1.231x / 1.226x |
| GSM8K decode | 6391 / 6384 | 5696 / 5634 | 1.122x / 1.133x |

GSM8K (200 questions, 5 shots): 181 stock, 184 Helion; mean accept length
4.117 stock, 4.107 Helion. Without the flag (SGLang's default runtime) the
served step is host-bound: 0.99-1.04x.

Per piece (4 ranks chained, slowest rank, steady state, us): KDA + MoE layer
81.0 vs 100.1 stock; KDA + dense layer 60.0 vs 76.6; draft decode step 142.6
vs 187.1 in-graph; draft extend 337 vs 404.
