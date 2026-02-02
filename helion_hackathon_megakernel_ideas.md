## Megakernel Workloads

The following sections cover advanced megakernel approaches that fuse entire model components or full forward passes into single GPU kernels, eliminating kernel launch overhead and enabling fine-grained pipelining.

---

### 1. LLM Inference - Full Model Megakernels

The primary driver of megakernel development today. The goal is to fuse an entire LLM forward pass into a single kernel launch.

#### Why Megakernels?

Traditional LLM inference involves:
- ~7 kernel launches per layer × 16+ layers = 100+ kernel launches per forward pass
- Each launch incurs ~5μs overhead (stragglers, launch latency, memory latency)
- At 770 forward passes/second theoretical limit, launch overhead dominates
- GPU spends more time waiting for memory than computing

Megakernels solve this by:
- Eliminating kernel launch overhead entirely
- Enabling fine-grained software pipelining across operators
- Overlapping computation with communication across GPUs
- Keeping intermediate data on-chip in shared memory

---

#### Mirage Persistent Kernel (MPK)

**The first compiler that automatically transforms LLM inference into a single megakernel.**

- **Repository**: [github.com/mirage-project/mirage](https://github.com/mirage-project/mirage) (branch: `mpk`)
- **Paper**: [arxiv.org/abs/2512.22219](https://arxiv.org/abs/2512.22219)
- **Team**: CMU, UW, Berkeley, NVIDIA, Tsinghua

##### How MPK Works

1. **SM-Level Graph Representation**: Captures data dependencies at individual Streaming Multiprocessor (SM) granularity
2. **Task Graph Compilation**: Lowers tensor programs into optimized SM-level task graphs
3. **In-Kernel Parallel Runtime**: Executes tasks using decentralized scheduling across SMs
4. **Cross-Operator Software Pipelining**: Pre-loads next operator's data while current operator computes


##### Models Supported

| Model | Size | Hardware | Speedup vs vLLM/SGLang |
|-------|------|----------|------------------------|
| Qwen3-0.6B | 0.6B | A100/H100/B200 | 1.0-1.7× |
| Qwen3-1.7B | 1.7B | A100/H100/B200 | 1.0-1.7× |
| Qwen3-8B | 8B | A100 | 14.5ms → 12.5ms (approaching 10ms theoretical limit) |


#### Hazy Research Megakernels (ThunderKittens)

**Stanford's approach: interpreter-based megakernels built on ThunderKittens primitives.**

- **Repository**: [github.com/HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels)
- **ThunderKittens**: [github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

##### Llama-1B Low-Latency Megakernel (May 2025)

**Fuses entire Llama-1B forward pass into a single CUDA kernel launch.**

The megakernel uses an **interpreter-based design**:
- **One persistent kernel** runs on the GPU and acts as an interpreter
- **Virtual instruction set**: The kernel reads "instruction tensors" from memory
- **Seven fused operations** are executed sequentially within this single kernel (no host round-trips)

This eliminates ~5μs kernel launch overhead per operation. With 100+ operations per forward pass in traditional approaches, this saves ~0.5ms per token.

```
Traditional approach (100+ kernel launches per forward pass):
  Host → Launch RMSNorm kernel → Host → Launch QKV kernel → Host → ...

Megakernel approach (1 kernel launch):
  Host → Launch Megakernel interpreter
         GPU: Execute instruction 1 (fused RMSNorm+QKV+RoPE)
         GPU: Execute instruction 2 (Attention)
         GPU: Execute instruction 3 (ThunderGQA reduction)
         ... (all within same kernel, no host synchronization)
```

The seven virtual instructions:
1. RMS norm + QKV projection + RoPE (fused)
2. Attention computation
3. Attention reduction (ThunderGQA)
4. O-projection + residual connection
5. RMS norm + up-projection + gating + SiLU (fused)
6. Down-projection + residual connection
7. RMS norm + LM head


##### ThunderMLA (March 2025)

**Fused megakernel for DeepSeek MLA (Multi-head Latent Attention).**

**Background: What is MLA?**

DeepSeek's MLA compresses the KV cache by ~93% compared to standard MHA:
- Down-projects K and V into a compressed latent vector C^KV (much smaller dimension)
- At inference time, up-projects C^KV back to full K and V before attention
- Trade-off: extra matmuls but massive memory savings

**The Problem: FlashMLA Uses Two Separate Kernels**

FlashMLA (the standard implementation) launches **two separate CUDA kernels**:
1. **Slot attention kernel**: Computes attention with decompressed K/V, produces partial results
2. **Token reduction kernel**: Aggregates partial results across tokens

This causes:
- Kernel launch overhead (~5μs × 2 = ~10μs per attention layer)
- Data cannot be reused between kernels (extra global memory round-trip)
- Poor GPU utilization on variable-length batches (tail effects)

Example: With batch=4, seq_len=variable, queries=4, FlashMLA achieves only 144 TFLOPS / 1199 GB/s on H100.

**ThunderMLA Solution: Fuse Into One Kernel**

ThunderMLA uses the same interpreter-based megakernel approach:
- Both slot attention and token reduction become **virtual instructions** in one persistent kernel
- Data stays in shared memory/registers between operations (no global memory round-trip)
- Uses **makespan backwards scheduler**: works backwards from final reduction task, runs heuristic rollouts to find optimal SM assignment (+10% over static scheduling)

```
FlashMLA (2 kernel launches):
  Host → Launch Slot Attention kernel → sync → Host → Launch Reduction kernel → sync

ThunderMLA (1 kernel launch):
  Host → Launch Megakernel
         GPU: Execute slot attention instruction
         GPU: Execute reduction instruction (data stays local)
```

**Performance**: 20-35% faster than FlashMLA across diverse workloads
**Code Size**: Just 250 lines of ThunderKittens device code

##### ThunderGQA (Grouped Query Attention)

**Fused decode attention kernel for GQA models (Llama 3, etc.).**

**Background: FlashDecoding's Two-Kernel Architecture**

Standard decode attention (FlashDecoding) uses **two separate kernels** for long sequences:

1. **Partial attention kernel**: Splits KV cache into chunks, computes attention for each chunk in parallel, outputs partial results + log-sum-exp (LSE) values
2. **Reduction kernel**: Combines partial results using LSE rescaling to get correct softmax output

This split-K approach parallelizes across the KV sequence length (essential when batch_size × num_heads < num_SMs), but the two-kernel design has overhead:
- Kernel launch overhead (~5μs × 2)
- Partial results must go to global memory between kernels
- Reduction step accounts for ~18.8% of attention computation time (per LeanAttention paper)

```
FlashDecoding (2 kernel launches):
  Host → Launch Partial Attention kernel (split KV into N chunks, compute in parallel)
       → sync, write partial outputs + LSE to global memory
  Host → Launch Reduction kernel (rescale and combine partial results)
       → sync
```

**ThunderGQA Solution: Fuse Into One Kernel**

ThunderGQA uses the interpreter-based megakernel to fuse both steps:
- Partial attention and reduction are **virtual instructions** in one persistent kernel
- Partial results stay in shared memory/registers (no global memory round-trip)
- Uses the same `interpreter::kernel<config, partial_template, reduction_template>` pattern as ThunderMLA

```
ThunderGQA (1 kernel launch):
  Host → Launch Megakernel
         GPU: Execute partial attention instruction (results stay in SMEM)
         GPU: Execute reduction instruction (combine locally)
         → write final output to global memory
```

**Why GQA Benefits More**

GQA has higher arithmetic intensity than MHA because multiple query heads share KV heads:
- Arithmetic intensity: O(H_qo / H_kv) where H_qo = query heads, H_kv = KV heads
- Example: Llama 3 uses 8 KV heads for 32 query heads (4:1 ratio)
- This allows better utilization of Tensor Cores (vs memory-bound standard MHA decode)

##### Models/Kernels Supported by Hazy Research

| Kernel/Model | Description | Hardware | Performance |
|--------------|-------------|----------|-------------|
| Llama-1B | Full forward pass megakernel | H100, B200 | 1.5-3.5× vs SGLang |
| ThunderMLA | DeepSeek MLA attention | H100, B200 | 20-35% vs FlashMLA |
| ThunderGQA | Grouped query attention | H100 | Near-cuDNN speeds |

---

