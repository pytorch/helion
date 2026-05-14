Helion Examples
===============

This directory contains examples demonstrating how to use Helion for high-performance tensor operations.
The examples are organized into the following categories:

Pretuned Kernels (run as-is, no autotuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`pretuned_kernels/ <https://github.com/pytorch/helion/tree/main/pretuned_kernels>`_
directory holds runnable kernels that ship with checked-in AOT heuristic
files (currently tuned for NVIDIA B200 / sm100). Helion picks the
checked-in config at startup, so these kernels run immediately without
any online autotuning — useful as copy/paste recipes for common patterns
or as a quick way to try Helion on a supported GPU. Each kernel module
has a ``main()`` that benchmarks against the PyTorch eager baseline.

- `vector_add <https://github.com/pytorch/helion/tree/main/pretuned_kernels/vector_add>`_
  — element-wise addition.
- `softmax <https://github.com/pytorch/helion/tree/main/pretuned_kernels/softmax>`_
  — softmax with a long-context shape sweep.
- `layer_norm <https://github.com/pytorch/helion/tree/main/pretuned_kernels/layer_norm>`_
  — layer normalization across realistic hidden sizes.
- `rms_norm <https://github.com/pytorch/helion/tree/main/pretuned_kernels/rms_norm>`_
  — RMS normalization with NPOT and LLM-shaped inputs.
- `cross_entropy <https://github.com/pytorch/helion/tree/main/pretuned_kernels/cross_entropy>`_
  — cross-entropy across LLM vocabulary sizes.

To pretune one of these kernels for a different GPU (or to ship a
heuristic for your own kernel), see the
:doc:`AOT Heuristic Tuning section </deployment_autotuning>` of the
deployment guide.

Basic Operations
~~~~~~~~~~~~~~~~

- :doc:`add.py <add>`: Element-wise addition with broadcasting support
- :doc:`exp.py <exp>`: Element-wise exponential function
- :doc:`sum.py <sum>`: Sum reduction along the last dimension
- :doc:`long_sum.py <long_sum>`: Efficient sum reduction along a long dimension
- :doc:`softmax.py <softmax>`: Different implementations of the softmax function
- :doc:`batch_softmax.py <batch_softmax>`: Batched (3D) softmax with arithmetic broadcasting
- :doc:`concatenate.py <concatenate>`: Tensor concatenation along a dimension
- :doc:`low_mem_dropout.py <low_mem_dropout>`: Memory-efficient dropout implementation

Matrix Multiplication Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`matmul.py <matmul>`: Basic matrix multiplication
- :doc:`bmm.py <bmm>`: Batch matrix multiplication
- :doc:`broadcast_matmul.py <broadcast_matmul>`: Batch matrix multiplication with broadcasting (weight without batch dimension)
- :doc:`matmul_split_k.py <matmul_split_k>`: Matrix multiplication using split-K algorithm for better parallelism
- :doc:`split_k_barrier.py <split_k_barrier>`: Split-K matmul with barrier synchronization for deterministic results
- :doc:`matmul_layernorm.py <matmul_layernorm>`: Fused matrix multiplication and layer normalization
- :doc:`fp8_gemm.py <fp8_gemm>`: Matrix multiplication using FP8 precision
- :doc:`bf16xint16_gemm.py <bf16xint16_gemm>`: BF16 x INT16 matrix multiplication
- :doc:`int4_gemm.py <int4_gemm>`: INT4 quantized matrix multiplication
- :doc:`nvfp4_gemm.py <nvfp4_gemm>`: NVFP4 (E2M1) quantized matrix multiplication
- :doc:`grouped_gemm.py <grouped_gemm>`: Grouped matrix multiplication
- :doc:`gather_gemv.py <gather_gemv>`: Gather-based matrix-vector multiplication

Attention Operations
~~~~~~~~~~~~~~~~~~~~

- :doc:`attention.py <attention>`: Scaled dot-product attention mechanism
- :doc:`fp8_attention.py <fp8_attention>`: Attention mechanism using FP8 precision
- :doc:`blackwell_attention.py <blackwell_attention>`: Attention optimized for Blackwell architecture
- :doc:`flex_attention.py <flex_attention>`: Flex attention with score modification and block masking

Normalization
~~~~~~~~~~~~~

- :doc:`rms_norm.py <rms_norm>`: Root Mean Square (RMS) normalization
- :doc:`layer_norm.py <layer_norm>`: Layer normalization

Activation Functions
~~~~~~~~~~~~~~~~~~~~

- :doc:`geglu.py <geglu>`: Gated Linear Unit (GEGLU) activation
- :doc:`swiglu.py <swiglu>`: SwiGLU activation function

Loss Functions
~~~~~~~~~~~~~~

- :doc:`cross_entropy.py <cross_entropy>`: Cross entropy loss function
- :doc:`grpo_loss.py <grpo_loss>`: Group Relative Policy Optimization (GRPO) loss function
- :doc:`jsd.py <jsd>`: Jensen-Shannon Divergence
- :doc:`fused_linear_jsd.py <fused_linear_jsd>`: Fused linear layer with JSD loss
- :doc:`kl_div.py <kl_div>`: Kullback-Leibler divergence

Sparse and Jagged Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`jagged_dense_add.py <jagged_dense_add>`: Addition between a jagged tensor and a dense tensor
- :doc:`jagged_dense_bmm.py <jagged_dense_bmm>`: Batch matrix multiplication with jagged tensors
- :doc:`jagged_mean.py <jagged_mean>`: Computing the mean of each row in a jagged tensor
- :doc:`jagged_sum.py <jagged_sum>`: Sum reduction for jagged tensors
- :doc:`jagged_softmax.py <jagged_softmax>`: Softmax for jagged tensors
- :doc:`jagged_layer_norm.py <jagged_layer_norm>`: Layer normalization for jagged tensors
- :doc:`jagged_hstu_attn.py <jagged_hstu_attn>`: HSTU attention for jagged tensors
- :doc:`segment_reduction.py <segment_reduction>`: Segmented reduction operation
- :doc:`moe_matmul_ogs.py <moe_matmul_ogs>`: Mixture-of-Experts matrix multiplication using Outer-Gather-Scatter

Sequence Models
~~~~~~~~~~~~~~~

- :doc:`mamba2_chunk_scan.py <mamba2_chunk_scan>`: Mamba2 chunk scan operation
- :doc:`mamba2_chunk_state.py <mamba2_chunk_state>`: Mamba2 chunk state operation

Statistics
~~~~~~~~~~

- :doc:`welford.py <welford>`: Welford's online algorithm for computing variance

Neural Network Components
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`embedding.py <embedding>`: Embedding lookup operation
- :doc:`squeeze_and_excitation_net.py <squeeze_and_excitation_net>`: Squeeze-and-Excitation network
- :doc:`gdn_fwd_h.py <gdn_fwd_h>`: Generalized Divisive Normalization (GDN) forward pass

Advanced Usage
~~~~~~~~~~~~~~

- :doc:`aot_example.py <aot_example>`: Ahead-of-time (AOT) autotuning workflow with batch-aware heuristics
- :doc:`acfs/softmax_acf.py <acfs/softmax_acf>`: Using Advanced Controls Files (ACFs) with kernel configurations and autotuning

Distributed Operations
~~~~~~~~~~~~~~~~~~~~~~

- :doc:`distributed/all_gather_matmul.py <distributed/all_gather_matmul>`: All-gather operation followed by matrix multiplication
- :doc:`distributed/all_reduce.py <distributed/all_reduce>`: All-reduce operation (one-shot)
- :doc:`distributed/matmul_reduce_scatter.py <distributed/matmul_reduce_scatter>`: Fused matmul with reduce-scatter
- :doc:`distributed/one_shot_allreduce_bias_rmsnorm.py <distributed/one_shot_allreduce_bias_rmsnorm>`: Fused all-reduce, bias add, and RMS normalization

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:
   :glob:

   *
