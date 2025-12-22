Helion Examples
===============

This directory contains examples demonstrating how to use Helion for high-performance tensor operations.
The examples are organized into the following categories:

Basic Operations
~~~~~~~~~~~~~~~~

- :doc:`add.py <add>`: Element-wise addition with broadcasting support
- :doc:`exp.py <exp>`: Element-wise exponential function
- :doc:`sum.py <sum>`: Sum reduction along the last dimension
- :doc:`long_sum.py <long_sum>`: Efficient sum reduction along a long dimension
- :doc:`softmax.py <softmax>`: Different implementations of the softmax function
- :doc:`concatenate.py <concatenate>`: Tensor concatenation along a dimension
- :doc:`low_mem_dropout.py <low_mem_dropout>`: Memory-efficient dropout implementation

Matrix Multiplication Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`matmul.py <matmul>`: Basic matrix multiplication
- :doc:`bmm.py <bmm>`: Batch matrix multiplication
- :doc:`matmul_split_k.py <matmul_split_k>`: Matrix multiplication using split-K algorithm for better parallelism
- :doc:`matmul_layernorm.py <matmul_layernorm>`: Fused matrix multiplication and layer normalization
- :doc:`fp8_gemm.py <fp8_gemm>`: Matrix multiplication using FP8 precision
- :doc:`bf16xint16_gemm.py <bf16xint16_gemm>`: BF16 x INT16 matrix multiplication
- :doc:`int4_gemm.py <int4_gemm>`: INT4 quantized matrix multiplication
- :doc:`grouped_gemm.py <grouped_gemm>`: Grouped matrix multiplication
- :doc:`gather_gemv.py <gather_gemv>`: Gather-based matrix-vector multiplication

Attention Operations
~~~~~~~~~~~~~~~~~~~~

- :doc:`attention.py <attention>`: Scaled dot-product attention mechanism
- :doc:`fp8_attention.py <fp8_attention>`: Attention mechanism using FP8 precision
- :doc:`blackwell_attention.py <blackwell_attention>`: Attention optimized for Blackwell architecture

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
