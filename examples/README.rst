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


Matrix Multiplication Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`matmul.py <matmul>`: Basic matrix multiplication
- :doc:`bmm.py <bmm>`: Batch matrix multiplication
- :doc:`matmul_split_k.py <matmul_split_k>`: Matrix multiplication using split-K algorithm for better parallelism
- :doc:`matmul_layernorm.py <matmul_layernorm>`: Fused matrix multiplication and layer normalization
- :doc:`fp8_gemm.py <fp8_gemm>`: Matrix multiplication using FP8 precision

Attention Operations
~~~~~~~~~~~~~~~~~~~~

- :doc:`attention.py <attention>`: Scaled dot-product attention mechanism
- :doc:`fp8_attention.py <fp8_attention>`: Attention mechanism using FP8 precision

Normalization
~~~~~~~~~~~~~

- :doc:`rms_norm.py <rms_norm>`: Root Mean Square (RMS) normalization

Sparse and Jagged Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`jagged_dense_add.py <jagged_dense_add>`: Addition between a jagged tensor and a dense tensor
- :doc:`jagged_mean.py <jagged_mean>`: Computing the mean of each row in a jagged tensor
- :doc:`segment_reduction.py <segment_reduction>`: Segmented reduction operation
- :doc:`moe_matmul_ogs.py <moe_matmul_ogs>`: Mixture-of-Experts matrix multiplication using Outer-Gather-Scatter

Other Operations
~~~~~~~~~~~~~~~~

- :doc:`concatenate.py <concatenate>`: Tensor concatenation along a dimension
- :doc:`cross_entropy.py <cross_entropy>`: Cross entropy loss function
- :doc:`embedding.py <embedding>`: Embedding lookup operation
- :doc:`all_gather_matmul.py <all_gather_matmul>`: All-gather operation followed by matrix multiplication
- :doc:`all_reduce.py <all_reduce>`: All-reduce operation (one-shot)

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:
   :glob:

   *
