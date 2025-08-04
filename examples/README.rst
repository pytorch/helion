Helion Examples
==============

This directory contains examples demonstrating how to use Helion for high-performance tensor operations.
The examples are organized into the following categories:

Basic Operations
~~~~~~~~~~~~~~~

- ``add.py``: Element-wise addition with broadcasting support
- ``exp.py``: Element-wise exponential function
- ``sum.py``: Sum reduction along the last dimension
- ``long_sum.py``: Efficient sum reduction along a long dimension
- ``softmax.py``: Different implementations of the softmax function

Matrix Multiplication Operations
~~~~~~~~~~~~~~~~

- ``matmul.py``: Basic matrix multiplication
- ``bmm.py``: Batch matrix multiplication
- ``matmul_split_k.py``: Matrix multiplication using split-K algorithm for better parallelism
- ``matmul_layernorm.py``: Fused matrix multiplication and layer normalization
- ``fp8_gemm.py``: Matrix multiplication using FP8 precision

Attention Operations
~~~~~~~~~~~~~~~~~~~

- ``attention.py``: Scaled dot-product attention mechanism
- ``fp8_attention.py``: Attention mechanism using FP8 precision

Normalization
~~~~~~~~~~~~

- ``rms_norm.py``: Root Mean Square (RMS) normalization

Sparse and Jagged Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``jagged_dense_add.py``: Addition between a jagged tensor and a dense tensor
- ``jagged_mean.py``: Computing the mean of each row in a jagged tensor
- ``segment_reduction.py``: Segmented reduction operation
- ``moe_matmul_ogs.py``: Mixture-of-Experts matrix multiplication using Outer-Gather-Scatter

Other Operations
~~~~~~~~~~~~~~~

- ``concatenate.py``: Tensor concatenation along a dimension
- ``cross_entropy.py``: Cross entropy loss function
- ``embedding.py``: Embedding lookup operation
- ``all_gather_matmul.py``: All-gather operation followed by matrix multiplication

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   add
   all_gather_matmul
   attention
   bmm
   concatenate
   cross_entropy
   embedding
   exp
   fp8_attention
   fp8_gemm
   jagged_dense_add
   jagged_mean
   long_sum
   matmul
   matmul_layernorm
   matmul_split_k
   moe_matmul_ogs
   rms_norm
   segment_reduction
   softmax
   sum
