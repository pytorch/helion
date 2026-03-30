Linear Attention Examples
=========================

Helion implementations of chunked linear attention covering the major linear
attention / state-space model variants.  Every example includes correctness
tests (forward + backward) against both a pure-PyTorch reference **and**
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(FLA) / `mamba <https://github.com/state-spaces/mamba>`_, plus benchmarks
comparing Helion kernel performance against those libraries.

This work builds on ideas from the
`Attention Engine <https://github.com/fla-org/attention-engine>`_ project.

Variants
--------

.. list-table::
   :header-rows: 1

   * - Example
     - Decay
     - Correction
     - FLA baseline
   * - ``example_simple_gla.py``
     - scalar
     - none
     - ``chunk_simple_gla``
   * - ``example_full_gla.py``
     - diagonal
     - none
     - ``chunk_gla``
   * - ``example_delta_rule.py``
     - none
     - rank-1
     - ``chunk_delta_rule``
   * - ``example_gated_delta_rule.py``
     - scalar
     - rank-1
     - ``chunk_gated_delta_rule``
   * - ``example_vanilla_linear_attn.py``
     - none
     - none
     - ``chunk_linear_attn``
   * - ``example_retention.py``
     - scalar (fixed)
     - none
     - ``chunk_retention``
   * - ``example_mamba2_ssd.py``
     - scalar
     - none
     - ``mamba_chunk_scan_combined``
   * - ``example_rwkv6.py``
     - diagonal
     - none (+ output gate)
     - ``chunk_gla`` (decay path)
   * - ``example_kda.py``
     - diagonal
     - rank-1
     - ``chunk_kda``

Architecture
------------

``linear_attention_engine.py``
    All Helion kernels (14 ``@helion.experimental.aot_kernel()`` functions),
    ``ChunkedLinearAttnFn`` autograd wrapper, ``LinearAttentionEngine`` class,
    and the ``chunked_linear_attn()`` public entry point.

``linear_attention_utils.py``
    Pure-PyTorch chunked reference implementation, naive recurrent reference,
    WY decomposition helpers, input generators, kernel config caching.

Running
-------

Run a single example::

    HELION_USE_DEFAULT_CONFIG=1 python -m examples.linear.example_simple_gla

Run all tests via pytest::

    pytest test/test_examples.py -k "test_linear" -v

Run monkey-patch tests (plugs our engine into FLA layers)::

    pytest test/test_examples.py -k "monkeypatch" -v

AOT autotuning
~~~~~~~~~~~~~~

The kernels use ``@helion.experimental.aot_kernel()`` and are ready for
ahead-of-time autotuning::

    # Single variant
    HELION_AUTOTUNE_PRECOMPILE=spawn \
    HELION_AUTOTUNE_IGNORE_ERRORS=1 \
    python -m helion.experimental.aot_runner --phase all \
      -- python -m examples.linear.aot_benchmark --variant simple_gla

    # All variants across 8 GPUs
    ./examples/linear/run_aot_tuning.sh

Acknowledgements
----------------

- `flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
  (FLA) by Songlin Yang, Yu Zhang *et al.* — the reference Triton
  implementations of GLA, DeltaNet, retention, and other linear attention
  variants that our examples test against.

- `mamba <https://github.com/state-spaces/mamba>`_ by Albert Gu and
  Tri Dao — the Mamba-2 SSD Triton kernel used as baseline for the
  Mamba-2 example.

- `Attention Engine <https://github.com/fla-org/attention-engine>`_ —
  a DSL-based approach to generating chunked linear attention kernels
  that inspired the generalized engine design.
