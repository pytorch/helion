# Gemma 4 megakernel probes

The maintained compiler-generated comparisons are documented in
`probes/README.md`:

- `helion_gemma4_e4b_megakernel.py` for the E4B layer;
- `helion_gemma4_a4b_moe_megakernel.py` for the A4B MoE sub-layer; and
- their corresponding standalone Helion baselines.

All maintained performance comparisons use cold L2 by default and report the
cache mode. The probe below is retained as a historical manual-scheduling
experiment, not as the current compiler benchmark.

`triton_gemma4_codegen_schedule_probe.py` preserves the Triton bodies generated
from the existing Helion layer roots and replaces only their cross-root
dispatch, waits, publications, and continuation schedule.

On GPU 0 of the current NVIDIA B200 host at context length 8192, three fresh
processes measured 74.72--74.78 microseconds for direct keyed activation
scheduling versus 80.11--80.16 microseconds for the tuned separate Helion CUDA
graph. Each process used 50 interleaved samples of 20 graph replays, and
repeated-launch correctness passed under the historical warm-cache protocol.
Absolute performance varies materially
between otherwise-idle GPUs on this host, so comparisons must be same-process
and same-GPU. Each run's `--lowered-output` provides lowered Triton for
inspection; generated lowerings are not checked in.

Run the measured configuration from the repository root:

```bash
export PYTHONPATH="$PWD"
MEGAKERNEL_CLEAR_L2=1 python -m \
  probes.gemma4.triton_gemma4_codegen_schedule_probe \
    --layer 0 \
    --workers 576 \
    --ffn-stream \
    --ffn-scheduled-activation \
    --compare-scheduled-activation \
    --ffn-first-groups 36 \
    --qkv-block-n 8 \
    --qkv-block-k 256 \
    --qkv-range-stages 4 \
    --qkv-unroll-factor 0 \
    --o-block-n 16 \
    --o-block-k 512 \
    --o-range-stages 3 \
    --o-unroll-factor 0 \
    --gate-block-n 32 \
    --gate-block-k 256 \
    --activation-block 256 \
    --gate-range-stages 2 \
    --gate-unroll-factor 2 \
    --match-gate-eviction \
    --stream-down-stages 3 \
    --stream-down-unroll 0 \
    --ple-gate-block-k 256 \
    --poll-delay 32 \
    --fused-signals \
    --benchmark \
    --compare-helion \
    --repeats 50 \
    --batch-replays 20 \
    --lowered-output /tmp/gemma4_codegen_schedule_best_lowered.py
```

The persistent waiting kernel is launched only when the CUDA driver proves all
workers can be co-resident. This residency check is a deadlock-safety
requirement, not a performance heuristic.

The gate-weight `evict_first` annotation selected by `--match-gate-eviction`
is performance-critical in this fused envelope. Omitting it measured 84.40
microseconds in the matched post-copy run; it is codegen configuration, not a
different cross-loop schedule.
