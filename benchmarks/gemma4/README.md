# Gemma 4 E4B megakernel probe

`triton_gemma4_codegen_schedule_probe.py` preserves the Triton bodies generated
from the existing Helion layer roots and replaces only their cross-root
dispatch, waits, publications, and continuation schedule.

On one NVIDIA B200 at context length 8192, the retained layer-0 configuration
measured 72.19 microseconds versus 79.80 microseconds for the separate Helion
CUDA graph. A clean run from this worktree measured 72.25 versus 80.04
microseconds. Repeated-launch correctness passed. The checked-in
`triton_gemma4_codegen_schedule_best_lowered.txt` is the corresponding lowered
Triton snapshot for inspection.

Run the measured configuration from the repository root:

```bash
export PYTHONPATH="$PWD"
python benchmarks/gemma4/triton_gemma4_codegen_schedule_probe.py \
    --layer 0 \
    --workers 576 \
    --ffn-stream \
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
    --repeats 20 \
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
