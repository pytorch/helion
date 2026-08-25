# Megakernel probes

This directory contains the self-contained Qwen3 and Gemma 4 experiments used
to evaluate Helion's cross-loop scheduler. Run commands from the repository
root with exactly one idle GPU visible:

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
```

Set `MEGAKERNEL_IDLE_MEMORY_LIMIT_MB` if the driver's idle allocation exceeds
the default 256 MiB. Every comparison below validates outputs, captures CUDA
graphs, and interleaves the megakernel and separate Helion launches in the same
process.

## Recommended comparisons

Qwen3-8B layer:

```bash
python -m probes.qwen3.helion_qwen3_granular_tile_dependency \
  --strict-validation --probe-config \
  --repeats 30 --batch-replays 20
```

The optional `--task-aligned-attention` mode is a dependency-analysis stress
probe; it is not the fastest batch-one lowering.

Gemma 4 E4B layer:

```bash
python -m probes.gemma4.helion_gemma4_e4b_megakernel \
  --layer 0 --config-mode fused --cross-loop-workers 576 \
  --benchmark --repeats 30 --batch-replays 20
```

Gemma 4 26B-A4B MoE sub-layer, preserving the separate GeGLU boundary:

```bash
python -m probes.gemma4.helion_gemma4_a4b_moe_megakernel \
  --batch 1 --route-skew 2 \
  --source-mode assignment_hierarchical_topk_unfused_geglu \
  --config-mode matched --workers 444 --worker-multiplier 4 \
  --num-warps 4 --maxnreg 128 --reduce-block 256 \
  --benchmark --repeats 30 --batch-replays 50
```

Use `--print-lowered` on either Gemma megakernel, or `--dump-triton` on the
Qwen3 probe, to print the complete lowered Triton source.

## Separate Helion baselines

The comparison commands above already run these baselines. They can also be
run alone:

```bash
python -m probes.qwen3.helion_qwen3_layer_baseline --benchmark
python -m probes.gemma4.helion_gemma4_e4b_layer --layer 0 --benchmark
python -m probes.gemma4.helion_gemma4_a4b_moe \
  --batch 1 --route-skew 2 --benchmark
```

Model-specific notes and exploratory Triton scheduling probes live in
`probes/gemma4/`; Qwen3 scheduler ablations and wait profiling live in
`probes/qwen3/`.
