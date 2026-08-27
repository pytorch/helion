# DeepSeek-V3 MoE probe

This directory contains the production-shape, batch-one DeepSeek-V3 BF16 MoE
probe used for the CLC comparison. The matched-boundary megakernel preserves
the ten standalone Helion operation boundaries and changes only their launch
and scheduling boundary.

Run the final B200 Helion comparison with one idle GPU visible:

```bash
python -m probes.deepseek_v3.helion_deepseek_v3_moe_megakernel
```

The defaults select the measured 592-worker configuration and use 200
single-replay observations with a 256 MiB L2 flush before every observation.

Compare the standalone Helion graph with vLLM's production FlashInfer TRT-LLM
MoE backend with:

```bash
VLLM_ROOT=/path/to/vllm python -m \
  probes.deepseek_v3.benchmark_deepseek_v3_moe_standalone \
  --moe-backend flashinfer_trtllm --repeats 200
```

`VLLM_ROOT` defaults to a sibling `vllm` checkout. Generated JSON, lowered
Triton, PTX, and Gantt artifacts are intentionally not part of the probe.
