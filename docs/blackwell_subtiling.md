# Blackwell Attention Subtiling Analysis

- In `examples/blackwell_attention.py` (see `examples/blackwell_attention.py:145` and `examples/blackwell_attention.py:170`), the `SUBTILING` flag reshapes the accumulator tile into a `(tile_m, Dv // 2, 2)` view so that each lane sees a packed pair of FP32 values. This layout is required to invoke the helper `_mul_f32x2`, which emits the Blackwell-only `mul.f32x2` instruction and performs two FP32 multiplies per 64-bit register. Without the subtiled layout, the kernel would fall back to scalar multiplies and lose half of the available FP32 throughput for this epilogue step.
- Blackwell’s 5th-generation Tensor Core stack (UMMA plus Tensor Memory) partitions accumulator fragments into warp-local subtiles. When the accumulator is viewed in the subtiled shape, each warp operates solely on data that resides within its own TMEM fragment. That avoids cross-warp shuffles, keeps data in the format the hardware already produces, and lets the kernel sustain high pipe utilization.
- Combining the vectorized `mul.f32x2` epilogue with the TMEM-aware layout is what unlocks the intended Blackwell speedup: the subtiled path keeps the epilogue aligned with the hardware’s register-file and memory layout, allowing the wider TMEM bandwidth and fused-precision units to be fully exercised.

## References

- NVIDIA, *Parallel Thread Execution ISA*, CUDA 13.0.1 (Blackwell preview).
- Rong Shen, “NVIDIA Blackwell UMMA Architecture Guide — Part One,” 2024.
