# Hopper TMA Misaligned-Address Debugging Notes

## Symptom
Running `python repro_634_jsd.py` (or the minimal `repro_tma_misaligned.py`) on a Hopper GPU aborts with

```
RuntimeError: Triton Error [CUDA]: misaligned address
```

`compute-sanitizer --error-exitcode=99 python repro_634_jsd.py` pinpoints the fault to `_helion_jsd_forward+0x64c0`, reporting *"Misaligned shared or local address"* for the TMA-backed tensor descriptor load loop in `repro_634_jsd.py:25`.

## Root Cause
The kernel uses `tl.make_tensor_descriptor(...).load` inside a `tl.range(..., num_stages=4, flatten=False)` for tiles of shape `1x4` (16 bytes in `float32`). Triton lowers this to the Hopper `UTMALDG.2D` instruction, copying a tile from global memory into shared memory via the TMA unit.

Disassembly of `~/.triton/cache/NBROAAXGWG46K2KYFBR4CB7FL73EN3V6KDVBCLFKA4C3E5LSQPUA/_helion_jsd_forward.cubin` shows the failing instruction:

```
/*64c0*/ UTMALDG.2D [UR44], [UR42]
```

The destination pointer `UR44` equals `global_smem + 0x7b8`. `global_smem` is declared as `.shared .align 16`, so the effective address is only 16-byte aligned, but `UTMALDG` requires the shared-memory target to be 128-byte aligned. Hopper therefore raises `cudaErrorMisalignedAddress`.

## Confirmation Tests
- Re-running the same kernel with `num_stages=1` (no pipelined stages) avoids the TMA path, and execution succeeds.
- Increasing the tile width so each transfer is ≥128 bytes (e.g. `_BLOCK_SIZE_1 = 32`) also succeeds.
- The standalone script `repro_tma_misaligned.py` (no Helion dependencies beyond the allocator helper) reproduces the failure, proving the bug is in Triton’s TMA lowering rather than Helion-specific logic.

## Workarounds and Fixes
1. **Immediate mitigation**: Guard generated kernels so that tile widths below 128 bytes fall back to `num_stages=1` (or avoid tensor descriptors entirely). This prevents TMA code generation and eliminates the misalignment.
2. **Alternative**: Increase the per-tile width to at least 128 bytes when multi-stage pipelines are desirable.
3. **Long-term fix**: Update Triton so that any shared-memory buffer used as a TMA destination is padded to 128-byte alignment (or otherwise adjusted). This requires an upstream Triton change; the minimal `repro_tma_misaligned.py` plus the disassembly snippet provide a concise reproducer for filing the bug.

## Useful Commands
- `compute-sanitizer --error-exitcode=99 python repro_634_jsd.py`
- `nvdisasm -c ~/.triton/cache/.../_helion_jsd_forward.cubin | sed -n '1760,1810p'`

These expose the misaligned shared-memory address and the specific `UTMALDG.2D` instruction causing the crash.
