# fp8 2-CTA cluster scaled_mm (Gluon) — WORKING base, needs WS tuning

Toolchain: triton from the Azure nightly index (has the 2-CTA cluster API):
  pip install --target=<dir> --no-deps --pre \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ \
    triton    # -> triton-3.7.0+gitfeb6c04a (main-style gluon API: tma.async_load, two_ctas barriers, tcgen05_mma)
  Run with PYTHONPATH=<dir>.

Status (on B200, M=512 compute-bound shapes):
- scaled_mm_2cta_fp8.py: CORRECT fp8 2-CTA cluster GEMM (rel ~0.003), TMEM acc, multi-stage TMA.
  Key: cross-CTA (two_ctas) barriers + mbarrier.fence_init_release_cluster() after init
  (required outside warp_specialize, else 'could not find insertion point' compile error).
- Perf: 2.1-2.9x SLOWER than cutlass -- it is NOT warp-specialized, so load/MMA/epilogue
  serialize on one warpgroup. (Single-CTA WS was actually faster, 1.1-1.5x.)

Remaining step to MATCH cutlass: port the full warp-specialized multicta structure
(agent_space/multicta_ref.py, main's 14-multicta.py): 4 partitions (load/MMA/epilogue/CLC
scheduler) via gl.warp_specialize, epilogue subtiling, + add the rowwise-scale epilogue
(load scale_a[m]/scale_b[n], acc *= sa*sb, cast bf16) to matmul_epilogue_partition.
