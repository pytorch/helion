"""tritonbench gold-standard bench: _do_bench_cudagraph_with_cache_clear (cudagraph + L2 flush).
Run on an IDLE GPU via CUDA_VISIBLE_DEVICES."""
import sys, torch
sys.path.insert(0, "benchmarks/gluon_2cta")
sys.path.insert(0, "/home/shangdiy/tritonbench")
import scaled_mm_2cta_persistent as mc  # noqa: E402
from vllm import _custom_ops as vllm_ops  # noqa: E402
from tritonbench.components.do_bench.run import _do_bench_cudagraph_with_cache_clear as dbcg  # noqa: E402

BEST = {
    (512, 2048, 4096): dict(block_size_m=128, block_size_n=128, block_size_k=128, stages=6, acc_stages=2, epilogue_size_n=32, subtile_stages=4),
    (512, 2048, 2048): dict(block_size_m=128, block_size_n=64, block_size_k=128, stages=6, acc_stages=2, epilogue_size_n=32, subtile_stages=4),
    (512, 6144, 2048): dict(block_size_m=128, block_size_n=128, block_size_k=128, stages=6, acc_stages=2, epilogue_size_n=32, subtile_stages=4),
    (512, 2048, 12288): dict(block_size_m=128, block_size_n=128, block_size_k=128, stages=6, acc_stages=2, epilogue_size_n=32, subtile_stages=4),
}
print("dev", torch.cuda.get_device_name(), flush=True)
tot = []
for (M, K, N), cfg in BEST.items():
    a = (torch.randn((M, K), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    b = (torch.randn((K, N), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    sa = torch.rand((M,), device="cuda") * 0.5 + 0.5
    sb = torch.rand((N,), device="cuda") * 0.5 + 0.5
    mc.scaled_mm_multicta(a, b, sa, sb, out=out, **cfg); torch.cuda.synchronize()
    ref = (a.float() @ b.float()) * sa[:, None] * sb[None, :]
    rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    sa2 = sa.view(M, 1); sb2 = sb.view(N, 1); bcm = b.t().contiguous().t()
    f_v = lambda: vllm_ops.cutlass_scaled_mm(a, bcm, sa2, sb2, out_dtype=torch.bfloat16)
    f_h = lambda: mc.scaled_mm_multicta(a, b, sa, sb, out=out, **cfg)
    tv = dbcg(f_v, rep=100, return_mode="median")
    th = dbcg(f_h, rep=100, return_mode="median")
    tot.append(th / tv)
    tag = "WIN " if th <= tv * 1.02 else "MISS"
    print(f"{tag} M={M} K={K} N={N}: vLLM={tv*1e3:.2f}us mc={th*1e3:.2f}us {th/tv:.3f}x rel={rel:.4f}", flush=True)
if tot:
    print(f"avg ratio {sum(tot)/len(tot):.3f}  max {max(tot):.3f}", flush=True)
