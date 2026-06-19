"""Benchmark fp8_gemm_skinny_m on a skinny-M shape, cold-L2.

Usage:
  python bench_skinny.py --shape 16x4096x14336 --who torch
  python bench_skinny.py --shape 16x4096x14336 --who helion            # autotune
  python bench_skinny.py --shape 16x4096x14336 --who helion --cfg f.py # pinned
"""
from __future__ import annotations
import argparse, importlib.util, os

os.environ.setdefault("HELION_BACKEND", "cute")

import torch
import helion
from examples import fp8_gemm as fp8_mod
from tritonbench.components.do_bench.run import do_bench_wrapper


def make_inputs(M, N, K):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 0.4).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda") * 0.4).to(torch.float8_e4m3fn)
    y = b.t()
    sa = (torch.rand(M, 1, device="cuda") + 0.5).float().expand(M, N)
    sb = (torch.rand(N, 1, device="cuda") + 0.5).float().reshape(-1).expand(N)
    return a, y, sa, sb


def torch_ref(a, y, sa, sb):
    return torch._scaled_mm(a, y, sa[:, :1].contiguous(), sb.reshape(1, -1).contiguous(),
                            use_fast_accum=False, out_dtype=torch.bfloat16)


def measure(fn, skip=False):
    lat = do_bench_wrapper(fn, warmup=None, rep=None, repcnt=None, grad_to_none=None,
                           device="cuda", use_cuda_graphs=True,
                           latency_measure_mode="triton_do_bench", skip_cache_clearing=skip)
    return lat.p50 if hasattr(lat, "p50") else float(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--who", default="helion")
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()
    M, N, K = (int(x) for x in args.shape.split("x"))
    a, y, sa, sb = make_inputs(M, N, K)
    ref = torch_ref(a, y, sa, sb)

    if args.who == "torch":
        fn = lambda: torch_ref(a, y, sa, sb)
        name = "torch._scaled_mm"
    else:
        k = fp8_mod.fp8_gemm_skinny_m
        k.reset()
        k.settings.static_shapes = True
        if args.cfg:
            os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
            spec = importlib.util.spec_from_file_location("cfgmod", args.cfg)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            bound = k.bind((a, y, sa, sb))
            bound.set_config(mod.FIXED_CFG)
            name = f"helion-skinny[{os.path.basename(args.cfg)}]"
        else:
            bound = k.bind((a, y, sa, sb))
            cfg = bound.autotune((a, y, sa, sb))
            print("AUTOTUNED:", repr(cfg))
            name = "helion-skinny[autotuned]"
        out = bound(a, y, sa, sb)
        relerr = (out.float() - ref.float()).norm() / ref.float().norm()
        print(f"relerr vs torch = {relerr:.2e}")
        fn = lambda: bound(a, y, sa, sb)

    p50 = measure(fn)
    flops = 2 * M * N * K
    print(f"[{name}] {args.shape}  p50={p50*1e3:.3f}us  {flops/p50/1e9:.0f}TFLOP/s  (cold-L2)")


if __name__ == "__main__":
    main()
