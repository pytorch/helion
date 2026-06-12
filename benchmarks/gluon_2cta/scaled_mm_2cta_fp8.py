import torch, triton, statistics, functools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from vllm import _custom_ops as vllm_ops
print("triton", triton.__version__)


@gluon.jit
def gemm_2cta(a_desc, b_desc, c_desc, M, N, K,
              CLUSTER_M: gl.constexpr, TILE_N: gl.constexpr, BK: gl.constexpr,
              STAGES: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 2)
    cta_m: gl.constexpr = CLUSTER_M // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout
    pid = gl.program_id(0)
    num_n = (N + TILE_N - 1) // TILE_N
    pid_m = pid // num_n
    pid_n = pid % num_n
    off_m = pid_m * CLUSTER_M
    off_n = pid_n * TILE_N

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, TILE_N), col_stride=1, cga_layout=cga_layout, two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [CLUSTER_M, TILE_N], acc_layout)

    smem_a = gl.allocate_shared_memory(a_desc.dtype, [STAGES] + a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, [STAGES] + b_desc.block_shape, b_desc.layout)
    full = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=True)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    for s in gl.static_range(STAGES):
        mbarrier.init(full.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    mbarrier.fence_init_release_cluster()

    n_k = K // BK
    # prologue: kick off the first STAGES loads
    for s in range(STAGES):
        mbarrier.expect(full.index(s), a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [off_m, s * BK], full.index(s), smem_a.index(s))
        tma.async_load(b_desc, [s * BK, off_n], full.index(s), smem_b.index(s))

    for ki in range(n_k):
        stage = ki % STAGES
        full_phase = (ki // STAGES) & 1
        mbarrier.wait(full.index(stage), phase=full_phase,
                      deps=[smem_a.index(stage), smem_b.index(stage)])
        tcgen05_mma(smem_a.index(stage), smem_b.index(stage), acc,
                    use_acc=(ki > 0), mbarriers=[empty.index(stage)])
        mbarrier.wait(empty.index(stage), phase=full_phase,
                      deps=[smem_a.index(stage), smem_b.index(stage)])
        nxt = ki + STAGES
        if nxt < n_k:
            mbarrier.expect(full.index(stage), a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_load(a_desc, [off_m, nxt * BK], full.index(stage), smem_a.index(stage))
            tma.async_load(b_desc, [nxt * BK, off_n], full.index(stage), smem_b.index(stage))

    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)


def run(a, b, c, CLUSTER_M, TILE_N, BK, STAGES):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    al = gl.NVMMASharedLayout.get_default_for([CLUSTER_M, BK], gl.float8e4nv, cga_layout=[(1, 0)])
    bl = gl.NVMMASharedLayout.get_default_for([BK, TILE_N], gl.float8e4nv, cga_layout=[(0, 1)])
    cl = gl.NVMMASharedLayout.get_default_for([CLUSTER_M, TILE_N], gl.bfloat16, cga_layout=[(1, 0)])
    ad = TensorDescriptor.from_tensor(a, [CLUSTER_M, BK], al)
    bd = TensorDescriptor.from_tensor(b, [BK, TILE_N], bl)
    cd = TensorDescriptor.from_tensor(c, [CLUSTER_M, TILE_N], cl)
    n_clusters = ((M + CLUSTER_M - 1) // CLUSTER_M) * ((N + TILE_N - 1) // TILE_N)
    gemm_2cta[(n_clusters,)](ad, bd, cd, M, N, K, CLUSTER_M, TILE_N, BK, STAGES,
                             num_warps=4, num_ctas=2)


def cap(fn, Nc=64):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(Nc):
            fn()
    torch.cuda.synchronize()
    return g, Nc


def tg(g, Nc, it=100):
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(it):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it / Nc * 1000


for (M, N, K) in [(512, 4096, 2048), (512, 2048, 2048), (512, 2048, 6144), (512, 12288, 2048)]:
    af = (torch.randn((M, K), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    bf = (torch.randn((K, N), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    CM, TN, BK, STG = 256, 128, 64, 4
    try:
        run(af, bf, c, CM, TN, BK, STG); torch.cuda.synchronize()
        ref = af.float() @ bf.float()
        rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"M={M} N={N} K={K}: ERR {str(e)[:80]}"); continue
    print(f"  [{M}x{N}x{K}] 2cta rel={rel:.4f}", flush=True)
    # vLLM scaled_mm (dummy scales) for time comparison; b must be col-major [K,N]
    sa = torch.ones((M, 1), device="cuda", dtype=torch.float32)
    sbt = torch.ones((N, 1), device="cuda", dtype=torch.float32)
    b_cm = bf.t().contiguous().t()
    gv, nv = cap(functools.partial(vllm_ops.cutlass_scaled_mm, af, b_cm, sa, sbt, out_dtype=torch.bfloat16))
    gh, nh = cap(functools.partial(run, af, bf, c, CM, TN, BK, STG))
    tvs, ths = [], []
    for _ in range(20):
        tvs.append(tg(gv, nv)); ths.append(tg(gh, nh))
    tv, th = statistics.median(tvs), statistics.median(ths)
    print(f"M={M} N={N} K={K}: vLLM={tv:.2f}us 2cta_gemm={th:.2f}us {th/tv:.3f}x rel={rel:.3f}", flush=True)
    del gv, gh; torch.cuda.empty_cache()
