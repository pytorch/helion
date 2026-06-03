import torch, triton, statistics, functools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma, tcgen05_commit,
    tcgen05_mma_barrier_count)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from vllm import _custom_ops as vllm_ops
print("triton", triton.__version__)


@gluon.aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@gluon.aggregate
class Args:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty: gl.shared_memory_descriptor
    load_ready: gl.shared_memory_descriptor
    acc_bufs: object
    acc_ready: gl.shared_memory_descriptor
    off_m: gl.tensor
    off_n: gl.tensor
    K: gl.tensor
    BK: gl.constexpr
    STAGES: gl.constexpr


@gluon.jit
def load_part(p):
    state = Counter.create(1, p.STAGES)
    nk = p.K // p.BK
    for ki in range(nk):
        pred = ki >= p.STAGES
        mbarrier.wait(p.load_empty.index(state.index), state.phase, pred=pred)
        bar = p.load_ready.index(state.index)
        mbarrier.expect(bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
        tma.async_load(p.a_desc, [p.off_m, ki * p.BK], bar, p.a_bufs.index(state.index), multicast=True)
        tma.async_load(p.b_desc, [ki * p.BK, p.off_n], bar, p.b_bufs.index(state.index), multicast=True)
        state = state.next()


@gluon.jit
def mma_part(p):
    state = Counter.create(0, p.STAGES)
    nk = p.K // p.BK
    acc = p.acc_bufs.index(0)
    use_acc = False
    for ki in range(nk):
        mbarrier.wait(p.load_ready.index(state.index), state.phase)
        tcgen05_mma(p.a_bufs.index(state.index), p.b_bufs.index(state.index), acc,
                    use_acc=use_acc, multicast=True, mbarriers=[p.load_empty.index(state.index)])
        state = state.next()
        use_acc = True
    tcgen05_commit(p.acc_ready.index(0), descs=[p.a_bufs.index(0), p.b_bufs.index(0)])


@gluon.jit
def epi_part(p):
    mbarrier.wait(p.acc_ready.index(0), 0)
    acc = p.acc_bufs.index(0)
    c_smem = gl.allocate_shared_memory(p.c_desc.dtype, p.c_desc.block_shape, p.c_desc.layout)
    c_smem.store(acc.load().to(p.c_desc.dtype))
    tma.async_copy_shared_to_global(p.c_desc, [p.off_m, p.off_n], c_smem)


@gluon.jit
def ws_gemm(a_desc, b_desc, c_desc, M, N, K,
            CLUSTER_M: gl.constexpr, TILE_N: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 2)
    cta_m: gl.constexpr = CLUSTER_M // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout
    pid = gl.program_id(0)
    num_n = (N + TILE_N - 1) // TILE_N
    off_m = (pid // num_n) * CLUSTER_M
    off_n = (pid % num_n) * TILE_N

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [STAGES] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [STAGES] + b_desc.block_shape, b_desc.layout)
    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, TILE_N), col_stride=1, cga_layout=cga_layout, two_ctas=True)
    acc_bufs = allocate_tensor_memory(gl.float32, [1, CLUSTER_M, TILE_N], acc_layout)
    mbc: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0)], multicast=True, two_ctas=True)

    load_empty = mbarrier.allocate_mbarrier(batch=STAGES)
    load_ready = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=True)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_empty.index(i), count=mbc)
        mbarrier.init(load_ready.index(i), count=1)
    acc_ready = mbarrier.allocate_mbarrier(batch=1)
    mbarrier.init(acc_ready.index(0), count=mbc)
    mbarrier.fence_init_release_cluster()

    p = Args(a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty, load_ready,
             acc_bufs, acc_ready, off_m, off_n, K, BK, STAGES)
    gl.warp_specialize([(epi_part, (p,)), (load_part, (p,)), (mma_part, (p,))], [1, 1], [24, 24])


def run(a, b, c, CM, TN, BK, STG):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    al = gl.NVMMASharedLayout.get_default_for([CM, BK], gl.float8e4nv, cga_layout=[(1, 0)])
    bl = gl.NVMMASharedLayout.get_default_for([BK, TN], gl.float8e4nv, cga_layout=[(0, 1)])
    cl = gl.NVMMASharedLayout.get_default_for([CM, TN], gl.bfloat16, cga_layout=[(1, 0)])
    ad = TensorDescriptor.from_tensor(a, [CM, BK], al)
    bd = TensorDescriptor.from_tensor(b, [BK, TN], bl)
    cd = TensorDescriptor.from_tensor(c, [CM, TN], cl)
    ncl = ((M + CM - 1) // CM) * ((N + TN - 1) // TN)
    ws_gemm[(ncl,)](ad, bd, cd, M, N, K, CM, TN, BK, STG, num_warps=4, num_ctas=2)


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
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it / Nc * 1000


for (M, N, K) in [(512, 4096, 2048), (512, 2048, 6144), (512, 12288, 2048)]:
    af = (torch.randn((M, K), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    bf = (torch.randn((K, N), device="cuda") * 0.3).to(torch.float8_e4m3fn)
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    CM, TN, BK, STG = 256, 128, 128, 6  # best-found; WS runs correctly (rel ~0.003)
    try:
        run(af, bf, c, CM, TN, BK, STG); torch.cuda.synchronize()
        ref = af.float() @ bf.float()
        rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    except Exception as ex:
        import traceback; traceback.print_exc(); print(f"M={M} N={N} K={K} ERR {str(ex)[:80]}"); continue
    sa = torch.ones((M, 1), device="cuda", dtype=torch.float32)
    sbt = torch.ones((N, 1), device="cuda", dtype=torch.float32)
    bcm = bf.t().contiguous().t()
    gv, nv = cap(functools.partial(vllm_ops.cutlass_scaled_mm, af, bcm, sa, sbt, out_dtype=torch.bfloat16))
    gh, nh = cap(functools.partial(run, af, bf, c, CM, TN, BK, STG))
    tvs, ths = [], []
    for _ in range(20):
        tvs.append(tg(gv, nv)); ths.append(tg(gh, nh))
    tv, th = statistics.median(tvs), statistics.median(ths)
    print(f"M={M} N={N} K={K}: vLLM={tv:.2f}us ws_2cta={th:.2f}us {th/tv:.3f}x rel={rel:.4f}", flush=True)
    del gv, gh; torch.cuda.empty_cache()
