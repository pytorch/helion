import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
print("triton", triton.__version__)


@gluon.jit
def kloop_2cta(a_desc, b_desc, c_desc, K, BK: gl.constexpr):
    gl.static_assert(gl.num_ctas() == 2)
    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, tile_n), col_stride=1, cga_layout=cga_layout, two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    n_k = K // BK
    for ki in range(n_k):
        mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, ki * BK], tma_bar, smem_a)
        tma.async_load(b_desc, [ki * BK, 0], tma_bar, smem_b)
        mbarrier.wait(tma_bar, phase=ki & 1, deps=[smem_a, smem_b])
        tcgen05_mma(smem_a, smem_b, acc, use_acc=(ki > 0), mbarriers=[mma_bar])
        mbarrier.wait(mma_bar, phase=ki & 1, deps=[smem_a, smem_b])

    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)


def run(a, b, c, BK):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    al = gl.NVMMASharedLayout.get_default_for([M, BK], gl.float8e4nv, cga_layout=[(1, 0)])
    bl = gl.NVMMASharedLayout.get_default_for([BK, N], gl.float8e4nv, cga_layout=[(0, 1)])
    cl = gl.NVMMASharedLayout.get_default_for([M, N], gl.bfloat16, cga_layout=[(1, 0)])
    ad = TensorDescriptor.from_tensor(a, [M, BK], al)
    bd = TensorDescriptor.from_tensor(b, [BK, N], bl)
    cd = TensorDescriptor.from_tensor(c, [M, N], cl)
    kloop_2cta[(1,)](ad, bd, cd, K, BK, num_warps=4, num_ctas=2)


# fp8 single tile: M=256, N=128, K=2048
M, N, K = 256, 128, 2048
af = torch.randn((M, K), device="cuda") * 0.3
bf = torch.randn((K, N), device="cuda") * 0.3
a = af.to(torch.float8_e4m3fn)
b = bf.to(torch.float8_e4m3fn)
c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
run(a, b, c, BK=64)
torch.cuda.synchronize()
ref = (a.float() @ b.float())
rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
print("fp8 K-loop 2CTA rel_err:", round(rel, 4), "OK" if rel < 0.05 else "BAD")
