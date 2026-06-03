import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
print("triton", triton.__version__)


@gluon.jit
def two_cta_kernel(a_desc, b_desc, c_desc):
    gl.static_assert(gl.num_ctas() == 2)
    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout
    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)
    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_load(a_desc, [0, 0], tma_bar, smem_a)
    tma.async_load(b_desc, [0, 0], tma_bar, smem_b)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)
    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, tile_n), col_stride=1, cga_layout=cga_layout, two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)
    tcgen05_mma(smem_a, smem_b, acc, use_acc=False, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)


def run(a, b, c):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    al = gl.NVMMASharedLayout.get_default_for([M, K], gl.float16, cga_layout=[(1, 0)])
    bl = gl.NVMMASharedLayout.get_default_for([K, N], gl.float16, cga_layout=[(0, 1)])
    cl = gl.NVMMASharedLayout.get_default_for([M, N], gl.float16, cga_layout=[(1, 0)])
    ad = TensorDescriptor.from_tensor(a, [M, K], al)
    bd = TensorDescriptor.from_tensor(b, [K, N], bl)
    cd = TensorDescriptor.from_tensor(c, [M, N], cl)
    two_cta_kernel[(1,)](ad, bd, cd, num_warps=4, num_ctas=2)


M, N, K = 256, 128, 64
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
c = torch.empty((M, N), device="cuda", dtype=torch.float16)
run(a, b, c)
torch.cuda.synchronize()
err = (c.float() - torch.matmul(a, b).float()).abs().max().item()
print("2CTA fp16 max_err:", round(err, 4), "OK" if err < 0.5 else "BAD")
