"""FP8 RowWise scaled_mm via the multicta persistent WS gluon kernel (committed copy) (adapted from
triton main's 14-multicta.py fp16 reference). Persistent + ACC double-buffer + epilogue
subtiling + CLC snake scheduler. fp8 e4m3 inputs, bf16 output, f32 rowwise scales.

Run: PYTHONPATH=agent_space/tnew TRITON_CACHE_DIR=agent_space/tc TMPDIR=agent_space/tt \
  conda run -n pytorch-3.12 --no-capture-output python agent_space/scaled_mm_multicta.py
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, clc, tcgen05_commit, tcgen05_mma,
    tcgen05_mma_barrier_count, tensor_memory_descriptor)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


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


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


def get_cga_layout(layout, op_idx, two_ctas):
    assert op_idx in (0, 1)
    if not layout:
        return layout

    def broadcast(b):
        mul = 2 if two_ctas else 1
        return (b[0], 0) if op_idx == 0 else (0, mul * b[1])

    if not two_ctas:
        return tuple(map(broadcast, layout))
    assert layout[0] == (1, 0)
    first = (1, 0) if op_idx == 0 else (0, 1)
    return (first, *map(broadcast, layout[1:]))


@gluon.jit
def _planar_snake(lin_idx, m_tiles, n_tiles, minor_dim: gl.constexpr, tile_width: gl.constexpr):
    major_size = n_tiles if minor_dim == 0 else m_tiles
    minor_size = m_tiles if minor_dim == 0 else n_tiles
    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_tiles * tile_width * major_size
    minor_tile_idx = lin_idx // (tile_width * major_size)
    full_minor_within = lin_idx % tile_width
    full_major_within = (lin_idx // tile_width) % major_size
    full_minor = minor_tile_idx * tile_width + full_minor_within
    full_major = gl.where((minor_tile_idx % 2) == 0, full_major_within, major_size - 1 - full_major_within)
    partial_width = minor_size - full_minor_size
    partial_width = gl.where(partial_width > 0, partial_width, 1)
    partial_lin = lin_idx - full_elements
    partial_minor_within = partial_lin % partial_width
    partial_major_within = (partial_lin // partial_width) % major_size
    partial_minor = minor_tile_idx * tile_width + partial_minor_within
    partial_major = gl.where((minor_tile_idx % 2) == 0, partial_major_within, major_size - 1 - partial_major_within)
    in_full_tile = lin_idx < full_elements
    minor = gl.where(in_full_tile, full_minor, partial_minor)
    major = gl.where(in_full_tile, full_major, partial_major)
    if minor_dim == 0:
        return minor, major
    return major, minor


@gluon.aggregate
class ClcTileSchedulerConsumer:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(M, N, TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr,
                   GRID_TILE_WIDTH: gl.constexpr, clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
                   clc_planar_ready_bars, clc_consumed_bars):
        tile_id = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, TILE_M)
        num_pid_n = gl.cdiv(N, TILE_N)
        pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, MINOR_DIM, GRID_TILE_WIDTH)
        return ClcTileSchedulerConsumer(
            gl.to_tensor(True), tile_id, pid_m, pid_n, num_pid_m, num_pid_n, TILE_M, TILE_N,
            MINOR_DIM, GRID_TILE_WIDTH, clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
            clc_planar_ready_bars, clc_consumed_bars,
            Counter.create(0, clc_barriers.shape[0]), Counter.create(0, clc_barriers.shape[0]))

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()
        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        mbarrier.wait(self.clc_planar_ready_bars.index(counter.index), counter.phase)
        planar_slot = self.clc_planar_pid_buffers.index(counter.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        packed_pid = planar_slot.load(planar_layout).reshape([])
        pid_m = ((packed_pid >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed_pid & 0xFFFFFFFF).to(gl.int32)
        has_work = clc_res.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = clc_res.program_id(0)
        return ClcTileSchedulerConsumer(
            has_work, tile_id, pid_m, pid_n, self.num_pid_m, self.num_pid_n, self.TILE_M, self.TILE_N,
            self.MINOR_DIM, self.GRID_TILE_WIDTH, self.clc_result_buffers, self.clc_barriers,
            self.clc_planar_pid_buffers, self.clc_planar_ready_bars, self.clc_consumed_bars,
            counter.next(), consumed_counter)


@gluon.aggregate
class MatmulPartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    sa_ptr: gl.tensor
    sb_ptr: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    SUBTILE_STAGES: gl.constexpr

    @gluon.jit
    def get_clc_consumer(self):
        return ClcTileSchedulerConsumer.initialize(
            self.c_desc.shape[0], self.c_desc.shape[1], self.a_desc.block_shape[0],
            self.b_desc.block_shape[1], self.MINOR_DIM, self.GRID_TILE_WIDTH,
            self.clc_result_buffers, self.clc_barriers, self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars, self.clc_consumed_bars)


@gluon.jit
def matmul_clc_partition(p):
    tile_m: gl.constexpr = p.a_desc.block_shape[0]
    tile_n: gl.constexpr = p.b_desc.block_shape[1]
    has_work = gl.to_tensor(True)
    num_pid_m = gl.cdiv(p.c_desc.shape[0], tile_m)
    num_pid_n = gl.cdiv(p.c_desc.shape[1], tile_n)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    acc_stages: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        mbarrier.wait(p.clc_consumed_bars.index(consumed_state.index), consumed_state.phase, pred=(i >= acc_stages))
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        mbarrier.expect(barrier, 16)
        clc.try_cancel(result, barrier)
        mbarrier.wait(barrier, state.phase)
        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = clc_res.program_id(0)
            pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, p.MINOR_DIM, p.GRID_TILE_WIDTH)
        packed_pid = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        planar_slot = p.clc_planar_pid_buffers.index(state.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        planar_slot.store(gl.full([1], packed_pid, gl.int64, layout=planar_layout))
        mbarrier.arrive(p.clc_planar_ready_bars.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


@gluon.jit
def matmul_load_partition(p):
    block_k: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]
    concurrent_loads: gl.constexpr = p.load_ready_bars.shape[0]
    state = Counter.create(1, concurrent_loads)
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        for k in range(0, K, block_k):
            pred = (i > 0) or (k >= block_k * concurrent_loads)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase, pred=pred)
            bar = p.load_ready_bars.index(state.index)
            mbarrier.expect(bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
            tma.async_load(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index), multicast=True)
            tma.async_load(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index), multicast=True)
            state = state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def matmul_mma_partition(p):
    block_k: gl.constexpr = p.a_desc.block_shape[1]
    K = p.a_desc.shape[1]
    acc_stages: gl.constexpr = p.acc_empty_bars.shape[0]
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, acc_stages)
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        acc_buf = p.acc_bufs.index(acc_state.index)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase, pred=(i >= acc_stages))
        use_acc = False
        for k in range(0, K, block_k):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index), acc_buf,
                        use_acc=use_acc, multicast=True, mbarriers=[p.load_empty_bars.index(load_state.index)])
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index), descs=[p.a_bufs.index(0), p.b_bufs.index(0)])
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs


@gluon.jit
def matmul_epilogue_partition(p):
    tile_m: gl.constexpr = p.a_desc.block_shape[0]
    tile_n: gl.constexpr = p.b_desc.block_shape[1]
    split_tile_n: gl.constexpr = p.c_desc.block_shape[1]
    subtile_factor: gl.constexpr = tile_n // split_tile_n
    subtile_stages: gl.constexpr = p.SUBTILE_STAGES
    acc_stages: gl.constexpr = p.acc_empty_bars.shape[0]
    dtype: gl.constexpr = p.c_desc.dtype
    acc_state = Counter.create(0, acc_stages)
    acc_smems = gl.allocate_shared_memory(dtype, [subtile_stages, tile_m, split_tile_n], p.c_desc.layout)
    sub_acc_state = Counter.create(0, subtile_stages)
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        # Load the FULL tile in its natural layout (like 08-WS), apply rowwise scale on the
        # full tile (layout derived from the valid full-tile load), then split in registers.
        accv = p.acc_bufs.index(acc_state.index).load()  # [tile_m, tile_n] f32
        rows = off_m + gl.arange(0, tile_m, layout=gl.SliceLayout(1, accv.type.layout))
        cols = off_n + gl.arange(0, tile_n, layout=gl.SliceLayout(0, accv.type.layout))
        sa = gl.load(p.sa_ptr + rows)
        sb = gl.load(p.sb_ptr + cols)
        accv = accv * sa[:, None] * sb[None, :]
        accs = _split_n(accv, subtile_factor)
        for s in gl.static_range(subtile_factor):
            acc_smem = acc_smems.index(sub_acc_state.index)
            tma.store_wait(pendings=subtile_stages - 1)
            acc_smem.store(accs[s].to(dtype))
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + split_tile_n * s], acc_smem)
            sub_acc_state = sub_acc_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def matmul_multicta_kernel(a_desc, b_desc, c_desc, sa_ptr, sb_ptr, M, N, K,
                           BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
                           GRID_MINOR_DIM: gl.constexpr, GRID_TILE_WIDTH: gl.constexpr, STAGES: gl.constexpr,
                           ACC_STAGES: gl.constexpr, CGA_LAYOUT: gl.constexpr, EPILOGUE_SIZE_N: gl.constexpr,
                           SUBTILE_STAGES: gl.constexpr):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_n: gl.constexpr = b_desc.block_shape[1]
    two_ctas: gl.constexpr = gl.num_ctas() > 1
    n_partitions: gl.constexpr = 4
    dtype: gl.constexpr = a_desc.dtype
    a_bufs = gl.allocate_shared_memory(dtype, [STAGES] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [STAGES] + b_desc.block_shape, b_desc.layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_SIZE_M, block_n // get_split_dim(CGA_LAYOUT, 1)], col_stride=1, cga_layout=CGA_LAYOUT, two_ctas=two_ctas)
    acc_bufs = allocate_tensor_memory(gl.float32, [ACC_STAGES, block_m, block_n], tmem_layout)
    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0)], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=two_ctas)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)
    acc_empty_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=two_ctas)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=mma_barrier_count)
    clc_barriers = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    clc_consumed_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=two_ctas)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(clc_barriers.index(i), count=1)
        mbarrier.init(clc_planar_ready_bars.index(i), count=1)
        mbarrier.init(clc_consumed_bars.index(i), count=n_partitions - 1)
    cga_layout: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout)
    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    p = MatmulPartitionArgs(
        a_desc, b_desc, c_desc, sa_ptr, sb_ptr, a_bufs, b_bufs, load_empty_bars, load_ready_bars,
        acc_bufs, acc_empty_bars, acc_ready_bars, clc_result_buffers, clc_barriers,
        clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars,
        GRID_MINOR_DIM, GRID_TILE_WIDTH, SUBTILE_STAGES)

    gl.warp_specialize([
        (matmul_epilogue_partition, (p, )),
        (matmul_load_partition, (p, )),
        (matmul_mma_partition, (p, )),
        (matmul_clc_partition, (p, )),
    ], [1, 1, 1], [24, 24, 24])


def scaled_mm_multicta(a, b, sa, sb, out=None, *, block_size_m=128, block_size_n=256, block_size_k=128,
                       grid_minor_dim=0, grid_tile_width=16, stages=6, acc_stages=2,
                       cga_layout=((1, 0), ), epilogue_size_n=32, subtile_stages=4, num_warps=4):
    M, K = a.shape
    K1, N = b.shape
    assert K == K1
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    tile_m = block_size_m * get_split_dim(cga_layout, 0)
    two_ctas = bool(cga_layout)
    a_layout = gl.NVMMASharedLayout.get_default_for([tile_m, block_size_k], gl.float8e4nv,
                                                    cga_layout=get_cga_layout(cga_layout, 0, two_ctas))
    b_layout = gl.NVMMASharedLayout.get_default_for([block_size_k, block_size_n], gl.float8e4nv,
                                                    cga_layout=get_cga_layout(cga_layout, 1, two_ctas))
    c_layout = gl.NVMMASharedLayout.get_default_for([tile_m, epilogue_size_n], gl.bfloat16, cga_layout=cga_layout)
    a_desc = TensorDescriptor.from_tensor(a, [tile_m, block_size_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_size_k, block_size_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(out, [tile_m, epilogue_size_n], c_layout)

    def grid(meta):
        tm = meta["BLOCK_SIZE_M"] * get_split_dim(meta["CGA_LAYOUT"], 0)
        tn = meta["BLOCK_SIZE_N"]
        return (triton.cdiv(M, tm) * triton.cdiv(N, tn), )

    matmul_multicta_kernel[grid](
        a_desc, b_desc, c_desc, sa, sb, M, N, K, block_size_m, block_size_n, block_size_k,
        grid_minor_dim, grid_tile_width, stages, acc_stages, cga_layout, epilogue_size_n, subtile_stages,
        num_warps=num_warps, num_ctas=2**len(cga_layout))
    return out


if __name__ == "__main__":
    print("triton", triton.__version__)
    for (M, K, N) in [(512, 2048, 4096), (512, 2048, 2048), (512, 6144, 2048), (512, 2048, 12288)]:
        a = (torch.randn((M, K), device="cuda") * 0.3).to(torch.float8_e4m3fn)
        b = (torch.randn((K, N), device="cuda") * 0.3).to(torch.float8_e4m3fn)
        sa = torch.rand((M,), device="cuda", dtype=torch.float32) * 0.5 + 0.5
        sb = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.5 + 0.5
        try:
            out = scaled_mm_multicta(a, b, sa, sb)
            torch.cuda.synchronize()
            ref = (a.float() @ b.float()) * sa[:, None] * sb[None, :]
            rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
            print(f"M={M} K={K} N={N}: rel={rel:.4f}", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"M={M} K={K} N={N} ERR {str(ex)[:120]}", flush=True)
