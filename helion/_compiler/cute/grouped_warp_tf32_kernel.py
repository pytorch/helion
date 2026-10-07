"""Sixteen-warp TF32 grouped contraction with ldmatrix operand transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grouped_warp_tf32_plan import GroupedWarpTF32Plan


def grouped_warp_tf32_module(plan: GroupedWarpTF32Plan) -> str:
    multiple_k = plan.reduction > 64
    multiple_n = plan.columns != 64
    parameters = (", k_base" if multiple_k else "") + (", n_base" if multiple_n else "")
    first_args = (", cutlass.Int32(0)" if multiple_k else "") + (
        ", n_base" if multiple_n else ""
    )
    global_n = "n + n_base" if multiple_n else "n"
    column = "column + n_base" if multiple_n else "column"
    output_column = "4 * (t % 16) + n_base" if multiple_n else "4 * (t % 16)"
    b_valid = "cutlass.Int64(offset) < length"
    store_valid = "cutlass.Int64(row) < length"
    if plan.columns % 64:
        b_valid = f"({b_valid}) & ({global_n} < {plan.columns})"
        store_valid = f"({store_valid}) & ({output_column} < {plan.columns})"
    bias_load = f"""            source = bias.iterator + cutlass.Int64(group) * bias.layout.stride[0] + cutlass.Int64({column}) * bias.layout.stride[1]
            bias_tensor = cute.make_tensor(source.align(8), cute.make_layout((2,)))
            cute.copy(bias_copy, bias_tensor, bias_values[None, half_n])"""
    if plan.columns % 64:
        bias_load = bias_load.replace("source", "bias_source")
        bias_load = (
            f"            if {column} < {plan.columns}:\n"
            + "\n".join("    " + line for line in bias_load.splitlines())
            + "\n            else:\n                bias_values[None, half_n].fill(0.0)"
        )
    refills = ""
    if multiple_k:
        args = ", k_base" + (", n_base" if multiple_n else "")
        refills = f"""        for k_block in cutlass.range(1, {plan.reduction // 64}, unroll=1):
            # All previous ldmatrix reads retire before any warp refills shared A/B.
            cute.arch.sync_threads()
            k_base = cutlass.Int32(k_block * 64)
            _grouped_warp_copy_ab(a, b, a_packed, b_grouped, start, length, group, offset, t{args})
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.copy(ca, pa, da)
            cute.copy(cb, pb, db)
            for ki in cutlass.range_constexpr(8):
                cute.gemm(mma, acc, ra[None, None, ki], rb[None, None, ki], acc)
"""
    return _TEMPLATE.format(
        copy_parameters=parameters,
        global_k="k + k_base" if multiple_k else "k",
        global_n=global_n,
        b_valid=b_valid,
        n_setup="    n_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64\n"
        if multiple_n
        else "",
        first_copy_args=first_args,
        refills=refills,
        bias_load=bias_load,
        store_valid=store_valid,
        output_column=output_column,
    )


_TEMPLATE = """from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def _grouped_warp_mma():
    return cute.make_tiled_mma(
        cute.nvgpu.warp.MmaTF32Op((16, 8, 8)),
        atom_layout_mnk=cute.make_layout((4, 4, 1), stride=(4, 1, 0)),
        permutation_mnk=(64, 32, 16),
    )


@cute.jit
def _grouped_warp_ab_layout(rows):
    return cute.make_composed_layout(
        cute.make_swizzle(3, 2, 4),
        0,
        cute.make_layout((rows, 64), stride=(64, 1)),
    )


@cute.jit
def _grouped_warp_shuffle_store_offset(t):
    q = ((t & 24) << 4) | ((t << 3) & 3088)
    u = (-(t & 1) & 16448) ^ (t & 96)
    return (q | u) ^ (-((t >> 2) & 1) & 8224)


@cute.jit
def _grouped_warp_shuffle_load_offset(t):
    q = ((t & 7) << 4) | ((t << 2) & 1920)
    return (q ^ (-((t >> 4) & 1) & 8224)) | ((t << 9) & 4096)


@cute.jit
def _grouped_warp_copy_ab(a, b, a_packed, b_grouped, start, length, group, offset, t{copy_parameters}):
    for packet in cutlass.range_constexpr(4):
        row = offset + t // 16 + packet * 32
        k = (t % 16) * 4
        source = a_packed.iterator + (start + cutlass.Int64(row)) * a_packed.layout.stride[0] + cutlass.Int64({global_k}) * a_packed.layout.stride[1]
        valid_bytes = cutlass.Int32(16) if cutlass.Int64(row) < length else cutlass.Int32(0)
        cute.arch.cp_async_shared_global(
            a.iterator + cute.crd2idx((t // 16 + packet * 32, k), a.layout),
            source, 16, "cg", cp_size=valid_bytes,
        )
    cute.arch.cp_async_commit_group()
    for packet in cutlass.range_constexpr(8):
        # Retain the original B packet issue order: 0, 32, 8, 40, 16, 48, 24, 56.
        k = t // 64 + (packet // 2) * 8 + (packet % 2) * 32
        n = t % 64
        source = b_grouped.iterator + cutlass.Int64(group) * b_grouped.layout.stride[0] + cutlass.Int64({global_n}) * b_grouped.layout.stride[1] + cutlass.Int64({global_k}) * b_grouped.layout.stride[2]
        valid_bytes = cutlass.Int32(4) if {b_valid} else cutlass.Int32(0)
        cute.arch.cp_async_shared_global(
            b.iterator + cute.crd2idx((n, k), b.layout),
            source, 4, "ca", cp_size=valid_bytes,
        )
    cute.arch.cp_async_commit_group()


@cute.kernel
def _helion_native_grouped_rna(offsets, a_packed, b_grouped, bias, out, tcgen05_grouped_ab_tensormaps):
    # Retain the grouped host ABI; this implementation does not use its workspace.
    storage = cute.arch.alloc_smem(cutlass.Uint8, 81920, alignment=16)
    ap = cute.recast_ptr(storage, dtype=cutlass.TFloat32)
    bp = cute.recast_ptr(storage + 65536, dtype=cutlass.TFloat32)
    cp = cute.recast_ptr(storage + 32768, dtype=cutlass.Float32)
    a = cute.make_tensor(ap, _grouped_warp_ab_layout(128))
    b = cute.make_tensor(bp, _grouped_warp_ab_layout(64))
    t = cutlass.Int32(cute.arch.thread_idx()[0])
    group = cutlass.Int32(cute.arch.block_idx()[0])
{n_setup}    start = cutlass.Int64(offsets[group])
    end = cutlass.Int64(offsets[group + 1])
    length = end - start
    mma = _grouped_warp_mma()
    thread = mma.get_slice(t)
    ra = thread.make_fragment_A(thread.partition_A(a))
    rb = thread.make_fragment_B(thread.partition_B(b))
    coords = thread.partition_C(cute.make_identity_tensor((128, 64)))
    acc = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
    values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
    ca = cute.make_tiled_copy_A(
        cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.TFloat32), mma
    )
    cb = cute.make_tiled_copy_B(
        cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.TFloat32), mma
    )
    pa = ca.get_slice(t).partition_S(a)
    pb = cb.get_slice(t).partition_S(b)
    da = ca.get_slice(t).retile(ra)
    db = cb.get_slice(t).retile(rb)
    vector_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
    bias_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64)
    bias_values = cute.make_rmem_tensor((2, 2), cutlass.Float32)
    packed = cute.make_rmem_tensor((4,), cutlass.Float32)
    loads = cute.make_rmem_tensor((4, 4), cutlass.Float32)
    store_offset = _grouped_warp_shuffle_store_offset(t) // 4
    load_offset = _grouped_warp_shuffle_load_offset(t)
    offset = cutlass.Int32(0)
    _grouped_warp_copy_ab(a, b, a_packed, b_grouped, start, length, group, offset, t{first_copy_args})
    while cutlass.Int64(offset) < length:
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        cute.copy(ca, pa, da)
        cute.copy(cb, pb, db)
        acc.fill(0.0)
        for ki in cutlass.range_constexpr(8):
            # All four fragment chains execute one K8 wave before their next wave.
            cute.gemm(mma, acc, ra[None, None, ki], rb[None, None, ki], acc)
{refills}        for half_n in cutlass.range_constexpr(2):
            column = cutlass.Int32(coords[half_n * 8][1])
{bias_load}
        for element in cutlass.range_constexpr(16):
            bias_value = bias_values[element % 2, element // 8]
            dot_plus_zero = acc[element] + cutlass.Float32(0.0)
            values[element] = dot_plus_zero + bias_value
        for packet in cutlass.range_constexpr(4):
            half_n = packet // 2
            half_row = packet % 2
            for item in cutlass.range_constexpr(4):
                index = half_n * 8 + half_row * 2 + (item // 2) * 4 + item % 2
                packed[item] = values[index]
            target = cute.make_tensor((cp + store_offset + half_n * 1024 + half_row * 128).align(16), cute.make_layout((4,)))
            cute.copy(vector_copy, packed, target)
        cute.arch.sync_threads()
        for packet in cutlass.range_constexpr(4):
            word_offset = (load_offset ^ ((packet // 2) * 16448)) // 4 + (packet % 2) * 512
            source = cute.make_tensor((cp + word_offset).align(16), cute.make_layout((4,)))
            cute.copy(vector_copy, source, loads[None, packet])
        for packet in cutlass.range_constexpr(4):
            row = offset + t // 16 + packet * 32
            if {store_valid}:
                for item in cutlass.range_constexpr(4):
                    packed[item] = loads[item % 2 + (packet // 2) * 2, (packet % 2) + (item // 2) * 2]
                destination = out.iterator + (start + cutlass.Int64(row)) * out.layout.stride[0] + cutlass.Int64({output_column}) * out.layout.stride[1]
                output_target = cute.make_tensor(destination.align(16), cute.make_layout((4,)))
                cute.copy(vector_copy, packed, output_target)
        offset = offset + cutlass.Int32(128)
        _grouped_warp_copy_ab(a, b, a_packed, b_grouped, start, length, group, offset, t{first_copy_args})
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()


_helion_native_grouped_rna._helion_cute_wrapper_plans = []
_helion_native_grouped_rna._helion_cute_disable_bake_tensor_shapes = True
"""
