from __future__ import annotations

import ast
import hashlib
import importlib
import math
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest

from test.test_cute_quotient_aux import _args
from test.test_cute_quotient_aux import _quotient_cols

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target


def flatten(value):
    if isinstance(value, tuple):
        for x in value:
            yield from flatten(x)
    else:
        yield value


def static_values(tensor, cute):
    shape = tensor.layout.shape
    stride = tensor.layout.stride
    origin = tensor.iterator
    assert isinstance(tensor.layout, cute.Layout) and cute.is_static(tensor.layout)
    sizes = list(flatten(shape))
    weights = []
    assert isinstance(origin, tuple) and all(type(x) is int for x in origin)
    for x in flatten(stride):
        if isinstance(x, cute.ScaledBasis):
            assert type(x.value) is int and len(x.mode) == 1
            weights.append(
                tuple(x.value if i == x.mode[0] else 0 for i in range(len(origin)))
            )
        else:
            assert x == 0
            weights.append((0,) * len(origin))
    assert len(weights) == len(sizes) and all(type(x) is int and x > 0 for x in sizes)
    result = []
    for linear in range(math.prod(sizes)):
        remaining = linear
        coords = []
        for size in sizes:
            coords.append(remaining % size)
            remaining //= size
        value = tuple(
            origin[i] + sum(c * w[i] for c, w in zip(coords, weights, strict=True))
            for i in range(len(origin))
        )
        assert tensor[linear] == value
        result.append(value)
    return result, (shape, stride)


def assignment(node, scope):
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "<actual emitted layout>", "exec"
        ),
        scope,
    )


def certificate(source, axis, c):
    import cutlass

    # The SDK's extension module exports IR classes dynamically (no stubs).
    ir: Any = importlib.import_module("cutlass._mlir.ir")
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute
    from cutlass.utils.gemm import sm100

    parsed = ast.parse(source)
    native = next(
        n
        for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    nodes = {
        n.targets[0].id: n
        for n in ast.walk(native)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    }
    plans = next(
        ast.literal_eval(n.value)
        for n in parsed.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_helion_cute_wrapper_plans"
            for t in n.targets
        )
    )
    ab = next(p for p in plans if p["kind"] == "tcgen05_ab_tma")
    bm, bn = ab["bm"], ab["bn"]
    tile = (bm, bn)
    assert ab["cluster_m"] == ab["cluster_n"] == 1
    assert tile[axis] % c == 0 or c % tile[axis] == 0
    m, n = ab["m_size"], ab["n_size"]
    qshape = (m, (n - 1) // c + 1) if axis == 1 else ((m - 1) // c + 1, n)
    sy, sx = qshape[1], 1
    with (
        patch.object(cute, "compile", side_effect=AssertionError("native forbidden")),
        patch.object(
            cutlass, "compile", side_effect=AssertionError("native forbidden")
        ),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("quotient_scalar_fragment_certificate", ([], []))
        with ir.InsertionPoint(fn.add_entry_block()):
            scope: dict[str, Any] = {"cutlass": cutlass, "cute": cute}
            for name in [
                "tiled_mma",
                "tcgen05_epilogue_rest_mode",
                "acc_frag_base",
                "tcgen05_kernel_desc",
                "tcgen05_store_epi_tile",
            ]:
                assignment(nodes[name], scope)
            thr = scope["tiled_mma"].get_slice(0)
            # The original per-thread C/T2R/R2S logical output coordinates.
            logical = cute.make_identity_tensor((bm, bn))
            import sympy

            from helion._compiler.cute.quotient_aux import prove_quotient_aux_load_map

            y, x = sympy.symbols("y x", integer=True, nonnegative=True)
            indices = [y, x]
            indices[axis] = sympy.floor(indices[axis] / c)
            mapping = prove_quotient_aux_load_map(
                indices,
                (y, x),
                (m, n),
                qshape,
                (sy, sx),
                storage_offset=0,
                storage_elements=qshape[0] * qshape[1],
            )
            assert mapping is not None
            scope["aux"] = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32,
                    0x1000000,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout(qshape, stride=(sy, sx)),
            )
            emitted, _ = mapping.view_source("actual_qview", "aux", "0", "0", bm, bn)
            exec(compile(emitted, "<actual quotient map scalar view>", "exec"), scope)
            view = scope["actual_qview"]

            def basis(value):
                if isinstance(value, tuple):
                    return tuple(basis(x) for x in value)
                return 0 if value == 0 else cute.ScaledBasis(value, 0)

            projected = cute.make_tensor(
                (0,),
                cute.make_layout(view.layout.shape, stride=basis(view.layout.stride)),
            )

            def pipeline(tensor):
                base = thr.partition_C(tensor)
                xfm = sm100.transform_partitioned_tensor_layout(base)
                rest = scope["tcgen05_epilogue_rest_mode"]
                planned = cute.make_tensor(
                    xfm.iterator,
                    cute.append(cute.append(cute.append(xfm.layout, rest), rest), rest),
                )
                return (
                    base,
                    planned,
                    cute.flat_divide(planned, scope["tcgen05_store_epi_tile"]),
                )

            lbase, lplanned, lepi = pipeline(logical)
            qbase, qplanned, qepi = pipeline(projected)
            acc = scope["acc_frag_base"]
            scope["tcgen05_epi_acc_frag_base"] = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                acc.layout,
            )
            assignment(nodes["tcgen05_tAcc"], scope)
            copy, ts, tr = sm100.epilogue_tmem_copy_and_partition(
                scope["tcgen05_kernel_desc"],
                cast("cutlass.Int32", 0),
                scope["tcgen05_tAcc"],
                lplanned,
                scope["tcgen05_store_epi_tile"],
                False,
            )
            r2s = None
            if "tcgen05_sD_layout" in nodes:
                assignment(nodes["tcgen05_sD_layout"], scope)
                scope["tcgen05_sD_ptr"] = cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.smem, assumed_align=1024
                )
                assignment(nodes["tcgen05_sD"], scope)
                rd = cute.make_rmem_tensor(tr.shape, cutlass.Float32)
                r2s, rsd, rssd = sm100.epilogue_smem_copy_and_partition(
                    scope["tcgen05_kernel_desc"],
                    copy,
                    rd,
                    cast("cutlass.Int32", 0),
                    scope["tcgen05_sD"],
                )
            count = 0
            logical_union = []
            for thread in range(128):
                ld = copy.get_slice(thread).partition_D(lepi)
                qd = copy.get_slice(thread).partition_D(qepi)
                paths = [(ld, qd)]
                if r2s is not None:
                    paths.append((r2s.retile(ld), r2s.retile(qd)))
                for lv, qv in paths:
                    lc, ls = static_values(lv, cute)
                    qc, qs = static_values(qv, cute)
                    assert len(lc) == len(qc)
                    # Compare actual single-linear-index semantics, not Python
                    # row-major tuple iteration across different factor shapes.

                    def numeric(value):
                        if isinstance(value, tuple):
                            return tuple(numeric(x) for x in value)
                        return (
                            value.value
                            if isinstance(value, cute.ScaledBasis)
                            else value
                        )

                    if thread == 0:
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                0x1000000 + qv.iterator[0] * 4,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(
                                qv.layout.shape, stride=numeric(qv.layout.stride)
                            ),
                        )
                        dest = cute.make_rmem_tensor(
                            cute.make_layout(lv.shape), cutlass.Float32
                        )
                        atom = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            cutlass.Float32,
                            num_bits_per_copy=32,
                        )
                        cute.copy(atom, src, dest)
                    for coord, offset in zip(lc, qc, strict=True):
                        y, x = coord
                        assert 0 <= y < bm and 0 <= x < bn
                        expected = (
                            (y * sy + (x // c) * sx)
                            if axis == 1
                            else ((y // c) * sy + x * sx)
                        )
                        assert offset == (expected,)
                        count += 1
                    if lv is ld:
                        logical_union.extend(lc)
            assert len(logical_union) == bm * bn and len(set(logical_union)) == bm * bn
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
            )
            assert atom is not None
            text = str(module)
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bm": bm,
        "bn": bn,
        "axis": axis,
        "divisor": c,
        "actual_T2R_and_used_R2S_slot_offsets_checked": count,
        "R2S_used_in_actual_source": r2s is not None,
        "all_output_coordinates_owned_once": True,
        "scalar_copy_atom_bits": 32,
        "no_native_compile_or_CUDA": True,
    }, text


@pytest.mark.parametrize("blocks", ([64, 32, 64], [128, 64, 64]))
@pytest.mark.parametrize("axis,c", [(1, 16), (0, 16), (1, 2), (1, 32)])
def test_actual_SDK_scalar_read_slots_match_C_T2R_R2S(blocks, axis, c):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        source = bound.to_code(helion.Config(block_sizes=blocks, pid_type="flat"))
        record, irtext = certificate(source, axis, c)
        assert record["all_output_coordinates_owned_once"]
        assert record["R2S_used_in_actual_source"]
        assert record["scalar_copy_atom_bits"] == 32
