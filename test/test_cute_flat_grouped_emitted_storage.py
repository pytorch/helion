from __future__ import annotations

import ast
from itertools import pairwise

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute

from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_tma_rn import schedule
from test.test_cute_tma_rn_two_stage import low_schedule

from helion._compiler.cute.tcgen05_flat_grouped_kernel import (
    render_flat_grouped_rna_kernel,
)
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_storage


def check_emitted_storage(source, selected):
    """Run the emitted SDK layouts/assertions, then bound every emitted view."""
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion_native_grouped_rna"
    )
    assignments = {
        node.targets[0].id: node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    pool_call = assignments["tcgen05_storage_pool"].value
    assert ast.unparse(pool_call.func) == "cute.arch.alloc_smem"
    pool_bytes = ast.literal_eval(pool_call.args[1])
    assert pool_bytes == selected.storage.bytes
    placements = {p.allocation.name: p for p in selected.storage.placements}
    pointers = {}
    for node in ast.walk(kernel):
        if not isinstance(node, ast.BinOp) or not (
            isinstance(node.left, ast.Name) and node.left.id == "tcgen05_storage_pool"
        ):
            continue
        assert isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant)
        owners = [
            name
            for name, assignment in assignments.items()
            if isinstance(assignment.value, ast.Call)
            and assignment.value.args
            and assignment.value.args[0] is node
        ]
        assert len(owners) == 1, "every pool view must be a direct typed assignment"
        name = owners[0]
        assert name not in pointers
        pointers[name] = node.right.value
        assert ast.unparse(assignments[name].value.func) == "cute.recast_ptr"
        placement = placements[name]
        assert pointers[name] == placement.offset, name
        assert placement.offset % placement.allocation.alignment == 0
        assert (
            0
            <= pointers[name]
            < pointers[name] + placement.allocation.bytes
            <= pool_bytes
        )
    omitted = (
        {"tcgen05_grouped_tensormap_smem_ptr", "tcgen05_grouped_d_tensormap_smem_ptr"}
        if selected.descriptors is not None
        else set()
    )
    assert set(pointers) == set(placements) - omitted
    spans = sorted(
        (placement.offset, placement.offset + placement.allocation.bytes)
        for placement in placements.values()
    )
    assert all(left[1] <= right[0] for left, right in pairwise(spans))

    names = {
        "tiled_mma",
        "sA_layout",
        "sB_layout",
        "sA",
        "sB",
        "tcgen05_store_epi_tile",
        "tcgen05_sD_layout",
        "tcgen05_sD",
        "tcgen05_work_tile_smem_tensor",
        "tcgen05_prefix",
        *pointers,
    }
    body = [
        node
        for node in kernel.body
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        )
        or isinstance(node, ast.Assert)
    ]
    assert {
        node.targets[0].id for node in body if isinstance(node, ast.Assign)
    } == names
    assert sum(isinstance(node, ast.Assert) for node in body) == 4
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            namespace = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05_storage_pool": cute.make_ptr(
                    cutlass.Uint8, 0, cute.AddressSpace.smem, assumed_align=1024
                ),
            }
            # Execute the real generated assignments and all four assertions;
            # no JIT wrapper defers them, and no native compiler is called.
            exec(
                compile(
                    ast.Module(body=body, type_ignores=[]), "emitted-layouts", "exec"
                ),
                namespace,
            )
            sizes = {
                "smem_a": cute.cosize(namespace["sA"].layout) * 4,
                "smem_b": cute.cosize(namespace["sB"].layout) * 4,
                "tcgen05_sD_ptr": cute.cosize(namespace["tcgen05_sD"].layout) * 4,
                "tcgen05_work_tile_smem_ptr": cute.cosize(
                    namespace["tcgen05_work_tile_smem_tensor"].layout
                )
                * 4,
                "tcgen05_prefix_ptr": cute.cosize(namespace["tcgen05_prefix"].layout)
                * 4,
                "tcgen05_tmem_holding_buf": 4,
                "tcgen05_tmem_dealloc_mbar_ptr": 8,
            }
            for operand in ("A", "B"):
                layout = namespace[f"s{operand}_layout"]
                stage_words = cute.cosize(
                    cute.slice_(layout, (None, None, None, 0)).outer
                )
                stage_stride = namespace[f"s{operand}"].layout.stride[3]
                assert stage_words == stage_stride == 128 * selected.block_k
                assert (
                    stage_words * selected.ab_stages * 4
                    == sizes[f"smem_{operand.lower()}"]
                )
                # Every ring index, including the last stage, ends in the view.
                assert [
                    (stage * stage_stride * 4, (stage * stage_stride + stage_words) * 4)
                    for stage in range(selected.ab_stages)
                ] == [
                    (stage * stage_words * 4, (stage + 1) * stage_words * 4)
                    for stage in range(selected.ab_stages)
                ]
    for pipeline, storage in (
        ("tcgen05_acc_pipeline", "tcgen05_acc_pipeline_barriers"),
        ("tcgen05_sched_pipeline", "tcgen05_sched_pipeline_mbars"),
        ("tcgen05_ab_pipeline", "tcgen05_ab_pipeline_mbars"),
    ):
        stages = next(
            ast.literal_eval(keyword.value)
            for keyword in assignments[pipeline].value.keywords
            if keyword.arg == "num_stages"
        )
        sizes[storage] = stages * 2 * 8
    if not selected.tma_rn:
        converted = assignments["tcgen05_converted_pipeline"].value
        assert ast.unparse(converted.func) == "make_converted_operand_pipeline"
        sizes["tcgen05_converted_full_ptr"] = ast.literal_eval(converted.args[1]) * 8
    if not omitted:
        assert (
            ast.unparse(assignments["tcgen05_grouped_tensormap_a_smem_ptr"].value)
            == "tcgen05_grouped_tensormap_smem_ptr"
        )
        assert (
            ast.unparse(assignments["tcgen05_grouped_tensormap_b_smem_ptr"].value)
            == "tcgen05_grouped_tensormap_a_smem_ptr + 16"
        )
        # Two 128-byte tensor maps: the second is +16 Int64 words.
        sizes["tcgen05_grouped_tensormap_smem_ptr"] = 2 * 16 * 8
        sizes["tcgen05_grouped_d_tensormap_smem_ptr"] = 16 * 8
    assert set(sizes) == set(pointers)
    for name, size in sizes.items():
        assert size == placements[name].allocation.bytes, name
    return {name: [pointers[name], sizes[name]] for name in sorted(pointers)}


@pytest.mark.parametrize("groups", [1, 5, 256, 1024])
def test_actual_emitted_layout_assertions_and_every_pointer_span(groups):
    with cuda_binding(
        static=True, original=True, groups=groups, rows=515, k=128, n=128
    ) as bound:
        choices = [
            schedule(bound, bk=bk, wrapped=wrapped, rn=rn)
            for bk in (32, 64)
            for wrapped in (False, True)
            for rn in (False, True)
            if groups <= 256
        ] + [
            low_schedule(bound, ctas, descriptor_policy=policy)
            for ctas in (1, 2)
            for policy in ("dynamic", "wrapped")
        ]
        if groups > 256:
            # The legacy K-total=192 ring no longer fits with this prefix.
            assert (
                flat_grouped_storage(groups, 32, 6, tma_rn=True).bytes + 1024 > 232448
            )
        for selected in choices:
            assert selected is not None
            check_emitted_storage(render_flat_grouped_rna_kernel(selected), selected)


def test_actual_emitted_storage_check_rejects_six_stage_regressions():
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        selected = low_schedule(bound, descriptor_policy="dynamic")
        source = render_flat_grouped_rna_kernel(selected)
        spans = check_emitted_storage(source, selected)
        bad_sizes = source.replace(
            "cute.cosize(sA_layout.outer) == 8192",
            "cute.cosize(sA_layout.outer) == 24576",
        )
        assert bad_sizes != source
        with pytest.raises(AssertionError):
            check_emitted_storage(bad_sizes, selected)
        # Includes within-pool wrong placements as well as out-of-pool offsets.
        for name, wrong in (
            ("smem_a", 0),
            ("smem_b", 131072),
            ("tcgen05_work_tile_smem_ptr", 229760),
            ("tcgen05_grouped_tensormap_smem_ptr", 229376),
            ("tcgen05_grouped_d_tensormap_smem_ptr", 229632),
            ("tcgen05_sD_ptr", 32768),
            ("tcgen05_prefix_ptr", 229808),
        ):
            correct = (
                f"{name} = cute.recast_ptr(tcgen05_storage_pool + {spans[name][0]},"
            )
            bad = source.replace(
                correct, f"{name} = cute.recast_ptr(tcgen05_storage_pool + {wrong},"
            )
            assert bad != source
            with pytest.raises(AssertionError):
                check_emitted_storage(bad, selected)
