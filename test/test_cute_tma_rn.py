from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_flat_grouped_public import _native_source
from test.test_cute_flat_grouped_public import _source

import helion
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_kernel_module,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_wrapper_plans,
)
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import (
    normalize_grouped_rna_config,
)
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._compiler.cute.tcgen05_operand_transform import Tcgen05OperandRolePlan
from helion._compiler.cute.tcgen05_operand_transform import prove_operand_stage_span
from helion._compiler.cute.tcgen05_tma_rn import AUTO
from helion._compiler.cute.tcgen05_tma_rn import CONVERSION_KEY
from helion._compiler.cute.tcgen05_tma_rn import TMA_RN
from helion._compiler.cute.tcgen05_tma_rn import WARP_RAW
from helion._compiler.cute.tcgen05_tma_rn import Tcgen05TmaRnRoles
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.cute.launcher import _append_cute_wrapper_plan


def rn_config(bound, bk=32, wrapped=True):
    values = bound.config_spec._base_default_config().config | {
        CONVERSION_KEY: TMA_RN,
        K_KEY: bk,
    }
    if wrapped:
        values[DESCRIPTOR_KEY] = WRAPPED
    return helion.Config.from_dict(values)


def schedule(bound, bk=32, wrapped=True, rn=True):
    proof = prove_flat_grouped_rna(bound.env, _host(bound).device_ir)
    assert proof is not None
    result = flat_grouped_rna_schedule(
        bound.env,
        proof,
        converter_warps=0 if rn else 8,
        block_k=bk,
        ab_stages={32: 6, 64: 3}[bk],
        shared_capacity=232448,
        descriptor_policy=WRAPPED if wrapped else "dynamic",
        tma_rn=rn,
    )
    assert result is not None
    return result


@pytest.mark.parametrize("bk", [32, 64])
@pytest.mark.parametrize("wrapped", [False, True])
def test_public_binding_uses_direct_TMA_ring_and_preserves_other_role_bodies(
    bk, wrapped
):
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        rn = schedule(bound, bk, wrapped)
        old = schedule(bound, bk, wrapped, rn=False)
        native = _native_source(_source(bound, rn_config(bound, bk, wrapped)))
        assert native == flat_grouped_kernel_module(rn)
        assert rn.thread_block == (32, 8, 1)
        assert type(rn.roles) is Tcgen05TmaRnRoles
        assert rn.roles.scheduler_consumer_warp_ids == (0, 1, 2, 3, 4, 5)
        assert rn.roles.scheduler_empty_warp_arrivals == 6
        assert {
            *rn.roles.scheduler_consumer_warp_ids,
            rn.roles.scheduler_warp,
            *rn.roles.padding_warp_ids,
        } == set(range(8))
        assert rn.roles.converter_warp_ids == ()
        with pytest.raises(ValueError):
            replace(old, tma_rn=True)
        with pytest.raises(ValueError):
            replace(rn, tma_rn=False)
        allocations = {p.allocation.name: p for p in rn.storage.placements}
        assert "tcgen05_converted_full_ptr" not in allocations
        for operand in ("smem_a", "smem_b"):
            assert allocations[operand].allocation.bytes == 128 * bk * rn.ab_stages * 4
        spans = sorted(
            (p.offset, p.offset + p.allocation.bytes) for p in rn.storage.placements
        )
        assert all(a[1] <= b[0] for a, b in pairwise(spans))

        old_tree, rn_tree = map(ast.parse, (flat_grouped_kernel_module(old), native))
        old_kernel, kernel = [
            next(
                n
                for n in t.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_helion_native_grouped_rna"
            )
            for t in (old_tree, rn_tree)
        ]
        names = {n.id for n in ast.walk(kernel) if isinstance(n, ast.Name)}
        assert not any("convert" in name for name in names)
        guards = {ast.unparse(n.test): n for n in kernel.body if isinstance(n, ast.If)}
        old_guards = {
            ast.unparse(n.test): n for n in old_kernel.body if isinstance(n, ast.If)
        }
        # TMA, epilogue and the common TMEM publication/wait remain complete,
        # identical subtrees. Only the operand consumer and converter differ.
        for guard in (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)",
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)",
            "tcgen05_exec_active or tcgen05_epi_active",
        ):
            assert ast.dump(guards[guard]) == ast.dump(old_guards[guard])
        mma = guards[
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4)"
        ]
        calls = [ast.unparse(n.func) for n in ast.walk(mma) if isinstance(n, ast.Call)]
        for method in ("consumer_try_wait", "consumer_wait", "consumer_release"):
            assert calls.count("tcgen05_ab_pipeline." + method) == 1
        k_loop = next(
            n
            for n in ast.walk(mma)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "tile_offset_2"
        )
        indices = {}
        for i, statement in enumerate(k_loop.body):
            for n in ast.walk(statement):
                if isinstance(n, ast.Call):
                    target = ast.unparse(n.func)
                    if target in (
                        "tcgen05_ab_pipeline.consumer_wait",
                        "cute.gemm",
                        "tcgen05_ab_pipeline.consumer_release",
                    ):
                        indices[target] = i
        assert (
            indices["tcgen05_ab_pipeline.consumer_wait"]
            < indices["cute.gemm"]
            < indices["tcgen05_ab_pipeline.consumer_release"]
        )
        plans = flat_grouped_wrapper_plans(rn)
        body, args = [], []
        for plan in plans:
            _append_cute_wrapper_plan(body, args, plan, num_sm=148)
        assert "\n".join(body).count("internal_type=cutlass.TFloat32") == 2
        assert plans[0]["operand_transform"] == "tf32_tma_rn"
        assert plans[0]["converter_warps"] == 0


@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
def test_non_tf32_precision_never_admits_RN(precision):
    with cuda_binding(
        static=True, original=True, k=128, n=128, dot_precision=precision
    ) as bound:
        assert not bound.config_spec.cute_grouped_rna_k_choices
        with pytest.raises(helion.exc.InvalidConfig, match="direct FP32 TF32"):
            bound.config_spec.normalized_config(rn_config(bound))
        invalid = rn_config(bound)
        bound.config_spec.normalize(invalid, _fix_invalid=True)
        assert invalid == bound.config_spec._base_default_config()


@pytest.mark.parametrize(
    "case",
    [
        "dynamic",
        "misalign_a",
        "misalign_b",
        "offset_stride",
        "wrong_stride",
        "extra_math",
        "fp16",
    ],
)
def test_unsupported_typed_domains_keep_the_ordinary_path(case):
    with cuda_binding(
        static=case != "dynamic",
        k=128,
        n=128,
        mode=case if case in ("wrong_stride", "extra_math") else "ordinary",
        misalign={"misalign_a": 1, "misalign_b": 2}.get(case),
        offsets_stride=2 if case == "offset_stride" else 1,
        dtype=torch.float16 if case == "fp16" else torch.float32,
    ) as bound:
        assert not bound.config_spec.cute_grouped_rna_k_choices
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalized_config(rn_config(bound))


@pytest.mark.parametrize("operand", [1, 2])
def test_explicit_cast_between_load_and_dot_is_not_a_TMA_conversion(operand):
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        ir = _host(bound).device_ir
        assert prove_flat_grouped_rna(bound.env, ir) is not None
        mm = next(
            n
            for info in ir.graphs
            for n in info.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.baddbmm.default
        )
        original_args = mm.args
        with mm.graph.inserting_before(mm):
            cast = mm.graph.call_function(
                torch.ops.aten._to_copy.default,
                (mm.args[operand],),
                {"dtype": torch.float32},
            )
        cast.meta.update(mm.args[operand].meta)
        try:
            changed = list(mm.args)
            changed[operand] = cast
            mm.args = tuple(changed)
            assert prove_flat_grouped_rna(bound.env, ir) is None
        finally:
            mm.args = original_args
            mm.graph.erase_node(cast)
        assert prove_flat_grouped_rna(bound.env, ir) is not None


def test_policy_normalization_search_roundtrip_and_original_seed_prefix():
    with pytest.raises(ValueError):
        Tcgen05OperandRolePlan(0)
    for invalid in (True, 0, "rna", "tf32x3"):
        with pytest.raises(helion.exc.InvalidConfig):
            normalize_grouped_rna_config(
                {CONVERSION_KEY: invalid}, available_k=(32, 64), fix_invalid=False
            )
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        spec = bound.config_spec
        old_config = helion.Config.from_dict(
            spec._base_default_config().config
            | {WARPS_KEY: 8, K_KEY: 32, DESCRIPTOR_KEY: WRAPPED}
        )
        auto_config = helion.Config.from_dict(
            old_config.config | {CONVERSION_KEY: AUTO}
        )
        assert spec.normalized_config(auto_config) == spec.normalized_config(old_config)
        assert _source(bound, auto_config) == _source(bound, old_config)
        seeds = CuteGroupedRnaHeuristic.get_seed_configs(
            bound.env, _host(bound).device_ir
        )
        assert len(seeds) == 25
        assert seeds[-1].get(CONVERSION_KEY) == WARP_RAW
        assert all(CONVERSION_KEY not in s.config for s in seeds[:16])
        assert [
            (s[WARPS_KEY], s[K_KEY], s.get(DESCRIPTOR_KEY)) for s in seeds[:16]
        ] == [
            (warps, k, descriptor)
            for descriptor in (None, WRAPPED)
            for k in (32, 64)
            for warps in (4, 8, 2, 1)
        ]
        generator = ConfigGeneration(spec)
        for seed in seeds[16:24]:
            normalized = spec.normalized_config(seed)
            assert (
                CONVERSION_KEY in normalized.config
                and WARPS_KEY not in normalized.config
            )
            mixed = helion.Config.from_dict(seed.config | {WARPS_KEY: 8})
            assert spec.normalized_config(mixed) == normalized
            restored = generator.unflatten(generator.flatten(normalized))
            assert (
                restored[CONVERSION_KEY] == TMA_RN
                and restored[K_KEY] == normalized[K_KEY]
            )
            assert restored.get(DESCRIPTOR_KEY) == normalized.get(DESCRIPTOR_KEY)
        snapshots = [deepcopy(s.config) for s in seeds]
        seeds[-1].config["block_sizes"][0] += 1
        assert [i for i, s in enumerate(seeds) if s.config != snapshots[i]] == [24]


@pytest.mark.parametrize(
    "field,value",
    [
        ("operand_transform", "tf32_rna"),
        ("converter_warps", 8),
        ("converter_warps", False),
        ("scheduler_self_consumer", True),
        ("cluster_n", 2),
        ("input_dtype", "cutlass.Float32"),
    ],
)
def test_wrapper_rejects_policy_geometry_mismatch(field, value):
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        plan = flat_grouped_wrapper_plans(schedule(bound))[0]
        plan[field] = value
        with pytest.raises(helion.exc.BackendUnsupported):
            _append_cute_wrapper_plan([], [], plan, num_sm=148)


@pytest.mark.parametrize("bk,stages", [(32, 6), (64, 3)])
@pytest.mark.parametrize("operand", ["A", "B"])
def test_actual_SDK_TMA_RN_format_and_all_stage_spans(bk, stages, operand):
    import cutlass
    from cutlass._mlir import ir
    import cutlass.cute as cute
    from cutlass.utils import blackwell_helpers as sm100

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = sm100.make_trivial_tiled_mma(
                cutlass.TFloat32,
                cutlass.TFloat32,
                cute.nvgpu.OperandMajorMode.MN,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                (128, 128),
            )
            layout_fn = (
                sm100.make_smem_layout_a if operand == "A" else sm100.make_smem_layout_b
            )
            layout = layout_fn(
                mma, (128, 128, bk), cutlass.Float32, stages, is_k_major=operand == "B"
            )
            span = prove_operand_stage_span(layout, stages=stages, alignment_bytes=1024)
            assert span is not None and span.words == span.stride_words == 128 * bk
            assert span.allocation_bytes == cute.cosize(layout.outer) * 4 == 98304
            shape, stride = (
                ((128, 128, 3), (1, 128, 16384))
                if operand == "A"
                else ((273, 128), (128, 1))
            )
            tensor = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout(shape, stride=stride),
            )
            atom_fn = (
                cute.nvgpu.make_tiled_tma_atom_A
                if operand == "A"
                else cute.nvgpu.make_tiled_tma_atom_B
            )
            common = (
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                    cute.nvgpu.tcgen05.CtaGroup.ONE
                ),
                tensor,
                cute.slice_(layout, (None, None, None, 0)),
                (128, 128, bk),
                mma,
                (1, 1, 1, 1),
            )
            raw, raw_coords = atom_fn(*common, internal_type=cutlass.Float32)
            rn, rn_coords = atom_fn(*common, internal_type=cutlass.TFloat32)
            assert "tma_format = TF32_RN" in str(rn.type)
            assert str(raw.type) == str(rn.type).replace("TF32_RN", "F32_RN")
            assert str(raw_coords) == str(rn_coords)
