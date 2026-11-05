"""Fail-closed, cache and allocation tests for the narrow shared-A M-edge path."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_a_tma_m_edge import INDUCTOR_NATIVE
from test.test_cute_shared_a_tma_m_edge import args_for
from test.test_cute_shared_a_tma_m_edge import bind
from test.test_cute_shared_a_tma_m_edge import capture_code
from test.test_cute_shared_a_tma_m_edge import cpu_only as cpu_only
from test.test_cute_shared_a_tma_m_edge import load_kernel
from test.test_cute_shared_a_tma_m_edge import native_calls
from test.test_cute_shared_a_tma_m_edge import pairs
from test.test_cute_shared_a_tma_m_edge import synthetic_source

import helion
from helion._compiler.cute import cute_mma_dag_tma_m_edge as edge
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.runtime_tensor_guards import (
    _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
)
from helion.autotuner.config_generation import ConfigGeneration


def axes(bound):
    facts = bound.config_spec.cute_mma_dag_facts
    return (
        None
        if facts is None or facts.tcgen05 is None
        else facts.tcgen05.shared_a_m_edge_axes
    )


@pytest.mark.parametrize(
    "change",
    [
        "output_exact_alias",
        "output_partial_alias",
        "B_alias",
        "A_B_alias",
        "A_offset",
        "B_offset",
        "D_offset",
        "A_pitch",
        "D_pitch",
        "partial_N",
        "partial_K",
        "row_major_A",
        "mixed_B_dtype",
    ],
)
def test_genuine_argument_refusals_do_not_declare_edge_authority(change):
    source = synthetic_source()
    args = list(args_for(n=128))
    if change == "output_exact_alias":
        args[-1] = args[-2]
    elif change == "output_partial_alias":
        base = torch.empty(1000 * 128 + 32, dtype=torch.float32)
        args[-2] = base[: 1000 * 128].view(1000, 128)
        args[-1] = base[32:].view(1000, 128)
    elif change == "B_alias":
        args[2] = args[1]
    elif change == "A_B_alias":
        base = torch.empty(1000 * 128, dtype=torch.bfloat16)
        args[0] = base.view(128, 1000).t()
        args[1] = base[: 128 * 128].view(128, 128)
    elif change.endswith("_offset"):
        index = {"A_offset": 0, "B_offset": 1, "D_offset": 3}[change]
        original = args[index]
        base = torch.empty(original.numel() + 1, dtype=original.dtype)
        args[index] = base.as_strided(
            original.shape, original.stride(), storage_offset=1
        )
    elif change == "A_pitch":
        args[0] = torch.empty_strided((1000, 128), (1, 1004), dtype=torch.bfloat16)
    elif change == "D_pitch":
        args[3] = torch.empty_strided((1000, 128), (130, 1), dtype=torch.float32)
    elif change == "partial_N":
        args = list(args_for(n=130))
    elif change == "partial_K":
        args = list(args_for(n=128, k=130))
    elif change == "row_major_A":
        args[0] = torch.empty((1000, 128), dtype=torch.bfloat16)
    elif change == "mixed_B_dtype":
        args[1] = torch.empty((128, 128), dtype=torch.float16)
    bound = bind(load_kernel(source), tuple(args))
    assert axes(bound) is None
    assert not any(c.get("cute_mma_dag") == "tcgen05" for _, c in pairs(bound))


@pytest.mark.parametrize(
    "change",
    [
        "computed_carry",
        "nonzero_init",
        "negative_zero_init",
        "cross_carry",
        "extra_store",
        "different_A_index",
        "explicit_mask",
        "extra_side_effect",
        "extra_aux",
    ],
)
def test_complete_live_program_refusals_not_pseudo_metadata(change):
    source = synthetic_source()
    if change == "computed_carry":
        source = source.replace(
            "torch.addmm(acc0,v,w0)", "torch.addmm(acc0 * 0.5,v,w0)"
        )
    elif change == "cross_carry":
        source = source.replace("torch.addmm(acc0,v,w0)", "torch.addmm(acc1,v,w0)")
    elif change == "nonzero_init":
        source = source.replace(
            "acc0=hl.zeros([m,n],dtype=torch.float32)",
            "acc0=hl.full([m,n],1.0,dtype=torch.float32)",
        )
    elif change == "negative_zero_init":
        source = source.replace(
            "acc0=hl.zeros([m,n],dtype=torch.float32)",
            "acc0=hl.full([m,n],-0.0,dtype=torch.float32)",
        )
    elif change == "extra_store":
        source = source.replace(
            "        d0[m,n]=", "        d1[m,n]=acc0\n        d0[m,n]="
        )
    elif change == "different_A_index":
        source = source.replace("acc1=", "acc1=")  # preserve the separate real carry
        source = source.replace(
            "torch.addmm(acc1,v,w1)",
            "torch.addmm(acc1,a[m,k].to(torch.float32).to(a.dtype) * 0.5,w1)",
        )
    elif change == "explicit_mask":
        source = source.replace(
            "a[m,k].to", "hl.load(a,[m,k],extra_mask=m.index[:,None] % 2 == 0).to"
        )
    elif change == "extra_side_effect":
        source = source.replace(
            "        d0[m,n]=", "        hl.atomic_add(d0,[m,n],acc1)\n        d0[m,n]="
        )
    elif change == "extra_aux":
        source = source.replace("d0[m,n]=acc0.to", "d0[m,n]=(acc0 + d1[m,n]).to")
    bound = bind(load_kernel(source), args_for(n=128))
    assert axes(bound) is None
    assert not any(c.get("cute_mma_dag") == "tcgen05" for _, c in pairs(bound))


def test_cached_separate_output_bind_cannot_authorize_later_overlap():
    kernel = load_kernel(synthetic_source())
    args = args_for(n=128)
    original = bind(kernel, args)
    assert axes(original) is not None
    base = torch.empty(1000 * 128 + 32, dtype=torch.float32)
    overlap = (*args[:3], base[: 1000 * 128].view(1000, 128), base[32:].view(1000, 128))
    rebound = bind(kernel, overlap)
    assert rebound is not original
    assert axes(rebound) is None
    assert pairs(original) and not pairs(rebound)
    assert bind(kernel, args) is original


@pytest.mark.parametrize("p", [1, 5])
def test_consumer_count_bound_is_general(p):
    bound = bind(load_kernel(synthetic_source(p)), args_for(p, n=128))
    assert axes(bound) is None


@pytest.mark.parametrize(
    "override",
    [
        "persistent",
        "pass_warps",
        "warps",
        "N_tail",
        "K_tail",
        "TMEM_overflow",
        "SMEM_overflow",
    ],
)
def test_explicit_invalid_schedule_remains_loud(override):
    bound = bind(load_kernel(INDUCTOR_NATIVE), args_for())
    config = helion.Config.from_dict(dict(pairs(bound)[0][1]))
    if override == "persistent":
        config.config["cute_mma_dag_persistent"] = 1
    elif override == "pass_warps":
        config.config["cute_mma_dag_pass_warps"] = "on"
    elif override == "warps":
        config.config["cute_mma_dag_warps"] = 8
    elif override in ("N_tail", "K_tail"):
        blocks = list(config.block_sizes)
        blocks[1 if override == "N_tail" else 2] = 512
        config.config["block_sizes"] = blocks
    elif override == "TMEM_overflow":
        # This config also breaks the full N tile guard; the P4 boundary below
        # checks the independent TMEM allocation budget itself.
        blocks = list(config.block_sizes)
        blocks[1] = 512
        config.config["block_sizes"] = blocks
    elif override == "SMEM_overflow":
        config.config["block_sizes"] = [128, 128, 128]
        config.config["cute_mma_dag_stages"] = 8
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(config)


def test_P4_TMEM_and_all_simultaneous_SMEM_priced_not_one_GEMM_reservation():
    bound = bind(load_kernel(synthetic_source(4)), args_for(4, n=256))
    config = helion.Config.from_dict(dict(pairs(bound)[0][1]))
    config.config["block_sizes"] = [128, 256, 128]
    config.config["cute_mma_dag_stages"] = 1
    with pytest.raises(helion.exc.InvalidConfig, match="TMEM"):
        bound.to_code(config)
    code, plan = capture_code(bound, pairs(bound)[0][1])
    assert len(plan.loop_tiles) == 5
    assert plan.stage_bytes == sum(
        __import__(
            "helion._compiler.cute.cute_mma_dag_tcgen05", fromlist=["_align"]
        )._align(t.bytes, 1024)
        for t in plan.loop_tiles
    )
    specs = [t.tma for t in plan.loop_tiles if t.tma is not None]
    assert len(specs) == len(plan.loop_tiles)
    assert plan.tx_bytes == sum(s.bytes for s in specs)
    assert plan.fixed_offset == plan.stages * plan.stage_bytes
    allocation = native_calls(code, "alloc_smem")
    emitted_bytes = sum(
        {"cutlass.Uint8": 1, "cutlass.Int32": 4, "cutlass.Int64": 8}[
            ast.unparse(c.args[0])
        ]
        * ast.literal_eval(c.args[1])
        for c in allocation
    )
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    # Obtain the actual resolved axes from the config, not positional shape
    # assumptions or aggregate payload-as-occupancy.
    size_map = dict(
        zip(
            bound.config_spec.block_sizes.valid_block_ids(),
            pairs(bound)[0][1].block_sizes,
            strict=True,
        )
    )
    sizes = [facts.block_size(i, size_map) for i in range(len(facts.axes))]
    from helion._compiler.cute.cute_mma_dag_tcgen05 import tcgen05_smem_bytes

    assert emitted_bytes <= tcgen05_smem_bytes(facts, sizes, plan.stages, True)
    assert emitted_bytes >= plan.stage_bytes * plan.stages


def test_selected_native_plan_with_missing_bound_disjoint_fact_is_not_scalar_fallback():
    bound = bind(load_kernel(INDUCTOR_NATIVE), args_for())
    config = pairs(bound)[0][1]
    saved = bound.env.bound_runtime_input_specialization_results.pop(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    try:
        with pytest.raises(
            helion.exc.BackendUnsupported, match="M-edge proof was lost"
        ):
            bound.to_code(config)
    finally:
        bound.env.bound_runtime_input_specialization_results[
            _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
        ] = saved


def test_optional_discovery_failure_keeps_default_schema_and_no_late_native_seed():
    with patch.object(
        edge,
        "prove_shared_a_m_edge_axes",
        side_effect=RuntimeError("optional early proof unavailable"),
    ):
        kernel = load_kernel(INDUCTOR_NATIVE)
        bound = bind(kernel, args_for())
    assert axes(bound) is None
    assert pairs(bound) == []
    with bound.env, bound.host_function:
        gen = ConfigGeneration(bound.config_spec)
        before = bound.config_spec.cache_fingerprint_hash()
        gen.initial_population_view()
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        assert edge.prove_shared_a_m_edge_axes(program) is not None
        assert gen.seed_flat_config_pairs() == []
        assert before == bound.config_spec.cache_fingerprint_hash()
    assert not native_calls(
        bound.to_code(bound.config_spec.default_config()), "tcgen05_mma"
    )
