"""CPU-only proof, source and search tests for SIMT-to-MMA PDL."""

from __future__ import annotations

import ast
from hashlib import sha256
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

from test.test_cute_materialize_operand import _args
from test.test_cute_materialize_operand import _bind
from test.test_cute_materialize_operand import _computed_rhs
from test.test_cute_materialize_operand import _cpu_b200 as _cpu_b200
from test.test_cute_materialize_operand import _direct_rhs
from test.test_cute_materialize_operand import _stage_sources

import helion
from helion._compiler.autotuner_heuristics.cute_materialized_operand import (
    CuteMaterializedOperandHeuristic,
)
from helion._compiler.cute.materialized_pdl import prove_materialized_operand_pdl
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from helion._testing import skipUnlessBackends
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.autotuner.compiler_coverage import CompilerCoverageGroup
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


KEY = "tcgen05_materialized_pdl"


def _group(bound: BoundKernel[Any]) -> CompilerCoverageGroup:
    return next(g for g in bound.config_spec.compiler_coverage_groups if g.key == KEY)


def _serial(bound: BoundKernel[Any]) -> helion.Config:
    return _group(bound).witnesses[2].carrier


def _device(source: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


@pytest.mark.parametrize(
    "kernel,packed", [(matmul_bf16_int4, True), (_computed_rhs, False)]
)
def test_pdl_source_changes_only_entry_release_and_consumer_launch(
    kernel: Kernel[Any], packed: bool
) -> None:
    bound = _bind(kernel, _args(packed=packed))
    serial = _serial(bound)
    pdl = helion.Config.from_dict(serial.config | {KEY: True})
    baseline = bound.to_code(serial)
    assert (
        bound.to_code(helion.Config.from_dict(serial.config | {KEY: False})) == baseline
    )
    changed = bound.to_code(pdl)
    before = _stage_sources(baseline)
    after = _stage_sources(changed)
    assert len(before) == len(after) == 2
    producer = _device(after[0])
    release = producer.body.pop(0)
    assert ast.unparse(release) == "cute.arch.griddepcontrol_launch_dependents()"
    assert ast.dump(producer) == ast.dump(_device(before[0]))
    assert ast.dump(_device(after[1])) == ast.dump(_device(before[1]))
    assert "cute.arch.griddepcontrol_wait()" in after[1]
    assert after[1].index("cute.arch.griddepcontrol_wait()") < after[1].index(
        "cute.copy(tma_atom_"
    )
    assert "sync_threads()" in after[1]
    assert after[1].count("wait_for_alloc()") == 1
    assert after[1].replace(", 'use_pdl': True", "") == before[1]
    # Full caller/allocation/ABI inverse, including the embedded source hashes.
    restored = changed
    for old, new in zip(before, after, strict=True):
        restored = restored.replace(repr(new), repr(old))
        restored = restored.replace(
            sha256(new.encode()).hexdigest(), sha256(old.encode()).hexdigest()
        )
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(baseline))


def test_coverage_is_additive_serial_default_and_acc1_is_capacity_derived() -> None:
    cuda_initialized = torch.cuda.is_initialized()
    bound = _bind(matmul_bf16_int4, _args(packed=True))
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    group = _group(bound)
    assert [w.value for w in group.witnesses] == [False, True, False, True]
    assert [w.carrier.config["tcgen05_acc_stages"] for w in group.witnesses] == [
        2,
        2,
        1,
        1,
    ]
    carrier = group.witnesses[2].carrier
    assert carrier.block_sizes[2:] == [128, 64, 128]
    assert carrier.config["tcgen05_ab_stages"] == 8
    assert KEY not in spec.default_config().config
    assert all(KEY not in seed.config for seed in spec.compiler_seed_configs)
    with bound.env:
        expanded = CuteMaterializedOperandHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        roots = state.materialized_operand_pdl_roots
        try:
            state.materialized_operand_pdl_roots = None
            original = CuteMaterializedOperandHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        finally:
            state.materialized_operand_pdl_roots = roots
        assert expanded[: len(original)] == original
        assert all(
            c.config["tcgen05_acc_stages"] == 1 for c in expanded[len(original) :]
        )
        generation = spec.create_config_generation()
        for witness in group.witnesses:
            requested = witness.carrier
            requested.config[KEY] = witness.value
            flat, effective = generation.strict_config_pair(requested)
            assert effective.config.get(KEY, False) is witness.value
            assert generation.unflatten(flat) == effective
        population, outcomes = append_compiler_coverage(
            [], generation, cached_configs=list, pin=lambda config: None
        )
        actual = [item for item in outcomes if item.mechanism == group.mechanism]
        assert len(actual) == 4
        assert all(item.outcome == "added" for item in actual)
        assert any(
            generation.unflatten(row).config.get(KEY) is True for row in population
        )
    assert torch.cuda.is_initialized() is cuda_initialized


@pytest.mark.parametrize(
    "kernel,packed", [(matmul_bf16_int4, True), (_computed_rhs, False)]
)
def test_coverage_strict_transfer_without_direct_entry_surface(
    kernel: Kernel[Any], packed: bool
) -> None:
    cuda_initialized = torch.cuda.is_initialized()
    # The smaller output still admits the paired pipeline, but cannot reach
    # the separate 256x256 direct-entry seed and its flat-role / FFI fields.
    bound = _bind(kernel, _args(shape=(128, 1024, 128), packed=packed))
    spec = bound.config_spec
    group = _group(bound)
    inactive = (
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
    )
    with bound.env:
        assert not spec._cute_tcgen05_config.full_tile_direct_entry_seed_eligible()
        assert all(key not in spec._flat_fields() for key in inactive)
        generation = spec.create_config_generation()
        assert [
            (w.carrier.config["tcgen05_acc_stages"], w.value) for w in group.witnesses
        ] == [(2, False), (2, True), (1, False), (1, True)]
        for witness in group.witnesses:
            requested = witness.carrier
            requested.config[KEY] = witness.value
            flat, effective = generation.strict_config_pair(requested)
            assert effective.config.get(KEY, False) is witness.value
            assert generation.unflatten(flat) == effective
            assert all(key not in effective.config for key in inactive)
            # Carrier construction may omit known neutral inactive settings;
            # strict transfer must still reject supplied fields that it drops.
            for key in inactive:
                supplied = helion.Config.from_dict(requested.config | {key: False})
                with pytest.raises(
                    InvalidConfig,
                    match=f"Coverage transfer changed supplied field '{key}'",
                ):
                    generation.strict_config_pair(supplied)
                supplied = helion.Config.from_dict(requested.config | {key: True})
                with pytest.raises(InvalidConfig):
                    generation.strict_config_pair(supplied)
        population, outcomes = append_compiler_coverage(
            [], generation, cached_configs=list, pin=lambda config: None
        )
        actual = [item for item in outcomes if item.mechanism == group.mechanism]
        assert len(actual) == 4
        assert all(item.outcome == "added" for item in actual)
        for item in actual:
            assert item.effective is not None
            assert generation.flatten(item.effective) in population
    assert torch.cuda.is_initialized() is cuda_initialized


@pytest.mark.parametrize(
    "update",
    [
        {KEY: 1},
        {"tcgen05_cta_group": "auto"},
        {"pid_type": "flat"},
        {"tcgen05_persistence_model": "clc"},
        {"tcgen05_strategy": "warp_specialized"},
        {"cute_materialized_operand_schedule": "packed_warp"},
        {"cute_materialized_schedule": "row_resident"},
    ],
)
def test_unsupported_schedules_cannot_inherit_pdl(update: dict[str, object]) -> None:
    bound = _bind(matmul_bf16_int4, _args(packed=True))
    config = helion.Config.from_dict(_serial(bound).config | {KEY: True} | update)
    with bound.env, pytest.raises(InvalidConfig):
        bound.config_spec.normalize(config)


def test_no_typed_materialization_means_no_pdl_fragment() -> None:
    bound = _bind(_direct_rhs, _args())
    assert bound.env.cute_fission_plan is None
    assert not any(g.key == KEY for g in bound.config_spec.compiler_coverage_groups)
    with bound.env, pytest.raises(InvalidConfig):
        bound.config_spec.normalize(
            helion.Config.from_dict(
                bound.config_spec.default_config().config | {KEY: True}
            )
        )


def test_missing_proof_rejects_an_otherwise_legal_pdl_carrier() -> None:
    bound = _bind(matmul_bf16_int4, _args(packed=True))
    state = bound.config_spec._cute_tcgen05_config
    config = helion.Config.from_dict(_serial(bound).config | {KEY: True})
    state.materialized_operand_pdl_roots = None
    with bound.env, pytest.raises(InvalidConfig):
        bound.config_spec.normalize(config)


def test_actual_retraced_scratch_edge_is_required() -> None:
    from helion.language import memory_ops

    with patch(
        "helion._compiler.cute.materialized_pdl.prove_materialized_operand_pdl",
        wraps=prove_materialized_operand_pdl,
    ) as observed:
        bound = _bind(matmul_bf16_int4, _args(packed=True))
    env, ir, candidate = observed.call_args.args
    producer, consumer = (
        env.config_spec._cute_tcgen05_config.materialized_operand_pdl_roots
    )
    assert producer != consumer
    stores = [
        node
        for node in ir.graphs[producer].graph.nodes
        if node.target is memory_ops.store
    ]
    assert stores
    scratch_node = stores[0].args[0]
    original = scratch_node.meta["val"]
    with bound.env, bound.host_function:
        assert prove_materialized_operand_pdl(env, ir, candidate) == (
            producer,
            consumer,
        )
        try:
            # A different input-backed tensor cannot stand in for the fresh RHS.
            scratch_node.meta["val"] = candidate.operands.lhs.source_fake
            assert prove_materialized_operand_pdl(env, ir, candidate) is None
        finally:
            scratch_node.meta["val"] = original
        try:
            for node in stores:
                node.target = memory_ops.load
            assert prove_materialized_operand_pdl(env, ir, candidate) is None
        finally:
            for node in stores:
                node.target = memory_ops.store


def test_unaligned_input_cannot_enable_a_tma_dependent_launch() -> None:
    a, b = _args(packed=True)
    a = torch.empty(a.numel() + 1, dtype=a.dtype)[1:].view(a.shape)
    bound = _bind(matmul_bf16_int4, (a, b))
    assert bound.env.cute_fission_plan is not None
    # Input specialization facts are published after semantic edge registration.
    # Actual codegen must reject the unaligned consumer instead of returning a
    # bundle whose sender releases a scalar/fallback consumer without its wait.
    config = helion.Config.from_dict(_serial(bound).config | {KEY: True})
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        bound.to_code(config)


def test_partial_native_tile_does_not_inherit_the_full_tile_pdl_schedule() -> None:
    bound = _bind(_computed_rhs, _args(shape=(384, 1024, 512)))
    config = _serial(bound)
    config.block_sizes[2] = 256
    config.config.update({KEY: True, "tcgen05_ab_stages": 2})
    assert (
        not bound.config_spec._cute_tcgen05_config.materialized_operand_pdl_supported(
            config.config
        )
    )
    with bound.env, pytest.raises(InvalidConfig):
        bound.config_spec.normalize(config)


@pytest.mark.parametrize("disabled", [False, True])
def test_full_search_declares_coverage_without_automatic_heuristics(
    disabled: bool,
) -> None:
    bound = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        autotune_effort="full",
        static_shapes=False,
        cute_materialize_transformed_operands=True,
        disable_autotuner_heuristics=disabled,
    )._bind_isolated(_args(packed=True))
    assert [w.value for w in _group(bound).witnesses] == [False, True, False, True]
