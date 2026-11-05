"""Paired PDL follows actual launch facts, never CtaGroup geometry alone."""

from __future__ import annotations

import ast
from dataclasses import replace
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test.test_autotuner_heuristics import _grouped_worklist_bind_patches
from test.test_autotuner_heuristics import _grouped_worklist_kernel_body
from test.test_cute_device_state import _lifecycle
from test.test_cute_materialize_operand import _args
from test.test_cute_materialize_operand import _bind
from test.test_cute_materialize_operand import _computed_rhs
from test.test_cute_materialize_operand import _stage_sources
from test.test_cute_materialized_pdl import _serial
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_paired_clc_carrier,
)
from helion._compiler.cute.tcgen05_lifecycle import tcgen05_launch_uses_pdl
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl
from helion.runtime.cute.launcher import _cute_use_pdl

grouped_gemm_jagged = import_example("grouped_gemm").grouped_gemm_jagged

WAIT = "cute.arch.griddepcontrol_wait"
RELEASE = "cute.arch.griddepcontrol_launch_dependents"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def plain_paired_gemm(a, b):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        out[row, col] = acc.to(out.dtype)
    return out


def managed_plain(dtype=torch.bfloat16, m=4096, n=1024, k=128):
    args = (torch.empty((m, k), dtype=dtype), torch.empty((n, k), dtype=dtype).T)
    kernel = helion.kernel(plain_paired_gemm.fn, settings=plain_paired_gemm.settings)
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, args)
    return bound, args


def managed_grouped(m=10240, n=1024, k=1024, groups=16):
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=torch.int32) * (m // groups),
    )
    kernel = helion.Kernel(
        grouped_gemm_jagged.fn,
        settings=helion.Settings(
            backend="cute",
            static_shapes=False,
            autotune_effort="full",
            cute_segmented_matmul_tiling=True,
        ),
    )
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, args)
    return bound, args


def managed_worklist():
    args = (
        torch.empty((6 * 256, 128), dtype=torch.bfloat16),
        torch.empty((6, 256, 128), dtype=torch.bfloat16),
        torch.tensor([[i, i * 256, 256, 256] for i in range(6)], dtype=torch.int32),
    )
    kernel = helion.kernel(
        _grouped_worklist_kernel_body, backend="cute", static_shapes=True
    )
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        bound = cpu_bind(kernel, args)
    return bound, args


def pdl_calls(source):
    return [
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (WAIT, RELEASE)
    ]


def wrapper_plans(source):
    plans = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_helion_cute_wrapper_plans"
            for target in node.targets
        ):
            plans.extend(ast.literal_eval(node.value))
    return plans


class StripPDL(ast.NodeTransformer):
    """Delete only exact bare dependency calls and their now-empty if shell."""

    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and not node.value.args
            and not node.value.keywords
            and ast.unparse(node.value.func) in (WAIT, RELEASE)
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)
        return node if node.body or node.orelse else None


def without_pdl(source):
    return ast.dump(StripPDL().visit(ast.parse(source)), include_attributes=False)


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@pytest.mark.parametrize(
    "general,materialized,clc,fission",
    tuple(itertools.product((False, True), repeat=4)),
)
def test_effective_launch_fact_matrix_preserves_CLC_and_materialized_forcing(
    general, materialized, clc, fission
):
    config = {"cute_pdl": general, "tcgen05_materialized_pdl": materialized}
    result = tcgen05_launch_uses_pdl(config, clc_persistent=clc, fission_bundle=fission)
    # Exactly the old host AB-plan condition: CLC OR materialized OR
    # the general-launch predicate, not a shape/CTA-group condition.
    old_general = general and not materialized and not fission
    assert result is (clc or materialized or old_general)
    assert (
        tcgen05_launch_uses_pdl({}, clc_persistent=False, fission_bundle=False) is False
    )


@pytest.mark.parametrize(
    "field",
    ("cute_pdl", "tcgen05_materialized_pdl", "clc_persistent", "fission_bundle"),
)
@pytest.mark.parametrize("value", (None, 1, "false"))
def test_unknown_launch_contract_fact_is_refused_not_guessed(field, value):
    cfg = {"cute_pdl": False, "tcgen05_materialized_pdl": False}
    kwargs = {"clc_persistent": False, "fission_bundle": False}
    if field in kwargs:
        kwargs[field] = value
    else:
        cfg[field] = value
    with pytest.raises(helion.exc.InvalidConfig, match="proved booleans"):
        tcgen05_launch_uses_pdl(cfg, **kwargs)


@pytest.mark.parametrize(
    "pdl_release,paired_pdl", tuple(itertools.product((False, True), repeat=2))
)
def test_lifecycle_legacy_direct_default_and_one_CTA_field_meaning_unchanged(
    pdl_release, paired_pdl
):
    one = _lifecycle(pdl_release=pdl_release, paired_pdl_release=paired_pdl)
    lines = one.render_store_post_loop_lines(
        tma_store_pipeline_tail="stores.producer_tail()"
    )
    assert any(RELEASE in x for x in lines) is pdl_release
    assert lines == replace(
        one, paired_pdl_release=not paired_pdl
    ).render_store_post_loop_lines(tma_store_pipeline_tail="stores.producer_tail()")
    legacy = _lifecycle(is_two_cta=True)
    assert any(RELEASE in x for x in legacy.render_store_post_loop_lines())
    actual = replace(legacy, paired_pdl_release=False)
    assert not any(RELEASE in x for x in actual.render_store_post_loop_lines())
    assert actual.render_store_post_loop_lines() == [
        line for line in legacy.render_store_post_loop_lines() if RELEASE not in line
    ]


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("pid", ("persistent_blocked", "persistent_interleaved"))
@pytest.mark.parametrize("small", (False, True))
def test_genuine_managed_paired_static_wait_release_and_host_launch_agree(
    dtype, pid, small
):
    bound, args = managed_plain(
        dtype, m=256 if small else 4096, n=128 if small else 1024
    )
    assert bound.config_spec.cute_tcgen05_search_enabled
    original_facts = dict(bound.env.bound_runtime_input_specialization_results)
    common = {
        "block_sizes": [256, 128, 64],
        "pid_type": pid,
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
    }
    for flag in (False, True):
        cfg = helion.Config.from_dict({**common, "cute_pdl": flag})
        source = bound.to_code(cfg)
        normal = dict(bound._normalized_config_copy(cfg))
        assert normal["cute_pdl"] is flag and normal["tcgen05_cta_group"] == "two"
        ab = next(p for p in wrapper_plans(source) if p["kind"] == "tcgen05_ab_tma")
        assert bool(ab.get("use_pdl", False)) is flag
        calls = pdl_calls(source)
        assert bool(calls) is flag
        if flag:
            assert WAIT in calls and RELEASE in calls
        else:
            assert not calls and "._helion_cute_use_pdl = True" not in source
            assert not _cute_use_pdl(SimpleNamespace())
        assert "CtaGroup.TWO" in source and "mcast_mask" in source
        assert source.count("PipelineTmaStore.create(") == 1
        assert "tcgen05_c_pipeline.producer_tail()" in source
        assert "tcgen05_tmem_allocator.free(" in source
        assert bound.env.bound_runtime_input_specialization_results == original_facts


@pytest.mark.parametrize("pdl", (False, True))
def test_managed_grouped_CLC_forces_PDL_even_when_general_flag_off(pdl):
    with mock_cuda_unavailable(), _target():
        bound, args = managed_grouped()
        with bound.env:
            cfg = grouped_row_union_paired_clc_carrier(
                bound.env, bound.host_function.device_ir
            )
        assert cfg is not None
        cfg = helion.Config.from_dict({**dict(cfg), "cute_pdl": pdl})
        source = bound.to_code(cfg)
        plans = wrapper_plans(source)
        ab = next(x for x in plans if x["kind"] == "tcgen05_ab_tma")
        assert ab["use_pdl"] is True
        assert WAIT in pdl_calls(source) and RELEASE in pdl_calls(source)
        assert (
            "clc_response" in source and "tcgen05_c_pipeline.producer_tail()" in source
        )
        assert "tcgen05_tmem_allocator.free(" in source
        assert "pipeline_init_wait(" in source or "cluster_wait(" in source


@pytest.mark.parametrize(
    "general,materialized", tuple(itertools.product((False, True), repeat=2))
)
def test_materialized_pair_general_inert_and_handoff_forced_PDLenable(
    general, materialized
):
    bound = _bind(_computed_rhs, _args())
    serial = _serial(bound)
    cfg = helion.Config.from_dict(
        {**dict(serial), "cute_pdl": general, "tcgen05_materialized_pdl": materialized}
    )
    source = bound.to_code(cfg)
    stages = _stage_sources(source)
    consumer = stages[-1]
    assert "CtaGroup.TWO" in consumer
    ab = next(x for x in wrapper_plans(consumer) if x["kind"] == "tcgen05_ab_tma")
    assert bool(ab.get("use_pdl", False)) is materialized
    assert bool(pdl_calls(consumer)) is materialized
    if materialized:
        assert WAIT in pdl_calls(consumer) and RELEASE in pdl_calls(consumer)
    else:
        assert "._helion_cute_use_pdl = True" not in consumer
    assert (
        "PipelineTmaStore.create(" in consumer
        and "producer_tail()" in consumer
        and "tcgen05_tmem_allocator.free(" in consumer
    )


def test_paired_emission_without_actual_plan_fails_closed():
    splitter = object.__new__(Tcgen05PersistentProgramIDs)
    with (
        patch.object(Tcgen05PersistentProgramIDs, "_tcgen05_plan", return_value=None),
        pytest.raises(
            helion.exc.BackendUnsupported, match="requires a tcgen05 launch plan"
        ),
    ):
        splitter._tcgen05_pair_launch_uses_pdl(
            SimpleNamespace(config={"cute_pdl": False})
        )


@pytest.mark.parametrize("pdl", (False, True))
def test_managed_grouped_worklist_static_role_and_host_flags_agree(pdl):
    with _grouped_worklist_bind_patches(analysis_runtime_n_ptx=True):
        bound, args = managed_worklist()
        original = bound.config_spec.default_config()
        cfg = helion.Config.from_dict({**dict(original), "cute_pdl": pdl})
        source = bound.to_code(cfg)
        assert "CtaGroup.TWO" in source
        ab = next(x for x in wrapper_plans(source) if x["kind"] == "tcgen05_ab_tma")
        assert bool(ab.get("use_pdl", False)) is pdl
        assert bool(pdl_calls(source)) is pdl
        assert (
            "tcgen05_c_pipeline.producer_tail()" in source
            and "tcgen05_tmem_allocator.free(" in source
        )
        assert (
            bound.config_spec._cute_tcgen05_config.grouped_worklist_smem_facts
            is not None
        )


@pytest.mark.parametrize("value", (None, 1, "off"))
def test_lifecycle_rejects_unknown_paired_release_fact(value):
    invalid = _lifecycle(is_two_cta=True, paired_pdl_release=value)
    with pytest.raises(helion.exc.InvalidConfig, match="proved boolean"):
        invalid.render_store_post_loop_lines()
