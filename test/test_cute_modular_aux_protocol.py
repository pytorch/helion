"""Source protocol constraints must agree with strict normalizer and lowering."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_mapped_aux_codegen import _mapped_rows
from test.test_cute_mapped_aux_codegen import _row_args
from test.test_cute_mixed_output_stores import _arguments as mixed_args
from test.test_cute_mixed_output_stores import mixed_outputs
from test.test_cute_modular_aux_codegen import _mod_cols
from test.test_cute_modular_aux_codegen import _mod_rows
from test.test_cute_modular_aux_codegen import arguments

import helion
from helion import exc
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import create
from helion._compiler.cute import cute_mma
from helion._compiler.device_ir import lower_to_device_ir
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_mixed(a, b, aux, narrow, wide):
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            row.index.reshape([row.index.size(0), 1]) % aux.size(0),
            col.index.reshape([1, col.index.size(0)]),
        ].to(torch.float32)
        narrow[row, col] = (acc + value).to(a.dtype)
        wide[row, col] = (acc * value).to(torch.float32)


def bind(function, args, managed=False):
    if managed:
        with patch.object(function, "_bind_isolated", side_effect=function.bind):
            return cpu_bind(function, args)
    return cpu_bind(function, args)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("managed", (False, True))
def test_subject_is_private_complete_immutable_and_default_configs_remain_valid(
    axis, managed
):
    b = bind(
        _mod_rows if axis == 0 else _mod_cols, arguments(torch.bfloat16, axis), managed
    )
    state = b.config_spec._cute_tcgen05_config
    subject = state._modular_aux_protocol_subject
    assert subject is not None
    assert subject.maps and subject.maps[0].split_axis == axis
    assert subject.stores and subject.anchor is not None
    assert not any(
        isinstance(getattr(subject, field), torch.Tensor)
        for field in subject.__dataclass_fields__
    )
    with pytest.raises(FrozenInstanceError):
        subject.block_ids = (1, 0)
    facts = dict(b.env.bound_runtime_input_specialization_results)
    registrations = dict(b.env.runtime_input_specializations)
    for c in (
        b.config_spec.default_config(),
        b.config_spec._base_default_config(),
        helion.Config(block_sizes=[128, 32, 64], pid_type="flat"),
    ):
        s = b.to_code(c)
        assert "tcgen05_ab_tma" in s and "tcgen05_aux_view2d" in s
        assert state._modular_aux_protocol_subject is subject
    assert facts == b.env.bound_runtime_input_specialization_results
    assert registrations == b.env.runtime_input_specializations
    fields = state.optional_fragments(for_search=True)
    for key, value in subject.inert_options():
        if key in fields:
            assert fields[key].choices == (value,)
    assert fields["tcgen05_cta_group"].choices == ("auto",)
    assert state.strategy_autotune_fragments()[
        "tcgen05_warp_spec_c_input_warps"
    ].choices == (0,)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize(
    "key,value",
    (
        ("tcgen05_cta_group", "two"),
        ("tcgen05_cluster_m", 2),
        ("tcgen05_cluster_n", 2),
        ("tcgen05_warp_spec_c_input_warps", 1),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_aux_load_mode", "epi_tma"),
        ("tcgen05_aux_prefetch_subtiles", 1),
        ("tcgen05_aux_fragment_hoist", True),
        ("tcgen05_epi_warp_groups", 2),
        ("tcgen05_epi_subtile_n", 32),
        ("tcgen05_layout_strategy", "explicit_epi_tile"),
    ),
)
def test_explicit_uncomposed_protocols_are_refused_during_normalization(
    axis, key, value
):
    b = bind(_mod_rows if axis == 0 else _mod_cols, arguments(torch.bfloat16, axis))
    c = helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=2,
    )
    c.config[key] = value
    if key == "tcgen05_layout_strategy":
        c.config.update(
            tcgen05_layout_overrides_epi_tile_m=128,
            tcgen05_layout_overrides_epi_tile_n=32,
            tcgen05_layout_overrides_d_store_box_n=32,
        )
    with pytest.raises(exc.InvalidConfig):
        b.config_spec.normalized_config(c)


@pytest.mark.parametrize(
    "axis,period,blocks",
    (
        (0, 192, [128, 32, 64]),
        (1, 128, [128, 256, 64]),
        (0, 128, [256, 32, 64]),
        (0, 128, [128, 128, 64]),
    ),
)
def test_geometry_is_rejected_early_not_by_late_modular_store_guard(
    axis, period, blocks
):
    b = bind(
        _mod_rows if axis == 0 else _mod_cols,
        arguments(torch.bfloat16, axis, period=period),
    )
    with pytest.raises(exc.InvalidConfig, match="modular"):
        b.config_spec.normalized_config(
            helion.Config(block_sizes=blocks, pid_type="flat")
        )


@pytest.mark.parametrize("axis", (0, 1))
def test_repaired_search_configs_are_fixed_points_and_late_render(axis):
    b = bind(_mod_rows if axis == 0 else _mod_cols, arguments(torch.bfloat16, axis))
    state = b.config_spec._cute_tcgen05_config
    c = helion.Config(
        block_sizes=[256, 256, 64],
        pid_type="persistent_interleaved",
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=2,
        tcgen05_ab_stages=2,
        tcgen05_aux_load_mode="epi_tma",
        tcgen05_warp_spec_c_input_warps=1,
        tcgen05_aux_prefetch_subtiles=1,
        tcgen05_aux_fragment_hoist=True,
        tcgen05_epi_warp_groups=2,
        tcgen05_epi_subtile_n=32,
    )
    b.config_spec.normalize(c, _fix_invalid=True)
    assert (
        state._modular_aux_protocol_subject.refusal(
            c.config, state._modular_aux_block_indices()
        )
        is None
    )
    assert b.config_spec.normalized_config(c) == c
    assert "tcgen05_aux_view2d" in b.to_code(c)
    assert c.config["tcgen05_cluster_m"] == c.config["tcgen05_cluster_n"] == 1
    assert c.config["tcgen05_warp_spec_c_input_warps"] == 0


@pytest.mark.parametrize("managed", (False, True))
def test_mixed_width_modular_source_retains_collective_domain_without_native_falsepositive(
    managed,
):
    a, b, aux, wide = arguments(torch.bfloat16)
    narrow = torch.empty_like(wide, dtype=torch.bfloat16)
    bound = bind(_mod_mixed, (a, b, aux, narrow, wide), managed)
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert bound.config_spec._cute_tcgen05_config._modular_aux_protocol_subject is None
    assert bound.config_spec._cute_tcgen05_config.mixed_output_store_plan is None
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        node = next(
            n
            for g in graphs
            for n in g.graph.nodes
            if n.target is torch.ops.aten.addmm.default
        )
        analysis = cute_mma.analyze_cute_mma_node(
            node, device_ir=bound.host_function.device_ir
        )
        assert analysis is not None
        assert not cute_mma.tcgen05_half_source_store_eligible(
            graphs, node, analysis.operands
        )
    s = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_collective_mma=True,
            cute_collective_compute="tcgen05",
            cute_collective_tmem_a=True,
        )
    )
    assert "collective_mma" in s and "tcgen05_ab_tma" not in s


@pytest.mark.parametrize("fault", ("alias", "fake"))
def test_no_modular_constraint_from_unproved_runtime_alias_subject(fault):
    a, b, aux, out = arguments(torch.bfloat16)
    if fault == "alias":
        aux = out[:128, :]
    args = (a, b, aux, out)
    if fault == "fake":
        with FakeTensorMode() as mode:
            args = tuple(mode.from_tensor(x) for x in args)
    bound = bind(_mod_rows, args)
    assert bound.config_spec._cute_tcgen05_config._modular_aux_protocol_subject is None
    assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("change", ("removed_mma", "no_loops", "unproved_modulo"))
def test_real_relowering_clears_old_protocol_subject_before_decline(change):
    b = bind(_mod_rows, arguments(torch.bfloat16))
    state = b.config_spec._cute_tcgen05_config
    old = state._modular_aux_protocol_subject
    assert old is not None
    facts = dict(b.env.bound_runtime_input_specialization_results)
    if change == "no_loops":
        original = b.host_function.body[:]
        b.host_function.body.clear()
        try:
            with b.env, pytest.raises(exc.NoDeviceLoopsInKernel):
                lower_to_device_ir(b.host_function)
        finally:
            b.host_function.body[:] = original
    else:
        changes = 0
        for statement in b.host_function.body:
            for node in ast.walk(statement):
                if (
                    change == "removed_mma"
                    and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "addmm"
                ):
                    node.func.attr = "add"
                    node.args = node.args[:1] + node.args[:1]
                    changes += 1
                elif (
                    change == "unproved_modulo"
                    and isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Mod)
                ):
                    assert isinstance(node, ExtendedAST)
                    with node:
                        node.left = create(
                            ast.BinOp,
                            left=node.left,
                            op=ast.Add(),
                            right=create(ast.Constant, value=1),
                        )
                    changes += 1
        assert changes == 1
        with b.env:
            lower_to_device_ir(b.host_function)
    assert state._modular_aux_protocol_subject is None
    assert b.env.bound_runtime_input_specialization_results == facts


@pytest.mark.parametrize("missing", ("bound", "registered"))
def test_protocol_subject_does_not_replace_final_runtime_proofs_or_publish_facts(
    missing,
):
    b = bind(_mod_rows, arguments(torch.bfloat16), managed=True)
    state = b.config_spec._cute_tcgen05_config
    subject = state._modular_aux_protocol_subject
    assert subject is not None
    key = "tensor_storage_disjoint_matrix_v1"
    table = (
        b.env.bound_runtime_input_specialization_results
        if missing == "bound"
        else b.env.runtime_input_specializations
    )
    saved = table.pop(key)
    config = helion.Config(block_sizes=[128, 32, 64], pid_type="flat")
    try:
        before = dict(table)
        b.config_spec.normalized_config(config)
        assert before == table
        source = b.to_code(config)
        # Losing either fact rejects the native mapped proof; the existing
        # universal fallback is legitimate and must not be changed to an error.
        assert "tcgen05_ab_tma" not in source
        assert "tcgen05_aux_view2d" not in source
        assert before == table
        assert state._modular_aux_protocol_subject is subject
    finally:
        table[key] = saved


@pytest.mark.parametrize("kind", ("rank3", "mixed"))
def test_nonmodular_subject_constraints_and_seed_admission_remain_untouched(kind):
    b = (
        bind(_mapped_rows, _row_args(torch.bfloat16))
        if kind == "rank3"
        else bind(mixed_outputs, mixed_args(torch.bfloat16))
    )
    state = b.config_spec._cute_tcgen05_config
    assert state._modular_aux_protocol_subject is None
    seeds = b.config_spec.compiler_seed_configs
    assert state.admit_modular_seed_configs(seeds) is seeds
