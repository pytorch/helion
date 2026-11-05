"""Compact seeds are a tagged readonly-bool source extension, not legacy fanout."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_bool_aux_native import _args
from test.test_cute_bool_aux_native import _chains
from test.test_cute_bool_aux_native import _external_bool_epilogue
from test.test_cute_mixed_output_stores import _arguments
from test.test_cute_mixed_output_stores import mixed_outputs

from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def test_compact_seed_subject_uses_complete_original_tagged_bool_source_identities():
    bound = cpu_bind(_external_bool_epilogue, _args())
    state = bound.config_spec._cute_tcgen05_config
    subject = state._readonly_bool_aux_seed_subject
    assert subject is not None
    anchor, stores, loads = subject
    graphs, actual_anchor, chains = _chains(bound)
    assert anchor is actual_anchor
    assert chains is not None and stores == tuple(store for store, _ in chains)
    tagged = tuple(
        dict.fromkeys(
            leaf.load_node
            for _, chain in chains
            for leaf in chain.auxiliary_tensor_loads
            if isinstance(leaf, cute_epilogue._ReadonlyBoolAuxLoadExpr)
        )
    )
    assert loads == tagged and len(loads) == 1 and len(stores) == 4
    assert all(isinstance(node, torch.fx.Node) for node in (anchor, *stores, *loads))
    assert (
        isinstance(subject, tuple)
        and isinstance(stores, tuple)
        and isinstance(loads, tuple)
    )
    before = dict(bound.env.bound_runtime_input_specialization_results)
    registered = dict(bound.env.runtime_input_specializations)
    seeds = state._multi_output_seed_configs()
    assert seeds
    assert state._readonly_bool_aux_seed_subject is subject
    assert before == bound.env.bound_runtime_input_specialization_results
    assert registered == bound.env.runtime_input_specializations
    # The certificate is seed-only, never a replacement for final alias facts.
    with bound.env, bound.host_function:
        key = "tensor_storage_disjoint_matrix_v1"
        saved = bound.env.bound_runtime_input_specialization_results.pop(key)
        try:
            assert (
                cute_epilogue._readonly_bool_auxiliary_cast(
                    next(
                        node
                        for node in loads[0].users
                        if node.target is torch.ops.prims.convert_element_type.default
                    ),
                    carrier_tile_shape=tuple(loads[0].meta["val"].shape),
                    carrier_tile_index_nodes=tuple(loads[0].args[1]),
                    carrier_global_shape=(128, 128),
                    graphs=graphs,
                )
                is None
            )
        finally:
            bound.env.bound_runtime_input_specialization_results[key] = saved


@pytest.mark.parametrize(
    "mode",
    (
        "implicit",
        "bool_half",
        "float_bool",
        "extra_mask",
        "offset_index",
        "transpose",
        "sin",
    ),
)
def test_unproved_bool_sources_receive_no_compact_seed_certificate(mode):
    bound = cpu_bind(_external_bool_epilogue, (*_args(), mode))
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is None
    assert state._multi_output_seed_configs() == []


@pytest.mark.parametrize("alias", ("boolean_output", "outputs", "residual"))
def test_alias_rejected_bool_sources_receive_no_compact_seed_subject(alias):
    bound = cpu_bind(_external_bool_epilogue, _args(alias=alias))
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is None
    assert state._multi_output_seed_configs() == []


def test_fake_only_alias_inequality_is_not_a_compact_bool_seed_subject():
    with FakeTensorMode() as mode:
        values = tuple(mode.from_tensor(value) for value in _args())
    bound = cpu_bind(_external_bool_epilogue, values)
    assert (
        bound.config_spec._cute_tcgen05_config._readonly_bool_aux_seed_subject is None
    )


def test_nonbool_mixed_certificate_and_ordinary_seed_planner_remain_independent():
    bound = cpu_bind(mixed_outputs, _arguments(torch.bfloat16, m=2048, n=8008, k=2560))
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is None
    assert state._multi_output_seed_configs() == []
    assert state.mixed_output_store_plan is not None
    assert bound.config_spec.compiler_seed_configs


def test_no_subject_default_false_preserves_legacy_even_if_arena_could_fit():
    state = CuteTcgen05Config(SimpleNamespace())
    assert state._readonly_bool_aux_seed_subject is None
    state.search_enabled = True
    with patch.object(
        state,
        "_matmul_block_fragments",
        side_effect=AssertionError("No legacy compact subject"),
    ):
        assert state._multi_output_seed_configs() == []
