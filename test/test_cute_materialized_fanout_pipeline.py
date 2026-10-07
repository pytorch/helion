from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_epilogue_fanout import _store_records
from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401
from test.test_cute_epilogue_fanout import original_bound

import helion
from helion._compiler.cute import materialized_mma
from helion._compiler.cute.epilogue_fanout import combine_stores
from helion._compiler.cute.epilogue_fanout import prove_paired_fanout
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import memory_ops

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel


def upstream_auxiliary(
    left: torch.Tensor, right: torch.Tensor, next_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    intermediate = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    rounded = torch.empty_like(intermediate)
    output = torch.empty_like(intermediate)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            accumulator = left[row, :] @ right[:, column]
            intermediate[row, column] = torch.relu(accumulator)
        for column in hl.tile(columns):
            accumulator = hl.zeros([row, column], dtype=torch.float32)
            for inner in hl.tile(columns):
                accumulator = torch.addmm(
                    accumulator, intermediate[row, inner], next_weight[inner, column]
                )
            value = torch.relu(accumulator + 0.25).to(left.dtype)
            rounded[row, column] = value
            output[row, column] = value + intermediate[row, column]
    return intermediate, rounded, output


def paired_config(
    bound: BoundKernel[Any], placement: str = "pre_acc_wait"
) -> helion.Config:
    return bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[128, 256, 128, 256, 128, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_cta_group="two",
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_epilogue_fanout="shared",
            tcgen05_region_ab_stages=[4, 8],
            tcgen05_region_c_stages=[4, 2],
            tcgen05_materialized_pdl=True,
            tcgen05_c_acquire_placement="before_store",
            tcgen05_aux_load_placement=placement,
        )
    )


def device_sources(source: str) -> list[str]:
    return [
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.args[0], ast.Constant)
    ]


def test_original_recipe_and_additive_seed_roundtrip() -> None:
    bound = original_bound()
    explicit = paired_config(bound)
    source = bound.to_code(explicit)
    groups = bound.config_spec.compiler_coverage_groups
    (group,) = [
        g for g in groups if g.mechanism == "cute.materialized_fanout_aux_placement"
    ]
    assert group.dependencies[0].value == "shared"
    assert [w.value for w in group.witnesses] == ["post_acc_wait", "pre_acc_wait"]
    assert all(
        "tcgen05_region_ab_stages" not in seed.config
        for seed in bound.config_spec.compiler_seed_configs
    )
    assert TCGEN05_C_ACQUIRE_PLACEMENTS[0] == "pre_loop"
    assert TCGEN05_C_ACQUIRE_PLACEMENTS[-1] == "before_store"
    selected_configs = []
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        for witness in group.witnesses:
            request = witness.carrier
            request.config[group.key] = witness.value
            flat, selected = generation.strict_config_pair(request)
            assert generation.unflatten(flat) == selected
            selected_configs.append((selected, witness.value))
    for selected, value in selected_configs:
        assert bound.to_code(selected) == bound.to_code(paired_config(bound, value))
    assert (
        sum(module.count("'use_pdl': True") for module in device_sources(source)) == 2
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_non_example_upstream_auxiliary_wait_and_region_projection(
    dtype: torch.dtype,
) -> None:
    kernel = helion.kernel(
        upstream_auxiliary,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
    )
    bound = kernel._bind_isolated(
        tuple(
            torch.empty(shape, dtype=dtype)
            for shape in ((512, 128), (128, 256), (256, 256))
        )
    )
    plan = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans[0]
    assert plan.pre_wait_aux_safe
    source = bound.to_code(paired_config(bound))
    # The real wrapper plans carry independent pipeline depths and launch PDL.
    modules = device_sources(source)
    assert len(modules) == 2
    first, second = modules
    assert first.count("'ab_stage_count': 4") == 1
    assert second.count("'ab_stage_count': 8") == 1
    assert first.count("'c_stage_count': 4") == 1
    assert second.count("'c_stage_count': 2") == 2
    assert first.count("'use_pdl': True") == second.count("'use_pdl': True") == 1
    assert "tcgen05_fanout" not in first
    # Both producer and fanout epilogue depend on the earlier materialized call.
    assert first.count("griddepcontrol_wait") == 1
    assert second.count("griddepcontrol_wait") == 2
    assert second.index("griddepcontrol_wait") < second.index("consumer_wait")


@pytest.mark.parametrize(
    "key,value",
    (
        ("tcgen05_region_ab_stages", [4]),
        ("tcgen05_region_ab_stages", [True, 8]),
        ("tcgen05_region_ab_stages", [4, 17]),
        ("tcgen05_region_ab_stages", [16, 16]),
        ("tcgen05_region_c_stages", [4, 4]),
        ("tcgen05_region_c_stages", [4, 3]),
        ("tcgen05_materialized_pdl", 1),
        ("tcgen05_cta_group", "auto"),
        ("tcgen05_cluster_n", 2),
        ("tcgen05_aux_load_mode", "tma"),
    ),
)
def test_invalid_region_budgets_and_protocols_reject(key: str, value: object) -> None:
    bound = original_bound()
    raw = deepcopy(paired_config(bound).config)
    raw[key] = value
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(helion.Config.from_dict(raw))


def test_zero_overrides_preserve_scalar_normalization() -> None:
    bound = original_bound()
    raw = deepcopy(paired_config(bound).config)
    for key in ("tcgen05_region_ab_stages", "tcgen05_region_c_stages"):
        raw[key] = [0, 0]
    raw["tcgen05_materialized_pdl"] = False
    selected = bound.config_spec.normalized_config(helion.Config.from_dict(raw))
    assert "tcgen05_region_ab_stages" not in selected.config
    assert "tcgen05_region_c_stages" not in selected.config
    assert "tcgen05_materialized_pdl" not in selected.config


@pytest.mark.parametrize("fault", ("extra_store", "reduction", "unknown"))
def test_whole_region_smem_proof_rejects_other_allocations(fault: str) -> None:
    captured = []
    original = materialized_mma._paired_region_smem_facts

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if result is not None:
            captured.append((args, kwargs))
        return result

    with patch.object(materialized_mma, "_paired_region_smem_facts", observe):
        original_bound()
    ((args, kwargs),) = captured
    candidate, anchor, graphs = args
    outer = next(
        g.graph
        for g in graphs
        if any(n.target is memory_ops.store for n in g.graph.nodes)
    )
    store = next(n for n in outer.nodes if n.target is memory_ops.store)
    output = next(n for n in outer.nodes if n.op == "output")
    with outer.inserting_before(output):
        if fault == "extra_store":
            # Unrelated to the MMA: chain discovery still finds the old pair.
            inserted = outer.call_function(
                memory_ops.store, (store.args[0], store.args[1], 0, None)
            )
        elif fault == "reduction":
            inserted = outer.call_function(
                torch.ops.aten.sum.dim_IntList, (store.args[2], [0])
            )
        else:
            inserted = outer.call_function(
                torch.ops.aten.sort.default, (store.args[2],)
            )
    try:
        assert original(candidate, anchor, graphs, **kwargs) is None
    finally:
        outer.erase_node(inserted)
    assert original(candidate, anchor, graphs, **kwargs) is not None


@pytest.mark.parametrize("effect", ("store", "mutable_op"))
def test_argument_backed_writer_declines_pre_wait_alias_proof(effect: str) -> None:
    bound = original_bound()
    host = bound.host_function
    assert host is not None
    plan = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans[0]
    graphs = host.device_ir.graphs
    # Add an unrelated argument writer in the first root. Its storage may alias
    # the later auxiliary argument on another call despite distinct fake IDs.
    root = next(
        g.graph
        for g in graphs
        if any(n.target is memory_ops.store for n in g.graph.nodes)
    )
    output = next(n for n in root.nodes if n.op == "output")
    store = next(n for n in root.nodes if n.target is memory_ops.store)
    argument = next(n for g in graphs for n in g.graph.nodes if n.name == "a")
    with root.inserting_before(output):
        if effect == "store":
            inserted = root.call_function(
                memory_ops.store, (argument, store.args[1], 0, None)
            )
        else:
            inserted = root.call_function(
                torch.ops.aten.add_.Tensor, (argument, argument)
            )
    anchor = next(
        n
        for g in graphs
        for n in g.graph.nodes
        if n.target is torch.ops.aten.addmm.default and n.name == "acc"
    )
    try:
        with host:
            actual = prove_paired_fanout(bound.env, host, graphs, plan.stores, {anchor})
        assert actual is not None and not actual.pre_wait_aux_safe
    finally:
        root.erase_node(inserted)


def test_late_acquire_requires_shared_fanout_during_normalization() -> None:
    bound = original_bound()
    raw = deepcopy(paired_config(bound).config)
    raw["tcgen05_epilogue_fanout"] = "off"
    for key in (
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
        "tcgen05_materialized_pdl",
    ):
        raw.pop(key)
    raw["tcgen05_cluster_m"] = 1
    raw["tcgen05_cta_group"] = "auto"
    raw["block_sizes"] = [128] * 6
    with pytest.raises(InvalidConfig, match="before_store"):
        bound.config_spec.normalized_config(helion.Config.from_dict(raw))
    repaired = helion.Config.from_dict(raw)
    bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert "tcgen05_c_acquire_placement" not in repaired.config


@pytest.mark.parametrize("fault", ("pre_mismatch", "late_mismatch", "unsafe"))
def test_new_pair_lifetime_rejections(fault: str) -> None:
    first, second = _store_records()
    if fault == "pre_mismatch":
        first = replace(first, iteration=replace(first.iteration, pre_wait_aux=True))
    elif fault == "late_mismatch":
        first = replace(first, late_acquire="late()")
    else:
        unsafe = replace(first.plan, pre_wait_aux_safe=False)
        first = replace(
            first, plan=unsafe, iteration=replace(first.iteration, pre_wait_aux=True)
        )
        second = replace(
            second, plan=unsafe, iteration=replace(second.iteration, pre_wait_aux=True)
        )
    with pytest.raises(BackendUnsupported, match="lifetime"):
        combine_stores(first, second)


def test_late_acquire_is_after_final_read_and_before_shared_publication() -> None:
    first, second = _store_records()
    first = replace(
        first,
        acquire="",
        late_acquire="        late()\n",
        iteration=replace(first.iteration, pre_wait_aux=True),
    )
    second = replace(
        second,
        acquire="",
        late_acquire="        late()\n",
        iteration=replace(
            second.iteration, pre_wait_aux=True, auxiliary_loads="        auxiliary()\n"
        ),
    )
    combine_stores(first, second)
    source = ast.unparse(first.main)
    assert source.count("late()") == 1
    assert (
        source.index("auxiliary()") < source.index("wait()") < source.index("read0()")
    )
    assert (
        source.index("release()")
        < source.index("late()")
        < source.index("barrier0.arrive_and_wait()")
        < source.index("r2s0()")
    )
    assert source.index("tma0()") < source.index("tma1()") < source.index("commit()")
