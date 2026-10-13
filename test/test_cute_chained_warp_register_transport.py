from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_root_warp_stage import WarpOperandEmission
from helion._compiler.cute.chained_root_warp_stage import WarpOperandLayout
from helion._compiler.cute.chained_warp_register_transport import (
    plan_warp_register_transport,
)
from helion._compiler.cute.chained_warp_register_transport import relocate_operands
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.contraction_region import collect_contraction_region
from helion._compiler.device_ir import RootGraphInfo
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language.matmul_ops import dot


def _fixture(
    rows=64,
    columns=128,
    reduction=128,
    threads=128,
    *,
    transpose_seed=False,
    escape=False,
    alias_raw=False,
):
    graph = Graph()

    def call(target, args, shape, dtype=torch.bfloat16):
        node = graph.call_function(target, args)
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    def read(name, shape):
        host = call(_tracing_ops._host_tensor, (name,), shape)
        return call(memory_ops.load, (host, [None, None]), shape)

    q, h = read("q", (rows, reduction)), read("h", (reduction, columns))
    k, v = read("k", (rows, reduction)), read("v", (rows, columns))
    first = call(dot, (q, h, None, torch.float32), (rows, columns), torch.float32)
    kt = call(torch.ops.aten.t.default, (k,), (reduction, rows))
    middle = call(dot, (q, kt, None, torch.float32), (rows, rows), torch.float32)
    attention = call(
        torch.ops.prims.convert_element_type.default,
        (middle, torch.bfloat16),
        (rows, rows),
    )
    seed = call(torch.ops.aten.mul.Tensor, (first, 0.5), (rows, columns), torch.float32)
    if transpose_seed:
        seed = call(torch.ops.aten.t.default, (seed,), (columns, rows), torch.float32)
    last = call(
        dot, (attention, v, seed, torch.float32), (rows, columns), torch.float32
    )
    out = call(_tracing_ops._host_tensor, ("out",), (rows, columns), torch.float32)
    if alias_raw:
        out.meta["val"] = cast("Node", k.args[0]).meta["val"]
    value = (
        call(torch.ops.aten.add.Tensor, (last, first), (rows, columns), torch.float32)
        if escape
        else last
    )
    store = graph.call_function(memory_ops.store, (out, [None, None], value))
    graph.output((out,))
    region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    return chain.ChainedMatmulPlan(
        0,
        (first, middle, last),
        store,
        (),
        ((rows, columns, reduction), (rows, rows, reduction), (rows, columns, rows)),
        torch.bfloat16,
        threads,
        region=region,
    )


@pytest.mark.parametrize(
    "shape,threads", [((64, 128, 128), 128), ((32, 64, 64), 128), ((32, 32, 32), 64)]
)
def test_exact_seed_and_dead_result_lease_are_derived_from_graph(shape, threads):
    plan = _fixture(shape[0], shape[1], shape[2], threads)
    selected = plan_warp_register_transport(plan)
    assert selected is not None
    assert (
        selected.source,
        selected.destination,
        selected.lease_stage,
        selected.lease_role,
    ) == (0, 2, 1, "b")
    assert selected.reuse == ((1, "a"),)
    assert not any(r.name == "chain_0_c" for r in selected.workspace.regions)
    assert (
        selected.workspace.allocated_bytes
        <= plan_contraction_workspace(plan, row_padding=4).allocated_bytes
    )
    lease = selected.workspace.region("chain_raw_lease")
    assert lease.live_from == 0 and lease.live_until == 3
    for left in selected.workspace.regions:
        for right in selected.workspace.regions:
            if left is not right:
                assert not (
                    left.overlaps_lifetime(right) and left.overlaps_storage(right)
                )


@pytest.mark.parametrize("change", ["transpose", "escape", "alias", "partial", "loop"])
def test_unproved_seed_or_raw_ownership_rejects(change):
    plan = _fixture(
        rows=64,
        columns=64,
        transpose_seed=change == "transpose",
        escape=change == "escape",
        alias_raw=change == "alias",
    )
    if change == "partial":
        plan = replace(plan, shapes=((63, 64, 128), *plan.shapes[1:]))
    if change == "loop":
        plan = replace(plan, loop=cast("chain.ChainedLoopPlan", object()))
    assert plan_warp_register_transport(plan) is None


def _receipts(plan):
    operands, stages = {}, {}
    for stage, (rows, columns, reduction) in enumerate(plan.shapes):
        roles = []
        layouts = []
        for index, (role, shape) in enumerate(
            (("a", (rows, reduction)), ("b", (columns, reduction)))
        ):
            prefix = f"chain_{stage}_{role}"
            roles.append(
                (
                    f"{prefix}_ptr = chain_{role}_workspace",
                    f"{prefix} = cute.make_tensor({prefix}_ptr, original_layout)",
                    f"copy_{prefix}()",
                )
            )
            layouts.append(
                WarpOperandLayout(
                    plan.dots[stage].args[index],
                    role,
                    shape,
                    (shape[1] + 8, 1),
                    1,
                    torch.bfloat16,
                )
            )
        lines = tuple(line for role in roles for line in role)
        item = WarpOperandEmission(
            lines,
            (("a", 1), ("b", 1)),
            True,
            (("original_copy",),),
            tuple(layouts),
            None,
            role_lines=tuple(roles),
        )
        item = replace(item, _selection=item.facts())
        operands[stage] = item
        stages[stage] = (
            *lines,
            "cute.arch.cp_async_commit_group()",
            "cute.arch.cp_async_wait_group(0)",
            "cute.arch.sync_threads()",
            f"gemm_{stage}()",
            "cute.arch.sync_threads()",
        )
    return operands, stages


def test_actual_owned_copy_blocks_move_once_and_metadata_follows():
    plan = _fixture()
    selected = plan_warp_register_transport(plan)
    assert selected is not None
    operands, stages = _receipts(plan)
    original = [
        "setup()",
        *(line for unit in stages.values() for line in unit),
        "final_shared_drain()",
    ]
    result, actual = relocate_operands(selected, plan, original, operands, stages, {})
    assert result[0] == original[0] and result[-1] == original[-1]
    assert result.count("copy_chain_1_b()") == 1
    assert "copy_chain_1_b()" in actual[0] and "copy_chain_1_b()" not in actual[1]
    assert "copy_chain_1_a()" not in result and result.count("copy_chain_0_a()") == 1
    assert actual[2] == stages[2]
    assert result.count("cute.arch.sync_threads()") == original.count(
        "cute.arch.sync_threads()"
    )
    assert result.count("cute.arch.cp_async_commit_group()") == original.count(
        "cute.arch.cp_async_commit_group()"
    )
    assert result == [
        "setup()",
        *(line for unit in actual.values() for line in unit),
        "final_shared_drain()",
    ]


@pytest.mark.parametrize(
    "change", ["source", "layout", "receipt", "reader_join", "source_async", "program"]
)
def test_relocation_does_not_recertify_mutated_emission(change):
    plan = _fixture()
    selected = plan_warp_register_transport(plan)
    assert selected is not None
    operands, stages = _receipts(plan)
    lines = [line for unit in stages.values() for line in unit]
    if change == "source":
        plan.dots[0].args = (*plan.dots[0].args[:2], 1, torch.float32)
    elif change == "layout":
        operands[1] = replace(
            operands[1],
            layouts=(
                replace(operands[1].layouts[0], stride=(1, 136)),
                operands[1].layouts[1],
            ),
        )
    elif change == "receipt":
        operands[1] = replace(operands[1], role_lines=())
    elif change == "source_async":
        changed = replace(operands[0], asynchronous=False)
        operands[0] = replace(changed, _selection=changed.facts())
    elif change == "reader_join":
        stages[1] = stages[1][:-1]
    else:
        lines.remove("copy_chain_1_b()")
    with pytest.raises(chain._UnsupportedChain):
        relocate_operands(selected, plan, lines, operands, stages, {})


def _normalize_cache(config, *, enabled=True, fix=False):
    spec = SimpleNamespace(
        cute_async_load_pipeline_enabled=False,
        cute_chained_matmul_search_enabled=enabled,
        cute_chained_loop_search_enabled=False,
    )
    ConfigSpec._normalize_cute_async_load_pipeline(
        cast("ConfigSpec", spec), config, fix_invalid=fix
    )
    return config


@pytest.mark.parametrize(
    "schedule",
    (
        "cp_async",
        "cp_async_register",
        "cp_async_register_reuse",
        "cp_async_register_reuse_scan",
    ),
)
def test_cache_admission_preserves_omission_and_all_explicit_legal_values(schedule):
    config = {"cute_chained_mma_schedule": schedule}
    assert _normalize_cache(dict(config)) == config
    for value in ("cg", "ca"):
        requested = {**config, "cute_async_load_cache": value}
        assert _normalize_cache(dict(requested)) == (
            requested if value == "cg" else config
        )


@pytest.mark.parametrize(
    "change", ("schedule", "undiscovered", "foreign_key", "invalid")
)
def test_cache_admission_does_not_enable_other_pipeline_fields(change):
    config: dict[str, object] = {
        "cute_chained_mma_schedule": "cp_async",
        "cute_async_load_cache": "cg",
    }
    if change == "schedule":
        config["cute_chained_mma_schedule"] = "coalesced"
    elif change == "foreign_key":
        config["cute_async_load_stages"] = 3
    elif change == "invalid":
        config["cute_async_load_cache"] = True
    with pytest.raises(InvalidConfig):
        _normalize_cache(config, enabled=change != "undiscovered")


def test_register_mode_is_explicit_and_keeps_original_final_shared_region():
    from helion._compiler.cute.chained_warp_result_transport import (
        plan_warp_result_transport,
    )

    plan = _fixture()
    assert plan_warp_result_transport(plan, "packed_register", "coalesced") is None
    selected = plan_warp_result_transport(plan, "packed_register", "cp_async")
    assert selected is not None and selected.terminal is None
    assert selected.workspace is not None
    assert selected.workspace.region("chain_2_c").byte_size == 64 * 132 * 4
    bound = replace(
        plan, warp_workspace=selected.workspace, warp_result_transport=selected
    )
    selected.check(bound)
    with pytest.raises(chain._UnsupportedChain):
        selected.check(replace(bound, warp_workspace=None))


def test_raw_storage_ownership_change_invalidates_selected_lease():
    plan = _fixture()
    selected = plan_warp_register_transport(plan)
    assert selected is not None
    raw = cast("Node", plan.dots[0].args[0])
    host = cast("Node", raw.args[0])
    host.meta["val"] = host.meta["val"].clone()
    with pytest.raises(chain._UnsupportedChain, match="graph changed"):
        selected.check(plan)


@pytest.mark.parametrize(
    "schedule,mode,cache",
    [
        ("coalesced", "packed", None),
        ("cp_async", "legacy", None),
        ("cp_async", "packed_register", "cg"),
        ("cp_async", "packed_register", "ca"),
    ],
)
def test_actual_discovery_keeps_old_and_new_flat_config_coordinates(
    schedule, mode, cache
):
    from ._cute_aux import _cpu_codegen
    from .test_cute_chained_warp_workspace import _args
    from .test_cute_chained_warp_workspace import _long_chain
    import helion

    values = {
        "num_warps": 4,
        "cute_chained_mma_schedule": schedule,
        "cute_chained_warp_result_transport": mode,
    }
    if cache is not None:
        values["cute_async_load_cache"] = cache
    with _cpu_codegen():
        bound = _long_chain._bind_isolated(_args(stages=3, size=64))
        spec = bound.config_spec
        config = spec.normalized_config(helion.Config.from_dict(values))
        generation = spec.create_config_generation()
        assert generation.unflatten(generation.flatten(config)) == config
        assert generation.strict_config_pair(config)[1] == config
        assert config.config.get("cute_async_load_cache") == (
            "cg" if cache == "cg" else None
        )
