from __future__ import annotations

import contextlib
import dataclasses
import gc
import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch
import weakref

from benchmarks.cute.kda_prefill_kernels import BT
from benchmarks.cute.kda_prefill_kernels import DK
from benchmarks.cute.kda_prefill_kernels import DV
from benchmarks.cute.kda_prefill_kernels import KDA_RECURRENCE_CONFIG as _CONFIG
from benchmarks.cute.kda_prefill_kernels import (
    kda_chunk_recurrence as _bt16_resident_chain,
)
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute.chunk_recurrence import _select_sm100_dv_partitions
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.cute.launcher import _chunk_recurrence_tensor_map_specs

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable


def _fake_inputs(
    *,
    tokens: int = 128,
    heads: int = 12,
    sequences: int = 2,
    chunks: int = 8,
    packed_workspace: bool = True,
) -> tuple[torch.Tensor, ...]:
    rows = chunks * BT
    with FakeTensorMode():
        vector_bytes = heads * rows * DK * 2
        aq_bytes = heads * chunks * BT * BT * 2
        gt_bytes = heads * chunks * DK * 4
        if packed_workspace:
            workspace = torch.empty(
                (3 * vector_bytes + aq_bytes + gt_bytes,),
                device=DEVICE,
                dtype=torch.uint8,
            )
            kd = workspace[:vector_bytes].view(torch.bfloat16).view(1, heads, rows, DK)
            qd = (
                workspace[vector_bytes : 2 * vector_bytes]
                .view(torch.bfloat16)
                .view(1, heads, rows, DK)
            )
            ak = (
                workspace[2 * vector_bytes : 3 * vector_bytes]
                .view(torch.bfloat16)
                .view(1, heads, rows, DK)
            )
            aq = (
                workspace[3 * vector_bytes : 3 * vector_bytes + aq_bytes]
                .view(torch.bfloat16)
                .view(1, heads, chunks, BT * BT)
            )
            g_total = (
                workspace[3 * vector_bytes + aq_bytes :]
                .view(torch.float32)
                .view(1, heads, chunks, DK)
            )
        else:
            kd = torch.empty((1, heads, rows, DK), device=DEVICE, dtype=torch.bfloat16)
            qd = torch.empty_like(kd)
            ak = torch.empty_like(kd)
            aq = torch.empty(
                (1, heads, chunks, BT * BT),
                device=DEVICE,
                dtype=torch.bfloat16,
            )
            g_total = torch.empty(
                (1, heads, chunks, DK), device=DEVICE, dtype=torch.float32
            )
        values = torch.empty(
            (1, tokens, heads, DV), device=DEVICE, dtype=torch.bfloat16
        )
        output = torch.empty_like(values)
        state = torch.empty(
            (sequences, heads, DV, DK), device=DEVICE, dtype=torch.bfloat16
        )
        cu_seqlens = torch.empty((sequences + 1,), device=DEVICE, dtype=torch.int32)
        cu_chunks = torch.empty_like(cu_seqlens)
    return kd, qd, ak, aq, g_total, values, output, state, cu_seqlens, cu_chunks


def _code(
    mutate: Callable[[object], None] | None = None,
    *,
    tokens: int = 128,
    heads: int = 12,
    sequences: int = 2,
    chunks: int = 8,
    capability: tuple[int, int] = (10, 3),
    num_sm: int = 152,
    dv_partitions: int | None = None,
    register_cap: int | None = None,
    packed_workspace: bool = True,
) -> str:
    with (
        patch(
            "helion.runtime.kernel.target_device_capability",
            return_value=capability,
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=capability,
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch(
            "helion.language.loops._supports_warp_specialize",
            return_value=capability >= (10, 0),
        ),
        patch(
            "helion._compat._supports_tensor_descriptor",
            return_value=capability >= (9, 0),
        ),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.runtime.get_num_sm", return_value=num_sm),
    ):
        bound = _bt16_resident_chain._bind_isolated(
            (
                *_fake_inputs(
                    tokens=tokens,
                    heads=heads,
                    sequences=sequences,
                    chunks=chunks,
                    packed_workspace=packed_workspace,
                ),
                128**-0.5,
            )
        )
        if mutate is not None:
            mutate(bound.host_function.device_ir)
        bound.config_spec.target_device_capability = capability
        bound.config_spec.num_sm = num_sm
        config_overrides: dict[str, object] = {}
        if dv_partitions is not None:
            config_overrides["cute_chunk_recurrence_dv_partitions"] = dv_partitions
        if register_cap is not None:
            config_overrides["cute_chunk_recurrence_register_cap"] = register_cap
        config = helion.Config.from_dict({**_CONFIG.config, **config_overrides})
        return bound.to_triton_code(config)


def _captured_plan(
    *,
    mutate: Callable[[object], None] | None = None,
    capability: tuple[int, int] = (10, 3),
    packed_workspace: bool = True,
) -> object:
    from helion._compiler.cute import chunk_recurrence

    captured: list[object] = []
    original = chunk_recurrence._plan_chunk_recurrence

    def capture(graphs: object, tile_strategy: object) -> object:
        result = original(graphs, tile_strategy)  # type: ignore[arg-type]
        captured.append(result)
        return result

    with (
        patch.object(chunk_recurrence, "_plan_chunk_recurrence", side_effect=capture),
        contextlib.suppress(exc.BackendUnsupported),
    ):
        _code(
            mutate=mutate,
            capability=capability,
            packed_workspace=packed_workspace,
        )
    assert len(captured) == 1
    return captured[0]


def _config_spec(
    *, tokens: int, heads: int, sequences: int, chunks: int, num_sm: int = 152
) -> object:
    with (
        patch(
            "helion.runtime.kernel.target_device_capability",
            return_value=(10, 3),
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.runtime.get_num_sm", return_value=num_sm),
    ):
        bound = _bt16_resident_chain._bind_isolated(
            (
                *_fake_inputs(
                    tokens=tokens,
                    heads=heads,
                    sequences=sequences,
                    chunks=chunks,
                ),
                128**-0.5,
            )
        )
        bound.config_spec.num_sm = num_sm
        return bound.config_spec


def test_bt16_resident_chain_codegen() -> None:
    code = _code()
    assert "from helion._compiler.cute.chunk_recurrence_sm100 import" in code
    assert "_prototype_bt16_chain" not in code
    assert "'kind': 'chunk_recurrence_sm100'" in code
    assert "'threads': 512" in code
    assert "'smem_bytes': 138240" in code
    assert "'input_stages': 8" in code
    assert "'tma_stages': 6" in code
    assert "'factor_tma_value_splits': 2" in code
    assert "'output_acc_stages': 2" in code
    assert "'output_smem_stages': 7" in code
    assert "'output_store_wait_groups': 6" in code
    assert "'tmem_cols': 512" in code
    assert "'workspace_layout_version': 2" in code
    assert "'factor_key_xor': 8" in code
    assert "'outputs_scaled': False" in code
    assert "'scale_idx': 10" in code
    assert "--maxrregcount=" not in code
    assert "cute.gemm(" not in code


def test_tmem_dv2_exposes_only_the_matched_state_contract() -> None:
    from helion._compiler.cute import chunk_recurrence_sm100 as device

    source = inspect.getsource(device.kernel_chain_dv2)
    kernel_parameters = inspect.signature(device.kernel_chain_dv2).parameters
    host_parameters = inspect.signature(device.host_chain_dv2).parameters
    for removed_parameter in (
        "state_indices",
        "initial_state",
        "final_state",
        "state_ckpt",
        "cu_ckpts",
        "checkpoint_stride_chunks",
    ):
        assert removed_parameter not in kernel_parameters
        assert removed_parameter not in host_parameters
    assert "checkpoint_read_done" not in source


@pytest.mark.parametrize("guard", ("_linear_offsets_fit_i32", "_xyz_grid_fits"))
def test_bt16_recurrence_unsafe_index_or_launch_geometry_fails_closed(
    guard: str,
) -> None:
    with patch(f"helion._compiler.cute.chunk_recurrence.{guard}", return_value=False):
        plan = _captured_plan()
    assert plan is None


@pytest.mark.parametrize(
    ("collision", "expected"),
    (
        (
            "_helion_sm100_warp_dv4_host",
            "import _recurrence_entry as _helion_sm100_warp_dv4_host_1",
        ),
        (
            "_chunk_recurrence_desc_factor",
            "'desc_args': ('_chunk_recurrence_desc_factor_1',",
        ),
    ),
)
def test_bt16_recurrence_generated_names_do_not_shadow_tensor_arguments(
    collision: str,
    expected: str,
) -> None:
    from helion._compiler.cute import chunk_recurrence

    original = chunk_recurrence._load_ref

    def colliding_name(node: object) -> object:
        ref = original(node)
        if ref is not None and ref.name == "value_rows":
            return dataclasses.replace(ref, name=collision)
        return ref

    with patch.object(chunk_recurrence, "_load_ref", side_effect=colliding_name):
        code = _code(dv_partitions=4)
    assert expected in code


def _recurrence_loop_nodes(device_ir: object) -> list[torch.fx.Node]:
    from helion._compiler.device_ir import ForLoopGraphInfo

    loop = next(
        graph
        for graph in device_ir.graphs
        if isinstance(graph, ForLoopGraphInfo)  # type: ignore[attr-defined]
    )
    return list(loop.graph.nodes)


def _reverse_residual_subtraction(device_ir: object) -> None:
    nodes = _recurrence_loop_nodes(device_ir)
    subtraction = next(
        node
        for node in nodes
        if node.op == "call_function" and node.target is torch.ops.aten.sub.Tensor
    )
    subtraction.args = (subtraction.args[1], subtraction.args[0])


def _change_factor_feature_xor(device_ir: object) -> None:
    nodes = _recurrence_loop_nodes(device_ir)
    xor = next(
        node
        for node in nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.bitwise_xor.Scalar
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target is torch.ops.prims.iota.default
        and node.args[1] == 8
    )
    xor.args = (xor.args[0], 4)


def _change_ak_row_xor(device_ir: object) -> None:
    nodes = _recurrence_loop_nodes(device_ir)
    xors = [
        node
        for node in nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.bitwise_xor.Scalar
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target.__name__ == "tile_index"
    ]
    assert len(xors) == 1
    xors[0].args = (xors[0].args[0], 4)


def _change_aq_swizzle_constant(device_ir: object) -> None:
    nodes = _recurrence_loop_nodes(device_ir)
    shift = next(
        node
        for node in nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.__rshift__.Scalar
    )
    shift.args = (shift.args[0], 6)


def _swap_update_operands(device_ir: object) -> None:
    from helion.language.matmul_ops import dot

    dots = [
        node
        for node in _recurrence_loop_nodes(device_ir)
        if node.op == "call_function" and node.target is dot
    ]
    update = dots[-1]
    update.args = (update.args[1], update.args[0], *update.args[2:])


def _change_residual_rounding(device_ir: object) -> None:
    nodes = _recurrence_loop_nodes(device_ir)
    subtraction = next(
        node
        for node in nodes
        if node.op == "call_function" and node.target is torch.ops.aten.sub.Tensor
    )
    rounding = next(
        node
        for node in subtraction.users
        if node.target is torch.ops.prims.convert_element_type.default
    )
    rounding.args = (rounding.args[0], torch.float16)


def _remove_output_mask(device_ir: object) -> None:
    from helion.language import memory_ops

    store = next(
        node
        for node in _recurrence_loop_nodes(device_ir)
        if node.op == "call_function" and node.target is memory_ops.store
    )
    store.args = (*store.args[:3], None)


def _reverse_loop_bound(device_ir: object) -> None:
    from helion._compiler.device_ir import RootGraphInfo
    from helion.language._tracing_ops import _for_loop

    root = next(
        graph
        for graph in device_ir.graphs
        if isinstance(graph, RootGraphInfo)  # type: ignore[attr-defined]
    )
    loop_call = next(
        node
        for node in root.graph.nodes
        if node.op == "call_function" and node.target is _for_loop
    )
    extent = loop_call.args[2][0]
    extent.args = (extent.args[1], extent.args[0])


def _replace_state_head_with_sequence_coordinate(device_ir: object) -> None:
    from helion._compiler.device_ir import RootGraphInfo
    from helion.language import memory_ops

    root = next(
        graph
        for graph in device_ir.graphs
        if isinstance(graph, RootGraphInfo)  # type: ignore[attr-defined]
    )
    state_accesses = [
        node
        for node in root.graph.nodes
        if node.op == "call_function"
        and node.target in (memory_ops.load, memory_ops.store)
        and node.args[0].args[0] == "state"
    ]
    assert len(state_accesses) == 2
    sequence_coordinate = state_accesses[0].args[1][0]
    for node in state_accesses:
        indices = list(node.args[1])
        indices[1] = sequence_coordinate
        node.args = (node.args[0], indices, *node.args[2:])


def _change_value_size_dimension(device_ir: object) -> None:
    size = next(
        node
        for node in _recurrence_loop_nodes(device_ir)
        if node.op == "call_function" and node.target is torch.ops.aten.sym_size.int
    )
    size.args = (size.args[0], 1)


def _replace_state_value_with_head_coordinate(device_ir: object) -> None:
    from helion._compiler.device_ir import RootGraphInfo
    from helion.language import memory_ops

    root = next(
        graph
        for graph in device_ir.graphs
        if isinstance(graph, RootGraphInfo)  # type: ignore[attr-defined]
    )
    state_accesses = [
        node
        for node in root.graph.nodes
        if node.op == "call_function"
        and node.target in (memory_ops.load, memory_ops.store)
        and node.args[0].args[0] == "state"
    ]
    assert len(state_accesses) == 2
    head_coordinate = state_accesses[0].args[1][1]
    for node in state_accesses:
        indices = list(node.args[1])
        indices[2] = head_coordinate
        node.args = (node.args[0], indices, *node.args[2:])


def _disconnect_token_tile_id_from_lane(device_ir: object) -> None:
    from helion.language.tile_ops import tile_id
    from helion.language.tile_ops import tile_index

    nodes = _recurrence_loop_nodes(device_ir)
    tile_ids = [node for node in nodes if node.target is tile_id]
    tile_indices = [node for node in nodes if node.target is tile_index]
    assert len(tile_ids) == 2
    assert len(tile_indices) == 4
    head_block_size = tile_ids[1].args[0]
    assert isinstance(head_block_size, torch.fx.Node)
    tile_ids[0].prepend(head_block_size)
    tile_ids[0].args = (head_block_size,)


def _permute_root_task_axes(device_ir: object) -> None:
    axes = list(device_ir.task_families[0].axes)  # type: ignore[attr-defined]
    axes[0], axes[1] = axes[1], axes[0]
    device_ir.task_families[0] = dataclasses.replace(  # type: ignore[attr-defined]
        device_ir.task_families[0],  # type: ignore[attr-defined]
        axes=tuple(axes),
    )


def _make_root_task_origin_noncanonical(device_ir: object) -> None:
    axes = list(device_ir.task_families[0].axes)  # type: ignore[attr-defined]
    axes[0] = dataclasses.replace(axes[0], canonical_origin=False)
    device_ir.task_families[0] = dataclasses.replace(  # type: ignore[attr-defined]
        device_ir.task_families[0],  # type: ignore[attr-defined]
        axes=tuple(axes),
    )


def _shorten_root_task_extent(device_ir: object) -> None:
    axes = list(device_ir.task_families[0].axes)  # type: ignore[attr-defined]
    extent = axes[0].extent
    assert extent is not None and not isinstance(extent, str)
    axes[0] = dataclasses.replace(axes[0], extent=extent - 1)
    device_ir.task_families[0] = dataclasses.replace(  # type: ignore[attr-defined]
        device_ir.task_families[0],  # type: ignore[attr-defined]
        axes=tuple(axes),
    )


@pytest.mark.parametrize(
    "mutate",
    (
        _reverse_residual_subtraction,
        _change_factor_feature_xor,
        _change_ak_row_xor,
        _change_aq_swizzle_constant,
        _swap_update_operands,
        _change_residual_rounding,
        _remove_output_mask,
        _reverse_loop_bound,
        _replace_state_head_with_sequence_coordinate,
        _change_value_size_dimension,
        _replace_state_value_with_head_coordinate,
        _disconnect_token_tile_id_from_lane,
        _permute_root_task_axes,
        _make_root_task_origin_noncanonical,
        _shorten_root_task_extent,
    ),
)
def test_bt16_recurrence_rejects_semantic_mutations(
    mutate: Callable[[object], None],
) -> None:
    assert _captured_plan(mutate=mutate) is None


def test_bt16_recurrence_rejects_sm80() -> None:
    assert _captured_plan(capability=(8, 0)) is None


def test_bt16_recurrence_split_workspace_falls_back_during_planning() -> None:
    assert _captured_plan(packed_workspace=False) is None


@pytest.mark.parametrize(
    ("total_tokens", "total_chunks"),
    ((0, 8), (128, 0)),
)
def test_bt16_recurrence_empty_geometry_falls_back_before_schedule_policy(
    total_tokens: int, total_chunks: int
) -> None:
    from helion._compiler.cute import chunk_recurrence

    original_match = chunk_recurrence._match_chunk_recurrence_graphs

    def match_all_empty(graphs: object) -> object:
        match = original_match(graphs)  # type: ignore[arg-type]
        assert match is not None
        return dataclasses.replace(
            match,
            total_tokens=total_tokens,
            total_chunks=total_chunks,
        )

    with (
        patch.object(
            chunk_recurrence,
            "_match_chunk_recurrence_graphs",
            side_effect=match_all_empty,
        ),
        patch.object(chunk_recurrence, "_select_sm100_dv_partitions") as select,
    ):
        assert _captured_plan() is None

    select.assert_not_called()


@pytest.mark.parametrize(
    ("total_tokens", "total_chunks"),
    ((0, 8), (128, 0)),
)
def test_bt16_recurrence_empty_geometry_disables_schedule_search(
    total_tokens: int, total_chunks: int
) -> None:
    from helion._compiler.cute import chunk_recurrence

    original_match = chunk_recurrence._match_chunk_recurrence_graphs

    def match_all_empty(graphs: object) -> object:
        match = original_match(graphs)  # type: ignore[arg-type]
        assert match is not None
        return dataclasses.replace(
            match,
            total_tokens=total_tokens,
            total_chunks=total_chunks,
        )

    with (
        patch.object(
            chunk_recurrence,
            "_match_chunk_recurrence_graphs",
            side_effect=match_all_empty,
        ),
        patch.object(chunk_recurrence, "_select_sm100_dv_partitions") as select,
    ):
        _config_spec(tokens=128, heads=12, sequences=2, chunks=8)

    select.assert_not_called()


@pytest.mark.parametrize("register_cap", (72, 76, 80))
def test_bt16_resident_chain_register_cap_reaches_compile_options(
    register_cap: int,
) -> None:
    code = _code(dv_partitions=4, register_cap=register_cap)
    assert f"--maxrregcount={register_cap}" in code


def test_sm100_dv_partition_policy_is_geometry_derived() -> None:
    assert (
        _select_sm100_dv_partitions(total_chunks=32, sequences=1, heads=12, num_sm=152)
        == 4
    )
    assert (
        _select_sm100_dv_partitions(total_chunks=515, sequences=6, heads=12, num_sm=152)
        == 2
    )


def test_bt16_recurrence_autotune_seeds_both_dv_schedules() -> None:
    fixed = _config_spec(tokens=512, heads=12, sequences=1, chunks=32)
    packed = _config_spec(tokens=8192, heads=12, sequences=6, chunks=515)
    assert fixed.cute_chunk_recurrence_dv_partitions.choices == (4, 2)
    assert packed.cute_chunk_recurrence_dv_partitions.choices == (2, 4)
    assert fixed.cute_chunk_recurrence_register_cap.choices == (None, 72, 76, 80)
    assert packed.cute_chunk_recurrence_register_cap.choices == (None, 72, 76, 80)
    assert fixed.compiler_default_config.config == {
        "cute_chunk_recurrence_dv_partitions": 4,
        "cute_chunk_recurrence_register_cap": 72,
    }
    assert packed.compiler_default_config.config == {
        "cute_chunk_recurrence_dv_partitions": 2,
        "cute_chunk_recurrence_register_cap": None,
    }
    recurrence_keys = {
        "cute_chunk_recurrence_dv_partitions",
        "cute_chunk_recurrence_register_cap",
    }
    fixed_recurrence_seeds = [
        seed
        for seed in fixed.compiler_seed_configs
        if set(seed.config) == recurrence_keys
    ]
    packed_recurrence_seeds = [
        seed
        for seed in packed.compiler_seed_configs
        if set(seed.config) == recurrence_keys
    ]
    assert all(
        seed.config["cute_chunk_recurrence_dv_partitions"] != 2
        or seed.config["cute_chunk_recurrence_register_cap"] is None
        for seed in (*fixed_recurrence_seeds, *packed_recurrence_seeds)
    )
    assert {
        seed.config["cute_chunk_recurrence_register_cap"]
        for seed in fixed_recurrence_seeds
        if seed.config["cute_chunk_recurrence_dv_partitions"] == 4
    } == {72, 76, 80, None}
    fixed_seed_schedules = [
        seed.config["cute_chunk_recurrence_dv_partitions"]
        for seed in fixed_recurrence_seeds
    ]
    packed_seed_schedules = [
        seed.config["cute_chunk_recurrence_dv_partitions"]
        for seed in packed_recurrence_seeds
    ]
    assert list(dict.fromkeys(fixed_seed_schedules)) == [4, 2]
    assert list(dict.fromkeys(packed_seed_schedules)) == [2, 4]
    fixed_seed_caps = [
        seed.config["cute_chunk_recurrence_register_cap"]
        for seed in fixed_recurrence_seeds
    ]
    assert list(dict.fromkeys(fixed_seed_caps)) == [72, 76, 80, None]
    generated_seed_schedules = [
        config.config["cute_chunk_recurrence_dv_partitions"]
        for _flat, config in ConfigGeneration(fixed).seed_flat_config_pairs()
    ]
    assert list(dict.fromkeys(generated_seed_schedules)) == [4, 2]
    assert [
        dimension.values
        for dimension in packed.iter_search_dimensions()
        if dimension.name == "cute_chunk_recurrence_dv_partitions"
    ] == [[2, 4]]
    assert [
        dimension.values
        for dimension in packed.iter_search_dimensions()
        if dimension.name == "cute_chunk_recurrence_register_cap"
    ] == [[None, 72, 76, 80]]
    assert (
        _select_sm100_dv_partitions(total_chunks=33, sequences=1, heads=12, num_sm=152)
        == 2
    )
    assert (
        _select_sm100_dv_partitions(total_chunks=32, sequences=1, heads=40, num_sm=152)
        == 2
    )


def test_bt16_recurrence_dv_partition_config_is_strictly_scoped() -> None:
    spec = _config_spec(tokens=512, heads=12, sequences=1, chunks=32)
    with pytest.raises(
        exc.InvalidConfig,
        match="cute_chunk_recurrence_dv_partitions must be one of",
    ):
        spec.normalized_config(
            helion.Config.from_dict(
                {
                    **_CONFIG.config,
                    "cute_chunk_recurrence_dv_partitions": 3,
                }
            )
        )

    spec.cute_chunk_recurrence_dv_partitions = None
    with pytest.raises(
        exc.InvalidConfig,
        match="available only for matched BT16 chunk-recurrence kernels",
    ):
        spec.normalized_config(
            helion.Config.from_dict(
                {
                    **_CONFIG.config,
                    "cute_chunk_recurrence_dv_partitions": 4,
                }
            )
        )

    repaired = helion.Config.from_dict(
        {
            **_CONFIG.config,
            "cute_chunk_recurrence_dv_partitions": 4,
        }
    )
    spec.normalize(repaired, _fix_invalid=True)
    assert "cute_chunk_recurrence_dv_partitions" not in repaired.config


def test_bt16_recurrence_register_cap_config_is_strictly_scoped() -> None:
    spec = _config_spec(tokens=512, heads=12, sequences=1, chunks=32)
    with pytest.raises(
        exc.InvalidConfig,
        match="cute_chunk_recurrence_register_cap must be one of",
    ):
        spec.normalized_config(
            helion.Config.from_dict(
                {
                    **_CONFIG.config,
                    "cute_chunk_recurrence_register_cap": 64,
                }
            )
        )

    spec.cute_chunk_recurrence_register_cap = None
    with pytest.raises(
        exc.InvalidConfig,
        match="available only for matched BT16 chunk-recurrence kernels",
    ):
        spec.normalized_config(
            helion.Config.from_dict(
                {
                    **_CONFIG.config,
                    "cute_chunk_recurrence_register_cap": 72,
                }
            )
        )

    repaired = helion.Config.from_dict(
        {
            **_CONFIG.config,
            "cute_chunk_recurrence_register_cap": 72,
        }
    )
    spec.normalize(repaired, _fix_invalid=True)
    assert "cute_chunk_recurrence_register_cap" not in repaired.config


def test_bt16_recurrence_rejects_register_caps_for_tmem_dv2() -> None:
    spec = _config_spec(tokens=8192, heads=12, sequences=6, chunks=515)
    unsafe = helion.Config.from_dict(
        {
            **_CONFIG.config,
            "cute_chunk_recurrence_dv_partitions": 2,
            "cute_chunk_recurrence_register_cap": 72,
        }
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="must be None.*TMEM schedule dynamically reallocates registers",
    ):
        spec.normalized_config(unsafe)

    spec.normalize(unsafe, _fix_invalid=True)
    assert unsafe["cute_chunk_recurrence_dv_partitions"] == 2
    assert unsafe["cute_chunk_recurrence_register_cap"] is None


def test_bt16_resident_chain_selects_warp_dv4_when_short_and_underfilled() -> None:
    code = _code(tokens=512, sequences=1, chunks=32, num_sm=152)
    assert "chunk_recurrence_warp_dv4" in code
    assert "chunk_recurrence_dv4_sm100 import _recurrence_entry" in code
    assert "'threads': 192" in code
    assert "'smem_bytes': 97536" in code
    assert "'dv_partitions': 4" in code
    assert "'workspace_layout_version': 2" in code
    assert "'factor_key_xor': 8" in code
    assert "'outputs_scaled': False" in code

    forced_dv2 = _code(
        tokens=512,
        sequences=1,
        chunks=32,
        num_sm=152,
        dv_partitions=2,
    )
    assert "'kind': 'chunk_recurrence_sm100'" in forced_dv2
    assert "chunk_recurrence_warp_dv4" not in forced_dv2


def test_bt16_resident_chain_keeps_tmem_dv2_for_packed_geometry() -> None:
    code = _code(tokens=8192, sequences=6, chunks=515, num_sm=152)
    assert "'kind': 'chunk_recurrence_sm100'" in code
    assert "'threads': 512" in code
    assert "'dv_partitions': 2" in code
    assert "chunk_recurrence_warp_dv4" not in code

    forced_dv4 = _code(
        tokens=8192,
        sequences=6,
        chunks=515,
        num_sm=152,
        dv_partitions=4,
    )
    assert "chunk_recurrence_warp_dv4" in forced_dv4
    assert "'dv_partitions': 4" in forced_dv4


def test_dv4_uses_semantic_length_when_chunk_metadata_disagrees() -> None:
    from helion._compiler.cute.chunk_recurrence_dv4_sm100 import emit_bt16_recurrence

    # Adversarial metadata: the chunk offsets claim one chunk for 17 tokens.
    # The Helion carrier and DV2 execute ceil(17 / 16) == 2 chunks, so DV4 must
    # derive the same bound from sequence lengths instead of trusting the delta.
    token_start, token_end = 0, 17
    chunk_start, chunk_end = 0, 1
    assert (token_end - token_start + BT - 1) // BT == 2
    assert chunk_end - chunk_start == 1

    source = inspect.getsource(emit_bt16_recurrence)
    assert source.startswith("@cute.kernel\n")
    assert "num_chunks = cute.ceil_div(token_end - token_start, BT)" in source
    assert "gcu_chunks[seq + 1]" not in source


def _runtime_plan(
    *, tokens: int = 128, heads: int = 12, sequences: int = 2, chunks: int = 8
) -> dict[str, object]:
    plan: dict[str, object] = {
        "kind": "chunk_recurrence_warp_dv4",
        "workspace_layout_version": 2,
        "outputs_scaled": False,
        "factor_key_xor": 8,
        "chunk_size": BT,
        "key_size": DK,
        "value_size": DV,
        "total_tokens": tokens,
        "total_chunks": chunks,
        "heads": heads,
        "sequences": sequences,
    }
    for index, name in enumerate(
        (
            "kd",
            "qd",
            "ak",
            "aq",
            "gt",
            "v",
            "out",
            "state",
            "cu_seqlens",
            "cu_chunks",
        )
    ):
        plan[f"{name}_idx"] = index
    return plan


def _runtime_view_args() -> tuple[torch.Tensor, ...]:
    args = _fake_inputs()
    return (
        args[0].view(-1, DK),
        args[1].view(-1, DK),
        args[2].view(-1, DK),
        args[3].view(-1, BT * BT),
        args[4].view(-1, DK),
        args[5].view(-1, DV),
        args[6].view(-1, DV),
        args[7],
        args[8],
        args[9],
    )


def _fake_pointer_patches(
    args: tuple[torch.Tensor, ...],
) -> tuple[object, object]:
    from torch._subclasses.fake_tensor import FakeTensor

    storage_bases = {
        storage._cdata: 0x10_0000 + index * 0x100_0000
        for index, storage in enumerate({tensor.untyped_storage() for tensor in args})
    }

    def storage_data_ptr(storage: object) -> int:
        return storage_bases[storage._cdata]  # type: ignore[attr-defined]

    def tensor_data_ptr(tensor: FakeTensor) -> int:
        return (
            storage_bases[tensor.untyped_storage()._cdata]
            + tensor.storage_offset() * tensor.element_size()
        )

    return (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
    )


def test_chunk_recurrence_runtime_validates_packed_workspace() -> None:
    args = _runtime_view_args()
    storage_patch, tensor_patch = _fake_pointer_patches(args)
    with storage_patch, tensor_patch:
        specs = _chunk_recurrence_tensor_map_specs(_runtime_plan(), args)
    assert specs["factor"].global_dim == (128, 128, 36)
    assert specs["aq"].global_dim == (256, 8, 12)
    assert specs["gt"].global_dim == (128, 8, 12)
    assert specs["v"].global_dim == (128, 128, 12)
    assert specs["out"].global_dim == (128, 128, 12)
    assert specs["state_in"].global_dim == (64, 256, 24)
    assert specs["state_in"] == specs["state_out"]


def test_chunk_recurrence_runtime_rejects_legacy_abi1() -> None:
    args = _runtime_view_args()
    plan = _runtime_plan()
    plan.update(
        {
            "kind": "chunk_recurrence_tma",
            "workspace_layout_version": 1,
            "outputs_scaled": True,
            "factor_key_xor": 0,
        }
    )
    with pytest.raises(
        exc.BackendUnsupported, match="unsupported chunk-recurrence ABI"
    ):
        _chunk_recurrence_tensor_map_specs(plan, args, validate_only=True)


def test_chunk_recurrence_runtime_builds_dv4_tensor_map_geometry() -> None:
    args = _runtime_view_args()
    plan = _runtime_plan()
    plan.update(
        {
            "kind": "chunk_recurrence_warp_dv4",
            "workspace_layout_version": 2,
            "outputs_scaled": False,
            "factor_key_xor": 8,
        }
    )
    storage_patch, tensor_patch = _fake_pointer_patches(args)
    with storage_patch, tensor_patch:
        specs = _chunk_recurrence_tensor_map_specs(plan, args)
    assert specs["v"].box_dim == (32, 16, 1)
    assert specs["out"].box_dim == (32, 16, 1)
    assert specs["state_in"].box_dim == (64, 64, 1)
    assert specs["state_in"] == specs["state_out"]


def test_chunk_recurrence_runtime_validates_sm100_dv2_without_raw_descriptors() -> None:
    args = _runtime_view_args()
    plan = _runtime_plan()
    plan.update({"kind": "chunk_recurrence_sm100", "dv_partitions": 2})
    storage_patch, tensor_patch = _fake_pointer_patches(args)
    with storage_patch, tensor_patch:
        assert _chunk_recurrence_tensor_map_specs(plan, args, validate_only=True) == {}
        with pytest.raises(
            exc.BackendUnsupported,
            match="raw chunk-recurrence TensorMaps require the warp-DV4 schedule",
        ):
            _chunk_recurrence_tensor_map_specs(plan, args)


def test_chunk_recurrence_runtime_allows_exact_value_output_alias() -> None:
    args = list(_runtime_view_args())
    args[6] = args[5]
    runtime_args = tuple(args)
    storage_patch, tensor_patch = _fake_pointer_patches(runtime_args)
    with storage_patch, tensor_patch:
        specs = _chunk_recurrence_tensor_map_specs(_runtime_plan(), runtime_args)
    assert specs["v"].base_ptr == specs["out"].base_ptr


@pytest.mark.parametrize(
    ("metadata_index", "metadata_name"),
    ((8, "cu_seqlens"), (9, "cu_chunks")),
)
def test_chunk_recurrence_runtime_rejects_output_metadata_alias(
    metadata_index: int,
    metadata_name: str,
) -> None:
    args = list(_runtime_view_args())
    original_output = args[6]
    original_metadata = args[metadata_index]
    assert isinstance(original_output, torch.Tensor)
    assert isinstance(original_metadata, torch.Tensor)
    with original_output.fake_mode:
        storage = torch.empty(
            original_output.numel() * original_output.element_size(),
            dtype=torch.uint8,
            device=original_output.device,
        )
        args[6] = storage.view(torch.bfloat16).view(original_output.shape)
        args[metadata_index] = (
            storage[: original_metadata.numel() * original_metadata.element_size()]
            .view(torch.int32)
            .view(original_metadata.shape)
        )
    runtime_args = tuple(args)
    storage_patch, tensor_patch = _fake_pointer_patches(runtime_args)
    with (
        storage_patch,
        tensor_patch,
        pytest.raises(
            exc.BackendUnsupported,
            match=rf"chunk-recurrence output aliases {metadata_name}",
        ),
    ):
        _chunk_recurrence_tensor_map_specs(_runtime_plan(), runtime_args)


def test_chunk_recurrence_sm100_dv2_enters_launch_schema() -> None:
    from helion.runtime.cute import launcher

    def dummy_kernel(*_args: object) -> None:
        pass

    args = _runtime_view_args()
    plan = _runtime_plan()
    plan.update({"kind": "chunk_recurrence_sm100", "dv_partitions": 2})
    dummy_kernel._helion_cute_wrapper_plans = [plan]  # type: ignore[attr-defined]

    def fake_imports() -> tuple[object, object, object]:
        def make_ptr(
            _dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, int]:
            assert assumed_align == 16
            return "ptr", data_ptr

        return object(), make_ptr, object()

    storage_patch, tensor_patch = _fake_pointer_patches(args)
    with (
        storage_patch,
        tensor_patch,
        patch.object(launcher, "_get_cute_launcher_imports", side_effect=fake_imports),
        patch.object(launcher, "_torch_dtype_to_cutlass", side_effect=str),
    ):
        built = launcher._build_cute_schema_and_args(dummy_kernel, args, (17, 12, 1))

    assert len(built.schema) == len(args)
    assert not any(entry[0] == "wrapper_host_scalar" for entry in built.schema)
    assert built.launch_args[-3:] == (17, 12, 1)
    assert built.owned_tensors == ()


def test_chunk_recurrence_dv4_wrapper_uses_external_host_schedule() -> None:
    from helion.runtime.cute.launcher import _create_cute_wrapper

    def dummy_kernel(*_args: object) -> None:
        pass

    descriptor_names = tuple(f"desc_{index}" for index in range(7))
    plan = _runtime_plan(tokens=512, heads=12, sequences=1, chunks=32)
    plan.update(
        {
            "kind": "chunk_recurrence_warp_dv4",
            "desc_args": descriptor_names,
            "workspace_layout_version": 2,
            "outputs_scaled": False,
            "factor_key_xor": 8,
            "threads": 192,
            "smem_bytes": 97_536,
            "device_abi": 3,
            "input_stages": 6,
            "tma_stages": 6,
            "factor_tma_value_splits": 1,
            "output_acc_stages": 2,
            "output_smem_stages": 3,
            "output_store_wait_groups": 0,
            "tmem_cols": 0,
            "dv_partitions": 4,
            "scale_idx": 10,
        }
    )
    dummy_kernel._helion_cute_wrapper_plans = [plan]  # type: ignore[attr-defined]
    tensor_schema = tuple(("tensor", torch.bfloat16, 1, (1,), (1,)) for _ in range(10))
    schema = (
        *tensor_schema,
        ("scalar", "float"),
        *(("wrapper_host_scalar", name, "int") for name in descriptor_names),
    )
    wrapper = _create_cute_wrapper(dummy_kernel, schema, (320, 1, 1))
    source = inspect.getsource(wrapper)
    assert "grid_x = cutlass.Int32(4)" in source
    assert "grid_y = cutlass.Int32(12)" in source
    assert "_helion_sm100_warp_dv4_host(" in source
    assert "cutlass.Float32(arg10)" in source
    assert "block=(320, 1, 1)" not in source


def test_chunk_recurrence_runtime_rejects_split_factor_storage() -> None:
    args = list(_runtime_view_args())
    with args[1].fake_mode:
        args[1] = torch.empty_like(args[1])
    runtime_args = tuple(args)
    storage_patch, tensor_patch = _fake_pointer_patches(runtime_args)
    with (
        storage_patch,
        tensor_patch,
        pytest.raises(exc.BackendUnsupported, match="packed workspace ABI"),
    ):
        _chunk_recurrence_tensor_map_specs(_runtime_plan(), runtime_args)


def _require_dv4_runtime() -> torch.device:
    from helion._compat import requires_cuda_version

    if not torch.cuda.is_available() or not requires_cuda_version("13"):
        pytest.skip("DV4 CUDA-graph runtime test requires CUDA >= 13")
    pytest.importorskip("cutlass.cute")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 10:
        pytest.skip("DV4 CUDA-graph runtime test requires SM100")
    return device


def _dv4_runtime_args(
    device: torch.device,
    *,
    value: float,
    decay: float,
    identity_aq: bool,
) -> tuple[object, ...]:
    from benchmarks.cute.kda_prefill_staged import allocate_kda_factor_workspace

    heads = 1
    tokens = 2 * BT
    chunks = 2
    workspace = allocate_kda_factor_workspace(heads, chunks, device)
    workspace.kd.zero_()
    workspace.qd.zero_()
    workspace.ak.zero_()
    workspace.aq.zero_()
    workspace.g_total.fill_(decay)
    if identity_aq:
        lane = torch.arange(BT, dtype=torch.int64, device=device)
        byte_offset = 2 * (lane[:, None] * BT + (lane[None, :] ^ 8))
        pair_index = (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2
        workspace.aq.view(-1, BT * BT)[:, pair_index] = torch.eye(
            BT, dtype=torch.bfloat16, device=device
        )
    value_output = torch.full(
        (1, tokens, heads, DV), value, dtype=torch.bfloat16, device=device
    )
    state = torch.full((1, heads, DV, DK), 0.25, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor((0, tokens), dtype=torch.int32, device=device)
    cu_chunks = torch.tensor((0, chunks), dtype=torch.int32, device=device)
    return (
        workspace.kd,
        workspace.qd,
        workspace.ak,
        workspace.aq,
        workspace.g_total,
        value_output,
        value_output,
        state,
        cu_seqlens,
        cu_chunks,
        2.0,
    )


def test_dv4_cuda_graph_descriptor_lifetime_after_launch_cache_eviction() -> None:
    """Captured raw TensorMap pointers survive launch-cache eviction."""

    from helion.runtime.cute import launcher

    device = _require_dv4_runtime()
    original_args = _dv4_runtime_args(device, value=0.125, decay=0.5, identity_aq=True)
    config = helion.Config.from_dict(
        {
            **_CONFIG.config,
            "cute_chunk_recurrence_dv_partitions": 4,
        }
    )
    bound = _bt16_resident_chain.bind(original_args)
    compiled = bound.compile_config(config)
    cute_kernel = compiled.__globals__["_helion_kda_chunk_recurrence"]
    assert cute_kernel._helion_cute_wrapper_plans[0]["kind"] == (
        "chunk_recurrence_warp_dv4"
    )

    def exercise_capture(*, managed: bool) -> None:
        for attribute in (
            "_helion_cute_launch_arg_cache",
            "_helion_cute_last_launch_cache",
            "_helion_cute_capture_owned_launch_tensors",
        ):
            cute_kernel.__dict__.pop(attribute, None)

        value_output = original_args[5]
        state = original_args[7]
        assert isinstance(value_output, torch.Tensor)
        assert value_output is original_args[6]
        assert isinstance(state, torch.Tensor)

        value_output.fill_(0.125)
        state.fill_(0.25)
        compiled(*original_args)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            value_output,
            torch.full_like(value_output, 0.25),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state,
            torch.full_like(state, 0.0625),
            rtol=0,
            atol=0,
        )

        launch_cache = cute_kernel._helion_cute_launch_arg_cache
        assert len(launch_cache) == 1
        captured_entry = next(iter(launch_cache.values()))
        assert len(captured_entry.owned_tensors) == 1
        descriptor = captured_entry.owned_tensors[0]
        descriptor_ref = weakref.ref(descriptor)
        descriptor_ptr = descriptor.data_ptr()

        capture_stream = torch.cuda.Stream(device=device)
        if managed:
            with launcher.cute_cuda_graph(stream=capture_stream) as graph:
                compiled(*original_args)
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                compiled(*original_args)
        torch.cuda.synchronize(device)

        capture_cache = cute_kernel._helion_cute_capture_owned_launch_tensors
        assert len(capture_cache) == 1
        assert any(
            tensor is descriptor
            for tensors in capture_cache.values()
            for tensor in tensors.values()
        )

        alternate_args = [
            _dv4_runtime_args(
                device,
                value=float(signature),
                decay=0.75,
                identity_aq=False,
            )
            for signature in range(1, launcher._CUTE_LAUNCH_ARG_CACHE_LIMIT + 2)
        ]
        for args in alternate_args:
            compiled(*args)
        torch.cuda.synchronize(device)

        assert len(launch_cache) == launcher._CUTE_LAUNCH_ARG_CACHE_LIMIT
        assert all(entry is not captured_entry for entry in launch_cache.values())
        del captured_entry, descriptor
        gc.collect()
        retained_descriptor = descriptor_ref()
        assert retained_descriptor is not None
        assert retained_descriptor.data_ptr() == descriptor_ptr
        del retained_descriptor

        value_output.fill_(0.25)
        state.fill_(0.25)
        torch.cuda.synchronize(device)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            value_output,
            torch.full_like(value_output, 0.5),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state,
            torch.full_like(state, 0.0625),
            rtol=0,
            atol=0,
        )

        del graph
        gc.collect()
        if managed:
            assert capture_cache == {}
            assert descriptor_ref() is None
        else:
            # Raw torch.cuda.graph does not expose its owner to the launcher,
            # so its descriptor remains conservatively retained.
            assert descriptor_ref() is not None
            capture_cache.clear()
            gc.collect()
            assert descriptor_ref() is None

    exercise_capture(managed=True)
    exercise_capture(managed=False)
