from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import operator
from typing import TYPE_CHECKING
from unittest.mock import patch

from benchmarks.cute.kda_prefill_kernels import BT
from benchmarks.cute.kda_prefill_kernels import DK
from benchmarks.cute.kda_prefill_kernels import KDA_PREPARE_CONFIG as _CONFIG
from benchmarks.cute.kda_prefill_kernels import (
    kda_chunk_prepare as _five_factor_prepare,
)
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import _compat
from helion import exc
from helion._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from helion._compiler.autotuner_heuristics.cute import CuteChunkPrepareHeuristic
from helion._compiler.cute.chunk_prepare import _packed_workspace_is_exact
from helion._compiler.cute.chunk_prepare import _preferred_split_alias_chunks_per_cta
from helion._compiler.cute.chunk_prepare import _TensorRef
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import _chunk_prepare_expected_tensor_map_specs
from helion.runtime.cute.launcher import _chunk_prepare_tensor_map_specs

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])
_split_alias_device = importlib.import_module(
    "helion._compiler.cute.chunk_prepare_split_alias_device"
)
PREPARE_DEVICE_THREADS = _split_alias_device.PREPARE_DEVICE_THREADS
emit_bt16_prepare = _split_alias_device.emit_bt16_prepare
issue_chunk_tma = _split_alias_device.issue_chunk_tma
mbar_spin_wait = _split_alias_device.mbar_spin_wait
mma_blockdiag_8x8_f16 = _split_alias_device.mma_blockdiag_8x8_f16

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import DeviceIR


def _fake_inputs(
    *,
    tokens: int = 128,
    heads: int = 2,
    chunks: int = 8,
    gate_dtype: torch.dtype = torch.bfloat16,
    packed_workspace: bool = True,
) -> tuple[object, ...]:
    rows = chunks * BT
    with FakeTensorMode():
        q = torch.empty((1, tokens, heads, DK), device=DEVICE, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        gate = torch.empty((1, tokens, heads, DK), device=DEVICE, dtype=gate_dtype)
        beta = torch.empty((1, tokens, heads), device=DEVICE, dtype=torch.bfloat16)
        a_log = torch.empty((heads,), device=DEVICE, dtype=torch.float32)
        dt_bias = torch.empty((heads, DK), device=DEVICE, dtype=torch.float32)
        cu_seqlens = torch.empty((3,), device=DEVICE, dtype=torch.int32)
        cu_chunks = torch.empty((3,), device=DEVICE, dtype=torch.int32)
        chunk_to_seq = torch.empty((chunks,), device=DEVICE, dtype=torch.int32)
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
    return (
        q,
        k,
        gate,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
        kd,
        qd,
        ak,
        aq,
        g_total,
        DK**-0.5,
        -5.0 * 1.4426950408889634,
    )


def _code(
    *,
    tokens: int = 128,
    heads: int = 2,
    chunks: int = 8,
    gate_dtype: torch.dtype = torch.bfloat16,
    outputs_scaled: bool = True,
    mutate: Callable[[DeviceIR], None] | None = None,
    config: helion.Config = _CONFIG,
    capability: tuple[int, int] = (10, 0),
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
    ):
        inputs = _fake_inputs(
            tokens=tokens,
            heads=heads,
            chunks=chunks,
            gate_dtype=gate_dtype,
            packed_workspace=packed_workspace,
        )
        if not outputs_scaled:
            inputs = (*inputs[:-2], None, inputs[-1])
        bound = _five_factor_prepare._bind_isolated(inputs)
        if mutate is not None:
            mutate(bound.host_function.device_ir)
        bound.config_spec.target_device_capability = capability
        return bound.to_triton_code(config)


def test_five_factor_prepare_codegen() -> None:
    code = _code()
    assert code.count("emit_bt16_prepare(") == 1
    assert "True, 0, 1, False)" in code
    assert "block=(128, 1, 1)" in code
    assert "'kind': 'chunk_prepare_tma'" in code
    assert "'chunks_per_cta': 1" in code
    assert "'schedule': 'split_alias_cpc1'" in code
    assert "chunk_prepare_split_alias_device import emit_bt16_prepare" in code
    assert "'outputs_scaled': True" in code
    assert "'device_abi': 3" in code
    assert "'smem_bytes': 21968" in code
    assert "--override-directive-values --maxrregcount=48" in code
    assert "'bf16_mma_count': 168" in code
    assert "'fp16_mma_count': 24" in code
    assert "'q_idx': 0" in code
    assert "'a_log_idx': 4" in code
    assert "'gt_idx': 13" in code
    assert "_helion_cute_disable_bake_tensor_shapes" not in code
    assert "cute.gemm(" not in code


def test_prepare_device_exposes_only_the_matched_gate_contract() -> None:
    source = inspect.getsource(emit_bt16_prepare)
    assert "SAFE_GATE" not in source
    assert "cute.math.log2" not in source


def test_prepare_uses_shared_device_primitives() -> None:
    from helion._compiler.cute import chunk_prepare_split_alias_device as device
    from helion._compiler.cute import kda_device_primitives as primitives

    for name in (
        "mma_m16n8k16_bf16",
        "movmatrix_b16",
        "pack_bf16x2",
        "store_vec8_bf16",
        "tma_store_3d",
        "tma_store_commit_group",
        "tma_store_wait_read",
        "vec4_f32",
        "vec8_bf16",
        "vec_at",
        "warp_arrive",
    ):
        assert getattr(device, name) is getattr(primitives, name)


def test_prepare_exact_lowering_requires_fast_math_policy() -> None:
    with patch.object(_five_factor_prepare.settings, "fast_math", False):
        plan = _captured_plan()
    assert plan is None


@pytest.mark.parametrize("guard", ("_linear_offsets_fit_i32", "_xyz_grid_fits"))
def test_prepare_unsafe_index_or_launch_geometry_fails_closed(guard: str) -> None:
    with patch(f"helion._compiler.cute.chunk_prepare.{guard}", return_value=False):
        plan = _captured_plan()
    assert plan is None


@pytest.mark.parametrize(
    ("collision", "expected"),
    (
        (
            "emit_bt16_prepare",
            (
                "import emit_bt16_prepare as emit_bt16_prepare_1",
                "    emit_bt16_prepare_1(",
            ),
        ),
        (
            "_chunk_prepare_arg_0",
            ("_chunk_prepare_arg_1 = cute.make_tensor(_chunk_prepare_arg_0.iterator",),
        ),
        (
            "_chunk_prepare_desc_q",
            ("'desc_args': ('_chunk_prepare_desc_q_1',",),
        ),
    ),
)
def test_prepare_generated_names_do_not_shadow_tensor_arguments(
    collision: str,
    expected: tuple[str, ...],
) -> None:
    from helion._compiler.cute import chunk_prepare

    original = chunk_prepare._host_tensor_ref

    def colliding_name(node: object) -> object:
        ref = original(node)
        if ref is not None and ref.name == "q_rows":
            return dataclasses.replace(ref, name=collision)
        return ref

    with patch.object(chunk_prepare, "_host_tensor_ref", side_effect=colliding_name):
        code = _code()
    assert all(snippet in code for snippet in expected)


def test_fake_arch_codegen_preserves_process_capability_state() -> None:
    capability_cache = _compat._target_device_capability
    tensor_descriptor_cache = _compat._supports_tensor_descriptor
    min_dot_cache = _compat._min_dot_size
    is_hip_cache = _compat._is_hip
    cache_info = (
        capability_cache.cache_info(),
        tensor_descriptor_cache.cache_info(),
        min_dot_cache.cache_info(),
        is_hip_cache.cache_info(),
    )
    get_device_capability = torch.cuda.get_device_capability

    _code(capability=(10, 0))

    assert (
        capability_cache.cache_info(),
        tensor_descriptor_cache.cache_info(),
        min_dot_cache.cache_info(),
        is_hip_cache.cache_info(),
    ) == cache_info
    assert torch.cuda.get_device_capability is get_device_capability


def test_five_factor_prepare_cpc5_split_alias_codegen() -> None:
    code = _code(tokens=8192, heads=12, chunks=515)
    assert "'chunks_per_cta': 5" in code
    assert "'schedule': 'split_alias_cpc5'" in code
    assert "'smem_bytes': 21968" in code


def test_five_factor_prepare_cpc4_split_alias_remains_available() -> None:
    config = helion.Config.from_dict(
        {**_CONFIG.config, "cute_chunk_prepare_schedule": "split_alias_cpc4"}
    )
    code = _code(tokens=8192, heads=12, chunks=515, config=config)
    assert "'chunks_per_cta': 4" in code
    assert "'schedule': 'split_alias_cpc4'" in code
    assert "'smem_bytes': 21968" in code


def test_prepare_autotuner_searches_all_split_alias_schedules() -> None:
    from helion.autotuner.config_generation import ConfigGeneration

    with (
        patch(
            "helion.runtime.kernel.target_device_capability",
            return_value=(10, 0),
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        bound = _five_factor_prepare._bind_isolated(_fake_inputs())

    fragment = bound.config_spec.cute_chunk_prepare_schedule
    assert fragment is not None
    expected = (
        "split_alias_cpc1",
        "split_alias_cpc2",
        "split_alias_cpc3",
        "split_alias_cpc4",
        "split_alias_cpc5",
    )
    assert fragment.choices == expected
    assert bound.config_spec.compiler_default_config is not None
    assert (
        bound.config_spec.compiler_default_config["cute_chunk_prepare_schedule"]
        == "split_alias_cpc1"
    )
    seeded = [
        config["cute_chunk_prepare_schedule"]
        for config in bound.config_spec.compiler_seed_configs
        if "cute_chunk_prepare_schedule" in config
    ]
    assert seeded == list(expected)
    dimensions = [
        dimension.values
        for dimension in bound.config_spec.iter_search_dimensions()
        if dimension.name == "cute_chunk_prepare_schedule"
    ]
    assert dimensions == [list(expected)]
    generated = [
        config["cute_chunk_prepare_schedule"]
        for _flat, config in ConfigGeneration(
            bound.config_spec
        ).seed_flat_config_pairs()
        if "cute_chunk_prepare_schedule" in config
    ]
    assert generated == list(expected)


def test_five_factor_prepare_fp32_gate_codegen() -> None:
    code = _code(gate_dtype=torch.float32)
    assert code.count("emit_bt16_prepare(") == 1
    assert ", True)" in code
    assert "'gate_is_fp32': True" in code


def test_five_factor_prepare_unscaled_workspace_codegen() -> None:
    code = _code(outputs_scaled=False)
    assert code.count("emit_bt16_prepare(") == 1
    assert "False, 8, 1, False)" in code
    assert "'outputs_scaled': False" in code
    assert "'factor_key_xor': 8" in code
    assert "'device_abi': 4" in code


def _captured_plan(
    *,
    mutate: Callable[[DeviceIR], None] | None = None,
    capability: tuple[int, int] = (10, 0),
    packed_workspace: bool = True,
) -> object:
    from helion._compiler.cute import chunk_prepare

    captured: list[object] = []
    original = chunk_prepare._plan_chunk_prepare

    def capture(graphs: object, tile_strategy: object) -> object:
        result = original(graphs, tile_strategy)  # type: ignore[arg-type]
        captured.append(result)
        return result

    with (
        patch.object(chunk_prepare, "_plan_chunk_prepare", side_effect=capture),
        contextlib.suppress(exc.BackendUnsupported),
    ):
        _code(
            mutate=mutate,
            capability=capability,
            packed_workspace=packed_workspace,
        )
    assert len(captured) == 1
    return captured[0]


def _captured_plan_after_mutation(mutate: Callable[[DeviceIR], None]) -> object:
    return _captured_plan(mutate=mutate)


def test_prepare_rejects_changed_scan_axis() -> None:
    from helion.language import scan_ops

    def mutate(device_ir: DeviceIR) -> None:
        scan = next(
            node
            for graph in device_ir.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is scan_ops._associative_scan
        )
        scan.args = (*scan.args[:2], 1, *scan.args[3:])

    assert _captured_plan_after_mutation(mutate) is None


def test_prepare_rejects_changed_ak_aq_permutation() -> None:
    def mutate(device_ir: DeviceIR) -> None:
        for graph in device_ir.graphs:
            for node in graph.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.bitwise_xor.Scalar
                    and node.args[1] == 8
                ):
                    node.args = (node.args[0], 4)

    assert _captured_plan_after_mutation(mutate) is None


def test_prepare_rejects_changed_dot_accumulator_dtype() -> None:
    from helion.language.matmul_ops import dot

    def mutate(device_ir: DeviceIR) -> None:
        candidate = next(
            node
            for graph in device_ir.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is dot
        )
        candidate.args = (*candidate.args[:3], torch.float16)

    assert _captured_plan_after_mutation(mutate) is None


def test_prepare_rejects_changed_chunk_guard() -> None:
    from helion.language._tracing_ops import _if

    def mutate(device_ir: DeviceIR) -> None:
        loop_if = next(
            node
            for graph in device_ir.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is _if
        )
        predicate = loop_if.args[0]
        assert isinstance(predicate, torch.fx.Node)
        predicate.args = (predicate.args[0], 7)

    assert _captured_plan_after_mutation(mutate) is None


def _prepare_body_nodes(device_ir: DeviceIR) -> list[torch.fx.Node]:
    from helion._compiler.device_ir import IfGraphInfo

    body = next(graph for graph in device_ir.graphs if isinstance(graph, IfGraphInfo))
    return list(body.graph.nodes)


def _change_gate_clamp(device_ir: DeviceIR) -> None:
    clamp = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function"
        and node.target is torch.ops.aten.clamp_min.default
        and node.args[1] == -126.0
    )
    clamp.args = (clamp.args[0], -120.0)


def _change_norm_epsilon(device_ir: DeviceIR) -> None:
    clamp = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function"
        and node.target is torch.ops.aten.clamp_min.default
        and node.args[1] == 1.0e-24
    )
    clamp.args = (clamp.args[0], 1.0e-20)


def _change_tanh_coefficient(device_ir: DeviceIR) -> None:
    tanh = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function" and node.target is torch.ops.aten.tanh.default
    )
    product = tanh.args[0]
    product.args = (product.args[0], 0.25)


def _bypass_a_log_exp2(device_ir: DeviceIR) -> None:
    decay_exp2 = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function"
        and node.target is torch.ops.aten.exp2.default
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target is torch.ops.aten.mul.Tensor
    )
    decay_exp2.target = torch.ops.aten.exp.default


def _bypass_sigmoid_input_cast(device_ir: DeviceIR) -> None:
    sigmoid = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function" and node.target is torch.ops.aten.sigmoid.default
    )
    converted = sigmoid.args[0]
    sigmoid.args = (converted.args[0],)


def _change_q_normalization_rounding(device_ir: DeviceIR) -> None:
    from helion.language import memory_ops

    nodes = _prepare_body_nodes(device_ir)
    q_load = next(
        node
        for node in nodes
        if node.op == "call_function"
        and node.target is memory_ops.load
        and node.args[0].args[0] == "q_rows"
    )
    q_raw = next(
        node
        for node in q_load.users
        if node.target is torch.ops.prims.convert_element_type.default
    )
    norm_product = next(
        node
        for node in q_raw.users
        if node.target is torch.ops.aten.mul.Tensor
        and node.args[0] is q_raw
        and node.args[1] is not None
    )
    rounding = next(
        node
        for node in norm_product.users
        if node.target is torch.ops.prims.convert_element_type.default
    )
    rounding.args = (rounding.args[0], torch.float16)


def _remove_beta_mask(device_ir: DeviceIR) -> None:
    from helion.language import memory_ops

    beta_load = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function"
        and node.target is memory_ops.load
        and node.args[0].args[0] == "beta_rows"
    )
    beta_load.args = (*beta_load.args[:2], None, beta_load.args[3])


def _change_inverse_half_boundary(device_ir: DeviceIR) -> None:
    boundary = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function"
        and node.target is torch.ops.aten.ge.Scalar
        and node.args[1] == 8
    )
    boundary.args = (boundary.args[0], 4)


def _swap_causal_operands(device_ir: DeviceIR) -> None:
    causal = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function" and node.target is torch.ops.aten.ge.Tensor
    )
    causal.args = (causal.args[1], causal.args[0])


def _swap_strict_operands(device_ir: DeviceIR) -> None:
    strict = next(
        node
        for node in _prepare_body_nodes(device_ir)
        if node.op == "call_function" and node.target is torch.ops.aten.gt.Tensor
    )
    strict.args = (strict.args[1], strict.args[0])


def _bypass_scan_helper_addition(device_ir: DeviceIR) -> None:
    from helion._compiler.device_ir import HelperFunctionGraphInfo

    helper = next(
        graph
        for graph in device_ir.graphs
        if isinstance(graph, HelperFunctionGraphInfo)
    )
    first_placeholder = next(
        node for node in helper.graph.nodes if node.op == "placeholder"
    )
    output = next(node for node in helper.graph.nodes if node.op == "output")
    output.args = (first_placeholder,)


def _offset_prepare_chunk_coordinate(device_ir: DeviceIR) -> None:
    from helion._compiler.device_ir import ForLoopGraphInfo
    from helion._compiler.device_ir import IfGraphInfo
    from helion.language import memory_ops
    from helion.language._tracing_ops import _if

    loop = next(
        graph for graph in device_ir.graphs if isinstance(graph, ForLoopGraphInfo)
    )
    loop_if = next(
        node
        for node in loop.graph.nodes
        if node.op == "call_function" and node.target is _if
    )
    predicate = loop_if.args[0]
    assert isinstance(predicate, torch.fx.Node)
    chunk = predicate.args[0]
    assert isinstance(chunk, torch.fx.Node)
    chunk_value = chunk.meta["val"]
    assert isinstance(chunk_value, torch.SymInt)
    with loop.graph.inserting_before(predicate):
        shifted = loop.graph.call_function(operator.add, (chunk, 1))
    shifted.meta = dict(chunk.meta)
    shifted.meta["val"] = chunk_value + 1
    predicate.args = (shifted, predicate.args[1])

    body = next(graph for graph in device_ir.graphs if isinstance(graph, IfGraphInfo))
    chunk_load = next(
        node
        for node in body.graph.nodes
        if node.op == "call_function"
        and node.target is memory_ops.load
        and node.args[0].args[0] == "chunk_to_seq"
    )
    body_chunk = chunk_load.args[1][0]
    assert isinstance(body_chunk, torch.fx.Node)
    shifted_value = shifted.meta["val"]
    assert isinstance(shifted_value, torch.SymInt)
    body_chunk.args = (str(shifted_value._sympy_()),)
    body_chunk.meta["val"] = shifted_value


def _permute_prepare_root_task_axes(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    axes[0], axes[1] = axes[1], axes[0]
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


def _make_prepare_root_task_origin_noncanonical(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    axes[0] = dataclasses.replace(axes[0], canonical_origin=False)
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


def _shorten_prepare_root_task_extent(device_ir: DeviceIR) -> None:
    axes = list(device_ir.task_families[0].axes)
    extent = axes[0].extent
    assert extent is not None and not isinstance(extent, str)
    axes[0] = dataclasses.replace(axes[0], extent=extent - 1)
    device_ir.task_families[0] = dataclasses.replace(
        device_ir.task_families[0],
        axes=tuple(axes),
    )


@pytest.mark.parametrize(
    "mutate",
    (
        _change_gate_clamp,
        _change_norm_epsilon,
        _change_tanh_coefficient,
        _bypass_a_log_exp2,
        _bypass_sigmoid_input_cast,
        _change_q_normalization_rounding,
        _remove_beta_mask,
        _change_inverse_half_boundary,
        _swap_causal_operands,
        _swap_strict_operands,
        _bypass_scan_helper_addition,
        _offset_prepare_chunk_coordinate,
        _permute_prepare_root_task_axes,
        _make_prepare_root_task_origin_noncanonical,
        _shorten_prepare_root_task_extent,
    ),
)
def test_prepare_rejects_semantic_mutations(
    mutate: Callable[[DeviceIR], None],
) -> None:
    assert _captured_plan_after_mutation(mutate) is None


@pytest.mark.parametrize(
    "component",
    ("rshift", "bitwise_and", "lshift", "divisor", "byte_multiplier"),
)
def test_prepare_rejects_changed_aq_physical_index_constants(
    component: str,
) -> None:
    def mutate(device_ir: DeviceIR) -> None:
        nodes = _prepare_body_nodes(device_ir)
        pair_index = next(
            node
            for node in nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten.div.Tensor_mode
        )
        pair_xor = pair_index.args[0]
        byte_offset = pair_xor.args[0]
        shifted = pair_xor.args[1]
        masked = shifted.args[0]
        rshift = masked.args[0]
        targets = {
            "rshift": (rshift, 6),
            "bitwise_and": (masked, 3),
            "lshift": (shifted, 3),
            "divisor": (pair_index, 4),
            "byte_multiplier": (byte_offset, 4),
        }
        node, replacement = targets[component]
        node.args = (node.args[0], replacement)

    assert _captured_plan_after_mutation(mutate) is None


@pytest.mark.parametrize("tensor_name", ("q_rows", "k_rows", "gate_rows", "beta_rows"))
def test_prepare_rejects_nondefault_masked_load_other(tensor_name: str) -> None:
    from helion.language import memory_ops

    def mutate(device_ir: DeviceIR) -> None:
        load = next(
            node
            for node in _prepare_body_nodes(device_ir)
            if node.op == "call_function"
            and node.target is memory_ops.load
            and node.args[0].args[0] == tensor_name
        )
        load.args = (*load.args[:3], 1.0)

    assert _captured_plan_after_mutation(mutate) is None


def test_prepare_rejects_sm80() -> None:
    assert _captured_plan(capability=(8, 0)) is None


def test_prepare_split_workspace_falls_back_during_planning() -> None:
    assert _captured_plan(packed_workspace=False) is None


def test_prepare_packed_workspace_contract() -> None:
    args = _fake_inputs()
    refs = tuple(
        _TensorRef(args[index], name)
        for index, name in zip(
            range(9, 14), ("kd", "qd", "ak", "aq", "g_total"), strict=True
        )
    )
    assert _packed_workspace_is_exact(*refs)
    broken = (*refs[:1], _TensorRef(torch.empty_like(args[10]), "qd"), *refs[2:])
    assert not _packed_workspace_is_exact(*broken)


def test_prepare_schedule_static_contract() -> None:
    assert PREPARE_DEVICE_THREADS == 128


def test_prepare_fuses_raw_a_log_exp2_into_device_kernel() -> None:
    source = inspect.getsource(emit_bt16_prepare)
    assert "cutlass.Float32(ga_log[head])" in source
    assert "* cutlass.Float32(LOG2_E), fastmath=True" in source


def test_prepare_block_sparse_inverse_has_six_native_mmas() -> None:
    emitter_source = inspect.getsource(emit_bt16_prepare)
    inverse_source = emitter_source.split(
        "The two diagonal 8x8 blocks share one native m16n8 issue", maxsplit=1
    )[1].split("# step 7 of the design", maxsplit=1)[0]
    assert inverse_source.count("mma_blockdiag_8x8_f16(") == 4
    assert inverse_source.count("mma_m16n8k16_f16(") == 2
    assert "mma_16x16_f16(" not in inverse_source
    assert inspect.getsource(mma_blockdiag_8x8_f16).count("mma_m16n8k16_f16(") == 1


def test_prepare_input_tma_is_three_full_width_transactions() -> None:
    source = inspect.getsource(issue_chunk_tma)
    assert source.count("tma_load_4d(") == 3
    assert "tma_load_3d(" not in source


def test_prepare_uses_tight_mbarrier_waits() -> None:
    source = inspect.getsource(emit_bt16_prepare)
    assert source.count("mbar_spin_wait(") == 14
    assert "cute.arch.mbarrier_wait(" not in source
    helper_source = inspect.getsource(mbar_spin_wait)
    assert "mbarrier.try_wait.parity.shared::cta.b64" in helper_source
    assert "nanosleep." not in helper_source.lower()


def test_prepare_aliases_qd_ki_after_split_release() -> None:
    source = inspect.getsource(emit_bt16_prepare)
    assert "p_qd = p_q" in source
    assert "p_ki = p_k" in source
    assert "allocate_array(cutlass.Int64, 10)" in source
    assert source.count("mbar_q_released") >= 4
    assert source.count("mbar_k_released") >= 4


@pytest.mark.parametrize(
    ("total_chunks", "heads", "num_sm", "expected"),
    (
        (32, 12, 152, 1),
        (192, 12, 152, 2),
        (323, 12, 152, 4),
        (515, 12, 152, 5),
        (32, 12, 0, 4),
    ),
)
def test_preferred_split_alias_chunks_per_cta(
    total_chunks: int, heads: int, num_sm: int, expected: int
) -> None:
    assert (
        _preferred_split_alias_chunks_per_cta(
            total_chunks=total_chunks,
            heads=heads,
            num_sm=num_sm,
        )
        == expected
    )


def test_prepare_seed_cache_specializes_on_effective_sm_count() -> None:
    assert compiler_seed_specialization_facts(
        "cute", [CuteChunkPrepareHeuristic.name]
    ) == frozenset({"config_num_sm", "input_tensor_metadata"})
    assert {
        num_sm: _preferred_split_alias_chunks_per_cta(
            total_chunks=92,
            heads=12,
            num_sm=num_sm,
        )
        for num_sm in (132, 148)
    } == {132: 2, 148: 1}


@pytest.mark.parametrize(
    ("gate_dtype", "gate_box"),
    ((torch.bfloat16, (64, 16, 1, 2)), (torch.float32, (32, 16, 1, 4))),
)
def test_prepare_descriptor_geometry(
    gate_dtype: torch.dtype,
    gate_box: tuple[int, ...],
) -> None:
    specs = _chunk_prepare_expected_tensor_map_specs(
        total_tokens=8192,
        total_chunks=515,
        heads=12,
        q_ptr=0x1000,
        k_ptr=0x2000,
        gate_ptr=0x3000,
        factor_ptr=0x4000,
        gate_dtype=gate_dtype,
    )
    assert tuple(spec.global_dim for spec in specs[:3]) == (
        (64, 8192, 12, 2),
        (64, 8192, 12, 2),
        ((32, 8192, 12, 4) if gate_dtype is torch.float32 else (64, 8192, 12, 2)),
    )
    assert specs[0].global_stride_bytes == (3072, 256, 128)
    assert specs[0].box_dim == (64, 16, 1, 2)
    assert specs[2].box_dim == gate_box
    assert specs[3].global_dim == (128, 8240, 36)
    assert specs[3].global_stride_bytes == (256, 2_109_440)
    assert specs[3].box_dim == (64, 16, 1)


def _runtime_plan() -> dict[str, object]:
    plan: dict[str, object] = {
        "kind": "chunk_prepare_tma",
        "device_abi": 3,
        "chunk_size": 16,
        "key_size": 128,
        "chunks_per_cta": 4,
        "schedule": "split_alias_cpc4",
        "outputs_scaled": True,
        "factor_key_xor": 0,
        "total_tokens": 128,
        "total_chunks": 8,
        "heads": 2,
        "gate_is_fp32": False,
    }
    for index, name in enumerate(
        (
            "q",
            "k",
            "g",
            "beta",
            "a_log",
            "dt",
            "cu_seqlens",
            "cu_chunks",
            "chunk_to_seq",
            "kd",
            "qd",
            "ak",
            "aq",
            "gt",
        )
    ):
        plan[f"{name}_idx"] = index
    return plan


def _runtime_view_args() -> tuple[torch.Tensor, ...]:
    args = _fake_inputs()
    return (
        args[0].view(256, 128),
        args[1].view(256, 128),
        args[2].view(256, 128),
        args[3].view(256),
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9].view(256, 128),
        args[10].view(256, 128),
        args[11].view(256, 128),
        args[12].view(16, 256),
        args[13].view(16, 128),
    )


def test_prepare_runtime_validates_packed_workspace_and_specs() -> None:
    from torch._subclasses.fake_tensor import FakeTensor

    args = _runtime_view_args()
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

    expected_bases = tuple(
        storage_bases[args[index].untyped_storage()._cdata] for index in (0, 1, 2, 9)
    )
    with (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
    ):
        specs = _chunk_prepare_tensor_map_specs(_runtime_plan(), args)
    assert tuple(spec.base_ptr for spec in specs) == expected_bases

    cpc1_plan = _runtime_plan()
    cpc1_plan["chunks_per_cta"] = 1
    cpc1_plan["schedule"] = "split_alias_cpc1"
    with (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
    ):
        cpc1_specs = _chunk_prepare_tensor_map_specs(cpc1_plan, args)
    assert cpc1_specs == specs

    cpc5_plan = _runtime_plan()
    cpc5_plan["chunks_per_cta"] = 5
    cpc5_plan["schedule"] = "split_alias_cpc5"
    with (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
    ):
        cpc5_specs = _chunk_prepare_tensor_map_specs(cpc5_plan, args)
    assert cpc5_specs == specs


def test_prepare_runtime_rejects_split_factor_allocations() -> None:
    from torch._subclasses.fake_tensor import FakeTensor

    args = list(_runtime_view_args())
    assert isinstance(args[10], FakeTensor)
    with args[10].fake_mode:
        args[10] = torch.empty_like(args[10])
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

    with (
        patch.object(torch.storage.UntypedStorage, "data_ptr", storage_data_ptr),
        patch.object(FakeTensor, "data_ptr", tensor_data_ptr),
        pytest.raises(exc.BackendUnsupported, match="packed workspace ABI"),
    ):
        _chunk_prepare_tensor_map_specs(_runtime_plan(), tuple(args))


@pytest.mark.parametrize(
    ("chunks_per_cta", "grid_x", "schedule"),
    (
        (1, 8, "split_alias_cpc1"),
        (2, 4, "split_alias_cpc2"),
        (3, 3, "split_alias_cpc3"),
        (4, 2, "split_alias_cpc4"),
        (5, 2, "split_alias_cpc5"),
    ),
)
def test_prepare_wrapper_overrides_grid_and_appends_descriptors(
    chunks_per_cta: int, grid_x: int, schedule: str
) -> None:
    from helion.runtime.cute.launcher import _create_cute_wrapper

    def dummy_kernel() -> None:
        pass

    dummy_kernel._helion_cute_wrapper_plans = [  # type: ignore[attr-defined]
        {
            "kind": "chunk_prepare_tma",
            "desc_args": ("desc_q", "desc_k", "desc_g", "desc_factor"),
            "total_chunks": 8,
            "heads": 2,
            "chunk_size": 16,
            "key_size": 128,
            "chunks_per_cta": chunks_per_cta,
            "schedule": schedule,
            "outputs_scaled": True,
            "factor_key_xor": 0,
            "smem_bytes": 21_968,
            "bf16_mma_count": 168,
            "fp16_mma_count": 24,
            "device_abi": 3,
        }
    ]
    schema = tuple(
        ("wrapper_host_scalar", name, "int")
        for name in ("desc_q", "desc_k", "desc_g", "desc_factor")
    )
    wrapper = _create_cute_wrapper(dummy_kernel, schema, (128, 1, 1))
    source = inspect.getsource(wrapper)
    assert f"grid_x = cutlass.Int32({grid_x})" in source
    assert "grid_y = cutlass.Int32(2)" in source
    assert "_kernel(desc_q, desc_k, desc_g, desc_factor).launch" in source
    assert "block=(128, 1, 1)" in source


def test_prepare_descriptor_args_enter_cached_launch_schema() -> None:
    from helion.runtime.cute import launcher

    def dummy_kernel(*_args: object) -> None:
        pass

    plan = {
        "kind": "chunk_prepare_tma",
        "desc_args": ("desc_q", "desc_k", "desc_g", "desc_factor"),
    }
    dummy_kernel._helion_cute_wrapper_plans = [plan]  # type: ignore[attr-defined]
    tensor = torch.empty(1)
    args = (*((tensor,) * 14), 0.125, -7.0)
    descriptor_storage = torch.empty(512, dtype=torch.uint8)

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

    with (
        patch.object(launcher, "_validate_cute_launcher_tensor"),
        patch.object(launcher, "_torch_dtype_to_cutlass", side_effect=str),
        patch.object(launcher, "_get_cute_launcher_imports", side_effect=fake_imports),
        patch.object(
            launcher,
            "_build_chunk_prepare_tensor_maps",
            return_value=(descriptor_storage, (0x1000, 0x1080, 0x1100, 0x1180)),
        ),
    ):
        built = launcher._build_cute_schema_and_args(dummy_kernel, args, (17, 1, 1))

    assert built.schema[-4:] == (
        ("wrapper_host_scalar", "desc_q", "int"),
        ("wrapper_host_scalar", "desc_k", "int"),
        ("wrapper_host_scalar", "desc_g", "int"),
        ("wrapper_host_scalar", "desc_factor", "int"),
    )
    assert built.launch_args[-7:] == (0x1000, 0x1080, 0x1100, 0x1180, 17, 1, 1)
    assert built.owned_tensors == (descriptor_storage,)
