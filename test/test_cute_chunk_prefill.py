from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._compiler.autotuner_heuristics.cute import CuteChunkPrefillHeuristic
from helion._compiler.cute.chunk_prefill import _disjoint_storage
from helion._compiler.cute.chunk_prefill import match_chunk_prefill_region
from helion._compiler.cute.chunk_prefill import match_chunk_prefill_step
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import HelperFunctionGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_SCHEDULE_KEY
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_TASK_ORDER_KEY
from helion.exc import InvalidConfig
from helion.language.matmul_ops import dot
from helion.language.memory_ops import store

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import GraphInfo
    from helion.runtime.kernel import BoundKernel

pytest.importorskip("cutlass.cute")
pytestmark = skipUnlessBackends(["cute"])


def _inputs(
    tokens: int = 64,
    heads: int = 2,
    sequences: int = 2,
    *,
    key_width: int = 128,
    value_width: int = 128,
    cu_dtype: torch.dtype = torch.int64,
    device: torch.device = DEVICE,
) -> tuple[object, ...]:
    with FakeTensorMode():
        q = torch.empty(
            (1, tokens, heads, key_width), device=device, dtype=torch.bfloat16
        )
        k, gate = (torch.empty_like(q) for _ in range(2))
        v = torch.empty(
            (1, tokens, heads, value_width), device=device, dtype=torch.bfloat16
        )
        beta = torch.empty((1, tokens, heads), device=device, dtype=torch.bfloat16)
        a_log = torch.empty((heads,), device=device, dtype=torch.float32)
        bias = torch.empty((heads, key_width), device=device, dtype=torch.float32)
        state = torch.empty(
            (sequences, heads, value_width, key_width),
            device=device,
            dtype=torch.float32,
        )
        output, final = torch.empty_like(v), torch.empty_like(state)
        cu = torch.empty((sequences + 1,), device=device, dtype=cu_dtype)
    return (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        state,
        output,
        final,
        cu,
        128**-0.5,
        -5 * 1.4426950408889634,
    )


@pytest.fixture(scope="module")
def bound() -> BoundKernel:
    return kda_prefill_native_math._bind_isolated(_inputs())


@pytest.fixture
def bt32_bound(prefill_cpu_target: None) -> BoundKernel:
    return kda_prefill_native_math_bt32._bind_isolated(
        _inputs(heads=8, device=torch.device("cpu"))
    )


def _mutated_plan(
    bound: BoundKernel, mutate: Callable[[list[GraphInfo]], None]
) -> object:
    assert bound.host_function is not None
    graphs = [g.copy() for g in bound.host_function.device_ir.cute_semantic_graphs]
    mutate(graphs)
    with bound.env, bound.host_function:
        return match_chunk_prefill_region(bound.host_function.device_ir, graphs)


def _loop(graphs: list[GraphInfo]) -> torch.fx.Graph:
    return next(g.graph for g in graphs if isinstance(g, ForLoopGraphInfo))


def _change_state_dtype(graphs: list[GraphInfo]) -> None:
    node = next(n for n in _loop(graphs).nodes if n.op == "output").args[0][0]
    node.args = (*node.args[:3], torch.bfloat16)


def _transpose_inverse_rhs(graphs: list[GraphInfo]) -> None:
    graph = _loop(graphs)
    node = next(
        n
        for n in graph.nodes
        if n.target is dot and n.args[0].meta["val"].dtype is torch.float16
    )
    with graph.inserting_before(node):
        rhs = graph.call_function(
            torch.ops.aten.permute.default, (node.args[1], [1, 0])
        )
    node.args = (node.args[0], rhs, *node.args[2:])


def _change_inverse_precision(graphs: list[GraphInfo]) -> None:
    node = next(
        n
        for n in _loop(graphs).nodes
        if n.target is dot and n.args[0].meta["val"].dtype is torch.float16
    )
    operand = node.args[0]
    operand.args = (operand.args[0], torch.bfloat16)


def _drop_projection_rounding(graphs: list[GraphInfo]) -> None:
    graph = _loop(graphs)
    projection = next(n for n in graph.nodes if n.name == "projected")
    rounded = next(
        n
        for n in projection.users
        if n.target is torch.ops.prims.convert_element_type.default
    )
    rounded.replace_all_uses_with(projection)
    graph.erase_node(rounded)


def _change_causal_mask(graphs: list[GraphInfo]) -> None:
    node = next(n for n in _loop(graphs).nodes if n.target is torch.ops.aten.ge.Tensor)
    node.target = torch.ops.aten.gt.Tensor


def _change_gate_scan(graphs: list[GraphInfo]) -> None:
    graph = next(g.graph for g in graphs if isinstance(g, HelperFunctionGraphInfo))
    node = next(n for n in graph.nodes if n.target is torch.ops.aten.add.Tensor)
    node.target = torch.ops.aten.mul.Tensor


def _change_key_coordinates(graphs: list[GraphInfo]) -> None:
    node = next(
        n
        for n in _loop(graphs).nodes
        if n.target is torch.ops.prims.iota.default and n.args == (128,)
    )
    node.kwargs = {**node.kwargs, "start": 1}


def _remove_output_mask(graphs: list[GraphInfo]) -> None:
    node = next(n for n in _loop(graphs).nodes if n.target is store)
    node.args = (*node.args[:3], None)


def _add_effect(graphs: list[GraphInfo]) -> None:
    graph = _loop(graphs)
    node = next(n for n in graph.nodes if n.target is store)
    with graph.inserting_after(node):
        graph.call_function(store, node.args)


def _transpose_final_state(graphs: list[GraphInfo]) -> None:
    graph = next(g.graph for g in graphs if isinstance(g, RootGraphInfo))
    node = next(n for n in graph.nodes if n.target is store)
    indices = list(node.args[1])
    indices[2], indices[3] = indices[3], indices[2]
    node.args = (node.args[0], indices, *node.args[2:])


@pytest.mark.parametrize(
    "mutate",
    [
        _change_state_dtype,
        _transpose_inverse_rhs,
        _change_inverse_precision,
        _drop_projection_rounding,
        _change_causal_mask,
        _change_gate_scan,
        _change_key_coordinates,
        _remove_output_mask,
        _add_effect,
        _transpose_final_state,
    ],
)
def test_prefill_rejects_semantic_mutation(
    bound: BoundKernel, mutate: Callable[[list[GraphInfo]], None]
) -> None:
    assert _mutated_plan(bound, mutate) is None


@pytest.mark.parametrize("scale_name", ["output_scale", "gate_scale"])
def test_prefill_rejects_loaded_tensor_scale(
    bound: BoundKernel, scale_name: str
) -> None:
    def use_loaded_scale(graphs: list[GraphInfo]) -> None:
        graph = _loop(graphs)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            step = match_chunk_prefill_step(graph)
        assert step is not None
        scalar = step.output_scale if scale_name == "output_scale" else step.gate_scale
        # A legal scalar tensor load can vary by head. It cannot be replaced
        # by a uniform host scalar, even though the rest of the DAG matches.
        scalar.replace_all_uses_with(step.a_log_load)
        graph.erase_node(scalar)

    assert _mutated_plan(bound, use_loaded_scale) is None


def test_prefill_does_not_match_node_names(bound: BoundKernel) -> None:
    def rename(graphs: list[GraphInfo]) -> None:
        for graph in graphs:
            for index, node in enumerate(graph.graph.nodes):
                node.name = f"anonymous_{index}"

    assert _mutated_plan(bound, rename) is not None


@pytest.mark.parametrize("geometry", [(64, 2, 2), (96, 3, 4), (512, 12, 1)])
def test_prefill_matches_geometry(geometry: tuple[int, int, int]) -> None:
    bound = kda_prefill_native_math._bind_isolated(_inputs(*geometry))
    assert bound.host_function is not None
    assert bound.host_function.device_ir.cute_chunk_prefill_region is not None


def test_prefill_codegen_uses_fused_host(bound: BoundKernel) -> None:
    code = bound.to_triton_code(
        helion.Config(
            block_sizes=[64],
            num_warps=4,
            num_stages=2,
            indexing="pointer",
            pid_type="flat",
        )
    )
    assert "chunk_prefill_sm100" in code
    assert "initial_state_idx" in code and "final_state_idx" in code


def test_prefill_storage_guard_rejects_output_alias():
    args = _inputs()
    q, k, v, gate, beta, a_log, bias, initial, output, final, cu = args[:11]
    values = [q, k, v, gate, beta, a_log, bias, initial, cu, output, final]
    assert _disjoint_storage(values)
    values[-1] = initial
    assert not _disjoint_storage(values)
    values[-1] = final
    values[-2] = q
    assert not _disjoint_storage(values)


@pytest.mark.parametrize(
    "order", ["identity", "longest_first", "longest_first_precompute"]
)
@pytest.mark.parametrize("schedule", ["single", "prefix_tail_2", "prefix_tail_4"])
def test_prefill_task_order_seed_and_normalization(
    bound: BoundKernel, order: str, schedule: str
) -> None:
    assert bound.host_function is not None
    seeds = CuteChunkPrefillHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    assert seeds is not None
    assert (order, schedule) in [
        (seed[CUTE_CHUNK_PREFILL_TASK_ORDER_KEY], seed[CUTE_CHUNK_PREFILL_SCHEDULE_KEY])
        for seed in seeds
    ]
    config = helion.Config(
        block_sizes=[64],
        cute_chunk_prefill_task_order=order,
        cute_chunk_prefill_schedule=schedule,
    )
    normalized = bound.env.config_spec.normalized_config(config)
    assert normalized[CUTE_CHUNK_PREFILL_TASK_ORDER_KEY] == order
    assert normalized[CUTE_CHUNK_PREFILL_SCHEDULE_KEY] == schedule
    assert CUTE_CHUNK_PREFILL_TASK_ORDER_KEY in bound.env.config_spec._flat_fields()


def test_prefill_task_order_rejects_unknown(bound: BoundKernel) -> None:
    with pytest.raises(InvalidConfig, match="must be one of"):
        bound.env.config_spec.normalized_config(
            helion.Config(block_sizes=[64], cute_chunk_prefill_task_order="unknown")
        )


def test_prefill_search_only_varies_effective_choices(bound: BoundKernel) -> None:
    assert set(bound.env.config_spec._flat_fields()) == {
        "block_sizes",
        CUTE_CHUNK_PREFILL_TASK_ORDER_KEY,
        CUTE_CHUNK_PREFILL_SCHEDULE_KEY,
    }
    generation = bound.env.config_spec.create_config_generation()
    assert generation.block_size_indices == [0]
    assert [fragment.cardinality() for fragment in generation.flat_spec] == [1, 3, 3]
    config = bound.env.config_spec.default_config()
    assert "chunk_prefill_sm100" in bound.to_triton_code(config)


def test_prefill_full_search_population_obeys_tensor_constraints(
    bound: BoundKernel,
) -> None:
    assert bound.host_function is not None
    spec = bound.env.config_spec
    assert spec.tensor_numel_constraints
    generation = spec.create_config_generation()
    seeds = CuteChunkPrefillHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    assert seeds is not None
    population = generation.random_population_flat(32, user_seed_configs=seeds)
    assert len(population) == 32
    configs = [generation.unflatten(flat) for flat in population]
    assert {config[CUTE_CHUNK_PREFILL_TASK_ORDER_KEY] for config in configs} == {
        "identity",
        "longest_first",
        "longest_first_precompute",
    }
    assert {config[CUTE_CHUNK_PREFILL_SCHEDULE_KEY] for config in configs} == {
        "single",
        "prefix_tail_2",
        "prefix_tail_4",
    }
    for flat, config in zip(population, configs, strict=True):
        block_sizes = config["block_sizes"]
        assert isinstance(block_sizes, list) and block_sizes == [64]
        assert generation.flatten(config) == flat
        assert all(
            constraint.check_fn(
                *(block_sizes[index] for index in constraint.block_indices)
            )
            for constraint in spec.tensor_numel_constraints
        )
    for flat in (
        generation.default_flat(),
        generation.random_flat(),
        generation.biased_random_flat(),
    ):
        assert flat[0] == 64
        for neighbor in generation.coordinate_neighbor_projections(flat):
            assert neighbor.key in (
                CUTE_CHUNK_PREFILL_TASK_ORDER_KEY,
                CUTE_CHUNK_PREFILL_SCHEDULE_KEY,
            )
    assert generation.unflatten(
        generation.differential_mutation(*population[:4], crossover_rate=0.5)
    )["block_sizes"] == [64]


@pytest.mark.parametrize("order", ["identity", "longest_first_precompute"])
def test_bt32_codegen_and_search_space(bt32_bound: BoundKernel, order: str) -> None:
    assert bt32_bound.host_function is not None
    region = bt32_bound.host_function.device_ir.cute_chunk_prefill_region
    assert region is not None
    assert (region.chunk_size, region.numerical_policy) == (
        32,
        "centered_bt32_fp32_rhs_v2",
    )
    spec = bt32_bound.config_spec
    assert spec.cute_chunk_prefill_schedule is not None
    assert spec.cute_chunk_prefill_task_order is not None
    assert spec.cute_chunk_prefill_schedule.choices == ("single",)
    assert spec.cute_chunk_prefill_task_order.choices == (
        "identity",
        "longest_first_precompute",
    )
    seeds = CuteChunkPrefillHeuristic.get_seed_configs(
        bt32_bound.env, bt32_bound.host_function.device_ir
    )
    assert seeds is not None
    assert {
        (
            seed[CUTE_CHUNK_PREFILL_TASK_ORDER_KEY],
            seed[CUTE_CHUNK_PREFILL_SCHEDULE_KEY],
        )
        for seed in seeds
    } == {
        ("identity", "single"),
        ("longest_first_precompute", "single"),
    }
    config = helion.Config(
        block_sizes=[64],
        cute_chunk_prefill_task_order=order,
        cute_chunk_prefill_schedule="single",
    )
    source = bt32_bound.to_triton_code(config)
    assert "'device_abi': 2" in source
    assert "'threads': 1024" in source
    assert "'smem_bytes': 227968" in source
    assert "'tmem_columns': 256" in source
    assert "'min_blocks_per_mp': 1" in source
    assert "'chunk_size': 32" in source
    assert "'numerical_policy': 'centered_bt32_fp32_rhs_v2'" in source
    assert f"'task_order': '{order}'" in source
    assert "'schedule': 'single'" in source
    assert "prefix_tail" not in source


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (CUTE_CHUNK_PREFILL_TASK_ORDER_KEY, "longest_first"),
        (CUTE_CHUNK_PREFILL_SCHEDULE_KEY, "prefix_tail_2"),
        (CUTE_CHUNK_PREFILL_SCHEDULE_KEY, "prefix_tail_4"),
    ],
)
def test_bt32_rejects_unimplemented_configs(
    bt32_bound: BoundKernel, key: str, value: str
) -> None:
    with pytest.raises(InvalidConfig, match="must be one of"):
        bt32_bound.config_spec.normalized_config(
            helion.Config.from_dict({"block_sizes": [64], key: value})
        )


@pytest.fixture
def prefill_cpu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake CPU tensors and an explicit target let these admission/cache tests
    # exercise real binding without CUDA allocation or device compilation.
    for module in ("helion.runtime.kernel", "helion._compiler.compile_environment"):
        monkeypatch.setattr(
            importlib.import_module(module),
            "target_device_capability",
            lambda device: (10, 0),
        )
    monkeypatch.setattr("helion.language.loops.use_tileir_tunables", lambda: False)
    monkeypatch.setattr("helion.language.loops._supports_warp_specialize", lambda: True)
    monkeypatch.setattr("helion._compat._supports_tensor_descriptor", lambda: True)
    monkeypatch.setattr("helion._compat._min_dot_size", lambda *args: (16, 16, 16))
    monkeypatch.setattr("helion._compat._is_hip", lambda: False)


def _assert_prefill_generic_search(
    bound: BoundKernel, *, expect_matched_region: bool = True
) -> None:
    assert bound.host_function is not None
    assert (
        bound.host_function.device_ir.cute_chunk_prefill_region is not None
    ) is expect_matched_region
    spec = bound.config_spec
    assert spec.cute_chunk_prefill_task_order is None
    assert spec.cute_chunk_prefill_schedule is None
    fields = spec._flat_fields()
    assert {"num_threads", "cute_vector_widths"} <= fields.keys()
    assert CUTE_CHUNK_PREFILL_TASK_ORDER_KEY not in fields
    assert spec.block_sizes[0].min_size == 64
    assert spec.block_sizes[0].autotuner_min < spec.block_sizes[0].max_size
    assert spec.block_sizes[0].max_size >= 128
    generation = spec.create_config_generation()
    cardinality = generation.flat_spec[generation.block_size_indices[0]].cardinality()
    assert cardinality is not None and cardinality > 1


@pytest.mark.usefixtures("prefill_cpu_target")
@pytest.mark.parametrize(
    ("heads", "cu_dtype", "expect_matched_region"),
    [(7, torch.int64, True), (8, torch.int32, False)],
)
def test_bt32_static_guards_keep_generic_search(
    heads: int, cu_dtype: torch.dtype, expect_matched_region: bool
) -> None:
    kernel = helion.kernel(
        kda_prefill_native_math_bt32.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
    )
    bound = kernel._bind_isolated(
        _inputs(heads=heads, cu_dtype=cu_dtype, device=torch.device("cpu"))
    )
    _assert_prefill_generic_search(bound, expect_matched_region=expect_matched_region)


@pytest.mark.usefixtures("prefill_cpu_target")
@pytest.mark.parametrize(
    "denied",
    [
        "fast_math",
        "sm90",
        "sm110",
        "key_width",
        "value_width",
        "i32",
        "grid",
        "pid",
        "persistent",
    ],
)
def test_prefill_fallback_keeps_generic_search(
    denied: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings: dict[str, Any] = {
        "backend": "cute",
        "static_shapes": True,
        "fast_math": True,
    }
    geometry: dict[str, Any] = {}
    if denied == "fast_math":
        settings["fast_math"] = False
    elif denied in ("sm90", "sm110"):
        capability = (9, 0) if denied == "sm90" else (11, 0)
        for module in ("helion.runtime.kernel", "helion._compiler.compile_environment"):
            monkeypatch.setattr(
                importlib.import_module(module),
                "target_device_capability",
                lambda device: capability,
            )
    elif denied == "key_width":
        geometry["key_width"] = 64
    elif denied == "value_width":
        geometry["value_width"] = 256
    elif denied == "i32":
        geometry.update(tokens=2**24, heads=2)
    elif denied == "grid":
        geometry["sequences"] = 65535
    elif denied == "pid":
        settings["autotune_config_overrides"] = {"pid_type": "xyz"}
    elif denied == "persistent":
        settings["autotune_force_persistent"] = True
    kernel = helion.kernel(kda_prefill_native_math.fn, **settings)
    bound = kernel._bind_isolated(_inputs(**geometry, device=torch.device("cpu")))
    _assert_prefill_generic_search(bound)


@pytest.mark.usefixtures("prefill_cpu_target")
@pytest.mark.parametrize("disable_heuristics", [False, True])
def test_prefill_storage_rebinding_keeps_separate_searches(
    disable_heuristics: bool,
) -> None:
    kernel = helion.kernel(
        kda_prefill_native_math.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        disable_autotuner_heuristics=disable_heuristics,
    )
    args = _inputs(device=torch.device("cpu"))
    valid = kernel.bind(args)
    assert valid.config_spec.cute_chunk_prefill_task_order is not None
    aliased = list(args)
    # A distinct view defeats identity-only alias checks; admission must use
    # the registered span classifier in the reusable specialization key.
    assert isinstance(args[0], torch.Tensor)
    aliased[8] = args[0].view_as(args[0])
    invalid = kernel.bind(tuple(aliased))
    assert invalid is not valid
    _assert_prefill_generic_search(invalid)
    assert kernel.bind(args) is valid
    assert kernel.bind(tuple(aliased)) is invalid
    assert valid.config_spec.block_sizes[0].max_size == 64
    if disable_heuristics:
        assert valid.config_spec.compiler_seed_configs == []
        assert invalid.config_spec.compiler_seed_configs == []


@pytest.mark.usefixtures("prefill_cpu_target")
def test_prefill_actual_pointer_alignment_rebinding() -> None:
    # Concrete CPU allocations exercise actual addresses, unlike FakeTensor's
    # storage identity model. Binding remains a CPU-only tracing operation.
    args = tuple(
        torch.empty(t.shape, dtype=t.dtype) if isinstance(t, torch.Tensor) else t
        for t in _inputs(device=torch.device("cpu"))
    )
    kernel = helion.kernel(
        kda_prefill_native_math.fn, backend="cute", static_shapes=True, fast_math=True
    )
    valid = kernel.bind(args)
    assert valid.config_spec.cute_chunk_prefill_task_order is not None
    q = args[0]
    assert isinstance(q, torch.Tensor)
    misaligned = list(args)
    offset_q = torch.empty(q.numel() + 1, dtype=q.dtype)[1:].view(q.shape)
    assert offset_q.is_contiguous() and offset_q.data_ptr() % 16 != 0
    misaligned[0] = offset_q
    invalid = kernel.bind(tuple(misaligned))
    assert invalid is not valid
    _assert_prefill_generic_search(invalid)
    assert kernel.bind(args) is valid
    assert kernel.bind(tuple(misaligned)) is invalid


@pytest.mark.parametrize("schedule", ["prefix_tail_2", "prefix_tail_4"])
def test_prefill_segmented_codegen(bound: BoundKernel, schedule: str) -> None:
    code = bound.to_triton_code(
        helion.Config(block_sizes=[64], cute_chunk_prefill_schedule=schedule)
    )
    assert "chunk_prefill_sm100" in code
    assert schedule in code
    assert "initial_state_idx" in code and "final_state_idx" in code


def test_prefill_schedule_rejects_unknown(bound: BoundKernel) -> None:
    with pytest.raises(InvalidConfig, match="must be one of"):
        bound.env.config_spec.normalized_config(
            helion.Config(block_sizes=[64], cute_chunk_prefill_schedule="unknown")
        )


@pytest.mark.parametrize(
    ("schedule", "order"),
    [
        ("prefix_tail_2", "identity"),
        ("prefix_tail_4", "longest_first"),
        ("single", "longest_first_precompute"),
        ("prefix_tail_2", "longest_first_precompute"),
    ],
)
def test_prefill_segmented_concurrent_streams_and_capture(
    schedule: str, order: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from helion.runtime.cute import chunk_prefill
    from helion.runtime.cute.launcher import cute_cuda_graph

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19)
    lengths = [0, 1, 17, 33, 513]
    tokens, heads = sum(lengths), 3
    shape = (1, tokens, heads, 128)
    q, k, v = (
        torch.randn(shape, generator=generator, device=device).bfloat16()
        for _ in range(3)
    )
    gate = torch.full_like(q, -8.0)
    beta = torch.randn(shape[:-1], generator=generator, device=device).bfloat16()
    a_log = torch.zeros(heads, device=device)
    dt = torch.zeros((heads, 128), device=device)
    initial = torch.randn(
        (len(lengths), heads, 128, 128), generator=generator, device=device
    ).mul_(0.1)
    output, final = torch.empty_like(v), torch.empty_like(initial)
    cu = torch.tensor([0, 0, 1, 18, 51, tokens], device=device, dtype=torch.int64)
    args = (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt,
        initial,
        output,
        final,
        cu,
        128**-0.5,
        -5 * 1.4426950408889634,
    )
    frozen = [tensor.clone() for tensor in (*args[:8], cu)]
    bound = kda_prefill_native_math._bind_isolated(args)
    single = bound.compile_config(helion.Config(block_sizes=[64]))
    single(*args)
    expected_output, expected_state = output.clone(), final.clone()
    compiled = bound.compile_config(
        helion.Config(
            block_sizes=[64],
            cute_chunk_prefill_schedule=schedule,
            cute_chunk_prefill_task_order=order,
        )
    )
    resources = []
    get_resources = chunk_prefill.prefill_resources

    def record_resources(*values):
        result = get_resources(*values)
        resources.append(result)
        return result

    monkeypatch.setattr(chunk_prefill, "prefill_resources", record_resources)
    compiled(*args)
    torch.cuda.synchronize()
    assert torch.equal(output, expected_output)
    assert torch.equal(final, expected_state)

    # Submit both calls before joining either origin. Their internal checkpoint
    # buffers must be independent even when all immutable input pointers match.
    origin = torch.cuda.current_stream()
    origins = [torch.cuda.Stream() for _ in range(2)]
    calls = []
    stream_resources = []
    for stream in origins:
        stream.wait_stream(origin)
        call = (
            *args[:8],
            torch.empty_like(output),
            torch.empty_like(final),
            *args[10:],
        )
        calls.append(call)
        with torch.cuda.stream(stream):
            compiled(*call)
        stream_resources.append(resources[-1])
    for stream in origins:
        origin.wait_stream(stream)
    torch.cuda.synchronize()
    assert stream_resources[0] is not stream_resources[1]
    assert (
        stream_resources[0].tensors[0].data_ptr()
        != stream_resources[1].tensors[0].data_ptr()
    )
    for call in calls:
        assert torch.equal(call[8], expected_output)
        assert torch.equal(call[9], expected_state)

    # A capture context gets separate ownership from eager warmup; poisoning
    # both checkpoints exposes any empty-prefix or tail that skips handoff.
    with cute_cuda_graph() as graph:
        compiled(*args)
    capture_resources = resources[-1]
    assert all(capture_resources is not resource for resource in stream_resources)
    for _ in range(5):
        output.fill_(float("nan"))
        final.fill_(float("nan"))
        for state in capture_resources.states:
            state.fill_(float("nan"))
        if capture_resources.order is not None:
            capture_resources.order.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected_output)
        assert torch.equal(final, expected_state)
        if capture_resources.order is not None:
            assert capture_resources.order.tolist() == sorted(
                range(len(lengths)), key=lambda i: -lengths[i]
            )
    if schedule == "single":
        assert capture_resources.states == () and capture_resources.streams == ()
    for tensor, snapshot in zip((*args[:8], cu), frozen, strict=True):
        assert torch.equal(tensor, snapshot)
    graph.reset()


def test_bt32_mixed_tails_and_task_orders() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(29)
    lengths = [0, 1, 31, 32, 33, 193]
    tokens, heads = sum(lengths), 8
    shape = (1, tokens, heads, 128)
    q, k, v = (
        torch.randn(shape, generator=generator, device=device).bfloat16()
        for _ in range(3)
    )
    gate = torch.full_like(q, -8.0)
    beta = torch.randn(shape[:-1], generator=generator, device=device).bfloat16()
    a_log = torch.zeros(heads, device=device)
    dt = torch.zeros((heads, 128), device=device)
    initial = torch.randn(
        (len(lengths), heads, 128, 128), generator=generator, device=device
    ).mul_(0.1)
    output, final = torch.empty_like(v), torch.empty_like(initial)
    cu = torch.tensor([0, 0, 1, 32, 64, 97, tokens], device=device, dtype=torch.int64)
    args = (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt,
        initial,
        output,
        final,
        cu,
        128**-0.5,
        -5 * 1.4426950408889634,
    )
    frozen = [tensor.clone() for tensor in (*args[:8], cu)]
    reference_output, reference_final = torch.empty_like(v), torch.empty_like(initial)
    reference_args = (
        *args[:8],
        reference_output,
        reference_final,
        *args[10:],
    )
    reference_kernel = helion.kernel(
        kda_prefill_native_math_bt32.fn,
        backend="triton",
        static_shapes=True,
        fast_math=True,
    )
    reference = reference_kernel._bind_isolated(reference_args).compile_config(
        helion.Config(
            block_sizes=[64],
            num_warps=4,
            num_stages=2,
            indexing="pointer",
            pid_type="flat",
        )
    )
    reference(*reference_args)
    bound = kda_prefill_native_math_bt32._bind_isolated(args)
    compiled = {
        order: bound.compile_config(
            helion.Config(
                block_sizes=[64],
                cute_chunk_prefill_schedule="single",
                cute_chunk_prefill_task_order=order,
            )
        )
        for order in ("identity", "longest_first_precompute")
    }
    compiled["identity"](*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference_output, atol=0.005, rtol=0.02)
    torch.testing.assert_close(final, reference_final, atol=0.01, rtol=0.02)
    assert torch.equal(final[0], initial[0])
    expected_output, expected_state = output.clone(), final.clone()
    for _ in range(5):
        output.fill_(float("nan"))
        final.fill_(float("nan"))
        compiled["longest_first_precompute"](*args)
        torch.cuda.synchronize()
        assert torch.equal(output, expected_output)
        assert torch.equal(final, expected_state)
    for tensor, snapshot in zip((*args[:8], cu), frozen, strict=True):
        assert torch.equal(tensor, snapshot)
