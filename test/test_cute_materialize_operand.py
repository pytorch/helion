"""CPU semantic and admission tests for general operand materialization."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_materialized_fission import _CpuRefMode

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.ref_mode import RefMode
from helion.runtime.ref_mode import RefModeContext

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


class _OperandCpuRefMode(_CpuRefMode):
    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: list[type[object]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> Any:
        if func is torch.mm and kwargs is not None and "out_dtype" in kwargs:
            lhs, rhs = args
            assert isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor)
            dtype = kwargs["out_dtype"]
            assert dtype is torch.float32
            return torch.mm(lhs.float(), rhs.float())
        return super().__torch_function__(func, types, args, kwargs)


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _computed_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = (b[inner, column].float() * 0.25 + 2).to(a.dtype)
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _int16_bare_cast_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[row, inner], b[inner, column].to(a.dtype), acc=acc)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _int16_bare_cast_lhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=b.dtype, device=b.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[row, inner].to(b.dtype), b[inner, column], acc=acc)
        out[row, column] = acc.to(b.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _int16_derived_cast_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = (b[inner, column] + 1).to(a.dtype)
            acc = hl.dot(a[row, inner], transformed, acc=acc)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _interleaved_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    physical_k = hl.register_block_size(k // 2)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k // 2, block_size=physical_k):
            start = inner.begin * 2
            left = a[row, start : start + physical_k * 2]
            value = b[inner, column].float()
            stacked = torch.stack(
                [(value + 1).to(a.dtype), (value * 2).to(a.dtype)], dim=1
            )
            right = stacked.reshape(inner.block_size * 2, column.block_size)
            acc = hl.dot(left, right, acc=acc)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _incomplete_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.zeros((m + 1, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * 2
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _nonzero_seed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.full([row, column], 1, dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * 2
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _tile_reversal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = torch.flip(b[inner, column], (0,))
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _aliased_output(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * 2
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _direct_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _rebound_input(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    b = b.reshape(b.shape)
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * 2
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _rebound_capture(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    dead = torch.empty_like(b)
    dead = 2
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * dead
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


_HOST_CALLBACKS: list[tuple[int, ...]] = []


def _opaque_allocation(a: torch.Tensor, m: int, n: int) -> torch.Tensor:
    _HOST_CALLBACKS.append((m, n))
    return torch.empty((m, n), dtype=a.dtype, device=a.device)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _opaque_prelude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = _opaque_allocation(a, m, n)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            transformed = b[inner, column] * 2
            acc = torch.addmm(acc, a[row, inner], transformed)
        out[row, column] = acc.to(a.dtype)
    return out


def _bind(
    kernel: Kernel[Any],
    args: tuple[torch.Tensor, ...],
    *,
    static_shapes: bool = False,
    enabled: bool = True,
) -> BoundKernel[Any]:
    return helion.kernel(
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        cute_materialize_transformed_operands=enabled,
    )(kernel.fn)._bind_isolated(args)


def _args(
    dtype: torch.dtype = torch.bfloat16,
    shape: tuple[int, int, int] = (512, 1024, 512),
    *,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k, n = shape
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty(
            (k // 2 if packed else k, n), dtype=torch.int8 if packed else dtype
        ),
    )


def _reference(
    bound: BoundKernel[Any],
    args: tuple[torch.Tensor, ...],
    *,
    transformed: bool,
    materialized_operand: bool = False,
) -> torch.Tensor:
    if transformed:
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            definition = bound.host_function.codegen_function_def(
                bound.host_function.body
            )
        definition = ast.parse(ast.unparse(ast.fix_missing_locations(definition))).body[
            0
        ]
        assert isinstance(definition, ast.FunctionDef)
        roots = [node for node in definition.body if isinstance(node, ast.For)]
        assert len(roots) == 2
        for root, blocks in zip(roots, ((4, 8), (8, 4)), strict=True):
            assert isinstance(root.iter, ast.Call)
            root.iter.keywords = [
                ast.keyword(
                    arg="block_size",
                    value=ast.List(
                        elts=[ast.Constant(value=value) for value in blocks],
                        ctx=ast.Load(),
                    ),
                )
            ]
        contraction = roots[1].body[1]
        assert isinstance(contraction, ast.For) and isinstance(
            contraction.iter, ast.Call
        )
        contraction.iter.keywords = [
            ast.keyword(arg="block_size", value=ast.Constant(value=8))
        ]
        if materialized_operand:
            plan = bound.env.cute_fission_plan
            assert plan is not None
            definition.body.remove(roots[1])
            result = definition.body[-1]
            assert isinstance(result, ast.Return)
            result.value = ast.Name(id=plan.materialized_names[-1], ctx=ast.Load())
        namespace = dict(bound.kernel.fn.__globals__)
        namespace["_default_cute_launcher"] = None
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[definition], type_ignores=[])
                ),
                "<materialization-reference>",
                "exec",
            ),
            namespace,
        )
        function = namespace[definition.name]
    else:
        function = bound.kernel.fn
    env = CompileEnvironment(
        torch.device("cpu"), dataclasses.replace(bound.settings, ref_mode=RefMode.EAGER)
    )
    context = RefModeContext(env, None)
    context.func_mode = _OperandCpuRefMode()
    with context:
        result = function(*args)
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("static_shapes", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_original_packed_example_has_independent_axes_and_guarded_shapes(
    static_shapes: bool, dtype: torch.dtype
) -> None:
    args = _args(dtype, packed=True)
    bound = _bind(matmul_bf16_int4, args, static_shapes=static_shapes)
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.region_count == 2
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) == 2
    assert len(bound.env.block_sizes) == 5
    assert plan.materialized_names == ("C", "_helion_materialized_operand")
    if not static_shapes:
        assert bound.env.specialized_vars
        assert bound.env.shape_env.guards
    fact = bound.config_spec.matmul_facts[0]
    assert (fact.m_block_id, fact.n_block_id, fact.k_block_id) == (2, 3, 4)
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        first_flat, first_config = generation.canonicalize_flat(
            generation.default_flat()
        )
        assert (
            len(first_config.l2_groupings) == len(bound.config_spec.l2_groupings) == 2
        )
        assert generation.canonicalize_flat(first_flat) == (first_flat, first_config)
    assert bound.config_spec._cute_tcgen05_config.pipeline_smem_facts is not None
    original = _bind(matmul_bf16_int4, args, static_shapes=static_shapes, enabled=False)
    assert original.env.cute_fission_plan is None
    assert (
        bound.config_spec.cache_fingerprint_hash()
        != original.config_spec.cache_fingerprint_hash()
    )


@pytest.mark.parametrize("kernel", (_computed_rhs, _interleaved_rhs, matmul_bf16_int4))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_cpu_reference_preserves_values_casts_and_unequal_stage_tiles(
    kernel: Kernel[Any], dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(1904)
    a = torch.randint(-3, 4, (13, 40), generator=generator).to(dtype) * 0.125
    if kernel is matmul_bf16_int4:
        b = torch.randint(-128, 128, (20, 19), dtype=torch.int8, generator=generator)
    else:
        k = 20 if kernel is _interleaved_rhs else 40
        b = torch.randint(-8, 8, (k, 19), generator=generator).to(dtype) * 0.5
    args = (a, b)
    bound = _bind(kernel, args)
    assert bound.env.cute_fission_plan is not None
    actual = _reference(bound, args, transformed=True)
    expected = _reference(bound, args, transformed=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_materialized_interleave_preserves_every_packed_byte(
    dtype: torch.dtype,
) -> None:
    packed = (torch.arange(256, dtype=torch.int16) - 128).to(torch.int8)
    packed = packed[:, None].expand(256, 19).contiguous()
    args = (torch.empty((1, 512), dtype=dtype), packed)
    bound = _bind(matmul_bf16_int4, args)
    assert bound.env.cute_fission_plan is not None
    actual = _reference(bound, args, transformed=True, materialized_operand=True)
    low = (packed & 15).int()
    low = torch.where(low >= 8, low - 16, low)
    high = (packed >> 4).int()
    expected = torch.stack((low.to(dtype), high.to(dtype)), dim=1).reshape(512, 19)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kernel",
    (
        _incomplete_output,
        _nonzero_seed,
        _tile_reversal,
        _direct_rhs,
        _rebound_input,
        _rebound_capture,
    ),
)
def test_unsupported_memory_and_tile_semantics_keep_original_path(
    kernel: Kernel[Any],
) -> None:
    args = _args(shape=(16, 32, 24))
    bound = _bind(kernel, args)
    assert bound.env.cute_fission_plan is None
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) == 1


def test_bare_integer_cast_of_loaded_tile_is_materialized() -> None:
    """``b[k, n].to(bf16)`` on an int16 input rounds every loaded value exactly as
    the program does, so the range proof required for derived integer recipes
    does not apply; the derived ``(b + 1).to(bf16)`` recipe still needs it."""
    m, k, n = 128, 256, 128
    args = (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.int16),
    )
    bound = _bind(_int16_bare_cast_rhs, args)
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.region_count == 2
    assert bound.config_spec.cute_tcgen05_search_enabled
    derived = _bind(_int16_derived_cast_rhs, args)
    assert derived.env.cute_fission_plan is None


def test_bare_integer_cast_on_the_lhs_is_materialized() -> None:
    """An ``[M, K]`` recipe on the LHS is reused across the output columns:
    the producer tiles ``[M, K]`` and the consumer contracts
    ``matrix[m, k] @ b[k, n]`` with the direct RHS input."""
    m, k, n = 128, 256, 128
    args = (
        torch.empty((m, k), dtype=torch.int16),
        torch.empty((k, n), dtype=torch.bfloat16),
    )
    bound = _bind(_int16_bare_cast_lhs, args)
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.region_count == 2
    assert plan.operand_interleave_factor == 1
    assert bound.config_spec.cute_tcgen05_search_enabled
    assert plan.replacement_body is not None
    producer = ast.unparse(plan.replacement_body[-3])
    assert "_helion_materialize_m, _helion_materialize_k" in producer
    assert (
        "_helion_materialized_operand[_helion_materialize_m, _helion_materialize_k]"
        in producer
    )
    consumer = ast.unparse(plan.replacement_body[-2])
    assert (
        "_helion_materialized_operand[row, _helion_materialize_logical_k]" in consumer
    )
    assert "b[_helion_materialize_logical_k, column]" in consumer


def test_input_output_aliases_are_not_materialized() -> None:
    args = _args(shape=(16, 32, 32))
    bound = _bind(_aliased_output, (*args, args[0]))
    assert bound.env.cute_fission_plan is None


def test_incomplete_output_keeps_its_observable_zero_initialization() -> None:
    args = tuple(torch.ones_like(value) for value in _args(shape=(3, 8, 5)))
    bound = _bind(_incomplete_output, args)
    assert bound.env.cute_fission_plan is None
    output = _reference(bound, args, transformed=False)
    assert torch.count_nonzero(output[-1]).item() == 0


def test_consumer_budget_and_seed_serialization_ignore_producer_store_rings() -> None:
    args = _args(packed=True)
    bound = _bind(matmul_bf16_int4, args)
    spec = bound.config_spec
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        seeds = [
            seed
            for seed in spec.autotune_seed_configs()
            if seed.config.get("tcgen05_cta_group") == "two"
        ]
        selected = next(
            seed
            for seed in seeds
            if seed.block_sizes[2:] == [128, 128, 64]
            and seed.config["tcgen05_ab_stages"] == 13
        )
        config = bound._normalized_config_copy(
            helion.Config.from_dict(spec.default_config().config | selected.config)
        )
        generation = ConfigGeneration(spec)
        assert generation.unflatten(generation.flatten(config)) == config
    code = bound.to_code(config)
    module = ast.parse(code)
    sources = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PyCodeCache"
        and node.func.attr == "load"
    ]
    assert len(sources) == 2
    assert "CtaGroup.TWO" not in sources[0]
    assert "CtaGroup.TWO" in sources[1]
    assert "num_stages=13" in sources[1]
    wrapper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "matmul_bf16_int4"
    )
    allocations = [
        node
        for node in wrapper.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "torch.empty"
    ]
    assert len(allocations) == 2
    assert all("torch.zeros" not in ast.unparse(node) for node in wrapper.body)


def test_dynamic_shape_specializations_separate_bound_kernel_cache_entries() -> None:
    kernel = helion.kernel(
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )(matmul_bf16_int4.fn)
    args = _args(shape=(16, 40, 24), packed=True)
    bound = kernel.bind(args)
    extractors = bound._specialize_extra()
    assert extractors
    key = tuple(extract(args) for extract in extractors)
    same_args = tuple(value.clone() for value in args)
    assert tuple(extract(same_args) for extract in extractors) == key
    assert kernel.bind(same_args) is bound
    for shape in ((17, 40, 24), (16, 48, 24), (16, 40, 32)):
        other_args = _args(shape=shape, packed=True)
        assert tuple(extract(other_args) for extract in extractors) != key
        other = kernel.bind(other_args)
        assert other is not bound
        assert other.env.cute_fission_plan is not None


def test_opaque_host_callback_is_not_executed_by_speculative_analysis() -> None:
    args = _args(shape=(16, 32, 24))
    _HOST_CALLBACKS.clear()
    _bind(_opaque_prelude, args, enabled=False)
    original_calls = len(_HOST_CALLBACKS)
    assert original_calls > 0
    _HOST_CALLBACKS.clear()
    bound = _bind(_opaque_prelude, args)
    assert bound.env.cute_fission_plan is None
    assert len(_HOST_CALLBACKS) == original_calls


def test_native_bundle_retains_producer_layout_in_search_and_vector_codegen() -> None:
    args = _args(packed=True)
    bound = _bind(matmul_bf16_int4, args)
    assert bound.host_function is not None
    spec = bound.config_spec
    assert spec.cute_pointwise_region_block_ids == {0, 1}
    with bound.env, bound.host_function:
        seed = next(
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("tcgen05_cta_group") == "two"
            and seed.block_sizes[2:] == [128, 128, 64]
            and seed.config.get("tcgen05_ab_stages") == 13
            and spec.cute_vector_widths.config_get(
                cast("list[int]", seed.config.get("cute_vector_widths", [])), 1
            )
            == 8
        )
        config = bound._normalized_config_copy(
            helion.Config.from_dict(spec.default_config().config | seed.config)
        )
        generation = ConfigGeneration(spec)
        roundtrip = generation.unflatten(generation.flatten(config))
        assert roundtrip == config
        for block_id in (2, 3, 4):
            assert spec.num_threads.block_id_lookup(block_id)._fragment(
                spec
            ).search_values() == [0]
            assert spec.cute_vector_widths.block_id_lookup(block_id)._fragment(
                spec
            ).search_values() == [1]
    code = bound.to_code(config)
    sources = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PyCodeCache"
        and node.func.attr == "load"
    ]
    assert len(sources) == 2
    assert "cutlass.range_constexpr(8)" in sources[0]
    assert "CtaGroup.TWO" in sources[1]
    assert "cutlass.range_constexpr(8)" not in sources[1]


def _stage_sources(code: str) -> list[str]:
    return [
        ast.literal_eval(node.args[0])
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PyCodeCache"
        and node.func.attr == "load"
    ]


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [
        (matmul_bf16_int4, (512, 1024, 512)),
        (matmul_bf16_int4, (1024, 2048, 1024)),
        (_computed_rhs, (128, 256, 128)),
    ],
)
def test_flat_materialization_seed_preserves_the_native_consumer(
    kernel: Kernel[Any], shape: tuple[int, int, int]
) -> None:
    bound = _bind(kernel, _args(shape=shape, packed=kernel is matmul_bf16_int4))
    spec = bound.config_spec
    with bound.env:
        generation = ConfigGeneration(spec)
        flat = next(
            seed
            for _, seed in generation.seed_flat_config_pairs()
            if seed.config.get("cute_pointwise_pid_type") == "flat"
            and seed.config.get("tcgen05_cta_group") == "two"
        )
        # The producer seed owns one 16-byte packet per thread in a 128-thread
        # CTA; the flat variant keeps that geometry and only changes the grid.
        threads = cast("list[int]", flat.config["num_threads"])
        row_threads = spec.num_threads.config_get(threads, 0)
        column_threads = spec.num_threads.config_get(threads, 1)
        assert flat.block_sizes[0] == row_threads >= 1
        assert row_threads * column_threads == 128
        assert generation.unflatten(generation.flatten(flat)) == flat
        assert any(
            generation.unflatten(config).config.get("cute_pointwise_pid_type") == "flat"
            for config in generation.random_population_flat(30)
        )
        inherited_values = flat.config.copy()
        inherited_values.pop("cute_pointwise_pid_type")
        inherited = helion.Config.from_dict(inherited_values)
        explicit_inherit = helion.Config.from_dict(
            inherited_values | {"cute_pointwise_pid_type": "inherit"}
        )
        assert spec.normalized_config(explicit_inherit) == inherited
        assert "cute_pointwise_pid_type" not in spec.default_config().config
    flat_sources = _stage_sources(bound.to_code(flat))
    inherited_sources = _stage_sources(bound.to_code(inherited))
    assert len(flat_sources) == len(inherited_sources) == 2
    assert flat_sources[1] == inherited_sources[1]
    assert "for virtual_pid in" in inherited_sources[0]
    assert "for virtual_pid in" not in flat_sources[0]
    assert "cute.arch.block_idx()[0]" in flat_sources[0]
    assert "cute_pointwise_pid_type" not in "".join(flat_sources)


@pytest.mark.parametrize("value", ["flat", "persistent_blocked", False, 1, None])
def test_pointwise_pid_override_requires_a_proved_region(value: object) -> None:
    bound = _bind(matmul_bf16_int4, _args(packed=True), enabled=False)
    assert bound.env.cute_fission_plan is None
    with bound.env:
        config = helion.Config.from_dict(
            bound.config_spec.default_config().config
            | {"cute_pointwise_pid_type": value}
        )
        with pytest.raises(InvalidConfig, match="proved pointwise region"):
            bound.config_spec.normalized_config(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert "cute_pointwise_pid_type" not in config.config


def _int16_args(
    shape: tuple[int, int, int], *, lhs: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k, n = shape
    if lhs:
        return (
            torch.empty((m, k), dtype=torch.int16),
            torch.empty((k, n), dtype=torch.bfloat16),
        )
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.int16),
    )


def _producer_seed(bound: BoundKernel[Any], **wanted: object) -> helion.Config:
    spec = bound.config_spec
    for seed in spec.compiler_seed_configs:
        values = seed.config
        widths = cast("list[int]", values.get("cute_vector_widths", []))
        if spec.cute_vector_widths.config_get(widths, 1) != 8:
            continue
        if all(values.get(key) == value for key, value in wanted.items()):
            return bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | values)
            )
    raise AssertionError(f"no producer seed matches {wanted}")


@pytest.mark.parametrize("lhs", [False, True], ids=["rhs_int16", "lhs_int16"])
def test_int16_cast_producer_loads_vector_packets(lhs: bool) -> None:
    kernel = _int16_bare_cast_lhs if lhs else _int16_bare_cast_rhs
    bound = _bind(kernel, _int16_args((128, 256, 128), lhs=lhs))
    assert bound.env.cute_fission_plan is not None
    # Both recipes prove the producer/consumer edge, so their seeds carry PDL.
    assert bound.config_spec._cute_tcgen05_config.materialized_operand_pdl_roots
    with bound.env:
        config = _producer_seed(
            bound, tcgen05_cta_group="two", tcgen05_materialized_pdl=True
        )
    producer, consumer = _stage_sources(bound.to_code(config))
    assert "cute.arch.griddepcontrol_launch_dependents()" in producer
    assert "cute.arch.griddepcontrol_wait()" in consumer
    # One 16-byte packet of int16 words per thread, each lane bitcast back to
    # Int16 before the program's own bf16 cast; the store keeps its u16 packet.
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in producer
    assert ".bitcast(cutlass.Int16)" in producer
    assert "cutlass.BFloat16(" in producer
    assert "_cute_store_u16_vec(" in producer
    assert ".load()" not in producer
    assert "CtaGroup.TWO" in consumer


def test_producer_seed_geometry_is_one_packet_per_thread_in_a_warp_quartet() -> None:
    bound = _bind(_int16_bare_cast_rhs, _int16_args((128, 256, 128)))
    spec = bound.config_spec
    with bound.env:
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if spec.cute_vector_widths.config_get(
                cast("list[int]", seed.config.get("cute_vector_widths", [])), 1
            )
            == 8
        ]
        assert seeds
        for seed in seeds:
            threads = cast("list[int]", seed.config["num_threads"])
            row_threads = spec.num_threads.config_get(threads, 0)
            column_threads = spec.num_threads.config_get(threads, 1)
            # 128 columns hold 16 packets, so eight rows stack across threads.
            assert (row_threads, column_threads) == (8, 16)
            assert seed.block_sizes[:2] == [8, 128]
            # Every paired seed asks for the programmatic dependent launch.
            assert seed.config.get("tcgen05_materialized_pdl") is True
        assert {seed.config.get("cute_pointwise_pid_type") for seed in seeds} == {
            None,
            "flat",
        }


def test_small_grid_seeds_add_one_cta_tiles_with_pdl() -> None:
    with patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148):
        bound = _bind(_int16_bare_cast_rhs, _int16_args((128, 256, 128)))
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.device_sm_count == 148
    with bound.env:
        one_cta = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("tcgen05_cta_group") == "auto"
        ]
        # A 128x128 output is one standard tile: 128-row tiles with 16/32/64
        # columns spread the operand re-reads over 16/8/4 CTAs, and the 64-row
        # tiles halve each CTA's A tile over twice the CTAs; both row heights
        # are seeded because the 64-row grids (32/16/8 CTAs) stay within one
        # wave of the 148 SMs.  The one-wave seed adds the same 64x32 tile
        # (the one-tile-per-CTA grid closest to the SM count) at the narrowest
        # K step with the whole K loop in flight (ab_stages = 256 / 64).
        assert sorted({tuple(seed.block_sizes[2:]) for seed in one_cta}) == [
            (64, 16, 128),
            (64, 32, 64),
            (64, 32, 128),
            (64, 64, 128),
            (128, 16, 128),
            (128, 32, 128),
            (128, 64, 128),
        ]
        for seed in one_cta:
            assert seed.config["pid_type"] == "persistent_interleaved"
            assert seed.config["tcgen05_cluster_m"] == 1
            assert seed.config["tcgen05_ab_stages"] == (
                4 if seed.block_sizes[4] == 64 else 2
            )
        # The producer-layout carriers of those seeds ask for the programmatic
        # dependent launch; the bare native seeds keep the search default.
        assert all(
            seed.config.get("tcgen05_materialized_pdl") is True
            for seed in one_cta
            if spec.cute_vector_widths.config_get(
                cast("list[int]", seed.config.get("cute_vector_widths", [])), 1
            )
            == 8
        )
        config = _producer_seed(bound, tcgen05_cta_group="auto")
        assert config.block_sizes[2:] == [64, 16, 128]
        assert config.config["tcgen05_materialized_pdl"] is True
        # The seed is representable in the search schema (the optional matmul
        # min-blocks coordinate has no public default key, so compare flat).
        generation = ConfigGeneration(spec)
        flat = generation.flatten(config)
        assert generation.flatten(generation.unflatten(flat)) == flat
    producer, consumer = _stage_sources(bound.to_code(config))
    assert "cute.arch.griddepcontrol_launch_dependents()" in producer
    assert "CtaGroup.ONE" in consumer
    assert "'use_pdl': True" in consumer
    # The one-CTA consumer issues the ordinary input's initial TMA loads before
    # the dependency wait; only the materialized operand's loads follow it.
    wait = "cute.arch.griddepcontrol_wait()"
    assert consumer.count(wait) == 1
    assert consumer.index("cute.copy(tma_atom_a") < consumer.index(wait)
    assert consumer.index(wait) < consumer.index("cute.copy(tma_atom_b")
    assert "tcgen05_pdl_producer_state = tcgen05_ab_producer_state.clone()" in consumer


def test_small_grid_seeds_absent_when_standard_tiles_fill_the_device() -> None:
    with patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148):
        bound = _bind(_int16_bare_cast_rhs, _int16_args((2048, 1024, 2048)))
    spec = bound.config_spec
    with bound.env:
        assert spec._cute_tcgen05_config.device_sm_count == 148
        # 256 standard tiles already fill 148 SMs: no small one-CTA tiles.
        assert not any(
            seed.config.get("tcgen05_cta_group") == "auto" and seed.block_sizes[2] == 64
            for seed in spec.compiler_seed_configs
        )
        assert any(
            seed.config.get("tcgen05_cta_group") == "two"
            for seed in spec.compiler_seed_configs
        )


def test_one_cta_pdl_requires_the_persistent_role_local_schedule() -> None:
    with patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148):
        bound = _bind(_int16_bare_cast_rhs, _int16_args((128, 256, 128)))
    spec = bound.config_spec
    with bound.env:
        config = _producer_seed(bound, tcgen05_cta_group="auto")
        flat = helion.Config.from_dict(
            config.config
            | {
                "pid_type": "flat",
                "tcgen05_persistence_model": "non_persistent",
                "tcgen05_materialized_pdl": True,
            }
        )
        with pytest.raises(InvalidConfig, match="dependency-wait schedule"):
            spec.normalized_config(flat)
        spec.normalize(flat, _fix_invalid=True)
        assert "tcgen05_materialized_pdl" not in flat.config
