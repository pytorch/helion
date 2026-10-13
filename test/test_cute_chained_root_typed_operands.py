from __future__ import annotations

import ast
from collections import Counter
import importlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_workspace import _assert_safe
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_root_typed_operands import _cast_path
from helion._compiler.cute.chained_tcgen05 import _shared_memory_bytes
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chain(
    left: torch.Tensor,
    right: torch.Tensor,
    initial: torch.Tensor,
    seed: hl.constexpr,
    mixed: hl.constexpr,
    transpose: hl.constexpr,
    fanout: hl.constexpr,
) -> torch.Tensor:
    out = torch.empty_like(initial)
    for batch in hl.tile(left.size(0), block_size=1):
        idx = batch.id
        first = hl.dot(left[idx, :, :], right[idx, :, :])
        narrow = first.to(torch.float16)
        if transpose:
            rhs = narrow.T
        else:
            rhs = narrow
        if seed:
            second = hl.dot(narrow, rhs, acc=initial[idx, :, :])
        else:
            second = hl.dot(narrow, rhs)
        if mixed:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        if seed:
            third = hl.dot(second.to(dtype), first.to(dtype), acc=second)
        else:
            third = hl.dot(second.to(dtype), first.to(dtype))
        if fanout:
            third = third + first * 0.125
        out[idx, :, :] = third
    return out


def _args(
    size: int = 64,
    *,
    seed: bool = True,
    mixed: bool = True,
    transpose: bool = True,
    fanout: bool = True,
    device: str = "cpu",
) -> tuple[object, ...]:
    return (
        torch.empty((2, size, size), dtype=torch.bfloat16, device=device),
        torch.empty((2, size, size), dtype=torch.bfloat16, device=device),
        torch.empty((2, size, size), dtype=torch.float32, device=device),
        seed,
        mixed,
        transpose,
        fanout,
    )


def _config(enabled: bool = True, /, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_typed_operands": enabled,
            **overrides,
        }
    )


def _code(
    args: tuple[object, ...], config: helion.Config | None = None
) -> tuple[str, ChainedMatmulPlan]:
    plans: list[ChainedMatmulPlan] = []
    original = chained_matmul.plan_chained_matmul

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            assert _shared_memory_bytes(plan) <= 232448
            plans.append(plan)
        return plan

    with _cpu_codegen(), patch.object(chained_matmul, "plan_chained_matmul", observe):
        bound = _chain._bind_isolated(args)
        assert bound.config_spec.cute_chained_typed_operands_search_enabled
        source = bound.to_code(config or _config())
    assert len(plans) == 1
    return source, plans[0]


@pytest.mark.parametrize(
    "seed,fanout", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("transpose,mixed", [(False, False), (True, True)])
def test_codegen_preserves_seed_and_fanout_publications(
    seed: bool, fanout: bool, transpose: bool, mixed: bool
) -> None:
    source, plan = _code(
        _args(seed=seed, fanout=fanout, transpose=transpose, mixed=mixed)
    )
    typed = plan.root_typed_operands
    assert typed is not None
    _assert_safe(typed.operands)
    _assert_safe(typed.fp32)
    assert len(typed.images) == (3 if mixed else 2)
    assert {region.name for region in typed.fp32.regions} == (
        {"chain_0_c"} if fanout else set()
    ) | ({"chain_1_c"} if seed else set())
    assert "chain_1_a_step" not in source
    assert "chain_1_b_step" not in source
    assert "chain_2_a_step" not in source
    assert "chain_2_b_step" not in source
    assert source.count("cute.autovec_copy(chain_typed_") == len(typed.images)
    assert ("chain_c_workspace =" in source) == (seed or fanout)
    # Every aliased publication follows the completed TMEM drain and CTA join.
    module = ast.parse(source)
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion__chain"
    )
    for image in typed.images:
        index = next(
            i
            for i, node in enumerate(kernel.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == f"{image.name}_values"
                for target in node.targets
            )
        )
        previous = ast.unparse(kernel.body[index - 1])
        assert previous == "cute.arch.sync_threads()" or previous.startswith(
            "cute.autovec_copy(chain_typed_"
        )


def test_default_off_preserves_source() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _chain._bind_isolated(args)
        original = bound.to_code(
            helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
        )
        disabled = bound.to_code(_config(False))
    assert original == disabled


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tail_chain(
    a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor, valid: int
) -> torch.Tensor:
    valid = hl.specialize(valid)
    out = torch.empty_like(initial)
    for tile in hl.tile(a.size(0), block_size=1):
        idx = tile.id
        columns = hl.arange(64)
        rhs = hl.load(
            b, [idx, hl.arange(128), columns], extra_mask=columns[None, :] < valid
        )
        first = hl.dot(a[idx, :, :], rhs)
        narrow = first.to(torch.float16)
        out[idx, :, :] = hl.dot(narrow, narrow.T, acc=initial[idx, :, :])
    return out


def _tail_args(size: int, device: str = "cpu") -> tuple[object, ...]:
    return (
        torch.empty((2, 64, 128), dtype=torch.bfloat16, device=device),
        torch.empty((2, 128, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 64, 64), dtype=torch.float32, device=device),
        size,
    )


@pytest.mark.parametrize("size", [33, 63])
def test_partial_logical_domain_remains_in_typed_publication(size: int) -> None:
    with _cpu_codegen():
        bound = _tail_chain._bind_isolated(_tail_args(size))
        source = bound.to_code(_config())
    assert f"= {size}\n" in source
    assert "operator.lt(" in source
    assert "chain_typed_0_values" in source


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_chained_vector_group", True),
        ("cute_chained_fragment_epilogues", True),
        ("cute_chained_startup_transfer", "tma"),
        ("cute_chained_pointwise_read_cache", True),
        ("cute_chained_tmem_early_release", True),
        ("cute_chained_k_schedule", "serial64"),
        ("num_warps", 8),
        ("cute_cluster_n", 2),
    ],
)
def test_rejects_unproved_ownership_interactions(key: str, value: object) -> None:
    with _cpu_codegen():
        bound = _chain._bind_isolated(_args())
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_config(**{key: value}))


@pytest.mark.parametrize("kind", ["arithmetic", "extra_cast", "reshape"])
def test_rejects_non_cast_permutation_paths(kind: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (64, 64), torch.bfloat16)
    right = _input(graph, "right", (64, 64), torch.bfloat16)
    first = _dot(graph, left, right)
    value = _convert(graph, first, torch.float16)
    if kind == "arithmetic":
        value = _call(
            graph, torch.ops.aten.mul.Scalar, (value, 2), (64, 64), torch.float16
        )
    elif kind == "extra_cast":
        value = _convert(graph, _convert(graph, value, torch.float32), torch.float16)
    else:
        value = _call(
            graph,
            torch.ops.aten.reshape.default,
            (value, (64, 64)),
            (64, 64),
            torch.float16,
        )
    second = _dot(graph, value, _convert(graph, right, torch.float16))
    assert _cast_path((first, second), 1, 0) is None


@pytest.mark.parametrize("keep", ["seed", "fanout", "export", "none"])
def test_skip_reads_never_omits_other_fp32_consumers(keep: str) -> None:
    graph = Graph()
    left = _input(graph, "left", (64, 64), torch.bfloat16)
    right = _input(graph, "right", (64, 64), torch.bfloat16)
    first = _dot(graph, left, right)
    second = _dot(
        graph,
        _convert(graph, first, torch.bfloat16),
        right,
        first if keep == "seed" else None,
    )
    third = _dot(
        graph,
        _convert(graph, second, torch.bfloat16),
        right,
        first if keep == "fanout" else None,
    )
    plan = _plan(graph, (first, third) if keep == "export" else (third,))
    workspace = plan_contraction_workspace(
        plan,
        include_final=False,
        skip_operand_reads=frozenset({(1, 0), (2, 0)}),
        omit_unused=True,
    )
    assert bool(workspace.regions) == (keep != "none")
    if keep != "none":
        assert len(workspace.regions) == 1
        assert workspace.regions[0].name == "chain_0_c"
        assert (
            workspace.regions[0].live_until
            == {"seed": 3, "fanout": 5, "export": 7}[keep]
        )
    _assert_safe(workspace)


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize("width", [32, 64, 96, 128, 256])
def test_actual_cute_native_transpose_and_sparse_publication(
    dtype_name: str, width: int
) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    tcgen = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
    helpers = importlib.import_module("cutlass.utils.blackwell_helpers")
    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = getattr(cutlass, dtype_name)
    atom = min(128, (width * 2) & -(width * 2))
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            a = cute.tile_to_shape(
                tcgen.make_smem_layout_atom(
                    getattr(tcgen.SmemLayoutAtomKind, f"K_SW{atom}"), dtype
                ),
                (64, width),
                order=(0, 1),
            )
            b = cute.tile_to_shape(
                tcgen.make_smem_layout_atom(
                    getattr(tcgen.SmemLayoutAtomKind, f"MN_SW{atom}"), dtype
                ),
                (width, 64),
                order=(1, 0),
            )
            # Pointer swizzles act on byte offsets, whereas tensor layouts
            # carry element offsets. Compare the actual native maps in both.
            assert str(a.inner) == str(b.inner)
            assert cute.cosize(a.outer) == cute.cosize(b.outer) == 64 * width
            for row in range(64):
                for column in range(width):
                    assert a.outer((row, column)) == b.outer((column, row))
            offsets = [
                int(a.outer((row, column)))
                for row in range(64)
                for column in range(width)
            ]
            bits, base, shift = a.inner.num_bits, a.inner.num_base, a.inner.num_shift
            mask = ((1 << bits) - 1) << base
            assert base >= 0 and shift > 0 and base + bits <= 7
            for phase in range(16):
                start = phase * 128
                addresses = [start + 2 * offset for offset in offsets]
                swizzled = [
                    address ^ ((address >> shift) & mask) for address in addresses
                ]
                assert sorted(swizzled) == list(range(start, start + 2 * 64 * width, 2))
            mma = helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                cutlass.Float32,
                tcgen.CtaGroup.ONE,
                (64, width),
                tcgen.OperandSource.SMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                mma.make_fragment_C(mma.partition_shape_C((64, width))).layout,
            )
            copy = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen.Ld16x256bOp(tcgen.Repetition(min(8, (width & -width) // 8))),
                    cutlass.Float32,
                ),
                acc,
            )
            identity = mma.get_slice(0).partition_C(
                cute.make_identity_tensor((64, width))
            )
            target = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(dtype, 0, cute.AddressSpace.smem, assumed_align=128),
                    a.inner,
                    dtype=dtype,
                ),
                a.outer,
            )
            owners = []
            for thread in range(128):
                sliced = copy.get_slice(thread)
                coords = sliced.partition_D(identity)
                destinations = sliced.partition_D(mma.get_slice(0).partition_C(target))
                assert cute.size(coords) == cute.size(destinations)
                owners.extend(
                    tuple(coords[index]) for index in range(cute.size(coords))
                )
            assert Counter(owners) == Counter(
                (r, c) for r in range(64) for c in range(width)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize(
    "size,seed,mixed,transpose,fanout",
    [
        (64, True, True, True, True),
        (64, False, False, False, False),
        (64, True, False, True, True),
        (64, False, True, False, True),
    ],
)
def test_typed_chain_gpu(
    size: int, seed: bool, mixed: bool, transpose: bool, fanout: bool
) -> None:
    args = _args(
        size, seed=seed, mixed=mixed, transpose=transpose, fanout=fanout, device="cuda"
    )
    left, right, initial = args[:3]
    assert (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and isinstance(initial, torch.Tensor)
    )
    generator = torch.Generator(device="cuda").manual_seed(551)
    for value in (left, right, initial):
        value.normal_(generator=generator).mul_(0.025)
    first = (left.double() @ right.double()).float()
    narrow = first.half().double()
    second = (
        narrow @ (narrow.transpose(-1, -2) if transpose else narrow)
        + (initial.double() if seed else 0)
    ).float()
    dtype = torch.bfloat16 if mixed else torch.float16
    third = (
        second.to(dtype).double() @ first.to(dtype).double()
        + (second.double() if seed else 0)
    ).float()
    expected = third + first * 0.125 if fanout else third
    bound = _chain._bind_isolated(args)
    bound.set_config(_config())
    actual = bound(*args)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("size", [33, 63])
def test_typed_tail_chain_gpu(size: int) -> None:
    args = _tail_args(size, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(557)
    left, right, initial = args[:3]
    assert (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and isinstance(initial, torch.Tensor)
    )
    for value in (left, right, initial):
        value.normal_(generator=generator).mul_(0.025)
    masked = right.double().clone()
    masked[:, :, size:] = 0
    first = (left.double() @ masked).float().half().double()
    expected = (first @ first.transpose(-1, -2) + initial.double()).float()
    bound = _tail_chain._bind_isolated(args)
    bound.set_config(_config())
    torch.testing.assert_close(bound(*args), expected, atol=2e-5, rtol=2e-5)


def test_search_keeps_default_and_original_seed_pool():
    from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
    from helion.autotuner.config_fragment import EnumFragment

    key = "cute_chained_typed_operands"
    with _cpu_codegen():
        bound = _chain._bind_isolated(_args())
        spec = bound.config_spec
        field = spec._flat_fields()[key]
        assert isinstance(field, EnumFragment)
        assert field.default() is False
        assert field.search_values() == [False, True]
        assert key not in spec.default_config()
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if not seed.get("cute_chained_scalar_operand_unroll", 0)
        ]
        added = [seed for seed in seeds if seed.get(key, False)]
        assert 0 < len(added) <= 2
        base = [seed for seed in seeds if not seed.get(key, False)]
        # The sibling builder retains every original seed object and order.
        assert bound.host_function is not None
        rebuilt = CuteChainedMatmulHeuristic._with_root_typed_operand_seeds(
            bound.env, bound.host_function.device_ir, base
        )
        assert [seed for seed in rebuilt if not seed.get(key, False)] == base
        selected = spec.normalized_config(_config())
        generation = spec.create_config_generation()
        assert generation.unflatten(generation.flatten(selected)) == selected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_consumers(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a, dtype=torch.float32)
    for tile in hl.tile(a.size(0), block_size=1):
        idx = tile.id
        first = hl.dot(a[idx, :, :], b[idx, :, :])
        pure = first.to(torch.float16)
        computed = (first * 0.5).to(torch.float16)
        out[idx, :, :] = hl.dot(pure, computed)
    return out


def test_computed_consumer_keeps_fp32_and_ordinary_staging() -> None:
    with _cpu_codegen():
        args = _args()[:2]
        bound = _mixed_consumers._bind_isolated(args)
        source = bound.to_code(_config())
    assert "chain_0_c =" in source
    assert "chain_1_a = chain_typed_0" in source
    assert "chain_1_b_step" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_partial_retention_gpu() -> None:
    args = _args(device="cuda")[:2]
    a, b = args
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    generator = torch.Generator(device="cuda").manual_seed(559)
    for value in (a, b):
        value.normal_(generator=generator).mul_(0.025)
    first = (a.double() @ b.double()).float()
    expected = first.half().double() @ (first * 0.5).half().double()
    bound = _mixed_consumers._bind_isolated(args)
    bound.set_config(_config())
    torch.testing.assert_close(bound(*args).double(), expected, atol=2e-5, rtol=2e-5)
