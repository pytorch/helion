from __future__ import annotations

import ast
import importlib
import re
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_startup import cpu_codegen
from test.test_cute_chained_startup import plans
from test.test_cute_chained_vector_export import _args
from test.test_cute_chained_vector_export import _vectors

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import _with_chained_startup_seed
from helion._compiler.cute import chained_tcgen05
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_startup_transfer"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _one(a, b):
    rows, reduction = a.shape
    columns = b.size(1)
    result = torch.empty((rows, columns), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([rows, columns], block_size=[None, columns]):
        kk = hl.arange(reduction)
        weighted = (b[kk, col].float() * 0.75).to(b.dtype)
        result[row, col] = hl.dot(a[row, kk], weighted) + 0.125
    return result


def _values(dtype=torch.bfloat16, n=64, major="KK"):
    a = torch.empty((64, 128), dtype=dtype)
    b = torch.empty((n, 128), dtype=dtype).T
    if major[0] == "M":
        a = a.T.contiguous().T
    if major[1] == "M":
        b = b.contiguous()
    return a, b


def _config(n=64, *, direct=False, startup="tma", **extra):
    values = {
        "block_sizes": [64],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_direct_output": direct,
        **extra,
    }
    if startup is not None:
        values[KEY] = startup
    return helion.Config.from_dict(values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("major", ["KK", "KM", "MK", "MM"])
def test_m64_startup_source(dtype, n, major):
    with cpu_codegen():
        bound = _one._bind_isolated(_values(dtype, n, major))
        config = _config(n)
        code = bound.to_code(config)
        assert code == bound.to_code(bound._normalized_config_copy(config))
    descriptors = plans(code)
    assert len(descriptors) == 2
    assert (
        sum(p["tile"][0] * p["tile"][1] * 2 for p in descriptors)
        == (64 * 128 + n * 128) * 2
    )
    assert "tcgen05.Ld16x256bOp" in code
    assert (
        code.index("tma_bar_ptr=chain_start_bar")
        < code.index("mbarrier_wait(chain_start_bar, 0)")
        < code.index("chain_start_b_index")
        < code.index("cute.gemm(")
    )
    assert "chain_0_a_async_copy" not in code
    assert "chain_0_b_async_copy" not in code
    ast.parse(code)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("direct", [False, True])
def test_m64_scan_exports_last_read_and_disabled(dtype, direct):
    with cpu_codegen():
        bound = _vectors._bind_isolated((*_args(dtype), "normal"))
        config = _config(
            block_sizes=[64, 64],
            direct=direct,
            cute_chained_tmem_early_release=True,
            cute_chained_tmem_free="last_read",
        )
        code = bound.to_code(config)
        assert code == bound.to_code(bound._normalized_config_copy(config))
        assert bound.to_code(
            _config(direct=direct, startup=None, block_sizes=[64, 64])
        ) == bound.to_code(
            _config(direct=direct, startup="legacy", block_sizes=[64, 64])
        )
    assert (
        code.index("tma_bar_ptr=chain_start_bar")
        < code.index("chain_scan_0_pointer")
        < code.index("mbarrier_wait(chain_start_bar, 0)")
    )
    assert code.count("chain_allocator.free(chain_tptr)") == 1
    assert "chain_export_1_index" in code and "chain_export_2_index" in code
    assert "chain_origin_1 == 0 and chain_origin_2 == 0" in code


@pytest.mark.parametrize("direct", [False, True])
def test_m64_resource_charge(direct):
    original = chained_tcgen05.supported_plan
    calls = []

    def checked(plan, padding=0, *, startup=False):
        if startup:
            assert chained_tcgen05.is_m64_plan(plan)
            needed = chained_tcgen05._shared_memory_bytes(plan, padding, startup=True)
            assert needed == chained_tcgen05._shared_memory_bytes(plan, padding) + 128
            for capacity, expected in ((needed - 1, False), (needed, True)):
                with patch.object(
                    chained_tcgen05.CuteTcgen05Config,
                    "per_cta_smem_capacity_bytes",
                    return_value=capacity,
                ):
                    assert original(plan, padding, startup=True) == expected
            calls.append(needed)
        return original(plan, padding, startup=startup)

    with cpu_codegen(), patch.object(chained_tcgen05, "supported_plan", checked):
        _one._bind_isolated(_values()).to_code(_config(direct=direct))
    assert calls


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_host_selected_fastpath": True},
        {"cute_chained_initialized_accumulator": True},
        {"cute_chained_late_rhs_reuse": True},
        {"cute_chained_k_schedule": "serial64"},
        {"cute_chained_k_schedule": "overlap64"},
        {"block_sizes": [32]},
    ],
)
def test_unsupported_m64_startup_fails_closed(extra):
    with cpu_codegen(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _one._bind_isolated(_values()).to_code(_config(**extra))


def test_m64_holey_sources_reject():
    a = torch.empty((64, 256), dtype=torch.bfloat16)[:, ::2]
    b = torch.empty((128, 128), dtype=torch.bfloat16)[:, ::2]
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _one._bind_isolated((a, b)).to_code(_config())


def test_m64_seed_adds_one_direct_sibling_preserving_legacy_objects():
    parent = _config(startup=None, direct=True)
    legacy = _config(startup=None)
    old = [helion.Config(), legacy, parent, parent, helion.Config(num_warps=4)]
    actual = _with_chained_startup_seed(old, [parent], prefer_direct=True)
    assert actual[3].config == parent.config | {KEY: "tma"}
    filtered = [s for s in actual if s.config.get(KEY) != "tma"]
    assert len(filtered) == len(old)
    assert all(a is b for a, b in zip(filtered, old, strict=True))
    assert _with_chained_startup_seed(old, [], prefer_direct=True) is old
    assert _with_chained_startup_seed([legacy], [legacy], prefer_direct=True) == [
        legacy
    ]


def test_m64_actual_initial100_normalizes_and_admits():
    with cpu_codegen():
        bound = _one._bind_isolated(_values())
        generation = ConfigGeneration(bound.env.config_spec)
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = generation
        population = generation.random_population_flat(100)
        expected = search.make_unbenchmarked(
            generation.flatten(
                next(
                    seed
                    for seed in bound.config_spec.compiler_seed_configs
                    if seed.config.get(KEY) == "tma"
                )
            )
        )
        assert expected is not None
        admitted = []
        for position, flat in enumerate(population[:100]):
            member = search.make_unbenchmarked(flat)
            if member is not None and member.config == expected.config:
                source = bound.to_code(member.config)
                assert member.config.config["cute_chained_direct_output"]
                assert "tcgen05.Ld16x256bOp" in source
                assert "tma_bar_ptr=chain_start_bar" in source
                admitted.append(position)
        assert len(population) >= 100 and admitted


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped(a, b, ratio: int):
    ratio = hl.specialize(ratio)
    groups, rows, reduction = a.shape
    columns = b.size(2)
    result = torch.empty((groups, rows, columns), dtype=torch.float32, device=a.device)
    for group, row, col in hl.tile(
        [groups, rows, columns], block_size=[1, None, columns]
    ):
        kk = hl.arange(reduction)
        weighted = (b[group.begin // ratio, kk, col].float() * 0.75).to(b.dtype)
        result[group.begin, row, col] = hl.dot(a[group.begin, row, kk], weighted)
    return result


@pytest.mark.parametrize("ratio", [2, 3, 4, 8])
def test_grouped_outer_quotient_keeps_bijective_matrix(ratio):
    args = (
        torch.empty((ratio * 3, 64, 128), dtype=torch.bfloat16),
        torch.empty((3, 128, 64), dtype=torch.bfloat16),
        ratio,
    )
    with cpu_codegen():
        bound = _grouped._bind_isolated(args)
        config = _config(direct=True)
        source = bound.to_code(config)
        assert source == bound.to_code(bound._normalized_config_copy(config))
        assert bound.to_code(_config(startup=None)) == bound.to_code(
            _config(startup="legacy")
        )
    assert len(plans(source)) == 2
    assert f"// {ratio}" in source


@pytest.mark.parametrize(
    "bad", ["matrix", "computed", "negative", "nested", "variable"]
)
def test_uniform_quotient_rejects_unproved_dependencies(bad):
    import sympy

    from helion._compiler.cute.chained_matmul import _UnsupportedChain
    from helion._compiler.cute.chained_startup import _uniform_quotients

    outer, row, divisor = sympy.symbols("outer row divisor", integer=True)
    values = {
        "matrix": sympy.floor(row / 4),
        "computed": sympy.floor((outer + 1) / 4),
        "negative": sympy.floor(-outer / 4),
        "nested": sympy.floor(
            sympy.Mul(sympy.floor(outer / 4), sympy.Rational(1, 4)), evaluate=False
        ),
        "variable": sympy.floor(outer / divisor),
    }
    with pytest.raises(_UnsupportedChain, match="outer origin"):
        _uniform_quotients([values[bad]], {outer: 11})


@pytest.mark.parametrize("divisor", [2, 3, 4, 8])
def test_quotient_independent_axis_is_conservative(divisor):
    import sympy

    from helion._compiler.cute.chained_startup import _interval
    from helion._compiler.cute.chained_startup import _uniform_quotients

    outer = sympy.Symbol("outer", integer=True)
    value = sympy.Add(
        sympy.Mul(128, sympy.floor(outer / divisor)), sympy.Mul(16, outer)
    )
    mapping = _uniform_quotients([value], {outer: 31})
    quotient = next(iter(mapping.values()))
    lifted = value.xreplace(mapping)
    low, high = _interval(lifted, {outer: 31, quotient: 31 // divisor})
    for origin in range(32):
        concrete = int(cast("sympy.Integer", value.subs(outer, origin)))
        assert low <= concrete <= high and concrete % 16 == 0


@pytest.mark.parametrize("shape", [(64, 128), (32, 128), (96, 128), (256, 128)])
@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
def test_m64_actual_tma_partition_and_swizzle(shape, inner, dtype_name):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    rows, columns = shape if inner else shape[::-1]
    atom = min(128, (columns * 2) & -(columns * 2))
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            original = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    getattr(
                        tcgen05.SmemLayoutAtomKind, f"{'K' if inner else 'MN'}_SW{atom}"
                    ),
                    dtype,
                ),
                shape,
                order=(0, 1) if inner else (1, 0),
            )
            descriptor = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    getattr(tcgen05.SmemLayoutAtomKind, f"K_SW{atom}"), dtype
                ),
                (rows, columns),
                order=(0, 1),
            )
            assert str(original.inner) == str(descriptor.inner)
            offsets = set()
            for row in range(rows):
                for col in range(columns):
                    value = int(original.outer((row, col) if inner else (col, row)))
                    assert value == int(descriptor.outer((row, col)))
                    offsets.add(value)
            assert offsets == set(range(rows * columns))
            global_tensor = cute.make_tensor(
                cute.make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((rows * 2, columns), stride=(columns, 1)),
            )
            tma, tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                global_tensor,
                descriptor,
                (rows, columns),
            )
            target = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(dtype, 0, cute.AddressSpace.smem, assumed_align=128),
                    original.inner,
                    dtype=dtype,
                ),
                original.outer,
            )
            if not inner:
                target = cute.make_tensor(
                    target.iterator, cute.select(target.layout, mode=[1, 0])
                )
            source = cute.local_tile(tensor, (rows, columns), (1, 0))
            shared, global_part = cute.nvgpu.cpasync.tma_partition(
                tma,
                0,
                cute.make_layout(1),
                cute.group_modes(target, 0, 2),
                cute.group_modes(source, 0, 2),
            )
            assert (
                int(cute.size(shared)) == int(cute.size(global_part)) == rows * columns
            )
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before


def _canonical(node):
    names = {}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if re.fullmatch(r"v_\d+|chain_value_\d+", node.id):
                if node.id not in names:
                    names[node.id] = f"temporary_{len(names)}"
                node.id = names[node.id]
            return node

    return ast.dump(Rename().visit(ast.parse(ast.unparse(node))))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("major", ["KK", "KM", "MK", "MM"])
def test_prestaged_vector_guard_body_fallback_and_unroll_identity(dtype, major):
    config = _config(
        direct=True,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_pointwise_inplace_async=True,
    )
    with cpu_codegen():
        bound = _one._bind_isolated(_values(dtype, 128, major))
        candidate = bound.to_code(config)
        config.config[KEY] = "legacy"
        control = bound.to_code(config)

    def guard(source):
        return next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.If)
            and "chain_0_b_pointwise_leaf_0_pointer" in ast.unparse(node.test)
        )

    old, new = guard(control), guard(candidate)
    assert _canonical(old.test) == _canonical(new.test)
    assert _canonical(ast.Module(body=old.orelse, type_ignores=[])) == _canonical(
        ast.Module(body=new.orelse, type_ignores=[])
    )
    old_loop = next(node for node in old.body if isinstance(node, ast.For))
    new_loop = next(node for node in new.body if isinstance(node, ast.For))
    assert _canonical(old_loop) == _canonical(new_loop)
    assert "unroll=8" in ast.unparse(new_loop.iter)
    assert "chain_0_b_raw_copy" in control and "chain_0_b_raw_copy" not in candidate
    assert "chain_0_b_raw_partition" in candidate
    assert candidate.index("mbarrier_wait(chain_start_bar, 0)") < candidate.index(
        "chain_0_b_raw_partition"
    )
    assert len(plans(candidate)) == 2


def test_multiple_coordinate_leaf_is_not_handed_to_vector_producer():
    original = chained_tcgen05._vector_leaf
    duplicated = []

    def duplicate(expression, node, coordinates, indices, *args):
        result = original(expression, node, coordinates, indices, *args)
        if result is not None and not duplicated:
            expression.loaded_inputs.append(
                (node, coordinates[::-1], indices, "duplicate")
            )
            duplicated.append(True)
        return result

    with (
        cpu_codegen(),
        patch.object(chained_tcgen05, "_vector_leaf", side_effect=duplicate),
        pytest.raises(exc.BackendUnsupported, match="pre-staged leaf"),
    ):
        _one._bind_isolated(_values()).to_code(
            _config(cute_chained_pointwise_vectorize=True)
        )
    assert duplicated


@pytest.mark.parametrize("kind", ["unknown", "dtype"])
def test_prestaged_leaf_handoff_rejects_unproved_identity(kind):
    original = chained_tcgen05._vector_leaf

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None:
            import dataclasses

            if kind == "unknown":
                other = result.node.args[0]
                assert isinstance(other, torch.fx.Node)
                return dataclasses.replace(result, node=other)
            # A different source dtype cannot inhabit the narrow operand arena.
            return dataclasses.replace(result, dtype=torch.float32)
        return result

    with (
        cpu_codegen(),
        patch.object(chained_tcgen05, "_vector_leaf", side_effect=changed),
        pytest.raises(exc.BackendUnsupported, match="pre-staged leaf"),
    ):
        _one._bind_isolated(_values()).to_code(
            _config(cute_chained_pointwise_vectorize=True)
        )
