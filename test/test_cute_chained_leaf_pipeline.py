from __future__ import annotations

import ast
from collections import Counter
import importlib
import re
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_initialized_accumulator import _cpu

import helion
from helion import exc
from helion._compiler.cute import chained_leaf_pipeline as leaf
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_leaf_pipeline"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pair(a, b, raw, d, weights, dt):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        first = hl.dot(a[row, kk], b[kk, col])
        prefix = hl.cumsum(weights[kk], dim=0)
        seed = first * torch.exp(prefix[row])[:, None]
        decay = torch.exp((prefix[row][:, None] - prefix[kk][None, :]).clamp(max=0.0))
        weighted = (raw[row, kk] * decay) * dt[kk][None, :].float()
        left = torch.where(row.index[:, None] >= kk[None, :], weighted, 0.0).to(a.dtype)
        second = hl.dot(left, d[kk, col])
        out[row, col] = (seed + second + d[row, col].float()).to(a.dtype)
    return out


def args(dtype=torch.bfloat16, m=128, n=64):
    return (
        torch.empty((m, 128), dtype=dtype),
        torch.empty((128, n), dtype=dtype),
        torch.empty((m, 128), dtype=torch.float32),
        torch.empty((128, n), dtype=dtype),
        torch.empty(128),
        torch.empty(128, dtype=dtype),
    )


def config(mode="paired_tma", schedule="overlap64", **kw):
    values = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_initialized_accumulator": True,
        "cute_chained_late_rhs_reuse": True,
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_unroll": 8,
        "cute_chained_pointwise_read_cache": True,
        "cute_chained_k_schedule": schedule,
    }
    if mode is not None:
        values[KEY] = mode
    return helion.Config.from_dict(values | kw)


def source(mode="paired_tma", values=None, **kw):
    with _cpu():
        return pair._bind_isolated(args() if values is None else values).to_code(
            config(mode, **kw)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("paired_tma", "paired_tma_coeff_prefetch"))
@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
def test_source(dtype, mode, schedule):
    code = source(mode, args(dtype), schedule=schedule)
    ast.parse(code)
    assert "chained_paired_leaf_tma" in code
    assert "16384" in code
    assert "chain_leaf_retained" in code
    assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in code
    assert "_helion_cute_disable_bake_tensor_shapes = True" in code


def test_legacy_identity():
    assert source(None) == source("legacy")


def test_enabled_keeps_original_host_and_tensor_argument_order():
    original, candidate = (ast.parse(source(mode)) for mode in (None, "paired_tma"))

    def functions(tree):
        return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    old, new = functions(original), functions(candidate)
    assert ast.dump(old["pair"]) == ast.dump(new["pair"])
    assert [a.arg for a in old["_helion_pair"].args.args] == [
        a.arg for a in new["_helion_pair"].args.args[:-2]
    ]
    assert [a.arg for a in new["_helion_pair"].args.args[-2:]] == [
        "leaf_tma_atom",
        "leaf_tma_tensor",
    ]


@pytest.mark.parametrize("value", (True, False, None, 0, 1, "", "TMA", "paired", []))
def test_invalid(value):
    with _cpu():
        bound = pair._bind_isolated(args())
        c = config()
        c.config[KEY] = value
        for fix in (False, True):
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.normalize(dict(c.config), _fix_invalid=fix)


@pytest.mark.parametrize(
    "shape,strides,want",
    [
        ((128, 128), (128, 1), 16384),
        ((128, 128), (1, 128), 16384),
        ((128, 128), (136, 1), None),
        ((128, 128), (1, 1), None),
        ((128, 128), (0, 1), None),
    ],
)
def test_compact(shape, strides, want):
    assert leaf.compact_elements(shape, strides) == want


@pytest.mark.parametrize("mode", ("paired_tma", "paired_tma_coeff_prefetch"))
def test_real_raw_canonical_stops(mode):
    class Stop(BaseException):
        pass

    with _cpu():
        bound = pair._bind_isolated(args())
        raw = config(mode)
        canonical = bound._normalized_config_copy(raw)
        expected = bound.to_code(raw)
        assert bound.to_code(canonical) == expected
        observed = []

        def stop(code, **kw):
            observed.append(code)
            raise Stop

        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for request in (raw, canonical):
                with pytest.raises(Stop):
                    bound.compile_config(request, allow_print=False)
        assert observed == [expected, expected]
        assert not bound._compile_cache


def test_seed_pool_and_real_first100():
    from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
    from helion.autotuner.base_search import PopulationBasedSearch
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu():
        bound = pair._bind_isolated(args())
        spec = bound.config_spec
        assert bound.host_function is not None
        with bound.env:
            pool = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            spec.cute_chained_leaf_pipeline_search_enabled = False
            try:
                old = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            finally:
                spec.cute_chained_leaf_pipeline_search_enabled = True
            assert pool is not None and old is not None
            assert [c for c in pool if KEY not in c.config] == old
            assert len(pool) == len(old) + 2 and pool[0] == old[0]
            generation = ConfigGeneration(spec)
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast(
                        "PopulationBasedSearch", SimpleNamespace(config_gen=generation)
                    ),
                    flat,
                )
                for flat in generation.random_population_flat(100)
            ]
        for mode in ("paired_tma", "paired_tma_coeff_prefetch"):
            member = next(
                m for m in members if m is not None and m.config.config.get(KEY) == mode
            )
            assert "chained_paired_leaf_tma" in bound.to_code(member.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cute_chained_k_schedule", "full"),
        ("cute_chained_initialized_accumulator", False),
        ("cute_chained_late_rhs_reuse", False),
        ("cute_chained_pointwise_vectorize", False),
        ("cute_chained_pointwise_inplace_async", True),
        ("cute_chained_direct_output", True),
        ("cute_chained_coefficient_cache", True),
        ("num_warps", 8),
    ],
)
def test_prerequisites_never_repaired(field, value):
    with _cpu():
        bound = pair._bind_isolated(args())
        requested = config(**{field: value})
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.normalize(dict(requested.config), _fix_invalid=repair)


@pytest.mark.parametrize("kind", ("stride", "transpose", "tail", "bf16"))
def test_bad_leaf_rejects(kind):
    values = list(args())
    values[2] = {
        "stride": lambda: torch.empty((128, 256))[:, ::2],
        "transpose": lambda: torch.empty((128, 128)).T,
        "tail": lambda: torch.empty((128, 127)),
        "bf16": lambda: torch.empty((128, 128), dtype=torch.bfloat16),
    }[kind]()
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        source(values=tuple(values))


def _scalar_fallback(code):
    tree = ast.parse(code)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and ast.unparse(n.target) == "chain_1_a_step"
    )
    names = {}

    class Alpha(ast.NodeTransformer):
        def visit_Name(self, node):
            if re.fullmatch(r"(v|chain_value)_\d+", node.id):
                node.id = names.setdefault(node.id, f"tmp{len(names)}")
            return node

    return ast.dump(Alpha().visit(node))


def test_exact_scalar_fallback_and_uniform_phase_drain():
    baseline, candidate = source(None), source()
    assert _scalar_fallback(candidate) == _scalar_fallback(baseline)
    tree = ast.parse(candidate)
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "chain_leaf_source_pointer.toint()" in ast.unparse(n.test)
    )
    fallback = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
    assert (
        fallback.index("mbarrier_wait(chain_leaf_bar, 0)")
        < fallback.index("16384")
        < fallback.index("mbarrier_wait(chain_leaf_bar, 1)")
        < fallback.index("for chain_1_a_step")
    )


@pytest.mark.parametrize("mode", ("paired_tma", "paired_tma_coeff_prefetch"))
def test_phase_order_and_typed_arithmetic(mode):
    code = source(mode)
    # First A retires before descriptor issue; this issue overlaps seed work.
    assert (
        code.index("mbarrier_wait(chain_bars + 0, 0)")
        < code.index("mbarrier_arrive_and_expect_tx(chain_leaf_bar, 16384)")
        < code.index("chain_0_copy =")
    )
    assert "propagate_nan=True" in code
    assert "cutlass.Float32" in code and "cutlass.BFloat16" in code
    assert code.index(
        "cute.copy(chain_leaf_raw_copy, chain_leaf_partition, chain_leaf_raw_values)"
    ) < code.index("chain_leaf_retained_target =")
    if mode.endswith("coeff_prefetch"):
        assert code.index("chain_leaf_read_cache_0 =") < code.index(
            "mbarrier_wait(chain_leaf_bar, chain_leaf_panel)"
        )
        assert "chain_leaf_row_cache_0" in code
    else:
        assert code.index("chain_leaf_read_cache_0 =") > code.index(
            "mbarrier_wait(chain_leaf_bar, chain_leaf_panel)"
        )


def test_current_metadata_alias_and_cache_guards():
    from helion.runtime.cute import launcher
    from helion.runtime.cute.chained_leaf_pipeline import validate_plan

    plan: dict[str, object] = {
        "kind": leaf.KIND,
        "lhs_idx": 0,
        "out_idx": 2,
        "shape": (128, 128),
        "strides": (128, 1),
    }
    raw = torch.empty(128 * 128 + 4)
    source0, source4 = raw[: 128 * 128].view(128, 128), raw[4:].view(128, 128)
    coefficient, output = torch.empty(128), torch.empty((128, 64), dtype=torch.bfloat16)
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    with (
        patch.object(
            torch.Tensor, "device", property(lambda self: torch.device("cuda", 0))
        ),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
    ):
        for current in (source0, source4):
            validate_plan(plan, (current, coefficient, output))
        for current in (
            source0.T,
            raw[1 : 1 + 128 * 128].view(128, 128),
            source0.to(torch.bfloat16),
        ):
            with pytest.raises(exc.BackendUnsupported):
                validate_plan(plan, (current, coefficient, output))
        with pytest.raises(exc.BackendUnsupported, match="aliases"):
            validate_plan(plan, (source0, coefficient, source0.view(torch.bfloat16)))
        # Neither a warm prepared cache nor default fast-relaunch can bypass
        # current alignment/shape validation.
        invalid = (source0.T, coefficient, output)
        with (
            patch.object(
                launcher,
                "_cute_dynamic_tensormap_contexts",
                side_effect=AssertionError("late cache reached"),
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            launcher._build_cached_cute_schema_and_args(kernel, invalid, (1, 1, 1))
        with (
            patch.object(
                launcher,
                "_cute_last_launch_cache_entry",
                side_effect=AssertionError("late cache reached"),
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            launcher.default_cute_launcher(kernel, (1,), *invalid, block=(128, 1, 1))
        output.untyped_storage().resize_(1)
        with pytest.raises(exc.BackendUnsupported, match="exceeds current storage"):
            validate_plan(plan, (source0, coefficient, output))


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
def test_actual_static_physical_panels(dtype_name):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            raw_layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float32
                ),
                (128, 32),
                order=(0, 1),
            )
            a_layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128, dtype
                ),
                (128, 128),
                order=(0, 1),
            )
            assert str(raw_layout.inner) == "S<3,4,3>" == str(a_layout.inner)
            assert cute.cosize(raw_layout.outer) * 4 == 16384
            assert cute.cosize(a_layout.outer) * 2 == 32768
            copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
                ),
                cute.make_layout((32, 4), stride=(4, 1)),
                cute.make_layout((1, 8)),
            )
            raw_sets, weighted_sets, cells = [set(), set()], {}, []
            for half in range(2):
                for panel in range(2):
                    identity = cute.local_tile(
                        cute.make_identity_tensor((128, 128)),
                        (128, 32),
                        (0, half * 2 + panel),
                    )
                    physical = set()
                    for thread in range(128):
                        coords = copy.get_slice(thread).partition_S(identity)
                        for step in range(4):
                            for element in range(8):
                                row, col = map(int, coords[element, step, 0])
                                assert (row, col) == (
                                    thread // 4 + step * 32,
                                    half * 64 + panel * 32 + thread % 4 * 8 + element,
                                )
                                cells.append((row, col))
                                raw = panel * 16384 + 4 * int(
                                    raw_layout.outer((row, col % 32))
                                )
                                raw ^= (raw >> 3) & 112
                                raw_sets[panel].update(range(raw, raw + 4))
                                weighted = 2 * int(a_layout.outer((row, col)))
                                weighted ^= (weighted >> 3) & 112
                                physical.update(range(weighted, weighted + 2))
                    weighted_sets[half, panel] = physical
            assert Counter(cells) == Counter(
                (m, k) for m in range(128) for k in range(128)
            )
            assert raw_sets == [set(range(16384)), set(range(16384, 32768))]
            assert set.union(*weighted_sets.values()) == set(range(32768))
            assert weighted_sets[1, 0] & raw_sets[1]  # Actual overwrite hazard.
    assert torch.cuda.is_initialized() == before


def test_real_wrapper_static_descriptor_and_partitions():
    import cutlass
    import cutlass.cute as cute

    from helion.runtime.cute.launcher import _append_cute_wrapper_plan

    ir = importlib.import_module("cutlass._mlir.ir")
    plan: dict[str, object] = {
        "kind": leaf.KIND,
        "lhs_idx": 0,
        "rows": 256,
        "columns": 128,
        "kernel_args": ["leaf_atom", "leaf_tensor"],
    }
    body, call_args = [], []
    _append_cute_wrapper_plan(body, call_args, plan)
    assert call_args == ["leaf_atom", "leaf_tensor"]
    assert "arg0.iterator.align(16)" in body[0]
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            arg = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((256, 128), stride=(128, 1)),
            )
            scope: dict[str, Any] = {"cutlass": cutlass, "cute": cute, "arg0": arg}
            exec("\n".join(line.removeprefix("    ") for line in body), scope)
            layout = scope["leaf_atom_layout"]
            target = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.Float32, 0, cute.AddressSpace.smem, assumed_align=128
                    ),
                    layout.inner,
                    dtype=cutlass.Float32,
                ),
                layout.outer,
            )
            for half in range(2):
                for panel in range(2):
                    source = cute.local_tile(
                        scope["leaf_tensor"], (128, 32), (1, half * 2 + panel)
                    )
                    shared, global_ = cute.nvgpu.cpasync.tma_partition(
                        scope["leaf_atom"],
                        0,
                        cute.make_layout(1),
                        cute.group_modes(target, 0, 2),
                        cute.group_modes(source, 0, 2),
                    )
                    assert int(cute.size(shared)) == int(cute.size(global_)) == 4096
        assert module.operation.verify()


@pytest.mark.parametrize("capacity", (32768, 49152, 50000))
def test_barrier_capacity_rejects(capacity):
    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    with (
        _cpu(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        pair._bind_isolated(args()).to_code(config())


@pytest.mark.parametrize("mode", ("paired_tma", "paired_tma_coeff_prefetch"))
@pytest.mark.parametrize("cache", (False, True))
@pytest.mark.parametrize("unroll", (1, 2, 4, 8))
def test_cache_unroll_interactions(mode, cache, unroll):
    code = source(
        mode,
        cute_chained_pointwise_read_cache=cache,
        cute_chained_pointwise_unroll=unroll,
    )
    assert f"cutlass.range(4, unroll={min(4, unroll)})" in code
    assert ("chain_leaf_read_cache" in code) == (
        cache or mode.endswith("coeff_prefetch")
    )


def test_affine_domain_floor_and_unknown_rejection():
    import sympy

    axis = sympy.Symbol("axis", integer=True)
    assert leaf.interval(
        sympy.floor(sympy.Mul(axis, sympy.Rational(1, 64))), {axis: 63}
    ) == (0, 0)
    for expression in (
        sympy.floor(sympy.Mul(axis, sympy.Rational(1, 32))),
        sympy.Pow(axis, 2),
        sympy.Symbol("unknown"),
    ):
        with pytest.raises(leaf.chain._UnsupportedChain, match="paired leaf"):
            leaf.interval(expression, {axis: 63})


def test_raw_snapshot_bits_and_half_retirement_model():
    import struct

    # Arbitrary raw FP32 payloads include signed zeros, infinities and NaNs.
    words = (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC54321, 0x3F800000)
    raw = {
        (r, c): words[(r * 128 + c) % len(words)] ^ ((r * 128 + c) & 31)
        for r in range(128)
        for c in range(128)
    }

    def address(row, col, size):
        atom = 128 // size
        byte = row * 128 + (col % atom) * size + (col // atom) * 128 * 128
        return byte ^ ((byte >> 3) & 112)

    arena = bytearray(32768)
    retired = True
    observed = {}
    for half in range(2):
        assert retired
        snapshots = []
        for panel in range(2):
            for row in range(128):
                for col in range(32):
                    struct.pack_into(
                        "<I",
                        arena,
                        panel * 16384 + address(row, col, 4),
                        raw[row, half * 64 + panel * 32 + col],
                    )
            snapshots.append(
                {
                    (row, col): struct.unpack_from(
                        "<I", arena, panel * 16384 + address(row, col, 4)
                    )[0]
                    for row in range(128)
                    for col in range(32)
                }
            )
        # Only after both panels were snapshotted may either be narrowed/stored.
        for panel in range(2):
            for row in range(128):
                for col in range(32):
                    original = raw[row, half * 64 + panel * 32 + col]
                    assert snapshots[panel][row, col] == original
                    struct.pack_into(
                        "<H",
                        arena,
                        address(row, half * 64 + panel * 32 + col, 2),
                        original >> 16,
                    )
        retired = False
        for row in range(128):
            for col in range(64):
                observed[row, half * 64 + col] = struct.unpack_from(
                    "<H", arena, address(row, half * 64 + col, 2)
                )[0]
        retired = True
    assert observed == {coordinate: bits >> 16 for coordinate, bits in raw.items()}
