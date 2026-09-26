from __future__ import annotations

import ast
from collections import Counter
import contextlib
import dataclasses
import importlib
import inspect
from itertools import product
import re
import types
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_accumulator import _cpu
from test.test_cute_chained_accumulator import _late_rhs_args
from test.test_cute_chained_accumulator import _late_rhs_code
from test.test_cute_chained_accumulator import _late_rhs_config
from test.test_cute_chained_accumulator import _late_rhs_pair
from test.test_cute_chained_scan_export import _scan_export_config
from test.test_cute_chained_scan_export import _scan_export_cpu_codegen
from test.test_cute_chained_scan_export import _vector_export_args
from test.test_cute_chained_scan_export import _vectors
from test.test_cute_chained_scan_input_export import _export
from test.test_cute_chained_tcgen05 import _tcgen_chain
from test.test_cute_chained_tcgen05 import _tcgen_inputs
from test.test_cute_chained_tcgen05 import _tcgen_single

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import _with_chained_startup_seed
from helion._compiler.cute import chained_leaf_pipeline as leaf
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


# Startup.


@contextlib.contextmanager
def cpu_codegen():
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def startup_config(mode: str | None = "tma", n: int = 64) -> helion.Config:
    values: dict[str, Any] = {
        "block_sizes": [128, n],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if mode is not None:
        values["cute_chained_startup_transfer"] = mode
    return helion.Config(**values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_direct_start_before_scan(dtype: torch.dtype, n: int) -> None:
    with cpu_codegen():
        bound = _tcgen_chain._bind_isolated(
            (*_tcgen_inputs("cpu", dtype, n=n), "decay")
        )
        source = bound.to_code(startup_config(n=n))
    assert "kind': 'chained_startup_tma'" in source
    assert source.index("tma_bar_ptr=chain_start_bar") < source.index(
        "chain_scan_0_pointer"
    )
    assert source.index("mbarrier_wait(chain_start_bar, 0)") < source.index(
        "cute.gemm("
    )
    assert "chain_0_a_async_copy" not in source
    assert "chain_0_b_async_copy" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_raw_start_before_scan(dtype: torch.dtype) -> None:
    values = tuple(
        torch.randn(s, dtype=dtype) for s in ((2, 128, 128), (2, 128, 64), (2, 128))
    )
    with cpu_codegen():
        source = _tcgen_single._bind_isolated((*values, True, True)).to_code(
            startup_config()
        )
    assert source.index("tma_bar_ptr=chain_start_bar") < source.index(
        "chain_scan_0_pointer"
    )
    assert "chain_start_b_index" in source
    assert source.index("mbarrier_wait(chain_start_bar, 0)") < source.index(
        "chain_start_b_index"
    )


def test_startup_legacy_identity() -> None:
    with cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu"), "decay"))
        assert bound.to_code(startup_config(None)) == bound.to_code(
            startup_config("legacy")
        )


@pytest.mark.parametrize("mode", [False, True, 1, None, "invalid"])
def test_invalid_enum(mode: Any) -> None:
    cfg = startup_config()
    cfg.config["cute_chained_startup_transfer"] = mode
    with cpu_codegen(), pytest.raises(exc.InvalidConfig):
        _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu"), "decay")).to_code(cfg)


def plans(source: str) -> list[dict[str, Any]]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_helion_cute_wrapper_plans"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("missing wrapper plans")


def test_raw_normalized_same() -> None:
    with cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu"), "decay"))
        raw = startup_config()
        normalized = bound.env.config_spec.normalized_config(raw)
        assert bound.to_code(raw) == bound.to_code(normalized)


def test_no_eligible_layout_fails_closed() -> None:
    args = list(_tcgen_inputs("cpu"))
    args[0] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    args[1] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _tcgen_chain._bind_isolated((*args, "decay")).to_code(startup_config())


def test_fp32_raw_leaf_does_not_borrow_a_narrow_arena() -> None:
    a = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    b = torch.empty((2, 128, 64), dtype=torch.float32)
    coeff = torch.empty((2, 128), dtype=torch.float32)
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _tcgen_single._bind_isolated((a, b, coeff, True, True)).to_code(
            startup_config()
        )


def test_holey_read_sources_do_not_get_a_startup_seed() -> None:
    args = list(_tcgen_inputs("cpu"))
    args[0] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    args[1] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    with cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*args, "decay"))
        assert not any(
            seed.config.get("cute_chained_startup_transfer") == "tma"
            for seed in bound.env.config_spec.compiler_seed_configs
        )


def test_nonaffine_index_proof_rejects() -> None:
    import sympy

    from helion._compiler.cute.chained_matmul import _UnsupportedChain
    from helion._compiler.cute.chained_startup import _interval

    index = sympy.Symbol("index", integer=True)
    assert _interval(sympy.Add(3, sympy.Mul(2, index)), {index: 127}) == (3, 257)
    with pytest.raises(_UnsupportedChain, match="affine"):
        _interval(sympy.Pow(index, 2), {index: 127})


@pytest.mark.parametrize("mode", ["coalesced", "cp_async"])
def test_other_schedule_rejects(mode: str) -> None:
    cfg = startup_config()
    cfg.config["cute_chained_mma_schedule"] = mode
    with cpu_codegen(), pytest.raises(exc.InvalidConfig, match="TCgen05"):
        _tcgen_chain._bind_isolated((*_tcgen_inputs("cpu"), "decay")).to_code(cfg)


def test_seed_preserves_objects_order_multiplicity() -> None:
    from helion._compiler.autotuner_heuristics.cute import _with_chained_startup_seed

    parent = startup_config(None)
    old = [helion.Config(), parent, parent, helion.Config(block_sizes=[16, 16])]
    result = _with_chained_startup_seed(old, [parent])
    filtered = [
        s for s in result if s.config.get("cute_chained_startup_transfer") != "tma"
    ]
    assert len(filtered) == len(old)
    assert all(a is b for a, b in zip(filtered, old, strict=True))
    assert result[2].config == {**parent.config, "cute_chained_startup_transfer": "tma"}


def guard_plan() -> dict[str, Any]:
    return {
        "kind": "chained_startup_tma",
        "lhs_idx": 0,
        "out_idx": 1,
        "dtype": "bfloat16",
        "shape": (128, 64),
        "strides": (64, 1),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "fresh",
        "aligned_offset",
        "unaligned",
        "stride",
        "dtype",
        "alias",
        "cross_dtype_alias",
    ],
)
def test_current_arguments(kind: str) -> None:
    from helion.runtime.cute.chained_startup import validate_arguments

    source = torch.empty((128, 64), dtype=torch.bfloat16)
    output = torch.empty((128, 64), dtype=torch.float32)
    if kind in ("aligned_offset", "unaligned"):
        offset = 8 if kind == "aligned_offset" else 1
        source = torch.empty(128 * 64 + offset, dtype=torch.bfloat16)[offset:].view(
            128, 64
        )
    if kind == "stride":
        source = torch.empty((128, 128), dtype=torch.bfloat16)[:, ::2]
    if kind == "dtype":
        source = source.float()
    if kind == "alias":
        output = source
    if kind == "cross_dtype_alias":
        output = source.view(torch.float32)
    kernel = types.SimpleNamespace(_helion_cute_wrapper_plans=[guard_plan()])
    with (
        patch.object(
            torch.Tensor,
            "device",
            new_callable=PropertyMock,
            return_value=torch.device("cuda:0"),
        ),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        if kind in ("fresh", "aligned_offset"):
            validate_arguments(kernel, (source, output))
            validate_arguments(kernel, (source.clone(), output.clone()))
        else:
            with pytest.raises(exc.BackendUnsupported):
                validate_arguments(kernel, (source, output))


def test_guard_precedes_builder_and_cache() -> None:
    from helion.runtime.cute import launcher

    kernel = types.SimpleNamespace(_helion_cute_wrapper_plans=[guard_plan()])
    args = (torch.empty((128, 64), dtype=torch.bfloat16), torch.empty((128, 64)))
    with (
        patch.object(
            launcher,
            "_get_cute_launcher_imports",
            side_effect=AssertionError("builder reached"),
        ),
        pytest.raises(exc.BackendUnsupported, match="current layout"),
    ):
        launcher._build_cute_schema_and_args(kernel, args, (1, 1, 1))
    source = inspect.getsource(launcher.default_cute_launcher)
    assert source.index("validate_arguments(cute_kernel, args_tuple)") < source.index(
        "_cute_last_launch_cache_entry("
    )


@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize("width", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
def test_actual_static_layout(inner: int, width: int, dtype_name: str) -> None:
    import importlib

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    shape = (128, width) if inner else (width, 128)
    atom = min(128, (2 * width) & -(2 * width))
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
                (128, width),
                order=(0, 1),
            )
            assert str(original.inner) == str(descriptor.inner)
            offsets = set()
            for row in range(128):
                for col in range(width):
                    coords = (row, col) if inner else (col, row)
                    actual = int(original.outer(coords))
                    assert actual == int(descriptor.outer((row, col)))
                    offsets.add(actual)
            assert offsets == set(range(128 * width))
            global_tensor = cute.make_tensor(
                cute.make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((256, width), stride=(width, 1)),
            )
            tma, tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                global_tensor,
                descriptor,
                (128, width),
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
            source = cute.local_tile(tensor, (128, width), (0, 0))
            shared, source_partition = cute.nvgpu.cpasync.tma_partition(
                tma,
                0,
                cute.make_layout(1),
                cute.group_modes(target, 0, 2),
                cute.group_modes(source, 0, 2),
            )
            assert (
                int(cute.size(shared))
                == int(cute.size(source_partition))
                == 128 * width
            )
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before


# K schedule.

K_SCHEDULE_KEY = "cute_chained_k_schedule"

FLAGS = {
    "cute_chained_pointwise_vectorize": True,
    "cute_chained_pointwise_unroll": 8,
    "cute_chained_pointwise_read_cache": True,
}


def k_schedule_source(mode="serial64", args=None, **extra):
    return _late_rhs_code(
        _late_rhs_args() if args is None else args,
        **(FLAGS | {K_SCHEDULE_KEY: mode} | extra),
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _earlier_raw(a, b, c, d, scale):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        first_left = (a[row, kk].float() * scale[kk][None, :]).to(a.dtype)
        first = hl.dot(first_left, b[kk, col])
        seed = first * torch.exp(scale[row])[:, None]
        final_left = (c[row, kk].float() * scale[kk][None, :]).to(a.dtype)
        second = hl.dot(final_left, d[kk, col])
        out[row, col] = (seed + second + d[row, col].float()).to(a.dtype)
    return out


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_raw_on_earlier_operand_only_is_preserved(dtype, mode):
    # FP32 final leaf cannot be raw-preloaded into a BF16/FP16 arena; the
    # already-supported same-dtype first leaf still activates the raw option.
    args = (
        torch.empty((128, 128), dtype=dtype),
        torch.empty((128, 64), dtype=dtype),
        torch.empty((128, 128)),
        torch.empty((128, 64), dtype=dtype),
        torch.empty(128),
    )
    with _cpu():
        bound = _earlier_raw._bind_isolated(args)
        config = _late_rhs_config(
            **(
                FLAGS
                | {
                    K_SCHEDULE_KEY: mode,
                    "cute_chained_pointwise_inplace_async": True,
                    "cute_chained_pointwise_read_cache": False,
                }
            )
        )
        result = bound.to_code(config)
    assert "chain_0_a_raw_copy" in result
    assert "chain_1_a_raw_copy" not in result
    assert "chain_k_half" in result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "first_a,first_b,final_b", tuple(product(("K", "MN"), repeat=3))
)
def test_independent_major_combinations(dtype, first_a, first_b, final_b):
    args = list(_late_rhs_args(dtype))
    if first_a == "MN":
        args[0] = torch.empty((128, 128), dtype=dtype).T
    if first_b == "K":
        args[1] = torch.empty((64, 128), dtype=dtype).T
    if final_b == "K":
        args[3] = torch.empty((64, 128), dtype=dtype).T
    result = k_schedule_source(args=tuple(args))
    calls = [
        n
        for n in ast.walk(ast.parse(result))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "chain_sm100.make_trivial_tiled_mma"
    ]
    assert [
        [ast.unparse(n.args[i]).rsplit(".", 1)[-1] for i in (2, 3)] for n in calls
    ] == [[first_a, first_b], ["K", final_b]]


def test_unsupported_reduction_has_no_k_seed_family():
    args = list(_late_rhs_args())
    args[2] = torch.empty((128, 64), dtype=torch.bfloat16)
    args[5] = torch.empty(64)
    with _cpu():
        bound = _late_rhs_pair._bind_isolated(tuple(args))
        assert not bound.config_spec.cute_chained_k_schedule_search_enabled
        assert not any(
            K_SCHEDULE_KEY in c.config for c in bound.config_spec.compiler_seed_configs
        )


def test_new_siblings_preserve_duplicate_object_order():
    from helion._compiler.autotuner_heuristics.cute import _with_k_schedule_seeds

    anchor = _late_rhs_config(cute_chained_pointwise_vectorize=False)
    parent = _late_rhs_config(**FLAGS)
    other = _late_rhs_config(**(FLAGS | {"cute_chained_auxiliary_cache": True}))
    seeds = [anchor, other, parent, other, parent]
    result = _with_k_schedule_seeds(seeds, enabled=True)
    assert [id(c) for c in result if K_SCHEDULE_KEY not in c.config] == [
        id(c) for c in seeds
    ]
    assert result[0] is seeds[0]
    assert [c.config[K_SCHEDULE_KEY] for c in result[1:3]] == ["serial64", "overlap64"]
    for child in result[1:3]:
        assert {
            k: v for k, v in child.config.items() if k != K_SCHEDULE_KEY
        } == other.config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", range(32, 257, 32))
@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_source_halves_seed_and_phases(dtype, n, mode):
    args = _late_rhs_args(dtype, n)
    control = k_schedule_source("full", args)
    new = k_schedule_source(mode, args)
    assert control == _late_rhs_code(args, **FLAGS)
    assert "cute.local_tile(chain_1_a, (128, 64), (0, chain_k_half))" in new
    assert "chain_k_half * 64 + chain_thread % 8 * 8" in new
    assert "chain_1_local_kk in cutlass.range_constexpr(4)" in new
    assert "chain_k_half * 4 + chain_1_local_kk" in new
    assert "cutlass.range(8, unroll=8)" in new
    assert "cutlass.range(64, unroll=1)" in new
    assert new.count("cute.gemm(") == control.count("cute.gemm(") == 2
    assert new.count("chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)") == 2
    assert "chain_0_c =" not in new
    start = "    for chain_k_half in cutlass.range_constexpr(2):"
    end = "    chain_1_copy ="
    body = new[new.index(start) : new.index(end)]
    assert body.count("cute.arch.cp_async_commit_group()") == 1
    assert body.count("cute.arch.cp_async_wait_group(0)") == 1
    assert body.count("cute.arch.fence_view_async_shared()") == 1
    assert body.count("tcgen05.commit(chain_bars + 1)") == 1
    if mode == "serial64":
        assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in body
        assert body.count("cute.arch.sync_threads()") == 2
    else:
        assert "mbarrier_wait(chain_bars + 1, 0)" in body
        assert "    if chain_warp == 0:" in body
        assert body.count("cute.arch.sync_threads()") == 1
    # Full allocations, initial dot/FP32 seed, final result/epilogue are exact.
    assert (
        new[: new.index("    chain_1_mma =")]
        == control[: control.index("    chain_1_mma =")]
    )
    assert new[new.index(end) :] == control[control.index(end) :]

    def allocations(text):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.alloc_smem"
        ]

    assert allocations(new) == allocations(control)


@pytest.mark.parametrize("value", (None, False, True, 0, 1, "", "serial", "FULL"))
@pytest.mark.parametrize("repair", (False, True))
def test_strict_enum_even_repair(value, repair):
    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args())
        config = _late_rhs_config(**(FLAGS | {K_SCHEDULE_KEY: value}))
        with pytest.raises(exc.InvalidConfig, match="must be full"):
            bound.config_spec.normalize(config, _fix_invalid=repair)
        assert config.config[K_SCHEDULE_KEY] == value


@pytest.mark.parametrize(
    "override",
    (
        {"num_warps": 8},
        {"cute_chained_initialized_accumulator": False},
        {"cute_chained_late_rhs_reuse": False},
        {"cute_chained_pointwise_vectorize": False},
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_direct_output": True},
    ),
)
@pytest.mark.parametrize("repair", (False, True))
def test_prerequisites_are_not_repaired(override, repair):
    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args())
        config = _late_rhs_config(**(FLAGS | {K_SCHEDULE_KEY: "overlap64"} | override))
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalize(config, _fix_invalid=repair)
        assert config.config[K_SCHEDULE_KEY] == "overlap64"


@pytest.mark.parametrize("mode", ("serial64", "overlap64"))
def test_selected_final_raw_preload_rejects(mode):
    with pytest.raises(exc.BackendUnsupported, match="raw preload"):
        k_schedule_source(mode, cute_chained_pointwise_inplace_async=True)


@pytest.mark.parametrize("mode", ("computed_rhs", "extra_first", "fp32out", "a_reader"))
def test_unsupported_pair_rejects(mode):
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        k_schedule_source(args=_late_rhs_args(mode=mode))


def test_exact_shared_accounting():
    plans = []
    original = chained_tcgen05.supported_plan

    def capture(plan):
        assert chained_tcgen05._shared_memory_bytes(
            plan
        ) == chained_tcgen05._shared_memory_bytes(
            dataclasses.replace(plan, k_schedule=None)
        )
        plans.append(plan)
        return original(plan)

    with patch.object(chained_tcgen05, "supported_plan", capture):
        k_schedule_source()
    plan = plans[-1]
    assert plan.k_schedule.byte_spans == ((0, 16384), (16384, 32768))


@pytest.mark.parametrize("kind", ("offset", "stride", "tail", "major"))
def test_full_leaf_guard_or_conservative_rejection(kind):
    args = list(_late_rhs_args())
    operand = cast("torch.Tensor", args[2])
    if kind == "offset":
        args[2] = torch.empty(128 * 128 + 1, dtype=operand.dtype)[1:].view(128, 128)
    elif kind == "stride":
        args[2] = torch.empty((128, 256), dtype=operand.dtype)[:, ::2]
    elif kind == "tail":
        args[2] = torch.empty((128, 127), dtype=operand.dtype)
        args[5] = torch.empty(127)
    else:
        args[2] = operand.T
    if kind != "offset":
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            k_schedule_source(args=tuple(args))
    else:
        result = k_schedule_source(args=tuple(args))
        assert "toint() % 16 == 0" in result
        assert (
            "chain_1_a_load = (chain_thread + chain_1_a_step * 128) // 64 * 128"
            in result
        )


@pytest.mark.parametrize("k", (64, 256))
def test_final_reduction_extent_rejects(k):
    args = list(_late_rhs_args())
    args[2] = torch.empty((128, k), dtype=torch.bfloat16)
    args[3] = torch.empty((max(128, k), 64), dtype=torch.bfloat16)
    args[5] = torch.empty(k)
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        k_schedule_source(args=tuple(args))


@pytest.mark.parametrize("mode", ("full", "serial64", "overlap64"))
def test_real_raw_canonical_preload_stops(mode):
    class StopBeforeLoad(BaseException):
        pass

    with _cpu():
        bound = _late_rhs_pair._bind_isolated(_late_rhs_args())
        raw = _late_rhs_config(**(FLAGS | {K_SCHEDULE_KEY: mode}))
        canonical = bound._normalized_config_copy(raw)
        expected = bound.to_code(raw)
        assert expected == bound.to_code(canonical)
        observed = []

        def stop(code, **kwargs):
            observed.append(code)
            raise StopBeforeLoad

        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for request in (raw, canonical):
                with pytest.raises(StopBeforeLoad):
                    bound.compile_config(request, allow_print=False)
        assert observed == [expected, expected] and not bound._compile_cache


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cache", (False, True))
@pytest.mark.parametrize("unroll", (1, 2, 4, 8))
def test_half_cache_refresh_and_other_knobs(dtype, cache, unroll):
    new = k_schedule_source(
        args=_late_rhs_args(dtype),
        cute_chained_pointwise_read_cache=cache,
        cute_chained_pointwise_unroll=unroll,
        cute_chained_auxiliary_cache=True,
        cute_chained_tmem_early_release=True,
    )
    if cache:
        half = new.index("    for chain_k_half")
        refresh = new.index(
            "chain_k_half * 64 + chain_thread % 8 * 8 + chain_1_a_pointwise_cache_element"
        )
        consume = new.index(
            f"for chain_1_a_pointwise_step in cutlass.range(8, unroll={unroll})"
        )
        assert half < refresh < consume
    assert "chain_allocator.relinquish_alloc_permit()" in new


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cpu_full_vs_half_typed_expression_and_poison(dtype):
    # Evaluate identical elementwise RHS on identical scalar CPU operands;
    # this is not a claim about GPU exp rounding or reduction association.
    raw = torch.linspace(-3, 3, 16384).view(128, 128).to(dtype)
    coefficient = torch.linspace(-8, 8, 128)
    coefficient[0], coefficient[63], coefficient[64], coefficient[-1] = (
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.0,
    )
    expected = (raw.float() * torch.exp(coefficient)[None, :]).to(dtype)
    actual = torch.full_like(expected, float("nan"))
    old = torch.full((8,), 12345.0)
    for half in range(2):
        seen = set()
        for thread in range(128):
            columns = half * 64 + 8 * (thread % 8) + torch.arange(8)
            cache = torch.exp(coefficient[columns])
            assert not torch.equal(cache, old)
            for step in range(8):
                row = thread // 8 + 16 * step
                actual[row, columns] = (raw[row, columns].float() * cache).to(dtype)
                seen.update((row, int(col)) for col in columns)
        assert len(seen) == 8192
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


# Leaf pipeline.

LEAF_KEY = "cute_chained_leaf_pipeline"


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


def leaf_config(mode="paired_tma", schedule="overlap64", **kw):
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
        values[LEAF_KEY] = mode
    return helion.Config.from_dict(values | kw)


def leaf_source(mode="paired_tma", values=None, **kw):
    with _cpu():
        return pair._bind_isolated(args() if values is None else values).to_code(
            leaf_config(mode, **kw)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
def test_source(dtype, schedule):
    code = leaf_source(values=args(dtype), schedule=schedule)
    ast.parse(code)
    assert "chained_paired_leaf_tma" in code
    assert "16384" in code
    assert "chain_leaf_retained" in code
    assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in code
    assert "_helion_cute_disable_bake_tensor_shapes = True" in code


def test_leaf_legacy_identity():
    assert leaf_source(None) == leaf_source("legacy")


def test_enabled_keeps_original_host_and_tensor_argument_order():
    original, candidate = (
        ast.parse(leaf_source(mode)) for mode in (None, "paired_tma")
    )

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
        c = leaf_config()
        c.config[LEAF_KEY] = value
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


def test_real_raw_canonical_stops():
    class Stop(BaseException):
        pass

    with _cpu():
        bound = pair._bind_isolated(args())
        raw = leaf_config()
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("cute_chained_k_schedule", "full"),
        ("cute_chained_initialized_accumulator", False),
        ("cute_chained_late_rhs_reuse", False),
        ("cute_chained_pointwise_vectorize", False),
        ("cute_chained_pointwise_inplace_async", True),
        ("cute_chained_direct_output", True),
        ("num_warps", 8),
    ],
)
def test_prerequisites_never_repaired(field, value):
    with _cpu():
        bound = pair._bind_isolated(args())
        requested = leaf_config(**{field: value})
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
        leaf_source(values=tuple(values))


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
    baseline, candidate = leaf_source(None), leaf_source()
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


def test_phase_order_and_typed_arithmetic():
    code = leaf_source()
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
        pair._bind_isolated(args()).to_code(leaf_config())


@pytest.mark.parametrize("cache", (False, True))
@pytest.mark.parametrize("unroll", (1, 2, 4, 8))
def test_cache_unroll_interactions(cache, unroll):
    code = leaf_source(
        cute_chained_pointwise_read_cache=cache,
        cute_chained_pointwise_unroll=unroll,
    )
    assert f"cutlass.range(4, unroll={min(4, unroll)})" in code
    assert ("chain_leaf_read_cache" in code) == cache


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


# Startup m64.

STARTUP_M64_KEY = "cute_chained_startup_transfer"


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


def _startup_m64_values(dtype=torch.bfloat16, n=64, major="KK"):
    a = torch.empty((64, 128), dtype=dtype)
    b = torch.empty((n, 128), dtype=dtype).T
    if major[0] == "M":
        a = a.T.contiguous().T
    if major[1] == "M":
        b = b.contiguous()
    return a, b


def _startup_m64_config(n=64, *, direct=False, startup="tma", **extra):
    values = {
        "block_sizes": [64],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_direct_output": direct,
        **extra,
    }
    if startup is not None:
        values[STARTUP_M64_KEY] = startup
    return helion.Config.from_dict(values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("major", ["KK", "KM", "MK", "MM"])
def test_m64_startup_source(dtype, n, major):
    with cpu_codegen():
        bound = _one._bind_isolated(_startup_m64_values(dtype, n, major))
        config = _startup_m64_config(n)
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
        bound = _vectors._bind_isolated((*_vector_export_args(dtype), "normal"))
        config = _startup_m64_config(
            block_sizes=[64, 64],
            direct=direct,
            cute_chained_tmem_early_release=True,
            cute_chained_tmem_free="last_read",
        )
        code = bound.to_code(config)
        assert code == bound.to_code(bound._normalized_config_copy(config))
        assert bound.to_code(
            _startup_m64_config(direct=direct, startup=None, block_sizes=[64, 64])
        ) == bound.to_code(
            _startup_m64_config(direct=direct, startup="legacy", block_sizes=[64, 64])
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

    def checked(plan, *, startup=False):
        if startup:
            assert chained_tcgen05.is_m64_plan(plan)
            needed = chained_tcgen05._shared_memory_bytes(plan, startup=True)
            assert needed == chained_tcgen05._shared_memory_bytes(plan) + 128
            for capacity, expected in ((needed - 1, False), (needed, True)):
                with patch.object(
                    chained_tcgen05.CuteTcgen05Config,
                    "per_cta_smem_capacity_bytes",
                    return_value=capacity,
                ):
                    assert original(plan, startup=True) == expected
            calls.append(needed)
        return original(plan, startup=startup)

    with cpu_codegen(), patch.object(chained_tcgen05, "supported_plan", checked):
        _one._bind_isolated(_startup_m64_values()).to_code(
            _startup_m64_config(direct=direct)
        )
    assert calls


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_chained_initialized_accumulator": True},
        {"cute_chained_late_rhs_reuse": True},
        {"cute_chained_k_schedule": "serial64"},
        {"cute_chained_k_schedule": "overlap64"},
        {"block_sizes": [32]},
    ],
)
def test_unsupported_m64_startup_fails_closed(extra):
    with cpu_codegen(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _one._bind_isolated(_startup_m64_values()).to_code(_startup_m64_config(**extra))


def test_m64_holey_sources_reject():
    a = torch.empty((64, 256), dtype=torch.bfloat16)[:, ::2]
    b = torch.empty((128, 128), dtype=torch.bfloat16)[:, ::2]
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _one._bind_isolated((a, b)).to_code(_startup_m64_config())


def test_m64_seed_adds_one_direct_sibling_preserving_legacy_objects():
    parent = _startup_m64_config(startup=None, direct=True)
    legacy = _startup_m64_config(startup=None)
    old = [helion.Config(), legacy, parent, parent, helion.Config(num_warps=4)]
    actual = _with_chained_startup_seed(old, [parent], prefer_direct=True)
    assert actual[3].config == parent.config | {STARTUP_M64_KEY: "tma"}
    filtered = [s for s in actual if s.config.get(STARTUP_M64_KEY) != "tma"]
    assert len(filtered) == len(old)
    assert all(a is b for a, b in zip(filtered, old, strict=True))
    assert _with_chained_startup_seed(old, [], prefer_direct=True) is old
    assert _with_chained_startup_seed([legacy], [legacy], prefer_direct=True) == [
        legacy
    ]


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
        config = _startup_m64_config(direct=True)
        source = bound.to_code(config)
        assert source == bound.to_code(bound._normalized_config_copy(config))
        assert bound.to_code(_startup_m64_config(startup=None)) == bound.to_code(
            _startup_m64_config(startup="legacy")
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
    config = _startup_m64_config(
        direct=True,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_pointwise_inplace_async=True,
    )
    with cpu_codegen():
        bound = _one._bind_isolated(_startup_m64_values(dtype, 128, major))
        candidate = bound.to_code(config)
        config.config[STARTUP_M64_KEY] = "legacy"
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
        _one._bind_isolated(_startup_m64_values()).to_code(
            _startup_m64_config(cute_chained_pointwise_vectorize=True)
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
        _one._bind_isolated(_startup_m64_values()).to_code(
            _startup_m64_config(cute_chained_pointwise_vectorize=True)
        )


# Startup vector.

STARTUP_VECTOR_KEY = "cute_chained_startup_transfer"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("last_read", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_startup_vector_exports_and_last_read(dtype, last_read, cache):
    with _scan_export_cpu_codegen():
        bound = _vectors._bind_isolated((*_vector_export_args(dtype), "permuted"))
        config = _scan_export_config()
        config.config.update(
            {
                STARTUP_VECTOR_KEY: "tma",
                "cute_chained_auxiliary_cache": cache,
                "cute_chained_tmem_early_release": True,
            }
        )
        if last_read:
            config.config["cute_chained_tmem_free"] = "last_read"
        source = bound.to_code(config)
        assert source == bound.to_code(bound._normalized_config_copy(config))
    assert source.index("tma_bar_ptr=chain_start_bar") < source.index(
        "chain_scan_0_pointer"
    )
    assert source.index("mbarrier_wait(chain_start_bar, 0)") < source.index(
        "cute.gemm("
    )
    assert source.count("chain_allocator.free(chain_tptr)") == 1
    assert source.count("chain_allocator.relinquish_alloc_permit()") == 1
    assert "chain_export_1_index" in source and "chain_export_2_index" in source
    if last_read:
        assert source.index("chain_allocator.free(chain_tptr)") < source.index(
            "for chain_export_1_step"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_startup_fp32_scan_input_expression(dtype):
    with _scan_export_cpu_codegen():
        args = (
            torch.zeros(2, 256, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, dtype=dtype),
            "chain",
        )
        config = _scan_export_config()
        config.config[STARTUP_VECTOR_KEY] = "tma"
        source = _export._bind_isolated(args).to_code(config)
    assert "chain_export_0_index" in source
    assert "tma_bar_ptr=chain_start_bar" in source


def test_startup_m64_domain():
    with _scan_export_cpu_codegen():
        bound = _vectors._bind_isolated((*_vector_export_args(), "normal"))
        config = _scan_export_config()
        config.config[STARTUP_VECTOR_KEY] = "tma"
        config.config["block_sizes"] = [64, 64]
        source = bound.to_code(config)
        assert "tcgen05.Ld16x256bOp" in source
        assert "tma_bar_ptr=chain_start_bar" in source


@pytest.mark.parametrize("mode", ["full", "serial64", "overlap64"])
def test_startup_rejects_initialized_late_k64(mode):
    from test.test_cute_chained_pipeline import k_schedule_source as source

    with pytest.raises(exc.BackendUnsupported, match="uninitialized M128"):
        source(mode, **{STARTUP_VECTOR_KEY: "tma"})


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [True, 1, None, "unknown"])
def test_startup_invalid_never_repairs(repair, value):
    with _scan_export_cpu_codegen():
        bound = _vectors._bind_isolated((*_vector_export_args(), "normal"))
        raw = _scan_export_config().config | {STARTUP_VECTOR_KEY: value}
        with pytest.raises(exc.InvalidConfig, match="legacy or tma"):
            bound.config_spec.normalize(raw, _fix_invalid=repair)


def test_startup_resource_charge_and_padding_are_additive():
    from test.test_cute_chained_tcgen05 import _tcgen_chain
    from test.test_cute_chained_tcgen05 import _tcgen_inputs as _inputs

    with _scan_export_cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay"))
        counts = []
        original = chained_tcgen05.supported_plan

        def record(plan, *, startup=False):
            plain = chained_tcgen05._shared_memory_bytes(plan)
            charged = chained_tcgen05._shared_memory_bytes(plan, startup=True)
            assert charged == plain + 128
            with patch.object(
                chained_tcgen05.CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=charged - 1,
            ):
                assert not original(plan, startup=True)
            with patch.object(
                chained_tcgen05.CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=charged,
            ):
                assert original(plan, startup=True)
            counts.append(startup)
            return original(plan, startup=startup)

        with patch.object(chained_tcgen05, "supported_plan", side_effect=record):
            bound.to_code(startup_config())
    assert True in counts and False in counts


def test_startup_charge_reduces_early_cache_budget_once():
    budgets = []
    original = chained_tcgen05.make_early_auxiliary_cache

    def record(cg, plan, scans, budget):
        budgets.append(budget)
        return original(cg, plan, scans, budget)

    with (
        _scan_export_cpu_codegen(),
        patch.object(chained_tcgen05, "make_early_auxiliary_cache", side_effect=record),
    ):
        bound = _vectors._bind_isolated((*_vector_export_args(), "normal"))
        config = _scan_export_config()
        config.config["cute_chained_auxiliary_cache"] = True
        bound.to_code(config)
        config.config[STARTUP_VECTOR_KEY] = "tma"
        bound.to_code(config)
    assert len(budgets) == 2 and budgets[1] == budgets[0] - 128


def test_startup_single_sibling_after_complete_legacy_pool():
    from helion._compiler.cute import chained_startup

    with _scan_export_cpu_codegen():
        with patch.object(chained_startup, "has_startup_leaf", return_value=False):
            old = _vectors._bind_isolated((*_vector_export_args(), "normal"))
            old_pool = [dict(c.config) for c in old.config_spec.compiler_seed_configs]
            old_default = dict(old.config_spec.default_config().config)
        new = _vectors._bind_isolated((*_vector_export_args(), "normal"))
        new_pool = list(new.config_spec.compiler_seed_configs)
        assert [
            dict(c.config) for c in new_pool if STARTUP_VECTOR_KEY not in c.config
        ] == old_pool
        assert sum(c.config.get(STARTUP_VECTOR_KEY) == "tma" for c in new_pool) == 1
        assert dict(new.config_spec.default_config().config) == old_default
        assert new.host_function is not None
        with new.env:
            parents = CuteChainedMatmulHeuristic._tcgen05_seed_configs_for_rows(
                new.env, new.host_function.device_ir, 128
            )
        assert parents


def test_startup_partial_and_unsupported_parents_remain_unchanged():
    parent = startup_config(None)
    partial = helion.Config()
    initialized = helion.Config.from_dict(
        parent.config | {"cute_chained_initialized_accumulator": True}
    )
    old = [partial, initialized, partial]
    assert _with_chained_startup_seed(old, [parent]) is old
    assert _with_chained_startup_seed([parent], []) == [parent]
