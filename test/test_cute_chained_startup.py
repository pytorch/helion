from __future__ import annotations

import ast
import contextlib
import inspect
import types
from typing import Any
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tcgen05 import _inputs
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_single
import helion
from helion import exc
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


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


def config(mode: str | None = "tma", n: int = 64) -> helion.Config:
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
        bound = _tcgen_chain._bind_isolated((*_inputs("cpu", dtype, n=n), "decay"))
        source = bound.to_code(config(n=n))
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
        source = _tcgen_single._bind_isolated((*values, True, True)).to_code(config())
    assert source.index("tma_bar_ptr=chain_start_bar") < source.index(
        "chain_scan_0_pointer"
    )
    assert "chain_start_b_index" in source
    assert source.index("mbarrier_wait(chain_start_bar, 0)") < source.index(
        "chain_start_b_index"
    )


def test_legacy_identity() -> None:
    with cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay"))
        assert bound.to_code(config(None)) == bound.to_code(config("legacy"))


@pytest.mark.parametrize("mode", [False, True, 1, None, "invalid"])
def test_invalid_enum(mode: Any) -> None:
    cfg = config()
    cfg.config["cute_chained_startup_transfer"] = mode
    with cpu_codegen(), pytest.raises(exc.InvalidConfig):
        _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay")).to_code(cfg)


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
        bound = _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay"))
        raw = config()
        normalized = bound.env.config_spec.normalized_config(raw)
        assert bound.to_code(raw) == bound.to_code(normalized)


def test_no_eligible_layout_fails_closed() -> None:
    args = list(_inputs("cpu"))
    args[0] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    args[1] = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _tcgen_chain._bind_isolated((*args, "decay")).to_code(config())


def test_fp32_raw_leaf_does_not_borrow_a_narrow_arena() -> None:
    a = torch.empty((2, 128, 256), dtype=torch.bfloat16)[..., ::2]
    b = torch.empty((2, 128, 64), dtype=torch.float32)
    coeff = torch.empty((2, 128), dtype=torch.float32)
    with (
        cpu_codegen(),
        pytest.raises(exc.BackendUnsupported, match="no full bijective"),
    ):
        _tcgen_single._bind_isolated((a, b, coeff, True, True)).to_code(config())


def test_holey_read_sources_do_not_get_a_startup_seed() -> None:
    args = list(_inputs("cpu"))
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
    cfg = config()
    cfg.config["cute_chained_mma_schedule"] = mode
    with cpu_codegen(), pytest.raises(exc.InvalidConfig, match="TCgen05"):
        _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay")).to_code(cfg)


def test_seed_preserves_objects_order_multiplicity() -> None:
    from helion._compiler.autotuner_heuristics.cute import _with_chained_startup_seed

    parent = config(None)
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
