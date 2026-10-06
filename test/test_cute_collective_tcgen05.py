from __future__ import annotations

import ast

from examples.moe_matmul_ogs import moe_matmul_ogs
import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_matmul import _aliased_matmul
from test.test_cute_collective_matmul import _computed_gathered_matmul
from test.test_cute_collective_matmul import _config
from test.test_cute_collective_matmul import _padded_inputs
from test.test_cute_collective_seeds import _independent_matmuls

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_tcgen05 import CollectiveTcgen05Plan
from helion._compiler.cute.collective_tcgen05 import CollectiveTmemResource
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

CUDA_DEVICE = "cuda"


def _native_config(
    bm: int = 64, bn: int = 32, bk: int = 64, *, copy: str = "async_cached"
) -> helion.Config:
    return helion.Config.from_dict(
        dict(_config(bm, bn, bk, copy=copy)) | {"cute_collective_compute": "tcgen05"}
    )


def _kernel_ast(source: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _scalar_epilogues(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.walk(_kernel_ast(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("lane_")
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [(64, 32, 16), (64, 64, 32), (128, 64, 128)])
@skipUnlessBackends(["cute"])
def test_native_compute_preserves_computed_operand_and_epilogue(
    dtype: torch.dtype, tile: tuple[int, int, int]
) -> None:
    bound = _cpu_bind(
        _computed_gathered_matmul,
        (
            torch.empty((193, 78), dtype=dtype),
            torch.empty((3, 78, 70), dtype=dtype),
            torch.empty((3, 130), dtype=torch.int32),
        ),
    )
    warp = bound.to_code(_config(*tile, copy="async_cached"))
    native = bound.to_code(_native_config(*tile))
    assert "tcgen05.CtaGroup.ONE" in native
    assert "LdMatrix8x8x16bOp" not in native
    # This exact equality includes indirect addressing, casts, scalar output
    # stores, and tail masks. The native choice must not rewrite that program.
    assert _scalar_epilogues(native) == _scalar_epilogues(warp)
    assert _scalar_epilogues(native)
    assert "0.5" in native
    assert "rows.iterator" in native


@skipUnlessBackends(["cute"])
def test_native_scatter_retains_serial_row_loop_and_masks() -> None:
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    bound = _cpu_bind(kernel, (*_padded_inputs("cpu", torch.float16), 97))
    native = bound.to_code(_native_config(128, 64, 128))
    warp = bound.to_code(_config(128, 64, 128, copy="async_cached"))
    assert _scalar_epilogues(native) == _scalar_epilogues(warp)
    fn = _kernel_ast(native)
    allocate = next(
        index
        for index, statement in enumerate(fn.body)
        if ".allocate(" in ast.unparse(statement)
    )
    traversal = next(
        index
        for index, statement in enumerate(fn.body)
        if any(
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "tile_offset_1"
            for node in ast.walk(statement)
        )
    )
    relinquish = next(
        index
        for index, statement in enumerate(fn.body)
        if ".relinquish_alloc_permit(" in ast.unparse(statement)
    )
    free = next(
        index
        for index, statement in enumerate(fn.body)
        if ".free(" in ast.unparse(statement)
    )
    assert allocate < relinquish < traversal < free
    assert native.count(".allocate(") == 1
    assert native.count(".relinquish_alloc_permit(") == 1
    assert "existing_values =" not in native


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (12, 0), None])
@skipUnlessBackends(["cute"])
def test_native_config_requires_supported_architecture(
    capability: tuple[int, int] | None,
) -> None:
    bound = _cpu_bind(
        _computed_gathered_matmul,
        (
            torch.empty((193, 78), dtype=torch.float16),
            torch.empty((3, 78, 70), dtype=torch.float16),
            torch.empty((3, 130), dtype=torch.int32),
        ),
    )
    bound.config_spec.target_device_capability = capability
    with pytest.raises(InvalidConfig, match="requires SM100-family"):
        bound.to_code(_native_config())


@skipUnlessBackends(["cute"])
def test_alias_rejection_still_applies_with_native_compute() -> None:
    bound = _cpu_bind(
        _aliased_matmul,
        (
            torch.empty((3, 64, 64), dtype=torch.float16),
            torch.empty((64, 64), dtype=torch.float16),
        ),
    )
    source = bound.to_code(_native_config())
    assert "tcgen05.CtaGroup.ONE" not in source
    assert "TmemAllocator" not in source


@skipUnlessBackends(["cute"])
def test_two_sequential_native_sites_share_one_allocation() -> None:
    bound = _cpu_bind(
        _independent_matmuls,
        tuple(
            torch.empty(shape, dtype=torch.float16)
            for shape in ((128, 256), (256, 64), (96, 64), (64, 256))
        ),
    )
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    native = [
        seed for seed in seeds if seed.get("cute_collective_compute") == "tcgen05"
    ]
    assert native
    source = bound.to_code(native[0])
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert source.count(".allocate(") == 1
    assert source.count(".relinquish_alloc_permit(") == 1
    assert source.count(".free(") == 1


def test_empty_reduction_has_no_unconditional_tmem_read() -> None:
    plan = CollectiveTcgen05Plan(
        "tile", "tid", 64, 32, 16, "cutlass.Float16", CollectiveTmemResource("mem", 32)
    )
    statements = plan.finish(ast.Constant(0), ast.Name(id="K", ctx=ast.Load()))
    conditional = next(node for node in statements if isinstance(node, ast.If))
    assert ast.unparse(conditional.test) == "0 < K"
    assert "cute.copy(" in ast.unparse(conditional)
    unconditional = ast.unparse(
        ast.Module(
            body=[node for node in statements if not isinstance(node, ast.If)],
            type_ignores=[],
        )
    )
    assert "cute.copy(" not in unconditional
    assert ".fill(0.0)" in unconditional


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bn", [32, 64])
def test_shared_result_layout_is_bijective_and_avoids_row_store_bank_conflicts(
    bm: int,
    bn: int,
) -> None:
    plan = CollectiveTcgen05Plan(
        "tile", "tid", bm, bn, 64, "cutlass.Float16", CollectiveTmemResource("mem", bn)
    )
    bits, base, shift = plan.c_swizzle

    def address(row: int, column: int) -> int:
        logical = row * bn + column
        return logical ^ ((logical >> shift) & (((1 << bits) - 1) << base))

    assert {address(row, col) for row in range(bm) for col in range(bn)} == set(
        range(bm * bn)
    )
    for row in range(bm):
        # The original scalar epilogue reads one N-column per lane.
        assert len({address(row, lane) % 32 for lane in range(32)}) == 32
        for col in range(0, bn, 4):
            vector = [address(row, col + element) for element in range(4)]
            assert vector == list(range(vector[0], vector[0] + 4))
    for row in range(0, bm, 8):
        for col in range(0, bn, 4):
            # One 128-byte shared-store transaction: eight lanes, one 16-byte
            # vector per lane, consecutive rows. Every bank is used once.
            assert (
                len(
                    {
                        address(row + lane, col + element) % 32
                        for lane in range(8)
                        for element in range(4)
                    }
                )
                == 32
            )


@pytest.mark.parametrize(
    "bm,bn,bk,dtype,columns",
    [
        (32, 32, 16, "cutlass.Float16", 32),
        (64, 128, 16, "cutlass.Float16", 128),
        (64, 32, 8, "cutlass.Float16", 32),
        (64, 32, 16, "cutlass.Float32", 32),
        (64, 32, 16, "cutlass.Float16", 16),
    ],
)
def test_native_plan_rejects_unsupported_geometry(
    bm: int, bn: int, bk: int, dtype: str, columns: int
) -> None:
    with pytest.raises(ValueError):
        CollectiveTcgen05Plan(
            "tile", "tid", bm, bn, bk, dtype, CollectiveTmemResource("mem", columns)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100-family CUDA",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty_k", [False, True])
@pytest.mark.parametrize(
    "tile",
    [(64, 32, 64), (128, 32, 64), (64, 64, 16), (64, 32, 32), (128, 64, 128)],
)
@pytest.mark.parametrize("copy", ["async", "async_cached"])
@skipUnlessBackends(["cute"])
def test_native_scatter_repeated_rows_multiple_tiles_and_empty_k(
    dtype: torch.dtype, empty_k: bool, tile: tuple[int, int, int], copy: str
) -> None:
    torch.manual_seed(827)
    a, b, counts, offsets, rows = _padded_inputs("cuda", dtype)
    counts = torch.tensor([0, 193, 0], device=CUDA_DEVICE, dtype=torch.int32)
    offsets = torch.tensor([0, 0, 193, 193], device=CUDA_DEVICE, dtype=torch.int32)
    # Duplicates occur within one expert, across M-tile boundaries. Each source
    # row has the same product at every occurrence; no uniqueness is assumed.
    rows = torch.randint(0, 193, (193,), device=CUDA_DEVICE, dtype=torch.int32)
    if empty_k:
        a = a[:, :0]
        b = b[:, :0]
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    inputs = (a, b, counts, offsets, rows, 193)
    bound = kernel._bind_isolated(inputs)
    config = _native_config(*tile, copy=copy)
    assert "tcgen05.CtaGroup.ONE" in bound.to_code(config)
    bound.set_config(config)
    expected = torch.zeros((193, 70), device=CUDA_DEVICE, dtype=dtype)
    expected[rows.long()] = (a[rows.long()].float() @ b[1].float()).to(dtype)
    for _ in range(2):
        torch.testing.assert_close(bound(*inputs), expected, rtol=0.02, atol=0.02)
