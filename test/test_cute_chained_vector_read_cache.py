from __future__ import annotations

import ast
import copy
import importlib
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_pointwise_cache import _expression
import helion
from helion._compiler.cute.chained_pointwise_cache import PointwiseReadCache
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_pointwise_read_cache"


@helion.kernel(backend="cute", static_shapes=True)
def _pair(a, b, raw, x, prefix, delta):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        ll = hl.arange(raw.size(1))
        first = hl.dot(a[row, kk], b[kk, col])
        seed = first * torch.exp(prefix[row])[:, None]
        scale = torch.exp((prefix[row][:, None] - prefix[ll][None, :]).clamp(max=0.0))
        weighted = raw[row, ll] * scale
        weighted = weighted * delta[ll][None, :]
        weighted = torch.where(row.index[:, None] >= ll[None, :], weighted, 0.0)
        second = hl.dot(weighted.to(a.dtype), x[ll, col])
        out[row, col] = (seed + second + x[row, col].float()).to(out.dtype)
    return out


def _args(
    dtype: torch.dtype = torch.bfloat16,
    n: int = 64,
    view: str = "dense",
) -> tuple[torch.Tensor, ...]:
    values = (
        torch.empty((128, 128), dtype=dtype),
        torch.empty((128, n), dtype=dtype),
        torch.empty((128, 128), dtype=torch.float32),
        torch.empty((128, n), dtype=dtype),
        torch.empty(128, dtype=torch.float32),
        torch.empty(128, dtype=torch.float32),
    )
    prefix, delta = values[-2:]
    if view == "offset":
        prefix = torch.empty(132)[4:]
        delta = torch.empty(132)[4:]
    elif view == "unaligned":
        prefix = torch.empty(129)[1:]
        delta = torch.empty(129)[1:]
    elif view == "stride":
        prefix = torch.empty(256)[::2]
        delta = torch.empty(256)[::2]
    elif view == "alias":
        delta = prefix
    return (*values[:-2], prefix, delta)


def _config(cache: bool | None = True, schedule: str = "serial64") -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_initialized_accumulator": True,
        "cute_chained_late_rhs_reuse": True,
        "cute_chained_k_schedule": schedule,
        "cute_chained_auxiliary_cache": False,
    }
    if cache is not None:
        values[KEY] = cache
    return helion.Config.from_dict(values)


def _source(args: tuple[torch.Tensor, ...], cache=True, schedule="serial64") -> str:
    with _cpu_codegen():
        return _pair._bind_isolated(args).to_code(_config(cache, schedule))


def _inverse_vector_hoist(source: str) -> str:
    tree = ast.parse(source)
    changed = 0
    for branch in ast.walk(tree):
        if not isinstance(branch, ast.If):
            continue
        loops = [
            (i, node)
            for i, node in enumerate(branch.body)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.endswith("_pointwise_step")
        ]
        if not loops:
            continue
        loop_index, loop = loops[0]
        preloads = [
            node
            for node in branch.body[:loop_index]
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.copy"
            and "_leaf_" in ast.unparse(node.value.args[1])
        ]
        if not preloads:
            continue
        changed += len(preloads)
        for node in preloads:
            assert isinstance(node.value, ast.Call)
            source_arg = node.value.args[1]
            assert isinstance(source_arg, ast.Subscript)
            assert isinstance(source_arg.slice, ast.Tuple)
            assert ast.unparse(source_arg.slice.elts[1]) == "0"
            source_arg.slice.elts[1] = copy.deepcopy(loop.target)
            branch.body.remove(node)
        # Existing row-dependent raw load remains first; original vectors are
        # ordered after it. No other statement may move in this inverse.
        assert ast.unparse(loop.body[1]).startswith("cute.copy(")
        loop.body[2:2] = preloads
    assert changed == 2
    return ast.unparse(ast.fix_missing_locations(tree))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("schedule", ["full", "serial64", "overlap64"])
def test_whole_source_inverse_and_defaults(dtype, n, schedule) -> None:
    args = _args(dtype, n)
    disabled = _source(args, False, schedule)
    assert disabled == _source(args, None, schedule)
    enabled = _source(args, True, schedule)
    assert _inverse_vector_hoist(enabled) == ast.unparse(ast.parse(disabled))


@pytest.mark.parametrize("view", ["offset", "unaligned", "alias"])
def test_guard_fallback_and_readonly_alias_are_unchanged(view: str) -> None:
    args = _args(view=view)
    old, new = _source(args, False), _source(args, True)
    assert _inverse_vector_hoist(new) == ast.unparse(ast.parse(old))
    assert "toint() % 16 == 0" in new and "layout.stride[0] == 1" in new


def test_strided_vectors_keep_original_scalar_cache() -> None:
    source = _source(_args(view="stride"), True)
    assert "_read_cache_" in source and "layout.stride[0]" in source


@pytest.mark.parametrize(
    "rhs",
    [
        "src[col + row]",
        "src[col] if row < 127 else cutlass.Float32(0)",
        "src[col] if col < limit else cutlass.Float32(0)",
        "src[row]",
        "src[0]",
    ],
)
def test_vector_complete_rhs_rejects_unproved_dependence(rhs: str) -> None:
    expression = _expression(rhs, torch.float32)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    expression.definitions["limit"] = "row + 1"
    old = list(expression.lines)
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    assert expression.lines == old


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int32])
def test_vector_transport_does_not_widen_dtype_family(dtype: torch.dtype) -> None:
    expression = _expression("src[col]", dtype)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128, dtype=dtype)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)


def test_vector_read_catalog_disabled_trip_rank_and_capacity() -> None:
    expression = _expression("src[col]", torch.float32)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    assert PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    for cache, trips in ((False, 8), (True, 1)):
        assert not PointwiseReadCache(cache).vector_reads(
            expression, ("row", "col"), trips
        )
    expression.loaded_inputs = [(node, ("row", "col"), indices, value)]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    expression.loaded_inputs = [
        (node, (f"col + {i}",), indices, value) for i in range(5)
    ]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)


def test_copy_coordinate_and_fp32_bit_identity() -> None:
    bits = torch.tensor(
        [0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2143289346, 1065353216],
        dtype=torch.int32,
    ).repeat(16)
    for half in range(2):
        for thread in range(128):
            cols = half * 64 + thread % 8 * 8 + torch.arange(8)
            retained = bits[cols].view(torch.float32).clone().view(torch.int32)
            for step in range(8):
                row = thread // 8 + step * 16
                assert 0 <= row < 128
                assert torch.equal(retained, bits[cols])


@pytest.mark.parametrize("width", [64, 128])
def test_actual_static_fp32_partition_has_identical_row_addresses(width: int) -> None:
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            columns, rows = width // 8, 128 // (width // 8)
            tiled_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.Float32,
                    num_bits_per_copy=128,
                ),
                cute.make_layout((rows, columns), stride=(columns, 1)),
                cute.make_layout((1, 8)),
            )
            physical = cute.make_layout((128, 128), stride=(0, 1))
            for half in range(128 // width):
                tile = cute.local_tile(
                    cute.make_identity_tensor((128, 128)), (128, width), (0, half)
                )
                for thread in range(128):
                    coords = tiled_copy.get_slice(thread).partition_S(tile)
                    assert int(cute.size(coords[None, 0, 0])) == 8
                    for element in range(8):
                        first = tuple(map(int, coords[element, 0, 0]))
                        for step in range(128 // rows):
                            current = tuple(map(int, coords[element, step, 0]))
                            assert current == (
                                thread // columns + step * rows,
                                half * width + thread % columns * 8 + element,
                            )
                            assert int(physical(current)) == int(physical(first))
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before
