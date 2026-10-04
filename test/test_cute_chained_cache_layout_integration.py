from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_cache_set_integration import _args
from .test_cute_chained_cache_set_integration import _config
from .test_cute_chained_cache_set_integration import _loop_cache_set
from .test_cute_chained_cache_set_integration import _root_cache_set
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config as _prefill_config
import helion
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

KEY = "cute_chained_pointwise_cache_layout"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp32_seed_cache(a, b):
    m, k = a.shape
    n = b.size(-1)
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        first = hl.dot(a[rows, kk], b[kk, cols])
        repeated = torch.sigmoid(first) * 0.125
        second = hl.dot(a[rows, kk], b[kk, cols], acc=repeated)
        output[rows, cols] = hl.dot(a[rows, kk], b[kk, cols], acc=repeated + second)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rank1_cache(a, b):
    m, k = a.shape
    n = b.size(-1)
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        first = hl.dot(a[rows, kk], b[kk, cols])
        coefficient = torch.sigmoid(torch.sum(first, dim=1))
        left = (a[rows, kk] * coefficient[:, None]).to(a.dtype)
        second = hl.dot(left, b[kk, cols])
        output[rows, cols] = hl.dot(left, b[kk, cols], acc=second)
    return output


def test_rank1_only_cache_cannot_activate_xor_cpu():
    args = (
        torch.empty((128, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=torch.bfloat16),
    )
    config = helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="coalesced",
        cute_chained_pointwise_cache_bytes=4096,
    )
    before = _source(_rank1_cache, args, config)
    assert len(_cache_assignments(before)) == 1
    assert "make_layout((128,))" in ast.unparse(_cache_assignments(before)[0])
    config.config[KEY] = "xor"
    with pytest.raises(
        helion.exc.BackendUnsupported, match="eligible materialized typed cache"
    ):
        _source(_rank1_cache, args, config)


def test_explicit_cache_layout_rejects_late_codegen_fallback_cpu():
    args = _args("cpu", False, torch.bfloat16, 1)
    config = _config(False, 2)
    config.config.update(cute_chained_mma_schedule="coalesced", **{KEY: "xor"})
    with (
        patch(
            "helion._compiler.cute.chained_pointwise_residency.allocate_pointwise_cache",
            side_effect=_UnsupportedChain("forced after plan admission"),
        ),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match="unsupported chain with explicit cache layout",
        ),
    ):
        _source(_root_cache_set, args, config)


def _cache_assignments(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("chain_pointwise_cache_")
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_tensor"
    ]


def _assert_layout_only(before, after):
    old = ast.parse(before)
    new = ast.parse(after)
    old_assignments = _cache_assignments(before)
    new_assignments = _cache_assignments(after)
    assert old_assignments and len(old_assignments) == len(new_assignments)
    replacements = {}
    for left, right in zip(old_assignments, new_assignments, strict=True):
        assert isinstance(left.value, ast.Call) and isinstance(right.value, ast.Call)
        assert isinstance(right.targets[0], ast.Name)
        assert ast.dump(left.targets[0]) == ast.dump(right.targets[0])
        assert ast.dump(left.value.args[0]) == ast.dump(right.value.args[0])
        assert "make_swizzle" in ast.unparse(right.value.args[1])
        replacements[right.targets[0].id] = left.value.args[1]
    for node in ast.walk(new):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in replacements
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.make_tensor"
        ):
            node.value.args[1] = replacements[node.targets[0].id]
    assert ast.dump(old) == ast.dump(new)


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_typed_cache_layout_is_the_only_generated_change_cpu(loop, dtype):
    kernel = _loop_cache_set if loop else _root_cache_set
    args = _args("cpu", loop, dtype, 3)
    config = _config(loop, 2)
    original = _source(kernel, args, config)
    explicit = _source(
        kernel, args, helion.Config.from_dict(config.config | {KEY: "auto"})
    )
    assert original == explicit
    changed = _source(
        kernel, args, helion.Config.from_dict(config.config | {KEY: "xor"})
    )
    _assert_layout_only(original, changed)


@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("leaf", ["legacy", "rectangular_tma"])
def test_prefill_layout_preserves_native_operands_carry_and_protocol_cpu(
    value_tile, leaf
):
    kernel, args = _kda_fixture()
    config = _prefill_config(0, pipeline=True, value_tile=value_tile)
    config.config["cute_chained_leaf_pipeline"] = leaf
    before = _source(kernel, args, config)
    config.config[KEY] = "xor"
    after = _source(kernel, args, config)
    _assert_layout_only(before, after)
    assert after.count("cute.make_swizzle(5, 0, 5), 0, cute.make_layout((32, 32)") >= 2


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_fp32_cache_layout_uses_the_same_logical_policy_cpu(schedule):
    args = (
        torch.empty((128, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=torch.bfloat16),
    )
    config = helion.Config(
        num_warps=4,
        cute_chained_mma_schedule=schedule,
        cute_chained_pointwise_cache_bytes=16384,
    )
    before = _source(_fp32_seed_cache, args, config)
    config.config[KEY] = "xor"
    after = _source(_fp32_seed_cache, args, config)
    _assert_layout_only(before, after)
    cache = _cache_assignments(after)[0].value
    assert isinstance(cache, ast.Call)
    assert "cutlass.Float32" in ast.unparse(cache.args[0])


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop,steps", [(False, 1), (True, 0), (True, 1), (True, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cache_layout_gpu_bitwise_replay_and_inputs(loop, steps, dtype):
    kernel = _loop_cache_set if loop else _root_cache_set
    args = _args(DEVICE, loop, dtype, steps)
    saved = tuple(arg.clone() for arg in args if isinstance(arg, torch.Tensor))
    functions = []
    for layout in ("auto", "xor"):
        config = _config(loop, 2)
        config.config[KEY] = layout
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            functions.append(bound.compile_config(config))
    expected, actual = functions[0](*args), functions[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(functions[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(
        tuple(arg for arg in args if isinstance(arg, torch.Tensor)),
        saved,
        atol=0,
        rtol=0,
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_fp32_cache_layout_gpu(schedule):
    torch.manual_seed(9981)
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.125
        for shape in ((128, 32), (32, 32))
    )
    saved = tuple(arg.clone() for arg in args)
    functions = []
    for layout in ("auto", "xor"):
        config = helion.Config(
            num_warps=4,
            cute_chained_mma_schedule=schedule,
            cute_chained_pointwise_cache_bytes=16384,
        )
        config.config[KEY] = layout
        functions.append(_fp32_seed_cache._bind_isolated(args).compile_config(config))
    expected, actual = functions[0](*args), functions[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(functions[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_cache_layout_prefill_gpu(bt32, value_tile, consumer_warps):
    _run_preparation_prefill(bt32, value_tile, consumer_warps, cache_layout="xor")


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
def test_cache_layout_combined_mechanisms_gpu(bt32):
    _run_preparation_prefill(
        bt32,
        128,
        4,
        rectangular_leaf=True,
        cache_entries=4,
        seed_columns=32,
        cache_layout="xor",
    )
