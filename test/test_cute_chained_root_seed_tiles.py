from __future__ import annotations

import pytest
import torch

from ._cute_prepared_source import expand_prepared_root_source
from .test_cute_chained_accumulator import _initialized_args
from .test_cute_chained_accumulator import _initialized_code
from .test_cute_chained_accumulator import _initialized_config
from .test_cute_chained_accumulator import _initialized_pair
from .test_cute_chained_loop_tmem_transport import _source
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

KEY = "cute_chained_seed_tile_columns"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _explicit_seed(a, b, seed):
    m, k = a.shape
    n = b.size(1)
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, columns in hl.tile([m, n], block_size=[128, n]):
        kk = hl.arange(k)
        left = (a[rows, kk].float() * 1.01).to(a.dtype)
        output[rows, columns] = hl.dot(
            left, b[kk, columns], acc=seed[rows, columns] * 2.0 - 1.0
        )
    return output


def _explicit_args(n, dtype, device="cpu"):
    return (
        torch.empty((128, 32), dtype=dtype, device=device),
        torch.empty((32, n), dtype=dtype, device=device),
        torch.empty((128, n), dtype=torch.float32, device=device),
    )


def _explicit_config(columns):
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_seed_tile_columns=columns,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,columns", [(64, 32), (96, 32), (128, 64), (256, 32)])
def test_root_initialized_panels_cpu(dtype, n, columns):
    args = _initialized_args(dtype=dtype, n=n)
    before = _initialized_code(args)
    assert before == _initialized_code(args, **{KEY: 0})
    after = expand_prepared_root_source(_initialized_code(args, **{KEY: columns}))
    assert "chain_0_values =" in before and "chain_0_values =" not in after
    wait = after.index("cute.arch.mbarrier_wait(chain_bars + 0, 0)")
    for offset in range(0, n, columns):
        prefix = f"chain_seed_panel_{offset}"
        load = after.index(f"cute.copy({prefix}_copy,")
        store = after.index(f"cute.copy({prefix}_store_copy,")
        assert wait < load < store
        assert f"{prefix}_row, {prefix}_col = {prefix}_coords[" in after
        assert f"cute.domain_offset(((0, {offset}), 0, 0), chain_0_acc)" in after
    assert after.count("chain_seed_panel_") > 0
    assert after.index("cute.arch.fence_view_async_tmem_store()") < after.index(
        "cute.gemm(chain_1_mma,"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,columns", [(64, 32), (96, 32), (128, 64)])
def test_root_explicit_panels_cpu(dtype, n, columns):
    args = _explicit_args(n, dtype)
    before = _source(_explicit_seed, args, _explicit_config(0))
    after = _source(_explicit_seed, args, _explicit_config(columns))
    assert before.count("cute.copy(chain_0_seed_copy,") == 1
    assert after.count("cute.copy(chain_0_seed_copy,") == n // columns
    for offset in range(0, n, columns):
        assert f"cute.domain_offset(((0, {offset}), 0, 0), chain_0_acc)" in after
    assert after.count("cute.arch.fence_view_async_tmem_store()") == before.count(
        "cute.arch.fence_view_async_tmem_store()"
    )


@pytest.mark.parametrize("implicit", [False, True])
def test_root_ineffective_seed_tiles_reject_cpu(implicit):
    with pytest.raises(helion.exc.BackendUnsupported, match="multi-panel"):
        if implicit:
            _initialized_code(_initialized_args(n=32), **{KEY: 32})
        else:
            _source(
                _explicit_seed, _explicit_args(32, torch.bfloat16), _explicit_config(32)
            )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,columns", [(64, 32), (96, 32), (128, 64)])
@pytest.mark.parametrize("implicit", [False, True])
def test_root_seed_tiles_gpu_bitwise_replay_inputs(dtype, n, columns, implicit):
    torch.manual_seed(821)
    if implicit:
        source_args = _initialized_args(dtype=dtype, n=n)
        args = tuple(
            torch.randn_like(arg, device=DEVICE) * 0.1
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in source_args
        )
        kernel = _initialized_pair
        config = _initialized_config(n)
    else:
        args = tuple(
            torch.randn_like(arg) * 0.1 for arg in _explicit_args(n, dtype, DEVICE)
        )
        kernel = _explicit_seed
        config = _explicit_config(0)
    originals = [arg.clone() for arg in args if isinstance(arg, torch.Tensor)]
    before = kernel._bind_isolated(args)
    before.set_config(config)
    after = kernel._bind_isolated(args)
    after.set_config(helion.Config.from_dict(config.config | {KEY: columns}))
    expected = before(*args)
    for _ in range(3):
        torch.testing.assert_close(after(*args), expected, rtol=0, atol=0)
    for actual, original in zip(
        (arg for arg in args if isinstance(arg, torch.Tensor)), originals, strict=True
    ):
        assert torch.equal(actual, original)
