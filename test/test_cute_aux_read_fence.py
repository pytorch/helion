from __future__ import annotations

import ast

import pytest
import torch

from test.test_cute_batched_aux_tma import _batched_aux
from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen
from test.test_cute_batched_aux_tma import _inputs
from test.test_cute_batched_aux_tma import _rank_two_aux

import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rank_two_multi_aux(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    m, k = lhs.shape
    n = rhs.shape[1]
    out = torch.empty_like(bias)
    for mi, ni in hl.tile([m, n]):
        acc = hl.zeros([mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.addmm(acc, lhs[mi, ki], rhs[ki, ni])
        out[mi, ni] = (acc * scale[mi, ni] + bias[mi, ni]).to(out.dtype)
    return out


def _rank_two_inputs(
    count: int,
    *,
    m: int = 256,
    n: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((m, 128), (128, n), *((m, n),) * count)
    )


def _assert_aux_read_fence_order(code: str, *, use_tma_load: bool = False) -> int:
    """Each staged release must immediately follow a per-reader fence."""
    count = 0
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.With):
            # The enclosing statement list checks a TMA elected release.
            # Do not count its inner expression again as a direct release.
            continue
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value):
                if isinstance(statement, ast.With) and len(statement.body) == 1:
                    body = statement.body[0]
                else:
                    body = statement
                if not isinstance(body, ast.Expr) or not isinstance(
                    body.value, ast.Call
                ):
                    continue
                function = body.value.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "consumer_release"
                    and isinstance(function.value, ast.Name)
                    and function.value.id.startswith("tcgen05_aux_pipeline")
                ):
                    continue
                assert index >= 2
                assert ast.unparse(value[index - 1]) == "cute.arch.fence_acq_rel_cta()"
                load = value[index - 2]
                assert isinstance(load, ast.Assign)
                assert ast.unparse(load.targets[0]).startswith("tcgen05_aux_loaded_")
                assert ast.unparse(load.value).endswith(".load()")
                assert isinstance(statement, ast.With) == use_tma_load
                if isinstance(statement, ast.With):
                    assert ast.unparse(statement.items[0].context_expr) == (
                        "cute.arch.elect_one()"
                    )
                count += 1
    return count


@pytest.mark.parametrize("kind", ("batched", "rank_two", "rank_two_multi"))
@pytest.mark.parametrize("mode", ("tma", "simt"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_staged_aux_read_completion_codegen(
    kind: str, mode: str, dtype: torch.dtype
) -> None:
    if kind == "batched":
        kernel = _batched_aux
        args = _inputs(dtype=dtype)
    else:
        multi = kind == "rank_two_multi"
        kernel = _rank_two_multi_aux if multi else _rank_two_aux
        args = _rank_two_inputs(2 if multi else 1, dtype=dtype)
    config = _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode=mode)
    with _cpu_codegen():
        code = kernel._bind_isolated(args).to_code(config)
    count = _assert_aux_read_fence_order(code, use_tma_load=mode == "tma")
    if kind == "batched" and mode == "simt":
        # Batched staging is currently TMA-only; the direct GMEM path has no
        # reusable shared ring and must not acquire this fence.
        assert count == 0
        assert "tcgen05_aux_pipeline.consumer_release" not in code
        assert "cute.arch.fence_acq_rel_cta()" not in code
    else:
        assert count > 0
        assert "tcgen05_aux_pipeline.consumer_wait" in code


@pytest.mark.parametrize("tails", (False, True))
def test_direct_aux_loads_do_not_gain_ring_fence(tails: bool) -> None:
    # One output fringe is supported by this role-local one-CTA strategy;
    # its unrelated double-edge geometry is intentionally not requested.
    args = _rank_two_inputs(2, m=257 if tails else 256, n=512)
    config = _config(
        block_sizes=[128, 128, 64],
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
    )
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(args).to_code(config)
    assert _assert_aux_read_fence_order(code) == 0
    assert "tcgen05_aux_pipeline.consumer_release" not in code
    assert "cute.arch.fence_acq_rel_cta()" not in code
    assert "tcgen05_aux_loaded_" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ("tma", "simt"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_rank_two_aux_ring_repeatability(mode: str, dtype: torch.dtype) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 F16/BF16 support")
    for seed in range(5):
        torch.manual_seed(seed)
        args = tuple(
            torch.randn_like(value, device=DEVICE) * 0.1
            for value in _rank_two_inputs(2, m=512, n=8192, dtype=dtype)
        )
        before = tuple(value.clone() for value in args)
        lhs, rhs, bias, scale = args
        expected = (lhs.double() @ rhs.double()) * scale.double() + bias.double()
        bound = _rank_two_multi_aux._bind_isolated(args)
        bound.set_config(
            _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode=mode)
        )
        first = bound(*args)
        second = bound(*args)
        assert torch.equal(second, first)
        for _ in range(3):
            assert torch.equal(bound(*args), first)
        torch.testing.assert_close(first.double(), expected, atol=0.002, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(*args)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            assert torch.equal(captured, first)
        for current, previous in zip(args, before, strict=True):
            assert torch.equal(current, previous)
