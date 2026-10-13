from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_parallel import _parallel_sequence
from .test_cute_chained_row_registers import _config
from .test_cute_chained_scan_input import _args as _scan_args
from .test_cute_chained_scan_input import _config as _scan_config
from .test_cute_chained_scan_input import _sequence as _scan_sequence
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
import helion.language as hl


def _args(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((3, 16, 64), (3, 64, 16), (3, 16, 32))
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _combined(raw, scale, a, b, rhs):
    output = torch.empty((a.size(0), 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.size(0), block_size=1):
        index = hl.arange(16)
        values = raw[batch.begin, :].float()
        steps = scale[batch.begin] * torch.sigmoid(values)
        prefix = hl.cumsum(steps, 0)
        base = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        total = hl.full([16, 32], 0, torch.float32)
        for island in hl.static_range(4):
            original = base * (1 + island / 8) + prefix[:, None]
            state = original
            for position in hl.static_range(1, 3):
                row = torch.where(index[:, None] == position, state, 0.0).sum(0)
                col = torch.where(index[None, :] == position, original, 0.0).sum(1)
                state = torch.tanh(state * 0.5 + col[:, None] * row[None, :] + 0.0625)
            total = hl.dot(state.to(a.dtype), rhs[batch.begin, :, :], acc=total)
        output[batch.begin, :, :] = total.to(a.dtype)
    return output


def _math_calls(source: str) -> list[str]:
    return [
        ast.dump(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in ("cute.gemm", "cute.arch.shuffle_sync", "execute_prepared_warp_k")
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["rows", "scan", "combined"])
def test_existing_raw_copy_composes_with_original_collectives(dtype, kind) -> None:
    if kind == "rows":
        kernel, args = _parallel_sequence, (*_args(dtype), 4, True)
        config = _config("warp_rows_parallel")
    elif kind == "scan":
        kernel, args = _scan_sequence, _scan_args(dtype)
        config = _scan_config()
    else:
        kernel = _combined
        args = (torch.empty((3, 16), dtype=dtype), torch.empty((3,)), *_args(dtype))
        config = _config("warp_rows_parallel", cute_chained_scan_input_materialize=True)
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        config = helion.Config.from_dict(
            config.config | {"cute_native_matmul_metadata": True}
        )
        original = bound.to_code(config)
        copied = bound.to_code(
            helion.Config.from_dict(
                config.config | {"cute_chained_mma_schedule": "cp_async"}
            )
        )
    assert "CopyG2SOp(" not in original
    assert "CopyG2SOp(" in copied
    assert "cp_async_commit_group()" in copied
    assert "cp_async_wait_group(0)" in copied
    assert copied.count("cute.arch.sync_threads()") == original.count(
        "cute.arch.sync_threads()"
    )
    assert _math_calls(copied) == _math_calls(original)


def test_combined_copy_seed_keeps_old_pool_and_obeys_explicit_choices() -> None:
    args = (
        torch.empty((3, 16), dtype=torch.bfloat16),
        torch.empty((3,)),
        *_args(torch.bfloat16),
    )
    with _cpu_codegen():
        bound = _combined._bind_isolated(args)
        seeds = bound.config_spec.compiler_seed_configs
        seed = seeds[-1]
        assert seed.config["cute_chained_mma_schedule"] == "cp_async"
        assert seed.config["cute_chained_pointwise_schedule"] == "warp_rows_parallel"
        assert seed.config["cute_chained_scan_input_materialize"] is True
        append = CuteChainedMatmulHeuristic._with_row_async_copy_seed
        assert append(bound.env, seeds[:-1]) == seeds
        assert append(bound.env, seeds) == seeds
        for choices in (
            {"cute_chained_mma_schedule": "coalesced"},
            {"cute_chained_pointwise_schedule": "shared"},
            {"cute_chained_scan_input_materialize": False},
            {"cute_native_matmul_metadata": False},
        ):
            with patch.dict(bound.env.settings.autotune_config_overrides, choices):
                assert append(bound.env, seeds[:-1]) == seeds[:-1]
        generation = bound.config_spec.create_config_generation()
        flat = generation.flatten(seed)
        assert generation.flatten(generation.unflatten(flat)) == flat
        assert "CopyG2SOp(" in bound.to_code(seed)
