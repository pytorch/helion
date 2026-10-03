from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_collectives import _inputs
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cooperative_loop(left, wide_rhs, rhs, mixed_rhs, initial, steps: int):
    steps = hl.specialize(steps)
    _, m, k = left.shape
    n = rhs.shape[-1]
    history = torch.empty(
        (max(steps, 1), m, n), dtype=torch.float32, device=left.device
    )
    final = torch.empty_like(initial)
    for rows, columns in hl.tile([m, n], block_size=[16, 8]):
        state = initial[rows, columns]
        for step in hl.tile(steps, block_size=1):
            kk, intermediate = hl.arange(k), hl.arange(16)
            common = left[step.id, rows, kk]
            first = hl.dot(
                common, wide_rhs[step.id, kk, intermediate], out_dtype=torch.float32
            )
            state = hl.dot(
                common, rhs[step.id, kk, columns], acc=state, out_dtype=torch.float32
            )
            mixed = hl.dot(
                first.to(mixed_rhs.dtype),
                mixed_rhs[step.id, intermediate, columns],
                out_dtype=torch.float32,
            )
            state = state + mixed
            history[step.id, rows, columns] = state
        final[rows, columns] = state
    return history, final


def _args(device: str | torch.device, steps: int, mixed: bool) -> tuple:
    generator = torch.Generator(device=device).manual_seed(816)
    return (
        *(
            torch.randn(shape, dtype=dtype, device=device, generator=generator) * 0.1
            for shape, dtype in (
                ((max(steps, 1), 19, 16), torch.bfloat16),
                ((max(steps, 1), 16, 16), torch.bfloat16),
                ((max(steps, 1), 16, 13), torch.bfloat16),
                ((max(steps, 1), 16, 13), torch.float16 if mixed else torch.bfloat16),
                ((19, 13), torch.float32),
            )
        ),
        steps,
    )


def _config(
    warps: int, grouped: bool = False, schedule: str = "tcgen05_tmem"
) -> helion.Config:
    return helion.Config(
        num_warps=warps,
        cute_chained_mma_schedule=schedule,
        cute_chained_group_contractions=grouped,
        cute_chained_scratch_layout="xor",
    )


@pytest.mark.parametrize("warps", [4, 8, 16, 32])
@pytest.mark.parametrize("grouped", [False, True])
def test_cooperative_team_codegen_keeps_copy_participants_and_guards_short_tiles(
    warps: int, grouped: bool
) -> None:
    with _cpu_codegen():
        bound = _cooperative_loop._bind_isolated(_args("cpu", 3, True))
        spec = bound.config_spec
        config = spec.normalized_config(_config(warps, grouped))
        generation = spec.create_config_generation()
        restored = generation.unflatten(generation.flatten(config))
        assert restored["num_warps"] == warps
        source = bound.to_code(restored)
    threads = 32 * warps
    assert f"block=({threads}, 1, 1)" in source
    assert f"NamedBarrier(barrier_id=1, num_threads={threads})" in source
    assert f"chain_thread + chain_0_a_0_step * {threads}" in source
    assert "chain_0_seed" in source if grouped else "chain_1_seed" in source
    if threads > 256:
        assert "if chain_0_b_0_index < 256:" in source
    tree = ast.parse(source)
    copy_regions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "chain_thread < 128"
    ]
    if warps > 4:
        assert copy_regions
        for node in copy_regions:
            assert not any(
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "cute.arch.sync_threads"
                for call in ast.walk(node)
            )
    assert "cutlass.Float16" in source and "cutlass.BFloat16" in source


@pytest.mark.parametrize("warps", [16, 32])
def test_wide_warp_mma_loop_honors_cta_size(warps: int) -> None:
    with _cpu_codegen():
        bound = _cooperative_loop._bind_isolated(_args("cpu", 1, True))
        source = bound.to_code(_config(warps, schedule="coalesced"))
    assert f"block=({32 * warps}, 1, 1)" in source
    assert f"chain_0_load_step * {32 * warps * 8}" in source
    assert "if chain_thread < 32:" in source


@pytest.mark.parametrize("warps", [8, 16, 32])
@pytest.mark.parametrize("repair", [False, True])
def test_root_tcgen_retains_rejection_and_legacy_repair_policy(
    warps: int, repair: bool
) -> None:
    from .test_cute_chained_caches import _aux_cache_args
    from .test_cute_chained_caches import _auxiliary_chain

    with _cpu_codegen():
        bound = _auxiliary_chain._bind_isolated(_aux_cache_args("cpu", "plain"))
        config = helion.Config(
            block_sizes=[128, 64],
            num_warps=warps,
            cute_chained_mma_schedule="tcgen05_tmem",
        )
        if repair:
            bound.config_spec.normalize(config, _fix_invalid=True)
            assert config.num_warps == 4
            assert "block=(128, 1, 1)" in bound.to_code(config)
        else:
            with pytest.raises(exc.InvalidConfig, match="requires num_warps"):
                bound.config_spec.normalize(config)


@pytest.mark.parametrize("warps", [1, 2, 3, 64, True])
def test_loop_tcgen_rejects_unsupported_teams_before_repair(warps: int) -> None:
    with _cpu_codegen():
        bound = _cooperative_loop._bind_isolated(_args("cpu", 1, False))
        with pytest.raises(exc.InvalidConfig, match="requires num_warps"):
            bound.config_spec.normalize(_config(warps), _fix_invalid=True)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "warps,grouped,steps,mixed,schedule",
    [
        (8, False, 3, False, "tcgen05_tmem"),
        (16, False, 3, True, "tcgen05_tmem"),
        (32, False, 3, True, "tcgen05_tmem"),
        (8, True, 1, True, "tcgen05_tmem"),
        (16, True, 3, False, "tcgen05_tmem"),
        (32, True, 3, True, "tcgen05_tmem"),
        (32, True, 0, True, "tcgen05_tmem"),
        (16, False, 3, True, "coalesced"),
        (32, False, 3, True, "coalesced"),
    ],
)
def test_wide_team_gpu_preserves_ragged_grouped_seeded_mixed_and_zero_trip(
    warps: int, grouped: bool, steps: int, mixed: bool, schedule: str
) -> None:
    args = _args(DEVICE, steps, mixed)
    left, wide_rhs, rhs, mixed_rhs, initial, _ = args
    initial_copy = initial.clone()
    bound = _cooperative_loop._bind_isolated(args)
    compiled = bound.compile_config(_config(warps, grouped, schedule))
    history, final = compiled(*args)
    expected = initial.clone()
    for step in range(steps):
        first = left[step].float() @ wide_rhs[step].float()
        expected = left[step].float() @ rhs[step].float() + expected
        expected = (
            expected + first.to(mixed_rhs.dtype).float() @ mixed_rhs[step].float()
        )
        torch.testing.assert_close(history[step], expected, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(final, expected, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(initial, initial_copy, rtol=0, atol=0)
    _, repeated = compiled(*args)
    torch.testing.assert_close(repeated, final, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("warps", [16, 32])
def test_wide_team_gpu_collectives_and_carries_use_full_cta(warps: int) -> None:
    args = _inputs(DEVICE, 3, 1)
    bound = _prefix_coefficient_recurrence._bind_isolated(args)
    reference = bound.compile_config(_config(4))(*args)
    actual = bound.compile_config(_config(warps))(*args)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
