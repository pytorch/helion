from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.active_phase_joins import participants
from helion._compiler.cute.backend import CuteBackend
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_staged_indexed_rows import selected_projection_pair

pytestmark = pytest.mark.cpu_only


def _phase(hi: int, lo: int) -> ast.For:
    loops = ast.parse(
        "for _ in range(cutlass.Int32(0), "
        f"cutlass.Int32(({hi} - {lo}) // _NUM_SM + "
        f"cutlass.Int32(cutlass.Int32(cute.arch.block_idx()[0]) < ({hi} - {lo}) % _NUM_SM))):\n"
        "    store_result()"
    ).body
    (loop,) = loops
    assert isinstance(loop, ast.For)
    return loop


def test_union_and_exact_interleaved_coverage() -> None:
    # Rebased starts are common after different phase tile sizes are installed.
    region = [_phase(81, 17), _phase(8, 4)]
    assert participants(region) == 64


def test_unknown_effects_and_changed_iteration_keep_full_grid() -> None:
    body = [_phase(8, 4)]
    assert participants([*body, *ast.parse("tmp = arg.iterator.load()").body]) is None
    assert participants([*body, *ast.parse("opaque.allocate_tensor()").body]) is None
    loop = _phase(8, 4)
    loop.orelse = ast.parse("publish()").body
    assert participants([loop]) is None
    loop = _phase(8, 4)
    loop.iter = ast.parse("range(some_other_stride)", mode="eval").body
    assert participants([loop]) is None


def test_descriptor_setup_does_not_enlarge_the_prefix() -> None:
    setup = ast.parse(
        "alloc = cutlass.utils.SmemAllocator()\n"
        "private = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((4,)))"
    ).body
    assert participants([*setup, _phase(9, 4)]) == 5
    assert participants(setup) is None
    assert (
        participants(
            [setup[0], *ast.parse("alloc = outside").body, setup[1], _phase(9, 4)]
        )
        is None
    )


def test_active_join_import_cannot_be_shadowed_and_requires_enough_scratch() -> None:
    # The shared projection fixture deliberately names its BF16 scale tensor
    # _cute_join_active. The generated CUDA function must rename the argument.
    args = (
        torch.empty((2, 256), dtype=torch.bfloat16),
        torch.empty((5, 64, 256), dtype=torch.bfloat16),
        torch.empty((5, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 78), dtype=torch.int32),
        torch.empty(2, 78),
        torch.empty(256, dtype=torch.bfloat16),
        3.0,
    )
    cfg = helion.Config(
        block_sizes=[16, 256, 64, 64],
        num_threads=[4, 32, 4, 2, 1],
        reduction_loops=[64],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_defer_tile_reduce=True,
        cute_staged_indexed_rows=True,
        cute_staged_projection_grid=2,
        cute_phase_participation=True,
    )
    with cpu_render_target(fake_device_sm_count=148, smem_budget_bytes=232448):
        bound = cpu_bind(selected_projection_pair, args)
        source = bound.to_code(cfg)
        tree = ast.parse(source)
        (device,) = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
        ]
        arguments = {a.arg for a in device.args.args}
        assert "_cute_join_active" not in arguments
        assert any(a.startswith("_cute_join_active") for a in arguments)
        assert source.count("import join_active as _cute_join_active") == 1
        assert (
            "_cute_join_active(x_grid_sem.iterator + 0, min(_NUM_SM, 312), min(_NUM_SM, 156))"
            in source
        )
        assert "torch.zeros((32,)" in source
        with (
            patch.object(CuteBackend, "grid_barrier_scratch_words", return_value=31),
            pytest.raises(exc.InvalidConfig, match="private counter for each edge"),
        ):
            bound.to_code(cfg)
