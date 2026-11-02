from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import (
    CutePersistentSubwarpRowsHeuristic,
)
from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds
from helion.autotuner.config_fragment import BooleanFragment
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, disable_autotuner_heuristics=True)
def _center_rows(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for rows in hl.tile(x.shape[0]):
        columns = hl.arange(x.shape[1])
        values = x[rows, columns].float()
        maximum = values.abs().amax(-1)
        output[rows, columns] = (values - maximum[:, None]).to(x.dtype)
    return output


def _config(
    bound: BoundKernel, width: int, *, enabled: bool, row_threads: int = 128
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    reduction_threads = max(2, width // 16)
    return helion.Config(
        block_sizes=[row_threads],
        num_threads=[
            reduction_threads if block == reduction else row_threads
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            width // reduction_threads if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_layouts=[
            "blocked" for block in spec.cute_lane_layouts.valid_block_ids()
        ],
        cute_reduction_reloads=["register"],
        cute_proven_bounds=enabled,
    )


def _comparisons(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Compare)
    ]


@pytest.mark.parametrize(
    ("width", "dtype"), [(32, torch.bfloat16), (64, torch.bfloat16)]
)
def test_seeded_persistent_reduction_exposes_bounds_without_autotuner_seeds(
    width: int, dtype: torch.dtype
) -> None:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_center_rows, (torch.empty(4096, width, dtype=dtype),))
        spec = bound.config_spec
        assert spec.cute_proven_bounds_enabled
        assert spec.autotuner_heuristics == []
        assert spec.compiler_seed_configs == []
        assert isinstance(spec._flat_fields()["cute_proven_bounds"], BooleanFragment)
        config = _config(bound, width, enabled=True)
        enabled = bound.to_code(config)
        control = bound.to_code(_config(bound, width, enabled=False))
        implicit_default = helion.Config.from_dict(
            {
                key: value
                for key, value in config.config.items()
                if key != "cute_proven_bounds"
            }
        )
        assert spec.normalized_config(implicit_default).cute_proven_bounds is False
        generation = spec.create_config_generation()
        flat = generation.flatten(config)
        restored = generation.unflatten(flat)
        assert restored.cute_proven_bounds is True
        assert generation.flatten(restored) == flat
        assert bound.host_function is not None
        assert CutePersistentSubwarpRowsHeuristic.is_eligible(
            bound.env, bound.host_function.device_ir
        )

    assert "_cute_proven_loop_bounds_" in enabled
    assert "_cute_proven_loop_bounds_" not in control
    assert len(_comparisons(enabled)) < len(_comparisons(control))
    assert not any(
        "reduction_lane_base" in expression for expression in _comparisons(enabled)
    )
    assert f"block=({max(2, width // 16)}, 128, 1)" in enabled
    assert f"threads_in_group={max(2, width // 16)}" in enabled


@pytest.mark.parametrize(
    ("width", "dtype"), [(32, torch.bfloat16), (64, torch.bfloat16)]
)
def test_bounds_keep_partial_outer_row_guards(width: int, dtype: torch.dtype) -> None:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_center_rows, (torch.empty(137, width, dtype=dtype),))
        enabled = bound.to_code(_config(bound, width, enabled=True))
        control = bound.to_code(_config(bound, width, enabled=False))
    enabled_comparisons = _comparisons(enabled)
    assert any("< 137" in expression for expression in enabled_comparisons)
    assert not any(
        "reduction_lane_base" in expression for expression in enabled_comparisons
    )
    assert len(enabled_comparisons) < len(_comparisons(control))


@pytest.mark.parametrize(
    ("width", "dtype", "admitted"),
    [
        (32, torch.bfloat16, True),
        (16, torch.int32, False),
        (512, torch.bfloat16, False),
    ],
)
def test_bounds_admission_follows_the_short_row_seed_plan(
    width: int, dtype: torch.dtype, admitted: bool
) -> None:
    # Reductions without a seed plan (too short for a vector group, or wider
    # than one warp of packets) keep the knob out of their search space.
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_center_rows, (torch.empty(4096, width, dtype=dtype),))
        assert bound.host_function is not None
        eligible = CutePersistentSubwarpRowsHeuristic.is_eligible(
            bound.env, bound.host_function.device_ir
        )
        enabled = bound.config_spec.cute_proven_bounds_enabled
    assert eligible is admitted
    assert enabled is admitted


@pytest.mark.parametrize(("width", "vector"), [(16, 8), (32, 16)])
@pytest.mark.parametrize("extent_delta", [0, -1])
def test_short_row_interval_proof_preserves_all_thread_addresses(
    width: int, vector: int, extent_delta: int
) -> None:
    extent = width + extent_delta
    source = f"""
thread = cutlass.Int32(cute.arch.thread_idx()[0])
base = thread * {vector}
for lane in cutlass.range_constexpr({vector}):
    column = base + lane
    if column < {extent}:
        record(column)
if base < {extent} and base + {vector - 1} < {extent}:
    packet(base)
"""
    transformed = ast.unparse(
        ast.Module(
            body=simplify_proven_loop_bounds(
                ast.parse(source).body,
                enabled=True,
                thread_block_dims=(2, 128, 1),
                constexpr_values={},
            ),
            type_ignores=[],
        )
    )
    if extent_delta == 0:
        assert "if " not in transformed
    else:
        assert "if " in transformed
    for thread in range(2):
        for row_thread in range(128):
            traces = []
            for program in (source, transformed):
                trace: list[tuple[str, int]] = []
                namespace = {
                    "cutlass": SimpleNamespace(Int32=int, range_constexpr=range),
                    "cute": SimpleNamespace(
                        arch=SimpleNamespace(
                            thread_idx=lambda thread=thread, row_thread=row_thread: (
                                thread,
                                row_thread,
                                0,
                            )
                        )
                    ),
                    "record": lambda column, trace=trace: trace.append(
                        ("scalar", column)
                    ),
                    "packet": lambda base, trace=trace: trace.append(("packet", base)),
                }
                exec(compile(program, "<persistent-bound-proof>", "exec"), namespace)
                traces.append(trace)
            assert traces[0] == traces[1]
