"""Code shape of the ``cute_pdl`` knob (programmatic dependent launch).

Knob on: the module tags the kernel with ``_helion_cute_use_pdl = True`` for
the launcher and the kernel executes ``cute.arch.griddepcontrol_wait()``
ahead of its first memory access; nothing else changes.  Passes that run
after the wait is spliced in may hoist register allocations above it, so the
invariant is "no load, store, atomic or bulk copy ahead of the wait, and no
release of the dependents", not "the wait is the first statement".  Knob off
(the default): neither line appears.  The kernels that already take part in
dependent launch (the fission producer that releases its dependents at entry,
the tcgen05 consumer with its role-local wait) are covered in
``test_cute_materialized_pdl.py``; GPU semantics live in
``test_cute_pdl_cuda.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from examples.aot_example import rms_norm_batched
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test._cute_vloop_sink_kernels import col_reduce_max_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_static
from test._cute_vloop_sink_kernels import col_reduce_sum_lower_triangle_static
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test._cute_vloop_sink_kernels import col_weighted_mean_static
from test.test_cute_dynamic_vector_reduction import _dynamic_kernel
from test.test_cute_dynamic_vector_reduction import _rms_norm_specialized_n
from test.test_cute_dynamic_vector_reduction import _rolled_config

import helion
from helion import exc
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scale_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * w[tile_n][None, :]
    return out


_COLUMN_SUM = {
    "block_sizes": [1024, 16],
    "num_threads": [128, 4],
    "cute_vector_widths": [1, 4],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}

_WAIT = "cute.arch.griddepcontrol_wait()"
_ATTRIBUTE = "._helion_cute_use_pdl = True"
# Attribute names of calls that touch memory or take part in dependent launch.
_MEMORY_TOKENS = (
    "load",
    "store",
    "copy",
    "atomic",
    "cp_async",
    "tma",
    "mbarrier",
    "fence",
    "griddepcontrol",
)


def _kernel_def(code: str) -> ast.FunctionDef:
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    ]
    return kernel


def _on_and_off(bound: Any, config: dict[str, object]) -> tuple[str, str]:
    spec = bound.config_spec
    base = {key: value for key, value in config.items() if key != "cute_pdl"}
    on = bound.to_code(
        spec.normalized_config(helion.Config.from_dict(base | {"cute_pdl": True}))
    )
    off = bound.to_code(spec.normalized_config(helion.Config.from_dict(base)))
    return on, off


def _codes(kernel: helion.Kernel, args: tuple, **config: object) -> tuple[str, str]:
    return _on_and_off(_cpu_bind(kernel, args), config)


def _statements_ahead_of_the_wait(kernel: ast.FunctionDef) -> list[str]:
    """The wait is a top-level statement of the kernel body and nothing ahead
    of it touches memory, at any nesting depth: no tensor indexing and no
    call that loads, stores, copies, syncs a barrier or speaks
    ``griddepcontrol``.  Returns what does precede it."""
    waits = [
        index
        for index, statement in enumerate(kernel.body)
        if ast.unparse(statement) == _WAIT
    ]
    assert len(waits) == 1, ast.unparse(kernel)[:600]
    ahead = kernel.body[: waits[0]]
    for statement in ahead:
        for node in ast.walk(statement):
            assert not isinstance(node, ast.Subscript), ast.unparse(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not any(token in node.func.attr for token in _MEMORY_TOKENS), (
                    ast.unparse(statement)
                )
    return [ast.unparse(statement) for statement in ahead]


def _assert_only_the_dependent_launch_differs(on: str, off: str) -> list[str]:
    kernel = _kernel_def(on)
    ahead = _statements_ahead_of_the_wait(kernel)
    assert "griddepcontrol_launch_dependents" not in on
    assert on.count(_WAIT) == 1
    assert on.count(_ATTRIBUTE) == 1
    assert f"{kernel.name}{_ATTRIBUTE}" in on
    assert _WAIT not in off and _ATTRIBUTE not in off
    stripped = [
        line for line in on.splitlines() if _WAIT not in line and _ATTRIBUTE not in line
    ]
    assert stripped == off.splitlines()
    return ahead


def test_the_column_sum_waits_before_its_first_global_access(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    on, off = _codes(col_reduce_sum_static, (x,), **_COLUMN_SUM)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    assert "_vsink_vec" in on
    # The attribute is set on the kernel object the host code launches.
    kernel = _kernel_def(on)
    assert f"_launcher({kernel.name}," in on


def test_dynamic_shapes_and_pointwise_kernels_get_the_same_two_lines(
    cpu_only: None,
) -> None:
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    on, off = _codes(col_reduce_sum_dynamic, (x,), **_COLUMN_SUM)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    w = torch.empty(1008, dtype=torch.bfloat16)
    on, off = _codes(scale_rows, (x, w), block_sizes=[16, 64], num_threads=[4, 64])
    assert _assert_only_the_dependent_launch_differs(on, off) == []


@pytest.mark.parametrize(
    "kernel,shapes",
    [
        (col_reduce_max_dynamic, [(1000, 1008)]),
        (col_reduce_sum_from8_static, [(1024, 1024)]),
        (col_reduce_sum_lower_triangle_static, [(1024, 1024)]),
        (col_reduce_sum_pair_dynamic, [(1000, 1008), (1000, 1008)]),
        (col_weighted_mean_static, [(1024, 1024), (1024,)]),
    ],
)
def test_column_reductions_take_the_knob_under_their_default_config(
    cpu_only: None, kernel: helion.Kernel, shapes: list[tuple[int, ...]]
) -> None:
    args = tuple(torch.empty(*shape, dtype=torch.bfloat16) for shape in shapes)
    bound = _cpu_bind(kernel, args)
    on, off = _on_and_off(bound, dict(bound.config_spec.default_config().config))
    assert _assert_only_the_dependent_launch_differs(on, off) == []


@pytest.mark.parametrize("static_shapes", [False, True])
def test_hoisted_register_allocations_may_precede_the_wait(
    cpu_only: None, static_shapes: bool
) -> None:
    # The two-pass load fuser runs after the wait is spliced in and hoists
    # its register cache to the top of the kernel.  That is no memory
    # access, so it may sit ahead of the wait; the first load still follows.
    fn = rms_norm_batched.fn if static_shapes else _rms_norm_specialized_n
    x = torch.empty((13, 4096), dtype=torch.bfloat16)
    args = (x, 1e-5)
    bound = _cpu_bind(_dynamic_kernel(fn, static_shapes=static_shapes), args)
    config = _rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024)
    on, off = _on_and_off(bound, dict(config.config))
    assert _assert_only_the_dependent_launch_differs(on, off) == [
        "_fuse_cache_0 = cute.make_rmem_tensor(64, cutlass.Uint16)"
    ]


def test_the_knob_is_a_searchable_boolean(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    assert spec.default_config().config.get("cute_pdl", False) is False
    with pytest.raises(exc.InvalidConfig, match="cute_pdl must be a boolean"):
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl="yes"))
    assert (
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=False)).config[
            "cute_pdl"
        ]
        is False
    )
    # The search space carries the knob as a boolean fragment and searches
    # both values for this kernel (no GEMM, sm_100 target).
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(spec)
    assert "cute_pdl" in {key for key, _count, _seq in spec.flat_key_layout()}
    assert spec._flat_fields()["cute_pdl"].search_choices == (False, True)
    values = {
        generation.unflatten(generation.flatten(config)).config.get("cute_pdl")
        for config in (
            helion.Config(**_COLUMN_SUM, cute_pdl=True),
            helion.Config(**_COLUMN_SUM, cute_pdl=False),
        )
    }
    assert values == {True, False}


def test_pre_hopper_targets_neither_search_nor_apply_the_knob(
    cpu_only: None,
) -> None:
    # ``griddepcontrol`` needs sm_90: below it the fragment offers only the
    # off value and an explicit ``cute_pdl=True`` generates the knob-off code.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    with patch.object(spec, "target_device_capability", (8, 0)):
        assert spec._flat_fields()["cute_pdl"].search_choices == (False,)
        on, off = _on_and_off(bound, _COLUMN_SUM)
    assert _WAIT not in on and _ATTRIBUTE not in on
    assert on == off
