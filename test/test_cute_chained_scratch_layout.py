from __future__ import annotations

import ast
import importlib
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_loop_collectives import _inputs
from .test_cute_chained_loop_collectives import _postdot_chain
from .test_cute_chained_loop_collectives import _postdot_inputs
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
import helion
from helion import exc
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_scratch_layout import xor_swizzle
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@pytest.mark.parametrize("rows", [2, 3, 16, 32, 128])
@pytest.mark.parametrize("columns", [2, 16, 32, 64, 128, 256])
def test_xor_layout_is_bounded_bijective_row_bank_permutation(
    rows: int, columns: int
) -> None:
    parameters = xor_swizzle((rows, columns))
    assert parameters is not None
    bits, base, shift = parameters
    assert base == 0 and bits <= shift
    mask = (1 << bits) - 1
    addresses = []
    for row in range(rows):
        for column in range(columns):
            linear = row * columns + column
            physical = linear ^ ((linear >> shift) & mask)
            assert physical // columns == row
            assert 0 <= physical < rows * columns
            addresses.append(physical)
    assert len(set(addresses)) == rows * columns
    assert addresses != list(range(rows * columns))
    if columns >= 32:
        banks = [addresses[row * columns] % 32 for row in range(min(rows, 32))]
        assert len(set(banks)) == len(banks)
    # Arena offsets belong to the pointer, not the swizzle input; an allocation
    # remains bounded even when its offset isn't a multiple of the row stride.
    for offset in (0, 32, 96, 16384):
        assert min(offset + value for value in addresses) >= offset
        assert max(offset + value for value in addresses) < offset + rows * columns


@pytest.mark.parametrize("shape", [(3, 2), (3, 16), (8, 32), (16, 128)])
def test_actual_cute_mapping_matches_the_bounded_bank_permutation(
    shape: tuple[int, int],
) -> None:
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    parameters = xor_swizzle(shape)
    assert parameters is not None
    bits, base, shift = parameters
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = cute.make_composed_layout(
                cute.make_swizzle(bits, base, shift),
                0,
                cute.make_layout(shape, stride=(shape[1], 1)),
            )
            for row in range(shape[0]):
                for column in range(shape[1]):
                    linear = row * shape[1] + column
                    expected = linear ^ ((linear >> shift) & ((1 << bits) - 1))
                    assert int(layout((row, column))) == expected


@pytest.mark.parametrize(
    "shape", [(1, 128), (0, 32), (32, 1), (32, 24), (32, 96), (128,)]
)
def test_identity_or_unsafe_xor_layouts_are_not_eligible(
    shape: tuple[int, ...],
) -> None:
    assert xor_swizzle(shape) is None
    attempt = ScratchLayouts("xor")
    assert "make_swizzle" not in attempt.layout("scratch", shape)
    with pytest.raises(exc.BackendUnsupported, match="materialized eligible"):
        attempt.validate()


def test_layout_activation_is_local_and_preserves_existing_padding_by_default() -> None:
    default = ScratchLayouts()
    assert default.layout("c", (16, 64), row_stride=68) == (
        "cute.make_layout((16, 64), stride=(68, 1))"
    )
    first = ScratchLayouts("xor")
    assert first.layout("c", (16, 64), row_stride=68) == (
        "cute.make_composed_layout(cute.make_swizzle(5, 0, 6), 0, "
        "cute.make_layout((16, 64), stride=(64, 1)))"
    )
    first.validate()
    second = ScratchLayouts("xor")
    assert "make_swizzle" not in second.layout("half_carry", (16, 64), torch.float16)
    with pytest.raises(exc.BackendUnsupported, match="materialized eligible"):
        second.validate()


def test_unused_materialized_result_cannot_activate_xor() -> None:
    graph = Graph()
    left = _input(graph, "left", (16, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 16), torch.bfloat16)
    _dot(graph, left, right)
    row = _input(graph, "row", (1, 16), torch.bfloat16)
    live = _dot(graph, row, right)
    attempt = ScratchLayouts.for_plan("xor", _plan(graph, (live,)))
    assert attempt.read_buffers is not None
    assert "chain_0_c" not in attempt.read_buffers
    assert "chain_1_c" in attempt.read_buffers
    attempt.layout("chain_0_c", (16, 16))
    attempt.layout("chain_1_c", (1, 16))
    with pytest.raises(exc.BackendUnsupported, match="materialized eligible"):
        attempt.validate()


def _config(layout: str, schedule: str = "tcgen05_tmem") -> helion.Config:
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule=schedule,
        cute_chained_scratch_layout=layout,
    )


def _allocations(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.alloc_smem"
    ]


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_shared_scratch_layout_roundtrips_and_changes_real_buffers(
    loop: bool, schedule: str
) -> None:
    kernel = _prefix_coefficient_recurrence if loop else _postdot_chain
    args = (
        _inputs("cpu", 3, 0)
        if loop
        else _postdot_inputs("cpu", loop=False, rank=2, axis=0)
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_scratch_layout_search_enabled
        assert spec.supports_config_key("cute_chained_scratch_layout")
        assert "cute_chained_scratch_layout" in spec._flat_fields()
        config = spec.normalized_config(_config("xor", schedule))
        generation = spec.create_config_generation()
        restored = generation.unflatten(generation.flatten(config))
        assert restored["cute_chained_scratch_layout"] == "xor"
        row_major = bound.to_code(_config("row_major", schedule))
        xor = bound.to_code(restored)
        again = bound.to_code(_config("row_major", schedule))
    assert row_major == again
    assert _allocations(row_major) == _allocations(xor)
    assert "make_swizzle" not in row_major
    assert "make_swizzle" in xor
    assert "chain_collective_" in xor
    if loop:
        assert "chain_loop_carry_" in xor
        assert "stride=(32, 1)" in row_major
    else:
        assert "chain_0_c =" in xor


@pytest.mark.parametrize("value", [None, True, 1, "unknown"])
@pytest.mark.parametrize("repair", [False, True])
def test_scratch_layout_rejects_invalid_values_before_repair(
    value: object, repair: bool
) -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 1, 0))
        with pytest.raises(exc.InvalidConfig, match="must be row_major or xor"):
            bound.config_spec.normalize(
                helion.Config.from_dict({"cute_chained_scratch_layout": value}),
                _fix_invalid=repair,
            )


def test_xor_request_cannot_silently_fall_back_to_another_lowering() -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 1, 0))
        with (
            patch(
                "helion._compiler.cute.chained_matmul.plan_chained_matmul",
                return_value=None,
            ),
            pytest.raises(
                exc.BackendUnsupported, match="XOR scratch layout requires a supported"
            ),
        ):
            bound.to_code(_config("xor", "coalesced"))


def test_xor_request_cannot_select_a_competing_affine_lowering() -> None:
    with _cpu_codegen():
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 1, 0))
        config = helion.Config.from_dict(
            _config("xor").config | {"cute_affine_scan_schedule": "direct_mma"}
        )
        with pytest.raises(exc.InvalidConfig, match="common contraction lowering"):
            bound.to_code(config)


def test_xor_requires_actual_materialization_after_register_bridges() -> None:
    from .test_cute_chained_caches import _aux_cache_args
    from .test_cute_chained_caches import _auxiliary_chain

    with _cpu_codegen():
        bound = _auxiliary_chain._bind_isolated(_aux_cache_args("cpu", "plain"))
        assert bound.config_spec.cute_chained_scratch_layout_search_enabled
        with pytest.raises(exc.BackendUnsupported, match="materialized eligible"):
            bound.to_code(
                helion.Config(
                    block_sizes=[128, 64],
                    num_warps=4,
                    cute_chained_mma_schedule="tcgen05_tmem",
                    cute_chained_scratch_layout="xor",
                )
            )


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_root_default_source_preserves_legacy_row_major_policy(schedule: str) -> None:
    from .test_cute_chained_caches import _aux_cache_args
    from .test_cute_chained_caches import _auxiliary_chain

    def old_layout(
        self: ScratchLayouts,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        *,
        row_stride: int | None = None,
    ) -> str:
        assert dtype == torch.float32 and len(shape) == 2
        stride = shape[1] if row_stride is None else row_stride
        return f"cute.make_layout({shape!r}, stride=({stride}, 1))"

    config = helion.Config(
        block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule=schedule
    )
    with _cpu_codegen():
        bound = _auxiliary_chain._bind_isolated(_aux_cache_args("cpu", "plain"))
        source = bound.to_code(config)
        with patch.object(ScratchLayouts, "layout", old_layout):
            legacy = bound.to_code(config)
    assert source == legacy


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("axis", [0, 1])
def test_xor_scratch_gpu_preserves_shared_loop_and_root_numerics(
    loop: bool, schedule: str, axis: int
) -> None:
    if loop:
        args = _inputs(DEVICE, 3, axis)
        a, b, decay, delta, initial = args[:5]
        state = initial.clone()
        expected_history = []
        for step in range(3):
            row_prefix = decay[:, step].cumsum(1)
            matrix_prefix = delta[:, step].cumsum(axis + 1)
            energy = matrix_prefix.square().sum(2)
            left = (a[:, step].float() * matrix_prefix.exp()).to(a.dtype)
            product = left.float() @ b[:, step].float()
            state = state * row_prefix.exp()[:, :, None] + product / (
                1.0 + energy[:, :, None]
            )
            expected_history.append(state.clone())
        expected = torch.stack(expected_history, dim=1), state
        kernel = _prefix_coefficient_recurrence
    else:
        args = _postdot_inputs(DEVICE, loop=False, rank=2, axis=axis)
        a, b, c = args[:3]
        first = (a.float() @ b.float()).cumsum(axis)
        second = first.to(c.dtype).float() @ c.float()
        expected = second.cumsum(axis)
        kernel = _postdot_chain
    bound = kernel._bind_isolated(args)
    config = _config("xor", schedule)
    assert "make_swizzle" in bound.to_code(config)
    actual = bound.compile_config(config)(*args)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
