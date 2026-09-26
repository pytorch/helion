from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test.test_cute_epilogue_fanout import _opaque_metadata
from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401

import helion
from helion._compiler.cute.row_resident import ROW_RESIDENT_KEY
from helion._compiler.cute.row_resident import _matrix
from helion._compiler.cute.row_resident import prove_row_resident
from helion._compiler.cute.row_resident_codegen import _scalar_epilogue
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import creation_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from helion._compiler.cute.row_resident import RowResidentPlan
    from helion.runtime.kernel import BoundKernel


def renamed_chain(
    left: torch.Tensor,
    hidden: torch.Tensor,
    expansion: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, inner = left.size()
    middle_width = hidden.size(1)
    width = expansion.size(1)
    result = torch.empty((height, width), dtype=left.dtype, device=left.device)
    middle = torch.empty((height, middle_width), dtype=left.dtype, device=left.device)
    gate_output = torch.empty((height, width), dtype=left.dtype, device=left.device)
    for row in hl.tile(height):
        for column in hl.tile(middle_width):
            first = left[row, :] @ hidden[:, column]
            middle[row, column] = torch.relu(first)
        for column in hl.tile(width):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            gate_output[row, column] = torch.sigmoid(value)
            result[row, column] = residual[row, column] * gate_output[row, column]
    return middle, gate_output, result


def effectful_host_chain(
    left: torch.Tensor, hidden: torch.Tensor, expansion: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = left.size()
    middle_width = hidden.size(1)
    result = torch.empty((height, width), dtype=left.dtype, device=left.device)
    middle = torch.empty((height, middle_width), dtype=left.dtype, device=left.device)
    gate_output = torch.empty_like(result)
    token = _opaque_metadata(gate_output, result)
    for row in hl.tile(height):
        for column in hl.tile(middle_width):
            middle[row, column] = torch.relu(left[row, :] @ hidden[:, column])
        for column in hl.tile(width + token):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            gate_output[row, column] = torch.sigmoid(value)
            result[row, column] = left[row, column] * gate_output[row, column]
    return middle, gate_output, result


def bind(
    *,
    rows: int = 17,
    reduction: int = 64,
    middle: int = 64,
    columns: int = 128,
    dtype: torch.dtype = torch.float16,
    original: bool = False,
    settings: dict[str, Any] | None = None,
) -> BoundKernel[Any]:
    if original:
        function = squeeze_and_excitation_net_fwd.fn
        shapes = ((rows, columns), (columns, middle), (middle, columns))
    else:
        function = renamed_chain
        shapes = (
            (rows, reduction),
            (reduction, middle),
            (middle, columns),
            (rows, columns),
        )
    values = tuple(torch.empty(shape, dtype=dtype) for shape in shapes)
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        **(
            {"cute_region_fission": True, "cute_full_slice_matmul_tiling": True}
            | (settings or {})
        ),
    )
    return kernel._bind_isolated(values)


def plan(bound: BoundKernel[Any]) -> RowResidentPlan | None:
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        return prove_row_resident(bound.env, host)


def row_config(bound: BoundKernel[Any], mode: object = "warp_rows2") -> helion.Config:
    ordinary = bound.config_spec.default_config()
    requested = helion.Config.from_dict(ordinary.config | {ROW_RESIDENT_KEY: mode})
    return bound.config_spec.normalized_config(requested)


@pytest.mark.parametrize("rows", (1, 2, 3, 17, 255, 256, 257))
@skipUnlessBackends(["cute"])
def test_two_row_tails_are_a_general_static_domain(rows: int) -> None:
    bound = bind(rows=rows)
    found = plan(bound)
    assert found is not None and found.rows == rows
    assert found.columns == (64, 128) and found.reductions == (64, 64)
    code = bound.to_code(row_config(bound))
    assert "_helion_row_resident" in code
    assert f"({(rows + 1) // 2}, 1, 1)" in code
    assert len(found.stages[0].stores) == 1 and len(found.stages[1].stores) == 2


@pytest.mark.parametrize(
    ("reduction", "middle", "columns", "admitted"),
    (
        (32, 64, 256, True),
        (256, 64, 64, True),
        (128, 128, 128, True),
        (32, 256, 64, True),
        (64, 256, 64, True),
        (256, 128, 64, False),
        (64, 128, 256, False),
        (64, 256, 128, False),
        (48, 64, 64, False),
        (64, 32, 64, False),
        (64, 64, 80, False),
    ),
)
@skipUnlessBackends(["cute"])
def test_packet_and_shared_memory_domains(
    reduction: int, middle: int, columns: int, admitted: bool
) -> None:
    bound = bind(reduction=reduction, middle=middle, columns=columns)
    assert (plan(bound) is not None) is admitted
    assert bound.config_spec.cute_materialized_schedule_available is admitted
    if admitted:
        row_config(bound)
    else:
        with pytest.raises(InvalidConfig, match=ROW_RESIDENT_KEY):
            row_config(bound)
        assert not any(
            g.key == ROW_RESIDENT_KEY
            for g in bound.config_spec.compiler_coverage_groups
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@skipUnlessBackends(["cute"])
def test_other_arithmetic_domains_keep_old_path(dtype: torch.dtype) -> None:
    bound = bind(dtype=dtype)
    assert plan(bound) is None
    assert not bound.config_spec.cute_materialized_schedule_available


@skipUnlessBackends(["cute"])
def test_original_and_renamed_materialized_graphs() -> None:
    for bound in (bind(), bind(original=True, rows=256, middle=64, columns=256)):
        found = plan(bound)
        assert found is not None
        first, second = found.stages
        assert second.lhs is first.stores[0].tensor
        assert second.lhs_name == first.stores[0].name
        assert second.stores[1].chain.auxiliary_tensor_loads
        ordinary = bound.config_spec.default_config()
        off = helion.Config.from_dict(ordinary.config | {ROW_RESIDENT_KEY: "off"})
        assert bound.to_code(ordinary) == bound.to_code(off)
        assert ROW_RESIDENT_KEY not in bound.config_spec.normalized_config(off).config
        assert bound.to_code(row_config(bound)) != bound.to_code(ordinary)


@skipUnlessBackends(["cute"])
def test_actual_effectful_host_metadata_keeps_the_original_path() -> None:
    args = tuple(torch.empty((64, 64), dtype=torch.float16) for _ in range(3))
    kernel = helion.kernel(
        effectful_host_chain,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    assert plan(bound) is None
    assert not bound.config_spec.cute_materialized_schedule_available
    code = bound.to_code(bound.config_spec.default_config())
    assert "_opaque_metadata(gate_output, result)" in code
    with pytest.raises(InvalidConfig, match=ROW_RESIDENT_KEY):
        row_config(bound)


@pytest.mark.parametrize("value", (None, False, True, 0, 1, "auto", "rows2", [], {}))
@skipUnlessBackends(["cute"])
def test_strict_mode_and_roundtrip(value: object) -> None:
    bound = bind()
    with pytest.raises(InvalidConfig, match=ROW_RESIDENT_KEY):
        row_config(bound, value)
    good = row_config(bound)
    restored = helion.Config.from_dict(deepcopy(good.config))
    assert bound.config_spec.normalized_config(restored).config == good.config
    generation = bound.config_spec.create_config_generation()
    _, effective = generation.strict_config_pair(good)
    assert effective.config[ROW_RESIDENT_KEY] == "warp_rows2"


@contextmanager
def mutate(node: torch.fx.Node, **values: Any) -> Iterator[None]:
    before = {key: getattr(node, key) for key in values}
    for key, value in values.items():
        setattr(node, key, value)
    try:
        yield
    finally:
        for key, value in before.items():
            setattr(node, key, value)


@pytest.mark.parametrize("value", (1, -1.0, -0.0, float("nan")))
@skipUnlessBackends(["cute"])
def test_accumulator_initialization_is_exact_positive_zero(value: float) -> None:
    bound = bind()
    assert plan(bound)
    assert bound.host_function is not None
    seed = next(
        n
        for info in bound.host_function.device_ir.graphs
        for n in info.graph.nodes
        if n.target is creation_ops.full
    )
    args = list(seed.args)
    args[1] = value
    with mutate(seed, args=tuple(args)):
        assert plan(bound) is None
        with pytest.raises(InvalidConfig, match="proved complete"):
            bound.to_code(row_config(bound))
    assert plan(bound)


@pytest.mark.parametrize(
    "fault",
    (
        "beta",
        "wrong_seed",
        "loop_start",
        "loop_stop",
        "output_mask",
        "input_mask",
        "missing_alignment_snapshot",
        "unregistered_alignment",
    ),
)
@skipUnlessBackends(["cute"])
def test_complete_typed_proof_rechecks_late_changes(fault: str) -> None:
    bound = bind()
    assert plan(bound)
    assert bound.host_function is not None
    nodes = [
        n for info in bound.host_function.device_ir.graphs for n in info.graph.nodes
    ]
    if fault == "missing_alignment_snapshot":
        with patch.object(bound.env, "bound_runtime_input_specialization_results", {}):
            assert plan(bound) is None
        return
    if fault == "unregistered_alignment":
        with patch.object(bound.env, "runtime_input_specializations", {}):
            assert plan(bound) is None
            with bound.env, bound.host_function:
                assert (
                    prove_row_resident(bound.env, bound.host_function, binding=True)
                    is None
                )
        return
    if fault in ("beta", "wrong_seed"):
        node = next(n for n in nodes if n.target is torch.ops.aten.addmm.default)
        changed = (
            {"kwargs": {"beta": 2}}
            if fault == "beta"
            else {"args": (node.args[1], *node.args[1:])}
        )
    elif fault.startswith("loop"):
        node = next(n for n in nodes if n.target is _tracing_ops._for_loop)
        args = list(node.args)
        args[1 if fault == "loop_start" else 2] = [1]
        changed = {"args": tuple(args)}
    else:
        target = memory_ops.store if fault == "output_mask" else memory_ops.load
        node = next(n for n in nodes if n.target is target)
        args = list(node.args)
        args[3 if target is memory_ops.store else 2] = True
        changed = {"args": tuple(args)}
    with mutate(node, **changed):
        assert plan(bound) is None
    assert plan(bound)


def test_matrix_stride_and_address_domain() -> None:
    assert _matrix(torch.empty((3, 64), dtype=torch.float16))
    assert not _matrix(torch.empty((3, 64), dtype=torch.float64))
    assert not _matrix(torch.empty((64, 3), dtype=torch.float16).T)
    assert not _matrix(torch.empty((3, 128), dtype=torch.float16)[:, ::2])
    assert not _matrix(torch.empty((0, 64), dtype=torch.float16))
    with torch.device("meta"):
        assert not _matrix(torch.empty((2**25, 64), dtype=torch.float16))


def test_scalar_conditionals_retain_both_ieee_relu_branches() -> None:
    source = (
        "out = cute.where(value != value, value, cute.where(value > 0.0, value, 0.0))"
    )
    transformed = _scalar_epilogue(source)
    assert (
        len([n for n in ast.walk(ast.parse(transformed)) if isinstance(n, ast.IfExp)])
        == 2
    )
    samples = (float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1.0, 1.0, 1e-40)
    for value in samples:
        namespace = {"value": value}
        exec(transformed, namespace)
        expected = value if value != value else max(0.0, value)
        observed = namespace["out"]
        assert observed == expected or observed != observed and expected != expected
