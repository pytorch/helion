from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.memory_ops import (
    _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
)
from helion._compiler.cute.memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language.scan_ops import _associative_scan
from helion.runtime.kernel import BoundKernel

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])
CPU_DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        yield


def _pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + y[tile]
    return out


@pytest.mark.parametrize("static_shapes", [False, True])
def test_unrelated_grid_has_no_alias_specialization(static_shapes: bool) -> None:
    kernel = helion.kernel(_pointwise, backend="cute", static_shapes=static_shapes)
    x = torch.empty(64, device=CPU_DEVICE)
    y = torch.empty_like(x)
    bound = kernel.bind((x, y))
    assert not bound.env.runtime_input_specializations
    assert not kernel._has_specialization_extras
    # This pointwise kernel does not reorder stores to an existing input, so
    # input aliasing should not create a redundant compiled variant.
    assert kernel.bind((x, x.view_as(x))) is bound


def test_explicit_specialization_schema_is_not_extended_by_unused_alias_facts() -> None:
    @helion.kernel(backend="cute", static_shapes=False)
    def specialized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        size = hl.specialize(y.size(0))
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile] + size
        return out

    args = (torch.empty(32, device=CPU_DEVICE), torch.empty(64, device=CPU_DEVICE))
    bound = specialized.bind(args)
    extractors = specialized._specialize_extra[bound._base_spec_key]
    assert tuple(fn(args) for fn in extractors) == (64,)


def test_container_grid_dispatch_does_not_rebind_without_alias_consumer() -> None:
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[64])
    )
    def add_list(xs: list[torch.Tensor]) -> torch.Tensor:
        out = torch.empty_like(xs[0])
        for tile in hl.tile(xs[0].numel()):
            out[tile] = xs[0][tile] + xs[1][tile]
        return out

    x, y = torch.randn(64, device=CPU_DEVICE), torch.randn(64, device=CPU_DEVICE)
    with (
        patch.object(
            BoundKernel, "compile_config", return_value=lambda xs: xs[0] + xs[1]
        ),
        patch.object(add_list, "_bind", wraps=add_list._bind) as bind,
    ):
        out = add_list([x, y])
    assert bind.call_count == 1
    torch.testing.assert_close(out, x + y)


def _scan_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel], dim=0)
    return out


@pytest.mark.parametrize("static_shapes", [False, True])
def test_scan_preserves_alias_specialization_and_serial_codegen(
    static_shapes: bool,
) -> None:
    kernel = helion.kernel(_scan_into, backend="cute", static_shapes=static_shapes)
    x = torch.empty((128, 3), device=CPU_DEVICE)
    out = torch.empty_like(x)
    bound = kernel.bind((x, out))
    assert _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY in (
        bound.env.runtime_input_specializations
    )
    config = bound.config_spec.default_config()
    assert "for scan_i in range(" in bound.to_code(config)
    for alias in (x, x.view_as(x)):
        alias_bound = kernel.bind((x, alias))
        assert alias_bound is not bound
        assert "for scan_i in range(" in alias_bound.to_code(config)


def test_nested_scan_registers_alias_facts_before_codegen() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def nested(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
            if channel.begin == 0:
                out[row, channel] = hl.cumsum(x[row, channel], dim=0)
            else:
                out[row, channel] = x[row, channel]
        return out

    x = torch.empty((128, 3), device=CPU_DEVICE)
    bound = nested.bind((x, torch.empty_like(x)))
    assert _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY in (
        bound.env.runtime_input_specializations
    )
    assert bound.host_function is not None
    ir = bound.host_function.device_ir
    scan_graphs = [
        info.graph_id
        for info in ir.graphs
        if any(node.target is _associative_scan for node in info.graph.nodes)
    ]
    assert scan_graphs and not set(scan_graphs).intersection(ir.root_ids)
    # Registration is conservative; divergent scans retain scalar lowering.
    assert "for scan_i in range(" in bound.to_code(bound.config_spec.default_config())


def test_reduction_specializations_are_unchanged() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def reduce_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        for row in hl.tile(x.size(0)):
            out[row] = x[row, :].sum(-1)
        return out

    bound = reduce_into.bind(
        (
            torch.empty((3, 128), device=CPU_DEVICE),
            torch.empty(3, device=CPU_DEVICE),
        )
    )
    assert {
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
    }.issubset(bound.env.runtime_input_specializations)
