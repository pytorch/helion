"""CPU contract for exact payload recovery from a complete radix key cache."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_radix_selection import _custom_payload
from test.test_cute_radix_selection import _payload_selection

from helion._compiler.cute.radix_selection import radix_schedule
from helion.autotuner.compiler_coverage import CoverageDependency
from helion.exc import InvalidConfig
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _args(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.empty((3, 33), dtype=dtype),
        torch.tensor([-(1 << 40), 1 << 40, (1 << 32) + 7], dtype=torch.int64),
        17,
    )


def _config(bound: BoundKernel[object], **changes: object) -> Config:
    result = bound.config_spec.default_config()
    result.config.update(
        cute_selection_algorithm="radix",
        cute_radix_threads=1024,
        cute_radix_cluster_ctas=4,
        cute_radix_candidate_capacity=1024,
        cute_radix_histogram_mode="direct",
        cute_radix_candidate_key_cache=True,
        cute_radix_scalar_candidates=True,
        cute_radix_valid_domain=True,
        cute_radix_ticket_stores=True,
        cute_radix_packet_bits=256,
        cute_radix_scan_unroll=4,
        cute_radix_cached_payload_inverse=True,
    )
    result.config.update(changes)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_inverse_uses_exact_typed_payload_and_preserves_masks(
    dtype: torch.dtype,
) -> None:
    bound = _cpu_bind(_custom_payload, (*_args(dtype), False, True, True))
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    enabled = _config(bound)
    disabled = _config(bound, cute_radix_cached_payload_inverse=False)
    enabled_plan = radix_schedule(graph, enabled.config)
    disabled_plan = radix_schedule(graph, disabled.config)
    assert enabled_plan.allocations == disabled_plan.allocations
    source = bound.to_code(enabled)
    assert "inverse_raw_bits" not in bound.to_code(disabled)
    width = 8 * graph.x.element_size()
    assert f".bitcast(cutlass.Int{width})" in source
    assert f"cutlass.Uint{width}({(1 << width) - 1})" in source
    assert "cached_payload_raw" in source
    assert " < 16:" in source
    assert source.count(" % ") >= 2
    tree = ast.parse(source)
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "inverse_key" in ast.unparse(node.test)
    )
    assert ".load()" in ast.unparse(fallback)
    assert "cute.arch" not in ast.unparse(fallback)


@pytest.mark.parametrize(
    "dependency",
    [
        "cute_radix_candidate_key_cache",
        "cute_radix_valid_domain",
        "cute_radix_ticket_stores",
    ],
)
def test_inverse_requires_complete_valid_ticket_plan(dependency: str) -> None:
    bound = _cpu_bind(_custom_payload, (*_args(), False, False, False))
    with pytest.raises(InvalidConfig):
        bound.to_code(_config(bound, **{dependency: False}))


def test_inverse_declines_shifted_reads_and_smallest_domain() -> None:
    shifted = _cpu_bind(_custom_payload, (*_args(), True, False, False))
    with pytest.raises(InvalidConfig):
        shifted.to_code(_config(shifted))
    smallest = _cpu_bind(_payload_selection, (*_args(), False, False))
    with pytest.raises(InvalidConfig):
        smallest.to_code(_config(smallest))


def test_inverse_has_a_scoped_deferred_coverage_witness() -> None:
    key = "cute_radix_cached_payload_inverse"
    bound = _cpu_bind(_custom_payload, (*_args(), False, False, False))
    groups = {group.key: group for group in bound.config_spec.compiler_coverage_groups}
    group = groups[key]
    assert group.deferred and group.legacy is False and group.domain == (False, True)
    for dependency in (
        "cute_radix_candidate_key_cache",
        "cute_radix_valid_domain",
        "cute_radix_ticket_stores",
    ):
        assert (
            CoverageDependency(f"cute.{dependency[5:]}", dependency, True)
            in group.dependencies
        )
    assert len(group.witnesses) == 1
    generation = bound.config_spec.create_config_generation()
    generation.strict_config_pair(group.witnesses[0].carrier)
    requested = group.witnesses[0].carrier.config | {key: True}
    generation.strict_config_pair(Config.from_dict(requested))
    for kernel, args in (
        (_custom_payload, (*_args(), True, False, False)),
        (_payload_selection, (*_args(), False, False)),
    ):
        declined = _cpu_bind(kernel, args)
        declined_group = next(
            group
            for group in declined.config_spec.compiler_coverage_groups
            if group.key == key
        )
        assert declined_group.deferred and not declined_group.witnesses
