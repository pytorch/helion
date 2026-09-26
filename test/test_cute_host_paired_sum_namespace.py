"""Real generated callers must retain user bindings of an injected helper name."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_host_paired_sum import _config
from test.test_cute_host_paired_sum import _literal
from test.test_cute_host_paired_sum_seeds import _population

import helion
from helion._compiler.cute.host_paired_sum import PAIRED_SUM_KEY
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _argument(
    _cute_try_paired_sum_cast: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(b)
    for i, j in hl.tile(b.shape):
        partial[i, j] = b[i, j]
    left = _cute_try_paired_sum_cast.sum(0).to(torch.float32)
    right = partial.sum(0).to(torch.float32)
    return left, right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _earlier_local(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _cute_try_paired_sum_cast = a
    partial = torch.empty_like(b)
    for i, j in hl.tile(b.shape):
        partial[i, j] = b[i, j]
    left = _cute_try_paired_sum_cast.sum(0).to(torch.float32)
    right = partial.sum(0).to(torch.float32)
    return left, right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _later_local(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(b)
    for i, j in hl.tile(b.shape):
        partial[i, j] = b[i, j]
    left = a.sum(0).to(torch.float32)
    right = partial.sum(0).to(torch.float32)
    _cute_try_paired_sum_cast = left
    return _cute_try_paired_sum_cast, right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _result_local(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(b)
    for i, j in hl.tile(b.shape):
        partial[i, j] = b[i, j]
    _cute_try_paired_sum_cast = a.sum(0).to(torch.float32)
    right = partial.sum(0).to(torch.float32)
    return _cute_try_paired_sum_cast, right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cute_try_paired_sum_cast(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(b)
    for i, j in hl.tile(b.shape):
        partial[i, j] = b[i, j]
    left = a.sum(0).to(torch.float32)
    right = partial.sum(0).to(torch.float32)
    return left, right


@pytest.mark.parametrize(
    "kernel",
    (_argument, _earlier_local, _later_local, _result_local, _cute_try_paired_sum_cast),
)
def test_helper_name_collisions_decline_before_search_and_execute_original_host(
    kernel: helion.Kernel, tmp_path: Path
) -> None:
    initial_cuda_state = torch.cuda.is_initialized()
    values = (
        torch.arange(17 * 20, dtype=torch.float32).view(17, 20),
        -torch.arange(17 * 20, dtype=torch.float32).view(17, 20),
    )
    bound = _cpu_bind(kernel, values)
    assert not bound.config_spec.cute_host_paired_sum_available
    population = _population(bound, values)
    assert all(PAIRED_SUM_KEY not in config for config in population)
    assert all(
        PAIRED_SUM_KEY not in config
        for config in bound.config_spec.compiler_seed_configs
    )
    code = bound.to_code(_config(bound, "off"))
    module = ast.parse(code)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_try_paired_sum_cast"
        for node in ast.walk(module)
    )
    for choice in ("mapped", "narrow"):
        with pytest.raises(InvalidConfig, match="typed independent pair"):
            bound.to_code(_config(bound, choice))
    bound.set_config(_config(bound, "off"))
    caller = bound._run
    assert caller is not None
    device = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    )
    calls = []

    def launcher(kernel, grid, *args, **kwargs):
        # Execute only the already-proved copy producer on CPU. The real
        # BoundKernel and its generated host/fallback/returns remain intact.
        bindings = dict(
            zip((argument.arg for argument in device.args.args), args, strict=True)
        )
        assert set(bindings) == {"b", "partial"}
        bindings["partial"].copy_(bindings["b"])
        calls.append((id(kernel), grid, kwargs))

    with patch.dict(caller.__kwdefaults__, {"_launcher": launcher}):
        first = bound(*values)
        replacement = tuple(value.clone() for value in values)
        second = bound(*replacement)
    for outputs in (first, second):
        for output, value in zip(outputs, values, strict=True):
            assert torch.equal(output, value.sum(0))
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert bound._run is caller
    assert torch.cuda.is_initialized() == initial_cuda_state
    assert (
        len({output.data_ptr() for outputs in (first, second) for output in outputs})
        == 4
    )
    (tmp_path / "generated.py").write_text(code)
    (tmp_path / "actual-bound-caller.json").write_text(
        json.dumps(
            {
                "kernel": kernel.fn.__name__,
                "initial_population": len(population),
                "availability": False,
                "explicit_choices": "mapped/narrow both InvalidConfig",
                "actual_bound_calls": len(calls),
                "fresh_pointer_sets": 2,
                "output_exact": True,
                "GPU_work": False,
            },
            indent=2,
        )
        + "\n"
    )


def test_no_collision_retains_typed_choice_and_initial_population() -> None:
    values = (torch.ones(17, 20), torch.ones(17, 20))
    bound = _cpu_bind(_literal, values)
    assert bound.config_spec.cute_host_paired_sum_available
    first = _population(bound, values)[:5]
    assert {config[PAIRED_SUM_KEY] for config in first} == {
        "off",
        "mapped",
        "narrow",
    }
    assert "_cute_try_paired_sum_cast(" in bound.to_code(_config(bound, "narrow"))
