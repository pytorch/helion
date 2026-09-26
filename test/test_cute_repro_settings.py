from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.settings import Settings

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield
    torch.set_num_threads(previous_threads)


def _host(bound: BoundKernel[Any]) -> str:
    assert bound.host_function is not None
    return ast.dump(ast.Module(body=bound.host_function.body, type_ignores=[]))


def _replay(decorator: str, function: Callable[..., Any]) -> Kernel[Any]:
    # Evaluate the real public decorator: config construction and Settings
    # precedence must work without injecting missing replay-only kwargs.
    return eval(decorator.removeprefix("@"), {"helion": helion, "torch": torch})(
        function
    )


def _check_roundtrip(
    bound: BoundKernel[Any],
    args: tuple[object, ...],
    monkeypatch: pytest.MonkeyPatch,
    *,
    conflicting_environment: bool,
) -> None:
    config = bound.config_spec.default_config()
    expected_config = json.dumps(config.config, sort_keys=True)
    expected_settings = (
        bound.settings.backend,
        bound.settings.static_shapes,
        bound.settings.index_dtype,
    )
    decorator = bound.format_kernel_decorator(config, bound.settings)
    expected_source = bound.to_code(config)
    assert json.dumps(config.config, sort_keys=True) == expected_config
    if conflicting_environment:
        monkeypatch.setenv("HELION_BACKEND", "triton")
    else:
        monkeypatch.delenv("HELION_BACKEND", raising=False)
    replay = _replay(decorator, bound.kernel.fn)
    assert replay.settings.backend == "cute"
    assert (
        replay.settings.backend,
        replay.settings.static_shapes,
        replay.settings.index_dtype,
    ) == expected_settings
    assert json.dumps(replay.configs[0].config, sort_keys=True) == expected_config
    rebound = replay._bind_isolated(args)
    assert _host(rebound) == _host(bound)
    assert (
        rebound.config_spec.cache_fingerprint_hash()
        == bound.config_spec.cache_fingerprint_hash()
    )
    assert rebound.to_code() == expected_source


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * 2 + 1
    return out


@pytest.fixture
def _pointwise_bound() -> BoundKernel[Any]:
    return helion.kernel(
        _pointwise, backend="cute", autotune_effort="none"
    )._bind_isolated((torch.empty(19),))


@pytest.mark.parametrize("index_dtype", [None, torch.int64])
def test_default_unrelated_kernel_roundtrip(
    index_dtype: torch.dtype | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = (torch.empty(19),)
    bound = helion.kernel(
        _pointwise,
        backend="cute",
        autotune_effort="none",
        index_dtype=index_dtype,
    )._bind_isolated(args)
    _check_roundtrip(bound, args, monkeypatch, conflicting_environment=True)


@pytest.mark.parametrize("backend", ["triton", "pallas", "tileir"])
@pytest.mark.parametrize("index_dtype", [None, torch.int64])
def test_other_backend_decorators_are_unchanged(
    backend: str,
    index_dtype: torch.dtype | None,
    _pointwise_bound: BoundKernel[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_TILE", "1")
    settings = Settings(
        backend=backend,
        static_shapes=False,
        index_dtype=index_dtype,
    )
    config = helion.Config(block_sizes=[32], num_warps=4)
    expected = (
        "@helion.kernel(config=helion.Config(block_sizes=[32], num_warps=4), "
        "static_shapes=False"
    )
    if index_dtype is not None:
        expected += ", index_dtype=torch.int64"
    expected += ")"
    assert _pointwise_bound.format_kernel_decorator(config, settings) == expected


def test_decorator_is_an_immutable_config_and_settings_snapshot(
    _pointwise_bound: BoundKernel[Any],
) -> None:
    config = helion.Config(block_sizes=[16], loop_orders=[[0]])
    expected_config = json.dumps(config.config, sort_keys=True)
    expected_static_shapes = _pointwise_bound.settings.static_shapes
    decorator = _pointwise_bound.format_kernel_decorator(
        config, _pointwise_bound.settings
    )
    config.block_sizes[0] = 32
    config.config["loop_orders"][0][0] = 1
    _pointwise_bound.settings.static_shapes = not expected_static_shapes
    first = _replay(decorator, _pointwise)
    assert json.dumps(first.configs[0].config, sort_keys=True) == expected_config
    first.configs[0].block_sizes[0] = 64
    first.configs[0].config["loop_orders"][0][0] = 2
    first.settings.static_shapes = not expected_static_shapes
    second = _replay(decorator, _pointwise)
    assert json.dumps(second.configs[0].config, sort_keys=True) == expected_config
    assert second.settings.static_shapes == expected_static_shapes
