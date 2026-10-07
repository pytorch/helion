from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.int4_gemm import matmul_bf16_int4
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_flatten_nested_reductions import _joint_sum
from test.test_cute_full_slice_matmul import _matmul
from test.test_cute_materialized_fission import _se_args
from test.test_cute_shared_rhs_grouped import _offset_mm

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


_STRUCTURAL_SETTINGS = (
    "cute_region_fission",
    "cute_full_slice_matmul_tiling",
    "cute_segmented_matmul_tiling",
    "cute_flatten_nested_reductions",
    "cute_materialize_transformed_operands",
)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in _STRUCTURAL_SETTINGS:
        monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
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


def _case(
    setting: str,
) -> tuple[Callable[..., Any], tuple[object, ...], str]:
    if setting == "cute_region_fission":
        return squeeze_and_excitation_net_fwd.fn, _se_args(), "fission"
    if setting == "cute_full_slice_matmul_tiling":
        return (
            _matmul,
            (
                torch.empty((32, 16), dtype=torch.float16),
                torch.empty((16, 64), dtype=torch.float16),
            ),
            "_helion_fullslice_k",
        )
    if setting == "cute_segmented_matmul_tiling":
        return (
            _offset_mm,
            (
                torch.empty((37, 16), dtype=torch.bfloat16),
                torch.empty((16, 24), dtype=torch.bfloat16),
                torch.tensor([0, 0, 1, 19, 37], dtype=torch.int64),
            ),
            "_helion_segment_group",
        )
    if setting == "cute_flatten_nested_reductions":
        return (
            _joint_sum,
            (torch.empty((37, 69)), torch.tensor([0, 0, 1, 19, 37], dtype=torch.int64)),
            "_helion_flat_reduction",
        )
    assert setting == "cute_materialize_transformed_operands"
    return (
        matmul_bf16_int4.fn,
        (
            torch.empty((16, 40), dtype=torch.bfloat16),
            torch.empty((20, 24), dtype=torch.int8),
        ),
        "_helion_materialized_operand",
    )


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
    expected_settings = {
        name: getattr(bound.settings, name) for name in _STRUCTURAL_SETTINGS
    }
    decorator = bound.format_kernel_decorator(config, bound.settings)
    expected_source = bound.to_code(config)
    assert json.dumps(config.config, sort_keys=True) == expected_config
    if conflicting_environment:
        monkeypatch.setenv("HELION_BACKEND", "triton")
        for name, enabled in expected_settings.items():
            monkeypatch.setenv(f"HELION_{name.upper()}", str(int(not enabled)))
    else:
        monkeypatch.delenv("HELION_BACKEND", raising=False)
        for name in _STRUCTURAL_SETTINGS:
            monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
    replay = _replay(decorator, bound.kernel.fn)
    assert replay.settings.backend == "cute"
    assert {
        name: getattr(replay.settings, name) for name in _STRUCTURAL_SETTINGS
    } == expected_settings
    assert json.dumps(replay.configs[0].config, sort_keys=True) == expected_config
    rebound = replay._bind_isolated(args)
    assert _host(rebound) == _host(bound)
    assert (
        rebound.config_spec.cache_fingerprint_hash()
        == bound.config_spec.cache_fingerprint_hash()
    )
    assert rebound.to_code() == expected_source


@pytest.mark.parametrize("setting", _STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("conflicting_environment", [False, True])
def test_structural_binding_and_config_roundtrip(
    setting: str,
    enabled: bool,
    conflicting_environment: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function, args, marker = _case(setting)
    switches = dict.fromkeys(_STRUCTURAL_SETTINGS, False) | {setting: enabled}
    bound = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **switches,
    )._bind_isolated(args)
    if marker == "fission":
        assert (bound.env.cute_fission_plan is not None) == enabled
    else:
        assert (marker in _host(bound)) == enabled
    _check_roundtrip(
        bound, args, monkeypatch, conflicting_environment=conflicting_environment
    )


@pytest.mark.parametrize("from_environment", [False, True])
def test_composed_structural_settings_roundtrip(
    from_environment: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    function, args, marker = _case("cute_full_slice_matmul_tiling")
    switches = dict.fromkeys(_STRUCTURAL_SETTINGS, True)
    if from_environment:
        for name in _STRUCTURAL_SETTINGS:
            monkeypatch.setenv(f"HELION_{name.upper()}", "1")
        switches = {}
    bound = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **switches,
    )._bind_isolated(args)
    assert marker in _host(bound)
    _check_roundtrip(bound, args, monkeypatch, conflicting_environment=True)


def test_dynamic_materialization_specializations_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting = "cute_materialize_transformed_operands"
    function, args, marker = _case(setting)
    bound = helion.kernel(
        function,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        **(dict.fromkeys(_STRUCTURAL_SETTINGS, False) | {setting: True}),
    )._bind_isolated(args)
    assert marker in _host(bound)
    assert bound.env.specialized_vars
    assert bound.env.shape_env.guards
    _check_roundtrip(bound, args, monkeypatch, conflicting_environment=True)


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
    assert all(not getattr(bound.settings, name) for name in _STRUCTURAL_SETTINGS)
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
        **dict.fromkeys(_STRUCTURAL_SETTINGS, True),
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
    decorator = _pointwise_bound.format_kernel_decorator(
        config, _pointwise_bound.settings
    )
    config.block_sizes[0] = 32
    config.config["loop_orders"][0][0] = 1
    for name in _STRUCTURAL_SETTINGS:
        setattr(_pointwise_bound.settings, name, True)
    first = _replay(decorator, _pointwise)
    assert json.dumps(first.configs[0].config, sort_keys=True) == expected_config
    first.configs[0].block_sizes[0] = 64
    first.configs[0].config["loop_orders"][0][0] = 2
    first.settings.cute_full_slice_matmul_tiling = True
    second = _replay(decorator, _pointwise)
    assert json.dumps(second.configs[0].config, sort_keys=True) == expected_config
    assert all(not getattr(second.settings, name) for name in _STRUCTURAL_SETTINGS)


@pytest.mark.parametrize("from_environment", [False, True])
def test_materialized_composed_structural_settings_roundtrip(
    from_environment: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _se_args()
    switches = dict.fromkeys(_STRUCTURAL_SETTINGS, True)
    if from_environment:
        for name in _STRUCTURAL_SETTINGS:
            monkeypatch.setenv(f"HELION_{name.upper()}", "1")
        switches = {}
    bound = helion.kernel(
        squeeze_and_excitation_net_fwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **switches,
    )._bind_isolated(args)
    assert bound.env.cute_fission_plan is not None
    assert "_helion_fullslice_k" in _host(bound)
    _check_roundtrip(bound, args, monkeypatch, conflicting_environment=True)


def test_materialized_decorator_is_an_immutable_config_and_settings_snapshot(
    _pointwise_bound: BoundKernel[Any],
) -> None:
    config = helion.Config(block_sizes=[16], loop_orders=[[0]])
    expected_config = json.dumps(config.config, sort_keys=True)
    decorator = _pointwise_bound.format_kernel_decorator(
        config, _pointwise_bound.settings
    )
    config.block_sizes[0] = 32
    config.config["loop_orders"][0][0] = 1
    for name in _STRUCTURAL_SETTINGS:
        setattr(_pointwise_bound.settings, name, True)
    first = _replay(decorator, _pointwise)
    assert json.dumps(first.configs[0].config, sort_keys=True) == expected_config
    first.configs[0].block_sizes[0] = 64
    first.configs[0].config["loop_orders"][0][0] = 2
    first.settings.cute_region_fission = True
    second = _replay(decorator, _pointwise)
    assert json.dumps(second.configs[0].config, sort_keys=True) == expected_config
    assert all(not getattr(second.settings, name) for name in _STRUCTURAL_SETTINGS)
