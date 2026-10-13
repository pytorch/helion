from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_fuser_occurrences import _args
from .test_cute_fuser_occurrences import _state_reduction
import helion._compiler.autotuner_heuristics as heuristics
from helion._compiler.autotuner_heuristics.cute_computed_sweep import (
    append_computed_sweep_seed,
)
from helion._compiler.autotuner_heuristics.cute_computed_sweep import (
    has_computed_sweep_candidate,
)
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import _CUTE_REDUCTION_RELOAD_CHOICES
from helion.language.memory_ops import store

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR


def test_old_choices_seed_prefix_and_default_are_unchanged() -> None:
    assert _CUTE_REDUCTION_RELOAD_CHOICES == ("auto", "register", "gmem", "computed")
    args = _args(torch.bfloat16, 32, 128)
    with (
        _cpu_codegen(),
        patch.object(
            heuristics,
            "append_vector_computed_sweep_seed",
            side_effect=lambda env, ir, seeds: seeds,
        ),
    ):
        with patch.object(
            heuristics,
            "append_computed_sweep_seed",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old = _state_reduction._bind_isolated(args)
        current = _state_reduction._bind_isolated(args)
        before = old.config_spec.compiler_seed_configs
        after = current.config_spec.compiler_seed_configs
        assert after[: len(before)] == before
        assert len(after) == len(before) + 1
        assert after[-1]["cute_reduction_reloads"] == ["computed"]
        assert (
            old.config_spec.compiler_default_config
            == current.config_spec.compiler_default_config
        )
        with current.env:
            generation = ConfigGeneration(current.config_spec)
            seed = generation.unflatten(generation.flatten(after[-1]))
            assert seed["cute_reduction_reloads"] == ["computed"]
        with current.env.use_runtime_arg_values(
            _runtime_values(_state_reduction, args)
        ):
            source = current.to_code(seed)
        assert "_computed_cache_" in source


@pytest.mark.parametrize("override", ["auto", "register", "gmem", "computed"])
def test_structural_seed_respects_explicit_reload_override(override: str) -> None:
    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(_args(torch.bfloat16, 32, 128))
        with (
            bound.env,
            patch.object(
                bound.settings,
                "autotune_config_overrides",
                {"cute_reduction_reloads": [override]},
            ),
        ):
            assert bound.host_function is not None
            result = append_computed_sweep_seed(
                bound.env, bound.host_function.device_ir, []
            )
    assert bool(result) == (override == "computed")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shared", [False, True])
def test_discovery_requires_shared_computed_fp32_ancestor(
    dtype: torch.dtype, shared: bool
) -> None:
    graph = torch.fx.Graph()
    raw = graph.placeholder("x")
    value = graph.call_function(torch.ops.aten.mul.Tensor, (raw, raw))
    value.meta["val"] = torch.empty((4, 128), dtype=dtype)
    reduction = graph.call_function(torch.ops.aten.sum.dim_IntList, (value, [-1]))
    target = graph.placeholder("out")
    graph.call_function(store, (target, [], value if shared else raw))
    graph.output(reduction)
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert has_computed_sweep_candidate(cast("DeviceIR", ir)) == (
        shared and dtype == torch.float32
    )
