"""Tests for the external autotuner API (helion.autotuner.external.autotune).

These tests use pure PyTorch -- no Helion DSL, no Triton, no CuTeDSL.
They verify that the search algorithms work correctly with user-defined
tunables and plain Python compile/baseline functions.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCpu
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.external import autotune

if TYPE_CHECKING:
    from helion.runtime.config import Config


def _scaled_add_compile(config: Config):
    scale = config["scale"]
    return lambda a, b: a * scale + b


@onlyBackends(["triton"])
@skipIfCpu("external autotuner requires GPU for benchmarking")
class TestExternalAutotune(TestCase):
    def test_basic_autotune(self):
        a = torch.randn(1024, device=DEVICE)
        b = torch.randn(1024, device=DEVICE)

        best = autotune(
            tunables={
                "scale": IntegerFragment(1, 4, 2),
                "tile": PowerOfTwoFragment(32, 256, 64),
            },
            compile_fn=_scaled_add_compile,
            baseline_fn=lambda a, b: a * 2 + b,
            args=(a, b),
            algorithm="RandomSearch",
            count=5,
        )

        assert "scale" in best
        assert "tile" in best
        assert isinstance(best["scale"], int)
        assert isinstance(best["tile"], int)

    def test_config_only_has_user_keys(self):
        a = torch.randn(256, device=DEVICE)
        b = torch.randn(256, device=DEVICE)

        best = autotune(
            tunables={
                "tile": PowerOfTwoFragment(32, 256, 64),
                "unroll": IntegerFragment(1, 4, 2),
            },
            compile_fn=lambda config: operator.add,
            baseline_fn=operator.add,
            args=(a, b),
            algorithm="RandomSearch",
            count=3,
        )

        config_keys = set(best.config.keys())
        assert config_keys == {"tile", "unroll"}, (
            f"Config should only have user keys, got {config_keys}"
        )

    def test_pattern_search(self):
        a = torch.randn(512, device=DEVICE)
        b = torch.randn(512, device=DEVICE)

        best = autotune(
            tunables={
                "block": PowerOfTwoFragment(32, 256, 64),
                "unroll": IntegerFragment(1, 4, 1),
            },
            compile_fn=lambda config: operator.add,
            baseline_fn=operator.add,
            args=(a, b),
            algorithm="PatternSearch",
            max_generations=2,
            initial_population=5,
        )

        assert "block" in best
        assert "unroll" in best

    def test_invalid_algorithm_raises(self):
        a = torch.randn(64, device=DEVICE)

        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            autotune(
                tunables={"x": IntegerFragment(1, 4, 2)},
                compile_fn=lambda config: lambda a: a,
                baseline_fn=lambda a: a,
                args=(a,),
                algorithm="DoesNotExist",
            )

    def test_unknown_search_kwarg_raises(self):
        a = torch.randn(64, device=DEVICE)

        with self.assertRaisesRegex(ValueError, "Unknown search kwargs"):
            autotune(
                tunables={"x": IntegerFragment(1, 4, 2)},
                compile_fn=lambda config: lambda a: a,
                baseline_fn=lambda a: a,
                args=(a,),
                algorithm="PatternSearch",
                max_generation=2,
            )

    def test_no_baseline_fn(self):
        a = torch.randn(256, device=DEVICE)
        b = torch.randn(256, device=DEVICE)

        best = autotune(
            tunables={"scale": IntegerFragment(1, 4, 2)},
            compile_fn=_scaled_add_compile,
            args=(a, b),
            algorithm="RandomSearch",
            count=3,
        )

        assert "scale" in best


if __name__ == "__main__":
    from helion._testing import main

    main()
