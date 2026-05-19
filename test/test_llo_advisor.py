"""Tests for the LLM-search LLO advisor integration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import textwrap
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

if TYPE_CHECKING:
    import types

# Load llo_advisor as a standalone module so the pure-string parse tests run
# even when torch isn't installed locally. Importing it normally would pull in
# helion/__init__.py, which depends on torch.
_HELION_ROOT = Path(__file__).resolve().parent.parent
_ADVISOR_PATH = _HELION_ROOT / "helion" / "autotuner" / "llm" / "llo_advisor.py"


def _load_advisor_standalone() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_llo_advisor_test", _ADVISOR_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_llo_advisor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class ParseAdviseOutputTest(unittest.TestCase):
    """Verify the advisor parses tpu_roofline.py --advise text into structured hints."""

    def test_parse_full_advise_output(self) -> None:
        _parse_advise_output = _load_advisor_standalone()._parse_advise_output

        sample = textwrap.dedent("""\
            === LLO stats ===
            === Per-lane busy (dynamic) ===
                      lane   cap        busy    util%
                       MXU     2  66,219,406    83.1%  ████ ← binding
                       XLU     2   8,382,566    10.5%
                    VSTORE     2  33,000,000    50.0%
                VLOAD:FILL     3  47,000,000    44.0%
              VSTORE:SPILL     2  35,000,000    49.0%
            === Roofline ===
              Predicted:        22128.60 µs   ← max(min_floor=30, additive=22128.60)
              Regime:           compute-bound (MXU)

            === Counterfactuals ===
                                     MXU=1.0  →   20876.1 µs (save  +1252.5 µs,  +5.7%)
                                     XLU=1.0  →   22043.3 µs (save    +85.3 µs,  +0.4%)
               -RP (kill pure-spill bundles)  →   21772.7 µs (save   +355.9 µs,  +1.6%)

            === Suggestions ===
              - MXU is the binding lane. Levers:
              - Larger inner-K block so the MXU pipeline amortizes setup
              - Reduce non-MXU work between matmuls to avoid stalling MXU
              - Register pressure: VLOAD:FILL 44%, VSTORE:SPILL 49%
        """)
        hints = _parse_advise_output(sample)
        assert hints.predicted_us is not None
        self.assertAlmostEqual(hints.predicted_us, 22128.60, places=1)
        self.assertEqual(hints.regime, "compute-bound (MXU)")
        self.assertEqual(hints.binding_lane, "MXU")
        self.assertAlmostEqual(hints.mxu_busy_pct or 0.0, 83.1, places=1)
        self.assertEqual(hints.register_pressure, "high")
        self.assertIn("MXU=1.0", hints.counterfactual_savings_us)
        self.assertGreater(hints.counterfactual_savings_us["MXU=1.0"], 1000)
        self.assertGreater(len(hints.suggestions), 0)
        self.assertTrue(any("MXU is the binding lane" in s for s in hints.suggestions))


class PredictorInputsFromArgsTest(unittest.TestCase):
    """Verify shape extraction from a kernel's argument tuple."""

    def test_extracts_bf16_tensors(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        predictor_inputs_from_args = (
            _load_advisor_standalone().predictor_inputs_from_args
        )

        q = torch.empty(8, 32, 8192, 256, dtype=torch.bfloat16, device="meta")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        spec = predictor_inputs_from_args([q, k, v])
        self.assertEqual(
            spec, "bf16:8x32x8192x256,bf16:8x32x8192x256,bf16:8x32x8192x256"
        )

    def test_skips_non_tensors(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        predictor_inputs_from_args = (
            _load_advisor_standalone().predictor_inputs_from_args
        )

        x = torch.empty(1024, 1024, dtype=torch.bfloat16, device="meta")
        # Mix in a Python scalar; advisor should ignore it
        spec = predictor_inputs_from_args([x, 42, "extra"])
        self.assertEqual(spec, "bf16:1024x1024")


class LloDumpDirFromEnvTest(unittest.TestCase):
    """Verify env-var parsing for LLO dump dir."""

    def test_parses_xla_jf_dump_to(self) -> None:
        llo_dump_dir_from_env = _load_advisor_standalone().llo_dump_dir_from_env

        with patch.dict(
            os.environ,
            {"LIBTPU_INIT_ARGS": "--foo=1 --xla_jf_dump_to=/tmp/llo --bar=2"},
        ):
            self.assertEqual(llo_dump_dir_from_env(), "/tmp/llo")

    def test_returns_none_when_unset(self) -> None:
        llo_dump_dir_from_env = _load_advisor_standalone().llo_dump_dir_from_env

        with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--other-flag=true"}):
            self.assertIsNone(llo_dump_dir_from_env())


class PromptIntegrationTest(unittest.TestCase):
    """Verify the refinement prompt picks up the bottleneck section."""

    def test_bottleneck_section_in_prompt(self) -> None:
        try:
            from helion.autotuner.llm.prompting import build_refinement_prompt
        except ImportError:
            self.skipTest("helion/torch not available")

        prompt = build_refinement_prompt(
            configs_per_round=4,
            compile_timeout_s=120,
            failed_count=0,
            total_count=10,
            search_state="  Best so far: 22.4 ms",
            anchor_configs="  Anchor 1 (best): 22.4 ms — block_sizes=[2,512,2048]",
            results="  22.4 ms — block_sizes=[2,512,2048]",
            top_patterns="  pallas_loop_type='unroll' (8/10)",
            failed_patterns="  None",
            bottleneck_analysis=(
                "  Anchor 1 (best): 22.4 ms — block_sizes=[2,512,2048]:\n"
                "  predicted: 22128.6 µs (compute-bound (MXU))\n"
                "  binding lane MXU @ 83% busy\n"
                "  register pressure: high\n"
                "  • MXU is the binding lane. Try larger inner-K block"
            ),
        )
        self.assertIn("Bottleneck Analysis (static from LLO)", prompt)
        self.assertIn("MXU @ 83% busy", prompt)
        self.assertIn("Try larger inner-K block", prompt)

    def test_empty_bottleneck_skips_section(self) -> None:
        try:
            from helion.autotuner.llm.prompting import build_refinement_prompt
        except ImportError:
            self.skipTest("helion/torch not available")

        prompt = build_refinement_prompt(
            configs_per_round=4,
            compile_timeout_s=120,
            failed_count=0,
            total_count=1,
            search_state="  Best so far: 1.0 ms",
            anchor_configs="  Anchor 1: 1.0 ms",
            results="  1.0 ms",
            top_patterns="  None",
            failed_patterns="  None",
            bottleneck_analysis="",  # advisor unavailable
        )
        self.assertNotIn("Bottleneck Analysis", prompt)


class RefinementStrategyOverrideTest(unittest.TestCase):
    """Verify the advisor produces actionable refinement instructions."""

    def test_mxu_bound_produces_directional_lines(self) -> None:
        mod = _load_advisor_standalone()
        hints = mod.BottleneckHints(
            predicted_us=22128.6,
            regime="compute-bound (MXU)",
            binding_lane="MXU",
            mxu_busy_pct=83.1,
            register_pressure="high",
            counterfactual_savings_us={"MXU=1.0": 1252.5},
            suggestions=["MXU is the binding lane."],
        )
        lines = mod.refinement_strategy_from_hints([("anchor1", hints)])
        joined = "\n".join(lines)
        # Kernel-agnostic phrasing: name the bottleneck and the lever, not
        # specific block_sizes positions.
        self.assertIn("compute-bound on MXU", joined)
        self.assertIn("reduction loops", joined)
        self.assertNotIn("block_b", joined)
        self.assertNotIn("first block_sizes entry", joined)
        self.assertIn("Multi-field changes are encouraged", joined)
        self.assertIn("Register pressure is high", joined)

    def test_empty_hints_returns_no_override(self) -> None:
        mod = _load_advisor_standalone()
        self.assertEqual(mod.refinement_strategy_from_hints([]), [])

    def test_strategy_override_used_in_prompt(self) -> None:
        try:
            from helion.autotuner.llm.prompting import build_refinement_prompt
        except ImportError:
            self.skipTest("helion/torch not available")

        prompt = build_refinement_prompt(
            configs_per_round=4,
            compile_timeout_s=120,
            failed_count=0,
            total_count=10,
            search_state="  Best so far: 22.4 ms",
            anchor_configs="  Anchor 1: 22.4 ms",
            results="  22.4 ms",
            top_patterns="  None",
            failed_patterns="  None",
            bottleneck_analysis="some bottleneck info",
            refinement_strategy_override=[
                "Best config is MXU-bound. Try larger inner-K block_sizes.",
                "Multi-field changes are encouraged this round.",
            ],
        )
        self.assertIn("MXU-bound", prompt)
        self.assertIn("Multi-field changes are encouraged", prompt)
        # Default strategy text should be suppressed
        self.assertNotIn(
            "About two thirds of configs should be 1-field mutations", prompt
        )


if __name__ == "__main__":
    unittest.main()
