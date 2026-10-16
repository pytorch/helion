from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

import helion
from helion.autotuner.benchmark_provider import _MAX_REFERENCE_BASELINE_ATTEMPTS
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider


class _Spec:
    """A config spec whose execution default may be promoted beyond the
    autotuning reference config."""

    def __init__(self, reference: dict[str, Any], execution: dict[str, Any]) -> None:
        self._reference = reference
        self._execution = execution
        self.backend = SimpleNamespace(classify_autotune_exception=lambda error: "skip")

    def autotune_reference_config(self) -> helion.Config:
        return helion.Config(**self._reference)

    def default_config(self) -> helion.Config:
        return helion.Config(**self._execution)

    def shrink_block_sizes_once(self, config: helion.Config) -> helion.Config | None:
        sizes = list(config.config["block_sizes"])
        if max(sizes) <= 1:
            return None
        sizes[sizes.index(max(sizes))] //= 2
        return helion.Config(**dict(config.config, block_sizes=sizes))


def _provider(
    spec: _Spec, runs: Any
) -> tuple[LocalBenchmarkProvider, list[helion.Config], list[str]]:
    """A provider around ``spec``; a compiled config runs ``runs(config)``:
    True adds one to the input, False raises."""
    compiled: list[helion.Config] = []
    warnings: list[str] = []

    def compile_config(config: helion.Config, allow_print: bool = False) -> object:
        compiled.append(config)

        def run(x: torch.Tensor) -> torch.Tensor:
            if not runs(config):
                raise RuntimeError(f"{config.config} cannot run")
            return x + 1

        return run

    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        compile_config=compile_config, env=SimpleNamespace(process_group_name=None)
    )
    provider.config_spec = spec  # pyrefly: ignore[bad-assignment]
    provider.args = (torch.zeros(4),)
    provider.log = SimpleNamespace(warning=warnings.append)  # pyrefly: ignore[bad-assignment]
    return provider, compiled, warnings


def test_the_execution_default_produces_the_baseline_when_promoted() -> None:
    # The fast candidates share the promoted default's arithmetic; the
    # conservative reference is not run at all.
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(spec, lambda config: True)
    output, args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma"]
    assert torch.equal(output, torch.ones(4)) and not warnings
    assert args[0] is not provider.args[0]


def test_the_reference_config_backs_off_a_failing_execution_default() -> None:
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(spec, lambda c: c.config["mode"] == "simt")
    output, _args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma", "simt"]
    assert torch.equal(output, torch.ones(4))
    assert len(warnings) == 1 and "execution default" in warnings[0]


def test_an_unpromoted_default_runs_the_reference_once() -> None:
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "simt"}
    )
    provider, compiled, _warnings = _provider(spec, lambda config: True)
    provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["simt"]


def test_the_reference_config_still_shrinks_its_block_sizes() -> None:
    # Only the smallest block size runs: the reference backs off three times
    # after its own attempt, as before.
    spec = _Spec(
        {"block_sizes": [8], "mode": "simt"}, {"block_sizes": [8], "mode": "simt"}
    )
    provider, compiled, warnings = _provider(
        spec, lambda c: c.config["block_sizes"][0] == 1
    )
    output, _args = provider._compute_reference_baseline()
    assert [c.config["block_sizes"][0] for c in compiled] == [8, 4, 2, 1]
    assert len(compiled) == _MAX_REFERENCE_BASELINE_ATTEMPTS
    assert torch.equal(output, torch.ones(4))
    assert warnings[-1].startswith("Using block sizes [1]")
