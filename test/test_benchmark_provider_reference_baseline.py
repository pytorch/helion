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
    spec: _Spec, runs: Any, value: Any = None, compiles: Any = None
) -> tuple[LocalBenchmarkProvider, list[helion.Config], list[str]]:
    """A provider around ``spec``; a compiled config runs ``runs(config)``:
    True adds one to the input (``value(config)`` when given), False
    raises (``runs`` may raise itself).  A config ``compiles(config)``
    refuses raises ``InvalidConfig`` at compile time, as the backends do."""
    compiled: list[helion.Config] = []
    warnings: list[str] = []

    def compile_config(config: helion.Config, allow_print: bool = False) -> object:
        compiled.append(config)
        if compiles is not None and not compiles(config):
            raise helion.exc.InvalidConfig(
                f"{config.config} is refused at compile time"
            )

        def run(x: torch.Tensor) -> torch.Tensor:
            if not runs(config):
                raise RuntimeError(f"{config.config} cannot run")
            return x + (1 if value is None else value(config))

        return run

    provider = object.__new__(LocalBenchmarkProvider)
    provider.kernel = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        compile_config=compile_config,
        # The baseline runs poison the device's cached memory first (CUDA only).
        env=SimpleNamespace(process_group_name=None, device=torch.device("cpu")),
        format_kernel_decorator=lambda config, settings: f"@helion.kernel({config})",
        maybe_log_repro=lambda *args, **kwargs: None,
    )
    provider.config_spec = spec  # pyrefly: ignore[bad-assignment]
    # The baseline runs poison the device's cached memory when the accuracy
    # check is on; the fake provider's kernels are plain Python.
    provider.settings = SimpleNamespace(autotune_accuracy_check=False)  # pyrefly: ignore[bad-assignment]
    provider.args = (torch.zeros(4),)
    provider.log = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        warning=warnings.append,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    return provider, compiled, warnings


def test_the_execution_default_produces_the_baseline_when_promoted() -> None:
    # The fast candidates share the promoted default's arithmetic: its output
    # is the baseline, once the conservative reference (run second) agrees.
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(
        spec, lambda config: True, lambda c: 1.0 if c.config["mode"] == "simt" else 1.01
    )
    output, args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma", "simt"]
    assert torch.equal(output, torch.full((4,), 1.01)) and not warnings
    assert args[0] is not provider.args[0]


def test_a_promoted_default_disagreeing_with_the_reference_is_refused() -> None:
    # The accuracy check judges every candidate against the baseline; a
    # promoted default whose lowering is wrong would reject the right
    # candidates and keep the wrong ones, so the search is refused, naming
    # both configs (full_gla's dqkg at 9534981b9: the tcgen05 default
    # dropped dk's intra-chunk dot and a wrong winner shipped).
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, _warnings = _provider(
        spec, lambda config: True, lambda c: 1.0 if c.config["mode"] == "simt" else 2.0
    )
    try:
        provider._compute_reference_baseline()
    except helion.exc.AutotuneError as error:
        message = str(error)
    else:
        raise AssertionError("the disagreement was not refused")
    assert [c.config["mode"] for c in compiled] == ["mma", "simt"]
    assert "disagree on the kernel's outputs" in message
    assert "relative L2 difference 5.000e-01" in message
    assert "Execution default: @helion.kernel(" in message
    assert "Autotuning reference config: @helion.kernel(" in message
    assert "mode='mma'" in message and "mode='simt'" in message


def test_a_reference_that_cannot_run_leaves_the_promoted_default_unchecked() -> None:
    # The reference config and every shrink of it fail where the default ran:
    # reported, and the default's output is the baseline as before the check.
    spec = _Spec(
        {"block_sizes": [8], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(spec, lambda c: c.config["mode"] == "mma")
    output, _args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma"] + ["simt"] * (
        _MAX_REFERENCE_BASELINE_ATTEMPTS
    )
    assert [c.config["block_sizes"][0] for c in compiled[1:]] == [8, 4, 2, 1]
    assert torch.equal(output, torch.ones(4))
    assert "accuracy baseline unchecked" in warnings[-1]


def test_the_reference_config_backs_off_a_failing_execution_default() -> None:
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(spec, lambda c: c.config["mode"] == "simt")
    output, _args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma", "simt"]
    assert torch.equal(output, torch.ones(4))
    assert len(warnings) == 1 and "execution default" in warnings[0]


def test_a_default_the_compiler_refuses_falls_back_to_the_reference() -> None:
    # The cute backend refuses a config at compile time with InvalidConfig
    # (a resident-storage or vector-width rule), which the candidate path
    # skips; a promoted default it refuses falls back to the reference, as
    # every other failure of the default does.
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    provider, compiled, warnings = _provider(
        spec, lambda config: True, compiles=lambda c: c.config["mode"] == "simt"
    )
    output, _args = provider._compute_reference_baseline()
    assert [c.config["mode"] for c in compiled] == ["mma", "simt"]
    assert torch.equal(output, torch.ones(4))
    assert len(warnings) == 1 and "execution default" in warnings[0]
    assert "is refused at compile time" in warnings[0]
    assert warnings[0].endswith("retrying with the autotuning reference config")


def test_an_unrecoverable_runtime_error_from_the_default_stops_the_search() -> None:
    # An illegal instruction leaves the device in a bad state: no fallback.
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )

    def runs(config: helion.Config) -> bool:
        if config.config["mode"] == "mma":
            raise RuntimeError("CUDA error: an illegal instruction was encountered")
        return True

    provider, compiled, warnings = _provider(spec, runs)
    try:
        provider._compute_reference_baseline()
    except helion.exc.InvalidConfig as error:
        assert str(error).startswith("Execution default failed while computing")
        assert "illegal instruction" in str(error.__cause__)
    else:
        raise AssertionError("the unrecoverable error was swallowed")
    assert [c.config["mode"] for c in compiled] == ["mma"] and not warnings


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


def test_a_refused_reference_falls_back_to_a_sampled_config() -> None:
    # The compiler refuses the reference config and every shrink of it (a
    # lowering rule no block size fixes); a config sampled from the search
    # space that runs becomes the baseline instead of aborting the search.
    spec = _Spec(
        {"block_sizes": [8], "mode": "simt"}, {"block_sizes": [8], "mode": "simt"}
    )
    provider, compiled, warnings = _provider(
        spec, lambda config: True, compiles=lambda c: c.config.get("sampled") is True
    )
    sampled = [
        helion.Config(block_sizes=[16], mode="simt"),
        helion.Config(block_sizes=[32], mode="simt", sampled=True),
    ]
    provider._sampled_baseline_configs = lambda count: sampled  # pyrefly: ignore[bad-assignment]
    output, _args = provider._compute_reference_baseline()
    assert [c.config["block_sizes"][0] for c in compiled] == [8, 4, 2, 1, 16, 32]
    assert torch.equal(output, torch.ones(4))
    assert warnings[-1].startswith("Using the sampled config")
    assert "is refused at compile time" in warnings[-1]


def test_a_refused_reference_without_a_running_sample_raises() -> None:
    spec = _Spec(
        {"block_sizes": [8], "mode": "simt"}, {"block_sizes": [8], "mode": "simt"}
    )
    provider, compiled, _warnings = _provider(
        spec, lambda config: True, compiles=lambda c: False
    )
    provider._sampled_baseline_configs = lambda count: [  # pyrefly: ignore[bad-assignment]
        helion.Config(block_sizes=[16], mode="simt")
    ]
    try:
        provider._compute_reference_baseline()
    except helion.exc.InvalidConfig as error:
        assert str(error).startswith("Autotuning reference config failed")
    else:
        raise AssertionError("the refused reference was not reported")
    assert [c.config["block_sizes"][0] for c in compiled] == [8, 4, 2, 1, 16]


def test_an_unrecoverable_error_from_a_sampled_config_stops_the_search() -> None:
    spec = _Spec(
        {"block_sizes": [8], "mode": "simt"}, {"block_sizes": [8], "mode": "simt"}
    )

    def runs(config: helion.Config) -> bool:
        if config.config.get("sampled"):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return True

    provider, _compiled, _warnings = _provider(
        spec, runs, compiles=lambda c: c.config.get("sampled") is True
    )
    provider._sampled_baseline_configs = lambda count: [  # pyrefly: ignore[bad-assignment]
        helion.Config(block_sizes=[16], mode="simt", sampled=True)
    ]
    try:
        provider._compute_reference_baseline()
    except helion.exc.InvalidConfig as error:
        assert "illegal memory access" in str(error.__cause__)
    else:
        raise AssertionError("the unrecoverable error was swallowed")
