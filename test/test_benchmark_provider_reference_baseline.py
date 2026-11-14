from __future__ import annotations

import contextlib
import dataclasses
import errno
import functools
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.utils._pytree import _deregister_pytree_node
from torch.utils._pytree import register_pytree_node

import helion
from helion.autotuner import AccuracyCheck
from helion.autotuner import benchmark_provider as benchmark_provider_module
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.accuracy import assert_close
from helion.autotuner.accuracy import entirely_nan
from helion.autotuner.accuracy import same_values
from helion.autotuner.benchmark_provider import _MAX_REFERENCE_BASELINE_ATTEMPTS
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import _BaselineAttemptFailure
from helion.autotuner.handoff_evaluation import _assert_exact
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.precompile_future import PrecompileFuture
from helion.autotuner.precompile_future import RemoteError
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _run_kernel_in_subprocess_spawn
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only


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
    # check is on; the fake provider's kernels are plain Python, compiled
    # in-process only (no precompile pool, see _pool_provider).
    provider.settings = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        autotune_accuracy_check=False, autotune_precompile=None
    )
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


def test_a_baseline_holding_only_nans_is_warned_about() -> None:
    # The accuracy check counts a NaN that matches a NaN as accurate: a
    # baseline config that produces only NaNs (an overflow, say) passes every
    # candidate that does too.  A partly NaN output is the kernel's own result
    # (a sqrt of negative inputs); a custom accuracy check, or none, does not
    # match NaNs this way.
    spec = _Spec(
        {"block_sizes": [64], "mode": "simt"}, {"block_sizes": [64], "mode": "simt"}
    )

    def nan_warnings(value: object, **settings: object) -> list[str]:
        provider, _compiled, warnings = _provider(
            spec, lambda config: True, lambda config: value
        )
        provider.settings = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
            **{
                "autotune_accuracy_check": True,
                "autotune_precompile": None,
                "autotune_baseline_fn": None,
                "autotune_baseline_accuracy_check_fn": None,
                **settings,
            }
        )
        provider._compute_baseline()
        return [warning for warning in warnings if "NaN" in warning]

    (warning,) = nan_warnings(math.nan)
    assert warning.startswith("Every element of the accuracy baseline's output is NaN.")
    assert nan_warnings(torch.tensor([math.nan, 0.0, math.nan, 0.0])) == []
    assert nan_warnings(math.nan, autotune_accuracy_check=False) == []
    assert (
        nan_warnings(math.nan, autotune_baseline_accuracy_check_fn=lambda a, b: None)
        == []
    )

    def fill_with_nan(x: torch.Tensor) -> None:
        # A baseline that writes its result into an argument.
        x.fill_(math.nan)

    (warning,) = nan_warnings(0.0, autotune_baseline_fn=fill_with_nan)
    assert warning.startswith(
        "Every element of the accuracy baseline's argument 0 is NaN."
    )


def _baseline_of(kernel: Any, *args: torch.Tensor) -> tuple[list[int], list[str]]:
    """The mutated-argument indices and the warnings of the baseline run of
    ``kernel`` on ``args``, under the default accuracy check."""
    provider, _compiled, warnings = _provider(
        _Spec({"block_sizes": [64]}, {"block_sizes": [64]}), lambda config: True
    )
    provider.kernel.compile_config = (  # pyrefly: ignore[missing-attribute]
        lambda config, allow_print=False: kernel
    )
    provider.args = args
    provider.settings = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        autotune_accuracy_check=True,
        autotune_precompile=None,
        autotune_baseline_fn=None,
        autotune_baseline_accuracy_check_fn=None,
    )
    _output, mutated_arg_idxs, _post_args = provider._compute_baseline()
    return list(mutated_arg_idxs), warnings


def test_an_input_holding_nans_is_compared_but_not_warned_about() -> None:
    # torch.equal counts a NaN unequal to itself, so an input holding a NaN
    # counts as written and is compared, a NaN matching a NaN: a read-only
    # one costs a clone per run, and an output buffer the baseline rewrites
    # with the values it held is checked.  Neither holds NaNs only, so
    # neither draws the warning about NaN baselines.
    values = [math.nan, 1.0, math.nan, 2.0]

    def rewrite(out: torch.Tensor) -> None:
        out.copy_(torch.tensor(values))

    for kernel in (lambda x: torch.zeros(4), rewrite):
        mutated, warnings = _baseline_of(kernel, torch.tensor(values))
        assert mutated == [0]
        assert not [warning for warning in warnings if "NaN" in warning]


def test_an_argument_of_nans_only_is_compared_and_warned_about() -> None:
    # Values cannot tell a read-only input of NaNs from an output the
    # baseline filled with NaNs (an overflow writing over a NaN prefill): it
    # is compared, so a candidate writing zeros there fails, and the warning
    # about NaN baselines names it.
    def fill(out: torch.Tensor) -> None:
        out.fill_(math.nan)

    mutated, warnings = _baseline_of(fill, torch.full((4,), math.nan))
    assert mutated == [0]
    assert [warning for warning in warnings if "argument 0 is NaN" in warning]


def test_a_written_conjugate_view_argument_is_mutated() -> None:
    # The baseline runs on a resolved clone of the caller's conjugate view.
    def write(z: torch.Tensor) -> None:
        z.copy_(torch.tensor([3 + 4j]))

    mutated, _warnings = _baseline_of(write, torch.tensor([1 + 2j]).conj())
    assert mutated == [0]


def test_a_flattened_argument_is_named_by_its_position() -> None:
    # The mutated-argument indices count the leaves of the flattened
    # arguments, one per tensor of a list; the warning names the argument.
    provider, _compiled, _warnings = _provider(
        _Spec({"block_sizes": [64]}, {"block_sizes": [64]}), lambda config: True
    )
    provider.args = (torch.zeros(4), [torch.zeros(2), (torch.zeros(2),)], 2.0)
    assert provider._flattened_argument_names() == [
        "argument 0",
        "argument 1[0]",
        "argument 1[1][0]",
        "argument 2",
    ]


def test_a_nested_output_of_nans_only_is_named_by_its_path() -> None:
    def nested(x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return torch.zeros(2), [torch.zeros(2), torch.full((2,), math.nan)]

    _mutated, warnings = _baseline_of(nested, torch.zeros(2))
    assert [warning for warning in warnings if "baseline's output[1][1] is" in warning]


@dataclasses.dataclass
class _Keyless:
    """A pytree node type registered without keys: ``tree_flatten`` takes it,
    ``tree_flatten_with_path`` raises for it."""

    value: torch.Tensor


def test_a_pytree_node_without_keys_is_named_by_its_leaf_index() -> None:
    # The warning about NaN baselines builds names only for the leaves it
    # reports, and names a leaf of such a node by its index: a baseline
    # holding one autotunes, where the paths of every leaf crashed it.
    register_pytree_node(
        _Keyless,
        lambda node: ([node.value], None),
        lambda values, _context: _Keyless(values[0]),
    )
    try:
        _mutated, warnings = _baseline_of(
            lambda x: _Keyless(torch.zeros(2)), torch.zeros(2)
        )
        assert not [warning for warning in warnings if "NaN" in warning]
        _mutated, warnings = _baseline_of(
            lambda x: _Keyless(torch.full((2,), math.nan)), torch.zeros(2)
        )
        assert [w for w in warnings if "baseline's output leaf 0 is NaN" in w]

        def fill(node: _Keyless) -> None:
            node.value.fill_(math.nan)

        mutated, warnings = _baseline_of(fill, _Keyless(torch.zeros(2)))
        assert mutated == [0]
        assert [w for w in warnings if "baseline's argument 0 leaf 0 is NaN" in w]
    finally:
        _deregister_pytree_node(_Keyless)


def test_a_written_fp4_argument_is_mutated() -> None:
    # torch.equal has no fp4 kernel: its error counted the argument
    # unchanged, so an fp4 argument written in place went unchecked.
    def write(packed: torch.Tensor) -> None:
        packed.view(torch.uint8).add_(1)

    packed = torch.zeros(4, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    mutated, _warnings = _baseline_of(write, packed)
    assert mutated == [0]
    mutated, _warnings = _baseline_of(lambda x: torch.zeros(4), packed)
    assert mutated == []


class _Log:
    """The provider's logger, keeping the warnings."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def __call__(self, *msg: object, **kwargs: object) -> None:
        pass

    def warning(self, *msg: object, **kwargs: object) -> None:
        self.warnings.append(" ".join(map(str, msg)))

    def debug(self, *msg: object, **kwargs: object) -> None:
        pass

    def error(self, *msg: object, **kwargs: object) -> None:
        pass


def _pool_provider(
    spec: _Spec,
    *,
    pool: bool = True,
    compiled: list[int] | None = None,
    args: tuple[torch.Tensor, ...] | None = None,
    classify: Callable[[Exception], str] = lambda error: "raise",
    **settings: Any,
) -> tuple[LocalBenchmarkProvider, list[int], _Log]:
    """A provider of ``args`` (four zeros by default) built as a search
    builds it, its constructor computing the accuracy baseline, on a backend
    with a precompile pool (``pool``) whose classifier is ``classify``, by
    default raising every error: only the precompile's own back-off
    applies.  A config adds its largest
    block size to the input; returns the block sizes that ran in-process,
    and adds those compiled to ``compiled``.  A compiled config is a plain
    callable, which reaches no Triton launch a fork precompile could
    intercept."""
    ran: list[int] = []
    log = _Log()

    def compile_config(config: helion.Config, allow_print: bool = False) -> object:
        size = max(config.config["block_sizes"])
        if compiled is not None:
            compiled.append(size)

        def run(x: torch.Tensor) -> torch.Tensor:
            ran.append(size)
            return x + size

        return run

    spec.backend = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
        supports_precompile=lambda: pool,
        classify_autotune_exception=classify,
    )
    provider_settings = Settings(
        autotune_accuracy_check=False,
        autotune_log_level=logging.CRITICAL,
        **settings,
    )
    kernel = SimpleNamespace(
        compile_config=compile_config,
        settings=provider_settings,
        env=SimpleNamespace(process_group_name=None, device=torch.device("cpu")),
        format_kernel_decorator=lambda config, settings: f"@helion.kernel({config})",
        maybe_log_repro=lambda *args, **kwargs: None,
    )
    provider = LocalBenchmarkProvider(
        kernel=kernel,  # pyrefly: ignore[bad-argument-type]
        settings=provider_settings,
        config_spec=spec,  # pyrefly: ignore[bad-argument-type]
        args=(torch.zeros(4),) if args is None else args,
        log=log,  # pyrefly: ignore[bad-argument-type]
        autotune_metrics=AutotuneMetrics(),
    )
    return provider, ran, log


def _compile_forever(compilers: str) -> None:
    """A precompile stuck as Triton is on ptxas, waiting on a compiler
    process -- one that ignores SIGTERM, so only the SIGKILL of the
    precompile's whole process group ends it.  Records the compiler's pid
    in the directory ``compilers``."""
    # The fork inherits the test runner's signal handlers; a compile has none.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    compiler = subprocess.Popen(["sh", "-c", "trap '' TERM; exec sleep 60"])
    Path(compilers, str(compiler.pid)).touch()
    compiler.wait()


def _exited(pid: int, timeout: float = 10.0) -> bool:
    """Whether process ``pid`` is gone, or a zombie awaiting its reaper,
    within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            # Gone before the open, or between the open and the read.
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def _skip_precompile(**kwargs: Any) -> PrecompileFuture:
    """``PrecompileFuture.create`` for a precompile that finished at once."""
    return PrecompileFuture.skip(kwargs["ctx"], kwargs["config"], True)


def test_a_baseline_compile_past_the_deadline_backs_off() -> None:
    # The observed hang: ptxas ran for two hours on the default config of an
    # hl.rand kernel (block size 32) in the baseline's in-process compile,
    # which no deadline reached.  Precompiled under the candidates' deadline
    # instead, the config is killed along with the compiler it started and
    # never compiles or runs in-process; the reference backs off to block
    # size 16, which becomes the baseline, and the search can start.
    precompiled: list[int] = []
    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with tempfile.TemporaryDirectory() as compilers:

        def precompiler(*args: object, **kwargs: object) -> Callable[[], object]:
            config = args[2]
            assert isinstance(config, helion.Config)
            size = max(config.config["block_sizes"])
            precompiled.append(size)
            if size >= 32:
                return functools.partial(_compile_forever, compilers)
            return lambda: True

        with patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=precompiler,
        ):
            provider, ran, log = _pool_provider(
                spec,
                compiled=compiled,
                autotune_precompile="fork",
                autotune_compile_timeout=2,
            )
        try:
            assert precompiled == [32, 16]
            # Each attempt compiles once: the run reuses its precompile's
            # compiled config.
            assert compiled == [32, 16]
            assert ran == [16]
            assert torch.equal(provider._baseline_output, torch.full((4,), 16.0))
            # The compiler the killed precompile started is gone too (on a
            # loaded host the deadline may come before the fork started it).
            pids = [int(name) for name in os.listdir(compilers)]
            assert len(pids) <= 1 and all(_exited(pid) for pid in pids)
        finally:
            provider.cleanup()
    (backoff,) = [w for w in log.warnings if "retrying the baseline" in w]
    assert "block_sizes=[32]" in backoff
    assert re.search(
        r"compile exceeded autotune_compile_timeout=2s "
        r"\(HELION_AUTOTUNE_COMPILE_TIMEOUT\) in the precompile subprocess, "
        r"killed after \d+\.\ds\); retrying the baseline with block sizes \[16\]$",
        backoff,
    ), backoff
    assert log.warnings[-1].startswith("Using block sizes [16] for the accuracy")


def _die_from_a_signal() -> None:
    """A fork precompile's compile that kills the child, as an LLVM or MLIR
    abort or a segfault in the backend compiler would."""
    os.kill(os.getpid(), signal.SIGKILL)


def test_a_baseline_precompile_that_dies_from_a_signal_backs_off() -> None:
    # A child that dies from a signal before writing a result crashed in
    # the backend compiler, which an in-process compile would repeat in
    # this process, ending the autotune: the attempt fails instead, and the
    # reference backs off to block size 16 as past the deadline.  The
    # crashed config never compiles or runs in-process again.
    precompiled: list[int] = []
    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )

    def precompiler(*args: object, **kwargs: object) -> Callable[[], object]:
        config = args[2]
        assert isinstance(config, helion.Config)
        size = max(config.config["block_sizes"])
        precompiled.append(size)
        return _die_from_a_signal if size >= 32 else lambda: True

    with patch(
        "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
        side_effect=precompiler,
    ):
        provider, ran, log = _pool_provider(
            spec, compiled=compiled, autotune_precompile="fork"
        )
    try:
        assert precompiled == [32, 16]
        assert compiled == [32, 16] and ran == [16]
        assert torch.equal(provider._baseline_output, torch.full((4,), 16.0))
    finally:
        provider.cleanup()
    (backoff,) = [w for w in log.warnings if "retrying the baseline" in w]
    # The signal's description is the platform's: "Killed", or "Killed: 9".
    killed = signal.strsignal(signal.SIGKILL)
    assert (
        f"(the precompile subprocess died from signal 9 ({killed}) before "
        "reporting a result); retrying the baseline with block sizes [16]"
    ) in backoff, backoff


def test_every_rank_backs_off_when_a_peer_rank_hits_the_deadline() -> None:
    # Ranks agree on the deadline as they do on a candidate's precompile: a
    # rank whose precompile finished still backs off when a peer's was
    # killed, so no rank runs a config its peers gave up.  The execution
    # default is killed on the peer and the reference takes over.  Each
    # attempt has two collectives: on its code generation, then on its
    # precompile.
    gathered: list[object] = []
    killed = (
        "compile exceeded autotune_compile_timeout=60s "
        "(HELION_AUTOTUNE_COMPILE_TIMEOUT) in the precompile subprocess, "
        "killed after 61.0s"
    )

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        peer = _BaselineAttemptFailure(killed, back_off=True, timed_out=True)
        return [obj, peer if len(gathered) == 2 else None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=_skip_precompile),
    ):
        provider, ran, log = _pool_provider(
            spec, autotune_precompile="fork", autotune_compile_timeout=60
        )
    try:
        assert gathered == [None] * 4
        assert ran == [32]
        assert torch.equal(provider._baseline_output, torch.full((4,), 32.0))
    finally:
        provider.cleanup()
    (fallback,) = log.warnings
    assert f"(on a peer rank, {killed})" in fallback
    assert fallback.endswith("retrying with the autotuning reference config")


def test_every_rank_fails_an_attempt_a_peer_rank_failed_before_its_run() -> None:
    # A rank-local failure ahead of the run -- an out-of-memory error on the
    # precompile's copy of the arguments, say -- fails the attempt on every
    # rank, as a killed precompile does: a rank that went on to clone the
    # arguments for the run (a rendezvous for symmetric memory) would wait
    # for the peer that gave the attempt up.  A failure a smaller config may
    # avoid backs every rank off; the execution default here, which falls
    # back to the reference.
    gathered: list[object] = []
    oom = _BaselineAttemptFailure(
        "OutOfMemoryError: CUDA out of memory",
        back_off=True,
        timed_out=False,
    )

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        # The execution default's precompile.
        return [obj, oom if len(gathered) == 2 else None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=_skip_precompile),
    ):
        provider, ran, log = _pool_provider(spec, autotune_precompile="fork")
    try:
        assert gathered == [None] * 4
        assert ran == [32]
    finally:
        provider.cleanup()
    (fallback,) = log.warnings
    assert "(on a peer rank, OutOfMemoryError: CUDA out of memory)" in fallback


def test_every_rank_stops_on_a_failure_a_peer_rank_cannot_back_off() -> None:
    # A peer's failure no config avoids (a bug) stops this rank's search too,
    # although its own precompile finished.
    peer = _BaselineAttemptFailure(
        "RuntimeError: a bug in the backend",
        back_off=False,
        timed_out=False,
    )
    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(
            benchmark_provider_module,
            "all_gather_object",
            lambda obj, process_group_name=None: [obj, peer],
        ),
        patch.object(PrecompileFuture, "create", side_effect=_skip_precompile),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(spec, compiled=compiled, autotune_precompile="fork")
    assert (
        str(raised.value.__cause__)
        == "on a peer rank, RuntimeError: a bug in the backend"
    )
    assert compiled == [32]


def test_a_failed_baseline_removes_its_precompile_directory() -> None:
    # The baseline's attempts make the precompile directory (for their result
    # files); a provider whose construction raised has no caller to clean it
    # up, and the directory stayed until garbage collection (a
    # ResourceWarning).
    peer = _BaselineAttemptFailure(
        "RuntimeError: a bug in the backend", back_off=False, timed_out=False
    )
    made: list[tempfile.TemporaryDirectory[str]] = []
    make = tempfile.TemporaryDirectory

    def record(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory[str]:
        made.append(make(*args, **kwargs))
        return made[-1]

    gathered: list[object] = []

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        # The peer fails the reference after its precompile.
        return [obj, peer if len(gathered) == 2 else None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=_skip_precompile),
        patch.object(tempfile, "TemporaryDirectory", side_effect=record),
        pytest.raises(helion.exc.InvalidConfig),
    ):
        _pool_provider(spec, autotune_precompile="fork")
    assert made
    assert not [d.name for d in made if os.path.exists(d.name)]


def test_a_rank_shares_its_failed_baseline_precompile_with_its_peers() -> None:
    # This rank's precompile raises (an out-of-memory error, say): the rank
    # tells its peers how the attempt failed, so they fail it alike.
    gathered: list[object] = []
    creates = iter([RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")])

    def create(**kwargs: Any) -> PrecompileFuture:
        error = next(creates, None)
        if error is not None:
            raise error
        return _skip_precompile(**kwargs)

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        return [obj, None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=create),
    ):
        provider, ran, _log = _pool_provider(spec, autotune_precompile="fork")
    try:
        assert gathered == [
            None,
            _BaselineAttemptFailure(
                "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
                # The provider's classifier raises every error.
                back_off=False,
                timed_out=False,
            ),
            None,
            None,
        ]
        assert ran == [32]
    finally:
        provider.cleanup()


def test_a_peer_ranks_failed_code_generation_keeps_this_rank_from_precompiling() -> (
    None
):
    # The ranks agree on an attempt's code generation before any of them
    # precompiles: the precompile's copy of the arguments rendezvouses
    # symmetric-memory tensors, a collective a peer whose code generation
    # failed would not join.  The execution default fails on the peer here:
    # this rank neither precompiles it nor copies its arguments for it, and
    # falls back to the reference with the peer.
    gathered: list[object] = []
    peer = _BaselineAttemptFailure(
        "InvalidConfig: too many registers",
        back_off=True,
        timed_out=False,
    )

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        return [obj, peer if len(gathered) == 1 else None]

    create = Mock(side_effect=_skip_precompile)
    reference = {"block_sizes": [32], "mode": "simt"}
    spec = _Spec(reference, {"block_sizes": [32], "mode": "mma"})
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", create),
        patch.object(
            benchmark_provider_module,
            "_clone_args",
            wraps=benchmark_provider_module._clone_args,
        ) as clone_args,
    ):
        provider, ran, log = _pool_provider(spec, autotune_precompile="fork")
    try:
        (precompiled,) = create.call_args_list
        assert precompiled.kwargs["config"] == helion.Config(**reference)
        # The reference's precompile copy and run clone, then the baseline's
        # copy of the arguments its run left.
        assert clone_args.call_count == 3
        assert gathered == [None] * 3 and ran == [32]
    finally:
        provider.cleanup()
    (fallback,) = log.warnings
    assert "(on a peer rank, InvalidConfig: too many registers)" in fallback


@pytest.mark.parametrize(
    ("here", "on_the_peer"),
    [(True, False), (False, True), (True, True)],
    ids=["here", "on-the-peer", "on-every-rank"],
)
def test_a_compile_error_a_precompile_reports_on_some_ranks_backs_them_all_off(
    here: bool, on_the_peer: bool
) -> None:
    # A precompile that reports a compile error leaves the attempt to its
    # run, which repeats the compile in-process and reports the error, as
    # without a precompile -- where every rank's precompile reports one.
    # Where a peer's finished, the rank whose compile failed would give the
    # run up while its peers launch it and wait for it: every rank backs
    # off, here the execution default to the reference.
    error = RuntimeError("out of resources: shared memory")
    gathered: list[object] = []
    reported = _BaselineAttemptFailure(
        f"RuntimeError: {error}",
        # The provider's classifier raises every error.
        back_off=False,
        timed_out=False,
        reported=True,
    )
    creates = iter([error if here else None])

    def create(**kwargs: Any) -> PrecompileFuture:
        failed = next(creates, None)
        if failed is None:
            return _skip_precompile(**kwargs)
        return PrecompileFuture.skip(
            kwargs["ctx"], kwargs["config"], False, prepare_error=failed
        )

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        # The execution default's precompile.
        return [obj, reported if on_the_peer and len(gathered) == 2 else None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "mma"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=create),
    ):
        provider, ran, log = _pool_provider(spec, autotune_precompile="fork")
    try:
        assert gathered[1] == (reported if here else None)
        if here and on_the_peer:
            # The default ran, then the reference, which agrees with it.
            assert ran == [32, 32] and log.warnings == []
            return
        assert ran == [32]
    finally:
        provider.cleanup()
    (fallback,) = log.warnings
    assert f"({'on a peer rank, ' * on_the_peer}RuntimeError: {error})" in fallback
    assert fallback.endswith("retrying with the autotuning reference config")


@pytest.mark.parametrize(
    ("classification", "back_off"), [("debug", True), ("raise", False)]
)
def test_a_precompile_childs_error_backs_off_as_the_child_classified_it(
    classification: str, back_off: bool
) -> None:
    # The child's error arrives as a stand-in of its class, which no type
    # rule matches.  Classified here, an OutOfResources error whose message
    # the expected-error pattern misses reads as a bug, so a rank reporting
    # it and a rank with a real bug count as alike, and the attempt goes to
    # their runs, which decide apart (one backs off, the other stops).
    error = RemoteError(
        exc_type="OutOfResources",
        exc_module="triton.runtime.errors",
        message="out of resource: registers, Required: 70000, Hardware limit: 65536",
        traceback=None,
        classification=classification,
    ).to_exception()
    provider = SimpleNamespace(
        config_spec=SimpleNamespace(
            backend=SimpleNamespace(classify_autotune_exception=lambda error: None)
        )
    )
    assert (
        LocalBenchmarkProvider._can_back_off_baseline(provider, error)  # pyrefly: ignore[bad-argument-type]
        is back_off
    )


@pytest.mark.parametrize("severe_here", [True, False], ids=["here", "on-the-peer"])
def test_compile_errors_the_ranks_classify_apart_stop_every_rank(
    severe_here: bool,
) -> None:
    # Every rank's precompile reported a compile error, but the ranks would
    # handle theirs apart: one a smaller config may avoid (out of resources),
    # the other a bug.  Left to its run, each rank would decide alone -- one
    # backing off into the next attempt's collectives, the other stopping --
    # and the search would hang: every rank stops, here at the reference.
    bug = RuntimeError("a bug in the backend")
    resources = RuntimeError("out of resources: shared memory")
    here, peer_error = (bug, resources) if severe_here else (resources, bug)
    peer = _BaselineAttemptFailure(
        f"RuntimeError: {peer_error}",
        back_off=severe_here,
        timed_out=False,
        reported=True,
    )
    creates = iter([here])

    def create(**kwargs: Any) -> PrecompileFuture:
        failed = next(creates, None)
        if failed is None:
            return _skip_precompile(**kwargs)
        return PrecompileFuture.skip(
            kwargs["ctx"], kwargs["config"], False, prepare_error=failed
        )

    gathered: list[object] = []

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        # The reference's precompile.
        return [obj, peer if len(gathered) == 2 else None]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=create),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(
            spec,
            classify=lambda error: "raise" if error is bug else "warn",
            autotune_precompile="fork",
        )
    assert gathered[1] == peer._replace(
        reason=f"RuntimeError: {here}", back_off=not severe_here
    )
    assert str(raised.value.__cause__) == (
        "RuntimeError: a bug in the backend"
        if severe_here
        else "on a peer rank, RuntimeError: a bug in the backend"
    )


def _timed_out_precompile(**kwargs: Any) -> PrecompileFuture:
    """``PrecompileFuture.create`` for a precompile killed at a 60 s deadline,
    without the wait."""
    future = PrecompileFuture.skip(kwargs["ctx"], kwargs["config"], False)
    future.timeout = 60
    future.failure_reason = "timeout"
    return future


def test_a_reference_past_the_deadline_at_every_block_size_tries_no_samples() -> None:
    # Every block size of the reference compiled past the deadline: a
    # sampled config is no likelier to compile in time, and each would wait
    # out the deadline again, so the search stops at once, its error naming
    # the setting to raise.
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    compiled: list[int] = []
    with (
        patch.object(PrecompileFuture, "create", side_effect=_timed_out_precompile),
        patch.object(
            LocalBenchmarkProvider,
            "_sampled_baseline_configs",
            side_effect=AssertionError("sampled a baseline config"),
        ),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(
            spec,
            compiled=compiled,
            autotune_precompile="fork",
            autotune_compile_timeout=60,
        )
    assert compiled == [32, 16, 8, 4]
    assert (
        "A compile past autotune_compile_timeout=60s was killed: set "
        "HELION_AUTOTUNE_COMPILE_TIMEOUT higher if the kernel's compile needs longer."
    ) in str(raised.value)
    assert "compile exceeded autotune_compile_timeout=60s" in str(
        raised.value.__cause__
    )


@pytest.mark.parametrize(
    ("precompile", "pool"),
    [(None, True), ("fork", False)],
    ids=["precompile-off", "backend-without-a-pool"],
)
def test_without_a_precompile_pool_the_baseline_compiles_in_process(
    precompile: str | None, pool: bool
) -> None:
    # Precompilation off, or a backend without a precompile pool (CuTe: its
    # DSL compiles at the first call, which candidates make in the benchmark
    # worker under its compile deadline): the baseline compiles and runs in
    # this process as before, without a subprocess or a deadline.
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    create = Mock(side_effect=_skip_precompile)
    with patch.object(PrecompileFuture, "create", create):
        provider, ran, log = _pool_provider(
            spec, pool=pool, autotune_precompile=precompile
        )
    try:
        create.assert_not_called()
        assert ran == [32] and not log.warnings
    finally:
        provider.cleanup()


def test_a_spawned_baseline_precompile_loads_the_arguments_as_cloned() -> None:
    # Triton specializes on a pointer's alignment.  The attempt runs on a
    # clone of the arguments, which gives a lone contiguous view such as
    # x[1:] an allocation of its own, aligned, while the candidates'
    # arguments file keeps the view's offset.  The spawned precompile loads
    # a file of the clone: on the candidates' file the child would compile
    # another specialization than the run's, which the run would then
    # compile in-process without the deadline.  The file is written once for
    # all attempts -- the promoted default's and the reference's here -- and
    # removed with the baseline computed (a full copy of the arguments, in
    # host memory on a tmpfs), and setup() writes the candidates' file as
    # before.
    loaded: list[tuple[str, torch.Tensor]] = []

    def create(**kwargs: Any) -> PrecompileFuture:
        path = kwargs["args_path"]
        loaded.append((path, *torch.load(path)))
        return _skip_precompile(**kwargs)

    view = torch.arange(5.0)[1:]
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "mma"}
    )
    with (
        patch.object(PrecompileFuture, "create", side_effect=create),
        patch.object(
            LocalBenchmarkProvider,
            "_save_kernel_args",
            autospec=True,
            side_effect=LocalBenchmarkProvider._save_kernel_args,
        ) as save_kernel_args,
    ):
        provider, ran, _log = _pool_provider(
            spec,
            args=(view,),
            autotune_precompile="spawn",
            autotune_benchmark_subprocess=False,
        )
    try:
        assert ran == [32, 32]
        (path, clone), (again, _) = loaded
        assert again == path and save_kernel_args.call_count == 1
        assert torch.equal(clone, view) and clone.storage_offset() == 0
        assert clone.untyped_storage().nbytes() == view.nbytes
        assert not os.path.exists(path)
        assert provider._baseline_precompile_args is None
        provider.setup()
        assert provider._precompile_args_path is not None
        (candidates,) = torch.load(provider._precompile_args_path)
        assert torch.equal(candidates, view) and candidates.storage_offset() == 1
    finally:
        provider.cleanup()


def test_a_baseline_arguments_file_that_failed_is_not_tried_again() -> None:
    # Copying the arguments for the file rendezvouses symmetric-memory
    # tensors: a rank whose copy or write raised (a full disk, say) must not
    # try again on a later attempt, where its peers, which wrote theirs,
    # would not join the copy's rendezvous.  The rank's spawned precompiles
    # are skipped instead.
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    provider, _ran, _log = _pool_provider(spec)
    try:
        with (
            patch.object(
                provider,
                "_save_kernel_args",
                side_effect=OSError(errno.ENOSPC, "No space left on device"),
            ) as save_kernel_args,
            patch.object(
                benchmark_provider_module,
                "_clone_args",
                wraps=benchmark_provider_module._clone_args,
            ) as clone_args,
        ):
            with pytest.raises(OSError, match="No space left on device"):
                provider._baseline_precompile_args_path()
            assert provider._baseline_precompile_args_path() is None
            assert save_kernel_args.call_count == clone_args.call_count == 1
    finally:
        provider.cleanup()


def test_a_spawned_baseline_precompile_without_its_arguments_file_is_skipped() -> None:
    # Arguments the file cannot hold leave the baseline's spawned precompile
    # out, and the attempt compiles in-process, as before the baseline had
    # a precompile.  The provider remembers the failed write without the
    # error, whose traceback would keep a copy of the arguments alive, and
    # does not try it again.
    create = Mock(side_effect=_skip_precompile)
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(PrecompileFuture, "create", create),
        patch.object(
            LocalBenchmarkProvider,
            "_save_kernel_args",
            return_value=TypeError("cannot pickle 'generator' object"),
        ) as save_kernel_args,
    ):
        provider, ran, _log = _pool_provider(
            spec, autotune_precompile="spawn", autotune_benchmark_subprocess=False
        )
        try:
            create.assert_not_called()
            assert ran == [32]
            assert provider._baseline_precompile_args is False
            assert provider._baseline_precompile_args_path() is None
            save_kernel_args.assert_called_once()
        finally:
            provider.cleanup()


def test_an_unrecoverable_error_from_the_baseline_precompile_stops_the_search() -> None:
    # A fork precompile runs the host code to find the launch, as a
    # candidate's does, and raises an unrecoverable runtime error from it:
    # the device may be in a bad state.  The attempt fails with it at once --
    # before the ranks' collective on the precompile (the one on code
    # generation came first), and with no argument clone of its own or run
    # in-process -- and the search stops with the error naming the config it
    # failed, as for an attempt that ran.
    error = RuntimeError("CUDA error: an illegal memory access was encountered")
    gathered: list[object] = []
    compiled: list[int] = []

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        return [obj]

    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(
            benchmark_provider_module,
            "_clone_args",
            wraps=benchmark_provider_module._clone_args,
        ) as clone_args,
        patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=error,
        ),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(spec, compiled=compiled, autotune_precompile="fork")
    assert raised.value.__cause__ is error
    assert compiled == [32] and gathered == [None]
    # The one copy is the precompile's, for the host code it ran.
    assert clone_args.call_count == 1


def test_an_unrecoverable_error_from_a_spawned_baseline_precompile_stops_the_search() -> (
    None
):
    # A spawned precompile runs the kernel and synchronizes, so its child can
    # report an illegal memory access.  It happened on the child's device
    # context, not this process's, so the rank shares it with its peers (here
    # one whose precompile finished): the attempt fails on every rank and
    # every rank stops, where a rank that stopped alone would leave its peers
    # waiting in the collective.  No rank clones the arguments for a run in
    # its process, which would repeat the error there.  The child is the
    # spawned one's, started by a fork to save the interpreter's start.
    fn_spec = SerializedCompiledFunction(
        function_name="call",
        source_code=(
            "def call(*args):\n"
            "    raise RuntimeError("
            "'CUDA error: an illegal memory access was encountered')\n"
        ),
        filename="<spawned baseline precompile>",
        module_name=None,
    )

    def create(**kwargs: Any) -> PrecompileFuture:
        process = torch.multiprocessing.get_context("fork").Process(
            target=_run_kernel_in_subprocess_spawn,
            args=(fn_spec, kwargs["args_path"], kwargs["result_path"], "@test"),
        )
        return PrecompileFuture(
            ctx=kwargs["ctx"],
            config=kwargs["config"],
            process=process,
            timeout=30,
            result_path=kwargs["result_path"],
        )

    gathered: list[object] = []

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        return [obj, None]

    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        patch.object(PrecompileFuture, "create", side_effect=create),
        patch("helion.autotuner.precompile_future.synchronize_device"),
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(
            benchmark_provider_module,
            "_clone_args",
            wraps=benchmark_provider_module._clone_args,
        ) as clone_args,
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(
            spec,
            compiled=compiled,
            autotune_precompile="spawn",
            autotune_benchmark_subprocess=False,
        )
    assert "illegal memory access" in str(raised.value.__cause__)
    assert compiled == [32]
    # The one copy is the spawned precompile's arguments file.
    assert clone_args.call_count == 1
    # Code generation went well; the precompile's failure is shared.
    no_failure, shared = gathered
    assert no_failure is None
    assert isinstance(shared, _BaselineAttemptFailure)
    assert not shared.back_off and "illegal memory access" in shared.reason


def test_every_rank_stops_on_an_unrecoverable_error_a_peers_precompile_reports() -> (
    None
):
    # The peer side of the test above: a peer's spawned precompile reported
    # an illegal memory access for the execution default, whose own
    # precompile finished here.  This rank stops too -- it does not fall back
    # to the reference config, which would leave the peer alone -- without
    # cloning the arguments for a run.
    peer = _BaselineAttemptFailure(
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        back_off=False,
    )
    gathered: list[object] = []

    def all_gather_object(
        obj: object, process_group_name: str | None = None
    ) -> list[object]:
        gathered.append(obj)
        # The execution default's precompile.
        return [obj, peer if len(gathered) == 2 else None]

    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [64], "mode": "mma"}
    )
    with (
        patch.object(benchmark_provider_module, "all_gather_object", all_gather_object),
        patch.object(PrecompileFuture, "create", side_effect=_skip_precompile),
        patch.object(
            benchmark_provider_module,
            "_clone_args",
            wraps=benchmark_provider_module._clone_args,
        ) as clone_args,
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Execution default failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(spec, compiled=compiled, autotune_precompile="fork")
    assert str(raised.value.__cause__) == f"on a peer rank, {peer.reason}"
    assert compiled == [64]
    assert gathered == [None, None]
    # The fork precompile's copy of the arguments, and no run's.
    assert clone_args.call_count == 1


def _child_compile_error(*args: object, **kwargs: object) -> Callable[[], None]:
    """``_prepare_precompiler_for_fork`` for a config whose compile fails in
    the fork child with an error the classifier does not expect."""

    def compile_in_child() -> None:
        raise RuntimeError("unsupported op")

    return compile_in_child


@pytest.mark.parametrize(
    ("precompile", "prepare"),
    [
        # The compiled config reaches no Triton launch to intercept, as the
        # stand-ins of the promoted-default tests in test_autotuner_accuracy.
        ("fork", None),
        ("fork", _child_compile_error),
        # A nested function cannot be serialized for a spawned precompile.
        ("spawn", None),
    ],
    ids=["no-launch-to-intercept", "compile-error-in-the-child", "cannot-serialize"],
)
def test_a_failed_baseline_precompile_reports_nothing(
    precompile: str, prepare: Callable[..., Callable[[], None]] | None
) -> None:
    # The attempt's run that follows reports a failure the precompile
    # shares, so the precompile reports nothing: it only tells whether the
    # compile finished, outran the deadline or crashed.  Reporting it there
    # too would warn, with a traceback, about a callable that reaches no
    # launch, and report a compile error twice.
    compiled: list[int] = []
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        contextlib.nullcontext()
        if prepare is None
        else patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=prepare,
        )
    ):
        provider, ran, log = _pool_provider(
            spec,
            compiled=compiled,
            autotune_precompile=precompile,
            autotune_benchmark_subprocess=False,
        )
    try:
        assert compiled == [32] and ran == [32]
        assert log.warnings == []
    finally:
        provider.cleanup()


@contextlib.contextmanager
def _baseline_precompile_start_raises(error: Exception) -> Iterator[None]:
    """A fork precompile of the baseline whose process start raises
    ``error``."""
    with (
        patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=lambda *args, **kwargs: lambda: None,
        ),
        patch.object(PrecompileFuture, "start", side_effect=error),
    ):
        yield


def test_a_baseline_precompile_the_system_cannot_start_compiles_in_process() -> None:
    # A fork the system refuses (no memory or process slots for it) leaves
    # the attempt to compile in-process without the deadline.
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with _baseline_precompile_start_raises(
        OSError(errno.EAGAIN, "Resource temporarily unavailable")
    ):
        provider, ran, log = _pool_provider(spec, autotune_precompile="fork")
    try:
        assert ran == [32] and log.warnings == []
    finally:
        provider.cleanup()


def test_another_error_of_the_baseline_precompile_fails_the_attempt() -> None:
    # Any other error from the precompile path is the attempt's failure,
    # rather than turning the deadline off unnoticed.
    error = RuntimeError("a bug in the precompile path")
    spec = _Spec(
        {"block_sizes": [32], "mode": "simt"}, {"block_sizes": [32], "mode": "simt"}
    )
    with (
        _baseline_precompile_start_raises(error),
        pytest.raises(
            helion.exc.InvalidConfig,
            match="Autotuning reference config failed while computing baseline",
        ) as raised,
    ):
        _pool_provider(spec, autotune_precompile="fork")
    assert raised.value.__cause__ is error


class TestNanAccuracy(unittest.TestCase):
    """``accuracy.assert_close`` on outputs that legitimately contain NaNs (CPU)."""

    @staticmethod
    def _values(dtype: torch.dtype) -> torch.Tensor:
        # A sqrt of half-negative inputs, as an equation-of-state kernel
        # produces on random data: NaN in about half of the outputs.
        generator = torch.Generator().manual_seed(0)
        return torch.randn(4096, generator=generator).sqrt().to(dtype)

    def test_matching_nans_pass_and_one_sided_nans_fail(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            expected = self._values(dtype)
            self.assertTrue(bool(torch.isnan(expected).any()))
            actual = expected.clone()
            assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            # The chunked comparison large outputs take.
            _chunked_assert_close(actual, expected, atol=1e-2, rtol=1e-2, chunk_size=64)
            finite = int(torch.isfinite(expected).nonzero()[0])
            actual[finite] = float("nan")
            with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
                assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
                _chunked_assert_close(
                    actual, expected, atol=1e-2, rtol=1e-2, chunk_size=64
                )
            # An AccuracyCheck compares as the default check does, under each
            # of its scalings (the rms and the peak skip the NaNs).
            for scale in (None, "rms", "peak"):
                check = AccuracyCheck(atol=1e-2, rtol=1e-2, atol_scale=scale)
                check(expected.clone(), expected)
                with self.assertRaisesRegex(
                    AssertionError, "Tensor-likes are not close"
                ):
                    check(actual, expected)

    def test_matching_scalar_nans_pass_and_one_sided_ones_fail(self) -> None:
        # Scalar leaves, such as the ConstExpr floats a handoff bundle stores
        # as plain floats, which handoff_evaluation._check_result compares.
        values = torch.arange(4.0)
        assert_close(math.nan, math.nan, atol=1e-2, rtol=1e-2)
        assert_close(
            (values.clone(), math.nan), (values, math.nan), atol=1e-2, rtol=1e-2
        )
        for actual, expected in ((math.nan, 1.0), (1.0, math.nan)):
            with self.assertRaisesRegex(AssertionError, "Scalars are not close"):
                assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            with self.assertRaisesRegex(AssertionError, "Scalars are not close"):
                assert_close(
                    (values.clone(), actual), (values, expected), atol=1e-2, rtol=1e-2
                )

    def test_a_nan_in_one_part_of_a_complex_value_hides_no_error(self) -> None:
        # torch.testing matches two complex values whenever either part of
        # each is a NaN: NaN+1j against NaN+999j, or against 1+NaNj.  The
        # parts compare as numbers of their own, as tensors and as scalars.
        nan = math.nan
        assert_close(
            torch.tensor([complex(nan, 1.0)]),
            torch.tensor([complex(nan, 1.0)]),
            atol=0.0,
            rtol=0.0,
        )
        assert_close(complex(nan, 1.0), complex(nan, 1.0), atol=0.0, rtol=0.0)
        for other in (complex(nan, 999.0), complex(1.0, nan)):
            with self.assertRaisesRegex(AssertionError, "differ in a part"):
                assert_close(
                    torch.tensor([complex(nan, 1.0)]),
                    torch.tensor([other]),
                    atol=0.0,
                    rtol=0.0,
                )
            with self.assertRaisesRegex(AssertionError, "are not equal"):
                assert_close(complex(nan, 1.0), other, atol=0.0, rtol=0.0)
        # The error names the first such value.
        with self.assertRaisesRegex(
            AssertionError,
            r"first at flat index 1: actual \(nan\+1j\), expected \(nan\+999j\)",
        ):
            assert_close(
                torch.tensor([1 + 1j, complex(nan, 1.0)]),
                torch.tensor([1 + 1j, complex(nan, 999.0)]),
                atol=0.0,
                rtol=0.0,
            )
        # Leaves on two devices fail as a mismatch, not with the RuntimeError
        # of a mask built across them.
        with self.assertRaisesRegex(AssertionError, "device"):
            assert_close(
                torch.tensor([complex(nan, 1.0)]),
                torch.tensor([complex(nan, 1.0)], device=torch.device("meta")),
                atol=0.0,
                rtol=0.0,
            )
        conjugate = torch.tensor([complex(nan, 1.0)]).conj()
        assert_close(conjugate, conjugate.resolve_conj(), atol=0.0, rtol=0.0)
        # The handoff's exact check of the inputs the reference leaves
        # unchanged compares the parts the same way.
        _assert_exact(
            torch.tensor([complex(nan, 1.0)]), torch.tensor([complex(nan, 1.0)])
        )
        for other in (complex(nan, 999.0), complex(1.0, nan)):
            with self.assertRaisesRegex(AssertionError, "are not equal"):
                _assert_exact(torch.tensor([complex(nan, 1.0)]), torch.tensor([other]))

    def test_a_complex_leaf_keeps_its_dtype_shape_and_tolerances(self) -> None:
        # Complex leaves stay complex: the real tensor of a reference's parts
        # does not pass for it, rtol applies to the magnitude, and the atol
        # floor is not scaled by the reference's RMS (only floating-point
        # leaves' is).
        for scaled in (False, True):
            with self.assertRaisesRegex(AssertionError, "shape mismatch"):
                assert_close(
                    torch.tensor([[1.0, 2.0]]),
                    torch.tensor([1 + 2j]),
                    atol=0.01,
                    rtol=0.01,
                    scale_atol_by_expected_rms=scaled,
                )
        assert_close(
            torch.tensor([1000 + 0.5j]),
            torch.tensor([1000 + 0.001j]),
            atol=0.01,
            rtol=0.01,
        )
        expected = torch.full((8,), 100 + 100j)
        with self.assertRaisesRegex(AssertionError, "are not close"):
            assert_close(
                expected + 0.5,
                expected,
                atol=0.01,
                rtol=0.0,
                scale_atol_by_expected_rms=True,
            )

    def test_entirely_nan_reads_a_number_first_without_a_copy(self) -> None:
        # An expanded row flattens to a copy of its logical size.
        row = torch.randn(1, 4096).expand(4096, 4096)
        with patch.object(torch.Tensor, "reshape", side_effect=AssertionError):
            self.assertFalse(entirely_nan(row))
        self.assertTrue(entirely_nan(torch.full((1, 4), math.nan).expand(4, 4)))

    def test_same_values_compares_fp4_by_its_bytes(self) -> None:
        # torch.equal has no fp4 kernel, which the handoff bundle's check of
        # unchanged inputs raised on.
        packed = torch.arange(4, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        self.assertTrue(same_values(packed.clone(), packed))
        changed = packed.view(torch.uint8).add(1).view(torch.float4_e2m1fn_x2)
        self.assertFalse(same_values(changed, packed))

    def test_entirely_nan(self) -> None:
        # The accuracy baseline is warned about when a tensor of it holds NaNs
        # only (see LocalBenchmarkProvider._warn_about_nan_baseline).
        nans = torch.full((10,), math.nan)
        self.assertTrue(entirely_nan(nans, chunk_size=3))
        nans[-1] = 0.0
        self.assertFalse(entirely_nan(nans, chunk_size=3))
        self.assertFalse(entirely_nan(torch.empty(0)))
        self.assertFalse(entirely_nan(torch.zeros(4, dtype=torch.int32)))
        strided = torch.full((8, 6), math.nan).t()
        self.assertTrue(entirely_nan(strided, chunk_size=16))
        strided[5, 7] = 0.0
        self.assertFalse(entirely_nan(strided, chunk_size=16))

    def test_same_values_matches_a_nan_to_a_nan(self) -> None:
        # The handoff bundle's check of the inputs a kernel left unchanged:
        # torch.equal counts a NaN unequal to itself.
        for dtype in (torch.float32, torch.bfloat16, torch.complex64):
            values = torch.tensor([1.0, math.nan, -0.0], dtype=torch.float32).to(dtype)
            self.assertTrue(same_values(values.clone(), values))
            changed = values.clone()
            changed[0] = 2.0
            self.assertFalse(same_values(changed, values))
            moved = values.clone()
            moved[1], moved[2] = 0.0, math.nan
            self.assertFalse(same_values(moved, values))
        # Once a NaN makes torch.equal fail, the bytes decide.
        signed = torch.tensor([math.nan, 0.0])
        self.assertFalse(same_values(signed * torch.tensor([1.0, -1.0]), signed))
        nans = torch.full((3,), math.nan).to(torch.float8_e4m3fn)
        self.assertTrue(same_values(nans.clone(), nans))
        self.assertTrue(same_values(torch.arange(3), torch.arange(3)))
        self.assertFalse(same_values(torch.arange(3), torch.arange(1, 4)))

    def test_same_values_resolves_conjugate_and_negative_views(self) -> None:
        # The baseline runs on resolved clones of the caller's arguments, and
        # a conjugate view's bytes cannot be read unresolved (view_as_real
        # raises).
        values = torch.tensor([complex(math.nan, 2.0), 3 - 4j])
        conjugate = values.conj()
        self.assertTrue(same_values(conjugate.resolve_conj(), conjugate))
        self.assertFalse(same_values(torch.zeros_like(values), conjugate))
        negative = conjugate.imag
        self.assertTrue(same_values(negative.resolve_neg(), negative))
        self.assertFalse(same_values(negative.resolve_neg() + 1, negative))
