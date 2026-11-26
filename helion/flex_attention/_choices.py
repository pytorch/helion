"""Inductor's flex_attention backend hooks, answered with the Helion kernels.

Inductor offers each compiled flex_attention forward and backward to the
registered ``InductorChoices`` contributors before it lowers the call itself
(``lower_flex_attention`` / ``lower_flex_attention_backward``).
:class:`HelionFlexAttentionChoices` lowers a call the Helion kernels run and
returns its outputs; it declines any other call with the reason, which it
logs at INFO (``HELION_LOGS=helion.flex_attention``). It claims
``BACKEND="HELION"``, whose ``BACKEND_OPTIONS`` override :func:`configure`
for that call.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import TYPE_CHECKING
from typing import Any

from torch._inductor.choices import InductorChoices
from torch._inductor.kernel.flex.backend import FlexAttentionBackwardOutputs
from torch._inductor.kernel.flex.backend import FlexAttentionOutputs

from . import _kernels
from . import _lowering
from helion._compiler.cute.cutedsl_compat import installed_cute_dsl_version
from helion.autotuner.base_cache import helion_key
from helion.runtime.settings import autotuning_disallowed_reason

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch._inductor.kernel.flex.backend import FlexAttentionBackwardCall
    from torch._inductor.kernel.flex.backend import FlexAttentionCall

log = logging.getLogger("helion.flex_attention")

# The flex_attention BACKEND name that selects the Helion kernels.
BACKEND = "HELION"

# HELION_* environment variables that code generation does not read: those
# of the settings that change no generated code or that Helion writes back
# while it autotunes (and the LLM search's compile timeout), and those that
# only log or dump, check, locate or skip caches, drive the test harness or
# reach the LLM endpoint.  Any other variable, the ones code generation and
# the autotuner read directly included, keys the caches of generated code
# (``_keyed_env_var``), so new variables are keyed by default.
_NON_CODEGEN_ENV_VARS: frozenset[str] = frozenset(
    {
        "HELION_AOT_VERBOSE",
        "HELION_ASSERT_CACHE_HIT",
        "HELION_AUTOTUNER_TRITON_FAILURE_DUMP_DIR",
        "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS",
        "HELION_AUTOTUNE_COMPILE_TIMEOUT",
        "HELION_AUTOTUNE_METRICS_JSON",
        "HELION_AUTOTUNE_PRECOMPILE",
        "HELION_AUTOTUNE_PRECOMPILE_JOBS",
        "HELION_AUTOTUNE_PROGRESS_BAR",
        "HELION_AUTOTUNE_RANDOM_SEED",
        "HELION_AUTOTUNE_SWEEP_RESULT_JSON",
        "HELION_CACHE_DIR",
        "HELION_CUTE_LAUNCH_SHAPE_CHECK",
        "HELION_CUTE_LAUNCH_SHAPE_LOG",
        "HELION_DEBUG_DISTRIBUTED",
        "HELION_DELETE_CACHE_AFTER_TEST",
        "HELION_DISALLOW_AUTOTUNING",
        "HELION_DIST_CHECK_CONFIG_CONSISTANCY",
        "HELION_IGNORE_WARNINGS",
        "HELION_LLM_API_BASE",
        "HELION_LLM_API_KEY",
        "HELION_LLM_CA_BUNDLE",
        "HELION_LLM_CLIENT_CERT",
        "HELION_LLM_CLIENT_KEY",
        "HELION_LLM_COMPILE_TIMEOUT_S",
        "HELION_LLM_EXTRA_HEADERS",
        "HELION_LLM_VERTEX_LOCATION",
        "HELION_LLM_VERTEX_PROJECT",
        "HELION_LOGS",
        "HELION_MEASURE_COMPILE_TIME",
        "HELION_REMOTE_CACHE_BACKEND",
        "HELION_SKIP_CACHE",
    }
)
_NON_CODEGEN_ENV_PREFIXES: tuple[str, ...] = (
    "HELION_AUTOREVIEW_",
    "HELION_AUTOTUNE_LOG",
    "HELION_KEEP_",
    "HELION_PRINT_",
    "HELION_TEST",
)


def _keyed_env_var(name: str) -> bool:
    """Whether environment variable ``name`` keys the caches of the code
    Helion generates (see ``_NON_CODEGEN_ENV_VARS``)."""
    return (
        name.startswith("HELION_")
        and name not in _NON_CODEGEN_ENV_VARS
        and not name.startswith(_NON_CODEGEN_ENV_PREFIXES)
    )


@dataclasses.dataclass(frozen=True)
class Options:
    """What :func:`helion.flex_attention.configure` sets."""

    forward_config: Mapping[str, Any] | None = None
    backward_config: Mapping[str, Any] | None = None
    autotune_tolerance: float = _kernels.AUTOTUNE_TOLERANCE

    def config(self, direction: str) -> Mapping[str, Any] | None:
        """The configured config of the ``direction`` kernel, if any."""
        return self.forward_config if direction == "forward" else self.backward_config


options = Options()


def json_config(name: str, config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``config`` as a dict of JSON values (a Helion config), or ValueError
    naming ``name``: the FX graph cache key encodes it."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"{name} must be a dict of config knobs, got {config!r}")
    try:
        return json.loads(json.dumps(config))
    except TypeError as e:
        raise ValueError(f"{name} must hold JSON values: {e}") from None


def tolerance(value: object) -> float:
    """``value`` as an autotune tolerance, or ValueError."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not value > 0  # also NaN
    ):
        raise ValueError(f"autotune_tolerance must be a positive number, got {value!r}")
    return float(value)


def call_options(backend_options: Mapping[str, Any]) -> Options:
    """:data:`options` with a ``BACKEND="HELION"`` call's ``BACKEND_OPTIONS``
    (the arguments of :func:`configure`) in place of the configured ones."""
    fields = {f.name for f in dataclasses.fields(Options)}
    if unknown := sorted(set(backend_options) - fields):
        raise ValueError(
            f"BACKEND_OPTIONS for BACKEND='{BACKEND}' takes {sorted(fields)}, "
            f"got {unknown}"
        )
    overrides: dict[str, Any] = {}
    for name, value in backend_options.items():
        if name == "autotune_tolerance":
            overrides[name] = tolerance(value)
        else:
            overrides[name] = json_config(f"BACKEND_OPTIONS[{name!r}]", value)
    return dataclasses.replace(options, **overrides)


def cache_key() -> str:
    """What the code generated for a graph's flex_attention calls depends on
    beyond the graph and PyTorch: Helion's source (``helion_key``), the CuTe
    DSL version, the ``HELION_*`` variables code generation reads,
    :data:`options`, and whether autotuning is disallowed (a kernel that
    would autotune then keeps Inductor's). A call's ``BACKEND_OPTIONS`` are
    part of its graph, so they need no place here."""
    env = sorted((k, v) for k, v in os.environ.items() if _keyed_env_var(k))
    return json.dumps(
        [
            "helion.flex_attention",
            helion_key(),
            str(installed_cute_dsl_version()),
            env,
            [
                options.forward_config,
                options.backward_config,
                options.autotune_tolerance,
            ],
            autotuning_disallowed_reason() is not None,
        ],
        sort_keys=True,
    )


class HelionFlexAttentionChoices(InductorChoices):
    def uuid(self) -> str:
        # Everything the lowering depends on is in flex_attention_cache_key,
        # which keys only graphs that call flex_attention.
        return "helion.flex_attention"

    def flex_attention_backends(self) -> tuple[str, ...]:
        return (BACKEND,)

    def flex_attention_cache_key(self) -> str:
        return cache_key()

    def lower_flex_attention(self, call: FlexAttentionCall) -> FlexAttentionOutputs:
        try:
            out, lse, max_scores = _lowering.lower_forward(
                call, call_options(call.backend_options)
            )
        except _lowering.Declined as e:
            log.info("Helion declines this flex_attention forward: %s", e)
            raise
        return FlexAttentionOutputs(out, lse, max_scores)

    def lower_flex_attention_backward(
        self, call: FlexAttentionBackwardCall
    ) -> FlexAttentionBackwardOutputs:
        try:
            dq, dk, dv = _lowering.lower_backward(
                call, call_options(call.backend_options)
            )
        except _lowering.Declined as e:
            log.info("Helion declines this flex_attention backward: %s", e)
            raise
        # The kernels decline score_mods whose captured tensors need
        # gradients, so no captured tensor has one.
        return FlexAttentionBackwardOutputs(
            dq, dk, dv, (None,) * len(call.score_mod_buffers)
        )
