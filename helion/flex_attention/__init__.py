"""Run ``torch.nn.attention.flex_attention`` on Helion's CuTe kernels under
``torch.compile``.

Importing this module registers Helion with Inductor's flex_attention
lowering::

    import helion.flex_attention  # noqa: F401
    from torch.nn.attention.flex_attention import flex_attention

    out = torch.compile(flex_attention)(q, k, v, score_mod=..., block_mask=...)

From then on, every compiled flex_attention forward and backward that
Inductor lowers with its Triton kernels (``kernel_options`` ``BACKEND``
``"TRITON"``, or ``"AUTO"`` when it picks them over flex decoding and FLASH)
runs Helion's fused tcgen05 flash kernels instead when they support the call,
and keeps the Triton kernels otherwise: the forward and backward decide on
their own, and ``HELION_LOGS=helion.flex_attention`` logs why a call keeps
Triton's. The decision is made while planning (before any kernel compiles):
once Helion's kernel is chosen, Triton's candidates are gone, so a failure
compiling or autotuning it fails the compile rather than falling back.

The kernels need a PyTorch nightly from 2026-11-19 or later (or a source
build with pytorch/pytorch#205348), whose Inductor boxes a template's
outputs; on PyTorch 2.15, the first release with the flex_attention choice
hooks, the module imports but every call keeps Triton's kernels, and
importing it on an older PyTorch raises ``ImportError``. They also need a
Blackwell (sm100) GPU, the CuTe DSL (``nvidia-cutlass-dsl``), float16 or
bfloat16 inputs, query and key / value of one batch size (no batch
broadcast), static shapes (a graph compiled with dynamic shapes keeps
Triton's kernels), and score_mod / mask_mod functions of the coordinates and of
captured tensors indexed by them (with integer indices, not slices).

The kernels autotune when compiled, at Helion's autotune effort
(``HELION_AUTOTUNE_EFFORT``: "full" by default, minutes per kernel; "none"
takes Helion's default config), and Helion caches tuned configs on disk;
:func:`configure` fixes their configs instead.

Import this module before the first ``torch.compile`` call that lowers with
Inductor: Inductor builds its choices handler once per thread, so a thread
that compiled before the import (such as the autograd thread, which compiles
the backward) keeps Inductor's kernels. Inductor's FX graph cache key does
not see that: such a thread caches its Triton-lowered graphs under the key of
the enabled routing, where a process that imported in time can load them.
:func:`disable` and :func:`enable` switch the routing for every thread.
Inductor's FX graph cache key of every graph compiled after the import, with
or without a flex_attention call, includes this module's key (Helion's source,
the CuTe DSL version, the ``HELION_*`` variables code generation reads,
:func:`configure` and whether autotuning is disallowed), so changing those
misses the cache for all of them. The contributor stays registered while
disabled, so a contributor overriding
the same hooks registered after this import still does not get them (Inductor
gives each hook to the first contributor that overrides it).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import Any

from torch._inductor import choices as _inductor_choices
from torch._inductor.virtualized import V

from helion._compat import supports_inductor_flex_attention_choices

if TYPE_CHECKING:
    from collections.abc import Mapping

if not supports_inductor_flex_attention_choices():
    raise ImportError(
        "helion.flex_attention needs a PyTorch with Inductor's flex_attention "
        "choice hooks (torch._inductor.choices.register_inductor_choices and "
        "InductorChoices.append_flex_attention_backward_choices)"
    )

from . import _choices

__all__ = ["configure", "disable", "enable"]

_KEY = "helion.flex_attention"
_registered = False


def enable() -> None:
    """Route compiled flex_attention calls to the Helion kernels (importing
    this module does it)."""
    global _registered
    _choices.enabled = True
    if not _registered:
        # Build this thread's handler the way every other thread does, first:
        # that runs Inductor's lazy setup (installing torchTLX in fbcode),
        # which registering skips on the calling thread, so the threads agree
        # on the contributors registered before this one.
        V.choices  # noqa: B018
        _inductor_choices.register_inductor_choices(  # pyrefly: ignore[missing-attribute]
            _KEY, _choices.HelionFlexAttentionChoices
        )
        _registered = True


def disable() -> None:
    """Undo :func:`enable`: flex_attention calls compiled from now on keep
    Inductor's kernels."""
    _choices.enabled = False


def configure(
    *,
    forward_config: Mapping[str, Any] | None = None,
    backward_config: Mapping[str, Any] | None = None,
    autotune_tolerance: float = _choices._kernels.AUTOTUNE_TOLERANCE,
) -> None:
    """Set how the Helion kernels pick their configs, for the flex_attention
    calls compiled from now on. Each call replaces the whole setting: an
    argument left out returns to its default (``configure()`` restores all
    of them).

    Args:
        forward_config: A Helion config for the forward kernel, as a dict of
            config knobs (for example the config Helion prints after
            autotuning); the forward uses it instead of autotuning. Compiling
            a call it does not apply to fails: Inductor's lowering error
            wraps the ``ValueError`` saying why.
        backward_config: The same for the backward kernel.
        autotune_tolerance: When autotuning, a candidate config is accepted
            when each output element is within this fraction of the output's
            peak magnitude of Helion's default config.
    """
    if (
        isinstance(autotune_tolerance, bool)
        or not isinstance(autotune_tolerance, (int, float))
        or not autotune_tolerance > 0
    ):
        raise ValueError(
            f"autotune_tolerance must be a positive number, got {autotune_tolerance!r}"
        )
    _choices.options = _choices.Options(
        forward_config=_json_config("forward_config", forward_config),
        backward_config=_json_config("backward_config", backward_config),
        autotune_tolerance=float(autotune_tolerance),
    )


def _json_config(name: str, config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``config`` as a dict of JSON values: the FX graph cache key of every
    compile encodes it, so a value JSON cannot encode is rejected here."""
    if config is None:
        return None
    try:
        return json.loads(json.dumps(dict(config)))
    except TypeError as e:
        raise ValueError(f"{name} must hold JSON values: {e}") from None


enable()
