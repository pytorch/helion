"""Run ``torch.nn.attention.flex_attention`` on Helion's CuTe kernels under
``torch.compile``.

Importing this module registers Helion as a flex_attention backend with
Inductor::

    import helion.flex_attention  # noqa: F401
    from torch.nn.attention.flex_attention import flex_attention

    out = torch.compile(flex_attention)(q, k, v, score_mod=..., block_mask=...)

From then on, Inductor offers every compiled flex_attention forward and
backward with ``kernel_options`` ``BACKEND`` ``"AUTO"`` (the default) to
Helion first. Helion's fused tcgen05 flash kernels run the calls they support,
and Inductor lowers the others as before: the forward and backward decide on
their own, and ``HELION_LOGS=helion.flex_attention`` logs why Helion declines
a call. ``kernel_options={"BACKEND": "HELION"}`` asks for the Helion kernels
by name: a call they do not support then fails to compile, with the reason,
and ``kernel_options["BACKEND_OPTIONS"]`` can take :func:`configure`'s
arguments for that call::

    flex_attention(
        q,
        k,
        v,
        kernel_options={
            "BACKEND": "HELION",
            "BACKEND_OPTIONS": {"forward_config": {...}},
        },
    )

The other built-in backends (``"TRITON"``, ``"FLASH"``, ...) keep Inductor's
kernels. The decision is made before any kernel compiles: once Helion takes a
call, a failure compiling or autotuning its kernel fails the compile rather
than falling back.

The kernels need a Blackwell (sm100) GPU, the CuTe DSL
(``nvidia-cutlass-dsl``), float16 or bfloat16 inputs, query and key / value of
one batch size (no batch broadcast), static shapes (a graph compiled with
dynamic shapes keeps Inductor's kernels), and score_mod / mask_mod functions
of the coordinates and of captured tensors indexed by them (with integer
indices, not slices).

The kernels follow Helion's ``fast_math`` setting: ``HELION_FAST_MATH=1``
lets them use approximations less accurate than eager (polynomial ``exp2``,
``tanh.approx``), as FlashAttention-4 does.

The kernels autotune when compiled, at Helion's autotune effort
(``HELION_AUTOTUNE_EFFORT``: "full" by default), and Helion caches tuned
configs on disk; :func:`configure` (or a call's ``BACKEND_OPTIONS``) fixes
their configs instead.

Inductor's FX graph cache key of a graph that calls flex_attention includes
Helion's key (Helion's source, the CuTe DSL version, the ``HELION_*``
variables code generation reads, :func:`configure` and whether autotuning is
disallowed); other graphs' keys do not depend on it. Enabling or disabling
Helion changes every graph's key once (the registered contributors are part
of it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from torch._inductor.choices import register_inductor_choices
from torch._inductor.choices import unregister_inductor_choices

from helion._compat import supports_inductor_flex_attention_backends

if TYPE_CHECKING:
    from collections.abc import Mapping

if not supports_inductor_flex_attention_backends():
    raise ImportError(
        "helion.flex_attention needs a PyTorch whose Inductor lets an "
        "InductorChoices contributor lower a whole flex_attention call "
        "(InductorChoices.lower_flex_attention, flex_attention_backends and "
        "flex_attention_cache_key)"
    )

from . import _choices

__all__ = ["configure", "disable", "enable"]

_KEY = "helion.flex_attention"


def enable() -> None:
    """Register Helion as a flex_attention backend (importing this module
    does it)."""
    register_inductor_choices(_KEY, _choices.HelionFlexAttentionChoices)


def disable() -> None:
    """Undo :func:`enable`: flex_attention calls compiled from now on keep
    Inductor's kernels, and ``BACKEND="HELION"`` is no longer accepted."""
    unregister_inductor_choices(_KEY)


def configure(
    *,
    forward_config: Mapping[str, Any] | None = None,
    backward_config: Mapping[str, Any] | None = None,
    autotune_tolerance: float = _choices._kernels.AUTOTUNE_TOLERANCE,
) -> None:
    """Set how the Helion kernels pick their configs, for the flex_attention
    calls compiled from now on. Each call replaces the whole setting: an
    argument left out returns to its default (``configure()`` restores all
    of them). A ``BACKEND="HELION"`` call's ``BACKEND_OPTIONS`` take the same
    arguments and override these for that call.

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
    _choices.options = _choices.Options(
        forward_config=_choices.json_config(
            "forward_config", None if forward_config is None else dict(forward_config)
        ),
        backward_config=_choices.json_config(
            "backward_config",
            None if backward_config is None else dict(backward_config),
        ),
        autotune_tolerance=_choices.tolerance(autotune_tolerance),
    )


enable()
