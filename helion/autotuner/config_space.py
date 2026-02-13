from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Any

import torch

from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment

if TYPE_CHECKING:
    from .config_spec import ConfigSpec


class ConfigSpace:
    """User-facing container describing constraints on the autotuner search space.

    Each keyword argument can be:
    - A plain value (pinned)
    - A ConfigSpecFragment (constrained search range)
    - A callable ``(proxy_args, config) -> value`` (derived)

    For list-valued parameters (e.g. ``block_sizes``), each element is
    independently classified. Shorter lists are padded with ``None``
    (unconstrained).

    **Serialization note:** ``config_space_fn`` and any derived callables are
    not serialized. The AOT cache stores only the final ``Config`` dicts
    produced after derived resolution. When loading a cached config, the
    ``config_space_fn`` is re-evaluated to rebuild the ``ConfigSpec`` but
    the stored config values take precedence.
    """

    def __init__(self, **kwargs: object) -> None:
        self.entries: dict[str, object] = kwargs


class _TensorProxy:
    """Lightweight proxy for tensor args in config_space_fn.

    Exposes ``.shape``, ``.dtype``, ``.device``, ``.size()``.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        if dim is None:
            return self.shape
        return self.shape[dim]

    def __repr__(self) -> str:
        return f"_TensorProxy(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def _make_config_space_proxy_args(
    real_args: tuple[object, ...],
) -> list[object]:
    """Build lightweight proxy args for config_space_fn."""
    proxy_args: list[object] = []
    for real_arg in real_args:
        if isinstance(real_arg, torch.Tensor):
            shape = tuple(int(s) for s in real_arg.shape)
            proxy_args.append(
                _TensorProxy(shape, real_arg.dtype, real_arg.device)
            )
        else:
            proxy_args.append(real_arg)
    return proxy_args


# Maps config dict keys (plural, e.g. "range_warp_specializes") to
# ConfigSpec attribute names (which may differ, e.g. "range_warp_specialize").
# The mismatch is intentional: config dicts use the plural form while
# ConfigSpec dataclass fields use the singular/base form.
_SEQUENCE_FIELDS: dict[str, str] = {
    "block_sizes": "block_sizes",
    "loop_orders": "loop_orders",
    "l2_groupings": "l2_groupings",
    "reduction_loops": "reduction_loops",
    "flatten_loops": "flatten_loops",
    "range_unroll_factors": "range_unroll_factors",
    "range_warp_specializes": "range_warp_specialize",
    "range_num_stages": "range_num_stages",
    "range_multi_buffers": "range_multi_buffers",
    "range_flattens": "range_flattens",
    "static_ranges": "static_ranges",
}


def _get_sequence(config_spec: ConfigSpec, key: str) -> Any:
    """Get the BlockIdSequence for a list-valued config key."""
    from ..exc import InvalidConfig

    attr = _SEQUENCE_FIELDS.get(key)
    if attr is not None:
        return getattr(config_spec, attr)
    raise InvalidConfig(
        f"List override not supported for {key!r}. "
        f"Supported list keys: {sorted(_SEQUENCE_FIELDS)!r}"
    )


def _validate_callable(fn: object, context: str) -> None:
    """Check that *fn* accepts at least 2 positional parameters (args, config)."""
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return  # built-ins / C functions – skip
    # Count parameters that can be filled positionally.
    positional = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    has_var_positional = any(
        p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if positional < 2 and not has_var_positional:
        from ..exc import InvalidConfig

        raise InvalidConfig(
            f"Derived callable for {context} must accept (args, config), "
            f"but {fn!r} only accepts {positional} positional argument(s)"
        )


def apply_config_space(
    config_spec: ConfigSpec,
    config_space: ConfigSpace,
    proxy_args: list[object],
) -> None:
    """Apply a ConfigSpace to a ConfigSpec, modifying it in place.

    Three data paths depending on the entry type:

    1. **List values** (e.g. ``block_sizes``): each element is matched to the
       corresponding ``BlockIdSequence`` item. Elements can be ``None``
       (unconstrained), a ``ConfigSpecFragment`` (sets ``_override_fragment``),
       a plain value (pinned via single-choice ``EnumFragment``), or a callable
       ``(args, config) -> value`` (resolved later by ``_resolve_derived``).

    2. **Built-in scalar keys** (e.g. ``num_warps``, ``pid_type``): handled by
       ``_get_effective_fragment`` during ``flat_config``. Fragments constrain
       the search range; plain values pin it; callables pin a placeholder
       default and are resolved in ``_resolve_derived``.

    3. **User-defined keys**: registered in ``user_defined_tunables`` so they
       participate in the normal ``flat_config`` / ``normalize`` flow.
    """
    from ..exc import InvalidConfig
    from .config_spec import VALID_KEYS

    config_spec._config_space = config_space
    config_spec._config_space_proxy_args = proxy_args

    for key, value in config_space.entries.items():
        if isinstance(value, list):
            sequence = _get_sequence(config_spec, key)
            if len(value) > len(sequence):
                raise InvalidConfig(
                    f"Too many elements for {key!r}: got {len(value)}, "
                    f"expected at most {len(sequence)}"
                )
            for i, elem in enumerate(value):
                if elem is None:
                    continue
                if callable(elem) and not isinstance(elem, ConfigSpecFragment):
                    _validate_callable(elem, f"{key}[{i}]")
                elif isinstance(elem, ConfigSpecFragment):
                    sequence[i]._override_fragment = elem
                else:
                    sequence[i]._override_fragment = EnumFragment(choices=(elem,))
        elif callable(value) and not isinstance(value, ConfigSpecFragment):
            _validate_callable(value, key)
        elif isinstance(value, ConfigSpecFragment):
            if key not in VALID_KEYS and key not in config_spec.user_defined_tunables:
                config_spec.user_defined_tunables[key] = value
        else:
            # Pinned scalar
            if key not in VALID_KEYS and key not in config_spec.user_defined_tunables:
                config_spec.user_defined_tunables[key] = EnumFragment(choices=(value,))
