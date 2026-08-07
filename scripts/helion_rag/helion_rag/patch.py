"""Warm-start Helion's LFBO autotuner from retrieved neighbour configs.

This is the experiment's own seeding path, driven by ``HELION_RAG_LOO_SEEDING``.
It predates the in-tree adapter at ``helion/autotuner/rag/`` (gated on
``HELION_RAG_ENABLED``) and is kept because it is the treatment that produced
the published leave-one-workload-out numbers. The two must never run together;
:func:`install` asserts that.
"""

from __future__ import annotations

import os

from helion_rag._util import _log
import helion_rag.corpus as corpus
from helion_rag.lookup import lookup

_ENABLE_ENV = "HELION_RAG_LOO_SEEDING"


def install() -> bool:
    """Patch ``BoundKernel.ensure_config_exists`` once."""
    from helion.runtime.kernel import BoundKernel

    assert os.environ.get("HELION_RAG_ENABLED", "0") not in {"1", "true", "True"}, (
        f"{_ENABLE_ENV} and HELION_RAG_ENABLED are both set: this seeding path "
        "and helion/autotuner/rag/ would each retrieve and seed. Enable one."
    )
    if getattr(BoundKernel, "_helion_rag_patched", False):
        return True
    original = BoundKernel.ensure_config_exists

    def wrapped(self, args, *rest, **kwargs):
        return apply(self, original, args, rest, kwargs)

    BoundKernel.ensure_config_exists = wrapped
    BoundKernel._helion_rag_patched = True
    return True


def _settings_dict(settings) -> dict:
    if not settings:
        return {}
    if isinstance(settings, dict):
        return {key: settings.get(key) for key in corpus._CODEGEN_SETTINGS}
    return {key: getattr(settings, key) for key in corpus._CODEGEN_SETTINGS}


def _extract(bound_kernel, args) -> dict | None:
    from helion import _hardware
    from helion.runtime.kernel import _find_device
    import torch

    source = bound_kernel.kernel.kernel_source()
    if not source:
        return None
    arg_tuple = tuple(args)
    hardware = _hardware.get_hardware_info(_find_device(arg_tuple))
    tensors = [arg for arg in arg_tuple if isinstance(arg, torch.Tensor)]
    return {
        "kernel_name": bound_kernel.kernel.fn.__name__,
        "kernel_source": source,
        "shapes": str([tuple(arg.shape) for arg in tensors]),
        "dtypes": str([str(arg.dtype) for arg in tensors]),
        "hardware": hardware.hardware_name,
        "settings": _settings_dict(bound_kernel.settings),
    }


def _to_config(raw):
    from helion import Config

    if isinstance(raw, Config):
        return raw
    return Config(**raw) if isinstance(raw, dict) else None


class _TemporarySeeds:
    """Scope ``autotune_seed_configs`` to one autotune run."""

    def __init__(self, settings, seeds) -> None:
        self.settings = settings
        self.seeds = seeds
        self.original_seeds = settings.autotune_seed_configs

    def __enter__(self):
        self.settings.autotune_seed_configs = self.seeds

    def __exit__(self, exc_type, exc_value, traceback):
        self.settings.autotune_seed_configs = self.original_seeds


_MAX_SEED_CONFIGS = 3


def _curated_seed_configs(result, *, limit: int = _MAX_SEED_CONFIGS):
    """Shape-closest retrieved configs, round-robin over neighbours, capped.

    Ordering neighbours by shape proximity and taking each one's best config
    first keeps the injected anchors both relevant to the target shape and
    diverse, while the cap preserves the FROM_RANDOM remainder so seeding cannot
    collapse LFBO's exploration.
    """
    neighbors = sorted(
        result.get("neighbors", []),
        key=lambda neighbor: (
            neighbor.get("shape_distance", float("inf")),
            -(neighbor.get("relevance") or 0.0),
        ),
    )
    max_rank = max(
        (len(neighbor.get("top_n") or []) for neighbor in neighbors), default=0
    )
    configs = []
    seen = set()
    for rank in range(max_rank):
        for neighbor in neighbors:
            entries = neighbor.get("top_n") or []
            if rank >= len(entries):
                continue
            config = _to_config(entries[rank].get("config"))
            if config is None:
                continue
            key = repr(config)
            if key in seen:
                continue
            seen.add(key)
            configs.append(config)
            if len(configs) >= limit:
                return configs
    return configs


def _seeded_autotune(bound_kernel, original, args, rest, kwargs, result):
    from helion.autotuner.base_search import normalize_autotune_seed_configs

    retrieved = _curated_seed_configs(result)
    if not retrieved:
        return original(bound_kernel, args, *rest, **kwargs)
    existing = list(normalize_autotune_seed_configs(bound_kernel.settings))
    unique = []
    seen = set()
    for config in [*existing, *retrieved]:
        key = repr(config)
        if key not in seen:
            seen.add(key)
            unique.append(config)
    with _TemporarySeeds(bound_kernel.settings, unique):
        return original(bound_kernel, args, *rest, **kwargs)


def apply(bound_kernel, original, args, rest, kwargs):
    """Replay an exact Tier-0 hit, else warm-start the search from Tier-1 seeds."""
    if os.environ.get(_ENABLE_ENV) != "1":
        return original(bound_kernel, args, *rest, **kwargs)
    if (
        bound_kernel._config is not None
        or bound_kernel.configs
        or bound_kernel.settings.force_autotune
    ):
        return original(bound_kernel, args, *rest, **kwargs)
    info = _extract(bound_kernel, args)
    if info is None:
        return original(bound_kernel, args, *rest, **kwargs)
    try:
        result = lookup(
            info["kernel_source"],
            info["shapes"],
            info["dtypes"],
            info["hardware"],
            settings=info["settings"],
            kernel_name=info["kernel_name"],
        )
        if result.get("tier") == 0 and (
            config := _to_config(result.get("best_config"))
        ):
            bound_kernel.set_config(config)
            return None
    except Exception as exc:
        # Degradation boundary: retrieval is an optimization, so a failed lookup
        # falls back to an unseeded autotune. The rate is reported as a
        # diagnostic rather than failing the cell.
        _log(
            f"apply: RAG lookup failed ({type(exc).__name__}: {exc}); using normal autotune"
        )
        return original(bound_kernel, args, *rest, **kwargs)
    if result.get("tier") != 1:
        return original(bound_kernel, args, *rest, **kwargs)
    bound_kernel._helion_rag_lookup = result
    return _seeded_autotune(bound_kernel, original, args, rest, kwargs, result)
