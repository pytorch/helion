"""Install RAG hook on BoundKernel so autotune can reuse cached configs."""

from __future__ import annotations

import os
from pathlib import Path

from helion_rag._util import _log
import helion_rag.corpus as corpus
from helion_rag.lookup import lookup
from helion_rag.lookup import merge_seed_configs

HOOK_BEGIN = "# >>> helion-rag hook >>>"
HOOK_END = "# <<< helion-rag <<<"
HOOK_BODY = """import helion_rag.patch as _hrp
_hrp.install()"""


def write_hook(target: Path | str) -> bool:
    """Append RAG hook to target Python file if not present. Returns True if written,
    False if already present. Raises FileNotFoundError if missing."""
    p = Path(target)
    if not p.is_file():
        raise FileNotFoundError(f"Patch target not found: {p}")
    text = p.read_text(encoding="utf-8")
    if HOOK_BEGIN in text:
        return False
    hook = f"\n{HOOK_BEGIN}\n{HOOK_BODY}\n{HOOK_END}\n"
    p.write_text(text.rstrip("\n") + hook, encoding="utf-8")
    return True


def install() -> bool:
    """Patch BoundKernel.ensure_config_exists to try RAG first. Returns True if patched."""
    from helion.runtime.kernel import BoundKernel

    if getattr(BoundKernel, "_helion_rag_patched", False):
        return True

    original = BoundKernel.ensure_config_exists

    def _wrapped(self, args, *rest, **kwargs):
        # Helion passes the full arg tuple as single positional; keep it intact.
        return apply(self, original, args, rest, kwargs)

    BoundKernel.ensure_config_exists = _wrapped
    BoundKernel._helion_rag_patched = True
    _log("patch: RAG runtime hook installed on BoundKernel.ensure_config_exists")
    return True


def _hardware_name(info) -> str:
    """Pull readable name out of hardware info dict or object."""
    keys = ("device_name", "name", "hardware", "gpu")
    if isinstance(info, dict):
        return next((str(info[k]) for k in keys if info.get(k)), str(info))
    return next(
        (str(getattr(info, a)) for a in keys if getattr(info, a, None)), str(info)
    )


def _extract(bound_kernel, args) -> dict | None:
    """Pull kernel source, shapes, dtypes, hardware and settings from a BoundKernel call.
    Returns None to fall back to normal autotune."""
    from helion import _hardware
    from helion.runtime.kernel import _find_device

    ks_fn = getattr(getattr(bound_kernel, "kernel", None), "kernel_source", None)
    kernel_source = ks_fn() if callable(ks_fn) else None
    if not kernel_source:
        return None

    import torch

    arg_tuple = tuple(args)
    device = _find_device(arg_tuple) if callable(_find_device) else None
    if device is None:
        return None

    hardware = (
        getattr(_hardware, "get_hardware_info", lambda _: None)(device)
        if hasattr(_hardware, "get_hardware_info")
        else None
    )
    # Use same shape/dtype string format as ingest so workload key matches.
    tensor_args = [a for a in arg_tuple if isinstance(a, torch.Tensor)]

    return {
        "kernel_source": kernel_source,
        "shapes": str([tuple(a.shape) for a in tensor_args]),
        "dtypes": str([str(a.dtype) for a in tensor_args]),
        "hardware": _hardware_name(hardware),
        "settings": _settings_dict(getattr(bound_kernel, "settings", None)),
    }


def _settings_dict(settings) -> dict:
    """Keep only codegen settings that affect workload key, filling missing with None."""
    if not settings:
        return {}
    keys = corpus._CODEGEN_SETTINGS
    if isinstance(settings, dict):
        return {k: settings.get(k) for k in keys}
    return {k: getattr(settings, k, None) for k in keys}


def _to_config(raw):
    """Turn stored dict into Helion Config object, or pass through if already good."""
    if not isinstance(raw, dict):
        return raw

    from helion import Config

    return Config(**raw)


class _TempSeedConfigs:
    """Swap in temporary seed configs for one autotune run then restore."""

    def __init__(self, settings, temp_seeds):
        self.settings = settings
        self.temp_seeds = temp_seeds
        self.orig_seeds = getattr(settings, "autotune_seed_configs", None)

    def __enter__(self):
        self.settings.autotune_seed_configs = self.temp_seeds

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.autotune_seed_configs = self.orig_seeds


def _tier1_seeded(bound_kernel, original, args, rest, kwargs, res):
    """Run original autotune but with RAG seeds merged in temporarily."""
    settings = getattr(bound_kernel, "settings", None)
    rag_seeds = [
        cfg
        for nb in res.get("neighbors", [])
        for e in nb.get("top_n", []) or []
        if (cfg := _to_config(e.get("config"))) is not None
    ]
    if not settings or not rag_seeds:
        return original(bound_kernel, args, *rest, **kwargs)

    orig = list(getattr(settings, "autotune_seed_configs", None) or [])
    temp = merge_seed_configs(orig, rag_seeds)
    with _TempSeedConfigs(settings, temp):
        return original(bound_kernel, args, *rest, **kwargs)


def apply(
    bound_kernel, original, args, rest, kwargs, *, _extract_fn=None, _lookup_fn=None
):
    """RAG wrapper around ensure_config_exists. Falls back to original on miss or error."""
    if os.environ.get("HELION_RAG_ENABLED") != "1":
        return original(bound_kernel, args, *rest, **kwargs)

    extract_fn = _extract_fn or _extract
    lookup_fn = _lookup_fn or lookup
    info = extract_fn(bound_kernel, args)
    if not info:
        return original(bound_kernel, args, *rest, **kwargs)

    res = (
        lookup_fn(
            info["kernel_source"],
            info["shapes"],
            info["dtypes"],
            info["hardware"],
            settings=info.get("settings"),
        )
        or {}
    )

    tier = res.get("tier")
    if tier == 0:
        cfg = _to_config(res.get("best_config"))
        if cfg is not None:
            bound_kernel.set_config(cfg)
            return None
        return original(bound_kernel, args, *rest, **kwargs)

    if tier == 1:
        return _tier1_seeded(bound_kernel, original, args, rest, kwargs, res)

    return original(bound_kernel, args, *rest, **kwargs)
