"""Map messy device strings to a small set of hardware families."""

from __future__ import annotations

_DEVICE_TOKENS: tuple[tuple[str, str], ...] = (
    ("h100", "h100"),
    ("b200", "b200"),
    ("gfx950", "mi350x"),
    ("mi350", "mi350x"),
    ("tpu", "tpu"),
)


def _family_from_device(device: str | None) -> str | None:
    """Match device string against known tokens, case-insensitive."""
    if not device:
        return None
    h = device.lower()
    return next((fam for tok, fam in _DEVICE_TOKENS if tok in h), None)


def _family_from_manifest_alias(
    device: str | None, manifest: dict | None
) -> str | None:
    """Try manifest aliases if token list misses."""
    if not device or not manifest:
        return None
    d = device.lower()
    families = manifest.get("families") or {}
    return next(
        (
            fam
            for fam, spec in families.items()
            for alias in spec.get("aliases") or []
            if (a := str(alias).lower()) == d or a in d
        ),
        None,
    )


def _family_from_compute_capability(
    cc: str | None, manifest: dict | None
) -> str | None:
    """Match by compute capability string from manifest."""
    if not cc or not manifest:
        return None
    cc_str = str(cc)
    families = manifest.get("families") or {}
    return next(
        (
            fam
            for fam, spec in families.items()
            if cc_str in map(str, spec.get("compute_capabilities") or [])
        ),
        None,
    )


def _helion_device_string() -> str | None:
    """Ask torch for current CUDA device name, or None if no CUDA."""
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def resolve_family(
    device: str | None = None,
    manifest: dict | None = None,
    *,
    env_family: str | None = None,
    override: str | None = None,
    compute_capability: str | None = None,
    _device_fn=None,
) -> str | None:
    """Pick family by precedence: override > env > device token > torch > manifest alias > compute capability."""
    device_fn = _device_fn or _helion_device_string
    return (
        override
        or env_family
        or _family_from_device(device)
        or _family_from_device(device_fn())
        or _family_from_manifest_alias(device, manifest)
        or _family_from_compute_capability(compute_capability, manifest)
    )
