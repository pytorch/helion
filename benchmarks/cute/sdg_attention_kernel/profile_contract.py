"""Validate profile lineage without confusing it with current source-byte hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# These identify the archived authored kernels that produced the retained tuning
# records, before source formatting and packaging. They are not hashes of the
# current Python files; exported provenance records those independently.
ORIGIN_FINGERPRINTS = {
    "recurrence": "30ca0984659208b06453ca4443640f12187af5ec1cc7ffd32d58618a74fd23ad",
    "rowwise": "1ac3b10aebc37e18e5e74447ba715825d35dcb17839555739faafc1ddbf9a979",
    "rowwise_extended": "f6bb935f917620ddb6ccfdcaf270c0d5f1b44f3cc21c8adde90860421ed21bc8",
    "headwise": "cb75b4ed9b6b5302fe793f6725b7c4e220ac22e37084bb175be2d16d51fdb055",
}


ROWWISE_THRESHOLDS = (2, 16, 32, 64, 4, 8)
INPUT_NAMES = ("q", "k", "v", "ks", "vs", "table", "lengths", "out")


def source_fingerprint(family: str) -> str:
    """Fingerprint the current authored family file for newly tuned profiles."""
    filename = "rowwise" if family == "rowwise_extended" else family
    if not isinstance(family, str) or family not in ORIGIN_FINGERPRINTS:
        raise ValueError(f"Unsupported SDG source family {family!r}")
    return hashlib.sha256(
        Path(__file__).with_name(filename + ".py").read_bytes()
    ).hexdigest()


def profile_signature(profile: dict[str, Any]) -> tuple[object, ...]:
    inputs = profile["inputs"]
    return (
        tuple(
            (
                tuple(inputs[name]["shape"]),
                tuple(inputs[name]["stride"]),
                inputs[name]["dtype"],
                profile["scale_storage_offset"] if name in ("ks", "vs") else 0,
            )
            for name in INPUT_NAMES
        ),
        profile["max_kv_len"],
        profile["window_left"],
        profile["return_lse"],
    )


def _validate_layout(profile: dict[str, Any], path: Path) -> None:
    inputs = profile.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Missing SDG profile inputs: {path}")
    for name in INPUT_NAMES:
        tensor = inputs.get(name)
        if not isinstance(tensor, dict):
            raise ValueError(f"Missing SDG profile input {name!r}: {path}")
        shape, stride = tensor.get("shape"), tensor.get("stride")
        if (
            not isinstance(shape, list)
            or not isinstance(stride, list)
            or len(shape) != len(stride)
            or any(type(value) is not int for value in (*shape, *stride))
            or not isinstance(tensor.get("dtype"), str)
        ):
            raise ValueError(f"Invalid SDG profile input layout {name!r}: {path}")
    if (
        type(profile.get("max_kv_len")) is not int
        or type(profile.get("scale_storage_offset")) is not int
        or "window_left" not in profile
        or (
            profile["window_left"] is not None
            and type(profile["window_left"]) is not int
        )
        or type(profile.get("return_lse")) is not bool
    ):
        raise ValueError(f"Missing or invalid SDG profile scalar layout: {path}")


def load_profiles(path: Path) -> list[dict[str, Any]]:
    """Read the external tuning artifact and check its supported source lineage."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(
            f"Missing SDG profiles at {path}. Set SDG_ATTENTION_PROFILES when "
            "using the source entry, or export a bundle with --profiles."
        ) from error
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("profiles"), list)
    ):
        raise ValueError(f"Unsupported SDG profile schema: {path}")
    profiles = data["profiles"]
    if not profiles:
        raise ValueError(f"Empty SDG profile dataset: {path}")
    names = set()
    signatures = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError(f"Invalid SDG profile record: {path}")
        name = profile.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Missing SDG profile name: {path}")
        _validate_layout(profile, path)
        signature = profile_signature(profile)
        if name in names or signature in signatures:
            raise ValueError(
                f"Duplicate SDG attention profile name or input layout: {path}"
            )
        names.add(name)
        signatures.add(signature)
        family = profile.get("family")
        if not isinstance(family, str) or family not in ORIGIN_FINGERPRINTS:
            raise ValueError(f"Unsupported SDG source family {family!r}: {path}")
        if profile.get("authored_source_sha256") not in (
            ORIGIN_FINGERPRINTS[family],
            source_fingerprint(family),
        ):
            raise ValueError(
                f"Unknown SDG tuning-origin fingerprint for {family}: {path}"
            )
        if not isinstance(profile.get("config"), dict):
            raise ValueError(f"Missing SDG config for {profile.get('name')!r}: {path}")
        if family in ("rowwise", "rowwise_extended"):
            threshold = profile["config"].get("rescale_threshold")
            if type(threshold) is not int or threshold not in ROWWISE_THRESHOLDS:
                raise ValueError(
                    f"The {family} family requires a supported integer threshold "
                    f"{ROWWISE_THRESHOLDS}: {path}"
                )
    return profiles
