from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Literal
from typing import cast

CUTE_STRUCTURAL_SETTINGS = (
    "cute_region_fission",
    "cute_full_slice_matmul_tiling",
    "cute_segmented_matmul_tiling",
    "cute_flatten_nested_reductions",
    "cute_materialize_transformed_operands",
)
_POLICY_SCHEMA = "helion.cute.structural_policy"
_POLICY_VERSION = 1
StructuralSettingOrigin = Literal["default", "environment", "explicit"]


@dataclasses.dataclass(frozen=True, slots=True)
class CuteStructuralPolicy:
    """An immutable snapshot of the effective structural settings.

    This record does not select a policy, activate a pass, or extend Config.
    Its versioned envelope is separate from the flattened autotuner knobs.
    Origin is deliberately absent from the effective policy's identity: an
    explicit False and a default False describe the same current structure.
    """

    cute_region_fission: bool = False
    cute_full_slice_matmul_tiling: bool = False
    cute_segmented_matmul_tiling: bool = False
    cute_flatten_nested_reductions: bool = False
    cute_materialize_transformed_operands: bool = False
    version: Literal[1] = dataclasses.field(default=1, init=False)

    def __post_init__(self) -> None:
        for name in CUTE_STRUCTURAL_SETTINGS:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

    def to_dict(self) -> dict[str, object]:
        """Return a detached, versioned envelope, including every False value."""
        return {
            "schema": _POLICY_SCHEMA,
            "version": self.version,
            "settings": {
                name: getattr(self, name) for name in CUTE_STRUCTURAL_SETTINGS
            },
        }

    @classmethod
    def from_dict(cls, envelope: Mapping[str, object]) -> CuteStructuralPolicy:
        """Read an exact envelope; never interpret flattened legacy Configs."""
        if set(envelope) != {"schema", "version", "settings"}:
            raise ValueError("Expected a CuTe structural policy envelope")
        if envelope["schema"] != _POLICY_SCHEMA:
            raise ValueError("Unsupported CuTe structural policy schema")
        version = envelope["version"]
        if type(version) is not int or version != _POLICY_VERSION:
            raise ValueError(f"Unsupported CuTe structural policy version: {version!r}")
        settings = envelope["settings"]
        if not isinstance(settings, Mapping) or set(settings) != set(
            CUTE_STRUCTURAL_SETTINGS
        ):
            raise ValueError("Expected exactly the five CuTe structural settings")
        return cls(**cast("dict[str, bool]", dict(settings)))

    def to_json(self) -> str:
        """Canonical serialization, independent of mapping insertion order."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def identity(self) -> str:
        """A stable digest of the versioned effective policy, not Python hash()."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class CuteStructuralOrigins:
    """Construction origins, updated only by explicit settings writes/copies.

    An environment value is an intentional request, including "0". A missing
    or blank environment value uses the default. This metadata is not a
    Settings dataclass field and does not participate in existing cache keys.
    """

    cute_region_fission: StructuralSettingOrigin = "default"
    cute_full_slice_matmul_tiling: StructuralSettingOrigin = "default"
    cute_segmented_matmul_tiling: StructuralSettingOrigin = "default"
    cute_flatten_nested_reductions: StructuralSettingOrigin = "default"
    cute_materialize_transformed_operands: StructuralSettingOrigin = "default"

    def __post_init__(self) -> None:
        for name in CUTE_STRUCTURAL_SETTINGS:
            value = getattr(self, name)
            if type(value) is not str or value not in (
                "default",
                "environment",
                "explicit",
            ):
                raise TypeError(f"Invalid origin for {name}: {value!r}")

    def with_explicit(self, *names: str) -> CuteStructuralOrigins:
        return dataclasses.replace(self, **dict.fromkeys(names, "explicit"))
