"""Successful scalar-lowering evidence for single-policy search neighbors."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel

_POLICIES = ("", "first", "last", "streaming", "l2_last", "l1_l2_first", "l1_l2_last")
_ACTIVE: ContextVar[PolicyUses | None] = ContextVar(
    "cute_eviction_relevance", default=None
)


def _config_key(config: Config) -> str:
    return json.dumps(config.config, sort_keys=True, separators=(",", ":"))


def authority(bound: BoundKernel) -> str | None:
    """Reuse source ownership checks, permitting only tensor-content changes."""
    import torch

    from ...runtime.cute.source_batch import _snapshot

    if (
        bound.settings.backend != "cute"
        or not bound.settings.static_shapes
        or bound.env.cute_fission_plan is not None
        or bound.env.cute_resolved_wrapper_plans
        or bound.env.process_group_name is not None
        or bound.host_function is None
        or len(bound.host_function.device_ir.root_ids) != 1
        or any(
            not spec.reusable_tensor_properties
            or not spec.reusable_tensor_properties <= {"data_ptr", "storage_span"}
            for spec in bound.env.runtime_input_specializations.values()
        )
    ):
        return None
    tensors = tuple(ref() for ref in bound._runtime_tensor_refs_by_name.values())
    if not tensors or any(type(tensor) is not torch.Tensor for tensor in tensors):
        return None
    try:
        facts = json.loads(_snapshot(bound, cast("tuple[torch.Tensor, ...]", tensors)))
    except (TypeError, ValueError, RuntimeError):
        return None
    # Classification is restricted above to metadata-only input specializations.
    # Original search reset/poison operations change values/versions, not lowering.
    facts["owners"] = [owner[:-1] for owner in facts["owners"]]
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyReceipt:
    authority: str
    config_key: str
    source_sha256: str
    classes: tuple[tuple[str, ...] | None, ...]

    def excludes(self, candidate: Config) -> bool:
        old = json.loads(self.config_key)
        new = json.loads(_config_key(candidate))
        before = old.pop("load_eviction_policies", None)
        after = new.pop("load_eviction_policies", None)
        # A config can omit this field when there is no load-policy domain.
        # Such a receipt supplies no ordinal evidence for excluding a neighbor.
        if (
            old != new
            or not isinstance(before, list)
            or not isinstance(after, list)
            or len(before) != len(after)
        ):
            return False
        changed = [
            i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
        ]
        if len(changed) != 1:
            return False
        index = changed[0]
        classes = self.classes[index]
        if (
            classes is None
            or before[index] not in _POLICIES
            or after[index] not in _POLICIES
        ):
            return False
        return (
            classes[_POLICIES.index(before[index])]
            == classes[_POLICIES.index(after[index])]
        )


class PolicyUses:
    def __init__(self, config: Config) -> None:
        self.config_key = _config_key(config)
        self.uses: list[list[tuple[str, ...]]] = [
            [] for _ in config.load_eviction_policies
        ]
        self.device_function: object | None = None
        self.unknown: set[int] = set()
        # Whole-plan decisions can consume policies independently of load spelling.
        self.disabled = (
            bool(config.config.get("cute_loop_vectorize"))
            or config.config.get("cute_grid_work_order", "current") != "current"
        )

    def record(
        self, owner: object, index: int, classes: tuple[str, ...] | None
    ) -> None:
        if self.device_function is None:
            self.device_function = owner
        elif self.device_function is not owner:
            # Includes retry/fallback passes and multi-function generation. Keeping
            # their candidates is safer than combining different ordinal domains.
            self.disabled = True
        if not 0 <= index < len(self.uses):
            self.disabled = True
        elif classes is None:
            self.unknown.add(index)
        else:
            self.uses[index].append(classes)

    def finish(self, identity: str | None, source: str) -> PolicyReceipt | None:
        if identity is None or self.disabled:
            return None
        result: list[tuple[str, ...] | None] = []
        for index, uses in enumerate(self.uses):
            if not uses or index in self.unknown:
                result.append(None)
            else:
                # Intersection of equivalence relations from every actual use.
                signatures = list(zip(*uses, strict=True))
                result.append(
                    tuple(str(signatures.index(value)) for value in signatures)
                )
        return PolicyReceipt(
            identity,
            self.config_key,
            hashlib.sha256(source.encode()).hexdigest(),
            tuple(result),
        )


@contextmanager
def collect(config: Config) -> Iterator[PolicyUses]:
    uses = PolicyUses(config)
    token = _ACTIVE.set(uses)
    try:
        yield uses
    finally:
        _ACTIVE.reset(token)


def record_load(owner: object, index: int, *, scalar_bytes: int | None) -> None:
    uses = _ACTIVE.get()
    if uses is None:
        return
    classes = None
    if scalar_bytes is not None:
        classes = (
            ("",) * len(_POLICIES)
            if scalar_bytes < 4
            else ("", "first", "last", "streaming", "", "", "")
        )
    uses.record(owner, index, classes)


def receipt_for(bound: BoundKernel, config: Config) -> PolicyReceipt | None:
    receipt = bound._cute_eviction_policy_receipts.get(_config_key(config))
    if receipt is None or authority(bound) != receipt.authority:
        return None
    return receipt


def export_receipt(
    bound: BoundKernel, config: Config, source: str
) -> dict[str, object] | None:
    """Export only evidence produced for this exact generated source."""
    receipt = receipt_for(bound, config)
    if (
        receipt is None
        or receipt.source_sha256 != hashlib.sha256(source.encode()).hexdigest()
    ):
        return None
    return {
        "version": 1,
        "authority": receipt.authority,
        "config": receipt.config_key,
        "source": receipt.source_sha256,
        "classes": receipt.classes,
    }


def import_receipt(
    bound: BoundKernel, config: Config, source: str, payload: object
) -> PolicyReceipt | None:
    """Validate optional worker evidence without changing source admission."""
    if type(payload) is not dict:
        return None
    if set(payload) != {"version", "authority", "config", "source", "classes"}:
        return None
    if type(payload["version"]) is not int or payload["version"] != 1:
        return None
    key = _config_key(config)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if payload["config"] != key or payload["source"] != digest:
        return None
    classes = payload["classes"]
    if type(classes) is not list or len(classes) != len(config.load_eviction_policies):
        return None
    decoded: list[tuple[str, ...] | None] = []
    for row in classes:
        if row is None:
            decoded.append(None)
            continue
        if type(row) is not list or len(row) != len(_POLICIES):
            return None
        if any(
            type(value) is not str
            or value not in {str(i) for i in range(len(_POLICIES))}
            for value in row
        ):
            return None
        # finish() names each class by its first policy position.
        if row != [str(row.index(value)) for value in row]:
            return None
        decoded.append(tuple(row))
    identity = authority(bound)
    if identity is None or payload["authority"] != identity:
        return None
    return PolicyReceipt(identity, key, digest, tuple(decoded))
