from __future__ import annotations

import ast
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import IdentityFragment
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import NumThreadsFragment
import helion.language as hl


class _CachedDefaultFragment(IdentityFragment):
    def __init__(self, value: int = 3) -> None:
        super().__init__(value)
        self.calls = 0
        self.state = {"opaque": object()}
        self.state["cycle"] = self.state

    def default(self) -> int | float | bool:
        self.calls += 1
        return self.value


class _ValueFragment(IdentityFragment):
    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self.value == other.value


_EXTERNAL_STATE: list[Any] = []

HIGH_LITERAL_GLOBAL = 128


def _changing_helper() -> NumThreadsFragment:
    return NumThreadsFragment(_EXTERNAL_STATE.pop(0))


def _mutating_helper() -> NumThreadsFragment:
    if not _EXTERNAL_STATE:
        fragment = NumThreadsFragment(32)
        _EXTERNAL_STATE.append(fragment)
        return fragment
    _EXTERNAL_STATE[0].high = 64
    return NumThreadsFragment(32)


class _ChangingConstructor(IdentityFragment):
    def __init__(self) -> None:
        super().__init__(_EXTERNAL_STATE.pop(0))


def _closed_thread_helper() -> NumThreadsFragment:
    return NumThreadsFragment(HIGH_LITERAL_GLOBAL)


def _stored_thread_helper(high: int) -> NumThreadsFragment:
    fragment = NumThreadsFragment(128)
    fragment.high = high
    return fragment


def _helper_fragment_domain() -> EnumFragment:
    return EnumFragment((2, 16))


@helion.kernel(autotune_effort="none")
def global_thread_fragment(
    x: torch.Tensor, form: hl.constexpr, high: int
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if form == "closed":
            fragment = _closed_thread_helper()
        elif form == "stored":
            fragment = _stored_thread_helper(HIGH_LITERAL_GLOBAL)
        elif form == "derived":
            fragment = _stored_thread_helper(HIGH_LITERAL_GLOBAL + 64)
        else:
            fragment = _stored_thread_helper(high)
        value = hl.register_tunable("value", fragment)
        out[tile] = x[tile] + value
    return out


@helion.kernel(autotune_effort="none")
def grid_fragment(x: torch.Tensor, kind: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if kind == "bool":
            fragment = BooleanFragment()
        elif kind == "threads":
            fragment = NumThreadsFragment(128)
        elif kind == "float":
            fragment = EnumFragment((2.0, 4.0))
        elif kind == "custom":
            fragment = IdentityFragment(value=3)
        elif kind == "value_custom":
            fragment = _ValueFragment(value=3)
        elif kind == "helper":
            fragment = _changing_helper()
        elif kind == "changing_custom":
            fragment = _ChangingConstructor()
        elif kind == "mutating_helper":
            fragment = _mutating_helper()
        else:
            fragment = EnumFragment((2, 16))
        value = hl.register_tunable("value", fragment)
        out[tile] = x[tile] + value
    return out


def _shared_revisit_helper() -> ConfigSpecFragment:
    fragment, field, remaining = _EXTERNAL_STATE
    if remaining:
        setattr(fragment, field, remaining.pop(0))
    return fragment


@helion.kernel(autotune_effort="none")
def shared_revisit_fragment(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = hl.register_tunable("value", _shared_revisit_helper())
        out[tile] = x[tile] + value
    return out


@helion.kernel(autotune_effort="none")
def shared_host_fragment(x: torch.Tensor) -> torch.Tensor:
    first = hl.register_tunable("value", _shared_revisit_helper())
    second = hl.register_tunable("value", _shared_revisit_helper())
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


def _fragment_pair(kind: object) -> tuple[ConfigSpecFragment, ConfigSpecFragment]:
    if kind == "bool":
        return BooleanFragment(), BooleanFragment()
    if kind == "custom":
        return IdentityFragment(), IdentityFragment()
    high = _EXTERNAL_STATE.pop(0) if _EXTERNAL_STATE else 128
    return NumThreadsFragment(high), NumThreadsFragment(high)


@helion.kernel(autotune_effort="none")
def container_fragment_revisit(
    x: torch.Tensor, kind: hl.constexpr, unpack: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if unpack:
            fragment, unused_fragment = _fragment_pair(kind)
        else:
            fragment = _fragment_pair(kind)[0]
        value = hl.register_tunable("value", fragment)
        out[tile] = x[tile] + value
    return out


def _return_builtin_with_added_state() -> ConfigSpecFragment:
    fragment, changes = _EXTERNAL_STATE
    if changes.pop(0):
        fragment.extra = 1
    return fragment


@helion.kernel(autotune_effort="none")
def added_state_declarations(x: torch.Tensor) -> torch.Tensor:
    first = hl.register_tunable("value", _return_builtin_with_added_state())
    second = hl.register_tunable("value", _return_builtin_with_added_state())
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


def check_shared_fragment_revisit(kind: str, mutate: bool) -> None:
    if kind == "threads":
        fragment, field, first, second = NumThreadsFragment(32), "high", 32, 64
    elif kind == "enum":
        fragment, field, first, second = EnumFragment((2, 4)), "choices", (2, 4), (4, 8)
    else:
        cls = _CachedDefaultFragment if kind == "cached_custom" else IdentityFragment
        fragment, field, first, second = cls(3), "value", 3, 7
    remaining = [first, second if mutate else first]
    with (
        patch.dict(
            shared_revisit_fragment.fn.__globals__,
            _EXTERNAL_STATE=[fragment, field, remaining],
        ),
        cpu_render_target(),
    ):
        if mutate:
            with pytest.raises(
                helion.exc.TypeInferenceError,
                match="Config fragment changed during compilation; keep registered domains immutable",
            ):
                # pyrefly: ignore [bad-argument-type]
                cpu_bind(shared_revisit_fragment, (torch.empty(37),))
            assert getattr(fragment, field) == second
        else:
            # pyrefly: ignore [bad-argument-type]
            bound = cpu_bind(shared_revisit_fragment, (torch.empty(37),))
            captured = bound.config_spec.user_defined_tunables["value"]
            assert (captured is fragment) is (kind in ("custom", "cached_custom"))
            assert getattr(captured, field) == first
            ast.parse(bound.to_code(bound.config_spec.default_config()))
            if kind == "cached_custom":
                # pyrefly: ignore [missing-attribute]
                assert captured.calls == fragment.calls > 0
                # pyrefly: ignore [missing-attribute]
                assert captured.state["cycle"] is captured.state
        assert not remaining


@helion.kernel(autotune_effort="none")
def helper_returned_fragment(x: torch.Tensor, on_device: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    if not on_device:
        host_value = hl.register_tunable("helper_factor", _helper_fragment_domain())
    for tile in hl.tile(x.numel()):
        if on_device:
            value = hl.register_tunable("helper_factor", _helper_fragment_domain())
        else:
            # pyrefly: ignore [unbound-name]
            value = host_value
        out[tile] = x[tile] + value
    return out
