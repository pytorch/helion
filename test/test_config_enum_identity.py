"""Search operations retain distinct scalar choices with Python-equal values."""

from __future__ import annotations

import pytest

import helion
from helion._compiler.backend_registry import get_backend
from helion._testing.cute_codegen import cpu_target
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_spec import ConfigSpec
from helion.runtime.settings import _get_backend

pytestmark = pytest.mark.cpu_only


def _tag(value: object) -> tuple[type, str]:
    return type(value), value.hex() if type(value) is float else repr(value)


# Each pair exercises a different compiler distinction; each method is a
# separate search consumer, so a passing neighbor check cannot prove encoding.
@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_enum_neighbors_keep_python_equal_alternative(
    values: tuple[object, object],
) -> None:
    fragment = EnumFragment(values)
    assert [_tag(v) for v in fragment.pattern_neighbors(values[0])] == [_tag(values[1])]
    assert [_tag(v) for v in fragment.pattern_neighbors(values[1])] == [_tag(values[0])]


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_enum_encodes_distinct_scalar_choices(values: tuple[object, object]) -> None:
    fragment = EnumFragment(values)
    assert fragment.encode(values[0]) == [1.0, 0.0]
    assert fragment.encode(values[1]) == [0.0, 1.0]


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_enum_mutation_keeps_changed_choice(values: tuple[object, object]) -> None:
    fragment = EnumFragment(values)
    assert _tag(
        fragment.differential_mutation(values[0], values[0], values[1])
    ) == _tag(values[1])


@pytest.mark.parametrize("implicit_backend", [False, True])
def test_enum_round_trip_and_uniform_list_neighbor(
    monkeypatch, implicit_backend
) -> None:
    if implicit_backend:
        monkeypatch.delenv("HELION_BACKEND", raising=False)
    values = (True, 1, 1.0, -0.0, 0.0)
    with cpu_target():
        spec = ConfigSpec(
            backend=get_backend(_get_backend()),
            target_device_capability=(10, 0),
            user_defined_tunables={"selected": EnumFragment(values)},
        )
        generation = spec.create_config_generation()
        for value in values:
            flat = generation.flatten(helion.Config(selected=value))
            assert _tag(generation.unflatten(flat)["selected"]) == _tag(value)
    fragment = ListOf(EnumFragment((-0.0, 0.0)), 2)
    neighbors = fragment.pattern_neighbors([-0.0, -0.0])
    assert [[_tag(v) for v in row] for row in neighbors] == [
        [_tag(0.0), _tag(-0.0)],
        [_tag(-0.0), _tag(0.0)],
        [_tag(0.0), _tag(0.0)],
    ]


def test_enum_search_preserves_opaque_value_equality() -> None:
    class Choice:
        def __init__(self, name: str) -> None:
            self.name = name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Choice) and self.name == other.name

    first, equivalent, other = Choice("first"), Choice("first"), Choice("other")
    fragment = EnumFragment((first, other))
    assert fragment.encode(equivalent) == [1.0, 0.0]
    assert fragment.pattern_neighbors(equivalent) == [other]
    assert fragment.differential_mutation(equivalent, first, other) is other
