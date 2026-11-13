from __future__ import annotations

from bisect import bisect_left
import dataclasses
import enum
import math
import random
import struct
from typing import TYPE_CHECKING
from typing import Iterable
from typing import TypeAlias
from typing import TypeGuard
from typing import cast

from ..exc import InvalidConfig
from ..runtime.config import _config_value_key

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Callable

    from . import ConfigSpec


FragmentFingerprint: TypeAlias = tuple[str | int, ...]


def _same_search_value(left: object, right: object) -> bool:
    """Search choices follow compiled identity, not Python numeric equality."""
    return _config_value_key(left) == _config_value_key(right)


def integer_power_of_two(n: object) -> TypeGuard[int]:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def assert_integer_power_of_two(n: object) -> int:
    if integer_power_of_two(n):
        return n
    raise InvalidConfig(f"Expected integer power of two, got {n}")


class Category(enum.Enum):
    UNSET = enum.auto()
    BLOCK_SIZE = enum.auto()
    NUM_WARPS = enum.auto()


class ConfigSpecFragment:
    def category(self) -> Category:
        return Category.UNSET

    def default(self) -> object:
        """Return the default value for this fragment."""
        raise NotImplementedError

    def random(self) -> object:
        """Return the default value for this fragment."""
        raise NotImplementedError

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        """Return neighbors for PatternSearch."""
        raise NotImplementedError

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        """Create a new value by combining a, b, and c with something like: `a + (b - c)`"""
        if b == c:
            return a
        return self.random()

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> object:
        return fn(self)

    def is_block_size(self) -> bool:
        return False

    def dim(self) -> int:
        """
        Returns the dimension of the output of encode
        """
        raise NotImplementedError

    def encode(self, value: object) -> list[float]:
        """
        Encode a configuration value into a list of floats for ML models.

        This is used by surrogate-assisted algorithms to convert configurations
        into numerical vectors for prediction models.

        Args:
            value: The configuration value to encode.

        Returns:
            A list of floats representing the encoded value.
        """
        raise NotImplementedError

    def _flat_key_info(self) -> tuple[int, bool]:
        """Return (num_flat_entries, is_sequence) for flat_key_layout().

        A scalar fragment is a single tunable parameter, so it always
        occupies exactly 1 flat config slot and is never a sequence.
        """
        return (1, False)

    def fingerprint(self) -> FragmentFingerprint:
        """Return structural metadata for this fragment used in ConfigSpec fingerprinting."""
        return ()

    def get_minimum(self) -> int:
        """
        Return the minimum allowed value for this fragment.
        """
        raise NotImplementedError

    def cardinality(self) -> int | None:
        """Number of distinct values this fragment can take during search.

        Returns ``None`` when the count is unbounded or unknown. Used by the
        search-space logger to describe the size of a tunable dimension.
        """
        return None

    def search_values(self, limit: int = 100) -> list[object] | None:
        """Explicit distinct search values when cheaply enumerable within ``limit``.

        Returns ``None`` when the values aren't usefully enumerable or exceed
        ``limit``.
        """
        return None


@dataclasses.dataclass
class PermutationFragment(ConfigSpecFragment):
    length: int

    def default(self) -> list[int]:
        return [*range(self.length)]

    def random(self) -> list[int]:
        return random.sample(range(self.length), self.length)

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        sequence = list(cast("Iterable[int]", current))
        if len(sequence) != self.length:
            raise ValueError(
                f"Expected permutation of length {self.length}, got {len(sequence)}"
            )
        if {*sequence} != {*range(self.length)}:
            raise ValueError(
                f"Expected permutation of range({self.length}), got {sequence!r}"
            )
        neighbors: list[object] = []
        for i in range(self.length):
            for j in range(i + 1, self.length):
                swapped = [*sequence]
                swapped[i], swapped[j] = swapped[j], swapped[i]
                neighbors.append(swapped)
        return neighbors

    def dim(self) -> int:
        return self.length

    def cardinality(self) -> int | None:
        return math.factorial(self.length)

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, list)
        encoded = []
        for val in value:
            assert isinstance(val, int)
            encoded.append(float(val))
        return encoded


@dataclasses.dataclass
class BaseIntegerFragment(ConfigSpecFragment):
    low: int  # minimum value (inclusive)
    high: int  # maximum value (inclusive)
    default_val: int

    def __init__(self, low: int, high: int, default_val: int | None = None) -> None:
        self.low = low
        self.high = high
        if default_val is None:
            default_val = low
        self.default_val = default_val

    def default(self) -> int:
        return self.clamp(self.default_val)

    def clamp(self, val: int) -> int:
        return max(min(val, self.high), self.low)

    def get_minimum(self) -> int:
        return self.low

    def dim(self) -> int:
        return 1

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if type(current) is not int:  # bool is not allowed
            raise TypeError(f"Expected int, got {type(current).__name__}")
        if type(radius) is not int or radius < 1:
            raise ValueError(f"Expected positive int radius, got {radius!r}")
        lower = max(self.low, current - radius)
        upper = min(self.high, current + radius)
        return [v for v in range(lower, upper + 1) if v != current]

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, int)
        return [float(value)]

    def cardinality(self) -> int | None:
        return self.high - self.low + 1

    def search_values(self, limit: int = 100) -> list[object] | None:
        card = self.cardinality()
        if card is None or card > limit:
            return None
        return list(range(self.low, self.high + 1))


class PowerOfTwoFragment(BaseIntegerFragment):
    def _power_values(self) -> list[int]:
        assert_integer_power_of_two(self.low)
        assert_integer_power_of_two(self.high)
        return [
            1 << exponent
            for exponent in range(self.low.bit_length() - 1, self.high.bit_length())
        ]

    def random(self) -> int:
        assert_integer_power_of_two(self.low)
        assert_integer_power_of_two(self.high)
        return 2 ** random.randrange(self.low.bit_length() - 1, self.high.bit_length())

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if type(current) is not int or current <= 0:
            raise TypeError(f"Expected positive power-of-two int, got {current!r}")
        if type(radius) is not int or radius < 1:
            raise ValueError(f"Expected positive int radius, got {radius!r}")

        assert_integer_power_of_two(self.high)
        assert_integer_power_of_two(self.low)
        choices = self._power_values()
        if current not in choices:
            insertion = bisect_left(choices, current)
            return cast(
                "list[object]",
                choices[
                    max(0, insertion - radius) : min(len(choices), insertion + radius)
                ],
            )

        cur_exp = current.bit_length() - 1
        low_exp = self.low.bit_length() - 1
        high_exp = self.high.bit_length() - 1
        lower = max(low_exp, cur_exp - radius)
        upper = min(high_exp, cur_exp + radius)
        return [2**e for e in range(lower, upper + 1) if e != cur_exp]

    def differential_mutation(self, a: object, b: object, c: object) -> int:
        assert isinstance(b, int)
        assert isinstance(c, int)
        if type(a) is not int or a <= 0:
            raise TypeError(f"Expected positive int, got {a!r}")
        choices = self._power_values()
        if a not in choices:
            insertion = bisect_left(choices, a)
            lower = choices[max(0, insertion - 1)]
            upper = choices[min(len(choices) - 1, insertion)]
            if b < c:
                return lower
            if b > c:
                return upper
            return min((lower, upper), key=lambda value: (abs(value - a), value))

        ai = assert_integer_power_of_two(a)
        # TODO(jansel): should we take more than one step at a time?
        # the logic of *2 or //2 is we are dealing with rather small ranges and overflows are likely
        if b < c:
            return self.clamp(ai // 2)
        if b > c:
            return self.clamp(ai * 2)
        return ai

    def encode(self, value: object) -> list[float]:
        """Encode power-of-2 values using log2 transformation."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected int/float for PowerOfTwoFragment, got {type(value).__name__}: {value!r}"
            )
        if value <= 0:
            raise ValueError(
                f"Expected positive value for PowerOfTwoFragment, got {value}"
            )
        return [math.log2(float(value))]

    def cardinality(self) -> int | None:
        return self.high.bit_length() - self.low.bit_length() + 1

    def search_values(self, limit: int = 100) -> list[object] | None:
        values = self._power_values()
        if len(values) > limit:
            return None
        return list(values)


class IntegerFragment(BaseIntegerFragment):
    def random(self) -> int:
        return random.randint(self.low, self.high)

    def differential_mutation(self, a: object, b: object, c: object) -> int:
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(c, int)
        # TODO(jansel): should we take more than one step at a time?
        # the logic of +/- 1 is we are dealing with rather small ranges and overflows are likely
        if b < c:
            return self.clamp(a - 1)
        if b > c:
            return self.clamp(a + 1)
        return a


@dataclasses.dataclass
class EnumFragment(ConfigSpecFragment):
    choices: tuple[object, ...]
    search_choices: tuple[object, ...] | None = None
    coverage_choices: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if self.search_choices is not None:
            if not self.search_choices:
                raise ValueError("search_choices must not be empty")
            invalid = [
                choice for choice in self.search_choices if choice not in self.choices
            ]
            if invalid:
                raise ValueError(
                    f"search_choices must be a subset of choices, got {invalid!r}"
                )
        if self.coverage_choices is not None:
            if not self.coverage_choices:
                raise ValueError("coverage_choices must not be empty")
            active_choices = self._active_choices()
            invalid = [
                choice
                for choice in self.coverage_choices
                if choice not in active_choices
            ]
            if invalid:
                raise ValueError(
                    "coverage_choices must be a subset of active search choices, "
                    f"got {invalid!r}"
                )

    def _active_choices(self) -> tuple[object, ...]:
        return self.choices if self.search_choices is None else self.search_choices

    def default(self) -> object:
        return self.choices[0]

    def random(self) -> object:
        return random.choice(self._active_choices())

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        # `current` can be outside `choices` when config normalization rewrote
        # the knob to a value off the searched surface (e.g. cute tcgen05
        # knobs on configs that opt out of tcgen05); every searched choice is
        # then a neighbor so the search can step back onto the surface.
        return [
            choice
            for choice in self._active_choices()
            if not _same_search_value(choice, current)
        ]

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        active_choices = self._active_choices()
        if _same_search_value(b, c):
            if not any(_same_search_value(a, choice) for choice in active_choices):
                return self.random()
            return a
        choices = [
            choice
            for choice in (b, c)
            if any(_same_search_value(choice, active) for active in active_choices)
        ]
        if not choices:
            return self.random()
        choices = [choice for choice in choices if not _same_search_value(a, choice)]
        if not choices:
            return self.random()
        return random.choice(choices)

    def dim(self) -> int:
        return len(self.choices)

    def cardinality(self) -> int | None:
        return len(self._active_choices())

    def search_values(self, limit: int = 100) -> list[object] | None:
        active = self._active_choices()
        if len(active) > limit:
            return None
        return list(active)

    def fingerprint(self) -> FragmentFingerprint:
        result = ["enum", *(repr(choice) for choice in self.choices)]
        if self.search_choices is not None:
            result.extend(("search", *(repr(choice) for choice in self.search_choices)))
        if self.coverage_choices is not None:
            result.extend(
                ("coverage", *(repr(choice) for choice in self.coverage_choices))
            )
        return tuple(result)

    def encode(self, value: object) -> list[float]:
        """Encode enum values as a one-hot vector.

        Values outside ``choices`` encode as all zeros rather than raising:
        config normalization can legally rewrite a knob to a value outside
        the searched surface (e.g. cute tcgen05 knobs on configs that opt
        out of tcgen05), and this encoding only feeds surrogate models.
        """
        choice_idx = next(
            (
                i
                for i, choice in enumerate(self.choices)
                if _same_search_value(choice, value)
            ),
            -1,
        )
        return [1.0 if i == choice_idx else 0.0 for i in range(len(self.choices))]


class BooleanFragment(ConfigSpecFragment):
    def default(self) -> bool:
        return False

    def random(self) -> bool:
        return random.choice((False, True))

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if type(current) is not bool:
            raise TypeError(f"Expected bool, got {type(current).__name__}")
        return [not current]

    def differential_mutation(self, a: object, b: object, c: object) -> bool:
        assert isinstance(a, bool)
        if b is c:
            return a
        return not a

    def dim(self) -> int:
        return 1

    def cardinality(self) -> int | None:
        return 2

    def search_values(self, limit: int = 100) -> list[object] | None:
        return [False, True]

    def encode(self, value: object) -> list[float]:
        """Encode enum values as their index."""
        assert isinstance(value, bool)
        return [1.0] if value else [0.0]


class BlockSizeFragment(PowerOfTwoFragment):
    def category(self) -> Category:
        return Category.BLOCK_SIZE


class NumWarpsFragment(PowerOfTwoFragment):
    def category(self) -> Category:
        return Category.NUM_WARPS


class NumThreadsFragment(ConfigSpecFragment):
    """CuTe launch-thread count for one tile axis.

    The value ``0`` means "auto": let the CuTe backend derive a thread count
    from the selected block size and shrink it as needed for the 1024-thread
    CTA limit. Randomly sampled positive values are powers of two. Explicit
    heuristic seeds may use any whole-warp multiple and are repaired against
    the paired block size by ConfigGeneration before benchmarking.
    """

    def __init__(self, high: int) -> None:
        self.high = assert_integer_power_of_two(max(high, 1))

    def default(self) -> int:
        return 0

    def random(self) -> int:
        if random.random() < 0.25:
            return 0
        return PowerOfTwoFragment(1, self.high, self.high).random()

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if current == 0:
            return [1] if self.high == 1 else [1, self.high]
        if type(current) is not int or current < 1:
            raise TypeError(f"Expected positive int, got {current!r}")
        neighbors = PowerOfTwoFragment(1, self.high, self.high).pattern_neighbors(
            current, radius
        )
        return [0, *neighbors]

    def differential_mutation(self, a: object, b: object, c: object) -> int:
        if b == c:
            return cast("int", a)
        if a == 0 or b == 0 or c == 0:
            return self.random()
        return PowerOfTwoFragment(1, self.high, self.high).differential_mutation(
            a, b, c
        )

    def dim(self) -> int:
        return 1

    def cardinality(self) -> int | None:
        # "0" (auto) plus every power of two up to ``high``.
        return 1 + self.high.bit_length()

    def search_values(self, limit: int = 100) -> list[object] | None:
        values: list[object] = [0, *(1 << e for e in range(self.high.bit_length()))]
        if len(values) > limit:
            return None
        return values

    def encode(self, value: object) -> list[float]:
        if value == 0:
            return [0.0]
        if not isinstance(value, int):
            raise TypeError(
                f"Expected int for NumThreadsFragment, got {type(value).__name__}: {value!r}"
            )
        return [math.log2(float(value)) + 1.0]

    def get_minimum(self) -> int:
        return 0


@dataclasses.dataclass
class ListOf(ConfigSpecFragment):
    """Wrapper that creates a list of independently tunable fragments.

    Example:
        ListOf(EnumFragment(choices=("a", "b", "c")), length=5)
        creates a list of 5 independently tunable enum values.
    """

    inner: ConfigSpecFragment
    length: int

    def default(self) -> list[object]:
        """Return a list of default values."""
        return [self.inner.default() for _ in range(self.length)]

    def random(self) -> list[object]:
        """Return a list of random values."""
        return [self.inner.random() for _ in range(self.length)]

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        """Return neighbors by changing one element at a time."""
        if not isinstance(current, list) or len(current) != self.length:
            raise ValueError(f"Expected list of length {self.length}, got {current!r}")

        neighbors: list[object] = []
        # For each position, try all neighbors from the inner fragment
        for i in range(self.length):
            for neighbor_value in self.inner.pattern_neighbors(current[i], radius):
                neighbor = current.copy()
                neighbor[i] = neighbor_value
                neighbors.append(neighbor)
        # Also propose uniform lists (every element set to the same value):
        # element-at-a-time moves can require crossing worse mixed
        # configurations (e.g. flipping five memory ops from pointer to
        # tensor_descriptor one by one), while the uniform move jumps straight
        # across the valley.
        if self.length > 1:
            values: list[object] = []
            for value in [
                self.inner.default(),
                *self.inner.pattern_neighbors(self.inner.default(), radius),
            ]:
                if not any(_same_search_value(value, prior) for prior in values):
                    values.append(value)
            for value in values:
                uniform = [value] * self.length
                if not _same_search_value(uniform, current) and not any(
                    _same_search_value(uniform, prior) for prior in neighbors
                ):
                    neighbors.append(uniform)
        return neighbors

    def differential_mutation(self, a: object, b: object, c: object) -> list[object]:
        """Create a new value by combining a, b, and c element-wise."""
        assert isinstance(a, list) and len(a) == self.length
        assert isinstance(b, list) and len(b) == self.length
        assert isinstance(c, list) and len(c) == self.length

        return [
            self.inner.differential_mutation(a[i], b[i], c[i])
            for i in range(self.length)
        ]

    def fingerprint(self) -> FragmentFingerprint:
        return (self.length, *self.inner.fingerprint())

    def dim(self) -> int:
        return self.length * self.inner.dim()

    def cardinality(self) -> int | None:
        inner = self.inner.cardinality()
        if inner is None:
            return None
        return inner**self.length

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, list)
        encoded = []
        for v in value:
            encoded.extend(self.inner.encode(v))
        return encoded


_BUILTIN_TUNABLE_FRAGMENTS = (
    BaseIntegerFragment,
    IntegerFragment,
    PowerOfTwoFragment,
    BlockSizeFragment,
    NumWarpsFragment,
    BooleanFragment,
    NumThreadsFragment,
    EnumFragment,
)

_FRAGMENT_CHANGED_MESSAGE = (
    "Config fragment changed during compilation; keep registered domains immutable."
)


def _fragment_domain_fields(
    fragment: ConfigSpecFragment,
) -> tuple[tuple[str, object], ...] | None:
    """Declared built-in domain fields, separate from flat-layout identity.

    Subclasses retain these common fields; custom extra domain state belongs
    in their existing fingerprint() implementation.
    """
    if isinstance(fragment, BaseIntegerFragment):
        return (
            ("low", fragment.low),
            ("high", fragment.high),
            ("default_val", fragment.default_val),
        )
    if isinstance(fragment, NumThreadsFragment):
        return (("high", fragment.high),)
    if isinstance(fragment, BooleanFragment):
        return ()
    if isinstance(fragment, EnumFragment):
        return (
            ("choices", fragment.choices),
            ("search_choices", fragment.search_choices),
            ("coverage_choices", fragment.coverage_choices),
        )
    return None


def _plain_builtin_fragment_fields(
    fragment: ConfigSpecFragment,
) -> tuple[tuple[str, object], ...] | None:
    """Complete domain fields for exact built-ins with no extra instance state.

    Unlike persistent fingerprint fields, these fields can replace object
    equality only when they describe the whole instance. Subclasses and
    instances with additional state retain their existing equality policy.
    Values are neither normalized nor copied here.
    """
    if type(fragment) not in _BUILTIN_TUNABLE_FRAGMENTS:
        return None
    fields = _fragment_domain_fields(fragment)
    assert fields is not None
    if set(vars(fragment)) != {name for name, _ in fields}:
        return None
    return fields


def _plain_builtin_fragment_key(
    fragment: ConfigSpecFragment,
    *,
    resolve: Callable[[object], object] | None = None,
    for_identity_equality: bool = False,
) -> Hashable | None:
    """Canonical concrete domain shared by binding, revisits and constexpr keys.

    A revisit may resolve a recorded literal before registration snapshots it.
    Other callers already have concrete fields. Extra state and subclasses
    continue to use their own equality rather than this value key. Identity
    equality replacement is limited to the two built-ins without value equality
    and their original concrete domains; constexpr keys encode all plain fields.
    """
    if for_identity_equality and type(fragment) not in (
        BooleanFragment,
        NumThreadsFragment,
    ):
        return None
    fields = _plain_builtin_fragment_fields(fragment)
    if fields is None:
        return None
    if resolve is not None:
        fields = tuple((name, resolve(value)) for name, value in fields)
    if (
        for_identity_equality
        and type(fragment) is NumThreadsFragment
        and type(fields[0][1]) is not int
    ):
        return None
    return _immutable_fragment_key(fields)


def _scalar_fragment_key(value: object) -> Hashable | None:
    """Exact scalar type and value, including float zero signs and NaN bits."""
    kind = type(value)
    if kind in (bool, int):
        return kind, value
    if kind is float:
        value = cast("float", value)
        # hex() erases NaN signs and payloads. Keep all other encodings stable.
        return float, struct.pack("!d", value) if math.isnan(value) else value.hex()
    return None


def _same_scalar(left: object, right: object) -> bool:
    key = _scalar_fragment_key(left)
    return key is not None and key == _scalar_fragment_key(right)


def _immutable_fragment_key(value: object) -> Hashable | None:
    """Encode only plain immutable metadata; never retain opaque objects."""
    kind = type(value)
    if value is None:
        return (type(None),)
    if kind in (bool, int, float):
        return _scalar_fragment_key(value)
    if kind is str:
        return str, value
    if kind in (tuple, list):
        result: list[Hashable] = []
        for item in cast("tuple[object, ...] | list[object]", value):
            key = _immutable_fragment_key(item)
            if key is None:
                return None
            result.append(key)
        return kind, tuple(result)
    if kind is dict:
        entries: list[tuple[str | int, Hashable]] = []
        for item, field in cast("dict[object, object]", value).items():
            if type(item) not in (str, int):
                return None
            key = _immutable_fragment_key(field)
            if key is None:
                return None
            entries.append((cast("str | int", item), key))
        try:
            entries.sort()
        except TypeError:
            return None
        return dict, tuple(entries)
    return None
