from __future__ import annotations

import dataclasses
import enum
import math
import random
from typing import TYPE_CHECKING
from typing import Iterable
from typing import TypeAlias
from typing import TypeGuard
from typing import cast

from ..exc import InvalidConfig

if TYPE_CHECKING:
    from typing import Callable

    from . import ConfigSpec
    from .config_generation import ConfigGeneration


FragmentFingerprint: TypeAlias = tuple[str | int, ...]


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


class PowerOfTwoFragment(BaseIntegerFragment):
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
        assert_integer_power_of_two(current)

        cur_exp = current.bit_length() - 1
        low_exp = self.low.bit_length() - 1
        high_exp = self.high.bit_length() - 1
        lower = max(low_exp, cur_exp - radius)
        upper = min(high_exp, cur_exp + radius)
        return [2**e for e in range(lower, upper + 1) if e != cur_exp]

    def differential_mutation(self, a: object, b: object, c: object) -> int:
        ai = assert_integer_power_of_two(a)
        assert isinstance(b, int)
        assert isinstance(c, int)
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

    def __post_init__(self) -> None:
        if self.search_choices is None:
            return
        if not self.search_choices:
            raise ValueError("search_choices must not be empty")
        invalid = [
            choice for choice in self.search_choices if choice not in self.choices
        ]
        if invalid:
            raise ValueError(
                f"search_choices must be a subset of choices, got {invalid!r}"
            )

    def _active_choices(self) -> tuple[object, ...]:
        return self.choices if self.search_choices is None else self.search_choices

    def default(self) -> object:
        return self.choices[0]

    def random(self) -> object:
        return random.choice(self._active_choices())

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if current not in self.choices:
            raise ValueError(f"{current!r} not a valid choice")
        return [choice for choice in self._active_choices() if choice != current]

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        active_choices = self._active_choices()
        if b == c:
            if a not in active_choices:
                return self.random()
            return a
        choices = [choice for choice in (b, c) if choice in active_choices]
        if not choices:
            return self.random()
        if a in choices:
            choices.remove(a)
        if not choices:
            return self.random()
        return random.choice(choices)

    def dim(self) -> int:
        return len(self.choices)

    def fingerprint(self) -> FragmentFingerprint:
        if self.search_choices is None:
            return ("enum", *(repr(choice) for choice in self.choices))
        return (
            "enum",
            *(repr(choice) for choice in self.choices),
            "search",
            *(repr(choice) for choice in self.search_choices),
        )

    def encode(self, value: object) -> list[float]:
        """Encode enum values as their index."""
        try:
            choice_idx = self.choices.index(value)
        except ValueError:
            raise ValueError(
                f"Invalid enum value {value!r} for EnumFragment. "
                f"Valid choices: {self.choices}"
            ) from None
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
    CTA limit. Positive values are powers of two and are repaired against the
    paired block size by ConfigGeneration before benchmarking.
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
        assert_integer_power_of_two(current)
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

    def encode(self, value: object) -> list[float]:
        if value == 0:
            return [0.0]
        if not isinstance(value, int):
            raise TypeError(
                f"Expected int for NumThreadsFragment, got {type(value).__name__}: {value!r}"
            )
        assert_integer_power_of_two(value)
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

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, list)
        encoded = []
        for v in value:
            encoded.extend(self.inner.encode(v))
        return encoded


@dataclasses.dataclass
class PerDimListOf(ConfigSpecFragment):
    """Like ListOf but each position has its own fragment with distinct choices.

    ListOf repeats a single inner fragment for every position, so all
    dimensions share the same valid values.  PerDimListOf holds a separate
    fragment per dimension, allowing each to have a different choice set
    (e.g. grid folding factors filtered by per-dim num_blocks).
    """

    fragments: list[ConfigSpecFragment]

    def default(self) -> list[object]:
        return [f.default() for f in self.fragments]

    def random(self) -> list[object]:
        return [f.random() for f in self.fragments]

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if not isinstance(current, list) or len(current) != len(self.fragments):
            raise ValueError(
                f"Expected list of length {len(self.fragments)}, got {current!r}"
            )
        neighbors: list[object] = []
        for i, frag in enumerate(self.fragments):
            for neighbor_value in frag.pattern_neighbors(current[i], radius):
                neighbor = current.copy()
                neighbor[i] = neighbor_value
                neighbors.append(neighbor)
        return neighbors

    def differential_mutation(self, a: object, b: object, c: object) -> list[object]:
        assert isinstance(a, list) and len(a) == len(self.fragments)
        assert isinstance(b, list) and len(b) == len(self.fragments)
        assert isinstance(c, list) and len(c) == len(self.fragments)
        return [
            self.fragments[i].differential_mutation(a[i], b[i], c[i])
            for i in range(len(self.fragments))
        ]

    def fingerprint(self) -> tuple[int, ...]:
        return (len(self.fragments),)

    def dim(self) -> int:
        return sum(f.dim() for f in self.fragments)

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, list)
        encoded = []
        for v, frag in zip(value, self.fragments, strict=True):
            encoded.extend(frag.encode(v))
        return encoded


@dataclasses.dataclass
class DynamicGridFoldingFragment(ConfigSpecFragment):
    """Grid folding fragment with generation-aware choice filtering.

    This fragment dynamically filters allowed folding factors based on
    the current search generation. Early generations are restricted to
    conservative folding (factor 0), while later generations can explore
    higher folding factors.

    Attributes:
        valid_factors: All possible folding factors (0, -1, 2, 4, 8, 16, 32, 64)
        min_generation: Generation threshold before allowing non-zero factors
        max_factor: Maximum allowed folding factor after min_generation
        _config_gen: Reference to ConfigGeneration for accessing current generation
    """

    valid_factors: tuple[int, ...]
    min_generation: int
    max_factor: int
    _config_gen: object | None = None  # ConfigGeneration, set later

    def _get_allowed_choices(self) -> tuple[int, ...]:
        """Compute allowed folding factors based on current generation."""

        if self._config_gen is None:
            # Fallback: if config_gen not set, allow all factors (shouldn't happen in normal use)
            return self.valid_factors

        config_gen = cast("ConfigGeneration", self._config_gen)
        current_gen = config_gen.current_generation

        if current_gen < self.min_generation:
            # Early generations: only allow no folding (factor 0)
            # Partial folding factors (2, 4, 8, ...) and full folding (-1) are restricted
            return (0,)

        # Later generations: allow factors up to max_factor
        if self.max_factor == 0:
            return (0,)
        if self.max_factor == -1:
            # No limit, use heuristics (all positive factors allowed)
            return self.valid_factors
        # Filter to factors <= max_factor
        return tuple(
            f for f in self.valid_factors if f <= 0 or (f > 0 and f <= self.max_factor)
        )

    def default(self) -> int:
        return self._get_allowed_choices()[0]

    def random(self) -> int:
        import random

        return random.choice(self._get_allowed_choices())

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        allowed = self._get_allowed_choices()
        if current not in allowed:
            # If current is not in allowed set, return all allowed choices
            return list(allowed)
        return [choice for choice in allowed if choice != current]

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        import random

        allowed = self._get_allowed_choices()
        if b == c:
            return a if a in allowed else allowed[0]
        choices = [x for x in (b, c) if x in allowed]
        if not choices:
            return random.choice(allowed)
        if a in choices:
            choices = [x for x in allowed if x not in choices]
            if not choices:
                return random.choice(allowed)
        return random.choice(choices) if choices else random.choice(allowed)

    def dim(self) -> int:
        return len(self.valid_factors)

    def encode(self, value: object) -> list[float]:
        """Encode folding factor as its index in valid_factors."""
        try:
            choice_idx = self.valid_factors.index(value)
        except ValueError:
            raise ValueError(
                f"Invalid folding factor {value!r}. Valid factors: {self.valid_factors}"
            ) from None
        return [1.0 if i == choice_idx else 0.0 for i in range(len(self.valid_factors))]
