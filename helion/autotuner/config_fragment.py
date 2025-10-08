from __future__ import annotations

import dataclasses
import enum
import random
from typing import Iterable
from typing import TypeGuard
from typing import cast

from ..exc import InvalidConfig


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

    def pattern_neighbors(self, current: object) -> list[object]:
        """Return neighbors for PatternSearch."""
        raise NotImplementedError

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        """Create a new value by combining a, b, and c with something like: `a + (b - c)`"""
        if b == c:
            return a
        return self.random()

    def is_block_size(self) -> bool:
        return False

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

    def pattern_neighbors(self, current: object) -> list[object]:
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

    def pattern_neighbors(self, current: object) -> list[object]:
        if type(current) is not int:  # bool is not allowed
            raise TypeError(f"Expected int, got {type(current).__name__}")
        neighbors: list[object] = []
        lower = current - 1
        upper = current + 1
        if lower >= self.low:
            neighbors.append(lower)
        if upper <= self.high:
            neighbors.append(upper)
        return neighbors


class PowerOfTwoFragment(BaseIntegerFragment):
    def random(self) -> int:
        assert_integer_power_of_two(self.low)
        assert_integer_power_of_two(self.high)
        return 2 ** random.randrange(self.low.bit_length() - 1, self.high.bit_length())

    def pattern_neighbors(self, current: object) -> list[object]:
        if type(current) is not int or current <= 0:
            raise TypeError(f"Expected positive power-of-two int, got {current!r}")
        neighbors: list[object] = []
        lower = current // 2
        if lower >= self.low:
            neighbors.append(lower)
        upper = current * 2
        if upper <= self.high:
            neighbors.append(upper)
        return neighbors

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

    def default(self) -> object:
        return self.choices[0]

    def random(self) -> object:
        return random.choice(self.choices)

    def pattern_neighbors(self, current: object) -> list[object]:
        if current not in self.choices:
            raise ValueError(f"{current!r} not a valid choice")
        return [choice for choice in self.choices if choice != current]

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        if b == c:
            return a
        choices = [b, c]
        if a in choices:
            choices.remove(a)
        return random.choice(choices)


class BooleanFragment(ConfigSpecFragment):
    def default(self) -> bool:
        return False

    def random(self) -> bool:
        return random.choice((False, True))

    def pattern_neighbors(self, current: object) -> list[object]:
        if type(current) is not bool:
            raise TypeError(f"Expected bool, got {type(current).__name__}")
        return [not current]

    def differential_mutation(self, a: object, b: object, c: object) -> bool:
        assert isinstance(a, bool)
        if b is c:
            return a
        return not a


class BlockSizeFragment(PowerOfTwoFragment):
    def category(self) -> Category:
        return Category.BLOCK_SIZE


class NumWarpsFragment(PowerOfTwoFragment):
    def category(self) -> Category:
        return Category.NUM_WARPS


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

    def pattern_neighbors(self, current: object) -> list[object]:
        """Return neighbors by changing one element at a time."""
        if not isinstance(current, list) or len(current) != self.length:
            raise ValueError(f"Expected list of length {self.length}, got {current!r}")

        neighbors: list[object] = []
        # For each position, try all neighbors from the inner fragment
        for i in range(self.length):
            for neighbor_value in self.inner.pattern_neighbors(current[i]):
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
