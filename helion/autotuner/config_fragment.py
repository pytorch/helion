from __future__ import annotations

import dataclasses
import enum
import random
from typing import TypeGuard

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

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        """Create a new value by combining a, b, and c with something like: `a + (b - c)`"""
        if b == c:
            return a
        return self.random()

    def is_block_size(self) -> bool:
        return False


@dataclasses.dataclass
class PermutationFragment(ConfigSpecFragment):
    length: int

    def normalize(self, ordering: object) -> list[int]:
        if type(ordering) is not list:
            if not isinstance(ordering, tuple):
                raise InvalidConfig(f"ordering must be a list: {ordering!r}")
            ordering = [*ordering]
        if len(ordering) != self.length:
            raise InvalidConfig(
                f"Expected {self.length} permutations, got {len(ordering)}"
            )
        if {*ordering} != {*range(self.length)}:
            raise InvalidConfig(f"Invalid permutation {ordering!r}")
        return ordering

    def default(self) -> list[int]:
        return [*range(self.length)]

    def random(self) -> list[int]:
        return random.sample(range(self.length), self.length)


@dataclasses.dataclass
class BaseIntegerFragment(ConfigSpecFragment):
    low: int  # minimum value (inclusive)
    high: int  # maximum value (inclusive)
    default_val: int

    def default(self) -> int:
        return self.clamp(self.default_val)

    def clamp(self, val: int) -> int:
        return max(min(val, self.high), self.low)


class PowerOfTwoFragment(BaseIntegerFragment):
    def random(self) -> int:
        assert_integer_power_of_two(self.low)
        assert_integer_power_of_two(self.high)
        return 2 ** random.randrange(self.low.bit_length() - 1, self.high.bit_length())

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

    def differential_mutation(self, a: object, b: object, c: object) -> object:
        if b == c:
            return a
        for candidate in random.sample(self.choices, 2):
            if candidate != a:
                return candidate
        return self.random()  # only reachable with duplicate choices


class BooleanFragment(ConfigSpecFragment):
    def default(self) -> bool:
        return False

    def random(self) -> bool:
        return random.choice((False, True))

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
