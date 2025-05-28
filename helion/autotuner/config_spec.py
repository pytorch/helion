from __future__ import annotations

import dataclasses
import functools
import operator
from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.runtime.triton_heuristics import get_max_y_grid

from ..exc import InvalidConfig
from .config_fragment import BlockSizeFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import NumWarpsFragment
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_fragment import assert_integer_power_of_two
import helion
from helion._compat import supports_tensor_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "loop_orders",
        "reduction_loops",
        "num_warps",
        "num_stages",
        "l2_grouping",
        "use_yz_grid",
        "indexing",
    ]
)


@dataclasses.dataclass
class ConfigSpec:
    block_size_specs: list[BlockSizeSpec] = dataclasses.field(default_factory=list)
    reduction_loop_specs: list[ReductionLoopSpec] = dataclasses.field(
        default_factory=list
    )

    def loop_order_specs(self) -> Sequence[PermutationFragment]:
        """Return the specs for the loop orders."""
        return [
            PermutationFragment(len(spec))
            for spec in self.block_size_specs
            if spec.allow_reorder
        ]

    def normalize(self, config: helion.Config | dict[str, object]) -> None:
        """Normalize the config to match the block_sizes and validate the config."""
        if isinstance(config, helion.Config):
            self.normalize(config.config)
            return

        for name in ("block_size", "loop_order", "reduction_loop"):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                config[names] = [config.pop(name)]

        config["block_sizes"] = self.normalize_block_sizes(
            config.get("block_sizes", None)
        )
        config["loop_orders"] = self.normalize_loop_orders(
            config.get("loop_orders", None)
        )
        if not config["loop_orders"]:
            config.pop("loop_orders")
        config["reduction_loops"] = self.normalize_reduction_loops(
            config.get("reduction_loops", None)
        )
        if not config["reduction_loops"]:
            config.pop("reduction_loops")
        config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        config.setdefault("num_stages", DEFAULT_NUM_STAGES)
        # TODO(jansel): include num_ctas and max_nreg

        if self.allow_l2_grouping:
            config.setdefault("l2_grouping", 1)
        if self.allow_use_yz_grid:
            config.setdefault("use_yz_grid", False)
        config.setdefault("indexing", "pointer")
        if invalid_keys := ({*config} - VALID_KEYS):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    @property
    def allow_l2_grouping(self) -> bool:
        return (
            len(self.block_size_specs) > 0
            and self.block_size_specs[0].allow_l2_grouping
        )

    @property
    def allow_use_yz_grid(self) -> bool:
        return (
            len(self.block_size_specs) > 0
            and 1 < len(self.block_size_specs[0]) <= 3
            and all(
                s < get_max_y_grid() for s in self.block_size_specs[0].size_hints[1:]
            )
        )

    def normalize_block_sizes(self, block_sizes: object) -> list[int | list[int]]:
        if len(self.block_size_specs) == 0:
            if block_sizes:
                raise InvalidConfig("block_sizes should be empty")
            return []
        if not block_sizes or not isinstance(block_sizes, (list, tuple)):
            raise InvalidConfig("block_sizes must be set to a list")
        idx = 0
        new_block_sizes: list[int | list[int]] = []
        for block_spec in self.block_size_specs:
            expected = len(block_spec)
            if idx >= len(block_sizes):
                raise InvalidConfig(
                    f"Not enough block sizes, expected {sum(map(len, self.block_size_specs))}, got {len(block_sizes)}"
                )
            val = block_sizes[idx]
            if (
                expected > 1
                and len(block_sizes[idx:]) == expected
                and block_spec is self.block_size_specs[-1]
            ):
                new_block_sizes.append(
                    [*map(assert_integer_power_of_two, block_sizes[idx:])]
                )
                idx += expected
            elif isinstance(val, int):
                if len(block_spec) == 1:
                    # go down the more general NDTileStrategy path
                    new_block_sizes.append([assert_integer_power_of_two(val)])
                else:
                    if not block_spec.can_be_int():
                        raise InvalidConfig(f"Block sizes must be list, got {val!r}")
                    new_block_sizes.append(assert_integer_power_of_two(val))
                idx += 1
            elif isinstance(val, (list, tuple)):
                if len(val) != expected:
                    raise InvalidConfig(f"Block size {idx} length {expected}: {val!r}")
                new_block_sizes.append([*map(assert_integer_power_of_two, val)])
                idx += 1
            else:
                raise InvalidConfig(f"Block size must be int/list, got {val!r}")
        if len(block_sizes) != idx:
            raise InvalidConfig(f"Extra block sizes, used {idx} of {len(block_sizes)}")
        return new_block_sizes

    def normalize_loop_orders(self, loop_orders: object) -> list[list[int]]:
        assert isinstance(loop_orders, (list, tuple, type(None)))
        loop_orders = [*(loop_orders or ())]
        specs = self.loop_order_specs()
        if len(loop_orders) > len(specs):
            raise InvalidConfig(
                f"Too many loop orders, expected {len(specs)}, got {len(loop_orders)}"
            )
        for i, spec in enumerate(specs):
            if i < len(loop_orders):
                loop_orders[i] = spec.normalize(loop_orders[i])
            else:
                loop_orders.append(spec.default())
        return loop_orders

    def normalize_reduction_loops(self, reduction_loops: object) -> list[int | None]:
        assert isinstance(reduction_loops, (list, tuple, type(None), int))
        loops = [spec for spec in self.reduction_loop_specs if spec.allow_loop]
        if reduction_loops is None:
            reduction_loops = [None for _ in loops]
        elif isinstance(reduction_loops, int):
            reduction_loops = [reduction_loops]
        if len(reduction_loops) != len(loops):
            raise InvalidConfig(
                f"Invalid number of reduction loops, expected {len(loops)} got {len(reduction_loops)}"
            )
        return [
            spec.normalize(value)
            for spec, value in zip(loops, reduction_loops, strict=True)
        ]

    def default_config(self) -> helion.Config:
        return self.flat_config(lambda x: x.default())

    def flat_config(self, fn: Callable[[ConfigSpecFragment], object]) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        total_ndim = sum([len(spec) for spec in self.block_size_specs])
        reduction_numel = product(
            [next_power_of_2(spec.size_hint) for spec in self.reduction_loop_specs]
        )
        config = {
            "block_sizes": (
                block_sizes := [
                    spec.flat_block_sizes(fn, total_ndim, reduction_numel)
                    for spec in self.block_size_specs
                ]
            ),
            "loop_orders": [
                spec.flat_loop_orders(fn)
                for spec in self.block_size_specs
                if spec.allow_reorder
            ],
            "reduction_loops": [
                spec.flat_reduction_loop(fn)
                for spec in self.reduction_loop_specs
                if spec.allow_loop
            ],
            "num_warps": fn(NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS)),
            "num_stages": fn(IntegerFragment(1, 8, DEFAULT_NUM_STAGES)),
            "indexing": fn(
                EnumFragment(
                    ("pointer", "block_ptr", "tensor_descriptor")
                    if supports_tensor_descriptor()
                    else ("pointer", "block_ptr")
                )
            ),
        }
        if not config["loop_orders"]:
            config.pop("loop_orders")
        if not config["reduction_loops"]:
            config.pop("reduction_loops")
        if self.allow_l2_grouping:
            config["l2_grouping"] = fn(PowerOfTwoFragment(1, 64, 1))
        if self.allow_use_yz_grid:
            use_yz_grid = fn(BooleanFragment())
            if config.get("l2_grouping", 1) == 1 and isinstance(block_sizes[0], list):
                config["use_yz_grid"] = use_yz_grid
        return helion.Config(**config)  # pyre-ignore[6]


class BlockSizeSpec:
    def __init__(
        self,
        size_hints: list[int],
        allow_flattened: bool,
        allow_reorder: bool,
        allow_l2_grouping: bool,
    ) -> None:
        self.size_hints = size_hints
        self.allow_flattened = allow_flattened
        self.allow_reorder = allow_reorder
        self.allow_l2_grouping = allow_l2_grouping
        self.min_sizes: list[int] = [1 for _ in size_hints]
        self.max_sizes: list[int] = [next_power_of_2(s) for s in size_hints]

    def __repr__(self) -> str:
        fields = [repr(self.size_hints)]
        for name in ("allow_flattened", "allow_reorder", "allow_l2_grouping"):
            if value := getattr(self, name):
                fields.append(f"{name}={value}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def update_min(self, i: int, value: int) -> None:
        self.min_sizes[i] = assert_integer_power_of_two(max(value, self.min_sizes[i]))

    def update_max(self, i: int, value: int) -> None:
        self.max_sizes[i] = assert_integer_power_of_two(min(value, self.max_sizes[i]))

    def update_hint(self, i: int, value: int) -> None:
        self.size_hints[i] = value
        self.update_max(i, next_power_of_2(value))

    def can_be_int(self) -> bool:
        return len(self.size_hints) == 1 or self.allow_flattened

    def __len__(self) -> int:
        return len(self.size_hints)

    def numel_hint(self) -> int:
        return product(self.size_hints)

    def flat_block_sizes(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        total_ndim: int,
        reduction_numel: int,
    ) -> object:
        """We turn the more complex list[int]|int config into smaller fragments that are easier to autotune over."""
        if total_ndim == 1 and reduction_numel == 1:
            default = 1024
        elif total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        block_sizes = [
            fn(BlockSizeFragment(low, high, default))
            for low, high in zip(self.min_sizes, self.max_sizes, strict=True)
        ]
        if self.allow_flattened:
            should_flatten = fn(BooleanFragment())
            flat_block_size = fn(
                BlockSizeFragment(
                    next_power_of_2(product(self.min_sizes)),
                    next_power_of_2(self.numel_hint()),
                    1024,
                )
            )
            if should_flatten:
                return flat_block_size
        return block_sizes

    def flat_loop_orders(self, fn: Callable[[ConfigSpecFragment], object]) -> object:
        assert self.allow_reorder
        return fn(PermutationFragment(len(self.size_hints)))


@dataclasses.dataclass
class ReductionLoopSpec:
    size_hint: int
    allow_loop: bool

    def normalize(self, value: int | None) -> int | None:
        if value is None:
            return None
        assert_integer_power_of_two(value)
        if value < 0 or value >= next_power_of_2(self.size_hint):
            raise InvalidConfig(
                f"Invalid reduction loop value {value!r}, expected 0 to {next_power_of_2(self.size_hint)}"
            )
        return value

    def flat_reduction_loop(self, fn: Callable[[ConfigSpecFragment], object]) -> object:
        assert self.allow_loop
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(self.size_hint)
        default = min(high, 4096)
        value = fn(BlockSizeFragment(low, high, default))
        if value == high:
            return None  # max size becomes persistent reduction
        return value


def product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
