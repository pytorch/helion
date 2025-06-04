from __future__ import annotations

from collections.abc import MutableSequence
import dataclasses
import functools
import operator
from typing import TYPE_CHECKING
from typing import TypeVar

from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch.fx.node import map_aggregate

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

    _T = TypeVar("_T")
    _D = TypeVar("_D")


DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "loop_orders",
        "l2_groupings",
        "reduction_loops",
        "flatten_loops",
        "num_warps",
        "num_stages",
        "use_yz_grid",
        "indexing",
    ]
)


@dataclasses.dataclass
class _BlockIdItem:
    # the block_indices used in the IR
    block_ids: list[int]

    @property
    def block_id(self) -> int:
        """Return the first block_id for this item."""
        return self.block_ids[0]

    def _fill_missing(self) -> object:
        """Provide a value when not provided by the user."""
        raise NotImplementedError

    def _normalize(self, name: str, value: object) -> object:
        """Validate and normalize the value for this item."""
        raise NotImplementedError

    def _fragment(self, base: ConfigSpec) -> ConfigSpecFragment:
        """Return the fragment used for autotunging for this item."""
        raise NotImplementedError

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> object:
        return fn(self._fragment(base))


_BlockIdItemT = TypeVar("_BlockIdItemT", bound=_BlockIdItem)


class BlockIdSequence(MutableSequence[_BlockIdItemT]):
    """
    A mapping from block_id to item that keeps track of the index of
    each item in the config for O(1) config lookups.
    """

    def __init__(self) -> None:
        self._data: list[_BlockIdItemT] = []
        self._block_id_to_index: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._data)

    def _reindex(self) -> None:
        """Rebuild the mapping from block_id to index."""
        new_index = {}
        for i, item in enumerate(self._data):
            for block_id in item.block_ids:
                new_index[block_id] = i
        self._block_id_to_index = new_index

    def __getitem__(self, index: int) -> _BlockIdItemT:
        return self._data[index]

    def __setitem__(self, index: int, value: _BlockIdItemT) -> None:
        self._data[index] = value
        self._reindex()  # could be faster, but uncommon case

    def __delitem__(self, index: int) -> None:
        del self._data[index]
        self._reindex()  # could be faster, but uncommon case

    def clear(self) -> None:
        self._data.clear()
        self._block_id_to_index.clear()

    def append(self, value: _BlockIdItemT) -> None:
        """Append a new item to the end of the sequence."""
        index = len(self._data)
        self._data.append(value)
        for block_id in value.block_ids:
            self._block_id_to_index[block_id] = index

    def insert(self, index: int, value: _BlockIdItemT) -> None:
        """Insert a new item at the given index."""
        if index == len(self._data):
            self.append(value)
            return
        self._data.insert(index, value)
        self._reindex()  # could be faster, but uncommon case

    def block_id_to_index(self, block_id: int) -> int:
        """Return the index of the block_id in the config."""
        return self._block_id_to_index[block_id]

    def block_id_lookup(self, block_id: int) -> _BlockIdItemT:
        """Return the index of the block_id in the config."""
        return self._data[self._block_id_to_index[block_id]]

    def disable_block_id(self, block_id: int) -> None:
        """Remove configuration choice for the given block_id."""
        self._data = [x for x in self._data if block_id not in x.block_ids]
        self._reindex()

    def config_get(
        self, config: list[_T], block_id: int, default: _D = None
    ) -> _T | _D:
        """
        Get the config value for the given block_id, or return default if not found.
        """
        index = self._block_id_to_index.get(block_id, None)
        if index is None:
            return default
        return config[index]

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> list[object]:
        """Map a flattened version of the config using the given function."""
        return [spec._flat_config(base, fn) for spec in self._data]

    def _normalize(
        self, name: str, values: object, *, flatten: bool = False
    ) -> list[object]:
        """Validate and normalize the values for this config item."""
        if flatten:
            if values is None:
                values = ()
            new_values = []
            # pyre-ignore[6]
            map_aggregate(values, new_values.append)
            values = new_values
        elif not isinstance(values, (list, tuple, type(None))):
            raise InvalidConfig(
                f"Unexpected type for config[{name!r}], expected list or None, got {type(values).__name__}"
            )
        values = [*(values or ())]
        size = len(self)
        if len(values) > size:
            raise InvalidConfig(
                f"Too many values for config[{name!r}], expected {size}, got {len(values)}"
            )
        if len(values) < size:
            try:
                for spec in self._data[len(values) :]:
                    values.append(spec._fill_missing())
            except NotImplementedError:
                raise InvalidConfig(
                    f"Not enough values for config[{name!r}], expected {size}, got {len(values)}"
                ) from None
        for i, spec in enumerate(self._data):
            values[i] = spec._normalize(f"config[{name}][{i}]", values[i])
        return values

    def _remove_duplicates(self) -> None:
        new_specs = []
        for spec in self:
            other = self.block_id_lookup(spec.block_id)
            if other is spec:
                new_specs.append(spec)
            elif len(spec.block_ids) != len(other.block_ids):
                # this will cause invalid config errors with loop orders
                # remove them both
                self.disable_block_id(spec.block_id)
                self._remove_duplicates()  # start over
                return
        if len(new_specs) != len(self):
            self._data = new_specs
            self._reindex()


@dataclasses.dataclass
class ConfigSpec:
    block_sizes: BlockIdSequence[BlockSizeSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    loop_orders: BlockIdSequence[LoopOrderSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    l2_groupings: BlockIdSequence[L2GroupingSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    flatten_loops: BlockIdSequence[FlattenLoopSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    reduction_loops: BlockIdSequence[ReductionLoopSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    allow_use_yz_grid: bool | None = None

    def _remove_duplicates(self) -> None:
        self.loop_orders._remove_duplicates()
        self.l2_groupings._remove_duplicates()
        self.flatten_loops._remove_duplicates()

    def normalize(self, config: helion.Config | dict[str, object]) -> None:
        """Normalize the config to match the block_sizes and validate the config."""
        if isinstance(config, helion.Config):
            self.normalize(config.config)
            return

        for name in (
            "block_size",
            "loop_order",
            "reduction_loop",
            "l2_grouping",
            "flatten_loop",
        ):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                config[names] = [config.pop(name)]

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("flatten_loops", self.flatten_loops, True),
            ("l2_groupings", self.l2_groupings, True),
            ("loop_orders", self.loop_orders, False),
            ("reduction_loops", self.reduction_loops, True),
        ]:
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )

        for name in ("loop_orders", "l2_groupings", "flatten_loops", "reduction_loops"):
            if not config[name]:
                config.pop(name)

        config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        config.setdefault("num_stages", DEFAULT_NUM_STAGES)
        # TODO(jansel): include num_ctas and max_nreg

        if self.allow_use_yz_grid:
            config.setdefault("use_yz_grid", False)

        config.setdefault("indexing", "pointer")
        if invalid_keys := ({*config} - VALID_KEYS):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def default_config(self) -> helion.Config:
        return self.flat_config(lambda x: x.default())

    def flat_config(self, fn: Callable[[ConfigSpecFragment], object]) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        config = {
            "block_sizes": self.block_sizes._flat_config(self, fn),
            "loop_orders": self.loop_orders._flat_config(self, fn),
            "flatten_loops": self.flatten_loops._flat_config(self, fn),
            "l2_groupings": self.l2_groupings._flat_config(self, fn),
            "reduction_loops": self.reduction_loops._flat_config(self, fn),
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
        if self.allow_use_yz_grid:
            use_yz_grid = fn(BooleanFragment())
            # pyre-ignore[16]
            if (not config["l2_groupings"] or config["l2_groupings"][0] == 1) and (
                not config["flatten_loops"] or not config["flatten_loops"][0]
            ):
                config["use_yz_grid"] = use_yz_grid
        for name in ("loop_orders", "flatten_loops", "reduction_loops", "l2_groupings"):
            if not config[name]:
                config.pop(name)
        return helion.Config(**config)  # pyre-ignore[6]


class LoopOrderSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PermutationFragment:
        return PermutationFragment(len(self.block_ids))

    def _normalize(self, name: str, value: object) -> list[int]:
        if type(value) is not list:
            if not isinstance(value, tuple):
                raise InvalidConfig(f"{name} must be a list, got {value!r}")
            value = [*value]
        length = len(self.block_ids)
        if len(value) != length:
            raise InvalidConfig(f"{name} must be length {length}, got {len(value)}")
        if {*value} != {*range(length)}:
            raise InvalidConfig(f"{name} must be permutation, got {value!r}")
        return value

    def _fill_missing(self) -> list[int]:
        """Provide a value when not provided by the user."""
        return [*range(len(self.block_ids))]


class _PowerOfTwoBlockIdItem(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int | None:
        try:
            return assert_integer_power_of_two(value)
        except InvalidConfig:
            raise InvalidConfig(
                f"{name} must be a power of two, got {value!r}"
            ) from None


class L2GroupingSpec(_PowerOfTwoBlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        return PowerOfTwoFragment(1, 64, 1)

    def _fill_missing(self) -> int:
        return 1


class BlockSizeSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        min_size: int = 1,
        max_size: int | None = None,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint
        self.min_size: int = min_size
        self.max_size: int = (
            next_power_of_2(size_hint) if max_size is None else max_size
        )

    def __repr__(self) -> str:
        fields = []
        for field, default in (
            ("block_id", None),
            ("size_hint", None),
            ("min_size", 1),
            ("max_size", next_power_of_2(self.size_hint)),
        ):
            value = getattr(self, field)
            if value != default:
                fields.append(f"{field}={value!r}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))

    def update_max(self, value: int) -> None:
        self.max_size = assert_integer_power_of_two(min(value, self.max_size))

    def update_hint(self, value: int) -> None:
        self.size_hint = value
        self.update_max(next_power_of_2(value))

    def _fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        total_ndim = len(base.block_sizes)
        reduction_numel = _product(
            [next_power_of_2(spec.size_hint) for spec in base.reduction_loops]
        )
        if total_ndim <= 1 and reduction_numel <= 1:
            default = 1024
        elif total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        return BlockSizeFragment(
            self.min_size,
            self.max_size,
            default,
        )


class FlattenLoopSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}") from None
        return value

    def _fill_missing(self) -> bool:
        return False


class ReductionLoopSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> int | None:
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(self.size_hint)
        default = min(high, 4096)
        value = fn(BlockSizeFragment(low, high, default))
        assert isinstance(value, int)
        if value >= self.size_hint:
            return None  # max size becomes persistent reduction
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        return super()._normalize(name, value)

    def _fill_missing(self) -> None:
        return None


def _product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
