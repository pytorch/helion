from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Callable
from typing import MutableSequence
from typing import TypeVar

from torch.fx.node import map_aggregate

from ..exc import InvalidConfig
from .config_fragment import ConfigSpecFragment
from .config_fragment import assert_integer_power_of_two

if TYPE_CHECKING:
    from . import ConfigSpec

    _T = TypeVar("_T")
    _D = TypeVar("_D")


@dataclasses.dataclass
class _BlockIdItem:
    # the block_ids used in the IR
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
    A sequence of _BlockIdItem subclasses that allows for efficient
    mapping from block_id to index in the sequence.  A generic data
    structure used to store different types of configuration specs.
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

    def __getitem__(self, index: int) -> _BlockIdItemT:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._data[index]

    def __setitem__(self, index: int, value: _BlockIdItemT) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self._data[index] = value
        self._reindex()  # could be faster, but uncommon case

    def __delitem__(self, index: int) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def valid_block_ids(self) -> list[int]:
        """Return the list of valid block_ids."""
        return list(self._block_id_to_index.keys())

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

    def _reset_config_to_default(
        self, name: str, values: object, *, block_ids: list[int] | None = None
    ) -> list[object]:
        """Set the config values to the default values. If block_ids is provided, only set those values."""
        if not values:
            return []
        assert isinstance(values, list)
        assert len(values) == len(self)

        if block_ids is None:
            block_ids = self.valid_block_ids()
        for block_id in block_ids:
            if block_id not in self._block_id_to_index:
                continue
            index = self._block_id_to_index[block_id]
            values[index] = self._data[index]._fill_missing()
        return values

    def _normalize(
        self, name: str, values: object, *, flatten: bool = False
    ) -> list[object]:
        """Validate and normalize the values for this config item."""
        if flatten:
            if values is None:
                values = ()
            new_values = []

            map_aggregate(values, new_values.append)  # pyright: ignore[reportArgumentType]
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
                    f"Not enough values for config[{name!r}]: expected {size} block sizes "
                    f"(one for each tiled dimension), got {len(values)}. "
                    f"Did you forget to specify block sizes for all your hl.tile() dimensions?"
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


class _PowerOfTwoBlockIdItem(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int | None:
        try:
            return assert_integer_power_of_two(value)
        except InvalidConfig:
            raise InvalidConfig(
                f"{name} must be a power of two, got {value!r}"
            ) from None
