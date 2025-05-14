from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torch._dynamo.source import AttrSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import GlobalSource
from torch._dynamo.source import LocalSource

if TYPE_CHECKING:
    from torch._guards import Source

    from .host_function import HostFunction
    from .source_location import SourceLocation


@dataclasses.dataclass
class Origin:
    """Keeps track of where a variable came from."""

    def is_host(self) -> bool:
        """
        Check if the origin is a host.
        """
        return issubclass(self.base_type(), HostOrigin)

    def is_global(self) -> bool:
        """
        Check if the origin is a global variable.

        :return: True if the origin is from a global variable, False otherwise.
        """
        return issubclass(self.base_type(), GlobalOrigin)

    def is_argument(self) -> bool:
        """
        Check if the origin is an argument.

        :return: True if the origin is from an argument, False otherwise.
        """
        return issubclass(self.base_type(), ArgumentOrigin)

    def is_device(self) -> bool:
        """
        Check if the origin is a device.

        :return: True if the origin is a device, False otherwise.
        """
        return not self.is_host()

    def base_type(self) -> type[Origin]:
        """
        Get the base type of the origin, unwrapping things like attributes.

        :return: The base type of the origin.
        """
        return type(self)

    def needs_rename(self) -> bool:
        """
        Check if the origin needs to be renamed (globals and closures).

        :return: True if the origin needs to be renamed, False otherwise.
        """
        return self.is_global()

    def depth(self) -> int:
        """
        Get the depth of the origin.

        :return: The depth of the origin, which is 1 by default and increases each wrapper.
        """
        return 1

    def host_str(self) -> str:
        """
        Get a string representation of the host origin.

        :raises NotImplementedError: Always raises this error as it should be implemented by subclasses.
        """
        raise NotImplementedError(type(self).__name__)

    def suggest_var_name(self) -> str:
        """
        Suggest a variable name based on the origin.

        :raises NotImplementedError: Always raises this error as it should be implemented by subclasses.
        """
        raise NotImplementedError(type(self).__name__)

    def to_source(self) -> Source:
        """Convert to a PyTorch source object."""
        raise NotImplementedError(type(self).__name__)


@dataclasses.dataclass
class HostOrigin(Origin):
    pass


@dataclasses.dataclass
class NameOrigin(HostOrigin):
    """A variable that came from an ast.Name node."""

    name: str

    def __init__(self, name: str, function: HostFunction | None = None) -> None:
        super().__init__()
        self.name = name

    def host_str(self) -> str:
        return self.name

    def suggest_var_name(self) -> str:
        return self.name


class BuiltinOrigin(NameOrigin):
    def to_source(self) -> Source:
        return GlobalSource(self.name)


class GlobalOrigin(NameOrigin):
    def to_source(self) -> Source:
        return GlobalSource(self.name)


class ArgumentOrigin(NameOrigin):
    def to_source(self) -> Source:
        return LocalSource(self.name, is_input=True)


@dataclasses.dataclass
class WrappedOrigin(Origin):
    """Keeps track of where a variable came from."""

    value: Origin

    def base_type(self) -> type[Origin]:
        return self.value.base_type()

    def needs_rename(self) -> bool:
        return self.value.needs_rename()

    def depth(self) -> int:
        return 1 + self.value.depth()


@dataclasses.dataclass
class AttributeOrigin(WrappedOrigin):
    key: str

    def host_str(self) -> str:
        return f"{self.value.host_str()}.{self.key}"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_attr_{self.key}"

    def to_source(self) -> Source:
        return AttrSource(self.value.to_source(), self.key)


@dataclasses.dataclass
class GetItemOrigin(WrappedOrigin):
    key: int | str

    def host_str(self) -> str:
        return f"{self.value.host_str()}[{self.key!r}]"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_item_{self.key}"

    def to_source(self) -> Source:
        return GetItemSource(self.value.to_source(), self.key)


@dataclasses.dataclass
class TensorSizeOrigin(WrappedOrigin):
    key: int

    def host_str(self) -> str:
        return f"{self.value.host_str()}.size({self.key!r})"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_size_{self.key}"

    def to_source(self) -> Source:
        return GetItemSource(AttrSource(self.value.to_source(), "shape"), self.key)


@dataclasses.dataclass
class ClosureOrigin(WrappedOrigin):
    key: int

    def needs_rename(self) -> bool:
        return True

    def host_str(self) -> str:
        return f"{self.value.host_str()}.__closure__[{self.key!r}].cell_contents"

    def suggest_var_name(self) -> str:
        return f"{self.value.suggest_var_name()}_closure_{self.key}"

    def to_source(self) -> Source:
        return AttrSource(
            GetItemSource(AttrSource(self.value.to_source(), "__closure__"), self.key),
            "cell_contents",
        )


@dataclasses.dataclass
class SourceOrigin(HostOrigin):
    location: SourceLocation


@dataclasses.dataclass
class DeviceOrigin(Origin):
    location: SourceLocation


@dataclasses.dataclass
class BlockSizeOrigin(Origin):
    block_size_idx: int

    def host_str(self) -> str:
        """
        Get the host-side string representation of a block size variable.
        If the block size variable was not created (e.g., block size == 1),
        return the literal '1'.
        """
        from .device_function import DeviceFunction

        # Look up the block size variable name; if not set (e.g., size==1), use literal 1
        var = DeviceFunction.current().block_size_var(self.block_size_idx)
        if var is None:
            return "1"
        return var


@dataclasses.dataclass
class ReductionDimensionOrigin(Origin):
    rdim_idx: int

    def host_str(self) -> str:
        raise NotImplementedError
