from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch._guards import Source

    from .host_function import HostFunction
    from .source_location import SourceLocation

    # PyTorch's runtime classes lack type stubs; define them here so pyrefly sees
    # the correct signatures.
    class AttrSource(Source):
        def __init__(self, base: Source, member: str) -> None: ...

    class GetItemSource(Source):
        def __init__(
            self, base: Source, index: object, index_is_slice: bool = False
        ) -> None: ...

    class GlobalSource(Source):
        def __init__(self, global_name: str) -> None: ...

    class LocalSource(Source):
        def __init__(
            self,
            local_name: str,
            is_input: bool = False,
            dynamism: frozenset[str] | None = None,
            is_derefed_cell_contents: bool = False,
        ) -> None: ...
else:
    from torch._dynamo.source import AttrSource
    from torch._dynamo.source import GetItemSource
    from torch._dynamo.source import GlobalSource
    from torch._dynamo.source import LocalSource


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

        Returns:
            bool: True if the origin is from a global variable, False otherwise.
        """
        return issubclass(self.base_type(), GlobalOrigin)

    def is_argument(self) -> bool:
        """
        Check if the origin is an argument.

        Returns:
            bool: True if the origin is from an argument, False otherwise.
        """
        return issubclass(self.base_type(), ArgumentOrigin)

    def is_device(self) -> bool:
        """
        Check if the origin is a device.

        Returns:
            bool: True if the origin is a device, False otherwise.
        """
        return not self.is_host()

    def base_type(self) -> type[Origin]:
        """
        Get the base type of the origin, unwrapping things like attributes.

        Returns:
            type[Origin]: The base type of the origin.
        """
        return type(self)

    def needs_rename(self) -> bool:
        """
        Check if the origin needs to be renamed (globals and closures).

        Returns:
            bool: True if the origin needs to be renamed, False otherwise.
        """
        return self.is_global()

    def depth(self) -> int:
        """
        Get the depth of the origin.

        Returns:
            int: The depth of the origin, which is 1 by default and increases each wrapper.
        """
        return 1

    def host_str(self) -> str:
        """
        Get a string representation of the host origin.

        Raises:
            NotImplementedError: Always raises this error as it should be implemented by subclasses.
        """
        raise NotImplementedError(type(self).__name__)

    def suggest_var_name(self) -> str:
        """
        Suggest a variable name based on the origin.

        Raises:
            NotImplementedError: Always raises this error as it should be implemented by subclasses.
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
    block_id: int

    def host_str(self) -> str:
        """
        Get the host-side string representation of a block size variable.
        If the block size variable was not created (e.g., block size == 1),
        return the literal '1'.
        """
        from .device_function import DeviceFunction

        # Look up the block size variable name; if not set (e.g., size==1), use literal 1
        var = DeviceFunction.current().block_size_var(self.block_id)
        if var is None:
            return "1"
        return var

    def suggest_var_name(self) -> str:
        return f"block_size_{self.block_id}"


@dataclasses.dataclass
class ReductionDimensionOrigin(Origin):
    rdim_idx: int

    def host_str(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass
class GridOrigin(Origin):
    """Note this represents the tile_begin() of the grid, not the block size (which is always 1)"""

    block_id: int

    def host_str(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass
class TileBeginOrigin(GridOrigin):
    def host_str(self) -> str:
        from .device_function import DeviceFunction

        return DeviceFunction.current().codegen.offset_var(self.block_id)


@dataclasses.dataclass
class TileEndOrigin(GridOrigin):
    def host_str(self) -> str:
        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction

        device_fn = DeviceFunction.current()
        codegen = device_fn.codegen
        offset = codegen.offset_var(self.block_id)
        block_size = device_fn.block_size_var(self.block_id) or "1"
        naive_end = f"{offset} + {block_size}"
        mask = codegen.mask_var(self.block_id)
        if mask is None:
            return naive_end
        end_var = (
            codegen.active_device_loops[self.block_id][-1]
            .block_id_to_info[self.block_id]
            .end_var_name
        )
        assert end_var is not None
        backend = CompileEnvironment.current().backend
        return backend.minimum_expr(naive_end, end_var)


@dataclasses.dataclass
class TileCountOrigin(GridOrigin):
    def host_str(self) -> str:
        from .compile_environment import CompileEnvironment
        from .device_function import DeviceFunction

        device_fn = DeviceFunction.current()
        loop_info = device_fn.codegen.active_device_loops[self.block_id][
            -1
        ].block_id_to_info[self.block_id]
        begin_var_name = loop_info.begin_var_name or "0"
        end_var = loop_info.end_var_name
        assert end_var is not None
        block_size = device_fn.block_size_var(self.block_id) or "1"
        backend = CompileEnvironment.current().backend
        extent = f"({end_var}) - ({begin_var_name})"
        return backend.cdiv_expr(extent, block_size, is_device=True)


@dataclasses.dataclass
class TileIdOrigin(GridOrigin):
    def host_str(self) -> str:
        from .device_function import DeviceFunction

        device_fn = DeviceFunction.current()
        offset = device_fn.codegen.offset_var(self.block_id)
        block_size = device_fn.block_size_var(self.block_id)
        if block_size is None:
            return offset
        return f"{offset} // {block_size}"
