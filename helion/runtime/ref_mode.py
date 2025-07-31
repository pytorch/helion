from __future__ import annotations

import enum
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import NoCurrentEnvironment
from helion._compiler.compile_environment import tls as ce_tls

if TYPE_CHECKING:
    from typing_extensions import Self

    from .settings import Settings


class RefMode(enum.Enum):
    """Reference mode for kernel execution."""

    OFF = "off"
    EAGER = "eager"


def is_ref_mode_enabled(settings: Settings) -> bool:
    """Check if ref mode is enabled based on settings."""
    return settings.ref_mode != RefMode.OFF


def is_in_ref_mode_context() -> bool:
    """Check if we're currently executing in ref mode context.

    This checks if there's a current CompileEnvironment with ref mode enabled.
    """
    try:
        env = CompileEnvironment.current()
        return is_ref_mode_enabled(env.settings)
    except NoCurrentEnvironment:
        return False


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __init__(self, env: CompileEnvironment) -> None:
        self.env = env
        self.func_mode = RefModeTorchFunctionMode()

    def __enter__(self) -> Self:
        ce_tls.env = self.env
        self.func_mode.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.func_mode.__exit__(exc_type, exc_val, exc_tb)
        ce_tls.env = None
        return False


class RefModeTorchFunctionMode(BaseTorchFunctionMode):
    """Torch function mode for Helion ref mode operations."""

    def __torch_function__(
        self,
        func: object,
        types: list[type[object]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if kwargs is None:
            kwargs = {}

        # Handle matrix multiplication operations
        if func == torch.addmm:
            return self._handle_addmm(args, kwargs)
        if func == torch.baddbmm:
            return self._handle_baddbmm(args, kwargs)
        
        # Handle expand with RefTile arguments
        if func == torch.Tensor.expand:
            return self._handle_expand(args, kwargs)
        
        # Handle view with RefTile arguments
        if func == torch.Tensor.view:
            return self._handle_view(args, kwargs)
        
        # Handle reshape with RefTile arguments
        if func == torch.reshape:
            return self._handle_reshape(args, kwargs)
        
        # Handle zeros/ones/full with RefTile arguments
        if func == torch.zeros:
            return self._handle_zeros(args, kwargs)
        
        if func == torch.ones:
            return self._handle_ones(args, kwargs)
        
        if func == torch.full:
            return self._handle_full(args, kwargs)
        
        # Handle tensor.new_zeros/new_ones/new_full methods
        if hasattr(func, '__name__'):
            if func.__name__ == 'new_zeros':
                return self._handle_new_zeros(args, kwargs)
            elif func.__name__ == 'new_ones':
                return self._handle_new_ones(args, kwargs)
            elif func.__name__ == 'new_full':
                return self._handle_new_full(args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _handle_addmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.addmm with mixed precision support (e.g. torch.addmm(fp32, bf16, bf16))."""
        assert len(args) >= 3, "addmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        mat1 = cast("torch.Tensor", args[1])
        mat2 = cast("torch.Tensor", args[2])
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert mat1.dtype == mat2.dtype, (
            f"Matrix dtypes must match for torch.addmm: "
            f"mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

        result = torch.mm(mat1, mat2, out_dtype=bias.dtype)
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        if beta == 0:
            return result
        return result + (beta * bias)

    def _handle_expand(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle tensor.expand() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1:] are the sizes
        tensor = cast("torch.Tensor", args[0])
        sizes = args[1:]
        
        # Convert RefTile objects to their sizes
        new_sizes = []
        for size in sizes:
            if isinstance(size, RefTile):
                # Use the size of the slice (end - begin)
                new_sizes.append(size._slice.stop - size._slice.start)
            else:
                new_sizes.append(size)
        
        # Call expand with the converted sizes
        return tensor.expand(*new_sizes, **kwargs)

    def _handle_view(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle tensor.view() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1:] are the sizes
        tensor = cast("torch.Tensor", args[0])
        sizes = args[1:]
        
        # Convert RefTile objects to their sizes
        new_sizes = []
        for size in sizes:
            if isinstance(size, RefTile):
                # Use the size of the slice (end - begin)
                new_sizes.append(size._slice.stop - size._slice.start)
            else:
                new_sizes.append(size)
        
        # Call view with the converted sizes
        return tensor.view(*new_sizes, **kwargs)
    
    def _handle_reshape(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.reshape() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1] is the shape (which may be a list/tuple)
        tensor = cast("torch.Tensor", args[0])
        if len(args) > 1:
            shape = args[1]
            # If shape is a list/tuple, convert any RefTile objects in it
            if isinstance(shape, (list, tuple)):
                new_shape = []
                for size in shape:
                    if isinstance(size, RefTile):
                        new_shape.append(size._slice.stop - size._slice.start)
                    else:
                        new_shape.append(size)
                return torch.reshape(tensor, new_shape, **kwargs)
            else:
                # Single dimension reshape
                if isinstance(shape, RefTile):
                    shape = shape._slice.stop - shape._slice.start
                return torch.reshape(tensor, shape, **kwargs)
        
        return torch.reshape(tensor, **kwargs)

    def _handle_baddbmm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.baddbmm with mixed precision support (e.g. torch.baddbmm(fp32, bf16, bf16))."""
        assert len(args) >= 3, "baddbmm requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        batch1 = cast("torch.Tensor", args[1])
        batch2 = cast("torch.Tensor", args[2])
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert batch1.dtype == batch2.dtype, (
            f"Matrix dtypes must match for torch.baddbmm: "
            f"mat1.dtype={batch1.dtype}, mat2.dtype={batch2.dtype}"
        )

        result = torch.bmm(batch1, batch2, out_dtype=bias.dtype)
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        if beta == 0:
            return result
        return result + (beta * bias)
    
    def _handle_zeros(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.zeros() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the size argument
        size = args[0]
        other_args = args[1:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.zeros(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.zeros(new_size, *other_args, **kwargs)
        else:
            return torch.zeros(size, *other_args, **kwargs)
    
    def _handle_ones(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.ones() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the size argument
        size = args[0]
        other_args = args[1:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.ones(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.ones(new_size, *other_args, **kwargs)
        else:
            return torch.ones(size, *other_args, **kwargs)
    
    def _handle_full(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.full() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the size argument, args[1] is the fill value
        size = args[0]
        fill_value = args[1]
        other_args = args[2:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return torch.full(new_size, fill_value, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return torch.full(new_size, fill_value, *other_args, **kwargs)
        else:
            return torch.full(size, fill_value, *other_args, **kwargs)
    
    def _handle_new_zeros(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle tensor.new_zeros() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1] is the size argument
        tensor = cast("torch.Tensor", args[0])
        size = args[1]
        other_args = args[2:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return tensor.new_zeros(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return tensor.new_zeros(new_size, *other_args, **kwargs)
        else:
            return tensor.new_zeros(size, *other_args, **kwargs)
    
    def _handle_new_ones(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle tensor.new_ones() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1] is the size argument
        tensor = cast("torch.Tensor", args[0])
        size = args[1]
        other_args = args[2:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return tensor.new_ones(new_size, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return tensor.new_ones(new_size, *other_args, **kwargs)
        else:
            return tensor.new_ones(size, *other_args, **kwargs)
    
    def _handle_new_full(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle tensor.new_full() with RefTile arguments."""
        from ..language.ref_tile import RefTile
        
        # args[0] is the tensor, args[1] is the size, args[2] is the fill value
        tensor = cast("torch.Tensor", args[0])
        size = args[1]
        fill_value = args[2]
        other_args = args[3:]
        
        # Convert RefTile objects to their sizes
        if isinstance(size, (list, tuple)):
            new_size = []
            for s in size:
                if isinstance(s, RefTile):
                    new_size.append(s._slice.stop - s._slice.start)
                else:
                    new_size.append(s)
            return tensor.new_full(new_size, fill_value, *other_args, **kwargs)
        elif isinstance(size, RefTile):
            # Single RefTile as size
            new_size = size._slice.stop - size._slice.start
            return tensor.new_full(new_size, fill_value, *other_args, **kwargs)
        else:
            return tensor.new_full(size, fill_value, *other_args, **kwargs)
