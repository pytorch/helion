from __future__ import annotations

import enum
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import NoCurrentEnvironment
from helion._compiler.compile_environment import tls as ce_tls
from helion.language.ref_tile import convert_reftiles_to_sizes
from helion.language.ref_tile import convert_size_arg

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

    def __init__(self) -> None:
        super().__init__()
        # Map functions to their handlers
        self._func_handlers = {
            torch.addmm: lambda args, kwargs: self._handle_mm_with_bias(
                args, kwargs, torch.mm, "addmm"
            ),
            torch.baddbmm: lambda args, kwargs: self._handle_mm_with_bias(
                args, kwargs, torch.bmm, "baddbmm"
            ),
            torch.Tensor.expand: lambda args, kwargs: self._handle_tensor_size_method(
                args, kwargs, "expand"
            ),
            torch.Tensor.view: lambda args, kwargs: self._handle_tensor_size_method(
                args, kwargs, "view"
            ),
            torch.reshape: self._handle_reshape,
            torch.zeros: lambda args, kwargs: self._handle_factory_func(
                args, kwargs, torch.zeros, has_fill=False
            ),
            torch.ones: lambda args, kwargs: self._handle_factory_func(
                args, kwargs, torch.ones, has_fill=False
            ),
            torch.full: lambda args, kwargs: self._handle_factory_func(
                args, kwargs, torch.full, has_fill=True
            ),
        }

        # Map method names to their handlers for tensor methods
        self._method_handlers = {
            "new_zeros": lambda args, kwargs: self._handle_factory_method(
                args, kwargs, "new_zeros", has_fill=False
            ),
            "new_ones": lambda args, kwargs: self._handle_factory_method(
                args, kwargs, "new_ones", has_fill=False
            ),
            "new_full": lambda args, kwargs: self._handle_factory_method(
                args, kwargs, "new_full", has_fill=True
            ),
        }

    def __torch_function__(
        self,
        func: object,
        types: list[type[object]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if kwargs is None:
            kwargs = {}

        # Check direct function mapping
        if func in self._func_handlers:
            return self._func_handlers[func](args, kwargs)

        # Check method name mapping
        if hasattr(func, "__name__"):
            func_name = getattr(func, "__name__", None)
            if func_name and func_name in self._method_handlers:
                return self._method_handlers[func_name](args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    def _handle_mm_with_bias(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        mm_func: object,
        op_name: str,
    ) -> torch.Tensor:
        """Handle torch.addmm/baddbmm with mixed precision support."""
        assert len(args) >= 3, f"{op_name} requires at least 3 arguments"
        bias = cast("torch.Tensor", args[0])
        mat1 = cast("torch.Tensor", args[1])
        mat2 = cast("torch.Tensor", args[2])
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert mat1.dtype == mat2.dtype, (
            f"Matrix dtypes must match for torch.{op_name}: "
            f"mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

        # Cast mm_func to callable since we know it's a function
        mm_func_callable = cast("Callable[..., torch.Tensor]", mm_func)
        result = mm_func_callable(mat1, mat2, out_dtype=bias.dtype)
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        if beta == 0:
            return result
        return result + (beta * bias)

    def _handle_tensor_size_method(
        self, args: tuple[object, ...], kwargs: dict[str, object], method_name: str
    ) -> torch.Tensor:
        """Handle tensor methods that take size arguments (expand, view)."""
        tensor = cast("torch.Tensor", args[0])
        sizes = args[1:]
        new_sizes = convert_reftiles_to_sizes(sizes)
        method = getattr(tensor, method_name)
        return method(*new_sizes, **kwargs)

    def _handle_reshape(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        """Handle torch.reshape() with RefTile arguments."""
        tensor = cast("torch.Tensor", args[0])
        if len(args) > 1:
            shape = convert_size_arg(args[1])
            # Reshape expects a sequence of ints, cast appropriately
            shape_seq = cast("list[int] | tuple[int, ...]", shape)
            return torch.reshape(tensor, shape_seq, **kwargs)
        # If shape is in kwargs, convert it
        if "shape" in kwargs:
            converted_shape = convert_size_arg(kwargs["shape"])
            kwargs["shape"] = cast("list[int] | tuple[int, ...]", converted_shape)
        # kwargs may contain shape that needs type casting
        return torch.reshape(tensor, **kwargs)  # type: ignore[arg-type]

    def _handle_factory_func(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        func: object,
        has_fill: bool,
    ) -> torch.Tensor:
        """Handle torch tensor factory functions (zeros, ones, full) with RefTile arguments."""
        size = convert_size_arg(args[0])
        # Cast func to callable
        func_callable = cast("Callable[..., torch.Tensor]", func)
        if has_fill:
            fill_value = args[1]
            return func_callable(size, fill_value, *args[2:], **kwargs)
        return func_callable(size, *args[1:], **kwargs)

    def _handle_factory_method(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        method_name: str,
        has_fill: bool,
    ) -> torch.Tensor:
        """Handle tensor.new_* factory methods (new_zeros, new_ones, new_full) with RefTile arguments."""
        tensor = cast("torch.Tensor", args[0])
        size = convert_size_arg(args[1])
        method = getattr(tensor, method_name)
        if has_fill:
            fill_value = args[2]
            return method(size, fill_value, *args[3:], **kwargs)
        return method(size, *args[2:], **kwargs)
