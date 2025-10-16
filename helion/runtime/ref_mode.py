from __future__ import annotations

import enum
import threading
import typing
from typing import TYPE_CHECKING
from typing import Callable
from typing import Protocol
from typing import cast

import torch
from torch.overrides import BaseTorchFunctionMode

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import NoCurrentEnvironment
from .._compiler.compile_environment import tls as ce_tls
from .._utils import convert_size_arg
from .._utils import create_shape_matching_slices

if TYPE_CHECKING:
    from typing_extensions import Self

    from .. import Config
    from .settings import Settings

    class _RefModeTLS(Protocol):
        context: RefModeContext | None


# Thread-local storage for RefModeContext
ref_mode_tls: _RefModeTLS = typing.cast("_RefModeTLS", threading.local())


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


class NoCurrentRefModeContext(RuntimeError):
    """Raised when RefModeContext.current() is called but no context is active."""


class RefModeContext:
    """Context manager to enable ref mode execution."""

    def __init__(self, env: CompileEnvironment, config: Config | None) -> None:
        self.env = env
        self.func_mode = RefModeTorchFunctionMode()
        self.config = config

    def __enter__(self) -> Self:
        assert getattr(ref_mode_tls, "context", None) is None, (
            "RefModeContext already active"
        )
        ce_tls.env = self.env
        ref_mode_tls.context = self
        self.func_mode.__enter__()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.func_mode.__exit__(exc_type, exc_val, exc_tb)
        ref_mode_tls.context = None
        ce_tls.env = None
        return False

    @staticmethod
    def current() -> RefModeContext:
        """Get the currently active RefModeContext.

        Returns:
            The active RefModeContext

        Raises:
            NoCurrentRefModeContext: If no RefModeContext is active
        """
        try:
            if (context := ref_mode_tls.context) is not None:
                return context
        except AttributeError:
            pass
        raise NoCurrentRefModeContext from None

    @staticmethod
    def has_current() -> bool:
        """Check if a RefModeContext is currently active."""
        try:
            RefModeContext.current()
            return True
        except NoCurrentRefModeContext:
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
            torch.Tensor.expand: lambda args, kwargs: self._handle_size_arg_method(
                args, kwargs, "expand"
            ),
            torch.Tensor.view: lambda args, kwargs: self._handle_size_arg_method(
                args, kwargs, "view"
            ),
            torch.Tensor.reshape: lambda args, kwargs: self._handle_size_arg_method(
                args, kwargs, "reshape"
            ),
            torch.reshape: lambda args, kwargs: self._handle_size_arg_method(
                args, kwargs, "reshape"
            ),
            # Factory functions with standard pattern
            **{
                func: lambda args, kwargs, f=func: self._handle_factory_func(
                    args, kwargs, f, has_fill=False
                )
                for func in [torch.zeros, torch.ones]
            },
            torch.full: lambda args, kwargs: self._handle_factory_func(
                args, kwargs, torch.full, has_fill=True
            ),
        }

        # Map method names to their handlers for tensor methods
        self._method_handlers = {
            **{
                method: lambda args, kwargs, m=method: self._handle_factory_method(
                    args, kwargs, m, has_fill=False
                )
                for method in ["new_zeros", "new_ones"]
            },
            "new_full": lambda args, kwargs: self._handle_factory_method(
                args, kwargs, "new_full", has_fill=True
            ),
        }

        # Initialize binary operation mappings
        self._setup_binary_ops_handling()

    def __torch_function__(
        self,
        func: object,
        types: list[type[object]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}

        if func in self._func_handlers:
            return self._func_handlers[func](args, kwargs)

        func_name = getattr(func, "__name__", None)
        if func_name:
            if func_name in self._method_handlers:
                return self._method_handlers[func_name](args, kwargs)
            if func_name in self._binary_op_names:
                return self._handle_binary_op(func, args, kwargs)

        if func in self._binary_ops:
            return self._handle_binary_op(func, args, kwargs)

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
        bias, mat1, mat2 = (cast("torch.Tensor", args[i]) for i in range(3))
        beta = cast("float", kwargs.get("beta", 1))
        alpha = cast("float", kwargs.get("alpha", 1))

        assert mat1.dtype == mat2.dtype, (
            f"Matrix dtypes must match for torch.{op_name}: "
            f"mat1.dtype={mat1.dtype}, mat2.dtype={mat2.dtype}"
        )

        result = cast("Callable[..., torch.Tensor]", mm_func)(
            mat1, mat2, out_dtype=bias.dtype
        )
        if alpha != 1:
            result = result * alpha
        if result.dtype != bias.dtype:
            result = result.to(bias.dtype)
        return result if beta == 0 else result + (beta * bias)

    def _handle_size_arg_method(
        self, args: tuple[object, ...], kwargs: dict[str, object], method_name: str
    ) -> torch.Tensor:
        """Handle tensor methods that take size arguments (expand, view, reshape)."""
        tensor = cast("torch.Tensor", args[0])

        if method_name == "reshape":
            # reshape can take shape as multiple positional args or as a single tuple/list
            # e.g., tensor.reshape(2, 3) or tensor.reshape((2, 3))
            if "shape" in kwargs:
                # Handle kwargs case: tensor.reshape(shape=(2, 3))
                shape = convert_size_arg(kwargs["shape"])
                kwargs = dict(kwargs)  # Make a copy to avoid modifying the original
                kwargs["shape"] = shape
                return torch.reshape(tensor, **kwargs)  # type: ignore[arg-type]
            # Handle positional args case
            sizes = args[1:]
            new_sizes = convert_size_arg(sizes)
            method = getattr(tensor, method_name)
            assert isinstance(new_sizes, list)
            return method(*new_sizes, **kwargs)

        # view/expand take sizes as positional args
        sizes = args[1:]
        new_sizes = convert_size_arg(sizes)
        method = getattr(tensor, method_name)
        assert isinstance(new_sizes, list)
        return method(*new_sizes, **kwargs)

    def _handle_factory_func(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        func: object,
        has_fill: bool,
    ) -> torch.Tensor:
        """Handle torch tensor factory functions (zeros, ones, full) with RefTile arguments."""
        size = convert_size_arg(args[0])
        func_callable = cast("Callable[..., torch.Tensor]", func)
        extra_args = args[1:]
        return func_callable(size, *extra_args, **kwargs)

    def _handle_factory_method(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        method_name: str,
        has_fill: bool,
    ) -> torch.Tensor:
        """Handle tensor.new_* factory methods (new_zeros, new_ones, new_full) with RefTile arguments."""
        tensor = cast("torch.Tensor", args[0])
        if "size" in kwargs:
            kwargs = dict(kwargs)
            size = kwargs.pop("size")
            assert len(args) == 1
        else:
            size = args[1]
        size = convert_size_arg(size)
        method = getattr(tensor, method_name)
        extra_args = args[2:]
        return method(size, *extra_args, **kwargs)

    def _handle_binary_op(
        self,
        func: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        """Handle binary operations with shape-based masking for power-of-2 tensors."""
        if len(args) < 2:
            return cast("Callable[..., torch.Tensor]", func)(*args, **kwargs)

        lhs, rhs = args[0], args[1]

        # Skip if either operand is not a tensor (e.g., scalar operations)
        if not (isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor)):
            return cast("Callable[..., torch.Tensor]", func)(*args, **kwargs)

        if not self._should_handle_binary_op(lhs, rhs):
            return cast("Callable[..., torch.Tensor]", func)(*args, **kwargs)

        # Check if this is an in-place operation
        func_name = getattr(func, "__name__", "")
        is_inplace = (
            func in self._inplace_binary_ops
            or func_name in self._inplace_binary_op_names
        )

        # Create slices that take the minimum size in each dimension
        slices = create_shape_matching_slices(lhs.shape, rhs.shape)

        # Apply the operation on the overlapping region
        result = cast("Callable[..., torch.Tensor]", func)(
            lhs[slices], rhs[slices], *args[2:], **kwargs
        )

        # For in-place ops, the operation already modified lhs, so just return it
        # For out-of-place ops, return the computed result
        return lhs if is_inplace else result

    def _should_handle_binary_op(self, lhs: object, rhs: object) -> bool:
        """Check if binary operation needs special handling.

        Only handle cases where both tensors have the same shape except for
        dimension size differences (for power-of-2 padding). Skip broadcasting
        cases where one dimension is 1.
        """
        if not (isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor)):
            return False

        if lhs.shape == rhs.shape or len(lhs.shape) != len(rhs.shape):
            return False

        # Check if this is a broadcasting case (any dimension is 1)
        for l_dim, r_dim in zip(lhs.shape, rhs.shape, strict=False):
            if l_dim == 1 or r_dim == 1:
                return False  # Let PyTorch handle broadcasting

        # Only handle shape-based masking for non-broadcasting cases
        return True

    def _setup_binary_ops_handling(self) -> None:
        """Initialize binary operation tracking sets and mappings."""
        # Define binary operations and their variants
        binary_op_info = [
            # (torch_func, tensor_method, inplace_method, inplace_name, func_names)
            (torch.add, torch.Tensor.__add__, torch.Tensor.__iadd__, "add_", ["add"]),
            (torch.sub, torch.Tensor.__sub__, torch.Tensor.__isub__, "sub_", ["sub"]),
            (torch.mul, torch.Tensor.__mul__, torch.Tensor.__imul__, "mul_", ["mul"]),
            (
                torch.div,
                torch.Tensor.__truediv__,
                torch.Tensor.__itruediv__,
                "div_",
                ["div", "__truediv__"],
            ),
            (
                torch.floor_divide,
                torch.Tensor.__floordiv__,
                torch.Tensor.__ifloordiv__,
                "floor_divide_",
                ["floordiv", "__floordiv__"],
            ),
            (
                torch.remainder,
                torch.Tensor.__mod__,
                torch.Tensor.__imod__,
                "remainder_",
                ["mod", "__mod__"],
            ),
            (torch.pow, torch.Tensor.__pow__, torch.Tensor.__ipow__, "pow_", ["pow"]),
        ]

        # Build sets from the info
        self._inplace_binary_ops = set()
        self._inplace_binary_op_names = set()
        self._binary_ops = set()
        self._binary_op_names = set()

        for (
            torch_func,
            method,
            inplace_method,
            inplace_name,
            func_names,
        ) in binary_op_info:
            # Add function objects
            self._binary_ops.add(torch_func)
            self._binary_ops.add(method)
            self._binary_ops.add(inplace_method)
            self._inplace_binary_ops.add(inplace_method)

            # Add inplace op string names
            self._inplace_binary_op_names.add(inplace_name)

            # Add all function names
            self._binary_op_names.update(func_names)
            self._binary_op_names.add(method.__name__)
            self._binary_op_names.add(inplace_method.__name__)
            self._binary_op_names.add(inplace_name)
