"""
AOT Kernel Decorator
====================

Provides a simplified decorator for creating kernels with AOT (Ahead-of-Time)
autotuning support. This decorator automatically configures the kernel for
heuristic-based config selection.

Usage:
    @helion.aot_kernel()
    def my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

The key function is loaded from the generated heuristic file:
- key_<kernel>(*args): Generated key function using only features that matter
- Falls back to all shape features if no heuristic is available
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Hashable
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing import overload

import torch

if TYPE_CHECKING:
    from .kernel import ConfigLike
    from .kernel import Kernel


_R = TypeVar("_R")

# Type alias for key functions
KeyFunction = Callable[..., Hashable]


def _get_dtype_category(dtype: torch.dtype) -> int:
    """Get numeric category for dtype."""
    if dtype == torch.bool:
        return 0
    if dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        return 1
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return 2
    if dtype in (torch.complex64, torch.complex128):
        return 3
    return 4


# Type alias for batched specification
# List with one entry per argument:
# - For tensors: list with one entry per dimension (None=not batched, int=batch index)
# - For non-tensors: None
BatchedSpec = Sequence[Sequence[int | None] | None] | None


def extract_shape_features(
    args: Sequence[object],
    batched: BatchedSpec = None,
) -> dict[str, Any]:
    """
    Extract numeric shape features from kernel arguments.

    This is the single source of truth for feature extraction, used by both:
    - AOT heuristic training (in aot_cache.py)
    - Specialization key generation (here)

    Features extracted:
    - arg{i}_ndim: number of dimensions
    - arg{i}_dim{j}: size of each dimension (skipped for batched dimensions)
    - arg{i}_numel: total number of elements
    - arg{i}_dtype: dtype string
    - arg{i}_dtype_size: element size in bytes
    - arg{i}_dtype_cat: dtype category (int/float/etc)
    - arg{i}_scalar: scalar value for numeric args

    Args:
        args: Kernel arguments
        batched: Optional batch dimension specification. List with one entry per
            argument. For tensor args, a list with one entry per dimension where
            None means not batched and an integer means batched. For non-tensor
            args, None. Example for rms_norm(weight, input, eps):
            [[None], [0, None], None] means input's first dim is batched.
    """
    features: dict[str, Any] = {}

    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            features[f"arg{i}_ndim"] = arg.ndim

            # Get batch info for this argument
            arg_batched = batched[i] if batched and i < len(batched) else None

            # Check if any dimension is batched
            has_batched_dim = arg_batched is not None and any(
                b is not None for b in arg_batched
            )

            for j, size in enumerate(arg.shape):
                # Skip batched dimensions
                is_batched = (
                    arg_batched is not None
                    and j < len(arg_batched)
                    and arg_batched[j] is not None
                )
                if not is_batched:
                    features[f"arg{i}_dim{j}"] = int(size)

            # Skip numel if tensor has any batched dimensions (numel includes batch)
            if not has_batched_dim:
                features[f"arg{i}_numel"] = int(arg.numel())
            features[f"arg{i}_dtype"] = str(arg.dtype)
            features[f"arg{i}_dtype_size"] = arg.element_size()
            features[f"arg{i}_dtype_cat"] = _get_dtype_category(arg.dtype)
        elif isinstance(arg, (int, float)):
            features[f"arg{i}_scalar"] = arg

    return features


# Simple fallback key function using all shape features
def aot_key(*args: object, batched: BatchedSpec = None) -> Hashable:
    """
    Simple AOT key function that uses all shape features.

    This is a fallback when no heuristic is available.

    Args:
        *args: Kernel arguments
        batched: Optional batch dimension specification (see extract_shape_features)
    """
    features = extract_shape_features(args, batched=batched)
    return tuple(sorted(features.items()))


class HeuristicKeyFunction:
    """
    Key function that loads key_<kernel> from the heuristic file.

    In evaluate mode, loads the generated key function from the heuristic file.
    In other modes, falls back to using all shape features.
    """

    # Class-level cache: (kernel_source_file, kernel_name) -> key_fn or None
    _key_fn_cache: ClassVar[dict[tuple[str, str], KeyFunction | None]] = {}

    def __init__(
        self,
        kernel_source_file: str,
        kernel_name: str,
        batched: BatchedSpec = None,
    ) -> None:
        self.kernel_source_file = kernel_source_file
        self.kernel_name = kernel_name
        self.batched = batched
        self._loaded: bool = False
        self._key_fn: KeyFunction | None = None

    def _load_key_function(self) -> KeyFunction | None:
        """Load key_<kernel> function from the heuristic file if available."""
        if self._loaded:
            return self._key_fn

        cache_key = (self.kernel_source_file, self.kernel_name)

        # Check class-level cache first
        if cache_key in HeuristicKeyFunction._key_fn_cache:
            self._key_fn = HeuristicKeyFunction._key_fn_cache[cache_key]
            self._loaded = True
            return self._key_fn

        # Only load heuristics in evaluate mode
        aot_mode = os.environ.get("HELION_AOT_MODE", "evaluate").lower()
        if aot_mode != "evaluate":
            self._key_fn = None
            self._loaded = True
            HeuristicKeyFunction._key_fn_cache[cache_key] = None
            return None

        # Use shared heuristic file discovery
        try:
            from ..autotuner.aot_cache import find_heuristic_file

            heuristic_path = find_heuristic_file(
                self.kernel_source_file, kernel_name=self.kernel_name
            )

            if heuristic_path is not None:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "heuristic", heuristic_path
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Load the key_<kernel> function
                    key_fn = getattr(module, f"key_{self.kernel_name}", None)
                    if key_fn is not None:
                        self._key_fn = key_fn
                        self._loaded = True
                        HeuristicKeyFunction._key_fn_cache[cache_key] = self._key_fn
                        return self._key_fn
        except Exception:
            pass  # Silently fall back to full features

        self._key_fn = None
        self._loaded = True
        HeuristicKeyFunction._key_fn_cache[cache_key] = None
        return None

    def __call__(self, *args: object) -> Hashable:
        """Generate specialization key from arguments."""
        key_fn = self._load_key_function()

        if key_fn is not None:
            # Use the heuristic's key function
            return key_fn(*args)

        # Fallback: use all features
        return aot_key(*args, batched=self.batched)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the key function cache (useful for testing)."""
        cls._key_fn_cache.clear()


def make_aot_key(
    kernel_source_file: str,
    kernel_name: str,
    batched: BatchedSpec = None,
) -> HeuristicKeyFunction:
    """
    Create an AOT key function for a specific kernel.

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Name of the kernel function
        batched: Optional batch dimension specification (see extract_shape_features)

    Returns:
        A callable that generates specialization keys from kernel arguments
    """
    return HeuristicKeyFunction(kernel_source_file, kernel_name, batched=batched)


class _AOTKernelDecorator:
    """Protocol for the aot_kernel decorator when called without arguments."""

    def __call__(self, fn: Callable[..., _R]) -> Kernel[_R]: ...


@overload
def aot_kernel(
    fn: Callable[..., _R],
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    batched: BatchedSpec = None,
    **settings: object,
) -> Kernel[_R]: ...


@overload
def aot_kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    batched: BatchedSpec = None,
    **settings: object,
) -> _AOTKernelDecorator: ...


def aot_kernel(
    fn: Callable[..., _R] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    batched: BatchedSpec = None,
    **settings: object,
) -> Kernel[_R] | _AOTKernelDecorator:
    """
    Decorator to create a Kernel with AOT (Ahead-of-Time) autotuning support.

    This decorator configures the kernel for heuristic-based config selection,
    allowing per-shape configs to be selected at runtime using pre-generated
    decision trees.

    Key features:
    - Automatically uses AOTAutotuneCache for heuristic support
    - Dynamic specialization key that adapts to available heuristics
    - In evaluate mode: uses only features the heuristic needs (minimal keys)
    - In collect/measure modes: uses all features (full coverage)

    The AOT workflow is:
    1. Run benchmarks with HELION_AOT_MODE=collect to tune each shape
    2. Run with HELION_AOT_MODE=measure to measure all configs across shapes
    3. Generate heuristics: python -m helion.autotuner.aot_runner --generate
    4. Deploy with HELION_AOT_MODE=evaluate (default) to use heuristics

    Args:
        fn: The function to be wrapped by the Kernel. If None, a decorator is returned.
        config: A single configuration to use for the kernel (optional).
        configs: A list of configurations to use for the kernel (optional).
        batched: Optional batch dimension specification. A list with one entry per
            argument. For tensor args, a list with one entry per dimension where
            None means not batched and an integer means batched. For non-tensor
            args, None. Example for rms_norm(weight, input, eps):
            [[None], [0, None], None] means input's first dim is batched.
            Batched dimensions are excluded from the heuristic key.
        **settings: Additional settings for the Kernel.

    Returns:
        Kernel: A Kernel object configured for AOT autotuning.

    Example:
        @helion.aot_kernel()
        def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.shape
            _, n = b.shape
            out = torch.empty((m, n), dtype=a.dtype, device=a.device)
            for tile in hl.tile(m, n):
                acc = hl.zeros([tile[0], tile[1]], dtype=torch.float32)
                for k_tile in hl.tile(k):
                    acc += a[tile[0], k_tile].to(torch.float32) @ b[k_tile, tile[1]].to(torch.float32)
                out[tile] = acc.to(out.dtype)
            return out

        # The kernel will automatically use heuristics when available
        result = matmul(x, y)

        # Example with batched dimension:
        @helion.aot_kernel(batched=[[0, None], None])
        def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
            # x has shape (batch, hidden), first dim is batched
            ...
    """
    from .kernel import kernel

    # Set AOT-specific defaults
    settings.setdefault("autotune_cache", "AOTAutotuneCache")
    settings.setdefault("static_shapes", False)

    # Check if user provided their own key
    user_key: KeyFunction | None = cast("KeyFunction | None", settings.pop("key", None))

    if fn is None:
        # Called as @aot_kernel() - return a decorator
        return cast(
            "_AOTKernelDecorator",
            functools.partial(
                aot_kernel,
                config=config,
                configs=configs,
                batched=batched,
                key=user_key,
                **settings,
            ),
        )

    # Get kernel source file and name for heuristic-aware key
    kernel_source_file = fn.__code__.co_filename
    kernel_name = fn.__name__

    # Use user's key if provided, otherwise create heuristic-aware key
    key_fn: KeyFunction = (
        user_key
        if user_key is not None
        else make_aot_key(kernel_source_file, kernel_name, batched=batched)
    )

    return kernel(fn, config=config, configs=configs, key=key_fn, **settings)
