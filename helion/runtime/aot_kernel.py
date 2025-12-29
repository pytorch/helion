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

The key function adapts based on available heuristics:
- In evaluate mode with heuristics: uses only features the heuristic needs
- Otherwise: uses all shape features for full coverage
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

# Sentinel for "not yet loaded"
_NOT_LOADED = object()


def _get_dtype_category(dtype: torch.dtype) -> int:
    """Get numeric category for dtype (same as aot_cache.py)."""
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


def extract_shape_features(args: Sequence[object]) -> dict[str, Any]:
    """
    Extract numeric shape features from kernel arguments.

    This is the single source of truth for feature extraction, used by both:
    - AOT heuristic training (in aot_cache.py)
    - Specialization key generation (here)

    Features extracted:
    - arg{i}_ndim: number of dimensions
    - arg{i}_dim{j}: size of each dimension
    - arg{i}_numel: total number of elements
    - arg{i}_dtype: dtype string
    - arg{i}_dtype_size: element size in bytes
    - arg{i}_dtype_cat: dtype category (int/float/etc)
    - arg{i}_scalar: scalar value for numeric args
    """
    features: dict[str, Any] = {}

    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            features[f"arg{i}_ndim"] = arg.ndim
            for j, size in enumerate(arg.shape):
                features[f"arg{i}_dim{j}"] = int(size)
            features[f"arg{i}_numel"] = int(arg.numel())
            features[f"arg{i}_dtype"] = str(arg.dtype)
            features[f"arg{i}_dtype_size"] = arg.element_size()
            features[f"arg{i}_dtype_cat"] = _get_dtype_category(arg.dtype)
        elif isinstance(arg, (int, float)):
            features[f"arg{i}_scalar"] = arg

    return features


class AOTKeyFunction:
    """
    Dynamic key function that uses heuristic features when available.

    In evaluate mode with heuristics, extracts only the features that the
    heuristic actually uses for decisions. This minimizes cache fragmentation
    by ensuring shapes that map to the same config share a cache entry.

    In other modes (collect, measure, disabled), extracts all features to
    ensure full coverage during training.
    """

    # Class-level cache: (kernel_source_file, kernel_name) -> feature_names or None
    _feature_cache: ClassVar[dict[tuple[str, str], list[str] | None]] = {}

    def __init__(self, kernel_source_file: str, kernel_name: str) -> None:
        self.kernel_source_file = kernel_source_file
        self.kernel_name = kernel_name
        self._loaded: bool = False
        self._feature_names: list[str] | None = None

    def _load_heuristic_features(self) -> list[str] | None:
        """Load feature names from the heuristic file if available."""
        if self._loaded:
            return self._feature_names

        cache_key = (self.kernel_source_file, self.kernel_name)

        # Check class-level cache first
        if cache_key in AOTKeyFunction._feature_cache:
            self._feature_names = AOTKeyFunction._feature_cache[cache_key]
            self._loaded = True
            return self._feature_names

        # Only load heuristics in evaluate mode
        aot_mode = os.environ.get("HELION_AOT_MODE", "evaluate").lower()
        if aot_mode != "evaluate":
            self._feature_names = None
            self._loaded = True
            AOTKeyFunction._feature_cache[cache_key] = None
            return None

        # Use shared heuristic file discovery
        try:
            from ..autotuner.aot_cache import find_heuristic_file

            heuristic_path = find_heuristic_file(
                self.kernel_source_file, kernel_name=self.kernel_name
            )

            if heuristic_path is not None:
                # Load the heuristic module and get FEATURE_NAMES
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "heuristic", heuristic_path
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try kernel-specific feature names first, then global
                    feature_names = getattr(
                        module, f"FEATURE_NAMES_{self.kernel_name.upper()}", None
                    )
                    if feature_names is None:
                        feature_names = getattr(module, "FEATURE_NAMES", None)

                    if feature_names is not None:
                        self._feature_names = list(feature_names)
                        self._loaded = True
                        AOTKeyFunction._feature_cache[cache_key] = self._feature_names
                        return self._feature_names
        except Exception:
            pass  # Silently fall back to full features

        self._feature_names = None
        self._loaded = True
        AOTKeyFunction._feature_cache[cache_key] = None
        return None

    def __call__(self, *args: object) -> Hashable:
        """Generate specialization key from arguments."""
        features = extract_shape_features(args)
        heuristic_features = self._load_heuristic_features()

        if heuristic_features:
            # Use only features the heuristic cares about
            # This ensures shapes that produce the same config share a cache entry
            key_parts = []
            for fname in heuristic_features:
                if fname in features:
                    key_parts.append((fname, features[fname]))
            return tuple(key_parts)
        # No heuristic available - use all features
        # Sort for deterministic ordering
        return tuple(sorted(features.items()))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the feature cache (useful for testing)."""
        cls._feature_cache.clear()


def make_aot_key(kernel_source_file: str, kernel_name: str) -> AOTKeyFunction:
    """
    Create an AOT key function for a specific kernel.

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Name of the kernel function

    Returns:
        A callable that generates specialization keys from kernel arguments
    """
    return AOTKeyFunction(kernel_source_file, kernel_name)


# Simple fallback for cases where we don't have kernel info
def aot_key(*args: object) -> Hashable:
    """
    Simple AOT key function that uses all shape features.

    This is a fallback when kernel source info is not available.
    Prefer using make_aot_key() when possible for heuristic-aware keying.
    """
    features = extract_shape_features(args)
    return tuple(sorted(features.items()))


class _AOTKernelDecorator:
    """Protocol for the aot_kernel decorator when called without arguments."""

    def __call__(self, fn: Callable[..., _R]) -> Kernel[_R]: ...


@overload
def aot_kernel(
    fn: Callable[..., _R],
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    **settings: object,
) -> Kernel[_R]: ...


@overload
def aot_kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    **settings: object,
) -> _AOTKernelDecorator: ...


def aot_kernel(
    fn: Callable[..., _R] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
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
                aot_kernel, config=config, configs=configs, key=user_key, **settings
            ),
        )

    # Get kernel source file and name for heuristic-aware key
    kernel_source_file = fn.__code__.co_filename
    kernel_name = fn.__name__

    # Use user's key if provided, otherwise create heuristic-aware key
    key_fn: KeyFunction = (
        user_key
        if user_key is not None
        else make_aot_key(kernel_source_file, kernel_name)
    )

    return kernel(fn, config=config, configs=configs, key=key_fn, **settings)
