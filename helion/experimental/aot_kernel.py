"""
AOT Kernel Decorator
====================

Provides a simplified decorator for creating kernels with AOT (Ahead-of-Time)
autotuning support. This decorator automatically configures the kernel for
heuristic-based config selection.

Usage:
    @helion.experimental.aot_kernel()
    def my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

The key function is loaded from the generated heuristic file:
- key_<kernel>(*args): Generated key function using only features that matter
- Falls back to all shape features if no heuristic is available
"""

from __future__ import annotations

from collections.abc import Iterable
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
    from ..runtime.kernel import ConfigLike
    from ..runtime.kernel import Kernel


_R = TypeVar("_R")

# Type alias for key functions
KeyFunction = Callable[..., Hashable]

# Type alias for input generator functions (collect_fn/measure_fn)
# Returns an iterable of argument tuples for the kernel
InputFn = Callable[[], Iterable[tuple[Any, ...]]]


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


def _flatten_key_value(value: object) -> list[int | float | str]:
    """
    Recursively flatten a key value into a list of primitives.

    Handles nested tuples, lists, and converts dtypes to their element size.
    """
    result: list[int | float | str] = []

    if isinstance(value, (tuple, list)):
        for item in value:
            result.extend(_flatten_key_value(item))
    elif isinstance(value, torch.dtype):
        # Convert dtype to element size (numeric)
        result.append(torch.tensor([], dtype=value).element_size())
    elif isinstance(value, (int, float, str)):
        result.append(value)
    elif value is None:
        pass  # Skip None values
    else:
        # Try to convert to string as fallback
        result.append(str(value))

    return result


def extract_key_features(key_value: object) -> dict[str, Any]:
    """
    Extract features from a user key function's output.

    Pytree flattens the key value and creates features named key_0, key_1, etc.

    Args:
        key_value: The output of a user's key function

    Returns:
        Dictionary of features: {key_0: val0, key_1: val1, ...}
    """
    flat = _flatten_key_value(key_value)
    return {f"key_{i}": v for i, v in enumerate(flat)}


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

    When a user_key is provided, the heuristic is trained on the flattened
    key output values, and this class composes the user key with the heuristic.
    """

    # Class-level cache: (kernel_source_file, kernel_name) -> key_fn or None
    _key_fn_cache: ClassVar[dict[tuple[str, str], KeyFunction | None]] = {}

    def __init__(
        self,
        kernel_source_file: str,
        kernel_name: str,
        batched: BatchedSpec = None,
        user_key: KeyFunction | None = None,
    ) -> None:
        self.kernel_source_file = kernel_source_file
        self.kernel_name = kernel_name
        self.batched = batched
        self.user_key = user_key
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
        heuristic_key_fn = self._load_key_function()

        if self.user_key is not None:
            # User provided a key function
            user_key_value = self.user_key(*args)

            if heuristic_key_fn is not None:
                # Heuristic loaded - flatten user key and pass to heuristic
                flat_key = _flatten_key_value(user_key_value)
                return heuristic_key_fn(*flat_key)

            # No heuristic - use user key value directly as cache key
            return user_key_value

        if heuristic_key_fn is not None:
            # Use the heuristic's key function directly on args
            return heuristic_key_fn(*args)

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
    user_key: KeyFunction | None = None,
) -> HeuristicKeyFunction:
    """
    Create an AOT key function for a specific kernel.

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Name of the kernel function
        batched: Optional batch dimension specification (see extract_shape_features)
        user_key: Optional user-provided key function. If provided, the heuristic
            will be trained on the flattened output of this function.

    Returns:
        A callable that generates specialization keys from kernel arguments
    """
    return HeuristicKeyFunction(
        kernel_source_file, kernel_name, batched=batched, user_key=user_key
    )


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
    collect_fn: InputFn | None = None,
    measure_fn: InputFn | None = None,
    **settings: object,
) -> Kernel[_R]: ...


@overload
def aot_kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    batched: BatchedSpec = None,
    collect_fn: InputFn | None = None,
    measure_fn: InputFn | None = None,
    **settings: object,
) -> _AOTKernelDecorator: ...


def aot_kernel(
    fn: Callable[..., _R] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    batched: BatchedSpec = None,
    collect_fn: InputFn | None = None,
    measure_fn: InputFn | None = None,
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
    - Optional collect_fn/measure_fn to specify inputs for collect/measure phases

    The AOT workflow is:
    1. Run benchmarks with HELION_AOT_MODE=collect to tune each shape
    2. Run with HELION_AOT_MODE=measure to measure all configs across shapes
    3. Generate heuristics: python -m helion.experimental.aot_runner --generate
    4. Deploy with HELION_AOT_MODE=evaluate (default) to use heuristics

    Using collect_fn and measure_fn:
    - If collect_fn is set: in collect mode, only collect_fn() inputs are autotuned
    - If measure_fn is set: in measure mode, only measure_fn() inputs are measured
    - If both are set in collect mode (one-shot): autotune collect_fn inputs,
      then measure all discovered configs across measure_fn inputs

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
        collect_fn: Optional function that returns input tuples for autotuning.
            Each tuple contains arguments for one kernel invocation.
            Used to define which shapes to autotune during the collect phase.
        measure_fn: Optional function that returns input tuples for measurement.
            If set, only these inputs are used for the measure phase.
        **settings: Additional settings for the Kernel.

    Returns:
        Kernel: A Kernel object configured for AOT autotuning.

    Example:
        @helion.experimental.aot_kernel()
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
        @helion.experimental.aot_kernel(batched=[[0, None], None])
        def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
            # x has shape (batch, hidden), first dim is batched
            ...

        # Example with collect_fn and measure_fn:
        def my_collect_inputs():
            return [(torch.randn(1024, size, device="cuda"), 1e-5)
                    for size in [512, 1024, 2048, 4096]]

        def my_measure_inputs():
            return [(torch.randn(1024, size, device="cuda"), 1e-5)
                    for size in range(128, 4096, 128)]

        @helion.experimental.aot_kernel(
            batched=[[0, None], None],
            collect_fn=my_collect_inputs,
            measure_fn=my_measure_inputs,
        )
        def my_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
            ...

        # Example with custom key function - see examples/aot_example.py for
        # matmul_custom_key which demonstrates using key= to control which
        # features the heuristic uses. Key output is pytree-flattened:
        # (1024, 512, 256, 2) -> {key_0: 1024, key_1: 512, key_2: 256, key_3: 2}
    """
    from ..runtime.kernel import kernel

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
                collect_fn=collect_fn,
                measure_fn=measure_fn,
                key=user_key,
                **settings,
            ),
        )

    # Get kernel source file and name for heuristic-aware key
    kernel_source_file = fn.__code__.co_filename
    kernel_name = fn.__name__

    # Create the key function
    if user_key is not None:
        # User provided a key - create a composed key that:
        # 1. During collect/measure: uses user key for cache, features extracted from key output
        # 2. During evaluate: loads heuristic that works on flattened key values
        heuristic_key = make_aot_key(
            kernel_source_file, kernel_name, batched=batched, user_key=user_key
        )
        key_fn: KeyFunction = heuristic_key
    else:
        key_fn = make_aot_key(kernel_source_file, kernel_name, batched=batched)

    k = kernel(fn, config=config, configs=configs, key=key_fn, **settings)

    # Store collect_fn/measure_fn on the Kernel object for AOTAutotuneCache to access
    # This avoids global state and keeps the functions scoped to this specific kernel
    k._aot_collect_fn = collect_fn  # type: ignore[attr-defined]
    k._aot_measure_fn = measure_fn  # type: ignore[attr-defined]

    # Store user key function for AOTAutotuneCache to extract features from
    k._aot_user_key = user_key  # type: ignore[attr-defined]

    return k
