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

from collections.abc import Iterable
import functools
import importlib.util
import logging
import os
from pathlib import Path
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

from .._argument_device import _ArgumentDeviceResolver
from .aot_compile import kernel_source_identity

if TYPE_CHECKING:
    from types import CodeType

    from .._hardware import HardwareInfo
    from ..runtime.kernel import ConfigLike
    from ..runtime.kernel import Kernel


_R = TypeVar("_R")

# Type alias for key functions
KeyFunction = Callable[..., Hashable]

# Type alias for input generator functions (collect_fn/measure_fn)
# Returns an iterable of argument tuples for the kernel
InputFn = Callable[[], Iterable[tuple[Any, ...]]]

log: logging.Logger = logging.getLogger(__name__)


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

    ``HELION_AOT_MODE`` is process configuration and must remain unchanged after
    an instance is first called. In non-disabled modes,
    ``HELION_AOT_DATA_DIR`` has the same requirement. If it is unset, the
    current working directory on first active use supplies the default; later
    working-directory changes do not retarget that pinned directory.
    """

    # Class-level cache: (kernel source, kernel name, resolved heuristic path)
    # -> exact (key function, artifact identity) resolution. The path varies
    # across hardware and override dirs.
    _key_fn_cache: ClassVar[
        dict[
            tuple[str, str, str | None],
            tuple[KeyFunction | None, str | None],
        ]
    ] = {}

    def __init__(
        self,
        kernel_source_file: str,
        kernel_name: str,
        batched: BatchedSpec = None,
        user_key: KeyFunction | None = None,
        code_object: CodeType | None = None,
    ) -> None:
        source_path, self._kernel_source_identity = kernel_source_identity(
            kernel_source_file,
            code_object,
        )
        self.kernel_source_file = None if source_path is None else str(source_path)
        self.kernel_name = kernel_name
        self.batched = batched
        self.user_key = user_key
        self._resolution_cache: dict[
            tuple[
                torch.device | None,
                tuple[str, str | None] | None,
                HardwareInfo,
            ],
            tuple[KeyFunction | None, str | None],
        ] = {}
        self._argument_device_resolver = _ArgumentDeviceResolver()
        self._aot_mode: str | None = None
        self._data_dir: Path | None = None
        self._aot_data_dir_setting: str | None = None
        self._aot_data_dir_initialized = False
        self._heuristic_dir_identities: dict[tuple[str, str | None], Path] = {}

    def _resolve_argument_device(self, args: Sequence[object]) -> torch.device | None:
        """Avoid a recursive container walk on the steady-state dispatch path."""
        return self._argument_device_resolver.resolve(args)

    def _resolve_aot_mode(self) -> str:
        """Pin and validate the process-wide AOT phase."""

        current_aot_mode = os.environ.get("HELION_AOT_MODE", "evaluate").lower()
        if self._aot_mode is None:
            self._aot_mode = current_aot_mode
        elif current_aot_mode != self._aot_mode:
            raise RuntimeError(
                "HELION_AOT_MODE changed from "
                f"{self._aot_mode!r} to {current_aot_mode!r} after this AOT key "
                "function was first used. Run each AOT mode in a fresh process."
            )
        return self._aot_mode

    def _validate_aot_data_dir_setting(self, current_setting: str | None) -> None:
        if current_setting != self._aot_data_dir_setting:
            raise RuntimeError(
                "HELION_AOT_DATA_DIR setting changed from "
                f"{self._aot_data_dir_setting!r} to {current_setting!r} after "
                "this AOT key function was first used. The resolved AOT data "
                "directory is pinned on first active use; run each setting in "
                "a fresh process."
            )

    def validate_process_settings(self) -> None:
        """Reject AOT process-setting changes on direct bound-kernel calls."""

        aot_mode = self._resolve_aot_mode()
        if aot_mode != "disabled":
            assert self._aot_data_dir_initialized
            self._validate_aot_data_dir_setting(os.environ.get("HELION_AOT_DATA_DIR"))

    def _resolve_aot_data_dir(self) -> Path:
        """Pin the first active directory without steady-state filesystem work.

        With no explicit setting, the default captures ``Path.cwd()`` on this
        first call. A later ``chdir`` intentionally does not retarget it.
        """
        current_setting = os.environ.get("HELION_AOT_DATA_DIR")
        if not self._aot_data_dir_initialized:
            from .aot_cache import get_aot_data_dir

            data_dir = get_aot_data_dir().expanduser().resolve()
            self._aot_data_dir_setting = current_setting
            self._data_dir = data_dir
            self._aot_data_dir_initialized = True
            return data_dir

        self._validate_aot_data_dir_setting(current_setting)

        assert self._data_dir is not None
        return self._data_dir

    def _resolve_heuristic(
        self, args: Sequence[object]
    ) -> tuple[KeyFunction | None, str | None]:
        """Return the key function and file identity for the current call."""
        # AOT phases and the active modes' data directory are process
        # configuration: the runner executes each phase in a fresh subprocess.
        # Device and HELION_HEURISTIC_DIR remain dynamic dispatch inputs because
        # callers may switch them within one process.
        aot_mode = self._resolve_aot_mode()

        # Evaluate uses the generated key function. Compile retains its full
        # fallback key plus the selected heuristic identity so BoundKernel
        # caching stays isolated across override files. Disabled mode must not
        # discover or validate AOT artifacts at all.
        if aot_mode == "disabled":
            return None, None

        data_dir = self._resolve_aot_data_dir()
        if aot_mode in ("collect", "measure"):
            return None, None

        device = self._resolve_argument_device(args)
        from .aot_cache import HEURISTIC_DIR_ENV
        from .aot_cache import HeuristicArtifactMetadataError
        from .aot_cache import find_heuristic_file
        from .aot_cache import get_heuristic_hardware
        from .aot_cache import heuristic_artifact_identity
        from .aot_cache import heuristic_module_supports_hardware
        from .aot_cache import validate_runner_hardware

        heuristic_dir_value = os.environ.get(HEURISTIC_DIR_ENV)
        heuristic_dir_key: tuple[str, str | None] | None = None
        heuristic_dir_path: Path | None = None
        if heuristic_dir_value is not None:
            heuristic_dir_path = Path(heuristic_dir_value).expanduser()
            heuristic_dir_key = (
                str(heuristic_dir_path),
                None if heuristic_dir_path.is_absolute() else os.getcwd(),
            )
        hardware = get_heuristic_hardware(device)
        resolution_key = (device, heuristic_dir_key, hardware)
        if resolution_key in self._resolution_cache:
            return self._resolution_cache[resolution_key]
        # The key function runs before AOTAutotuneCache is constructed. Guard
        # source-adjacent and override artifacts here so no heuristic code is
        # loaded before the runner validates the child argument device.
        validate_runner_hardware(data_dir, hardware)

        resolved_heuristic_dir: Path | None = None
        if heuristic_dir_key is not None:
            resolved_heuristic_dir = self._heuristic_dir_identities.get(
                heuristic_dir_key
            )
            if resolved_heuristic_dir is None:
                assert heuristic_dir_path is not None
                resolved_heuristic_dir = heuristic_dir_path.resolve()
                self._heuristic_dir_identities[heuristic_dir_key] = (
                    resolved_heuristic_dir
                )

        # Use shared heuristic file discovery. Hardware identity failures and
        # programming errors remain visible; only filesystem discovery failures
        # fall back to the full specialization key.
        try:
            heuristic_path = find_heuristic_file(
                self.kernel_source_file,
                kernel_name=self.kernel_name,
                data_dir=data_dir,
                device=device,
                resolved_heuristic_dir=resolved_heuristic_dir,
            )
        except OSError:
            log.debug(
                "Failed to discover an AOT heuristic for %s",
                self.kernel_name,
                exc_info=True,
            )
            result: tuple[KeyFunction | None, str | None] = (None, None)
            self._resolution_cache[resolution_key] = result
            return result

        if heuristic_path is None:
            heuristic_identity = None
        else:
            heuristic_identity = heuristic_artifact_identity(heuristic_path, hardware)
            if heuristic_identity is None:
                heuristic_path = None

        if aot_mode != "evaluate":
            result = (None, heuristic_identity)
            self._resolution_cache[resolution_key] = result
            return result

        cache_key = (
            self._kernel_source_identity,
            self.kernel_name,
            heuristic_identity,
        )

        # Check class-level cache after resolving the current hardware/path.
        if cache_key in HeuristicKeyFunction._key_fn_cache:
            result = HeuristicKeyFunction._key_fn_cache[cache_key]
            self._resolution_cache[resolution_key] = result
            return result

        if heuristic_path is not None:
            try:
                spec = importlib.util.spec_from_file_location(
                    "heuristic", heuristic_path
                )
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    try:
                        supports_hardware = heuristic_module_supports_hardware(
                            module,
                            heuristic_path,
                            hardware,
                        )
                    except HeuristicArtifactMetadataError:
                        log.warning(
                            "Skipping heuristic with invalid artifact metadata: %s",
                            heuristic_path,
                            exc_info=True,
                        )
                        result = (None, None)
                        HeuristicKeyFunction._key_fn_cache[cache_key] = result
                        self._resolution_cache[resolution_key] = result
                        return result
                    if not supports_hardware:
                        result = (None, None)
                        HeuristicKeyFunction._key_fn_cache[cache_key] = result
                        self._resolution_cache[resolution_key] = result
                        return result

                    # find_heuristic_file already filtered unsupported hardware;
                    # the loaded-module check above validates runtime metadata.
                    key_fn = getattr(module, f"key_{self.kernel_name}", None)
                    if key_fn is not None:
                        result = (key_fn, heuristic_identity)
                        HeuristicKeyFunction._key_fn_cache[cache_key] = result
                        self._resolution_cache[resolution_key] = result
                        return result
            except Exception:
                log.warning(
                    "Failed to load AOT heuristic key from %s",
                    heuristic_path,
                    exc_info=True,
                )

        result = (None, heuristic_identity)
        HeuristicKeyFunction._key_fn_cache[cache_key] = result
        self._resolution_cache[resolution_key] = result
        return result

    def __call__(self, *args: object) -> Hashable:
        """Generate specialization key from arguments."""
        heuristic_key_fn, heuristic_identity = self._resolve_heuristic(args)

        if self.user_key is not None:
            user_key_value = self.user_key(*args)
            if heuristic_key_fn is not None:
                # User-key heuristics are trained on the flattened key output,
                # so their generated key is the minimal runtime projection.
                key_value = heuristic_key_fn(*_flatten_key_value(user_key_value))
            else:
                key_value = user_key_value

        elif heuristic_key_fn is not None:
            # Use the heuristic's key function directly on args
            key_value = heuristic_key_fn(*args)
        else:
            # Fallback: use all features
            key_value = aot_key(*args, batched=self.batched)

        if heuristic_identity is None:
            return key_value
        return ("helion_aot_heuristic", heuristic_identity, key_value)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear only the shared cache of heuristic key resolutions.

        Existing instances retain their resolved AOT mode, data directory,
        device path, and per-hardware resolution cache.
        """
        cls._key_fn_cache.clear()


def make_aot_key(
    kernel_source_file: str,
    kernel_name: str,
    batched: BatchedSpec = None,
    user_key: KeyFunction | None = None,
    code_object: CodeType | None = None,
) -> HeuristicKeyFunction:
    """
    Create an AOT key function for a specific kernel.

    Args:
        kernel_source_file: Path to the kernel's source file
        kernel_name: Name of the kernel function
        batched: Optional batch dimension specification (see extract_shape_features)
        user_key: Optional user-provided key function. If provided, the heuristic
            will be trained on the flattened output of this function.
        code_object: Optional code object used to isolate non-file-backed kernels.

    Returns:
        A callable that generates specialization keys from kernel arguments
    """
    return HeuristicKeyFunction(
        kernel_source_file,
        kernel_name,
        batched=batched,
        user_key=user_key,
        code_object=code_object,
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
    standalone: bool = True,
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
    standalone: bool = True,
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
    standalone: bool = True,
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
    - In evaluate mode: minimizes automatic keys to the features the heuristic needs
    - In evaluate/compile mode: namespaces every specialization key with the
      resolved heuristic artifact identity
    - In collect/measure modes: uses all features (full coverage)
    - Optional collect_fn/measure_fn to specify inputs for collect/measure phases

    The AOT workflow is:
    1. Run benchmarks with HELION_AOT_MODE=collect to tune each shape
    2. Run with HELION_AOT_MODE=measure to measure all configs across shapes
    3. Generate heuristics: python -m helion.autotuner.aot_runner --generate
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
        standalone: Whether ``HELION_AOT_MODE=compile`` may emit a standalone
            dispatcher for this kernel. Set this to false when different
            runtime specializations can share one heuristic configuration.
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

        # Example with collect_fn and measure_fn:
        def my_collect_inputs():
            return [(torch.randn(1024, size, device="cuda"), 1e-5)
                    for size in [512, 1024, 2048, 4096]]

        def my_measure_inputs():
            return [(torch.randn(1024, size, device="cuda"), 1e-5)
                    for size in range(128, 4096, 128)]

        @helion.aot_kernel(
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
                standalone=standalone,
                key=user_key,
                **settings,
            ),
        )

    # Get kernel source file and name for heuristic-aware key
    kernel_source_file = fn.__code__.co_filename
    kernel_name = fn.__name__

    # A user key is composed inside HeuristicKeyFunction: collect/measure use it
    # directly, while evaluate applies the generated heuristic to its flattened
    # output.
    heuristic_key = make_aot_key(
        kernel_source_file,
        kernel_name,
        batched=batched,
        user_key=user_key,
        code_object=fn.__code__,
    )

    k = kernel(
        fn,
        config=config,
        configs=configs,
        key=heuristic_key,
        _bound_call_validator=heuristic_key.validate_process_settings,
        **settings,
    )

    # Store collect_fn/measure_fn on the Kernel object for AOTAutotuneCache to access
    # This avoids global state and keeps the functions scoped to this specific kernel
    k._aot_collect_fn = collect_fn
    k._aot_measure_fn = measure_fn
    k._aot_standalone = standalone

    # Store user key function for AOTAutotuneCache to extract features from
    k._aot_user_key = user_key

    return k
