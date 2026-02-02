# Common utilities for FLA-based kernels
# Extracted from the fla package for standalone use

import contextlib
import functools
import inspect
import os
from functools import lru_cache

import torch
import triton
from packaging import version

# Environment variables
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
TRITON_ABOVE_3_4_0 = version.parse(triton.__version__) >= version.parse("3.4.0")

# Check if autotune cache is supported
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


@functools.cache
def get_available_device() -> str:
    """Get the available device backend."""
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        return 'cpu'


def map_triton_backend_to_torch_device() -> str:
    """Map triton backend to torch device name."""
    backend = get_available_device()
    return {'cuda': 'cuda', 'xpu': 'xpu'}.get(backend, backend)


# Device detection
device_platform = get_available_device()
device_name = map_triton_backend_to_torch_device()
device = device_platform
device_torch_lib = getattr(torch, device, torch.cuda)

# Platform flags
IS_INTEL = (device_platform == 'xpu')
IS_NVIDIA = (device_platform == 'cuda')
IS_INTEL_ALCHEMIST = False
IS_NVIDIA_HOPPER = False
IS_NVIDIA_BLACKWELL = False
IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')
IS_TMA_SUPPORTED = False

# Try to get more detailed device info
try:
    if IS_INTEL and torch.xpu.is_available():
        IS_INTEL_ALCHEMIST = 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0)
    if IS_NVIDIA and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        IS_NVIDIA_HOPPER = ('NVIDIA H' in torch.cuda.get_device_name(0) or cap[0] >= 9)
        IS_NVIDIA_BLACKWELL = (cap[0] == 10)
        # TMA support requires Hopper (sm90+) and explicit opt-in
        IS_TMA_SUPPORTED = (cap[0] >= 9) and \
            os.getenv('FLA_USE_TMA', '0') == '1' and \
            (hasattr(triton.language, '_experimental_make_tensor_descriptor') or
             hasattr(triton.language, 'make_tensor_descriptor'))
except Exception:
    pass

# TF32 support
IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'


class Backend:
    """Backend shared memory sizes."""
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        arch_map = {
            'ada': cls.ADA,
            'ampere': cls.AMPERE,
            'hopper': cls.HOPPER,
        }
        return arch_map.get(arch.lower(), cls.DEFAULT)


def get_all_max_shared_mem():
    """Get max shared memory for all devices."""
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Check if device has enough shared memory."""
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


@functools.cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    """Check if PyTorch version is at least the specified version."""
    return version.parse(torch.__version__) >= version.parse(version_s)


# Custom autocast decorators
if check_pytorch_version('2.4'):
    _device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=_device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=_device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


def tensor_cache(fn):
    """
    A decorator that caches the most recent result of a function with tensor inputs.
    """
    last_args = None
    last_kwargs = None
    last_result = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result

        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(fn=None, *, no_guard_contiguous=False):
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """
    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def contiguous(fn):
    """Alias for input_guard() without parameters."""
    return input_guard(fn)


def get_abs_err(x, y):
    """Get absolute error between two tensors."""
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    """Get error ratio between two tensors."""
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    """Assert two tensors are close within the given ratio."""
    import logging
    import warnings
    logger = logging.getLogger(__name__)

    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning:
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


# CUDA graph support
USE_CUDA_GRAPH = os.getenv('FLA_USE_CUDA_GRAPH', '0') == '1'


def get_multiprocessor_count():
    """Get the number of SMs on the current device."""
    try:
        if IS_NVIDIA and torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        elif IS_INTEL and hasattr(torch, 'xpu') and torch.xpu.is_available():
            # XPU doesn't have a direct equivalent, use a reasonable default
            return 128
    except Exception:
        pass
    return 128  # Default fallback


# Lower case aliases for compatibility
is_intel = IS_INTEL
is_nvidia = IS_NVIDIA
is_intel_alchemist = IS_INTEL_ALCHEMIST
is_nvidia_hopper = IS_NVIDIA_HOPPER
is_nvidia_blackwell = IS_NVIDIA_BLACKWELL
is_gather_supported = IS_GATHER_SUPPORTED
is_tf32_supported = IS_TF32_SUPPORTED
is_tma_supported = IS_TMA_SUPPORTED
