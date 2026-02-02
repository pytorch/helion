# VLLM Compatibility Layer
# Provides minimal functionality needed for standalone tests without requiring full vllm installation

import random
import torch
import numpy as np


# STR_DTYPE_TO_TORCH_DTYPE from vllm.utils.torch_utils
STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
}


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


class Platform:
    """Minimal platform detection for tests."""

    def __init__(self):
        self._is_cuda = torch.cuda.is_available()
        self._is_xpu = False

        # Try to detect XPU
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self._is_xpu = True

    def is_cuda(self) -> bool:
        return self._is_cuda

    def is_xpu(self) -> bool:
        return self._is_xpu

    def is_cpu(self) -> bool:
        return not (self._is_cuda or self._is_xpu)

    def get_device_name(self) -> str:
        if self._is_cuda:
            return torch.cuda.get_device_name(0)
        elif self._is_xpu:
            return "Intel XPU"
        return "CPU"

    def get_device_capability(self, device_id: int = 0):
        """Get CUDA device capability."""
        if self._is_cuda:
            return torch.cuda.get_device_capability(device_id)
        return (0, 0)

    def is_cuda_alike(self) -> bool:
        """Returns True if CUDA."""
        return self._is_cuda

    @staticmethod
    def inference_mode():
        """Context manager for inference mode."""
        return torch.inference_mode()


# Global platform instance
current_platform = Platform()


# Logger stub
class Logger:
    def info(self, msg, *args):
        pass

    def debug(self, msg, *args):
        pass

    def warning(self, msg, *args):
        pass

    def error(self, msg, *args):
        pass


def init_logger(name: str) -> Logger:
    return Logger()
