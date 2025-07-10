from __future__ import annotations

import torch


def get_gpu_memory_info(device_id: int | None = None) -> tuple[float, float]:
    """
    Get total and available GPU memory in GB.

    Args:
        device_id: GPU device ID. If None, uses current device.

    Returns:
        Tuple of (total_memory_gb, available_memory_gb)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)

    if device_id is None:
        device_id = torch.cuda.current_device()

    # Get total memory
    total_memory = torch.cuda.get_device_properties(device_id).total_memory

    # Get reserved memory (memory allocated by the caching allocator)
    reserved_memory = torch.cuda.memory_reserved(device_id)

    # Available memory is approximately total - reserved
    available_memory = total_memory - reserved_memory

    # Convert to GB
    total_gb = total_memory / (1024**3)
    available_gb = available_memory / (1024**3)

    return (total_gb, available_gb)
