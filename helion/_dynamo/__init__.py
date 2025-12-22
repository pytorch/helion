"""Helion Dynamo integration module.

This module provides integration with PyTorch's torch.compile (Dynamo)
by registering Helion kernel handling at import time.
"""
from __future__ import annotations

from helion._dynamo.higher_order_ops import (
    helion_kernel_wrapper_mutation,
    get_helion_kernel,
)
from helion._dynamo.variables import (
    HelionKernelSideTable,
    HelionKernelVariable,
    helion_kernel_side_table,
)
from helion._dynamo.registration import register_with_dynamo

__all__ = [
    "HelionKernelSideTable",
    "HelionKernelVariable",
    "helion_kernel_side_table",
    "helion_kernel_wrapper_mutation",
    "get_helion_kernel",
    "register_with_dynamo",
]

# Register with Dynamo on import
register_with_dynamo()
