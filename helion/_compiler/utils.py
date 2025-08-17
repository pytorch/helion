from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export compute_slice_size from type_utils for backward compatibility
from .type_utils import compute_slice_size

if TYPE_CHECKING:
    import torch