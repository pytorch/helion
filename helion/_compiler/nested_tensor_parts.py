from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    import torch


class NestedTensorParts(NamedTuple):
    """Decomposed representation of a PyTorch NestedTensor for compilation.

    Stores the values buffer, offsets array, and precomputed max jagged length
    so the compiler can generate flat indexing and masking automatically.
    """

    values: torch.Tensor  # [total_elements, *dense_dims]
    offsets: torch.Tensor  # [num_rows + 1]
    max_length: int  # max(offsets[1:] - offsets[:-1])
