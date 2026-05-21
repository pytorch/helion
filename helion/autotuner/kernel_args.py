from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@functools.cache
def load_trusted_kernel_args(path: str) -> Sequence[object]:
    # Cached so re-spawning configs don't re-read the same args off disk.
    # This file is a trusted temporary artifact written by the parent
    # autotuner process. Kernel args can include user Python objects such as
    # callable epilogues, which PyTorch's weights-only loader rejects.
    return cast("Sequence[object]", torch.load(path, weights_only=False))
