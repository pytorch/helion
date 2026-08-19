from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class FlashHardwareCapabilities:
    """Hardware features used by CuTe flash-attention lowerings."""

    supports_tmem_row_reduce: bool = False
    supports_packed_f16x2_exp2: bool = False
