from __future__ import annotations

import torch

import helion
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


@helion.kernel()
def test(a):
    o = torch.empty_like(a)
    _triton_maxnreg = hl.register_tunable(
        "_triton_maxnreg", EnumFragment(choices=[32, 64])
    )
    for t in hl.tile(a.size(0)):
        o[t] = a[t]
    return o


test(torch.zeros(1024, device="cuda"))
