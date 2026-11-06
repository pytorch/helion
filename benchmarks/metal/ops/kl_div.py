"""kl_div: ``sum(target * (log(target) - pred)) / BT`` (batchmean)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from . import REFERENCE_ROWS
from helion._testing import import_example

_examples_kl_div = import_example("kl_div")
HelionKLDivLoss = _examples_kl_div.HelionKLDivLoss
kl_div_forward = _examples_kl_div.kl_div_forward

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (kl_div_forward,)
_helion_kl = HelionKLDivLoss(reduction="batchmean", log_target=False, eps=1e-10)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    bt, v = case.params["BT"], case.params["V"]
    return {
        "pred": torch.randn(bt, v, generator=gen).log_softmax(dim=-1),
        "target": torch.randn(bt, v, generator=gen).softmax(dim=-1),
    }


def reference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, pred.shape[0], REFERENCE_ROWS):
        rows = slice(start, start + REFERENCE_ROWS)
        t = target[rows].double()
        total += (t * (t.log() - pred[rows].double())).sum()
    return total / pred.shape[0]


def eager(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(pred, target, reduction="batchmean")


def helion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _helion_kl(pred, target)


def mlx(pred: mx.array, target: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.sum(target * (mx.log(target) - pred)) / pred.shape[0]
