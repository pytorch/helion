"""jsd: generalized Jensen-Shannon divergence (beta = 0.5) of log-probabilities.

``loss = sum(beta * P * (log_p - log M) + (1 - beta) * Q * (log_q - log M)) / BT``
with ``P = exp(log_p)``, ``Q = exp(log_q)`` and ``M = beta * P + (1 - beta) * Q``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import REFERENCE_ROWS
from helion._testing import import_example

_examples_jsd = import_example("jsd")
HelionJSD = _examples_jsd.HelionJSD
TorchJSDBaseline = _examples_jsd.TorchJSDBaseline
jsd_forward = _examples_jsd.jsd_forward

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (jsd_forward,)
BETA = 0.5
_helion_jsd = HelionJSD(beta=BETA, ignore_index=-100, dtype=torch.float)
_torch_jsd = TorchJSDBaseline(beta=BETA, ignore_index=-100, dtype=torch.float)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    bt, v = case.params["BT"], case.params["V"]
    return {
        "log_q": torch.randn(bt, v, generator=gen).log_softmax(dim=-1),
        "log_p": torch.randn(bt, v, generator=gen).log_softmax(dim=-1),
    }


def _jsd_sum(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    p, q = log_p.exp(), log_q.exp()
    log_m = torch.lerp(q, p, BETA).log()
    return (BETA * p * (log_p - log_m) + (1 - BETA) * q * (log_q - log_m)).sum()


def reference(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, log_q.shape[0], REFERENCE_ROWS):
        rows = slice(start, start + REFERENCE_ROWS)
        total += _jsd_sum(log_q[rows].double(), log_p[rows].double())
    return total / log_q.shape[0]


def eager(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    return _torch_jsd(log_q, log_p)


def helion(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    return _helion_jsd(log_q, log_p)


def mlx(log_q: mx.array, log_p: mx.array) -> mx.array:
    import mlx.core as mx

    p, q = mx.exp(log_p), mx.exp(log_q)
    log_m = mx.log(q + BETA * (p - q))
    terms = BETA * p * (log_p - log_m) + (1 - BETA) * q * (log_q - log_m)
    return mx.sum(terms) / log_q.shape[0]
