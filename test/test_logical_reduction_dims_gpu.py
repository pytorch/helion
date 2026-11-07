from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["triton", "cute"])


@helion.kernel(static_shapes=False)
def _logical_unmerged(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    out = torch.empty((x.size(0),), device=x.device, dtype=torch.bool)
    for row in hl.tile(x.size(0)):
        if operation == "any":
            out[row] = x[row, :, :].any((1, 2))
        else:
            out[row] = x[row, :, :].all((1, 2))
    return out


@onlyBackends(["triton"])
def test_dynamic_logical_reductions_execute_unmerged_chains():
    # The CPU test_dynamic_vote_keeps_existing_layout_admission covers CuTe's
    # layout admission. Triton executes both single-axis stages here, detecting
    # incorrect original-overload axes.
    values = torch.ones((5, 7, 9), device=DEVICE)
    values[0] = 0
    values[1, 3, 4] = 0
    values[2] = 0
    values[2, -1, -1] = float("nan")
    for operation, reference in (
        ("any", values.any((1, 2))),
        ("all", values.all((1, 2))),
    ):
        _, actual = code_and_output(
            _logical_unmerged, (values, operation), block_sizes=[2]
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
