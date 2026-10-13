from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_warp_bridge import _code as _bridge_code
import helion
from helion import exc
from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_matmul as chain
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _retained_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    n = b.shape[2]
    out = torch.empty((batches, m, n), dtype=a.dtype, device=a.device)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        value = hl.dot(a[bi, row, kk], b[bi, kk, col])
        out[bi, row, col] = (value + a[bi, row, col].float()).to(a.dtype)
    return out


def _code(route: str, dtype: torch.dtype = torch.bfloat16) -> str:
    if route == "bridge":
        return _bridge_code(dtype=dtype, scan=True)
    with _cpu_codegen():
        return _retained_single._bind_isolated(
            (
                torch.empty((2, 32, 64), dtype=dtype),
                torch.empty((2, 64, 64), dtype=dtype),
            )
        ).to_code(
            helion.Config(
                num_warps=4, cute_chained_mma_schedule="cp_async_register_reuse"
            )
        )


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_final_staged_original_owner_and_values(route: str, dtype: torch.dtype) -> None:
    original = bodies.emit_body_program
    seen = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        body = kwargs["root_warp"]
        assert body.staged and body.consumed
        body.validate_return(result)
        seen.append(body)
        return result

    with patch.object(bodies, "emit_body_program", observe):
        source = _code(route, dtype)
    assert "chain_store" in source and len(seen) == 1


@pytest.mark.parametrize("route", ("singleton", "bridge"))
@pytest.mark.parametrize(
    "change", ("list", "clear", "shared", "coordinates", "inverse_axes")
)
def test_final_staged_mutation_rejects_before_store(route: str, change: str) -> None:
    original = bodies.emit_body_program
    seen = []

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        body = kwargs["root_warp"]
        assert body.staged and body.consumed
        if change == "list":
            body.staged = list(body.staged)
        elif change == "clear":
            body.staged.clear()
        elif change == "shared":
            object.__setattr__(body.staged[0], "shared", "unapproved")
        elif change == "coordinates":
            item = body.staged[0]
            object.__setattr__(item, "coordinates", item.coordinates[::-1])
        else:
            item = body.staged[0]
            object.__setattr__(item, "inverse_axes", item.inverse_axes[::-1])
        seen.append(body)
        return result

    with (
        patch.object(bodies, "emit_body_program", mutate),
        patch.object(chain, "_emit_store", side_effect=AssertionError("store reached")),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(route)
    assert len(seen) == 1
