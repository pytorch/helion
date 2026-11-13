from __future__ import annotations

from typing import Any

import pytest
import torch

import helion
from helion._compiler.cute.state import cute_env_state
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_promote_half_atomic_outputs=True,
)
def _aliased_output_matmul(
    a: torch.Tensor, b: torch.Tensor, broadcast: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    alias = out[:1, :].expand(m, n) if broadcast else out.view(m, n)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        previous = alias[row, column]
        hl.atomic_add(out, [row, column], acc + previous, sem="relaxed")
    return out


def _metadata_receiver(method: Any) -> torch.Tensor:
    return method.__self__


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_promote_half_atomic_outputs=True,
)
def _metadata_alias_output_matmul(
    a: torch.Tensor, b: torch.Tensor, broadcast: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    receiver = _metadata_receiver(out.size)
    alias = receiver[:1, :].expand(m, n) if broadcast else receiver.view(m, n)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        previous = alias[row, column]
        hl.atomic_add(out, [row, column], acc + previous, sem="relaxed")
    return out


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("metadata_alias", [False, True])
def test_output_storage_aliases_preserve_requested_half_atomics(
    collective: bool, broadcast: bool, metadata_alias: bool
) -> None:
    with cpu_target():
        kernel = (
            _metadata_alias_output_matmul if metadata_alias else _aliased_output_matmul
        )
        args = (
            torch.empty((64, 512), dtype=torch.float16),
            torch.empty((512, 128), dtype=torch.float16),
            broadcast,
        )
        bound = cpu_bind(kernel, args)
        assert not cute_env_state(bound.env).half_atomic_output_promotions
        config = helion.Config(
            block_sizes=[32, 64, 32],
            num_threads=[2, 64, 1],
            cute_collective_mma=collective,
            cute_collective_compute="warp",
            cute_collective_copy="async_cached",
            cute_collective_stages=4,
        )
        if metadata_alias:
            # Recovering the receiver executes opaque Python, which may change any
            # captured storage. It cannot authorize collective memory reordering.
            for name in ("a", "b", "out"):
                tensor = bound.host_function.local_types[name].fake_value
                assert bound.env.tensor_storage_provenance(tensor) is None
            if collective:
                with pytest.raises(
                    helion.exc.BackendUnsupported,
                    match="operand loads may alias row-loop writes",
                ):
                    bound.to_code(config)
                return
        code = bound.to_code(config)
        assert "dtype=a.dtype, device=a.device" in code
        assert "val=cutlass.Float16(" in code
        assert "return out.to(torch.float16)" not in code
