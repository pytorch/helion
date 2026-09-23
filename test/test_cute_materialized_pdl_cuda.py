"""Real CUDA coverage for the ordinary transformed-RHS materialized pipeline."""

from __future__ import annotations

import pytest
import torch

from test.test_cute_materialize_operand import _computed_rhs

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pdl", [False, True], ids=["serial", "pdl"])
def test_transformed_rhs_numerics_and_graph(pdl: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    generator = torch.Generator().manual_seed(257)
    a_cpu = torch.randn((128, 256), generator=generator, dtype=torch.bfloat16)
    b_cpu = torch.randn((256, 128), generator=generator, dtype=torch.bfloat16)
    a, b = a_cpu.to(DEVICE), b_cpu.to(DEVICE)
    bound = helion.kernel(
        _computed_rhs.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )._bind_isolated((a, b))
    group = next(
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == "tcgen05_materialized_pdl"
    )
    config = group.witnesses[2].carrier
    config.config["tcgen05_materialized_pdl"] = pdl
    bound.set_config(config)

    def call() -> torch.Tensor:
        return bound(a, b)

    def expected() -> torch.Tensor:
        # CPU reference keeps whole-process sanitizer runs free of an unrelated
        # vendor GEMM. Preserve the source's BF16 rounding before contraction.
        rhs = (b_cpu.float() * 0.25 + 2).to(a_cpu.dtype)
        return (a_cpu.float() @ rhs.float()).to(a_cpu.dtype)

    torch.testing.assert_close(call().cpu(), expected(), atol=0.25, rtol=1e-2)
    replay = _make_cudagraph_replay(call)
    for _ in range(2):
        a_cpu.normal_(generator=generator)
        b_cpu.normal_(generator=generator)
        a.copy_(a_cpu)
        b.copy_(b_cpu)
        reference = expected()
        torch.testing.assert_close(call().cpu(), reference, atol=0.25, rtol=1e-2)
        for _ in range(3):
            actual = replay()
            torch.testing.assert_close(actual.cpu(), reference, atol=0.25, rtol=1e-2)
            actual.fill_(float("nan"))
    torch.cuda.synchronize()
