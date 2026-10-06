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


def _producer_seeds(bound: helion.runtime.kernel.BoundKernel) -> list[helion.Config]:
    """The one-CTA small-grid seed and the deepest 128x64 paired seed, both
    with the flat producer grid and the programmatic dependent launch."""
    spec = bound.config_spec
    chosen: dict[str, helion.Config] = {}
    with bound.env:
        for seed in spec.compiler_seed_configs:
            values = seed.config
            widths = values.get("cute_vector_widths", [])
            if (
                spec.cute_vector_widths.config_get(widths, 1) != 8
                or values.get("cute_pointwise_pid_type") != "flat"
                or values.get("tcgen05_materialized_pdl") is not True
            ):
                continue
            if values.get("tcgen05_cta_group") == "auto" and tuple(
                seed.block_sizes[2:4]
            ) < tuple(
                chosen.get(
                    "one_cta", helion.Config(block_sizes=[0, 0, 512, 512])
                ).block_sizes[2:4]
            ):
                # The smallest one-CTA tile: 64 rows where the search admits
                # them (M < 256), otherwise 128 rows.
                chosen["one_cta"] = seed
            if (
                values.get("tcgen05_cta_group") == "two"
                and seed.block_sizes[2:] == [128, 64, 64]
                and values.get("tcgen05_acc_stages") == 2
                and values.get("tcgen05_ab_stages", 0)
                > chosen.get("paired", helion.Config()).config.get(
                    "tcgen05_ab_stages", 0
                )
            ):
                chosen["paired"] = seed
        assert set(chosen) == {"one_cta", "paired"}, list(chosen)
        return [
            bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | seed.config)
            )
            for seed in chosen.values()
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape", [(128, 256, 128), (512, 1024, 512), (1024, 2048, 1024)]
)
@pytest.mark.parametrize("transpose", [False, True], ids=["bf16xint16", "int16xbf16"])
def test_bf16xint16_seeds_match_torch_matmul(
    shape: tuple[int, int, int], transpose: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    from examples.bf16xint16_gemm import _bf16xint16_gemm
    from examples.bf16xint16_gemm import _int16xbf16_gemm

    m, k, n = shape
    generator = torch.Generator().manual_seed(m + k + n)

    def inputs() -> tuple[torch.Tensor, torch.Tensor]:
        if transpose:
            x = torch.randint(
                -(2**15), 2**15 - 1, (m, k), generator=generator, dtype=torch.int16
            )
            w = torch.randn((k, n), generator=generator, dtype=torch.bfloat16)
        else:
            x = torch.randn((m, k), generator=generator, dtype=torch.bfloat16)
            w = torch.randint(
                -(2**15), 2**15 - 1, (k, n), generator=generator, dtype=torch.int16
            )
        return x.to(DEVICE), w.to(DEVICE)

    def expected(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if transpose:
            return torch.matmul(x.to(torch.bfloat16), w)
        return torch.matmul(x, w.to(torch.bfloat16))

    x, w = inputs()
    example = _int16xbf16_gemm if transpose else _bf16xint16_gemm
    bound = helion.kernel(
        example.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )._bind_isolated((x, w))
    for config in _producer_seeds(bound):
        bound.set_config(config)
        torch.testing.assert_close(bound(x, w), expected(x, w), atol=1e-2, rtol=1e-2)
        replay = _make_cudagraph_replay(lambda: bound(x, w))
        for _ in range(2):
            fresh_x, fresh_w = inputs()
            x.copy_(fresh_x)
            w.copy_(fresh_w)
            reference = expected(x, w)
            actual = replay()
            torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
            actual.fill_(float("nan"))
    torch.cuda.synchronize()
