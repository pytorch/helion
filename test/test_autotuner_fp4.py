from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.accuracy import assert_close
from helion.autotuner.finite_search import FiniteSearch
import helion.language as hl


@pytest.mark.parametrize("chunk_size", [7, 1024])
@pytest.mark.parametrize("bit", [1, 16])
def test_packed_fp4_accuracy_checks_both_nibbles_exactly(
    chunk_size: int, bit: int
) -> None:
    expected = torch.arange(128, dtype=torch.uint8)[::2].view(torch.float4_e2m1fn_x2)
    actual = expected.clone()
    _chunked_assert_close(
        actual,
        expected,
        atol=100,
        rtol=100,
        chunk_size=chunk_size,
        scale_atol_by_expected_rms=True,
    )
    actual.view(torch.uint8)[-1] ^= bit
    with pytest.raises(AssertionError):
        _chunked_assert_close(
            actual,
            expected,
            atol=100,
            rtol=100,
            chunk_size=chunk_size,
            scale_atol_by_expected_rms=True,
        )


def test_packed_fp4_does_not_change_other_leaves_tolerances() -> None:
    packed = torch.tensor([0x17, 0xF2], dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    expected = (packed, torch.tensor([1.0]))
    assert_close(
        (packed.clone(), torch.tensor([1.005])),
        expected,
        atol=0.01,
        rtol=0,
        scale_atol_by_expected_rms=True,
    )
    with pytest.raises(AssertionError):
        assert_close(
            (packed.clone(), torch.tensor([1.5])),
            expected,
            atol=0.01,
            rtol=0,
            scale_atol_by_expected_rms=True,
        )
    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_close(packed, packed.view(torch.uint8), atol=0.01, rtol=0.01)


@skipIfNotCUDA()
@skipUnlessBackends(["triton", "cute"])
@skipIfRefEager("autotuning compiles candidates; ref mode runs the kernel eagerly")
def test_autotune_validates_packed_fp4_inputs_without_numeric_cast() -> None:
    @helion.kernel(
        static_shapes=True,
        autotune_benchmark_subprocess=False,
        autotune_precompile=None,
        autotune_log_level=0,
    )
    def write_output(
        packed: torch.Tensor, values: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        for tile in hl.tile(values.numel()):
            out[tile] = values[tile] + 1
        return out

    packed = torch.randint(256, (64,), device=DEVICE, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
    values = torch.randn(64, device=DEVICE)
    out = torch.empty_like(values)
    args = (packed, values, out)
    configs = [helion.Config(block_sizes=[16]), helion.Config(block_sizes=[32])]
    bound = write_output.bind(args)
    search = FiniteSearch(bound, args, configs=configs)
    best = search.autotune()
    assert best in configs
    # Writing an argument makes the autotuner validate all argument leaves,
    # including the unchanged FP4 input, with its default RMS-scaled tolerance.
    torch.testing.assert_close(bound.compile_config(best)(*args), values + 1)
