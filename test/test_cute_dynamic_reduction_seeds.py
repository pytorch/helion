from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteRolledRowLadderHeuristic
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@skipUnlessBackends(["cute"])
@skipIfRefEager("asserts generated CuTe source, which ref eager does not emit")
def test_cute_dynamic_reduction_seeds_vector_memory(dtype: torch.dtype) -> None:
    @helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
    def softmax(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for rows in hl.tile(x.size(0)):
            out[rows, :] = torch.softmax(x[rows, :].float(), dim=-1).to(x.dtype)
        return out

    with _mock_cuda_unavailable():
        x = torch.empty(64, 4096, dtype=dtype)
        bound = _cpu_bind(softmax, (x,))
        assert bound.host_function is not None
        spec = bound.config_spec
        seed = CuteRolledRowLadderHeuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        assert seed is not None
        generation = ConfigGeneration(spec)
        _, seed = generation.canonicalize_flat(generation.flatten(seed))
        assert seed in [config for _, config in generation.seed_flat_config_pairs()]
        reduction_id = spec.reduction_loops[0].block_id
        vector_index = spec.cute_vector_widths.block_id_to_index(reduction_id)
        assert seed.config["cute_vector_widths"][vector_index] == 16 // dtype.itemsize
        assert "cute.arch.load(" in bound.to_code(seed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "offset"])
@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@skipIfRefEager("checks compiled dynamic seed execution")
def test_dynamic_seed_handles_changed_extent_and_alignment(
    dtype: torch.dtype, layout: str
) -> None:
    def softmax(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for rows in hl.tile(x.size(0)):
            out[rows, :] = torch.softmax(x[rows, :].float(), dim=-1).to(x.dtype)
        return out

    kernel = helion.kernel(
        softmax, backend="cute", static_shapes=False, autotune_effort="none"
    )
    first = torch.randn(5, 4096, device="cuda", dtype=dtype)
    bound = kernel.bind((first,))
    assert bound.host_function is not None
    seed = CuteRolledRowLadderHeuristic.get_seed_config(
        bound.env, bound.host_function.device_ir
    )
    assert seed is not None
    # Use public dispatch so a changed pointer/stride residue gets its own
    # binding rather than bypassing the runtime's alignment specialization.
    seeded = helion.kernel(
        softmax,
        config=seed,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    for width in (4096, 4100):
        if layout == "padded":
            x = torch.randn(5, width + 8, device="cuda", dtype=dtype)[:, :width]
        elif layout == "offset":
            x = torch.randn(5, width + 1, device="cuda", dtype=dtype)[:, 1:]
        else:
            x = torch.randn(5, width, device="cuda", dtype=dtype)
        actual = seeded(x)
        expected = torch.softmax(x.float(), dim=-1).to(dtype)
        torch.testing.assert_close(actual, expected)
        assert seeded.bind((x,)) is seeded.bind((x,))
