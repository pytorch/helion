from __future__ import annotations

from examples.moe_matmul_ogs import moe_matmul_ogs
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    skipUnlessBackends(["cute"]),
]


def _routing(
    rows: int, groups: int, variant: int
) -> tuple[list[int], list[int], list[int]]:
    """Generate routing without assuming sorted offsets or unique row IDs."""
    counts = [3 * rows // 4, *([0] * (groups - 1))]
    for position in range(rows - counts[0]):
        counts[1 + position % (groups - 2)] += 1
    if variant == 1:
        counts = counts[-1:] + counts[:-1]
    elif variant == 2:
        counts = [0] * groups
    elif variant == 3:
        counts = [max(0, count - 7) for count in counts]
    order = list(range(groups))
    if variant % 2:
        order.reverse()
    starts = [0] * (groups + 1)
    indices = [0] * rows
    generator = torch.Generator().manual_seed(411 + variant)
    cursor = 0
    for group in order:
        starts[group] = cursor
        count = counts[group]
        values = (torch.randperm(count, generator=generator) + cursor).tolist()
        # Equal gathered rows in different M CTAs must write equal values.
        if count > 128:
            values[128] = values[0]
        if count > 260:
            values[260] = values[130]
        indices[cursor : cursor + count] = values
        cursor += count
    starts[-1] = cursor
    return counts, starts, indices


def _check_pipeline(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    bn: int,
    stages: int,
    *,
    padded: bool,
    static: bool = False,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family TMA gather required")
    rows, k, n, groups = shape
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(901)
    a_pitch = (k + 7) // 8 * 8 if padded else k
    b_pitch = (n + 7) // 8 * 8 if padded else n
    a = torch.randn((rows, a_pitch), device=device, dtype=dtype, generator=generator)[
        :, :k
    ]
    b = torch.randn(
        (groups, a_pitch, b_pitch), device=device, dtype=dtype, generator=generator
    )[:, :k, :n]
    routing = _routing(rows, groups, 0)
    metadata = tuple(
        torch.tensor(values, device=device, dtype=torch.int32) for values in routing
    )
    maximum = max(routing[0]) + 17
    args = (a, b, *metadata, maximum)
    kernel = helion.kernel(
        moe_matmul_ogs.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
    )
    bound = kernel.bind(args)
    config = helion.Config(
        block_sizes=[128, 32, 64],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute="tma_gather",
        cute_gathered_mma_n=bn,
        cute_gathered_mma_stages=stages,
    )
    assert "gathered_mma_tma" in bound.to_code(config)
    bound.set_config(config)

    def expected(left: torch.Tensor, right: torch.Tensor, limit: int) -> torch.Tensor:
        result = torch.zeros((rows, n), device=device, dtype=dtype)
        counts, starts, indices = routing
        for group, count in enumerate(counts):
            count = min(count, max(0, limit))
            if count == 0:
                continue
            chosen = torch.tensor(
                indices[starts[group] : starts[group] + count],
                device=device,
                dtype=torch.int64,
            )
            values = (left.index_select(0, chosen).float() @ right[group].float()).to(
                dtype
            )
            result.index_copy_(0, chosen, values)
        return result

    torch.testing.assert_close(
        bound(*args), expected(a, b, maximum), rtol=1e-2, atol=1e-1
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for variant in (1, 2, 3):
        routing = _routing(rows, groups, variant)
        for target, values in zip(metadata, routing, strict=True):
            target.copy_(target.new_tensor(values))
        a.normal_(generator=generator)
        b.normal_(generator=generator)
        graph.replay()
        torch.testing.assert_close(
            captured, expected(a, b, maximum), rtol=1e-2, atol=1e-1
        )

    # The cached wrapper must build descriptors for the replacement pointers.
    replacement_a = torch.empty_strided(a.shape, a.stride(), device=device, dtype=dtype)
    replacement_b = torch.empty_strided(b.shape, b.stride(), device=device, dtype=dtype)
    replacement_a.normal_(generator=generator)
    replacement_b.normal_(generator=generator)
    shorter = min(maximum, 129)
    replaced = bound(replacement_a, replacement_b, *metadata, shorter)
    torch.testing.assert_close(
        replaced, expected(replacement_a, replacement_b, shorter), rtol=1e-2, atol=1e-1
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(256, 128, 128, 4), (1024, 512, 512, 8), (4096, 1024, 1024, 16)]
)
def test_original_shapes_with_mutated_routing_and_replacement_pointers(
    shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    _check_pipeline(shape, dtype, 512, 2, padded=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("bn", "stages"),
    [(128, 2), (128, 3), (128, 4), (256, 2), (256, 3), (256, 4), (512, 2)],
)
def test_padded_input_and_unaligned_output_row_tails(
    dtype: torch.dtype, bn: int, stages: int
) -> None:
    _check_pipeline((385, 78, 70, 4), dtype, bn, stages, padded=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
def test_static_shapes_with_mutated_routing_and_replacement_pointers(
    dtype: torch.dtype, padded: bool
) -> None:
    shape = (385, 78, 70, 4) if padded else (256, 128, 128, 4)
    _check_pipeline(shape, dtype, 128, 3, padded=padded, static=True)
