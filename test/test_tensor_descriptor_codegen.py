"""A reduction dim's block size and extent in generated Triton code, rendered on CPU."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.row_iota_kernels import half_row_fill
from helion._testing.row_iota_kernels import onehot_row_update
from helion._utils import triton_is_available

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton"]),
    pytest.mark.skipif(
        not triton_is_available(), reason="Triton is required for source generation"
    ),
]


@pytest.fixture
def sm90_descriptors() -> Iterator[None]:
    """Render for an sm90 target, which offers tensor descriptors, in a
    process that must not initialize CUDA."""
    with (
        cpu_render_target(capability=(9, 0), num_sm=132),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        yield


@skipIfRefEager("renders generated code")
@pytest.mark.parametrize(
    ("cols", "reduction_loop", "static_shapes", "block", "count"),
    [
        (200, 64, True, "_REDUCTION_BLOCK_1", 2),
        (256, 64, True, "_REDUCTION_BLOCK_1", 2),
        (256, None, True, "_RDIM_SIZE_1", 2),
        # With dynamic shapes the iota lowers to the size's name; out, whose
        # row stride is the dynamic size, keeps pointer indexing.
        (200, 64, False, "_REDUCTION_BLOCK_1", 1),
    ],
)
def test_descriptor_block_of_a_row_beside_its_iota(
    sm90_descriptors: None,
    cols: int,
    reduction_loop: int | None,
    static_shapes: bool,
    block: str,
    count: int,
) -> None:
    # The iota's arithmetic, carrying the row's block size symbol, lowered
    # to the row's extent, which used to stand for that symbol in the
    # descriptors' block shapes: [_BLOCK_SIZE_0, 200] fails Triton's
    # power-of-two check, [_BLOCK_SIZE_0, 256] a looped reduction's chunk of
    # 64, and [_BLOCK_SIZE_0, cols] (a runtime size) Triton's constexpr block
    # shape.  A row stride of cols + 8 elements keeps x's descriptor aligned.
    x = torch.rand(64, cols + 8, dtype=torch.bfloat16)[:, :cols]
    label = torch.randint(0, cols, (64,))
    kernel = helion.kernel(
        onehot_row_update,
        backend="triton",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((x, label))
    default = bound.config_spec.default_config()
    config = helion.Config(
        **(
            default.config
            | {
                "block_sizes": [16],
                "reduction_loops": [reduction_loop],
                "indexing": ["tensor_descriptor"] * len(default.config["indexing"]),
            }
        )
    )
    code = bound.to_triton_code(config)
    descriptors = [
        line.strip() for line in code.splitlines() if "make_tensor_descriptor" in line
    ]
    assert len(descriptors) == count, descriptors
    for line in descriptors:
        assert line.endswith(f"[_BLOCK_SIZE_0, {block}])"), line


@skipIfRefEager("renders generated code")
@pytest.mark.parametrize("reduction_loop", [None, 64])
@pytest.mark.parametrize("static_shapes", [True, False])
def test_nested_loop_reads_a_scalar_of_the_rows_extent(
    static_shapes: bool, reduction_loop: int | None
) -> None:
    # The interpreter's skip stays limited to values that are exactly the
    # block size symbol: n = row.size(1) // 2 still reaches the nested loop
    # as a capture of the extent's half, the constant 100 with static shapes
    # and, with dynamic ones, the outer floordiv its record names.
    x = torch.rand(64, 200)
    kernel = helion.kernel(
        half_row_fill,
        backend="triton",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((x,))
    config = helion.Config(
        **(
            bound.config_spec.default_config().config
            | {"block_sizes": [16], "reduction_loops": [reduction_loop]}
        )
    )
    code = bound.to_triton_code(config)
    assert not re.search(r"(_RDIM_SIZE_1|_REDUCTION_BLOCK_1)( // |, )2\b", code), code
    if static_shapes:
        assert "tl.range(0, tl.cast(100, tl.int32))" in code, code
    else:
        assert "tl.cast(floordiv, tl.float32)" in code, code
