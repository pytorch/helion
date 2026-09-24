from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.indexing_strategy import SubscriptIndexing
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize(
    "index,expected",
    [
        ([slice(None), slice(None)], [3, 5]),
        ([None, slice(None), slice(None)], [1, 3, 5]),
        ([slice(None), None, slice(None)], [3, 1, 5]),
        ([slice(None), slice(None), None], [3, 5, 1]),
        ([None, slice(None), None, slice(None), None], [1, 3, 1, 5, 1]),
    ],
)
def test_shape_only_views_do_not_allocate_reduction_dimensions(
    index: list[object], expected: list[int]
) -> None:
    tensor = torch.empty((3, 5))
    with patch.object(
        SubscriptIndexing,
        "compute_shape",
        side_effect=AssertionError("shape-only views must preserve dimensions"),
    ):
        assert CuteBackend().fake_subscript_shape(tensor, index) == expected


@pytest.mark.parametrize(
    "index",
    [
        [1, slice(None)],
        [slice(1, 3), slice(None)],
        [slice(None), slice(0, 4, 1)],
    ],
)
def test_narrowing_keeps_general_index_shape_path(index: list[object]) -> None:
    tensor = torch.empty((3, 5))
    with patch.object(
        SubscriptIndexing, "compute_shape", return_value=[2, 5]
    ) as compute:
        assert CuteBackend().fake_subscript_shape(tensor, index) == [2, 5]
    compute.assert_called_once_with(tensor, index)


@pytest.mark.parametrize(
    "index", [[slice(None), slice(None, None, 2)], [-1, slice(None)]]
)
def test_unsupported_narrowing_still_fails_closed(index: list[object]) -> None:
    with pytest.raises(exc.InvalidIndexingType):
        CuteBackend().fake_subscript_shape(torch.empty((3, 5)), index)


@pytest.mark.parametrize("topology", ["fa4", "ws_overlap"])
def test_causal_flash_host_has_no_undefined_block_size(topology: str) -> None:
    from .test_cute_backend import cute_causal_biased_attention

    values = [torch.empty((1, 2, 256, 64), dtype=torch.float16) for _ in range(3)]
    values.append(torch.empty((1, 2, 256, 256), dtype=torch.float16))
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        cute_causal_biased_attention.reset()
        code = cute_causal_biased_attention.bind(tuple(values)).to_code(
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_topology=topology,
                cute_flash_causal_kv_order="descending",
                cute_flash_causal_loop_split=True,
            )
        )
    assert "'kind': 'helion_flash'" in code
    tree = ast.parse(code)
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    read = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id.startswith("_BLOCK_SIZE_")
    }
    assert not read - assigned
