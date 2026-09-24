from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _config
from .test_cute_grid_launch_extents import _grid_argreduce
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _bounds_code(rows: int, columns: int, largest: bool) -> str:
    # Match NVIDIA's FP32 minimum (1, 1, 8), not the generic CPU fallback
    # (16, 16, 16), which silently widens BM2 and hides the second-row CTA.
    with patch("helion._compat._min_dot_size", return_value=(1, 1, 8)):
        return _code(
            _grid_argreduce,
            (torch.empty((rows, 16)), torch.empty((16, columns)), largest),
            _config("collective"),
        )


@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rows,columns", [(3, 128), (3, 129), (17, 259)])
def test_argreduce_validity_uses_local_coordinates_and_global_masks(
    rows: int, columns: int, largest: bool
) -> None:
    code = _bounds_code(rows, columns, largest)
    assert "_BLOCK_SIZE_0 = 2" in code
    assert "indices_0 - tile_offset_0" in code
    assert "indices_1 - tile_offset_1" in code
    predicates = [
        node.value.test
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.IfExp)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id.startswith("argreduce_valid_smem")
            for target in node.targets
        )
    ]
    assert len(predicates) == 1
    predicate = compile(ast.Expression(body=predicates[0]), "<argreduce-valid>", "eval")
    for row, column, valid in (
        (0, 1, True),
        (2, 1, True),
        (rows - 1, columns - 1, True),
        (rows, columns - 1, False),
        (rows - 1, columns, False),
    ):
        namespace = {
            "cutlass": SimpleNamespace(Int32=int),
            "indices_0": row,
            "indices_1": column,
            "tile_offset_0": row // 2 * 2,
            "tile_offset_1": (min(column, columns - 1) // 128) * 128,
            "_BLOCK_SIZE_0": 2,
            "_BLOCK_SIZE_1": 128,
            "mask_0": row < rows,
            "mask_1": column < columns,
        }
        assert bool(eval(predicate, namespace)) == valid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "rows,columns,pattern",
    [
        (3, 128, "ties"),
        (3, 129, "finite"),
        (3, 259, "finite"),
        (17, 259, "ties"),
        (3, 129, "infinite"),
    ],
)
def test_argreduce_offset_ctas_and_tail_validity(
    rows: int, columns: int, pattern: str, largest: bool
) -> None:
    a = torch.ones((rows, 16), device=DEVICE)
    if pattern == "infinite":
        b = torch.full(
            (16, columns), -float("inf") if largest else float("inf"), device=DEVICE
        )
    else:
        fill = (-9.0 if largest else 9.0) if pattern == "finite" else 0.0
        b = torch.full((16, columns), fill, device=DEVICE)
        column = torch.arange(columns, device=DEVICE)
        if pattern == "finite":
            extreme = -1.0 if largest else 1.0
        else:
            extreme = 9.0 if largest else -9.0
        b[:, ((column % 128) == 1) | ((column % 128) == 32)] = extreme
    fn = torch.argmax if largest else torch.argmin
    expected = torch.stack(
        [fn(part, dim=1) for part in (a @ b).split(128, dim=1)], dim=1
    )
    args = (a, b, largest)
    saved = (a.clone(), b.clone())
    run = _grid_argreduce._bind_isolated(args).compile_config(_config("collective"))
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(run(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close((a, b), saved, atol=0, rtol=0)
