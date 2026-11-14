from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_headwide import check_headwide_native
from helion._testing.cute_paged_headwide import fixed_root_matmul


def _raw_path(tmp_path):
    root = os.environ.get("HELION_HEADWIDE_NATIVE_RAW")
    if root is None:
        return tmp_path
    result = Path(root) / f"pid-{os.getpid()}" / tmp_path.name
    result.mkdir(parents=True, exist_ok=False)
    return result


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "query_len,scoped", [(4, False), (5, False), (4, True), (5, True)]
)
def test_complete_head_outputs_and_changed_graph(query_len, scoped, tmp_path):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("The paired/scoped attention helpers use SM100 tensor memory")
    check_headwide_native(query_len, scoped, DEVICE, _raw_path(tmp_path))


@skipIfNotCUDA()
@pytest.mark.parametrize("rows,block", [(16, 16), (40, 64)])
def test_complete_fixed_root_matmul(rows, block, tmp_path):
    tmp_path = _raw_path(tmp_path)
    # Row stride is noncontiguous; the MMA reduction axis remains contiguous.
    a = ((torch.arange(rows * 32, device=DEVICE).reshape(rows * 2, 16) % 7 - 3) / 4).to(
        torch.bfloat16
    )[::2]
    b = ((torch.arange(16 * 16, device=DEVICE).reshape(16, 16) % 5 - 2) / 4).to(
        torch.bfloat16
    )
    bound = fixed_root_matmul.bind((a, b, block))
    config = helion.Config(block_sizes=[], cute_mma_dag="warp", cute_mma_dag_warps=1)
    source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    (tmp_path / "generated.py").write_text(source)
    compiled = bound.compile_config(config)
    for index in range(2):
        before = (a.clone(), b.clone())
        actual = compiled(a, b, block)
        expected = a.float() @ b.float()
        torch.cuda.synchronize()
        record = {
            "actual": actual.cpu(),
            "expected": expected.cpu(),
            "input_before": [x.cpu() for x in before],
            "input_after": [a.cpu(), b.cpu()],
        }
        torch.save(record, tmp_path / f"matmul-{index}.pt")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(a, before[0]) and torch.equal(b, before[1])
        a.neg_()
