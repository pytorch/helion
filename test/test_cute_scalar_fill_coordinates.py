"""CPU coverage of scalar SMEM fill participants in native tcgen05 launches."""

from __future__ import annotations

import ast
from collections import Counter
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import vector_scaled_mm

import helion
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _participant_ids(source: str) -> tuple[list[int], int]:
    tree = ast.parse(source)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    shape = ast.literal_eval(
        next(kw.value for kw in launch.keywords if kw.arg == "block")
    )
    names = {
        "tcgen05_warp_idx",
        "tcgen05_lane_idx",
        "mma_tidx",
        "mma_copy_tidx",
        "mma_active",
    }
    assignments = [
        node
        for node in device.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(assignments) == len(names)
    strides = {
        ast.literal_eval(node.value.right.right.args[0])
        for node in ast.walk(device)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_flat"
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.left, ast.Name)
        and node.value.left.id == "mma_copy_tidx"
    }
    assert len(strides) == 1
    (stride,) = strides
    ids = []
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                linear = x + shape[0] * (y + shape[1] * z)
                arch = SimpleNamespace(
                    thread_idx=lambda coords=(x, y, z): coords,
                    warp_idx=lambda tid=linear: tid // 32,
                    lane_idx=lambda tid=linear: tid % 32,
                    make_warp_uniform=lambda value: value,
                )
                scope = {
                    "cutlass": SimpleNamespace(Int32=int),
                    "cute": SimpleNamespace(arch=arch),
                }
                for statement in assignments:
                    expression = ast.Expression(statement.value)
                    scope[statement.targets[0].id] = eval(
                        compile(expression, "<generated participant>", "eval"),
                        {"__builtins__": {}},
                        scope,
                    )
                if scope["mma_active"]:
                    ids.append(scope["mma_copy_tidx"])
    return ids, stride


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("order", ([0, 1], [1, 0]))
@pytest.mark.parametrize("shape", ((128, 256, 288), (192, 256, 256), (128, 288, 256)))
def test_scalar_smem_fill_covers_each_element_once(dtype, order, shape):
    args = scaled_args(*shape, dtype=dtype)
    config = helion.Config(
        block_sizes=[128, 64, 128],
        loop_orders=[order],
        l2_groupings=[8],
        indexing=["pointer"] * 5,
        pid_type="flat",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=1,
        tcgen05_c_stages=2,
        tcgen05_c_store_mode="direct",
        tcgen05_persistence_model="non_persistent",
    )
    # The scalar SMEM fill is the producer the flat edge TMA path replaces;
    # pin the legacy path to test its coordinates.
    kernel = helion.kernel(
        vector_scaled_mm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_flat_edge_tma=False,
    )
    source = kernel._bind_isolated(args).to_code(config)
    ids, stride = _participant_ids(source)
    assert sorted(ids) == list(range(stride))
    for size in (128 * 128, 64 * 128):
        writes = Counter(
            lane + iteration * stride
            for lane in ids
            for iteration in range((size + stride - 1) // stride)
            if lane + iteration * stride < size
        )
        assert writes == Counter(range(size))
