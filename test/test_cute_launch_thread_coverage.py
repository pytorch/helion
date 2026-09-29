"""A SIMT launch must retain the threads required by root and inner tiles."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch

from test.test_cute_collective_tf32 import _config as _collective_config
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel as _collective_kernel
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CPU_DEVICE = "cpu"

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import Kernel


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with _cpu_target():
        yield


def _selected_config() -> helion.Config:
    # Full-search winner whose root N axis was omitted from the launch guard.
    return helion.Config(
        block_sizes=[1, 1024, 32, 64],
        num_threads=[1, 1024, 0, 0],
        loop_orders=[[0, 1]],
        load_eviction_policies=["l2_last", "streaming", "", "l2_last", "l2_last"],
        cute_vector_widths=[2, 4, 1, 2],
        cute_lane_layouts=["blocked", "strided", "strided", "strided"],
        cute_collective_mma=False,
        cute_collective_compute="warp",
        cute_collective_native_seeded=True,
        cute_collective_stages=4,
        cute_collective_copy="scalar",
        cute_gathered_mma_n=256,
        cute_gathered_mma_stages=3,
    )


def _jagged_kernel(*, static_shapes: bool) -> Kernel[torch.Tensor]:
    return helion.kernel(
        jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        dot_precision="tf32",
    )


def _launch_shape(source: str) -> tuple[int, int, int]:
    shapes = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "block"
    ]
    assert len(shapes) == 1
    return shapes[0]


def test_selected_jagged_config_cannot_drop_1024_row_threads() -> None:
    inputs = _jagged_inputs(batch=256, rows=30856, k=128, n=128)
    bound = _jagged_kernel(static_shapes=True)._bind_isolated(inputs)
    config = _selected_config()
    before = config.to_json()
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match=r"under-dimension required_thread_block_dims=\(32, 1024, 1\)",
    ):
        bound.to_code(config)
    assert config.to_json() == before


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize(
    ("schedule", "static_launch", "dynamic_launch"),
    [
        ("reference", (16, 1, 1), (1, 1, 1)),
        ("serial_n", (1024, 1, 1), (1024, 1, 1)),
        ("auto_n", (32, 32, 1), (32, 1, 1)),
        ("lane_rows", (32, 4, 1), (4, 32, 1)),
    ],
)
def test_jagged_simt_schedules_keep_row_coverage(
    static_shapes: bool,
    schedule: str,
    static_launch: tuple[int, int, int],
    dynamic_launch: tuple[int, int, int],
) -> None:
    inputs = _jagged_inputs(batch=256, rows=30856, k=128, n=128)
    bound = _jagged_kernel(static_shapes=static_shapes)._bind_isolated(inputs)
    if schedule == "reference":
        config = bound.config_spec.autotune_reference_config()
    else:
        settings = _selected_config().config
        settings["num_threads"] = {
            "serial_n": [1, 1024, 1, 1],
            "auto_n": [1, 32, 0, 0],
            "lane_rows": [1, 4, 32, 1],
        }[schedule]
        if schedule == "lane_rows":
            settings["block_sizes"] = [1, 64, 32, 64]
        if not static_shapes:
            # A dynamic N extent stays inside the row loop, so this binding
            # has no root-axis permutation to configure.
            settings.pop("loop_orders")
        config = helion.Config.from_dict(settings)
    source = bound.to_code(config)
    assert _launch_shape(source) == (static_launch if static_shapes else dynamic_launch)
    assert "cute.gemm(" not in source
    if schedule == "serial_n":
        assert f"for lane_2 in range({32 if static_shapes else 8})" in source
        assert (
            "indices_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0])"
            in source
        )
    else:
        assert "for lane_1" in source


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_collective_matmul_keeps_its_cooperative_launch(
    static_shapes: bool, compute: str
) -> None:
    inputs = (torch.empty((3, 130, 78)), torch.empty((3, 78, 70)))
    bound = _collective_kernel(
        batched=True, static_shapes=static_shapes
    )._bind_isolated(inputs)
    source = bound.to_code(_collective_config(compute=compute))
    assert _launch_shape(source) == (4, 32, 1)
    assert "cute.gemm(" in source
    if compute == "tcgen05":
        assert "tcgen05.CtaGroup.ONE" in source
    else:
        assert "warp.MmaTF32Op((16, 8, 8))" in source


def _nested_copy(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    output = torch.empty_like(x)
    for column in hl.tile(columns):
        for row in hl.tile(rows):
            output[row, column] = x[row, column]
    return output


def test_plain_nested_simt_loop_cannot_drop_an_explicit_inner_axis() -> None:
    inputs = (torch.empty((1536, 128), device=CPU_DEVICE),)
    kernel = helion.kernel(
        _nested_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated(inputs)
    with pytest.raises(helion.exc.BackendUnsupported):
        bound.to_code(helion.Config(block_sizes=[32, 1024], num_threads=[0, 1024]))
