from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from test._cute_binding import _cpu_bind
from test.test_cute_collective_b_major import _b_copy
from test.test_cute_collective_dot import _full_slice_dot
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.memory_ops import _cute_resolved_load_mask
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


@dataclass
class _MaskStrategy:
    index: str
    mask: str | None

    def index_var(self, block_id: int) -> str:
        return self.index

    def mask_var(self, block_id: int) -> str | None:
        return self.mask


@pytest.mark.parametrize("k_mask", [None, "mask_k"])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("alias", ["none", "same_mask", "different_mask"])
def test_full_slice_preserves_resolved_axis_mask(
    k_mask: str | None, extra: bool, alias: str
) -> None:
    shape_env = ShapeEnv()
    sizes = [shape_env.create_unbacked_symint() for _ in range(3)]
    env = SimpleNamespace(
        get_block_id=lambda size: next(
            (index for index, candidate in enumerate(sizes) if size is candidate),
            None,
        ),
        known_equal=operator.eq,
        block_sizes=[
            SimpleNamespace(block_id=index, size=64, var=size)
            for index, size in enumerate(sizes)
        ],
    )
    codegen = SimpleNamespace(
        active_device_loops={
            index: [
                SimpleNamespace(
                    strategy=_MaskStrategy(
                        f"index_{2 if index == 0 and alias != 'none' else index}",
                        k_mask if index == 0 and alias == "same_mask" else mask,
                    ),
                )
            ]
            for index, mask in enumerate(("mask_m", "mask_n", k_mask))
        },
        lift=lambda value, **kwargs: value,
    )
    state = cast(
        "CodegenState",
        SimpleNamespace(
            codegen=codegen,
            device_function=SimpleNamespace(
                cute_state=SimpleNamespace(
                    matmul_operand_block_remap={}, matmul_operand_index_override={}
                )
            ),
        ),
    )
    predicate = ast.Name(id="predicate", ctx=ast.Load()) if extra else None
    with patch.object(CompileEnvironment, "current", return_value=env):
        if alias == "different_mask":
            with pytest.raises(
                helion.exc.BackendUnsupported, match="ambiguous coordinate bounds"
            ):
                _cute_resolved_load_mask(
                    state, torch.empty(64), [None, slice(None)], ["index_2"], predicate
                )
            return
        actual = _cute_resolved_load_mask(
            state, torch.empty(64), [None, slice(None)], ["index_2"], predicate
        )
    # Neither an unrelated M/N tail nor an explicit false user mask may be
    # lost by matching equal extents. The K mask may legitimately be elided.
    for m_valid in (False, True):
        for k_valid in (False, True):
            for user_valid in (False, True):
                values = {
                    "mask_m": m_valid,
                    "mask_n": not m_valid,
                    "mask_k": k_valid,
                    "predicate": user_valid,
                }
                assert eval(actual or "True", {}, values) == (
                    (k_mask is None or k_valid) and (not extra or user_valid)
                )


def _config(*, compute: str, copy: str) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_threads=[2, 64, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy=copy,
    )


def _assert_operand_coordinates_are_local(source: str) -> None:
    loops = {
        node.target.id: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
    }
    for loop, other_axis in (
        ("collective_ai", "indices_1"),
        ("collective_bi", "indices_0"),
    ):
        assert all(
            not isinstance(node, ast.Name) or node.id != other_axis
            for node in ast.walk(loops[loop])
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 70, 64), (130, 21, 21)])
@pytest.mark.parametrize(
    "compute,copy", [("warp", "scalar"), ("tcgen05", "async_cached")]
)
@skipUnlessBackends(["cute"])
def test_full_slice_equal_extents_do_not_capture_another_output_axis(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    compute: str,
    copy: str,
) -> None:
    m, n, k = shape
    inputs = (torch.empty((m, k), dtype=dtype), torch.empty((k, n), dtype=dtype))
    kernel = helion.kernel(
        _full_slice_dot.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    source = _cpu_bind(kernel, inputs).to_code(_config(compute=compute, copy=copy))
    assert source.count("cute.gemm(") == 1
    _assert_operand_coordinates_are_local(source)
    # N=70/21 still needs a real RHS/output bound after its unrelated M mask
    # is removed; a non-divisible synthetic K keeps its zero-fill guard.
    assert "collective_n" in ast.unparse(_b_copy(source))
    assert "else cutlass." in ast.unparse(_b_copy(source))
    if k == 21:
        assert "collective_k_offset + collective_k < 21" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_original_seeded_mamba_oversized_m_tile_has_no_escaped_row_mask(
    dtype: torch.dtype, static_shapes: bool
) -> None:
    inputs = tuple(torch.empty(size, dtype=dtype) for size in _mamba_sizes(1))
    config = helion.Config.from_dict(
        _config(compute="tcgen05", copy="async_cached").config
        | {
            "block_sizes": [128, 64, 128],
            "num_threads": [2, 64, 1, 1],
            "cute_vector_widths": [1] * 7,
            "cute_lane_layouts": ["blocked"] * 7,
            "cute_collective_native_seeded": True,
        }
    )
    source = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), inputs).to_code(
        config
    )
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert "fence_view_async_tmem_store" in source
    _assert_operand_coordinates_are_local(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 70, 64), (130, 21, 21)])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_full_slice_equal_extent_tails_and_mutated_graphs(
    dtype: torch.dtype, shape: tuple[int, int, int], compute: str
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2011)
    m, n, k = shape
    a = torch.randn((m, k), device=CUDA_DEVICE, dtype=dtype)
    b = torch.randn((k, n), device=CUDA_DEVICE, dtype=dtype)
    bound = _full_slice_dot._bind_isolated((a, b))
    bound.set_config(_config(compute=compute, copy="async_cached"))

    def expected() -> torch.Tensor:
        return torch.relu((a * 0.5).float() @ b.float()).to(dtype)

    def run() -> torch.Tensor:
        return bound(a, b)

    torch.testing.assert_close(run(), expected(), rtol=0.02, atol=0.02)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(replay(), expected(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        bound(a.clone(), b.clone()), expected(), rtol=0.02, atol=0.02
    )
