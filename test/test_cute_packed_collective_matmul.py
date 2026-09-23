from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING

from examples.int4_gemm import _unpack_int4_matrix
from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_matmul import _tensor_roots
from helion._compiler.cute.packed_matmul import packed_matmul_axis
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
import helion.language as hl
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _expanded_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    factor: hl.constexpr,
    use_addmm: hl.constexpr,
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    packed_size = hl.register_block_size(k // factor)  # pyrefly: ignore[unsupported-operation]
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for packed in hl.tile(k // factor, block_size=packed_size):  # pyrefly: ignore[unsupported-operation]
            start = packed.begin * factor
            left = a[row, start : start + packed_size * factor]
            base = b[packed, col].to(a.dtype)
            if factor == 2:
                terms = torch.stack([base, base * 0.5], dim=1)
            elif factor == 3:
                terms = torch.stack([base, base * 0.5, base], dim=1)
            else:
                terms = torch.stack([base, base * 0.5, base, base * 0.5], dim=1)
            right = terms.reshape(packed.block_size * factor, col.block_size)  # pyrefly: ignore[bad-argument-type]
            if use_addmm:
                acc = torch.addmm(acc, left, right)
            else:
                acc = hl.dot(left, right, acc=acc)
        out[row, col] = acc.to(out.dtype)
    return out


def _config(
    *, factor: int = 2, compute: str = "tcgen05", n_first: bool = False
) -> helion.Config:
    return helion.Config(
        block_sizes=[32 // factor, 64, 32],
        num_threads=[1, 4, 32],
        cute_vector_widths=[1, 1, 1],
        cute_lane_layouts=["blocked", "blocked", "blocked"],
        loop_orders=[[1, 0] if n_first else [0, 1]],
        cute_collective_mma=True,
        cute_collective_compute=compute,
    )


def _original_bound(
    shape: tuple[int, int, int] = (73, 78, 41), dtype: torch.dtype = torch.bfloat16
) -> BoundKernel:
    m, k, n = shape
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=matmul_bf16_int4.settings.static_shapes,
        autotune_effort="none",
    )
    return _cpu_bind(
        kernel,
        (torch.empty((m, k), dtype=dtype), torch.empty((k // 2, n), dtype=torch.int8)),
    )


@pytest.mark.parametrize(
    "shape", [(128, 256, 128), (512, 1024, 512), (1024, 2048, 1024)]
)
@skipUnlessBackends(["cute"])
def test_original_int4_has_searchable_packed_collective_seeds(
    shape: tuple[int, int, int],
) -> None:
    bound = _original_bound(shape)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        generation = ConfigGeneration(bound.config_spec)
        for seed in seeds:
            normalized = helion.Config.from_dict(dict(seed))
            bound.config_spec.normalize(normalized)
            restored = generation.unflatten(generation.flatten(normalized))
            assert restored["cute_collective_mma"]
            assert (
                restored["cute_collective_compute"]
                == normalized["cute_collective_compute"]
            )
    assert seeds
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    # Direct-load native facts must retain their logical, expanded K semantics.
    assert bound.config_spec.matmul_facts[0].k_block_id is None
    for compute in ("warp", "tcgen05"):
        seed = next(
            value
            for value in seeds
            if value.get("cute_collective_compute", "warp") == compute
        )
        source = bound.to_code(seed)
        assert "cute.gemm(" in source
        assert ("tcgen05.CtaGroup.ONE" in source) == (compute == "tcgen05")
        assert "dot_acc" not in source
        assert "_helion_pending_collective_mma" not in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("use_addmm", [False, True])
@skipUnlessBackends(["cute"])
def test_packed_collective_is_not_specific_to_int4(
    dtype: torch.dtype, factor: int, use_addmm: bool
) -> None:
    inputs = (
        torch.empty((73, 39 * factor), dtype=dtype),
        torch.empty((39, 41), dtype=torch.int8),
        torch.empty((73, 41), dtype=dtype),
        factor,
        use_addmm,
    )
    source = _cpu_bind(_expanded_matmul, inputs).to_code(_config(factor=factor))
    assert "tcgen05.CtaGroup.ONE" in source
    assert "cute.gemm(" in source
    for lane in range(factor):
        assert f"collective_k * {factor} + {lane}]" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_first", [False, True])
@skipUnlessBackends(["cute"])
def test_packed_collective_preserves_input_alias_rejection(
    dtype: torch.dtype, n_first: bool
) -> None:
    a = torch.empty((73, 78), dtype=dtype)
    inputs = (a, torch.empty((39, 41), dtype=torch.int8), a[:, :41], 2, False)
    bound = _cpu_bind(_expanded_matmul, inputs)
    with pytest.raises(BackendUnsupported, match="operand loads may alias"):
        bound.to_code(_config(n_first=n_first))


def test_tensor_subscript_reads_participate_in_alias_proof() -> None:
    body = ast.parse("value = a[row, col] + (b.iterator + index).load()").body
    assert _tensor_roots(body, "load", tensor_names=frozenset({"a", "b"})) == {"a", "b"}


@pytest.mark.parametrize(
    "mutation", ["interleave", "extent", "term_dtype", "affine_start"]
)
@skipUnlessBackends(["cute"])
def test_packed_axis_rejects_unproved_interleaving(mutation: str) -> None:
    bound = _original_bound()
    assert bound.host_function is not None
    node = next(
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is dot
    )
    lhs, rhs = node.args[:2]
    assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
    stack = rhs.args[0]
    assert isinstance(stack, torch.fx.Node)
    if mutation == "interleave":
        stack.args = (stack.args[0], 0)
    elif mutation == "extent":
        rhs.meta["val"] = torch.empty((64, 64), dtype=torch.bfloat16)
    elif mutation == "term_dtype":
        terms = stack.args[0]
        assert isinstance(terms, (tuple, list))
        term = terms[0]
        assert isinstance(term, torch.fx.Node)
        term.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
    else:
        indices = lhs.args[1]
        assert isinstance(indices, (tuple, list))
        index = indices[1]
        assert isinstance(index, torch.fx.Node)
        index.kwargs = dict(index.kwargs) | {"start": 0}
    assert packed_matmul_axis(bound.env, lhs, rhs) is None


@skipUnlessBackends(["cute"])
def test_nonintegral_mma_expansion_has_no_collective_seed() -> None:
    inputs = (
        torch.empty((73, 117), dtype=torch.bfloat16),
        torch.empty((39, 41), dtype=torch.int8),
        torch.empty((73, 41), dtype=torch.bfloat16),
        3,
        False,
    )
    bound = _cpu_bind(_expanded_matmul, inputs)
    assert bound.host_function is not None
    with bound.env:
        assert not CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


@dataclass
class _Pointer:
    tensor: torch.Tensor
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + offset)

    def load(self) -> int | float:
        flat = self.tensor.flatten()
        assert 0 <= self.offset < flat.numel(), "unmasked pointer read"
        return flat[self.offset].item()


class _ScalarTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.iterator = _Pointer(tensor)
        self.layout = SimpleNamespace(stride=tensor.stride())

    def __getitem__(self, index: tuple[int, int]) -> int | float:
        assert all(
            0 <= coordinate < extent
            for coordinate, extent in zip(index, self.tensor.shape, strict=True)
        ), "unmasked tensor read"
        return self.tensor[index].item()

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        self.tensor[index] = value


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_actual_staging_replays_all_nibbles_and_mnk_tail_masks(
    dtype: torch.dtype, compute: str
) -> None:
    source = _original_bound(dtype=dtype).to_code(_config(compute=compute))
    tree = ast.parse(source)
    a_copy, b_copy = (
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
        for name in ("collective_ai", "collective_bi")
    )
    staging = compile(
        ast.fix_missing_locations(ast.Module(body=[a_copy, b_copy], type_ignores=[])),
        "<actual-packed-staging>",
        "exec",
    )
    a = (torch.arange(73 * 78).reshape(73, 78).float() % 33 - 16).to(dtype)
    b = (torch.arange(39 * 41).reshape(39, 41) % 256 - 128).to(torch.int8)
    unpacked = _unpack_int4_matrix(b).to(dtype)
    for packed_offset in (0, 16, 32):
        staged_a = torch.full((64, 32), float("nan"), dtype=dtype)
        staged_b = torch.full((32, 32), float("nan"), dtype=dtype)
        for tid in range(128):
            namespace = {
                "cutlass": SimpleNamespace(
                    Int8=int,
                    Int32=int,
                    Int64=int,
                    Float32=float,
                    Float16=lambda value: float(
                        torch.tensor(value, dtype=torch.float16)
                    ),
                    BFloat16=lambda value: float(
                        torch.tensor(value, dtype=torch.bfloat16)
                    ),
                    range=lambda *args, **kwargs: range(*args),
                ),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(
                        thread_idx=lambda coordinates=(tid % 4, tid // 4, 0): (
                            coordinates
                        )
                    )
                ),
                "operator": operator,
                "A": _ScalarTensor(a),
                "B": _ScalarTensor(b),
                "M": 73,
                "K": 78,
                "N": 41,
                "tile_offset_0": packed_offset,
                "tile_offset_1": 64,
                "tile_offset_2": 32,
                "collective_tid": tid,
                "collective_a": _ScalarTensor(staged_a),
                "collective_b": _ScalarTensor(staged_b),
            }
            exec(staging, namespace)
        expected_a = torch.zeros_like(staged_a)
        expected_b = torch.zeros_like(staged_b)
        logical_start = 2 * packed_offset
        valid_k = min(32, 78 - logical_start)
        expected_a[:9, :valid_k] = a[64:, logical_start : logical_start + valid_k]
        expected_b[:9, :valid_k] = unpacked[
            logical_start : logical_start + valid_k, 32:
        ].T
        torch.testing.assert_close(staged_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(staged_b, expected_b, rtol=0, atol=0)
