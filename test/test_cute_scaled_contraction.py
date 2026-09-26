from __future__ import annotations

import ast
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.nvfp4_gemm import _nvfp4_w4a4_matmul_kernel
import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_packed_collective_matmul import _ScalarTensor

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_matmul import _tensor_roots
from helion._compiler.cute.packed_matmul import interleaved_matrix_terms
from helion._compiler.cute.scaled_contraction import expose_scaled_contractions
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.language import creation_ops
from helion.language import memory_ops
from helion.language.matmul_ops import dot
from helion.language.quantized_ops import float4_e2m1fn_x2_to_float32

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR
    from helion.runtime.kernel import BoundKernel


def _inputs(m: int = 73, groups: int = 5, n: int = 41) -> tuple[object, ...]:
    return (
        torch.empty((m, groups, 8), dtype=torch.float4_e2m1fn_x2),
        torch.empty((groups, 8, n), dtype=torch.float4_e2m1fn_x2),
        torch.empty(
            ((m + 127) // 128 * ((groups + 3) // 4) * 512,),
            dtype=torch.float8_e4m3fn,
        ),
        torch.empty(
            ((n + 127) // 128 * ((groups + 3) // 4) * 512,),
            dtype=torch.float8_e4m3fn,
        ),
        torch.empty((m, n), dtype=torch.bfloat16),
        1.0,
    )


def _bind(
    shape: tuple[int, int, int] = (73, 5, 41),
) -> tuple[BoundKernel, tuple[object, ...]]:
    inputs = _inputs(*shape)
    kernel = helion.kernel(
        _nvfp4_w4a4_matmul_kernel.fn,
        backend="cute",
        static_shapes=_nvfp4_w4a4_matmul_kernel.settings.static_shapes,
        autotune_effort="none",
    )
    with _cpu_target():
        return kernel._bind_isolated(inputs), inputs


def _config(compute: str = "tcgen05", *, bk: int = 2) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 32, bk],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_lane_layouts=["blocked"] * 3,
        loop_orders=[[0, 1]],
        cute_collective_mma=compute != "scalar",
        cute_collective_compute="warp" if compute == "scalar" else compute,
    )


@pytest.mark.parametrize("shape", [(128, 16, 128), (512, 64, 512), (1024, 128, 1024)])
@skipUnlessBackends(["cute"])
def test_original_nvfp4_has_ordinary_mma_seeds(shape: tuple[int, int, int]) -> None:
    bound, inputs = _bind(shape)
    assert bound.host_function is not None
    ir = bound.host_function.device_ir
    facts = bound.config_spec.matmul_facts
    assert len(facts) == 1
    assert facts[0].lhs_dtype is facts[0].rhs_dtype is torch.bfloat16
    assert facts[0].static_k == shape[1] * 16
    assert facts[0].k_block_id is None
    assert ir.codegen_active_block_ids == frozenset({0, 1, 2})
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(bound.env, ir)
        assert seeds
        assert any(seed.get("cute_collective_compute") == "tcgen05" for seed in seeds)
        assert len({seed.block_sizes[-1] for seed in seeds}) > 1
        assert all(seed.get("cute_collective_copy") == "scalar" for seed in seeds)
        generation = ConfigGeneration(bound.config_spec)
        for seed in seeds:
            bound.config_spec.normalize(seed)
            restored = generation.unflatten(generation.flatten(seed))
            assert all(restored[key] == value for key, value in seed.items())
    with _cpu_target():
        source = bound.to_code(_config())
    assert "cute.gemm(" in source
    assert "tcgen05.CtaGroup.ONE" in source
    assert "synthetic_lane_3" not in source and "synthetic_lane_4" not in source
    assert "_helion_pending_collective_mma" not in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_pairs(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, groups, pairs = a.shape
    n = b.size(2)
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for group in hl.tile(groups):
            left_scale = sa[row, group].to(torch.float32)
            right_scale = sb[group, col].to(torch.float32)
            for byte in hl.static_range(pairs):
                a0, a1 = hl.float4_e2m1fn_x2_to_float32(a[row, group, byte])
                b0, b1 = hl.float4_e2m1fn_x2_to_float32(b[group, byte, col])
                acc = acc + (
                    (
                        a0.unsqueeze(2) * b0.unsqueeze(0)
                        + a1.unsqueeze(2) * b1.unsqueeze(0)
                    )
                    * left_scale.unsqueeze(2)
                    * right_scale.unsqueeze(0)
                ).sum(1)
        out[row, col] = acc
    return out


@pytest.mark.parametrize("pairs", [1, 2, 3])
@skipUnlessBackends(["cute"])
def test_pair_count_and_scale_layout_are_not_fixed(pairs: int) -> None:
    args = (
        torch.empty((73, 7, pairs), dtype=torch.float4_e2m1fn_x2),
        torch.empty((7, pairs, 41), dtype=torch.float4_e2m1fn_x2),
        torch.empty((73, 7), dtype=torch.float8_e4m3fn),
        torch.empty((7, 41), dtype=torch.float8_e4m3fn),
        torch.empty((73, 41), dtype=torch.float32),
    )
    with _cpu_target():
        bound = _scaled_pairs._bind_isolated(args)
        source = bound.to_code(_config("scalar"))
    assert bound.host_function is not None
    nodes = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is dot
    ]
    assert len(nodes) == 1
    lhs = nodes[0].args[0]
    assert isinstance(lhs, torch.fx.Node)
    assert len(interleaved_matrix_terms(lhs, lhs=True)) == pairs * 2
    assert "reshape_smem" not in source
    assert "dot_product" in source


class _StopProbe(Exception):
    pass


@pytest.mark.parametrize(
    "change",
    [
        "scale_fp32",
        "scale_e5m2",
        "computed_scale",
        "arbitrary_fp32",
        "pair_alpha",
        "half_product",
        "nonzero_seed",
        "negative_zero_seed",
        "half_seed",
        "multiple_reduction_uses",
        "effect",
        "extra_output",
    ],
)
@skipUnlessBackends(["cute"])
def test_refuses_unproved_precision_or_effects(change: str) -> None:
    checked = False

    def probe(ir: DeviceIR) -> int:
        nonlocal checked
        info = next(
            info
            for info in ir.graphs
            if any(n.target is torch.ops.aten.sum.dim_IntList for n in info.graph.nodes)
        )
        graph = info.graph
        nodes = list(graph.nodes)
        reduction = next(n for n in nodes if n.target is torch.ops.aten.sum.dim_IntList)
        scale = next(
            n
            for n in nodes
            if n.target is torch.ops.prims.convert_element_type.default
            and n.args[0].meta["val"].dtype is torch.float8_e4m3fn
        )
        seed = next(
            n
            for info in ir.graphs
            for n in info.graph.nodes
            if n.target is creation_ops.full
        )
        if change in {"scale_fp32", "scale_e5m2"}:
            value = scale.args[0].meta["val"]
            scale.args[0].meta["val"] = value.new_empty(
                value.shape,
                dtype=torch.float32 if change == "scale_fp32" else torch.float8_e5m2,
            )
        elif change == "computed_scale":
            scale.target = torch.ops.aten.add.Tensor
            scale.args = (scale.args[0], 0.0)
        elif change == "arbitrary_fp32":
            pair = next(n for n in nodes if n.target is float4_e2m1fn_x2_to_float32)
            pair.target = tuple
        elif change in {"pair_alpha", "half_product"}:
            pair = next(
                n
                for n in nodes
                if n.target is torch.ops.aten.add.Tensor and n.meta["val"].ndim == 3
            )
            if change == "pair_alpha":
                pair.kwargs = {"alpha": 2}
            else:
                operand = pair.args[0]
                with graph.inserting_before(pair):
                    rounded = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        (operand, torch.float16),
                    )
                rounded.meta = {**operand.meta, "val": operand.meta["val"].half()}
                pair.args = (rounded, pair.args[1])
        elif change in {"nonzero_seed", "negative_zero_seed"}:
            seed.args = (
                seed.args[0],
                1.0 if change == "nonzero_seed" else -0.0,
                *seed.args[2:],
            )
        elif change == "half_seed":
            seed.meta["val"] = seed.meta["val"].half()
        elif change == "multiple_reduction_uses":
            with graph.inserting_after(reduction):
                extra = graph.call_function(torch.ops.aten.clone.default, (reduction,))
            extra.meta = dict(reduction.meta)
        elif change == "effect":
            load = next(
                n
                for n in nodes
                if n.target is memory_ops.load
                and n.args[0].meta["val"].dtype is torch.float4_e2m1fn_x2
            )
            with graph.inserting_before(reduction):
                effect = graph.call_function(
                    memory_ops.store, (load.args[0], load.args[1], load, None)
                )
            effect.meta = {**load.meta, "val": None}
        elif change == "extra_output":
            output = next(iter(graph.find_nodes(op="output")))
            output.args = ([output.args[0][0], reduction],)
        before = str(graph)
        assert expose_scaled_contractions(ir) == 0
        assert str(graph) == before
        checked = True
        raise _StopProbe

    with (
        patch(
            "helion._compiler.cute.scaled_contraction.expose_scaled_contractions",
            side_effect=probe,
        ),
        pytest.raises(_StopProbe),
    ):
        _bind()
    assert checked


_FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def test_all_fp4_and_e4m3_encodings_fit_bf16_exactly() -> None:
    scales = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    exact = _FP4[:, None] * scales[None, :]
    torch.testing.assert_close(
        exact.bfloat16().float(), exact, rtol=0, atol=0, equal_nan=True
    )
    pair = torch.tensor([1.0, -1.0]).sum() * -1.0
    distributed = torch.tensor([-1.0, 1.0]).sum()
    assert torch.signbit(pair) and not torch.signbit(distributed)
    assert not torch.signbit(torch.tensor(0.0) + pair)


def _scale_offset(row: int, group: int, groups: int) -> int:
    return (
        ((row // 128) * ((groups + 3) // 4) + group // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + group % 4
    )


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("bk", [1, 2, 4])
@skipUnlessBackends(["cute"])
def test_actual_packed_staging_preserves_scales_and_nan_padding(
    compute: str, bk: int
) -> None:
    bound, inputs = _bind()
    with _cpu_target():
        source = bound.to_code(_config(compute, bk=bk))
    tree = ast.parse(source)
    loops = [
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == name
        )
        for name in ("collective_ai", "collective_bi")
    ]
    staging = compile(
        ast.fix_missing_locations(ast.Module(body=loops, type_ignores=[])),
        "<scaled-packed-staging>",
        "exec",
    )
    a = (torch.arange(73 * 5 * 8).reshape(73, 5, 8) % 256).to(torch.uint8)
    b = (torch.arange(5 * 8 * 41).reshape(5, 8, 41) % 256).to(torch.uint8)
    sa = torch.full((1024,), 127, dtype=torch.uint8)
    sb = torch.full((1024,), 255, dtype=torch.uint8)
    for raw, rows in ((sa, 73), (sb, 41)):
        for row in range(rows):
            for group in range(5):
                bits = (row * 13 + group * 7) % 254
                raw[_scale_offset(row, group, 5)] = bits if bits < 127 else bits + 1
    a_decoded = torch.stack(
        (_FP4[(a & 15).long()], _FP4[(a >> 4).long()]), dim=3
    ).reshape(73, 80)
    b_decoded = torch.stack(
        (_FP4[(b & 15).long()], _FP4[(b >> 4).long()]), dim=2
    ).reshape(80, 41)
    for row in range(73):
        for group in range(5):
            a_decoded[row, group * 16 : (group + 1) * 16] *= (
                sa[_scale_offset(row, group, 5)].view(torch.float8_e4m3fn).float()
            )
    for col in range(41):
        for group in range(5):
            b_decoded[group * 16 : (group + 1) * 16, col] *= (
                sb[_scale_offset(col, group, 5)].view(torch.float8_e4m3fn).float()
            )
    logical_bk = bk * 16
    for group_offset in range(0, 5, bk):
        staged_a = torch.full((64, logical_bk), float("nan"))
        staged_b = torch.full((32, logical_bk), float("nan"))
        for tid in range(128):
            namespace = {
                "cutlass": SimpleNamespace(
                    Int32=int,
                    Int64=int,
                    Uint8=int,
                    Float32=float,
                    BFloat16=lambda value: (
                        torch.tensor(value).bfloat16().float().item()
                    ),
                    range=lambda *args, **kwargs: range(*args),
                ),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(
                        load=lambda pointer, dtype: pointer.load(),
                        thread_idx=lambda coordinates=(tid % 4, tid // 4, 0): (
                            coordinates
                        ),
                    )
                ),
                "operator": operator,
                "_cute_float4_e2m1fn_x2_to_float32": lambda value: (
                    _FP4[value & 15].item(),
                    _FP4[value >> 4].item(),
                ),
                "_cute_fp8e4m3fn_to_float32": lambda value: (
                    torch.tensor(value, dtype=torch.uint8)
                    .view(torch.float8_e4m3fn)
                    .float()
                    .item()
                ),
                "a_fp4x2": _ScalarTensor(a),
                "b_fp4x2": _ScalarTensor(b),
                "act_scale": _ScalarTensor(sa),
                "weight_scale": _ScalarTensor(sb),
                "tile_offset_0": 64,
                "tile_offset_1": 32,
                "tile_offset_2": group_offset,
                "collective_tid": tid,
                "collective_a": _ScalarTensor(staged_a),
                "collective_b": _ScalarTensor(staged_b),
            }
            exec(staging, namespace)
        expected_a, expected_b = torch.zeros_like(staged_a), torch.zeros_like(staged_b)
        width = min(logical_bk, 80 - group_offset * 16)
        expected_a[:9, :width] = a_decoded[
            64:, group_offset * 16 : group_offset * 16 + width
        ]
        expected_b[:9, :width] = b_decoded[
            group_offset * 16 : group_offset * 16 + width, 32:
        ].T
        torch.testing.assert_close(
            staged_a, expected_a, rtol=0, atol=0, equal_nan=False
        )
        torch.testing.assert_close(
            staged_b, expected_b, rtol=0, atol=0, equal_nan=False
        )


def test_raw_byte_loads_participate_in_alias_proof() -> None:
    value = ast.parse("cute.arch.load(A.iterator + k, cutlass.Uint8)", mode="eval").body
    assert _tensor_roots([value], "load") == {"A"}
    unknown = ast.parse("cute.arch.load(pointer, cutlass.Uint8)", mode="eval").body
    assert _tensor_roots([unknown], "load") is None
