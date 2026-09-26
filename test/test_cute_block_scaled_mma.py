from __future__ import annotations

import ast
from dataclasses import replace
import operator
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.nvfp4_gemm import _nvfp4_w4a4_matmul_kernel
import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_scaled_contraction import _config
from test.test_cute_scaled_contraction import _inputs

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute_block_scaled import (
    CuteBlockScaledMmaHeuristic,
)
from helion._compiler.cute.block_scaled_config import BLOCK_SCALED_CONFIG_KEYS
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
from helion._compiler.cute.block_scaled_mma import _plan_block_scaled
from helion._compiler.cute.block_scaled_mma import render_integer_recipe
from helion._compiler.cute.block_scaled_provenance import BLOCK_SCALED_PROVENANCE
from helion._compiler.cute.block_scaled_provenance import IntegerRecipe
from helion._compiler.cute.block_scaled_provenance import _plain_load
from helion._compiler.cute.block_scaled_provenance import integer_recipe
from helion._compiler.cute.block_scaled_provenance import operand_fingerprint
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.language import creation_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import GraphInfo
    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", autotune_effort="none")
def _plain_scales(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, groups, _ = a.shape
    n = b.size(2)
    for row, col in hl.tile([m, n]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for group in hl.tile(groups):
            left_scale = sa[row, group].to(torch.float32)
            right_scale = sb[group, col].to(torch.float32)
            for byte in hl.static_range(8):
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


def _native_config(**kwargs: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            **_config(),
            "cute_scaled_mma": True,
            "cute_collective_mma": False,
            **kwargs,
        }
    )


def _bind_native(
    shape: tuple[int, int, int] = (73, 5, 41),
    *,
    static: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    generic: bool = False,
) -> tuple[BoundKernel, tuple[object, ...]]:
    m, groups, n = shape
    args = _inputs(*shape)
    if generic:
        args = (
            args[0],
            args[1],
            torch.empty((m, groups), dtype=torch.float8_e4m3fn),
            torch.empty((groups, n), dtype=torch.float8_e4m3fn),
            torch.empty((m, n), dtype=dtype),
        )
    else:
        args = (*args[:4], torch.empty((m, n), dtype=dtype), args[5])
    with _cpu_target():
        kernel = helion.kernel(
            _plain_scales.fn if generic else _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=static,
            autotune_effort="none",
        )
        return kernel._bind_isolated(args), args


def _wrapper_plan(source: str) -> dict[str, object]:
    plans = [
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_helion_cute_wrapper_plans"
            for target in node.targets
        )
    ]
    assert len(plans) == 1 and len(plans[0]) == 1
    assert plans[0][0]["kind"] == "block_scaled_mma"
    return plans[0][0]


@pytest.mark.parametrize(
    "shape", [(128, 16, 128), (512, 64, 512), (1024, 128, 1024), (73, 5, 41)]
)
@pytest.mark.parametrize("static", [False, True])
@skipUnlessBackends(["cute"])
def test_original_and_dynamic_metadata_are_proved(
    shape: tuple[int, int, int], static: bool
) -> None:
    bound, _ = _bind_native(shape, static=static)
    with _cpu_target():
        plan = _wrapper_plan(bound.to_code(_native_config()))
    assert (plan["m"], plan["k"], plan["n"]) == (shape[0], shape[1] * 16, shape[2])
    assert plan["lhs_row_fast"] is False and plan["rhs_row_fast"] is True
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
def test_generic_scale_layout_and_output_dtype(dtype: torch.dtype) -> None:
    bound, _ = _bind_native(generic=True, static=False, dtype=dtype)
    with _cpu_target():
        plan = _wrapper_plan(
            bound.to_code(
                _native_config(
                    cute_scaled_cluster_m=1,
                    cute_scaled_tile_n=128,
                    cute_scaled_tile_k=64,
                    cute_scaled_stages=3,
                )
            )
        )
    assert plan["lhs_scale_span"] == 73 * 5
    assert plan["rhs_scale_span"] == 5 * 41
    assert plan["scale_value"] == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@skipUnlessBackends(["cute"])
def test_explicit_intermediate_rounding_is_retained(dtype: torch.dtype) -> None:
    bound, _ = _bind_native(dtype=dtype)
    with _cpu_target():
        plan = _wrapper_plan(bound.to_code(_native_config()))
    assert plan["round_dtypes"] == ("torch.bfloat16", str(dtype))


@pytest.mark.parametrize("mask,other", [(True, None), (None, 1), (False, 0)])
def test_source_load_mask_and_fill_require_the_original_proof(
    mask: bool | None, other: int | None
) -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty((8,), dtype=torch.float4_e2m1fn_x2)
    load = graph.call_function(memory_ops.load, (source, [0], mask, other))
    assert _plain_load(load, torch.float4_e2m1fn_x2) is None


@skipUnlessBackends(["cute"])
def test_output_aliasing_an_input_declines_native_reordering() -> None:
    args = list(_inputs(4, 3, 4))
    packed = cast("torch.Tensor", args[0])
    args[4] = packed.view(torch.bfloat16).reshape(-1)[:16].view(4, 4)
    with _cpu_target():
        kernel = helion.kernel(
            _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        bound = kernel._bind_isolated(tuple(args))
        with pytest.raises(
            exc.BackendUnsupported, match="block-scaled contraction proof declined"
        ):
            bound.to_code(_native_config())


@pytest.mark.parametrize("strides", [(0, 1), (1, 0), (1, 1)])
@skipUnlessBackends(["cute"])
def test_overlapping_output_elements_decline_native_reordering(
    strides: tuple[int, int],
) -> None:
    args = list(_inputs(4, 3, 4))
    args[4] = torch.empty_strided((4, 4), strides, dtype=torch.bfloat16)
    with _cpu_target():
        kernel = helion.kernel(
            _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
        )
        bound = kernel._bind_isolated(tuple(args))
        with pytest.raises(
            exc.BackendUnsupported, match="block-scaled contraction proof declined"
        ):
            bound.to_code(_native_config())


@pytest.mark.parametrize("strides", [(1, 6), (7, 1), (11, 2)])
@skipUnlessBackends(["cute"])
def test_noncontiguous_disjoint_output_layout_is_supported(
    strides: tuple[int, int],
) -> None:
    args = list(_inputs(4, 3, 4))
    args[4] = torch.empty_strided((4, 4), strides, dtype=torch.bfloat16)
    with _cpu_target():
        kernel = helion.kernel(
            _nvfp4_w4a4_matmul_kernel.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
        )
        bound = kernel._bind_isolated(tuple(args))
        assert _wrapper_plan(bound.to_code(_native_config()))["m"] == 4


@skipUnlessBackends(["cute"])
def test_all_native_seeds_round_trip_and_emit() -> None:
    bound, _ = _bind_native()
    assert bound.host_function is not None
    with bound.env, _cpu_target():
        seeds = CuteBlockScaledMmaHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        generation = ConfigGeneration(bound.config_spec)
        assert len(seeds) == 8
        for seed in seeds:
            bound.config_spec.normalize(seed)
            restored = generation.unflatten(generation.flatten(seed))
            assert all(restored[key] == value for key, value in seed.items())
    with _cpu_target():
        for seed in seeds:
            assert _wrapper_plan(bound.to_code(seed))["kind"] == "block_scaled_mma"


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_scaled_mma", 1),
        ("cute_scaled_cluster_m", True),
        ("cute_scaled_tile_k", 32),
        ("cute_scaled_tile_n", 64),
        ("cute_scaled_stages", 1),
        ("cute_scaled_persistent", 0),
    ],
)
def test_invalid_native_configuration_is_not_silently_accepted(
    key: str, value: object
) -> None:
    with pytest.raises(exc.InvalidConfig):
        normalize_block_scaled_config(
            {key: value}, available=True, capability=(10, 0), fix_invalid=False
        )


def test_native_configuration_requires_proven_format_and_hardware() -> None:
    for available, capability in ((False, (10, 0)), (True, (9, 0)), (True, None)):
        with pytest.raises(exc.InvalidConfig):
            normalize_block_scaled_config(
                {"cute_scaled_mma": True},
                available=available,
                capability=capability,
                fix_invalid=False,
            )
    config: dict[str, object] = dict.fromkeys(BLOCK_SCALED_CONFIG_KEYS, 123)
    normalize_block_scaled_config(
        config, available=False, capability=(10, 0), fix_invalid=True
    )
    assert not config


@pytest.mark.parametrize(
    "change",
    [
        "loop_effect",
        "root_effect",
        "carried_result",
        "negative_zero",
        "partial_output",
        "unknown_index",
        "unavailable_coordinate",
        "float_epilogue",
        "extra_loop_result",
        "changed_operand",
        "changed_scale_arithmetic",
    ],
)
@skipUnlessBackends(["cute"])
def test_native_region_proof_rejects_changed_semantics(change: str) -> None:
    bound, _ = _bind_native()
    checked = False

    def probe(graphs: list[GraphInfo]) -> None:
        nonlocal checked
        assert _plan_block_scaled(graphs) is not None
        info, contraction = next(
            (info, node)
            for info in graphs
            for node in info.graph.nodes
            if BLOCK_SCALED_PROVENANCE in node.meta
        )
        root = next(
            info
            for info in graphs
            if any(node.target is memory_ops.store for node in info.graph.nodes)
        )
        store = next(
            node for node in root.graph.nodes if node.target is memory_ops.store
        )
        if change in ("loop_effect", "root_effect"):
            graph = info.graph if change == "loop_effect" else root.graph
            with graph.inserting_before(
                next(node for node in graph.nodes if node.op == "output")
            ):
                graph.call_function(memory_ops.store, store.args)
        elif change == "carried_result":
            contraction.args = (*contraction.args[:2], 0.0, contraction.args[3])
        elif change == "negative_zero":
            seed = next(
                node for node in root.graph.nodes if node.target is creation_ops.full
            )
            seed.args = (seed.args[0], -0.0, *seed.args[2:])
        elif change == "partial_output":
            store.args = (*store.args[:3], True)
        elif change in ("unknown_index", "unavailable_coordinate"):
            provenance = contraction.meta[BLOCK_SCALED_PROVENANCE]
            if change == "unknown_index":
                invalid = IntegerRecipe("opaque", value=0)
            else:
                invalid = IntegerRecipe("coordinate", value=99999)
            lhs = replace(provenance.lhs, scale_indices=(invalid,))
            contraction.meta[BLOCK_SCALED_PROVENANCE] = replace(provenance, lhs=lhs)
        elif change == "float_epilogue":
            multiply = next(
                node
                for node in root.graph.nodes
                if node.target is torch.ops.aten.mul.Tensor
            )
            multiply.meta["val"] = multiply.meta["val"].double()
        elif change == "extra_loop_result":
            output = next(node for node in info.graph.nodes if node.op == "output")
            output.args = ([contraction, contraction],)
        elif change == "changed_operand":
            contraction.args = (
                contraction.args[1],
                contraction.args[0],
                *contraction.args[2:],
            )
        elif change == "changed_scale_arithmetic":
            multiply = next(
                node
                for node in info.graph.nodes
                if node.target is torch.ops.aten.mul.Tensor
                and isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].dtype == torch.float32
            )
            multiply.args = (multiply.args[0], 2.0)
        assert _plan_block_scaled(graphs) is None
        checked = True
        raise _StoppedProof

    with (
        _cpu_target(),
        patch(
            "helion._compiler.cute.block_scaled_mma.plan_block_scaled",
            side_effect=probe,
        ),
        pytest.raises(_StoppedProof),
    ):
        bound.to_code(_native_config())
    assert checked


class _StoppedProof(Exception):
    pass


@pytest.mark.parametrize(
    "target", [operator.getitem, operator.neg, torch.ops.aten.item.default]
)
@skipUnlessBackends(["cute"])
def test_integer_recipe_rejects_opaque_values_even_with_concrete_metadata(
    target: Callable[..., object],
) -> None:
    bound, _ = _bind_native()
    graph = torch.fx.Graph()
    node = graph.call_function(target, (0,))
    node.meta["val"] = 7
    with bound.env:
        assert integer_recipe(node, bound.env) is None


def test_integer_recipe_uses_strict_bounds_and_expansion_limit() -> None:
    row = IntegerRecipe("coordinate", value=0, bits=32)
    literal = IntegerRecipe("constant", value=1)

    def render(recipe: IntegerRecipe, extent: int = 17):
        return render_integer_recipe(
            recipe,
            coordinates={0: ("row", extent)},
            resolve=lambda value: value if type(value) is int else None,
        )

    rendered = render(row)
    assert rendered is not None and rendered.high == 16
    assert render(IntegerRecipe("add", (row, literal), bits=32), (1 << 31)) is None
    assert (
        render(IntegerRecipe("floordiv", (row, IntegerRecipe("constant", value=0))))
        is None
    )
    tree = row
    for _ in range(15):
        tree = IntegerRecipe(
            "mod",
            (
                IntegerRecipe("mul", (tree, tree), bits=32),
                IntegerRecipe("constant", value=17),
            ),
            bits=32,
        )
    assert render(tree) is None
    for _ in range(256):
        row = IntegerRecipe("cast", (row,), bits=32)
    assert render(row) is None


@skipUnlessBackends(["cute"])
def test_source_proof_declines_deep_dags_without_python_recursion_failure() -> None:
    bound, _ = _bind_native()
    graph = torch.fx.Graph()
    node = graph.call_function(operator.add, (0, 1))
    node.meta["val"] = 1
    for _ in range(256):
        node = graph.call_function(operator.add, (node, 1))
        node.meta["val"] = 1
    with bound.env:
        assert integer_recipe(node, bound.env) is None
    assert operand_fingerprint((node, node)) is None


def test_signed_scale_normalization_preserves_every_finite_operand_and_zero_sign() -> (
    None
):
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    encoded_scales = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    scales = encoded_scales.view(torch.float8_e4m3fn).float()
    bytes_ = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    normalized = bytes_[:, None] ^ ((encoded_scales[None, :] >> 7) * 136)
    positive_scales = (encoded_scales & 127).view(torch.float8_e4m3fn).float()
    for shift in (0, 4):
        original = values[((bytes_.to(torch.int64) >> shift) & 15)[:, None]] * scales
        prepared = values[(normalized.to(torch.int64) >> shift) & 15] * positive_scales
        finite = ~torch.isnan(original)
        assert torch.equal(
            original.view(torch.int32)[finite], prepared.view(torch.int32)[finite]
        )
        assert torch.equal(torch.isnan(original), torch.isnan(prepared))
