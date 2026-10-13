from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_coordinates import _split_view_sequence
from .test_cute_chained_coordinates import _view_args
from .test_cute_chained_row_registers import _fixture
from .test_cute_chained_split_join import _quadrants
from .test_cute_chained_warp_k8 import _inputs
from .test_cute_chained_warp_k8 import _seeded_chain
import helion
from helion._compiler.cute.chained_coordinates import CoordinateDomain
from helion._compiler.cute.chained_coordinates import IndexExpr
from helion._compiler.cute.chained_coordinates import binary
from helion._compiler.cute.chained_coordinates import constant
from helion._compiler.cute.chained_coordinates import original_coordinate
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_operand_coordinates import OperandCoordinates
from helion._compiler.cute.chained_operand_coordinates import operand_scope
from helion._compiler.cute.chained_root_warp_stage import WarpOperandLayout
from helion.autotuner.config_spec import EnumFragment
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from torch.fx import Node

    from helion._compiler.generate_ast import GenerateAST


@pytest.mark.parametrize("shape", [(8, 16), (16, 8), (32, 32)])
def test_composed_runtime_coordinates_and_domain_at_every_point(shape) -> None:
    rows, columns = shape
    domain = CoordinateDomain("r", "c", (0, rows - 1), (0, columns - 1))
    context = OperandCoordinates(domain)
    split = context.reshape(("r", "c"), (2, rows // 2, 2, columns // 2), shape)
    assert split is not None
    joined = context.reshape(split, shape, (2, rows // 2, 2, columns // 2))
    assert joined is not None
    assert context.in_bounds(joined[0], rows)
    assert context.in_bounds(joined[1], columns)
    assert not context.in_bounds(joined[0], rows - 1)
    assert not context.in_bounds(joined[1], columns - 1)
    points = domain.points()
    assert points is not None
    for r, c in points:
        ns = {
            "r": np.int32(r),
            "c": np.int32(c),
            "cutlass": SimpleNamespace(Int32=np.int32),
        }
        assert tuple(int(eval(s, ns)) for s in joined) == (r, c)
    assert original_coordinate(joined[0]) != "r"
    column = context.lookup("c")
    assert column is not None
    zero = context.normalize(binary("mul", column, constant(0)), "(c * 0)")
    assert isinstance(
        eval(zero, {"c": np.int32(7), "cutlass": SimpleNamespace(Int32=np.int32)}),
        np.int32,
    )


@pytest.mark.parametrize(
    "case",
    ["shift", "negative", "overflow", "cancel", "unknown", "cast", "variable_divisor"],
)
def test_partial_or_unproved_domains_are_not_erased(case) -> None:
    context = OperandCoordinates(CoordinateDomain("r", "c", (0, 31), (0, 31)))
    r = IndexExpr("row")
    expressions = {
        "shift": binary("add", r, constant(1)),
        "negative": binary("sub", r, constant(1)),
        "overflow": binary("mul", r, constant(2**30)),
        "cancel": binary("div", binary("mul", r, constant(2**30)), constant(2**30)),
        "unknown": IndexExpr("memory_index"),
        "cast": IndexExpr("int64", (r,)),
        "variable_divisor": binary("div", r, IndexExpr("column")),
    }
    result = context.normalize(expressions[case], "original")
    assert not context.in_bounds(result, 32)
    for text in ("r + 1", "cutlass.Int32(r)", "loaded_index", "data[r]"):
        assert not context.in_bounds(text, 32)


def test_nonaffine_piece_is_retained_typed_without_an_affine_rewrite() -> None:
    context = OperandCoordinates(CoordinateDomain("r", "c", (0, 31), (0, 31)))
    result = context.normalize(binary("mod", IndexExpr("row"), constant(3)), "(r % 3)")
    assert result == "(r % 3)"
    assert context.in_bounds(result, 3)
    assert not context.in_bounds(result, 2)


def test_scope_rejects_stale_stage_layout_and_loop_bound() -> None:
    plan, _, _ = _fixture()
    cg = SimpleNamespace(
        device_function=SimpleNamespace(
            config=SimpleNamespace(config={"cute_chained_operand_coordinates": True})
        )
    )
    layout = WarpOperandLayout(
        cast("Node", plan.dots[0].args[0]), "a", (16, 16), (20, 1), 1, torch.bfloat16
    )
    with _cpu_codegen():
        scope = operand_scope(
            cast("GenerateAST", cg),
            plan,
            0,
            layout,
            "coalesced",
            8,
            1,
            "cutlass.range(1, unroll=1)",
        )
        assert scope is not None
        for changed in (
            replace(scope, stage=99),
            replace(scope, vector=4),
            replace(scope, steps=2),
            replace(scope, load_range="cutlass.range(2, unroll=1)"),
            replace(scope, domain=replace(scope.domain, rows=(0, 31))),
            replace(scope, layout=replace(layout, stride=(16, 1))),
            replace(scope, layout=replace(layout, role="b")),
        ):
            with pytest.raises(_UnsupportedChain):
                changed.check(cast("GenerateAST", cg), plan, 0, layout)
        # An equal but separately constructed owner is another emit attempt.
        with pytest.raises(_UnsupportedChain, match="foreign"):
            scope.check(cast("GenerateAST", cg), plan, 0, replace(layout))
        cg.device_function.config.config["num_warps"] = 8
        with pytest.raises(_UnsupportedChain):
            scope.check(cast("GenerateAST", cg), plan, 0, layout)


def _config(**kwargs):
    return helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced", **kwargs)


def _assert_quadrant_staging_values(source: str, dtype: torch.dtype) -> None:
    """Evaluate every thread's view staging with symbolic shared values/casts."""
    loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_1_load_step"
    ]
    assert len(loops) == 2
    program = compile(
        ast.Module(body=loops, type_ignores=[]), "operand-staging", "exec"
    )
    reads = []

    class Shared:
        def __getitem__(self, index):
            row, column = map(int, index)
            assert 0 <= row < 32 and 0 <= column < 32
            reads.append((row, column))
            return ("source", row, column)

    class Operand:
        def __init__(self):
            self.values = {}

        def __setitem__(self, index, value):
            row, column = map(int, index)
            assert 0 <= row < 16 and 0 <= column < 16
            assert (row, column) not in self.values
            self.values[row, column] = value

    a, b = Operand(), Operand()
    namespace = {
        "cutlass": SimpleNamespace(
            range=lambda count, **kwargs: range(count),
            range_constexpr=range,
            Int32=np.int32,
            Float16=lambda value: ("Float16", value),
            BFloat16=lambda value: ("BFloat16", value),
            Float32=lambda value: ("Float32", value),
        ),
        "chain_0_c": Shared(),
        "chain_1_a": a,
        "chain_1_b": b,
    }
    for thread in range(128):
        namespace["chain_thread"] = thread
        exec(program, namespace)
    scalar = "Float16" if dtype is torch.float16 else "BFloat16"
    expected_a = {}
    expected_b = {}
    for row in range(16):
        for column in range(16):
            expected_a[row, column] = (scalar, (scalar, ("source", row, column)))
            expected_b[column, row] = (
                scalar,
                (scalar, ("source", row + 16, column + 16)),
            )
    assert a.values == expected_a and b.values == expected_b
    assert len(reads) == 512 and len(set(reads)) == 512


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_typed_view_codegen_preserves_original_schedule(dtype) -> None:
    with _cpu_codegen():
        args = (
            torch.empty((2, 32, 32), dtype=dtype),
            torch.empty((2, 32, 32), dtype=dtype),
            0.0,
        )
        bound = _quadrants._bind_isolated(args)
        config = _config()
        old = bound.to_code(config)
        assert (
            bound.to_code(
                helion.Config.from_dict(
                    {**config.config, "cute_chained_operand_coordinates": False}
                )
            )
            == old
        )
        new = bound.to_code(
            helion.Config.from_dict(
                {**config.config, "cute_chained_operand_coordinates": True}
            )
        )
    assert len(new) < len(old)
    for call in (
        "cute.gemm(",
        "sync_threads()",
        "make_rmem_tensor(",
        "allocate_tensor(",
    ):
        assert new.count(call) == old.count(call)
    _assert_quadrant_staging_values(old, dtype)
    _assert_quadrant_staging_values(new, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_original_masked_rows_remain_masked(dtype) -> None:
    args, _ = _inputs(dtype, rows=19, columns=16, reduction=8, transpose=True)
    with _cpu_codegen():
        bound = _seeded_chain._bind_isolated(args)
        config = _config(block_sizes=[32])
        old = bound.to_code(config)
        new = bound.to_code(
            helion.Config.from_dict(
                {**config.config, "cute_chained_operand_coordinates": True}
            )
        )
    assert "< 19" in old and "< 19" in new
    assert new.count("cute.gemm(") == old.count("cute.gemm(")
    assert new.count("sync_threads()") == old.count("sync_threads()")


def test_option_validation_and_default_off_search_field() -> None:
    with _cpu_codegen():
        bound = _split_view_sequence._bind_isolated(_view_args(torch.float16))
        spec = bound.config_spec
        field = spec._flat_fields()["cute_chained_operand_coordinates"]
        assert isinstance(field, EnumFragment)
        assert field.default() is False
        assert field.choices == (False, True)
        default = spec.default_config()
        assert "cute_chained_operand_coordinates" not in default.config
        for value in (1, "true", None):
            with pytest.raises(InvalidConfig, match="must be bool"):
                spec.normalize(_config(cute_chained_operand_coordinates=value))
        for schedule in ("tcgen05_tmem", "tcgen05"):
            with pytest.raises(InvalidConfig, match="ordinary warp"):
                spec.normalize(
                    helion.Config(
                        num_warps=4,
                        cute_chained_mma_schedule=schedule,
                        cute_chained_operand_coordinates=True,
                    )
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tail", [False, True])
def test_gpu_original_typed_products_and_guarded_inputs(dtype, tail) -> None:
    torch.manual_seed(423)
    if tail:
        args, backing = _inputs(
            dtype, rows=19, columns=16, reduction=8, transpose=True, device="cuda"
        )
        for raw in backing:
            raw.fill_(-19)
        for tensor in args:
            tensor.normal_(0, 0.0625)
        a, b, c, seed = args
        first = a.double() @ b.double() + seed.double()
        expected = first.to(dtype).double() @ c.double() + first * 0.125
        kernel, config = _seeded_chain, _config(block_sizes=[32])
    else:
        backing = tuple(
            torch.full((2, 34, 36), -19, dtype=dtype, device="cuda") for _ in range(2)
        )
        a, b = tuple(raw[:, 1:33, 2:34] for raw in backing)
        a.normal_(0, 0.0625)
        b.normal_(0, 0.0625)
        args = (a, b, -1.25)
        first = a.double() @ b.double()
        expected = first.clone()
        expected[:, :16, :16] = (
            first[:, :16, :16].to(dtype).double()
            @ first[:, 16:, 16:].to(dtype).double()
        )
        expected[:, :16, 16:] = -1.25
        kernel, config = _quadrants, _config()
    before = tuple(raw.clone() for raw in backing)
    bound = kernel._bind_isolated(args)
    old = bound.compile_config(config)(*args)
    changed = helion.Config.from_dict(
        {**config.config, "cute_chained_operand_coordinates": True}
    )
    new = bound.compile_config(changed)(*args)
    torch.testing.assert_close(old.double(), expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    for raw, snapshot in zip(backing, before, strict=True):
        assert torch.equal(raw, snapshot)
