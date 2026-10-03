from __future__ import annotations

import ast
import hashlib
import importlib
import json
import operator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_producer_teams import _args
from .test_cute_chained_producer_teams import _cooperative_loop
from .test_cute_chained_vector_stage import _config
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_vector_expression import emit_vector_expression
from helion._compiler.cute.chained_vector_stage import emit_vector_stage
import helion.language as hl

# Captured from the original stage emitter before caller integration. Mixed
# downstream operands change the whole kernel, but not these host-vector fills.
_STAGE_DIGESTS = {
    (False, 4): "a0368216e2d0182f7c7693b4293a05eb4a64b49eb7fe266553401410aa495e76",
    (False, 16): "ca742b76687f3c651acab5b346086e162ee259b5b372da1a21862dec2968dbc6",
    (False, 32): "3a8f7e74d7fb7a9ad21e3010d0990a0752e2cfbe14cf177acbe8c3f24e247d49",
    (True, 4): "3a853ae3581aec573418f0a9cb5e6ece2bd7cae3ffcd0e33e1f0f56cede1f656",
    (True, 16): "fc87092704c77ef537589fa7c337460bd48a3bce08cf7d08d1c188a8ef0d0ab9",
    (True, 32): "d6154cbf9f1dd2aa3ad8baca219759e570f2fd7cf3a76e152e88c4bf0c732b66",
}


def _stage_adapter(cg, plan, boundaries, operand, geometry, *, role, **kwargs):
    return emit_vector_expression(
        cg,
        plan,
        boundaries,
        operand,
        coordinates=lambda row, column: geometry.operand(role, row, column)[1],
        final_value=lambda value, dtype, coords: chain._masked_operand(
            value, dtype, chain._operand_domain(cg, operand, coords, plan)
        ),
        **kwargs,
    )


@pytest.mark.parametrize("warps", [4, 16, 32])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_real_stage_extraction_preserves_complete_generated_source(
    warps, grouped, mixed
):
    def generate(emitter):
        emitted = []

        def record(*args, **kwargs):
            lines = emitter(*args, **kwargs)
            emitted.append(lines)
            return lines

        with _cpu_codegen():
            bound = _cooperative_loop._bind_isolated(_args("cpu", 3, mixed))
            with patch.object(chained_tcgen_stage, "emit_vector_stage", record):
                source = bound.to_code(_config(warps, True, grouped))
        return source, emitted

    before, original = generate(emit_vector_stage)
    after, extracted = generate(_stage_adapter)
    assert any(lines is not None for lines in original)
    assert extracted == original
    assert (
        hashlib.sha256(json.dumps(extracted).encode()).hexdigest()
        == _STAGE_DIGESTS[grouped, warps]
    )
    assert (
        hashlib.sha256(after.encode()).digest()
        == hashlib.sha256(before.encode()).digest()
    )
    assert after == before


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _frontier_loop(
    a, b, initial, control, dtype, varying: hl.constexpr, sigmoid: hl.constexpr
):
    steps, m, k = a.shape
    n = b.shape[-1]
    values = torch.empty((steps, (n + 7) // 8, m, k), dtype=dtype, device=a.device)
    final = torch.empty_like(initial)
    for rows, columns in hl.tile([m, n], block_size=[16, 8]):
        state = initial[rows, columns]
        shift = control[0].to(torch.int32)
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            indices = (rows.index + shift) % m
            if varying:
                mask = (kk % 2 == 0)[None, :]
            else:
                mask = (indices < m - 1)[:, None]
            loaded = hl.load(a, [step.id, indices, kk], extra_mask=mask)
            rounded = (loaded.float() * 0.125).to(torch.float16).float()
            if sigmoid:
                frontier = torch.sigmoid(rounded).to(dtype)
            else:
                frontier = torch.exp(rounded).to(dtype)
            values[step.id, columns.id, rows, kk] = frontier
            state = hl.dot(
                frontier.to(torch.bfloat16),
                b[step.id, kk, columns],
                acc=state,
                out_dtype=torch.float32,
            )
        final[rows, columns] = state
    return values, final


def _frontier_args(
    dtype=torch.float32, varying=False, sigmoid=False, host_dtype=torch.bfloat16
):
    return (
        torch.empty((3, 19, 16), dtype=host_dtype),
        torch.empty((3, 16, 16), dtype=torch.bfloat16),
        torch.empty((19, 16), dtype=torch.float32),
        torch.tensor([1], dtype=torch.int32),
        dtype,
        varying,
        sigmoid,
    )


def _frontier_capture(
    dtype=torch.float32,
    varying=False,
    *,
    shape=None,
    transposed=False,
    boundary=False,
    final_domain=False,
    execution=None,
    unroll=None,
    select_dtype=None,
    sigmoid=False,
    host_dtype=torch.bfloat16,
):
    captured = []
    original = chained_tcgen_stage.emit_stage

    def observe(cg, plan, boundaries, *args, **kwargs):
        if not captured:
            assert plan.loop is not None
            node = plan.region.stores[0].args[2]
            assert node.meta["val"].dtype == dtype
            if select_dtype is not None:
                node = next(
                    candidate
                    for candidate in plan.region.nodes
                    if isinstance(candidate.meta.get("val"), torch.Tensor)
                    and candidate.meta["val"].dtype == select_dtype
                )
            logical = chain._shape(node)
            coords = (
                (lambda row, column: (column, row))
                if transposed
                else (lambda row, column: (row, column))
            )
            selected = (
                {**boundaries, node: "existing_boundary"} if boundary else boundaries
            )
            lines = emit_vector_expression(
                cg,
                plan,
                selected,
                node,
                shape=shape or (16, 16),
                coordinates=coords,
                offset=32,
                tag="frontier",
                target="prepared_image",
                final_value=(
                    lambda value, dtype_name, coordinates: chain._masked_operand(
                        value,
                        dtype_name,
                        chain._operand_domain(cg, node, coordinates, plan),
                    )
                )
                if final_domain
                else None,
                execution=execution,
                producer_unroll=unroll,
            )
            captured.append((lines, plan.loop.captures(), logical))
        return original(cg, plan, boundaries, *args, **kwargs)

    with _cpu_codegen():
        bound = _frontier_loop._bind_isolated(
            _frontier_args(dtype, varying, sigmoid, host_dtype)
        )
        with patch.object(chained_tcgen_stage, "emit_stage", observe):
            bound.to_code(_config(4, False, False))
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_frontier_preserves_typed_masked_values_captures_and_original_casts(dtype):
    lines, captures, shape = _frontier_capture(dtype)
    assert lines is not None and captures and shape == (16, 16)
    source = "\n".join(lines)
    ast.parse(source)
    assert "cute.domain_offset((32, 0), prepared_image)" in source
    assert any(name in source for name in captures.values())
    assert "cutlass.Int32" in source
    assert "cutlass.Float16" in source and "cutlass.Float32" in source
    assert "_last_pointer =" in source and "_last_address ==" in source
    assert "_vectorized = cutlass.Boolean(False)" in source
    assert "exp" in source
    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "frontier_values[frontier_element]"
    ]
    assert len(assignments) == 2
    assert all(isinstance(node.value, ast.Call) for node in assignments)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_nonfloating_frontier_rejects_instead_of_packing(dtype):
    assert _frontier_capture(select_dtype=dtype)[0] is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("host_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_host_leaf_dtype_and_final_storage_dtype_keep_separate_provenance(
    dtype, host_dtype
):
    names = {
        torch.bfloat16: "cutlass.BFloat16",
        torch.float16: "cutlass.Float16",
        torch.float32: "cutlass.Float32",
    }
    lines, _, _ = _frontier_capture(dtype, host_dtype=host_dtype)
    assert lines is not None
    source = "\n".join(lines)
    assert (
        f"frontier_leaf_0_values = cute.make_rmem_tensor((8,), {names[host_dtype]})"
        in source
    )
    assert f"CopyUniversalOp(), {names[dtype]}, num_bits_per_copy=128)" in lines[0]
    byte_delta = 28 if host_dtype == torch.float32 else 14
    assert f"_last_address == frontier_leaf_0_address + {byte_delta}" in source


@pytest.mark.parametrize("reason", ["boundary", "varying_mask", "transposed"])
def test_boundary_only_or_unproved_host_vector_retains_scalar_path(reason):
    assert (
        _frontier_capture(
            varying=reason == "varying_mask",
            boundary=reason == "boundary",
            transposed=reason == "transposed",
        )[0]
        is None
    )


@pytest.mark.parametrize("width", [0, 7, 24, 2048])
def test_unsupported_physical_width_rejects_without_unroll_activation(width):
    unroll = BoundedProducerUnroll(8)
    assert _frontier_capture(shape=(16, width), unroll=unroll)[0] is None
    assert not unroll.activated


def test_explicit_final_domain_is_separate_from_original_leaf_mask():
    plain = "\n".join(_frontier_capture()[0])
    domain = "\n".join(_frontier_capture(final_domain=True)[0])
    assert "frontier_values[frontier_element] = (cutlass.Float32(" not in plain
    assert "frontier_values[frontier_element] = (cutlass.Float32(" in domain
    assert "else cutlass.Float32(0))" in domain


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("final_domain", [False, True])
def test_actual_scalar_fallback_keeps_masked_zero_pointwise_semantics(
    dtype, sigmoid, final_domain
):
    lines, captures, _ = _frontier_capture(
        dtype, sigmoid=sigmoid, final_domain=final_domain
    )
    assert lines is not None
    tree = ast.parse("\n".join(lines))
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not frontier_leaf_0_vectorized"
    )
    pointwise = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "frontier_element"
    )
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Float32=lambda value: torch.tensor(value, dtype=torch.float32).item(),
            Float16=lambda value: torch.tensor(value, dtype=torch.float16).item(),
            BFloat16=lambda value: torch.tensor(value, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                exp2=lambda value, **kwargs: 2.0**value,
                rcp=lambda value, **kwargs: 1 / value,
            )
        ),
        "operator": operator,
        "frontier_leaf_0_vectorized": False,
        "frontier_leaf_0_values": [float("nan")] * 8,
        "frontier_values": [float("nan")] * 8,
        "frontier_row": 5,
        "frontier_base": 0,
        "chain_origin_0": 16,
        "chain_loop_index": 0,
        **dict.fromkeys(captures.values(), 16),
    }
    # Execute the generated scalar fallback and original pointwise statements.
    # Row (16 + 5 + 16) % 19 == 18 is masked out. No host tensor or pointer is
    # supplied: touching memory instead of taking that mask would fail here.
    executable = ast.Module(body=[fallback, pointwise], type_ignores=[])
    exec(compile(executable, "<vector-expression-fallback>", "exec"), namespace)
    assert namespace["frontier_leaf_0_values"] == [0] * 8
    expected = 0 if final_domain else (0.5 if sigmoid else 1)
    assert namespace["frontier_values"] == [expected] * 8


def test_role_local_threads_padded_rows_and_bounded_unroll():
    unroll = BoundedProducerUnroll(8)
    lines, _, _ = _frontier_capture(
        shape=(257, 16),
        execution=ChainedExecution(256, thread="prep_thread", sync="forbidden()"),
        unroll=unroll,
    )
    assert lines is not None and unroll.activated
    source = "\n".join(lines)
    assert "get_slice(prep_thread)" in source
    assert "cutlass.range(3, unroll=3)" in source
    assert "if frontier_row < 257:" in source
    assert "((frontier_row) < 16)" in source
    assert "forbidden" not in source and "sync_threads" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "height,width,threads", [(16, 16, 128), (257, 16, 256), (19, 64, 128)]
)
def test_actual_cute_destination_partition_keeps_eight_values_and_row_offset_cpu(
    dtype, height, width, threads
):
    import cutlass
    import cutlass.cute as cute

    lines, _, _ = _frontier_capture(
        dtype, shape=(height, width), execution=ChainedExecution(threads)
    )
    assert lines is not None
    native_dtype = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = cute.make_layout((height + 32, width), stride=(width, 1))
            target = cute.make_tensor(
                cute.make_ptr(native_dtype, 0, cute.AddressSpace.smem), layout
            )
            identity = cute.domain_offset(
                (32, 0), cute.make_identity_tensor((height + 32, width))
            )
            columns = width // 8
            rows = threads // columns
            coverage = set()
            for thread in range(threads):
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "prepared_image": target,
                    "chain_thread": thread,
                }
                exec("\n".join(lines[:4]), namespace)
                values = namespace["frontier_values"]
                assert values.element_type == native_dtype
                assert int(cute.size(values)) == 8
                coordinates = namespace["frontier_thread"].partition_D(identity)
                for step in range((height + rows - 1) // rows):
                    row = thread // columns + step * rows
                    if row >= height:
                        continue
                    for element in range(8):
                        actual = tuple(map(int, coordinates[None, step, 0][element]))
                        expected = (32 + row, thread % columns * 8 + element)
                        assert actual == expected and actual not in coverage
                        coverage.add(actual)
            assert coverage == {
                (32 + row, column) for row in range(height) for column in range(width)
            }
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before
