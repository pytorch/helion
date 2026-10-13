from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import importlib
import operator
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage as stage_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.chained_vector_expression import emit_vector_expression
from helion._compiler.cute.chained_vector_group import VectorGroupOutput
from helion._compiler.cute.chained_vector_group import _conjunction
from helion._compiler.cute.chained_vector_group import emit_vector_group
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shared_loop(a, initial, control, dtype, varying: hl.constexpr, steps: int):
    steps = hl.specialize(steps)
    m, k = a.shape[1:]
    history = torch.empty((max(steps, 1), m, k), dtype=torch.float32, device=a.device)
    result = torch.empty_like(initial)
    for rows, columns in hl.tile([m, k], block_size=[128, 128]):
        state = initial[rows, columns]
        shift = control[0].to(torch.int32)
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            indices = (rows.index + shift) % m
            if varying:
                mask = (kk % 2 == 0)[None, :]
            else:
                mask = (indices < m - 1)[:, None]
            values = hl.load(a, [step.id, indices, kk], extra_mask=mask)
            rounded = (values.float() * 0.125).to(torch.float16).float()
            common = torch.exp(rounded)
            left = (common * 2).to(dtype)
            right = (common + 1).to(dtype).T
            state = hl.dot(left, right, acc=state, out_dtype=torch.float32)
            history[step.id, rows, columns] = state
        result[rows, columns] = state
    return history, result


def _args(device="cpu", steps=3, dtype=torch.bfloat16, varying=False, host_dtype=None):
    generator = torch.Generator(device=device).manual_seed(761)
    return (
        torch.randn(
            (max(1, steps), 259, 128),
            dtype=host_dtype or dtype,
            device=device,
            generator=generator,
        )
        * 0.1,
        torch.randn(
            (259, 128), dtype=torch.float32, device=device, generator=generator
        ),
        torch.tensor([1], dtype=torch.int32, device=device),
        dtype,
        varying,
        steps,
    )


def _config(warps=4, vectorize=True):
    return helion.Config(
        num_warps=warps,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=vectorize,
    )


def _outputs(plan, stage, boundaries, cg, *, scalar=False, final_domain=True):
    geometry = stage_module.stage_geometry(plan.shapes[stage])
    assert geometry is not None
    result = []
    for role in ("a", "b"):
        arg, _ = geometry.operand(role, "row", "column")
        node = plan.dots[stage].args[arg]
        result.append(
            VectorGroupOutput(
                node,
                f"chain_{stage}_{role}",
                lambda row, column, role=role: geometry.operand(role, row, column)[1],
                final_value=(
                    lambda value, dtype, coords, node=node: chain._masked_operand(
                        value, dtype, chain._operand_domain(cg, node, coords, plan)
                    )
                )
                if final_domain
                else None,
                vector_store=not scalar,
            )
        )
    return result


def _capture(
    *,
    dtype=torch.bfloat16,
    host_dtype=None,
    varying=False,
    change=None,
    shape=(128, 128),
    threads=128,
    final_domain=True,
    scalar=False,
    tracker=None,
    ownership=None,
):
    captured = []
    original = stage_module.emit_stage

    def observe(cg, plan, boundaries, stage, *args, **kwargs):
        original_boundaries = boundaries
        if not captured:
            outputs = _outputs(
                plan, stage, boundaries, cg, scalar=scalar, final_domain=final_domain
            )
            if change is not None:
                outputs, boundaries = change(outputs, boundaries, plan)
            unroll = tracker if tracker is not None else BoundedProducerUnroll(8)
            merged = emit_vector_group(
                cg,
                plan,
                boundaries,
                outputs,
                shape=shape,
                tag="joined",
                execution=ChainedExecution(threads),
                producer_unroll=unroll,
                ownership=ownership,
            )
            ordinary = []
            for index, item in enumerate(outputs):
                ordinary.append(
                    emit_vector_expression(
                        cg,
                        plan,
                        boundaries,
                        item.node,
                        shape=shape,
                        coordinates=item.coordinates,
                        offset=item.offset,
                        target=item.target,
                        tag=f"ordinary_{index}",
                        final_value=item.final_value,
                        vector_store=item.vector_store,
                        execution=ChainedExecution(threads),
                        ownership=ownership,
                    )
                )
            captured.append((merged, ordinary, unroll))
        return original(cg, plan, original_boundaries, stage, *args, **kwargs)

    with _cpu_codegen():
        bound = _shared_loop._bind_isolated(
            _args(dtype=dtype, varying=varying, host_dtype=host_dtype)
        )
        with patch.object(stage_module, "emit_stage", observe):
            bound.to_code(_config(vectorize=False))
    assert len(captured) == 1
    return captured[0]


def _calls(source):
    return Counter(
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    )


def _expanded_outputs(lines, element):
    tree = ast.parse("\n".join(lines))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == element
    )
    definitions = {}
    results = []

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in definitions:
                return self.visit(ast.parse(definitions[node.id], mode="eval").body)
            if node.id.endswith("_row"):
                return ast.Name(id="row", ctx=ast.Load())
            if node.id.endswith("_base"):
                return ast.Name(id="base", ctx=ast.Load())
            if node.id == element:
                return ast.Name(id="element", ctx=ast.Load())
            return node

        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and "_leaf_0_values" in node.value.id:
                return ast.Name(id="original_leaf_value", ctx=ast.Load())
            return self.generic_visit(node)

    condition = loop.body[0]
    assert isinstance(condition, ast.If)
    for statement in condition.body:
        assert isinstance(statement, ast.Assign)
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            definitions[target.id] = ast.unparse(statement.value)
        else:
            results.append(
                ast.dump(
                    Expand().visit(
                        ast.parse(ast.unparse(statement.value), mode="eval").body
                    )
                )
            )
    return results


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("host_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("final_domain", [False, True])
def test_common_expression_exact_per_output_with_original_casts_and_masks(
    dtype, host_dtype, final_domain
):
    merged, separate, unroll = _capture(
        dtype=dtype, host_dtype=host_dtype, final_domain=final_domain
    )
    assert merged is not None and all(lines is not None for lines in separate)
    joined = "\n".join(merged)
    assert _expanded_outputs(merged, "joined_element") == [
        _expanded_outputs(lines, f"ordinary_{index}_element")[0]
        for index, lines in enumerate(separate)
    ]
    assert _calls(joined)["cute.math.exp2"] == 1
    assert sum(_calls("\n".join(lines))["cute.math.exp2"] for lines in separate) == 2
    assert joined.count("_vectorized = cutlass.Boolean(False)") == 1
    assert "_last_address ==" in joined and "chain_loop_capture_" in joined
    assert "cutlass.Float16" in joined and "cutlass.Float32" in joined
    assert "sync_threads" not in joined and "alloc_smem" not in joined
    assert unroll.activated


@pytest.mark.parametrize("threads", [128, 256, 384])
@pytest.mark.parametrize("scalar", [False, True])
def test_role_ownership_scalar_or_native_publication_and_unroll(threads, scalar):
    lines, _, unroll = _capture(threads=threads, scalar=scalar)
    assert lines is not None and unroll.activated
    source = "\n".join(lines)
    assert f"chain_thread // 16 + joined_step * {threads // 16}" in source
    trips = (128 + threads // 16 - 1) // (threads // 16)
    assert f"cutlass.range({trips}, unroll={min(8, trips)})" in source
    if scalar:
        assert "chain_0_a[joined_row + 0, joined_base + joined_element]" in source
        assert "joined_output_0_copy" not in source
    else:
        assert "joined_output_0_copy" in source and "joined_output_1_copy" in source


@pytest.mark.parametrize(
    "reason",
    [
        "single",
        "overlap",
        "negative_offset",
        "boundary",
        "coordinates",
        "domain",
        "no_shared",
    ],
)
def test_rejected_group_never_activates_unroll(reason):
    def change(outputs, boundaries, plan):
        left, right = outputs
        if reason == "single":
            return [left], boundaries
        if reason == "overlap":
            return [left, replace(right, target=left.target)], boundaries
        if reason == "negative_offset":
            return [left, replace(right, offset=-1)], boundaries
        if reason == "boundary":
            return outputs, {
                **boundaries,
                left.node: "already_materialized",
                right.node: "other_materialized",
            }
        if reason == "coordinates":
            return [
                left,
                replace(
                    right, coordinates=lambda row, column: (column, f"({row} + 1)")
                ),
            ], boundaries
        if reason == "domain":
            # The emitter's exact domain comparator is tested independently of
            # source shape; this controlled difference models a padded origin.
            return outputs, boundaries
        if reason == "no_shared":
            return [
                left,
                replace(right, coordinates=lambda row, column: (row, column)),
            ], boundaries
        raise AssertionError(reason)

    if reason == "domain":
        original = chain._operand_domain

        def different(cg, node, coords, plan):
            result = original(cg, node, coords, plan)
            return (
                [*result, "different_extent < 7"]
                if node is plan.dots[0].args[1]
                else result
            )

        with patch.object(chain, "_operand_domain", different):
            lines, _, unroll = _capture(change=change)
    else:
        lines, _, unroll = _capture(change=change)
    assert lines is None and not unroll.activated


@pytest.mark.parametrize("width", [0, 7, 24, 2048])
def test_unsupported_width_rejects_before_unroll(width):
    lines, _, unroll = _capture(shape=(128, width))
    assert lines is None and not unroll.activated


def test_varying_mask_has_no_proven_host_vector():
    lines, _, unroll = _capture(varying=True)
    assert lines is None and not unroll.activated


def test_exact_conjunction_only_reorders_atoms_never_arithmetic():
    assert _conjunction(["a < 3", "(b < 7) & (c == 2)"]) == _conjunction(
        ["c == 2 and a < 3", "b < 7"]
    )
    assert _conjunction(["a + 1 < 3"]) != _conjunction(["a < 2"])
    assert _conjunction(["cutlass.Int32(a + 1) < 3"]) != _conjunction(["a + 1 < 3"])
    assert _conjunction(["a < 3 or b < 7"]) != _conjunction(["a < 3", "b < 7"])


def test_root_unroll_factor_two_and_strict_full_group_contract_are_preserved():
    tracker = PointwiseUnroll(2)
    lines, _, returned = _capture(tracker=tracker)
    assert lines is not None and returned is tracker and tracker.activated
    assert "cutlass.range(16, unroll=2)" in "\n".join(lines)
    strict = PointwiseUnroll(8)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="whole number of unroll groups"
    ):
        _capture(tracker=strict, threads=384)
    assert not strict.activated


def test_fp32_output_keeps_final_storage_distinct_from_half_host_leaf():
    def change(outputs, boundaries, plan):
        left, right = outputs
        assert left.node.target is torch.ops.prims.convert_element_type.default
        assert right.node.target is torch.ops.aten.permute.default
        return [
            replace(left, node=left.node.args[0], final_value=None),
            replace(
                right,
                node=right.node.args[0].args[0],
                coordinates=lambda row, column: (row, column),
                final_value=None,
            ),
        ], boundaries

    lines, _, _ = _capture(change=change)
    assert lines is not None
    source = "\n".join(lines)
    assert (
        source.count("CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128") == 2
    )
    assert (
        "joined_leaf_0_values = cute.make_rmem_tensor((8,), cutlass.BFloat16)" in source
    )


def test_masked_host_zero_remains_input_to_both_original_pointwise_outputs():
    lines, _, _ = _capture(final_domain=False)
    assert lines is not None
    tree = ast.parse("\n".join(lines))
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not joined_leaf_0_vectorized"
    )
    elements = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "joined_element"
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Int64=int,
            Float32=lambda x: torch.tensor(x, dtype=torch.float32).item(),
            Float16=lambda x: torch.tensor(x, dtype=torch.float16).item(),
            BFloat16=lambda x: torch.tensor(x, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(math=SimpleNamespace(exp2=lambda x, **kwargs: 2.0**x)),
        "operator": operator,
        "joined_leaf_0_vectorized": False,
        "joined_leaf_0_values": [float("nan")] * 8,
        "joined_output_0_values": [float("nan")] * 8,
        "joined_output_1_values": [float("nan")] * 8,
        "joined_row": 1,
        "joined_base": 0,
        "chain_loop_index": 0,
        **{name: 1 for name in names if name.startswith("chain_loop_capture_")},
        **{name: 0 for name in names if name.startswith("chain_origin_")},
    }
    namespace["chain_origin_0"] = 256
    exec(
        compile(
            ast.Module(body=[fallback, elements], type_ignores=[]),
            "<masked-vector-group>",
            "exec",
        ),
        namespace,
    )
    assert namespace["joined_leaf_0_values"] == [0] * 8
    assert namespace["joined_output_0_values"] == [2] * 8
    assert namespace["joined_output_1_values"] == [2] * 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("threads", [128, 384])
@pytest.mark.parametrize("tile_columns", [0, 32])
def test_actual_cute_native_copy_coordinates_with_disjoint_row_offset(
    dtype, threads, tile_columns
):
    initialized_before = torch.cuda.is_initialized()

    def offset(outputs, boundaries, plan):
        return [outputs[0], replace(outputs[1], offset=128)], boundaries

    ownership = plan_vector_ownership((128, 128), threads, tile_columns=tile_columns)
    assert ownership is not None
    lines, _, _ = _capture(
        dtype=dtype, threads=threads, change=offset, ownership=ownership
    )
    assert lines is not None
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    cute_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    with _cpu_codegen(), ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            def target(height, address):
                layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(
                        tcgen05.SmemLayoutAtomKind.K_SW128, cute_dtype
                    ),
                    (height, 128),
                    order=(0, 1),
                )
                pointer = cute.make_ptr(
                    cute_dtype, address, cute.AddressSpace.smem, assumed_align=128
                )
                return cute.make_tensor(
                    cute.recast_ptr(pointer, layout.inner, dtype=cute_dtype),
                    layout.outer,
                )

            targets = [target(128, 0), target(256, 32768)]
            coverage = [set(), set()]
            for thread in range(threads):
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "chain_thread": thread,
                    "chain_0_a": targets[0],
                    "chain_0_b": targets[1],
                }
                exec("\n".join(lines[:8]), namespace)
                for output in range(2):
                    copy = namespace[f"joined_output_{output}_copy"]
                    owned = copy.get_slice(thread).partition_D(
                        cute.domain_offset(
                            (128 * output, 0),
                            cute.make_identity_tensor((128 * (output + 1), 128)),
                        )
                    )
                    for step in range(ownership.trips):
                        row_tile, column_tile = divmod(step, ownership.column_tiles)
                        row = (
                            thread // ownership.thread_columns
                            + row_tile * ownership.thread_rows
                        )
                        base = (
                            thread % ownership.thread_columns * 8
                            + column_tile * ownership.tile_columns
                        )
                        if row >= 128:
                            continue
                        for element in range(8):
                            coord = tuple(
                                map(int, owned[None, row_tile, column_tile][element])
                            )
                            assert coord == (
                                128 * output + row,
                                base + element,
                            )
                            assert coord not in coverage[output]
                            coverage[output].add(coord)
            assert all(len(cells) == 16384 for cells in coverage)
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == initialized_before


def test_real_preparation_group_shares_original_exp_and_host_vector():
    # Import only the benchmark namespace needed by this existing fixture;
    # another installed package named benchmarks must not intercept it.
    root = Path(__file__).resolve().parents[1]
    package = ModuleType("benchmarks")
    package.__path__ = [str(root / "benchmarks")]
    with patch.dict(sys.modules, {"benchmarks": package}):
        from .test_cute_chained_preparation_cut import _kda_fixture
        from .test_cute_chained_preparation_frame import _capture as capture_plan

        kernel, args = _kda_fixture()
        captured = []
        original = stage_module.emit_stage

        def observe(cg, plan, boundaries, stage, *args, **kwargs):
            if stage == 0 and not captured:
                group = plan.contraction_groups[0]
                assert group.stages == (0, 1)
                outputs = []
                for role in ("a", "b"):
                    selected = list(
                        zip(group.stages, group.geometries, group.offsets, strict=True)
                    )
                    for member, geometry, offset in (
                        selected[:1] if role == "a" else selected
                    ):
                        arg, _ = geometry.operand(role, "r", "c")
                        node = plan.dots[member].args[arg]
                        outputs.append(
                            VectorGroupOutput(
                                node,
                                f"chain_0_{role}",
                                lambda row, column, geometry=geometry, role=role: (
                                    geometry.operand(role, row, column)[1]
                                ),
                                offset=offset if role == "b" else 0,
                                final_value=lambda value, dtype, coords, node=node: (
                                    chain._masked_operand(
                                        value,
                                        dtype,
                                        chain._operand_domain(cg, node, coords, plan),
                                    )
                                ),
                            )
                        )
                merged = emit_vector_group(
                    cg,
                    plan,
                    boundaries,
                    outputs,
                    shape=(32, 128),
                    tag="actual_group",
                    execution=ChainedExecution(384),
                )
                captured.append(merged)
            return original(cg, plan, boundaries, stage, *args, **kwargs)

        with patch.object(stage_module, "emit_stage", observe):
            capture_plan(
                kernel,
                args,
                helion.Config(
                    block_sizes=[128],
                    num_warps=16,
                    cute_chained_mma_schedule="tcgen05_tmem",
                    cute_chained_group_contractions=True,
                    cute_chained_scratch_layout="xor",
                    cute_chained_pointwise_vectorize=True,
                    cute_chained_scan_schedule="warp",
                    cute_chained_pointwise_cache_bytes=4096,
                    cute_chained_pointwise_unroll=8,
                    cute_chained_warp_mma_rows=32,
                ),
            )
    assert len(captured) == 1 and captured[0] is not None
    source = "\n".join(captured[0])
    assert source.count("_vectorized = cutlass.Boolean(False)") == 2
    assert _calls(source)["cute.math.exp2"] == 1
    assert "cute.domain_offset((32, 0), chain_0_b)" in source


@contextmanager
def _use_group_emission():
    """Test-only integration into an otherwise unchanged full loop kernel."""
    original = stage_module.emit_stage

    def replace_fills(cg, plan, boundaries, stage, *args, **kwargs):
        ordinary = []
        emit = stage_module.emit_vector_stage

        def record(*args, **kwargs):
            lines = emit(*args, **kwargs)
            ordinary.append(lines)
            return lines

        with patch.object(stage_module, "emit_vector_stage", record):
            lines = original(cg, plan, boundaries, stage, *args, **kwargs)
        assert (
            stage == 0
            and len(ordinary) == 2
            and all(fill is not None for fill in ordinary)
        )
        outputs = _outputs(plan, stage, boundaries, cg)
        merged = emit_vector_group(
            cg, plan, boundaries, outputs, shape=(128, 128), tag="joined_test"
        )
        assert merged is not None
        source = "\n".join(lines)
        # Bind both original native views before the joint fill. No barrier
        # originally exists between the two separate materializations.
        for index, fill in enumerate(ordinary):
            assert fill is not None
            old = "\n".join(fill)
            assert source.count(old) == 1
            source = source.replace(old, "" if index == 0 else "\n".join(merged), 1)
        return source.splitlines()

    with patch.object(stage_module, "emit_stage", replace_fills):
        yield


@pytest.mark.parametrize("warps", [4, 16])
def test_test_only_runtime_route_contains_group_without_new_storage_or_sync(warps):
    with _cpu_codegen():
        plain = _shared_loop._bind_isolated(_args()).to_code(_config(warps))
        with _use_group_emission():
            grouped = _shared_loop._bind_isolated(_args()).to_code(_config(warps))
    assert "joined_test_output_0" in grouped
    for name in (
        "cute.arch.alloc_smem",
        "cute.gemm",
        "cute.arch.sync_threads",
        "cute.arch.mbarrier_wait",
    ):
        assert _calls(plain)[name] == _calls(grouped)[name]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps,warps", [(0, 4), (1, 4), (3, 16)])
def test_gpu_grouped_expression_matches_original_replay_and_immutability(
    dtype, steps, warps
):
    args = _args("cuda", steps, dtype)
    saved = tuple(value.clone() for value in args[:3])
    original = _shared_loop._bind_isolated(args).compile_config(_config(warps))(*args)
    with _use_group_emission():
        grouped = _shared_loop._bind_isolated(args).compile_config(_config(warps))
    actual = grouped(*args)
    replay = grouped(*args)
    for index, (value, expected, repeated) in enumerate(
        zip(actual, original, replay, strict=True)
    ):
        if steps == 0 and index == 0:
            continue  # The unchanged source leaves zero-trip history unused.
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
        torch.testing.assert_close(repeated, value, atol=0, rtol=0)
    torch.testing.assert_close(args[:3], saved, atol=0, rtol=0)
