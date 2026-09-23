"""Execute the emitted partition coordinates and FP32 reducer on the CPU."""

from __future__ import annotations

import ast
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_split_k_workspace import _args
from test.test_cute_split_k_workspace import _embedded_sources
from test.test_cute_split_k_workspace import _fp32_bias
from test.test_cute_split_k_workspace import _proof

import helion
from helion._compiler.cute.split_k_workspace import workspace_schedule
from helion._compiler.cute.split_k_workspace_codegen import _reducer_source
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_protocol():
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matrix_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    rows, inner_size = a.shape
    columns = b.size(1)
    chunk = hl.register_tunable("tile_extent", PowerOfTwoFragment(1, 256))
    output = torch.zeros_like(bias, dtype=torch.float32)
    for row, column, partition in hl.tile(
        (rows, columns, inner_size), block_size=(None, None, chunk)
    ):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(partition.begin, partition.end):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        if partition.begin == 0:
            acc = acc + bias[row, column]
        hl.atomic_add(output, [row, column], acc)
    return output


class _Values(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(_Values)


class _MaskIfs(ast.NodeTransformer):
    """Vectorize only the emitted pure predicates, preserving nested masks."""

    def visit_BoolOp(self, node):
        assert isinstance(node.op, ast.And)
        values = [self.visit(value) for value in node.values]
        result = values[0]
        for value in values[1:]:
            result = ast.BinOp(result, ast.BitAnd(), value)
        return result

    def visit_If(self, node):
        assert not node.orelse
        body = []
        for statement in node.body:
            transformed = self.visit(statement)
            body.extend(transformed if isinstance(transformed, list) else [transformed])
        return [
            ast.Expr(
                ast.Call(ast.Name("push_mask", ast.Load()), [self.visit(node.test)], [])
            ),
            *body,
            ast.Expr(ast.Call(ast.Name("pop_mask", ast.Load()), [], [])),
        ]


def _execute_reducer(source, partial_values, output_dtype, bias_storage=None):
    """Execute the actual arithmetic/copy/shuffle AST across all 128 lanes."""
    partitions, rows, columns = partial_values.shape
    partials = partial_values.reshape(-1)
    output = np.full((rows * columns,), np.nan, dtype=output_dtype)
    reads = np.zeros(partials.size, dtype=np.int32)
    writes = np.zeros(output.size, dtype=np.int32)
    bias_reads = np.zeros(
        0 if bias_storage is None else bias_storage.size, dtype=np.int32
    )
    state = {"block": 0, "masks": [np.ones(128, dtype=np.bool_)]}

    class Fragment:
        def __init__(self, shape, dtype):
            # Inactive lanes are unspecified and never consumed by the kernel.
            # Give those lanes a neutral value in this vectorized interpreter.
            self.values = np.zeros((128, *shape), dtype=dtype)

        def fill(self, value):
            self.values.fill(value)

        def load(self):
            return self.values.copy().view(_Values)

        def store(self, values):
            active = state["masks"][-1]
            self.values[active] = values[active]

        def __getitem__(self, element):
            return self.values[:, element].copy().view(_Values)

        def __setitem__(self, element, values):
            active = state["masks"][-1]
            self.values[active, element] = values[active]

    class Pointer:
        def __init__(self, allocation, offset=0):
            self.allocation = allocation
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.allocation, self.offset + offset)

    class Tensor:
        def __init__(self, pointer, shape):
            self.iterator = pointer
            self.shape = shape

        def __getitem__(self, index):
            assert self.shape == 1 and index == 0
            assert bias_storage is not None
            pointer = self.iterator
            assert pointer.allocation is bias_storage
            indices = np.asarray(pointer.offset)
            active = state["masks"][-1]
            result = np.zeros(128, dtype=np.float32)
            assert np.all(indices[active] >= 0)
            assert np.all(indices[active] < bias_storage.size)
            result[active] = bias_storage[indices[active]]
            np.add.at(bias_reads, indices[active], 1)
            return result

    def copy_values(atom, source, destination):
        active = state["masks"][-1]
        if isinstance(destination, Fragment):
            pointer = source.iterator
            assert pointer.allocation is partials and atom == (np.float32, 128)
            indices = pointer.offset[:, None] + np.arange(4)
            assert np.all(indices[active] >= 0) and np.all(
                indices[active] < partials.size
            )
            assert np.all(pointer.offset[active] * 4 % 16 == 0)
            values = np.zeros((128, 4), dtype=np.float32)
            values[active] = partials[indices[active]]
            destination.store(values)
            np.add.at(reads, indices[active], 1)
        else:
            pointer = destination.iterator
            assert pointer.allocation is output
            assert atom == (output_dtype, np.dtype(output_dtype).itemsize * 32)
            indices = pointer.offset[:, None] + np.arange(4)
            assert np.all(indices[active] >= 0) and np.all(
                indices[active] < output.size
            )
            assert np.all(pointer.offset[active] % 4 == 0)
            output[indices[active]] = source.load()[active]
            np.add.at(writes, indices[active], 1)

    def assume(value, *, divby):
        assert np.all(value % divby == 0)
        return value

    def push_mask(predicate):
        state["masks"].append(state["masks"][-1] & predicate)

    def pop_mask():
        state["masks"].pop()

    kernel = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    )
    kernel.decorator_list = []
    kernel = _MaskIfs().visit(kernel)
    namespace: dict[str, Any] = {
        "push_mask": push_mask,
        "pop_mask": pop_mask,
        "cutlass": SimpleNamespace(
            Int32=lambda value: np.asarray(value, dtype=np.int32),
            Float32=np.float32,
            Float16=np.float16,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            assume=assume,
            make_rmem_tensor=Fragment,
            make_layout=lambda shape, **kwargs: shape,
            make_tensor=Tensor,
            make_copy_atom=lambda operation, dtype, num_bits_per_copy: (
                dtype,
                num_bits_per_copy,
            ),
            copy=copy_values,
            nvgpu=SimpleNamespace(CopyUniversalOp=lambda: None),
            arch=SimpleNamespace(
                thread_idx=lambda: (np.arange(128, dtype=np.int32), 0, 0),
                block_idx=lambda: (state["block"], 0, 0),
                shuffle_sync_bfly=lambda value, offset: value[np.arange(128) ^ offset],
            ),
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([kernel], [])),
            "<emitted-reducer>",
            "exec",
        ),
        namespace,
    )
    arguments = [
        Tensor(Pointer(partials), partial_values.shape),
        Tensor(Pointer(output), output.shape),
    ]
    if bias_storage is not None:
        arguments.append(Tensor(Pointer(bias_storage), bias_storage.shape))
    for block in range(helion.cdiv(output.size, 128)):
        state["block"] = block
        namespace[kernel.name](*arguments)
        assert len(state["masks"]) == 1
    np.testing.assert_array_equal(reads, 1)
    np.testing.assert_array_equal(writes, 1)
    if bias_storage is not None:
        assert bias_reads.sum() == rows * columns
    return output.reshape(rows, columns)


def _ordered_reference(partials, bias=None):
    sums = np.zeros((4, *partials.shape[1:]), dtype=np.float32)
    for partition, values in enumerate(partials):
        if partition == 0 and bias is not None:
            values = np.float32(values + bias)
        sums[partition % 4] = np.float32(sums[partition % 4] + values)
    return np.float32(np.float32(sums[0] + sums[1]) + np.float32(sums[2] + sums[3]))


@pytest.mark.parametrize("partitions", (1, 2, 3, 4, 5, 16, 32))
def test_emitted_reducer_reads_each_partial_once_and_writes_each_output_once(
    partitions,
):
    args = _args((128, 64 * partitions, 128))
    bound = _direct_chunk_matmul._bind_isolated(args)
    proof = _proof(bound)
    assert proof is not None
    config = helion.Config(block_sizes=[128, 128, 64], tile_extent=64)
    with bound.env:
        schedule = workspace_schedule(proof, bound.env, config, capacity_bytes=232448)
    assert schedule is not None and schedule.partitions == partitions
    values = (
        np.random.default_rng(972 + partitions)
        .standard_normal((partitions, 128, 128))
        .astype(np.float32)
    )
    actual = _execute_reducer(_reducer_source(proof, schedule), values, np.float16)
    np.testing.assert_array_equal(actual, _ordered_reference(values).astype(np.float16))


@pytest.mark.parametrize("stride", (0, 1, 3))
def test_emitted_bias_address_uses_physical_stride_once_and_adds_only_to_partition_zero(
    stride,
):
    a, b = _args((128, 192, 128))
    bias_storage = torch.empty((128 * max(1, stride),), dtype=torch.float32)
    bias = bias_storage.as_strided((128,), (stride,))
    bound = _fp32_bias._bind_isolated((a, b, bias))
    proof = _proof(bound)
    assert proof is not None and proof.bias is not None
    with bound.env:
        schedule = workspace_schedule(
            proof,
            bound.env,
            helion.Config(block_sizes=[128, 128, 64], partitions=4),
            capacity_bytes=232448,
        )
    assert schedule is not None and schedule.partitions == 3
    storage = np.full(bias_storage.numel(), -1e8, dtype=np.float32)
    partials = np.zeros((3, 128, 128), dtype=np.float32)
    partials[0].fill(1e8)
    partials[1].fill(1)
    actual = _execute_reducer(
        _reducer_source(proof, schedule), partials, np.float32, storage
    )
    np.testing.assert_array_equal(actual, np.ones((128, 128), dtype=np.float32))
    np.testing.assert_array_equal(
        actual, _ordered_reference(partials, storage[:: max(stride, 1)][:128])
    )


def test_original_zeros_like_host_factory_preserves_arguments_and_final_dtype():
    a, b = _args()
    bias = torch.empty((128, 128), dtype=torch.float32)
    bound = _matrix_bias._bind_isolated((a, b, bias))
    proof = _proof(bound)
    assert proof is not None and proof.bias is not None
    config = helion.Config(
        block_sizes=[128, 128, 64], tile_extent=128, cute_split_k_workspace=True
    )
    source = bound.to_code(config)
    module = PyCodeCache.load(source)
    calls = []
    output = module._matrix_bias(
        a, b, bias, _launcher=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert len(calls) == 2 and output.shape == bias.shape
    assert output.dtype is torch.float32 and output.data_ptr() != bias.data_ptr()
    assert "torch.empty_like(bias, dtype=torch.float32)" in source
    producer, reducer = _embedded_sources(source)
    assert "CtaGroup.TWO" in producer
    with bound.env:
        schedule = workspace_schedule(proof, bound.env, config, capacity_bytes=232448)
    assert schedule is not None
    partials = (
        np.random.default_rng(871)
        .standard_normal((schedule.partitions, 128, 128))
        .astype(np.float32)
    )
    bias_values = np.arange(128 * 128, dtype=np.float32) / 100
    actual = _execute_reducer(reducer, partials, np.float32, bias_values)
    np.testing.assert_array_equal(
        actual, _ordered_reference(partials, bias_values.reshape(128, 128))
    )


@pytest.mark.parametrize(
    "shape,partitions,tile",
    (
        ((128, 32768, 256), 32, (128, 128, 64)),
        ((512, 1024, 512), 16, (128, 128, 64)),
    ),
)
def test_actual_cute_scheduler_pairs_cover_logical_partitions_once(
    tmp_path, shape, partitions, tile
):
    import cutlass
    import cutlass.cute as cute

    from test.test_cute_collective_chunk_seeds import _partitioned_matmul

    args = _args(shape)
    bound = _partitioned_matmul._bind_isolated(args)
    config = helion.Config(
        block_sizes=list(tile), partitions=partitions, cute_split_k_workspace=True
    )
    source = bound.to_code(config)
    producer, _reducer = _embedded_sources(source)
    module = PyCodeCache.load(source)
    captures = []
    module._partitioned_matmul(
        *args,
        _launcher=lambda kernel, grid, *args, **kwargs: captures.append(
            (kernel, grid, args, kwargs)
        ),
    )
    grid = captures[0][1]
    assert grid[0:2] == (2, 1)
    m, k, n = shape
    bm, bn, bk = tile
    cluster_total = partitions * (m // bm) * (n // bn)
    assert grid[2] == min(cluster_total, 74)
    tree = ast.parse(producer)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    )
    params = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.utils.PersistentTileSchedulerParams"
    ]
    assert len(params) == 3 and len({ast.dump(node) for node in params}) == 1
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_BLOCK_SIZE_")
    ]
    assert all(isinstance(node.value, ast.Constant) for node in constants)
    rounds = helion.cdiv(cluster_total, grid[2])
    slots = grid[2] * rounds * 2
    driver = f"""\n@cute.jit
def scheduler_cpu(address: cutlass.Int64):
    output = cute.make_tensor(cute.make_ptr(cutlass.Int32, address, cute.AddressSpace.generic, assumed_align=4), cute.make_layout(({slots},4),stride=(4,1)))
    for launch_cluster in range({grid[2]}):
        for cta in range(2):
            params = {ast.unparse(params[0])}
            scheduler = cutlass.utils.StaticPersistentTileScheduler.create(params, (cta,0,launch_cluster), (cutlass.Int32({grid[0]}),cutlass.Int32({grid[1]}),cutlass.Int32({grid[2]})))
            work = scheduler.initial_work_tile_info()
            for step in range({rounds}):
                slot = (launch_cluster * {rounds} + step) * 2 + cta
                output[slot,0] = work.tile_idx[0]
                output[slot,1] = work.tile_idx[1]
                output[slot,2] = work.tile_idx[2]
                output[slot,3] = cutlass.Int32(work.is_valid_tile)
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
"""
    path = tmp_path / "scheduler_cpu.py"
    path.write_text(
        "import cutlass\nimport cutlass.cute as cute\n"
        + "\n".join(ast.unparse(node) for node in constants)
        + driver
    )
    spec = importlib.util.spec_from_file_location("split_workspace_scheduler_cpu", path)
    assert spec is not None and spec.loader is not None
    driver_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver_module
    spec.loader.exec_module(driver_module)
    observed = np.full((slots, 4), -12345, dtype=np.int32)
    with patch.object(cute.arch, "make_warp_uniform", side_effect=lambda value: value):
        compiled = cute.compile(
            driver_module.scheduler_cpu,
            cutlass.Int64(observed.ctypes.data),
            options="--gpu-arch sm_100a",
        )
    compiled(observed.ctypes.data)
    valid = observed[observed[:, 3] == 1]
    expected = np.array(
        [
            (2 * p + cta, row, column, 1)
            for column in range(n // bn)
            for row in range(m // bm)
            for p in range(partitions)
            for cta in range(2)
        ],
        dtype=np.int32,
    )
    assert sorted(map(tuple, valid)) == sorted(map(tuple, expected))
    # The native batch descriptor is [M,K,P] / [N,K,P] / [M,N,P].
    # Each CTA half writes a disjoint M interval within its own logical P.
    owners = np.zeros((partitions, m, n), dtype=np.uint8)
    for packed_partition, row_tile, column_tile, _valid in valid:
        partition, cta = divmod(int(packed_partition), 2)
        row = int(row_tile) * bm + cta * (bm // 2)
        column = int(column_tile) * bn
        owners[partition, row : row + bm // 2, column : column + bn] += 1
    np.testing.assert_array_equal(owners, 1)
    if shape == (128, 32768, 256):
        assert partitions == 32 and grid[2] == 64


@pytest.mark.parametrize("chunk,k,padded", ((64, 192, False), (128, 512, True)))
def test_generated_batch_views_preserve_every_original_partition_address(
    chunk, k, padded
):
    rows, columns = 256, 128
    a_storage = torch.empty((rows, k + (8 if padded else 0)), dtype=torch.float16)
    b_storage = torch.empty((k, columns + (8 if padded else 0)), dtype=torch.float16)
    a, b = a_storage[:, :k], b_storage[:, :columns]
    bound = _direct_chunk_matmul._bind_isolated((a, b))
    config = helion.Config(
        block_sizes=[128, 128, 64], tile_extent=chunk, cute_split_k_workspace=True
    )
    module = PyCodeCache.load(bound.to_code(config))
    calls = []
    module._direct_chunk_matmul(
        a, b, _launcher=lambda kernel, grid, *args, **kwargs: calls.append(args)
    )
    av, bv, partials = calls[0]
    partitions = k // chunk
    assert av.data_ptr() == a.data_ptr() and bv.data_ptr() == b.data_ptr()
    assert av.shape == (partitions, rows, chunk)
    assert bv.shape == (partitions, chunk, columns)
    assert partials.shape == (partitions, rows, columns)
    global_k = np.arange(partitions)[:, None] * chunk + np.arange(chunk)[None, :]
    np.testing.assert_array_equal(global_k.reshape(-1), np.arange(k))
    for partition in range(partitions):
        original_k = global_k[partition]
        a_actual = (
            av.storage_offset()
            + partition * av.stride(0)
            + np.arange(rows)[:, None] * av.stride(1)
            + np.arange(chunk)[None, :] * av.stride(2)
        )
        a_expected = (
            int(a.storage_offset())
            + np.arange(rows)[:, None] * a.stride(0)
            + original_k[None, :] * a.stride(1)
        )
        b_actual = (
            bv.storage_offset()
            + partition * bv.stride(0)
            + np.arange(chunk)[:, None] * bv.stride(1)
            + np.arange(columns)[None, :] * bv.stride(2)
        )
        b_expected = (
            b.storage_offset()
            + original_k[:, None] * b.stride(0)
            + np.arange(columns)[None, :] * b.stride(1)
        )
        np.testing.assert_array_equal(a_actual, a_expected)
        np.testing.assert_array_equal(b_actual, b_expected)


def test_column_major_and_overlapping_inputs_retain_original_lowering():
    a, b = _args()
    for rhs in (b.T.contiguous().T, b[:1].expand_as(b)):
        bound = _direct_chunk_matmul._bind_isolated((a, rhs))
        assert _proof(bound) is None
        assert not bound.config_spec.cute_split_k_workspace_available


def test_bundle_serialization_and_cleanup_keep_both_stage_sources(
    tmp_path, monkeypatch
):
    import hashlib
    from unittest.mock import Mock

    from helion.autotuner.precompile_future import SerializedCompiledFunction
    from helion.autotuner.precompile_future import _load_compiled_fn
    from helion.autotuner.precompile_future import _serialize_compiled_fn
    from helion.autotuner.precompile_future import _unload_compiled_fn

    args = _args()
    bound = _direct_chunk_matmul._bind_isolated(args)
    config = helion.Config(
        block_sizes=[128, 128, 64], tile_extent=128, cute_split_k_workspace=True
    )
    source = bound.to_code(config)
    filename = tmp_path / "split_workspace_wrapper.py"
    filename.write_text(source)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    spec = SerializedCompiledFunction(
        bound.kernel.name, source, str(filename), None, source_hash
    )
    compiled = _load_compiled_fn(spec)
    kernels = compiled.__dict__["_helion_cute_kernels"]
    assert len(kernels) == 2
    assert [kernel._helion_cute_source_hash for kernel in kernels] == [
        hashlib.sha256(stage.encode()).hexdigest()
        for stage in _embedded_sources(source)
    ]
    saved = _serialize_compiled_fn(compiled)
    assert saved.source_code == source and saved.source_hash == source_hash
    assert bound.env.backend.generated_source_hash(compiled) == source_hash
    calls = []
    output = compiled(*args, _launcher=lambda *args, **kwargs: calls.append(args))
    assert len(calls) == 2 and output.dtype is torch.float16
    for kernel in kernels:
        monkeypatch.setattr(kernel, "clear_fast_path_caches", Mock(), raising=False)
        monkeypatch.setattr(
            kernel,
            "_helion_cute_compiled_launchers",
            {0: SimpleNamespace()},
            raising=False,
        )
        monkeypatch.setattr(
            kernel, "_helion_cute_launch_arg_cache", {0: object()}, raising=False
        )
    module_name = compiled.__module__
    _unload_compiled_fn(compiled)
    assert module_name not in sys.modules
    for kernel in kernels:
        kernel.clear_fast_path_caches.assert_called_once_with()
        assert not kernel._helion_cute_compiled_launchers
        assert not kernel._helion_cute_launch_arg_cache
