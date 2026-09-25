from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import sympy
import torch

from test.test_cute_chained_initialized_accumulator import _args as pair_args
from test.test_cute_chained_initialized_accumulator import _cpu
from test.test_cute_chained_initialized_accumulator import _pair as pair_kernel

import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_result_transport import _index
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_result_transport import prove_layout
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
tcgen05 = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
blackwell_helpers = importlib.import_module("cutlass.utils.blackwell_helpers")

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_direct_output"
# The MLIR extension's Python re-export omits these native types from its stub.
# Import the real module dynamically, without suppressing product diagnostics.
ir = importlib.import_module("cutlass._mlir.ir")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        weighted = (b[kk, col].float() * 0.75).to(b.dtype)
        out[row, col] = hl.dot(a[row, kk], weighted) + 0.125
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched(
    a: torch.Tensor, b: torch.Tensor, coefficient: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, m, k = a.shape
    n = b.size(2)
    out = torch.empty((batch, m, n), dtype=torch.float32, device=a.device)
    terminal = torch.empty((batch,), dtype=torch.float32, device=a.device)
    for bi, row, col in hl.tile([batch, m, n], block_size=[1, None, n]):
        kk = hl.arange(k)
        prefix = hl.cumsum(coefficient[kk].float(), dim=0)
        weighted = (b[bi.begin, kk, col].float() * prefix[:, None]).to(a.dtype)
        value = hl.dot(a[bi.begin, row, kk], weighted) + prefix[k - 1]
        out[bi.begin, row, col] = value
        row_begin: Any = row.begin
        col_begin: Any = col.begin
        hl.store(
            terminal,
            [bi.begin],
            prefix[k - 1],
            extra_mask=(row_begin == 0) & (col_begin == 0),
        )
    return out, terminal


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_narrow(
    a: torch.Tensor, b: torch.Tensor, coefficient: torch.Tensor
) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.size(2)
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    for bi, row, col in hl.tile([batch, m, n], block_size=[1, None, n]):
        kk = hl.arange(k)
        prefix = hl.cumsum(coefficient[kk].float(), dim=0)
        weighted = (b[bi.begin, kk, col].float() * prefix[:, None]).to(a.dtype)
        out[bi.begin, row, col] = hl.dot(a[bi.begin, row, kk], weighted) + prefix[k - 1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _colliding(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for bi, row, col in hl.tile([batch, m, n], block_size=[1, None, n]):
        kk = hl.arange(k)
        out[row, col] = hl.dot(
            a[bi.begin, row, kk], (b[kk, col].float() * 0.5).to(b.dtype)
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        out[row, col] = hl.dot(a[row, kk], (b[kk, col].float() * 0.5).to(b.dtype))
    return out


def _config(rows: int = 64, **extra: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [rows],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        }
        | extra
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_m64_shared_sparse_load(dtype: torch.dtype, n: int) -> None:
    before = torch.cuda.is_initialized()
    with _cpu(), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")):
        bound = _single._bind_isolated(
            (torch.empty((64, 128), dtype=dtype), torch.empty((128, n), dtype=dtype))
        )
        source = bound.to_code(_config())
        assert source == bound.to_code(_config(**{KEY: False}))
    assert load_operation((64, n)) in source
    assert "(64, " + str(n) + ")" in source
    assert f"chain_allocator.allocate({max(32, 1 << (n - 1).bit_length())})" in source
    assert "chain_output_ptr = cute.arch.alloc_smem" in source
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_m64_direct_typed_values_and_fallback(dtype: torch.dtype, n: int) -> None:
    with _cpu(), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")):
        bound = _single._bind_isolated(
            (torch.empty((128, 128), dtype=dtype), torch.empty((128, n), dtype=dtype))
        )
        source = bound.to_code(_config(**{KEY: True}))
    assert load_operation((64, n)) in source
    assert "chain_output" not in source
    assert "cute.autovec_copy(chain_epi_values, chain_direct_target)" in source
    assert "= chain_epi_values[chain_direct_index]" in source
    assert "chain_direct_pointer.toint() % 16 == 0" in source


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("narrow", (False, True))
@pytest.mark.parametrize("n", (32, 96, 256))
def test_batched_scan_export_typed_epilogue(
    dtype: torch.dtype, narrow: bool, n: int
) -> None:
    plans = []
    original = chained_tcgen05.supported_plan

    def record(plan):
        plans.append(
            (
                plan,
                chained_tcgen05._shared_memory_bytes(plan),
                chained_tcgen05._shared_memory_bytes(
                    dataclasses.replace(plan, direct_output=False)
                ),
            )
        )
        return original(plan)

    with _cpu(), patch.object(chained_tcgen05, "supported_plan", side_effect=record):
        bound = (_batched_narrow if narrow else _batched)._bind_isolated(
            (
                torch.empty((3, 128, 128), dtype=dtype),
                torch.empty((3, 128, n), dtype=dtype),
                torch.empty((128,), dtype=torch.float32),
            )
        )
        shared = bound.to_code(_config())
        direct = bound.to_code(_config(**{KEY: True}))
    plan, actual_bytes, legacy_bytes = plans[-1]
    assert plan.direct_output and len(plan.scan_exports) == (0 if narrow else 1)
    assert legacy_bytes - actual_bytes == 64 * n * (2 if narrow else 4)
    for source in (shared, direct):
        assert "fence_view_async_tmem_load" in source
        assert source.count("chain_allocator.free(") == 1
        assert "chain_scan" in source
    expected_dtype = "BFloat16" if dtype == torch.bfloat16 else "Float16"
    expected_dtype = expected_dtype if narrow else "Float32"
    assert (
        f"chain_epi_values = cute.make_rmem_tensor(chain_0_coords.shape, cutlass.{expected_dtype})"
        in direct
    )
    assert "= chain_epi_values[chain_direct_index]" in direct
    assert "chain_output" not in direct
    ast.parse(direct)


@pytest.mark.parametrize("bad", (0, 1, None, "true", [], {}))
def test_direct_strict_bool(bad: object) -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((64, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        with pytest.raises(exc.InvalidConfig, match="must be bool"):
            bound.to_code(_config(64, **{KEY: bad}))


@pytest.mark.parametrize(
    "rows,n,warps",
    ((32, 64, 4), (64, 160, 4), (64, 192, 4), (64, 224, 4), (64, 64, 8), (64, 64, 2)),
)
def test_unsupported_geometry_fails_closed(rows: int, n: int, warps: int) -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((rows, 128), dtype=torch.bfloat16),
                torch.empty((128, n), dtype=torch.bfloat16),
            )
        )
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config(rows, num_warps=warps, **{KEY: True}))


@pytest.mark.parametrize("mode", ("coalesced", "cp_async", "cp_async_register"))
def test_direct_rejects_other_schedules(mode: str) -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((64, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        with pytest.raises(exc.InvalidConfig, match="resident one-dot M64"):
            bound.to_code(_config(cute_chained_mma_schedule=mode, **{KEY: True}))


def test_colliding_cta_map_and_external_alias_reject() -> None:
    with _cpu():
        dtype = torch.bfloat16
        b = torch.empty((128, 64), dtype=dtype)
        bound = _colliding._bind_isolated((torch.empty((2, 64, 128), dtype=dtype), b))
        assert "chain_output" in bound.to_code(_config())
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config(**{KEY: True}))
        a = torch.empty((64, 128), dtype=dtype)
        bound = _external._bind_isolated((a, b, a[:, :64]))
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config(**{KEY: True}))


@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_full_grid_span_proof_and_padding(n: int) -> None:
    row, col, batch, ro = sympy.symbols("r c b ro", integer=True)
    for stride in (n, n + 16):
        layout = prove_layout(
            (batch, row + ro, col),
            (3, 128, n),
            (128 * stride, stride, 1),
            row,
            col,
            (64, n),
            ((batch, 3, 1), (ro, 128, 64)),
            3 * 128 * stride,
            32,
        )
        assert layout.row_stride == stride
        for storage in (1, 128 * stride):
            with pytest.raises(chain._UnsupportedChain, match="storage span"):
                prove_layout(
                    (batch, row + ro, col),
                    (3, 128, n),
                    (128 * stride, stride, 1),
                    row,
                    col,
                    (64, n),
                    ((batch, 3, 1), (ro, 128, 64)),
                    storage,
                    32,
                )
    with pytest.raises(chain._UnsupportedChain, match="overlapping"):
        prove_layout(
            (row, col), (64, n), (n, 1), row, col, (64, n), ((batch, 3, 1),), 64 * n, 32
        )
    with pytest.raises(chain._UnsupportedChain):
        prove_layout((row, col), (64, n), (n - 1, 1), row, col, (64, n), (), 64 * n, 32)


@pytest.mark.parametrize(
    "text",
    (
        "r * 2147483648 - r * 2147483648",
        "cutlass.Int32(r) * 2147483648",
        "r % 64",
        "r // 2",
        "data[r]",
        "r * r",
    ),
)
def test_index_uncertainty_and_overflow_reject(text: str) -> None:
    row = sympy.Symbol("r", integer=True)
    with pytest.raises(chain._UnsupportedChain):
        _index(text, {}, {"r": row}, {row: (0, 63)}, 32)


def test_lossless_cast_and_nonzero_origins() -> None:
    row, origin = sympy.symbols("r origin", integer=True)
    actual = _index(
        "cutlass.Int32(r + origin)",
        {},
        {"r": row, "origin": origin},
        {row: (0, 63), origin: (0, 64)},
        32,
    )
    assert sympy.expand(actual - row - origin) == 0


@pytest.mark.parametrize("dtype", (cutlass.BFloat16, cutlass.Float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_actual_sparse_tmem_and_final_store_ownership(dtype: type, n: int) -> None:
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.MN,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (64, n),
                tcgen05.OperandSource.SMEM,
            )
            fragment = mma.make_fragment_C(mma.partition_shape_C((64, n))).layout
            acc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                fragment,
            )
            operation = eval(load_operation((64, n)), {"tcgen05": tcgen05})
            copy = tcgen05.make_tmem_copy(
                cute.make_copy_atom(operation, cutlass.Float32), acc
            )
            identity = mma.get_slice(0).partition_C(cute.make_identity_tensor((64, n)))
            ownership = []
            for target_dtype in (dtype, cutlass.Float32):
                bits = 32 if target_dtype is cutlass.Float32 else 16
                width_bytes = n * bits // 8
                atom_bytes = min(128, width_bytes & -width_bytes)
                shared_layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(
                        getattr(tcgen05.SmemLayoutAtomKind, f"K_SW{atom_bytes}"),
                        target_dtype,
                    ),
                    (64, n),
                    order=(0, 1),
                )
                shared = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            target_dtype, 0, cute.AddressSpace.smem, assumed_align=128
                        ),
                        shared_layout.inner,
                        dtype=target_dtype,
                    ),
                    shared_layout.outer,
                )
                shared_c = mma.get_slice(0).partition_C(shared)
                target = cute.make_tensor(
                    cute.make_ptr(
                        target_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout((64, n), stride=(n, 1)),
                )
                tensor = mma.get_slice(0).partition_C(target)
                for lane in range(128):
                    thread = copy.get_slice(lane)
                    coords = thread.partition_D(identity)
                    destination = thread.partition_D(tensor)
                    values = cute.make_tensor(
                        cute.make_ptr(
                            target_dtype, 0, cute.AddressSpace.rmem, assumed_align=16
                        ),
                        cute.make_layout(coords.shape),
                    )
                    assert cute.size(coords) == cute.size(destination) == 64 * n // 128
                    assert (
                        cute.max_common_vector(values.layout, destination.layout) == 2
                    )
                    cute.autovec_copy(values, destination)
                    shared_target = thread.partition_D(shared_c)
                    assert cute.size(shared_target) == cute.size(values)
                    cute.autovec_copy(values, shared_target)
                    if target_dtype is cutlass.Float32:
                        ownership.extend(
                            tuple(int(x) for x in coords[slot])
                            for slot in range(cute.size(coords))
                        )
            assert Counter(ownership) == Counter(
                (r, c) for r in range(64) for c in range(n)
            )
            addresses = [int(fragment(i)) for i in range(cute.size(fragment))]
            assert len(set(addresses)) == 64 * n
            assert {address % 65536 for address in addresses} == set(range(n))
            assert {address // 65536 for address in addresses} == {
                warp * 32 + row for warp in range(4) for row in range(16)
            }
    assert torch.cuda.is_initialized() == before


def test_actual_seed_prefix_contains_admitted_shared_direct_and_early() -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((64, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            count = min(100, len(list(generation.seed_flat_config_pairs())))
            population = [
                generation.unflatten(row)
                for row in generation.random_population_flat(count)
            ]
        selected = {}
        for index, config in enumerate(population):
            if config.config.get(
                "cute_chained_mma_schedule"
            ) == "tcgen05_tmem" and not config.config.get(
                "cute_chained_pointwise_read_cache"
            ):
                name = (
                    "direct"
                    if config.config.get(KEY)
                    else "early"
                    if config.config.get("cute_chained_tmem_early_release")
                    else "shared"
                )
                selected.setdefault(name, (index, config))
        assert set(selected) == {"shared", "direct", "early"}
        for index, config in selected.values():
            assert index < 100
            assert load_operation((64, 64)) in bound.to_code(config)


def test_direct_does_not_spend_freed_output_bytes_on_auxiliary_cache() -> None:
    original = chained_tcgen05.make_early_auxiliary_cache
    capacities = []

    def record(cg, plan, scans, available):
        capacities.append((plan.direct_output, available))
        return original(cg, plan, scans, available)

    with (
        _cpu(),
        patch.object(chained_tcgen05, "make_early_auxiliary_cache", side_effect=record),
    ):
        bound = _single._bind_isolated(
            (
                torch.empty((64, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        for direct in (False, True):
            bound.to_code(
                _config(**{KEY: direct, "cute_chained_auxiliary_cache": True})
            )
    assert capacities[0][0] is False and capacities[1][0] is True
    assert capacities[0][1] == capacities[1][1]


@pytest.mark.parametrize(
    "extra",
    (
        {"cute_chained_initialized_accumulator": True},
        {"cute_chained_late_rhs_reuse": True},
    ),
)
def test_m64_does_not_admit_pair_transformations(extra: dict[str, bool]) -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((64, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config(**{KEY: True, **extra}))


@pytest.mark.parametrize("direct", (False, True))
def test_m64_multi_dot_rejects(direct: bool) -> None:
    with _cpu():
        bound = pair_kernel._bind_isolated(pair_args())
        config = helion.Config(
            block_sizes=[64, 64],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_direct_output=direct,
        )
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(config)


def test_direct_m128_is_explicitly_rejected() -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((128, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        assert "Ld32x32bOp" in bound.to_code(_config(128))
        with pytest.raises(exc.BackendUnsupported, match="one-dot M64"):
            bound.to_code(_config(128, **{KEY: True}))


@pytest.mark.parametrize("m", (0, 63, 65, 96))
def test_m64_full_root_tail_requirement(m: int) -> None:
    with _cpu():
        bound = _single._bind_isolated(
            (
                torch.empty((m, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
            )
        )
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config())


@pytest.mark.parametrize("strides", ((0, 1), (-64, 1), (1, 64), (64, 0)))
def test_direct_unproved_physical_layout_rejects(strides: tuple[int, int]) -> None:
    row, col = sympy.symbols("r c", integer=True)
    with pytest.raises(chain._UnsupportedChain):
        prove_layout((row, col), (64, 64), strides, row, col, (64, 64), (), 4096, 32)


def test_narrow_index_overflow_not_cancelled_in_int64_environment() -> None:
    row = sympy.Symbol("r", integer=True)
    with pytest.raises(chain._UnsupportedChain, match="overflow"):
        _index(
            "cutlass.Int32(r) * 2147483648 - cutlass.Int32(r) * 2147483648",
            {},
            {"r": row},
            {row: (0, 63)},
            64,
        )
