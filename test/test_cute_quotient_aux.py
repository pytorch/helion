from __future__ import annotations

import pytest
import sympy
import torch

import helion
from helion._compiler.cute.quotient_aux import prove_quotient_aux_load_map
from helion._compiler.cute.quotient_aux import quotient_aux_load_map
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quotient_cols(a, b, aux, first, second, c: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]), x.index.reshape([1, x]) // c]
        second[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quotient_rows(a, b, aux, first, second, c: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]) // c, x.index.reshape([1, x])]
        second[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _quotient_rank3(a, b, aux, out):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        v = aux[0, y.index.reshape([y, 1]), x.index.reshape([1, x]) // 16]
        out[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


def _args(c=16, axis=1, dtype=torch.float32):
    m, n, k = 128, 64, 64
    shape = (m, (n - 1) // c + 1) if axis == 1 else ((m - 1) // c + 1, n)
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty(shape, dtype=dtype),
        torch.empty((m, n)),
        torch.empty((m, n)),
        c,
    )


def _mapping(c=16, axis=1, rank=2, padded=False, offset=0):
    row, col = sympy.symbols("row col", integer=True, nonnegative=True)
    extents = (128, 64)
    shape = (128, (63 // c) + 1) if axis == 1 else ((127 // c) + 1, 64)
    indices = [row, col]
    indices[axis] = sympy.floor(indices[axis] / c)
    strides = (shape[1] + (7 if padded else 0), 1)
    if rank == 3:
        indices.insert(0, sympy.Integer(0))
        strides = (shape[0] * strides[0], *strides)
        shape = (2, *shape)
    span = offset + sum((n - 1) * s for n, s in zip(shape, strides, strict=True)) + 1
    return prove_quotient_aux_load_map(
        indices,
        (row, col),
        extents,
        shape,
        strides,
        storage_offset=offset,
        storage_elements=span,
    )


@pytest.mark.parametrize("c", (1, 2, 3, 5, 8, 16, 31, 32, 128))
@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("rank", (2, 3))
def test_quotient_read_bounds_no_bijection_or_dense_requirement(c, axis, rank):
    mapping = _mapping(c, axis, rank, padded=True, offset=1)
    assert mapping is not None
    for y in range(128):
        for x in range(64):
            coords = [y, x]
            coords[axis] //= c
            assert mapping.offset(y, x) == sum(
                i * mapping.strides[a]
                for i, a in zip(coords, mapping.source_axes, strict=True)
            )
    if c > 1:
        assert mapping.shape[-1] * mapping.shape[-2] != 128 * 64


@pytest.mark.parametrize("c", (3, 5, 31))
def test_valid_quotient_math_can_lack_regular_transport(c):
    mapping = _mapping(c)
    assert mapping is not None
    reason = mapping.tile_reason(64, 32)
    assert reason is not None and "valid quotient read" in reason


@pytest.mark.parametrize("blocks", ((63, 32), (64, 24)))
def test_quotient_transport_refuses_partial_tiles(blocks):
    mapping = _mapping()
    assert mapping is not None
    reason = mapping.tile_reason(*blocks)
    assert reason is not None and "complete M/N" in reason


@pytest.mark.parametrize("bad", (0, -1, 16.0, True, 2**31))
def test_typed_quotient_divisor_refusal(bad):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        with bound.env, bound.host_function:
            load = next(
                n
                for gi in bound.host_function.device_ir.graphs
                for n in gi.graph.nodes
                if n.target is hl.load and n.args[0].meta.get("val").shape == (128, 4)
            )
            store = next(n for n in load.graph.nodes if n.target is hl.store)
            carriers = tuple(store.args[1])
            options = {
                "carrier_tile_shape": tuple(load.meta["val"].shape),
                "carrier_tile_index_nodes": carriers,
                "carrier_global_shape": (128, 64),
            }
            assert quotient_aux_load_map(load, **options) is not None
            quotient = load.args[1][1]
            quotient.args = (quotient.args[0], bad)
            assert quotient_aux_load_map(load, **options) is None


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("c", (1, 2, 8, 16, 32, 128))
def test_quotient_scalar_codegen_admits_regular_tiles(axis, c):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols if axis else _quotient_rows, _args(c, axis))
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(helion.Config(block_sizes=[64, 32, 64], pid_type="flat"))
        assert "tcgen05_ab_tma" in source
        assert "assumed_align=4" in source
        assert "num_bits_per_copy=32" in source
        assert (
            "cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32)"
            in source
        )
        assert ".layout.stride[" in source


def test_rank3_fixed_plane_uses_original_pointer_and_strides():
    a, b, aux, _first, out, _c = _args()
    aux = torch.empty((2, 128, 4))
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_rank3, (a, b, aux, out))
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(helion.Config(block_sizes=[64, 32, 64], pid_type="flat"))
        assert "aux.layout.stride[1]" in source
        assert "aux.layout.stride[2]" in source
        assert "num_bits_per_copy=32" in source


@pytest.mark.parametrize("c", (3, 5, 31))
def test_untransported_nonpower_quotient_keeps_original_fallback(c):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args(c))
        assert not bound.config_spec.cute_tcgen05_search_enabled


def test_aliasing_writer_not_proved_by_fake_storage_inequality():
    a, b, _aux, first, second, c = _args()
    aux = torch.as_strided(first, (128, 4), (64, 1))
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, (a, b, aux, first, second, c))
        assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("mode", ("tma", "epi_tma"))
def test_no_quotient_TMA_ring_is_registered_or_requested(mode):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        with pytest.raises(helion.exc.InvalidConfig):
            bound.to_code(
                helion.Config(
                    block_sizes=[64, 32, 64],
                    pid_type="flat",
                    tcgen05_aux_load_mode=mode,
                )
            )


@pytest.mark.parametrize(
    "extra",
    (
        {"tcgen05_aux_fragment_hoist": True},
        {"tcgen05_epi_warp_groups": 2},
        {"tcgen05_aux_prefetch_subtiles": 1},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_epi_subtile_n": 32},
    ),
)
def test_unsupported_quotient_protocol_requests_fail_closed(extra):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
            bound.to_code(
                helion.Config(block_sizes=[64, 32, 64], pid_type="flat", **extra)
            )
