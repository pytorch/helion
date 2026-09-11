"""Tests for the CuTe ``fuse_two_pass_loads`` AST pass.

When a kernel reads the same gmem tensor in two sequential inner-tile
loops over the same range, the pass detects the redundant load and
caches the first sweep's values in a small ``cute.make_rmem_tensor(...)``.
The second sweep then reads from the fragment instead of issuing a
second LDG, eliminating the duplicate HBM/L1 traffic.

The pass canonicalizes per-sweep variable names (``mask_<N>``,
``lane_base_<N>``, ``vec_lane_<N>``, ``_tile_unroll_vec_<N>_<M>``) so
the two sweeps' load expressions compare equal modulo the rename
suffix. Without canonicalization, ``vec_lane_1`` vs ``vec_lane_2``
would mis-key the fuser's match table and the cache would never fire.

Lives in ``helion/_compiler/cute/fuse_two_pass_loads.py``.
"""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


def _nested_persistent_sweeps() -> list[ast.stmt]:
    """Three guarded persistent sweeps with one later-origin load."""
    return ast.parse(
        """
if active:
    for synthetic_lane_7 in range(8):
        a0 = (x.iterator + slot + synthetic_lane_7).load()
    for synthetic_lane_7 in range(8):
        slot_copy_0 = slot
        a1 = (x.iterator + slot_copy_0 + synthetic_lane_7).load()
        b1 = (y.iterator + slot_copy_0 + synthetic_lane_7).load()
    for synthetic_lane_7 in range(8):
        slot_copy_1 = slot
        a2 = (x.iterator + slot_copy_1 + synthetic_lane_7).load()
        b2 = (y.iterator + slot_copy_1 + synthetic_lane_7).load()
"""
    ).body


def test_nested_persistent_sweeps_register_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    result = fuse_two_pass_loads(
        _nested_persistent_sweeps(),
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # Allocations stay at kernel scope rather than under the dynamic guard.
    assert len(result) == 3
    assert all(isinstance(stmt, ast.Assign) for stmt in result[:2])
    assert isinstance(result[2], ast.If)
    assert code.count("cute.make_rmem_tensor(8, cutlass.Float32)") == 2
    # A is loaded only in sweep one.  B is first loaded in sweep two, cached
    # there, and reused in sweep three despite the copy aliases.
    assert code.count(".load()") == 2
    assert "a2 = _fuse_cache_0" in code
    assert "b2 = _fuse_cache_1" in code


def test_nested_persistent_sweeps_gmem_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    result = fuse_two_pass_loads(
        _nested_persistent_sweeps(),
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: "gmem"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 5


@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
def test_persistent_sweep_cache_invalidated_by_intervening_store(
    monkeypatch: pytest.MonkeyPatch,
    prove_x_y_disjoint: bool,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
if active:
    for synthetic_lane_7 in range(8):
        a0 = (x.iterator + synthetic_lane_7).load()
        b0 = (y.iterator + synthetic_lane_7).load()
    (x.iterator + slot).store(replacement)
    for synthetic_lane_7 in range(8):
        a1 = (x.iterator + synthetic_lane_7).load()
        b1 = (y.iterator + synthetic_lane_7).load()
    for synthetic_lane_7 in range(8):
        a2 = (x.iterator + synthetic_lane_7).load()
        b2 = (y.iterator + synthetic_lane_7).load()
"""
    ).body

    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y"))} if prove_x_y_disjoint else None
        ),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # The x store always invalidates x.  A distinct argument name is not an
    # aliasing proof: y can cross the write only when the caller supplies a
    # runtime-backed disjointness fact.
    assert code.count("(x.iterator + synthetic_lane_7).load()") == 2
    assert "a1 = (x.iterator + synthetic_lane_7).load()" in code
    assert "a2 = _fuse_cache_" in code
    expected_y_loads = 1 if prove_x_y_disjoint else 2
    assert code.count("(y.iterator + synthetic_lane_7).load()") == expected_y_loads
    if prove_x_y_disjoint:
        assert "b1 = _fuse_cache_" in code
    else:
        assert "b1 = (y.iterator + synthetic_lane_7).load()" in code
    assert "b2 = _fuse_cache_" in code


def test_auto_reload_stops_at_disjoint_write_phase_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
for synthetic_lane_7 in range(8):
    before = (x.iterator + synthetic_lane_7).load()
(out.iterator + slot).store(replacement)
for synthetic_lane_7 in range(8):
    after = (x.iterator + synthetic_lane_7).load()
"""
    ).body

    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: "auto"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "_fuse_cache_" not in code
    assert code.count("(x.iterator + synthetic_lane_7).load()") == 2


@pytest.mark.parametrize(
    "atomic",
    (
        "cute.arch.atomic_add(x.iterator + slot, replacement)",
        "_cute_atomic_add(x.iterator + slot, replacement)",
        "_cute_atomic_add(opaque_ptr, replacement)",
    ),
)
def test_persistent_sweep_cache_invalidated_by_intervening_atomic(
    monkeypatch: pytest.MonkeyPatch,
    atomic: str,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        f"""
if active:
    for synthetic_lane_7 in range(8):
        before = (x.iterator + synthetic_lane_7).load()
    {atomic}
    for synthetic_lane_7 in range(8):
        after = (x.iterator + synthetic_lane_7).load()
"""
    ).body

    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "_fuse_cache_" not in code
    assert code.count("(x.iterator + synthetic_lane_7).load()") == 2


@pytest.mark.parametrize(
    ("reload_mode", "expected_loads", "expect_cache"),
    (("auto", 2, False), ("register", 1, True)),
)
def test_branch_local_exact_fragment_vector_load_store(
    monkeypatch: pytest.MonkeyPatch,
    reload_mode: str,
    expected_loads: int,
    expect_cache: bool,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
if active:
    for synthetic_lane_7 in cutlass.range_constexpr(8):
        lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
        before = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + cutlass.Int32(slot) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(lane_index) * cutlass.Int32(x.layout.stride[1]), (x.iterator + cutlass.Int32(slot) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(lane_index) * cutlass.Int32(x.layout.stride[1])).load() if active_index and lane_index < 128 else cutlass.BFloat16(0))
    (out.iterator + slot).store(result)
    for synthetic_lane_7 in cutlass.range_constexpr(8):
        slot_copy = slot
        lane_index_copy = lane_base + cutlass.Int32(synthetic_lane_7)
        after = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + cutlass.Int32(slot_copy) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(lane_index_copy) * cutlass.Int32(x.layout.stride[1]), (x.iterator + cutlass.Int32(slot_copy) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(lane_index_copy) * cutlass.Int32(x.layout.stride[1])).load() if active_index and lane_index_copy < 128 else cutlass.BFloat16(0))
        updated = after + delta
        _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', x.iterator + cutlass.Int32(slot_copy) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(lane_index_copy) * cutlass.Int32(x.layout.stride[1]), cutlass.BFloat16(updated), active_index and lane_index_copy < 128)
"""
    ).body
    fused = fuse_two_pass_loads(
        body,
        tensor_dtypes={
            "x": "cutlass.BFloat16",
            "out": "cutlass.BFloat16",
        },
        reload_modes={7: reload_mode},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    result = vectorize_branch_local_persistent_fragments(fused)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # The producer sweep snapshots all lanes before the consumer writes any of
    # them. The exact-fragment marker proves that consumer iteration i cannot
    # affect iteration j, so the final read reuses that register snapshot.
    assert code.count("cute.arch.load(x.iterator") == expected_loads
    assert ("cute.make_rmem_tensor(8, cutlass.BFloat16)" in code) is expect_cache
    assert ("after = _fuse_cache_0" in code) is expect_cache
    assert "_cute_store_u16_vec(x.iterator" in code
    assert ").store(cutlass.BFloat16(updated))" not in code
    assert "if active_index and" in code
    assert "_helion_persistent_branch_vec_" not in code


@pytest.mark.parametrize(
    "consumer",
    (
        (
            "_helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', "
            "x.iterator + lane_index + 1, updated, None)"
        ),
        "cute.arch.atomic_add(x.iterator + lane_index, updated)",
    ),
    ids=("shifted-store", "atomic"),
)
def test_inplace_snapshot_fusion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    before = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    after = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
    updated = after + delta
    {consumer}
"""
    ).body

    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.BFloat16"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert code.count("_helion_persistent_branch_vec_load") == 2
    assert "_fuse_cache_" not in code


def test_inplace_snapshot_fusion_rejects_store_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    before = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', x.iterator + lane_index, replacement, None)
    after = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
"""
    ).body

    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.BFloat16"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert code.count("_helion_persistent_branch_vec_load") == 2
    assert "_fuse_cache_" not in code


def test_branch_local_shifted_exact_markers_stay_scalar() -> None:
    body = ast.parse(
        """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    before = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
    updated = before + 1
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', x.iterator + lane_index + 1, cutlass.BFloat16(updated), None)
"""
    ).body
    result = vectorize_branch_local_persistent_fragments(body)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "cute.arch.load(x.iterator" not in code
    assert "(x.iterator + lane_index).load()" in code
    assert "_cute_store_u16_vec(x.iterator" not in code
    assert "(x.iterator + lane_index + 1).store" in code
    assert "_helion_persistent_branch_vec_" not in code


@pytest.mark.parametrize("write_kind", ("store", "atomic"))
@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
def test_branch_local_vector_load_treats_later_write_as_backedge_barrier(
    write_kind: str,
    prove_x_y_disjoint: bool,
) -> None:
    write = (
        "(y.iterator + lane_index + 1).store(replacement)"
        if write_kind == "store"
        else "cute.arch.atomic_add(y.iterator + lane_index + 1, replacement)"
    )
    body = ast.parse(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = _helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', x.iterator + lane_index, (x.iterator + lane_index).load())
    {write}
"""
    ).body
    result = vectorize_branch_local_persistent_fragments(
        body,
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y"))} if prove_x_y_disjoint else None
        ),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert code.count("cute.arch.load(x.iterator") == (1 if prove_x_y_disjoint else 0)
    assert code.count("(x.iterator + lane_index).load()") == (
        0 if prove_x_y_disjoint else 1
    )
    assert "_helion_persistent_branch_vec_" not in code


@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
def test_branch_local_vector_store_treats_earlier_load_as_backedge_barrier(
    prove_x_y_disjoint: bool,
) -> None:
    body = ast.parse(
        """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    observed = (x.iterator + lane_index - 1).load() if synthetic_lane_7 > 0 else cutlass.BFloat16(0)
    replacement = observed + 1
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', y.iterator + lane_index, cutlass.BFloat16(replacement), None)
"""
    ).body
    result = vectorize_branch_local_persistent_fragments(
        body,
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y"))} if prove_x_y_disjoint else None
        ),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert ("_cute_store_u16_vec(y.iterator" in code) is prove_x_y_disjoint
    assert (").store(cutlass.BFloat16(replacement))" in code) is not prove_x_y_disjoint
    assert "_helion_persistent_branch_vec_" not in code


@pytest.mark.parametrize("write_kind", ("store", "atomic"))
@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
def test_branch_local_vector_store_treats_other_write_as_backedge_barrier(
    write_kind: str,
    prove_x_y_disjoint: bool,
) -> None:
    other_write = (
        "(x.iterator + lane_index + 1).store(replacement)"
        if write_kind == "store"
        else "cute.arch.atomic_add(x.iterator + lane_index + 1, replacement)"
    )
    body = ast.parse(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    replacement = cutlass.BFloat16(synthetic_lane_7)
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', y.iterator + lane_index, replacement, None)
    {other_write}
"""
    ).body
    result = vectorize_branch_local_persistent_fragments(
        body,
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y"))} if prove_x_y_disjoint else None
        ),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert ("_cute_store_u16_vec(y.iterator" in code) is prove_x_y_disjoint
    assert (
        "(y.iterator + lane_index).store(replacement)" in code
    ) is not prove_x_y_disjoint
    assert "_helion_persistent_branch_vec_" not in code


def test_non_copy_rename_is_not_address_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    body = ast.parse(
        """
for synthetic_lane_7 in range(8):
    before = (x.iterator + index + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    index_1 = index + 1
    after = (x.iterator + index_1 + synthetic_lane_7).load()
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.parametrize("prove_x_y_disjoint", (False, True))
@pytest.mark.parametrize("write_kind", ("store", "atomic"))
def test_consumer_loop_write_is_a_backedge_barrier(
    monkeypatch: pytest.MonkeyPatch,
    prove_x_y_disjoint: bool,
    write_kind: str,
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    write = (
        "(y.iterator + synthetic_lane_7).store(after)"
        if write_kind == "store"
        else "cute.arch.atomic_add(y.iterator + synthetic_lane_7, after)"
    )
    body = ast.parse(
        f"""
for synthetic_lane_7 in range(8):
    before = (x.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    after = (x.iterator + synthetic_lane_7).load()
    {write}
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y"))} if prove_x_y_disjoint else None
        ),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # The consumer's store executes before its next loop iteration.  Reusing
    # the producer cache is therefore legal only with an explicit x/y
    # disjointness proof.
    assert code.count("(x.iterator + synthetic_lane_7).load()") == (
        1 if prove_x_y_disjoint else 2
    )


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@onlyBackends(["cute"])
class TestCuteFuseTwoPassLoads(TestCase):
    def test_fuser_fires_with_vec_hoist(self) -> None:
        """When the vec hoist runs, the two sweeps' loads use names like
        ``lane_base_1`` vs ``lane_base_2``, ``vec_lane_1`` vs ``vec_lane_2``,
        and ``_tile_unroll_vec_1_0`` vs ``_tile_unroll_vec_1_1``. The
        alias map canonicalizes these so the fuser matches and emits one
        ``_fuse_cache_*`` fragment instead of re-reading from gmem.

        The cache_size cap is 64; for inner-tile trip=8 and lane_trip=1
        the cache_size=8 fits.
        """
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Cache fragment allocated.
        self.assertIn("_fuse_cache_", code)
        # When the load-pipeline pass is OFF the consume sweep just
        # reads from cache so the kernel has 1 ``cute.arch.load`` total.
        # With pipelining ON the reduce sweep's load is hoisted into a
        # prologue + per-iter prefetch (2 load sites), so the total is
        # 2. The consume sweep emits NO load (cache hit) either way.
        load_count = code.count("cute.arch.load(")
        self.assertTrue(
            load_count in {1, 2},
            f"expected 1 or 2 cute.arch.load sites, got {load_count}",
        )
        # The consume sweep (second top-level ``for tile_offset`` loop)
        # must not contain a gmem load.
        consume_marker = "for tile_offset_2 in range"
        consume_start = code.rfind(consume_marker)
        self.assertGreater(consume_start, 0)
        self.assertNotIn(
            "cute.arch.load(",
            code[consume_start:],
            "consume sweep must read from _fuse_cache_, not gmem",
        )

    def test_fuser_skips_when_cache_size_too_large(self) -> None:
        """The fuser caps cache_size at 64 to avoid the register-pressure
        regression measured earlier when the per-thread fragment grew
        beyond the register budget. So for trip > 64 the consume sweep
        still loads from gmem.

        For (4096, 12672) with block_size 128, trip = 99, V = 4, so the
        cache_size would be 99 — fuser must bail.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Both sweeps load from gmem (cache fragment NOT allocated since
        # the fuser bails on cache_size > 64).
        self.assertNotIn("_fuse_cache_", code)
        # When the load-pipeline pass is OFF the kernel has exactly
        # 2 ``cute.arch.load`` calls (one per sweep).  With pipelining
        # ON each load site is hoisted into a prologue and a per-iter
        # prefetch, so the total is 4.  Both forms are correct; assert
        # at least 2.
        self.assertGreaterEqual(code.count("cute.arch.load("), 2)
        # Each sweep must contain at least one gmem load.
        first_sweep_marker = "for tile_offset_2 in range"
        first_sweep_start = code.find(first_sweep_marker)
        second_sweep_start = code.find(first_sweep_marker, first_sweep_start + 1)
        self.assertGreater(second_sweep_start, first_sweep_start)
        first_sweep = code[first_sweep_start:second_sweep_start]
        second_sweep = code[second_sweep_start:]
        self.assertIn("cute.arch.load(", first_sweep)
        self.assertIn("cute.arch.load(", second_sweep)
