"""Tests for the CuTe ``fuse_two_pass_loads`` AST pass.

When a kernel reads the same gmem tensor in two sequential inner-tile
loops over the same range, the pass detects the redundant load and
caches the first sweep's values in a small ``cute.make_rmem_tensor(...)``.
The second sweep then reads from the fragment instead of issuing a
second LDG, eliminating the duplicate HBM/L1 traffic.

The two sweeps spell their loads with per-sweep names (``mask_<N>``,
``lane_base_<N>``, ``vec_lane_<N>``). A load key spells a name the sweep
assigns by its definition, and the pass renames the later sweep's
vector-lane loop variable to the first sweep's, so the two sweeps' load
expressions compare equal. Without the rename, ``vec_lane_1`` vs
``vec_lane_2`` would mis-key the fuser's match table and the cache would
never fire.

Lives in ``helion/_compiler/cute/fuse_two_pass_loads.py``.
"""

from __future__ import annotations

import ast
import re

import pytest
import torch

from test._cute_kernels import _reduction_kernel

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
from helion._testing.cute_sweep_kernels import center_fp8_row
from helion._testing.cute_sweep_kernels import center_fp8_row_config
from helion._testing.cute_sweep_kernels import dropout_layernorm
from helion._testing.cute_sweep_kernels import scaled_workspace_config
from helion._testing.cute_sweep_kernels import scaled_workspace_reload
from helion._testing.cute_sweep_kernels import workspace_chunk_config
from helion._testing.cute_sweep_kernels import workspace_reload

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


# A word0 dropout mask drawn once per row and needed by all three sweeps
# (``test_cute_sweep_values.py`` checks the codegen on CPU).
_dropout_layernorm = helion.kernel(
    backend="cute", static_shapes=True, cute_rng_stream="word0"
)(dropout_layernorm)


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


def _fused(source: str) -> str:
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


@pytest.mark.cpu_only
def test_only_a_load_of_one_element_takes_a_single_slot() -> None:
    # ``s`` reads one element in every lane: one slot, written in the first
    # sweep and read in the second.  ``v`` reads through ``t``, which the lane
    # index defines, and ``w`` through the lane index itself: a slot per lane
    # each, indexed by the lane.
    code = _fused(
        """
for synthetic_lane_7 in range(8):
    t = synthetic_lane_7 * 2
    s = (x.iterator + 3).load()
    v = (x.iterator + t).load()
    w = (x.iterator + synthetic_lane_7).load()
    acc = acc + s * v * w
for synthetic_lane_7 in range(8):
    t = synthetic_lane_7 * 2
    s = (x.iterator + 3).load()
    v = (x.iterator + t).load()
    w = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(s * v * w)
"""
    )
    assert "_fuse_cache_0 = cute.make_rmem_tensor(1, cutlass.Float32)" in code
    assert "_fuse_cache_0[0] = s" in code
    assert "s = _fuse_cache_0[0]" in code
    for cache, name in (("_fuse_cache_1", "v"), ("_fuse_cache_2", "w")):
        assert f"{cache} = cute.make_rmem_tensor(8, cutlass.Float32)" in code
        assert f"{name} = {cache}[(synthetic_lane_7 - 0) // 1]" in code


@pytest.mark.cpu_only
@pytest.mark.parametrize("per_lane", (False, True))
def test_smem_mode_syncs_only_for_a_cache_in_shared_memory(
    monkeypatch: pytest.MonkeyPatch, per_lane: bool
) -> None:
    # ``HELION_FUSER_MODE=smem`` puts a per-lane cache in shared memory, which
    # the sweeps share through a barrier; a one-element load's single slot
    # stays in registers and needs none.
    monkeypatch.setenv("HELION_FUSER_MODE", "smem")
    index = "synthetic_lane_7" if per_lane else "3"
    code = _fused(
        f"""
for synthetic_lane_7 in range(8):
    v = (x.iterator + {index}).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    v = (x.iterator + {index}).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert ("cute.make_rmem_tensor(1, cutlass.Float32)" in code) is not per_lane
    assert ("cute.arch.alloc_smem(" in code) is per_lane
    assert ("cute.arch.sync_threads()" in code) is per_lane


@pytest.mark.cpu_only
def test_load_through_a_carried_index_is_not_fused() -> None:
    # ``idx`` is carried from one iteration to the next: ``x[idx]`` is not
    # ``x[0]``, the address ``idx = 0`` gives it in the first iteration
    # only, so the second sweep must not read the cache of ``a = x[0]``.
    code = _fused(
        """
idx = 0
for synthetic_lane_7 in range(8):
    b = (x.iterator + idx).load()
    a = (x.iterator + 0).load()
    acc = acc + a + b
    idx = idx + 1
idx = 0
for synthetic_lane_7 in range(8):
    c = (x.iterator + idx).load()
    (out.iterator + synthetic_lane_7).store(c)
    idx = idx + 1
"""
    )
    assert "c = (x.iterator + idx).load()" in code


@pytest.mark.cpu_only
def test_load_through_a_name_the_sweep_redefines_is_keyed_by_it() -> None:
    # Each sweep iteration redefines ``base`` before its lanes, so the
    # ``base = 0`` before the sweep is not the address of ``v``: ``v`` and
    # ``w`` read different elements, and the second sweep's ``v`` reads the
    # cache of the first sweep's ``v``.
    code = _fused(
        """
base = 0
for roffset_7 in range(0, 64, 32):
    base = roffset_7 * 1
    for reduction_lane_7 in range(8):
        v = (x.iterator + base + reduction_lane_7).load()
        w = (x.iterator + 0 + reduction_lane_7).load()
        acc = acc + v + w
base = 0
for roffset_7 in range(0, 64, 32):
    base = roffset_7 * 1
    for reduction_lane_7 in range(8):
        v = (x.iterator + base + reduction_lane_7).load()
        (out.iterator + base + reduction_lane_7).store(v)
"""
    )
    first, second = code[: code.rindex("base = 0")], code[code.rindex("base = 0") :]
    match = re.search(r"v = (_fuse_cache_\d+)\[", second)
    assert match is not None
    # Only the first sweep's ``v`` fills the cache the second sweep reads.
    assert re.findall(rf"{match.group(1)}\[.*\] = (\w+)$", first, re.MULTILINE) == ["v"]


@pytest.mark.cpu_only
def test_load_through_a_definition_of_a_rewritten_name_is_keyed_by_it() -> None:
    # ``c = k * 2`` holds the ``k`` from before the loop, not the ``k`` the
    # loop rebinds: ``x[c]`` is not ``x[lane * 2]``.  Each sweep reads its
    # own ``c``, rebound between them, so they read different elements.
    code = _fused(
        """
k = 0
c = k * 2
for synthetic_lane_7 in range(8):
    k = synthetic_lane_7
    v = (x.iterator + c).load()
    acc = acc + v
c = k * 2
for synthetic_lane_7 in range(8):
    k = synthetic_lane_7
    v = (x.iterator + c).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert "_fuse_cache_" not in code
    assert code.count("v = (x.iterator + c).load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("rebind", (False, True))
def test_load_through_a_name_one_sweep_defines_is_not_fused(rebind: bool) -> None:
    # The first sweep reads ``x[t]`` at each lane's own ``t``; the second
    # reads the ``t`` left after the first sweep, or rebound between them.
    between = "t = (y.iterator + 0).load()" if rebind else ""
    code = _fused(
        f"""
for synthetic_lane_7 in range(8):
    t = (y.iterator + synthetic_lane_7).load()
    v = (x.iterator + t).load()
    acc = acc + v
{between}
for synthetic_lane_7 in range(8):
    v = (x.iterator + t).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert code.count("v = (x.iterator + t).load()") == 2


@pytest.mark.cpu_only
def test_load_through_a_name_rebound_between_the_sweeps_is_not_fused() -> None:
    code = _fused(
        """
t = (y.iterator + 0).load()
for synthetic_lane_7 in range(8):
    v = (x.iterator + t + synthetic_lane_7).load()
    acc = acc + v
t = (y.iterator + 1).load()
for synthetic_lane_7 in range(8):
    v = (x.iterator + t + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert "_fuse_cache_" not in code


@pytest.mark.cpu_only
def test_sweep_whose_range_moves_is_not_fused() -> None:
    # Equal range text, but ``start`` moves between the sweeps: the second
    # sweep reads ``x[8:16]``, not the first sweep's ``x[0:8]``.
    code = _fused(
        """
start = 0
for synthetic_lane_7 in range(start, start + 8):
    before = (x.iterator + synthetic_lane_7).load()
    acc = acc + before
start = 8
for synthetic_lane_7 in range(start, start + 8):
    after = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(after)
"""
    )
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
def test_lane_loop_not_counting_from_zero_is_not_fused() -> None:
    # The cache index takes the lane variable for the slot: lanes 1..8 would
    # index past the 16 slots of two trips.
    sweep = """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(1, 9):
        value = (x.iterator + roffset_7 + reduction_lane_7).load()
        {use}
"""
    code = _fused(
        sweep.format(use="acc = acc + value")
        + sweep.format(use="(out.iterator + roffset_7).store(value)")
    )
    assert "_fuse_cache_" not in code


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("later", "between", "fuses"),
    [
        # A lane count rebound between the sweeps runs sixteen lanes over the
        # first sweep's eight slots per trip.
        ("K", "K = 16", False),
        # Another spelling of the same eight lanes.
        ("8", "", True),
    ],
    ids=("rebound", "respelled"),
)
def test_lane_ranges_must_keep_their_trip_counts(
    later: str, between: str, fuses: bool
) -> None:
    sweep = """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range({lanes}):
        value = (x.iterator + roffset_7 + reduction_lane_7).load()
        {use}
"""
    result = fuse_two_pass_loads(
        ast.parse(
            sweep.format(lanes="K", use="acc = acc + value")
            + between
            + sweep.format(
                lanes=later, use="(out.iterator + roffset_7).store(value + acc)"
            )
        ).body,
        {"K": 8},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert ("_fuse_cache_" in code) is fuses


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "source",
    [
        # The first sweep sets its index to 0 before its load: every
        # iteration writes slot 0, and the later sweep reads seven slots
        # never written.
        """
for synthetic_lane_7 in range(8):
    synthetic_lane_7 = 0
    v = (x.iterator + synthetic_lane_7).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    w = (x.iterator + 0).load()
    (out.iterator + synthetic_lane_7).store(w)
""",
        # The later sweep advances its index before its load: its last
        # iteration names a slot past the cache.
        """
for synthetic_lane_7 in range(8):
    v = (x.iterator + synthetic_lane_7).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    synthetic_lane_7 += 1
    w = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(w)
""",
        # The same for the lane loop's index.
        """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        reduction_lane_7 = 0
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        w = (x.iterator + roffset_7 + 0).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(w)
""",
    ],
)
def test_sweep_that_rebinds_its_index_is_not_fused(source: str) -> None:
    # A cache slot is the one the sweep's indices name.
    code = _fused(source)
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
def test_lane_loop_reusing_the_sweep_index_is_not_fused() -> None:
    # The lane loop hides the sweep index, which the cache slot reads too:
    # slot ``(lane - 0) // 1 * 8 + lane`` would reach 63 of 16.
    sweep = """
for synthetic_lane_7 in range(2):
    for synthetic_lane_7 in range(8):
        value = (x.iterator + synthetic_lane_7).load()
        {use}
"""
    code = _fused(
        sweep.format(use="acc = acc + value")
        + sweep.format(use="(out.iterator + synthetic_lane_7).store(value + acc)")
    )
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
def test_lane_index_read_after_its_loop_keeps_the_fusion() -> None:
    # ``last = reduction_lane_7`` reads the lane index after the lane loop;
    # both sweeps still read ``x[roffset_7 + lane]`` at equal iterations.
    code = _fused(
        """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        acc = acc + v
    last = reduction_lane_7
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        w = (x.iterator + roffset_7 + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(w)
"""
    )
    second = code[code.index("last = reduction_lane_7") :]
    assert "_fuse_cache_" in second
    assert ".load()" not in second


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("before", "first", "later"),
    [
        # The definitions before the sweeps differ only where each sweep
        # loop overwrites them.
        (("base = 0", "base = 0"), "base = roffset_7", "base = roffset_7 + 16"),
        (("", ""), "base = roffset_7 * 1", "base = roffset_7 * 1 + 8"),
    ],
)
def test_load_through_prefix_names_the_sweeps_define_apart_is_not_fused(
    before: tuple[str, str], first: str, later: str
) -> None:
    # Each sweep's prefix defines ``base`` its own way, so the two sweeps'
    # ``x[base + lane]`` read different elements.
    code = _fused(
        f"""
{before[0]}
for roffset_7 in range(0, 64, 32):
    {first}
    for reduction_lane_7 in range(8):
        v = (x.iterator + base + reduction_lane_7).load()
        acc = acc + v
{before[1]}
for roffset_7 in range(0, 64, 32):
    {later}
    for reduction_lane_7 in range(8):
        v = (x.iterator + base + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(v)
"""
    )
    assert "_fuse_cache_" not in code
    assert code.count("v = (x.iterator + base + reduction_lane_7).load()") == 2


@pytest.mark.cpu_only
def test_load_through_an_index_the_sweeps_load_apart_is_not_fused() -> None:
    # Both sweeps write ``t``, from different tensors: ``x[t]`` is another
    # element in each.
    code = _fused(
        """
for synthetic_lane_7 in range(8):
    t = (y.iterator + synthetic_lane_7).load()
    v = (x.iterator + t).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    t = (z.iterator + synthetic_lane_7).load()
    v = (x.iterator + t).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert code.count("v = (x.iterator + t).load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("first", "later"),
    [
        # In the lane body: ``c`` keeps ``k``'s value from before ``k``
        # moves on, so it is ``x[lane * 2]``, not ``x[(lane + 1) * 2]``.
        (
            """
for synthetic_lane_7 in range(8):
    k = synthetic_lane_7
    c = k * 2
    k = synthetic_lane_7 + 1
    v = (x.iterator + c).load()
    acc = acc + v
""",
            """
for synthetic_lane_7 in range(8):
    v = (x.iterator + (synthetic_lane_7 + 1) * 2).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        ),
        # Before the sweeps: the first sweep's ``c`` is ``0 * 2``, the second
        # sweep's ``1 * 2``.
        (
            """
k = 0
c = k * 2
k = 1
for synthetic_lane_7 in range(8):
    v = (x.iterator + c + synthetic_lane_7).load()
    acc = acc + v
""",
            """
c = k * 2
for synthetic_lane_7 in range(8):
    v = (x.iterator + c + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        ),
    ],
)
def test_definition_whose_operand_is_rebound_is_not_inlined(
    first: str, later: str
) -> None:
    code = _fused(first + later)
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
def test_load_through_a_fragment_written_between_the_sweeps_is_not_fused() -> None:
    code = _fused(
        """
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    acc = acc + v
offs[0] = 8
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "source",
    [
        # The store between the sweeps.
        """
q = offs
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    acc = acc + v
q[0] = 8
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        # The store in the first sweep: its later iterations read another
        # element than its first.
        """
q = offs
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    acc = acc + v
    q[0] = 8
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        # The store in the later sweep, before its load.
        """
q = offs
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    q[0] = 8
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        # The other name bound between the sweeps.
        """
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    acc = acc + v
q = offs
q[0] = 8
for synthetic_lane_7 in range(8):
    v = (x.iterator + offs[0] + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
        # The sweeps' range read from the fragment: the later sweep runs
        # other indices.
        """
q = bounds
for synthetic_lane_7 in range(bounds[0], bounds[0] + 8):
    v = (x.iterator + synthetic_lane_7).load()
    acc = acc + v
q[0] = 8
for synthetic_lane_7 in range(bounds[0], bounds[0] + 8):
    v = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
""",
    ],
)
def test_load_through_a_fragment_stored_through_another_name_is_not_fused(
    source: str,
) -> None:
    # ``q = offs`` makes ``q`` another name for the fragment ``offs``, so
    # ``q[0] = 8`` changes the element ``offs[0]`` the loads read.
    code = _fused(source)
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "effect",
    [
        "cute.arch.atomic_add(out.iterator + 0, cutlass.Float32(1.0))",
        "(out.iterator + 0).atomic_add(cutlass.Float32(1.0))",
        (
            "_cute_inline_asm_elementwise((), asm='mov.u32 $0, %clock;', "
            "constraints='=r', dtype=cutlass.Int32, is_pure=False)"
        ),
        "_cute_flex_load_vec(y.iterator + 0, cutlass.Int32)",
    ],
    ids=("atomic", "atomic-method", "impure-asm", "helper"),
)
def test_load_through_a_value_with_effects_is_not_fused(effect: str) -> None:
    # Each sweep computes ``slot`` again: an atomic returns another old value
    # each time, and asm not declared pure (a clock read) or a helper that
    # reads memory (a flex load) may too, so the sweeps' ``x[slot * 8 +
    # lane]`` need not read the same elements.
    code = _fused(
        f"""
for synthetic_lane_7 in range(8):
    slot = {effect}
    v = (x.iterator + slot * 8 + synthetic_lane_7).load()
    acc = acc + v
for synthetic_lane_7 in range(8):
    slot = {effect}
    w = (x.iterator + slot * 8 + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(w + acc)
"""
    )
    assert "w = (x.iterator + slot * 8 + synthetic_lane_7).load()" in code


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("source", "fuses"),
    [
        # A break of a loop nested in the lane loop ends only that loop.
        (
            """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        for k in range(4):
            if k == 2:
                break
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(v)
""",
            True,
        ),
        (
            """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        while acc > 0:
            acc = acc - 1
            if acc < 3:
                break
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(v)
""",
            True,
        ),
        # A loop's ``else`` runs in the lane loop: its ``continue`` skips
        # the rest of the lane's iteration.
        (
            """
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        for k in range(4):
            acc = acc + k
        else:
            continue
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        v = (x.iterator + roffset_7 + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(v)
""",
            False,
        ),
    ],
)
def test_only_an_exit_of_the_sweep_or_lane_loop_leaves_an_iteration_early(
    source: str, fuses: bool
) -> None:
    code = _fused(source)
    assert ("_fuse_cache_" in code) is fuses


@pytest.mark.cpu_only
@pytest.mark.parametrize("leave", ("break", "continue"))
def test_first_sweep_that_leaves_an_iteration_early_is_not_fused(leave: str) -> None:
    # The first sweep skips the load of lane 3 (``continue``) or of lanes 3
    # to 7 (``break``): the second sweep would read slots no load wrote.
    code = _fused(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    if cutlass.const_expr(synthetic_lane_7 == 3):
        {leave}
    v = (x.iterator + synthetic_lane_7).load()
    acc = acc + v
for synthetic_lane_7 in cutlass.range_constexpr(8):
    v = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(v)
"""
    )
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "source",
    [
        # ``for base in range(2)`` rebinds ``base`` when ``cond`` holds: the
        # first sweep's ``x[base + lane]`` is not the later ``x[0 + lane]``.
        """
for roffset_7 in range(0, 64, 32):
    base = 0
    if cond:
        for base in range(2):
            pass
    for reduction_lane_7 in range(8):
        v = (x.iterator + base + reduction_lane_7).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for reduction_lane_7 in range(8):
        w = (x.iterator + 0 + reduction_lane_7).load()
        (out.iterator + roffset_7 + reduction_lane_7).store(w)
""",
        # The later sweep loops over ``lane_b``: its ``lane_a`` is the first
        # sweep's last lane, not the lane of the iteration.
        """
for roffset_7 in range(0, 64, 32):
    for lane_a in range(8):
        v = (x.iterator + roffset_7 + lane_a).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for lane_b in range(8):
        w = (x.iterator + roffset_7 + lane_a).load()
        (out.iterator + roffset_7 + lane_b).store(w)
""",
        # The same with the vector-lane names the pass renames from sweep to
        # sweep: the later ``vec_lane_1`` is still the first sweep's last lane.
        """
for roffset_7 in range(0, 64, 32):
    for vec_lane_1 in range(8):
        v = (x.iterator + roffset_7 + vec_lane_1).load()
        acc = acc + v
for roffset_7 in range(0, 64, 32):
    for vec_lane_2 in range(8):
        w = (x.iterator + roffset_7 + vec_lane_1).load()
        (out.iterator + roffset_7 + vec_lane_2).store(w)
""",
        # The same through definitions the later keys inline: a name bound to
        # the first sweep's lane between the sweeps, read in the later lane
        # loop, through the later sweep's prefix, or beside the later lane.
        """
for roffset_7 in range(0, 64, 32):
    for vec_lane_1 in range(8):
        v = (x.iterator + roffset_7 + vec_lane_1).load()
        acc = acc + v
last = vec_lane_1
for roffset_7 in range(0, 64, 32):
    for vec_lane_2 in range(8):
        w = (x.iterator + roffset_7 + last).load()
        (out.iterator + roffset_7 + vec_lane_2).store(w)
""",
        """
for roffset_7 in range(0, 64, 32):
    for vec_lane_1 in range(8):
        v = (x.iterator + roffset_7 + vec_lane_1).load()
        acc = acc + v
saved = vec_lane_1
for roffset_7 in range(0, 64, 32):
    base = saved
    for vec_lane_2 in range(8):
        w = (x.iterator + roffset_7 + base).load()
        (out.iterator + roffset_7 + vec_lane_2).store(w)
""",
        """
for roffset_7 in range(0, 64, 32):
    for vec_lane_1 in range(8):
        v = (x.iterator + roffset_7 + vec_lane_1 + vec_lane_1).load()
        acc = acc + v
t = vec_lane_1
for roffset_7 in range(0, 64, 32):
    for vec_lane_2 in range(8):
        w = (x.iterator + roffset_7 + t + vec_lane_2).load()
        (out.iterator + roffset_7 + vec_lane_2).store(w)
""",
        # The later sweep advances ``start`` after its load, while its cache
        # index reads ``start`` in every iteration.
        """
start = 0
for synthetic_lane_7 in range(start, start + 8):
    before = (x.iterator + synthetic_lane_7).load()
    acc = acc + before
for synthetic_lane_7 in range(start, start + 8):
    after = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(after)
    start = start + 1
""",
        # ``offs[0] = 8`` changes ``p[0]`` through the other name: the first
        # sweep's ``c`` is the element before the store, the later one's the
        # element after it.
        """
p = offs
c = p[0]
offs[0] = 8
for synthetic_lane_7 in range(8):
    v = (x.iterator + c + synthetic_lane_7).load()
    acc = acc + v
c = p[0]
for synthetic_lane_7 in range(8):
    w = (x.iterator + c + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(w)
""",
    ],
)
def test_sweeps_that_may_read_other_elements_are_not_fused(source: str) -> None:
    code = _fused(source)
    assert "_fuse_cache_" not in code
    assert code.count(".load()") == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("pure", (False, True))
@pytest.mark.parametrize("where", ("between", "consume", "in_place"))
def test_inline_asm_of_unknown_effects_orders_the_sweeps(
    where: str, pure: bool
) -> None:
    # Inline asm not declared pure may write any tensor: between the sweeps,
    # or in the consume loop, it keeps the later sweep's loads, also where
    # that loop stores each element back into ``x`` (the in-place snapshot
    # proof).  Declared pure, it writes nothing.
    asm = (
        "_cute_inline_asm_elementwise((acc,), asm='add.f32 $0, $1, $1;', "
        f"constraints='=f,f', dtype=cutlass.Float32, is_pure={pure})"
    )
    between = f"v = {asm}" if where == "between" else ""
    inside = "" if where == "between" else f"v = {asm}"
    target = "x" if where == "in_place" else "out"
    code = _fused(
        f"""
for synthetic_lane_7 in range(8):
    before = (x.iterator + synthetic_lane_7).load()
    acc = acc + before
{between}
for synthetic_lane_7 in range(8):
    after = (x.iterator + synthetic_lane_7).load()
    {inside}
    ({target}.iterator + synthetic_lane_7).store(after)
"""
    )
    assert ("_fuse_cache_" in code) is pure


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


def test_hinted_vector_load_helpers_key_like_plain_loads() -> None:
    # Cross-sweep matching keys on the load text with hints stripped: a
    # hinted helper packet keys as the plain ``cute.arch.load`` it stands for.
    from helion._compiler.cute.fuse_two_pass_loads import _load_kind
    from helion._compiler.cute.fuse_two_pass_loads import _normalized_load_text

    vector_type = "ir.VectorType.get([4], cutlass.Uint16.mlir_type)"
    plain = ast.parse(f"cute.arch.load(x.iterator + i, {vector_type})", mode="eval")
    hinted = ast.parse(
        f"_cute_load_l2_evict_last_8b(x.iterator + i, {vector_type})", mode="eval"
    )
    assert _load_kind(plain.body) == "vec"
    assert _load_kind(hinted.body) == "vec"

    def key(node: ast.Expression) -> str:
        return _normalized_load_text(node.body, {})

    assert key(plain) == key(hinted)
    other = ast.parse(f"cute.arch.load(y.iterator + i, {vector_type})", mode="eval")
    assert key(other) != key(hinted)


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@onlyBackends(["cute"])
def test_fp8_raw_bytes_carried_across_sweeps_match_eager() -> None:
    # The consume sweep reads the reduce sweep's raw fp8 bytes back from a
    # ``Uint8`` fragment instead of loading them again, and decodes them
    # exactly as the load it replaces would.
    torch.manual_seed(0)
    x = torch.randn(8, 4096, device=DEVICE).to(torch.float8_e4m3fn)
    kernel = helion.kernel(center_fp8_row, backend="cute", static_shapes=True)
    bound = kernel.bind((x,))
    config = center_fp8_row_config(bound.config_spec)
    code = bound.to_code(config)
    assert "cute.make_rmem_tensor(32, cutlass.Uint8)" in code
    assert code.count("cute.arch.load(x.iterator") == 1
    out = bound.compile_config(config)(x)
    v = x.float()
    expected = (v - v.sum(-1, keepdim=True)).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@onlyBackends(["cute"])
def test_workspace_reload_matches_eager() -> None:
    # The barrier ordering the first sweep's workspace stores before the
    # second sweep's loads sits in each lane.  Each thread reads back only its
    # own stores here, and the row sum between the sweeps synchronizes the CTA
    # too.
    torch.manual_seed(0)
    x = torch.randn(8, 4096, device=DEVICE, dtype=torch.bfloat16)
    kernel = helion.kernel(workspace_reload, backend="cute", static_shapes=True)
    bound = kernel.bind((x,))
    config = workspace_chunk_config(bound.config_spec)
    out = bound.compile_config(config)(x)
    v = x.float()
    expected = (v * v - v.sum(-1, keepdim=True)).to(x.dtype)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)


@onlyBackends(["cute"])
def test_scaled_workspace_reload_ragged_byte_packets_match_eager() -> None:
    # Rows that end inside their last 1024-element chunk load ``keep`` as
    # byte packets too: 3000 columns (whole packets in a masked chunk) and
    # 3001 of rows padded to 3008 (the straddling packet's in-bounds lanes
    # reread one by one), as ``grad``'s carrier packets do.
    torch.manual_seed(0)
    kernel = helion.kernel(scaled_workspace_reload, backend="cute", static_shapes=True)
    for n, padded in ((3000, 3000), (3001, 3008)):
        x = torch.randn(8, padded, device=DEVICE, dtype=torch.bfloat16)[:, :n]
        grad = torch.randn(8, padded, device=DEVICE, dtype=torch.bfloat16)[:, :n]
        keep = (torch.rand(8, padded, device=DEVICE) > 0.1)[:, :n]
        bound = kernel.bind((x, grad, keep))
        config = scaled_workspace_config(
            bound.config_spec, "auto", bool_byte_packets=True
        )
        assert "cute.arch.load(keep.iterator" in bound.to_code(config)
        out = bound.compile_config(config)(x, grad, keep)
        v = x.float()
        scaled = grad.float() * (keep.float() * 1.1111111111111112)
        expected = (v * v * scaled - (scaled * v).sum(-1, keepdim=True)).to(x.dtype)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)


@onlyBackends(["cute"])
def test_scaled_workspace_reload_packets_match_eager() -> None:
    # Both sweeps load ``grad`` and, under ``cute_bool_byte_packets``, the
    # bool ``keep`` as packets: the reduce sweep's after the workspace store's
    # append, the consume sweep's behind the barrier that orders the workspace
    # reload; with register reloads the consume sweep reads the reduce sweep's
    # packets from a fragment (``test_cute_rolled_packet_hoist.py`` checks the
    # code).
    torch.manual_seed(0)
    rows, n = 8, 4096
    x = torch.randn(rows, n, device=DEVICE, dtype=torch.bfloat16)
    grad = torch.randn(rows, n, device=DEVICE, dtype=torch.bfloat16)
    keep = torch.rand(rows, n, device=DEVICE) > 0.1
    kernel = helion.kernel(scaled_workspace_reload, backend="cute", static_shapes=True)
    bound = kernel.bind((x, grad, keep))
    v = x.float()
    scaled = grad.float() * (keep.float() * 1.1111111111111112)
    expected = (v * v * scaled - (scaled * v).sum(-1, keepdim=True)).to(x.dtype)
    for reloads in ("auto", "register"):
        for bool_byte_packets in (False, True):
            config = scaled_workspace_config(
                bound.config_spec, reloads, bool_byte_packets=bool_byte_packets
            )
            out = bound.compile_config(config)(x, grad, keep)
            torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)


class TestCuteFuseTwoPassLoads(TestCase):
    def test_fuser_fires_with_vec_hoist(self) -> None:
        """When the vec hoist runs, the two sweeps' loads use names like
        ``lane_base_1`` vs ``lane_base_2`` and ``vec_lane_1`` vs
        ``vec_lane_2``. The keys spell ``lane_base_<N>`` by its definition
        and the pass renames ``vec_lane_2`` to ``vec_lane_1``, so the fuser
        matches and emits one ``_fuse_cache_*`` fragment instead of
        re-reading from gmem.

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

    def test_fuser_fires_across_a_cache_hinted_consume_load(self) -> None:
        """A consume-sweep load carrying an L2 hint (an 8-byte
        ``_cute_load_l2_evict_last_8b`` packet here) is the same logical load
        as the reduce sweep's plain ``cute.arch.load``: the fuser still
        matches the pair and the consume sweep reads the cache."""
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
            load_eviction_policies=["", "l2_last"],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertIn("_fuse_cache_", code)
        consume_start = code.rfind("for tile_offset_2 in range")
        self.assertGreater(consume_start, 0)
        self.assertNotIn("cute.arch.load(", code[consume_start:])
        self.assertNotIn("_cute_load_", code[consume_start:])

    def test_carried_word0_mask_matches_the_recomputed_mask(self) -> None:
        """The later sweeps read the first sweep's word0 mask and dropped-out
        row from fragments instead of redrawing them; the mask matches the
        ``gmem`` policy's recomputation bit for bit, the other outputs to the
        last place."""
        torch.manual_seed(0)
        x = torch.randn(64, 256, device=DEVICE, dtype=torch.bfloat16)
        residual = torch.randn(64, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        bias = torch.randn(256, device=DEVICE)
        seeds = torch.tensor([3, 1234567, 99, 7], device=DEVICE, dtype=torch.int64)
        args = (x, seeds, residual, weight, bias, 1, 255)
        config = {"block_sizes": [4], "num_threads": [4, 32]}
        code, carried = code_and_output(
            _dropout_layernorm, args, cute_reduction_reloads=["register"], **config
        )
        self.assertRegex(code, r"keep = cutlass\.Boolean\(_fuse_cache_\d+\[")
        recomputed_code, recomputed = code_and_output(
            _dropout_layernorm, args, cute_reduction_reloads=["gmem"], **config
        )
        self.assertNotIn("_fuse_cache_", recomputed_code)
        (keep, *values), (recomputed_keep, *recomputed_values) = carried, recomputed
        self.assertTrue(torch.equal(keep, recomputed_keep))
        # Under ``gmem`` the mean sweep's dropped-out product feeds only the
        # sum, and the compiler contracts the two into an FMA (one rounding
        # fewer); here the product also fills a fragment and is rounded
        # first.  The means can differ in the last place, and so can any
        # output after them: by one unit in the last place, or near zero,
        # where the terms cancel, by their fp32 rounding.
        for got, expected in zip(values, recomputed_values, strict=True):
            torch.testing.assert_close(
                got, expected, rtol=torch.finfo(expected.dtype).eps, atol=1e-5
            )
        self.assertGreater(keep.float().mean().item(), 0.8)

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
