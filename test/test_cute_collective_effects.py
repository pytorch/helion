from __future__ import annotations

import ast
import itertools
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion import exc
from helion._compiler.cute.collective_matmul import CollectiveMmaSite
from helion._compiler.cute.collective_matmul import _lower_site
from helion._compiler.cute.collective_matmul import _region_write_roots

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction


@pytest.mark.parametrize(
    "iterator", ["range", "cutlass.range", "cutlass.range_constexpr"]
)
def test_region_effects_accept_scalar_dataflow_and_known_stores(iterator: str) -> None:
    body = ast.parse(
        f"""
row = cutlass.Int32(cute.arch.block_idx()[0])
stride = a.layout.stride[0]
for i in {iterator}(4):
    value = (a.iterator + row * stride + i).load() if i < size else cutlass.Float16(0)
    value = cute.math.exp2(value).to(cutlass.Float32)
    acc = _helion_pending_collective_mma(0, value, other, acc)
    if i < size:
        (out.iterator + i).store(cutlass.Float16(acc))
    else:
        pass
"""
    ).body
    assert _region_write_roots(body, 0) == {"out"}


@pytest.mark.parametrize(
    "source",
    [
        "cute.arch.atomic_add(ptr, val=value)",
        "cute.arch.atomic_exch(ptr, val=value)",
        "cute.arch.atomic_cas(ptr, cmp=old, val=value)",
        "result = cute.arch.atomic_add(ptr, val=value)",
        "opaque_effect(out)",
        "result = opaque_effect(out)",
        "(out.iterator + i).unknown_write(value)",
        "(out.iterator + i).store(opaque_effect(value))",
        "(out.iterator + i).store(value, unknown_flag=True)",
        "ptr.store(value)",
        "(a.iterator + b.iterator).store(value)",
        "out[i] = value",
        "out[i] += value",
        "out.field = value",
        "left, out[i] = values",
        "del out[i]",
        "cute.arch.sync_threads()",
        "cute.arch.cp_async_commit_group()",
        "if opaque_predicate():\n    pass",
        "for i in opaque_iterator():\n    pass",
        "while predicate:\n    pass",
        "acc = _helion_pending_collective_mma(1, left, right, acc)",
    ],
)
def test_region_effects_refuse_unclassified_effects(source: str) -> None:
    assert _region_write_roots(ast.parse(source).body, 0) is None


def _lower(
    prefix: str = "",
    *,
    root_prefix: str = "",
    outer_header: str | None = None,
    root_suffix: str = "",
) -> str:
    prefix = "\n".join(f"    {line}" for line in prefix.splitlines())
    source = f"""
n_offset = 0
for m_offset in range(0, m, 64):
{prefix}
    k_limit = (limits.iterator + cutlass.Int32(0)).load()
    for m_lane in range(16):
        m_index = m_offset + cutlass.Int32(cute.arch.thread_idx()[1]) * 16 + m_lane
        n_index = n_offset + cutlass.Int32(cute.arch.thread_idx()[0])
        acc = cutlass.Float32(0)
        for k_offset in range(0, k_limit, 16):
            k_index = k_offset
            left = (a.iterator + m_index * 64 + k_index).load()
            right = (b.iterator + k_index * 32 + n_index).load()
            acc = _helion_pending_collective_mma(0, left, right, acc)
        (out.iterator + m_index * 32 + n_index).store(cutlass.Float16(acc))
"""
    if outer_header is not None:
        source = f"{outer_header}:\n{textwrap.indent(source, '    ')}"
    body = ast.parse(f"{root_prefix}\n{source}\n{root_suffix}\n").body
    site = CollectiveMmaSite(
        identity=0,
        m_index="m_index",
        n_index="n_index",
        k_index="k_index",
        m_offset="m_offset",
        n_offset="n_offset",
        k_offset="k_offset",
        bm=64,
        bn=32,
        bk=16,
        m_axis=1,
        n_axis=0,
        dtype="cutlass.Float16",
        grid_row_lane=None,
    )
    counter = itertools.count()
    df = SimpleNamespace(
        # These AST fixtures use explicit pointer loads, not tensor subscripts.
        arguments=[],
        config={"cute_collective_copy": "scalar"},
        tile_strategy=SimpleNamespace(thread_block_dims=lambda: (32, 4, 1)),
        codegen=SimpleNamespace(max_thread_block_dims=(32, 4, 1)),
        new_var=lambda hint: f"{hint}_{next(counter)}",
        cute_state=SimpleNamespace(
            collective_mma_sites=[site],
            collective_mma_emitted_stmt_ids=set(),
            collective_mma_shared_results=set(),
            collective_mma_pure_stmt_ids=set(),
        ),
    )
    tensors = ("a", "b", "out", "limits", "flags", "rows")
    _lower_site(
        body,
        site,
        cast("DeviceFunction", df),
        {*tensors, "m", "active", "seed"},
        {frozenset(pair) for pair in itertools.combinations(tensors, 2)},
    )
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )


def test_hoisting_preserves_disjoint_row_prefix_store() -> None:
    source = _lower("(out.iterator + cutlass.Int32(0)).store(cutlass.Float16(0))")
    assert "cute.gemm(" in source
    assert "_helion_pending_collective_mma" not in source
    assert "out.iterator" in source


def test_hoisting_checks_bound_reads_even_when_operands_do_not_read_them() -> None:
    # Neither operand's full-tile load uses limits. The reduction bound itself
    # must participate in the alias proof, independently of operand tail masks.
    with pytest.raises(exc.BackendUnsupported, match="may alias row-loop writes"):
        _lower("(limits.iterator + cutlass.Int32(0)).store(cutlass.Int32(16))")


@pytest.mark.parametrize(
    "prefix",
    [
        "cute.arch.atomic_exch((limits.iterator + 0).llvm_ptr, val=cutlass.Int32(16))",
        "opaque_effect(limits)",
        "limits[0] = cutlass.Int32(16)",
    ],
)
def test_hoisting_refuses_effects_hidden_from_pointer_store_scan(prefix: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="unclassified effects"):
        _lower(prefix)


@pytest.mark.parametrize(
    ("root_prefix", "outer_header"),
    [
        ("", "if active"),
        ("predicate = cutlass.Int32(cute.arch.block_idx()[0]) < 4", "if predicate"),
        ("predicate = (flags.iterator + 0).load() > 0", "if predicate"),
        (
            "row = (rows.iterator + 0).load()\npredicate = (flags.iterator + row).load() > 0",
            "if predicate",
        ),
        ("", "for repetition in range(4)"),
        ("count = (flags.iterator + 0).load()", "for repetition in range(count)"),
    ],
)
def test_uniform_enclosing_control_accepts_readonly_metadata(
    root_prefix: str, outer_header: str
) -> None:
    assert "cute.gemm(" in _lower(root_prefix=root_prefix, outer_header=outer_header)


@pytest.mark.parametrize(
    "expression",
    [
        "cutlass.Int32(cute.arch.thread_idx()[0]) < 16",
        "cute.arch.lane_idx() < 16",
        "cute.arch.warp_idx() < 2",
        "opaque_predicate(seed)",
        "_cute_inline_asm_elementwise((seed,), asm='mov.u32 $0, %tid.x;', constraints='=r,r', dtype=cutlass.Int32, is_pure=True) < 16",
    ],
)
def test_scalar_enclosing_predicate_requires_cta_uniformity(expression: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="enclosing control flow"):
        _lower(root_prefix=f"predicate = {expression}", outer_header="if predicate")


@pytest.mark.parametrize(
    "expression",
    [
        "cutlass.Int32(cute.arch.thread_idx()[0]) + 1",
        "cute.arch.lane_idx() + 1",
        "cute.arch.warp_idx() + 1",
        "opaque_count(seed)",
    ],
)
def test_enclosing_loop_trip_count_requires_cta_uniformity(expression: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="enclosing control flow"):
        _lower(
            root_prefix=f"count = {expression}",
            outer_header="for repetition in range(count)",
        )


@pytest.mark.parametrize(
    "root_suffix",
    [
        "(flags.iterator + 0).store(cutlass.Int32(0))",
        "cute.arch.atomic_exch((flags.iterator + 0).llvm_ptr, val=cutlass.Int32(0))",
        "opaque_effect(flags)",
    ],
)
@pytest.mark.parametrize(
    "outer_header", ["if count > 0", "for repetition in range(count)"]
)
def test_load_derived_control_checks_writes_outside_row_loop(
    root_suffix: str, outer_header: str
) -> None:
    with pytest.raises(exc.BackendUnsupported, match="control.flow"):
        _lower(
            root_prefix="count = (flags.iterator + 0).load()",
            outer_header=outer_header,
            root_suffix=root_suffix,
        )


def test_indirect_predicate_checks_metadata_gather_source_writes() -> None:
    with pytest.raises(exc.BackendUnsupported, match="control.flow"):
        _lower(
            root_prefix="row = (rows.iterator + 0).load()\npredicate = (flags.iterator + row).load() > 0",
            outer_header="if predicate",
            root_suffix="(rows.iterator + 0).store(cutlass.Int32(0))",
        )


def test_reduction_bound_checks_writes_outside_row_loop() -> None:
    with pytest.raises(exc.BackendUnsupported, match="control.flow"):
        _lower(root_suffix="(limits.iterator + 0).store(cutlass.Int32(0))")


def test_enclosing_loop_induction_cannot_be_mutated_in_body() -> None:
    with pytest.raises(exc.BackendUnsupported, match="enclosing loop"):
        _lower(
            "repetition = cutlass.Int32(cute.arch.thread_idx()[0])",
            outer_header="for repetition in range(4)",
        )
