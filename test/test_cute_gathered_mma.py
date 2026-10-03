from __future__ import annotations

import ast
import dataclasses
from types import SimpleNamespace

import pytest

from helion._compiler.cute.collective_matmul import CollectiveMmaSite
from helion._compiler.cute.collective_matmul import _region_write_roots
from helion._compiler.cute.gathered_mma import GatherTensorFacts
from helion._compiler.cute.gathered_mma import _expanded
from helion._compiler.cute.gathered_mma import analyze_gathered_mma_region

SOURCE = """
group = cutlass.Int32(cute.arch.block_idx()[0]) % groups
column_offset = cutlass.Int32(cute.arch.block_idx()[0]) // groups * BN
n = column_offset + cutlass.Int32(cute.arch.thread_idx()[0])
column_mask = cutlass.Int32(cute.arch.thread_idx()[0]) < BN and n < width
begin = (starts.iterator + cutlass.Int32(group) * cutlass.Int32(starts.layout.stride[0])).load()
count = (lengths.iterator + cutlass.Int32(group) * cutlass.Int32(lengths.layout.stride[0])).load()
if count != 0:
    for m_offset in range(cutlass.Int32(0), cutlass.Int32(maximum), cutlass.Int32(BM)):
        for m_lane in range(32):
            m = m_offset + cutlass.Int32(cute.arch.thread_idx()[1]) * 32 + cutlass.Int32(m_lane)
            row_mask = cutlass.Int32(cute.arch.thread_idx()[1]) < 4 and m < maximum
            active = m < count
            safe_m = cutlass.Int32(m) if active else cutlass.Int32(0)
            position = begin + safe_m
            row = (indices.iterator + cutlass.Int32(position) * cutlass.Int32(indices.layout.stride[0])).load() if position < index_length else cutlass.Int32(0)
            total = cutlass.Float32(0.0)
            for k_offset in range(cutlass.Int32(0), cutlass.Int32(inner), cutlass.Int32(BK)):
                for k_lane in range(64):
                    k = k_offset + cutlass.Int32(k_lane)
                    k_mask = k < inner
                    a = (left.iterator + cutlass.Int32(row) * cutlass.Int32(left.layout.stride[0]) + cutlass.Int32(k) * cutlass.Int32(left.layout.stride[1])).load() if cutlass.Int32(row) < source_rows and k_mask else cutlass.Float16(0)
                    b = (right.iterator + cutlass.Int32(group) * cutlass.Int32(right.layout.stride[0]) + cutlass.Int32(k) * cutlass.Int32(right.layout.stride[1]) + cutlass.Int32(n) * cutlass.Int32(right.layout.stride[2])).load() if k_mask and column_mask else cutlass.Float16(0)
                    total = _helion_pending_collective_mma(0, a, b, total)
            rounded = cutlass.Float16(total)
            if active and row_mask and column_mask:
                (result.iterator + cutlass.Int32(row) * cutlass.Int32(result.layout.stride[0]) + cutlass.Int32(n) * cutlass.Int32(result.layout.stride[1])).store(cutlass.Float16(rounded))
"""


def analyze(source: str = SOURCE, *, updates=None, site_updates=None):
    tensors = {
        "left": GatherTensorFacts("cutlass.Float16", (4096, 1024), (1024, 1), 16),
        "right": GatherTensorFacts(
            "cutlass.Float16", (16, 1024, 1024), (1048576, 1024, 1), 16
        ),
        "result": GatherTensorFacts("cutlass.Float16", (4096, 1024), (1024, 1), 16),
        "indices": GatherTensorFacts("cutlass.Int32", (4096,), (1,), 16),
        "lengths": GatherTensorFacts("cutlass.Int32", (16,), (1,), 16),
        "starts": GatherTensorFacts("cutlass.Int32", (17,), (1,), 16),
    }
    constants = {
        "source_rows": 4096,
        "index_length": 4096,
        "inner": 1024,
        "width": 1024,
        "groups": 16,
        "BM": 128,
        "BN": 32,
        "BK": 64,
    }
    site = CollectiveMmaSite(
        0,
        "m",
        "n",
        "k",
        "m_offset",
        "column_offset",
        "k_offset",
        128,
        32,
        64,
        1,
        0,
        "cutlass.Float16",
        None,
    )
    arguments = {
        "boundary_names": set(tensors) | set(constants) | {"maximum"},
        "tensors": tensors,
        "constants": constants,
        "thread_dims": (32, 4, 1),
        "disjoint_pairs": {
            frozenset((a, b)) for a in tensors for b in tensors if a != b
        },
    }
    if updates:
        arguments.update(updates(tensors, constants, arguments))
    if site_updates:
        site = dataclasses.replace(site, **site_updates)
    return analyze_gathered_mma_region(ast.parse(source).body, site, **arguments)


def test_direct_gather_is_admitted_without_index_values() -> None:
    plan = analyze()
    assert plan is not None
    assert (plan.lhs, plan.rhs, plan.output) == ("left", "right", "result")
    assert (plan.group_count, plan.n_size, plan.k_size) == (16, 1024, 1024)
    assert "indices.iterator" in ast.unparse(plan.row)
    # Keep the original masked metadata access and explicit index casts.
    assert "else cutlass.Int32(0)" in ast.unparse(plan.row)
    assert ast.unparse(plan.m_extent) == "cutlass.Int32(maximum)"
    assert "maximum" in ast.unparse(plan.row_predicate)


def _with_tensor_index_domain_masks() -> str:
    return SOURCE.replace(
        "if position < index_length", "if row_mask and position < index_length"
    ).replace("source_rows and k_mask", "source_rows and k_mask and row_mask")


@pytest.mark.parametrize(
    "thread_bound", ["4", "cutlass.Int32(4)", "cutlass.Int64(cutlass.Int32(4))"]
)
def test_tensor_index_domain_masks_preserve_metadata_guard(thread_bound: str) -> None:
    source = _with_tensor_index_domain_masks().replace(
        "cutlass.Int32(cute.arch.thread_idx()[1]) < 4",
        f"cutlass.Int32(cute.arch.thread_idx()[1]) < {thread_bound}",
    )
    plan = analyze(source)
    assert plan is not None
    row = ast.unparse(plan.row)
    assert "m < maximum" in row
    assert "else cutlass.Int32(0)" in row
    assert "thread_idx" not in row
    assert "m < maximum" in ast.unparse(plan.row_predicate)
    # Outside the logical domain, the row recipe must return zero without even
    # evaluating the metadata addresses (no tensor bindings are supplied).
    for m in (17, 18, 127):
        assert (
            eval(
                compile(
                    ast.fix_missing_locations(ast.Expression(plan.row)),
                    "<gather-row>",
                    "eval",
                ),
                {"__builtins__": {}},
                {"m": m, "maximum": 17, "cutlass": SimpleNamespace(Int32=int)},
            )
            == 0
        )


@pytest.mark.parametrize(
    "predicate",
    [
        "m < maximum - 1",
        "m <= maximum",
        "m < inner",
        "n < maximum",
        "row_mask and active",
    ],
)
def test_tensor_index_domain_mask_rejects_different_a_bound(predicate: str) -> None:
    source = _with_tensor_index_domain_masks().replace(
        "source_rows and k_mask and row_mask",
        f"source_rows and k_mask and ({predicate})",
    )
    assert analyze(source) is None


def test_tensor_index_domain_mask_requires_matching_scatter_bound() -> None:
    source = _with_tensor_index_domain_masks().replace(
        "if active and row_mask and column_mask:",
        "if active and column_mask:",
    )
    assert analyze(source) is None


@pytest.mark.parametrize(
    "predicate",
    [
        "cutlass.Int32(cute.arch.thread_idx()[1]) < 3",
        "cutlass.Int32(cute.arch.thread_idx()[0]) < 31",
        "_gather_thread_1 < 4",
    ],
)
def test_metadata_domain_rejects_partial_or_impostor_thread_masks(
    predicate: str,
) -> None:
    source = _with_tensor_index_domain_masks().replace(
        "if row_mask and position < index_length",
        f"if ({predicate}) and m < maximum and position < index_length",
    )
    assert analyze(source) is None


@pytest.mark.parametrize(
    "bound",
    [
        "cutlass.Int32(4294967296)",
        "cutlass.Int64(18446744073709551616)",
        "cutlass.Int32(cutlass.Int64(4294967296))",
        "cutlass.Int64(cutlass.Int32(4294967296))",
        "cutlass.Int32(-4294967296)",
        "cutlass.Int64(-18446744073709551616)",
        "cutlass.Int32(thread_bound)",
        "4294967296",
    ],
)
def test_metadata_thread_mask_rejects_unrepresentable_casts(bound: str) -> None:
    source = _with_tensor_index_domain_masks().replace(
        "if row_mask and position < index_length",
        f"if cutlass.Int32(cute.arch.thread_idx()[1]) < {bound} and m < maximum and position < index_length",
    )

    def update(tensors, constants, arguments):
        constants["thread_bound"] = 1 << 32
        return {}

    assert analyze(source, updates=update) is None


def test_user_scalar_cannot_impersonate_normalized_block_index() -> None:
    source = SOURCE.replace(
        "column_offset = cutlass.Int32(cute.arch.block_idx()[0])",
        "column_offset = cutlass.Int32(_gather_block)",
    )

    def update(tensors, constants, arguments):
        arguments["boundary_names"].add("_gather_block")
        return {}

    assert analyze(source, updates=update) is None


def test_extra_integer_row_predicate_is_preserved() -> None:
    source = SOURCE.replace(
        "if active and row_mask and column_mask:",
        "if active and row_mask and column_mask and (m + begin) % 3 == 0:",
    )
    plan = analyze(source)
    assert plan is not None
    assert "% 3 == 0" in ast.unparse(plan.row_predicate)


@pytest.mark.parametrize(
    "predicate",
    [
        "cutlass.Float32(m) * cutlass.Float32(1.1) - cutlass.Float32(m) * cutlass.Float32(1.1) == cutlass.Float32(0.0)",
        "(left.iterator + cutlass.Int32(row) * cutlass.Int32(left.layout.stride[0]) + cutlass.Int32(0) * cutlass.Int32(left.layout.stride[1])).load() > cutlass.Float16(0)",
        "left[0, 0] > 0",
        "left[0, 0] * m - left[0, 0] * m == 0",
        "indices[0] > 0",
    ],
)
def test_floating_row_predicate_is_not_rematerialized(predicate: str) -> None:
    source = SOURCE.replace(
        "if active and row_mask and column_mask:",
        f"if active and row_mask and column_mask and ({predicate}):",
    )
    assert analyze(source) is None


def test_floating_subscript_cannot_hide_in_integer_row_cast() -> None:
    source = SOURCE.replace(
        "position = begin + safe_m",
        "position = begin + safe_m + cutlass.Int32(left[0, 0])",
    )
    assert analyze(source) is None


def test_known_tensor_shape_metadata_is_integer() -> None:
    source = SOURCE.replace(
        "if active and row_mask and column_mask:",
        "if active and row_mask and column_mask and starts.shape[0] > 0:",
    )
    assert analyze(source) is not None


def test_small_integer_dag_preserves_values() -> None:
    statements = ast.parse(
        "\n".join(
            ["h0 = seed"]
            + [f"h{i} = (h{i - 1} * h{i - 1} + 1) % 32" for i in range(1, 5)]
        )
    ).body
    expression = _expanded(ast.Name(id="h4", ctx=ast.Load()), statements, {"seed"})
    assert expression is not None
    code = compile(
        ast.fix_missing_locations(ast.Expression(expression)), "<recipe>", "eval"
    )
    for seed in (0, 1, 7, 15, 31):
        expected = seed
        for _iteration in range(4):
            expected = (pow(expected, 2) + 1) % 32
        assert eval(code, {"__builtins__": {}, "seed": seed}) == expected


@pytest.mark.parametrize("depth", [8, 12, 40])
def test_large_integer_dag_declines_complete_admission(depth: int) -> None:
    steps = ["h0 = safe_m"]
    steps.extend(f"h{i} = (h{i - 1} * h{i - 1} + 1) % 32" for i in range(1, depth + 1))
    steps.append(f"position = begin + h{depth}")
    source = SOURCE.replace("position = begin + safe_m", "\n            ".join(steps))
    assert analyze(source) is None


def test_padded_bfloat16_tails() -> None:
    def update(tensors, constants, arguments):
        for name, shape, strides in (
            ("left", (4096, 78), (80, 1)),
            ("right", (16, 78, 70), (5760, 72, 1)),
            ("result", (4096, 70), (70, 1)),
        ):
            tensors[name] = GatherTensorFacts("cutlass.BFloat16", shape, strides, 16)
        constants.update(inner=78, width=70)
        return {}

    assert (
        analyze(
            SOURCE.replace("cutlass.Float16", "cutlass.BFloat16"),
            updates=update,
            site_updates={"dtype": "cutlass.BFloat16"},
        )
        is not None
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("cutlass.Float16(total)", "cutlass.Float16(total + 1.0)"),
        (
            "cutlass.Int32(row) * cutlass.Int32(result.layout.stride[0])",
            "cutlass.Int32(row + 1) * cutlass.Int32(result.layout.stride[0])",
        ),
        (
            "cutlass.Int32(n) * cutlass.Int32(result.layout.stride[1])",
            "cutlass.Int32(n % 16) * cutlass.Int32(result.layout.stride[1])",
        ),
        (
            "cutlass.Int32(group) * cutlass.Int32(right.layout.stride[0])",
            "cutlass.Int32(group + m) * cutlass.Int32(right.layout.stride[0])",
        ),
        (
            "cutlass.Int32(n) * cutlass.Int32(right.layout.stride[2])",
            "cutlass.Int32(n + 1) * cutlass.Int32(right.layout.stride[2])",
        ),
        ("source_rows and k_mask", "source_rows and k_mask and active"),
        ("if k_mask and column_mask", "if k_mask and column_mask and n % 2 == 0"),
        (
            "cutlass.Int32(inner), cutlass.Int32(BK)",
            "cutlass.Int32(inner - 1), cutlass.Int32(BK)",
        ),
        (
            "range(cutlass.Int32(0), cutlass.Int32(inner)",
            "range(cutlass.Int32(16), cutlass.Int32(inner)",
        ),
        ("if count != 0:", "if cute.arch.thread_idx()[0] == 0:"),
        ("if count != 0:", "if cute.arch.thread_idx()[0] < 32:"),
        ("if count != 0:", "if cute.arch.block_dim()[0] == 128:"),
        (
            "if active and row_mask and column_mask:",
            "if (active if cute.arch.thread_idx()[0] < 32 else False) and row_mask and column_mask:",
        ),
        (
            "safe_m = cutlass.Int32(m)",
            "safe_m = cutlass.Int32(m + cute.arch.thread_idx()[0])",
        ),
        ("position = begin + safe_m", "position = cutlass.Int64(begin) + safe_m"),
        ("position = begin + safe_m", "position = cutlass.Int16(begin + safe_m)"),
        ("position = begin + safe_m", "position = cutlass.Int32((begin + safe_m) / 2)"),
        (
            "position = begin + safe_m",
            "position = cute.arch.inline_ptx('mov.u32 %0, %tid.x;', read_only_args=[])",
        ),
        (
            "rounded = cutlass.Float16(total)",
            "rounded = (result.iterator + cutlass.Int32(row) * cutlass.Int32(result.layout.stride[0]) + cutlass.Int32(n) * cutlass.Int32(result.layout.stride[1])).load()",
        ),
        (
            "total = cutlass.Float32(0.0)",
            "total = cutlass.Float32(0.0)\n            cute.arch.atomic_add(result.iterator, cutlass.Float16(1.0))",
        ),
        (
            "total = cutlass.Float32(0.0)",
            "total = cutlass.Float32(0.0)\n            opaque_effect()",
        ),
        ("if count != 0:", "if count != 0:\n    count = count + 1"),
        ("total = cutlass.Float32(0.0)", "total = cutlass.Float32(1.0)"),
        (
            "rounded = cutlass.Float16(total)",
            "total = total + 1.0\n            rounded = cutlass.Float16(total)",
        ),
        (
            "total = _helion_pending_collective_mma(0, a, b, total)",
            "total = _helion_pending_collective_mma(0, a, b, total)\n                    total += 1.0",
        ),
        (
            "_helion_pending_collective_mma(0, a, b, total)",
            "_helion_pending_collective_mma(0, a, b, cutlass.Float32(0.0))",
        ),
        ("cutlass.Int32(maximum)", "cutlass.Int64(maximum)"),
        ("for m_lane in range(32):", "for m_lane in range(16):"),
        ("for k_lane in range(64):", "for k_lane in range(32):"),
        (
            "column_offset + cutlass.Int32(cute.arch.thread_idx()[0])",
            "column_offset + cutlass.Int32(cute.arch.thread_idx()[0]) * 2",
        ),
        (
            "// groups * BN",
            "// groups * BN + 1",
        ),
    ],
)
def test_semantic_mismatches_reject(old: str, new: str) -> None:
    assert old in SOURCE
    assert analyze(SOURCE.replace(old, new)) is None


def test_second_store_rejects_even_when_same_tensor() -> None:
    assert (
        analyze(SOURCE + "\n(result.iterator + 0).store(cutlass.Float16(1.0))\n")
        is None
    )


@pytest.mark.parametrize("scope", ["cta", "gpu"])
def test_relaxed_atomic_to_same_output_is_not_discarded(scope: str) -> None:
    module = ast.parse(SOURCE)
    store = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    )
    parent = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Expr) and statement.value is store
            for statement in node.body
        )
    )
    assert isinstance(store.func, ast.Attribute)
    pointer = ast.unparse(store.func.value)
    parent.body.extend(
        ast.parse(
            f"cute.arch.atomic_add(({pointer}).llvm_ptr, "
            f"val=cutlass.Float16(1), sem='relaxed', scope={scope!r})"
        ).body
    )
    # This is a classified effect, unlike an opaque or malformed atomic call.
    # For unique routing, each source element receives its store then +1.
    assert _region_write_roots(module.body, 0) == {"result"}
    assert analyze(ast.unparse(module)) is None


@pytest.mark.parametrize("extent", [0, 1, 2])
def test_additional_epilogue_loop_rejects(extent: int) -> None:
    module = ast.parse(SOURCE)
    parent = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "m_lane"
    )
    statement = parent.body[-1]
    parent.body[-1] = ast.For(
        target=ast.Name(id="extra", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Constant(extent)],
            keywords=[],
        ),
        body=[statement],
        orelse=[],
    )
    assert analyze(ast.unparse(ast.fix_missing_locations(module))) is None


def test_ordered_carry_rejects() -> None:
    source = SOURCE.replace("if count != 0:", "cursor = 0\nif count != 0:")
    source = source.replace(
        "for m_lane in range(32):",
        "cursor = cursor + 1\n        for m_lane in range(32):",
    )
    assert analyze(source) is None


def test_current_k_iteration_accumulator_alias_is_admitted() -> None:
    source = SOURCE.replace(
        "total = _helion_pending_collective_mma(0, a, b, total)",
        "current = total\n                    total = _helion_pending_collective_mma(0, a, b, current)",
    )
    assert analyze(source) is not None


@pytest.mark.parametrize("inside_k", [False, True])
def test_stale_or_shadowed_accumulator_alias_rejects(inside_k: bool) -> None:
    source = SOURCE.replace(
        "total = cutlass.Float32(0.0)",
        "total = cutlass.Float32(0.0)\n            current = total",
    )
    replacement = "total = _helion_pending_collective_mma(0, a, b, current)"
    if inside_k:
        replacement = (
            "current = total\n                    current = cutlass.Float32(0.0)\n                    "
            + replacement
        )
    source = source.replace(
        "total = _helion_pending_collective_mma(0, a, b, total)", replacement
    )
    assert analyze(source) is None


@pytest.mark.parametrize("missing", ["a_k", "b_k", "b_n"])
def test_missing_tail_load_masks_reject(missing: str) -> None:
    def update(tensors, constants, arguments):
        for name, shape, strides in (
            ("left", (4096, 78), (80, 1)),
            ("right", (16, 78, 70), (5760, 72, 1)),
            ("result", (4096, 70), (70, 1)),
        ):
            tensors[name] = dataclasses.replace(
                tensors[name], shape=shape, strides=strides
            )
        constants.update(inner=78, width=70)
        return {}

    replacements = {
        "a_k": ("source_rows and k_mask", "source_rows"),
        "b_k": ("if k_mask and column_mask", "if column_mask"),
        "b_n": ("if k_mask and column_mask", "if k_mask"),
    }
    before, after = replacements[missing]
    assert analyze(SOURCE.replace(before, after), updates=update) is None


@pytest.mark.parametrize(
    ("tensor", "field", "value"),
    [
        ("left", "alignment", 2),
        ("right", "strides", (1048576, 1025, 1)),
        ("left", "strides", (-1024, 1)),
        ("result", "strides", (512, 1)),
        ("right", "strides", (1 << 31, 1024, 1)),
        ("indices", "dtype", "cutlass.Int64"),
        ("starts", "dtype", "cutlass.Int64"),
        ("lengths", "dtype", "cutlass.Int64"),
    ],
)
def test_metadata_contract_rejects(tensor, field, value) -> None:
    def update(tensors, constants, arguments):
        tensors[tensor] = dataclasses.replace(tensors[tensor], **{field: value})
        return {}

    assert analyze(updates=update) is None


@pytest.mark.parametrize("name", ["left", "right", "indices", "starts", "lengths"])
def test_all_read_roots_require_disjoint_output(name: str) -> None:
    def update(tensors, constants, arguments):
        arguments["disjoint_pairs"].remove(frozenset((name, "result")))
        return {}

    assert analyze(updates=update) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"zero_seed": False},
        {"bm": 64},
        {"bk": 128},
        {"k_factor": 2},
        {"synthetic_k_lane": "synthetic"},
        {"grid_row_lane": "row_lane"},
    ],
)
def test_other_collective_envelopes_remain_on_existing_path(updates) -> None:
    assert analyze(site_updates=updates) is None
