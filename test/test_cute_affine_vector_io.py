from __future__ import annotations

import ast
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from helion._compiler.cute.affine_vector_io import _Vectorizer
from helion._compiler.cute.scalar_integer import _integer_expression


class _Float32(float):
    def bitcast(self, target):
        assert target is _Uint32
        return _Uint32(struct.unpack("I", struct.pack("f", self))[0])


class _Uint32(int):
    def bitcast(self, target):
        assert target is _Float32
        return _Float32(struct.unpack("f", struct.pack("I", self))[0])


class _Pointer:
    def __init__(self, values: list[float], stats: dict[str, int], offset: int = 0):
        self.values = values
        self.offset = offset
        self.stats = stats

    def __add__(self, offset: int):
        return _Pointer(self.values, self.stats, self.offset + offset)

    def load(self):
        assert 0 <= self.offset < len(self.values)
        self.stats["scalar_loads"] += 1
        return _Float32(self.values[self.offset])

    def store(self, value):
        assert 0 <= self.offset < len(self.values)
        self.stats["scalar_stores"] += 1
        self.values[self.offset] = float(value)


class _Tensor(SimpleNamespace):
    def __getitem__(self, index):
        return (self.iterator + index).load()


def _vector_load(pointer, vector_type):
    pointer.stats["vector_loads"] += 1
    width = vector_type[0][0]
    assert pointer.offset >= 0 and pointer.offset + width <= len(pointer.values)
    return [
        _Float32(value).bitcast(_Uint32)
        for value in pointer.values[pointer.offset : pointer.offset + width]
    ]


def _vector_store(pointer, values):
    pointer.stats["vector_stores"] += 1
    assert pointer.offset >= 0 and pointer.offset + len(values) <= len(pointer.values)
    for index, value in enumerate(values):
        pointer.values[pointer.offset + index] = float(value.bitcast(_Float32))


# Model the generated Int32 coordinates explicitly; unknown scalar arithmetic
# is tested separately and must not inherit an integer type from Python inputs.
_SOURCE = """
def kernel(src, out, start, length):
    for tile_offset_1 in range(0, cutlass.Int32(length), 16):
        for lane_1 in range(4):
            lane_base_1 = cutlass.Int32(tile_offset_1) + cutlass.Int32(lane_1) * 4
            for vec_lane_1 in cutlass.range_constexpr(4):
                indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)
                mask_1 = indices_1 < cutlass.Int32(length)
                address = cutlass.Int32(start) + indices_1
                value = (src.iterator + cutlass.Int32(address) * cutlass.Int32(src.layout.stride[0])).load() if mask_1 else cutlass.Float32(0)
                result = value * 1.25 + 2
                if mask_1:
                    (out.iterator + cutlass.Int32(address) * cutlass.Int32(out.layout.stride[0])).store(cutlass.Float32(result))
"""


def _rewrite(
    source: str,
    *,
    aligned: bool = True,
    disjoint: bool = True,
    integer_tensors: frozenset[str] = frozenset(),
):
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    rewriter = _Vectorizer(
        function.body,
        strides={("src", 0): 1, ("out", 0): 1},
        alignments={"src": 16 if aligned else 1, "out": 16},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))} if disjoint else set(),
        constexpr={},
        integer_tensors=integer_tensors,
    )
    function.body = rewriter.body(function.body)
    return ast.fix_missing_locations(tree), rewriter.changed


def _execute(
    tree: ast.Module, start: int, length: int, *, int32=int, int64=int, extra_args=()
):
    stats = dict.fromkeys(
        ("scalar_loads", "scalar_stores", "vector_loads", "vector_stores"), 0
    )
    data = [float(value) for value in range(80)]
    output = [-123.0] * 80
    src = _Tensor(iterator=_Pointer(data, stats), layout=SimpleNamespace(stride=(1,)))
    out = _Tensor(iterator=_Pointer(output, stats), layout=SimpleNamespace(stride=(1,)))
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=int32,
            Int64=int64,
            Float32=_Float32,
            Uint32=_Uint32,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(load=_vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape, dtype))
        ),
        "_cute_store_u32_vec": _vector_store,
    }
    _Uint32.mlir_type = "i32"
    exec(compile(tree, "<affine-lane-check>", "exec"), namespace)
    namespace["kernel"](src, out, start, length, *extra_args)
    return output, stats


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 17, 32])
@pytest.mark.parametrize("start", [0, 1, 4])
def test_vector_path_and_scalar_tails_preserve_memory_values(
    length: int, start: int
) -> None:
    transformed, changed = _rewrite(_SOURCE)
    assert changed == 1
    expected, before = _execute(ast.parse(_SOURCE), start, length)
    actual, after = _execute(transformed, start, length)
    assert actual == expected
    assert before["scalar_loads"] == length
    expected_vectors = length // 4 if start % 4 == 0 else 0
    assert after["vector_loads"] == after["vector_stores"] == expected_vectors
    assert after["scalar_loads"] == length - expected_vectors * 4


@pytest.mark.parametrize(
    "predicate",
    [
        "indices_1 < cutlass.Int32(length) and indices_1 % 4 != 2",
        "indices_1 < cutlass.Int32(length) and indices_1 != 2",
    ],
)
def test_mask_with_holes_never_uses_endpoint_only_proof(predicate: str) -> None:
    source = _SOURCE.replace("indices_1 < cutlass.Int32(length)", predicate)
    tree, changed = _rewrite(source)
    assert changed == 1
    expected, _stats = _execute(ast.parse(source), 0, 17)
    actual, stats = _execute(tree, 0, 17)
    assert actual == expected
    assert stats["vector_loads"] == (3 if "%" not in predicate else 0)


def test_full_fragment_guard_removes_redundant_active_memory_masks() -> None:
    tree, changed = _rewrite(_SOURCE)
    assert changed == 1
    fast = next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("_helion_affine_load")
                for target in item.targets
            )
            for item in node.body
        )
    )
    # The enclosing guard proves every lane active; selects in the fast body
    # must fold, while the existing partial-tail semantic tests cover fallback.
    assert not any(
        isinstance(item, ast.IfExp)
        for statement in fast
        for item in ast.walk(statement)
    )
    assert "mask_1 = True" in ast.unparse(ast.Module(body=fast, type_ignores=[]))


@pytest.mark.parametrize(
    "mask",
    ["cutlass.Int32(2)", "cutlass.Float32(2.5)", "cutlass.Float32(-0.0)"],
)
def test_numeric_mask_truthiness_preserves_its_arithmetic_value(mask: str) -> None:
    source = (
        _SOURCE.replace("indices_1 < cutlass.Int32(length)", mask)
        .replace("address = cutlass.Int32(start) + indices_1", "address = indices_1")
        .replace("result = value * 1.25 + 2", "result = value * mask_1 + 2")
        + "    (out.iterator + cutlass.Int32(79)).store(cutlass.Float32(mask_1))\n"
    )
    tree, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 0, 16)
    actual, _after = _execute(tree, 0, 16)
    assert changed == 1
    assert actual == expected
    assert struct.pack("f", actual[-1]) == struct.pack("f", expected[-1])


@pytest.mark.parametrize(("aligned", "disjoint"), [(False, True), (True, False)])
def test_missing_alignment_or_alias_proofs_preserve_scalar_loop(
    aligned: bool, disjoint: bool
) -> None:
    tree, changed = _rewrite(_SOURCE, aligned=aligned, disjoint=disjoint)
    assert changed == 0
    actual, stats = _execute(tree, 0, 17)
    expected, _stats = _execute(ast.parse(_SOURCE), 0, 17)
    assert actual == expected
    assert stats["vector_loads"] == stats["vector_stores"] == 0


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "address = cutlass.Int32(start) + indices_1",
            "address = cutlass.Int32(start) + indices_1 * 2",
        ),
        (
            "address = cutlass.Int32(start) + indices_1",
            "address = cutlass.Int32(start) + indices_1 % 3",
        ),
        (
            "result = value * 1.25 + 2",
            "cute.arch.sync_threads()\n                result = value * 1.25 + 2",
        ),
        (
            "cutlass.Int32(address) * cutlass.Int32(src.layout.stride[0])",
            "cutlass.Int64(cutlass.Int32(address + 2)) - 2",
        ),
        (
            "result = value * 1.25 + 2",
            "(out.iterator + cutlass.Int32(address)).store(value)\n                result = value * 1.25 + 2",
        ),
    ],
)
def test_nonaffine_effectful_or_mixed_width_loops_are_unchanged(
    before: str, after: str
) -> None:
    tree, changed = _rewrite(_SOURCE.replace(before, after))
    assert changed == 0
    assert "_helion_affine_load" not in ast.unparse(tree)


@pytest.mark.parametrize("length", [0, 1, 3, 4, 8])
def test_uniform_inactive_fragments_preserve_reduction_carries(length: int) -> None:
    source = _SOURCE.replace(
        "for tile_offset_1 in range(0, cutlass.Int32(length), 16):",
        "total = cutlass.Float32(-0.0)\n    for tile_offset_1 in range(0, 32, 16):",
    ).replace(
        "indices_1 < cutlass.Int32(length)", "indices_1 < cutlass.Int32(length) * 4"
    )
    source = source.replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    source = source.replace(
        "result = value * 1.25 + 2",
        "total = total + (value if mask_1 else cutlass.Float32(0))\n                result = value * 1.25 + 2",
    )
    source += "    (out.iterator + cutlass.Int32(79)).store(cutlass.Float32(total))\n"
    tree, changed = _rewrite(source)
    assert changed == 1
    expected, _stats = _execute(ast.parse(source), 0, length)
    actual, stats = _execute(tree, 0, length)
    assert actual == expected
    # Even an entirely inactive loop must execute the original additions of
    # +0, which turn an incoming -0 carry into +0.
    assert struct.pack("f", actual[-1]) == struct.pack("f", expected[-1])
    assert stats["scalar_loads"] == 0
    vector_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_helion_affine_load"
        in ast.unparse(ast.Module(body=node.body, type_ignores=[]))
    ]
    assert len(vector_guards) == 1
    inactive = ast.unparse(ast.Module(body=vector_guards[0].orelse, type_ignores=[]))
    assert ".load()" not in inactive and ".store(" not in inactive
    assert "total = total + cutlass.Float32(0)" in inactive


def test_different_uniform_masks_retain_scalar_fallback() -> None:
    source = _SOURCE.replace(
        "indices_1 < cutlass.Int32(length)", "indices_1 < cutlass.Int32(length) * 4"
    )
    source = source.replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    source = source.replace(
        "if mask_1:", "if indices_1 < cutlass.Int32(length) * 4 - 4:"
    )
    tree, changed = _rewrite(source)
    assert changed == 1
    expected, _stats = _execute(ast.parse(source), 0, 4)
    actual, _stats = _execute(tree, 0, 4)
    assert actual == expected
    assert ".load()" in ast.unparse(tree)


def test_implicit_widening_does_not_hide_a_signed_mask_discontinuity() -> None:
    source = _SOURCE.replace(
        "indices_1 < cutlass.Int32(length)",
        "cutlass.Int64(2) + cutlass.Int32(2147483646 + cutlass.Int32(vec_lane_1)) < cutlass.Int64(2147483648)",
    ).replace("address = cutlass.Int32(start) + indices_1", "address = indices_1")
    tree, changed = _rewrite(source)
    assert changed == 1
    # The mask is false,false,true,true despite the aligned widened endpoints.
    # Real int32 wrapping followed by int64 promotion is essential to this case.
    with np.errstate(over="ignore"):
        expected, before = _execute(
            ast.parse(source), 0, 16, int32=np.int32, int64=np.int64
        )
        actual, after = _execute(tree, 0, 16, int32=np.int32, int64=np.int64)
    assert before["scalar_stores"] == after["scalar_stores"] == 8
    assert actual == expected
    assert after["vector_stores"] == 0


@pytest.mark.parametrize("root", ["out", "alias"])
def test_tensor_subscripts_preserve_dependent_read_store_order(root: str) -> None:
    source = _SOURCE.replace(
        "for tile_offset_1 in range(0, cutlass.Int32(length), 16):",
        "alias = out\n    for tile_offset_1 in range(0, cutlass.Int32(length), 16):",
    ).replace(
        "(src.iterator + cutlass.Int32(address) * cutlass.Int32(src.layout.stride[0])).load()",
        f"{root}[(cutlass.Int32(address) + 3) % 4]",
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), 0, 4)
    actual, after = _execute(tree, 0, 4)
    # Lanes 1..3 must read the value stored by their preceding lane, even when
    # the input is spelled as a tensor subscript or an alias of the output.
    assert len(set(expected[:4])) == 4
    assert actual == expected
    assert before == after
    assert changed == 0


@pytest.mark.parametrize(
    "replacement",
    [
        "for tile_offset_1 in range(0, 16, 16):\n        tile_offset_1 = cutlass.Int32(1)",
        "for tile_offset_1 in range(0, 16, 16):\n        for tile_offset_1 in range(1, 2):\n            pass",
    ],
)
def test_reassigned_induction_variable_loses_alignment_fact(replacement: str) -> None:
    source = (
        _SOURCE.replace(
            "for tile_offset_1 in range(0, cutlass.Int32(length), 16):", replacement
        )
        .replace("address = cutlass.Int32(start) + indices_1", "address = indices_1")
        .replace("indices_1 < cutlass.Int32(length)", "indices_1 < 4")
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), 0, 4)
    actual, after = _execute(tree, 0, 4)
    assert changed == 1
    assert actual == expected
    assert before["scalar_stores"] == after["scalar_stores"] == 3
    assert after["vector_stores"] == 0


def test_loop_carried_offset_does_not_reuse_outer_definition() -> None:
    source = (
        _SOURCE.replace(
            "for tile_offset_1 in range(0, cutlass.Int32(length), 16):",
            "start = cutlass.Int32(0)\n    for tile_offset_1 in range(0, 32, 16):",
        ).replace("indices_1 < cutlass.Int32(length)", "indices_1 < 32")
        + "        start = cutlass.Int32(1)\n"
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), 0, 32)
    actual, after = _execute(tree, 0, 32)
    assert changed == 1
    assert actual == expected
    assert before["scalar_stores"] == 32
    assert after["vector_stores"] == 4
    assert after["scalar_stores"] == 16


@pytest.mark.parametrize("start", [-0.5, 0.5])
def test_float_arithmetic_before_index_cast_preserves_scalar_addresses(start: float):
    source = _SOURCE.replace(
        "address = cutlass.Int32(start) + indices_1", "address = start + indices_1"
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), start, 4)
    actual, after = _execute(tree, start, 4)
    # Truncation toward zero maps -0.5 + [0, 1, 2, 3] to [0, 0, 1, 2].
    # A final Int32 result does not prove adjacent input lane addresses.
    assert actual == expected
    assert after == before
    assert changed == 0
    assert ast.dump(tree) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("operation", ["//", "%"])
@pytest.mark.parametrize("start", [0, 1, 2])
def test_final_offset_alignment_retains_original_mask_protection(
    operation: str, start: int
):
    source = _SOURCE.replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start != 0",
    )
    for name in ("src", "out"):
        source = source.replace(
            f"cutlass.Int32(address) * cutlass.Int32({name}.layout.stride[0])",
            f"(cutlass.Int32(8 {operation} cutlass.Int32(start)) + cutlass.Int32(indices_1))",
        )
    original = ast.parse(source)
    before_ast = ast.dump(original)
    tree, changed = _rewrite(source)
    expected, before = _execute(original, start, 4)
    actual, after = _execute(tree, start, 4)
    assert ast.dump(original) == before_ast
    assert changed == 1
    assert actual == expected
    if start == 0:
        assert not any(before.values())
        assert not any(after.values())
    else:
        assert before["scalar_loads"] == before["scalar_stores"] == 4
        assert after["vector_loads"] == after["vector_stores"] == 1


@pytest.mark.parametrize("start", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_final_offset_float_conversion_retains_original_mask_protection(start: float):
    source = _SOURCE.replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start == start and start < 100 and start > -100",
    )
    # Keep the conversion inside the masked pointer, so the original program
    # never evaluates it for NaN/infinity. The address assignment is unused.
    for name in ("src", "out"):
        source = source.replace(
            f"cutlass.Int32(address) * cutlass.Int32({name}.layout.stride[0])",
            "(cutlass.Int32(start) + cutlass.Int32(indices_1))",
        )
    source = source.replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), start, 4)
    actual, after = _execute(tree, start, 4)
    assert changed == 1
    assert actual == expected
    if start == 0:
        assert after["vector_loads"] == after["vector_stores"] == 1
    else:
        assert not any(before.values())
        assert not any(after.values())


@pytest.mark.parametrize(
    "mask",
    [
        "cutlass.Int32(start * 4 + indices_1) < 4",
        "indices_1 < cutlass.Int32(start * 4)",
    ],
)
def test_float_arithmetic_cannot_prove_uniform_fragment_masks(mask: str):
    source = _SOURCE.replace("indices_1 < cutlass.Int32(length)", mask).replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    tree, changed = _rewrite(source)
    expected, before = _execute(ast.parse(source), 0.5, 4)
    actual, after = _execute(tree, 0.5, 4)
    assert before["scalar_loads"] == before["scalar_stores"] == 2
    assert actual == expected
    assert after == before
    assert changed == 1


@pytest.mark.parametrize("rebind", [False, True])
def test_integer_parameter_facts_do_not_survive_writes(rebind: bool):
    source = _SOURCE.replace(
        "address = cutlass.Int32(start) + indices_1", "address = start + indices_1"
    )
    if rebind:
        source = source.replace(
            "    for tile_offset_1", "    start = -0.5\n    for tile_offset_1", 1
        )
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    rewriter = _Vectorizer(
        function.body,
        strides={("src", 0): 1, ("out", 0): 1},
        alignments={"src": 16, "out": 16},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))},
        constexpr={},
        integer_parameters=frozenset({"start"}),
    )
    function.body = rewriter.body(function.body)
    tree = ast.fix_missing_locations(tree)
    changed = rewriter.changed
    expected, before = _execute(ast.parse(source), 0, 4)
    actual, after = _execute(tree, 0, 4)
    assert actual == expected
    if rebind:
        assert changed == 0
        assert after == before
        assert ast.dump(tree) == ast.dump(ast.parse(source))
    else:
        assert changed == 1
        assert after["vector_loads"] == after["vector_stores"] == 1


@pytest.mark.parametrize(
    ("typed", "offset"), ((True, 0), (True, 4), (False, 0), (False, 0.5))
)
def test_loaded_integer_offset_preserves_values_and_single_read(
    typed: bool, offset: float
) -> None:
    class OffsetPointer:
        reads = 0

        def __add__(self, index):
            assert index == 0
            return self

        def load(self):
            self.reads += 1
            return offset

    source = _SOURCE.replace(
        "def kernel(src, out, start, length):",
        "def kernel(src, out, start, length, offsets):\n"
        "    start = (offsets.iterator + cutlass.Int32(0)).load()",
    ).replace(
        "address = cutlass.Int32(start) + indices_1", "address = start + indices_1"
    )
    tree, changed = _rewrite(
        source, integer_tensors=frozenset({"offsets"}) if typed else frozenset()
    )
    before_pointer = OffsetPointer()
    after_pointer = OffsetPointer()
    expected, before = _execute(
        ast.parse(source), 0, 17, extra_args=(SimpleNamespace(iterator=before_pointer),)
    )
    actual, after = _execute(
        tree, 0, 17, extra_args=(SimpleNamespace(iterator=after_pointer),)
    )
    assert actual == expected
    assert before_pointer.reads == after_pointer.reads == 1
    assert changed == int(typed)
    if typed:
        assert after["vector_loads"] == after["vector_stores"] == 4
    else:
        assert after == before
        assert ast.dump(tree) == ast.dump(ast.parse(source))


@pytest.mark.parametrize(
    "prefix",
    (
        "offsets = other",
        "offsets.iterator = other.iterator",
        "if flag:\n        offsets = other",
    ),
)
def test_integer_tensor_facts_do_not_survive_binding_mutations(prefix: str) -> None:
    source = _SOURCE.replace(
        "    for tile_offset_1",
        f"    {prefix}\n"
        "    start = (offsets.iterator + cutlass.Int32(0)).load()\n"
        "    for tile_offset_1",
        1,
    ).replace(
        "address = cutlass.Int32(start) + indices_1", "address = start + indices_1"
    )
    tree, changed = _rewrite(source, integer_tensors=frozenset({"offsets"}))
    assert changed == 0
    assert ast.dump(tree) == ast.dump(ast.parse(source))


def test_tensor_integer_kind_does_not_authorize_load_replay_or_casts() -> None:
    node = ast.parse("(offsets.iterator + 1).load()", mode="eval").body
    assert not _integer_expression(node, "lane", frozenset())
    assert _integer_expression(
        node, "lane", frozenset(), integer_tensors=frozenset({"offsets"})
    )
    for source in (
        "(unknown.iterator + 1).load()",
        "reinterpret_float(offsets.iterator).load()",
        "(offsets.iterator + 1).load(dtype=cutlass.Float32)",
    ):
        node = ast.parse(source, mode="eval").body
        assert not _integer_expression(
            node, "lane", frozenset(), integer_tensors=frozenset({"offsets"})
        )


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("fallback", ["cutlass.Int64(0)", "cutlass.Float32(0)"])
def test_predicated_load_kind_preserves_mask_and_load_evaluations(
    active: bool, typed: bool, fallback: str
) -> None:
    class OffsetPointer:
        reads = 0
        predicates = 0

        def __add__(self, index):
            assert index == 0
            return self

        def load(self):
            self.reads += 1
            return 4

        def predicate(self):
            self.predicates += 1
            return active

    source = _SOURCE.replace(
        "def kernel(src, out, start, length):",
        "def kernel(src, out, start, length, offsets, predicate):\n"
        "    start = (offsets.iterator + cutlass.Int32(0)).load() "
        f"if predicate() else {fallback}",
    ).replace(
        "address = cutlass.Int32(start) + indices_1", "address = start + indices_1"
    )
    tree, changed = _rewrite(
        source, integer_tensors=frozenset({"offsets"}) if typed else frozenset()
    )
    before_pointer, after_pointer = OffsetPointer(), OffsetPointer()
    expected, before = _execute(
        ast.parse(source),
        0,
        17,
        extra_args=(SimpleNamespace(iterator=before_pointer), before_pointer.predicate),
    )
    actual, after = _execute(
        tree,
        0,
        17,
        extra_args=(SimpleNamespace(iterator=after_pointer), after_pointer.predicate),
    )
    assert actual == expected
    assert before_pointer.predicates == after_pointer.predicates == 1
    assert before_pointer.reads == after_pointer.reads == int(active)
    assert changed == int(typed and fallback == "cutlass.Int64(0)")
    if changed:
        assert after["vector_loads"] == after["vector_stores"] == 4
    else:
        assert ast.dump(tree) == ast.dump(ast.parse(source))
        assert before == after


def test_integer_conditional_kind_does_not_authorize_predicate_replay() -> None:
    node = ast.parse("1 if predicate() else 0", mode="eval").body
    assert not _integer_expression(node, "lane", frozenset())


def test_successful_vectorization_does_not_mutate_original_lane_loop():
    loop = ast.parse("""
for vec_lane_1 in cutlass.range_constexpr(4):
    value = (src.iterator + cutlass.Int32(vec_lane_1)).load() if vec_lane_1 < length else cutlass.Float32(0)
    if vec_lane_1 < length:
        (out.iterator + cutlass.Int32(vec_lane_1)).store(value)
""").body[0]
    assert isinstance(loop, ast.For)
    before = ast.dump(loop)
    rewriter = _Vectorizer(
        [loop],
        strides={},
        alignments={"src": 16, "out": 16},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))},
        constexpr={},
    )
    assert rewriter.loop(loop, {}, {}, frozenset()) is not None
    assert rewriter.changed == 1
    assert ast.dump(loop) == before
