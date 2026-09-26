from __future__ import annotations

import ast
from dataclasses import replace
from itertools import count
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion import exc
from helion._compiler.cute.collective_register_chain import ProducerLayout
from helion._compiler.cute.collective_register_chain import RegisterChainEpilogue
from helion._compiler.cute.collective_register_chain import RegisterChainSite
from helion._compiler.cute.collective_register_chain import _axis_projection
from helion._compiler.cute.collective_register_chain import _Builder
from helion._compiler.cute.collective_register_chain import _Names
from helion._compiler.cute.collective_register_chain import _packet_alignment
from helion._compiler.cute.collective_register_chain import _packet_plan
from helion._compiler.cute.collective_register_chain import _RegisterCache
from helion._compiler.cute.collective_register_chain import _SharedLayout
from helion._compiler.cute.collective_register_chain import _Transport
from helion._compiler.cute.collective_register_chain import _transport_batches
from helion._compiler.cute.collective_register_chain import _transports
from helion._compiler.cute.collective_register_chain import emit_register_chain
from helion._compiler.cute.collective_register_chain import plan_register_chain
from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.cute.scalar_recipe import ScalarRecipe
from helion._compiler.cute.scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _recipe(source: str, value: str) -> ScalarRecipe:
    statements = ast.parse(source).body
    written = {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    boundaries = {
        node.id
        for statement in [*statements, _expr(value)]
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - written
    recipe = build_recipe(_expr(value), statements, boundaries)
    assert recipe is not None
    return recipe


def _chain() -> tuple[
    tuple[RegisterChainSite, RegisterChainSite],
    RegisterChainEpilogue,
    dict[str, CopyTensorFacts],
]:
    first = RegisterChainSite(
        identity=1,
        bm=16,
        bn=16,
        bk=32,
        dtype="cutlass.BFloat16",
        m_index="m",
        n_index="n",
        k_index="k",
        m_offset="m0",
        n_offset="n0",
        k_offset="k0",
        reduction_iterator=cast(
            "ast.Call", _expr("cutlass.range(0, 32, 32, unroll=1)")
        ),
        static_k_extent=32,
        a_recipe=_recipe("a_value = (left.iterator + m * 32 + k).load()", "a_value"),
        b_recipe=_recipe("b_value = (right.iterator + n * 32 + k).load()", "b_value"),
    )
    second = replace(
        first,
        identity=2,
        bk=64,
        k_offset="k1",
        reduction_iterator=cast(
            "ast.Call", _expr("cutlass.range(0, 64, 64, unroll=1)")
        ),
        static_k_extent=64,
        a_recipe=_recipe(
            """
row_value = (row_scale.iterator + m).load()
column_value = (column_scale.iterator + k).load()
loaded = (middle.iterator + m * 64 + k).load()
scaled = cutlass.Float32(loaded) * cutlass.Float32(column_value)
rounded = cutlass.BFloat16(scaled)
selected = rounded if m >= k else cutlass.BFloat16(0.0)
""",
            "selected",
        ),
        b_recipe=_recipe("b_value = (last.iterator + k * 16 + n).load()", "b_value"),
        seed_recipe=_recipe(
            "scaled_seed = cutlass.Float32(previous * scale)", "scaled_seed"
        ),
        seed_from=1,
        seed_name="previous",
    )
    epilogue = RegisterChainEpilogue(
        m_index="m",
        n_index="n",
        accumulator_name="acc",
        value_recipe=_recipe(
            "residual_value = (residual.iterator + m * 16 + n).load()\n"
            "result = cutlass.BFloat16(acc + cutlass.Float32(residual_value))",
            "result",
        ),
        pointer_recipe=_recipe("", "output.iterator + m * 16 + n"),
        predicate_recipe=_recipe("", "m < rows and n < columns"),
        dtype="cutlass.BFloat16",
        output_tensor="output",
        output_facts=CopyTensorFacts("cutlass.BFloat16", (16, 1), 16),
    )
    facts = {
        name: CopyTensorFacts("cutlass.BFloat16", (width, 1), 16)
        for name, width in (
            ("left", 32),
            ("right", 32),
            ("middle", 64),
            ("last", 16),
            ("residual", 16),
            ("row_scale", 1),
            ("column_scale", 1),
        )
    }
    return (first, second), epilogue, facts


def _fresh() -> Callable[[str], str]:
    sequence = count()
    return lambda hint: f"{hint}_{next(sequence)}"


def _execute_register_b_transpose(words: list[list[int]]) -> list[list[int]]:
    """Execute the actual emitted shuffle loops with inert per-warp banks."""
    builder = _Builder(_Names({"tid"}), "tid", "unused", {}, True, 64)
    layout = ProducerLayout(len(words[0]) * 4, 32, 64, 8)
    statements = builder.register_b(
        layout,
        [],
        _expr("0.0"),
        row="n",
        column="k",
        dtype="cutlass.BFloat16",
        fragment="fragment",
        thread="thread",
        caches={},
        aligned_names={},
    )
    lane_name = next(
        cast("ast.Name", statement.targets[0]).id
        for statement in statements
        if isinstance(statement, ast.Assign)
        and ast.unparse(statement.value) == "tid % 32"
    )

    class Word:
        def __init__(self, bank: list[list[int]], lane: int, index: int) -> None:
            self.bank, self.lane, self.index = bank, lane, index

        @property
        def value(self) -> int:
            return self.bank[self.lane][self.index]

    class Source:
        def __init__(self, bank: list[list[int]], lane: int) -> None:
            self.bank, self.lane = bank, lane

        def __getitem__(self, index: int) -> Word:
            return Word(self.bank, self.lane, index)

    class Destination:
        def __init__(self, row: list[int]) -> None:
            self.row = row

        def __setitem__(self, index: int, value: Word | int) -> None:
            self.row[index] = value.value if isinstance(value, Word) else value

    def shuffle(value: Word, offset: int, *, mask: int, mask_and_clamp: int) -> int:
        assert mask == -1 and mask_and_clamp == 31
        return value.bank[offset][value.index]

    values = words
    phases = 0
    for statement in statements:
        calls = [
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.shuffle_sync"
        ]
        if not isinstance(statement, ast.For) or not calls:
            continue
        assert len(calls) == 1
        source = cast("ast.Name", cast("ast.Subscript", calls[0].args[0]).value).id
        destination = next(
            cast("ast.Name", cast("ast.Subscript", node.targets[0]).value).id
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)
        )
        program = compile(
            ast.fix_missing_locations(ast.Module([statement], [])),
            "<emitted-shuffle-phase>",
            "exec",
        )
        next_values = [[0] * len(words[0]) for _ in range(32)]
        for lane in range(32):
            exec(
                program,
                {
                    lane_name: lane,
                    source: Source(values, lane),
                    destination: Destination(next_values[lane]),
                    "cutlass": SimpleNamespace(range_constexpr=range),
                    "cute": SimpleNamespace(arch=SimpleNamespace(shuffle_sync=shuffle)),
                },
            )
        values = next_values
        phases += 1
    assert phases == 2
    return values


@pytest.mark.parametrize(
    "rows,columns,width",
    [(16, 32, 8), (16, 64, 8), (64, 16, 8), (16, 16, 4), (32, 32, 8)],
)
def test_producer_ownership_covers_each_original_cell_once(
    rows: int, columns: int, width: int
) -> None:
    layout = ProducerLayout(rows, columns, 64, width)
    coordinates = [
        compile(ast.Expression(expr), "<coordinate>", "eval")
        for expr in layout.coordinates("tid", "v")
    ]
    observed = [
        tuple(eval(expr, {"tid": tid, "v": v}) for expr in coordinates)
        for tid in range(64)
        for v in range(layout.values_per_thread)
    ]
    assert len(observed) == len(set(observed)) == rows * columns
    assert set(observed) == {
        (row, column) for row in range(rows) for column in range(columns)
    }


def test_rhs_register_permutation_preserves_every_word_and_warp_group() -> None:
    original = [[lane * 4 + word for word in range(4)] for lane in range(32)]
    actual = _execute_register_b_transpose(original)
    for lane in range(32):
        for word in range(4):
            assert actual[lane][word] == original[(lane & ~3) | word][lane & 3]
    assert sorted(value for row in actual for value in row) == list(range(128))


@pytest.mark.parametrize("rows", [16, 32, 64])
def test_rhs_register_transport_maps_complete_producer_tiles_to_mma_cells(
    rows: int,
) -> None:
    # The independent m16n8k16 B-cell formula supplies the expected consumer;
    # this also checks the repeated N-tile term used with SDK coordinates.
    observed = set()
    for warp in range(2):
        source: list[list[int]] = [[] for _ in range(32)]
        for group in range(rows // 16):
            for lane in range(32):
                tid = 32 * warp + lane
                row = tid // 4 + group * 16
                first_column = (tid % 4) * 8
                source[lane].extend(
                    [
                        (row * 32 + first_column + 2 * word)
                        | ((row * 32 + first_column + 2 * word + 1) << 16)
                        for word in range(4)
                    ]
                )
        destination = _execute_register_b_transpose(source)
        for group in range(rows // 16):
            for lane in range(32):
                row = 8 * warp + lane // 4 + group * 16
                for value in range(8):
                    column = 2 * (lane % 4) + value % 2 + 8 * (value // 2)
                    packed = destination[lane][group * 4 + value // 2]
                    actual = (packed >> (16 * (value % 2))) & 0xFFFF
                    assert actual == row * 32 + column
                    observed.add((row, column))
    assert observed == {(row, column) for row in range(rows) for column in range(32)}


def test_complete_schedule_has_full_fragments_typed_transfers_and_no_shared_fp32_c() -> (
    None
):
    sites, epilogue, facts = _chain()
    before = [ast.dump(site.reduction_iterator) for site in sites]
    plan = plan_register_chain(sites, epilogue, thread_count=64, tensor_facts=facts)
    assert plan.shared_bytes == 4096
    assert plan.b_k_major == (True, False)
    assert plan.b_register_transpose == (True, False)
    assert plan.slice_transports >= 1
    assert plan.matrix_transports == 1
    emitted = emit_register_chain(
        plan, prefix="chain", tid="physical_tid", fresh_name=_fresh()
    )
    source = ast.unparse(ast.Module(emitted, []))
    compile(
        ast.fix_missing_locations(ast.Module(emitted, [])), "<generated-chain>", "exec"
    )
    assert "alloc_smem(cutlass.Uint8, 4096" in source
    assert "alloc_smem(cutlass.Float32" not in source
    assert "partition_shape_A((16, 32))" in source
    assert "partition_shape_A((16, 64))" in source
    assert "partition_C(cute.make_identity_tensor((16, 16)))" in source
    assert "partition_B(cute.make_identity_tensor((16, 32)))" in source
    assert "cute.arch.shuffle_sync" in source
    assert "assumed_align=8" in source
    assert "output.iterator.toint() % 8 == 0" in source
    assert "output.layout.stride[0] == 16" in source
    gemm_loops = [
        node
        for node in ast.walk(ast.Module(emitted, []))
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call) and ast.unparse(child.func) == "cute.gemm"
            for statement in node.body
            for child in ast.walk(statement)
        )
        and "k_fragment" in ast.unparse(node.target)
    ]
    assert len(gemm_loops) == 2
    for loop in gemm_loops:
        assert not any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.copy"
            for node in ast.walk(loop)
        )
    assert [ast.dump(site.reduction_iterator) for site in sites] == before


@pytest.mark.parametrize("integer_type", ["Int32", "Int64", "Uint32", "Uint64"])
def test_typed_packet_iterator_is_emitted_unchanged(integer_type: str) -> None:
    sites, epilogue, facts = _chain()
    typed_sites = tuple(
        replace(
            site,
            reduction_iterator=cast(
                "ast.Call",
                _expr(
                    f"{iterator}(cutlass.{integer_type}(0), "
                    f"cutlass.{integer_type}(cutlass.Int32({site.static_k_extent})), "
                    f"cutlass.{integer_type}({site.bk}){keyword})"
                ),
            ),
        )
        for site, iterator, keyword in zip(
            sites, ("cutlass.range", "range"), (", unroll=1", ""), strict=True
        )
    )
    before = [ast.dump(site.reduction_iterator) for site in typed_sites]
    plan = plan_register_chain(
        typed_sites, epilogue, thread_count=64, tensor_facts=facts
    )
    emitted = emit_register_chain(
        plan, prefix="typed_chain", tid="physical_tid", fresh_name=_fresh()
    )
    iterators = [
        ast.dump(node.iter)
        for node in ast.walk(ast.Module(emitted, []))
        if isinstance(node, ast.For)
    ]
    assert all(iterators.count(iterator) == 1 for iterator in before)
    assert [ast.dump(site.reduction_iterator) for site in typed_sites] == before


@pytest.mark.parametrize("cast_type", ["cutlass.Float32", "opaque_integer"])
def test_rejects_float_or_opaque_packet_bound_cast(cast_type: str) -> None:
    sites, epilogue, facts = _chain()
    first, second = sites
    first = replace(
        first,
        reduction_iterator=cast(
            "ast.Call", _expr(f"cutlass.range(0, cutlass.Int32({cast_type}(32)), 32)")
        ),
    )
    before = ast.dump(first.reduction_iterator)
    with pytest.raises(exc.BackendUnsupported):
        plan_register_chain(
            (first, second), epilogue, thread_count=64, tensor_facts=facts
        )
    assert ast.dump(first.reduction_iterator) == before


@pytest.mark.parametrize(
    "fault",
    [
        "thread_count",
        "dtype",
        "tile",
        "k_tail",
        "k_domain",
        "seed_identity",
        "seed_frontier",
        "different_cell",
        "output_address",
    ],
)
def test_rejects_whole_unsupported_plan_before_emission(fault: str) -> None:
    sites, epilogue, facts = _chain()
    first, second = sites
    threads = 64
    if fault == "thread_count":
        threads = 128
    elif fault == "dtype":
        first = replace(first, dtype="cutlass.TFloat32")
    elif fault == "tile":
        first = replace(first, bm=15)
    elif fault == "k_tail":
        first = replace(
            first,
            static_k_extent=31,
            reduction_iterator=cast("ast.Call", _expr("cutlass.range(0, 31, 32)")),
        )
    elif fault == "k_domain":
        first = replace(
            first,
            reduction_iterator=cast("ast.Call", _expr("cutlass.range(0, 16, 32)")),
        )
    elif fault == "seed_identity":
        second = replace(second, seed_from=99)
    elif fault == "seed_frontier":
        second = replace(second, seed_name="other")
    elif fault == "different_cell":
        second = replace(second, m_offset="other_m0")
    elif fault == "output_address":
        epilogue = replace(
            epilogue, pointer_recipe=_recipe("", "output.iterator + cutlass.Int32(acc)")
        )
    before = ast.dump(
        ast.Module(
            [ast.Expr(first.reduction_iterator), ast.Expr(second.reduction_iterator)],
            [],
        )
    )
    with pytest.raises(exc.BackendUnsupported):
        plan_register_chain(
            (first, second), epilogue, thread_count=threads, tensor_facts=facts
        )
    assert (
        ast.dump(
            ast.Module(
                [
                    ast.Expr(first.reduction_iterator),
                    ast.Expr(second.reduction_iterator),
                ],
                [],
            )
        )
        == before
    )


def test_emitted_trees_and_names_are_independent_of_the_plan_and_each_other() -> None:
    sites, epilogue, facts = _chain()
    plan = plan_register_chain(sites, epilogue, thread_count=64, tensor_facts=facts)
    before = ast.dump(ast.Module(list(plan._statements), []))
    first = emit_register_chain(plan, prefix="one", tid="tid", fresh_name=_fresh())
    second = emit_register_chain(plan, prefix="two", tid="tid", fresh_name=_fresh())
    assert "one_" in ast.unparse(ast.Module(first, []))
    assert "two_" in ast.unparse(ast.Module(second, []))
    first.clear()
    assert ast.dump(ast.Module(list(plan._statements), [])) == before
    assert second


def test_narrow_output_packet_reuses_original_mask_and_checks_real_alignment() -> None:
    recipe = _recipe(
        "",
        "(output.iterator + m * 16 + n).load() if n < limit else cutlass.BFloat16(0)",
    )
    statements, value = recipe.emit({}, _fresh())
    facts = {"output": CopyTensorFacts("cutlass.BFloat16", (16, 1), 8)}
    plan = _packet_plan(
        statements,
        value,
        coordinate="n",
        width=4,
        tensors=facts,
        aligned_names={"m": 1, "n": 4},
    )
    assert plan is not None and plan.width == 4
    emitted = _packet_alignment(plan.emit_from_registers("values", _fresh()), 8)
    source = ast.unparse(ast.Module(emitted, []))
    assert "assumed_align=8" in source and "assumed_align=16" not in source
    assert "n + 3 < limit" in source
    assert "for epilogue_store_lane" in source
    assert ".load(" not in source
    assert (
        _packet_plan(
            statements,
            value,
            coordinate="n",
            width=4,
            tensors={"output": replace(facts["output"], alignment_bytes=4)},
            aligned_names={"m": 1, "n": 4},
        )
        is None
    )
    assert (
        _packet_plan(
            statements,
            value,
            coordinate="n",
            width=4,
            tensors=facts,
            aligned_names={"m": 1, "n": 2},
        )
        is None
    )


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def test_packet_recipe_preserves_masked_read_domain_and_cast_before_select() -> None:
    # Execute only generated scalar recipe statements with inert Python storage.
    # There is no CUDA/SDK call, compiler, kernel binding or numerical corpus.
    recipe = _recipe(
        """
loaded = (source.iterator + m * 64 + k).load() if k < limit else cutlass.Float32(0)
rounded = cutlass.Float32(loaded * cutlass.Float32(0.33333334))
selected = rounded if m >= k else cutlass.Float32(0)
""",
        "selected",
    )
    statements, value = recipe.emit({}, _fresh())
    names = _Names({"m", "k", "source", "limit", "destination", "tid"})
    builder = _Builder(names, "tid", "unused", {}, True, 64)
    loop = builder.packet_loop(
        ProducerLayout(16, 64, 64, 8),
        statements,
        value,
        row="m",
        column="k",
        dtype="cutlass.Float32",
        destination="destination",
    )
    program = compile(
        ast.fix_missing_locations(ast.Module([loop], [])), "<packet-recipe>", "exec"
    )
    reads: list[int] = []

    class Pointer:
        def __init__(self, offset: int = 0) -> None:
            self.offset = offset

        def __add__(self, value: int) -> Pointer:
            return Pointer(self.offset + value)

        def load(self) -> float:
            reads.append(self.offset)
            return _f32(self.offset / 7)

    output: dict[tuple[int, int], float] = {}
    namespace = {
        "cutlass": SimpleNamespace(Int32=int, Float32=_f32, range_constexpr=range),
        "source": SimpleNamespace(iterator=Pointer()),
        "limit": 61,
        "destination": output,
    }
    for tid in range(64):
        exec(program, {**namespace, "tid": tid})
    assert len(output) == 1024
    assert sorted(reads) == [m * 64 + k for m in range(16) for k in range(61)]
    for (m, k), actual in output.items():
        expected = (
            _f32(_f32((m * 64 + k) / 7) * _f32(0.33333334))
            if k < 61 and m >= k
            else _f32(0)
        )
        assert struct.pack("f", actual) == struct.pack("f", expected)


def _execute_shared_packet(
    program: ast.Module, *, pointer_swizzle: bool, threads: int
) -> tuple[dict[int, float], list[int], list[float], list[tuple[int, int]]]:
    """Execute emitted indexing/copies, with swizzling in either SDK location.

    Only storage and the small CuTe descriptor interface are modeled. Every
    recipe, cast, mask, coordinate and packet copy comes from the emitted AST.
    A copy rejects incomplete rmem packets and noncontiguous physical addresses.
    """
    memory: dict[int, float] = {}
    reads: list[int] = []
    casts: list[float] = []
    copies: list[tuple[int, int]] = []

    def bf16(value: float) -> float:
        bits = struct.unpack("I", struct.pack("f", value))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        result = struct.unpack("f", struct.pack("I", rounded))[0]
        casts.append(result)
        return result

    class Layout:
        def __init__(
            self,
            shape: tuple[int, ...],
            *,
            stride: tuple[int, ...],
            swizzle: Callable[[int], int] = lambda value: value,
        ) -> None:
            self.shape, self.stride, self.swizzle = shape, stride, swizzle

        def index(self, coordinate: int | tuple[int, ...]) -> int:
            crd = (coordinate,) if isinstance(coordinate, int) else coordinate
            return self.swizzle(
                sum(a * b for a, b in zip(crd, self.stride, strict=True))
            )

    class Pointer:
        def __init__(
            self,
            base: int = 0,
            offset: int = 0,
            element_bytes: int = 1,
            swizzle: Callable[[int], int] = lambda value: value,
        ) -> None:
            self.base, self.offset = base, offset
            self.element_bytes, self.swizzle = element_bytes, swizzle

        def __add__(self, value: int) -> Pointer:
            return Pointer(
                self.base, self.offset + value, self.element_bytes, self.swizzle
            )

        def address(self, offset: int) -> int:
            return self.base + self.swizzle(self.offset + offset) * self.element_bytes

    class Tensor:
        def __init__(self, pointer: Pointer, layout: Layout) -> None:
            self.iterator, self.layout = pointer, layout

        def __setitem__(self, coordinate: int | tuple[int, ...], value: float) -> None:
            address = self.iterator.address(self.layout.index(coordinate))
            assert address not in memory
            memory[address] = value

    def make_tensor(pointer: Pointer, layout: Layout) -> Tensor:
        if pointer_swizzle and len(layout.shape) == 2:
            # make_view may transfer the composed swizzle into the iterator.
            pointer = Pointer(pointer.address(0), 0, 2, layout.swizzle)
            layout = Layout(layout.shape, stride=layout.stride)
        return Tensor(pointer, layout)

    def recast_ptr(pointer: Pointer, *, dtype: Callable[[float], float]) -> Pointer:
        assert dtype is bf16
        return Pointer(pointer.address(0), 0, 2)

    def make_swizzle(bits: int, base: int, shift: int) -> Callable[[int], int]:
        mask = ((1 << bits) - 1) << (base + shift)
        return lambda offset: offset ^ ((offset & mask) >> shift)

    def composed(swizzle: Callable[[int], int], offset: int, layout: Layout) -> Layout:
        assert offset == 0
        return Layout(layout.shape, stride=layout.stride, swizzle=swizzle)

    def assume(value: int, *, divby: int) -> int:
        assert value % divby == 0
        return value

    def autovec_copy(source: list[float | None], destination: Tensor) -> None:
        assert source and all(value is not None for value in source)
        addresses = [
            destination.iterator.address(destination.layout.index(i))
            for i in range(len(source))
        ]
        assert addresses == list(range(addresses[0], addresses[0] + len(source) * 2, 2))
        assert addresses[0] % (len(source) * 2) == 0
        copies.append((addresses[0], len(source)))
        for i, value in enumerate(source):
            assert value is not None
            destination[i] = value

    class SourcePointer:
        def __init__(self, offset: int = 0) -> None:
            self.offset = offset

        def __add__(self, value: int) -> SourcePointer:
            return SourcePointer(self.offset + value)

        def load(self) -> float:
            reads.append(self.offset)
            return _f32(self.offset / 7)

    namespace = {
        "cute": SimpleNamespace(
            make_tensor=make_tensor,
            recast_ptr=recast_ptr,
            make_layout=Layout,
            make_swizzle=make_swizzle,
            make_composed_layout=composed,
            make_rmem_tensor=lambda shape, dtype: [None] * shape[0],
            crd2idx=lambda coordinate, layout: layout.index(coordinate),
            assume=assume,
            autovec_copy=autovec_copy,
        ),
        "cutlass": SimpleNamespace(
            Int32=int, Float32=_f32, BFloat16=bf16, range_constexpr=range
        ),
        "source": SimpleNamespace(iterator=SourcePointer()),
        "arena": Pointer(),
    }
    code = compile(ast.fix_missing_locations(program), "<shared-packet>", "exec")
    for tid in range(threads):
        exec(code, {**namespace, "tid": tid})
    return memory, reads, casts, copies


@pytest.mark.parametrize("pointer_swizzle", [False, True])
@pytest.mark.parametrize(
    ("rows", "columns", "width", "bits", "transpose"),
    [
        (16, 64, 8, 3, False),
        (16, 32, 8, 2, False),
        (64, 16, 8, 1, True),
        (32, 32, 4, 2, True),
    ],
)
def test_shared_packet_publication_preserves_swizzled_addresses_and_recipe(
    rows: int,
    columns: int,
    width: int,
    bits: int,
    transpose: bool,
    pointer_swizzle: bool,
) -> None:
    producer = ProducerLayout(rows, columns, 64, width)
    shared = _SharedLayout(
        (columns, rows) if transpose else (rows, columns),
        (1, columns) if transpose else (columns, 1),
        (bits, 3, 3),
    )
    recipe = _recipe(
        f"""
loaded = (source.iterator + m * {columns} + k).load() if k < {columns - 3} else cutlass.Float32(0)
rounded = cutlass.BFloat16(cutlass.Float32(loaded * cutlass.Float32(0.33333334)))
selected = rounded if m >= k else cutlass.BFloat16(0)
""",
        "selected",
    )
    statements, value = recipe.emit({}, _fresh())
    before = ast.dump(ast.Module(cast("list[ast.stmt]", statements), []))
    results = []
    for vector in (False, True):
        builder = _Builder(
            _Names({"tid", "m", "k", "source", "destination", "arena"}),
            "tid",
            "arena",
            {},
            True,
            64,
        )
        view = builder.view(
            "destination", "cutlass.BFloat16", "", offset=1024, layout=shared
        )
        if not vector:
            # The same view is emitted, but this producer has no layout proof.
            builder = _Builder(builder.names, "tid", "arena", {}, True, 64)
        loop = builder.packet_loop(
            producer,
            statements,
            value,
            row="m",
            column="k",
            dtype="cutlass.BFloat16",
            destination="destination",
            transpose=transpose,
        )
        results.append(
            _execute_shared_packet(
                ast.Module([*view, loop], []),
                pointer_swizzle=pointer_swizzle,
                threads=64,
            )
        )
    scalar, vector = results
    assert scalar[:3] == vector[:3]  # Complete bytes, ordered reads and typed casts.
    assert len(vector[0]) == rows * columns
    assert sorted(vector[1]) == [
        m * columns + k for m in range(rows) for k in range(columns - 3)
    ]
    assert scalar[3] == [] and len(vector[3]) == rows * columns // width
    assert ast.dump(ast.Module(cast("list[ast.stmt]", statements), [])) == before


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "stride",
        "transpose",
        "swizzle",
        "alignment",
        "row_tail",
        "partition_tail",
        "scalar_width",
        "registers",
    ],
)
def test_shared_packet_store_falls_back_without_complete_proof(fault: str) -> None:
    builder = _Builder(
        _Names({"tid", "m", "k", "destination"}), "tid", "arena", {}, True, 64
    )
    layout = ProducerLayout(16, 64, 64, 8)
    shared = _SharedLayout((16, 64), (64, 1), (3, 3, 3))
    offset = 0
    if fault == "stride":
        shared = replace(shared, stride=(128, 2))
    elif fault == "swizzle":
        shared = replace(shared, swizzle=(3, 2, 3))
    elif fault == "alignment":
        offset = 2
    elif fault == "row_tail":
        layout = replace(layout, columns=62)
        shared = replace(shared, shape=(16, 62), stride=(62, 1))
    elif fault == "partition_tail":
        layout = replace(layout, threads=48)
    elif fault == "scalar_width":
        layout = replace(layout, width=1)
    if fault != "unknown":
        builder.view(
            "destination", "cutlass.BFloat16", "", offset=offset, layout=shared
        )
    loop = builder.packet_loop(
        layout,
        [],
        _expr("m + k"),
        row="m",
        column="k",
        dtype="cutlass.BFloat16",
        destination="destination",
        transpose=fault == "transpose",
        registers=fault == "registers",
    )
    source = ast.unparse(ast.fix_missing_locations(loop))
    assert "autovec_copy" not in source and "make_rmem_tensor" not in source
    assert "destination[" in source


def test_shared_operand_packets_complete_before_existing_mma_barriers() -> None:
    sites, epilogue, facts = _chain()
    plan = plan_register_chain(sites, epilogue, thread_count=64, tensor_facts=facts)
    assert plan.b_register_transpose == (True, False)
    assert plan.shared_bytes == 4096
    reductions = [
        statement
        for statement in plan._statements
        if isinstance(statement, ast.For)
        and ast.unparse(statement.target) in {site.k_offset for site in sites}
    ]
    assert len(reductions) == 2
    for reduction, expected_count in zip(reductions, (1, 2), strict=True):
        publications = [
            index
            for index, statement in enumerate(reduction.body)
            if isinstance(statement, ast.For)
            and ast.unparse(statement.body[-1]).startswith("cute.autovec_copy(")
        ]
        assert len(publications) == expected_count
        after = reduction.body[publications[-1] + 1 :]
        barriers = [
            index
            for index, statement in enumerate(after)
            if ast.unparse(statement) == "cute.arch.sync_threads()"
        ]
        copies = [
            index
            for index, statement in enumerate(after)
            if ast.unparse(statement).startswith("cute.copy(")
        ]
        assert len(barriers) == 2 and copies
        assert barriers[0] < min(copies) <= max(copies) < barriers[1]


def test_every_transfer_has_both_publication_and_retirement_barriers() -> None:
    sites, epilogue, facts = _chain()
    plan = plan_register_chain(sites, epilogue, thread_count=64, tensor_facts=facts)
    source = ast.unparse(ast.Module(list(plan._statements), []))
    assert "transfer_shared" in source and "transfer_values" in source
    # Read the complete phase sequence in each direct statement list. This
    # catches removal or movement of the post-read barrier before shared reuse.
    scopes = [list(plan._statements)]
    scopes.extend(
        node.body
        for statement in plan._statements
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
    )
    found = 0
    for scope in scopes:
        for index, statement in enumerate(scope):
            if isinstance(statement, ast.For) and "transfer_element" in ast.unparse(
                statement.target
            ):
                assert ast.unparse(scope[index - 2]) == "cute.arch.sync_threads()"
                assert ast.unparse(scope[index + 1]) == "cute.arch.sync_threads()"
                found += 1
    assert found == plan.slice_transports + plan.matrix_transports


@pytest.mark.parametrize("rows,columns", [(16, 64), (32, 32), (64, 16), (16, 128)])
def test_axis_cache_projection_preserves_every_consumer_coordinate(
    rows: int, columns: int
) -> None:
    producer = ProducerLayout(rows, columns, 64, 8)
    original = [
        compile(ast.Expression(expr), "<original-coordinate>", "eval")
        for expr in producer.coordinates("tid", "element")
    ]
    for axis, coordinate in (("m", 0), ("k", 1)):
        projection = _axis_projection(producer, (axis,), row="m", column="k")
        assert projection is not None
        index = compile(
            ast.Expression(projection.index("element")), "<cache-index>", "eval"
        )
        representative = compile(
            projection.representative("cached"), "<representative>", "eval"
        )
        for tid in range(64):
            for element in range(producer.values_per_thread):
                cached = eval(index, {"element": element})
                assert 0 <= cached < projection.size
                observed = eval(original[coordinate], {"tid": tid, "element": element})
                retained = eval(
                    original[coordinate],
                    {"tid": tid, "element": eval(representative, {"cached": cached})},
                )
                assert observed == retained
    if (rows, columns) == (16, 64):
        row = _axis_projection(producer, ("m",), row="m", column="k")
        column = _axis_projection(producer, ("k",), row="m", column="k")
        assert row is not None and row.size == 2
        assert column is not None and column.size == 8


@pytest.mark.parametrize(
    "producer,axes",
    [
        (ProducerLayout(16, 1024, 64, 8), ("k",)),  # Columns differ across slots.
        (ProducerLayout(16, 62, 64, 8), ("m",)),  # Packet crosses a row boundary.
        (ProducerLayout(16, 64, 48, 8), ("m",)),  # Incomplete per-thread partition.
        (ProducerLayout(1 << 26, 64, 64, 8), ("k",)),  # Int32 flat arithmetic wraps.
        (ProducerLayout(16, 64, 64, 8), ("m", "k")),
    ],
)
def test_unproved_axis_projection_keeps_the_complete_cache(
    producer: ProducerLayout, axes: tuple[str, ...]
) -> None:
    assert _axis_projection(producer, axes, row="m", column="k") is None


def _axis_transfer(name: str, axis: str) -> _Transport:
    return _Transport(
        name,
        "cutlass.BFloat16",
        (axis,),
        _recipe(f"{name} = ({name}_source.iterator + {axis}).load()", name),
    )


def test_independent_slice_batch_is_disjoint_and_retires_after_all_reads() -> None:
    producer = ProducerLayout(16, 64, 64, 8)
    transfers = (
        _axis_transfer("row_value", "m"),
        _axis_transfer("key", "k"),
        _axis_transfer("dt", "k"),
    )
    builder = _Builder(
        _Names({"tid", "arena", "m", "k"}), "tid", "arena", {}, False, 64
    )
    builder.shared_bytes = 4096
    statements, caches = builder.transport(
        transfers,
        row="m",
        column="k",
        rows=16,
        columns=64,
        consumer_size="16",
        coordinates=lambda element: producer.coordinates("tid", element),
        aligned_names={},
        producer_layout=producer,
    )
    assert builder.shared_bytes == 4096
    assert [entry[2] for entry in builder.shared_views.values()] == [0, 32, 160]
    assert [
        cache.projection.size
        for cache in caches.values()
        if cache.projection is not None
    ] == [2, 8, 8]
    intervals = [
        set(range(offset, offset + size))
        for offset, size in ((0, 32), (32, 128), (160, 128))
    ]
    assert len(set.union(*intervals)) == sum(map(len, intervals)) == 288
    ast.fix_missing_locations(ast.Module(statements, []))
    barriers = [
        i
        for i, statement in enumerate(statements)
        if ast.unparse(statement) == "cute.arch.sync_threads()"
    ]
    assert len(barriers) == 2 and barriers[-1] == len(statements) - 1
    producers = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and "slice_slot" in ast.unparse(statement.target)
    ]
    readers = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and "transfer_element" in ast.unparse(statement.target)
    ]
    assert len(producers) == len(readers) == 3
    assert max(producers) < barriers[0] < min(readers) <= max(readers) < barriers[1]


@pytest.mark.parametrize("fault", ["dependency", "capacity"])
def test_transport_batch_refuses_dependent_or_over_budget_publications(
    fault: str,
) -> None:
    first, second = _axis_transfer("first", "m"), _axis_transfer("second", "k")
    capacity = 128 if fault == "capacity" else 4096
    if fault == "dependency":
        second = replace(second, dependencies=frozenset({"first"}))
    batches = _transport_batches(
        (first, second), row="m", rows=16, columns=64, arena_bytes=capacity
    )
    assert batches == (((first, 0),), ((second, 0),))
    with pytest.raises(exc.BackendUnsupported, match="bounded shared arena"):
        _transport_batches(
            (first,), row="m", rows=65536, columns=64, arena_bytes=capacity
        )


def test_source_load_dependency_prevents_batching_even_with_matching_axis() -> None:
    statements = ast.parse(
        "first = (source.iterator + k).load()\nsecond = (other.iterator + k).load() if first > cutlass.BFloat16(0) else cutlass.BFloat16(0)"
    ).body
    transfers = _transports(
        statements,
        row="m",
        column="k",
        tensors={
            name: CopyTensorFacts("cutlass.BFloat16", (1,), 2)
            for name in ("source", "other")
        },
        forbidden=frozenset(),
        matrices=False,
    )
    assert len(transfers) == 2
    assert "first" in transfers[1].dependencies
    batches = _transport_batches(
        transfers, row="m", rows=16, columns=64, arena_bytes=4096
    )
    assert len(batches) == 2


def test_axis_dag_reuse_keeps_masked_typed_results_and_each_rounding_boundary() -> None:
    # Compare emitted scalar evaluation against the complete per-element cache.
    # Only inert arrays and scalar arithmetic run; no SDK or kernel is invoked.
    producer = ProducerLayout(16, 64, 64, 8)
    recipe = _recipe(
        """
row_word = (row_source.iterator + m).load() if m < 13 else cutlass.BFloat16(0)
key_word = (key_source.iterator + k).load() if k < 61 else cutlass.BFloat16(0)
dt_word = (dt_source.iterator + k).load() if k < 61 else cutlass.BFloat16(0)
row_f32 = cutlass.Float32(row_word)
key_f32 = cutlass.Float32(key_word)
p = 1.44269504
row_key = row_f32 * cutlass.Float32(p)
column_key = key_f32 * cutlass.Float32(p)
delta = row_key - column_key
scale = cute.math.exp2(delta)
loaded = (matrix_source.iterator + m * 64 + k).load() if k < 61 else cutlass.BFloat16(0)
scaled = cutlass.Float32(loaded) * scale
weighted = scaled * cutlass.Float32(dt_word)
rounded = cutlass.BFloat16(weighted)
selected = rounded if m >= k else cutlass.BFloat16(0)
""",
        "selected",
    )
    facts = {
        name: CopyTensorFacts("cutlass.BFloat16", (1,), 2)
        for name in ("row_source", "key_source", "dt_source", "matrix_source")
    }
    builder = _Builder(
        _Names({"m", "k", "tid", "destination"}), "tid", "unused", facts, False, 64
    )
    statements, value = builder.materialize(recipe, {})
    before = ast.dump(ast.Module(statements, []), include_attributes=True)
    transfers = _transports(
        statements,
        row="m",
        column="k",
        tensors=facts,
        forbidden=frozenset(),
        matrices=False,
    )
    assert len(transfers) == 3

    def bf16(value: float) -> float:
        bits = struct.unpack("I", struct.pack("f", value))[0]
        return struct.unpack(
            "f", struct.pack("I", (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000)
        )[0]

    reads: list[tuple[str, int]] = []
    multiplies: list[tuple[float, float]] = []

    class Pointer:
        def __init__(self, name: str, offset: int = 0) -> None:
            self.name, self.offset = name, offset

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.name, self.offset + offset)

        def load(self) -> float:
            reads.append((self.name, self.offset))
            base = {
                "row_source": -0.125,
                "key_source": 0.25,
                "dt_source": 0.5,
                "matrix_source": 1.0,
            }[self.name]
            return bf16(base + self.offset / 1024)

    def multiply(operands: tuple[float, float], **kwargs: object) -> float:
        assert kwargs == {
            "asm": "mul.rn.f32 $0, $1, $2;",
            "constraints": "=f,f,f",
            "dtype": _f32,
            "is_pure": True,
        }
        multiplies.append(operands)
        return _f32(operands[0] * operands[1])

    namespace = {
        "cutlass": SimpleNamespace(
            Int32=int, Float32=_f32, BFloat16=bf16, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda shape, dtype: [None] * shape[0],
            math=SimpleNamespace(exp2=lambda v: _f32(2.0**v)),
        ),
        "_cute_inline_asm_elementwise": multiply,
        **{name: SimpleNamespace(iterator=Pointer(name)) for name in facts},
    }
    observed = []
    for compressed in (False, True):
        caches = {
            transfer.target: _RegisterCache(
                f"cache_{i}",
                transfer.dtype,
                _axis_projection(producer, transfer.axes, row="m", column="k")
                if compressed
                else None,
            )
            for i, transfer in enumerate(transfers)
        }
        setup, replay, used_caches = builder.cache_axis_recipes(
            producer, statements, value, column="k", caches=caches
        )
        assert bool(setup) is compressed
        loop = builder.packet_loop(
            producer,
            replay,
            value,
            row="m",
            column="k",
            dtype="cutlass.BFloat16",
            destination="destination",
            caches=used_caches,
        )
        program = compile(
            ast.fix_missing_locations(ast.Module([*setup, loop], [])),
            "<axis-reuse>",
            "exec",
        )
        output: dict[tuple[int, int], float] = {}
        matrix_reads: list[tuple[str, int]] = []
        key_multiplies = 0
        for tid in range(64):
            values: dict[str, list[float]] = {}
            for transfer in transfers:
                cache = caches[transfer.target]
                projection = cache.projection
                count = 16 if projection is None else projection.size
                transfer_statements, transfer_value = transfer.recipe.emit({}, _fresh())
                transfer_program = compile(
                    ast.fix_missing_locations(
                        ast.Module(
                            [
                                *transfer_statements,
                                ast.Assign(
                                    [ast.Name("result", ast.Store())], transfer_value
                                ),
                            ],
                            [],
                        )
                    ),
                    "<original-transfer-recipe>",
                    "exec",
                )
                slots = []
                for item in range(count):
                    element = (
                        item
                        if projection is None
                        else eval(projection.representative("item"), {"item": item})
                    )
                    m, k = [
                        eval(
                            compile(ast.Expression(expr), "<coordinate>", "eval"),
                            {"tid": tid, "element": element},
                        )
                        for expr in producer.coordinates("tid", "element")
                    ]
                    scope = {**namespace, "m": m, "k": k}
                    exec(transfer_program, scope)
                    slots.append(cast("float", scope["result"]))
                values[cache.name] = slots
            reads.clear()
            multiplies.clear()
            exec(program, {**namespace, **values, "tid": tid, "destination": output})
            matrix_reads.extend(reads)
            key_multiplies += sum(right == _f32(1.44269504) for _, right in multiplies)
        observed.append((output, matrix_reads, key_multiplies))
    assert observed[0][:2] == observed[1][:2]
    assert len(observed[1][0]) == 1024
    assert observed[0][2] == 64 * 16 * 2
    assert observed[1][2] == 64 * (2 + 8)
    assert sorted(observed[1][1]) == [
        ("matrix_source", m * 64 + k) for m in range(16) for k in range(61)
    ]
    assert ast.dump(ast.Module(statements, []), include_attributes=True) == before


@pytest.mark.parametrize(
    "fault",
    [
        "mixed_axis",
        "coordinate",
        "unrounded",
        "unknown",
        "conditional",
        "rebound",
        "rounding",
        "side_effect",
        "other_layout",
        "dtype",
    ],
)
def test_axis_dag_reuse_rejects_missing_type_dependency_or_lifetime_proof(
    fault: str,
) -> None:
    producer = ProducerLayout(16, 64, 64, 8)
    projection = _axis_projection(producer, ("k",), row="m", column="k")
    row_projection = _axis_projection(producer, ("m",), row="m", column="k")
    assert projection is not None and row_projection is not None
    caches = {
        "key": _RegisterCache("column_cache", "cutlass.BFloat16", projection),
        "row": _RegisterCache("row_cache", "cutlass.BFloat16", row_projection),
    }
    left = "cutlass.Float32(key)"
    right = "cutlass.Float32(1.44269504)"
    if fault == "mixed_axis":
        right = "cutlass.Float32(row)"
    elif fault == "coordinate":
        left = "cutlass.Float32(k)"
    elif fault == "dtype":
        left = "cutlass.Float64(key)"
    expression = f"_cute_inline_asm_elementwise(({left}, {right}), asm='mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', dtype=cutlass.Float32, is_pure=True)"
    if fault == "unrounded":
        expression = f"{left} * {right}"
    elif fault == "unknown":
        expression = f"opaque({left})"
    elif fault == "rounding":
        expression = expression.replace("mul.rn", "mul.rz")
    elif fault == "side_effect":
        expression = expression.replace("is_pure=True", "is_pure=False")
    elif fault == "other_layout":
        caches["key"] = replace(
            caches["key"],
            projection=replace(projection, producer=replace(producer, columns=32)),
        )
    statements = ast.parse(
        f"key = original_column_load()\nrow = original_row_load()\nresult = {expression}"
    ).body
    if fault == "conditional":
        statements[-1] = ast.If(_expr("predicate"), [statements[-1]], [])
    elif fault == "rebound":
        statements.insert(
            2, ast.Assign([ast.Name("key", ast.Store())], _expr("other_value"))
        )
    before = ast.dump(ast.Module(statements, []), include_attributes=True)
    builder = _Builder(_Names({"m", "k", "tid"}), "tid", "arena", {}, False, 64)
    setup, replay, retained = builder.cache_axis_recipes(
        producer, statements, _expr("result"), column="k", caches=caches
    )
    assert not setup and retained == caches
    assert ast.dump(ast.Module(replay, []), include_attributes=True) == before
    assert ast.dump(ast.Module(statements, []), include_attributes=True) == before


def test_terminal_arena_proof_removes_only_the_last_retirement_barrier() -> None:
    sites, epilogue, facts = _chain()
    retained = plan_register_chain(sites, epilogue, thread_count=64, tensor_facts=facts)
    terminal = plan_register_chain(
        sites, epilogue, thread_count=64, tensor_facts=facts, terminal_arena_dead=True
    )
    assert ast.unparse(retained._statements[-1]) == "cute.arch.sync_threads()"
    assert [ast.dump(node) for node in retained._statements[:-1]] == [
        ast.dump(node) for node in terminal._statements
    ]
