from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from itertools import product
from unittest.mock import patch

import pytest
import torch

from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401

import helion
from helion._compiler.cute.row_matrix_transport import ROW_MATRIX_BARRIERS
from helion._compiler.cute.row_matrix_transport import ROW_MATRIX_MODE
from helion._compiler.cute.row_matrix_transport import RowMatrixLayout
from helion._compiler.cute.row_matrix_transport import prove_row_matrix_transport
from helion._compiler.cute.row_matrix_transport_codegen import _contraction
from helion._compiler.cute.row_matrix_transport_codegen import row_matrix_module
from helion._compiler.cute.row_resident import ROW_RESIDENT_KEY
from helion._compiler.cute.row_resident import ROW_RESIDENT_MODES
from helion._compiler.cute.row_resident import prove_row_resident
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl


def materialized_pair(left, hidden, expansion, auxiliary):
    rows, inner = left.size()
    middle_width = hidden.size(1)
    columns = expansion.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    middle = torch.empty((rows, middle_width), dtype=left.dtype, device=left.device)
    carrier = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row in hl.tile(rows):
        for column in hl.tile(middle_width):
            first = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(inner):
                first = torch.addmm(
                    first, left[row, reduction], hidden[reduction, column]
                )
            middle[row, column] = torch.relu(first)
        for column in hl.tile(columns):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            carrier[row, column] = torch.sigmoid(value)
            result[row, column] = auxiliary[row, column] * carrier[row, column]
    return result, middle, carrier


def linear_pair(left, hidden, expansion, auxiliary):
    rows, inner = left.size()
    middle_width = hidden.size(1)
    columns = expansion.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    middle = torch.empty((rows, middle_width), dtype=left.dtype, device=left.device)
    carrier = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row in hl.tile(rows):
        for column in hl.tile(middle_width):
            first = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(inner):
                first = torch.addmm(
                    first, left[row, reduction], hidden[reduction, column]
                )
            middle[row, column] = -first
        for column in hl.tile(columns):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            carrier[row, column] = -value
            result[row, column] = carrier[row, column] * auxiliary[row, column]
    return middle, carrier, result


def _bind(function=materialized_pair, *, rows=17, offset=0, dimensions=(32, 64, 128)):
    k, n0, n1 = dimensions
    auxiliary = torch.empty(rows * n1 + offset, dtype=torch.float16)[offset:].view(
        rows, n1
    )
    arguments = (
        torch.empty((rows, k), dtype=torch.float16),
        torch.empty((k, n0), dtype=torch.float16),
        torch.empty((n0, n1), dtype=torch.float16),
        auxiliary,
    )
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
    )
    return kernel._bind_isolated(arguments)


def _plan(bound):
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        row = prove_row_resident(bound.env, host)
        if row is None:
            return None
        return prove_row_matrix_transport(bound.env, host, row)


def _config(bound, mode=ROW_MATRIX_MODE):
    return helion.Config.from_dict(
        deepcopy(bound.config_spec.default_config().config)
        | {ROW_RESIDENT_KEY: mode, "cute_min_blocks_per_mp": 1}
    )


def _device(source):
    return next(
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "def _helion_row_resident(" in node.args[0].value
    )


def _ldmatrix(memory, row_addresses, lane, matrix, *, transpose=False):
    # PTX m8n8.b16: address lanes [8*j,8*j+8) supply matrix j's rows.
    # A destination lane owns two adjacent columns (or rows for .trans).
    return tuple(
        memory[
            row_addresses[
                8 * matrix + (2 * (lane % 4) + half if transpose else lane // 4)
            ]
            + 2 * (lane // 4 if transpose else 2 * (lane % 4) + half)
        ]
        for half in range(2)
    )


def _stmatrix(memory, owners, row_addresses, thread, words):
    lane = thread % 32
    for matrix, word in enumerate(words):
        for half, value in enumerate(word):
            address = row_addresses[8 * matrix + lane // 4] + 2 * (
                2 * (lane % 4) + half
            )
            if address in memory:
                assert memory[address] == value
            memory[address] = value
            owners[address].append(thread)


@pytest.mark.parametrize(
    ("columns", "reduction"), tuple(product((64, 128, 256), (32, 64, 128, 256)))
)
def test_complete_producers_and_mma_fragment_values(columns, reduction):
    layout = RowMatrixLayout.create(columns, reduction)
    assert layout is not None
    a, a_owners = {}, defaultdict(list)
    for warp in range(reduction // 32):
        addresses = [
            layout.linear_matrix_address(warp * 32 + lane, reduction)
            for lane in range(32)
        ]
        assert all(address % 16 == 0 for address in addresses)
        for lane in range(32):
            thread = warp * 32 + lane
            word = tuple(
                ("a", *divmod(2 * thread + half, reduction)) for half in range(2)
            )
            _stmatrix(a, a_owners, addresses, thread, (word,))
    assert len(a) == 2 * reduction
    assert all(len(owners) == 1 for owners in a_owners.values())
    assert set(a) == set(range(0, 4 * reduction, 2))

    b = {}
    for packet in range(layout.weight_packets):
        for thread in range(256):
            k, n = divmod(thread * 8 + packet * 2048, columns)
            address = layout.weight_offset(k, n)
            assert address % 16 == 0
            for half in range(8):
                assert address + 2 * half == layout.weight_offset(k, n + half)
                assert address + 2 * half not in b
                b[address + 2 * half] = ("b", k, n + half)
    assert set(b) == set(range(0, 2 * columns * reduction, 2))

    for warp in range(8):
        for pair in range(reduction // 32):
            a_addresses = [
                layout.a_matrix_address(warp * 32 + lane, pair) for lane in range(32)
            ]
            for packet in range(layout.column_packets):
                b_addresses = [
                    layout.b_matrix_address(warp * 32 + lane, pair, packet)
                    for lane in range(32)
                ]
                assert all(
                    address % 16 == 0 for address in (*a_addresses, *b_addresses)
                )
                for step in range(2):
                    mma_a, mma_b = {}, {}
                    for lane in range(32):
                        a_words = [
                            _ldmatrix(a, a_addresses, lane, step * 2 + j)
                            for j in range(2)
                        ]
                        b_words = [
                            _ldmatrix(
                                b, b_addresses, lane, step * 2 + j, transpose=True
                            )
                            for j in range(2)
                        ]
                        # Published m16n8k16 A/B register layout, independently
                        # of the transport's coordinate formulas.
                        for j, word in enumerate(
                            (a_words[0], a_words[0], a_words[1], a_words[1])
                        ):
                            for half, value in enumerate(word):
                                mma_a[
                                    lane // 4 + 8 * (j % 2),
                                    2 * (lane % 4) + 8 * (j // 2) + half,
                                ] = value
                        for j, word in enumerate(b_words):
                            for half, value in enumerate(word):
                                mma_b[2 * (lane % 4) + 8 * j + half, lane // 4] = value
                    k_start = pair * 32 + step * 16
                    assert mma_a == {
                        (m, k): ("a", m % 2, k_start + k)
                        for m in range(16)
                        for k in range(16)
                    }
                    assert mma_b == {
                        (k, n): ("b", k_start + k, warp * 8 + packet * 64 + n)
                        for k in range(16)
                        for n in range(8)
                    }


@pytest.mark.parametrize("columns", (64, 128, 256))
def test_redistribution_alias_equality_and_unique_global_owners(columns):
    layout = RowMatrixLayout.create(columns, 64)
    assert layout is not None
    middle = {}
    final, owners = {}, defaultdict(list)
    for warp in range(8):
        addresses = [
            layout.result_matrix_address(warp * 32 + lane) for lane in range(32)
        ]
        assert all(
            0 <= address < 4 * columns and address % 16 == 0 for address in addresses
        )
        for lane in range(32):
            thread = warp * 32 + lane
            words = []
            for packet in range(layout.column_packets):
                word = []
                for half in range(2):
                    row, column = layout.result_coordinate(thread, packet, half)
                    # All physical replicas share the same positive-zero
                    # seed, ordered K terms, and typed per-element recipe.
                    value = ("RN16", tuple((row, k, column) for k in range(64)))
                    word.append(value)
                    if lane < 8:
                        address = layout.middle_offset(row, column)
                        assert address not in middle
                        middle[address] = value
                words.append(tuple(word))
            _stmatrix(final, owners, addresses, thread, words)
    assert len(middle) == len(final) == 2 * columns
    assert set(middle) == set(final) == set(range(0, 4 * columns, 2))
    assert all(
        len(group) == 4 and len({thread // 32 for thread in group}) == 1
        for group in owners.values()
    )
    global_owners = {}
    for warp in range(columns // 32):
        addresses = [
            layout.linear_matrix_address(warp * 32 + lane, columns)
            for lane in range(32)
        ]
        for lane in range(32):
            thread = warp * 32 + lane
            word = _ldmatrix(final, addresses, lane, 0)
            for half, value in enumerate(word):
                row, column = divmod(2 * thread + half, columns)
                assert value == middle[layout.middle_offset(row, column)]
                assert (row, column) not in global_owners
                global_owners[row, column] = thread
    assert len(global_owners) == 2 * columns
    # Both scalar middle transport and packed final transport retain only
    # actual rows, including an entirely inactive physical row at odd M.
    for rows in (1, 2, 3, 17, 256, 257):
        coordinates = [
            (start + row, column)
            for start in range(0, rows, 2)
            for row, column in global_owners
            if start + row < rows
        ]
        assert len(coordinates) == len(set(coordinates)) == rows * columns


@pytest.mark.parametrize(
    ("columns", "reduction"),
    ((32, 64), (96, 64), (512, 64), (64, 16), (64, 48), (64, 512), (True, 64)),
)
def test_unsupported_matrix_geometries(columns, reduction):
    assert RowMatrixLayout.create(columns, reduction) is None


@pytest.mark.parametrize("function", (materialized_pair, linear_pair))
@skipUnlessBackends(["cute"])
def test_typed_admission_additive_config_and_complete_source(function):
    bound = _bind(function)
    found = _plan(bound)
    assert found is not None, [
        str(graph.graph) for graph in bound.host_function.device_ir.graphs
    ]
    assert found.materialized_is_left_operand is (function is linear_pair)
    assert found.shared_bytes == 16640
    spec = bound.config_spec
    assert spec.cute_materialized_schedule_choices == (
        *ROW_RESIDENT_MODES,
        ROW_MATRIX_MODE,
    )
    group = next(
        group
        for group in spec.compiler_coverage_groups
        if group.key == ROW_RESIDENT_KEY
    )
    assert group.domain == (*ROW_RESIDENT_MODES, ROW_MATRIX_MODE)
    config = _config(bound)
    generation = spec.create_config_generation()
    _, effective = generation.strict_config_pair(config)
    assert effective.config[ROW_RESIDENT_KEY] == ROW_MATRIX_MODE
    source = bound.to_code(config)
    device = _device(source)
    assert device == row_matrix_module(found, 1)
    assert device.count("cute.arch.sync_threads()") == len(ROW_MATRIX_BARRIERS) == 8
    assert "block=(256, 1, 1)" in source and "(9, 1, 1)" in source
    assert (
        "_row_product(auxiliary_word, materialized_word)" in device
        if function is materialized_pair
        else "_row_product(materialized_word, auxiliary_word)" in device
    )
    assert "cute.arch.cp_async" not in device
    assert (
        "cute.math.exp" in device
        if function is materialized_pair
        else "cute.math.exp" not in device
    )
    assert ROW_RESIDENT_KEY not in spec.default_config().config
    ordinary = bound.to_code(spec.default_config())
    assert bound.to_code(_config(bound, "off")) != source
    assert "def _helion_row_resident(" not in ordinary
    # Each independent accumulator sees every K packet exactly once in its
    # original increasing order, with no partial cast or extra K split.
    for index, layout in enumerate(found.layouts):
        tree = ast.parse(_contraction(found, index))
        calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_row_mma"
        ]
        assert len(calls) == layout.reduction // 16 * layout.column_packets
        for step in range(layout.reduction // 16):
            for packet in range(layout.column_packets):
                call = calls[step * layout.column_packets + packet].value
                assert (
                    ast.unparse(call.args[0])
                    == f"s{index}_a{step // 2}[{step % 2 * 2}]"
                )
                assert (
                    ast.unparse(call.args[2])
                    == f"s{index}_b{packet}_{step // 2}[{step % 2 * 2}]"
                )


@skipUnlessBackends(["cute"])
def test_alignment_architecture_and_late_graph_rejection():
    unaligned = _bind(offset=1)
    assert unaligned.config_spec.cute_materialized_schedule_available
    assert not unaligned.config_spec.cute_row_matrix_transport_available
    assert (
        unaligned.config_spec.cute_materialized_schedule_choices == ROW_RESIDENT_MODES
    )
    assert _plan(unaligned) is None
    with pytest.raises(InvalidConfig, match=ROW_RESIDENT_KEY):
        unaligned.config_spec.normalized_config(_config(unaligned))
    # A four-byte residue is sufficient; do not require unused 16-byte aux
    # alignment. Original contraction vector operands still require 16.
    aligned = _bind(offset=2)
    assert _plan(aligned) is not None
    assert aligned.config_spec.cute_row_matrix_transport_available
    with patch.object(aligned.config_spec, "target_device_capability", (9, 0)):
        assert _plan(aligned) is not None
    with patch.object(aligned.config_spec, "target_device_capability", (8, 0)):
        assert _plan(aligned) is None
        with pytest.raises(InvalidConfig, match="sm90"):
            aligned.to_code(_config(aligned))
    found = _plan(aligned)
    assert found is not None
    before = found.product_value.target
    try:
        found.product_value.target = torch.ops.aten.add.Tensor
        assert _plan(aligned) is None
        with pytest.raises(InvalidConfig, match="materialized FP16"):
            aligned.to_code(_config(aligned))
    finally:
        found.product_value.target = before
    before_meta = found.product_value.meta["val"]
    try:
        # An FP32 product followed by a final half cast is a different typed
        # recipe and must not use the direct half product admission.
        found.product_value.meta["val"] = before_meta.to(torch.float32)
        assert _plan(aligned) is None
    finally:
        found.product_value.meta["val"] = before_meta


class _HalfPointer:
    def __init__(self, byte_offset):
        self.byte_offset = byte_offset

    def __add__(self, half_offset):
        return _HalfPointer(self.byte_offset + 2 * half_offset)

    def toint(self):
        return self.byte_offset


@skipUnlessBackends(["cute"])
def test_emitted_addresses_are_the_proved_layout_expressions():
    found = _plan(_bind(rows=3, dimensions=(256, 64, 256)))
    assert found is not None
    tree = ast.parse(row_matrix_module(found))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion_row_resident"
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    stores = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_row_matrix_store"
    ]
    assert len(stores) == 2
    # AST traversal reaches the unconditional final store before the nested
    # input store. Classify by their tuple-valued word operand, not traversal.
    input_store = next(
        node for node in stores if ast.unparse(node.args[1]) == "(input_word,)"
    )
    output_store = next(node for node in stores if node is not input_store)
    evaluations = []
    for stage, layout in enumerate(found.layouts):
        for pair in range(layout.reduction // 32):
            call = assignments[f"s{stage}_a{pair}"]
            assert isinstance(call, ast.Call)
            evaluations.append(
                (
                    call.args[0],
                    65536,
                    lambda tid, layout=layout, pair=pair: layout.a_matrix_address(
                        tid, pair
                    ),
                    256,
                )
            )
            for packet in range(layout.column_packets):
                call = assignments[f"s{stage}_b{packet}_{pair}"]
                assert isinstance(call, ast.Call)
                evaluations.append(
                    (
                        call.args[0],
                        0,
                        lambda tid, layout=layout, pair=pair, packet=packet: (
                            layout.b_matrix_address(tid, pair, packet)
                        ),
                        256,
                    )
                )
        packets = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_row_shared_packet"
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id.startswith(f"s{stage}_weight_")
        ]
        assert len(packets) == layout.weight_packets
        for packet, call in enumerate(packets):
            evaluations.append(
                (
                    call.args[0],
                    0,
                    lambda tid, layout=layout, packet=packet: layout.weight_offset(
                        *divmod(tid * 8 + packet * 2048, layout.columns)
                    ),
                    256,
                )
            )
    evaluations.extend(
        (
            (
                input_store.args[0],
                65536,
                lambda tid: found.layouts[0].linear_matrix_address(
                    tid, found.layouts[0].reduction
                ),
                found.layouts[0].reduction,
            ),
            (output_store.args[0], 0, found.layouts[1].result_matrix_address, 256),
        )
    )
    final_word = assignments["final_word"]
    assert isinstance(final_word, ast.Subscript) and isinstance(
        final_word.value, ast.Call
    )
    evaluations.append(
        (
            final_word.value.args[0],
            0,
            lambda tid: found.layouts[1].linear_matrix_address(
                tid, found.layouts[1].columns
            ),
            found.layouts[1].columns,
        )
    )
    for expression, base, expected, threads in evaluations:
        code = compile(ast.Expression(expression), "<emitted-address>", "eval")
        for tid in range(threads):
            actual = eval(
                code,
                {
                    "tid": tid,
                    "lane": tid % 32,
                    "warp": tid // 32,
                    "arena_ptr": _HalfPointer(0),
                    "operand_a_ptr": _HalfPointer(65536),
                },
            )
            assert actual == base + expected(tid)


@skipUnlessBackends(["cute"])
def test_resource_limit_keeps_previous_path():
    bound = _bind(dimensions=(256, 128, 64))
    assert not bound.config_spec.cute_materialized_schedule_available
    assert not bound.config_spec.cute_row_matrix_transport_available
    with pytest.raises(InvalidConfig, match=ROW_RESIDENT_KEY):
        bound.config_spec.normalized_config(_config(bound))
