"""CPU checks for semantic matching and exact packed-carrier composition."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph

from helion._compiler.autotuner_heuristics.cute_signed_bitfield import (
    add_signed_bitfield_seeds,
)
from helion._compiler.cute import signed_bitfield as sb
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.tile_strategy import DeviceGridState
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
from helion.language import memory_ops
from helion.runtime.config import Config


def _node(graph, target, args, dtype, kwargs=None):
    node = graph.call_function(target, args, kwargs or {})
    node.meta["val"] = torch.empty((2, 8), dtype=dtype)
    return node


def _graph(offset=0, width=4, *, identity=False, shift_only=False, wide=torch.int32):
    graph = Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty((2, 8), dtype=torch.int8)
    output = graph.placeholder("destination")
    output.meta["val"] = torch.empty((2, 8), dtype=torch.bfloat16)
    load = _node(graph, memory_ops.load, (source, [0, 0]), torch.int8)
    current = load
    if offset:
        current = _node(
            graph,
            torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
            (current, offset),
            torch.int8,
        )
    if not identity and not shift_only:
        # The original INT4 source masks in INT8 and then widens.
        if width == 8:
            current = _node(
                graph,
                torch.ops.aten._to_copy.default,
                (current,),
                wide,
                {"dtype": wide},
            )
        current = _node(
            graph,
            torch.ops.aten.bitwise_and.Scalar,
            (current, (1 << width) - 1),
            current.meta["val"].dtype,
        )
    current = _node(
        graph, torch.ops.aten._to_copy.default, (current,), wide, {"dtype": wide}
    )
    if not identity and not shift_only:
        compare = _node(
            graph, torch.ops.aten.ge.Scalar, (current, 1 << (width - 1)), torch.bool
        )
        subtract = _node(graph, torch.ops.aten.sub.Tensor, (current, 1 << width), wide)
        current = _node(
            graph, torch.ops.aten.where.self, (compare, subtract, current), wide
        )
    value = _node(
        graph,
        torch.ops.aten._to_copy.default,
        (current,),
        torch.bfloat16,
        {"dtype": torch.bfloat16},
    )
    store = _node(graph, memory_ops.store, (output, [0, 0], value), torch.bfloat16)
    graph.output(())
    return SimpleNamespace(
        graph=graph, source=source, output=output, load=load, value=value, store=store
    )


@pytest.mark.parametrize(
    "offset,width", [(o, w) for w in range(1, 9) for o in range(9 - w)]
)
def test_contained_fields_match(offset, width):
    case = _graph(offset, width)
    field = sb.match_signed_byte_field(case.value)
    assert field is not None
    assert (field.load, field.bit_offset, field.bit_width) == (case.load, offset, width)
    assert field.signed and sb.has_signed_byte_field(case.graph)


@pytest.mark.parametrize("offset", range(8))
def test_identity_and_arithmetic_high_fields(offset):
    case = _graph(offset, 8 - offset, identity=(offset == 0), shift_only=offset != 0)
    field = sb.match_signed_byte_field(case.value)
    assert field is not None and (field.bit_offset, field.bit_width) == (
        offset,
        8 - offset,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "threshold",
        "subtrahend",
        "alpha",
        "wrong_compare",
        "unsigned_load",
        "unsigned_cast",
        "narrow_wrap",
        "float_intermediate",
        "float_destination",
        "extra_arithmetic",
        "dynamic_shift",
        "cross_byte",
        "unproved_alias",
        "different_raw",
        "mask255_int8",
        "graph_copy",
    ],
)
def test_semantic_near_misses_decline(fault):
    case = _graph(1, 4)
    nodes = list(case.graph.nodes)
    compare = next(n for n in nodes if n.target is torch.ops.aten.ge.Scalar)
    subtract = next(n for n in nodes if n.target is torch.ops.aten.sub.Tensor)
    shift = next(
        n for n in nodes if n.target is torch.ops.aten.bitwise_right_shift.Tensor_Scalar
    )
    masked = next(n for n in nodes if n.target is torch.ops.aten.bitwise_and.Scalar)
    if fault == "threshold":
        compare.args = (compare.args[0], 7)
    elif fault == "subtrahend":
        subtract.args = (subtract.args[0], 15)
    elif fault == "alpha":
        subtract.kwargs = {"alpha": 2}
    elif fault == "wrong_compare":
        compare.target = torch.ops.aten.gt.Scalar
    elif fault == "unsigned_load":
        case.source.meta["val"] = torch.empty((2, 8), dtype=torch.uint8)
    elif fault == "unsigned_cast":
        source = case.value.args[0]
        source.meta["val"] = torch.empty((2, 8), dtype=torch.uint8)
    elif fault == "narrow_wrap":
        subtract.meta["val"] = torch.empty((2, 8), dtype=torch.int8)
    elif fault == "float_intermediate":
        source = compare.args[0]
        source.meta["val"] = torch.empty((2, 8), dtype=torch.float32)
        source.kwargs = {"dtype": torch.float32}
    elif fault == "float_destination":
        case.value.meta["val"] = torch.empty((2, 8), dtype=torch.float16)
        case.value.kwargs = {"dtype": torch.float16}
    elif fault == "extra_arithmetic":
        with case.graph.inserting_before(case.value):
            extra = _node(
                case.graph,
                torch.ops.aten.mul.Tensor,
                (case.value.args[0], 2),
                torch.int32,
            )
        case.value.args = (extra,)
    elif fault == "dynamic_shift":
        shift.args = (shift.args[0], compare.args[0])
    elif fault == "cross_byte":
        shift.args = (shift.args[0], 5)
    elif fault == "unproved_alias":
        shift.target = torch.ops.aten.alias.default
        shift.args = (shift.args[0],)
    elif fault == "different_raw":
        subtract.args = (masked, 16)
    elif fault == "mask255_int8":
        masked.args = (masked.args[0], 255)
    elif fault == "graph_copy":
        replacement = _graph()
        shift.args = (replacement.load, 1)
    assert sb.match_signed_byte_field(case.value) is None


class _U16(int):
    def __new__(cls, value):
        return int.__new__(cls, value & 0xFFFF)


class _U32(int):
    def __new__(cls, value):
        return int.__new__(cls, value & 0xFFFFFFFF)


def _bf16(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _held_helper():
    """Execute only the helper's Python template with a strict PTX interpreter.

    This checks its actual masks/selectors/grouping, without SDK/IR/native code.
    The archived exhaustive field algebra is reused, not re-enumerated here.
    """
    path = Path(sb.__file__).with_name("vec_utils.py")
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "signed_bitfield_to_bf16_packed"
    )
    calls = []

    def interpreter(code, *, write_only_types, read_only_args):
        calls.append(code)
        registers = {"{$r0}": int(read_only_args[0])}

        def value(token):
            return registers[token] if token in registers else int(token, 0)

        for part in code.removeprefix("{").removesuffix("}").split(";"):
            part = part.strip()
            if not part or part.startswith(".reg "):
                continue
            opcode, operands = part.split(" ", 1)
            args = [s.strip() for s in operands.split(",")]
            dst = args[0]
            if opcode == "prmt.b32":
                src = value(args[1]) | (value(args[2]) << 32)
                selector = value(args[3])
                result = 0
                for i in range(4):
                    index = (selector >> (i * 4)) & 15
                    assert index < 8
                    result |= ((src >> (8 * index)) & 255) << (8 * i)
            elif opcode == "shr.b32":
                result = value(args[1]) >> value(args[2])
            elif opcode == "and.b32":
                result = value(args[1]) & value(args[2])
            elif opcode == "sub.bf16x2":
                result = sum(
                    _bf16(
                        _f32((value(args[1]) >> (16 * i)) & 65535)
                        - _f32((value(args[2]) >> (16 * i)) & 65535)
                    )
                    << (16 * i)
                    for i in range(2)
                )
            else:
                raise AssertionError(opcode)
            registers[dst] = result & 0xFFFFFFFF
        results = [
            dtype(registers[f"{{$w{i}}}"]) for i, dtype in enumerate(write_only_types)
        ]
        return results[0] if len(results) == 1 else tuple(results)

    namespace = {
        "cutlass": SimpleNamespace(Uint16=_U16, Uint32=_U32),
        "cute": SimpleNamespace(arch=SimpleNamespace(inline_ptx=interpreter)),
    }
    exec(compile(ast.Module([function], []), str(path), "exec"), namespace)
    return namespace[function.name], calls


@pytest.mark.parametrize("lanes", (2, 4, 8))
def test_exact_helper_bits_and_physical_lane_order(lanes):
    helper, calls = _held_helper()
    boundary = [0, 1, 127, 128, 255, 85, 170, 2]
    for width in range(1, 9):
        for offset in range(9 - width):
            for rotation in range(8):
                values = (boundary[rotation:] + boundary[:rotation])[:lanes]
                result = helper(
                    int.from_bytes(bytes(values), "little"), offset, width, lanes
                )
                raw = [(v >> offset) & ((1 << width) - 1) for v in values]
                signed = [
                    v - (1 << width) if v & (1 << (width - 1)) else v for v in raw
                ]
                assert result == list(map(_bf16, signed))
                assert len(result) == lanes
                assert all(
                    bits == 0 for v, bits in zip(signed, result, strict=True) if v == 0
                )
    expected_pairs = 1 if lanes == 2 else 2
    assert all(code.count("sub.bf16x2") == expected_pairs for code in calls)
    assert all(("h<3>" in code) == (lanes != 2) for code in calls)


def _physical(monkeypatch):
    case = _graph()
    definition = ast.parse("packet = byte_load()").body[0]
    vloop = ast.parse("for vi in cutlass.range_constexpr(8):\n    pass").body[0]
    body = [definition, vloop]
    statements = []
    strategy = SimpleNamespace(
        _cute_lane_body_by_block={1: body},
        _cute_lane_vloop_by_block={1: vloop},
        _cute_lane_axis_pos_by_block={1: 1},
        _cute_lane_base_index_var_by_block={1: "base"},
        _cute_flat_multi=False,
    )
    grid = object.__new__(DeviceGridState)
    codegen = SimpleNamespace(
        _cute_branch_path=[],
        active_device_loops={1: [grid]},
        statements_stack=[statements],
    )
    env = {
        n: ast.Name(id=n.name, ctx=ast.Load())
        for n in case.graph.nodes
        if n is not case.store
    }
    cute_state = SimpleNamespace(signed_byte_packets={})
    serial = iter(range(20))
    device = SimpleNamespace(
        cute_state=cute_state,
        new_var=lambda prefix, **kwargs: prefix + str(next(serial)),
    )
    state = SimpleNamespace(
        config=Config(**{sb.KEY: True}),
        fx_node=case.load,
        env=env,
        codegen=codegen,
        device_function=device,
    )
    from helion._compiler.compile_environment import CompileEnvironment

    monkeypatch.setattr(
        CompileEnvironment,
        "current",
        lambda: SimpleNamespace(index_type=lambda: "cutlass.Int32"),
    )
    monkeypatch.setattr(sb.HostFunction, "current", lambda: None)
    monkeypatch.setattr(sb, "_fresh_tensors", lambda host: {case.output.meta["val"]})
    monkeypatch.setattr(
        "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
        lambda *args: False,
    )
    sb.record_signed_byte_packet(
        state,
        strategy,
        1,
        case.source.meta["val"],
        8,
        ["row", "column"],
        "packet",
        definition,
        "lane_ok",
    )
    assert len(cute_state.signed_byte_packets) == 1
    state.fx_node = case.store
    return case, state, strategy


def test_actual_store_protocol_preserves_order_and_scalar_fanout(monkeypatch):
    case, state, strategy = _physical(monkeypatch)
    value = sb.packed_store_value(
        state,
        strategy,
        1,
        8,
        case.output.meta["val"],
        ["different_row", "column"],
        "row_ok and lane_ok",
    )
    assert value == "_cute_signed_bitfield_to_bf16_packed(packet, 0, 4, 8)"
    original_scalar = deepcopy(state.env[case.value])
    for row in ("row*2", "row*2+1"):
        statement = memory_ops._cute_register_tile_unroll_vec_store(
            state,
            strategy,
            1,
            "destination",
            [row, "column"],
            "scalar",
            "row_ok and lane_ok",
            torch.bfloat16,
            packed_values_expr=value,
        )
        assert isinstance(statement, ast.Pass)
    body = strategy._cute_lane_body_by_block[1]
    assert ast.dump(state.env[case.value]) == ast.dump(original_scalar)
    assert len(body) == 4 and body[1] is strategy._cute_lane_vloop_by_block[1]
    assert "row * 2" in ast.unparse(body[2]) and "row * 2 + 1" in ast.unparse(body[3])
    assert not any(
        isinstance(node, ast.List) for stmt in body for node in ast.walk(stmt)
    )


@pytest.mark.parametrize(
    "fault",
    [
        "off",
        "dtype",
        "width",
        "lane",
        "mask",
        "environment",
        "loop",
        "statements",
        "strategy",
        "definition_after",
        "reassignment",
        "branch",
        "alias",
        "effect",
        "stale_graph",
        "control_graph",
    ],
)
def test_physical_proof_declines_before_store_change(monkeypatch, fault):
    case, state, strategy = _physical(monkeypatch)
    kwargs = {
        "state": state,
        "strategy": strategy,
        "block_id": 1,
        "width": 8,
        "tensor": case.output.meta["val"],
        "index_exprs": ["row", "column"],
        "mask": "lane_ok",
    }
    if fault == "off":
        state.config.config[sb.KEY] = False
    elif fault == "dtype":
        kwargs["tensor"] = torch.empty((2, 8), dtype=torch.float16)
    elif fault == "width":
        kwargs["width"] = 4
    elif fault == "lane":
        kwargs["index_exprs"] = ["row", "column+1"]
    elif fault == "mask":
        kwargs["mask"] = "other_ok"
    elif fault == "environment":
        state.env = dict(state.env)
    elif fault == "loop":
        state.codegen.active_device_loops[1] = [object.__new__(DeviceGridState)]
    elif fault == "statements":
        state.codegen.statements_stack = [[]]
    elif fault == "strategy":
        kwargs["strategy"] = deepcopy(strategy)
    elif fault == "definition_after":
        strategy._cute_lane_body_by_block[1].reverse()
    elif fault == "reassignment":
        strategy._cute_lane_body_by_block[1].append(ast.parse("packet = 0").body[0])
    elif fault == "branch":
        state.codegen._cute_branch_path = [(7, 0)]
    elif fault == "alias":
        case.output.meta["val"] = case.source.meta["val"]
        monkeypatch.setattr(sb, "_fresh_tensors", lambda host: set())
    elif fault == "effect":
        with case.graph.inserting_before(case.store):
            _node(case.graph, torch.ops.aten.zero_.default, (case.source,), torch.int8)
    elif fault == "control_graph":
        with case.graph.inserting_before(case.store):
            _node(case.graph, sb._tracing_ops._if, (case.source,), torch.int8)
    elif fault == "stale_graph":
        replacement = _graph()
        state.fx_node = replacement.store
    before = ast.dump(ast.Module(strategy._cute_lane_body_by_block[1], []))
    assert sb.packed_store_value(**kwargs) is None
    assert before == ast.dump(ast.Module(strategy._cute_lane_body_by_block[1], []))


@pytest.mark.parametrize(
    "source,output,expected",
    [
        (None, None, True),
        (None, "x", True),
        ("x", None, False),
        ("x", "x and y", True),
        ("x and y", "x", False),
        ("x", "y", False),
        ("x", "x or y", False),
    ],
)
def test_predicate_implication(source, output, expected):
    assert sb.mask_implies(output, source) is expected


def test_config_roundtrip_and_no_default_promotion():
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
        num_sm=148,
        device=torch.device("cpu"),
    )
    assert sb.KEY not in spec._flat_fields()
    config = {sb.KEY: True}
    with pytest.raises(InvalidConfig):
        spec._normalize_cute_signed_bitfield_bf16(config, fix_invalid=False)
    spec.cute_signed_bitfield_bf16_available = True
    before = spec._flat_fields()
    assert before[sb.KEY].search_values() == [False, True]
    assert before[sb.KEY].pattern_neighbors(False) == [True]
    assert before[sb.KEY].pattern_neighbors(True) == [False]
    assert spec.flatten_missing_field_default(sb.KEY, {}) == (True, False)
    for value in (False, True):
        config = {sb.KEY: value}
        spec._normalize_cute_signed_bitfield_bf16(config, fix_invalid=False)
        assert config == ({sb.KEY: True} if value else {})
        assert (
            Config.from_dict(Config(**{sb.KEY: value}).config).config[sb.KEY] is value
        )
    for bad in (0, 1, None, "true", [], {}):
        with pytest.raises(InvalidConfig):
            spec._normalize_cute_signed_bitfield_bf16({sb.KEY: bad}, fix_invalid=False)
    seeds = [
        Config(cute_vector_widths=[1]),
        Config(cute_vector_widths=[1, 8]),
        Config(cute_vector_widths=[4]),
    ]
    env = SimpleNamespace(config_spec=spec)
    augmented = add_signed_bitfield_seeds(env, seeds)
    assert augmented == [
        seeds[0],
        seeds[1],
        Config(cute_vector_widths=[1, 8], **{sb.KEY: True}),
        seeds[2],
    ]
    assert (
        spec.compiler_default_config is None
        and spec._flat_fields().keys() == before.keys()
    )


@pytest.mark.parametrize("shift", range(8))
def test_actual_rshift_operator_and_prims_cast(shift):
    case = _graph(shift, 8 - shift, identity=shift == 0, shift_only=shift != 0)
    for node in case.graph.nodes:
        if node.target is torch.ops.aten.bitwise_right_shift.Tensor_Scalar:
            node.target = torch.ops.aten.__rshift__.Scalar
        if node.target is torch.ops.aten._to_copy.default:
            node.target = torch.ops.prims.convert_element_type.default
            node.args = (*node.args, node.kwargs["dtype"])
            node.kwargs = {}
    field = sb.match_signed_byte_field(case.value)
    assert field is not None and (field.bit_offset, field.bit_width) == (
        shift,
        8 - shift,
    )
    values = torch.tensor([-128, -127, -16, -1, 0, 1, 15, 127], dtype=torch.int8)
    result = torch.ops.aten.__rshift__.Scalar(values, shift)
    assert result.dtype is torch.int8
    assert result.tolist() == [value >> shift for value in values.tolist()]


def test_bad_definition_never_becomes_a_packet(monkeypatch):
    case, state, strategy = _physical(monkeypatch)
    state.device_function.cute_state.signed_byte_packets.clear()
    state.fx_node = case.load
    sb.record_signed_byte_packet(
        state,
        strategy,
        1,
        case.source.meta["val"],
        8,
        ["row", "column"],
        "packet",
        ast.parse("unknown_effect()").body[0],
        "lane_ok",
    )
    assert not state.device_function.cute_state.signed_byte_packets
