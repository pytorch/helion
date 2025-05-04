from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import import_path
import helion.language as hl

if TYPE_CHECKING:
    from helion import Kernel

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
examples_dir = Path(__file__).parent.parent / "examples"


def type_propagation_report(fn: Kernel, *args, ignore=False):
    return fn.bind(args)._debug_str()


class TestTypePropagation(TestCase):
    maxDiff = 16384

    def test_add(self):
        output = type_propagation_report(
            basic_kernels.add,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def add(x, y):
    # Call: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32))) SourceOrigin(location=<SourceLocation basic_kernels.py:10>)
    # Attribute: CallableType(broadcast_tensors) AttributeOrigin(value=GlobalOrigin(name='torch'), key='broadcast_tensors')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    x, y = torch.broadcast_tensors(x, y)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:11>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:10>), key=0)
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation basic_kernels.py:12>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation basic_kernels.py:12>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:11>), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:11>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:13>)
        # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:11>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:12>)
        # BinOp: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:13>)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:13>)
        # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:10>), key=0)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:12>)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:13>)
        # Name: TensorType([y_size0, x_size1], torch.int32) GetItemOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:10>), key=1)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:12>)
        out[tile] = x[tile] + y[tile]
    return out

def root_graph_0():
    # File: .../basic_kernels.py:13 in add, code: out[tile] = x[tile] + y[tile]
    x: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    load: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    y: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('y')
    load_1: "i32[u0, u1]" = helion_language_memory_ops_load(y, [block_size_0, block_size_1]);  y = None
    add: "i32[u0, u1]" = torch.ops.aten.add.Tensor(load, load_1);  load = load_1 = None
    out: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], add);  out = block_size_0 = block_size_1 = add = store = None
    return None""",
        )

    def test_torch_ops_pointwise(self):
        output = type_propagation_report(
            basic_kernels.torch_ops_pointwise,
            torch.ones([1024], dtype=torch.int32),
            torch.ones([1024], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def torch_ops_pointwise(x, y):
    # Call: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:19>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0)])) SourceOrigin(location=<SourceLocation basic_kernels.py:20>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s77), )) SourceOrigin(location=<SourceLocation basic_kernels.py:20>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:19>), key='size')
    # Name: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:19>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size_0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Name: TensorType([x_size0], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:19>)
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:20>)
        # Call: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Attribute: CallableType(_VariableFunctionsClass.sigmoid) AttributeOrigin(value=GlobalOrigin(name='torch'), key='sigmoid')
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Call: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Attribute: CallableType(_VariableFunctionsClass.add) AttributeOrigin(value=GlobalOrigin(name='torch'), key='add')
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Call: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Attribute: CallableType(_VariableFunctionsClass.sin) AttributeOrigin(value=GlobalOrigin(name='torch'), key='sin')
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Subscript: TensorType([block_size_0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Name: TensorType([x_size0], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:20>)
        # Call: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Attribute: CallableType(_VariableFunctionsClass.cos) AttributeOrigin(value=GlobalOrigin(name='torch'), key='cos')
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # Subscript: TensorType([block_size_0], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:21>)
        # Name: TensorType([y_size0], torch.int32) ArgumentOrigin(name='y')
        # Name: SequenceType([TileIndexType(0)]) SourceOrigin(location=<SourceLocation basic_kernels.py:20>)
        out[tile] = torch.sigmoid(torch.add(torch.sin(x[tile]), torch.cos(y[tile])))
    return out

def root_graph_0():
    # File: .../basic_kernels.py:21 in torch_ops_pointwise, code: out[tile] = torch.sigmoid(torch.add(torch.sin(x[tile]), torch.cos(y[tile])))
    x: "i32[s77]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    load: "i32[u0]" = helion_language_memory_ops_load(x, [block_size_0]);  x = None
    sin: "f32[u0]" = torch.ops.aten.sin.default(load);  load = None
    y: "i32[s17]" = helion_language__tracing_ops__host_tensor('y')
    load_1: "i32[u0]" = helion_language_memory_ops_load(y, [block_size_0]);  y = None
    cos: "f32[u0]" = torch.ops.aten.cos.default(load_1);  load_1 = None
    add: "f32[u0]" = torch.ops.aten.add.Tensor(sin, cos);  sin = cos = None
    sigmoid: "f32[u0]" = torch.ops.aten.sigmoid.default(add);  add = None
    convert_element_type: "i32[u0]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.int32);  sigmoid = None
    out: "i32[s77]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0], convert_element_type);  out = block_size_0 = convert_element_type = store = None
    return None""",
        )

    def test_all_ast_nodes(self):
        output = type_propagation_report(
            import_path(datadir / "all_ast_nodes.py").all_ast_nodes,
            torch.ones([5, 5], dtype=torch.int32),
            torch.ones([5, 5], dtype=torch.int32),
            ignore=True,
        )
        self.assertExpectedInline(
            output,
            """\
def all_ast_nodes(x, y):
    # Constant: LiteralType(1024) SourceOrigin(location=<SourceLocation all_ast_nodes.py:23>)
    int_literal = 1024
    # JoinedStr: UnsupportedType('str is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:24>)
    formatted_value = f'prefix{int_literal}suffix'
    # Constant: LiteralType('abc') SourceOrigin(location=<SourceLocation all_ast_nodes.py:25>)
    joined_string = 'abc'
    # List: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: LiteralType(1024) SourceOrigin(location=<SourceLocation all_ast_nodes.py:23>)
    list_literal0 = [x, y, int_literal]
    # Tuple: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    tuple_literal0 = (x, y, 1, 2)
    # List: SequenceType([LiteralType(5), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    # Name: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:28>)
    list_literal1 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # List: SequenceType([LiteralType(5), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024), LiteralType(3), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    # Name: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1), LiteralType(2))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:27>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:29>)
    tuple_literal2 = [5, *list_literal0, 3, *tuple_literal0, 4]
    # Set: UnsupportedType('set is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:30>)
    set_literal = {1, 2, 3}
    # Dict: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:31>)
    dict_literal0 = {}
    # Name: DictType({1: LiteralType(2)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:31>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:32>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:32>)
    dict_literal0[1] = 2
    # Dict: DictType({1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType('y') SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    dict_literal1 = {1: x, 'y': y}
    # Dict: DictType({'foo': LiteralType('bar'), 1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Constant: LiteralType('foo') SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Constant: LiteralType('bar') SourceOrigin(location=<SourceLocation all_ast_nodes.py:34>)
    # Name: DictType({1: TensorType([y_size0, x_size1], torch.int32), 'y': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:33>)
    dict_literal2 = {'foo': 'bar', **dict_literal1}
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:35>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    uadd = +x
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:36>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    usub = -x
    # UnaryOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:37>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    invert = ~x
    # UnaryOp: SymBoolType(Eq(u0, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:38>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    not_ = not x
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:39>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add = x + y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:40>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    sub = x - y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:41>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    mul = x * y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:43>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    truediv = x / y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:44>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    floordiv = x // y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:45>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    mod = x % y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:46>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    pow = x ** y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:47>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    lshift = x << y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:48>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    rshift = x >> y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:49>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_and = x & y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:50>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_xor = x ^ y
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:51>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    bitwise_or = x | y
    # BoolOp: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:52>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    and_ = x and y
    # BoolOp: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:53>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    or_ = x and y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:54>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    eq = x == y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:55>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ne = x != y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:56>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    lt = x < y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:57>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    le = x <= y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:58>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    gt = x > y
    # Compare: TensorType([y_size0, x_size1], torch.bool) SourceOrigin(location=<SourceLocation all_ast_nodes.py:59>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ge = x >= y
    # Compare: SymBoolType(Eq(u1, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:60>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    is_ = x is y
    # Compare: SymBoolType(Eq(u2, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:61>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    is_not = x is not y
    # Compare: SymBoolType(Eq(u3, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:62>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    in_ = x in y
    # Compare: SymBoolType(Eq(u4, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:63>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    not_in = x not in y
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:64>)
    # Name: CallableType(func) AttributeOrigin(value=GlobalOrigin(name='_source_module'), key='func')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:64>)
    call0 = func(x, y, 3)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:65>)
    # Name: CallableType(func) AttributeOrigin(value=GlobalOrigin(name='_source_module'), key='func')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:65>)
    call1 = func(x, y, c=3)
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:66>)
    # Name: CallableType(func) AttributeOrigin(value=GlobalOrigin(name='_source_module'), key='func')
    # Tuple: SequenceType((TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:66>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    call2 = func(*(x, y, y))
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: CallableType(func) AttributeOrigin(value=GlobalOrigin(name='_source_module'), key='func')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Dict: DictType({'b': TensorType([y_size0, x_size1], torch.int32), 'c': TensorType([y_size0, x_size1], torch.int32)}) SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Constant: LiteralType('b') SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType('c') SourceOrigin(location=<SourceLocation all_ast_nodes.py:67>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    call3 = func(x, **{'b': y, 'c': y})
    # IfExp: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: UnknownType('And not supported on TensorType([y_size0, x_size1], torch.int32) and TensorType([y_size0, x_size1], torch.int32)') SourceOrigin(location=<SourceLocation all_ast_nodes.py:53>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    ifexp = x if or_ else y
    # Attribute: LiteralType(torch.int32) AttributeOrigin(value=ArgumentOrigin(name='x'), key='dtype')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr0 = x.dtype
    # Attribute: SequenceType((SymIntType(s17), SymIntType(s27))) AttributeOrigin(value=ArgumentOrigin(name='x'), key='shape')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr1 = x.shape
    # Call: SymIntType(s17) SourceOrigin(location=<SourceLocation all_ast_nodes.py:75>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:75>)
    attr2 = x.size(0)
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:76>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr3 = x.size()
    # Call: SequenceType((SymIntType(s27), LiteralType(1))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:77>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='stride')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    attr4 = x.stride()
    # Call: SymIntType(s27) SourceOrigin(location=<SourceLocation all_ast_nodes.py:78>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='stride')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:78>)
    attr5 = x.stride(0)
    # NamedExpr: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    named_expr = (z := (y + 1))
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    zzz = zz = z - 1
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:81>)
    # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:81>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:80>)
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:79>)
    q = zzz + zz + z
    # Subscript: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:82>)
    subscript0 = list_literal0[0]
    # Subscript: SequenceType([TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Slice: SliceType(LiteralType(1):LiteralType(None):LiteralType(None)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:83>)
    subscript1 = list_literal0[1:]
    # Subscript: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # Name: SequenceType([TensorType([y_size0, x_size1], torch.int32), TensorType([y_size0, x_size1], torch.int32), LiteralType(1024)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:26>)
    # Slice: SliceType(LiteralType(None):LiteralType(-1):LiteralType(None)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # UnaryOp: LiteralType(-1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:84>)
    subscript2 = list_literal0[:-1]
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add += y
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    add -= y
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    a, b, c = [1, 2, 3]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:88>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Name: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    # Name: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:87>)
    tmp0 = [a, b, c]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    a, *bc = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4), LiteralType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:90>)
    # Name: SequenceType([LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:89>)
    tmp1 = [*bc, a]
    # List: SequenceType([LiteralType(1), LiteralType(2), LiteralType(3), LiteralType(4)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Constant: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    a, *ab, c = [1, 2, 3, 4]
    # List: SequenceType([LiteralType(1), LiteralType(4), LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:92>)
    # Name: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Name: LiteralType(4) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    # Name: SequenceType([LiteralType(2), LiteralType(3)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:91>)
    tmp2 = [a, c, *ab]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Constant: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Constant: LiteralType(6) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    a, *ab, c = [5, 6]
    # List: SequenceType([LiteralType(5), LiteralType(6)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:94>)
    # Name: LiteralType(5) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Name: LiteralType(6) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    # Name: SequenceType([]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:93>)
    tmp2 = [a, c, *ab]
    try:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:97>)
        e0 = 1
        # Call: UnknownType('Exception is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:98>)
        # Name: CallableType(Exception) BuiltinOrigin(name='Exception')
        raise Exception()
    except Exception as e:
        e1 = 1
    else:
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:102>)
        e2 = 1
        pass
    # Compare: SymBoolType(Eq(u5, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:105>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    assert x is not y
    # Compare: SymBoolType(Eq(u6, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:106>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Constant: LiteralType('msg') SourceOrigin(location=<SourceLocation all_ast_nodes.py:106>)
    assert x is not y, 'msg'
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:86>)
    del add
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:109>)
    import torch as torch2
    # alias: UnknownType('ast.alias is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:110>)
    from torch import Tensor
    # Compare: SymBoolType(Eq(u7, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:112>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    if x is y:
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        join_var0 = x
        # BinOp: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:114>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        join_var1 = x + y
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:115>)
        join_var2 = 1
    else:
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        join_var0 = y
        # Constant: LiteralType(None) SourceOrigin(location=<SourceLocation all_ast_nodes.py:118>)
        join_var1 = None
        # Constant: LiteralType(2) SourceOrigin(location=<SourceLocation all_ast_nodes.py:119>)
        join_var2 = 2
    # List: SequenceType([TensorType([y_size0, x_size1], torch.int32), UnknownType("Can't combine types from control flow: TensorType([y_size0, x_size1], torch.int32) and LiteralType(None)"), SymIntType(u8)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:120>)
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
    # Name: UnknownType("Can't combine types from control flow: TensorType([y_size0, x_size1], torch.int32) and LiteralType(None)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:118>)
    # Name: SymIntType(u8) SourceOrigin(location=<SourceLocation all_ast_nodes.py:115>)
    combined = [join_var0, join_var1, join_var2]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:122>)
    v = 0
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:123>)
    # For: loop_type=HOST
    z = 0
    # Call: LiteralType(range(0, 3)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
    # Name: CallableType(range) BuiltinOrigin(name='range')
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
    for i in range(3):
        # BinOp: SymIntType(u12) SourceOrigin(location=<SourceLocation all_ast_nodes.py:125>)
        # Name: SymIntType(u11) SourceOrigin(location=<SourceLocation all_ast_nodes.py:122>)
        # Name: SymIntType(u10) SourceOrigin(location=<SourceLocation all_ast_nodes.py:124>)
        v = v + i
        # BinOp: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
        # Name: UnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        z = z + x
        break
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:129>)
        t = 0
    # List: SequenceType([SymIntType(u13), ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)")]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:130>)
    # Name: SymIntType(u13) SourceOrigin(location=<SourceLocation all_ast_nodes.py:125>)
    # Name: ChainedUnknownType("Can't combine types from control flow: LiteralType(0) and TensorType([y_size0, x_size1], torch.int32)") SourceOrigin(location=<SourceLocation all_ast_nodes.py:126>)
    combined = [v, z]
    # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
    i = 0
    # Compare: SymBoolType(Eq(u16, 1)) SourceOrigin(location=<SourceLocation all_ast_nodes.py:133>)
    # Name: SymIntType(u14) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
    # Constant: LiteralType(3) SourceOrigin(location=<SourceLocation all_ast_nodes.py:133>)
    while i < 3:
        # BinOp: SymIntType(u18) SourceOrigin(location=<SourceLocation all_ast_nodes.py:134>)
        # Name: SymIntType(u17) SourceOrigin(location=<SourceLocation all_ast_nodes.py:132>)
        # Constant: LiteralType(1) SourceOrigin(location=<SourceLocation all_ast_nodes.py:134>)
        i = i + 1
        continue
    else:
        # Constant: LiteralType(0) SourceOrigin(location=<SourceLocation all_ast_nodes.py:137>)
        t = 0
    with contextlib.nullcontext():
    # Global: UnknownType('ast.Global is not supported') SourceOrigin(location=<SourceLocation all_ast_nodes.py:142>)
        e3 = 1
    global global0
    # Call: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s17), SymIntType(s27))) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>), key='size')
    # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
    for tile in hl.tile(out.size()):
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation all_ast_nodes.py:144>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        # BinOp: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation all_ast_nodes.py:146>)
        # Name: TensorType([y_size0, x_size1], torch.int32) ArgumentOrigin(name='y')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation all_ast_nodes.py:145>)
        out[tile] = x[tile] + y[tile]
    return out

def root_graph_0():
    # File: .../all_ast_nodes.py:146 in all_ast_nodes, code: out[tile] = x[tile] + y[tile]
    x: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u21)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u22)" = helion_language__tracing_ops__get_symnode('block_size_1')
    load: "i32[u21, u22]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    y: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('y')
    load_1: "i32[u21, u22]" = helion_language_memory_ops_load(y, [block_size_0, block_size_1]);  y = None
    add: "i32[u21, u22]" = torch.ops.aten.add.Tensor(load, load_1);  load = load_1 = None
    out: "i32[s17, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], add);  out = block_size_0 = block_size_1 = add = store = None
    return None""",
        )

    def test_hl_zeros_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_zeros_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def hl_zeros_usage(x: torch.Tensor):
    # Call: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:27>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s77), SymIntType(s27))) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:27>), key='size')
    # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:27>)
    for tile in hl.tile(out.size()):
        # Call: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:29>)
        # Attribute: CallableType(zeros) AttributeOrigin(value=GlobalOrigin(name='hl'), key='zeros')
        # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
        # Attribute: LiteralType(torch.int32) AttributeOrigin(value=ArgumentOrigin(name='x'), key='dtype')
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        tmp = hl.zeros(tile, dtype=x.dtype)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:30>)
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
        tmp += x[tile]
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:31>)
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
        tmp += x[tile]
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:32>)
        # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:27>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:28>)
        # Name: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:31>)
        out[tile] = tmp
    return out

def root_graph_0():
    # File: .../basic_kernels.py:29 in hl_zeros_usage, code: tmp = hl.zeros(tile, dtype=x.dtype)
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    tmp: "i32[u0, u1]" = helion_language_creation_ops_full([block_size_0, block_size_1], 0, torch.int32)

    # File: .../basic_kernels.py:30 in hl_zeros_usage, code: tmp += x[tile]
    x: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    load: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1])
    tmp_1: "i32[u0, u1]" = torch.ops.aten.add.Tensor(tmp, load);  tmp = load = None

    # File: .../basic_kernels.py:31 in hl_zeros_usage, code: tmp += x[tile]
    load_1: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    tmp_2: "i32[u0, u1]" = torch.ops.aten.add.Tensor(tmp_1, load_1);  tmp_1 = load_1 = None

    # File: .../basic_kernels.py:32 in hl_zeros_usage, code: out[tile] = tmp
    out: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], tmp_2);  out = block_size_0 = block_size_1 = tmp_2 = store = None
    return None""",
        )

    def test_hl_full_usage(self):
        output = type_propagation_report(
            basic_kernels.hl_full_usage,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def hl_full_usage(x: torch.Tensor):
    # Call: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:38>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s77), SymIntType(s27))) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
    # Attribute: TensorAttributeType AttributeOrigin(value=SourceOrigin(location=<SourceLocation basic_kernels.py:38>), key='size')
    # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:38>)
    for tile in hl.tile(out.size()):
        # Call: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:40>)
        # Attribute: CallableType(full) AttributeOrigin(value=GlobalOrigin(name='hl'), key='full')
        # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
        # Constant: LiteralType(1) DeviceOrigin(location=<SourceLocation basic_kernels.py:40>)
        # Attribute: LiteralType(torch.int32) AttributeOrigin(value=ArgumentOrigin(name='x'), key='dtype')
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        tmp = hl.full(tile, 1, dtype=x.dtype)
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:41>)
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
        tmp += x[tile]
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:42>)
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
        tmp += x[tile]
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:43>)
        # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:38>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation basic_kernels.py:39>)
        # Name: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:42>)
        out[tile] = tmp
    return out

def root_graph_0():
    # File: .../basic_kernels.py:40 in hl_full_usage, code: tmp = hl.full(tile, 1, dtype=x.dtype)
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    tmp: "i32[u0, u1]" = helion_language_creation_ops_full([block_size_0, block_size_1], 1, torch.int32)

    # File: .../basic_kernels.py:41 in hl_full_usage, code: tmp += x[tile]
    x: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    load: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1])
    tmp_1: "i32[u0, u1]" = torch.ops.aten.add.Tensor(tmp, load);  tmp = load = None

    # File: .../basic_kernels.py:42 in hl_full_usage, code: tmp += x[tile]
    load_1: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    tmp_2: "i32[u0, u1]" = torch.ops.aten.add.Tensor(tmp_1, load_1);  tmp_1 = load_1 = None

    # File: .../basic_kernels.py:43 in hl_full_usage, code: out[tile] = tmp
    out: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], tmp_2);  out = block_size_0 = block_size_1 = tmp_2 = store = None
    return None""",
        )

    def test_pointwise_device_loop(self):
        output = type_propagation_report(
            basic_kernels.pointwise_device_loop,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def pointwise_device_loop(x: torch.Tensor):
    # Call: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:49>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    out = torch.empty_like(x)
    # Attribute: SequenceType((SymIntType(s77), SymIntType(s27))) AttributeOrigin(value=ArgumentOrigin(name='x'), key='shape')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    n, m = x.shape
    # Call: IterType(TileIndexType(0)) SourceOrigin(location=<SourceLocation basic_kernels.py:51>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Name: SymIntType(s77) GetItemOrigin(value=AttributeOrigin(value=ArgumentOrigin(name='x'), key='shape'), key=0)
        # For: loop_type=DEVICE
    for tile_n in hl.tile(n):
        # Call: IterType(TileIndexType(1)) DeviceOrigin(location=<SourceLocation basic_kernels.py:52>)
        # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
        # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
        # Name: SymIntType(s27) GetItemOrigin(value=AttributeOrigin(value=ArgumentOrigin(name='x'), key='shape'), key=1)
        for tile_m in hl.tile(m):
            # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:53>)
            # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation basic_kernels.py:49>)
            # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation basic_kernels.py:51>)
            # Name: TileIndexType(1) DeviceOrigin(location=<SourceLocation basic_kernels.py:52>)
            # Call: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation basic_kernels.py:53>)
            # Attribute: CallableType(_VariableFunctionsClass.sigmoid) AttributeOrigin(value=GlobalOrigin(name='torch'), key='sigmoid')
            # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
            # BinOp: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:53>)
            # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation basic_kernels.py:53>)
            # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
            # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation basic_kernels.py:51>)
            # Name: TileIndexType(1) DeviceOrigin(location=<SourceLocation basic_kernels.py:52>)
            # Constant: LiteralType(1) DeviceOrigin(location=<SourceLocation basic_kernels.py:53>)
            out[tile_n, tile_m] = torch.sigmoid(x[tile_n, tile_m] + 1)
    return out

def for_loop_0():
    # File: .../basic_kernels.py:53 in pointwise_device_loop, code: out[tile_n, tile_m] = torch.sigmoid(x[tile_n, tile_m] + 1)
    x: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    load: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    add: "i32[u0, u1]" = torch.ops.aten.add.Tensor(load, 1);  load = None
    sigmoid: "f32[u0, u1]" = torch.ops.aten.sigmoid.default(add);  add = None
    convert_element_type: "i32[u0, u1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.int32);  sigmoid = None
    out: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], convert_element_type);  out = block_size_0 = block_size_1 = convert_element_type = store = None
    return []

def root_graph_1():
    # File: .../basic_kernels.py:52 in pointwise_device_loop, code: for tile_m in hl.tile(m):
    _for_loop = helion_language__tracing_ops__for_loop(0, []);  _for_loop = None
    return None""",
        )

    def test_method_call(self):
        @helion.kernel
        def fn(x):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile].sin()
            return out

        output = type_propagation_report(
            fn,
            torch.ones([512, 512], dtype=torch.int32),
        )
        self.assertExpectedInline(
            output,
            """\
def fn(x):
    # Call: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation test_type_propagation.py:784>)
    # Attribute: CallableType(_VariableFunctionsClass.empty_like) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty_like')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty_like(x)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation test_type_propagation.py:785>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Call: SequenceType((SymIntType(s77), SymIntType(s27))) SourceOrigin(location=<SourceLocation test_type_propagation.py:785>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
    for tile in hl.tile(x.size()):
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation test_type_propagation.py:786>)
        # Name: TensorType([x_size0, x_size1], torch.int32) SourceOrigin(location=<SourceLocation test_type_propagation.py:784>)
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation test_type_propagation.py:785>)
        # Call: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation test_type_propagation.py:786>)
        # Attribute: TensorAttributeType AttributeOrigin(value=DeviceOrigin(location=<SourceLocation test_type_propagation.py:786>), key='sin')
        # Subscript: TensorType([block_size_0, block_size_1], torch.int32) DeviceOrigin(location=<SourceLocation test_type_propagation.py:786>)
        # Name: TensorType([x_size0, x_size1], torch.int32) ArgumentOrigin(name='x')
        # Name: SequenceType([TileIndexType(0), TileIndexType(1)]) SourceOrigin(location=<SourceLocation test_type_propagation.py:785>)
        out[tile] = x[tile].sin()
    return out

def root_graph_0():
    # File: .../test_type_propagation.py:786 in fn, code: out[tile] = x[tile].sin()
    x: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    load: "i32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, block_size_1]);  x = None
    sin: "f32[u0, u1]" = torch.ops.aten.sin.default(load);  load = None
    convert_element_type: "i32[u0, u1]" = torch.ops.prims.convert_element_type.default(sin, torch.int32);  sin = None
    out: "i32[s77, s27]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], convert_element_type);  out = block_size_0 = block_size_1 = convert_element_type = store = None
    return None""",
        )

    def test_matmul(self):
        output = type_propagation_report(
            import_path(examples_dir / "matmul.py").matmul,
            torch.ones([512, 512]),
            torch.ones([512, 512]),
        )
        self.assertExpectedInline(
            output,
            """\
def matmul(x: torch.Tensor, y: torch.Tensor):
    # Call: SequenceType((LiteralType(512), LiteralType(512))) SourceOrigin(location=<SourceLocation matmul.py:12>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='x')
    m, k = x.size()
    # Call: SequenceType((LiteralType(512), LiteralType(512))) SourceOrigin(location=<SourceLocation matmul.py:13>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='y'), key='size')
    # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='y')
    k2, n = y.size()
    # Compare: LiteralType(True) SourceOrigin(location=<SourceLocation matmul.py:14>)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:12>), key=1)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:13>), key=0)
    # JoinedStr: UnsupportedType('str is not supported') SourceOrigin(location=<SourceLocation matmul.py:14>)
    assert k == k2, f'size mismatch {k} != {k2}'
    # Call: TensorType([512, 512], torch.float32) SourceOrigin(location=<SourceLocation matmul.py:15>)
    # Attribute: CallableType(_VariableFunctionsClass.empty) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # List: SequenceType([LiteralType(512), LiteralType(512)]) SourceOrigin(location=<SourceLocation matmul.py:16>)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:12>), key=0)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:13>), key=1)
    # Call: LiteralType(torch.float32) SourceOrigin(location=<SourceLocation matmul.py:16>)
    # Attribute: CallableType(_VariableFunctionsClass.promote_types) AttributeOrigin(value=GlobalOrigin(name='torch'), key='promote_types')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # Attribute: LiteralType(torch.float32) AttributeOrigin(value=ArgumentOrigin(name='x'), key='dtype')
    # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='x')
    # Attribute: LiteralType(torch.float32) AttributeOrigin(value=ArgumentOrigin(name='y'), key='dtype')
    # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='y')
    # Attribute: LiteralType(device(type='cpu')) AttributeOrigin(value=ArgumentOrigin(name='x'), key='device')
    # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    # Call: IterType(SequenceType([TileIndexType(0), TileIndexType(1)])) SourceOrigin(location=<SourceLocation matmul.py:18>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # List: SequenceType([LiteralType(512), LiteralType(512)]) SourceOrigin(location=<SourceLocation matmul.py:18>)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:12>), key=0)
    # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:13>), key=1)
    for tile_m, tile_n in hl.tile([m, n]):
        # Call: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:19>)
        # Attribute: CallableType(zeros) AttributeOrigin(value=GlobalOrigin(name='hl'), key='zeros')
        # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
        # List: SequenceType([TileIndexType(0), TileIndexType(1)]) DeviceOrigin(location=<SourceLocation matmul.py:19>)
        # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation matmul.py:18>)
        # Name: TileIndexType(1) SourceOrigin(location=<SourceLocation matmul.py:18>)
        # Attribute: LiteralType(torch.float32) AttributeOrigin(value=GlobalOrigin(name='torch'), key='float32')
        # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
        # For: loop_type=DEVICE
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        # Call: IterType(TileIndexType(2)) DeviceOrigin(location=<SourceLocation matmul.py:20>)
        # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
        # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
        # Name: LiteralType(512) GetItemOrigin(value=SourceOrigin(location=<SourceLocation matmul.py:12>), key=1)
        for tile_k in hl.tile(k):
            # Call: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:21>)
            # Attribute: CallableType(_VariableFunctionsClass.addmm) AttributeOrigin(value=GlobalOrigin(name='torch'), key='addmm')
            # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
            # Name: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:21>)
            # Subscript: TensorType([block_size_0, block_size_2], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:21>)
            # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='x')
            # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation matmul.py:18>)
            # Name: TileIndexType(2) DeviceOrigin(location=<SourceLocation matmul.py:20>)
            # Subscript: TensorType([block_size_2, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:21>)
            # Name: TensorType([512, 512], torch.float32) ArgumentOrigin(name='y')
            # Name: TileIndexType(2) DeviceOrigin(location=<SourceLocation matmul.py:20>)
            # Name: TileIndexType(1) SourceOrigin(location=<SourceLocation matmul.py:18>)
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        # Subscript: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:22>)
        # Name: TensorType([512, 512], torch.float32) SourceOrigin(location=<SourceLocation matmul.py:15>)
        # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation matmul.py:18>)
        # Name: TileIndexType(1) SourceOrigin(location=<SourceLocation matmul.py:18>)
        # Name: TensorType([block_size_0, block_size_1], torch.float32) DeviceOrigin(location=<SourceLocation matmul.py:21>)
        out[tile_m, tile_n] = acc
    return out

def for_loop_0(arg0_1: "f32[u0, u1]"):
    # File: .../matmul.py:21 in matmul, code: acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    x: "f32[512, 512]" = helion_language__tracing_ops__host_tensor('x')
    sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
    block_size_2: "Sym(u2)" = helion_language__tracing_ops__get_symnode('block_size_2')
    load: "f32[u0, u2]" = helion_language_memory_ops_load(x, [sym_size_int, block_size_2]);  x = sym_size_int = None
    y: "f32[512, 512]" = helion_language__tracing_ops__host_tensor('y')
    sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(arg0_1, 1)
    load_1: "f32[u2, u1]" = helion_language_memory_ops_load(y, [block_size_2, sym_size_int_1]);  y = block_size_2 = sym_size_int_1 = None
    acc: "f32[u0, u1]" = torch.ops.aten.addmm.default(arg0_1, load, load_1);  arg0_1 = load = load_1 = None
    return [acc]

def root_graph_1():
    # File: .../matmul.py:19 in matmul, code: acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    block_size_1: "Sym(u1)" = helion_language__tracing_ops__get_symnode('block_size_1')
    acc: "f32[u0, u1]" = helion_language_creation_ops_full([block_size_0, block_size_1], 0.0, torch.float32)

    # File: .../matmul.py:20 in matmul, code: for tile_k in hl.tile(k):
    _for_loop = helion_language__tracing_ops__for_loop(0, [acc])
    getitem: "f32[u0, u1]" = _for_loop[0];  _for_loop = None
    _phi: "f32[u0, u1]" = helion_language__tracing_ops__phi(acc, getitem);  acc = getitem = None

    # File: .../matmul.py:22 in matmul, code: out[tile_m, tile_n] = acc
    out: "f32[512, 512]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0, block_size_1], _phi);  out = block_size_0 = block_size_1 = _phi = store = None
    return None""",
        )


if __name__ == "__main__":
    unittest.main()
