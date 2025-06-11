# testing inputs for test_type_propagation.py
from __future__ import annotations

import contextlib

import torch
import helion
import helion.language as hl


def func(a, b, c):
    return a + b + c


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def all_ast_nodes(x, y):
    # def func0(q, v):
    #    nonlocal int_literal
    #    return q + v
    #
    # class ClassDef:
    #     pass
    int_literal = 1024
    formatted_value = f"prefix{int_literal}suffix"
    joined_string = "abc"
    list_literal0 = [x, y, int_literal]
    tuple_literal0 = (x, y, 1, 2)
    list_literal1 = [5, *list_literal0, 3, *tuple_literal0, 4]
    tuple_literal2 = [5, *list_literal0, 3, *tuple_literal0, 4]
    set_literal = {1, 2, 3}
    dict_literal0 = {}
    dict_literal0[1] = 2
    dict_literal1 = {1: x, "y": y}
    dict_literal2 = {"foo": "bar", **dict_literal1}
    uadd = +x
    usub = -x
    invert = ~x
    not_ = not x
    add = x + y
    sub = x - y
    mul = x * y
    # matmul = x @ y
    truediv = x / y
    floordiv = x // y
    mod = x % y
    pow = x**y
    lshift = x << y
    rshift = x >> y
    bitwise_and = x & y
    bitwise_xor = x ^ y
    bitwise_or = x | y
    and_ = x and y
    or_ = x and y
    eq = x == y
    ne = x != y
    lt = x < y
    le = x <= y
    gt = x > y
    ge = x >= y
    is_ = x is y
    is_not = x is not y
    in_ = x in y
    not_in = x not in y
    call0 = func(x, y, 3)
    call1 = func(x, y, c=3)
    call2 = func(*(x, y, y))
    call3 = func(x, **{"b": y, "c": y})
    ifexp = x if or_ else y
    # listcomp = [v for v in (1, 2, 3)]
    # dictcomp = {k: v for k, v in [(1, 2), (3, 4)]}
    # setcomp = {v for v in (1, 2, 3)}
    # generator = (v for v in (1, 2, 3))
    attr0 = x.dtype
    attr1 = x.shape
    attr2 = x.size(0)
    attr3 = x.size()
    attr4 = x.stride()
    attr5 = x.stride(0)
    named_expr = (z := y + 1)
    zzz = zz = z - 1
    q = zzz + zz + z
    subscript0 = list_literal0[0]
    subscript1 = list_literal0[1:]
    subscript2 = list_literal0[:-1]
    add += y
    add -= y
    a, b, c = [1, 2, 3]
    tmp0 = [a, b, c]
    a, *bc = [1, 2, 3, 4]
    tmp1 = [*bc, a]
    a, *ab, c = [1, 2, 3, 4]
    tmp2 = [a, c, *ab]
    a, *ab, c = [5, 6]
    tmp2 = [a, c, *ab]

    try:
        e0 = 1
        raise Exception()
    except Exception as e:
        e1 = 1
    else:
        e2 = 1
        pass

    assert x is not y
    assert x is not y, "msg"
    del add

    import torch as torch2
    from torch import Tensor

    if x is y:
        join_var0 = x
        join_var1 = x + y
        join_var2 = 1
        join_var3 = {"x": 0}
    else:
        join_var0 = y
        join_var1 = None
        join_var2 = 2
        join_var3 = {"x": 1}
    combined = [join_var0, join_var1, join_var2, join_var3]

    i = 0
    while i < 3:
        i = i + 1
        continue
    else:
        t = 0

    with contextlib.nullcontext():
        e3 = 1

    global global0

    out = torch.empty_like(x)
    v = 0
    z = 0
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    for i in range(3):
        v = v + i
        z = z + x
        break
    else:
        t = 0
    combined = [v, z]

    return out
