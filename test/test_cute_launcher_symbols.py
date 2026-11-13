"""Generated host wrapper symbols must accept every runtime float value."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest
import torch

from helion.runtime.cute.launcher import _create_cute_wrapper

pytestmark = pytest.mark.cpu_only


def _symbol(wrapper: object, value: float) -> str:
    from cuda.bindings.driver import CUstream
    from cutlass.base_dsl.dsl import BaseDSL

    func = cast("Any", wrapper).__wrapped__
    BaseDSL._lazy_initialize_dsl(func)
    dsl = func._dsl_object
    options = dsl._get_name_options(func)
    default = (
        dsl.mangle_name(
            func.__name__,
            (value, 1, 1, 1, CUstream(0)),
            inspect.signature(func, eval_str=True),
        )
        if options.keep_mangled_name
        else ""
    )
    return cast("str", dsl._customize_generated_name(default, options))


def _parse_global(symbol: str) -> None:
    from cutlass._mlir import ir

    with ir.Context(), ir.Location.unknown():
        # This is the SDK's actual TVM FFI state declaration, before GPU use.
        ir.Operation.parse(
            f"llvm.mlir.global private @__{symbol}_cuda_state(0 : i64) : i64"
        )


@pytest.mark.parametrize("value", [1e16, -1e16, 0.0, -0.0, float("inf"), float("nan")])
def test_runtime_float_wrapper_symbol_is_valid(value: float) -> None:
    wrapper = _create_cute_wrapper(
        SimpleNamespace(__name__="scalar_kernel"), (("scalar", "float"),), (32, 1, 1)
    )
    _parse_global(_symbol(wrapper, value))


def test_wrapper_symbol_is_independent_of_runtime_scalar() -> None:
    wrapper = _create_cute_wrapper(
        SimpleNamespace(__name__="scalar_kernel"), (("scalar", "float"),), (32, 1, 1)
    )
    assert _symbol(wrapper, 1e16) == _symbol(wrapper, 2.0)


def test_wrapper_symbol_preserves_kernel_schema_and_launch_distinctions() -> None:
    kernel = SimpleNamespace(__name__="scalar_kernel")
    other = SimpleNamespace(__name__="scalar_kernel")
    wrappers = [
        _create_cute_wrapper(owner, (("scalar", kind),), block)
        for owner, kind, block in (
            (kernel, "float", (32, 1, 1)),
            (other, "float", (32, 1, 1)),
            (kernel, "int", (32, 1, 1)),
            (kernel, "float", (64, 1, 1)),
        )
    ]
    symbols = [_symbol(wrapper, 2.0) for wrapper in wrappers]
    assert len(set(symbols)) == len(symbols)
    repeated = _create_cute_wrapper(kernel, (("scalar", "float"),), (32, 1, 1))
    assert _symbol(repeated, 2.0) == symbols[0]
    for symbol in symbols:
        _parse_global(symbol)


@pytest.mark.parametrize("kind", ["dynamic", "baked", "wrapper", "runtime_leading"])
def test_pointer_abi_specializations_have_distinct_symbols(kind: str) -> None:
    # Both pointer type facts vary independently across each supported tensor
    # schema encoding. They are absent from the wrapper's generic Pointer AST.
    kernel = SimpleNamespace(__name__="pointer_kernel")
    schemas = []
    symbols = []
    for dtype in (torch.float16, torch.float32):
        for alignment in (16, 8):
            pointer_dtype = str(dtype)
            if kind == "dynamic":
                entry = ("tensor", pointer_dtype, 2)
            elif kind == "baked":
                entry = ("tensor", pointer_dtype, 2, (4, 8), (8, 1))
            elif kind == "wrapper":
                entry = ("wrapper_tensor", "extra", pointer_dtype, 2, (4, 8), (8, 1))
            else:
                entry = (
                    "wrapper_tensor_runtime_leading_extent",
                    "extra",
                    pointer_dtype,
                    2,
                    (8,),
                    (8, 1),
                )
            if alignment != 16:
                entry = (*entry, alignment)
            schema = (entry,)
            schemas.append(schema)
            wrapper = _create_cute_wrapper(kernel, schema, (32, 1, 1))
            symbols.append(_symbol(wrapper, 2.0))
    assert len(set(symbols)) == 4
    for schema, symbol in zip(schemas, symbols, strict=True):
        repeated = _create_cute_wrapper(kernel, schema, (32, 1, 1))
        assert _symbol(repeated, 2.0) == symbol
        _parse_global(symbol)
