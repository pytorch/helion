from __future__ import annotations

from pathlib import Path
import re
from typing import Callable
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler._dynamo.variables import torch_compile_captures_kernel_hop
from helion._compiler.host_function import HostFunction
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.kernel import Kernel

basic_kernels = import_path(Path(__file__).parent / "data/basic_kernels.py")

_INDEX_DIVISOR = 4
_SPLIT = 4
_SCALE = 0.5
_FIRST = 2
_SECOND = 3
_SHARED = 4


def _plus_shared(v: torch.Tensor) -> torch.Tensor:
    return v + _SHARED


@helion.kernel(static_shapes=True, autotune_effort="none")
def add_block_index(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + (tile.index // _INDEX_DIVISOR).to(x.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def add_split_thrice(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, 2 * _SPLIT)
    for tile in hl.tile(x.size(0)):
        out[tile] = out[tile] + x[tile] + _SPLIT
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * _SCALE
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def apply_fn(x: torch.Tensor, fn) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = fn(x[tile], tile)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def add_shared_and_apply(x: torch.Tensor, fn) -> torch.Tensor:
    """Reads ``_SHARED`` in the body and, through ``fn``'s lifted globals,
    again: two reads that each register the global."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = fn(x[tile]) + _SHARED
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def pick(x: torch.Tensor) -> torch.Tensor:
    if hl.specialize(x.size(0)) > 32:
        factor = _FIRST
    else:
        factor = _SECOND
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * factor
    return out


# Where torch.compile traces a kernel call inline, a module-level int global
# stays a kernel argument (host_function._specializes_module_ints), so only
# its values are checked there.
requires_specialization = unittest.skipUnless(
    torch_compile_captures_kernel_hop(_get_backend()),
    "module-level ints stay kernel arguments where torch.compile traces calls",
)


def _host_code(code: str) -> str:
    """The generated code without its source-line comments."""
    return "\n".join(line for line in code.splitlines() if "src[" not in line)


def _expected_index(block: int) -> torch.Tensor:
    return (torch.arange(64, device=DEVICE) // block).float()


class TestGlobalIntConstants(TestCase):
    @skipIfRefEager("checks the generated code, which ref mode does not build")
    @requires_specialization
    def test_int_global_is_a_compile_time_constant(self) -> None:
        x = torch.zeros(64, device=DEVICE)
        code, out = code_and_output(add_block_index, (x,))
        torch.testing.assert_close(out, _expected_index(4))
        self.assertNotIn("_INDEX_DIVISOR", _host_code(code))

    @skipIfRefEager("checks bindings, which ref mode does not key on globals")
    @requires_specialization
    def test_assigning_an_int_global_compiles_again(self) -> None:
        x = torch.zeros(64, device=DEVICE)
        torch.testing.assert_close(add_block_index(x), _expected_index(4))
        first = add_block_index.bind((x,))
        with patch.dict(add_block_index.fn.__globals__, _INDEX_DIVISOR=8):
            torch.testing.assert_close(add_block_index(x), _expected_index(8))
            self.assertIsNot(add_block_index.bind((x,)), first)
        torch.testing.assert_close(add_block_index(x), _expected_index(4))
        self.assertIs(add_block_index.bind((x,)), first)

    @skipIfRefEager("checks bindings, which ref mode does not key on globals")
    @requires_specialization
    def test_binding_is_keyed_by_the_value_it_compiled(self) -> None:
        # Another thread assigns the global between a binding's trace and its
        # caching: a cold bind keys after the trace (4 traced, 8 keyed), a
        # warm one looks up before it (8 looked up, 16 traced).  Either way
        # the binding is filed under the value its code holds, so a call once
        # the global settles at 8 computes with 8.
        x = torch.zeros(64, device=DEVICE)
        globals_ = add_block_index.fn.__globals__
        for method, warm in (
            ("_create_bound_kernel_cache_key", False),
            ("_get_bound_kernel_cache_key", True),
        ):
            with self.subTest(method=method), patch.dict(globals_, _INDEX_DIVISOR=4):
                kernel = helion.kernel(
                    add_block_index.fn, static_shapes=True, autotune_effort="none"
                )
                if warm:
                    kernel.bind((x,))
                    globals_["_INDEX_DIVISOR"] = 8
                original = getattr(Kernel, method)
                flips = [16 if warm else 8]

                def racing(
                    self: Kernel,
                    *args: object,
                    _original: Callable[..., object] = original,
                    _warm: bool = warm,
                    _flips: list[int] = flips,
                    **kwargs: object,
                ) -> object:
                    if not _warm and _flips:
                        globals_["_INDEX_DIVISOR"] = _flips.pop()
                    key = _original(self, *args, **kwargs)
                    if _warm and _flips:
                        globals_["_INDEX_DIVISOR"] = _flips.pop()
                    return key

                with patch.object(Kernel, method, racing):
                    kernel.bind((x,))
                self.assertEqual(flips, [])
                globals_["_INDEX_DIVISOR"] = 8
                torch.testing.assert_close(kernel(x), _expected_index(8))
                if warm:
                    globals_["_INDEX_DIVISOR"] = 16
                    torch.testing.assert_close(kernel(x), _expected_index(16))

    @skipIfRefEager("checks bindings, which ref mode does not key on globals")
    @requires_specialization
    def test_late_specialization_keeps_the_compiled_value(self) -> None:
        # A specialization a backend publishes after the trace (here at once,
        # after the global moved from 4 to 8) re-keys the binding: under the
        # value it compiled, so a call at 8 compiles its own.
        x = torch.zeros(64, device=DEVICE)
        globals_ = add_block_index.fn.__globals__
        kernel = helion.kernel(
            add_block_index.fn, static_shapes=True, autotune_effort="none"
        )
        with patch.dict(globals_, _INDEX_DIVISOR=4):
            bound = kernel.bind((x,))
            globals_["_INDEX_DIVISOR"] = 8
            kernel._extend_bound_kernel_specializations(
                bound, bound._base_spec_key, [lambda args: 0], (x,)
            )
            torch.testing.assert_close(kernel(x), _expected_index(8))
            globals_["_INDEX_DIVISOR"] = 4
            torch.testing.assert_close(kernel(x), _expected_index(4))

    @skipIfRefEager("checks bindings, which ref mode does not key on globals")
    @requires_specialization
    def test_one_value_per_global_in_a_trace(self) -> None:
        # The body and the lifted helper each read _SHARED, and another thread
        # assigns 8 after the first read.  The trace keeps the first read's 4
        # for both, so the program never mixes 4 and 8 (4 + 8) and is keyed
        # by 4: once the global settles at 8, a call computes 8 + 8.
        x = torch.zeros(64, device=DEVICE)
        globals_ = add_shared_and_apply.fn.__globals__
        kernel = helion.kernel(
            add_shared_and_apply.fn, static_shapes=True, autotune_effort="none"
        )
        original = HostFunction.register_fake
        reads: list[object] = []

        def racing(self: HostFunction, obj: object, origin: object) -> object:
            value = original(self, obj, origin)  # pyrefly: ignore[bad-argument-type]
            if getattr(origin, "key", None) == "_SHARED":
                reads.append(obj)
                globals_["_SHARED"] = 8
            return value

        with patch.dict(globals_, _SHARED=4):
            with patch.object(HostFunction, "register_fake", racing):
                bound = kernel.bind((x, _plus_shared))
            self.assertEqual(reads[:2], [4, 8])
            self.assertEqual(list(bound.env.specialized_globals.values()), [4])
            # A call with a function argument binds afresh (at 8); the
            # binding's own program runs on 4 alone.
            run = bound.compile_config(bound.config_spec.default_config())
            torch.testing.assert_close(run(x, _plus_shared), x + 8)
            torch.testing.assert_close(kernel(x, _plus_shared), x + 16)

    @skipIfRefEager("checks bindings, which ref mode does not key on globals")
    @requires_specialization
    def test_dispatch_entry_is_keyed_by_the_value_it_compiled(self) -> None:
        # The fast path publishes a dispatch entry read before the binding it
        # maps to; an entry that read the global at another value than the
        # binding compiled (assigned in between) is not published.
        x = torch.zeros(64, device=DEVICE)
        globals_ = add_block_index.fn.__globals__
        kernel = helion.kernel(
            add_block_index.fn, static_shapes=True, autotune_effort="none"
        )
        with patch.dict(globals_, _INDEX_DIVISOR=8):
            torch.testing.assert_close(kernel(x), _expected_index(8))
            entry = kernel._fast_dispatch_key_and_guards((x,))
            assert entry is not None
            at_8 = kernel.bind((x,))
            globals_["_INDEX_DIVISOR"] = 16
            torch.testing.assert_close(kernel(x), _expected_index(16))
            at_16 = kernel.bind((x,))
            self.assertIsNot(at_16, at_8)
            self.assertIsNone(kernel._prepare_dispatch_entry((x,), at_16, entry))
            self.assertIsNotNone(kernel._prepare_dispatch_entry((x,), at_8, entry))

    @skipIfRefEager("checks the generated code, which ref mode does not build")
    @requires_specialization
    def test_host_and_device_code_read_one_value(self) -> None:
        # _SPLIT is read on the host (the fill value) and on the device.  The
        # generated program reads the value it was compiled with in both,
        # never the module's, so it cannot mix two values of the global.
        x = torch.zeros(64, device=DEVICE)
        code, out = code_and_output(add_split_thrice, (x,))
        self.assertNotIn("_SPLIT", _host_code(code))
        torch.testing.assert_close(out, x + 12)
        with patch.dict(add_split_thrice.fn.__globals__, _SPLIT=2):
            torch.testing.assert_close(add_split_thrice(x), x + 6)
        torch.testing.assert_close(add_split_thrice(x), x + 12)

    @skipIfRefEager("checks the generated code, which ref mode does not build")
    @requires_specialization
    def test_int_global_of_a_function_argument_is_a_constant(self) -> None:
        x = torch.zeros(64, device=DEVICE)
        code, out = code_and_output(apply_fn, (x, basic_kernels.add_global_int))
        torch.testing.assert_close(out, x + 3)
        self.assertIsNone(re.search(r"\.global_int\b", _host_code(code)))
        with patch.dict(vars(basic_kernels), global_int=5):
            torch.testing.assert_close(apply_fn(x, basic_kernels.add_global_int), x + 5)
        torch.testing.assert_close(apply_fn(x, basic_kernels.add_global_int), x + 3)

    def test_global_read_by_a_later_binding_keys_it(self) -> None:
        # The specialized size picks which of _FIRST and _SECOND a binding
        # reads, so the binding of the short input reads a global the first
        # binding did not, and it has to key the schema that one published.
        long, short = torch.ones(64, device=DEVICE), torch.ones(16, device=DEVICE)
        globals_ = pick.fn.__globals__
        torch.testing.assert_close(pick(long), long * 2)
        torch.testing.assert_close(pick(short), short * 3)
        with patch.dict(globals_, _SECOND=5):
            torch.testing.assert_close(pick(short), short * 5)
            torch.testing.assert_close(pick(long), long * 2)
        with patch.dict(globals_, _FIRST=7):
            torch.testing.assert_close(pick(long), long * 7)
        torch.testing.assert_close(pick(short), short * 3)

    def test_float_global_stays_a_runtime_argument(self) -> None:
        x = torch.ones(64, device=DEVICE)
        torch.testing.assert_close(scale(x), x * 0.5)
        first = scale.bind((x,))
        with patch.dict(scale.fn.__globals__, _SCALE=1.5):
            torch.testing.assert_close(scale(x), x * 1.5)
            self.assertIs(scale.bind((x,)), first)
