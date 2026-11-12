"""What the later loops of a rolled reduction compute again.

The reduction roller splits a row's work into loops over chunks of the
reduction dim and adds to a later loop the values of an earlier one that it
reads, computing them again chunk by chunk.  A value the earlier loop read
from outside the loops is passed in instead.  A value with a side effect (an
atomic, asm not declared pure), or a load of memory written since it first
ran, cannot run again: such a reduction dim stays persistent, and a config
saved while it rolled still loads.
"""

from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.cpu_only,
    # The tests read the device IR and the generated code, which
    # HELION_INTERPRET=1 bypasses.
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1", reason="requires compiler IR"
    ),
    skipUnlessBackends(["triton", "cute"]),
]


def _kernel_body(code: str) -> list[ast.stmt]:
    """The statements of the device function in generated ``code``."""
    return next(
        node.body
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(item) in ("triton.jit", "cute.kernel")
            for item in node.decorator_list
        )
    )


def _atomic_result_after_a_row_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        old = hl.atomic_add(x, [row, slice(None)], 1.0)
        total = y[row, :].sum(-1, keepdim=True)
        out[row, :] = old * total
    return out


def _row_zeroed_before_its_reuse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1, keepdim=True)
        x[row, :] = 0.0
        out[row, :] = r - top
    return out


def _row_head_overwritten_before_its_reuse(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        x[row, 0] = top
        out[row, :] = r - top[:, None]
    return out


def _clock_read_before_its_row_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        stamp = hl.inline_asm_elementwise(
            "mov.u32 $0, %clock;", "=r", [x[row, :]], torch.int32, False, 1
        ).to(torch.float32)
        out[row, :] = stamp - stamp.amax(-1, keepdim=True)
    return out


@pytest.mark.parametrize(
    "fn",
    [
        _atomic_result_after_a_row_sum,
        _row_zeroed_before_its_reuse,
        _row_head_overwritten_before_its_reuse,
        _clock_read_before_its_row_max,
    ],
)
def test_a_value_that_cannot_run_again_keeps_the_dim_persistent(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    # The loop after the reduction needs the row's ``old``, ``r`` or
    # ``stamp``.  Rolled, it ran the atomic again (``x`` incremented twice,
    # ``old`` read back incremented), reloaded the row after the store of
    # zeros, or (the store placed between the loops) of its maximum into
    # column 0, or read the clock again.  The dim keeps its slot, pinned
    # persistent, for the configs older versions saved.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (torch.zeros(4, 32), torch.ones(4, 32)),
        )
        slots = bound.config_spec.reduction_loops
        assert [(slot.size_hint, slot.pinned) for slot in slots] == [(32, True)]
        # A config saved while the dim rolled loads and runs it persistent.
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    body = _kernel_body(code)
    assert not [statement for statement in body if isinstance(statement, ast.For)]
    atomics = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "atomic_add"
    ]
    assert len(atomics) == (fn is _atomic_result_after_a_row_sum)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(("columns", "compiles"), [(512, True), (8192, False)])
def test_a_refused_row_past_the_thread_budget_fails_on_cute(
    columns: int, compiles: bool
) -> None:
    # The refused dim stays persistent, with no rolled config to fall back
    # to, and CuTe's persistent lowering cannot share a row the atomic
    # writes among threads once it is wider than the thread budget: at 8192
    # columns the kernel no longer compiles, where older versions compiled
    # a rolled config with wrong results.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_atomic_result_after_a_row_sum, static_shapes=True),
            (torch.zeros(64, columns), torch.ones(64, columns)),
        )
        slots = bound.config_spec.reduction_loops
        assert [(slot.size_hint, slot.pinned) for slot in slots] == [(columns, True)]
        config = bound.config_spec.default_config()
        if compiles:
            bound.to_code(config)
        else:
            with pytest.raises(helion.exc.BackendUnsupported):
                bound.to_code(config)


def _row_zeroed_through_another_argument(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1, keepdim=True)
        y[row, :] = 0.0
        out[row, :] = r - top
    return out


@pytest.mark.parametrize(
    ("args", "rolls"),
    [
        (lambda: (torch.ones(4, 32), torch.ones(4, 32)), True),
        (lambda: (x := torch.ones(4, 32), x), False),
        (lambda: ((b := torch.ones(4, 48))[:, :32], b[:, 16:]), False),
    ],
    ids=["distinct", "same-tensor", "overlapping-views"],
)
def test_a_store_through_an_aliased_argument_keeps_the_dim_persistent(
    args: Callable[[], tuple[torch.Tensor, torch.Tensor]], rolls: bool
) -> None:
    # The store of zeros through ``y`` comes before the loop that reloads
    # the row of ``x``.  Different arguments may still be one tensor, or
    # overlapping views: they count as disjoint only by the runtime storage
    # fact the bound kernel is keyed on, so an aliased call binds again and
    # keeps the row persistent.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_zeroed_through_another_argument, static_shapes=True),
            args(),
        )
        slots = bound.config_spec.reduction_loops
        assert [(slot.size_hint, slot.pinned) for slot in slots] == [(32, not rolls)]
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    sweeps = [node for node in _kernel_body(code) if isinstance(node, ast.For)]
    assert bool(sweeps) == rolls


def test_an_aliased_call_binds_a_kernel_of_its_own() -> None:
    # One kernel object: views of two tensors roll, and the storage fact
    # their bind registered keys that bound kernel, so views of one tensor
    # with the same layout bind again and keep the row persistent; a second
    # call with distinct tensors reuses the first bound kernel.
    kernel = helion.kernel(_row_zeroed_through_another_argument, static_shapes=True)

    def views(shared: bool) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.ones(4, 48)
        return base[:, :32], (base if shared else torch.ones(4, 48))[:, 16:]

    with cpu_render_target():
        distinct = kernel.bind(views(False))
        overlapping = kernel.bind(views(True))
        assert kernel.bind(views(False)) is distinct
        x = torch.ones(4, 32)
        same = kernel.bind((x, x))
    assert overlapping is not distinct
    for bound, rolls in ((distinct, True), (overlapping, False), (same, False)):
        slots = bound.config_spec.reduction_loops
        assert [slot.pinned for slot in slots] == [not rolls]


def _row_zeroed_through_a_stack_tensor(
    x: torch.Tensor, template: torch.Tensor, dev_ptrs: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1, keepdim=True)
        rows = hl.stacktensor_like(template, dev_ptrs[0])
        rows[row, :] = 0.0
        out[row, :] = r - top
    return out


def _row_stored_by_asm(x: torch.Tensor, addrs: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    flags = torch.empty(x.size(0), dtype=torch.int32, device=x.device)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        flags[row] = hl.inline_asm_elementwise(
            "st.global.f32 [$1], $2; mov.u32 $0, 0;",
            "=r,l,f",
            [addrs[row], top],
            torch.int32,
            False,
            1,
        )
        out[row, :] = r - top[:, None]
    return out


def _row_stored_by_inline_triton(x: torch.Tensor, addrs: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        hl.inline_triton(
            "tl.store({0}.to(tl.pointer_type(tl.float32)), {1})\n{1}",
            args=(addrs[row], top),
            output_like=top,
        )
        out[row, :] = r - top[:, None]
    return out


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (
            _row_zeroed_through_a_stack_tensor,
            lambda: (
                torch.ones(4, 32),
                torch.ones(4, 32),
                torch.zeros(1, dtype=torch.uint64),
            ),
        ),
        (
            _row_stored_by_asm,
            lambda: (torch.ones(4, 32), torch.zeros(4, dtype=torch.int64)),
        ),
        pytest.param(
            _row_stored_by_inline_triton,
            lambda: (torch.ones(4, 32), torch.zeros(4, dtype=torch.int64)),
            marks=skipUnlessBackends(["triton"]),
        ),
    ],
    ids=["stack-tensor", "asm", "inline-triton"],
)
def test_a_store_through_a_computed_address_keeps_the_dim_persistent(
    fn: Callable[..., torch.Tensor], args: Callable[[], tuple[torch.Tensor, ...]]
) -> None:
    # The store of zeros through a stack tensor's pointer, the asm not
    # declared pure and the snippet given 64-bit integers may write the row
    # of ``x`` (the pointer and the addresses can be ``x``'s) between the
    # loops, before the one that reloads it: they count as writing any of
    # the kernel's memory.
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(fn, static_shapes=True), args())
    slots = bound.config_spec.reduction_loops
    assert [(slot.size_hint, slot.pinned) for slot in slots] == [(32, True)]


def _flag_stored_by_asm_before_a_row_sum_lands(
    x: torch.Tensor, addrs: torch.Tensor
) -> torch.Tensor:
    sums = torch.empty(x.size(0), device=x.device)
    flags = torch.empty(x.size(0), dtype=torch.int32, device=x.device)
    for row in hl.tile(x.size(0)):
        total = x[row, :].sum(-1)
        flags[row] = hl.inline_asm_elementwise(
            "st.global.f32 [$1], $2; mov.u32 $0, 0;",
            "=r,l,f",
            [addrs[row], x[row, 0]],
            torch.int32,
            False,
            1,
        )
        sums[row] = total
    return sums


@skipUnlessBackends(["triton"])
def test_an_outer_write_closes_the_pending_loop() -> None:
    # The asm stores through ``addrs`` after the row's loads of ``x`` in
    # program order; it needs nothing from the loop, but placed outside the
    # loops ahead of it, it would store before those loads (``addrs`` can be
    # ``x``'s).  An outer write closes the pending loop first.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(
                _flag_stored_by_asm_before_a_row_sum_lands, static_shapes=True
            ),
            (torch.ones(4, 64), torch.zeros(4, dtype=torch.int64)),
        )
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    body = _kernel_body(code)
    (loop,) = [index for index, node in enumerate(body) if isinstance(node, ast.For)]
    (asm,) = [
        index
        for index, node in enumerate(body)
        if "tl.inline_asm_elementwise(" in ast.unparse(node)
    ]
    assert loop < asm, code


def _row_cleared_through_a_stack_tensor(
    x: torch.Tensor, template: torch.Tensor, dev_ptrs: torch.Tensor
) -> torch.Tensor:
    tops = torch.empty(x.size(0), device=x.device)
    for row in hl.tile(x.size(0)):
        tops[row] = x[row, :].amax(-1)
        rows = hl.stacktensor_like(template, dev_ptrs[0])
        rows[row, :] = 0.0
    return tops


@skipUnlessBackends(["triton"])
def test_a_scalar_store_through_a_stack_tensor_rolls_over_the_row() -> None:
    # The store of a scalar spans the row through its subscript, which
    # indexes the stack tensor's tensor-like: a later loop stores every
    # chunk, rather than one chunk's width outside the loops.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_cleared_through_a_stack_tensor, static_shapes=True),
            (torch.ones(4, 64), torch.ones(4, 64), torch.zeros(1, dtype=torch.uint64)),
        )
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    loops = [node for node in _kernel_body(code) if isinstance(node, ast.For)]
    assert len(loops) == 2, code
    assert "0.0" in ast.unparse(loops[1]), code
    assert "tl.arange(0, 16)" not in code, code


def _row_prefetched_before_its_reuse(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        hl.prefetch(x, [row, slice(None)])
        top = r.amax(-1, keepdim=True)
        out[row, :] = r - top
    return out


def test_a_prefetch_keeps_the_roll() -> None:
    # A prefetch writes nothing and orders nothing: the loop after the
    # row's maximum reloads the row as it does without one.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_prefetched_before_its_reuse, static_shapes=True),
            (torch.randn(4, 64),),
        )
    slots = bound.config_spec.reduction_loops
    assert [(slot.size_hint, slot.pinned) for slot in slots] == [(64, False)]
    assert bound.host_function.device_ir.rolled_reductions


def _inline_triton_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        e = hl.inline_triton("tl.exp({0})", args=(r,), output_like=r)
        out[row, :] = e / e.sum(-1, keepdim=True)
    return out


@skipUnlessBackends(["triton"])
def test_an_inline_triton_snippet_keeps_the_dim_persistent() -> None:
    # A snippet may write the pointers it is given and has no purity flag,
    # so the loop after the reduction does not run it again, pure or not.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_inline_triton_softmax, static_shapes=True),
            (torch.randn(4, 64),),
        )
    slots = bound.config_spec.reduction_loops
    assert [(slot.size_hint, slot.pinned) for slot in slots] == [(64, True)]


def _row_head_overwritten_by_inline_triton(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        top = hl.inline_triton(
            "tl.store({0} + {1} * 32, {2})\n{2}",
            args=(x, row.index, top),
            output_like=top,
        )
        out[row, :] = r - top[:, None]
    return out


def _row_head_overwritten_by_inline_triton_in_a_loop(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        for _ in hl.tile(1):
            hl.inline_triton(
                "tl.store({0} + {1} * 32, {2})\n{2}",
                args=(x, row.index, top),
                output_like=top,
            )
        out[row, :] = r - top[:, None]
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    "fn",
    [
        _row_head_overwritten_by_inline_triton,
        _row_head_overwritten_by_inline_triton_in_a_loop,
    ],
)
def test_a_snippet_storing_to_a_reloaded_row_keeps_the_dim_persistent(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    # The snippet between the loops stores the row's maximum into its first
    # element, through the pointer it gets, as ``x[row, 0] = top`` would.
    # It names no store target, so the loop after the reduction reloaded the
    # row after it; inside control flow too.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (torch.zeros(4, 32), torch.ones(4, 32)),
        )
    slots = bound.config_spec.reduction_loops
    assert [(slot.size_hint, slot.pinned) for slot in slots] == [(32, True)]


def _two_rows_rerunning_atomics(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out2 = torch.empty_like(y)
    for row in hl.tile(x.size(0)):
        a = hl.atomic_add(x, [row, slice(None)], 1.0)
        out[row, :] = a * a.sum(-1, keepdim=True)
        b = hl.atomic_add(y, [row, slice(None)], 1.0)
        out2[row, :] = b * b.sum(-1, keepdim=True)
    return out, out2


def _rerunning_atomic_beside_a_row_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)
    for row in hl.tile(x.size(0)):
        a = hl.atomic_add(x, [row, slice(None)], 1.0)
        out[row, :] = a * a.sum(-1, keepdim=True)
        sums[row] = y[row, :].sum(-1)
    return out, sums


@pytest.mark.parametrize(
    "fn", [_two_rows_rerunning_atomics, _rerunning_atomic_beside_a_row_sum]
)
def test_a_refused_dim_keeps_its_saved_slot_and_graph(
    fn: Callable[[torch.Tensor, torch.Tensor], object],
) -> None:
    # Two dims share the row loop's graph, and the first refuses its roll.
    # Older versions rolled the first and so skipped the second: their
    # configs have one entry, for the first.  The first keeps that slot,
    # pinned persistent, so the entry still loads and is not applied to the
    # second, which still gets no slot.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (torch.zeros(4, 32), torch.zeros(4, 64)),
        )
        spec = bound.config_spec
        slots = spec.reduction_loops
        assert [(slot.size_hint, slot.pinned) for slot in slots] == [(32, True)]
        for saved in ([16], [None]):
            config = helion.Config(block_sizes=[1], reduction_loops=saved)
            spec.normalize(config)
            assert config.config.get("reduction_loops", [None]) == [None]


def _two_rerunning_atomics_then_a_row_sum(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    out_z = torch.empty_like(z)
    for row in hl.tile(x.size(0)):
        a = hl.atomic_add(x, [row, slice(None)], 1.0)
        out_x[row, :] = a * a.sum(-1, keepdim=True)
    for row in hl.tile(y.size(0)):
        b = hl.atomic_add(y, [row, slice(None)], 1.0)
        out_y[row, :] = b * b.sum(-1, keepdim=True)
    for row in hl.tile(z.size(0)):
        out_z[row, :] = z[row, :] / z[row, :].sum(-1, keepdim=True)
    return out_x, out_y, out_z


def test_a_saved_config_with_trailing_entries_left_off_loads() -> None:
    # Older versions offered each loop's row a slot: the configs they saved
    # have three entries, the first two for the rows the roller now refuses.
    # Those rows keep their slots, pinned persistent, so the entries line up
    # by position, also in a config that leaves off the last.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_two_rerunning_atomics_then_a_row_sum, static_shapes=True),
            (torch.zeros(4, 32), torch.zeros(4, 64), torch.ones(4, 128)),
        )
        spec = bound.config_spec
        assert [(loop.size_hint, loop.pinned) for loop in spec.reduction_loops] == [
            (32, True),
            (64, True),
            (128, False),
        ]
        for saved, loaded in (
            ([16, None, 64], [None, None, 64]),
            ([16, 16], [None, None, None]),
        ):
            config = helion.Config(block_sizes=[1, 1, 1], reduction_loops=saved)
            spec.normalize(config)
            assert config.config.get("reduction_loops", [None] * 3) == loaded


def _row_stored_then_read_back(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1)
        y[row, :] = r - top[:, None]
        out[row] = y[row, 0]
    return out


def test_an_outer_load_follows_the_loop_storing_its_tensor() -> None:
    # ``y[row, 0]`` has no reduction-dim axis, so it goes outside the loops,
    # and it went there as soon as its inputs were ready: ahead of the loop
    # that stores the row it reads, so ``out`` got the old ``y[row, 0]``.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_stored_then_read_back, static_shapes=True),
            (torch.randn(4, 64), torch.randn(4, 64)),
        )
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    touching_y = [
        statement
        for statement in _kernel_body(code)
        if any(
            isinstance(node, ast.Name) and node.id == "y"
            for node in ast.walk(statement)
        )
    ]
    assert isinstance(touching_y[0], ast.For)
    assert not isinstance(touching_y[-1], ast.For)


def _load_fusion_moves_beside_another_store(
    x: torch.Tensor, b: torch.Tensor, c: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    out2 = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        b[row, :] = r * 2
        t = r.sum(-1) * 3
        c0 = c[row, 0]
        out[row] = t + c0
        out2[row] = q[row, :].sum(-1)
    return out, out2


@skipUnlessBackends(["triton"])
def test_codegen_trusts_no_alias_fact_the_bind_did_not_record() -> None:
    # Fusing the two sums keeps the loop storing ``b`` open when ``c[row, 0]``
    # comes, which the natural order the bind analyzes never compared.
    # Registered at codegen, the storage fact did not key the bound kernel:
    # a later call passing a view of ``b`` as ``c`` reused the code that
    # loads ``c`` ahead of that loop.  The load now follows it.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_load_fusion_moves_beside_another_store, static_shapes=True),
            tuple(torch.randn(4, 64) for _ in range(4)),
        )
        facts = set(bound.env.runtime_input_specializations)
        code = bound.to_code(
            helion.Config(block_sizes=[1], reduction_loops=[16], reduction_fusion=True)
        )
        assert set(bound.env.runtime_input_specializations) == facts
    body = _kernel_body(code)

    def first(name: str, kind: type[ast.stmt]) -> int:
        return next(
            index
            for index, statement in enumerate(body)
            if isinstance(statement, kind)
            and any(
                isinstance(node, ast.Name) and node.id == name
                for node in ast.walk(statement)
            )
        )

    assert first("c", ast.stmt) > first("b", ast.For)


def _scale_read_before_its_store(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        s = scale[row, :]
        r = x[row, :] * s
        top = r.amax(-1, keepdim=True)
        scale[row, :] = top
        out[row, :] = torch.exp(r - top)
    return out


def test_a_value_read_from_outside_the_loops_is_passed_to_the_next_one() -> None:
    # ``s`` is one value per row, loaded outside the loops.  The loop after
    # the reduction recomputes ``r`` and loaded ``s`` again on every
    # iteration, after the store between the loops had replaced it with the
    # row's maximum: exp(x * top - top) instead of exp(x * s - top).
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_scale_read_before_its_store, static_shapes=True),
            (torch.randn(4, 64), torch.randn(4, 1)),
        )
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    sweeps = [
        statement for statement in _kernel_body(code) if isinstance(statement, ast.For)
    ]
    assert len(sweeps) == 2
    assert not any(
        isinstance(node, ast.Name) and node.id == "scale"
        for sweep in sweeps
        for node in ast.walk(sweep)
    )


def _layer_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        mean = r.sum(-1, keepdim=True) / x.size(1)
        centered = r - mean
        var = (centered * centered).sum(-1, keepdim=True) / x.size(1)
        out[row, :] = centered * torch.rsqrt(var + 1e-5) * w[None, :]
    return out


def test_a_value_computed_outside_the_loops_is_passed_to_the_last_one() -> None:
    # ``mean`` is computed outside the loops from the first loop's sum.  The
    # second loop reads it as an argument, and the third, which recomputes
    # ``centered``, divided the sum again.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_layer_norm, static_shapes=True),
            (torch.randn(4, 64), torch.randn(64)),
        )
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    sweeps = [s for s in _kernel_body(code) if isinstance(s, ast.For)]
    assert len(sweeps) == 3
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        for node in ast.walk(sweeps[-1])
    )


def _softmax_in_place(x: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        top = r.amax(-1, keepdim=True)
        e = torch.exp(r - top)
        x[row, :] = e / e.sum(-1, keepdim=True)
    return x


def _normalize_with_stats_args(
    x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        row_mean = r.mean(-1)
        mean[row] = row_mean
        centered = r - row_mean[:, None]
        row_rstd = torch.rsqrt((centered * centered).mean(-1) + 1e-5)
        rstd[row] = row_rstd
        out[row, :] = centered * row_rstd[:, None]
    return out


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (_softmax_in_place, lambda: (torch.randn(4, 64),)),
        (
            _normalize_with_stats_args,
            lambda: (torch.randn(4, 64), torch.empty(4), torch.empty(4)),
        ),
    ],
)
def test_stores_that_no_rerun_reads_keep_the_roll(
    fn: Callable[..., torch.Tensor], args: Callable[[], tuple[torch.Tensor, ...]]
) -> None:
    # The in-place store comes after the last loop reloads the row; the
    # stores between the loops write other tensors.
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(fn, static_shapes=True), args())
        assert len(bound.config_spec.reduction_loops) == 1
        code = bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[16]))
    assert len([s for s in _kernel_body(code) if isinstance(s, ast.For)]) == 3
