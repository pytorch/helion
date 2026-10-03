from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch
from torch._inductor.utils import fresh_cache

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _multiple(values, initial, steps):
    steps = hl.specialize(steps)
    batch, _, features = values.shape
    history = torch.empty_like(values)
    final = torch.empty_like(initial)
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        total = initial[bi.begin, fi].float()
        rounded = hl.full([fi], 0.25, dtype=torch.bfloat16)
        count = hl.full([fi], 2, dtype=torch.int32)
        for qi in hl.grid(steps):
            total = total + values[bi.begin, qi, fi].float()
            rounded = (rounded.float() * 0.5 + total).to(torch.bfloat16)
            count = count + 1
            history[bi.begin, qi, fi] = rounded.to(history.dtype)
        final[bi.begin, fi] = total + rounded.float() + count.float()
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_final(values, final):
    batch, steps, features = values.shape
    history = torch.empty_like(values)
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        total = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            total = total + values[bi.begin, qi, fi]
            history[bi.begin, qi, fi] = total
        final[bi.begin, fi] = total
    return history


def _config(width=4, mode=None):
    extra = (
        {}
        if mode is None
        else {
            "cute_loop_vectorize": True,
            "cute_loop_load_schedule": mode,
        }
    )
    return helion.Config(
        block_sizes=[64],
        num_threads=[64 // width],
        cute_vector_widths=[1, width, 1],
        **extra,
    )


def _inputs(steps=7, features=64, device="cpu", stride=1):
    values = torch.randn(2, max(steps, 1), features * stride, device=device)[
        ..., ::stride
    ]
    initial = torch.randn(2, features * stride, device=device)[..., ::stride]
    return values, initial, steps


def test_typed_loop_interface_and_single_kernel():
    with _cpu_codegen():
        code = _multiple._bind_isolated(_inputs()).to_code(_config())
    assert "tile_loop_value" in code
    for dtype in ("Float32", "BFloat16", "Int32"):
        assert f"cute.make_rmem_tensor((4,), cutlass.{dtype})" in code
    assert "serial_lane_guard" not in code
    assert "select_kernel" not in code
    tree = ast.parse(code)
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    ]
    assert len(kernels) == 1


def test_schedule_is_planned_before_scalar_body_emission():
    from helion._compiler.cute import loop_state

    plans = []
    original = loop_state.plan_root

    def capture(cg, root, grid):
        plan = original(cg, root, grid)
        if plan is not None:
            # The plan references typed FX values and the assigned layout;
            # no root/body scalar operation has been emitted to inspect.
            for node in (*root.graph.nodes, *plan.loop.graph.nodes):
                assert not cg.statements_owned_by_node(node)
            assert plan.layout.elements == 4
            assert len(plan.inputs) == 3
            assert plan.memory.depth == 4
            assert plan.memory.prefetch
            plans.append(plan)
        return plan

    with _cpu_codegen(), patch.object(loop_state, "plan_root", capture):
        _multiple._bind_isolated(_inputs()).to_code(_config(mode="prefetch4"))
    assert len(plans) == 1


def test_external_final_store_keeps_lane_major():
    with _cpu_codegen():
        values, initial, _ = _inputs()
        code = _external_final._bind_isolated((values, initial)).to_code(_config())
    assert "tile_loop_value" not in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [0, 1, 3, 7])
@pytest.mark.parametrize("features,stride,width", [(64, 1, 4), (67, 1, 2), (64, 2, 4)])
@pytest.mark.parametrize(
    "mode", [None, "current", "group2", "prefetch2", "group4", "prefetch4"]
)
def test_multiple_typed_carries_and_final_consumer(
    steps, features, stride, width, mode
):
    args = _inputs(steps, features, DEVICE, stride)
    values, initial, _ = args
    before = values.clone(), initial.clone()
    bound = _multiple._bind_isolated(args)
    bound.set_config(_config(width, mode))
    actual = bound(*args)
    total = initial.float().clone()
    rounded = torch.full_like(initial, 0.25, dtype=torch.bfloat16)
    history = torch.empty_like(values)
    for step in range(steps):
        total = total + values[:, step].float()
        rounded = (rounded.float() * 0.5 + total).to(torch.bfloat16)
        history[:, step] = rounded
    torch.testing.assert_close(actual[0][:, :steps], history[:, :steps], atol=0, rtol=0)
    torch.testing.assert_close(
        actual[1], total + rounded.float() + (2 + steps), atol=0, rtol=0
    )
    repeated = bound(*args)
    torch.testing.assert_close(repeated[1], actual[1], atol=0, rtol=0)
    torch.testing.assert_close((values, initial), before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fusion", [False, True])
def test_torch_compile_preserves_loop_state(fusion):
    torch._dynamo.reset()
    kernel = helion.kernel(
        _multiple.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        config=_config(mode="prefetch4"),
        torch_compile_fusion=fusion,
    )

    def run(values, initial):
        history, final = kernel(values * 0.5, initial + 0.25, 7)
        return history + 1, final + 1

    values, initial, _ = _inputs(device=DEVICE)
    expected = run(values, initial)
    compiled = torch.compile(run, fullgraph=True)
    with fresh_cache():
        actual = compiled(values, initial)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(compiled(values, initial), expected, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _recurrence(states, coefficient, output_dtype):
    batch, steps, features = states.shape
    out = torch.empty_like(states, dtype=output_dtype)
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], 0.25, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi].float()) * carry
                + states[bi.begin, qi, fi].float()
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather(values, steps, dependent):
    steps = hl.specialize(steps)
    dependent = hl.specialize(dependent)
    size = values.shape[0]
    flat = values.flatten()
    out = torch.empty((steps, size), dtype=values.dtype, device=values.device)
    for fi in hl.tile(size):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            if dependent == 1:
                row = carry.to(torch.int32) % size
            elif dependent == 2:
                row = values[fi, 0].to(torch.int32) % size
            elif dependent == 3:
                row = fi + qi
            else:
                row = (fi + 1) % size
            # One flat index expresses paired/diagonal indexing; two 1-D
            # Helion tile indexers preserve separate dimensions.
            carry = carry + flat[row * size + fi.index]
            out[qi, fi] = carry
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matrix(values):
    steps, rows, columns = values.shape
    out = torch.empty_like(values)
    final = torch.empty((rows, columns), dtype=values.dtype, device=values.device)
    for mi, ni in hl.tile([rows, columns]):
        first = hl.full([mi, ni], 0.25, dtype=torch.float32)
        second = values[0, mi, ni]
        for qi in hl.grid(steps):
            first, second = second, first + values[qi, mi, ni]
            out[qi, mi, ni] = first + second
        final[mi, ni] = first - second
    return out, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _swapped_store(values):
    out = torch.empty_like(values)
    transposed = out.T
    for mi, ni in hl.tile(values.shape):
        carry = hl.zeros([mi, ni], dtype=torch.float32)
        for _qi in hl.grid(3):
            carry = carry + values[mi, ni]
            out[mi, ni] = carry
            transposed[mi, ni] = carry
    return out


@pytest.mark.parametrize(
    "mode", ["current", "group2", "prefetch2", "group4", "prefetch4"]
)
@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 7, 64, 65])
def test_memory_schedule_codegen(mode, steps):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(
            (torch.empty(2, steps, 64), torch.empty(2, steps), torch.bfloat16)
        )
        code = bound.to_code(_config(mode=mode))
        assert code == bound.to_code(bound._normalized_config_copy(_config(mode=mode)))
    assert "cute.make_copy_atom" in code
    assert ("loop_prefetch" in code) == mode.startswith("prefetch")
    assert ("loop_group" in code) == (mode != "current")
    assert "serial_lane_guard" not in code


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_loop_vectorize", "step_major"),
        ("cute_loop_vectorize", 1),
        ("cute_loop_load_schedule", "prefetch8"),
        ("cute_loop_load_schedule", True),
    ],
)
def test_invalid_schedule_configuration(key, value):
    with _cpu_codegen():
        bound = _multiple._bind_isolated(_inputs())
        config = _config(mode="prefetch4")
        config.config[key] = value
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(config)


def test_swapped_fresh_output_views_keep_original_lowering():
    with _cpu_codegen():
        bound = _swapped_store._bind_isolated((torch.empty(64, 64),))
        assert not bound.config_spec.cute_loop_schedule_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", [None, "current", "prefetch4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("alias", [False, True])
def test_raw_calls_offsets_readonly_alias_and_typed_coefficients(mode, dtype, alias):
    # Reuse the same compiled function with a nonzero storage offset while
    # preserving the ordinary launcher's 16-byte input alignment contract.
    def inputs(offset):
        backing = torch.rand(2 * 7 * 64 + 16, device=DEVICE, dtype=dtype) * -0.25
        states = backing[offset : offset + 2 * 7 * 64].view(2, 7, 64)
        coefficient = (
            states[..., 0] if alias else -torch.rand(2, 7, device=DEVICE, dtype=dtype)
        )
        return states, coefficient, dtype

    args = inputs(0)
    bound = _recurrence._bind_isolated(args)
    fn = bound.compile_config(_config(mode=mode))
    for values in (args, inputs(16 // dtype.itemsize)):
        states, coefficient, _ = values
        before = states.clone(), coefficient.clone()
        actual = fn(*values)
        expected = torch.empty_like(states)
        carry = torch.full_like(states[:, 0], 0.25, dtype=torch.float32)
        for step in range(7):
            expected[:, step] = carry
            carry = (
                torch.exp(coefficient[:, step].float())[:, None] * carry
                + states[:, step].float()
            )
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)
        torch.testing.assert_close(fn(*values), actual, atol=0, rtol=0)
        torch.testing.assert_close((states, coefficient), before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dependent", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", [None, "current", "prefetch4"])
def test_gather_and_dependent_addresses(dependent, mode):
    values = (
        torch.arange(64 * 64, device=DEVICE, dtype=torch.float32).reshape(64, 64) % 13
    )
    args = values, 7, dependent
    config = _config(mode=mode)
    config.config["cute_vector_widths"] = [4, 1]
    bound = _gather._bind_isolated(args)
    code = bound.to_code(config)
    if dependent:
        assert "loop_prefetch" not in code
    else:
        # Gather loads must not be replaced by contiguous vector copies.
        assert "tile_loop_value" in code
    bound.set_config(config)
    actual = bound(*args)
    carry = torch.zeros(64, device=DEVICE)
    expected = []
    indices = torch.arange(64, device=DEVICE)
    for _ in range(7):
        row = (
            carry.int() % 64
            if dependent == 1
            else values[:, 0].int() % 64
            if dependent == 2
            else indices + len(expected)
            if dependent == 3
            else (indices + 1) % 64
        )
        carry = carry + torch.where(row < 64, values[row.clamp(max=63), indices], 0)
        expected.append(carry)
    torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("vectorize", [False, True])
def test_multiple_element_groups_two_axes_and_simultaneous_updates(vectorize):
    values = torch.randn(7, 8, 64, device=DEVICE)
    config = helion.Config(
        block_sizes=[4, 64],
        num_threads=[2, 8],
        cute_vector_widths=[1, 4, 1],
        **(
            {"cute_loop_vectorize": True, "cute_loop_load_schedule": "prefetch4"}
            if vectorize
            else {}
        ),
    )
    bound = _matrix._bind_isolated((values,))
    assert "tile_loop_value" in bound.to_code(config)
    bound.set_config(config)
    actual = bound(values)
    first, second = torch.full_like(values[0], 0.25), values[0]
    expected = []
    for step in range(7):
        first, second = second, first + values[step]
        expected.append(first + second)
    torch.testing.assert_close(
        actual, (torch.stack(expected), first - second), atol=0, rtol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _colliding_store(values):
    steps, features = values.shape
    out = torch.empty_like(values)
    for fi in hl.tile(features):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            carry = carry + values[qi, fi]
            out[qi, fi] = carry
            hl.store(out, [qi, fi.begin], 0.0)
    return out


def test_tile_begin_does_not_prove_element_ownership():
    with _cpu_codegen():
        bound = _colliding_store._bind_isolated((torch.empty(7, 64),))
        assert not bound.config_spec.cute_loop_schedule_enabled


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_begin(values):
    steps, features = values.shape
    out = torch.empty_like(values)
    for fi in hl.tile(features):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            carry = carry + values[qi, fi.begin]
            out[qi, fi] = carry
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode", [None, "current", "group2", "prefetch2", "group4", "prefetch4"]
)
def test_tile_begin_load_broadcasts_across_element_groups(mode):
    values = torch.arange(5 * 144, device=DEVICE, dtype=torch.float32).view(5, 144)
    config = _config(mode=mode)
    config.config.update(num_threads=[8], cute_vector_widths=[4, 1])
    bound = _broadcast_begin._bind_isolated((values,))
    code = bound.to_code(config)
    assert "tile_loop_value" in code
    bound.set_config(config)
    columns = torch.arange(144, device=DEVICE) // 64 * 64
    expected = values[:, columns].cumsum(0)
    torch.testing.assert_close(bound(values), expected, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rotate_with_live_initial(values, initial, steps):
    steps = hl.specialize(steps)
    out = torch.empty_like(initial)
    for fi in hl.tile(initial.numel()):
        bias = initial[fi].float()
        first = bias
        second = hl.full([fi], 1.25, dtype=torch.float32)
        third = hl.full([fi], 2.5, dtype=torch.float32)
        for qi in hl.grid(steps):
            first, second, third = second, third, first + values[qi, fi].float()
        out[fi] = first + second * 2 + third * 3 + bias
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("mode", [None, "current", "prefetch4"])
def test_parallel_rotation_preserves_original_prefix_value(steps, mode):
    values = torch.randn(max(steps, 1), 134, device=DEVICE)[:, ::2]
    initial = torch.randn(134, device=DEVICE)[::2]
    before = values.clone(), initial.clone()
    config = _config(mode=mode)
    config.config["cute_vector_widths"] = [4, 1]
    bound = _rotate_with_live_initial._bind_isolated((values, initial, steps))
    assert "tile_loop_initial" in bound.to_code(config)
    bound.set_config(config)
    first = initial.clone()
    second, third = torch.full_like(first, 1.25), torch.full_like(first, 2.5)
    for qi in range(steps):
        first, second, third = second, third, first + values[qi]
    expected = first + second * 2 + third * 3 + initial
    actual = bound(values, initial, steps)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bound(values, initial, steps), actual, atol=0, rtol=0)
    torch.testing.assert_close((values, initial), before, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _subrange(values, begin, end):
    begin, end = hl.specialize(begin), hl.specialize(end)
    out = torch.empty_like(values)
    out.fill_(-123)
    for fi in hl.tile(begin, end):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(values.size(0)):
            carry = carry + values[qi, fi]
            out[qi, fi] = carry
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("begin,end", [(0, 63), (5, 67), (64, 127)])
@pytest.mark.parametrize("mode", [None, "current", "prefetch4"])
def test_vector_transfers_preserve_logical_subrange_masks(begin, end, mode):
    values = torch.arange(5 * 128, device=DEVICE, dtype=torch.float32).view(5, 128)
    config = _config(mode=mode)
    config.config["cute_vector_widths"] = [4, 1]
    bound = _subrange._bind_isolated((values, begin, end))
    code = bound.to_code(config)
    assert "tile_loop_value" in code
    if mode is not None:
        assert "loop_all_valid" in code
    bound.set_config(config)
    expected = torch.full_like(values, -123)
    expected[:, begin:end] = values[:, begin:end].cumsum(0)
    torch.testing.assert_close(bound(values, begin, end), expected, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ranged(values, begin, end, step: hl.constexpr, eviction: hl.constexpr):
    begin, end = hl.specialize(begin), hl.specialize(end)
    out = torch.empty((values.size(1),), dtype=values.dtype, device=values.device)
    for fi in hl.tile(values.size(1)):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(begin, end, step):
            carry = carry + hl.load(values, [qi, fi], eviction_policy=eviction)
        out[fi] = carry
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("begin,step", [(0, None), (0, 2), (2, None)])
@pytest.mark.parametrize("eviction", [None, "evict_last"])
@pytest.mark.parametrize("mode", ["current", "prefetch4"])
@pytest.mark.parametrize("width", [4, 8])
def test_loop_control_and_explicit_eviction_policy(begin, step, eviction, mode, width):
    values = torch.arange(7 * 67, device=DEVICE, dtype=torch.float32).view(7, 67)
    args = values, begin, 7, step, eviction
    # Explicit steps retain the ordinary path and its configuration domain.
    config = _config(width, mode if step is None else None)
    config.config["cute_vector_widths"] = [width, 1]
    bound = _ranged._bind_isolated(args)
    code = bound.to_code(config)
    assert ("tile_loop_value" in code) == (step is None)
    assert ("loop_prefetch" in code) == (
        mode == "prefetch4" and begin == 0 and step is None and eviction is None
    )
    if eviction is not None:
        assert "loop_load" not in code
    bound.set_config(config)
    torch.testing.assert_close(
        bound(*args), values[begin:7:step].sum(0), atol=0, rtol=0
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shared_initial(values, initial, steps):
    steps = hl.specialize(steps)
    out = torch.empty_like(initial)
    for fi in hl.tile(initial.numel()):
        bias = initial[fi]
        first = bias
        second = bias
        coefficient = bias + 2
        for qi in hl.grid(steps):
            first, second = second + values[qi, fi], first + coefficient
        out[fi] = first + 2 * second + bias + coefficient
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", [None, "current", "prefetch4"])
@pytest.mark.parametrize("steps", [0, 5])
@pytest.mark.parametrize("lane_layout", ["blocked", "strided"])
def test_ssa_fragments_shared_initial_captures_and_transport_width(
    width, mode, steps, lane_layout
):
    values = torch.randn(max(steps, 1), 67, device=DEVICE)
    initial = torch.randn(67, device=DEVICE)
    before = values.clone(), initial.clone()
    args = values, initial, steps
    config = _config(width, mode)
    # The thread owns eight values regardless of the transfer width.
    config.config.update(
        num_threads=[8],
        cute_vector_widths=[width, 1],
        cute_lane_layouts=[lane_layout, "blocked"],
    )
    bound = _shared_initial._bind_isolated(args)
    code = bound.to_code(config)
    assert "cute.make_rmem_tensor((8,), cutlass.Float32)" in code
    assert "tile_next_value = tile_next_buffer.load()" in code
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            assert not (
                isinstance(node.value, ast.Name)
                and node.value.id.startswith("tile_loop_value")
            )
    bound.set_config(config)
    first, second = initial, initial
    coefficient = initial + 2
    for qi in range(steps):
        first, second = second + values[qi], first + coefficient
    expected = first + 2 * second + initial + coefficient
    actual = bound(*args)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bound(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close((values, initial), before, atol=0, rtol=0)
