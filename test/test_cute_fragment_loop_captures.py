from __future__ import annotations

import ast
import builtins
import inspect
import operator
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.fragment_context import Fragment
from helion._compiler.cute.fragment_context import FragmentContext
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_fragment_loop_captures import loop_scans
from helion._testing.cute_fragment_loop_captures import loop_tensor_bound
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
import helion.language as hl
from helion.runtime.ref_mode import RefMode

pytestmark = pytest.mark.cpu_only


def test_fragment_root_coordinate_is_wide_before_the_stride_uses_it():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(2**31 - 1, 2**31 + 1):
            for _i in hl.grid(1, 2, row + 1):
                out[row % 2, :] = out[row % 2, :] + torch.cumsum(
                    x[row % 2, :] + 1, dim=0
                )

    x = torch.tensor([[2, 4], [2, 4]], dtype=torch.int64)
    protected = torch.tensor([[-73, 0, 0, 97], [-73, 0, 0, 97]], dtype=x.dtype)
    out = protected[:, 1:-1]
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, out)
    expected = protected.clone()
    assert expected.tolist() == [[-73, 3, 8, 97], [-73, 3, 8, 97]]
    out.zero_()
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(
        source, {"x": x}, {"out": out}, 2, int32=np.int32, int64=np.int64
    )
    torch.testing.assert_close(protected, expected)


@pytest.mark.parametrize(
    "lo,hi,step,blocks", [(0, 2, -2, 0), (2, 0, 1, 0), (3, 0, -2, 4)]
)
def test_flat_grid_multiplies_nonnegative_axis_counts(tmp_path, lo, hi, step, blocks):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, start: hl.constexpr, stop: hl.constexpr, stride: hl.constexpr):
        for i, j in hl.grid([start, start], [stop, stop], [stride, stride]):
            out[i + j, :] = torch.cumsum(x[i + j, :] + 1, dim=0)

    x = torch.ones(7, 2, dtype=torch.int64)
    args = (x, torch.zeros_like(x), lo, hi, step)
    with cpu_render_target():
        source = cpu_bind(scan, args).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "flat_grid.py"
    path.write_text(source)
    grids = []
    load_generated_module(path).scan(
        *args, _launcher=lambda fn, grid, *ops, block: grids.append(grid)
    )
    assert grids == [(blocks,)]


def test_flat_grid_rechecks_each_runtime_axis_on_warm_calls(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, out, stop):
        for i, j in hl.grid([3, 3], [stop, stop]):
            out[(i + j) % 2, :] = torch.cumsum(x[(i + j) % 2, :] + 1, dim=0)

    x = torch.ones(2, 2, dtype=torch.int64)
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, 5)).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "warm_grid.py"
    path.write_text(source)
    wrapper = load_generated_module(path).scan
    grids = []
    for stop in (5, 3, 2, -5):
        wrapper(
            x, out, stop, _launcher=lambda fn, grid, *ops, block: grids.append(grid)
        )
    assert grids == [(4,), (0,), (0,), (0,)]


@pytest.mark.parametrize("start, counter", [(7, "Int32"), (2**31 - 1, "Int64")])
def test_descending_grid_includes_the_sdk_final_increment(start, counter):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, begin: hl.constexpr):
        for row in hl.grid(1):
            for i in hl.grid(begin, begin - 3, -2):
                out[row, :] = torch.cumsum(x[row, :] + i, dim=0)

    x = torch.zeros((1, 2), dtype=torch.int64)
    out = torch.zeros_like(x)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, out, start)
    assert out.tolist() == [[start - 2, 2 * (start - 2)]]
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, start)).to_code(helion.Config(block_sizes=[]))
    (header,) = (
        n.iter
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("fragment_tile")
    )
    # The SDK counts forward from begin - 3. After its second iteration it
    # reaches begin + 1, even though the Python body never sees that value.
    assert isinstance(header, ast.Call)
    assert all(
        isinstance(bound, ast.Call) and ast.unparse(bound.func) == f"cutlass.{counter}"
        for bound in header.args[:2]
    )


@pytest.mark.parametrize(
    "escape,readonly_alias", [(False, False), (True, False), (False, True)]
)
def test_actual_loop_copies_retire_only_consumed_initials(escape, readonly_alias):
    observations = []
    original = FragmentContext.retired_loop_initials

    def inspect(context, node, values, copied, readonly):
        # Use the real context manager and inspect both sides of its boundary.
        class Scope:
            def __enter__(self):
                self.original_values = dict(values)
                self.before = context.live_buffers()
                self.manager = original(context, node, values, copied, readonly)
                self.manager.__enter__()
                observations.append(
                    (
                        self.before,
                        context.live_buffers(),
                        set(self.original_values) - set(values),
                    )
                )

            def __exit__(self, *args):
                result = self.manager.__exit__(*args)
                assert values == self.original_values
                return result

        return Scope()

    with (
        cpu_render_target(),
        patch.object(FragmentContext, "retired_loop_initials", inspect),
    ):
        bound = cpu_bind(loop_scans, (torch.ones(2, 17), 3, escape, readonly_alias))
        source = bound.to_code(bound.config_spec.default_config())
    assert len(observations) == 1
    before, after, removed = observations[0]
    if escape:
        # A dead `state` binding can retire while the escaping `initial`
        # binding keeps the same backing storage live throughout the loop.
        assert before == after
    elif readonly_alias:
        assert removed and before == after
    else:
        assert removed and len(before - after) == 1
    parsed = ast.parse(source)
    # Runtime loop and both result stores remain present. The copy preceding
    # the loop covers steps=0; the loop still handles arbitrary positive steps.
    assert any(
        isinstance(node, ast.For) and "steps" in ast.unparse(node.iter)
        for node in ast.walk(parsed)
    )
    assert ".store(" in source


def _scope_fixture(later_use=False, readonly_alias=False):
    graph = torch.fx.Graph()
    source = graph.placeholder("initial")
    call = graph.call_function(operator.neg, (source,))
    if later_use:
        graph.call_function(operator.add, (source, 1))
    value = Fragment((4,), torch.float32, lambda _: "initial[0]", True, (), "initial")
    alias = Fragment((4,), torch.float32, lambda _: "initial[0] + 7", True, (value,))
    context = object.__new__(FragmentContext)
    values = {source: value}
    context.scopes = [values]
    context.scope_uses = [{source: 2 if later_use else 1}]
    context.held = [
        Fragment((4,), torch.float32, lambda _: "carry[0]", True, (), "carry")
    ]
    return context, call, values, source, [alias] if readonly_alias else []


@pytest.mark.parametrize(
    "later_use,readonly_alias", [(False, False), (True, False), (False, True)]
)
def test_initial_scope_restores_after_zero_trip_or_lowering_failure(
    later_use, readonly_alias
):
    context, call, values, source, readonly = _scope_fixture(later_use, readonly_alias)
    initial = dict(values)
    for fail in (False, True):
        try:
            with context.retired_loop_initials(call, values, {source}, readonly):
                assert (source in values) == later_use
                assert ("initial" in context.live_buffers()) == (
                    later_use or readonly_alias
                )
                assert "carry" in context.live_buffers()
                if fail:
                    raise ValueError("lowering failure")
        except ValueError:
            assert fail
        assert values == initial and len(context.held) == 1


def test_loop_retirement_requires_its_active_parent_scope():
    context, call, values, source, readonly = _scope_fixture()
    context.scope_uses = []
    with (
        pytest.raises(AssertionError),
        context.retired_loop_initials(call, values, {source}, readonly),
    ):
        pytest.fail("an inconsistent internal scope must not be lowered")


@pytest.mark.parametrize("bound_use", ["range", "carry", "readonly"])
def test_only_independently_read_bounds_remain_live_in_loop_body(bound_use):
    observations = []
    original = FragmentContext.retired_loop_initials

    def inspect(context, node, values, copied, readonly):
        class Scope:
            def __enter__(self):
                ranges = [
                    value
                    for args in node.args[1:3]
                    for value in args
                    if isinstance(value, torch.fx.Node)
                ]
                self.range_buffers = context.referenced_buffers(
                    [values[value] for value in ranges]
                )
                self.manager = original(context, node, values, copied, readonly)
                self.manager.__enter__()
                observations.append(
                    (
                        self.range_buffers,
                        context.live_buffers(),
                        context.referenced_buffers(readonly),
                    )
                )

            def __exit__(self, *args):
                return self.manager.__exit__(*args)

        return Scope()

    with (
        cpu_render_target(),
        patch.object(FragmentContext, "retired_loop_initials", inspect),
    ):
        bound = cpu_bind(
            loop_tensor_bound,
            (
                torch.ones(1, 17),
                torch.tensor([2, 3, 0, 0], dtype=torch.int64),
                bound_use,
            ),
        )
        source = bound.to_code(bound.config_spec.default_config())
    assert len(observations) == 1
    range_buffers, live_buffers, readonly_buffers = observations[0]
    assert range_buffers, "the loop bound must actually use resident storage"
    if bound_use == "readonly":
        assert range_buffers <= readonly_buffers <= live_buffers
    else:
        assert range_buffers.isdisjoint(live_buffers)
    # This zero-based range counts whole steps at full width. Its stop
    # must still come from the snapshotted register.
    tree = ast.parse(source)
    (stop,) = (
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id.startswith("fragment_stop")
    )
    assert any(
        isinstance(n, ast.Name) and n.id.startswith("fragment_control")
        for n in ast.walk(stop.value)
    )
    assert any(
        isinstance(n, ast.For) and ast.unparse(n.target).startswith("fragment_ordinal")
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_grid_induction_product_uses_kernel_index_type(index_dtype):
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=index_dtype)
    def scan(x, out, scale: hl.constexpr):
        for row in hl.grid(x.size(0)):
            for i in hl.grid(2, 4):
                out[row, i, :] = torch.cumsum(x[row, :] + i * scale, dim=0)

    scale = 2**30 if index_dtype is torch.int64 else 7
    x = torch.arange(4, dtype=torch.int64).unsqueeze(0)
    out = torch.full((1, 4, 4), -1, dtype=torch.int64)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, out, scale)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, scale)).to_code(helion.Config(block_sizes=[]))
    (loop,) = (
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.For)
        and ast.unparse(n.iter) == "range(cutlass.Int32(2), cutlass.Int32(4), 1)"
    )
    # Replay the emitted expression with the actual Int32 range counters.
    (product,) = (
        n
        for n in ast.walk(loop)
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.Mult)
        and any(
            isinstance(arg, ast.Constant) and arg.value == scale
            for arg in (n.left, n.right)
        )
    )
    counters = torch.arange(2, 4, dtype=torch.int32)
    values = eval(
        compile(ast.Expression(product), "<fragment-grid-product>", "eval"),
        {
            "cutlass": SimpleNamespace(Int64=torch.Tensor.long),
            ast.unparse(loop.target): counters,
        },
    )
    assert values.dtype == index_dtype
    for i, value in zip((2, 3), values, strict=True):
        assert value.item() == i * scale
        torch.testing.assert_close(out[0, i, :], (x[0, :] + value).cumsum(0))


@pytest.mark.parametrize(
    "dtype,step,count_only,fixed_start",
    [
        (torch.int64, -2, True, False),
        (torch.int64, -2, False, False),
        (torch.int64, -3, False, False),
        (torch.int64, -(1 << 63), False, False),
        (torch.int32, -2, False, False),
        (torch.int64, -2, False, True),
        (torch.int64, -(1 << 63), False, True),
    ],
)
def test_runtime_descending_fragment_terminates_at_signed_endpoints(
    dtype, step, count_only, fixed_start
):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(
        x,
        bounds,
        out,
        step: hl.constexpr,
        count_only: hl.constexpr,
        fixed_start: hl.constexpr,
    ):
        for row in hl.grid(1):
            start = -3 if fixed_start else bounds[row, 0]
            for i in hl.grid(start, bounds[row, 1], step):
                if count_only:
                    # Eager and the formerly narrowed counter both ran twice.
                    # The computed input selects the real fragment scan path.
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
                else:
                    out[row, :] = torch.cumsum(x[row, :] + (i % 7), dim=0)

    limit = torch.iinfo(dtype)
    pairs = [
        (limit.max, limit.max - 3),
        (limit.max, limit.max - 1),
        (limit.min + 3, limit.min),
        (4, -3),
        (1, 1),
        (-2, 5),
    ]
    if step == -(1 << 63):
        pairs.append((limit.max, limit.min))
    if fixed_start:
        pairs = [(-3, -9), (-3, -3), (-3, 1)]
        if step == -(1 << 63):
            pairs.append((-3, limit.min))
    x = torch.tensor([[2, 4, 9, 19]], dtype=torch.int64)
    b = torch.tensor([pairs[0]], dtype=dtype)
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(
            scan, (x, b, out, step, count_only, fixed_start), capability=(10, 3)
        ).to_code(helion.Config(block_sizes=[]))
    # Python range cannot reproduce the SDK's overflowing reflected backedge.
    # Check the actual loop form in addition to replaying its values.
    assert "fragment_tile" in source
    assert any(isinstance(n, ast.While) for n in ast.walk(ast.parse(source))) is (
        dtype == torch.int64 and (not fixed_start or step == -(1 << 63))
    )
    for first, end in pairs:
        b[0] = torch.tensor([first, end], dtype=dtype)
        expected = torch.zeros_like(x)
        for i in range(first, end, step):
            if count_only:
                expected += torch.cumsum(x + 1, dim=1)
            else:
                expected = torch.cumsum(x + i % 7, dim=1)
        out.zero_()
        with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
            scan(x, b, out, step, count_only, fixed_start)
        torch.testing.assert_close(out, expected)
        out.zero_()
        simulate_independent_fragment(source, {"x": x, "bounds": b}, {"out": out}, 1)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("aliases_output", [False, True])
def test_varlen_bounds_keep_uniform_immutable_loads_in_registers(aliases_output):
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int32)
    def scan(x, limits, out):
        for row in hl.grid(1):
            for _i in hl.grid(limits[row], limits[row + 1], 2):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    backing = torch.zeros(6, dtype=torch.int64)
    limits = backing[:2] if aliases_output else torch.zeros(2, dtype=torch.int64)
    out = backing[2:].view(1, 4)
    x = torch.tensor([[3, 1, 4, 2]])
    with cpu_render_target():
        src = cpu_bind(scan, (x, limits, out)).to_code(helion.Config(block_sizes=[]))
    kernel = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
    loop = next(n for n in kernel.body if isinstance(n, ast.While))
    # Only a potentially mutable table needs a single publishing reader.
    assert sum(
        isinstance(n, ast.For) and ".load()" in ast.unparse(n)
        for n in kernel.body[: kernel.body.index(loop)]
    ) == (2 if aliases_output else 0)
    for begin, end in [
        (3, 8),
        (2**63 - 4, 2**63 - 1),
        (-(2**63), -(2**63) + 1),
        (8, 0),
    ]:
        limits[:] = torch.tensor([begin, end])
        out.zero_()
        with np.errstate(over="ignore"):  # The dead final update may wrap.
            simulate_independent_fragment(
                src,
                {"x": x, "limits": limits},
                {"out": out},
                1,
                int32=np.int32,
                int64=np.int64,
            )
        torch.testing.assert_close(out, len(range(begin, end, 2)) * (x + 1).cumsum(1))


def test_zero_based_fragment_counts_an_empty_tensors_wide_axis(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False, index_dtype=torch.int32)
    def scan(x, empty, out, step: hl.constexpr):
        for row in hl.grid(1):
            for _i in hl.grid(0, empty.size(1), step):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    # numel=0 permits an Int32 kernel even though this particular dimension
    # exceeds Int32. Use a wide step to exercise the real loop in two trips.
    x = torch.tensor([[3, 1, 4, 2]])
    out = torch.zeros_like(x)
    step = 2**62
    empty = x.as_strided((0, 2**63 - 1), (0, 0))
    with cpu_render_target():
        src = cpu_bind(scan, (x, empty, out, step)).to_code(
            helion.Config(block_sizes=[])
        )
    generated = tmp_path / "zero_size.py"
    generated.write_text(src)
    assert not any(isinstance(n, ast.While) for n in ast.walk(ast.parse(src)))

    def launch(fn, grid, *args, block):
        operands = dict(zip(inspect.signature(fn).parameters, args, strict=True))
        assert operands.pop("out") is out
        simulate_independent_fragment(
            src, operands, {"out": out}, int(grid[0]), int32=np.int32, int64=np.int64
        )

    wrapper = load_generated_module(generated).scan
    for size in (2**63 - 1, 7, 0):
        out.zero_()
        wrapper(x, x.as_strided((0, size), (0, 0)), out, step, _launcher=launch)
        torch.testing.assert_close(out, len(range(0, size, step)) * (x + 1).cumsum(1))


@pytest.mark.parametrize(
    "start,end,step,runtime_dtype,guarded",
    [
        (2**63 - 4, 2**63 - 1, 2, torch.int64, True),
        (2**63 - 1, 2**63 - 4, -2, None, True),
        (2**63 - 4, 2**63 - 1, 2, None, True),
        (2**63 - 5, 2**63 - 1, 2, None, False),
        (2**31 - 4, 2**31 - 1, 2, torch.int32, False),
    ],
)
def test_fragment_scan_counts_signed_endpoint_steps(
    start, end, step, runtime_dtype, guarded
):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(
        x, bounds, out, begin: hl.constexpr, finish: hl.constexpr, step: hl.constexpr
    ):
        for row in hl.grid(1):
            lo = bounds[row, 0] if begin is None else begin
            hi = bounds[row, 1] if finish is None else finish
            for _i in hl.grid(lo, hi, step):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4, 9, 19]], dtype=torch.int64)
    bounds = torch.tensor([[start, end]], dtype=runtime_dtype or torch.int64)
    out = torch.zeros_like(x)
    args = (
        x,
        bounds,
        out,
        None if runtime_dtype else start,
        None if runtime_dtype else end,
        step,
    )
    with cpu_render_target():
        source = cpu_bind(scan, args, capability=(10, 3)).to_code(
            helion.Config(block_sizes=[])
        )
    # Inspect the real control form first. Never run an unguarded overflow witness.
    assert any(isinstance(n, ast.While) for n in ast.walk(ast.parse(source))) == guarded
    epochs = [(start, end)]
    if runtime_dtype:
        epochs += [(end, end), (3, 8), (-5, -4)]
    for first, last in epochs:
        bounds[0] = torch.tensor([first, last], dtype=bounds.dtype)
        expected = len(range(first, last, step)) * (x + 1).cumsum(1)
        out.zero_()
        with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
            scan(*args)
        torch.testing.assert_close(out, expected)
        out.zero_()
        simulate_independent_fragment(
            source, {"x": x, "bounds": bounds}, {"out": out}, 1
        )
        torch.testing.assert_close(out, expected)


def test_fragment_loop_live_host_step_reuses_wrapper_and_rejects_zero(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def ValueError(x, bounds, out, OverflowError):  # noqa: A001, A002 - wrapper and argument shadow builtins
        for row in hl.grid(1):
            for _i in hl.grid(bounds[row, 0], bounds[row, 1], OverflowError):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4, 9, 19]], dtype=torch.int64)
    bounds = torch.tensor([[0, 1]], dtype=torch.int64)
    out = torch.zeros_like(x)
    with cpu_render_target():
        bound = cpu_bind(ValueError, (x, bounds, out, 2), capability=(10, 3))
        src = bound.to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "live_fragment_step.py"
    path.write_text(src)
    wrapper = load_generated_module(path).ValueError

    def replay(kernel, grid, *args, block):
        operands = dict(zip(inspect.signature(kernel).parameters, args, strict=True))
        assert operands.pop("out") is out
        simulate_independent_fragment(src, operands, {"out": out}, 1)

    for first, last, step in (
        (2**63 - 4, 2**63 - 1, 2),
        (2**63 - 1, 2**63 - 4, -2),
        (-(2**63), -(2**63) + 1, 3),
        (2**63 - 1, -(2**63), -(2**63)),
        (5, 2, 2),
        (2, 5, -2),
    ):
        bounds[0] = torch.tensor([first, last])
        expected = len(range(first, last, step)) * (x + 1).cumsum(1)
        out.zero_()
        with patch.object(ValueError.settings, "ref_mode", RefMode.EAGER):
            ValueError(x, bounds, out, step)
        torch.testing.assert_close(out, expected)
        out.zero_()
        wrapper(x, bounds, out, step, _launcher=replay)
        torch.testing.assert_close(out, expected)
    with pytest.raises(builtins.ValueError, match="must not be zero"):
        wrapper(x, bounds, out, 0, _launcher=lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(OverflowError, match="signed Int64"):
        wrapper(
            x, bounds, out, 2**63, _launcher=lambda *a, **kw: pytest.fail("launched")
        )


def test_compound_fragment_step_is_evaluated_on_the_host(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, out, n, fragment_host_step):
        fragment_host_step_1 = fragment_host_step + 1
        for row in hl.grid(1):
            for i in hl.grid(1, 2, (n + 2**63) // 3):
                for j in hl.grid(i, i + 1):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + j, dim=0)

        return fragment_host_step, fragment_host_step_1

    x = torch.tensor([[2, 4]])
    out = torch.zeros_like(x)
    n = 3  # The intermediate exceeds Int64, but the Python step fits.
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, n, 17)).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "compound.py"
    path.write_text(source)
    wrapper = load_generated_module(path).scan

    def replay(kernel, grid, *args, block):
        operands = dict(zip(inspect.signature(kernel).parameters, args, strict=True))
        assert operands.pop("out") is out
        assert "n" not in operands
        assert (n + 2**63) // 3 in [v for v in operands.values() if type(v) is int]
        simulate_independent_fragment(source, operands, {"out": out}, 1, int64=np.int64)

    assert wrapper(x, out, n, 17, _launcher=replay) == (17, 18)
    assert out.tolist() == [[3, 8]]


def test_fragment_grid_evaluates_all_steps_on_entry_to_the_device_branch(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, flags, out, stop, step):
        for row in hl.grid(1):
            if flags[row]:
                for _i, _j in hl.grid([0, 1], [stop, 4], [1, step]):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    flags = torch.tensor([False])
    out = torch.tensor([[-11, 23]])
    with cpu_render_target():
        source = cpu_bind(scan, (x, flags, out, 3, 2)).to_code(
            helion.Config(block_sizes=[])
        )
    path = tmp_path / "conditional.py"
    path.write_text(source)
    wrapper = load_generated_module(path).scan

    def invalid_step(*args, **kwargs):
        assert "trap;" in kwargs["asm"]
        raise ValueError("reached invalid step")

    def replay(kernel, grid, *args, block):
        operands = dict(zip(inspect.signature(kernel).parameters, args, strict=True))
        assert operands.pop("out") is out
        simulate_independent_fragment(
            source, operands, {"out": out}, 1, inline_asm=invalid_step
        )

    for step in (0, 2**63):
        wrapper(x, flags, out, 0, step, _launcher=replay)
        assert out.tolist() == [[-11, 23]]
    flags.fill_(True)
    with pytest.raises(ValueError, match="reached invalid step"):
        wrapper(x, flags, out, 0, 0, _launcher=replay)
    wrapper(x, flags, out, 0, 2, _launcher=replay)
    assert out.tolist() == [[-11, 23]]
    wrapper(x, flags, out, 1, 2, _launcher=replay)
    assert out.tolist() == [[-5, 39]]


@pytest.mark.parametrize(
    "form", ["root", "nested", "tile", "divide", "product", "modulo"]
)
def test_fragment_scan_with_grid_derived_stride(form):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, form: hl.constexpr):
        for row in hl.grid(4):
            if form == "nested":
                for col in hl.grid(2):
                    for _i in hl.grid(1, 8, col + 2):
                        out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
            elif form == "tile":
                for t in hl.tile(2, block_size=1):
                    for _i in hl.grid(1, 8, t.begin + 1):
                        out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
            else:
                if form == "root":
                    step = row + 2
                elif form == "divide":
                    step = row // 2 + 1
                elif form == "modulo":
                    step = (row - 3) % 2 + 1
                else:
                    step = row * row + 1
                for _i in hl.grid(1, 8, step):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]]).repeat(4, 1)
    expected = torch.zeros_like(x)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected, form)
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, form)).to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(source, {"x": x}, {"out": out}, 4)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("start", [1, 2])
@pytest.mark.parametrize("operand", ["begin", "end", "step"])
def test_fragment_grid_checks_the_product_before_division(start, operand):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, start: hl.constexpr, operand: hl.constexpr):
        # Row start + 1 is outside the *visited* coordinates even though
        # it is below the stop. Only start == 2 overflows before division.
        for row in hl.grid(start, start + 2, 2):
            value = (row * 2**62) // 3
            if operand == "step":
                for _i in hl.grid(1, 2, value):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
            elif operand == "end":
                for _i in hl.grid(0, value, 2**62):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
            else:
                for _i in hl.grid(value, 2**62, 2**62):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 3)
    out = torch.zeros_like(x)
    expected = torch.zeros_like(x)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected, start, operand)
    assert expected[start].tolist() == [3, 8]
    with cpu_render_target():
        bound = cpu_bind(scan, (x, out, start, operand))
        if start == 2:
            with pytest.raises(exc.InvalidConfig, match="not proved to fit Int64"):
                bound.to_code(helion.Config(block_sizes=[]))
            return
        source = bound.to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(
        source, {"x": x}, {"out": out}, 1, int32=np.int32, int64=np.int64
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "listed, start, stop, step",
    [
        (True, 1, 3, 2),
        (True, -1, -3, -2),
        (False, -1, -3, -2),
        (False, 1, 2, 1),
        (True, 2, 4, 2),
        (False, -3, -5, -2),
    ],
)
def test_fragment_range_uses_logical_root_step(listed, start, stop, step):
    @helion.kernel(backend="cute", static_shapes=True)
    def scalar_grid(
        x, out, start: hl.constexpr, stop: hl.constexpr, step: hl.constexpr
    ):
        end = -1 if start < 0 else 1
        for row in hl.grid(start, stop, step):
            for _i in hl.grid(0, end, (row * 2**62) // 3):
                out[0, :] = torch.cumsum(x[0, :] + 1, dim=0)

    @helion.kernel(backend="cute", static_shapes=True)
    def listed_grid(
        x, out, start: hl.constexpr, stop: hl.constexpr, step: hl.constexpr
    ):
        end = -1 if start < 0 else 1
        for (row,) in hl.grid([start], [stop], [step]):
            for _i in hl.grid(0, end, (row * 2**62) // 3):
                out[0, :] = torch.cumsum(x[0, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.full_like(x, -73)
    with cpu_render_target():
        bound = cpu_bind(
            listed_grid if listed else scalar_grid, (x, out, start, stop, step)
        )
        if not -(2**63) <= start * 2**62 < 2**63:
            with pytest.raises(exc.InvalidConfig, match="not proved to fit Int64"):
                bound.to_code(helion.Config(block_sizes=[]))
            return
        source = bound.to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            source, {"x": x}, {"out": out}, 1, int32=np.int32, int64=np.int64
        )
    assert out.tolist() == [[3, 8]]


@pytest.mark.parametrize("mode", ["bitwise", "nested"])
def test_fragment_range_checks_bounded_nonliteral_intermediates(mode):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, mode: hl.constexpr):
        for row in hl.grid(1, 3):
            if mode == "nested":
                for j in hl.grid(row, row + 1):
                    for _i in hl.grid(0, 1, (j * 2**62) // 3):
                        out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)
            else:
                for _i in hl.grid(0, 1, ((row | 1) * 2**62) // 3):
                    out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.ones(3, 2, dtype=torch.int64)
    with (
        cpu_render_target(),
        pytest.raises(exc.InvalidConfig, match="not proved to fit Int64"),
    ):
        cpu_bind(scan, (x, torch.zeros_like(x), mode)).to_code(
            helion.Config(block_sizes=[])
        )


def test_nested_remainder_respects_the_proven_residue():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(1):
            for y in hl.grid(0, 6, 4):
                for j in hl.grid(-y, 1 - y):
                    divisor = 9 + y + 2 * j
                    for k in hl.grid(24 % divisor):
                        for i in hl.grid(k, k + 1):
                            out[row, :] = torch.cumsum(x[row, :] + i, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.zeros_like(x)
    with cpu_render_target():
        src = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            src, {"x": x}, {"out": out}, 1, int32=np.int32, int64=np.int64
        )
    # The last divisor is 5 and the last k/i is 3.
    assert out.tolist() == [[5, 12]]


def test_nested_guarded_endpoint_retains_the_unchecked_divisor_range():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(-1, 2, 2):
            for j in hl.grid(1, 2 + 1 // row):
                for _i in hl.grid(0, 1, (j * 2**62) // 3):
                    out[0, :] = torch.cumsum(x[0, :] + 1, dim=0)

    x = torch.zeros(1, 2)
    with (
        cpu_render_target(),
        pytest.raises(exc.InvalidConfig, match="not proved to fit Int64"),
    ):
        cpu_bind(scan, (x, torch.zeros_like(x))).to_code(helion.Config(block_sizes=[]))


@pytest.mark.parametrize(
    "mode", ["dependent", "varying_stride", "xor", "signed_xor", "alias_divisor"]
)
def test_nested_range_checks_only_reachable_integer_values(mode):
    @helion.kernel(backend="cute", static_shapes=True)
    def dependent(x, out):
        for row in hl.grid(3):
            for j in hl.grid(row, row + 1):
                for _i in hl.grid(0, 1, (2**63 - 1) + ((row - j) // 2)):
                    out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    @helion.kernel(backend="cute", static_shapes=True)
    def varying_stride(x, out):
        for row in hl.grid(2, 4):
            for j in hl.grid(1, 3, row):
                for _i in hl.grid(0, 1, (j * 2**62) // 3):
                    out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    @helion.kernel(backend="cute", static_shapes=True)
    def xor(x, out):
        for row in hl.grid(33):
            for _i in hl.grid(0, 1, (row ^ 1) + 1):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    @helion.kernel(backend="cute", static_shapes=True)
    def signed_xor(x, out):
        for row in hl.grid(2):
            for _i in hl.grid(0, 1, (2**63 - 1) + (row ^ -2)):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    @helion.kernel(backend="cute", static_shapes=True)
    def alias_divisor(x, out):
        for row in hl.grid(-2, 3, 4):
            for j in hl.grid(row, row + 1):
                for _i in hl.grid(0, 1, -(2**63) // j):
                    out[row + 2, :] = torch.cumsum(x[row + 2, :] + 1, dim=0)

    scan, ctas = {
        "dependent": (dependent, 3),
        "varying_stride": (varying_stride, 2),
        "signed_xor": (signed_xor, 2),
        "alias_divisor": (alias_divisor, 2),
        "xor": (xor, 33),
    }[mode]
    x = torch.arange(66).reshape(33, 2)
    out = torch.full_like(x, -37)
    expected = out.clone()
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected)
    with cpu_render_target():
        src = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            src, {"x": x}, {"out": out}, ctas, int32=np.int32, int64=np.int64
        )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("mode", ["fixed_tile", "square"])
def test_fragment_range_uses_reachable_coordinates_in_safe_products(mode):
    @helion.kernel(backend="cute", static_shapes=True)
    def fixed_tile(x, out):
        for t in hl.tile(1, 3, block_size=2):
            for _i in hl.grid(0, 1, (t.begin * 2**62) // 3):
                out[t, :] = torch.cumsum(x[t, :] + 1, dim=1)

    @helion.kernel(backend="cute", static_shapes=True)
    def square(x, out):
        for row in hl.grid(-35, 36, 2):
            for _i in hl.grid(0, 1, (2**63 - 1) - row * row):
                out[(row + 35) // 2, :] = torch.cumsum(x[(row + 35) // 2, :] + 1, dim=0)

    scan = fixed_tile if mode == "fixed_tile" else square
    x = (
        torch.arange(72).reshape(36, 2)
        if mode == "square"
        else torch.tensor([[2, 4], [3, 7], [5, 9]])
    )
    out = torch.full_like(x, -37)
    expected = out.clone()
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected)
    with cpu_render_target():
        src = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            src,
            {"x": x},
            {"out": out},
            1 if mode == "fixed_tile" else 36,
            int32=np.int32,
            int64=np.int64,
        )
    torch.testing.assert_close(out, expected)


def test_nested_descending_stride_proves_only_visited_coordinates():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(1):
            for col in hl.grid(-1, -3, -2):
                for _i in hl.grid(0, (-col * 2**62) // 3, 2**62):
                    out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(
        source, {"x": x}, {"out": out}, 1, int32=np.int32, int64=np.int64
    )
    assert out.tolist() == [[3, 8]]


@pytest.mark.parametrize("divide,stop", [(True, 2), (False, 2), (True, 1)])
def test_fragment_grid_refuses_a_provably_zero_divisor(divide, stop, tmp_path):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, divide: hl.constexpr, stop: hl.constexpr):
        for row in hl.grid(1, stop):
            divisor = row - 1
            step = 1 + (4 // divisor if divide else 4 % divisor)
            for _i in hl.grid(0, 1, step):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.ones(2, 2, dtype=torch.int64)
    args = x, torch.empty_like(x), divide, stop
    with cpu_render_target():
        bound = cpu_bind(scan, args)
        config = helion.Config(block_sizes=[])
        if stop == 2:
            with pytest.raises(exc.InvalidConfig, match="stride has a zero divisor"):
                bound.to_code(config)
            return
        source = bound.to_code(config)
    path = tmp_path / "empty.py"
    path.write_text(source)
    grids = []
    load_generated_module(path).scan(
        *args, _launcher=lambda fn, grid, *ops, block: grids.append(grid)
    )
    assert grids == [(0,)]


@pytest.mark.parametrize("divide", [True, False])
def test_fragment_range_guards_a_divisor_before_using_it(divide):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, divide: hl.constexpr):
        for row in hl.grid(1, 3):
            divisor = row - 1
            step = 1 + (4 // divisor if divide else 4 % divisor)
            for _i in hl.grid(0, 1, step):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.ones(3, 2, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(scan, (x, torch.zeros_like(x), divide)).to_code(
            helion.Config(block_sizes=[])
        )
    body = ast.parse(source)
    guards = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If) and "fragment_divisor" in ast.unparse(node.test)
    ]
    (guard,) = guards
    assert "trap;" in ast.unparse(guard)
    assert "Int64(1)" in ast.unparse(guard)
    operands = [
        node
        for node in ast.walk(body)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.FloorDiv, ast.Mod))
            and "fragment_divisor" in ast.unparse(node.right)
        )
        or (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_cute_python_mod"
            and "fragment_divisor" in ast.unparse(node.args[1])
        )
    ]
    assert operands and all(n.lineno > guard.end_lineno for n in operands)


def test_fragment_range_refuses_signed_division_overflow_across_zero():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(0, 3, 2):
            divisor = row - 1
            step = (-(2**63) // divisor) + divisor
            for _i in hl.grid(0, 1, step):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.ones(3, 2, dtype=torch.int64)
    with cpu_render_target():
        bound = cpu_bind(scan, (x, torch.zeros_like(x)))
        with pytest.raises(
            exc.InvalidConfig, match="stride is not proved to fit Int64"
        ):
            bound.to_code(helion.Config(block_sizes=[]))


@pytest.mark.parametrize("affine", [False, True])
def test_fragment_range_proves_quotients_at_reached_strided_coordinates(affine):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, affine: hl.constexpr):
        start, stop, stride = (1, -1, -1) if affine else (-2, 3, 4)
        for row in hl.grid(start, stop, stride):
            divisor = 4 * row - 2 if affine else row
            step = -(2**63) // divisor
            for _i in hl.grid(0, step, step):
                out[divisor + 2, :] = torch.cumsum(x[divisor + 2, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 5)
    out = torch.full_like(x, -73)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, affine)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            source, {"x": x}, {"out": out}, 2, int32=np.int32, int64=np.int64
        )
    assert out.tolist() == [[3, 8], [-73, -73], [-73, -73], [-73, -73], [3, 8]]


@pytest.mark.parametrize("upper", [True, False])
def test_guarded_remainder_cannot_hide_an_overflowing_sum(upper):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out, upper: hl.constexpr):
        for row in hl.grid(0, 4, 3):
            if upper:
                step = ((2**63 - 1) + (1 % (row - 1))) // 2
            else:
                # A factor of two could move the divisible constant outside
                # the sum. Three keeps the underflow in the emitted operand.
                step = (-(2**63) + (-1 % (row - 2))) // 3
            for _i in hl.grid(0, 1, step):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.ones(4, 2, dtype=torch.int64)
    with cpu_render_target():
        bound = cpu_bind(scan, (x, torch.zeros_like(x), upper))
        with pytest.raises(
            exc.InvalidConfig, match="stride is not proved to fit Int64"
        ):
            bound.to_code(helion.Config(block_sizes=[]))


def test_guarded_remainder_proves_an_exact_multiple_on_reached_rows():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(0, 4, 3):
            step = ((2**63 - 1) + (2 % (row - 1))) // 2
            for _i in hl.grid(0, 1, step):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 4)
    out = torch.full_like(x, -73)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            source, {"x": x}, {"out": out}, 2, int32=np.int32, int64=np.int64
        )
    assert out.tolist() == [[3, 8], [-73, -73], [-73, -73], [3, 8]]


def test_descending_root_row_controls_the_inner_trip_count():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(4, 0, -2):
            for _i in hl.grid(0, 8, row):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 5)
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    with np.errstate(over="raise"):
        simulate_independent_fragment(
            source, {"x": x}, {"out": out}, 2, int32=np.int32, int64=np.int64
        )
    assert out.tolist() == [[0, 0], [0, 0], [12, 32], [0, 0], [6, 16]]


def test_fragment_range_preserves_reached_nonzero_divisors_on_both_sides():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(0, 3, 2):
            for _i in hl.grid(0, 1, 5 + 4 // (row - 1)):
                out[row, :] = torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 3)
    out = torch.full_like(x, -73)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(source, {"x": x}, {"out": out}, 2)
    assert out.tolist() == [[3, 8], [-73, -73], [3, 8]]


def test_empty_cartesian_axis_skips_an_unrelated_invalid_operand(tmp_path):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row, col in hl.grid([1, 0], [2, 0]):
            for _i in hl.grid(0, 1, 1 + 4 // (row - 1)):
                out[row, :] = torch.cumsum(x[row, :] + col, dim=0)

    x = torch.ones(2, 2, dtype=torch.int64)
    args = x, torch.full_like(x, -73)
    with cpu_render_target():
        source = cpu_bind(scan, args).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "empty_cartesian.py"
    path.write_text(source)
    grids = []
    load_generated_module(path).scan(
        *args, _launcher=lambda fn, grid, *ops, block: grids.append(grid)
    )
    assert grids == [(0,)]
    assert args[1].tolist() == [[-73, -73], [-73, -73]]


@pytest.mark.parametrize("form", ["host", "remainder", "nested", "tensor_root"])
def test_unknown_fragment_stride_domains_keep_wide_runtime_arithmetic(form, tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, out, bounds, limit, n, form: hl.constexpr):
        root_end = limit if form == "tensor_root" else 2
        for row in hl.grid(root_end):
            if form == "nested":
                for col in hl.grid(bounds[0], bounds[1]):
                    for _i in hl.grid(1, 8, col + 2):
                        out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)
            else:
                if form == "host":
                    stride = row + n
                elif form == "remainder":
                    stride = row + (n % 2) + 1
                else:
                    stride = row + 2
                for _i in hl.grid(1, 8, stride):
                    out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 2)
    bounds = torch.tensor([0, 2], dtype=torch.int64)
    limit = torch.tensor(2)
    step = 2 if form == "host" else 3
    out = torch.zeros_like(x)
    expected = torch.zeros_like(x)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, expected, bounds, limit, step, form)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, bounds, limit, step, form)).to_code(
            helion.Config(
                block_sizes=[],
                **({"pid_type": "persistent_blocked"} if form == "tensor_root" else {}),
            )
        )
    path = tmp_path / "unknown.py"
    path.write_text(source)

    def launch(fn, grid, *args, block):
        ops = dict(zip(inspect.signature(fn).parameters, args, strict=True))
        assert ops.pop("out") is out
        if form == "tensor_root":
            # A persistent grid reads the scalar root bound as limit[0].
            ops = {
                key: [int(limit)] if arg is limit else arg for key, arg in ops.items()
            }
        simulate_independent_fragment(
            source, ops, {"out": out}, int(grid[0]), int32=np.int32, int64=np.int64
        )

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).scan(
            x, out, bounds, limit, step, form, _launcher=launch
        )
    torch.testing.assert_close(out, expected)


def test_empty_descending_root_does_not_validate_an_unreached_step(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, out, step):
        for row in hl.grid(0, 1, -2):
            for _i in hl.grid(1, 4, step):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.full_like(x, -13)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, 2)).to_code(helion.Config(block_sizes=[]))
    f = tmp_path / "empty_root.py"
    f.write_text(source)
    grids = []

    def capture(kernel, grid, *args, block):
        grids.append(grid)

    load_generated_module(f).scan(x, out, 0, _launcher=capture)
    assert grids == [(0,)]
    assert out.tolist() == [[-13, -13]]


def test_tensor_bound_root_checks_step_only_for_live_blocks(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def scan(x, out, limit, step):
        for row in hl.grid(limit):
            for _i in hl.grid(1, 4, step):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.zeros_like(x)
    limit = torch.tensor(1)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out, limit, 2)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "persistent.py"
    path.write_text(source)
    wrapper = load_generated_module(path).scan

    def capture(kernel, grid, *args, block):
        operands = dict(zip(inspect.signature(kernel).parameters, args, strict=True))
        assert operands["fragment_host_step"] == 0

    limit.zero_()
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        wrapper(x, out, limit, 0, _launcher=capture)
    device = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    virtual_pid_loop = next(
        n
        for n in ast.walk(device)
        if isinstance(n, ast.For) and ast.unparse(n.target) == "virtual_pid"
    )
    assert "trap;" in ast.unparse(virtual_pid_loop)


def test_fragment_accepts_a_uint64_endpoint_widened_from_bytes():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, limits, out):
        for row in hl.grid(1):
            for _i in hl.grid(0, limits[row].to(torch.uint64)):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    limits = torch.tensor([3], dtype=torch.uint8)
    out = torch.zeros_like(x)
    with cpu_render_target():
        source = cpu_bind(scan, (x, limits, out)).to_code(helion.Config(block_sizes=[]))
    simulate_independent_fragment(source, {"x": x, "limits": limits}, {"out": out}, 1)
    assert out.tolist() == [[9, 24]]


def test_fragment_does_not_use_signed_order_for_uint64_bounds():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, bounds, out):
        for row in hl.grid(1):
            for _i in hl.grid(bounds[row, 0], bounds[row, 1], 2):
                out[row, :] = out[row, :] + torch.cumsum(x[row, :] + 1, dim=0)

    x = torch.tensor([[2, 4]])
    out = torch.zeros_like(x)
    bounds = torch.tensor([[2**63 - 1, 2**63 + 2]], dtype=torch.uint64)
    with patch.object(scan.settings, "ref_mode", RefMode.EAGER):
        scan(x, bounds, out)
    assert out.tolist() == [[6, 16]]
    with (
        cpu_render_target(),
        pytest.raises(exc.InvalidConfig, match="UInt64 fragment loop"),
    ):
        cpu_bind(scan, (x, bounds, out)).to_code(helion.Config(block_sizes=[]))
