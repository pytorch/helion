from __future__ import annotations

import ast
from functools import partial
import inspect
import operator
from types import CodeType
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion._testing import _get_backend
from helion._testing import cpu_render_target
from helion._testing import matchesBackends
from helion._testing import skipIfRefEagerMark
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
from helion._testing.cute_output_addresses import generated_output_address_coverage
from helion._testing.triton_codegen import interpret_triton_kernel
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipIfRefEagerMark("checks generated Cartesian grid code"),
]


def _persistent_count(source: str) -> CodeType:
    (count,) = [
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "total_pids"
    ]
    return compile(ast.Expression(count), "<persistent trips>", "eval")


@skipUnlessBackends(["triton", "cute"])
def test_python_stride_checks_its_surviving_device_argument(tmp_path):
    @helion.kernel(backend=_get_backend(), static_shapes=False)
    def write(stop, stride, out):
        for i in hl.grid(0, stop, stride):
            out[i % 8] = i

    out = torch.full((8,), -1)
    stop = torch.tensor(8, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(write, (stop, 2, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "python_step.py"
    path.write_text(source)
    run = load_generated_module(path).write
    calls = []
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        for step in (2**63 - 1, -(2**63)):
            run(stop, step, out, _launcher=lambda *a, **kw: calls.append(a))
        for step in (2**63, -(2**63) - 1):
            with pytest.raises(OverflowError, match="grid argument"):
                run(stop, step, out, _launcher=lambda *a, **kw: calls.append(a))
    assert len(calls) == 2
    assert all(-(2**63) <= v < 2**63 for call in calls for v in call if type(v) is int)


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("step", [2, -2])
def test_persistent_range_counts_every_strided_coordinate(step, tmp_path):
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=False)
    def write(out, limit, stride: hl.constexpr):
        for i in hl.grid(0, limit, stride):
            out[i + 5] = i + 7

    out = torch.full((11,), -73)
    limit = torch.tensor(5 if step > 0 else -5, dtype=torch.int32)
    args = (out, limit, step)
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "bounded.py"
    path.write_text(source)
    if matchesBackends(["triton"]):
        m = load_generated_module(path)

        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            m.write(*args, _launcher=interpret_triton_kernel)
        expected = torch.full_like(out, -73)
        for i in range(0, int(limit), step):
            expected[i + 5] = i + 7
        torch.testing.assert_close(out, expected)
    else:
        # Count in the generated device program, before its persistent loop.
        # In particular a false billion-trip grid must never be dispatched.
        count = _persistent_count(source)
        for end in (int(limit), 0, -(2**31), 2**31 - 1):
            scope = {
                "limit": [np.int32(end)],
                "cutlass": SimpleNamespace(Int64=np.int64),
            }
            got = eval(count, scope)
            assert got == len(range(0, end, step))


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "lo,hi,step_value,dtype",
    [
        (0, 8, 2, torch.int64),
        (7, -1, -2, torch.int64),
        (0, 300, 100, torch.int8),
        (0, 8, 2**63, torch.uint64),
    ],
)
def test_root_grid_scalar_tensor_step(tmp_path, lo, hi, step_value, dtype):
    @helion.kernel(backend=_get_backend(), static_shapes=True)
    def write(out, step, lo: hl.constexpr, hi: hl.constexpr):
        for i in hl.grid(lo, hi, step):
            out[i] = i

    out = torch.full((max(lo + 1, hi),), -73, dtype=torch.int64)
    step = torch.tensor(step_value, dtype=dtype)
    with cpu_render_target():
        bound = cpu_bind(write, (out, step, lo, hi))
        assert "flat" not in bound.config_spec.allowed_pid_types
        source = bound.to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    if matchesBackends(["triton"]):
        from triton.runtime.errors import InterpreterError

        path = tmp_path / "tensor_step.py"
        path.write_text(source)
        module = load_generated_module(path)
        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            module.write(out, step, lo, hi, _launcher=interpret_triton_kernel)
            # Replace the fatal device assertion with a catchable CPU signal.
            with (
                patch.object(
                    module.helion_triton_helpers,
                    "device_error",
                    side_effect=ValueError("zero device stride"),
                ) as fail,
                pytest.raises(InterpreterError, match="zero device stride"),
            ):
                module.write(
                    out,
                    torch.zeros_like(step),
                    lo,
                    hi,
                    _launcher=interpret_triton_kernel,
                )
            fail.assert_called_once_with("range() arg 3 must not be zero")
        coords = range(lo, hi, step_value)
        assert out.tolist() == [i if i in coords else -73 for i in range(out.numel())]
    else:
        count = _persistent_count(source)
        for stride in (-(2**63), -2, 1, 2) if dtype == torch.int64 else (step_value,):
            scope = {
                "step": [
                    np.uint64(stride).view(np.int64)
                    if dtype == torch.uint64
                    else np.int64(stride)
                ],
                "_BLOCK_SIZE_0": 1,
                "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
            }
            assert eval(count, scope) == len(range(lo, hi, stride))


@skipUnlessBackends(["triton"])
def test_signed_grid_coordinates_with_unsigned_operands(tmp_path):
    @helion.kernel(backend="triton")
    def visit(out, cols, step, lo, hi, u):
        for i in hl.grid(-3, 8, step):
            if i < 0:
                out[i + 3] = i + 17
            else:
                out[i + 3] = i
        for row, col in hl.grid([lo, -1], [hi, u]):
            cols[row, col + 1] = col + 5

    out = torch.full((11,), -73, dtype=torch.int64)
    cols = torch.zeros((1, 4), dtype=torch.int64)
    args = (
        out,
        cols,
        torch.tensor(2, dtype=torch.uint64),
        torch.tensor(0),
        torch.tensor(1),
        torch.tensor(3, dtype=torch.uint32),
    )
    with cpu_render_target():
        source = cpu_bind(visit, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "signed_axes.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).visit(*args, _launcher=interpret_triton_kernel)
    assert out.tolist() == [14, -73, 16, -73, 1, -73, 3, -73, 5, -73, 7]
    assert cols.tolist() == [[4, 5, 6, 7]]


@skipUnlessBackends(["cute"])
def test_grid_bool_stride_uses_the_logical_value():
    @helion.kernel(backend="cute")
    def write(out, step):
        for i in hl.grid(0, 4, step):
            out[i] = i

    # torch.bool permits any nonzero backing byte; True is a unit stride.
    step = torch.tensor(2, dtype=torch.uint8).view(torch.bool)
    with cpu_render_target():
        source = cpu_bind(write, (torch.empty(4), step)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    counts = generated_output_address_coverage(
        source,
        "out",
        1,
        1,
        scalar_args={"step": [2], "_NUM_SM": 1, "min": min, "max": max},
    )
    assert counts == dict.fromkeys(range(4), 1)


@skipUnlessBackends(["cute"])
def test_tensor_stride_counts_int32_min_without_launching_the_grid():
    @helion.kernel(backend="cute")
    def write(out, end, step):
        for i in hl.grid(0, end, step):
            out[0] = i

    end, step = (
        torch.tensor(-(2**31), dtype=torch.int32),
        torch.tensor(-1, dtype=torch.int32),
    )
    with cpu_render_target():
        src = cpu_bind(write, (torch.zeros(1), end, step)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    count = eval(
        _persistent_count(src),
        {
            "end": [np.int32(end.item())],
            "step": [np.int32(-1)],
            "_BLOCK_SIZE_0": 1,
            "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
        },
    )
    assert count == len(range(0, end.item(), step.item())) == 2**31


@skipUnlessBackends(["triton", "cute"])
def test_literal_zero_grid_stride():
    @helion.kernel(backend=_get_backend())
    def write(out):
        for i in hl.grid(0, 8, 0):
            out[i] = i

    with (
        cpu_render_target(),
        pytest.raises(helion.exc.InvalidAPIUsage, match="range.*must not be zero"),
    ):
        cpu_bind(write, (torch.empty(8),)).to_code(helion.Config(block_sizes=[]))


@skipUnlessBackends(["cute"])
def test_flattened_root_preserves_signed_coordinate_before_branch():
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for row in hl.grid(2**31 - 1, 2**31 + 1):
            if row > 0:
                out[row % 2, :] = torch.cumsum(x[row % 2, :] + 1, dim=0)

    x = torch.tensor([[2, 4]] * 2)
    with cpu_render_target():
        source = cpu_bind(scan, (x, torch.zeros_like(x))).to_code(
            helion.Config(block_sizes=[])
        )
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    branch = next(n for n in fn.body if isinstance(n, ast.If))
    index = ast.unparse(branch.test.left)
    # Replay the physical coordinate with the DSL's signed widths. A cast
    # after the shift would leave block 1 negative and skip its store.
    prefix = []
    for n in fn.body:
        prefix.append(n)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == index:
            break
    constants = [n for n in tree.body if isinstance(n, ast.Assign)]
    mod = compile(ast.Module([*constants, *prefix], []), "<coordinates>", "exec")
    seen = []
    for pid in range(2):
        scope = {
            "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(block_idx=lambda pid=pid: (pid, 0, 0))
            ),
        }
        exec(mod, scope)
        seen.append(int(scope[index]))
    assert seen == list(range(2**31 - 1, 2**31 + 1))


@skipUnlessBackends(["cute"])
def test_empty_grouped_grid_has_no_device_decode_division(tmp_path):
    @helion.kernel(backend="cute", static_shapes=True)
    def scan(x, out):
        for i, j in hl.grid([0, 0], [2, 2], [-2, -2]):
            out[i + j, :] = torch.cumsum(x[i + j, :] + 1, dim=0)

    x = torch.ones(4, 2, dtype=torch.int64)
    out = torch.full_like(x, -73)
    with cpu_render_target():
        source = cpu_bind(scan, (x, out)).to_code(
            helion.Config(block_sizes=[], pid_type="flat", l2_groupings=[2])
        )
    path = tmp_path / "empty_grouped.py"
    path.write_text(source)
    grids = []
    load_generated_module(path).scan(
        x, out, _launcher=lambda fn, grid, *ops, block: grids.append(grid)
    )
    assert grids == [(0,)]
    # Evaluate the decoder up to the two PID assignments. A static zero
    # divisor is a compilation error even when the host never dispatches it.
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    scope = {
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: (0, 0, 0))),
    }
    for n in fn.body:
        exec(compile(ast.Module([n], []), "<PID decode>", "exec"), scope)
        if "pid_0" in scope and "pid_1" in scope:
            break
    assert (scope["pid_0"], scope["pid_1"]) == (0, 0)


@helion.kernel(backend="cute", static_shapes=False)
def _persistent_tensor_grid(out, lo, hi, stride: hl.constexpr):
    for i in hl.grid(lo, hi, stride):
        out[i] = i


@helion.kernel(backend="cute", static_shapes=True)
def _crossing_tile_end(out):
    for tile in hl.tile(2**31 - 4, 2**31, block_size=4):
        out[0] = tile.end


@helion.kernel(backend="cute", static_shapes=True)
def _empty_flattened_tiles(x, out):
    for i, j in hl.tile([64, 0], block_size=[64, 32]):
        out[i, j] = x[i, j] + 1


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("step", [3, -3])
def test_full_width_persistent_range_counts_without_signed_subtraction(step):
    out = torch.zeros(8, dtype=torch.int64)
    lo, hi = torch.tensor(1), torch.tensor(6)
    with cpu_render_target():
        source = cpu_bind(_persistent_tensor_grid, (out, lo, hi, step)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    expression = _persistent_count(source)
    low, high = -(2**63), 2**63 - 1
    # Two empty directions, full-width distances with representable quotients,
    # and ordinary reached ranges. Never launch the extreme cases.
    for begin, end in ((1, low), (-1, high), (low, high), (high, low), (1, 6), (6, 1)):
        scope = {
            "lo": [np.int64(begin)],
            "hi": [np.int64(end)],
            "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
        }
        with np.errstate(over="ignore"):
            assert eval(expression, scope) == len(range(begin, end, step))


@skipUnlessBackends(["triton"])
def test_signed_device_range_to_exclusive_int64_ceiling(tmp_path):
    from triton.runtime.errors import InterpreterError

    @helion.kernel(backend="triton", static_shapes=False)
    def write(lo, step, out):
        for i in hl.grid(lo, 2**63, step):
            out[0] = i

    out = torch.zeros(1, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(write, (torch.tensor(0), torch.tensor(1), out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    # Interpret the real count and guard. Replace only the work loop so a
    # representable count does not require trillions of iterations.
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    loop = next(i for i, stmt in enumerate(fn.body) if isinstance(stmt, ast.For))
    fn.body[loop:] = ast.parse("tl.store(out, total_pids)").body
    replay = tmp_path / "count_only.py"
    replay.write_text(ast.unparse(tree))
    module = load_generated_module(replay)
    with (
        patch.object(helion.runtime, "get_num_sm", return_value=1),
        patch.object(
            module.helion_triton_helpers,
            "device_error",
            side_effect=RuntimeError("range overflow"),
        ) as fail,
    ):
        for start in (-(2**63), 0):
            for step in (1, 2, 3):
                args = (torch.tensor(start), torch.tensor(step), out)
                count = -(-(2**63 - start) // step)
                if count >= 2**63:
                    fail.reset_mock()
                    with pytest.raises(InterpreterError, match="range overflow"):
                        module.write(*args, _launcher=interpret_triton_kernel)
                    fail.assert_called_once_with("root grid count exceeds signed Int64")
                else:
                    module.write(*args, _launcher=interpret_triton_kernel)
                    assert out.item() == count


@skipUnlessBackends(["triton"])
def test_cartesian_sibling_uses_signed_start_below_wide_exclusive_stop(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(begin, stop, last, out):
        for i, j in hl.grid([begin, last], [stop, 2**63]):
            out[i] = j

    out = torch.full((2,), -1, dtype=torch.int64)
    args = torch.tensor(0), torch.tensor(1), torch.tensor(2**63 - 1), out
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    # The native type checker cannot compare an Int64 load to this literal.
    assert f"< {2**63}" not in source
    path = tmp_path / "wide_sibling.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).write(*args, _launcher=interpret_triton_kernel)
    assert out.tolist() == [2**63 - 1, -1]


@skipUnlessBackends(["cute"])
def test_device_grid_checks_the_signed_count_before_scheduling():
    out = torch.zeros(8, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(
            _persistent_tensor_grid, (out, torch.tensor(1), torch.tensor(6), 2)
        ).to_code(helion.Config(block_sizes=[], pid_type="persistent_interleaved"))
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    preamble = compile(
        ast.Module(fn.body[: fn.body.index(loop)], []), "<grid guard>", "exec"
    )

    def trap(*args, **kwargs):
        raise OverflowError("device grid overflow")

    scope = {
        "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64, Uint64=np.uint64),
        "_cute_inline_asm_elementwise": trap,
    }
    with np.errstate(over="ignore"):
        for lo, hi, total in (
            (0, 2**32, 2**31),
            (6, 1, 0),
            (-(2**63), 2**63 - 2, 2**63 - 1),
        ):
            scope.update(lo=[np.int64(lo)], hi=[np.int64(hi)])
            exec(preamble, scope)
            assert int(scope["total_pids"]) == total
        scope.update(lo=[np.int64(-(2**63))], hi=[np.int64(2**63 - 1)])
        with pytest.raises(OverflowError, match="device grid overflow"):
            exec(preamble, scope)


@skipUnlessBackends(["cute"])
def test_descending_device_grid_skips_overflow_for_an_empty_sibling():
    @helion.kernel(backend="cute", static_shapes=False)
    def reverse(out, stop, rows):
        for i, j in hl.grid([0, 0], [stop, rows], [-1, 1]):
            out[(i + j) % 8] = i

    args = (torch.zeros(8, dtype=torch.int64), torch.tensor(-3), torch.tensor(2))
    with cpu_render_target():
        source = cpu_bind(reverse, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    guard = next(
        n for n in fn.body if isinstance(n, ast.If) and "trap;" in ast.unparse(n)
    )
    test = compile(ast.Expression(guard.test), "<grid overflow>", "eval")
    scope = {
        "stop": [np.int64(-(2**63))],
        "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
    }
    with np.errstate(over="ignore"):
        for rows in (-1, 0, 1):
            scope["rows"] = [np.int64(rows)]
            assert bool(eval(test, scope)) == (rows > 0)


@skipUnlessBackends(["cute"])
def test_cartesian_overflow_guard_uses_the_dynamic_sibling_direction():
    @helion.kernel(backend="cute", static_shapes=False)
    def write(a, b, c, step, out):
        for i, j in hl.grid([a, b], [c, -1], [1, step]):
            out[(i + j) % 8] = 1

    args = (
        torch.tensor(0),
        torch.tensor(0, dtype=torch.uint8),
        torch.tensor(1),
        torch.tensor(-1, dtype=torch.int32),
        torch.zeros(8),
    )
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    # Evaluate the emitted guard, including CuTe's conditional expression
    # for the sibling, without scheduling the intentional overflow case.
    guard = next(
        n.test
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.If)
        and {"a", "c", "step"}
        <= {t.id for t in ast.walk(n.test) if isinstance(t, ast.Name)}
    )
    test = compile(ast.Expression(guard), "<Cartesian overflow>", "eval")
    scope = {
        "b": [np.uint8(0)],
        "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
    }
    with np.errstate(over="ignore"):
        for a, c in ((0, 1), (-(2**63), 2**63 - 1)):
            for step in (-1, 1):
                scope.update(a=[np.int64(a)], c=[np.int64(c)], step=[np.int32(step)])
                assert bool(eval(test, scope)) == (c - a >= 2**63 and step < 0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("pid_type", ["persistent_blocked", "persistent_interleaved"])
def test_persistent_worker_partition_near_the_signed_limit(pid_type):
    out = torch.zeros(8, dtype=torch.int64)
    args = (out, torch.tensor(1), torch.tensor(6), 2)
    with cpu_render_target():
        source = cpu_bind(_persistent_tensor_grid, args).to_code(
            helion.Config(block_sizes=[], pid_type=pid_type)
        )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    assert isinstance(loop.iter, ast.Call)
    preamble = compile(
        ast.Module(fn.body[: fn.body.index(loop)], []), "<grid partition>", "exec"
    )
    bounds = [
        compile(ast.Expression(n), "<worker range>", "eval") for n in loop.iter.args
    ]
    assert isinstance(loop.target, ast.Name)
    # The CuTe front-end may lower a wide range to an iteration count.
    # Replay its actual induction prefix to recover the logical program ID.
    prefix = []
    if loop.target.id != "virtual_pid":
        for stmt in loop.body:
            assert isinstance(stmt, ast.Assign)
            prefix.append(stmt)
            if any(
                isinstance(target, ast.Name) and target.id == "virtual_pid"
                for target in stmt.targets
            ):
                break
        else:
            raise AssertionError("the scheduling loop did not bind virtual_pid")
    induction = compile(ast.Module(prefix, []), "<virtual PID>", "exec")
    # Three physical workers make the last signed update wrap on the old
    # interleaved schedule, and overflow the old (total + workers - 1) on
    # the blocked schedule. Replay just the two endpoints of each worker.
    for total in (2**31, 2**63 - 1):
        for worker in range(3):
            scope = {
                "lo": [np.int64(-(2**63))],
                "hi": [np.int64(-(2**63) + 2 * total)],
                "_NUM_SM": 3,
                "cutlass": SimpleNamespace(
                    Int32=np.int32, Int64=np.int64, Uint64=np.uint64
                ),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(
                        block_idx=lambda worker=worker: (np.int32(worker), 0, 0)
                    )
                ),
            }
            with np.errstate(over="ignore"):
                exec(preamble, scope)
                endpoints = [int(eval(n, scope)) for n in bounds]
                r = range(*endpoints)
                assert r and r[-1] + r.step <= 2**63 - 1
                logical = []
                for iteration in (r[0], r[-1]):
                    scope[loop.target.id] = np.int64(iteration)
                    exec(induction, scope)
                    logical.append(int(scope["virtual_pid"]))
                if pid_type == "persistent_blocked":
                    width = -(-total // 3)
                    assert logical == [
                        worker * width,
                        min((worker + 1) * width, total) - 1,
                    ]
                else:
                    assert logical == [worker, worker + ((total - 1 - worker) // 3) * 3]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("group", [1, 2])
def test_persistent_cartesian_product_and_pid_decode_use_wide_counts(group, tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(n, m, k, out):
        for i, j, h in hl.grid([1, 1, 1], [n, m, k]):
            out[(i + j + h) % 8] = i

    out = torch.zeros(8)
    with cpu_render_target():
        source = cpu_bind(write, (5, 7, 9, out)).to_code(
            helion.Config(
                block_sizes=[], pid_type="persistent_blocked", l2_groupings=[group]
            )
        )
    count = _persistent_count(source)
    scope = {n: np.int32(65536) for n in count.co_names if n.startswith("grid_length")}
    scope.update(
        tl=SimpleNamespace(cast=lambda x, dtype: dtype(x), int64=np.int64),
        virtual_pid=np.int64(2**32),
    )
    with np.errstate(over="raise", divide="raise"):
        assert eval(count, scope) == 2**48
        loop = next(
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
        )
        for stmt in loop.body:
            exec(compile(ast.Module([stmt], []), "<PID decode>", "exec"), scope)
            if "pid_2" in scope:
                break
        assert [scope[f"pid_{i}"] for i in range(3)] == [0, 0, 1]
    path = tmp_path / "host_product.py"
    path.write_text(source)
    run = load_generated_module(path).write
    with patch("helion.runtime.get_num_sm", return_value=3):
        with pytest.raises(OverflowError, match="root grid count"):
            run(
                2**32 + 1,
                2**32 + 1,
                2,
                out,
                _launcher=lambda *args, **kwargs: pytest.fail("oversized launch"),
            )
        run(2**32 + 1, 2**32 + 1, 1, out, _launcher=lambda *args, **kwargs: None)


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_grouped_xyz_flattens_physical_coordinates_in_signed_int64(ndim):
    @helion.kernel(backend=_get_backend(), static_shapes=True)
    def write_2d(out):
        for i, j in hl.grid([65535, 65535]):
            out[(i + j) % 8] = i - j

    @helion.kernel(backend=_get_backend(), static_shapes=True)
    def write_3d(out):
        for i, j, k in hl.grid([65535, 65535, 2]):
            out[(i + j + k) % 8] = i - j + k

    write = write_2d if ndim == 2 else write_3d
    with cpu_render_target():
        source = cpu_bind(write, (torch.zeros(8),)).to_code(
            helion.Config(block_sizes=[], pid_type="xyz", l2_groupings=[8])
        )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    x, y, z = physical = (np.int32(0), np.int32(32769), np.int32(1))
    scope = {
        "tl": SimpleNamespace(
            program_id=physical.__getitem__,
            cast=lambda x, dtype: dtype(x),
            int64=np.int64,
        ),
        "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
        "cute": SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: physical)),
    }
    # Replay only this physical block, using the types actually emitted.
    with np.errstate(over="raise"):
        for stmt in fn.body:
            exec(compile(ast.Module([stmt], []), "<grouped XYZ>", "exec"), scope)
            if f"pid_{ndim - 1}" in scope:
                break
    if ndim == 3:
        # The folded divisor would otherwise be a UInt32 literal in Triton.
        assert isinstance(scope["inner_2d_size"], np.int64)
        assert scope["pid_2"] == z
    flat = int(x) + int(y) * 65535
    group, lane = divmod(flat, 8 * 65535)
    assert flat == 2**31 + 32767
    assert (scope["pid_0"], scope["pid_1"]) == (group * 8 + lane % 8, lane // 8)


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("static", [False, True])
def test_short_l2_group_in_a_flat_launch_uses_only_live_rows(static):
    @helion.kernel(backend=_get_backend(), static_shapes=static)
    def write(shape, out):
        for i, j in hl.grid([2, shape.size(0)]):
            out[i] = j

    n = 2**29
    with cpu_render_target():
        source = cpu_bind(write, (torch.empty(1).expand(n), torch.zeros(2))).to_code(
            helion.Config(block_sizes=[], pid_type="flat", l2_groupings=[8])
        )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    # The full launch fits Int32 (2*N), but counting eight rows in its only
    # two-row group would wrap the divisor to zero before the PID is decoded.
    for pid in (np.int32(0), np.int32(2 * n - 1)):
        scope = {
            "shape_size_0": np.int32(n),
            "tl": SimpleNamespace(program_id=lambda dim, pid=pid: pid),
            "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(block_idx=lambda pid=pid: (pid, 0, 0))
            ),
        }
        with np.errstate(over="raise", divide="raise"):
            for stmt in fn.body:
                assert isinstance(stmt, ast.Assign)
                if isinstance(stmt.value, ast.Constant):
                    scope[ast.unparse(stmt.targets[0])] = np.int32(stmt.value.value)
                else:
                    exec(compile(ast.Module([stmt], []), "<flat PID>", "exec"), scope)
                if "pid_1" in scope:
                    break
        assert (scope["pid_0"], scope["pid_1"]) == (int(pid) % 2, int(pid) // 2)


@skipUnlessBackends(["cute"])
def test_device_cartesian_product_checks_before_it_wraps():
    @helion.kernel(backend="cute", static_shapes=False)
    def write(lo, hi, out):
        for i, j in hl.grid([0, 0], [lo, hi]):
            out[(i + j) % 8] = i

    with cpu_render_target():
        source = cpu_bind(
            write, (torch.tensor(3), torch.tensor(4), torch.zeros(8))
        ).to_code(helion.Config(block_sizes=[], pid_type="persistent_interleaved"))
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    stmts = fn.body[: next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For))]
    preamble = compile(ast.Module(stmts, []), "<product guard>", "exec")

    def trap(*args, **kwargs):
        raise OverflowError("grid product")

    scope = {
        "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64, Uint64=np.uint64),
        "_cute_inline_asm_elementwise": trap,
    }
    with np.errstate(over="raise", divide="raise"):
        for x, y in [(65536, 65536), (2**63 - 1, 1), (2**63 - 1, 0)]:
            scope.update(lo=[np.int64(x)], hi=[np.int64(y)])
            exec(preamble, scope)
            assert scope["total_pids"] == x * y
        scope.update(lo=[np.int64(2**32)], hi=[np.int64(2**32)])
        with pytest.raises(OverflowError, match="grid product"):
            exec(preamble, scope)


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    "index_dtype,bound_dtype,width",
    [
        (torch.int8, torch.int8, 4),
        (torch.int64, torch.int8, 4),
        (torch.int64, torch.int64, 1),
        (torch.int64, torch.int64, 4),
    ],
)
def test_observed_root_tile_count_uses_the_full_signed_interval(
    tmp_path, index_dtype, bound_dtype, width
):
    @helion.kernel(backend="triton", static_shapes=True, index_dtype=index_dtype)
    def write(lo, hi, out, block: hl.constexpr):
        for tile in hl.tile(lo, hi, block_size=block):
            out[1] = tile.count
            out[2] = tile.count - 34 < 0

    # The largest legal root extent yields INT64_MAX tiles at width one;
    # at width four, its rounded-up numerator overflows. Sample one tile.
    sample = bound_dtype == torch.int64
    lo, hi = (-(2**63), -1) if sample else (-64, 65)
    out = torch.full((4,), -73, dtype=torch.int64)
    args = (
        torch.tensor(lo, dtype=bound_dtype),
        torch.tensor(hi, dtype=bound_dtype),
        out,
        width,
    )
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    if sample:
        tree = ast.parse(source)
        (loop,) = (n for n in ast.walk(tree) if isinstance(n, ast.For))
        loop.body.append(ast.Break())
        source = ast.unparse(tree)
    path = tmp_path / "count.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).write(*args, _launcher=interpret_triton_kernel)
    expected = -(-(hi - lo) // width)
    assert out.tolist() == [-73, expected, int(expected < 34), -73]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("pid_type", ["flat", "xyz"])
def test_root_count_uses_each_logical_axis(tmp_path, pid_type):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(lo, hi, table, out):
        for m, n in hl.tile([lo, 0], [hi, out.size(1)], block_size=[4, 2]):
            out[m, n] = m.count * 10 + table[n.count]

    table, out = torch.arange(12), torch.full((10, 5), -73)
    with cpu_render_target():
        source = cpu_bind(write, (1, 10, table, out)).to_code(
            helion.Config(block_sizes=[], pid_type=pid_type)
        )
    path = tmp_path / "axes.py"
    path.write_text(source)
    run = load_generated_module(path).write
    for lo, hi in [(1, 10), (3, 7)]:
        out.fill_(-73)
        run(lo, hi, table, out, _launcher=interpret_triton_kernel)
        expected = torch.full_like(out, -73)
        expected[lo:hi] = len(range(lo, hi, 4)) * 10 + len(range(0, out.size(1), 2))
        torch.testing.assert_close(out, expected)


@skipUnlessBackends(["triton"])
def test_l2_grouped_root_count_uses_the_logical_axis(tmp_path):
    @helion.kernel(backend="triton", static_shapes=True, index_dtype=torch.int8)
    def write(lo, hi, table, out):
        for m, n in hl.tile([lo, 0], [hi, 4], block_size=[4, 2]):
            if m.begin == -64 and n.begin == 0:
                out[0] = m.count
                out[1] = table[(m.count + 64) // 32] + n.count

    lo, hi = (torch.tensor(x, dtype=torch.int8) for x in (-64, 65))
    table, out = torch.arange(10, 14), torch.full((2,), -73)
    with cpu_render_target():
        source = cpu_bind(write, (lo, hi, table, out)).to_code(
            helion.Config(
                block_sizes=[], pid_type="persistent_blocked", l2_groupings=[2]
            )
        )
    path = tmp_path / "grouped_count.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).write(
            lo, hi, table, out, _launcher=interpret_triton_kernel
        )
    assert out.tolist() == [33, 13 + 2]


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("threads", [4, 1])
def test_single_flattened_root_reports_its_full_count(threads):
    @helion.kernel(backend="cute", static_shapes=False, index_dtype=torch.int64)
    def write(n, table, counts, lookups):
        for tile in hl.tile(n):
            counts[tile.index & 3] = tile.count
            lookups[tile.index & 3] = table[tile.count // (2**61) + 1]

    table, counts = torch.arange(10, 14), torch.full((4,), -73)
    lookups = torch.full_like(counts, -73)
    with cpu_render_target():
        source = cpu_bind(write, (9, table, counts, lookups)).to_code(
            helion.Config(
                block_sizes=[4], num_threads=[threads], pid_type="persistent_blocked"
            )
        )
    # Sample one virtual tile of the real persistent program, including its
    # count and stores, without traversing the full Int64 input domain.
    tree = ast.parse(source)
    (device,) = (
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    (loop,) = (stmt for stmt in device.body if isinstance(stmt, ast.For))
    loop.body.append(ast.Break())
    source = ast.unparse(tree)
    for n in (9, 2**63 - 1):
        for t in range(threads):
            simulate_independent_fragment(
                source,
                {
                    "n": np.int64(n),
                    "grid_extent": np.int64(n),
                    "table": table,
                    "_NUM_SM": 1,
                },
                {"counts": counts, "lookups": lookups},
                1,
                thread=(t, 0, 0),
                int32=np.int32,
                int64=np.int64,
            )
        count = len(range(0, n, 4))
        assert counts.tolist() == [count] * 4
        assert lookups.tolist() == [table[count // (2**61) + 1].item()] * 4


@skipUnlessBackends(["triton"])
def test_root_count_samples_tensor_bounds_before_the_loop(tmp_path):
    @helion.kernel(backend="triton", static_shapes=True)
    def write(lo, hi, out):
        for t in hl.tile(lo, hi, block_size=4):
            if t.begin == 0:
                hi[()] = 4
            out[t.begin // 4] = t.count

    lo, hi, out = torch.tensor(0), torch.tensor(9), torch.full((4,), -73)
    with cpu_render_target():
        source = cpu_bind(write, (lo, hi, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "count_at_entry.py"
    path.write_text(source)
    # One persistent worker reads the bound for all three of its tiles.
    # With two workers, this store could race the other worker's first read.
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).write(
            lo, hi, out, _launcher=interpret_triton_kernel
        )
    assert hi.item() == 4
    assert out.tolist() == [3, 3, 3, -73]


@skipUnlessBackends(["cute", "triton"])
def test_device_tile_count_preserves_the_last_partial_tile_at_signed_max():
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=False)
    def write(lo, hi, out):
        for tile in hl.tile(lo, hi, block_size=4):
            out[tile.index % 8] = tile.index

    with cpu_render_target():
        source = cpu_bind(
            write, (torch.tensor(0), torch.tensor(7), torch.zeros(8, dtype=torch.int64))
        ).to_code(helion.Config(block_sizes=[], pid_type="persistent_blocked"))
    expr = _persistent_count(source)
    scope = {
        "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
        "tl": SimpleNamespace(
            int64=np.int64,
            uint64=np.uint64,
            cast=lambda x, dtype: dtype(x),
            load=operator.itemgetter(0),
        ),
        "_BLOCK_SIZE_0": 4,
    }
    with np.errstate(over="raise"):
        for n in (0, 4, 7, 2**63 - 1):
            scope.update(lo=[np.int64(0)], hi=[np.int64(n)])
            assert eval(expr, scope) == -(-n // 4)

    @helion.kernel(backend=backend, static_shapes=False)
    def from_shape(out):
        for tile in hl.tile(out.size(0), block_size=4):
            out[tile] = 1

    with cpu_render_target():
        source = cpu_bind(from_shape, (torch.zeros(8),)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    expr = _persistent_count(source)
    size_name = next(n for n in expr.co_names if "size_0" in n)
    with np.errstate(over="raise"):
        for dtype in (np.int32, np.int64):
            max_extent = int(np.iinfo(dtype).max)
            scope[size_name] = dtype(max_extent)
            assert eval(expr, scope) == -(-max_extent // 4)


@skipUnlessBackends(["cute"])
def test_descending_grid_widens_int32_minimum_before_counting():
    @helion.kernel(backend="cute", static_shapes=False)
    def write(end, out):
        for i in hl.grid(0, end, -1):
            out[i % 8] = i

    with cpu_render_target():
        source = cpu_bind(
            write, (torch.tensor(-3, dtype=torch.int32), torch.zeros(8))
        ).to_code(helion.Config(block_sizes=[], pid_type="persistent_interleaved"))
    scope = {"end": [np.int32(-(2**31))], "cutlass": SimpleNamespace(Int64=np.int64)}
    with np.errstate(over="raise"):
        assert eval(_persistent_count(source), scope) == 2**31


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("pid_type", ["flat", "persistent_blocked"])
def test_root_sum_checked_before_dispatch(tmp_path, pid_type):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(n, m, out, other):
        for i in hl.grid(1, n):
            out[i % 8] = i
        for j in hl.grid(1, m):
            other[j % 8] = j

    args = torch.zeros(8), torch.zeros(8)
    with cpu_render_target():
        source = cpu_bind(write, (4, 5, *args)).to_code(
            helion.Config(block_sizes=[], pid_type=pid_type)
        )
    path = tmp_path / "root_sum.py"
    path.write_text(source)
    run = load_generated_module(path).write
    with patch("helion.runtime.get_num_sm", return_value=3):
        with pytest.raises(OverflowError, match="root grid count"):
            run(
                2**62 + 1,
                2**62 + 1,
                *args,
                _launcher=lambda *args, **kwargs: pytest.fail("oversized launch"),
            )
        run(2**62 + 1, 2**62, *args, _launcher=lambda *args, **kwargs: None)


@skipUnlessBackends(["triton"])
def test_loaded_tile_extent_controls_coordinate_and_count_width(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(end, out, width: hl.constexpr):
        for tile in hl.tile(end, block_size=width):
            out[tile.index % 8] = tile.index

    out = torch.full((8,), -9, dtype=torch.int64)
    with cpu_render_target():
        wide = cpu_bind(write, (torch.tensor(7), out, 4)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
        host = cpu_bind(write, (7, out, 4)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
        narrow = cpu_bind(write, (torch.tensor(7, dtype=torch.int8), out, 128)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    scope = {
        "tl": SimpleNamespace(
            cast=lambda v, dtype: torch.as_tensor(v).to(dtype),
            int64=torch.int64,
            int32=torch.int32,
            arange=torch.arange,
            load=lambda x: x,
        ),
        "virtual_pid": torch.tensor(2**29),
        "end": torch.tensor(2**31 + 1),
        "_BLOCK_SIZE_0": 4,
    }
    for source in (wide, host):
        scope.pop("mask_0", None)
        loop = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For))
        for stmt in loop.body:
            exec(compile(ast.Module([stmt], []), "<last tile>", "exec"), scope)
            if "mask_0" in scope:
                break
        assert scope["mask_0"].tolist() == [True, False, False, False]
        assert int(scope["indices_0"][0]) == 2**31
    path = tmp_path / "narrow_extent.py"
    path.write_text(narrow)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).write(
            torch.tensor(7, dtype=torch.int8),
            out,
            128,
            _launcher=interpret_triton_kernel,
        )
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6, -9]


@skipUnlessBackends(["triton"])
def test_empty_input_can_supply_a_wide_nonempty_tile_axis():
    from helion.runtime.kernel import _tensor_key

    @helion.kernel(backend="triton", static_shapes=False)
    def write(x, out):
        for tile in hl.tile(x.size(0), block_size=4):
            out[tile.index % 8] = tile.index

    out = torch.zeros(8, dtype=torch.int64)
    empty = torch.empty(1, 0)
    small, wide = empty.expand(7, 0), empty.expand(2**31 + 1, 0)
    with cpu_render_target():
        assert write._specialization_key(small) != write._specialization_key(wide)
        assert _tensor_key(write, wide) == write._specialization_key(wide)
        source = cpu_bind(write, (wide, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
        for dtype in (torch.int16, torch.int32, torch.uint32):
            forced = helion.kernel(
                backend="triton",
                index_dtype=dtype,
                static_shapes=False,
                config=helion.Config(block_sizes=[], pid_type="persistent_blocked"),
            )(write.fn)
            first = forced.bind((small, out))
            assert forced.bind((small, out)) is first
            too_wide = empty.expand(torch.iinfo(dtype).max + 2, 0)
            wide_bound = forced.bind((too_wide, out))
            assert wide_bound is not first
            assert wide_bound.env.index_dtype == dtype
            assert wide_bound.env.wide_empty_grid
            assert forced.bind((too_wide, out)) is wide_bound
            nonempty = torch.empty(1, 1).expand(torch.iinfo(dtype).max + 2, 1)
            with pytest.raises(helion.exc.InputTensorNumelExceedsIndexType):
                forced.bind((nonempty, out))
    loop = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For))
    scope = {
        "tl": SimpleNamespace(int64=torch.int64, arange=torch.arange),
        "virtual_pid": torch.tensor(2**29),
        "_BLOCK_SIZE_0": 4,
        "x_size_0": 2**31 + 1,
    }
    for stmt in loop.body:
        exec(compile(ast.Module([stmt], []), "<last empty-source tile>", "exec"), scope)
        if "mask_0" in scope:
            break
    assert scope["mask_0"].tolist() == [True, False, False, False]
    assert int(scope["indices_0"][0]) == 2**31


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("pid_type", ["flat", "persistent_blocked"])
@pytest.mark.parametrize("static", [False, True])
def test_narrow_storage_indices_do_not_narrow_empty_tensor_grid(
    pid_type, static, tmp_path
):
    def write(x, out):
        for t in hl.tile(x.size(0), block_size=1):
            if t.begin == 65536:
                out[0] = t.begin

    config = helion.Config(block_sizes=[], pid_type=pid_type)
    empty = torch.empty(1, 0)
    wide = empty.expand(65537, 0)
    out = torch.empty(1, dtype=torch.int64)
    sources = []
    for warm in (False, True):
        kernel = helion.kernel(
            backend="triton",
            index_dtype=torch.int16,
            static_shapes=static,
            config=config,
        )(write)
        with cpu_render_target():
            if warm:
                kernel.bind((empty.expand(7, 0), out))
            bound = kernel.bind((wide, out))
            assert kernel.bind((wide, out)) is bound
            sources.append(bound.to_code(config))
    # The same compiled program must work for a direct wide binding and one
    # populated after the narrow entry, including its actual scheduler.
    assert sources[0] == sources[1]
    path = tmp_path / "last_pid.py"
    path.write_text(sources[0])
    worker = 65536 if pid_type == "flat" else 65536 // -(-65537 // 152)
    out.fill_(-1)
    with patch.object(helion.runtime, "get_num_sm", return_value=152):
        load_generated_module(path).write(
            wide, out, _launcher=partial(interpret_triton_kernel, physical_pid=worker)
        )
    assert out.item() == 65536


@skipUnlessBackends(["triton"])
def test_narrow_storage_indices_keep_loaded_int32_tile_mask(tmp_path):
    @helion.kernel(backend="triton", index_dtype=torch.int16, static_shapes=False)
    def write(end, out):
        for tile in hl.tile(end, block_size=4):
            if tile.begin == 32768:
                out[tile.index & 3] = 1

    end = torch.tensor(32769, dtype=torch.int32)
    out = torch.zeros(4, dtype=torch.int32)
    with cpu_render_target():
        src = cpu_bind(write, (end, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    path = tmp_path / "loaded_stop.py"
    path.write_text(src)
    with patch.object(helion.runtime, "get_num_sm", return_value=152):
        load_generated_module(path).write(
            end,
            out,
            _launcher=partial(interpret_triton_kernel, physical_pid=8192 % 152),
        )
    assert out.bool().tolist() == [True, False, False, False]


@skipUnlessBackends(["triton", "cute"])
def test_narrow_loaded_stop_masks_padded_physical_lanes(tmp_path):
    backend = _get_backend()

    @helion.kernel(backend=backend, index_dtype=torch.int8, static_shapes=True)
    def write(end, out):
        for tile in hl.tile(end, block_size=256):
            out[tile.index & 1] = 1

    end = torch.tensor(1, dtype=torch.int8)
    out = torch.zeros(2, dtype=torch.int32)
    with cpu_render_target():
        source = cpu_bind(write, (end, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_interleaved")
        )
    if matchesBackends(["triton"]):
        path = tmp_path / "padded_lanes.py"
        path.write_text(source)
        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            load_generated_module(path).write(
                end, out, _launcher=interpret_triton_kernel
            )
    else:
        # Lane 129 would wrap to -127 and write element 1 under an Int8 mask.
        for lane in (0, 1, 129):
            simulate_independent_fragment(
                source,
                {"end": [np.int8(1)], "_NUM_SM": 1},
                {"out": out},
                1,
                thread=(np.int32(lane), 0, 0),
                int32=np.int32,
                int64=np.int64,
            )
    assert out.tolist() == [1, 0]


@skipUnlessBackends(["triton", "cute"])
def test_unsigned_storage_index_keeps_flat_pid_division_signed(tmp_path):
    backend = _get_backend()

    @helion.kernel(backend=backend, index_dtype=torch.uint32, static_shapes=True)
    def write(out):
        for i, j in hl.grid([3, 4]):
            out[i * 4 + j] = i * 4 + j

    out = torch.full((12,), -1, dtype=torch.int32)
    with cpu_render_target():
        source = cpu_bind(write, (out,)).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    if matchesBackends(["triton"]):
        path = tmp_path / "unsigned_storage.py"
        path.write_text(source)
        load_generated_module(path).write(out, _launcher=interpret_triton_kernel)
    else:
        simulate_independent_fragment(source, {}, {"out": out}, 12, int32=np.int32)
    assert out.tolist() == list(range(12))


@skipUnlessBackends(["triton"])
def test_unsigned_tile_end_widens_the_unwrapped_origin(tmp_path):
    @helion.kernel(backend="triton", index_dtype=torch.uint32, static_shapes=True)
    def write(out):
        for tile in hl.tile(2**31 + 1, block_size=4):
            out[tile.index & 3] = tile.end

    out = torch.full((4,), -1, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(write, (out,)).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "unsigned_tile_end.py"
    path.write_text(source)
    load_generated_module(path).write(
        out, _launcher=partial(interpret_triton_kernel, physical_pid=2**29)
    )
    assert out.tolist() == [2**31 + 1, -1, -1, -1]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("n", [2**31, 2**31 + 1])
def test_single_static_pid_divisor_is_signed(n):
    @helion.kernel(backend="triton", static_shapes=True)
    def write(n: hl.constexpr, out):
        for i, j in hl.grid([n, 2]):
            out[(i + j) % 8] = j - i

    with cpu_render_target():
        src = cpu_bind(write, (n, torch.zeros(8))).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    fn = next(
        x
        for x in ast.parse(src).body
        if isinstance(x, ast.FunctionDef) and x.name.startswith("_helion_")
    )
    loop = next(x for x in fn.body if isinstance(x, ast.For))
    scope = {
        "virtual_pid": np.int64(n + 1),
        "tl": SimpleNamespace(cast=lambda v, t: t(v), int64=np.int64),
    }
    for stmt in loop.body:
        exec(compile(ast.Module([stmt], []), "<signed PID divisor>", "exec"), scope)
        if "pid_1" in scope:
            break
    assert isinstance(scope["num_blocks_0"], np.int64)
    assert (scope["pid_0"], scope["pid_1"]) == (1, 1)


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("step_on_empty", [False, True])
def test_empty_root_checks_a_device_step_before_folding(tmp_path, step_on_empty):
    from triton.runtime.errors import InterpreterError

    @helion.kernel(backend="triton", static_shapes=False)
    def write(n, step, out, on_empty: hl.constexpr):
        steps = [1, step] if on_empty else [step, 1]
        for i, j in hl.grid([0, 0], [n, 0], steps):
            out[(i + j) % 8] = i

    out = torch.full((8,), -1)
    with cpu_render_target():
        source = cpu_bind(write, (8, torch.tensor(2), out, step_on_empty)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "empty_device_step.py"
    path.write_text(source)
    module = load_generated_module(path)
    with (
        patch.object(helion.runtime, "get_num_sm", return_value=1),
        patch.object(
            module.helion_triton_helpers,
            "device_error",
            side_effect=RuntimeError("zero device stride"),
        ) as fail,
    ):
        module.write(
            2**63,
            torch.tensor(1),
            out,
            step_on_empty,
            _launcher=interpret_triton_kernel,
        )
        assert out.tolist() == [-1] * 8
        with pytest.raises(InterpreterError, match="zero device stride"):
            module.write(
                2**63,
                torch.tensor(0),
                out,
                step_on_empty,
                _launcher=interpret_triton_kernel,
            )
        fail.assert_called_once_with("range() arg 3 must not be zero")


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint8])
def test_grid_checks_host_endpoints_with_a_device_stride(tmp_path, dtype):
    from triton.runtime.errors import InterpreterError

    @helion.kernel(backend="triton", static_shapes=False)
    def write(n, a, b, step, out):
        for i, j in hl.grid([-1, a], [n, b], [1, step]):
            out[(i + j) % 8] = 1

    out = torch.full((8,), -1)
    with cpu_render_target():
        source = cpu_bind(write, (9, 2, 5, torch.tensor(1, dtype=dtype), out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "dynamic_empty_sibling.py"
    path.write_text(source)
    module = load_generated_module(path)
    with (
        patch.object(helion.runtime, "get_num_sm", return_value=1),
        patch.object(
            module.helion_triton_helpers,
            "device_error",
            side_effect=RuntimeError("zero device stride"),
        ),
    ):
        # Equal endpoints are empty for both signs. Unsigned device strides
        # also prove that a reversed range is empty, without a scalar readback.
        empty = [(7, 7, 1), (7, 7, -1)] if dtype.is_signed else [(7, 7, 1), (9, 7, 1)]
        for a, b, step in empty:
            module.write(
                2**63 - 1,
                a,
                b,
                torch.tensor(step, dtype=dtype),
                out,
                _launcher=interpret_triton_kernel,
            )
        assert out.tolist() == [-1] * 8
        with pytest.raises(InterpreterError, match="zero device stride"):
            module.write(
                2**63 - 1,
                7,
                7,
                torch.tensor(0, dtype=dtype),
                out,
                _launcher=interpret_triton_kernel,
            )
        # A forward range with a positive stride still cannot excuse overflow.
        with pytest.raises(OverflowError, match="root grid axis length"):
            module.write(
                2**63 - 1,
                7,
                9,
                torch.tensor(1, dtype=dtype),
                out,
                _launcher=lambda *args, **kwargs: pytest.fail("unexpected launch"),
            )


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    ("tiles", "workers", "active"), [(2**31 - 257, 128, 128), (2049, 2048, 1025)]
)
def test_static_xcd_active_workers_use_safe_ceiling_division(tiles, workers, active):
    @helion.kernel(backend="triton", static_shapes=True)
    def write(out, count: hl.constexpr):
        for i in hl.grid(count):
            out[i % 8] = i

    with (
        cpu_render_target(),
        patch("helion._compat.supports_amd_cdna_tunables", return_value=True),
    ):
        bound = cpu_bind(write, (torch.zeros(8), tiles))
        bound.config_spec.num_xcd = 8
        source = bound.to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked", xcd_remap=True)
        )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    scope = {
        "tl": SimpleNamespace(maximum=max, cdiv=lambda a, b: (a + b - 1) // b),
        "_NUM_SM": np.int32(workers),
    }
    with np.errstate(over="raise"):
        for stmt in fn.body:
            exec(compile(ast.Module([stmt], []), "<static XCD workers>", "exec"), scope)
            if isinstance(stmt, ast.Assign):
                name = ast.unparse(stmt.targets[0])
                scope[name] = np.int32(scope[name])
            if "xcd_active_workers" in scope:
                break
    assert scope["xcd_active_workers"] == active


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("pid_type", ["flat", "xyz"])
def test_nonpersistent_python_grid_count_is_checked_before_launch(pid_type, tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def flat(n, out):
        for i in hl.grid(n):
            out[i % 8] = i

    @helion.kernel(backend="triton", static_shapes=False)
    def xyz(n, out):
        for i, j in hl.grid([n, 3]):
            out[(i + j) % 8] = i

    kernel = xyz if pid_type == "xyz" else flat
    out = torch.zeros(8)
    with cpu_render_target():
        source = cpu_bind(kernel, (7, out)).to_code(
            helion.Config(block_sizes=[], pid_type=pid_type)
        )
    path = tmp_path / "host_launch.py"
    path.write_text(source)
    with pytest.raises(OverflowError, match="signed Int64"):
        getattr(load_generated_module(path), kernel.fn.__name__)(
            2**63,
            out,
            _launcher=lambda *args, **kwargs: pytest.fail("oversized launch"),
        )


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("computed", [False, True])
def test_host_stop_checks_the_signed_device_bound(tmp_path, computed):
    @helion.kernel(static_shapes=False)
    def edges(n, stop, out, computed: hl.constexpr):
        for t in hl.tile(n, n + 5 if computed else stop, block_size=4):
            out[t.index & 7] = t.end // 2

    out = torch.zeros(8, dtype=torch.int64)
    with cpu_render_target():
        src = cpu_bind(edges, (7, 12, out, computed)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "edges.py"
    path.write_text(src)
    run = load_generated_module(path).edges
    calls = []
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        for n in (-7, 7, 2**63 - 6):
            run(n, n + 5, out, computed, _launcher=lambda *a, **kw: calls.append(a))
        with pytest.raises(OverflowError, match="runtime root grid bound"):
            run(
                2**63 - 5,
                2**63,
                out,
                computed,
                _launcher=lambda *a, **kw: calls.append(a),
            )
    assert len(calls) == 3


@skipUnlessBackends(["triton", "cute"])
def test_device_endpoint_does_not_hide_a_wide_host_bound(tmp_path):
    @helion.kernel(backend=_get_backend(), static_shapes=False)
    def write(lo, n, out, start_on_device: hl.constexpr):
        start, stop = (lo, n + 5) if start_on_device else (n - 5, lo)
        for tile in hl.tile(start, stop, block_size=4):
            out[tile.index % 8] = tile.end // 2

    out, lo = torch.zeros(8, dtype=torch.int64), torch.tensor(7)
    for start_on_device in (True, False):
        with cpu_render_target():
            source = cpu_bind(write, (lo, 7, out, start_on_device)).to_code(
                helion.Config(block_sizes=[], pid_type="persistent_blocked")
            )
        path = tmp_path / f"mixed_bound_{start_on_device}.py"
        path.write_text(source)
        run = load_generated_module(path).write
        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            run(lo, -7, out, start_on_device, _launcher=lambda *args, **kwargs: None)
            with pytest.raises(OverflowError, match="runtime root grid bound"):
                run(
                    lo,
                    2**63 - 5 if start_on_device else -(2**63) + 4,
                    out,
                    start_on_device,
                    _launcher=lambda *args, **kwargs: pytest.fail(
                        "oversized bound reached launcher"
                    ),
                )


@skipUnlessBackends(["triton", "cute"])
def test_zero_origin_tile_checks_plain_python_stop_before_rounding(tmp_path):
    @helion.kernel(backend=_get_backend(), static_shapes=False)
    def edges(n, m, out):
        for t, col in hl.tile([n, m], block_size=[4, 1]):
            out[t.index % 8, col] = t.end // 2

    out = torch.zeros(8, 1, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(edges, (7, 1, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "plain_stop.py"
    path.write_text(source)
    run = load_generated_module(path).edges
    calls = []

    def record(*args, **kwargs):
        assert all(-(2**63) <= v < 2**63 for v in args if type(v) is int)
        calls.append(args)

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        run(2**63 - 1, 1, out, _launcher=record)
        run(2**63 - 1, 0, out, _launcher=record)
        for n, m in ((2**63, 1), (2**64, 0)):
            with pytest.raises(OverflowError, match="signed Int64"):
                run(n, m, out, _launcher=record)
    assert len(calls) == 2


@skipUnlessBackends(["triton"])
def test_an_empty_root_does_not_narrow_a_later_live_root(tmp_path):
    @helion.kernel(backend="triton", static_shapes=True)
    def write(out, other):
        bs = hl.register_block_size(4)
        for t in hl.tile(0, block_size=bs):
            out[t.index % 8] = t.index
        for t in hl.tile(2**31, 2**31 + 2, block_size=bs):
            other[t.index % 8] = t.index

    args = (
        torch.full((8,), -1, dtype=torch.int64),
        torch.full((8,), -1, dtype=torch.int64),
    )
    with cpu_render_target():
        bound = cpu_bind(write, args)
        assert len(bound.config_spec.block_sizes) == 1
        source = bound.to_code(
            helion.Config(block_sizes=[4], pid_type="persistent_blocked")
        )
    path = tmp_path / "distinct_roots.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=2):
        load_generated_module(path).write(*args, _launcher=interpret_triton_kernel)
    assert args[0].tolist() == [-1] * 8
    assert args[1].tolist() == [2**31, 2**31 + 1] + [-1] * 6


@skipUnlessBackends(["triton"])
def test_old_static_range_config_keeps_the_descending_loop_slot():
    @helion.kernel(backend="triton", static_shapes=False)
    def add(out):
        for tile in hl.tile(out.size(0)):
            result = out[tile]
            for i in hl.grid(8, 0, -1):
                result = result + i
            for j in hl.grid(3):
                result = result + j
            out[tile] = result

    with cpu_render_target():
        source = cpu_bind(add, (torch.zeros(8, dtype=torch.int32),)).to_code(
            helion.Config(block_sizes=[4], static_ranges=[True])
        )
    loops = [
        ast.unparse(n.iter.func)
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
    ]
    # A preexisting one-slot config unrolls the second loop. Inserting a slot
    # for the eight descending trips would move that setting to the wrong loop.
    assert loops == ["tl.range", "tl.static_range"]


@skipUnlessBackends(["triton"])
def test_scalar_grid_checks_host_count_and_remaps_wide_xcd_workers(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(n, out):
        for i in hl.grid(n):
            out[i % 8] = i

    out = torch.zeros(8)
    with (
        cpu_render_target(),
        patch("helion._compat.supports_amd_cdna_tunables", return_value=True),
    ):
        bound = cpu_bind(write, (7, out))
        bound.config_spec.num_xcd = 8
        source = bound.to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked", xcd_remap=True)
        )
    path = tmp_path / "single_xcd.py"
    path.write_text(source)
    run = load_generated_module(path).write
    with (
        patch.object(helion.runtime, "get_num_sm", return_value=128),
        patch.object(helion.runtime, "get_num_xcd", return_value=8),
        pytest.raises(OverflowError, match="runtime root grid bound"),
    ):
        run(
            2**63,
            out,
            _launcher=lambda *args, **kwargs: pytest.fail("oversized host grid"),
        )
    device = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    loop = next(n for n in device.body if isinstance(n, ast.For))
    preamble = compile(
        ast.Module(device.body[: device.body.index(loop)], []),
        "<XCD partition>",
        "exec",
    )
    scope = {
        "_NUM_SM": 128,
        "_NUM_XCDS": 8,
        "grid_extent": np.int64(2**63 - 1),
        "tl": SimpleNamespace(
            int64=np.int64,
            uint64=np.uint64,
            cast=lambda x, dtype: dtype(x),
            maximum=max,
            where=lambda c, a, b: a if c else b,
            program_id=lambda _: np.int32(127),
        ),
    }
    with np.errstate(over="raise"):
        exec(preamble, scope)
        assert scope["xcd_active_workers"] == 128
        assert scope["end_pid"] == 2**63 - 1
        assert scope["start_pid"] == 127 * (2**56)
        # The interleaved XCD mapper uses the same per-XCD ceiling division
        # over the virtual programs, rather than over the physical workers.
        stmt = next(
            n
            for n in device.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "xcd_pids_per"
        )
        scope["xcd_active_workers"] = np.int64(2**63 - 1)
        exec(compile(ast.Module([stmt], []), "<virtual XCD groups>", "exec"), scope)
        assert scope["xcd_pids_per"] == 2**60


@skipUnlessBackends(["cute"])
def test_shifted_lane_scheduler_checks_extent_before_tile_division(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(x, out):
        for tile in hl.tile(8 - 2**63, x.size(0)):
            out[tile.index & 3] = tile.index

    out = torch.zeros(4, dtype=torch.int64)

    def check(pid):
        with cpu_render_target():
            source = cpu_bind(write, (torch.empty(7), out)).to_code(
                helion.Config(block_sizes=[128], num_threads=[32], pid_type=pid)
            )
        path = tmp_path / f"{pid}.py"
        path.write_text(source)
        module = load_generated_module(path)
        counts = []

        def record(entry, grid, *args, block):
            if pid == "flat":
                counts.append(grid[0])
                return
            # Match the launch ABI instead of evaluating the Python host's
            # unbounded expression in place of the generated device arithmetic.
            scope = {
                name: np.int64(value)
                for name, value in inspect.signature(entry)
                .bind(*args)
                .arguments.items()
                if isinstance(value, int)
            }
            scope.update(_BLOCK_SIZE_0=128, cutlass=SimpleNamespace(Int64=np.int64))
            counts.append(eval(_persistent_count(source), scope))

        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            module.write(torch.empty(7), out, _launcher=record)
            if pid == "flat":
                module.write(torch.empty(8), out, _launcher=record)
                assert counts == [2**56, 2**56]
            else:
                with pytest.raises(OverflowError, match="does not fit signed Int64"):
                    module.write(torch.empty(8), out, _launcher=record)
                assert counts == [2**56]

    check("flat")
    check("persistent_blocked")


@skipUnlessBackends(["triton", "cute"])
def test_flat_grid_uses_signed_product_before_rounding_tile_count(tmp_path):
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=False)
    def write(x, y, out):
        for i, j in hl.tile([x.size(0), y.size(0)]):
            out[i, j] = 1

    out = torch.zeros(5, 7)
    with cpu_render_target():
        source = cpu_bind(write, (torch.zeros(5), torch.zeros(7), out)).to_code(
            helion.Config(
                block_sizes=[16, 16],
                flatten_loops=[True],
                pid_type="persistent_blocked",
            )
        )
    count = _persistent_count(source)
    scope = {
        name: np.int32(65536) for name in count.co_names if name.endswith("_size_0")
    }
    scope.update(
        {name: 256 for name in count.co_names if name.startswith("_BLOCK_SIZE")}
    )
    scope.update(
        tl=SimpleNamespace(cast=lambda x, dtype: dtype(x), int64=np.int64),
        cutlass=SimpleNamespace(Int64=np.int64),
    )
    with np.errstate(over="raise"):
        assert eval(count, scope) == 2**24
    path = tmp_path / "flat_grid.py"
    path.write_text(source)
    run = load_generated_module(path).write
    x = torch.empty(1).expand(2**31)
    y = torch.empty(1).expand(2**32)
    error = "before dividing by the tile width"
    with patch.object(helion.runtime, "get_num_sm", return_value=128):
        # The boundary has only 2**55 programs, but its raw count is 2**63.
        for left in (x, y):
            with pytest.raises(OverflowError, match=error):
                run(
                    left,
                    y,
                    out,
                    _launcher=lambda *args, **kwargs: pytest.fail(
                        "launched oversized grid"
                    ),
                )
    static = helion.kernel(backend=backend, static_shapes=True)(write.fn)
    with cpu_render_target(), pytest.raises(helion.exc.InvalidConfig, match=error):
        cpu_bind(static, (x, y, out)).to_code(
            helion.Config(
                block_sizes=[16, 16],
                flatten_loops=[True],
                pid_type="persistent_blocked",
            )
        )


@skipUnlessBackends(["triton"])
def test_flat_xyz_checks_raw_grid_and_shifted_product_before_launch(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def tiles(x, y, out):
        for i, j in hl.tile([x.size(0), y.size(0)]):
            out[i, j] = 1

    @helion.kernel(backend="triton", static_shapes=False)
    def shifted(n, m, out):
        for i, j in hl.grid([1, 1], [n, m]):
            out[(i + j) % 8] = 1

    out = torch.zeros(5, 7)
    with cpu_render_target():
        xyz = cpu_bind(tiles, (torch.zeros(5), torch.zeros(7), out)).to_code(
            helion.Config(block_sizes=[16, 16], flatten_loops=[True], pid_type="xyz")
        )
        flat = cpu_bind(shifted, (5, 7, torch.zeros(8))).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "xyz.py"
    path.write_text(xyz)
    run = load_generated_module(path).tiles
    y = torch.empty(1).expand(2**32)
    x = torch.empty(1).expand(2**31)
    with pytest.raises(OverflowError, match="before dividing by the tile width"):
        run(
            x, y, out, _launcher=lambda *a, **kw: pytest.fail("launched oversized grid")
        )
    calls = []
    run(x[:-1], y, out, _launcher=lambda *a, **kw: calls.append(a))
    assert calls
    path = tmp_path / "shifted.py"
    path.write_text(flat)
    with pytest.raises(OverflowError, match="root grid count"):
        load_generated_module(path).shifted(
            2**32 + 1,
            2**32 + 1,
            torch.zeros(8),
            _launcher=lambda *a, **kw: pytest.fail("launched oversized grid"),
        )


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_flat_grid_checks_device_scalar_factors_before_multiplying(tmp_path, dtype):
    from triton.runtime.errors import InterpreterError

    @helion.kernel(backend="triton", static_shapes=False)
    def write(m, n, out):
        for i, j in hl.tile([m, n]):
            out[i, j] = 1

    out = torch.zeros(8, 8)
    args = torch.tensor(5, dtype=dtype), torch.tensor(7, dtype=dtype), out
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(
                block_sizes=[4, 4], flatten_loops=[True], pid_type="persistent_blocked"
            )
        )
    path = tmp_path / "device_product.py"
    path.write_text(source)
    module = load_generated_module(path)
    with patch.object(helion.runtime, "get_num_sm", return_value=2):
        module.write(*args, _launcher=interpret_triton_kernel)
        assert torch.equal(out, (torch.arange(8)[:, None] < 5) & (torch.arange(8) < 7))
        with (
            patch.object(
                module.helion_triton_helpers,
                "device_error",
                side_effect=RuntimeError("unsupported raw grid"),
            ) as fail,
            pytest.raises(InterpreterError, match="unsupported raw grid"),
        ):
            module.write(
                torch.tensor(2**31, dtype=dtype),
                torch.tensor(2**32, dtype=dtype),
                out,
                _launcher=interpret_triton_kernel,
            )
        fail.assert_called_once_with(
            "flattened root grid count exceeds signed Int64 before dividing by the tile width"
        )


@skipUnlessBackends(["cute", "triton"])
def test_unreachable_literal_axis_is_not_validated():
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=True)
    def write(out):
        for i, j in hl.grid([-(2**62), 0], [2**62 + 1, 0]):
            out[(i + j) % 8] = i

    with cpu_render_target():
        source = cpu_bind(write, (torch.zeros(8),)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    assert eval(_persistent_count(source), {}) == 0


@skipUnlessBackends(["triton"])
def test_uint32_grid_axes_decode_with_signed_virtual_pid(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(x, y, out):
        for i, j in hl.grid([x, y]):
            out[i, j] = i * 4 + j

    args = (
        torch.tensor(3, dtype=torch.uint32),
        torch.tensor(4, dtype=torch.uint32),
        torch.full((3, 4), -1),
    )
    with cpu_render_target():
        source = cpu_bind(write, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "unsigned_decode.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=4):
        load_generated_module(path).write(*args, _launcher=interpret_triton_kernel)
    torch.testing.assert_close(args[-1], torch.arange(12).view(3, 4))


@skipUnlessBackends(["triton"])
def test_oversized_literal_root_rejected_before_launch():
    @helion.kernel(backend="triton", static_shapes=True)
    def write(out):
        for i in hl.grid(-(2**62), 2**62):
            out[i % 8] = i

    with (
        cpu_render_target(),
        pytest.raises(helion.exc.InvalidConfig, match="root grid count exceeds"),
    ):
        cpu_bind(write, (torch.zeros(8),)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )


@skipUnlessBackends(["triton"])
def test_symbolic_stride_uses_host_extents_for_each_launch(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def copy(out, stride):
        for i, j in hl.grid([0, 0], [out.size(0), 1], [stride, 1]):
            out[i + j] = i

    out = torch.zeros(5, dtype=torch.int32)
    with cpu_render_target():
        source = cpu_bind(copy, (out, 2)).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "symbolic_grid.py"
    path.write_text(source)
    fn = load_generated_module(path).copy
    grids = []
    for tensor, stride in ((out, 2), (out, -2), (torch.zeros_like(out)[:3], 2)):
        fn(
            tensor,
            stride,
            _launcher=lambda kernel, grid, *args, **kwargs: grids.append(grid),
        )
    assert grids == [(3,), (0,), (2,)]


@skipUnlessBackends(["triton"])
def test_flattened_launch_uses_host_shape_for_each_axis(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def copy(x):
        matrix = x.view(-1, x.size(-1))
        for row, col in hl.tile(matrix.size()):
            matrix[row, col] = matrix[row, col] + 1

    x = torch.zeros(3, 5, 7)
    with cpu_render_target():
        source = cpu_bind(copy, (x,)).to_code(
            helion.Config(block_sizes=[16, 16], flatten_loops=[True], pid_type="flat")
        )
    path = tmp_path / "flattened.py"
    path.write_text(source)
    run = load_generated_module(path).copy
    grids = []
    for tensor in (x, torch.zeros(5, 7, 11)):
        run(tensor, _launcher=lambda fn, grid, *op, **kw: grids.append(grid))
    assert grids == [(1,), (2,)]


@skipUnlessBackends(["triton"])
def test_flat_grid_extent_can_shadow_max(tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def write(out, max):  # noqa: A002 - exercise a legal argument name
        for i in hl.grid(max):
            out[i] = i

    out = torch.zeros(8)
    with cpu_render_target():
        source = cpu_bind(write, (out, 4)).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "extent.py"
    path.write_text(source)
    run = load_generated_module(path).write
    grids = []
    for extent in (4, -2):
        run(out, extent, _launcher=lambda fn, grid, *op, **kw: grids.append(grid))
    assert grids == [(4,), (0,)]


@skipUnlessBackends(["triton"])
def test_shifted_tile_masks_padded_indices_past_int32(tmp_path):
    @helion.kernel(backend="triton", static_shapes=True)
    def write(out):
        for tile in hl.tile(2**31 - 6, 2**31 - 1, block_size=4):
            # All eight addresses are valid, even for a wrapped lane.
            # Only the five live indices should write.
            out[(tile.index.to(torch.int64) - (2**31 - 6)) % 8] = 1

    out = torch.zeros(8, dtype=torch.int32)
    with cpu_render_target():
        source = cpu_bind(write, (out,)).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "padded.py"
    path.write_text(source)
    load_generated_module(path).write(
        out,
        _launcher=interpret_triton_kernel,
    )
    assert out.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    ("backend", "bound_source"),
    [
        ("triton", "loaded"),
        ("triton", "mixed"),
        ("triton", "host"),
        ("tileir", "loaded"),
    ],
)
def test_nested_tile_keeps_four_writes_above_uint32(
    tmp_path, monkeypatch, backend, bound_source
):
    if backend == "tileir":
        monkeypatch.setenv("ENABLE_TILE", "1")

    @helion.kernel(backend=backend, static_shapes=False, index_dtype=torch.int32)
    def mark(out, bounds, lo, hi, source: hl.constexpr):
        for row in hl.grid(out.size(0)):
            begin = bounds[0] if source == "loaded" else lo
            end = hi if source == "host" else bounds[1]
            for tile in hl.tile(begin, end, block_size=4):
                out[row, 1 + (tile.index & 3)] = row + 1

    bounds = torch.tensor([2**32, 2**32 + 4], dtype=torch.int64)
    out = torch.full((2, 6), -73, dtype=torch.int32)
    args = (out, bounds, 2**32, 2**32 + 4, bound_source)
    with cpu_render_target():
        source = cpu_bind(mark, args).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "nested_tile.py"
    path.write_text(source)
    load_generated_module(path).mark(*args, _launcher=interpret_triton_kernel)
    assert out.tolist() == [[-73, 1, 1, 1, 1, -73], [-73, 2, 2, 2, 2, -73]]


@skipUnlessBackends(["cute"])
def test_strided_grid_leaves_thread_axis_zero_for_free_lanes(tmp_path):
    @helion.kernel(backend="cute", static_shapes=True)
    def write(out):
        for row in hl.grid(0, 4, 2):
            out[row, hl.arange(4)] = 1

    out = torch.zeros(10, 4)
    with cpu_render_target():
        source = cpu_bind(write, (out,)).to_code(
            helion.Config(block_sizes=[], pid_type="flat")
        )
    path = tmp_path / "strided.py"
    path.write_text(source)
    launches = []
    load_generated_module(path).write(
        out, _launcher=lambda fn, grid, *args, block: launches.append((grid, block))
    )
    assert launches == [((2,), (4, 1, 1))]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("bounds", ["host", "literal_start", "device"])
def test_nd_mask_uses_relative_offsets_for_a_shifted_dynamic_tile(tmp_path, bounds):
    @helion.kernel(backend="triton", static_shapes=False)
    def fill(out, origin, stop, literal: hl.constexpr):
        start = 2**63 - 6 if literal else origin
        for row, col in hl.tile([start, 0], [stop, 2]):
            out[(row.index.to(torch.int64) - (2**63 - 6)) % 8, col] = 1

    out = torch.zeros(8, 2, dtype=torch.int32)
    origin, stop = 2**63 - 6, 2**63 - 1
    device = bounds == "device"
    args = (
        out,
        torch.tensor(origin) if device else origin,
        torch.tensor(stop) if device else stop,
        bounds == "literal_start",
    )
    cfg = helion.Config(
        block_sizes=[4, 2], pid_type="persistent_blocked" if device else "flat"
    )
    with cpu_render_target():
        source = cpu_bind(fill, args).to_code(cfg)
    path = tmp_path / "shifted_nd.py"
    path.write_text(source)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        load_generated_module(path).fill(*args, _launcher=interpret_triton_kernel)
    assert out.tolist() == [[1, 1]] * 5 + [[0, 0]] * 3


@skipUnlessBackends(["triton"])
def test_flattened_explicit_block_sizes_fill_each_cartesian_element(tmp_path):
    @helion.kernel(backend="triton", static_shapes=True)
    def fill(out):
        for row, col in hl.tile([8, 8], block_size=[2, 2]):
            out[row, col] = 1

    out = torch.zeros(8, 8, dtype=torch.int32)
    with cpu_render_target():
        src = cpu_bind(fill, (out,)).to_code(
            helion.Config(block_sizes=[], flatten_loops=[True])
        )
    path = tmp_path / "flat_tiles.py"
    path.write_text(src)
    load_generated_module(path).fill(out, _launcher=interpret_triton_kernel)
    torch.testing.assert_close(out, torch.ones_like(out))


@skipUnlessBackends(["cute"])
def test_flattened_tile_rejects_padded_index_past_int64():
    @helion.kernel(backend="cute", static_shapes=True)
    def write(out):
        for tile in hl.tile(2**63 - 6, 2**63 - 1):
            out[(tile.index - (2**63 - 6)) % 8] = 1

    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.InvalidConfig, match="coordinate arithmetic exceeds Int64"
        ),
    ):
        cpu_bind(write, (torch.zeros(8),)).to_code(
            helion.Config(block_sizes=[4], num_threads=[2])
        )


@skipUnlessBackends(["cute"])
def test_empty_single_lane_grid_launches_no_blocks(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(out, n):
        for tile in hl.tile(n):
            out[tile] = 1

    out = torch.zeros(512)
    with cpu_render_target():
        source = cpu_bind(write, (out, 128)).to_code(
            helion.Config(block_sizes=[128], num_threads=[32], pid_type="flat")
        )
    path = tmp_path / "single_axis.py"
    path.write_text(source)
    run = load_generated_module(path).write
    grids = []
    for extent in (128, -256, 129):
        run(out, extent, _launcher=lambda fn, grid, *op, **kw: grids.append(grid))
    assert grids == [(1,), (0,), (2,)]


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_live_tensor_origin_runs_the_reached_coordinates(tmp_path, dtype):
    @helion.kernel(backend="triton", static_shapes=False)
    def copy(ValueError, lo, hi, stride):  # noqa: A002 - exercise the generated scope
        for i in hl.grid(lo, hi, stride):
            ValueError[i] = i

    out = torch.full((8,), -73)
    args = (out, torch.tensor(1, dtype=dtype), torch.tensor(6, dtype=dtype), 2)
    with cpu_render_target():
        source = cpu_bind(copy, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "tensor_origin.py"
    path.write_text(source)

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        run = load_generated_module(path).copy
        run(*args, _launcher=interpret_triton_kernel)
        assert out.tolist() == [-73, 1, -73, 3, -73, 5, -73, -73]
        out.fill_(-73)
        run(out, args[2], args[1], -2, _launcher=interpret_triton_kernel)
        assert out.tolist() == [-73, -73, 2, -73, 4, -73, 6, -73]

        with pytest.raises(ValueError, match="step|stride|arg 3"):
            run(
                out,
                args[1],
                args[2],
                0,
                _launcher=lambda *a, **kw: pytest.fail("launched with zero stride"),
            )


@skipUnlessBackends(["cute"])
def test_flat_host_grid_counts_elements_once_for_runtime_tile_width(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def copy(x, out, width):
        for t in hl.tile(out.size(0), block_size=width):
            out[t] = x[t]

    x = torch.arange(16)
    with cpu_render_target():
        source = cpu_bind(copy, (x, x.clone(), 4)).to_code(
            helion.Config(block_sizes=[])
        )
    path = tmp_path / "runtime_tile.py"
    path.write_text(source)
    grids = []
    run = load_generated_module(path).copy
    for size, width in ((16, 4), (13, 4), (13, 2)):
        run(
            x[:size],
            x[:size],
            width,
            _launcher=lambda fn, grid, *op, **kw: grids.append(grid),
        )
    assert grids == [(4,), (4,), (7,)]


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("step", [2, -3])
def test_persistent_grid_with_wide_literal_and_int32_limit(step):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(out, limit, stride: hl.constexpr):
        for i in hl.grid(2**63 - 1, limit, stride):
            out[i % out.size(0)] = i

    with cpu_render_target():
        source = cpu_bind(
            write, (torch.zeros(8), torch.tensor(-2, dtype=torch.int32), step)
        ).to_code(helion.Config(block_sizes=[], pid_type="persistent_blocked"))
    expression = _persistent_count(source)
    for limit in (-2, -(2**31), 2**31 - 1):
        scope = {
            "limit": [np.int32(limit)],
            "cutlass": SimpleNamespace(Int64=np.int64, Uint64=np.uint64),
        }
        with np.errstate(over="ignore"):
            assert eval(expression, scope) == len(range(2**63 - 1, limit, step))


@skipUnlessBackends(["cute"])
def test_last_tiles_exclusive_end_fits_its_coordinate_type():
    with cpu_render_target():
        source = cpu_bind(
            _crossing_tile_end, (torch.zeros(1, dtype=torch.int64),)
        ).to_code(helion.Config(block_sizes=[]))
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    # Replay the emitted coordinates of the final lane of the last tile using
    # the selected widths. Widening only the final store is too late.
    scope = {
        "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (0, 0, 0), thread_idx=lambda: (3, 0, 0)
            )
        ),
    }
    assignments = [n for n in [*tree.body, *fn.body] if isinstance(n, ast.Assign)]
    with np.errstate(over="ignore"):
        exec(compile(ast.Module(assignments, []), "<tile end>", "exec"), scope)
    assert int(scope["tile_end"]) == 2**31


@skipUnlessBackends(["cute"])
def test_empty_flattened_lane_grid_has_no_static_zero_divisor(tmp_path):
    x = torch.empty(64, 0)
    with cpu_render_target():
        source = cpu_bind(_empty_flattened_tiles, (x, x.clone())).to_code(
            helion.Config(block_sizes=[], flatten_loops=[True])
        )
    path = tmp_path / "empty_lanes.py"
    path.write_text(source)
    grids = []
    load_generated_module(path)._empty_flattened_tiles(
        x, x, _launcher=lambda kernel, grid, *args, **kwargs: grids.append(grid)
    )
    assert grids == [(0,)]
    assert not any(
        isinstance(n, ast.BinOp)
        and isinstance(n.op, (ast.FloorDiv, ast.Mod))
        and isinstance(n.right, ast.Constant)
        and n.right.value == 0
        for n in ast.walk(ast.parse(source))
    )


@skipUnlessBackends(["triton", "cute"])
def test_one_axis_list_grid_obeys_its_live_stride(tmp_path):
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=False)
    def copy(out, lo, hi, step):
        for (i,) in hl.grid([lo], [hi], [step]):
            out[i] = i

    out = torch.full((10,), -73)
    with cpu_render_target():
        source = cpu_bind(copy, (out, 0, 8, 2)).to_code(helion.Config(block_sizes=[]))
    path = tmp_path / "listed.py"
    path.write_text(source)
    run = load_generated_module(path).copy
    launches = []

    def launch(fn, grid, *args, **kwargs):
        launches.append((grid, kwargs.get("block")))
        if matchesBackends(["triton"]):
            interpret_triton_kernel(fn, grid, *args)

    for lo, hi, step in ((0, 8, 2), (8, 0, -3), (1, 8, 40)):
        run(out, lo, hi, step, _launcher=launch)
        if backend == "cute":
            assert launches[-1] == ((len(range(lo, hi, step)),), (1, 1, 1))
        else:
            expected = torch.full_like(out, -73)
            for i in range(lo, hi, step):
                expected[i] = i
            torch.testing.assert_close(out, expected)
            out.fill_(-73)


@skipUnlessBackends(["triton"])
def test_uint64_root_coordinates_stay_unsigned_until_a_bounded_use(
    tmp_path, monkeypatch
):
    from helion._testing.triton_codegen import interpret_triton_selection

    @helion.kernel(backend="triton", static_shapes=False)
    def crossing(lo, hi, out, loaded: hl.constexpr):
        start = lo if loaded else 2**63 - 1
        for i in hl.grid(start, hi, 2):
            out[i % 4] = i // 2

    @helion.kernel(backend="triton", static_shapes=False)
    def bounded(lo, out):
        for i in hl.grid(lo, 2):
            out[i] = (i - 1) // 3

    @helion.kernel(backend="triton", static_shapes=False)
    def arithmetic(lo, hi, out, reduced: hl.constexpr):
        for i in hl.grid(lo, hi):
            value = i % 7 if reduced else i
            out[i % 4] = (value - 4) // 3

    config = helion.Config(block_sizes=[], pid_type="persistent_blocked")
    limit = 2**63 - 1
    lo = torch.tensor(limit, dtype=torch.uint64)
    hi = torch.tensor(limit + 3, dtype=torch.uint64)
    zero = torch.tensor(0, dtype=torch.uint64)
    two = torch.tensor(2, dtype=torch.uint64)
    out = torch.full((4,), -73, dtype=torch.int64)
    monkeypatch.setattr(helion.runtime, "get_num_sm", lambda device: 1)
    cases = [
        (crossing, (lo, hi, out, False), [-73, 2**62, -73, limit // 2]),
        (crossing, (lo, hi, out, True), [-73, 2**62, -73, limit // 2]),
        (bounded, (zero, out), [-1, 0, -73, -73]),
        (arithmetic, (zero, two, out, True), [-2, -1, -73, -73]),
    ]
    for index, (kernel, args, expected) in enumerate(cases):
        with cpu_render_target():
            source = cpu_bind(kernel, args).to_code(config)
        out.fill_(-73)
        interpret_triton_selection(
            source, args, tmp_path / f"grid{index}.py", monkeypatch
        )
        assert out.tolist() == expected
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.InvalidConfig, match="unbounded UInt64"),
    ):
        cpu_bind(arithmetic, (zero, two, out, False)).to_code(config)


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("loaded", [False, True])
def test_unsigned_exclusive_tile_edge_is_narrowed_after_division(tmp_path, loaded):
    backend = _get_backend()

    @helion.kernel(backend=backend, static_shapes=False)
    def write(out, lo, hi, loaded: hl.constexpr):
        stop = hi if loaded else 2**63
        for tile in hl.tile(lo, stop, block_size=4):
            out[tile.index % 8] = tile.end // 2

    stop, start = 2**63, 2**63 - 5
    bound = torch.tensor(start, dtype=torch.uint64)
    hi = torch.tensor(stop, dtype=torch.uint64)
    out = torch.full((8,), -1, dtype=torch.int64)
    with cpu_render_target():
        compiled = cpu_bind(write, (out, bound, hi, loaded))
        config = helion.Config(block_sizes=[], pid_type="persistent_blocked")
        if backend == "tileir":
            with pytest.raises(helion.exc.InvalidConfig, match="unbounded UInt64"):
                compiled.to_code(config)
            return
        source = compiled.to_code(config)
    if matchesBackends(["triton"]):
        path = tmp_path / "exclusive_edge.py"
        path.write_text(source)
        with patch.object(helion.runtime, "get_num_sm", return_value=1):
            load_generated_module(path).write(
                out, bound, hi, loaded, _launcher=interpret_triton_kernel
            )
    else:

        class SignedCarrier(np.int64):
            def ir_value(self):
                return self

        # CuTe constructs an unsigned high-half literal from its signed IR bits.
        with np.errstate(over="ignore"):
            for lane in range(4):
                simulate_independent_fragment(
                    source,
                    {
                        "lo": [np.int64(start)],
                        "hi": [np.uint64(stop).view(np.int64)],
                        "_NUM_SM": 1,
                    },
                    {"out": out},
                    1,
                    thread=(lane, 0, 0),
                    int32=np.int32,
                    int64=SignedCarrier,
                    uint64=np.uint64,
                )
    expected = torch.full_like(out, -1)
    for i in range(start, stop):
        expected[i % 8] = min(start + (1 + (i - start) // 4) * 4, stop) // 2
    torch.testing.assert_close(out, expected)


@skipUnlessBackends(["triton"])
def test_high_tile_mean_uses_bounded_extent_and_checks_only_reached_tiles(tmp_path):
    from triton.runtime.errors import InterpreterError

    @helion.kernel(backend="triton", static_shapes=False)
    def mean(x, out, lo, hi):
        for tile in hl.tile(lo, hi, block_size=4):
            i = tile.index % 8
            out[i] = x[i].mean()

    x = torch.arange(8, dtype=torch.float32) + 1
    out = torch.full_like(x, -1)

    def bound(n):
        return torch.tensor(n, dtype=torch.uint64)

    args = x, out, bound(2**63 - 5), bound(2**63)
    with cpu_render_target():
        source = cpu_bind(mean, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "high_mean.py"
    path.write_text(source)
    module = load_generated_module(path)
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        module.mean(*args, _launcher=interpret_triton_kernel)
        torch.testing.assert_close(
            out, torch.tensor([-1.0, -1.0, -1.0, 5.5, 5.5, 5.5, 5.5, 8.0])
        )
        module.mean(
            x,
            out,
            bound(2**63 + 2),
            bound(2**63 + 1),
            _launcher=interpret_triton_kernel,
        )
        with (
            patch.object(
                module.helion_triton_helpers,
                "device_error",
                side_effect=RuntimeError("unsupported coordinates"),
            ) as fail,
            pytest.raises(InterpreterError, match="unsupported coordinates"),
        ):
            module.mean(
                x,
                out,
                bound(2**63),
                bound(2**63 + 1),
                _launcher=interpret_triton_kernel,
            )
        fail.assert_called_once_with("tiled root coordinates exceed signed Int64")


@skipUnlessBackends(["cute"])
def test_persistent_uint64_root_preserves_its_coordinate_through_the_body():
    @helion.kernel(backend="cute", static_shapes=False)
    def from_literal(hi, out):
        for i in hl.grid(2**63 - 3, hi, 2):
            out[i % 4] = i // 2

    @helion.kernel(backend="cute", static_shapes=False)
    def from_tensors(lo, hi, out, step: hl.constexpr):
        for i in hl.grid(lo, hi, step):
            out[i % 4] = i // 2

    m = 2**63 - 1
    for kernel, begin, end, step in (
        (from_literal, m - 2, m + 1, 2),
        (from_tensors, m, m + 3, 2),
        (from_tensors, m + 2, m - 1, -2),
    ):
        bounds = {"hi": end} if kernel is from_literal else {"lo": begin, "hi": end}
        tensors = [torch.tensor(n, dtype=torch.uint64) for n in bounds.values()]
        out = torch.full((4,), -73)
        args = (*tensors, out) if kernel is from_literal else (*tensors, out, step)
        with cpu_render_target():
            source = cpu_bind(kernel, args).to_code(
                helion.Config(block_sizes=[], pid_type="persistent_blocked")
            )
        # UInt64 tensor storage is a signed 64-bit carrier on CuTe. Replay the
        # actual count, persistent loop, index arithmetic and protected stores.
        registers = {key: [np.uint64(n).view(np.int64)] for key, n in bounds.items()}
        with np.errstate(over="ignore"):
            for hi in (end, begin):
                registers["hi"] = [np.uint64(hi).view(np.int64)]
                expected = torch.full_like(out, -73)
                for i in range(begin, hi, step):
                    expected[i % 4] = i // 2
                out.fill_(-73)
                simulate_independent_fragment(
                    source,
                    {**registers, "_NUM_SM": 1},
                    {"out": out},
                    1,
                    int32=np.int32,
                    int64=np.int64,
                    uint64=np.uint64,
                )
                torch.testing.assert_close(out, expected)

    # A genuinely mixed signed/unsigned domain may span more than 64 bits.
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.InvalidConfig, match="known nonnegative"),
    ):
        cpu_bind(from_tensors, (torch.tensor(-1), tensors[-1], out, 2)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )


@skipUnlessBackends(["cute"])
def test_uint64_grid_keeps_negative_intermediate_results_signed():
    @helion.kernel(backend="cute", static_shapes=False)
    def bounded(lo, out):
        for i in hl.grid(lo, 2):
            out[i % 4] = (i - 1) // 3

    @helion.kernel(backend="cute", static_shapes=False)
    def reduced(lo, hi, out):
        for i in hl.grid(lo, hi):
            out[i % 4] = (i % 7 - 4) // 3

    @helion.kernel(backend="cute", static_shapes=False)
    def unbounded(lo, hi, out):
        for i in hl.grid(lo, hi):
            out[i % 4] = (i - 1) // 3

    config = helion.Config(block_sizes=[], pid_type="persistent_blocked")
    lo = torch.tensor(0, dtype=torch.uint64)
    hi = torch.tensor(2, dtype=torch.uint64)
    out = torch.full((4,), -73)
    with cpu_render_target():
        bounded_src = cpu_bind(bounded, (lo, out)).to_code(config)
        reduced_src = cpu_bind(reduced, (lo, hi, out)).to_code(config)
        with pytest.raises(
            helion.exc.InvalidConfig, match="unbounded UInt64 grid coordinate"
        ):
            cpu_bind(unbounded, (lo, hi, out)).to_code(config)

    for source, stops, expected in (
        (bounded_src, {}, [-1, 0, -73, -73]),
        (reduced_src, {"hi": [np.int64(2)]}, [-2, -1, -73, -73]),
    ):
        out.fill_(-73)
        simulate_independent_fragment(
            source,
            {"lo": [np.int64(0)], **stops, "_NUM_SM": 1},
            {"out": out},
            1,
            int32=np.int32,
            int64=np.int64,
            uint64=np.uint64,
        )
        assert out.tolist() == expected


@skipUnlessBackends(["cute"])
def test_persistent_root_evaluates_host_range_operands_once(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def from_hosts(lo, hi, out):
        for _ in hl.grid(lo, hi):
            out[0] = 1

    @helion.kernel(backend="cute", static_shapes=False)
    def from_tensors(lo, hi, n, out):
        for i in hl.grid(lo, hi, (n * 2**62) // 3):
            out[i % 8] = i

    out = torch.full((8,), -73)
    tensors = (torch.tensor(1), torch.tensor(6))
    for kernel, traced, cases in (
        (from_hosts, (2, 6, out), [((2**63 - 1, -(2**63), out), 0), ((2, 6, out), 4)]),
        (
            from_tensors,
            (*tensors, 4, out),
            [((*tensors, 4, out), 1), ((*tensors, -4, out), 0)],
        ),
    ):
        with cpu_render_target():
            source = cpu_bind(kernel, traced).to_code(
                helion.Config(block_sizes=[], pid_type="persistent_blocked")
            )
        path = tmp_path / (kernel.fn.__name__ + ".py")
        path.write_text(source)
        module = load_generated_module(path)
        run = getattr(module, kernel.fn.__name__)
        count = _persistent_count(source)
        for args, expected in cases:
            captured = []

            def capture(device, grid, *ops, records=captured, **kw):
                records.append((device, ops))

            with patch.object(helion.runtime, "get_num_sm", return_value=1):
                run(*args, _launcher=capture)
            ((device, ops),) = captured
            scope = dict(
                vars(module), cutlass=SimpleNamespace(Int64=np.int64, Uint64=np.uint64)
            )
            for name, value in zip(
                inspect.signature(device).parameters, ops, strict=True
            ):
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    scope[name] = [np.int64(value.item())]
                elif isinstance(value, int):
                    scope[name] = np.int64(value)
            with np.errstate(over="ignore", divide="raise", invalid="raise"):
                assert eval(count, scope) == expected


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("case", ["end", "begin", "shadow"])
def test_compound_root_bounds_use_the_python_value_in_the_body(case, tmp_path):
    @helion.kernel(backend="triton", static_shapes=False)
    def from_end(out, n):
        for i in hl.grid((n * 2**62) // (2**62 + 1)):
            out[i] = i + 1

    @helion.kernel(backend="triton", static_shapes=False)
    def from_begin(out, n):
        for i in hl.grid((n * 2**62) // (2**62 + 1), 5):
            out[i] = i + 1

    @helion.kernel(backend="triton", static_shapes=False)
    def shadow(out, len):  # noqa: A002 - a kernel argument may shadow a builtin
        for i in hl.grid(1, len):
            out[i] = i + 1

    kernel, visited = {
        "end": (from_end, range(3)),
        "begin": (from_begin, range(3, 5)),
        "shadow": (shadow, range(1, 4)),
    }[case]
    out = torch.full((8,), -73)
    with cpu_render_target():
        source = cpu_bind(kernel, (out, 4)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / (kernel.fn.__name__ + ".py")
    path.write_text(source)
    module = load_generated_module(path)

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        getattr(module, kernel.fn.__name__)(out, 4, _launcher=interpret_triton_kernel)
    expected = torch.full_like(out, -73)
    for index in visited:
        expected[index] = index + 1
    torch.testing.assert_close(out, expected)


@skipUnlessBackends(["cute"])
def test_empty_cartesian_grid_never_narrows_an_unreachable_axis(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(lo, hi, rows, out):
        for i, j in hl.grid([lo, 0], [hi, rows]):
            out[(i + j) % 8] = i

    out = torch.full((8,), -73)
    with cpu_render_target():
        source = cpu_bind(write, (2, 6, 2, out)).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "cartesian.py"
    path.write_text(source)
    module = load_generated_module(path)

    def capture(fn, grid, *args, **kwargs):
        scope = dict(
            vars(module), cutlass=SimpleNamespace(Int32=np.int32, Int64=np.int64)
        )
        for name, value in zip(inspect.signature(fn).parameters, args, strict=True):
            if isinstance(value, int):
                scope[name] = np.int64(value)
        assert eval(_persistent_count(source), scope) == 0

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        module.write(-(2**63), 2**63 - 1, 0, out, _launcher=capture)
        with pytest.raises(OverflowError, match="grid axis length"):
            module.write(-(2**63), 2**63 - 1, 1, out, _launcher=capture)


@skipUnlessBackends(["cute"])
def test_persistent_lane_loop_checks_its_raw_extent_before_launch(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(lo, hi, out):
        for t in hl.tile(lo, hi):
            out[t.index % 8] = t.index

    out = torch.zeros(8, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(write, (3, 22, out)).to_code(
            helion.Config(
                block_sizes=[128], num_threads=[32], pid_type="persistent_blocked"
            )
        )
    path = tmp_path / "lane_extent.py"
    path.write_text(source)
    run = load_generated_module(path).write
    calls = []
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        run(-(2**63) + 1, 0, out, _launcher=lambda *a, **kw: calls.append(a))
        with pytest.raises(OverflowError, match="grid argument"):
            run(-(2**63), 0, out, _launcher=lambda *a, **kw: calls.append(a))
    (args,) = calls
    assert 2**63 - 1 in [v for v in args if type(v) is int]


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("shifted", [False, True])
def test_flattened_lane_loop_preflights_signed_bounds(tmp_path, shifted):
    @helion.kernel(backend="cute", static_shapes=False)
    def edges(n, out, shifted: hl.constexpr):
        start, stop = (n, n + 5) if shifted else (0, n)
        for t in hl.tile(start, stop):
            out[t.index % 8] = t.end // 2

    out = torch.zeros(8, dtype=torch.int64)
    with cpu_render_target():
        source = cpu_bind(edges, (7, out, shifted)).to_code(
            helion.Config(block_sizes=[128], num_threads=[32])
        )
    path = tmp_path / "lane_edges.py"
    path.write_text(source)
    run = load_generated_module(path).edges
    calls = []
    n = 2**63 - 5 if shifted else 2**63
    run(n - 1, out, shifted, _launcher=lambda *a, **kw: calls.append(True))
    with pytest.raises(OverflowError, match="runtime root grid bound"):
        run(n, out, shifted, _launcher=lambda *a, **kw: calls.append(True))
    assert calls == [True]


@skipUnlessBackends(["cute"])
def test_flattened_lane_loop_uses_the_host_start(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def shifted(out, n):
        for row in hl.tile((n * 2**62) // (2**62 + 1), 131):
            out[row] = row.index + 1

    out = torch.full((135,), -73)
    with cpu_render_target():
        source = cpu_bind(shifted, (out, 4)).to_code(
            helion.Config(block_sizes=[128], num_threads=[32])
        )
    path = tmp_path / "shifted_flat.py"
    path.write_text(source)
    run = load_generated_module(path).shifted

    captured = []

    def capture(fn, grid, *args, block):
        scalars = {
            name: np.int64(value)
            for name, value in zip(inspect.signature(fn).parameters, args, strict=True)
            if isinstance(value, int)
        }
        assert block == (32, 1, 1)
        with np.errstate(over="ignore"):
            captured.append(
                generated_output_address_coverage(
                    source, "out", grid[0], block[0], scalar_args=scalars
                )
            )

    for n in (4, 5):
        run(out, n, _launcher=capture)
        start = (n * 2**62) // (2**62 + 1)
        assert captured.pop() == dict.fromkeys(range(start, 131), 1)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("matrix", [False, True])
def test_persistent_flattened_lane_loop_uses_the_host_extent(tmp_path, matrix):
    @helion.kernel(backend="cute", static_shapes=False)
    def write(out, n):
        for row in hl.tile(1, 128 + (n * 2**62) // (2**62 + 1)):
            out[row] = row.index

    @helion.kernel(backend="cute", static_shapes=False)
    def copy(data, out, n):
        for row, col in hl.tile([128 + (n * 2**62) // (2**62 + 1), 2]):
            out[row, col] = data[row, col]

    if matrix:
        data = torch.arange(270).reshape(135, 2)
        inputs = (data, torch.full_like(data, -73))
        kernel = copy
        config = helion.Config(
            block_sizes=[64, 2],
            num_threads=[16, 2],
            flatten_loops=[True],
            pid_type="persistent_blocked",
        )
    else:
        inputs = (torch.full((135,), -73),)
        kernel = write
        config = helion.Config(
            block_sizes=[128], num_threads=[32], pid_type="persistent_blocked"
        )
    with cpu_render_target():
        source = cpu_bind(kernel, (*inputs, 4)).to_code(config)
    path = tmp_path / "persistent_flat.py"
    path.write_text(source)
    module = load_generated_module(path)
    count = _persistent_count(source)

    def capture(fn, grid, *args, block):
        scope = {
            **vars(module),
            "cutlass": SimpleNamespace(Int64=np.int64),
            **{
                name: np.int64(value)
                for name, value in zip(
                    inspect.signature(fn).parameters, args, strict=True
                )
                if isinstance(value, int)
            },
        }
        with np.errstate(over="ignore"):
            assert eval(count, scope) == (3 if matrix else 2)

    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        for n in (4, 5):
            getattr(module, kernel.fn.__name__)(*inputs, n, _launcher=capture)


@skipUnlessBackends(["cute"])
def test_persistent_flattened_tile_clamps_each_axis_before_multiplying(tmp_path):
    @helion.kernel(backend="cute", static_shapes=False)
    def copied(data, out, height, max):  # noqa: A002 - kernel can shadow max
        for row, col in hl.tile([height - 10, max - 10]):
            out[row, col] = data[row, col]

    data = torch.arange(600).reshape(20, 30)
    out = torch.full_like(data, -73)
    with cpu_render_target():
        source = cpu_bind(copied, (data, out, 25, 20)).to_code(
            helion.Config(
                block_sizes=[64, 2],
                num_threads=[16, 2],
                flatten_loops=[True],
                pid_type="persistent_blocked",
            )
        )
    path = tmp_path / "clipped.py"
    path.write_text(source)
    module = load_generated_module(path)
    count = _persistent_count(source)

    def capture(fn, grid, *args, block):
        scope = dict(vars(module), cutlass=SimpleNamespace(Int64=np.int64))
        for name, value in zip(inspect.signature(fn).parameters, args, strict=True):
            if isinstance(value, int):
                scope[name] = np.int64(value)
        observed.append(int(eval(count, scope)))

    observed = []
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        for height, width in ((25, 20), (6, 7), (6, 12), (10, 20), (30, 30)):
            module.copied(data, out, height, width, _launcher=capture)
    assert observed == [2, 0, 0, 0, 4]
