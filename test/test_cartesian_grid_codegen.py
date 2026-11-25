from __future__ import annotations

import ast
import inspect
from types import CodeType
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion._testing import _get_backend
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
from helion._testing.cute_output_addresses import generated_output_address_coverage
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _persistent_count(source: str) -> CodeType:
    (count,) = [
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "total_pids"
    ]
    return compile(ast.Expression(count), "<persistent trips>", "eval")


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
    if backend == "triton":
        import triton.language as tl
        from triton.runtime.interpreter import InterpretedFunction

        m = load_generated_module(path)

        def launch(fn, grid, *op, **kwargs):
            InterpretedFunction(fn.fn)[grid](*op)

        with (
            patch.object(helion.runtime, "get_num_sm", return_value=1),
            patch.object(tl, "cdiv", InterpretedFunction(tl.cdiv.fn)),
        ):
            m.write(*args, _launcher=launch)
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
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_live_tensor_origin_runs_the_reached_coordinates(tmp_path, dtype):
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction

    @helion.kernel(backend="triton", static_shapes=False)
    def copy(out, lo, hi, stride):
        for i in hl.grid(lo, hi, stride):
            out[i] = i

    out = torch.full((8,), -73)
    args = (out, torch.tensor(1, dtype=dtype), torch.tensor(6, dtype=dtype), 2)
    with cpu_render_target():
        source = cpu_bind(copy, args).to_code(
            helion.Config(block_sizes=[], pid_type="persistent_blocked")
        )
    path = tmp_path / "tensor_origin.py"
    path.write_text(source)

    def launch(fn, grid, *op, **kwargs):
        InterpretedFunction(fn.fn)[grid](*op)

    with (
        patch.object(helion.runtime, "get_num_sm", return_value=1),
        patch.object(tl, "cdiv", InterpretedFunction(tl.cdiv.fn)),
    ):
        run = load_generated_module(path).copy
        run(*args, _launcher=launch)
        assert out.tolist() == [-73, 1, -73, 3, -73, 5, -73, -73]
        out.fill_(-73)
        run(out, args[2], args[1], -2, _launcher=launch)
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
        if backend == "triton":
            from triton.runtime.interpreter import InterpretedFunction

            InterpretedFunction(fn.fn)[grid](*args)

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
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction

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

    def launch(fn, grid, *args, **kwargs):
        InterpretedFunction(fn.fn)[grid](*args)

    with (
        patch.object(helion.runtime, "get_num_sm", return_value=1),
        patch.object(tl, "cdiv", InterpretedFunction(tl.cdiv.fn)),
    ):
        getattr(module, kernel.fn.__name__)(out, 4, _launcher=launch)
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
        scope = dict(vars(module))
        for name, value in zip(inspect.signature(fn).parameters, args, strict=True):
            if isinstance(value, int):
                scope[name] = np.int64(value)
        observed.append(int(eval(count, scope)))

    observed = []
    with patch.object(helion.runtime, "get_num_sm", return_value=1):
        for height, width in ((25, 20), (6, 7), (6, 12), (10, 20), (30, 30)):
            module.copied(data, out, height, width, _launcher=capture)
    assert observed == [2, 0, 0, 0, 4]
