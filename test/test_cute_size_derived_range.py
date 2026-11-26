"""CuTe ranges whose length is computed from a reduction dim's size (CPU codegen).

CuTe runs an ``hl.arange`` over the lanes of the dim registered for its
shape, which keeps the length traced over the reduction dim's block size:
``hl.arange(row.size(1) // 2)`` for ``row = x[tile, :]`` at 129 columns, in
64-column chunks, ran 32 lanes where the row's extent gives 64.  CuTe refuses
such a length unless the block size gives it the value the extent gives, any
tile size in it taken from the config on both sides.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.program_id import FlatProgramIDs
from helion._compiler.program_id import PIDInfo
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def half_row_iota(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        n = row.size(1) // 2
        out[tile] = row.sum(-1) + hl.arange(n).to(torch.float32).sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_per_tile_row_iota(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        n = row.size(1) // row.size(0)
        out[tile] = row.sum(-1) + hl.arange(n).to(torch.float32).sum(-1)
    return out


@pytest.mark.parametrize(
    "kernel",
    [
        # Half of 128 columns, in one 128-column chunk: 64.
        pytest.param(half_row_iota, id="half"),
        # 128 columns over the 16-row tile: 8, the tile's size from the config
        # on the extent's side too.
        pytest.param(row_per_tile_row_iota, id="per_tile_row"),
    ],
)
def test_a_length_the_block_gives_as_the_extent_does_renders(
    kernel: helion.Kernel,
) -> None:
    bound = cpu_bind(kernel, (torch.empty(64, 128),))
    bound.to_code(helion.Config(block_sizes=[16], reduction_loops=[128]))


@pytest.mark.parametrize(
    "kernel,columns",
    [
        # Half a 64-column chunk is 32; half of 129 columns is 64.
        pytest.param(half_row_iota, 129, id="half"),
        # A 64-column chunk over the 16-row tile is 4; 128 columns give 8.
        pytest.param(row_per_tile_row_iota, 128, id="per_tile_row"),
    ],
)
def test_a_length_the_block_gives_otherwise_is_refused(
    kernel: helion.Kernel, columns: int
) -> None:
    bound = cpu_bind(kernel, (torch.empty(64, columns),))
    with pytest.raises(
        helion.exc.BackendUnsupported, match="computed from a reduction dim's size"
    ):
        bound.to_code(helion.Config(block_sizes=[16], reduction_loops=[64]))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _wide_phase_copy(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel(), block_size=1):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(x.numel(), block_size=2):
        out[tile] = tmp[tile] * 2
    return out


@pytest.mark.parametrize("pid_type", ["persistent_blocked", "persistent_interleaved"])
@pytest.mark.parametrize("extent", [2**31 + 7, 3 * 2**29])
def test_wide_persistent_pid_preserves_the_range_bound_and_induction_type(
    pid_type: str, extent: int
) -> None:
    # The first root alone has more program IDs than Int32 can represent.
    # The expanded input represents the extent with one real element.
    x = torch.empty(1).expand(extent)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_wide_phase_copy, (x,))
        assert bound.env.index_dtype == (
            torch.int64 if extent >= 2**31 else torch.int32
        )
        config = helion.Config(
            pid_type=pid_type,
            cross_loop_pipeline="barrier",
            num_sm_multiplier=1,
        )
        tree = ast.parse(bound.to_code(config))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith(("program_iteration", "phase_iteration"))
    ]
    assert len(loops) == 2
    for loop in loops:
        # The SDK's dynamic multibound range inserts a bare Int32 zero;
        # here both the trip domain and the computed absolute ID stay wide.
        assert isinstance(loop.iter, ast.Call) and len(loop.iter.args) == 1
        assert "cutlass.Int64" in ast.unparse(loop.iter.args[0])
        assert isinstance(loop.body[0], ast.Assign)
        assert loop.target.id in ast.unparse(loop.body[0].value)


def test_blocked_last_worker_near_the_signed_limit() -> None:
    extent = 1_431_655_764
    total = extent + (extent + 1) // 2
    assert total < 2**31 <= ((total + 147) // 148) * 148
    with torch.device("meta"):
        x = torch.empty(extent)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_wide_phase_copy, (x,))
        config = helion.Config(
            pid_type="persistent_blocked",
            cross_loop_pipeline="barrier",
            num_sm_multiplier=1,
        )
        source = bound.to_code(config)
    assert "cutlass.Int64(1431655764)" in source


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _shifted_plane(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    result = torch.empty_like(x)
    for row, column in hl.tile([-1, 0], [n, m], block_size=[2, 1]):
        valid = row.index[:, None] >= 0
        value = hl.load(x, [row, column], extra_mask=valid)
        hl.store(result, [row, column], value + 1, extra_mask=valid)
    return result


@pytest.mark.parametrize("kind", ["persistent_blocked", "persistent_interleaved"])
def test_shifted_extent_and_decomposition_at_int32_max(kind: str) -> None:
    # Small genuine bind, legal Int32 dimensions. Replay the actual scheduling
    # expressions at an Int32 boundary without allocating an enormous tensor.
    with cpu_render_target(fake_device_sm_count=148):
        b = cpu_bind(_shifted_plane, (torch.empty(17, 4),))
        assert b.env.index_dtype is torch.int32
        tree = ast.parse(b.to_code(helion.Config(pid_type=kind)))
    expressions = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    # Fixed-width operations must overflow if promotion is moved outside
    # (n + 1). The second axis starts exactly one complete first axis later.
    area = {
        "n": np.int32(2**31 - 1),
        "m": np.int32(2),
        "_BLOCK_SIZE_0": np.int32(2),
        "virtual_pid": np.int64(2**30),
        "cutlass": SimpleNamespace(Int64=np.int64),
    }
    # The wrapper computes the shifted Cartesian length with Python shape
    # integers before passing it to the device's fixed-width PID arithmetic.
    host_length = compile(
        ast.Expression(expressions["grid_length"]), "<host-grid>", "eval"
    )
    area["grid_length"] = eval(
        host_length,
        {"__builtins__": {}},
        {"helion": helion, "n": int(area["n"]), "m": int(area["m"])},
    )
    assert area["grid_length"] == 2**31
    for name, expected in [
        ("total_pids", 2**31),
        ("num_blocks_0", 2**30),
        ("pid_0", 0),
        ("pid_1", 1),
    ]:
        code = compile(ast.Expression(expressions[name]), "<rendered-pid>", "eval")
        area[name] = eval(code, {"__builtins__": {}}, area)
        assert area[name] == expected


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_plane(x: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for r, c in hl.tile([limit, x.size(1)], block_size=[2, 1]):
        out[r, c] = x[r, c]
    return out


def test_grouped_device_scalar_bound_widens_before_dividing() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        b = cpu_bind(
            _bounded_plane, (torch.empty(8, 4), torch.tensor(8, dtype=torch.int32))
        )
        source = b.to_code(
            helion.Config(pid_type="persistent_interleaved", l2_grouping=8)
        )
    expressions = {
        node.targets[0].id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    scope = {
        "max": max,
        "limit": [np.int32(2**31 - 1)],
        "_BLOCK_SIZE_0": 2,
        "_BLOCK_SIZE_1": 1,
        "cutlass": SimpleNamespace(Int64=np.int64),
    }
    counts = [
        eval(
            compile(ast.Expression(expressions[name]), "<grouped-axis>", "eval"),
            {"__builtins__": {}},
            scope,
        )
        for name in ("num_pid_m", "num_pid_n")
    ]
    assert sorted(counts) == [4, 2**30]


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _shape_bounded_plane(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row, col in hl.tile([x.size(0), x.size(1)], block_size=[2, 1]):
        out[row, col] = x[row, col]
    return out


@pytest.mark.parametrize("group", [1, 8])
def test_flat_two_axis_decoder_narrows_only_after_wide_division(group: int) -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shape_bounded_plane, (torch.empty(8, 4),))
        source = bound.to_code(helion.Config(pid_type="flat", l2_grouping=group))
    exprs = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    if group == 1:
        assert exprs["num_blocks_0"].startswith("cutlass.Int32(")
        assert "cutlass.Int64(x_size_0)" in exprs["num_blocks_0"]
    else:
        assert "cutlass.Int64(x_size_0)" in exprs["num_pid_m"]
        assert all(exprs[f"pid_{i}"].startswith("cutlass.Int32(") for i in (0, 1))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _single_count(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for i in hl.tile(x.numel(), block_size=1):
        result[i] = x[i]
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_independent_counts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for i in hl.tile(x.numel() // 2, block_size=1):
        a[i] = x[i]
    for j in hl.tile((x.numel() + 1) // 2, block_size=1):
        b[j] = x[j]
    return a, b


@pytest.mark.parametrize("kernel", [_single_count, _two_independent_counts])
def test_last_interleaved_increment_preserves_the_int32_limit(
    kernel: helion.Kernel,
) -> None:
    with torch.device("meta"):
        x = torch.empty(2**31 - 1)
    with cpu_render_target(fake_device_sm_count=148):
        b = cpu_bind(kernel, (x,))
        assert b.env.index_dtype is torch.int32
        src = b.to_code(
            helion.Config(pid_type="persistent_interleaved", num_sm_multiplier=1)
        )
    loops = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("program_iteration")
    ]
    # Inspect the actually generated induction, not just a large literal or
    # the memory address width. A 148-CTA worker's final old step would
    # advance 2,147,483,552 to 2,147,483,700.
    assert len(loops) == 1
    loop = loops[0]
    assert isinstance(loop.iter, ast.Call) and len(loop.iter.args) == 1
    assert isinstance(loop.body[0], ast.Assign)
    assert "cutlass.Int64(_NUM_SM)" in ast.unparse(loop.body[0].value)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _flat_independent_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    for i in hl.tile(x.numel(), block_size=16):
        left[i] = x[i] + 1
    for i in hl.tile(x.numel(), block_size=32):
        right[i] = x[i] - 2
    return left, right


def test_flat_symbolic_dispatch_stays_in_its_original_index_type() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        b = cpu_bind(_flat_independent_rows, (torch.empty(317),))
        source = b.to_code(helion.Config(pid_type="flat"))
    statements = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("pid_shared")
    ]
    assert len(statements) >= 2
    assert ast.unparse(statements[0].value) == "cutlass.Int32(cute.arch.block_idx()[0])"
    for node in statements[1:]:
        # Compute each extent's ceiling division safely first; the result
        # re-enters the original Int32 flat PID at the subtraction.
        assert isinstance(node.value, ast.BinOp)
        assert ast.unparse(node.value.right).startswith("cutlass.Int32(")


@pytest.mark.cuda
def test_native_flat_and_persistent_dynamic_phases() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from helion._testing import DEVICE

    x = torch.arange(317, device=DEVICE, dtype=torch.float32)
    b = _flat_independent_rows.bind((x,))
    for kind in ("flat", "persistent_interleaved", "persistent_blocked"):
        f = b.compile_config(helion.Config(pid_type=kind))
        a, b_ = f(x)
        torch.testing.assert_close(a, x + 1, rtol=0, atol=0)
        torch.testing.assert_close(b_, x - 2, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("extent", "block", "pack", "count"),
    [
        ("end", "1", 1, 2**31 - 1),
        ("7", "1", 1, 7),
        ("(end + 1) // 2", "1", 1, 2**30),
        ("end", "8", 1, 2**28),
        ("end - begin", "1", 2, 2**30),
    ],
)
def test_flat_divisor_at_the_int32_limit(
    extent: str, block: str, pack: int, count: int
) -> None:
    # The inputs fit Int32, as does each final divisor, but the intermediate
    # sum/difference in the last three cases does not.
    grid = FlatProgramIDs(
        pid_info=[PIDInfo("i", block, extent, 0, pack), PIDInfo("j", "1", "1", 1)]
    )
    state = Mock()
    state.device_function.new_var.side_effect = lambda name: name
    with CompileEnvironment(
        torch.device("cpu"), helion.Settings(backend="cute"), index_dtype=torch.int32
    ):
        divisor = grid._decompose_pid_to_statements("pid", state)[0]
    assert isinstance(divisor, ast.Assign)
    if extent in ("end", "7") and block == "1":
        assert ast.unparse(divisor.value) == f"cutlass.Int32({extent})"
    namespace = {
        "end": np.int32(2**31 - 1),
        "begin": np.int32(-1),
        "cutlass": SimpleNamespace(Int64=np.int64, Int32=np.int32),
    }
    code = compile(ast.fix_missing_locations(ast.Expression(divisor.value)), "", "eval")
    value = eval(code, {"__builtins__": {}}, namespace)
    assert isinstance(value, np.int32)
    assert int(value) == count
