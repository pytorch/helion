from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.metal.backend import _lane_floor
from helion._compiler.metal.lane_fold import unroll_lane_folds
from helion._compiler.metal.mpp_matmul import names_read
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@pytest.fixture(autouse=True)
def _no_cuda_work() -> Iterator[None]:
    with forbid_cuda_work():
        yield


@pytest.mark.parametrize("block_size", [16, 32])
def test_configured_free_reduction_reaches_free_axis_refusal(block_size: int) -> None:
    @helion.kernel(backend="metal", static_shapes=True)
    def free_reduce(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.zeros([m], dtype=x.dtype, device=x.device)
        bn = hl.register_block_size(16, 32)
        for g in hl.grid(m):
            cols = hl.arange(bn)
            out[g] = hl.load(x, [g, cols], extra_mask=cols < n).sum(0)
        return out

    bound = cpu_bind(free_reduce, (torch.randn(64, 64),))
    with pytest.raises(exc.BackendUnsupported, match="kernel has a reduction"):
        bound.to_code(helion.Config(block_sizes=[block_size]))


def test_runtime_persistent_reduction_remains_refused() -> None:
    @helion.kernel(backend="metal", static_shapes=False)
    def row_sum(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
        for rows in hl.tile(x.size(0)):
            out[rows] = x[rows, :].sum(-1)
        return out

    bound = cpu_bind(row_sum, (torch.randn(8, 256),))
    with pytest.raises(exc.BackendUnsupported, match="dimension of runtime extent"):
        bound.to_code(helion.Config(block_sizes=[1], reduction_loops=[None]))


def test_specialized_persistent_reduction_still_renders() -> None:
    @helion.kernel(backend="metal", static_shapes=False)
    def row_sum(x: torch.Tensor) -> torch.Tensor:
        hl.specialize(x.size(1))
        out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
        for rows in hl.tile(x.size(0)):
            out[rows] = x[rows, :].sum(-1)
        return out

    bound = cpu_bind(row_sum, (torch.randn(8, 256),))
    code = bound.to_code(
        helion.Config(block_sizes=[1], num_warps=4, reduction_loops=[None])
    )
    assert "_block_dims=(128, 1, 1)" in code
    assert "in range(2)" in code


@helion.kernel(backend="metal", static_shapes=True)
def _configured_broadcast_sum(x: torch.Tensor) -> torch.Tensor:
    bn = hl.register_block_size(16, 32768)
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for g in hl.grid(x.size(0)):
        values = hl.zeros([bn], dtype=x.dtype) + x[g, 0]
        out[g] = values.sum(0)
    return out


@pytest.mark.parametrize(
    "block_size, floor, reserved", [(16, 1, 2), (256, 4, 32), (16384, 256, 256)]
)
def test_configured_reduction_budgets_use_selected_extent(
    block_size: int, floor: int, reserved: int
) -> None:
    bound = cpu_bind(_configured_broadcast_sum, (torch.ones(2, 1),))
    config = helion.Config(block_sizes=[block_size], num_warps=4)
    bound.config_spec.normalize(config)
    with bound.env, bound.host_function:
        assert _lane_floor(config, bound.host_function.device_ir) == floor
        backend = bound.env.backend
        assert isinstance(backend, MetalBackend)
        assert (
            backend._reserved_reduction_threads(
                config, combined=True, device_ir=bound.host_function.device_ir
            )
            == reserved
        )


@pytest.mark.parametrize("block_size", [16, 256, 16384])
def test_configured_broadcast_reduction_folds_lanes(block_size: int) -> None:
    x = torch.tensor([[1.25], [-2.5]])
    bound = cpu_bind(_configured_broadcast_sum, (x,))
    code = bound.to_code(helion.Config(block_sizes=[block_size], num_warps=4))
    torch.testing.assert_close(_broadcast_model(code, x), x[:, 0] * block_size)
    assert ("_metal_lane_fold(" in code) == (block_size > 128)


@dataclass
class _Pointer:
    tensor: torch.Tensor
    offset: int | torch.Tensor = 0

    def __add__(self, offset: int | torch.Tensor) -> _Pointer:
        return _Pointer(self.tensor, self.offset + offset)


def _broadcast_model(code: str, x: torch.Tensor) -> torch.Tensor:
    """Evaluate this scalar-broadcast device IR on CPU, including its lane jam.

    Loads/stores keep their emitted indices. The collective sums the modeled
    threads; it never knows the authored reduction extent, so a missing lane
    fold produces the same truncated multiple as the generated Metal code.
    """
    tree = ast.parse(code)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    )
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    block_dims = ast.literal_eval(
        next(kw.value for kw in launch.keywords if kw.arg == "_block_dims")
    )
    threads = block_dims[0]
    device.decorator_list = []
    device.body = unroll_lane_folds(device.body, names_read)

    def load(pointer: _Pointer, mask: object) -> torch.Tensor:
        return pointer.tensor.flatten()[pointer.offset]

    def store(pointer: _Pointer, value: torch.Tensor, mask: object) -> None:
        pointer.tensor.flatten()[pointer.offset] = value

    def group_sum(
        site: object,
        scratch: object,
        value: torch.Tensor,
        tid: object,
        axis: int,
        span: int,
        tx: int,
        ty: int,
        tz: int,
    ) -> torch.Tensor:
        assert axis == 0 and span == threads
        return value.expand(threads).sum()

    namespace = {
        "tl": SimpleNamespace(load=load, store=store),
        "helion_red": SimpleNamespace(group_sum=group_sum),
        "static_cast": lambda dtype, value: torch.as_tensor(value, dtype=torch.float32),
        "tid": (torch.arange(threads), 0, 0),
        "_BLOCK_SIZE_1": 1,
        "_TG_SIZE_X": threads,
        "_TG_SIZE_Y": 1,
        "_TG_SIZE_Z": 1,
        "_red_site": None,
        "_red_scratch": None,
    }
    module = ast.fix_missing_locations(ast.Module(body=[device], type_ignores=[]))
    exec(compile(module, "<metal-broadcast-model>", "exec"), namespace)
    out = torch.empty(x.size(0))
    for row in range(x.size(0)):
        namespace["tgid"] = (row, 0, 0)
        namespace[device.name](_Pointer(x), _Pointer(out))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_lane_argmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        out[tile_m] = torch.argmax(x[tile_m, tile_n], -1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_lane_sum_in_branch(x: torch.Tensor, flag: int) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        if flag > 0:
            out[tile_m] = x[tile_m, tile_n].sum(-1)
    return out


@pytest.mark.parametrize(
    ("kernel", "args", "op"),
    [
        (grid_lane_argmax, (), "argmax"),
        (grid_lane_sum_in_branch, (1,), "sum"),
    ],
    ids=["argmax", "sum_in_branch"],
)
def test_grid_lane_loop_reduction_rejected(
    kernel: helion.Kernel, args: tuple[object, ...], op: str
) -> None:
    """A root tile's lane loop is split around its reductions
    (``GridLaneSplit``) except where no split applies: argmax has no lane
    fold, and a reduction in a branch is not a top-level statement of the
    lane loop."""
    bound = cpu_bind(kernel, (torch.randn(4, 64), *args))
    with pytest.raises(
        exc.BackendUnsupported,
        match=re.escape(f"{op} reduction over an axis strided by a tile lane loop"),
    ):
        bound.to_code(helion.Config(block_sizes=[64], num_threads=[16]))


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_where_positive(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A K loop's row sums under an if on a per-element predicate."""
    m, k = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for _g in hl.grid(1):
        for tile_m in hl.tile(m):
            acc = hl.zeros([tile_m], dtype=torch.float32)
            if c[tile_m] > 0:
                for tile_k in hl.tile(k):
                    acc = acc + x[tile_m, tile_k].sum(1)
            out[tile_m] = acc
    return out


def test_collective_under_a_per_element_predicate_rejected() -> None:
    """A reduction whose block spans threads combines through a threadgroup
    collective, whose barriers every thread must reach; under an if on a
    predicate over M with M's rows on threads, some threads skip them, so
    it is refused.  With K on one thread it combines in registers, and with
    M's rows on one thread's lanes every thread tests the same element, so
    both lower."""
    bound = cpu_bind(row_sums_where_positive, (torch.randn(16, 64), torch.randn(16)))
    with pytest.raises(
        exc.BackendUnsupported,
        match=re.escape("a reduction across threads (a threadgroup collective) under"),
    ):
        bound.to_code(helion.Config(block_sizes=[16, 64], num_threads=[16, 64]))
    for num_threads in ([16, 1], [1, 64]):
        bound.to_code(helion.Config(block_sizes=[16, 64], num_threads=num_threads))


@helion.kernel(backend="metal")
def row_sum_where(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A row sum over a ':' dim under an if on a per-row predicate."""
    m, _ = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        value = hl.zeros([tile_m], dtype=torch.float32)
        if c[tile_m] > 0:
            value = x[tile_m, :].sum(1)
        out[tile_m] = value
    return out


@helion.kernel(backend="metal")
def row_sum_if(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """A row sum over a ':' dim under an if on a scalar predicate."""
    m, _ = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        value = hl.zeros([tile_m], dtype=torch.float32)
        if flag[0] > 0:
            value = x[tile_m, :].sum(1)
        out[tile_m] = value
    return out


def test_rolled_reduction_under_an_if_rejected() -> None:
    """A rolled reduction's loop wraps the if and its result is combined
    after the loop, where the if's body can't read it: refused at every
    config, on a per-row predicate (with M on threads, on one thread's
    lanes, at one row) and on a scalar one, which the persistent form
    lowers."""
    match = re.escape("a rolled reduction (reduction_loops) under an if")
    x = torch.randn(16, 256)
    per_row = cpu_bind(row_sum_where, (x, torch.randn(16)))
    scalar = cpu_bind(row_sum_if, (x, torch.randn(1)))
    for bound, config in (
        (per_row, helion.Config(block_sizes=[16], reduction_loops=[64])),
        (
            per_row,
            helion.Config(block_sizes=[16], reduction_loops=[64], num_threads=[1]),
        ),
        (per_row, helion.Config(block_sizes=[1], reduction_loops=[64])),
        (scalar, helion.Config(block_sizes=[16], reduction_loops=[64])),
    ):
        with pytest.raises(exc.BackendUnsupported, match=match):
            bound.to_code(config)
    scalar.to_code(helion.Config(block_sizes=[16], reduction_loops=[None]))


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_while(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """``n[0]`` times each row's sum, in a device while on a scalar."""
    out = torch.zeros([x.size(0)], dtype=x.dtype, device=x.device)
    for t in hl.tile(x.size(0)):
        c = hl.load(n, [0])
        total = hl.zeros([t], dtype=x.dtype)
        i = torch.zeros([], device=x.device, dtype=torch.int32)
        while i < c:
            total = total + x[t, :].sum(-1)
            i = i + 1
        out[t] = total
    return out


def test_device_while_refused() -> None:
    """Metal emits no ``while``: a device while is refused when its device
    IR is built, at bind time, before any config (the fallback's and the
    autotuner's included).  The refusal reads no config, so one bind covers
    every layout; it used to reach the MSL emitter's refusal at the first
    launch, or an MPP planner refusal of its own."""
    args = (torch.randn(8, 64), torch.tensor([3], dtype=torch.int32))
    with pytest.raises(exc.BackendUnsupported, match="a device while loop"):
        cpu_bind(row_sums_while, args)
