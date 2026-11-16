from __future__ import annotations

import ast
from dataclasses import dataclass
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
