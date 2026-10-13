from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_fuser_occurrences import _args
from .test_cute_fuser_occurrences import _state_reduction
import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

if TYPE_CHECKING:
    from collections.abc import Callable


def _config(mode: str, vectorized: bool = False) -> helion.Config:
    if vectorized:
        return helion.Config(
            block_sizes=[16],
            num_threads=[16, 16],
            loop_orders=[[1, 2, 0]],
            pid_type="flat",
            cute_vector_widths=[8, 1, 1, 1],
            cute_reduction_reloads=[mode],
        )
    return helion.Config(
        block_sizes=[16],
        num_threads=[1, 32],
        loop_orders=[[1, 2, 0]],
        pid_type="flat",
        cute_reduction_reloads=[mode],
    )


def _check_cache(
    source: str, control: str, vectorized: bool = False, fast_math: bool = False
) -> None:
    declarations = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_computed_cache_")
    ]
    assert declarations
    for declaration in declarations:
        assert isinstance(declaration.value, ast.Call)
        assert ast.unparse(declaration.value.func) == "cute.make_rmem_tensor"
        assert ast.unparse(declaration.value.args[1]) == "cutlass.Float32"
    assert "_computed_cache_" not in control
    assert source.count("state.iterator") == 2  # One original read and one write.
    # Fast vector controls factor the first scalar sweep into two FP32 lanes;
    # the second sweep still has its original vector reload and store.
    assert control.count("state.iterator") == (4 if fast_math and vectorized else 3)
    assert source.count("cute.math.exp2(") < control.count("cute.math.exp2(")
    assert source.count("warp_reduction_sum(") == control.count("warp_reduction_sum(")
    if vectorized:
        assert source.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == (
            4 if fast_math else 3
        )
        assert "_cute_store_u16_vec(state.iterator" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,state_dim", [(32, 128), (17, 128)])
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("fast_math", [False, True])
def test_actual_computed_cache_codegen(
    dtype: torch.dtype, width: int, state_dim: int, vectorized: bool, fast_math: bool
) -> None:
    arguments = _args(dtype, width, state_dim)
    with _cpu_codegen():
        kernel = helion.kernel(
            _state_reduction.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            fast_math=fast_math,
        )
        bound = kernel._bind_isolated(arguments)
        source = bound.to_code(_config("computed", vectorized))
        control = bound.to_code(_config("gmem", vectorized))
    _check_cache(source, control, vectorized, fast_math)


@dataclass
class _Guarded:
    storage: torch.Tensor
    tensor: torch.Tensor
    active: torch.Tensor

    def check(self) -> None:
        assert torch.all(self.storage[~self.active] == -29.0).item()


def _guarded_copy(source: torch.Tensor) -> _Guarded:
    # Preserve zero strides of broadcast inputs as well as arbitrary positive
    # strides; copying the physical span avoids writing an expanded view.
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(source.shape, source.stride(), strict=True)
    )
    storage = torch.full((span + 256,), -29.0, dtype=source.dtype, device=source.device)
    tensor = storage.as_strided(source.shape, source.stride(), 128)
    active = torch.zeros_like(storage, dtype=torch.bool)
    active.as_strided(source.shape, source.stride(), 128).fill_(True)
    storage[128 : 128 + span].copy_(source.as_strided((span,), (1,)))
    return _Guarded(storage, tensor, active)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,state_dim", [(32, 128), (17, 128)])
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("fast_math", [False, True])
def test_computed_state_cache_preserves_guards_and_changing_graph_inputs(
    dtype: torch.dtype, width: int, state_dim: int, vectorized: bool, fast_math: bool
) -> None:
    torch.manual_seed(8462)
    original = _args(dtype, width, state_dim, "cuda")
    protected = tuple(_guarded_copy(tensor) for tensor in original)
    arguments = tuple(item.tensor for item in protected)
    state, x, dt, a, b, c, d, bias, slots, output = arguments
    kernel = helion.kernel(
        _state_reduction.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=fast_math,
    )
    bound = kernel._bind_isolated(arguments)
    source = bound.to_code(_config("computed", vectorized))
    _check_cache(
        source, bound.to_code(_config("gmem", vectorized)), vectorized, fast_math
    )
    compiled = bound.compile_config(_config("computed", vectorized))
    group_map = torch.tensor([0, 0, 1, 1], device="cuda")

    def invoke() -> torch.Tensor:
        return compiled(*arguments)

    def validate(call: Callable[[], torch.Tensor], generation: int) -> None:
        choices = ((2, 0), (0, 2), (4, 1), (1, 4))[generation]
        slots.copy_(torch.tensor(choices, device="cuda"))
        x.copy_(original[1] * (generation + 1))
        if fast_math:
            # Include underflow, the softplus threshold, and overflow in the
            # unselected exp branch. The FP64 oracle retains the same tolerance.
            raw = torch.tensor(
                [-110.0, -20.0, -1.0, 0.0, 19.9, 20.0, 20.1, 90.0],
                device="cuda",
            ).roll(generation)
            dt[:, :, 0].copy_(raw.view(2, 4) - bias[:, 0])
        before = state.clone()
        readonly = [item.storage.clone() for item in protected[1:-1]]
        # Use independent FP64 math. Output consumes the unrounded updated
        # state; only the final state store is rounded to the input dtype.
        delta = torch.nn.functional.softplus(dt.double() + bias.double())
        current = before[slots].double() * torch.exp(delta[..., None] * a.double())
        current = (
            current
            + (delta * x.double())[..., None] * b.double()[:, group_map, None, :]
        )
        expected_output = (current * c.double()[:, group_map, None, :]).sum(-1)
        expected_output = expected_output + d.double() * x.double()
        expected_state = before.clone()
        expected_state[slots] = current.to(dtype)
        output.fill_(float("nan"))
        call()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, expected_output.to(dtype), atol=0.02, rtol=0.01
        )
        torch.testing.assert_close(state, expected_state, atol=0.02, rtol=0.01)
        unselected = [slot for slot in range(5) if slot not in choices]
        torch.testing.assert_close(
            state[unselected], before[unselected], atol=0, rtol=0
        )
        for item, expected in zip(protected[1:-1], readonly, strict=True):
            torch.testing.assert_close(item.storage, expected, atol=0, rtol=0)
        for item in protected:
            item.check()

    validate(invoke, 0)
    validate(invoke, 1)
    replay = _make_cudagraph_replay(invoke)
    validate(replay, 2)
    validate(replay, 3)
