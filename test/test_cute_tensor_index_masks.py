from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.memory_ops import _cute_stack_tensor_mask_expr
from helion._testing import skipUnlessBackends
from helion.language.memory_ops import _cute_combined_mask

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


@dataclass
class _MaskStrategy:
    mask: str | None

    def mask_var(self, block_id: int) -> str | None:
        return self.mask


def _predicate(
    *,
    jagged: bool,
    k_mask: str | None = "mask_k",
    extra: bool = False,
    stack: bool = False,
    routing: str = "ordinary",
) -> str:
    shape_env = ShapeEnv()
    axes = [shape_env.create_unbacked_symint() for _ in range(4)]
    unknown = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        index = torch.empty(
            (unknown,) if routing == "unknown" else (axes[0], axes[2]),
            dtype=torch.int64,
        )

    def block_id(size: object) -> int | None:
        if isinstance(size, torch.SymInt):
            return next(
                (i for i, axis in enumerate(axes) if size.node.expr == axis.node.expr),
                None,
            )
        return None

    # M/N/K have the same logical extent but independent symbolic provenance.
    # The unknown shape also has that extent, without belonging to any axis.
    def equal_extent(left: object, right: object) -> bool:
        if isinstance(right, torch.SymInt):
            return left == (8 if block_id(right) == 3 else 64)
        return left == right

    env = SimpleNamespace(
        get_block_id=block_id,
        known_equal=equal_extent,
        block_sizes=[
            SimpleNamespace(block_id=i, size=8 if i == 3 else 64, var=axis)
            for i, axis in enumerate(axes)
        ],
        is_jagged_tile=lambda bid: jagged and bid == 0,
        index_dtype=torch.int32,
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int32"),
    )
    state = cast(
        "CodegenState",
        SimpleNamespace(
            fx_node=None,
            ast_args=[None, [ast.Name(id="address", ctx=ast.Load())], None],
            codegen=SimpleNamespace(
                active_device_loops={
                    i: [SimpleNamespace(strategy=_MaskStrategy(mask))]
                    for i, mask in enumerate(("mask_m", "mask_n", k_mask, "mask_s"))
                },
                lift=lambda value, **kwargs: value,
            ),
            device_function=SimpleNamespace(
                tensor_size=lambda tensor, dim: SimpleNamespace(
                    name=str(tensor.size(dim))
                ),
                cute_state=SimpleNamespace(
                    matmul_operand_block_remap={2: 1} if routing == "remap" else {},
                    matmul_operand_index_override={2: "override"}
                    if routing == "override"
                    else {},
                ),
            ),
        ),
    )
    tensor = torch.empty(4096)
    user_mask = ast.Name(id="user_mask", ctx=ast.Load()) if extra else None
    with patch.object(CompileEnvironment, "current", return_value=env):
        if stack:
            result = _cute_stack_tensor_mask_expr(
                state, tensor, torch.empty(8, dtype=torch.int64), [index], user_mask
            )
        else:
            result = _cute_combined_mask(
                state,
                [index],
                user_mask,
                tensor=tensor,
                include_tensor_index_masks=False,
            )
    assert result is not None
    return result


@pytest.mark.parametrize("jagged", [False, True])
@pytest.mark.parametrize("k_mask", [None, "mask_k"])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("stack", [False, True])
def test_exact_index_axes_preserve_tail_and_existing_masks(
    jagged: bool, k_mask: str | None, extra: bool, stack: bool
) -> None:
    predicate = _predicate(jagged=jagged, k_mask=k_mask, extra=extra, stack=stack)
    for m, n, k, s, user, in_bounds in itertools.product((False, True), repeat=6):
        values = {
            "mask_m": m,
            "mask_n": n,
            "mask_k": k,
            "mask_s": s,
            "user_mask": user,
            "address": 1 if in_bounds else 4096,
            "cutlass": SimpleNamespace(Int32=int),
        }
        assert bool(eval(predicate, {}, values)) == (
            m
            and (k_mask is None or k)
            and (not extra or user)
            and (not stack or s)
            and (jagged or in_bounds)
        )


@pytest.mark.parametrize("routing", ["remap", "override", "unknown"])
def test_index_axes_follow_provenance_and_active_overrides(routing: str) -> None:
    predicate = _predicate(jagged=False, routing=routing)
    for m, n, k, in_bounds in itertools.product((False, True), repeat=4):
        values = {
            "mask_m": m,
            "mask_n": n,
            "mask_k": k,
            "address": 1 if in_bounds else 4096,
            "cutlass": SimpleNamespace(Int32=int),
        }
        expected = in_bounds
        if routing != "unknown":
            expected = expected and m and (routing != "remap" or n)
        assert bool(eval(predicate, {}, values)) == expected


@skipUnlessBackends(["cute"])
def test_flattened_jagged_load_keeps_its_k_tail_mask() -> None:
    # Fixed CPU capabilities must not fill process-wide hardware caches.
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch("helion._compat._supports_maxnreg", return_value=False),
        patch("helion._compat._supports_tensor_descriptor", return_value=False),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.autotuner.config_spec.num_compute_units", return_value=128),
    ):
        kernel = helion.kernel(
            jagged_dense_bmm.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        arguments = (
            torch.empty(17, dtype=torch.int64),
            torch.empty((319, 32)),
            torch.empty((16, 32, 32)),
            torch.empty((16, 32)),
        )
        source = _cpu_bind(kernel, arguments).to_code(
            helion.Config(
                block_sizes=[1, 32, 32, 64],
                num_threads=[0, 4, 32, 1],
                cute_vector_widths=[1] * 4,
                cute_lane_layouts=["blocked"] * 4,
            )
        )
    loads = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "jagged_data" for t in node.targets)
    ]
    assert len(loads) == 1 and isinstance(loads[0], ast.IfExp)
    predicate = compile(ast.Expression(loads[0].test), "<jagged load mask>", "eval")
    for row_valid, k_valid in itertools.product((False, True), repeat=2):
        assert bool(eval(predicate, {}, {"mask_1": row_valid, "mask_3": k_valid})) == (
            row_valid and k_valid
        )
