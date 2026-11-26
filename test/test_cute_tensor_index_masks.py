from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.emitters.memory_access import _cute_combined_mask
from helion._compiler.cute.memory_ops import _cute_stack_tensor_mask_expr
from helion._compiler.cute.state import CuteCodegenState
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
from helion._testing.cute_thread_simulator import execute_pointwise_thread_program
import helion.language as hl

pytestmark = pytest.mark.cpu_only

jagged_dense_bmm = import_example("jagged_dense_bmm").jagged_dense_bmm

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
                backend_state=CuteCodegenState(),
                lift=lambda value, **kwargs: value,
            ),
            device_function=SimpleNamespace(
                tensor_size=lambda tensor, dim: SimpleNamespace(
                    name=str(tensor.size(dim))
                ),
                cute_state=SimpleNamespace(
                    tcgen05=SimpleNamespace(
                        matmul_operand_block_remap={2: 1} if routing == "remap" else {},
                        matmul_operand_index_override={2: "override"}
                        if routing == "override"
                        else {},
                    )
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


def test_flattened_jagged_load_keeps_its_k_tail_mask() -> None:
    # Fixed CPU capabilities must not fill process-wide hardware caches.
    with (
        mock_cuda_unavailable(),
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
        source = cpu_bind(kernel, arguments).to_code(
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


CUDA_DEVICE = "cuda"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_jagged_reduction_tail_ignores_backing_storage() -> None:
    torch.manual_seed(20260912)
    batch, rows, k, n, bk = 16, 319, 32, 32, 64
    lengths = torch.tensor([19, 0, 40] + [20] * 13, dtype=torch.int64)
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=CUDA_DEVICE)
    offsets[1:].copy_(lengths.cumsum(0))
    # Poison only storage outside the contiguous logical view. Keeping the
    # backing allocation alive removes allocator history from this test.
    backing = torch.empty(rows * k + bk, device=CUDA_DEVICE)
    a = backing[: rows * k].view(rows, k)
    a.normal_()
    b = torch.randn((batch, k, n), device=CUDA_DEVICE)
    bias = torch.randn((batch, n), device=CUDA_DEVICE)
    arguments = (offsets, a, b, bias)
    original = tuple(value.clone() for value in arguments)
    assert a.is_contiguous() and a.stride() == (k, 1)
    assert a.data_ptr() == backing.data_ptr()

    left, right = a.double(), b.double()
    pieces = []
    begin = 0
    for group, count in enumerate(lengths.tolist()):
        end = begin + count
        pieces.append((left[begin:end] @ right[group]).float() + bias[group])
        begin = end
    expected = torch.cat(pieces)
    assert torch.isfinite(expected).all()

    kernel = helion.kernel(
        jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="ieee",
    )
    bound = kernel.bind(arguments)
    config = helion.Config(
        block_sizes=[1, 32, n, bk],
        num_threads=[0, 4, 32, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
    )
    source = bound.to_code(config)
    assert "dot_acc" in source
    assert "cute.gemm(" not in source
    assert "cvt_f32_tf32" not in source
    bound.set_config(config)

    outputs = []
    for padding in (0.0, float("nan"), 0.0):
        backing[rows * k :].fill_(padding)
        output = bound(*arguments)
        outputs.append(output)
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
        for before, after in zip(original, arguments, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)


@skipIfRefEager("checks compile-time rejection")
@pytest.mark.parametrize("value", [1.75, 1 + 2j, True])
@pytest.mark.parametrize("local", [False, True])
def test_direct_scalar_index_requires_integer_dtype(value: object, local: bool) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, index, out, local: hl.constexpr):
        for col in hl.tile(out.size(0), block_size=1):
            if local:
                p = index
                out[col] = hl.load(x, [p, col])
            else:
                out[col] = hl.load(x, [index, col])

    index = torch.tensor(value)
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.InvalidIndexingType, match="must have an integer dtype"
        ),
    ):
        cpu_bind(
            gather, (torch.tensor([[10.0], [20.0]]), index, torch.empty(1), local)
        ).to_code(helion.Config(block_sizes=[]))


@skipIfRefEager("executes generated CuTe")
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_scalar_index_reads_its_buffer_after_an_alias_store(dtype: torch.dtype) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def gather(x, plane, alias, out):
        for col in hl.tile(1, block_size=1):
            p = plane
            before = hl.load(x, [plane, col])
            alias[0] = 1
            out[col] = before + hl.load(x, [p, col])

    x = torch.tensor([[10.0], [20.0]])
    plane = torch.tensor(0, dtype=dtype)
    alias = plane.view(1)
    out = torch.empty(1)
    with cpu_render_target():
        code = cpu_bind(gather, (x, plane, alias, out)).to_code(
            helion.Config(block_sizes=[])
        )
    # The launcher marshals a scalar as a one-element view. NumPy and the
    # writable alias share the same CPU storage in the generated program.
    unsigned = dtype == torch.uint64
    inputs = [0, 1, 2, 2**32, 2**32 + 1]
    inputs.extend([2**63, 2**64 - 1] if unsigned else [-1, -(2**32), -(2**63)])
    for initial in inputs:
        alias.numpy()[0] = initial
        simulate_independent_fragment(
            code,
            {"x": x, "plane": alias.numpy(), "alias": alias},
            {"out": out},
            1,
            int32=np.int32,
            int64=np.int64,
            uint64=np.uint64,
        )
        assert out.item() == 20 + (x[initial, 0].item() if 0 <= initial < 2 else 0)


@skipIfRefEager("executes generated CuTe")
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_direct_scalar_bounds_guard_stores_and_atomics(dtype: torch.dtype) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def update(x, plane):
        for _ in hl.grid(1):
            hl.store(x, [plane], 3.0)
            hl.atomic_add(x, [plane], 4.0)

    x = torch.empty(2)
    plane = torch.tensor(0, dtype=dtype)
    with cpu_render_target():
        code = cpu_bind(update, (x, plane)).to_code(helion.Config(block_sizes=[]))
    scalar = plane.view(1).numpy()
    for index in [
        0,
        1,
        2,
        2**32,
        2**32 + 1,
        2**64 - 1 if dtype == torch.uint64 else -1,
    ]:
        scalar[0] = index
        x.fill_(11.0)
        simulate_independent_fragment(
            code,
            {"plane": scalar},
            {"x": x},
            1,
            int32=np.int32,
            int64=np.int64,
            uint64=np.uint64,
        )
        assert x.tolist() == [7.0 if i == index else 11.0 for i in range(2)]


@skipIfRefEager("executes generated CuTe")
def test_repeated_rank_two_index_has_one_pair_of_output_coordinates() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def diagonal(x, ids, out):
        for row, col in hl.tile(out.shape):
            idx = ids[row, col]
            out[row, col] = x[idx, idx]
        return out

    ids = torch.tensor([[6, 3, 1, 7], [0, 5, 2, 4]])
    x = torch.arange(64.0).reshape(8, 8)
    args = (x, ids, torch.empty(2, 4))
    with cpu_render_target():
        code = cpu_bind(diagonal, args).to_code(helion.Config(block_sizes=[2, 4]))
    result, _ = execute_pointwise_thread_program(code, args)
    torch.testing.assert_close(result, x[ids, ids])

    @helion.kernel(backend="cute", static_shapes=True)
    def split_diagonal(x, ids, out):
        for row in hl.tile(ids.size(0)):
            half = ids.size(1) // 2
            idx, _ = hl.split(ids[row, :].reshape([row, 2, half]).permute(0, 2, 1))
            out[row, :] = x[idx, idx]

    # A split index and a regular result use different lane coordinates for
    # columns. The conflict is on the second axis of the rank-two gather.
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match="at position 1 with different lane coordinates",
        ),
    ):
        cpu_bind(
            split_diagonal,
            (x, torch.zeros(8, 16, dtype=torch.int64), torch.empty(8, 8)),
        ).to_code(helion.Config(block_sizes=[8]))
