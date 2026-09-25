from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language import scan_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo

pytestmark = skipUnlessBackends(["cute"])


# Matmul.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 16]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        result = hl.dot(first.to(a.dtype), v[bi, qq, col])
        out[bi, row, col] = result.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _modified_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    residual_scale: torch.Tensor,
) -> torch.Tensor:
    batches, length, heads, n = v.shape
    groups, k = a.shape[2:]
    out = torch.empty_like(v)
    for task, row, col in hl.tile([batches * heads, length, n], block_size=[1, 16, 16]):
        bi = task.begin // heads
        head = task.begin % heads
        group = head // (heads // groups)
        kk = hl.arange(k)
        qq = hl.arange(length)
        first = hl.dot(a[bi, row, group, kk], b[bi, qq, group, kk].T)
        weights = first * torch.exp(
            decay[bi, head, row][:, None] - decay[bi, head, qq][None, :]
        )
        weights *= dt[bi, qq, head][None, :].float()
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        result = hl.dot(weights.to(v.dtype), v[bi, qq, head, col])
        result += v[bi, row, head, col].float() * residual_scale[head].float()
        out[bi, row, head, col] = result.to(v.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three_dots(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 8]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        result = hl.dot(torch.relu(first).to(a.dtype), v[bi, qq, col])
        past = hl.dot(a[bi, row, kk], state[bi, kk, col])
        result += past * torch.exp(decay[bi, row])[:, None]
        out[bi, row, col] = result.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_operand(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    batches, k, m = a.shape
    n = b.shape[2]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 32, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        weighted = (b[bi, kk, col].float() * torch.exp(scale[bi, kk])[:, None]).to(
            a.dtype
        )
        out[bi, row, col] = hl.dot(a[bi, kk, row].T, weighted).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _offset_scaled_operand(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    batches, length, m = a.shape
    n = b.shape[2]
    chunks, k = scale.shape[1:]
    out = torch.empty((batches, chunks, m, n), device=a.device, dtype=a.dtype)
    for task, row, col in hl.tile([batches * chunks, m, n], block_size=[1, None, None]):
        bi = task.begin // chunks
        ci = task.begin % chunks
        kk = hl.arange(k)
        time = ci * k + kk
        weighted = (
            b[bi, time, col].float() * torch.exp(scale[bi, ci, kk])[:, None]
        ).to(a.dtype)
        out[bi, ci, row, col] = hl.dot(a[bi, time, row].T, weighted).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_modified_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    batches, length, k = a.shape
    n = v.shape[2]
    out = torch.empty_like(v)
    for batch, row, col in hl.tile([batches, length, n], block_size=[1, 16, 16]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(length)
        decay = hl.cumsum(delta[bi, qq].float(), dim=0)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        out[bi, row, col] = hl.dot(weights.to(v.dtype), v[bi, qq, col]).to(v.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _colliding_names(
    chain_thread: torch.Tensor, chain_0_a: torch.Tensor, chain_store: torch.Tensor
) -> torch.Tensor:
    m, k = chain_thread.shape
    q, n = chain_store.shape
    out = torch.empty((m, n), device=chain_thread.device, dtype=chain_thread.dtype)
    for row, col in hl.tile([m, n], block_size=[16, 16]):
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(chain_thread[row, kk], chain_0_a[qq, kk].T)
        out[row, col] = hl.dot(first.to(chain_thread.dtype), chain_store[qq, col]).to(
            chain_thread.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _native_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, m, groups, k = a.shape
    n = b.shape[1]
    out = torch.empty((batches, groups, m, n), device=a.device, dtype=torch.float32)
    for bg, row, col in hl.tile([batches * groups, m, n], block_size=[1, 16, 16]):
        bi = bg.begin // groups
        group = bg.begin % groups
        kk = hl.arange(k)
        out[bi, group, row, col] = hl.dot(
            a[bi, row, group, kk], b[bi, col, group, kk].T
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conventional_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for row, col in hl.tile([m, n], block_size=[64, 64]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for kk in hl.tile(k, block_size=64):
            acc = hl.dot(a[row, kk], b[kk, col], acc)
        out[row, col] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _noop_squeeze_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, axis: int
) -> torch.Tensor:
    axis = hl.specialize(axis)
    m, k = a.shape
    q, n = v.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for row, col in hl.tile([m, n], block_size=[16, 16]):
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(a[row, kk].squeeze(axis), b[qq, kk].T)
        out[row, col] = hl.dot(first.to(a.dtype), v[qq, col]).to(a.dtype)
    return out


def _matmul_compile(kernel: Any, args: tuple[torch.Tensor, ...], dots: int = 2):
    bound = kernel.bind(args)
    config = bound.env.config_spec.default_config()
    code = bound.to_triton_code(config)
    for index in range(dots):
        assert f"chain_{index}_mma" in code
    return bound.compile_config(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 32, 64, 48, 32), (1, 35, 80, 33, 24)])
def test_plain_chained_matmul(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    batches, m, k, q, n = shape
    torch.manual_seed(123)
    args = tuple(
        torch.randn(size, dtype=dtype, device=DEVICE) * 0.2
        for size in ((batches, m, k), (batches, q, k), (batches, q, n))
    )
    frozen = tuple(arg.clone() for arg in args)
    fn = _matmul_compile(_plain_chain, args)
    result = fn(*args)
    intermediate = (args[0].float() @ args[1].float().transpose(-1, -2)).to(dtype)
    reference = (intermediate.float() @ args[2].float()).to(dtype)
    torch.testing.assert_close(result, reference, atol=0.01, rtol=0.01)
    torch.testing.assert_close(fn(*args), result, atol=0, rtol=0)
    for actual, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "length,heads,groups,k,n", [(64, 4, 2, 64, 32), (128, 8, 8, 128, 64)]
)
@pytest.mark.parametrize(
    "schedule", ["coalesced", "cp_async_register", "cp_async_register_reuse"]
)
def test_chained_matmul_causal_broadcast(
    length: int, heads: int, groups: int, k: int, n: int, schedule: str
) -> None:
    torch.manual_seed(456)
    dtype = torch.bfloat16
    a = torch.randn((2, length, groups, k), device=DEVICE, dtype=dtype) / k**0.5
    b = torch.randn_like(a)
    v = torch.randn((2, length, heads, n), device=DEVICE, dtype=dtype)
    dt = (0.01 + 0.09 * torch.rand((2, length, heads), device=DEVICE)).to(dtype)
    decay = -dt.float().cumsum(1).transpose(1, 2).contiguous()
    scale = torch.randn(heads, device=DEVICE, dtype=dtype)
    args = (a, b, v, decay, dt, scale)
    frozen = tuple(arg.clone() for arg in args)
    bound = _modified_chain.bind(args)
    config = helion.Config(cute_chained_mma_schedule=schedule)
    code = bound.to_code(config)
    if schedule in ("cp_async_register", "cp_async_register_reuse"):
        assert "chain_1_a_bridge_coords" in code
        assert "chain_0_c_ptr" not in code
    fn = bound.compile_config(config)
    result = fn(*args)
    scores = a.double().permute(0, 2, 1, 3) @ b.double().permute(0, 2, 3, 1)
    scores = scores.repeat_interleave(heads // groups, dim=1)
    weights = scores * torch.exp(
        decay.double()[:, :, :, None] - decay.double()[:, :, None, :]
    )
    weights *= dt.double().transpose(1, 2)[:, :, None, :]
    weights = torch.tril(weights)
    reference = weights @ v.double().transpose(1, 2)
    reference = (
        reference.transpose(1, 2) + v.double() * scale.double()[None, None, :, None]
    )
    torch.testing.assert_close(result.double(), reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(fn(*args), result, atol=0, rtol=0)
    for actual, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_three_contraction_dag() -> None:
    torch.manual_seed(789)
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.2
        for shape in ((2, 32, 64), (2, 48, 64), (2, 48, 24), (2, 64, 24))
    )
    decay = -torch.rand((2, 32), device=DEVICE)
    fn = _matmul_compile(_three_dots, (*args, decay), dots=3)
    out = fn(*args, decay)
    first = torch.relu(args[0].float() @ args[1].float().transpose(-1, -2)).to(
        torch.bfloat16
    )
    reference = first.float() @ args[2].float()
    reference += (args[0].float() @ args[3].float()) * decay.exp()[:, :, None]
    torch.testing.assert_close(out, reference.to(out.dtype), atol=0.01, rtol=0.01)


def test_single_contraction_computed_operand() -> None:
    torch.manual_seed(987)
    a = torch.randn((2, 80, 48), device=DEVICE, dtype=torch.bfloat16) * 0.2
    b = torch.randn((2, 80, 40), device=DEVICE, dtype=torch.bfloat16) * 0.2
    scale = -torch.rand((2, 80), device=DEVICE)
    fn = _matmul_compile(_scaled_operand, (a, b, scale), dots=1)
    out = fn(a, b, scale)
    weighted = (b.float() * scale.exp()[:, :, None]).to(b.dtype)
    reference = (a.float().transpose(-1, -2) @ weighted.float()).to(a.dtype)
    torch.testing.assert_close(out, reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_computed_operand_scalar_offset_and_search(warps: int) -> None:
    torch.manual_seed(654)
    a = torch.randn((2, 128, 32), device=DEVICE, dtype=torch.bfloat16) * 0.2
    b = torch.randn((2, 128, 24), device=DEVICE, dtype=torch.bfloat16) * 0.2
    scale = -torch.rand((2, 2, 64), device=DEVICE)
    bound = _offset_scaled_operand.bind((a, b, scale))
    spec = bound.env.config_spec
    assert spec.cute_chained_matmul_search_enabled
    assert not spec.cute_tcgen05_search_enabled
    assert {dim.name for dim in spec.iter_search_dimensions()} == {
        "block_sizes",
        "num_warps",
        "cute_chained_mma_schedule",
    }
    config = helion.Config(block_sizes=[16, 32], num_warps=warps)
    code = bound.to_code(config)
    assert "chain_0_mma" in code
    fn = bound.compile_config(config)
    out = fn(a, b, scale)
    weighted = (b.reshape(2, 2, 64, 24).float() * scale.exp()[..., None]).to(b.dtype)
    reference = (
        a.reshape(2, 2, 64, 32).float().transpose(-1, -2) @ weighted.float()
    ).to(a.dtype)
    torch.testing.assert_close(out, reference, atol=0.01, rtol=0.01)
    torch.testing.assert_close(fn(a, b, scale), out, atol=0, rtol=0)


@pytest.mark.parametrize("length", [64, 128])
@pytest.mark.parametrize("schedule", ["coalesced", "cp_async_register"])
@pytest.mark.parametrize("warps", [4, 8])
def test_chained_matmul_fused_additive_scan(
    length: int, schedule: str, warps: int
) -> None:
    torch.manual_seed(246)
    a = torch.randn((2, length, 64), device=DEVICE, dtype=torch.bfloat16) * 0.1
    b = torch.randn_like(a)
    v = torch.randn((2, length, 32), device=DEVICE, dtype=torch.bfloat16)
    delta = -torch.rand((2, length), device=DEVICE, dtype=torch.bfloat16) * 0.05
    args = (a, b, v, delta)
    frozen = tuple(arg.clone() for arg in args)
    fn = _scan_modified_chain.bind(args).compile_config(
        helion.Config(cute_chained_mma_schedule=schedule, num_warps=warps)
    )
    out = fn(*args)
    decay = delta.double().cumsum(-1)
    weights = torch.tril(
        (a.double() @ b.double().transpose(-1, -2))
        * (decay[:, :, None] - decay[:, None, :]).exp()
    ).to(a.dtype)
    reference = (weights.double() @ v.double()).to(v.dtype)
    torch.testing.assert_close(out, reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(fn(*args), out, atol=0, rtol=0)
    for actual, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


def test_chained_matmul_generated_name_hygiene() -> None:
    args = tuple(
        torch.randn((32, 32), device=DEVICE, dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    )
    fn = _matmul_compile(_colliding_names, args)
    intermediate = (args[0].float() @ args[1].float().T).to(torch.bfloat16)
    reference = (intermediate.float() @ args[2].float()).to(torch.bfloat16)
    torch.testing.assert_close(fn(*args), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("groups", [1, 3])
def test_native_nt_fp32_fallback(groups: int) -> None:
    a = torch.randn((2, 35, groups, 64), device=DEVICE, dtype=torch.bfloat16) * 0.1
    b = torch.randn((2, 27, groups, 64), device=DEVICE, dtype=torch.bfloat16) * 0.1
    fn = _matmul_compile(_native_nt, (a, b), dots=1)
    out = fn(a, b)
    reference = a.float().permute(0, 2, 1, 3) @ b.float().permute(0, 2, 3, 1)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)


def test_conventional_gemm_keeps_standard_mma() -> None:
    a = torch.empty((128, 128), device=DEVICE, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    bound = _conventional_gemm.bind((a, b))
    assert not bound.env.config_spec.cute_chained_matmul_search_enabled
    code = bound.to_code(bound.env.config_spec.default_config())
    assert "chain_0_mma" not in code


@pytest.mark.parametrize(
    "schedule",
    [
        "coalesced",
        "coalesced_unrolled",
        "k_major",
        "k_major_padded",
        "cp_async",
        "cp_async_register",
    ],
)
def test_chained_matmul_memory_schedules(schedule: str) -> None:
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.1
        for shape in ((1, 35, 80), (1, 33, 80), (1, 33, 24))
    )
    bound = _plain_chain.bind(args)
    config = helion.Config(cute_chained_mma_schedule=schedule)
    fn = bound.compile_config(config)
    intermediate = (args[0].float() @ args[1].float().transpose(-1, -2)).to(
        args[0].dtype
    )
    reference = (intermediate.float() @ args[2].float()).to(args[0].dtype)
    torch.testing.assert_close(fn(*args), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("axis", [0, -2])
def test_chained_matmul_noop_squeeze(axis: int) -> None:
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.1
        for shape in ((35, 64), (48, 64), (48, 32))
    )
    bound = _noop_squeeze_chain.bind((*args, axis))
    config = bound.env.config_spec.default_config()
    assert "chain_0_mma" in bound.to_code(config)
    fn = bound.compile_config(config)
    intermediate = (args[0].float() @ args[1].float().T).to(args[0].dtype)
    reference = (intermediate.float() @ args[2].float()).to(args[0].dtype)
    torch.testing.assert_close(fn(*args, axis), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
def test_chained_matmul_cooperative_width_is_not_narrowed(warps: int) -> None:
    with FakeTensorMode():
        args = tuple(
            torch.empty(shape, device=DEVICE, dtype=torch.bfloat16)
            for shape in ((1, 32, 64), (1, 128, 64), (1, 128, 32))
        )
        bound = _plain_chain.bind(args)
        code = bound.to_code(helion.Config(num_warps=warps))
    assert f"atom_layout_mnk=(1, {warps}, 1)" in code
    assert f"atom_layout_mnk=(1, {min(warps, 2)}, 1)" in code
    tree = ast.parse(code)
    guarded_compute = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id.startswith("chain_thread")
        and ast.unparse(node.test).endswith("< 64")
    ]
    assert bool(guarded_compute) == (warps >= 4)
    for region in guarded_compute:
        assert "cute.arch.sync_threads" not in ast.unparse(region)
        assert "cute.copy(" in ast.unparse(region)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("schedule", ["cp_async", "cp_async_register"])
def test_chained_matmul_async_copy(dtype: torch.dtype, schedule: str) -> None:
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=dtype) * 0.1
        for shape in ((1, 32, 64), (1, 64, 64), (1, 64, 32))
    )
    bound = _plain_chain.bind(args)
    config = helion.Config(cute_chained_mma_schedule=schedule)
    code = bound.to_code(config)
    assert code.count("cpasync.CopyG2SOp()") == 3
    assert code.count(".toint() % 16 == 0") == 3
    assert code.count("cp_async_wait_group(0)") == 2
    fn = bound.compile_config(config)
    intermediate = (args[0].float() @ args[1].float().transpose(-1, -2)).to(dtype)
    reference = (intermediate.float() @ args[2].float()).to(dtype)
    torch.testing.assert_close(fn(*args), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("schedule", ["cp_async", "cp_async_register"])
def test_chained_matmul_async_copy_codegen(schedule: str) -> None:
    with FakeTensorMode():
        args = tuple(
            torch.empty(shape, device=DEVICE, dtype=torch.bfloat16)
            for shape in ((1, 32, 64), (1, 64, 64), (1, 64, 32))
        )
        code = _plain_chain.bind(args).to_code(
            helion.Config(cute_chained_mma_schedule=schedule)
        )
    assert code.count("cpasync.CopyG2SOp()") == 3
    assert code.count(".toint() % 16 == 0") == 3
    if schedule == "cp_async_register":
        assert "chain_0_c_ptr" not in code
        assert "chain_1_a_bridge_coords" in code


def test_chained_matmul_register_scan_codegen() -> None:
    with FakeTensorMode():
        args = tuple(
            torch.empty(shape, device=DEVICE, dtype=torch.bfloat16)
            for shape in ((1, 64, 64), (1, 64, 64), (1, 64, 32), (1, 64))
        )
        code = _scan_modified_chain.bind(args).to_code(
            helion.Config(cute_chained_mma_schedule="cp_async_register")
        )
    assert "chain_0_c_ptr" not in code
    assert "chain_1_a_bridge_coords" in code
    assert "chain_scan_0_values" in code


def test_chained_matmul_async_guards_are_eager() -> None:
    # Four-dimensional leaves need enough bounds/stride predicates to expose
    # exponential expansion in CuTe's short-circuit AST preprocessing.
    shapes = (
        (2, 64, 2, 64),
        (2, 64, 2, 64),
        (2, 64, 4, 32),
        (2, 4, 64),
        (2, 64, 4),
        (4,),
    )
    with FakeTensorMode():
        args = tuple(
            torch.empty(
                shape,
                device=DEVICE,
                dtype=torch.float32 if index == 3 else torch.bfloat16,
            )
            for index, shape in enumerate(shapes)
        )
        code = _modified_chain.bind(args).to_code(
            helion.Config(cute_chained_mma_schedule="cp_async_register")
        )
    guards = [
        node.test
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If) and ".toint()" in ast.unparse(node.test)
    ]
    assert len(guards) == 3
    for guard in guards:
        assert not any(isinstance(node, ast.BoolOp) for node in ast.walk(guard))
        assert isinstance(guard, ast.BinOp)
        assert isinstance(guard.op, ast.BitAnd)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _wide_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


def test_chained_matmul_eight_warp_codegen() -> None:
    with FakeTensorMode():
        args = tuple(
            torch.empty(shape, device=DEVICE, dtype=torch.bfloat16)
            for shape in ((1, 32, 128), (1, 128, 128), (1, 128, 64))
        )
        bound = _wide_chain.bind(args)
        config = helion.Config(
            block_sizes=[16, 64],
            num_warps=8,
            cute_chained_mma_schedule="cp_async_register",
        )
        code = bound.to_code(config)
    assert code.count("atom_layout_mnk=(1, 8, 1)") == 2
    assert code.count("cpasync.CopyG2SOp()") == 3
    assert "block=(256, 1, 1)" in code
    assert "chain_1_a_bridge_coords" in code
    assert any(
        seed.config.get("num_warps") == 8
        for seed in bound.env.config_spec.compiler_seed_configs
    )


def test_chained_matmul_eight_warp_runtime() -> None:
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.1
        for shape in ((1, 32, 128), (1, 128, 128), (1, 128, 64))
    )
    config = helion.Config(
        block_sizes=[16, 64],
        num_warps=8,
        cute_chained_mma_schedule="cp_async_register",
    )
    fn = _wide_chain.bind(args).compile_config(config)
    first = (args[0].float() @ args[1].float().transpose(-1, -2)).to(args[0].dtype)
    expected = (first.float() @ args[2].float()).to(args[0].dtype)
    torch.testing.assert_close(fn(*args), expected, atol=0.01, rtol=0.01)


def test_chained_matmul_epilogue_staged_input_codegen() -> None:
    shapes = (
        (2, 64, 2, 64),
        (2, 64, 2, 64),
        (2, 64, 4, 32),
        (2, 4, 64),
        (2, 64, 4),
        (4,),
    )
    with FakeTensorMode():
        args = tuple(
            torch.empty(
                shape, device=DEVICE, dtype=torch.float32 if i == 3 else torch.bfloat16
            )
            for i, shape in enumerate(shapes)
        )
        code = _modified_chain.bind(args).to_code(
            helion.Config(cute_chained_mma_schedule="cp_async_register_reuse")
        )
    stores = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For) and "chain_store_step" in ast.unparse(node.target)
    ]
    assert len(stores) == 1
    epilogue = ast.unparse(stores[0])
    assert "chain_1_b[" in epilogue
    assert ".load()" in epilogue  # Guarded global fallback remains available.


# Expression.

CPU_DEVICE = torch.device("cpu")

_SCHEDULES = ("coalesced", "cp_async", "cp_async_register")

_OPERATIONS = ("left_add", "right_add", "both_add", "bias", "exp")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _expression_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    a_bias: torch.Tensor,
    b_bias: torch.Tensor,
    operation: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        left = a[bi, row, kk].float()
        right = b[bi, qq, kk].float()
        if operation == "left_add" or operation == "both_add":
            left = left + 1.0
        if operation == "right_add" or operation == "both_add":
            right = right + 1.0
        if operation == "bias":
            left = left + a_bias[bi, row][:, None].float()
            right = right + b_bias[bi, qq][:, None].float()
        if operation == "exp":
            left = torch.exp(left)
            right = torch.exp(right)
        first = hl.dot(left.to(a.dtype), right.to(a.dtype).T)
        # Both operands resurrect a padded zero at the second contraction too.
        # This exercises the register bridge as well as shared-memory staging.
        second = hl.dot(
            (first + 1.0).to(a.dtype), (v[bi, qq, col].float() + 1.0).to(a.dtype)
        )
        out[bi, row, col] = second.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _negative_view_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, view: hl.constexpr
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        left = a[bi, row, kk]
        right = b[bi, qq, kk]
        if view == "identity":
            left = left.permute(-2, -1)
            right = right.T
        elif view == "permute":
            right = right.permute(-1, -2)
        else:
            right = right.transpose(-1, -2)
        first = hl.dot(left, right)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


def _expression_config(schedule: str) -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)


def _expression_cpu_code(kernel: Any, args: tuple[Any, ...], schedule: str) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        kernel.reset()
        return kernel.bind(args).to_code(_expression_config(schedule))


def _expression_inputs(
    device: Any, *, seed: int | None = None
) -> tuple[torch.Tensor, ...]:
    shapes = ((2, 35, 49), (2, 49, 49), (2, 49, 24))
    if seed is None:
        values = tuple(
            torch.zeros(s, device=device, dtype=torch.bfloat16) for s in shapes
        )
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        values = tuple(
            torch.randn(s, device=device, dtype=torch.bfloat16, generator=generator)
            * 0.03
            for s in shapes
        )
    return (
        *values,
        torch.ones((2, 35), device=device, dtype=torch.bfloat16),
        torch.ones((2, 49), device=device, dtype=torch.bfloat16),
    )


def _expression_reference(
    args: tuple[torch.Tensor, ...], operation: str
) -> torch.Tensor:
    a, b, v, a_bias, b_bias = args
    left, right = a.float(), b.float()
    if operation in ("left_add", "both_add"):
        left = left + 1.0
    if operation in ("right_add", "both_add"):
        right = right + 1.0
    if operation == "bias":
        left = left + a_bias.float().unsqueeze(-1)
        right = right + b_bias.float().unsqueeze(-1)
    if operation == "exp":
        left, right = left.exp(), right.exp()
    first = (
        left.to(a.dtype).double() @ right.to(a.dtype).double().transpose(-1, -2)
    ).float()
    middle = (first + 1.0).to(a.dtype).double()
    last = (v.float() + 1.0).to(a.dtype).double()
    return (middle @ last).to(a.dtype)


def _expression_check(
    fn: Any, args: tuple[torch.Tensor, ...], mode: str, expected: torch.Tensor
) -> None:
    originals = tuple(x.clone() for x in args)
    output = fn(*args, mode)
    repeated = fn(*args, mode)
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(repeated, output, atol=0, rtol=0)
    assert output.data_ptr() != repeated.data_ptr()
    for x, original in zip(args, originals, strict=True):
        torch.testing.assert_close(x, original, atol=0, rtol=0)


def _predicate_value(condition: ast.expr, **coordinates: int) -> bool:
    values = {
        node.id: 0
        for node in ast.walk(condition)
        if isinstance(node, ast.Name) and node.id.startswith("chain_origin_")
    }
    values.update(coordinates)
    # Only the emitted integer-coordinate predicate is evaluated, never GPU
    # operations or generated expression values.
    return bool(
        eval(compile(ast.Expression(condition), "<operand-domain>", "eval"), {}, values)
    )


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_expression_codegen_masks_after_pointwise(
    operation: str, schedule: str
) -> None:
    code = _expression_cpu_code(
        _expression_chain, (*_expression_inputs("cpu"), operation), schedule
    )
    assert "chain_0_mma" in code and "chain_1_mma" in code
    assignments = {
        target.value.id: node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
    }
    for name in ("chain_0_a", "chain_0_b", "chain_1_b"):
        value = assignments[name]
        assert isinstance(value, ast.IfExp), (name, ast.unparse(value))
        # The final dtype conversion, not merely each global load, is masked.
        assert isinstance(value.body, ast.Call)
        assert ast.unparse(value.orelse) == "cutlass.BFloat16(0)"
        assert "< 49" in ast.unparse(value.test)
        load = "chain_0_load" if name.startswith("chain_0") else "chain_1_load"
        valid = 48 if name != "chain_1_b" else 48 * 32
        invalid = 49 if name != "chain_1_b" else 49 * 32
        assert _predicate_value(value.test, **{load: valid})
        assert not _predicate_value(value.test, **{load: invalid})
    name = "chain_1_a_bridge_values" if schedule == "cp_async_register" else "chain_1_a"
    value = assignments[name]
    assert isinstance(value, ast.IfExp)
    assert ast.unparse(value.orelse) == "cutlass.BFloat16(0)"
    assert "< 49" in ast.unparse(value.test)
    assert "< 35" in ast.unparse(value.test)
    if schedule == "cp_async_register":
        assert _predicate_value(
            value.test, chain_1_a_bridge_row=0, chain_1_a_bridge_col=48
        )
        assert not _predicate_value(
            value.test, chain_1_a_bridge_row=0, chain_1_a_bridge_col=49
        )
    else:
        assert _predicate_value(value.test, chain_1_load=48)
        assert not _predicate_value(value.test, chain_1_load=49)
    assert ("chain_0_c_ptr" not in code) is (schedule == "cp_async_register")


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_expression_padded_reduction_lanes(operation: str, schedule: str) -> None:
    args = _expression_inputs(DEVICE)
    _expression_chain.reset()
    fn = _expression_chain.bind((*args, operation)).compile_config(
        _expression_config(schedule)
    )
    # Exact zeros make the unmasked-lane error deterministic and very large:
    # (49 + 1) * 49 rather than (64 + 1) * 64 for both_add/bias/exp.
    _expression_check(fn, args, operation, _expression_reference(args, operation))
    for seed in range(5):
        args = _expression_inputs(DEVICE, seed=seed)
        _expression_check(fn, args, operation, _expression_reference(args, operation))


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("view", ["identity", "permute", "transpose"])
def test_negative_view_codegen(view: str, schedule: str) -> None:
    code = _expression_cpu_code(
        _negative_view_chain, (*_expression_inputs("cpu")[:3], view), schedule
    )
    assert "chain_0_mma" in code and "chain_1_mma" in code


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("view", ["identity", "permute", "transpose"])
def test_negative_view_operands(view: str, schedule: str) -> None:
    args = _expression_inputs(DEVICE, seed=0)[:3]
    _negative_view_chain.reset()
    fn = _negative_view_chain.bind((*args, view)).compile_config(
        _expression_config(schedule)
    )
    for seed in range(5):
        a, b, v = _expression_inputs(DEVICE, seed=seed)[:3]
        first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
        expected = (first.double() @ v.double()).to(a.dtype)
        _expression_check(fn, (a, b, v), view, expected)


@pytest.mark.parametrize("schedule", ["cp_async", "cp_async_register"])
def test_async_codegen_large_offsets_promote_before_multiply(schedule: str) -> None:
    # Fake CPU storage avoids allocating three multi-gigabyte tensors or CUDA
    # initialization. The first dimension needs Int64 byte/element offsets.
    with FakeTensorMode():
        args = tuple(
            torch.empty((1048577, 64, 64), device=CPU_DEVICE, dtype=torch.bfloat16)
            for _ in range(3)
        )
        code = _expression_cpu_code(_negative_view_chain, (*args, "identity"), schedule)
    pointers = [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.endswith("_async_pointer")
            for target in node.targets
        )
    ]
    assert len(pointers) == 3
    for pointer in pointers:
        products = [
            node
            for node in ast.walk(pointer)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and any(
                isinstance(x, ast.Constant) and x.value == 4096
                for x in (node.left, node.right)
            )
        ]
        assert products, ast.unparse(pointer)
        for product in products:
            operand = (
                product.right
                if isinstance(product.left, ast.Constant)
                else product.left
            )
            assert isinstance(operand, ast.Call), ast.unparse(product)
            assert ast.unparse(operand.func) == "cutlass.Int64", ast.unparse(product)


# Reuse.

REUSE_MODES = (
    "same",
    "shift",
    "reverse",
    "other_batch",
    "other_source",
    "computed",
    "earlier",
    "gather",
    "constant_col",
)

_REUSES = {"same", "shift", "reverse", "constant_col"}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reuse_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    other: torch.Tensor,
    permutation: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = b.shape[1], v.shape[2]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        last = v[bi, qq, col]
        if mode == "computed":
            last = (last.float() * 0.5).to(v.dtype)
        if mode == "gather":
            last = v[bi, permutation[qq], col]
        result = hl.dot(first.to(a.dtype), last)
        if mode == "shift":
            residual = v[bi, row.index + 32, col]
        elif mode == "reverse":
            residual = v[bi, 79 - row.index, col]
        elif mode == "other_batch":
            residual = v[(bi + 1) % batches, row, col]
        elif mode == "other_source":
            residual = other[bi, row, col]
        elif mode == "earlier":
            residual = b[bi, row, col]
        elif mode == "constant_col":
            residual = v[bi, row, 0][:, None]
        else:
            residual = v[bi, row, col]
        out[bi, row, col] = (result + residual.float()).to(a.dtype)
    return out


def _reuse_config(warps: int) -> helion.Config:
    return helion.Config(
        num_warps=warps, cute_chained_mma_schedule="cp_async_register_reuse"
    )


def _reuse_inputs(device: Any, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    # K=64, Q=49 and M=35 stress padded domains. V's physical length80 means
    # indices49..63 exist globally but must not read masked shared padding.
    values = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        for shape in ((2, 35, 64), (2, 49, 64), (2, 80, 64), (2, 80, 64))
    )
    return (*values, torch.randperm(49, device=device, generator=generator))


def _reuse_cpu_code(args: tuple[torch.Tensor, ...], mode: str, warps: int) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        _reuse_chain.reset()
        return _reuse_chain.bind((*args, mode)).to_code(_reuse_config(warps))


def _reuse_expressions(code: str) -> list[ast.IfExp]:
    epilogue = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_store_step"
    )
    return [
        node
        for node in ast.walk(epilogue)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Subscript)
        and isinstance(node.body.value, ast.Name)
        and node.body.value.id in {"chain_1_a", "chain_1_b"}
    ]


def _predicate(
    condition: ast.expr, *, row_origin: int, col_origin: int, store: int = 0
) -> bool:
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": row_origin,
        "chain_origin_2": col_origin,
        "chain_store": store,
    }
    return bool(
        eval(compile(ast.Expression(condition), "<reuse-domain>", "eval"), {}, values)
    )


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("mode", REUSE_MODES)
def test_staged_reuse_codegen_eligibility(mode: str, warps: int) -> None:
    code = _reuse_cpu_code(_reuse_inputs("cpu"), mode, warps)
    assert f"block=({warps * 32}, 1, 1)" in code
    expressions = _reuse_expressions(code)
    assert bool(expressions) is (mode in _REUSES)
    for expression in expressions:
        assert ".load()" in ast.unparse(expression.orelse)
        assert "chain_0_b[" not in ast.unparse(expression)
    if mode == "shift":
        assert len(expressions) == 1
        condition = expressions[0].test
        assert _predicate(condition, row_origin=0, col_origin=0)
        # Global V[49] is valid, but outside the 49 staged values. Padding
        # shared memory to64 must not replace a valid global value with zero.
        assert not _predicate(condition, row_origin=16, col_origin=0, store=32)
        assert "< 49" in ast.unparse(condition)
    if mode == "constant_col":
        assert len(expressions) == 1
        condition = expressions[0].test
        assert _predicate(condition, row_origin=0, col_origin=0)
        assert not _predicate(condition, row_origin=0, col_origin=32)


def _reuse_reference(args: tuple[torch.Tensor, ...], mode: str) -> torch.Tensor:
    a, b, v, other, permutation = args
    first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
    last = v[:, :49, :]
    if mode == "computed":
        last = (last.float() * 0.5).to(v.dtype)
    if mode == "gather":
        last = v[:, permutation, :]
    result = first.double() @ last.double()
    if mode == "shift":
        residual = v[:, 32:67, :]
    elif mode == "reverse":
        residual = v[:, 45:80, :].flip(1)
    elif mode == "other_batch":
        residual = v.flip(0)[:, :35, :]
    elif mode == "other_source":
        residual = other[:, :35, :]
    elif mode == "earlier":
        residual = b[:, :35, :]
    elif mode == "constant_col":
        residual = v[:, :35, :1]
    else:
        residual = v[:, :35, :]
    return (result + residual.double()).to(a.dtype)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("mode", REUSE_MODES)
def test_staged_reuse_numerics(mode: str, warps: int) -> None:
    args = _reuse_inputs(DEVICE)
    _reuse_chain.reset()
    fn = _reuse_chain.bind((*args, mode)).compile_config(_reuse_config(warps))
    for seed in range(5):
        args = _reuse_inputs(DEVICE, seed)
        if mode == "other_source" and seed % 2:
            # Reusing the same compiled callable must also remain correct
            # when two previously distinct read-only arguments alias.
            args = (*args[:3], args[2], args[4])
        originals = tuple(value.clone() for value in args)
        result = fn(*args, mode)
        repeated = fn(*args, mode)
        assert result.dtype is torch.bfloat16
        assert result.data_ptr() != repeated.data_ptr()
        torch.testing.assert_close(
            result, _reuse_reference(args, mode), atol=0.01, rtol=0.01
        )
        torch.testing.assert_close(repeated, result, atol=0, rtol=0)
        for value, original in zip(args, originals, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)


# Scan cache.

SCAN_CACHE_SCHEDULE = "cp_async_register_reuse_scan"

SCAN_CACHE_MODES = (
    "same",
    "shift",
    "other_batch",
    "other_source",
    "gather",
    "multiple",
    "scan_only",
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_cached_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    other: torch.Tensor,
    permutation: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, length, k = a.shape
    n = v.shape[2]
    out = torch.empty_like(v)
    for batch, row, col in hl.tile([batches, length, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(length)
        scan_value = delta[bi, qq].float()
        if mode == "multiple":
            scan_value *= other[bi, qq].float()
        decay = hl.cumsum(scan_value, dim=0)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        if mode == "shift":
            scale = delta[bi, qq + 16]
            residual = delta[bi, row.index + 16]
        elif mode == "other_batch":
            scale = delta[(bi + 1) % batches, qq]
            residual = delta[(bi + 1) % batches, row]
        elif mode == "other_source":
            scale = other[bi, qq]
            residual = other[bi, row]
        elif mode == "gather":
            scale = delta[bi, permutation[qq]]
            residual = delta[bi, permutation[row]]
        elif mode == "multiple":
            scale = other[bi, qq]
            residual = delta[bi, row]
        else:
            scale = delta[bi, qq]
            residual = delta[bi, row]
        if mode != "scan_only":
            weights *= scale[None, :].float()
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        if mode != "scan_only":
            result += residual[:, None].float()
        out[bi, row, col] = result.to(v.dtype)
    return out


def _scan_cache_inputs(
    device: Any, length: int, stride: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(481)
    operands = tuple(
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
        for shape in ((2, length, 64), (2, length, 64), (2, length, 32))
    )
    vectors = tuple(
        -torch.rand((2, (length + 32) * stride), device=device, generator=generator).to(
            torch.bfloat16
        )[:, ::stride]
        * 0.005
        for _ in range(2)
    )
    # Multiplication would make a contiguous tensor; retain noncontiguous views.
    if stride > 1:
        vectors = tuple(
            vector.repeat_interleave(stride, dim=1)[:, ::stride] for vector in vectors
        )
    permutation = torch.randperm(length, device=device, generator=generator)
    return (*operands, *vectors, permutation)


def _scan_cache_cpu_code(
    args: tuple[torch.Tensor, ...],
    mode: str,
    schedule: str = SCAN_CACHE_SCHEDULE,
    capacity: int = 232448,
    warps: int = 4,
) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        _scan_cached_chain.reset()
        return _scan_cached_chain.bind((*args, mode)).to_triton_code(
            helion.Config(cute_chained_mma_schedule=schedule, num_warps=warps)
        )


@pytest.mark.parametrize("mode", SCAN_CACHE_MODES)
@pytest.mark.parametrize("length,stride", [(64, 1), (49, 2)])
def test_scan_input_cache_codegen(mode: str, length: int, stride: int) -> None:
    code = _scan_cache_cpu_code(_scan_cache_inputs("cpu", length, stride), mode)
    tree = ast.parse(code)
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id.startswith("chain_scan_0_input_")
            for target in node.targets
        )
    ]
    expected = (
        2 if mode == "multiple" else 0 if mode in ("scan_only", "other_source") else 1
    )
    assert len(stores) == expected
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
    ]
    if mode in ("same", "shift", "multiple"):
        assert reads
        assert any("chain_store" in ast.unparse(node.test) for node in reads)
        assert any("bridge" in ast.unparse(node.test) for node in reads)
        if length == 49:
            assert all("< 49" in ast.unparse(node.test) for node in reads)
    else:
        assert not reads
    for store in stores:
        # Store the existing loaded register, never issue another global read.
        assert ".load()" not in ast.unparse(store.value)


def test_scan_input_cache_is_optional() -> None:
    code = _scan_cache_cpu_code(
        _scan_cache_inputs("cpu", 64, 1), "same", "cp_async_register_reuse"
    )
    assert "chain_scan_0_input_" not in code


def test_scan_input_cache_shared_memory_admission() -> None:
    args = _scan_cache_inputs("cpu", 64, 1)
    # Two dot shapes (16,64,64), (16,32,64), four scan warp totals.
    allocations = (
        2 * (16 * 64 + 8 * 64),
        2 * (64 * 64 + 8 * 64),
        4 * 16 * (64 + 4),
        4 * 16 * (32 + 4),
        4 * (64 + 4),
    )
    capacity = sum((size + 127) // 128 * 128 for size in allocations)
    assert "chain_1_mma" in _scan_cache_cpu_code(
        args, "same", "cp_async_register_reuse", capacity
    )
    with pytest.raises(exc.BackendUnsupported, match="associative_scan input"):
        _scan_cache_cpu_code(args, "same", capacity=capacity)
    # The non-cache schedule must also count scan allocation alignment.
    with pytest.raises(exc.BackendUnsupported, match="associative_scan input"):
        _scan_cache_cpu_code(args, "same", "cp_async_register_reuse", sum(allocations))


def test_scan_input_cache_logical_tail_predicate() -> None:
    code = _scan_cache_cpu_code(_scan_cache_inputs("cpu", 49, 2), "shift")
    reads = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
        and "chain_store" in ast.unparse(node.test)
    ]
    assert reads
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": 32,
        "chain_origin_2": 0,
    }
    for read in reads:
        condition = compile(ast.Expression(read.test), "<scan-cache-domain>", "eval")
        # Global indices48 and49 both exist and fit shared64; logical49 is
        # outside the scan's original vector and must use global fallback.
        assert eval(condition, {}, {**values, "chain_store": 0})
        assert not eval(condition, {}, {**values, "chain_store": 32})


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scan_input_cache_dtype_and_eight_warps_codegen(dtype: torch.dtype) -> None:
    args = _scan_cache_inputs("cpu", 49, 2)
    args = (*args[:3], args[3].to(dtype), args[4].to(dtype), args[5])
    code = _scan_cache_cpu_code(args, "multiple", warps=8)
    assert "chain_scan_0_input_1" in code
    dtype_name = "cutlass.Float16" if dtype is torch.float16 else "cutlass.Float32"
    assert f"{dtype_name}(chain_scan_0_input_0[" in code
    assert "if chain_scan_0_index < 64:" in code
    assert "cute.make_layout(8)" in code


@pytest.mark.parametrize("mode", SCAN_CACHE_MODES)
@pytest.mark.parametrize("length,stride", [(64, 1), (49, 2)])
def test_scan_input_cache_runtime(mode: str, length: int, stride: int) -> None:
    args = _scan_cache_inputs(DEVICE, length, stride)
    frozen = tuple(arg.clone() for arg in args)
    fn = _scan_cached_chain.bind((*args, mode)).compile_config(
        helion.Config(cute_chained_mma_schedule=SCAN_CACHE_SCHEDULE)
    )
    output = fn(*args, mode)
    a, b, v, delta, other, permutation = args
    scan = delta[:, :length].double()
    if mode == "multiple":
        scan *= other[:, :length].double()
    decay = scan.cumsum(-1)
    weights = (a.double() @ b.double().transpose(-1, -2)) * (
        decay[:, :, None] - decay[:, None, :]
    ).exp()
    selected = delta[:, :length]
    if mode == "shift":
        selected = delta[:, 16 : length + 16]
    elif mode == "other_batch":
        selected = delta.flip(0)[:, :length]
    elif mode in ("other_source", "multiple"):
        selected = other[:, :length]
    elif mode == "gather":
        selected = delta[:, permutation]
    if mode != "scan_only":
        weights *= selected[:, None, :].double()
    result = torch.tril(weights).to(v.dtype).double() @ v.double()
    if mode != "scan_only":
        residual = delta[:, :length] if mode == "multiple" else selected
        result += residual[:, :, None].double()
    torch.testing.assert_close(
        output,
        result.to(v.dtype),
        atol=0.003 if mode == "scan_only" else 0.0001,
        rtol=0.02,
    )
    torch.testing.assert_close(fn(*args, mode), output, atol=0, rtol=0)
    assert output.data_ptr() not in {arg.data_ptr() for arg in args}
    for actual, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_NAMES = {
    torch.bfloat16: "cutlass.BFloat16",
    torch.float16: "cutlass.Float16",
    torch.float32: "cutlass.Float32",
}


def _typed_inputs(
    device: str | torch.device, length: int, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    args = _scan_cache_inputs(device, length, 2)
    # Keep independently typed sources and noncontiguous physical strides.
    delta = args[3].to(dtype).repeat_interleave(2, dim=1)[:, ::2]
    other = args[4].to(torch.float32).repeat_interleave(2, dim=1)[:, ::2]
    return (*args[:3], delta, other, args[5])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("length", [49, 64])
def test_scan_cache_original_dtype_codegen(dtype: torch.dtype, length: int) -> None:
    code = _scan_cache_cpu_code(_typed_inputs("cpu", length, dtype), "multiple")
    tree = ast.parse(code)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    caches = {
        name: value
        for name, value in assignments.items()
        if name.startswith("chain_scan_0_input_")
    }
    assert len(caches) == 2
    cache_dtypes = []
    for name, tensor in caches.items():
        assert isinstance(tensor, ast.Call)
        allocation = tensor.args[0]
        assert isinstance(allocation, ast.Call)
        assert ast.unparse(allocation.func) == "cute.arch.alloc_smem"
        cache_dtype = ast.unparse(allocation.args[0])
        cache_dtypes.append(cache_dtype)
        assert ast.literal_eval(allocation.args[1]) == 64
        assert allocation.keywords[0].arg == "alignment"
        assert ast.literal_eval(allocation.keywords[0].value) == 128
        stores = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript) and ast.unparse(target.value) == name
                for target in node.targets
            )
        ]
        assert len(stores) == 1
        stored = stores[0].body if isinstance(stores[0], ast.IfExp) else stores[0]
        assert isinstance(stored, ast.Call)
        assert ast.unparse(stored.func) == cache_dtype
        # The cache stores exactly the raw leaf register, not its FP32 scan
        # conversion or product. This also prevents a duplicate global read.
        raw = stored.args[0]
        assert isinstance(raw, ast.Name)
        loaded = assignments[raw.id]
        assert isinstance(loaded, ast.IfExp)
        assert isinstance(loaded.body, ast.Call)
        assert isinstance(loaded.body.func, ast.Attribute)
        assert loaded.body.func.attr == "load"
        assert isinstance(loaded.orelse, ast.Call)
        assert ast.unparse(loaded.orelse.func) == cache_dtype
    assert sorted(cache_dtypes) == sorted([_NAMES[dtype], "cutlass.Float32"])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scan_cache_original_dtype_exact_shared_admission(dtype: torch.dtype) -> None:
    initialized_before = torch.cuda.is_initialized()
    args = _typed_inputs("cpu", 49, dtype)
    allocations = (
        2 * (16 * 64 + 8 * 64),
        2 * (64 * 64 + 8 * 64),
        4 * 16 * (64 + 4),
        4 * 16 * (32 + 4),
        4 * (64 + 4),
        64 * args[3].element_size(),
        64 * args[4].element_size(),
    )
    capacity = sum((size + 127) // 128 * 128 for size in allocations)
    original = chained_matmul.plan_chained_matmul
    decisions: list[ChainedMatmulPlan | None] = []

    def observe_plan(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        decisions.append(plan)
        return plan

    with patch.object(
        chained_matmul, "plan_chained_matmul", side_effect=observe_plan
    ) as planner:
        code = _scan_cache_cpu_code(args, "multiple", capacity=capacity)
        assert "chain_1_mma" in code
        assert planner.call_count == 1
        assert len(decisions) == 1 and decisions[0] is not None
        decisions.clear()
        planner.reset_mock()
        # A fallback may reject for a different reason after this planner
        # declines the chain. Check the actual admission decision directly.
        with pytest.raises(exc.BackendUnsupported):
            _scan_cache_cpu_code(args, "multiple", capacity=capacity - 1)
        assert planner.call_count == 1
        assert decisions == [None]
    assert torch.cuda.is_initialized() is initialized_before


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scan_cache_original_dtype_runtime(dtype: torch.dtype) -> None:
    args = _typed_inputs(DEVICE, 49, dtype)
    originals = tuple(value.clone() for value in args)
    _scan_cached_chain.reset()
    fn = _scan_cached_chain.bind((*args, "multiple")).compile_config(
        helion.Config(cute_chained_mma_schedule="cp_async_register_reuse_scan")
    )
    actual = fn(*args, "multiple")
    a, b, v, delta, other, permutation = args
    decay = (delta[:, :49].double() * other[:, :49].double()).cumsum(-1)
    weights = (a.double() @ b.double().transpose(-1, -2)) * (
        decay[:, :, None] - decay[:, None, :]
    ).exp()
    weights *= other[:, None, :49].double()
    expected = torch.tril(weights).to(v.dtype).double() @ v.double()
    expected += delta[:, :49, None].double()
    torch.testing.assert_close(actual, expected.to(v.dtype), atol=0.0001, rtol=0.02)
    repeated = fn(*args, "multiple")
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert repeated.data_ptr() != actual.data_ptr()
    for value, original in zip(args, originals, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


# Scan cache guards.

SCAN_GUARD_MODES = ("scalar_last", "scalar_outside", "broadcast", "reverse", "nested")

SCAN_GUARD_SCHEDULE = "cp_async_register_reuse_scan"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_guard_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, length, k = a.shape
    n = v.shape[2]
    out = torch.empty_like(v)
    for batch, row, col in hl.tile([batches, length, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(length)
        scan_value = delta[bi, qq].float()
        if mode == "reverse":
            scan_value = delta[bi, length - 1 - qq].float()
        decay = hl.cumsum(scan_value * 0.01, dim=0)
        if mode == "nested":
            decay = hl.cumsum(decay * 0.01 + delta[bi, qq].float() * 0.01, dim=0)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        weights *= delta[bi, qq][None, :].float()
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        if mode == "scalar_outside":
            result += delta[bi, length].float()
        elif mode == "broadcast":
            result += delta[bi, 0].float()
        else:
            result += delta[bi, length - 1].float()
        out[bi, row, col] = result.to(v.dtype)
    return out


def _scan_guard_inputs(device: Any, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    operands = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        for shape in ((2, 49, 64), (2, 49, 64), (2, 49, 32))
    )
    delta = (
        torch.randn((2, 80), device=device, dtype=torch.float32, generator=generator)
        * 0.1
    )
    return (*operands, delta)


def _scan_guard_cpu_code(args: tuple[torch.Tensor, ...], mode: str) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        _scan_guard_chain.reset()
        return _scan_guard_chain.bind((*args, mode)).to_code(
            helion.Config(num_warps=8, cute_chained_mma_schedule=SCAN_GUARD_SCHEDULE)
        )


@pytest.mark.parametrize("mode", SCAN_GUARD_MODES)
def test_scan_cache_scalar_and_nested_codegen(mode: str) -> None:
    code = _scan_guard_cpu_code(_scan_guard_inputs("cpu"), mode)
    assert "chain_scan_0_input_0" in code
    assert ("chain_scan_1_input_0" in code) is (mode == "nested")
    # Scalar host loads can safely read the vector cache, including broadcast
    # to every output coordinate. An index49 request is globally valid in the
    # 80-element input but outside the original49-element scan vector.
    epilogue = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For) and ast.unparse(node.target) == "chain_store_step"
    )
    reads = [
        node
        for node in ast.walk(epilogue)
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
    ]
    assert reads
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": 0,
        "chain_origin_2": 0,
        "chain_store": 0,
    }
    for read in reads:
        assert ".load()" in ast.unparse(read.orelse)
        actual = bool(
            eval(
                compile(ast.Expression(read.test), "<scalar-cache>", "eval"), {}, values
            )
        )
        assert actual is (mode != "scalar_outside")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.bool,
    ],
)
def test_scan_cache_lossless_dtype_policy(dtype: torch.dtype) -> None:
    graph = torch.fx.Graph()
    source = graph.call_function(_tracing_ops._host_tensor, ("delta",))
    source.meta["val"] = torch.empty(49, dtype=dtype)
    leaf = graph.call_function(memory_ops.load, (source, [slice(None)], None))
    leaf.meta["val"] = torch.empty(49, dtype=dtype)
    scan = graph.call_function(scan_ops._associative_scan, (0, leaf, 0, False, False))
    later = graph.call_function(torch.add, (scan, leaf))
    output = graph.call_function(_tracing_ops._host_tensor, ("out",))
    store = graph.call_function(memory_ops.store, (output, [], later, None))
    # Isolate the cache admission policy from tile-shape configuration. The
    # supported floating leaf dtypes are cached without conversion.
    with patch.object(
        chained_matmul, "_shape", side_effect=lambda node: tuple(node.meta["val"].shape)
    ):
        candidates = chained_matmul._scan_cache_candidates(scan, (scan,), store)
    assert (leaf in candidates) is (
        dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _scan_guard_reference(args: tuple[torch.Tensor, ...], mode: str) -> torch.Tensor:
    a, b, v, delta = args
    scan = delta[:, :49].double()
    if mode == "reverse":
        scan = scan.flip(1)
    decay = (scan * 0.01).cumsum(-1)
    if mode == "nested":
        decay = (decay * 0.01 + delta[:, :49].double() * 0.01).cumsum(-1)
    first = a.double() @ b.double().transpose(-1, -2)
    weights = first * torch.exp(decay[:, :, None] - decay[:, None, :])
    weights = torch.tril(weights * delta[:, None, :49].double()).to(v.dtype)
    scalar = 49 if mode == "scalar_outside" else 0 if mode == "broadcast" else 48
    result = weights.double() @ v.double() + delta[:, scalar, None, None].double()
    return result.to(v.dtype)


@pytest.mark.parametrize("mode", SCAN_GUARD_MODES)
def test_scan_cache_scalar_and_nested_runtime(mode: str) -> None:
    args = _scan_guard_inputs(DEVICE)
    _scan_guard_chain.reset()
    fn = _scan_guard_chain.bind((*args, mode)).compile_config(
        helion.Config(num_warps=8, cute_chained_mma_schedule=SCAN_GUARD_SCHEDULE)
    )
    for seed in range(5):
        args = _scan_guard_inputs(DEVICE, seed)
        frozen = tuple(value.clone() for value in args)
        actual = fn(*args, mode)
        repeated = fn(*args, mode)
        torch.testing.assert_close(
            actual, _scan_guard_reference(args, mode), atol=0.003, rtol=0.02
        )
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        assert actual.data_ptr() != repeated.data_ptr()
        for value, original in zip(args, frozen, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
