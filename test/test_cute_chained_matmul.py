from __future__ import annotations

import ast
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


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


def _compile(kernel: Any, args: tuple[torch.Tensor, ...], dots: int = 2):
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
    fn = _compile(_plain_chain, args)
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
    fn = _compile(_three_dots, (*args, decay), dots=3)
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
    fn = _compile(_scaled_operand, (a, b, scale), dots=1)
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
    fn = _compile(_colliding_names, args)
    intermediate = (args[0].float() @ args[1].float().T).to(torch.bfloat16)
    reference = (intermediate.float() @ args[2].float()).to(torch.bfloat16)
    torch.testing.assert_close(fn(*args), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("groups", [1, 3])
def test_native_nt_fp32_fallback(groups: int) -> None:
    a = torch.randn((2, 35, groups, 64), device=DEVICE, dtype=torch.bfloat16) * 0.1
    b = torch.randn((2, 27, groups, 64), device=DEVICE, dtype=torch.bfloat16) * 0.1
    fn = _compile(_native_nt, (a, b), dots=1)
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
