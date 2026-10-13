from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gathered_masked_loop(a, b, initial, gather: hl.constexpr, mask_kind: hl.constexpr):
    steps, m, k = a.shape
    n = b.shape[1]
    history = torch.empty((steps, m, n), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[16, 16]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            if mask_kind == 1:
                value = hl.load(
                    a, [step.id, rows, kk], extra_mask=(rows.index % 2 == 0)[:, None]
                )
            elif mask_kind == 2:
                value = hl.load(
                    a, [step.id, rows, kk], extra_mask=(kk % 2 == 0)[None, :]
                )
            else:
                value = a[step.id, rows, kk]
            if gather:
                gathered = hl.load(value, [(rows.index - rows.begin) ^ 1, slice(None)])
            else:
                gathered = value
            state = hl.dot(
                gathered, b[step.id, cols, kk].T, acc=state, out_dtype=torch.float32
            )
            history[step.id, rows, cols] = state
        final[rows, cols] = state
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_fanout(a, other, common, initial):
    steps, m, k = a.shape
    n = common.shape[1]
    history = torch.empty((steps, m, n), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[16, 128]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            rhs = common[step.id, cols, kk].T
            first = hl.dot(
                a[step.id, rows, kk], rhs, acc=state, out_dtype=torch.float32
            )
            second = hl.dot(other[step.id, rows, kk], rhs, out_dtype=torch.float32)
            state = first + second
            history[step.id, rows, cols] = state
        final[rows, cols] = state
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_masked_chain(a, b, c):
    m, k = a.shape
    n = c.shape[1]
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 64]):
        kk = hl.arange(k)
        masked = hl.load(a, [rows, kk], extra_mask=(rows.index % 2 == 0)[:, None])
        first = hl.dot(masked, b[kk, :], out_dtype=torch.float32)
        output[rows, cols] = hl.dot(
            first.to(a.dtype), c[kk, cols], out_dtype=torch.float32
        )
    return output


def _root_args(device: str | torch.device) -> tuple:
    generator = torch.Generator(device=device).manual_seed(380)
    return tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        for shape in ((128, 128), (128, 128), (128, 64))
    )


def _root_config(vectorize: bool) -> helion.Config:
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=vectorize,
    )


def _config(vectorize: bool, *, grouped: bool = False) -> helion.Config:
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=grouped,
        cute_chained_pointwise_vectorize=vectorize,
        cute_chained_scratch_layout="xor",
    )


def _inputs(
    device: str | torch.device, gather: bool, mask_kind: int, storage: str = "dense"
) -> tuple:
    generator = torch.Generator(device=device).manual_seed(377)
    if storage == "strided":
        a = torch.randn(
            (3, 16, 32), dtype=torch.bfloat16, device=device, generator=generator
        )[..., ::2]
    elif storage == "unaligned":
        a = torch.randn(
            (3 * 16 * 16 + 1,), dtype=torch.bfloat16, device=device, generator=generator
        )[1:].view(3, 16, 16)
    else:
        a = torch.randn(
            (3, 16, 16), dtype=torch.bfloat16, device=device, generator=generator
        )
    return (
        a * 0.1 if storage == "dense" else a,
        torch.randn(
            (3, 16, 16), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.1,
        torch.randn((16, 16), dtype=torch.float32, device=device, generator=generator)
        * 0.1,
        gather,
        mask_kind,
    )


def _fanout_inputs(device: str | torch.device) -> tuple:
    generator = torch.Generator(device=device).manual_seed(379)
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.1
        for shape, dtype in (
            ((3, 16, 16), torch.float16),
            ((3, 16, 16), torch.float16),
            ((3, 128, 16), torch.float16),
            ((16, 128), torch.float32),
        )
    )


def _undefined_temporaries(source: str) -> set[str]:
    tree = ast.parse(source)
    reads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    writes = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return {name for name in reads - writes if name.startswith("chain_value")}


@pytest.mark.parametrize(
    "gather,mask_kind,storage",
    [
        (True, 0, "dense"),
        (False, 1, "dense"),
        (False, 2, "dense"),
        (False, 0, "strided"),
        (False, 0, "unaligned"),
    ],
)
def test_internal_gather_masks_and_storage_preserve_scalar_fallbacks(
    gather: bool, mask_kind: int, storage: str
) -> None:
    with _cpu_codegen():
        bound = _gathered_masked_loop._bind_isolated(
            _inputs("cpu", gather, mask_kind, storage)
        )
        source = bound.to_code(_config(True))
    assert not _undefined_temporaries(source)
    assert "chain_0_b_0_vector" in source  # A separate legal vector always activates.
    if gather or mask_kind == 2 or storage == "strided":
        assert "chain_0_a_0_vector" not in source
    else:
        assert "chain_0_a_0_vector" in source
    if mask_kind:
        assert "% 2" in source
    assert "_last_address ==" in source and "_address % 16 == 0" in source


def test_transposed_grouped_rhs_keeps_member_offset_and_layout() -> None:
    with _cpu_codegen():
        bound = _transposed_fanout._bind_isolated(_fanout_inputs("cpu"))
        source = bound.to_code(_config(True, grouped=True))
    assert "chain_1_mma" not in source
    assert "cute.domain_offset((16, 0), chain_0_b)" in source
    assert "chain_0_b_0_vector" in source and "chain_0_b_1_vector" in source
    assert "chain_0_b_1_vector_row < 16" in source
    assert not _undefined_temporaries(source)


def test_root_masked_chain_uses_shared_vector_fallback() -> None:
    with _cpu_codegen():
        bound = _root_masked_chain._bind_isolated(_root_args("cpu"))
        source = bound.to_code(_root_config(True))
    assert "_vectorized = cutlass.Boolean(False)" in source
    assert "_last_address ==" in source
    assert "% 2" in source
    assert not _undefined_temporaries(source)


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_full_prefill_vector_stage_retains_runtime_token_bounds(
    chunk_size: int,
) -> None:
    from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
    from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32

    from .test_cute_chunk_prefill import _inputs as prefill_inputs

    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    grouped = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides={
            "cute_chained_group_contractions": True,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        },
    )
    with _cpu_codegen():
        bound = grouped._bind_isolated(
            prefill_inputs(tokens=67, heads=8, device=torch.device("cpu"))
        )
        config = _config(True, grouped=True)
        config.config["block_sizes"] = [128]
        source = bound.to_code(config)
    assert "chain_loop_capture_" in source and "chain_loop_index" in source
    assert "_vectorized = cutlass.Boolean(False)" in source
    assert "_last_address ==" in source
    assert "_address % 16 == 0" in source
    vector_guards = [
        ast.unparse(node.test)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and "_vector_leaf_" in ast.unparse(node.test)
        and "_index_" in ast.unparse(node.test)
    ]
    assert any(
        "chain_loop_capture_" in guard
        and "operator.lt" in guard
        and "chain_loop_index" in guard
        and "< 536" in guard  # Flattened 67-token, eight-head host storage.
        for guard in vector_guards
    )
    assert not _undefined_temporaries(source)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "gather,mask_kind,storage",
    [
        (True, 0, "dense"),
        (False, 1, "dense"),
        (False, 2, "dense"),
        (False, 0, "strided"),
        (False, 0, "unaligned"),
    ],
)
def test_vector_stage_gpu_internal_gather_masks_and_storage(
    gather: bool, mask_kind: int, storage: str
) -> None:
    args = _inputs(DEVICE, gather, mask_kind, storage)
    a, b, initial = args[:3]
    expected = initial.clone()
    for step in range(a.shape[0]):
        left = a[step].float()
        if mask_kind == 1:
            left = left * (torch.arange(16, device=DEVICE)[:, None] % 2 == 0)
        elif mask_kind == 2:
            left = left * (torch.arange(16, device=DEVICE)[None, :] % 2 == 0)
        if gather:
            left = left[torch.arange(16, device=DEVICE) ^ 1]
        expected = left @ b[step].float().T + expected
    bound = _gathered_masked_loop._bind_isolated(args)
    actual = bound.compile_config(_config(True))(*args)
    scalar = bound.compile_config(_config(False))(*args)
    torch.testing.assert_close(actual, scalar, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected, rtol=5e-4, atol=5e-5)


@skipUnlessBackends(["cute"])
def test_vector_stage_gpu_transposed_grouped_offset() -> None:
    args = _fanout_inputs(DEVICE)
    a, other, common, initial = args
    expected = initial.clone()
    for step in range(a.shape[0]):
        expected = a[step].float() @ common[step].float().T + expected
        expected = expected + other[step].float() @ common[step].float().T
    bound = _transposed_fanout._bind_isolated(args)
    actual = bound.compile_config(_config(True, grouped=True))(*args)
    scalar = bound.compile_config(_config(False, grouped=True))(*args)
    torch.testing.assert_close(actual, scalar, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected, rtol=5e-4, atol=5e-5)


@skipUnlessBackends(["cute"])
def test_vector_stage_gpu_root_masked_chain() -> None:
    args = _root_args(DEVICE)
    a, b, c = args
    masked = a.float() * (torch.arange(128, device=DEVICE)[:, None] % 2 == 0)
    expected = (masked @ b.float()).to(a.dtype).float() @ c.float()
    bound = _root_masked_chain._bind_isolated(args)
    actual = bound.compile_config(_root_config(True))(*args)
    scalar = bound.compile_config(_root_config(False))(*args)
    torch.testing.assert_close(actual, scalar, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
