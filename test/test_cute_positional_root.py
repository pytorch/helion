from __future__ import annotations

import re

import pytest
import torch

from ._cute_aux import _cpu_codegen
from ._positional_scan_kernels import affine
from ._positional_scan_kernels import padded_reverse_exp_cumsum
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import output_only
import helion.language as hl

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@helion.kernel(backend="cute", static_shapes=True)
def square_mask_only(beta: torch.Tensor) -> torch.Tensor:
    # Previously silent: ordinary lowering wrote only the diagonal.
    bhn, c = beta.size(0), hl.specialize(beta.size(1))
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=beta.device)
    for tile in hl.tile(bhn, block_size=1):
        b = beta[tile, :].to(torch.float32)
        idx = hl.arange(c)
        lower = (idx[:, None] > idx[None, :]).to(torch.float32)
        eye = (idx[:, None] == idx[None, :]).to(torch.float32)
        out[tile, :, :] = b[:, :, None] * lower[None, :, :] + eye[None, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_gram(k: torch.Tensor) -> torch.Tensor:
    # M == N == K share one canonical id (C == D).
    bhn, c = k.size(0), hl.specialize(k.size(1))
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=k.device)
    for tile in hl.tile(bhn, block_size=1):
        kt = k[tile, :, :].float()
        out[tile, :, :] = hl.dot(kt, kt.transpose(-2, -1))
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_swap_carry(a: torch.Tensor) -> torch.Tensor:
    # Simultaneous carry update: x, y = y, x @ y must read old x and old y.
    bhn, c = a.size(0), hl.specialize(a.size(1))
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=a.device)
    for tile in hl.tile(bhn, block_size=1):
        x = a[tile, :, :].float()
        idx = hl.arange(c)
        y = (
            (idx[:, None] == idx[None, :])
            .to(torch.float32)[None, :, :]
            .broadcast_to([tile, c, c])
        )
        for _ in range(3):
            x, y = y, hl.dot(x, y)
        out[tile, :, :] = x + 2 * y
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_gram_tail(k: torch.Tensor) -> torch.Tensor:
    # D=40 with a 16-wide K loop: the last K tile is partial (masks/_mask_to).
    bhn, c, d = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=k.device)
    for tile in hl.tile(bhn, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for tk in hl.tile(d, block_size=16):
            kt = k[tile, :, tk]
            acc = hl.dot(kt, kt.transpose(-2, -1), acc=acc)
        out[tile, :, :] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_reduce(x: torch.Tensor) -> torch.Tensor:
    # Summation must reduce the complete positional row, not its diagonal.
    bhn, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([bhn, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(bhn, block_size=1):
        out[tile, :] = x[tile, :, :].float().sum(-1)
    return out


def _ref_mask(beta: torch.Tensor) -> torch.Tensor:
    i = torch.arange(beta.shape[1], device=beta.device)
    return (
        beta.float()[:, :, None] * (i[:, None] > i[None, :]).float()
        + (i[:, None] == i[None, :]).float()
    )


def _ref_swap(a: torch.Tensor) -> torch.Tensor:
    x = a.float()
    y = torch.eye(a.shape[1], device=a.device).expand_as(x)
    for _ in range(3):
        x, y = y, x @ y
    return x + 2 * y


@requires_cuda
def test_square_mask_matches_eager() -> None:
    beta = torch.randn(4, 64, device=DEVICE)
    code, out = code_and_output(square_mask_only, (beta,))
    assert "ptp_thread" in code  # positional route owns the root
    torch.testing.assert_close(out, _ref_mask(beta))


@requires_cuda
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_square_gram_matches_eager(num_warps: int) -> None:
    k = torch.randn(4, 32, 32, device=DEVICE, dtype=torch.bfloat16)
    _, out = code_and_output(square_gram, (k,), num_warps=num_warps)
    torch.testing.assert_close(out, k.float() @ k.float().transpose(-1, -2))


@requires_cuda
def test_square_simultaneous_carry() -> None:
    a = torch.randn(2, 16, 16, device=DEVICE) / 4
    _, out = code_and_output(square_swap_carry, (a,))
    torch.testing.assert_close(out, _ref_swap(a), rtol=1e-4, atol=1e-4)


@requires_cuda
def test_square_partial_k_tile() -> None:
    k = torch.randn(3, 40, 40, device=DEVICE, dtype=torch.bfloat16)
    code, out = code_and_output(square_gram_tail, (k,))
    assert "ptp_thread" in code
    torch.testing.assert_close(out, k.float() @ k.float().transpose(-1, -2))


@requires_cuda
def test_delta_rule_wy_matches_eager() -> None:
    from examples.linear import linear_attention_engine as engine

    torch.manual_seed(0)
    k = torch.nn.functional.normalize(
        torch.randn(16, 64, 64, device=DEVICE), dim=-1
    ).bfloat16()
    v = torch.randn(16, 64, 64, device=DEVICE).bfloat16()
    beta = torch.rand(16, 64, device=DEVICE).bfloat16()
    kernel = helion.kernel(
        engine.chunk_fwd_wy_delta_helion.fn, backend="cute", static_shapes=True
    )
    bound = kernel.bind((k, v, beta))
    result = bound.compile_config(bound.config_spec.default_config())(k, v, beta)
    reference = helion.kernel(
        engine.chunk_fwd_wy_delta_helion.fn, ref_mode=helion.RefMode.EAGER
    )(k, v, beta)
    for actual, expected in zip(result, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-4)


@requires_cuda
def test_reduction_collision_matches_eager() -> None:
    x = torch.randn(2, 32, 32, device=DEVICE)
    _, out = code_and_output(square_reduce, (x,))
    torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-5)


def test_shared_budget_rejects_oversized_root() -> None:
    from unittest.mock import patch

    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    with (
        _cpu_codegen(),
        patch.object(CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=1),
    ):
        k = torch.empty(2, 64, 64, dtype=torch.bfloat16)
        bound = square_swap_carry._bind_isolated((k,))
        with pytest.raises(exc.BackendUnsupported, match="shared bytes exceed"):
            bound.to_code(bound.config_spec.default_config())


def test_non_collision_kernel_never_plans() -> None:
    # Route ownership: equal tile *sizes* on distinct tiles are not collisions.
    @helion.kernel(backend="cute", static_shapes=True)
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, kk = a.size()
        _, n = b.size()
        out = torch.empty([m, n], dtype=torch.float32, device=a.device)
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(kk):
                acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
            out[tm, tn] = acc
        return out

    with _cpu_codegen():
        a = torch.empty(64, 64)
        code = matmul._bind_isolated((a, a)).to_code(
            helion.Config(block_sizes=[32, 32, 32])
        )
    assert "ptp_" not in code


@helion.kernel(backend="cute", static_shapes=True)
def square_in_place_snapshot(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        target[tile, :, :] = old - old
        out[tile, :, :] = old
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_transpose_in_place(x: torch.Tensor) -> torch.Tensor:
    # One cooperative store phase must not read elements it is overwriting.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        x[tile, :, :] = x[tile, :, :].transpose(-1, -2)
        out[tile, :, :] = x[tile, :, :] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_loop_mutation(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    # A loop-invariant value and a loop carry both cross in-loop stores.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] * 10
        acc = hl.zeros([tile, x.size(1), x.size(2)], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
            x[tile, :, :] = x[tile, :, :] * 2
        out[tile, :, :] = old + acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_fresh_read_after_write(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    out2 = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        out[tile, :, :] = x[tile, :, :] + 1
        out2[tile, :, :] = out[tile, :, :].transpose(-1, -2) * 2
    return out2


@helion.kernel(backend="cute", static_shapes=True)
def square_gram_then_scale(k: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(k)
    for tile in hl.tile(k.size(0), block_size=1):
        kt = k[tile, :, :]
        gram = hl.dot(kt, kt.transpose(-2, -1))
        k[tile, :, :] = kt * 2
        out[tile, :, :] = gram + kt
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_bool_snapshot(m: torch.Tensor) -> torch.Tensor:
    out = torch.empty(m.shape, dtype=torch.float32, device=m.device)
    for tile in hl.tile(m.size(0), block_size=1):
        flag = m[tile, :, :]
        m[tile, :, :] = torch.logical_not(flag)
        out[tile, :, :] = flag.to(torch.float32) * 2
    return out


def _aliased(x: torch.Tensor, alias: str) -> torch.Tensor:
    if alias == "same":
        return x
    if alias == "view":
        return x.view_as(x)
    return torch.empty_like(x)


def _snapshot_reference(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    old = x + 1
    target.copy_(old - old)
    return old


def _cpu_source(kernel, args: tuple[torch.Tensor, ...]) -> str:
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        return bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize(
    ("alias", "snapshots"), [("distinct", 0), ("same", 1), ("view", 1)]
)
def test_host_snapshots_follow_storage_alias_proof(alias: str, snapshots: int) -> None:
    # Distinct arguments are disjoint only through the cache-keyed storage
    # matrix; one storage passed twice (or a view) must snapshot the load.
    x = torch.empty(2, 8, 8)
    source = _cpu_source(square_in_place_snapshot, (x, _aliased(x, alias)))
    assert "ptp_thread" in source
    assert source.count("alloc_smem") == snapshots


def test_fresh_read_after_write_needs_no_snapshot() -> None:
    source = _cpu_source(square_fresh_read_after_write, (torch.empty(2, 8, 8),))
    assert "ptp_thread" in source
    assert "alloc_smem" not in source


def test_host_snapshots_count_against_shared_capacity() -> None:
    from unittest.mock import patch

    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    def generate(limit: int) -> str:
        with (
            _cpu_codegen(),
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=limit
            ),
        ):
            bound = square_gram_then_scale._bind_isolated((torch.empty(2, 8, 8),))
            return bound.to_code(bound.config_spec.default_config())

    # The admitted 8x8 FP32 gram tile fits exactly; the snapshot of ``kt``
    # (another 256 bytes) must not bypass the capacity check.
    with pytest.raises(
        exc.BackendUnsupported,
        match=r"512 shared bytes exceed 256 \(256 for host-effect snapshots\)",
    ):
        generate(256)
    assert generate(512).count("alloc_smem") == 2


def test_bool_host_snapshot_requires_typed_restore() -> None:
    with _cpu_codegen():
        bound = square_bool_snapshot._bind_isolated(
            (torch.zeros(2, 8, 8, dtype=torch.bool),)
        )
        with pytest.raises(exc.BackendUnsupported, match="requires a typed restore"):
            bound.to_code(bound.config_spec.default_config())


@requires_cuda
@pytest.mark.parametrize("alias", ["distinct", "same", "view"])
def test_mutating_root_matches_eager(alias: str) -> None:
    x = torch.randn(2, 16, 16, device=DEVICE)
    target = _aliased(x, alias)
    x_ref = x.clone()
    target_ref = _aliased(x_ref, alias)
    out = output_only(square_in_place_snapshot, (x, target))
    torch.testing.assert_close(out, _snapshot_reference(x_ref, target_ref))
    torch.testing.assert_close(x, x_ref)
    torch.testing.assert_close(target, target_ref)


@requires_cuda
def test_aliasing_launch_does_not_reuse_disjoint_code() -> None:
    # The first launch binds with disjoint storages; the second passes one
    # storage twice.  The runtime storage-matrix guard must select new code.
    kernel = helion.kernel(
        square_in_place_snapshot.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    x = torch.randn(2, 16, 16, device=DEVICE)
    kernel(x.clone(), torch.empty_like(x))
    aliased, reference = x.clone(), x.clone()
    out = kernel(aliased, aliased)
    torch.testing.assert_close(out, _snapshot_reference(reference, reference))
    torch.testing.assert_close(aliased, reference)


@requires_cuda
def test_transpose_and_loop_mutations_match_eager() -> None:
    x = torch.randn(2, 16, 16, device=DEVICE)
    expected = x.clone()
    expected.copy_(expected.transpose(-1, -2).clone())
    torch.testing.assert_close(
        output_only(square_transpose_in_place, (x,)), expected * 2
    )
    torch.testing.assert_close(x, expected)

    x = torch.randn(2, 16, 16, device=DEVICE)
    steps = torch.zeros(3, device=DEVICE)
    reference = x.clone()
    old, acc = reference * 10, torch.zeros_like(reference)
    for _ in range(3):
        acc = acc + reference
        reference.mul_(2)
    torch.testing.assert_close(output_only(square_loop_mutation, (x, steps)), old + acc)
    torch.testing.assert_close(x, reference)


@pytest.mark.parametrize(
    "kernel,shape",
    [
        (square_mask_only, (2, 8)),
        (square_gram, (2, 8, 8)),
        (square_swap_carry, (2, 8, 8)),
        (square_gram_tail, (2, 8, 20)),
        (square_reduce, (2, 8, 8)),
    ],
)
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_positional_root_owns_its_launch_geometry(
    kernel, shape, num_warps: int
) -> None:
    with _cpu_codegen():
        bound = kernel._bind_isolated((torch.empty(shape),))
        config = bound.config_spec.default_config()
        config.config["num_warps"] = num_warps
        source = bound.to_code(config)
    assert "ptp_thread" in source
    assert f"block=({32 * num_warps}, 1, 1)" in source


@helion.kernel(backend="cute", static_shapes=True)
def square_nonzero_grid(beta: torch.Tensor) -> torch.Tensor:
    bhn, c = beta.shape
    c = hl.specialize(c)
    out = torch.empty([bhn, c, c], dtype=beta.dtype, device=beta.device)
    for tile in hl.tile(1, bhn - 1, block_size=2):
        idx = hl.arange(c)
        out[tile, :, :] = beta[tile, :, None] * (idx[:, None] > idx[None, :])
    return out


def test_noncanonical_grid_cannot_silently_change_origin() -> None:
    with _cpu_codegen():
        bound = square_nonzero_grid._bind_isolated((torch.empty(8, 8),))
        with pytest.raises(exc.BackendUnsupported, match="noncanonical grid origin"):
            bound.to_code(bound.config_spec.default_config())


# --------------------------------------------------------------------------
# Conditional host effects
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True)
def branch_snapshot(x: torch.Tensor, flag: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        if flag > 0:
            x[tile, :, :] = old * 0
        out[tile, :, :] = old
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_first_use_in_arm(x: torch.Tensor, flag: int) -> torch.Tensor:
    first = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        v = x[tile, :, :]
        if flag > 0:
            first[tile, :, :] = v * 2
            x[tile, :, :] = v * 0
        out[tile, :, :] = v + 1 + first[tile, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_else_writes(x: torch.Tensor, flag: int) -> torch.Tensor:
    side = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        if flag > 0:
            side[tile, :, :] = old
        else:
            x[tile, :, :] = old * 0
        out[tile, :, :] = old + side[tile, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_arm_local(x: torch.Tensor, flag: int) -> torch.Tensor:
    side = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        if flag > 0:
            w = x[tile, :, :]
            x[tile, :, :] = w * 0
            side[tile, :, :] = w
        else:
            u = x[tile, :, :]
            x[tile, :, :] = u + 1
            side[tile, :, :] = u * 2
        out[tile, :, :] = x[tile, :, :] + side[tile, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def static_arm_local(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    side = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        if mode:
            w = x[tile, :, :]
            x[tile, :, :] = w * 0
            side[tile, :, :] = w
        else:
            u = x[tile, :, :]
            x[tile, :, :] = u + 1
            side[tile, :, :] = u * 2
        out[tile, :, :] = x[tile, :, :] + side[tile, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def static_dead_store(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        if mode:
            x[tile, :, :] = old * 0
        out[tile, :, :] = old
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_loop_mutation(
    x: torch.Tensor, steps: torch.Tensor, flag: int
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] * 10
        acc = hl.zeros([tile, x.size(1), x.size(2)], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
            if flag > 0:
                x[tile, :, :] = x[tile, :, :] * 2
        out[tile, :, :] = old + acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_fresh_accumulate(
    x: torch.Tensor, steps: torch.Tensor, flag: int
) -> torch.Tensor:
    acc = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        for _ in hl.tile(steps.size(0), block_size=1):
            if flag > 0:
                acc[tile, :, :] = acc[tile, :, :] + x[tile, :, :]
        out[tile, :, :] = acc[tile, :, :] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True)
def branch_cross_argument(
    x: torch.Tensor, target: torch.Tensor, flag: int
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        if flag > 0:
            target[tile, :, :] = old * 0
        out[tile, :, :] = old
    return out


@helion.kernel(backend="cute", static_shapes=True)
def out_kwarg_factory(x: torch.Tensor, y: torch.Tensor, flag: int) -> torch.Tensor:
    fresh = torch.zeros_like(x)
    reused = torch.zeros(x.shape, dtype=x.dtype, device=x.device, out=y)
    for tile in hl.tile(x.size(0), block_size=1):
        old = x[tile, :, :] + 1
        if flag > 0:
            reused[tile, :, :] = old * 0
        fresh[tile, :, :] = old
    return fresh


def _code_with_capacity(kernel, args: tuple[object, ...], capacity: int) -> str:
    from unittest.mock import patch

    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

    # ``args`` stay referenced by the caller: the runtime storage-matrix proof
    # reads live argument values through weak references during codegen.
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        bound = kernel._bind_isolated(args)
        return bound.to_code(bound.config_spec.default_config())


def test_runtime_branch_snapshots_preceding_loads_before_the_branch() -> None:
    args = (torch.empty(2, 8, 8), 1)
    source = _cpu_source(branch_snapshot, args)
    lines = source.splitlines()
    branch = next(i for i, line in enumerate(lines) if "if flag > 0:" in line)
    snapshot = next(
        i for i, line in enumerate(lines) if "ptp_tile_" in line and "] = " in line
    )
    assert source.count("alloc_smem") == 1
    assert snapshot < branch  # the read is fixed before either arm can store


@pytest.mark.parametrize(
    ("alias", "snapshots"), [("distinct", 0), ("same", 1), ("view", 1)]
)
def test_branch_writes_follow_storage_alias_proof(alias: str, snapshots: int) -> None:
    x = torch.empty(2, 8, 8)
    target = _aliased(x, alias)
    source = _cpu_source(branch_cross_argument, (x, target, 1))
    assert "if flag > 0:" in source
    assert source.count("alloc_smem") == snapshots


def test_static_dead_arm_emits_no_snapshot() -> None:
    x = torch.empty(2, 8, 8)
    dead = _cpu_source(static_dead_store, (x, False))
    live = _cpu_source(static_dead_store, (x, True))
    # The dead arm's store is never emitted, so no read needs a snapshot.
    assert "alloc_smem" not in dead
    assert dead.count(".store(") == 1  # only ``out``
    assert live.count("alloc_smem") == 1
    assert live.count(".store(") == 2


def test_branch_snapshots_count_exact_shared_capacity() -> None:
    # Each runtime arm snapshots one 8x8 FP32 tile; both are allocated
    # statically outside the branch and both count against capacity.
    args = (torch.empty(2, 8, 8), 1)
    with pytest.raises(
        exc.BackendUnsupported,
        match=r"512 shared bytes exceed 256 \(512 for host-effect snapshots\)",
    ):
        _code_with_capacity(branch_arm_local, args, 256)
    assert _code_with_capacity(branch_arm_local, args, 512).count("alloc_smem") == 2
    # A config-selected branch emits, and counts, only its live arm.
    static_args = (torch.empty(2, 8, 8), True)
    assert (
        _code_with_capacity(static_arm_local, static_args, 256).count("alloc_smem") == 1
    )
    with pytest.raises(
        exc.BackendUnsupported,
        match=r"256 shared bytes exceed 128 \(256 for host-effect snapshots\)",
    ):
        _code_with_capacity(static_arm_local, static_args, 128)


def test_out_kwarg_factory_is_not_fresh_provenance() -> None:
    from helion.language import _tracing_ops

    x, y = torch.empty(2, 8, 8), torch.empty(2, 8, 8)
    with _cpu_codegen():
        bound = out_kwarg_factory._bind_isolated((x, y, 1))
        env = bound.env
        hosts = {
            node.args[0]: node.meta["val"].untyped_storage()
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is _tracing_ops._host_tensor
        }
    # ``out=y`` writes the argument storage: it must not become a disjoint
    # fresh allocation.  Initializing factories stay out of the native set.
    assert hosts["y"] not in env.fresh_initialized_storages
    assert hosts["y"] not in env.fresh_allocation_storages
    assert hosts["fresh"] in env.fresh_initialized_storages
    assert hosts["fresh"] not in env.fresh_allocation_storages


@requires_cuda
@pytest.mark.parametrize("flag", [0, 1])
def test_conditional_host_effects_match_eager(flag: int) -> None:
    x = torch.randn(2, 16, 16, device=DEVICE)
    reference = x.clone()
    old = reference + 1
    if flag:
        reference.copy_(old * 0)
    torch.testing.assert_close(output_only(branch_snapshot, (x, flag)), old)
    torch.testing.assert_close(x, reference)

    x = torch.randn(2, 16, 16, device=DEVICE)
    v = x.clone()
    first = v * 2 if flag else torch.zeros_like(v)
    expected_x = v * 0 if flag else v.clone()
    torch.testing.assert_close(
        output_only(branch_first_use_in_arm, (x, flag)), v + 1 + first
    )
    torch.testing.assert_close(x, expected_x)

    x = torch.randn(2, 16, 16, device=DEVICE)
    old = x + 1
    side = old if flag else torch.zeros_like(old)
    expected_x = x.clone() if flag else old * 0
    torch.testing.assert_close(output_only(branch_else_writes, (x, flag)), old + side)
    torch.testing.assert_close(x, expected_x)

    x = torch.randn(2, 16, 16, device=DEVICE)
    before = x.clone()
    expected_x = before * 0 if flag else before + 1
    side = before if flag else before * 2
    torch.testing.assert_close(
        output_only(branch_arm_local, (x, flag)), expected_x + side
    )
    torch.testing.assert_close(x, expected_x)


@requires_cuda
@pytest.mark.parametrize("flag", [0, 1])
def test_conditional_loop_effects_match_eager(flag: int) -> None:
    steps = torch.zeros(3, device=DEVICE)
    x = torch.randn(2, 16, 16, device=DEVICE)
    reference = x.clone()
    old, acc = reference * 10, torch.zeros_like(reference)
    for _ in range(3):
        acc = acc + reference
        if flag:
            reference.mul_(2)
    torch.testing.assert_close(
        output_only(branch_loop_mutation, (x, steps, flag)), old + acc
    )
    torch.testing.assert_close(x, reference)

    x = torch.randn(2, 16, 16, device=DEVICE)
    expected = x * 3 * 2 if flag else torch.zeros_like(x)
    torch.testing.assert_close(
        output_only(branch_fresh_accumulate, (x, steps, flag)), expected
    )


@requires_cuda
@pytest.mark.parametrize("alias", ["distinct", "same", "view"])
def test_conditional_cross_argument_matches_eager(alias: str) -> None:
    for flag in (0, 1):
        x = torch.randn(2, 16, 16, device=DEVICE)
        target = _aliased(x, alias)
        x_ref = x.clone()
        target_ref = target.clone() if alias == "distinct" else _aliased(x_ref, alias)
        old = x_ref + 1
        if flag:
            target_ref.copy_(old * 0)
        torch.testing.assert_close(
            output_only(branch_cross_argument, (x, target, flag)), old
        )
        torch.testing.assert_close(x, x_ref)
        torch.testing.assert_close(target, target_ref)


@helion.kernel(backend="cute", static_shapes=True)
def shared_grid_canonical(a: torch.Tensor) -> torch.Tensor:
    # Two grid tiles take one registered block size: one canonical id.
    n = a.size(0)
    out = torch.empty([n, n], dtype=torch.float32, device=a.device)
    block = hl.register_block_size(16, 16)
    for tm, tn in hl.tile([n, n], block_size=[block, block]):
        out[tm, tn] = a[tm, tn] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def shared_loop_canonical(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # The device-loop tile shares the grid tile's canonical id.
    n = q.size(0)
    out = torch.empty([n, n], dtype=torch.float32, device=q.device)
    block = hl.register_block_size(16, 16)
    for tq in hl.tile(n, block_size=block):
        qt = q[tq, :]
        for tk in hl.tile(n, block_size=block):
            out[tq, tk] = hl.dot(qt, k[tk, :].transpose(0, 1))
    return out


@pytest.mark.parametrize(
    "kernel,shapes,message",
    [
        (shared_grid_canonical, [(32, 32)], "grid tiles share a canonical block id"),
        (
            shared_loop_canonical,
            [(32, 16), (32, 16)],
            "loop tile shares an active canonical block id",
        ),
    ],
)
def test_shared_canonical_tiles_are_rejected(kernel, shapes, message: str) -> None:
    # Coordinates are keyed by canonical id; sharing one used to rebind an
    # origin and silently compute the wrong blocks.
    with _cpu_codegen():
        bound = kernel._bind_isolated(tuple(torch.empty(s) for s in shapes))
        with pytest.raises(exc.BackendUnsupported, match=message):
            bound.to_code(bound.config_spec.default_config())


@helion.kernel(backend="cute", static_shapes=True)
def padded_exp_sum(x: torch.Tensor) -> torch.Tensor:
    # C=40 full dims evaluate 64 positions; exp(0) at padding is not zero.
    bhn, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([bhn, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(bhn, block_size=1):
        out[tile, :] = torch.exp(x[tile, :, :].float()).sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def padded_exp_gram(k: torch.Tensor) -> torch.Tensor:
    # Both dot operands are nonzero at padded K positions.
    bhn, c = k.size(0), hl.specialize(k.size(1))
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=k.device)
    for tile in hl.tile(bhn, block_size=1):
        e = torch.exp(k[tile, :, :].float())
        out[tile, :, :] = hl.dot(e, e.transpose(-2, -1))
    return out


@helion.kernel(backend="cute", static_shapes=True)
def exp_gram_k_tail(k: torch.Tensor) -> torch.Tensor:
    # D=40 with 16-wide K tiles: transformed operands are nonzero in the tail.
    bhn, c, d = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([bhn, c, c], dtype=torch.float32, device=k.device)
    for tile in hl.tile(bhn, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for tk in hl.tile(d, block_size=16):
            e = torch.exp(k[tile, :, tk].float())
            acc = hl.dot(e, e.transpose(-2, -1), acc=acc)
        out[tile, :, :] = acc
    return out


def _accumulation_lines(source: str, prefix: str) -> list[str]:
    # Self-referencing updates (``acc = acc + ...``), not initializations.
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(prefix) and line.count(prefix) >= 2
    ]


@pytest.mark.parametrize(
    "kernel,prefix",
    [
        (padded_exp_sum, "ptp_sum_"),
        (padded_exp_gram, "ptp_acc_"),
        (padded_reverse_exp_cumsum, "ptp_scan_seen_"),
    ],
)
def test_padded_full_dims_are_masked(kernel, prefix: str) -> None:
    # Every accumulation (and the reverse-scan seed predicate) over the
    # 64-wide padded C=40 axis is bounded by 40; power-of-two C needs no bound.
    padded = _cpu_source(kernel, (torch.empty(2, 40, 40),))
    assert "ptp_thread" in padded
    lines = _accumulation_lines(padded, prefix)
    assert lines and all("< 40" in line for line in lines), lines
    exact = _cpu_source(kernel, (torch.empty(2, 32, 32),))
    lines = _accumulation_lines(exact, prefix)
    assert lines and not any("< 32" in line for line in lines), lines


def test_dot_k_tail_is_masked() -> None:
    # The K loop tail bound guards each contraction term, not only the loads.
    source = _cpu_source(exp_gram_k_tail, (torch.empty(2, 16, 40),))
    assert "ptp_thread" in source
    lines = _accumulation_lines(source, "ptp_acc_")
    assert lines and all(" if " in line for line in lines), lines


@requires_cuda
def test_padded_full_dims_match_eager() -> None:
    x = torch.randn(3, 40, 40, device=DEVICE) / 4
    code, out = code_and_output(padded_exp_sum, (x,))
    assert "ptp_thread" in code
    torch.testing.assert_close(out, torch.exp(x).sum(-1), rtol=1e-5, atol=1e-4)
    code, out = code_and_output(padded_exp_gram, (x,))
    assert "ptp_thread" in code
    e = torch.exp(x)
    torch.testing.assert_close(out, e @ e.transpose(-1, -2), rtol=1e-5, atol=1e-4)
    k = torch.randn(3, 16, 40, device=DEVICE) / 4
    _, out = code_and_output(exp_gram_k_tail, (k,))
    e = torch.exp(k)
    torch.testing.assert_close(out, e @ e.transpose(-1, -2), rtol=1e-5, atol=1e-4)


@helion.kernel(backend="cute", static_shapes=True)
def square_broadcast_store(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    rows = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        value = x[tile, :, :]
        out[tile, :, :] = hl.dot(value, value.transpose(-2, -1))
        rows[tile, :, :] = value.sum(-1, keepdim=True)
    return out, rows


@helion.kernel(backend="cute", static_shapes=True)
def square_scalar_store(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    filled = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        value = x[tile, :, :]
        out[tile, :, :] = hl.dot(value, value.transpose(-2, -1))
        filled[tile, :, :] = 3.0
    return out, filled


@pytest.mark.parametrize(
    "kernel,message",
    [
        (square_broadcast_store, "broadcast-valued store"),
        (square_scalar_store, "scalar store"),
    ],
)
def test_unsupported_store_domain_fails_closed(kernel, message: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match=message):
        _cpu_source(kernel, (torch.empty(2, 8, 8),))


@helion.kernel(backend="cute", static_shapes=True)
def square_grid_signed_mod(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for index in hl.grid(x.size(0)):
        out[index, :, :] = x[(index - 1) % x.size(0), :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_grid_signed_div(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for index in hl.grid(x.size(0)):
        out[index, :, :] = x[(index - 1) // 2 + 1, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_tile_signed_mod(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        out[tile, :, :] = x[(tile.id - 1) % x.size(0), :, :][None, :, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def square_tile_signed_div(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=1):
        out[tile, :, :] = x[(tile.id - 1) // 2 + 1, :, :][None, :, :]
    return out


@requires_cuda
@pytest.mark.parametrize(
    "kernel,divide",
    [
        (square_grid_signed_mod, False),
        (square_tile_signed_mod, False),
        (square_grid_signed_div, True),
        (square_tile_signed_div, True),
    ],
)
def test_signed_grid_scalar_index_matches_python(kernel, divide: bool) -> None:
    # At index zero, C-style truncation/remainder selects the wrong row.
    x = torch.arange(3 * 8 * 8, device=DEVICE, dtype=torch.float32).reshape(3, 8, 8)
    code, out = code_and_output(kernel, (x,))
    assert "ptp_thread" in code
    index = torch.arange(3, device=DEVICE)
    source = (index - 1) // 2 + 1 if divide else (index - 1) % 3
    torch.testing.assert_close(out, x[source], rtol=0, atol=0)


def test_padded_literal_carry_axes_use_positional_coordinates() -> None:
    # The carry's two C=40 axes become literal 64s, losing block-id metadata.
    code = _cpu_source(
        square_gram_tail, (torch.empty(3, 40, 40, dtype=torch.bfloat16),)
    )
    assert "ptp_thread" in code


def test_padded_index_store_keeps_linear_score_route() -> None:
    from examples.linear import linear_attention_engine as engine

    kernel = helion.kernel(
        engine.chunk_fwd_A_diag_anchored_helion.fn, backend="cute", static_shapes=True
    )
    q, k, g = (torch.empty(2, 64, 32) for _ in range(3))
    code = _cpu_source(kernel, (q, k, g, 0.125, False))
    assert "ptp_thread" in code


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_positional_views_need_only_element_alignment(dtype: torch.dtype) -> None:
    x = torch.randn(2 * 16 * 16 + 1, device=DEVICE, dtype=dtype)[1:].view(2, 16, 16)
    target = torch.randn(2 * 16 * 16 + 3, device=DEVICE, dtype=dtype)[3:].view(
        2, 16, 16
    )
    assert x.data_ptr() % 16 and target.data_ptr() % 16
    expected = x + 1
    code, out = code_and_output(square_in_place_snapshot, (x, target))
    assert "_helion_cute_pointer_alignment = 1" in code
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(target, torch.zeros_like(target))


# --------------------------------------------------------------------------
# Logical extents of literal padded dims
# --------------------------------------------------------------------------
# ``hl.zeros([tile, 40, 40])`` and ``x[tile, :, hl.arange(48)]`` carry literal
# 64s in their FakeTensor shapes.  Their logical lengths come from FX
# provenance (factory shape, iota length, host slice) through loop carries,
# branch ports, views, reductions and scans.


@helion.kernel(backend="cute", static_shapes=True)
def literal_carry_sum(
    x: torch.Tensor, steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = x.size(0), hl.specialize(x.size(1))
    rows = torch.empty([b, c], dtype=torch.float32, device=x.device)
    square = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        rows[tile, :] = torch.exp(acc).sum(-1)
        square[tile, :, :] = acc
    return rows, square


@helion.kernel(backend="cute", static_shapes=True)
def literal_carry_reverse_scan(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        out[tile, :, :] = hl.cumsum(torch.exp(acc), dim=2, reverse=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def literal_carry_gram(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        e = torch.exp(acc)
        out[tile, :, :] = hl.dot(e, e.transpose(-2, -1))
    return out


@helion.kernel(backend="cute", static_shapes=True)
def literal_index_reductions(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = x.size(0), hl.specialize(x.size(1))
    rows = torch.empty([b, c], dtype=torch.float32, device=x.device)
    gram = torch.empty_like(x)
    for tile in hl.tile(b, block_size=1):
        e = torch.exp(x[tile, :, hl.arange(48)])
        rows[tile, :] = e.sum(-1)
        gram[tile, :, :] = hl.dot(e, e.transpose(-2, -1))
    return rows, gram


@helion.kernel(backend="cute", static_shapes=True)
def literal_carry_tuple_scan(
    x: torch.Tensor, steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = x.size(0), hl.specialize(x.size(1))
    scale = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    shift = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        e = torch.exp(acc)
        a, s = hl.associative_scan(affine, (e * 0.5, e), dim=2, reverse=True)
        scale[tile, :, :] = a
        shift[tile, :, :] = s
    return scale, shift


@helion.kernel(backend="cute", static_shapes=True)
def literal_branch_port_sum(
    x: torch.Tensor, steps: torch.Tensor, flag: int
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = x.size(0), hl.specialize(x.size(1))
    rows = torch.zeros([b, c], dtype=torch.float32, device=x.device)
    square = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        e = torch.exp(acc)
        if flag > 0:
            rows[tile, :] = e.sum(-1)
        square[tile, :, :] = acc
    return rows, square


@helion.kernel(backend="cute", static_shapes=True)
def literal_keepdim_expand(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    b, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([b, c, c], dtype=torch.float32, device=x.device)
    for tile in hl.tile(b, block_size=1):
        acc = hl.zeros([tile, c, c], dtype=torch.float32)
        for _ in hl.tile(steps.size(0), block_size=1):
            acc = acc + x[tile, :, :]
        e = torch.exp(acc)
        normalized = (e / e.sum(-1, keepdim=True)).transpose(-2, -1)
        # ``e.shape`` is the padded 64: an expanded dim takes its co-operand's
        # logical length instead of asserting one.
        out[tile, :, :] = normalized.sum(-1, keepdim=True).expand(e.shape) + e
    return out


def _literal_reference(x: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.exp(x * steps)


def _reverse_affine(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
    scale, shift = torch.empty_like(a), torch.empty_like(b)
    sa, sb = a[..., -1], b[..., -1]
    scale[..., -1], shift[..., -1] = sa, sb
    for j in range(a.shape[-1] - 2, -1, -1):
        sa, sb = sa * a[..., j], sb * a[..., j] + b[..., j]
        scale[..., j], shift[..., j] = sa, sb
    return scale, shift


@pytest.mark.parametrize(
    "kernel,args,prefixes",
    [
        (
            literal_carry_sum,
            lambda c: (torch.empty(2, c, c), torch.empty(2)),
            ("ptp_sum_",),
        ),
        (
            literal_carry_reverse_scan,
            lambda c: (torch.empty(2, c, c), torch.empty(2)),
            ("ptp_scan_seen_",),
        ),
        (
            literal_carry_gram,
            lambda c: (torch.empty(2, c, c), torch.empty(2)),
            ("ptp_acc_",),
        ),
        (
            literal_carry_tuple_scan,
            lambda c: (torch.empty(2, c, c), torch.empty(2)),
            ("ptp_scan_seen_",),
        ),
        (
            literal_branch_port_sum,
            lambda c: (torch.empty(2, c, c), torch.empty(2), 1),
            ("ptp_sum_",),
        ),
    ],
)
def test_literal_padded_carry_dims_are_bounded(kernel, args, prefixes) -> None:
    # C=40 carries are literal 64-wide; each reduction, scan seed and dot K
    # term is bounded by 40.  An exact power-of-two C needs no bound.
    padded = _cpu_source(kernel, args(40))
    assert "ptp_thread" in padded
    for prefix in prefixes:
        lines = _accumulation_lines(padded, prefix)
        assert lines and all("< 40" in line for line in lines), (prefix, lines)
    exact = _cpu_source(kernel, args(32))
    for prefix in prefixes:
        lines = _accumulation_lines(exact, prefix)
        assert lines and not any("< 32" in line for line in lines), (prefix, lines)


def test_literal_padded_index_dims_are_bounded() -> None:
    # ``hl.arange(48)`` indexes a 64-wide host dim: columns 48..63 are real
    # host data, so the logical length must bound the load and reductions.
    source = _cpu_source(literal_index_reductions, (torch.empty(2, 64, 64),))
    assert "ptp_thread" in source
    for prefix in ("ptp_sum_", "ptp_acc_"):
        lines = _accumulation_lines(source, prefix)
        assert lines and all("< 48" in line for line in lines), (prefix, lines)


def test_expanded_padded_shape_uses_co_operand_extent() -> None:
    source = _cpu_source(
        literal_keepdim_expand, (torch.empty(2, 40, 40), torch.empty(2))
    )
    assert "ptp_thread" in source
    lines = _accumulation_lines(source, "ptp_sum_")
    assert lines and all("< 40" in line for line in lines), lines


@requires_cuda
@pytest.mark.parametrize("c", [40, 32])
def test_literal_padded_dims_match_eager(c: int) -> None:
    steps = torch.zeros(2, device=DEVICE)
    x = torch.randn(3, c, c, device=DEVICE) / 4
    e = _literal_reference(x, 2)
    tol = {"rtol": 1e-5, "atol": 1e-4}
    code, (rows, _) = code_and_output(literal_carry_sum, (x, steps))
    assert "ptp_thread" in code
    torch.testing.assert_close(rows, e.sum(-1), **tol)
    _, scan = code_and_output(literal_carry_reverse_scan, (x, steps))
    torch.testing.assert_close(scan, e.flip(-1).cumsum(-1).flip(-1), **tol)
    _, gram = code_and_output(literal_carry_gram, (x, steps))
    torch.testing.assert_close(gram, e @ e.transpose(-1, -2), **tol)
    x = torch.randn(3, c, c, device=DEVICE) / 8
    e = _literal_reference(x, 2)
    _, (scale, shift) = code_and_output(literal_carry_tuple_scan, (x, steps))
    expected = _reverse_affine(e * 0.5, e)
    torch.testing.assert_close(scale, expected[0], **tol)
    torch.testing.assert_close(shift, expected[1], **tol)
    for flag in (0, 1):
        x = torch.randn(3, c, c, device=DEVICE) / 4
        e = _literal_reference(x, 2)
        _, (rows, _) = code_and_output(literal_branch_port_sum, (x, steps, flag))
        torch.testing.assert_close(
            rows, e.sum(-1) if flag else torch.zeros_like(rows), **tol
        )
    x = torch.randn(3, c, c, device=DEVICE) / 4
    e = _literal_reference(x, 2)
    _, out = code_and_output(literal_keepdim_expand, (x, steps))
    normalized = (e / e.sum(-1, keepdim=True)).transpose(-1, -2)
    torch.testing.assert_close(
        out, normalized.sum(-1, keepdim=True).expand_as(e) + e, **tol
    )


@requires_cuda
def test_literal_padded_index_matches_eager() -> None:
    x = torch.randn(3, 64, 64, device=DEVICE) / 4
    code, (rows, gram) = code_and_output(literal_index_reductions, (x,))
    assert "ptp_thread" in code
    e = torch.exp(x[:, :, :48])
    tol = {"rtol": 1e-5, "atol": 1e-4}
    torch.testing.assert_close(rows, e.sum(-1), **tol)
    torch.testing.assert_close(gram, e @ e.transpose(-1, -2), **tol)


# --------------------------------------------------------------------------
# Contractions over a synthetic reduction-lane K
# --------------------------------------------------------------------------
# The delta-rule state pass contracts over C=64 (``k_i.T @ v_new``) inside a
# serial ``hl.grid`` carry.  When D != C nothing collides, but the ordinary
# layout splits C across threads and a synthetic lane loop wrapping the whole
# root: each lane carried its own partial state (step 0 exact, later steps
# wrong).  The positional root owns such contractions.


def _delta_state_args(d: int, device: object = "cpu") -> tuple[object, ...]:
    gen = torch.Generator(device=device).manual_seed(0)

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, device=device, generator=gen) * 0.25).bfloat16()

    bh, n, c = 2, 3, 64
    k, w, u, h0 = (
        rand(bh, n, c, d),
        rand(bh, n, c, d),
        rand(bh, n, c, d),
        rand(bh, d, d),
    )
    g_cs = (-torch.rand(bh, n, c, device=device, generator=gen) * 0.05).cumsum(-1)
    return k, w, u, h0, g_cs, g_cs[:, :, -1].contiguous(), True


def _delta_state_reference(k, w, u, h0, g_cs, last, _scalar_decay):
    from examples.linear import linear_attention_engine as engine

    state, states, corrected = h0.float(), [], []
    for i in range(k.size(1)):
        states.append(state.bfloat16())
        v_new = u[:, i].float() - w[:, i].float() @ state.bfloat16().float()
        corrected.append(v_new.bfloat16())
        dl = last[:, i]
        k_i = (
            k[:, i].float()
            * torch.exp2((dl[:, None] - g_cs[:, i]) * engine.RCP_LN2)[..., None]
        )
        state = state * torch.exp2(dl * engine.RCP_LN2)[:, None, None] + (
            k_i.bfloat16().float().transpose(-1, -2) @ v_new.bfloat16().float()
        )
    return torch.stack(states, 1), torch.stack(corrected, 1)


def _delta_state_kernel():
    from examples.linear import linear_attention_engine as engine

    return helion.kernel(
        engine.chunk_fwd_h_delta_helion.fn, backend="cute", static_shapes=True
    )


@pytest.mark.parametrize("d", [32, 128])
def test_lane_split_contraction_takes_positional_route(d: int) -> None:
    source = _cpu_source(_delta_state_kernel(), _delta_state_args(d))
    assert "ptp_thread" in source


def test_lane_split_dot_fails_closed_without_positional(monkeypatch) -> None:
    # The scalar hl.dot fallback must never silently drop synthetic K lanes.
    from helion._compiler.cute import positional_root

    monkeypatch.setattr(
        positional_root, "find_lane_split_contractions", lambda graphs, df: ()
    )
    with pytest.raises(exc.BackendUnsupported, match="split across synthetic lanes"):
        _cpu_source(_delta_state_kernel(), _delta_state_args(32))


@requires_cuda
@pytest.mark.parametrize("d", [32, 128, 256])
def test_lane_split_contraction_matches_reference(d: int) -> None:
    args = _delta_state_args(d, DEVICE)
    code, (h_all, v_new) = code_and_output(_delta_state_kernel(), args)
    assert "ptp_thread" in code
    h_ref, v_ref = _delta_state_reference(*args)
    torch.testing.assert_close(h_all.float(), h_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_new.float(), v_ref.float(), rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------
# Direct-K aten.mm operands own their repeated axes
# --------------------------------------------------------------------------
# ``y[:, :]`` (K == N == 128) repeats a static axis, but the per-node direct mm
# lowering reads it by position from host memory.  Only when an operand is
# consumed solely by such mm nodes does the root stay with that lowering.


@helion.kernel(backend="cute", static_shapes=True)
def direct_mm_square_rhs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m, y.size(1)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m, block_size=32):
        out[tile_m, :] = x[tile_m, :] @ y[:, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def direct_mm_square_rhs_reused(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.size()
    out = torch.empty([m, y.size(1)], dtype=x.dtype, device=x.device)
    scaled = torch.empty_like(y)
    for tile_m in hl.tile(m, block_size=32):
        rhs = y[:, :]
        out[tile_m, :] = x[tile_m, :] @ rhs
        scaled[:, :] = rhs * 2
    return out, scaled


def _direct_mm_args() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(64, 32, dtype=torch.float16), torch.empty(
        32, 32, dtype=torch.float16
    )


def test_direct_mm_repeated_axis_operand_keeps_per_node_lowering() -> None:
    source = _cpu_source(direct_mm_square_rhs, _direct_mm_args())
    assert "ptp_thread" not in source
    assert "dot_serial_result" in source or "cute.gemm" in source


def test_repeated_axis_operand_with_other_users_stays_positional() -> None:
    # ``rhs`` also feeds a pointwise store, so its diagonal-only per-thread
    # value would be wrong: the root must not fall back to the ordinary direct
    # mm path; the positional root lowers the mm and the store by position.
    source = _cpu_source(direct_mm_square_rhs_reused, _direct_mm_args())
    assert "ptp_thread" in source
    assert "dot_serial_result" not in source
    assert "cute.gemm" not in source


@requires_cuda
def test_repeated_axis_operand_with_other_users_matches_eager() -> None:
    x = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    y = torch.randn(32, 32, device=DEVICE, dtype=torch.float16)
    code, (out, scaled) = code_and_output(direct_mm_square_rhs_reused, (x, y))
    assert "ptp_thread" in code
    torch.testing.assert_close(out, x @ y, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(scaled, y * 2)


def test_direct_mm_owner_fails_closed_off_the_direct_path(monkeypatch) -> None:
    from helion._compiler.cute import aten_lowering

    monkeypatch.setattr(
        aten_lowering, "cute_static_serial_matmul_k_extent", lambda lhs, rhs: None
    )
    with pytest.raises(
        exc.BackendUnsupported, match="repeated-axis operand requires the direct mm"
    ):
        _cpu_source(direct_mm_square_rhs, _direct_mm_args())


@helion.kernel(backend="cute")
def stack_store_2d(dev_ptrs: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
    # ``ptrs[:, :]`` repeats a static axis, but the native 2D stack store
    # loops its second axis and re-reads ``dev_ptrs`` by position.
    m1, m2 = dev_ptrs.size()
    n = example.size(0)
    out = torch.empty(m1, m2, n, dtype=torch.bfloat16, device=dev_ptrs.device)
    for tile in hl.tile(n, block_size=4):
        ptr_tile = dev_ptrs[:, :]
        tensors = hl.stacktensor_like(example, ptr_tile)
        out[:, :, tile] = tensors[tile]
    return out


@helion.kernel(backend="cute")
def stack_store_2d_plus_one(
    dev_ptrs: torch.Tensor, example: torch.Tensor
) -> torch.Tensor:
    # The stack value reaches the store through a pointwise op: no owner.
    m1, m2 = dev_ptrs.size()
    n = example.size(0)
    out = torch.empty(m1, m2, n, dtype=torch.bfloat16, device=dev_ptrs.device)
    for tile in hl.tile(n, block_size=4):
        ptr_tile = dev_ptrs[:, :]
        tensors = hl.stacktensor_like(example, ptr_tile)
        out[:, :, tile] = tensors[tile] + 1
    return out


def _stack_args() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(4, 4, dtype=torch.uint64), torch.zeros(4, dtype=torch.bfloat16)


def test_native_stack_store_pointer_tile_is_not_a_collision() -> None:
    with _cpu_codegen():
        bound = stack_store_2d._bind_isolated(_stack_args())
        code = bound.to_code(bound.config_spec.default_config())
    assert "ptp_" not in code
    assert "for stack_dim in range(4):" in code
    assert "cutlass.Int32(stack_dim) * cutlass.Int32(1)).load()" in code


def test_native_stack_store_owner_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion._compiler.cute import memory_ops

    monkeypatch.setattr(memory_ops, "_codegen_cute_store_stack_load", lambda *a: None)
    with _cpu_codegen():
        bound = stack_store_2d._bind_isolated(_stack_args())
        with pytest.raises(exc.BackendUnsupported, match="native stack lowering"):
            bound.to_code(bound.config_spec.default_config())


def test_unowned_repeated_axis_stack_load_is_rejected() -> None:
    with _cpu_codegen():
        bound = stack_store_2d_plus_one._bind_isolated(_stack_args())
        with pytest.raises(exc.BackendUnsupported, match="stack tensor access"):
            bound.to_code(bound.config_spec.default_config())


@requires_cuda
def test_native_stack_store_permuted_transposed_pointers() -> None:
    sources = [
        torch.arange(4, device=DEVICE, dtype=torch.bfloat16) + 4 * k for k in range(16)
    ]
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(0))
    table = torch.tensor(
        [sources[int(p)].data_ptr() for p in perm], device=DEVICE, dtype=torch.uint64
    )
    dev_ptrs = table.reshape(4, 4).t()  # strides (1, 4)
    expected_src = perm.reshape(4, 4).t()
    _, result = code_and_output(stack_store_2d, (dev_ptrs, sources[0]))
    expected = torch.stack(
        [
            torch.stack([sources[int(expected_src[i, j])] for j in range(4)])
            for i in range(4)
        ]
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@helion.kernel(backend="cute")
def stack_store_2d_masked(
    dev_ptrs: torch.Tensor, example: torch.Tensor
) -> torch.Tensor:
    m1, m2 = dev_ptrs.size()
    n = example.size(0)
    out = torch.zeros(m1, m2, n, dtype=torch.bfloat16, device=dev_ptrs.device)
    for tile in hl.tile(n, block_size=4):
        tensors = hl.stacktensor_like(example, dev_ptrs[:, :])
        out[:, :, tile] = hl.load(tensors, [tile], extra_mask=tile.index < 2)
    return out


def test_native_stack_store_does_not_drop_load_mask() -> None:
    with _cpu_codegen():
        bound = stack_store_2d_masked._bind_isolated(_stack_args())
        with pytest.raises(exc.BackendUnsupported, match="stack tensor access"):
            bound.to_code(bound.config_spec.default_config())


# --------------------------------------------------------------------------
# Batched contractions (aten.bmm / aten.baddbmm)
# --------------------------------------------------------------------------
# These share the hl.dot contraction lowering: batch coordinates broadcast,
# K is bounded by its logical extent (padded full dim or loop tail), and the
# baddbmm input joins the product sum with weight one.


@helion.kernel(backend="cute", static_shapes=True)
def baddbmm_full_slice_loop(q: torch.Tensor) -> torch.Tensor:
    # test_indexing.py::test_full_slice_in_reduction_loop: attn is [t, C, C].
    n, c, d = q.size(0), q.size(1), q.size(2)
    out = torch.empty([n, c], dtype=q.dtype, device=q.device)
    for (tile_n,) in hl.tile([n]):
        attn = hl.zeros([tile_n, c, c], dtype=torch.float32)
        for tile_d in hl.tile(d):
            qt = q[tile_n, :, tile_d]
            attn = torch.baddbmm(attn, qt, qt.transpose(-2, -1))
        out[tile_n, :] = attn.sum(-1).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def baddbmm_nonsquare(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # [n, C, D] @ [n, D, E] + bias with C != E and a padded K (D=20 -> 32);
    # ``sq`` adds a square value so the positional root owns the kernel.
    n, c, e = a.size(0), a.size(1), b.size(2)
    out = torch.empty([n, c, e], dtype=torch.float32, device=a.device)
    square = torch.empty_like(sq)
    for tile_n in hl.tile(n):
        out[tile_n, :, :] = torch.baddbmm(
            bias[tile_n, :, :], a[tile_n, :, :], b[tile_n, :, :]
        )
        square[tile_n, :, :] = sq[tile_n, :, :].transpose(-2, -1) * 2
    return out, square


@helion.kernel(backend="cute", static_shapes=True)
def bmm_square(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, c = x.size(0), hl.specialize(x.size(1))
    out = torch.empty([n, c, c], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n, :, :] = torch.bmm(x[tile_n, :, :], y[tile_n, :, :])
    return out


def _contraction_source(kernel, args, block_sizes) -> str:
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = block_sizes
        return bound.to_code(config)


def _contraction_terms(source: str) -> list[str]:
    return [
        line.strip()
        for line in source.splitlines()
        if re.search(r"ptp_acc_\d+ = ptp_acc_\d+ \+", line)
    ]


@pytest.mark.parametrize(
    "kernel,args,block_sizes,guard",
    [
        # Batch tail (3 rows, tile 2) and a partial last K loop tile (20 / 16).
        (
            baddbmm_full_slice_loop,
            (torch.empty(3, 8, 20),),
            [2, 16],
            r"ptp_loop_\d+ \+ ptp_k_\d+ < ptp_end_\d+",
        ),
        (
            baddbmm_nonsquare,
            (
                torch.empty(3, 8, 20),
                torch.empty(3, 20, 6),
                torch.empty(3, 8, 6),
                torch.empty(3, 8, 8),
            ),
            [2],
            r"ptp_k_\d+ < 20",
        ),
        (
            bmm_square,
            (torch.empty(3, 8, 8, dtype=torch.bfloat16),) * 2,
            [2],
            None,
        ),
    ],
)
def test_batched_contraction_lowers_positionally(
    kernel, args, block_sizes, guard
) -> None:
    source = _contraction_source(kernel, args, block_sizes)
    assert "ptp_thread" in source
    (term,) = _contraction_terms(source)
    assert (re.search(guard, term) is not None) if guard else " if " not in term


@requires_cuda
def test_batched_contractions_match_eager() -> None:
    q = torch.randn(16, 16, 16, device=DEVICE)
    code, out = code_and_output(baddbmm_full_slice_loop, (q,), block_sizes=[16, 16])
    assert "ptp_thread" in code
    expected = torch.baddbmm(torch.zeros_like(q), q, q.transpose(-2, -1)).sum(-1)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    q = torch.randn(3, 8, 20, device=DEVICE)
    _, out = code_and_output(baddbmm_full_slice_loop, (q,), block_sizes=[2, 16])
    expected = torch.baddbmm(q.new_zeros(3, 8, 8), q, q.transpose(-2, -1)).sum(-1)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    a, b = torch.randn(3, 8, 20, device=DEVICE), torch.randn(3, 20, 6, device=DEVICE)
    bias, sq = torch.randn(3, 8, 6, device=DEVICE), torch.randn(3, 8, 8, device=DEVICE)
    _, (out, square) = code_and_output(
        baddbmm_nonsquare, (a, b, bias, sq), block_sizes=[2]
    )
    torch.testing.assert_close(out, torch.baddbmm(bias, a, b), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(square, sq.transpose(-2, -1) * 2)
    x = torch.randn(3, 8, 8, device=DEVICE).bfloat16()
    y = torch.randn(3, 8, 8, device=DEVICE).bfloat16()
    _, out = code_and_output(bmm_square, (x, y), block_sizes=[2])
    torch.testing.assert_close(out, torch.bmm(x, y), atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# Rank-2 aten.mm
# --------------------------------------------------------------------------


@helion.kernel(backend="cute")
def mm_padded_constants(x: torch.Tensor) -> torch.Tensor:
    # test_specialize.py::test_dynamic_size_block_non_power_of_two_matmul: the
    # K of acc @ acc2 is the explicit next_power_of_2 extent, not x.size(1).
    hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        acc = hl.full(
            [tile, helion.next_power_of_2(x.size(1))],
            1.0 / helion.next_power_of_2(x.size(1)),
        )
        acc2 = hl.full(
            [helion.next_power_of_2(x.size(1)), helion.next_power_of_2(x.size(1))],
            1.0,
        )
        acc = torch.matmul(acc, acc2)
        acc = x[tile, :] + acc + 1
        out[tile, :] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def mm_nonsquare(
    a: torch.Tensor, w: torch.Tensor, sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # [t, C] @ [C, E] with C != E and a padded K (C=20 -> 32); ``sq`` adds a
    # square value so the positional root owns the kernel.
    m, e = a.size(0), w.size(1)
    out = torch.empty([m, e], dtype=torch.float32, device=a.device)
    square = torch.empty_like(sq)
    for tile in hl.tile(m):
        out[tile, :] = torch.mm(a[tile, :], w[:, :]) * 2
        square[tile, :, :] = sq[tile, :, :].transpose(-2, -1)
    return out, square


def _mm_source(kernel, args, block_size: int) -> str:
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(
            helion.Config(block_sizes=[block_size])
        )


@pytest.mark.parametrize("c,padded", [(7, 8), (20, 32), (500, 512)])
def test_mm_explicit_padded_k_is_not_shrunk(c: int, padded: int) -> None:
    # The constant operands span the padded extent: K runs over all of it
    # unguarded, while the store is still bounded by the host extent.
    source = _mm_source(mm_padded_constants, (torch.empty(10, c),), 4)
    assert "ptp_thread" in source
    assert re.search(rf"for ptp_k_\d+ in cutlass\.range\({padded},", source)
    (term,) = _contraction_terms(source)
    assert " if " not in term


def test_mm_nonsquare_bounds_padded_k() -> None:
    args = (torch.empty(10, 20), torch.empty(20, 6), torch.empty(10, 4, 4))
    source = _mm_source(mm_nonsquare, args, 4)
    assert "ptp_thread" in source
    (term,) = _contraction_terms(source)
    assert re.search(r"if ptp_k_\d+ < 20 else", term)


@requires_cuda
def test_mm_positional_matches_eager() -> None:
    for c in (7, 20):
        x = torch.randn(10, c, device=DEVICE)
        code, out = code_and_output(mm_padded_constants, (x,), block_size=4)
        assert "ptp_thread" in code
        torch.testing.assert_close(out, x + 2)
    a, w = torch.randn(10, 20, device=DEVICE), torch.randn(20, 6, device=DEVICE)
    sq = torch.randn(10, 4, 4, device=DEVICE)
    _, (out, square) = code_and_output(mm_nonsquare, (a, w, sq), block_size=4)
    torch.testing.assert_close(out, (a @ w) * 2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(square, sq.transpose(-2, -1))


# --------------------------------------------------------------------------
# Contraction K defined by an operand spanning the padded extent
# --------------------------------------------------------------------------
# ``exp(x[tile, :])`` with C=20 is padded to 32 lanes, whose values are
# exp(0) = 1.  Against ``hl.full([32, 4], 1)`` the only definition is the
# padded one (eager rejects 20 x 32), and Helion's Triton codegen runs
# ``tl.dot`` over all 32 lanes, exactly like the elementwise ``e * full``.
# Two partial K extents, and an active loop tile K with a runtime end, keep
# their bounds.


@helion.kernel(backend="cute", static_shapes=True)
def k_promoted_contractions(
    x: torch.Tensor, sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, c = x.size(0), x.size(1)
    p = helion.next_power_of_2(c)
    rows = torch.empty([m], dtype=torch.float32, device=x.device)
    mm = torch.empty([m, 4], dtype=torch.float32, device=x.device)
    dot = torch.empty([m, 4], dtype=torch.float32, device=x.device)
    square = torch.empty_like(sq)
    for tile in hl.tile(m):
        e = torch.exp(x[tile, :])
        rows[tile] = (e * hl.full([tile, p], 1.0, dtype=torch.float32)).sum(-1)
        mm[tile, :] = torch.mm(e, hl.full([p, 4], 1.0, dtype=torch.float32))
        dot[tile, :] = hl.dot(e, hl.full([p, 4], 1.0, dtype=torch.float32))
        square[tile, :, :] = sq[tile, :, :].transpose(-2, -1)
    return rows, mm, dot, square


@helion.kernel(backend="cute", static_shapes=True)
def k_partial_contractions(
    x: torch.Tensor, w: torch.Tensor, sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, d = x.size(0), x.size(1)
    both = torch.empty([m, 4], dtype=torch.float32, device=x.device)
    tail = torch.empty([m, 4], dtype=torch.float32, device=x.device)
    square = torch.empty_like(sq)
    for tile in hl.tile(m):
        both[tile, :] = torch.mm(torch.exp(x[tile, :]), torch.exp(w[:, :]))
        acc = hl.zeros([tile, 4], dtype=torch.float32)
        for tk in hl.tile(d, block_size=16):
            ones = hl.full([tk, 4], 1.0, dtype=torch.float32)
            acc = hl.dot(torch.exp(x[tile, tk]), ones, acc=acc)
        tail[tile, :] = acc
        square[tile, :, :] = sq[tile, :, :].transpose(-2, -1)
    return both, tail, square


def _padded_exp_rows(x: torch.Tensor) -> torch.Tensor:
    # Explicit padding: the masked load reads 0, so padded lanes are exp(0).
    padded = torch.nn.functional.pad(x, (0, 12))
    return torch.exp(padded).sum(-1)


def test_full_padded_operand_defines_contraction_k() -> None:
    args = (torch.empty(6, 20), torch.empty(6, 4, 4))
    source = _contraction_source(k_promoted_contractions, args, [2])
    assert "ptp_thread" in source
    terms = _contraction_terms(source)
    assert len(terms) == 2  # aten.mm and hl.dot
    assert all(" if " not in term for term in terms), terms


def test_partial_contraction_k_keeps_bounds() -> None:
    args = (torch.empty(6, 20), torch.empty(20, 4), torch.empty(6, 4, 4))
    source = _contraction_source(k_partial_contractions, args, [2])
    assert "ptp_thread" in source
    both, tail = _contraction_terms(source)
    assert re.search(r"if ptp_k_\d+ < 20 else", both)
    assert re.search(r"ptp_loop_\d+ \+ ptp_k_\d+ < ptp_end_\d+", tail)


@requires_cuda
def test_contraction_k_padding_matches_reference() -> None:
    x = torch.randn(6, 20, device=DEVICE) / 4
    w = torch.randn(20, 4, device=DEVICE) / 4
    sq = torch.randn(6, 4, 4, device=DEVICE)
    code, (rows, mm, dot, square) = code_and_output(
        k_promoted_contractions, (x, sq), block_sizes=[2]
    )
    assert "ptp_thread" in code
    padded = _padded_exp_rows(x)
    torch.testing.assert_close(rows, padded, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mm, padded[:, None].expand(6, 4), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dot, padded[:, None].expand(6, 4), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(square, sq.transpose(-2, -1))
    _, (both, tail, _) = code_and_output(
        k_partial_contractions, (x, w, sq), block_sizes=[2]
    )
    torch.testing.assert_close(both, torch.exp(x) @ torch.exp(w), atol=1e-4, rtol=1e-4)
    logical = torch.exp(x).sum(-1)[:, None].expand(6, 4)
    torch.testing.assert_close(tail, logical, atol=1e-4, rtol=1e-4)
