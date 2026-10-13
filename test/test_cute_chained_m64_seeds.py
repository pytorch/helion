from __future__ import annotations

from collections import Counter
import dataclasses
import importlib
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_warp_workspace import _args
from .test_cute_chained_warp_workspace import _code
from .test_cute_chained_warp_workspace import _long_chain
from .test_cute_chained_workspace import _assert_safe
import helion
from helion import exc
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_tcgen05 import _shared_memory_bytes
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _seeded(a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    batch, m, _k = a.shape
    n = b.size(2)
    out = torch.empty((batch, m, n), dtype=torch.float32, device=a.device)
    for tile in hl.tile(batch, block_size=1):
        row = tile.id
        out[row, :, :] = hl.dot(a[row, :, :], b[row, :, :], acc=seed[row, :, :].T)
    return out


def _config() -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")


@pytest.mark.parametrize("n", [32, 64, 96, 128, 256])
def test_sparse_seed_store_covers_exact_result_coordinates(n: int) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    tcgen = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
    helpers = importlib.import_module("cutlass.utils.blackwell_helpers")
    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = helpers.make_trivial_tiled_mma(
                cutlass.Float16,
                cutlass.Float16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                cutlass.Float32,
                tcgen.CtaGroup.ONE,
                (64, n),
                tcgen.OperandSource.SMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                mma.make_fragment_C(mma.partition_shape_C((64, n))).layout,
            )
            repetition = min(8, (n & -n) // 8)
            store = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen.St16x256bOp(tcgen.Repetition(repetition)), cutlass.Float32
                ),
                acc,
            )
            load = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen.Ld16x256bOp(tcgen.Repetition(repetition)), cutlass.Float32
                ),
                acc,
            )
            identity = mma.get_slice(0).partition_C(cute.make_identity_tensor((64, n)))
            owners: list[tuple[int, int]] = []
            for thread in range(128):
                written = store.get_slice(thread).partition_S(identity)
                read = load.get_slice(thread).partition_D(identity)
                write_coords = [tuple(written[i]) for i in range(cute.size(written))]
                assert write_coords == [tuple(read[i]) for i in range(cute.size(read))]
                owners.extend(write_coords)
            assert Counter(owners) == Counter(
                (i, j) for i in range(64) for j in range(n)
            )


@pytest.mark.parametrize("n", [32, 64, 96, 128, 256])
def test_sparse_seed_codegen_and_search(n: int) -> None:
    with _cpu_codegen():
        bound = _seeded._bind_isolated(
            (
                torch.empty((2, 64, 128), dtype=torch.bfloat16),
                torch.empty((2, 128, n), dtype=torch.bfloat16),
                torch.empty((2, n, 64)),
            )
        )
        assert bound.config_spec.cute_chained_tcgen05_search_enabled
        source = bound.to_code(_config())
    # Full-slice lowering pads a non-power-of-two logical dimension; the masks
    # stay in the generated seed expression and final store.
    assert (
        load_operation((64, 1 << (n - 1).bit_length())).replace(".Ld", ".St") in source
    )
    assert "cute.arch.fence_view_async_tmem_store()" in source


@pytest.mark.parametrize("size,stages,fanout", [(64, 14, True), (128, 4, False)])
def test_root_workspace_rescues_long_seeded_chain(
    size: int, stages: int, fanout: bool
) -> None:
    footprints = []
    original = chained_tcgen05.supported_plan

    def record(plan, *, startup=False):
        footprints.append(
            (
                _shared_memory_bytes(plan),
                _shared_memory_bytes(dataclasses.replace(plan, root_workspace=None)),
            )
        )
        return original(plan, startup=startup)

    with patch.object(chained_tcgen05, "supported_plan", record):
        source, plan = _code(
            size=size, stages=stages, mixed=True, fanout=fanout, schedule="tcgen05_tmem"
        )
    assert plan.root_workspace is not None
    _assert_safe(plan.root_workspace)
    assert len(plan.root_workspace.regions) == stages - 1
    assert footprints[-1][0] <= 232448 < footprints[-1][1]
    assert "chain_c_workspace" in source
    assert "bridge" not in source
    with pytest.raises(exc.BackendUnsupported):
        _code(
            size=size,
            stages=stages,
            mixed=True,
            fanout=fanout,
            schedule="tcgen05_tmem",
            capacity=4096,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n", [32, 64, 96, 128, 256])
def test_sparse_explicit_seed_gpu(dtype: torch.dtype, n: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(503)
    a = torch.randn((2, 64, 128), generator=gen, device="cuda").mul_(0.1).to(dtype)
    b = torch.randn((2, 128, n), generator=gen, device="cuda").mul_(0.1).to(dtype)
    seed = torch.arange(2 * n * 64, device="cuda", dtype=torch.float32).view(2, n, 64)
    seed = seed.remainder(127).mul_(0.03125)
    bound = _seeded._bind_isolated((a, b, seed))
    bound.set_config(_config())
    actual = bound(a, b, seed)
    expected = (a.double() @ b.double()) + seed.transpose(-1, -2).double()
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("size,stages,fanout", [(64, 14, True), (128, 4, False)])
def test_root_workspace_seed_transpose_and_fanout_gpu(
    size: int, stages: int, fanout: bool
) -> None:
    args = _args(size=size, stages=stages, mixed=True, fanout=fanout, device="cuda")
    left, right, initial = args[:3]
    assert isinstance(left, torch.Tensor)
    assert isinstance(right, torch.Tensor)
    assert isinstance(initial, torch.Tensor)
    gen = torch.Generator(device="cuda").manual_seed(508)
    for value in (left, right, initial):
        value.normal_(generator=gen).mul_(0.05)
    state = (left.double() @ right.double() + initial.double()).float()
    first = state
    for _step in range(stages - 1):
        state = (
            state.half().double() @ right.half().double()
            + (state.transpose(-1, -2) * 0.125).double()
        ).float()
    expected = state + first * 0.0625 if fanout else state
    bound = _long_chain._bind_isolated(args)
    bound.set_config(_config())
    actual = bound(*args)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
