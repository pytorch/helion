from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import os
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tcgen05 import _compile
from .test_cute_chained_tcgen05 import _inputs as _bridge_inputs
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05_config import _without_early_release_seed
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_c_smem_padding"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_boundary(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), device=a.device, dtype=a.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        first = hl.dot(a[bi, row, kk], b[bi, kk, col])
        second = hl.dot(c[bi, row, kk], d[bi, kk, col])
        out[bi, row, col] = (first + second).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cached_resident_boundary(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), device=a.device, dtype=a.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        left = (a[bi, row, kk].float() * scale[kk][None, :]).to(a.dtype)
        first = hl.dot(left, b[bi, kk, col])
        second = hl.dot(c[bi, row, kk], d[bi, kk, col])
        out[bi, row, col] = (first + second).to(a.dtype)
    return out


def _args(n: int = 64, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(976)
    # Wide N needs a shallow K to remain inside the unchanged resident-SMEM
    # envelope; unlike final-B borrowing, output storage still needs M*N.
    reduction = 32 if n == 256 else 128
    return tuple(
        torch.randn(shape, dtype=dtype, generator=generator) * 0.05
        for shape in (
            (2, 128, reduction),
            (2, reduction, n),
            (2, 128, reduction),
            (2, reduction, n),
        )
    )


def _config(padding: int | None = 4, n: int = 64) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [128, n],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if padding is not None:
        values[KEY] = padding
    return helion.Config.from_dict(values)


@contextmanager
def _cpu_codegen() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=152),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


def _code(padding: int | None, n: int = 64, dtype: torch.dtype = torch.bfloat16) -> str:
    with _cpu_codegen():
        return _resident_boundary._bind_isolated(_args(n, dtype)).to_code(
            _config(padding, n)
        )


@pytest.mark.parametrize("value", [True, False, -4, 1, 8, 4.0, "4", None])
def test_padding_schema_rejects_non_integer_domain(value: object) -> None:
    with _cpu_codegen():
        bound = _resident_boundary._bind_isolated(_args())
        with bound.env, pytest.raises(InvalidConfig, match="must be 0 or 4"):
            bound.config_spec.normalize(dict(_config()) | {KEY: value})


@pytest.mark.parametrize("padding", [0, 4])
def test_padding_schema_roundtrip_and_search(padding: int) -> None:
    with _cpu_codegen():
        bound = _resident_boundary._bind_isolated(_args())
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            _, restored = generation.canonicalize_flat(
                generation.flatten(_config(padding))
            )
            assert restored.config[KEY] == padding
            fragment = bound.config_spec._flat_fields()[KEY]
            assert isinstance(fragment, EnumFragment)
            assert fragment.choices == (0, 4)
            assert bound.config_spec.default_config().config[KEY] == 0


def test_padding_schema_inactive_schedule_and_unsupported_dag() -> None:
    with _cpu_codegen():
        bound = _resident_boundary._bind_isolated(_args())
        with bound.env:
            values = dict(_config()) | {"cute_chained_mma_schedule": "coalesced"}
            bound.config_spec.normalize(values)
            assert values[KEY] == 0
            with patch.object(
                bound.config_spec, "cute_chained_tcgen05_search_enabled", False
            ):
                values.pop("cute_chained_pointwise_vectorize", None)
                values.pop("cute_chained_auxiliary_cache", None)
                with pytest.raises(InvalidConfig, match="padding requires"):
                    bound.config_spec.normalize(dict(values))
                bound.config_spec.normalize(values, _fix_invalid=True)
                assert KEY not in values


def test_padding_seeds_keep_same_pool_and_add_only_tcgen_siblings() -> None:
    with _cpu_codegen():
        bound = _resident_boundary._bind_isolated(_args())
        with bound.env:
            assert bound.host_function is not None
            device_ir = bound.host_function.device_ir
            seeds = CuteChainedMatmulHeuristic.get_seed_configs(bound.env, device_ir)
            assert seeds is not None
            seeds = _without_early_release_seed(seeds)
            old = [seed for seed in seeds if KEY not in seed.config]
            padded = [seed for seed in seeds if KEY in seed.config]
            original_tcgen = [
                seed
                for seed in old
                if seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
            ]
            assert padded and seeds[0] is old[0]
            assert len(seeds) == len(old) + len(padded)
            assert [seed.config for seed in padded] == [
                seed.config | {KEY: 4} for seed in original_tcgen
            ]
            assert (
                CuteChainedMatmulHeuristic.get_seed_config(bound.env, device_ir)
                == old[0]
            )
            generation = ConfigGeneration(bound.config_spec)
            for seed in padded:
                _, restored = generation.canonicalize_flat(generation.flatten(seed))
                assert restored.config[KEY] == 4


@pytest.mark.parametrize("n", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_padding_changes_only_intermediate_allocation_and_pitch(
    n: int, dtype: torch.dtype
) -> None:
    omitted = _code(None, n, dtype)
    zero = _code(0, n, dtype)
    padded = _code(4, n, dtype)
    assert omitted == zero
    old = f"chain_0_c = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {128 * n}, alignment=128), cute.make_layout((128, {n}), stride=({n}, 1)))"
    new = f"chain_0_c = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {128 * (n + 4)}, alignment=128), cute.make_layout((128, {n}), stride=({n + 4}, 1)))"
    assert zero.count(old) == padded.count(new) == 1
    assert padded.replace(new, old) == zero


def test_padding_does_not_change_packed_tmem_bridge_or_final_output() -> None:
    with _cpu_codegen():
        args = (*_bridge_inputs("cpu"), "plain")
        sources = [
            _tcgen_chain._bind_isolated(args).to_code(_config(pad)) for pad in (0, 4)
        ]
    assert sources[0] == sources[1]
    assert "chain_1_bridge_layout" in sources[0]
    assert "chain_0_c =" not in sources[0]


def test_padding_capacity_includes_extra_bytes() -> None:
    from helion._compiler.cute.chained_tcgen05 import _shared_memory_bytes

    observed: dict[int, int] = {}

    def capture(plan: Any, c_smem_padding: int = 0, *, startup: bool = False) -> int:
        size = _shared_memory_bytes(plan, c_smem_padding, startup=startup)
        observed[c_smem_padding] = size
        return size

    with patch(
        "helion._compiler.cute.chained_tcgen05._shared_memory_bytes",
        side_effect=capture,
    ):
        _code(0)
        _code(4)
    assert observed[4] - observed[0] == 128 * 4 * 4


def test_padding_combined_cache_capacity_and_source() -> None:
    from helion._compiler.cute.chained_early_aux_cache import make_early_auxiliary_cache
    from helion._compiler.cute.chained_tcgen05 import _shared_memory_bytes

    args = (*_args(), torch.ones(128, dtype=torch.float32))
    budgets: list[int] = []
    footprints: list[int] = []

    def observe(cg: Any, plan: Any, existing: Any, budget_bytes: int) -> Any:
        budgets.append(budget_bytes)
        footprints.append(_shared_memory_bytes(plan, 0))
        return make_early_auxiliary_cache(cg, plan, existing, budget_bytes)

    def emit(padding: int, capacity: int = 232448) -> str:
        config = _config(padding)
        config.config["cute_chained_auxiliary_cache"] = True
        config.config["cute_chained_pointwise_vectorize"] = True
        with (
            _cpu_codegen(),
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
            ),
            patch(
                "helion._compiler.cute.chained_tcgen05.make_early_auxiliary_cache",
                side_effect=observe,
            ),
        ):
            return _cached_resident_boundary._bind_isolated(args).to_code(config)

    zero, padded = emit(0), emit(4)
    assert budgets[0] - budgets[1] == 2048
    assert footprints[0] == footprints[1]
    assert "chain_early_aux_0 =" in zero and "chain_early_aux_0 =" in padded
    old = "alloc_smem(cutlass.Float32, 8192, alignment=128), cute.make_layout((128, 64), stride=(64, 1))"
    new = "alloc_smem(cutlass.Float32, 8704, alignment=128), cute.make_layout((128, 64), stride=(68, 1))"
    assert padded.replace(new, old) == zero
    # Capacity admits the padded baseline but not its optional 512-byte cache.
    limit = footprints[0] + 2048
    assert "chain_early_aux_0 =" in emit(0, limit)
    assert "chain_early_aux_0 =" not in emit(4, limit)
    assert budgets[-2:] == [2048, 0]
    # Below the mandatory padded footprint, codegen fails closed instead of
    # emitting a too-large resident allocation; the old geometry still fits.
    emit(0, limit - 128)
    with pytest.raises(BackendUnsupported, match="supported full-tile contraction DAG"):
        emit(4, limit - 128)


@pytest.mark.parametrize("n", range(32, 257, 32))
def test_padding_bank_model(n: int) -> None:
    for padding, expected in ((0, 32), (4, 4), (8, 8)):
        pitch = n + padding
        for slot in range(0, n, 4):
            waves = sum(
                max(
                    Counter(
                        (lane * pitch + slot + word) % 32
                        for lane in range(first, first + 8)
                        for word in range(4)
                    ).values()
                )
                for first in range(0, 32, 8)
            )
            assert waves == expected
        addresses = {row * pitch + col for row in range(128) for col in range(n)}
        assert len(addresses) == 128 * n and max(addresses) < 128 * pitch


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("cache", [False, True])
def test_padding_real_cpu_compile(dtype: str, cache: bool, tmp_path: Path) -> None:
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
        "CUTE_DSL_DUMP_DIR": str(tmp_path),
        "CUTE_DSL_CACHE_DIR": str(tmp_path / "cute-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        "PYTHONPATH": os.pathsep.join((repository, os.environ.get("PYTHONPATH", ""))),
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, dtype, str(int(cache))],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_padding_actual_tmem_coordinate_mapping() -> None:
    import cutlass
    from cutlass._mlir import ir
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    mlir: Any = ir
    with mlir.Context(), mlir.Location.unknown():
        module = mlir.Module.create()
        with mlir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, 64),
                tcgen05.OperandSource.SMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
                ),
                mma.make_fragment_C(mma.partition_shape_C((128, 64))).layout,
            )
            copy = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), cutlass.Float32
                ),
                acc,
            )
            identity = mma.get_slice(0).partition_C(
                cute.make_identity_tensor((128, 64))
            )
            for thread in range(128):
                coords = copy.get_slice(thread).partition_D(identity)
                assert cute.size(coords) == 64
                for slot in range(64):
                    assert coords[slot] == (thread, slot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_padding_runtime_exact_identity_dots_and_graphs(dtype: torch.dtype) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    identity = torch.eye(128, device=DEVICE, dtype=dtype).repeat(2, 8, 1)
    fn = None
    for seed in range(5):
        generator = torch.Generator(device=DEVICE).manual_seed(980 + seed)
        b = (
            torch.randn((2, 128, 1024), device=DEVICE, dtype=dtype, generator=generator)
            * 0.125
        )
        d = (
            torch.randn((2, 128, 1024), device=DEVICE, dtype=dtype, generator=generator)
            * 0.125
        )
        args = (identity, b, identity, d)
        before = tuple(value.clone() for value in args)
        if fn is None:
            fn = _resident_boundary._bind_isolated(args).compile_config(_config())
        expected = (b.float() + d.float()).to(dtype).repeat(1, 8, 1)
        output = fn(*args)
        saved = output.clone()
        repeated = fn(*args)
        assert output.data_ptr() != repeated.data_ptr()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(repeated, saved, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn(*args)
        assert captured.data_ptr() not in (output.data_ptr(), repeated.data_ptr())
        graph.replay()
        torch.testing.assert_close(captured, saved, atol=0, rtol=0)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(captured, saved, atol=0, rtol=0)
        torch.testing.assert_close(output, saved, atol=0, rtol=0)
        torch.testing.assert_close(args, before, atol=0, rtol=0)


if __name__ == "__main__":
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_initialized()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
    cache = bool(int(sys.argv[2]))
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        if cache:
            args = (*_args(dtype=dtype), torch.ones(128, dtype=torch.float32))
            config = _config(4)
            config.config["cute_chained_auxiliary_cache"] = True
            config.config["cute_chained_pointwise_vectorize"] = True
            with _cpu_codegen():
                code = _cached_resident_boundary._bind_isolated(args).to_code(config)
            ptx = _compile(code, args, None, entry="_cached_resident_boundary")
        else:
            ptx = _compile(
                _code(4, dtype=dtype),
                _args(dtype=dtype),
                None,
                entry="_resident_boundary",
            )
    assert "tcgen05.mma" in ptx
    assert "st.shared.v4.b32" in ptx and "ld.shared.v4.b32" in ptx
