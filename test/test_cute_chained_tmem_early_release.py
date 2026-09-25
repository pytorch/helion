from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_initialized_accumulator import _cpu
from test.test_cute_chained_late_rhs import _args as late_args
from test.test_cute_chained_late_rhs import _config as late_config
from test.test_cute_chained_late_rhs import _pair as late_kernel
from test.test_cute_chained_tcgen05 import _inputs as bridge_args
from test.test_cute_chained_tcgen05 import _tcgen_chain

import helion
from helion import exc
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import CUTE_CHAINED_TMEM_EARLY_RELEASE_KEY
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan

pytestmark = skipUnlessBackends(["cute"])
KEY = CUTE_CHAINED_TMEM_EARLY_RELEASE_KEY
RELEASE = "    chain_allocator.relinquish_alloc_permit()\n"
RETRIEVE = "    chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)\n"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([m, n], block_size=[None, n]):
        kk = hl.arange(k)
        out[row, col] = hl.dot(a[row, kk], b[kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for row in hl.tile(a.numel()):
        out[row] = a[row] + 1
    return out


def _values(dtype: torch.dtype, n: int, m: int = 128) -> tuple[torch.Tensor, ...]:
    return torch.empty((m, 128), dtype=dtype), torch.empty((128, n), dtype=dtype)


def _config(value: object = "missing", **extra: object) -> helion.Config:
    config: dict[str, object] = {
        "block_sizes": [128],
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
    }
    if value != "missing":
        config[KEY] = value
    return helion.Config.from_dict(config | extra)


def _assert_only_release(old: str, new: str) -> None:
    assert old.count(RELEASE) == new.count(RELEASE) == 1
    assert old.count(RETRIEVE) == new.count(RETRIEVE) == 1
    assert RETRIEVE + RELEASE in old
    assert RETRIEVE + RELEASE not in new
    assert new.replace(RELEASE, "", 1).replace(RETRIEVE, RETRIEVE + RELEASE, 1) == old
    for source, early in ((old, False), (new, True)):
        module = ast.parse(source)
        body = next(
            node.body
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(item, ast.Expr)
                and ast.unparse(item).startswith("chain_allocator.allocate(")
                for item in node.body
            )
        )
        statements = [ast.unparse(node) for node in body]
        allocate = next(
            i
            for i, text in enumerate(statements)
            if text.startswith("chain_allocator.allocate(")
        )
        release = statements.index(RELEASE.strip())
        wait = statements.index("chain_allocator.wait_for_alloc()")
        retrieve = statements.index(RETRIEVE.strip())
        free = next(
            i
            for i, text in enumerate(statements)
            if text.startswith("chain_allocator.free(")
        )
        assert allocate < wait < retrieve < free
        assert (release == allocate + 1) is early
        assert retrieve == wait + 1
        assert statements[free - 1] == "cute.arch.sync_threads()"
        for method in (
            "allocate",
            "relinquish_alloc_permit",
            "wait_for_alloc",
            "retrieve_ptr",
            "free",
        ):
            calls = [
                node
                for node in ast.walk(module)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) == f"chain_allocator.{method}"
            ]
            assert len(calls) == 1


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 96, 128, 256))
def test_single_allocation_domain_and_default_identity(
    dtype: torch.dtype, n: int
) -> None:
    before = torch.cuda.is_initialized()
    with _cpu():
        bound = _single._bind_isolated(_values(dtype, n))
        old = bound.to_code(_config())
        assert old == bound.to_code(_config(False))
        new = bound.to_code(_config(True))
    _assert_only_release(old, new)
    columns = max(32, 1 << (n - 1).bit_length())
    assert f"chain_allocator.allocate({columns})" in new
    assert new.count("cute.gemm(") == 1
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("n", (32, 64, 128, 256))
def test_packed_bridge_keeps_doubled_columns(dtype: torch.dtype, n: int) -> None:
    args = (*bridge_args("cpu", dtype, n=n), "plain")
    config = helion.Config(
        block_sizes=[128, n], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )
    with _cpu():
        bound = _tcgen_chain._bind_isolated(args)
        old = bound.to_code(config)
        new = bound.to_code(helion.Config.from_dict(config.config | {KEY: True}))
    _assert_only_release(old, new)
    assert "OperandSource.TMEM" in new
    assert f"chain_allocator.allocate({max(128, n) * 2})" in new


@pytest.mark.parametrize("initialized", (False, True))
@pytest.mark.parametrize(
    "raw,cache,aux", tuple(itertools.product((False, True), repeat=3))
)
def test_two_dot_interactions_exact_inverse(
    initialized: bool, raw: bool, cache: bool, aux: bool
) -> None:
    config = late_config(
        False,
        cute_chained_initialized_accumulator=initialized,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_inplace_async=raw,
        cute_chained_pointwise_read_cache=cache,
        cute_chained_auxiliary_cache=aux,
        cute_chained_pointwise_unroll=8,
    )
    with _cpu():
        bound = late_kernel._bind_isolated(late_args())
        old = bound.to_code(config)
        new = bound.to_code(helion.Config.from_dict(config.config | {KEY: True}))
    _assert_only_release(old, new)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("kind", ("dense", "offset", "stride", "tail"))
def test_late_rhs_guard_and_lifetime_unchanged(dtype: torch.dtype, kind: str) -> None:
    config = late_config(
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_read_cache=True,
        cute_chained_pointwise_inplace_async=True,
    )
    with _cpu():
        bound = late_kernel._bind_isolated(late_args(dtype, kind=kind))
        old = bound.to_code(config)
        new = bound.to_code(helion.Config.from_dict(config.config | {KEY: True}))
    _assert_only_release(old, new)
    assert "chain_output_ptr = chain_a_workspace" in new


@pytest.mark.parametrize("value", (0, 1, None, "true", 2))
@pytest.mark.parametrize("fix", (False, True))
def test_strict_bool_even_fix_invalid(value: object, fix: bool) -> None:
    with _cpu():
        spec = _single._bind_isolated(_values(torch.bfloat16, 64)).config_spec
        with pytest.raises(exc.InvalidConfig, match="must be bool"):
            spec.normalize(_config(value), _fix_invalid=fix)


@pytest.mark.parametrize("fix", (False, True))
@pytest.mark.parametrize("schedule", ("coalesced", "cp_async", "unknown"))
def test_unsupported_true_never_repaired_away(fix: bool, schedule: str) -> None:
    with _cpu():
        spec = _single._bind_isolated(_values(torch.bfloat16, 64)).config_spec
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(
                _config(True, cute_chained_mma_schedule=schedule), _fix_invalid=fix
            )
        spec.cute_chained_tcgen05_search_enabled = False
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(True), _fix_invalid=fix)


def test_ordinary_pointwise_false_identity_true_rejection() -> None:
    with _cpu():
        bound = _pointwise._bind_isolated((torch.empty(128),))
        old = helion.Config(block_sizes=[128])
        assert bound.to_code(old) == bound.to_code(
            helion.Config.from_dict(old.config | {KEY: False})
        )
        assert KEY not in bound.config_spec._flat_fields()
        for fix in (False, True):
            with pytest.raises(exc.InvalidConfig, match="early TMEM"):
                bound.config_spec.normalize(
                    helion.Config.from_dict(old.config | {KEY: True}), _fix_invalid=fix
                )


@pytest.mark.parametrize("m", (64, 129))
def test_no_geometry_admission_expansion(m: int) -> None:
    with _cpu(), pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
        _single._bind_isolated(_values(torch.bfloat16, 64, m)).to_code(_config(True))


def test_unavailable_hardware_rejects() -> None:
    with (
        _cpu(),
        patch_cute_mma_support(default_cute_mma_support(tcgen05_f16bf16=False)),
    ):
        bound = _single._bind_isolated(_values(torch.bfloat16, 64))
        assert KEY not in bound.config_spec._flat_fields()
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_config(True))


def test_real_resource_boundary_and_no_plan_rejection() -> None:
    footprints: list[int] = []
    original = chained_tcgen05.supported_plan

    def capture(plan: ChainedMatmulPlan) -> bool:
        footprints.append(chained_tcgen05._shared_memory_bytes(plan))
        return original(plan)

    with _cpu(), patch.object(chained_tcgen05, "supported_plan", side_effect=capture):
        bound = _single._bind_isolated(_values(torch.bfloat16, 256))
        old = bound.to_code(_config())
        new = bound.to_code(_config(True))
    _assert_only_release(old, new)
    assert len(set(footprints)) == 1
    for enabled in (False, True):
        with (
            _cpu(),
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=footprints[0],
            ),
        ):
            _single._bind_isolated(_values(torch.bfloat16, 256)).to_code(
                _config(enabled)
            )
        with (
            _cpu(),
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=footprints[0] - 1,
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            _single._bind_isolated(_values(torch.bfloat16, 256)).to_code(
                _config(enabled)
            )
    with (
        _cpu(),
        patch(
            "helion._compiler.cute.chained_matmul.plan_chained_matmul",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="early TMEM"),
    ):
        _single._bind_isolated(_values(torch.bfloat16, 64)).to_code(_config(True))


@pytest.mark.parametrize("enabled", (False, True))
def test_real_compile_config_raw_normalized_stops_at_load(enabled: bool) -> None:
    class StopBeforeModuleLoad(BaseException):
        pass

    sources: list[str] = []

    def stop(source: str, *args: object, **kwargs: object) -> None:
        sources.append(source)
        raise StopBeforeModuleLoad

    with _cpu():
        bound = _single._bind_isolated(_values(torch.bfloat16, 128))
        raw = _config(enabled)
        before = dict(raw.config)
        normalized = bound._normalized_config_copy(raw)
        with (
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch("helion.runtime.kernel.PyCodeCache.load", side_effect=stop),
        ):
            for config in (raw, normalized):
                with pytest.raises(StopBeforeModuleLoad):
                    bound.compile_config(config)
        assert raw.config == before
        assert sources[0] == sources[1] == bound.to_code(raw)


def test_one_seed_twin_preserves_objects_and_all_old_priority() -> None:
    seeds = [
        helion.Config(cute_chained_mma_schedule="coalesced"),
        helion.Config(cute_chained_mma_schedule="tcgen05_tmem", block_sizes=[128, 64]),
        helion.Config(cute_chained_mma_schedule="tcgen05_tmem", block_sizes=[128, 32]),
    ]
    seeds.append(seeds[1])
    result = heuristics._with_early_tmem_release_seed(seeds)
    assert len(result) == len(seeds) + 1
    assert result[2].config == seeds[1].config | {KEY: True}
    assert all(
        a is b
        for a, b in zip(
            (x for x in result if not x.config.get(KEY)), seeds, strict=True
        )
    )
    assert heuristics._with_early_tmem_release_seed([]) == []
    assert heuristics._with_early_tmem_release_seed(seeds[:1]) == seeds[:1]
