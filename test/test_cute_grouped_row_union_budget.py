from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_grouped_row_union_cluster4 import selected
from test.test_cute_grouped_row_union_cluster4 import wrapper
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.cute_mma import _tcgen05_candidate_exceeds_smem
from helion._compiler.cute.grouped_row_union import CONFIG_KEY
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def cpu_only() -> Generator[None, None, None]:
    initialized = torch.cuda.is_initialized()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("torch.cuda.current_device", side_effect=AssertionError("CUDA metadata")),
        patch(
            "torch.cuda.get_device_properties",
            side_effect=AssertionError("CUDA metadata"),
        ),
        patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("CUDA metadata"),
        ),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        _target(),
    ):
        yield
    assert torch.cuda.is_initialized() is initialized
    torch.set_num_threads(previous)


@pytest.fixture
def case() -> tuple[BoundKernel, Config]:
    bound, _args = _bound(1)
    return bound, selected(bound)


def test_capacity_is_not_the_ordinary_ab_budget() -> None:
    device = torch.device("cuda", 0)
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(device)
    budget = CuteTcgen05Config.per_cta_smem_budget_bytes(device)
    assert capacity == 232448
    assert budget == 203776
    assert CuteTcgen05Config.per_cta_ab_smem_budget_bytes(device) == 203776
    required = tcgen05_ab_smem_bytes_per_cta(
        bm=TRANSPOSED.mma_m,
        bn=TRANSPOSED.mma_n,
        bk=TRANSPOSED.block_k,
        dtype_bytes=2,
        ab_stages=TRANSPOSED.ab_stages,
        cluster_m=TRANSPOSED.cluster_m,
    )
    assert required == 215040
    assert TRANSPOSED.shared_upper_bound == 225280
    assert budget < required < TRANSPOSED.shared_upper_bound < capacity


@skipUnlessBackends(["cute"])
def test_actual_set_config_keeps_normalization_and_ordinary_budget(
    case: tuple[BoundKernel, Config],
) -> None:
    bound, config = case
    requested = deepcopy(config)
    compile_source = bound.to_triton_code
    emitted = []

    class CodegenComplete(Exception):
        pass

    def stop_after_codegen(
        normalized: Config,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        assert normalized is not config
        assert normalized == bound.config_spec.normalized_config(config)
        emitted.append(
            compile_source(
                normalized,
                emit_repro_caller=emit_repro_caller,
                output_origin_lines=output_origin_lines,
            )
        )
        raise CodegenComplete

    # Run the actual set_config -> compile_config -> to_code path, stopping only
    # after complete source emission and before any host/native compilation.
    with (
        patch.object(bound, "to_triton_code", side_effect=stop_after_codegen),
        pytest.raises(CodegenComplete),
    ):
        bound.set_config(config)
    assert config == requested
    assert len(emitted) == 1
    assert "block=(32, 6, 1)" in emitted[0]
    assert "StaticPersistentTileScheduler.create(" in emitted[0]


@skipUnlessBackends(["cute"])
def test_exact_source_and_wrapper_match_prior_capacity_facade(
    case: tuple[BoundKernel, Config],
) -> None:
    bound, config = case
    ordinary = bound.to_code(config)
    # Retain the former facade only as a comparison witness, never as the
    # admission target. Both complete generated modules must be identical.
    with patch.object(
        CuteTcgen05Config, "per_cta_smem_budget_bytes", return_value=232448
    ):
        former_facade = bound.to_code(config)
    assert ordinary == former_facade
    assert ast.dump(wrapper(ordinary)) == ast.dump(wrapper(former_facade))


@pytest.mark.parametrize("defer", (False, True))
@skipUnlessBackends(["cute"])
def test_only_explicit_defer_uses_the_complete_ledger(
    case: tuple[BoundKernel, Config], defer: bool
) -> None:
    bound, config = case
    with bound.env:
        projected = bound.env.backend.codegen_config(config)
    assert (
        _tcgen05_candidate_exceeds_smem(
            torch.bfloat16,
            input_device=torch.device("cuda", 0),
            bm=256,
            bn=80,
            bk=64,
            config=projected,
            defer_grouped_worklist_smem_check=defer,
        )
        is not defer
    )


@pytest.mark.parametrize("defer", (False, True))
def test_existing_worklist_deferral_is_preserved(defer: bool) -> None:
    config = {
        "block_sizes": [256, 128, 64],
        "tcgen05_cluster_m": 2,
        "tcgen05_ab_stages": 7,
        "tcgen05_grouped_mode": "worklist_nm",
        "tcgen05_grouped_worklist_source_m_tile": 224,
    }
    assert (
        _tcgen05_candidate_exceeds_smem(
            torch.bfloat16,
            input_device=torch.device("cuda", 0),
            bm=256,
            bn=224,
            bk=64,
            config=config,
            defer_grouped_worklist_smem_check=defer,
        )
        is not defer
    )


@pytest.mark.parametrize(
    "changes,dtype,bk",
    [
        ({CONFIG_KEY: False}, torch.bfloat16, 64),
        ({CONFIG_KEY: 1}, torch.bfloat16, 64),
        ({SCHEDULE_KEY: "legacy"}, torch.bfloat16, 64),
        ({SCHEDULE_KEY: "unknown"}, torch.bfloat16, 64),
        ({"tcgen05_acc_stages": 3}, torch.bfloat16, 64),
        ({"tcgen05_ab_stages": 11}, torch.bfloat16, 64),
        ({"tcgen05_cluster_n": 1}, torch.bfloat16, 64),
        ({"tcgen05_layout_overrides_d_store_box_n": 32}, torch.bfloat16, 64),
        ({}, torch.float16, 64),
        ({}, torch.bfloat16, 128),
    ],
)
@skipUnlessBackends(["cute"])
def test_unproved_protocols_retain_the_ordinary_budget(
    case: tuple[BoundKernel, Config],
    changes: dict[str, object],
    dtype: torch.dtype,
    bk: int,
) -> None:
    bound, config = case
    with bound.env:
        projected = bound.env.backend.codegen_config(config)
    projected.config.update(changes)
    assert _tcgen05_candidate_exceeds_smem(
        dtype,
        input_device=torch.device("cuda", 0),
        bm=256,
        bn=80,
        bk=bk,
        config=projected,
        defer_grouped_worklist_smem_check=True,
    )


@pytest.mark.parametrize("capacity", (0, 203776, TRANSPOSED.shared_upper_bound - 1))
@skipUnlessBackends(["cute"])
def test_complete_capacity_failure_still_rejects(
    case: tuple[BoundKernel, Config], capacity: int
) -> None:
    bound, config = case
    with (
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
        pytest.raises(BackendUnsupported, match="row-union"),
    ):
        bound.to_code(config)
