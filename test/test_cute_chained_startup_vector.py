from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_scan_export import _config
from test.test_cute_chained_scan_export import _cpu_codegen
from test.test_cute_chained_scan_input_export import _export
from test.test_cute_chained_startup import config as startup_config
from test.test_cute_chained_vector_export import _args
from test.test_cute_chained_vector_export import _vectors

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import _with_chained_startup_seed
from helion._compiler.cute import chained_tcgen05
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_startup_transfer"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("last_read", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_startup_vector_exports_and_last_read(dtype, last_read, cache):
    with _cpu_codegen():
        bound = _vectors._bind_isolated((*_args(dtype), "permuted"))
        config = _config()
        config.config.update(
            {
                KEY: "tma",
                "cute_chained_auxiliary_cache": cache,
                "cute_chained_tmem_early_release": True,
            }
        )
        if last_read:
            config.config["cute_chained_tmem_free"] = "last_read"
        source = bound.to_code(config)
        assert source == bound.to_code(bound._normalized_config_copy(config))
    assert source.index("tma_bar_ptr=chain_start_bar") < source.index(
        "chain_scan_0_pointer"
    )
    assert source.index("mbarrier_wait(chain_start_bar, 0)") < source.index(
        "cute.gemm("
    )
    assert source.count("chain_allocator.free(chain_tptr)") == 1
    assert source.count("chain_allocator.relinquish_alloc_permit()") == 1
    assert "chain_export_1_index" in source and "chain_export_2_index" in source
    if last_read:
        assert source.index("chain_allocator.free(chain_tptr)") < source.index(
            "for chain_export_1_step"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_startup_fp32_scan_input_expression(dtype):
    with _cpu_codegen():
        args = (
            torch.zeros(2, 256, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, 128, dtype=torch.bfloat16),
            torch.zeros(2, 128, dtype=dtype),
            "chain",
        )
        config = _config()
        config.config[KEY] = "tma"
        source = _export._bind_isolated(args).to_code(config)
    assert "chain_export_0_index" in source
    assert "tma_bar_ptr=chain_start_bar" in source


def test_startup_m64_domain():
    with _cpu_codegen():
        bound = _vectors._bind_isolated((*_args(), "normal"))
        config = _config()
        config.config[KEY] = "tma"
        config.config["block_sizes"] = [64, 64]
        source = bound.to_code(config)
        assert "tcgen05.Ld16x256bOp" in source
        assert "tma_bar_ptr=chain_start_bar" in source


@pytest.mark.parametrize("mode", ["full", "serial64", "overlap64"])
def test_startup_rejects_initialized_late_k64(mode):
    from test.test_cute_chained_k_schedule import source

    with pytest.raises(exc.BackendUnsupported, match="uninitialized M128"):
        source(mode, **{KEY: "tma"})


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("value", [True, 1, None, "unknown"])
def test_startup_invalid_never_repairs(repair, value):
    with _cpu_codegen():
        bound = _vectors._bind_isolated((*_args(), "normal"))
        raw = _config().config | {KEY: value}
        with pytest.raises(exc.InvalidConfig, match="legacy or tma"):
            bound.config_spec.normalize(raw, _fix_invalid=repair)


def test_startup_resource_charge_and_padding_are_additive():
    from test.test_cute_chained_startup import _inputs
    from test.test_cute_chained_startup import _tcgen_chain

    with _cpu_codegen():
        bound = _tcgen_chain._bind_isolated((*_inputs("cpu"), "decay"))
        counts = []
        original = chained_tcgen05.supported_plan

        def record(plan, *, startup=False):
            plain = chained_tcgen05._shared_memory_bytes(plan)
            charged = chained_tcgen05._shared_memory_bytes(plan, startup=True)
            assert charged == plain + 128
            with patch.object(
                chained_tcgen05.CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=charged - 1,
            ):
                assert not original(plan, startup=True)
            with patch.object(
                chained_tcgen05.CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=charged,
            ):
                assert original(plan, startup=True)
            counts.append(startup)
            return original(plan, startup=startup)

        with patch.object(chained_tcgen05, "supported_plan", side_effect=record):
            bound.to_code(startup_config())
    assert True in counts and False in counts


def test_startup_charge_reduces_early_cache_budget_once():
    budgets = []
    original = chained_tcgen05.make_early_auxiliary_cache

    def record(cg, plan, scans, budget):
        budgets.append(budget)
        return original(cg, plan, scans, budget)

    with (
        _cpu_codegen(),
        patch.object(chained_tcgen05, "make_early_auxiliary_cache", side_effect=record),
    ):
        bound = _vectors._bind_isolated((*_args(), "normal"))
        config = _config()
        config.config["cute_chained_auxiliary_cache"] = True
        bound.to_code(config)
        config.config[KEY] = "tma"
        bound.to_code(config)
    assert len(budgets) == 2 and budgets[1] == budgets[0] - 128


def test_startup_single_sibling_after_complete_legacy_pool():
    from helion._compiler.cute import chained_startup

    with _cpu_codegen():
        with patch.object(chained_startup, "has_startup_leaf", return_value=False):
            old = _vectors._bind_isolated((*_args(), "normal"))
            old_pool = [dict(c.config) for c in old.config_spec.compiler_seed_configs]
            old_default = dict(old.config_spec.default_config().config)
        new = _vectors._bind_isolated((*_args(), "normal"))
        new_pool = list(new.config_spec.compiler_seed_configs)
        assert [dict(c.config) for c in new_pool if KEY not in c.config] == old_pool
        assert sum(c.config.get(KEY) == "tma" for c in new_pool) == 1
        assert dict(new.config_spec.default_config().config) == old_default
        assert new.host_function is not None
        with new.env:
            parents = CuteChainedMatmulHeuristic._tcgen05_seed_configs_for_rows(
                new.env, new.host_function.device_ir, 128
            )
        assert parents


def test_startup_partial_and_unsupported_parents_remain_unchanged():
    parent = startup_config(None)
    partial = helion.Config()
    initialized = helion.Config.from_dict(
        parent.config | {"cute_chained_initialized_accumulator": True}
    )
    old = [partial, initialized, partial]
    assert _with_chained_startup_seed(old, [parent]) is old
    assert _with_chained_startup_seed([parent], []) == [parent]
