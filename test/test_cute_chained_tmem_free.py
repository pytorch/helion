from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from .test_cute_chained_tcgen05 import _inputs
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_single
import helion
from helion import exc
from helion._compiler.backend import TritonBackend
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_tmem_free"


def _config(mode: str | None = "last_read", n: int = 64) -> helion.Config:
    result = helion.Config(
        block_sizes=[128, n],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
    )
    if mode is not None:
        result.config[KEY] = mode
    return result


def _bound(kind: str = "scan", dtype: torch.dtype = torch.bfloat16, n: int = 64):
    if kind == "single":
        values = (
            torch.empty(2, 128, 128, dtype=dtype),
            torch.empty(2, 128, n, dtype=dtype),
            torch.empty(2, 128, dtype=dtype),
            True,
            True,
        )
        return _tcgen_single._bind_isolated(values)
    return _tcgen_chain._bind_isolated((*_inputs("cpu", dtype, n=n), kind))


def _inverse(source: str) -> str:
    device = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert device.end_lineno is not None
    lines = source.splitlines(keepends=True)
    free = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "chain_allocator.free(chain_tptr)"
    )
    assert lines[free - 2].strip() == "cute.arch.fence_view_async_tmem_load()"
    assert lines[free - 1].strip() == "cute.arch.sync_threads()"
    statement = lines.pop(free)
    # Removing the early statement shifts the final device line by one.
    end = device.end_lineno - 1
    lines[end:end] = ["    cute.arch.sync_threads()\n", statement]
    return "".join(lines)


@pytest.mark.parametrize("kind", ["single", "plain", "scan", "three"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_whole_source_inverse_and_defaults(kind, dtype):
    with _cpu_codegen():
        bound = _bound(kind, dtype)
        legacy = bound.to_code(_config(None))
        assert bound.to_code(_config("legacy")) == legacy
        raw = _config()
        early = bound.to_code(raw)
        assert early == bound.to_code(bound._normalized_config_copy(raw))
        assert _inverse(early) == legacy
        assert early.count("chain_allocator.free(chain_tptr)") == 1
        assert early.count("cute.arch.sync_threads()") + 1 == legacy.count(
            "cute.arch.sync_threads()"
        )


@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_all_ordinary_widths_and_config_roundtrip(n):
    with _cpu_codegen():
        bound = _bound("single", n=n)
        raw = _config(n=n)
        normalized = bound._normalized_config_copy(raw)
        assert normalized.config[KEY] == "last_read"
        assert bound.to_code(normalized) == bound.to_code(raw)
        assert bound.config_spec.flatten_missing_field_default(KEY, {}) == (
            True,
            "legacy",
        )
        fragment = bound.config_spec._flat_fields()[KEY]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices == ("legacy", "last_read")


@pytest.mark.parametrize("value", [None, True, False, 0, 1, "early", "LEGACY", [], {}])
@pytest.mark.parametrize("repair", [False, True])
def test_invalid_values_never_repair(value, repair):
    with _cpu_codegen():
        bound = _bound()
        config = _config().config | {KEY: value}
        with pytest.raises(exc.InvalidConfig, match="must be legacy or last_read"):
            bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize(
    "delta",
    [
        {"cute_chained_mma_schedule": "coalesced"},
        {"num_warps": 8},
        {"pid_type": "persistent_interleaved"},
        {"cute_cluster_n": 2},
    ],
)
@pytest.mark.parametrize("repair", [False, True])
def test_unsupported_route_never_repair(delta, repair):
    with _cpu_codegen():
        bound = _bound()
        with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
            bound.config_spec.normalize(_config().config | delta, _fix_invalid=repair)


def test_non_cute_rejects_enabled_and_omits_legacy():
    spec = ConfigSpec(backend=TritonBackend())
    with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
        spec.normalize({KEY: "last_read"}, _fix_invalid=True)
    default: dict[str, object] = {}
    legacy: dict[str, object] = {KEY: "legacy"}
    spec.normalize(default)
    spec.normalize(legacy)
    assert legacy == default


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] + 1
    return out


def test_unrelated_kernel_rejects_enabled():
    with _cpu_codegen():
        bound = _copy._bind_isolated((torch.empty(128),))
        with pytest.raises(exc.InvalidConfig, match="resident one-CTA"):
            bound.to_code(
                helion.Config.from_dict({"block_sizes": [128], KEY: "last_read"})
            )


@pytest.mark.parametrize("m,n", [(64, 64), (127, 64), (128, 96)])
def test_existing_shape_admission_remains_strict(m, n):
    with _cpu_codegen():
        values = (
            torch.empty(2, 128, m, dtype=torch.bfloat16),
            torch.empty(2, 128, n, dtype=torch.bfloat16),
            torch.empty(2, 128, dtype=torch.bfloat16),
            True,
            True,
        )
        bound = _tcgen_single._bind_isolated(values)
        config = _config(n=n)
        config.config["block_sizes"] = [m, n]
        if m == 64:
            source = bound.to_code(config)
            legacy = helion.Config.from_dict(
                {key: value for key, value in config.config.items() if key != KEY}
            )
            assert _inverse(source) == bound.to_code(legacy)
        else:
            with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
                bound.to_code(config)


def test_physical_capacity_is_not_waived():
    with _cpu_codegen():
        bound = _bound()
        with (
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=1024
            ),
            pytest.raises(exc.BackendUnsupported),
        ):
            bound.to_code(_config())


def _body():
    return [
        "chain_0_acc = cute.make_tensor(chain_tptr, layout)",
        "chain_0_copy = tcgen05.make_tmem_copy(atom, chain_0_acc)",
        "chain_0_thread = chain_0_copy.get_slice(chain_thread)",
        "chain_0_source = chain_0_thread.partition_S(chain_0_acc)",
        "chain_0_values = cute.make_rmem_tensor(shape, cutlass.Float32)",
        "cute.copy(chain_0_copy, chain_0_source, chain_0_values)",
        "cute.arch.fence_view_async_tmem_load()",
        "cute.arch.sync_threads()",
    ]


@pytest.mark.parametrize("index", [-1, -2, -3])
def test_final_snapshot_fence_and_rendezvous_are_required(index):
    body = _body()
    body.pop(index)
    with pytest.raises(chained_matmul._UnsupportedChain, match="final TMEM load"):
        chained_tcgen05._last_read_epilogue(body, ["x = chain_0_values[0]"], 0)


@pytest.mark.parametrize(
    "use",
    [
        "x = chain_tptr",
        "x = chain_0_acc[0]",
        "cute.copy(atom, chain_0_source, values)",
        "tcgen05.commit(bar)",
        "cute.gemm(mma, a, b, c)",
        "cute.arch.fence_view_async_tmem_store()",
        "chain_allocator.free(chain_tptr)",
    ],
)
def test_later_tmem_use_is_rejected(use):
    with pytest.raises(chained_matmul._UnsupportedChain, match="later TMEM use"):
        chained_tcgen05._last_read_epilogue(_body(), [use], 0)


def test_indirect_prior_alias_is_rejected():
    body = _body()
    body[1:1] = [
        "other = cute.make_tensor(chain_0_acc.iterator, layout)",
        "offset = other.iterator + 8",
        "nested = cute.make_tensor(offset, layout)",
    ]
    with pytest.raises(chained_matmul._UnsupportedChain, match="later TMEM use"):
        chained_tcgen05._last_read_epilogue(body, ["x = nested[0]"], 0)


@pytest.mark.parametrize(
    "alias",
    [
        "hidden: object = chain_0_source",
        "hidden, other = (chain_0_source, 0)",
        "hidden += chain_tptr",
        "if (hidden := chain_0_source):\n    pass",
    ],
)
def test_unsupported_tmem_alias_forms_fail_closed(alias):
    body = _body()
    body.insert(-3, alias)
    with pytest.raises(chained_matmul._UnsupportedChain, match="unproven TMEM alias"):
        chained_tcgen05._last_read_epilogue(body, ["x = hidden[0]"], 0)


def test_register_snapshot_and_layout_metadata_remain_usable():
    epi = [
        "coords = chain_0_thread.partition_D(shared)",
        "x = chain_0_values[0]",
        "shared[0] = x",
        "cute.arch.sync_threads()",
        "output[0] = shared[0]",
    ]
    assert chained_tcgen05._last_read_epilogue(_body(), epi, 0) == [
        "chain_allocator.free(chain_tptr)",
        *epi,
    ]


def test_real_codegen_rejects_injected_future_tmem_read():
    original = chained_tcgen05._epilogue

    def changed(*args, **kwargs):
        return [*original(*args, **kwargs), "later = chain_tptr.toint()"]

    with (
        _cpu_codegen(),
        patch.object(chained_tcgen05, "_epilogue", side_effect=changed),
        pytest.raises(exc.BackendUnsupported, match="later TMEM use"),
    ):
        _bound().to_code(_config())


@pytest.mark.parametrize("kind", ["single", "plain", "scan", "three"])
def test_exact_one_typed_sibling_preserves_old_pool_and_first100(kind):
    from test.test_cute_chained_tcgen05_config import _without_startup_seed

    from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic

    with _cpu_codegen():
        bound = _bound(kind)
        spec = bound.config_spec
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            pool = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            assert pool is not None
            pool = _without_startup_seed(pool)
            enabled = [
                (i, seed)
                for i, seed in enumerate(pool)
                if seed.config.get(KEY) == "last_read"
            ]
            assert len(enabled) == 1
            index, sibling = enabled[0]
            assert index > 0
            assert sibling.config == pool[index - 1].config | {KEY: "last_read"}
            with patch(
                "helion._compiler.autotuner_heuristics.cute._with_last_read_seed",
                side_effect=lambda seeds: seeds,
            ):
                legacy_pool = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            assert legacy_pool is not None
            legacy_pool = _without_startup_seed(legacy_pool)
            assert len(pool) == len(legacy_pool) + 1
            assert [dict(seed) for seed in pool if KEY not in seed.config] == [
                dict(seed) for seed in legacy_pool
            ]
            assert dict(spec.default_config()).get(KEY) is None
            generation = ConfigGeneration(spec)
            flat = generation.random_population_flat(100)
            user = _config("legacy")
            priority = generation.random_population_flat(100, user_seed_configs=[user])
            assert priority[0] == generation.default_flat()
            assert generation.unflatten(priority[1]) == bound._normalized_config_copy(
                user
            )
            members = [
                PopulationBasedSearch.make_unbenchmarked(
                    cast(
                        "PopulationBasedSearch", SimpleNamespace(config_gen=generation)
                    ),
                    row,
                )
                for row in flat
            ]
        selected = [
            (i, member)
            for i, member in enumerate(members)
            if member is not None and member.config.config.get(KEY) == "last_read"
        ]
        assert selected and selected[0][0] < 100
        source = bound.to_code(selected[0][1].config)
        assert "chain_allocator.free(chain_tptr)" in source
        assert _inverse(source) == bound.to_code(
            helion.Config.from_dict(
                {
                    key: value
                    for key, value in selected[0][1].config.config.items()
                    if key != KEY
                }
            )
        )
