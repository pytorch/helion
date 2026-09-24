from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_batched_aux_tma import _batched_aux
from .test_cute_batched_aux_tma import _config
from .test_cute_batched_aux_tma import _cpu_codegen
from .test_cute_batched_aux_tma import _inputs
from .test_cute_fragment_where import _args
from .test_cute_fragment_where import _fragment_where
from .test_cute_scheduler_mailbox import _plain_matmul
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05ThreadLocalEpilogueHeuristic,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])


def _plain_config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 64],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 2,
            **overrides,
        }
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("aux", [False, True])
def test_consumer_register_choice_roundtrip_and_source(
    dtype: torch.dtype, aux: bool
) -> None:
    args = (
        _inputs(dtype=dtype)
        if aux
        else (
            torch.empty((256, 128), dtype=dtype),
            torch.empty((128, 512), dtype=dtype),
        )
    )
    kernel = _batched_aux if aux else _plain_matmul
    config = _config() if aux else _plain_config()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        old = bound.to_code(config)
        explicit = bound.to_code(
            helion.Config.from_dict(config.config | {"tcgen05_consumer_regs": 256})
        )
        low = helion.Config.from_dict(config.config | {"tcgen05_consumer_regs": 128})
        code = bound.to_code(low)
        assert old == explicit
        assert (
            code.replace("setmaxregister_increase(128)", "setmaxregister_increase(256)")
            == old
        )
        assert "cute.gemm(" in code
        with bound.env:
            field = bound.config_spec._flat_fields()["tcgen05_consumer_regs"]
            assert isinstance(field, EnumFragment)
            assert field.default() == 256
            search_values = field.search_values()
            assert search_values is not None and 128 in search_values
            generation = ConfigGeneration(bound.config_spec)
            _, restored = generation.canonicalize_flat(generation.flatten(low))
        assert restored.config["tcgen05_consumer_regs"] == 128


@pytest.mark.parametrize("value", TCGEN05_CONSUMER_REGS_CHOICES)
def test_older_register_choices_remain_legal(value: int) -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        code = bound.to_code(_config(tcgen05_consumer_regs=value))
    assert f"setmaxregister_increase({value})" in code
    # Grouped policy's established domain must not widen with the new domain.
    assert TCGEN05_CONSUMER_REGS_CHOICES == (224, 232, 240, 256)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pid_type": "flat"},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_persistence_model": "clc_persistent"},
        {"tcgen05_strategy": "legacy_shared_loop"},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_num_epi_warps": 2},
        {"tcgen05_grouped_mode": "static"},
        {"tcgen05_layout_overrides_epi_tile_m": 64},
        {"block_sizes": [128, 128, 256]},
    ],
)
def test_low_register_guard_excludes_unvalidated_topologies(
    overrides: dict[str, object],
) -> None:
    with _cpu_codegen():
        bound = _plain_matmul._bind_isolated(
            (
                torch.empty((256, 128), dtype=torch.bfloat16),
                torch.empty((128, 512), dtype=torch.bfloat16),
            )
        )
        state = bound.config_spec._cute_tcgen05_config
        assert state._low_consumer_regs_config_supported(_plain_config().config)
        assert not state._low_consumer_regs_config_supported(
            _plain_config(**overrides).config
        )


@pytest.mark.parametrize("field,value", [("m", 127), ("n", 511), ("k", 127)])
def test_low_register_guard_preserves_logical_tails(field: str, value: int) -> None:
    with _cpu_codegen():
        extents = {"m": 128, "n": 512, "k": 128, field: value}
        bound = _batched_aux._bind_isolated(
            _inputs(m=extents["m"], n=extents["n"], k=extents["k"])
        )
        assert not bound.config_spec._cute_tcgen05_config._low_consumer_regs_config_supported(
            _config().config
        )


def test_low_register_explicit_reject_and_search_repair() -> None:
    with _cpu_codegen():
        bound = _plain_matmul._bind_isolated(
            (
                torch.empty((256, 128), dtype=torch.bfloat16),
                torch.empty((128, 512), dtype=torch.bfloat16),
            )
        )
        with pytest.raises(
            exc.InvalidConfig, match="tcgen05_consumer_regs=128 requires"
        ):
            bound.to_code(_plain_config(pid_type="flat", tcgen05_consumer_regs=128))
        state = bound.config_spec._cute_tcgen05_config
        with bound.env:
            values = _plain_config().config
            bound.config_spec.normalize(values)
            values.update(pid_type="flat", tcgen05_consumer_regs=128)
            state.normalize_strategy(values, fix_invalid=True)
        assert values["tcgen05_consumer_regs"] == 256
        with patch.object(state, "matmul_input_dtype", torch.float8_e4m3fn):
            assert not state._low_consumer_regs_search_enabled()
        with patch.object(state, "matmul_has_non_tcgen05_operand", True):
            assert not state._low_consumer_regs_search_enabled()
        with patch.object(
            state, "matmul_compile_time_static_extents", (256, 512, None)
        ):
            assert not state._low_consumer_regs_search_enabled()


def test_low_register_aux_requires_productive_tma() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        state = bound.config_spec._cute_tcgen05_config
        with pytest.raises(
            exc.InvalidConfig, match="tcgen05_consumer_regs=128 requires"
        ):
            bound.to_code(
                _config(tcgen05_consumer_regs=128, tcgen05_aux_load_mode="simt")
            )
        with patch.object(state, "_aux_tma_edge_search_enabled", return_value=True):
            fragment = state.consumer_regs_autotune_fragments()["tcgen05_consumer_regs"]
            assert isinstance(fragment, EnumFragment)
            assert fragment.choices == TCGEN05_CONSUMER_REGS_CHOICES
            assert fragment.default() == 224


def test_structural_seeds_keep_original_order_and_one_sibling() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        state = bound.config_spec._cute_tcgen05_config
        all_seeds = state.autotune_seed_configs()
        seeds = [
            seed
            for seed in all_seeds
            if not seed.config.get("tcgen05_aux_role_local_scheduler")
        ]
        assert all_seeds[: len(seeds)] == seeds
        old = [
            seed
            for seed in seeds
            if seed.config.get("tcgen05_consumer_regs", 256) == 256
        ]
        low = [
            seed for seed in seeds if seed.config.get("tcgen05_consumer_regs") == 128
        ]
        assert len(old) == len(low) == 3
        assert seeds == old + low
        for original, sibling in zip(old, low, strict=True):
            assert sibling.config == original.config | {"tcgen05_consumer_regs": 128}
            with bound.env:
                generation = ConfigGeneration(bound.config_spec)
                _, restored = generation.canonicalize_flat(generation.flatten(sibling))
            assert restored.config["tcgen05_consumer_regs"] == 128
        assert state.consumer_register_seed_configs(low) == low


def test_fragment_seed_zero_and_promotion_remain_unchanged() -> None:
    with _cpu_codegen():
        bound = _fragment_where._bind_isolated((*_args("cpu"), "causal"))
        heuristic = CuteTcgen05ThreadLocalEpilogueHeuristic
        with bound.env:
            assert bound.host_function is not None
            device_ir = bound.host_function.device_ir
            primary = heuristic.get_seed_config(bound.env, device_ir)
            seeds = heuristic.get_seed_configs(bound.env, device_ir)
        assert primary is not None
        assert len(seeds) == 2
        assert seeds[0] == primary
        assert seeds[1].config == primary.config | {"tcgen05_consumer_regs": 128}
        assert heuristic.promote_seed_to_default


def test_low_register_runtime_fixture_has_multiple_emitted_waves() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs(m=1024, n=2048))
        code = bound.to_code(_config(tcgen05_consumer_regs=128))
    tree = ast.parse(code)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    mma_shapes = [
        ast.literal_eval(node.args[-2])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "make_trivial_tiled_mma"
    ]
    # The admitted eight-warp layout normalizes the requested M64 to M128.
    assert mma_shapes == [(128, 256)]
    assert (constants["_BLOCK_SIZE_1"], constants["_BLOCK_SIZE_2"]) == mma_shapes[0]
    assert (
        3 * (1024 // constants["_BLOCK_SIZE_1"]) * (2048 // constants["_BLOCK_SIZE_2"])
        == 192
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("aux", [False, True])
def test_low_consumer_register_runtime(dtype: torch.dtype, aux: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell TCgen05")
    for seed in range(5):
        generator = torch.Generator(device=DEVICE).manual_seed(912320 + seed)

        def random(
            shape: tuple[int, ...], generator: torch.Generator = generator
        ) -> torch.Tensor:
            return (
                torch.randn(shape, device=DEVICE, dtype=dtype, generator=generator)
                * 0.1
            )

        if aux:
            # 3 * (1024 / 128) * (2048 / 256) = 192 work tiles: mailbox reuse.
            lhs, rhs = random((3, 1024, 128)), random((3, 128, 2048))
            bias, scale = random((3, 1024, 2048)), random((3, 1024, 2048))
            out = torch.empty_like(bias)
            args = (lhs, rhs, bias, scale, out)
            reference = (
                torch.bmm(lhs.double(), rhs.double()) * scale.double() + bias.double()
            )
            config = _config(tcgen05_consumer_regs=128)
            kernel = _batched_aux
            inputs = args[:-1]
        else:
            # (2048 / 128) ** 2 = 256 work tiles: more than one resident wave.
            lhs, rhs = random((2048, 128)), random((128, 2048))
            args = (lhs, rhs)
            reference = lhs.double() @ rhs.double()
            config = _plain_config(tcgen05_consumer_regs=128)
            kernel = _plain_matmul
            inputs = args
        before = [tensor.clone() for tensor in inputs]
        bound = kernel._bind_isolated(args).compile_config(config)
        result = bound(*args)
        torch.testing.assert_close(result.double(), reference, atol=0.002, rtol=0.01)
        expected = result.clone()
        assert torch.equal(bound(*args), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_result = bound(*args)
        graph.replay()
        assert torch.equal(graph_result, expected)
        for _repeat in range(3):
            graph_result.fill_(float("nan"))
            graph.replay()
            assert torch.equal(graph_result, expected)
        for tensor, previous in zip(inputs, before, strict=True):
            assert torch.equal(tensor, previous)
