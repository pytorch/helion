from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_warp_workspace import _args
from .test_cute_chained_warp_workspace import _long_chain
from .test_cute_chained_workspace import _assert_safe
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_warp_result_transport as transport
from helion._compiler.cute.chained_workspace import plan_contraction_workspace
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo

KEY = transport.KEY


def _config(mode: object = "legacy", **extra: object) -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, "cute_chained_mma_schedule": "coalesced", KEY: mode, **extra}
    )


def _code(mode: str, **extra: object) -> tuple[str, ChainedMatmulPlan]:
    plans = []
    original = chained_matmul.plan_chained_matmul

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
        return plan

    with _cpu_codegen(), patch.object(chained_matmul, "plan_chained_matmul", observe):
        bound = _long_chain._bind_isolated(_args(stages=3, size=64, mixed=True))
        source = bound.to_code(_config(mode, **extra))
    assert len(plans) == 1
    return source, plans[0]


def _gemms(source: str) -> list[str]:
    return [
        ast.dump(node, include_attributes=False)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.gemm"
    ]


@pytest.mark.parametrize("schedule", transport.SCHEDULES)
def test_original_mma_seed_and_live_results_survive_transport(schedule: str) -> None:
    sources, plans = {}, {}
    for mode in (mode for mode in transport.MODES if mode != "packed_register"):
        sources[mode], plans[mode] = _code(mode, cute_chained_mma_schedule=schedule)
    assert (
        _gemms(sources["legacy"])
        == _gemms(sources["packed"])
        == _gemms(sources["packed_direct"])
    )
    assert len(_gemms(sources["legacy"])) == 3
    assert plans["legacy"].warp_workspace is None
    for mode in ("packed", "packed_direct"):
        plan = plans[mode]
        assert plan.warp_workspace is not None
        _assert_safe(plan.warp_workspace)
        assert plan.warp_workspace == plan_contraction_workspace(
            plan,
            row_padding=0 if schedule == "k_major" else 4,
            include_final=mode == "packed",
        )
        # The first result remains live through terminal fanout; transposed
        # explicit seeds read all previous C lanes before the added CTA join.
        assert plan.warp_workspace.region("chain_0_c").live_until == 7
        assert "chain_1_seed_coords" in sources[mode]
        assert (
            sources[mode].count("cute.arch.sync_threads()")
            == sources["legacy"].count("cute.arch.sync_threads()") + 3
        )
    assert "chain_2_c_ptr" in sources["packed"]
    assert not any(
        isinstance(node, ast.Name) and node.id == "chain_2_c"
        for node in ast.walk(ast.parse(sources["packed_direct"]))
    )
    assert "chain_2_terminal_coords" in sources["packed_direct"]
    assert "chain_store_step" not in sources["packed_direct"]


def test_default_source_and_original_seed_prefix_are_unchanged() -> None:
    with _cpu_codegen():
        bound = _long_chain._bind_isolated(_args(stages=3, size=64))
        spec = bound.config_spec
        assert KEY in spec._flat_fields()
        without = helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        assert bound.to_code(without) == bound.to_code(_config())
        seeds = spec.compiler_seed_configs
        added = [seed for seed in seeds if KEY in seed]
        assert len(added) == 1 and added[0][KEY] == "packed_direct"
        base = [seed for seed in seeds if KEY not in seed]
        assert seeds == [*base, *added]
        assert (
            CuteChainedMatmulHeuristic._with_warp_result_transport_seed(bound.env, base)
            == seeds
        )
        for mode in (mode for mode in transport.MODES if mode != "packed_register"):
            config = spec.normalized_config(_config(mode))
            generation = spec.create_config_generation()
            assert generation.unflatten(generation.flatten(config)) == config
        with patch.dict(bound.env.settings.autotune_config_overrides, {KEY: "legacy"}):
            assert (
                CuteChainedMatmulHeuristic._with_warp_result_transport_seed(
                    bound.env, base
                )
                == base
            )


@pytest.mark.parametrize("mode", [True, 1, None, "direct_invalid", "PACKED"])
def test_invalid_enum_does_not_disappear(mode: object) -> None:
    with _cpu_codegen():
        bound = _long_chain._bind_isolated(_args(stages=3, size=64))
        with pytest.raises(
            exc.InvalidConfig, match="must be legacy, packed, packed_direct or direct"
        ):
            bound.to_code(_config(mode))


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_chained_mma_schedule": "tcgen05_tmem"},
        {"num_warps": 16},
        {"cute_chained_fragment_epilogues": True},
    ],
)
def test_other_owners_are_not_silently_erased(extra: dict[str, object]) -> None:
    with _cpu_codegen():
        bound = _long_chain._bind_isolated(_args(stages=3, size=64))
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_config("packed_direct", **extra))


def test_missing_plan_and_capacity_cannot_fall_back() -> None:
    with _cpu_codegen():
        bound = _long_chain._bind_isolated(_args(stages=3, size=64))
        for patcher in (
            patch.object(chained_matmul, "plan_chained_matmul", return_value=None),
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=128
            ),
        ):
            with (
                patcher,
                pytest.raises(exc.BackendUnsupported, match="admitted complete root"),
            ):
                bound.to_code(_config("packed_direct"))


@pytest.mark.parametrize(
    "shape,threads,accepted",
    [
        ((16, 8, 16), 32, True),
        ((80, 96, 48), 128, True),
        ((64, 64, 16), 256, True),
        ((64, 48, 16), 128, False),
        ((65, 64, 16), 128, False),
        ((64, 64, 17), 128, False),
        ((64, 16, 16), 128, False),
        ((64, 64, 16), 512, False),
    ],
)
def test_physical_complete_tiles_reject_tails_and_narrow_warps(
    shape: tuple[int, int, int],
    threads: int,
    accepted: bool,
) -> None:
    from types import SimpleNamespace

    assert (
        transport.complete_warp_tiles(
            cast("ChainedMatmulPlan", SimpleNamespace(threads=threads, shapes=(shape,)))
        )
        == accepted
    )


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize(
    "warps,shape", [(1, (16, 24)), (2, (32, 48)), (4, (80, 96)), (8, (64, 128))]
)
def test_actual_cute_ordered_fragment_is_complete(
    dtype_name: str, warps: int, shape: tuple[int, int]
) -> None:
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    owners = []
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            dtype = cutlass.Float16 if dtype_name == "Float16" else cutlass.BFloat16
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
                ),
                atom_layout_mnk=(1, warps, 1),
            )
            for thread in range(warps * 32):
                coords = mma.get_slice(thread).partition_C(
                    cute.make_identity_tensor(shape)
                )
                assert cute.shape(coords) == mma.partition_shape_C(shape)
                owners.extend(
                    tuple(int(x) for x in coords[index])
                    for index in range(cute.size(coords))
                )
    assert len(owners) == len(set(owners)) == shape[0] * shape[1]
    assert set(owners) == {(m, n) for m in range(shape[0]) for n in range(shape[1])}


@pytest.mark.parametrize(
    "change", ["workspace", "graph", "config", "publication", "join"]
)
def test_completed_binding_rechecks_source_and_selection(change: str) -> None:
    original = transport.TerminalStoreCompletion.validate

    def invalidate(self: transport.TerminalStoreCompletion, lines: list[str]) -> None:
        if change == "workspace":
            self.plan = replace(self.plan, warp_workspace=None)
        elif change == "graph":
            assert self.transport.terminal is not None
            self.transport.terminal.output.store.args = (
                *self.transport.terminal.output.store.args[:3],
                True,
            )
        elif change == "config":
            self.cg.device_function.config.config[KEY] = "packed"
        elif change == "publication":
            self.emitted = ("pass",)
        else:
            # Remove the all-read join immediately after the completed MMA.
            at = lines.index(self.stage_lines[-1])
            assert lines[at + 1] == "cute.arch.sync_threads()"
            lines.pop(at + 1)
        original(self, lines)

    with (
        patch.object(transport.TerminalStoreCompletion, "validate", invalidate),
        pytest.raises(exc.BackendUnsupported, match="warp result transport changed"),
    ):
        _code("packed_direct")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _terminal_cases(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    out_arg: torch.Tensor,
    kind: hl.constexpr,
    output_dtype: hl.constexpr,
) -> torch.Tensor:
    dtype = cast("torch.dtype", output_dtype)
    out = torch.empty(out_arg.shape, dtype=dtype, device=left.device)
    for ba, gr in hl.tile([left.size(0), left.size(1)], block_size=[1, 1]):
        b = ba.id
        g = gr.id
        i = hl.arange(64)
        first = hl.dot(left[b, g, :, :], right[b, g, :, :])
        # Zero-padded input tail and original select/scale/cast stay in the
        # terminal recipe, while output ownership is complete in both axes.
        value = hl.dot(first.to(left.dtype), right[b, g, :, :])
        loaded = torch.where(
            (i[:, None] < 61) & (i[None, :] < 59), bias[b, g, :, :], 0.0
        )
        result = torch.where(value > 0, value * 0.25 + loaded, value + 0.0)
        if kind == "transpose":
            result = result.T
        if kind == "scan":
            result = hl.cumsum(result, 0)
        out[b, g, :, :] = result.to(dtype)
    return out


def _terminal_args(
    kind: str, dtype: torch.dtype, device: str = "cpu"
) -> tuple[object, ...]:
    return (
        torch.empty((2, 2, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 2, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((2, 2, 64, 64), dtype=torch.float32, device=device),
        torch.empty((2, 2, 64, 64), dtype=dtype, device=device),
        kind,
        dtype,
    )


@pytest.mark.parametrize("kind", ["transpose", "scan"])
def test_unsafe_terminal_stores_keep_rejection(kind: str) -> None:
    with _cpu_codegen():
        bound = _terminal_cases._bind_isolated(_terminal_args(kind, torch.float32))
        with pytest.raises((exc.BackendUnsupported, exc.InvalidConfig)):
            bound.to_code(_config("packed_direct"))


def test_complete_pointwise_terminal_retains_input_masks_and_casts() -> None:
    with _cpu_codegen():
        bound = _terminal_cases._bind_isolated(_terminal_args("full", torch.float16))
        source = bound.to_code(_config("packed_direct"))
    assert "chain_1_terminal_coords" in source and "cutlass.Float16" in source
    assert "operator.lt(" in source and "operator.gt(" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_terminal_typed_masked_multiple_axes_gpu(dtype: torch.dtype) -> None:
    args = _terminal_args("full", dtype, "cuda")
    left, right, bias, unused, _, _ = args
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    generator = torch.Generator(device="cuda").manual_seed(1701)
    # Dyadic operands avoid BF16-tie amplification while still changing every
    # batch/group, lane, and the masked terminal tail independently.
    left.copy_(
        torch.randint(-2, 3, left.shape, device="cuda", generator=generator) / 32
    )
    right.copy_(
        torch.randint(-2, 3, right.shape, device="cuda", generator=generator) / 32
    )
    bias.copy_(torch.randn(bias.shape, device="cuda", generator=generator) / 128)
    first = (left.double() @ right.double()).to(left.dtype).double()
    value = (first @ right.double()).float()
    padded = torch.zeros_like(value)
    padded[..., :61, :59] = bias[..., :61, :59]
    expected = torch.where(value > 0, value * 0.25 + padded, value + 0.0).to(dtype)
    outputs = []
    for mode in (mode for mode in transport.MODES if mode != "packed_register"):
        bound = _terminal_cases._bind_isolated(args)
        bound.set_config(_config(mode))
        result = bound(*args)
        torch.testing.assert_close(result, expected, atol=0.002, rtol=0.002)
        outputs.append(result)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("mixed", [False, True])
def test_seed_fanout_arena_gpu(mixed: bool) -> None:
    args = _args(stages=3, size=64, mixed=mixed, device="cuda")
    left, right, initial, _, _, _ = args
    assert (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and isinstance(initial, torch.Tensor)
    )
    generator = torch.Generator(device="cuda").manual_seed(1703)
    for tensor in (left, right, initial):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator) / 32
        )
    state = (left.double() @ right.double() + initial.double()).float()
    first = state.clone()
    for _ in range(2):
        typed = torch.float16 if mixed else left.dtype
        state = (
            state.to(typed).double() @ right.to(typed).double()
            + state.transpose(-1, -2).double() * 0.125
        ).float()
    expected = state + first * 0.0625
    outputs = []
    for mode in (mode for mode in transport.MODES if mode != "packed_register"):
        bound = _long_chain._bind_isolated(args)
        bound.set_config(_config(mode))
        value = bound(*args)
        torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
        outputs.append(value)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])


@pytest.mark.parametrize(
    "violation", ["mask", "missing_axis", "nonfresh", "read_output"]
)
def test_terminal_ownership_proof_rejects_mutated_original_graph(
    violation: str,
) -> None:
    original = transport.classify_terminal_store

    def invalidate(plan: ChainedMatmulPlan) -> transport.WarpTerminalStore | None:
        store = plan.store
        if violation == "mask":
            store.args = (*store.args[:3], plan.dots[0])
        elif violation == "missing_axis":
            target, indices, value, mask = store.args
            assert isinstance(indices, (tuple, list))
            store.args = (target, [0, *indices[1:]], value, mask)
        elif violation == "nonfresh":
            with patch(
                "helion._compiler.cute.chained_root_exports._has_fresh_output_allocation",
                return_value=False,
            ):
                return original(plan)
        else:
            assert plan.region is not None
            load = next(node for node in plan.region.nodes if node.target is hl.load)
            target = store.args[0]
            load.args = (target, *load.args[1:])
        return original(plan)

    with (
        patch.object(transport, "classify_terminal_store", invalidate),
        pytest.raises(exc.BackendUnsupported, match="admitted complete root"),
    ):
        _code("packed_direct")


def test_masked_input_tail_with_complete_output_codegen() -> None:
    from .test_cute_chained_root_typed_operands import _tail_args
    from .test_cute_chained_root_typed_operands import _tail_chain

    with _cpu_codegen():
        bound = _tail_chain._bind_isolated(_tail_args(33))
        original = bound.to_code(_config())
        direct = bound.to_code(_config("packed_direct"))
    assert _gemms(original) == _gemms(direct)
    assert "operator.lt(" in direct and "chain_1_terminal_coords" in direct


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_masked_input_tail_gpu() -> None:
    from .test_cute_chained_root_typed_operands import _tail_args
    from .test_cute_chained_root_typed_operands import _tail_chain

    args = _tail_args(33, "cuda")
    left, right, initial, valid = args
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
    assert isinstance(initial, torch.Tensor) and isinstance(valid, int)
    generator = torch.Generator(device="cuda").manual_seed(1707)
    for tensor in (left, right, initial):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator) / 32
        )
    masked = right.clone()
    masked[..., valid:] = 0
    first = (left.double() @ masked.double()).float().half().double()
    expected = (first @ first.transpose(-1, -2) + initial.double()).float()
    outputs = []
    for mode in (mode for mode in transport.MODES if mode != "packed_register"):
        bound = _tail_chain._bind_isolated(args)
        bound.set_config(_config(mode))
        value = bound(*args)
        torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
        outputs.append(value)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])
