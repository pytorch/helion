from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_collective_dot import _full_slice_dot
from test.test_cute_collective_seeds import _independent_matmuls
from test.test_cute_collective_tf32 import _config
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel
from test.test_cute_collective_warp_tf32 import _seeded_fp32

import helion

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _packet_config(
    config: helion.Config, enabled: bool, *, recipe: str = "vector_unrolled"
) -> helion.Config:
    return helion.Config.from_dict(
        config.config
        | {"cute_collective_recipe": recipe, "cute_collective_operand_packets": enabled}
    )


def _recover_scalar_stores(
    packet_source: str, control_source: str
) -> tuple[ast.Module, int]:
    control = ast.parse(control_source)
    packet = ast.parse(packet_source)
    control_loops = {
        ast.unparse(node.target): node
        for node in ast.walk(control)
        if isinstance(node, ast.For)
    }
    count = 0
    for parent in ast.walk(packet):
        for field in ("body", "orelse"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            for index in range(len(statements) - 4, -1, -1):
                if index + 4 > len(statements):
                    continue
                allocation, loop, view, store = statements[index : index + 4]
                if not (
                    isinstance(allocation, ast.Assign)
                    and isinstance(allocation.targets[0], ast.Name)
                    and allocation.targets[0].id.startswith("operand_packet")
                    and isinstance(loop, ast.For)
                ):
                    continue
                original = control_loops[ast.unparse(loop.target)]
                assert isinstance(loop.body[-1], ast.Assign)
                assert ast.dump(loop.body[-1].value) == ast.dump(
                    original.body[-1].value
                )
                assert isinstance(view, ast.Assign)
                assert isinstance(store, ast.Expr) and isinstance(store.value, ast.Call)
                assert ast.unparse(store.value.func) == "cute.copy"
                loop.body[-1].targets = copy.deepcopy(original.body[-1].targets)
                statements[index : index + 4] = [loop]
                count += 1
    return packet, count


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_packet_source_changes_only_the_two_operand_store_groups(
    compute: str, recipe: str, static_shapes: bool
) -> None:
    args = (torch.empty((3, 130, 64)), torch.empty((3, 64, 70)))
    bound = _cpu_bind(_kernel(batched=True, static_shapes=static_shapes), args)
    config = _config(64, 32, 32, compute=compute)
    control = bound.to_code(_packet_config(config, False, recipe=recipe))
    packet = bound.to_code(_packet_config(config, True, recipe=recipe))
    restored, count = _recover_scalar_stores(packet, control)
    assert count == 2
    assert ast.dump(restored) == ast.dump(ast.parse(control))
    assert packet.count("cvt_f32_tf32") == control.count("cvt_f32_tf32") == 2
    assert "cutlass.Float16" not in packet and "cutlass.BFloat16" not in packet


@pytest.mark.parametrize(
    "mode",
    ["half", "scalar_recipe", "static_partial_k", "async_ring", "ieee", "tf32x3"],
)
def test_packet_request_preserves_unsupported_scalar_or_copy_paths(mode: str) -> None:
    dtype = torch.float16 if mode == "half" else torch.float32
    precision = mode if mode in ("ieee", "tf32x3") else "tf32"
    k = 64
    args = (torch.empty((3, 130, k), dtype=dtype), torch.empty((3, k, 64), dtype=dtype))
    bound = _cpu_bind(
        _kernel(batched=True, static_shapes=True, precision=precision), args
    )
    config = _config(
        64,
        32,
        32,
        compute="warp",
        copy="async_cached" if mode == "async_ring" else "scalar",
    )
    if mode == "static_partial_k":
        args = (torch.empty((130, 21)), torch.empty((21, 70)))
        bound = _cpu_bind(_full_slice_dot, args)
        config = helion.Config(
            block_sizes=[64, 32],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1],
            cute_collective_mma=True,
            cute_collective_compute="warp",
        )
    if mode == "async_ring":
        # A computed A cannot use an async ring, so use the direct jagged input.
        args = _jagged_inputs(batch=3, rows=137, k=64, n=64)
        bound = _cpu_bind(_kernel(), args)
        config = helion.Config.from_dict(config.config | {"cute_collective_stages": 2})
    recipe = "scalar" if mode == "scalar_recipe" else "vector_unrolled"
    control = bound.to_code(_packet_config(config, False, recipe=recipe))
    packet = bound.to_code(_packet_config(config, True, recipe=recipe))
    assert "operand_packet" not in packet
    assert ast.dump(ast.parse(packet)) == ast.dump(ast.parse(control))


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_packet_keeps_fp32_seed_writes_and_seed_before_mma(compute: str) -> None:
    args = (torch.empty((65, 64)), torch.empty((64, 70)), torch.empty((65, 70)))
    bound = _cpu_bind(_seeded_fp32, args)
    config = helion.Config(
        block_sizes=[64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1] * 3,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_native_seeded=True,
    )
    control = bound.to_code(_packet_config(config, False))
    packet = bound.to_code(_packet_config(config, True))
    restored, count = _recover_scalar_stores(packet, control)
    assert count == 2
    assert ast.dump(restored) == ast.dump(ast.parse(control))
    assert "collective_seed" in packet


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_packet_config_rejects_non_boolean_choices(value: object) -> None:
    bound = _cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=65, k=64, n=64))
    config = helion.Config.from_dict(
        _config().config | {"cute_collective_operand_packets": value}
    )
    with pytest.raises(
        helion.exc.InvalidConfig, match="cute_collective_operand_packets"
    ):
        bound.to_code(config)


@pytest.mark.parametrize("value", [False, True])
def test_packet_choice_survives_flattening(value: bool) -> None:
    bound = _cpu_bind(_kernel(), _jagged_inputs(batch=3, rows=65, k=64, n=64))
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        _flat, restored = generation.canonicalize_flat(
            generation.flatten(_packet_config(_config(), value))
        )
    assert restored["cute_collective_operand_packets"] is value


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_partial_global_k_masks_remain_inside_each_packet(
    compute: str, static_shapes: bool
) -> None:
    arguments = (torch.empty((3, 130, 78)), torch.empty((3, 78, 70)))
    bound = _cpu_bind(_kernel(batched=True, static_shapes=static_shapes), arguments)
    config = _config(64, 32, 32, compute=compute)
    control = bound.to_code(_packet_config(config, False))
    packet = bound.to_code(_packet_config(config, True))
    restored, count = _recover_scalar_stores(packet, control)
    assert count == 2
    assert ast.dump(restored) == ast.dump(ast.parse(control))


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_mixed_dtype_operand_pool_retains_both_lifetimes(compute: str) -> None:
    arguments = (
        torch.empty((65, 78), dtype=torch.float32),
        torch.empty((78, 71), dtype=torch.float32),
        torch.empty((131, 37), dtype=torch.bfloat16),
        torch.empty((37, 39), dtype=torch.bfloat16),
    )
    bound = _cpu_bind(_independent_matmuls, arguments)
    config = helion.Config(
        block_sizes=[64, 32, 32, 64, 32, 32],
        num_threads=[4, 32, 1, 4, 32, 1],
        cute_vector_widths=[1] * 6,
        cute_collective_mma=True,
        cute_collective_compute=compute,
    )
    control = bound.to_code(_packet_config(config, False))
    packet = bound.to_code(_packet_config(config, True))
    restored, count = _recover_scalar_stores(packet, control)
    assert count == 2
    assert ast.dump(restored) == ast.dump(ast.parse(control))
    assert "cute.arch.alloc_smem(cutlass.Uint8" in packet
