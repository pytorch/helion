"""Block sizes and extents a host helper over tunables computes, inline.

``hl.tile([n], block_size=[helion.next_power_of_2(foo)])`` passes a value no
host variable names.  The tile takes the value the helper gives under the
config being compiled, as a static block size like a literal one, on each
axis of a tile; a loop extent written the same way is that value too.  Code
generation only; test_register_tunable.py runs one.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton", "cute"]),
    # Every test binds a kernel and reads its config spec or generated code.
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="inspects config specs and generated code; ref mode has neither",
    ),
]


@helion.kernel(autotune_effort="none")
def scale_by_power_of_2_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[helion.next_power_of_2(foo)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_cdiv_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile([n], block_size=[helion.cdiv(foo, 4)]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_2d_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, helion.next_power_of_2(foo)]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_helper_extent(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_n in hl.tile(helion.next_power_of_2(foo)):
        out[tile_n] = x[tile_n] * 2.0
    return out


def _render(kernel: helion.Kernel, x: torch.Tensor, foo: int) -> str:
    """``kernel``'s code for ``x`` under its default config with ``foo``."""
    with cpu_render_target():
        bound = cpu_bind(kernel, (x,))
        config = bound.config_spec.default_config().config | {"foo": foo}
        return bound.to_code(helion.Config(**config))


def _literal_block_sizes(code: str) -> dict[int, int]:
    """Each ``_BLOCK_SIZE_<i>`` the code defines as a literal: a Triton
    ``tl.constexpr`` or a CuTe host constant."""
    return {
        int(block): int(value)
        for block, value in re.findall(
            r"_BLOCK_SIZE_(\d+) = (?:tl\.constexpr\()?(\d+)\b", code
        )
    }


@pytest.mark.parametrize(
    ("kernel", "shape", "block_sizes"),
    [
        (scale_by_power_of_2_tile, (1000,), {0: 64}),
        (scale_by_cdiv_tile, (1000,), {0: 16}),
        # CuTe sizes its thread block by both axes: the helper's had stayed
        # symbolic there while device code printed its value (refused).
        (scale_by_2d_tile, (64, 1000), {0: 16, 1: 64}),
    ],
)
def test_block_takes_the_helper_value_under_the_config(
    kernel: helion.Kernel, shape: tuple[int, ...], block_sizes: dict[int, int]
) -> None:
    # foo=64 is not the default (16): a block that ignored the config shows.
    code = _render(kernel, torch.empty(shape), foo=64)
    assert _literal_block_sizes(code) == block_sizes, code


@helion.kernel(autotune_effort="none")
def scale_by_cdiv_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for i in hl.grid(helion.cdiv(foo, 2)):
        out[i] = x[i] * 2.0
    return out


def test_grid_takes_the_helper_value_under_the_config() -> None:
    code = _render(scale_by_cdiv_grid, torch.empty(1000), foo=64)
    # The launch covers the 32 programs ``helion.cdiv(foo, 2)`` gives.
    assert re.search(r"_launcher\(\w+, \(\(?32\b", code), code


@helion.kernel(autotune_effort="none")
def scale_by_device_helper_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=helion.next_power_of_2(foo)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_device_helper_extent(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.zeros_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(helion.next_power_of_2(foo)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@pytest.mark.parametrize(
    "kernel", [scale_by_device_helper_tile, scale_by_device_helper_extent]
)
def test_a_device_helper_over_a_tunable_is_refused(kernel: helion.Kernel) -> None:
    """In device code next_power_of_2 ran on foo's hint, so the inner loop's
    block size or extent was 8192 whatever the config: refused, naming the
    tunable."""
    with pytest.raises(
        helion.exc.NotAllowedOnDevice, match="next_power_of_2 over the tunable 'foo'"
    ):
        _render(kernel, torch.empty(64, 1024), foo=64)


@helion.kernel(autotune_effort="none")
def row_sum_by_tunable_tile(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    for tile_m in hl.tile(m, block_size=foo):
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    ("kernel", "args", "heuristic"),
    [
        # ``block_size=[None, None, k_block]``: a named helper over a tunable.
        (
            import_example("matmul_split_k").matmul_split_k,
            (
                torch.empty(512, 4096, dtype=torch.float16),
                torch.empty(4096, 512, dtype=torch.float16),
            ),
            "triton_b200_formula_matmul",
        ),
        (row_sum_by_tunable_tile, (torch.empty(4096, 1024),), "triton_reduction"),
    ],
    ids=["split_k", "tunable_tile"],
)
@skipIfTileIR("names the Triton backend's seed heuristics; TileIR registers its own")
def test_seed_heuristics_read_a_tunable_block_size(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], heuristic: str
) -> None:
    # The seed heuristics read fixed block sizes outside code generation,
    # with no current host function: a block size a host variable names
    # stays symbolic for them to resolve.
    with cpu_render_target():
        spec = cpu_bind(kernel, args).config_spec
    assert heuristic in spec.autotuner_heuristics
    assert spec.compiler_seed_configs


@helion.kernel(
    static_shapes=False, config=helion.Config(indexing="tensor_descriptor", foo=16)
)
def scale_by_tunable_descriptor_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(8, 64))
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, foo * 2]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@skipUnlessBackends(["triton"])
def test_descriptor_bind_reads_a_tunable_block_size() -> None:
    # Binding caps the descriptors' extents by the config's block sizes,
    # outside code generation too.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        bound = scale_by_tunable_descriptor_tile.bind((torch.empty(256, 256),))
    assert bound.env.tensor_descriptor_layout_guards


@helion.kernel(autotune_effort="none", static_shapes=False)
def scale_by_named_mixed_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = helion.next_power_of_2(foo) * helion.next_power_of_2(n)
    for tile_n in hl.tile(n, block_size=block):
        out[tile_n] = x[tile_n] * block
    return out


def test_a_named_block_size_over_an_inline_helper_keeps_its_variable() -> None:
    # ``block`` names the product, but neither power of two alone: the host's
    # block size and the device's scale take the variable, rather than foo's
    # value times n's power of two, which nothing names (KeyError).
    code = _render(scale_by_named_mixed_tile, torch.empty(1000), foo=64)
    assert re.search(r"_BLOCK_SIZE_0 = block\b", code), code
    assert re.search(r"def _helion_\w+\([^)]*\bblock\b", code), code


@helion.kernel(autotune_effort="none")
def scale_by_scaled_helper_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = 2 * helion.next_power_of_2(foo)
    for tile_n in hl.tile([n], block_size=[block]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@helion.kernel(autotune_effort="none")
def scale_by_rebound_helper_tile(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
    block = helion.next_power_of_2(foo)
    block = block * 2
    for tile_n in hl.tile([n], block_size=[block]):
        out[tile_n] = x[tile_n] * 2.0
    return out


@pytest.mark.parametrize(
    "kernel", [scale_by_scaled_helper_tile, scale_by_rebound_helper_tile]
)
def test_a_variable_over_an_inline_helper_takes_the_config_value(
    kernel: helion.Kernel,
) -> None:
    # No variable names the helper's result alone, so the block takes 2 * 64
    # as a literal under foo=64, where origin/main passed ``block`` in.
    code = _render(kernel, torch.empty(1000), foo=64)
    assert _literal_block_sizes(code) == {0: 128}, code


def test_bind_time_facts_leave_a_helper_block_unfixed() -> None:
    # Bind computes some facts once for every config (``_immovable_extent``,
    # read by the affine ranges of cross-loop dependencies and by trip
    # counts): a block the config sizes has no width they may assume, while
    # a block a host variable names keeps the symbolic block it had.
    from helion._compiler.device_ir_analysis import _immovable_extent

    with cpu_render_target():
        helper = cpu_bind(scale_by_power_of_2_tile, (torch.empty(1000),))
        named = cpu_bind(row_sum_by_tunable_tile, (torch.empty(4096, 1024),))
    (block,) = helper.env.block_sizes
    assert block.block_size_source.varies_with_config
    with helper.env:
        assert _immovable_extent(helper.env, helper.config_spec, block.block_id) is None
    assert not named.env.block_sizes[0].block_size_source.varies_with_config


def test_extent_takes_the_helper_value_under_the_config() -> None:
    code = _render(scale_helper_extent, torch.empty(1000), foo=64)
    # The tile mask bounds the 1000 elements at the 64 the loop covers.
    assert re.search(r"< 64\b", code), code
