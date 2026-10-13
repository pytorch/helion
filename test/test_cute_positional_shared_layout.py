from __future__ import annotations

from copy import deepcopy
import math
import re
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_positional_pointwise_cache import mutated_weight
from .test_cute_positional_root import square_gram
from .test_cute_positional_warp_mma import loop_exported_state
import helion
from helion import exc
from helion._compiler.cute.positional_root import PositionalEmitter
from helion._compiler.cute.positional_shared_layout import positional_xor_swizzle
from helion._compiler.cute.positional_shared_layout import shared_tile_layout
from helion.autotuner.config_spec import CUTE_POSITIONAL_SHARED_LAYOUT_KEY as KEY
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def matrix_cache_gram(x: torch.Tensor) -> torch.Tensor:
    batch, rows, _ = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        values = x[tb, :, :].float().exp().to(x.dtype)
        out[tb, :, :] = hl.dot(values, values.transpose(-1, -2))
    return out


def _swizzle(offset: int, parameters: tuple[int, int, int]) -> int:
    bits, base, shift = parameters
    return offset ^ ((offset >> shift) & (((1 << bits) - 1) << base))


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.int64]
)
@pytest.mark.parametrize(
    "shape", [(3, 64), (1, 3, 32), (2, 3, 16), (3, 5, 128), (2, 1, 3, 64)]
)
def test_xor_is_a_bijection_in_each_row_and_preserves_storage(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    element_bytes = torch.empty((), dtype=dtype).element_size()
    parameters = positional_xor_swizzle(shape, element_bytes)
    if parameters is None:
        # A 16-byte row has no column bits above the preserved grain.
        assert shape[-1] * element_bytes <= 16
        return
    bits, base, shift = parameters
    assert bits <= shift
    assert base + bits <= shape[-1].bit_length() - 1
    offsets = [_swizzle(offset, parameters) for offset in range(math.prod(shape))]
    assert sorted(offsets) == list(range(math.prod(shape)))
    grain_mask = 16 // element_bytes - 1
    for source, target in enumerate(offsets):
        assert source // shape[-1] == target // shape[-1]
        assert source & grain_mask == target & grain_mask
        assert _swizzle(target, parameters) == source


@pytest.mark.parametrize(
    "shape", [(), (64,), (1, 64), (3, 1), (3, 2), (3, 4), (3, 5), (3, 40), (2, 7, 40)]
)
@pytest.mark.parametrize("element_bytes", [1, 2, 4])
def test_unsupported_xor_uses_row_without_padding(
    shape: tuple[int, ...], element_bytes: int
) -> None:
    assert positional_xor_swizzle(shape, element_bytes) is None
    assert shared_tile_layout(shape, element_bytes, "xor") == shared_tile_layout(
        shape, element_bytes, "row"
    )


def test_explicit_element_width_and_swizzle_formulas() -> None:
    assert positional_xor_swizzle((3, 64), 4) == (3, 2, 4)
    assert positional_xor_swizzle((3, 64), 2) == (3, 3, 3)
    assert positional_xor_swizzle((3, 64), 1) == (2, 4, 2)
    assert positional_xor_swizzle((3, 64), 8) == (3, 1, 5)
    for width in (0, 3, 32):
        assert positional_xor_swizzle((3, 64), width) is None
    assert shared_tile_layout((), 4, "column") == "cute.make_layout(1)"
    assert (
        shared_tile_layout((2, 3, 64), 4, "row")
        == "cute.make_layout((2, 3, 64), stride=(192, 64, 1))"
    )
    with pytest.raises(ValueError, match="unsupported positional shared layout"):
        shared_tile_layout((3, 64), 4, "unknown")


def test_column_default_normalization_and_source_are_unchanged() -> None:
    with _cpu_codegen():
        bound = square_gram._bind_isolated((torch.empty(2, 32, 32),))
        spec = bound.config_spec
        assert spec.cute_positional_shared_layout_search_enabled
        default = spec.default_config()
        assert KEY not in default
        explicit = deepcopy(default)
        explicit.config[KEY] = "column"
        assert spec.normalized_config(explicit) == default
        assert bound.to_code(default) == bound.to_code(explicit)
        assert spec.flatten_missing_field_default(KEY, {}) == (True, "column")
        assert spec.normalized_config(default.config | {KEY: None}) == default
        for value in (0, True, "swizzle"):
            with pytest.raises(exc.InvalidConfig, match=KEY):
                spec.normalized_config(default.config | {KEY: value})


@pytest.mark.parametrize("mma", [False, True])
def test_layouts_preserve_allocation_bytes_cache_scope_and_snapshots(mma: bool) -> None:
    original = PositionalEmitter.emit
    allocations: list[tuple[int, int, int]] = []

    def observe(self: PositionalEmitter) -> list[str]:
        source = original(self)
        allocations.append(
            (self.allocated_bytes, self.snapshot_bytes, self.pointwise_cache_bytes)
        )
        return source

    sources = []
    with _cpu_codegen(), patch.object(PositionalEmitter, "emit", observe):
        bound = mutated_weight._bind_isolated(
            (torch.empty(2, 32, 32), torch.empty(2, 32))
        )
        for layout in ("column", "row", "xor"):
            config = bound.config_spec.default_config()
            config.config.update(
                {
                    KEY: layout,
                    "cute_positional_mma": mma,
                    "cute_positional_pointwise_cache_bytes": 256,
                }
            )
            sources.append(bound.to_code(config))
    assert allocations == [(4480, 256, 128)] * 3
    assert "stride=" in sources[1]
    assert "cute.make_swizzle(3, 2, 3)" in sources[2]
    allocated = [
        re.findall(r"cute.arch.alloc_smem\([^\n]+?alignment=128\)", source)
        for source in sources
    ]
    assert allocated[0] == allocated[1] == allocated[2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_matrix_cache_uses_element_width_swizzle(dtype: torch.dtype) -> None:
    with _cpu_codegen():
        bound = matrix_cache_gram._bind_isolated((torch.empty(2, 8, 20, dtype=dtype),))
        config = bound.config_spec.default_config()
        config.config.update(
            {
                KEY: "xor",
                "cute_positional_mma": True,
                "cute_positional_pointwise_cache_bytes": 1024,
            }
        )
        source = bound.to_code(config)
    assert "ptp_cache_" in source
    assert "cute.make_swizzle(2, 3, 2)" in source


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@requires_cuda
@pytest.mark.parametrize("layout", ["row", "xor"])
@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("cache", [0, 256])
def test_gpu_layout_changes_only_addresses(layout: str, mma: bool, cache: int) -> None:
    torch.manual_seed(456)
    x = (torch.randn(2, 3, 8, 20, device="cuda") * 0.125).bfloat16()
    gate = torch.randn(2, 3, 20, device="cuda") * 0.05
    state = torch.randn(2, 8, 8, device="cuda") * 0.1

    def run(mode: str) -> object:
        args = (
            x,
            gate,
            state.clone(),
            torch.full((2, 3, 8, 8), torch.nan, device="cuda"),
        )
        bound = loop_exported_state._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config.update(
            {
                KEY: mode,
                "cute_positional_mma": mma,
                "cute_positional_pointwise_cache_bytes": cache,
            }
        )
        return bound.compile_config(config)(*args)

    torch.testing.assert_close(run(layout), run("column"), rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["row", "xor"])
def test_gpu_half_matrix_cache_layout(dtype: torch.dtype, layout: str) -> None:
    torch.manual_seed(567)
    x = (torch.randn(2, 8, 20, device="cuda") * 0.125).to(dtype)

    def run(mode: str) -> object:
        bound = matrix_cache_gram._bind_isolated((x,))
        config = bound.config_spec.default_config()
        config.config.update(
            {
                KEY: mode,
                "cute_positional_mma": True,
                "cute_positional_pointwise_cache_bytes": 1024,
            }
        )
        return bound.compile_config(config)(x)

    torch.testing.assert_close(run(layout), run("column"), rtol=0, atol=0)
