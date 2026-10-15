"""Explicit full-loop configs stay distinct from persistent None (CPU only)."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target

import helion
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.backend import CuteBackend
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # These tests inspect compiler IR, which eager interpretation does not build.
    monkeypatch.setenv("HELION_INTERPRET", "0")
    initialized = torch.cuda.is_initialized()
    with _cpu_target(), _forbid_native_compile() as calls:
        yield
        assert all(call.call_count == 0 for call in calls)
    assert torch.cuda.is_initialized() == initialized


def make_spec(width: int, *, full: bool, backend: str = "cute") -> ConfigSpec:
    spec = ConfigSpec(backend=CuteBackend() if backend == "cute" else TritonBackend())
    spec.block_sizes.append(
        BlockSizeSpec(block_id=0, size_hint=1, min_size=1, max_size=1)
    )
    spec.reduction_loops.append(
        ReductionLoopSpec(
            block_id=1,
            size_hint=width,
            allow_non_power_of_two=backend == "cute",
            allow_full_extent=full,
        )
    )
    return spec


@pytest.mark.parametrize("width", [8, 128, 1024, 4096, 8192, 32768])
def test_full_and_persistent_have_distinct_flat_tokens(width: int) -> None:
    base = make_spec(width, full=True)
    item = base.reduction_loops[0]
    fragment = item._flat_fragment(base)
    assert fragment.high == width * 2
    assert item._normalize("reduction_loops", width) == width
    assert item._normalize("reduction_loops", None) is None
    assert item._normalize("reduction_loops", width * 2) is None
    assert item._encode_flat_value(base, width) == width
    assert item._encode_flat_value(base, None) == width * 2
    assert item._flat_config(base, lambda _: width) == width
    assert item._flat_config(base, lambda _: width * 2) is None
    choices = fragment.search_values()
    assert choices is not None and width in choices and width * 2 in choices
    assert width in fragment.pattern_neighbors(width * 2)
    if width >= 16:
        assert item._flat_config(base, lambda _: width // 2) == width // 2
        assert item._normalize("reduction_loops", width // 2) == width // 2
    with pytest.raises(InvalidConfig):
        item._flat_config(base, lambda _: width * 4)


@pytest.mark.parametrize("width", [129, 513, 3584, 7168])
def test_exact_padded_and_persistent_full_widths_remain_distinct(width: int) -> None:
    old, new = (make_spec(width, full=value) for value in (False, True))
    old_item, new_item = old.reduction_loops[0], new.reduction_loops[0]
    padded = 1 << (width - 1).bit_length()
    assert old_item._normalize("reduction_loops", padded) is None
    assert old_item._flat_fragment(old).high == padded
    assert new_item._flat_fragment(new).high == padded * 2
    assert new_item._normalize("reduction_loops", width) == width
    assert new_item._flat_config(new, lambda _: width) == width
    assert new_item._normalize("reduction_loops", padded) == padded
    assert new_item._flat_config(new, lambda _: padded) == padded
    sentinel = new_item._encode_flat_value(new, None)
    assert sentinel == padded * 2
    assert new_item._flat_config(new, lambda _: sentinel) is None
    assert new_item._normalize("reduction_loops", sentinel) is None


@pytest.mark.parametrize("width", [1, 2, 4])
def test_subfloor_extents_keep_persistent_encoding(width: int) -> None:
    old, new = (make_spec(width, full=value) for value in (False, True))
    assert new.reduction_loops[0]._normalize("reduction_loops", width) is None
    assert vars(old.reduction_loops[0]._flat_fragment(old)) == vars(
        new.reduction_loops[0]._flat_fragment(new)
    )


@pytest.mark.parametrize("value", [0, -1, 3, 12, 4096.0, "4096"])
def test_invalid_integer_chunks_are_not_newly_admitted(value: object) -> None:
    spec = ReductionLoopSpec(block_id=0, size_hint=4096, allow_full_extent=True)
    with pytest.raises(InvalidConfig):
        spec._normalize("reduction_loops", value)


@pytest.mark.parametrize(
    "width", [8, 128, 129, 513, 1024, 3584, 4096, 7168, 8192, 32768]
)
def test_default_and_reference_configs_keep_their_old_meaning(width: int) -> None:
    old, new = (make_spec(width, full=value) for value in (False, True))
    assert old.default_config() == new.default_config()
    assert old.autotune_reference_config() == new.autotune_reference_config()


@pytest.mark.parametrize("width", [128, 129, 513, 3584, 4096, 7168])
def test_complete_configs_roundtrip_without_hiding_persistent_caps(width: int) -> None:
    spec = make_spec(width, full=True)
    generation = ConfigGeneration(spec)
    padded = 1 << (width - 1).bit_length()
    for requested, expected in (
        (width, width),
        (padded, padded),
        (None, None if width <= 1024 else 1024),
    ):
        config = spec.default_config()
        config.config["reduction_loops"] = [requested]
        flat, effective = generation.strict_config_pair(config)
        assert effective.config["reduction_loops"] == [expected]
        assert generation.strict_unflatten(deepcopy(flat)) == effective
        assert generation.flatten(effective) == flat


def test_backend_chunk_and_indexed_reduction_caps_still_apply() -> None:
    spec = make_spec(4096, full=True)
    spec.max_reduction_loop = 2048
    config = spec.default_config()
    config.config["reduction_loops"] = [4096]
    spec.normalize(config.config)
    assert config.config["reduction_loops"] == [1024]
    indexed = make_spec(64, full=True)
    indexed.cute_indexed_reduction_block_ids.add(1)
    config = indexed.default_config()
    config.config["reduction_loops"] = [64]
    indexed.normalize(config.config)
    assert config.config["reduction_loops"] == [32]


@pytest.mark.parametrize("width", [128, 4096, 8192])
def test_legacy_backend_specs_keep_full_alias_and_flat_schema(width: int) -> None:
    spec = make_spec(width, full=False, backend="triton")
    item = spec.reduction_loops[0]
    assert item._normalize("reduction_loops", width) is None
    assert item._encode_flat_value(spec, None) == width
    assert item._flat_config(spec, lambda _: width) is None
    assert not any(
        row[0] == "reduction_loop_full_extent_encoding_v2"
        for row in spec.structural_fingerprint()
    )


def test_legacy_cached_flat_sentinel_has_a_different_schema_identity() -> None:
    old, new = (make_spec(128, full=value) for value in (False, True))
    # The identical numeric token used to mean None and now means full. A warm
    # transfer must re-encode its Config rather than trust a matching flat hash.
    assert old.reduction_loops[0]._flat_config(old, lambda _: 128) is None
    assert new.reduction_loops[0]._flat_config(new, lambda _: 128) == 128
    assert old.structural_fingerprint_hash() != new.structural_fingerprint_hash()
    assert old.cache_fingerprint_hash() != new.cache_fingerprint_hash()
    assert (
        old.projected_cache_fingerprint_hash() != new.projected_cache_fingerprint_hash()
    )


@pytest.mark.parametrize("width", [1, 2, 4, 7])
def test_capability_versions_all_shapes_even_when_local_encoding_is_unchanged(
    width: int,
) -> None:
    old, new = (make_spec(width, full=value) for value in (False, True))
    assert vars(old.reduction_loops[0]._flat_fragment(old)) == vars(
        new.reduction_loops[0]._flat_fragment(new)
    )
    assert old.structural_fingerprint_hash() != new.structural_fingerprint_hash()
    assert old.cache_fingerprint_hash() != new.cache_fingerprint_hash()
    assert (
        old.projected_cache_fingerprint_hash() != new.projected_cache_fingerprint_hash()
    )


@pytest.mark.parametrize("widths", [(1, 8), (4, 8), (128, 129), (4096, 4097)])
def test_encoding_capability_fingerprint_is_shape_invariant(
    widths: tuple[int, int],
) -> None:
    first, second = (make_spec(width, full=True) for width in widths)
    assert first.structural_fingerprint() == second.structural_fingerprint()
    assert (
        "reduction_loop_full_extent_encoding_v2",
        1,
    ) in first.structural_fingerprint()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for rows in hl.tile(m):
        values = x[rows, :].float()
        mean = values.sum(-1) / n
        centered = values - mean[:, None]
        variance = (centered * centered).sum(-1) / n
        out[rows, :] = (
            centered * torch.rsqrt(variance + eps)[:, None] * weight[:].float()
        ).to(x.dtype)
    return out


@pytest.mark.parametrize("widths", [(4, 8), (128, 129), (4096, 4097), (513, 7168)])
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_mixed_shape_autotune_reaches_search_boundary(
    widths: tuple[int, int], reverse: bool
) -> None:
    class ReachedSearch(Exception):
        pass

    ordered = tuple(reversed(widths)) if reverse else widths
    kernel = helion.kernel(rms_norm, backend="cute", static_shapes=True)
    args = [(torch.empty(1, width), torch.empty(width), 1e-6) for width in ordered]
    bounds = [kernel.bind(arg) for arg in args]
    keys = [
        kernel._get_bound_kernel_cache_key(arg, kernel._base_specialization_key(arg))
        for arg in args
    ]
    assert all(key is not None for key in keys)
    assert all(
        item.allow_full_extent
        for bound in bounds
        for item in bound.config_spec.reduction_loops
    )
    # Preserve real bindings/specs while supplying accelerator identity for the
    # CPU-only API check. Stop at autotune entry, before any search or execution.
    with (
        patch.object(kernel, "bind", side_effect=bounds),
        patch.object(kernel, "_get_bound_kernel_cache_key", side_effect=keys),
        patch(
            "helion.runtime.kernel._canonicalize_multi_shape_device",
            return_value=torch.device("cuda", 0),
        ),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch.object(CuteBackend, "autotune", side_effect=ReachedSearch) as search,
        pytest.raises(ReachedSearch),
    ):
        kernel.autotune_multi(args, cache_tag="full-loop-mixed-shape-cpu")
    search.assert_called_once()
    anchor, carrier = search.call_args.args
    assert anchor is bounds[0]
    assert isinstance(carrier, _MultiShapeAutotuneArgs)
    assert all(
        case[0] is bound and case[1][0] is arg[0]
        for case, bound, arg in zip(carrier.cases, bounds, args, strict=True)
    )


@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_binding_enables_full_encoding_only_for_cute(backend: str) -> None:
    kernel = helion.kernel(rms_norm, backend=backend, static_shapes=True)
    bound = kernel._bind_isolated((torch.empty(1, 4096), torch.empty(4096), 1e-6))
    assert len(bound.config_spec.reduction_loops) == 1
    assert bound.config_spec.reduction_loops[0].allow_full_extent == (backend == "cute")


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize(
    "shape,rows,threads",
    [
        ((1, 4096), 1, 512),
        ((3, 4096), 2, 256),
        ((16, 7168), 1, 1024),
        ((3, 7168), 2, 512),
    ],
)
def test_full_loop_codegen_keeps_packet_cache_and_two_reduction_path(
    name: str, shape: tuple[int, int], rows: int, threads: int
) -> None:
    function = rms_norm if name == "rms_norm" else layer_norm
    kernel = helion.kernel(function, backend="cute", static_shapes=True)
    bound = kernel._bind_isolated(
        (
            torch.empty(shape, dtype=torch.bfloat16),
            torch.empty(shape[1], dtype=torch.bfloat16),
            1e-6,
        )
    )
    spec = bound.config_spec
    owner = spec.reduction_loops[0].block_id
    chunk = 1 << (shape[1] - 1).bit_length()
    config = helion.Config(
        block_sizes=[rows],
        num_threads=[
            threads if block == owner else rows
            for block in spec.num_threads.valid_block_ids()
        ],
        reduction_loops=[chunk],
        cute_vector_widths=[
            8 if block == owner else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_reduction_reloads=["register"],
    )
    with bound.env:
        flat, effective = ConfigGeneration(spec).strict_config_pair(config)
        assert effective.config["reduction_loops"] == [chunk]
        assert ConfigGeneration(spec).strict_unflatten(deepcopy(flat)) == effective
    source = bound.to_code(effective)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(ast.unparse(node.func) == "cute.make_rmem_tensor" for node in calls)
    assert f"_REDUCTION_BLOCK_1 = {chunk}" in source
    assert "cute.arch.load(x.iterator" in source
    assert "_fuse_cache_" in source
    assert f"block=({threads}, {rows}, 1)" in source
    expected = 1 if name == "rms_norm" else 2
    assert (
        sum(
            ast.unparse(node.func) == "_cute_grouped_reduce_shared_two_stage"
            for node in calls
        )
        == expected
    )


@pytest.mark.parametrize("width", [9, 129, 4097, 7168])
def test_padded_full_chunk_is_reachable_from_persistent_search_neighbor(
    width: int,
) -> None:
    spec = make_spec(width, full=True)
    item = spec.reduction_loops[0]
    padded = 1 << (width - 1).bit_length()
    fragment = item._flat_fragment(spec)
    persistent = item._encode_flat_value(spec, None)
    choices = fragment.search_values()
    assert choices is not None and padded in choices and persistent in choices
    assert padded in fragment.pattern_neighbors(persistent)
    assert persistent in fragment.pattern_neighbors(padded)
    assert item._flat_config(spec, lambda _: padded) == padded
    assert item._flat_config(spec, lambda _: persistent) is None


@pytest.mark.parametrize(
    "width,max_loop,threshold",
    [(129, 128, 64), (513, 256, 128), (7168, 2048, 1024)],
)
def test_padded_chunk_does_not_bypass_live_backend_caps(
    width: int, max_loop: int, threshold: int
) -> None:
    old, new = (make_spec(width, full=value) for value in (False, True))
    for spec in (old, new):
        spec.max_reduction_loop = max_loop
        spec.reduction_loop_force_threshold = threshold
    assert old.default_config() == new.default_config()
    assert old.autotune_reference_config() == new.autotune_reference_config()
    padded = 1 << (width - 1).bit_length()
    for requested in (None, padded):
        config = new.default_config()
        config.config["reduction_loops"] = [requested]
        generation = ConfigGeneration(new)
        flat, effective = generation.strict_config_pair(config)
        assert effective.config["reduction_loops"] == [threshold]
        assert generation.strict_unflatten(deepcopy(flat)) == effective


def test_padded_indexed_reduction_retains_single_warp_cap() -> None:
    spec = make_spec(65, full=True)
    spec.cute_indexed_reduction_block_ids.add(1)
    config = spec.default_config()
    config.config["reduction_loops"] = [128]
    generation = ConfigGeneration(spec)
    flat, effective = generation.strict_config_pair(config)
    assert effective.config["reduction_loops"] == [32]
    assert generation.strict_unflatten(deepcopy(flat)) == effective
