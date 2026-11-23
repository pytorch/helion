"""The component-packet seeds of the CuTe pointwise heuristic
(``CutePointwiseVecHeuristic._component_packet_seeds``): one packet of the
device's widest store per thread for a store whose last two subscripts are a
lane tile and the tile of its components (``test_cute_component_packets.py``
covers the lowering)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import CutePointwiseVecHeuristic
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.unfold_kernels import sliding_windows
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scatter_component(
    pair: torch.Tensor, grad: torch.Tensor, component: hl.constexpr
) -> torch.Tensor:
    elements, components = pair.shape
    out = torch.empty([elements, components], dtype=pair.dtype, device=pair.device)
    for tile_e, tile_c in hl.tile([elements, components]):
        chosen = (tile_c.index == component)[None, :]
        out[tile_e, tile_c] = torch.where(
            chosen, grad[tile_e][:, None], pair[tile_e, tile_c]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cube_scale(x: torch.Tensor) -> torch.Tensor:
    m, n, k = x.shape
    out = torch.empty([m, n, k], dtype=x.dtype, device=x.device)
    for tile_k, tile_m, tile_n in hl.tile([k, m, n]):
        out[tile_m, tile_n, tile_k] = x[tile_m, tile_n, tile_k] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_scale(x: torch.Tensor) -> torch.Tensor:
    # The component tile in a loop of its own: no common loop order.
    elements, components = x.shape
    out = torch.empty_like(x)
    for tile_e in hl.tile(elements):
        for tile_c in hl.tile(components):
            out[tile_e, tile_c] = x[tile_e, tile_c] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sibling_pairs(x: torch.Tensor) -> torch.Tensor:
    # Literal components: a one-dim tile, which the contiguous-axis seeds fit.
    n = x.size(0)
    out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile, 0] = x[tile]
        out[tile, 1] = x[tile] * 2
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_in_place(x: torch.Tensor) -> torch.Tensor:
    for tile_e, tile_c in hl.tile(x.size()):
        x[tile_e, tile_c] = x[tile_e, tile_c] * 2
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_pairs_and_vector(
    pair: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Two root grids: a store of every component, then a pointwise loop.
    elements, components = pair.shape
    out = torch.empty_like(pair)
    scaled = torch.empty_like(y)
    for tile_e, tile_c in hl.tile([elements, components]):
        out[tile_e, tile_c] = pair[tile_e, tile_c] * 2
    for tile in hl.tile(y.size(0)):
        scaled[tile] = y[tile] * 2
    return out, scaled


def _scatter_args(
    elements: int = 1 << 20, components: int = 2, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.empty(elements, components, dtype=dtype),
        torch.empty(elements, dtype=dtype),
        0,
    )


_SEED_KEYS = {"block_sizes", "num_threads", "cute_vector_widths", "loop_orders"}


def _component_seeds(bound: BoundKernel) -> list[dict[str, object]]:
    """The seeds that set the tiles, threads, vector widths and loop order
    alone and keep one tile on a single thread: the component-packet seeds."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if set(seed.config) == _SEED_KEYS and 1 in seed.config["num_threads"]
    ]


def _bind(
    kernel: object, args: tuple[object, ...], capability: tuple[int, int] = (10, 0)
) -> BoundKernel:
    return cpu_bind(kernel, args, capability=capability)  # type: ignore[arg-type]


def test_seeds_read_the_environment_they_are_given() -> None:
    # A heuristic's seeds are also asked of a bound kernel, outside its bind
    # (the pointwise seed tests of test_autotuner_heuristics.py call
    # ``get_seed_configs`` that way): the extent's proof must come from the
    # environment passed in, not from a current one.
    with cpu_render_target():
        bound = _bind(_scatter_component, _scatter_args())
        seeds = _component_seeds(bound)
    assert bound.host_function is not None
    again = CutePointwiseVecHeuristic.get_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    assert [dict(seed.config) for seed in again][-2:] == seeds and len(seeds) == 2


def test_seeds_give_each_thread_one_widest_packet() -> None:
    with cpu_render_target():
        bound = _bind(_scatter_component, _scatter_args())
        seeds = _component_seeds(bound)
        # 128 and 256 threads of one 32-byte packet: 4 pairs of fp32.
        assert [seed["block_sizes"] for seed in seeds] == [[512, 2], [1024, 2]]
        assert [seed["num_threads"] for seed in seeds] == [[128, 1], [256, 1]]
        assert [seed["cute_vector_widths"] for seed in seeds] == [[4, 1]] * 2
        assert [seed["loop_orders"] for seed in seeds] == [[[0, 1]]] * 2
        # They follow the seeds the kernel had.
        every = [dict(seed.config) for seed in bound.config_spec.compiler_seed_configs]
        assert every[-2:] == seeds
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    # The seed renders the packet: one flush of every component of 4 lanes.
    assert "for lane_1 in cutlass.range_constexpr(2):" in code
    assert code.count("_cute_store_u32_vec(out.iterator") == 1
    assert ".store(" not in code


# The packet is the device's widest store of whole elements: its lanes follow
# the dtype, the components and the device.
@pytest.mark.parametrize(
    ("components", "dtype", "capability", "width"),
    [
        (2, torch.float32, (9, 0), 2),
        (2, torch.bfloat16, (10, 0), 8),
        (4, torch.bfloat16, (10, 0), 4),
        (4, torch.bfloat16, (9, 0), 2),
    ],
    ids=["sm90", "bf16", "four-bf16", "four-bf16-sm90"],
)
def test_packet_lanes_follow_the_store(
    components: int, dtype: torch.dtype, capability: tuple[int, int], width: int
) -> None:
    with cpu_render_target():
        bound = _bind(
            _scatter_component, _scatter_args(1 << 20, components, dtype), capability
        )
        seeds = _component_seeds(bound)
    assert [seed["cute_vector_widths"] for seed in seeds] == [[width, 1]] * 2
    assert [seed["block_sizes"] for seed in seeds] == [
        [128 * width, components],
        [256 * width, components],
    ]


def test_short_lane_axis_gets_fewer_threads() -> None:
    # 64 elements: one tile of 16 threads, a packet each.
    with cpu_render_target():
        seeds = _component_seeds(_bind(_scatter_component, _scatter_args(64)))
    assert [seed["block_sizes"] for seed in seeds] == [[64, 2]]
    assert [seed["num_threads"] for seed in seeds] == [[16, 1]]


def test_lane_and_components_go_innermost_in_a_wider_tile() -> None:
    # The tile lists the components first: the seed orders the rows, then
    # the lane, then the components.
    with cpu_render_target():
        bound = _bind(_cube_scale, (torch.empty(64, 4096, 2),))
        seeds = _component_seeds(bound)
        assert [seed["loop_orders"] for seed in seeds] == [[[1, 2, 0]]] * 2
        assert [seed["num_threads"][::2] for seed in seeds] == [[1, 128], [1, 256]]
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 1, 4]] * 2
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    assert code.count("_cute_store_u32_vec(out.iterator") == 1


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        # Four fp32 or eight bf16 components are a 16-byte packet already:
        # a vector loop over them stores and loads whole packets.
        (_scatter_component, _scatter_args(1 << 20, 4)),
        (_scatter_component, _scatter_args(1 << 20, 8, torch.bfloat16)),
        # 1024 columns are no components of an element.
        (_scatter_component, _scatter_args(4096, 1024)),
        (_nested_scale, (torch.empty(4096, 2),)),
        (_sibling_pairs, (torch.empty(4096),)),
        # The lowering packs fp16, bf16 and fp32 stores only.
        (_scatter_component, _scatter_args(1 << 20, 2, torch.int32)),
        (_scatter_component, _scatter_args(1 << 20, 2, torch.float64)),
        # No chunk of 4 pairs may straddle the extent: the lowering asks for
        # an extent that is a multiple of the vector width.
        (_scatter_component, _scatter_args((1 << 20) + 2)),
    ],
    ids=[
        "four-fp32",
        "eight-bf16",
        "wide-rows",
        "nested-tiles",
        "literal-components",
        "int32",
        "fp64",
        "extent-not-multiple",
    ],
)
def test_no_seeds_without_a_packet_of_several_elements(
    kernel: object, args: tuple[object, ...]
) -> None:
    with cpu_render_target():
        assert _component_seeds(_bind(kernel, args)) == []


def test_search_space_does_not_depend_on_the_lane_extent() -> None:
    # Neighbouring shapes share one search space, and the same seeds where
    # the extent is a multiple of the packet's lanes; an extent that is not
    # (no packet can form) has the search space without the seeds.
    with cpu_render_target():
        bounds = [
            _bind(_scatter_component, _scatter_args(elements))
            for elements in (1 << 20, 3 << 19, 1 << 21, (1 << 20) + 2)
        ]
        fingerprints = {
            repr(bound.config_spec.structural_fingerprint()) for bound in bounds
        }
        seeds = [_component_seeds(bound) for bound in bounds]
    assert len(fingerprints) == 1
    assert seeds[0] == seeds[1] == seeds[2] and len(seeds[0]) == 2
    assert seeds[3] == []


def test_seeds_render_their_packet_on_each_dtype_and_device() -> None:
    # The seed's vector width follows the store's dtype and the device's
    # widest store: the first seed of each kind renders one flush of every
    # component and no scalar store
    # (``test_seeds_give_each_thread_one_widest_packet`` renders fp32 pairs
    # on sm_100).
    for args, capability in (
        (_scatter_args(), (9, 0)),
        (_scatter_args(1 << 20, 4, torch.bfloat16), (10, 0)),
        (_scatter_args(1 << 20, 2, torch.float16), (10, 0)),
    ):
        with cpu_render_target():
            bound = _bind(_scatter_component, args, capability)
            seeds = _component_seeds(bound)
            assert len(seeds) == 2
            code = bound.to_code(
                bound.config_spec.normalized_config(helion.Config(**seeds[0]))
            )
        assert len(re.findall(r"_cute_store_u(?:16|32)_vec\(out", code)) == 1
        assert ".store(" not in code


def test_a_lane_body_that_reads_its_output_gets_the_layout_with_scalar_stores() -> None:
    # The seeds follow the store's own facts.  What the lowering proves on
    # the lane body is not known to the heuristic: an in-place update reads
    # the tensor in the lane loop, so its packet is refused
    # (``component_group_refusal``) and the seeds are the same thread layout
    # with scalar stores.
    with cpu_render_target():
        bound = _bind(_scale_in_place, (torch.empty(1 << 20, 2),))
        seeds = _component_seeds(bound)
        assert [seed["num_threads"] for seed in seeds] == [[128, 1], [256, 1]]
        assert [seed["cute_vector_widths"] for seed in seeds] == [[4, 1]] * 2
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    assert not re.search(r"_cute_store_u(?:16|32)_vec\(", code)
    assert code.count(".store(") == 1


def test_a_kernel_of_several_root_grids_keeps_its_multi_root_seeds() -> None:
    # A component-packet seed sizes the store's tile alone: the pointwise
    # root would stay at its default blocks, where every multi-root seed
    # gives it a thread layout.  Such a kernel has its multi-root seeds only.
    with cpu_render_target():
        bound = _bind(
            _scale_pairs_and_vector, (torch.empty(1 << 20, 2), torch.empty(65536))
        )
        spec = bound.config_spec
        assert spec.kernel_grid_fact is not None
        assert len(spec.kernel_grid_fact.roots) == 2
        seeds = [dict(seed.config) for seed in spec.compiler_seed_configs]
    assert seeds
    assert _component_seeds(bound) == []
    assert all(seed["num_threads"][-1] > 0 for seed in seeds)


def test_a_two_slot_window_gets_a_kernel_from_each_seed_family() -> None:
    # ``out[n, k] = flat[n + channels * (k - pad)]`` over two slots is a
    # gather stored along another tile and a store of an element's two
    # components at once.  The register tiles of the first
    # (``_lane_gather_seed_configs``, with ``cute_vloop_sink`` and
    # ``cute_staged_store``) and the component packets tile it alike at 128
    # threads, and are two kernels: the sink keeps the slots for its packet
    # loads and the staged image, and without it the slots are one packet
    # store fed by element loads.  Neither seed carries a knob that does
    # nothing.
    args = (torch.empty(3, 40, 16), 2, 1)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _bind(sliding_windows, args)
        tiles = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_staged_store")
        ]
        packets = _component_seeds(bound)
        layouts = [
            [(seed["block_sizes"], seed["num_threads"]) for seed in family]
            for family in (tiles, packets)
        ]
        staged, packet = (
            bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
            for seed in (tiles[1], packets[0])
        )
    assert layouts == [
        [([256, 2], [64, 1]), ([512, 2], [128, 1])],
        [([512, 2], [128, 1]), ([1024, 2], [256, 1])],
    ]
    assert staged.count("_cute_load_v16b_if(flat.iterator") == 2
    assert "_cta_stage" in staged and "_tile_store_vals" not in staged
    assert len(re.findall(r"_cute_store_u32_vec\(out", packet)) == 1
    assert "_cute_load_v16b_if(" not in packet and "_cta_stage" not in packet


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seed_matches_select_scatter() -> None:
    # 5000 pairs in tiles of 512: a partial last tile.
    pair = torch.randn(5000, 2, device=DEVICE)
    grad = torch.randn(5000, device=DEVICE)
    for component in (0, 1):
        bound = _scatter_component.bind((pair, grad, component))
        seed = _component_seeds(bound)[0]
        config = bound.config_spec.normalized_config(helion.Config(**seed))
        assert "_cute_store_u32_vec(out.iterator" in bound.to_code(config)
        out = bound.compile_config(config)(pair, grad, component)
        expected = torch.select_scatter(pair, grad, 1, component)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
