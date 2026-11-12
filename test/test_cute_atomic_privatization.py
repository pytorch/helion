"""Per-CTA shared-memory privatization of small atomic-add targets
(``cute_privatize_atomics``, ``helion/_compiler/cute/atomic_privatization.py``)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute import atomic_privatization
from helion._compiler.cute.atomic_pointer import cute_coord_pointer_expr
from helion._compiler.cute.atomic_privatization import _atomic_site
from helion._compiler.cute.atomic_privatization import _atomic_sites
from helion._compiler.cute.atomic_privatization import _point_sites_at
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_grid_lane_jam_kernels import bucket_grad_scatter
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bucket_sums(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out[buckets[r, c], c] += values[r, c]``: a bucketed column sum, the
    shape of T5's relative-position-bias gradient."""
    rows, cols = values.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        hl.atomic_add(
            out,
            [buckets[tile_r, tile_c], tile_c.index[None, :]],
            values[tile_r, tile_c],
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _heads_inner_scatter(
    grads: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``bucket_grad_scatter`` with the heads innermost:
    ``out[buckets[k, q], h] += grads[:, k, q, h].sum(0)``."""
    _, k_len, q_len, heads = grads.shape
    for tile_k, tile_q, tile_h in hl.tile([k_len, q_len, heads]):
        total = torch.sum(grads[:, tile_k, tile_q, tile_h].to(torch.float32), dim=0)
        bucket = buckets[tile_k, tile_q].unsqueeze(2)
        hl.atomic_add(out, [bucket, tile_h.index[None, None, :]], total)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sum_scatter(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out[buckets[r]] += values[r, :].sum()``: only the index tensor is
    loaded along the tile."""
    for tile_r in hl.tile(values.size(0)):
        total = values[tile_r, :].sum(-1)
        hl.atomic_add(out, [buckets[tile_r]], total)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bucket_sums_then_read(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """The same adds, plus a read of the target: privatization would hide the
    adds of other CTAs from it."""
    rows, cols = values.shape
    seen = torch.empty_like(values)
    for tile_r, tile_c in hl.tile([rows, cols]):
        hl.atomic_add(
            out,
            [buckets[tile_r, tile_c], tile_c.index[None, :]],
            values[tile_r, tile_c],
        )
        seen[tile_r, tile_c] = out[buckets[tile_r, tile_c], tile_c.index[None, :]]
    return seen


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bucket_sums_acq_rel(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    rows, cols = values.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        hl.atomic_add(
            out,
            [buckets[tile_r, tile_c], tile_c.index[None, :]],
            values[tile_r, tile_c],
            sem="acq_rel",
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _histogram_then_release(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor, done: torch.Tensor
) -> torch.Tensor:
    """Relaxed adds, then a release on another tensor that publishes them to
    an observer acquiring it (another kernel, a peer rank)."""
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(out, [buckets[tile]], values[tile])
        hl.atomic_add(done, [0], 1, sem="release")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _histogram_then_fence(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Relaxed adds, then asm not declared pure: it may hold a fence."""
    fenced = torch.empty_like(values)
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(out, [buckets[tile]], values[tile])
        fenced[tile] = hl.inline_asm_elementwise(
            "fence.acq_rel.gpu; mov.b32 $0, $1;",
            "=r,r",
            [values[tile]],
            dtype=torch.float32,
            is_pure=False,
            pack=1,
        )
    return fenced


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _histogram(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(out, [buckets[tile]], values[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _allocated_histogram(values: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
    # The target is allocated here, so no other tensor can be its storage.
    out = torch.zeros(32, dtype=values.dtype, device=values.device)
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(out, [buckets[tile]], values[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count(buckets: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(buckets.size(0)):
        hl.atomic_add(counts, [buckets[tile]], 1)
    return counts


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _count_tiles(values: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    for _tile in hl.tile(values.size(0)):
        hl.atomic_add(total, [], 1.0)
    return total


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_histograms(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor, wide: torch.Tensor
) -> torch.Tensor:
    """Two sibling loops, eight elements into ``out`` and all into ``wide``."""
    for tile in hl.tile(8):
        hl.atomic_add(out, [buckets[tile]], values[tile])
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(wide, [buckets[tile]], values[tile])
    return wide


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sibling_inner_histograms(
    short: torch.Tensor,
    long: torch.Tensor,
    buckets: torch.Tensor,
    out: torch.Tensor,
    wide: torch.Tensor,
) -> torch.Tensor:
    """Two inner loops under one grid tile, the shorter one first."""
    for tile_g in hl.tile(short.size(0)):
        for tile_s in hl.tile(short.size(1)):
            hl.atomic_add(out, [buckets[tile_g, tile_s]], short[tile_g, tile_s])
        for tile_n in hl.tile(long.size(1)):
            hl.atomic_add(wide, [buckets[tile_g, tile_n]], long[tile_g, tile_n])
    return wide


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_dtype_adds(
    floats: torch.Tensor, buckets: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """An int32 and an fp32 view of one buffer, both added to."""
    words = floats.view(torch.int32)
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(words, [buckets[tile]], 1065353216)
        hl.atomic_add(floats, [buckets[tile]], values[tile])
    return floats


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add_then_read(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Adds into ``out``, reads ``x``: one tensor when a caller passes it
    twice, and each thread then reads back its own update."""
    seen = torch.empty_like(x)
    for tile in hl.tile(out.size(0)):
        hl.atomic_add(out, [tile], 1.0)
        seen[tile] = x[tile]
    return seen


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add_then_read_through_pointers(
    out: torch.Tensor, example: torch.Tensor, dev_ptrs: torch.Tensor
) -> torch.Tensor:
    """Adds into ``out``, reads through a stack tensor's pointers, which a
    caller may aim at ``out``: no storage of the kernel bounds them."""
    seen = torch.empty_like(out)
    for tile in hl.tile(out.size(0)):
        hl.atomic_add(out, [tile], 1.0)
        rows = hl.stacktensor_like(example, dev_ptrs[:])
        seen[tile] = rows[tile].sum(0)
    return seen


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _histogram_after_barrier(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    scaled = torch.empty_like(values)
    for tile in hl.tile(values.size(0)):
        scaled[tile] = values[tile] * 2
    hl.barrier()
    for tile in hl.tile(values.size(0)):
        hl.atomic_add(out, [buckets[tile]], scaled[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _split_k_matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        hl.atomic_add(
            out, [tile_m, tile_n], torch.matmul(a[tile_m, tile_k], b[tile_k, tile_n])
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_bucket_sums(
    values: torch.Tensor, rows: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out[rows[r], c] += values[r, c]``, contiguous along ``c``: vector
    atomics without the knob."""
    n, cols = values.shape
    for tile_r, tile_c in hl.tile([n, cols]):
        hl.atomic_add(out, [rows[tile_r], tile_c], values[tile_r, tile_c])
    return out


def _args(
    buckets: int = 32, cols: int = 8, rows: int = 4096, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.randn(rows, cols, device=device)
    indices = torch.randint(0, buckets, (rows, cols), device=device)
    return values, indices, torch.zeros(buckets, cols, device=device)


def _histogram_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(4096), torch.randint(0, 32, (4096,)), torch.zeros(32)


# One tensor for both arguments of ``_add_then_read``.
_SHARED = torch.zeros(64)

_PRIVATE = {"block_sizes": [256, 8], "num_threads": [32, 8]}


def _codes(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], *configs: dict[str, object]
) -> list[str]:
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        return [
            bound.to_code(bound.config_spec.normalized_config(helion.Config(**config)))
            for config in configs
        ]


@pytest.mark.parametrize(
    ("kernel", "args", "block_sizes", "num_threads"),
    [
        # Eight columns take eight threads; the other 16 of the 128 go to
        # the rows, the next axis outward.
        (_bucket_sums, _args(), [256, 8], [16, 8]),
        # Sibling loops each fill their own tile: the short one to its extent.
        (
            _two_histograms,
            (*_histogram_args(), torch.zeros(32)),
            [8, 256],
            [8, 128],
        ),
        # Sibling inner loops share the grid tile's 16 threads; the second
        # loop gets what the first left of the budget.
        (
            _sibling_inner_histograms,
            (
                torch.randn(256, 8),
                torch.randn(256, 1024),
                torch.randint(0, 32, (256, 1024)),
                torch.zeros(32),
                torch.zeros(32),
            ),
            [256, 8, 8],
            [16, 8, 8],
        ),
        # The batch sum runs serially in each of the tile's 128 threads.
        (
            helion.kernel(
                bucket_grad_scatter,
                backend="cute",
                static_shapes=True,
                autotune_effort="none",
            ),
            (
                torch.randn(4, 2, 64, 64),
                torch.randint(0, 32, (64, 64)),
                torch.zeros(32, 2),
            ),
            [1, 32, 64],
            [1, 2, 64, 1],
        ),
    ],
)
def test_the_seed_fills_each_loop_nest_from_the_inside(
    kernel: helion.Kernel,
    args: tuple[torch.Tensor, ...],
    block_sizes: list[int],
    num_threads: list[int],
) -> None:
    with cpu_render_target():
        spec = cpu_bind(kernel, args).config_spec
    assert spec.cute_privatize_atomics_enabled
    (seed,) = [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_privatize_atomics") is True
        and "block_sizes" in seed.config
        and not seed.config.get("cute_grid_lane_jam")
    ]
    assert seed["block_sizes"] == block_sizes
    assert seed["num_threads"] == num_threads


def _scatter_bound(batch: int) -> BoundKernel:
    """``bucket_grad_scatter`` bound on CPU, in the caller's
    ``cpu_render_target``, to ``batch`` x 2 x 8 x 256 bf16 gradients."""
    kernel = helion.kernel(bucket_grad_scatter, backend="cute", static_shapes=True)
    args = (
        torch.empty(batch, 2, 8, 256, dtype=torch.bfloat16),
        torch.empty(8, 256, dtype=torch.int64),
        torch.zeros(32, 2),
    )
    return cpu_bind(kernel, args)


def _scatter_jam_seed(bound: BoundKernel) -> dict[str, object]:
    (seed,) = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_grid_lane_jam") is True
    ]
    return seed


def test_the_jam_seed_puts_a_packet_of_keys_on_each_thread() -> None:
    # The keys, the gradients' stride-1 dimension, as 32 threads of eight
    # (one 16-byte packet of bf16), the batch sum a rolled loop of eight
    # around them, consecutive CTAs on consecutive query rows.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _scatter_bound(16)
        seed = _scatter_jam_seed(bound)
        code = bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
    assert seed["cute_privatize_atomics"] is True
    assert seed["block_sizes"] == [1, 1, 256]
    assert seed["num_threads"] == [1, 1, 32, 1]
    assert seed["reduction_loops"] == [8]
    assert set(seed["cute_vector_widths"]) == {1}
    assert seed["loop_orders"] == [[2, 1, 0]]
    assert "for lane_2 in range(8)" not in code
    assert "_jam_cell" in code


def test_a_batch_too_short_to_roll_keeps_the_seed_and_its_lane_loop() -> None:
    # Four rows of the batch sum stay a persistent reduction, with no device
    # loop to interleave the lanes into; the seed stays, so the knob is
    # searchable at every shape (one search space for autotune_multi).
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bounds = [_scatter_bound(batch) for batch in (4, 16)]
        seed = _scatter_jam_seed(bounds[0])
        code = bounds[0].to_code(
            bounds[0].config_spec.normalized_config(helion.Config(**seed))
        )
        fingerprints = [bound.config_spec.structural_fingerprint() for bound in bounds]
    assert "for lane_2 in range(8)" in code
    assert fingerprints[0] == fingerprints[1]
    (jam_entry,) = [
        entry for entry in fingerprints[0] if entry[0] == "cute_grid_lane_jam"
    ]
    assert jam_entry[jam_entry.index("search") + 1 :] == ("False", "True")


def test_a_short_inner_axis_gets_a_warp_from_the_axes_outward() -> None:
    # Eight heads innermost are one 16-byte packet of bf16, one thread; the
    # query axis gives the CTA its warp, a thread per query.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(
            _heads_inner_scatter,
            (
                torch.empty(16, 64, 64, 8, dtype=torch.bfloat16),
                torch.empty(64, 64, dtype=torch.int64),
                torch.zeros(32, 8),
            ),
        )
        seed = _scatter_jam_seed(bound)
        code = bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
    assert seed["block_sizes"] == [1, 32, 8]
    assert seed["num_threads"][:3] == [1, 32, 1]
    assert "_jam_cell" in code


def test_an_index_tensor_along_the_tile_gets_no_jam_seed() -> None:
    # The rows are loaded along their reduction, 512 elements apart on the
    # tile; the int64 buckets, the only tensor stride-1 along it, are the
    # atomic's index, not what it adds.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        spec = cpu_bind(
            _row_sum_scatter,
            (
                torch.empty(4096, 512, dtype=torch.bfloat16),
                torch.empty(4096, dtype=torch.int64),
                torch.zeros(32),
            ),
        ).config_spec
    assert spec.cute_privatize_atomics_enabled
    assert not [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_grid_lane_jam")
    ]


def test_adds_go_to_a_shared_copy_flushed_once_per_cta() -> None:
    (code,) = _codes(
        _bucket_sums, _args(), {**_PRIVATE, "cute_privatize_atomics": True}
    )
    module = ast.parse(code)
    atomics = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.atomic_add"
    ]
    shared = [
        call
        for call in atomics
        if ast.unparse(call.args[0]).startswith("(_private_out.iterator +")
    ]
    flushes = [
        call
        for call in atomics
        if ast.unparse(call.args[0]).startswith("(out.iterator +")
    ]
    assert len(shared) == 1 and len(flushes) == 1, code
    assert any(
        keyword.arg == "scope" and ast.literal_eval(keyword.value) == "cta"
        for keyword in shared[0].keywords
    )
    # The flush passes its value positionally (DSL versions name it ``val``
    # or ``value``).
    assert len(flushes[0].args) == 2
    # The copy starts at -0.0, the identity of the add, behind one barrier;
    # after another, a cell is flushed only when its bits changed.
    assert code.count("cute.arch.alloc_smem(cutlass.Float32, 256)") == 1
    assert code.count("cute.arch.sync_threads()") == 2
    assert code.index("= cutlass.Float32(-0.0)") < code.index("_private_out.iterator +")
    assert ".bitcast(cutlass.Uint32) != cutlass.Uint32(2147483648)" in code


def test_the_knob_off_keeps_the_global_atomics() -> None:
    (code,) = _codes(
        _bucket_sums, _args(), {**_PRIVATE, "cute_privatize_atomics": False}
    )
    assert "_private_out" not in code
    assert "alloc_smem" not in code


def test_warp_aggregation_leaves_the_shared_copy_scalar_atomics() -> None:
    """Both knobs apply to an int32 count; the privatized target's adds stay
    the scalar atomics the rewrite redirects to its shared copy."""
    (code,) = _codes(
        _count,
        (torch.randint(0, 64, (8192,)), torch.zeros(64, dtype=torch.int32)),
        {
            "block_sizes": [256],
            "cute_warp_aggregated_atomics": True,
            "cute_privatize_atomics": True,
        },
    )
    assert "_private_counts" in code, code
    assert "_cute_atomic_warp_fetch_add_int32" not in code, code


def test_vector_atomic_sites_stay_scalar_for_the_shared_copy() -> None:
    args = (torch.randn(4096, 64), torch.randint(0, 16, (4096,)), torch.zeros(16, 64))
    config = {
        "block_sizes": [16, 64],
        "num_threads": [16, 16],
        "cute_vector_widths": [1, 4],
    }
    off, on = _codes(
        _row_bucket_sums,
        args,
        {**config, "cute_privatize_atomics": False},
        {**config, "cute_privatize_atomics": True},
    )
    assert "_cute_red_add_f32_vec" in off and "_private_out" not in off
    assert "_cute_red_add" not in on and "_private_out" in on


def test_distinct_arguments_offer_the_knob() -> None:
    with cpu_render_target():
        spec = cpu_bind(_add_then_read, (torch.zeros(64), torch.zeros(64))).config_spec
    assert spec.cute_privatize_atomics_enabled


def test_a_target_the_kernel_allocates_is_privatized() -> None:
    # A storage no argument holds is disjoint from every other tensor without
    # the cache-keyed fact the arguments need.
    args = (torch.randn(4096), torch.randint(0, 32, (4096,)))
    with cpu_render_target():
        assert cpu_bind(
            _allocated_histogram, args
        ).config_spec.cute_privatize_atomics_enabled
    (code,) = _codes(
        _allocated_histogram,
        args,
        {"block_sizes": [256], "cute_privatize_atomics": True},
    )
    assert "_private_out" in code, code
    assert code.count("cute.arch.alloc_smem(cutlass.Float32, 32)") == 1, code


def test_an_explicit_false_drops_where_warp_aggregation_is_not_offered() -> None:
    # ``cute_warp_aggregated_atomics`` normalizes like
    # ``cute_privatize_atomics``: where the kernel offers it no relaxed Int32
    # tickets, an explicit False (the knob-off code a tuned config spells
    # out) is dropped and True is rejected.
    with cpu_render_target():
        spec = cpu_bind(_histogram, _histogram_args()).config_spec
    assert not spec.cute_warp_aggregated_atomics_enabled
    default = spec.default_config().config
    config = spec.normalized_config(
        helion.Config(**default, cute_warp_aggregated_atomics=False)
    )
    assert "cute_warp_aggregated_atomics" not in config.config
    with pytest.raises(
        helion.exc.InvalidConfig, match="cute_warp_aggregated_atomics requires"
    ):
        spec.normalized_config(
            helion.Config(**default, cute_warp_aggregated_atomics=True)
        )


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        # Read back in the kernel.
        (_bucket_sums_then_read, _args()),
        # Ordered atomics must stay global.
        (_bucket_sums_acq_rel, _args()),
        # A release on another tensor orders the adds before it.
        (_histogram_then_release, (*_histogram_args(), torch.zeros(1))),
        # So may a fence in asm not declared pure.
        (_histogram_then_fence, _histogram_args()),
        # Past PRIVATE_MAX_BYTES (16 KiB of copies).
        (_bucket_sums, _args(buckets=1024)),
        # The target also passed as the tensor the kernel reads.
        (_add_then_read, (_SHARED, _SHARED)),
        # A stack tensor's pointers may name the target.
        (
            _add_then_read_through_pointers,
            (torch.zeros(1024), torch.zeros(1024), torch.zeros(1, dtype=torch.uint64)),
        ),
        # A grid barrier between hl.barrier() phases.
        (_histogram_after_barrier, _histogram_args()),
        # A 0-d target.
        (_count_tiles, (torch.randn(4096), torch.zeros(()))),
        # Adds of two dtypes to one buffer do not commute.
        (
            _mixed_dtype_adds,
            (torch.zeros(8), torch.randint(0, 8, (4096,)), torch.randn(4096)),
        ),
        # A matmul, whose families generate their own kernel bodies.
        (
            _split_k_matmul,
            (torch.randn(32, 512), torch.randn(512, 32), torch.zeros(32, 32)),
        ),
    ],
)
def test_ineligible_targets_get_no_knob(kernel: helion.Kernel, args: tuple) -> None:
    with cpu_render_target():
        spec = cpu_bind(kernel, args).config_spec
    assert not spec.cute_privatize_atomics_enabled
    with pytest.raises(helion.exc.InvalidConfig, match="cute_privatize_atomics"):
        spec.normalize(spec.default_config().config | {"cute_privatize_atomics": True})
    # ``False``, which every tuned config spells out, is the knob-off code:
    # a tuned config stays valid where the kernel no longer offers the knob.
    tuned = spec.default_config().config | {"cute_privatize_atomics": False}
    spec.normalize(tuned)
    assert "cute_privatize_atomics" not in tuned


@pytest.mark.parametrize("index", [["i"], ["i", "j"]])
def test_the_rewrite_matches_the_lowering_pointer(index: list[str]) -> None:
    """The rewrite matches the pointer the atomic lowering spells
    (``atomic_pointer.cute_coord_pointer_expr``), which its flush emits too."""
    pointer = cute_coord_pointer_expr("out", index)
    (statement,) = ast.parse(
        f"cute.arch.atomic_add({pointer}, val=v, sem='relaxed')"
    ).body
    assert isinstance(statement, ast.Expr)
    site = _atomic_site(statement, "out")
    assert site is not None
    assert [reference.id for reference in site[1]] == ["out", "out"]


def test_other_references_to_the_target_are_not_redirected() -> None:
    site = (
        "cute.arch.atomic_add((out.iterator + cute.crd2idx((i,), out.layout))"
        ".llvm_ptr, val=v, sem='relaxed')"
    )
    body = ast.parse(f"if c:\n    {site}\nn = cute.size(out.layout)").body
    assert _atomic_sites(body, "out") is None
    # A reference in the coordinate is not the pointer's: the copy's layout
    # has other strides.
    strided = site.replace("((i,), out.layout)", "((i * out.stride[0],), out.layout)")
    assert _atomic_sites(ast.parse(strided).body, "out") is None
    body = ast.parse(f"if c:\n    {site}").body
    sites = _atomic_sites(body, "out")
    assert sites is not None
    _point_sites_at(sites, "_private_out")
    assert ast.unparse(body[0]).endswith(
        "cute.arch.atomic_add((_private_out.iterator + cute.crd2idx((i,), "
        "_private_out.layout)).llvm_ptr, val=v, sem='relaxed', scope='cta')"
    )


_PREAMBLE_ATOMIC = (
    f"cute.arch.atomic_add({cute_coord_pointer_expr('out', ['0', '0'])}, "
    "val=cutlass.Float32(1.0), sem='relaxed')"
)


@pytest.mark.parametrize(
    ("where", "source", "reason"),
    [
        ("preamble", "_cells = cute.size(out.layout)", "is reached other than"),
        # The copy is filled at the top of the body: an atomic before it
        # would add into a copy that does not exist yet.
        ("preamble", _PREAMBLE_ATOMIC, "is reached other than"),
        # The flush goes at the end of the body.
        (
            "body",
            "if cute.arch.thread_idx()[0] > 1024:\n    return",
            "returns before its flush",
        ),
    ],
)
def test_a_kernel_the_rewrite_cannot_redirect_rejects_the_config(
    where: str, source: str, reason: str
) -> None:
    """The lowering kept the target's atomics scalar for the rewrite, so a
    kernel that defeats it (a reference to the target in the preamble, an
    atomic into it there, or a return before the flush) is an invalid config,
    not the knob-off code without vector atomics."""
    rewrite = atomic_privatization.privatize_atomic_targets

    def with_statement(codegen: object, host: object) -> None:
        fn = codegen.device_function  # pyrefly: ignore [missing-attribute]
        getattr(fn, where).insert(0, statement_from_string(source))
        rewrite(codegen, host)  # pyrefly: ignore [bad-argument-type]

    with (
        patch.object(atomic_privatization, "privatize_atomic_targets", with_statement),
        pytest.raises(helion.exc.InvalidConfig, match=reason),
    ):
        _codes(_bucket_sums, _args(), {**_PRIVATE, "cute_privatize_atomics": True})


def test_a_target_past_the_shared_budget_keeps_its_global_atomics() -> None:
    """The targets' copies share PRIVATE_MAX_BYTES: after ``out``'s 12 KiB,
    ``wide``'s 8 KiB no longer fit, so ``out`` alone is privatized."""
    args = (
        torch.randn(4096),
        torch.randint(0, 32, (4096,)),
        torch.zeros(3072),
        torch.zeros(2048),
    )
    (code,) = _codes(
        _two_histograms,
        args,
        {"block_sizes": [8, 256], "cute_privatize_atomics": True},
    )
    assert "_private_out" in code, code
    assert "_private_wide" not in code, code


def test_the_targets_are_found_once_per_codegen() -> None:
    """Every vector atomic site and the rewrite read one cached decision."""
    args = (torch.randn(4096, 64), torch.randint(0, 16, (4096,)), torch.zeros(16, 64))
    config = helion.Config(
        block_sizes=[16, 64],
        num_threads=[16, 16],
        cute_vector_widths=[1, 4],
        cute_privatize_atomics=True,
    )
    with cpu_render_target():
        bound = cpu_bind(_row_bucket_sums, args)
        config = bound.config_spec.normalized_config(config)
        find = atomic_privatization.privatizable_atomic_targets
        with patch.object(
            atomic_privatization, "privatizable_atomic_targets", side_effect=find
        ) as found:
            code = bound.to_code(config)
    assert "_private_out" in code
    assert found.call_count == 1


def _bucket_sums_reference(
    values: torch.Tensor, buckets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    columns = torch.arange(values.size(1), device=values.device)
    return out.double().index_put_(
        (buckets.reshape(-1), columns.repeat(values.size(0))),
        values.double().reshape(-1),
        accumulate=True,
    )


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteAtomicPrivatization(TestCase):
    def test_privatized_sums_match_eager(self) -> None:
        values, buckets, out = _args(device=DEVICE)
        expected = _bucket_sums_reference(values, buckets, out)
        code, result = code_and_output(
            _bucket_sums,
            (values, buckets, out),
            **_PRIVATE,
            cute_privatize_atomics=True,
        )
        self.assertIn("_private_out", code)
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=1e-4)

    def test_persistent_ctas_flush_after_their_last_tile(self) -> None:
        # 1024 tiles over at most one CTA per SM: every CTA adds several tiles
        # into its copy before the flush.
        values, buckets, out = _args(rows=65536, device=DEVICE)
        expected = _bucket_sums_reference(values, buckets, out)
        code, result = code_and_output(
            _bucket_sums,
            (values, buckets, out),
            block_sizes=[64, 8],
            num_threads=[8, 8],
            pid_type="persistent_interleaved",
            cute_privatize_atomics=True,
        )
        self.assertIn("_private_out", code)
        self.assertIn("virtual_pid", code)
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=1e-3)

    def test_cells_left_unchanged_keep_their_bits(self) -> None:
        # From -0.0, the global atomics turn the cells that get +0.0 adds
        # (0-7) into +0.0 and leave the ones that get -0.0 adds (8-15) or none
        # (16-31) at -0.0; the flush must skip exactly the last two groups.
        buckets = torch.arange(4096, device=DEVICE) % 16
        values = torch.where(buckets < 8, 0.0, -0.0)
        out = torch.full((32,), -0.0, device=DEVICE)
        expected = torch.full((32,), -0.0)
        expected[:8] = 0.0
        code, result = code_and_output(
            _histogram,
            (values, buckets, out),
            block_sizes=[256],
            cute_privatize_atomics=True,
        )
        self.assertIn("_private_out", code)
        self.assertEqual(
            result.cpu().view(torch.int32).tolist(), expected.view(torch.int32).tolist()
        )

    def test_privatized_sums_into_an_allocated_target_match_eager(self) -> None:
        values = torch.randn(65536, device=DEVICE)
        buckets = torch.randint(0, 32, (65536,), device=DEVICE)
        code, result = code_and_output(
            _allocated_histogram,
            (values, buckets),
            block_sizes=[256],
            cute_privatize_atomics=True,
        )
        self.assertIn("_private_out", code)
        expected = torch.zeros(32, dtype=torch.float64, device=DEVICE)
        expected.index_add_(0, buckets, values.double())
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=1e-3)

    def test_privatized_int32_counts_are_exact(self) -> None:
        buckets = torch.randint(0, 64, (8192,), device=DEVICE)
        counts = torch.zeros(64, dtype=torch.int32, device=DEVICE)
        code, result = code_and_output(
            _count, (buckets, counts), block_sizes=[256], cute_privatize_atomics=True
        )
        self.assertIn("_private_counts", code)
        self.assertEqual(
            result.tolist(), torch.bincount(buckets, minlength=64).tolist()
        )

    def _assert_the_jam_seed_matches(
        self,
        kernel: helion.Kernel,
        grads: torch.Tensor,
        buckets: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        """Run the jam seed ``kernel`` plants with the jam on and off: integer
        gradients sum exactly in any order, so combining the lanes' adds per
        cell must give ``expected`` bit for bit either way."""
        out = torch.zeros(expected.shape, device=DEVICE)
        (seed,) = [
            seed.config
            for seed in kernel.bind(
                (grads, buckets, out)
            ).config_spec.compiler_seed_configs
            if seed.config.get("cute_grid_lane_jam") is True
        ]
        for jam in (False, True):
            out = torch.zeros(expected.shape, device=DEVICE)
            code, result = code_and_output(
                kernel, (grads, buckets, out), **{**seed, "cute_grid_lane_jam": jam}
            )
            self.assertEqual("_jam_cell" in code, jam)
            self.assertTrue(torch.equal(result, expected.float()))

    def test_the_jam_seed_with_the_keys_innermost_matches_eager(self) -> None:
        # A 16-byte packet of keys per thread; four buckets give each
        # thread's lanes cells in common.
        torch.manual_seed(0)
        grads = torch.randint(-4, 5, (16, 2, 8, 256), device=DEVICE).to(torch.bfloat16)
        buckets = torch.randint(0, 4, (8, 256), device=DEVICE) * 7
        total = grads.double().sum(0)
        expected = torch.zeros(32, 2, device=DEVICE, dtype=torch.float64)
        for head in range(2):
            expected[:, head].index_add_(0, buckets.flatten(), total[head].flatten())
        self._assert_the_jam_seed_matches(
            helion.kernel(bucket_grad_scatter, backend="cute", static_shapes=True),
            grads,
            buckets,
            expected,
        )

    def test_the_jam_seed_with_the_heads_innermost_matches_eager(self) -> None:
        # The eight heads are one thread's packet, beside a thread per query.
        torch.manual_seed(0)
        grads = torch.randint(-4, 5, (16, 64, 64, 8), device=DEVICE).to(torch.bfloat16)
        buckets = torch.randint(0, 4, (64, 64), device=DEVICE) * 7
        expected = torch.zeros(32, 8, device=DEVICE, dtype=torch.float64)
        expected.index_add_(0, buckets.flatten(), grads.double().sum(0).flatten(0, 1))
        self._assert_the_jam_seed_matches(
            _heads_inner_scatter, grads, buckets, expected
        )
