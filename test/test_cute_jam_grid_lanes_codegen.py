"""CPU codegen tests: ``cute_grid_lane_jam`` interleaves a grid lane loop's lanes.

A grid tile axis without threads runs its elements as a lane loop around
the whole tile body, device loops included.  With the knob,
``cute/jam_grid_lanes.py`` unrolls that loop and jams the lanes' copies of
each statement side by side: the lanes' loads of consecutive elements of a
stride-1 dimension become one vector load (T5's heads-innermost bias: the
eight heads of a key in 16 bytes) and the lanes' shared-memory reduces one
fragment reduce.  A lane loop whose lanes are not provably independent keeps
the knob-off code.
"""

from __future__ import annotations

import ast
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteGridLaneJamHeuristic
from helion._compiler.cute.autotune.heuristics.cute import CuteReductionLaneJamHeuristic
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_grid_lane_jam_kernels import HEAD_ROWS_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import ROW_NORM_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import SCALAR_ROWS_CONFIG
from helion._testing.cute_grid_lane_jam_kernels import bias_softmax
from helion._testing.cute_grid_lane_jam_kernels import doubled_rows_plus_one
from helion._testing.cute_grid_lane_jam_kernels import head_bias
from helion._testing.cute_grid_lane_jam_kernels import head_bias_softmax_bwd
from helion._testing.cute_grid_lane_jam_kernels import heads_config
from helion._testing.cute_grid_lane_jam_kernels import hinted_head_rows
from helion._testing.cute_grid_lane_jam_kernels import invariant_rows_over_heads
from helion._testing.cute_grid_lane_jam_kernels import kept_row_norm
from helion._testing.cute_grid_lane_jam_kernels import leading_batch_bias_exp
from helion._testing.cute_grid_lane_jam_kernels import nested_bias_exp
from helion._testing.cute_grid_lane_jam_kernels import row_norm
from helion._testing.cute_grid_lane_jam_kernels import rows_into_two
from helion._testing.cute_grid_lane_jam_kernels import scale_rows_in_place
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

_BIAS_PACKET = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"


def _bind(
    keys: int,
    *,
    heads: int = 8,
    queries: int = 4,
    heads_innermost: bool = True,
    static_shapes: bool = True,
    bias_dtype: torch.dtype = torch.bfloat16,
) -> helion.runtime.kernel.BoundKernel:
    """``bias_softmax`` bound on CPU, in the caller's ``cpu_render_target``,
    to ``[2, heads, queries, keys]`` bf16 scores and a ``bias_dtype`` bias
    laid out heads-innermost (keys-innermost when ``heads_innermost`` is
    false)."""
    kernel = helion.kernel(bias_softmax, backend="cute", static_shapes=static_shapes)
    scores = torch.empty(2, heads, queries, keys, dtype=torch.bfloat16)
    bias = (
        torch.empty(2, queries, keys, heads, dtype=bias_dtype).permute(0, 3, 1, 2)
        if heads_innermost
        else torch.empty(2, heads, queries, keys, dtype=bias_dtype)
    )
    return cpu_bind(kernel, (scores, bias))


def _bias_softmax_seeds(
    keys: int,
    heads_innermost: bool = True,
    *,
    static_shapes: bool = True,
    bias_dtype: torch.dtype = torch.bfloat16,
) -> list[helion.Config]:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _bind(
            keys,
            heads_innermost=heads_innermost,
            static_shapes=static_shapes,
            bias_dtype=bias_dtype,
        )
        return list(bound.config_spec.compiler_seed_configs)


def _bias_softmax_code(keys: int = 256, **knobs: object) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _bind(keys)
        return bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**heads_config(2, keys), **knobs)
            )
        )


def test_the_key_loops_take_packets_of_the_scores() -> None:
    # The lanes are one 16-byte packet of the fp32 bias (four heads); the
    # bias's loads along the keys gather at a heads stride either way, so the
    # keys' vector width is a packet of the bf16 scores they walk (eight).
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        seed = _jam_seed_of(_bind(512, bias_dtype=torch.float32))
    assert seed["block_sizes"] == [1, 4, 1, 512, 512]
    assert seed["num_threads"] == [1, 1, 1, 64, 64]
    assert seed["cute_vector_widths"] == [1, 1, 1, 8, 8]


def _jam_seed_of(bound: helion.runtime.kernel.BoundKernel) -> dict[str, object]:
    """The config of ``bound``'s one ``cute_grid_lane_jam`` seed."""
    (seed,) = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_grid_lane_jam")
    ]
    return seed


def _jam_plan(bound: helion.runtime.kernel.BoundKernel) -> object:
    """``CuteGridLaneJamHeuristic._plan`` for ``bound``: asked directly,
    since ``compiler_seed_configs`` would hide a crash as no seed."""
    return CuteGridLaneJamHeuristic._plan(bound.env, bound.host_function.device_ir)


def test_one_lane_still_seeds_and_keeps_the_search_space() -> None:
    # Seven heads or one head leave one lane per thread, and one key a
    # one-key tile, which the jam leaves as they are; the seed still turns
    # the knob on, so that it is searchable at every shape and
    # ``autotune_multi`` sees one search space.  One query and one key give
    # the scores stride 1 on the heads, the queries and the keys, one head
    # and one key the bias on the heads, the queries and the keys: the seed
    # still finds the keys walked and the heads as lanes.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        # (heads, queries, keys, lanes)
        shapes = [
            (8, 4, 512, 8),
            (7, 4, 512, 1),
            (1, 4, 512, 1),
            (8, 4, 1, 8),
            (8, 1, 1, 8),
            (1, 4, 1, 1),
        ]
        bounds = [
            _bind(keys, heads=heads, queries=queries)
            for heads, queries, keys, _ in shapes
        ]
        for (_, _, keys, lanes), bound in zip(shapes, bounds, strict=True):
            assert _jam_seed_of(bound)["block_sizes"] == [1, lanes, 1, keys, keys]
        assert (
            len({bound.config_spec.structural_fingerprint() for bound in bounds}) == 1
        )


@pytest.mark.parametrize(("keys", "chunk"), [(512, 512), (768, 256)])
def test_the_seed_jams_the_head_lanes(keys: int, chunk: int) -> None:
    # 768 keys take tiles of 256, which divide them: a masked key tile would
    # leave the head lanes rolled.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = _bind(keys)
        seed = _jam_seed_of(bound)
        code = bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
    assert seed["block_sizes"][3:] == [chunk, chunk]
    assert "for lane_1 in range(8)" not in code
    assert _BIAS_PACKET in code


def test_a_scalar_indexed_batch_still_seeds() -> None:
    # ``bias[0, ...]`` drops the batch dimension from the loaded value; the
    # heads are still the bias's stride-1 dimension.
    kernel = helion.kernel(leading_batch_bias_exp, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        scores = torch.empty(2, 8, 4, 512, dtype=torch.bfloat16)
        bias = torch.empty(1, 4, 512, 8, dtype=torch.bfloat16).permute(0, 3, 1, 2)
        bound = cpu_bind(kernel, (scores, bias))
        assert _jam_seed_of(bound)["block_sizes"] == [1, 8, 1, 512]


def test_the_jam_seed_guards_no_dynamic_shape() -> None:
    # Under dynamic shapes the seed declines, reading the strides and extents
    # without comparing them: a shape guard would become a dispatch fact that
    # no generated code needs.
    kernel = helion.kernel(
        bias_softmax,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        scores = torch.empty(2, 8, 4, 512, dtype=torch.bfloat16)
        bias = torch.empty(2, 4, 512, 8, dtype=torch.bfloat16).permute(0, 3, 1, 2)
        bound = cpu_bind(kernel, (scores, bias))
        host = bound.host_function
        guards = len(bound.env.shape_env.guards)
        with bound.env, host:
            assert _jam_plan(bound) is None
        assert len(bound.env.shape_env.guards) == guards


def test_a_keys_innermost_bias_gets_no_jam_seed() -> None:
    # No heads to load together.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        assert _jam_plan(_bind(512, heads_innermost=False)) is None


def test_nested_device_loops_get_no_jam_seed() -> None:
    # The seed puts each device loop's elements on threads of their own:
    # nested, the queries' threads would multiply the keys' (32 x 64).
    kernel = helion.kernel(nested_bias_exp, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        scores = torch.empty(2, 8, 256, 512, dtype=torch.bfloat16)
        bias = torch.empty(2, 256, 512, 8, dtype=torch.bfloat16).permute(0, 3, 1, 2)
        assert _jam_plan(cpu_bind(kernel, (scores, bias))) is None


def _bias_loads(code: str) -> tuple[list[ast.Call], list[ast.Call]]:
    """The vector and the scalar loads of ``bias`` in ``code``."""
    vector: list[ast.Call] = []
    scalar: list[ast.Call] = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        text = ast.unparse(node)
        if text.startswith("cute.arch.load(bias.iterator") and _BIAS_PACKET in text:
            vector.append(node)
        elif text.startswith("(bias.iterator") and text.endswith(".load()"):
            scalar.append(node)
    return vector, scalar


def test_jam_loads_the_heads_of_a_key_in_one_vector() -> None:
    # Plain division: no redo branch, so every bias load is in the passes.
    code = _bias_softmax_code(cute_exact_div_form="plain", cute_grid_lane_jam=True)
    assert "for lane_1 in range(8)" not in code
    vector, scalar = _bias_loads(code)
    # One packet of the eight heads per key, at lane 0's head.
    assert len(vector) == 1
    assert all("indices_1_lane0" in ast.unparse(call) for call in vector)
    assert scalar == []
    # Each lane reads its head out of the packet.
    for lane in range(8):
        assert f"cutlass.Uint16(_jam_vec[{lane}]).bitcast(cutlass.BFloat16)" in code
    # The tile spans the row, so each key loop runs once: the second pass
    # reads the scores packets and the heads' bias packets the first loaded.
    assert "for tile_offset_4 in range(" not in code
    for lane in range(8):
        assert (
            f"_tile_unroll_vec_4_0_lane{lane} = _tile_unroll_vec_3_0_lane{lane}" in code
        )
    assert "_jam_stash.append(_jam_vec)" in code
    assert "_jam_vec_1 = _jam_stash[vec_lane_4]" in code


def test_knob_off_keeps_the_rolled_head_loop() -> None:
    code = _bias_softmax_code(cute_exact_div_form="plain")
    assert "for lane_1 in range(8)" in code
    vector, scalar = _bias_loads(code)
    assert vector == []
    assert len(scalar) == 2


def test_each_lane_keeps_its_own_division_redo() -> None:
    code = _bias_softmax_code(cute_exact_div_form="redo", cute_grid_lane_jam=True)
    tree = ast.parse(code)
    redos = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith("_cute_exact_div_redo_f32(")
    ]
    # The branch reruns one lane's normalization: a branch per lane, each
    # testing that lane's quotient bounds.
    assert len(redos) == 8
    for lane, redo in enumerate(redos):
        assert f"_helion_exdiv_min_0_lane{lane}" in ast.unparse(redo.test)
    # The passes themselves load the heads as one packet per key.
    vector, _ = _bias_loads(code)
    assert len(vector) == 1


def test_jam_batches_the_lanes_shared_memory_reduces() -> None:
    # 512 keys on 64 threads: the max and the sum are two-warp shared-memory
    # reduces, one fragment reduce of the eight heads each.
    code = _bias_softmax_code(
        keys=512, cute_exact_div_form="plain", cute_grid_lane_jam=True
    )
    assert code.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 2
    assert "_cute_grouped_reduce_shared_two_stage(" not in code
    off = _bias_softmax_code(keys=512, cute_exact_div_form="plain")
    assert off.count("_cute_grouped_reduce_shared_two_stage(") == 2


def test_jam_interleaves_rows_that_share_no_packet() -> None:
    # Rows are the outer dimension: the jam interleaves the four rows and
    # batches their sums, and loads each row's packets on its own.
    kernel = helion.kernel(row_norm, backend="cute", static_shapes=True)
    codes = []
    for jam in (False, True):
        with cpu_render_target(reduction_smem_budget_bytes=232448):
            bound = cpu_bind(kernel, (torch.empty(64, 2048, dtype=torch.bfloat16),))
            codes.append(
                bound.to_code(
                    bound.config_spec.normalized_config(
                        helion.Config(**ROW_NORM_CONFIG, cute_grid_lane_jam=jam)
                    )
                )
            )
    off, on = codes
    assert "for lane_0 in range(4)" in off
    assert "for lane_0 in range(4)" not in on
    assert on.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 1
    assert "_jam_vec" not in on
    for lane in range(4):
        assert f"indices_0_lane{lane} = tile_offset_0 + cutlass.Int32({lane})" in on


@pytest.mark.parametrize("overlapping", [False, True])
def test_stores_into_overlapping_views_keep_the_rolled_head_loop(
    overlapping: bool,
) -> None:
    # ``a`` and ``b`` view one buffer one head apart: ``a``'s head 1 is
    # ``b``'s head 0.  The rolled loop stores ``a[1]`` after ``b[0]``; the
    # jam would store every head of ``a`` first and let ``b[0]`` win, so it
    # keeps the rolled loop.  Separate outputs still jam.
    kernel = helion.kernel(rows_into_two, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        x = torch.empty(2, 8, 4, 256)
        if overlapping:
            buf = torch.empty(2, 9, 4, 256)
            a, b = buf[:, :-1], buf[:, 1:]
        else:
            a, b = torch.empty(2, 8, 4, 256), torch.empty(2, 8, 4, 256)
        bound = cpu_bind(kernel, (x, a, b))
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**heads_config(1), cute_grid_lane_jam=True)
            )
        )
    assert ("for lane_1 in range(8)" in code) == overlapping


def test_a_head_invariant_store_keeps_one_packet_per_lane() -> None:
    # The stored row does not vary with the head, but each lane stores it to
    # its own row: each lane fills a packet of its own (one list run once
    # would collect every lane's elements, and each store would write them
    # all).
    kernel = helion.kernel(
        invariant_rows_over_heads, backend="cute", static_shapes=True
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        rows = torch.empty(2, 4, 256, dtype=torch.bfloat16)
        bound = cpu_bind(
            kernel, (rows, torch.empty(2, 8, 4, 256, dtype=torch.bfloat16))
        )
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**heads_config(1), cute_grid_lane_jam=True)
            )
        )
    assert "for lane_1 in range(8)" not in code
    for lane in range(8):
        assert f"_tile_store_vals_3_0_lane{lane} = []" in code
        assert f"_tile_store_vals_3_0_lane{lane})" in code


def test_a_loaded_mask_is_no_mask_of_the_lane_index() -> None:
    # A boolean tensor's load compares the loaded byte with zero
    # (``keep[...].load() != 0``), addressing the row by the lane index but
    # not comparing the index: the rows still jam.
    kernel = helion.kernel(kept_row_norm, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        args = (
            torch.empty(64, 2048, dtype=torch.bfloat16),
            torch.empty(64, 2048, dtype=torch.bool),
        )
        bound = cpu_bind(kernel, args)
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**ROW_NORM_CONFIG, cute_grid_lane_jam=True)
            )
        )
    assert "!= cutlass.Uint8(0)" in code
    assert "for lane_0 in range(4)" not in code
    assert "indices_0_lane3 = tile_offset_0 + cutlass.Int32(3)" in code


def test_rows_updated_in_place_are_each_lanes_own() -> None:
    # Every load and store of ``x`` names the lane's head: the lanes touch
    # disjoint rows and jam.  (The lane loop nest check already rejects a
    # row another lane updates.)
    kernel = helion.kernel(scale_rows_in_place, backend="cute", static_shapes=True)
    codes = []
    for jam in (False, True):
        with cpu_render_target(reduction_smem_budget_bytes=232448):
            bound = cpu_bind(kernel, (torch.empty(2, 8, 4, 256),))
            codes.append(
                bound.to_code(
                    bound.config_spec.normalized_config(
                        helion.Config(**heads_config(1), cute_grid_lane_jam=jam)
                    )
                )
            )
    assert "for lane_1 in range(8)" in codes[0]
    assert "for lane_1 in range(8)" not in codes[1]


def test_a_row_stored_in_place_is_read_back_from_memory() -> None:
    # The second loop's loads of ``x`` address what the first loop loaded,
    # but the first loop stored to it in between: no load of a stored tensor
    # is served from an earlier one.
    kernel = helion.kernel(doubled_rows_plus_one, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (torch.empty(64, 512, dtype=torch.bfloat16),))
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**SCALAR_ROWS_CONFIG, cute_grid_lane_jam=True)
            )
        )
    assert "for lane_0 in range(4)" not in code
    loads = [
        line for line in code.splitlines() if "x.iterator" in line and ".load()" in line
    ]
    assert len(loads) == 8


@contextlib.contextmanager
def _head_rows_bound(
    bias: torch.Tensor,
    rows: int = 16,
    keys: int = 512,
    *,
    static_shapes: bool = True,
    capability: tuple[int, int] = (10, 0),
) -> Iterator[BoundKernel]:
    """``head_bias_softmax_bwd`` bound on CPU for ``rows`` rows of four
    queries over ``keys`` keys and the ``[heads, q, k]`` ``bias``, for a
    target of ``capability``."""
    kernel = helion.kernel(
        head_bias_softmax_bwd, backend="cute", static_shapes=static_shapes
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        scores = torch.empty(rows, 4, keys, dtype=torch.bfloat16)
        args = (scores, scores, bias, *[torch.empty(rows, 4)] * 2)
        yield cpu_bind(kernel, args, capability=capability)


def _head_rows_code(**knobs: object) -> str:
    with _head_rows_bound(head_bias(8, 4, 512, "cpu")) as bound:
        config = {**HEAD_ROWS_CONFIG, **knobs}
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**config))
        )


def _bias_packets(code: str, width: int) -> int:
    packet = f"ir.VectorType.get([{width}], cutlass.Uint32.mlir_type)"
    return sum(
        1
        for line in code.splitlines()
        if "cute.arch.load(bias.iterator" in line and packet in line
    )


def test_head_rows_jam_around_their_workspace() -> None:
    # Each head lane stores its row's probabilities to the workspace and
    # reads them back after the row sum: every access of the workspace names
    # the lane's row.  The head ``bh % 8`` of consecutive rows is consecutive,
    # so the four heads' bias elements of a key are one 16-byte packet, and
    # the lanes' read-after-write barriers are one barrier.
    off = _head_rows_code(cute_grid_lane_jam=False)
    on = _head_rows_code(cute_grid_lane_jam=True)
    assert "for lane_0 in range(4)" in off
    assert "for lane_0 in range(4)" not in on
    assert on.count("bias.iterator") == 1
    assert _bias_packets(on, 4) == 1
    assert on.count("cute.arch.sync_threads()") == 1


def test_eight_fp32_head_lanes_load_two_packets() -> None:
    code = _head_rows_code(block_sizes=[8, 1], cute_grid_lane_jam=True)
    assert "for lane_0 in range(8)" not in code
    assert _bias_packets(code, 4) == 2


def _jammed_head_rows(fn: object, *args: torch.Tensor) -> str:
    """``fn`` (a head-rows kernel) at ``HEAD_ROWS_CONFIG`` with the jam."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**HEAD_ROWS_CONFIG, cute_grid_lane_jam=True)
            )
        )


def test_heads_of_a_period_not_a_multiple_of_the_lanes_load_one_by_one() -> None:
    # ``bh % 6`` of four consecutive rows wraps inside a run of four: the
    # heads are not consecutive, and each lane loads its own.
    rows = torch.empty(16, 4, 512, dtype=torch.bfloat16)
    bias = head_bias(6, 4, 512, "cpu")
    stats = torch.empty(16, 4)
    code = _jammed_head_rows(head_bias_softmax_bwd, rows, rows, bias, stats, stats)
    assert "for lane_0 in range(4)" not in code
    assert _bias_packets(code, 4) == 0
    assert code.count("bias.iterator") == 4


def test_a_hinted_load_keeps_its_hint_in_the_packet() -> None:
    rows = torch.empty(16, 4, 512, dtype=torch.bfloat16)
    code = _jammed_head_rows(hinted_head_rows, rows, head_bias(8, 4, 512, "cpu"))
    (packet,) = {
        line.strip().partition(" = ")[2]
        for line in code.splitlines()
        if line.strip().startswith("_jam_vec")
    }
    assert "ir.VectorType.get([4], cutlass.Uint32.mlir_type)" in packet
    assert packet.endswith("level1_eviction_priority='evict_last')")


def test_each_head_lane_reloads_from_its_own_register_cache() -> None:
    # The second sweep reads the gradient from the first sweep's register
    # cache: each lane writes its slots before reading them, so each lane's
    # copy gets a cache of its own, which replaces the cache allocated before
    # the loop.
    code = _head_rows_code(cute_reduction_reloads=["register"], cute_grid_lane_jam=True)
    assert "for lane_0 in range(4)" not in code
    assert "_fuse_cache_0 = cute.make_rmem_tensor(" not in code
    for lane in range(4):
        assert f"_fuse_cache_0_lane{lane} = cute.make_rmem_tensor(" in code
        assert f"_fuse_cache_0_lane{lane}[" in code


def test_jammed_head_rows_load_their_keys_in_packets() -> None:
    # The seed's alternate at four keys per thread: each head lane loads its
    # row's four bf16 scores and gradients as one packet and keeps its own
    # register cache, and the heads' bias elements of a key stay one packet.
    code = _head_rows_code(
        cute_reduction_reloads=["register"],
        cute_vector_widths=[4, 1, 1],
        cute_grid_lane_jam=True,
    )
    assert "for lane_0 in range(4)" not in code
    packet = "ir.VectorType.get([4], cutlass.Uint16.mlir_type)"
    for tensor in ("scores", "grad"):
        loads = [
            line
            for line in code.splitlines()
            if f"cute.arch.load({tensor}.iterator" in line
        ]
        assert len(loads) == 4
        assert all(packet in line for line in loads)
    assert _bias_packets(code, 4) == 1
    for lane in range(4):
        assert f"_fuse_cache_0_lane{lane} = cute.make_rmem_tensor(" in code


def test_heads_innermost_bias_seeds_the_jammed_head_lanes() -> None:
    jammed = [
        seed.config
        for seed in _bias_softmax_seeds(512, heads_innermost=True)
        if seed.config.get("cute_grid_lane_jam")
    ]
    # The eight bf16 heads of one 16-byte packet as one thread's lanes, the
    # keys of each loop on threads of 16-byte packets.
    assert len(jammed) == 1
    seed = jammed[0]
    assert seed["block_sizes"] == [1, 8, 1, 512, 512]
    assert seed["num_threads"] == [1, 1, 1, 64, 64]
    assert seed["cute_vector_widths"] == [1, 1, 1, 8, 8]
    # A keys-innermost bias has no heads to load together.
    assert not any(
        seed.config.get("cute_grid_lane_jam")
        for seed in _bias_softmax_seeds(512, heads_innermost=False)
    )


def _head_rows_seeds(
    bias: torch.Tensor, keys: int = 512, capability: tuple[int, int] = (10, 0)
) -> list[helion.Config]:
    with _head_rows_bound(bias, keys=keys, capability=capability) as bound:
        # The workspace's forwarding twin (``cute_scratch_forwarding``) is
        # ``test_cute_scratch_forwarding.py``'s.
        return [
            seed.config
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_grid_lane_jam")
            and not seed.config.get("cute_scratch_forwarding")
        ]


def test_a_heads_innermost_bias_row_seeds_jammed_rows_around_the_reduction() -> None:
    # Four rows as one thread's lanes (four fp32 bias elements of a key fill
    # a 16-byte packet), each row's 512 keys on 128 threads in one chunk of a
    # looped reduction that reloads from registers.
    seeds = _head_rows_seeds(head_bias(8, 4, 512, "cpu"))
    seed = seeds[0]
    assert seed["block_sizes"] == [4, 1]
    assert seed["num_threads"] == [1, 1, 128]
    assert seed["reduction_loops"] == [512]
    assert seed["cute_reduction_reloads"] == ["register"]
    # The same seed at four keys per thread (eight would leave threads
    # without keys in a 512-key chunk); the primary keeps one.
    assert "cute_vector_widths" not in seed
    assert [alternate["cute_vector_widths"] for alternate in seeds[1:]] == [[4, 1, 1]]
    assert {**seeds[1], "cute_vector_widths": None} == {
        **seed,
        "cute_vector_widths": None,
    }
    # 300 keys fill 300 slots of their 512-key chunk: at four keys per thread
    # 53 of the 128 threads would get none.
    assert len(_head_rows_seeds(head_bias(8, 4, 300, "cpu"), keys=300)) == 1
    # T5's 1024-key rows take both alternates; eight-key packets would not
    # tile rows of 1500 keys.
    for keys, widths in ((1024, [[4, 1, 1], [8, 1, 1]]), (1500, [[4, 1, 1]])):
        alternates = _head_rows_seeds(head_bias(8, 4, keys, "cpu"), keys=keys)[1:]
        assert [alternate["cute_vector_widths"] for alternate in alternates] == widths
    # Before sm_100 a lane loads at most 16 bytes whole: eight keys of the
    # fp32 workspace the second sweep reads back would be 32.
    alternates = _head_rows_seeds(
        head_bias(8, 4, 1024, "cpu"), keys=1024, capability=(9, 0)
    )[1:]
    assert [alternate["cute_vector_widths"] for alternate in alternates] == [[4, 1, 1]]
    # A keys-innermost bias row is no gather: no seed.
    assert _head_rows_seeds(torch.empty(8, 4, 512)) == []


@pytest.mark.parametrize(("keys", "chunk"), [(768, 256), (1536, 512)])
def test_the_head_rows_chunk_divides_the_row(keys: int, chunk: int) -> None:
    # A 1024-key chunk leaves the last chunk of these rows masked, and each
    # lane's bias element of a key a masked scalar load: no packet forms.
    with _head_rows_bound(head_bias(8, 4, keys, "cpu"), keys=keys) as bound:
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["reduction_loops"] == [chunk]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code
    assert _bias_packets(code, 4) == 1


def test_a_row_divided_only_by_one_key_per_thread_keeps_its_chunk() -> None:
    # 2176 keys are 17 chunks of 128, one key per thread each: the seed
    # keeps the 1024-key chunk, its last one masked.
    with _head_rows_bound(head_bias(8, 4, 2176, "cpu"), keys=2176) as bound:
        seed = _jam_seed(bound)
    assert seed is not None
    assert seed.config["reduction_loops"] == [1024]


def _jam_seed(bound: BoundKernel) -> helion.Config | None:
    """``cute_reduction_lane_jam``'s primary seed for ``bound``, when the
    heuristic applies."""
    host = bound.host_function
    assert host is not None
    heuristic = CuteReductionLaneJamHeuristic
    if not heuristic.is_eligible(bound.env, host.device_ir):
        return None
    return heuristic.get_seed_config(bound.env, host.device_ir)


def _assert_reaches_the_search(bound: BoundKernel, seed: helion.Config) -> None:
    """``seed`` is a compiler seed that survives the search's transfer."""
    assert seed in bound.config_spec.compiler_seed_configs
    generation = ConfigGeneration(bound.config_spec)
    surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
    _flat, roundtrip = generation.canonicalize_flat(generation.flatten(seed))
    assert roundtrip in surviving


def test_the_head_rows_seed_reaches_the_search_jammed() -> None:
    with _head_rows_bound(head_bias(8, 4, 512, "cpu")) as bound:
        seed = _jam_seed(bound)
        assert seed is not None
        _assert_reaches_the_search(bound, seed)
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    # The four rows' lane loop is jammed: their bias elements of a key are
    # one 16-byte packet.
    assert "for lane_0 in range(4)" not in code
    assert _bias_packets(code, 4) == 1


def test_the_head_rows_alternates_reach_the_search() -> None:
    # The search's transfer keeps each alternate's vector width (T5's
    # 1024-key rows plant both).
    with _head_rows_bound(head_bias(8, 4, 1024, "cpu"), keys=1024) as bound:
        alternates = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_grid_lane_jam")
            and seed.config.get("cute_vector_widths")
            and not seed.config.get("cute_scratch_forwarding")
        ]
        assert len(alternates) == 2
        for seed in alternates:
            _assert_reaches_the_search(bound, seed)


def test_twelve_bf16_heads_take_four_lanes() -> None:
    # Eight lanes fill a 16-byte packet, but heads 8 to 11 and 0 to 3 of the
    # next batch are no packet: the lanes also divide the bias's heads.
    bias = torch.empty(4, 512, 12, dtype=torch.bfloat16).permute(2, 0, 1)
    with _head_rows_bound(bias, rows=24) as bound:
        seed = _jam_seed(bound)
    assert seed is not None
    assert seed.block_sizes == [4, 1]


def test_wide_head_rows_reload_from_memory() -> None:
    # Four lanes of 4096 keys on 128 threads would cache 128 values per
    # thread, which spills: the later sweeps reload from memory.
    with _head_rows_bound(head_bias(8, 4, 4096, "cpu"), keys=4096) as bound:
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        _assert_reaches_the_search(bound, seed)


def _head_bias_softmax(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Softmax of each (batch * head) row of ``x`` plus its head's ``bias``
    row (``head_bias``) in three sweeps: the max, the sum of the
    exponentials, and the output, which reads the exponentials again."""
    bh, q_len, _ = x.shape
    heads = bias.size(0)
    out = torch.empty_like(x)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        v = x[tile_bh, tile_q, :].to(torch.float32) + bias[head, tile_q, :]
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        out[tile_bh, tile_q, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(out.dtype)
    return out


def test_a_three_sweep_softmax_counts_the_exponentials_its_output_reads() -> None:
    # Each of the four lanes caches its eight elements per thread of x and
    # of the bias for the sum sweep, and the sum sweep's exponentials for the
    # output sweep: 96 values, past the 64 one thread's caches hold, so the
    # later sweeps reload from memory and the lanes jam.
    kernel = helion.kernel(_head_bias_softmax, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        x = torch.empty(16, 4, 1024, dtype=torch.bfloat16)
        bound = cpu_bind(kernel, (x, head_bias(8, 4, 1024, "cpu")))
        host = bound.host_function
        assert host is not None
        loop = bound.config_spec.reduction_loops[0]
        heuristic = CuteReductionLaneJamHeuristic
        assert heuristic._cached_values(host.device_ir, loop.block_id) == 3
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code


def _sweeps_cache(*sweeps: Callable[[torch.fx.Graph], torch.fx.Node]) -> int:
    """``_cached_values`` of a looped reduction whose sweeps each output the
    node their builder adds to a graph."""
    graphs = []
    for graph_id, build in enumerate(sweeps):
        graph = torch.fx.Graph()
        graph.output(build(graph))
        graphs.append(
            ReductionLoopGraphInfo(
                graph_id=graph_id, graph=graph, node_args=[], block_ids=[1]
            )
        )
    device_ir = SimpleNamespace(graphs=graphs)
    return CuteReductionLaneJamHeuristic._cached_values(device_ir, 1)


def test_the_cache_count_compares_a_shared_operand_once() -> None:
    # An exp of 64 doublings in two sweeps: each value is numbered once, so
    # the comparison walks each shared operand once, not once per path.
    def doubled_exp(graph: torch.fx.Graph) -> torch.fx.Node:
        value = graph.placeholder("x")
        for _ in range(64):
            value = graph.call_function(torch.ops.aten.add.Tensor, (value, value))
        return graph.call_function(torch.ops.aten.exp.default, (value,))

    assert _sweeps_cache(doubled_exp, doubled_exp) == 1


_SIXTEEN_ADDS = "add.u32 $0, $1, 1;" + " add.u32 $0, $0, 1;" * 15


@pytest.mark.parametrize(
    ("text", "pure", "cached"),
    [
        ("mov.b32 $0, $1;", True, 0),
        (_SIXTEEN_ADDS, True, 1),
        (_SIXTEEN_ADDS, False, 0),
        ("ld.global.u32 $0, [$1];", True, 0),
    ],
    ids=["one_instruction", "sixteen_instructions", "impure", "memory_read"],
)
def test_the_cache_count_includes_pure_inline_asm(
    text: str, pure: bool, cached: int
) -> None:
    # A later sweep reads pure inline asm (a word0 ``hl.rand`` draw's Philox
    # chain) an earlier sweep computed from a cache, as it does an exp, when
    # it saves ``MIN_SAVED_COST`` (16) instructions; a shorter one, impure asm
    # and asm that is not scalar arithmetic run again.
    def asm(graph: torch.fx.Graph) -> torch.fx.Node:
        x = graph.placeholder("x")
        operands = (text, "=r,r", [x], torch.int32, pure, 1)
        return graph.call_function(hl.inline_asm_elementwise, operands)

    assert _sweeps_cache(asm, asm) == cached


def _head_bias_branches(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Three values built from one exponential of each (batch * head) row of
    ``x`` plus its head's ``bias`` row, reduced together and stored after."""
    bh, q_len, _ = x.shape
    heads = bias.size(0)
    out = torch.empty_like(x)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        e = torch.exp(x[tile_bh, tile_q, :].to(torch.float32) + bias[head, tile_q, :])
        a, b, c = e + 1.0, e + 2.0, e + 3.0
        total = torch.sum(a * b * c, dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (a * total + b / total + c).to(out.dtype)
    return out


def test_values_built_from_one_exponential_are_cached_each() -> None:
    # The output sweep reads ``e + 1``, ``e + 2`` and ``e + 3`` from three
    # caches of eight elements per thread for each of the four lanes, past
    # the 64 one thread's caches hold: the later sweep reloads from memory and
    # the lanes jam.
    kernel = helion.kernel(_head_bias_branches, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        x = torch.empty(16, 4, 1024, dtype=torch.bfloat16)
        bound = cpu_bind(kernel, (x, head_bias(8, 4, 1024, "cpu")))
        host = bound.host_function
        assert host is not None
        loop = bound.config_spec.reduction_loops[0]
        heuristic = CuteReductionLaneJamHeuristic
        assert heuristic._cached_values(host.device_ir, loop.block_id) == 3
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code


def test_the_cache_count_numbers_the_nodes_of_a_list_operand() -> None:
    # Pure inline asm of one value in two sweeps, its operand named
    # differently in the second: equal values, one cache.
    def asm(rename: bool) -> Callable[[torch.fx.Graph], torch.fx.Node]:
        def build(graph: torch.fx.Graph) -> torch.fx.Node:
            x = graph.placeholder("x")
            if rename:
                graph.call_function(torch.ops.aten.add.Tensor, (x, 2))
            y = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
            operands = (_SIXTEEN_ADDS, "=r,r", [y], torch.int32, True, 1)
            return graph.call_function(hl.inline_asm_elementwise, operands)

        return build

    assert _sweeps_cache(asm(False), asm(True)) == 1


def test_the_cache_count_numbers_a_deep_chain() -> None:
    # An exp of a chain of 1000 additions in two sweeps, numbered in node
    # order: no recursion per link.
    def chain_exp(graph: torch.fx.Graph) -> torch.fx.Node:
        value = graph.placeholder("x")
        for step in range(1000):
            value = graph.call_function(torch.ops.aten.add.Tensor, (value, step))
        return graph.call_function(torch.ops.aten.exp.default, (value,))

    assert _sweeps_cache(chain_exp, chain_exp) == 1


def test_the_carried_ops_are_the_fusers_transcendentals() -> None:
    # The seed counts the aten forms of what ``fuse_two_pass_loads`` carries
    # (``sweep_values.TRANSCENDENTALS``), and sigmoid, which lowers through
    # exp.
    from helion._compiler.cute.autotune.heuristics.cute import _SWEEP_CARRIED_OPS
    from helion._compiler.cute.sweep_values import TRANSCENDENTALS

    names = {op.__name__.split(".")[0] for op in _SWEEP_CARRIED_OPS}
    assert names == {name.rsplit(".", 1)[-1] for name in TRANSCENDENTALS} | {"sigmoid"}


@helion.kernel(backend="cute", static_shapes=True)
def _gathered_row_dot_plus_narrow_sum(
    x: torch.Tensor, xt: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # ``xt[tile_b, tile_m, :]`` gathers a row of a transposed tensor; ``y``'s
    # 16 columns are a second, static reduction dim with a reload slot of its
    # own.
    b, m = x.size(0), x.size(1)
    out = torch.empty([b, m], dtype=torch.float32, device=x.device)
    for tile_b, tile_m in hl.tile([b, m]):
        row = x[tile_b, tile_m, :].to(torch.float32)
        dot = torch.sum(row * xt[tile_b, tile_m, :].to(torch.float32), dim=-1)
        narrow = torch.sum(y[tile_b, tile_m, :].to(torch.float32), dim=-1)
        out[tile_b, tile_m] = dot + narrow
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _gathered_rows_over_heads(x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    # ``xt[tile_b, tile_m, :, :]`` gathers rows too, but both operands span
    # the keys and a second, narrow reduction dim, the heads: each key's
    # values are normalized over the keys, then summed over the heads.
    b, m, k = x.size(0), x.size(1), x.size(2)
    out = torch.empty([b, m, k], dtype=torch.float32, device=x.device)
    for tile_b, tile_m in hl.tile([b, m]):
        e = torch.exp(x[tile_b, tile_m, :, :] + xt[tile_b, tile_m, :, :])
        out[tile_b, tile_m, :] = torch.sum(
            e / torch.sum(e, dim=-2, keepdim=True), dim=-1
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _gathered_row_dot(x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.sum(
            x[tile_m, :].to(torch.float32) * xt[tile_m, :].to(torch.float32), dim=-1
        )
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _gathered_row_scale(x: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        dot = torch.sum(
            x[tile_m, :].to(torch.float32) * xt[tile_m, :].to(torch.float32), dim=-1
        )
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] * dot[:, None]).to(out.dtype)
    return out


def _gathered_rows(*batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """64 rows of 512 bf16 keys under ``batch``, and their transpose's
    view: a row of the view is a gather."""
    x = torch.empty(*batch, 64, 512, dtype=torch.bfloat16)
    return x, torch.empty(*batch, 512, 64, dtype=torch.bfloat16).transpose(-1, -2)


def _packet_loads(code: str, tensor: str) -> int:
    """Loads in ``code`` of a 16-byte packet of eight bf16 of ``tensor``."""
    return sum(
        1
        for line in code.splitlines()
        if f"cute.arch.load({tensor}.iterator" in line and _BIAS_PACKET in line
    )


def test_the_seed_sets_the_reload_mode_of_the_looped_reduction_only() -> None:
    # The second reduction dim keeps its reload slot and runs on one thread:
    # an automatic count there does not lower beside the lanes' reduced
    # stores.  The seed renders with the rows' lane loop jammed.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        y = torch.empty(2, 64, 16, dtype=torch.bfloat16)
        bound = cpu_bind(_gathered_row_dot_plus_narrow_sum, (*_gathered_rows(2), y))
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["num_threads"] == [1, 1, 128, 1]
        assert seed.config["cute_reduction_reloads"] == ["register", "auto"]
        _assert_reaches_the_search(bound, seed)
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_1 in range(8)" not in code
    assert _packet_loads(code, "xt") == 1


def test_a_second_reduction_dim_the_rows_span_gets_no_seed() -> None:
    # The seed runs a second reduction dim on one thread, which lowers for a
    # narrow sum of its own (above), not for the heads both operands span
    # here: rendering it raised "reduction marker is nested in a different
    # lane owner".  So the seed is not offered, and the default renders.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        x = torch.empty(2, 16, 4, 1024).transpose(-1, -2)
        xt = torch.empty(2, 1024, 4, 16).permute(0, 3, 1, 2)
        bound = cpu_bind(_gathered_rows_over_heads, (x, xt))
        assert _jam_seed(bound) is None
        assert not any(
            seed.config.get("cute_grid_lane_jam")
            for seed in bound.config_spec.compiler_seed_configs
        )
        bound.to_code(bound.config_spec.default_config())


def test_a_one_axis_grid_gets_no_seed() -> None:
    # A one-axis grid tile lays out flattened, whose lanes the jam keeps
    # rolled: the seed's knob would change nothing.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_gathered_row_dot, _gathered_rows())
        assert _jam_seed(bound) is None
        config = helion.Config(
            block_sizes=[8],
            num_threads=[1, 128],
            reduction_loops=[512],
            cute_grid_lane_jam=True,
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    assert "for lane_0 in range(8)" in code


@helion.kernel(backend="cute", static_shapes=True)
def _weighted_column_sums(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    b, n = x.size(0), x.size(2)
    out = torch.empty([b, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_n in hl.tile([b, n]):
        weights = w[:].to(torch.float32)
        out[tile_b, tile_n] = torch.sum(
            x[tile_b, :, tile_n].to(torch.float32) * weights[None, :, None], dim=1
        )
    return out


def test_weighted_column_sums_seed_jammed_columns() -> None:
    # The columns are ``x``'s stride-1 dim, so the lanes' elements of a row
    # are one packet.  ``w[:]`` walks the reduced rows without the lanes'
    # axis, and each lane stores its column's sum: the jam takes both.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        x = torch.empty(2, 512, 64, dtype=torch.bfloat16)
        w = torch.empty(512, dtype=torch.bfloat16)
        bound = cpu_bind(_weighted_column_sums, (x, w))
        seed = _jam_seed(bound)
        assert seed is not None
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_1 in range(8)" not in code
    assert _packet_loads(code, "x") == 1


@helion.kernel(backend="cute", static_shapes=True)
def _two_reloaded_rows(
    grad: torch.Tensor, scores: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # The second sweep reads ``grad`` and ``scores`` again, each on its own:
    # each lane caches two values per key.
    bh, q_len = scores.size(0), scores.size(1)
    heads = bias.size(0)
    out = torch.empty_like(scores)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        g = grad[tile_bh, tile_q, :].to(torch.float32)
        s = scores[tile_bh, tile_q, :].to(torch.float32)
        total = torch.sum(g * (s + bias[head, tile_q, :]), dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (g * total + s).to(out.dtype)
    return out


def test_two_reloaded_loads_count_against_the_register_budget() -> None:
    # Four lanes of 2048 keys on 128 threads: one cached load would hold 64
    # values per thread, but two hold 128, past what the jam's lanes may
    # copy.  The seed reloads from memory, and its lanes jam.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        rows = torch.empty(16, 4, 2048, dtype=torch.bfloat16)
        bias = head_bias(8, 4, 2048, "cpu")
        bound = cpu_bind(_two_reloaded_rows, (rows, rows, bias))
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code


@helion.kernel(backend="cute", static_shapes=True)
def _row_plus_bias_by_asm(rows: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # A one-instruction pure asm adds each row's head bias row, which the
    # sum and the output both read.
    bh, q_len = rows.size(0), rows.size(1)
    heads = bias.size(0)
    out = torch.empty_like(rows)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        v = hl.inline_asm_elementwise(
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            [rows[tile_bh, tile_q, :].to(torch.float32), bias[head, tile_q, :]],
            torch.float32,
            True,
            1,
        )
        total = torch.sum(v, dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (v * total).to(out.dtype)
    return out


def test_a_cheap_pure_asm_counts_the_loads_it_reads() -> None:
    # The output sweep computes the one-instruction asm again, so the fuser
    # caches its two loads, not the asm: 128 values per thread for four
    # lanes of 2048 keys, past what the jam's lanes may copy.  The seed
    # reloads from memory, and its lanes jam.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        rows = torch.empty(16, 4, 2048, dtype=torch.bfloat16)
        bias = head_bias(8, 4, 2048, "cpu")
        bound = cpu_bind(_row_plus_bias_by_asm, (rows, bias))
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code


@helion.kernel(backend="cute", static_shapes=True)
def _three_reloaded_rows(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # The second sweep reads ``a``, ``b`` and ``c`` again: three cached
    # values per key.
    bh, q_len = a.size(0), a.size(1)
    heads = bias.size(0)
    out = torch.empty_like(a)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        x = a[tile_bh, tile_q, :].to(torch.float32)
        y = b[tile_bh, tile_q, :].to(torch.float32)
        z = c[tile_bh, tile_q, :].to(torch.float32)
        total = torch.sum(x * y * (z + bias[head, tile_q, :]), dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (x * total + y * z).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _exp_rows(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Both sweeps take three exponentials of one row, which register reloads
    # carry from the first: three cached values per key.
    bh, q_len = x.size(0), x.size(1)
    heads = bias.size(0)
    out = torch.empty_like(x)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        v = x[tile_bh, tile_q, :].to(torch.float32) + bias[head, tile_q, :]
        total = torch.sum(
            torch.exp(v) + torch.exp(-v) + torch.exp(2 * v), dim=-1, keepdim=True
        )
        out[tile_bh, tile_q, :] = (
            (torch.exp(v) - torch.exp(-v) + torch.exp(2 * v)) / total
        ).to(out.dtype)
    return out


@pytest.mark.parametrize(
    ("fn", "keys", "rows"),
    [(_three_reloaded_rows, 600, 3), (_exp_rows, 1024, 1)],
    ids=["padded_chunk", "carried_exponentials"],
)
def test_the_register_budget_counts_every_cache(
    fn: helion.Kernel, keys: int, rows: int
) -> None:
    # Four lanes on 128 threads.  600 keys take a 1024-key chunk, and each
    # cache holds a thread's eight elements of it, padding included: three
    # reread rows are 96 values per thread, not 60.  Three exponentials of a
    # row are three caches besides its loads.  The lanes' caches would not
    # fit, so the seed reloads from memory, and its lanes jam.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        loaded = torch.empty(16, 4, keys, dtype=torch.bfloat16)
        bound = cpu_bind(fn, (*[loaded] * rows, head_bias(8, 4, keys, "cpu")))
        seed = _jam_seed(bound)
        assert seed is not None
        assert seed.config["cute_reduction_reloads"] == ["gmem"]
        code = bound.to_code(bound.config_spec.normalized_config(seed))
    assert "for lane_0 in range(4)" not in code


def test_a_device_loop_beside_the_reduction_gets_no_seed() -> None:
    # The seed tiles every block but the lanes' by 1 on one thread: a device
    # loop's 512 columns would run on one thread per row.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_gathered_row_scale, _gathered_rows())
        assert _jam_seed(bound) is None


def test_an_odd_row_count_still_seeds_one_lane() -> None:
    # Fifteen rows leave one lane per CTA; the seed keeps the knob
    # searchable at every row count (one search space for autotune_multi).
    fingerprints = set()
    for rows in (15, 16):
        with _head_rows_bound(head_bias(8, 4, 512, "cpu"), rows=rows) as bound:
            seed = _jam_seed(bound)
            assert seed is not None
            assert seed.block_sizes == [1 if rows == 15 else 4, 1]
            fingerprints.add(bound.config_spec.structural_fingerprint())
    assert len(fingerprints) == 1


def test_a_dynamic_shape_gets_no_head_rows_seed() -> None:
    # The rows' tile keeps a mask, which the jam does not take: the seed's
    # knob would change nothing.
    with _head_rows_bound(head_bias(8, 4, 512, "cpu"), static_shapes=False) as bound:
        assert _jam_seed(bound) is None
