"""Later reduction sweeps reuse the first sweep's expensive pure values.

A persistent reduction runs one lane loop per sweep and re-emits the row's
elementwise chain in each.  ``fuse_two_pass_loads`` (with ``sweep_values``)
reads a later sweep's copy of the first sweep's pure inline-asm results, the
Philox chain of a word0 ``hl.rand`` draw, from a per-thread fragment, and with
the explicit ``register`` reload policy its transcendentals too; the ``gmem``
policy still recomputes them.  A lane-invariant load takes one fragment slot,
and a scalar load that names its dtype (an fp8 row's raw bytes) caches in it.
The kernels render on CPU; ``test_cute_fuse_two_pass_loads.py`` checks on the
GPU that the carried masks match the recomputed ones bit for bit.
"""

from __future__ import annotations

import ast
import functools
import re
import textwrap
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.cut_carried_values import _read_carried_values
from helion._compiler.cute.cut_carried_values import cut_carried_values
from helion._compiler.cute.fuse_fma import fuse_fma
from helion._compiler.cute.fuse_two_pass_loads import FUSE_CARRIED_READ_ATTR
from helion._compiler.cute.fuse_two_pass_loads import FUSE_CARRY_ATTR
from helion._compiler.cute.fuse_two_pass_loads import FUSE_CARRY_DEAD_ATTR
from helion._compiler.cute.fuse_two_pass_loads import _arch_scalar_dtype
from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.cute.fuse_two_pass_loads import _load_kind
from helion._compiler.cute.fuse_two_pass_loads import _scalar_load_key
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.sweep_values import CACHE_DTYPES
from helion._compiler.cute.sweep_values import SweepValues
from helion._compiler.cute.sweep_values import _promoted
from helion._compiler.cute.sweep_values import _scalar_dtype
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_sweep_kernels import THIRD_SOFTMAX_CONFIG
from helion._testing.cute_sweep_kernels import center_fp8_row
from helion._testing.cute_sweep_kernels import center_fp8_row_config
from helion._testing.cute_sweep_kernels import dropout_layernorm
from helion._testing.cute_sweep_kernels import third_softmax
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        top = torch.amax(row, dim=-1, keepdim=True)
        e = torch.exp(row - top)
        out[tile_r, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(out.dtype)
    return out


def scaled_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :]
        top = torch.amax(row, dim=-1, keepdim=True)
        # A bf16 exp times a Python float: the DSL computes the product in
        # fp32, and it is rounded to bf16 as torch computes it.
        e = torch.exp(row - top) * 2.0
        out[tile_r, :] = e / torch.sum(e, dim=-1, keepdim=True)
    return out


def scaled_layernorm(
    x: torch.Tensor, scales: torch.Tensor, scale_index: hl.constexpr
) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        y = x[tile_r, :].to(torch.float32) * scales[scale_index]
        mean = torch.mean(y, dim=-1, keepdim=True)
        centered = y - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        out[tile_r, :] = centered * torch.rsqrt(var + 1e-5)
    return out


_KERNELS = {
    "dropout_layernorm": (
        dropout_layernorm,
        lambda: (
            torch.empty(64, 256, dtype=torch.bfloat16),
            torch.zeros(4, dtype=torch.int64),
            torch.empty(64, 256),
            torch.empty(256),
            torch.empty(256),
            2,
            255,
        ),
        {"cute_rng_stream": "word0"},
    ),
    "softmax": (softmax, lambda: (torch.empty(64, 512, dtype=torch.bfloat16),), {}),
    "scaled_softmax": (
        scaled_softmax,
        lambda: (torch.empty(64, 512, dtype=torch.bfloat16),),
        {},
    ),
    "scaled_layernorm": (
        scaled_layernorm,
        lambda: (torch.empty(64, 256, dtype=torch.bfloat16), torch.empty(4), 1),
        {},
    ),
}


@functools.cache
def _sweeps(
    name: str, reload: str, *, exact_division: bool = False, **options: object
) -> tuple[str, ...]:
    """The source of each top-level loop of ``name``'s kernel, rendered once
    per reload policy (and further config ``options``): one persistent sweep
    per loop."""
    fn, args, settings = _KERNELS[name]
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            cute_exact_division=exact_division,
            **settings,
        )
        bound = cpu_bind(kernel, args())
        config = helion.Config(
            block_sizes=[4],
            num_threads=[4, 32],
            cute_reduction_reloads=[reload],
            **options,
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == f"_helion_{name}"
    )
    return (
        ast.unparse(device),
        *(ast.unparse(node) for node in device.body if isinstance(node, ast.For)),
    )


@pytest.mark.parametrize("reload", ["auto", "register"])
def test_word0_draw_runs_in_the_first_sweep_only(reload: str) -> None:
    code, *sweeps = _sweeps("dropout_layernorm", reload)
    # Mean, variance and store sweeps; the Philox rounds' multiplies run in
    # the mean sweep only.
    assert len(sweeps) == 3
    assert sweeps[0].count("mul.hi.u32") > 0
    assert [sweep.count("mul.hi.u32") for sweep in sweeps[1:]] == [0, 0]
    # So does the key schedule (its 0x9E3779B9 increments): the store sweep's
    # hoisted lane-invariant half of the chain is gone too, with its seed load.
    assert code.count("2654435769") == sweeps[0].count("2654435769") > 0
    assert code.count("seeds.iterator") == 1
    # The variance sweep reads the dropped-out row; the store sweep also reads
    # the mask, which it stores.
    assert re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", sweeps[1])
    assert re.search(r"keep = cutlass\.Boolean\(_fuse_cache_\d+\[", sweeps[2])
    # The variance sweep needs neither input load any more.
    assert ".load()" not in sweeps[1]


def test_word0_draw_carries_the_latest_value_under_register_policy() -> None:
    _, *sweeps = _sweeps("dropout_layernorm", "register")
    # The register policy proves the store sweep's inputs unchanged too, so
    # it reads the dropped-out row instead of recombining the inputs.
    assert "x.iterator" not in sweeps[2]
    assert "residual.iterator" not in sweeps[2]


def test_mask_is_carried_as_a_boolean_inside_a_persistent_pid_loop() -> None:
    # The sweeps run in the persistent pid's loop, below the cutoffs that
    # ``lower_uniform_comparisons`` hoists to the kernel's root: the mask
    # still takes their DSL type, so the store sweep reads it from a Boolean
    # fragment instead of reading the draw and recomputing the mask.
    code, *_ = _sweeps("dropout_layernorm", "register", pid_type="persistent_blocked")
    assert "for virtual_pid in" in code
    assert re.search(r"= cute\.make_rmem_tensor\(8, cutlass\.Boolean\)", code)
    assert re.search(r"keep = cutlass\.Boolean\(_fuse_cache_\d+\[", code)
    assert "cutlass.Int64(_fuse_cache_" not in code


def test_gmem_policy_recomputes_the_draw() -> None:
    code, *sweeps = _sweeps("dropout_layernorm", "gmem")
    assert "_fuse_cache_" not in code
    assert len(sweeps) == 3
    multiplies = sweeps[0].count("mul.hi.u32")
    assert multiplies > 0
    assert [sweep.count("mul.hi.u32") for sweep in sweeps[1:]] == [multiplies] * 2


def test_register_policy_carries_a_transcendental() -> None:
    _, *sweeps = _sweeps("softmax", "register")
    assert [sweep.count("cute.math.exp(") for sweep in sweeps] == [0, 1, 0]
    assert re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", sweeps[2])
    assert " / " in sweeps[2]
    # The carried read stays a provable fp32 dividend, so under
    # ``cute_exact_division`` the normalize sweep keeps the exact division's
    # fast path.
    _, *sweeps = _sweeps("softmax", "register", exact_division=True)
    assert re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", sweeps[2])
    assert "_cute_exact_div_fast_f32" in sweeps[2]


@functools.cache
def _rolled_sweeps(name: str, reload: str, vector: int) -> tuple[str, ...]:
    """``_sweeps`` for a rolled reduction: each row's 512 elements in one
    chunk over 128 threads, each thread's elements in packets of ``vector``
    lanes, a V-loop per sweep when ``vector > 1``."""
    fn, args, settings = _KERNELS[name]
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(fn, backend="cute", static_shapes=True, **settings)
        bound = cpu_bind(kernel, args())
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[512],
            cute_vector_widths=[vector, 1],
            cute_reduction_reloads=[reload],
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == f"_helion_{name}"
    )
    return (
        ast.unparse(device),
        *(ast.unparse(node) for node in device.body if isinstance(node, ast.For)),
    )


@pytest.mark.parametrize("vector", [1, 4])
def test_register_policy_carries_a_transcendental_of_a_rolled_sweep(
    vector: int,
) -> None:
    # In packets, each sweep's elements run in a V-loop: the normalize
    # sweep reads the sum sweep's exp from a fragment of a slot per lane.
    _, *sweeps = _rolled_sweeps("softmax", "register", vector)
    assert [sweep.count("cute.math.exp(") for sweep in sweeps] == [0, 1, 0]
    read = re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[([^\]]*)\]\)", sweeps[2])
    assert read is not None
    assert (f"* {vector} + reduction_vec_lane_" in read.group(1)) == (vector > 1)


def test_gmem_policy_recomputes_a_transcendental_of_a_rolled_sweep() -> None:
    _, *sweeps = _rolled_sweeps("softmax", "gmem", 4)
    assert [sweep.count("cute.math.exp(") for sweep in sweeps] == [0, 1, 1]


@functools.cache
def _rolled_cluster_softmax(fast_math: bool, pair: bool) -> str:
    """The softmax of 32768-element rows over a cluster of four CTAs (two
    4096-element chunks each, 128 threads x 8-lane packets) under the
    register policy, with ``cute_cluster_warp_online`` set to ``pair``."""
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(
            softmax, backend="cute", static_shapes=True, fast_math=fast_math
        )
        bound = cpu_bind(kernel, (torch.empty(64, 32768, dtype=torch.float16),))
        config = helion.Config(
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[4096],
            cute_vector_widths=[8, 1],
            cute_reduction_reloads=["register"],
            cute_cluster_n=4,
            cute_cluster_warp_online=pair,
            cute_exact_div_form="plain",
        )
        return bound.to_code(bound.config_spec.normalized_config(config))


@pytest.mark.parametrize("fast_math", [False, True])
def test_rolled_cluster_pair_keeps_its_sweeps_values(fast_math: bool) -> None:
    # The rolled online-softmax pair moves the sum sweep's exp into the frame
    # of a CTA's or warp's own maximum after the fuser runs (under fast_math
    # the write sweep rescales cached copies of it), so an exp carried from
    # there would skip the rescale.  Under cute_cluster_warp_online no
    # V-loop value is carried and the pair forms; without the knob the write
    # sweep reads the carried exp.
    paired = _rolled_cluster_softmax(fast_math, pair=True)
    assert "_cute_grouped_reduce_cluster(" not in paired
    assert not re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", paired)
    if fast_math:
        assert "_pair_exp_cache_0[" in paired.split("_pair_rescale_0 = ")[1]
    else:
        assert paired.count("_cute_grouped_reduce_cluster_warp_online_pair(") == 1
    unpaired = _rolled_cluster_softmax(fast_math, pair=False)
    assert unpaired.count("_cute_grouped_reduce_cluster(") == 2
    assert re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", unpaired)


def test_auto_policy_recomputes_a_transcendental() -> None:
    _, *sweeps = _sweeps("softmax", "auto")
    assert sweeps[1].count("cute.math.exp(") == 1
    assert sweeps[2].count("cute.math.exp(") == 1


def _vloop_sweeps(
    first: list[str],
    later: list[str],
    *,
    between: str = "",
    trips: int = 2,
    lanes: int = 8,
    fma: bool = False,
    rename_groups: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Two rolled sweeps of ``trips`` chunks over ``lanes`` lanes of 4-lane
    V-loops (``first`` and ``later`` are their V-loop bodies, over the lane
    variables ``vec_lane_1`` and ``vec_lane_2``), through the fusion and the
    cut (both comparing names through ``rename_groups``), with ``fuse_fma``
    in between when ``fma``: the device body and the later sweep."""

    def sweep(lane: str, inner: list[str]) -> str:
        return (
            f"for roffset_7 in range(0, {32 * trips}, 32):\n"
            f"    for reduction_lane_7 in range({lanes}):\n"
            f"        for {lane} in cutlass.range_constexpr(4):\n"
            + "".join(f"            {line}\n" for line in inner)
        )

    body = fuse_two_pass_loads(
        ast.parse(
            sweep("vec_lane_1", first) + between + sweep("vec_lane_2", later)
        ).body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
        rename_groups=rename_groups,
    )
    body = cut_carried_values(fuse_fma(body) if fma else body, rename_groups)
    return ast.unparse(ast.Module(body=body, type_ignores=[])), ast.unparse(body[-1])


def _vloop_load(lane: str, offset: str = "") -> str:
    return (
        f"w = (x.iterator + {offset}roffset_7 + reduction_lane_7 * 4 + {lane}).load()"
    )


_SUM = "acc = acc + e"
_STORE = "(out.iterator + roffset_7 + reduction_lane_7 * 4 + vec_lane_2).store(e)"


@pytest.mark.parametrize("first_reads_the_product", [False, True])
def test_a_carried_value_a_later_pass_rewrote_differently_is_recomputed(
    first_reads_the_product: bool,
) -> None:
    # fuse_fma contracts ``p - 1.1`` into one fma only where nothing else
    # reads ``p``: when the first sweep also sums ``p``, its exp rounds the
    # product and the later sweep's does not, so the later sweep keeps its
    # own (the redo copy's, and the gmem policy's, value).
    def chain(lane: str, n: int) -> list[str]:
        # Each sweep's own names, as the codegen spells them.
        return [
            _vloop_load(lane),
            f"f_{n} = cutlass.Float32(w)",
            f"p_{n} = f_{n} * 1.1",
            f"d_{n} = p_{n} - 1.1",
            f"e = cute.math.exp(d_{n})",
        ]

    extra = ["acc2 = acc2 + p_1"] if first_reads_the_product else []
    code, later = _vloop_sweeps(
        [*chain("vec_lane_1", 1), _SUM, *extra],
        [*chain("vec_lane_2", 2), _STORE],
        fma=True,
    )
    assert ("_fuse_cache_" in later) is not first_reads_the_product
    assert ("cute.math.fma(" in later) is first_reads_the_product
    # The cut given up, the fragment goes too: its declaration and the first
    # sweep's writes.
    assert ("_fuse_cache_" in code) is not first_reads_the_product


def test_a_v_loop_load_through_a_fragment_an_alias_rewrites_is_recomputed() -> None:
    # ``q[0] = 8`` changes ``offs[0]`` between the sweeps: the later sweep's
    # loads read other elements than the first sweep's.
    def inner(lane: str) -> list[str]:
        return [_vloop_load(lane, "offs[0] + "), "e = cute.math.exp(w)"]

    for between, carried in (("", True), ("q = offs\nq[0] = 8\n", False)):
        code, later = _vloop_sweeps(
            [*inner("vec_lane_1"), _SUM],
            [*inner("vec_lane_2"), _STORE],
            between="q = offs\n" + between,
        )
        assert ("_fuse_cache_" in later) is carried
        assert ("cute.math.exp(" in later) is not carried


@pytest.mark.parametrize(
    ("between", "lane"),
    [("", "vec_lane_1"), ("saved = vec_lane_1\n", "saved")],
    ids=["directly", "through-a-name"],
)
def test_a_later_v_loop_reading_the_first_lane_is_recomputed(
    between: str, lane: str
) -> None:
    # After its loop ``vec_lane_1`` holds the first sweep's last lane: a later
    # V-loop loading at it, directly or through a name bound between the
    # sweeps, reads one element in every lane.
    code, later = _vloop_sweeps(
        [_vloop_load("vec_lane_1"), "e = cute.math.exp(w)", _SUM],
        [_vloop_load(lane), "e = cute.math.exp(w)", _STORE],
        between=between,
    )
    assert "_fuse_cache_" not in code
    assert "cute.math.exp(" in later


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("operand", ["load", "index"])
@pytest.mark.parametrize("sweep", ["first", "later"])
def test_a_v_loop_that_rebinds_its_lane_carries_nothing(
    sweep: str, operand: str, renamed: bool
) -> None:
    # A value's fragment slot is the lane its V-loop's variable names where
    # the slot is written or read.  Past ``vec_lane_1 = 3 - vec_lane_1`` (or a
    # ``vec_lane_1_next`` the final rename merges into it), the first sweep
    # would write each lane's exp, of an operand read at the lane before, to
    # the mirrored lane's slot, or the later one read the mirrored lane's.
    def inner(lane: str, rebinds: bool) -> list[str]:
        target = f"{lane}_next" if renamed else lane
        return [
            _vloop_load(lane)
            if operand == "load"
            else f"w = cutlass.Float32(roffset_7 + reduction_lane_7 * 4 + {lane})",
            *([f"{target} = 3 - {lane}"] if rebinds else []),
            "e = cute.math.exp(w)",
        ]

    code, later = _vloop_sweeps(
        [*inner("vec_lane_1", sweep == "first"), _SUM],
        [*inner("vec_lane_2", sweep == "later"), _STORE],
        rename_groups={f"vec_lane_{n}_next": f"vec_lane_{n}" for n in (1, 2)}
        if renamed
        else None,
    )
    assert "_fuse_cache_" not in code
    assert "cute.math.exp(" in later


def test_a_carried_value_with_one_of_two_alike_operands_rewritten_is_recomputed() -> (
    None
):
    # ``da`` and ``db`` compute one value in each sweep.  The first sweep also
    # sums both products, the later one only ``pb_2``, so fuse_fma contracts
    # ``da_2`` alone: the later exp rounds differently and keeps its own.
    def chain(lane: str, n: int) -> list[str]:
        return [
            _vloop_load(lane),
            f"f_{n} = cutlass.Float32(w)",
            f"pa_{n} = f_{n} * 1.1",
            f"pb_{n} = f_{n} * 1.1",
            f"da_{n} = pa_{n} - 1.1",
            f"db_{n} = pb_{n} - 1.1",
            f"e = cute.math.exp(da_{n} + db_{n})",
        ]

    code, later = _vloop_sweeps(
        [*chain("vec_lane_1", 1), _SUM, "acc2 = acc2 + pa_1", "acc3 = acc3 + pb_1"],
        [*chain("vec_lane_2", 2), _STORE, "acc4 = acc4 + pb_2"],
        fma=True,
    )
    # No fragment is left behind either, in the later sweep or the first.
    assert "_fuse_cache_" not in code
    assert "cute.math.fma(" in later


def test_a_carried_exp_whose_product_rounds_differently_is_recomputed() -> None:
    # Only the store sweep's ``x / 3 - 0.5`` contracts into an FMA, so its exp
    # rounds differently from the first sweep's and is not carried.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(third_softmax, backend="cute", static_shapes=True)
        bound = cpu_bind(kernel, (torch.empty(64, 512, dtype=torch.float16),))
        code = bound.to_code(
            helion.Config(**THIRD_SOFTMAX_CONFIG, cute_reduction_reloads=["register"])
        )
    assert "cute.math.fma(" in code
    assert not re.search(r"cutlass\.Float32\(_fuse_cache_\d+\[", code)


def _scalar_sweeps(first: list[str], later: list[str], after: str = "") -> str:
    """Two sweeps of 2 chunks over 8 lanes (``first`` and ``later`` are
    their lane-loop bodies), then ``after``, through the fusion and the cut."""

    def sweep(inner: list[str]) -> str:
        return (
            "for roffset_7 in range(0, 64, 32):\n"
            "    for reduction_lane_7 in range(8):\n"
            + "".join(f"        {line}\n" for line in inner)
        )

    body = fuse_two_pass_loads(
        ast.parse(sweep(first) + sweep(later) + after).body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    return ast.unparse(ast.Module(body=cut_carried_values(body), type_ignores=[]))


@pytest.mark.parametrize("read_after", [False, True])
def test_a_load_read_after_its_sweep_stays_a_cache_read(read_after: bool) -> None:
    # The later sweep's exp is carried, so only a read after the loop keeps
    # its load of ``w``: from the fragment, as without the carry.
    load = "w = (x.iterator + roffset_7 + reduction_lane_7).load()"
    store = "(out.iterator + roffset_7 + reduction_lane_7).store(e)"
    code = _scalar_sweeps(
        [load, "e = cute.math.exp(w)", "acc = acc + e"],
        [load, "e = cute.math.exp(w)", store],
        after="last = w\n" if read_after else "",
    )
    assert code.count("(x.iterator") == 1
    assert ("w = _fuse_cache_" in code) is read_after


def test_a_marked_statement_binding_no_plain_name_keeps_what_it_reads() -> None:
    # A later pass made a marked assignment augmented: it stays, and so does
    # the marked statement it reads.  A loop the cut empties keeps a pass.
    body = ast.parse(
        "x = cute.math.exp(w)\nacc += x\nfor i in range(4):\n    t = cute.math.exp(w)\n"
    ).body
    loop = cast("ast.For", body[2])
    for stmt in (body[0], body[1], loop.body[0]):
        setattr(stmt, FUSE_CARRY_DEAD_ATTR, True)
    code = ast.unparse(ast.Module(body=cut_carried_values(body), type_ignores=[]))
    assert code == "x = cute.math.exp(w)\nacc += x\nfor i in range(4):\n    pass"


def test_a_carried_value_takes_its_nearest_read() -> None:
    # Copies of a V-loop body share a read's name: each carried statement
    # takes the read right before it.
    block = ast.parse(
        "c = _fuse_cache_0[0]\n"
        "e = cute.math.exp(w)\n"
        "c = _fuse_cache_0[1]\n"
        "e = cute.math.exp(w)\n"
    ).body
    for read in (block[0], block[2]):
        setattr(read, FUSE_CARRIED_READ_ATTR, True)
    for stmt in (block[1], block[3]):
        setattr(stmt, FUSE_CARRY_ATTR, "c")
    _read_carried_values(block, {id(block[3])})
    assert ast.unparse(ast.Module(body=block, type_ignores=[])) == (
        "c = _fuse_cache_0[0]\ne = cute.math.exp(w)\ne = _fuse_cache_0[1]"
    )


@pytest.mark.parametrize(("trips", "carried"), [(256, True), (512, False)])
def test_a_carried_v_loop_fragment_stays_within_the_register_cap(
    trips: int, carried: bool
) -> None:
    # A value of a 4-lane V-loop takes a slot per lane and chunk: 1024 slots
    # at 256 chunks, as many as a register fragment holds, 2048 at 512.
    code, later = _vloop_sweeps(
        [_vloop_load("vec_lane_1"), "e = cute.math.exp(w)", _SUM],
        [_vloop_load("vec_lane_2"), "e = cute.math.exp(w)", _STORE],
        trips=trips,
        lanes=1,
    )
    assert ("_fuse_cache_" in later) is carried
    assert all(
        int(slots) <= 1024 for slots in re.findall(r"make_rmem_tensor\((\d+)", code)
    )


def test_carried_value_of_a_python_scalar_product_is_rounded_once() -> None:
    code, *sweeps = _sweeps("scaled_softmax", "register")
    # The fragment holds the product as the kernel rounds it to bf16, and
    # the reload reads that value back unchanged.
    assert re.search(r"e = cutlass\.BFloat16\(v_\d+\)", sweeps[1])
    assert re.search(r"= cutlass\.BFloat16\(_fuse_cache_\d+\[", sweeps[2])
    assert "cutlass.Float32(_fuse_cache_" not in code


def test_operand_promotion_matches_the_dsl() -> None:
    # The DSL's operators on Python-valued numerics report the type they
    # compute in; a Python scalar operand converts to a DSL value first (an
    # int beyond Int32 to Int64).
    cutlass = pytest.importorskip("cutlass")
    names = [
        name.removeprefix("cutlass.")
        for name in sorted(CACHE_DTYPES)
        if name != "cutlass.Boolean"
    ]
    for left in names:
        value = getattr(cutlass, left)(1)
        operands = [
            *((f"cutlass.{right}", getattr(cutlass, right)(1)) for right in names),
            *((_scalar_dtype(scalar), scalar) for scalar in (2.0, 5, 3449720151)),
        ]
        for dtype, operand in operands:
            assert dtype is not None
            expected = f"cutlass.{type(value + operand).__name__}"
            assert _promoted(f"cutlass.{left}", dtype) == expected, (left, operand)


def test_lane_invariant_load_takes_one_slot() -> None:
    code, *sweeps = _sweeps("scaled_layernorm", "register")
    # The row keeps a slot per lane; the scale factor takes one.
    assert "cute.make_rmem_tensor(8, cutlass.BFloat16)" in code
    assert re.search(r"= _fuse_cache_\d+\[\(synthetic_lane_\d+ - 0\) // 1\]", sweeps[1])
    assert "cute.make_rmem_tensor(1, cutlass.Float32)" in code
    assert re.search(r"_fuse_cache_\d+\[0\] = ", sweeps[0])
    assert re.search(r"= _fuse_cache_\d+\[0\]", sweeps[1])


def test_raw_byte_loads_cache_in_the_dtype_they_name() -> None:
    """A rolled fp8 row loads raw bytes (``cute.arch.load(ptr, cutlass.Uint8)``);
    the consume sweep reads the reduce sweep's from a ``Uint8`` fragment
    instead of loading them again (the fp8 dtype would convert them)."""
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(center_fp8_row, backend="cute", static_shapes=True)
        bound = cpu_bind(kernel, (torch.empty(8, 4096, dtype=torch.float8_e4m3fn),))
        code = bound.to_code(center_fp8_row_config(bound.config_spec))
    # Four 1024-element chunks of eight lanes per thread.
    assert "cute.make_rmem_tensor(32, cutlass.Uint8)" in code
    assert code.count("cute.arch.load(x.iterator") == 1


def _fp8_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty([rows, n], dtype=torch.bfloat16, device=x.device)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        out[tile_r, :] = (e / e.sum(-1, keepdim=True)).to(out.dtype)
    return out


def test_raw_byte_lane_loads_take_the_dtype_they_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At V = 8 an fp8 row loads raw bytes in its V-loops
    (``cute.arch.load(ptr, cutlass.Uint8)``).  The numbering of the carried
    values types each such load as the Uint8 it names, as a cached load's
    fragment does, not as the tensor's fp8."""
    seen: list[dict[str, str]] = []
    plan = _CuteFuseTwoPassLoads._plan_sweep_values

    def recording(
        self: _CuteFuseTwoPassLoads, *args: object, **kwargs: object
    ) -> object:
        seen.append(dict(cast("dict[str, str]", kwargs["load_dtypes"])))
        return plan(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_plan_sweep_values", recording)
    kernel = helion.kernel(_fp8_softmax, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (torch.empty(8, 4096, dtype=torch.float8_e4m3fn),))
        spec = bound.config_spec
        (rdim,) = spec.reduction_loops.valid_block_ids()
        code = bound.to_code(
            spec.normalized_config(
                helion.Config(
                    block_sizes=[1],
                    reduction_loops=[1024],
                    num_threads=[
                        128 if block_id == rdim else 1
                        for block_id in spec.num_threads.valid_block_ids()
                    ],
                    cute_vector_widths=[
                        8 if block_id == rdim else 1
                        for block_id in spec.cute_vector_widths.valid_block_ids()
                    ],
                    cute_reduction_reloads=["register"],
                )
            )
        )
    assert "cutlass.range_constexpr(8)" in code
    raw = {
        key: dtype
        for dtypes in seen
        for key, dtype in dtypes.items()
        if key.endswith(", cutlass.Uint8)")
    }
    assert raw
    assert set(raw.values()) == {"cutlass.Uint8"}


@pytest.mark.parametrize(
    ("expr", "kind", "dtype"),
    [
        # A plain scalar load caches in its tensor's dtype (looked up later).
        ("(x.iterator + i).load()", "unmasked", None),
        # A scalar ``cute.arch.load`` names its dtype: a hinted fp32 load, a
        # byte-packed bool/int8/fp8 packet, an fp8 raw byte.
        (
            "cute.arch.load(x.iterator + i, cutlass.Float32, cop='cs')",
            "unmasked",
            "cutlass.Float32",
        ),
        (
            "cute.arch.load(x.iterator + i, cutlass.Uint16)",
            "unmasked",
            "cutlass.Uint16",
        ),
        ("cute.arch.load(x.iterator + i, cutlass.Uint8)", "unmasked", "cutlass.Uint8"),
        # A packet caches lane by lane in its element type.
        (
            "cute.arch.load(x.iterator + i, ir.VectorType.get([2], cutlass.Uint16.mlir_type))",
            "vec",
            None,
        ),
        # A vector whose width is no literal is neither: no cache.
        (
            "cute.arch.load(x.iterator + i, ir.VectorType.get([n], cutlass.Uint16.mlir_type))",
            None,
            None,
        ),
    ],
)
def test_scalar_arch_loads_cache_in_the_dtype_they_name(
    expr: str, kind: str | None, dtype: str | None
) -> None:
    node = ast.parse(expr, mode="eval").body
    assert _load_kind(node) == kind
    assert _arch_scalar_dtype(node) == dtype


@pytest.mark.parametrize(
    ("later", "fused"),
    [
        # The unhinted twin of a hinted fp32 load reads its cache.
        ("(x.iterator + lane).load()", True),
        ("cute.arch.load(x.iterator + lane, cutlass.Float32, cop='cs')", True),
        # A load naming another dtype at the same address is no reuse: a
        # ``Uint64`` from the ``Float32`` fragment would keep 4 of 8 bytes,
        # an ``Int32`` would convert the bits it reinterprets.
        ("cute.arch.load(x.iterator + lane, cutlass.Uint64)", False),
        ("cute.arch.load(x.iterator + lane, cutlass.Uint32)", False),
        ("cute.arch.load(x.iterator + lane, cutlass.Int32)", False),
        ("cute.arch.load(x.iterator + lane, cutlass.Float64)", False),
    ],
)
def test_a_later_load_reuses_a_cache_only_of_its_own_dtype(
    later: str, fused: bool
) -> None:
    code = _fused_twin(
        "cute.arch.load(x.iterator + lane, cutlass.Float32, cop='cs')",
        later,
        "cutlass.Float32",
    )
    assert ("after = _fuse_cache_" in code) is fused, code
    assert code.count("x.iterator + lane") == (1 if fused else 2), code


def test_a_vector_load_of_unknown_width_has_no_scalar_key() -> None:
    # ``ir.VectorType.get([n], ...)`` is a vector whatever ``n`` is: it keys
    # as no scalar load, so it cannot collide with the plain load's key.
    vector = "cute.arch.load(x.iterator + i, ir.VectorType.get([n], cutlass.Int16.mlir_type))"
    plain = "(x.iterator + i).load()"
    dtypes = {"x": "cutlass.Int16"}
    assert _scalar_load_key(ast.parse(vector, mode="eval").body, dtypes) is None
    assert (
        _scalar_load_key(ast.parse(plain, mode="eval").body, dtypes)
        == "__scalar_load__(x.iterator + i, cutlass.Int16)"
    )


def _fused_twin(before: str, later: str, dtype: str) -> str:
    """Two lane sweeps loading the same address of ``x`` (``dtype``),
    ``before`` in the first and ``later`` in the second, through the fusion."""
    body = ast.parse(
        f"""
for synthetic_lane_3 in range(8):
    before = {before}
for synthetic_lane_3 in range(8):
    after = {later}
"""
    ).body
    return ast.unparse(
        ast.Module(
            body=fuse_two_pass_loads(
                body, tensor_dtypes={"x": dtype}, reload_modes={3: "register"}
            ),
            type_ignores=[],
        )
    )


# A pure asm chain long enough to be worth carrying (16 instructions).
_CHAIN = (
    "_cute_inline_asm_elementwise(({value},), asm='{{\\n"
    + "mul.lo.u32 $0, $1, 3449720151;\\nadd.u32 $0, $0, 7;\\n" * 8
    + "}}', constraints='=r,r', dtype=cutlass.Uint32, is_pure=True)"
)


def _carried_chain_sweeps(
    first: list[ast.stmt], second: list[ast.stmt], between: str, **kwargs: object
) -> list[str]:
    body = [
        ast.parse("for synthetic_lane_7 in range(8):\n    pass").body[0],
        *ast.parse(between).body,
        ast.parse("for synthetic_lane_7 in range(8):\n    pass").body[0],
    ]
    assert isinstance(body[0], ast.For) and isinstance(body[-1], ast.For)
    body[0].body, body[-1].body = first, second
    result = fuse_two_pass_loads(body, reload_modes={7: "register"}, **kwargs)
    return [ast.unparse(node) for node in result if isinstance(node, ast.For)]


@pytest.mark.parametrize("prove_disjoint", (False, True))
def test_chain_over_a_load_a_store_may_change_is_recomputed(
    prove_disjoint: bool,
) -> None:
    def sweep(store: str) -> list[ast.stmt]:
        return ast.parse(
            f"""
word = (x.iterator + synthetic_lane_7).load()
mixed = {_CHAIN.format(value="word")}
{store}
"""
        ).body

    sweeps = _carried_chain_sweeps(
        sweep("acc = acc + mixed"),
        sweep("(out.iterator + synthetic_lane_7).store(mixed)"),
        "(y.iterator + 0).store(acc)",
        tensor_dtypes={
            "x": "cutlass.Uint32",
            "y": "cutlass.Uint32",
            "out": "cutlass.Uint32",
        },
        proven_disjoint_tensor_pairs=(
            {frozenset(("x", "y")), frozenset(("x", "out"))}
            if prove_disjoint
            else set()
        ),
    )
    # Only a proof that the store between the sweeps leaves ``x`` alone lets
    # the second sweep reuse the first sweep's chain.
    assert ("mul.lo.u32" in sweeps[1]) is not prove_disjoint
    assert ("mixed = cutlass.Uint32(_fuse_cache_" in sweeps[1]) is prove_disjoint


def _jump_sweeps(source: str) -> str:
    """``source`` through the fusion, its tensors ``x``, ``y``, ``z`` and
    ``out`` disjoint words."""
    tensors = ("x", "y", "z", "out")
    body = fuse_two_pass_loads(
        ast.parse(source).body,
        reload_modes={7: "register"},
        tensor_dtypes=dict.fromkeys(tensors, "cutlass.Uint32"),
        proven_disjoint_tensor_pairs={
            frozenset((a, b)) for a in tensors for b in tensors if a < b
        },
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_a_later_sweep_that_breaks_keeps_what_the_break_path_reads() -> None:
    # The liveness the cut deletes by lets a break fall through to
    # ``last = 0``: the load only the break path stores would look dead.
    code = _jump_sweeps(
        f"""
for synthetic_lane_7 in range(8):
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word")}
    acc = acc + mixed
for synthetic_lane_7 in range(8):
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word")}
    last = (z.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(mixed)
    if synthetic_lane_7 == 3:
        break
    last = cutlass.Uint32(0)
(out.iterator + 9).store(last)
"""
    )
    assert "last = (z.iterator + synthetic_lane_7).load()" in code


def test_a_break_after_the_group_keeps_what_follows_its_loop_reading() -> None:
    # After the enclosing loop breaks, the store reads the ``seed2`` the
    # statements between the sweeps loaded.
    code = _jump_sweeps(
        f"""
for outer in range(2):
    for synthetic_lane_7 in range(8):
        seed = (y.iterator + 0).load()
        word = (x.iterator + synthetic_lane_7).load()
        mixed = {_CHAIN.format(value="word + seed")}
        acc = acc + mixed
    seed2 = (y.iterator + 0).load()
    for synthetic_lane_7 in range(8):
        word = (x.iterator + synthetic_lane_7).load()
        mixed = {_CHAIN.format(value="word + seed2")}
        (out.iterator + synthetic_lane_7).store(mixed)
    if flag:
        break
    seed2 = cutlass.Uint32(0)
(out.iterator + 8).store(seed2)
"""
    )
    assert "seed2 = (y.iterator + 0).load()" in code


def test_tile_loop_sweeps_carry_their_loads_only() -> None:
    # Tile loops are no reduction sweeps: no reload knob could turn a costly
    # carry off, so the second sweep reads only the load from the fragment
    # and recomputes the chain.
    body = ast.parse(
        f"""
for offset_7 in range(8):
    word = (x.iterator + offset_7).load()
    mixed = {_CHAIN.format(value="word")}
    acc = acc + mixed
for offset_7 in range(8):
    word = (x.iterator + offset_7).load()
    mixed = {_CHAIN.format(value="word")}
    (out.iterator + offset_7).store(mixed)
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={"x": "cutlass.Uint32", "out": "cutlass.Uint32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    second = ast.unparse(result[-1])
    assert re.search(r"word = _fuse_cache_\d+\[", second)
    assert "mul.lo.u32" in second


# The store sweep reads the chain over ``word + seed2`` from the first sweep,
# where ``seed`` loaded the same element, so the group no longer needs the
# ``seed2`` between the sweeps.
_SEEDED_SWEEPS = f"""
for synthetic_lane_7 in range(8):
    seed = (y.iterator + 0).load()
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word + seed")}
    acc = acc + mixed
seed2 = (y.iterator + 0).load()
for synthetic_lane_7 in range(8):
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word + seed2")}
    (out.iterator + synthetic_lane_7).store(mixed)
"""


@pytest.mark.parametrize(
    ("head", "tail", "kept"),
    [
        (None, None, False),
        ("try:", "finally:", True),
        ("for outer in range(2):", "else:", True),
    ],
    ids=["alone", "try-finally", "for-else"],
)
def test_cut_keeps_a_name_the_rest_of_an_enclosing_statement_reads(
    head: str | None, tail: str | None, kept: bool
) -> None:
    # A ``finally`` or a loop's ``else`` runs after the group and still
    # reads ``seed2``.
    source = _SEEDED_SWEEPS
    if head is not None:
        source = (
            f"{head}\n{textwrap.indent(source, '    ')}"
            f"{tail}\n    (out.iterator + 8).store(seed2)\n"
        )
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        reload_modes={7: "register"},
        tensor_dtypes={
            "x": "cutlass.Uint32",
            "y": "cutlass.Uint32",
            "out": "cutlass.Uint32",
        },
        proven_disjoint_tensor_pairs={
            frozenset(("x", "out")),
            frozenset(("y", "out")),
        },
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "mixed = cutlass.Uint32(_fuse_cache_" in code
    assert ("seed2 = " in code) is kept


def test_statement_shared_by_two_sweeps_is_two_values() -> None:
    # A body may hold one statement object in two sweeps (the lane split
    # once emitted its sweeps that way).  The load below is not tracked (no
    # dtype for ``z``) and ``z`` changes between the sweeps, so the chain
    # over it must not be carried.
    load = ast.parse("word = (z.iterator + synthetic_lane_7).load()").body[0]
    chain = ast.parse(f"mixed = {_CHAIN.format(value='word')}").body
    sweeps = _carried_chain_sweeps(
        [load, *chain, *ast.parse("acc = acc + mixed").body],
        [
            load,
            *chain,
            *ast.parse("(out.iterator + synthetic_lane_7).store(mixed)").body,
        ],
        "(z.iterator + 0).store(acc)",
        tensor_dtypes={"out": "cutlass.Uint32"},
        proven_disjoint_tensor_pairs={frozenset(("z", "out"))},
    )
    assert "mul.lo.u32" in sweeps[1]
    assert "_fuse_cache_" not in sweeps[1]


def test_chain_over_a_load_rebound_between_the_sweeps_is_recomputed() -> None:
    # ``u`` between the sweeps loads ``x[t]`` for the ``t`` loaded after the
    # first sweep, not the first sweep's ``w``: the second sweep's chain
    # over it is not the first sweep's.
    body = ast.parse(
        f"""
t = (y.iterator + 0).load()
for synthetic_lane_7 in range(8):
    w = (x.iterator + t).load()
    mixed = {_CHAIN.format(value="w")}
    acc = acc + mixed
t = (y.iterator + 1).load()
u = (x.iterator + t).load()
for synthetic_lane_7 in range(8):
    mixed = {_CHAIN.format(value="u")}
    (out.iterator + synthetic_lane_7).store(mixed)
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={
            "x": "cutlass.Uint32",
            "y": "cutlass.Int32",
            "out": "cutlass.Uint32",
        },
        proven_disjoint_tensor_pairs={
            frozenset(("x", "out")),
            frozenset(("y", "out")),
        },
    )
    second = ast.unparse(result[-1])
    assert "mul.lo.u32" in second
    assert "_fuse_cache_" not in second


@pytest.mark.parametrize(
    ("store", "carried"),
    [("", True), ("q[0] = 1", False), ("    q[0] = 1", False)],
    ids=["no-store", "between", "in-first-loop"],
)
def test_chain_over_a_load_an_alias_store_changes_is_recomputed(
    store: str, carried: bool
) -> None:
    # ``q[0] = 1`` changes ``offs[0]`` through another name for the fragment,
    # between the sweeps or in the first one, so ``seed2`` need not be the
    # first sweep's ``seed``, and the second sweep's chain over it is
    # recomputed, with ``seed2`` loaded again.
    body = ast.parse(
        f"""
q = offs
for synthetic_lane_7 in range(8):
    seed = (y.iterator + offs[0]).load()
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word + seed")}
    acc = acc + mixed
{store}
seed2 = (y.iterator + offs[0]).load()
for synthetic_lane_7 in range(8):
    word = (x.iterator + synthetic_lane_7).load()
    mixed = {_CHAIN.format(value="word + seed2")}
    (out.iterator + synthetic_lane_7).store(mixed)
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={
            "x": "cutlass.Uint32",
            "y": "cutlass.Uint32",
            "out": "cutlass.Uint32",
        },
        proven_disjoint_tensor_pairs={
            frozenset(("x", "out")),
            frozenset(("y", "out")),
            frozenset(("x", "y")),
        },
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    second = ast.unparse(result[-1])
    assert ("mul.lo.u32" in second) is not carried
    assert ("seed2 = (y.iterator + offs[0]).load()" in code) is not carried


def test_name_the_final_rename_merges_is_carried_around_the_sweep() -> None:
    # ``a_next`` renames to ``a``, which so changes every iteration: the
    # second sweep's ``exp(a)`` is not the first sweep's last one.
    body = ast.parse(
        """
a = cutlass.Float32(0.0)
for synthetic_lane_7 in range(8):
    v = cute.math.exp(a)
    acc = acc + v
    a_next = a + 1.0
a = cutlass.Float32(0.0)
for synthetic_lane_7 in range(8):
    v = cute.math.exp(a)
    (out.iterator + synthetic_lane_7).store(v)
    a_next = a + 1.0
"""
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        rename_groups={"a_next": "a"},
        tensor_dtypes={"out": "cutlass.Float32"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "_fuse_cache_" not in code
    assert code.count("v = cute.math.exp(a)") == 2
    assert code.count("a = cutlass.Float32(0.0)") == 2


def test_name_the_lane_loop_rebinds_is_carried_from_the_prefix_value() -> None:
    def sweep(use: str) -> str:
        return f"""
for roffset_7 in range(0, 64, 32):
    t = cutlass.Uint32(0)
    for reduction_lane_7 in range(8):
        word = (x.iterator + roffset_7 + reduction_lane_7).load()
        mixed = {_CHAIN.format(value="t")}
        {use}
        t = mixed
"""

    body = ast.parse(
        sweep("acc = acc + mixed + word")
        + sweep("(out.iterator + roffset_7 + reduction_lane_7).store(mixed + word)")
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Uint32", "out": "cutlass.Uint32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    # ``t`` holds the prefix's value in the first lane iteration only, so the
    # chain over it is recomputed (the loads are still served from a fragment).
    second = ast.unparse(result[-1])
    assert "mul.lo.u32" in second
    assert "t = mixed" in second
    assert "word = _fuse_cache_" in second


@pytest.mark.parametrize("renamed", (False, True))
def test_load_through_a_carried_index_takes_no_single_slot(renamed: bool) -> None:
    # ``idx`` is carried from one lane to the next (directly, or as the
    # ``idx_next`` the final rename merges into it): ``x[idx]`` reads another
    # element in every iteration, not the ``x[0]`` of ``idx = 0``, and is not
    # fused.  ``w`` beside it reads its lane's element: a slot per lane.
    step = "idx_next = idx + 1" if renamed else "idx = idx + 1"
    sweep = f"""
idx = 0
for synthetic_lane_7 in range(8):
    v = (x.iterator + idx).load()
    w = (x.iterator + synthetic_lane_7).load()
    {{use}}
    {step}
"""
    body = ast.parse(
        sweep.format(use="acc = acc + v + w")
        + sweep.format(use="(out.iterator + synthetic_lane_7).store(v + w)")
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
        rename_groups={"idx_next": "idx"} if renamed else {},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "w = _fuse_cache_0[(synthetic_lane_7 - 0) // 1]" in code
    assert "cute.make_rmem_tensor(1," not in code
    assert code.count("v = (x.iterator + idx).load()") == 2


@pytest.mark.parametrize("renamed", (False, True))
def test_load_after_its_index_is_rebound_takes_no_single_slot(renamed: bool) -> None:
    # ``idx = 0`` stops holding once ``idx`` is rebound to ``idx + lane``
    # (directly, or as the ``idx_next`` the final rename merges into it):
    # ``x[idx]`` then reads another element in every lane, not ``x[0]``.
    # ``w`` beside it reads its lane's element: a slot per lane.
    step = (
        "idx_next = idx + synthetic_lane_7"
        if renamed
        else "idx = idx + synthetic_lane_7"
    )
    sweep = f"""
for synthetic_lane_7 in range(8):
    idx = 0
    {step}
    v = (x.iterator + idx).load()
    w = (x.iterator + synthetic_lane_7).load()
    {{use}}
"""
    body = ast.parse(
        sweep.format(use="acc = acc + v + w")
        + sweep.format(use="(out.iterator + synthetic_lane_7).store(v + w)")
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
        rename_groups={"idx_next": "idx"} if renamed else {},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "w = _fuse_cache_0[(synthetic_lane_7 - 0) // 1]" in code
    assert "cute.make_rmem_tensor(1," not in code


@pytest.mark.parametrize("renamed", (False, True))
def test_sweep_over_other_indices_recomputes_its_values(renamed: bool) -> None:
    # Equal range text, but ``start`` moves between the sweeps: the second
    # sweep's ``exp`` of its index is not the first sweep's.
    move = "start_next = start + 8" if renamed else "start = 8"
    sweep = """
for synthetic_lane_7 in range(start, start + 8):
    v = cute.math.exp(cutlass.Float32(synthetic_lane_7))
    {use}
"""
    body = ast.parse(
        "start = 0"
        + sweep.format(use="acc = acc + v")
        + move
        + sweep.format(use="(out.iterator + synthetic_lane_7).store(v)")
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={"out": "cutlass.Float32"},
        rename_groups={"start_next": "start"} if renamed else {},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "_fuse_cache_" not in code
    assert code.count("cute.math.exp(") == 2


def test_values_of_a_first_sweep_that_leaves_lanes_early_are_recomputed() -> None:
    # The first sweep leaves its lane loop after lane 0, so its ``exp`` ran
    # for lane 0 only: the second sweep's lanes recompute theirs.
    body = ast.parse(
        """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    v = cute.math.exp(cutlass.Float32(synthetic_lane_7))
    acc = acc + v
    if cutlass.const_expr(synthetic_lane_7 == 0):
        break
for synthetic_lane_7 in cutlass.range_constexpr(8):
    v = cute.math.exp(cutlass.Float32(synthetic_lane_7))
    (out.iterator + synthetic_lane_7).store(v)
"""
    ).body
    result = fuse_two_pass_loads(
        body, reload_modes={7: "register"}, tensor_dtypes={"out": "cutlass.Float32"}
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "_fuse_cache_" not in code
    assert code.count("cute.math.exp(") == 2


def test_carried_value_fragment_follows_a_constexpr_trip_count() -> None:
    # A rolled reduction over a symbolic extent: the value fragment holds the
    # exact trip count's slots, under the trace-time budget the size hint
    # admitted, and the original sweeps run beyond it.
    def sweep(use: str) -> str:
        return f"""
for roffset_7 in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(1024)):
    for reduction_lane_7 in range(8):
        word = (x.iterator + roffset_7 + reduction_lane_7).load()
        mixed = {_CHAIN.format(value="word")}
        {use}
"""

    body = ast.parse(
        sweep("acc = acc + mixed")
        + sweep("(out.iterator + roffset_7 + reduction_lane_7).store(mixed)")
    ).body
    result = fuse_two_pass_loads(
        body,
        reload_modes={7: "register"},
        tensor_dtypes={"x": "cutlass.Uint32", "out": "cutlass.Uint32"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
        dynamic_trip_counts={"roffset_7": (2, "_REDUCTION_TRIPS_7")},
    )
    cap = "cutlass.const_expr(_REDUCTION_TRIPS_7 * 8 <= 16)"
    declaration, group = [
        node
        for node in result
        if isinstance(node, ast.If) and ast.unparse(node.test) == cap
    ]
    assert [ast.unparse(statement) for statement in declaration.body] == [
        "_fuse_cache_0 = cute.make_rmem_tensor(_REDUCTION_TRIPS_7 * 8, cutlass.Uint32)"
    ]
    fast = [ast.unparse(statement) for statement in group.body]
    fallback = [ast.unparse(statement) for statement in group.orelse]
    assert ["mul.lo.u32" in sweep for sweep in fast] == [True, False]
    assert "mixed = cutlass.Uint32(_fuse_cache_0[" in fast[1]
    assert ["mul.lo.u32" in sweep for sweep in fallback] == [True, True]
    assert "_fuse_cache_" not in "".join(fallback)


def test_call_results_take_the_dsl_types() -> None:
    # ``cute.arch.fmax`` computes in Float32 whatever its operands are; a
    # ``cute.math`` function keeps its operand's type, except the predicates
    # (an ``i1``) and ``sincos`` (a pair), whose results fill no fragment.
    values = SweepValues(
        transcendentals=True,
        cache_dtypes={},
        binding_dtypes={},
        canonical=lambda name: name,
    )
    env = {"x": values.leaf("x", dtype="cutlass.BFloat16")}

    def dtype(source: str) -> str | None:
        number = values.expression(ast.parse(source, mode="eval").body, env)
        return values.values[number].dtype

    assert dtype("cute.arch.fmax(cute.math.exp(x), x)") == "cutlass.Float32"
    assert dtype("cute.arch.fmin(x, 0.0) * x") == "cutlass.Float32"
    assert dtype("cute.math.exp(x)") == "cutlass.BFloat16"
    assert dtype("cute.math.fma(x, x, x)") == "cutlass.BFloat16"
    assert dtype("cute.math.isnan(x)") is None
    assert dtype("cute.math.sincos(x)") is None
