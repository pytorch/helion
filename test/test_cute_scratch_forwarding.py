"""CPU codegen tests: CuTe serves a later sweep's reload of a scratch store
from registers.

With ``cute_scratch_forwarding``, ``fuse_two_pass_loads`` treats an unmasked
store into a kernel-local scratch tensor (``helion._compiler.scratch_tensors``)
as the first sweep's load of its element: a later sweep's load of that element
reads the stored value from the register cache, through an opaque move as wide
as the element (``_opaque_copy``), under the reload policy loads follow.  The
GPU twin is ``test_cute_scratch_forwarding_cuda.py``.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import _reloads_scratch
from helion._compiler.cute.fuse_two_pass_loads import _AnalysisCache
from helion._compiler.cute.fuse_two_pass_loads import _canonical_load_text
from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.device_function import DeviceFunction
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_grid_lane_jam_kernels import head_bias
from helion._testing.cute_grid_lane_jam_kernels import head_bias_softmax_bwd
from helion._testing.scratch_kernels import ROW_SWEEPS_CONFIG
from helion._testing.scratch_kernels import row_softmax_bwd
from helion._testing.scratch_kernels import row_softmax_bwd_keeping_workspace
from helion._testing.scratch_kernels import row_softmax_bwd_recomputing
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST

pytestmark = pytest.mark.cpu_only


def _code(
    fn: object,
    reloads: str,
    dtype: torch.dtype = torch.float32,
    *,
    forwarding: bool = True,
    rows: torch.Tensor | None = None,
    exact_division: bool = False,
    **knobs: object,
) -> str:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, cute_exact_division=exact_division
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        if rows is None:
            rows = torch.empty(64, 1024, dtype=torch.bfloat16)
        args = (
            rows,
            rows,
            torch.empty(64),
            torch.empty(64),
            torch.empty(1, dtype=dtype),
        )
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(
                    **ROW_SWEEPS_CONFIG,
                    cute_reduction_reloads=[reloads],
                    cute_scratch_forwarding=forwarding,
                    **knobs,
                )
            )
        )


def _workspace_loads(code: str) -> int:
    return sum(
        1
        for line in code.splitlines()
        if "probs_ws.iterator" in line and ".load()" in line
    )


@pytest.mark.parametrize(
    ("dtype", "move"),
    [
        (torch.float32, "asm='mov.b32 $0, $1;', constraints='=r,r'"),
        (torch.bfloat16, "asm='mov.b16 $0, $1;', constraints='=h,h'"),
        (torch.float64, "asm='mov.b64 $0, $1;', constraints='=l,l'"),
    ],
    ids=["fp32", "bf16", "fp64"],
)
def test_the_second_sweep_reads_the_stored_value_through_an_opaque_move(
    dtype: torch.dtype, move: str
) -> None:
    # One move per forwarded load, as wide as the workspace's element.
    code = _code(row_softmax_bwd, "register", dtype)
    assert _workspace_loads(code) == 0
    assert code.count(move) == 1


@pytest.mark.parametrize(
    ("dtype", "element", "move"),
    [
        (torch.float32, "Float32", "mov.b32"),
        (torch.bfloat16, "BFloat16", "mov.b16"),
    ],
    ids=["fp32", "bf16"],
)
@pytest.mark.parametrize("width", [2, 8])
def test_a_packet_store_is_forwarded_lane_by_lane(
    width: int, dtype: torch.dtype, element: str, move: str
) -> None:
    # Each thread's eight elements in packets of ``width``: the first sweep
    # flushes the workspace in packets of the values its V-loop's lanes
    # gather, the second loads it one lane at a time in its own V-loop.
    # Each lane caches its value next to its append and reads it back; the
    # unread flush goes.  Under ``cute_exact_division`` the exact division's
    # copy of the first sweep caches it too.
    for exact_division in (False, True):
        code = _code(
            row_softmax_bwd,
            "register",
            dtype,
            exact_division=exact_division,
            cute_vector_widths=[width, 1],
        )
        assert _workspace_loads(code) == 0
        assert "probs_ws.iterator" not in code
        assert code.count(f"asm='{move} $0, $1;'") == 1
        cache_writes = [
            line.strip()
            for line in code.splitlines()
            if line.strip().startswith("_fuse_cache_")
            and f"] = cutlass.{element}(v_" in line
        ]
        assert len(cache_writes) == 1 + exact_division
        assert all(
            f"* {width} + reduction_vec_lane_1]" in line for line in cache_writes
        )


@pytest.mark.skip(
    reason="needs the read-after-write barrier out of the lane loops (hoist dropped)"
)
def test_a_recomputed_exp_is_carried_beside_a_forwarded_packet() -> None:
    # T5's encoder shape in packets of eight: the second sweep reads the
    # product the first flushed and the exp the first computed, each from a
    # fragment of a slot per lane, instead of reloading the one and
    # recomputing the other.
    kernel = helion.kernel(
        row_softmax_bwd_recomputing, backend="cute", static_shapes=True
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        rows = torch.empty(64, 1024, dtype=torch.bfloat16)
        bound = cpu_bind(
            kernel,
            (rows, rows, torch.empty(64, 1024), torch.empty(64), torch.empty(64)),
        )
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(
                    **ROW_SWEEPS_CONFIG,
                    cute_reduction_reloads=["register"],
                    cute_scratch_forwarding=True,
                    cute_vector_widths=[8, 1],
                )
            )
        )
    assert "product_ws.iterator" not in code
    # The first sweep and the exact division's copies of both sweeps; the
    # second sweep itself reads it.
    assert code.count("cute.math.exp(") == 3


def test_gmem_reloads_keep_reading_a_packet_store() -> None:
    code = _code(row_softmax_bwd, "gmem", cute_vector_widths=[8, 1])
    assert _workspace_loads(code) == 1
    assert "mov.b" not in code


@pytest.mark.parametrize(("columns", "forwarded"), [(16384, True), (262144, False)])
def test_a_packet_fragment_stays_within_the_cap(columns: int, forwarded: bool) -> None:
    # Rows one element off alignment load in scalars, while the workspace
    # takes packets of eight: a slot per lane, 8 per sweep step, is 128 or
    # 2048 elements, which ``register`` caps at 1024.
    rows = torch.empty(4, columns + 1, dtype=torch.bfloat16)[:, 1:]
    code = _code(row_softmax_bwd, "register", rows=rows, cute_vector_widths=[8, 1])
    assert "VectorType.get(" not in code
    assert (_workspace_loads(code) == 0) is forwarded
    assert ("cute.make_rmem_tensor(128, cutlass.Float32)" in code) is forwarded


_LANE_INDEX = "rindex = base + cutlass.Int32(vl)"
_APPEND = "vals.append(cutlass.Float32(v).bitcast(cutlass.Uint32))"


@pytest.mark.parametrize(
    ("before", "body", "forwarded"),
    [
        (["vals = []"], [_LANE_INDEX, _APPEND], True),
        # Nothing empties the list here: it may hold earlier values.
        ([], [_LANE_INDEX, _APPEND], False),
        # A value appended before the loop's lands in lane 0's element.
        (["vals = []", "vals.append(y)"], [_LANE_INDEX, _APPEND], False),
        # Emptied in the loop: the flush writes the last lane's value alone.
        (["vals = []"], [_LANE_INDEX, "vals = []", _APPEND], False),
        # An expression, which the cache write would evaluate again.
        (
            ["vals = []"],
            [
                _LANE_INDEX,
                "vals.append(cutlass.Float32(v * y).bitcast(cutlass.Uint32))",
            ],
            False,
        ),
    ],
    ids=["emptied", "not-emptied", "appended-before", "emptied-in-loop", "expression"],
)
def test_a_packet_store_forwards_the_values_its_lanes_append(
    before: list[str], body: list[str], forwarded: bool
) -> None:
    container = ast.parse(
        "\n".join(
            [
                *before,
                "for vl in cutlass.range_constexpr(2):",
                *(f"    {line}" for line in body),
                "_cute_store_u32_vec(ws.iterator + cutlass.Int32(base), vals)",
            ]
        )
    ).body
    fuser = _CuteFuseTwoPassLoads(
        tensor_dtypes={"ws": "cutlass.Float32"}, forwardable_tensors={"ws"}
    )
    packet = fuser._forwardable_packet_store(container, len(container) - 1)
    assert (packet is not None) is forwarded


@pytest.mark.parametrize(
    ("loop", "body", "between", "pointer", "forwarded"),
    [
        ("range_constexpr(2)", [_LANE_INDEX, _APPEND], [], "base", True),
        # The base moves before the flush, in the loop or after it.
        (
            "range_constexpr(2)",
            [_LANE_INDEX, _APPEND, "base = base + 8"],
            [],
            "base",
            False,
        ),
        (
            "range_constexpr(2)",
            [_LANE_INDEX, _APPEND],
            ["base = base + 8"],
            "base",
            False,
        ),
        # Bound in the loop before the lane's index: the key adds each lane to
        # its own base, while the flush starts at the last lane's.
        (
            "range_constexpr(2)",
            ["base = off + cutlass.Int32(vl)", _LANE_INDEX, _APPEND],
            [],
            "base",
            False,
        ),
        # The pointer reads the loop's variable, the last lane at the flush.
        ("range_constexpr(2)", [_LANE_INDEX, _APPEND], [], "base + vl", False),
        # The loop rebinds its variable: its lanes run in another order.
        (
            "range_constexpr(2)",
            ["vl = 1 - vl", _LANE_INDEX, _APPEND],
            [],
            "base",
            False,
        ),
        # A lane that ends early appends nothing, and the flush packs fewer
        # values than the later sweep reads lanes.
        (
            "range_constexpr(2)",
            [_LANE_INDEX, "if vl == 0:", "    continue", _APPEND],
            [],
            "base",
            False,
        ),
        (
            "range_constexpr(2)",
            [_LANE_INDEX, _APPEND, "if vl == 0:", "    break"],
            [],
            "base",
            False,
        ),
        # Two elements per step of the base, while the packet's lanes are one
        # element apart.
        ("range_constexpr(2)", [_LANE_INDEX, _APPEND], [], "2 * base", False),
        ("range_constexpr(0)", [_LANE_INDEX, _APPEND], [], "base", False),
    ],
    ids=[
        "one-apart",
        "base-moved-in-loop",
        "base-moved-after",
        "base-moved-before-append",
        "lane-in-pointer",
        "rebound-lane",
        "continue",
        "break",
        "scaled",
        "no-lanes",
    ],
)
def test_a_packet_store_forwards_the_elements_its_lanes_store(
    loop: str, body: list[str], between: list[str], pointer: str, forwarded: bool
) -> None:
    container = ast.parse(
        "\n".join(
            [
                "vals = []",
                f"for vl in cutlass.{loop}:",
                *(f"    {line}" for line in body),
                *between,
                f"_cute_store_u32_vec(ws.iterator + cutlass.Int32({pointer}), vals)",
            ]
        )
    ).body
    fuser = _CuteFuseTwoPassLoads(
        tensor_dtypes={"ws": "cutlass.Float32"}, forwardable_tensors={"ws"}
    )
    packet = fuser._forwardable_packet_store(container, len(container) - 1)
    assert (packet is not None) is forwarded
    # Each lane's element is the base plus the lane, whatever names it.
    if packet is not None:
        assert ast.unparse(packet.load) == (
            "(ws.iterator + cutlass.Int32(base + cutlass.Int32(vl))).load()"
        )


def test_a_packet_base_spelled_like_an_attribute_is_substituted_as_a_name() -> None:
    # The base ``iterator`` is the pointer's one name ``iterator``; the
    # attribute of ``ws.iterator`` stays what it is.
    container = ast.parse(
        "\n".join(
            [
                "vals = []",
                "for vl in cutlass.range_constexpr(2):",
                "    rindex = iterator + cutlass.Int32(vl)",
                f"    {_APPEND}",
                "_cute_store_u32_vec(ws.iterator + cutlass.Int32(iterator), vals)",
            ]
        )
    ).body
    fuser = _CuteFuseTwoPassLoads(
        tensor_dtypes={"ws": "cutlass.Float32"}, forwardable_tensors={"ws"}
    )
    packet = fuser._forwardable_packet_store(container, len(container) - 1)
    assert packet is not None
    assert ast.unparse(packet.load) == (
        "(ws.iterator + cutlass.Int32(iterator + cutlass.Int32(vl))).load()"
    )


@pytest.mark.parametrize(
    ("element", "distinct"),
    [
        # Every sweep step, lane and V-loop lane its own element.
        ("roffset + lane * 4 + cutlass.Int32(vl)", True),
        # The same elements at every sweep step: a reload reads the last
        # step's values.
        ("lane * 4 + cutlass.Int32(vl)", False),
        # Lane 1's first element is lane 0's second.
        ("roffset + lane + cutlass.Int32(vl)", False),
        # Of two threads one element apart, thread 1's first element is
        # thread 0's second, a packet lane's.  Two apart, thread 1's
        # elements fill thread 0's gaps.
        (
            "roffset + lane * 4 + cutlass.Int32(vl) + cute.arch.thread_idx()[0]",
            False,
        ),
        (
            "roffset + lane * 4 + cutlass.Int32(vl) + 2 * cute.arch.thread_idx()[0]",
            True,
        ),
    ],
)
def test_a_packet_store_writes_distinct_elements(element: str, distinct: bool) -> None:
    # Eight sweep steps of two lanes, each flushing a packet of two, on two
    # threads.
    first_loop = ast.parse(
        f"""
for roffset in range(0, 64, 8):
    for lane in range(2):
        vals = []
        for vl in cutlass.range_constexpr(2):
            rindex = {element}
            vals.append(cutlass.Float32(v).bitcast(cutlass.Uint32))
"""
    ).body[0]
    assert isinstance(first_loop, ast.For)
    fuser = _CuteFuseTwoPassLoads(
        thread_block_dims=(2, 1, 1), tensor_dtypes={"ws": "cutlass.Float32"}
    )
    assert (
        fuser._stores_distinct_elements(
            f"load(ws.iterator + ({element}))",
            first_loop,
            ast.Constant(8),
            8,
            16,
            vector_lane=("vl", 2),
        )
        is distinct
    )


@pytest.mark.parametrize(
    ("between", "inner", "lane", "matched"),
    [
        ([], [], "vl2", True),
        # After its loop ``vl`` holds the first sweep's last lane, one element
        # for every lane, read directly or through a name bound in between.
        ([], [], "vl", False),
        (["saved = vl"], [], "saved", False),
        # The loop rebinds its variable: its lanes run in another order.
        ([], ["vl2 = 1 - vl2"], "vl2", False),
    ],
    ids=["own-lane", "first-lane", "first-lane-through-a-name", "rebound-lane"],
)
def test_a_later_v_loop_reads_a_packet_only_at_its_own_lanes(
    between: list[str], inner: list[str], lane: str, matched: bool
) -> None:
    first = ast.parse(
        "\n".join(
            [
                "vals = []",
                "for vl in cutlass.range_constexpr(2):",
                f"    {_LANE_INDEX}",
                f"    {_APPEND}",
                "_cute_store_u32_vec(ws.iterator + cutlass.Int32(base), vals)",
            ]
        )
    ).body
    later = ast.parse(
        "\n".join(
            [
                *between,
                "for vl2 in cutlass.range_constexpr(2):",
                *(f"    {line}" for line in inner),
                f"    rindex2 = base + cutlass.Int32({lane})",
                "    w = (ws.iterator + cutlass.Int32(rindex2)).load()",
            ]
        )
    ).body
    fuser = _CuteFuseTwoPassLoads(
        tensor_dtypes={"ws": "cutlass.Float32"}, forwardable_tensors={"ws"}
    )
    packet = fuser._forwardable_packet_store(first, len(first) - 1)
    assert packet is not None
    # The packet's key as the fusion spells it: the element of lane ``vl``.
    lane_index = cast("ast.Assign", ast.parse(_LANE_INDEX).body[0])
    key = _canonical_load_text(
        packet.load,
        {},
        {"rindex": lane_index.value},
        tensor_dtypes={"ws": "cutlass.Float32"},
    )
    matches = fuser._packet_lane_matches(
        later, {key: packet}, {}, {}, analysis=_AnalysisCache(fuser._canonical)
    )
    assert bool(matches) is matched


def test_the_knob_off_keeps_the_reload() -> None:
    code = _code(row_softmax_bwd, "register", forwarding=False)
    assert _workspace_loads(code) == 1
    assert "mov.b" not in code


def test_gmem_reloads_keep_reading_the_workspace() -> None:
    code = _code(row_softmax_bwd, "gmem")
    assert _workspace_loads(code) == 1
    assert "mov.b" not in code


def test_a_returned_workspace_is_reloaded() -> None:
    # The caller reads it: not kernel-local scratch.
    code = _code(row_softmax_bwd_keeping_workspace, "register")
    assert _workspace_loads(code) == 1
    assert "mov.b" not in code


def test_a_placeholder_argument_is_reloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # The kernel may also reach a placeholder argument's memory through
    # objects the host wrapper builds (TMA descriptors), under other names:
    # not kernel-local scratch.
    init = DeviceFunction.__init__

    def placeholder_workspace(
        self: DeviceFunction, name: str, config: helion.Config, codegen: GenerateAST
    ) -> None:
        init(self, name, config, codegen)
        self.placeholder_args.add("probs_ws")

    monkeypatch.setattr(DeviceFunction, "__init__", placeholder_workspace)
    code = _code(row_softmax_bwd, "register")
    assert _workspace_loads(code) == 1
    assert "mov.b" not in code


def row_softmax_bwd_accumulating(
    grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    workspace_like: torch.Tensor,
) -> torch.Tensor:
    """``row_softmax_bwd`` whose first sweep adds the probabilities to the
    workspace (``+=``): it loads each element and then stores to it."""
    rows, n = logits.shape
    probs_ws = torch.empty([rows, n], dtype=workspace_like.dtype, device=logits.device)
    out = torch.empty_like(logits)
    for tile_r in hl.tile(rows):
        probs = (
            torch.exp(logits[tile_r, :].to(torch.float32) - row_max[tile_r][:, None])
            / row_sum[tile_r][:, None]
        )
        probs_ws[tile_r, :] += probs
        g = grad[tile_r, :].to(torch.float32)
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_r, :] = (probs_ws[tile_r, :].to(torch.float32) * (g - total)).to(
            out.dtype
        )
    return out


def test_a_workspace_loaded_and_stored_in_one_sweep_forwards_the_store() -> None:
    # The first sweep's load and store of an element share a key; the load
    # stays (it reads what was there) and the second sweep reads what the
    # store wrote, bound to a name once.
    code = _code(row_softmax_bwd_accumulating, "register")
    assert _workspace_loads(code) == 1
    assert code.count("asm='mov.b32 $0, $1;'") == 1
    # Only the store writes its cache slot: no later sweep reads the load's
    # value from it.
    (cache,) = re.findall(
        r"(_fuse_cache_\d+)\[[^\]]*\] = cutlass\.\w+\(_fuse_stored_", code
    )
    assert len(re.findall(rf"{cache}\[[^\]]*\] = ", code)) == 1


def _jam_seeds(fn: object, *args: torch.Tensor) -> list[dict[str, object]]:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return [
            seed.config
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_grid_lane_jam")
        ]


def test_a_reloaded_workspace_seeds_both_settings_of_the_knob() -> None:
    # T5's softmax-dropout backward shape: the jam seed leaves the knob off
    # and its forwarding twin turns it on, so the autotuner picks per kernel.
    rows = torch.empty(16, 4, 512, dtype=torch.bfloat16)
    seeds = _jam_seeds(
        head_bias_softmax_bwd,
        rows,
        rows,
        head_bias(8, 4, 512, "cpu"),
        torch.empty(16, 4),
        torch.empty(16, 4),
    )
    off = [seed for seed in seeds if not seed.get("cute_scratch_forwarding")]
    on = [seed for seed in seeds if seed.get("cute_scratch_forwarding")]
    assert off and len(on) == len(off)
    assert [
        {key: value for key, value in seed.items() if key != "cute_scratch_forwarding"}
        for seed in on
    ] == off


def test_gmem_reloads_seed_no_forwarding_twin() -> None:
    # Rows past the jam seed's register budget reload from memory, where
    # nothing is forwarded: the twin would compile the jam seed's code again.
    rows = torch.empty(16, 4, 4096, dtype=torch.bfloat16)
    seeds = _jam_seeds(
        head_bias_softmax_bwd,
        rows,
        rows,
        head_bias(8, 4, 4096, "cpu"),
        torch.empty(16, 4),
        torch.empty(16, 4),
    )
    assert seeds and all("gmem" in seed["cute_reduction_reloads"] for seed in seeds)
    assert not any(seed.get("cute_scratch_forwarding") for seed in seeds)


def test_a_masked_chunk_seeds_no_forwarding_twin() -> None:
    # 600 keys in the jam seed's 1024-key chunk: its stores are masked, and
    # a masked store is not forwarded.
    rows = torch.empty(16, 4, 600, dtype=torch.bfloat16)
    seeds = _jam_seeds(
        head_bias_softmax_bwd,
        rows,
        rows,
        head_bias(8, 4, 600, "cpu"),
        torch.empty(16, 4),
        torch.empty(16, 4),
    )
    assert seeds and all("register" in seed["cute_reduction_reloads"] for seed in seeds)
    assert not any(seed.get("cute_scratch_forwarding") for seed in seeds)


def head_bias_softmax_bwd_kept(
    grad: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """``head_bias_softmax_bwd`` whose gradient both sweeps read through a
    mask: the second sweep reads two of the first sweep's loads again."""
    bh, q_len, k_len = scores.shape
    heads = bias.size(0)
    probs_ws = torch.empty(
        [bh, q_len, k_len], dtype=torch.float32, device=scores.device
    )
    out = torch.empty_like(scores)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        logits = scores[tile_bh, tile_q, :].to(torch.float32) + bias[head, tile_q, :]
        probs = (
            torch.exp(logits - row_max[tile_bh, tile_q][:, :, None])
            / row_sum[tile_bh, tile_q][:, :, None]
        )
        probs_ws[tile_bh, tile_q, :] = probs
        g = grad[tile_bh, tile_q, :].to(torch.float32) * keep[tile_bh, tile_q, :]
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (probs_ws[tile_bh, tile_q, :] * (g - total)).to(
            out.dtype
        )
    return out


def test_a_twin_past_the_jams_cache_budget_is_not_seeded() -> None:
    # 1024 keys on 128 threads: each of four lanes caches the gradient and
    # the mask, eight elements each, the jam's budget of 64 together.  The
    # forwarded workspace's cache would take the lanes' copies to 96, which
    # the jam declines: the twin would time a kernel without the jam.
    rows = torch.empty(16, 4, 1024, dtype=torch.bfloat16)
    seeds = _jam_seeds(
        head_bias_softmax_bwd_kept,
        rows,
        rows,
        head_bias(8, 4, 1024, "cpu"),
        torch.empty(16, 4),
        torch.empty(16, 4),
        torch.empty(16, 4, 1024),
    )
    assert seeds and all("register" in seed["cute_reduction_reloads"] for seed in seeds)
    assert not any(seed.get("cute_scratch_forwarding") for seed in seeds)


def test_the_knob_is_searchable_at_every_extent() -> None:
    # The twin seeds the knob at 512 keys and not at 4096, past the jam
    # seed's register budget; the kernel's structure opens it at both, so the
    # two shapes keep one search space (``autotune_multi`` compares their
    # ``structural_fingerprint``).
    fingerprints = set()
    for keys in (512, 4096):
        rows = torch.empty(16, 4, keys, dtype=torch.bfloat16)
        kernel = helion.kernel(
            head_bias_softmax_bwd, backend="cute", static_shapes=True
        )
        args = (
            rows,
            rows,
            head_bias(8, 4, keys, "cpu"),
            torch.empty(16, 4),
            torch.empty(16, 4),
        )
        with cpu_render_target(reduction_smem_budget_bytes=232448):
            bound = cpu_bind(kernel, args)
        fingerprints.add(bound.config_spec.structural_fingerprint())
    assert len(fingerprints) == 1


def row_softmax_bwd_unread_accumulation(
    grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    workspace_like: torch.Tensor,
) -> torch.Tensor:
    """``row_softmax_bwd_accumulating`` whose second sweep does not read the
    workspace back: its only load comes before its store."""
    rows, n = logits.shape
    probs_ws = torch.empty([rows, n], dtype=workspace_like.dtype, device=logits.device)
    out = torch.empty_like(logits)
    for tile_r in hl.tile(rows):
        probs = (
            torch.exp(logits[tile_r, :].to(torch.float32) - row_max[tile_r][:, None])
            / row_sum[tile_r][:, None]
        )
        probs_ws[tile_r, :] += probs
        g = grad[tile_r, :].to(torch.float32)
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_r, :] = (probs * (g - total)).to(out.dtype)
    return out


@pytest.mark.parametrize(
    ("fn", "workspace", "reloaded"),
    [
        (row_softmax_bwd, torch.float32, True),
        (row_softmax_bwd_accumulating, torch.float32, True),
        (row_softmax_bwd_unread_accumulation, torch.float32, False),
        (row_softmax_bwd, torch.int8, False),
        (row_softmax_bwd, torch.uint16, False),
        (row_softmax_bwd, torch.uint32, False),
    ],
    ids=["stored", "accumulated", "accumulated-unread", "int8", "uint16", "uint32"],
)
def test_the_forwarding_twin_needs_a_forwardable_workspace(
    fn: object, workspace: torch.dtype, reloaded: bool
) -> None:
    # A load before the workspace's only store reads nothing a store
    # forwards, nor does a load of bytes (no opaque move is that narrow) or
    # of unsigned integers (``_OPAQUE_MOVES``): the twin would compile and
    # time the jam seed's code again.
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    rows = torch.empty(64, 1024, dtype=torch.bfloat16)
    args = (
        rows,
        rows,
        torch.empty(64),
        torch.empty(64),
        torch.empty(1, dtype=workspace),
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
    assert _reloads_scratch(bound.env, bound.host_function.device_ir) is reloaded


def test_a_forwarded_store_keeps_its_source_line() -> None:
    # The store whose value the rewrite binds to a name stays the original
    # statement: it keeps its source line, and the access regions later
    # passes read.
    lines = _code(row_softmax_bwd_accumulating, "register").splitlines()
    (index,) = [
        i
        for i, line in enumerate(lines)
        if "probs_ws.iterator" in line and ".store(_fuse_stored_" in line
    ]
    assert lines[index - 1].strip().endswith("probs_ws[tile_r, :] += probs")
