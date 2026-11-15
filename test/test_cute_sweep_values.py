"""Later reduction sweeps reuse the first sweep's expensive pure values.

A persistent reduction runs one lane loop per sweep and re-emits the row's
elementwise chain in each.  ``fuse_two_pass_loads`` (with ``sweep_values``)
reads a later sweep's copy of the first sweep's pure inline-asm results, the
Philox chain of a word0 ``hl.rand`` draw, from a per-thread fragment, and with
the explicit ``register`` reload policy its transcendentals too; the ``gmem``
policy still recomputes them.  A lane-invariant load takes one fragment slot.
The kernels render on CPU; ``test_cute_fuse_two_pass_loads.py`` checks on the
GPU that the carried masks match the recomputed ones bit for bit.
"""

from __future__ import annotations

import ast
import functools
import re
import textwrap

import pytest
import torch

import helion
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._compiler.cute.sweep_values import CACHE_DTYPES
from helion._compiler.cute.sweep_values import SweepValues
from helion._compiler.cute.sweep_values import _promoted
from helion._compiler.cute.sweep_values import _scalar_dtype
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_sweep_kernels import dropout_layernorm
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
def _sweeps(name: str, reload: str, **options: object) -> tuple[str, ...]:
    """The source of each top-level loop of ``name``'s kernel, rendered once
    per reload policy (and further config ``options``): one persistent sweep
    per loop."""
    fn, args, settings = _KERNELS[name]
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        kernel = helion.kernel(fn, backend="cute", static_shapes=True, **settings)
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
    # The carried read stays a provable fp32 dividend, so the normalize
    # sweep keeps the exact division's fast path.
    assert "_cute_exact_div_fast_f32" in sweeps[2]


def test_auto_policy_recomputes_a_transcendental() -> None:
    _, *sweeps = _sweeps("softmax", "auto")
    assert sweeps[1].count("cute.math.exp(") == 1
    assert sweeps[2].count("cute.math.exp(") == 1


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
