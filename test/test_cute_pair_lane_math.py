"""Adjacent V-lanes' exps, exact divisions and fp32 arithmetic paired into
packed fp32 ops.

``helion/_compiler/cute/pair_lane_math.py`` (the pass and why each lane
stays bit for bit what the scalar code computes) with its device helpers
``exp_f32x2_parts`` and ``exact_div_fast_f32x2`` (``device/ptx.py``).

CPU: the pass on AST snippets (which lane ``b`` statements move up to a
pair, which keep their order, what keeps a pair scalar) and renders of a
softmax-statistics row kernel, paired in both sweeps on sm_100 and scalar
under ``fast_math`` or before sm_100.  GPU (``cuda``): the paired kernels
against the same kernels with the pass switched off, bit for bit: an
elementwise ``exp`` over every fp32 exponent with random significands, the
specials and the subnormal results, and the softmax statistics on rows with
large, tiny and -inf entries.
"""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute import pair_lane_math
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import with_exact_division
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

EXP_PAIR = "_cute_exp_f32x2_parts("
DIV_PAIR = "_cute_exact_div_fast_f32x2("
MUL_PAIR = "cute.arch.mul_packed_f32x2("
ADD_PAIR = "cute.arch.add_packed_f32x2("
SUB_PAIR = "cute.arch.sub_packed_f32x2("


def _run(
    source: str,
    *,
    fast_math: bool = False,
    capability: tuple[int, int] | None = (10, 0),
    tensor_dtypes: dict[str, str] | None = None,
    constants: dict[str, int] | None = None,
    parameters: frozenset[str] | None = None,
) -> str:
    body = ast.parse(textwrap.dedent(source)).body
    body = pair_lane_math.pair_lane_math(
        body,
        fast_math=fast_math,
        target_device_capability=capability,
        tensor_dtypes=tensor_dtypes,
        constants=constants,
        parameters=parameters,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _same(source: str) -> str:
    return ast.unparse(ast.parse(textwrap.dedent(source)))


# --------------------------------------------------------------------------
# The pass on AST snippets
# --------------------------------------------------------------------------

_EXP_SUM = """\
acc = cutlass.Float32(0)
for v in cutlass.range_constexpr(4):
    x = cutlass.Float32(frag[v]) - m
    e = cute.math.exp(cutlass.Float32(x))
    acc = acc + e
total = acc
"""


def test_exp_lanes_pair_and_keep_the_running_sum_in_lane_order() -> None:
    expected = """\
        acc = cutlass.Float32(0)
        for v_pair in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[2 * v_pair]) - m
            x_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(x), cutlass.Float32(x_b))
            e = e_pa * e_sa
            acc = acc + e
            e_b = e_pb * e_sb
            acc = acc + e_b
        total = acc
        """
    assert _run(_EXP_SUM) == _same(expected)


@pytest.mark.parametrize(
    ("fast_math", "capability"),
    [(True, (10, 0)), (False, (9, 0)), (False, None)],
    ids=["fast_math", "sm90", "unknown_target"],
)
def test_fast_math_and_targets_without_packed_fp32_stay_scalar(
    fast_math: bool, capability: tuple[int, int] | None
) -> None:
    assert _run(_EXP_SUM, fast_math=fast_math, capability=capability) == _same(_EXP_SUM)


@pytest.mark.parametrize(
    "loop",
    [
        "for v in cutlass.range_constexpr(3):",
        "for v in range(4):",
        "for v in cutlass.range_constexpr(0, 4, 1):",
    ],
    ids=["odd_extent", "lane_loop", "three_argument_range"],
)
def test_only_even_constexpr_v_loops_pair(loop: str) -> None:
    source = _EXP_SUM.replace("for v in cutlass.range_constexpr(4):", loop)
    assert _run(source) == _same(source)


def test_an_exp_folded_into_a_sum_reuses_the_assignment_beside_it() -> None:
    # The reduction folds keep the dead ``e = exp(d)`` beside ``acc + exp(d)``:
    # the fold reads the paired ``e`` instead of a second exp.
    source = """\
        for v in cutlass.range_constexpr(2):
            d = cutlass.Float32(frag[v]) - m
            e = cute.math.exp(cutlass.Float32(d))
            acc = acc + cutlass.Float32(cute.math.exp(cutlass.Float32(d)))
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            d = cutlass.Float32(frag[2 * v_pair]) - m
            d_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(d), cutlass.Float32(d_b))
            e = e_pa * e_sa
            acc = acc + cutlass.Float32(e)
            e_b = e_pb * e_sb
            acc = acc + cutlass.Float32(e_b)
        """
    assert _run(source) == _same(expected)


def test_a_nested_exp_gets_its_own_assignment() -> None:
    source = """\
        for v in cutlass.range_constexpr(2):
            vals.append(cutlass.Float32(cute.math.exp(cutlass.Float32(frag[v]))).bitcast(cutlass.Uint32))
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            _helion_paired_call_pa, _helion_paired_call_pb, _helion_paired_call_sa, _helion_paired_call_sb = _cute_exp_f32x2_parts(cutlass.Float32(frag[2 * v_pair]), cutlass.Float32(frag[2 * v_pair + 1]))
            _helion_paired_call = _helion_paired_call_pa * _helion_paired_call_sa
            vals.append(cutlass.Float32(_helion_paired_call).bitcast(cutlass.Uint32))
            _helion_paired_call_b = _helion_paired_call_pb * _helion_paired_call_sb
            vals.append(cutlass.Float32(_helion_paired_call_b).bitcast(cutlass.Uint32))
        """
    assert _run(source) == _same(expected)


def test_an_exp_assigned_over_its_operand_is_not_reused() -> None:
    # ``y = exp(y)`` holds the exp of the earlier ``y``, and the later
    # ``exp(y)`` reads the new one: it gets an assignment of its own (the
    # pair of ``z`` rewrites the loop).
    source = """\
        c = cutlass.Float32(cp)
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        out = cute.make_rmem_tensor(2, cutlass.Float32)
        for v in cutlass.range_constexpr(2):
            y = cute.math.exp(cutlass.Float32(y))
            z = cutlass.Float32(frag[v]) + c
            out[v] = cutlass.Float32(cute.math.exp(cutlass.Float32(y))) + z
        """
    code = _run(source)
    assert "z, z_b = " + ADD_PAIR in code, code
    assert "_helion_paired_call = cute.math.exp(cutlass.Float32(y))" in code, code
    assert "_helion_paired_call_b = cute.math.exp(cutlass.Float32(y))" in code, code


def test_conditional_arms_and_nested_scopes_are_not_hoisted() -> None:
    # Hoisting an exp out of a conditional arm would evaluate it always, out
    # of a lambda would read its parameter outside it.
    source = """\
        for v in cutlass.range_constexpr(2):
            y = cute.math.exp(cutlass.Float32(frag[v])) if flag else cutlass.Float32(0)
            f = lambda q: cute.math.exp(cutlass.Float32(q))
            out[v] = y + f(frag[v])
        """
    assert _run(source) == _same(source)


def test_exact_divisions_by_an_invariant_divisor_pair() -> None:
    source = """\
        for v in cutlass.range_constexpr(2):
            e = cutlass.Float32(frag[v])
            q = cutlass.Float32(_cute_exact_div_fast_f32(e, s, y))
            lo = cute.arch.fmin(lo, cute.math.abs(e), nan=True)
            out[v] = q
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            e = cutlass.Float32(frag[2 * v_pair])
            e_b = cutlass.Float32(frag[2 * v_pair + 1])
            q, q_b = _cute_exact_div_fast_f32x2(e, e_b, s, y)
            lo = cute.arch.fmin(lo, cute.math.abs(e), nan=True)
            out[2 * v_pair] = q
            lo = cute.arch.fmin(lo, cute.math.abs(e_b), nan=True)
            out[2 * v_pair + 1] = q_b
        """
    assert _run(source) == _same(expected)
    # A divisor the loop assigns is no invariant.
    varying = """\
        for v in cutlass.range_constexpr(2):
            e = cutlass.Float32(frag[v])
            s = e + 1.0
            q = cutlass.Float32(_cute_exact_div_fast_f32(e, s, y))
            out[v] = q
        """
    assert DIV_PAIR not in _run(varying)


def test_an_exp_feeding_a_division_pairs_both() -> None:
    # Lane ``b``'s exp product is emitted right after the pair when the next
    # pair reads it.
    source = """\
        for v in cutlass.range_constexpr(2):
            d = cutlass.Float32(frag[v]) - m
            e = cute.math.exp(cutlass.Float32(d))
            q = _cute_exact_div_fast_f32(e, s, y)
            acc = acc + q
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            d = cutlass.Float32(frag[2 * v_pair]) - m
            d_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(d), cutlass.Float32(d_b))
            e = e_pa * e_sa
            e_b = e_pb * e_sb
            q, q_b = _cute_exact_div_fast_f32x2(e, e_b, s, y)
            acc = acc + q
            acc = acc + q_b
        """
    assert _run(source) == _same(expected)
    # NVVM contracts the product into an add that also reads the exp
    # wherever the product sits in the unrolled body: the division pairs
    # all the same.
    summed = """\
        for v in cutlass.range_constexpr(2):
            d = cutlass.Float32(frag[v]) - m
            e = cute.math.exp(cutlass.Float32(d))
            q = _cute_exact_div_fast_f32(e, s, y)
            acc = acc + q
            sum_e = sum_e + e
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            d = cutlass.Float32(frag[2 * v_pair]) - m
            d_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(d), cutlass.Float32(d_b))
            e = e_pa * e_sa
            e_b = e_pb * e_sb
            q, q_b = _cute_exact_div_fast_f32x2(e, e_b, s, y)
            acc = acc + q
            sum_e = sum_e + e
            acc = acc + q_b
            sum_e = sum_e + e_b
        """
    assert _run(summed) == _same(expected)


def test_a_statement_between_an_exp_and_its_division_runs_after_the_product() -> None:
    # The division's lane ``b`` operand reads the exp through a rounding to
    # bf16: lane ``b``'s deferred exp product is emitted before the moved
    # statements that read it, in the order of the loop body.
    source = """\
        for v in cutlass.range_constexpr(2):
            d = cutlass.Float32(frag[v]) - m
            e = cute.math.exp(cutlass.Float32(d))
            r = cutlass.Float32(cutlass.BFloat16(e))
            q = _cute_exact_div_fast_f32(r, s, y)
            acc = acc + q
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            d = cutlass.Float32(frag[2 * v_pair]) - m
            d_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(d), cutlass.Float32(d_b))
            e = e_pa * e_sa
            r = cutlass.Float32(cutlass.BFloat16(e))
            e_b = e_pb * e_sb
            r_b = cutlass.Float32(cutlass.BFloat16(e_b))
            q, q_b = _cute_exact_div_fast_f32x2(r, r_b, s, y)
            acc = acc + q
            acc = acc + q_b
        """
    assert _run(source) == _same(expected)


def test_lane_b_reads_of_a_running_value_keep_the_pair_scalar() -> None:
    # Lane ``b``'s ``m_new`` reads the running max lane ``a`` updates later
    # in the trip: computing it early would read the stale max.
    source = """\
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            m_new = cute.arch.fmax(m, x)
            acc = acc * cute.math.exp(cutlass.Float32(m - m_new)) + cute.math.exp(cutlass.Float32(x - m_new))
            m = m_new
        """
    assert _run(source) == _same(source)


def test_a_running_value_a_lane_updates_twice_keeps_lane_order() -> None:
    # The window trackers of a speculated quotient: each lane updates the
    # running minimum twice.  Those statements keep their order, lane ``a``'s
    # then lane ``b``'s, and the exps still pair.
    source = """\
        lo = cutlass.Float32(float('inf'))
        for v in cutlass.range_constexpr(4):
            x = cutlass.Float32(frag[v])
            e = cute.math.exp(cutlass.Float32(x))
            lo = cute.arch.fmin(lo, cute.math.abs(x))
            lo = cute.arch.fmin(lo, cute.math.abs(e))
            out[v] = e
        total = lo
        """
    expected = """\
        lo = cutlass.Float32(float('inf'))
        for v_pair in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[2 * v_pair])
            x_b = cutlass.Float32(frag[2 * v_pair + 1])
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(x), cutlass.Float32(x_b))
            e = e_pa * e_sa
            lo = cute.arch.fmin(lo, cute.math.abs(x))
            lo = cute.arch.fmin(lo, cute.math.abs(e))
            out[2 * v_pair] = e
            e_b = e_pb * e_sb
            lo = cute.arch.fmin(lo, cute.math.abs(x_b))
            lo = cute.arch.fmin(lo, cute.math.abs(e_b))
            out[2 * v_pair + 1] = e_b
        total = lo
        """
    assert _run(source) == _same(expected)


def test_a_local_assigned_twice_keeps_the_loop_scalar() -> None:
    # Lane ``b``'s copy of ``x`` would need both bindings renamed in order.
    source = """\
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            x = x * 2.0
            e = cute.math.exp(cutlass.Float32(x))
            out[v] = e
        """
    assert _run(source) == _same(source)


def test_loads_and_stored_tensors_keep_the_pair_scalar() -> None:
    load = """\
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32((src.iterator + v).load())
            e = cute.math.exp(cutlass.Float32(x))
            out[v] = e
        """
    assert _run(load) == _same(load)
    # Lane ``b`` indexes a register tensor lane ``a`` stores into.
    stored = """\
        for v in cutlass.range_constexpr(2):
            frag[v] = cutlass.Float32(frag[v]) * 2.0
            e = cute.math.exp(cutlass.Float32(frag[v]))
            out[v] = e
        """
    assert _run(stored) == _same(stored)
    # ... or a packet a call statement appends to.
    appended = """\
        for v in cutlass.range_constexpr(2):
            parts.append(cutlass.Float32(frag[v]))
            e = cute.math.exp(cutlass.Float32(parts[v]))
            out[v] = e
        """
    assert _run(appended) == _same(appended)


@pytest.mark.parametrize(
    "setup",
    [
        "alias = frag",
        "alias = frag[None]",
        "alias = cute.make_rmem_tensor(2, cutlass.Float32)\nfrag = alias",
    ],
    ids=["alias", "view", "aliased_by"],
)
def test_lane_b_reads_of_storage_lane_a_writes_under_another_name_stay_scalar(
    setup: str,
) -> None:
    # Lane ``a`` stores through ``alias`` into the storage lane ``b``'s exp
    # reads as ``frag``: run early, that exp would read the element before
    # lane ``a``'s trip writes it.  Two fragments of their own pair.
    source = """\
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        SETUP
        for v in cutlass.range_constexpr(2):
            alias[v] = cutlass.Float32(1.0)
            e = cute.math.exp(cutlass.Float32(frag[v]))
            vals.append(e)
        """
    shared = textwrap.dedent(source).replace("SETUP", setup)
    assert _run(shared) == _same(shared)
    own = textwrap.dedent(source).replace(
        "SETUP", "alias = cute.make_rmem_tensor(2, cutlass.Float32)"
    )
    assert EXP_PAIR in _run(own)


def test_a_fragment_appended_to_a_list_may_be_written_through_it() -> None:
    # ``zero_fragments(frags)`` may write ``frag`` through the list it was
    # appended to: run early, lane ``b``'s exp would read its element before
    # lane ``a``'s call writes it.  An appended loaded packet is a value.
    source = """\
        frags = []
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        frags.append(frag)
        for v in cutlass.range_constexpr(2):
            e = cute.math.exp(cutlass.Float32(frag[v]))
            zero_fragments(frags)
            vals.append(e)
        """
    assert _run(source) == _same(source)
    loaded = source.replace(
        "cute.make_rmem_tensor(2, cutlass.Float32)", "cute.arch.load(ptr, vector_type)"
    )
    assert EXP_PAIR in _run(loaded)


def test_a_loaded_packet_read_back_from_its_list_pairs() -> None:
    # The loads-first lane unroll appends each lane's packet to a list and
    # reads it back: a loaded packet, which no store reaches.
    source = """\
        packets = []
        for lane in range(2):
            vec = cute.arch.load(ptr, vector_type)
            packets.append(vec)
        for lane in range(2):
            vec = packets[lane]
            for v in cutlass.range_constexpr(2):
                e = cute.math.exp(cutlass.Float32(vec[v]))
                vals.append(e)
        """
    assert EXP_PAIR in _run(source)


@pytest.mark.parametrize(
    ("write", "read"),
    [
        ("frag[v] = cutlass.Float32(src[v])", "operator.getitem(frag, v)"),
        ("write_into(holder)", "holder.frag[v]"),
    ],
    ids=["getitem", "attribute"],
)
def test_reads_of_a_container_spelled_another_way_keep_their_place(
    write: str, read: str
) -> None:
    # Lane ``b``'s exp reads the element lane ``b``'s store (or call) has
    # just changed: run early, it would read the old one.
    source = f"""\
        for v in cutlass.range_constexpr(2):
            {write}
            e = cute.math.exp(cutlass.Float32({read}))
            vals.append(e)
        """
    assert _run(source) == _same(source)


def test_an_exp_of_storage_an_alias_writes_is_not_reused() -> None:
    # ``alias[v] = 0`` writes the element the second exp reads, so that exp
    # is no repeat of ``old``.  The plain sum pairs, keeping the rewrite.
    source = """\
        c = cutlass.Float32(cp)
        alias = frag
        for v in cutlass.range_constexpr(2):
            old = cute.math.exp(cutlass.Float32(frag[v]))
            alias[v] = cutlass.Float32(0.0)
            y = cutlass.Float32(cute.math.exp(cutlass.Float32(frag[v]))) + c
            s = cutlass.Float32(src[v]) + c
            vals.append(old)
            vals.append(y)
            vals.append(s)
        """
    code = _run(source)
    assert ADD_PAIR in code and "cutlass.Float32(old) + c" not in code, code


@pytest.mark.parametrize(
    ("setup", "read"),
    [
        ("frags = [frag]", "frags[0][v]"),
        ("frags = []\nfrags.append(frag)", "frags[0][v]"),
        ("view = cute.make_tensor(frag.iterator, layout)\nalias = view", "alias[v]"),
    ],
    ids=["listed", "appended", "alias_of_a_view"],
)
def test_lane_b_reads_through_a_list_or_an_alias_of_storage_lane_a_writes(
    setup: str, read: str
) -> None:
    # Lane ``b``'s exp reads ``frag`` through the list it is in or through an
    # alias of a view of it: run early, it would read the element before
    # lane ``b``'s own store writes it.  A list of another fragment pairs.
    source = """\
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        SETUP
        for v in cutlass.range_constexpr(2):
            frag[v] = cutlass.Float32(src[v])
            e = cute.math.exp(cutlass.Float32(READ))
            vals.append(e)
        """
    shared = textwrap.dedent(source).replace("SETUP", setup).replace("READ", read)
    assert _run(shared) == _same(shared)
    own = (
        textwrap.dedent(source)
        .replace(
            "SETUP",
            "other = cute.make_rmem_tensor(2, cutlass.Float32)\nfrags = [other]",
        )
        .replace("READ", "frags[0][v]")
    )
    assert EXP_PAIR in _run(own)


def test_an_exp_of_storage_written_through_its_list_is_not_reused() -> None:
    # ``frag[v] = 0`` writes the element the second exp reads through
    # ``frags``, so that exp is no repeat of ``old``.
    source = """\
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        frags = [frag]
        for v in cutlass.range_constexpr(2):
            old = cute.math.exp(cutlass.Float32(frags[0][v]))
            frag[v] = cutlass.Float32(0.0)
            y = cute.math.exp(cutlass.Float32(frags[0][v])) + c
            vals.append(old)
            vals.append(y)
        """
    assert _run(source) == _same(source)


def test_the_lane_index_is_no_invariant_divisor() -> None:
    # Each lane divides by its own index: the exps pair, the divisions not.
    source = """\
        y = cutlass.Float32(yp)
        for v in cutlass.range_constexpr(2):
            e = cute.math.exp(cutlass.Float32(frag[v]))
            q = _cute_exact_div_fast_f32(e, v, y)
            vals.append(q)
        """
    code = _run(source)
    assert EXP_PAIR in code and DIV_PAIR not in code, code


def test_an_exp_assigned_through_a_conversion_pairs() -> None:
    source = """\
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            y = cutlass.Float32(cute.math.exp(cutlass.Float32(x)))
            vals.append(y)
        """
    assert EXP_PAIR in _run(source)


def test_names_read_after_the_loop_keep_their_names() -> None:
    source = """\
        for v in cutlass.range_constexpr(2):
            d = cutlass.Float32(frag[v]) - m
            e = cute.math.exp(cutlass.Float32(d))
            last = e
        result = last
        """
    expected = """\
        for v_pair in cutlass.range_constexpr(1):
            d = cutlass.Float32(frag[2 * v_pair]) - m
            d_b = cutlass.Float32(frag[2 * v_pair + 1]) - m
            e_pa, e_pb, e_sa, e_sb = _cute_exp_f32x2_parts(cutlass.Float32(d), cutlass.Float32(d_b))
            e = e_pa * e_sa
            last = e
            e_b = e_pb * e_sb
            last = e_b
        result = last
        """
    assert _run(source) == _same(expected)


def test_a_v_loop_whose_index_is_read_after_it_stays_scalar() -> None:
    # After the loop ``v`` holds its last lane; the paired loop binds
    # ``v_pair`` instead.
    source = """\
        for v in cutlass.range_constexpr(2):
            e = cute.math.exp(cutlass.Float32(frag[v]))
            vals.append(e)
        last = v
        """
    assert _run(source) == _same(source)
    assert EXP_PAIR in _run(source.replace("last = v", "last = 1"))


def test_a_lambda_keeps_its_parameters_in_the_lane_copy() -> None:
    # Lane ``b``'s copy renames the local ``a``, but not the parameter of the
    # combine, which shadows it.
    source = """\
        for v in cutlass.range_constexpr(2):
            a = cute.math.exp(cutlass.Float32(frag[v]))
            r = cute.arch.warp_reduction(a, lambda a, b: cute.arch.fmax(a, b))
            vals.append(r)
        """
    code = _run(source)
    assert EXP_PAIR in code
    assert "warp_reduction(a_b, lambda a, b: cute.arch.fmax(a, b))" in code, code


def test_nested_v_loops_pair_inside_lane_loops_and_branches() -> None:
    source = """\
        for lane in range(2):
            if lane_ok:
                for v in cutlass.range_constexpr(2):
                    e = cute.math.exp(cutlass.Float32(frag[lane * 2 + v]))
                    out[lane * 2 + v] = e
        """
    code = _run(source)
    assert code.count(EXP_PAIR) == 1
    assert "for v_pair in cutlass.range_constexpr(1):" in code
    assert "out[lane * 2 + (2 * v_pair + 1)] = e_b" in code


_ARITHMETIC = """\
w = cutlass.Float32(wp)
b = cutlass.Float32(bp)
vals = []
for v in cutlass.range_constexpr(2):
    x = cutlass.Float32(frag[v])
    y = x * w
    z = y * y
    s = x * w
    t = s + b
    u = cute.math.fma(x, w, b)
    vals.append(cutlass.BFloat16(z))
    vals.append(cutlass.BFloat16(t))
    vals.append(cutlass.BFloat16(u))
"""


def test_fp32_arithmetic_pairs_unless_it_could_contract() -> None:
    # ``y`` feeds only a product and ``z`` only a conversion, so they pair,
    # as does the explicit fma.  ``s`` feeds the sum ``t``, which NVVM fuses
    # into an FFMA: both stay scalar in each lane.
    expected = """\
        w = cutlass.Float32(wp)
        b = cutlass.Float32(bp)
        vals = []
        for v_pair in cutlass.range_constexpr(1):
            x = cutlass.Float32(frag[2 * v_pair])
            x_b = cutlass.Float32(frag[2 * v_pair + 1])
            y, y_b = cute.arch.mul_packed_f32x2((cutlass.Float32(x), cutlass.Float32(x_b)), (cutlass.Float32(w), cutlass.Float32(w)))
            z, z_b = cute.arch.mul_packed_f32x2((cutlass.Float32(y), cutlass.Float32(y_b)), (cutlass.Float32(y), cutlass.Float32(y_b)))
            s = x * w
            t = s + b
            u, u_b = cute.arch.fma_packed_f32x2((cutlass.Float32(x), cutlass.Float32(x_b)), (cutlass.Float32(w), cutlass.Float32(w)), (cutlass.Float32(b), cutlass.Float32(b)))
            vals.append(cutlass.BFloat16(z))
            vals.append(cutlass.BFloat16(t))
            vals.append(cutlass.BFloat16(u))
            s_b = x_b * w
            t_b = s_b + b
            vals.append(cutlass.BFloat16(z_b))
            vals.append(cutlass.BFloat16(t_b))
            vals.append(cutlass.BFloat16(u_b))
        """
    assert _run(_ARITHMETIC) == _same(expected)


def test_a_sum_of_a_fragment_holding_a_product_stays_scalar() -> None:
    # The fragment's element is the product the first loop stored: NVVM may
    # fuse it into the second loop's sum, which therefore stays scalar.  An
    # element of a fragment holding loaded values pairs.
    source = """\
        w = cutlass.Float32(wp)
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        held = cute.make_rmem_tensor(2, cutlass.Float32)
        for v in cutlass.range_constexpr(2):
            frag[v] = cutlass.Float32(src[v]) * w
            held[v] = cutlass.Float32(src[v])
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(other[v])
            t = x + frag[v]
            u = x + held[v]
            vals.append(cutlass.BFloat16(t))
            vals.append(cutlass.BFloat16(u))
        """
    code = _run(source)
    assert "t = x + frag[2 * v_pair]" in code
    assert ADD_PAIR in code and code.count(ADD_PAIR) == 1


@pytest.mark.parametrize(
    ("statement", "read"),
    [
        ("y = x * q", "y"),
        ("y = x * 1", "y"),
        ("y = cutlass.Float32(cute.math.tanh(x)) + w", "y"),
        ("k = v * 4", "frag[k]"),
    ],
    ids=[
        "operand_of_unknown_type",
        "identity_product",
        "sum_of_a_math_call",
        "integers",
    ],
)
def test_arithmetic_that_is_not_plain_fp32_stays_scalar(
    statement: str, read: str
) -> None:
    # An operand whose type the pass cannot tell, a product NVVM folds to its
    # operand, a sum whose operand may end in an inlined product, and index
    # arithmetic.
    source = f"""\
        w = cutlass.Float32(wp)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            {statement}
            vals.append(cutlass.BFloat16({read}))
        """
    assert _run(source) == _same(source)


@pytest.mark.parametrize(
    "fold",
    [
        "m / 1.0",
        "m * -1.0",
        "m / -1.0",
        "-1.0 * m",
        "m / cutlass.Float32(1.0)",
        "m - 0.0",
        "m - 0",
        "m + -0.0",
        "-0.0 + m",
        "-0.0 - m",
        "m + cutlass.Float32(-0.0)",
        "m - z",
        "m - v",
        "cutlass.Float32(cutlass.Float64(m))",
    ],
)
def test_a_product_through_a_unit_fold_still_feeds_its_sum(fold: str) -> None:
    # NVVM folds ``m / 1`` and ``m - 0.0`` to ``m``, ``-0.0 - m`` and the -1
    # forms to ``-m``, and ``Float32(Float64(m))`` back to ``m``, whatever
    # spells the constant, then fuses ``m``'s product into the sum: the
    # product and the sum stay scalar.  ``z`` may be either zero, and the
    # lane index is 0 in lane 0.
    source = f"""\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        z = 0.0
        if flag:
            z = -0.0
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            m = x * w
            n = {fold}
            t = n + c
            vals.append(cutlass.BFloat16(t))
        """
    assert _run(source) == _same(source)


def test_a_product_through_another_constexpr_loop_s_index_feeds_its_sum() -> None:
    # ``u`` is 1 in the outer loop's second trip: NVVM folds ``t * u`` to
    # ``t`` and fuses ``t``'s product into the sum, so ``t`` stays scalar.
    # A factor that folds nowhere leaves ``t`` to pair.
    source = """\
        w = cutlass.Float32(x)
        acc = cutlass.Float32(0.0)
        for u in cutlass.range_constexpr(2):
            for v in cutlass.range_constexpr(2):
                t = cutlass.Float32(frag[v]) * w
                s = t * u
                acc = acc + s
        """
    assert _run(source) == _same(source)
    assert MUL_PAIR in _run(source.replace("t * u", "t * 2.0"))


def test_a_product_through_a_bitcast_round_trip_feeds_its_sum() -> None:
    # LLVM folds ``bitcast(bitcast(p))`` back to ``p`` and fuses the product
    # into the sum, through the integer conversion that keeps its bits; an
    # integer change of the bits ends the product.
    source = """\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        for v in cutlass.range_constexpr(2):
            p = cutlass.Float32(frag[v]) * w
            t = cutlass.Float32(p).bitcast(cutlass.Uint32)
            q = cutlass.Uint32(t).bitcast(cutlass.Float32)
            s = q + c
            vals.append(cutlass.BFloat16(s))
        """
    assert _run(source) == _same(source)
    assert ADD_PAIR in _run(source.replace("Uint32(t)", "Uint32(t + 1)"))


@pytest.mark.parametrize("fold", ["m + 0.0", "m - -0.0", "m - 1.0", "0.0 - m"])
def test_a_sum_whose_constant_does_not_fold_is_no_product(fold: str) -> None:
    # ``m + 0.0`` turns -0.0 into +0.0, so NVVM keeps it, and so it keeps the
    # others: ``n`` is a sum, and the sum reading it pairs.
    source = f"""\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            m = x * w
            n = {fold}
            t = n + c
            vals.append(cutlass.BFloat16(t))
        """
    code = _run(source)
    assert code.count(ADD_PAIR) == 1 and "t, t_b = " in code, code


def test_operands_of_unknown_origin_may_be_products() -> None:
    # A product of operands of unknown type, an element of a list a product
    # was appended to and of a fragment written by a call: each may be a
    # product NVVM fuses into the sum, which stays scalar (and so does the
    # appended product).  An element of a loaded packet is no product.
    source = """\
        c = cutlass.Float32(cp)
        w = cutlass.Float32(wp)
        stash = []
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        frag.store(vector)
        packet = cute.arch.load(ptr, vector_type)
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(src[v])
            stash.append(x * w)
        vals = []
        for v in cutlass.range_constexpr(2):
            y = p[v] * q[v]
            s = cutlass.Float32(y) + c
            t = cutlass.Float32(stash[v]) + c
            u = frag[v] + c
            z = cutlass.Float32(packet[v]) + c
            vals.append(cutlass.BFloat16(s))
            vals.append(cutlass.BFloat16(t))
            vals.append(cutlass.BFloat16(u))
            vals.append(cutlass.BFloat16(z))
        """
    code = _run(source)
    assert MUL_PAIR not in code, code
    assert code.count(ADD_PAIR) == 1 and "z, z_b = " in code, code


def test_a_call_may_end_in_a_product_unless_known_not_to() -> None:
    # A grouped reduction by product ends in a multiply, and so may a helper
    # NVVM inlines: the sums reading them stay scalar.  A reduction by sum
    # ends in its add, and the sum reading it pairs.
    source = """\
        r = cutlass.Float32(_cute_grouped_reduce_warp(acc, 'prod', cutlass.Float32(1.0), lane, pre=1, group_span=32))
        s = cutlass.Float32(_cute_grouped_reduce_warp(acc, 'sum', cutlass.Float32(0.0), lane, pre=1, group_span=32))
        h = cutlass.Float32(_scaled(acc))
        packet = cute.arch.load(ptr, vector_type)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(packet[v])
            t = x + r
            u = x + s
            w = x + h
            vals.append(cutlass.BFloat16(t))
            vals.append(cutlass.BFloat16(u))
            vals.append(cutlass.BFloat16(w))
        """
    code = _run(source)
    assert code.count(ADD_PAIR) == 1 and "u, u_b = " in code, code


@pytest.mark.parametrize(
    "store",
    [
        "    frag[v] *= w\n",
        "    view = frag[None]\n    view[v] = cutlass.Float32(src[v]) * w\n",
    ],
    ids=["augmented", "view"],
)
def test_a_product_stored_another_way_reaches_the_fragment(store: str) -> None:
    # ``frag[v] *= w`` stores ``frag[v] * w``, and a store through the view
    # ``view`` of ``frag`` lands in ``frag``: either way NVVM may contract
    # the product into the last loop's sum, which stays scalar.
    source = (
        "w = cutlass.Float32(wp)\n"
        "c = cutlass.Float32(cp)\n"
        "frag = cute.make_rmem_tensor(2, cutlass.Float32)\n"
        "for v in cutlass.range_constexpr(2):\n"
        "    frag[v] = cutlass.Float32(src[v])\n"
        "for v in cutlass.range_constexpr(2):\n"
        f"{store}"
        "vals = []\n"
        "for v in cutlass.range_constexpr(2):\n"
        "    t = frag[v] + c\n"
        "    vals.append(cutlass.BFloat16(t))\n"
    )
    assert ADD_PAIR not in _run(source)
    assert ADD_PAIR in _run(source.replace(store, "    out[v] = frag[v]\n"))


@pytest.mark.parametrize(
    "store",
    [
        (
            "for v in cutlass.range_constexpr(2):\n"
            "    frags[0][v] = cutlass.Float32(src[v]) * w\n"
        ),
        (
            "for frag in frags:\n"
            "    for v in cutlass.range_constexpr(2):\n"
            "        frag[v] = cutlass.Float32(src[v]) * w\n"
        ),
    ],
    ids=["index_chain", "loop_over_the_list"],
)
def test_a_product_stored_through_a_list_reaches_the_fragment(store: str) -> None:
    # Either store puts a product into ``c`` through the list it was
    # appended to: NVVM may contract it into the last loop's sum, which
    # stays scalar.  Without the list the sum pairs.
    source = (
        "w = cutlass.Float32(wp)\n"
        "b = cutlass.Float32(bp)\n"
        "c = cute.make_rmem_tensor(2, cutlass.Float32)\n"
        "for v in cutlass.range_constexpr(2):\n"
        "    c[v] = cutlass.Float32(src[v])\n"
        "frags = []\n"
        "frags.append(c)\n"
        f"{store}"
        "vals = []\n"
        "for v in cutlass.range_constexpr(2):\n"
        "    t = cutlass.Float32(c[v]) + b\n"
        "    vals.append(cutlass.BFloat16(t))\n"
    )
    assert ADD_PAIR not in _run(source)
    assert ADD_PAIR in _run(source.replace("frags.append(c)\n", "").replace(store, ""))


def test_a_shuffle_reduction_by_max_is_no_product() -> None:
    # A warp reduction ends in its combine: a max is no product, so the
    # difference reading it pairs; a reduction by product may end in one.
    source = """\
        m = cutlass.Float32(cute.arch.warp_reduction_max(acc, op=_cute_nan_max, threads_in_group=32))
        p = cutlass.Float32(cute.arch.warp_reduction(acc, lambda a, b: a * b, threads_in_group=32))
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            d = x - m
            e = x - p
            vals.append(cutlass.BFloat16(d))
            vals.append(cutlass.BFloat16(e))
        """
    code = _run(source)
    assert code.count(SUB_PAIR) == 1 and "d, d_b = " in code, code


def test_a_scalar_load_of_an_fp32_tensor_is_a_plain_operand() -> None:
    # A partial row tile's row value, ``(t.iterator + i).load() if mask else
    # 0``: an fp32 value (the tensor is fp32) and a memory read, no product,
    # so the difference reading it pairs.  Of a tensor of another dtype the
    # load is no fp32 value, and the difference stays scalar.
    source = """\
        m = (row_amax.iterator + cutlass.Int32(row)).load() if mask_1 else cutlass.Float32(0)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            y = x - m
            vals.append(cutlass.BFloat16(y))
        """
    code = _run(source, tensor_dtypes={"row_amax": "cutlass.Float32"})
    assert SUB_PAIR in code, code
    assert _run(source) == _same(source)


def test_a_fragment_holds_what_its_store_assigned_in_the_store_s_loop() -> None:
    # Both loops assign ``load``.  ``frag`` holds the first loop's product,
    # whatever the second loop's ``load`` is, so the second loop's sum of
    # ``frag`` stays scalar (NVVM may fuse the product into it); ``held``
    # holds the first loop's loaded value, so the difference of it pairs.
    source = """\
        w = cutlass.Float32(wp)
        m = cutlass.Float32(mp)
        frag = cute.make_rmem_tensor(2, cutlass.Float32)
        held = cute.make_rmem_tensor(2, cutlass.Float32)
        for v in cutlass.range_constexpr(2):
            load = cutlass.Float32(src[v]) * w
            frag[v] = cutlass.Float32(load)
            load = cutlass.Float32(src[v])
            held[v] = cutlass.Float32(load)
        vals = []
        for v in cutlass.range_constexpr(2):
            load = cutlass.Float32(other[v])
            t = load + frag[v]
            u = held[v] - m
            vals.append(cutlass.BFloat16(t))
            vals.append(cutlass.BFloat16(u))
        """
    code = _run(source)
    assert "t = load + frag[2 * v_pair]" in code, code
    assert ADD_PAIR not in code and code.count(SUB_PAIR) == 1


def test_a_widened_bf16_lane_stays_in_its_scalar_add() -> None:
    # ptxas reads a bf16 lane straight into a scalar add (FHADD.BF16); a
    # packed add needs both lanes widened first.  So a difference of a
    # widened bf16 lane stays scalar, from a packet, an fp32 fragment holding
    # widened lanes or a scalar load of a bf16 tensor; the difference of an
    # fp32 lane and a widened bf16 loop invariant (``m``) pairs.
    source = """\
        held = cute.make_rmem_tensor(2, cutlass.Float32)
        for v in cutlass.range_constexpr(2):
            held[v] = cutlass.Float32(cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16))
        m = cutlass.Float32(cutlass.Uint16(row[0]).bitcast(cutlass.BFloat16))
        vals = []
        for v in cutlass.range_constexpr(2):
            load = cutlass.Uint16(packet[v]).bitcast(cutlass.BFloat16)
            x = cutlass.Float32(load)
            d = x - m
            e = held[v] - m
            s = cutlass.Float32((scores.iterator + v).load()) - m
            f = cutlass.Float32(frag[v]) - m
            vals.append(cutlass.BFloat16(d))
            vals.append(cutlass.BFloat16(e))
            vals.append(cutlass.BFloat16(s))
            vals.append(cutlass.BFloat16(f))
        """
    code = _run(source, tensor_dtypes={"scores": "cutlass.BFloat16"})
    assert code.count(SUB_PAIR) == 1 and "f, f_b = " in code, code


def test_a_quotient_by_a_constant_may_be_a_product() -> None:
    # LLVM rewrites a division by a power of two as the multiply by its exact
    # reciprocal, which a following sum fuses into an FFMA: the sum reading
    # it stays scalar.  A quotient by a DSL value stays a division.
    source = """\
        w = cutlass.Float32(wp)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            q = x / {divisor}
            y = q + w
            vals.append(cutlass.BFloat16(y))
        """
    assert ADD_PAIR not in _run(source.format(divisor="2.0"))
    assert ADD_PAIR in _run(source.format(divisor="w"))
    # 192 has no exact reciprocal, so the quotient stays a division (a
    # layernorm's mean over 192 columns); 256 does.
    assert ADD_PAIR in _run(source.format(divisor="cutlass.Float32(192)"))
    assert ADD_PAIR not in _run(source.format(divisor="cutlass.Float32(256)"))
    # So does the value of a module constant (an rdim of 1024 or 192); one of
    # unknown value, a name neither the body nor the kernel's parameters
    # bind, may be a power of two.
    module = source.format(divisor="cutlass.Float32(_RDIM_SIZE_1)")
    assert ADD_PAIR not in _run(module, constants={"_RDIM_SIZE_1": 1024})
    assert ADD_PAIR in _run(module, constants={"_RDIM_SIZE_1": 192})
    assert ADD_PAIR not in _run(module, parameters=frozenset({"wp", "frag"}))


@pytest.mark.parametrize(
    ("constants", "paired"),
    [({"_BLOCK_SIZE_0": 1}, False), (None, False), ({"_BLOCK_SIZE_0": 32}, True)],
    ids=["one", "unknown", "thirty-two"],
)
def test_a_product_through_a_module_constant_of_one_feeds_its_sum(
    constants: dict[str, int] | None, paired: bool
) -> None:
    # NVVM folds ``m / _BLOCK_SIZE_0`` to ``m`` for a block size of 1 and
    # fuses ``m``'s product into the sum: ``m`` stays scalar, as it does for
    # a block size of unknown value.  32 folds nowhere.
    source = """\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        frag = cute.arch.load(ptr, vector_type)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            m = x * w
            n = m / _BLOCK_SIZE_0
            t = n + c
            vals.append(cutlass.BFloat16(t))
        """
    parameters = frozenset({"wp", "cp", "ptr", "vector_type"})
    code = _run(source, constants=constants, parameters=parameters)
    assert (MUL_PAIR in code) is paired


def test_kernel_parameters_may_be_one_tensor() -> None:
    # Lane ``b``'s exp reads ``x`` after its own store into ``out``: two
    # parameters of the kernel may be one tensor, so the exp keeps its place.
    source = """\
        for v in cutlass.range_constexpr(2):
            out[v] = cutlass.Float32(1.0)
            e = cute.math.exp(cutlass.Float32(x[v]))
            vals.append(e)
        """
    assert _run(source, parameters=frozenset({"x", "out", "vals"})) == _same(source)
    assert EXP_PAIR in _run(source)


@pytest.mark.parametrize(
    "fma",
    [
        "cute.math.fma(m, 1.0, c)",
        "cute.math.fma(cutlass.Float32(-1.0), m, c)",
        "cute.math.fma(m, v, c)",
    ],
)
def test_an_fma_by_one_is_a_sum_its_product_operand_fuses_into(fma: str) -> None:
    # LLVM folds an fma by 1 or -1 into a sum, and NVVM fuses ``m``'s
    # product into it: neither pairs.  The lane index is 1 in lane 1.  By
    # 2.0, the fma stays one and both pair.
    source = f"""\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            m = x * w
            u = {fma}
            vals.append(cutlass.BFloat16(u))
        """
    assert _run(source) == _same(source)
    code = _run(source.replace(fma, "cute.math.fma(m, 2.0, c)"))
    assert MUL_PAIR in code and "cute.arch.fma_packed_f32x2(" in code


@pytest.mark.parametrize("addend", ["-0.0", "cutlass.Float32(-0.0)"])
def test_an_fma_adding_minus_zero_is_a_product(addend: str) -> None:
    # LLVM folds ``fma(x, w, -0.0)`` into ``x * w``, which NVVM fuses into
    # the sum reading it: neither pairs.  Adding +0.0 is no fold (it turns
    # -0.0 into +0.0), and both pair.
    source = f"""\
        w = cutlass.Float32(wp)
        c = cutlass.Float32(cp)
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            t = cute.math.fma(x, w, {addend})
            s = t + c
            vals.append(cutlass.BFloat16(s))
        """
    assert _run(source) == _same(source)
    code = _run(source.replace(addend, "0.0"))
    assert ADD_PAIR in code and "cute.arch.fma_packed_f32x2(" in code


@pytest.mark.parametrize("fold", ["x - 0.0", "x + -0.0", "-0.0 - x"])
def test_a_sum_nvvm_folds_away_stays_scalar(fold: str) -> None:
    # NVVM folds these to ``x`` or ``-x``, which keep a NaN's payload, where
    # a packed sum gives the canonical NaN.
    source = f"""\
        vals = []
        for v in cutlass.range_constexpr(2):
            x = cutlass.Float32(frag[v])
            y = {fold}
            vals.append(cutlass.BFloat16(y))
        """
    assert _run(source) == _same(source)


def test_a_read_through_a_display_of_packets_pairs_unless_one_is_written() -> None:
    # The persistent-branch vectorizer reads lane ``v`` of its packets as
    # ``(vec0, vec1)[v // 2][v % 2]``: the display indexes loaded packets,
    # which no store reaches.  A fragment the body writes keeps it scalar.
    source = """\
        vec0 = cute.arch.load(ptr0, vector_type)
        vec1 = SECOND
        for v in cutlass.range_constexpr(4):
            WRITE
            e = cute.math.exp(cutlass.Float32((vec0, vec1)[v // 2][v % 2]))
            vals.append(e)
        """
    packets = (
        textwrap.dedent(source)
        .replace("SECOND", "cute.arch.load(ptr1, vector_type)")
        .replace("    WRITE\n", "")
    )
    assert EXP_PAIR in _run(packets)
    written = (
        textwrap.dedent(source)
        .replace("SECOND", "cute.make_rmem_tensor(2, cutlass.Float32)")
        .replace("WRITE", "vec1[v % 2] = cutlass.Float32(src[v])")
    )
    assert _run(written) == _same(written)


# --------------------------------------------------------------------------
# Renders
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    row_max_out = torch.empty([m], dtype=torch.float32, device=x.device)
    row_sum_out = torch.empty([m], dtype=torch.float32, device=x.device)
    prob_sums = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(32, n)
    for tile_m in hl.tile(m):
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            vals = x[tile_m, tile_n].to(torch.float32)
            block_max = torch.amax(vals, dim=-1)
            delta = torch.where(
                block_max[:, None] == float("-inf"), 0.0, vals - block_max[:, None]
            )
            block_sum = torch.sum(torch.exp(delta), dim=-1)
            new_max = torch.maximum(row_max, block_max)
            scale = torch.exp(-torch.abs(row_max - block_max))
            row_scale = torch.where(row_max == new_max, 1.0, scale)
            block_scale = torch.where(block_max == new_max, 1.0, scale)
            row_sum = row_sum * row_scale + block_sum * block_scale
            row_max = new_max
        row_max_out[tile_m] = row_max
        row_sum_out[tile_m] = row_sum
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + (
                torch.exp(x[tile_m, tile_n].to(torch.float32) - row_max[:, None])
                / row_sum[:, None]
            )
        prob_sums[tile_m] = torch.sum(acc, dim=-1)
    return row_max_out, row_sum_out, prob_sums


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        out[tile, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.exp(x[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile].to(torch.float32)
        inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        out[tile] = (0.5 * v * (1.0 + torch.tanh(inner))).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sums_of_folded_products(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sums of a product NVVM gets back from ``m - 0.0``, ``m + -0.0`` and an
    fp64 round trip, and a sum with nothing to fuse."""
    minus_zero = torch.empty_like(x)
    plus_negative_zero = torch.empty_like(x)
    round_trip = torch.empty_like(x)
    plain = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        m = x[tile] * w[tile]
        minus_zero[tile] = (m - 0.0) + b[tile]
        plus_negative_zero[tile] = (m + (-0.0)) + b[tile]
        round_trip[tile] = m.to(torch.float64).to(torch.float32) + b[tile]
        plain[tile] = x[tile] + b[tile]
    return minus_zero, plus_negative_zero, round_trip, plain


_STATS = {
    "block_sizes": [512, 1],
    "num_threads": [32, 0],
    "cute_vector_widths": [16, 8],
}
_EXP = {"block_sizes": [1024], "num_threads": [128], "cute_vector_widths": [8]}


def _config(values: dict[str, object], paired: bool = True) -> helion.Config:
    return helion.Config(**values, cute_pair_lane_math=paired)


def _bind(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    *,
    capability: tuple[int, int] = (10, 0),
) -> BoundKernel:
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(kernel, args, capability=capability)


def _render(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config,
    *,
    capability: tuple[int, int] = (10, 0),
) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(kernel, args, capability=capability).to_code(config)


def _stats_args() -> tuple[torch.Tensor]:
    return (torch.empty(4, 2048, dtype=torch.bfloat16),)


def test_softmax_stats_pair_in_both_sweeps() -> None:
    code = _render(_softmax_stats, _stats_args(), _config(_STATS))
    # The block sums' exps and the probability sweep's; its quotients are
    # the plain ``/`` (``div.full``), one per lane.
    assert code.count(EXP_PAIR) == 2
    assert DIV_PAIR not in code
    assert code.count("cutlass.range_constexpr(8)") == 2
    # Under ``cute_exact_division``: the probability sweep's exact divisions
    # pair too, and so do the exps of its exact-division redo copy.
    code = _render(with_exact_division(_softmax_stats), _stats_args(), _config(_STATS))
    assert code.count(EXP_PAIR) == 3
    assert code.count(DIV_PAIR) == 1
    assert code.count("cutlass.range_constexpr(8)") == 3
    # The knob's default keeps the scalar code.
    for plain in (
        _render(_softmax_stats, _stats_args(), _config(_STATS, paired=False)),
        _render(_softmax_stats, _stats_args(), helion.Config(**_STATS)),
    ):
        assert EXP_PAIR not in plain and DIV_PAIR not in plain
        assert "_exp_f32x2_parts as _cute_exp_f32x2_parts" not in plain


def test_the_knob_is_offered_where_the_pass_can_apply() -> None:
    """Searchable for exp (and sigmoid), fp32-division and fp32-arithmetic
    kernels on sm_100 without ``fast_math``; elsewhere absent, a config
    naming it rejected."""
    key = "cute_pair_lane_math"

    def flat_keys(bound: BoundKernel) -> set[str]:
        return {name for name, _, _ in bound.config_spec.flat_key_layout()}

    assert key in flat_keys(_bind(_softmax_stats, _stats_args()))
    assert key in flat_keys(_bind(_exp, (torch.empty(1 << 14),)))
    # A lone sigmoid, which the FastMathUnused warning does not count on
    # this backend (its lowering ignores fast_math), still has an exp.
    assert key in flat_keys(_bind(_sigmoid, (torch.empty(1 << 14),)))
    assert key in flat_keys(_bind(_add, (torch.empty(1 << 14), torch.empty(1 << 14))))
    integers = torch.empty(1 << 14, dtype=torch.int32)
    fast = helion.kernel(
        _softmax_stats.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=True,
    )
    for bound in (
        _bind(_softmax_stats, _stats_args(), capability=(9, 0)),
        _bind(fast, _stats_args()),
        _bind(_add, (integers, integers)),
    ):
        assert key not in flat_keys(bound)
        config = _config(_EXP if bound.kernel is _add else _STATS)
        with pytest.raises(helion.exc.InvalidConfig, match=key):
            bound.config_spec.normalize(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert key not in config.config


def test_an_absent_knob_stays_absent() -> None:
    """Off unless a config sets it, like the other default-off lowering
    booleans: normalizing a config without the knob (a compiler seed) adds
    nothing, and an explicit value stays."""
    spec = _bind(_softmax_stats, _stats_args()).config_spec
    key = "cute_pair_lane_math"
    assert key not in spec.normalized_config(helion.Config(**_STATS)).config
    for value in (False, True):
        normalized = spec.normalized_config(_config(_STATS, paired=value))
        assert normalized.config[key] is value


def test_an_elementwise_exp_pairs_around_its_packet_appends() -> None:
    code = _render(_exp, (torch.empty(1 << 14, dtype=torch.float32),), _config(_EXP))
    assert code.count(EXP_PAIR) == 1
    assert "cutlass.range_constexpr(4)" in code
    assert "cute.math.exp(" not in code


def test_gelu_pairs_its_products_and_fma_but_not_what_could_contract() -> None:
    # ``0.044715 * v * v`` and ``0.5 * v`` feed products, the cube term the
    # emitted fma, and the result a conversion: they pair.  The product the
    # tanh reads (it may contract inside the inlined tanh) and the sum of the
    # tanh's result stay scalar in each lane.
    args = (torch.empty(1 << 14, dtype=torch.bfloat16),)
    code = _render(_gelu_tanh, args, _config(_EXP))
    assert code.count(MUL_PAIR) == 4
    assert code.count("cute.arch.fma_packed_f32x2(") == 1
    assert code.count("cute.math.tanh(") == 2
    assert "_packed_f32x2(" not in _render(_gelu_tanh, args, _config(_EXP, False))
    code = _render(_add, (torch.empty(1 << 14), torch.empty(1 << 14)), _config(_EXP))
    assert code.count(ADD_PAIR) == 1
    # Only the plain sum: the others read a product NVVM folds them back to.
    args = tuple(torch.empty(1 << 14) for _ in range(3))
    code = _render(_sums_of_folded_products, args, _config(_EXP))
    assert code.count(ADD_PAIR) == 1, code


# --------------------------------------------------------------------------
# GPU: the paired kernels against the pass switched off, bit for bit
# --------------------------------------------------------------------------


def _exp_inputs() -> torch.Tensor:
    """Every fp32 exponent with random significands and signs, a dense
    sweep of the arguments where ``exp`` overflows or goes subnormal, and
    the specials; 1024-element aligned."""
    generator = torch.Generator().manual_seed(0)
    exponents = torch.arange(256, dtype=torch.int32).repeat_interleave(256)
    significands = torch.randint(
        0, 1 << 23, exponents.shape, generator=generator, dtype=torch.int32
    )
    signs = torch.randint(0, 2, exponents.shape, generator=generator, dtype=torch.int32)
    patterns = ((signs << 31) | (exponents << 23) | significands).view(torch.float32)
    edges = torch.empty(1 << 16).uniform_(-110.0, 95.0, generator=generator)
    specials = torch.tensor(
        [
            *(0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45),
            *(88.72283, 88.72284, -87.33654, -87.33655, -103.27893, -103.97208),
            *(-103.97209, 1.0, -1.0, 0.5, -0.5, 2.0**-24, -(2.0**-24)),
        ]
    )
    values = torch.cat([patterns, edges, specials])
    pad = -values.numel() % 1024
    return torch.cat([values, torch.zeros(pad)])


def _stats_inputs() -> torch.Tensor:
    """Rows of normal values, of values spread wide enough that the
    probabilities leave the exact-division window (the redo fires), with
    -inf entries, and all -inf."""
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(8, 4096, generator=generator)
    rows[1] *= 40.0
    rows[2] = torch.empty(4096).uniform_(-120.0, 0.0, generator=generator)
    rows[3, ::7] = float("-inf")
    rows[4] = float("-inf")
    rows[5] *= 1e-3
    return rows.to(torch.bfloat16)


def _run_both(
    kernel: helion.Kernel,
    args: tuple[torch.Tensor, ...],
    values: dict[str, object],
    marker: str = EXP_PAIR,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``kernel``'s outputs with the knob on and off (``marker`` in the code
    only with it on)."""

    def outputs(paired: bool) -> list[torch.Tensor]:
        bound = kernel._bind_isolated(args)
        config = _config(values, paired)
        assert (marker in bound.to_code(config)) is paired
        result = bound.compile_config(config)(*args)
        return list(result) if isinstance(result, tuple) else [result]

    return outputs(True), outputs(False)


def _assert_same_bits(left: list[torch.Tensor], right: list[torch.Tensor]) -> None:
    for a, b in zip(left, right, strict=True):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_cuda_paired_exp_matches_the_scalar_exp_bit_for_bit() -> None:
    x = _exp_inputs().to(DEVICE)
    paired, plain = _run_both(_exp, (x,), _EXP)
    _assert_same_bits(paired, plain)
    # The sweep reaches overflow, subnormal results and zero.
    (out,) = plain
    assert torch.isinf(out).any() and (out == 0).any()
    assert ((out > 0) & (out < torch.finfo(torch.float32).tiny)).any()


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_cuda_paired_softmax_stats_match_the_scalar_kernel_bit_for_bit() -> None:
    x = _stats_inputs().to(DEVICE)
    paired, plain = _run_both(_softmax_stats, (x,), _STATS)
    _assert_same_bits(paired, plain)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_cuda_paired_arithmetic_matches_the_scalar_kernels_bit_for_bit() -> None:
    generator = torch.Generator().manual_seed(0)
    x = (torch.randn(1 << 16, generator=generator) * 8).to(torch.bfloat16)
    paired, plain = _run_both(_gelu_tanh, (x.to(DEVICE),), _EXP, MUL_PAIR)
    _assert_same_bits(paired, plain)
    a, b = (torch.randn(1 << 16, generator=generator).to(DEVICE) for _ in range(2))
    paired, plain = _run_both(_add, (a, b), _EXP, ADD_PAIR)
    _assert_same_bits(paired, plain)


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0))
def test_cuda_sums_of_folded_products_match_the_scalar_kernel_bit_for_bit() -> None:
    # 4097 * 4097 = 16785409 rounds to 16785408 in fp32: added to -16785408,
    # the fused product gives 1 and the rounded one 0.  The scalar kernel
    # fuses each folded spelling of the product into its sum, so only the
    # plain sum may pair.
    x = torch.full((1 << 14,), 4097.0, device=DEVICE)
    args = (x, x.clone(), torch.full_like(x, -16785408.0))
    paired, plain = _run_both(_sums_of_folded_products, args, _EXP, ADD_PAIR)
    _assert_same_bits(paired, plain)


def test_the_first_seeds_with_vector_lanes_cover_the_knob() -> None:
    """The compiler seeds stay as they were; the first four with lanes to
    pair carry the knob's coverage witnesses."""
    spec = _bind(_softmax_stats, _stats_args()).config_spec
    assert not any(
        "cute_pair_lane_math" in seed.config for seed in spec.compiler_seed_configs
    )
    # The carriers are the seeds as the search canonicalizes them.
    seeds = [
        seed for _, seed in spec.create_config_generation().seed_flat_config_pairs()
    ]
    pairable = [
        seed
        for seed in seeds
        if any(width >= 2 for width in seed.config.get("cute_vector_widths", ()))
        or seed.config.get("cute_lane_unroll", 1) >= 2
    ]
    (group,) = [
        group
        for group in spec.compiler_coverage_groups
        if group.key == "cute_pair_lane_math"
    ]
    assert [witness.value for witness in group.witnesses] == [True] * 4
    assert [witness.carrier for witness in group.witnesses] == pairable[:4]
    fast = helion.kernel(
        _softmax_stats.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=True,
    )
    assert not any(
        group.key == "cute_pair_lane_math"
        for group in _bind(fast, _stats_args()).config_spec.compiler_coverage_groups
    )


def test_a_kernel_reducing_a_full_slice_is_witnessed_too() -> None:
    """Witnesses go wherever a V-loop can pair: a softmax over ``x[tile, :]``
    rows carries them like an elementwise kernel, the knob still off in
    every ordinary seed."""
    key = "cute_pair_lane_math"
    for kernel, args in (
        (_row_softmax, (torch.empty(64, 512, dtype=torch.bfloat16),)),
        (_exp, (torch.empty(1 << 14),)),
    ):
        spec = _bind(kernel, args).config_spec
        (group,) = [g for g in spec.compiler_coverage_groups if g.key == key]
        assert group.witnesses and all(w.value is True for w in group.witnesses)
        assert not any(seed.config.get(key) for seed in spec.compiler_seed_configs)
