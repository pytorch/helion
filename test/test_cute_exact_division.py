"""The exact loop-invariant fp32 division hoist of the CuTe backend.

``helion/_compiler/cute/exact_division.py`` (the device helpers and the
bit-identity argument) and the ``2'`` sub-pass of
``helion/_compiler/cute/hoist_loop_invariant_recip.py``: an fp32 ``x / d``
whose divisor is invariant across a ``for`` loop hoists ``y = 1.0 / d`` and the
admission window of ``d`` out of the loop and divides with the Markstein
sequence ``RN(q + RN(x - d*q) * y)``, ``q = RN(x*y)``.  Loops that can be
re-executed verbatim speculate the whole sweep and redo it with ``div.rn``
when a dividend left the window (running min/max of ``|x|``); a loop the redo
cannot take keeps its plain division.  The result is ``x / d`` bit for bit.

CPU: render pins (where the redo form applies and where the plain division
must stay, which loop-carried names a redo copy restores under the DSL's
staging rules, the knob's plain form and the retired guard form rendering the
plain kernel).  GPU: bit identity against ``torch`` division over structured
operand pairs (every exponent pair, dense random bit patterns, specials,
near-midpoint quotients; more than 10**8 pairs), the softmax / JSD rows
against the plain render on the wave-5 input classes, and the rows kernels
against their plain renders on hard rows under pinned and random configs with
one and with several rows per thread -- on ragged shapes (a partial row block,
a reduction axis the lane sweeps and reduction chunks do not divide), where
the masked tails are what a change of code shape perturbs.
"""

from __future__ import annotations

import ast
import random
import re
import textwrap
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.hoist_loop_invariant_recip import (
    hoist_exact_invariant_divisions,
)
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytest.importorskip("cutlass")

FAST = "_cute_exact_div_fast_f32("
BOUNDS = "_cute_exact_div_bounds_f32("
REDO = "if _cute_exact_div_redo_f32("
ANY_EXACT = "_cute_exact_div"
HOISTED = "_helion_exdiv_"
# Ragged shapes: a row count no row block above one divides (every
# several-rows config sees a partial row block, with its ``if mask else``
# selects in the sweeps) and a reduction axis the lane sweeps (threads x
# vector width) and the reduction chunks do not divide.
RAGGED_ROWS, RAGGED_COLS = 53, 4000


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """examples/acfs/softmax_acf.py's body."""
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


def _jsd(
    beta: float, temperature: float, student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """examples/fused_linear_jsd.py::fused_linear_jsd_kernel's row body."""
    batch, _vocab = student.size()
    loss = student.new_empty(batch, dtype=torch.float32)
    for tile in hl.tile(batch):
        s = student[tile, :] / temperature
        t = teacher[tile, :] / temperature
        sp = torch.softmax(s.to(torch.float32), dim=-1)
        tp = torch.softmax(t.to(torch.float32), dim=-1)
        slp = torch.log_softmax(s.to(torch.float32), dim=-1)
        tlp = torch.log_softmax(t.to(torch.float32), dim=-1)
        m = (1 - beta) * sp + beta * tp
        log_m = torch.log(m)
        skl = (sp * (slp - log_m)).sum(dim=-1)
        tkl = (tp * (tlp - log_m)).sum(dim=-1)
        loss[tile] = (1 - beta) * skl + beta * tkl
    return loss


def _rows_div(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Every row divided by its own scalar: the operand-pair harness."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = x[tile_m, :] / s[tile_m][:, None]
    return out


def _divide_by_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] / scale[0]
    return out


def _divide_inside_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n]
            local_sum = torch.sum(values, dim=1, keepdim=True)
            out[tile_m, tile_n] = values / local_sum
    return out


def _int_divs(x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fd = torch.empty_like(x)
    md = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        fd[tile] = x[tile] // d[0]
        md[tile] = x[tile] % d[0]
    return fd, md


def _sum_and_store(
    x: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a row block, store it and accumulate its sum: codegen emits
    the store sweep with the reduction sweep's statement objects."""
    m, n = x.size()
    out = torch.empty_like(x)
    res = torch.empty(m, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n] / s[tile_m][:, None]
            out[tile_m, tile_n] = v
            acc = acc + v.sum(dim=1)
        res[tile_m] = acc
    return out, res


def _normalize_and_sum(
    x: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A rolled row kernel with a store sweep and a reduction sweep."""
    m, n = x.size()
    out = torch.empty_like(x)
    res = torch.empty(m, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :] / s[tile_m][:, None]
        out[tile_m, :] = row
        res[tile_m] = row.sum(dim=1)
    return out, res


def _carried_alias(x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The divisor is rebound from another row value at the end of every
    inner iteration: the first block divides by ``s``, the rest by ``t``."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        d = s[tile_m]
        e = t[tile_m]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / d[:, None]
            d = e
    return out


def _carried_acc(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The divisor accumulates the quotients it produced."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        d = s[tile_m]
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n] / d[:, None]
            out[tile_m, tile_n] = v
            d = d + v.sum(dim=1)
    return out


def _rolled_rows_two_sums(
    x: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A rolled row kernel whose quotient feeds a store and two reductions."""
    m, n = x.size()
    out = torch.empty_like(x)
    r1 = torch.empty(m, dtype=torch.float32, device=x.device)
    r2 = torch.empty(m, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :] / s[tile_m][:, None]
        out[tile_m, :] = row
        r1[tile_m] = row.sum(dim=1)
        r2[tile_m] = (row * row).sum(dim=1)
    return out, r1, r2


def _chained_divisions(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The first quotient's row sum divides the row again: a speculative
    pass that kept a stale accumulator poisons the whole row."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :] / s[tile_m][:, None]
        d2 = row.sum(dim=1)
        out[tile_m, :] = row / d2[:, None]
    return out


softmax = helion.kernel(backend="cute", static_shapes=True)(_softmax)
softmax_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_softmax)
softmax_fast_math = helion.kernel(backend="cute", static_shapes=True, fast_math=True)(
    _softmax
)
jsd = helion.kernel(backend="cute", static_shapes=True)(_jsd)
jsd_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_jsd)
rows_div = helion.kernel(backend="cute", static_shapes=True)(_rows_div)
divide_by_scale = helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none"
)(_divide_by_scale)
divide_by_scale_fast_math = helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", fast_math=True
)(_divide_by_scale)
divide_inside_loop = helion.kernel(backend="cute", static_shapes=True)(
    _divide_inside_loop
)
int_divs = helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")(
    _int_divs
)
sum_and_store = helion.kernel(backend="cute", static_shapes=True)(_sum_and_store)
sum_and_store_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_sum_and_store)
normalize_and_sum = helion.kernel(backend="cute", static_shapes=True)(
    _normalize_and_sum
)
normalize_and_sum_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_normalize_and_sum)
carried_alias = helion.kernel(backend="cute", static_shapes=True)(_carried_alias)
carried_alias_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_carried_alias)
carried_acc = helion.kernel(backend="cute", static_shapes=True)(_carried_acc)
carried_acc_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_carried_acc)
rolled_rows_two_sums = helion.kernel(backend="cute", static_shapes=True)(
    _rolled_rows_two_sums
)
rolled_rows_two_sums_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_rolled_rows_two_sums)
chained_divisions = helion.kernel(backend="cute", static_shapes=True)(
    _chained_divisions
)
chained_divisions_plain = helion.kernel(
    backend="cute", static_shapes=True, cute_exact_division=False
)(_chained_divisions)
# The rows kernels with a store sweep and one or more reduction sweeps.
ROWS_KERNELS: list[tuple[helion.Kernel, helion.Kernel]] = [
    (normalize_and_sum, normalize_and_sum_plain),
    (rolled_rows_two_sums, rolled_rows_two_sums_plain),
    (chained_divisions, chained_divisions_plain),
]

SOFTMAX_CONFIG: dict[str, Any] = {
    # The campaign's softmax_acf shape-1 winner (rolled sweeps, 128 threads).
    "block_sizes": [1],
    "num_threads": [1, 128],
    "reduction_loops": [4096],
    "cute_vector_widths": [4, 1],
    "cute_lane_layouts": ["strided", "strided"],
    "cute_reduction_reloads": ["register_prefetch"],
}
JSD_CONFIG: dict[str, Any] = {
    "block_sizes": [1],
    "num_threads": [1, 256],
    "reduction_loops": [4096],
    "cute_vector_widths": [4, 1],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_reduction_reloads": ["register"],
}
ROWS_CONFIG: dict[str, Any] = {
    "block_sizes": [1],
    "num_threads": [1, 128],
    "reduction_loops": [1024],
    "cute_vector_widths": [4, 1],
}
LANE_LOOP_CONFIG: dict[str, Any] = {"block_sizes": [1024], "num_threads": [128]}
# A row block of one, 256-column inner tiles, two 4-wide lanes per thread.
NESTED_CONFIG: dict[str, Any] = {
    "block_sizes": [1, 256],
    "num_threads": [0, 128],
    "cute_vector_widths": [1, 4],
}
# ``cute_independent_reduction``: four partial accumulators per rolled sweep.
INDEPENDENT_SOFTMAX_CONFIG: dict[str, Any] = {
    **SOFTMAX_CONFIG,
    "cute_independent_reduction": True,
}
INDEPENDENT_JSD_CONFIG: dict[str, Any] = {
    **JSD_CONFIG,
    "cute_independent_reduction": True,
}
INDEPENDENT_ROWS_CONFIG: dict[str, Any] = {
    **ROWS_CONFIG,
    "cute_independent_reduction": True,
}
# JSD configs whose masked tails the retired guard form perturbed (1-2 ulp in
# the reductions over the quotients on 50 x 4096 and 48 x 4000): the sum of
# products ``(1-beta)*sp + beta*tp`` and ``acc + sp*(slp - log_m)`` after
# the quotient is what ptxas contracts differently once the loop's code
# shape changes.  One row per thread with 128 reduction threads, and two
# several-rows configs (the row block on one thread, 2- and 8-wide row
# vectors).  Masked tails now fail closed: the first takes the redo form on
# an exact vocabulary only, the other two (a partial row block on 50 rows)
# never do.
JSD_RAGGED_CONFIGS: list[dict[str, Any]] = [
    {
        **JSD_CONFIG,
        "num_threads": [1, 128],
        "cute_reduction_reloads": ["register_prefetch"],
        "cute_independent_reduction": True,
    },
    {
        "block_sizes": [8],
        "reduction_loops": [1024],
        "cute_vector_widths": [2, 1],
        "cute_lane_layouts": ["blocked", "blocked"],
        "cute_reduction_reloads": ["auto"],
        "cute_independent_reduction": True,
    },
    {
        "block_sizes": [16],
        "num_threads": [1, 128],
        "reduction_loops": [4096],
        "cute_vector_widths": [8, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["register_prefetch"],
        "cute_independent_reduction": True,
    },
]
# Several rows per thread (a row block over fewer row threads than rows, or
# row vectors wider than one): codegen nests each row's rolled sweep in a
# ``cutlass.range_constexpr`` lane loop and defines the sweep's accumulator
# inside that lane body.  The sweep takes the redo form there and must roll
# the lane's accumulator back before the redo (the shapes the autotuner
# sampled when the redo form lost that accumulator).
SEVERAL_ROWS_CONFIGS: list[dict[str, Any]] = [
    {
        # 32 rows over 2 row threads, 8-wide row vectors, a 2-CTA cluster.
        "block_sizes": [32],
        "num_threads": [2, 16],
        "reduction_loops": [32],
        "cute_vector_widths": [1, 8],
        "cute_lane_layouts": ["blocked", "blocked"],
        "cute_reduction_reloads": ["gmem"],
        "cute_cluster_n": 2,
    },
    {
        # 16 rows over 2 row threads, 4-wide, an 8-CTA cluster.
        "block_sizes": [16],
        "num_threads": [2, 8],
        "reduction_loops": [4096],
        "cute_vector_widths": [1, 4],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 8,
    },
    {
        # 64 rows over 8 row threads, 8-wide row vectors, one reduction
        # thread per row, independent and replicated partial accumulators.
        "block_sizes": [64],
        "num_threads": [8, 1],
        "reduction_loops": [512],
        "cute_vector_widths": [8, 2],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 4,
        "cute_independent_reduction": True,
        "cute_replicated_reduction": True,
    },
    {
        # 32 rows on one row thread, 16-wide vectors on both axes.
        "block_sizes": [32],
        "num_threads": [1, 0],
        "reduction_loops": [128],
        "cute_vector_widths": [16, 16],
        "cute_lane_layouts": ["blocked", "blocked"],
        "cute_reduction_reloads": ["register"],
        "cute_independent_reduction": True,
    },
]


# One row per thread, vector packets on the reduction axis: the family the
# autotuner draws for a 53 x 4000 shape.  The first is corpus-4 config 43 of
# the fourth review: ``reduction_loops [2000]`` over 4000 columns gives the
# second trip's last lane a masked packet on 24 of the 128 threads, the
# masked zero dividend (``... if mask else cutlass.Float32(0)``) made the
# redo fire for those threads in every row while the rest kept the
# speculative pass, and the sum of the quotients' squares differed from the
# plain render by 1 ulp in two rows.  Every operand a select reaches now
# keeps the plain division; on an exact axis (no select) the same configs
# take the redo form.
ONE_ROW_PACKET_CONFIGS: list[dict[str, Any]] = [
    {
        "block_sizes": [1],
        "num_threads": [1, 128],
        "reduction_loops": [2000],
        "cute_vector_widths": [4, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["auto"],
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 128],
        "reduction_loops": [2000],
        "cute_vector_widths": [4, 1],
        "cute_lane_layouts": ["strided", "strided"],
        "cute_reduction_reloads": ["register"],
    },
    {
        "block_sizes": [1],
        "num_threads": [1, 256],
        "reduction_loops": [4000],
        "cute_vector_widths": [8, 1],
        "cute_lane_layouts": ["blocked", "blocked"],
        "cute_reduction_reloads": ["register_prefetch"],
    },
]


def _on_exact_axis(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` with its 4000-column reduction chunks sized for 4096."""
    chunks = {2000: 2048, 4000: 4096}
    return {
        **config,
        "reduction_loops": [chunks.get(n, n) for n in config["reduction_loops"]],
    }


def _several_rows_per_thread(config: dict[str, Any]) -> bool:
    """Whether ``config`` gives a thread several rows of the row block
    (``num_threads[0]`` of 0 means one thread per row)."""
    block = config["block_sizes"][0]
    threads = config.get("num_threads", [0])[0] or block
    width = config.get("cute_vector_widths", [1])[0]
    return block > 1 and (threads < block or width > 1)


def _hard_rows(
    device: torch.device | str, rows: int = 64, cols: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rows`` fp32 rows of ``cols`` whose first 16 rows leave the admitted
    window: infinite, signed-zero, subnormal, NaN and zero dividends,
    dividends below the window of a ``2**120`` divisor, zero, subnormal,
    ``2**-120``, infinite, NaN, negative, ``2**100`` and ``2**126``
    divisors, overflowing (``1e38 / 0.3``) and underflowing
    (``2**-126 / 2**100``) quotients, a ``-inf`` row -- and the last three
    rows (the tail of every row block that does not divide ``rows``) hard
    cases of their own."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(rows, cols, generator=g) * 3
    s = torch.rand(rows, generator=g) + 0.5
    x[0, 0] = float("inf")
    x[1, 1] = -0.0
    x[2, :] = 1e-40
    x[3, 100] = float("nan")
    x[4, 5] = 0.0
    s[4] = 0.0
    s[5] = 1e-39
    s[6] = 2.0**120
    x[6, ::7] = 1e-3
    s[7] = 2.0**-120
    s[8] = float("inf")
    s[9] = float("nan")
    s[10] = -0.75
    x[11, :] = 1e38
    s[11] = 0.3
    x[12, ::3] = 1e-39
    s[12] = 3.0
    x[13, :] = float("-inf")
    x[14, :] = 2.0**-126
    s[14] = 2.0**100
    s[15] = 2.0**126
    x[15, ::5] = 0.5
    x[rows - 1, 7] = float("inf")
    s[rows - 1] = 0.0
    x[rows - 2, :] = 1e-40
    s[rows - 3] = 2.0**120
    x[rows - 3, ::9] = 1e-3
    return x.to(device), s.to(device)


def _bind(kernel: helion.Kernel, args: tuple[object, ...]) -> BoundKernel:
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        return _cpu_bind(kernel, args)


def _render(kernel: helion.Kernel, args: tuple[object, ...], **config: Any) -> str:
    return _kernel_body(_bind(kernel, args).to_code(helion.Config(**config)))


def _kernel_body(code: str) -> str:
    """The device function text: no module header (imports of helpers used
    by other kernels rendered in this process can appear there) and no
    ``# src[...]`` comments (which quote the kernel's own ``/``)."""
    body = code.split("@cute.kernel", 1)[1]
    return "\n".join(line for line in body.splitlines() if "# src[" not in line)


def _redo_copies(code: str) -> list[str]:
    """The bodies of the ``if _cute_exact_div_redo_f32(...)`` statements."""
    tree = ast.parse(textwrap.dedent(code))
    copies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_cute_exact_div_redo_f32" in ast.unparse(
            node.test
        ):
            copies.append("\n".join(ast.unparse(stmt) for stmt in node.body))
    return copies


# --------------------------------------------------------------------------
# CPU render pins
# --------------------------------------------------------------------------


def test_softmax_row_sum_takes_the_redo_form() -> None:
    x = torch.empty(4096, 4096, dtype=torch.float16)
    code = _render(softmax, (x,), **SOFTMAX_CONFIG)
    assert "_helion_exdiv_y_0 = 1.0 / sum_1" in code
    assert f"_helion_exdiv_lo_0, _helion_exdiv_hi_0 = {BOUNDS}sum_1)" in code
    assert code.count(FAST) == 1 and f"{FAST}v_6, sum_1, _helion_exdiv_y_0)" in code
    assert (
        "_helion_exdiv_min_0 = cute.arch.fmin(_helion_exdiv_min_0, cute.math.abs(v_6), nan=True)"
        in code
    )
    assert (
        "_helion_exdiv_max_0 = cute.arch.fmax(_helion_exdiv_max_0, cute.math.abs(v_6), nan=True)"
        in code
    )
    (copy,) = _redo_copies(code)
    # The redo copy is the original sweep: one plain divide, no helpers.
    assert "v_7 = v_6 / sum_1" in copy and ANY_EXACT not in copy
    assert code.count("/ sum_1") == 2
    # The consume sweep has no loop-carried value to roll back.
    assert "_helion_exdiv_save_" not in code


def test_jsd_hoists_every_divisor_once_per_scope_and_rolls_back_accumulators() -> None:
    student = torch.empty(128, 4096, dtype=torch.float32)
    teacher = torch.empty(128, 4096, dtype=torch.float32)
    code = _render(jsd, (0.5, 1.0, student, teacher), **JSD_CONFIG)
    # temperature (a float kernel argument) once at the top, the two row
    # sums once each after their reductions.
    assert code.count("= 1.0 / temperature") == 1
    assert code.count("= 1.0 / sum_1") == 1 and code.count("= 1.0 / sum_2") == 1
    assert code.count(BOUNDS) == 3
    # Every sweep speculates; its redo copy keeps the plain divides.
    copies = _redo_copies(code)
    assert len(copies) == 5
    assert all(ANY_EXACT not in copy for copy in copies)
    # Outside the copies the only remaining divides are the three hoisted
    # reciprocals: every site took the fast form.
    in_copies = sum(copy.count("/ temperature") for copy in copies)
    assert in_copies >= 5 and code.count("/ temperature") == in_copies + 1
    in_copies_sums = sum(
        copy.count("/ sum_1") + copy.count("/ sum_2") for copy in copies
    )
    assert in_copies_sums >= 2
    assert code.count("/ sum_1") + code.count("/ sum_2") == in_copies_sums + 2
    assert code.count(FAST) == in_copies + in_copies_sums
    # The sweeps' accumulators are snapshotted before and restored in the redo.
    saves = [
        line.strip() for line in code.splitlines() if "_helion_exdiv_save_" in line
    ]
    assert any(line.endswith("= amax_acc") for line in saves)
    assert any(line.startswith("amax_acc = _helion_exdiv_save_") for line in saves)
    assert any(line.endswith("= skl_acc") for line in saves)
    # Per-iteration temporaries are not rolled back.
    assert not any("rindex" in line or "reduction_lane" in line for line in saves)


def test_scalar_argument_and_tensor_element_divisors_in_lane_loops() -> None:
    x = torch.empty(1 << 16, dtype=torch.float32)
    code = _render(
        divide_by_scale, (x, torch.empty(1, dtype=torch.float32)), **LANE_LOOP_CONFIG
    )
    assert "_helion_exdiv_y_0 = 1.0 / load_1" in code
    assert f"{FAST}load, load_1, _helion_exdiv_y_0)" in code
    assert len(_redo_copies(code)) == 1
    code = _render(rows_div, (torch.empty(256, 1024), torch.empty(256)), **ROWS_CONFIG)
    assert code.count(FAST) == 1 and len(_redo_copies(code)) == 1


def test_element_varying_divisor_is_left_alone() -> None:
    x = torch.empty(4096, 1024, dtype=torch.float16)
    code = _render(
        divide_inside_loop,
        (x,),
        block_sizes=[1, 128],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
    )
    assert ANY_EXACT not in code and "_helion_exdiv_" not in code


def test_fast_math_keeps_its_approximate_division() -> None:
    x = torch.empty(1 << 16, dtype=torch.float32)
    code = _render(divide_by_scale_fast_math, (x, torch.empty(1)), **LANE_LOOP_CONFIG)
    assert "cute.math.div(" in code and "approx=True" in code
    assert ANY_EXACT not in code
    xh = torch.empty(4096, 4096, dtype=torch.float16)
    code = _render(softmax_fast_math, (xh,), **SOFTMAX_CONFIG)
    assert ANY_EXACT not in code


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_divisions_are_left_alone(dtype: torch.dtype) -> None:
    x = torch.empty(1 << 16, dtype=dtype)
    code = _render(
        divide_by_scale, (x, torch.empty(1, dtype=dtype)), **LANE_LOOP_CONFIG
    )
    assert ANY_EXACT not in code
    assert "v_0 = load / load_1" in code


def test_integer_divisions_are_left_alone() -> None:
    x = torch.empty(1 << 16, dtype=torch.int32)
    code = _render(int_divs, (x, torch.empty(1, dtype=torch.int32)), **LANE_LOOP_CONFIG)
    assert ANY_EXACT not in code
    assert " // " in code and "_cute_python_mod(" in code or " % " in code


def test_the_setting_and_the_pass_switch_keep_the_plain_divide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.empty(4096, 4096, dtype=torch.float16)
    code = _render(softmax_plain, (x,), **SOFTMAX_CONFIG)
    assert ANY_EXACT not in code and "v_7 = v_6 / sum_1" in code
    # The knob's plain form renders exactly what the setting off renders,
    # and so does the retired guard form (repaired to plain by the config
    # normalization: a recorded config naming it fails closed).
    assert _render(softmax, (x,), **SOFTMAX_CONFIG, cute_exact_div_form="plain") == code
    retired: dict[str, Any] = {**SOFTMAX_CONFIG, "cute_exact_div_form": "guard"}
    assert _render(softmax, (x,), **retired) == code
    spec = _bind(softmax, (x,)).config_spec
    config = helion.Config(**retired)
    spec.normalize(config)
    assert config.config["cute_exact_div_form"] == "plain"
    bogus: dict[str, Any] = {**SOFTMAX_CONFIG, "cute_exact_div_form": "bogus"}
    config = helion.Config(**bogus)
    with pytest.raises(helion.exc.InvalidConfig):
        spec.normalize(config)
    spec.normalize(config, _fix_invalid=True)
    assert config.config["cute_exact_div_form"] == "plain"
    assert spec.flatten_missing_field_default("cute_exact_div_form", {}) == (
        True,
        "redo",
    )
    # The autotuner's surface offers the two forms only.
    fragment = spec._flat_fields()["cute_exact_div_form"]
    assert tuple(fragment.choices) == ("redo", "plain")  # pyrefly: ignore
    monkeypatch.setenv("HELION_DISABLE_HOIST_RECIP", "1")
    code = _render(
        helion.kernel(backend="cute", static_shapes=True)(_softmax),
        (x,),
        **SOFTMAX_CONFIG,
    )
    assert ANY_EXACT not in code and "v_7 = v_6 / sum_1" in code


# --------------------------------------------------------------------------
# AST-level pins of the eligibility rules (no binding needed)
# --------------------------------------------------------------------------

_TENSORS = {"x": "cutlass.Float32", "out": "cutlass.Float32", "y": "cutlass.Float32"}


def _run(
    source: str,
    *,
    tensor_dtypes: dict[str, str] | None = None,
    float_scalar_names: tuple[str, ...] = (),
    disjoint: tuple[frozenset[str], ...] = (),
    rename_groups: dict[str, str] | None = None,
) -> str:
    body = ast.parse(textwrap.dedent(source)).body
    body = hoist_exact_invariant_divisions(
        body,
        rename_groups,
        float_scalar_names=float_scalar_names,
        tensor_dtypes=_TENSORS if tensor_dtypes is None else tensor_dtypes,
        argument_names=("x", "out", "y", "scale"),
        proven_disjoint_tensor_pairs=disjoint,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


_SWEEP = """\
d = cutlass.Float32(total)
acc = cutlass.Float32(0)
for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
    load = (x.iterator + cutlass.Int32(i)).load()
    v_0 = load / d
    acc = acc + cutlass.Float32(v_0)
    (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
"""


def test_redo_form_hoists_rolls_back_and_copies_the_loop() -> None:
    code = _run(_SWEEP, disjoint=(frozenset({"x", "out"}),))
    expected = """\
        d = cutlass.Float32(total)
        acc = cutlass.Float32(0)
        _helion_exdiv_y_0 = 1.0 / d
        _helion_exdiv_lo_0, _helion_exdiv_hi_0 = _cute_exact_div_bounds_f32(d)
        _helion_exdiv_min_0 = cutlass.Float32(float('inf'))
        _helion_exdiv_max_0 = cutlass.Float32(0.0)
        _helion_exdiv_save_1_acc = acc
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            load = (x.iterator + cutlass.Int32(i)).load()
            v_0 = cutlass.Float32(_cute_exact_div_fast_f32(load, d, _helion_exdiv_y_0))
            _helion_exdiv_min_0 = cute.arch.fmin(_helion_exdiv_min_0, cute.math.abs(load), nan=True)
            _helion_exdiv_max_0 = cute.arch.fmax(_helion_exdiv_max_0, cute.math.abs(load), nan=True)
            acc = acc + cutlass.Float32(v_0)
            (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
        if _cute_exact_div_redo_f32(_helion_exdiv_min_0, _helion_exdiv_max_0, _helion_exdiv_lo_0, _helion_exdiv_hi_0):
            acc = _helion_exdiv_save_1_acc
            for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
                load = (x.iterator + cutlass.Int32(i)).load()
                v_0 = load / d
                acc = acc + cutlass.Float32(v_0)
                (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
        """
    assert code == ast.unparse(ast.parse(textwrap.dedent(expected)))


_LANE_SWEEP = """\
for lane_0 in cutlass.range_constexpr(2):
    load_1 = (y.iterator + cutlass.Int32(lane_0)).load()
    acc = cutlass.Float32(0)
    for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
        load = (x.iterator + cutlass.Int32(i + lane_0)).load()
        v_0 = load / load_1
        acc = acc + cutlass.Float32(v_0)
        (out.iterator + cutlass.Int32(i + lane_0)).store(cutlass.Float32(v_0))
    (y.iterator + cutlass.Int32(lane_0)).store(acc)
"""
_LANE_DISJOINT = (
    frozenset({"x", "out"}),
    frozenset({"x", "y"}),
    frozenset({"out", "y"}),
)


def test_redo_form_inside_a_constexpr_lane_loop_rolls_back_the_lane_accumulator() -> (
    None
):
    """Several rows per thread: the sweep sits in a ``cutlass.range_constexpr``
    lane loop (Python runs it while tracing, so its bindings stay visible)
    and its accumulator is defined in the lane body.  The redo copy must
    restore that accumulator too, or a hard dividend re-accumulates on top
    of the speculative pass's partial sum."""
    code = _run(_LANE_SWEEP, disjoint=_LANE_DISJOINT)
    expected = """\
        for lane_0 in cutlass.range_constexpr(2):
            load_1 = (y.iterator + cutlass.Int32(lane_0)).load()
            acc = cutlass.Float32(0)
            _helion_exdiv_y_0 = 1.0 / load_1
            _helion_exdiv_lo_0, _helion_exdiv_hi_0 = _cute_exact_div_bounds_f32(load_1)
            _helion_exdiv_min_0 = cutlass.Float32(float('inf'))
            _helion_exdiv_max_0 = cutlass.Float32(0.0)
            _helion_exdiv_save_1_acc = acc
            for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
                load = (x.iterator + cutlass.Int32(i + lane_0)).load()
                v_0 = cutlass.Float32(_cute_exact_div_fast_f32(load, load_1, _helion_exdiv_y_0))
                _helion_exdiv_min_0 = cute.arch.fmin(_helion_exdiv_min_0, cute.math.abs(load), nan=True)
                _helion_exdiv_max_0 = cute.arch.fmax(_helion_exdiv_max_0, cute.math.abs(load), nan=True)
                acc = acc + cutlass.Float32(v_0)
                (out.iterator + cutlass.Int32(i + lane_0)).store(cutlass.Float32(v_0))
            if _cute_exact_div_redo_f32(_helion_exdiv_min_0, _helion_exdiv_max_0, _helion_exdiv_lo_0, _helion_exdiv_hi_0):
                acc = _helion_exdiv_save_1_acc
                for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
                    load = (x.iterator + cutlass.Int32(i + lane_0)).load()
                    v_0 = load / load_1
                    acc = acc + cutlass.Float32(v_0)
                    (out.iterator + cutlass.Int32(i + lane_0)).store(cutlass.Float32(v_0))
            (y.iterator + cutlass.Int32(lane_0)).store(acc)
        """
    assert code == ast.unparse(ast.parse(textwrap.dedent(expected)))


def _saved_names(code: str) -> tuple[set[str], set[str]]:
    """(names snapshotted before a speculative pass, names restored in a
    redo copy)."""
    saves: set[str] = set()
    restores: set[str] = set()
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("_helion_exdiv_save_"):
            saves.add(stripped.split(" = ", 1)[1])
        elif " = _helion_exdiv_save_" in stripped:
            restores.add(stripped.split(" = ", 1)[0])
    return saves, restores


_INNER_SWEEP = """\
for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
    load = (x.iterator + cutlass.Int32(i)).load()
    v_0 = load / load_1
    (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
    acc = acc + cutlass.Float32(v_0)
"""
_INNER_SWEEP_TWO = (
    _INNER_SWEEP + "    acc2 = acc2 + cutlass.Float32(v_0) * cutlass.Float32(v_0)\n"
)


def _nested(prefix: str, sweep: str, suffix: str, depth: int) -> str:
    return prefix + textwrap.indent(sweep, " " * depth) + suffix


@pytest.mark.parametrize(
    ("title", "source", "carried"),
    [
        (
            "two constexpr levels, the accumulator at the outer level",
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    acc = cutlass.Float32(0)\n"
                "    for vl in cutlass.range_constexpr(4):\n"
                "        load_1 = (y.iterator + cutlass.Int32(lane + vl)).load()\n",
                _INNER_SWEEP,
                "    (y.iterator + cutlass.Int32(lane)).store(acc)\n",
                8,
            ),
            {"acc"},
        ),
        (
            "two constexpr levels, the accumulator inside the inner one",
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    for vl in cutlass.range_constexpr(4):\n"
                "        load_1 = (y.iterator + cutlass.Int32(lane + vl)).load()\n"
                "        acc = cutlass.Float32(0)\n",
                _INNER_SWEEP,
                "        (y.iterator + cutlass.Int32(lane + vl)).store(acc)\n",
                8,
            ),
            {"acc"},
        ),
        (
            "two accumulators defined in the lane body",
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    acc = cutlass.Float32(0)\n"
                "    acc2 = cutlass.Float32(0)\n",
                _INNER_SWEEP_TWO,
                "    (y.iterator + cutlass.Int32(lane)).store(acc + acc2)\n",
                4,
            ),
            {"acc", "acc2"},
        ),
        (
            (
                "a staged loop inside the lane loop before the sweep: its own names"
                " are not exported, the accumulator it updates stays carried"
            ),
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    acc = cutlass.Float32(0)\n"
                "    for k in range(cutlass.Int32(0), cutlass.Int32(m), cutlass.Int32(1)):\n"
                "        tmp = cutlass.Float32(k)\n"
                "        acc = acc + tmp\n",
                _INNER_SWEEP,
                "    (y.iterator + cutlass.Int32(lane)).store(acc)\n",
                4,
            ),
            {"acc"},
        ),
        (
            "the sweep inside a dynamic loop inside the lane loop",
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    acc = cutlass.Float32(0)\n"
                "    for j in range(cutlass.Int32(0), cutlass.Int32(m), cutlass.Int32(1)):\n",
                _INNER_SWEEP,
                "    (y.iterator + cutlass.Int32(lane)).store(acc)\n",
                8,
            ),
            {"acc"},
        ),
        (
            (
                "the accumulator first bound in a cutlass.const_expr if chain (Python"
                " runs it while tracing: exported)"
            ),
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    if cutlass.const_expr(lane % 2 < 1):\n"
                "        acc = cutlass.Float32(0)\n"
                "    else:\n"
                "        acc = cutlass.Float32(0) + load_1\n",
                _INNER_SWEEP,
                "    (y.iterator + cutlass.Int32(lane)).store(acc)\n",
                4,
            ),
            {"acc"},
        ),
        (
            (
                "a temporary first bound in a dynamic if before the sweep is not"
                " exported and not rolled back"
            ),
            (
                "load_1 = (y.iterator + cutlass.Int32(0)).load()\n"
                "acc = cutlass.Float32(0)\n"
                "if m > 0:\n"
                "    tmp = cutlass.Float32(1)\n"
                "for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):\n"
                "    load = (x.iterator + cutlass.Int32(i)).load()\n"
                "    v_0 = load / load_1\n"
                "    tmp = cutlass.Float32(v_0)\n"
                "    (out.iterator + cutlass.Int32(i)).store(tmp)\n"
                "    acc = acc + tmp\n"
                "(y.iterator + cutlass.Int32(0)).store(acc)\n"
            ),
            {"acc"},
        ),
        (
            (
                "an accumulator bound before the lane loop and rebound in the lane"
                " body before the sweep"
            ),
            _nested(
                "acc = cutlass.Float32(0)\n"
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    acc = acc + load_1\n",
                _INNER_SWEEP,
                "(y.iterator + cutlass.Int32(0)).store(acc)\n",
                4,
            ),
            {"acc"},
        ),
        (
            (
                "a KERNEL PARAMETER as the accumulator: bound before any statement,"
                " so the sweep's rebinding is rolled back"
            ),
            (
                "load_1 = (y.iterator + cutlass.Int32(0)).load()\n"
                + _INNER_SWEEP.replace(
                    "    acc = acc + cutlass.Float32(v_0)\n",
                    "    scale = scale + cutlass.Float32(v_0)\n",
                )
                + "(y.iterator + cutlass.Int32(0)).store(scale)\n"
            ),
            {"scale"},
        ),
        (
            (
                "an accumulator updated only by an AUGMENTED assignment (its"
                " target is a Store the read collector does not see)"
            ),
            (
                "load_1 = (y.iterator + cutlass.Int32(0)).load()\n"
                "acc = cutlass.Float32(0)\n"
                + _INNER_SWEEP.replace(
                    "    acc = acc + cutlass.Float32(v_0)\n",
                    "    acc += cutlass.Float32(v_0)\n",
                )
                + "(y.iterator + cutlass.Int32(0)).store(acc)\n"
            ),
            {"acc"},
        ),
        (
            "a kernel parameter updated only by an augmented assignment",
            (
                "load_1 = (y.iterator + cutlass.Int32(0)).load()\n"
                + _INNER_SWEEP.replace(
                    "    acc = acc + cutlass.Float32(v_0)\n",
                    "    scale *= cutlass.Float32(v_0)\n",
                )
                + "(y.iterator + cutlass.Int32(0)).store(scale)\n"
            ),
            {"scale"},
        ),
        (
            "a lane accumulator updated only by an augmented assignment",
            _nested(
                "for lane in cutlass.range_constexpr(2):\n"
                "    load_1 = (y.iterator + cutlass.Int32(lane)).load()\n"
                "    acc = cutlass.Float32(0)\n",
                _INNER_SWEEP.replace(
                    "    acc = acc + cutlass.Float32(v_0)\n",
                    "    acc += cutlass.Float32(v_0)\n",
                ),
                "    (y.iterator + cutlass.Int32(lane)).store(acc)\n",
                4,
            ),
            {"acc"},
        ),
    ],
)
def test_redo_form_rolls_back_every_accumulator_bound_before_the_sweep(
    title: str, source: str, carried: set[str]
) -> None:
    """The names defined when the sweep starts follow the DSL's staging
    rules: the kernel's parameters are bound before any statement, unrolled
    regions (``range_constexpr`` loops, ``const_expr`` ifs) export their
    bindings, staged regions (``range`` loops, dynamic ifs) export only
    names that already existed.  Every loop-carried accumulator among them
    is snapshotted and restored, nothing else is."""
    code = _run(source, disjoint=_LANE_DISJOINT)
    assert FAST in code, (title, code)
    copies = _redo_copies(code)
    assert len(copies) == 1 and "load / load_1" in copies[0], (title, code)
    assert _saved_names(code) == (carried, carried), (title, code)


def _unchanged(source: str) -> str:
    return ast.unparse(ast.parse(textwrap.dedent(source)))


def test_unproven_aliasing_and_in_place_loops_keep_the_plain_division() -> None:
    """A loop the redo form cannot take is left exactly as written: no
    helper, and no hoisted reciprocal or window either (the hoist alone
    would add a division and two live registers for nothing)."""
    # x and out are both tensor arguments with no disjointness proof: a redo
    # could re-read what the speculative pass stored.
    code = _run(_SWEEP)
    assert code == _unchanged(_SWEEP)
    assert ANY_EXACT not in code and HOISTED not in code
    # The same tensor loaded and stored.
    source = _SWEEP.replace("(out.iterator", "(x.iterator")
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    assert code == _unchanged(source)


@pytest.mark.parametrize(
    "statement",
    [
        "    cute.arch.sync_threads()",
        "    acc = _cute_grouped_reduce_warp(acc, 'sum')",
        "    if v_0 > 0:\n        acc = acc + 1.0",
        "    scratch[i] = v_0",
        "    vals.append(v_0)",
        "    w = (y.iterator + cutlass.Int32(v_0)).load()",
    ],
)
def test_loops_the_redo_cannot_re_execute_keep_the_plain_division(
    statement: str,
) -> None:
    source = "scratch = cute.make_tensor(ptr, (4,))\nvals = []\n" + _SWEEP.replace(
        "    acc = acc + cutlass.Float32(v_0)\n",
        "    acc = acc + cutlass.Float32(v_0)\n" + statement + "\n",
    )
    code = _run(
        source,
        disjoint=(
            frozenset({"x", "out"}),
            frozenset({"x", "y"}),
            frozenset({"out", "y"}),
        ),
    )
    assert code == _unchanged(source), code
    assert ANY_EXACT not in code and HOISTED not in code


def test_register_fragments_augmented_in_the_sweep_keep_the_plain_division() -> None:
    # ``cache[0] += v`` reads and rewrites a register fragment inside the
    # sweep; a redo would accumulate into it twice and no name identifies it
    # for the rollback, so the loop keeps the plain division.
    source = (
        "cache = cute.make_rmem_tensor(16, cutlass.Float32)\n"
        "cache[0] = cutlass.Float32(0)\n"
        + _SWEEP.replace(
            "    acc = acc + cutlass.Float32(v_0)\n",
            "    acc = acc + cutlass.Float32(v_0)\n    cache[0] += cutlass.Float32(v_0)\n",
        )
    )
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    assert code == _unchanged(source), code
    assert ANY_EXACT not in code and HOISTED not in code


def test_register_fragments_written_but_not_read_keep_the_redo() -> None:
    source = "cache = cute.make_rmem_tensor(16, cutlass.Uint32)\n" + _SWEEP.replace(
        "    acc = acc + cutlass.Float32(v_0)\n",
        "    acc = acc + cutlass.Float32(v_0)\n    cache[0] = cutlass.Float32(v_0).bitcast(cutlass.Uint32)\n",
    )
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    assert REDO in code
    source = source.replace(
        "    load = (x.iterator + cutlass.Int32(i)).load()\n",
        "    load = cutlass.Uint32(cache[0]).bitcast(cutlass.Float32)\n",
    )
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    assert code == _unchanged(source)


def test_child_loops_take_the_redo_when_their_parent_cannot() -> None:
    """The outer loop has a barrier, so it cannot be re-executed; its inner
    sweep can.  The divisor's hoist sits above the outer loop (it is
    invariant there), the inner sweep speculates and rolls back the
    accumulator, and the outer loop's own quotient keeps the plain
    division."""
    source = """\
        d = cutlass.Float32(total)
        acc = cutlass.Float32(0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            for lane in cutlass.range_constexpr(4):
                load = (x.iterator + cutlass.Int32(i + lane)).load()
                v_0 = load / d
                acc = acc + cutlass.Float32(v_0)
                (out.iterator + cutlass.Int32(i + lane)).store(cutlass.Float32(v_0))
            load_2 = (y.iterator + cutlass.Int32(i)).load()
            v_1 = load_2 / d
            cute.arch.sync_threads()
            (y.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_1))
        """
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    expected = """\
        d = cutlass.Float32(total)
        acc = cutlass.Float32(0)
        _helion_exdiv_y_0 = 1.0 / d
        _helion_exdiv_lo_0, _helion_exdiv_hi_0 = _cute_exact_div_bounds_f32(d)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            _helion_exdiv_min_0 = cutlass.Float32(float('inf'))
            _helion_exdiv_max_0 = cutlass.Float32(0.0)
            _helion_exdiv_save_1_acc = acc
            for lane in cutlass.range_constexpr(4):
                load = (x.iterator + cutlass.Int32(i + lane)).load()
                v_0 = cutlass.Float32(_cute_exact_div_fast_f32(load, d, _helion_exdiv_y_0))
                _helion_exdiv_min_0 = cute.arch.fmin(_helion_exdiv_min_0, cute.math.abs(load), nan=True)
                _helion_exdiv_max_0 = cute.arch.fmax(_helion_exdiv_max_0, cute.math.abs(load), nan=True)
                acc = acc + cutlass.Float32(v_0)
                (out.iterator + cutlass.Int32(i + lane)).store(cutlass.Float32(v_0))
            if _cute_exact_div_redo_f32(_helion_exdiv_min_0, _helion_exdiv_max_0, _helion_exdiv_lo_0, _helion_exdiv_hi_0):
                acc = _helion_exdiv_save_1_acc
                for lane in cutlass.range_constexpr(4):
                    load = (x.iterator + cutlass.Int32(i + lane)).load()
                    v_0 = load / d
                    acc = acc + cutlass.Float32(v_0)
                    (out.iterator + cutlass.Int32(i + lane)).store(cutlass.Float32(v_0))
            load_2 = (y.iterator + cutlass.Int32(i)).load()
            v_1 = load_2 / d
            cute.arch.sync_threads()
            (y.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_1))
        """
    assert code == _unchanged(expected)
    # Two children that can both be re-executed each take the redo form,
    # sharing the hoist above their parent.
    source = textwrap.dedent(source)
    assert "\n    load_2 = (y.iterator + cutlass.Int32(i)).load()\n" in source
    source = source.replace(
        "    load_2 = (y.iterator + cutlass.Int32(i)).load()\n    v_1 = load_2 / d\n",
        "    for vl in cutlass.range_constexpr(2):\n"
        "        load_2 = (y.iterator + cutlass.Int32(i + vl)).load()\n"
        "        v_1 = load_2 / d\n"
        "        acc = acc + cutlass.Float32(v_1)\n",
    ).replace(
        "    (y.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_1))\n",
        "    (y.iterator + cutlass.Int32(i)).store(acc)\n",
    )
    assert "for vl in" in source and "store(acc)" in source
    code = _run(source, disjoint=_LANE_DISJOINT)
    _exact_names_defined_before_use(code)
    assert code.count("= 1.0 / d") == 1 and code.count(FAST) == 2
    assert code.count(REDO) == 2 and len(_redo_copies(code)) == 2
    assert "v_1 = load_2 / d" in code and "v_0 = load / d" in code
    assert _saved_names(code) == ({"acc"}, {"acc"})


def test_sibling_loops_share_one_hoist_and_rebinding_starts_another() -> None:
    source = _SWEEP + _SWEEP.replace("for i", "for j").replace("v_0", "v_1")
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    # ``d`` is rebound between the sweeps (the second ``d = ...``): two hoists.
    assert code.count("= 1.0 / d") == 2
    source = _SWEEP + _SWEEP.replace("for i", "for j").replace("v_0", "v_1").replace(
        "d = cutlass.Float32(total)\nacc = cutlass.Float32(0)\n", ""
    )
    code = _run(source, disjoint=(frozenset({"x", "out"}),))
    assert code.count("= 1.0 / d") == 1 and code.count(REDO) == 2


def test_only_proved_fp32_scalar_operands_are_rewritten() -> None:
    # The divisor is a tensor element of unknown dtype.
    code = _run(
        _SWEEP, tensor_dtypes={"x": "cutlass.Float16", "out": "cutlass.Float16"}
    )
    assert ANY_EXACT not in code and "v_0 = load / d" in code
    # The divisor is a kernel argument proved fp32 only through
    # ``float_scalar_names``.
    source = _SWEEP.replace("d = cutlass.Float32(total)\n", "").replace(
        "/ d", "/ scale"
    )
    assert ANY_EXACT not in _run(source, disjoint=(frozenset({"x", "out"}),))
    code = _run(
        source, float_scalar_names=("scale",), disjoint=(frozenset({"x", "out"}),)
    )
    assert "_helion_exdiv_y_0 = 1.0 / scale" in code and FAST in code
    # Not a generic-lowering temporary (a hand-emitted name), a reciprocal.
    code = _run(_SWEEP.replace("v_0", "flash_q"), disjoint=(frozenset({"x", "out"}),))
    assert ANY_EXACT not in code
    code = _run(
        _SWEEP.replace("load / d", "cutlass.Float32(1.0) / d"),
        disjoint=(frozenset({"x", "out"}),),
    )
    assert ANY_EXACT not in code


_PACKET_PREFIX = """\
d = cutlass.Float32(total)
acc = cutlass.Float32(0)
for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
    mask = i < cutlass.Int32(n)
    _unroll_vec = cute.arch.load(x.iterator + cutlass.Int32(i), ir.VectorType.get([4], cutlass.Uint32.mlir_type))
"""
_PACKET_SUFFIX = """\
    _mask_to = cutlass.Float32(v_0) if mask else cutlass.Float32(0)
    acc = acc + cutlass.Float32(_mask_to)
"""
_PACKET_SELECT = (
    "    load = cutlass.Uint32(_unroll_vec[0]).bitcast(cutlass.Float32)"
    " if mask else cutlass.Float32(0)\n"
)


def test_a_select_after_the_quotient_keeps_the_redo() -> None:
    """The masked contribution of a reduction (``v if mask else 0`` AFTER
    the quotient) is the plain lowering's own shape and no operand of the
    division: the sweep takes the redo form."""
    source = (
        _PACKET_PREFIX
        + "    load = cutlass.Uint32(_unroll_vec[0]).bitcast(cutlass.Float32)\n"
        + "    v_0 = load / d\n"
        + _PACKET_SUFFIX
    )
    code = _run(source)
    assert FAST in code and len(_redo_copies(code)) == 1, code
    assert _saved_names(code) == ({"acc"}, {"acc"})


@pytest.mark.parametrize(
    ("title", "body"),
    [
        (
            "the packet form of a masked load",
            _PACKET_SELECT + "    v_0 = load / d\n",
        ),
        (
            "the scalar form of a masked load",
            (
                "    load = (x.iterator + cutlass.Int32(i)).load() if mask else"
                " cutlass.Float32(0)\n    v_0 = load / d\n"
            ),
        ),
        (
            "the select inline",
            (
                "    v_0 = (cutlass.Uint32(_unroll_vec[0]).bitcast(cutlass.Float32)"
                " if mask else cutlass.Float32(0)) / d\n"
            ),
        ),
        (
            "derived through arithmetic",
            _PACKET_SELECT
            + "    tmp = load * cutlass.Float32(2.0)\n    v_0 = tmp / d\n",
        ),
        (
            "derived through a cutlass.Float32 conversion",
            _PACKET_SELECT + "    tmp = cutlass.Float32(load)\n    v_0 = tmp / d\n",
        ),
        (
            "a half-precision packet converted after the select",
            (
                "    load = cutlass.Uint16(_unroll_vec[0]).bitcast(cutlass.Float16)"
                " if mask else cutlass.Float16(0)\n"
                "    tmp = load.to(cutlass.Float32)\n    v_0 = tmp / d\n"
            ),
        ),
        (
            "derived through a cute.math call",
            _PACKET_SELECT + "    tmp = cute.math.exp2(load)\n    v_0 = tmp / d\n",
        ),
    ],
)
def test_masked_select_operands_keep_the_plain_division(title: str, body: str) -> None:
    """A dividend that is, or descends from, a masked select -- in any
    spelling, through arithmetic, conversions (which ``_kind`` takes as
    fp32 whatever their operand) and math calls -- keeps the plain division
    and the render is left exactly as written.  The masked zero would make
    the redo fire for the tail threads of a ragged axis only, and the kernel
    with both loop copies live compiled the fp32 chains around the quotient
    differently from the plain kernel (the fourth review's config 43)."""
    source = _PACKET_PREFIX + body + _PACKET_SUFFIX
    code = _run(source)
    assert code == _unchanged(source), (title, code)
    assert ANY_EXACT not in code and HOISTED not in code


def test_a_register_cache_written_from_a_select_taints_its_reloads() -> None:
    """Codegen's ``_fuse_cache`` reloads are masked selects themselves; an
    unmasked reload of a cache a masked value was stored to is tainted
    through the store (``cache[k] = ...``) and keeps the plain division."""
    source = (
        "cache = cute.make_rmem_tensor(16, cutlass.Uint32)\n"
        + _PACKET_PREFIX
        + _PACKET_SELECT
        + "    cache[0] = cutlass.Float32(load).bitcast(cutlass.Uint32)\n"
        "for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):\n"
        "    tmp = cutlass.Uint32(cache[0]).bitcast(cutlass.Float32)\n"
        "    v_0 = tmp / d\n"
        "    acc = acc + cutlass.Float32(v_0)\n"
    )
    code = _run(source)
    assert code == _unchanged(source), code
    assert ANY_EXACT not in code and HOISTED not in code


def test_a_masked_divisor_keeps_the_redo() -> None:
    """The divisor is one value for every thread of the sweep (a row scale,
    a row sum): a masked one (``s[tile_m]`` loaded as a masked packet for a
    partial row block) fires the redo for all of them alike, which mixes
    nothing, so only the dividend is held to the select rule."""
    source = (
        "_dvec = cute.arch.load(y.iterator, ir.VectorType.get([4], cutlass.Uint32.mlir_type))\n"
        "d = cutlass.Uint32(_dvec[0]).bitcast(cutlass.Float32) if m < n else cutlass.Float32(0)\n"
        + _SWEEP.replace("d = cutlass.Float32(total)\n", "")
    )
    code = _run(source, disjoint=_LANE_DISJOINT)
    assert FAST in code and len(_redo_copies(code)) == 1, code
    assert "_helion_exdiv_y_0 = 1.0 / d" in code


@pytest.mark.parametrize(
    ("title", "source"),
    [
        (
            "a clamped prefetch address (the pipelined loads)",
            (
                "d = cutlass.Float32(total)\n"
                "acc = cutlass.Float32(0)\n"
                "_pipe_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 4\n"
                "_pipe_vec = cute.arch.load(x.iterator + cutlass.Int32(_pipe_base), ir.VectorType.get([4], cutlass.Uint32.mlir_type))\n"
                "for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):\n"
                "    _vec = _pipe_vec\n"
                "    _pipe_base = i + cutlass.Int32(16) + cutlass.Int32(cute.arch.thread_idx()[0]) * 4 if i + cutlass.Int32(16) < cutlass.Int32(n) else _pipe_base\n"
                "    _pipe_vec = cute.arch.load(x.iterator + cutlass.Int32(_pipe_base) if _pipe_base < cutlass.Int32(n) else x.iterator + cutlass.Int32(0), ir.VectorType.get([4], cutlass.Uint32.mlir_type))\n"
                "    load = cutlass.Uint32(_vec[0]).bitcast(cutlass.Float32)\n"
                "    v_0 = load / d\n"
                "    acc = acc + cutlass.Float32(v_0)\n"
            ),
        ),
        (
            "a select whose value passes through a cross-thread reduction",
            (
                "d = cutlass.Float32(total)\n"
                "acc = cutlass.Float32(0)\n"
                "where = cutlass.Float32(peek) if full else cutlass.Float32(float('-inf'))\n"
                "mi = cutlass.Float32(_cute_grouped_reduce_warp(where, 'max', cutlass.Float32(float('-inf'))))\n"
                "for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):\n"
                "    load = (x.iterator + cutlass.Int32(i)).load()\n"
                "    tmp = cute.math.exp2(load - mi)\n"
                "    v_0 = tmp / d\n"
                "    acc = acc + cutlass.Float32(v_0)\n"
            ),
        ),
    ],
)
def test_address_selects_and_reductions_do_not_taint(title: str, source: str) -> None:
    """A select on an ADDRESS (where a packet is read, not what it holds)
    and a select whose value reaches the dividend only through a
    cross-thread reduction (row-uniform afterwards: the online softmax's
    first-tile max) leave the sweep its redo form."""
    code = _run(source)
    assert FAST in code and len(_redo_copies(code)) == 1, (title, code)
    saves, restores = _saved_names(code)
    # The pipelined prefetch state is loop-carried too and rolled back.
    assert "acc" in saves and saves == restores, (title, code)


def test_renamed_loop_carried_divisor_is_not_invariant() -> None:
    """``v_8`` renames to ``di`` after the pass: a sweep that updates ``v_8``
    divides by a value that changes every iteration."""
    source = """\
        di = cutlass.Float32(0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            load = (x.iterator + cutlass.Int32(i)).load()
            v_0 = load / di
            v_8 = di + cutlass.Float32(v_0)
        """
    code = _run(source, rename_groups={"v_8": "di"})
    assert ANY_EXACT not in code


def test_loop_carried_alias_divisor_is_not_invariant() -> None:
    """``d = e`` at the end of the loop rebinds the divisor for the next
    iteration: the first iteration divides by the pre-loop ``d``, the rest
    by ``e``.  Neither spelling (direct, or through a rename group) is
    hoisted, and the per-iteration snapshot ``d_copy_0`` is no invariant
    either when its root is loop-carried."""
    source = """\
        d = cutlass.Float32(s0)
        e = cutlass.Float32(s1)
        acc = cutlass.Float32(0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            d_copy_0 = d
            load = (x.iterator + cutlass.Int32(i)).load()
            v_0 = load / d_copy_0
            acc = acc + cutlass.Float32(v_0)
            (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
            d = e
        """
    disjoint = (frozenset({"x", "out"}),)
    code = _run(source, disjoint=disjoint)
    assert ANY_EXACT not in code and "v_0 = load / d_copy_0" in code
    code = _run(
        source.replace("    d = e\n", "    v_9 = e\n"),
        rename_groups={"v_9": "d"},
        disjoint=disjoint,
    )
    assert ANY_EXACT not in code
    # A read before the write within the iteration is loop-carried too.
    source = """\
        d = cutlass.Float32(s0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            load = (x.iterator + cutlass.Int32(i)).load()
            v_0 = load / d2
            d2 = d
            (out.iterator + cutlass.Int32(i)).store(cutlass.Float32(v_0))
        """
    assert ANY_EXACT not in _run(source, disjoint=disjoint)


def test_per_iteration_copy_of_an_invariant_resolves_to_its_root() -> None:
    """``d_copy_0 = d`` at the top of the iteration precedes every read:
    the lane loop's divisor resolves to ``d`` and the hoist sits above the
    outer loop.  A copy written inside a branch does not dominate its
    reads; the lane loop then hoists the copy itself, inside the outer
    loop."""
    source = """\
        d = cutlass.Float32(s0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            d_copy_0 = d
            for lane in cutlass.range_constexpr(4):
                load = (x.iterator + cutlass.Int32(i + lane)).load()
                v_0 = load / d_copy_0
                (out.iterator + cutlass.Int32(i + lane)).store(cutlass.Float32(v_0))
        """
    disjoint = (frozenset({"x", "out"}),)
    code = _run(source, disjoint=disjoint)
    assert code.index("_helion_exdiv_y_0 = 1.0 / d\n") < code.index("for i in")
    assert code.count(FAST) == 1 and code.count(REDO) == 1
    code = _run(
        source.replace(
            "            d_copy_0 = d\n",
            "            if i > 0:\n                d_copy_0 = d\n",
        ),
        disjoint=disjoint,
    )
    assert "= 1.0 / d\n" not in code
    assert code.index("for i in") < code.index("_helion_exdiv_y_0 = 1.0 / d_copy_0")


def _exact_names_defined_before_use(code: str) -> None:
    """Every ``_helion_exdiv_*`` name is bound before the statement that
    reads it, in program order: the DSL traces the body top to bottom, and
    a sweep reading a name another sweep's hoist defines later is a
    ``NameError`` at trace time."""
    bound: set[str] = set()

    def reads(node: ast.AST) -> set[str]:
        return {
            sub.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name)
            and isinstance(sub.ctx, ast.Load)
            and sub.id.startswith("_helion_exdiv_")
        }

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                walk(stmt.body)
                continue
            if isinstance(stmt, ast.For):
                header: list[ast.AST] = [stmt.iter]
            elif isinstance(stmt, (ast.If, ast.While)):
                header = [stmt.test]
            elif isinstance(stmt, ast.With):
                header = list(stmt.items)
            else:
                header = [stmt]
            missing = {name for node in header for name in reads(node)} - bound
            assert not missing, (missing, ast.unparse(stmt).splitlines()[0])
            if isinstance(stmt, (ast.For, ast.If, ast.While, ast.With)):
                if isinstance(stmt, ast.For):
                    bound.update(
                        node.id
                        for node in ast.walk(stmt.target)
                        if isinstance(node, ast.Name)
                    )
                walk(stmt.body)
                walk(getattr(stmt, "orelse", []))
            else:
                bound.update(
                    node.id
                    for node in ast.walk(stmt)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                )

    walk(ast.parse(textwrap.dedent(code)).body)


def test_statement_objects_shared_between_sweeps_are_rewritten_per_sweep() -> None:
    """Codegen emits a reduction sweep and its rematerialized store sweep
    with the SAME statement objects (and body list).  Each sweep is
    rewritten in its own context -- the first loads its divisor per
    iteration (hoist inside it), the second divides by the value loaded
    before it (hoist above it) -- so no sweep reads the other's hoisted
    names."""
    source = """\
        acc = cutlass.Float32(0)
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            d = (y.iterator + cutlass.Int32(i)).load()
            for lane in cutlass.range_constexpr(4):
                load = (x.iterator + cutlass.Int32(i + lane)).load()
                v_0 = load / d
                acc = acc + cutlass.Float32(v_0)
                (out.iterator + cutlass.Int32(i + lane)).store(cutlass.Float32(v_0))
        d = (y.iterator + cutlass.Int32(0)).load()
        for i in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(16)):
            for lane in cutlass.range_constexpr(4):
                pass
        """
    disjoint = (
        frozenset({"x", "out"}),
        frozenset({"x", "y"}),
        frozenset({"out", "y"}),
    )
    for share_list in (False, True):
        body = ast.parse(textwrap.dedent(source)).body
        reduction, store = body[1], body[3]
        assert isinstance(reduction, ast.For) and isinstance(store, ast.For)
        lane_loop, store_lane_loop = reduction.body[1], store.body[0]
        assert isinstance(lane_loop, ast.For) and isinstance(store_lane_loop, ast.For)
        store_lane_loop.body = lane_loop.body if share_list else list(lane_loop.body)
        body = hoist_exact_invariant_divisions(
            body,
            None,
            tensor_dtypes=_TENSORS,
            argument_names=("x", "out", "y"),
            proven_disjoint_tensor_pairs=disjoint,
        )
        code = ast.unparse(ast.Module(body=body, type_ignores=[]))
        _exact_names_defined_before_use(code)
        assert code.count("= 1.0 / d") == 2 and code.count(FAST) == 2
        first, second = code.split("d = (y.iterator + cutlass.Int32(0)).load()")
        names_first = set(re.findall(r"_helion_exdiv_y_\d+", first))
        names_second = set(re.findall(r"_helion_exdiv_y_\d+", second))
        assert len(names_first) == 1 and len(names_second) == 1
        assert names_first.isdisjoint(names_second)
        # The reduction sweep's hoist follows its per-iteration load.
        assert first.index("d = (y.iterator + cutlass.Int32(i)).load()") < first.index(
            f"{names_first.pop()} = 1.0 / d"
        )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def test_loop_carried_divisor_kernels_hoist_only_the_iteration_snapshot() -> None:
    """The nested-loop kernels whose divisor changes between inner
    iterations: the only hoists sit inside the inner tile loop, after the
    iteration's snapshot of the divisor, never above it."""
    x = torch.empty(64, 2048)
    s = torch.empty(64)
    for kernel, args in (
        (carried_alias, (x, s, s.clone())),
        (carried_acc, (x, s)),
    ):
        code = _render(kernel, args, **NESTED_CONFIG)
        _exact_names_defined_before_use(code)
        lines = code.splitlines()
        (tile_loop,) = [
            i for i, line in enumerate(lines) if "for tile_offset_1 in" in line
        ]
        hoists = [i for i, line in enumerate(lines) if "= 1.0 / " in line]
        assert hoists, code
        for index in hoists:
            assert index > tile_loop and _indent(lines[index]) > _indent(
                lines[tile_loop]
            ), lines[index]


def _without_exact_additions(code: str) -> str:
    """The render with the hoisted statements, the trackers and the redo
    copies removed and every helper call mapped back to the divide it
    replaced: equal to the plain render iff the hoist changed nothing else
    (in particular not the structure another pass gave the sweeps)."""
    kept: list[str] = []
    skip_indent: int | None = None
    for line in code.splitlines():
        stripped = line.strip()
        indent = _indent(line)
        if skip_indent is not None and stripped and indent > skip_indent:
            continue
        skip_indent = None
        if stripped.startswith(REDO):
            skip_indent = indent
            continue
        if stripped.startswith("_helion_exdiv_"):
            continue
        line = re.sub(
            r"cutlass\.Float32\(_cute_exact_div_fast_f32\((\w+), (\w+), _helion_exdiv_y_\d+\)\)",
            r"\1 / \2",
            line,
        )
        kept.append(line)
    return "\n".join(kept)


def test_independent_reduction_keeps_its_partial_accumulators() -> None:
    """``cute_independent_reduction`` splits each rolled sweep into four
    partial accumulators; the hoist runs after it and leaves that structure
    (and so the summation order) exactly as the plain render has it."""
    x = torch.empty(4096, 4096, dtype=torch.float16)
    plain = _render(softmax_plain, (x,), **INDEPENDENT_SOFTMAX_CONFIG)
    assert "_cute_rolled_sum_" in plain
    exact = _render(softmax, (x,), **INDEPENDENT_SOFTMAX_CONFIG)
    assert ANY_EXACT in exact
    assert _without_exact_additions(exact) == plain
    student = torch.empty(128, 4096, dtype=torch.float32)
    args = (0.5, 1.0, student, student.clone())
    plain = _render(jsd_plain, args, **INDEPENDENT_JSD_CONFIG)
    assert "_cute_rolled_sum_" in plain
    exact = _render(jsd, args, **INDEPENDENT_JSD_CONFIG)
    assert ANY_EXACT in exact
    assert _without_exact_additions(exact) == plain


def test_several_rows_per_thread_renders_roll_back_the_lane_accumulators() -> None:
    """Every rows kernel under every several-rows config: the plain render
    nests the sweeps in a ``cutlass.range_constexpr`` lane loop, the redo
    render snapshots and restores each sweep's accumulator and keeps the
    plain render's structure."""
    x = torch.empty(64, 4096, dtype=torch.float32)
    s = torch.empty(64, dtype=torch.float32)
    for kernel, plain_kernel in ROWS_KERNELS:
        for config in SEVERAL_ROWS_CONFIGS:
            assert _several_rows_per_thread(config)
            plain = _render(plain_kernel, (x, s), **config)
            assert "cutlass.range_constexpr" in plain, (kernel.fn.__name__, config)
            exact = _render(kernel, (x, s), **config)
            assert ANY_EXACT in exact, (kernel.fn.__name__, config)
            _exact_names_defined_before_use(exact)
            assert _without_exact_additions(exact) == plain, (
                kernel.fn.__name__,
                config,
            )
            saves, restores = _saved_names(exact)
            assert saves and saves == restores, (kernel.fn.__name__, config, exact)
            assert all(name.endswith("_acc") for name in saves), saves


def test_ragged_shapes_render_the_plain_kernel() -> None:
    """Ragged shapes (a partial row block, a reduction axis the lane sweeps
    do not divide) put ``if mask else`` selects into the sweeps: every
    packet load and register-cache reload is one, and every operand of
    every division descends from one (the rows kernels' loads directly,
    JSD's and softmax's exponentials and row sums through the arithmetic).
    The pass fails closed on every such operand, whatever the spelling, so
    the render IS the plain render: no helper, no hoisted reciprocal, for
    every rows kernel, JSD and softmax under every config -- the
    several-rows configs, the one-row configs and the one-row packet family
    (corpus-4 config 43).  The same packet configs on an exact axis take
    the redo form: the pin is about the masked tail, not the config."""
    x = torch.empty(RAGGED_ROWS, RAGGED_COLS, dtype=torch.float32)
    s = torch.empty(RAGGED_ROWS, dtype=torch.float32)
    configs = (
        *SEVERAL_ROWS_CONFIGS,
        ROWS_CONFIG,
        INDEPENDENT_ROWS_CONFIG,
        *ONE_ROW_PACKET_CONFIGS,
    )
    for kernel, plain_kernel in ROWS_KERNELS:
        for config in configs:
            plain = _render(plain_kernel, (x, s), **config)
            assert " else " in plain, (kernel.fn.__name__, config)
            assert _render(kernel, (x, s), **config) == plain, (
                kernel.fn.__name__,
                config,
            )
    xh = torch.empty(RAGGED_ROWS, RAGGED_COLS, dtype=torch.float16)
    for config in (SOFTMAX_CONFIG, *ONE_ROW_PACKET_CONFIGS):
        plain = _render(softmax_plain, (xh,), **config)
        assert " else " in plain, config
        assert _render(softmax, (xh,), **config) == plain, config
    args = (0.3, 2.0, x, x.clone())
    for config in (JSD_CONFIG, INDEPENDENT_JSD_CONFIG, *JSD_RAGGED_CONFIGS):
        plain = _render(jsd_plain, args, **config)
        assert " else " in plain, config
        assert _render(jsd, args, **config) == plain, config
    # The control: on an exact axis the one-row packet configs take the
    # redo form in every rows kernel, rolling the sweep accumulators back.
    x = torch.empty(64, 4096, dtype=torch.float32)
    s = torch.empty(64, dtype=torch.float32)
    for kernel, plain_kernel in ROWS_KERNELS:
        for config in map(_on_exact_axis, ONE_ROW_PACKET_CONFIGS):
            plain = _render(plain_kernel, (x, s), **config)
            assert " else " not in plain, (kernel.fn.__name__, config)
            exact = _render(kernel, (x, s), **config)
            assert REDO in exact, (kernel.fn.__name__, config)
            assert _without_exact_additions(exact) == plain, (
                kernel.fn.__name__,
                config,
            )
            _exact_names_defined_before_use(exact)
            saves, restores = _saved_names(exact)
            assert saves and saves == restores, (kernel.fn.__name__, config)


# --------------------------------------------------------------------------
# GPU: bit identity
# --------------------------------------------------------------------------


def _float_from_parts(
    exponent: torch.Tensor, significand: torch.Tensor, sign: torch.Tensor
) -> torch.Tensor:
    bits = (
        (sign.to(torch.int32) << 31)
        | (exponent.to(torch.int32) << 23)
        | significand.to(torch.int32)
    )
    return bits.view(torch.float32)


@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestExactDivisionCuda(TestCase):
    def _check_rows(self, x: torch.Tensor, s: torch.Tensor) -> None:
        code, out = code_and_output(rows_div, (x, s), **ROWS_CONFIG)
        assert REDO in code, code
        expected = x / s[:, None]
        if not torch.equal(out.view(torch.int32), expected.view(torch.int32)):
            bad = (out.view(torch.int32) != expected.view(torch.int32)).nonzero()[:8]
            self.fail(
                f"{len(bad)}+ mismatches, first {bad.tolist()}: a={x[bad[:, 0], bad[:, 1]].tolist()} b={s[bad[:, 0]].tolist()}"
            )

    def test_every_exponent_pair_with_random_significands(self) -> None:
        """256 x 256 biased-exponent pairs (zero/subnormal and inf/NaN fields
        included), random significands and signs: 2048 divisors x 65536
        dividends = 134M quotients, in four row chunks."""
        g = torch.Generator(device=DEVICE).manual_seed(0)
        cols = 65536
        a_exp = torch.arange(256, device=DEVICE).repeat_interleave(cols // 256)
        for chunk in range(4):
            rows = 512
            b_exp = (torch.arange(64, device=DEVICE) + 64 * chunk).repeat_interleave(
                rows // 64
            )
            b = _float_from_parts(
                b_exp,
                torch.randint(0, 1 << 23, (rows,), device=DEVICE, generator=g),
                torch.randint(0, 2, (rows,), device=DEVICE, generator=g),
            )
            a = _float_from_parts(
                a_exp.expand(rows, cols),
                torch.randint(0, 1 << 23, (rows, cols), device=DEVICE, generator=g),
                torch.randint(0, 2, (rows, cols), device=DEVICE, generator=g),
            )
            self._check_rows(a.contiguous(), b)

    def test_dense_random_bit_patterns(self) -> None:
        g = torch.Generator(device=DEVICE).manual_seed(1)
        rows, cols = 1024, 65536
        a = (
            torch.randint(
                -(1 << 31),
                (1 << 31) - 1,
                (rows, cols),
                device=DEVICE,
                dtype=torch.int64,
                generator=g,
            )
            .to(torch.int32)
            .view(torch.float32)
        )
        b = (
            torch.randint(
                -(1 << 31),
                (1 << 31) - 1,
                (rows,),
                device=DEVICE,
                dtype=torch.int64,
                generator=g,
            )
            .to(torch.int32)
            .view(torch.float32)
        )
        self._check_rows(a, b)

    def test_special_values_and_near_midpoint_quotients(self) -> None:
        g = torch.Generator(device=DEVICE).manual_seed(2)
        cols = 4096
        specials = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0,
                0.5,
                float("inf"),
                -float("inf"),
                float("nan"),
                torch.finfo(torch.float32).max,
                torch.finfo(torch.float32).tiny,
                2.0**-149,
                3.0,
                1.0 / 3.0,
                2.0**126,
                2.0**-126,
            ],
            device=DEVICE,
        )
        powers = torch.exp2(torch.arange(-140.0, 128.0, device=DEVICE))
        b = torch.cat(
            [specials, powers, torch.randn(256, device=DEVICE, generator=g) * 10]
        )
        rows = b.numel()
        a = torch.randn(rows, cols, device=DEVICE, generator=g) * torch.exp2(
            torch.randint(-60, 60, (rows, cols), device=DEVICE, generator=g).float()
        )
        a[:, : specials.numel()] = specials
        a[:, 20] = b  # a == b
        # Quotients next to a rounding boundary: a = RN(m * b) with m a
        # midpoint of the fp32 grid (24 significant bits plus a half ulp).
        m_bits = (
            torch.randint(0, 1 << 23, (rows, 1024), device=DEVICE, generator=g).to(
                torch.int32
            )
            | (127 << 23)
        ).view(torch.float32)
        m = m_bits.double() + 2.0**-24
        a[:, 32:1056] = (m * b.double()[:, None]).float()
        self._check_rows(a.contiguous(), b)

    def test_softmax_rows_match_the_plain_render(self) -> None:
        """fp16 rows (the softmax_acf cells): the wave-5 input classes."""
        n, m = 256, 4096
        g = torch.Generator().manual_seed(0)

        def randn() -> torch.Tensor:
            return torch.randn(n, m, generator=g)

        cases = {
            "randn": randn(),
            "scale_1e4": randn() * 1e4,
            "offset_6e4": randn() + 6e4,
        }
        x = randn()
        x[:, ::3] = float("-inf")
        cases["neginf_columns"] = x
        x = randn()
        x[0, :] = float("-inf")
        cases["neginf_row"] = x
        x = randn()
        x[1, 5] = float("inf")
        cases["posinf"] = x
        x = randn()
        x[2, 7] = float("nan")
        cases["nan"] = x
        x = randn()
        x[3, :] = 0.0
        x[3, 0] = -0.0
        cases["signed_zero"] = x
        cases["wide"] = randn() * 30
        x = randn()
        x[4, :] = -65504.0
        x[4, 1] = 65504.0
        cases["fp16_extremes"] = x
        for name, case in cases.items():
            xh = case.to(torch.float16).to(DEVICE)
            code, out = code_and_output(softmax, (xh,), **SOFTMAX_CONFIG)
            assert REDO in code
            code0, out0 = code_and_output(softmax_plain, (xh,), **SOFTMAX_CONFIG)
            assert ANY_EXACT not in _kernel_body(code0)
            with self.subTest(case=name):
                self.assertTrue(
                    torch.equal(out.view(torch.int16), out0.view(torch.int16))
                )
        xh = cases["randn"].to(torch.float16).to(DEVICE)
        _code, out = code_and_output(softmax, (xh,), **SOFTMAX_CONFIG)
        torch.testing.assert_close(
            out, torch.softmax(xh.float(), dim=1).half(), rtol=0, atol=1e-3
        )

    def test_jsd_rows_match_the_plain_render(self) -> None:
        """fp32 rows with a scalar temperature and two row sums per row."""
        g = torch.Generator().manual_seed(0)
        n, m = 64, 4096

        def randn() -> torch.Tensor:
            return torch.randn(n, m, generator=g)

        cases = {"randn": (randn(), randn()), "scale_1e30": (randn() * 1e30, randn())}
        s = randn()
        s[0, :] = -1e30
        s[1, 5] = float("nan")
        s[2, ::7] = float("-inf")
        cases["hard_rows"] = (s, randn())
        for name, (student, teacher) in cases.items():
            args = (0.3, 2.0, student.to(DEVICE), teacher.to(DEVICE))
            code, out = code_and_output(jsd, args, **JSD_CONFIG)
            assert REDO in code and "_helion_exdiv_save_" in code
            code0, out0 = code_and_output(jsd_plain, args, **JSD_CONFIG)
            assert ANY_EXACT not in _kernel_body(code0)
            with self.subTest(case=name):
                self.assertTrue(
                    torch.equal(out.view(torch.int32), out0.view(torch.int32))
                )

    def test_configs_do_not_change_the_quotients(self) -> None:
        """The hoist fires only in the loops a config creates; every config
        divides exactly like eager (a ragged length: the last block is
        partial)."""
        torch.manual_seed(0)
        x = torch.randn((1 << 16) - 1000, device=DEVICE) * 4
        x[::97] = 0.0
        x[5::101] = 1e-40
        s = torch.tensor([0.0517], device=DEVICE)
        for num_threads in (1024, 128):
            _code, out = code_and_output(
                divide_by_scale, (x, s), block_sizes=[1024], num_threads=[num_threads]
            )
            self.assertTrue(torch.equal(out, x / s[0]))

    def test_loops_the_redo_cannot_take_keep_the_plain_division(self) -> None:
        """A loop the redo cannot re-execute -- it stores into the tensor it
        loads, so a second pass would read its own results -- renders
        exactly as the plain kernel does (no helper, no hoist) and divides
        like eager."""

        def _normalize_rows_in_place(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            for tile_m in hl.tile(m):
                x[tile_m, :] = x[tile_m, :] / s[tile_m][:, None]
            return x

        hoisted = helion.kernel(backend="cute", static_shapes=True)(
            _normalize_rows_in_place
        )
        plain = helion.kernel(
            backend="cute", static_shapes=True, cute_exact_division=False
        )(_normalize_rows_in_place)
        x, s = _hard_rows(DEVICE, RAGGED_ROWS, 1000)
        code, out = code_and_output(hoisted, (x.clone(), s), **ROWS_CONFIG)
        code0, out0 = code_and_output(plain, (x.clone(), s), **ROWS_CONFIG)
        body = _kernel_body(code)
        assert ANY_EXACT not in body and HOISTED not in body, body
        assert body == _kernel_body(code0)
        self.assertTrue(torch.equal(out.view(torch.int32), out0.view(torch.int32)))
        self.assertTrue(
            torch.equal(out0.view(torch.int32), (x / s[:, None]).view(torch.int32))
        )

    def test_loop_carried_divisors_match_the_plain_render(self) -> None:
        """A divisor rebound inside the inner loop -- from another row
        value, or from its own accumulated quotients -- is not invariant
        there; only the iteration's snapshot is hoisted, and the result
        equals the plain render (and eager, for the rebinding)."""
        torch.manual_seed(0)
        m, n, block = 64, 2048, NESTED_CONFIG["block_sizes"][1]
        x = torch.randn(m, n, device=DEVICE)
        s = torch.rand(m, device=DEVICE) + 0.5
        t = torch.rand(m, device=DEVICE) + 2.0
        code, out = code_and_output(carried_alias, (x, s, t), **NESTED_CONFIG)
        assert ANY_EXACT in _kernel_body(code)
        expected = x.clone()
        expected[:, :block] = x[:, :block] / s[:, None]
        expected[:, block:] = x[:, block:] / t[:, None]
        self.assertTrue(torch.equal(out.view(torch.int32), expected.view(torch.int32)))
        _code, out0 = code_and_output(carried_alias_plain, (x, s, t), **NESTED_CONFIG)
        self.assertTrue(torch.equal(out.view(torch.int32), out0.view(torch.int32)))
        code, out = code_and_output(carried_acc, (x, s), **NESTED_CONFIG)
        assert ANY_EXACT in _kernel_body(code)
        _code, out0 = code_and_output(carried_acc_plain, (x, s), **NESTED_CONFIG)
        self.assertTrue(torch.equal(out.view(torch.int32), out0.view(torch.int32)))

    def test_shared_sweep_statements_compile_and_match_for_the_redo_form(
        self,
    ) -> None:
        """The quotient feeds a store and a reduction: codegen shares the
        statement objects between the two sweeps.  The redo render compiles
        and matches the plain render bit for bit, hard dividends included."""
        x, s = _hard_rows(DEVICE, 64, 2048)
        code0, (out0, res0) = code_and_output(
            sum_and_store_plain, (x, s), **NESTED_CONFIG
        )
        assert ANY_EXACT not in _kernel_body(code0)
        self.assertTrue(
            torch.equal(out0.view(torch.int32), (x / s[:, None]).view(torch.int32))
        )
        code, (out, res) = code_and_output(sum_and_store, (x, s), **NESTED_CONFIG)
        body = _kernel_body(code)
        assert REDO in body
        _exact_names_defined_before_use(body)
        self.assertTrue(torch.equal(out.view(torch.int32), out0.view(torch.int32)))
        self.assertTrue(torch.equal(res.view(torch.int32), res0.view(torch.int32)))

    def _forms_match_the_plain_render(
        self,
        kernel: helion.Kernel,
        plain: helion.Kernel,
        args: tuple[object, ...],
        configs: list[dict[str, Any]],
        *,
        redo_markers: tuple[str, ...] = (),
    ) -> list[int]:
        """The redo form equals the plain render bit for bit on every
        config; ``redo_markers`` must appear in every redo render.  A config
        the backend rejects for this shape (``BackendUnsupported``, a
        structural refusal) must be rejected identically with the exact
        hoist on.  A config on which the pass fails closed (no loop it can
        take: a masked operand or packet store, a store packet list bound
        outside the lane loop) must render exactly the plain kernel's body,
        which then needs no second compile.  Returns the indices of the
        configs on which the redo form fired; at least one config must
        run."""

        def bits(result: object) -> list[torch.Tensor]:
            tensors = result if isinstance(result, tuple) else (result,)
            return [
                t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)
                for t in tensors
                if isinstance(t, torch.Tensor)
            ]

        ran = 0
        fired: list[int] = []
        for index, config in enumerate(configs):
            plain_config = {
                k: v for k, v in config.items() if k != "cute_exact_div_form"
            }
            try:
                code0, out0 = code_and_output(plain, args, **plain_config)
            except BackendUnsupported as rejection:
                with (
                    self.subTest(config=plain_config, rejected=str(rejection)),
                    self.assertRaises(BackendUnsupported),
                ):
                    code_and_output(kernel, args, **plain_config)
                continue
            ran += 1
            body0 = _kernel_body(code0)
            assert ANY_EXACT not in body0
            rendered = _kernel_body(
                kernel.bind(args).to_code(helion.Config(**plain_config))
            )
            if ANY_EXACT not in rendered:
                assert rendered == body0, plain_config
                continue
            fired.append(index)
            code, out = code_and_output(kernel, args, **plain_config)
            body = _kernel_body(code)
            assert REDO in body, plain_config
            for marker in redo_markers:
                assert marker in body, (marker, plain_config)
            with self.subTest(config=plain_config):
                for exact, expected in zip(bits(out), bits(out0), strict=True):
                    self.assertTrue(torch.equal(exact, expected))
        assert ran, configs
        return fired

    def _random_configs(
        self, kernel: helion.Kernel, args: tuple[object, ...], count: int
    ) -> list[dict[str, Any]]:
        """``count`` configs with one row per thread plus up to ``count``
        with several rows per thread (``_several_rows_per_thread``: the
        sweeps then sit in a constexpr lane loop) where the population
        draws them -- on a GPU every kernel's does (a row block above one
        with fewer row threads than rows, or row vectors wider than one, is
        most of the draw), the CPU stub's softmax / JSD populations keep a
        row block of one -- from the kernel's own spec (a fixed seed), each
        with the independent reduction on where the kernel has it."""
        random.seed(0)
        spec = kernel.bind(args).config_spec
        buckets: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
        for config in ConfigGeneration(spec).random_population(16 * count):
            values = dict(config.config)
            if "cute_independent_reduction" in values:
                values["cute_independent_reduction"] = True
            bucket = buckets[_several_rows_per_thread(values)]
            if len(bucket) < count:
                bucket.append(values)
            if len(buckets[False]) == len(buckets[True]) == count:
                break
        assert len(buckets[False]) == count, buckets
        return [*buckets[False], *buckets[True]]

    def test_softmax_configs_with_independent_reductions_match_the_plain_render(
        self,
    ) -> None:
        """The independent partial accumulators and random configs: the
        redo form equals the plain render on the adversarial fp16 rows, on
        an exact and on a ragged shape.  A config whose consume sweep the
        redo cannot take (its store packet list is bound outside the lane
        loop; every config with a masked tail) renders the plain render."""
        for rows, cols in ((96, 4096), (RAGGED_ROWS, RAGGED_COLS)):
            g = torch.Generator().manual_seed(0)
            x = torch.randn(rows, cols, generator=g)
            x[0, ::3] = float("-inf")
            x[1, :] = float("-inf")
            x[2, 7] = float("nan")
            x[3, :] = 0.0
            x[4, :] += 6e4
            x[5, :] *= 1e4
            x[6, :] = -65504.0
            x[6, 1] = 65504.0
            x[rows - 1, ::2] = float("-inf")
            args = (x.to(torch.float16).to(DEVICE),)
            configs = [
                INDEPENDENT_SOFTMAX_CONFIG,
                *self._random_configs(softmax, args, 2),
            ]
            with self.subTest(shape=(rows, cols)):
                fired = self._forms_match_the_plain_render(
                    softmax, softmax_plain, args, configs
                )
                if cols == 4096:
                    # The campaign's pinned config takes the redo form.
                    assert 0 in fired, fired

    def test_jsd_configs_with_independent_reductions_match_the_plain_render(
        self,
    ) -> None:
        """JSD under the pinned configs, the configs whose masked tails the
        retired guard form perturbed by 1-2 ulp, and random configs.  On an
        exact vocabulary with a partial row block (50 x 4096) the one-row
        configs take the redo form and equal the plain render bit for bit;
        the several-rows configs (row-masked loads) and every config on a
        vocabulary the lane sweeps do not divide (50 x 4000: the masked
        packet loads reach every operand) render the plain kernel."""
        for cols in (4096, RAGGED_COLS):
            g = torch.Generator().manual_seed(0)
            student = torch.randn(RAGGED_ROWS, cols, generator=g)
            teacher = torch.randn(RAGGED_ROWS, cols, generator=g)
            student[0, :] = -1e30
            student[1, 5] = float("nan")
            student[2, ::7] = float("-inf")
            student[3, :] *= 1e30
            teacher[RAGGED_ROWS - 1, :] = -1e30
            args = (0.3, 2.0, student.to(DEVICE), teacher.to(DEVICE))
            configs = [
                INDEPENDENT_JSD_CONFIG,
                *JSD_RAGGED_CONFIGS,
                *self._random_configs(jsd, args, 1),
            ]
            with self.subTest(cols=cols):
                fired = self._forms_match_the_plain_render(
                    jsd, jsd_plain, args, configs
                )
                if cols == 4096:
                    assert set(fired) >= {0, 1}, fired
                    assert not set(fired) & {2, 3}, fired
                else:
                    assert fired == [], fired

    def test_one_row_packet_configs_on_a_ragged_axis_render_the_plain_kernel(
        self,
    ) -> None:
        """Every rows kernel on the hard rows of 53 x 4000 under the one-row
        packet family (corpus-4 config 43, whose redo render differed from
        the plain render by 1 ulp in a sum of squares) and random one-row
        configs: the masked packet loads reach every operand, so the packet
        family renders the plain kernel; a random config that still takes
        the redo form (a lane sweep that divides 4000 has no masked tail)
        must equal the plain render bit for bit."""
        args = _hard_rows(DEVICE, RAGGED_ROWS, RAGGED_COLS)
        for kernel, plain in ROWS_KERNELS:
            random_configs = [
                config
                for config in self._random_configs(kernel, args, 3)
                if not _several_rows_per_thread(config)
            ]
            assert len(random_configs) == 3, random_configs
            configs = [*ONE_ROW_PACKET_CONFIGS, *random_configs]
            with self.subTest(kernel=kernel.fn.__name__):
                fired = self._forms_match_the_plain_render(kernel, plain, args, configs)
                assert not set(fired) & set(range(len(ONE_ROW_PACKET_CONFIGS))), fired

    def test_reduction_configs_with_independent_reductions_match_the_plain_render(
        self,
    ) -> None:
        """A rolled row kernel with a store sweep and a reduction sweep:
        one-row and several-rows random configs on the hard rows, on an
        exact shape (every config takes the redo form) and on a ragged
        shape (the masked operand loads keep the plain division)."""
        for shape in ((64, 4096), (RAGGED_ROWS, RAGGED_COLS)):
            args = _hard_rows(DEVICE, *shape)
            configs = [
                INDEPENDENT_ROWS_CONFIG,
                *self._random_configs(normalize_and_sum, args, 2),
            ]
            assert sum(map(_several_rows_per_thread, configs)) == 2, configs
            with self.subTest(shape=shape):
                fired = self._forms_match_the_plain_render(
                    normalize_and_sum, normalize_and_sum_plain, args, configs
                )
                if shape == (64, 4096):
                    assert fired == list(range(len(configs))), fired

    def _several_rows_configs_match_the_plain_render(
        self, shape: tuple[int, int]
    ) -> None:
        args = _hard_rows(DEVICE, *shape)
        for kernel, plain in ROWS_KERNELS:
            with self.subTest(kernel=kernel.fn.__name__, shape=shape):
                fired = self._forms_match_the_plain_render(
                    kernel,
                    plain,
                    args,
                    SEVERAL_ROWS_CONFIGS,
                    redo_markers=("cutlass.range_constexpr", "_helion_exdiv_save_"),
                )
                if shape == (64, 4096):
                    assert fired == list(range(len(SEVERAL_ROWS_CONFIGS))), fired

    def test_several_rows_per_thread_configs_match_the_plain_render(self) -> None:
        """The rows kernels under the several-rows configs (the sweeps inside
        a ``cutlass.range_constexpr`` lane loop, the accumulator defined in
        the lane body) on the hard rows: the redo render snapshots the lane
        accumulator and equals the plain render bit for bit, the rows whose
        dividends leave the window included."""
        self._several_rows_configs_match_the_plain_render((64, 4096))

    def test_several_rows_per_thread_configs_on_a_ragged_shape(self) -> None:
        """The same on a ragged shape (a partial row block, a reduction axis
        the lanes do not divide): the masked operand loads keep the plain
        division (the render is the plain render; were a sweep to take the
        redo form it would have to equal the plain render bit for bit)."""
        self._several_rows_configs_match_the_plain_render((RAGGED_ROWS, RAGGED_COLS))
