"""CPU codegen of Inductor's pointwise GEMM epilogues as tcgen05 chains.

Each form of ``helion._testing.cute_inductor_epilogues`` (0-d constants, a
repeated operand, boolean masks, ``where`` selects) takes the tcgen05 pair
default and renders its chain step; ``test_cute_inductor_epilogues_gpu`` runs
the same forms against torch.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import checked_initial_population

import helion
from helion import exc
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_inductor_epilogues import INDUCTOR_EPILOGUES
from helion._testing.cute_inductor_epilogues import InductorEpilogue
from helion._testing.cute_inductor_epilogues import inductor_epilogue_args
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    # B200's 148 SMs, which the pair defaults' wave counts are sized from
    with (
        cpu_codegen_target(),
        forbid_cuda_work(),
        patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148),
    ):
        yield


# The transcendental forms over an exact-shape operand: the ring's depth.
_RING_PREFETCH = {"gelu_grad": 2, "gelu_tanh_grad": 1, "selu_grad": 1}


# One axis, one case per epilogue form.
@pytest.mark.parametrize(
    "epilogue", INDUCTOR_EPILOGUES, ids=[e.name for e in INDUCTOR_EPILOGUES]
)
def test_inductor_pointwise_epilogues_join_the_chain(
    epilogue: InductorEpilogue,
) -> None:
    """Inductor's pointwise epilogues are tcgen05 chains: a 0-d ``hl.full`` is
    the literal it holds (in either operand order), a repeated operand (the
    lowering's ``mul(y, None)`` for ``y * y``) is one leaf, a boolean mask
    converts to the exact 0/1 promotion, ``where(cond, 0, grad)`` selects
    between a literal and the carrier, and a step may read the carrier more
    than once (GELU, SELU and Hardswish backward). Each takes the pair default
    and renders its chain step, instead of raising ``BackendUnsupported`` on
    every config."""
    bound = epilogue.kernel._bind_isolated(
        inductor_epilogue_args(epilogue, 4096, 2048, 768, device="cpu", random=False)
    )
    config = bound.config_spec.default_config()
    code = bound.to_code(config)
    assert config.block_sizes[:2] == [256, 256]
    assert re.search(epilogue.step, code, re.MULTILINE), epilogue.step
    assert ("tcgen05_epi_group = " in code) is epilogue.multiple_epilogue_groups
    if epilogue.name in _RING_PREFETCH:
        # Its exact-shape operand two subtiles ahead under two transcendental
        # ops, one under one, on three groups at the short K loop's three AB
        # stages.
        assert config.config["tcgen05_epi_warp_groups"] == 3
        assert config.config["tcgen05_epi_subtile_n"] == 32
        assert (
            config.config["tcgen05_aux_prefetch_subtiles"]
            == _RING_PREFETCH[epilogue.name]
        )
        assert config.config["tcgen05_ab_stages"] == 3
    # one load per auxiliary tensor, however often the epilogue reads it (a
    # rank-0 scalar binds once per tile instead)
    loads = set(re.findall(r"\btcgen05_aux_loaded(?:_\d+)*\b", code))
    assert len(loads) == len([kind for kind in epilogue.aux if kind != "scalar"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp16_masked_fill_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    fill: hl.constexpr,
) -> None:
    """``masked_fill(mask, fill)`` on an FP16 carrier."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        value = hl.full([], fill, dtype=torch.float32)
        out[rows, cols] = torch.where(
            mask[rows, cols], value, acc.to(torch.float16)
        ).to(out.dtype)


@pytest.mark.parametrize(
    ("fill", "select"),
    (
        (-1e4, "cute.where(tcgen05_aux_loaded_0, -10000.0, "),
        (-1e9, "cute.where(tcgen05_aux_loaded_0, float('-inf'), "),
    ),
    ids=("finite", "infinite"),
)
def test_where_literal_rounds_to_the_carrier_type(fill: float, select: str) -> None:
    """The select's literal is the value the carrier's type holds: -1e9
    selected against an FP16 carrier is ``-inf``, which the select spells
    ``float('-inf')`` (``repr`` would splice an undefined ``inf``)."""
    args = (
        torch.empty((4096, 768), dtype=torch.float16),
        torch.empty((768, 2048), dtype=torch.float16),
        torch.empty((4096, 2048), dtype=torch.bool),
        torch.empty((4096, 2048), dtype=torch.float16),
        fill,
    )
    bound = _fp16_masked_fill_mm._bind_isolated(args)
    assert select in bound.to_code(bound.config_spec.default_config())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scalar_condition_mm(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> None:
    """``where(scale[()] > 0, 0, grad)``: a tile-uniform condition."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        zero = hl.full([], 0.0, dtype=torch.float32)
        out[rows, cols] = torch.where(scale[()] > zero, zero, acc).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _column_condition_mm(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """``where(x <= 0, 0, grad)`` with ``x`` a per-row (column) broadcast."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        zero = hl.full([], 0.0, dtype=torch.float32)
        dead = x[rows, cols].to(torch.float32) <= zero
        out[rows, cols] = torch.where(dead, zero, acc).to(out.dtype)


@pytest.mark.parametrize("condition", ("scalar", "column"))
def test_where_needs_a_tensor_condition(condition: str) -> None:
    """``cute.where`` takes a tensor condition: a condition over tile-uniform
    scalars (``scale[()] > 0``) or a per-row value read once per thread (a
    column broadcast) renders as a scalar, so the chain refuses it."""
    a = torch.empty((4096, 768), dtype=torch.bfloat16)
    b = torch.empty((768, 2048), dtype=torch.bfloat16)
    out = torch.empty((4096, 2048), dtype=torch.bfloat16)
    if condition == "scalar":
        scale = torch.empty((), dtype=torch.float32)
        bound = _scalar_condition_mm._bind_isolated((a, b, scale, out))
    else:
        column = torch.empty((4096, 1), dtype=torch.bfloat16).expand(4096, 2048)
        bound = _column_condition_mm._bind_isolated((a, b, column, out))
    with pytest.raises(exc.BackendUnsupported, match="non-whitelisted"):
        bound.to_code(bound.config_spec.default_config())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _y_pow32_mm(
    a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> None:
    """``grad * y**32`` as Inductor's repeated squaring."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        y2 = y[rows, cols].to(torch.float32)
        y2 = y2 * y2
        y2 = y2 * y2
        y2 = y2 * y2
        y2 = y2 * y2
        y2 = y2 * y2
        out[rows, cols] = (acc * y2).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sigmoid_grad_mm(
    a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> None:
    """Sigmoid backward ``grad * y * (1 - y)``: one load reused indirectly."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        yv = y[rows, cols].to(torch.float32)
        one = hl.full([], 1.0, dtype=torch.float32)
        out[rows, cols] = (acc * (yv * (one - yv))).to(out.dtype)


@pytest.mark.parametrize(("kernel", "lines"), ((_y_pow32_mm, 6), (_sigmoid_grad_mm, 3)))
def test_repeated_operands_load_and_render_once(
    kernel: helion.Kernel, lines: int
) -> None:
    """Every use of an FX node within one expression is the same subtree: its
    input loads once and the renderer binds the subtree to one local, so
    repeated squaring stays linear (``y**32`` is five products) and the
    indirect reuse of ``y * (1 - y)`` loads ``y`` once."""
    args = (
        torch.empty((4096, 768), dtype=torch.bfloat16),
        torch.empty((768, 2048), dtype=torch.bfloat16),
        torch.empty((4096, 2048), dtype=torch.bfloat16),
        torch.empty((4096, 2048), dtype=torch.bfloat16),
    )
    bound = kernel._bind_isolated(args)
    code = bound.to_code(bound.config_spec.default_config())
    assert len(set(re.findall(r"\btcgen05_aux_loaded(?:_\d+)*\b", code))) == 1
    expr_lines = {
        line.strip()
        for line in code.splitlines()
        if re.match(r"\s*tcgen05_aux_(?:expr|product)(?:_\d+)* = ", line)
    }
    assert len(expr_lines) == lines, sorted(expr_lines)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bf16_constant_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``(x @ w).to(bf16) * hl.full([], 0.1)``: a BF16 product of a 0-d
    constant."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        tenth = hl.full([], 0.1, dtype=torch.float32)
        out[rows, cols] = (acc.to(torch.bfloat16) * tenth).to(out.dtype)


def test_zero_dim_constant_rounds_to_a_16_bit_result_type() -> None:
    """Unlike a Python number, a 0-d tensor operand converts to the BF16
    result type first: the product scales by ``bf16(0.1) = 0.10009765625``
    (torch's ``bf16_x * fp32_0d``), not by ``0.1``."""
    bound = _bf16_constant_mm._bind_isolated(
        (
            torch.empty((4096, 768), dtype=torch.bfloat16),
            torch.empty((768, 2048), dtype=torch.bfloat16),
            torch.empty((4096, 2048), dtype=torch.bfloat16),
        )
    )
    code = bound.to_code(bound.config_spec.default_config())
    assert "* 0.10009765625" in code
    assert "* 0.1)" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scalar_branches_mm(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """``acc * where(mask, 1.25, 0.0)``: a select between two numbers."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        keep = torch.where(mask[rows, cols], 1.25, 0.0)
        out[rows, cols] = (acc * keep).to(out.dtype)


def test_select_needs_a_tensor_branch() -> None:
    """``cute.where`` takes at least one tensor branch: a select between two
    numbers is refused rather than failing in the DSL."""
    bound = _scalar_branches_mm._bind_isolated(
        (
            torch.empty((4096, 768), dtype=torch.bfloat16),
            torch.empty((768, 2048), dtype=torch.bfloat16),
            torch.empty((4096, 2048), dtype=torch.bool),
            torch.empty((4096, 2048), dtype=torch.bfloat16),
        )
    )
    with pytest.raises(exc.BackendUnsupported, match="non-whitelisted"):
        bound.to_code(bound.config_spec.default_config())


def test_select_condition_forms_ahead_of_the_accumulator_wait() -> None:
    """Under ``pre_acc_wait`` ReLU backward's ``x <= 0`` forms with the
    auxiliary loads, ahead of the accumulator wait, like the auxiliary side of
    ``acc + aux``."""
    (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "relu_grad"]
    bound = form.kernel._bind_isolated(
        inductor_epilogue_args(form, 4096, 2048, 768, device="cpu", random=False)
    )
    config = bound.config_spec.default_config()
    config.config["tcgen05_aux_load_placement"] = "pre_acc_wait"
    code = bound.to_code(config)
    compare = re.search(r"tcgen05_aux_expr(?:_\d+)* = \S+ <= 0\.0", code)
    assert compare is not None
    assert compare.start() < code.index("tcgen05_acc_pipeline.consumer_wait")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_mask_mm(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, out: torch.Tensor
) -> None:
    """``acc * mask`` with a per-row mask (``(m, 1)`` expanded over N)."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        keep = mask[rows, cols].to(torch.bool).to(torch.float32)
        out[rows, cols] = (acc * keep).to(out.dtype)


def test_per_row_mask_reads_its_byte_as_nonzero() -> None:
    """A per-row mask element reads as its byte storage, which torch reads as
    ``!= 0`` (a byte other than 0/1 is true)."""
    mask = torch.empty((4096, 1), dtype=torch.bool).expand(4096, 2048)
    bound = _row_mask_mm._bind_isolated(
        (
            torch.empty((4096, 768), dtype=torch.bfloat16),
            torch.empty((768, 2048), dtype=torch.bfloat16),
            mask,
            torch.empty((4096, 2048), dtype=torch.bfloat16),
        )
    )
    (seed,) = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("pid_type") != "flat"
    ][:1]
    code = bound.to_code(seed)
    assert re.search(
        r"tcgen05_aux_loaded(?:_\d+)* = "
        r"tcgen05_tTR_gAux_grouped(?:_\d+)*\[0, 0, 0, [^\]]*\] != 0$",
        code,
        re.MULTILINE,
    )


def test_boolean_mask_on_tma_store_edge_tiles_reads_its_int8_storage() -> None:
    """The hybrid TMA-store epilogue copies an exact-shape operand of its SIMT
    edge tiles through a copy atom of the operand's type, and a Boolean atom
    is narrower than the mask's byte storage (``cute.copy`` refuses unequal
    widths): a mask copies element-wise into an Int8 fragment and is read as
    ``!= 0``."""
    (dropout,) = [e for e in INDUCTOR_EPILOGUES if e.name == "dropout"]
    bound = dropout.kernel._bind_isolated(
        inductor_epilogue_args(dropout, 3000, 2048, 256, device="cpu", random=False)
    )
    # The persistent pair over the M edge (3000 rows): whole tiles through
    # TMA, the last row of tiles through the SIMT edge store.
    code = bound.to_code(
        helion.Config(
            block_sizes=[256, 256, 64],
            pid_type="persistent_interleaved",
            l2_groupings=[4],
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=2,
            tcgen05_acc_stages=2,
        )
    )
    assert "_edge_atom" not in code
    assert re.search(r"cute\.make_rmem_tensor\([^\n]*, cutlass\.Int8\)", code)
    assert re.search(r"= tcgen05_aux_rmem(?:_\d+)*\.load\(\) != 0$", code, re.MULTILINE)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp32_erf_mm(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, out: torch.Tensor
) -> None:
    """``acc * erf(x)`` into an FP32 output: no aux hoist at FP32, so the
    L2 swizzle's coverage carriers come from the seeds."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = acc * torch.erf(x[rows, cols])


def test_masked_ring_reads_one_ahead_on_two_groups() -> None:
    """Beside a mask the ring's operand reads one subtile ahead, and the
    chain's two exact-shape operands keep two groups at the deep AB ring
    (no short-K cap)."""
    (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "masked_gelu_grad"]
    bound = form.kernel._bind_isolated(
        inductor_epilogue_args(form, 8192, 4096, 1024, device="cpu", random=False)
    )
    default = bound.config_spec.default_config()
    assert default.block_sizes[:2] == [256, 256]
    assert default.config["tcgen05_epi_warp_groups"] == 2
    assert default.config["tcgen05_ab_stages"] > 3
    assert default.config["tcgen05_aux_prefetch_subtiles"] == 1


def test_two_masks_take_the_capped_three_group_ring() -> None:
    """Two masks over a short K loop take the ring's AB cap (four stages on
    the 256x128 pair) and its third group, where a lone mask keeps two groups
    at the deepest ring
    (``test_register_mask_epilogue_runs_two_groups_over_a_short_k_loop``)."""
    (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "two_mask"]
    bound = form.kernel._bind_isolated(
        inductor_epilogue_args(form, 8192, 2048, 512, device="cpu", random=False)
    )
    default = bound.config_spec.default_config()
    assert default.block_sizes[:2] == [256, 128]
    assert default.config["tcgen05_epi_warp_groups"] == 3
    assert default.config["tcgen05_ab_stages"] == 4
    assert default.config["tcgen05_aux_prefetch_subtiles"] == 1


def test_long_k_ring_keeps_the_deepest_ab_ring() -> None:
    """Over 64 k-blocks the ring keeps two groups and the AB stages their
    rings leave (the MMA wants them), not the short loop's three groups at
    three stages."""
    (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "gelu_grad"]
    bound = form.kernel._bind_isolated(
        inductor_epilogue_args(form, 4096, 2048, 4096, device="cpu", random=False)
    )
    default = bound.config_spec.default_config()
    assert default.block_sizes[:2] == [256, 256]
    assert default.config["tcgen05_epi_warp_groups"] == 2
    assert default.config["tcgen05_aux_prefetch_subtiles"] == 2
    assert default.config["tcgen05_ab_stages"] > 3


@pytest.mark.parametrize(
    ("k", "groups", "ab_stages"), ((1024, 2, 5), (4096, 1, 6)), ids=("short", "long")
)
def test_register_mask_epilogue_runs_two_groups_over_a_short_k_loop(
    k: int, groups: int, ab_stages: int
) -> None:
    """A boolean mask read on registers bounds the 256x256 pair by its
    epilogue over at most 16 k-blocks: two groups and the one-deep ring at
    the five AB stages the groups' rings leave (a mask leaves L1 alone, so
    the ring's AB cap stays off); a long K loop keeps one group's deeper
    ring."""
    (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "dropout"]
    bound = form.kernel._bind_isolated(
        inductor_epilogue_args(form, 8192, 4096, k, device="cpu", random=False)
    )
    default = bound.config_spec.default_config()
    assert default.block_sizes[:2] == [256, 256]
    assert default.config["tcgen05_epi_warp_groups"] == groups
    assert default.config["tcgen05_ab_stages"] == ab_stages
    if groups == 2:
        assert default.config["tcgen05_epi_subtile_n"] == 32
        assert default.config["tcgen05_aux_prefetch_subtiles"] == 1


# One-CTA multi-group defaults whose coverage carriers come from the aux-hoist
# witnesses (GELU backward, two geometries) or from the seeds (FP32 erf).
@pytest.mark.parametrize(
    ("kernel", "shape"),
    (
        ("gelu_grad", (6272, 4096, 1024)),
        ("gelu_grad", (25216, 768, 192)),
        ("fp32_erf", (6272, 4096, 1024)),
    ),
)
def test_multi_group_default_keeps_the_coverage_population(
    kernel: str, shape: tuple[int, int, int]
) -> None:
    """The default's epilogue warp groups stay out of the coverage
    carriers drawn from the seeds (``coverage_seed_carriers``): a carrier
    holds every group it does not declare at its legacy mode, or the
    population step refuses its witnesses (both aux-hoist witnesses and the
    swizzle's seed carriers were refused). The hoist stays off where one
    group does not admit it, and every declared witness reaches the initial
    population."""
    m, n, k = shape
    if kernel == "gelu_grad":
        (form,) = [e for e in INDUCTOR_EPILOGUES if e.name == "gelu_grad"]
        args = inductor_epilogue_args(form, m, n, k, device="cpu", random=False)
        bound = form.kernel._bind_isolated(args)
    else:
        args = (
            torch.empty((m, k), dtype=torch.bfloat16),
            torch.empty((k, n), dtype=torch.bfloat16),
            torch.empty((m, n), dtype=torch.float32),
            torch.empty((m, n), dtype=torch.float32),
        )
        bound = _fp32_erf_mm._bind_isolated(args)
    spec = bound.config_spec
    default = spec.default_config()
    # GELU backward's capped short-K ring runs a third group
    assert default.config["tcgen05_epi_warp_groups"] == (
        3 if kernel == "gelu_grad" else 2
    )
    if kernel == "gelu_grad" and shape == (6272, 4096, 1024):
        # The one-CTA AB envelope prices one group's C ring: beside several
        # groups' rings the 128x256 tile fits three 64-deep AB stages at most
        # (four need 262 KB of the 232 KB B200 cap beside two groups, which
        # NVVM refused), and over the 16 k-blocks the ring keeps two of them
        # (96 KiB), beside three groups.
        assert default.block_sizes[:2] == [128, 256]
        assert default.config["tcgen05_ab_stages"] == 2
        # The exact-shape operand reads two 32-column subtiles ahead.
        assert default.config["tcgen05_epi_subtile_n"] == 32
        assert default.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
        assert default.config["tcgen05_aux_prefetch_subtiles"] == 2
        code = bound.to_code(default)
        assert (
            "tcgen05_ab_pipeline = cutlass.pipeline.PipelineTmaUmma.create("
            "num_stages=2," in code
        )
        assert "tcgen05_epi_group = " in code
        assert "tcgen05_aux_ring_0_2" in code
    groups = {group.key: group for group in spec.compiler_coverage_groups}
    assert "tcgen05_aux_fragment_hoist" not in groups
    assert all(
        witness.carrier.config.get("tcgen05_epi_warp_groups", 1) == 1
        for witness in groups["tcgen05_l2_swizzle_size"].witnesses
    )
    state = random.getstate()
    try:
        with bound.env:
            search = LFBOTreeSearch(
                bound,
                args,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            random.seed(2026091539)
            checked_initial_population(search)
    finally:
        random.setstate(state)
