"""CPU cross-domain tests for the combined batch-digit, shared-A M-edge and quotient-aux source.

Each line's own tests cover its own feature. These tests cover how the features meet in
one tree:
  - a batch-digit store never composes with an auxiliary (quotient) read;
  - quotient kernels keep their register-MMA, seed, default and fixed-axis fallback facts;
  - shared-A MMA-DAG programs get no plane-store seeds;
  - the `_cute_python_mod` pure-call generalization omits only proof-owned index
    statements. Every refused store or quotient-grammar variant stays loud under a
    persistent pid instead of compiling with an omitted remainder.

No CUDA work and no native compilation happen here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion import exc
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import cute_mma
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

Z, M, N, K, H = 8, 256, 64, 128, 4
PERSISTENT = ("persistent_interleaved", "persistent_blocked")


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _bind(kernel: helion.Kernel, args: tuple[object, ...]):
    fresh = helion.kernel(
        kernel.fn,
        settings=kernel.settings,
        cute_structural_policy=kernel.cute_structural_policy,
    )
    with patch.object(fresh, "_bind_isolated", side_effect=fresh.bind):
        return cpu_bind(fresh, args)


def _default(bound) -> helion.Config:
    with bound.env, bound.host_function:
        generation = ConfigGeneration(bound.config_spec).initial_population_view()
        return generation.unflatten(generation.default_flat())


def _with_pid(bound, config: helion.Config, pid_type: str) -> helion.Config:
    raw = {k: v for k, v in config.config.items() if k != "tcgen05_persistence_model"}
    raw["pid_type"] = pid_type
    adjusted = helion.Config.from_dict(raw)
    bound.config_spec.normalize(adjusted)
    return adjusted


def _mma_chains(bound, target):
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        anchor = next(
            node
            for info in graphs
            for node in info.graph.nodes
            if node.target is target
        )
        return analyze_tcgen05_matmul_store_chains(graphs, anchor)


def _digit_map(bound):
    device_ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        (node,) = [
            node
            for info in device_ir.graphs
            for node in info.graph.nodes
            if node.op == "call_function"
            and cute_mma._decode_cute_mma_target(node, device_ir=device_ir) is not None
        ]
        candidate = cute_mma.analyze_cute_mma_node(node, device_ir=device_ir)
        # No analyzed candidate (a refused store) has no digit map either.
        return cute_mma._candidate_batch_digit_output(candidate)


# ---------------------------------------------------------------- kernels


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def head_split(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, yi, zi % heads, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def head_split_with_quotient_read(
    A: torch.Tensor, B: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        v = aux[yi, xi // 16]
        out[zi // heads, yi, zi % heads, xi] = (acc * v.to(torch.float32)).to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def head_split_with_exact_aux(
    A: torch.Tensor, B: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    """A batch-digit store whose value chain reads a same-shape (exact) aux tile."""
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        v = aux[tz, ty, tx]
        out[zi // heads, yi, zi % heads, xi] = (acc + v.to(torch.float32)).to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def wrapped_row(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, (yi + 1) % m, zi % heads, xi] = acc.to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def rank1_m_index(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        out[zi // heads, (ty.index + m * zi) % m, zi % heads, xi] = acc.to(
            torch.bfloat16
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def masked_store(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    z, m, k = A.size()
    n = B.size(2)
    heads = out.size(2)
    for tz, ty, tx in hl.tile([z, m, n]):
        acc = hl.zeros([tz, ty, tx], dtype=torch.float32)
        for tr in hl.tile(k):
            acc = torch.baddbmm(acc, A[tz, ty, tr], B[tz, tr, tx])
        zi = tz.index.reshape([tz.index.size(0), 1, 1])
        yi = ty.index.reshape([1, ty.index.size(0), 1])
        xi = tx.index.reshape([1, 1, tx.index.size(0)])
        hl.store(
            out,
            [zi // heads, yi, zi % heads, xi],
            acc.to(torch.bfloat16),
            extra_mask=xi < n - 1,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def quotient_cols(a, b, aux, first, second, c: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]), x.index.reshape([1, x]) // c]
        second[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def quotient_with_mod(a, b, aux, first, second, c: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]), (x.index.reshape([1, x]) % 32) // c]
        second[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def quotient_with_mod_store(a, b, aux, first, second, c: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n]):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]), x.index.reshape([1, x]) // c]
        rows = y.index.reshape([y.index.size(0), 1])
        cols = (x.index.reshape([1, x.index.size(0)]) + n) % n
        second[rows, cols] = (acc * v.to(torch.float32)).to(torch.float32)


SHAREDA_INDUCTOR = """import torch
import helion
import helion.language as hl

@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)
def helion_op1_op3_op2_op4(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
        tmp6 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp6.to(torch.float32)
        tmp7 = acc1.to(torch.float32)
        out1[tile_y, tile_x] = tmp7.to(torch.float32)
"""


SHAREDA_MM53_DEFAULT = {
    "block_sizes": [16, 16, 16],
    "loop_orders": [[0, 1]],
    "load_eviction_policies": ["", "", ""],
    "cute_store_policy": "default",
    "cute_proven_bounds": False,
    "cute_pdl": False,
    "cute_vector_widths": [1, 1, 1],
    "cute_lane_layouts": ["blocked", "blocked", "blocked"],
    "cute_collective_mma": False,
    "cute_collective_compute": "warp",
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_seed": False,
    "cute_collective_tmem_a": False,
    "cute_collective_stages": 1,
    "cute_collective_copy": "scalar",
    "cute_collective_recipe": "scalar",
    "cute_collective_operand_packets": False,
    "cute_collective_epilogue": "scalar",
    "cute_gathered_mma_n": 256,
    "cute_gathered_mma_stages": 3,
    "cute_register_chain": False,
    "cute_mma_dag": "off",
    "cute_mma_dag_warps": 4,
    "cute_mma_dag_stages": 4,
    "cute_mma_dag_store": "staged",
    "cute_mma_dag_exchange": "auto",
    "cute_mma_dag_pass_warps": "off",
    "cute_mma_dag_loads": "tma",
    "cute_mma_dag_derived": "exchange",
    "cute_mma_dag_epilogue": "row",
    "cute_mma_dag_lookahead": "off",
    "cute_mma_dag_persistent": 0,
    "cute_mma_dag_aux": "register",
    "cute_mma_dag_cluster": 1,
    "cute_mma_dag_split": "cluster",
}


def _head_split_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((Z, M, K), dtype=torch.bfloat16),
        torch.empty((Z, K, N), dtype=torch.bfloat16),
        torch.empty((Z // H, M, H, N), dtype=torch.bfloat16),
    )


def _quotient_args(c: int = 16):
    m, n, k = 128, 64, 64
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((m, (n - 1) // c + 1)),
        torch.empty((m, n)),
        torch.empty((m, n)),
        c,
    )


# ---------------------------------------------------------------- batch-digit x quotient


def test_batch_digit_store_with_an_aux_read_is_not_a_plane_store() -> None:
    a, b, out = _head_split_args()
    aux = torch.empty((M, N // 16), dtype=torch.float32)
    bound = _bind(head_split_with_quotient_read, (a, b, aux, out))
    assert _digit_map(bound) is None
    tcgen05 = bound.config_spec._cute_tcgen05_config
    assert not tcgen05.batch_digit_plane_store_seeds_available
    assert tcgen05._optional_batch_digit_plane_store_seed_configs() == []
    # No aux leaf is accepted in a batched (rank-3) carrier's store chain, so
    # the chain analysis itself declines: no quotient leaf (rank-2 carriers only)
    # and no analyzed store reach the batch-digit proof.
    assert _mma_chains(bound, torch.ops.aten.baddbmm.default) is None
    code = bound.to_code(_default(bound))
    assert ".iterator + cute.crd2idx((" not in code
    assert "_cute_python_mod(" in code


def test_batch_digit_store_with_an_exact_aux_read_is_refused_by_the_chain() -> None:
    # The plain batch-digit kernel has an analyzed store chain ...
    plain = _bind(head_split, _head_split_args())
    chains = _mma_chains(plain, torch.ops.aten.baddbmm.default)
    assert chains is not None and len(chains) == 1
    assert not chains[0][1].auxiliary_tensor_loads
    # ... but adding a same-shape aux read makes the chain analysis decline, which
    # is the first refusal; the proof's own aux exclusion stays as a second guard.
    a, b, out = _head_split_args()
    aux = torch.empty((Z, M, N), dtype=torch.float32)
    bound = _bind(head_split_with_exact_aux, (a, b, aux, out))
    assert _mma_chains(bound, torch.ops.aten.baddbmm.default) is None
    assert _digit_map(bound) is None
    tcgen05 = bound.config_spec._cute_tcgen05_config
    assert not tcgen05.batch_digit_plane_store_seeds_available
    for pid_type in ("flat", *PERSISTENT):
        code = bound.to_code(_with_pid(bound, _default(bound), pid_type))
        assert ".iterator + cute.crd2idx((" not in code
        assert "_cute_python_mod(" in code


def test_batch_digit_output_aliasing_an_operand_keeps_the_generic_route() -> None:
    a, b, _out = _head_split_args()
    out = a.view(-1)[: Z * M * N].view(Z // H, M, H, N)
    bound = _bind(head_split, (a, b, out))
    tcgen05 = bound.config_spec._cute_tcgen05_config
    assert not tcgen05.batch_digit_plane_store_seeds_available
    for pid_type in ("flat", *PERSISTENT):
        code = bound.to_code(_with_pid(bound, _default(bound), pid_type))
        assert "tcgen05_tma_warp" not in code
        assert ".iterator + cute.crd2idx((" not in code
        assert "_cute_python_mod(" in code


# ---------------------------------------------------------------- quotient facts under C6


def test_quotient_kernel_keeps_its_register_mma_seed_and_default_facts() -> None:
    bound = _bind(quotient_cols, _quotient_args(16))
    spec = bound.config_spec
    tcgen05 = spec._cute_tcgen05_config
    assert spec.cute_tcgen05_search_enabled
    # A quotient leaf is a mapped leaf: register MMA is excluded before C6's own
    # store-contract check is consulted, and no plane-store seed is offered.
    assert not tcgen05.warp_mma_admitted
    assert not tcgen05.batch_digit_plane_store_seeds_available
    assert list(spec.compiler_seed_configs) == []
    chains = _mma_chains(bound, torch.ops.aten.addmm.default)
    assert any(
        isinstance(leaf, epi._QuotientAuxiliaryTensorLoadExpr)
        for _store, chain in chains or ()
        for leaf in chain.auxiliary_tensor_loads
    )
    code = bound.to_code(_default(bound))
    assert "'kind': 'tcgen05_ab_tma'" in code
    assert "_cute_python_mod(" not in code


def test_untransported_quotient_keeps_the_fixed_axis_fallback() -> None:
    bound = _bind(quotient_cols, _quotient_args(3))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert not bound.config_spec._cute_tcgen05_config.batch_digit_plane_store_seeds_available


# ---------------------------------------------------------------- shared-A DAG under C6


def test_shared_a_dag_program_gets_no_plane_store_seeds() -> None:
    original = PyCodeCache.load(SHAREDA_INDUCTOR).helion_op1_op3_op2_op4
    args = (
        torch.empty_strided((1000, 128), (1, 1000), dtype=torch.bfloat16),
        torch.empty((128, 768), dtype=torch.bfloat16),
        torch.empty((128, 768), dtype=torch.bfloat16),
        torch.empty((1000, 768), dtype=torch.float32),
        torch.empty((1000, 768), dtype=torch.float32),
    )
    bound = _bind(original, args)
    spec = bound.config_spec
    tcgen05 = spec._cute_tcgen05_config
    assert not tcgen05.batch_digit_plane_store_seeds_available
    assert tcgen05._optional_batch_digit_plane_store_seed_configs() == []
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.shared_a_m_edge_axes is not None
    seeds = list(spec.compiler_seed_configs)
    assert seeds and all("tcgen05_strategy" not in s.config for s in seeds)
    # The SharedA line's own banked default for this module
    # (shared_A_TMA_Medge_r2_20261104_1247/actual_CPU_probe.json): C6 must not move it.
    assert dict(spec.default_config()) == SHAREDA_MM53_DEFAULT


# ---------------------------------------------------------------- the `%` pure-call generalization


@pytest.mark.parametrize("pid_type", PERSISTENT)
def test_persistent_plane_store_omits_only_proof_owned_remainders(pid_type) -> None:
    bound = _bind(head_split, _head_split_args())
    mapping = _digit_map(bound)
    assert mapping is not None and mapping.period == H
    # Every remainder node the store parses is owned by the proved digit map.
    owned = set(mapping.owned_index_nodes)
    remainders = [
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.remainder.Scalar
    ]
    assert remainders and all(node in owned for node in remainders)
    code = bound.to_code(_with_pid(bound, _default(bound), pid_type))
    assert "'kind': 'tcgen05_ab_tma'" in code and "tcgen05_tma_warp" in code
    # The plane view recomputes the digits from the leading index; the original
    # Python-semantics remainder statement beside the warp roles is dead and omitted.
    assert ".iterator + cute.crd2idx((" in code
    assert "_cute_python_mod(" not in code


@pytest.mark.parametrize(
    "kernel", [wrapped_row, masked_store], ids=lambda k: k.fn.__name__
)
@pytest.mark.parametrize("pid_type", ("flat", *PERSISTENT))
def test_refused_batch_digit_variants_keep_their_remainders_under_every_pid(
    kernel, pid_type
) -> None:
    bound = _bind(kernel, _head_split_args())
    assert _digit_map(bound) is None
    config = helion.Config(block_sizes=[1, 128, 64, 64], pid_type=pid_type)
    bound.config_spec.normalize(config)
    code = bound.to_code(config)
    # No native role pipeline exists here, so nothing may omit the remainder:
    # the generic body keeps every Python-semantics `%` it computes.
    assert "tcgen05_tma_warp" not in code
    assert ".iterator + cute.crd2idx((" not in code
    assert "_cute_python_mod(" in code


def test_rank1_index_on_another_axis_is_never_a_digit_map() -> None:
    args = (
        torch.empty((Z, N, K), dtype=torch.bfloat16),
        torch.empty((Z, K, N), dtype=torch.bfloat16),
        torch.empty((Z // H, N, H, N), dtype=torch.bfloat16),
    )
    bound = _bind(rank1_m_index, args)
    assert _digit_map(bound) is None


@pytest.mark.parametrize(
    "kernel", [quotient_with_mod, quotient_with_mod_store], ids=lambda k: k.fn.__name__
)
def test_remainders_outside_the_quotient_grammar_keep_the_generic_route(kernel) -> None:
    bound = _bind(kernel, _quotient_args(16))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    # Under every pid the generic body compiles and keeps its remainder statement;
    # nothing omits it.
    for pid_type in ("flat", *PERSISTENT):
        code = bound.to_code(_with_pid(bound, _default(bound), pid_type))
        assert "tcgen05_tma_warp" not in code
        assert "_cute_python_mod(" in code


@pytest.mark.parametrize("pid_type", PERSISTENT)
def test_explicit_persistent_tiles_on_the_generic_route_stay_loud(pid_type) -> None:
    # The existing generic-route rule: an explicit persistent tcgen05 pid with
    # explicit tiles needs the tcgen05 matmul plan, so it refuses loudly instead
    # of compiling. This guards the rule; it is not evidence about remainders.
    bound = _bind(quotient_with_mod, _quotient_args(16))
    config = helion.Config(block_sizes=[64, 32, 64], pid_type=pid_type)
    bound.config_spec.normalize(config)
    with pytest.raises(
        exc.BackendUnsupported, match="requires the tcgen05 matmul plan"
    ):
        bound.to_code(config)
