"""The ``warp_mma`` matmul family: register-MMA GEMM tiles for tiny problems.

``cute_matmul_family="warp_mma"`` replaces the tcgen05 pipeline of a plain
dense GEMM (``acc = zeros; for k: acc = dot(A[m, k], B[k, n]); out[m, n] =
epilogue(acc)``) by the structure the Helion-Triton winner of the 256^3 fp8
GEMM uses: a CTA of ``cute_warp_mma_warps`` warps owns one ``bm x bn`` output
tile (16x8 .. 64x64), stages the whole K extent of its A and B tiles into
XOR-swizzled shared memory with ``cp.async`` packets issued at kernel entry
(one commit group per ``bk`` K chunk, all issued before the first wait), and
runs ``ldmatrix`` + ``mma.sync.m16n8k16`` (fp16/bf16) or ``m16n8k32`` (e4m3)
with fp32 accumulators in a fixed K order.  The epilogue chain the store
analysis admits (casts, pointwise ops, bias rows, residual tiles) runs on the
fp32 fragments in registers and the outputs are stored as 4- or 8-byte pairs.
No TMEM, no TMA descriptors, no mbarrier pipeline: the per-CTA chain is one
DRAM latency plus a few hundred nanoseconds.

Measured in the campaign regime on B200 (device span, us; hand kernels of
this exact structure): fp8 256^3 e4m3 -> fp16 16x8 one-warp tiles 2.24 vs
cuBLAS 2.62 and the Helion-Triton winner 2.27 (tcgen05 64x16 one-CTA tile
2.51); fp16 256^3 + bias 64x8 four-warp tiles 2.64 vs cuBLAS 2.88 (tcgen05
2.85); bmm (4, 64, 128, 128) fp16 64x8 four-warp tiles 2.19 vs cuBLAS 2.66
(tcgen05 2.58).  The family loses once the problem stops being
latency-bound (fp16 512^3: 3.81 vs cuBLAS 3.49; 1024^3: 3-5x slower), so it
is admitted to the search only below ``WARP_MMA_MAX_MACS`` multiply-adds and
below one 128x128 tile per SM; explicit configs outside that region fail
closed with the reason.

Fissioned kernels (``cute_materialize_transformed_operands``, e.g. the
bf16 x int16 example at (128, 256, 128)) are admitted: the family replaces the
device body of the region that holds the GEMM's root tile loop and K loop,
while the other regions (the operand materialization) stay separate launches
with their own lowering.  The admission therefore gates on the single matmul,
not on the kernel's graph count; ``_require_complete_root`` proves the GEMM's
own root complete at codegen.
"""

from __future__ import annotations

import ast
import dataclasses
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .cute_epilogue import Tcgen05UnaryEpilogueChain
    from .cute_epilogue import _AuxiliaryTensorLoadExpr

WARP_MMA_FAMILY_KEY = "cute_matmul_family"
MATMUL_FAMILY_TCGEN05 = "tcgen05"
MATMUL_FAMILY_WARP_MMA = "warp_mma"
MATMUL_FAMILIES: tuple[str, ...] = (MATMUL_FAMILY_TCGEN05, MATMUL_FAMILY_WARP_MMA)
WARP_MMA_WARPS_KEY = "cute_warp_mma_warps"
WARP_MMA_WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
WARP_MMA_DEFAULT_WARPS = 1
WARP_MMA_PLAN_KIND = "helion_warp_mma_gemm"
WARP_MMA_MIN_BM = 16
WARP_MMA_MAX_BM = 64
WARP_MMA_MIN_BN = 8
WARP_MMA_MAX_BN = 64
# Static shared memory one CTA may stage (the DSL opt-in maximum is 227 KiB;
# the margin covers the allocator's alignment padding).
WARP_MMA_SMEM_BUDGET = 200 * 1024
# The register-MMA tiles re-read their operands once per tile and run the
# K loop on one warp's mma.sync chain; both scale with the problem's
# multiply-adds while the tcgen05 pipeline amortizes them.  256^3 (16.8M
# MACs) wins by 0.25-0.4 us, 512^3 (134M) loses by 0.3-0.6 us, 1024^3 loses
# 3-5x; the cut sits between the two measured regimes.
WARP_MMA_MAX_MACS = 1 << 25
# Seeds for admitted problems (measured winners, campaign regime): fp8 256^3
# 16x8 one-warp; fp16 256^3 + bias and bmm (4, 64, 128, 128) 64x8 four-warp,
# then 32x32 four-warp, 32x16 two-warp and 16x16 one-warp within 0.1 us.
WARP_MMA_SEED_TILES: tuple[tuple[int, int, int], ...] = (
    (16, 8, 1),
    (64, 8, 4),
    (32, 32, 4),
    (32, 16, 2),
    (16, 16, 1),
)
_RUNTIME = "_helion_warp_mma"
_PREAMBLE = f"import helion._compiler.cute._warp_mma_runtime as {_RUNTIME}\n"

_DTYPE_KIND: dict[torch.dtype, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
}
_OUT_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)
_AUX_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)


def warp_mma_dtype_kind(dtype: torch.dtype) -> str | None:
    """``f16`` / ``bf16`` / ``e4m3`` for the operand dtypes the family runs."""
    return _DTYPE_KIND.get(dtype)


def warp_mma_k_step(dtype: torch.dtype) -> int:
    """K extent of one ``mma.sync`` (32 for e4m3, 16 for 16-bit operands)."""
    return 32 if dtype == torch.float8_e4m3fn else 16


def warp_mma_warp_layout(bm: int, bn: int, warps: int) -> tuple[int, int] | None:
    """``(wm, wn)``: warps along M and N (M split first, each warp at least a
    16x8 atom), or None when ``warps`` cannot tile ``bm x bn``."""
    if warps < 1 or bm % 16 or bn % 8:
        return None
    wm = min(warps, bm // 16)
    if warps % wm:
        return None
    wn = warps // wm
    if (bn // 8) % wn:
        return None
    return wm, wn


def warp_mma_max_legal_warps(bm: int, bn: int) -> int:
    """The largest searchable warp count that tiles ``bm x bn``."""
    legal = [w for w in WARP_MMA_WARP_CHOICES if warp_mma_warp_layout(bm, bn, w)]
    return max(legal) if legal else 1


def warp_mma_smem_bytes(*, bm: int, bn: int, k: int, elem_bytes: int) -> int:
    """Shared memory of the whole-K A and B stages of one ``bm x bn`` tile."""
    return (bm + bn) * k * elem_bytes


def warp_mma_tile_reason(
    *,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    warps: int,
    dtype: torch.dtype,
) -> str | None:
    """Why ``(bm, bn, bk, warps)`` cannot run the family on ``m x n x k``."""
    kind = warp_mma_dtype_kind(dtype)
    for name, value in (("block_m", bm), ("block_n", bn), ("block_k", bk)):
        # The swizzle is a bijection and the warp layout splits evenly only
        # for power-of-two tiles (``BlockSizeSpec`` rejects the rest first;
        # the invariant is kept local to the family).
        if value <= 0 or value & (value - 1):
            return f"{name} {value} must be a power of two"
    if kind is None:
        return f"operand dtype {dtype} is not fp16, bf16 or e4m3"
    k_step = warp_mma_k_step(dtype)
    if not WARP_MMA_MIN_BM <= bm <= WARP_MMA_MAX_BM or bm % 16:
        return f"block_m {bm} must be a multiple of 16 in [{WARP_MMA_MIN_BM}, {WARP_MMA_MAX_BM}]"
    if not WARP_MMA_MIN_BN <= bn <= WARP_MMA_MAX_BN or bn % 8:
        return f"block_n {bn} must be a multiple of 8 in [{WARP_MMA_MIN_BN}, {WARP_MMA_MAX_BN}]"
    if bk % k_step or bk > k:
        return f"block_k {bk} must be a multiple of {k_step} no larger than K={k}"
    if m % bm or n % bn or k % bk:
        return f"tile {bm}x{bn}x{bk} must divide the {m}x{n}x{k} problem"
    if warp_mma_warp_layout(bm, bn, warps) is None:
        return f"{warps} warps cannot tile a {bm}x{bn} block into 16x8 atoms"
    smem = warp_mma_smem_bytes(bm=bm, bn=bn, k=k, elem_bytes=dtype.itemsize)
    if smem > WARP_MMA_SMEM_BUDGET:
        return f"{bm}x{bn} tiles stage {smem} B of shared memory over K={k}"
    return None


def warp_mma_admission_reason(
    *,
    static_m: int | None,
    static_n: int | None,
    static_k: int | None,
    leading: int,
    dtype: torch.dtype,
    lhs_major: str | None,
    rhs_major: str | None,
    num_sms: int,
    lhs_strides: tuple[int, ...] | None = None,
    rhs_strides: tuple[int, ...] | None = None,
    operands_permuted: bool = False,
) -> str | None:
    """Why the family is not admitted to a GEMM's search surface (None: it is).

    ``lhs_strides`` / ``rhs_strides`` are the static element strides of the
    operands' source tensors (None when unknown: not admitted) and
    ``operands_permuted`` says whether either operand reaches the matmul
    through a permute; both mirror the detector's gates so no admitted
    kernel receives seeds that fail at codegen.

    The structural facts here are the ones the search plan knows before the
    kernel body is analyzed; the body match (``detect_warp_mma_gemm``) is the
    second, complete gate.  Fissioned kernels are admitted when their GEMM
    region is one plain matmul (the caller passes ``warp_mma_plain_kernel`` =
    exactly one matmul candidate, regardless of how many fission regions the
    kernel has): the family lowers that region's root tile loop and leaves the
    other regions to their own launches.
    """
    kind = warp_mma_dtype_kind(dtype)
    if kind is None:
        return f"operand dtype {dtype} is not fp16, bf16 or e4m3"
    if static_m is None or static_n is None or static_k is None:
        return "the problem extents must be static"
    if lhs_major != "row":
        return "A must be K-contiguous"
    if rhs_major not in ("row", "col"):
        return "B must be N- or K-contiguous"
    if kind == "e4m3" and rhs_major != "col":
        return "an e4m3 B must be K-contiguous (ldmatrix.trans moves 16-bit elements)"
    if operands_permuted:
        return "permuted operands are not supported"
    if lhs_strides is None or rhs_strides is None:
        return "the operand strides must be static"
    elem = dtype.itemsize
    if lhs_strides[-1] != 1 or (lhs_strides[-2] * elem) % 16:
        return "A rows must be K-contiguous and 16-byte aligned"
    if rhs_major == "col":
        if rhs_strides[-2] != 1 or (rhs_strides[-1] * elem) % 16:
            return "a K-contiguous B needs 16-byte aligned columns"
    elif rhs_strides[-1] != 1 or (rhs_strides[-2] * elem) % 16:
        return "an N-contiguous B needs 16-byte aligned rows"
    for strides in (lhs_strides, rhs_strides):
        if len(strides) == 3 and (strides[0] * elem) % 16:
            return "batch strides must be 16-byte aligned"
    if static_m % WARP_MMA_MIN_BM or static_n % WARP_MMA_MIN_BN:
        return f"M must be a multiple of {WARP_MMA_MIN_BM} and N of {WARP_MMA_MIN_BN}"
    if static_k % warp_mma_k_step(dtype):
        return f"K must be a multiple of {warp_mma_k_step(dtype)}"
    if num_sms <= 0:
        return "the device SM count is unknown"
    tiles = leading * -(-static_m // 128) * -(-static_n // 128)
    if tiles >= num_sms:
        return "the 128x128 tiling fills the machine"
    if leading * static_m * static_n * static_k > WARP_MMA_MAX_MACS:
        return "the problem is beyond the latency-bound regime"
    smem = warp_mma_smem_bytes(
        bm=WARP_MMA_MIN_BM, bn=WARP_MMA_MIN_BN, k=static_k, elem_bytes=dtype.itemsize
    )
    if smem > WARP_MMA_SMEM_BUDGET:
        return "the K extent does not fit the shared-memory budget"
    return None


@dataclasses.dataclass(frozen=True)
class WarpMmaAuxMatch:
    """One auxiliary load leaf of a store's epilogue chain."""

    leaf: _AuxiliaryTensorLoadExpr
    fake: torch.Tensor
    # rowvec: rank-1 ``bias[tile_n]``; row0: ``bias[tile_m, tile_n]`` on a
    # (1, N) tensor; colvec: a (M, N) view with stride 0 along N; exact: the
    # output's own (batch,) m, n tile.
    kind: str
    strides: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class WarpMmaStoreMatch:
    store_node: Node
    chain: Tcgen05UnaryEpilogueChain
    fake: torch.Tensor
    strides: tuple[int, ...]
    aux: tuple[WarpMmaAuxMatch, ...]


@dataclasses.dataclass(frozen=True)
class CuteWarpMmaGemmMatch:
    """The matched plain GEMM and the tile the config selected."""

    mma_node: Node
    lhs_fake: torch.Tensor
    rhs_fake: torch.Tensor
    lhs_strides: tuple[int, ...]
    rhs_strides: tuple[int, ...]
    rhs_k_major: bool
    batch: int
    m: int
    n: int
    k: int
    bm: int
    bn: int
    bk: int
    warps: int
    stores: tuple[WarpMmaStoreMatch, ...]

    @property
    def dtype(self) -> torch.dtype:
        return self.lhs_fake.dtype

    @property
    def threads(self) -> int:
        return 32 * self.warps

    @property
    def total_tiles(self) -> int:
        return self.batch * (self.m // self.bm) * (self.n // self.bn)

    @property
    def warp_layout(self) -> tuple[int, int]:
        layout = warp_mma_warp_layout(self.bm, self.bn, self.warps)
        assert layout is not None
        return layout


def _decline(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"warp_mma GEMM family: {reason}")


def _static_int(env: CompileEnvironment, value: object) -> int | None:
    from ...language.matmul_ops import _static_dim_value

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        return _static_dim_value(env, value)
    return None


def _static_strides(env: CompileEnvironment, fake: torch.Tensor) -> tuple[int, ...]:
    strides = []
    for stride in fake.stride():
        value = _static_int(env, stride)
        if value is None:
            raise _decline("tensor strides must be static")
        strides.append(value)
    return tuple(strides)


def _canonical_block_ids(
    env: CompileEnvironment, block_ids: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    if block_ids is None:
        return None
    return tuple(env.canonical_block_id(block_id) for block_id in block_ids)


def _config_int(config: object, key: str, default: int) -> int:
    value = cast("dict[str, object]", config).get(key, default)  # pyrefly: ignore[missing-attribute]
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _aux_kind(
    leaf: _AuxiliaryTensorLoadExpr, fake: torch.Tensor, strides: tuple[int, ...]
) -> str:
    axis = leaf.broadcast_axis
    if axis == 1:
        if fake.ndim != 1:
            raise _decline("a trailing-axis broadcast aux row must be rank 1")
        return "rowvec"
    if axis == 0:
        if fake.ndim != 2 or fake.size(0) != 1:
            raise _decline("a leading-axis broadcast aux must be a (1, N) tensor")
        return "row0"
    if axis == 2:
        if fake.ndim != 2 or strides[1] != 0:
            raise _decline("a per-row aux must be a stride-0-N view")
        return "colvec"
    if axis is None:
        if strides[-1] != 1:
            raise _decline("an exact-shape aux tile must be N-contiguous")
        return "exact"
    raise _decline(f"unsupported aux broadcast axis {axis}")


def detect_warp_mma_gemm(
    fn: DeviceFunction, block_ids: list[int], *, config: object
) -> bool:
    """Match the K device loop ``block_ids`` of a plain dense GEMM and record
    the family's plan on ``fn.cute_state``.

    Called for the K loop when the config selects the family; every mismatch
    raises ``BackendUnsupported`` with the reason (the knob is never silently
    traded for another lowering).
    """
    from ..host_function import HostFunction
    from ..indexing_strategy import exact_tile_block_ids
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_epilogue import _AuxiliaryTensorLoadExpr
    from .cute_mma import _tcgen05_tma_matrix_major
    from .cute_mma import analyze_cute_mma_node

    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        raise _decline("the kernel must have one root tile grid")
    root = _canonical_block_ids(env, tuple(device_ir.grid_block_ids[0]))
    assert root is not None
    if len(block_ids) != 1 or env.canonical_block_id(block_ids[0]) in root:
        raise _decline("the K reduction must be one device loop")
    k_block_id = env.canonical_block_id(block_ids[0])

    graphs = fn.codegen.codegen_graphs
    candidates = []
    for graph_info in graphs:
        if getattr(graph_info, "block_ids", None) != block_ids:
            continue
        for node in graph_info.graph.nodes:
            candidate = analyze_cute_mma_node(node, graphs=graphs)
            if (
                candidate is not None
                and env.canonical_block_id(candidate.operands.k_block_id) == k_block_id
            ):
                candidates.append((node, candidate))
    if len(candidates) != 1:
        raise _decline("the K loop must hold exactly one dense matmul")
    mma_node, candidate = candidates[0]
    if candidate.requires_accumulator_seed:
        raise _decline("the accumulator must start at zero")
    operands = candidate.operands
    lhs, rhs = operands.lhs, operands.rhs
    if (
        lhs.source_to_logical_order is not None
        or rhs.source_to_logical_order is not None
    ):
        raise _decline("permuted operands are not supported")
    if rhs.rhs_is_grouped or rhs.grouped_k_mask is not None:
        raise _decline("grouped operands are not supported")
    m_block_id = env.canonical_block_id(operands.m_block_id)
    n_block_id = env.canonical_block_id(operands.n_block_id)
    leading_block_id = operands.leading_passthrough_block_id
    expected_root = (
        (m_block_id, n_block_id)
        if leading_block_id is None
        else (env.canonical_block_id(leading_block_id), m_block_id, n_block_id)
    )
    if root != expected_root:
        raise _decline("the root grid must tile (batch,) M, N in that order")
    lhs_fake, rhs_fake = lhs.source_fake, rhs.source_fake
    if lhs_fake.dtype != rhs_fake.dtype or warp_mma_dtype_kind(lhs_fake.dtype) is None:
        raise _decline("operands must share an fp16, bf16 or e4m3 dtype")
    lhs_major = _tcgen05_tma_matrix_major(lhs_fake)
    rhs_major = _tcgen05_tma_matrix_major(rhs_fake)
    m = _static_int(env, lhs.matrix_rows)
    k = _static_int(env, lhs.matrix_cols)
    n = _static_int(env, rhs.matrix_cols)
    if m is None or n is None or k is None:
        raise _decline("the problem extents must be static")
    batch = 1
    if leading_block_id is not None:
        leading_operand = lhs if lhs.is_leading_passthrough else rhs
        batch_extent = _static_int(env, leading_operand.logical_fake.shape[0])
        if batch_extent is None:
            raise _decline("the batch extent must be static")
        batch = batch_extent
        batch_block = env.block_sizes[leading_block_id].from_config(config)  # pyrefly: ignore[bad-argument-type]
        if batch_block != 1:
            raise _decline("the batch tile must have block size 1")
    lhs_strides = _static_strides(env, lhs_fake)
    rhs_strides = _static_strides(env, rhs_fake)
    reason = warp_mma_admission_reason(
        static_m=m,
        static_n=n,
        static_k=k,
        leading=batch,
        dtype=lhs_fake.dtype,
        lhs_major=lhs_major,
        rhs_major=rhs_major,
        num_sms=_device_sm_count(lhs_fake.device),
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
    )
    if reason is not None:
        raise _decline(reason)
    bm = env.block_sizes[m_block_id].from_config(config)  # pyrefly: ignore[bad-argument-type]
    bn = env.block_sizes[n_block_id].from_config(config)  # pyrefly: ignore[bad-argument-type]
    bk = env.block_sizes[k_block_id].from_config(config)  # pyrefly: ignore[bad-argument-type]
    if not isinstance(bm, int) or not isinstance(bn, int) or not isinstance(bk, int):
        raise _decline("the tile sizes must be static")
    warps = _config_int(config, WARP_MMA_WARPS_KEY, WARP_MMA_DEFAULT_WARPS)
    tile_reason = warp_mma_tile_reason(
        m=m, n=n, k=k, bm=bm, bn=bn, bk=bk, warps=warps, dtype=lhs_fake.dtype
    )
    if tile_reason is not None:
        raise _decline(tile_reason)
    # (The operand stride and batch-stride alignment gates are part of the
    # admission above.)
    rhs_k_major = rhs_major == "col"

    analyzed = analyze_tcgen05_matmul_store_chains(graphs, mma_node)
    if analyzed is None:
        raise _decline("every store of the accumulator must be a supported epilogue")
    stores: list[WarpMmaStoreMatch] = []
    for store_node, chain in analyzed:
        out_node = store_node.args[0]
        out_fake = (
            out_node.meta.get("val") if isinstance(out_node, torch.fx.Node) else None
        )
        if not isinstance(out_fake, torch.Tensor):
            raise _decline("the output must be a tensor")
        if out_fake.dtype not in _OUT_DTYPES:
            raise _decline(f"output dtype {out_fake.dtype} is not fp16, bf16 or fp32")
        if out_fake.ndim != len(expected_root):
            raise _decline("the output rank must match the (batch,) M, N grid")
        out_shape = tuple(_static_int(env, size) for size in out_fake.shape)
        expected_shape = (m, n) if leading_block_id is None else (batch, m, n)
        if out_shape != expected_shape:
            raise _decline("the output must cover the whole (batch,) M x N problem")
        store_index = store_node.args[1] if len(store_node.args) > 1 else None
        if not isinstance(store_index, (list, tuple)):
            raise _decline("the output store must index whole tiles")
        store_ids = _canonical_block_ids(env, exact_tile_block_ids(env, store_index))
        if store_ids != expected_root:
            raise _decline(
                "the output store must index (batch,) tile_m, tile_n in order"
            )
        out_strides = _static_strides(env, out_fake)
        oes = out_fake.dtype.itemsize
        pair_bytes = 2 * oes
        if out_strides[-1] != 1 or (out_strides[-2] * oes) % pair_bytes:
            raise _decline("output rows must be N-contiguous with even-aligned rows")
        if out_fake.ndim == 3 and (out_strides[0] * oes) % pair_bytes:
            raise _decline("the output batch stride must keep the store pairs aligned")
        aux: list[WarpMmaAuxMatch] = []
        for leaf in chain.auxiliary_tensor_loads:
            if not isinstance(leaf, _AuxiliaryTensorLoadExpr):
                raise _decline("unexpected epilogue leaf")
            aux_node = leaf.load_node.args[0] if leaf.load_node.args else None
            aux_fake = (
                aux_node.meta.get("val")
                if isinstance(aux_node, torch.fx.Node)
                else None
            )
            if not isinstance(aux_fake, torch.Tensor):
                raise _decline("an epilogue aux operand must be a tensor")
            if aux_fake.dtype not in _AUX_DTYPES:
                raise _decline(f"aux dtype {aux_fake.dtype} is not fp16, bf16 or fp32")
            aux_strides = _static_strides(env, aux_fake)
            kind = _aux_kind(leaf, aux_fake, aux_strides)
            if kind == "exact":
                if (
                    aux_fake.ndim != len(expected_root)
                    or tuple(_static_int(env, size) for size in aux_fake.shape)
                    != expected_shape
                ):
                    raise _decline("an exact-shape aux must cover the whole problem")
                aux_index = (
                    leaf.load_node.args[1] if len(leaf.load_node.args) > 1 else None
                )
                if not isinstance(aux_index, (list, tuple)):
                    raise _decline("an exact-shape aux must index whole tiles")
                aux_ids = _canonical_block_ids(
                    env, exact_tile_block_ids(env, aux_index)
                )
                if aux_ids != expected_root:
                    raise _decline(
                        "an exact-shape aux must index (batch,) tile_m, tile_n"
                    )
                aes = aux_fake.dtype.itemsize
                if (aux_strides[-2] * aes) % (2 * aes) or (
                    aux_fake.ndim == 3 and (aux_strides[0] * aes) % (2 * aes)
                ):
                    raise _decline(
                        "exact-shape aux rows must keep the load pairs aligned"
                    )
            elif kind in ("rowvec", "row0"):
                if aux_strides[-1] != 1:
                    raise _decline("an aux row must be contiguous")
                if kind == "row0" and aux_fake.size(1) != n:
                    raise _decline("an aux row must span N")
                if kind == "rowvec" and _static_int(env, aux_fake.shape[0]) != n:
                    raise _decline("an aux row must span N")
            elif _static_int(env, aux_fake.shape[0]) != m:
                raise _decline("a per-row aux must span M")
            aux.append(
                WarpMmaAuxMatch(
                    leaf=leaf, fake=aux_fake, kind=kind, strides=aux_strides
                )
            )
        if len(aux) > 4:
            raise _decline("at most four aux operands per store")
        stores.append(
            WarpMmaStoreMatch(
                store_node=store_node,
                chain=chain,
                fake=out_fake,
                strides=out_strides,
                aux=tuple(aux),
            )
        )
    _require_complete_root(
        graphs,
        k_block_ids=block_ids,
        store_nodes={store.store_node for store in stores},
    )
    fn.cute_state.warp_mma_gemm_plan = CuteWarpMmaGemmMatch(
        mma_node=mma_node,
        lhs_fake=lhs_fake,
        rhs_fake=rhs_fake,
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
        rhs_k_major=rhs_k_major,
        batch=batch,
        m=m,
        n=n,
        k=k,
        bm=bm,
        bn=bn,
        bk=bk,
        warps=warps,
        stores=tuple(stores),
    )
    return True


def _require_complete_root(
    graphs: Sequence[object], *, k_block_ids: list[int], store_nodes: set[Node]
) -> None:
    """The family replaces the whole device body: decline unless the kernel is
    exactly the root tile loop around the matched GEMM.

    The K loop body is already exclusive (``analyze_cute_mma_node``); here the
    root graph that calls the K loop may hold only the matched stores and
    their read-only chains, the accumulator init, the K-loop node and
    read-only (pure) nodes -- an unrelated store or atomic in that root fails
    closed, where the whole-body replacement would otherwise drop it silently.
    Other root graphs are separate fission regions (their own launches, e.g.
    the operand materialization of the bf16 x int16 example) and are left to
    their own lowering; any other device loop must belong to one of them.
    """
    from ...language import _tracing_ops
    from ...language import creation_ops
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from .cute_mma import _is_tracing_for_loop_node
    from .fold_noop_stores import _is_read_only

    k_graphs = [
        graph for graph in graphs if getattr(graph, "block_ids", None) == k_block_ids
    ]
    if len(k_graphs) != 1 or not isinstance(k_graphs[0], ForLoopGraphInfo):
        raise _decline("the K reduction must be one device loop")
    k_graph_id = k_graphs[0].graph_id
    roots = [
        graph
        for graph in graphs
        if isinstance(graph, RootGraphInfo)
        and any(
            _is_tracing_for_loop_node(node) and node.args[0] == k_graph_id
            for node in graph.graph.nodes
        )
    ]
    if len(roots) != 1:
        raise _decline("exactly one root tile loop must call the K loop")
    root = roots[0]
    loops = [node for node in root.graph.nodes if _is_tracing_for_loop_node(node)]
    if len(loops) != 1:
        raise _decline("the root tile loop must hold exactly the K loop")
    for node in root.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        if node in store_nodes or node is loops[0]:
            continue
        if node.op == "call_function" and node.target in (
            creation_ops.full,
            _tracing_ops._new_var,
            _tracing_ops._phi,
        ):
            continue
        if not _is_read_only(node):
            raise _decline(
                "the root tile loop may only hold the GEMM and its stores "
                f"({node.target} is not part of it)"
            )


def _device_sm_count(device: torch.device) -> int:
    from ...language.matmul_ops import _cuda_num_sms_or_zero

    return _cuda_num_sms_or_zero(device)


def emit_warp_mma_gemm_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the register-MMA runtime helpers."""
    if getattr(cg, "_helion_warp_mma_module_emitted", False):
        return
    cg._helion_warp_mma_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast.parse(_PREAMBLE).body:
        cg.module_statements.append(stmt)


def codegen_warp_mma_gemm(cg: GenerateAST) -> bool:
    """Replace the device body by the register-MMA GEMM of the recorded match."""
    from .memory_ops import cute_tensor_base_is_aligned

    df = cg.device_function
    match = df.cute_state.warp_mma_gemm_plan
    if match is None:
        return False
    env = CompileEnvironment.current()
    a_arg = df.tensor_arg(match.lhs_fake)
    b_arg = df.tensor_arg(match.rhs_fake)
    for fake, what in ((match.lhs_fake, "A"), (match.rhs_fake, "B")):
        if not cute_tensor_base_is_aligned(env, fake, 16):
            raise _decline(f"the {what} base must be provably 16-byte aligned")
    out_args = []
    aux_args: list[list[str]] = []
    # Outputs and aux operands move as 4- / 8-byte pairs when their base is
    # provably pair-aligned (inputs, exact views of inputs, fresh wrapper
    # allocations); anything else (a closure-captured bias row, say) moves
    # element by element, which torch's element alignment always allows.
    out_pairs: list[bool] = []
    aux_pairs: list[list[bool]] = []
    for store in match.stores:
        out_arg = df.tensor_arg(store.fake)
        out_args.append(out_arg.name)
        out_pairs.append(
            cute_tensor_base_is_aligned(env, store.fake, 2 * store.fake.dtype.itemsize)
        )
        names: list[str] = []
        pairs: list[bool] = []
        for aux in store.aux:
            aux_arg = df.tensor_arg(aux.fake)
            names.append(aux_arg.name)
            pairs.append(
                cute_tensor_base_is_aligned(env, aux.fake, 2 * aux.fake.dtype.itemsize)
            )
        aux_args.append(names)
        aux_pairs.append(pairs)
    df.placeholder_args.update((a_arg.name, b_arg.name, *out_args))
    for names in aux_args:
        df.placeholder_args.update(names)
    emit_warp_mma_gemm_module_statements(cg)
    plan: dict[str, object] = {
        "kind": WARP_MMA_PLAN_KIND,
        "total_tiles": match.total_tiles,
        "threads": match.threads,
        "bm": match.bm,
        "bn": match.bn,
        "bk": match.bk,
        "warps": match.warps,
        # ``*_name`` keys resolve to launcher argument positions (the body
        # bakes every stride, so the launcher never stages these operands
        # through copies: under-aligned A / B bases are refused at compile
        # time, outputs and aux rows fall back to element accesses).
        "lhs_name": a_arg.name,
        "rhs_name": b_arg.name,
        "out_name": out_args[0],
    }
    first_aux = aux_args[0] if aux_args else []
    if first_aux:
        plan["epi_aux_count"] = len(first_aux)
        for index, name in enumerate(first_aux):
            plan[f"epi_aux{index}_name"] = name
    cg.cute_wrapper_plans.append(plan)
    df.body = emit_warp_mma_gemm_device_body(
        df,
        match,
        a_name=a_arg.name,
        b_name=b_arg.name,
        out_names=out_args,
        aux_names=aux_args,
        out_pairs=out_pairs,
        aux_pairs=aux_pairs,
    )
    df.preamble = []
    return True


def _swizzled_chunk(row: str, chunk: str, chunks_per_row: int) -> str:
    """Swizzled 16-byte chunk index: the eight consecutive rows one ldmatrix
    reads at one logical chunk land on eight distinct bank groups (rows
    narrower than 128 B already spread over the banks and share a group)."""
    if chunks_per_row >= 8:
        return f"(({chunk}) ^ (({row}) % 8))"
    if chunks_per_row == 1:
        return f"({chunk})"
    rows_per_group = 8 // chunks_per_row
    return f"(({chunk}) ^ ((({row}) // {rows_per_group}) % {chunks_per_row}))"


def emit_warp_mma_gemm_device_body(
    df: DeviceFunction,
    match: CuteWarpMmaGemmMatch,
    *,
    a_name: str,
    b_name: str,
    out_names: list[str],
    aux_names: list[list[str]],
    out_pairs: list[bool] | None = None,
    aux_pairs: list[list[bool]] | None = None,
) -> list[ast.AST]:
    """Render the register-MMA GEMM body (see the module docstring).

    ``out_pairs`` / ``aux_pairs`` say which outputs and aux operands may move
    as element pairs (pair-aligned bases); the rest move element by element.
    """
    if out_pairs is None:
        out_pairs = [True] * len(match.stores)
    if aux_pairs is None:
        aux_pairs = [[True] * len(store.aux) for store in match.stores]
    from .row_resident_codegen import _scalar_epilogue

    rt = _RUNTIME
    dtype = match.dtype
    kind = warp_mma_dtype_kind(dtype)
    assert kind is not None
    es = dtype.itemsize
    bm, bn, bk, warps = match.bm, match.bn, match.bk, match.warps
    m, n, k = match.m, match.n, match.k
    wm, wn = match.warp_layout
    wtm, wtn = bm // wm, bn // wn
    atoms_m, atoms_n = wtm // 16, wtn // 8
    k_step = warp_mma_k_step(dtype)
    steps = bk // k_step
    nk = k // bk
    threads = match.threads
    npn, npm = n // bn, m // bm
    mma = {"e4m3": f"{rt}.mma_e4m3", "f16": f"{rt}.mma_f16", "bf16": f"{rt}.mma_bf16"}[
        kind
    ]
    # ---- global strides in bytes
    a_batch = match.lhs_strides[0] * es if match.lhs_fake.ndim == 3 else 0
    a_row = match.lhs_strides[-2] * es
    b_batch = match.rhs_strides[0] * es if match.rhs_fake.ndim == 3 else 0
    if match.rhs_k_major:
        b_row = match.rhs_strides[-1] * es  # one n column of B (K contiguous)
    else:
        b_row = match.rhs_strides[-2] * es  # one k row of B (N contiguous)
    # ---- smem tiles per K chunk
    ra = bk * es
    cpr_a = ra // 16
    a_chunk = bm * ra
    if match.rhs_k_major:
        rb, b_rows = bk * es, bn
    else:
        rb, b_rows = bn * es, bk
    cpr_b = rb // 16
    b_chunk = b_rows * rb
    smem_bytes = nk * (a_chunk + b_chunk)
    a_packets = a_chunk // 16
    b_packets = b_chunk // 16
    lines: list[str] = []
    emit = lines.append
    emit("wm_tidx, _, _ = cute.arch.thread_idx()")
    emit("wm_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
    emit("wm_lane = wm_tidx % 32")
    emit("wm_g = wm_lane // 4")
    emit("wm_c = wm_lane % 4")
    emit("wm_l8 = wm_lane % 8")
    emit("wm_lq = wm_lane // 8")
    emit("wm_pid = cutlass.Int32(cute.arch.block_idx()[0])")
    emit(f"wm_pn = wm_pid % {npn}")
    emit(f"wm_pm = (wm_pid // {npn}) % {npm}")
    emit(f"wm_pb = wm_pid // {npn * npm}")
    emit(f"wm_wmi = wm_warp // {wn}")
    emit(f"wm_wni = wm_warp % {wn}")
    emit(f"wm_m0 = wm_pm * {bm}")
    emit(f"wm_n0 = wm_pn * {bn}")
    emit(f"wm_smem = cute.arch.alloc_smem(cutlass.Uint8, {smem_bytes}, alignment=128)")
    emit("wm_s_a = cutlass.Int32(wm_smem.toint())")
    emit(f"wm_s_b = wm_s_a + {nk * a_chunk}")
    emit(
        f"wm_a_g = {a_name}.iterator.toint() + cutlass.Int64(wm_pb) * {a_batch} "
        f"+ cutlass.Int64(wm_m0) * {a_row}"
    )
    if match.rhs_k_major:
        emit(
            f"wm_b_g = {b_name}.iterator.toint() + cutlass.Int64(wm_pb) * {b_batch} "
            f"+ cutlass.Int64(wm_n0) * {b_row}"
        )
    else:
        emit(
            f"wm_b_g = {b_name}.iterator.toint() + cutlass.Int64(wm_pb) * {b_batch} "
            f"+ cutlass.Int64(wm_n0) * {es}"
        )
    # The lane's output / aux coordinates: rows g and g+8 of every atom row,
    # columns 2c and 2c+1 of every atom column.
    emit(f"wm_row = wm_m0 + wm_wmi * {wtm} + wm_g")
    emit(f"wm_col = wm_n0 + wm_wni * {wtn} + 2 * wm_c")

    # ---- aux loads: issued at entry so their latency overlaps the operands.
    aux_value: dict[tuple[int, int, int, int, int], str] = {}
    for si, store in enumerate(match.stores):
        for li, aux in enumerate(store.aux):
            name = aux_names[si][li]
            adt = aux.fake.dtype
            aes = adt.itemsize
            if aux.kind in ("rowvec", "row0"):
                col_stride = aux.strides[-1] * aes
                emit(
                    f"wm_x{si}_{li} = {name}.iterator.toint() + cutlass.Int64(wm_col) * {col_stride}"
                )
                for an in range(atoms_n):
                    off = an * 8 * col_stride
                    lo, hi = f"wm_v{si}_{li}_{an}_0", f"wm_v{si}_{li}_{an}_1"
                    _emit_pair_load(
                        emit,
                        lo,
                        hi,
                        f"wm_x{si}_{li} + cutlass.Int64({off})",
                        adt,
                        rt,
                        paired=aux_pairs[si][li],
                    )
                    for am in range(atoms_m):
                        for j in range(4):
                            aux_value[(si, li, am, an, j)] = lo if j % 2 == 0 else hi
            elif aux.kind == "colvec":
                row_stride = aux.strides[0] * aes
                emit(
                    f"wm_x{si}_{li} = {name}.iterator.toint() + cutlass.Int64(wm_row) * {row_stride}"
                )
                for am in range(atoms_m):
                    for half in range(2):
                        off = (am * 16 + 8 * half) * row_stride
                        value = f"wm_v{si}_{li}_{am}_{half}"
                        _emit_scalar_load(
                            emit,
                            value,
                            f"wm_x{si}_{li} + cutlass.Int64({off})",
                            adt,
                            rt,
                        )
                        for an in range(atoms_n):
                            for j in (0, 1) if half == 0 else (2, 3):
                                aux_value[(si, li, am, an, j)] = value
            else:
                x_batch = aux.strides[0] * aes if aux.fake.ndim == 3 else 0
                x_row = aux.strides[-2] * aes
                emit(
                    f"wm_x{si}_{li} = {name}.iterator.toint() + cutlass.Int64(wm_pb) * {x_batch} "
                    f"+ cutlass.Int64(wm_row) * {x_row} + cutlass.Int64(wm_col) * {aes}"
                )
                for am in range(atoms_m):
                    for an in range(atoms_n):
                        for half in range(2):
                            off = (am * 16 + 8 * half) * x_row + an * 8 * aes
                            lo = f"wm_v{si}_{li}_{am}_{an}_{half}_0"
                            hi = f"wm_v{si}_{li}_{am}_{an}_{half}_1"
                            _emit_pair_load(
                                emit,
                                lo,
                                hi,
                                f"wm_x{si}_{li} + cutlass.Int64({off})",
                                adt,
                                rt,
                                paired=aux_pairs[si][li],
                            )
                            aux_value[(si, li, am, an, 2 * half)] = lo
                            aux_value[(si, li, am, an, 2 * half + 1)] = hi

    # ---- cp.async issue: one commit group per K chunk, all before the first wait.
    def issue(*, tile: str, packets: int, cpr: int, row_bytes: int, gsrc: str) -> None:
        reps = max(1, packets // threads)
        for rep in range(reps):
            if packets < threads:
                emit(f"if wm_tidx < {packets}:")
                indent, idx = "    ", "wm_tidx"
            else:
                indent, idx = "", (f"({rep * threads} + wm_tidx)" if rep else "wm_tidx")
            emit(f"{indent}wm_prow = {idx} // {cpr}")
            emit(f"{indent}wm_pch = {idx} % {cpr}")
            swz = _swizzled_chunk("wm_prow", "wm_pch", cpr)
            emit(
                f"{indent}{rt}.cp_async_16({tile} + wm_prow * {row_bytes} + {swz} * 16, {gsrc})"
            )

    for i in range(nk):
        issue(
            tile=f"(wm_s_a + {i * a_chunk})",
            packets=a_packets,
            cpr=cpr_a,
            row_bytes=ra,
            gsrc=f"wm_a_g + cutlass.Int64(wm_prow) * {a_row} + cutlass.Int64({i * bk * es} + wm_pch * 16)",
        )
        if match.rhs_k_major:
            gsrc = f"wm_b_g + cutlass.Int64(wm_prow) * {b_row} + cutlass.Int64({i * bk * es} + wm_pch * 16)"
        else:
            gsrc = f"wm_b_g + cutlass.Int64({i * bk} + wm_prow) * {b_row} + cutlass.Int64(wm_pch * 16)"
        issue(
            tile=f"(wm_s_b + {i * b_chunk})",
            packets=b_packets,
            cpr=cpr_b,
            row_bytes=rb,
            gsrc=gsrc,
        )
        emit("cute.arch.cp_async_commit_group()")
    # ---- accumulators
    for am in range(atoms_m):
        for an in range(atoms_n):
            emit(
                "; ".join(
                    f"wm_acc_{am}_{an}_{j} = cutlass.Float32(0.0)" for j in range(4)
                )
            )
    # ---- per-lane ldmatrix row / chunk terms.
    # A x4: lanes 0-7 rows 0-7 chunk 2s, 8-15 rows 8-15 chunk 2s, 16-23 rows 0-7
    # chunk 2s+1, 24-31 rows 8-15 chunk 2s+1 -> (a0, a1, a2, a3).
    emit(f"wm_arow = wm_wmi * {wtm} + wm_l8 + 8 * (wm_lq % 2)")
    emit("wm_achunk = wm_lq // 2")
    if match.rhs_k_major:
        # x4 over two n-tiles: lanes 0-7 (n-tile j, chunk 2s) b0; 8-15 (j, 2s+1)
        # b1; 16-23 (j+1, 2s) b0; 24-31 (j+1, 2s+1) b1.  x2 uses lanes 0-15.
        emit(f"wm_brow = wm_wni * {wtn} + wm_l8 + 8 * (wm_lq // 2)")
        emit("wm_bchunk = wm_lq % 2")
    else:
        # x4.trans over two n-tiles: lanes 0-7 (k rows 0-7, n-tile j) b0; 8-15
        # (k rows 8-15, j) b1; 16-23 (k 0-7, j+1) b0; 24-31 (k 8-15, j+1) b1.
        emit("wm_brow = wm_l8 + 8 * (wm_lq % 2)")
        emit(f"wm_bchunk = (wm_wni * {wtn}) // 8 + wm_lq // 2")
    sync = "cute.arch.barrier()" if warps > 1 else "cute.arch.sync_warp()"
    for i in range(nk):
        emit(f"cute.arch.cp_async_wait_group({nk - 1 - i})")
        emit(sync)
        for st in range(steps):
            for am in range(atoms_m):
                row = f"(wm_arow + {am * 16})"
                chunk = f"({2 * st} + wm_achunk)"
                swz = _swizzled_chunk(row, chunk, cpr_a)
                emit(
                    f"wm_a_{am}_0, wm_a_{am}_1, wm_a_{am}_2, wm_a_{am}_3 = {rt}.ldmatrix_x4("
                    f"wm_s_a + {i * a_chunk} + {row} * {ra} + {swz} * 16)"
                )
            an = 0
            while an < atoms_n:
                pair = an + 1 < atoms_n
                if match.rhs_k_major:
                    row = (
                        f"(wm_brow + {an * 8})"
                        if pair
                        else f"(wm_wni * {wtn} + {an * 8} + wm_l8)"
                    )
                    chunk = (
                        f"({2 * st} + wm_bchunk)" if pair else f"({2 * st} + wm_lq % 2)"
                    )
                    op = "ldmatrix_x4" if pair else "ldmatrix_x2"
                else:
                    row = f"({16 * st} + wm_brow)"
                    chunk = (
                        f"(wm_bchunk + {an})"
                        if pair
                        else f"((wm_wni * {wtn} + {an * 8}) * {es} // 16)"
                    )
                    op = "ldmatrix_x4_trans" if pair else "ldmatrix_x2_trans"
                swz = _swizzled_chunk(row, chunk, cpr_b)
                address = f"wm_s_b + {i * b_chunk} + {row} * {rb} + {swz} * 16"
                if pair:
                    emit(
                        f"wm_b_{an}_0, wm_b_{an}_1, wm_b_{an + 1}_0, wm_b_{an + 1}_1 = {rt}.{op}({address})"
                    )
                else:
                    emit(f"wm_b_{an}_0, wm_b_{an}_1 = {rt}.{op}({address})")
                an += 2 if pair else 1
            for am in range(atoms_m):
                for an in range(atoms_n):
                    acc = ", ".join(f"wm_acc_{am}_{an}_{j}" for j in range(4))
                    emit(
                        f"{acc} = {mma}(wm_a_{am}_0, wm_a_{am}_1, wm_a_{am}_2, wm_a_{am}_3, "
                        f"wm_b_{an}_0, wm_b_{an}_1, {acc})"
                    )
    # ---- epilogue: the chain of every store on the fp32 fragments, then pairs.
    for si, store in enumerate(match.stores):
        odt = store.fake.dtype
        oes = odt.itemsize
        o_batch = store.strides[0] * oes if store.fake.ndim == 3 else 0
        o_row = store.strides[-2] * oes
        emit(
            f"wm_o{si} = {out_names[si]}.iterator.toint() + cutlass.Int64(wm_pb) * {o_batch} "
            f"+ cutlass.Int64(wm_row) * {o_row} + cutlass.Int64(wm_col) * {oes}"
        )
        for am in range(atoms_m):
            for an in range(atoms_n):
                values: list[str] = []
                for j in range(4):
                    carrier = f"wm_acc_{am}_{an}_{j}"
                    if store.chain.steps:
                        aux_locals = {
                            aux.leaf: aux_value[(si, li, am, an, j)]
                            for li, aux in enumerate(store.aux)
                        }
                        prelude, value = store.chain.render_prelude_and_expr(
                            carrier, df.new_var, "", aux_locals or None
                        )
                        for line in _scalar_epilogue(prelude).split("\n"):
                            if line.strip():
                                emit(line)
                        values.append(value)
                    else:
                        values.append(carrier)
                for half in range(2):
                    off = (am * 16 + 8 * half) * o_row + an * 8 * oes
                    lo, hi = values[2 * half], values[2 * half + 1]
                    address = f"wm_o{si} + cutlass.Int64({off})"
                    if not out_pairs[si]:
                        scalar = {
                            torch.float32: "stg_f32",
                            torch.float16: "stg_f16",
                            torch.bfloat16: "stg_bf16",
                        }[odt]
                        emit(f"{rt}.{scalar}({address}, {lo})")
                        emit(f"{rt}.{scalar}({address} + {oes}, {hi})")
                    elif odt == torch.float32:
                        emit(f"{rt}.stg_v2_f32({address}, {lo}, {hi})")
                    else:
                        pack = "pack_f16x2" if odt == torch.float16 else "pack_bf16x2"
                        emit(f"{rt}.stg_b32({address}, {rt}.{pack}({lo}, {hi}))")
    source = "\n".join(lines) + "\n"
    return list(ast.parse(textwrap.dedent(source)).body)


def _emit_pair_load(
    emit: Callable[[str], object],
    lo: str,
    hi: str,
    address: str,
    dtype: torch.dtype,
    rt: str,
    *,
    paired: bool = True,
) -> None:
    """Two adjacent elements at ``address`` as fp32 scalars (one pair load
    when the base is pair-aligned, two element loads otherwise)."""
    if not paired:
        _emit_scalar_load(emit, lo, address, dtype, rt)
        _emit_scalar_load(emit, hi, f"{address} + {dtype.itemsize}", dtype, rt)
    elif dtype == torch.float32:
        emit(f"{lo}, {hi} = {rt}.ldg_v2_f32({address})")
    else:
        unpack = "unpack_f16x2" if dtype == torch.float16 else "unpack_bf16x2"
        emit(f"{lo}, {hi} = {rt}.{unpack}({rt}.ldg_b32({address}))")


def _emit_scalar_load(
    emit: Callable[[str], object], value: str, address: str, dtype: torch.dtype, rt: str
) -> None:
    """One element at ``address`` as an fp32 scalar."""
    if dtype == torch.float32:
        emit(f"{value} = {rt}.ldg_f32({address})")
    elif dtype == torch.float16:
        emit(f"{value} = {rt}.ldg_f16_as_f32({address})")
    else:
        emit(f"{value} = {rt}.ldg_bf16_as_f32({address})")
