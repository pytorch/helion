from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Sequence
from typing import cast

if TYPE_CHECKING:
    from ...autotuner.config_spec import BlockSizeSpec
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment

HardwareTarget = tuple[str, str | None]

# Reduction op name tokens used to detect a reduction in a traced graph. Shared
# by the Triton split/join heuristic and the LLM workload-trait detection.
REDUCTION_TARGET_NAMES = frozenset({"amax", "sum", "softmax", "logsumexp"})

# Floor the attention query/M tile to >= 64 rows: tensor cores underutilize the
# MMA M-dimension below 64 (a hardware fact, not a shape preference), so a 32-row
# query tile in flash-attention runs ~5x slower than 64-512 at an otherwise
# identical config. This is the search-space floor applied by
# ``raise_attention_block_minimums``.
ATTENTION_MIN_QUERY_BLOCK = 64


def attention_query_block_id(env: CompileEnvironment) -> int | None:
    """Block id of the flash-attention query/M tile (Q@Kᵀ → softmax → P@V), else None.

    Structural — keyed on the matmul dataflow, not shape/dtype/name: the query tile
    is the M of two matmuls chained through a kv tile over an *untiled* head dim
    (the untiled-head-dim clause excludes an ordinary X@W1 → H@W2 MLP chain).
    """
    facts = env.config_spec.matmul_facts
    if len(facts) < 2:
        return None
    # A query tile is any block id used as the M of >= 2 matmuls.
    m_block_counts = Counter(f.m_block_id for f in facts if f.m_block_id is not None)
    for m_block_id, count in m_block_counts.items():
        if count < 2:
            continue
        chain = [f for f in facts if f.m_block_id == m_block_id]
        # scores GEMM has N=kv (untiled K); value GEMM has K=kv (untiled N).
        scores_kv = {f.n_block_id for f in chain if f.k_block_id is None}
        value_kv = {f.k_block_id for f in chain if f.n_block_id is None}
        if scores_kv & value_kv - {None}:
            return m_block_id
    return None


def op_name_parts(target: object) -> frozenset[str]:
    """Coarse name fragments for a traced FX call target.

    Collects the target's ``__name__``/``name``/``str()`` and their dot-split
    pieces, so callers can match a node against a set of op names.
    """
    parts: set[str] = set()
    for raw in (
        getattr(target, "__name__", None),
        getattr(target, "name", None),
        str(target),
    ):
        if not isinstance(raw, str):
            continue
        parts.add(raw)
        parts.update(piece for piece in raw.split(".") if piece)
    return frozenset(parts)


def is_canonical_row_reduction(env: CompileEnvironment) -> bool:
    """Whether the kernel is a canonical single-tile row reduction.

    A single non-reduction tile + a single reduction loop, with no matmul facts,
    and an M-axis that admits one row per program. Shared by the CuTe and Triton
    reduction heuristics so the structural gate cannot drift between backends.
    """
    spec = env.config_spec
    # Single non-reduction tile + single reduction dim.
    if len(spec.block_sizes) != 1 or len(spec.reduction_loops) != 1:
        return False
    # No matmul facts (this seeds reduction kernels, not GEMMs, and not a fused
    # matmul+reduction, which keeps its own tiling).
    if spec.matmul_facts:
        return False
    bs_spec = spec.block_sizes[0]
    # M-axis must accept block_size=1 (one row per program).
    return max(bs_spec.min_size, bs_spec.autotuner_min) <= 1


def dedupe_configs(configs: Iterable[Config]) -> list[Config]:
    result: list[Config] = []
    seen: set[Config] = set()
    for config in configs:
        if config in seen:
            continue
        seen.add(config)
        result.append(config)
    return result


def matches_hardware(
    env: CompileEnvironment,
    targets: tuple[HardwareTarget, ...],
) -> bool:
    from ..._hardware import get_hardware_info

    hardware = get_hardware_info(env.device)
    return (hardware.device_kind, hardware.compute_capability) in targets or (
        hardware.device_kind,
        None,
    ) in targets


def clamp_block_size_targets(
    env: CompileEnvironment,
    block_dims: Sequence[tuple[int, int, int]],
) -> list[int] | None:
    """Clamp block-size targets against the live ConfigSpec constraints.

    Each entry in *block_dims* is ``(block_id, static_dim, target)``.
    Returns the clamped block sizes, or ``None`` if any axis cannot
    satisfy its floor/ceiling constraints.
    """
    block_sizes: list[int] = []
    for block_id, static_dim, target in block_dims:
        try:
            spec = cast(
                "BlockSizeSpec",
                env.config_spec.block_sizes.block_id_lookup(block_id),
            )
        except KeyError:
            return None
        candidate = min(target, static_dim)
        if candidate < 1:
            return None
        candidate = 1 << (candidate.bit_length() - 1)
        floor = max(spec.min_size, spec.autotuner_min)
        if candidate < floor:
            return None
        candidate = min(candidate, spec.max_size)
        if candidate < floor:
            return None
        block_sizes.append(candidate)
    return block_sizes
