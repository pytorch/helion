"""Compiler-owned seeds for the exact fp32 SIMT GEMM (``cute_simt_mma``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ...runtime.config import Config
from ..cute.matmul_utils import cute_matmul_root_placements
from ..cute.mma_support import cute_fp32_dot_uses_tf32
from ..cute.simt_gemm import MAX_REGISTER_BLOCK
from ..cute.simt_gemm import SIMT_MMA_KEY
from ..cute.simt_gemm import SIMT_MMA_LAYOUT_KEY
from ..cute.simt_gemm import SIMT_MMA_LAYOUTS
from .common import dedupe_configs
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import ConfigValueExpression
    from ..device_ir import DeviceIR

# (bm, bn, threads_m, threads_n): the cuBLAS SIMT SGEMM shapes first (a
# 128x128 tile over 16x16 threads is an 8x8 register block per thread), then
# smaller tiles for short M/N extents or more CTAs per problem.
_TILE_LADDER: tuple[tuple[int, int, int, int], ...] = (
    (128, 128, 16, 16),
    (64, 64, 16, 16),
    (128, 64, 16, 16),
    (64, 128, 16, 16),
    # Smaller register blocks (8x4): half the accumulators, 2 CTAs per SM.
    (128, 128, 16, 32),
    (64, 64, 8, 16),
    # 16x4 blocks over 128 threads: the s2 cold-autotune winner of
    # split_k_barrier (128x64 tile, 8x16 threads).
    (128, 64, 8, 16),
    (64, 128, 16, 8),
    (64, 64, 8, 8),
    (32, 32, 8, 8),
    (32, 32, 16, 16),
)
_K_STEPS: tuple[int, ...] = (8, 16)
# CTAs per SM the grid-scaling tunable (a split-K factor) is sized for: one
# wave with a little slack, and a second seed with enough CTAs for two
# resident CTAs per SM.
_WAVE_TARGETS: tuple[float, ...] = (0.75, 1.5)


def _grid_scaling_tunable(
    env: CompileEnvironment, root: tuple[int, ...], m: int, n: int
) -> tuple[str, int, ConfigValueExpression] | None:
    """A user tunable that sets the extent of the GEMM root's extra axis.

    A split-K kernel tiles K with ``block_size=[..., block_k]`` where
    ``block_k`` is a pure function of ``hl.register_tunable(...)`` (recorded in
    ``config_value_expressions``); the axis only bounds the inner K loop, and
    the number of CTAs along it is ``ceil(extent / block_k(tunable))``.
    Returns ``(tunable name, axis extent, block expression)``.
    """
    from ..compile_environment import ConfigValueExpression
    from ..compile_environment import FixedBlockSizeSource

    extra = [block_id for block_id in root if block_id not in (m, n)]
    if len(extra) != 1:
        return None
    info = env.block_sizes[extra[0]]
    extent = info.size
    source = info.block_size_source
    if not isinstance(extent, int) or not isinstance(source, FixedBlockSizeSource):
        return None
    value = source.value
    if not isinstance(value, torch.SymInt):
        return None
    symbol = value._sympy_()
    if not isinstance(symbol, sympy.Expr):
        return None
    expression = env.config_value_expressions.get(symbol)
    if not isinstance(expression, ConfigValueExpression):
        return None
    names: set[str] = set()

    def collect(item: int | str | ConfigValueExpression) -> None:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, ConfigValueExpression):
            for argument in item.arguments:
                collect(argument)

    collect(expression)
    if len(names) != 1:
        return None
    (name,) = names
    if name not in env.config_spec.user_defined_tunables:
        return None
    return name, extent, expression


def _grid_scaling_values(
    env: CompileEnvironment,
    tunable: tuple[str, int, ConfigValueExpression],
    tiles: int,
) -> list[int]:
    """Tunable values whose CTA count first reaches each wave target."""
    name, extent, expression = tunable
    fragment = env.config_spec.user_defined_tunables[name]
    low = getattr(fragment, "low", None)
    high = getattr(fragment, "high", None)
    if type(low) is not int or type(high) is not int or low < 1:
        return []
    candidates = []
    value = 1
    while value <= high:
        if value >= low:
            candidates.append(value)
        value *= 2
    if not candidates:
        return []
    num_sm = max(env.config_spec.num_sm, 1)
    chosen = []
    for target in _WAVE_TARGETS:
        for candidate in candidates:
            block = expression.evaluate(Config.from_dict({name: candidate}))
            if block <= 0:
                continue
            ctas = tiles * -(-extent // block)
            if ctas >= target * num_sm:
                chosen.append(candidate)
                break
    return chosen


class CuteSimtMmaHeuristic(AutotunerHeuristic):
    """Seed the staged, register-blocked SIMT GEMM for exact fp32 matmuls.

    ``dot_precision="ieee"`` keeps fp32 contractions off the tensor cores, so
    without these seeds the search only ever sees the per-element scalar dot.
    A kernel whose phases are separated by ``hl.barrier()`` is seeded as the
    ordered per-phase launch (``pid_type="flat"``): the single persistent
    launch shares one thread block across phases and cannot host the GEMM's
    thread layout.
    """

    name = SIMT_MMA_KEY
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        spec = env.config_spec
        if (
            spec.cute_tcgen05_search_enabled
            or spec.cute_flash_search_enabled
            or not spec.cute_simt_mma_enabled
            or cute_fp32_dot_uses_tf32()
        ):
            return []
        placements = cute_matmul_root_placements(env, device_ir)
        if len(placements) != 1:
            return []
        fact, root = placements[0]
        m, n, k = fact.m_block_id, fact.n_block_id, fact.k_block_id
        if (
            m is None
            or n is None
            or k is None
            or len({m, n, k}) != 3
            or fact.lhs_ndim != 2
            or fact.rhs_ndim != 2
            or fact.lhs_dtype != torch.float32
            or fact.rhs_dtype != torch.float32
            or m not in root
            or n not in root
            or k in root
        ):
            return []
        blocks = {
            item.block_id: item for item in spec.block_sizes if len(item.block_ids) == 1
        }
        if any(block_id not in blocks for block_id in (m, n, k)):
            return []
        fragments = {item.block_id: item._fragment(spec) for item in spec.block_sizes}
        # Every other tile (another phase's output tile, say) gets a modest
        # fixed size; the search owns it from there.
        other_values = {
            block_id: max(fragment.low, min(fragment.high, 16))
            for block_id, fragment in fragments.items()
            if block_id not in (m, n, k)
        }
        other_threads = {item.block_id: 0 for item in spec.num_threads}
        # A reduction over a tunable-sized extent (the split-K workspace's
        # last dim) runs as one pass over the tunable's default; its generic
        # size hint would otherwise default to a 4096-wide masked loop.
        reduction_loops: dict[int, object] = {}
        for item in spec.reduction_loops:
            size = env.block_sizes[item.block_id].size
            if not isinstance(size, torch.SymInt):
                continue
            hint = env.tunable_default_hint(size._sympy_())
            fragment = item._flat_fragment(spec)
            if hint is not None and fragment.low <= hint <= fragment.high:
                reduction_loops[item.block_id] = hint
        lane_axes = {m, n, k}
        # A complete seed: a kernel's own tunables (a split-K factor) keep
        # their declared defaults, so the seed also compiles as-is.
        base: dict[str, object] = {
            name: fragment.default()
            for name, fragment in spec.user_defined_tunables.items()
        }
        scaling = _grid_scaling_tunable(env, root, m, n)
        base[SIMT_MMA_KEY] = True
        if env.has_barrier and "flat" in spec.allowed_pid_types:
            base["pid_type"] = "flat"
        result = []
        for bm, bn, threads_m, threads_n in _TILE_LADDER:
            bm = min(bm, fragments[m].high)
            bn = min(bn, fragments[n].high)
            threads_m = min(threads_m, bm)
            threads_n = min(threads_n, bn)
            if (
                bm < fragments[m].low
                or bn < fragments[n].low
                or bm % threads_m
                or bn % threads_n
                or (bm // threads_m) * (bn // threads_n) > MAX_REGISTER_BLOCK
            ):
                continue
            for bk in _K_STEPS:
                if not fragments[k].low <= bk <= fragments[k].high:
                    continue
                values = {**other_values, m: bm, n: bn, k: bk}
                threads = {**other_threads, m: threads_m, n: threads_n, k: 0}
                # Both register-block layouts: 128-bit blocked reads win for
                # 16x16-thread tiles, strided scalar reads for 8x16 (bank pattern).
                layouts = (
                    SIMT_MMA_LAYOUTS
                    if threads_m == threads_n
                    else SIMT_MMA_LAYOUTS[::-1]
                )
                # The grid-scaling tunable (split_k) sized so this tile's
                # CTAs fill the SMs, beside its declared default.
                scaled: list[dict[str, object]] = [{}]
                if scaling is not None and fact.static_m and fact.static_n:
                    tiles = -(-fact.static_m // bm) * -(-fact.static_n // bn)
                    scaled.extend(
                        {scaling[0]: value}
                        for value in _grid_scaling_values(env, scaling, tiles)
                    )
                result.extend(
                    Config.from_dict(
                        {
                            **base,
                            **override,
                            SIMT_MMA_LAYOUT_KEY: layout,
                            "block_sizes": _seq_config_list(spec.block_sizes, values),
                            "num_threads": _seq_config_list(spec.num_threads, threads),
                            "cute_vector_widths": _seq_config_list(
                                spec.cute_vector_widths, dict.fromkeys(lane_axes, 1)
                            ),
                            "cute_lane_layouts": _seq_config_list(
                                spec.cute_lane_layouts,
                                dict.fromkeys(lane_axes, "blocked"),
                            ),
                            **(
                                {
                                    "reduction_loops": _seq_config_list(
                                        spec.reduction_loops, reduction_loops
                                    )
                                }
                                if reduction_loops
                                else {}
                            ),
                        }
                    )
                    for layout in layouts
                    for override in scaled
                )
        return dedupe_configs(result)
