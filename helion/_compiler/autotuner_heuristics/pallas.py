from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from ...runtime.config import Config
from .common import clamp_block_size_targets
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


# Square bf16/fp16 cubes that get the no-tiling ``[N, N, N]`` seed (dims per the
# device-us ablation; ~17% on-device speedup on bf16 1024^3 via prefetch).
_PALLAS_NO_TILING_DIMS: frozenset[int] = frozenset({1024, 2048, 4096})

# f32 stays pinned to 1024: the ablation showed forced no-tiling regresses
# ~2-2.5% on 2048/4096 vs the autotuner's tiled picks.
_PALLAS_F32_NO_TILING_DIMS: frozenset[int] = frozenset({1024})


class _PallasMatmulFactDims(NamedTuple):
    """Static-shape view of a 2D matmul fact (all fields narrowed to ``int``)."""

    m: int
    k: int
    n: int
    m_block_id: int
    k_block_id: int
    n_block_id: int


_BF16_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
_F32_DTYPES: tuple[torch.dtype, ...] = (torch.float32,)


def _pallas_matmul_seed_dims_or_none(
    env: CompileEnvironment,
    *,
    allowed_dtypes: tuple[torch.dtype, ...] = _BF16_DTYPES,
) -> _PallasMatmulFactDims | None:
    """Return the single 2D matmul fact's static dims, or ``None``.

    Shared eligibility gate: exactly one matmul fact, 2D x 2D, same dtype in
    ``allowed_dtypes`` (default bf16/fp16; pass ``(float32,)`` for f32), all
    dims and block-ids known.
    """
    facts = env.config_spec.matmul_facts
    if len(facts) != 1:
        return None
    fact = facts[0]
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.lhs_dtype not in allowed_dtypes:
        return None
    if fact.rhs_dtype != fact.lhs_dtype:
        return None
    if (
        fact.static_m is None
        or fact.static_n is None
        or fact.static_k is None
        or fact.m_block_id is None
        or fact.n_block_id is None
        or fact.k_block_id is None
    ):
        return None
    return _PallasMatmulFactDims(
        m=fact.static_m,
        k=fact.static_k,
        n=fact.static_n,
        m_block_id=fact.m_block_id,
        k_block_id=fact.k_block_id,
        n_block_id=fact.n_block_id,
    )


def _pallas_matmul_supports_loop_type_and_pre_broadcast(
    env: CompileEnvironment,
) -> bool:
    spec = env.config_spec
    return spec.supports_config_key("pallas_loop_type") and spec.supports_config_key(
        "pallas_pre_broadcast"
    )


class _PallasNoTilingSeedHeuristic(AutotunerHeuristic):
    """Seed ``block_sizes == [N, N, N]`` for a square matmul whose dim is in
    ``_dims``, so the backend lowers via ``lax.dot_general`` instead of
    ``pl.pallas_call`` (see ``PallasBackend._detect_matmul_dot_general_lowering``)
    -- the former is XLA-visible, so ``cross_program_prefetch`` applies.  It falls
    back to ``pl.pallas_call`` when the autotuner picks a tiled config.  Subclasses
    set ``_allowed_dtypes`` and ``_dims``.
    """

    backend = "pallas"
    _allowed_dtypes: tuple[torch.dtype, ...]
    _dims: frozenset[int]

    @classmethod
    def _seed_block_sizes(cls, env: CompileEnvironment) -> list[int] | None:
        """The ``[N, N, N]`` no-tiling block sizes for an eligible matmul, else None."""
        dims = _pallas_matmul_seed_dims_or_none(env, allowed_dtypes=cls._allowed_dtypes)
        if dims is None or not (dims.m == dims.k == dims.n) or dims.m not in cls._dims:
            return None
        return clamp_block_size_targets(
            env,
            [
                (dims.m_block_id, dims.m, dims.m),
                (dims.k_block_id, dims.k, dims.k),
                (dims.n_block_id, dims.n, dims.n),
            ],
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._seed_block_sizes(
            env
        ) is not None and _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        block_sizes = cls._seed_block_sizes(env)
        if block_sizes is None:
            return None
        return Config(
            block_sizes=block_sizes,
            pallas_loop_type="unroll",
            pallas_pre_broadcast=True,
        )


class PallasMatmulNoTilingSeedHeuristic(_PallasNoTilingSeedHeuristic):
    """bf16/fp16 square cubes in ``_PALLAS_NO_TILING_DIMS``."""

    name = "pallas_matmul_no_tiling_seed"
    _allowed_dtypes = _BF16_DTYPES
    _dims = _PALLAS_NO_TILING_DIMS


class PallasMatmulF32NoTilingSeedHeuristic(_PallasNoTilingSeedHeuristic):
    """f32 square cubes in ``_PALLAS_F32_NO_TILING_DIMS`` (narrower than bf16 --
    the ablation showed forced no-tiling regresses on f32 2048/4096)."""

    name = "pallas_matmul_f32_no_tiling_seed"
    _allowed_dtypes = _F32_DTYPES
    _dims = _PALLAS_F32_NO_TILING_DIMS
