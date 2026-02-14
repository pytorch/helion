"""CuteDSL pipeline state management utilities.

Provides helpers for generating multi-stage software-pipelined loops
in CuteDSL kernels. Supports three pipeline modes:

1. **Single-stage** (num_stages=1): Simple synchronous loop.
2. **CP.async pipeline** (SM80, num_stages>1): Uses ``cp_async_commit_group``
   / ``cp_async_wait_group`` for async GMEM->SMEM copies.
3. **TMA pipeline** (SM90+, num_stages>1): Uses TMA + mbarrier for
   hardware-accelerated async copies.

Reference: ``flash_attn/cute/pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cutedsl_arch_utils import SM90

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a software-pipelined loop.

    Attributes:
        num_stages: Number of pipeline stages (1 = no pipelining).
        arch: SM version (80, 90, 100).
        use_tma: Whether to use TMA-based pipeline (SM90+).
        num_buffers: Number of SMEM buffers (usually == num_stages).
    """

    num_stages: int
    arch: int
    use_tma: bool = False
    num_buffers: int | None = None

    @property
    def effective_buffers(self) -> int:
        """Number of SMEM buffers (defaults to num_stages)."""
        return self.num_buffers if self.num_buffers is not None else self.num_stages

    @property
    def is_pipelined(self) -> bool:
        """Whether this pipeline uses multi-stage buffering."""
        return self.num_stages > 1

    @property
    def pipeline_mode(self) -> str:
        """Return the pipeline mode string."""
        if not self.is_pipelined:
            return "single"
        if self.use_tma and self.arch >= SM90:
            return "tma"
        return "cp_async"

    @staticmethod
    def from_config(
        num_stages: int,
        arch: int,
        *,
        prefer_tma: bool = True,
    ) -> PipelineConfig:
        """Create a PipelineConfig from Config parameters.

        Args:
            num_stages: Number of pipeline stages.
            arch: SM version.
            prefer_tma: Whether to prefer TMA over cp.async on SM90+.

        Returns:
            Configured PipelineConfig.
        """
        use_tma = prefer_tma and arch >= SM90 and num_stages > 1
        return PipelineConfig(
            num_stages=num_stages,
            arch=arch,
            use_tma=use_tma,
        )


def emit_pipeline_state_init(
    config: PipelineConfig,
    var_prefix: str = "",
) -> list[str]:
    """Emit code to initialize pipeline state variables.

    Args:
        config: Pipeline configuration.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        return lines

    lines.append(f"# Pipeline state (mode={config.pipeline_mode}, stages={config.num_stages})")
    lines.append(f"{p}pipe_stage = 0")

    if config.pipeline_mode == "tma":
        lines.append(f"{p}pipe_phase = 0")
        lines.append(f"# Allocate mbarriers for TMA pipeline")
        lines.append(
            f"{p}mbar = cute.arch.mbarrier_init("
            f"count={config.num_stages})"
        )
    else:
        lines.append(f"# cp.async pipeline with {config.num_stages} stages")

    return lines


def emit_pipeline_prologue(
    config: PipelineConfig,
    tile_k: int,
    k_extent: str,
    var_prefix: str = "",
) -> list[str]:
    """Emit pipeline prologue: pre-fill stages before the main loop.

    For pipelined loops, issues the first ``num_stages - 1`` async copies
    before the main loop starts, so the first iteration has data ready.

    Args:
        config: Pipeline configuration.
        tile_k: K tile dimension.
        k_extent: Expression for total K extent.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        return lines

    prefill_stages = config.num_stages - 1
    lines.append(f"# Pipeline prologue: pre-fill {prefill_stages} stage(s)")

    if config.pipeline_mode == "tma":
        lines.append(f"for {p}prefill_i in range({prefill_stages}):")
        lines.append(f"    {p}stage = {p}prefill_i % {config.num_stages}")
        lines.append(f"    # Issue TMA copy for sA[..., {p}stage] and sB[..., {p}stage]")
        lines.append(f"    cute.arch.mbarrier_arrive({p}mbar, {p}stage)")
    else:
        lines.append(f"for {p}prefill_i in range({prefill_stages}):")
        lines.append(f"    {p}stage = {p}prefill_i % {config.num_stages}")
        lines.append(f"    # Issue cp.async for sA[..., {p}stage] and sB[..., {p}stage]")
        lines.append(f"    cute.arch.cp_async_commit_group()")

    return lines


def emit_pipeline_wait(
    config: PipelineConfig,
    wait_count: int | None = None,
    var_prefix: str = "",
) -> list[str]:
    """Emit code to wait for async copies to complete.

    Args:
        config: Pipeline configuration.
        wait_count: Number of outstanding groups to wait for.
            Defaults to ``num_stages - 2`` for overlap.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        lines.append(f"{p}cute.arch.syncthreads()")
        return lines

    if wait_count is None:
        wait_count = max(0, config.num_stages - 2)

    if config.pipeline_mode == "tma":
        lines.append(f"{p}cute.arch.mbarrier_wait({p}mbar, {p}pipe_stage, {p}pipe_phase)")
    else:
        lines.append(f"{p}cute.arch.cp_async_wait_group({wait_count})")
        lines.append(f"{p}cute.arch.syncthreads()")

    return lines


def emit_pipeline_advance(
    config: PipelineConfig,
    var_prefix: str = "",
) -> list[str]:
    """Emit code to advance the pipeline stage.

    Args:
        config: Pipeline configuration.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        return lines

    lines.append(f"{p}pipe_stage = ({p}pipe_stage + 1) % {config.num_stages}")

    if config.pipeline_mode == "tma":
        lines.append(
            f"{p}pipe_phase = ({p}pipe_stage == 0).select("
            f"1 - {p}pipe_phase, {p}pipe_phase)"
        )

    return lines


def emit_pipeline_commit(
    config: PipelineConfig,
    var_prefix: str = "",
) -> list[str]:
    """Emit code to commit the current async copy group.

    Args:
        config: Pipeline configuration.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        return lines

    if config.pipeline_mode == "tma":
        lines.append(f"{p}cute.arch.mbarrier_arrive({p}mbar, {p}pipe_stage)")
    else:
        lines.append(f"{p}cute.arch.cp_async_commit_group()")

    return lines


def emit_pipeline_drain(
    config: PipelineConfig,
    var_prefix: str = "",
) -> list[str]:
    """Emit code to drain the pipeline after the main loop.

    Waits for all outstanding async copies to complete.

    Args:
        config: Pipeline configuration.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines.
    """
    p = var_prefix
    lines: list[str] = []

    if not config.is_pipelined:
        return lines

    lines.append(f"# Pipeline drain: wait for all outstanding copies")
    if config.pipeline_mode == "tma":
        lines.append(f"{p}cute.arch.mbarrier_wait({p}mbar, {p}pipe_stage, {p}pipe_phase)")
    else:
        lines.append(f"{p}cute.arch.cp_async_wait_group(0)")
        lines.append(f"{p}cute.arch.syncthreads()")

    return lines


def emit_pipelined_mainloop(
    config: PipelineConfig,
    tile_k: int,
    k_extent: str,
    var_prefix: str = "",
) -> list[str]:
    """Emit a complete pipelined K-loop skeleton.

    Generates the full main loop structure with:
    - Pipeline prologue (pre-filling)
    - Main loop with wait/compute/commit/advance
    - Pipeline drain

    The actual copy and MMA code must be inserted at the marked positions.

    Args:
        config: Pipeline configuration.
        tile_k: K tile dimension.
        k_extent: Expression for total K extent.
        var_prefix: Optional variable name prefix.

    Returns:
        List of code lines with insertion markers.
    """
    p = var_prefix
    lines: list[str] = []

    lines.append(f"{p}num_k_tiles = ({k_extent} + {tile_k} - 1) // {tile_k}")
    lines.append("")

    if not config.is_pipelined:
        # Single-stage: simple synchronous loop
        lines.append(f"for {p}k_tile in range({p}num_k_tiles):")
        lines.append(f"    # [INSERT: GMEM -> SMEM copy for A and B]")
        lines.append(f"    cute.arch.syncthreads()")
        lines.append(f"    # [INSERT: SMEM -> REG copy and MMA]")
        lines.append(f"    cute.arch.syncthreads()")
        return lines

    # Pipeline state init
    lines.extend(emit_pipeline_state_init(config, var_prefix))
    lines.append("")

    # Prologue
    lines.extend(emit_pipeline_prologue(config, tile_k, k_extent, var_prefix))
    lines.append("")

    # Main loop
    lines.append(f"for {p}k_tile in range({p}num_k_tiles):")
    lines.append(f"    {p}compute_stage = {p}k_tile % {config.num_stages}")

    # Wait for current compute stage data
    for line in emit_pipeline_wait(config, var_prefix=f"    {p}"):
        lines.append(line)

    lines.append(f"    # [INSERT: SMEM -> REG copy and MMA for stage {p}compute_stage]")
    lines.append("")

    # Issue async copy for next tile
    lines.append(f"    {p}next_tile = {p}k_tile + {config.num_stages - 1}")
    lines.append(f"    if {p}next_tile < {p}num_k_tiles:")
    lines.append(f"        # [INSERT: GMEM -> SMEM copy for next tile]")

    for line in emit_pipeline_commit(config, var_prefix=f"        {p}"):
        lines.append(line)

    # Advance stage
    for line in emit_pipeline_advance(config, var_prefix=f"    {p}"):
        lines.append(line)

    lines.append("")

    # Drain
    lines.extend(emit_pipeline_drain(config, var_prefix))

    return lines
