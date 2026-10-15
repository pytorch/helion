"""Pinned handwritten CuTe baseline for the K2 FP8 scaled GEMM campaign.

The Quack dependency is loaded only when this provider is requested.  Each
prepared call owns an explicit configuration and reads the original device
scale storage on every launch.  Preparation never autotunes or reads scales
back to the host.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import cache
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_QUACK_ROOT = (
    Path(
        os.environ.get(
            "K2_QUACK_ROOT", str(Path(__file__).resolve().parents[2] / "quack")
        )
    )
    .expanduser()
    .resolve()
)
QUACK_COMMIT = "c8ec3170057987da0ec99883736f381ea1937cf3"


@dataclass(frozen=True)
class PreparedBaseline:
    call: Callable[[], torch.Tensor]
    config: dict[str, Any]
    provenance: dict[str, Any]
    out: torch.Tensor


def _scaled_epilogue(acc: Any, sa: Any, sb: Any) -> dict[str, Any]:  # noqa: ANN401 - Quack traces dynamically supplied symbolic operands
    # Keep the multiplication order used by the library reference.  Forming
    # sa * sb first can overflow despite the final result being representable.
    return {"D": (acc * sa) * sb}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@cache
def _quack_dependencies(root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if commit != QUACK_COMMIT:
        raise ValueError(f"Quack baseline requires {QUACK_COMMIT}, got {commit}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"Quack baseline checkout has tracked changes: {dirty}")
    sys.path.insert(0, str(root))
    quack = importlib.import_module("quack")
    quack_file = quack.__file__
    assert quack_file is not None
    loaded_root = Path(quack_file).resolve().parent.parent
    if loaded_root != root:
        raise ValueError(
            f"Quack was already imported from {loaded_root}, expected {root}"
        )
    frontend = importlib.import_module("quack.epilogue.frontend")
    resources = importlib.import_module("quack.epilogue.ops")
    config_module = importlib.import_module("quack.gemm_config")
    files = [
        "quack/gemm_sm100.py",
        "quack/gemm_config.py",
        "quack/gemm_runtime/host.py",
        "quack/epilogue/frontend.py",
        "quack/epilogue/ops.py",
        "quack/epilogue/visit.py",
        "quack/epilogue/math.py",
    ]
    provenance = {
        "provider": "quack-cute-fp8-scaled-mm",
        "root": str(root),
        "commit": commit,
        "tracked_changes": dirty,
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch_version": torch.__version__,
        "sources": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "semantics": "BF16((FP32(A @ B) * scale_a) * scale_b)",
        "scale_values": (
            "Live FP32 device tensors; views only; "
            "no host reads or precomputed products"
        ),
        "preparation": (
            "Explicit GemmConfig; EpiMod.plan compiles without launching or autotuning"
        ),
    }
    return frontend, resources, config_module, provenance


@cache
def _epilogue(root: Path, token_scale: bool, channel_scale: bool) -> Any:  # noqa: ANN401 - external Quack frontend returns a dynamically generated module
    frontend, resources, config_module, provenance = _quack_dependencies(root)
    sa_op = resources.ColVecLoad("sa") if token_scale else resources.Scalar("sa")
    sb_op = resources.RowVecLoad("sb") if channel_scale else resources.Scalar("sb")
    return frontend.gemm_epilogue(ops={"sa": sa_op, "sb": sb_op})(_scaled_epilogue)


def quack_config_candidates(m: int, k: int, n: int) -> list[dict[str, Any]]:
    """A finite baseline sweep, including all relevant vLLM tile families.

    Quack's tile_m denotes the two-CTA MMA tile, so a (128,128,128)
    configuration with cluster_m=2 has a per-CTA M tile of 64.
    Every candidate is returned, including any rejected later by Quack;
    callers must record preparation/execution failures instead of dropping them.
    """
    tile_clusters = [
        (128, 128, 128, 2, 1, False),  # vLLM 64 < M <= 256
        (256, 128, 128, 2, 2, False),  # vLLM M > 256
        (256, 256, 128, 2, 1, False),  # Quack SM100 default tile
        (128, 128, 128, 1, 1, False),
        (128, 64, 128, 1, 1, False),
        (128, 128, 128, 2, 2, False),
        (256, 128, 128, 2, 1, False),
    ]
    if m <= 256:
        tile_clusters.extend(
            [
                (128, 64, 256, 4, 1, True),  # vLLM M64,K>=4096
                (128, 64, 256, 2, 1, True),
                (128, 128, 128, 2, 1, True),
            ]
        )
    return [
        {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "cluster_m": cluster_m,
            "cluster_n": cluster_n,
            "swap_ab": swap_ab,
            "pingpong": False,
            "is_dynamic_persistent": dynamic,
            "device_capacity": 10,
        }
        for tile_m, tile_n, tile_k, cluster_m, cluster_n, swap_ab in tile_clusters
        for dynamic in (False, True)
    ]


def prepare_quack_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    config: dict[str, Any],
    *,
    out: torch.Tensor | None = None,
    quack_root: Path = DEFAULT_QUACK_ROOT,
) -> PreparedBaseline:
    """Compile a pinned Quack configuration; B uses the logical (K,N) layout.

    Unsupported configurations raise their original exception.  The campaign
    should persist that exception alongside the attempted configuration.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected A[M,K] and B[K,N]")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("The K2 baseline requires FP8 E4M3FN operands")
    m, k = a.shape
    n = b.shape[1]
    if scale_a.numel() not in (1, m) or scale_b.numel() not in (1, n):
        raise ValueError("Scales must contain 1 or M / 1 or N elements")
    if any(
        scale.dtype != torch.float32 or scale.device != a.device
        for scale in (scale_a, scale_b)
    ):
        raise ValueError("Scales must be live FP32 tensors on the operand device")
    if not scale_a.is_contiguous() or not scale_b.is_contiguous():
        raise ValueError("Scale vectors must be contiguous so normalization is a view")
    token_scale = scale_a.numel() != 1
    channel_scale = scale_b.numel() != 1
    sa = scale_a.view(1, m) if token_scale else scale_a.view(())
    sb = scale_b.view(1, n) if channel_scale else scale_b.view(())
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16 or out.device != a.device:
        raise ValueError("Output must be a BF16 [M,N] tensor on the operand device")
    root = quack_root.resolve()
    frontend, resources, config_module, source_provenance = _quack_dependencies(root)
    chosen = config_module.GemmConfig(**config)
    mod = _epilogue(root, token_scale, channel_scale)
    output = {"D": out}
    plan = mod.plan(a, b, out=output, config=chosen, sa=sa, sb=sb)

    def call() -> torch.Tensor:
        plan.run(a, b, out=output, sa=sa, sb=sb)
        return out

    provenance = dict(source_provenance)
    provenance.update(
        {
            "shape_mkn": [m, k, n],
            "scale_mode": (
                f"{'token' if token_scale else 'tensor'}-"
                f"{'channel' if channel_scale else 'tensor'}"
            ),
            "epilogue_semantic_key": mod.semantic_key,
            "configuration": asdict(chosen),
        }
    )
    return PreparedBaseline(call, asdict(chosen), provenance, out)
