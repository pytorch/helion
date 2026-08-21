from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import cast
from urllib.parse import unquote
from urllib.parse import urlparse

if TYPE_CHECKING:
    from types import ModuleType

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation


QUACK_B_LAYOUTS = ("k_major", "n_major")
QUACK_REPOSITORY = "https://github.com/Dao-AILab/quack"
QUACK_VERSION = "0.6.4"
QUACK_COMMIT = "60d88082272a256fa9b3b2ab631c82cfa78337c6"


def _editable_root(distribution: importlib.metadata.Distribution) -> Path:
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("QuACK must be installed from an editable checkout")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("QuACK direct_url.json is invalid") from error
    url = direct_url.get("url") if isinstance(direct_url, dict) else None
    parsed = urlparse(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
    ):
        raise RuntimeError("QuACK editable checkout path is unavailable")
    return Path(unquote(parsed.path)).resolve(strict=True)


def verify_quack_installation() -> dict[str, str]:
    """Verify the pinned clean editable QuACK distribution and module origin."""

    try:
        distribution = importlib.metadata.distribution("quack-kernels")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("QuACK requires the quack-kernels distribution") from error
    if distribution.version != QUACK_VERSION:
        raise RuntimeError(
            f"QuACK distribution is {distribution.version!r}, expected {QUACK_VERSION!r}"
        )
    root = _editable_root(distribution)
    from benchmarks.cute import grouped_gemm_benchmark as common

    checkout = common.clean_checkout(root, QUACK_COMMIT, "QuACK")
    module = _import_quack_from_root(root)
    origin = Path(str(module.__file__)).resolve(strict=True)
    if not origin.is_relative_to(root):
        raise RuntimeError("QuACK imported outside its editable checkout")

    return {
        "repository": QUACK_REPOSITORY,
        "upstream_commit": str(checkout["commit"]),
        "distribution_version": distribution.version,
        "installation": "editable",
        "module": origin.relative_to(root).as_posix(),
    }


def _import_quack_from_root(root: Path) -> ModuleType:
    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        return importlib.import_module("quack")
    finally:
        sys.path.remove(root_text)


def _package_identity() -> dict[str, str]:
    identity = verify_quack_installation()
    import quack  # pyrefly: ignore [missing-import]

    module_version = str(quack.__version__)
    if module_version != QUACK_VERSION:
        raise RuntimeError(
            f"QuACK module is {module_version!r}, expected {QUACK_VERSION!r}"
        )
    return {
        **identity,
        "module_version": module_version,
    }


def prepare_quack_default(
    inputs: GroupedGemmInputs,
    *,
    b_layout: str,
) -> PreparedImplementation:
    """Use QuACK's public GEMM with its untuned per-architecture default."""

    from benchmarks.cute import grouped_gemm_benchmark as common
    import torch

    if b_layout not in QUACK_B_LAYOUTS:
        raise ValueError(f"unsupported QuACK B layout {b_layout!r}")
    package = _package_identity()
    try:
        from quack.gemm_interface import (  # pyrefly: ignore [missing-import]
            default_config,
        )
        from quack.gemm_interface import gemm  # pyrefly: ignore [missing-import]
    except ImportError as error:
        raise RuntimeError(
            "QuACK requires the optional quack-kernels package"
        ) from error

    b_kn = inputs.b_for_layout(b_layout).transpose(1, 2)
    output = torch.empty(
        (inputs.case.total_m, inputs.case.n),
        device=inputs.compact_a.device,
        dtype=torch.bfloat16,
    )
    resolved_config = default_config(inputs.compact_a.device)
    if (
        resolved_config.device_capacity != 10
        or resolved_config.swap_ab
        or resolved_config.use_tma_gather
    ):
        raise RuntimeError("QuACK's public SM100 default config changed")
    serialized = asdict(resolved_config)
    resolved_dynamic_scheduler = bool(resolved_config.is_dynamic_persistent)

    def call() -> torch.Tensor:
        # The public wrapper has no ``config`` keyword. ``tuned=False`` routes
        # internally to ``gemm_tuned.fn(..., config=None)``.
        result = gemm(
            inputs.compact_a,
            b_kn,
            out=output,
            cu_seqlens_m=inputs.offsets,
            dynamic_scheduler=False,
            tuned=False,
            split_k=1,
        )
        if result is not output:
            raise RuntimeError("QuACK public GEMM replaced the provided output")
        return output

    return common.PreparedImplementation(
        name=f"quack-public-default-{b_layout}",
        call=call,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: inputs.compact_output_slices(
            cast("torch.Tensor", result)
        ),
        config={
            "provider": "quack",
            "selection_mode": "public_default_no_tune",
            "b_layout": b_layout,
            "requested_config": None,
            "config": {**serialized, "b_layout": b_layout},
            "requested_dynamic_scheduler": False,
            "resolved_dynamic_scheduler": resolved_dynamic_scheduler,
            "tuned": False,
            "split_k": 1,
            "package": package,
            "a_layout": common.compact_contiguous_a_layout(),
            "preprocessing_timed": False,
            "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
        },
        owners=(
            inputs,
            inputs.compact_a,
            inputs.offsets,
            b_kn,
            output,
            resolved_config,
        ),
    )
