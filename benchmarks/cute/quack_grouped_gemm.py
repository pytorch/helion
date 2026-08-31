from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
from enum import Enum
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from urllib.parse import unquote
from urllib.parse import urlparse

from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute.grouped_gemm_workloads import BENCHMARK_B_LAYOUT
import torch

if TYPE_CHECKING:
    from types import ModuleType

    from benchmarks.cute.grouped_gemm_benchmark import GroupedGemmInputs
    from benchmarks.cute.grouped_gemm_benchmark import PreparedImplementation

_NATIVE_FILE_MARKERS = (".so", ".dylib", ".dll", ".pyd")
QUACK_REPOSITORY = "https://github.com/Dao-AILab/quack"
# The pinned post-release source still reports the preceding release version.
QUACK_PACKAGE_METADATA_VERSION = "0.6.4"
QUACK_BASE_RELEASE_TAG = "v0.6.4"
QUACK_COMMIT = "c8ec3170057987da0ec99883736f381ea1937cf3"
QUACK_BENCHMARK_LABEL = (
    f"quack-main@{QUACK_COMMIT[:8]} (post-{QUACK_BASE_RELEASE_TAG}, non-release)"
)
QUACK_DEPENDENCY_VERSIONS = {
    "nvidia-cutlass-dsl": "4.7.0",
    "apache-tvm-ffi": "0.1.13.post3",
    "torch-c-dlpack-ext": "0.1.5",
    "einops": "0.8.2",
}


def _source_provenance() -> dict[str, object]:
    return {
        "kind": "upstream_main_snapshot",
        "repository": QUACK_REPOSITORY,
        "commit": QUACK_COMMIT,
        "base_release_tag": QUACK_BASE_RELEASE_TAG,
        "is_formal_release": False,
        "benchmark_label": QUACK_BENCHMARK_LABEL,
    }


def _validate_loaded_quack_modules(root: Path) -> None:
    for name, module in sorted(sys.modules.items()):
        if module is not None and (name == "quack" or name.startswith("quack.")):
            common.validate_module_source(module, root, f"QuACK module {name!r}")


def _editable_root(distribution: importlib.metadata.Distribution) -> Path:
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("QuACK must be installed from an editable checkout")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("QuACK direct_url.json is invalid") from error
    url = direct_url.get("url") if isinstance(direct_url, dict) else None
    directory = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
    parsed = urlparse(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or not isinstance(directory, dict)
        or directory.get("editable") is not True
    ):
        raise RuntimeError(
            "QuACK distribution is not installed from an editable checkout"
        )
    return Path(unquote(parsed.path)).resolve(strict=True)


def _native_distribution_files(
    distribution: importlib.metadata.Distribution,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path)
            for path in distribution.files or ()
            if any(marker in Path(str(path)).name for marker in _NATIVE_FILE_MARKERS)
        )
    )


def _dependency_versions() -> dict[str, str]:
    distributions = common.require_pinned_distributions(
        QUACK_DEPENDENCY_VERSIONS,
        "QuACK dependencies are not pinned",
    )
    return {name: distribution.version for name, distribution in distributions.items()}


def verify_quack_installation(
    quack_root: Path | None = None,
) -> dict[str, object]:
    """Verify the installed QuACK distribution and pinned Python source."""

    distribution = common.require_pinned_distribution(
        "quack-kernels",
        QUACK_PACKAGE_METADATA_VERSION,
    )
    native_files = _native_distribution_files(distribution)
    installation = "editable"
    root = _editable_root(distribution) if quack_root is None else quack_root.resolve()
    if quack_root is not None:
        if native_files:
            raise RuntimeError(
                "--quack-root cannot override a distribution with native artifacts: "
                f"{native_files}"
            )
        installation = "installed_distribution_with_source_override"
    if str(root) not in sys.path:
        raise RuntimeError(
            "QuACK checkout must be present on the worker import path so "
            "compile subprocesses use the validated source"
        )
    dependency_versions = _dependency_versions()
    commit = common.clean_checkout(root, QUACK_COMMIT, "QuACK")
    module = _import_quack_from_root(root)
    common.validate_module_source(
        module,
        root,
        "QuACK module 'quack'",
    )

    return {
        "repository": QUACK_REPOSITORY,
        "upstream_commit": commit,
        "source_provenance": _source_provenance(),
        "distribution_version": distribution.version,
        "distribution_requirements": sorted(distribution.requires or ()),
        "dependency_versions": dependency_versions,
        "installation": installation,
        "source_root": str(root),
    }


def _import_quack_from_root(root: Path, name: str = "quack") -> ModuleType:
    _validate_loaded_quack_modules(root)
    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    _validate_loaded_quack_modules(root)
    return module


def _package_identity(quack_root: Path | None = None) -> dict[str, object]:
    identity = verify_quack_installation(quack_root)
    module = cast("Any", sys.modules["quack"])
    module_version = str(module.__version__)
    if module_version != QUACK_PACKAGE_METADATA_VERSION:
        raise RuntimeError(
            "QuACK module is "
            f"{module_version!r}, expected {QUACK_PACKAGE_METADATA_VERSION!r}"
        )
    return {
        **identity,
        "module_version": module_version,
    }


def _stable_dispatch_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
            "value": _stable_dispatch_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _stable_dispatch_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _stable_dispatch_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_dispatch_value(item) for item in value]
    as_dict = getattr(value, "_asdict", None)
    if as_dict is not None:
        return _stable_dispatch_value(as_dict())
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _dispatch_identity(dispatch_plan: object) -> dict[str, object]:
    fields = cast("Any", dispatch_plan)._asdict()
    compiled_fn = fields.pop("compiled_fn")
    return {
        "type": f"{type(dispatch_plan).__module__}.{type(dispatch_plan).__qualname__}",
        "compiled_callable_type": (
            f"{type(compiled_fn).__module__}.{type(compiled_fn).__qualname__}"
        ),
        "fields": _stable_dispatch_value(fields),
    }


def prepare_quack_default(
    inputs: GroupedGemmInputs,
    *,
    quack_root: Path | None = None,
) -> PreparedImplementation:
    """Select QuACK's public tuned default before capture, then replay it."""

    package = _package_identity(quack_root)
    source_root = Path(cast("str", package.pop("source_root")))
    try:
        interface = cast(
            "Any", _import_quack_from_root(source_root, "quack.gemm_interface")
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "QuACK requires the optional quack-kernels package"
        ) from error
    _validate_loaded_quack_modules(source_root)
    gemm = interface.gemm
    gemm_tuned = interface.gemm_tuned

    b_kn = inputs.b.transpose(1, 2)
    output = torch.empty(
        (inputs.case.total_m, inputs.case.n),
        device=inputs.compact_a.device,
        dtype=torch.bfloat16,
    )
    # This is the public API with the snapshot's actual defaults:
    # ``tuned=True``, ``dynamic_scheduler=False``, and ``split_k=1``. Run it
    # once during preparation so candidate compilation and benchmarking cannot
    # enter this benchmark's captured/timed replay.
    selected_output = gemm(
        inputs.compact_a,
        b_kn,
        out=output,
        cu_seqlens_m=inputs.offsets,
    )
    if selected_output is not output:
        raise RuntimeError("QuACK public GEMM replaced the provided output")
    selected_config = gemm_tuned.best_config.kwargs["config"]
    resolved_config, resolved_split_k, resolved_dynamic_scheduler, dispatch_plan = (
        gemm_tuned.fn(
            inputs.compact_a,
            b_kn,
            output,
            cu_seqlens_m=inputs.offsets,
            config=selected_config,
        )
    )
    if resolved_config != selected_config:
        raise RuntimeError("QuACK replay did not preserve the autotuned config")
    serialized = asdict(resolved_config)
    dispatch_identity = _dispatch_identity(dispatch_plan)

    def call() -> torch.Tensor:
        replay_config, replay_split_k, replay_dynamic, replay_dispatch = gemm_tuned.fn(
            inputs.compact_a,
            b_kn,
            output,
            cu_seqlens_m=inputs.offsets,
            config=resolved_config,
        )
        if (
            replay_config != resolved_config
            or replay_split_k != resolved_split_k
            or replay_dynamic != resolved_dynamic_scheduler
            or replay_dispatch is not dispatch_plan
        ):
            raise RuntimeError("QuACK resolved replay plan changed after selection")
        return output

    return common.PreparedImplementation(
        name=(
            f"quack-main-{QUACK_COMMIT[:8]}-public-default-tuned-{BENCHMARK_B_LAYOUT}"
        ),
        call=call,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: inputs.compact_output_slices(
            cast("torch.Tensor", result)
        ),
        config=common.provider_config(
            "quack",
            {
                "benchmark_label": QUACK_BENCHMARK_LABEL,
                "selection_api": "gemm(default tuned=True)",
                "replay_api": "gemm_tuned.fn(resolved config)",
                "requested_config": None,
                "selected_config": {**serialized, "b_layout": BENCHMARK_B_LAYOUT},
                "requested_dynamic_scheduler": False,
                "resolved_dynamic_scheduler": bool(resolved_dynamic_scheduler),
                "tuned": True,
                "resolved_split_k": int(resolved_split_k),
                "dispatch_plan": dispatch_identity,
                "package": package,
                "a_layout": common.compact_contiguous_a_layout(),
                "numerics": "BF16 inputs, FP32 accumulation, BF16 output",
            },
        ),
    )
