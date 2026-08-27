"""Compare attention backends per-shape on B200+ hardware.

Mirrors the methodology of ``compare_matmul_backends.py`` (steady-state
do_bench, thermal warmup, fresh-subprocess isolation per impl) for scaled
dot-product attention.

Impls:
    sdpa          torch.nn.functional.scaled_dot_product_attention -- the gold
                  reference baseline (B200 fused flash). Used as the
                  correctness reference for the other impls.
    fa4           FlashAttention-4 (the CuTe-DSL fwd, flash_attn.cute) -- the
                  upstream design our cute flash kernel is modelled on, so the
                  most meaningful throughput target. Loaded from HELION_FA4_ROOT,
                  an existing tritonbench flash-attention submodule, or, when
                  HELION_FA4_AUTO_DOWNLOAD=1, an auto-cloned
                  benchmarks/flash-attention checkout with a small CUDA-12.9 ->
                  CUDA-13 cutlass-ABI shim (see _import_fa4);
                  supports causal too.
    helion-triton examples.attention attention kernels with the DEFAULT
                  (triton) backend.
    helion-tileir examples.attention attention kernels with NVIDIA's
                  Triton-to-Tile-IR backend (ENABLE_TILE=1).
    tilegym-tileir NVIDIA TileGym's handwritten Triton FMHA variant compiled
                   with the active NV Triton TileIR backend.
    helion-cute   examples.attention attention kernels with
                  HELION_BACKEND=cute. By default this uses output-only
                  variants for dense, causal, and biased attention so Helion
                  does not compute an aux output that SDPA/FA4 omit.
                  NOTE: the cute attention path is being fixed in parallel and
                  may currently be numerically WRONG or slow. The harness flags
                  a cute correctness mismatch (accuracy=FAIL) but still reports
                  timing -- it never crashes on cute's current state.
    flexattention torch.nn.attention.flex_attention.flex_attention under
                  torch.compile(fullgraph=True), forced to Inductor's Triton
                  template. Causal uses a BlockMask; biased uses score_mod.
    flexattention-cute
                  The same PyTorch FlexAttention API forced to its experimental
                  CuTeDSL/FA4 FLASH backend.
    gluon        The Triton Gluon Blackwell attention-forward example loaded
                 from HELION_GLUON_ATTENTION_PATH (or TRITON_ROOT). Supports
                 dense and causal softmax attention.
    tlx          Meta TLX's Blackwell warp-specialized, pipelined persistent
                 attention tutorial. Loaded from an isolated TLX Triton runtime
                 selected with HELION_TLX_RUNTIME_ROOT.
    kernelagent-1x / kernelagent-2x / kernelagent-10x
                 Shape-specific kernels produced by KernelAgent with one or
                 more times the corresponding Helion full-autotune wall-clock
                 time. Loaded from --kernelagent-results-root.
    kernelagent-closed-1x / kernelagent-closed-2x
                 Shape-specific CuTeDSL kernels produced by KernelAgent Closed
                 v3-20260730 with the corresponding Helion full-autotune
                 wall-clock time. Loaded from
                 --kernelagent-closed-results-root.
Each impl runs in a fresh subprocess (so the HELION_BACKEND env mutation and
example imports do not leak between impls), with steady-state methodology
(10 s thermal warmup, do_bench warmup=1 s + rep=500 ms, 5 runs) and reports
best plus mom-median ms/TFLOP/s plus speedup vs sdpa.

Default is non-causal. Use ``--causal 1`` to benchmark causal kernels.

Examples:

    # Single-shape A/B across all impls
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --impl all --z 2 --h 8 --seq-len 512 --head-dim 64 --dtype float16

    # One impl, JSON line (used by --impl all subprocess collection)
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --impl sdpa --z 2 --h 32 --seq-len 1024 --head-dim 64 --json

    # Helion result from a strict cold full-LFBO search
    env -u PYTHONPATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \\
        python benchmarks/cute/compare_attention_backends.py \\
        --impl helion-cute --z 2 --h 32 --seq-len 32768 --head-dim 64 \\
        --helion-require-full-autotune 1 --helion-autotune-effort full

    # Representative shape sweep -> Markdown table
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --all-shapes --output attention_sweep.md \\
        --csv-output attention_sweep.csv --plot-output attention_sweep.png

    # Variant-focused sweep -> compact table across dense/causal/biased
    CUDA_VISIBLE_DEVICES=6 HELION_AUTOTUNE_EFFORT=full \\
        python benchmarks/cute/compare_attention_backends.py \\
        --all-shapes --shape-suite variants --helion-force-autotune 0 \\
        --stream-subprocesses \\
        --csv-output attention_variants.csv --plot-output attention_variants.png
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
from itertools import product
from itertools import starmap
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator
from typing import NoReturn
from typing import Protocol
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
# Running this file directly puts benchmarks/cute, rather than the checkout,
# first on sys.path. Pin imports to the checkout whose benchmark is running.
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion import Config
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.config_spec import ConfigSpec

DEFAULT_IMPLS = (
    "helion-triton",
    "helion-cute",
    "flexattention",
    "flexattention-cute",
    "sdpa",
    "fa4",
)
ALL_IMPLS = (
    "helion-triton",
    "helion-cute",
    "helion-tileir",
    "tilegym-tileir",
    "gluon",
    "tlx",
    "flexattention",
    "flexattention-cute",
    "sdpa",
    "fa4",
    "kernelagent-1x",
    "kernelagent-2x",
    "kernelagent-10x",
    "kernelagent-closed-1x",
    "kernelagent-closed-2x",
)
HELION_IMPLS = ("helion-cute", "helion-tileir", "helion-triton")
_STRICT_FINAL_CORRECTNESS_LAUNCHES = 64
_CUTE_FLASH_LANE_POLICY_VERSION = 14
_FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION = 2
_FLASH_TERMINAL_REFINEMENT_POLICY_VERSION = 2
_FLASH_TERMINAL_COORDINATE_POLICY = "same_leaf_full_surface_normalized_coordinate_v2"
_FLASH_TERMINAL_MEASUREMENT_POLICY = "mirrored_rotating_batched_wall_v2"
_FLASH_TERMINAL_COORDINATE_RADIUS = 2
_FLASH_TERMINAL_MINIMUM_IMPROVEMENT_FRACTION = 0.001
_FLASH_TERMINAL_REFINEMENT_TARGET_MS = 200.0
_FLASH_TERMINAL_CONFIRMATION_TARGET_MS = 5000.0
_FLASH_TERMINAL_REPEAT_MAX = 20_000
_FLASH_TERMINAL_MAX_SWEEPS = 64
_FLASH_TERMINAL_SURFACE_SCHEMA_VERSION = 1
_CORRECTNESS_MAX_TEMP_BYTES = 512 * 1024 * 1024
_CORRECTNESS_TEMP_BYTES_PER_ELEMENT = 20
_STRICT_FORBIDDEN_RUNTIME_ENV = frozenset(
    {
        "CUBLAS_FORCE_TF32",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_AUTO_BOOST",
        "CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_DISABLE_PTX_JIT",
        "CUDA_FORCE_PTX_JIT",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
        "CUDA_MODULE_LOADING",
        "NVIDIA_TF32_OVERRIDE",
        "PYTHONPATH",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    }
)
_STRICT_FORBIDDEN_RUNTIME_ENV_PREFIXES = (
    "CUDA_MPS_",
    "CUDNN_",
    "PYTORCH_TUNABLEOP_",
    "TORCH_CUDNN_",
)
_FULL_AUTOTUNE_ALLOWED_EXPLICIT_ENV = frozenset(
    {
        "CUTE_DSL_CACHE_DIR",
        "HELION_AUTOTUNER",
        "HELION_AUTOTUNER_INITIAL_POPULATION",
        "HELION_AUTOTUNE_ACCURACY_CHECK",
        "HELION_AUTOTUNE_ADAPTIVE_TIMEOUT",
        "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS",
        "HELION_AUTOTUNE_BENCHMARK_TIMEOUT",
        "HELION_AUTOTUNE_BEST_OF_K",
        "HELION_AUTOTUNE_BUDGET_SECONDS",
        "HELION_AUTOTUNE_CACHE",
        "HELION_AUTOTUNE_COMPILE_TIMEOUT",
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNE_FINAL_PICK_TOP_K",
        "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
        "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
        "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
        "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
        "HELION_AUTOTUNE_FORCE_PERSISTENT",
        "HELION_AUTOTUNE_IGNORE_ERRORS",
        "HELION_AUTOTUNE_MAX_GENERATIONS",
        "HELION_AUTOTUNE_RANDOM_SEED",
        "HELION_AUTOTUNE_SUSPICIOUS_REBENCHMARK_RATIO",
        "HELION_CACHE_DIR",
        "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS",
        "HELION_CAP_REBENCHMARK_REPEAT",
        "HELION_DISABLE_AUTOTUNER_HEURISTICS",
        "HELION_REBENCHMARK_THRESHOLD",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    }
)
_FULL_AUTOTUNE_ALLOWED_AMBIENT_HELION_ENV = _FULL_AUTOTUNE_ALLOWED_EXPLICIT_ENV | {
    "HELION_AUTOTUNE_LOG",
    "HELION_AUTOTUNE_LOG_DETAILS",
    "HELION_BACKEND",
    "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS",
}
_SOURCE_STABILITY_KEYS = (
    "helion_import_path",
    "helion_expected_package_path",
    "helion_import_root_matches_repo",
    "attention_example_import_path",
    "attention_example_expected_module_path",
    "attention_example_import_matches_repo",
    "helion_checkout_git_commit",
    "helion_source_tree_sha256",
    "helion_source_tree_file_count",
    "helion_source_tree_dirty",
)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _flash_terminal_coordinate_refinement_policy(
    effort_profile: object,
) -> dict[str, object] | None:
    structural = getattr(effort_profile, "flash_structural_search", None)
    lfbo = getattr(effort_profile, "lfbo_pattern_search", None)
    if structural is None or lfbo is None:
        return None
    return {
        "schema_version": _FLASH_TERMINAL_REFINEMENT_SCHEMA_VERSION,
        "policy_version": _FLASH_TERMINAL_REFINEMENT_POLICY_VERSION,
        "lane_policy_version": _CUTE_FLASH_LANE_POLICY_VERSION,
        "coordinate_policy": _FLASH_TERMINAL_COORDINATE_POLICY,
        "measurement_policy": _FLASH_TERMINAL_MEASUREMENT_POLICY,
        "rounds": structural.terminal_coordinate_rounds,
        "beam_width": structural.terminal_coordinate_beam_width,
        "radius": _FLASH_TERMINAL_COORDINATE_RADIUS,
        "minimum_improvement_fraction": (_FLASH_TERMINAL_MINIMUM_IMPROVEMENT_FRACTION),
        "round_target_ms": _FLASH_TERMINAL_REFINEMENT_TARGET_MS,
        "confirmation_target_ms": _FLASH_TERMINAL_CONFIRMATION_TARGET_MS,
    }


def _module_source_path(module_name: str) -> Path | None:
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        source = loaded.__file__
    else:
        spec = importlib.util.find_spec(module_name)
        source = None if spec is None else spec.origin
    return None if source is None else Path(source).resolve()


def _helion_source_provenance() -> dict[str, object]:
    helion_module_path = _module_source_path("helion")
    helion_expected_package_path = (REPO_ROOT / "helion").resolve()
    attention_module_path = _module_source_path("examples.attention")
    attention_expected_module_path = (REPO_ROOT / "examples" / "attention.py").resolve()
    source_snapshot = _git_source_snapshot(
        REPO_ROOT,
        (
            "helion",
            "examples/__init__.py",
            "examples/attention.py",
            "benchmarks/cute/compare_attention_backends.py",
            "sitecustomize.py",
            "usercustomize.py",
        ),
    )
    return {
        "helion_import_path": (
            None if helion_module_path is None else str(helion_module_path)
        ),
        "helion_expected_package_path": str(helion_expected_package_path),
        "helion_import_root_matches_repo": (
            helion_module_path is not None
            and helion_module_path.parent == helion_expected_package_path
        ),
        "attention_example_import_path": (
            None if attention_module_path is None else str(attention_module_path)
        ),
        "attention_example_expected_module_path": str(attention_expected_module_path),
        "attention_example_import_matches_repo": (
            attention_module_path == attention_expected_module_path
        ),
        "helion_checkout_git_commit": _git_commit(REPO_ROOT, "HEAD"),
        "helion_checkout_git_describe": _git_describe(REPO_ROOT),
        **source_snapshot,
    }


def _validate_helion_source_checkout(provenance: dict[str, object]) -> None:
    mismatches: list[str] = []
    if not provenance.get("helion_import_root_matches_repo"):
        mismatches.append("Helion")
    if not provenance.get("attention_example_import_matches_repo"):
        mismatches.append("the attention example")
    if mismatches:
        raise SystemExit(
            "--helion-require-full-autotune rejected this run: "
            "source modules were not imported from the checkout containing this "
            "benchmark: " + ", ".join(mismatches)
        )
    if provenance.get("helion_source_tree_dirty") is not False:
        raise SystemExit(
            "--helion-require-full-autotune rejected this run: the measured "
            "source checkout has tracked or untracked source changes"
        )


def _validate_helion_source_unchanged(
    initial_source: dict[str, object], current_source: dict[str, object]
) -> None:
    _validate_helion_source_checkout(current_source)
    changed = [
        key
        for key in _SOURCE_STABILITY_KEYS
        if initial_source.get(key) != current_source.get(key)
    ]
    if changed:
        raise SystemExit(
            "--helion-require-full-autotune rejected this run: the measured "
            "source checkout changed during tuning or measurement: "
            + ", ".join(changed)
        )


def _validate_post_measurement_source(provenance: dict[str, object]) -> None:
    final_source = _helion_source_provenance()
    _validate_helion_source_unchanged(provenance, final_source)
    provenance["post_measurement_source"] = final_source
    provenance["post_measurement_source_verified"] = True


@contextlib.contextmanager
def _strict_helion_runtime_environment(
    enabled: bool,
) -> Iterator[dict[str, object] | None]:
    if not enabled:
        yield None
        return

    forbidden = {
        key: value
        for key, value in os.environ.items()
        if key in _STRICT_FORBIDDEN_RUNTIME_ENV
        or key.startswith(_STRICT_FORBIDDEN_RUNTIME_ENV_PREFIXES)
    }
    if forbidden:
        raise SystemExit(
            "--helion-require-full-autotune conflicts with ambient runtime "
            "overrides: " + ", ".join(sorted(forbidden))
        )
    cuda_device_order = os.environ.get("CUDA_DEVICE_ORDER")
    if cuda_device_order != "PCI_BUS_ID":
        raise SystemExit(
            "--helion-require-full-autotune requires CUDA_DEVICE_ORDER=PCI_BUS_ID"
        )

    worker_pythonpath = str(REPO_ROOT)
    evidence: dict[str, object] = {
        "cuda_device_order": cuda_device_order,
        "forbidden_overrides": forbidden,
        "startup_pythonpath": None,
        "worker_pythonpath": worker_pythonpath,
    }
    os.environ["PYTHONPATH"] = worker_pythonpath
    try:
        yield evidence
    finally:
        os.environ.pop("PYTHONPATH", None)


class _ConfigLike(Protocol):
    config: dict[str, object]


class _ConfigSpecWithFlashSeeds(Protocol):
    compiler_default_config: object | None
    compiler_seed_configs: list[_ConfigLike]

    def default_config(self) -> _ConfigLike: ...


class _BoundWithConfigSpec(Protocol):
    @property
    def config_spec(self) -> _ConfigSpecWithFlashSeeds: ...


class _AutotuneMetricsLike(Protocol):
    def to_dict(self) -> dict[str, object]: ...


_FA4_REPO = "https://github.com/Dao-AILab/flash-attention.git"
_FA4_DEFAULT_REF = "v2.8.3"
_FA4_ROOT_ENV = "HELION_FA4_ROOT"
_FA4_REF_ENV = "HELION_FA4_REF"
_FA4_AUTO_DOWNLOAD_ENV = "HELION_FA4_AUTO_DOWNLOAD"
_TRITONBENCH_ROOT = REPO_ROOT / "benchmarks" / "tritonbench"
_FA4_TRITONBENCH_ROOT = _TRITONBENCH_ROOT / "submodules" / "flash-attention"
_FA4_STANDALONE_ROOT = REPO_ROOT / "benchmarks" / "flash-attention"
_GLUON_ATTENTION_PATH_ENV = "HELION_GLUON_ATTENTION_PATH"
_GLUON_VERSION_ENV = "HELION_GLUON_VERSION"
_TLX_ATTENTION_MODULE = (
    "triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent"
)
_TLX_ATTENTION_PATH_ENV = "HELION_TLX_ATTENTION_PATH"
_TLX_RUNTIME_ROOT_ENV = "HELION_TLX_RUNTIME_ROOT"
_TLX_REVISION_ENV = "HELION_TLX_REVISION"
_HELION_VERSION_ENV = "HELION_BENCHMARK_HELION_VERSION"
_KERNELAGENT_RESULTS_ROOT_ENV = "HELION_KERNELAGENT_RESULTS_ROOT"
_KERNELAGENT_CLOSED_RESULTS_ROOT_ENV = "HELION_KERNELAGENT_CLOSED_RESULTS_ROOT"
# (z, h, seq_len, head_dim, dtype, causal, biased)
_REPRESENTATIVE_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (1, 4, 512, 64, "float16", 0, 0),
    (2, 8, 512, 64, "float16", 0, 0),
    (2, 32, 1024, 64, "float16", 0, 0),
    (2, 32, 2048, 64, "float16", 0, 0),
    (4, 32, 4096, 128, "bfloat16", 0, 0),
    (8, 32, 8192, 128, "bfloat16", 0, 0),
    # causal variants
    (2, 32, 1024, 64, "float16", 1, 0),
    (2, 32, 4096, 64, "float16", 1, 0),
    (4, 32, 4096, 128, "bfloat16", 1, 0),
    # biased variant
    (2, 32, 1024, 64, "float16", 0, 1),
)

_VARIANT_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (2, 32, 2048, 64, "float16", 0, 0),
    (2, 32, 4096, 64, "float16", 1, 0),
    (1, 2, 128, 64, "float16", 0, 1),
)

_DENSE_CAUSAL8_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (2, 32, 32768, 64, "float16", 0, 0),
    (2, 32, 65536, 64, "float16", 0, 0),
    (2, 32, 131072, 64, "float16", 0, 0),
    (2, 32, 262144, 64, "float16", 0, 0),
    (2, 32, 65536, 64, "float16", 1, 0),
    (2, 32, 131072, 64, "float16", 1, 0),
    (2, 32, 262144, 64, "float16", 1, 0),
    (2, 32, 524288, 64, "float16", 1, 0),
)

_SHAPE_SUITES = {
    "representative": _REPRESENTATIVE_SHAPES,
    "variants": _VARIANT_SHAPES,
    "dense_causal8": _DENSE_CAUSAL8_SHAPES,
}

_DISPLAY_IMPLS = (
    "helion-triton",
    "helion-tileir",
    "tilegym-tileir",
    "flexattention",
    "gluon",
    "tlx",
    "flexattention-cute",
    "fa4",
    "sdpa",
    "helion-cute",
    "kernelagent-1x",
    "kernelagent-2x",
    "kernelagent-10x",
    "kernelagent-closed-1x",
    "kernelagent-closed-2x",
)
_IMPL_LABELS = {
    "helion-triton": "Helion (backend=Triton)",
    "helion-tileir": "Helion (backend=TileIR)",
    "tilegym-tileir": "TileGym+TileIR",
    "helion-cute": "Helion (backend=CuTe)",
    "gluon": "Gluon attention",
    "tlx": "TLX attention",
    "flexattention": "FlexAttention (backend=Triton)",
    "flexattention-cute": "FlexAttention (backend=CuTe)",
    "sdpa": "torch SDPA",
    "fa4": "FA4",
    "kernelagent-1x": "KernelAgent Public (1x Helion tuning time)",
    "kernelagent-2x": "KernelAgent Public (2x Helion tuning time)",
    "kernelagent-10x": "KernelAgent Public (10x Helion tuning time)",
    "kernelagent-closed-1x": "KernelAgent Closed (1x Helion tuning time)",
    "kernelagent-closed-2x": "KernelAgent Closed (2x Helion tuning time)",
}
_IMPL_KEYS = {
    "helion-triton": "helion_triton",
    "helion-tileir": "helion_tileir",
    "tilegym-tileir": "tilegym_tileir",
    "helion-cute": "helion_cute",
    "gluon": "gluon",
    "tlx": "tlx",
    "flexattention": "flexattention_triton",
    "flexattention-cute": "flexattention_cute",
    "sdpa": "torch_sdpa",
    "fa4": "fa4",
    "kernelagent-1x": "kernelagent_1x",
    "kernelagent-2x": "kernelagent_2x",
    "kernelagent-10x": "kernelagent_10x",
    "kernelagent-closed-1x": "kernelagent_closed_1x",
    "kernelagent-closed-2x": "kernelagent_closed_2x",
}
_FLEXATTENTION_BACKENDS = {
    "flexattention": "TRITON",
    "flexattention-cute": "FLASH",
}
_KERNELAGENT_BUDGET_LABELS = {
    "kernelagent-1x": "1x",
    "kernelagent-2x": "2x",
    "kernelagent-10x": "10x",
    "kernelagent-closed-1x": "1x",
    "kernelagent-closed-2x": "2x",
}
_KERNELAGENT_CLOSED_IMPLS = {
    "kernelagent-closed-1x",
    "kernelagent-closed-2x",
}

_ALLOWED_PHYSICAL_GPUS_ENV = "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS"


def _parse_key_value(value: str) -> tuple[str, str]:
    key, sep, raw_value = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, raw_value


def _parse_config_override(value: str) -> tuple[str, object]:
    key, raw_value = _parse_key_value(value)
    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed_value = raw_value
    return key, parsed_value


def _parse_plot_impl_label(value: str) -> tuple[str, str]:
    impl, label = _parse_key_value(value)
    if impl not in _IMPL_LABELS:
        raise argparse.ArgumentTypeError(
            f"unknown implementation {impl!r}; expected one of "
            f"{', '.join(_IMPL_LABELS)}"
        )
    if not label:
        raise argparse.ArgumentTypeError(f"plot label must not be empty for {impl!r}")
    return impl, label


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "unavailable"
    return torch.cuda.get_device_name()


def _physical_gpu_selection() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def _validate_strict_gpu_selection(require_full_autotune: bool) -> None:
    if not require_full_autotune:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = [] if visible is None else [part.strip() for part in visible.split(",")]
    if len(requested) != 1 or not requested[0]:
        raise SystemExit(
            "--helion-require-full-autotune requires CUDA_VISIBLE_DEVICES to "
            "select exactly one physical GPU"
        )


def _verify_power_cap_w(requested: int | None) -> int | None:
    """Return the measured power limit, rejecting mislabeled benchmark runs."""
    if requested is None:
        return None

    visible = _physical_gpu_selection()
    physical_gpu = visible.split(",", 1)[0].strip()
    if not physical_gpu:
        if not torch.cuda.is_available():
            raise SystemExit("cannot verify a GPU power cap without CUDA")
        physical_gpu = str(torch.cuda.current_device())
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                physical_gpu,
                "--query-gpu=power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        output_lines = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
        actual = float(output_lines[0])
    except (IndexError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(
            f"failed to verify the power limit for physical GPU {physical_gpu}"
        ) from exc
    if abs(actual - requested) > 0.5:
        raise SystemExit(
            f"physical GPU {physical_gpu} has a {actual:g} W power limit; "
            f"requested benchmark label is {requested} W"
        )
    return int(round(actual))


def _attention_flops(args: argparse.Namespace) -> float:
    """Attention FLOPs = 4 * z * h * seq^2 * head_dim (x0.5 if causal)."""
    flops = 4.0 * args.z * args.h * args.seq_len * args.seq_len * args.head_dim
    if args.causal:
        flops *= 0.5
    return flops


def _tflops(args: argparse.Namespace, ms: float) -> float:
    return _attention_flops(args) / (ms * 1e9)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "on", "yes")


def _valid_fa4_root(root: Path) -> bool:
    return (root / "flash_attn" / "cute").is_dir()


def _fa4_ref() -> str:
    return os.environ.get(_FA4_REF_ENV, _FA4_DEFAULT_REF)


def _git_commit(root: Path, rev: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return proc.stdout.strip()


def _git_source_snapshot(root: Path, pathspecs: tuple[str, ...]) -> dict[str, object]:
    """Fingerprint tracked and untracked source content used by the benchmark."""
    try:
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *pathspecs,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *pathspecs,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "helion_source_tree_sha256": None,
            "helion_source_tree_file_count": None,
            "helion_source_tree_dirty": None,
        }

    relative_paths = sorted(set(listed.rstrip("\0").split("\0"))) if listed else []
    digest = hashlib.sha256()
    file_count = 0
    for relative_path in relative_paths:
        source_path = root / relative_path
        if not source_path.is_file():
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(source_path.read_bytes()).digest())
        file_count += 1
    return {
        "helion_source_tree_sha256": digest.hexdigest(),
        "helion_source_tree_file_count": file_count,
        "helion_source_tree_dirty": bool(status),
    }


def _fa4_checkout_matches_ref(root: Path, ref: str) -> bool:
    head = _git_commit(root, "HEAD")
    target = _git_commit(root, ref)
    return head is not None and head == target


def _run_fa4_setup(cmd: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        command = " ".join(cmd)
        raise SystemExit(f"failed to set up FA4 checkout with `{command}`") from exc


def _checkout_fa4_ref(root: Path, ref: str) -> None:
    _run_fa4_setup(["git", "fetch", "--tags", "origin"], cwd=root)
    checkout_ref = ref
    if _git_commit(root, ref) is None:
        _run_fa4_setup(["git", "fetch", "origin", ref], cwd=root)
        if _git_commit(root, ref) is None:
            checkout_ref = "FETCH_HEAD"
    _run_fa4_setup(["git", "checkout", checkout_ref], cwd=root)
    if not _valid_fa4_root(root):
        raise SystemExit(
            f"{root} is checked out at {ref}, but flash_attn/cute is missing; "
            f"try another {_FA4_REF_ENV} value"
        )


def _fa4_root_at_ref(
    root: Path,
    ref: str,
    *,
    auto_checkout: bool,
    label: str,
) -> Path | None:
    if not _valid_fa4_root(root):
        return None
    if auto_checkout:
        try:
            _checkout_fa4_ref(root, ref)
        except SystemExit as exc:
            print(f"Skipping {label}: {exc}", file=sys.stderr)
            return None
        return root
    if _fa4_checkout_matches_ref(root, ref):
        return root
    print(
        f"Skipping {label}: {root} is not checked out at {ref}",
        file=sys.stderr,
    )
    return None


def _ensure_fa4_checkout(root: Path) -> Path:
    ref = _fa4_ref()
    if _valid_fa4_root(root):
        _checkout_fa4_ref(root, ref)
        return root
    if root.exists():
        raise SystemExit(
            f"{root} exists but does not look like a flash-attention checkout; "
            f"set {_FA4_ROOT_ENV}=<path> or remove the incomplete directory"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning FlashAttention for FA4 benchmark into {root}", file=sys.stderr)
    _run_fa4_setup(["git", "clone", "--filter=blob:none", _FA4_REPO, str(root)])
    _checkout_fa4_ref(root, ref)
    return root


def _ensure_fa4_tritonbench_submodule() -> Path | None:
    ref = _fa4_ref()
    if not (_TRITONBENCH_ROOT / ".git").exists():
        return None
    print(
        "Initializing TritonBench flash-attention submodule for FA4 benchmark",
        file=sys.stderr,
    )
    try:
        _run_fa4_setup(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "submodules/flash-attention",
            ],
            cwd=_TRITONBENCH_ROOT,
        )
    except SystemExit as exc:
        print(f"Skipping TritonBench FA4 submodule: {exc}", file=sys.stderr)
        return None
    tritonbench_root = _fa4_root_at_ref(
        _FA4_TRITONBENCH_ROOT,
        ref,
        auto_checkout=True,
        label="TritonBench FA4 submodule",
    )
    if tritonbench_root is not None:
        return tritonbench_root
    print(
        f"Skipping TritonBench FA4 submodule: {_FA4_TRITONBENCH_ROOT} "
        "is missing flash_attn/cute after submodule update",
        file=sys.stderr,
    )
    return None


def _resolve_fa4_root() -> Path:
    for env_name in (_FA4_ROOT_ENV, "FLASH_ATTENTION_ROOT"):
        root_str = os.environ.get(env_name)
        if root_str:
            root = Path(root_str).expanduser().resolve()
            if _valid_fa4_root(root):
                return root
            raise SystemExit(f"{env_name}={root} does not contain flash_attn/cute")
    auto_download = _env_flag(_FA4_AUTO_DOWNLOAD_ENV, default=False)
    ref = _fa4_ref()
    tritonbench_root = _fa4_root_at_ref(
        _FA4_TRITONBENCH_ROOT,
        ref,
        auto_checkout=auto_download,
        label="TritonBench FA4 submodule",
    )
    if tritonbench_root is not None:
        return tritonbench_root
    if _valid_fa4_root(_FA4_STANDALONE_ROOT):
        if auto_download:
            return _ensure_fa4_checkout(_FA4_STANDALONE_ROOT)
        if _fa4_checkout_matches_ref(_FA4_STANDALONE_ROOT, ref):
            return _FA4_STANDALONE_ROOT
        raise SystemExit(
            f"{_FA4_STANDALONE_ROOT} is not checked out at {ref}; "
            f"set {_FA4_AUTO_DOWNLOAD_ENV}=1 to update it or set "
            f"{_FA4_ROOT_ENV}=<path>"
        )
    if not auto_download:
        raise SystemExit(
            "FlashAttention checkout not found; set "
            f"{_FA4_ROOT_ENV}=<path>, initialize the TritonBench submodule, "
            f"or enable {_FA4_AUTO_DOWNLOAD_ENV}=1"
        )
    tritonbench_fa4_root = _ensure_fa4_tritonbench_submodule()
    if tritonbench_fa4_root is not None:
        return tritonbench_fa4_root
    return _ensure_fa4_checkout(_FA4_STANDALONE_ROOT)


def _import_tilegym_fmha() -> tuple[
    Callable[..., torch.Tensor],
    Callable[[], object | None],
]:
    # TileGym's recommended TileIR path enables these before its Triton kernels
    # compile. ``setdefault`` preserves explicit per-run overrides.
    os.environ.setdefault("TILEIR_ENABLE_APPROX", "1")
    os.environ.setdefault("TILEIR_ENABLE_FTZ", "1")

    from benchmarks.cute import tilegym_attention

    if tilegym_attention._get_available_triton_backend() != "nvt":
        raise RuntimeError(
            "TileGym attention did not detect the NV Triton TileIR backend; "
            "set ENABLE_TILE=1 and put NV Triton first on PYTHONPATH"
        )

    def get_best_config() -> object | None:
        return getattr(tilegym_attention._prefill_fmha, "best_config", None)

    return (
        cast(
            "Callable[..., torch.Tensor]",
            tilegym_attention.fmha_variant_triton,
        ),
        get_best_config,
    )


def _tilegym_attention_kwargs(
    args: argparse.Namespace, bias: torch.Tensor | None
) -> dict[str, object]:
    return {
        "scaling": None,
        "is_causal": bool(args.causal),
        "bias_type": "matrix" if bias is not None else None,
        "bias": bias,
        "layout": "bnsd",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_describe(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "describe",
                "--tags",
                "--always",
                "--dirty",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def _format_git_development_version(git_describe: str) -> str:
    """Convert ``v1.2.3-4-gabc`` into ``1.2.3.dev4+gabc``."""
    dirty = git_describe.endswith("-dirty")
    version = git_describe.removesuffix("-dirty").removeprefix("v")
    parts = version.rsplit("-", 2)
    if (
        len(parts) == 3
        and parts[1].isdigit()
        and parts[2].startswith("g")
        and len(parts[2]) > 1
    ):
        base, commit_count, commit = parts
        version = f"{base}.dev{commit_count}+{commit}"
    if dirty:
        return f"{version}.dirty" if "+" in version else f"{version}+dirty"
    return version


def _cudnn_version() -> str:
    version = torch.backends.cudnn.version()
    if version is None:
        return "unknown"
    major, remainder = divmod(version, 10000)
    minor, patch = divmod(remainder, 100)
    return f"{major}.{minor}.{patch}"


def _tileir_toolchain_version() -> str:
    try:
        tileir_conf = importlib.import_module("triton.backends.tileir.conf")
        tileiras = tileir_conf.TileIREnvConf.get_tileiras_path()
        proc = subprocess.run(
            [tileiras, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (ImportError, OSError, subprocess.CalledProcessError):
        return "unknown"
    for line in proc.stdout.splitlines():
        prefix = "Cuda compilation tools, release "
        if line.startswith(prefix):
            return line.removeprefix(prefix).split(",", 1)[0]
    return "unknown"


def _resolve_gluon_attention_path() -> Path:
    configured = os.environ.get(_GLUON_ATTENTION_PATH_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise SystemExit(f"{_GLUON_ATTENTION_PATH_ENV}={path} is not a file")

    triton_root = os.environ.get("TRITON_ROOT")
    if triton_root:
        path = (
            Path(triton_root).expanduser().resolve()
            / "python"
            / "examples"
            / "gluon"
            / "01-attention-forward.py"
        )
        if path.is_file():
            return path
    raise SystemExit(
        "Gluon attention example not found; set "
        f"{_GLUON_ATTENTION_PATH_ENV}=<triton/python/examples/gluon/"
        "01-attention-forward.py> or TRITON_ROOT=<triton checkout>"
    )


def _resolve_tlx_attention_path() -> Path:
    configured = os.environ.get(_TLX_ATTENTION_PATH_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise SystemExit(f"{_TLX_ATTENTION_PATH_ENV}={path} is not a file")

    runtime_root = os.environ.get(_TLX_RUNTIME_ROOT_ENV)
    if runtime_root:
        path = (
            Path(runtime_root).expanduser().resolve()
            / "triton"
            / "language"
            / "extra"
            / "tlx"
            / "tutorials"
            / "blackwell_fa_ws_pipelined_persistent.py"
        )
        if path.is_file():
            return path

    try:
        spec = importlib.util.find_spec(_TLX_ATTENTION_MODULE)
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is not None and spec.origin is not None:
        path = Path(spec.origin).resolve()
        if path.is_file():
            return path
    raise SystemExit(
        "TLX attention example not found; set "
        f"{_TLX_RUNTIME_ROOT_ENV}=<isolated TLX runtime> or "
        f"{_TLX_ATTENTION_PATH_ENV}=<blackwell attention source>"
    )


def _implementation_version(
    impl: str, *, resolve_external_sources: bool = True
) -> dict[str, str]:
    helion_version = os.environ.get(_HELION_VERSION_ENV)
    if helion_version is None:
        helion_describe = _git_describe(REPO_ROOT)
        helion_version = (
            _format_git_development_version(helion_describe)
            if helion_describe is not None
            else "unknown"
        )
    triton_version = _package_version("triton")
    triton_label_version = triton_version.split("+", 1)[0]
    torch_version = torch.__version__
    torch_label_version = torch_version.split("+", 1)[0]
    if impl in _KERNELAGENT_BUDGET_LABELS:
        return {
            "version": "KernelAgent metadata is supplied by the run manifest",
            "version_label": "run manifest metadata",
        }
    if impl == "helion-cute":
        cute_version = _package_version("nvidia-cutlass-dsl")
        return {
            "version": f"Helion {helion_version}; CuTe {cute_version}",
            "version_label": f"Helion {helion_version} / CuTe {cute_version}",
        }
    if impl == "helion-triton":
        return {
            "version": f"Helion {helion_version}; Triton {triton_version}",
            "version_label": (
                f"Helion {helion_version} / Triton {triton_label_version}"
            ),
        }
    if impl == "helion-tileir":
        nvtriton_version = _package_version("nvtriton")
        tileir_version = _tileir_toolchain_version()
        return {
            "version": (
                f"Helion {helion_version}; nvtriton {nvtriton_version}; "
                f"TileIR {tileir_version}"
            ),
            "version_label": (
                f"Helion {helion_version} / nvtriton {nvtriton_version} / "
                f"TileIR {tileir_version}"
            ),
        }
    if impl == "tilegym-tileir":
        nvtriton_version = _package_version("nvtriton")
        tileir_version = _tileir_toolchain_version()
        return {
            "version": (
                f"TileGym attention; nvtriton {nvtriton_version}; "
                f"TileIR {tileir_version}"
            ),
            "version_label": (
                f"TileGym / nvtriton {nvtriton_version} / TileIR {tileir_version}"
            ),
        }
    if impl == "gluon":
        revision = os.environ.get(_GLUON_VERSION_ENV, triton_version)
        if not resolve_external_sources:
            return {
                "version": (
                    f"Triton {triton_version}; example {revision}; "
                    "source not resolved (implementation skipped)"
                ),
                "version_label": f"Triton {triton_label_version}",
            }
        source_path = _resolve_gluon_attention_path()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        return {
            "version": (
                f"Triton {triton_version}; example {revision}; "
                f"source sha256 {source_hash}"
            ),
            "version_label": f"Triton {triton_label_version}",
        }
    if impl == "tlx":
        if not resolve_external_sources:
            return {
                "version": "TLX version not resolved (implementation skipped)",
                "version_label": "version not resolved",
            }
        source_path = _resolve_tlx_attention_path()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        triton_module = importlib.import_module("triton")
        meta_triton_version = str(getattr(triton_module, "__version__", triton_version))
        revision = os.environ.get(_TLX_REVISION_ENV)
        revision_text = f"; revision {revision}" if revision else ""
        return {
            "version": (
                f"Meta Triton {meta_triton_version}; integrated TLX; "
                f"package {triton_version}"
                f"{revision_text}; source sha256 {source_hash}"
            ),
            "version_label": f"Meta Triton {meta_triton_version}",
        }
    if impl in _FLEXATTENTION_BACKENDS:
        backend = _FLEXATTENTION_BACKENDS[impl]
        backend_version = (
            f"Triton {triton_version}"
            if backend == "TRITON"
            else (
                f"FA4 {_git_describe(_resolve_fa4_root()) or 'unknown'}; "
                f"CuTe {_package_version('nvidia-cutlass-dsl')}"
            )
        )
        backend_version_label = (
            f"Triton {triton_label_version}"
            if backend == "TRITON"
            else (
                f"FA4 {_git_describe(_resolve_fa4_root()) or 'unknown'}; "
                f"CuTe {_package_version('nvidia-cutlass-dsl')}"
            )
        )
        return {
            "version": f"PyTorch {torch_version}; {backend_version}",
            "version_label": (
                f"PyTorch {torch_label_version}; {backend_version_label}"
            ),
        }
    if impl == "sdpa":
        cudnn_version = _cudnn_version()
        cudnn_package_version = _package_version("nvidia-cudnn-cu13")
        cudnn_label = (
            cudnn_package_version
            if cudnn_package_version != "unknown"
            else cudnn_version
        )
        return {
            "version": (
                f"PyTorch {torch_version}; cuDNN runtime {cudnn_version}; "
                f"nvidia-cudnn-cu13 {cudnn_package_version}"
            ),
            "version_label": f"cuDNN {cudnn_label}",
        }
    if impl == "fa4":
        if not resolve_external_sources:
            return {
                "version": "FlashAttention version not resolved (implementation skipped)",
                "version_label": "version not resolved",
            }
        root = _resolve_fa4_root()
        revision = _git_describe(root) or (_git_commit(root, "HEAD") or "unknown")[:12]
        cute_version = _package_version("nvidia-cutlass-dsl")
        return {
            "version": f"FlashAttention {revision}; CuTe {cute_version}",
            "version_label": f"{revision}; CuTe {cute_version}",
        }
    raise ValueError(f"unknown implementation {impl!r}")


def _uses_bias(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "biased", 0))


def _epilogue(args: argparse.Namespace) -> str:
    epilogue = str(getattr(args, "epilogue", "none"))
    if epilogue not in ("none", "relu"):
        raise ValueError(f"unsupported attention epilogue {epilogue!r}")
    return epilogue


def _validate_epilogue_workload(args: argparse.Namespace) -> None:
    if _epilogue(args) == "none":
        return
    if args.dtype != "bfloat16":
        raise SystemExit("--epilogue relu requires --dtype bfloat16")
    if _uses_bias(args):
        raise SystemExit("--epilogue relu is not compatible with --biased 1")
    if bool(getattr(args, "helion_return_lse", 0)):
        raise SystemExit("--epilogue relu requires --helion-return-lse 0")


def _apply_output_epilogue(output: torch.Tensor, epilogue: str) -> torch.Tensor:
    if epilogue == "none":
        return output
    if epilogue == "relu":
        return torch.relu(output)
    raise AssertionError(f"unsupported attention epilogue {epilogue!r}")


def _check_gpu_policy() -> None:
    """Optionally restrict sweeps to a caller-provided CUDA_VISIBLE_DEVICES set."""
    allowed_raw = os.environ.get(_ALLOWED_PHYSICAL_GPUS_ENV, "").strip()
    if not allowed_raw:
        return
    allowed = tuple(item.strip() for item in allowed_raw.split(",") if item.strip())
    if not allowed:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise SystemExit(
            "Refusing to run without CUDA_VISIBLE_DEVICES set because "
            f"{_ALLOWED_PHYSICAL_GPUS_ENV}={allowed_raw!r} is configured."
        )
    requested = [item.strip() for item in visible.split(",") if item.strip()]
    bad = [gpu for gpu in requested if gpu not in allowed]
    if bad:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={visible!r} selects disallowed GPU(s) {bad}; "
            f"{_ALLOWED_PHYSICAL_GPUS_ENV} allows only {allowed}."
        )


def _make_inputs(
    args: argparse.Namespace, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed)
    shape = (args.z, args.h, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def _check_kernelagent_output(actual: object, expected: torch.Tensor) -> bool:
    if not isinstance(actual, torch.Tensor):
        return False
    if (
        actual.shape != expected.shape
        or actual.dtype is not torch.float16
        or actual.device.type != "cuda"
    ):
        return False
    return _check_close(actual, expected, torch.float16)


def _check_kernelagent_repeat(first: object, repeated: object) -> bool:
    return (
        isinstance(first, torch.Tensor)
        and isinstance(repeated, torch.Tensor)
        and first.shape == repeated.shape
        and first.dtype is repeated.dtype
        and first.device == repeated.device
        and bool(torch.equal(first, repeated))
    )


def _check_kernelagent_stress_case(
    run: Callable[..., torch.Tensor], args: argparse.Namespace, dtype: torch.dtype
) -> bool:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    shape = (args.z, args.h, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    q.mul_(2.0)
    k.mul_(2.0)
    v.add_(
        torch.randn(
            (args.z, args.h, 1, args.head_dim),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
    )
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    ):
        expected = _sdpa_reference(q, k, v, causal=bool(args.causal))
    actual = run(q, k, v)
    repeated = run(q, k, v)
    return (
        _check_kernelagent_output(actual, expected)
        and _check_kernelagent_output(repeated, expected)
        and _check_kernelagent_repeat(actual, repeated)
    )


def _kernelagent_evaluation_note(
    backend_name: str,
    selection_version: str,
    evaluation_version: str,
    *,
    standard_executed: bool,
    repeat_executed: bool,
    stress_executed: bool,
    passed: bool,
    measured: bool,
) -> str:
    prefix = (
        f"Source selected with {backend_name} {selection_version}; recompiled with "
        f"{backend_name} {evaluation_version}. Performance was "
        f"{'measured' if measured else 'not measured'}."
    )
    if not standard_executed:
        return f"{prefix} Final-harness correctness checks were skipped."
    if not repeat_executed:
        return f"{prefix} The standard full-output check failed; repeat and stress were not run."
    if not stress_executed:
        return f"{prefix} The standard check passed, but exact repeatability failed."
    if not passed:
        return f"{prefix} Standard and repeat checks passed, but stress failed."
    return (
        f"{prefix} Standard and stress full-output checks passed with exact "
        "repeatability."
    )


def _make_bias(args: argparse.Namespace, dtype: torch.dtype) -> torch.Tensor | None:
    if not _uses_bias(args):
        return None
    torch.manual_seed(args.seed + 1)
    shape = (args.z, args.h, args.seq_len, args.seq_len)
    return torch.randn(shape, device="cuda", dtype=dtype) * 0.25


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias,
        is_causal=causal,
    )


def _attention_output_reference(
    args: argparse.Namespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output = _sdpa_reference(q, k, v, causal=bool(args.causal), bias=bias)
    return _apply_output_epilogue(output, _epilogue(args))


def _comparison_chunk_rows(
    tensor: torch.Tensor, *, max_temp_bytes: int = _CORRECTNESS_MAX_TEMP_BYTES
) -> int:
    """Bound temporary memory used by a full-output floating-point comparison."""
    if tensor.ndim == 0:
        return 1
    row_dim = tensor.ndim - 2 if tensor.ndim >= 2 else 0
    rows = int(tensor.shape[row_dim])
    if rows <= 0:
        return 1
    elements_per_row = max(1, tensor.numel() // rows)
    bytes_per_row = elements_per_row * _CORRECTNESS_TEMP_BYTES_PER_ELEMENT
    return max(1, min(rows, max_temp_bytes // bytes_per_row))


def _check_tensor_close_in_chunks(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_temp_bytes: int = _CORRECTNESS_MAX_TEMP_BYTES,
) -> bool:
    if actual.ndim == 0:
        actual_float = actual.float()
        expected_float = expected.float()
        return bool(
            (
                torch.isfinite(actual_float)
                & torch.isfinite(expected_float)
                & torch.isclose(
                    actual_float,
                    expected_float,
                    atol=5e-2,
                    rtol=2e-2,
                )
            ).item()
        )

    row_dim = actual.ndim - 2 if actual.ndim >= 2 else 0
    rows = int(actual.shape[row_dim])
    chunk_rows = _comparison_chunk_rows(actual, max_temp_bytes=max_temp_bytes)
    for start in range(0, rows, chunk_rows):
        index = [slice(None)] * actual.ndim
        index[row_dim] = slice(start, min(start + chunk_rows, rows))
        actual_float = actual[tuple(index)].float()
        expected_float = expected[tuple(index)].float()
        chunk_is_valid = (
            torch.isfinite(actual_float).all()
            & torch.isfinite(expected_float).all()
            & torch.isclose(
                actual_float,
                expected_float,
                atol=5e-2,
                rtol=2e-2,
            ).all()
        )
        if not bool(chunk_is_valid.item()):
            return False
    return True


def _check_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Dtype-aware correctness check; returns True on pass.

    bf16/fp16 attention accumulates softmax rounding noise, so benchmark smoke
    checks use a looser threshold than unit tests.
    """
    if (
        actual.shape != expected.shape
        or actual.dtype is not expected.dtype
        or actual.device != expected.device
    ):
        return False
    return _check_tensor_close_in_chunks(actual, expected)


def _check_attention_result_close(
    actual: object, expected: object, dtype: torch.dtype
) -> bool:
    """Check output-only or output-plus-LSE attention results."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _check_close(actual, expected, dtype)
    if not (
        isinstance(actual, tuple)
        and isinstance(expected, tuple)
        and len(actual) == len(expected)
        and actual
    ):
        return False
    return all(
        isinstance(actual_value, torch.Tensor)
        and isinstance(expected_value, torch.Tensor)
        and _check_close(actual_value, expected_value, dtype)
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _attention_results_equal(first: object, second: object) -> bool:
    """Check exact repeatability for output-only or output-plus-LSE results."""
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return (
            first.shape == second.shape
            and first.dtype is second.dtype
            and first.device == second.device
            and bool(torch.equal(first, second))
        )
    if not (
        isinstance(first, tuple)
        and isinstance(second, tuple)
        and len(first) == len(second)
        and first
    ):
        return False
    return all(starmap(_attention_results_equal, zip(first, second, strict=True)))


def _check_attention_result_repeatedly(
    run: Callable[[], object],
    expected: object,
    dtype: torch.dtype,
    *,
    launches: int,
) -> bool:
    """Check every launch numerically and require bitwise repeatability."""
    assert launches >= 1
    first: object | None = None
    for _ in range(launches):
        actual = run()
        if not _check_attention_result_close(actual, expected, dtype):
            return False
        if first is not None and not _attention_results_equal(first, actual):
            return False
        if first is None:
            if isinstance(actual, torch.Tensor):
                first = actual.clone()
            else:
                first = tuple(
                    value.clone() if isinstance(value, torch.Tensor) else value
                    for value in cast("tuple[object, ...]", actual)
                )
    return True


def _gpu_warmup(duration_ms: int = 10000) -> None:
    """Drive the GPU to a stable clock state with sustained matmul work.

    Without warmup the first benchmark in a new process is at the mercy of the
    GPU's current clock state (idle vs sustained boost on B200, with a ~5-7 s
    cold-to-boost ramp). The steady-state number under sustained load is the
    canonical one; the warmup ensures we start there.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000.0
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(50):
            a = a @ a
        torch.cuda.synchronize()


def _bench_steady(
    fn: Callable[[], object],
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    do_bench_fn: Callable[..., Any] | None = None,
    cache_warmup_calls: int = 5,
    thermal_warmup_ms: int = 10000,
) -> dict[str, Any]:
    """Steady-state benchmark.

    1. Cache warmup: call fn() a few times to populate per-launch caches.
    2. Thermal warmup: drive the GPU to a stable clock state.
    3. Measurement: ``num_runs`` of do_bench(warmup, rep). Returns
       best/mom-median/mean across runs; mom-median is the gate metric,
       best-of-N is diagnostic only. CuTe can inject the backend wall-clock
       timer here because CUDA-event timing mis-times CuTe kernels on Blackwell.
    """
    bench_fn = do_bench_fn
    if bench_fn is None:
        from triton.testing import do_bench

        bench_fn = cast("Callable[..., Any]", do_bench)

    for _ in range(cache_warmup_calls):
        fn()
    torch.cuda.synchronize()

    _gpu_warmup(thermal_warmup_ms)

    runs: list[float] = []
    for _ in range(num_runs):
        # Every backend must be scored on the same per-run statistic. Triton
        # do_bench defaults to the mean, which is right-skewed for latency and
        # would diverge from the median reported by the CuTe wall-clock timer
        # (the statistic Helion autotune also uses to validate winners).
        ms = bench_fn(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        if isinstance(ms, tuple):
            ms = ms[0]
        assert isinstance(ms, float)
        runs.append(ms)

    return {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "mean_ms": sum(runs) / len(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs_ms": runs,
    }


def _result(
    impl: str,
    args: argparse.Namespace,
    stats: dict[str, Any] | None,
    *,
    accuracy: str,
    benchmark_timer: str,
    config: object = None,
    codegen: dict[str, bool] | None = None,
    helion_overrides: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    version_info: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "impl": impl,
        **(version_info if version_info is not None else _implementation_version(impl)),
        "shape": _shape_dict(args),
        "gpu": _gpu_name(),
        "physical_gpu": _physical_gpu_selection(),
        "power_cap_w": getattr(args, "power_cap_w", None),
        "input_seed": args.seed,
        "flop_model": "softmax_attention_forward",
        "accuracy": accuracy,
        "benchmark_timer": benchmark_timer,
    }
    if _epilogue(args) != "none":
        payload["epilogue_flops_included"] = False
    if stats is not None:
        payload.update(
            {
                "best_ms": stats["best_ms"],
                "median_ms": stats["median_ms"],
                "mom_median_ms": stats["median_ms"],
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
                "runs_ms": stats["runs_ms"],
                "best_tflops": _tflops(args, stats["best_ms"]),
                "median_tflops": _tflops(args, stats["median_ms"]),
                "mom_median_tflops": _tflops(args, stats["median_ms"]),
            }
        )
    if config is not None:
        payload["config"] = config
    if codegen is not None:
        payload["codegen"] = codegen
    if helion_overrides is not None:
        payload["helion_overrides"] = helion_overrides
    if notes:
        payload["notes"] = notes
    return payload


def _skipped_result(impl: str, args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "impl": impl,
        **_implementation_version(impl, resolve_external_sources=False),
        "shape": _shape_dict(args),
        "gpu": _gpu_name(),
        "physical_gpu": _physical_gpu_selection(),
        "power_cap_w": getattr(args, "power_cap_w", None),
        "input_seed": args.seed,
        "flop_model": "not_comparable",
        "accuracy": "SKIP",
        "skipped_reason": reason,
        "notes": [reason],
    }


def _shape_dict(args: argparse.Namespace) -> dict[str, Any]:
    shape = {
        "z": args.z,
        "h": args.h,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": int(args.causal),
        "biased": int(_uses_bias(args)),
    }
    epilogue = _epilogue(args)
    if epilogue != "none":
        shape["epilogue"] = epilogue
    return shape


def _helion_codegen_markers(code: str) -> dict[str, bool]:
    tcgen05_prefixes = ("cute.nvgpu.tcgen05.", "cute_tcgen05_flash.")
    return {
        "uses_tcgen05": any(prefix in code for prefix in tcgen05_prefixes),
        "uses_tcgen05_two_cta": any(
            f"{prefix}CtaGroup.TWO" in code for prefix in tcgen05_prefixes
        )
        or "is_two_cta=True" in code
        or "'use_2cta_instrs': True" in code,
        "uses_tma_umma_pipeline": "PipelineTmaUmma.create(" in code,
        "uses_relu_epilogue": (
            "_helion_flash_rt.relu_fragment_inplace(" in code
            or "relu_output=True" in code
        ),
    }


def _helion_selected_source_code(bound: object, active_config: object) -> str:
    typed_bound = cast("Any", bound)
    return cast(
        "str",
        typed_bound.to_triton_code(
            active_config,
            emit_repro_caller=bool(typed_bound.settings.print_output_code),
        ),
    )


def _helion_override_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for key, value in _helion_env_overrides(args).items():
        result.extend(["--helion-env", f"{key}={value}"])
    for key, value in getattr(args, "helion_config", ()):
        result.extend(["--helion-config", f"{key}={json.dumps(value)}"])
    for key, value in getattr(args, "helion_seed_config", ()):
        result.extend(["--helion-seed-config", f"{key}={json.dumps(value)}"])
    return result


def _codegen_override_env_keys(explicit_env_keys: set[str]) -> list[str]:
    """Ambient or explicit env keys that alter CuTe flash or runtime codegen."""
    ambient_helion_overrides = {
        key
        for key in os.environ
        if key.startswith("HELION_")
        and key not in _FULL_AUTOTUNE_ALLOWED_AMBIENT_HELION_ENV
    }
    return sorted(
        key
        for key in set(os.environ) | explicit_env_keys
        if (
            key in ambient_helion_overrides
            or key.startswith("HELION_CUTE_")
            or (
                key.startswith("CUTE_DSL_")
                and key != "CUTE_DSL_CACHE_DIR"
                # The launcher normalizes an ambient CUTE_DSL_ARCH to the
                # tensor device before compilation. It can remain after an
                # earlier CuTe launch in this process, but an explicit CLI
                # override still indicates an uncontrolled experiment.
                and (key != "CUTE_DSL_ARCH" or key in explicit_env_keys)
            )
            or key
            in {
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "CUDA_LAUNCH_BLOCKING",
            }
        )
    )


def _helion_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = dict(getattr(args, "helion_env", ()))
    require_full_autotune = bool(getattr(args, "helion_require_full_autotune", 0))
    if require_full_autotune:
        if getattr(args, "helion_seed_config", ()):
            raise SystemExit(
                "--helion-require-full-autotune does not allow user-provided "
                "--helion-seed-config values"
            )
        if getattr(args, "helion_config", ()):
            raise SystemExit(
                "--helion-require-full-autotune does not allow fixed "
                "--helion-config values"
            )
        if bool(getattr(args, "skip_correctness", 0)):
            raise SystemExit(
                "--helion-require-full-autotune requires final correctness checking"
            )
        explicit_env_keys = set(env_overrides)
        codegen_overrides = _codegen_override_env_keys(explicit_env_keys)
        if codegen_overrides:
            raise SystemExit(
                "--helion-require-full-autotune conflicts with CuTe flash or "
                "runtime codegen overrides: " + ", ".join(codegen_overrides)
            )
        unknown_explicit_env = sorted(
            explicit_env_keys - _FULL_AUTOTUNE_ALLOWED_EXPLICIT_ENV
        )
        if unknown_explicit_env:
            raise SystemExit(
                "--helion-require-full-autotune does not allow unknown "
                "--helion-env overrides: " + ", ".join(unknown_explicit_env)
            )
        disable_heuristics = env_overrides.get("HELION_DISABLE_AUTOTUNER_HEURISTICS")
        if disable_heuristics is not None and disable_heuristics.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            raise SystemExit(
                "--helion-require-full-autotune conflicts with "
                "HELION_DISABLE_AUTOTUNER_HEURISTICS=1"
            )
        env_overrides["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "0"
        autotuner_name = env_overrides.get("HELION_AUTOTUNER")
        if autotuner_name:
            raise SystemExit(
                "--helion-require-full-autotune conflicts with a custom "
                "HELION_AUTOTUNER"
            )
        env_overrides["HELION_AUTOTUNER"] = ""
        neighbor_cap = env_overrides.get("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS")
        if neighbor_cap not in (None, "-1"):
            raise SystemExit(
                "--helion-require-full-autotune requires "
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS=-1"
            )
        env_overrides["HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"] = "-1"
        benchmark_subprocess = env_overrides.get(
            "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS",
            os.environ.get("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS"),
        )
        if (
            benchmark_subprocess is not None
            and benchmark_subprocess.strip().lower()
            in {
                "0",
                "false",
                "no",
                "off",
            }
        ):
            raise SystemExit(
                "--helion-require-full-autotune requires the default isolated "
                "autotune benchmark subprocess"
            )
        rebenchmark_env = sorted(
            key
            for key in (
                "HELION_AUTOTUNE_FINAL_PICK_TOP_K",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
                "HELION_AUTOTUNE_SUSPICIOUS_REBENCHMARK_RATIO",
                "HELION_CAP_REBENCHMARK_REPEAT",
                "HELION_REBENCHMARK_THRESHOLD",
            )
            if key in os.environ or key in env_overrides
        )
        if rebenchmark_env:
            raise SystemExit(
                "--helion-require-full-autotune conflicts with rebenchmark "
                "overrides: " + ", ".join(rebenchmark_env)
            )
    helion_autotune_effort = getattr(args, "helion_autotune_effort", None)
    helion_autotune_budget_seconds = getattr(
        args, "helion_autotune_budget_seconds", None
    )
    helion_autotune_max_generations = getattr(
        args, "helion_autotune_max_generations", None
    )
    helion_autotune_best_of_k = getattr(args, "helion_autotune_best_of_k", None)
    helion_autotune_benchmark_timeout = getattr(
        args, "helion_autotune_benchmark_timeout", None
    )
    helion_autotune_accuracy_check = getattr(
        args, "helion_autotune_accuracy_check", None
    )
    helion_autotuner_initial_population = getattr(
        args, "helion_autotuner_initial_population", None
    )
    if helion_autotune_effort is not None:
        env_overrides["HELION_AUTOTUNE_EFFORT"] = helion_autotune_effort
    if helion_autotune_budget_seconds is not None:
        env_overrides["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(
            helion_autotune_budget_seconds
        )
    if helion_autotune_max_generations is not None:
        env_overrides["HELION_AUTOTUNE_MAX_GENERATIONS"] = str(
            helion_autotune_max_generations
        )
    if helion_autotune_best_of_k is not None:
        env_overrides["HELION_AUTOTUNE_BEST_OF_K"] = str(helion_autotune_best_of_k)
    if helion_autotune_benchmark_timeout is not None:
        env_overrides["HELION_AUTOTUNE_BENCHMARK_TIMEOUT"] = str(
            helion_autotune_benchmark_timeout
        )
    if helion_autotune_accuracy_check is not None:
        env_overrides["HELION_AUTOTUNE_ACCURACY_CHECK"] = str(
            int(helion_autotune_accuracy_check)
        )
    if require_full_autotune:
        explicit_initial_population = (
            helion_autotuner_initial_population
            or env_overrides.get("HELION_AUTOTUNER_INITIAL_POPULATION")
        )
        if explicit_initial_population not in (None, "from_random"):
            raise SystemExit(
                "--helion-require-full-autotune requires "
                "HELION_AUTOTUNER_INITIAL_POPULATION=from_random"
            )
        env_overrides["HELION_AUTOTUNER_INITIAL_POPULATION"] = "from_random"
    elif helion_autotuner_initial_population is not None:
        env_overrides["HELION_AUTOTUNER_INITIAL_POPULATION"] = (
            helion_autotuner_initial_population
        )
    return env_overrides


def _apply_helion_env(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = _helion_env_overrides(args)
    os.environ.update(env_overrides)
    return env_overrides


@contextlib.contextmanager
def _capture_helion_autotune_metrics() -> Iterator[list[dict[str, object]]]:
    from helion.autotuner.metrics import register_post_autotune_hook
    from helion.autotuner.metrics import remove_post_autotune_hook

    captured: list[dict[str, object]] = []

    def capture(metrics: _AutotuneMetricsLike) -> None:
        values = metrics.to_dict()
        trial = {
            key: values[key]
            for key in (
                "input_shapes",
                "dtypes",
                "hardware",
                "random_seed",
                "search_algorithm",
                "num_configs_tested",
                "num_compile_failures",
                "num_worker_failures",
                "num_isolated_rebenchmark_timeouts",
                "num_accuracy_failures",
                "num_successful_candidate_measurements",
                "num_unique_sources",
                "num_source_deduplications",
                "num_generations",
                "autotune_time",
                "best_perf_ms",
                "selected_config",
                "selected_source_hash",
                "selected_source_was_measured",
            )
        }
        if "search_phase_metrics" in values:
            trial["search_phase_metrics"] = values["search_phase_metrics"]
        captured.append(trial)

    register_post_autotune_hook(capture)
    try:
        yield captured
    finally:
        remove_post_autotune_hook(capture)


def _flash_normalization_context(config_spec: ConfigSpec) -> dict[str, object]:
    """Return the shape and active ConfigSpec identity used by normalization."""
    if not config_spec.cute_flash_search_enabled:
        return {
            "schema_version": 1,
            "backend": config_spec.backend_name,
            "enabled": False,
        }
    default_config = dict(config_spec.default_config().config)
    context: dict[str, object] = {
        "schema_version": 1,
        "backend": config_spec.backend_name,
        "config_spec_structural_fingerprint_sha256": (
            config_spec.structural_fingerprint_hash()
        ),
        "default_config_sha256": hashlib.sha256(
            json.dumps(default_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "dtype": str(config_spec._cute_flash_dtype),
        "head_dim": config_spec._cute_flash_head_dim,
        "num_kv": config_spec._cute_flash_num_kv,
        "num_bh": config_spec._cute_flash_num_bh,
        "tensor_4d_heads": config_spec._cute_flash_tensor_4d_heads,
        "is_causal": config_spec._cute_flash_is_causal,
        "has_kv_tile_pruning": config_spec._cute_flash_has_kv_tile_pruning,
        "requires_ws_overlap": config_spec._cute_flash_requires_ws_overlap,
        "small_biased_candidate": config_spec._cute_flash_small_biased_candidate,
        "standard_dense_output": config_spec._cute_flash_standard_dense_output,
        "standard_causal_output": config_spec._cute_flash_standard_causal_output,
        "output_requires_tma": config_spec._cute_flash_output_requires_tma,
        "supports_tensor_4d_tma": config_spec._cute_flash_supports_tensor_4d_tma,
        "block_size_targets": [
            [block_id, target]
            for block_id, target in sorted(
                config_spec._cute_flash_block_size_targets.items()
            )
        ],
        "flat_key_layout": [
            [key, count, is_sequence]
            for key, count, is_sequence in config_spec.flat_key_layout()
        ],
    }
    return context


def _live_canonical_cute_flash_compiler_seeds(
    config_spec: ConfigSpec,
) -> list[Config]:
    """Regenerate compiler-owned seeds through ConfigSpec's canonical API."""
    return list(config_spec.autotune_seed_configs())


def _ordered_effective_config_ids(
    config_generation: ConfigGeneration,
    configs: Sequence[Config],
) -> tuple[list[str], int]:
    """Normalize configs exactly as generation zero does, preserving rank order."""
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.exc import InvalidConfig

    result: list[str] = []
    seen: set[object] = set()
    invalid_count = 0
    for config in configs:
        try:
            _flat, normalized = config_generation.canonicalize_flat(
                config_generation.flatten(config)
            )
        except (InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
            invalid_count += 1
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(canonical_config_id(normalized))
    return result, invalid_count


def _compiler_seed_policy(
    config_spec: ConfigSpec,
    config_generation: ConfigGeneration,
) -> dict[str, object]:
    """Describe and authenticate the only compiler seeds allowed in strict runs."""
    actual_configs = list(config_spec.compiler_seed_configs)
    expected_configs = _live_canonical_cute_flash_compiler_seeds(config_spec)
    effective_config_ids, invalid_config_count = _ordered_effective_config_ids(
        config_generation, actual_configs
    )
    (
        expected_effective_config_ids,
        expected_invalid_config_count,
    ) = _ordered_effective_config_ids(config_generation, expected_configs)
    heuristic_names = list(config_spec.autotuner_heuristics)
    timeout_retry_repetitions = config_spec.compiler_seed_timeout_retry_repetitions
    is_canonical = (
        config_spec.cute_flash_search_enabled
        and heuristic_names == ["cute_flash_attention"]
        and [dict(config.config) for config in actual_configs]
        == [dict(config.config) for config in expected_configs]
        and invalid_config_count == 0
        and expected_invalid_config_count == 0
        and effective_config_ids == expected_effective_config_ids
        and timeout_retry_repetitions == (3 if expected_configs else None)
        and bool(effective_config_ids)
    )
    return {
        "schema_version": 1,
        "kind": "canonical_cute_flash" if is_canonical else "noncanonical",
        "heuristic_names": heuristic_names,
        "raw_config_count": len(actual_configs),
        "effective_config_ids": effective_config_ids,
        "effective_config_ids_sha256": hashlib.sha256(
            json.dumps(effective_config_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "timeout_retry_repetitions": timeout_retry_repetitions,
    }


def _canonical_flash_projection(
    config_generation: ConfigGeneration,
    source: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    """Project a config through the same normalization path used by search."""
    from helion.runtime.config import Config

    requested = Config.from_dict(source)
    requested.config.update(overrides)
    flat = config_generation.flatten(requested)
    _flat, normalized = config_generation.canonicalize_flat(flat)
    return dict(normalized.config)


def _flash_clc_lane_provenance(
    config_generation: ConfigGeneration,
    *,
    leaf_catalog: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Serialize ConfigGeneration's deterministic CLC catalog and witnesses."""
    from helion.autotuner.search_space_logger import canonical_config_id

    catalog = config_generation.flash_clc_lane_catalog()
    witnesses = config_generation.flash_clc_lane_witnesses()
    allowed = (
        None
        if leaf_catalog is None
        else {
            (leaf["family"], leaf["compound_packet"], leaf["softmax_disc"])
            for leaf in leaf_catalog
            if isinstance(leaf, dict)
            and set(leaf) == {"family", "compound_packet", "softmax_disc"}
        }
    )
    records: list[dict[str, object]] = []
    for leaf in config_generation.flash_structural_leaf_catalog():
        if leaf not in catalog or (
            allowed is not None
            and (
                leaf.pipeline_family,
                leaf.compound_exp2_packet,
                leaf.softmax_disc,
            )
            not in allowed
        ):
            continue
        lane = catalog[leaf]
        attempted = lane["attempted_values"]
        records.append(
            {
                "family": leaf.pipeline_family,
                "compound_packet": leaf.compound_exp2_packet,
                "softmax_disc": leaf.softmax_disc,
                "legal_values": list(lane["legal_values"]),
                "search_values": list(lane["search_values"]),
                "anchor_values": list(lane["anchor_values"]),
                "refinement_values": list(lane["refinement_values"]),
                "planned_values": list(attempted),
                "witness_config_ids": {
                    str(value): canonical_config_id(witnesses[(leaf, value)])
                    for value in attempted
                },
            }
        )
    return records


def _helion_autotune_provenance(
    args: argparse.Namespace,
    bound: object,
    fixed_config: dict[str, object] | None,
    expected_baseline_fn: Callable[..., object],
    strict_runtime_environment: dict[str, object] | None = None,
) -> dict[str, object]:
    from helion._compiler.cute.cute_flash import FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import flash_exp2_packet_is_compound
    from helion.autotuner.base_cache import should_skip_cache
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.effort_profile import get_effort_profile
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.runtime.settings import default_autotuner_fn

    settings = cast("Any", bound).settings
    config_spec = cast("Any", bound).config_spec
    kernel = cast("Any", bound).kernel
    active_fields = config_spec._flat_fields()
    active_value_prior_keys = sorted(
        set(config_spec.backend.config_value_priors(config_spec)) & set(active_fields)
    )
    fragment_default_config = dict(config_spec.default_config().config)
    fragment_default_json = json.dumps(
        fragment_default_config, sort_keys=True, separators=(",", ":")
    )
    flash_normalization_context = _flash_normalization_context(config_spec)
    flash_normalization_context_json = json.dumps(
        flash_normalization_context, sort_keys=True, separators=(",", ":")
    )
    config_generation = ConfigGeneration(config_spec)
    compiler_seed_policy = _compiler_seed_policy(config_spec, config_generation)
    structural_coverage_design_configs = [
        dict(config.config)
        for config in config_generation.flash_deterministic_population_configs()
    ]
    structural_coverage_uncovered_values = [
        {"key": key, "value": value}
        for key, value in config_generation.flash_structural_coverage_uncovered_values()
    ]
    structural_coverage_underqualified_values = [
        {"key": key, "value": value, "witness_count": witness_count}
        for key, value, witness_count in (
            config_generation.flash_structural_coverage_underqualified_values()
        )
    ]
    structural_leaves = config_generation.flash_structural_leaf_catalog()
    structural_leaf_catalog = [
        {
            "family": leaf.pipeline_family,
            "compound_packet": leaf.compound_exp2_packet,
            "softmax_disc": leaf.softmax_disc,
        }
        for leaf in structural_leaves
    ]
    pipeline_lanes_by_leaf = config_generation.flash_pipeline_lane_catalog()
    pipeline_lane_catalog = [
        {
            "family": leaf.pipeline_family,
            "compound_packet": leaf.compound_exp2_packet,
            "softmax_disc": leaf.softmax_disc,
            "pipeline_lanes": [
                {"key": key, "value": value}
                for key, value in pipeline_lanes_by_leaf[leaf]
            ],
        }
        for leaf in structural_leaves
    ]
    clc_lane_catalog = _flash_clc_lane_provenance(config_generation)
    clc_lane_catalog_json = json.dumps(
        clc_lane_catalog, sort_keys=True, separators=(",", ":")
    )
    structural_coverage_underqualified_leaves = [
        {
            "family": leaf.pipeline_family,
            "compound_packet": leaf.compound_exp2_packet,
            "softmax_disc": leaf.softmax_disc,
            "witness_count": witness_count,
        }
        for leaf, witness_count in (
            config_generation.flash_structural_coverage_underqualified_leaves()
        )
    ]
    structural_coverage_active_interactions = [
        {"keys": list(keys), "values": list(values)}
        for keys, values in (
            config_generation.flash_structural_coverage_active_interactions()
        )
    ]
    structural_coverage_uncovered_interactions = [
        {"keys": list(keys), "values": list(values)}
        for keys, values in (
            config_generation.flash_structural_coverage_uncovered_interactions()
        )
    ]
    structural_coverage_active_values = [
        {"key": key, "value": value}
        for key, value in config_generation.flash_structural_coverage_active_values()
    ]
    structural_qualification_values = [
        active
        for active in structural_coverage_active_values
        if active["key"] == FLASH_PIPELINE_FAMILY_KEY
        or (
            active["key"] == FLASH_EXP2_PACKET_KEY
            and flash_exp2_packet_is_compound(active["value"])
        )
    ]
    structural_coverage_design = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        for config in structural_coverage_design_configs
    ]
    structural_coverage_design_json = json.dumps(
        structural_coverage_design_configs, sort_keys=True, separators=(",", ":")
    )
    requested_force = bool(args.helion_force_autotune)
    effective_force = requested_force or bool(settings.force_autotune)
    skip_cache = should_skip_cache()
    autotuner_fn = settings.autotuner_fn
    selected_effort_profile = get_effort_profile(settings.autotune_effort)
    effort_profile = selected_effort_profile.lfbo_pattern_search
    flash_structural_search = selected_effort_profile.flash_structural_search
    terminal_coordinate_policy = (
        _flash_terminal_coordinate_refinement_policy(selected_effort_profile)
        if config_spec.cute_flash_search_enabled
        else None
    )
    terminal_coordinate_surface_catalog = (
        config_generation.flash_terminal_coordinate_surface_catalog(
            radius=_FLASH_TERMINAL_COORDINATE_RADIUS
        )
        if terminal_coordinate_policy is not None
        else None
    )
    initial_population_size = (
        effort_profile.initial_population if effort_profile is not None else None
    )
    exact_effective_configs = (
        config_generation.flash_exact_effective_search_space_configs(
            initial_population_size
        )
        if initial_population_size is not None
        else None
    )
    exact_effective_config_ids = (
        [canonical_config_id(config) for config in exact_effective_configs]
        if exact_effective_configs is not None
        else None
    )
    structural_population_budget = (
        config_generation.flash_structural_population_budget(initial_population_size)
        if initial_population_size is not None
        else None
    )
    structural_injected_design_count = (
        min(structural_population_budget, len(structural_coverage_design))
        if structural_population_budget is not None
        else None
    )
    retained_family_cap = (
        flash_structural_search.retained_families
        if flash_structural_search is not None
        else None
    )
    retained_family_limit = (
        config_generation.flash_structural_effective_family_limit(retained_family_cap)
        if flash_structural_search is not None
        else None
    )
    promoted_path_limit = (
        config_generation.flash_structural_starting_path_limit(
            minimum=max(
                effort_profile.copies,
                flash_structural_search.starting_paths,
            ),
            retained_families=flash_structural_search.retained_families,
            retained_candidates_per_leaf=(
                flash_structural_search.retained_candidates_per_leaf
            ),
        )
        if effort_profile is not None and flash_structural_search is not None
        else None
    )
    family_probe_path_limit = (
        config_generation.flash_structural_family_probe_path_limit(
            flash_structural_search.retained_families,
            flash_structural_search.family_probe_generations,
        )
        if flash_structural_search is not None
        else None
    )
    return {
        **_helion_source_provenance(),
        "physical_gpu_selection": _physical_gpu_selection(),
        "strict_runtime_environment": strict_runtime_environment,
        "require_full_autotune": bool(getattr(args, "helion_require_full_autotune", 0)),
        "effort": settings.autotune_effort,
        "requested_force_autotune": requested_force,
        "effective_force_autotune": effective_force,
        "fixed_config": fixed_config is not None,
        "autotune_budget_seconds": settings.autotune_budget_seconds,
        "autotune_max_generations": settings.autotune_max_generations,
        "autotune_best_of_k": settings.autotune_best_of_k,
        "autotune_accuracy_check": settings.autotune_accuracy_check,
        "autotune_compile_timeout": settings.autotune_compile_timeout,
        "autotune_benchmark_subprocess": settings.autotune_benchmark_subprocess,
        "autotune_benchmark_subprocess_env": os.environ.get(
            "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", ""
        ),
        "autotune_benchmark_timeout": settings.autotune_benchmark_timeout,
        "autotune_adaptive_timeout": settings.autotune_adaptive_timeout,
        "autotune_force_persistent": settings.autotune_force_persistent,
        "autotune_finishing_rounds_env": os.environ.get(
            "HELION_AUTOTUNE_FINISHING_ROUNDS", ""
        ),
        "autotune_ignore_errors": settings.autotune_ignore_errors,
        "autotune_random_seed": settings.autotune_random_seed,
        "autotune_cache": settings.autotune_cache,
        "disable_autotuner_heuristics": settings.disable_autotuner_heuristics,
        "autotune_initial_population_strategy_override": (
            settings.autotune_initial_population_strategy
        ),
        "autotune_initial_population_size": initial_population_size,
        "flash_exact_effective_search_space_size": (
            len(exact_effective_config_ids)
            if exact_effective_config_ids is not None
            else None
        ),
        "flash_exact_effective_search_space_config_ids": exact_effective_config_ids,
        "flash_exact_effective_search_space_sha256": (
            hashlib.sha256(
                json.dumps(exact_effective_config_ids, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            if exact_effective_config_ids is not None
            else None
        ),
        "autotune_lfbo_max_generations": (
            effort_profile.max_generations if effort_profile is not None else None
        ),
        "autotuner_initial_population_env": os.environ.get(
            "HELION_AUTOTUNER_INITIAL_POPULATION", ""
        ),
        "autotuner_env": os.environ.get("HELION_AUTOTUNER", ""),
        "autotune_num_neighbors_cap_env": os.environ.get(
            "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", ""
        ),
        "autotuner_fn": (
            f"{getattr(autotuner_fn, '__module__', '')}."
            f"{getattr(autotuner_fn, '__qualname__', type(autotuner_fn).__qualname__)}"
        ),
        "autotuner_fn_is_default": autotuner_fn is default_autotuner_fn,
        "autotune_baseline_fn": (
            f"{getattr(settings.autotune_baseline_fn, '__module__', '')}."
            f"{getattr(settings.autotune_baseline_fn, '__qualname__', '')}"
            if settings.autotune_baseline_fn is not None
            else ""
        ),
        "autotune_baseline_fn_is_expected": (
            settings.autotune_baseline_fn is expected_baseline_fn
        ),
        "autotune_baseline_atol": settings.autotune_baseline_atol,
        "autotune_baseline_rtol": settings.autotune_baseline_rtol,
        "autotune_baseline_accuracy_check_fn": (
            settings.autotune_baseline_accuracy_check_fn is not None
        ),
        "autotune_benchmark_fn": settings.autotune_benchmark_fn is not None,
        "autotune_rebenchmark_threshold": settings.autotune_rebenchmark_threshold,
        "autotune_suspicious_rebenchmark_ratio": (
            settings.autotune_suspicious_rebenchmark_ratio
        ),
        "autotune_config_overrides": settings.autotune_config_overrides,
        "autotune_search_acf": settings.autotune_search_acf,
        "autotune_config_filter": settings.autotune_config_filter is not None,
        "user_seed_configs": bool(settings.autotune_seed_configs),
        "compiler_seed_config_count": len(config_spec.compiler_seed_configs),
        "compiler_seed_policy": compiler_seed_policy,
        "compiler_default_config": config_spec.compiler_default_config is not None,
        "kernel_declared_config_count": len(kernel.configs),
        "active_value_prior_keys": active_value_prior_keys,
        "flash_value_prior_keys": [
            key for key in active_value_prior_keys if key.startswith("cute_flash_")
        ],
        "flash_fragment_default_config": fragment_default_config,
        "flash_fragment_default_sha256": hashlib.sha256(
            fragment_default_json.encode("utf-8")
        ).hexdigest(),
        "flash_normalization_context": flash_normalization_context,
        "flash_normalization_context_sha256": hashlib.sha256(
            flash_normalization_context_json.encode("utf-8")
        ).hexdigest(),
        "flash_structural_coverage_design_source": (
            "normalized active ConfigSpec fragments"
            if config_spec.cute_flash_search_enabled
            else "disabled"
        ),
        "flash_structural_coverage_active_values": structural_coverage_active_values,
        "flash_structural_coverage_design": structural_coverage_design,
        "flash_structural_coverage_design_count": len(structural_coverage_design),
        "flash_structural_coverage_design_sha256": hashlib.sha256(
            structural_coverage_design_json.encode("utf-8")
        ).hexdigest(),
        "flash_structural_coverage_uncovered_values": (
            structural_coverage_uncovered_values
        ),
        "flash_structural_coverage_underqualified_values": (
            structural_coverage_underqualified_values
        ),
        "flash_structural_leaf_catalog": structural_leaf_catalog,
        "flash_pipeline_lane_catalog": pipeline_lane_catalog,
        "flash_clc_lane_catalog": clc_lane_catalog,
        "flash_clc_lane_catalog_sha256": hashlib.sha256(
            clc_lane_catalog_json.encode("utf-8")
        ).hexdigest(),
        "flash_structural_coverage_underqualified_leaves": (
            structural_coverage_underqualified_leaves
        ),
        "flash_structural_coverage_interaction_key_groups": [
            list(group) for group in FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
        ],
        "flash_structural_coverage_active_interactions": (
            structural_coverage_active_interactions
        ),
        "flash_structural_coverage_uncovered_interactions": (
            structural_coverage_uncovered_interactions
        ),
        "flash_structural_qualification_values": structural_qualification_values,
        "flash_structural_parent_coverage_prefix_count": (
            config_generation.flash_structural_parent_coverage_prefix_count()
        ),
        "flash_structural_qualification_prefix_count": (
            config_generation.flash_structural_qualification_prefix_count()
        ),
        "flash_structural_population_budget": structural_population_budget,
        "flash_structural_injected_design_count": (structural_injected_design_count),
        "flash_structural_qualification_rounds": (
            flash_structural_search.qualification_rounds
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": (
            flash_structural_search.pipeline_candidates_per_leaf_per_round
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_family_probe_generations": (
            flash_structural_search.family_probe_generations
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_family_probe_candidates_per_path": (
            flash_structural_search.family_probe_candidates_per_path
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_retained_candidates_per_leaf": (
            flash_structural_search.retained_candidates_per_leaf
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_retained_family_cap": retained_family_cap,
        "flash_structural_retained_family_limit": retained_family_limit,
        "flash_structural_retained_family_slowdown_limit": (
            flash_structural_search.retained_family_slowdown_limit
            if flash_structural_search is not None
            else None
        ),
        "flash_structural_starting_path_limit": promoted_path_limit,
        "flash_structural_family_probe_path_limit": family_probe_path_limit,
        "flash_structural_maximum_path_capacity": (
            max(promoted_path_limit, family_probe_path_limit)
            if promoted_path_limit is not None and family_probe_path_limit is not None
            else None
        ),
        "flash_structural_unrestricted_path_exhausts_generation_budget": (
            flash_structural_search.exhaust_unrestricted_path
            if flash_structural_search is not None
            else None
        ),
        "flash_terminal_coordinate_refinement_policy": terminal_coordinate_policy,
        "flash_terminal_coordinate_refinement_policy_sha256": (
            _canonical_json_sha256(terminal_coordinate_policy)
            if terminal_coordinate_policy is not None
            else None
        ),
        "flash_terminal_coordinate_surface_catalog": (
            terminal_coordinate_surface_catalog
        ),
        "flash_terminal_coordinate_surface_catalog_sha256": (
            _canonical_json_sha256(terminal_coordinate_surface_catalog)
            if terminal_coordinate_surface_catalog is not None
            else None
        ),
        "cute_flash_env_overrides": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("HELION_CUTE_FLASH")
        },
        "final_correctness_enabled": not bool(args.skip_correctness),
        "cache_read_policy": ("bypass" if effective_force or skip_cache else "allow"),
        "cache_write_policy": "bypass" if skip_cache else "write",
        "skip_cache_env": skip_cache,
        "rebenchmark_env_overrides": {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "HELION_AUTOTUNE_FINAL_PICK_TOP_K",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
                "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
                "HELION_AUTOTUNE_SUSPICIOUS_REBENCHMARK_RATIO",
                "HELION_CAP_REBENCHMARK_REPEAT",
                "HELION_REBENCHMARK_THRESHOLD",
            }
        },
    }


def _structural_coverage_design(
    provenance: dict[str, object],
) -> list[dict[str, object]]:
    """Read the complete design, including artifacts from prior schemas."""
    design = provenance.get("flash_structural_coverage_design")
    if design is None:
        design = provenance.get("flash_structural_coverage_candidates")
    if design is None:
        design = provenance.get("deterministic_flash_population_anchors", [])
    return cast("list[dict[str, object]]", design)


def _record_selected_structural_coverage_design_provenance(
    provenance: dict[str, object],
) -> None:
    """Record how far the measured winner moved from the covering design."""
    selected = cast("dict[str, object]", provenance["selected_config"])
    design = _structural_coverage_design(provenance)
    if not design:
        provenance["selected_config_is_structural_coverage_design_member"] = False
        provenance[
            "selected_config_nearest_structural_coverage_design_field_distance"
        ] = None
        provenance[
            "selected_config_nearest_structural_coverage_design_config_sha256"
        ] = []
        return

    distances: list[tuple[int, str]] = []
    for candidate in design:
        candidate_config = cast("dict[str, object]", candidate["config"])
        keys = selected.keys() | candidate_config.keys()
        distance = sum(
            key not in selected
            or key not in candidate_config
            or selected[key] != candidate_config[key]
            for key in keys
        )
        distances.append((distance, cast("str", candidate["config_sha256"])))
    nearest_distance = min(distance for distance, _sha256 in distances)
    provenance["selected_config_is_structural_coverage_design_member"] = (
        nearest_distance == 0
    )
    provenance["selected_config_nearest_structural_coverage_design_field_distance"] = (
        nearest_distance
    )
    provenance["selected_config_nearest_structural_coverage_design_config_sha256"] = [
        sha256 for distance, sha256 in distances if distance == nearest_distance
    ]


def _flash_structural_leaf_dict(
    config: dict[str, object],
) -> dict[str, object] | None:
    """Return the JSON representation used by structural-search provenance."""
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config

    leaf = flash_structural_leaf_from_config(config)
    if leaf is None:
        return None
    return {
        "family": leaf.pipeline_family,
        "compound_packet": leaf.compound_exp2_packet,
        "softmax_disc": leaf.softmax_disc,
    }


def _flash_terminal_surface_catalog_error(catalog: object) -> str | None:
    if not isinstance(catalog, dict) or set(catalog) != {
        "schema_version",
        "radius",
        "leaves",
    }:
        return "the terminal coordinate surface catalog has an invalid envelope"
    if (
        type(catalog.get("schema_version")) is not int
        or catalog.get("schema_version") != _FLASH_TERMINAL_SURFACE_SCHEMA_VERSION
        or type(catalog.get("radius")) is not int
        or catalog.get("radius") != _FLASH_TERMINAL_COORDINATE_RADIUS
    ):
        return "the terminal coordinate surface catalog has an unexpected policy"
    leaves = catalog.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        return "the terminal coordinate surface catalog contains no leaves"
    seen_leaves: set[str] = set()
    for leaf_record in leaves:
        if not isinstance(leaf_record, dict) or set(leaf_record) != {
            "leaf",
            "coordinates",
        }:
            return "the terminal coordinate surface catalog has a malformed leaf"
        leaf = leaf_record.get("leaf")
        if (
            not isinstance(leaf, dict)
            or set(leaf) != {"family", "compound_packet", "softmax_disc"}
            or not isinstance(leaf.get("family"), str)
            or not isinstance(leaf.get("softmax_disc"), bool)
            or (
                leaf.get("compound_packet") is not None
                and not isinstance(leaf.get("compound_packet"), str)
            )
        ):
            return "the terminal coordinate surface catalog has a malformed leaf ID"
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        if leaf_key in seen_leaves:
            return "the terminal coordinate surface catalog contains duplicate leaves"
        seen_leaves.add(leaf_key)
        coordinates = leaf_record.get("coordinates")
        if not isinstance(coordinates, list) or not coordinates:
            return "the terminal coordinate surface catalog has an empty leaf"
        for expected_flat_index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, dict) or set(coordinate) != {
                "flat_index",
                "key",
                "sequence_index",
                "fragment_type",
                "overridden",
                "active_values",
                "neighbors_by_value",
            }:
                return (
                    "the terminal coordinate surface catalog has a malformed coordinate"
                )
            sequence_index = coordinate.get("sequence_index")
            if (
                type(coordinate.get("flat_index")) is not int
                or coordinate.get("flat_index") != expected_flat_index
                or not isinstance(coordinate.get("key"), str)
                or (
                    sequence_index is not None
                    and (
                        isinstance(sequence_index, bool)
                        or not isinstance(sequence_index, int)
                        or sequence_index < 0
                    )
                )
                or not isinstance(coordinate.get("fragment_type"), str)
                or not isinstance(coordinate.get("overridden"), bool)
                or not isinstance(coordinate.get("active_values"), list)
            ):
                return (
                    "the terminal coordinate surface catalog has an invalid coordinate"
                )
            neighbors = coordinate.get("neighbors_by_value")
            if not isinstance(neighbors, list) or any(
                not isinstance(row, dict)
                or set(row) != {"from_value", "to_values"}
                or not isinstance(row.get("to_values"), list)
                for row in neighbors
            ):
                return "the terminal coordinate surface catalog has malformed neighbors"
    return None


def _validate_required_full_autotune(provenance: dict[str, object]) -> None:
    if not provenance["require_full_autotune"]:
        return

    from helion._compiler.cute.cute_flash import FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
    from helion.autotuner.effort_profile import get_effort_profile

    violations: list[str] = []
    physical_gpu_selection = provenance.get("physical_gpu_selection")
    if (
        not isinstance(physical_gpu_selection, str)
        or len(physical_gpu_selection.split(",")) != 1
        or not physical_gpu_selection.strip()
    ):
        violations.append("the run does not identify exactly one physical GPU")
    strict_runtime_environment = provenance.get("strict_runtime_environment")
    if strict_runtime_environment != {
        "cuda_device_order": "PCI_BUS_ID",
        "forbidden_overrides": {},
        "startup_pythonpath": None,
        "worker_pythonpath": str(REPO_ROOT),
    }:
        violations.append("the strict runtime environment is not controlled")
    if not provenance.get("helion_import_root_matches_repo"):
        violations.append(
            "Helion was not imported from the checkout containing this benchmark"
        )
    if not provenance.get("attention_example_import_matches_repo"):
        violations.append(
            "the attention example was not imported from the checkout containing "
            "this benchmark"
        )
    source_tree_sha256 = provenance.get("helion_source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256) is None
    ):
        violations.append("the benchmark source tree has no valid content fingerprint")
    source_tree_file_count = provenance.get("helion_source_tree_file_count")
    if (
        isinstance(source_tree_file_count, bool)
        or not isinstance(source_tree_file_count, int)
        or source_tree_file_count <= 0
    ):
        violations.append("the benchmark source tree fingerprint contains no files")
    checkout_commit = provenance.get("helion_checkout_git_commit")
    if (
        not isinstance(checkout_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", checkout_commit) is None
    ):
        violations.append("the benchmark source tree has no Git revision")
    if provenance.get("helion_source_tree_dirty") is not False:
        violations.append("the measured source checkout is not clean")
    if provenance["effort"] != "full":
        violations.append("effective autotune effort is not full")
    if not provenance["effective_force_autotune"]:
        violations.append("cache reads are not bypassed with force autotune")
    if provenance["fixed_config"]:
        violations.append("a fixed Helion config was requested")
    if provenance["autotune_budget_seconds"] is not None:
        violations.append("an autotune wall-clock budget is set")
    if provenance["autotune_max_generations"] is not None:
        violations.append("an autotune generation cap is set")
    if provenance["autotune_best_of_k"] != 1:
        violations.append(
            "strict evidence requires exactly one full autotune trial because "
            "best-of-K final-rebenchmark provenance is not recorded"
        )
    if provenance["autotune_config_overrides"]:
        violations.append("autotune config fields are pinned")
    if provenance["user_seed_configs"]:
        violations.append("a user-provided autotune seed is set")
    if provenance["disable_autotuner_heuristics"]:
        violations.append("compiler autotuner heuristics are disabled")
    compiler_seed_policy = provenance.get("compiler_seed_policy")
    if not isinstance(compiler_seed_policy, dict) or set(compiler_seed_policy) != {
        "schema_version",
        "kind",
        "heuristic_names",
        "raw_config_count",
        "effective_config_ids",
        "effective_config_ids_sha256",
        "timeout_retry_repetitions",
    }:
        violations.append("the canonical CuTe flash compiler-seed policy is missing")
    else:
        raw_config_count = compiler_seed_policy.get("raw_config_count")
        effective_config_ids = compiler_seed_policy.get("effective_config_ids")
        valid_effective_ids = (
            isinstance(effective_config_ids, list)
            and bool(effective_config_ids)
            and all(
                isinstance(config_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
                for config_id in effective_config_ids
            )
            and len(effective_config_ids) == len(set(effective_config_ids))
        )
        effective_config_count = (
            len(effective_config_ids) if isinstance(effective_config_ids, list) else -1
        )
        expected_digest = (
            hashlib.sha256(
                json.dumps(effective_config_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if valid_effective_ids
            else None
        )
        if (
            type(compiler_seed_policy.get("schema_version")) is not int
            or compiler_seed_policy.get("schema_version") != 1
            or compiler_seed_policy.get("kind") != "canonical_cute_flash"
            or compiler_seed_policy.get("heuristic_names") != ["cute_flash_attention"]
            or isinstance(raw_config_count, bool)
            or not isinstance(raw_config_count, int)
            or raw_config_count < effective_config_count
            or provenance["compiler_seed_config_count"] != raw_config_count
            or not valid_effective_ids
            or compiler_seed_policy.get("effective_config_ids_sha256")
            != expected_digest
            or compiler_seed_policy.get("timeout_retry_repetitions") != 3
        ):
            violations.append("the CuTe flash compiler-seed policy is not canonical")
    if provenance["compiler_default_config"]:
        violations.append("a compiler-generated default config is present")
    if provenance["kernel_declared_config_count"]:
        violations.append("kernel-declared autotune configs are present")
    required_structural_fields = {
        "flash_structural_coverage_design_source",
        "flash_structural_coverage_active_values",
        "flash_structural_coverage_design",
        "flash_structural_coverage_design_count",
        "flash_structural_coverage_design_sha256",
        "flash_structural_coverage_uncovered_values",
        "flash_structural_coverage_underqualified_values",
        "flash_structural_leaf_catalog",
        "flash_pipeline_lane_catalog",
        "flash_clc_lane_catalog",
        "flash_clc_lane_catalog_sha256",
        "flash_structural_coverage_underqualified_leaves",
        "flash_structural_coverage_interaction_key_groups",
        "flash_structural_coverage_active_interactions",
        "flash_structural_coverage_uncovered_interactions",
        "flash_structural_qualification_values",
        "flash_structural_parent_coverage_prefix_count",
        "flash_structural_qualification_prefix_count",
        "flash_structural_population_budget",
        "flash_structural_injected_design_count",
        "flash_structural_qualification_rounds",
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round",
        "flash_structural_family_probe_generations",
        "flash_structural_family_probe_candidates_per_path",
        "flash_structural_retained_candidates_per_leaf",
        "flash_structural_retained_family_cap",
        "flash_structural_retained_family_limit",
        "flash_structural_retained_family_slowdown_limit",
        "flash_structural_starting_path_limit",
        "flash_structural_family_probe_path_limit",
        "flash_structural_maximum_path_capacity",
        "flash_structural_unrestricted_path_exhausts_generation_budget",
        "flash_terminal_coordinate_refinement_policy",
        "flash_terminal_coordinate_refinement_policy_sha256",
        "flash_terminal_coordinate_surface_catalog",
        "flash_terminal_coordinate_surface_catalog_sha256",
    }
    missing_structural_fields = sorted(required_structural_fields - provenance.keys())
    if missing_structural_fields:
        violations.append(
            "the current structural provenance schema is incomplete: "
            + ", ".join(missing_structural_fields)
        )
    terminal_policy = provenance.get("flash_terminal_coordinate_refinement_policy")
    expected_terminal_policy = _flash_terminal_coordinate_refinement_policy(
        get_effort_profile("full")
    )
    if terminal_policy != expected_terminal_policy or provenance.get(
        "flash_terminal_coordinate_refinement_policy_sha256"
    ) != _canonical_json_sha256(expected_terminal_policy):
        violations.append("the terminal coordinate refinement policy is inconsistent")
    terminal_surface_catalog = provenance.get(
        "flash_terminal_coordinate_surface_catalog"
    )
    terminal_surface_error = _flash_terminal_surface_catalog_error(
        terminal_surface_catalog
    )
    if terminal_surface_error is not None:
        violations.append(terminal_surface_error)
    elif provenance.get("flash_terminal_coordinate_surface_catalog_sha256") != (
        _canonical_json_sha256(terminal_surface_catalog)
    ):
        violations.append(
            "the terminal coordinate surface catalog digest is inconsistent"
        )
    else:
        typed_terminal_surface = cast("dict[str, object]", terminal_surface_catalog)
        surface_leaves = [
            leaf_record["leaf"]
            for leaf_record in cast(
                "list[dict[str, object]]", typed_terminal_surface["leaves"]
            )
        ]
        if surface_leaves != provenance.get("flash_structural_leaf_catalog"):
            violations.append(
                "the terminal coordinate surface catalog does not cover every live leaf"
            )
    clc_lane_catalog = provenance.get("flash_clc_lane_catalog")
    if not isinstance(clc_lane_catalog, list) or any(
        not isinstance(record, dict)
        or set(record)
        != {
            "family",
            "compound_packet",
            "softmax_disc",
            "legal_values",
            "search_values",
            "anchor_values",
            "refinement_values",
            "planned_values",
            "witness_config_ids",
        }
        for record in clc_lane_catalog or ()
    ):
        violations.append("the deterministic CLC lane catalog is malformed")
    elif any(
        record["search_values"] != record["legal_values"] for record in clc_lane_catalog
    ):
        violations.append("the CLC lane catalog does not expose every legal value")
    elif (
        provenance.get("flash_clc_lane_catalog_sha256")
        != hashlib.sha256(
            json.dumps(clc_lane_catalog, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        violations.append("the deterministic CLC lane catalog digest is inconsistent")
    coverage_design = _structural_coverage_design(provenance)
    if not coverage_design:
        violations.append(
            "the CuTe flash search surface has no structural coverage design"
        )
    if provenance.get("flash_structural_coverage_uncovered_values") != []:
        violations.append(
            "the CuTe flash structural design does not cover every active value"
        )
    if provenance.get("flash_structural_coverage_underqualified_values") != []:
        violations.append(
            "the CuTe flash structural design has a family or compound packet "
            "with fewer than two witnesses"
        )
    if provenance.get("flash_structural_coverage_underqualified_leaves") != []:
        violations.append(
            "the CuTe flash structural design has an exact nonsingleton leaf "
            "with fewer than two witnesses"
        )
    if provenance.get("flash_structural_coverage_uncovered_interactions") != []:
        violations.append(
            "the CuTe flash structural design does not cover every declared "
            "field interaction"
        )
    active_values = cast(
        "list[dict[str, object]]",
        provenance.get("flash_structural_coverage_active_values", []),
    )
    if not active_values:
        violations.append("the CuTe flash search recorded no active structural values")
    elif any(
        not any(
            cast("dict[str, object]", candidate["config"]).get(
                cast("str", active["key"])
            )
            == active["value"]
            for candidate in coverage_design
        )
        for active in active_values
    ):
        violations.append(
            "the recorded structural design does not cover its active-value manifest"
        )
    if "flash_structural_coverage_design_count" in provenance:
        if provenance["flash_structural_coverage_design_count"] != len(coverage_design):
            violations.append("the structural coverage design count is inconsistent")
        parent_prefix_count = provenance.get(
            "flash_structural_parent_coverage_prefix_count"
        )
        qualification_prefix_count = provenance.get(
            "flash_structural_qualification_prefix_count"
        )
        initial_population_size = provenance.get("autotune_initial_population_size")
        structural_population_budget = provenance.get(
            "flash_structural_population_budget"
        )
        structural_injected_design_count = provenance.get(
            "flash_structural_injected_design_count"
        )
        if (
            isinstance(parent_prefix_count, bool)
            or not isinstance(parent_prefix_count, int)
            or parent_prefix_count < 0
            or isinstance(qualification_prefix_count, bool)
            or not isinstance(qualification_prefix_count, int)
            or qualification_prefix_count < parent_prefix_count
            or qualification_prefix_count > len(coverage_design)
        ):
            violations.append("the structural qualification prefix is inconsistent")
        if (
            isinstance(initial_population_size, bool)
            or not isinstance(initial_population_size, int)
            or initial_population_size <= 0
        ):
            violations.append("the full-search initial population size is invalid")
        if (
            isinstance(structural_population_budget, bool)
            or not isinstance(structural_population_budget, int)
            or structural_population_budget < 0
        ):
            violations.append("the structural population budget is inconsistent")
        elif (
            isinstance(initial_population_size, int)
            and not isinstance(initial_population_size, bool)
            and structural_population_budget > initial_population_size
        ):
            violations.append(
                "the structural population budget exceeds the full initial population"
            )
        if (
            isinstance(structural_injected_design_count, bool)
            or not isinstance(structural_injected_design_count, int)
            or not isinstance(structural_population_budget, int)
            or isinstance(structural_population_budget, bool)
            or structural_injected_design_count
            != min(structural_population_budget, len(coverage_design))
        ):
            violations.append("the injected structural design count is inconsistent")
        coverage_configs = [
            cast("dict[str, object]", candidate["config"])
            for candidate in coverage_design
        ]
        leaf_catalog = provenance.get("flash_structural_leaf_catalog")
        design_leaves = [
            leaf
            for config in coverage_configs
            if (leaf := _flash_structural_leaf_dict(config)) is not None
        ]
        if (
            not isinstance(leaf_catalog, list)
            or not leaf_catalog
            or any(
                not isinstance(leaf, dict)
                or set(leaf) != {"family", "compound_packet", "softmax_disc"}
                or not isinstance(leaf.get("family"), str)
                or not isinstance(leaf.get("softmax_disc"), bool)
                or (
                    leaf.get("compound_packet") is not None
                    and not isinstance(leaf.get("compound_packet"), str)
                )
                for leaf in leaf_catalog
            )
        ):
            violations.append("the exact structural leaf catalog is malformed")
            leaf_catalog = []
        else:
            catalog_keys = [
                json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                for leaf in leaf_catalog
            ]
            design_leaf_keys = {
                json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                for leaf in design_leaves
            }
            compound_owners = [
                (
                    cast("str", cast("dict[str, object]", leaf)["family"]),
                    cast("str", cast("dict[str, object]", leaf)["compound_packet"]),
                    cast("bool", cast("dict[str, object]", leaf)["softmax_disc"]),
                )
                for leaf in leaf_catalog
                if cast("dict[str, object]", leaf)["compound_packet"] is not None
            ]
            if (
                len(catalog_keys) != len(set(catalog_keys))
                or set(catalog_keys) != design_leaf_keys
                or len(compound_owners) != len(set(compound_owners))
            ):
                violations.append(
                    "the exact structural leaf catalog is inconsistent with its "
                    "family/packet owners"
                )

        full_effort_profile = get_effort_profile("full")
        structural_policy = full_effort_profile.flash_structural_search
        full_lfbo_policy = full_effort_profile.lfbo_pattern_search
        assert structural_policy is not None
        assert full_lfbo_policy is not None
        ordinary_widths: dict[str, int] = {}
        compound_count = 0
        for leaf in cast("list[dict[str, object]]", leaf_catalog):
            if leaf["compound_packet"] is None:
                family = cast("str", leaf["family"])
                ordinary_widths[family] = ordinary_widths.get(family, 0) + 1
            else:
                compound_count += 1
        retained_family_cap = structural_policy.retained_families
        retained_family_limit = (
            len(ordinary_widths)
            if retained_family_cap is None
            else min(retained_family_cap, len(ordinary_widths))
        )
        promoted_count = min(retained_family_limit, len(ordinary_widths))
        promoted_protocol_count = sum(
            sorted(ordinary_widths.values(), reverse=True)[:promoted_count]
        )
        starting_path_limit = max(
            full_lfbo_policy.copies,
            structural_policy.starting_paths,
            1
            + promoted_protocol_count
            + (
                promoted_count
                if structural_policy.retained_candidates_per_leaf > 1
                else 0
            )
            + compound_count,
        )
        family_probe_path_limit = (
            1 + len(ordinary_widths) + compound_count
            if structural_policy.family_probe_generations > 0
            and retained_family_cap is not None
            and len(ordinary_widths) > retained_family_cap
            else 0
        )
        expected_policy = {
            "flash_structural_qualification_rounds": (
                structural_policy.qualification_rounds
            ),
            "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": (
                structural_policy.pipeline_candidates_per_leaf_per_round
            ),
            "flash_structural_family_probe_generations": (
                structural_policy.family_probe_generations
            ),
            "flash_structural_family_probe_candidates_per_path": (
                structural_policy.family_probe_candidates_per_path
            ),
            "flash_structural_retained_candidates_per_leaf": (
                structural_policy.retained_candidates_per_leaf
            ),
            "flash_structural_retained_family_cap": retained_family_cap,
            "flash_structural_retained_family_limit": retained_family_limit,
            "flash_structural_retained_family_slowdown_limit": (
                structural_policy.retained_family_slowdown_limit
            ),
            "flash_structural_starting_path_limit": starting_path_limit,
            "flash_structural_family_probe_path_limit": family_probe_path_limit,
            "flash_structural_maximum_path_capacity": max(
                starting_path_limit, family_probe_path_limit
            ),
            "flash_structural_unrestricted_path_exhausts_generation_budget": (
                structural_policy.exhaust_unrestricted_path
            ),
            "autotune_lfbo_max_generations": full_lfbo_policy.max_generations,
        }
        if any(provenance.get(key) != value for key, value in expected_policy.items()):
            violations.append(
                "the exact structural qualification policy is inconsistent"
            )
        if (
            structural_policy.exhaust_unrestricted_path is not True
            or provenance.get(
                "flash_structural_unrestricted_path_exhausts_generation_budget"
            )
            is not True
        ):
            violations.append(
                "the unrestricted full-search path does not exhaust its generation "
                "budget"
            )
        active_interactions = provenance.get(
            "flash_structural_coverage_active_interactions"
        )
        declared_interaction_groups = [
            list(group) for group in FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
        ]
        if (
            provenance.get("flash_structural_coverage_interaction_key_groups")
            != declared_interaction_groups
        ):
            violations.append(
                "the structural interaction-group manifest is inconsistent"
            )
        if not isinstance(active_interactions, list):
            violations.append("the structural interaction manifest is missing")
            active_interactions = []
        elif any(
            not isinstance(interaction, dict)
            or not isinstance(interaction.get("keys"), list)
            or not isinstance(interaction.get("values"), list)
            or len(cast("list[object]", interaction["keys"]))
            != len(cast("list[object]", interaction["values"]))
            or not cast("list[object]", interaction["keys"])
            for interaction in active_interactions
        ):
            violations.append("the structural interaction manifest is malformed")
            active_interactions = []
        elif any(
            cast("list[object]", interaction["keys"]) not in declared_interaction_groups
            for interaction in active_interactions
        ):
            violations.append(
                "the structural interaction manifest contains an undeclared group"
            )
            active_interactions = []
        elif len(
            {
                json.dumps(interaction, sort_keys=True, separators=(",", ":"))
                for interaction in active_interactions
            }
        ) != len(active_interactions):
            violations.append("the structural interaction manifest contains duplicates")
            active_interactions = []
        active_value_keys = {
            cast("str", active["key"])
            for active in active_values
            if isinstance(active, dict) and isinstance(active.get("key"), str)
        }
        expected_interactions = {
            json.dumps(
                {
                    "keys": list(group),
                    "values": [config.get(key) for key in group],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for group in FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS
            if any(key in active_value_keys for key in group)
            for config in coverage_configs
        }
        recorded_interactions = {
            json.dumps(interaction, sort_keys=True, separators=(",", ":"))
            for interaction in active_interactions
        }
        if recorded_interactions != expected_interactions:
            violations.append(
                "the structural interaction manifest is incomplete for the coverage design"
            )
        if active_interactions and any(
            not any(
                all(
                    config.get(cast("str", key)) == value
                    for key, value in zip(
                        cast("list[object]", interaction["keys"]),
                        cast("list[object]", interaction["values"]),
                        strict=True,
                    )
                )
                for config in coverage_configs
            )
            for interaction in cast("list[dict[str, object]]", active_interactions)
        ):
            violations.append(
                "the recorded structural design omits a declared interaction"
            )
        if (
            isinstance(structural_injected_design_count, int)
            and not isinstance(structural_injected_design_count, bool)
            and 0 <= structural_injected_design_count <= len(coverage_configs)
            and any(
                not any(
                    config.get(cast("str", active["key"])) == active["value"]
                    for config in coverage_configs[:structural_injected_design_count]
                )
                for active in active_values
            )
        ):
            violations.append(
                "the injected structural design does not cover every active value"
            )
        if (
            active_interactions
            and isinstance(structural_injected_design_count, int)
            and not isinstance(structural_injected_design_count, bool)
            and 0 <= structural_injected_design_count <= len(coverage_configs)
            and any(
                not any(
                    all(
                        config.get(cast("str", key)) == value
                        for key, value in zip(
                            cast("list[object]", interaction["keys"]),
                            cast("list[object]", interaction["values"]),
                            strict=True,
                        )
                    )
                    for config in coverage_configs[:structural_injected_design_count]
                )
                for interaction in cast("list[dict[str, object]]", active_interactions)
            )
        ):
            violations.append(
                "the injected structural design omits a declared interaction"
            )
        if (
            isinstance(parent_prefix_count, int)
            and not isinstance(parent_prefix_count, bool)
            and isinstance(qualification_prefix_count, int)
            and not isinstance(qualification_prefix_count, bool)
            and 0 <= parent_prefix_count <= qualification_prefix_count
            and qualification_prefix_count <= len(coverage_configs)
        ):
            from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
            from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
            from helion._compiler.cute.cute_flash import flash_exp2_packet_is_compound

            parent_goals = [
                (cast("str", active["key"]), active["value"])
                for active in active_values
                if active["key"] in (FLASH_PIPELINE_FAMILY_KEY, FLASH_EXP2_PACKET_KEY)
            ]
            qualification_goals = [
                (key, value)
                for key, value in parent_goals
                if key == FLASH_PIPELINE_FAMILY_KEY
                or flash_exp2_packet_is_compound(value)
            ]
            recorded_qualification_values = provenance.get(
                "flash_structural_qualification_values"
            )
            expected_qualification_values = [
                {"key": key, "value": value} for key, value in qualification_goals
            ]
            if recorded_qualification_values != expected_qualification_values:
                violations.append(
                    "the structural qualification manifest is inconsistent"
                )
            if any(
                sum(
                    config.get(key) == value
                    for config in coverage_configs[:parent_prefix_count]
                )
                < 1
                for key, value in parent_goals
            ):
                violations.append(
                    "the structural parent prefix does not cover every family and packet"
                )
            if any(
                sum(
                    config.get(key) == value
                    for config in coverage_configs[:qualification_prefix_count]
                )
                < (2 if key == FLASH_PIPELINE_FAMILY_KEY else 1)
                for key, value in qualification_goals
            ):
                violations.append(
                    "the structural qualification prefix does not contain the "
                    "required family/compound-packet witnesses"
                )
            if (
                isinstance(leaf_catalog, list)
                and leaf_catalog
                and any(
                    sum(
                        _flash_structural_leaf_dict(config) == leaf
                        for config in coverage_configs[:qualification_prefix_count]
                    )
                    < min(
                        2,
                        sum(
                            _flash_structural_leaf_dict(config) == leaf
                            for config in coverage_configs
                        ),
                    )
                    for leaf in leaf_catalog
                )
            ):
                violations.append(
                    "the structural qualification prefix does not contain the "
                    "required exact-leaf witnesses"
                )
        coverage_json = json.dumps(
            coverage_configs, sort_keys=True, separators=(",", ":")
        )
        expected_coverage_hash = hashlib.sha256(
            coverage_json.encode("utf-8")
        ).hexdigest()
        if (
            provenance.get("flash_structural_coverage_design_sha256")
            != expected_coverage_hash
        ):
            violations.append("the structural coverage design digest is inconsistent")
        if any(
            candidate.get("config_sha256")
            != hashlib.sha256(
                json.dumps(
                    candidate["config"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            for candidate in coverage_design
        ):
            violations.append(
                "a structural coverage design entry digest is inconsistent"
            )
    if provenance.get("active_value_prior_keys"):
        violations.append("autotuner value priors are active")
    if provenance.get("flash_value_prior_keys"):
        violations.append("CuTe flash-specific value priors are active")
    if provenance.get("dense_d64_2cta_performance_anchor_present"):
        violations.append("a legacy CuTe flash performance anchor is active")
    if provenance["cute_flash_env_overrides"]:
        violations.append("CuTe flash codegen environment overrides are active")
    if provenance["autotune_initial_population_strategy_override"] not in (
        None,
        "from_random",
    ):
        violations.append("kernel settings override the initial population")
    if provenance["autotuner_initial_population_env"] != "from_random":
        violations.append("initial population is not forced to random")
    if provenance["autotuner_env"]:
        violations.append("a custom HELION_AUTOTUNER is set")
    if provenance["autotune_num_neighbors_cap_env"] != "-1":
        violations.append("LFBO neighborhood search is capped")
    if not provenance["autotuner_fn_is_default"]:
        violations.append("the kernel uses a custom autotuner function")
    if not provenance["autotune_baseline_fn_is_expected"]:
        violations.append("the kernel does not use its expected attention baseline")
    if provenance["autotune_baseline_atol"] != 5e-2:
        violations.append("the attention baseline atol is not standard")
    if provenance["autotune_baseline_rtol"] != 2e-2:
        violations.append("the attention baseline rtol is not standard")
    if provenance["autotune_baseline_accuracy_check_fn"]:
        violations.append("a custom candidate accuracy checker is enabled")
    if provenance["autotune_benchmark_fn"]:
        violations.append("a custom autotune benchmark function is enabled")
    if provenance["autotune_rebenchmark_threshold"] is not None:
        violations.append("the standard rebenchmark threshold is overridden")
    if provenance["autotune_suspicious_rebenchmark_ratio"] is not None:
        violations.append("the suspicious-rebenchmark ratio is overridden")
    if not provenance["autotune_accuracy_check"]:
        violations.append("candidate accuracy checking is disabled")
    if cast("int", provenance["autotune_compile_timeout"]) < 60:
        violations.append("candidate compile timeout is below the standard default")
    if not provenance["autotune_benchmark_subprocess"]:
        violations.append("isolated autotune benchmark subprocesses are disabled")
    if cast("int", provenance["autotune_benchmark_timeout"]) < 30:
        violations.append("candidate benchmark timeout is below the standard default")
    if not provenance["autotune_adaptive_timeout"]:
        violations.append("adaptive candidate compile timeouts are disabled")
    if provenance["autotune_force_persistent"]:
        violations.append("the search is globally restricted to persistent schedules")
    if provenance["autotune_finishing_rounds_env"]:
        violations.append("the LFBO finishing-round count is overridden")
    if provenance["autotune_ignore_errors"]:
        violations.append("autotune errors are ignored")
    if provenance["autotune_search_acf"]:
        violations.append("kernel-specific PTXAS ACF search is enabled")
    if provenance["autotune_config_filter"]:
        violations.append("a kernel-level autotune config filter is enabled")
    if not provenance["final_correctness_enabled"]:
        violations.append("final output correctness checking is disabled")
    if provenance["autotune_cache"] != "LocalAutotuneCache":
        violations.append("the standard local autotune cache wrapper is not active")
    if provenance["rebenchmark_env_overrides"]:
        violations.append("rebenchmark environment overrides are active")
    if violations:
        raise SystemExit(
            "--helion-require-full-autotune rejected this run: " + "; ".join(violations)
        )


_FlashPipelineLane = tuple[str, int]
_FlashQualifiedMember = tuple[str, float, frozenset[_FlashPipelineLane]]
_FlashLeafQualification = tuple[
    str,
    str | None,
    bool,
    list[_FlashQualifiedMember],
    tuple[_FlashPipelineLane, ...],
]


def _flash_qualified_member_from_measurement_state(
    config_id: str,
    state: dict[str, object] | None,
) -> _FlashQualifiedMember | None:
    """Return a live retention member from the final measurement state."""
    if state is None or state.get("status") not in {"ok", "deduplicated"}:
        return None
    selection_perf = state.get("selection_perf")
    if (
        isinstance(selection_perf, bool)
        or not isinstance(selection_perf, (int, float))
        or not math.isfinite(selection_perf)
        or selection_perf <= 0
    ):
        return None
    return config_id, float(selection_perf), frozenset()


def _flash_pipeline_lane_metric(
    lane: _FlashPipelineLane | None,
) -> dict[str, object] | None:
    if lane is None:
        return None
    return {"key": lane[0], "value": lane[1]}


def _expected_flash_clc_depth_candidate_ids(
    successful_config_ids: set[str],
    manifest_leaves: dict[str, dict[str, object]],
    leaf: dict[str, object],
    pre_combination_ids: set[str],
) -> set[str]:
    """Recover the successful candidates visible before CLC composition."""
    return {
        config_id
        for config_id in successful_config_ids & pre_combination_ids
        if manifest_leaves[config_id] == leaf
    }


def _expected_flash_lane_diverse_members(
    members: list[_FlashQualifiedMember],
    lanes: tuple[_FlashPipelineLane, ...],
    *,
    limit: int,
    pipeline_qualification_keys: tuple[str, ...],
) -> list[tuple[_FlashQualifiedMember, _FlashPipelineLane | None]]:
    """Mirror the autotuner's deterministic marginal-lane retention."""
    remaining = sorted(members, key=operator.itemgetter(1, 0))
    if limit <= 0 or not remaining:
        return []
    selected: list[tuple[_FlashQualifiedMember, _FlashPipelineLane | None]] = [
        (remaining.pop(0), None)
    ]
    covered = set(selected[0][0][2]) & set(lanes)
    while remaining and len(selected) < limit:

        def rank(member: _FlashQualifiedMember) -> tuple[object, ...]:
            newly_covered = (set(member[2]) & set(lanes)) - covered
            coverage_by_key = tuple(
                -sum(lane[0] == key for lane in newly_covered)
                for key in pipeline_qualification_keys
            )
            return (*coverage_by_key, member[1], member[0])

        member = min(remaining, key=rank)
        remaining.remove(member)
        newly_covered = [
            lane for lane in lanes if lane not in covered and lane in member[2]
        ]
        selected.append((member, newly_covered[0] if newly_covered else None))
        covered.update(set(member[2]) & set(lanes))
    return selected


def _expected_flash_structural_retention(
    qualified_by_leaf: list[_FlashLeafQualification],
    *,
    retained_per_leaf: int,
    retained_family_cap: int | None,
    retained_family_limit: int,
    retained_family_slowdown_limit: float,
    starting_path_limit: int,
    pipeline_qualification_keys: tuple[str, ...],
    parent_score_config_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Reconstruct lane-diverse retention and deterministic path promotion."""
    family_leaves: dict[
        str,
        list[
            tuple[
                str | None,
                bool,
                list[_FlashQualifiedMember],
                tuple[_FlashPipelineLane, ...],
            ]
        ],
    ] = {}
    retained_by_leaf: dict[
        tuple[str, str | None, bool],
        list[tuple[_FlashQualifiedMember, _FlashPipelineLane | None]],
    ] = {}
    for family, compound_packet, softmax_disc, members, lanes in qualified_by_leaf:
        if not members:
            continue
        sorted_members = sorted(members, key=operator.itemgetter(1, 0))
        family_leaves.setdefault(family, []).append(
            (compound_packet, softmax_disc, sorted_members, lanes)
        )
        leaf_key = (family, compound_packet, softmax_disc)
        retained_by_leaf[leaf_key] = _expected_flash_lane_diverse_members(
            sorted_members,
            lanes,
            limit=retained_per_leaf,
            pipeline_qualification_keys=pipeline_qualification_keys,
        )

    family_queues: dict[
        str,
        list[tuple[str | None, bool, str, float, _FlashPipelineLane | None]],
    ] = {}
    family_scores: dict[str, tuple[str | None, bool, str, float]] = {}
    parent_score_families: set[str] = set()
    for family, leaves in family_leaves.items():
        queue: list[tuple[str | None, bool, str, float, _FlashPipelineLane | None]] = []
        for rank in range(retained_per_leaf):
            layer = [
                (
                    packet,
                    softmax_disc,
                    retained[rank][0][0],
                    retained[rank][0][1],
                    retained[rank][1],
                )
                for packet, softmax_disc, _members, _lanes in leaves
                for retained in [retained_by_leaf[(family, packet, softmax_disc)]]
                if rank < len(retained)
            ]
            queue.extend(
                sorted(
                    layer,
                    key=lambda item: (item[3], item[0] or "", item[1], item[2]),
                )
            )
        family_queues[family] = queue
        ordinary = []
        for packet, softmax_disc, members, _lanes in leaves:
            if packet is not None:
                continue
            score_member = next(
                (
                    member
                    for member in members
                    if parent_score_config_ids is None
                    or member[0] in parent_score_config_ids
                ),
                None,
            )
            if score_member is not None:
                ordinary.append(
                    (packet, softmax_disc, score_member[0], score_member[1])
                )
        if ordinary:
            parent_score_families.add(family)
        family_scores[family] = min(
            ordinary or [queue[0][:4]], key=operator.itemgetter(3, 2)
        )

    ranked_families = sorted(
        parent_score_families,
        key=lambda family: (family_scores[family][3], family),
    )
    if not family_queues:
        return []
    competitive_families: list[str] = []
    if retained_family_cap is None:
        competitive_families = ranked_families
    elif ranked_families:
        best_family_perf = family_scores[ranked_families[0]][3]
        competitive_families = [
            family
            for family in ranked_families
            if family_scores[family][3]
            <= best_family_perf * retained_family_slowdown_limit
        ]
    best_family, best_packet, best_softmax_disc, best_config_id, _best_perf = min(
        (
            (family, packet, softmax_disc, members[0][0], members[0][1])
            for family, leaves in family_leaves.items()
            for packet, softmax_disc, members, _lanes in leaves
        ),
        key=lambda item: (item[4], item[0], item[1] or "", item[2], item[3]),
    )
    best_leaf = (best_family, best_packet, best_softmax_disc)
    family_score_leaf = (
        best_family,
        family_scores[best_family][0],
        family_scores[best_family][1],
    )
    alternate_leaf_order = [best_leaf]
    if family_score_leaf != best_leaf:
        alternate_leaf_order.append(family_score_leaf)
    alternate_leaf_order.extend(
        leaf
        for leaf in sorted(
            (
                (best_family, packet, softmax_disc)
                for packet, softmax_disc, _members, _lanes in family_leaves[best_family]
                if (best_family, packet, softmax_disc) not in alternate_leaf_order
            ),
            key=lambda leaf: (
                retained_by_leaf[leaf][0][0][1],
                retained_by_leaf[leaf][0][0][0],
            ),
        )
    )
    best_lane_alternate = next(
        (
            (
                family,
                packet,
                softmax_disc,
                member[0],
                member[1],
                lane,
            )
            for family, packet, softmax_disc in alternate_leaf_order
            for member, lane in retained_by_leaf[(family, packet, softmax_disc)]
            if member[0] != best_config_id and lane is not None
        ),
        None,
    )
    constrained_limit = max(0, starting_path_limit - 1)
    retained_parent_families = competitive_families[
        : min(retained_family_limit, constrained_limit)
    ]
    selected: list[
        tuple[str, str | None, bool, str, bool, _FlashPipelineLane | None]
    ] = []
    for family in retained_parent_families:
        packet, softmax_disc, config_id, _perf = family_scores[family]
        if len(selected) >= constrained_limit:
            break
        selected.append((family, packet, softmax_disc, config_id, False, None))
    selected_ids = {
        config_id
        for _family, _packet, _softmax, config_id, _unrestricted, _lane in selected
    }
    selected_leaves = {
        (family, packet, softmax_disc)
        for family, packet, softmax_disc, _config_id, _unrestricted, _lane in selected
    }
    ordinary_leaf_candidates = sorted(
        (
            (family, packet, softmax_disc, members[0][0], members[0][1])
            for family in retained_parent_families
            for packet, softmax_disc, members, _lanes in family_leaves[family]
            if packet is None and (family, packet, softmax_disc) not in selected_leaves
        ),
        key=operator.itemgetter(4, 0, 2, 3),
    )
    for family, packet, softmax_disc, config_id, _perf in ordinary_leaf_candidates:
        if len(selected) >= constrained_limit:
            break
        if config_id in selected_ids:
            continue
        selected.append((family, packet, softmax_disc, config_id, False, None))
        selected_ids.add(config_id)
        selected_leaves.add((family, packet, softmax_disc))

    if (
        best_lane_alternate is not None
        and best_family in retained_parent_families
        and len(selected) < constrained_limit
    ):
        family, packet, softmax_disc, config_id, _perf, lane = best_lane_alternate
        if config_id not in selected_ids:
            selected.append((family, packet, softmax_disc, config_id, False, lane))
            selected_ids.add(config_id)
            selected_leaves.add((family, packet, softmax_disc))

    primary_config_ids = {
        (family, packet, softmax_disc): members[0][0]
        for family, leaves in family_leaves.items()
        for packet, softmax_disc, members, _lanes in leaves
    }
    families_with_ordinary_secondary = {
        family
        for family, packet, softmax_disc, config_id, _unrestricted, _lane in selected
        if packet is None
        and config_id != primary_config_ids[(family, packet, softmax_disc)]
    }
    for family in retained_parent_families:
        if (
            len(selected) >= constrained_limit
            or family in families_with_ordinary_secondary
        ):
            continue
        secondary = next(
            (
                (packet, softmax_disc, config_id, perf, lane)
                for packet, softmax_disc, config_id, perf, lane in family_queues[family]
                if packet is None
                and config_id != primary_config_ids[(family, packet, softmax_disc)]
                and config_id not in selected_ids
            ),
            None,
        )
        if secondary is None:
            continue
        packet, softmax_disc, config_id, _perf, lane = secondary
        selected.append((family, packet, softmax_disc, config_id, False, lane))
        selected_ids.add(config_id)
        selected_leaves.add((family, packet, softmax_disc))
        families_with_ordinary_secondary.add(family)

    compound_leaf_candidates = sorted(
        (
            (family, packet, softmax_disc, members[0][0], members[0][1])
            for family, leaves in family_leaves.items()
            for packet, softmax_disc, members, _lanes in leaves
            if packet is not None
            and (family, packet, softmax_disc) not in selected_leaves
        ),
        key=lambda item: (
            item[4],
            item[0],
            item[1] or "",
            item[2],
            item[3],
        ),
    )
    for family, packet, softmax_disc, config_id, _perf in compound_leaf_candidates:
        if len(selected) >= constrained_limit:
            break
        if config_id in selected_ids:
            continue
        selected.append((family, packet, softmax_disc, config_id, False, None))
        selected_ids.add(config_id)
        selected_leaves.add((family, packet, softmax_disc))

    offsets = dict.fromkeys(retained_parent_families, 0)
    while len(selected) < constrained_limit:
        added = False
        for family in retained_parent_families:
            queue = family_queues[family]
            offset = offsets[family]
            while offset < len(queue) and queue[offset][2] in selected_ids:
                offset += 1
            offsets[family] = offset
            if offset >= len(queue):
                continue
            compound_packet, softmax_disc, config_id, _perf, lane = queue[offset]
            offsets[family] += 1
            selected.append(
                (family, compound_packet, softmax_disc, config_id, False, lane)
            )
            selected_ids.add(config_id)
            added = True
            if len(selected) >= constrained_limit:
                break
        if not added:
            break

    if starting_path_limit > 0:
        selected.append(
            (best_family, best_packet, best_softmax_disc, best_config_id, True, None)
        )

    reported_families = list(
        dict.fromkeys(
            (
                best_family,
                *retained_parent_families,
                *(family for family, *_rest in selected),
            )
        )
    )
    return [
        {
            "family": family,
            "score": family_scores[family][3],
            "score_compound_packet": family_scores[family][0],
            "score_softmax_disc": family_scores[family][1],
            "parent_promoted": family in retained_parent_families,
            "starting_paths": [
                {
                    "family": selected_family,
                    "compound_packet": compound_packet,
                    "softmax_disc": softmax_disc,
                    "config_id": config_id,
                    "unrestricted": unrestricted,
                    "pipeline_lane": _flash_pipeline_lane_metric(lane),
                }
                for (
                    selected_family,
                    compound_packet,
                    softmax_disc,
                    config_id,
                    unrestricted,
                    lane,
                ) in selected
                if selected_family == family
            ],
        }
        for family in reported_families
    ]


def _flash_repair_passes_are_compact(
    decision_passes: list[int], *, per_pass_limit: int
) -> bool:
    """Return whether ordered repair decisions occupy consecutive full batches."""
    if not decision_passes:
        return True
    if per_pass_limit <= 0:
        return False
    first_pass = decision_passes[0]
    return decision_passes == [
        first_pass + decision_index // per_pass_limit
        for decision_index in range(len(decision_passes))
    ]


def _independent_flash_schedule_anchor_configs(
    config_generation: ConfigGeneration,
) -> list[_ConfigLike]:
    """Rebuild live schedule anchors without using the producer helper."""
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion._compiler.cute.cute_flash import flash_exp2_packet_is_compound
    from helion.autotuner.config_fragment import EnumFragment
    from helion.exc import InvalidConfig

    if not config_generation.config_spec.cute_flash_search_enabled:
        return []

    keys = (
        FLASH_PIPELINE_FAMILY_KEY,
        FLASH_EXP2_PACKET_KEY,
        FLASH_SOFTMAX_DISC_KEY,
    )
    axes: dict[str, tuple[int, tuple[object, ...]]] = {}
    for key in keys:
        layout = config_generation._key_to_flat_indices.get(key)
        if layout is None or layout[1] or len(layout[0]) != 1:
            return []
        index = layout[0][0]
        fragment = config_generation.flat_spec[index]
        if not isinstance(fragment, EnumFragment):
            return []
        choices = (
            (config_generation._override_values[key],)
            if key in config_generation._override_values
            else fragment.choices
            if fragment.search_choices is None
            else fragment.search_choices
        )
        axes[key] = (index, tuple(choices))

    base = [fragment.default() for fragment in config_generation.flat_spec]
    config_generation._shrink_for_numel_constraints(base)
    config_generation._repair_cute_num_threads(base)
    result: list[_ConfigLike] = []
    seen_configs: set[object] = set()
    for family, packet, softmax_disc in product(
        axes[FLASH_PIPELINE_FAMILY_KEY][1],
        axes[FLASH_EXP2_PACKET_KEY][1],
        axes[FLASH_SOFTMAX_DISC_KEY][1],
    ):
        if flash_exp2_packet_is_compound(packet):
            continue
        flat = copy.deepcopy(base)
        flat[axes[FLASH_PIPELINE_FAMILY_KEY][0]] = family
        flat[axes[FLASH_EXP2_PACKET_KEY][0]] = packet
        flat[axes[FLASH_SOFTMAX_DISC_KEY][0]] = softmax_disc
        try:
            config = config_generation.unflatten(flat)
        except InvalidConfig:
            continue
        values = config.config
        if (
            values.get(FLASH_PIPELINE_FAMILY_KEY) != family
            or values.get(FLASH_EXP2_PACKET_KEY) != packet
            or values.get(FLASH_SOFTMAX_DISC_KEY) != softmax_disc
            or config in seen_configs
        ):
            continue
        seen_configs.add(config)
        result.append(config)
    return result


def _strict_flash_schedule_anchor_configs(
    config_generation: ConfigGeneration,
    *,
    trial_index: int,
) -> list[_ConfigLike]:
    """Return independently rebuilt anchors after auditing producer output."""
    independent = _independent_flash_schedule_anchor_configs(config_generation)
    produced = config_generation.flash_low_confound_schedule_anchor_configs()
    independent_configs = [dict(config.config) for config in independent]
    produced_configs = [dict(config.config) for config in produced]
    if produced_configs != independent_configs:
        raise RuntimeError(
            f"required full autotune trial {trial_index} recorded a CuTe flash "
            "schedule-anchor producer result that differs from independent live "
            "fragment enumeration"
        )
    return independent


def _flash_pre_probe_pass_index(
    qualification_pass_count: int,
    family_probe_generations: int,
    family_probe_required: bool,
) -> int:
    return qualification_pass_count - (
        family_probe_generations if family_probe_required else 0
    )


def _measurement_snapshot_matches(
    record: dict[str, object],
    measurement_states_by_pass: list[dict[str, dict[str, object]]],
    *,
    config_id: str,
    expected_pass_index: int | None = None,
) -> bool:
    pass_index = record.get("measurement_pass_index")
    snapshot = {
        key: record.get(key)
        for key in ("attempt_perf", "selection_perf", "status", "source_hash")
    }
    return bool(
        type(pass_index) is int
        and 0 <= pass_index < len(measurement_states_by_pass)
        and (expected_pass_index is None or pass_index == expected_pass_index)
        and measurement_states_by_pass[pass_index].get(config_id) == snapshot
    )


def _validate_flash_structural_qualification_phase(
    provenance: dict[str, object],
    phase: dict[str, object],
    *,
    trial_index: int,
    expected_initial_config_ids: list[str],
    expected_initial_population_count: int,
    config_generation: ConfigGeneration,
) -> tuple[set[str], set[str]]:
    """Validate qualification and return timeout sources and final invalid configs."""
    from helion.autotuner.effort_profile import get_effort_profile
    from helion.autotuner.surrogate_pattern_search import (
        flash_terminal_measurement_is_valid,
    )
    from helion.exc import InvalidConfig

    def fail(detail: str) -> NoReturn:
        raise RuntimeError(
            f"required full autotune trial {trial_index} recorded {detail}"
        )

    def valid_config_id(value: object) -> bool:
        return (
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None
        )

    def valid_source_hash(value: object) -> bool:
        return (
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        )

    def valid_perf(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
        )

    def config_ids(value: object, detail: str) -> list[str]:
        if (
            not isinstance(value, list)
            or any(not valid_config_id(config_id) for config_id in value)
            or len(value) != len(set(value))
        ):
            fail(detail)
        return cast("list[str]", value)

    def canonical_config_id(config: dict[str, object]) -> str:
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    if (
        phase.get("phase") != "cute_flash_structural_qualification_v22"
        or phase.get("cute_flash_lane_policy_version")
        != _CUTE_FLASH_LANE_POLICY_VERSION
        or phase.get("completed") is not True
        or phase.get("budget_exhausted") is not False
    ):
        fail("an incomplete or non-v22 structural qualification phase")
    if (
        phase.get("conditional_candidates_per_pipeline_lane") != 1
        or phase.get("qualification_failure_retries") != 1
        or phase.get("qualification_passes_planned")
        != phase.get("qualification_passes_started")
        or phase.get("qualification_passes_planned")
        != phase.get("qualification_passes_completed")
        or phase.get("qualification_rounds_started")
        != phase.get("qualification_passes_completed")
        or phase.get("qualification_rounds_completed")
        != phase.get("qualification_passes_completed")
    ):
        fail("inconsistent v22 qualification pass accounting")
    rounds = provenance.get("flash_structural_qualification_rounds")
    candidate_limit = provenance.get(
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round"
    )
    family_probe_generations = provenance.get(
        "flash_structural_family_probe_generations"
    )
    family_probe_candidates_per_path = provenance.get(
        "flash_structural_family_probe_candidates_per_path"
    )
    retained_per_leaf = provenance.get("flash_structural_retained_candidates_per_leaf")
    retained_family_cap = provenance.get("flash_structural_retained_family_cap")
    retained_family_limit = provenance.get("flash_structural_retained_family_limit")
    retained_family_slowdown_limit = provenance.get(
        "flash_structural_retained_family_slowdown_limit"
    )
    starting_path_limit = provenance.get("flash_structural_starting_path_limit")
    family_probe_path_limit = provenance.get("flash_structural_family_probe_path_limit")
    maximum_path_capacity = provenance.get("flash_structural_maximum_path_capacity")
    if (
        type(rounds) is not int
        or rounds <= 0
        or type(candidate_limit) is not int
        or candidate_limit <= 0
        or type(family_probe_generations) is not int
        or family_probe_generations < 0
        or type(family_probe_candidates_per_path) is not int
        or family_probe_candidates_per_path <= 1
        or type(retained_per_leaf) is not int
        or retained_per_leaf <= 0
        or (
            retained_family_cap is not None
            and (type(retained_family_cap) is not int or retained_family_cap <= 0)
        )
        or type(retained_family_limit) is not int
        or retained_family_limit <= 0
        or not valid_perf(retained_family_slowdown_limit)
        or cast("float", retained_family_slowdown_limit) < 1.0
        or type(starting_path_limit) is not int
        or starting_path_limit <= 0
        or type(family_probe_path_limit) is not int
        or family_probe_path_limit < 0
        or type(maximum_path_capacity) is not int
        or maximum_path_capacity <= 0
        or phase.get("qualification_rounds") != rounds
        or phase.get("pipeline_candidate_limit_per_leaf_per_round") != candidate_limit
        or phase.get("family_probe_generations") != family_probe_generations
        or phase.get("family_probe_candidates_per_path")
        != family_probe_candidates_per_path
        or phase.get("retained_candidates_per_leaf") != retained_per_leaf
        or phase.get("retained_family_cap") != retained_family_cap
        or phase.get("retained_family_limit") != retained_family_limit
        or phase.get("retained_family_slowdown_limit") != retained_family_slowdown_limit
        or phase.get("starting_path_limit") != starting_path_limit
        or phase.get("family_probe_path_limit") != family_probe_path_limit
        or phase.get("maximum_path_capacity") != maximum_path_capacity
        or phase.get("pipeline_qualification_keys")
        != ["cute_flash_kv_stage", "cute_flash_s_stage"]
        or phase.get("neighbor_generation_limit_per_leaf_per_round") != 200
        or provenance.get(
            "flash_structural_unrestricted_path_exhausts_generation_budget"
        )
        is not True
        or phase.get("unrestricted_path_exhausts_generation_budget") is not True
    ):
        fail("inconsistent exact structural qualification bounds")
    qualification_neighbor_limit = cast(
        "int", phase["neighbor_generation_limit_per_leaf_per_round"]
    )
    if (
        phase.get("initial_config_count") != expected_initial_population_count
        or not isinstance(phase.get("initial_config_ids"), list)
        or len(cast("list[object]", phase["initial_config_ids"]))
        != expected_initial_population_count
        or any(
            not valid_config_id(config_id)
            for config_id in cast("list[object]", phase["initial_config_ids"])
        )
        or len(set(cast("list[str]", phase["initial_config_ids"])))
        != expected_initial_population_count
        or expected_initial_config_ids != cast("list[str]", phase["initial_config_ids"])
    ):
        fail("inconsistent v22 initial population")

    provenance_exact_ids = provenance.get(
        "flash_exact_effective_search_space_config_ids"
    )
    exact_enumerated = provenance_exact_ids is not None
    if exact_enumerated:
        provenance_exact_ids = config_ids(
            provenance_exact_ids, "an invalid exact-space provenance list"
        )
    phase_exact_ids = config_ids(
        phase.get("exact_space_config_ids"), "an invalid phase exact-space list"
    )
    initial_id_set = set(cast("list[str]", phase["initial_config_ids"]))
    exact_ids_measured = (
        exact_enumerated
        and set(cast("list[str]", provenance_exact_ids)) <= initial_id_set
    )
    reported_exact_space_exhausted = phase.get("exact_space_exhausted")
    expected_exact_raw_budget = max(
        1,
        cast("int", provenance["autotune_initial_population_size"]),
        expected_initial_population_count,
    )
    if (
        phase.get("exact_space_enumerated") is not exact_enumerated
        or phase_exact_ids != (provenance_exact_ids or [])
        or not isinstance(reported_exact_space_exhausted, bool)
        or (reported_exact_space_exhausted and not exact_ids_measured)
        or phase.get("exact_space_raw_budget") != expected_exact_raw_budget
    ):
        fail("inconsistent exact-space enumeration or exhaustion evidence")

    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    leaf_results = phase.get("leaf_results")
    compound_transfers = phase.get("compound_transfers")
    if not isinstance(leaf_catalog, list) or not leaf_catalog:
        fail("no exact structural leaf catalog")
    if not isinstance(leaf_results, list) or not isinstance(compound_transfers, list):
        fail("missing v22 leaf or compound qualification records")
    if (
        phase.get("compound_catalog_complete") is not True
        or phase.get("compound_catalog_errors") != []
    ):
        fail("an incomplete compound structural catalog")
    catalog_keys = [
        json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        for leaf in leaf_catalog
        if isinstance(leaf, dict)
        and set(leaf) == {"family", "compound_packet", "softmax_disc"}
        and isinstance(leaf.get("family"), str)
        and isinstance(leaf.get("softmax_disc"), bool)
        and (
            leaf.get("compound_packet") is None
            or isinstance(leaf.get("compound_packet"), str)
        )
    ]
    if len(catalog_keys) != len(leaf_catalog) or len(catalog_keys) != len(
        set(catalog_keys)
    ):
        fail("an invalid exact structural leaf catalog")
    pipeline_lane_catalog = provenance.get("flash_pipeline_lane_catalog")
    if not isinstance(pipeline_lane_catalog, list) or len(pipeline_lane_catalog) != len(
        leaf_catalog
    ):
        fail("no deterministic pipeline lane catalog")
    lanes_by_leaf: dict[str, list[tuple[str, int]]] = {}
    for leaf, entry in zip(leaf_catalog, pipeline_lane_catalog, strict=True):
        if not isinstance(entry, dict):
            fail("an invalid deterministic pipeline lane catalog")
        lanes = entry.get("pipeline_lanes")
        if (
            set(entry)
            != {"family", "compound_packet", "softmax_disc", "pipeline_lanes"}
            or {
                "family": entry.get("family"),
                "compound_packet": entry.get("compound_packet"),
                "softmax_disc": entry.get("softmax_disc"),
            }
            != leaf
            or not isinstance(lanes, list)
        ):
            fail("an invalid deterministic pipeline lane catalog")
        lane_pairs: list[tuple[str, int]] = []
        for lane in lanes:
            if (
                not isinstance(lane, dict)
                or set(lane) != {"key", "value"}
                or lane.get("key") not in {"cute_flash_kv_stage", "cute_flash_s_stage"}
                or type(lane.get("value")) is not int
                or cast("int", lane["value"]) <= 0
            ):
                fail("an invalid deterministic pipeline lane catalog")
            lane_pairs.append((cast("str", lane["key"]), cast("int", lane["value"])))
        if len(lane_pairs) != len(set(lane_pairs)):
            fail("an invalid deterministic pipeline lane catalog")
        if cast("dict[str, object]", leaf).get("compound_packet") is not None and lanes:
            fail("an invalid deterministic pipeline lane catalog")
        lanes_by_leaf[json.dumps(leaf, sort_keys=True, separators=(",", ":"))] = (
            lane_pairs
        )
    ordinary_catalog = [
        leaf
        for leaf in leaf_catalog
        if isinstance(leaf, dict) and leaf.get("compound_packet") is None
    ]
    compound_catalog = [
        leaf
        for leaf in leaf_catalog
        if isinstance(leaf, dict) and leaf.get("compound_packet") is not None
    ]
    ordinary_widths: dict[str, int] = {}
    for leaf in ordinary_catalog:
        family = cast("str", leaf["family"])
        ordinary_widths[family] = ordinary_widths.get(family, 0) + 1
    expected_retained_family_limit = (
        len(ordinary_widths)
        if retained_family_cap is None
        else min(cast("int", retained_family_cap), len(ordinary_widths))
    )
    if retained_family_limit != expected_retained_family_limit:
        fail("a retained family limit inconsistent with its configured cap")
    promoted_count = min(cast("int", retained_family_limit), len(ordinary_widths))
    promoted_protocol_count = sum(
        sorted(ordinary_widths.values(), reverse=True)[:promoted_count]
    )
    full_profile = get_effort_profile("full")
    assert full_profile.lfbo_pattern_search is not None
    assert full_profile.flash_structural_search is not None
    expected_starting_path_limit = max(
        full_profile.lfbo_pattern_search.copies,
        full_profile.flash_structural_search.starting_paths,
        1
        + promoted_protocol_count
        + (promoted_count if cast("int", retained_per_leaf) > 1 else 0)
        + len(compound_catalog),
    )
    if starting_path_limit != expected_starting_path_limit:
        fail("a non-live-derived structural starting-path limit")
    expected_family_probe_path_limit = (
        1 + len(ordinary_widths) + len(compound_catalog)
        if cast("int", family_probe_generations) > 0
        and retained_family_cap is not None
        and len(ordinary_widths) > cast("int", retained_family_cap)
        else 0
    )
    if (
        family_probe_path_limit != expected_family_probe_path_limit
        or maximum_path_capacity
        != max(expected_starting_path_limit, expected_family_probe_path_limit)
    ):
        fail("a non-live-derived structural probe-path capacity")
    expected_family_probe_required = bool(
        expected_family_probe_path_limit and not reported_exact_space_exhausted
    )
    family_probe_paths = phase.get("family_probe_paths")
    if (
        phase.get("family_probe_required") is not expected_family_probe_required
        or phase.get("family_probe_complete") is not True
        or phase.get("family_probe_generations_started")
        != (family_probe_generations if expected_family_probe_required else 0)
        or phase.get("family_probe_generations_completed")
        != (family_probe_generations if expected_family_probe_required else 0)
        or not isinstance(family_probe_paths, list)
        or len(family_probe_paths)
        != (expected_family_probe_path_limit if expected_family_probe_required else 0)
    ):
        fail("inconsistent structural family-probe accounting")
    initial_results = phase.get("initial_results")
    if not isinstance(initial_results, list):
        fail("missing v22 generation-zero measurement records")
    initial_record_by_id: dict[str, dict[str, object]] = {}
    initial_ids_by_leaf: dict[str, list[str]] = {
        json.dumps(leaf, sort_keys=True, separators=(",", ":")): []
        for leaf in leaf_catalog
    }
    initial_lane_ids: dict[tuple[str, _FlashPipelineLane], list[str]] = {}
    successful_statuses = {"ok", "deduplicated"}
    known_statuses = successful_statuses | {
        "error",
        "timeout",
        "peer_compilation_fail",
        "accuracy_error",
        "source_rejected",
        "filtered",
    }
    config_manifest = phase.get("config_manifest")
    if not isinstance(config_manifest, dict):
        fail("missing v22 canonical config manifest")
    manifest_entries: dict[str, dict[str, object]] = {}
    manifest_configs: dict[str, dict[str, object]] = {}
    manifest_leaves: dict[str, dict[str, object]] = {}
    for raw_config_id, raw_entry in config_manifest.items():
        if (
            not valid_config_id(raw_config_id)
            or not isinstance(raw_entry, dict)
            or set(raw_entry) != {"config"}
            or not isinstance(raw_entry.get("config"), dict)
        ):
            fail("a malformed v22 canonical config manifest")
        config_id = cast("str", raw_config_id)
        entry = cast("dict[str, object]", raw_entry)
        config = cast("dict[str, object]", entry["config"])
        try:
            normalized = _canonical_flash_projection(config_generation, config, {})
        except InvalidConfig:
            fail("an invalid config in the v22 canonical manifest")
        leaf = _flash_structural_leaf_dict(config)
        if (
            canonical_config_id(config) != config_id
            or normalized != config
            or leaf not in leaf_catalog
        ):
            fail("an inconsistent v22 canonical config manifest")
        manifest_entries[config_id] = entry
        manifest_configs[config_id] = config
        manifest_leaves[config_id] = cast("dict[str, object]", leaf)

    measurement_timeline = phase.get("measurement_timeline")
    qualification_pass_count = phase.get("qualification_passes_completed")
    if (
        not isinstance(measurement_timeline, list)
        or type(qualification_pass_count) is not int
        or len(measurement_timeline) != qualification_pass_count + 1
    ):
        fail("a malformed v22 measurement timeline")
    pre_probe_pass = _flash_pre_probe_pass_index(
        cast("int", qualification_pass_count),
        cast("int", family_probe_generations),
        expected_family_probe_required,
    )
    if pre_probe_pass < 0:
        fail("a malformed v22 pre-probe measurement boundary")
    measurement_states_by_pass: list[dict[str, dict[str, object]]] = []
    current_measurement_states: dict[str, dict[str, object]] = {}
    retryable_failure_statuses = {"error", "timeout", "peer_compilation_fail"}
    isolated_timeout_source_hashes: set[str] = set()
    isolated_invalidated_source_hashes: set[str] = set()
    for expected_pass_index, raw_pass in enumerate(measurement_timeline):
        if (
            not isinstance(raw_pass, dict)
            or set(raw_pass) != {"pass_index", "updates"}
            or raw_pass.get("pass_index") != expected_pass_index
            or not isinstance(raw_pass.get("updates"), list)
        ):
            fail("a malformed v22 measurement timeline")
        updates = cast("list[object]", raw_pass["updates"])
        update_ids: list[str] = []
        next_states = dict(current_measurement_states)
        introduced_success_hashes: set[str] = set()
        source_repair_hashes: list[str] = []
        pass_invalidated_source_hashes: set[str] = set()
        for raw_update in updates:
            if (
                not isinstance(raw_update, dict)
                or set(raw_update)
                != {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                }
                or not valid_config_id(raw_update.get("config_id"))
                or raw_update.get("status") not in known_statuses
            ):
                fail("a malformed v22 measurement timeline update")
            update = cast("dict[str, object]", raw_update)
            config_id = cast("str", update["config_id"])
            succeeded = update["status"] in successful_statuses
            state = {
                key: update[key]
                for key in (
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                )
            }
            if (
                config_id not in manifest_entries
                or config_id in update_ids
                or (
                    state["source_hash"] is not None
                    and not valid_source_hash(state["source_hash"])
                )
                or (succeeded and not valid_source_hash(state["source_hash"]))
                or succeeded
                != (
                    valid_perf(state["attempt_perf"])
                    and valid_perf(state["selection_perf"])
                )
                or (
                    not succeeded
                    and (
                        state["attempt_perf"] is not None
                        or state["selection_perf"] is not None
                    )
                )
                or current_measurement_states.get(config_id) == state
            ):
                fail("an invalid v22 measurement timeline update")
            previous = current_measurement_states.get(config_id)
            if previous is not None:
                previous_succeeded = previous["status"] in successful_statuses
                rebenchmarked = bool(
                    previous_succeeded
                    and succeeded
                    and state["status"] == previous["status"]
                    and state["attempt_perf"] == previous["attempt_perf"]
                    and state["source_hash"] == previous["source_hash"]
                )
                source_repaired = bool(
                    previous["status"] in retryable_failure_statuses
                    and state["status"] == "deduplicated"
                    and succeeded
                    and valid_source_hash(previous["source_hash"])
                    and state["source_hash"] == previous["source_hash"]
                )
                isolated_rebenchmark_invalidated = bool(
                    previous_succeeded
                    and state["status"] in {"error", "timeout"}
                    and state["attempt_perf"] is None
                    and state["selection_perf"] is None
                    and state["source_hash"] == previous["source_hash"]
                )
                if not (
                    rebenchmarked or source_repaired or isolated_rebenchmark_invalidated
                ):
                    fail("an invalid v22 measurement state transition")
                if source_repaired:
                    source_repair_hashes.append(cast("str", state["source_hash"]))
                if isolated_rebenchmark_invalidated:
                    source_hash = cast("str", state["source_hash"])
                    pass_invalidated_source_hashes.add(source_hash)
                    isolated_invalidated_source_hashes.add(source_hash)
                    if state["status"] == "timeout":
                        isolated_timeout_source_hashes.add(source_hash)
            elif succeeded:
                introduced_success_hashes.add(cast("str", state["source_hash"]))
            update_ids.append(config_id)
            next_states[config_id] = state
        if not set(source_repair_hashes) <= introduced_success_hashes:
            fail("an unproven v22 effective-source repair")
        if any(
            state["status"] in successful_statuses
            and state["source_hash"] in pass_invalidated_source_hashes
            for state in next_states.values()
        ):
            fail("an incomplete v22 effective-source invalidation")
        if update_ids != sorted(update_ids):
            fail("a reordered v22 measurement timeline update")
        if expected_pass_index == 0 and set(update_ids) != initial_id_set:
            fail("the v22 measurement timeline omits the initial population")
        current_measurement_states = next_states
        measurement_states_by_pass.append(current_measurement_states)

    decision_statuses = known_statuses | {"unknown"}

    def validate_measurement_snapshot(
        record: dict[str, object],
        detail: str,
        *,
        config_id: str,
        allow_unmeasured: bool = False,
        expected_pass_index: int | None = None,
    ) -> None:
        pass_index = record.get("measurement_pass_index")
        snapshot = {
            key: record.get(key)
            for key in ("attempt_perf", "selection_perf", "status", "source_hash")
        }
        if pass_index is None:
            if not (
                allow_unmeasured
                and snapshot
                == {
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "unknown",
                    "source_hash": None,
                }
                and (
                    expected_pass_index is None
                    or config_id not in measurement_states_by_pass[expected_pass_index]
                )
            ):
                fail(detail)
            return
        if not _measurement_snapshot_matches(
            record,
            measurement_states_by_pass,
            config_id=config_id,
            expected_pass_index=expected_pass_index,
        ):
            fail(detail)

    def decision_results(
        value: object,
        detail: str,
        *,
        allow_unmeasured: bool = False,
        allow_failed: bool = False,
        require_ranked: bool = True,
        expected_pass_index: int | None = None,
    ) -> list[dict[str, object]]:
        if not isinstance(value, list):
            fail(detail)
        parsed: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for item in value:
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }
                or not valid_config_id(item.get("config_id"))
                or item.get("status") not in decision_statuses
                or (
                    item.get("source_hash") is not None
                    and not valid_source_hash(item.get("source_hash"))
                )
            ):
                fail(detail)
            record = cast("dict[str, object]", item)
            config_id = cast("str", record["config_id"])
            succeeded = record["status"] in successful_statuses
            if (
                config_id in seen_ids
                or config_id not in manifest_entries
                or succeeded
                != (
                    valid_perf(record["attempt_perf"])
                    and valid_perf(record["selection_perf"])
                )
                or (
                    not succeeded
                    and (
                        record["attempt_perf"] is not None
                        or record["selection_perf"] is not None
                    )
                )
                or (record["status"] == "unknown" and not allow_unmeasured)
                or (succeeded and not valid_source_hash(record["source_hash"]))
                or (
                    record["status"] not in successful_statuses
                    and record["status"] != "unknown"
                    and not allow_failed
                )
            ):
                fail(detail)
            validate_measurement_snapshot(
                record,
                detail,
                config_id=config_id,
                allow_unmeasured=allow_unmeasured,
                expected_pass_index=expected_pass_index,
            )
            seen_ids.add(config_id)
            parsed.append(record)
        measured_passes = {
            record["measurement_pass_index"]
            for record in parsed
            if record["measurement_pass_index"] is not None
        }
        if len(measured_passes) > 1:
            fail(detail)
        if require_ranked and parsed != sorted(
            parsed,
            key=lambda record: (
                (
                    cast("float", record["selection_perf"])
                    if record["status"] in successful_statuses
                    else math.inf
                ),
                cast("str", record["config_id"]),
            ),
        ):
            fail(detail)
        return parsed

    def repair_id_mapping(
        value: object,
        allowed_values: list[int],
        detail: str,
    ) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            fail(detail)
        parsed: dict[str, list[str]] = {}
        expected_order = [str(item) for item in allowed_values]
        for raw_key, raw_ids in value.items():
            if raw_key not in expected_order:
                fail(detail)
            ids = config_ids(raw_ids, detail)
            if len(ids) > cast("int", phase["qualification_failure_retries"]):
                fail(detail)
            parsed[cast("str", raw_key)] = ids
        if list(parsed) != [key for key in expected_order if key in parsed]:
            fail(detail)
        flat_ids = [config_id for ids in parsed.values() for config_id in ids]
        if len(flat_ids) != len(set(flat_ids)):
            fail(detail)
        return parsed

    def add_snapshot_pass(
        passes: set[int], snapshot: dict[str, object], detail: str
    ) -> None:
        pass_index = snapshot.get("measurement_pass_index")
        if type(pass_index) is not int:
            fail(detail)
        passes.add(cast("int", pass_index))

    expected_anchor_configs = _strict_flash_schedule_anchor_configs(
        config_generation,
        trial_index=trial_index,
    )
    expected_anchor_ids = [
        canonical_config_id(dict(config.config)) for config in expected_anchor_configs
    ]
    expected_new_anchor_ids = set(expected_anchor_ids) - initial_id_set
    schedule_anchor_pass_count = int(bool(expected_new_anchor_ids))
    schedule_anchor_results = phase.get("schedule_anchor_results")
    if (
        phase.get("schedule_anchor_design_source")
        != "live family x ordinary packet x softmax protocol from fragment defaults"
        or phase.get("schedule_anchor_pass_planned")
        is not bool(expected_new_anchor_ids)
        or phase.get("schedule_anchor_pass_started")
        is not bool(expected_new_anchor_ids)
        or phase.get("schedule_anchor_count") != len(expected_anchor_ids)
        or phase.get("schedule_anchor_complete") is not True
        or not isinstance(schedule_anchor_results, list)
        or len(schedule_anchor_results) != len(expected_anchor_ids)
    ):
        fail("an incomplete or fabricated low-confound schedule-anchor design")
    for expected_config, expected_id, raw_result in zip(
        expected_anchor_configs,
        expected_anchor_ids,
        cast("list[object]", schedule_anchor_results),
        strict=True,
    ):
        if (
            not isinstance(raw_result, dict)
            or set(raw_result)
            != {
                "config_id",
                "family",
                "compound_packet",
                "softmax_disc",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
            }
            or raw_result.get("config_id") != expected_id
            or manifest_configs.get(expected_id) != dict(expected_config.config)
            or {
                "family": raw_result.get("family"),
                "compound_packet": raw_result.get("compound_packet"),
                "softmax_disc": raw_result.get("softmax_disc"),
            }
            != manifest_leaves.get(expected_id)
            or not flash_terminal_measurement_is_valid(raw_result)
        ):
            fail("an incomplete or fabricated low-confound schedule-anchor result")
        validate_measurement_snapshot(
            cast("dict[str, object]", raw_result),
            "an inconsistent low-confound schedule-anchor measurement snapshot",
            config_id=expected_id,
            expected_pass_index=schedule_anchor_pass_count,
        )

    for record in initial_results:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "config_id",
                "family",
                "compound_packet",
                "softmax_disc",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
                "pipeline_lanes",
            }
            or not valid_config_id(record.get("config_id"))
            or record.get("status") not in known_statuses
            or not isinstance(record.get("pipeline_lanes"), list)
        ):
            fail("a malformed v22 generation-zero measurement record")
        config_id = cast("str", record["config_id"])
        leaf = {
            "family": record["family"],
            "compound_packet": record["compound_packet"],
            "softmax_disc": record["softmax_disc"],
        }
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        entry = manifest_entries.get(config_id)
        config = manifest_configs.get(config_id)
        if (
            config_id in initial_record_by_id
            or leaf_key not in lanes_by_leaf
            or entry is None
            or config is None
            or manifest_leaves[config_id] != leaf
        ):
            fail("a malformed v22 generation-zero measurement record")
        validate_measurement_snapshot(
            record,
            "an inconsistent generation-zero measurement snapshot",
            config_id=config_id,
            expected_pass_index=0,
        )
        succeeded = record["status"] in successful_statuses
        if succeeded != (
            valid_perf(record["attempt_perf"]) and valid_perf(record["selection_perf"])
        ) or (
            not succeeded
            and (
                record["attempt_perf"] is not None
                or record["selection_perf"] is not None
            )
        ):
            fail("an inconsistent generation-zero result status/performance pair")
        record_memberships = cast("list[object]", record["pipeline_lanes"])
        expected_lanes = lanes_by_leaf[leaf_key]
        parsed_memberships: list[_FlashPipelineLane] = []
        for membership in record_memberships:
            if (
                not isinstance(membership, dict)
                or set(membership) != {"key", "value"}
                or (membership.get("key"), membership.get("value"))
                not in expected_lanes
            ):
                fail("an invalid generation-zero pipeline lane membership")
            parsed_memberships.append(
                (cast("str", membership["key"]), cast("int", membership["value"]))
            )
        if parsed_memberships != [
            lane for lane in expected_lanes if lane in parsed_memberships
        ] or parsed_memberships != [
            lane for lane in expected_lanes if config.get(lane[0]) == lane[1]
        ]:
            fail("an invalid generation-zero pipeline lane membership")
        initial_record_by_id[config_id] = record
        initial_ids_by_leaf[leaf_key].append(config_id)
        for lane in parsed_memberships:
            initial_lane_ids.setdefault((leaf_key, lane), []).append(config_id)
    if [record["config_id"] for record in initial_results] != cast(
        "list[str]", phase["initial_config_ids"]
    ):
        fail("generation-zero records do not partition the initial population")
    active_values = provenance.get("flash_structural_coverage_active_values")
    if not isinstance(active_values, list):
        fail("no structural active-value manifest")
    active_pipeline_lanes = {
        (active.get("key"), active.get("value"))
        for active in active_values
        if isinstance(active, dict)
        and active.get("key") in {"cute_flash_kv_stage", "cute_flash_s_stage"}
    }
    if any(
        lane not in active_pipeline_lanes
        for leaf in ordinary_catalog
        for lane in lanes_by_leaf[
            json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        ]
    ):
        fail("a normalized pipeline lane absent from the active-value manifest")
    if (
        phase.get("leaf_count") != len(leaf_catalog)
        or phase.get("ordinary_leaf_count") != len(ordinary_catalog)
        or phase.get("compound_leaf_count") != len(compound_catalog)
        or len(leaf_results) != len(ordinary_catalog)
        or len(compound_transfers) != len(compound_catalog)
    ):
        fail("inconsistent v22 ordinary/compound leaf counts")

    for leaf, result in zip(ordinary_catalog, leaf_results, strict=True):
        if (
            not isinstance(result, dict)
            or {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
                "softmax_disc": result.get("softmax_disc"),
            }
            != leaf
            or not isinstance(result.get("pipeline_lanes"), list)
            or not isinstance(result.get("qualified_results"), list)
            or not isinstance(result.get("retained_config_ids"), list)
            or result.get("complete") is not True
        ):
            fail("a malformed v22 ordinary leaf result")
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        if result.get("initial_config_ids") != initial_ids_by_leaf[leaf_key]:
            fail("an ordinary leaf omits a generation-zero measurement")
        for lane in cast("list[object]", result["pipeline_lanes"]):
            if (
                not isinstance(lane, dict)
                or lane.get("witness_attempted") is not True
                or not valid_config_id(lane.get("witness_config_id"))
                or not isinstance(lane.get("witness_succeeded"), bool)
                or lane.get("complete") is not True
                or not isinstance(lane.get("space_exhausted"), bool)
                or (
                    lane.get("space_config_count") is not None
                    and type(lane.get("space_config_count")) is not int
                )
                or not isinstance(lane.get("conditional_required"), bool)
                or lane.get("conditional_required") is lane.get("space_exhausted")
                or (lane.get("space_exhausted") is True and not exact_enumerated)
                or not isinstance(lane.get("conditional_candidate_ids"), list)
                or len(cast("list[object]", lane["conditional_candidate_ids"]))
                != (1 if lane.get("conditional_required") else 0)
                or any(
                    not valid_config_id(config_id)
                    for config_id in cast(
                        "list[object]", lane["conditional_candidate_ids"]
                    )
                )
                or not isinstance(
                    lane.get("successful_conditional_candidate_ids"), list
                )
                or not set(
                    cast("list[object]", lane["successful_conditional_candidate_ids"])
                )
                <= set(cast("list[object]", lane["conditional_candidate_ids"]))
                or not isinstance(lane.get("repair_candidate_ids"), list)
                or len(cast("list[object]", lane["repair_candidate_ids"])) > 1
                or any(
                    not valid_config_id(config_id)
                    for config_id in cast("list[object]", lane["repair_candidate_ids"])
                )
                or not isinstance(lane.get("successful_repair_candidate_ids"), list)
                or not set(
                    cast("list[object]", lane["successful_repair_candidate_ids"])
                )
                <= set(cast("list[object]", lane["repair_candidate_ids"]))
                or not isinstance(lane.get("repair_parent_decisions"), list)
                or len(cast("list[object]", lane["repair_parent_decisions"])) > 1
                or not isinstance(lane.get("terminal_failure_exhausted"), bool)
                or not (
                    lane["witness_succeeded"]
                    or lane["successful_conditional_candidate_ids"]
                    or lane["successful_repair_candidate_ids"]
                    or lane["terminal_failure_exhausted"]
                )
                or bool(lane["terminal_failure_exhausted"])
                is bool(
                    lane["witness_succeeded"]
                    or lane["successful_conditional_candidate_ids"]
                    or lane["successful_repair_candidate_ids"]
                )
            ):
                fail("an incomplete v22 pipeline lane")
            lane_key = (cast("str", lane["key"]), cast("int", lane["value"]))
            if lane.get("initial_config_ids") != initial_lane_ids.get(
                (leaf_key, lane_key), []
            ):
                fail("a pipeline lane omits a generation-zero measurement")

    clc_families = phase.get("clc_families")
    if not isinstance(clc_families, list):
        fail("missing v22 CLC family records")
    recorded_clc_catalog = provenance.get("flash_clc_lane_catalog")
    expected_clc_catalog = _flash_clc_lane_provenance(
        config_generation,
        leaf_catalog=cast("list[dict[str, object]]", leaf_catalog),
    )
    if recorded_clc_catalog != expected_clc_catalog or len(clc_families) != len(
        expected_clc_catalog
    ):
        fail("a seed-dependent or fabricated CLC lane catalog")
    clc_ids_by_leaf: dict[str, set[str]] = {}
    clc_witness_ids_by_leaf: dict[str, set[str]] = {}
    clc_records_by_leaf: dict[str, dict[str, object]] = {}
    clc_depth_ids_by_leaf: dict[str, list[str]] = {}
    clc_depth_candidate_ids_by_leaf: dict[str, list[str]] = {}
    clc_pre_combination_ids_by_leaf: dict[str, set[str]] = {}
    max_clc_planned = 0
    has_clc_conditional = False
    has_clc_combination = False
    clc_witness_repair_decision_passes: set[int] = set()
    clc_conditional_repair_decision_passes: set[int] = set()
    clc_witness_repair_first_passes: set[int] = set()
    clc_conditional_repair_first_passes: set[int] = set()
    clc_witness_snapshot_passes: set[int] = set()
    clc_conditional_parent_passes: set[int] = set()
    clc_retained_snapshot_passes: set[int] = set()
    clc_depth_snapshot_passes: set[int] = set()
    pipeline_generated_ids = {
        config_id
        for leaf_result in leaf_results
        if isinstance(leaf_result, dict)
        for round_result in (
            leaf_result.get("rounds")
            if isinstance(leaf_result.get("rounds"), list)
            else []
        )
        if isinstance(round_result, dict)
        for config_id in (
            round_result.get("candidate_config_ids")
            if isinstance(round_result.get("candidate_config_ids"), list)
            else []
        )
        if valid_config_id(config_id)
    }
    seen_clc_conditional_ids = (
        initial_id_set
        | set(expected_anchor_ids)
        | pipeline_generated_ids
        | {
            config_id
            for record in expected_clc_catalog
            for config_id in cast(
                "dict[str, str]", record["witness_config_ids"]
            ).values()
        }
    )
    for expected_clc, result in zip(expected_clc_catalog, clc_families, strict=True):
        if not isinstance(result, dict):
            fail("a malformed v22 CLC family record")
        if set(result) != {
            "family",
            "softmax_disc",
            "space_exhausted",
            "legal_values",
            "search_values",
            "anchor_values",
            "refinement_values",
            "planned_values",
            "attempted_values",
            "witness_config_ids",
            "witness_repair_candidate_ids",
            "witness_repair_parent_decisions",
            "value_space_exhausted",
            "witness_candidate_results",
            "witness_selection_results",
            "selected_values",
            "selected_config_ids",
            "conditional_values",
            "conditional_neighbor_generation_limit",
            "conditional_parent_decisions",
            "conditional_repair_candidate_ids",
            "conditional_repair_parent_decisions",
            "retained_values",
            "retained_config_ids",
            "retained_value_decisions",
            "retained_ranking_results",
            "conditional_candidate_ids",
            "combination_required",
            "depth_selection",
            "combination_candidate_ids",
            "combination_depth_config_ids",
            "combination_divisor_values",
            "combination_cells",
            "combination_projection_complete",
            "successful_combination_depth_config_ids",
            "successful_combination_divisor_values",
            "combination_row_coverage_complete",
            "combination_column_coverage_complete",
            "combination_failure_statuses_allowed",
            "complete",
        }:
            fail("a malformed v22 CLC family record")
        legal_values = result.get("legal_values")
        search_values = result.get("search_values")
        anchor_values = result.get("anchor_values")
        refinement_values = result.get("refinement_values")
        planned_values = result.get("planned_values")
        attempted_values = result.get("attempted_values")
        witness_ids = result.get("witness_config_ids")
        witness_repair_ids = result.get("witness_repair_candidate_ids")
        witness_repair_decisions = result.get("witness_repair_parent_decisions")
        value_space_exhausted = result.get("value_space_exhausted")
        witness_candidate_results = result.get("witness_candidate_results")
        witness_selection_results = result.get("witness_selection_results")
        selected_values = result.get("selected_values")
        selected_config_ids = result.get("selected_config_ids")
        conditional_values = result.get("conditional_values")
        conditional_neighbor_generation_limit = result.get(
            "conditional_neighbor_generation_limit"
        )
        conditional_parent_decisions = result.get("conditional_parent_decisions")
        conditional_repair_ids = result.get("conditional_repair_candidate_ids")
        conditional_repair_decisions = result.get("conditional_repair_parent_decisions")
        retained_values = result.get("retained_values")
        retained_config_ids = result.get("retained_config_ids")
        retained_value_decisions = result.get("retained_value_decisions")
        retained_ranking_results = result.get("retained_ranking_results")
        conditional_ids = result.get("conditional_candidate_ids")
        depth_selection = result.get("depth_selection")
        combination_ids = result.get("combination_candidate_ids")
        combination_depth_ids = result.get("combination_depth_config_ids")
        combination_divisor_values = result.get("combination_divisor_values")
        combination_cells = result.get("combination_cells")
        successful_combination_depth_ids = result.get(
            "successful_combination_depth_config_ids"
        )
        successful_combination_divisor_values = result.get(
            "successful_combination_divisor_values"
        )
        value_lists = (
            legal_values,
            search_values,
            anchor_values,
            refinement_values,
            planned_values,
            attempted_values,
            selected_values,
            conditional_values,
            retained_values,
        )
        if not all(isinstance(values, list) for values in value_lists):
            fail("a malformed v22 CLC family record")
        typed_legal = cast("list[int]", legal_values)
        typed_search = cast("list[int]", search_values)
        typed_anchors = cast("list[int]", anchor_values)
        typed_refinements = cast("list[int]", refinement_values)
        typed_planned = cast("list[int]", planned_values)
        typed_attempted = cast("list[int]", attempted_values)
        typed_selected_values = cast("list[int]", selected_values)
        typed_conditional_values = cast("list[int]", conditional_values)
        typed_retained_values = cast("list[int]", retained_values)
        if any(
            any(type(value) is not int or value <= 0 for value in values)
            or len(values) != len(set(values))
            for values in (
                typed_legal,
                typed_search,
                typed_anchors,
                typed_refinements,
                typed_planned,
                typed_attempted,
                typed_selected_values,
                typed_conditional_values,
                typed_retained_values,
            )
        ):
            fail("a malformed v22 CLC family record")
        if (
            not isinstance(witness_ids, dict)
            or not isinstance(witness_repair_ids, dict)
            or not isinstance(value_space_exhausted, dict)
            or not isinstance(conditional_ids, dict)
            or not isinstance(conditional_repair_ids, dict)
        ):
            fail("a malformed v22 CLC family record")
        typed_witness_ids = cast("dict[str, object]", witness_ids)
        typed_witness_repair_ids = cast("dict[str, object]", witness_repair_ids)
        typed_value_space_exhausted = cast("dict[str, object]", value_space_exhausted)
        typed_conditional_ids = cast("dict[str, object]", conditional_ids)
        typed_conditional_repair_ids = cast("dict[str, object]", conditional_repair_ids)
        typed_combination_ids = config_ids(
            combination_ids, "an incomplete v22 CLC family record"
        )
        typed_combination_depth_ids = config_ids(
            combination_depth_ids, "an incomplete v22 CLC family record"
        )
        typed_successful_combination_depth_ids = config_ids(
            successful_combination_depth_ids,
            "an incomplete v22 CLC family record",
        )
        typed_selected_config_ids = config_ids(
            selected_config_ids, "an incomplete v22 CLC family record"
        )
        typed_retained_config_ids = config_ids(
            retained_config_ids, "an incomplete v22 CLC family record"
        )
        if (
            not isinstance(combination_divisor_values, list)
            or not isinstance(successful_combination_divisor_values, list)
            or not isinstance(combination_cells, list)
        ):
            fail("incomplete CLC family record or depth/divisor coverage")
        typed_combination_divisor_values = cast("list[int]", combination_divisor_values)
        typed_successful_combination_divisor_values = cast(
            "list[int]", successful_combination_divisor_values
        )
        parsed_witness_repair_ids = repair_id_mapping(
            typed_witness_repair_ids,
            typed_planned,
            "an invalid immutable CLC witness repair",
        )
        parsed_conditional_repair_ids = repair_id_mapping(
            typed_conditional_repair_ids,
            typed_conditional_values,
            "an invalid immutable CLC conditional repair",
        )
        retained_count = len(typed_planned)
        if (
            result.get("complete") is not True
            or result.get("family") != expected_clc["family"]
            or result.get("softmax_disc") != expected_clc["softmax_disc"]
            or typed_legal != expected_clc["legal_values"]
            or typed_search != expected_clc["search_values"]
            or typed_search != typed_legal
            or typed_anchors != expected_clc["anchor_values"]
            or typed_refinements != expected_clc["refinement_values"]
            or typed_planned != expected_clc["planned_values"]
            or typed_attempted != typed_planned
            or typed_planned != [*typed_anchors, *typed_refinements]
            or set(typed_planned) != set(typed_legal)
            or not set(typed_anchors) <= set(typed_search)
            or not set(typed_refinements) <= set(typed_search)
            or bool(set(typed_anchors) & set(typed_refinements))
            or typed_witness_ids != expected_clc["witness_config_ids"]
            or len(typed_selected_values) != retained_count
            or len(typed_selected_config_ids) != retained_count
            or set(typed_selected_values) != set(typed_planned)
            or set(typed_value_space_exhausted)
            != {str(value) for value in typed_planned}
            or any(
                not isinstance(exhausted, bool)
                for exhausted in typed_value_space_exhausted.values()
            )
            or typed_conditional_values
            != [
                value
                for value in typed_selected_values
                if not typed_value_space_exhausted[str(value)]
            ]
            or len(typed_retained_values) != retained_count
            or len(typed_retained_config_ids) != retained_count
            or set(typed_retained_values) != set(typed_selected_values)
            or type(conditional_neighbor_generation_limit) is not int
            or conditional_neighbor_generation_limit
            != (
                max(qualification_neighbor_limit, len(typed_conditional_values))
                if typed_conditional_values
                else 0
            )
            or set(typed_witness_ids) != {str(value) for value in typed_planned}
            or list(typed_witness_ids) != [str(value) for value in typed_planned]
            or any(
                not valid_config_id(config_id)
                for config_id in typed_witness_ids.values()
            )
            or set(typed_conditional_ids)
            != {str(value) for value in typed_conditional_values}
            or list(typed_conditional_ids)
            != [str(value) for value in typed_conditional_values]
            or any(
                not isinstance(candidate_ids, list)
                or len(candidate_ids) > 1
                or any(not valid_config_id(config_id) for config_id in candidate_ids)
                or (not candidate_ids and not parsed_conditional_repair_ids.get(value))
                for value, candidate_ids in typed_conditional_ids.items()
            )
            or not isinstance(result.get("space_exhausted"), bool)
            or not isinstance(result.get("combination_required"), bool)
            or not isinstance(witness_candidate_results, list)
            or not isinstance(witness_selection_results, list)
            or not isinstance(conditional_parent_decisions, list)
            or not isinstance(witness_repair_decisions, list)
            or not isinstance(conditional_repair_decisions, list)
            or not isinstance(retained_value_decisions, list)
            or not isinstance(retained_ranking_results, list)
            or not isinstance(depth_selection, dict)
            or any(
                type(value) is not int or value <= 0
                for value in [
                    *typed_combination_divisor_values,
                    *typed_successful_combination_divisor_values,
                ]
            )
            or len(typed_combination_divisor_values)
            != len(set(typed_combination_divisor_values))
            or len(typed_successful_combination_divisor_values)
            != len(set(typed_successful_combination_divisor_values))
            or not set(typed_successful_combination_depth_ids)
            <= set(typed_combination_depth_ids)
            or not set(typed_successful_combination_divisor_values)
            <= set(typed_combination_divisor_values)
            or type(result.get("combination_projection_complete")) is not bool
            or type(result.get("combination_row_coverage_complete")) is not bool
            or type(result.get("combination_column_coverage_complete")) is not bool
            or type(result.get("combination_failure_statuses_allowed")) is not bool
            or result.get("combination_required") is result.get("space_exhausted")
            or (
                not (
                    0
                    < len(typed_combination_ids)
                    <= len(typed_combination_depth_ids)
                    * len(typed_combination_divisor_values)
                    and 0
                    < len(typed_combination_depth_ids)
                    <= cast("int", retained_per_leaf)
                    and typed_combination_divisor_values == typed_retained_values
                    and len(cast("list[object]", combination_cells))
                    == len(typed_combination_depth_ids)
                    * len(typed_combination_divisor_values)
                    and result.get("combination_projection_complete") is True
                    and result.get("combination_row_coverage_complete") is True
                    and result.get("combination_column_coverage_complete") is True
                )
                if result.get("combination_required")
                else bool(
                    typed_combination_ids
                    or typed_combination_depth_ids
                    or typed_combination_divisor_values
                    or cast("list[object]", combination_cells)
                    or typed_successful_combination_depth_ids
                    or typed_successful_combination_divisor_values
                    or result.get("combination_projection_complete") is not True
                    or result.get("combination_row_coverage_complete") is not True
                    or result.get("combination_column_coverage_complete") is not True
                )
            )
        ):
            fail("incomplete CLC family record or depth/divisor coverage")
        family = result.get("family")
        if not isinstance(family, str):
            fail("a malformed v22 CLC family record")
        expected_leaf: dict[str, object] = {
            "family": family,
            "compound_packet": None,
            "softmax_disc": result["softmax_disc"],
        }
        clc_key = json.dumps(expected_leaf, sort_keys=True, separators=(",", ":"))
        if expected_leaf not in ordinary_catalog or clc_key in clc_ids_by_leaf:
            fail("a malformed v22 CLC family record")
        if any(
            config_id not in manifest_entries
            or manifest_leaves[config_id] != expected_leaf
            or manifest_configs[config_id].get("cute_flash_clc_heads_per_batch")
            != value
            for value in typed_planned
            for config_id in (cast("str", typed_witness_ids[str(value)]),)
        ):
            fail("a CLC witness has the wrong canonical family or divisor")

        def validate_clc_repairs(
            raw_decisions: object,
            repair_ids_by_value: dict[str, list[str]],
            primary_ids_by_value: dict[str, object],
            values: list[int],
            *,
            kind: str,
            detail: str,
            decision_passes: set[int],
            first_passes: set[int],
            expected_leaf: dict[str, object],
            missing_generation_parent_ids_by_value: dict[str, str] | None = None,
        ) -> None:
            if not isinstance(raw_decisions, list):
                fail(detail)
            expected_values = [
                value for value in values if str(value) in repair_ids_by_value
            ]
            if len(raw_decisions) != len(expected_values):
                fail(detail)
            local_pass_counts: dict[int, int] = {}
            neighbor_limits_by_pass: dict[int, list[int]] = {}
            ordered_decision_passes: list[int] = []
            for value, raw_decision in zip(expected_values, raw_decisions, strict=True):
                if not isinstance(raw_decision, dict) or set(raw_decision) != {
                    "kind",
                    "value",
                    "repair_index",
                    "candidate_results",
                    "selected_config_id",
                    "generated_config_ids",
                    "neighbor_generation_limit",
                }:
                    fail(detail)
                repair_index = raw_decision["repair_index"]
                if type(repair_index) is not int:
                    fail(detail)
                candidates = decision_results(
                    raw_decision["candidate_results"],
                    detail,
                    allow_failed=True,
                )
                generated_ids = config_ids(raw_decision["generated_config_ids"], detail)
                candidate_passes = {
                    cast("int", candidate["measurement_pass_index"])
                    for candidate in candidates
                    if candidate["measurement_pass_index"] is not None
                }
                if len(candidate_passes) != 1:
                    fail(detail)
                decision_pass = next(iter(candidate_passes))
                primary = primary_ids_by_value.get(str(value))
                if isinstance(primary, str):
                    primary_ids = [primary]
                elif isinstance(primary, list) and all(
                    isinstance(config_id, str) for config_id in primary
                ):
                    primary_ids = primary
                elif primary is None:
                    primary_ids = []
                else:
                    fail(detail)
                prior_repair_ids = repair_ids_by_value[str(value)][:repair_index]
                tracked_ids = [*primary_ids, *prior_repair_ids]
                state_at_decision = measurement_states_by_pass[decision_pass]
                retryable_ids = sorted(
                    config_id
                    for config_id in tracked_ids
                    if state_at_decision.get(config_id, {}).get("status")
                    in retryable_failure_statuses
                )
                tracked_statuses = [
                    state_at_decision.get(config_id, {}).get("status")
                    for config_id in tracked_ids
                ]
                fallback_parent_id = (
                    None
                    if missing_generation_parent_ids_by_value is None
                    else missing_generation_parent_ids_by_value.get(str(value))
                )
                missing_generation = not tracked_ids and fallback_parent_id is not None
                expected_parent_ids = (
                    [fallback_parent_id]
                    if missing_generation
                    and state_at_decision.get(fallback_parent_id, {}).get("status")
                    in successful_statuses
                    else retryable_ids
                )
                expected_generated = repair_ids_by_value[str(value)][
                    repair_index : repair_index + 1
                ]
                if (
                    raw_decision["kind"] != kind
                    or raw_decision["value"] != value
                    or not 0
                    <= repair_index
                    < cast("int", phase["qualification_failure_retries"])
                    or expected_parent_ids
                    != [cast("str", candidate["config_id"]) for candidate in candidates]
                    or (
                        not missing_generation
                        and (
                            not tracked_ids
                            or any(status is None for status in tracked_statuses)
                            or any(
                                status in successful_statuses
                                or status
                                in {"accuracy_error", "source_rejected", "filtered"}
                                for status in tracked_statuses
                            )
                            or not retryable_ids
                        )
                    )
                    or not expected_parent_ids
                    or raw_decision["selected_config_id"] != expected_parent_ids[0]
                    or generated_ids != expected_generated
                    or len(generated_ids) > 1
                    or (missing_generation and not generated_ids)
                    or type(raw_decision["neighbor_generation_limit"]) is not int
                    or not 0
                    < raw_decision["neighbor_generation_limit"]
                    <= qualification_neighbor_limit
                ):
                    fail(detail)
                if not generated_ids and not missing_generation:
                    if decision_pass + 1 >= len(measurement_states_by_pass):
                        fail(detail)
                    completed_states = measurement_states_by_pass[decision_pass + 1]
                    if not any(
                        completed_states.get(config_id, {}).get("status")
                        == "deduplicated"
                        and valid_perf(
                            completed_states.get(config_id, {}).get("attempt_perf")
                        )
                        and valid_perf(
                            completed_states.get(config_id, {}).get("selection_perf")
                        )
                        for config_id in retryable_ids
                    ):
                        fail(detail)
                if generated_ids:
                    generated_id = generated_ids[0]
                    if (
                        generated_id in tracked_ids
                        or generated_id in state_at_decision
                        or manifest_leaves.get(generated_id) != expected_leaf
                        or manifest_configs.get(generated_id, {}).get(
                            "cute_flash_clc_heads_per_batch"
                        )
                        != value
                    ):
                        fail(detail)
                decision_passes.add(decision_pass)
                ordered_decision_passes.append(decision_pass)
                local_pass_counts[decision_pass] = (
                    local_pass_counts.get(decision_pass, 0) + 1
                )
                neighbor_limits_by_pass.setdefault(decision_pass, []).append(
                    cast("int", raw_decision["neighbor_generation_limit"])
                )
            if not _flash_repair_passes_are_compact(
                ordered_decision_passes,
                per_pass_limit=cast("int", candidate_limit),
            ) or any(
                count > cast("int", candidate_limit)
                for count in local_pass_counts.values()
            ):
                fail(detail)
            for limits in neighbor_limits_by_pass.values():
                expected_limits = [
                    (index + 1) * qualification_neighbor_limit // len(limits)
                    - index * qualification_neighbor_limit // len(limits)
                    for index in range(len(limits))
                ]
                if limits != expected_limits:
                    fail(detail)
            if ordered_decision_passes:
                first_passes.add(ordered_decision_passes[0])

        validate_clc_repairs(
            witness_repair_decisions,
            parsed_witness_repair_ids,
            typed_witness_ids,
            typed_planned,
            kind="witness_failure_repair",
            detail="an invalid immutable CLC witness repair",
            decision_passes=clc_witness_repair_decision_passes,
            first_passes=clc_witness_repair_first_passes,
            expected_leaf=expected_leaf,
        )
        validate_clc_repairs(
            conditional_repair_decisions,
            parsed_conditional_repair_ids,
            typed_conditional_ids,
            typed_conditional_values,
            kind="conditional_failure_repair",
            detail="an invalid immutable CLC conditional repair",
            decision_passes=clc_conditional_repair_decision_passes,
            first_passes=clc_conditional_repair_first_passes,
            expected_leaf=expected_leaf,
            missing_generation_parent_ids_by_value={
                str(value): config_id
                for value, config_id in zip(
                    typed_selected_values,
                    typed_selected_config_ids,
                    strict=True,
                )
            },
        )

        parsed_witness_candidates: list[tuple[int, dict[str, object]]] = []
        expected_witness_candidate_count = len(typed_planned) + sum(
            len(ids) for ids in parsed_witness_repair_ids.values()
        )
        if (
            len(cast("list[object]", witness_candidate_results))
            != expected_witness_candidate_count
        ):
            fail("an invalid immutable CLC witness candidate snapshot")
        witness_candidate_index = 0
        for expected_value in typed_planned:
            expected_attempt_ids = {
                cast("str", typed_witness_ids[str(expected_value)]),
                *parsed_witness_repair_ids.get(str(expected_value), ()),
            }
            value_candidates: list[dict[str, object]] = []
            for _ in range(len(expected_attempt_ids)):
                raw_snapshot = cast("list[object]", witness_candidate_results)[
                    witness_candidate_index
                ]
                witness_candidate_index += 1
                if not isinstance(raw_snapshot, dict) or set(raw_snapshot) != {
                    "value",
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }:
                    fail("an invalid immutable CLC witness candidate snapshot")
                snapshot = decision_results(
                    [
                        {
                            key: raw_snapshot[key]
                            for key in (
                                "config_id",
                                "attempt_perf",
                                "selection_perf",
                                "status",
                                "source_hash",
                                "measurement_pass_index",
                            )
                        }
                    ],
                    "an invalid immutable CLC witness candidate snapshot",
                    allow_failed=True,
                    require_ranked=False,
                )[0]
                add_snapshot_pass(
                    clc_witness_snapshot_passes,
                    snapshot,
                    "an invalid immutable CLC witness candidate snapshot",
                )
                if raw_snapshot["value"] != expected_value:
                    fail("an invalid immutable CLC witness candidate snapshot")
                value_candidates.append(snapshot)
            if {
                cast("str", item["config_id"]) for item in value_candidates
            } != expected_attempt_ids or value_candidates != sorted(
                value_candidates,
                key=lambda item: (
                    cast("float", item["selection_perf"])
                    if item["status"] in successful_statuses
                    else math.inf,
                    cast("str", item["config_id"]),
                ),
            ):
                fail("an invalid immutable CLC witness candidate snapshot")
            parsed_witness_candidates.extend(
                (expected_value, snapshot) for snapshot in value_candidates
            )
        parsed_witness_selection: list[tuple[int, dict[str, object]]] = []
        for raw_snapshot in cast("list[object]", witness_selection_results):
            if not isinstance(raw_snapshot, dict) or set(raw_snapshot) != {
                "value",
                "config_id",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
            }:
                fail("an invalid immutable CLC witness decision")
            value = raw_snapshot["value"]
            if type(value) is not int:
                fail("an invalid immutable CLC witness decision")
            snapshot = {
                key: raw_snapshot[key]
                for key in (
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                )
            }
            parsed = decision_results(
                [snapshot], "an invalid immutable CLC witness decision"
            )[0]
            add_snapshot_pass(
                clc_witness_snapshot_passes,
                parsed,
                "an invalid immutable CLC witness decision",
            )
            parsed_witness_selection.append((cast("int", value), parsed))
        witness_success_by_value = {
            value: next(
                (
                    snapshot
                    for candidate_value, snapshot in parsed_witness_candidates
                    if candidate_value == value
                    and snapshot["status"] in successful_statuses
                ),
                None,
            )
            for value in typed_planned
        }
        expected_witness_selection = sorted(
            [
                (value, snapshot)
                for value, snapshot in witness_success_by_value.items()
                if snapshot is not None
            ],
            key=lambda item: (
                cast("float", item[1]["selection_perf"]),
                cast("str", item[1]["config_id"]),
                item[0],
            ),
        )
        if (
            parsed_witness_selection != expected_witness_selection
            or typed_selected_values
            != [value for value, _snapshot in parsed_witness_selection]
            or typed_selected_config_ids
            != [
                cast("str", snapshot["config_id"])
                for _value, snapshot in parsed_witness_selection
            ]
        ):
            fail("an invalid immutable CLC witness decision")

        witness_snapshot_by_value = dict(parsed_witness_selection)
        conditional_decisions = cast("list[object]", conditional_parent_decisions)
        if len(conditional_decisions) != len(typed_conditional_values):
            fail("an invalid immutable CLC conditional-parent decision")
        conditional_neighbor_limits = [
            (index + 1)
            * cast("int", conditional_neighbor_generation_limit)
            // len(typed_conditional_values)
            - index
            * cast("int", conditional_neighbor_generation_limit)
            // len(typed_conditional_values)
            for index in range(len(typed_conditional_values))
        ]
        for value, raw_decision, expected_neighbor_limit in zip(
            typed_conditional_values,
            conditional_decisions,
            conditional_neighbor_limits,
            strict=True,
        ):
            if not isinstance(raw_decision, dict) or set(raw_decision) != {
                "value",
                "candidate_results",
                "selected_config_id",
                "generated_config_ids",
                "neighbor_generation_limit",
            }:
                fail("an invalid immutable CLC conditional-parent decision")
            candidates = decision_results(
                raw_decision["candidate_results"],
                "an invalid immutable CLC conditional-parent decision",
            )
            for candidate in candidates:
                add_snapshot_pass(
                    clc_conditional_parent_passes,
                    candidate,
                    "an invalid immutable CLC conditional-parent decision",
                )
            generated_ids = config_ids(
                raw_decision["generated_config_ids"],
                "an invalid immutable CLC conditional-parent decision",
            )
            witness_snapshot = witness_snapshot_by_value[value]
            if (
                raw_decision["value"] != value
                or candidates != [witness_snapshot]
                or raw_decision["selected_config_id"] != witness_snapshot["config_id"]
                or generated_ids
                != cast("dict[str, list[str]]", typed_conditional_ids)[str(value)]
                or type(raw_decision["neighbor_generation_limit"]) is not int
                or raw_decision["neighbor_generation_limit"] != expected_neighbor_limit
            ):
                fail("an invalid immutable CLC conditional-parent decision")

        value_decisions = cast("list[object]", retained_value_decisions)
        if len(value_decisions) != len(typed_selected_values):
            fail("an invalid immutable CLC retained-value decision")
        chosen_by_value: dict[int, dict[str, object]] = {}
        for value, raw_decision in zip(
            typed_selected_values, value_decisions, strict=True
        ):
            if not isinstance(raw_decision, dict) or set(raw_decision) != {
                "value",
                "candidate_results",
                "selected_config_id",
            }:
                fail("an invalid immutable CLC retained-value decision")
            candidates = decision_results(
                raw_decision["candidate_results"],
                "an invalid immutable CLC retained-value decision",
                allow_failed=True,
            )
            for candidate in candidates:
                add_snapshot_pass(
                    clc_retained_snapshot_passes,
                    candidate,
                    "an invalid immutable CLC retained-value decision",
                )
            expected_candidate_ids = {
                cast("str", typed_witness_ids[str(value)]),
                *parsed_witness_repair_ids.get(str(value), ()),
                *cast("dict[str, list[str]]", typed_conditional_ids).get(
                    str(value), []
                ),
                *parsed_conditional_repair_ids.get(str(value), ()),
            }
            if (
                raw_decision["value"] != value
                or not candidates
                or {cast("str", candidate["config_id"]) for candidate in candidates}
                != expected_candidate_ids
                or any(
                    manifest_configs[cast("str", candidate["config_id"])].get(
                        "cute_flash_clc_heads_per_batch"
                    )
                    != value
                    for candidate in candidates
                )
                or not any(
                    candidate["status"] in successful_statuses
                    for candidate in candidates
                )
                or raw_decision["selected_config_id"]
                != next(
                    (
                        candidate["config_id"]
                        for candidate in candidates
                        if candidate["status"] in successful_statuses
                    ),
                    None,
                )
            ):
                fail("an invalid immutable CLC retained-value decision")
            chosen_by_value[value] = next(
                candidate
                for candidate in candidates
                if candidate["status"] in successful_statuses
            )

        parsed_retained_ranking: list[tuple[int, dict[str, object]]] = []
        for raw_snapshot in cast("list[object]", retained_ranking_results):
            if not isinstance(raw_snapshot, dict) or set(raw_snapshot) != {
                "value",
                "config_id",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
            }:
                fail("an invalid immutable CLC retained ranking")
            value = raw_snapshot["value"]
            if type(value) is not int or value not in chosen_by_value:
                fail("an invalid immutable CLC retained ranking")
            snapshot = decision_results(
                [
                    {
                        key: raw_snapshot[key]
                        for key in (
                            "config_id",
                            "attempt_perf",
                            "selection_perf",
                            "status",
                            "source_hash",
                            "measurement_pass_index",
                        )
                    }
                ],
                "an invalid immutable CLC retained ranking",
            )[0]
            add_snapshot_pass(
                clc_retained_snapshot_passes,
                snapshot,
                "an invalid immutable CLC retained ranking",
            )
            if snapshot != chosen_by_value[cast("int", value)]:
                fail("an invalid immutable CLC retained ranking")
            parsed_retained_ranking.append((cast("int", value), snapshot))
        if (
            parsed_retained_ranking
            != sorted(
                parsed_retained_ranking,
                key=lambda item: (
                    cast("float", item[1]["selection_perf"]),
                    cast("str", item[1]["config_id"]),
                    item[0],
                ),
            )
            or {value for value, _snapshot in parsed_retained_ranking}
            != set(typed_selected_values)
            or typed_retained_values
            != [value for value, _snapshot in parsed_retained_ranking]
            or typed_retained_config_ids
            != [
                cast("str", snapshot["config_id"])
                for _value, snapshot in parsed_retained_ranking
            ]
        ):
            fail("an invalid immutable CLC retained ranking")

        if set(cast("dict[str, object]", depth_selection)) != {
            "candidate_results",
            "selected_representatives",
        }:
            fail("an invalid immutable CLC depth decision")
        depth_candidates = decision_results(
            cast("dict[str, object]", depth_selection)["candidate_results"],
            "an invalid immutable CLC depth decision",
        )
        for snapshot in depth_candidates:
            add_snapshot_pass(
                clc_depth_snapshot_passes,
                snapshot,
                "an invalid immutable CLC depth decision",
            )
        raw_depth_representatives = cast("dict[str, object]", depth_selection)[
            "selected_representatives"
        ]
        if not isinstance(raw_depth_representatives, list):
            fail("an invalid immutable CLC depth decision")
        depth_members = [
            (
                cast("str", snapshot["config_id"]),
                cast("float", snapshot["selection_perf"]),
                frozenset(
                    lane
                    for lane in lanes_by_leaf[
                        json.dumps(expected_leaf, sort_keys=True, separators=(",", ":"))
                    ]
                    if manifest_configs[cast("str", snapshot["config_id"])].get(lane[0])
                    == lane[1]
                ),
            )
            for snapshot in depth_candidates
        ]
        expected_depth_representatives = (
            _expected_flash_lane_diverse_members(
                depth_members,
                tuple(
                    lanes_by_leaf[
                        json.dumps(expected_leaf, sort_keys=True, separators=(",", ":"))
                    ]
                ),
                limit=cast("int", retained_per_leaf),
                pipeline_qualification_keys=(
                    "cute_flash_kv_stage",
                    "cute_flash_s_stage",
                ),
            )
            if result["combination_required"]
            else []
        )
        expected_depth_metrics = [
            {
                "config_id": member[0],
                "assigned_pipeline_lane": _flash_pipeline_lane_metric(lane),
            }
            for member, lane in expected_depth_representatives
        ]
        if (
            cast("list[object]", raw_depth_representatives) != expected_depth_metrics
            or (
                not result["combination_required"]
                and (depth_candidates or raw_depth_representatives)
            )
            or any(
                manifest_leaves[member[0]] != expected_leaf for member in depth_members
            )
        ):
            fail("an invalid immutable CLC depth decision")
        clc_depth_ids_by_leaf[clc_key] = [
            member[0] for member, _lane in expected_depth_representatives
        ]
        clc_depth_candidate_ids_by_leaf[clc_key] = [
            cast("str", snapshot["config_id"]) for snapshot in depth_candidates
        ]
        family_ids = set(cast("dict[str, str]", typed_witness_ids).values())
        witness_repair_flat_ids = [
            config_id for ids in parsed_witness_repair_ids.values() for config_id in ids
        ]
        conditional_flat_ids = [
            config_id
            for ids in cast("dict[str, list[str]]", typed_conditional_ids).values()
            for config_id in ids
        ]
        conditional_repair_flat_ids = [
            config_id
            for ids in parsed_conditional_repair_ids.values()
            for config_id in ids
        ]
        all_new_clc_ids = [
            *witness_repair_flat_ids,
            *conditional_flat_ids,
            *conditional_repair_flat_ids,
        ]
        if (
            len(conditional_flat_ids) != len(set(conditional_flat_ids))
            or len(all_new_clc_ids) != len(set(all_new_clc_ids))
            or set(all_new_clc_ids) & seen_clc_conditional_ids
            or any(
                config_id not in manifest_entries
                or manifest_leaves[config_id] != expected_leaf
                or manifest_configs[config_id].get("cute_flash_clc_heads_per_batch")
                != value
                for value in typed_conditional_values
                for config_id in cast("list[str]", typed_conditional_ids[str(value)])
            )
            or any(
                manifest_leaves.get(config_id) != expected_leaf
                or manifest_configs.get(config_id, {}).get(
                    "cute_flash_clc_heads_per_batch"
                )
                != value
                for mapping in (
                    parsed_witness_repair_ids,
                    parsed_conditional_repair_ids,
                )
                for value in typed_planned
                for config_id in mapping.get(str(value), ())
            )
        ):
            fail("a nonnovel or reordered v22 CLC candidate")
        clc_pre_combination_ids_by_leaf[clc_key] = (
            initial_id_set
            | set(expected_anchor_ids)
            | pipeline_generated_ids
            | family_ids
            | set(all_new_clc_ids)
        )
        seen_clc_conditional_ids.update(all_new_clc_ids)
        if any(
            config_id not in manifest_entries
            or manifest_leaves[config_id] != expected_leaf
            or manifest_configs[config_id].get("cute_flash_clc_heads_per_batch")
            not in typed_retained_values
            for config_id in typed_combination_ids
        ):
            fail("a CLC combination has the wrong canonical family or divisor")
        family_ids.update(all_new_clc_ids)
        family_ids.update(typed_combination_ids)
        clc_ids_by_leaf[clc_key] = family_ids
        clc_records_by_leaf[clc_key] = result
        clc_witness_ids_by_leaf[clc_key] = {
            *cast("dict[str, str]", typed_witness_ids).values(),
            *witness_repair_flat_ids,
            *conditional_repair_flat_ids,
        }
        max_clc_planned = max(max_clc_planned, len(typed_planned))
        has_clc_conditional |= bool(typed_conditional_values)
        has_clc_combination |= cast("bool", result["combination_required"])

    leaf_result_by_leaf = {
        json.dumps(
            {
                "family": result["family"],
                "compound_packet": result["compound_packet"],
                "softmax_disc": result["softmax_disc"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ): result
        for result in leaf_results
        if isinstance(result, dict)
    }
    if any(
        result["space_exhausted"]
        is not leaf_result_by_leaf[clc_key].get("space_exhausted")
        for clc_key, result in clc_records_by_leaf.items()
    ):
        fail("inconsistent v22 CLC leaf exhaustion evidence")
    if reported_exact_space_exhausted is not (
        exact_ids_measured
        and all(
            cast("bool", result["space_exhausted"])
            for result in clc_records_by_leaf.values()
        )
    ):
        fail("inconsistent hierarchical exact-space exhaustion evidence")

    transferred_compound_ids: set[str] = set()
    compound_source_ids: set[str] = set()
    compound_qualified_by_leaf: list[_FlashLeafQualification] = []
    compound_source_candidate_ids_by_leaf: dict[str, list[str]] = {}
    compound_source_pass_by_leaf: dict[str, int] = {}
    compound_backfill_pass_count = 0
    compound_leaf_keys_with_candidates: set[str] = set()
    for leaf, result in zip(compound_catalog, compound_transfers, strict=True):
        compound_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        ordinary_leaf = {
            "family": leaf["family"],
            "compound_packet": None,
            "softmax_disc": leaf["softmax_disc"],
        }
        ordinary_key = json.dumps(ordinary_leaf, sort_keys=True, separators=(",", ":"))
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "family",
                "compound_packet",
                "softmax_disc",
                "limit",
                "transfer_target_count",
                "transfer_count",
                "primary_transfer_config_ids",
                "backfill_rounds",
                "successful_transfer_config_ids",
                "qualified_transfer_config_ids",
                "failure_statuses_allowed",
                "source_selection",
                "transfers",
                "complete",
            }
            or {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
                "softmax_disc": result.get("softmax_disc"),
            }
            != leaf
            or result.get("complete") is not True
            or not isinstance(result.get("transfers"), list)
            or type(result.get("limit")) is not int
            or result.get("limit") != retained_per_leaf
            or type(result.get("transfer_count")) is not int
            or result.get("transfer_count")
            != len(cast("list[object]", result["transfers"]))
            or result.get("transfer_count") <= 0
            or result.get("transfer_count")
            > result.get("limit")
            * (1 + cast("int", phase["qualification_failure_retries"]))
            or type(result.get("transfer_target_count")) is not int
            or not 0 < result.get("transfer_target_count") <= result.get("limit")
            or not isinstance(result.get("backfill_rounds"), list)
            or result.get("failure_statuses_allowed") is not True
        ):
            fail("an incomplete v22 compound transfer")
        primary_transfer_ids = config_ids(
            result["primary_transfer_config_ids"],
            "an incomplete v22 compound transfer",
        )
        successful_transfer_ids = config_ids(
            result["successful_transfer_config_ids"],
            "an incomplete v22 compound transfer",
        )
        qualified_transfer_ids = config_ids(
            result["qualified_transfer_config_ids"],
            "an incomplete v22 compound transfer",
        )
        if (
            len(primary_transfer_ids) != result["transfer_target_count"]
            or len(successful_transfer_ids) < result["transfer_target_count"]
            or len(qualified_transfer_ids) != result["transfer_target_count"]
            or qualified_transfer_ids
            != successful_transfer_ids[: result["transfer_target_count"]]
        ):
            fail("an incomplete v22 compound transfer")
        source_selection = result.get("source_selection")
        if not isinstance(source_selection, dict) or set(source_selection) != {
            "candidate_results",
            "combination_prefix_count",
            "attempted_config_ids",
            "selected_config_ids",
        }:
            fail("an invalid immutable compound source decision")
        source_candidates = decision_results(
            source_selection["candidate_results"],
            "an invalid immutable compound source decision",
            require_ranked=False,
        )
        source_candidate_ids = [
            cast("str", snapshot["config_id"]) for snapshot in source_candidates
        ]
        source_snapshot_passes = {
            cast("int", snapshot["measurement_pass_index"])
            for snapshot in source_candidates
            if snapshot["measurement_pass_index"] is not None
        }
        if len(source_snapshot_passes) != 1:
            fail("an invalid immutable compound source decision")
        source_snapshot_pass = next(iter(source_snapshot_passes))
        compound_source_pass_by_leaf[compound_key] = source_snapshot_pass
        combination_prefix_count = source_selection["combination_prefix_count"]
        attempted_source_ids = config_ids(
            source_selection["attempted_config_ids"],
            "an invalid immutable compound source decision",
        )
        selected_source_ids = config_ids(
            source_selection["selected_config_ids"],
            "an invalid immutable compound source decision",
        )
        clc_result = clc_records_by_leaf.get(ordinary_key)
        expected_combination_source_ids = (
            []
            if clc_result is None
            else [
                config_id
                for config_id in cast(
                    "list[str]", clc_result["combination_candidate_ids"]
                )
                if any(
                    isinstance(cell, dict)
                    and cell.get("config_id") == config_id
                    and cell.get("status") in successful_statuses
                    for cell in cast("list[object]", clc_result["combination_cells"])
                )
            ]
        )
        if (
            type(combination_prefix_count) is not int
            or not 0 <= combination_prefix_count <= len(source_candidates)
            or set(source_candidate_ids[:combination_prefix_count])
            != set(expected_combination_source_ids)
            or combination_prefix_count != len(expected_combination_source_ids)
            or attempted_source_ids != source_candidate_ids[: len(attempted_source_ids)]
            or len(selected_source_ids) != result["transfer_count"]
            or any(
                manifest_leaves[config_id]
                != {
                    "family": leaf["family"],
                    "compound_packet": None,
                    "softmax_disc": leaf["softmax_disc"],
                }
                for config_id in source_candidate_ids
            )
        ):
            fail("an invalid immutable compound source decision")
        for segment in (
            source_candidates[:combination_prefix_count],
            source_candidates[combination_prefix_count:],
        ):
            if segment != sorted(
                segment,
                key=lambda snapshot: (
                    cast("float", snapshot["selection_perf"]),
                    cast("str", snapshot["config_id"]),
                ),
            ):
                fail("an invalid immutable compound source decision")
        compound_source_candidate_ids_by_leaf[compound_key] = source_candidate_ids
        compound_members: list[_FlashQualifiedMember] = []
        for transfer in cast("list[object]", result["transfers"]):
            if (
                not isinstance(transfer, dict)
                or set(transfer)
                != {
                    "source_config_id",
                    "source_config",
                    "transferred_config_id",
                    "projected_config",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                    "projection_overrides",
                    "projected_config_id",
                    "preserved_pipeline_values",
                }
                or not valid_config_id(transfer.get("source_config_id"))
                or not valid_config_id(transfer.get("transferred_config_id"))
                or not isinstance(transfer.get("source_config"), dict)
                or not isinstance(transfer.get("projected_config"), dict)
            ):
                fail("a malformed v22 compound transfer")
            source_id = cast("str", transfer["source_config_id"])
            target_id = cast("str", transfer["transferred_config_id"])
            source_config = cast("dict[str, object]", transfer["source_config"])
            projected_config = cast("dict[str, object]", transfer["projected_config"])
            projection_overrides = {"cute_flash_exp2_packet": leaf["compound_packet"]}
            try:
                expected_projection = _canonical_flash_projection(
                    config_generation,
                    source_config,
                    projection_overrides,
                )
            except InvalidConfig:
                fail("an invalid v22 compound projection")
            if (
                source_id == target_id
                or target_id in transferred_compound_ids
                or transfer.get("projected_config_id") != target_id
                or transfer.get("projection_overrides") != projection_overrides
                or canonical_config_id(source_config) != source_id
                or canonical_config_id(projected_config) != target_id
                or manifest_configs.get(source_id) != source_config
                or manifest_configs.get(target_id) != projected_config
                or projected_config != expected_projection
                or _flash_structural_leaf_dict(source_config)
                != {
                    "family": leaf["family"],
                    "compound_packet": None,
                    "softmax_disc": leaf["softmax_disc"],
                }
                or _flash_structural_leaf_dict(projected_config) != leaf
                or not flash_terminal_measurement_is_valid(transfer)
                or not isinstance(transfer.get("preserved_pipeline_values"), dict)
                or not set(
                    cast("dict[object, object]", transfer["preserved_pipeline_values"])
                )
                <= {"cute_flash_kv_stage", "cute_flash_s_stage"}
                or transfer.get("preserved_pipeline_values")
                != {
                    key: source_config[key]
                    for key in ("cute_flash_kv_stage", "cute_flash_s_stage")
                    if key in source_config
                }
            ):
                fail("a malformed v22 compound transfer")
            validate_measurement_snapshot(
                transfer,
                "an inconsistent compound transfer measurement snapshot",
                config_id=target_id,
                expected_pass_index=pre_probe_pass,
            )
            compound_source_ids.add(source_id)
            transferred_compound_ids.add(target_id)
            if target_id in qualified_transfer_ids:
                final_member = _flash_qualified_member_from_measurement_state(
                    target_id,
                    current_measurement_states.get(target_id),
                )
                if final_member is not None:
                    compound_members.append(final_member)
        if selected_source_ids != [
            cast("str", transfer["source_config_id"])
            for transfer in cast("list[dict[str, object]]", result["transfers"])
        ]:
            fail("an invalid immutable compound source decision")
        transfer_target_ids = [
            cast("str", transfer["transferred_config_id"])
            for transfer in cast("list[dict[str, object]]", result["transfers"])
        ]
        if any(
            target_id not in measurement_states_by_pass[source_snapshot_pass]
            for target_id in transfer_target_ids
        ):
            compound_leaf_keys_with_candidates.add(compound_key)
        expected_successful_transfer_ids = [
            cast("str", transfer["transferred_config_id"])
            for transfer in cast("list[dict[str, object]]", result["transfers"])
            if transfer["status"] in successful_statuses
        ]
        if (
            primary_transfer_ids
            != transfer_target_ids[: cast("int", result["transfer_target_count"])]
            or successful_transfer_ids != expected_successful_transfer_ids
            or qualified_transfer_ids
            != expected_successful_transfer_ids[
                : cast("int", result["transfer_target_count"])
            ]
        ):
            fail("an incomplete v22 compound transfer")
        expected_selected_source_ids: list[str] = []
        seen_projected_configs: set[str] = set()
        for source_id in attempted_source_ids:
            source_config = manifest_configs[source_id]
            try:
                projected = _canonical_flash_projection(
                    config_generation,
                    source_config,
                    {"cute_flash_exp2_packet": leaf["compound_packet"]},
                )
            except InvalidConfig:
                continue
            projected_key = json.dumps(projected, sort_keys=True, separators=(",", ":"))
            if (
                projected_key in seen_projected_configs
                or _flash_structural_leaf_dict(projected) != leaf
                or any(
                    key in source_config and projected.get(key) != source_config[key]
                    for key in ("cute_flash_kv_stage", "cute_flash_s_stage")
                )
            ):
                continue
            seen_projected_configs.add(projected_key)
            projected_id = canonical_config_id(projected)
            if (
                measurement_states_by_pass[source_snapshot_pass]
                .get(projected_id, {})
                .get("status")
                in retryable_failure_statuses
            ):
                continue
            expected_selected_source_ids.append(source_id)
        if selected_source_ids != expected_selected_source_ids:
            fail("an invalid immutable compound source decision")
        if (
            len(selected_source_ids) == cast("int", result["limit"])
            and (
                not attempted_source_ids
                or attempted_source_ids[-1] != selected_source_ids[-1]
            )
        ) or (
            len(selected_source_ids) < cast("int", result["limit"])
            and attempted_source_ids != source_candidate_ids
        ):
            fail("an invalid immutable compound source decision")

        raw_backfill_rounds = cast("list[object]", result["backfill_rounds"])
        compound_backfill_pass_count = max(
            compound_backfill_pass_count, len(raw_backfill_rounds)
        )
        if len(raw_backfill_rounds) > cast(
            "int", phase["qualification_failure_retries"]
        ):
            fail("an invalid compound transfer backfill")
        consumed_transfer_count = len(primary_transfer_ids)
        backfill_attempted_source_ids: list[str] = []
        for backfill_index, raw_backfill in enumerate(raw_backfill_rounds):
            if not isinstance(raw_backfill, dict) or set(raw_backfill) != {
                "repair_index",
                "required_successes",
                "failed_transfer_config_ids",
                "attempted_source_config_ids",
                "generated_config_ids",
            }:
                fail("an invalid compound transfer backfill")
            failed_ids = config_ids(
                raw_backfill["failed_transfer_config_ids"],
                "an invalid compound transfer backfill",
            )
            backfill_source_ids = config_ids(
                raw_backfill["attempted_source_config_ids"],
                "an invalid compound transfer backfill",
            )
            generated_ids = config_ids(
                raw_backfill["generated_config_ids"],
                "an invalid compound transfer backfill",
            )
            decision_pass = source_snapshot_pass + 1 + backfill_index
            if decision_pass >= len(measurement_states_by_pass):
                fail("an invalid compound transfer backfill")
            attempted_target_ids = transfer_target_ids[:consumed_transfer_count]
            states = measurement_states_by_pass[decision_pass]
            expected_failed_ids = [
                config_id
                for config_id in attempted_target_ids
                if states.get(config_id, {}).get("status") not in successful_statuses
            ]
            successful_count = sum(
                states.get(config_id, {}).get("status") in successful_statuses
                for config_id in attempted_target_ids
            )
            missing = cast("int", result["transfer_target_count"]) - successful_count
            if (
                raw_backfill["repair_index"] != backfill_index
                or raw_backfill["required_successes"] != missing
                or missing <= 0
                or failed_ids != expected_failed_ids
                or not failed_ids
                or any(
                    states.get(config_id, {}).get("status")
                    not in retryable_failure_statuses
                    for config_id in failed_ids
                )
                or generated_ids
                != transfer_target_ids[
                    consumed_transfer_count : consumed_transfer_count
                    + len(generated_ids)
                ]
                or len(generated_ids) > missing
            ):
                fail("an illegitimate compound transfer backfill")
            consumed_transfer_count += len(generated_ids)
            backfill_attempted_source_ids.extend(backfill_source_ids)
            completed_states = measurement_states_by_pass[decision_pass + 1]
            if sum(
                completed_states.get(config_id, {}).get("status") in successful_statuses
                for config_id in transfer_target_ids[:consumed_transfer_count]
            ) < cast("int", result["transfer_target_count"]):
                fail("an incomplete compound transfer backfill")
        primary_attempted_end = (
            attempted_source_ids.index(
                selected_source_ids[len(primary_transfer_ids) - 1]
            )
            + 1
        )
        if (
            consumed_transfer_count != len(transfer_target_ids)
            or attempted_source_ids[primary_attempted_end:]
            != backfill_attempted_source_ids
        ):
            fail("an invalid compound transfer backfill")
        primary_states = measurement_states_by_pass[source_snapshot_pass + 1]
        primary_success_count = sum(
            primary_states.get(config_id, {}).get("status") in successful_statuses
            for config_id in primary_transfer_ids
        )
        primary_failed_ids = [
            config_id
            for config_id in primary_transfer_ids
            if primary_states.get(config_id, {}).get("status")
            not in successful_statuses
        ]
        expected_backfill = bool(
            primary_success_count < cast("int", result["transfer_target_count"])
            and primary_failed_ids
            and all(
                primary_states.get(config_id, {}).get("status")
                in retryable_failure_statuses
                for config_id in primary_failed_ids
            )
        )
        if bool(raw_backfill_rounds) is not expected_backfill:
            fail("an invalid compound transfer backfill")
        compound_qualified_by_leaf.append(
            (
                cast("str", leaf["family"]),
                cast("str", leaf["compound_packet"]),
                cast("bool", leaf["softmax_disc"]),
                compound_members,
                (),
            )
        )

    qualification_rounds = cast("int", rounds)
    per_pass_limit = cast("int", candidate_limit)
    retained_limit = cast("int", retained_per_leaf)
    family_limit = cast("int", retained_family_limit)
    family_slowdown_limit = cast("float", retained_family_slowdown_limit)
    path_limit = cast("int", starting_path_limit)
    base_pipeline_pass_count = max(
        (
            max(
                (
                    0
                    if not lanes_by_leaf[leaf_key]
                    and result.get("space_exhausted") is True
                    else qualification_rounds
                ),
                math.ceil(len(lanes_by_leaf[leaf_key]) / per_pass_limit)
                + math.ceil(
                    sum(
                        lane.get("conditional_required") is True
                        for lane in cast(
                            "list[dict[str, object]]", result["pipeline_lanes"]
                        )
                    )
                    / per_pass_limit
                ),
            )
            for leaf, result in zip(ordinary_catalog, leaf_results, strict=True)
            for leaf_key in (json.dumps(leaf, sort_keys=True, separators=(",", ":")),)
        ),
        default=0,
    )
    pipeline_repair_pass_count = max(
        (
            math.ceil(
                sum(
                    bool(lane.get("repair_parent_decisions"))
                    for lane in cast(
                        "list[dict[str, object]]", result["pipeline_lanes"]
                    )
                )
                / per_pass_limit
            )
            for result in cast("list[dict[str, object]]", leaf_results)
        ),
        default=0,
    )
    pipeline_pass_count = base_pipeline_pass_count + pipeline_repair_pass_count
    clc_witness_repair_pass_count = max(
        (
            math.ceil(
                len(cast("list[object]", result["witness_repair_parent_decisions"]))
                / per_pass_limit
            )
            for result in cast("list[dict[str, object]]", clc_families)
        ),
        default=0,
    )
    clc_conditional_repair_pass_count = max(
        (
            math.ceil(
                len(cast("list[object]", result["conditional_repair_parent_decisions"]))
                / per_pass_limit
            )
            for result in cast("list[dict[str, object]]", clc_families)
        ),
        default=0,
    )
    clc_witness_pass_count = int(max_clc_planned > 0)
    clc_conditional_pass_count = int(has_clc_conditional)
    clc_combination_pass_count = int(has_clc_combination)
    compound_primary_pass_count = int(bool(compound_catalog))
    expected_passes = (
        schedule_anchor_pass_count
        + pipeline_pass_count
        + clc_witness_pass_count
        + clc_witness_repair_pass_count
        + clc_conditional_pass_count
        + clc_conditional_repair_pass_count
        + clc_combination_pass_count
        + compound_primary_pass_count
        + compound_backfill_pass_count
        + (
            cast("int", family_probe_generations)
            if expected_family_probe_required
            else 0
        )
    )
    if phase.get("qualification_passes_planned") != expected_passes:
        fail("inconsistent v22 qualification pass accounting")
    witness_repair_start = (
        schedule_anchor_pass_count + pipeline_pass_count + clc_witness_pass_count
    )
    conditional_repair_start = (
        witness_repair_start
        + clc_witness_repair_pass_count
        + clc_conditional_pass_count
    )
    if (
        clc_witness_repair_decision_passes
        != set(
            range(
                witness_repair_start,
                witness_repair_start + clc_witness_repair_pass_count,
            )
        )
        or clc_conditional_repair_decision_passes
        != set(
            range(
                conditional_repair_start,
                conditional_repair_start + clc_conditional_repair_pass_count,
            )
        )
        or clc_witness_repair_first_passes
        != ({witness_repair_start} if clc_witness_repair_pass_count else set())
        or clc_conditional_repair_first_passes
        != ({conditional_repair_start} if clc_conditional_repair_pass_count else set())
    ):
        fail("inconsistent v22 CLC repair pass accounting")
    post_witness_pass = witness_repair_start + clc_witness_repair_pass_count
    post_conditional_pass = conditional_repair_start + clc_conditional_repair_pass_count
    if (
        clc_witness_snapshot_passes
        != ({post_witness_pass} if max_clc_planned > 0 else set())
        or clc_conditional_parent_passes
        != ({post_witness_pass} if has_clc_conditional else set())
        or clc_retained_snapshot_passes
        != ({post_conditional_pass} if max_clc_planned > 0 else set())
        or clc_depth_snapshot_passes
        != ({post_conditional_pass} if has_clc_combination else set())
    ):
        fail("inconsistent v22 CLC decision snapshot passes")
    compound_source_pass = (
        conditional_repair_start
        + clc_conditional_repair_pass_count
        + clc_combination_pass_count
    )
    if any(
        pass_index != compound_source_pass
        for pass_index in compound_source_pass_by_leaf.values()
    ):
        fail("an invalid immutable compound source decision")

    scheduled_ids_by_completion_pass = [
        set() for _ in range(cast("int", qualification_pass_count) + 1)
    ]
    if schedule_anchor_pass_count:
        scheduled_ids_by_completion_pass[1].update(expected_new_anchor_ids)
    for leaf_result in cast("list[dict[str, object]]", leaf_results):
        raw_rounds = leaf_result.get("rounds")
        if not isinstance(raw_rounds, list):
            fail("an invalid exact structural qualification round")
        for pass_index, round_result in enumerate(
            raw_rounds, start=schedule_anchor_pass_count + 1
        ):
            if not isinstance(round_result, dict):
                fail("an invalid exact structural qualification round")
            scheduled_ids_by_completion_pass[pass_index].update(
                config_ids(
                    round_result.get("candidate_config_ids"),
                    "an invalid exact structural qualification round",
                )
            )
    witness_completion_pass = (
        schedule_anchor_pass_count + pipeline_pass_count + clc_witness_pass_count
    )
    for clc_result in cast("list[dict[str, object]]", clc_families):
        scheduled_ids_by_completion_pass[witness_completion_pass].update(
            cast("dict[str, str]", clc_result["witness_config_ids"]).values()
        )
        for decision_key in (
            "witness_repair_parent_decisions",
            "conditional_repair_parent_decisions",
        ):
            for decision in cast("list[dict[str, object]]", clc_result[decision_key]):
                candidates = cast(
                    "list[dict[str, object]]", decision["candidate_results"]
                )
                decision_pass = cast("int", candidates[0]["measurement_pass_index"])
                scheduled_ids_by_completion_pass[decision_pass + 1].update(
                    cast("list[str]", decision["generated_config_ids"])
                )
        for decision in cast(
            "list[dict[str, object]]", clc_result["conditional_parent_decisions"]
        ):
            scheduled_ids_by_completion_pass[post_witness_pass + 1].update(
                cast("list[str]", decision["generated_config_ids"])
            )
        if clc_result["combination_required"]:
            scheduled_ids_by_completion_pass[post_conditional_pass + 1].update(
                cast("list[str]", clc_result["combination_candidate_ids"])
            )
    for transfer_result in cast("list[dict[str, object]]", compound_transfers):
        compound_key = json.dumps(
            {
                "family": transfer_result["family"],
                "compound_packet": transfer_result["compound_packet"],
                "softmax_disc": transfer_result["softmax_disc"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        source_pass = compound_source_pass_by_leaf[compound_key]
        scheduled_ids_by_completion_pass[source_pass + 1].update(
            cast("list[str]", transfer_result["primary_transfer_config_ids"])
        )
        for backfill_index, backfill in enumerate(
            cast("list[dict[str, object]]", transfer_result["backfill_rounds"])
        ):
            scheduled_ids_by_completion_pass[source_pass + 2 + backfill_index].update(
                cast("list[str]", backfill["generated_config_ids"])
            )

    family_probe_candidate_ids: set[str] = set()
    family_probe_candidate_ids_by_leaf: dict[str, set[str]] = {}
    family_probe_leaf_keys_with_candidates: set[str] = set()
    family_probe_compound_members: dict[str, list[_FlashQualifiedMember]] = {}
    probe_pass_start = pre_probe_pass
    pre_probe_states = measurement_states_by_pass[probe_pass_start]
    family_probe_parent_score_ids = {
        config_id
        for config_id, state in pre_probe_states.items()
        if state["status"] in successful_statuses
        and manifest_leaves[config_id]["compound_packet"] is None
    }
    if expected_family_probe_required:
        pre_probe_ordinary: dict[str, list[str]] = {}
        for config_id, state in pre_probe_states.items():
            leaf = manifest_leaves[config_id]
            if (
                state["status"] in successful_statuses
                and leaf["compound_packet"] is None
            ):
                family = cast("str", leaf["family"])
                pre_probe_ordinary.setdefault(family, []).append(config_id)
        qualified_compound_ids = {
            config_id
            for transfer in cast("list[dict[str, object]]", compound_transfers)
            for config_id in cast(
                "list[str]", transfer["qualified_transfer_config_ids"]
            )
        }
        probe_eligible_ids = {
            *family_probe_parent_score_ids,
            *(
                config_id
                for config_id in qualified_compound_ids
                if pre_probe_states.get(config_id, {}).get("status")
                in successful_statuses
            ),
        }

        def pre_probe_rank(config_id: str) -> tuple[float, str]:
            return (
                cast("float", pre_probe_states[config_id]["selection_perf"]),
                config_id,
            )

        expected_probe_starts: list[tuple[str, dict[str, object], bool]] = []
        family_starts = [
            min(config_ids, key=pre_probe_rank)
            for config_ids in pre_probe_ordinary.values()
        ]
        for config_id in sorted(
            family_starts,
            key=lambda item: (
                pre_probe_rank(item),
                cast("str", manifest_leaves[item]["family"]),
            ),
        ):
            expected_probe_starts.append((config_id, manifest_leaves[config_id], False))
        for leaf in compound_catalog:
            leaf_ids = [
                config_id
                for config_id in qualified_compound_ids
                if manifest_leaves[config_id] == leaf
                and pre_probe_states.get(config_id, {}).get("status")
                in successful_statuses
            ]
            if not leaf_ids:
                fail("a compound leaf has no measured family-probe start")
            config_id = min(leaf_ids, key=pre_probe_rank)
            expected_probe_starts.append(
                (config_id, cast("dict[str, object]", leaf), False)
            )
        expected_probe_starts[len(family_starts) :] = sorted(
            expected_probe_starts[len(family_starts) :],
            key=lambda item: (
                pre_probe_rank(item[0]),
                cast("str", item[1]["family"]),
                cast("str", item[1]["compound_packet"]),
                cast("bool", item[1]["softmax_disc"]),
            ),
        )
        global_start = min(probe_eligible_ids, key=pre_probe_rank)
        expected_probe_starts.append(
            (global_start, manifest_leaves[global_start], True)
        )
        if len(expected_probe_starts) != expected_family_probe_path_limit:
            fail("an incomplete live family-probe start catalog")

        for path_index, (raw_path, expected_start) in enumerate(
            zip(family_probe_paths, expected_probe_starts, strict=True)
        ):
            if not isinstance(raw_path, dict) or set(raw_path) != {
                "family",
                "compound_packet",
                "softmax_disc",
                "starting_config_id",
                "unrestricted",
                "rounds",
            }:
                fail("a malformed structural family-probe path")
            path = cast("dict[str, object]", raw_path)
            start_id, start_leaf, unrestricted = expected_start
            if (
                path.get("starting_config_id") != start_id
                or path.get("family") != start_leaf["family"]
                or path.get("compound_packet") != start_leaf["compound_packet"]
                or path.get("softmax_disc") != start_leaf["softmax_disc"]
                or path.get("unrestricted") is not unrestricted
                or not isinstance(path.get("rounds"), list)
                or len(cast("list[object]", path["rounds"]))
                != cast("int", family_probe_generations)
            ):
                fail("an inconsistent structural family-probe start")
            for generation_index, raw_round in enumerate(
                cast("list[object]", path["rounds"]), start=1
            ):
                expected_pass = probe_pass_start + generation_index
                if (
                    not isinstance(raw_round, dict)
                    or set(raw_round)
                    != {
                        "probe_generation",
                        "measurement_pass_index",
                        "candidate_ids",
                        "results",
                    }
                    or raw_round.get("probe_generation") != generation_index
                    or raw_round.get("measurement_pass_index") != expected_pass
                ):
                    fail("a malformed structural family-probe round")
                round_record = cast("dict[str, object]", raw_round)
                candidate_ids = config_ids(
                    round_record.get("candidate_ids"),
                    "invalid structural family-probe candidate IDs",
                )
                results = decision_results(
                    round_record.get("results"),
                    "invalid structural family-probe results",
                    allow_failed=True,
                    require_ranked=False,
                    expected_pass_index=expected_pass,
                )
                if (
                    [cast("str", result["config_id"]) for result in results]
                    != candidate_ids
                    or len(candidate_ids)
                    > cast("int", family_probe_candidates_per_path) - 1
                    or family_probe_candidate_ids.intersection(candidate_ids)
                    or any(config_id in pre_probe_states for config_id in candidate_ids)
                    or any(
                        not flash_terminal_measurement_is_valid(result)
                        for result in results
                    )
                ):
                    fail("an inconsistent structural family-probe result set")
                family_probe_candidate_ids.update(candidate_ids)
                scheduled_ids_by_completion_pass[expected_pass].update(candidate_ids)
                for result in results:
                    config_id = cast("str", result["config_id"])
                    leaf = manifest_leaves[config_id]
                    leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                    family_probe_candidate_ids_by_leaf.setdefault(leaf_key, set()).add(
                        config_id
                    )
                    family_probe_leaf_keys_with_candidates.add(leaf_key)
                    if not unrestricted and leaf != start_leaf:
                        fail("a constrained family probe changed structural leaf")
                    if not unrestricted and start_leaf["compound_packet"] is None:
                        family_probe_parent_score_ids.add(config_id)
                    if (
                        leaf["compound_packet"] is not None
                        and result["status"] in successful_statuses
                    ):
                        final_member = _flash_qualified_member_from_measurement_state(
                            config_id,
                            current_measurement_states.get(config_id),
                        )
                        if final_member is not None:
                            family_probe_compound_members.setdefault(
                                leaf_key, []
                            ).append(final_member)
                if path_index == len(expected_probe_starts) - 1 and not unrestricted:
                    fail("the final family-probe path is not unrestricted")
    initial_ids = set(cast("list[str]", phase["initial_config_ids"]))
    successful_phase_ids: set[str] = set()
    selection_perf_by_id: dict[str, float] = {}
    all_explicit_ids: set[str] = set(expected_new_anchor_ids)
    leaves_with_candidate_keys = {
        json.dumps(manifest_leaves[config_id], sort_keys=True, separators=(",", ":"))
        for config_id in expected_new_anchor_ids
    }
    qualified_by_leaf: list[_FlashLeafQualification] = []
    memberships_by_id: dict[str, frozenset[_FlashPipelineLane]] = {}
    seen_pipeline_conditional_ids: set[str] = set(expected_anchor_ids)
    live_pipeline_witnesses = {
        (
            json.dumps(
                {
                    "family": witness_leaf.pipeline_family,
                    "compound_packet": witness_leaf.compound_exp2_packet,
                    "softmax_disc": witness_leaf.softmax_disc,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            cast("str", key),
            cast("int", value),
        ): dict(config.config)
        for (witness_leaf, key, value), config in (
            config_generation.flash_pipeline_lane_witnesses().items()
        )
    }
    for leaf, result in zip(ordinary_catalog, leaf_results, strict=True):
        assert isinstance(result, dict)
        if set(result) != {
            "family",
            "compound_packet",
            "softmax_disc",
            "initial_config_ids",
            "space_exhausted",
            "space_config_count",
            "ordinary_search_required",
            "rounds",
            "pipeline_lanes",
            "qualified_results",
            "retained_config_ids",
            "complete",
        }:
            fail("an invalid exact structural leaf result")
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        expected_lanes = lanes_by_leaf[leaf_key]
        if (
            not isinstance(result["space_exhausted"], bool)
            or (
                result["space_config_count"] is None
                if exact_enumerated
                else result["space_config_count"] is not None
            )
            or (
                exact_enumerated
                and (
                    type(result["space_config_count"]) is not int
                    or cast("int", result["space_config_count"]) < 0
                )
            )
            or result["ordinary_search_required"]
            is not (not expected_lanes and not result["space_exhausted"])
            or (result["space_exhausted"] and not exact_enumerated)
            or (
                result["space_exhausted"]
                and cast("int", result["space_config_count"]) == 0
            )
        ):
            fail("invalid exact-space evidence for a structural leaf")
        result_initial_ids = config_ids(
            result["initial_config_ids"], "an invalid exact structural leaf result"
        )
        if not set(result_initial_ids) <= initial_ids:
            fail("an invalid exact structural leaf result")
        result_rounds = result["rounds"]
        if (
            not isinstance(result_rounds, list)
            or len(result_rounds) != pipeline_pass_count
        ):
            fail("an invalid exact structural qualification round")
        round_ids_by_pass: list[list[str]] = []
        parent_decisions_by_pass: list[object] = []
        seen_round_ids: set[str] = set()
        for pass_result in result_rounds:
            if (
                not isinstance(pass_result, dict)
                or set(pass_result)
                != {
                    "candidate_config_ids",
                    "neighbor_generation_limit",
                    "ordinary_neighbor_generation_limit",
                    "parent_decisions",
                }
                or type(pass_result.get("neighbor_generation_limit")) is not int
                or type(pass_result.get("ordinary_neighbor_generation_limit"))
                is not int
                or not 0 <= cast("int", pass_result["neighbor_generation_limit"]) <= 200
            ):
                fail("an invalid exact structural qualification round")
            pass_ids = config_ids(
                pass_result["candidate_config_ids"],
                "an invalid exact structural qualification round",
            )
            if len(pass_ids) > per_pass_limit or seen_round_ids & set(pass_ids):
                fail("an invalid exact structural qualification round")
            seen_round_ids.update(pass_ids)
            round_ids_by_pass.append(pass_ids)
            parent_decisions_by_pass.append(pass_result["parent_decisions"])

        result_lanes = cast("list[object]", result["pipeline_lanes"])
        if [
            (lane.get("key"), lane.get("value"))
            for lane in result_lanes
            if isinstance(lane, dict)
        ] != expected_lanes:
            fail("a seed-dependent or fabricated pipeline lane catalog")
        lane_records: dict[_FlashPipelineLane, dict[str, object]] = {}
        lane_ids_by_pass: list[list[str]] = [[] for _ in range(pipeline_pass_count)]
        lane_neighbor_by_pass = [0 for _ in range(pipeline_pass_count)]
        lane_explicit_ids: set[str] = set()
        for lane_object, lane in zip(result_lanes, expected_lanes, strict=True):
            if not isinstance(lane_object, dict) or set(lane_object) != {
                "key",
                "value",
                "initial_config_ids",
                "space_exhausted",
                "space_config_count",
                "conditional_required",
                "rounds",
                "witness_attempted",
                "witness_config_id",
                "witness_succeeded",
                "conditional_candidate_ids",
                "successful_conditional_candidate_ids",
                "repair_candidate_ids",
                "successful_repair_candidate_ids",
                "repair_parent_decisions",
                "terminal_failure_exhausted",
                "complete",
            }:
                fail("an invalid normalized pipeline lane")
            lane_records[lane] = lane_object
            lane_initial_ids = config_ids(
                lane_object["initial_config_ids"],
                "an invalid normalized pipeline lane",
            )
            if not set(lane_initial_ids) <= set(result_initial_ids):
                fail("an invalid normalized pipeline lane")
            witness_id = lane_object["witness_config_id"]
            conditional_ids = config_ids(
                lane_object["conditional_candidate_ids"],
                "an incomplete v22 pipeline lane",
            )
            successful_conditional_ids = config_ids(
                lane_object["successful_conditional_candidate_ids"],
                "an incomplete v22 pipeline lane",
            )
            repair_ids = config_ids(
                lane_object["repair_candidate_ids"],
                "an incomplete v22 pipeline lane",
            )
            successful_repair_ids = config_ids(
                lane_object["successful_repair_candidate_ids"],
                "an incomplete v22 pipeline lane",
            )
            repair_decisions = lane_object["repair_parent_decisions"]
            if (
                lane_object["witness_attempted"] is not True
                or not valid_config_id(witness_id)
                or not isinstance(lane_object["witness_succeeded"], bool)
                or not isinstance(lane_object["space_exhausted"], bool)
                or (
                    lane_object["space_config_count"] is None
                    if exact_enumerated
                    else lane_object["space_config_count"] is not None
                )
                or (
                    exact_enumerated
                    and (
                        type(lane_object["space_config_count"]) is not int
                        or cast("int", lane_object["space_config_count"]) < 0
                    )
                )
                or not isinstance(lane_object["conditional_required"], bool)
                or lane_object["conditional_required"] is lane_object["space_exhausted"]
                or (lane_object["space_exhausted"] and not exact_enumerated)
                or (
                    lane_object["space_exhausted"]
                    and cast("int", lane_object["space_config_count"]) == 0
                )
                or len(conditional_ids)
                != (1 if lane_object["conditional_required"] else 0)
                or bool(set(conditional_ids) & initial_ids)
                or witness_id in conditional_ids
                or bool(set(conditional_ids) & seen_pipeline_conditional_ids)
                or not set(successful_conditional_ids) <= set(conditional_ids)
                or len(repair_ids) > 1
                or bool(set(repair_ids) & initial_ids)
                or witness_id in repair_ids
                or bool(set(repair_ids) & seen_pipeline_conditional_ids)
                or bool(set(repair_ids) & set(conditional_ids))
                or not set(successful_repair_ids) <= set(repair_ids)
                or not isinstance(repair_decisions, list)
                or len(repair_decisions) > 1
                or (bool(repair_ids) and not repair_decisions)
                or not isinstance(lane_object["terminal_failure_exhausted"], bool)
                or lane_object["complete"] is not True
            ):
                fail("an incomplete v22 pipeline lane")
            seen_pipeline_conditional_ids.update([*conditional_ids, *repair_ids])
            lane_rounds = lane_object["rounds"]
            if (
                not isinstance(lane_rounds, list)
                or len(lane_rounds) != pipeline_pass_count
            ):
                fail("an invalid per-lane qualification round")
            seen_lane_ids: set[str] = set()
            witness_pass: int | None = None
            conditional_pass: int | None = None
            repair_pass: int | None = None
            for pass_index, lane_round in enumerate(lane_rounds):
                if (
                    not isinstance(lane_round, dict)
                    or set(lane_round)
                    != {"candidate_config_ids", "neighbor_generation_limit"}
                    or type(lane_round.get("neighbor_generation_limit")) is not int
                    or not 0
                    <= cast("int", lane_round["neighbor_generation_limit"])
                    <= 200
                ):
                    fail("an invalid per-lane qualification round")
                pass_ids = config_ids(
                    lane_round["candidate_config_ids"],
                    "an invalid per-lane qualification round",
                )
                if seen_lane_ids & set(pass_ids):
                    fail("an invalid per-lane qualification round")
                seen_lane_ids.update(pass_ids)
                lane_ids_by_pass[pass_index].extend(pass_ids)
                lane_neighbor_by_pass[pass_index] += cast(
                    "int", lane_round["neighbor_generation_limit"]
                )
                if witness_id in pass_ids:
                    witness_pass = pass_index
                if conditional_ids and conditional_ids[0] in pass_ids:
                    conditional_pass = pass_index
                if repair_ids and repair_ids[0] in pass_ids:
                    repair_pass = pass_index
            if (
                seen_lane_ids
                != {cast("str", witness_id), *conditional_ids, *repair_ids}
                or witness_pass is None
                or (
                    bool(conditional_ids)
                    and (conditional_pass is None or witness_pass >= conditional_pass)
                )
                or (
                    bool(repair_ids)
                    and (
                        repair_pass is None
                        or repair_pass < base_pipeline_pass_count
                        or (
                            conditional_pass is not None
                            and conditional_pass >= repair_pass
                        )
                    )
                )
            ):
                fail("an incomplete v22 pipeline lane")
            lane_explicit_ids.update(seen_lane_ids)

        recorded_witness_ids = {
            cast("str", lane_record["witness_config_id"])
            for lane_record in lane_records.values()
        }
        recorded_repair_ids = {
            config_id
            for lane_record in lane_records.values()
            for config_id in cast("list[str]", lane_record["repair_candidate_ids"])
        }
        if recorded_witness_ids & recorded_repair_ids:
            fail("a pipeline repair reused a tracked witness")

        if expected_lanes:
            for pass_index, pass_ids in enumerate(round_ids_by_pass):
                pre_pass_ids = set(
                    measurement_states_by_pass[schedule_anchor_pass_count + pass_index]
                )
                if (
                    not set(pass_ids) <= set(lane_ids_by_pass[pass_index])
                    or not (set(lane_ids_by_pass[pass_index]) - set(pass_ids))
                    <= pre_pass_ids
                ):
                    fail("inconsistent per-lane qualification round IDs")
                if (
                    result_rounds[pass_index]["neighbor_generation_limit"]
                    != lane_neighbor_by_pass[pass_index]
                    or result_rounds[pass_index]["ordinary_neighbor_generation_limit"]
                    != 0
                ):
                    fail("an invalid per-lane qualification round")
        else:
            ordinary_pass_count = (
                0 if result["space_exhausted"] else qualification_rounds
            )
            for pass_index, pass_result in enumerate(result_rounds):
                expected_limit = 200 if pass_index < ordinary_pass_count else 0
                if (
                    pass_result["neighbor_generation_limit"] != expected_limit
                    or pass_result["ordinary_neighbor_generation_limit"]
                    != expected_limit
                ):
                    fail("an invalid exact structural qualification round")

        witness_jobs: list[tuple[str, _FlashPipelineLane | None, int | None]] = [
            ("witness", lane, None) for lane in expected_lanes
        ]
        conditional_jobs: list[tuple[str, _FlashPipelineLane | None, int | None]] = [
            ("conditional", lane, None)
            for lane in expected_lanes
            if lane_records[lane]["conditional_required"] is True
        ]
        expected_jobs_by_pass: list[
            list[tuple[str, _FlashPipelineLane | None, int | None]]
        ] = [
            jobs[offset : offset + per_pass_limit]
            for jobs in (witness_jobs, conditional_jobs)
            for offset in range(0, len(jobs), per_pass_limit)
        ]
        if not expected_lanes:
            expected_jobs_by_pass.extend(
                [("ordinary", None, None)]
                for _ in range(0 if result["space_exhausted"] else qualification_rounds)
            )
        expected_jobs_by_pass.extend(
            [] for _ in range(base_pipeline_pass_count - len(expected_jobs_by_pass))
        )
        repair_jobs: list[tuple[str, _FlashPipelineLane | None, int | None]] = [
            ("failure_repair", lane, repair_index)
            for repair_index in range(
                cast("int", phase["qualification_failure_retries"])
            )
            for lane in expected_lanes
            if len(cast("list[object]", lane_records[lane]["repair_parent_decisions"]))
            > repair_index
        ]
        repair_passes: list[list[tuple[str, _FlashPipelineLane | None, int | None]]] = [
            repair_jobs[offset : offset + per_pass_limit]
            for offset in range(0, len(repair_jobs), per_pass_limit)
        ]
        expected_jobs_by_pass.extend(repair_passes)
        expected_jobs_by_pass.extend(
            [] for _ in range(pipeline_repair_pass_count - len(repair_passes))
        )
        if len(expected_jobs_by_pass) != pipeline_pass_count:
            fail("inconsistent v22 qualification pass accounting")
        for pass_index, expected_jobs in enumerate(expected_jobs_by_pass):
            repair_lanes = [
                cast("_FlashPipelineLane", lane)
                for kind, lane, _repair_index in expected_jobs
                if kind == "failure_repair" and lane is not None
            ]
            if repair_lanes:
                repair_limits = [
                    (index + 1) * 200 // len(repair_lanes)
                    - index * 200 // len(repair_lanes)
                    for index in range(len(repair_lanes))
                ]
                for lane, expected_limit in zip(
                    repair_lanes, repair_limits, strict=True
                ):
                    lane_round = cast(
                        "list[dict[str, object]]", lane_records[lane]["rounds"]
                    )[pass_index]
                    if lane_round["neighbor_generation_limit"] != expected_limit:
                        fail("an invalid per-lane qualification round")
        for pass_index, (raw_decisions, expected_jobs, pass_ids) in enumerate(
            zip(
                parent_decisions_by_pass,
                expected_jobs_by_pass,
                round_ids_by_pass,
                strict=True,
            )
        ):
            if not isinstance(raw_decisions, list) or len(raw_decisions) != len(
                expected_jobs
            ):
                fail("an invalid immutable pipeline parent decision")
            emitted_ids: list[str] = []
            for job_index, (raw_decision, expected_job) in enumerate(
                zip(raw_decisions, expected_jobs, strict=True)
            ):
                if not isinstance(raw_decision, dict):
                    fail("an invalid immutable pipeline parent decision")
                decision = cast("dict[str, object]", raw_decision)
                kind, lane, repair_index = expected_job
                expected_decision_keys = {
                    "job_index",
                    "kind",
                    "pipeline_lane",
                    "selection_kind",
                    "candidate_results",
                    "selected_config_id",
                    "generated_config_ids",
                }
                if kind == "failure_repair":
                    expected_decision_keys.add("repair_index")
                if set(decision) != expected_decision_keys:
                    fail("an invalid immutable pipeline parent decision")
                expected_lane_metric = _flash_pipeline_lane_metric(lane)
                allow_unmeasured = (
                    kind == "witness"
                    and decision.get("selection_kind") == "catalog_witness"
                )
                candidates = decision_results(
                    decision.get("candidate_results"),
                    "an invalid immutable pipeline parent decision",
                    allow_unmeasured=allow_unmeasured,
                    allow_failed=True,
                    expected_pass_index=schedule_anchor_pass_count + pass_index,
                )
                generated_ids = config_ids(
                    decision.get("generated_config_ids"),
                    "an invalid immutable pipeline parent decision",
                )
                selected_id = decision.get("selected_config_id")
                expected_selection_kinds = (
                    {"ranked_existing", "catalog_witness"}
                    if kind == "witness"
                    else (
                        {"ranked_failed_parent"}
                        if kind == "failure_repair"
                        else {"ranked_parent"}
                    )
                )
                if (
                    decision.get("job_index") != job_index
                    or decision.get("kind") != kind
                    or decision.get("pipeline_lane") != expected_lane_metric
                    or decision.get("repair_index") != repair_index
                    or decision.get("selection_kind") not in expected_selection_kinds
                    or (selected_id is not None and not valid_config_id(selected_id))
                    or selected_id
                    != (candidates[0]["config_id"] if candidates else None)
                    or (not candidates and generated_ids)
                    or (
                        kind == "witness"
                        and (
                            generated_ids not in ([], [cast("str", selected_id)])
                            or (
                                decision.get("selection_kind") == "ranked_existing"
                                and generated_ids
                            )
                        )
                    )
                ):
                    fail("an invalid immutable pipeline parent decision")
                decision_ids = [
                    cast("str", candidate["config_id"]) for candidate in candidates
                ]
                if any(
                    manifest_leaves[config_id] != leaf
                    or (
                        lane is not None
                        and manifest_configs[config_id].get(lane[0]) != lane[1]
                    )
                    for config_id in [*decision_ids, *generated_ids]
                ):
                    fail("an invalid immutable pipeline parent decision")
                if kind == "witness":
                    emitted_ids.extend(generated_ids)
                else:
                    emitted_ids.extend(generated_ids)
                if kind == "conditional" and generated_ids != cast(
                    "list[str]",
                    lane_records[cast("_FlashPipelineLane", lane)][
                        "conditional_candidate_ids"
                    ],
                ):
                    fail("an invalid immutable pipeline parent decision")
                if kind == "failure_repair":
                    assert lane is not None and repair_index is not None
                    lane_record = lane_records[lane]
                    lane_decisions = cast(
                        "list[dict[str, object]]",
                        lane_record["repair_parent_decisions"],
                    )
                    lane_decision = lane_decisions[repair_index]
                    expected_lane_decision = {
                        key: decision[key]
                        for key in (
                            "repair_index",
                            "candidate_results",
                            "selected_config_id",
                            "generated_config_ids",
                        )
                    }
                    if (
                        lane_decision != expected_lane_decision
                        or generated_ids
                        != cast("list[str]", lane_record["repair_candidate_ids"])
                        or len(generated_ids) > 1
                    ):
                        fail("an invalid immutable pipeline repair decision")
                    if not generated_ids:
                        completed_states = measurement_states_by_pass[
                            schedule_anchor_pass_count + pass_index + 1
                        ]
                        if not any(
                            completed_states.get(config_id, {}).get("status")
                            == "deduplicated"
                            and valid_perf(
                                completed_states.get(config_id, {}).get("attempt_perf")
                            )
                            and valid_perf(
                                completed_states.get(config_id, {}).get(
                                    "selection_perf"
                                )
                            )
                            for config_id in decision_ids
                        ):
                            fail("an invalid immutable pipeline repair decision")
            if emitted_ids != pass_ids:
                fail("an invalid immutable pipeline parent decision")

        clc_ids = clc_ids_by_leaf.get(leaf_key, set())
        leaf_anchor_ids = {
            config_id
            for config_id in expected_anchor_ids
            if manifest_leaves[config_id] == leaf
        }
        expected_qualified_ids = (
            set(result_initial_ids)
            | leaf_anchor_ids
            | seen_round_ids
            | lane_explicit_ids
            | clc_ids
            | family_probe_candidate_ids_by_leaf.get(leaf_key, set())
        )
        qualified_results = cast("list[object]", result["qualified_results"])
        successful: list[_FlashQualifiedMember] = []
        qualified_ids: list[str] = []
        for qualified in qualified_results:
            if (
                not isinstance(qualified, dict)
                or set(qualified)
                != {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                    "pipeline_lanes",
                }
                or not valid_config_id(qualified.get("config_id"))
                or qualified.get("status") not in known_statuses
                or not isinstance(qualified.get("pipeline_lanes"), list)
            ):
                fail("an invalid measured result for an exact structural leaf")
            config_id = cast("str", qualified["config_id"])
            status = cast("str", qualified["status"])
            succeeded = status in successful_statuses
            manifest_entry = manifest_entries.get(config_id)
            manifest_config = manifest_configs.get(config_id)
            if (
                succeeded
                != (
                    valid_perf(qualified["attempt_perf"])
                    and valid_perf(qualified["selection_perf"])
                )
                or (
                    not succeeded
                    and (
                        qualified["attempt_perf"] is not None
                        or qualified["selection_perf"] is not None
                    )
                )
                or (
                    manifest_entry is None
                    or manifest_config is None
                    or manifest_leaves[config_id] != leaf
                )
            ):
                fail("an inconsistent result status/performance pair")
            validate_measurement_snapshot(
                qualified,
                "an inconsistent measured result timeline snapshot",
                config_id=config_id,
                expected_pass_index=cast("int", qualification_pass_count),
            )
            memberships: list[_FlashPipelineLane] = []
            for membership in cast("list[object]", qualified["pipeline_lanes"]):
                if (
                    not isinstance(membership, dict)
                    or set(membership) != {"key", "value"}
                    or (membership.get("key"), membership.get("value"))
                    not in expected_lanes
                ):
                    fail("an invalid measured pipeline lane membership")
                memberships.append(
                    (cast("str", membership["key"]), cast("int", membership["value"]))
                )
            if memberships != [
                lane
                for lane in expected_lanes
                if manifest_config.get(lane[0]) == lane[1]
            ]:
                fail("an invalid measured pipeline lane membership")
            frozen_memberships = frozenset(memberships)
            qualified_ids.append(config_id)
            memberships_by_id[config_id] = frozen_memberships
            if succeeded:
                selection_perf = cast("float", qualified["selection_perf"])
                successful_phase_ids.add(config_id)
                selection_perf_by_id[config_id] = selection_perf
                successful.append((config_id, selection_perf, frozen_memberships))
        if (
            len(qualified_ids) != len(set(qualified_ids))
            or set(qualified_ids) != expected_qualified_ids
        ):
            fail("inconsistent measured IDs for an exact structural leaf")

        available_ids = set(result_initial_ids) | leaf_anchor_ids
        for pass_index, (raw_decisions, expected_jobs, pass_ids) in enumerate(
            zip(
                parent_decisions_by_pass,
                expected_jobs_by_pass,
                round_ids_by_pass,
                strict=True,
            )
        ):
            assert isinstance(raw_decisions, list)
            for raw_decision, (kind, lane, repair_index) in zip(
                raw_decisions, expected_jobs, strict=True
            ):
                assert isinstance(raw_decision, dict)
                raw_candidates = cast(
                    "list[dict[str, object]]", raw_decision["candidate_results"]
                )
                candidate_ids = [
                    cast("str", candidate["config_id"]) for candidate in raw_candidates
                ]
                scoped_available = {
                    config_id
                    for config_id in available_ids
                    if manifest_leaves[config_id] == leaf
                    and (
                        lane is None
                        or manifest_configs[config_id].get(lane[0]) == lane[1]
                    )
                }
                state_at_decision = measurement_states_by_pass[
                    schedule_anchor_pass_count + pass_index
                ]
                expected_candidate_order: list[str] | None = None
                if kind == "failure_repair":
                    assert lane is not None and repair_index is not None
                    lane_record = lane_records[lane]
                    tracked_attempt_ids = {
                        cast("str", lane_record["witness_config_id"]),
                        *cast("list[str]", lane_record["conditional_candidate_ids"]),
                        *cast("list[str]", lane_record["repair_candidate_ids"])[
                            :repair_index
                        ],
                    }
                    tracked_statuses = {
                        config_id: state_at_decision.get(config_id, {}).get("status")
                        for config_id in tracked_attempt_ids
                    }
                    if not tracked_statuses or not all(
                        status in retryable_failure_statuses
                        for status in tracked_statuses.values()
                    ):
                        fail("an illegitimate pipeline failure repair")
                    expected_candidate_order = sorted(tracked_statuses)
                    if not expected_candidate_order:
                        fail("an illegitimate pipeline failure repair")
                    expected_candidate_ids = set(expected_candidate_order)
                    expected_selection_kind = "ranked_failed_parent"
                elif kind == "ordinary":
                    expected_candidate_ids = {
                        config_id
                        for config_id in scoped_available
                        if state_at_decision.get(config_id, {}).get("status")
                        in successful_statuses
                    }
                    expected_selection_kind = "ranked_parent"
                elif kind == "conditional":
                    expected_candidate_ids = scoped_available
                    expected_selection_kind = "ranked_parent"
                elif successful_scoped := {
                    config_id
                    for config_id in scoped_available
                    if state_at_decision.get(config_id, {}).get("status")
                    in successful_statuses
                }:
                    expected_candidate_ids = successful_scoped
                    expected_selection_kind = "ranked_existing"
                else:
                    assert lane is not None
                    witness_config = live_pipeline_witnesses.get(
                        (leaf_key, lane[0], lane[1])
                    )
                    expected_candidate_ids = (
                        set()
                        if witness_config is None
                        else {canonical_config_id(witness_config)}
                    )
                    expected_selection_kind = "catalog_witness"
                    if witness_config is not None:
                        witness_id = canonical_config_id(witness_config)
                        if manifest_configs.get(witness_id) != witness_config:
                            fail("an invalid live-catalog pipeline witness decision")
                if (
                    set(candidate_ids) != expected_candidate_ids
                    or (
                        expected_candidate_order is not None
                        and candidate_ids != expected_candidate_order
                    )
                    or raw_decision["selection_kind"] != expected_selection_kind
                ):
                    fail("an incomplete immutable pipeline parent decision")
            available_ids.update(pass_ids)

        for lane in expected_lanes:
            lane_record = lane_records[lane]
            expected_lane_initial_ids = [
                config_id
                for config_id in result_initial_ids
                if lane in memberships_by_id[config_id]
            ]
            if lane_record["initial_config_ids"] != expected_lane_initial_ids:
                fail("a normalized pipeline lane lacks a successful measured witness")
            witness_id = cast("str", lane_record["witness_config_id"])
            conditional_ids = cast(
                "list[str]", lane_record["conditional_candidate_ids"]
            )
            repair_ids = cast("list[str]", lane_record["repair_candidate_ids"])
            if lane not in memberships_by_id[witness_id] or any(
                lane not in memberships_by_id[config_id]
                for config_id in [*conditional_ids, *repair_ids]
            ):
                fail("a per-lane round contains a nonmember candidate")
            witness_succeeded = witness_id in successful_phase_ids
            successful_conditional_ids = [
                config_id
                for config_id in conditional_ids
                if config_id in successful_phase_ids
            ]
            successful_repair_ids = [
                config_id
                for config_id in repair_ids
                if config_id in successful_phase_ids
            ]
            has_success = bool(
                witness_succeeded or successful_conditional_ids or successful_repair_ids
            )
            tracked_ids = [witness_id, *conditional_ids, *repair_ids]
            repair_decisions = cast(
                "list[dict[str, object]]", lane_record["repair_parent_decisions"]
            )
            terminal_failure_exhausted = bool(
                not has_success
                and len(repair_ids)
                == cast("int", phase["qualification_failure_retries"])
                and len(repair_decisions)
                == cast("int", phase["qualification_failure_retries"])
                and all(
                    measurement_states_by_pass[-1].get(config_id, {}).get("status")
                    in retryable_failure_statuses
                    for config_id in tracked_ids
                )
            )
            if (
                lane_record["witness_succeeded"] is not witness_succeeded
                or lane_record["successful_conditional_candidate_ids"]
                != successful_conditional_ids
                or lane_record["successful_repair_candidate_ids"]
                != successful_repair_ids
                or lane_record["terminal_failure_exhausted"]
                is not terminal_failure_exhausted
                or not (has_success or terminal_failure_exhausted)
            ):
                fail("a normalized pipeline lane lacks a successful measured witness")

        if result["complete"] is not True or not successful:
            fail("an incomplete exact structural leaf")
        expected_retained = _expected_flash_lane_diverse_members(
            successful,
            tuple(expected_lanes),
            limit=retained_limit,
            pipeline_qualification_keys=(
                "cute_flash_kv_stage",
                "cute_flash_s_stage",
            ),
        )
        if result["retained_config_ids"] != [
            member[0] for member, _lane in expected_retained
        ]:
            fail("incorrect retained candidates for an exact structural leaf")
        qualified_by_leaf.append(
            (
                cast("str", result["family"]),
                cast("str | None", result["compound_packet"]),
                cast("bool", result["softmax_disc"]),
                successful,
                tuple(expected_lanes),
            )
        )
        explicit_ids = expected_qualified_ids - set(result_initial_ids)
        all_explicit_ids.update(explicit_ids)
        # The producer marks an ordinary leaf only when add_members() adds a
        # pipeline candidate or a CLC witness. Conditional CLC children and
        # combinations do not update leaves_with_candidates.
        clc_witness_ids = clc_witness_ids_by_leaf.get(leaf_key, set())
        if seen_round_ids or (clc_witness_ids - initial_ids):
            leaves_with_candidate_keys.add(leaf_key)

    for compound_key, candidate_ids in compound_source_candidate_ids_by_leaf.items():
        source_states = measurement_states_by_pass[
            compound_source_pass_by_leaf[compound_key]
        ]
        compound_leaf = cast("dict[str, object]", json.loads(compound_key))
        source_leaf = {
            "family": compound_leaf["family"],
            "compound_packet": None,
            "softmax_disc": compound_leaf["softmax_disc"],
        }
        expected_source_ids = {
            config_id
            for config_id, state in source_states.items()
            if state["status"] in successful_statuses
            if manifest_leaves[config_id] == source_leaf
        }
        if set(candidate_ids) != expected_source_ids:
            fail(f"an incomplete immutable compound source decision for {compound_key}")

    for clc_key, result in clc_records_by_leaf.items():
        witness_ids = cast("dict[str, str]", result["witness_config_ids"])
        conditional_ids = cast(
            "dict[str, list[str]]", result["conditional_candidate_ids"]
        )
        witness_repair_ids = cast(
            "dict[str, list[str]]", result["witness_repair_candidate_ids"]
        )
        conditional_repair_ids = cast(
            "dict[str, list[str]]", result["conditional_repair_candidate_ids"]
        )
        if (
            any(
                not {
                    witness_ids[str(value)],
                    *witness_repair_ids.get(str(value), ()),
                }
                & successful_phase_ids
                for value in cast("list[int]", result["planned_values"])
            )
            or any(
                not {
                    *conditional_ids.get(str(value), ()),
                    *conditional_repair_ids.get(str(value), ()),
                }
                & successful_phase_ids
                for value in cast("list[int]", result["conditional_values"])
            )
            or not set(cast("list[str]", result["retained_config_ids"]))
            <= successful_phase_ids
        ):
            fail(f"an immutable CLC decision references a failed config for {clc_key}")
        clc_leaf: dict[str, object] = {
            "family": result["family"],
            "compound_packet": None,
            "softmax_disc": result["softmax_disc"],
        }
        post_conditional_success_ids = {
            config_id
            for config_id, state in measurement_states_by_pass[
                post_conditional_pass
            ].items()
            if state["status"] in successful_statuses
        }
        expected_depth_candidate_ids = _expected_flash_clc_depth_candidate_ids(
            post_conditional_success_ids,
            manifest_leaves,
            clc_leaf,
            clc_pre_combination_ids_by_leaf[clc_key],
        )
        if set(clc_depth_candidate_ids_by_leaf[clc_key]) != (
            expected_depth_candidate_ids if result["combination_required"] else set()
        ):
            fail(f"an incomplete immutable CLC depth decision for {clc_key}")
        combination_ids = cast("list[str]", result["combination_candidate_ids"])
        combination_depth_ids = cast(
            "list[str]", result["combination_depth_config_ids"]
        )
        combination_divisor_values = cast(
            "list[int]", result["combination_divisor_values"]
        )
        combination_cells = cast("list[object]", result["combination_cells"])
        if combination_depth_ids != (
            clc_depth_ids_by_leaf[clc_key] if result["combination_required"] else []
        ) or combination_divisor_values != (
            cast("list[int]", result["retained_values"])
            if result["combination_required"]
            else []
        ):
            fail(f"incorrect CLC combination axes for {clc_key}")
        expected_combination_ids: list[str] = []
        seen_combination_configs: set[str] = set()
        expected_cells: list[dict[str, object]] = []
        for depth_config_id in combination_depth_ids:
            source = manifest_configs[depth_config_id]
            for value in combination_divisor_values:
                cell_prefix: dict[str, object] = {
                    "depth_config_id": depth_config_id,
                    "divisor_value": value,
                }
                raw_cell = combination_cells[len(expected_cells)]
                if not isinstance(raw_cell, dict) or set(raw_cell) != {
                    "depth_config_id",
                    "divisor_value",
                    "projected_config_id",
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }:
                    fail(f"a malformed CLC combination cell for {clc_key}")
                typed_cell = cast("dict[str, object]", raw_cell)
                try:
                    projected = _canonical_flash_projection(
                        config_generation,
                        source,
                        {"cute_flash_clc_heads_per_batch": value},
                    )
                except InvalidConfig:
                    projected = None
                if projected is not None and (
                    _flash_structural_leaf_dict(projected) != clc_leaf
                    or projected.get("cute_flash_clc_heads_per_batch") != value
                    or any(
                        key in source and projected.get(key) != source[key]
                        for key in ("cute_flash_kv_stage", "cute_flash_s_stage")
                    )
                ):
                    projected = None
                if projected is None:
                    expected_cell = {
                        **cell_prefix,
                        "projected_config_id": None,
                        "config_id": None,
                        "attempt_perf": None,
                        "selection_perf": None,
                        "status": "projection_rejected",
                        "source_hash": None,
                        "measurement_pass_index": None,
                    }
                    if typed_cell != expected_cell:
                        fail(f"an invalid rejected CLC combination cell for {clc_key}")
                    expected_cells.append(expected_cell)
                    continue
                projected_key = json.dumps(
                    projected, sort_keys=True, separators=(",", ":")
                )
                projected_id = canonical_config_id(projected)
                if (
                    manifest_configs.get(projected_id) != projected
                    or manifest_leaves.get(projected_id) != clc_leaf
                ):
                    fail(f"an unmeasured CLC combination cell for {clc_key}")
                cell_succeeded = typed_cell.get("status") in successful_statuses
                if (
                    typed_cell.get("depth_config_id") != depth_config_id
                    or typed_cell.get("divisor_value") != value
                    or typed_cell.get("projected_config_id") != projected_id
                    or typed_cell.get("config_id") != projected_id
                    or cell_succeeded
                    != (
                        valid_perf(typed_cell.get("attempt_perf"))
                        and valid_perf(typed_cell.get("selection_perf"))
                    )
                    or (
                        not cell_succeeded
                        and (
                            typed_cell.get("attempt_perf") is not None
                            or typed_cell.get("selection_perf") is not None
                        )
                    )
                ):
                    fail(f"an invalid measured CLC combination cell for {clc_key}")
                validate_measurement_snapshot(
                    typed_cell,
                    f"an invalid measured CLC combination cell for {clc_key}",
                    config_id=projected_id,
                    expected_pass_index=(
                        conditional_repair_start + clc_conditional_repair_pass_count + 1
                    ),
                )
                expected_cells.append(typed_cell)
                if projected_key not in seen_combination_configs:
                    seen_combination_configs.add(projected_key)
                    expected_combination_ids.append(projected_id)
        successful_depth_ids = [
            depth_config_id
            for depth_config_id in combination_depth_ids
            if any(
                cell.get("depth_config_id") == depth_config_id
                and cell.get("config_id") in successful_phase_ids
                for cell in expected_cells
            )
        ]
        successful_divisor_values = [
            value
            for value in combination_divisor_values
            if any(
                cell.get("divisor_value") == value
                and cell.get("config_id") in successful_phase_ids
                for cell in expected_cells
            )
        ]
        projection_complete = len(expected_cells) == len(combination_depth_ids) * len(
            combination_divisor_values
        ) and len(
            {
                (cell["depth_config_id"], cell["divisor_value"])
                for cell in expected_cells
            }
        ) == len(expected_cells)
        row_coverage_complete = successful_depth_ids == combination_depth_ids
        column_coverage_complete = (
            successful_divisor_values == combination_divisor_values
        )
        failure_statuses_allowed = all(
            flash_terminal_measurement_is_valid(cell, allow_projection_rejected=True)
            for cell in expected_cells
        )
        if (
            combination_ids != expected_combination_ids
            or combination_cells != expected_cells
            or result["combination_projection_complete"] is not projection_complete
            or result["successful_combination_depth_config_ids"] != successful_depth_ids
            or result["successful_combination_divisor_values"]
            != successful_divisor_values
            or result["combination_row_coverage_complete"] is not row_coverage_complete
            or result["combination_column_coverage_complete"]
            is not column_coverage_complete
            or result["combination_failure_statuses_allowed"]
            is not failure_statuses_allowed
            or (
                result["combination_required"]
                and not (
                    row_coverage_complete
                    and column_coverage_complete
                    and failure_statuses_allowed
                )
            )
        ):
            fail(f"incorrect CLC depth/divisor combinations for {clc_key}")

    for index, qualification in enumerate(compound_qualified_by_leaf):
        family, compound_packet, softmax_disc, members, lanes = qualification
        leaf_key = json.dumps(
            {
                "family": family,
                "compound_packet": compound_packet,
                "softmax_disc": softmax_disc,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        probe_members = family_probe_compound_members.get(leaf_key, [])
        if len({member[0] for member in [*members, *probe_members]}) != len(
            members
        ) + len(probe_members):
            fail("a duplicate compound family-probe candidate")
        compound_qualified_by_leaf[index] = (
            family,
            compound_packet,
            softmax_disc,
            [*members, *probe_members],
            lanes,
        )
    qualified_by_leaf.extend(compound_qualified_by_leaf)
    if not compound_source_ids <= successful_phase_ids:
        fail("an incomplete v22 compound transfer")
    referenced_config_ids = (
        initial_ids | set(phase_exact_ids) | set(expected_anchor_ids)
    )
    referenced_config_ids.update(
        cast("str", qualified["config_id"])
        for result in cast("list[dict[str, object]]", leaf_results)
        for qualified in cast("list[dict[str, object]]", result["qualified_results"])
    )
    referenced_config_ids.update(compound_source_ids)
    referenced_config_ids.update(transferred_compound_ids)
    referenced_config_ids.update(family_probe_candidate_ids)
    if set(manifest_entries) != referenced_config_ids:
        fail("the canonical config manifest does not match phase membership")
    all_explicit_ids.update(transferred_compound_ids - initial_ids)
    all_explicit_ids.update(family_probe_candidate_ids)
    if phase.get("candidate_count") != len(all_explicit_ids):
        fail("inconsistent exact structural qualification candidates")
    leaves_with_candidate_keys.update(compound_leaf_keys_with_candidates)
    leaves_with_candidate_keys.update(family_probe_leaf_keys_with_candidates)
    if phase.get("leaves_with_candidates") != len(leaves_with_candidate_keys):
        fail("inconsistent exact structural qualification candidates")

    expected_families = _expected_flash_structural_retention(
        qualified_by_leaf,
        retained_per_leaf=retained_limit,
        retained_family_cap=cast("int | None", retained_family_cap),
        retained_family_limit=family_limit,
        retained_family_slowdown_limit=family_slowdown_limit,
        starting_path_limit=path_limit,
        pipeline_qualification_keys=(
            "cute_flash_kv_stage",
            "cute_flash_s_stage",
        ),
        parent_score_config_ids=family_probe_parent_score_ids,
    )
    if not expected_families or phase.get("retained_families") != expected_families:
        fail("incorrect retained structural family ranking")
    expected_path_count = sum(
        len(cast("list[object]", family["starting_paths"]))
        for family in expected_families
    )
    if phase.get("retained_path_count") != expected_path_count:
        fail("an inconsistent retained structural path count")
    for pass_index in range(1, len(measurement_states_by_pass)):
        pre_state_ids = set(measurement_states_by_pass[pass_index - 1])
        actual_new_ids = set(measurement_states_by_pass[pass_index]) - pre_state_ids
        expected_new_ids = scheduled_ids_by_completion_pass[pass_index] - pre_state_ids
        raw_updates = cast(
            "list[object]",
            cast("dict[str, object]", measurement_timeline[pass_index])["updates"],
        )
        if actual_new_ids != expected_new_ids or (not expected_new_ids and raw_updates):
            fail("an inconsistent v22 measurement introduction timeline")
    final_invalidated_config_ids = {
        config_id
        for config_id, state in current_measurement_states.items()
        if state["status"] not in successful_statuses
        and state["source_hash"] in isolated_invalidated_source_hashes
    }
    return isolated_timeout_source_hashes, final_invalidated_config_ids


def _replay_strict_attention_initial_population_config_ids(
    config_generation: ConfigGeneration,
    *,
    random_seed: int,
    initial_population_size: int,
) -> list[str]:
    """Recreate a strict attention trial's normalized generation-zero configs."""
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.exc import InvalidConfig

    saved_random_state = random.getstate()
    try:
        random.seed(random_seed)
        raw_population = config_generation.random_population_flat(
            initial_population_size,
            user_seed_configs=(),
        )
        replayed_configs: set[object] = set()
        replayed_ids: list[str] = []
        for flat_config in raw_population:
            try:
                _canonical_flat, config = config_generation.canonicalize_flat(
                    flat_config
                )
            except InvalidConfig:
                continue
            if config in replayed_configs:
                continue
            replayed_configs.add(config)
            replayed_ids.append(canonical_config_id(config))
        return replayed_ids
    finally:
        random.setstate(saved_random_state)


def _validate_compiler_seed_generation_zero(
    compiler_seed_policy: dict[str, object],
    initial_results: object,
    *,
    trial_index: int,
    invalidated_config_ids: set[str] | None = None,
) -> None:
    """Require every canonical seed to have valid pass-zero measurement evidence."""
    if not isinstance(initial_results, list):
        raise RuntimeError(
            f"required full autotune trial {trial_index} has no generation-zero "
            "compiler-seed evidence"
        )
    records = {
        record.get("config_id"): record
        for record in initial_results
        if isinstance(record, dict) and record.get("measurement_pass_index") == 0
    }
    invalidated_config_ids = invalidated_config_ids or set()
    failures: dict[str, object] = {}
    for config_id in cast("list[str]", compiler_seed_policy["effective_config_ids"]):
        if config_id in invalidated_config_ids:
            failures[config_id] = "invalidated"
            continue
        record = records.get(config_id)
        if record is None:
            failures[config_id] = "missing"
            continue
        valid_perf = all(
            not isinstance(record.get(key), bool)
            and isinstance(record.get(key), (int, float))
            and math.isfinite(cast("float", record[key]))
            and cast("float", record[key]) > 0
            for key in ("attempt_perf", "selection_perf")
        )
        source_hash = record.get("source_hash")
        if (
            record.get("status") not in {"ok", "deduplicated"}
            or not valid_perf
            or not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
        ):
            failures[config_id] = record.get("status")
    if failures:
        raise RuntimeError(
            f"required full autotune trial {trial_index} did not successfully measure "
            f"every canonical compiler seed in pass 0: {failures}"
        )


def _validate_flash_terminal_measurement(
    raw_measurement: object,
    *,
    expected_base_order: list[str],
    expected_target_ms: float,
    fail: Callable[[str], NoReturn],
) -> dict[str, float]:
    if not isinstance(raw_measurement, dict) or set(raw_measurement) != {
        "base_order",
        "target_ms",
        "repeat_reference_perf_ms",
        "sweep_count",
        "calls_per_sample",
        "total_calls",
        "elapsed_ms",
        "median_ms",
    }:
        fail("a malformed terminal coordinate measurement")
    measurement = cast("dict[str, object]", raw_measurement)
    if measurement.get("base_order") != expected_base_order:
        fail("a terminal coordinate measurement with the wrong base order")
    target_ms = measurement.get("target_ms")
    repeat_reference_perf_ms = measurement.get("repeat_reference_perf_ms")
    sweep_count = measurement.get("sweep_count")
    calls_per_sample = measurement.get("calls_per_sample")
    total_calls = measurement.get("total_calls")
    if (
        isinstance(target_ms, bool)
        or not isinstance(target_ms, (int, float))
        or float(target_ms) != expected_target_ms
        or isinstance(repeat_reference_perf_ms, bool)
        or not isinstance(repeat_reference_perf_ms, (int, float))
        or not math.isfinite(repeat_reference_perf_ms)
        or repeat_reference_perf_ms <= 0
        or type(sweep_count) is not int
        or type(calls_per_sample) is not int
        or type(total_calls) is not int
    ):
        fail("a terminal coordinate measurement with invalid repeat provenance")
    base_repeat_float = expected_target_ms / float(repeat_reference_perf_ms)
    base_repeat = (
        _FLASH_TERMINAL_REPEAT_MAX
        if not math.isfinite(base_repeat_float)
        else int(base_repeat_float)
    )
    desired_calls = min(
        _FLASH_TERMINAL_REPEAT_MAX,
        max(3, base_repeat),
    )
    desired_calls = max(2, desired_calls + desired_calls % 2)
    expected_calls_per_sample = max(
        1,
        math.ceil(desired_calls / _FLASH_TERMINAL_MAX_SWEEPS),
    )
    expected_sweep_count = math.ceil(desired_calls / expected_calls_per_sample)
    if expected_sweep_count % 2:
        expected_sweep_count += 1
    expected_total_calls = expected_sweep_count * expected_calls_per_sample
    if (
        sweep_count != expected_sweep_count
        or calls_per_sample != expected_calls_per_sample
        or total_calls != expected_total_calls
        or sweep_count < 2
        or sweep_count > _FLASH_TERMINAL_MAX_SWEEPS
        or sweep_count % 2
        or total_calls < desired_calls
    ):
        fail("a terminal coordinate measurement with inconsistent batched call sizing")
    elapsed_rows = measurement.get("elapsed_ms")
    if (
        not isinstance(elapsed_rows, list)
        or len(elapsed_rows) != sweep_count
        or not expected_base_order
    ):
        fail("an incomplete terminal coordinate measurement trace")
    samples: list[list[float]] = [[] for _ in expected_base_order]
    indices = list(range(len(expected_base_order)))
    for sweep_index, elapsed in enumerate(elapsed_rows):
        offset = (sweep_index // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        expected_order = rotated if sweep_index % 2 == 0 else list(reversed(rotated))
        if not isinstance(elapsed, list) or len(elapsed) != len(expected_order):
            fail("a malformed terminal coordinate measurement elapsed row")
        for position, base_index in enumerate(expected_order):
            value = elapsed[position]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                fail("a terminal coordinate measurement with an invalid elapsed time")
            samples[base_index].append(float(value))
    expected_medians = [statistics.median(values) for values in samples]
    if float(repeat_reference_perf_ms) > 4.0 * max(expected_medians):
        fail(
            "a terminal coordinate measurement whose repeat reference is "
            "inconsistent with the raw timings"
        )
    if (
        max(sum(values) for values in samples) * calls_per_sample
        < 0.25 * expected_target_ms
    ):
        fail("a terminal coordinate measurement with insufficient raw timing work")
    raw_medians = measurement.get("median_ms")
    if not isinstance(raw_medians, list) or len(raw_medians) != len(
        expected_base_order
    ):
        fail("a terminal coordinate measurement with incomplete medians")
    medians: dict[str, float] = {}
    for config_id, expected_value, raw_median in zip(
        expected_base_order,
        expected_medians,
        raw_medians,
        strict=True,
    ):
        if not isinstance(raw_median, dict) or set(raw_median) != {
            "config_id",
            "value",
        }:
            fail("a malformed terminal coordinate median")
        value = raw_median.get("value")
        if (
            raw_median.get("config_id") != config_id
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or not math.isclose(
                float(value), expected_value, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            fail("a terminal coordinate measurement with an inconsistent median")
        medians[config_id] = float(value)
    return medians


def _validate_flash_terminal_coordinate_refinement(
    provenance: dict[str, object],
    phase: dict[str, object],
    trial: dict[str, object],
    *,
    trial_index: int,
    config_generation: ConfigGeneration,
) -> None:
    """Replay the strict CuTe-flash terminal coordinate search transcript."""
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion.autotuner.search_space_logger import canonical_config_id
    from helion.autotuner.surrogate_pattern_search import (
        flash_terminal_refinement_result_is_valid,
    )
    from helion.exc import InvalidConfig
    from helion.runtime.config import Config

    def fail(detail: str) -> NoReturn:
        raise RuntimeError(
            f"required full autotune trial {trial_index} recorded {detail}"
        )

    def valid_config_id(value: object) -> bool:
        return (
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None
        )

    def config_ids(value: object, detail: str) -> list[str]:
        if (
            not isinstance(value, list)
            or any(not valid_config_id(config_id) for config_id in value)
            or len(value) != len(set(value))
        ):
            fail(detail)
        return cast("list[str]", value)

    transcript = phase.get("terminal_coordinate_refinement")
    top_fields = {
        "schema_version",
        "policy_version",
        "lane_policy_version",
        "coordinate_policy",
        "measurement_policy",
        "rounds_planned",
        "beam_width",
        "maximum_projection_parent_count",
        "projection_parent_count",
        "rounds_started",
        "rounds_completed",
        "completed",
        "budget_exhausted",
        "termination_reason",
        "search_generation",
        "preterminal_num_configs_tested",
        "preterminal_registry_config_count",
        "preterminal_registry_config_ids_hash_policy",
        "preterminal_registry_config_ids_sha256",
        "radius",
        "minimum_improvement_fraction",
        "initial_incumbent_config_id",
        "refined_config_id",
        "final_config_id",
        "projection_attempt_count",
        "unique_candidate_count",
        "new_candidate_count",
        "reused_candidate_count",
        "intra_terminal_reused_candidate_count",
        "prior_failed_candidate_count",
        "accepted_config_ids",
        "config_manifest_sha256",
        "config_manifest",
        "rounds",
        "confirmation",
    }
    if not isinstance(transcript, dict) or set(transcript) != top_fields:
        fail("a missing or malformed terminal coordinate refinement transcript")
    terminal = cast("dict[str, object]", transcript)
    policy = provenance.get("flash_terminal_coordinate_refinement_policy")
    if not isinstance(policy, dict):
        fail("terminal coordinate refinement without a recorded policy")
    if (
        terminal.get("schema_version") != policy.get("schema_version")
        or terminal.get("policy_version") != policy.get("policy_version")
        or terminal.get("lane_policy_version") != policy.get("lane_policy_version")
        or terminal.get("coordinate_policy") != policy.get("coordinate_policy")
        or terminal.get("measurement_policy") != policy.get("measurement_policy")
        or terminal.get("rounds_planned") != policy.get("rounds")
        or terminal.get("beam_width") != policy.get("beam_width")
        or terminal.get("radius") != policy.get("radius")
        or terminal.get("minimum_improvement_fraction")
        != policy.get("minimum_improvement_fraction")
        or terminal.get("maximum_projection_parent_count")
        != 1
        + cast("int", policy["beam_width"]) * max(cast("int", policy["rounds"]) - 1, 0)
    ):
        fail("a terminal coordinate transcript with mismatched policy fields")
    integer_fields = (
        "schema_version",
        "policy_version",
        "lane_policy_version",
        "rounds_planned",
        "beam_width",
        "maximum_projection_parent_count",
        "projection_parent_count",
        "rounds_started",
        "rounds_completed",
        "search_generation",
        "preterminal_num_configs_tested",
        "preterminal_registry_config_count",
        "radius",
        "projection_attempt_count",
        "unique_candidate_count",
        "new_candidate_count",
        "reused_candidate_count",
        "intra_terminal_reused_candidate_count",
        "prior_failed_candidate_count",
    )
    minimum_improvement = terminal.get("minimum_improvement_fraction")
    if any(type(terminal.get(field)) is not int for field in integer_fields) or (
        isinstance(minimum_improvement, bool)
        or not isinstance(minimum_improvement, (int, float))
        or not math.isfinite(minimum_improvement)
    ):
        fail("a terminal coordinate transcript with invalid field types")
    if (
        terminal.get("completed") is not True
        or terminal.get("budget_exhausted") is not False
        or terminal.get("termination_reason") not in {"round_limit", "no_candidates"}
        or terminal.get("search_generation") != trial.get("num_generations")
    ):
        fail("an incomplete terminal coordinate refinement")
    preterminal_tested = terminal.get("preterminal_num_configs_tested")
    registry_count = terminal.get("preterminal_registry_config_count")
    registry_digest = terminal.get("preterminal_registry_config_ids_sha256")
    if (
        isinstance(preterminal_tested, bool)
        or not isinstance(preterminal_tested, int)
        or preterminal_tested < 0
        or isinstance(registry_count, bool)
        or not isinstance(registry_count, int)
        or registry_count <= 0
        or terminal.get("preterminal_registry_config_ids_hash_policy")
        != "sorted_compact_json_sha256_v1"
        or not isinstance(registry_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry_digest) is None
    ):
        fail("invalid preterminal coordinate-refinement boundary provenance")

    raw_manifest = terminal.get("config_manifest")
    if (
        not isinstance(raw_manifest, dict)
        or not raw_manifest
        or list(raw_manifest) != sorted(raw_manifest)
        or terminal.get("config_manifest_sha256")
        != _canonical_json_sha256(raw_manifest)
    ):
        fail("an invalid terminal coordinate config manifest")
    manifest = cast("dict[str, object]", raw_manifest)
    configs: dict[str, Config] = {}
    for config_id, raw_entry in manifest.items():
        if (
            not valid_config_id(config_id)
            or not isinstance(raw_entry, dict)
            or set(raw_entry) != {"config"}
            or not isinstance(raw_entry.get("config"), dict)
        ):
            fail("a malformed terminal coordinate config manifest entry")
        config = Config.from_dict(
            copy.deepcopy(cast("dict[str, object]", raw_entry["config"]))
        )
        if canonical_config_id(config) != config_id:
            fail("a terminal coordinate manifest entry with a noncanonical ID")
        try:
            _flat, normalized = config_generation.canonicalize_flat(
                config_generation.flatten(config)
            )
        except (InvalidConfig, KeyError, TypeError, ValueError):
            fail("a terminal coordinate manifest entry outside the live search surface")
        if normalized != config:
            fail("a non-normalized terminal coordinate manifest entry")
        configs[config_id] = config

    referenced_ids: set[str] = set()

    def one_config_id(value: object, detail: str) -> str:
        if not valid_config_id(value) or value not in configs:
            fail(detail)
        result = cast("str", value)
        referenced_ids.add(result)
        return result

    initial_id = one_config_id(
        terminal.get("initial_incumbent_config_id"),
        "an invalid terminal coordinate initial incumbent",
    )
    refined_id = one_config_id(
        terminal.get("refined_config_id"),
        "an invalid terminal coordinate refined config",
    )
    final_id = one_config_id(
        terminal.get("final_config_id"),
        "an invalid terminal coordinate final config",
    )
    accepted_ids = config_ids(
        terminal.get("accepted_config_ids"),
        "invalid accepted terminal coordinate config IDs",
    )
    for config_id in accepted_ids:
        one_config_id(
            config_id, "an accepted terminal coordinate config outside the manifest"
        )
    initial_leaf = _flash_structural_leaf_dict(dict(configs[initial_id].config))
    if initial_leaf is None:
        fail("a terminal coordinate incumbent without a CuTe flash leaf")
    surface = cast(
        "dict[str, object]", provenance["flash_terminal_coordinate_surface_catalog"]
    )
    if initial_leaf not in [
        cast("dict[str, object]", leaf)["leaf"]
        for leaf in cast("list[dict[str, object]]", surface["leaves"])
    ]:
        fail("a terminal coordinate incumbent outside the recorded leaf catalog")

    overrides = dict(config_generation._override_values)
    overrides[FLASH_PIPELINE_FAMILY_KEY] = initial_leaf["family"]
    overrides[FLASH_SOFTMAX_DISC_KEY] = initial_leaf["softmax_disc"]
    if initial_leaf["compound_packet"] is not None:
        overrides[FLASH_EXP2_PACKET_KEY] = initial_leaf["compound_packet"]
    leaf_generation = config_generation.config_spec.create_config_generation(
        overrides=overrides,
        advanced_controls_files=config_generation._advanced_controls_files,
        process_group_name=config_generation.process_group_name,
    )

    raw_rounds = terminal.get("rounds")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        fail("a terminal coordinate refinement without round evidence")
    rounds = cast("list[dict[str, object]]", raw_rounds)
    if (
        terminal.get("rounds_started") != len(rounds)
        or terminal.get("rounds_completed") != len(rounds)
        or len(rounds) > cast("int", policy["rounds"])
    ):
        fail("inconsistent terminal coordinate round accounting")

    round_fields = {
        "round_index",
        "incumbent_config_id",
        "leaf",
        "parent_config_ids",
        "parent_projections",
        "candidate_config_ids",
        "new_candidate_ids",
        "reused_candidate_ids",
        "intra_terminal_reused_candidate_ids",
        "prior_failed_candidate_ids",
        "candidate_results",
        "comparison_config_ids",
        "measurement",
        "round_best_config_id",
        "selected_config_id",
        "accepted",
        "improvement_fraction",
        "beam_config_ids",
    }
    request_fields = {
        "flat_index",
        "key",
        "sequence_index",
        "from_value",
        "to_value",
        "outcome",
        "config_id",
    }
    result_fields = {
        "config_id",
        "attempt_perf",
        "selection_perf",
        "status",
        "source_hash",
    }
    expected_incumbent_id = initial_id
    expected_parent_ids = [initial_id]
    all_unique_candidates: set[str] = set()
    all_new_candidates: set[str] = set()
    all_reused_candidates: set[str] = set()
    all_intra_terminal_reused: set[str] = set()
    all_prior_failed: set[str] = set()
    expected_accepted_ids: list[str] = []
    accepted_seen: set[str] = set()
    projection_attempt_count = 0
    projection_parent_count = 0
    terminal_source_hashes: dict[str, str] = {}
    stopped_for_no_candidates = False

    for round_index, raw_round in enumerate(rounds, 1):
        if not isinstance(raw_round, dict) or set(raw_round) != round_fields:
            fail("a malformed terminal coordinate round")
        round_metric = cast("dict[str, object]", raw_round)
        incumbent_id = one_config_id(
            round_metric.get("incumbent_config_id"),
            "an invalid terminal coordinate round incumbent",
        )
        parent_ids = config_ids(
            round_metric.get("parent_config_ids"),
            "invalid terminal coordinate parent IDs",
        )
        for config_id in parent_ids:
            one_config_id(
                config_id, "a terminal coordinate parent outside the manifest"
            )
        if (
            type(round_metric.get("round_index")) is not int
            or round_metric.get("round_index") != round_index
            or round_metric.get("leaf") != initial_leaf
            or incumbent_id != expected_incumbent_id
            or parent_ids != expected_parent_ids
            or not parent_ids
            or len(parent_ids) > cast("int", policy["beam_width"])
        ):
            fail("an inconsistent terminal coordinate round chain")
        parent_configs = {configs[config_id] for config_id in parent_ids}
        raw_parent_projections = round_metric.get("parent_projections")
        if not isinstance(raw_parent_projections, list) or len(
            raw_parent_projections
        ) != len(parent_ids):
            fail("incomplete terminal coordinate parent projections")
        expected_candidate_ids: list[str] = []
        round_seen_configs: set[Config] = set()
        for parent_id, raw_parent_projection in zip(
            parent_ids, raw_parent_projections, strict=True
        ):
            if not isinstance(raw_parent_projection, dict) or set(
                raw_parent_projection
            ) != {"parent_config_id", "coordinate_requests"}:
                fail("a malformed terminal coordinate parent projection")
            if raw_parent_projection.get("parent_config_id") != parent_id:
                fail("a terminal coordinate projection attached to the wrong parent")
            raw_requests = raw_parent_projection.get("coordinate_requests")
            if not isinstance(raw_requests, list):
                fail("missing terminal coordinate projection requests")
            projections = config_generation.canonicalize_coordinate_projections(
                leaf_generation.coordinate_neighbor_projections(
                    leaf_generation.flatten(configs[parent_id]),
                    radius=cast("int", policy["radius"]),
                ),
                base_config=configs[parent_id],
            )
            if len(raw_requests) != len(projections):
                fail("an incomplete terminal coordinate projection enumeration")
            projection_parent_count += 1
            projection_attempt_count += len(projections)
            for raw_request, projection in zip(raw_requests, projections, strict=True):
                if (
                    not isinstance(raw_request, dict)
                    or set(raw_request) != request_fields
                ):
                    fail("a malformed terminal coordinate projection request")
                request_sequence_index = raw_request.get("sequence_index")
                if (
                    type(raw_request.get("flat_index")) is not int
                    or not isinstance(raw_request.get("key"), str)
                    or (
                        request_sequence_index is not None
                        and type(request_sequence_index) is not int
                    )
                ):
                    fail("a terminal coordinate projection with invalid field types")
                outcome = projection.outcome
                projected_config = projection.config
                if (
                    outcome == "candidate"
                    and projected_config is not None
                    and _flash_structural_leaf_dict(dict(projected_config.config))
                    != initial_leaf
                ):
                    outcome = "different_leaf"
                elif outcome == "candidate" and projected_config in parent_configs:
                    outcome = "beam_alias"
                elif outcome == "candidate" and projected_config in round_seen_configs:
                    outcome = "round_candidate_alias"
                projected_id = (
                    canonical_config_id(projected_config)
                    if projected_config is not None
                    else None
                )
                expected_request = {
                    "flat_index": projection.flat_index,
                    "key": projection.key,
                    "sequence_index": projection.sequence_index,
                    "from_value": projection.from_value,
                    "to_value": projection.to_value,
                    "outcome": outcome,
                    "config_id": projected_id,
                }
                if raw_request != expected_request:
                    fail("a terminal coordinate projection that does not replay live")
                if projected_id is not None:
                    one_config_id(
                        projected_id,
                        "a terminal coordinate projection outside the manifest",
                    )
                    if configs[projected_id] != projected_config:
                        fail("a terminal coordinate projection with the wrong config")
                if outcome == "candidate":
                    assert projected_config is not None and projected_id is not None
                    round_seen_configs.add(projected_config)
                    expected_candidate_ids.append(projected_id)

        candidate_ids = config_ids(
            round_metric.get("candidate_config_ids"),
            "invalid terminal coordinate candidate IDs",
        )
        for config_id in candidate_ids:
            one_config_id(
                config_id, "a terminal coordinate candidate outside the manifest"
            )
        if candidate_ids != expected_candidate_ids:
            fail("terminal coordinate candidates that do not match their projections")
        all_unique_candidates.update(candidate_ids)
        categories = [
            config_ids(
                round_metric.get("new_candidate_ids"),
                "invalid new terminal coordinate candidate IDs",
            ),
            config_ids(
                round_metric.get("reused_candidate_ids"),
                "invalid reused terminal coordinate candidate IDs",
            ),
            config_ids(
                round_metric.get("intra_terminal_reused_candidate_ids"),
                "invalid intra-terminal reused candidate IDs",
            ),
            config_ids(
                round_metric.get("prior_failed_candidate_ids"),
                "invalid prior-failed terminal coordinate candidate IDs",
            ),
        ]
        if (
            set().union(*(set(category) for category in categories))
            != set(candidate_ids)
            or sum(len(category) for category in categories) != len(candidate_ids)
            or any(
                category
                != [config_id for config_id in candidate_ids if config_id in category]
                for category in categories
            )
        ):
            fail("an inconsistent terminal coordinate candidate classification")
        new_ids, reused_ids, intra_reused_ids, prior_failed_ids = categories
        if set(new_ids) & all_new_candidates:
            fail("a terminal coordinate candidate classified as new more than once")
        if not set(intra_reused_ids) <= all_new_candidates:
            fail("an intra-terminal reuse without an earlier new candidate")
        if set(reused_ids) & all_new_candidates:
            fail("a preterminal reuse that was created by terminal refinement")
        all_new_candidates.update(new_ids)
        all_reused_candidates.update(reused_ids)
        all_intra_terminal_reused.update(intra_reused_ids)
        all_prior_failed.update(prior_failed_ids)

        raw_results = round_metric.get("candidate_results")
        if not isinstance(raw_results, list):
            fail("missing terminal coordinate candidate results")
        result_ids: list[str] = []
        successful_result_ids: list[str] = []
        results_by_id: dict[str, dict[str, object]] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict) or set(raw_result) != result_fields:
                fail("a malformed terminal coordinate candidate result")
            result = cast("dict[str, object]", raw_result)
            config_id = one_config_id(
                result.get("config_id"),
                "a terminal coordinate result outside the manifest",
            )
            source_hash = result.get("source_hash")
            if (
                config_id not in candidate_ids
                or config_id in results_by_id
                or not flash_terminal_refinement_result_is_valid(result)
                or (
                    source_hash is not None
                    and (
                        not isinstance(source_hash, str)
                        or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
                    )
                )
            ):
                fail("an invalid terminal coordinate candidate result")
            result_ids.append(config_id)
            results_by_id[config_id] = result
            if result.get("status") in {"ok", "deduplicated"}:
                if not isinstance(source_hash, str):
                    fail(
                        "a successful terminal coordinate result without source identity"
                    )
                existing_source = terminal_source_hashes.get(config_id)
                if existing_source is not None and existing_source != source_hash:
                    fail("a terminal coordinate config with unstable generated source")
                terminal_source_hashes[config_id] = source_hash
                successful_result_ids.append(config_id)
        if result_ids != [
            config_id for config_id in candidate_ids if config_id in results_by_id
        ]:
            fail("terminal coordinate candidate results in the wrong order")
        if not set(new_ids + reused_ids + intra_reused_ids) <= set(result_ids):
            fail("a measured terminal coordinate candidate without a result")
        if not set(candidate_ids) - set(result_ids) <= set(prior_failed_ids):
            fail("an unexplained missing terminal coordinate candidate result")

        measured_ids: list[str] = []
        for config_id in [*parent_ids, *successful_result_ids]:
            if config_id not in measured_ids:
                measured_ids.append(config_id)
        comparison_ids = config_ids(
            round_metric.get("comparison_config_ids"),
            "invalid terminal coordinate comparison IDs",
        )
        for config_id in comparison_ids:
            one_config_id(
                config_id, "a terminal coordinate comparison outside the manifest"
            )
        round_best_id = one_config_id(
            round_metric.get("round_best_config_id"),
            "an invalid terminal coordinate round-best config",
        )
        selected_id = one_config_id(
            round_metric.get("selected_config_id"),
            "an invalid terminal coordinate selected config",
        )
        beam_ids = config_ids(
            round_metric.get("beam_config_ids"),
            "invalid terminal coordinate beam IDs",
        )
        for config_id in beam_ids:
            one_config_id(
                config_id, "a terminal coordinate beam member outside the manifest"
            )
        if len(measured_ids) < 2:
            if (
                comparison_ids
                or round_metric.get("measurement") is not None
                or round_best_id != incumbent_id
                or selected_id != incumbent_id
                or round_metric.get("accepted") is not False
                or round_metric.get("improvement_fraction") != 0.0
                or beam_ids != parent_ids
                or round_index != len(rounds)
            ):
                fail("an inconsistent no-candidate terminal coordinate round")
            stopped_for_no_candidates = True
        else:
            if comparison_ids != measured_ids:
                fail("a terminal coordinate comparison that omits a live candidate")
            medians = _validate_flash_terminal_measurement(
                round_metric.get("measurement"),
                expected_base_order=comparison_ids,
                expected_target_ms=_FLASH_TERMINAL_REFINEMENT_TARGET_MS,
                fail=fail,
            )
            for config_id in successful_result_ids:
                selection_perf = results_by_id[config_id].get("selection_perf")
                if not math.isclose(
                    cast("float", selection_perf),
                    medians[config_id],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    fail("a terminal candidate result with a stale selection median")
            expected_round_best_id = min(
                comparison_ids, key=lambda config_id: (medians[config_id], config_id)
            )
            improvement = 1.0 - (
                medians[expected_round_best_id] / medians[incumbent_id]
            )
            accepted = bool(
                expected_round_best_id != incumbent_id
                and improvement >= cast("float", policy["minimum_improvement_fraction"])
            )
            expected_selected_id = expected_round_best_id if accepted else incumbent_id
            raw_improvement = round_metric.get("improvement_fraction")
            if (
                round_best_id != expected_round_best_id
                or selected_id != expected_selected_id
                or round_metric.get("accepted") is not accepted
                or isinstance(raw_improvement, bool)
                or not isinstance(raw_improvement, (int, float))
                or not math.isclose(
                    float(raw_improvement),
                    improvement,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                fail("an invalid terminal coordinate improvement decision")
            ranked_ids = sorted(
                comparison_ids, key=lambda config_id: (medians[config_id], config_id)
            )
            expected_beam_ids = [expected_selected_id]
            for config_id in ranked_ids:
                if config_id in expected_beam_ids:
                    continue
                expected_beam_ids.append(config_id)
                if len(expected_beam_ids) >= cast("int", policy["beam_width"]):
                    break
            if beam_ids != expected_beam_ids:
                fail("a terminal coordinate beam that does not match measured ranking")
            if accepted and selected_id not in accepted_seen:
                accepted_seen.add(selected_id)
                expected_accepted_ids.append(selected_id)
        expected_incumbent_id = selected_id
        expected_parent_ids = beam_ids

    if stopped_for_no_candidates:
        if terminal.get("termination_reason") != "no_candidates":
            fail("a terminal coordinate no-candidate round with the wrong termination")
    elif (
        len(rounds) != cast("int", policy["rounds"])
        or terminal.get("termination_reason") != "round_limit"
    ):
        fail("a terminal coordinate refinement that stopped before its round limit")
    if (
        refined_id != expected_incumbent_id
        or accepted_ids != expected_accepted_ids
        or terminal.get("projection_parent_count") != projection_parent_count
        or terminal.get("projection_attempt_count") != projection_attempt_count
        or terminal.get("unique_candidate_count") != len(all_unique_candidates)
        or terminal.get("new_candidate_count") != len(all_new_candidates)
        or terminal.get("reused_candidate_count") != len(all_reused_candidates)
        or terminal.get("intra_terminal_reused_candidate_count")
        != len(all_intra_terminal_reused)
        or terminal.get("prior_failed_candidate_count") != len(all_prior_failed)
    ):
        fail("inconsistent terminal coordinate aggregate accounting")
    final_tested = trial.get("num_configs_tested")
    if type(final_tested) is not int or not cast("int", preterminal_tested) <= cast(
        "int", final_tested
    ) <= cast("int", preterminal_tested) + len(all_new_candidates):
        fail("inconsistent terminal coordinate benchmark accounting")

    final_beam_ids = expected_parent_ids
    expected_confirmation_ids: list[str] = []
    for config_id in [initial_id, *expected_accepted_ids, *final_beam_ids]:
        if config_id not in expected_confirmation_ids:
            expected_confirmation_ids.append(config_id)
    raw_confirmation = terminal.get("confirmation")
    confirmation_fields = {
        "candidate_config_ids",
        "measurement",
        "best_config_id",
        "selected_config_id",
        "accepted",
        "improvement_fraction",
        "skipped_reason",
    }
    if (
        not isinstance(raw_confirmation, dict)
        or set(raw_confirmation) != confirmation_fields
    ):
        fail("a malformed terminal coordinate confirmation")
    confirmation = cast("dict[str, object]", raw_confirmation)
    confirmation_ids = config_ids(
        confirmation.get("candidate_config_ids"),
        "invalid terminal coordinate confirmation IDs",
    )
    for config_id in confirmation_ids:
        one_config_id(
            config_id, "a terminal confirmation candidate outside the manifest"
        )
    confirmation_best_id = one_config_id(
        confirmation.get("best_config_id"),
        "an invalid terminal confirmation best config",
    )
    confirmation_selected_id = one_config_id(
        confirmation.get("selected_config_id"),
        "an invalid terminal confirmation selected config",
    )
    if confirmation_ids != expected_confirmation_ids:
        fail("a terminal confirmation with an inconsistent candidate set")
    if len(confirmation_ids) == 1:
        if (
            confirmation.get("measurement") is not None
            or confirmation_best_id != initial_id
            or confirmation_selected_id != initial_id
            or confirmation.get("accepted") is not False
            or confirmation.get("improvement_fraction") != 0.0
            or confirmation.get("skipped_reason") != "single_candidate"
        ):
            fail("an inconsistent single-candidate terminal confirmation")
    else:
        confirmation_medians = _validate_flash_terminal_measurement(
            confirmation.get("measurement"),
            expected_base_order=confirmation_ids,
            expected_target_ms=_FLASH_TERMINAL_CONFIRMATION_TARGET_MS,
            fail=fail,
        )
        expected_confirmation_best_id = min(
            confirmation_ids,
            key=lambda config_id: (confirmation_medians[config_id], config_id),
        )
        confirmation_improvement = 1.0 - (
            confirmation_medians[expected_confirmation_best_id]
            / confirmation_medians[initial_id]
        )
        confirmation_accepted = bool(
            expected_confirmation_best_id != initial_id
            and confirmation_improvement
            >= cast("float", policy["minimum_improvement_fraction"])
        )
        expected_confirmation_selected_id = (
            expected_confirmation_best_id if confirmation_accepted else initial_id
        )
        raw_confirmation_improvement = confirmation.get("improvement_fraction")
        if (
            confirmation_best_id != expected_confirmation_best_id
            or confirmation_selected_id != expected_confirmation_selected_id
            or confirmation.get("accepted") is not confirmation_accepted
            or isinstance(raw_confirmation_improvement, bool)
            or not isinstance(raw_confirmation_improvement, (int, float))
            or not math.isclose(
                float(raw_confirmation_improvement),
                confirmation_improvement,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or confirmation.get("skipped_reason") is not None
        ):
            fail("an invalid terminal coordinate confirmation decision")
    selected_config = trial.get("selected_config")
    selected_source_hash = trial.get("selected_source_hash")
    final_manifest_entry = cast("dict[str, object]", manifest[final_id])
    if (
        final_id != confirmation_selected_id
        or not isinstance(selected_config, dict)
        or selected_config != final_manifest_entry["config"]
        or not isinstance(selected_source_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", selected_source_hash) is None
        or (
            final_id != initial_id
            and terminal_source_hashes.get(final_id) != selected_source_hash
        )
    ):
        fail("a final config/source not selected by terminal confirmation")
    if set(manifest) != referenced_ids:
        fail("a terminal coordinate manifest with unreferenced or missing configs")


def _validate_required_full_autotune_trials(
    provenance: dict[str, object],
    trials: list[dict[str, object]],
    *,
    config_spec: ConfigSpec,
    expected_input_shapes: str,
    expected_dtypes: str,
    expected_hardware: str,
    config_generation: ConfigGeneration | None = None,
) -> None:
    if not provenance["require_full_autotune"]:
        return
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.effort_profile import get_effort_profile

    normalization_context = _flash_normalization_context(config_spec)
    normalization_context_json = json.dumps(
        normalization_context, sort_keys=True, separators=(",", ":")
    )
    if (
        provenance.get("flash_normalization_context") != normalization_context
        or provenance.get("flash_normalization_context_sha256")
        != hashlib.sha256(normalization_context_json.encode("utf-8")).hexdigest()
    ):
        raise RuntimeError(
            "required full autotune recorded a mismatched flash normalization context"
        )
    if config_generation is None:
        config_generation = ConfigGeneration(config_spec)
    expected_terminal_policy = _flash_terminal_coordinate_refinement_policy(
        get_effort_profile("full")
    )
    live_terminal_surface_catalog = (
        config_generation.flash_terminal_coordinate_surface_catalog(
            radius=_FLASH_TERMINAL_COORDINATE_RADIUS
        )
    )
    if (
        provenance.get("flash_terminal_coordinate_refinement_policy")
        != expected_terminal_policy
        or provenance.get("flash_terminal_coordinate_refinement_policy_sha256")
        != _canonical_json_sha256(expected_terminal_policy)
        or provenance.get("flash_terminal_coordinate_surface_catalog")
        != live_terminal_surface_catalog
        or provenance.get("flash_terminal_coordinate_surface_catalog_sha256")
        != _canonical_json_sha256(live_terminal_surface_catalog)
    ):
        raise RuntimeError(
            "required full autotune terminal coordinate provenance does not match "
            "the live ConfigGeneration"
        )
    live_compiler_seed_policy = _compiler_seed_policy(config_spec, config_generation)
    if (
        live_compiler_seed_policy.get("kind") != "canonical_cute_flash"
        or provenance.get("compiler_seed_policy") != live_compiler_seed_policy
    ):
        raise RuntimeError(
            "required full autotune compiler-seed provenance does not match the "
            "live canonical CuTe flash heuristic"
        )
    expected_trials = cast("int", provenance["autotune_best_of_k"])
    if len(trials) != expected_trials:
        raise RuntimeError(
            "required full autotune recorded "
            f"{len(trials)} trials, expected {expected_trials}"
        )
    base_seed = provenance.get("autotune_random_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise RuntimeError(
            "required full autotune did not record a valid base random seed"
        )
    parsed_shapes = ast.literal_eval(expected_input_shapes)
    parsed_dtypes = ast.literal_eval(expected_dtypes)
    if (
        not isinstance(parsed_shapes, list)
        or not parsed_shapes
        or not isinstance(parsed_shapes[0], tuple)
        or not parsed_shapes[0]
        or not isinstance(parsed_dtypes, list)
        or not parsed_dtypes
        or not isinstance(parsed_dtypes[0], str)
        or type(parsed_shapes[0][-1]) is not int
    ):
        raise RuntimeError("required full autotune has invalid shape/dtype identity")
    attention_head_dim = cast("int", parsed_shapes[0][-1])
    attention_causal = "causal_attention" in str(
        provenance.get("autotune_baseline_fn", "")
    )
    if (
        normalization_context.get("dtype") != parsed_dtypes[0]
        or normalization_context.get("head_dim") != attention_head_dim
        or normalization_context.get("is_causal") is not attention_causal
    ):
        raise RuntimeError(
            "required full autotune normalization context does not match the trial"
        )
    expected_generations = provenance.get("autotune_lfbo_max_generations")
    if (
        isinstance(expected_generations, bool)
        or not isinstance(expected_generations, int)
        or expected_generations <= 0
    ):
        raise RuntimeError(
            "required full autotune did not record a valid LFBO generation budget"
        )
    initial_population_size = provenance.get("autotune_initial_population_size")
    if (
        isinstance(initial_population_size, bool)
        or not isinstance(initial_population_size, int)
        or initial_population_size <= 0
    ):
        raise RuntimeError(
            "required full autotune did not record a valid initial population size"
        )
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import flash_exp2_packet_is_compound
    from helion.autotuner.search_space_logger import canonical_config_id

    expected_design_configs = [
        dict(config.config)
        for config in config_generation.flash_deterministic_population_configs()
    ]
    expected_design = [
        {
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for config in expected_design_configs
    ]
    expected_leaves = config_generation.flash_structural_leaf_catalog()
    expected_leaf_catalog = [
        {
            "family": leaf.pipeline_family,
            "compound_packet": leaf.compound_exp2_packet,
            "softmax_disc": leaf.softmax_disc,
        }
        for leaf in expected_leaves
    ]
    expected_pipeline_lanes = config_generation.flash_pipeline_lane_catalog()
    expected_active_values = [
        {"key": key, "value": value}
        for key, value in config_generation.flash_structural_coverage_active_values()
    ]
    expected_structural_provenance = {
        "flash_structural_coverage_design": expected_design,
        "flash_structural_coverage_design_count": len(expected_design),
        "flash_structural_coverage_design_sha256": hashlib.sha256(
            json.dumps(
                expected_design_configs, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "flash_structural_coverage_active_values": expected_active_values,
        "flash_structural_coverage_uncovered_values": [
            {"key": key, "value": value}
            for key, value in (
                config_generation.flash_structural_coverage_uncovered_values()
            )
        ],
        "flash_structural_coverage_underqualified_values": [
            {"key": key, "value": value, "witness_count": count}
            for key, value, count in (
                config_generation.flash_structural_coverage_underqualified_values()
            )
        ],
        "flash_structural_leaf_catalog": expected_leaf_catalog,
        "flash_pipeline_lane_catalog": [
            {
                "family": leaf.pipeline_family,
                "compound_packet": leaf.compound_exp2_packet,
                "softmax_disc": leaf.softmax_disc,
                "pipeline_lanes": [
                    {"key": key, "value": value}
                    for key, value in expected_pipeline_lanes[leaf]
                ],
            }
            for leaf in expected_leaves
        ],
        "flash_structural_coverage_underqualified_leaves": [
            {
                "family": leaf.pipeline_family,
                "compound_packet": leaf.compound_exp2_packet,
                "softmax_disc": leaf.softmax_disc,
                "witness_count": count,
            }
            for leaf, count in (
                config_generation.flash_structural_coverage_underqualified_leaves()
            )
        ],
        "flash_structural_coverage_active_interactions": [
            {"keys": list(keys), "values": list(values)}
            for keys, values in (
                config_generation.flash_structural_coverage_active_interactions()
            )
        ],
        "flash_structural_coverage_uncovered_interactions": [
            {"keys": list(keys), "values": list(values)}
            for keys, values in (
                config_generation.flash_structural_coverage_uncovered_interactions()
            )
        ],
        "flash_structural_qualification_values": [
            active
            for active in expected_active_values
            if active["key"] == FLASH_PIPELINE_FAMILY_KEY
            or (
                active["key"] == FLASH_EXP2_PACKET_KEY
                and flash_exp2_packet_is_compound(active["value"])
            )
        ],
        "flash_structural_parent_coverage_prefix_count": (
            config_generation.flash_structural_parent_coverage_prefix_count()
        ),
        "flash_structural_qualification_prefix_count": (
            config_generation.flash_structural_qualification_prefix_count()
        ),
        "flash_structural_population_budget": (
            config_generation.flash_structural_population_budget(
                initial_population_size
            )
        ),
    }
    expected_structural_provenance["flash_structural_injected_design_count"] = min(
        cast(
            "int", expected_structural_provenance["flash_structural_population_budget"]
        ),
        len(expected_design),
    )
    mismatched_structural_fields = [
        key
        for key, value in expected_structural_provenance.items()
        if provenance.get(key) != value
    ]
    if mismatched_structural_fields:
        raise RuntimeError(
            "required full autotune structural provenance does not match the live "
            "ConfigGeneration: " + ", ".join(mismatched_structural_fields)
        )
    expected_clc_catalog = _flash_clc_lane_provenance(config_generation)
    if provenance.get("flash_clc_lane_catalog") != expected_clc_catalog:
        raise RuntimeError(
            "required full autotune CLC provenance does not match the live "
            "ConfigGeneration"
        )
    live_exact_configs = config_generation.flash_exact_effective_search_space_configs(
        initial_population_size
    )
    live_exact_ids = (
        None
        if live_exact_configs is None
        else [canonical_config_id(config) for config in live_exact_configs]
    )
    exact_size = provenance.get("flash_exact_effective_search_space_size")
    exact_config_ids = provenance.get("flash_exact_effective_search_space_config_ids")
    exact_sha256 = provenance.get("flash_exact_effective_search_space_sha256")
    expected_exact_size = None if live_exact_ids is None else len(live_exact_ids)
    expected_exact_sha256 = (
        None
        if live_exact_ids is None
        else hashlib.sha256(
            json.dumps(live_exact_ids, separators=(",", ":")).encode()
        ).hexdigest()
    )
    if (
        exact_size != expected_exact_size
        or exact_config_ids != live_exact_ids
        or exact_sha256 != expected_exact_sha256
    ):
        raise RuntimeError(
            "required full autotune effective-space provenance does not match the "
            "live ConfigGeneration"
        )
    if exact_size is None:
        if exact_config_ids is not None or exact_sha256 is not None:
            raise RuntimeError(
                "required full autotune recorded inconsistent effective-space provenance"
            )
        exhaustive_config_ids: list[str] | None = None
        expected_initial_population_count = initial_population_size
    else:
        if (
            isinstance(exact_size, bool)
            or not isinstance(exact_size, int)
            or exact_size <= 0
            or exact_size > initial_population_size
            or not isinstance(exact_config_ids, list)
            or len(exact_config_ids) != exact_size
            or len(set(exact_config_ids)) != exact_size
            or any(
                not isinstance(config_id, str)
                or re.fullmatch(r"[0-9a-f]{16}", config_id) is None
                for config_id in exact_config_ids
            )
            or exact_sha256
            != hashlib.sha256(
                json.dumps(exact_config_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ):
            raise RuntimeError(
                "required full autotune recorded invalid exact effective search space"
            )
        exhaustive_config_ids = cast("list[str]", exact_config_ids)
        expected_initial_population_count = exact_size
    distance = provenance.get(
        "selected_config_nearest_structural_coverage_design_field_distance",
        provenance.get(
            "selected_config_nearest_structural_coverage_field_distance",
            provenance.get("selected_config_nearest_anchor_field_distance"),
        ),
    )
    nearest_hashes = provenance.get(
        "selected_config_nearest_structural_coverage_design_config_sha256",
        provenance.get(
            "selected_config_nearest_structural_coverage_config_sha256",
            provenance.get("selected_config_nearest_anchor_config_sha256"),
        ),
    )
    is_coverage_design_member = provenance.get(
        "selected_config_is_structural_coverage_design_member",
        provenance.get(
            "selected_config_is_structural_coverage_candidate",
            provenance.get("selected_config_is_deterministic_anchor"),
        ),
    )
    if (
        not isinstance(is_coverage_design_member, bool)
        or not isinstance(distance, int)
        or distance < 0
        or not isinstance(nearest_hashes, list)
        or not nearest_hashes
    ):
        raise RuntimeError(
            "required full autotune did not record complete "
            "winner-to-structural-coverage-design provenance"
        )
    injected_design_count = provenance.get("flash_structural_injected_design_count")
    coverage_design = _structural_coverage_design(provenance)
    if (
        isinstance(injected_design_count, bool)
        or not isinstance(injected_design_count, int)
        or not 0 <= injected_design_count <= len(coverage_design)
    ):
        raise RuntimeError(
            "required full autotune did not record a valid injected structural design"
        )
    from helion.runtime.config import Config

    expected_injected_config_ids = {
        canonical_config_id(
            Config.from_dict(cast("dict[str, object]", candidate["config"]))
        )
        for candidate in coverage_design[:injected_design_count]
    }
    if exhaustive_config_ids is not None:
        expected_injected_config_ids.update(exhaustive_config_ids)
    for index, trial in enumerate(trials, 1):
        expected_seed = base_seed + index - 1
        trial_seed = trial.get("random_seed")
        if type(trial_seed) is not int or trial_seed != expected_seed:
            raise RuntimeError(
                f"required full autotune trial {index} recorded random seed "
                f"{trial_seed!r}, expected {expected_seed}"
            )
        isolated_rebenchmark_timeouts = trial.get("num_isolated_rebenchmark_timeouts")
        if (
            type(isolated_rebenchmark_timeouts) is not int
            or isolated_rebenchmark_timeouts < 0
        ):
            raise RuntimeError(
                f"required full autotune trial {index} recorded an invalid isolated "
                "rebenchmark timeout count"
            )
        replayed_initial_config_ids = (
            _replay_strict_attention_initial_population_config_ids(
                config_generation,
                random_seed=trial_seed,
                initial_population_size=initial_population_size,
            )
        )
        if (
            len(replayed_initial_config_ids) != expected_initial_population_count
            or not expected_injected_config_ids <= set(replayed_initial_config_ids)
        ):
            raise RuntimeError(
                f"required full autotune trial {index} replayed an inconsistent "
                "generation-zero population"
            )
        tested = cast("int", trial["num_configs_tested"])
        source_deduplications = cast("int", trial["num_source_deduplications"])
        required_candidates = (
            initial_population_size
            if exhaustive_config_ids is None
            else len(exhaustive_config_ids)
        )
        if tested + source_deduplications < required_candidates:
            raise RuntimeError(
                f"required full autotune trial {index} covered only "
                f"{tested + source_deduplications}/{required_candidates} effective "
                "initial candidates"
            )
        successful = cast("int", trial["num_successful_candidate_measurements"])
        if exhaustive_config_ids is None and successful < required_candidates:
            raise RuntimeError(
                f"required full autotune trial {index} recorded only {successful} "
                "successful candidate measurements; expected at least "
                f"{required_candidates} for the large effective search space"
            )
        if cast("int", trial["num_unique_sources"]) <= 0:
            raise RuntimeError(
                f"required full autotune trial {index} recorded no unique generated sources"
            )
        generations = trial.get("num_generations")
        generations_valid = type(generations) is int and (
            0 <= generations <= expected_generations
            if exhaustive_config_ids is not None
            else generations == expected_generations
        )
        if not generations_valid:
            raise RuntimeError(
                f"required full autotune trial {index} recorded {generations!r} "
                "search generations, expected "
                + (
                    f"at most {expected_generations} after exact-space exhaustion"
                    if exhaustive_config_ids is not None
                    else f"the full LFBO budget of {expected_generations}"
                )
            )
        if not trial["selected_source_was_measured"]:
            raise RuntimeError(
                f"required full autotune trial {index} did not link its winner to a "
                "measured generated source"
            )
        phase = trial.get("search_phase_metrics")
        if not isinstance(phase, dict):
            raise RuntimeError(
                f"required full autotune trial {index} did not record structural "
                "qualification metrics"
            )
        expected_identity = {
            "input_shapes": expected_input_shapes,
            "dtypes": expected_dtypes,
            "hardware": expected_hardware,
        }
        for metadata_field, expected in expected_identity.items():
            if trial.get(metadata_field) != expected:
                raise RuntimeError(
                    f"required full autotune trial {index} recorded mismatched "
                    f"{metadata_field}: expected {expected!r}, got "
                    f"{trial.get(metadata_field)!r}"
                )
        (
            isolated_timeout_source_hashes,
            invalidated_config_ids,
        ) = _validate_flash_structural_qualification_phase(
            provenance,
            phase,
            trial_index=index,
            expected_initial_config_ids=replayed_initial_config_ids,
            expected_initial_population_count=expected_initial_population_count,
            config_generation=config_generation,
        )
        _validate_flash_terminal_coordinate_refinement(
            provenance,
            phase,
            trial,
            trial_index=index,
            config_generation=config_generation,
        )
        _validate_compiler_seed_generation_zero(
            live_compiler_seed_policy,
            phase["initial_results"],
            trial_index=index,
            invalidated_config_ids=invalidated_config_ids,
        )
        if isolated_rebenchmark_timeouts < len(isolated_timeout_source_hashes):
            raise RuntimeError(
                f"required full autotune trial {index} recorded fewer isolated "
                "rebenchmark timeouts than distinct timed-out generated sources"
            )
        if exhaustive_config_ids is not None:
            measured_initial = {
                cast("str", result["config_id"]): result["status"]
                for result in cast("list[dict[str, object]]", phase["initial_results"])
            }
            unsuccessful = {
                config_id: measured_initial.get(config_id)
                for config_id in exhaustive_config_ids
                if measured_initial.get(config_id) not in {"ok", "deduplicated"}
            }
            if unsuccessful:
                raise RuntimeError(
                    f"required full autotune trial {index} did not successfully "
                    f"exhaust its exact effective search space: {unsuccessful}"
                )
    algorithms = {str(trial["search_algorithm"]) for trial in trials}
    if algorithms != {"LFBOTreeSearch"}:
        raise RuntimeError(
            "required full autotune used unexpected search algorithms: "
            + ", ".join(sorted(algorithms))
        )
    selected_source = provenance.get("selected_source_sha256")
    selected_config = provenance.get("selected_config")
    if not any(
        trial["selected_source_hash"] == selected_source
        and trial["selected_config"] == selected_config
        for trial in trials
    ):
        raise RuntimeError(
            "required full autotune returned a config/source that was not the measured "
            "winner of any recorded trial"
        )


def _validate_required_full_autotune_correctness(
    provenance: dict[str, object], accuracy: str
) -> None:
    if not provenance["require_full_autotune"]:
        return
    if accuracy != "PASS":
        raise RuntimeError(
            "required full autotune selected a kernel that failed final correctness"
        )
    if provenance.get("final_correctness_launches") != (
        _STRICT_FINAL_CORRECTNESS_LAUNCHES
    ):
        raise RuntimeError(
            "required full autotune did not run the standard repeated final "
            "correctness check"
        )
    if not provenance.get("final_repeatability_passed"):
        raise RuntimeError(
            "required full autotune selected a kernel that failed exact repeatability"
        )


def _write_json_output(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload) + "\n")


def _write_strict_prevalidation_output(
    args: argparse.Namespace, provenance: dict[str, object]
) -> None:
    """Persist completed search evidence before strict in-process validation."""
    if not args.json_output or not provenance.get("require_full_autotune"):
        return
    result_path = Path(args.json_output)
    output_path = result_path.with_name(f"{result_path.stem}.strict-prevalidation.json")
    payload = {
        "schema_version": 1,
        "status": "autotune_complete_prevalidation",
        "autotune_provenance": provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle)
        handle.write("\n")
    try:
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _preserve_streamed_strict_prevalidation(
    args: argparse.Namespace, temporary_directory: Path
) -> None:
    """Copy child prevalidation evidence beside the caller-owned JSON output."""
    json_output = getattr(args, "json_output", None)
    if not json_output:
        return
    result_path = Path(json_output)
    for source_path in sorted(
        temporary_directory.glob("result*.strict-prevalidation.json")
    ):
        suffix = source_path.name.removeprefix("result")
        output_path = result_path.with_name(f"{result_path.stem}{suffix}")
        if source_path.resolve() == output_path.resolve():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(source_path.read_bytes())
        try:
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _make_helion_config(
    args: argparse.Namespace,
    compiler_seed_config: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    config_overrides = dict(getattr(args, "helion_config", ()))
    if not args.helion_force_flash_config and not config_overrides:
        return None, config_overrides

    config: dict[str, object] = {}
    if args.helion_force_flash_config:
        if compiler_seed_config is not None:
            config.update(compiler_seed_config)
        else:
            config["block_sizes"] = [1, 128, 128]
    config.update(config_overrides)
    return config, config_overrides


def _compiler_flash_seed_config(
    bound: object, backend: str
) -> dict[str, object] | None:
    if backend != "cute":
        return None
    config_spec = cast("_BoundWithConfigSpec", bound).config_spec
    if config_spec.compiler_default_config is not None:
        default_config = dict(config_spec.default_config().config)
        if any(key.startswith("cute_flash_") for key in default_config):
            return default_config
    for seed in config_spec.compiler_seed_configs:
        seed_config = dict(seed.config)
        if any(key.startswith("cute_flash_") for key in seed_config):
            return seed_config
    return None


def _benchmark_sdpa(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    fn = lambda: _attention_output_reference(args, q, k, v, bias=bias)  # noqa: E731
    # sdpa is the reference, so it is "PASS" by definition.
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    ):
        stats = _bench_steady(
            fn,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    return _result(
        "sdpa",
        args,
        stats,
        accuracy="PASS",
        benchmark_timer="event",
        config=None,
        notes=[
            "Forced torch SDPBackend.CUDNN_ATTENTION.",
            *(
                ["Timed eager torch.relu after SDPA; ReLU FLOPs are excluded."]
                if _epilogue(args) == "relu"
                else []
            ),
        ],
    )


def _manifest_text(manifest: dict[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"KernelAgent manifest is missing nonempty {key!r}")
    return value


def _kernelagent_version_info(
    impl: str,
    manifest: dict[str, Any],
    *,
    evaluation_backend_version: str | None,
) -> dict[str, str]:
    model = _manifest_text(manifest, "model")
    display_version = _manifest_text(manifest, "kernelagent_display_version")
    model_display_name = _manifest_text(manifest, "model_display_name")
    if impl in _KERNELAGENT_CLOSED_IMPLS:
        agent_version = _manifest_text(manifest, "kernelagent_version")
        selected_backend_version = _manifest_text(manifest, "cutlass_dsl_version")
        backend_name = "CuTe"
    else:
        agent_version = manifest.get("kernelagent_version")
        if not isinstance(agent_version, str) or not agent_version:
            commit = _manifest_text(manifest, "kernelagent_commit")
            agent_version = f"commit {commit[:8]}"
        selected_backend_version = _manifest_text(manifest, "triton_version")
        backend_name = "Triton"

    measured_backend_version = evaluation_backend_version or selected_backend_version
    backend_label_version = (
        measured_backend_version.split("+", 1)[0]
        if backend_name == "Triton"
        else measured_backend_version
    )
    selection_suffix = (
        f"; selected with {backend_name} {selected_backend_version}"
        if selected_backend_version != measured_backend_version
        else ""
    )
    return {
        "version": (
            f"KernelAgent {agent_version}; model {model}; {backend_name} "
            f"{measured_backend_version}{selection_suffix}"
        ),
        "version_label": (
            f"KernelAgent {display_version} / {model_display_name} / {backend_name} "
            f"{backend_label_version}"
        ),
    }


def _declared_kernelagent_source_hashes(
    manifest: dict[str, Any],
) -> dict[str, str]:
    declared: dict[str, str] = {}
    for field, container in (
        ("source_sha256", manifest),
        ("selection.source_sha256", manifest.get("selection")),
        (
            "posthoc_correctness_validation.source_sha256",
            manifest.get("posthoc_correctness_validation"),
        ),
    ):
        if isinstance(container, dict):
            value = container.get("source_sha256")
            if isinstance(value, str) and value:
                declared[field] = value
    return declared


def _validate_kernelagent_manifest(
    impl: str,
    manifest: object,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SystemExit(f"KernelAgent manifest is not an object in {run_dir}")

    budget_label = _KERNELAGENT_BUDGET_LABELS[impl]
    physical_gpu = _physical_gpu_selection()
    power_cap_w = getattr(args, "power_cap_w", None)
    if not physical_gpu:
        raise SystemExit("KernelAgent evaluation requires CUDA_VISIBLE_DEVICES")
    if power_cap_w is None:
        raise SystemExit("KernelAgent evaluation requires --power-cap-w")
    expected_manifest: dict[str, object] = {
        "budget_label": budget_label,
        "shape": _shape_dict(args),
        "physical_gpu": physical_gpu,
        "power_cap_w": power_cap_w,
        "seed": args.seed,
    }
    if impl in _KERNELAGENT_CLOSED_IMPLS:
        expected_manifest["kernelagent_family"] = "closed_binary"

    mismatches: dict[str, tuple[object, object]] = {}
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        if key == "physical_gpu" and actual is not None:
            actual = str(actual)
        exact_shape_match = (
            key != "shape"
            or isinstance(actual, dict)
            and isinstance(expected, dict)
            and actual.keys() == expected.keys()
            and all(
                type(actual[field]) is type(expected[field])
                and actual[field] == expected[field]
                for field in expected
            )
        )
        if actual != expected or not exact_shape_match:
            mismatches[key] = (actual, expected)
    if mismatches:
        raise SystemExit(f"KernelAgent manifest mismatch in {run_dir}: {mismatches}")

    _manifest_text(manifest, "kernelagent_display_version")
    _manifest_text(manifest, "model_display_name")
    return manifest


def _kernelagent_run_dir(args: argparse.Namespace) -> Path:
    if args.impl in _KERNELAGENT_CLOSED_IMPLS:
        root_value = args.kernelagent_closed_results_root or os.environ.get(
            _KERNELAGENT_CLOSED_RESULTS_ROOT_ENV
        )
        option = "--kernelagent-closed-results-root"
        environment_variable = _KERNELAGENT_CLOSED_RESULTS_ROOT_ENV
    else:
        root_value = args.kernelagent_results_root or os.environ.get(
            _KERNELAGENT_RESULTS_ROOT_ENV
        )
        option = "--kernelagent-results-root"
        environment_variable = _KERNELAGENT_RESULTS_ROOT_ENV
    if not root_value:
        raise SystemExit(
            f"KernelAgent results root is required; pass {option} or set "
            f"{environment_variable}"
        )
    variant = "causal" if args.causal else "dense"
    budget_label = _KERNELAGENT_BUDGET_LABELS[args.impl]
    return Path(root_value).expanduser().resolve() / (
        f"{variant}_{args.seq_len}_{budget_label}"
    )


def _benchmark_kernelagent(args: argparse.Namespace) -> dict[str, Any]:
    impl = str(args.impl)
    if (
        args.z != 2
        or args.h != 32
        or args.head_dim != 64
        or args.dtype != "float16"
        or _uses_bias(args)
        or _epilogue(args) != "none"
    ):
        return _skipped_result(
            impl,
            args,
            "KernelAgent artifacts cover only identity-epilogue, output-only "
            "FP16 B=2 H=32 D=64",
        )

    run_dir = _kernelagent_run_dir(args)
    source_path = run_dir / "selected_kernel.py"
    if not source_path.is_file():
        source_path = run_dir / "selected_kernel.py.txt"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"incomplete KernelAgent result directory: {run_dir}")

    manifest = _validate_kernelagent_manifest(
        impl,
        json.loads(manifest_path.read_text()),
        args,
        run_dir,
    )
    budget_label = _KERNELAGENT_BUDGET_LABELS[impl]

    selection_version_info = _kernelagent_version_info(
        impl, manifest, evaluation_backend_version=None
    )

    if impl in _KERNELAGENT_CLOSED_IMPLS and manifest.get("status") == "FAIL":
        reason = str(manifest.get("failure_reason", "KernelAgent campaign failed"))
        selection_cute_version = str(manifest.get("cutlass_dsl_version", "unknown"))
        return {
            "impl": impl,
            **selection_version_info,
            "shape": _shape_dict(args),
            "gpu": _gpu_name(),
            "physical_gpu": _physical_gpu_selection(),
            "power_cap_w": getattr(args, "power_cap_w", None),
            "flop_model": "softmax_attention_forward",
            "accuracy": "FAIL",
            "error": reason,
            "config": {
                "budget_label": budget_label,
                "budget_seconds": manifest["budget_seconds"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "selection_cute_version": selection_cute_version,
                "evaluation_cute_version": None,
            },
            "notes": [
                reason,
                (
                    f"Campaign ran with CuTe {selection_cute_version}; no source "
                    "was selected, so there was no kernel to re-evaluate."
                ),
            ],
        }
    if not source_path.is_file():
        raise SystemExit(f"incomplete KernelAgent result directory: {run_dir}")

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    declared_hashes = _declared_kernelagent_source_hashes(manifest)
    if not declared_hashes:
        raise SystemExit(
            f"KernelAgent manifest has no declared source hash in {run_dir}"
        )
    hash_mismatches = {
        field: declared_hash
        for field, declared_hash in declared_hashes.items()
        if declared_hash != source_hash
    }
    if hash_mismatches:
        raise SystemExit(
            f"KernelAgent source hash mismatch in {run_dir}: loaded {source_hash}, "
            f"manifest declares {hash_mismatches}"
        )
    family = "closed" if impl in _KERNELAGENT_CLOSED_IMPLS else "public"
    module_name = (
        f"_kernelagent_{family}_{budget_label}_{args.seq_len}_{source_hash[:12]}"
    )
    if source_path.suffix == ".txt":
        loader = importlib.machinery.SourceFileLoader(module_name, str(source_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
    else:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load KernelAgent kernel: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _scrubbed_argv():
        spec.loader.exec_module(module)

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    run = cast("Callable[..., torch.Tensor]", module.kernel_function)
    evaluation_backend_version = _package_version(
        "nvidia-cutlass-dsl" if impl in _KERNELAGENT_CLOSED_IMPLS else "triton"
    )
    version_info = _kernelagent_version_info(
        impl,
        manifest,
        evaluation_backend_version=evaluation_backend_version,
    )

    def fn() -> torch.Tensor:
        return run(q, k, v)

    standard_correctness_executed = False
    repeat_determinism_executed = False
    stress_correctness_executed = False
    with _scrubbed_argv():
        accuracy = "PASS"
        if not args.skip_correctness:
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
            ):
                expected = _sdpa_reference(q, k, v, causal=bool(args.causal))
            standard_correctness_executed = True
            actual = fn()
            accuracy = "PASS" if _check_kernelagent_output(actual, expected) else "FAIL"
            if accuracy == "PASS":
                repeat_determinism_executed = True
                repeated = fn()
                accuracy = (
                    "PASS"
                    if _check_kernelagent_output(repeated, expected)
                    and _check_kernelagent_repeat(actual, repeated)
                    else "FAIL"
                )
            if accuracy == "PASS":
                stress_correctness_executed = True
                accuracy = (
                    "PASS"
                    if _check_kernelagent_stress_case(run, args, dtype)
                    else "FAIL"
                )
        stats = None
        if accuracy == "PASS":
            stats = _bench_steady(
                fn,
                num_runs=args.num_runs,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )

    selection = manifest.get("selection", {})
    evaluation_provenance: dict[str, object] = {
        "standard_correctness_executed": standard_correctness_executed,
        "repeat_determinism_executed": repeat_determinism_executed,
        "stress_correctness_executed": stress_correctness_executed,
    }
    if impl in _KERNELAGENT_CLOSED_IMPLS:
        selection_cute_version = str(manifest.get("cutlass_dsl_version", "unknown"))
        evaluation_cute_version = evaluation_backend_version
        evaluation_provenance.update(
            {
                "selection_cute_version": selection_cute_version,
                "evaluation_cute_version": evaluation_cute_version,
            }
        )
        selection_name = f"candidate {selection.get('candidate_id', 'unknown')}"
        internal_time = selection.get("median_ms", "unknown")
        notes = [
            (
                "KernelAgent Closed wall-clock tuning budget "
                f"{manifest['budget_seconds']:.1f}s ({budget_label}); elapsed "
                f"{manifest['elapsed_seconds']:.1f}s."
            ),
            (
                f"Selected source sha256 {source_hash}; {selection_name}; "
                f"internal search time {internal_time} ms."
            ),
            _kernelagent_evaluation_note(
                "CuTe",
                selection_cute_version,
                evaluation_cute_version,
                standard_executed=standard_correctness_executed,
                repeat_executed=repeat_determinism_executed,
                stress_executed=stress_correctness_executed,
                passed=accuracy == "PASS",
                measured=stats is not None,
            ),
        ]
    else:
        selection_triton_version = _manifest_text(manifest, "triton_version")
        evaluation_provenance.update(
            {
                "selection_triton_version": selection_triton_version,
                "evaluation_triton_version": evaluation_backend_version,
            }
        )
        notes = [
            (
                f"KernelAgent Public wall-clock tuning budget "
                f"{manifest['budget_seconds']:.1f}s ({budget_label}); elapsed "
                f"{manifest['elapsed_seconds']:.1f}s."
            ),
            (
                f"Selected source sha256 {source_hash}; program "
                f"{selection.get('program_id', 'unknown')}; internal time "
                f"{selection.get('internal_time_ms', 'unknown')} ms."
            ),
            _kernelagent_evaluation_note(
                "Triton",
                selection_triton_version,
                evaluation_backend_version,
                standard_executed=standard_correctness_executed,
                repeat_executed=repeat_determinism_executed,
                stress_executed=stress_correctness_executed,
                passed=accuracy == "PASS",
                measured=stats is not None,
            ),
        ]
    result = _result(
        impl,
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config={
            "budget_label": budget_label,
            "budget_seconds": manifest["budget_seconds"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "selection": selection,
            "source_sha256": source_hash,
            **evaluation_provenance,
        },
        notes=notes,
        version_info=version_info,
    )
    if accuracy == "FAIL":
        if impl in _KERNELAGENT_CLOSED_IMPLS:
            result["error"] = (
                "Selected KernelAgent source failed final-harness correctness "
                f"under CuTe {evaluation_backend_version}."
            )
        else:
            result["error"] = (
                "Selected KernelAgent source failed final-harness correctness."
            )
    return result


def _benchmark_flexattention(args: argparse.Namespace) -> dict[str, Any]:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    impl = str(args.impl)
    backend = _FLEXATTENTION_BACKENDS[impl]
    epilogue = _epilogue(args)
    with _scrubbed_argv():
        if backend == "FLASH":
            _import_fa4()
        flex_attention_impl = cast("Callable[..., torch.Tensor]", flex_attention)
        compile_target = flex_attention_impl
        if epilogue == "relu":

            def flex_attention_with_relu(
                *inputs: torch.Tensor, **kwargs: object
            ) -> torch.Tensor:
                return torch.relu(flex_attention_impl(*inputs, **kwargs))

            compile_target = flex_attention_with_relu
        compiled = cast(
            "Callable[..., torch.Tensor]",
            torch.compile(compile_target, fullgraph=True),
        )
        kernel_options = {"BACKEND": backend}

        if causal:
            compiled_create_block_mask = torch.compile(create_block_mask)

            def causal_mask(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                return q_idx >= kv_idx

            # Eager mask creation materializes the full token-level mask before
            # reducing it to blocks, which is infeasible for the long suite.
            block_mask = compiled_create_block_mask(
                causal_mask,
                None,
                None,
                args.seq_len,
                args.seq_len,
                device=q.device,
                BLOCK_SIZE=256 if backend == "FLASH" else 128,
            )
            fn = lambda: compiled(  # noqa: E731
                q, k, v, block_mask=block_mask, kernel_options=kernel_options
            )
        elif bias is not None:
            bias_tensor = bias

            def bias_score_mod(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                return score + bias_tensor[b, h, q_idx, kv_idx]

            fn = lambda: compiled(  # noqa: E731
                q,
                k,
                v,
                score_mod=bias_score_mod,
                kernel_options=kernel_options,
            )
        else:
            fn = lambda: compiled(  # noqa: E731
                q, k, v, kernel_options=kernel_options
            )

        accuracy = "PASS"
        if not args.skip_correctness:
            expected = _attention_output_reference(args, q, k, v, bias=bias)
            out = fn()
            accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"
            del expected, out
        stats = _bench_steady(
            fn,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    return _result(
        impl,
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config=None,
        notes=[
            f"Forced PyTorch FlexAttention BACKEND={backend!r}.",
            *(
                ["torch.compile(fullgraph=True) captured FlexAttention and ReLU."]
                if epilogue == "relu"
                else []
            ),
        ],
    )


def _import_gluon_attention() -> types.ModuleType:
    path = _resolve_gluon_attention_path()
    module_name = "_helion_gluon_attention_forward"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load Gluon attention example from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _scrubbed_argv():
        spec.loader.exec_module(module)
    return module


def _import_tlx_attention() -> types.ModuleType:
    configured = os.environ.get(_TLX_ATTENTION_PATH_ENV)
    if configured:
        path = _resolve_tlx_attention_path()
        module_name = "_helion_tlx_blackwell_attention"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"unable to load TLX attention example from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _scrubbed_argv():
            spec.loader.exec_module(module)
        return module
    with _scrubbed_argv():
        return importlib.import_module(_TLX_ATTENTION_MODULE)


def _benchmark_gluon(args: argparse.Namespace) -> dict[str, Any]:
    if _epilogue(args) != "none":
        return _skipped_result(
            "gluon", args, "Gluon attention harness does not support output epilogues"
        )
    if _uses_bias(args):
        return _skipped_result(
            "gluon",
            args,
            "Gluon attention example does not support additive score bias",
        )

    module = _import_gluon_attention()
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    causal = bool(args.causal)
    sm_scale = args.head_dim**-0.5

    def run() -> torch.Tensor:
        out, _lse = module.attention_forward(
            q, k, v, causal, sm_scale, use_tmem_red=False
        )
        return out

    accuracy = "PASS"
    if not args.skip_correctness:
        expected = _attention_output_reference(args, q, k, v)
        out = run()
        accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"
        del expected, out
    stats = _bench_steady(
        run,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result(
        "gluon",
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        notes=[
            (
                "The referenced Gluon entry point returns both output and LSE; its "
                "timing therefore includes the auxiliary LSE write."
            )
        ],
    )


def _benchmark_tlx(args: argparse.Namespace) -> dict[str, Any]:
    if _epilogue(args) != "none":
        return _skipped_result(
            "tlx", args, "TLX attention harness does not support output epilogues"
        )
    if _uses_bias(args):
        return _skipped_result(
            "tlx",
            args,
            "TLX Blackwell attention does not support additive score bias",
        )

    module = _import_tlx_attention()
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    causal = bool(args.causal)
    sm_scale = args.head_dim**-0.5

    def run() -> torch.Tensor:
        return cast("torch.Tensor", module.attention(q, k, v, sm_scale, causal))

    accuracy = "PASS"
    if not args.skip_correctness:
        expected = _attention_output_reference(args, q, k, v)
        out = run()
        accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"
        del expected, out
    stats = _bench_steady(
        run,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    best_config = getattr(module._attn_fwd_ws, "best_config", None)
    return _result(
        "tlx",
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config=str(best_config) if best_config is not None else None,
        notes=[
            (
                "Meta TLX Blackwell ws_pipelined_persistent attention used its "
                "upstream shape-keyed autotuner."
            ),
            (
                "The forward kernel writes FP32 softmax state for backward even "
                "though the wrapper returns only the attention output; timing "
                "includes that auxiliary write."
            ),
        ],
    )


def _import_fa4() -> types.ModuleType:
    """Import FlashAttention-4 (``flash_attn.cute.flash_attn_func``) in this env.

    The FA4 checkout and its Quack dependency use CuTe APIs that moved across
    the 4.5 and 4.6 releases. The checkout is resolved lazily from
    HELION_FA4_ROOT, an existing tritonbench submodule, or, when
    HELION_FA4_AUTO_DOWNLOAD=1, an auto-cloned benchmarks/flash-attention tree.
    The compatibility aliases below bridge that API skew, and the top-level
    package must be bypassed:

    * The top-level ``flash_attn/__init__`` eagerly imports the unbuilt FA2 CUDA
      extension (``flash_attn_2_cuda``); register a bare ``flash_attn`` namespace
      package first so only the independent ``flash_attn.cute`` subpackage loads.
    * Inject missing nvvm enums (``ProxyKind`` etc.) into ``cute.arch`` and
      expose ``ThrMma`` and ``ThrCopy`` through their former ``cute.core`` path.
    * FA4's primitive wrappers use the old binding ABI: ``nvvm.fmax`` takes an
      explicit result-type first arg (4.5.1 infers it), the packed-f32x2 ops take
      a ``RoundingModeKind`` enum (4.5.1 wants the ``'rn'`` string), and
      ``fence_proxy`` takes ``ProxyKind``/``SharedSpace`` enums (4.5.1 wants string
      literals). Shim each. These are numerically exact (max/min associativity,
      same rounding mode, same proxy/space).

    Returns the ``flash_attn.cute`` module (``.flash_attn_func`` is the fwd entry).
    """
    import cutlass._mlir.dialects.nvvm as nvvm
    import cutlass.cute as cute

    fa4_root = _resolve_fa4_root()
    for sym in ("ThrMma", "ThrCopy"):
        if not hasattr(cute.core, sym):
            setattr(cute.core, sym, getattr(cute, sym))
    for sym in (
        "ProxyKind",
        "SharedSpace",
        "RoundingModeKind",
        "ReduxKind",
        "AtomicOpKind",
    ):
        if not hasattr(cute.arch, sym) and hasattr(nvvm, sym):
            setattr(cute.arch, sym, getattr(nvvm, sym))

    def _strip_type_arg(orig: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object, **kw: object) -> object:
            if len(args) == 3:  # (result_type, a, b) -> (a, b)
                args = args[1:]
            return orig(*args, **kw)

        return wrapped

    nvvm.fmax = _strip_type_arg(nvvm.fmax)  # pyrefly: ignore[bad-assignment]
    nvvm.fmin = _strip_type_arg(nvvm.fmin)  # pyrefly: ignore[bad-assignment]

    proxy_str = {
        "alias": "alias",
        "async_": "async",
        "async_global": "async.global",
        "async_shared": "async.shared",
        "generic": "generic",
        "tensormap": "tensormap",
    }
    space_str = {"shared_cta": "cta", "shared_cluster": "cluster"}
    orig_fence_proxy = cute.arch.fence_proxy

    def _fence_proxy(kind: object, *, space: object = None, **kw: object) -> object:
        if hasattr(kind, "name"):
            kind = proxy_str.get(kind.name, kind.name)
        if space is not None and hasattr(space, "name"):
            space = space_str.get(space.name, space.name)
        return orig_fence_proxy(kind, space=space, **kw)

    cute.arch.fence_proxy = _fence_proxy

    fa4_root_str = str(fa4_root)
    if fa4_root_str not in sys.path:
        sys.path.insert(0, fa4_root_str)
    if "flash_attn" not in sys.modules:
        pkg = types.ModuleType("flash_attn")
        pkg.__path__ = [str(fa4_root / "flash_attn")]
        pkg.__package__ = "flash_attn"
        sys.modules["flash_attn"] = pkg

    import functools

    import flash_attn.cute as fc  # pyrefly: ignore[missing-import]
    import flash_attn.cute.utils as fu  # pyrefly: ignore[missing-import]

    fu.fma_packed_f32x2 = functools.partial(cute.arch.fma_packed_f32x2, rnd="rn")
    fu.mul_packed_f32x2 = functools.partial(cute.arch.mul_packed_f32x2, rnd="rn")
    fu.add_packed_f32x2 = functools.partial(cute.arch.add_packed_f32x2, rnd="rn")
    fu.sub_packed_f32x2 = functools.partial(
        cute.arch.calc_packed_f32x2_op,
        src_c=None,
        calc_func=nvvm.sub_packed_f32x2,
        rnd="rn",
    )
    return fc


def _benchmark_fa4(args: argparse.Namespace) -> dict[str, Any]:
    """FlashAttention-4 (CuTe-DSL fwd) baseline -- the upstream design target.

    FA4's tensor layout is ``(B, S, H, D)``; our harness builds ``(B, H, S, D)``
    (the SDPA convention), so we transpose in and out. FA4 returns ``(out, lse)``.
    """
    if _uses_bias(args):
        return _skipped_result(
            "fa4", args, "FA4 harness does not support additive score bias"
        )
    fc = _import_fa4()
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    causal = bool(args.causal)
    epilogue = _epilogue(args)

    expected: torch.Tensor | None = None
    if not args.skip_correctness:
        expected = _attention_output_reference(args, q, k, v)

    qt = q.transpose(1, 2).contiguous()  # (B, H, S, D) -> (B, S, H, D)
    del q
    kt = k.transpose(1, 2).contiguous()
    del k
    vt = v.transpose(1, 2).contiguous()
    del v

    def run() -> torch.Tensor:
        out, _lse = fc.flash_attn_func(qt, kt, vt, softmax_scale=None, causal=causal)
        return _apply_output_epilogue(out, epilogue)

    with _scrubbed_argv():
        accuracy = "PASS"
        if expected is not None:
            out = run()  # (B, S, H, D)
            got = out.transpose(1, 2)  # back to (B, H, S, D)
            accuracy = "PASS" if _check_close(got, expected, dtype) else "FAIL"
            del expected, got, out
        stats = _bench_steady(
            run,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    notes = (
        ["Timed eager torch.relu after FA4; ReLU FLOPs are excluded."]
        if epilogue == "relu"
        else None
    )
    return _result(
        "fa4",
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config=None,
        notes=notes,
    )


def _benchmark_tilegym_tileir(args: argparse.Namespace) -> dict[str, Any]:
    """TileGym's handwritten Triton FMHA running through NV Triton TileIR."""
    try:
        fmha, get_best_config = _import_tilegym_fmha()
    except (ImportError, RuntimeError) as exc:
        return {
            "impl": "tilegym-tileir",
            "shape": _shape_dict(args),
            "accuracy": "SKIP",
            "skipped_reason": str(exc),
        }

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    kwargs = _tilegym_attention_kwargs(args, bias)

    def run() -> torch.Tensor:
        return fmha(q, k, v, **kwargs)

    accuracy = "PASS"
    if not args.skip_correctness:
        expected = _sdpa_reference(q, k, v, causal=bool(args.causal), bias=bias)
        accuracy = "PASS" if _check_close(run(), expected, dtype) else "FAIL"

    stats = _bench_steady(
        run,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    best_config = get_best_config()
    config = (
        dict(cast("Any", best_config).all_kwargs()) if best_config is not None else None
    )
    return _result(
        "tilegym-tileir",
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config=config,
        notes=[
            "Kernel source: benchmarks/cute/tilegym_attention.py",
            (
                "TILEIR_ENABLE_APPROX="
                f"{os.environ.get('TILEIR_ENABLE_APPROX', '<unset>')}; "
                "TILEIR_ENABLE_FTZ="
                f"{os.environ.get('TILEIR_ENABLE_FTZ', '<unset>')}"
            ),
        ],
    )


def _helion_benchmark_timer(args: argparse.Namespace, backend: str) -> str:
    if backend == "cute":
        return str(getattr(args, "helion_cute_benchmark_timer", "wall"))
    return "event"


def _validate_strict_helion_benchmark_timer(
    args: argparse.Namespace, backend: str
) -> None:
    if (
        backend == "cute"
        and bool(getattr(args, "helion_require_full_autotune", 0))
        and _helion_benchmark_timer(args, backend) != "wall"
    ):
        raise SystemExit(
            "--helion-require-full-autotune requires "
            "--helion-cute-benchmark-timer=wall because CUDA events can mis-time "
            "CuTe kernels on Blackwell"
        )


def _helion_do_bench_fn(
    bound: object, args: argparse.Namespace, backend: str
) -> Callable[..., Any] | None:
    if _helion_benchmark_timer(args, backend) == "wall":
        return cast("Any", bound).env.backend.get_do_bench()
    return None


@contextlib.contextmanager
def _scrubbed_argv() -> Iterator[None]:
    """Hide our CLI argv from libraries that parse ``sys.argv`` on import/compile.

    The CuTe DSL's ``base_dsl.dsl.diagnostic()`` calls ``parse_known_args()`` on
    the process argv during kernel compilation/launch and aborts the process
    (printing its own ``-diagnostic`` usage banner) when it trips over our
    ``--impl``/``--dtype`` flags. Reducing argv to ``[argv0]`` for the duration
    of the cute work makes that parser a no-op without affecting our own parse,
    which has already completed.
    """
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = saved


def _benchmark_helion(args: argparse.Namespace) -> dict[str, Any]:
    """Helion attention via examples/attention.py.

    Backend is determined by ``args.helion_backend``. The cute
    path may currently be numerically wrong; we never let that crash the
    harness -- we record accuracy=FAIL and still report timing.

    When ``--helion-force-flash-config`` is set, skip autotune and directly use
    the generic compiler-promoted flash seed. This is useful for benchmarking
    individual knob variants without waiting for autotuner search.
    """
    _validate_epilogue_workload(args)
    require_full_autotune = bool(getattr(args, "helion_require_full_autotune", 0))
    _validate_strict_helion_benchmark_timer(
        args, str(getattr(args, "helion_backend", "cute"))
    )
    _validate_strict_gpu_selection(require_full_autotune)
    initial_source: dict[str, object] | None = None
    if require_full_autotune:
        initial_source = _helion_source_provenance()
        _validate_helion_source_checkout(initial_source)

    with _strict_helion_runtime_environment(
        require_full_autotune
    ) as strict_runtime_environment:
        return _benchmark_helion_in_environment(
            args,
            require_full_autotune=require_full_autotune,
            initial_source=initial_source,
            strict_runtime_environment=strict_runtime_environment,
        )


def _benchmark_helion_in_environment(
    args: argparse.Namespace,
    *,
    require_full_autotune: bool,
    initial_source: dict[str, object] | None,
    strict_runtime_environment: dict[str, object] | None,
) -> dict[str, Any]:

    env_overrides = _apply_helion_env(args)
    backend = args.helion_backend
    os.environ["HELION_BACKEND"] = backend
    impl_label = f"helion-{backend}"

    from examples.attention import _attention_baseline
    from examples.attention import _attention_output_baseline
    from examples.attention import _attention_relu_output_baseline
    from examples.attention import _biased_attention_baseline
    from examples.attention import _biased_attention_output_baseline
    from examples.attention import _causal_attention_baseline
    from examples.attention import _causal_attention_output_baseline
    from examples.attention import _causal_attention_relu_output_baseline
    from examples.attention import attention
    from examples.attention import attention_output
    from examples.attention import attention_relu_output
    from examples.attention import biased_attention
    from examples.attention import biased_attention_output
    from examples.attention import causal_attention
    from examples.attention import causal_attention_output
    from examples.attention import causal_attention_relu_output

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    epilogue = _epilogue(args)
    output_only = not bool(args.helion_return_lse)
    if bias is not None:
        kernel = biased_attention_output if output_only else biased_attention
        expected_baseline_fn = (
            _biased_attention_output_baseline
            if output_only
            else _biased_attention_baseline
        )
        kernel_args = (q, k, v, bias)
    elif output_only:
        if epilogue == "relu":
            kernel = causal_attention_relu_output if causal else attention_relu_output
            expected_baseline_fn = (
                _causal_attention_relu_output_baseline
                if causal
                else _attention_relu_output_baseline
            )
        else:
            kernel = causal_attention_output if causal else attention_output
            expected_baseline_fn = (
                _causal_attention_output_baseline
                if causal
                else _attention_output_baseline
            )
        kernel_args = (q, k, v)
    elif causal:
        kernel = causal_attention
        expected_baseline_fn = _causal_attention_baseline
        kernel_args = (q, k, v)
    else:
        kernel = attention
        expected_baseline_fn = _attention_baseline
        kernel_args = (q, k, v)

    seed_config_overrides = dict(getattr(args, "helion_seed_config", ()))
    kernel.settings.autotune_seed_configs = (
        [seed_config_overrides] if seed_config_overrides else None
    )

    with _scrubbed_argv():
        bound = kernel.bind(kernel_args)
        compiler_seed_config = _compiler_flash_seed_config(bound, backend)
        fixed_config, config_overrides = _make_helion_config(args, compiler_seed_config)
        notes: list[str] = []
        if backend == "tileir":
            notes.append(
                "TileIR ran with "
                f"TILEIR_ENABLE_APPROX={os.environ.get('TILEIR_ENABLE_APPROX', '')} "
                f"and TILEIR_ENABLE_FTZ={os.environ.get('TILEIR_ENABLE_FTZ', '')}."
            )
            if env_overrides.get("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS") == "0":
                notes.append(
                    "TileIR autotune measurements ran in the parent process; "
                    "spawned benchmark workers cannot inherit the isolated "
                    "toolchain runtime."
                )
        if bias is not None and backend == "cute":
            biased_seed_config = cast(
                "dict[str, object]",
                compiler_seed_config or {"block_sizes": [1, 128, 128]},
            )
            fixed_config = {**biased_seed_config, **(fixed_config or {})}
            notes.append(
                "CuTe biased attention starts from the fixed 128x128 flash seed "
                "and applies user overrides; biased autotune search is not "
                "characterized yet."
            )
        autotune_provenance = _helion_autotune_provenance(
            args,
            bound,
            fixed_config,
            expected_baseline_fn,
            strict_runtime_environment,
        )
        if require_full_autotune:
            assert initial_source is not None
            _validate_helion_source_unchanged(initial_source, autotune_provenance)
        _validate_required_full_autotune(autotune_provenance)
        autotune_trials: list[dict[str, object]] = []
        if fixed_config is not None:
            bound.set_config(fixed_config)
            active_config = fixed_config
            autotuned = False
        else:
            with _capture_helion_autotune_metrics() as autotune_trials:
                active_config = bound.autotune(
                    kernel_args, force=bool(args.helion_force_autotune)
                )
            autotuned = True

        code = _helion_selected_source_code(bound, active_config)
        codegen = _helion_codegen_markers(code)
        if (
            backend == "cute"
            and epilogue == "relu"
            and not codegen["uses_relu_epilogue"]
        ):
            raise SystemExit(
                "Helion-CuTe did not lower the requested ReLU into the flash "
                "attention epilogue"
            )
        autotune_provenance["trials"] = autotune_trials
        autotune_provenance["selected_source_sha256"] = hashlib.sha256(
            code.encode("utf-8")
        ).hexdigest()
        autotune_provenance["selected_config"] = (
            dict(active_config)
            if isinstance(active_config, dict)
            else dict(active_config.config)
        )
        _record_selected_structural_coverage_design_provenance(autotune_provenance)
        _write_strict_prevalidation_output(args, autotune_provenance)
        input_shape = (args.z, args.h, args.seq_len, args.head_dim)
        _validate_required_full_autotune_trials(
            autotune_provenance,
            autotune_trials,
            config_spec=cast("ConfigSpec", cast("Any", bound).config_spec),
            expected_input_shapes=repr([input_shape] * len(kernel_args)),
            expected_dtypes=repr([str(tensor.dtype) for tensor in kernel_args]),
            expected_hardware=_gpu_name(),
        )

        accuracy = "PASS"
        correctness_launches = 0
        repeatability_passed = False
        if not args.skip_correctness:
            expected = expected_baseline_fn(*kernel_args)
            correctness_launches = (
                _STRICT_FINAL_CORRECTNESS_LAUNCHES
                if autotune_provenance["require_full_autotune"]
                else 1
            )
            accuracy = (
                "PASS"
                if _check_attention_result_repeatedly(
                    lambda: bound(*kernel_args),
                    expected,
                    dtype,
                    launches=correctness_launches,
                )
                else "FAIL"
            )
            del expected
            repeatability_passed = accuracy == "PASS" and correctness_launches > 1
        autotune_provenance["final_correctness_launches"] = correctness_launches
        autotune_provenance["final_repeatability_passed"] = repeatability_passed
        autotune_provenance["final_correctness_passed"] = accuracy == "PASS"
        _validate_required_full_autotune_correctness(autotune_provenance, accuracy)

        fn = lambda: bound(*kernel_args)  # noqa: E731
        benchmark_timer = _helion_benchmark_timer(args, backend)
        stats = _bench_steady(
            fn,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            do_bench_fn=_helion_do_bench_fn(bound, args, backend),
        )
        if require_full_autotune:
            _validate_post_measurement_source(autotune_provenance)
    return _result(
        impl_label,
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer=benchmark_timer,
        config=repr(active_config),
        codegen=codegen,
        helion_overrides={
            "env_overrides": env_overrides,
            "config_overrides": config_overrides,
            "seed_config_overrides": seed_config_overrides,
            "autotuned": autotuned,
            "benchmark_timer": benchmark_timer,
            "force_autotune": bool(args.helion_force_autotune),
            "return_lse": not output_only,
            **({"epilogue": epilogue} if epilogue != "none" else {}),
            "autotune_provenance": autotune_provenance,
        },
        notes=[
            *notes,
            *(
                ["ReLU was fused into the generated attention epilogue."]
                if epilogue == "relu"
                else []
            ),
        ],
    )


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.impl == "helion-cute":
        args.helion_backend = "cute"
        return _benchmark_helion(args)
    if args.impl == "helion-triton":
        args.helion_backend = "triton"
        args.helion_require_full_autotune = 0
        return _benchmark_helion(args)
    if args.impl == "helion-tileir":
        args.helion_backend = "tileir"
        args.helion_require_full_autotune = 0
        return _benchmark_helion(args)
    if args.impl == "tilegym-tileir":
        return _benchmark_tilegym_tileir(args)
    if args.impl == "sdpa":
        return _benchmark_sdpa(args)
    if args.impl in _KERNELAGENT_BUDGET_LABELS:
        return _benchmark_kernelagent(args)
    if args.impl == "gluon":
        return _benchmark_gluon(args)
    if args.impl == "tlx":
        return _benchmark_tlx(args)
    if args.impl in _FLEXATTENTION_BACKENDS:
        return _benchmark_flexattention(args)
    if args.impl == "fa4":
        return _benchmark_fa4(args)
    raise SystemExit(f"unknown impl {args.impl!r}")


def _build_subprocess_cmd(args: argparse.Namespace, impl: str) -> list[str]:
    require_full_autotune = bool(
        impl == "helion-cute" and getattr(args, "helion_require_full_autotune", 0)
    )
    helion_args = argparse.Namespace(
        **{
            **vars(args),
            "helion_require_full_autotune": int(require_full_autotune),
        }
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--z",
        str(args.z),
        "--h",
        str(args.h),
        "--seq-len",
        str(args.seq_len),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--causal",
        str(int(args.causal)),
        "--biased",
        str(int(_uses_bias(args))),
        "--epilogue",
        _epilogue(args),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--skip-correctness",
        str(int(args.skip_correctness)),
        "--helion-force-flash-config",
        str(int(getattr(args, "helion_force_flash_config", 0))),
        "--helion-force-autotune",
        str(int(getattr(args, "helion_force_autotune", 1))),
        "--helion-require-full-autotune",
        str(int(require_full_autotune)),
        "--helion-return-lse",
        str(int(getattr(args, "helion_return_lse", 0))),
        "--helion-cute-benchmark-timer",
        str(getattr(args, "helion_cute_benchmark_timer", "wall")),
        "--json",
    ]
    power_cap_w = getattr(args, "power_cap_w", None)
    if power_cap_w is not None:
        cmd.extend(["--power-cap-w", str(power_cap_w)])
    kernelagent_results_root = getattr(args, "kernelagent_results_root", None)
    if kernelagent_results_root:
        cmd.extend(["--kernelagent-results-root", str(kernelagent_results_root)])
    kernelagent_closed_results_root = getattr(
        args, "kernelagent_closed_results_root", None
    )
    if kernelagent_closed_results_root:
        cmd.extend(
            [
                "--kernelagent-closed-results-root",
                str(kernelagent_closed_results_root),
            ]
        )
    cmd.extend(_helion_override_args(helion_args))
    return cmd


def _run_json_subprocess(
    cmd: list[str], args: argparse.Namespace
) -> tuple[int, dict[str, Any] | None, str, str]:
    env = None
    impl_index = cmd.index("--impl") if "--impl" in cmd else -1
    impl = cmd[impl_index + 1] if impl_index >= 0 else None
    tlx_runtime_root = os.environ.get(_TLX_RUNTIME_ROOT_ENV)
    if impl == "tlx" and tlx_runtime_root:
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{tlx_runtime_root}{os.pathsep}{current_pythonpath}"
            if current_pythonpath
            else tlx_runtime_root
        )
    if args.stream_subprocesses:
        with tempfile.TemporaryDirectory() as tmpdir:
            temporary_directory = Path(tmpdir)
            json_path = temporary_directory / "result.json"
            proc = subprocess.run(
                [*cmd, "--json-output", str(json_path)],
                cwd=REPO_ROOT,
                check=False,
                env=env,
            )
            _preserve_streamed_strict_prevalidation(args, temporary_directory)
            if proc.returncode != 0:
                return proc.returncode, None, "", ""
            try:
                return proc.returncode, json.loads(json_path.read_text()), "", ""
            except (FileNotFoundError, json.JSONDecodeError) as err:
                return proc.returncode, None, "", f"failed to read {json_path}: {err}"

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        return proc.returncode, None, proc.stdout, proc.stderr
    stdout_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        return proc.returncode, json.loads(stdout_lines[-1]), proc.stdout, proc.stderr
    except (IndexError, json.JSONDecodeError):
        return proc.returncode, None, proc.stdout, proc.stderr


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    impls = args.impls or list(DEFAULT_IMPLS)
    results: list[dict[str, Any]] = []
    for impl in impls:
        if impl not in ALL_IMPLS:
            print(f"unknown impl {impl!r}, skipping", file=sys.stderr)
            continue
        print(f"\n=== Running {impl} ===", flush=True)
        cmd = _build_subprocess_cmd(args, impl)
        returncode, payload, stdout, stderr = _run_json_subprocess(cmd, args)
        require_full_autotune = bool(
            impl == "helion-cute" and getattr(args, "helion_require_full_autotune", 0)
        )
        if returncode != 0:
            print(
                f"impl {impl} failed (rc={returncode})\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}",
                file=sys.stderr,
            )
            if require_full_autotune:
                raise SystemExit(
                    "required full autotune helion-cute subprocess failed with "
                    f"exit code {returncode}"
                )
            results.append(
                {
                    "impl": impl,
                    "shape": _shape_dict(args),
                    "accuracy": "ERROR",
                    "error": f"subprocess rc={returncode}",
                }
            )
            continue
        if payload is None:
            print(
                f"impl {impl} produced no JSON output\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}",
                file=sys.stderr,
            )
            if require_full_autotune:
                raise SystemExit(
                    "required full autotune helion-cute subprocess produced no "
                    "JSON output"
                )
            results.append(
                {
                    "impl": impl,
                    "shape": _shape_dict(args),
                    "accuracy": "ERROR",
                    "error": "no JSON output",
                }
            )
        else:
            results.append(payload)
    return {"shape": _shape_dict(args), "results": results}


def _shape_label(shape: dict[str, Any]) -> str:
    label = (
        f"z{shape['z']}_h{shape['h']}_s{shape['seq_len']}_d{shape['head_dim']}"
        f"_{shape['dtype']}_causal{shape['causal']}_biased{shape.get('biased', 0)}"
    )
    epilogue = str(shape.get("epilogue", "none"))
    return label if epilogue == "none" else f"{label}_epilogue{epilogue}"


def _print_summary(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    epilogue = str(shape.get("epilogue", "none"))
    epilogue_text = "" if epilogue == "none" else f" epilogue={epilogue}"
    print(
        f"\nshape z={shape['z']} h={shape['h']} seq={shape['seq_len']} "
        f"head_dim={shape['head_dim']} dtype={shape['dtype']} "
        f"causal={shape['causal']} biased={shape.get('biased', 0)}{epilogue_text}"
    )
    impl_width = max(
        16,
        max(
            (len(str(result.get("impl", ""))) for result in payload["results"]),
            default=0,
        ),
    )
    separator_width = impl_width + 70
    print("=" * separator_width)
    header = (
        f"{'impl':>{impl_width}}  {'acc':>6}  {'best ms':>10}  "
        f"{'mom-med ms':>10}  {'best TF/s':>10}  {'mom-med TF/s':>12}"
    )
    print(header)
    print("-" * separator_width)
    sdpa_mom: float | None = None
    for r in payload["results"]:
        if r.get("impl") == "sdpa":
            sdpa_mom = r.get("mom_median_tflops")
    for r in payload["results"]:
        impl = r.get("impl", "")
        acc = r.get("accuracy", "?")
        if "best_ms" not in r:
            print(
                f"{impl:>{impl_width}}  {acc:>6}  "
                f"{'--':>10}  {'--':>10}  {'--':>10}  {'--':>12}"
            )
            for note in r.get("notes", ()):
                print(f"{'':>{impl_width}}  note: {note}")
            continue
        mom_ms = r.get("mom_median_ms", r["median_ms"])
        mom_tflops = r.get("mom_median_tflops", r["median_tflops"])
        line = (
            f"{impl:>{impl_width}}  {acc:>6}  "
            f"{r['best_ms']:>10.4f}  {mom_ms:>10.4f}  "
            f"{r['best_tflops']:>10.1f}  {mom_tflops:>12.1f}"
        )
        if sdpa_mom is not None and impl != "sdpa" and mom_tflops:
            line += f"   {mom_tflops / sdpa_mom * 100:>6.1f}% sdpa"
        print(line)
        details: list[str] = []
        if "benchmark_timer" in r:
            details.append(f"timer={r['benchmark_timer']}")
        helion_overrides = r.get("helion_overrides")
        if isinstance(helion_overrides, dict) and "autotuned" in helion_overrides:
            details.append(f"autotuned={helion_overrides['autotuned']}")
        if "config" in r:
            details.append(f"config={r['config']}")
        if details:
            print(f"{'':>{impl_width}}  {'; '.join(details)}")
        for note in r.get("notes", ()):
            print(f"{'':>{impl_width}}  note: {note}")
    print()


_MARKDOWN_COLUMNS = (
    "shape",
    "dtype",
    "causal",
    "biased",
    "impl",
    "version",
    "acc",
    "reason",
    "timer",
    "best_ms",
    "mom_med_ms",
    "best_tflops",
    "mom_med_tflops",
    "pct_sdpa",
)


def _markdown_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    shape = payload["shape"]
    shape_str = f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}"
    if shape.get("epilogue", "none") != "none":
        shape_str += f"+{shape['epilogue']}"
    sdpa_mom: float | None = None
    for r in payload["results"]:
        if r.get("impl") == "sdpa":
            sdpa_mom = r.get("mom_median_tflops")
    rows: list[dict[str, Any]] = []
    for r in payload["results"]:
        impl = r.get("impl", "")
        acc = r.get("accuracy", "?")
        if "best_ms" not in r:
            rows.append(
                {
                    "shape": shape_str,
                    "dtype": shape["dtype"],
                    "causal": shape["causal"],
                    "biased": shape.get("biased", 0),
                    "impl": impl,
                    "version": r.get("version", ""),
                    "acc": acc,
                    "reason": r.get("skipped_reason", r.get("error", "")),
                    "timer": r.get("benchmark_timer", ""),
                    "best_ms": "",
                    "mom_med_ms": "",
                    "best_tflops": "",
                    "mom_med_tflops": "",
                    "pct_sdpa": "",
                }
            )
            continue
        mom_ms = r.get("mom_median_ms", r["median_ms"])
        mom_tflops = r.get("mom_median_tflops", r["median_tflops"])
        pct = ""
        if sdpa_mom is not None and impl != "sdpa" and mom_tflops:
            pct = f"{mom_tflops / sdpa_mom * 100:.1f}%"
        rows.append(
            {
                "shape": shape_str,
                "dtype": shape["dtype"],
                "causal": shape["causal"],
                "biased": shape.get("biased", 0),
                "impl": impl,
                "version": r.get("version", ""),
                "acc": acc,
                "reason": "",
                "timer": r.get("benchmark_timer", ""),
                "best_ms": f"{r['best_ms']:.4f}",
                "mom_med_ms": f"{mom_ms:.4f}",
                "best_tflops": f"{r['best_tflops']:.1f}",
                "mom_med_tflops": f"{mom_tflops:.1f}",
                "pct_sdpa": pct,
            }
        )
    return rows


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(_MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _MARKDOWN_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[col]) for col in _MARKDOWN_COLUMNS) + " |"
        )
    return "\n".join(lines)


def _render_report_notes(payloads: list[dict[str, Any]]) -> str:
    error_rows: list[str] = []
    skipped_rows: list[str] = []
    fixed_config_rows: list[str] = []
    for payload in payloads:
        shape = payload["shape"]
        shape_str = f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}"
        variant = _variant_label(shape)
        for result in payload["results"]:
            impl = result.get("impl", "<unknown>")
            if result.get("accuracy") == "ERROR":
                error_rows.append(f"- {variant} {shape_str} {impl}")
                continue
            if result.get("accuracy") == "SKIP":
                skipped_rows.append(
                    f"- {variant} {shape_str} {impl}: "
                    f"{result.get('skipped_reason', 'unsupported')}"
                )
                continue
            helion_overrides = result.get("helion_overrides")
            if (
                isinstance(helion_overrides, dict)
                and helion_overrides.get("autotuned") is False
            ):
                fixed_config_rows.append(f"- {variant} {shape_str} {impl}")
    sections: list[str] = []
    if error_rows:
        sections.append(
            "Rows marked `ERROR` did not produce timing data and are omitted "
            "from the bar graph:\n" + "\n".join(error_rows)
        )
    if skipped_rows:
        sections.append(
            "Rows marked `SKIP` are not comparable and are omitted from the bar "
            "graph:\n" + "\n".join(skipped_rows)
        )
    if fixed_config_rows:
        sections.append(
            "Rows below used a fixed Helion config rather than full autotuning:\n"
            + "\n".join(fixed_config_rows)
        )
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(_MARKDOWN_COLUMNS), lineterminator="\n"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _variant_label(shape: dict[str, Any]) -> str:
    if shape.get("biased"):
        variant = "biased"
    elif shape["causal"]:
        variant = "causal"
    else:
        variant = "dense"
    if shape.get("epilogue", "none") == "relu":
        return f"{variant} + ReLU"
    return variant


def _wide_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        shape = payload["shape"]
        row: dict[str, Any] = {
            "variant": _variant_label(shape),
            "shape": f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}",
            "z": shape["z"],
            "h": shape["h"],
            "seq_len": shape["seq_len"],
            "head_dim": shape["head_dim"],
            "dtype": shape["dtype"],
            "causal": shape["causal"],
            "biased": shape.get("biased", 0),
        }
        if "epilogue" in shape:
            row["epilogue"] = shape["epilogue"]
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        input_seeds = {
            result["input_seed"]
            for result in payload["results"]
            if result.get("accuracy") == "PASS" and "input_seed" in result
        }
        row["input_seed"] = next(iter(input_seeds)) if len(input_seeds) == 1 else ""
        environment_result = next(
            (result for result in payload["results"] if result.get("gpu")), {}
        )
        row["gpu"] = environment_result.get("gpu", "")
        row["physical_gpu"] = environment_result.get("physical_gpu", "")
        row["power_cap_w"] = environment_result.get("power_cap_w", "")
        sdpa_mom = results_by_impl.get("sdpa", {}).get("mom_median_tflops")
        for impl in _DISPLAY_IMPLS:
            key = _IMPL_KEYS[impl]
            result = results_by_impl.get(impl, {})
            row[f"{key}_acc"] = result.get("accuracy", "")
            row[f"{key}_version"] = result.get("version", "")
            row[f"{key}_reason"] = result.get("skipped_reason", result.get("error", ""))
            row[f"{key}_flop_model"] = result.get("flop_model", "")
            row[f"{key}_timer"] = result.get("benchmark_timer", "")
            notes = result.get("notes")
            row[f"{key}_notes"] = json.dumps(notes, sort_keys=True) if notes else ""
            helion_overrides = result.get("helion_overrides")
            row[f"{key}_helion_overrides"] = (
                json.dumps(helion_overrides, sort_keys=True) if helion_overrides else ""
            )
            if "best_ms" not in result:
                row[f"{key}_best_ms"] = ""
                row[f"{key}_mom_med_ms"] = ""
                row[f"{key}_best_tflops"] = ""
                row[f"{key}_mom_med_tflops"] = ""
                row[f"{key}_pct_sdpa"] = ""
                continue
            row[f"{key}_best_ms"] = f"{result['best_ms']:.4f}"
            mom_ms = result.get("mom_median_ms", result["median_ms"])
            row[f"{key}_mom_med_ms"] = f"{mom_ms:.4f}"
            row[f"{key}_best_tflops"] = f"{result['best_tflops']:.1f}"
            mom_tflops = result.get("mom_median_tflops", result["median_tflops"])
            row[f"{key}_mom_med_tflops"] = f"{mom_tflops:.1f}"
            if sdpa_mom and impl != "sdpa":
                row[f"{key}_pct_sdpa"] = f"{mom_tflops / sdpa_mom * 100:.1f}%"
            else:
                row[f"{key}_pct_sdpa"] = ""
        rows.append(row)
    return rows


def _write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base_columns = [
        "variant",
        "shape",
        "z",
        "h",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "biased",
        "input_seed",
        "gpu",
        "physical_gpu",
        "power_cap_w",
    ]
    if any("epilogue" in row for row in rows):
        base_columns.insert(base_columns.index("input_seed"), "epilogue")
    impl_columns: list[str] = []
    for impl in _DISPLAY_IMPLS:
        key = _IMPL_KEYS[impl]
        impl_columns.extend(
            [
                f"{key}_acc",
                f"{key}_version",
                f"{key}_reason",
                f"{key}_flop_model",
                f"{key}_timer",
                f"{key}_notes",
                f"{key}_helion_overrides",
                f"{key}_best_ms",
                f"{key}_mom_med_ms",
                f"{key}_best_tflops",
                f"{key}_mom_med_tflops",
                f"{key}_pct_sdpa",
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*base_columns, *impl_columns],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_plot_cell(result: dict[str, Any]) -> str:
    accuracy = result.get("accuracy", "")
    if "best_ms" not in result:
        return str(accuracy)
    mom_ms = result.get("mom_median_ms", result["median_ms"])
    mom_tflops = result.get("mom_median_tflops", result["median_tflops"])
    label = f"{mom_tflops:.1f}\n{mom_ms:.4f} ms"
    if accuracy and accuracy != "PASS":
        label += f"\n{accuracy}"
    return label


def _shape_plot_label(shape: dict[str, Any]) -> str:
    variant = _variant_label(shape)
    seq_len = int(shape["seq_len"])
    seq_label = f"{seq_len // 1024}K" if seq_len % 1024 == 0 else str(seq_len)
    return f"{variant}\n{shape['z']}x{shape['h']}\n{seq_label}x{shape['head_dim']}"


def _versioned_impl_label(
    impl: str,
    payloads: list[dict[str, Any]],
    label_overrides: dict[str, str] | None = None,
) -> str:
    versions: list[str] = []
    fallback_versions: list[str] = []
    for payload in payloads:
        for result in payload["results"]:
            if result.get("impl") != impl:
                continue
            version = result.get("version_label") or result.get("version")
            if version and version not in fallback_versions:
                fallback_versions.append(str(version))
            if result.get("accuracy") == "PASS" and version and version not in versions:
                versions.append(str(version))
    if not versions:
        versions = fallback_versions
    impl_label = (
        label_overrides.get(impl, _IMPL_LABELS[impl])
        if label_overrides is not None
        else _IMPL_LABELS[impl]
    )
    if not versions:
        return impl_label
    return f"{impl_label}\n{' / '.join(versions)}"


def _benchmark_setup_label(payloads: list[dict[str, Any]]) -> str:
    gpu, power_cap_w = _validate_report_environment(payloads)
    if gpu is None:
        return "GPU setup not recorded"
    if power_cap_w is None:
        return gpu
    return f"{gpu} | {power_cap_w} W power cap"


def _benchmark_dtype_label(payloads: list[dict[str, Any]]) -> str:
    dtypes = {str(payload["shape"]["dtype"]) for payload in payloads}
    labels = {
        "float16": "FP16",
        "bfloat16": "BF16",
        "float32": "FP32",
    }
    if len(dtypes) == 1:
        dtype = next(iter(dtypes))
        return labels.get(dtype, dtype)
    return "mixed dtypes"


def _report_shape_key(shape: object, *, context: str) -> tuple[object, ...]:
    if not isinstance(shape, dict):
        raise ValueError(f"{context} has no shape object")
    required = ("z", "h", "seq_len", "head_dim", "dtype", "causal")
    missing = [field for field in required if field not in shape]
    if missing:
        raise ValueError(f"{context} shape is missing fields {missing}")
    return (
        shape["z"],
        shape["h"],
        shape["seq_len"],
        shape["head_dim"],
        shape["dtype"],
        int(shape["causal"]),
        int(shape.get("biased", 0)),
        str(shape.get("epilogue", "none")),
    )


def _validate_report_payloads(payloads: list[dict[str, Any]]) -> None:
    successful_metadata: dict[tuple[str, str], set[str]] = {}
    strict_helion_identity: set[tuple[object, ...]] = set()
    strict_terminal_surfaces: dict[tuple[object, ...], set[str]] = {}
    helion_provenance_modes: dict[str, set[str]] = {}
    for payload_index, payload in enumerate(payloads):
        input_seeds: set[int] = set()
        missing_input_seed = False
        payload_impls: set[str] = set()
        payload_shape = _report_shape_key(
            payload.get("shape"), context=f"payload {payload_index}"
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"payload {payload_index} has no results list")
        for result_index, result in enumerate(results):
            context = f"payload {payload_index} result {result_index}"
            if not isinstance(result, dict):
                raise ValueError(f"{context} is not an object")
            result_shape = _report_shape_key(result.get("shape"), context=context)
            if result_shape != payload_shape:
                raise ValueError(
                    f"{context} shape {result_shape} does not match payload shape "
                    f"{payload_shape}"
                )
            impl = result.get("impl")
            if not isinstance(impl, str) or not impl:
                raise ValueError(f"{context} has no implementation name")
            if impl in payload_impls:
                raise ValueError(
                    f"payload {payload_index} contains duplicate implementation {impl!r}"
                )
            payload_impls.add(impl)
            if result.get("accuracy") != "PASS":
                continue
            input_seed = result.get("input_seed")
            if input_seed is None:
                missing_input_seed = True
            else:
                if isinstance(input_seed, bool) or not isinstance(input_seed, int):
                    raise ValueError(f"{context} has invalid input_seed metadata")
                input_seeds.add(input_seed)
            if impl == "helion-cute":
                overrides = result.get("helion_overrides")
                provenance = (
                    overrides.get("autotune_provenance")
                    if isinstance(overrides, dict)
                    else None
                )
                strict = bool(
                    isinstance(provenance, dict)
                    and provenance.get("require_full_autotune") is True
                )
                helion_provenance_modes.setdefault(impl, set()).add(
                    "strict" if strict else "legacy"
                )
                if strict:
                    assert isinstance(provenance, dict)
                    source_hash = provenance.get("helion_source_tree_sha256")
                    checkout_commit = provenance.get("helion_checkout_git_commit")
                    post_measurement_source = provenance.get("post_measurement_source")
                    normalization = provenance.get("flash_normalization_context")
                    terminal_policy = provenance.get(
                        "flash_terminal_coordinate_refinement_policy"
                    )
                    terminal_policy_sha256 = provenance.get(
                        "flash_terminal_coordinate_refinement_policy_sha256"
                    )
                    terminal_surface_sha256 = provenance.get(
                        "flash_terminal_coordinate_surface_catalog_sha256"
                    )
                    terminal_surface_catalog = provenance.get(
                        "flash_terminal_coordinate_surface_catalog"
                    )
                    trials = provenance.get("trials")
                    phase = (
                        trials[0].get("search_phase_metrics")
                        if isinstance(trials, list)
                        and len(trials) == 1
                        and isinstance(trials[0], dict)
                        else None
                    )
                    terminal = (
                        phase.get("terminal_coordinate_refinement")
                        if isinstance(phase, dict)
                        else None
                    )
                    if (
                        not isinstance(source_hash, str)
                        or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
                        or not isinstance(checkout_commit, str)
                        or re.fullmatch(r"[0-9a-f]{40}", checkout_commit) is None
                        or provenance.get("post_measurement_source_verified")
                        is not True
                        or not isinstance(post_measurement_source, dict)
                        or post_measurement_source.get("helion_source_tree_sha256")
                        != source_hash
                        or post_measurement_source.get("helion_checkout_git_commit")
                        != checkout_commit
                        or post_measurement_source.get("helion_source_tree_dirty")
                        is not False
                        or not isinstance(normalization, dict)
                        or type(normalization.get("schema_version")) is not int
                        or not isinstance(phase, dict)
                        or not isinstance(phase.get("phase"), str)
                        or type(phase.get("cute_flash_lane_policy_version")) is not int
                        or not isinstance(terminal_policy, dict)
                        or terminal_policy_sha256
                        != _canonical_json_sha256(terminal_policy)
                        or not isinstance(terminal_surface_sha256, str)
                        or re.fullmatch(r"[0-9a-f]{64}", terminal_surface_sha256)
                        is None
                        or _flash_terminal_surface_catalog_error(
                            terminal_surface_catalog
                        )
                        is not None
                        or terminal_surface_sha256
                        != _canonical_json_sha256(terminal_surface_catalog)
                        or not isinstance(terminal, dict)
                        or terminal.get("schema_version")
                        != terminal_policy.get("schema_version")
                        or terminal.get("policy_version")
                        != terminal_policy.get("policy_version")
                        or terminal.get("lane_policy_version")
                        != terminal_policy.get("lane_policy_version")
                        or phase.get("cute_flash_lane_policy_version")
                        != terminal_policy.get("lane_policy_version")
                        or terminal.get("coordinate_policy")
                        != terminal_policy.get("coordinate_policy")
                        or terminal.get("measurement_policy")
                        != terminal_policy.get("measurement_policy")
                        or terminal.get("radius") != terminal_policy.get("radius")
                    ):
                        raise ValueError(
                            f"{context} has incomplete strict full-autotune provenance"
                        )
                    shape_seq_len = result_shape[2]
                    shape_head_dim = result_shape[3]
                    shape_dtype = result_shape[4]
                    shape_causal = result_shape[5]
                    if (
                        type(shape_seq_len) is not int
                        or shape_seq_len <= 0
                        or type(shape_head_dim) is not int
                        or shape_head_dim <= 0
                        or not isinstance(shape_dtype, str)
                        or not shape_dtype
                        or shape_causal not in {0, 1}
                    ):
                        raise ValueError(f"{context} has an invalid attention shape")
                    expected_normalization_dtype = f"torch.{shape_dtype}"
                    # This benchmark uses self-attention, and CuTe records the
                    # K length as its count of 128-wide KV tiles.
                    expected_num_kv = (shape_seq_len + 127) // 128
                    if (
                        normalization.get("dtype") != expected_normalization_dtype
                        or normalization.get("head_dim") != shape_head_dim
                        or normalization.get("num_kv") != expected_num_kv
                        or normalization.get("is_causal") is not bool(shape_causal)
                    ):
                        raise ValueError(
                            f"{context} normalization context does not match its shape"
                        )
                    strict_helion_identity.add(
                        (
                            source_hash,
                            checkout_commit,
                            cast("int", normalization["schema_version"]),
                            cast("str", phase["phase"]),
                            cast("int", phase["cute_flash_lane_policy_version"]),
                            terminal_policy["schema_version"],
                            terminal_policy["policy_version"],
                            terminal_policy["coordinate_policy"],
                            terminal_policy["measurement_policy"],
                            terminal_policy["radius"],
                            terminal_policy_sha256,
                        )
                    )
                    surface_shape_key = (
                        *result_shape,
                        cast("int", normalization["schema_version"]),
                    )
                    strict_terminal_surfaces.setdefault(surface_shape_key, set()).add(
                        cast("str", terminal_surface_sha256)
                    )
            for field in ("gpu", "physical_gpu"):
                value = result.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{context} has no {field} metadata")
            power_cap_w = result.get("power_cap_w")
            if (
                isinstance(power_cap_w, bool)
                or not isinstance(power_cap_w, (int, float))
                or not math.isfinite(power_cap_w)
                or power_cap_w <= 0
            ):
                raise ValueError(f"{context} has no power_cap_w metadata")
            for field in ("version", "benchmark_timer", "flop_model"):
                value = result.get(field)
                if value is None or value == "":
                    raise ValueError(f"{context} has no {field} metadata")
                successful_metadata.setdefault((impl, field), set()).add(str(value))
        if len(input_seeds) > 1:
            raise ValueError(
                f"payload {payload_index} mixes input seeds: {sorted(input_seeds)}"
            )
        if input_seeds and missing_input_seed:
            raise ValueError(
                f"payload {payload_index} mixes recorded and missing input seeds"
            )

    for (impl, field), values in successful_metadata.items():
        if len(values) > 1:
            raise ValueError(
                f"report mixes {field} metadata for {impl}: {sorted(values)}"
            )
    if any(len(modes) > 1 for modes in helion_provenance_modes.values()):
        raise ValueError("report mixes strict and legacy Helion autotune provenance")
    if len(strict_helion_identity) > 1:
        raise ValueError(
            "report mixes Helion source or structural-schema provenance across rows"
        )
    if any(len(hashes) > 1 for hashes in strict_terminal_surfaces.values()):
        raise ValueError(
            "report mixes terminal coordinate surfaces within one validated report shape"
        )


def _validate_report_environment(
    payloads: list[dict[str, Any]],
) -> tuple[str | None, int | float | None]:
    _validate_report_payloads(payloads)
    gpu_names: set[str] = set()
    power_caps: set[int | float | None] = set()
    for payload in payloads:
        physical_gpus: set[str] = set()
        for result in payload["results"]:
            gpu = result.get("gpu")
            if not gpu:
                continue
            gpu_names.add(str(gpu))
            raw_power_cap = result.get("power_cap_w")
            if raw_power_cap is None:
                power_caps.add(None)
            else:
                power_cap = float(raw_power_cap)
                power_caps.add(int(power_cap) if power_cap.is_integer() else power_cap)
            physical_gpu = result.get("physical_gpu")
            if physical_gpu:
                physical_gpus.add(str(physical_gpu))
        if len(physical_gpus) > 1:
            raise ValueError(
                "one shape contains results from multiple physical GPU selections: "
                f"{sorted(physical_gpus)}"
            )
    if len(gpu_names) > 1:
        raise ValueError(f"report mixes GPU models: {sorted(gpu_names)}")
    if len(power_caps) > 1:
        raise ValueError(
            "report mixes GPU power limits: "
            f"{sorted(str(power_cap) for power_cap in power_caps)}"
        )
    gpu = next(iter(gpu_names), None)
    power_cap_w = next(iter(power_caps), None)
    return gpu, power_cap_w


def _impls_by_average_performance(
    values_by_impl: dict[str, list[float]],
) -> list[str]:
    successful_impls = [
        impl
        for impl, values in values_by_impl.items()
        if any(map(math.isfinite, values))
    ]
    return sorted(
        successful_impls,
        key=lambda impl: statistics.fmean(
            value for value in values_by_impl[impl] if math.isfinite(value)
        ),
    )


def _geomean_performance_by_impl(
    values_by_impl: dict[str, list[float]],
) -> dict[str, float]:
    return {
        impl: statistics.geometric_mean(values)
        for impl, values in values_by_impl.items()
        if values and all(math.isfinite(value) and value > 0.0 for value in values)
    }


def _write_matplotlib_bar_graph(
    path: Path,
    payloads: list[dict[str, Any]],
    label_overrides: dict[str, str] | None = None,
) -> None:
    import matplotlib  # pyrefly: ignore[missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]
    import numpy as np

    labels: list[str] = []
    values_by_impl: dict[str, list[float]] = {impl: [] for impl in _DISPLAY_IMPLS}
    for payload in payloads:
        shape = payload["shape"]
        labels.append(_shape_plot_label(shape))
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        for impl in _DISPLAY_IMPLS:
            result = results_by_impl.get(impl, {})
            if result.get("accuracy") == "PASS" and "mom_median_tflops" in result:
                values_by_impl[impl].append(float(result["mom_median_tflops"]))
            else:
                values_by_impl[impl].append(np.nan)

    plot_impls = _impls_by_average_performance(values_by_impl)
    if not plot_impls:
        raise ValueError("no successful benchmark results to plot")

    x = np.arange(len(labels))
    width = 0.82 / len(plot_impls)
    fig_width = max(19.0, 1.8 * len(labels) + 7.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, impl in enumerate(plot_impls):
        offsets = x + (index - (len(plot_impls) - 1) / 2) * width
        color = colors[index % len(colors)]
        ax.bar(
            offsets,
            values_by_impl[impl],
            width,
            label=_versioned_impl_label(impl, payloads, label_overrides),
            color=color,
        )
        for offset, value in zip(offsets, values_by_impl[impl], strict=True):
            if not math.isfinite(value):
                ax.scatter(offset, 0, marker="x", s=20, color=color, clip_on=False)
                ax.annotate(
                    "FAIL",
                    (offset, 0),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                    color=color,
                )

    ax.set_ylabel("Throughput (TFLOP/s)", fontsize=13)
    ax.set_title(
        f"Attention forward throughput ({_benchmark_dtype_label(payloads)}) | "
        f"{_benchmark_setup_label(payloads)}",
        fontsize=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_matplotlib_geomean_bar_graph(
    path: Path,
    payloads: list[dict[str, Any]],
    label_overrides: dict[str, str] | None = None,
) -> None:
    import matplotlib  # pyrefly: ignore[missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]

    values_by_impl: dict[str, list[float]] = {impl: [] for impl in _DISPLAY_IMPLS}
    for payload in payloads:
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        for impl in _DISPLAY_IMPLS:
            result = results_by_impl.get(impl, {})
            if result.get("accuracy") == "PASS" and "mom_median_tflops" in result:
                values_by_impl[impl].append(float(result["mom_median_tflops"]))
            else:
                values_by_impl[impl].append(math.nan)

    geomean_by_impl = _geomean_performance_by_impl(values_by_impl)
    if not geomean_by_impl:
        raise ValueError("no complete successful benchmark series to plot")
    plot_impls = sorted(
        _impls_by_average_performance(values_by_impl),
        key=lambda impl: (
            impl in geomean_by_impl,
            geomean_by_impl.get(impl, 0.0),
        ),
    )

    color_order = _impls_by_average_performance(values_by_impl)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_by_impl = {
        impl: colors[index % len(colors)] for index, impl in enumerate(color_order)
    }

    fig_height = max(8.2, 0.9 * len(plot_impls))
    fig, ax = plt.subplots(figsize=(18.5, fig_height))
    bars = ax.barh(
        range(len(plot_impls)),
        [geomean_by_impl.get(impl, 0.0) for impl in plot_impls],
        color=[color_by_impl[impl] for impl in plot_impls],
    )
    ax.set_yticks(range(len(plot_impls)))
    ax.set_yticklabels(
        [_versioned_impl_label(impl, payloads, label_overrides) for impl in plot_impls],
        fontsize=11,
    )
    value_labels = []
    for impl in plot_impls:
        if impl in geomean_by_impl:
            value_labels.append(f"{geomean_by_impl[impl]:.1f}")
            continue
        completed = sum(math.isfinite(value) for value in values_by_impl[impl])
        value_labels.append(f"INCOMPLETE ({completed}/{len(payloads)})")
    ax.bar_label(bars, labels=value_labels, padding=4, fontsize=12)
    for index, impl in enumerate(plot_impls):
        if impl not in geomean_by_impl:
            ax.scatter(
                0,
                index,
                marker="x",
                s=35,
                color=color_by_impl[impl],
                clip_on=False,
            )
    ax.set_xlim(0, max(geomean_by_impl.values()) * 1.1)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("Geometric mean throughput (TFLOP/s)", fontsize=13)
    ax.set_title(
        f"Attention forward geometric-mean throughput across {len(payloads)} shapes "
        f"({_benchmark_dtype_label(payloads)}) | {_benchmark_setup_label(payloads)}",
        fontsize=15,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_matplotlib_table(
    path: Path,
    payloads: list[dict[str, Any]],
    label_overrides: dict[str, str] | None = None,
) -> None:
    import matplotlib  # pyrefly: ignore[missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]

    columns = [
        "Variant",
        "Shape\nzxhxSxD",
        "dtype",
        *(
            _versioned_impl_label(impl, payloads, label_overrides)
            for impl in _DISPLAY_IMPLS
        ),
    ]
    cell_text: list[list[str]] = []
    for payload in payloads:
        shape = payload["shape"]
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        cell_text.append(
            [
                _variant_label(shape),
                f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}",
                shape["dtype"],
                *(
                    _format_plot_cell(results_by_impl.get(impl, {}))
                    for impl in _DISPLAY_IMPLS
                ),
            ]
        )

    fig_height = max(3.0, 0.72 * len(cell_text) + 1.4)
    fig_width = 13.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.75)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9eef6")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    ax.set_title(
        f"Attention backend performance: TFLOP/s, "
        f"{_benchmark_dtype_label(payloads)} "
        f"({_benchmark_setup_label(payloads)})",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_shape_subprocess(
    args: argparse.Namespace, shape: tuple[int, int, int, int, str, int, int]
) -> dict[str, Any]:
    z, h, seq_len, head_dim, dtype, causal, biased = shape
    require_full_autotune = bool(
        getattr(args, "helion_require_full_autotune", 0)
        and (not args.impls or "helion-cute" in args.impls)
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        "all",
        "--z",
        str(z),
        "--h",
        str(h),
        "--seq-len",
        str(seq_len),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--causal",
        str(causal),
        "--biased",
        str(biased),
        "--epilogue",
        _epilogue(args),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--skip-correctness",
        str(int(args.skip_correctness)),
        "--helion-force-flash-config",
        str(int(getattr(args, "helion_force_flash_config", 0))),
        "--helion-force-autotune",
        str(int(getattr(args, "helion_force_autotune", 1))),
        "--helion-require-full-autotune",
        str(int(require_full_autotune)),
        "--helion-return-lse",
        str(int(getattr(args, "helion_return_lse", 0))),
        "--helion-cute-benchmark-timer",
        str(getattr(args, "helion_cute_benchmark_timer", "wall")),
        "--json",
    ]
    power_cap_w = getattr(args, "power_cap_w", None)
    if power_cap_w is not None:
        cmd.extend(["--power-cap-w", str(power_cap_w)])
    if args.impls:
        cmd.extend(["--impls", *args.impls])
    kernelagent_results_root = getattr(args, "kernelagent_results_root", None)
    if kernelagent_results_root:
        cmd.extend(["--kernelagent-results-root", kernelagent_results_root])
    kernelagent_closed_results_root = getattr(
        args, "kernelagent_closed_results_root", None
    )
    if kernelagent_closed_results_root:
        cmd.extend(
            [
                "--kernelagent-closed-results-root",
                kernelagent_closed_results_root,
            ]
        )
    if args.stream_subprocesses:
        cmd.append("--stream-subprocesses")
    if require_full_autotune:
        # Validate strict-only conflicts here, but leave the strict-derived
        # environment for the final helion-cute child. Materializing it in this
        # intermediate --impl all process would also feed it to Triton/TileIR.
        _helion_env_overrides(args)
    outer_helion_args = argparse.Namespace(
        **{
            **vars(args),
            "helion_require_full_autotune": 0,
        }
    )
    cmd.extend(_helion_override_args(outer_helion_args))
    shape_dict = {
        "z": z,
        "h": h,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": dtype,
        "causal": causal,
        "biased": biased,
    }
    if _epilogue(args) != "none":
        shape_dict["epilogue"] = _epilogue(args)
    returncode, payload, stdout, stderr = _run_json_subprocess(cmd, args)
    if returncode != 0:
        print(
            f"shape {_shape_label(shape_dict)} failed (rc={returncode})\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        if require_full_autotune:
            raise SystemExit(
                "required full autotune shape subprocess failed with "
                f"exit code {returncode}: {_shape_label(shape_dict)}"
            )
        return {"shape": shape_dict, "results": []}
    if payload is None:
        print(
            f"shape {_shape_label(shape_dict)} produced no JSON output\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        if require_full_autotune:
            raise SystemExit(
                "required full autotune shape subprocess produced no JSON output: "
                f"{_shape_label(shape_dict)}"
            )
        return {"shape": shape_dict, "results": []}
    return payload


def _validate_all_shapes_full_autotune(args: argparse.Namespace) -> None:
    """Reject strict suites containing unsupported fixed-config workloads."""
    require_full_autotune = bool(
        getattr(args, "helion_require_full_autotune", 0)
        and (not args.impls or "helion-cute" in args.impls)
    )
    if not require_full_autotune:
        return
    biased_shapes = [shape for shape in _SHAPE_SUITES[args.shape_suite] if shape[-1]]
    if biased_shapes:
        raise SystemExit(
            "--helion-require-full-autotune cannot run --shape-suite "
            f"{args.shape_suite!r}: it includes biased attention, whose CuTe path "
            "uses a fixed config rather than the characterized full search"
        )


def _write_sweep_outputs(
    args: argparse.Namespace, payloads: list[dict[str, Any]]
) -> None:
    _validate_report_environment(payloads)
    all_rows: list[dict[str, Any]] = []
    for payload in payloads:
        all_rows.extend(_markdown_rows(payload))

    table = _render_markdown_table(all_rows)
    notes = _render_report_notes(payloads)
    print("\n## Attention backend sweep\n")
    print(table + notes)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text("## Attention backend sweep\n\n" + table + notes + "\n")
        print(f"\nWrote Markdown table to {out_path}", file=sys.stderr)
    if args.append_csv:
        _append_csv(Path(args.append_csv), all_rows)
        print(f"Appended {len(all_rows)} rows to {args.append_csv}", file=sys.stderr)
    wide_rows = _wide_rows(payloads)
    if args.csv_output:
        _write_wide_csv(Path(args.csv_output), wide_rows)
        print(f"Wrote wide CSV table to {args.csv_output}", file=sys.stderr)
    label_overrides = dict(args.plot_impl_label)
    if args.plot_output:
        _write_matplotlib_bar_graph(Path(args.plot_output), payloads, label_overrides)
        print(
            f"Wrote matplotlib TFLOP/s bar graph to {args.plot_output}", file=sys.stderr
        )
    if args.summary_plot_output:
        _write_matplotlib_geomean_bar_graph(
            Path(args.summary_plot_output), payloads, label_overrides
        )
        print(
            f"Wrote matplotlib geomean TFLOP/s bar graph to {args.summary_plot_output}",
            file=sys.stderr,
        )


def _run_all_shapes(args: argparse.Namespace) -> None:
    _check_gpu_policy()
    payloads: list[dict[str, Any]] = []
    for shape in _SHAPE_SUITES[args.shape_suite]:
        print(f"\n##### shape {shape} #####", flush=True)
        payload = _run_shape_subprocess(args, shape)
        payloads.append(payload)
        _print_summary(payload)

    _write_sweep_outputs(args, payloads)


def _run_merge_json(args: argparse.Namespace) -> None:
    if getattr(args, "helion_require_full_autotune", 0):
        raise SystemExit(
            "--merge-json cannot verify --helion-require-full-autotune; rerun the "
            "benchmark directly in strict mode"
        )
    payloads = [json.loads(Path(path).read_text()) for path in args.merge_json]
    for payload in payloads:
        _print_summary(payload)
    _write_sweep_outputs(args, payloads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--impl", choices=("all", *ALL_IMPLS), default="all")
    parser.add_argument(
        "--impls",
        nargs="*",
        default=None,
        help=(
            "Override DEFAULT_IMPLS for --impl all "
            f"(default: {' '.join(DEFAULT_IMPLS)})"
        ),
    )
    parser.add_argument("--z", type=int, default=2, help="batch size")
    parser.add_argument("--h", type=int, default=8, help="number of heads")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--epilogue",
        choices=("none", "relu"),
        default="none",
        help=(
            "Output epilogue included in correctness and timing. ReLU is supported "
            "for output-only, unbiased BF16 attention."
        ),
    )
    parser.add_argument(
        "--biased",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use an additive attention score bias. Not compatible with --causal.",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--power-cap-w",
        type=int,
        default=None,
        help="Record the verified GPU power cap in JSON, CSV, and plot titles.",
    )
    parser.add_argument("--skip-correctness", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--kernelagent-results-root",
        default=None,
        help=(
            "Root containing shape-specific KernelAgent run directories. "
            f"Defaults to {_KERNELAGENT_RESULTS_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--kernelagent-closed-results-root",
        default=None,
        help=(
            "Root containing shape-specific KernelAgent Closed run "
            "directories. Defaults to "
            f"{_KERNELAGENT_CLOSED_RESULTS_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (used by --impl all subprocess collection).",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Write the JSON payload to this file, in addition to normal output.",
    )
    parser.add_argument(
        "--stream-subprocesses",
        action="store_true",
        help=(
            "Stream nested subprocess stdout/stderr directly. Useful for long "
            "Helion autotune runs; JSON is collected through a sidecar file."
        ),
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help=(
            "Sweep the selected shape list (each via an --impl all "
            "subprocess) and emit a Markdown table to stdout."
        ),
    )
    parser.add_argument(
        "--merge-json",
        nargs="+",
        default=None,
        help=(
            "Merge saved per-shape JSON payloads from --impl all into the "
            "--output/--append-csv/--csv-output/--plot-output/"
            "--summary-plot-output reports without rerunning benchmarks."
        ),
    )
    parser.add_argument(
        "--shape-suite",
        choices=tuple(_SHAPE_SUITES),
        default="representative",
        help="(--all-shapes) Shape list to sweep.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(--all-shapes) Write the Markdown table to this file.",
    )
    parser.add_argument(
        "--append-csv",
        default=None,
        help="(--all-shapes) Append sweep rows to this CSV file.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="(--all-shapes) Write one wide CSV row per shape to this file.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="(--all-shapes) Write a matplotlib-rendered TFLOP/s bar graph to this file.",
    )
    parser.add_argument(
        "--summary-plot-output",
        default=None,
        help=(
            "(--all-shapes) Write one geometric-mean TFLOP/s bar per complete "
            "implementation series to this file."
        ),
    )
    parser.add_argument(
        "--plot-impl-label",
        action="append",
        type=_parse_plot_impl_label,
        default=[],
        metavar="IMPL=LABEL",
        help=(
            "Override an implementation's display label in generated plots. "
            "Repeat for multiple implementations."
        ),
    )
    # Internal: backend selector threaded through the helion dispatch.
    parser.add_argument("--helion-backend", default="triton", help=argparse.SUPPRESS)
    parser.add_argument(
        "--helion-force-flash-config",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Skip autotune and use the compiler-promoted flash seed when one "
            "exists, otherwise Config(block_sizes=[1,128,128]). Fast compile "
            "path; verifies the flash kernel fires without full search."
        ),
    )
    parser.add_argument(
        "--helion-force-autotune",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Pass force=True to bound.autotune for Helion impls. Set to 0 to "
            "allow full-effort cache reads while still autotuning cache misses."
        ),
    )
    parser.add_argument(
        "--helion-require-full-autotune",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Fail Helion runs unless they perform a fresh, cache-bypassed standard "
            "full autotune with candidate accuracy checks, a deterministic structural "
            "coverage design, and random exploration with no flash-specific priors. "
            "No config is fixed as the winner. Fixed configs, user seeds, "
            "shape-specific measured-winner presets, and flash-specific value "
            "priors are rejected. Only the live canonical CuTe-flash compiler "
            "seeds are admitted, and every seed must be measured in pass zero. "
            "Records the exact built-in "
            "normalized coverage design and aggregate measured-source provenance "
            "in JSON."
        ),
    )
    parser.add_argument(
        "--helion-return-lse",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Use the LSE-returning Helion attention examples. The default "
            "uses output-only kernels for dense, causal, and biased attention so "
            "Helion does not compute an aux output that SDPA/FA4 omit."
        ),
    )
    parser.add_argument(
        "--helion-cute-benchmark-timer",
        choices=("wall", "event"),
        default="wall",
        help=(
            "Timer for Helion-CuTe benchmark samples. The default wall-clock "
            "path matches CuTe autotune timing; event mode uses the same CUDA "
            "event timing path as FlexAttention/SDPA/FA4 for opt-in comparisons."
        ),
    )
    parser.add_argument(
        "--helion-autotune-effort",
        choices=("none", "quick", "full"),
        default=None,
        help="Set HELION_AUTOTUNE_EFFORT for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-budget-seconds",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_BUDGET_SECONDS for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-max-generations",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_MAX_GENERATIONS for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-best-of-k",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_BEST_OF_K for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-benchmark-timeout",
        type=int,
        default=None,
        help=(
            "Set HELION_AUTOTUNE_BENCHMARK_TIMEOUT for Helion impl subprocesses. "
            "Use a larger value for very long attention shapes whose slow "
            "candidate configs can exceed the default per-config timeout."
        ),
    )
    parser.add_argument(
        "--helion-autotune-accuracy-check",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Set HELION_AUTOTUNE_ACCURACY_CHECK for Helion impl subprocesses. "
            "Set to 0 for performance-only sweeps after correctness is checked "
            "separately."
        ),
    )
    parser.add_argument(
        "--helion-autotuner-initial-population",
        choices=("from_random", "from_best_available"),
        default=None,
        help=(
            "Set HELION_AUTOTUNER_INITIAL_POPULATION for Helion impl "
            "subprocesses. Use from_best_available for long full-effort "
            "attention sweeps so cached and compiler-seeded configs are tried "
            "before random exploration."
        ),
    )
    parser.add_argument(
        "--helion-env",
        action="append",
        type=_parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set an environment variable for Helion impls only. Repeat for "
            "multiple vars; forwarded through --impl all and --all-shapes."
        ),
    )
    parser.add_argument(
        "--helion-config",
        action="append",
        type=_parse_config_override,
        default=[],
        metavar="KEY=JSON",
        help=(
            "Set a helion.Config kwarg for Helion impls only, parsing VALUE as "
            "JSON when possible and as a string otherwise. Repeat to sweep "
            "prospective knobs, e.g. --helion-config block_sizes='[1,128,128]'."
        ),
    )
    parser.add_argument(
        "--helion-seed-config",
        action="append",
        type=_parse_config_override,
        default=[],
        metavar="KEY=JSON",
        help=(
            "Seed a Helion autotune candidate without fixing the search space. "
            "Repeat for each helion.Config field."
        ),
    )
    return parser.parse_args()


def _validate_requested_impls(impls: list[str] | None) -> None:
    if impls and len(impls) != len(set(impls)):
        raise SystemExit("--impls contains duplicate implementations")


def main() -> None:
    args = parse_args()
    _validate_requested_impls(args.impls)
    if args.merge_json:
        _run_merge_json(args)
        return
    _validate_epilogue_workload(args)
    if args.all_shapes:
        _validate_all_shapes_full_autotune(args)

    _check_gpu_policy()
    args.power_cap_w = _verify_power_cap_w(args.power_cap_w)

    if args.causal and _uses_bias(args):
        raise SystemExit("--biased 1 is not compatible with --causal 1")

    if args.all_shapes:
        _run_all_shapes(args)
        return

    if args.impl == "all":
        payload = _run_all(args)
        _write_json_output(args, payload)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_summary(payload)
        return

    result = _run_impl(args)
    _write_json_output(args, result)
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
