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
import contextlib
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Protocol
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

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


class _ConfigLike(Protocol):
    config: dict[str, object]


class _ConfigSpecWithFlashSeeds(Protocol):
    compiler_default_config: object | None
    compiler_seed_configs: list[_ConfigLike]

    def default_config(self) -> _ConfigLike: ...


class _BoundWithConfigSpec(Protocol):
    @property
    def config_spec(self) -> _ConfigSpecWithFlashSeeds: ...


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


def _check_kernelagent_output(
    actual: object, expected: torch.Tensor, *, chunk_rows: int = 4096
) -> bool:
    if not isinstance(actual, torch.Tensor):
        return False
    if (
        actual.shape != expected.shape
        or actual.dtype is not torch.float16
        or actual.device.type != "cuda"
    ):
        return False
    for start in range(0, actual.shape[-2], chunk_rows):
        stop = min(start + chunk_rows, actual.shape[-2])
        actual_float = actual[..., start:stop, :].float()
        expected_float = expected[..., start:stop, :].float()
        if not bool(
            torch.all(
                torch.isclose(
                    actual_float,
                    expected_float,
                    atol=5e-2,
                    rtol=2e-2,
                )
            ).item()
        ):
            return False
    return True


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


def _check_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Dtype-aware correctness check; returns True on pass.

    bf16/fp16 attention accumulates softmax rounding noise, so benchmark smoke
    checks use a looser threshold than unit tests.
    """
    try:
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=5e-2, rtol=2e-2
        )
    except AssertionError:
        return False
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
    use_backend_timer = bench_fn is not None
    if bench_fn is None:
        from triton.testing import do_bench

        bench_fn = cast("Callable[..., Any]", do_bench)

    for _ in range(cache_warmup_calls):
        fn()
    torch.cuda.synchronize()

    _gpu_warmup(thermal_warmup_ms)

    runs: list[float] = []
    for _ in range(num_runs):
        if use_backend_timer:
            # Helion autotune also requests return_mode="median" from backend
            # timers, so use the same statistic when validating the winner.
            ms = bench_fn(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        else:
            ms = bench_fn(fn, warmup=warmup_ms, rep=rep_ms)
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
        "flop_model": "softmax_attention_forward",
        "accuracy": accuracy,
        "benchmark_timer": benchmark_timer,
    }
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
        "flop_model": "not_comparable",
        "accuracy": "SKIP",
        "skipped_reason": reason,
        "notes": [reason],
    }


def _shape_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "z": args.z,
        "h": args.h,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": int(args.causal),
        "biased": int(_uses_bias(args)),
    }


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
    }


def _helion_override_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for key, value in _helion_env_overrides(args).items():
        result.extend(["--helion-env", f"{key}={value}"])
    for key, value in getattr(args, "helion_config", ()):
        result.extend(["--helion-config", f"{key}={json.dumps(value)}"])
    for key, value in getattr(args, "helion_seed_config", ()):
        result.extend(["--helion-seed-config", f"{key}={json.dumps(value)}"])
    return result


def _helion_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = dict(getattr(args, "helion_env", ()))
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
    if helion_autotuner_initial_population is not None:
        env_overrides["HELION_AUTOTUNER_INITIAL_POPULATION"] = (
            helion_autotuner_initial_population
        )
    return env_overrides


def _apply_helion_env(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = _helion_env_overrides(args)
    os.environ.update(env_overrides)
    return env_overrides


def _write_json_output(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload) + "\n")


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
    causal = bool(args.causal)
    fn = lambda: _sdpa_reference(q, k, v, causal=causal, bias=bias)  # noqa: E731
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
        notes=["Forced torch SDPBackend.CUDNN_ATTENTION."],
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
    ):
        return _skipped_result(
            impl,
            args,
            "KernelAgent artifacts cover only output-only FP16 B=2 H=32 D=64",
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
    with _scrubbed_argv():
        if backend == "FLASH":
            _import_fa4()
        compiled = cast(
            "Callable[..., torch.Tensor]",
            torch.compile(flex_attention, fullgraph=True),
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
            expected = _sdpa_reference(q, k, v, causal=causal, bias=bias)
            out = fn()
            accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"
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
        notes=[f"Forced PyTorch FlexAttention BACKEND={backend!r}."],
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
        expected = _sdpa_reference(q, k, v, causal=causal)
        accuracy = "PASS" if _check_close(run(), expected, dtype) else "FAIL"
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
        expected = _sdpa_reference(q, k, v, causal=causal)
        accuracy = "PASS" if _check_close(run(), expected, dtype) else "FAIL"
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

    qt = q.transpose(1, 2).contiguous()  # (B, H, S, D) -> (B, S, H, D)
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    def run() -> torch.Tensor:
        out, _lse = fc.flash_attn_func(qt, kt, vt, softmax_scale=None, causal=causal)
        return out

    with _scrubbed_argv():
        accuracy = "PASS"
        if not args.skip_correctness:
            expected = _sdpa_reference(q, k, v, causal=causal)
            out = run()  # (B, S, H, D)
            got = out.transpose(1, 2)  # back to (B, H, S, D)
            accuracy = "PASS" if _check_close(got, expected, dtype) else "FAIL"
        stats = _bench_steady(
            run,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    return _result(
        "fa4", args, stats, accuracy=accuracy, benchmark_timer="event", config=None
    )


def _helion_benchmark_timer(args: argparse.Namespace, backend: str) -> str:
    if backend == "cute":
        return str(getattr(args, "helion_cute_benchmark_timer", "wall"))
    return "event"


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
    the compiler-promoted flash seed, including any shape-specific heuristic
    keys. This is useful for benchmarking individual knob variants without
    waiting for autotuner search.
    """
    env_overrides = _apply_helion_env(args)
    backend = args.helion_backend
    os.environ["HELION_BACKEND"] = backend
    impl_label = f"helion-{backend}"

    from examples.attention import attention
    from examples.attention import attention_output
    from examples.attention import biased_attention
    from examples.attention import biased_attention_output
    from examples.attention import causal_attention
    from examples.attention import causal_attention_output

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    output_only = not bool(args.helion_return_lse)
    if bias is not None:
        kernel = biased_attention_output if output_only else biased_attention
        kernel_args = (q, k, v, bias)
    elif output_only:
        kernel = causal_attention_output if causal else attention_output
        kernel_args = (q, k, v)
    elif causal:
        kernel = causal_attention
        kernel_args = (q, k, v)
    else:
        kernel = attention
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
        if fixed_config is not None:
            bound.set_config(fixed_config)
            active_config = fixed_config
            autotuned = False
        else:
            active_config = bound.autotune(
                kernel_args, force=bool(args.helion_force_autotune)
            )
            autotuned = True

        code = bound.to_triton_code(active_config)
        codegen = _helion_codegen_markers(code)

        accuracy = "PASS"
        if not args.skip_correctness:
            expected = _sdpa_reference(q, k, v, causal=causal, bias=bias)
            actual = bound(*kernel_args)
            out = cast("torch.Tensor", actual if output_only else actual[0])
            accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"

        fn = lambda: bound(*kernel_args)  # noqa: E731
        benchmark_timer = _helion_benchmark_timer(args, backend)
        stats = _bench_steady(
            fn,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            do_bench_fn=_helion_do_bench_fn(bound, args, backend),
        )
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
        },
        notes=notes,
    )


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.impl == "helion-cute":
        args.helion_backend = "cute"
        return _benchmark_helion(args)
    if args.impl == "helion-triton":
        args.helion_backend = "triton"
        return _benchmark_helion(args)
    if args.impl == "helion-tileir":
        args.helion_backend = "tileir"
        return _benchmark_helion(args)
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
    cmd.extend(_helion_override_args(args))
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
            json_path = Path(tmpdir) / "result.json"
            proc = subprocess.run(
                [*cmd, "--json-output", str(json_path)],
                cwd=REPO_ROOT,
                check=False,
                env=env,
            )
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
        if returncode != 0:
            print(
                f"impl {impl} failed (rc={returncode})\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}",
                file=sys.stderr,
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
    return (
        f"z{shape['z']}_h{shape['h']}_s{shape['seq_len']}_d{shape['head_dim']}"
        f"_{shape['dtype']}_causal{shape['causal']}_biased{shape.get('biased', 0)}"
    )


def _print_summary(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    print(
        f"\nshape z={shape['z']} h={shape['h']} seq={shape['seq_len']} "
        f"head_dim={shape['head_dim']} dtype={shape['dtype']} "
        f"causal={shape['causal']} biased={shape.get('biased', 0)}"
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
        return "biased"
    if shape["causal"]:
        return "causal"
    return "dense"


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
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
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
    base_columns = (
        "variant",
        "shape",
        "z",
        "h",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "biased",
        "gpu",
        "physical_gpu",
        "power_cap_w",
    )
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
    )


def _validate_report_payloads(payloads: list[dict[str, Any]]) -> None:
    successful_metadata: dict[tuple[str, str], set[str]] = {}
    for payload_index, payload in enumerate(payloads):
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
            if result.get("accuracy") != "PASS":
                continue
            impl = result.get("impl")
            if not isinstance(impl, str) or not impl:
                raise ValueError(f"{context} has no implementation name")
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

    for (impl, field), values in successful_metadata.items():
        if len(values) > 1:
            raise ValueError(
                f"report mixes {field} metadata for {impl}: {sorted(values)}"
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
    cmd.extend(_helion_override_args(args))
    shape_dict = {
        "z": z,
        "h": h,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": dtype,
        "causal": causal,
        "biased": biased,
    }
    returncode, payload, stdout, stderr = _run_json_subprocess(cmd, args)
    if returncode != 0:
        print(
            f"shape {_shape_label(shape_dict)} failed (rc={returncode})\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        return {"shape": shape_dict, "results": []}
    if payload is None:
        print(
            f"shape {_shape_label(shape_dict)} produced no JSON output\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        return {"shape": shape_dict, "results": []}
    return payload


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


def main() -> None:
    args = parse_args()
    if args.merge_json:
        _run_merge_json(args)
        return

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
