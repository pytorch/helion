"""Small, reproducible helpers for the grouped-GEMM DeepGEMM comparison."""

from __future__ import annotations

from hashlib import sha256
import importlib
import importlib.machinery
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from typing import Protocol
from typing import Sequence
from typing import cast

import torch

M_ALIGNMENT = 224
DEEPGEMM_VERSION = "2.6.1"
DEEPGEMM_COMMIT = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
DEEPGEMM_CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
DEEPGEMM_FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
_ALLOWED_DEEPGEMM_ENVIRONMENT = frozenset({"DG_JIT_CACHE_DIR"})


class _DeepGemmRuntime(Protocol):
    def set_num_sms(self, value: int) -> None: ...

    def get_num_sms(self) -> int: ...

    def set_tc_util(self, value: int) -> None: ...

    def get_tc_util(self) -> int: ...

    def set_pdl(self, value: bool) -> None: ...

    def get_pdl(self) -> bool: ...

    def set_ignore_compile_dims(self, value: bool) -> None: ...

    def set_block_size_multiple_of(self, value: int) -> None: ...


def align(value: int, alignment: int) -> int:
    """Round a value up to an alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the scale-independent error used by the dashboard checks."""
    actual64 = actual.double()
    expected64 = expected.double()
    denominator = (actual64.square() + expected64.square()).sum()
    if float(denominator) == 0.0:
        return 0.0
    return float((1 - 2 * (actual64 * expected64).sum() / denominator).item())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
    ).rstrip("\n")


def _clean_checkout(root: Path, expected_head: str, label: str) -> str:
    if not root.is_dir():
        raise RuntimeError(f"{label} root does not exist: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(f"{label} HEAD is {head}, expected {expected_head}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"{label} checkout is dirty: {status.splitlines()}")
    return head


def _native_extension(root: Path) -> tuple[Path, dict[str, object]]:
    extensions = sorted((root / "deep_gemm").glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError(
            "DeepGEMM requires exactly one native _C*.so artifact; "
            f"found {len(extensions)}"
        )
    extension = extensions[0]
    if not extension.is_file() or extension.is_symlink():
        raise RuntimeError("DeepGEMM native extension must be a regular file")
    suffixes = [
        suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if extension.name == f"_C{suffix}"
    ]
    if len(suffixes) != 1:
        raise RuntimeError(
            "DeepGEMM native extension does not match the active Python ABI"
        )
    return extension.resolve(), {
        "path": extension.relative_to(root).as_posix(),
        "sha256": file_sha256(extension),
        "size_bytes": extension.stat().st_size,
        "python_extension_suffix": suffixes[0],
    }


def _reset_public_runtime(module: _DeepGemmRuntime) -> dict[str, object]:
    """Reset process-global DeepGEMM controls to the upstream public defaults."""
    module.set_num_sms(0)
    module.set_tc_util(100)
    module.set_pdl(False)
    module.set_ignore_compile_dims(False)
    module.set_block_size_multiple_of(1)
    runtime: dict[str, object] = {
        "num_sms": int(module.get_num_sms()),
        "tc_util": int(module.get_tc_util()),
        "pdl": bool(module.get_pdl()),
        "ignore_compile_dims": False,
        "block_size_multiple_of": 1,
    }
    if runtime["tc_util"] != 100 or runtime["pdl"] is not False:
        raise RuntimeError(f"DeepGEMM runtime controls did not reset: {runtime}")
    return runtime


def make_case(
    groups: int,
    n: int,
    k: int,
    actual_ms: Sequence[int],
    device: torch.device,
    m_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one logical case packed for a chosen source-M alignment."""
    aligned_ms = [align(value, m_alignment) for value in actual_ms]
    m_total = sum(aligned_ms)
    a = torch.randn((m_total, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((groups, n, k), device=device, dtype=torch.bfloat16)
    layout = torch.empty(m_total, device=device, dtype=torch.int32)
    reference = torch.zeros((m_total, n), device=device, dtype=torch.float32)
    rows: list[tuple[int, int, int, int]] = []
    start = 0
    for group, (actual_m, aligned_m) in enumerate(
        zip(actual_ms, aligned_ms, strict=True)
    ):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        layout[start:actual_end] = group
        layout[actual_end:aligned_end] = -1
        a[actual_end:aligned_end] = 0
        reference[start:actual_end] = a[start:actual_end].float() @ b[group].float().T
        rows.append((group, start, actual_m, aligned_m))
        start = aligned_end
    worklist = torch.tensor(rows, device=device, dtype=torch.int32)
    return a.contiguous(), b.contiguous(), layout, reference, worklist


def repack_case_alignment(
    a: torch.Tensor,
    b: torch.Tensor,
    reference: torch.Tensor,
    worklist: torch.Tensor,
    actual_ms: Sequence[int],
    target_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack the same logical A values without redrawing inputs or changing B."""
    source_rows = worklist.detach().cpu().tolist()
    aligned_ms = [align(value, target_alignment) for value in actual_ms]
    m_total = sum(aligned_ms)
    repacked_a = torch.zeros((m_total, a.size(1)), device=a.device, dtype=a.dtype)
    repacked_reference = torch.zeros(
        (m_total, reference.size(1)),
        device=reference.device,
        dtype=reference.dtype,
    )
    layout = torch.full((m_total,), -1, device=a.device, dtype=torch.int32)
    rows: list[tuple[int, int, int, int]] = []
    target_start = 0
    for group, (actual_m, aligned_m, source_row) in enumerate(
        zip(actual_ms, aligned_ms, source_rows, strict=True)
    ):
        source_group, source_start, source_actual_m, _source_aligned_m = source_row
        if source_group != group or source_actual_m != actual_m:
            raise ValueError("source worklist does not match logical group sizes")
        source_end = source_start + actual_m
        target_end = target_start + actual_m
        repacked_a[target_start:target_end] = a[source_start:source_end]
        repacked_reference[target_start:target_end] = reference[source_start:source_end]
        layout[target_start:target_end] = group
        rows.append((group, target_start, actual_m, aligned_m))
        target_start += aligned_m
    repacked_worklist = torch.tensor(rows, device=a.device, dtype=torch.int32)
    return repacked_a, b, layout, repacked_reference, repacked_worklist


def correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
    layout: torch.Tensor,
    *,
    max_diff: float,
    require_zero_padding: bool,
) -> dict[str, object]:
    """Check valid rows and, for Helion, its zero-padding contract."""
    valid = layout >= 0
    padding_output = output[~valid]
    valid_diff = normalized_difference(output[valid], reference[valid])
    padding_max = (
        float(padding_output.float().abs().max()) if padding_output.numel() else 0.0
    )
    return {
        "valid_rows": int(valid.sum().item()),
        "padding_rows": int((~valid).sum().item()),
        "calc_diff_valid": valid_diff,
        "max_abs_padding_vs_zero": padding_max,
        "require_zero_padding": require_zero_padding,
        "ok": valid_diff <= max_diff
        and (padding_max == 0.0 or not require_zero_padding),
    }


def import_deepgemm(root: Path, m_alignment: int) -> tuple[Any, dict[str, object]]:
    """Import the pinned public DeepGEMM API and record its native artifact."""
    unexpected_environment = sorted(
        name
        for name in os.environ
        if name.startswith("DG_") and name not in _ALLOWED_DEEPGEMM_ENVIRONMENT
    )
    if unexpected_environment:
        raise RuntimeError(
            f"DeepGEMM control environment must be unset: {unexpected_environment}"
        )
    root = root.expanduser().resolve(strict=True)
    source = {
        "git_head": _clean_checkout(root, DEEPGEMM_COMMIT, "DeepGEMM"),
        "cutlass_head": _clean_checkout(
            root / "third-party" / "cutlass",
            DEEPGEMM_CUTLASS_COMMIT,
            "DeepGEMM CUTLASS",
        ),
        "fmt_head": _clean_checkout(
            root / "third-party" / "fmt",
            DEEPGEMM_FMT_COMMIT,
            "DeepGEMM fmt",
        ),
    }
    extension, extension_identity = _native_extension(root)
    original_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("deep_gemm")
    finally:
        sys.path[:] = original_path
    module_file = module.__file__
    native_file = module._C.__file__
    if module_file is None or native_file is None:
        raise RuntimeError("DeepGEMM module has no import origin")
    module_origin = Path(module_file).resolve()
    native_origin = Path(native_file).resolve()
    if not module_origin.is_relative_to(root) or native_origin != extension:
        raise RuntimeError("DeepGEMM imported outside the validated checkout")
    if str(module.__version__) != DEEPGEMM_VERSION:
        raise RuntimeError(
            f"DeepGEMM version is {module.__version__}, expected {DEEPGEMM_VERSION}"
        )
    theoretical_alignment = int(
        module.get_theoretical_mk_alignment_for_contiguous_layout()
    )
    if theoretical_alignment != m_alignment:
        raise RuntimeError(
            "DeepGEMM theoretical contiguous alignment is "
            f"{theoretical_alignment}, expected {m_alignment}"
        )
    module.set_mk_alignment_for_contiguous_layout(m_alignment)
    effective = int(module.get_mk_alignment_for_contiguous_layout())
    if effective != m_alignment:
        raise RuntimeError(f"DeepGEMM alignment is {effective}, expected {m_alignment}")
    runtime = _reset_public_runtime(cast("_DeepGemmRuntime", module))
    if _native_extension(root)[1] != extension_identity:
        raise RuntimeError("DeepGEMM native extension changed while importing")
    return module, {
        **source,
        "version": str(module.__version__),
        "module": module_origin.relative_to(root).as_posix(),
        "native_extension": extension_identity,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "m_alignment": effective,
        "theoretical_m_alignment": theoretical_alignment,
        "runtime_controls": runtime,
        "environment_controls": dict.fromkeys(
            sorted(_ALLOWED_DEEPGEMM_ENVIRONMENT & os.environ.keys()), True
        ),
    }
