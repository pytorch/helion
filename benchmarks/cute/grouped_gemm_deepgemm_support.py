"""Small, reproducible helpers for the grouped-GEMM DeepGEMM comparison."""

from __future__ import annotations

import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
from typing import Any
from typing import Protocol
from typing import cast

from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_CUTLASS_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_FMT_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_M_ALIGNMENT as M_ALIGNMENT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_VERSION
import torch

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


def _native_extension(root: Path) -> Path:
    extensions = sorted((root / "deep_gemm").glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError(
            "DeepGEMM requires exactly one native _C*.so artifact; "
            f"found {len(extensions)}"
        )
    extension_link = extensions[0]
    extension = extension_link.resolve(strict=True)
    if not extension.is_file() or not extension.is_relative_to(root.resolve()):
        raise RuntimeError(
            "DeepGEMM native extension must resolve to a regular file within "
            "the checkout"
        )
    suffixes = [
        suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if extension_link.name == f"_C{suffix}"
    ]
    if len(suffixes) != 1:
        raise RuntimeError(
            "DeepGEMM native extension does not match the active Python ABI"
        )
    return extension


def _reset_public_runtime(module: _DeepGemmRuntime) -> dict[str, object]:
    """Reset process-global DeepGEMM controls to the upstream public defaults."""

    requested = {
        "num_sms": 0,
        "tc_util": 100,
        "pdl": False,
        "ignore_compile_dims": False,
        "block_size_multiple_of": 1,
    }
    module.set_num_sms(0)
    module.set_tc_util(100)
    module.set_pdl(False)
    module.set_ignore_compile_dims(False)
    module.set_block_size_multiple_of(1)
    observed: dict[str, object] = {
        "num_sms": int(module.get_num_sms()),
        "tc_util": int(module.get_tc_util()),
        "pdl": bool(module.get_pdl()),
    }
    if observed["tc_util"] != 100 or observed["pdl"] is not False:
        raise RuntimeError(f"DeepGEMM runtime controls did not reset: {observed}")
    return {"requested": requested, "observed": observed}


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
        "git_head": common.clean_checkout(root, DEEPGEMM_COMMIT, "DeepGEMM"),
        "cutlass_head": common.clean_checkout(
            root / "third-party" / "cutlass",
            DEEPGEMM_CUTLASS_COMMIT,
            "DeepGEMM CUTLASS",
        ),
        "fmt_head": common.clean_checkout(
            root / "third-party" / "fmt",
            DEEPGEMM_FMT_COMMIT,
            "DeepGEMM fmt",
        ),
    }
    extension = _native_extension(root)
    extension_sha256 = common.file_sha256(extension)
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
    if (
        _native_extension(root) != extension
        or common.file_sha256(extension) != extension_sha256
    ):
        raise RuntimeError("DeepGEMM native extension changed while importing")
    return module, {
        **source,
        "version": str(module.__version__),
        "native_extension_sha256": extension_sha256,
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


def prepare_deepgemm_default(
    inputs: common.GroupedGemmInputs,
    *,
    deepgemm_root: Path,
) -> common.PreparedImplementation:
    """Prepare DeepGEMM's pinned public contiguous grouped API."""

    deep_gemm, provenance = import_deepgemm(deepgemm_root, M_ALIGNMENT)
    packed = common.pack_compact_rows(inputs, M_ALIGNMENT)
    layout = torch.full(
        (packed.a.size(0),),
        -1,
        device=packed.a.device,
        dtype=torch.int32,
    )
    for group, (start, actual_m) in enumerate(
        zip(packed.starts, packed.actual_ms, strict=True)
    ):
        layout[start : start + actual_m] = group
    output = torch.empty(
        (packed.a.size(0), inputs.case.n),
        device=packed.a.device,
        dtype=packed.a.dtype,
    )

    def call() -> torch.Tensor:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            packed.a,
            inputs.b,
            output,
            layout,
            compiled_dims="nk",
            use_psum_layout=False,
            ensure_zero_padding=False,
        )
        return output

    return common.PreparedImplementation(
        name="deepgemm-public-default",
        call=call,
        output_tensors=lambda result: (cast("torch.Tensor", result),),
        logical_outputs=lambda result: packed.output_slices(
            cast("torch.Tensor", result)
        ),
        config=common.provider_config(
            "deepgemm",
            {
                "api": {
                    "function": "m_grouped_bf16_gemm_nt_contiguous",
                    "compiled_dims": "nk",
                    "use_psum_layout": False,
                    "ensure_zero_padding": False,
                },
                "a_layout": {
                    "kind": "aligned_contiguous_layout",
                    "alignment": M_ALIGNMENT,
                    "logical_values_bitwise_equal": True,
                },
                "provenance": provenance,
            },
        ),
    )
