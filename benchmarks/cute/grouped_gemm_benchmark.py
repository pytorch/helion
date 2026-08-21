from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
import json
import math
import os
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute import grouped_gemm_deepgemm_support as deepgemm_support
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path


ORACLE_FLOAT32_MATMUL_PRECISION = "highest"
CORRECTNESS_MAX_NORMALIZED_DIFF = 1e-5
CANONICAL_DEVICE_NAME = "NVIDIA B200"
CANONICAL_DEVICE_CAPABILITY = (10, 0)


@dataclass(frozen=True, slots=True)
class GroupedGemmCase:
    """One official BF16 variable-M grouped ``A @ B.T`` case."""

    id: str
    row_index: int
    groups: int
    expected_m_per_group: int
    n: int
    k: int
    actual_ms: tuple[int, ...]

    @property
    def total_m(self) -> int:
        return sum(self.actual_ms)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def official_cases() -> tuple[GroupedGemmCase, ...]:
    """Return the eight reviewed shapes and their deterministic seed-0 M values."""

    from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles

    return tuple(
        GroupedGemmCase(
            id=(
                f"deepgemm-large-g{shape.groups}-m{shape.expected_m_per_group}-"
                f"n{shape.n}-k{shape.k}"
            ),
            row_index=shape.row_index,
            groups=shape.groups,
            expected_m_per_group=shape.expected_m_per_group,
            n=shape.n,
            k=shape.k,
            actual_ms=actual_ms,
        )
        for shape, actual_ms in zip(
            reviewed_profiles.OFFICIAL_SHAPES,
            reviewed_profiles.official_actual_ms(seed=0),
            strict=True,
        )
    )


@dataclass(frozen=True)
class GroupedGemmInputs:
    """Deterministic compact logical BF16 inputs and their FP32 oracle."""

    case: GroupedGemmCase
    compact_a: torch.Tensor
    b: torch.Tensor
    b_n_major: torch.Tensor
    offsets: torch.Tensor
    oracle: tuple[torch.Tensor, ...]

    def compact_a_slices(self) -> tuple[torch.Tensor, ...]:
        ends = tuple(accumulate(self.case.actual_ms))
        return tuple(
            self.compact_a[start:end]
            for start, end in zip((0, *ends[:-1]), ends, strict=True)
        )

    def compact_output_slices(self, output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        expected_shape = (self.case.total_m, self.case.n)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"compact output has shape {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        ends = tuple(accumulate(self.case.actual_ms))
        return tuple(
            output[start:end] for start, end in zip((0, *ends[:-1]), ends, strict=True)
        )

    def b_for_layout(self, layout: str) -> torch.Tensor:
        if layout == "k_major":
            return self.b
        if layout == "n_major":
            return self.b_n_major
        raise ValueError(f"unsupported grouped B layout {layout!r}")


@dataclass(frozen=True)
class PackedRows:
    """One aligned physical-A representation prepared outside timing."""

    a: torch.Tensor
    worklist: torch.Tensor
    starts: tuple[int, ...]
    actual_ms: tuple[int, ...]

    def output_slices(self, output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if output.ndim != 2:
            raise ValueError(f"packed output must be rank two, got {output.shape}")
        return tuple(
            output[start : start + valid_m]
            for start, valid_m in zip(self.starts, self.actual_ms, strict=True)
        )

    def output_padding_is_zero(self, output: torch.Tensor) -> bool:
        """Return whether every physical row outside the logical groups is zero."""

        if output.ndim != 2:
            raise ValueError(f"packed output must be rank two, got {output.shape}")
        if output.size(0) != self.a.size(0):
            raise ValueError(
                f"packed output has {output.size(0)} rows, expected {self.a.size(0)}"
            )
        if not self.starts or self.starts[0] != 0:
            raise ValueError("packed output row metadata must start at row zero")
        group_ends = (*self.starts[1:], output.size(0))
        for start, valid_m, stored_end in zip(
            self.starts,
            self.actual_ms,
            group_ends,
            strict=True,
        ):
            valid_end = start + valid_m
            if not 0 <= start <= valid_end <= stored_end <= output.size(0):
                raise ValueError("packed output row metadata is inconsistent")
            padding = output[valid_end:stored_end]
            if not torch.equal(padding, torch.zeros_like(padding)):
                return False
        return True


@dataclass
class PreparedImplementation:
    """A compiled or compilable implementation with stable output mapping."""

    name: str
    call: Callable[[], object]
    output_tensors: Callable[[object], tuple[torch.Tensor, ...]]
    logical_outputs: Callable[[object], tuple[torch.Tensor, ...]]
    config: dict[str, Any]
    owners: tuple[object, ...] = ()
    track_cute_graph: bool = False


@dataclass
class CapturedImplementation:
    """A prepared implementation captured into one CUDA graph."""

    prepared: PreparedImplementation
    graph: torch.cuda.CUDAGraph
    result: object
    owners: tuple[object, ...]

    def replay(self) -> None:
        self.graph.replay()


def configure_oracle_precision() -> str:
    """Force IEEE FP32 internal matmul precision for the shared oracle."""

    torch.set_float32_matmul_precision(ORACLE_FLOAT32_MATMUL_PRECISION)
    actual = torch.get_float32_matmul_precision()
    if actual != ORACLE_FLOAT32_MATMUL_PRECISION:
        raise RuntimeError(
            "FP32 oracle precision is "
            f"{actual!r}, expected {ORACLE_FLOAT32_MATMUL_PRECISION!r}"
        )
    return actual


align = deepgemm_support.align


def compact_contiguous_a_layout() -> dict[str, str | bool]:
    layout: dict[str, str | bool] = {"kind": "compact_contiguous"}
    layout["shared_compact_allocation"] = True
    layout["logical_values_bitwise_equal"] = True
    return layout


def pack_compact_rows(inputs: GroupedGemmInputs, alignment: int) -> PackedRows:
    """Repack compact A into aligned physical rows without changing values."""

    stored_ms = tuple(align(actual_m, alignment) for actual_m in inputs.case.actual_ms)
    packed = torch.zeros(
        (sum(stored_ms), inputs.case.k),
        device=inputs.compact_a.device,
        dtype=inputs.compact_a.dtype,
    )
    starts: list[int] = []
    rows: list[tuple[int, int, int, int]] = []
    compact_start = 0
    packed_start = 0
    for group, (actual_m, stored_m) in enumerate(
        zip(inputs.case.actual_ms, stored_ms, strict=True)
    ):
        compact_end = compact_start + actual_m
        packed[packed_start : packed_start + actual_m].copy_(
            inputs.compact_a[compact_start:compact_end]
        )
        starts.append(packed_start)
        rows.append((group, packed_start, actual_m, stored_m))
        compact_start = compact_end
        packed_start += stored_m
    worklist = torch.tensor(
        rows,
        device=inputs.compact_a.device,
        dtype=torch.int32,
    )
    return PackedRows(packed, worklist, tuple(starts), inputs.case.actual_ms)


def capture_implementation(
    prepared: PreparedImplementation,
    *,
    warmups: int,
) -> CapturedImplementation:
    """Warm, capture, and retain one implementation and all pointer owners."""

    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    for _ in range(warmups):
        prepared.call()
    torch.cuda.synchronize()
    if prepared.track_cute_graph:
        import helion.runtime as helion_runtime

        with helion_runtime.cute_cuda_graph() as graph:
            result = prepared.call()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = prepared.call()
    torch.cuda.synchronize()
    return CapturedImplementation(
        prepared=prepared,
        graph=graph,
        result=result,
        owners=(*prepared.owners, result),
    )


def poison_and_replay(captured: CapturedImplementation) -> bool:
    """Poison captured outputs and prove that replay rewrites logical rows."""

    for output in captured.prepared.output_tensors(captured.result):
        output.fill_(float("nan"))
    captured.replay()
    torch.cuda.synchronize()
    logical = captured.prepared.logical_outputs(captured.result)
    return all(not bool(torch.isnan(output).any().item()) for output in logical)


normalized_difference = deepgemm_support.normalized_difference


def check_correctness(
    captured: CapturedImplementation,
    oracle: Sequence[torch.Tensor],
    *,
    max_diff: float = CORRECTNESS_MAX_NORMALIZED_DIFF,
) -> dict[str, Any]:
    """Compare every logical group against the shared FP32 oracle."""

    actual = captured.prepared.logical_outputs(captured.result)
    if len(actual) != len(oracle):
        raise ValueError(
            f"implementation produced {len(actual)} groups, expected {len(oracle)}"
        )
    groups: list[dict[str, Any]] = []
    passed = True
    max_abs = 0.0
    max_normalized_diff = 0.0
    for group, (output, expected) in enumerate(zip(actual, oracle, strict=True)):
        shape_ok = output.shape == expected.shape
        dtype_ok = output.dtype is torch.bfloat16 and expected.dtype is torch.float32
        device_ok = output.device == expected.device
        if not (shape_ok and dtype_ok and device_ok):
            groups.append(
                {
                    "group": group,
                    "ok": False,
                    "shape_ok": shape_ok,
                    "dtype_ok": dtype_ok,
                    "device_ok": device_ok,
                    "normalized_diff": math.inf,
                    "max_abs": math.inf,
                }
            )
            passed = False
            max_abs = math.inf
            max_normalized_diff = math.inf
            continue
        output_fp32 = output.float()
        difference = (output_fp32 - expected).abs()
        group_max_abs = float(difference.max().item()) if difference.numel() else 0.0
        finite = bool(
            torch.isfinite(output_fp32).all().item()
            and torch.isfinite(expected).all().item()
        )
        if not finite:
            group_max_abs = math.inf
        normalized_diff = (
            normalized_difference(output_fp32, expected) if finite else math.inf
        )
        group_ok = normalized_diff <= max_diff
        groups.append(
            {
                "group": group,
                "ok": group_ok,
                "shape_ok": shape_ok,
                "dtype_ok": dtype_ok,
                "device_ok": device_ok,
                "normalized_diff": normalized_diff,
                "max_abs": group_max_abs,
            }
        )
        passed = passed and group_ok
        max_abs = max(max_abs, group_max_abs)
        max_normalized_diff = max(max_normalized_diff, normalized_diff)
    return {
        "ok": passed,
        "max_normalized_diff": max_normalized_diff,
        "max_abs": max_abs,
        "groups": groups,
    }


def validate_capture(
    captured: CapturedImplementation,
    oracle: Sequence[torch.Tensor],
) -> dict[str, Any]:
    rewritten = poison_and_replay(captured)
    correctness = check_correctness(captured, oracle)
    return {
        **correctness,
        "poisoned_replay_rewrote_output": rewritten,
        "ok": rewritten and bool(correctness["ok"]),
    }


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash one canonical JSON provider configuration without timing metadata."""

    identity = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(identity.encode()).hexdigest()


def require_single_visible_device(value: str | None = None) -> str:
    """Require one explicit ``CUDA_VISIBLE_DEVICES`` entry."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    entries = tuple(item.strip() for item in visible.split(",") if item.strip())
    if len(entries) != 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must contain exactly one entry, got {visible!r}"
        )
    return entries[0]


def is_canonical_b200(
    device_kind: object,
    name: object,
    capability: object,
) -> bool:
    """Return whether identity exactly matches the validated CUDA B200 target."""

    return (
        name == CANONICAL_DEVICE_NAME
        and device_kind == "cuda"
        and isinstance(capability, list | tuple)
        and tuple(capability) == CANONICAL_DEVICE_CAPABILITY
    )


def clean_checkout(root: Path, expected_commit: str, label: str) -> dict[str, object]:
    """Record a provider checkout after requiring its expected clean commit."""

    root = root.expanduser().resolve(strict=True)
    head = deepgemm_support.clean_checkout(root, expected_commit, label)
    return {"path": str(root), "commit": head, "clean": True}


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
