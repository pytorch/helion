"""One or four original SSD calls; explicit original callback/full-oracle checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.cute.mamba2_ssd_inputs import RAW_NAMES
from benchmarks.cute.mamba2_ssd_inputs import SHORT
from benchmarks.cute.mamba2_ssd_inputs import sha
from benchmarks.cute.mamba2_ssd_inputs import stages
from benchmarks.cute.mamba2_ssd_storage import _canonical
from benchmarks.cute.mamba2_ssd_storage import _private_storages
from benchmarks.cute.mamba2_ssd_storage import _storage_bytes
from benchmarks.cute.mamba2_ssd_storage import _tensor_abi
import torch

from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.kernel_args import _clone_args

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import FunctionType

StageResult = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]


def abi(values: Sequence[torch.Tensor]) -> str:
    groups = {}
    rows = []
    for x in values:
        s = x.untyped_storage()
        rows.append(
            {
                "group": groups.setdefault(s._cdata, len(groups)),
                "bytes": s.nbytes(),
                **_tensor_abi(x),
            }
        )
    return _canonical(rows)


def function_record(fn: FunctionType) -> dict[str, str]:
    return {
        "name": fn.__qualname__,
        "path": str(Path(fn.__code__.co_filename).resolve()),
        "sha256": sha(fn.__code__.co_filename),
    }


def request_check(request: BenchmarkRequest) -> None:
    if sha(request.compiled.__code__.co_filename) != request.source_hash:
        raise BenchmarkObjectiveError("SSD compiled member source changed")


def stage_arguments(
    raw: dict[str, torch.Tensor], case: dict[str, Any], index: int, values: list[Any]
) -> tuple[Any, ...]:
    x, dt, a, b, c, d = (raw[n] for n in RAW_NAMES)
    chunk = case["chunk_size"]
    if stages(case) == SHORT:
        return x, dt, a, b, c, d
    if index == 0:
        return x, b, dt, a, chunk
    if index == 1:
        return values[0][0], values[0][1]
    if index == 2:
        return b, c, chunk
    return x, c, d, values[0][2], values[0][3], values[2], values[1], chunk


def expected_stage(
    case: dict[str, Any], index: int
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    b, t, h, p, g, n, c = (
        case[k]
        for k in (
            "batch",
            "tokens",
            "heads",
            "value_dim",
            "groups",
            "state_dim",
            "chunk_size",
        )
    )
    chunks = t // c
    if stages(case) == SHORT or index == 3:
        return (((b, t, h, p), torch.bfloat16),)
    if index == 0:
        return (
            ((b, chunks, h, p, n), torch.float32),
            ((b, chunks, h), torch.float32),
            ((b, chunks, h, c), torch.float32),
            ((b, chunks, h, c), torch.float32),
        )
    if index == 1:
        return (((b, chunks, h, p, n), torch.bfloat16),)
    return (((b, chunks, g, c, c), torch.float32),)


def flat_result(result: StageResult) -> tuple[torch.Tensor, ...]:
    return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def check_result(
    result: StageResult,
    case: dict[str, Any],
    index: int,
    used: set[int],
    device: torch.device,
) -> None:
    values = flat_result(result)
    spec = expected_stage(case, index)
    assert len(values) == len(spec)
    for x, (shape, dtype) in zip(values, spec, strict=True):
        assert isinstance(x, torch.Tensor)
        assert tuple(x.shape) == shape and x.dtype == dtype and x.device == device
        assert x.is_contiguous() and x.storage_offset() == 0
        assert x.untyped_storage().nbytes() == x.numel() * x.element_size()
        identity = x.untyped_storage()._cdata
        assert identity not in used, "SSD output aliases raw or earlier live allocation"
        used.add(identity)


def compare(actual: StageResult, expected: StageResult) -> None:
    left = flat_result(actual)
    right = flat_result(expected)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        assert (a.shape, a.dtype, a.device) == (b.shape, b.dtype, b.device)
        torch.testing.assert_close(a, b, atol=0.002, rtol=0.01, equal_nan=False)


@dataclasses.dataclass(frozen=True, init=False, eq=False)
class MambaObjective:
    case_json: str
    raw: tuple[torch.Tensor, ...]
    raw_abi: str
    pristine: tuple[torch.Tensor, ...]
    selected: int
    partners: tuple[tuple[int, BenchmarkRequest], ...]
    references: tuple[Any, ...]
    expected: tuple[torch.Tensor, ...]
    record: str

    @property
    def case(self) -> dict[str, Any]:
        return json.loads(self.case_json)

    def __init__(
        self,
        bundle: dict[str, Any],
        *,
        case: dict[str, Any],
        selected: int,
        partners: dict[int, BenchmarkRequest],
        provenance: dict[str, Any],
        timing_policy: dict[str, Any],
    ) -> None:
        case = {k: v for k, v in case.items() if k != "device"}
        order = stages(case)
        assert (
            case["family"] == "mamba2-ssd" and case["tokens"] % case["chunk_size"] == 0
        )
        assert case["heads"] % case["groups"] == 0 and case["tokens"] > 0
        assert tuple(bundle["kernels"]) == order
        assert (bundle["atol"], bundle["rtol"]) == (0.002, 0.01)
        assert selected in range(len(order)) and set(partners) == set(
            range(len(order))
        ) - {selected}
        references = tuple(bundle["objective_binding"]["references"])
        assert len(references) == len(order)
        for ref, original in zip(references, bundle["kernels"].values(), strict=True):
            assert ref is original.settings.autotune_baseline_fn
        raw = cast(
            "tuple[torch.Tensor, ...]",
            tuple(_clone_args(tuple(bundle["inputs"][n] for n in RAW_NAMES), None)),
        )
        assert len({x.untyped_storage()._cdata for x in raw}) == 6
        b, t, h, p, g, n = (
            case[k]
            for k in ("batch", "tokens", "heads", "value_dim", "groups", "state_dim")
        )
        specs = (
            ((b, t, h, p), torch.bfloat16),
            ((b, t, h), torch.float32),
            ((h,), torch.float32),
            ((b, t, g, n), torch.bfloat16),
            ((b, t, g, n), torch.bfloat16),
            ((h,), torch.bfloat16),
        )
        for x, (shape, dtype) in zip(raw, specs, strict=True):
            assert (
                tuple(x.shape) == shape
                and x.dtype == dtype
                and x.device == raw[0].device
                and x.is_contiguous()
            )
        expected = tuple(x.clone() for x in bundle["reference"]())
        # Replay the original per-stage callbacks once as a complete pipeline,
        # then compare to the original separate full SSD oracle. Validation also
        # checks every actual compiled stage against its unchanged callback.
        reference_values = []
        named = dict(zip(RAW_NAMES, raw, strict=True))
        for i, ref in enumerate(references):
            reference_values.append(
                ref(*stage_arguments(named, case, i, reference_values))
            )
        compare(reference_values[-1], expected)
        stores = []
        for x in raw:
            data = _storage_bytes(x).cpu().numpy().tobytes()
            stores.append(
                {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        for request in partners.values():
            request_check(request)
        fields = {
            "case_json": _canonical(case),
            "raw": raw,
            "raw_abi": abi(raw),
            "pristine": tuple(_storage_bytes(x).clone() for x in raw),
            "selected": selected,
            "partners": tuple(sorted(partners.items())),
            "references": references,
            "expected": expected,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        record = {
            "schema": "helion.experimental.mamba_ssd_objective.v1",
            "case": case,
            "stage_order": order,
            "selected_stage": order[selected],
            "partners": [
                {
                    "stage": order[i],
                    "config": r.config.config,
                    "source_hash": r.source_hash,
                    "request_digest": r.digest,
                }
                for i, r in self.partners
            ],
            "raw_abi": json.loads(self.raw_abi),
            "raw_storage": stores,
            "raw_names": RAW_NAMES,
            "contract": bundle["metadata"],
            "atol": 0.002,
            "rtol": 0.01,
            "output_only": True,
            "callbacks": [function_record(ref) for ref in references],
            "oracle": function_record(bundle["reference"]),
            "recipe_sha256": sha(__file__),
            "storage_sha256": sha(Path(__file__).with_name("mamba2_ssd_storage.py")),
            "provenance": provenance,
            "timing": timing_policy,
        }
        object.__setattr__(self, "record", _canonical(record))

    @property
    def raw_args(self) -> tuple[torch.Tensor, ...]:
        return self.raw

    def cache_record(self) -> dict[str, Any]:
        assert abi(self.raw) == self.raw_abi
        for raw, pristine in zip(self.raw, self.pristine, strict=True):
            if not torch.equal(_storage_bytes(raw), pristine):
                raise BenchmarkObjectiveError("Frozen factory raw values changed")
        for _, request in self.partners:
            request_check(request)
        for ref, expected in zip(
            self.references, json.loads(self.record)["callbacks"], strict=True
        ):
            assert function_record(ref) == expected
        return json.loads(self.record)

    def bind(
        self, request: BenchmarkRequest, private_args: Sequence[object]
    ) -> BenchmarkWorkload:
        self.cache_record()
        request_check(request)
        private_args = cast("Sequence[torch.Tensor]", private_args)
        assert abi(private_args) == self.raw_abi
        assert not {x.untyped_storage()._cdata for x in private_args} & {
            x.untyped_storage()._cdata for x in self.raw
        }
        raw = dict(zip(RAW_NAMES, private_args, strict=True))
        storage = _private_storages(raw, frozenset())
        callables = {i: r.compiled for i, r in self.partners}
        callables[self.selected] = request.compiled
        last = None
        retained = []
        closed = False

        def reset() -> None:
            assert not closed
            # Original output-only SSD reset is a no-op: all scratch/output is
            # freshly allocated and fully overwritten by the original kernels.

        def run() -> tuple[torch.Tensor, ...]:
            nonlocal last, retained
            assert not closed
            values = []
            traces = []
            used = {x.untyped_storage()._cdata for x in private_args}
            for i in range(len(callables)):
                args = stage_arguments(raw, self.case, i, values)
                result = cast("StageResult", callables[i](*args))
                check_result(result, self.case, i, used, raw["x"].device)
                values.append(result)
                traces.append((args, result))
            retained = traces
            last = (values[-1],)
            return last

        def validate(output: object) -> None:
            assert not closed and last is not None
            assert (
                isinstance(output, tuple) and len(output) == 1 and output[0] is last[0]
            )
            for ref, (args, result) in zip(self.references, retained, strict=True):
                compare(result, ref(*args))
            compare(output, self.expected)
            for s in storage:
                s.validate()

        def close() -> None:
            nonlocal closed
            closed = True

        return BenchmarkWorkload(run, reset, validate, close)


def selected_guards(objective: MambaObjective, request: BenchmarkRequest) -> list[str]:
    """Original output-only checks, with and without the no-op reset."""
    private = tuple(_clone_args(objective.raw_args, None))
    member = objective.bind(request, private)
    original_record = objective.cache_record()
    labels = ("original", "repeat_reset", "repeat_without_reset")
    try:
        for label in labels:
            if label != "repeat_without_reset":
                member.reset()
            member.validate(member.run())
            assert objective.cache_record() == original_record
    finally:
        member.close()
    return list(labels)


def baseline_workload(bundle: dict[str, Any]) -> BenchmarkWorkload:
    """Retain baseline outputs/metadata and protect every readonly physical byte."""
    expected = bundle["reference"]()
    raw = bundle["inputs"]

    def identity(x: torch.Tensor) -> tuple[Any, ...]:
        return (
            id(x),
            x.untyped_storage()._cdata,
            x.data_ptr(),
            tuple(x.shape),
            tuple(x.stride()),
            x.storage_offset(),
            x.dtype,
        )

    owners = [
        (name, x, x.untyped_storage(), identity(x))
        for name, x in raw.items()
        if isinstance(x, torch.Tensor)
    ]
    protected = [
        (name, x, _storage_bytes(x).clone())
        for name, x, _, _ in owners
        if name != "work_output"
    ]

    def validate(actual: object) -> None:
        actual = cast("tuple[torch.Tensor, ...]", actual)
        assert len(actual) == len(expected) == 1
        for value, reference in zip(actual, expected, strict=True):
            assert value.shape == reference.shape and value.dtype == reference.dtype
            torch.testing.assert_close(
                value, reference, atol=0.002, rtol=0.01, equal_nan=False
            )
        for name, x, owner, before in owners:
            assert raw[name] is x and identity(x) == before
            assert x.untyped_storage()._cdata == owner._cdata
        for name, x, before in protected:
            assert torch.equal(_storage_bytes(x), before), name

    def close() -> None:
        pass

    return BenchmarkWorkload(bundle["run"], bundle["reset"], validate, close)
