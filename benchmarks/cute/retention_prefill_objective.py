"""Full one-call fused Retention workload using the original objective interface."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from benchmarks.cute import mamba2_ssd_storage as storage_helpers
from benchmarks.cute import retention_prefill_inputs as fixture
from benchmarks.cute.mamba2_ssd_storage import _canonical
from benchmarks.cute.mamba2_ssd_storage import _private_storages
from benchmarks.cute.mamba2_ssd_storage import _storage_bytes
from benchmarks.cute.mamba2_ssd_storage import _tensor_abi
from benchmarks.cute.retention_prefill_inputs import STAGES
from benchmarks.cute.retention_prefill_inputs import arguments
from benchmarks.cute.retention_prefill_inputs import prepare_fixed_prefix
from benchmarks.cute.retention_prefill_inputs import sha
from examples.linear import retention_prefill
import torch

from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.kernel_args import _clone_args

if TYPE_CHECKING:
    from collections.abc import Sequence

RAW_NAMES = (
    "q",
    "k",
    "v",
    "state",
    "output",
    "final_state",
    "retention_prepared_log_gamma",
    "retention_prepared_prefix",
)
WRITTEN = frozenset(("output", "final_state"))


def alias_abi(values: Sequence[torch.Tensor]) -> str:
    groups = {}
    rows = []
    for value in values:
        storage = value.untyped_storage()
        rows.append(
            {
                "group": groups.setdefault(storage._cdata, len(groups)),
                "bytes": storage.nbytes(),
                **_tensor_abi(value),
            }
        )
    return _canonical(rows)


def check_request(request: BenchmarkRequest) -> None:
    if sha(request.compiled.__code__.co_filename) != request.source_hash:
        raise BenchmarkObjectiveError("Compiled retention member source changed")


def check_contract(
    raw: dict[str, torch.Tensor], chunk: torch.Tensor, case: dict[str, Any]
) -> None:
    b, t, h, k, v, c = (
        case[n]
        for n in ("batch", "tokens", "heads", "key_dim", "value_dim", "chunk_size")
    )
    assert case["family"] == "retention-prefill" and case["value_heads"] == h
    assert min(b, t, h, k, v, c) > 0
    shapes = {
        "q": (b, t, h, k),
        "k": (b, t, h, k),
        "v": (b, t, h, v),
        "state": (b, h, k, v),
        "output": (b, t, h, v),
        "final_state": (b, h, k, v),
        "retention_prepared_log_gamma": (h,),
        "retention_prepared_prefix": (h, (t + c - 1) // c, c),
    }
    ids = []
    for name, shape in shapes.items():
        x = raw[name]
        dtype = torch.bfloat16 if name in ("q", "k", "v", "output") else torch.float32
        assert (
            tuple(x.shape) == shape
            and x.dtype == dtype
            and x.device == raw["q"].device
            and x.is_contiguous()
        )
        ids.append(x.untyped_storage()._cdata)
    assert len(set(ids)) == len(ids), "Canonical retention inputs must not alias"
    assert (
        tuple(chunk.shape) == (b * h, (t + c - 1) // c, k, v)
        and chunk.dtype == torch.bfloat16
    )
    assert (
        chunk.is_contiguous()
        and chunk.device == raw["q"].device
        and chunk.untyped_storage()._cdata not in ids
    )


@dataclasses.dataclass(frozen=True, init=False, eq=False)
class RetentionObjective:
    """One candidate executes the entire recurrence and output on every replay."""

    case_json: str
    selected_stage: str
    partner: None
    scale: float
    raw: tuple[torch.Tensor, ...]
    chunk_template: torch.Tensor
    chunk_abi: str
    expected: tuple[torch.Tensor, ...]
    raw_abi: str
    pristine: tuple[torch.Tensor, ...]
    raw_owners: tuple[Any, ...]
    record: str

    @property
    def case(self) -> dict[str, Any]:
        return json.loads(self.case_json)

    def __init__(
        self,
        bundle: dict[str, Any],
        *,
        case: dict[str, Any],
        selected_stage: str,
        partner: None,
        provenance: dict[str, Any],
        timing_policy: dict[str, Any],
    ) -> None:
        assert selected_stage in STAGES and tuple(bundle["kernels"]) == STAGES
        assert bundle["atol"] == 0.02 and bundle["rtol"] == 0.01
        assert (
            bundle["metadata"]["initial_state_readonly"] is True
            and bundle["metadata"]["return_final_state"] is True
        )
        assert partner is None
        object.__setattr__(
            self,
            "case_json",
            _canonical({k: v for k, v in case.items() if k != "device"}),
        )
        object.__setattr__(self, "selected_stage", selected_stage)
        object.__setattr__(self, "partner", partner)
        object.__setattr__(self, "scale", bundle["metadata"]["scale"])
        assert self.scale == case["key_dim"] ** -0.5
        object.__setattr__(
            self,
            "raw",
            cast(
                "tuple[torch.Tensor, ...]",
                tuple(
                    _clone_args(
                        tuple(bundle["inputs"][name] for name in RAW_NAMES), None
                    )
                ),
            ),
        )
        object.__setattr__(
            self, "raw_owners", tuple(x.untyped_storage() for x in self.raw)
        )
        raw = dict(zip(RAW_NAMES, self.raw, strict=True))
        for name in WRITTEN:
            raw[name].zero_()
        object.__setattr__(
            self, "chunk_template", bundle["inputs"]["retention_chunk_states"]
        )
        check_contract(raw, self.chunk_template, self.case)
        object.__setattr__(self, "chunk_abi", alias_abi((self.chunk_template,)))
        # Reuse the exact invariant recipe, with no raw tensor argument. This is
        # admission-time work only, and cannot enter a measured invocation.
        gamma, prefix = prepare_fixed_prefix(
            case["heads"], case["tokens"], case["chunk_size"], raw["q"].device
        )
        assert torch.equal(gamma, raw["retention_prepared_log_gamma"])
        assert torch.equal(prefix, raw["retention_prepared_prefix"])
        object.__setattr__(
            self, "expected", tuple(x.clone() for x in bundle["reference"]())
        )
        object.__setattr__(self, "raw_abi", alias_abi(self.raw))
        object.__setattr__(
            self, "pristine", tuple(_storage_bytes(x).clone() for x in self.raw)
        )
        records = []
        for x in self.raw:
            data = _storage_bytes(x).cpu().numpy().tobytes()
            records.append(
                {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        object.__setattr__(
            self,
            "record",
            _canonical(
                {
                    "schema": "helion.experimental.retention_fused_prefill_objective.v1",
                    "case": self.case,
                    "stage_order": STAGES,
                    "selected_stage": selected_stage,
                    "partner": None,
                    "raw_names": RAW_NAMES,
                    "raw_abi": json.loads(self.raw_abi),
                    "raw_storage": records,
                    "chunk_abi": json.loads(self.chunk_abi),
                    "scale": self.scale,
                    "atol": 0.02,
                    "rtol": 0.01,
                    "equal_nan": False,
                    "contract": bundle["metadata"],
                    "recipe_sha256": sha(__file__),
                    "storage_sha256": sha(storage_helpers.__file__),
                    "oracle_sha256": sha(fixture.__file__),
                    "kernel_sha256": sha(retention_prefill.__file__),
                    "provenance": provenance,
                    "timing": timing_policy,
                }
            ),
        )

    @property
    def raw_args(self) -> tuple[torch.Tensor, ...]:
        return self.raw

    def cache_record(self) -> dict[str, Any]:
        assert self.partner is None
        if any(
            x.untyped_storage()._cdata != owner._cdata
            for x, owner in zip(self.raw, self.raw_owners, strict=True)
        ):
            raise BenchmarkObjectiveError("Frozen factory raw owner changed")
        if (
            alias_abi(self.raw) != self.raw_abi
            or alias_abi((self.chunk_template,)) != self.chunk_abi
        ):
            raise BenchmarkObjectiveError("Retention raw or representative ABI changed")
        for raw, pristine in zip(self.raw, self.pristine, strict=True):
            if not torch.equal(_storage_bytes(raw), pristine):
                raise BenchmarkObjectiveError("Frozen factory raw values changed")
        return json.loads(self.record)

    def bind(
        self, request: BenchmarkRequest, private_args: Sequence[object]
    ) -> BenchmarkWorkload:
        private_args = cast("Sequence[torch.Tensor]", private_args)
        self.cache_record()
        check_request(request)
        if alias_abi(private_args) != self.raw_abi:
            raise BenchmarkObjectiveError("Retention private ABI changed")
        if {x.untyped_storage()._cdata for x in private_args} & {
            x.untyped_storage()._cdata for x in self.raw
        }:
            raise BenchmarkObjectiveError("Private retention inputs alias factory")
        raw = dict(zip(RAW_NAMES, private_args, strict=True))
        check_contract(raw, self.chunk_template, self.case)
        storage = _private_storages(raw, WRITTEN)
        call_args = arguments(raw, self.case, self.scale)[0]
        originals = tuple(private_args)
        owners = tuple(x.untyped_storage() for x in originals)
        last = None
        closed = False

        def check_private() -> None:
            assert alias_abi(private_args) == self.raw_abi
            assert all(x is y for x, y in zip(private_args, originals, strict=True))
            assert all(
                x.untyped_storage()._cdata == owner._cdata
                for x, owner in zip(private_args, owners, strict=True)
            )

        def reset() -> None:
            assert not closed
            check_private()
            raw["output"].zero_()
            raw["final_state"].zero_()
            if last is not None:
                last[0].zero_()

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal last
            assert not closed
            result = request.compiled(*call_args)
            assert isinstance(result, tuple) and len(result) == 2
            output, final = result
            assert final is raw["final_state"]
            assert isinstance(output, torch.Tensor)
            assert _tensor_abi(output) == _tensor_abi(raw["output"])
            assert output.untyped_storage()._cdata not in {
                x.untyped_storage()._cdata for x in private_args
            }
            last = (output, final)
            return last

        def validate(output: object) -> None:
            assert not closed and last is not None
            check_private()
            assert isinstance(output, tuple) and len(output) == 2
            assert output[0] is last[0] and output[1] is raw["final_state"]
            for actual, expected in zip(output, self.expected, strict=True):
                assert (actual.shape, actual.dtype, actual.device) == (
                    expected.shape,
                    expected.dtype,
                    expected.device,
                )
                torch.testing.assert_close(
                    actual, expected, atol=0.02, rtol=0.01, equal_nan=False
                )
            for item in storage:
                item.validate()

        def close() -> None:
            nonlocal closed
            closed = True

        return BenchmarkWorkload(run, reset, validate, close)


def selected_guards(
    objective: RetentionObjective, request: BenchmarkRequest
) -> list[str]:
    private = cast("tuple[torch.Tensor, ...]", _clone_args(objective.raw_args, None))
    owners = tuple(x.untyped_storage() for x in private)
    raw = dict(zip(RAW_NAMES, private, strict=True))
    job = objective.bind(request, private)
    rows = []
    try:
        for name, poison in (
            ("original", 0.0),
            ("output_nan", float("nan")),
            ("output_finite", 17.0),
            ("repeat_reset", 0.0),
            ("repeat_without_reset", None),
        ):
            if poison is not None:
                job.reset()
                raw["output"].fill_(poison)
                raw["final_state"].fill_(poison)
            result = job.run()
            job.validate(result)
            assert all(
                x.untyped_storage()._cdata == owner._cdata
                for x, owner in zip(private, owners, strict=True)
            )
            rows.append(name)
        objective.cache_record()
        return rows
    finally:
        job.close()


def make_objective(
    bundle: dict[str, Any], case: dict[str, Any], timing: dict[str, Any]
) -> RetentionObjective:
    return RetentionObjective(
        bundle,
        case=case,
        selected_stage=STAGES[0],
        partner=None,
        provenance={
            "public_program": "examples.linear.retention_prefill.retention_prefill_fixed_decay",
            "public_program_sha256": sha(retention_prefill.__file__),
        },
        timing_policy=timing,
    )


def baseline_workload(bundle: dict[str, Any]) -> BenchmarkWorkload:
    """The original complete baseline call and physical-owner/readonly checks."""
    expected = bundle["reference"]()
    raw = bundle["inputs"]
    originals = dict(raw)
    owners = tuple(
        (name, value, value.untyped_storage()) for name, value in raw.items()
    )
    abi = alias_abi(tuple(raw.values()))
    storage = _private_storages(raw, WRITTEN)

    def validate(values: object) -> None:
        assert isinstance(values, tuple) and len(values) == len(expected) == 2
        for actual, reference in zip(values, expected, strict=True):
            assert (actual.shape, actual.dtype, actual.device) == (
                reference.shape,
                reference.dtype,
                reference.device,
            )
            torch.testing.assert_close(
                actual, reference, atol=0.02, rtol=0.01, equal_nan=False
            )
        assert set(raw) == set(originals) and alias_abi(tuple(raw.values())) == abi
        for name, value, owner in owners:
            assert raw[name] is originals[name] is value
            assert value.untyped_storage()._cdata == owner._cdata
        for item in storage:
            item.validate()

    def close() -> None:
        pass

    return BenchmarkWorkload(bundle["run"], bundle["reset"], validate, close)
