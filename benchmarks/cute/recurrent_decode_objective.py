"""One raw decode invocation, private physical storage and complete output/state oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast

from benchmarks.cute.recurrent_decode_inputs import _oracle
from examples.linear.recurrent_decode import kernel_args
import torch
from torch.utils._pytree import tree_flatten

from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.kernel_args import _clone_args


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def function_record(fn: Callable[..., object]) -> dict[str, str]:
    path = Path(fn.__code__.co_filename).resolve()
    return {
        "name": fn.__qualname__,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def storage_bytes(t: torch.Tensor) -> torch.Tensor:
    s = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        s, 0, (s.nbytes(),), (1,)
    )


def abi(args: tuple[Any, ...]) -> str:
    flat, spec = tree_flatten(args)
    groups: dict[int, int] = {}
    identities: dict[int, int] = {}
    rows = []
    for value in flat:
        if isinstance(value, torch.Tensor):
            s = value.untyped_storage()
            rows.append(
                {
                    "tensor": identities.setdefault(id(value), len(identities)),
                    "storage": groups.setdefault(s._cdata, len(groups)),
                    "bytes": s.nbytes(),
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "offset": value.storage_offset(),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "base_residue256": s.data_ptr() % 256,
                }
            )
        else:
            assert type(value) in (int, float, bool, str, type(None))
            rows.append({"type": type(value).__name__, "value": value})
    return canonical({"tree": str(spec), "leaves": rows})


def results(raw: tuple[Any, ...], indices: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    args, _kernel_args, _inputs, pool = raw
    values = tuple(args[index] for index in indices)
    return (*values, pool[:1], pool[-1:]) if pool is not None else values


def compare(
    actual: object, expected: tuple[torch.Tensor, ...], atol: float, rtol: float
) -> None:
    assert isinstance(actual, tuple) and len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        assert isinstance(value, torch.Tensor)
        assert (value.shape, value.dtype, value.device) == (
            reference.shape,
            reference.dtype,
            reference.device,
        )
        torch.testing.assert_close(
            value, reference, atol=atol, rtol=rtol, equal_nan=False
        )


class DecodeObjective:
    """The existing objective interface, with no stage-local ranking or partners."""

    def __init__(
        self,
        bundle: dict[str, Any],
        *,
        case: dict[str, Any],
        provenance: dict[str, Any],
        timing_policy: dict[str, Any],
    ) -> None:
        binding = bundle["objective_binding"]
        assert len(bundle["kernels"]) == 1
        assert (bundle["atol"], bundle["rtol"]) == (0.02, 0.01)
        self.mutable = tuple(binding["mutable_indices"])
        self.written = self.mutable
        if case["family"] in ("gdn-decode", "gdn-mtp"):
            args = binding["canonical_args"]
            assert type(args[13]) is bool and type(args[14]) is bool
            assert args[13] == bundle["metadata"]["update_state"]
            assert args[14] == bundle["metadata"]["checkpoints"]
            self.written = tuple(
                index
                for index in self.mutable
                if not (index == 7 and not args[13])
                and not (index == 10 and not args[14])
            )
        self.indices = tuple(binding["result_indices"])
        self.atol, self.rtol = bundle["atol"], bundle["rtol"]
        self._raw = tuple(
            _clone_args(
                (
                    binding["canonical_args"],
                    binding["kernel_args"],
                    tuple(bundle["inputs"].values()),
                    binding["state_pool"],
                ),
                None,
            )
        )
        self._abi = abi(self._raw)
        seen = set()
        raw_storage = []
        pristine = []
        for value in tree_flatten(self._raw)[0]:
            if (
                isinstance(value, torch.Tensor)
                and value.untyped_storage()._cdata not in seen
            ):
                seen.add(value.untyped_storage()._cdata)
                pristine.append(storage_bytes(value).clone())
                data = storage_bytes(value).cpu().numpy().tobytes()
                raw_storage.append(
                    {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                )
        self._pristine = tuple(pristine)
        expected = tuple(value.clone() for value in bundle["reference"]())
        # Re-run the original per-kernel callback on an alias-preserving private
        # argument tree and compare every canonical output/state/canary. This is
        # an explicit migration of the callback, including KDA's split ABI.
        callback_raw = tuple(_clone_args(self._raw, None))
        binding["kernel_reference"](*callback_raw[1])
        compare(results(callback_raw, self.indices), expected, self.atol, self.rtol)
        self._expected = expected
        self._record = canonical(
            {
                "raw_storage": raw_storage,
                "schema": "canonical.decode.objective.v1",
                "execution": "one original canonical call; candidate directly substituted",
                "case": {k: v for k, v in case.items() if k != "device"},
                "stage_order": list(bundle["kernels"]),
                "partners": [],
                "raw_abi": self._abi,
                "input_names": list(bundle["inputs"]),
                "mutable_indices": self.mutable,
                "semantic_written_indices": self.written,
                "result_indices": self.indices,
                "contract": bundle["metadata"],
                "atol": self.atol,
                "rtol": self.rtol,
                "equal_nan": False,
                "original_baseline": function_record(binding["baseline"]),
                "migrated_kernel_reference": function_record(
                    binding["kernel_reference"]
                ),
                "recipe_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "reset": "original mutable views and complete original state pool before every invocation, outside events",
                "readonly": "all remaining physical storage bytes exact, including padding and pool guards",
                "provenance": provenance,
                "timing": timing_policy,
            }
        )

    @property
    def raw_args(self) -> tuple[Any, ...]:
        return self._raw

    def cache_record(self) -> dict[str, Any]:
        if abi(self._raw) != self._abi:
            raise BenchmarkObjectiveError("Canonical raw argument ABI changed")
        unique = {
            value.untyped_storage()._cdata: value
            for value in tree_flatten(self._raw)[0]
            if isinstance(value, torch.Tensor)
        }
        for value, pristine in zip(unique.values(), self._pristine, strict=True):
            if not torch.equal(storage_bytes(value), pristine):
                raise BenchmarkObjectiveError("Frozen canonical raw values changed")
        return json.loads(self._record)

    def bind(
        self, request: BenchmarkRequest, private_args: tuple[Any, ...]
    ) -> BenchmarkWorkload:
        self.cache_record()
        if abi(private_args) != self._abi:
            raise BenchmarkObjectiveError("Private canonical argument ABI changed")
        source = Path(request.compiled.__code__.co_filename)
        if hashlib.sha256(source.read_bytes()).hexdigest() != request.source_hash:
            raise BenchmarkObjectiveError("Candidate source differs from its request")
        args, kernel_args, _inputs, pool = private_args
        flat, _ = tree_flatten(private_args)
        unique: dict[int, torch.Tensor] = {}
        for value in flat:
            if isinstance(value, torch.Tensor):
                unique.setdefault(value.untyped_storage()._cdata, value)
        originals = {
            value.untyped_storage()._cdata
            for value in tree_flatten(self._raw)[0]
            if isinstance(value, torch.Tensor)
        }
        if originals.intersection(unique):
            raise BenchmarkObjectiveError("Candidate aliases objective-owned inputs")
        saved = {
            key: (storage_bytes(value), storage_bytes(value).clone())
            for key, value in unique.items()
        }
        writable: dict[int, list[tuple[int, int]]] = {key: [] for key in unique}
        pristine = {}
        for index in self.mutable:
            value = args[index]
            if not value.is_contiguous():
                raise BenchmarkObjectiveError(
                    "Canonical writable view must be contiguous"
                )
            pristine[index] = value.clone()
            start = value.storage_offset() * value.element_size()
            if index in self.written:
                writable[value.untyped_storage()._cdata].append(
                    (start, start + value.numel() * value.element_size())
                )
        pool_pristine = pool.clone() if pool is not None else None
        protected = []
        for key, (current, before) in saved.items():
            cursor = 0
            for start, end in sorted(writable[key]):
                if start > cursor:
                    protected.append((current[cursor:start], before[cursor:start]))
                cursor = max(cursor, end)
            if cursor < current.numel():
                protected.append((current[cursor:], before[cursor:]))
        closed = False

        def reset() -> None:
            assert not closed
            for index, before in pristine.items():
                args[index].copy_(before)
            if pool is not None:
                pool.copy_(pool_pristine)

        def run() -> tuple[torch.Tensor, ...]:
            assert not closed
            request.compiled(*kernel_args)
            return results(private_args, self.indices)

        def validate(output: object) -> None:
            assert not closed
            assert abi(private_args) == self._abi, "Candidate changed input metadata"
            expected_views = results(private_args, self.indices)
            assert isinstance(output, tuple) and len(output) == len(expected_views)
            for returned, actual in zip(output, expected_views, strict=True):
                assert (
                    returned.untyped_storage()._cdata == actual.untyped_storage()._cdata
                )
                assert (
                    returned.shape,
                    returned.stride(),
                    returned.storage_offset(),
                    returned.dtype,
                ) == (
                    actual.shape,
                    actual.stride(),
                    actual.storage_offset(),
                    actual.dtype,
                )
            compare(output, self._expected, self.atol, self.rtol)
            for current, before in protected:
                assert torch.equal(current, before), (
                    "Candidate changed readonly input, padding or state-pool guard"
                )

        def close() -> None:
            nonlocal closed
            closed = True

        return BenchmarkWorkload(run, reset, validate, close)


class ProjectedObjective:
    """Retain serving arguments and all projected views for one compiled workload."""

    def __init__(self, original: DecodeObjective, family: str) -> None:
        self.original = original
        self.family = family
        self.projection_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        kernel_args(family, original.raw_args[1])

    @property
    def raw_args(self) -> tuple[Any, ...]:
        return self.original.raw_args

    def cache_record(self) -> dict[str, Any]:
        return {
            "original_objective": self.original.cache_record(),
            "projection_source_sha256": self.projection_hash,
            "family": self.family,
            "projection": "raw serving arguments to retained zero-copy kernel views",
        }

    def bind(
        self, request: BenchmarkRequest, private: tuple[Any, ...]
    ) -> BenchmarkWorkload:
        self.cache_record()
        source = Path(request.compiled.__code__.co_filename)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == request.source_hash
        prepared = kernel_args(self.family, private[1])
        tensors = tuple(
            x for x in tree_flatten(prepared)[0] if isinstance(x, torch.Tensor)
        )
        owners = tuple(x.untyped_storage() for x in tensors)
        facts = tuple(
            (
                id(x),
                x.untyped_storage()._cdata,
                x.data_ptr(),
                x.shape,
                x.stride(),
                x.storage_offset(),
                x.dtype,
            )
            for x in tensors
        )

        def projected_call(*raw: object) -> torch.Tensor:
            # These exact view objects outlive native argument binding and replay.
            request.compiled(*prepared)
            return cast("torch.Tensor", raw[9 if self.family == "gdn-decode" else 10])

        adapted = BenchmarkRequest(
            request.config_json, projected_call, self.projection_hash
        )
        work = self.original.bind(adapted, private)

        def validate(output: object) -> None:
            work.validate(output)
            assert owners
            assert facts == tuple(
                (
                    id(x),
                    x.untyped_storage()._cdata,
                    x.data_ptr(),
                    x.shape,
                    x.stride(),
                    x.storage_offset(),
                    x.dtype,
                )
                for x in tensors
            )
            self.cache_record()

        return BenchmarkWorkload(work.run, work.reset, validate, work.close)


def make_objective(
    bundle: dict[str, Any], case: dict[str, Any], timing_policy: dict[str, Any]
) -> DecodeObjective | ProjectedObjective:
    original = DecodeObjective(
        bundle,
        case=case,
        provenance={"program": "examples.linear.recurrent_decode"},
        timing_policy=timing_policy,
    )
    if case["family"] in ("gdn-decode", "kda-decode"):
        return ProjectedObjective(original, case["family"])
    return original


def selected_guards(
    objective: DecodeObjective | ProjectedObjective,
    request: BenchmarkRequest,
    bundle: dict[str, Any],
    case: dict[str, Any],
    timing_policy: dict[str, Any],
) -> list[str]:
    """Original poison/reset cases followed by an independently referenced update."""
    factory = (
        objective.original if isinstance(objective, ProjectedObjective) else objective
    )
    private: tuple[Any, ...] = tuple(_clone_args(factory.raw_args, None))
    work = objective.bind(request, private)
    rows = []
    try:
        for label, poison in (
            ("original", None),
            ("output_nan", float("nan")),
            ("output_finite", 17.0),
            ("repeat_reset", None),
        ):
            work.reset()
            if poison is not None:
                private[0][factory.indices[0]].fill_(poison)
                if case["family"] == "gdn-mtp":
                    private[0][10].fill_(poison)
            work.validate(work.run())
            rows.append(label)
        reference_raw: tuple[Any, ...] = tuple(_clone_args(factory.raw_args, None))
        binding = bundle["objective_binding"]
        binding["baseline"](*reference_raw[0])
        args2, kernel_args2, inputs2, pool2 = reference_raw
        second = dict(bundle)
        second["inputs"] = dict(zip(bundle["inputs"], inputs2, strict=True))
        second["objective_binding"] = dict(
            binding, canonical_args=args2, kernel_args=kernel_args2, state_pool=pool2
        )
        second_oracle = _oracle(
            binding["baseline"], args2, factory.mutable, factory.indices
        )
        canaries = () if pool2 is None else (pool2[:1].clone(), pool2[-1:].clone())
        second["reference"] = lambda: (*second_oracle(), *(x.clone() for x in canaries))
        next_objective = make_objective(second, case, timing_policy)
        work.validate(results(private, factory.indices))
        next_work = next_objective.bind(request, private)
        try:
            next_work.validate(next_work.run())
            objective.cache_record()
            next_objective.cache_record()
        finally:
            next_work.close()
        rows.append("second_update_without_reset")
    finally:
        work.reset()
        work.close()
    return rows


def baseline_workload(bundle: dict[str, Any]) -> BenchmarkWorkload:
    """Preserve all readonly bytes, including packed storage and state-pool guards."""
    binding = bundle["objective_binding"]
    args = binding["canonical_args"]
    expected = tuple(x.clone() for x in bundle["reference"]())
    tensors = tuple(x for x in bundle["inputs"].values() if isinstance(x, torch.Tensor))
    owners = tuple(x.untyped_storage() for x in tensors)
    facts = abi(tensors)
    writable: dict[int, list[tuple[int, int]]] = {}
    for index in binding["mutable_indices"]:
        if bundle["metadata"]["family"] in ("gdn-decode", "gdn-mtp"):
            if index == 7 and not args[13] or index == 10 and not args[14]:
                continue
        x = args[index]
        start = x.storage_offset() * x.element_size()
        writable.setdefault(x.untyped_storage()._cdata, []).append(
            (start, start + x.numel() * x.element_size())
        )
    protected = []
    seen = set()
    for tensor in tensors:
        key = tensor.untyped_storage()._cdata
        if key in seen:
            continue
        seen.add(key)
        current = storage_bytes(tensor)
        before = current.clone()
        cursor = 0
        for start, end in sorted(writable.get(key, [])):
            if start > cursor:
                protected.append((current[cursor:start], before[cursor:start]))
            cursor = max(cursor, end)
        if cursor < current.numel():
            protected.append((current[cursor:], before[cursor:]))

    def validate(output: object) -> None:
        assert owners and abi(tensors) == facts
        compare(output, expected, bundle["atol"], bundle["rtol"])
        for current, before in protected:
            assert torch.equal(current, before), "Baseline changed readonly storage"

    return BenchmarkWorkload(bundle["run"], bundle["reset"], validate, lambda: None)
