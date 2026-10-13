"""First legal existing partner, without timing, ranking or search mutation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import traceback
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute.mamba2_ssd_storage import _canonical
from benchmarks.cute.mamba2_ssd_storage import _private_storages
from benchmarks.cute.mamba2_ssd_storage import _tensor_abi
import torch
from torch._inductor.runtime.triton_compat import OutOfResources
from torch.utils._pytree import tree_flatten

from helion import exc
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.logger import match_unrecoverable_runtime_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel


class NoLegalPartner(BenchmarkObjectiveError):
    """The default and finite original normalized seed list were exhausted."""


@dataclass(frozen=True)
class BootstrapMember:
    request: BenchmarkRequest
    result: Any
    arguments: tuple[Any, ...]
    bound: BoundKernel
    envelope: dict[str, Any]
    receipt: dict[str, Any]


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def argument_signature(arguments: Sequence[Any]) -> str:
    leaves, structure = tree_flatten(tuple(arguments))
    owners: dict[int, int] = {}
    records = []
    for value in leaves:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            records.append(
                {
                    "storage": owners.setdefault(storage._cdata, len(owners)),
                    "bytes": storage.nbytes(),
                    **_tensor_abi(value),
                }
            )
        elif (
            value is None
            or type(value) in (bool, int, float, str)
            or isinstance(value, (torch.dtype, torch.device))
        ):
            records.append({"type": str(type(value)), "value": repr(value)})
        else:
            raise TypeError("Unsupported mutable bootstrap argument")
    return _canonical({"structure": repr(structure), "leaves": records})


def _tensors(arguments: Sequence[Any]) -> dict[str, torch.Tensor]:
    return {
        str(i): value
        for i, value in enumerate(tree_flatten(tuple(arguments))[0])
        if isinstance(value, torch.Tensor)
    }


class ReadonlyArguments:
    """Retain exact tensor owners, ABI and every original physical byte."""

    def __init__(self, arguments: Sequence[Any]) -> None:
        self.signature = argument_signature(arguments)
        self.tensors = tuple(_tensors(arguments).values())
        self.owners = tuple(x.untyped_storage()._cdata for x in self.tensors)
        self.storages = _private_storages(_tensors(arguments), frozenset())

    def validate(self, arguments: Sequence[Any]) -> None:
        tensors = tuple(_tensors(arguments).values())
        if (
            argument_signature(arguments) != self.signature
            or len(tensors) != len(self.tensors)
            or any(x is not y for x, y in zip(tensors, self.tensors, strict=False))
            or tuple(x.untyped_storage()._cdata for x in tensors) != self.owners
        ):
            raise BenchmarkObjectiveError("Readonly argument owner or ABI changed")
        try:
            for storage in self.storages:
                storage.validate()
        except AssertionError as error:
            raise BenchmarkObjectiveError("Readonly argument bytes changed") from error


def _expected_failure(bound: BoundKernel, error: BaseException) -> bool:
    if not isinstance(error, Exception) or match_unrecoverable_runtime_error(error):
        return False
    if isinstance(
        error,
        (
            exc.InvalidConfig,
            exc.BackendUnsupported,
            OutOfResources,
            torch.cuda.OutOfMemoryError,
        ),
    ):
        return True
    # Reuse the owning backend's existing native exception policy. Never use
    # ignore_errors or turn ordinary Python failures into candidate declines.
    return bound.config_spec.backend.classify_autotune_exception(error) == "debug"


def _not_cancelled() -> None:
    pass


def bootstrap_partner(
    bound: BoundKernel,
    arguments: Sequence[Any],
    *,
    validate_result: Callable[[tuple[Any, ...], Any], None],
    synchronize: Callable[[], None],
    output: Path,
    writable_arg_indices: tuple[int, ...] = (),
    check_cancelled: Callable[[], None] = _not_cancelled,
) -> BootstrapMember:
    """Compile, execute and validate the first legal default/original seed.

    ``validate_result`` enforces the caller's original role callback, output
    ABI/ownership and any role oracle. Whole-tuple validation is additionally
    required before full search. Writable arguments must be contiguous tensor
    views; all other physical bytes (including guards) remain protected.
    This helper is single-rank; it does not implement distributed retry votes.
    """
    if bound.env.process_group_name is not None:
        raise BenchmarkObjectiveError("Partner bootstrap requires a single rank")
    output.mkdir(parents=True, exist_ok=False)
    arguments = tuple(arguments)
    signature = argument_signature(arguments)
    original = ReadonlyArguments(arguments)
    if len(set(writable_arg_indices)) != len(writable_arg_indices) or any(
        type(i) is not int
        or not 0 <= i < len(arguments)
        or not isinstance(arguments[i], torch.Tensor)
        for i in writable_arg_indices
    ):
        raise BenchmarkObjectiveError("Invalid bootstrap writable argument indices")

    def unchanged() -> None:
        original.validate(arguments)

    def candidates() -> Iterator[tuple[str, Config]]:
        nonlocal phase
        phase = "candidate_generation"
        # Legal defaults do not even ask for the heuristic seed list.
        yield "default", bound.config_spec.default_config()
        phase = "cancellation"
        check_cancelled()
        phase = "candidate_generation"
        messages: list[str] = []
        pairs = bound.config_spec.create_config_generation().seed_flat_config_pairs(
            messages.append
        )
        _write(
            output / "SEED_ORDER.json",
            {"configs": [c.config for _, c in pairs], "transfer_logs": messages},
        )
        for _flat, config in pairs:
            yield "compiler_seed", config

    attempts = []
    phase = "candidate_generation"
    try:
        for origin, config in candidates():
            check_cancelled()
            unchanged()
            attempt = output / f"attempt-{len(attempts):04d}"
            attempt.mkdir()
            row: dict[str, Any] = {
                "origin": origin,
                "config": config.config,
                "status": "started",
            }
            _write(attempt / "START.json", row)
            private = tuple(_clone_args(arguments, None))
            if argument_signature(private) != signature:
                raise BenchmarkObjectiveError("Bootstrap private argument ABI changed")
            source_ids = {
                x.untyped_storage()._cdata for x in _tensors(arguments).values()
            }
            if source_ids & {
                x.untyped_storage()._cdata for x in _tensors(private).values()
            }:
                raise BenchmarkObjectiveError("Bootstrap arguments were not isolated")
            writable = {id(private[i]) for i in writable_arg_indices}
            tensors = _tensors(private)
            protected = _private_storages(
                tensors, frozenset(n for n, x in tensors.items() if id(x) in writable)
            )
            private_owners = tuple(x.untyped_storage()._cdata for x in tensors.values())

            def private_abi() -> None:
                current_tensors = _tensors(private)  # noqa: B023 - synchronous callback, never retained
                assert argument_signature(private) == signature, "Private ABI changed"  # noqa: B023 - synchronous callback, never retained
                assert tuple(current_tensors) == tuple(tensors), (  # noqa: B023 - synchronous callback, never retained
                    "Private leaves changed"
                )
                assert all(current_tensors[n] is x for n, x in tensors.items()), (  # noqa: B023 - synchronous callback, never retained
                    "Private tensor identity changed"
                )
                assert (
                    tuple(x.untyped_storage()._cdata for x in current_tensors.values())
                    == private_owners  # noqa: B023 - synchronous callback, never retained
                ), "Private storage identity changed"

            try:
                phase = "cancellation"
                check_cancelled()
                phase = "compile"
                compiled = bound.compile_config(config)
                source = Path(compiled.__code__.co_filename)
                digest = bound.config_spec.backend.generated_source_hash(compiled)
                if digest is None or _sha(source) != digest:
                    raise BenchmarkObjectiveError(
                        "Bootstrap compiled source identity changed"
                    )
                (attempt / "executed-source.py").write_bytes(source.read_bytes())
                row["source_hash"] = digest
                phase = "cancellation"
                check_cancelled()
                phase = "execute"
                result = compiled(*private)
                synchronize()
                unchanged()
                phase = "cancellation"
                check_cancelled()
                phase = "validate"
                private_abi()
                for storage in protected:
                    storage.validate()
                validate_result(private, result)
                synchronize()
                for storage in protected:
                    storage.validate()
                private_abi()
                unchanged()
                if _sha(source) != digest:
                    raise BenchmarkObjectiveError("Bootstrap executed source changed")
                phase = "cancellation"
                check_cancelled()
            except BaseException as error:
                expected = (
                    phase == "validate" and isinstance(error, AssertionError)
                ) or (
                    phase in ("compile", "execute") and _expected_failure(bound, error)
                )
                row.update(
                    status="rejected" if expected else "aborted",
                    phase=phase,
                    error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                    error=str(error),
                )
                (attempt / "ERROR.txt").write_text(traceback.format_exc())
                _write(attempt / "RESULT.json", row)
                attempts.append(row)
                unchanged()
                if expected:
                    continue
                raise
            request = BenchmarkRequest.create(config, compiled, digest)
            envelope = bound.config_envelope(config).to_dict()
            devices = sorted({x.device.type for x in tensors.values()})
            row.update(
                status="accepted",
                request_digest=request.digest,
                config_envelope=envelope,
                argument_signature=json.loads(signature),
                execution_device_types=devices,
                native_validation="compiled_callable_executed"
                if devices == ["cuda"]
                else "pending_cpu_standin_only",
            )
            _write(attempt / "RESULT.json", row)
            attempts.append(row)
            _write(
                output / "RESULT.json",
                {"status": "accepted", "attempts": attempts, "timed_or_ranked": False},
            )
            return BootstrapMember(request, result, private, bound, envelope, row)
        _write(
            output / "RESULT.json",
            {
                "status": "exhausted",
                "attempts": attempts,
                "unattempted": (
                    "No configurations outside the original finite default/seed "
                    "sequence were considered"
                ),
                "timed_or_ranked": False,
            },
        )
        raise NoLegalPartner(
            "Default and unchanged normalized compiler seeds exhausted"
        )
    except BaseException as error:
        if not (output / "RESULT.json").exists():
            _write(
                output / "RESULT.json",
                {
                    "status": "aborted",
                    "phase": phase,
                    "attempts": attempts,
                    "error_type": type(error).__qualname__,
                    "error": str(error),
                    "timed_or_ranked": False,
                },
            )
        raise
