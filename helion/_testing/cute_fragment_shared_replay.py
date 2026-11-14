"""Shared native lifecycle for ordinary and dedicated two-phase diagnostics.

The exact source, full backing copies, CPU oracle, snapshots and capture lifetime
are common. Each caller supplies a mandatory generated-source admission check.
No timer or production dispatch behavior is changed by this helper.
"""

from __future__ import annotations

import hashlib
from itertools import starmap
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from .cute_conditional_phase_binding import phase_fixture
from .cute_fragment_storage_oracle import assert_attention_result
from .cute_fragment_storage_oracle import attention_cpu
from .cute_fragment_storage_oracle import backing_snapshots
from .cute_fragment_storage_oracle import copy_backings
from .cute_fragment_storage_oracle import move_backings
from .cute_fragment_storage_oracle import raw_equal
from .cute_paged_conditional_semantics import written_conditional_attention

if TYPE_CHECKING:
    from collections.abc import Callable


def require_sm100_native() -> int:
    """Skip this instruction-specific fixture outside the current SM100 device."""
    if not torch.cuda.is_available():
        pytest.skip("conditional native fixture requires CUDA SM100")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("conditional native fixture requires CUDA SM100")
    return device


def begin_record(name: str, default_root: Path) -> tuple[Path, dict[str, Any]]:
    """Record on the current device; HELION_SHARED_STORAGE_RAW overrides raw output.

    The optional diagnostic variable names an output directory. Otherwise pytest's
    temporary directory owns the raw evidence. It never selects a device.
    """
    device = require_sm100_native()
    root = Path(os.environ.get("HELION_SHARED_STORAGE_RAW", default_root))
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=name + "-", dir=root))
    return output, {
        "schema": 1,
        "case": name,
        "device": device,
        "complete": False,
        "observations": [],
        "sources": {},
    }


def checkpoint_record(output: Path, record: dict[str, Any]) -> None:
    temporary = output / "raw.pending"
    torch.save(record, temporary)
    temporary.replace(output / "raw.pt")


def finish_record(output: Path, record: dict[str, Any]) -> None:
    error = sys.exc_info()[1]
    if error is not None:
        record.setdefault("error", repr(error))
    checkpoint_record(output, record)
    logging.getLogger(__name__).info("Shared-storage raw evidence: %s", output)


def snapshot(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().clone() for value in values)


def same_inputs(
    before: list[torch.Tensor], args: tuple[object, ...]
) -> list[torch.Tensor]:
    after = backing_snapshots(args)
    assert len(before) == len(after) and all(
        starmap(raw_equal, zip(before, after, strict=True))
    )
    return after


def run_authored_storage_replay(
    output: Path,
    record: dict[str, Any],
    validate_source: Callable[[str], None],
) -> None:
    """Run the exact authored fixture; callers must prove their emission choice."""
    graph = None
    retained = []
    try:
        original = phase_fixture()
        assert all(isinstance(value, torch.Tensor) for value in original[:9])
        cast("torch.Tensor", original[7]).fill_(float("nan"))
        cast("torch.Tensor", original[8]).fill_(float("nan"))
        changed = move_backings(original, torch.device("cpu"))
        tensors = tuple(cast("torch.Tensor", value) for value in changed[:9])
        tensors[0].copy_(-tensors[0] + 0.03125)
        tensors[1].neg_()
        tensors[2].add_(1)
        tensors[4].fill_(128)
        tensors[5].add_(1).remainder_(7)
        tensors[6].copy_(torch.tensor([2049, 0, 257], dtype=torch.int32))
        expected = {
            name: snapshot(attention_cpu(move_backings(values, torch.device("cpu"))))
            for name, values in (("original", original), ("changed", changed))
        }
        record["expected"] = expected
        record["reference_inputs"] = {
            "original": backing_snapshots(original[:7]),
            "changed": backing_snapshots(changed[:7]),
        }
        args = move_backings(original, torch.device("cuda"))
        bound = written_conditional_attention.bind(args)
        assert bound.host_function is not None
        assert len(bound.host_function.device_ir.graphs) == 16
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
        validate_source(source)
        (output / "generated.py").write_text(source)
        record["sources"]["generated"] = hashlib.sha256(source.encode()).hexdigest()
        compiled = bound.compile_config(config)
        cached_path = bound.get_cached_path(config)
        assert cached_path is not None and Path(cached_path).read_text() == source
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            before = backing_snapshots(args[:7])
            eager = compiled(*args)
            assert eager[0] is args[7] and eager[1] is args[8]
            retained.append(eager)
            stream.synchronize()
            actual = snapshot(eager)
            record["observations"].append(
                {"phase": "eager", "actual": actual, "inputs": before}
            )
            checkpoint_record(output, record)
            assert_attention_result(actual, expected["original"])
            record["observations"][-1]["inputs_after"] = same_inputs(before, args[:7])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = compiled(*args)
            retained.append(captured)
            original_outputs = None
            for phase, values in (
                ("original", original),
                ("changed", changed),
                ("restored", original),
            ):
                copy_backings(args, values)
                phase_inputs = backing_snapshots(args[:7])
                graph.replay()
                stream.synchronize()
                actual = snapshot(captured)
                record["observations"].append(
                    {"phase": phase, "actual": actual, "inputs": phase_inputs}
                )
                checkpoint_record(output, record)
                assert_attention_result(
                    actual, expected["changed" if phase == "changed" else "original"]
                )
                record["observations"][-1]["inputs_after"] = same_inputs(
                    phase_inputs, args[:7]
                )
                if phase == "original":
                    original_outputs = actual
                    assert all(
                        starmap(
                            raw_equal,
                            zip(
                                actual, record["observations"][0]["actual"], strict=True
                            ),
                        )
                    )
                elif phase == "restored":
                    assert original_outputs is not None
                    assert all(
                        starmap(raw_equal, zip(actual, original_outputs, strict=True))
                    )
            stream.synchronize()
            graph.reset()
            graph = None
        record["complete"] = True
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            record["error"] = repr(error)
        try:
            if graph is not None:
                torch.cuda.synchronize()
                graph.reset()
        except BaseException as error:
            record["cleanup_error"] = repr(error)
            raise
        finally:
            finish_record(output, record)
