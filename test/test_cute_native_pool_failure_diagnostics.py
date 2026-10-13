from __future__ import annotations

import json
import os
import signal
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from helion.runtime.cute import native_pool as pool
from helion.runtime.cute.native_pool_protocol import run_workers
from helion.runtime.cute.native_pool_protocol import write
from helion.runtime.kernel import BoundKernel

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.skipif(sys.platform != "linux", reason="owned POSIX workers")
@pytest.mark.parametrize("mode", ["timeout", "malformed_result", "startup_failure"])
def test_failure_evidence_survives_cleanup_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    bound = object.__new__(BoundKernel)
    bound.kernel = SimpleNamespace(
        settings=SimpleNamespace(
            backend="cute",
            static_shapes=True,
            autotune_cute_native_workers=2,
            autotune_cute_native_required=False,
            autotune_compile_timeout=0.5,
        )
    )
    bound._env = SimpleNamespace(
        process_group_name=None,
        cute_fission_plan=None,
        cute_resolved_wrapper_plans=(),
        device=torch.device("cuda:0"),
    )
    requests = [
        SimpleNamespace(
            spec={
                "cache_key": str(i),
                "source_sha256": f"source{i}",
                "source": f"{i}.py",
            }
        )
        for i in range(4)
    ]
    monkeypatch.setattr(pool, "capture", lambda b, c, f, a: requests[c])
    monkeypatch.setattr(pool, "sdk_identity", lambda: {"metadata": {"version": 1}})
    monkeypatch.setattr(pool.signal, "getsignal", lambda s: signal.SIG_DFL)

    def no_verification(*args: object) -> None:
        raise AssertionError("timeout diagnostics must not authorize cache reuse")

    monkeypatch.setattr(pool, "verify_result", no_verification)
    paths = []

    def failed_workers(commands: object, output: Path, **kwargs: object) -> object:
        paths.append(output)
        if mode != "timeout":
            write(
                output / "REAPS.json",
                {
                    "records": [{"pid": 123, "exitcode": -15}],
                    "all_direct_children_reaped": True,
                },
            )
            if mode == "malformed_result":
                (output / "process-0" / "RESULT.json").write_text("incomplete json")
            raise TimeoutError("original timeout reason")
        scripts = []
        for index in range(2):
            directory = output / f"process-{index}"
            # Real owned processes exercise timeout/drain; only stdlib metadata.
            script = f"""
import json,os,pathlib,signal,time
signal.pthread_sigmask(signal.SIG_UNBLOCK, {{signal.SIGINT,signal.SIGTERM}})
p=pathlib.Path({str(directory)!r})
specs=json.loads((p/'BATCH.json').read_text())['requests']
rows=[dict(cache_key=s['cache_key'],source_sha256=s['source_sha256'],compile_seconds=.01) for s in specs[:{2 if index == 0 else 1}]]
(p/'RESULT.json').write_text(json.dumps(dict(status={"passed" if index == 0 else "running"!r},pid=os.getpid(),requests=rows)))
(p/'ACTIVE.json').write_text(json.dumps(dict(index=1,started=time.monotonic())))
{"time.sleep(5)" if index else ""}
"""
            scripts.append(
                (
                    [sys.executable, "-B", "-c", script],
                    dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                )
            )
        return run_workers(scripts, output, **kwargs)

    monkeypatch.setattr(pool, "run_workers", failed_workers)
    result = pool.warm_native_batch(
        bound, [0, 1, 2, 3], [lambda: None] * 4, (), budget_exceeded=lambda: False
    )
    assert result["status"] == "serial_fallback" and result["reloads"] == 0
    if mode == "timeout":
        # These diagnostic-only records omit guard metadata, so even the clean
        # terminal process cannot authorize recovery after the typed deadline.
        assert result["failure"] == "KeyError: 'forbidden_calls'"
        assert result["request_timeout"]["worker"] == 1
        assert result["request_timeout"]["request"] == 1
    else:
        assert result["failure"].startswith("TimeoutError:")
    assert paths and not paths[0].exists()
    # Retained records are JSON serializable after worker-directory deletion.
    result = json.loads(json.dumps(result))
    evidence = result["failure_diagnostics"]
    assert evidence["supervisor"]["all_direct_children_reaped"]
    assert result["reaps"] == evidence["supervisor"]["records"]
    assert len(evidence["processes"]) == 2
    assert all(
        not p["parent_verified"] and not p["publication_authorized"]
        for p in evidence["processes"]
    )
    if mode == "timeout":
        assert [p["reported_completed_count"] for p in evidence["processes"]] == [2, 1]
        active = evidence["processes"][1]["active"]
        assert (
            active["worker_request_index"] == 1 and active["batch_request_index"] == 3
        )
        assert (active["cache_key"], active["source_sha256"], active["source"]) == (
            "3",
            "source3",
            "3.py",
        )
        assert evidence["processes"][1]["reported_completed_requests"] == [
            {"cache_key": "1", "source_sha256": "source1", "compile_seconds": 0.01}
        ]
        for row in result["reaps"]:
            with pytest.raises(ChildProcessError):
                os.waitpid(row["pid"], os.WNOHANG)
    elif mode == "malformed_result":
        assert evidence["read_errors"] == ["RESULT.json: JSONDecodeError"]
    else:
        assert all("active" not in p for p in evidence["processes"])
