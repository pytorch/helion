from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


def _fresh_main(mutate: bool) -> None:
    import torch

    from .test_cute_chunk_recurrence import _code
    from helion import exc
    from helion._compiler import backend_registry
    from helion._compiler.aten_lowering import iota_lowering
    from helion._compiler.cute import chunk_recurrence
    from helion._compiler.cute.chained_matmul import _UnsupportedChain
    from helion._compiler.cute.prepared_tcgen_binding import PreparedProjectionHost

    assert not torch.cuda.is_initialized()
    assert "cute" not in backend_registry._REPAIRED_CODEGEN_NAMES
    planner = chunk_recurrence._plan_chunk_recurrence
    hosts = []

    def retain(*args, **kwargs):
        plan = planner(*args, prepared_edge=True, **kwargs)
        assert plan is not None
        assert "cute" in backend_registry._REPAIRED_CODEGEN_NAMES
        host = PreparedProjectionHost(plan)
        hosts.append(host)
        if mutate:
            iota_lowering.codegen_impls["cute"] = lambda _context, _node: None
        return plan

    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch.object(chunk_recurrence, "_plan_chunk_recurrence", retain),
    ):
        if mutate:
            with pytest.raises(exc.InternalError) as caught:
                _code(fp32_state=True, dv_partitions=2, pipeline="wide")
            assert type(caught.value.__cause__) is _UnsupportedChain
            assert str(caught.value.__cause__) == "matched projection graph changed"
            with pytest.raises(
                _UnsupportedChain, match="matched projection graph changed"
            ):
                hosts[0].check()
        else:
            source = _code(fp32_state=True, dv_partitions=2, pipeline="wide")
            assert hashlib.sha256(source.encode()).hexdigest() == (
                "c95ceb30c1e17c6522364b2fa42def36081dc2f0ffbda95e138fa0197511f2f5"
            )
            hosts[0].check()
    assert len(hosts) == 1 and not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutate", [False, True])
def test_selected_original_host_first_use_and_stale_handler(mutate):
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys, types; "
        f"sys.path.insert(0, {str(root)!r}); "
        "benchmarks = types.ModuleType('benchmarks'); "
        f"benchmarks.__path__ = [{str(root / 'benchmarks')!r}]; "
        "sys.modules['benchmarks'] = benchmarks; "
        "from test.test_cute_prepared_tcgen_edge_initialization import _fresh_main; "
        f"_fresh_main({mutate!r})"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
