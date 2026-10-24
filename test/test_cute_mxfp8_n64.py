"""N64 native MXFP8 tiles, including adjacent 64-row scale windows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test.test_cute_mxfp8_mma import _config
from test.test_cute_mxfp8_mma import _cpu_inputs
from test.test_cute_mxfp8_mma import _cpu_target
from test.test_cute_mxfp8_mma import _mxfp8
from test.test_cute_mxfp8_mma import _wrapper_plan

from helion._compiler.cute.block_scaled_config import BLOCK_SCALED_CHOICES
from helion._compiler.cute.block_scaled_config import block_scaled_choices
from helion._compiler.cute.block_scaled_config import normalize_block_scaled_config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.runtime import cute_cuda_graph

pytestmark = skipUnlessBackends(["cute"])


def test_cpu_n64_is_in_both_config_vocabularies() -> None:
    assert block_scaled_choices() == BLOCK_SCALED_CHOICES
    assert block_scaled_choices(mxfp8=True)["cute_scaled_tile_n"] == (
        256,
        224,
        128,
        64,
        32,
        16,
        8,
    )
    raw_config: dict[str, object] = {"cute_scaled_mma": True, "cute_scaled_tile_n": 64}
    normalize_block_scaled_config(
        dict(raw_config), available=True, capability=(10, 3), fix_invalid=False
    )
    normalize_block_scaled_config(
        raw_config, available=True, capability=(10, 3), fix_invalid=False, mxfp8=True
    )
    assert raw_config["cute_scaled_tile_n"] == 64
    with _cpu_target():
        bound = _mxfp8._bind_isolated(_cpu_inputs((137, 273, 288)))
        config = _config(bound, cute_scaled_tile_n=64)
        source = bound.to_code(config)
        assert _wrapper_plan(source)["bn"] == 64
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            assert (
                generation.unflatten(generation.flatten(config))["cute_scaled_tile_n"]
                == 64
            )


@pytest.mark.parametrize("cluster,bk", [(1, 128), (1, 256), (2, 128), (2, 256)])
def test_cpu_n64_native_pipeline_compiles(
    cluster: int, bk: int, tmp_path: Path
) -> None:
    pytest.importorskip("cutlass")
    initialized = torch.cuda.is_initialized()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_ARCH="sm_100a",
        CUTE_DSL_CACHE_DIR=str(tmp_path / "cutlass-cache"),
        HELION_BACKEND="cute",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    # CUTLASS caches the selected architecture in its process-wide DSL object.
    # A subprocess keeps fake SM100 compilation away from later CUDA tests.
    command = (
        "from test.test_cute_mxfp8_n64 import _compile_n64_pipeline_gpu_free; "
        f"_compile_n64_pipeline_gpu_free({cluster}, {bk})"
    )
    child = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert torch.cuda.is_initialized() is initialized
    assert not list(tmp_path.rglob("*.mlir"))
    assert not list(tmp_path.rglob("*.json"))


def _compile_n64_pipeline_gpu_free(cluster: int, bk: int) -> None:
    assert not torch.cuda.is_initialized()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["CUTE_DSL_ARCH"] == "sm_100a"
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    driver = pytest.importorskip("cuda.bindings.driver")
    from helion._compiler.cute.block_scaled_runtime import Sm100BlockScaledGemm

    m, n, k = 137, 273, 288
    kernel = Sm100BlockScaledGemm(
        m, n, k, 64, bk, 2, cluster, True, 148, operand_format="mxfp8"
    )
    assert kernel.sf_n_chunks == 1
    specifications = (
        (cutlass.Float8E4M3FN, (m, k), (k, 1)),
        (cutlass.Float8E4M3FN, (n, k), (k, 1)),
        (cutlass.Uint8, (kernel.workspace_bytes,), (1,)),
        (cutlass.BFloat16, (m, n), (n, 1)),
    )
    tensors = [
        cute.runtime.make_fake_tensor(dtype, shape, stride, assumed_align=16)
        for dtype, shape, stride in specifications
    ]
    cute.compile(
        kernel.mxfp8,
        *tensors,
        cutlass.Float32(1),
        driver.CUstream(0),
        no_jit_engine=True,
        no_cache=True,
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "m,n,k,cluster,bk,persistent",
    [
        (1, 129, 160, 1, 128, True),
        (137, 273, 288, 1, 128, True),
        (73, 273, 288, 1, 256, False),
        (137, 193, 288, 2, 256, True),
        (128, 2048, 4096, 1, 128, True),
        (128, 2048, 4096, 1, 256, True),
        (128, 2048, 4096, 2, 256, True),
    ],
)
def test_cuda_n64_observes_both_scale_halves(
    m: int, n: int, k: int, cluster: int, bk: int, persistent: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    torch.manual_seed(0)
    a = (torch.randn((m, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    groups = torch.arange(k // 32, device=DEVICE)
    sa = (124 + (torch.arange(m, device=DEVICE)[:, None] + groups) % 5).to(torch.uint8)
    sb = (125 + (torch.arange(n, device=DEVICE)[:, None] // 64 + groups * 3) % 4).to(
        torch.uint8
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
    values = (a, b, sa, sb, out)
    bound = _mxfp8._bind_isolated(values)
    config = _config(
        bound,
        cute_scaled_tile_n=64,
        cute_scaled_tile_k=bk,
        cute_scaled_cluster_m=cluster,
        cute_scaled_stages=2,
        cute_scaled_persistent=persistent,
    )
    run = bound.compile_config(config)
    expected = (
        (a.double() * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1))
        @ (b.double() * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1)).T
    ).bfloat16()
    actual = run(*values)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.016)
    assert _wrapper_plan(bound.to_code(config))["bn"] == 64


def test_cuda_n64_graph_replay_observes_typed_live_scales() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native MXFP8 required")
    m, n, k = 137, 273, 288
    torch.manual_seed(1)
    a = (torch.randn((m, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device=DEVICE) / 3).to(torch.float8_e4m3fn)
    sa = torch.randint(125, 129, (m, k // 32), dtype=torch.uint8, device=DEVICE)
    sb = torch.randint(125, 129, (n, k // 32), dtype=torch.uint8, device=DEVICE)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
    values = (a, b, sa.view(torch.float8_e8m0fnu), sb.view(torch.float8_e8m0fnu), out)
    bound = _mxfp8._bind_isolated(values)
    run = bound.compile_config(_config(bound, cute_scaled_tile_n=64))
    run(*values)
    with cute_cuda_graph() as graph:
        captured = run(*values)
    for scales in (sa, sb):
        original = scales.clone()
        scales.add_(1)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            (a.double() * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1))
            @ (
                b.double() * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1)
            ).T
        ).bfloat16()
        torch.testing.assert_close(captured, expected, atol=0.02, rtol=0.016)
        scales.copy_(original)
