from __future__ import annotations

import json
import os
from pathlib import Path

from examples.aot_example import row_softmax
from examples.welford import welford
import pytest
import torch
import torch.nn.functional as F

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "kind,shape,dtype,static,tile,threads,vector,schedule,nonfinite",
    [
        pytest.param(
            "welford",
            (1024, 512),
            torch.float32,
            True,
            (4, 512),
            32,
            4,
            "resident",
            False,
            id="welford-original0",
        ),
        pytest.param(
            "welford",
            (16384, 1024),
            torch.float32,
            False,
            (4, 1024),
            64,
            4,
            "resident",
            False,
            id="welford-original1",
        ),
        pytest.param(
            "welford",
            (65536, 2048),
            torch.float32,
            True,
            (4, 2048),
            32,
            4,
            "resident",
            False,
            id="welford-original2",
        ),
        pytest.param(
            "welford",
            (65536, 2048),
            torch.float32,
            True,
            (4, 2048),
            32,
            4,
            "reload",
            False,
            id="welford-original2-reload",
        ),
        pytest.param(
            "welford",
            (17, 77),
            torch.float32,
            False,
            (4, 128),
            32,
            1,
            "resident",
            False,
            id="welford-dynamic-tail",
        ),
        pytest.param(
            "welford",
            (129, 2057),
            torch.float32,
            True,
            (1, 512),
            128,
            1,
            "resident",
            False,
            id="welford-multiple-tiles",
        ),
        pytest.param(
            "welford",
            (17, 80),
            torch.bfloat16,
            True,
            (4, 512),
            32,
            8,
            "resident",
            False,
            id="welford-bf16-tail",
        ),
        pytest.param(
            "welford",
            (17, 256),
            torch.float16,
            False,
            (1, 512),
            64,
            4,
            "resident",
            False,
            id="welford-fp16-tail",
        ),
        pytest.param(
            "softmax",
            (256, 256),
            torch.bfloat16,
            False,
            (1, 256),
            32,
            4,
            "resident",
            False,
            id="softmax-original0",
        ),
        pytest.param(
            "softmax",
            (1024, 1024),
            torch.bfloat16,
            False,
            (1, 1024),
            128,
            4,
            "resident",
            False,
            id="softmax-original1",
        ),
        pytest.param(
            "softmax",
            (4096, 4096),
            torch.bfloat16,
            False,
            (1, 4096),
            128,
            8,
            "resident",
            False,
            id="softmax-original-large",
        ),
        pytest.param(
            "softmax",
            (128, 4096),
            torch.float16,
            True,
            (1, 4096),
            128,
            8,
            "reload",
            False,
            id="softmax-fp16-reload",
        ),
        pytest.param(
            "softmax",
            (17, 77),
            torch.float32,
            False,
            (4, 128),
            32,
            1,
            "resident",
            False,
            id="softmax-dynamic-tail",
        ),
        pytest.param(
            "softmax",
            (17, 513),
            torch.bfloat16,
            False,
            (1, 256),
            64,
            1,
            "resident",
            False,
            id="softmax-multiple-tiles",
        ),
        pytest.param(
            "welford",
            (17, 77),
            torch.float32,
            False,
            (4, 128),
            32,
            1,
            "resident",
            True,
            id="welford-nonfinite",
        ),
        pytest.param(
            "softmax",
            (17, 77),
            torch.float32,
            False,
            (4, 128),
            32,
            1,
            "resident",
            True,
            id="softmax-nonfinite",
        ),
    ],
)
def test_resident_sequence_original_examples_and_mutated_graphs(
    kind: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    static: bool,
    tile: tuple[int, int],
    threads: int,
    vector: int,
    schedule: str,
    nonfinite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These original cases exercise the dependent max/sum phase cache. The
    # active online rewrite has separate sequence and bounded-cache coverage.
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")
    kernel = helion.kernel(
        welford.fn if kind == "welford" else row_softmax.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
    )
    rows, columns = shape
    args = (torch.empty(shape, dtype=dtype, device=CUDA_DEVICE),)
    if kind == "welford":
        args = (
            torch.empty(columns, dtype=dtype, device=CUDA_DEVICE),
            torch.empty(columns, dtype=dtype, device=CUDA_DEVICE),
            *args,
        )
    count = 2 if kind == "welford" else 1
    config = helion.Config(
        block_sizes=[tile[0], *([tile[1]] * count)],
        num_threads=[tile[0], *([threads] * count)],
        cute_vector_widths=[1, *([vector] * count)],
        cute_lane_layouts=["blocked", *(["strided"] * count)],
        cute_reduction_sequence=schedule,
        cute_cluster_n=1,
    )
    bound = kernel.bind(args)
    code = bound.to_code(config)
    assert "sequence_acc" in code
    assert "_helion_sequence_reduce" not in code
    if schedule == "resident":
        assert "sequence_values" in code
    if artifact_root := os.environ.get("HELION_SEQUENCE_ARTIFACT_DIR"):
        artifact = Path(artifact_root)
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "generated.py").write_text(code)
        (artifact / "config.json").write_text(
            json.dumps(dict(bound.config_spec.normalized_config(config)), indent=2)
            + "\n"
        )
    compiled = bound.compile_config(config)

    def mutate(values: tuple[torch.Tensor, ...], seed: int) -> None:
        torch.manual_seed(seed)
        for value in values:
            value.uniform_(-0.5, 0.5)
        if seed == 1802:
            values[-1].zero_()
        if nonfinite:
            values[-1][0, 0] = float("nan")
            values[-1][1, 0] = float("inf")

    def check(actual: torch.Tensor, values: tuple[torch.Tensor, ...]) -> None:
        if kind == "welford":
            weight, bias, x = values
            expected = F.layer_norm(
                x.float(), (columns,), weight.float(), bias.float(), eps=1e-5
            ).to(dtype)
        else:
            expected = torch.softmax(values[0].float(), dim=-1).to(dtype)
        # Stricter than the study's default 1e-2 relative/absolute floor;
        # do not relax this if any configuration fails correctness.
        atol = 2e-5 if dtype == torch.float32 else 1e-3
        torch.testing.assert_close(
            actual, expected, rtol=1e-2, atol=atol, equal_nan=True
        )
        if kind == "softmax":
            finite_rows = torch.isfinite(expected).all(dim=-1)
            torch.testing.assert_close(
                actual[finite_rows].float().sum(-1),
                torch.ones_like(actual[finite_rows, 0], dtype=torch.float32),
                rtol=2e-3,
                atol=2e-3,
            )

    for seed in (1801, 1802, 1803):
        mutate(args, seed)
        actual = compiled(*args)
        torch.cuda.synchronize()
        check(actual, args)
    replacements = tuple(torch.empty_like(value) for value in args)
    mutate(replacements, 1804)
    actual = compiled(*replacements)
    torch.cuda.synchronize()
    check(actual, replacements)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    for seed in (1805, 1806, 1807):
        mutate(args, seed)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(captured, args)
