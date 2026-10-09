"""Pretuned top-k and routing recipes, including selected-logit softmax."""

from __future__ import annotations

import ast
import importlib
from typing import TYPE_CHECKING
from unittest.mock import patch

from pretuned_kernels import run as pretuned_run
from pretuned_kernels.moe_softmax_routing import moe_softmax_routing as pretuned_moe
from pretuned_kernels.topk import topk as pretuned_topk
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.aot_structural_policy import model_configs

if TYPE_CHECKING:
    from collections.abc import Iterator


topk_heuristic = importlib.import_module(
    "pretuned_kernels.topk._helion_aot_topk_cuda_sm103"
    f"__policy_{pretuned_topk.STRUCTURAL_POLICY.identity()}"
)


_PRETUNED_MOE_CASES = [("moe_softmax_routing", 0)]


@pytest.fixture
def cpu_codegen() -> Iterator[None]:
    pytest.importorskip("cutlass.cute")
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        yield


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.timeout(300)
@pytest.mark.parametrize("name,index", _PRETUNED_MOE_CASES)
def test_pretuned_moe_routing_aot_correctness(
    name: str, index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("MoE routing AOT configs are pretuned for GB300")
    from helion.autotuner.base_cache import AutotuneCacheBase

    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    assert name in pretuned_run.KERNELS
    assert pretuned_run._supported_hardware(name) == {"sm103"}
    module = pretuned_run._import_kernel_module(name)
    with patch.object(
        AutotuneCacheBase,
        "_run_autotune_trials",
        side_effect=AssertionError("pretuned recipes must not launch autotuning"),
    ):
        module.check_case(module.SHAPES[index])


def test_pretuned_topk_registration_and_keys() -> None:
    assert "topk" in pretuned_run.KERNELS
    assert pretuned_run._supported_hardware("topk") == {"sm103"}
    keys = set()
    for rows, width, k, softmax in pretuned_topk.SHAPES:
        x = torch.empty(
            (rows, width), dtype=torch.bfloat16, device=torch.device("meta")
        )
        key = topk_heuristic.key_topk(x, k, softmax)
        assert key not in keys
        keys.add(key)
        if not softmax:
            assert topk_heuristic.key_topk(x, k) == key
    assert len(keys) == 30
    assert topk_heuristic.STRUCTURAL_POLICY == pretuned_topk.STRUCTURAL_POLICY
    assert (
        list(
            model_configs(topk_heuristic, "topk", pretuned_topk.STRUCTURAL_POLICY) or ()
        )
        == topk_heuristic.CONFIGS
    )
    assert all(
        config.policy == pretuned_topk.STRUCTURAL_POLICY
        for config in topk_heuristic.CONFIGS
    )


@pytest.mark.parametrize("unsupported", ["rows", "k", "dtype", "stride"])
def test_pretuned_topk_rejects_untuned_inputs(unsupported: str) -> None:
    rows = 17 if unsupported == "rows" else 65536
    dtype = torch.float32 if unsupported == "dtype" else torch.bfloat16
    inner_stride = 2 if unsupported == "stride" else 1
    x = torch.empty_strided(
        (rows, 64),
        (64 * inner_stride, inner_stride),
        dtype=dtype,
        device=torch.device("meta"),
    )
    k = 3 if unsupported == "k" else 8
    with pytest.raises(ValueError):
        topk_heuristic.key_topk(x, k)
    with pytest.raises(ValueError):
        topk_heuristic.autotune_topk(x, k)


@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "rows,width,k,softmax", [shape for shape in pretuned_topk.SHAPES if shape[1] == 64]
)
def test_pretuned_topk_codegen_is_one_fused_launch(
    rows: int, width: int, k: int, softmax: bool
) -> None:
    with FakeTensorMode():
        x = torch.empty((rows, width), dtype=torch.bfloat16)
    # Bind the source without loading hardware-dependent AOT caches on the CPU.
    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=pretuned_topk.STRUCTURAL_POLICY,
    )(pretuned_topk.topk.fn)
    bound = kernel._bind_isolated((x, k, softmax))
    config = topk_heuristic.autotune_topk(x, k, softmax)
    assert len(bound.config_spec.reduction_loops.valid_block_ids()) == int(softmax)
    reduction_loops = config.config.get("reduction_loops", [])
    assert isinstance(reduction_loops, list) and len(reduction_loops) == int(softmax)
    code = bound.to_code(config)
    if not softmax:
        stale = helion.Config.from_dict({**config.config, "reduction_loops": [None]})
        with pytest.raises(exc.InvalidConfig, match="Too many values.*reduction_loops"):
            bound.to_code(stale)
    assert code.count("@cute.kernel") == 1
    assert "sort_rank" not in code
    assert ("row_known_maximum" in code) == softmax
    launches = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 1


@pytest.mark.usefixtures("cpu_codegen")
def test_pretuned_moe_codegen_is_one_fused_launch() -> None:
    heuristic = importlib.import_module(
        "pretuned_kernels.moe_softmax_routing."
        "_helion_aot_moe_softmax_routing_cuda_sm103"
        f"__policy_{pretuned_moe.STRUCTURAL_POLICY.identity()}"
    )
    with FakeTensorMode():
        signature = pretuned_moe.make_inputs(pretuned_moe.SHAPES[0], device="cpu")
        # Validate the shipped signature/config, then compile a small instance
        # of the same program so this unit test stays inexpensive.
        config = heuristic.autotune_moe_softmax_routing(*signature)
        args = (
            torch.empty((8, 64), dtype=torch.float32),
            torch.empty((64,), dtype=torch.float32),
            *signature[2:],
        )
    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=pretuned_moe.STRUCTURAL_POLICY,
    )(pretuned_moe.moe_softmax_routing.fn)
    bound = kernel._bind_isolated(args)
    code = bound.to_code(config)
    assert code.count("@cute.kernel") == 1
    assert "sort_rank" not in code
    launches = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width,k", [(64, 8), (256, 16), (1024, 32)])
@pytest.mark.parametrize("softmax", [False, True])
def test_pretuned_topk_aot_correctness(
    width: int, k: int, softmax: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("top-k AOT configs are pretuned for GB300 (sm103)")
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    generator = torch.Generator(device=DEVICE).manual_seed(20260929)
    x = torch.randn(
        (65536, width), dtype=torch.bfloat16, device=DEVICE, generator=generator
    )
    x[0].zero_()
    x[1] = (torch.arange(width, device=DEVICE) % 5 - 2).to(x.dtype)
    original = x.clone()
    values, indices = pretuned_topk.topk(x, k, softmax)
    assert values.shape == indices.shape == (65536, k)
    assert values.dtype == torch.bfloat16 and indices.dtype == torch.int32
    assert bool(((indices >= 0) & (indices < width)).all())
    ordered_indices = indices.sort(dim=-1).values
    assert bool((ordered_indices[:, 1:] != ordered_indices[:, :-1]).all())
    selected = original.gather(1, indices.long())
    expected = torch.topk(original, k, dim=-1).values
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    if softmax:
        expected = torch.softmax(expected.float(), dim=-1)
        torch.testing.assert_close(values.float(), expected, rtol=0.004, atol=1e-7)
        torch.testing.assert_close(
            values.float().sum(-1),
            torch.ones(65536, device=DEVICE),
            rtol=0,
            atol=0.004,
        )
    else:
        assert torch.equal(values.view(torch.int16), selected.view(torch.int16))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))
