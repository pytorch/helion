from __future__ import annotations

import ast
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

from test._cute_binding import _require_cute_codegen
from test.test_cute_materialized_fission import _HostModules
from test.test_cute_materialized_fission import _sources

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
import helion.language as hl

triton = pytest.importorskip("triton")


def _multi_root_reduction(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for first in hl.tile(x.size(0)):
        out[first] = x[first] + x[first].sum()
    hl.barrier()
    for second in hl.tile(x.size(0)):
        out[second] = out[second] + out[second].sum()
    return out


def _single_root_reduction(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + x[tile].sum()
    return out


def _gather_after_partial_reduction(x: torch.Tensor) -> torch.Tensor:
    rows = x.numel() // 13
    block_rows = hl.register_block_size(rows)
    block_cols = hl.register_block_size(13)
    partial = torch.empty((rows, (13 + block_cols - 1) // block_cols), device=x.device)
    out = torch.empty((rows,), device=x.device)
    for row, col in hl.tile([rows, 13], block_size=[block_rows, block_cols]):
        values = x[row.index[:, None] * 13 + col.index[None, :]]
        partial[row, col.id] = values.sum(-1)
    hl.barrier()
    for row in hl.tile(rows):
        columns = hl.arange(block_cols)
        values = hl.load(
            x,
            [row.index[:, None] * 13 + columns[None, :]],
            extra_mask=columns[None, :] < 13,
        )
        out[row] = partial[row, :].sum(-1) + values.sum(-1)
    return out


def _host_result(code: str, name: str, x: torch.Tensor) -> torch.Tensor:
    """Execute emitted host wrappers, stopping at their kernel launches."""
    launcher = Mock()
    if _sources(code):
        modules = _HostModules()
        namespace = vars(modules.load(code))
        assert len(modules.markers) == 2
        expected_launches = 2
    else:
        function = next(
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        namespace = {
            "torch": torch,
            "triton": triton,
            "helion": helion,
            "_default_launcher": launcher,
            "_default_cute_launcher": launcher,
            f"_helion_{name}": object(),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "<host>", "exec"),
            namespace,
        )
        expected_launches = 1
    with patch("helion.runtime.get_num_sm", return_value=148):
        result = namespace[name](x, _launcher=launcher)
    assert launcher.call_count == expected_launches
    return result


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.parametrize(
    ("backend", "kind"),
    [
        ("triton", "single"),
        ("triton", "multi"),
        ("triton", "gather"),
        ("cute", "single"),
        ("cute", "multi"),
    ],
)
@pytest.mark.parametrize("blocks", [[16, 32], [32, 16], [1, 16]])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_later_root_reduction_host_dependencies(
    backend: str, blocks: list[int], static_shapes: bool, kind: str
) -> None:
    if backend == "cute":
        _require_cute_codegen()
    function = {
        "single": _single_root_reduction,
        "multi": _multi_root_reduction,
        "gather": _gather_after_partial_reduction,
    }[kind]
    kernel = helion.kernel(
        backend=backend, static_shapes=static_shapes, autotune_effort="none"
    )(function)
    config = helion.Config(
        block_sizes=(
            [blocks[0], 4, blocks[1]]
            if kind == "gather"
            else blocks[:1]
            if kind == "single"
            else blocks
        ),
        indexing="pointer",
        # Different reduction widths need separate CuTe phase launches.
        pid_type="flat"
        if backend == "cute" and kind == "multi"
        else "persistent_blocked",
    )
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.current_device", return_value=0))
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        stack.enter_context(
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("unexpected CUDA")
            )
        )
        stack.enter_context(
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(
                    major=10,
                    minor=3,
                    multi_processor_count=148,
                    shared_memory_per_block=49152,
                    shared_memory_per_block_optin=232448,
                ),
            )
        )
        for target in (
            "helion.runtime.kernel.target_device_capability",
            "helion._compiler.compile_environment.target_device_capability",
        ):
            stack.enter_context(patch(target, return_value=(10, 3)))
        stack.enter_context(patch("helion._compat._is_hip", return_value=False))
        stack.enter_context(patch("helion.runtime.get_num_sm", return_value=148))
        stack.enter_context(
            patch("helion.autotuner.config_spec.num_compute_units", return_value=148)
        )
        mode = FakeTensorMode()
        x = FakeTensor(
            mode, torch.empty(65, device=torch.device("meta")), torch.device("cuda:0")
        )
        code = kernel._bind_isolated((x,)).to_code(config)
        result = _host_result(code, function.__name__, torch.empty(65))
        assert result.shape == ((5,) if kind == "gather" else (65,))
        assert result.device.type == "cpu"


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_multi_root_reduction_native(backend: str) -> None:
    if backend == "cute":
        _require_cute_codegen()
    kernel = helion.kernel(backend=backend, static_shapes=True, autotune_effort="none")(
        _multi_root_reduction
    )
    x = torch.arange(65, dtype=torch.float32, device=DEVICE)
    bound = kernel.bind((x,))
    for blocks in ([16, 32], [32, 16], [1, 16]):
        expected = x.clone()
        for block in blocks:
            for begin in range(0, x.numel(), block):
                values = expected[begin : begin + block]
                values.add_(values.sum())
        actual = bound.compile_config(
            helion.Config(
                block_sizes=blocks,
                indexing="pointer",
                pid_type="flat" if backend == "cute" else "persistent_blocked",
            )
        )(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_gather_after_partial_reduction_native() -> None:
    kernel = helion.kernel(
        backend="triton", static_shapes=True, autotune_effort="none"
    )(_gather_after_partial_reduction)
    x = torch.arange(65, dtype=torch.float32, device=DEVICE)
    bound = kernel.bind((x,))
    expected = x.view(5, 13).sum(-1) + x.view(5, 13)[:, :4].sum(-1)
    for blocks in ([16, 4, 32], [32, 4, 16], [1, 4, 16]):
        config = helion.Config(
            block_sizes=blocks, indexing="pointer", pid_type="persistent_blocked"
        )
        actual = bound.compile_config(config)(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
