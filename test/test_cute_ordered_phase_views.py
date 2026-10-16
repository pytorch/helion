from __future__ import annotations

import ast
import importlib.util
import sys

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _view_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    flat = torch.reshape(x, (-1,))
    chained = flat.view(n)
    temporary = torch.empty_like(x)
    temporary_flat = temporary.flatten()
    output = torch.empty_like(x)
    output_flat = output.reshape(-1)
    for index in hl.tile(n):
        temporary_flat[index] = chained[index] + 1
    hl.barrier()
    for index in hl.tile(n):
        output_flat[index] = temporary_flat[index] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _write_view_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    flat = x.flatten()
    output = torch.empty((n,), device=x.device, dtype=x.dtype)
    for index in hl.tile(n):
        flat[index] = flat[index] + 1
    hl.barrier()
    for index in hl.tile(n):
        output[index] = flat[index] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _dtype_view_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    flat = x.view(torch.int32)
    output = torch.empty_like(flat)
    for index in hl.tile(n):
        output[index] = flat[index]
    hl.barrier()
    for index in hl.tile(n):
        output[index] = output[index] + 1
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _detach_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    flat = x.detach()
    output = torch.empty_like(flat)
    for index in hl.tile(n):
        output[index] = flat[index]
    hl.barrier()
    for index in hl.tile(n):
        output[index] = output[index] + 1
    return output


@pytest.fixture
def cpu_only():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        yield


def _code(kernel, x):
    bound = _cpu_bind(kernel, (x,))
    config = bound.config_spec.default_config()
    config.config["pid_type"] = "flat"
    return bound.to_code(config)


@pytest.mark.parametrize("shape,offset", [((3, 4), 0), ((5, 7), 3), ((1, 65), 5)])
@pytest.mark.usefixtures("cpu_only")
def test_ordered_contiguous_view_captures(shape, offset, tmp_path):
    n = shape[0] * shape[1]
    x = torch.arange(n + offset, dtype=torch.float32)[offset:].reshape(shape)
    original = x.clone()
    code = _code(_view_phases, x)
    tree = ast.parse(code)
    bodies = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "PyCodeCache.load"
    ]
    assert len(bodies) == 2
    for body in bodies:
        host = next(
            node
            for node in ast.parse(body).body
            if isinstance(node, ast.FunctionDef) and not node.decorator_list
        )
        names = {a.arg for a in host.args.kwonlyargs}
        assert {"flat", "chained", "temporary_flat", "output_flat"} <= names
        assert not any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
            for node in host.body
        )
    path = tmp_path / "views_generated.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("views_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calls = []

    def launch(kernel, grid, *args, **kwargs):
        calls.append((kernel.__name__, args))

    # Run the actual generated outer and both generated stage hosts. Device
    # bodies are not executed; this checks capture/alias/offset/launch plumbing.
    output = module._view_phases(x, _launcher=launch)
    assert len(calls) == 2
    assert torch.equal(x, original)
    assert output.shape == x.shape and not torch._C._is_alias_of(output, x)
    tensors = [a for _, args in calls for a in args if isinstance(a, torch.Tensor)]
    input_views = [a for a in tensors if torch._C._is_alias_of(a, x)]
    assert input_views and all(a.storage_offset() == offset for a in input_views)
    assert any(torch._C._is_alias_of(a, output) for a in tensors)


@pytest.mark.usefixtures("cpu_only")
def test_ordered_writable_input_view_is_not_fresh():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    code = _code(_write_view_phases, x)
    assert code.count("PyCodeCache.load(") == 2
    expected = (x.clone().flatten() + 1) * 2
    eager = helion.kernel(
        _write_view_phases.fn, backend="cute", static_shapes=True, ref_mode="eager"
    )
    actual = _cpu_bind(eager, (x,)).run_ref(x)
    assert torch.equal(actual, expected)
    assert torch.equal(x.flatten(), expected / 2)


@pytest.mark.parametrize("kind", ["transpose", "stride", "expand"])
@pytest.mark.usefixtures("cpu_only")
def test_ordered_view_declines_noncontiguous_or_copying(kind):
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    if kind == "transpose":
        x = x.t()
    elif kind == "stride":
        x = x[:, ::2]
    else:
        x = x[:1].expand(4, 6)
    with pytest.raises(exc.BackendUnsupported, match="proved contiguous view"):
        _code(_view_phases, x)


@pytest.mark.parametrize("kernel", [_dtype_view_phases, _detach_phases])
@pytest.mark.usefixtures("cpu_only")
def test_ordered_view_declines_unproved_operations(kernel):
    with pytest.raises(exc.BackendUnsupported, match="proved contiguous view"):
        _code(kernel, torch.arange(12, dtype=torch.float32))


@helion.kernel(backend="cute", static_shapes=True)
def _effect_view_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    changed = x.add_(1)
    flat = changed.flatten()
    output = torch.empty_like(flat)
    for index in hl.tile(n):
        output[index] = flat[index]
    hl.barrier()
    for index in hl.tile(n):
        output[index] = output[index] + 1
    return output


@pytest.mark.usefixtures("cpu_only")
def test_ordered_view_does_not_admit_host_mutation():
    with pytest.raises(exc.BackendUnsupported, match="proved contiguous view"):
        _code(_effect_view_phases, torch.arange(12, dtype=torch.float32))


@pytest.mark.parametrize("shape", [(3, 4), (5, 7), (1, 65)])
@pytest.mark.usefixtures("cpu_only")
def test_ordered_dynamic_contiguous_view(shape):
    kernel = helion.kernel(_view_phases.fn, backend="cute", static_shapes=False)
    code = _code(kernel, torch.ones(shape))
    assert code.count("PyCodeCache.load(") == 2


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("write_input", [False, True])
def test_ordered_contiguous_views_native(write_input):
    original = torch.arange(3 * 65 + 3, device=DEVICE, dtype=torch.float32)
    x = original[3:].reshape(3, 65)
    before = x.clone()
    definition = _write_view_phases.fn if write_input else _view_phases.fn
    kernel = helion.kernel(
        definition,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        config=helion.Config(block_sizes=[32, 32], pid_type="flat"),
    )
    output = kernel(x)
    torch.testing.assert_close(
        output.flatten(), (before.flatten() + 1) * 2, rtol=0, atol=0
    )
    torch.testing.assert_close(x, before + 1 if write_input else before, rtol=0, atol=0)
    assert not torch._C._is_alias_of(output, x)
