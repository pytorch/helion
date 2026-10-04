from __future__ import annotations

import hashlib
import inspect
import json
import linecache
import sys
import types
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_snapshot_runtime import _config
from .test_cute_chained_snapshot_runtime import _inputs
from .test_cute_chained_snapshot_runtime import _sequence


def _module(name: str, source: str, **bindings: Any) -> types.ModuleType:
    path = f"<{name}>"
    module = types.ModuleType(name)
    module.__file__ = path
    module.__dict__.update(bindings)
    sys.modules[name] = module
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def _compile_original_host(source: str, values: tuple[Any, ...], directory):
    """Run the original CPU host to capture its exact launch, then compile only.

    Real cute.compile is essential here: direct scf construction bypasses the
    DSL's static/dynamic loop-container validation. No launch or oracle change.
    """
    from cuda.bindings.driver import CUstream
    import cutlass
    import cutlass.cute as cute

    from helion.runtime.cute.launcher import _create_cute_wrapper
    from helion.runtime.cute.launcher import _cute_bake_tensor_shapes_guard
    from helion.runtime.cute.launcher import _cute_tensor_pointer_alignment

    name = "_snapshot_original_host_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    module = _module(name, source)
    compiled = []
    launches = []

    def capture(kernel, grid, *tensors, **options):
        parameters = tuple(inspect.signature(kernel.__wrapped__).parameters)
        assert len(parameters) == len(tensors)
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = options["block"]
        assert set(options) == {"block"}
        dtypes = {
            torch.float32: cutlass.Float32,
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
        }
        # Match the actual launcher schema, including dynamic shape/stride
        # parameters for empty tensors. A static fake tensor cannot represent
        # the original zero-trip history: CuTe rejects a static zero extent.
        schema: list[tuple[object, ...]] = []
        arguments: list[object] = []
        bake = _cute_bake_tensor_shapes_guard(kernel)
        for tensor in tensors:
            shape, stride = tuple(tensor.shape), tuple(tensor.stride())
            arguments.append(
                cute.runtime.make_ptr(
                    dtypes[tensor.dtype],
                    0,
                    cute.AddressSpace.gmem,
                    assumed_align=_cute_tensor_pointer_alignment(kernel, tensor),
                )
            )
            if bake and all(s > 0 for s in shape):
                schema.append(("tensor", str(tensor.dtype), tensor.ndim, shape, stride))
            else:
                schema.append(("tensor", str(tensor.dtype), tensor.ndim))
                arguments.extend(cutlass.Int64(x) for x in (*shape, *stride))
        wrapper = cast("Any", _create_cute_wrapper(kernel, tuple(schema), block))
        wrapper_source = inspect.getsource(wrapper.__wrapped__)
        arguments.extend(cutlass.Int32(x) for x in grid)
        arguments.append(CUstream(0))
        launches.append(
            {
                "parameters": parameters,
                "grid": grid,
                "block": block,
                "tensors": [
                    {
                        "dtype": str(t.dtype),
                        "shape": tuple(t.shape),
                        "stride": tuple(t.stride()),
                    }
                    for t in tensors
                ],
                "wrapper": wrapper_source,
                "schema": schema,
            }
        )
        compiled.append(
            cute.compile(
                wrapper,
                *arguments,
                options=(
                    f"--dump-dir {directory} --keep-cubin --keep-ptx --gpu-arch sm_103a"
                ),
            ).__ptx__
        )

    module._sequence(*values, _launcher=capture)
    assert len(compiled) == len(launches) == 1
    return compiled[0], launches[0]


@pytest.mark.parametrize(
    "dtype,rows,steps,warps",
    [
        (torch.bfloat16, 128, 0, 4),
        (torch.float16, 129, 1, 8),
    ],
)
def test_snapshot_original_host_native_compile(
    dtype, rows, steps, warps, tmp_path, monkeypatch
):
    before = torch.cuda.is_initialized()
    values = _inputs(dtype, rows, steps)
    source = _source(_sequence, values, _config(warps, 32))
    monkeypatch.chdir(tmp_path)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        ptx, launch = _compile_original_host(source, values, tmp_path)
    assert "tcgen05.ld" in ptx and "tcgen05.st" in ptx
    assert launch["grid"] == ((rows + 127) // 128, 1, 1)
    assert launch["block"] == (512, 1, 1)
    assert torch.cuda.is_initialized() == before
    (tmp_path / "generated.py").write_text(source)
    (tmp_path / "wrapper.py").write_text(launch["wrapper"])
    (tmp_path / "launch.json").write_text(json.dumps(launch, indent=2))
    (tmp_path / "kernel.ptx").write_text(ptx)
