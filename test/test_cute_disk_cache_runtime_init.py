"""Fresh-process runtime initialization for ordinary CuTe disk-cache hits."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ("ffi", "plain", "miss", "corrupt"))
def test_fresh_process_cache_runtime_init(tmp_path: Path, mode: str) -> None:
    pytest.importorskip("cutlass")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_100a")
    result = subprocess.run(
        [sys.executable, "-B", __file__, "--probe", mode, str(tmp_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "pass"' in result.stdout


def _probe(mode: str, directory: str) -> None:
    # A real CRC cache round trip and loader are used. Only the execution
    # engine is a boundary: resolve actual process-global FFI symbols there,
    # without compiling or invoking host/device machine code.
    import ctypes
    import importlib
    import json
    from types import SimpleNamespace
    from typing import Any
    from typing import cast
    from unittest.mock import patch

    from cutlass._mlir import ir
    from cutlass.base_dsl.compiler import Compiler
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
    import torch

    from helion.runtime.cute import launcher

    assert not torch.cuda.is_initialized() and "tvm_ffi" not in sys.modules
    os.environ["CUTE_DSL_CACHE_DIR"] = directory
    options = "--generate-line-info --enable-tvm-ffi" if mode != "plain" else None
    wrapper = launcher._CompiledCuteLauncher(lambda: None, options, "runtime-init")
    # The SDK's extension-provided MLIR classes are absent from its type stub.
    mlir = cast("Any", ir)
    with mlir.Context():
        module = mlir.Module.parse("module {}")
    if mode != "miss":
        wrapper._persist_to_disk(
            SimpleNamespace(
                ir_module=module, function_name="saved_entry", has_gpu_module=False
            )
        )
    if mode == "corrupt":
        path = Path(wrapper._cache_file_paths()[1])
        path.write_bytes(path.read_bytes()[:-1] + b"x")
    calls: list[str] = []

    class Engine:
        def lookup(self, name: str):
            assert name == "saved_entry"
            calls.append("lookup")
            return SimpleNamespace(restype=None)

    def jit(self, module, **kwargs):
        calls.append("jit")
        assert str(module).strip() == "module {\n}"
        if mode == "ffi":
            assert "tvm_ffi" in sys.modules
            library = ctypes.CDLL(None)
            assert library.TVMFFIErrorSetRaisedFromCStr
            assert library.TVMFFIErrorSetRaisedFromCStrParts
        else:
            assert "tvm_ffi" not in sys.modules
        return Engine()

    def forbidden(*args, **kwargs):
        raise AssertionError("no compilation or CUDA initialization")

    with (
        patch.object(Compiler, "jit", jit),
        patch.object(Compiler, "compile", forbidden),
        patch.object(torch.cuda, "_lazy_init", forbidden),
        patch.object(
            importlib, "import_module", wraps=importlib.import_module
        ) as imports,
    ):
        value = wrapper._reload_from_disk()
        ffi_imports = [
            call for call in imports.call_args_list if call.args == ("tvm_ffi",)
        ]
        assert len(ffi_imports) == int(mode == "ffi")
        if mode in ("miss", "corrupt"):
            assert value is None and not calls and "tvm_ffi" not in sys.modules
        else:
            assert isinstance(value, CudaDialectJitCompiledFunction)
            assert value.function_name == "saved_entry" and not value.has_gpu_module
            assert calls == ["jit", "lookup"]
        # Ordinary repeated dispatch does not enter the loader or import path.
        wrapper._compiled = lambda x: x
        imports.reset_mock()
        with patch.object(
            launcher._CompiledCuteLauncher, "_reload_from_disk", forbidden
        ):
            assert wrapper(17) == 17 and wrapper(19) == 19
        assert not imports.call_args_list
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            {"status": "pass", "mode": mode, "native_compiles": 0, "GPU_calls": 0}
        )
    )


if __name__ == "__main__":
    assert sys.argv[1] == "--probe"
    _probe(sys.argv[2], sys.argv[3])
