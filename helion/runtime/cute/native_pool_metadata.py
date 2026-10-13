"""Observe only the installed SDK loader driver-version query in fresh workers."""

from __future__ import annotations

from contextlib import ExitStack
import ctypes
import inspect
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from typing_extensions import Self
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType
    from types import TracebackType


class DriverMetadata:
    def __init__(
        self, loader: str | Path, *, allow_query: bool, calls: list[str]
    ) -> None:
        self.loader = Path(loader).resolve()
        self.allow_query = allow_query
        self.calls = calls
        self.queries = []
        self.blocked_imports = 0
        self.stack = ExitStack()

    def __enter__(self) -> Self:
        original = ctypes.CDLL
        observer = self

        class ObservedCDLL(original):
            def __init__(self, name: str, *args: object, **kwargs: object) -> None:
                basename = os.path.basename(os.fsdecode(name)) if name else ""
                self._proof_driver = basename.startswith(("libcuda.so", "libcudart.so"))
                self._proof_name = basename
                if self._proof_driver and (not observer.allow_query):
                    observer.blocked_imports += 1
                    raise OSError("CPU test: CUDA library import blocked")
                try:
                    super().__init__(
                        name,
                        *cast("tuple[Any, ...]", args),
                        **cast("dict[str, Any]", kwargs),
                    )
                except OSError as error:
                    if self._proof_driver:
                        raise RuntimeError(
                            "Required driver metadata library unavailable"
                        ) from error
                    raise

            # The observed SDK symbol is a Python callable wrapping the C function.
            # pyrefly: ignore [bad-override]
            def __getitem__(self, name_or_ordinal: str) -> Callable[..., Any]:
                name = name_or_ordinal
                if not self.__dict__.get("_proof_driver", False):
                    return super().__getitem__(name)
                if self._proof_name != "libcuda.so.1" or name != "cuDriverGetVersion":
                    observer.calls.append("ctypes CUDA symbol: " + str(name))
                    raise RuntimeError("Forbidden CUDA ctypes symbol: " + str(name))
                function = original.__getitem__(self, name)

                def query(pointer: object) -> int:
                    caller = cast(
                        "FrameType", cast("FrameType", inspect.currentframe()).f_back
                    )
                    assert Path(caller.f_code.co_filename).resolve() == observer.loader
                    assert (
                        caller.f_code.co_name == "_select_and_load_cutlass_ir_toolkit"
                    )
                    assert not observer.queries, "Repeated driver metadata query"
                    assert type(cast("Any", pointer)._obj) is ctypes.c_int
                    result = function(pointer)
                    row = {
                        "result": int(result),
                        "driver_version": cast("Any", pointer)._obj.value,
                    }
                    observer.queries.append(row)
                    if result != 0 or row["driver_version"] <= 0:
                        raise RuntimeError(
                            "Actual driver version query failed; no fallback allowed"
                        )
                    return result

                return query

            # Same deliberate callable replacement as __getitem__.
            # pyrefly: ignore [bad-override]
            def __getattr__(self, name: str) -> Callable[..., Any]:
                if self.__dict__.get("_proof_driver", False):
                    return self[name]
                return super().__getattr__(name)

        self.stack.enter_context(patch.object(ctypes, "CDLL", ObservedCDLL))
        self.stack.enter_context(patch.object(ctypes.cdll, "_dlltype", ObservedCDLL))
        return self

    def finish(self) -> dict[str, Any]:
        assert self.allow_query and len(self.queries) == 1
        extension = Path(
            cast("str", sys.modules["cutlass._mlir._mlir_libs._cutlass_ir"].__file__)
        ).resolve()
        libs = os.environ.get("CUTE_DSL_LIBS", "")
        assert extension.name.startswith("_cutlass_ir.cu")
        toolkit = int(extension.name.split(".cu", 1)[1].split(".", 1)[0])
        assert self.queries[0]["driver_version"] // 1000 >= toolkit
        runtime = self.loader.parents[4] / f"cu{toolkit}/lib/libcute_dsl_runtime.so"
        assert runtime.is_file() and str(runtime) in libs.split(":")
        return {
            "queries": list(self.queries),
            "query_count": 1,
            "toolkit": toolkit,
            "extension": str(extension),
            "runtime_library": str(runtime),
            "cute_dsl_libs": libs,
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.stack.__exit__(exc_type, exc_value, traceback)
