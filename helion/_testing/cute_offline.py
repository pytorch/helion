"""Binds and compiles of rendered CuTe kernels without a device.

``compile_offline`` compiles a render of a kernel bound on CPU the way a
launch does, through the launcher's own compile function and the CuTe DSL's
assembler, and returns the PTX the assembler was given and the SASS it
produced: no CUDA context is created. ``partition_bound`` is the paged
attention partition fixture's kernel bound for the modelled device, which
the CPU tests of the paired schedule's options render and compile.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import struct
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

# pyrefly: ignore [missing-import]
import cuda.bindings.driver as cuda
from cutlass.base_dsl.compiler import DumpDir
from cutlass.base_dsl.compiler import KeepPTX
import cutlass.cute as cute

from . import cpu_render_target
from .cute_codegen import cpu_bind
from .cute_codegen import cpu_target
from .cute_codegen import load_generated_module
from .cute_paged_partition import partition_attention_args
from .cute_paged_partition import partition_attention_kernel
from helion.runtime.cute import launcher
from helion.runtime.cute import sass_check
from helion.runtime.cute.compile_cache import compile_with_options

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

# The host launch function is named after a hash of the wrapper's source.
_LAUNCH_NAME = re.compile(r"_helion_cute_launch_[0-9a-f]{64}")
_PTX_SETMAXNREG = re.compile(r"setmaxnreg\.(inc|dec)\.sync\.aligned\.u32\s+(\d+);")


@dataclasses.dataclass(frozen=True)
class OfflineKernel:
    """What one offline compile was given and what it produced."""

    source: str
    # The PTX handed to the assembler, the host launch name masked.
    ptx: str
    # Threads of the CTA the recorded launch asks for.
    threads: int
    # Registers per thread the assembler has the kernel launch with.
    launch_registers: int
    # ``nvdisasm -c`` of the cubin, or None without the disassembler.
    sass: str | None

    def ptx_quotas(self) -> list[tuple[str, int]]:
        """``setmaxnreg`` directions and counts in program order."""
        return [(kind, int(count)) for kind, count in _PTX_SETMAXNREG.findall(self.ptx)]


def cubin_launch_registers(cubin: bytes) -> list[int]:
    """Registers per thread each kernel of ``cubin`` launches with: the
    ``EIATTR_REGCOUNT`` entries of its ``.nv.info`` section (what
    ``cuobjdump --dump-resource-usage`` prints as ``REG``)."""
    section_table = struct.unpack_from("<Q", cubin, 0x28)[0]
    entry_size, entries, names_index = struct.unpack_from("<HHH", cubin, 0x3A)

    def section(index: int) -> tuple[int, int, int]:
        name, _kind, _flags, _address, offset, size = struct.unpack_from(
            "<IIQQQQ", cubin, section_table + index * entry_size
        )
        return name, offset, size

    _, names, _ = section(names_index)
    counts = []
    for index in range(entries):
        name, offset, size = section(index)
        if cubin[names + name : cubin.index(b"\0", names + name)] != b".nv.info":
            continue
        position = offset
        while position < offset + size:
            form, attribute, value = struct.unpack_from("<BBH", cubin, position)
            position += 4
            if form == 4:  # A value of ``value`` bytes follows the entry.
                if attribute == 0x2F:  # Register count: symbol index, count.
                    counts.append(struct.unpack_from("<II", cubin, position)[1])
                position += value
    return counts


class _OfflineCompile:
    """``cute.compile`` as the launcher's compile function indexes it: the
    option objects that function chose, plus the PTX dump, compiled without
    the DSL's JIT engine or cache."""

    def __init__(self, compile_callable: object, dump_dir: Path) -> None:
        self._compile = cast("Any", compile_callable)
        self._dump_dir = dump_dir

    def __getitem__(self, options: tuple[object, ...]) -> Callable[..., object]:
        kept = self._compile[(*options, KeepPTX(True), DumpDir(str(self._dump_dir)))]
        return functools.partial(kept, no_jit_engine=True, no_cache=True)


def compile_offline(
    bound: BoundKernel[object],
    args: tuple[object, ...],
    source: str,
    dump_dir: Path,
) -> OfflineKernel:
    """Compile ``source`` (a render of ``bound``) the way a launch does,
    without a device: the launcher's own wrapper for the recorded launch,
    through the launcher's compile function (``compile_with_options``) and so
    under its options, the FFI bridge and its division. The DSL's own
    defaults assemble another kernel."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / "kernel.py"
    path.write_text(source)
    calls: list[tuple[object, tuple[int, ...], tuple[object, ...], dict]] = []

    def record(
        entry: object, grid: tuple[int, ...], *launch_args: object, **kwargs: object
    ) -> None:
        calls.append((entry, grid, launch_args, kwargs))

    with cpu_render_target(fake_device_sm_count=148, forbid_native=False):
        module = load_generated_module(path)
        getattr(module, bound.kernel.name)(*args, _launcher=record)
    ((entry, grid, launch_args, kwargs),) = calls
    block = cast("tuple[int, int, int]", kwargs.pop("block"))
    compile_options = cast("str | None", kwargs.pop("cute_compile_options", None))
    assert not kwargs, kwargs
    grid3 = cast("tuple[int, int, int]", tuple(int(g) for g in (*grid, 1, 1)[:3]))
    # The wrapper is compiled, never called: CPU addresses stand in for the ABI.
    with patch.object(launcher, "_validate_cute_launcher_tensor", lambda tensor: None):
        built = launcher._build_cute_schema_and_args(entry, tuple(launch_args), grid3)
    wrapper = launcher._create_cute_wrapper(
        entry, built.schema, block, num_sm=148, device=None
    )
    with (
        cpu_target(forbid_native=False),
        patch.object(cute, "compile", _OfflineCompile(cute.compile, dump_dir)),
    ):
        compiled = cast(
            "Any",
            compile_with_options(
                wrapper, (*built.launch_args, cuda.CUstream(0)), compile_options
            ),
        )
    (ptx_path,) = dump_dir.glob("*.ptx")
    ptx = _LAUNCH_NAME.sub("_helion_cute_launch", ptx_path.read_text())
    (cubin,) = sass_check._cubin_blobs(compiled.ir_module)
    (launch_registers,) = cubin_launch_registers(cubin)
    sass = None
    if sass_check.cutedsl_compat.nvdisasm_path() is not None:
        sass = sass_check.disassemble(cubin)
    return OfflineKernel(source, ptx, math.prod(block), launch_registers, sass)


# The partition fixture's units in ``partition_bound``: seventeen head pairs.
# On the modelled 148 SMs a persistent launch runs one CTA per unit.
PARTITION_UNITS = 17


@functools.cache
def partition_bound() -> tuple[BoundKernel[object], tuple[object, ...]]:
    """The partition fixture's kernel bound on CPU for the modelled device."""
    args = partition_attention_args(units=PARTITION_UNITS)
    with cpu_render_target(fake_device_sm_count=148, num_sm=148):
        bound = cpu_bind(
            cast("Kernel[object]", partition_attention_kernel()), args, num_sm=148
        )
    return bound, args
