"""Compile the toolkit's public cooperative-grid sync into an inline device operation."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import functools
import importlib
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import TYPE_CHECKING

from ...exc import InvalidConfig

if TYPE_CHECKING:
    from types import ModuleType


def _leaf_body(source: str) -> str | None:
    """Accept one void, argument-free leaf with register operands and local branches.

    Treat the implementation as CUDA's: never assume a driver workspace address,
    register number, synchronization algorithm or instruction sequence.
    """
    matches = list(
        re.finditer(r"(?m)^\s*\.visible\s+\.func\s+helion_grid_sync\(\)\s*\{", source)
    )
    if len(matches) != 1:
        return None
    raw, sep, _ = source[matches[0].end() :].partition("}")
    if not sep or "{" in raw:
        return None
    lines = [line.split("//", 1)[0].strip() for line in raw.splitlines()]
    code = [line for line in lines if line]
    if not code or code[-1] != "ret;":
        return None
    code.pop()
    label_re = re.compile(r"([\w$]+):")
    labels = [m[1] for line in code if (m := label_re.fullmatch(line))]
    if len(set(labels)) != len(labels):
        return None
    renamed = {
        label: f"helion_cg_{i}_" + "$" + "{:uid}" for i, label in enumerate(labels)
    }
    registers = re.compile(r"%\w+(?:\.\w+)*")
    groups: dict[str, int] = {}
    for line in code:
        if decl := re.fullmatch(r"\.reg\s+\.\w+\s+(%[A-Za-z_]\w*?)<(\d+)>;", line):
            groups[decl[1]] = int(decl[2])
    # These PTX special registers are the inputs of NVIDIA's grid.sync.
    # Other global names (including a percent-prefixed external) cannot be
    # copied into an otherwise self-contained inline assembly operation.

    def local_register(name: str) -> bool:
        if re.fullmatch(r"%(?:tid|ctaid|ntid|nctaid)\.[xyz]", name):
            return True
        if match := re.fullmatch(r"%envreg(\d+)", name):
            return match[1] == str(int(match[1])) and int(match[1]) < 32
        return any(
            name.startswith(group)
            and (n := name[len(group) :]).isdigit()
            and n == str(int(n))
            and int(n) < size
            for group, size in groups.items()
        )

    emitted: list[str] = []
    for line in code:
        if definition := label_re.fullmatch(line):
            emitted.append(renamed[definition[1]] + ":")
            continue
        if re.fullmatch(r"\.reg\s+\.\w+\s+%\w+<\d+>;", line):
            emitted.append(line)
            continue
        inst = re.fullmatch(r"(?:@!?%\w+\s+)?([a-z][a-z0-9_.]*)(?:\s+(.*?))?;", line)
        if not inst or inst[1].split(".")[0] in {"call", "ret", "exit"}:
            return None
        if not all(map(local_register, registers.findall(line))):
            return None
        if inst[1] in {"bra", "bra.uni"}:
            label = inst[2]
            if label not in renamed:
                return None
            emitted.append(line[: inst.start(2)] + renamed[label] + line[inst.end(2) :])
            continue
        operands = registers.sub("", inst[2] or "")
        operands = re.sub(r"\b0[xX][0-9a-fA-F]+\b", "", operands)
        # No references to an external global, function or register alias.
        if re.search(r"[$a-zA-Z_][\w$]*", operands):
            return None
        emitted.append(line)
    asm = "\n".join(emitted)
    return "{\n" + asm + "\n}"


def _toolkit_roots(version: tuple[int, int]) -> tuple[Path, ...]:
    roots = [
        Path(value)
        for key in ("CUDA_HOME", "CUDA_PATH")
        if (value := os.environ.get(key))
    ]
    roots.append(Path(f"/usr/local/cuda-{version[0]}.{version[1]}"))
    if nvcc := shutil.which("nvcc"):
        roots.append(Path(nvcc).resolve().parents[1])
    roots.append(Path("/usr/local/cuda"))
    # CUDA runtime wheels install headers in a namespace (without __init__.py).
    # The same exact CUDART_VERSION check below applies to every candidate.
    for entry in sys.path:
        if entry:
            namespace = Path(entry) / "nvidia"
            roots.extend((namespace / f"cu{version[0]}", namespace / "cuda_runtime"))
    return tuple(dict.fromkeys(roots))


def _include_options(version: tuple[int, int]) -> tuple[tuple[Path, ...], ...]:
    # Headers from a different release can reject NVRTC's internal builtins.
    # Prefer an explicitly installed matching toolkit, then the versioned root.
    expected = 1000 * version[0] + 10 * version[1]
    options: list[tuple[Path, ...]] = []
    for root in _toolkit_roots(version):
        inc = root / "include"
        try:
            if not all(
                (inc / rel).is_file()
                for rel in (
                    "cooperative_groups.h",
                    "cooperative_groups/details/info.h",
                    "cooperative_groups/details/sync.h",
                )
            ):
                continue
            header = (inc / "cuda_runtime_api.h").read_text(encoding="utf8")
            version_match = re.search(
                r"(?m)^\s*#\s*define\s+CUDART_VERSION\s+(\d+)\b", header
            )
            if version_match and int(version_match[1]) == expected:
                candidates = [inc, inc / "cccl"]
                if root.name == "cuda_runtime" and root.parent.name == "nvidia":
                    candidates.append(root.parent / "cuda_cccl" / "include")
                paths = tuple(p for p in candidates if p.is_dir())
                if any((p / "nv" / "target").is_file() for p in paths):
                    options.append(paths)
        except (OSError, UnicodeError):
            continue
    return tuple(dict.fromkeys(options))


@dataclass(frozen=True)
class _Toolkit:
    nvrtc: ModuleType
    includes: tuple[Path, ...]
    architectures: frozenset[int]
    alternate_includes: tuple[tuple[Path, ...], ...] = ()
    # Absent until a config requests the leaf. A present None retires that
    # architecture after a rejected implementation, without probing other GPUs.
    bodies: dict[int, str | None] = field(default_factory=dict, compare=False)
    failures: dict[int, str] = field(default_factory=dict, compare=False)


_discovery_lock = threading.Lock()
_leaf_lock = threading.Lock()


def _reset_locks_in_child() -> None:
    global _discovery_lock, _leaf_lock
    _discovery_lock = threading.Lock()
    _leaf_lock = threading.Lock()


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_reset_locks_in_child)


def _toolkit() -> _Toolkit | None:
    # lru_cache alone can run the first factory on more than one thread.
    # A single cache owner must receive every compilation and rejection.
    with _discovery_lock:
        return _installed_toolkit()


@functools.lru_cache(maxsize=1)
def _installed_toolkit() -> _Toolkit | None:
    """Check the local compiler, architecture table and headers, without compiling."""
    try:
        nvrtc = importlib.import_module("cuda.bindings.nvrtc")
        ok = nvrtc.nvrtcResult.NVRTC_SUCCESS
        status, major, minor = nvrtc.nvrtcVersion()
        if status != ok:
            return None
        status, arches = nvrtc.nvrtcGetSupportedArchs()
    except (ImportError, OSError, RuntimeError):
        return None
    if status != ok or not (options := _include_options((major, minor))):
        return None
    return _Toolkit(nvrtc, options[0], frozenset(arches), options[1:])


def grid_toolkit_installed(capability: tuple[int, int]) -> bool:
    """Whether the installed compiler and headers support the architecture."""
    toolkit = _toolkit()
    return (
        toolkit is not None
        and capability[0] * 10 + capability[1] in toolkit.architectures
    )


def grid_sync_rejected(capability: tuple[int, int]) -> bool:
    """Whether compiling the actual leaf already failed for this architecture."""
    toolkit = _toolkit()
    arch = capability[0] * 10 + capability[1]
    return (
        toolkit is not None and arch in toolkit.bodies and toolkit.bodies[arch] is None
    )


def grid_sync_requirement(capability: tuple[int, int] | None) -> str:
    requirement = "CUDA grid-group support (matching NVRTC and CUDA headers)"
    toolkit = _toolkit()
    if toolkit is not None and capability is not None:
        if reason := toolkit.failures.get(capability[0] * 10 + capability[1]):
            return f"{requirement}: {reason}"
    return requirement


def grid_sync_asm(capability: tuple[int, int]) -> str | None:
    """Compile the selected public operation once for this toolkit and architecture."""
    toolkit = _toolkit()
    arch = capability[0] * 10 + capability[1]
    if toolkit is None or arch not in toolkit.architectures:
        return None
    if arch not in toolkit.bodies:
        with _leaf_lock:
            if arch not in toolkit.bodies:
                toolkit.bodies[arch] = _compile_grid_sync(toolkit, arch)
    return toolkit.bodies[arch]


class GridCompilerUnavailable(InvalidConfig):
    """The installed compiler could not run; this is not a rejected leaf."""


def _checked_status(nvrtc: ModuleType, status: object, operation: str) -> None:
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise GridCompilerUnavailable(f"CUDA cooperative grid {operation}: {status!s}")


def _compile_grid_sync(toolkit: _Toolkit, arch: int) -> str | None:
    # A matched toolkit may still be incomplete. Recreate the NVRTC program
    # per include set; do not silently combine two toolkit header trees.
    for includes in (toolkit.includes, *toolkit.alternate_includes):
        if body := _compile_with_includes(toolkit, arch, includes):
            toolkit.failures.pop(arch, None)
            return body
    return None


def _compile_with_includes(
    toolkit: _Toolkit, arch: int, includes: tuple[Path, ...]
) -> str | None:
    nvrtc = toolkit.nvrtc
    src = b'#include <cooperative_groups.h>\nextern "C" __device__ void helion_grid_sync() { cooperative_groups::this_grid().sync(); }\n'
    status, program = nvrtc.nvrtcCreateProgram(src, b"helion_grid_sync.cu", 0, [], [])
    _checked_status(nvrtc, status, "nvrtcCreateProgram")
    args = [
        b"--std=c++17",
        f"--gpu-architecture=compute_{arch}".encode(),
        b"--relocatable-device-code=true",
        *(os.fsencode(f"-I{d}") for d in includes),
    ]
    try:
        status = nvrtc.nvrtcCompileProgram(program, len(args), args)[0]
        # A source rejected by this toolchain is stable; memory pressure,
        # compiler initialization and internal errors are not.
        if status in (
            nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION,
            nvrtc.nvrtcResult.NVRTC_ERROR_INVALID_OPTION,
        ):
            reason = f"nvrtcCompileProgram: {status!s}"
            log_status, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            if log_status == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size:
                buf = bytearray(log_size)
                if (
                    nvrtc.nvrtcGetProgramLog(program, buf)[0]
                    == nvrtc.nvrtcResult.NVRTC_SUCCESS
                ):
                    reason += ": " + buf.rstrip(b"\0").decode(errors="replace").strip()
            toolkit.failures[arch] = reason
            return None
        _checked_status(nvrtc, status, "nvrtcCompileProgram")
        status, size = nvrtc.nvrtcGetPTXSize(program)
        _checked_status(nvrtc, status, "nvrtcGetPTXSize")
        ptx = bytearray(size)
        _checked_status(nvrtc, nvrtc.nvrtcGetPTX(program, ptx)[0], "nvrtcGetPTX")
        result = _leaf_body(ptx.rstrip(b"\0").decode("ascii"))
        if result is None:
            toolkit.failures[arch] = (
                "the compiled cooperative operation is not a self-contained PTX leaf"
            )
        return result
    finally:
        nvrtc.nvrtcDestroyProgram(program)
