"""The call a Helion kernel's first call is timed behind
(``helion.inductor.kernel_sequence``), named where the wrapper code is
written: the Helion kernel call just before it, when nothing between the two
may launch other work and that call is its kernel's first, the call the
runtime records. In the output code::

    helion_fused_0(arg0_1, buf0)
    buf1 = empty_strided_cuda((128, 96), (96, 1), torch.float32)
    helion_fused_sum_1(buf0, buf1)

defines ``helion_fused_sum_1`` as::

    helion_fused_sum_1 = helion_in_sequence(
        helion_fused_sum_1, globals(), "helion_fused_sum_1", after="helion_fused_0"
    )

A first call after ``extern_kernels.mm(buf0, arg1_1, out=buf2)`` gets
``after=None`` and is tuned alone: timed behind ``helion_fused_0``, its
search would replay a call that does not run just before it. So does one
after the second call of a kernel: in ``K(a, b); K(b, c); L(c, d)`` the
runtime records only ``K(a, b)``, and behind it ``L`` would find in L2 what
``K(a, b)`` left, not the ``c`` that ``K(b, c)`` has just written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import TYPE_CHECKING
import weakref

from torch._inductor.codegen.wrapper import CommentLine
from torch._inductor.codegen.wrapper import FreeLine
from torch._inductor.codegen.wrapper import MemoryPlanningLine
from torch._inductor.utils import LineContext

if TYPE_CHECKING:
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

# Wrapper lines that launch no GPU work: buffer allocations, reuses, views
# and frees, and comments.  Any other line may (an extern kernel, a fallback
# op, a copy_), and so may a string other than a comment or a ``del``.  A
# device context counts as a launch: the call after it may run on another
# device than the call before.
_NO_LAUNCH = (MemoryPlanningLine, FreeLine, CommentLine, LineContext)


def _launches(line: object) -> bool:
    if isinstance(line, _NO_LAUNCH):
        return False
    if isinstance(line, str):
        stripped = line.strip()
        return bool(stripped) and not stripped.startswith(("#", "del "))
    return True


@dataclasses.dataclass
class _Calls:
    """The Helion kernel calls written so far for one generated module: its
    root wrapper's and those of its subgraphs and graph partitions, which
    share the root's kernels (``src_to_kernel``), so a kernel's first call
    in one partition is not its first in the next."""

    called: set[str] = dataclasses.field(default_factory=set)
    # The last one, as (kernel name, the wrapper written to, index of its
    # line in that wrapper's lines), when it was its kernel's first call;
    # None when it was a later one.
    last: tuple[str, weakref.ref[PythonWrapperCodegen], int] | None = None
    # The buffers each kernel's first call passed (``_buffer``), by position
    args: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # The key each kernel's module is loaded under (module_key)
    keys: dict[str, str] = dataclasses.field(default_factory=dict)


_CALLS: weakref.WeakKeyDictionary[PythonWrapperCodegen, _Calls] = (
    weakref.WeakKeyDictionary()
)


def predecessor(wrapper: PythonWrapperCodegen) -> str | None:
    """The kernel whose first call the next call written to ``wrapper``, a
    kernel's first, is timed behind (module docstring); None: tuned alone,
    also when the call before was written to another wrapper (another graph
    partition, across the op that split the graph)."""
    calls = _CALLS.get(wrapper.get_root_graph())
    if calls is None or calls.last is None:
        return None
    name, written_to, index = calls.last
    if written_to() is not wrapper or any(
        _launches(line) for line in wrapper.lines[index + 1 :]
    ):
        return None
    return name


def _buffer(arg: str) -> str:
    """The buffer a call argument passes: ``buf0`` of ``buf0``,
    ``buf0.view(64, 128, 3072)`` or ``reinterpret_tensor(buf0, ...)``."""
    inner = re.match(r"reinterpret_tensor\(\s*(\w+)", arg)
    if inner is not None:
        return inner.group(1)
    match = re.match(r"\w+", arg)
    return match.group(0) if match is not None else arg


def module_key(
    wrapper: PythonWrapperCodegen,
    name: str,
    after: str,
    after_source: str,
    args: list[str],
) -> str:
    """The key kernel ``name``, called with ``args``, loads its module under,
    timed behind kernel ``after`` (of source ``after_source``): a hash of
    that source, of the key ``after``'s own module has, and of which of
    ``after``'s buffers each argument passes. C behind B behind A and C
    behind B behind D get modules of their own: B runs at the config
    selected behind A in one and behind D in the other. So do B in
    ``A(x, tmp); B(tmp, out)`` and in ``A(x, tmp); B(y, out)``: behind the
    first A, B reads the buffer A has just written; behind the second, a
    buffer A did not touch."""
    calls = _CALLS.setdefault(wrapper.get_root_graph(), _Calls())
    before = [_buffer(arg) for arg in calls.args.get(after, [])]
    shared = [
        before.index(buffer) if buffer in before else -1
        for buffer in map(_buffer, args)
    ]
    chained = f"{after_source}{calls.keys.get(after, '')}{shared}"
    calls.keys[name] = hashlib.sha256(chained.encode()).hexdigest()[:16]
    return calls.keys[name]


def note_call(wrapper: PythonWrapperCodegen, name: str, args: list[str]) -> None:
    """Record the call of kernel ``name`` with ``args`` just written to
    ``wrapper``."""
    calls = _CALLS.setdefault(wrapper.get_root_graph(), _Calls())
    first = name not in calls.called
    calls.called.add(name)
    calls.last = (name, weakref.ref(wrapper), len(wrapper.lines) - 1) if first else None
    if first:
        calls.args[name] = list(args)
