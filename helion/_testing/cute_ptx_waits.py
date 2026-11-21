"""Static checks of a compiled kernel's PTX against the capture guard's
dependency wait, and of what its paths pass between two instructions.

A kernel that takes the guard's wait mode (its last parameter, an Int32:
zero waits at entry, nonzero lets the launch read its inputs ahead of the
wait) must still execute ``griddepcontrol.wait`` ahead of every write to
global memory. ``global_accesses_ahead_of_a_wait`` searches the kernel's
control-flow graph for the writes, and for information the reads, that
some path from the entry reaches without one. A write through a generic
address, which names no state space, counts as one to global memory.

``instructions_reached_without`` searches the same graph from the
instructions a pattern names and returns what a path reaches without
executing an instruction of a second pattern: a barrier that every path
from a decision to a store passes leaves no such store.

The graph's edges are those of ``bra``, ``ret`` and ``trap``. Both searches
refuse a kernel with a ``call`` or a ``brx.idx``: what a function that was
not inlined writes, and where an indexed branch goes, they would not see.
"""

from __future__ import annotations

import re

# A store names its state space behind its qualifiers: ``st.global``,
# ``st.volatile.global``, ``st.relaxed.gpu.global``, ``st.release.sys.global``.
# The bulk copies write global memory from shared memory, plain and through
# a tensor map, and so do their reductions.
_WRITE = re.compile(
    r"\bst\.(?:[\w:]+\.)*global|\batom\.[a-z_.]*global|\bred\.[a-z_.]*global"
    r"|cp\.async\.bulk\.global\.shared|cp\.reduce\.async\.bulk\.global"
    r"|cp\.(?:reduce\.)?async\.bulk\.tensor\.[0-9]d\.global\.shared"
)
# A store, an atomic or a reduction that names no state space addresses
# generically (``st.u32 [%rd1], %r2;``, ``atom.add.u32``, ``red.add.u32``):
# its address may be one of global memory, so it counts as a write there.
# The instruction is a statement's mnemonic, behind a predicate if it has
# one: ``tcgen05.st`` and ``bar.red`` are other instructions. One that
# names shared, local or parameter memory writes there.
_ADDRESSED = re.compile(
    r"(?:^|[;{])\s*(?:@!?%p\d+\s+)?((?:st|atom|red)(?:\.[\w:]+)+)(?=\s)"
)
_STATE_SPACE = re.compile(r"\.(?:global|shared|local|param|const)\b")
# A load names its state space likewise. The bulk copies read global memory
# into shared memory, plain and through a tensor map, and a prefetch reads
# it into the cache.
_READ = re.compile(
    r"\bld\.[a-z_.]*global|cp\.async\.bulk\.tensor\.[0-9]d\.shared::cta\.global"
    r"|cp\.async\.bulk\.shared::(?:cta|cluster)\.global|cp\.async\.bulk\.prefetch"
)
_BRANCH = re.compile(r"^(?:@(!?)(%p\d+)\s+)?bra(?:\.uni)?\s+(\$L__BB\d+_\d+);")
_LABEL = re.compile(r"^(\$L__BB\d+_\d+):")
# A call or an indexed branch as a statement's mnemonic, behind a predicate
# if it has one: control flow the graph has no edges for.
_UNFOLLOWED = re.compile(r"(?:^|[;{])\s*(?:@!?%p\d+\s+)?(?:call|brx\.idx)\b")
# The wait as an instruction of its own: under a predicate (``@%p1
# griddepcontrol.wait;``) some threads pass it without waiting.
_WAIT = re.compile(r"^griddepcontrol\.wait\b")


def _writes_global(text: str) -> bool:
    """Whether the PTX line ``text`` holds an instruction that writes
    global memory, or may: one that names it, or a store, an atomic or a
    reduction through a generic address."""
    return _WRITE.search(text) is not None or any(
        _STATE_SPACE.search(mnemonic) is None for mnemonic in _ADDRESSED.findall(text)
    )


def _entry(lines: list[str]) -> tuple[int, list[tuple[int, str]]]:
    """Where the body of the PTX module's kernel entry starts, the line
    behind its opening brace, and the entry's parameters by position."""
    entries = [
        index
        for index, line in enumerate(lines)
        if line.startswith((".visible .entry", ".entry"))
    ]
    assert entries
    start = entries[-1]
    parameters: list[tuple[int, str]] = []
    body = None
    for index in range(start, len(lines)):
        parameter = re.search(r"\.param\s+\.\w+\s+(\S+_param_(\d+))", lines[index])
        if parameter:
            parameters.append((int(parameter.group(2)), parameter.group(1).rstrip(",")))
        if lines[index].strip() == "{":
            body = index + 1
            break
    assert body is not None and parameters
    return body, parameters


def _blocks(
    lines: list[str], body: int
) -> tuple[list[tuple[str | None, list[str]]], dict[str, int]]:
    """The basic blocks of the entry whose body starts at line ``body``,
    in program order, each with its label and its instructions, and the
    block of every label. A block ends behind a branch, a return or a
    trap, so it falls through to the next unless its last instruction is
    an unconditional one of these. An entry with a call or an indexed
    branch is refused: these are the only edges."""
    blocks: list[tuple[str | None, list[str]]] = [(None, [])]
    depth = 0
    for index in range(body, len(lines)):
        text = lines[index].strip()
        if not text or text.startswith(("//", ".loc")):
            continue
        if text.startswith(".reg") and depth == 0:
            continue
        label = _LABEL.match(text)
        if label:
            blocks.append((label.group(1), []))
            continue
        if text == "{":
            depth += 1
            continue
        if text == "}":
            if depth == 0:
                break
            depth -= 1
            continue
        assert _UNFOLLOWED.search(text) is None, f"unsupported control flow: {text}"
        blocks[-1][1].append(text)
        if _BRANCH.match(text) or text.startswith(("ret", "trap")):
            blocks.append((None, []))
    where = {label: index for index, (label, _text) in enumerate(blocks) if label}
    return blocks, where


def instructions_reached_without(
    ptx: str, *, start: re.Pattern[str], stop: re.Pattern[str]
) -> list[str]:
    """The instructions of the PTX module's kernel entry that some path
    reaches from behind an instruction ``start`` matches without executing
    one that ``stop`` matches, in program order.

    Every conditional branch is taken both ways, a superset of the
    feasible paths, so an instruction that is absent here is behind a
    ``stop`` instruction on every path from every ``start`` instruction:
    with a barrier of a whole group as ``stop``, the group has met between
    the two. A ``start`` instruction is listed only where a path from
    behind one reaches it, as around a loop, and the module must have one.
    """
    lines = ptx.splitlines()
    body, _parameters = _entry(lines)
    blocks, where = _blocks(lines, body)
    stack = [
        (block, position + 1)
        for block, (_label, instructions) in enumerate(blocks)
        for position, text in enumerate(instructions)
        if start.search(text)
    ]
    assert stack
    seen: set[tuple[int, int]] = set()
    reached: set[tuple[int, int]] = set()
    while stack:
        block, first = stack.pop()
        if (block, first) in seen or block >= len(blocks):
            continue
        seen.add((block, first))
        falls = True
        instructions = blocks[block][1]
        for position in range(first, len(instructions)):
            text = instructions[position]
            if stop.search(text):
                falls = False
                break
            reached.add((block, position))
            branch = _BRANCH.match(text)
            if branch:
                _inverted, predicate, target = branch.groups()
                stack.append((where[target], 0))
                falls = predicate is not None
                break
            if text.startswith(("ret", "trap")):
                falls = False
                break
        if falls:
            stack.append((block + 1, 0))
    return [blocks[block][1][position] for block, position in sorted(reached)]


def global_accesses_ahead_of_a_wait(
    ptx: str, *, mode_nonzero: bool
) -> tuple[list[str], list[str]]:
    """The global writes and the global reads of the PTX module's kernel
    entry that a path from the entry reaches without executing
    ``griddepcontrol.wait``, under the wait mode ``mode_nonzero`` names.

    The mode is the entry's last parameter. A branch on a predicate that
    compares it with zero is resolved for the assumed mode; every other
    conditional branch is taken both ways, a superset of the feasible
    paths. A block that executes the wait ends the paths through it; a
    wait under a predicate is no wait, since the path goes on for the
    threads the predicate excludes. Inline assembly counts by its text, so
    a store or a load written there is seen as the instruction it is. A
    store, an atomic or a reduction through a generic address (no state
    space in its name) counts as a write: its address may be global.
    """
    lines = ptx.splitlines()
    body, parameters = _entry(lines)
    _last, mode_name = max(parameters)
    mode_registers = set()
    for line in lines[body:]:
        loaded = re.match(r"\s*ld\.param\.\w+\s+(%\w+), \[(\S+)\];", line)
        if loaded and loaded.group(2) == mode_name:
            mode_registers.add(loaded.group(1))
    assert mode_registers
    # Predicates "mode != 0" (True) and "mode == 0" (False), also negated.
    predicates: dict[str, bool] = {}
    for line in lines[body:]:
        compared = re.match(r"\s*setp\.(eq|ne)\.[bsu]32\s+(%p\d+), (%\w+), 0;", line)
        if compared and compared.group(3) in mode_registers:
            predicates[compared.group(2)] = compared.group(1) == "ne"
    changed = True
    while changed:
        changed = False
        for line in lines[body:]:
            negated = re.match(r"\s*not\.pred\s+(%p\d+), (%p\d+);", line)
            if (
                negated
                and negated.group(2) in predicates
                and negated.group(1) not in predicates
            ):
                predicates[negated.group(1)] = not predicates[negated.group(2)]
                changed = True
    assert predicates
    blocks, where = _blocks(lines, body)
    seen: set[int] = set()
    stack = [0]
    writes: list[str] = []
    reads: list[str] = []
    while stack:
        block = stack.pop()
        if block in seen or block >= len(blocks):
            continue
        seen.add(block)
        falls = True
        for text in blocks[block][1]:
            if _WAIT.match(text):
                falls = False
                break
            if _writes_global(text):
                writes.append(text)
            if _READ.search(text):
                reads.append(text)
            branch = _BRANCH.match(text)
            if branch:
                inverted, predicate, target = branch.groups()
                if predicate is None:
                    stack.append(where[target])
                    falls = False
                elif predicate in predicates:
                    taken = predicates[predicate] == mode_nonzero
                    if inverted:
                        taken = not taken
                    if taken:
                        stack.append(where[target])
                        falls = False
                else:
                    stack.append(where[target])
                break
            if text.startswith(("ret", "trap")):
                falls = False
                break
        if falls:
            stack.append(block + 1)
    return writes, reads
