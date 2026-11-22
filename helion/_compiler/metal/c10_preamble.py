"""The c10 Metal declarations a kernel uses, in place of whole-header includes.

Helion's MSL includes torch's ``c10/metal`` headers for the helpers its
kernels call (``c10::metal::erf``, the SIMD reductions, ...), and with them
everything those headers pull in: about 160 KB of source, of which a kernel
uses a few declarations.  Where a kernel's source is regenerated and hashed
per call (ExecuTorch's MLX delegate rebuilds each custom kernel's source on
every execution), its size is host time; and each compile parses it all.

:func:`prune` replaces the include lines with a self-contained preamble,
read statically from torch's own headers, with no trial compiles:

- the headers are expanded, each once (``#pragma once``), from torch's
  include directory;
- conditionals on macros whose value is known when compiling Metal are
  resolved (``__METAL__`` is set; the stable-ABI guards each header is
  wrapped in, and the CUDA and HIP macros, are not), others kept as written;
- of the declarations at namespace scope, those reachable from the
  identifiers the rest of the kernel names are kept, in their order: a
  name's overloads and specializations together, since dropping one can
  send a call through an implicit conversion to another; a macro wherever a
  kept declaration names it; an operator wherever it names a kept type.

``metal_jit`` compiles the whole includes instead when the pruned source
doesn't compile, so a kernel that compiles with them still compiles.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
import re
from typing import NamedTuple

import torch
from torch._utils_internal import get_file_path

#: Macros the headers' conditionals test whose value is known when compiling
#: Metal: ``__METAL__`` is set; the stable-ABI guards, and the host, CUDA and
#: HIP macros, are not.  A conditional on any other macro is kept as written.
DEFINED = frozenset({"__METAL__"})
UNDEFINED = frozenset(
    {
        "TORCH_STABLE_ONLY",
        "TORCH_TARGET_VERSION",
        "__CUDACC__",
        "__CUDA_ARCH__",
        "__HIPCC__",
        "__HIP_DEVICE_COMPILE__",
    }
)

_C10_INCLUDE = re.compile(
    r"^[ \t]*#[ \t]*include[ \t]*<(c10/[^>]+)>[ \t]*$", re.MULTILINE
)
_IDENT = re.compile(r"[A-Za-z_]\w*")
# A name in Metal's own namespace, which no c10 declaration declares: ``::metal::``
# anywhere, and ``metal::`` outside c10's namespaces (in the kernel).
_GLOBAL_METAL = re.compile(r"(?<![\w:])::metal::(?:\w+::)*\w+")
_KERNEL_METAL = re.compile(r"(?<![\w:])(?:::)?metal::(?:\w+::)*\w+")
# String and character literals.
_LITERAL = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|(?<!\w)'(?:\\.|[^'\\\n])*'")
# Comments, and the literals a comment marker inside them doesn't open.
_LEXEME = re.compile(r"//[^\n]*|/\*.*?\*/|" + _LITERAL.pattern, re.DOTALL)
#: A comment block that licenses or attributes code, which the code it
#: covers keeps.
_NOTICE = re.compile(
    r"(?i)copyright|licen[cs]e|permission is hereby|redistribution|\bNOTICE\b|"
    r"(?:ported|taken|copied|derived|adapted|borrowed) from|copy-(?:and|n)-paste"
)
#: Lines :func:`_expand` marks a header's start and its notices with.
_FILE_MARK = "#pragma helion_c10_file "
_NOTICE_MARK = "#pragma helion_c10_notice "
_DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*)$", re.DOTALL)
_NAMESPACE = re.compile(r"^(?:inline\s+)?namespace\s*([\w:]*)\s*\{$")
_RECORD = re.compile(
    r"^(?:typedef\s+)?(struct|class|union|enum(?:\s+(?:class|struct))?)\s+"
    r"(?:alignas\s*\((?:[^()]|\([^()]*\))*\)\s*|__attribute__\s*\(\([^)]*\)\)\s*)*"
    r"([A-Za-z_]\w*)\s*(?=[{:;<]|$)"
)
_SPECIFIERS = re.compile(
    r"^(?:(?:inline|static|constexpr|constant|const|extern|thread_local|"
    r"METAL_FUNC|C10_\w+)\s+)+"
)
#: Words no declaration is named: a name read as one was misread.
_KEYWORDS = frozenset(
    [
        "auto",
        "bool",
        "char",
        "class",
        "const",
        "constexpr",
        "constant",
        "device",
        "double",
        "enum",
        "extern",
        "float",
        "half",
        "inline",
        "int",
        "long",
        "short",
        "signed",
        "static",
        "struct",
        "template",
        "thread",
        "threadgroup",
        "typedef",
        "typename",
        "uint",
        "uchar",
        "ushort",
        "ulong",
        "union",
        "unsigned",
        "using",
        "void",
        "volatile",
    ]
)
#: Words that precede ``(`` without naming a function.
_NOT_FUNCTIONS = frozenset(
    {
        "alignas",
        "alignof",
        "decltype",
        "noexcept",
        "sizeof",
        "static_assert",
        "__attribute__",
    }
)


class _Unparsed(Exception):
    """The headers hold a construct :func:`_pieces` can't split."""


@dataclasses.dataclass
class _Piece:
    """One unit of a header at namespace scope: a declaration, a directive or
    an unresolved conditional block, with the namespaces it sits in."""

    text: str
    namespaces: tuple[str, ...]
    kind: str  # "decl", "operator", "define", "undef", "cond", "keep"
    names: frozenset[str] = frozenset()
    refs: frozenset[str] = frozenset()
    #: The license and attribution comments over it (:func:`_expand`).
    notices: tuple[int, ...] = ()


def include_dir() -> Path:
    """The directory torch's headers are installed in, as
    ``torch.mps.compile_shader`` finds them."""
    return Path(get_file_path("torch")) / "include"


def _strip_comments(text: str) -> str:
    """``text`` without comments, keeping literals and line breaks."""

    def repl(match: re.Match[str]) -> str:
        lexeme = match.group(0)
        if lexeme[0] in "\"'":
            return lexeme
        return "\n" * lexeme.count("\n") if lexeme.startswith("/*") else ""

    return _LEXEME.sub(repl, text)


def _logical_lines(text: str) -> list[str]:
    """``text``'s lines, a directive joined with its continuation lines."""
    lines: list[str] = []
    for line in text.split("\n"):
        if lines and lines[-1].endswith("\\"):
            lines[-1] += "\n" + line
        else:
            lines.append(line)
    return lines


def _evaluate(expr: str) -> bool | None:
    """A conditional's value, or None when it tests a macro whose value
    isn't known (:data:`DEFINED`, :data:`UNDEFINED`).  Only ``defined``,
    ``!``, ``&&``, ``||`` and parentheses are evaluated; anything else (a
    comparison, ``__has_include``) is unknown."""
    tokens = re.findall(
        r"defined\s*\(\s*\w+\s*\)|defined\s+\w+|&&|\|\||[!()]|[^\s!()&|]+|[&|]", expr
    )
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def either() -> bool | None:
        nonlocal pos
        values = [both()]
        while peek() == "||":
            pos += 1
            values.append(both())
        if True in values:
            return True
        return None if None in values else False

    def both() -> bool | None:
        nonlocal pos
        values = [unary()]
        while peek() == "&&":
            pos += 1
            values.append(unary())
        if False in values:
            return False
        return None if None in values else True

    def unary() -> bool | None:
        nonlocal pos
        token = peek()
        if token == "!":
            pos += 1
            value = unary()
            return None if value is None else not value
        if token == "(":
            pos += 1
            value = either()
            if peek() == ")":
                pos += 1
                return value
            return None
        if token is not None and token.startswith("defined"):
            pos += 1
            name = re.findall(r"\w+", token)[-1]
            if name in DEFINED:
                return True
            return False if name in UNDEFINED else None
        # An atom this doesn't evaluate: skip to the next operator at this
        # level.
        depth = 0
        while (token := peek()) is not None:
            if depth == 0 and token in ("&&", "||", ")"):
                break
            depth += {"(": 1, ")": -1}.get(token, 0)
            pos += 1
        return None

    value = either()
    return value if pos == len(tokens) else None


def _resolve(text: str) -> str:
    """``text`` with each conditional whose value is known (:func:`_evaluate`)
    replaced by its taken branch; the others are kept as written."""
    out: list[str] = []
    # Each open conditional: [kept as written, emitting, a branch taken].
    stack: list[list[bool]] = []

    def emitting() -> bool:
        return all(frame[1] for frame in stack if not frame[0])

    for line in _logical_lines(text):
        match = _DIRECTIVE.match(line)
        word = match.group(1) if match else ""
        expr = match.group(2).strip() if match else ""
        if word in ("if", "ifdef", "ifndef"):
            if word == "ifdef":
                expr = f"defined({expr})"
            elif word == "ifndef":
                expr = f"!defined({expr})"
            value = _evaluate(expr)
            if value is None:
                if emitting():
                    out.append(line)
                stack.append([True, True, False])
            else:
                stack.append([False, value, value])
            continue
        if word in ("elif", "else", "endif") and stack:
            frame = stack[-1]
            if word == "endif":
                stack.pop()
                if frame[0] and emitting():
                    out.append(line)
                continue
            if frame[0]:
                if emitting():
                    out.append(line)
                continue
            if word == "else":
                frame[1] = not frame[2]
                frame[2] = True
                continue
            value = False if frame[2] else _evaluate(expr)
            if value is None:
                # The branches before were all false: the rest of the chain
                # is a conditional of its own, kept as written.
                stack[-1] = [True, True, False]
                if emitting():
                    out.append(f"#if {expr}")
                continue
            frame[1] = value
            frame[2] = frame[2] or value
            continue
        if emitting():
            out.append(line)
    return "\n".join(out)


def _mark_notices(text: str, notices: list[str]) -> str:
    """``text`` with each comment block that licenses or attributes code
    (:data:`_NOTICE`), a block comment or a run of line comments, replaced
    by a line marking it, and the block added to ``notices``."""
    blocks: list[tuple[int, int]] = []
    for match in _LEXEME.finditer(text):
        if match.group(0)[0] in "\"'":
            continue
        start, end = match.span()
        if (
            blocks
            and match.group(0).startswith("//")
            and text[blocks[-1][0] : blocks[-1][0] + 2] == "//"
            and not text[blocks[-1][1] : start].strip()
            and text[blocks[-1][1] : start].count("\n") == 1
        ):
            blocks[-1] = (blocks[-1][0], end)
        else:
            blocks.append((start, end))
    out: list[str] = []
    last = 0
    for start, end in blocks:
        block = text[start:end]
        if not _NOTICE.search(block):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        out.append(text[last:start])
        if text[line_start:start].strip():
            out.append("\n")
        out.append(f"{_NOTICE_MARK}{len(notices)}\n" + "\n" * block.count("\n"))
        notices.append(block)
        last = end
    out.append(text[last:])
    return "".join(out)


def _expand(rel: str, seen: set[str], notices: list[str]) -> str:
    """The header ``rel`` under :func:`include_dir`, its conditionals resolved
    and its c10 includes expanded, each header once; an include whose file
    isn't there stays an include.  Its license and attribution comments
    stay, in ``notices``, as lines marking where they were, and lines mark
    which header the text that follows comes from."""
    if rel in seen:
        return ""
    path = include_dir() / rel
    if not path.is_file():
        return f"#include <{rel}>"
    seen.add(rel)
    text = _mark_notices(path.read_text(encoding="utf-8"), notices)
    text = _resolve(_strip_comments(text))
    text = re.sub(
        r"^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*$", "", text, flags=re.MULTILINE
    )
    text = _C10_INCLUDE.sub(
        lambda m: f"{_expand(m.group(1), seen, notices)}\n{_FILE_MARK}{rel}", text
    )
    return f"{_FILE_MARK}{rel}\n{text}"


def _refs(text: str) -> frozenset[str]:
    """The identifiers a header's ``text`` names, but for Metal's own."""
    return frozenset(_IDENT.findall(_GLOBAL_METAL.sub(" ", text)))


def _pasted(text: str, names: frozenset[str]) -> frozenset[str]:
    """The ``names`` that a macro in ``text`` may form by token pasting
    (``simd_##NAME`` forms ``simd_sum``), which no identifier spells: those
    with the literal prefix and suffix around the macro's parameters, or all
    of them when a pasted name is parameters alone."""
    found: set[str] = set()
    for line in _logical_lines(text):
        match = re.match(r"\s*#\s*define\s+\w+(\(([^)]*)\))?(.*)$", line, re.DOTALL)
        if match is None or "##" not in line:
            continue
        params = set(re.findall(r"\w+", match.group(2) or "")) | {"__VA_ARGS__"}
        for chain in re.findall(r"\w+(?:\s*##\s*\w+)+", match.group(3)):
            parts = [part.strip() for part in chain.split("##")]
            unknown = [i for i, part in enumerate(parts) if part in params]
            if not unknown:
                found.add("".join(parts))
                continue
            prefix = "".join(parts[: unknown[0]])
            suffix = "".join(parts[unknown[-1] + 1 :])
            if not prefix and not suffix:
                return names
            found.update(
                name
                for name in names
                if name.startswith(prefix)
                and name.endswith(suffix)
                and len(name) > len(prefix) + len(suffix)
            )
    return frozenset(found & names)


def _count(text: str, opening: str, closing: str) -> int:
    """How many more ``opening`` than ``closing`` characters ``text`` has
    outside literals."""
    text = _LITERAL.sub("", text)
    return text.count(opening) - text.count(closing)


def _angle(text: str, i: int, paren: int) -> int:
    """How ``text[i]`` changes the depth of template brackets: not at all
    inside parentheses (``paren``), where ``<`` and ``>`` compare, nor as
    part of ``<<``, ``<=``, ``>=`` or ``->``."""
    c = text[i]
    if paren or c not in "<>":
        return 0
    before, after = text[i - 1 : i], text[i + 1 : i + 2]
    if c == "<":
        return 0 if after in ("<", "=") or before == "<" else 1
    return 0 if after == "=" or before == "-" else -1


def _skip_template_head(text: str) -> str:
    """``text`` after a leading ``template <...>`` parameter list."""
    while (match := re.match(r"\s*template\s*<", text)) is not None:
        depth, paren, i = 1, 0, match.end()
        while i < len(text) and depth:
            paren += {"(": 1, ")": -1}.get(text[i], 0)
            depth += _angle(text, i, paren)
            i += 1
        text = text[i:]
    return text.lstrip()


def _head(text: str) -> str:
    """A declaration's text before its first ``{`` outside parentheses."""
    depth = 0
    for i, c in enumerate(text):
        depth += {"(": 1, ")": -1}.get(c, 0)
        if c == "{" and depth == 0:
            return text[:i]
    return text


def _ends_at_brace(text: str) -> bool:
    """Whether a declaration ends with its closing brace, as a function
    definition does, rather than with a ``;`` after it (a class, an enum, a
    variable's braced initializer)."""
    head = _skip_template_head(_head(text))
    head = _SPECIFIERS.sub("", head)
    if _RECORD.match(head) or re.match(r"(?:typedef|using)\b", head):
        return False
    head = re.sub(r"operator\s*\S+|==|!=|<=|>=", "", head)
    while (stripped := re.sub(r"\([^()]*\)|<[^<>()]*>", "", head)) != head:
        head = stripped
    return "=" not in head


def _strip_trailing_angles(text: str) -> str:
    """``text`` without a trailing ``<...>`` (a specialization's template
    arguments)."""
    text = text.rstrip()
    if not text.endswith(">"):
        return text
    depth, paren = 0, 0
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        paren += {")": 1, "(": -1}.get(c, 0)
        if paren == 0 and c in "<>" and text[i - 1 : i] != "-":
            depth += 1 if c == ">" else -1
            if depth == 0:
                return text[:i].rstrip()
    return text


def _enumerators(body: str) -> frozenset[str]:
    """The enumerators an enum's ``body`` declares."""
    inside = body[body.index("{") + 1 : body.rindex("}")]
    return frozenset(
        m.group(1) for m in re.finditer(r"(?:^|,)\s*([A-Za-z_]\w*)", inside)
    )


def _names(text: str) -> tuple[str, frozenset[str]]:
    """A declaration's kind (``"decl"``, ``"operator"``, ``"keep"``) and the
    names it declares at its namespace's scope: every overload and
    specialization of a function or class shares its unqualified name; an
    unscoped enum declares its enumerators too.  ``"keep"`` marks a
    declaration naming nothing (``using namespace``, ``static_assert``).  A
    declaration whose name this can't read raises :class:`_Unparsed`: left
    out, it could leave its overloads to take its calls."""
    body = _skip_template_head(text)
    if re.match(r"using\s+namespace\b", body):
        return "keep", frozenset()
    if match := re.match(r"using\s+([A-Za-z_]\w*)\s*=", body):
        return "decl", frozenset({match.group(1)})
    if match := re.match(r"using\s+[\w:]*::([A-Za-z_]\w*)\s*;", body):
        return "decl", frozenset({match.group(1)})
    head = _SPECIFIERS.sub("", _head(body))
    while (
        stripped := _SPECIFIERS.sub(
            "",
            re.sub(r"^(?:\[\[.*?\]\]|__attribute__\s*\(\(.*?\)\))\s*", "", head),
        )
    ) != head:
        head = stripped
    if re.match(r"enum\s*(?::\s*[\w:]+\s*)?$", head) and "{" in body:
        # An anonymous enum declares its enumerators alone.
        return "decl", _enumerators(body)
    if match := _RECORD.match(head):
        names = {match.group(2)}
        if match.group(1) == "enum" and "{" in body:
            names |= _enumerators(body)
        if match.group(0).startswith("typedef"):
            names.update(re.findall(r"([A-Za-z_]\w*)\s*;\s*$", body))
        return "decl", frozenset(names)
    if match := re.match(
        r"typedef\b.*?([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*;\s*$", body, re.DOTALL
    ):
        return "decl", frozenset({match.group(1)})
    if re.match(r"static_assert\s*\(", head):
        return "keep", frozenset()
    if match := re.search(r"\boperator\s*(\(\)|\[\]|[^\s(]+)\s*\(", head):
        return "operator", frozenset({f"operator{match.group(1)}"})
    # The first ``(`` outside template arguments past a ``decltype(...)``
    # return type, unless an ``=`` comes first (a variable's initializer).
    depth = paren = 0
    i = 0
    while i < len(head):
        c = head[i]
        if depth and c in "()":
            paren += 1 if c == "(" else -1
        depth += _angle(head, i, paren)
        if depth == 0 and paren == 0 and c in "=(;[":
            before = _strip_trailing_angles(head[:i])
            match = re.search(r"([A-Za-z_]\w*)$", before)
            if c == "(" and match is not None and match.group(1) in _NOT_FUNCTIONS:
                # Past the parenthesized operand of a return type's
                # ``decltype``.
                level, i = 1, i + 1
                while i < len(head) and level:
                    level += {"(": 1, ")": -1}.get(head[i], 0)
                    i += 1
                continue
            if match is None or match.group(1) in _KEYWORDS:
                raise _Unparsed(text[:200])
            return "decl", frozenset({match.group(1)})
        i += 1
    match = re.search(r"([A-Za-z_]\w*)\s*$", _strip_trailing_angles(head))
    if match is None or match.group(1) in _KEYWORDS:
        raise _Unparsed(text[:200])
    return "decl", frozenset({match.group(1)})


def _single(text: str) -> bool:
    """Whether a declaration's ``text`` is one declaration: no ``;`` at
    namespace scope before its end, and its braces closing once, at its end
    (or before a ``;``)."""
    text = _LITERAL.sub(
        "",
        "\n".join(
            line for line in text.split("\n") if not line.lstrip().startswith("#")
        ),
    ).rstrip()
    depth = paren = 0
    for i, c in enumerate(text):
        if c in "{}":
            depth += 1 if c == "{" else -1
            if depth == 0 and text[i + 1 :].strip() not in ("", ";"):
                return False
        elif c in "()":
            paren += 1 if c == "(" else -1
        elif c == ";" and depth == 0 and paren == 0 and text[i + 1 :].strip():
            return False
    return True


def _pieces(text: str) -> list[_Piece]:
    """``text`` split into pieces at namespace scope (:class:`_Piece`).
    Raises :class:`_Unparsed` where the split could be wrong: a line that
    ends a declaration's parentheses without ending the declaration (a
    macro's use without ``;``, a brace on the next line), two declarations
    on a line, a declaration named after a macro (a macro's use)."""
    pieces: list[_Piece] = []
    namespaces: list[str] = []
    decl: list[str] = []
    depth = paren = 0
    cond: list[str] = []
    nesting = 0
    header = ""
    seen: dict[str, list[int]] = {}

    def add(
        text: str,
        kind: str,
        names: frozenset[str] = frozenset(),
        refs: frozenset[str] = frozenset(),
    ) -> None:
        notices = tuple(seen.get(header, ()))
        pieces.append(_Piece(text, tuple(namespaces), kind, names, refs, notices))

    for line in _logical_lines(text):
        stripped = line.strip()
        if stripped.startswith(_FILE_MARK):
            header = stripped[len(_FILE_MARK) :]
            continue
        if stripped.startswith(_NOTICE_MARK):
            seen.setdefault(header, []).append(int(stripped[len(_NOTICE_MARK) :]))
            continue
        match = _DIRECTIVE.match(line)
        if cond:
            cond.append(line)
            if match and match.group(1) in ("if", "ifdef", "ifndef"):
                nesting += 1
            elif match and match.group(1) == "endif":
                nesting -= 1
            if nesting == 0:
                # Kept whole, under the names it defines and declares in any
                # branch: a declaration in it may overload one outside.
                block = "\n".join(cond)
                inner = _pieces(
                    "\n".join(
                        block_line
                        for block_line in _logical_lines(block)
                        if not block_line.lstrip().startswith("#")
                    )
                )
                declared = {name for piece in inner for name in piece.names}
                names = set(re.findall(r"#\s*define\s+(\w+)", block)) | declared
                # One naming nothing (a conditional include or pragma, an
                # assertion) is kept, or kept with what it names.
                add(block, "cond" if names else "keep", frozenset(names), _refs(block))
                cond = []
            continue
        if not decl:
            if not stripped:
                continue
            if match:
                word, rest = match.group(1), match.group(2)
                if word in ("if", "ifdef", "ifndef"):
                    cond, nesting = [line], 1
                    continue
                if word == "define":
                    name = re.match(r"\w+", rest)
                    assert name is not None, line
                    add(
                        line,
                        "define",
                        frozenset({name.group(0)}),
                        _refs(rest[name.end() :]),
                    )
                elif word == "undef":
                    add(line, "undef", frozenset({rest.strip()}))
                else:
                    add(line, "keep")
                continue
            if match := _NAMESPACE.match(stripped):
                if stripped.startswith("inline"):
                    # Its members are the enclosing namespace's too, which
                    # the rebuilt namespaces wouldn't say.
                    raise _Unparsed(line)
                namespaces.append(match.group(1))
                continue
            if stripped in ("}", "};"):
                if not namespaces:
                    raise _Unparsed(line)
                namespaces.pop()
                continue
        decl.append(line)
        if stripped.startswith("#"):
            continue
        depth += _count(line, "{", "}")
        paren += _count(line, "(", ")")
        if depth < 0 or paren < 0:
            raise _Unparsed(line)
        if depth or paren:
            continue
        whole = "\n".join(decl)
        if stripped.endswith(";") or (stripped.endswith("}") and _ends_at_brace(whole)):
            if not _single(whole):
                raise _Unparsed(whole[:200])
            kind, names = _names(whole.strip())
            add(whole, kind, names, _refs(whole))
            decl = []
        elif stripped.endswith(")"):
            raise _Unparsed(whole[:200])
    if decl or cond:
        raise _Unparsed("\n".join(decl or cond)[:200])
    macros = {
        name for piece in pieces if piece.kind == "define" for name in piece.names
    }
    for piece in pieces:
        if piece.kind in ("decl", "operator") and piece.names & macros:
            raise _Unparsed(piece.text[:200])
    return pieces


def prune_source(text: str, used: frozenset[str] | set[str]) -> str:
    """The pieces of ``text`` (header source, expanded and resolved) that the
    identifiers ``used`` reach, in their order, with their namespaces."""
    return _select(_pieces(_strip_comments(text)), used, [])


def _select(
    pieces: list[_Piece], used: frozenset[str] | set[str], notices: list[str]
) -> str:
    """:func:`prune_source` of parsed ``pieces``, the ``notices`` over what
    stays first."""
    by_name: dict[str, list[int]] = {}
    for i, piece in enumerate(pieces):
        for name in piece.names:
            by_name.setdefault(name, []).append(i)
    keep = [piece.kind == "keep" for piece in pieces]
    needed: set[str] = set()
    pending = [name for name in used if name in by_name]
    # Declarations naming nothing in the headers (``static_assert``) are kept
    # when every header name they name is.
    guards = [
        i
        for i, piece in enumerate(pieces)
        if piece.kind == "keep" and piece.refs & by_name.keys()
    ]
    for i in guards:
        keep[i] = False

    def take(i: int) -> None:
        if not keep[i]:
            keep[i] = True
            pending.extend(name for name in pieces[i].refs if name in by_name)
            if pieces[i].kind == "define" and "##" in pieces[i].text:
                pending.extend(_pasted(pieces[i].text, frozenset(by_name)))

    while True:
        while pending:
            name = pending.pop()
            if name in needed:
                continue
            needed.add(name)
            for i in by_name[name]:
                if pieces[i].kind != "undef":
                    take(i)
        # A macro's ``#undef`` goes with its ``#define``; an operator with
        # the types it names; a guard with all it names.
        for i, piece in enumerate(pieces):
            if keep[i]:
                continue
            operator = piece.kind == "operator" or (
                piece.kind == "cond"
                and any(name.startswith("operator") for name in piece.names)
            )
            if (
                piece.kind == "undef"
                and piece.names & needed
                or operator
                and piece.refs & needed - piece.names
                or i in guards
                and piece.refs & by_name.keys() <= needed
            ):
                take(i)
        if not pending:
            break
    # A ``using namespace`` of one of the headers' namespaces goes where no
    # kept declaration declares it; where none does before it, it is
    # declared, empty, first.
    kept_in = {
        piece.namespaces[:n]
        for piece, kept in zip(pieces, keep, strict=True)
        if kept
        for n in range(1, len(piece.namespaces) + 1)
    }
    declared = {
        piece.namespaces[:n]
        for piece in pieces
        for n in range(1, len(piece.namespaces) + 1)
    }
    targets: dict[int, tuple[str, ...]] = {}
    for i, piece in enumerate(pieces):
        match = re.match(r"\s*using\s+namespace\s+([\w:]+)\s*;", piece.text)
        if not (keep[i] and match):
            continue
        named = tuple(match.group(1).strip(":").split("::"))
        # ``::x`` names the global namespace's ``x``, from any namespace.
        start = 0 if match.group(1).startswith("::") else len(piece.namespaces)
        for n in range(start, -1, -1):
            target = piece.namespaces[:n] + named
            if target in declared:
                keep[i] = target in kept_in
                targets[i] = target
                break
    out = [
        notices[n].strip("\n")
        for n in sorted(
            {
                n
                for piece, kept in zip(pieces, keep, strict=True)
                # Code, not the directives a header always keeps.
                if kept and piece.kind != "keep"
                for n in piece.notices
            }
        )
    ]
    current: tuple[str, ...] = ()
    opened: set[tuple[str, ...]] = set()
    included: set[str] = set()
    for i, (piece, kept) in enumerate(zip(pieces, keep, strict=True)):
        if not kept:
            continue
        if (target := targets.get(i)) is not None and target not in opened:
            out.extend("}" for _ in current)
            out.extend(f"namespace {name} {{" for name in target)
            out.extend("}" for _ in target)
            current = ()
            opened.update(target[:n] for n in range(1, len(target) + 1))
        if re.fullmatch(r"\s*#\s*include\s*<[^>]+>\s*", piece.text):
            # A system header's second include adds nothing.
            if piece.text.strip() in included:
                continue
            included.add(piece.text.strip())
        common = 0
        while (
            common < min(len(current), len(piece.namespaces))
            and current[common] == piece.namespaces[common]
        ):
            common += 1
        out.extend("}" for _ in current[common:])
        out.extend(f"namespace {name} {{" for name in piece.namespaces[common:])
        current = piece.namespaces
        opened.update(current[:n] for n in range(1, len(current) + 1))
        out.append(piece.text.strip("\n"))
    out.extend("}" for _ in current)
    return "\n".join(out)


class _Headers(NamedTuple):
    """Headers expanded and split (:func:`_headers`)."""

    text: str
    pieces: list[_Piece]
    names: frozenset[str]
    notices: list[str]


@functools.cache
def _headers(includes: tuple[str, ...], version: str) -> _Headers | None:
    """The headers ``includes`` expanded and resolved (:func:`_expand`), their
    pieces, the names those declare and the headers' notices; None when they
    hold a construct the split doesn't read.  ``version`` (torch's) keys the
    cache."""
    seen: set[str] = set()
    notices: list[str] = []
    text = "\n".join(_expand(rel, seen, notices) for rel in includes)
    try:
        pieces = _pieces(text)
    except _Unparsed:
        return None
    names = frozenset(name for piece in pieces for name in piece.names)
    return _Headers(text, pieces, names, notices)


@functools.lru_cache(maxsize=512)
def _pruned(includes: tuple[str, ...], used: frozenset[str], version: str) -> str:
    """The pieces of the headers ``includes`` that ``used`` reaches."""
    headers = _headers(includes, version)
    assert headers is not None
    return _select(headers.pieces, used, headers.notices)


def prune(msl: str) -> str:
    """``msl`` with its ``#include <c10/...>`` lines replaced, at the first,
    by the c10 declarations the rest of it uses (module docstring); ``msl``
    itself where the headers hold a construct the split doesn't read."""
    includes = tuple(dict.fromkeys(_C10_INCLUDE.findall(msl)))
    if not includes:
        return msl
    version = torch.__version__
    headers = _headers(includes, version)
    if headers is None:
        # Headers the split doesn't read stay whole.
        return msl
    declared = headers.names
    first = _C10_INCLUDE.search(msl)
    assert first is not None
    before, after = msl[: first.start()], _C10_INCLUDE.sub("", msl[first.end() :])
    rest = before + after
    used = frozenset(_IDENT.findall(_KERNEL_METAL.sub(" ", rest))) & declared
    used |= _pasted(rest, declared)
    # Without the system includes the kernel has already made.
    made = set(re.findall(r"^#include <[^>]+>$", before, re.MULTILINE))
    preamble = "\n".join(
        line
        for line in _pruned(includes, used, version).split("\n")
        if line not in made
    )
    return before + preamble + after
