#!/usr/bin/env python3
"""The examples of a pull request description: each one's program, and the code
generated from it before and after the change, with `# <-` on the lines that
differ (Step 4 of the skill).

An examples file is a Python module with a dict EXAMPLES of name -> dict:

    EXAMPLES = {
        "embedding_lookup": dict(
            title="An embedding lookup: the table read at the value",
            fn=embedding_lookup,          # a PyTorch function or a helion.Kernel
            inputs=lambda: (torch.randint(0, 1000, (64, 32), device="cuda"),
                            torch.randn(1000, 256, device="cuda")),
            # optional:
            backend="cute",               # helion.backend of a PyTorch function
                                          # (default: helion.inductor's); a
                                          # helion.Kernel takes its decorator's
            inductor_config={"assert_indirect_indexing": False},
            config=helion.Config(...),    # for a helion.Kernel (default: its default config)
            notes={"after": {"in1[tmp1, tile_x]": "w's view at the value"}},
            # for sweep_variants.py: variants=..., fired=...
        ),
    }

A PyTorch function compiles with helion.inductor (torch.compile, cuda_backend
"helion", autotune effort none) and shows the Helion kernels it generates, in
full; a helion.Kernel shows the Triton or CuTe code it generates, abridged to
the changed lines and three lines around them. Every changed line gets a
`# <-`; `notes` puts an explanation on its own line above the first line
holding each substring. Explain the bare ones by hand.

    PYTHONPATH=<tree before> python pr_examples.py compile EXAMPLES.py before.json
    PYTHONPATH=<tree after>  python pr_examples.py compile EXAMPLES.py after.json
    PYTHONPATH=<tree after>  python pr_examples.py markdown EXAMPLES.py \
        before.json after.json out.md
"""

from __future__ import annotations

import difflib
import importlib.util
import inspect
import json
import pathlib
import re
import sys
import traceback
from typing import Any

import torch
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code

import helion
import helion.inductor  # noqa: F401

CONTEXT = 3  # lines of backend code kept around each change
RUN_HEAD, RUN_TAIL = 6, 3  # changed lines a long run keeps at each end
# A run of changed backend lines longer than this keeps only its RUN_HEAD first
# and RUN_TAIL last lines; a shorter one is shown whole, as the CONTEXT lines
# around those would cover it anyway.
LONG_RUN = RUN_HEAD + RUN_TAIL + 2 * CONTEXT


def load_examples(path: str) -> dict[str, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("pr_examples_input", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXAMPLES


def helion_sources(code: str) -> list[str]:
    """The Helion kernels in an Inductor output module, from their decorator
    on, without the autotune effort the examples compile at."""
    sources = []
    for src in re.findall(r'_source = """\\\n(.*?)"""', code, re.DOTALL):
        lines = src.rstrip("\n").split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("@helion"))
        sources.append("\n".join(lines[start:]).replace(', autotune_effort="none"', ""))
    return sources


def helion_options(example: dict[str, Any]) -> dict[str, Any]:
    """torch.compile options for helion.inductor: `example`'s backend if it
    names one, else helion.inductor's default (helion/inductor/config.py)."""
    options = {"cuda_backend": "helion", "helion.autotune_effort": "none"}
    if "backend" in example:
        options["helion.backend"] = example["backend"]
    return options


def compile_one(example: dict[str, Any]) -> dict[str, Any]:
    fn = example["fn"]
    torch.manual_seed(0)
    args = example["inputs"]()
    if isinstance(fn, helion.Kernel):
        if "backend" in example:
            raise ValueError("a helion.Kernel takes its backend from its decorator")
        bound = fn.bind(args)
        config = example.get("config") or bound.config_spec.default_config()
        # Without Helion's source-map comments (# src[file:line]: ...).
        code = bound.to_code(config, output_origin_lines=False)
        return {"kind": "kernel", "code": [code]}
    torch._dynamo.reset()
    with inductor_config.patch(example.get("inductor_config", {})):
        compiled = torch.compile(
            fn, fullgraph=True, dynamic=False, options=helion_options(example)
        )
        _, codes = run_and_get_code(compiled, *args)
    return {"kind": "inductor", "code": helion_sources("\n".join(codes))}


def compile_all(examples_path: str, out_path: str) -> None:
    results = {}
    for name, example in load_examples(examples_path).items():
        try:
            results[name] = compile_one(example)
        except Exception:
            results[name] = {"error": traceback.format_exc()[-2000:]}
        print(name, "ERROR" if "error" in results[name] else "ok", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


# Numbered names: helion.inductor's temporaries and kernel arguments (tmp3,
# acc, acc1, in0, out1), and the backend code's numbered variables (v_0,
# load_1, sum_1_acc, offsets_0_lane2, _jam_vec_1) and constants
# (_BLOCK_SIZE_2, _RDIM_SIZE_1). Not dtypes: float32 has no _ before its
# digits.
NAME = re.compile(
    r"\b(?:tmp\d+|acc\d*|in\d+|out\d+)\b"
    r"|(?<![\w.])_jam_vec(?:_\d+)?\b"
    r"|(?<![\w.])_[A-Z][A-Z0-9_]*?_\d+\b"
    r"|(?<![\w.])_?[a-z][a-z0-9_]*?_\d+(?:_[a-z][a-z0-9_]*)?\b"
)


def changed_lines(lines: list[str], other: list[str]) -> set[int]:
    """The lines of `lines` that `other` does not have. A line counts as kept
    when it matches up to its numbered names and they rename consistently: a
    temporary the change inserts renumbers the ones after it, which is no
    change, while tmp2 = tmp0 - tmp1 against tmp2 = tmp1 - tmp0, or a store to
    out1 rather than out0, is. The renaming starts from the names that
    aligned lines assign (tmp2 = ... against tmp3 = ...), even where the
    lines changed or a temporary was inserted among them, so that a swap of
    operands whose loads changed still counts. Each kernel numbers its names
    afresh, so the renaming is per pair of kernels (counted by their @helion
    lines)."""
    matcher = difflib.SequenceMatcher(
        a=[NAME.sub("#", x) for x in lines],
        b=[NAME.sub("#", x) for x in other],
        autojunk=False,
    )

    def kernel_index(code: list[str]) -> list[int]:
        index, count = [], 0
        for line in code:
            count += line.startswith("@helion")
            index.append(count)
        return index

    def target(line: str) -> str | None:
        assigned = re.match(r"\s*(\w+) = ", line)
        if assigned and NAME.fullmatch(assigned[1]):
            return assigned[1]
        return None

    kernel_a, kernel_b = kernel_index(lines), kernel_index(other)
    changed = set(range(len(lines)))
    renaming: dict[tuple[int, int], tuple[dict[str, str], dict[str, str]]] = {}

    def pairs(i1: int, i2: int, j1: int, j2: int) -> list[tuple[int, int]]:
        """Aligned lines of a changed block: in order where the block keeps
        its length, else each line with the most alike one after the last
        pair (a temporary inserted among changed loads)."""
        if i2 - i1 == j2 - j1:
            return list(zip(range(i1, i2), range(j1, j2), strict=True))
        found, j = [], j1
        for i in range(i1, i2):
            alike = [
                (difflib.SequenceMatcher(a=matcher.a[i], b=matcher.b[k]).ratio(), k)
                for k in range(j, j2)
            ]
            # The most alike, the first of equals.
            ratio, k = max(alike, key=lambda t: (t[0], -t[1]), default=(0.0, j2))
            if ratio >= 0.6:
                found.append((i, k))
                j = k + 1
        return found

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert" or tag == "delete":
            continue
        for i, j in pairs(i1, i2, j1, j2):
            x, y = target(lines[i]), target(other[j])
            fwd, bwd = renaming.setdefault((kernel_a[i], kernel_b[j]), ({}, {}))
            if x and y and fwd.get(x, y) == y and bwd.get(y, x) == x:
                fwd[x], bwd[y] = y, x
    for block in matcher.get_matching_blocks():
        for i, j in zip(
            range(block.a, block.a + block.size),
            range(block.b, block.b + block.size),
            strict=True,
        ):
            names, other_names = NAME.findall(lines[i]), NAME.findall(other[j])
            if len(names) != len(other_names):
                continue
            forward, backward = renaming.get((kernel_a[i], kernel_b[j]), ({}, {}))
            # Extended pair by pair, so that two pairs of one line conflict too.
            fwd, bwd = dict(forward), dict(backward)
            if all(
                fwd.setdefault(x, y) == y and bwd.setdefault(y, x) == x
                for x, y in zip(names, other_names, strict=True)
            ):
                changed.discard(i)
                renaming[kernel_a[i], kernel_b[j]] = (fwd, bwd)
    return changed


def annotate(code: str, other: str, notes: dict[str, str], abridge: bool) -> str:
    """`code` with `# <-` on each line that differs from `other`, a note on its
    own line above the line it explains, and, if `abridge`, only the changed
    or explained lines, and where the other side has lines this one lacks,
    with CONTEXT lines around them (of a run of more than LONG_RUN changed
    lines, its first RUN_HEAD and last RUN_TAIL)."""
    lines = code.rstrip("\n").split("\n")
    changed = changed_lines(lines, other.rstrip("\n").split("\n"))
    noted: dict[int, str] = {}
    for key, note in notes.items():
        hit = next((i for i, line in enumerate(lines) if key in line), None)
        if hit is None:
            raise SystemExit(f"note {key!r} matches no line")
        noted[hit] = note
    blocks = []
    for i, line in enumerate(lines):
        block = [f"{line}  # <-" if i in changed else line]
        if i in noted:
            indent = line[: len(line) - len(line.lstrip())]
            block.insert(0, f"{indent}# {noted[i]}")
        blocks.append(block)
    keep: list[int] = list(range(len(lines)))
    if abridge:
        shown = set(noted)
        # Where the other side has lines this one lacks: the lines around.
        matcher = difflib.SequenceMatcher(
            a=[NAME.sub("#", x) for x in lines],
            b=[NAME.sub("#", x) for x in other.rstrip("\n").split("\n")],
            autojunk=False,
        )
        for tag, i1, _, _, _ in matcher.get_opcodes():
            if tag == "insert":
                shown |= {min(i1, len(lines) - 1), max(i1 - 1, 0)}
        run: list[int] = []
        for i in [*sorted(changed), None]:
            if run and (i is None or i != run[-1] + 1):
                long = len(run) > LONG_RUN
                shown |= set(run[:RUN_HEAD] + run[-RUN_TAIL:]) if long else set(run)
                run = []
            if i is not None:
                run.append(i)
        keep = sorted(
            {j for i in shown for j in range(i - CONTEXT, i + CONTEXT + 1)}
            & set(range(len(lines)))
        )
    out, last = [], -1
    for i in keep:
        if i != last + 1:
            out.append("...")
        out += blocks[i]
        last = i
    if keep and last != len(lines) - 1:
        out.append("...")
    return "\n".join(out)


def fence(code: str) -> str:
    return f"```python\n{code}\n```"


def markdown(
    examples_path: str, before_path: str, after_path: str, out_path: str
) -> None:
    examples = load_examples(examples_path)
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)
    parts = []
    for name, example in examples.items():
        b, a = before[name], after[name]
        for side, result in (("before", b), ("after", a)):
            if "error" in result:
                raise SystemExit(f"{name} failed to compile {side}:\n{result['error']}")
        abridge = a["kind"] == "kernel"
        fn = example["fn"]
        program = inspect.getsource(
            fn.fn if isinstance(fn, helion.Kernel) else fn
        ).rstrip()
        notes = example.get("notes", {})
        code_b, code_a = "\n\n".join(b["code"]), "\n\n".join(a["code"])
        if code_b == code_a:
            # An example the change does not reach: no fences of the same code.
            print(f"{name}: the same code before and after, left out", file=sys.stderr)
            continue
        parts += [
            f"**{example.get('title', name)}**",
            "",
            fence(program),
            "",
            "Before this pull request:",
            "",
            fence(annotate(code_b, code_a, notes.get("before", {}), abridge)),
            "",
            "After:",
            "",
            fence(annotate(code_a, code_b, notes.get("after", {}), abridge)),
            "",
        ]
    pathlib.Path(out_path).write_text("\n".join(parts))


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "compile":
        compile_all(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6 and sys.argv[1] == "markdown":
        markdown(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
