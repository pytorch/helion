---
description: Verify changes to the Helion CuTe backend by running both test suites, lint, and a code review subagent — then fix any issues found.
---

Run all four checks, then fix any issues:

1. **Lint** — `./lint.sh` (Ruff + Pyrefly).
2. **Default test suite** — `pytest test/ -n4 -q --tb=line`.
3. **CuTe test suite** — `HELION_BACKEND=cute pytest test/ -n4 -q --tb=line`.
4. **Code review** — spawn an Opus-class `general-purpose` subagent with the diff (`git diff HEAD~` covers staged + uncommitted, or pick the right scope for the situation).

**Parallelism:** Run lint and the code-review subagent in parallel with the test suites is fine, but **never run the two pytest suites simultaneously** — both use `-n4` and would put 8 workers on the GPU, causing OOMs and flaky failures. Run pytest suites sequentially.

**Review subagent prompt** (use `subagent_type: general-purpose`, `model: opus`):

> Review `git diff HEAD~` in `/data/users/jansel/helion`. Focus on:
> - **Correctness:** wrong logic, missed edge cases, broken invariants.
> - **Simplification:** redundant code, layers that could collapse, premature abstractions.
> - **Hacky things:** ad-hoc workarounds, type-casts that hide bugs, magic constants without comments, stub APIs that aren't really plumbed.
> - **Disabled / regressed coverage:** tests that were deleted or `@skip`ed without an equivalent replacement; functionality that silently lost a code path (e.g. an `if` branch removed without checking callers).
>
> Report concrete `file:line` issues. Under 400 words. Say "LGTM" plainly if clean.

If any check fails, fix the underlying issue (don't suppress it) and re-run only that check. Repeat until all four are clean.

Final report:

| Check | Result |
| ----- | ------ |
| ./lint.sh | clean / N issues fixed |
| pytest (default) | <pass>/<skip> |
| pytest (cute) | <pass>/<skip>/<xfail> |
| code review | LGTM / <N> nits applied |

Notes:
- Don't pass `HELION_AUTOTUNE_EFFORT=none` to a full suite run; it changes execution paths and breaks tests. Use it only for targeted local iteration.
- Use `pytest -ra` if a skip count looks suspicious — it prints skip reasons.
