---
name: pr-review
description: Review Helion pull requests for code quality, test coverage, kernel correctness, and performance. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# Helion PR Review Skill

Review Helion pull requests focusing on what CI cannot check: code quality, test coverage adequacy, kernel correctness, Triton codegen quality, and performance implications. Linting, formatting (ruff), and type checking (pyrefly) are handled by CI.

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, **do not perform a review**. Instead, ask the user what they would like to review:

> What would you like me to review?
> - A PR number or URL (e.g., `/pr-review 12345`)
> - A local branch (e.g., `/pr-review branch`)

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/pytorch/helion/pull/12345
```

For a detailed review with line-by-line specific comments:

```
/pr-review 12345 detailed
```

Use `gh` CLI to fetch PR data:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits

# Get the diff
gh pr diff <PR_NUMBER>

# Get PR comments
gh pr view <PR_NUMBER> --json comments,reviews
```

### Local Branch Mode

Review changes in the current branch that are not in `main`:

```
/pr-review branch
/pr-review branch detailed
```

Use git commands to get branch changes:

```bash
# Get current branch name
git branch --show-current

# Get list of changed files compared to main
git diff --name-only main...HEAD

# Get full diff compared to main
git diff main...HEAD

# Get commit log for the branch
git log main..HEAD --oneline

# Get diff stats (files changed, insertions, deletions)
git diff --stat main...HEAD
```

For local branch reviews:
- The "Summary" should describe what the branch changes accomplish based on commit messages and diff
- Use the current branch name in the review header instead of a PR number
- All other review criteria apply the same as PR reviews

### GitHub Actions Mode

When invoked via `@claude /pr-review` on a GitHub PR, the action pre-fetches PR
metadata and injects it into the prompt. Detect this mode by the presence of
`<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in the prompt.

The prompt already contains:
- PR metadata (title, author, branch names, additions/deletions, file count)
- PR body/description
- All comments and review comments (with file/line references)
- List of changed files with paths and change types

Use git commands to get the diff and commit history. The base branch name is in the
prompt context (look for `PR Branch: <head> -> <base>` or the `baseBranch` field).

```bash
# Get the full diff against the base branch
git diff origin/<baseBranch>...HEAD

# Get diff stats
git diff --stat origin/<baseBranch>...HEAD

# Get commit history for this PR
git log origin/<baseBranch>..HEAD --oneline

# If the base branch ref is not available, fetch it first
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI commands in this mode -- only git commands are available.
All PR metadata, comments, and reviews are already in the prompt context;
only the diff and commit log need to be fetched via git.

## Review Philosophy

Helion is a DSL that compiles to Triton kernels. A single line of code can have deep implications: incorrect tile indexing causes silent data corruption, improper dtype handling breaks precision, missing masking creates out-of-bounds memory access, and poor configuration choices cause dramatic performance regressions. **Treat every line as potentially load-bearing.**

Do not skim. Do not summarize the diff and move on. Read every changed line and ask: *does this interact with existing Helion infrastructure that the author may not know about?* When uncertain, **investigate** — spawn a sub-agent to read the surrounding code, the infrastructure the PR should be using, or the tests that should exist. The cost of a false negative (missing a real issue) is much higher than the cost of investigation.

## Review Workflow

### Step 1: Fetch PR Information

**Local CLI mode**: Use `gh` commands to get PR metadata, changed files, full diff,
existing comments/reviews, and associated issue information.

**Local Branch mode**: Use `git diff` and `git log` against `main` as shown in the
Local Branch Mode section above.

**GitHub Actions mode**: PR metadata, comments, and reviews are already in the prompt.
Use `git diff origin/<baseBranch>...HEAD` for the full diff and
`git log origin/<baseBranch>..HEAD --oneline` for the commit log.

### Step 2: Understand Context

Before reviewing, build understanding of what the PR touches and why:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (kernel code, compiler, runtime, tests, examples, docs)
3. Note the scope of changes (files affected, lines changed)
4. **Spawn sub-agents to read the unchanged code surrounding each changed file.** The diff alone is not enough — you need to understand the existing patterns, infrastructure, and invariants in the files being modified.

### Step 3: Deep Review — Line-by-Line with Investigation

This is the core of the review. Go through **every changed line** in the diff and evaluate it against the review checklist in [review-checklist.md](review-checklist.md).

**How to use sub-agents during review:**

When you encounter a changed line that touches a checklist area, **spawn a sub-agent** to investigate whether the checklist item applies. For example:

- A PR adds a new kernel → spawn a sub-agent to check: Does it use proper `hl.tile` patterns? Does it handle dtype promotion correctly? Does it have tests with `code_and_output`?
- A PR modifies the compiler → spawn a sub-agent to check: Are type propagation patterns correct? Does it handle edge cases (zero-size tensors, scalar inputs)?
- A PR adds a new example → spawn a sub-agent to check: Does it define `main()`? Does it follow Helion import conventions? Is there a corresponding test in `test_examples.py`?
- A PR modifies autotuner code → spawn a sub-agent to check: Does it handle all config options? Are search bounds reasonable?

**Spawn sub-agents in parallel** for independent investigation areas. A typical review of a medium PR should spawn 3-8 sub-agents.

**Checklist areas** (see [review-checklist.md](review-checklist.md) for full details):
- Kernel correctness (tile indexing, dtype handling, masking, reductions)
- Helion DSL patterns (`hl.tile`, `hl.zeros`, `hl.dot`, `hl.atomic_add`)
- Configuration handling (block_sizes, indexing strategies, pid_type)
- Type propagation and dtype promotion
- Generated Triton code quality
- Testing adequacy
- Security considerations
- Performance implications

### Step 4: Check Backward Compatibility

Evaluate BC implications:
- Changes to `@helion.kernel` decorator behavior
- Changes to `helion.Config` options
- Changes to `helion.language` (hl) API
- Changes to autotuning behavior or defaults

For non-trivial BC questions, spawn a sub-agent to search for existing usage patterns.

### Step 5: Formulate Review

Structure your review with actionable feedback organized by category. Every finding should be traceable to a specific line in the diff and a specific checklist item.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Code Quality | Abstractions, patterns, complexity, Python 3.10+ style | [review-checklist.md](review-checklist.md) |
| Kernel Correctness | Tile indexing, dtype handling, masking, reductions | [review-checklist.md](review-checklist.md) |
| DSL Patterns | `hl.tile`, `hl.zeros`, `hl.dot`, `hl.register_block_size` | [review-checklist.md](review-checklist.md) |
| Configuration | block_sizes, indexing, pid_type, num_warps, num_stages | [review-checklist.md](review-checklist.md) |
| Type Propagation | dtype promotion, autocast handling, accumulator types | [review-checklist.md](review-checklist.md) |
| Triton Codegen | Generated code quality, masking, memory access patterns | [review-checklist.md](review-checklist.md) |
| Autotuner | Search space, config validation, performance | [review-checklist.md](review-checklist.md) |
| Testing | pytest patterns, `code_and_output`, golden files, device requirements | [review-checklist.md](review-checklist.md) |
| Examples | `main()` function, import patterns, documentation | [review-checklist.md](review-checklist.md) |
| Security | Input validation, memory safety | [review-checklist.md](review-checklist.md) |
| Performance | Tile sizes, memory access patterns, L2 cache usage | [review-checklist.md](review-checklist.md) |

## Output Format

Structure your review as follows. **Omit sections where you have no findings** — don't write "No concerns" for every empty section. Only include sections with actual observations.

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Code Quality
[Issues and suggestions]

### Kernel Correctness
[Tile indexing, dtype, masking, reduction issues]

### Helion Infrastructure
[DSL patterns, config handling, type propagation issues]

### Testing
[Testing adequacy findings — missing tests, inadequate coverage, etc.]

### Security
[Issues if any]

### Performance
[Performance concerns if any]

### Backward Compatibility
[BC concerns if any]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification for recommendation]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

**Do not repeat observations already made in other sections.** This section is for additional file-specific feedback that doesn't fit into the categorized sections above.

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `helion/runtime/kernel.py:142` - Consider extracting this logic into a helper
- `test/test_matmul.py:100-105` - Missing test for transposed input case
- `examples/softmax.py:78` - This could benefit from a reduction_loops config
```

## Key Principles

1. **Investigate, don't guess** - When uncertain whether a checklist item applies, spawn a sub-agent to read the relevant infrastructure code. A reviewer who guesses wrong provides negative value.
2. **Every line matters** - A single incorrect tile index, a missing dtype cast, a wrong reduction pattern — each of these is a real bug. Do not skip lines.
3. **No repetition** - Each observation appears in exactly one section.
4. **Focus on what CI cannot check** - Don't comment on formatting (ruff handles it) or type errors (pyrefly handles it)
5. **Be specific** - Reference file paths and line numbers. Every finding should point to a concrete line in the diff.
6. **Be actionable** - Provide concrete suggestions with the right pattern to use, not vague concerns.
7. **Be proportionate** - Minor issues shouldn't block, but note them
8. **Assume competence** - The author knows Helion; explain only non-obvious context.

## Files to Reference

When reviewing, consult these project files for context. **Spawn sub-agents to read these** rather than relying on memory:

- `AGENTS.md` / `CLAUDE.md` - Coding style and development guidelines
- `CONTRIBUTING.md` - PR requirements and coding standards
- `helion/_testing.py` - Test patterns and utilities (`TestCase`, `code_and_output`, `check_example`)
- `helion/language/__init__.py` - DSL API (`hl.tile`, `hl.zeros`, `hl.dot`, etc.)
- `helion/runtime/config.py` - Configuration options and validation
- `helion/runtime/kernel.py` - Kernel decorator and compilation
- `helion/autotuner/` - Autotuning infrastructure
- `examples/` - Canonical example patterns (each must define `main()`)
- `test/` - Test patterns and golden file conventions
