---
name: fix-pr
description: Address CI failures and unresolved review comments on a Helion pull request. Auto-activate when the user mentions a URL like https://github.com/pytorch/helion/pull/<number>.
---

# Fix a Helion pull request

Goal: bring a PR green by fixing CI failures and addressing unresolved review comments. Leave the fixes **uncommitted and unstaged** — the user handles committing and updating the PR.

## 1. Identify the target PR

Run `git log -n1` to see the current commit. The local commit must correspond to the PR being fixed.

Resolve the PR number using whichever sources are available:

- **User-provided URL** (e.g. `https://github.com/pytorch/helion/pull/1234`): extract the PR number directly.
- **Commit message stack-info line**: look for a line of the form
  ```
  stack-info: PR: https://github.com/pytorch/helion/pull/<number>, branch: ...
  ```
  Use the URL on that line. Ignore the `branch:` value — do **not** run any `checkout`, `branch`, or `switch` commands.
- **Both present**: if the user-provided URL and the stack-info URL disagree, abort with an error.
- **Neither present**: abort with an error explaining that the PR could not be identified.

Then fetch the PR metadata with `gh pr view <number> --repo pytorch/helion --json title,body,...`. **Verify the PR title matches the local commit's subject line.** If they differ, abort with an error — the local commit does not match the PR.

## 2. Fix CI failures

Use `gh` to list the PR's check runs and pull failure logs:

```
gh pr checks <number> --repo pytorch/helion
gh run view <run-id> --log-failed --repo pytorch/helion
```

For each failing check:

- **Infra failures** (most commonly `CUDA Compute Check`, runner provisioning errors, transient network issues): do not attempt to fix. Note them and report at the end.
- **Real failures** (lint, type-check, test failures, build errors): read the logs, locate the root cause, and fix it in the working tree. Re-run the relevant check locally when feasible (e.g. `./lint.sh`, `pytest <file>::<test>`) to confirm the fix.

## 3. Address unresolved review comments

Fetch review comments and resolve any that haven't been addressed:

```
gh api repos/pytorch/helion/pulls/<number>/comments
gh api repos/pytorch/helion/pulls/<number>/reviews
gh pr view <number> --repo pytorch/helion --comments
```

For each unresolved comment, apply the requested change in the working tree. Skip comments that are already resolved, are non-actionable (praise, questions answered in thread), or that the author explicitly waved off.

## 4. Wrap up

- Leave all fixes **uncommitted and unstaged**. Do not run `git add`, `git commit`, `git push`, or any rebase/checkout commands.
- Report a short summary covering:
  - PR identified (number + title)
  - CI failures fixed (with file:line where useful)
  - Review comments addressed
  - Any infra failures skipped (so the user knows to retry them)
  - Anything that could not be fixed and why
