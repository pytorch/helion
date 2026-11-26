---
name: fix-ci-main
description: Fix a CI job failing on pytorch/helion's main branch, and, when only main's CI runs the failing test, prove the fix by running main's CI on its pull request (the ciflow/main tag). Auto-activate when the user reports a failing CI job or test on main.
---

# Fix a CI failure on main

1. **Find it.** The failing tests, and the same job in earlier main runs:
   failing since one commit is a regression, now and then a race or a
   timeout. Main's runs cancel each other, but a cancelled run's finished
   jobs still count. A worker that crashed with no traceback may have hit
   the timeout, the OOM killer or a native crash: confirm which.
2. **Check for a fix in flight** (`gh pr list --search`). If there is one,
   verify it on origin/main and report instead of writing another.
3. **Fix.** Reproduce with the job's settings, fix in a stack of its own off
   origin/main, show the failing case failing before and passing after, run
   the lanes, autoreview, commit.
4. **Main-only failure.** Pull requests skip the slow tier and the
   `"pr": false` jobs, so the fix's pull request can pass without running the
   failing test. Then run Tests on it as on main: the `ciflow/main` label
   (the bot moves the tag on each update), or the tag pushed by hand,
   `git push -f origin <head sha>:refs/tags/ciflow/main/<PR number>` (again
   after each update). Either starts a full main run, so both are the
   user's call. The run shows in the pull request's checks.
5. **Confirm.** Creating the pull request is the user's call. The jobs that
   failed on main must pass in the ciflow/main run. Then remove the label,
   or delete the tag (`git push origin :refs/tags/ciflow/main/<PR number>`).
