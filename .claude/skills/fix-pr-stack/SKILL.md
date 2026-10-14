---
name: fix-pr-stack
description: Address CI failures and unresolved review comments on every PR in a stack-pr stack, amending each fix into its PR's commit, then rebase the stack onto origin/main and resolve conflicts. Use when the user asks to fix a stack of Helion PRs.
---

# Fix a stack of Helion pull requests

Goal: bring every PR in a stack-pr stack green, with each fix amended into the commit for its PR, then rebase the stack onto `origin/main`. The fix-pr skill (`.claude/skills/fix-pr/SKILL.md`) defines how to fix a single PR; this skill runs it on each commit and handles the history rewriting. This skill authorizes `git commit --amend` and `git rebase`; do not run `git push` or `stack-pr submit` unless the user asks.

## 1. Map the stack

The working tree must be clean: if `git status` shows changes, stop and ask the user, since they would otherwise end up in the wrong commit. HEAD should be the top commit of the stack.

```
git rev-parse HEAD   # record it: the rollback point for the report
base=$(git merge-base origin/main HEAD)
git log --reverse --format='%h%x09%(trailers:key=stack-info,valueonly,separator=)%x09%s' "$base"..HEAD
```

Each line is a commit (bottom first), its `stack-info` PR URL, and its subject. For each PR, run `gh pr view <number> --repo pytorch/helion --json title,state,headRefOid`:

- Leave commits unchanged if their PR is `MERGED` or `CLOSED`, or if they have no stack-info (not submitted yet).
- If `headRefOid` differs from the local sha, the commit changed locally after the last `stack-pr submit`, so its CI results may be stale. Reproduce those failures locally before fixing them. Do this check now; the shas change once the rebase starts.

## 2. Fix each commit, bottom-up

Start an interactive rebase that stops after every commit (a `break` after each `pick`, so no editor is needed):

```
GIT_SEQUENCE_EDITOR="sed -i '/^pick /a break'" git rebase -i "$base"
```

At each stop, HEAD is the commit for one PR:

1. Follow steps 1–3 of the fix-pr skill for HEAD: identify the PR from stack-info and check its title, fix CI failures, and address unresolved review comments. Local test runs here exercise this commit's code, which is what CI tested. Let them finish before continuing; the rebase rewrites the working tree.
2. If anything was fixed, amend it into the commit, keeping the message and its stack-info line:
   ```
   git add -u   # plus any new files the fix created; don't stage scratch files
   git commit --amend --no-edit
   ```
3. Move to the next commit with `GIT_EDITOR=true git rebase --continue`.

CI on a PR tests it together with every commit below it. A failure that also shows up on a lower PR belongs to the lowest PR where it appears: fix it there, and when you reach the higher PRs, confirm locally that the fix carried through instead of fixing it again.

If replaying the next commit conflicts with a fix you just amended, resolve the conflict keeping both intents, `git add` the files, and run `GIT_EDITOR=true git rebase --continue`; the rebase then stops at that commit's `break` as usual. Never `git rebase --skip` here, since that drops a PR's commit.

## 3. Rebase onto origin/main

```
git fetch origin main
git rebase origin/main
```

For each conflict, resolve it keeping both main's change and the PR's intent, `git add` the files, and run `GIT_EDITOR=true git rebase --continue`. The only commits you may `git rebase --skip` are ones whose PR is already `MERGED`; git usually drops those automatically.

Then check the result:

```
git log --reverse --format='%h%x09%(trailers:key=stack-info,valueonly,separator=)%x09%s' origin/main..HEAD
git range-diff "$base"..<orig-head> origin/main..HEAD
```

Every unmerged PR from step 1 must appear exactly once, in the same order, with its stack-info line intact.

Run `./lint.sh` and the tests the stack touches at HEAD, because main may have changed code the stack depends on. Fix any breakage in the commit that introduced the broken code:

```
git commit --fixup=<sha>
git rebase --autosquash origin/main
```

## 4. Wrap up

- Do not run `git push` or `stack-pr submit` unless the user asks; `stack-pr submit` updates all the PRs.
- If a rebase goes badly wrong, `git rebase --abort` returns to the state before that rebase.
- Report:
  - The original HEAD sha (roll back with `git reset --hard <sha>`) and the new HEAD sha
  - Per PR (number + title): CI failures fixed, review comments addressed, infra failures skipped, and anything that could not be fixed and why
  - Stale-CI warnings from step 1
  - Each rebase conflict and how it was resolved, plus any commits dropped as already merged
  - Lint and test results at the new HEAD
