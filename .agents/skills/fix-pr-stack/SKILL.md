---
name: fix-pr-stack
description: Bring a submitted stack-pr stack green on GitHub — map the stack, fix CI failures and review comments bottom-up by amending each pull request's commit, rebase onto origin/main when needed, resubmit with stack-pr, watch CI, and repeat until every pull request is green or the loop is stuck. Use when the user asks to fix, update, or babysit a stack of Helion pull requests.
---

# Fix a stack of Helion pull requests

Goal: every pull request in the stack green on GitHub, each fix amended into
the commit of its pull request. The loop below runs without asking in between,
until the stack is green or the loop is stuck; the report at the end says what
happened.

This skill authorizes (within `AGENTS.md`, Working Autonomously): amending,
`--fixup`/`--autosquash` and rebasing the stack's commits, `stack-pr submit` to
update the existing pull requests, and re-running failed CI jobs. It does not
authorize squashing or reordering pull requests, landing, or pushing outside
`stack-pr submit`. Above all, it does not authorize creating pull requests:
a submit may only cover commits that already have a `stack-info:` line, so
commits without one (new work above the stack, a new commit at the bottom)
are left out of the submit or the loop stops for the user.

## 0. GitHub access

```
gh auth status
timeout 30 gh api repos/pytorch/helion/pulls?per_page=1 --jq '.[0].number'
```

Some sessions on Meta devservers cannot reach `github.com`. Then
`meta github.pull-request checks --full-name=pytorch/helion --pull-request-number=N -o json`
gives every check's conclusion, the `head_sha` and a `details_url`, and raw job
logs are at `https://ossci-raw-job-status.s3.amazonaws.com/log/pytorch/helion/<job_id>`
(gzip; the job id is the last path component of `details_url`). If neither
route works, stop and report: this loop cannot run blind.

## 1. Map the stack

A dirty tree is unrelated work that would end up in the wrong commit. Commit
it as a `WIP:` commit on top first; it rides along through the rebases and
stays out of every submit (below). In step 6, `git reset HEAD~1` turns it back
into uncommitted changes.

Shell variables do not survive between tool calls, so write the literal SHAs
down (in your notes or the reply) rather than relying on `$old` later:

```
git rev-parse HEAD                           # the rollback point: record it
git merge-base origin/main HEAD              # the base of the stack
git log --reverse --format='%h%x09%(trailers:key=stack-info,valueonly,separator=)%x09%s' <base>..HEAD
```

The top commit that has a `stack-info:` line is the submit head: every submit
below is `stack-pr submit -H <its current sha>` (check with `stack-pr view -H
<sha>`; stack-pr rebases the commits above it, the WIP commit included). A
commit without stack-info below the submit head would become a new pull
request, so if one exists, stop and report instead of submitting.

Each line is a commit (bottom first), its `stack-info` pull request URL, and
its subject. For each pull request:

```
gh pr view <number> --repo pytorch/helion --json title,state,headRefOid,mergeable
```

- `MERGED` or `CLOSED`, or no stack-info (never submitted): leave the commit alone.
- `headRefOid` differs from the local sha: the commit changed after the last
  submit, so its CI results are stale; reproduce locally instead of trusting
  them. Check this now, before the shas change.
- Record the failing checks and the unresolved review threads (the fix-pr
  skill, steps 2 and 3, has the commands) so the loop knows what it started with.

Classify every failing check before fixing anything:

- **Infrastructure**: `CUDA Compute Check`, runner provisioning, `cancelled`,
  network or package-index timeouts. Not a code fix; rerun in step 5.
- **Already failing on main**: compare with the latest runs of `main`:
  ```
  gh run list --repo pytorch/helion --branch main --workflow Tests --limit 3 --json databaseId,conclusion,headSha
  gh run view <run-id> --repo pytorch/helion --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'
  ```
  A job failing there too is not the stack's doing. Fix it at the bottom of
  the stack if you can reproduce it (`AGENTS.md`, Fix What You Find); otherwise
  note it in the report.
- **Real**: read the log (`gh run view <run-id> --repo pytorch/helion --log-failed`)
  and fix it in step 2.

## 2. Fix each commit, bottom-up

Start an interactive rebase that stops after every commit (a `break` after
each `pick`, so no editor opens):

```
GIT_SEQUENCE_EDITOR="sed -i '/^pick /a break'" git rebase -i <base>
```

At each stop HEAD is the commit of one pull request:

1. Fix its real CI failures and address its unresolved review comments (fix-pr
   skill, steps 2 and 3). Reproduce with this commit's code, which is what CI
   tested: `scripts/run_tests.py --backend <lane's backend> --changed --since HEAD~1`
   runs the modules this commit touches, or pass the failing node ids. Let
   runs finish before continuing; the rebase rewrites the working tree.
2. If anything changed, amend it, keeping the message and its stack-info line:
   ```
   git add -u                       # plus new files the fix created; no scratch files
   git commit --amend --no-edit
   ```
3. `GIT_EDITOR=true git rebase --continue` moves to the next commit.

CI on a pull request tests it together with every commit below it. A failure
that also shows on a lower pull request belongs to the lowest one where it
appears: fix it there, and at the higher ones confirm the fix carried through
instead of fixing it again. A fix that belongs to no pull request (a failure
on `main`) goes into the commit of the bottom pull request when it is small:
amend it there and add a line about it to that commit's message (above the
stack-info line). A larger fix becomes its own commit at the bottom of the
stack (commit it at the first stop, with a conventional message and no
stack-info line), which needs a new pull request: finish the round without
submitting and stop, so the user can submit it.

If replaying the next commit conflicts with a fix you just amended, resolve the
conflict keeping both intents, `git add` the files, and `GIT_EDITOR=true git
rebase --continue`; the rebase then stops at that commit's `break` as usual.
Never `git rebase --skip` here: it drops a pull request's commit.

## 3. Rebase onto origin/main only when needed

Every rebase re-runs CI for the whole stack, so rebase when GitHub reports a
conflict (`mergeable` is `CONFLICTING`), when a lower pull request merged, or
when `main` carries a fix a failing job needs; otherwise leave the base alone.
The stack-history skill has the recipe and the checks (every unmerged pull
request once, in order, trailer intact; the only commits you may `--skip` are
ones whose pull request is `MERGED`).

## 4. Verify locally, then resubmit

Lint, and run the lanes for the files the fixes touched (verify skill). Then,
only if something changed since the last submit:

```
stack-pr submit -H <submit head sha>    # --update-body when a commit message changed
```

Record the new head sha of every pull request.

## 5. Watch CI, then loop

CI takes 20 to 60 minutes. Poll instead of idling: `gh pr checks <number>
--repo pytorch/helion` every five to ten minutes (`gh pr checks <number>
--watch --fail-fast` in the background, or `meta github.pull-request await`
on a devserver), and use the wait for other work, committed before the next
fix round and not in this checkout while a local run still reads it. When the
checks have finished:

- every required check green on every pull request: done, go to step 6;
- an infrastructure failure: `gh run rerun <run-id> --repo pytorch/helion --failed`
  once; a second infrastructure failure of the same job goes in the report;
- a real failure: back to step 2 for the lowest pull request it appears on.

Stop and report when the same failure survives two fix attempts, after three
submit rounds, when a failure reproduces only on hardware you do not have and
cannot be skipped responsibly, or when the next submit would create a pull
request.

## 6. Report

- First `git reset HEAD~1` if you committed a WIP commit in step 1.
- The original and new head shas (roll back with `git reset --hard <old>`),
  and the number of submit rounds.
- Per pull request (number and title): CI failures fixed, review comments
  addressed, infrastructure jobs rerun, and anything still open and why.
- Stale-CI warnings from step 1, rebases and how each conflict was resolved,
  commits dropped as merged, `main` fixes folded into the bottom pull request,
  and any new commit waiting for the user to submit.
- Whether a WIP commit was restored to uncommitted changes.
- Lint and lane results at the final head, and jobs whose logs you could not read.
