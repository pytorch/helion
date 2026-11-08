---
name: stack-history
description: Recipes for rewriting the local commit stack above origin/main without an interactive editor — fold a fix into an earlier commit, squash, reorder, split, reword while keeping stack-info trailers, rebase onto origin/main with a check for duplicated upstream work, and work in worktrees. Use when shaping a stack before stack-pr submit or amending fixes into already-submitted pull requests.
---

# Rewriting the stack

`AGENTS.md` (Commits and the Stack) says when to rewrite: freely before the
stack is submitted; afterwards only by amending into the commit of the pull
request a change belongs to, keeping each commit's `stack-info:` trailer as
its last line and never squashing or reordering pull requests.

Before any rewrite:

- find out now whether the stack is submitted; never rely on what you saw
  earlier, because the user may have run `stack-pr submit` since. Record every
  commit's trailer:

  ```
  git log --format='%h %s%n    %(trailers:key=stack-info,valueonly)' origin/main..HEAD
  ```

  A commit with a `stack-info:` line has a pull request. That line must
  survive every rewrite: a commit that loses it looks new to the next submit,
  which opens a duplicate pull request;
- the tree is clean (`git status`), and no test run or autoreview is reading
  this checkout;
- record the rollback point as a literal SHA: run `git rev-parse HEAD` and
  write the output down. Shell variables do not survive between tool calls,
  and `git reset --hard $old` with an empty `$old` resets to HEAD instead of
  rolling back. `git reset --hard <that sha>` undoes everything below, and
  `git rebase --abort` undoes a rebase in progress. These are your own
  commits; `reset --hard` is never for changes you did not make.

Each commit message ends with its `autoreview passed: ...` line, above any
`stack-info:` trailer. Rewording keeps both. A rewrite that changes a commit's
content (a squash, a split, a fold of anything but fixes for that commit's own
review, a conflict resolution) makes the line stale: remove it and review the
commit again (AGENTS.md, Recording the review).

Rewrite only with the recipes below. They keep each commit's message, and so
its trailer, unless you edit it. Do not rebuild commits from saved copies of
their messages or with your own scripts: that is how trailers get lost. If a
script is unavoidable, it must read each commit's current message (`git log
-1 --format=%B <sha>`), trailers included, and write it back unchanged except
for the edit you mean.

`<base>` below is the output of `git merge-base origin/main HEAD`, the bottom
of the stack. `GIT_SEQUENCE_EDITOR` edits the rebase todo list
non-interactively and `GIT_EDITOR=true` accepts default messages, so no editor
opens. Scratch files come from `mktemp`: fixed names under /tmp collide with
other accounts on a shared machine.

## Fold a fix into an earlier commit

```
git add -u                         # plus new files that belong to the fix
git commit --fixup=<sha>
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash <base>
```

The fix lands in `<sha>` with its message (and trailer) untouched.

## Squash commits

For the top two commits, keep the older message:

```
git reset --soft HEAD~1
git commit --amend --no-edit
```

For commits lower in the stack, write the todo list: move the newer commit's
line directly below the older one and change its `pick` to `fixup` (a
`fixup` folds into the line just above it, so the two must be adjacent in the
list, wherever they were in the history):

```
todo=$(mktemp)
git log --reverse --format='pick %h %s' <base>..HEAD > "$todo"
# edit "$todo" (with a script or the Edit tool), then, in the same command or with the literal path:
GIT_SEQUENCE_EDITOR="cp $todo" git rebase -i <base>
```

Then reword the survivor if its message should mention the folded work. Use
`fixup` rather than `squash`: `squash` opens an editor for the combined
message.

## Reorder or drop commits

The same todo file, in the order you want; delete a line to drop that commit:

```
todo=$(mktemp)
git log --reverse --format='pick %h %s' <base>..HEAD > "$todo"
GIT_SEQUENCE_EDITOR="cp $todo" git rebase -i <base>
```

## Reword a commit

The trailer that stack-pr added must survive, so start from the current
message and edit only what is above the `stack-info:` line. For HEAD:

```
msg=$(mktemp)
git log -1 --format=%B > "$msg"            # edit everything above the stack-info line
git commit --amend -F "$msg"
```

For an older commit, stop the rebase there and amend:

```
GIT_SEQUENCE_EDITOR="sed -i 's/^pick <sha>/edit <sha>/'" git rebase -i <base>
msg=$(mktemp); git log -1 --format=%B > "$msg"   # edit it, then:
git commit --amend -F "$msg"
GIT_EDITOR=true git rebase --continue
```

Check afterwards that `git log --format=%B -1 <new sha> | tail -1` still
starts with `stack-info:` for a submitted commit.

## Split a commit

```
GIT_SEQUENCE_EDITOR="sed -i 's/^pick <sha>/edit <sha>/'" git rebase -i <base>
git reset HEAD~1                       # the commit's changes are back in the tree, unstaged
git add <files of the first part>      # for part of a file: write a patch and `git apply --cached`
git commit -F <message file>           # or -m; a bare `git commit` opens an editor
git add -A
git commit -F <message file>
GIT_EDITOR=true git rebase --continue
```

## Rebase onto origin/main

```
git fetch origin
git log --oneline HEAD..origin/main -- $(git diff --name-only <base> HEAD)   # upstream work on the stack's files
git rebase origin/main
```

Read each listed upstream commit: if it does the same thing as one of yours,
drop or shrink yours and say so in its message. On a conflict, keep both
main's change and the commit's intent, `git add` the files, and `GIT_EDITOR=true
git rebase --continue`; never `--skip` a commit of yours (git drops commits
whose pull request already merged by itself). Afterwards:

```
git range-diff <base>..<rollback sha> origin/main..HEAD
```

shows what the rebase changed per commit. Rerun the lanes for files the
upstream commits also touched.

## Check the result of any rewrite

- `git log --oneline origin/main..HEAD`: the commits you expect, in order,
  with conventional titles and no `fixup!` or `WIP` left.
- `git diff <rollback sha> HEAD`: empty when you only reorganized history;
  otherwise exactly the content you meant to change.
- Every `stack-info:` line you recorded before the rewrite is still there, on
  the commit with the same title, and still last. Check it every time, not
  only when you believe the stack is submitted:

  ```
  git log --format='%h %s%n    %(trailers:key=stack-info,valueonly)' origin/main..HEAD
  ```

  A missing one is an error: put it back (reword recipe) before anything
  else, and before any submit. `stack-pr view` should show every submitted
  commit with its pull request number; a commit without one would get a new,
  duplicate pull request.

## Worktrees

`git worktree add ../helion-<name> <commit>` checks a commit out beside the
repository. Use one to run a lane or an autoreview on a fixed commit while you
keep editing, or to compare generated code between two commits. Run its tests
with its own `scripts/run_tests.py`, which keeps the interpreters tests start
on that tree's `helion` too; `git worktree remove ../helion-<name>` cleans up.
