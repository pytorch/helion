---
name: runpod
description: Auto-activate when the user mentions runpod or scripts/runpod.py
---

# Runpod

- Start a persistent pod early with `python scripts/runpod.py --start` when you expect to need remote GPU runs. This avoids paying startup latency on every command.
- Run remote commands with `python scripts/runpod.py ...`. If a matching persistent pod is already running, the script reuses it automatically. If not, it falls back to one-shot start-run-stop behavior.
- For normal commands, pass argv directly. Inline env prefixes work too: `python scripts/runpod.py HELION_BACKEND=cute pytest test`.
- For compound shell commands, pass one quoted string: `python scripts/runpod.py './lint.sh && pytest test'`. Do not wrap it in `bash -lc`; the script handles shell execution itself.
- Before waiting on the user, shut down any persistent pod you started or reused with `python scripts/runpod.py --cleanup` so it does not keep billing while idle.
- If the script warns that a pod was left running, do not ignore it. Clean it up before waiting for user input.
