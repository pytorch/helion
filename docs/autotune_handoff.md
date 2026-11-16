---
orphan: true
---

# Native-Source Hand-off

Configuration search stops at what Helion's config space can express. The
hand-off continues from there: when the search finishes (or reaches
`autotune_budget_seconds`), Helion exports the best kernel as standalone
native source, such as a CuTe DSL module, and repeatedly asks an agent you
supply to rewrite it. Helion checks each proposal against the reference
outputs and times it next to the current best source. It keeps the proposal
only if it is faster by more than the measurement noise.

Helion does not run or confine the agent. The agent is a Python callable: it
can call a model API, drive a coding-agent CLI, or do anything else.
Sandboxing that work is up to the user.

The hand-off is experimental: Helion warns when it runs, and its API, prompt
and records may change.

## The agent

Helion calls the agent once per round with a
`helion.autotuner.HandoffRequest`:

| Field | Contents |
| --- | --- |
| `prompt` | The task: objective, cases, accuracy contract, and session deadline. |
| `sources` | The current best sources, keyed by path, e.g. `{"case_0/kernel.py": "..."}`. |
| `feedback` | The previous round's outcome (`None` in the first round): `accepted`, `reason`, timings of the candidate and the incumbent, any `error`, and `remaining_seconds`. |
| `deadline` | The session's end, in `time.monotonic()` seconds. |

The agent returns complete new contents for one or more of the request's
paths, or `None` to end the session. Calls are synchronous, so the agent
must return before `deadline`: a proposal returned late is not accepted.
If the agent raises, the session ends, as it does after 20 invalid proposals
in a row. Each autotuned specialization runs its own session with the same
agent; a session's first request has `feedback` `None`.

## Example

This agent runs a coding-agent CLI, Codex or Claude Code, in a scratch
directory that holds the current sources, and returns the files as the CLI
left them.

```python
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch

import helion
from helion.autotuner import HandoffRequest
import helion.language as hl

# Reads the task on stdin and edits files in its working directory.
COMMAND = ["codex", "exec", "-s", "workspace-write", "--skip-git-repo-check", "-"]
# COMMAND = ["claude", "-p", "--permission-mode", "acceptEdits"]


def agent(request: HandoffRequest) -> dict[str, str] | None:
    task = request.prompt + "\n\nEdit the listed files in the current directory."
    if request.feedback is not None:
        task += f"\n\nResult of your previous edit: {json.dumps(request.feedback)}"
    with tempfile.TemporaryDirectory() as workspace:
        for path, source in request.sources.items():
            (Path(workspace) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workspace) / path).write_text(source)
        try:
            subprocess.run(
                COMMAND,
                input=task,
                text=True,
                cwd=workspace,
                timeout=request.deadline - time.monotonic(),
            )
        except subprocess.TimeoutExpired:
            return None  # out of time: end the session
        return {path: (Path(workspace) / path).read_text() for path in request.sources}


@helion.kernel(
    backend="cute",
    autotune_handoff=True,
    autotune_handoff_agent=agent,
    autotune_handoff_budget_seconds=1800,
)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


if __name__ == "__main__":  # autotuning workers re-import this module
    x = torch.randn(8192, 8192, device="cuda")
    y = torch.randn(8192, 8192, device="cuda")
    add(x, y)  # config search, then up to 1800 s of agent rounds
```

Each call starts a fresh CLI session that learns about earlier rounds only
from `feedback`. An agent can instead keep one session open across calls.
The CLI's own sandbox (`workspace-write`, `acceptEdits`) limits where it
writes files; any further confinement is up to the user.

The hand-off options are set on the kernel only; there are no environment
variables for them. With the hand-off enabled and no agent, Helion raises an
error before it starts the search.

## Results

The kernel call still returns the configuration search's result. The
hand-off's output is files: it writes the standalone kernel to
`<autotune_log>.handoff/<id>/`, or to `$TMPDIR/helion-handoff/<id>/` when
`autotune_log` is unset, and the log reports the path. After the session,
that directory holds the best accepted sources. `agent_runs/<id>/` inside it
records the prompt, the baseline timing, and each round's candidate,
incumbent, decision (`round.json`) and feedback (`feedback.json`).

See {doc}`api/settings` for `autotune_handoff`,
`autotune_handoff_budget_seconds` and `autotune_handoff_agent`.
