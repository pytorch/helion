#!/usr/bin/env bash
# Source this file to activate a validated, isolated paper runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script: source setup_env.sh forward|backward [artifact_root]" >&2
    exit 2
fi

_paper_attention_activate() {
    local paper_target="${1:-forward}"
    local paper_artifact="${2:-${PAPER_REPRO_ARTIFACT_ROOT:-}}"
    local paper_script_dir
    local paper_exports
    local paper_bootstrap="${PAPER_REPRO_BOOTSTRAP_PYTHON:-python3}"
    paper_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    paper_exports="$("${paper_bootstrap}" - "${paper_script_dir}/runtime.json" "${paper_target}" "${paper_artifact}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

pointer, requested_target, override = sys.argv[1:]
target_name = {
    "forward": "forward_fused",
    "forward_sparse": "forward_sparse",
    "backward": "backward",
}[requested_target]
pointer_path = Path(pointer).resolve()
artifact = (
    Path(override)
    if override
    else pointer_path.parent / json.loads(pointer_path.read_text())["artifact_root"]
).resolve()
manifest_path = artifact / "runtime_manifest.json"
manifest = json.loads(manifest_path.read_text())
target = manifest["targets"][target_name]
if not target["software_validated"]:
    raise SystemExit(f"Paper runtime {target_name} has not passed its version checks: {manifest_path}")
values = dict(target["environment"])
values.update({
    "PAPER_REPRO_ARTIFACT_ROOT": str(artifact),
    "PAPER_REPRO_TARGET": target_name,
    "PAPER_REPRO_MANIFEST": str(manifest_path),
    "PAPER_REPRO_PYTHON": target["python"],
    "PAPER_REPRO_PROBE": target["software_probe_file"],
    "PAPER_HELION": target["helion_root"],
    "PAPER_ATTN": target["harness_root"],
    "HELION_FA4_ROOT": target["fa4_pythonpath"],
    "HELION_BACKEND": "cute",
})
for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
PY
)" || return
    eval "${paper_exports}"
    source "$(dirname "${PAPER_REPRO_PYTHON}")/activate" || return
    "${PAPER_REPRO_PYTHON}" "${PAPER_REPRO_PROBE}" \
        --manifest "${PAPER_REPRO_MANIFEST}" --target "${PAPER_REPRO_TARGET}" --quiet
}

_paper_attention_activate "$@"
_paper_attention_status=$?
unset -f _paper_attention_activate
if [[ "${_paper_attention_status}" -ne 0 ]]; then
    return "${_paper_attention_status}"
fi
unset _paper_attention_status
