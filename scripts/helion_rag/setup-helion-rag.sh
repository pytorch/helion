#!/usr/bin/env bash
# setup-helion-rag.sh - configure standalone RAG tools for a Helion checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$SCRIPT_DIR"
# Helper runner: use PYTHONPATH fallback before pip install, then bare command after
helion_rag_py() {
  PYTHONPATH="$PKG_ROOT:${PYTHONPATH:-}" "$INTERP" -m helion_rag "$@"
}
helion_rag_cmd() {
  if command -v helion-rag >/dev/null 2>&1; then
    helion-rag "$@"
  else
    helion_rag_py "$@"
  fi
}

HELION_REPO=""; PYTHON_BIN=""; MANIFOLD_BASE=""; HW_FAMILY=""
INTERACTIVE=1; DRY_RUN=0; ALLOW_GLOBAL=0
EMBED_MODEL="${HELION_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
HF_HOME_VALUE="${HF_HOME:-}"
COLLECT_AUTOTUNE=0
MANIFEST_TMP=""

die()  { echo "[setup-helion-rag] error: $*" >&2; exit 1; }
info() { echo "[setup-helion-rag] $*" >&2; }
plan() { echo "[setup-helion-rag] (plan) $*" >&2; }
shq()  { local s=${1//\'/\'\\\'\'}; printf "'%s'" "$s"; }  # shell-quote for env.sh safety

usage() {
  cat >&2 <<'USAGE'
Usage: setup-helion-rag.sh --helion-repo <path> [options]
  --helion-repo <path>     REQUIRED Helion checkout to configure
  --python <path>          interpreter default python then python3
  --manifold-base <uri>    Manifold base with manifest artifacts contrib
  --hardware-family <fam>  required non-interactive if detection fails
  --non-interactive        take values from flags/env only
  --dry-run                validate and print plan no mutation
  --allow-global-python    permit install into non-venv
USAGE
}

while [ $# -gt 0 ]; do case "$1" in
  --helion-repo) HELION_REPO="${2:-}"; shift 2;;
  --python) PYTHON_BIN="${2:-}"; shift 2;;
  --manifold-base) MANIFOLD_BASE="${2:-}"; shift 2;;
  --hardware-family) HW_FAMILY="${2:-}"; shift 2;;
  --non-interactive) INTERACTIVE=0; shift;;
  --dry-run) DRY_RUN=1; shift;;
  --allow-global-python) ALLOW_GLOBAL=1; shift;;
  -h|--help) usage; exit 0;;
  *) usage; die "unknown argument $1";;
esac; done

# Stage 1: repo validation
[ -n "$HELION_REPO" ] || { usage; die "--helion-repo required"; }
[ -n "$MANIFOLD_BASE" ] || { usage; die "--manifold-base required"; }
[ -d "$HELION_REPO" ] || die "helion repo not found: $HELION_REPO"
HELION_REPO="$(cd "$HELION_REPO" && pwd)"
[ -e "$HELION_REPO/helion/__init__.py" ] || die "not a Helion checkout"
RAG_ROOT="$HELION_REPO/.helion-rag"

# Stage 2: interpreter
resolve_interp() {
  [ -n "$PYTHON_BIN" ] && { echo "$PYTHON_BIN"; return; }
  command -v python >/dev/null && { command -v python; return; }
  command -v python3 >/dev/null && { command -v python3; return; }
  die "no python found on PATH"
}
INTERP="$(resolve_interp)"
[ -x "$INTERP" ] || command -v "$INTERP" >/dev/null || die "interpreter not executable $INTERP"

if ! PYTHONPATH="$HELION_REPO:${PYTHONPATH:-}" "$INTERP" -m helion_rag setup-helper check-import --repo "$HELION_REPO" >/dev/null 2>&1; then
  # fallback to old inline check for clearer error code handling
  if ! PYTHONPATH="$HELION_REPO:${PYTHONPATH:-}" "$INTERP" <<'PY' "$HELION_REPO" >/dev/null 2>&1
import os, sys
repo=os.path.realpath(sys.argv[1]); sys.path.insert(0,repo); import helion; path=os.path.realpath(helion.__file__); sys.exit(0 if path.startswith(repo) else 3)
PY
  then
    rc=$?
    [ $rc -eq 3 ] && die "interpreter imports Helion from wrong location (expected $HELION_REPO)"
    die "interpreter cannot import Helion from $HELION_REPO"
  fi
fi
# also verify via helper returns expected code
helion_rag_py setup-helper check-import --repo "$HELION_REPO" >/dev/null || {
  rc=$?
  [ $rc -eq 3 ] && die "interpreter imports Helion from wrong location"
  [ $rc -ne 0 ] && die "interpreter cannot import Helion from $HELION_REPO"
}
info "interpreter: $INTERP"

# Stage 3: manifold and manifest
command -v manifold >/dev/null || die "manifold CLI not found"
manifold ls "$MANIFOLD_BASE" >/dev/null || die "cannot access Manifold base $MANIFOLD_BASE"
MANIFEST_DIR="$(mktemp -d)"; MANIFEST_TMP="$MANIFEST_DIR/manifest.json"
trap 'rm -rf "$MANIFEST_DIR"' EXIT
if manifold get "$MANIFOLD_BASE/manifest.json" "$MANIFEST_TMP" >/dev/null 2>&1; then
  info "fetched published manifest.json"
else
  info "no manifest.json in bucket; synthesizing from family-dir layout"
  FAMILIES="$(manifold ls "$MANIFOLD_BASE" 2>/dev/null | awk '/^DIR/ {print $NF}')"
  [ -n "$FAMILIES" ] || die "no family dirs under $MANIFOLD_BASE to synthesize a manifest"
  # shellcheck disable=SC2086
  helion_rag_py setup-helper synthesize --out "$MANIFEST_TMP" $FAMILIES || die "manifest synthesis failed"
  info "synthesized manifest for families: $(echo $FAMILIES | tr '\n' ' ')"
fi
helion_rag_py setup-helper validate --manifest "$MANIFEST_TMP" || die "manifest failed schema validation"
info "manifold manifest valid"

# Stage 4: venv guard
is_venv() { "$INTERP" -c "import sys; sys.exit(0 if sys.prefix!=sys.base_prefix else 1)" >/dev/null 2>&1; }
if [ "$DRY_RUN" -eq 0 ] && ! is_venv; then
  if [ "$INTERACTIVE" -eq 0 ] && [ "$ALLOW_GLOBAL" -eq 0 ]; then
    die "interpreter looks global; re-run with --allow-global-python"
  fi
  info "warning: '$INTERP' has no venv marker"
fi

# Stage 5: hardware family
resolve_family() {
  helion_rag_py setup-helper resolve-family --manifest "$MANIFEST_TMP"
}
if [ -z "$HW_FAMILY" ]; then HW_FAMILY="$(resolve_family || true)"; fi
if [ -z "$HW_FAMILY" ]; then
  [ "$INTERACTIVE" -eq 0 ] && die "could not detect hardware family; pass --hardware-family"
  read -r -p "[setup-helion-rag] enter hardware family (h100/b200/tpu/mi350x): " HW_FAMILY
  [ -n "$HW_FAMILY" ] || die "no hardware family provided"
fi
info "hardware family: $HW_FAMILY"

is_represented_family() {
  helion_rag_py setup-helper is-represented --manifest "$MANIFEST_TMP" --family "$HW_FAMILY"
}
artifact_path_for_family() {
  helion_rag_py setup-helper artifact-path --manifest "$MANIFEST_TMP" --family "$HW_FAMILY"
}

FAMILY_REPRESENTED=0
if is_represented_family; then FAMILY_REPRESENTED=1; info "hardware family represented: $HW_FAMILY"
else info "hardware family not represented: $HW_FAMILY"; fi

if [ "$DRY_RUN" -eq 1 ]; then
  plan "would create $RAG_ROOT/{ci_artifacts,rag_index,rag_writeback,autotune_logs,uploads}"
  plan "would pip install -e $PKG_ROOT"
  if [ "$FAMILY_REPRESENTED" -eq 1 ]; then
    plan "would install deps, write env.sh, fetch corpus, and build index"
  else
    plan "would write env.sh and skip fetch/index"
  fi
  info "dry-run complete"; exit 0
fi

# Stage 6: gitignore
GITIGNORE="$HELION_REPO/.gitignore"
if [ ! -f "$GITIGNORE" ] || ! grep -qxF ".helion-rag/" "$GITIGNORE"; then
  echo ".helion-rag/" >> "$GITIGNORE"; info "added .helion-rag/ ignore"
else info ".helion-rag/ already ignored"; fi

# Stage 7: layout and environment
mkdir -p "$RAG_ROOT"/{ci_artifacts,rag_index,rag_writeback,autotune_logs,uploads}
cp "$MANIFEST_TMP" "$RAG_ROOT/manifest.json"
info "created $RAG_ROOT layout"

write_env() {
  local collect="${1:-0}"
  {
    echo "# Managed by setup-helion-rag.sh"
    printf 'export HELION_RAG_HARDWARE_FAMILY=%s\n' "$(shq "$HW_FAMILY")"
    printf 'export HELION_RAG_MANIFOLD_BASE=%s\n' "$(shq "$MANIFOLD_BASE")"
    printf 'export HELION_RAG_MANIFEST=%s\n' "$(shq "$RAG_ROOT/manifest.json")"
    printf 'export HELION_RAG_DATA_DIR=%s\n' "$(shq "$RAG_ROOT/ci_artifacts")"
    printf 'export HELION_RAG_INDEX_DIR=%s\n' "$(shq "$RAG_ROOT/rag_index")"
    printf 'export HELION_RAG_WRITEBACK_DIR=%s\n' "$(shq "$RAG_ROOT/rag_writeback")"
    printf 'export HELION_RAG_AUTOTUNE_LOG_DIR=%s\n' "$(shq "$RAG_ROOT/autotune_logs")"
    printf 'export HELION_RAG_UPLOADS_DIR=%s\n' "$(shq "$RAG_ROOT/uploads")"
    printf 'export HELION_EMBED_MODEL=%s\n' "$(shq "$EMBED_MODEL")"
  } > "$RAG_ROOT/env.sh"
  [ -n "$HF_HOME_VALUE" ] && printf 'export HF_HOME=%s\n' "$(shq "$HF_HOME_VALUE")" >> "$RAG_ROOT/env.sh"
  if [ "$collect" -eq 1 ]; then
    printf 'export HELION_AUTOTUNE_LOG=%s\n' "$(shq "$RAG_ROOT/autotune_logs/helion-rag")" >> "$RAG_ROOT/env.sh"
    printf 'export HELION_AUTOTUNE_LOG_DETAILS=1\n' >> "$RAG_ROOT/env.sh"
  fi
}
write_env 0
info "wrote RAG environment to $RAG_ROOT/env.sh"

if [ "$FAMILY_REPRESENTED" -eq 0 ]; then
  if [ "$INTERACTIVE" -eq 1 ]; then
    read -r -p "[setup-helion-rag] hardware not covered; enable local autotune collection? [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then write_env 1; info "enabled local autotune collection"; fi
  fi
  info "uncovered hardware: skipping corpus fetch and index build"; exit 0
fi

# pip install package editable early for helpers and CLI availability.
# This also installs the RAG dependencies declared in pyproject.toml.
info "installing helion-rag package editable from $PKG_ROOT"
"$INTERP" -m pip install -e "$PKG_ROOT" >&2 || die "pip install -e failed"

if [ -z "${HF_TOKEN:-}" ]; then
  info "note: no HF_TOKEN — first embedding download may be slow. Use huggingface-cli login for faster downloads; RAG does not store token."
fi

# Stage 8: corpus fetch + index
ARTIFACT_PATH="$(artifact_path_for_family)"
info "fetching corpus for $HW_FAMILY from $MANIFOLD_BASE/$ARTIFACT_PATH"
stage_rc=0
# shellcheck disable=SC1091
( set -a; . "$RAG_ROOT/env.sh"; set +a
  manifold getr "$MANIFOLD_BASE/$ARTIFACT_PATH" "$RAG_ROOT/ci_artifacts/$HW_FAMILY" || exit 16
  helion_rag_cmd extract || exit 18
  helion_rag_cmd index || exit 17
) || stage_rc=$?
case "$stage_rc" in
  0) ;;
  16) die "corpus FETCH failed manifold getr $MANIFOLD_BASE/$ARTIFACT_PATH";;
  18) die "corpus EXTRACT failed under $RAG_ROOT/ci_artifacts";;
  17) die "index BUILD failed helion_rag index";;
  *) die "fetch/extract/index failed exit $stage_rc";;
esac

# Stage 9: shell rc
RC_FILE="${HELION_RAG_RC_FILE:-$HOME/.bashrc}"
BEGIN="# >>> helion-rag >>>"; END="# <<< helion-rag <<<"
if [ -f "$RC_FILE" ] && grep -qF "$BEGIN" "$RC_FILE"; then
  info "shell-rc already present in $RC_FILE"
else
  { echo "$BEGIN"; printf '[ -f %s ] && . %s\n' "$(shq "$RAG_ROOT/env.sh")" "$(shq "$RAG_ROOT/env.sh")"; echo "$END"; } >> "$RC_FILE"
  info "added shell-rc to $RC_FILE"
fi
info "setup complete — open new shell or source $RAG_ROOT/env.sh"
