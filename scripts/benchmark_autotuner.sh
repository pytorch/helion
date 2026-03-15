#!/usr/bin/env bash
# benchmark_autotuner.sh — Run a Helion example with a selected autotuner.
#
# Usage:
#   ./scripts/benchmark_autotuner.sh [OPTIONS]
#
# Options:
#   -a, --autotuner NAME          Autotuner to use (default: LFBOTreeSearch)
#                                 Valid: LFBOTreeSearch, LFBOPatternSearch,
#                                        PatternSearch, DifferentialEvolutionSearch,
#                                        DESurrogateHybrid, RandomSearch,
#                                        EmbeddingBeamSearch
#   -e, --example FILE            Example script to run (default: examples/softmax.py)
#   --endpoint URL                Embedding endpoint URL (required for EmbeddingBeamSearch)
#   --model MODEL                 Embedding model name (required for EmbeddingBeamSearch)
#                                 Example: intfloat/multilingual-e5-large-instruct
#   --token TOKEN                 Bearer token for the embedding endpoint (optional)
#   --beam-width N                Beam width for EmbeddingBeamSearch (default: 5)
#   --num-neighbors N             Neighbor configs generated per generation (default: 100)
#   --frac-selected F             Fraction of candidates to benchmark, e.g. 0.3 (default: 0.3)
#   --initial-ratio F             Starting exploration ratio, e.g. 0.8 (default: 0.8)
#   --final-ratio F               Ending exploration ratio, e.g. 0.2 (default: 0.2)
#   --effort LEVEL                Autotuning effort: quick|full (default: quick)
#   --max-generations N           Override maximum number of generations
#   --logs                        Enable full autotuning logs (HELION_LOGS=all)
#   -h, --help                    Show this help message and exit
#
# Examples:
#   ./scripts/benchmark_autotuner.sh
#   ./scripts/benchmark_autotuner.sh --autotuner LFBOTreeSearch
#   ./scripts/benchmark_autotuner.sh --autotuner EmbeddingBeamSearch --endpoint https://api.together.xyz/v1/embeddings --model intfloat/multilingual-e5-large-instruct
#   ./scripts/benchmark_autotuner.sh --autotuner EmbeddingBeamSearch --endpoint https://api.together.xyz/v1/embeddings --model intfloat/multilingual-e5-large-instruct --token sk-abc123 --beam-width 7 --frac-selected 0.2 --initial-ratio 0.9 --final-ratio 0.1
#   ./scripts/benchmark_autotuner.sh --autotuner RandomSearch --effort quick --example examples/softmax.py --logs

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
AUTOTUNER="LFBOTreeSearch"
EXAMPLE="examples/softmax.py"

# ENDPOINT="https://api.together.xyz/v1/embeddings"
# TOKEN="tgp_v1_UnUn_sE14R10Cg7F2pTx5u9x2v5I3JMEHkXwQQtXp_c"
# MODEL="intfloat/multilingual-e5-large-instruct"

ENDPOINT="http://46.243.145.238:8000/v1/embeddings"
TOKEN=""
MODEL="Qwen/Qwen3-Embedding-8B"

BEAM_WIDTH=""
NUM_NEIGHBORS="100"
FRAC_SELECTED=""
INITIAL_RATIO="0.5"
FINAL_RATIO="0.0"
EFFORT="quick"
MAX_GENERATIONS="5"
ENABLE_LOGS=0

VALID_AUTOTUNERS="LFBOTreeSearch LFBOPatternSearch PatternSearch DifferentialEvolutionSearch DESurrogateHybrid RandomSearch EmbeddingBeamSearch"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--autotuner)
            AUTOTUNER="$2"; shift 2 ;;
        -e|--example)
            EXAMPLE="$2"; shift 2 ;;
        --endpoint)
            ENDPOINT="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --token)
            TOKEN="$2"; shift 2 ;;
        --beam-width)
            BEAM_WIDTH="$2"; shift 2 ;;
        --num-neighbors)
            NUM_NEIGHBORS="$2"; shift 2 ;;
        --frac-selected)
            FRAC_SELECTED="$2"; shift 2 ;;
        --initial-ratio)
            INITIAL_RATIO="$2"; shift 2 ;;
        --final-ratio)
            FINAL_RATIO="$2"; shift 2 ;;
        --effort)
            EFFORT="$2"; shift 2 ;;
        --max-generations)
            MAX_GENERATIONS="$2"; shift 2 ;;
        --logs)
            ENABLE_LOGS=1; shift ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if ! echo "$VALID_AUTOTUNERS" | grep -qw "$AUTOTUNER"; then
    echo "Error: unknown autotuner '$AUTOTUNER'." >&2
    echo "Valid options: $VALID_AUTOTUNERS" >&2
    exit 1
fi

if [[ "$AUTOTUNER" == "EmbeddingBeamSearch" && -z "$ENDPOINT" ]]; then
    echo "Error: --endpoint is required when using EmbeddingBeamSearch." >&2
    echo "  Example: --endpoint https://api.together.xyz/v1/embeddings" >&2
    exit 1
fi

if [[ "$AUTOTUNER" == "EmbeddingBeamSearch" && -z "$MODEL" ]]; then
    echo "Error: --model is required when using EmbeddingBeamSearch." >&2
    echo "  Example: --model intfloat/multilingual-e5-large-instruct" >&2
    exit 1
fi

if [[ "$EFFORT" != "quick" && "$EFFORT" != "full" ]]; then
    echo "Error: --effort must be 'quick' or 'full', got '$EFFORT'." >&2
    exit 1
fi

if [[ ! -f "$EXAMPLE" ]]; then
    echo "Error: example script not found: $EXAMPLE" >&2
    exit 1
fi

# ── Build environment ─────────────────────────────────────────────────────────
declare -a ENV_VARS=(
    "HELION_AUTOTUNER=$AUTOTUNER"
    "HELION_AUTOTUNE_EFFORT=$EFFORT"
)

if [[ "$AUTOTUNER" == "EmbeddingBeamSearch" ]]; then
    ENV_VARS+=("HELION_EMBEDDING_ENDPOINT=$ENDPOINT")
    ENV_VARS+=("HELION_EMBEDDING_MODEL=$MODEL")
    [[ -n "$TOKEN" ]]         && ENV_VARS+=("HELION_EMBEDDING_API_TOKEN=$TOKEN")
    [[ -n "$BEAM_WIDTH" ]]    && ENV_VARS+=("HELION_EMBEDDING_BEAM_WIDTH=$BEAM_WIDTH")
    [[ -n "$NUM_NEIGHBORS" ]] && ENV_VARS+=("HELION_EMBEDDING_NUM_NEIGHBORS=$NUM_NEIGHBORS")
    [[ -n "$FRAC_SELECTED" ]] && ENV_VARS+=("HELION_EMBEDDING_FRAC_SELECTED=$FRAC_SELECTED")
    [[ -n "$INITIAL_RATIO" ]] && ENV_VARS+=("HELION_EMBEDDING_INITIAL_EXPLORATION=$INITIAL_RATIO")
    [[ -n "$FINAL_RATIO" ]]   && ENV_VARS+=("HELION_EMBEDDING_FINAL_EXPLORATION=$FINAL_RATIO")
fi

[[ -n "$MAX_GENERATIONS" ]] && ENV_VARS+=("HELION_AUTOTUNE_MAX_GENERATIONS=$MAX_GENERATIONS")
[[ "$ENABLE_LOGS" -eq 1 ]]  && ENV_VARS+=("HELION_LOGS=all")

# ── Print summary ─────────────────────────────────────────────────────────────
echo "┌─ Autotuner benchmark ────────────────────────────────────────"
echo "│  Autotuner : $AUTOTUNER"
echo "│  Example   : $EXAMPLE"
echo "│  Effort    : $EFFORT"
[[ -n "$MAX_GENERATIONS" ]] && echo "│  Max gens  : $MAX_GENERATIONS"
if [[ "$AUTOTUNER" == "EmbeddingBeamSearch" ]]; then
    echo "│  Endpoint  : $ENDPOINT"
    echo "│  Model     : $MODEL"
    [[ -n "$TOKEN" ]]         && echo "│  API token : (set)"
    [[ -n "$BEAM_WIDTH" ]]    && echo "│  Beam width: $BEAM_WIDTH"
    [[ -n "$NUM_NEIGHBORS" ]] && echo "│  Neighbors : $NUM_NEIGHBORS"
    [[ -n "$FRAC_SELECTED" ]] && echo "│  Frac sel. : $FRAC_SELECTED"
    [[ -n "$INITIAL_RATIO" ]] && echo "│  Init ratio: $INITIAL_RATIO"
    [[ -n "$FINAL_RATIO" ]]   && echo "│  Final rat.: $FINAL_RATIO"
fi
[[ "$ENABLE_LOGS" -eq 1 ]] && echo "│  Logs      : enabled"
echo "└──────────────────────────────────────────────────────────────"

# ── Run ───────────────────────────────────────────────────────────────────────
env "${ENV_VARS[@]}" python "$EXAMPLE"
