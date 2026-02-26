#!/bin/bash
#
# Test OpenEvolve Autotuner on B200 Machines
# ===========================================
#
# This script runs a series of tests to verify the OpenEvolve autotuner
# works correctly on NVIDIA B200 (Blackwell) GPUs.
#
# Usage:
#   ./test_openevolve_b200.sh [quick|full]
#
# Options:
#   quick - Run only fast tests (no GPU, no API calls)
#   full  - Run all tests including GPU benchmarking (default)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test mode
MODE="${1:-full}"

echo "========================================================================"
echo "OpenEvolve Autotuner Test Suite for B200"
echo "========================================================================"
echo ""

# Function to print test status
print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if we're on a B200
print_test "Checking GPU model..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    print_info "Detected GPU: $GPU_NAME"

    if [[ $GPU_NAME == *"B200"* ]] || [[ $GPU_NAME == *"Blackwell"* ]]; then
        print_pass "Running on B200 GPU"
        IS_B200=true
    else
        print_skip "Not running on B200 (detected: $GPU_NAME)"
        print_info "Tests will still run but may not be B200-optimized"
        IS_B200=false
    fi
else
    print_fail "nvidia-smi not found. Are you on a GPU machine?"
    exit 1
fi
echo ""

# Check Python
print_test "Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    print_pass "Python found: $PYTHON_VERSION"
else
    print_fail "Python not found"
    exit 1
fi
echo ""

# Check dependencies
print_test "Checking dependencies..."

if python -c "import torch" 2>/dev/null; then
    print_pass "torch is installed"
else
    print_fail "torch is not installed. Run: pip install torch"
    exit 1
fi

if python -c "import triton" 2>/dev/null; then
    print_pass "triton is installed"
else
    print_fail "triton is not installed. Run: pip install triton"
    exit 1
fi

if python -c "import openevolve" 2>/dev/null; then
    OPENEVOLVE_VERSION=$(python -c "import openevolve; print(openevolve.__version__)")
    print_pass "openevolve is installed (version: $OPENEVOLVE_VERSION)"
else
    print_fail "openevolve is not installed. Run: pip install openevolve"
    exit 1
fi

if python -c "import helion" 2>/dev/null; then
    print_pass "helion is installed"
else
    print_fail "helion is not installed"
    exit 1
fi
echo ""

# Check OpenAI API key
print_test "Checking OpenAI API key..."
if [[ -n "$OPENAI_API_KEY" ]]; then
    KEY_PREFIX=$(echo $OPENAI_API_KEY | head -c 10)
    print_pass "OPENAI_API_KEY is set (${KEY_PREFIX}...)"
    HAS_API_KEY=true
else
    print_skip "OPENAI_API_KEY is not set"
    print_info "Real tuning will not be possible, but structure tests will run"
    HAS_API_KEY=false
fi
echo ""

# TEST 1: Structure validation
print_test "TEST 1: Validating OpenEvolveTuner structure..."
python << 'EOF'
import sys
import importlib.util

# Import directly to avoid torch dependency in __init__
spec = importlib.util.spec_from_file_location(
    "openevolve_tuner",
    "/home/user/helion/helion/autotuner/openevolve_tuner.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OpenEvolveTuner = module.OpenEvolveTuner

# Test initialization
config_space = {
    'block_size': [32, 64, 128],
    'num_warps': [2, 4]
}

tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=lambda c: 100.0,
    max_evaluations=10,
    verbose=False
)

assert tuner.config_space == config_space
assert tuner.max_evaluations == 10
print("✓ OpenEvolveTuner class structure is valid")
EOF

if [ $? -eq 0 ]; then
    print_pass "Structure validation passed"
else
    print_fail "Structure validation failed"
    exit 1
fi
echo ""

# TEST 2: Initial program generation
print_test "TEST 2: Testing initial program generation..."
python << 'EOF'
import sys
import importlib.util

spec = importlib.util.spec_from_file_location(
    "openevolve_tuner",
    "/home/user/helion/helion/autotuner/openevolve_tuner.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OpenEvolveTuner = module.OpenEvolveTuner

config_space = {'block_size': [64, 128], 'num_warps': [2, 4]}
tuner = OpenEvolveTuner(config_space, lambda c: 0.0, 10, verbose=False)

# Generate and execute initial program
initial_program = tuner._generate_initial_program()
exec_globals = {}
exec(initial_program, exec_globals)

assert 'get_kernel_config' in exec_globals
config = exec_globals['get_kernel_config']()
assert isinstance(config, dict)
assert 'block_size' in config
assert 'num_warps' in config
print("✓ Initial program generation works correctly")
EOF

if [ $? -eq 0 ]; then
    print_pass "Initial program generation passed"
else
    print_fail "Initial program generation failed"
    exit 1
fi
echo ""

# TEST 3: Vector add simple test (no tuning)
if [[ "$MODE" == "full" ]]; then
    print_test "TEST 3: Running simple vector add test..."

    if python examples/helion_vector_add_tuning.py --simple 2>&1 | grep -q "working correctly"; then
        print_pass "Vector add kernel test passed"
    else
        print_fail "Vector add kernel test failed"
        exit 1
    fi
    echo ""
else
    print_skip "TEST 3: Skipped in quick mode"
    echo ""
fi

# TEST 4: Mock tuning (no GPU/API required)
print_test "TEST 4: Running mock tuning test..."
unset OPENAI_API_KEY
TIMEOUT=60  # 60 seconds timeout

if timeout $TIMEOUT python << 'EOF'
import sys
import importlib.util

spec = importlib.util.spec_from_file_location(
    "openevolve_tuner",
    "/home/user/helion/helion/autotuner/openevolve_tuner.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OpenEvolveTuner = module.OpenEvolveTuner

config_space = {
    'block_size': [64, 128, 256],
    'num_warps': [2, 4, 8]
}

def mock_objective(config):
    return 100.0 * (config['block_size'] / 128) * (config['num_warps'] / 4)

tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=mock_objective,
    max_evaluations=10,
    verbose=False
)

best_config = tuner.tune()
assert best_config is not None
assert 'block_size' in best_config
assert 'num_warps' in best_config
print(f"✓ Mock tuning completed. Best: {best_config}")
EOF
then
    print_pass "Mock tuning test passed"
else
    print_fail "Mock tuning test failed or timed out"
    exit 1
fi
echo ""

# TEST 5: Real GPU tuning (if API key available and full mode)
if [[ "$MODE" == "full" ]] && [[ "$HAS_API_KEY" == true ]]; then
    print_test "TEST 5: Running real GPU tuning (small scale)..."
    print_info "This will make OpenAI API calls (estimated cost: $0.01-0.02)"
    print_info "Press Ctrl+C within 5 seconds to skip this test..."

    sleep 5

    # Run with small number of evaluations
    if python examples/helion_vector_add_tuning.py 2>&1 | tee /tmp/tuning_output.log | grep -q "Best configuration found"; then
        print_pass "Real GPU tuning completed successfully"

        # Extract and display results
        echo ""
        print_info "Tuning results:"
        grep "Best configuration" /tmp/tuning_output.log || true
        grep "Best performance" /tmp/tuning_output.log || true
    else
        print_fail "Real GPU tuning failed"
        echo ""
        print_info "Check /tmp/tuning_output.log for details"
        exit 1
    fi
    echo ""
elif [[ "$MODE" == "full" ]]; then
    print_skip "TEST 5: Skipped (no API key)"
    echo ""
else
    print_skip "TEST 5: Skipped in quick mode"
    echo ""
fi

# TEST 6: B200-specific test
if [[ "$MODE" == "full" ]] && [[ "$IS_B200" == true ]]; then
    print_test "TEST 6: Running B200-specific attention tuning test..."

    # Set mock mode if no API key
    if [[ "$HAS_API_KEY" != true ]]; then
        unset OPENAI_API_KEY
        print_info "Running in mock mode (no API key)"
    fi

    if timeout 120 python examples/helion_b200_attention_tuning.py 2>&1 | tee /tmp/b200_tuning.log | grep -q "Tuning complete"; then
        print_pass "B200 attention tuning completed"

        # Check for B200-specific features in results
        if grep -q "tensor_descriptor\|persistent_interleaved" /tmp/b200_tuning.log; then
            print_pass "B200-specific features were tuned"
        fi
    else
        print_fail "B200 attention tuning failed"
        echo ""
        print_info "Check /tmp/b200_tuning.log for details"
        exit 1
    fi
    echo ""
else
    print_skip "TEST 6: Skipped (not B200 or not full mode)"
    echo ""
fi

# Summary
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo ""
print_pass "All tests completed successfully!"
echo ""
print_info "Next steps:"
echo "  1. Run full tuning: python examples/helion_vector_add_tuning.py"
if [[ "$IS_B200" == true ]]; then
    echo "  2. Run B200 tuning: python examples/helion_b200_attention_tuning.py"
fi
echo "  3. Integrate OpenEvolveTuner into your kernels"
echo "  4. See TESTING_B200.md for more details"
echo ""
echo "========================================================================"
