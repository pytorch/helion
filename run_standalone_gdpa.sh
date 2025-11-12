#!/bin/bash
# Simple run script for standalone GDPA test
# This matches the environment variables from your original buck command

HELION_PRINT_OUTPUT_CODE=1 \
TORCH_COMPILE_FORCE_DISABLE_CACHES=1 \
TRITON_LOCAL_BUILD=1 \
HELION_AUTOTUNE_LOG_LEVEL=DEBUG \
HELION_SKIP_CACHE=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python standalone_gdpa.py 2>&1 | tee /tmp/helion_standalone.log

echo ""
echo "Output saved to /tmp/helion_standalone.log"
