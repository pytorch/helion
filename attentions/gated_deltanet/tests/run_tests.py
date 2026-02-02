#!/usr/bin/env python
# Run the standalone tests for GatedDeltaNet

import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch

if not torch.cuda.is_available():
    print("CUDA not available, skipping tests")
    exit(0)

from test_gated_deltanet_standalone import (
    test_fused_recurrent_gated_delta_rule_vs_reference,
    test_fused_recurrent_gated_delta_rule_basic,
    test_fused_recurrent_gated_delta_rule_without_final_state,
)

# Run quick tests
print("Testing Gated Delta Rule fused recurrent vs reference...")
test_fused_recurrent_gated_delta_rule_vs_reference(B=2, T=128, H=4, K=64, V=64)
print("  Fused recurrent vs reference test passed!")

print("Testing Gated Delta Rule basic fused recurrent...")
test_fused_recurrent_gated_delta_rule_basic(B=2, T=128, H=4, K=64, V=64)
print("  Basic fused recurrent test passed!")

print("Testing Gated Delta Rule without final state...")
test_fused_recurrent_gated_delta_rule_without_final_state(B=2, T=128, H=4, K=64, V=64)
print("  Without final state test passed!")

print("\nAll Gated Delta Rule standalone tests passed!")
