import torch
import importlib
import argparse

# Import the tritonbench operator
tritonbench_module = importlib.import_module("tritonbench.operators.jagged_softmax.operator")
BenchmarkOperator = tritonbench_module.Operator

# Import the helion kernel
helion_module = importlib.import_module("examples.jagged_softmax")
helion_fn = getattr(helion_module, "jagged_softmax_tritonbench")

# Create a minimal argparse namespace
tb_args = argparse.Namespace(
    mode="fwd",
    device="cuda",
    precision="fp32",
    metrics="latency",
    warmup=1,
    iter=1,
    only=None,
    skip=None,
    baseline=None,
    input_id=0,
    num_inputs=1,
    cudagraph=False,
    inductor=False,
    B=32,
    M=8,
    seqlen=64,
    sparsity=0.5
)

# Create operator instance
print("Creating operator...")
op = BenchmarkOperator(tb_args=tb_args, extra_args=[])

print(f"Input shape info: B={op.B}, M={op.M}, seqlen={op.max_seqlen}")

# Get the input
print("Getting input...")
input_tensor = op.get_input()
print(f"Input type: {type(input_tensor)}")
print(f"Input values shape: {input_tensor.values().shape}")
print(f"Input offsets shape: {input_tensor.offsets().shape}")

# Test helion kernel
print("\nTesting Helion kernel...")
try:
    import time
    start = time.time()
    result = helion_fn(input_tensor, op.B, op.M, op.max_seqlen, op.sparsity)
    end = time.time()
    print(f"Result shape: {result.shape}")
    print(f"Helion kernel time: {(end-start)*1000:.3f}ms")
    print("Helion kernel works!")
except Exception as e:
    print(f"Error in Helion kernel: {e}")
    import traceback
    traceback.print_exc()

# Test reference implementation
print("\nTesting reference implementation...")
ref_fn = op.torch_jagged_softmax_unbind_torch_softmax
start = time.time()
ref_result = ref_fn()
end = time.time()
print(f"Reference result shape: {ref_result.values().shape}")
print(f"Reference time: {(end-start)*1000:.3f}ms")

# Check if results match
if result is not None:
    print("\nChecking accuracy...")
    torch.testing.assert_close(result, ref_result.values(), rtol=1e-4, atol=1e-4)
    print("Accuracy test PASSED!")