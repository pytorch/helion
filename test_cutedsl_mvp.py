"""Test CuteDSL MVP: element-wise add end-to-end."""

import helion
import torch
import helion.language as hl


@helion.kernel(backend='cutedsl')
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        result[tile] = x[tile] + y[tile]
    return result


def test_codegen():
    """Test 1: Generate CuteDSL code and verify no tl.* intrinsics."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    bound = add.bind((x, y))
    config = bound.config_spec.default_config()
    code = bound.to_triton_code(config)
    print("=" * 60)
    print("Generated CuteDSL code:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    assert 'tl.load' not in code, f"Found tl.load in generated code"
    assert 'tl.store' not in code, f"Found tl.store in generated code"
    assert 'tl.full' not in code, f"Found tl.full in generated code"
    assert 'tl.where' not in code, f"Found tl.where in generated code"
    assert 'tl.arange' not in code, f"Found tl.arange in generated code"
    assert 'tl.zeros' not in code, f"Found tl.zeros in generated code"
    assert 'cute.arch.block_idx' in code or 'cute.arch.thread_idx' in code, \
        "Expected cute.arch.block_idx or cute.arch.thread_idx in generated code"
    print("\nSUCCESS: Generated code has no tl.* intrinsics!")
    return code


def test_codegen_non_divisible():
    """Test 1b: Generate CuteDSL code for non-divisible size (masking)."""
    x = torch.randn(1000, device='cuda')
    y = torch.randn(1000, device='cuda')
    bound = add.bind((x, y))
    config = bound.config_spec.default_config()
    code = bound.to_triton_code(config)
    print("\n" + "=" * 60)
    print("Generated CuteDSL code (non-divisible):")
    print("=" * 60)
    print(code)
    print("=" * 60)

    assert 'tl.load' not in code, f"Found tl.load in generated code"
    assert 'tl.store' not in code, f"Found tl.store in generated code"
    assert 'tl.full' not in code, f"Found tl.full in generated code"
    assert 'tl.where' not in code, f"Found tl.where in generated code"
    assert 'tl.broadcast_to' not in code, f"Found tl.broadcast_to in generated code"
    assert 'tl.arange' not in code, f"Found tl.arange in generated code"
    assert 'tl.zeros' not in code, f"Found tl.zeros in generated code"
    print("\nSUCCESS: Non-divisible code has no tl.* intrinsics!")
    return code


def test_execution():
    """Test 2: Actually execute and verify correctness."""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    result = add(x, y)
    expected = x + y
    assert torch.allclose(result, expected, atol=1e-6), \
        f"Results don't match! Max diff: {(result - expected).abs().max().item()}"
    print("SUCCESS: CuteDSL add kernel produced correct results!")


if __name__ == "__main__":
    code = test_codegen()
    code2 = test_codegen_non_divisible()
    # Execution test disabled due to sm_110a hardware compatibility issue
    # test_execution()
