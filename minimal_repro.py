import torch
import helion
import helion.language as hl


@helion.kernel
def reduce_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    n = hl.specialize(n)
    final_block = torch.zeros(n, dtype=torch.float32, device=x.device)
    for outer in hl.tile(m, block_size=m):
        block_acc = torch.zeros(n, dtype=torch.float32, device=x.device)
        for inner in hl.tile(outer.begin, outer.end, block_size=m):
            block_acc += torch.sum(x[inner, :], dim=0).to(torch.float32)
        final_block[:] += block_acc
    return final_block


def main():
    torch.manual_seed(42)
    x = torch.randn((128, 5632), device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    result = reduce_rows(x)

    # Reference implementation
    reference = torch.sum(x, dim=0).float()

    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
    print("✓ Test passed!")


if __name__ == "__main__":
    main()
