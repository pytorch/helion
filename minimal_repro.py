import torch
import helion
import helion.language as hl


@helion.kernel
def reduce_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    n = hl.specialize(n)

    m_block = hl.register_block_size(m)

    final_block = torch.zeros(n, dtype=torch.float32, device=x.device)

    for outer in hl.tile(m, block_size=m_block):
        block_acc = torch.zeros(n, dtype=torch.float32, device=x.device)
        for inner in hl.tile(outer.begin, outer.end):
            block_acc += torch.sum(x[inner, :], dim=0)
        final_block[:] = block_acc
    return final_block


def main():
    x = torch.randn((128, 5632), device="cuda", dtype=torch.float16)
    reduce_rows(x)


if __name__ == "__main__":
    main()
