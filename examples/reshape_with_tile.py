import helion
import helion.language as hl
import torch

@helion.kernel()
def reshape_with_tile(x: torch.Tensor) -> torch.Tensor:
    # hl.specialize(x.size(0))
    # hl.specialize(x.size(1))
    # hl.specialize(x.size(2))
    out = x.new_empty([x.size(0)])
    for tile0 in hl.tile(x.size(0)):
        acc = hl.zeros([tile0], dtype=x.dtype)
        for tile1, tile2 in hl.tile([x.size(1), x.size(2)]):
            acc += x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)
        out[tile0] = acc
    return out

def check(m: int, n: int, k: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, n, k], device="cuda", dtype=torch.float16)

    result = reshape_with_tile(x)
    reference = x.reshape(x.size(0), -1).sum(-1)

    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-1)

    sec = do_bench(lambda: reshape_with_tile(x))
    baseline_sec = do_bench(lambda: x.reshape(x.size(0), -1).sum(-1))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


def main() -> None:
    check(100, 200, 300)


if __name__ == "__main__":
    main()
