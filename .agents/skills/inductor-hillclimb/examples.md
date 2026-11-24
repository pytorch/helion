# Pull request description examples

One example per change leads the description (SKILL.md, Step 4). These are
abridged from real ones; the full descriptions are the linked pull requests.
Each changed line carries `# <-`, and its explanation sits on the line above.

## A helion.inductor change: PyTorch program, Helion kernels before and after

From https://github.com/pytorch/helion/pull/4584 (`gather_subscript()`):

**An embedding lookup: the table read at the value**

```python
def fn(i, w):  # i: [64, 32] int64, w: [1000, 256]
    return F.embedding(i, w) * 2
```

Before this pull request:

```python
@helion.kernel(backend="triton", static_shapes=True)
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([2048, 256]):
        tmp0 = in0[tile_y]
        tmp1 = tmp0.to(torch.int64)
        # a flat gather into the whole table
        tmp2 = in1[tile_x.index + 256*tmp1.reshape([tile_y, 1])]  # <-
        tmp3 = hl.full([], 2.0, dtype=torch.float32)
        tmp4 = tmp2 * tmp3
        out0[tile_y, tile_x] = tmp4
```

After:

```python
@helion.kernel(backend="triton", static_shapes=True)
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([2048, 256]):
        tmp0 = in0[tile_y]
        tmp1 = tmp0.to(torch.int64)
        # w's [1000, 256] view at the value (gather_subscript())
        tmp2 = in1[tmp1, tile_x]  # <-
        tmp3 = hl.full([], 2.0, dtype=torch.float32)
        tmp4 = tmp2 * tmp3
        out0[tile_y, tile_x] = tmp4
```

## An FX pass: PyTorch before and after the pass

From https://github.com/pytorch/helion/pull/4585 (`split_cat_reductions()`):

**Softmax with attention sinks: the cat's parts reduced apart**

```python
def fn(scores, sinks):  # scores: [8, 64, 100], sinks: [8, 1, 1]
    x = torch.cat([scores, sinks.expand(8, 64, 1)], -1)  # rdims of 101
    return torch.softmax(x, -1)[..., :-1].contiguous()
```

After split_cat_reductions(), the graph computes:

```python
def fn(scores, sinks):
    sink = sinks.expand(8, 64, 1)
    # amax(cat) = the maximum of the parts' amax
    m = torch.maximum(scores.amax(-1, keepdim=True), sink)  # <-
    e = torch.exp(scores - m)
    # sum(cat) = the sum of the parts' sums
    s = e.sum(-1, keepdim=True) + torch.exp(sink - m)  # <-
    # [..., :-1] keeps the scores' part: no rdim of 101
    return (e / s).contiguous()  # <-
```

## A Helion compiler change: Helion kernel, generated code before and after

The generated CuTe (or Triton) code, abridged to the changed lines and a few
around them (`<skill>/scripts/pr_examples.py` does this). From the CuTe lane jam of
https://github.com/pytorch/helion/pull/4579:

**A one-axis tile's lanes, jammed**

```python
@helion.kernel(backend="cute", static_shapes=True)
def dot_rdim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # a, b: [64, 512] bf16
    out = torch.empty([a.size(0)], dtype=torch.float32, device=a.device)
    for tile_x in hl.tile(a.size(0)):
        out[tile_x] = (a[tile_x, :].to(torch.float32) * b[tile_x, :].to(torch.float32)).sum(-1)
    return out
# config: block_sizes=[8], num_threads=[1, 128], reduction_loops=[512], cute_grid_lane_jam=True
```

Before:

```python
...
    # the tile's 8 x indices, one after another
    for lane_0 in range(8):  # <-
        offsets_0 = offsets_base_0 + cutlass.Int32(lane_0)  # <-
        indices_0 = offsets_0  # <-
        sum_1_acc = cutlass.Float32(0)  # <-
...
                # a scalar load per x
                load = (a.iterator + cutlass.Int32(indices_0) * ...).load()  # <-
...
```

After:

```python
...
    # the 8 x indices jammed into one pass
    offsets_0_lane0 = offsets_base_0 + cutlass.Int32(0)  # <-
    offsets_0_lane1 = offsets_base_0 + cutlass.Int32(1)  # <-
...
    offsets_0_lane7 = offsets_base_0 + cutlass.Int32(7)  # <-
...
```

## Below the examples

A table of the graphs the change affects, timed in one process on one GPU
(the GPU, backend and autotune effort named above it):

| repro | before (us) | after (us) | baseline, e.g. inductor_cd (us) |
|---|---:|---:|---:|

Then the details, `Generality:`, `Numerics:` and `Testing:`.
