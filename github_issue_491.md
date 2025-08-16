# Issue #491: My `helion_vmap` is crashing

## Metadata
- **State**: OPEN
- **Author**: [v0i0](https://github.com/v0i0)
- **Created**: August 12, 2025 at 23:41 UTC
- **Updated**: August 12, 2025 at 23:41 UTC

## Description

Note: Please write your bug report in English to ensure it can be understood and addressed by the development team.

**Describe the bug**
The code below attempts to provide a 1D vmap. While it is crashing, it is crashing in a way that seems fixable (a portion of code expects ComputeBuffers but gets an InputBuffer). Would be awesome to enable this level of metaprogramming!

**To Reproduce**

```
import helion
import helion.language as hl
import torch

def deduce_outputs(fn, *args):
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake_mode = FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    fake_args = [converter.from_real_tensor(fake_mode, arg) for arg in args]
    with fake_mode:
        fake_outs = fn(*fake_args)
    return fake_outs

def allocate_fakes(outs):
    return [torch.empty_like(out) for out in outs]

def tile_args(args, in_dims, tile):
    _slice = slice(None, None, None)
    tiled_args = []
    for arg, in_dim in zip(args, in_dims):
        if in_dim is None:
            tiled_args.append(arg)
        else:
            tiled_args.append(torch.exp(arg[tuple(_slice if i != in_dim else tile for i in range(arg.ndim))]))
    return tiled_args

def tile_outs(tile, global_outs, local_outs, out_dims):
    _slice = slice(None, None, None)
    for out, out_dim, local_out in zip(global_outs, out_dims, local_outs):
        assert out_dim is not None
        out[tuple(_slice if i != out_dim else tile for i in range(out.ndim))] = local_out


def helion_vmap(fn, in_dims, out_dims):
    packed_fn = fn
    if type(in_dims) is int:
        in_dims = (in_dims,)
    unpack = lambda args: args
    if type(out_dims) is int:
        out_dims = (out_dims,)
        packed_fn = lambda *args: (fn(*args),)
        unpack = lambda args: args[0]
    import functools
    @functools.wraps(fn)
    def wrapper(*args):
        outs = allocate_fakes(deduce_outputs(torch.vmap(packed_fn, in_dims, out_dims), *args))
        @helion.kernel()
        def kernel(kernel_function, kernel_args, out_dims, kernel_out_0, kernel_outs):
            for tile in hl.tile(kernel_out_0.shape[0]):
                kernel_function(tile, kernel_args, kernel_outs)

        def tiled_function(tile, kernel_args, kernel_outs):
            tiled_args = tile_args(kernel_args, in_dims, tile)
            local_outs = packed_fn(*tiled_args)
            tile_outs(tile, kernel_outs, local_outs, out_dims)

        kernel(tiled_function, args, out_dims, outs[0], outs)

        return unpack(outs)

    return wrapper

helion_vmap(torch.exp, 0, 0)(torch.zeros(100, 100, device='cuda'))
```

```
[...]
File ~/.bento/kernels/bento_kernel_helion/57/bento_kernel_helion_binary-inplace#link-tree/helion/_compiler/inductor_lowering.py:178, in prepare_node_lowering(graph_lowering, node)
    174             raise InductorLoweringError(
    175                 f"Lowering {node.target} returned {type(r)}, expected TensorBox(StorageBox(...)): {r}"
    176             )
    177         if not isinstance(buffer := r.data.data, ComputedBuffer):
--> 178             raise InductorLoweringError(
    179                 f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer: {buffer}"
    180             )
    181         buffer_name_to_output_index[buffer.get_name()] = i
    183 new_buffers = graph_lowering.buffers[prior_buffers:]
InductorLoweringError: Lowering aten.slice.Tensor returned buffer type <class 'torch._inductor.ir.InputBuffer'>, expected ComputedBuffer: InputBuffer(name='slice_1_input0', layout=FixedLayout('cuda:0', torch.float32, size=[s41, s49], stride=[s49, 1]))
```

**Expected behavior**
It works

**Versions**
bento helion kernel / main

**Additional context**
-

## Comments

*No comments yet.*
