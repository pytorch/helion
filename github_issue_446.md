# Issue #446: [helion day] Bunch of issues while trying out Helion

## Metadata
- **State**: OPEN
- **Author**: [anijain2305](https://github.com/anijain2305)
- **Created**: August 07, 2025 at 19:56 UTC
- **Updated**: August 07, 2025 at 20:13 UTC

## Description

Prioritize yourself whats important. Just writing down what I found.


### Issue 1

If there is number of tiles mismatch - then it raises an internal Helion error.  Maybe better error message?

```
Traceback (most recent call last):
  File "/home/anijain/local/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
  File "/home/anijain/local/helion/helion/_compiler/type_propagation.py", line 2060, in visit_For
    self._assign(node.target, iter_type.propagate_iter(self.origin()))
  File "/home/anijain/local/helion/helion/_compiler/type_propagation.py", line 1703, in _assign
    self._assign(elt, elements[idx])
IndexError: list index out of range

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/anijain/local/helion/examples/try_softmax.py", line 31, in <module>
    softmax(torch.randn(32, 64, device="cuda"))
  File "/home/anijain/local/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
  File "/home/anijain/local/helion/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
  File "/home/anijain/local/helion/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
  File "/home/anijain/local/helion/helion/_compiler/host_function.py", line 108, in __init__
    propagate_types(self, fake_args)
  File "/home/anijain/local/helion/helion/_compiler/type_propagation.py", line 2251, in propagate_types
    prop.visit(stmt)
  File "/home/anijain/local/helion/helion/_compiler/type_propagation.py", line 1588, in visit
    raise exc.InternalError(e) from e
helion.exc.InternalError: IndexError: list index out of range
While processing:
  File "/home/anijain/local/helion/examples/try_softmax.py", line 26, in softmax
    for tile_m, tile_n, tile_p in hl.tile(out.shape):
```



### Issue 2
This is wrong Helion code (I think?) but the error message is not helpful.
Update - the missing thing was keepdims=True in amax and sum operation. 

```

@helion.kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Simple Helion kernel wrapping PyTorch's softmax function.
    Args:
        x (torch.Tensor): Input tensor of shape [n, m].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """

    # m, n = x.shape

    out = torch.empty_like(x)

    for tile_m, tile_n in hl.tile(out.shape):
        # So, I was thinking that we can only work on tiles here, but that not true, 
        amax = torch.amax(x[tile_m, :], dim=1)
        out_rows = torch.exp(x[tile_m, :] - amax)
        out_rows = out_rows / out_rows.sum(dim=1)
        out[tile_m, :] = out_rows
    
    return out

softmax(torch.randn(32, 64, device="cuda"))
```

The message is here - https://www.internalfb.com/phabricator/paste/view/P1896098361




## Comments

*No comments yet.*
