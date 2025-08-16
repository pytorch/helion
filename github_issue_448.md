# Issue #448: helion crashes with torch.lerp

## Metadata
- **State**: OPEN
- **Author**: [v0i0](https://github.com/v0i0)
- **Created**: August 07, 2025 at 21:29 UTC
- **Updated**: August 07, 2025 at 21:29 UTC

## Description

Note: Please write your bug report in English to ensure it can be understood and addressed by the development team.

**Describe the bug**
helion crashes with torch.lerp

**To Reproduce**
```
@helion.kernel
def lerp(a, b, w):
    for tile_a in hl.tile(a.shape[0]):
        a[tile_a,:] = torch.lerp(a[tile_a,:], b[tile_a,:], w)
lerp(torch.zeros(1000, 1000, device="cuda"), torch.zeros(1000, 1000, device="cuda"), 0.5)
```

```
InternalError: AssertionError: While executing %full : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], %zuf0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
```

**Expected behavior**
Runs

**Versions**
helion bento kernel

**Additional context**
-

## Comments

*No comments yet.*
