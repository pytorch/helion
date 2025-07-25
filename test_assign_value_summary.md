# Test Assignment Value Summary

## Test Cases

- [ ] buf[:,i] = 0.0 - Assign to slice with specific index
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[1]
    for i in hl.grid(N):
        buf[:,i] = 0.0
    return buf
```

- [ ] buf[:,:] = 0.0 - Assign to full slice
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[1]
    for i in hl.grid(N):
        buf[:,:] = 0.0
    return buf
```

- [ ] buf[i] = 0.0 - Simple 1D index assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[i] = 0.0
    return buf
```

- [ ] buf[:] = 0.0 - Assign to full 1D slice
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[:] = 0.0
    return buf
```

- [ ] buf[:] = zeros[i] - Assign slice from indexed value
```python
@helion.kernel()
def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[:] = zeros[i]
    return buf
```

- [ ] buf[i] = zeros[i] - Index to index assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[i] = zeros[i]
    return buf
```

- [ ] buf[i,:] = 0.0 - Mixed index and slice in 2D
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[i,:] = 0.0
    return buf
```

- [ ] buf[i:i+1] = 0.0 - Partial slice assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N-1):
        buf[i:i+1] = 0.0
    return buf
```

- [ ] buf[::2] = 0.0 - Strided slice assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[::2] = 0.0
    return buf
```

- [ ] buf[-1] = 0.0 - Negative index assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[-1] = 0.0
    return buf
```

- [ ] buf[..., i] = 0.0 - Ellipsis indexing
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[-1]
    for i in hl.grid(N):
        buf[..., i] = 0.0
    return buf
```

- [ ] buf[mask] = 0.0 - Boolean mask assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[mask] = 0.0
    return buf
```

- [ ] buf[indices] = 0.0 - Advanced indexing with tensor
```python
@helion.kernel()
def kernel(buf: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[indices] = 0.0
    return buf
```

- [ ] buf[:, :, i] = 0.0 - Multiple dimension slicing
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[-1]
    for i in hl.grid(N):
        buf[:, :, i] = 0.0
    return buf
```

- [ ] buf[i::N//4] = 0.0 - Slice with step from index
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N//4):
        buf[i::N//4] = 0.0
    return buf
```

- [ ] buf[i] = tensor_val - Assign tensor value to index
```python
@helion.kernel()
def kernel(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[i] = val
    return buf
```

- [ ] buf[:] = zeros[:] - Full slice to slice assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N):
        buf[:] = zeros[:]
    return buf
```

- [x] buf[:, i] = scalar_tensor - Broadcast scalar assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    N = buf.shape[1]
    for i in hl.grid(N):
        buf[:, i] = val
    return buf
```

- [ ] buf[i:i+2] = 0.0 - Range slice assignment
```python
@helion.kernel()
def kernel(buf: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N-2):
        buf[i:i+2] = 0.0
    return buf
```

- [ ] buf[i] = zeros[i:i+1] - Assign from slice
```python
@helion.kernel()
def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    N = buf.shape[0]
    for i in hl.grid(N-1):
        buf[i] = zeros[i:i+1]
    return buf
```