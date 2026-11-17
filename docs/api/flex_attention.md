# flex_attention on Helion

`helion.flex_attention` runs PyTorch's
{func}`torch.nn.attention.flex_attention.flex_attention` on Helion's fused CuTe
flash kernels under {func}`torch.compile`. It needs a PyTorch nightly from
2026-11-19 or later and a Blackwell (sm100) GPU. PyTorch 2.15 has the
flex_attention choice hooks it registers with, so the import works there, but
every call keeps Triton's kernels: its Inductor leaves a template's outputs
unboxed (fixed by pytorch/pytorch#205348). Import it before the first
compiled call:

```python
import helion.flex_attention  # noqa: F401
from torch.nn.attention.flex_attention import flex_attention

out = torch.compile(flex_attention)(q, k, v, score_mod=..., block_mask=...)
```

```{eval-rst}
.. automodule:: helion.flex_attention
   :no-members:

.. autofunction:: helion.flex_attention.enable

.. autofunction:: helion.flex_attention.disable

.. autofunction:: helion.flex_attention.configure
```
