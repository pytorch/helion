# flex_attention on Helion

`helion.flex_attention` runs PyTorch's
{func}`torch.nn.attention.flex_attention.flex_attention` on Helion's fused CuTe
flash kernels under {func}`torch.compile`. It needs a Blackwell (sm100) GPU and
a PyTorch whose Inductor lets an `InductorChoices` contributor lower a whole
flex_attention call (`InductorChoices.lower_flex_attention`); importing it on
an older PyTorch raises `ImportError`. Importing registers Helion as a
flex_attention backend, at any time before or between compiles:

```python
import helion.flex_attention  # noqa: F401
from torch.nn.attention.flex_attention import flex_attention

out = torch.compile(flex_attention)(q, k, v, score_mod=..., block_mask=...)

# Only Helion's kernels, with a config for this call:
out = torch.compile(flex_attention)(
    q, k, v, kernel_options={
        "BACKEND": "HELION",
        "BACKEND_OPTIONS": {"forward_config": {...}},
    },
)
```

```{eval-rst}
.. automodule:: helion.flex_attention
   :no-members:

.. autofunction:: helion.flex_attention.enable

.. autofunction:: helion.flex_attention.disable

.. autofunction:: helion.flex_attention.configure
```
