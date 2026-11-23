"""The post-grad pass helion.inductor adds to Inductor's, for cuda_backend
"helion": Inductor's own passes keep some rewrites to Triton, which Helion
kernels can take too."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.custom_graph_pass import get_hash_for_files

from .online_softmax import online_softmax_patterns

if TYPE_CHECKING:
    import torch


class HelionPostGradPass(CustomGraphModulePass):
    """Runs after Inductor's post-grad passes on a graph with a CUDA tensor, as
    the CUDA device's custom backend pass, after the one that was registered
    before it (`inner`). With cuda_backend "helion" it rewrites softmax to
    online softmax, as Inductor does for Triton: torch.softmax(x, -1) is then
    one pass over each row for the max and the sum of exps (online_softmax_reduce),
    and one that writes exp(x - max) / sum, rather than three."""

    def __init__(self, inner: CustomGraphModulePass | None) -> None:
        self.inner = inner

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        if self.inner is not None:
            self.inner(gm)
        # As Inductor applies its replacement, under pattern_matcher too.
        if not (
            inductor_config.cuda_backend == "helion"
            and inductor_config.online_softmax
            and inductor_config.pattern_matcher
        ):
            return
        if online_softmax_patterns().apply(gm.graph):
            gm.graph.lint()
            gm.recompile()

    def uuid(self) -> object:
        # Inductor's FX graph cache key holds it, with cuda_backend,
        # online_softmax and pattern_matcher, which __call__ reads. None, which
        # asks to skip caching, stays None.
        inner = None
        if self.inner is not None and (inner := self.inner.uuid()) is None:
            return None
        return (get_hash_for_files((__file__,)), inner)
