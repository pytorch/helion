"""The post-grad pass helion.inductor adds to Inductor's, for cuda_backend
"helion": Inductor's own passes keep some rewrites to Triton, which Helion
kernels can take too."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphModulePass
from torch._inductor.custom_graph_pass import get_hash_for_files
from torch._inductor.fx_passes import post_grad
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.pattern_matcher import fwd_only
from torch._inductor.pattern_matcher import init_once_fakemode
from torch._inductor.pattern_matcher import register_replacement

if TYPE_CHECKING:
    from torch._inductor.pattern_matcher import Match


def online_softmax_check(match: Match) -> bool:
    """Inductor's prepare_softmax_extra_check, which rewrites softmax to online
    softmax only for Triton ("We only have triton online softmax kernels
    currently"), for a floating-point x on CUDA (HelionPostGradPass checks the
    settings). An integer x, as in (x - x.amax(-1, keepdim=True)).exp().sum(-1),
    would sum its exps in an accumulator of x's dtype: prepare_softmax_online's
    reductions take it."""
    x = match.kwargs["x"].meta["val"]
    return x.device.type == "cuda" and x.dtype.is_floating_point


@init_once_fakemode
def online_softmax_patterns() -> PatternMatcherPass:
    """The replacement Inductor registers in its post-grad passes, as there:
    x.amax(dim, keepdim=True) and the sum of exp(x - amax) become
    prepare_softmax_online(x, dim), which lowers to online_softmax_reduce."""
    patterns = PatternMatcherPass(pass_name="helion_online_softmax")
    register_replacement(
        post_grad.prepare_softmax_pattern,  # pyrefly: ignore[bad-argument-type]
        post_grad.prepare_softmax_replacement,  # pyrefly: ignore[bad-argument-type]
        [torch.empty(4, 8)],
        scalar_workaround={"dim": -1},
        trace_fn=fwd_only,  # pyrefly: ignore[bad-argument-type]
        pass_dicts=patterns,  # pyrefly: ignore[bad-argument-type]
        extra_check=online_softmax_check,
    )
    return patterns


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
