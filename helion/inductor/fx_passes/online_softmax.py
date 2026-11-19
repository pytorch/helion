"""The online-softmax rewrite that HelionPostGradPass applies. Inductor
registers it in its own post-grad passes for Triton kernels only, and Helion
kernels can take it too."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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
