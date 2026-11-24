"""Graph passes helion.inductor adds to Inductor's, for cuda_backend "helion":
rewrites of the post-grad FX graph that run before lowering. post_grad.py
holds the pass Inductor runs (HelionPostGradPass), and each pattern family it
applies has a module of its own (cat_reductions.py, online_softmax.py)."""

from __future__ import annotations
