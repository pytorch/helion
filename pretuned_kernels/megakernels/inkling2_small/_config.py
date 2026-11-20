"""Kernel configs for the Inkling megakernel variants built at serving time: a saved config's scalar choices with
tensor-descriptor indexing on the named tensors' memory ops and warp specialization of the roots whose dots read
them (the variants add memory ops, so a saved per-op indexing list does not carry over)."""

from __future__ import annotations

import json
import os
import sys

import helion

# The served scalar choices (vllm_helion_verify.py's defaults): "attn" for the attention kernels, "moe_nvfp4" /
# "moe_mxfp8" for the MoE kernels and the layer kernels of those layers.
CONFIGS = {
    name: {
        "block_sizes": [],
        "num_warps": num_warps,
        "num_stages": 3,
        "pid_type": "persistent_blocked",
        "cross_loop_pipeline": "dynamic",
        "num_sm_multiplier": 1,
    }
    for name, num_warps in (("attn", 8), ("moe_nvfp4", 4), ("moe_mxfp8", 4))
}


def tma_indexing(kern_fn, args, names_wanted, base, backend="gluon"):
    """Indexing per memory op: tensor descriptors for the loads/stores of names_wanted (codegen order)."""
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.host_function import HostFunction

    names = {}
    orig = DeviceFunction.get_indexing_strategy

    def wrap(self, index):
        tensor = sys._getframe(1).f_locals.get("tensor")
        origin = (
            HostFunction.current().tensor_to_origin.get(tensor)
            if tensor is not None
            else None
        )
        names.setdefault(
            index, origin.suggest_var_name() if origin is not None else "?"
        )
        return orig(self, index)

    DeviceFunction.get_indexing_strategy = wrap
    try:
        k = helion.kernel(
            kern_fn,
            config=helion.Config(**base, indexing="tensor_descriptor"),
            static_shapes=True,
            backend=backend,
        )
        k.bind(args).to_triton_code(k.configs[0])
    finally:
        DeviceFunction.get_indexing_strategy = orig
    order = [names.get(i) for i in range(max(names) + 1)]
    if os.environ.get("INK_PRINT_CFG"):
        print("indexing order", order, flush=True)
    seen: dict = {}
    out = []
    for n in order:
        k = seen[n] = seen.get(n, -1) + 1
        out.append(
            "tensor_descriptor"
            if n in names_wanted or f"{n}:{k}" in names_wanted
            else "pointer"
        )
    return out


def ws_auto(kern_fn, args, tma, base, backend="gluon"):
    """Warp-specialize the roots whose dots read tensor-descriptor operands."""
    spec = (
        helion.kernel(
            kern_fn,
            config=helion.Config(**{**base, "indexing": "tensor_descriptor"}),
            static_shapes=True,
            backend=backend,
        )
        .bind(args)
        .config_spec
    )
    roots = {
        spec.kernel_grid_fact.group_for_graph(f.graph_id)
        for f in spec.memory_op_facts
        if f.matmul_operand is not None and f.tensor_name in tma
    }
    return [
        True if tuple(sp.block_ids) in roots else None
        for sp in spec.range_warp_specialize
    ]


def variant_kernel(kern_fn, args, saved_config: str, tma: str):
    """helion.kernel for kern_fn with saved_config's choices (a CONFIGS name or a JSON file) and indexing / warp
    specialization for these args."""
    base = (
        dict(CONFIGS[saved_config])
        if saved_config in CONFIGS
        else json.load(open(saved_config))
    )
    for key in ("indexing", "range_warp_specializes", "load_eviction_policies"):
        base.pop(key, None)
    names = set(tma.split(","))
    base["indexing"] = tma_indexing(kern_fn, args, names, base)
    base["range_warp_specializes"] = ws_auto(kern_fn, args, names, base)
    return helion.kernel(
        kern_fn, config=helion.Config(**base), static_shapes=True, backend="gluon"
    )
