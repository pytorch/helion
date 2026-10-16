"""The 29 attention-gym variants used by the GB300 figure comparison.

The 87 cases are a representative selection, not the unpublished B200 shape
manifest. Ordinary attention uses three lengths from attention-gym's B200
preset; video, graph, and speculative layouts scale its specialized examples.
All cases use BF16, batch 4, and 16 query heads. See attngym_figure.md for the
complete shapes, numerical contract, and adaptations.

Mask and score generators are adapted from meta-pytorch/attention-gym at
4d3bdcf2ea511f335fcf789d7ca1ff88e325c395 (BSD-3-Clause; see
attention_gym.LICENSE). Torch and Helion functions share the same arithmetic;
only captured-tensor indexing changes to hl.load for Helion tracing.
"""

from __future__ import annotations

import functools
import math
import types
from typing import Any

import torch

from . import attngym_variants as agv
from .attngym_report import FIGURE_GROUPS
import helion.language as hl

UPSTREAM_COMMIT = "4d3bdcf2ea511f335fcf789d7ca1ff88e325c395"
SUITE_VERSION = 1
SDPA_VARIANTS = frozenset(
    name
    for _, names in FIGURE_GROUPS
    for name in names
    if name not in {"softcap", "sigmoid-act", "gemma2"}
)


def _torch_gather(
    tensor: torch.Tensor, indices: list[torch.Tensor | int]
) -> torch.Tensor:
    return tensor[tuple(indices)]


def _helion_gather(
    tensor: torch.Tensor, indices: list[torch.Tensor | int]
) -> torch.Tensor:
    index: list[object] = list(indices)
    return hl.load(tensor, index)


def _specialize_hook(hook: agv.ScoreMod | None) -> agv.ScoreMod | None:
    """Keep immutable case parameters constant while preserving live tensors.

    Helion normally lifts a callable's scalar closure values into runtime
    arguments. The benchmark's mask geometry, loop bounds, and score constants
    are fixed by its manifest, so specialize those cells explicitly. Tensor
    cells retain their identity and contents as ordinary captured inputs.
    """
    if hook is None:
        return None
    assert isinstance(hook, types.FunctionType)
    if hook.__closure__ is None:
        return hook
    cells = []
    for cell in hook.__closure__:
        value = cell.cell_contents
        scalar = isinstance(value, (bool, int, float, str))
        scalar_tuple = isinstance(value, tuple) and all(
            isinstance(item, (bool, int, float, str)) for item in value
        )
        cells.append(
            types.CellType(hl.constexpr(value) if scalar or scalar_tuple else value)
        )
    specialized = types.FunctionType(
        hook.__code__, hook.__globals__, hook.__name__, hook.__defaults__, tuple(cells)
    )
    specialized.__kwdefaults__ = hook.__kwdefaults__
    specialized.__annotations__ = hook.__annotations__
    return specialized


def _sta3d(
    canvas: tuple[int, int, int],
    kernel: tuple[int, int, int],
    tile: tuple[int, int, int],
) -> agv.MaskMod:
    ct, ch, cw = (c // t for c, t in zip(canvas, tile, strict=True))
    kt, kh, kw = (k // t for k, t in zip(kernel, tile, strict=True))
    tile_numel = math.prod(tile)
    lt, lh, lw = kt // 2, kh // 2, kw // 2
    rt, rh, rw = kt // 2 + kt % 2 - 1, kh // 2 + kh % 2 - 1, kw // 2 + kw % 2 - 1

    def sta3d(
        b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        qi = q // tile_numel
        ki = kv // tile_numel
        qt, qh, qw = qi // (ch * cw), (qi // cw) % ch, qi % cw
        nt, nh, nw = ki // (ch * cw), (ki // cw) % ch, ki % cw
        mt = qt.clamp(lt, ct - 1 - rt)
        mh = qh.clamp(lh, ch - 1 - rh)
        mw = qw.clamp(lw, cw - 1 - rw)
        return (
            (nt >= mt - lt)
            & (nt <= mt + rt)
            & (nh >= mh - lh)
            & (nh <= mh + rh)
            & (nw >= mw - lw)
            & (nw <= mw + rw)
        )

    return sta3d


def _graph_tables(
    b: int, heads: int, nodes: int, params: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shortest paths on an undirected ring, without cubic preprocessing.

    Ring topology follows the upstream visualization. Edge labels vary by
    batch, and the learnable bias tables vary by head. Ties take the clockwise
    path, consistently for every implementation.
    """
    max_distance, path_len, edge_types = (
        params["max_distance"],
        params["path_len"],
        params["edge_types"],
    )
    q = torch.arange(nodes, device=device, dtype=torch.int32)[:, None]
    k = torch.arange(nodes, device=device, dtype=torch.int32)[None, :]
    clockwise = (k - q) % nodes
    counterclockwise = (q - k) % nodes
    distance = torch.minimum(clockwise, counterclockwise)
    distances = distance.clamp_max(max_distance + 1).expand(b, nodes, nodes)
    lengths = distance.clamp_max(path_len).expand(b, nodes, nodes).contiguous()
    direction = torch.where(clockwise <= counterclockwise, 1, -1)
    batch = torch.arange(b, device=device, dtype=torch.int32)[:, None, None]
    paths = []
    for position in range(path_len):
        # An undirected edge is labelled by its smaller endpoint. The wrap
        # edge is assigned its own label, so both directions agree.
        start = (q + position * direction) % nodes
        stop = (start + direction) % nodes
        edge = torch.minimum(start, stop)
        edge = torch.where((start - stop).abs() == nodes - 1, nodes - 1, edge)
        paths.append(((edge[None, :, :] + batch * 3) % edge_types).to(torch.int32))
    path_types = torch.stack(paths, dim=-1)
    gen = torch.Generator(device=device).manual_seed(params["modifier_seed"])
    spatial_bias = (
        torch.randn(
            heads, max_distance + 2, generator=gen, device=device, dtype=torch.float32
        )
        * 0.1
    )
    edge_bias = (
        torch.randn(
            heads,
            path_len * edge_types,
            generator=gen,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    return distances, spatial_bias, lengths, path_types, edge_bias


def _mods(
    name: str,
    batch: int,
    heads: int,
    m: int,
    n: int,
    params: dict[str, Any],
    device: torch.device,
    *,
    helion: bool = False,
) -> tuple[agv.ScoreMod | None, agv.MaskMod | None]:
    gather = _helion_gather if helion else _torch_gather
    if name == "noop":
        return None, None
    if name in {"causal", "gqa-causal", "mla-causal"}:
        return None, agv.causal_mask
    if name == "alibi":
        return agv.make_alibi(heads), None
    if name == "rel-bias":

        def relative(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q: torch.Tensor,
            kv: torch.Tensor,
        ) -> torch.Tensor:
            return score + (q - kv)

        return relative, None
    if name == "softcap":
        return agv.make_tanh_softcap(params["softcap"]), None
    if name == "sigmoid-act":
        return agv.make_sigmoid_act(), None
    if name == "gemma2":
        return agv.make_tanh_softcap(params["softcap"]), agv.make_sliding_window_causal(
            params["window"]
        )
    if name == "sandwich":
        bias, head_scale, offset = agv._sandwich_tables(device, heads, max(m, n))

        def sandwich(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q: torch.Tensor,
            kv: torch.Tensor,
        ) -> torch.Tensor:
            return score + gather(bias, [q - kv + offset]) * gather(head_scale, [h])

        return sandwich, None
    if name == "sliding-window":
        return None, agv.make_sliding_window_causal(params["window"])
    if name == "dilated-sw":
        return None, agv.make_dilated_sliding_window(
            params["window"], params["dilation"]
        )
    if name == "prefix-lm":
        return None, agv.make_prefix_lm(params["prefix"])
    if name == "block-diffusion":
        return None, agv.make_block_diffusion(m // 2, params["block_size"])
    if name == "batchify":
        group_size = params["group_size"]

        def batchify(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            return q // group_size == kv // group_size

        return None, batchify
    if name == "natten2d":
        side = params["canvas"][0]
        return None, agv.make_natten2d(side, side, *params["kernel"])
    if name == "sta2d":
        return None, agv.make_sta2d(
            tuple(params["canvas"]), tuple(params["kernel"]), tuple(params["tile"])
        )
    if name == "sta3d":
        return None, _sta3d(
            tuple(params["canvas"]), tuple(params["kernel"]), tuple(params["tile"])
        )
    if name in {"svg-spatial", "svg-temporal"}:
        prompt, frames, tokens, width = (
            params["prompt"],
            params["frames"],
            params["tokens_per_frame"],
            params["width"],
        )
        if name == "svg-spatial":
            span = (width * tokens // params["round_width"]) * params["round_width"]
            sink = prompt + tokens if params["attn_sink"] else prompt

            def spatial(
                b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
            ) -> torch.Tensor:
                return (q < prompt) | (kv < sink) | ((q - kv).abs() < span)

            return None, spatial
        span = (width * tokens) // frames

        def temporal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            qi, ki = (q - prompt) % tokens, (kv - prompt) % tokens
            return (q < prompt) | (kv < prompt) | ((qi - ki).abs() < span)

        return None, temporal
    if name == "global-sw":
        is_global = torch.zeros(n, dtype=torch.bool, device=device)
        is_global[: params["global_prefix"]] = True
        is_global[:: params["global_stride"]] = True
        window = params["window"]

        def global_window(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            return (
                ((q - kv).abs() <= window)
                | gather(is_global, [q.clamp(0, m - 1)])
                | gather(is_global, [kv.clamp(0, n - 1)])
            )

        return None, global_window
    if name in {"document", "shared-prefix"}:
        lengths = params["document_lengths"]
        offsets = torch.tensor(
            [0, *list(_cumulative(lengths))], dtype=torch.int32, device=device
        )
        document = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32, device=device),
            torch.tensor(lengths, device=device),
        )
        if name == "document":

            def document_mask(
                b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
            ) -> torch.Tensor:
                qdoc = gather(document, [q.clamp(0, m - 1)])
                kdoc = gather(document, [kv.clamp(0, n - 1)])
                return (qdoc == kdoc) & (q >= kv)

            return None, document_mask
        prefix_documents = torch.tensor(
            params["prefix_documents"], dtype=torch.int32, device=device
        )

        def shared_prefix(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            qdoc = gather(document, [q.clamp(0, m - 1)])
            start = gather(offsets, [qdoc])
            prefix_doc = gather(prefix_documents, [qdoc])
            prefix_start = gather(offsets, [prefix_doc])
            prefix_stop = gather(offsets, [prefix_doc + 1])
            own = (kv >= start) & (kv <= q)
            prefix = (qdoc != prefix_doc) & (kv >= prefix_start) & (kv < prefix_stop)
            return own | prefix

        return None, shared_prefix
    if name == "flamingo":
        images = params["images"]
        image_tokens = n // images
        starts = torch.arange(images, dtype=torch.int32, device=device) * (m // images)
        ends = (starts + params["visible_images"] * (m // images)).clamp_max(m)
        image = torch.repeat_interleave(
            torch.arange(images, dtype=torch.int32, device=device), image_tokens
        )

        def flamingo(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            idx = gather(image, [kv.clamp(0, n - 1)])
            start, stop = gather(starts, [idx]), gather(ends, [idx])
            return (q >= start) & (q < stop)

        return None, flamingo
    if name == "vsa":
        tile_numel, top_k = params["tile_numel"], params["top_k"]
        remaining_positions = tuple(range(1, top_k))
        tiles = n // tile_numel
        gen = torch.Generator(device=device).manual_seed(params["modifier_seed"])
        topk = (
            torch.rand(
                batch, heads, m // tile_numel, tiles, generator=gen, device=device
            )
            .topk(top_k, dim=-1)
            .indices.to(torch.int32)
        )

        def vsa(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            qt = (q // tile_numel).clamp(0, m // tile_numel - 1)
            kt = kv // tile_numel
            selected = gather(topk, [b, h, qt, 0]) == kt
            for index in remaining_positions:
                selected = selected | (gather(topk, [b, h, qt, index]) == kt)
            return selected

        return None, vsa
    if name == "jetspec-tree":
        prefix = params["prefix"]
        parents = params["parents"]
        ancestors = torch.eye(m, dtype=torch.bool)
        for node, parent in enumerate(parents[1:], start=1):
            ancestors[node] |= ancestors[parent]
        ancestors = ancestors.to(device)

        def tree(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            ki = (kv - prefix).clamp(0, m - 1)
            path = gather(ancestors, [q.clamp(0, m - 1), ki])
            return (kv < prefix) | ((kv >= prefix) & (kv < prefix + m) & path)

        return None, tree
    if name == "jetspec-training":
        prefix, block = params["prefix"], params["block_size"]

        def training(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            ki = kv - prefix
            own = (
                (kv >= prefix) & (q // block == ki // block) & (ki % block <= q % block)
            )
            return (kv < prefix) | own

        return None, training
    if name in {"graphormer-spatial", "graphormer-edge"}:
        distances, spatial_bias, lengths, path_types, edge_bias = _graph_tables(
            batch, heads, m, params, device
        )
        if name == "graphormer-spatial":

            def graph_spatial(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q: torch.Tensor,
                kv: torch.Tensor,
            ) -> torch.Tensor:
                distance = gather(distances, [b, q, kv])
                return score + gather(spatial_bias, [h, distance])

            return graph_spatial, None
        path_len, edge_types = params["path_len"], params["edge_types"]
        path_positions = tuple(range(path_len))

        def graph_edge(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q: torch.Tensor,
            kv: torch.Tensor,
        ) -> torch.Tensor:
            length = gather(lengths, [b, q, kv])
            total = 0.0
            for position in path_positions:
                typ = gather(path_types, [b, q, kv, position])
                contribution = gather(edge_bias, [h, typ + position * edge_types])
                total = total + torch.where(length > position, contribution, 0.0)
            return score + total / length.clamp_min(1).to(torch.float32)

        return graph_edge, None
    raise ValueError(f"Unknown figure variant {name!r}")


def _cumulative(lengths: list[int]) -> list[int]:
    total = 0
    result = []
    for length in lengths:
        total += length
        result.append(total)
    return result


def _case(
    name: str,
    label: str,
    m: int,
    *,
    n: int | None = None,
    hkv: int = 16,
    d: int = 64,
    dv: int | None = None,
    params: dict[str, Any] | None = None,
    notes: str = "",
) -> agv.Variant:
    n = m if n is None else n
    extras = {"modifier_seed": 0, **(params or {})}
    extras["sdpa_supported"] = name in SDPA_VARIANTS
    return agv.Variant(
        name=name,
        b=4,
        hq=16,
        hkv=hkv,
        m=m,
        n=n,
        d=d,
        dtype=torch.bfloat16,
        make_mods=functools.partial(_mods, name, 4, 16, m, n, extras),
        notes=notes,
        extras=extras,
        dv=dv,
        suite="figure",
        shape_id=f"{name}.{label}",
        mask_batch=4 if name == "vsa" else 1,
        mask_heads=16 if name == "vsa" else 1,
    )


def _cases() -> dict[str, agv.Variant]:
    result = []
    for _, names in FIGURE_GROUPS:
        for name in names:
            if name == "natten2d":
                for side in (32, 64, 128):
                    result.append(
                        _case(
                            name,
                            f"c{side}x{side}",
                            side * side,
                            params={"canvas": [side, side], "kernel": [13, 13]},
                        )
                    )
                continue
            if name == "sta2d":
                for canvas in ((32, 64), (64, 64), (64, 128)):
                    result.append(
                        _case(
                            name,
                            f"c{canvas[0]}x{canvas[1]}",
                            math.prod(canvas),
                            params={
                                "canvas": list(canvas),
                                "kernel": [24, 24],
                                "tile": [8, 8],
                            },
                        )
                    )
                continue
            if name == "sta3d":
                for canvas in ((16, 16, 16), (16, 16, 32), (16, 32, 32)):
                    result.append(
                        _case(
                            name,
                            "c" + "x".join(map(str, canvas)),
                            math.prod(canvas),
                            params={
                                "canvas": list(canvas),
                                "kernel": [12, 12, 12],
                                "tile": [4, 4, 4],
                            },
                        )
                    )
                continue
            if name in {"svg-spatial", "svg-temporal"}:
                for frames in (8, 16, 32):
                    result.append(
                        _case(
                            name,
                            f"f{frames}",
                            128 + frames * 256,
                            params={
                                "prompt": 128,
                                "frames": frames,
                                "tokens_per_frame": 256,
                                "width": 2,
                                "attn_sink": True,
                                "round_width": 128,
                            },
                        )
                    )
                continue
            if name in {"jetspec-tree", "jetspec-training"}:
                m = 64 if name == "jetspec-tree" else 128
                for prefix in (2048, 4096, 8192):
                    result.append(
                        _case(
                            name,
                            f"p{prefix}",
                            m,
                            n=prefix + m,
                            params={
                                "prefix": prefix,
                                "block_size": 16,
                                "parents": [-1, *[(i - 1) // 2 for i in range(1, m)]],
                            },
                        )
                    )
                continue
            if name in {"graphormer-spatial", "graphormer-edge"}:
                for nodes in (512, 1024, 2048):
                    result.append(
                        _case(
                            name,
                            f"s{nodes}",
                            nodes,
                            params={"max_distance": 4, "path_len": 3, "edge_types": 8},
                            notes="undirected ring; precomputed shortest-path tables",
                        )
                    )
                continue
            lengths = (
                (4096, 8192, 16384)
                if name in {"document", "shared-prefix"}
                else (2048, 4096, 8192)
            )
            for seq in lengths:
                params: dict[str, Any] = {}
                hkv, d, dv, n = 16, 64, None, seq
                if name in {"causal", "gqa-causal", "mla-causal"}:
                    params["causal"] = True
                if name == "gqa-causal":
                    hkv, d = 4, 128
                if name == "mla-causal":
                    hkv, d, dv = 1, 192, 128
                    params.update({"latent_dim": 128, "rope_dim": 64})
                if name == "sigmoid-act":
                    d = 128
                if name in {"sliding-window", "gemma2"}:
                    params.update(
                        {"window": 1024, "window_size": [1024, 0], "causal": True}
                    )
                if name == "gemma2":
                    hkv, d = 4, 128
                if name in {"softcap", "gemma2"}:
                    params["softcap"] = 30.0 if name == "softcap" else 50.0
                if name == "dilated-sw":
                    params.update({"window": 512, "dilation": 2})
                if name == "prefix-lm":
                    params["prefix"] = 1024
                if name == "block-diffusion":
                    params["block_size"] = 128
                if name == "batchify":
                    params["group_size"] = 256
                if name == "global-sw":
                    params.update(
                        {"window": 512, "global_prefix": 64, "global_stride": 1024}
                    )
                if name == "document":
                    # Fixed seeded jagged lengths. The same offsets are used
                    # for all heads/batches and for FA4's varlen adapter.
                    gen = torch.Generator().manual_seed(0)
                    cuts = torch.sort(
                        torch.randperm(seq - 1, generator=gen)[:11] + 1
                    ).values
                    params["document_lengths"] = torch.diff(
                        torch.cat((torch.tensor([0]), cuts, torch.tensor([seq])))
                    ).tolist()
                    params["causal"] = True
                if name == "shared-prefix":
                    params.update(
                        {
                            "document_lengths": [seq // 8] * 8,
                            "prefix_documents": [0, 0, 0, 0, 4, 4, 4, 4],
                        }
                    )
                if name == "flamingo":
                    n = seq // 2
                    params.update({"images": 16, "visible_images": 4})
                if name == "vsa":
                    params.update({"tile_numel": 128, "top_k": 4})
                result.append(
                    _case(name, f"s{seq}", seq, n=n, hkv=hkv, d=d, dv=dv, params=params)
                )
    cases: dict[str, agv.Variant] = {}
    for case in result:
        assert case.shape_id is not None
        cases[case.shape_id] = case
    return cases


CASES = _cases()


def helion_mods(
    variant: agv.Variant, device: torch.device
) -> tuple[agv.ScoreMod | None, agv.MaskMod | None]:
    score_mod, mask_mod = _mods(
        variant.name,
        variant.b,
        variant.hq,
        variant.m,
        variant.n,
        variant.extras,
        device,
        helion=True,
    )
    return _specialize_hook(score_mod), _specialize_hook(mask_mod)


def dense_bias(variant: agv.Variant, device: torch.device) -> torch.Tensor | None:
    """Precompute SDPA's additive bias, with exact batch/head broadcasting.

    This is setup work for fixed masks and positional tables, shared with the
    other backends' BlockMask/table preparation and excluded from timing.
    Nonlinear modifiers cannot be expressed by an additive SDPA bias.
    """
    if variant.name not in SDPA_VARIANTS:
        raise ValueError(f"{variant.name} is not expressible as an SDPA bias")
    score_mod, mask_mod = variant.make_mods(device)
    if score_mod is None and mask_mod is None:
        return None
    score_batch = variant.b if variant.name.startswith("graphormer-") else 1
    score_heads = (
        variant.hq
        if variant.name
        in {"alibi", "sandwich", "graphormer-spatial", "graphormer-edge"}
        else 1
    )
    batches, heads = (
        max(score_batch, variant.mask_batch),
        max(score_heads, variant.mask_heads),
    )
    b = torch.arange(batches, device=device)[:, None, None, None]
    h = torch.arange(heads, device=device)[None, :, None, None]
    q = torch.arange(variant.m, device=device)[None, None, :, None]
    k = torch.arange(variant.n, device=device)[None, None, None, :]
    bias = torch.zeros(
        (batches, heads, variant.m, variant.n), device=device, dtype=torch.float32
    )
    if score_mod is not None:
        bias = score_mod(bias, b, h, q, k)
    if mask_mod is not None:
        bias = torch.where(mask_mod(b, h, q, k), bias, float("-inf"))
    return bias.contiguous()


def manifest() -> dict[str, Any]:
    return {
        "suite": "figure",
        "suite_version": SUITE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "description": "87 representative upstream-based cases; not the original B200 manifest",
        "cases": [
            {
                "shape_id": case.shape_id,
                "variant": case.name,
                "shape": {
                    "b": case.b,
                    "hq": case.hq,
                    "hkv": case.hkv,
                    "m": case.m,
                    "n": case.n,
                    "d": case.d,
                    "dv": case.value_dim,
                    "dtype": "bfloat16",
                },
                "parameters": case.extras,
                "notes": case.notes,
            }
            for case in CASES.values()
        ],
    }
