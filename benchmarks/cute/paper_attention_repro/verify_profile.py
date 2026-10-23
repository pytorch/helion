"""Verify the original B200 workload definitions on CPU.

The profile records the original harness registry and source hashes. Benchmark
execution must use that harness's generators; this module supplies no alternate
mask, score, or tensor factory. Importing this module does not import PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch.nn.attention.flex_attention import BlockMask

PROFILE_PATH = Path(__file__).with_name("reference_cases.json")


class _Shape(Protocol):
    b: int
    hq: int
    hkv: int
    m: int
    n: int
    d: int
    d_v: int
    label: str
    params: dict[str, Any]


class _Variant(Protocol):
    name: str
    dtype: object
    helion_template: str
    shapes: tuple[_Shape, ...]
    make_mods: Callable[
        ...,
        tuple[Callable[..., torch.Tensor] | None, Callable[..., torch.Tensor] | None],
    ]
    make_block_mask: Callable[..., BlockMask]


class _Harness(Protocol):
    VARIANTS: dict[str, _Variant]
    vsa_topk: Callable[..., torch.Tensor]
    _flex_block_mask_vsa: Callable[..., BlockMask]
    make_qkv: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    variant_flops: Callable[..., float]


def load_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_value(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_variants(harness_root: Path) -> _Harness:
    path = harness_root / "benchmarks/helion_attn/variants.py"
    sys.path.insert(0, str(harness_root))
    name = "_paper_variants_" + _sha256(path)[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return cast("_Harness", module)


def case_metadata(
    variant: _Variant, shape: _Shape, index: int, batch: str
) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "shape_index": index,
        "label": shape.label,
        "shape": {
            "b": shape.b,
            "hq": shape.hq,
            "hkv": shape.hkv,
            "m": shape.m,
            "n": shape.n,
            "d": shape.d,
            "dv": shape.d_v,
            "dtype": str(variant.dtype).removeprefix("torch."),
        },
        "parameters": _json_value(shape.params),
        "batch": batch,
        "helion_template": variant.helion_template,
        "modifier_factory": variant.make_mods.__name__,
        "modifier_factory_sha256": hashlib.sha256(
            inspect.getsource(variant.make_mods).encode()
        ).hexdigest(),
    }


def assert_case_matches(reference: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in (
        "variant",
        "shape_index",
        "label",
        "shape",
        "parameters",
        "batch",
        "helion_template",
        "modifier_factory",
        "modifier_factory_sha256",
    ):
        assert reference[key] == actual[key], (
            f"{reference['variant']}.{reference['shape_index']} {key}: "
            f"expected {reference[key]!r}, got {actual[key]!r}"
        )


def _verify_vsa(
    module: _Harness, variant: _Variant, shape: _Shape, expected_flops: float
) -> dict[str, Any]:
    import torch

    device = torch.device("cpu")
    tile = shape.params["tile_numel"]
    top_k = shape.params["topk"]
    topk = module.vsa_topk(shape, device)
    assert tuple(topk.shape) == (shape.b, shape.hq, shape.m // tile, top_k)
    self_tiles = torch.arange(shape.m // tile)[None, None, :, None]
    assert bool((topk == self_tiles).any(dim=-1).all())
    masks = (
        variant.make_block_mask(shape, device, "TRITON"),
        module._flex_block_mask_vsa(shape, device, (128, 128)),
    )
    selected = topk.sort(dim=-1).values
    for mask in masks:
        assert tuple(mask.BLOCK_SIZE) == (128, 128)
        assert bool((mask.kv_num_blocks == 0).all())
        assert bool((mask.full_kv_num_blocks == top_k).all())
        full_kv_indices = mask.full_kv_indices
        assert full_kv_indices is not None
        assert torch.equal(full_kv_indices[..., :top_k], selected)
    flops = 2.0 * (shape.d + shape.d_v) * shape.b * shape.hq * shape.m * tile * top_k
    assert flops == expected_flops, (variant.name, shape.label, flops, expected_flops)
    return {
        "tile_numel": tile,
        "top_k": top_k,
        "self_tile_forced": True,
        "topk_sha256": hashlib.sha256(topk.numpy().tobytes()).hexdigest(),
        "block_size": [128, 128],
        "partial_blocks": 0,
        "full_blocks_per_query": top_k,
        "paper_flops": flops,
    }


def verify_profile(
    harness_root: Path, batch: str, profile: dict[str, Any]
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    for filename, digest in profile["sources"][batch]["files"].items():
        assert _sha256(harness_root / filename) == digest, filename
    module = _load_variants(harness_root)
    verified = []
    for case in profile["cases"]:
        if case["batch"] != batch:
            continue
        variant = module.VARIANTS[case["variant"]]
        shape = variant.shapes[case["shape_index"]]
        assert_case_matches(
            case, case_metadata(variant, shape, case["shape_index"], batch)
        )
        tensors = module.make_qkv(
            variant, shape, torch.device("cpu"), seed=profile["input_seed"]
        )
        dimensions = (
            (shape.b, shape.hq, shape.m, shape.d),
            (shape.b, shape.hkv, shape.n, shape.d),
            (shape.b, shape.hkv, shape.n, shape.d_v),
        )
        for tensor, dims in zip(tensors, dimensions, strict=True):
            assert tuple(tensor.shape) == dims
            assert tensor.dtype == torch.bfloat16
            assert tensor.is_contiguous()
        result: dict[str, Any] = {
            "variant": variant.name,
            "shape_index": case["shape_index"],
            "label": shape.label,
            "tensor_shapes": [list(tensor.shape) for tensor in tensors],
            "dtype": "bfloat16",
        }
        if variant.name == "vsa":
            result["vsa"] = _verify_vsa(module, variant, shape, case["paper_flops"])
        elif not variant.name.startswith("graphormer-"):
            # Invoke the original mask factories and count their actual visible
            # pairs, including rectangular STA/NAT canvases and document layouts.
            flops = module.variant_flops(variant, shape, torch.device("cpu"))
            assert flops == case["paper_flops"], (
                variant.name,
                shape.label,
                flops,
                case["paper_flops"],
            )
            result["paper_flops"] = flops
            if variant.name in {"sta2d", "sta3d"}:
                _, mask = variant.make_mods(shape, torch.device("cpu"))
                assert mask is not None
                result["generated_mask_name"] = mask.__name__
        else:
            result["score_tables"] = (
                "original generator sources verified; no CPU all-pairs preprocessing"
            )
        verified.append(result)
    expected_count = 33 if batch == "fused" else 54
    assert len(verified) == expected_count
    return {
        "status": "PASS",
        "batch": batch,
        "harness_root": str(harness_root.resolve()),
        "harness_commit": profile["sources"][batch]["harness_commit"],
        "torch": torch.__version__,
        "device": "cpu",
        "case_count": len(verified),
        "cases": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--batch", choices=("fused", "sparse"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_profile(args.harness_root, args.batch, load_profile(args.profile))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['batch']}: {result['case_count']} cases PASS on CPU ({result['torch']})"
    )


if __name__ == "__main__":
    main()
