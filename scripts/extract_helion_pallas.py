"""Emit the generated Pallas source for a Helion kernel + config.

Used to populate `source/generated_pallas.py` for each Helion LLO database
entry, so the database stores a directly-runnable Pallas snapshot rather
than just the Helion DSL.

Usage:
    python scripts/extract_helion_pallas.py \\
        --example examples/attention.py \\
        --kernel attention \\
        --shapes '8x32x8192x256;8x32x8192x256;8x32x8192x256' \\
        --dtype bf16 \\
        --config '{"block_sizes": [8, 512, 512], "pallas_loop_type": "emit_pipeline", "pallas_pre_broadcast": true}' \\
        --out llo/<entry>/source/generated_pallas.py

How to also capture the LLO dump (paired artifact in the same entry):
    # On the pod, with HELION_BACKEND=pallas already set:
    rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump
    LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \\
        ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=<N> \\
        HELION_BACKEND=pallas \\
        python examples/<kernel>.py   # or your own runner that pins this config
    # Then pick the kernel's named bundle file (e.g. `custom_kernel.*-final_bundles.txt`)
    # and its companion `*-final_hlo-static-per-bundle-utilization.txt`, and copy
    # both into `llo/<entry>/{final_bundles,utilization}.txt`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_kernel(example_path: Path, kernel_name: str) -> object:
    spec = importlib.util.spec_from_file_location("hl_example", example_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hl_example"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name)


DTYPE_MAP = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "f16": "float16",
    "fp32": "float32",
    "f32": "float32",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--example", required=True, help="Path to Helion example .py")
    p.add_argument("--kernel", required=True, help="Symbol name in the example file")
    p.add_argument(
        "--shapes",
        required=True,
        help=(
            "Semicolon-separated input shapes (e.g. '8x32x8192x256;8x32x8192x256;"
            "8x32x8192x256' for attention's Q/K/V)"
        ),
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument(
        "--config",
        help=(
            "JSON dict of config kwargs. Omit to use the kernel's default "
            "config (matches running the example without autotune)."
        ),
    )
    p.add_argument("--out", help="Output path (default: stdout)")
    args = p.parse_args()

    os.environ.setdefault("HELION_BACKEND", "pallas")

    import torch  # pyrefly: ignore[missing-import]

    kernel = load_kernel(Path(args.example), args.kernel)

    torch_dtype = getattr(torch, DTYPE_MAP[args.dtype])
    tensors = []
    for shape_str in args.shapes.split(";"):
        shape = [int(x) for x in shape_str.split("x")]
        tensors.append(torch.empty(*shape, dtype=torch_dtype, device="cpu"))

    import helion  # pyrefly: ignore[missing-import]

    # pyrefly: ignore[missing-attribute]  # Helion kernel has .bind at runtime
    bound = kernel.bind(tuple(tensors))
    if args.config:
        cfg = helion.Config(**json.loads(args.config))
    else:
        cfg = bound.config_spec.default_config()
    code = bound.to_code(cfg)

    if args.out:
        Path(args.out).write_text(code)
        print(f"Wrote {len(code)} chars to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()
