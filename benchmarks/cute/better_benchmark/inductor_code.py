"""Write the Triton code Inductor generates for a better-benchmark repro.

Compiles the repro at one shape point (``coordinate_descent_tuning=True``,
better-benchmark's tuned configuration) in a private Inductor cache and copies
the generated output module(s) to ``--output`` (stdout by default) under a
header that lists the launched kernels (also printed when writing to
``--output``). Use it to see which Triton kernels a Helion port has to
reproduce.

Example::

    python benchmarks/cute/better_benchmark/inductor_code.py \\
        var_mean_004f9a9ea198 --output /tmp/var_mean_004f9a9ea198.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repro_id")
    parser.add_argument(
        "--shape",
        default=None,
        help="shape hash (default: the repro's selection.json point, else its first)",
    )
    parser.add_argument("--no-cd", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cache = tempfile.mkdtemp(prefix="bb-inductor-")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
    try:
        dump(args, cache)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def dump(args: argparse.Namespace, cache: str) -> None:
    # torch is imported only after main() points Inductor at the private cache.
    import torch

    shape = args.shape or common.default_shape(args.repro_id)
    instance = common.load_repro_module(args.repro_id).Repro()
    inputs = common.make_inputs(args.repro_id, shape, seed=0)
    compiled = common.compile_inductor(
        args.repro_id, shape, instance, cd=not args.no_cd
    )
    with torch.no_grad():
        compiled(*inputs)
    torch.cuda.synchronize()
    modules = []
    for path in sorted(Path(cache).rglob("*.py"), key=os.path.getmtime):
        text = path.read_text()
        if "def call(" in text and ".run(" in text:
            modules.append(text)
    if not modules:
        raise RuntimeError(f"no Inductor output module found under {cache}")
    launches = [
        line.strip().split(".run(")[0]
        for text in modules
        for line in text.splitlines()
        if ".run(" in line and not line.strip().startswith("#")
    ]
    header = (
        f"# better-benchmark repro {args.repro_id} shape {shape}\n"
        f"# Inductor {'default' if args.no_cd else 'coordinate_descent_tuning'}; "
        f"{len(launches)} kernel launch(es): {', '.join(launches)}\n"
    )
    body = header + "\n\n".join(modules)
    # The Inductor cache path in triton_heuristics(filename=...) is noise.
    body = re.sub(r"filename=os\.path\.join\([^)]*\)\),?", "", body)
    if args.output:
        args.output.write_text(body)
        print(header, end="")
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
