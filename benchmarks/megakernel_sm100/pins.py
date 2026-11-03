"""Record (and check) the versions one environment of this benchmark runs.

    python pins.py --system helion --out results/pins_helion.json [--strict]

Each system's environment records its own Python packages and source checkouts;
``--strict`` fails on a version that differs from ``EXPECTED`` (the versions
the checked-in numbers were measured with).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "helion": {"torch": "2.14.1+cu130", "triton": "3.8.0", "flashinfer": "0.7.0.post1"},
    # vLLM main@6a95c89c37 from its wheel (the version names the commit).  It
    # pins flashinfer 0.7.2.post1, whose TRT-LLM MoE cubins could not be
    # downloaded on the machine the numbers were taken on (README.md).
    "vllm": {
        "vllm_version": "0.33.0rc1.dev375+g6a95c89c3",
        "torch": "2.14.0+cu130",
        "triton": "3.8.0",
        "flashinfer": "0.7.0.post1",
    },
    "sglang": {
        "sglang": "02b983603",
        "torch": "2.14.1+cu130",
        "triton": "3.8.0",
        "flashinfer": "0.7.0.post1",
    },
}


def _commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short=9", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def collect(system: str) -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("torch", "triton", "flashinfer"):
        try:
            found[name] = importlib.import_module(name).__version__
        except ImportError:
            found[name] = None
    import torch

    found["cuda"] = torch.version.cuda
    if system == "helion":
        found["helion"] = _commit(HERE.parents[1])
    elif system in ("vllm", "sglang"):
        # This directory's sglang/ runner would shadow the sglang package.
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
        module = importlib.import_module(system)
        found[f"{system}_version"] = getattr(
            module, "__version__", None
        ) or importlib.metadata.version(system)
        # The source checkout the package is installed from (editable install;
        # SGLang's package is a namespace package, so use its path).
        found[system] = _commit(Path(next(iter(module.__path__))).resolve())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    cli = parser.parse_args()
    found = collect(cli.system)
    mismatches = {
        name: (expected, found.get(name))
        for name, expected in EXPECTED[cli.system].items()
        if not (found.get(name) or "").startswith(expected)
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(
        json.dumps(
            {"system": cli.system, "found": found, "mismatches": mismatches}, indent=1
        )
    )
    for name, (expected, actual) in mismatches.items():
        print(f"{cli.system}: {name} is {actual}, the recorded numbers used {expected}")
    if mismatches and cli.strict:
        raise SystemExit(1)
    print(f"{cli.system}: " + ", ".join(f"{k}={v}" for k, v in found.items()))


if __name__ == "__main__":
    main()
