"""Prepare the pinned environments and bundled Helion source packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((ROOT / "pins.json").read_text())
    observed = subprocess.check_output(
        [
            str(args.python),
            "-c",
            "import platform; print(platform.python_version()); print(platform.machine())",
        ],
        text=True,
    ).splitlines()
    assert observed == [pins["python"], pins["architecture"]], observed
    for mode in ("forward", "backward"):
        source = pins["sources"][mode]
        archive = ROOT / source["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == source["sha256"]
        engine = ROOT / ".engines" / mode
        engine.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as package:
            package.extractall(engine, filter="data")
        python = ROOT / ".envs" / mode / "bin/python"
        if not python.is_file():
            subprocess.run(
                ["uv", "venv", "--python", str(args.python), str(python.parent.parent)],
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "-r",
                str(ROOT / f"requirements-{mode}.txt"),
            ],
            check=True,
        )
        print(f"Prepared {mode}: {python}")


if __name__ == "__main__":
    main()
