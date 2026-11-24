"""Build the experiment's CPython 3.12.14 and its local library dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".bootstrap"
DEPS = WORK / "dependencies"
PREFIX = WORK / "python-3.12.14"
JOBS = 8


def run(command: list[str], source: Path, env: dict[str, str], log: object) -> None:
    subprocess.run(
        command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
    )


def fetch(entry: dict) -> Path:
    downloads = WORK / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / entry["url"].rsplit("/", 1)[1]
    if not archive.is_file():
        urllib.request.urlretrieve(entry["url"], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry["sha256"]
    build = WORK / "build"
    build.mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        directory = build / package.getnames()[0].split("/")[0]
        if not directory.is_dir():
            package.extractall(build, filter="data")
    return directory


def main() -> None:
    lock = json.loads((ROOT / "python-build-lock.json").read_text())
    python = PREFIX / "bin/python3.12"
    ready = PREFIX / ".ready"
    if ready.is_file():
        print(python)
        return
    WORK.mkdir(exist_ok=True)
    logs = WORK / "logs"
    logs.mkdir(exist_ok=True)
    DEPS.mkdir(exist_ok=True)
    env = dict(os.environ)
    for name in ("zlib", "libffi", "bzip2", "xz", "sqlite"):
        marker = DEPS / (name + ".installed")
        if marker.is_file():
            continue
        print(f"Building Python dependency {name}", file=sys.stderr, flush=True)
        source = fetch(lock["dependencies"][name])
        if name == "zlib":
            configure = ["./configure", f"--prefix={DEPS}"]
        elif name == "libffi":
            configure = [
                "./configure",
                f"--prefix={DEPS}",
                f"--libdir={DEPS}/lib",
                "--disable-docs",
            ]
        elif name == "xz":
            configure = [
                "./configure",
                f"--prefix={DEPS}",
                "--disable-doc",
                "--disable-nls",
            ]
        elif name == "sqlite":
            configure = ["./configure", f"--prefix={DEPS}", "--disable-readline"]
        else:
            configure = []
        with (logs / (name + ".log")).open("w") as log:
            if configure:
                run(configure, source, env, log)
            make = ["make", f"-j{JOBS}"]
            if name == "bzip2":
                make.append("CFLAGS=-O2 -fPIC")
            run(make, source, env, log)
            install = ["make", "install"]
            if name == "bzip2":
                install.append(f"PREFIX={DEPS}")
            run(install, source, env, log)
        marker.touch()
    print("Building CPython 3.12.14", file=sys.stderr, flush=True)
    source = fetch(lock["python"])
    env.update(
        CPPFLAGS=f"-I{DEPS}/include",
        LDFLAGS=f"-L{DEPS}/lib -Wl,-rpath,{DEPS}/lib -Wl,-rpath,{PREFIX}/lib",
        PKG_CONFIG_PATH=f"{DEPS}/lib/pkgconfig:{DEPS}/lib64/pkgconfig",
    )
    with (logs / "python.log").open("w") as log:
        run(
            [
                "./configure",
                f"--prefix={PREFIX}",
                "--enable-shared",
                "--with-ensurepip=install",
            ],
            source,
            env,
            log,
        )
        run(["make", f"-j{JOBS}"], source, env, log)
        run(["make", "install"], source, env, log)
    subprocess.run(
        [
            str(python),
            "-c",
            "import bz2, ctypes, lzma, sqlite3, ssl, venv, zlib; "
            "import platform; assert platform.python_version() == '3.12.14'",
        ],
        check=True,
    )
    ready.touch()
    print(python)


if __name__ == "__main__":
    main()
