"""Print the ROCR_VISIBLE_DEVICES index of the GPU with the least VRAM in use.

The MI350X CI runner pods share a node's eight GPUs: the job container gets
all of /dev/dri, and pinning every job to GPU 0 stacked up to four test jobs
on one GPU while the other seven sat idle.  The pod's own GPU assignment is
not visible inside the job container, so pick the least-used GPU instead.

ROCr numbers GPU agents in KFD topology node order, skipping CPU nodes, which
is the order read here.  The chosen GPU's UUID goes to stderr so the log can
be compared against what rocminfo reports afterwards.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _properties(node: pathlib.Path) -> dict[str, int]:
    props = {}
    for line in (node / "properties").read_text().splitlines():
        key, value = line.split(maxsplit=1)
        props[key] = int(value)
    return props


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs", default="/sys", type=pathlib.Path)
    args = parser.parse_args()

    nodes = sorted(
        (args.sysfs / "class/kfd/kfd/topology/nodes").iterdir(),
        key=lambda node: int(node.name),
    )
    used = []
    for node in nodes:
        if int((node / "gpu_id").read_text()) == 0:
            continue  # CPU node
        props = _properties(node)
        device = args.sysfs / f"class/drm/renderD{props['drm_render_minor']}/device"
        vram_used = int((device / "mem_info_vram_used").read_text())
        uuid = f"GPU-{props['unique_id']:016x}"
        print(
            f"GPU {len(used)}: {uuid} {vram_used >> 20} MiB used",
            file=sys.stderr,
        )
        used.append((vram_used, len(used), uuid))
    vram_used, index, uuid = min(used)
    print(f"selected GPU {index}: {uuid}", file=sys.stderr)
    print(index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
