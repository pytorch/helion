from __future__ import annotations

import csv
import glob
import sys

out = csv.writer(open(sys.argv[1] + "/data.csv", "w"))
out.writerow(["batch", "heads", "seqlen", "seqlen_kv", "dhead", "variant", "tflops"])
for f in glob.glob(sys.argv[1] + "/dhead_*"):
    lines = list(reversed(list(open(f))))
    i = -1
    for i in range(len(lines)):
        if lines[i].startswith("--------------"):
            i -= 1
            break
    line = lines[i].replace("(", "").replace(")", ",")
    line = line.split(",")
    if len(line) == 6:
        batch, heads, seqlen, seqlen_kv, dhead, tflops = line
    else:
        batch, heads, heads_kv, seqlen, seqlen_kv, dhead, tflops = line
        assert heads.strip() == heads_kv.strip()
    print(lines)

    variant = f.split("/")[-1].split(".log")[0].split("only_")[1]
    out.writerow(
        [
            int(batch.strip()),
            int(heads.strip()),
            int(seqlen.strip()),
            int(seqlen_kv.strip()),
            int(dhead.strip()),
            variant,
            float(tflops.strip()),
        ]
    )
