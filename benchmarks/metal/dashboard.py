"""A self-contained HTML dashboard of one run's correctness.

``render`` turns a run's ``results.json`` data into a single page with no
external dependencies: one stat tile per provider, a kernel-by-case grid of
Helion's verdicts, Helion's failures grouped by cause, and the same grid for
each baseline.  Every mark has a hover tooltip; the table view is the text
alternative.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from html import escape
import re
from typing import Any

from .manifest import BASELINES
from .manifest import PROVIDERS

# Each status maps to one of five verdicts: a status color, a glyph so no mark
# relies on color alone, and the glyph's ink.
VERDICTS = {
    "correct": ("Correct", "var(--good)", "✓", "#0b0b0b"),
    "wrong": ("Wrong result", "var(--critical)", "✕", "#ffffff"),
    "unsupported": ("Unsupported", "var(--warning)", "–", "#0b0b0b"),
    "error": ("Error", "var(--serious)", "!", "#0b0b0b"),
    "none": ("Not run", "var(--none)", "", "#0b0b0b"),
}
_VERDICT_OF = {
    "ok": "correct",
    "wrong_result": "wrong",
    "unsupported": "unsupported",
    "compile_error": "error",
    "runtime_error": "error",
    "oom": "error",
    "crash": "error",
    "timeout": "error",
    "cell failed": "error",
}
_PREFIXES = (
    "BackendUnsupported: Backend 'metal' does not support: ",
    "BackendImplementationMissing: Backend 'metal' is missing required implementation: ",
)
_CELL = 22
_GAP = 3
_LABEL_W = 150
_BLOCK_PAD = 24


def _record(cell: dict[str, Any], provider: str) -> dict[str, Any]:
    record = cell["providers"].get(provider)
    if record is not None:
        return record
    if "error" in cell:
        return {"status": "cell failed", "error": cell["error"]}
    return {"status": "not run"}


def _verdict(record: dict[str, Any]) -> str:
    return _VERDICT_OF.get(record["status"], "none")


def cause(record: dict[str, Any]) -> str:
    """A failure's first error line up to its first ';', with its numbers
    generalized, for grouping."""
    if record["status"] == "wrong_result":
        return "wrong result"
    line = (record.get("error") or record["status"]).splitlines()[0]
    if record["status"] == "cell failed":
        line = f"setup: {line}"
    for prefix in _PREFIXES:
        line = line.replace(prefix, "")
    line = re.sub(
        r"Aten lowering codegen not registered for <OpOverload\(op='([^']+)'.*",
        r"no lowering for \1",
        line,
    )
    line = line.split(";")[0]
    return re.sub(r"\b\d+\b", "N", line).strip()


def _tip(cell: dict[str, Any], provider: str) -> str:
    record = _record(cell, provider)
    lines = [f"{cell['kernel']} case {cell['index']} {cell['label']}", record["status"]]
    if record.get("error"):
        lines.append(record["error"].splitlines()[0][:240])
    return escape("\n".join(lines))


def _kernels(cells: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_kernel[cell["kernel"]].append(cell)
    return by_kernel


def _stat_tile(cells: list[dict[str, Any]], provider: str) -> str:
    counts = Counter(_verdict(_record(c, provider)) for c in cells)
    total = len(cells)
    segments = "".join(
        f'<span class="seg" style="flex:{counts[v]};background:{VERDICTS[v][1]}" '
        f'data-tip="{escape(VERDICTS[v][0])}: {counts[v]}"></span>'
        for v in VERDICTS
        if counts[v]
    )
    hero = " hero" if provider == "helion" else ""
    return (
        f'<div class="tile{hero}"><div class="tile-name">{provider}</div>'
        f'<div class="tile-num">{counts["correct"]}<span>/{total}</span></div>'
        f'<div class="bar">{segments}</div></div>'
    )


def _grid(
    cells: list[dict[str, Any]], providers: list[str], size: int, counts: bool
) -> str:
    """Kernel rows by case columns, one labelled block per provider, one shared
    label column; ``counts`` replaces the block label with per-kernel counts."""
    by_kernel = _kernels(cells)
    if not by_kernel:
        return '<p class="empty">No finished cells.</p>'
    width = max(len(v) for v in by_kernel.values())
    step = size + _GAP
    block_w = width * step + _BLOCK_PAD
    header = 0 if counts else 20
    marks = []
    for b, provider in enumerate(providers):
        left = _LABEL_W + b * block_w
        if not counts:
            marks.append(
                f'<text class="block-label" x="{left}" y="12">{provider}</text>'
            )
        for row, kernel_cells in enumerate(by_kernel.values()):
            y = header + row * step
            for col, cell in enumerate(kernel_cells):
                _, color, glyph, ink = VERDICTS[_verdict(_record(cell, provider))]
                x = left + col * step
                marks.append(
                    f'<g class="mark" data-tip="{_tip(cell, provider)}">'
                    f'<rect x="{x}" y="{y}" width="{size}" height="{size}" '
                    f'rx="{min(4, size // 4)}" fill="{color}"/>'
                    f'<text class="glyph" x="{x + size / 2}" y="{y + size * 0.74}" '
                    f'font-size="{size * 0.6:.0f}" fill="{ink}">{glyph}</text></g>'
                )
            if counts:
                ok = sum(
                    _verdict(_record(c, provider)) == "correct" for c in kernel_cells
                )
                marks.append(
                    f'<text class="row-count" x="{left + width * step + 6}" '
                    f'y="{y + size * 0.7}">{ok}/{len(kernel_cells)}</text>'
                )
    for row, kernel in enumerate(by_kernel):
        marks.append(
            f'<text class="row-label" x="{_LABEL_W - 8}" '
            f'y="{header + row * step + size * 0.7}">{escape(kernel)}</text>'
        )
    w = _LABEL_W + len(providers) * block_w + (_BLOCK_PAD if counts else 0)
    h = header + len(by_kernel) * step
    return (
        f'<svg class="grid" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="verdict per kernel and case: {", ".join(providers)}">'
        + "".join(marks)
        + "</svg>"
    )


def _causes(cells: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        record = _record(cell, "helion")
        if _verdict(record) in ("wrong", "unsupported", "error"):
            groups[cause(record)].append(cell)
    if not groups:
        return '<p class="empty">No Helion failures.</p>'
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    most = len(ordered[0][1])
    bars = []
    for name, members in ordered:
        v = _verdict(_record(members[0], "helion"))
        kernels = sorted({c["kernel"] for c in members})
        tip = escape(f"{name}\n{len(members)} cells: {', '.join(kernels)}")
        chips = "".join(f'<span class="chip">{escape(k)}</span>' for k in kernels)
        bars.append(
            f'<div class="cause" data-tip="{tip}">'
            f'<div class="cause-name">{VERDICTS[v][2]} {escape(name)}</div>'
            f'<div class="cause-track"><div class="cause-bar" style="width:'
            f'{100 * len(members) / most:.1f}%;background:{VERDICTS[v][1]}"></div>'
            f'<span class="cause-count">{len(members)}</span></div>'
            f'<div class="chips">{chips}</div></div>'
        )
    return "".join(bars)


def _legend() -> str:
    return "".join(
        f'<span class="key"><span class="swatch" style="background:{color};'
        f'color:{ink}">{glyph}</span>{escape(label)}</span>'
        for label, color, glyph, ink in VERDICTS.values()
    )


def _table(cells: list[dict[str, Any]], providers: list[str]) -> str:
    head = "".join(f"<th>{p}</th>" for p in providers)
    rows = []
    for cell in cells:
        statuses = "".join(
            f"<td>{escape(_record(cell, p)['status'])}</td>" for p in providers
        )
        rows.append(
            f"<tr><td>{escape(cell['kernel'])}</td><td>{cell['index']}</td>"
            f"<td>{escape(cell['label'])}</td>{statuses}</tr>"
        )
    return (
        "<details><summary>Table view</summary><table><thead><tr><th>kernel</th>"
        f"<th>case</th><th>shape</th>{head}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></details>"
    )


_STYLE = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#ffffff;--line:#e4e3df;
--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;--none:#dcdbd6;
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;
--panel:#222220;--line:#383835;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;
--none:#3a3a37}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 24px 48px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:15px;margin:0 0 12px;color:var(--text2)}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:22px}
.meta span{font-size:12px;color:var(--text2);background:var(--panel);
border:1px solid var(--line);border-radius:999px;padding:2px 10px}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin-bottom:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.tile{border:1px solid var(--line);border-radius:10px;padding:14px}
.tile.hero{border-color:var(--text2)}
.tile-name{font-size:12px;color:var(--text2);text-transform:uppercase;letter-spacing:.04em}
.tile-num{font-size:34px;font-weight:650;margin:2px 0 10px}
.tile-num span{font-size:16px;color:var(--muted);font-weight:500}
.bar{display:flex;gap:2px;height:10px}.seg{border-radius:4px;min-width:3px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0 0}
.key{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text2)}
.swatch{width:16px;height:16px;border-radius:4px;display:inline-flex;
align-items:center;justify-content:center;font-size:11px;font-weight:700}
.grid{display:block;max-width:100%;height:auto;overflow:visible}
.row-label{font-size:12px;fill:var(--text2);text-anchor:end}
.row-count{font-size:12px;fill:var(--muted)}
.glyph{font-weight:700;text-anchor:middle;pointer-events:none}
.mark:hover rect{stroke:var(--text);stroke-width:2}
.cause{display:grid;grid-template-columns:minmax(260px,38%) 1fr;gap:4px 14px;
padding:8px 0;border-top:1px solid var(--line)}.cause:first-child{border-top:0}
.cause-name{font-size:13px}.cause-track{display:flex;align-items:center;gap:8px}
.cause-bar{height:14px;border-radius:0 4px 4px 0;min-width:4px}
.cause-count{font-size:12px;color:var(--text2);font-weight:600}
.chips{grid-column:2;display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:11px;color:var(--text2);border:1px solid var(--line);
border-radius:999px;padding:0 8px}
.block-label{font-size:12px;fill:var(--text2);font-weight:600}
.empty{color:var(--muted)}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);
font-size:12px;padding:6px 9px;border-radius:6px;white-space:pre-line;max-width:420px;
display:none;z-index:9}
details{margin-top:4px}summary{cursor:pointer;color:var(--text2)}
table{border-collapse:collapse;font-size:12px;margin-top:10px;width:100%}
td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left}
"""

_SCRIPT = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{
const t=e.target.closest('[data-tip]');
if(!t){tip.style.display='none';return}
tip.textContent=t.dataset.tip;tip.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);
tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'});
"""


def render(data: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    """The page for a run's ``results.json`` data and its finished cells."""
    env = data["environment"]
    run = data["run"]
    providers = [p for p in PROVIDERS if p in run["providers"]]
    baselines = [p for p in BASELINES if p in run["providers"]]
    helion = (
        "<section><h2>Helion, per kernel and case</h2>"
        f"{_grid(cells, ['helion'], _CELL, counts=True)}</section>"
        f"<section><h2>Why Helion fails</h2>{_causes(cells)}</section>"
        if "helion" in providers
        else ""
    )
    baseline_grid = (
        f"<section><h2>Baselines</h2>"
        f"{_grid(cells, baselines, 12, counts=False)}</section>"
        if baselines
        else ""
    )
    meta = [
        env["chip"],
        f"macOS {env['macos']}",
        f"torch {env['torch']}",
        f"MLX {env['mlx']}",
        f"Helion {env['helion_commit']}",
        f"manifest {data['manifest']['version']}",
        f"autotune {run['autotune_effort']}",
        f"{len(cells)} cells",
    ]
    if run.get("started"):
        meta.insert(0, run["started"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helion Metal correctness</title><style>{_STYLE}</style></head>
<body><main>
<h1>Helion Metal · correctness</h1>
<div class="meta">{"".join(f"<span>{escape(m)}</span>" for m in meta)}</div>
<section><h2>Correct cells</h2>
<div class="tiles">{"".join(_stat_tile(cells, p) for p in providers)}</div>
<div class="legend">{_legend()}</div></section>
{helion}{baseline_grid}<section>{_table(cells, providers)}</section>
</main><div id="tip"></div><script>{_SCRIPT}</script></body></html>
"""
