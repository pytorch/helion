"""
AOT Standalone Compilation
==========================

Generates a standalone ``.py`` file from Helion kernels that has zero
Helion dependencies at runtime.  The output contains only Triton code,
a heuristic dispatcher, and standard ``torch`` / ``triton`` imports.

Usage::

    python -m helion.experimental.aot_runner --standalone \\
        -- python examples/aot_compile_example.py

Writes ``<source>_<kernel>_standalone.py`` next to each kernel source file.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

log: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helion runtime helpers inlined into standalone output
# ---------------------------------------------------------------------------

_INLINED_LAUNCHER = """
def _default_launcher(
    triton_kernel,
    grid,
    *args,
    num_warps,
    num_stages,
    launch_cooperative_grid=False,
    **kwargs,
):
    return triton_kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        **kwargs,
    )
"""

_INLINED_GET_NUM_SM = """
def _get_num_sm(device, reserved_sms=0):
    if device.type == "cuda":
        available = torch.cuda.get_device_properties(device.index).multi_processor_count
    elif device.type == "xpu":
        available = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    else:
        raise NotImplementedError(f"_get_num_sm not implemented for {device.type}")
    if reserved_sms <= 0:
        return available
    return max(available - reserved_sms, 1)
"""


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_imports_and_body(code: str) -> tuple[list[str], str]:
    """Split generated Triton code into import lines and everything after."""
    lines = code.split("\n")
    imports: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("from __future__", "import ", "from ")):
            imports.append(stripped)
            body_start = i + 1
        elif stripped == "" or stripped.startswith("#"):
            body_start = i + 1
        else:
            break
    return imports, "\n".join(lines[body_start:])


def _replace_helion_deps(body: str) -> str:
    """Replace ``helion.runtime.get_num_sm(`` with ``_get_num_sm(``."""
    return body.replace("helion.runtime.get_num_sm(", "_get_num_sm(")


def _rename_config_symbols(body: str, kernel_name: str, config_idx: int) -> str:
    """
    Rename module-level symbols so multiple configs coexist in one file.

    Appends ``_c<N>`` to: ``_helion_<kernel>``, the host wrapper
    ``def <kernel>(``, and every ``tl.constexpr`` constant.
    """
    sfx = f"_c{config_idx}"

    body = body.replace(f"_helion_{kernel_name}", f"_helion_{kernel_name}{sfx}")

    body = re.sub(
        rf"^(def ){kernel_name}\(",
        rf"\g<1>_{kernel_name}{sfx}(",
        body,
        flags=re.MULTILINE,
    )

    # Word-boundary rename, longest first so _BLOCK_SIZE_0_1 is renamed
    # before _BLOCK_SIZE_0.
    constexpr_names = re.findall(
        r"^(_[A-Z][A-Z0-9_]*)\s*=\s*tl\.constexpr\(", body, re.MULTILINE
    )
    for name in sorted(constexpr_names, key=len, reverse=True):
        body = re.sub(rf"\b{re.escape(name)}\b", f"{name}{sfx}", body)

    return body


def _extract_heuristic_body(heuristic_code: str, kernel_name: str) -> str:
    """
    Extract the config-index selection logic from generated heuristic code.

    Keeps ``key_<kernel>`` (decision-tree backend) or ``_extract_features``
    / ``_predict`` (nearest-neighbor backend).  Strips everything else.
    """
    lines = heuristic_code.split("\n")
    out: list[str] = []

    in_docstring = False
    in_multiline_list = False
    skip_fn: str | None = None
    skip_header = True

    for line in lines:
        stripped = line.strip()

        # --- docstrings ---
        if stripped.startswith('"""'):
            if stripped.count('"""') >= 2 and len(stripped) > 3:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # --- leading imports / blanks / comments ---
        if skip_header and (
            stripped.startswith(("import ", "from ", "#")) or stripped == ""
        ):
            continue
        skip_header = False

        # --- module-level constants ---
        if stripped.startswith(("CONFIGS", "FEATURE_NAMES", "USED_FEATURES")):
            if "[" in stripped and "]" not in stripped:
                in_multiline_list = True
            continue
        if in_multiline_list:
            if "]" in stripped:
                in_multiline_list = False
            continue

        # --- functions: keep only those relevant to this kernel ---
        if stripped.startswith("def "):
            keep_prefixes = (
                f"def key_{kernel_name}(",
                "def _extract_features(",
                "def _predict(",
                "def _get_dtype_cat(",
            )
            if not any(stripped.startswith(p) for p in keep_prefixes):
                skip_fn = stripped
                continue
        if skip_fn is not None:
            if stripped and not line[0].isspace():
                skip_fn = None
            else:
                continue

        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_standalone_file(
    kernel_name: str,
    triton_codes: list[str],
    heuristic_code: str,
    output_dir: Path,
    kernel_source_file: str | None = None,
) -> Path:
    """
    Generate a single standalone ``.py`` file with no Helion dependencies.

    Each config's symbols get a ``_c<N>`` suffix to avoid collisions.
    A public ``<kernel>`` function dispatches to the right variant.

    Args:
        kernel_name: Name of the kernel function.
        triton_codes: Triton code strings, one per selected config.
        heuristic_code: Generated heuristic Python source.
        output_dir: Fallback directory when *kernel_source_file* is ``None``.
        kernel_source_file: When set, writes next to the source file.

    Returns:
        Path to the generated file.
    """
    n = len(triton_codes)

    # -- collect imports & bodies -------------------------------------------
    all_imports: set[str] = set()
    bodies: list[str] = []
    needs_launcher = False
    needs_get_num_sm = False

    for i, code in enumerate(triton_codes):
        imports, body = _split_imports_and_body(code)
        for imp in imports:
            if "helion" in imp:
                if "default_launcher" in imp:
                    needs_launcher = True
                continue
            all_imports.add(imp)
        if "helion.runtime.get_num_sm(" in body:
            needs_get_num_sm = True
        body = _replace_helion_deps(body)
        bodies.append(_rename_config_symbols(body, kernel_name, i))

    # -- assemble -----------------------------------------------------------
    parts: list[str] = [
        f"# Auto-generated standalone Triton kernel for '{kernel_name}'.",
        "# No Helion dependency required at runtime.",
        "",
        "from __future__ import annotations\n",
    ]
    for imp in sorted(all_imports):
        if "from __future__" not in imp:
            parts.append(imp)
    parts.append("")

    if needs_launcher:
        parts.append(_INLINED_LAUNCHER)
    if needs_get_num_sm:
        parts.append(_INLINED_GET_NUM_SM)

    sep = "=" * 65
    for i, body in enumerate(bodies):
        parts.extend([f"\n# {sep}", f"# Config {i}", f"# {sep}\n", body])

    if n > 1:
        # Heuristic dispatch for multi-config
        parts.extend([f"\n# {sep}", "# Heuristic dispatch", f"# {sep}\n"])
        parts.append(_extract_heuristic_body(heuristic_code, kernel_name))

    if n == 1:
        select_expr = "0"
    elif f"def key_{kernel_name}(" in heuristic_code:
        select_expr = f"key_{kernel_name}(*args)"
    else:
        select_expr = "_predict(_extract_features(*args))"
    parts.extend([f"\ndef {kernel_name}(*args, **kwargs):", "    return ["])
    for i in range(n):
        parts.append(f"        _{kernel_name}_c{i},")
    parts.extend([f"    ][{select_expr}](*args, **kwargs)", ""])

    content = "\n".join(parts)

    # -- write --------------------------------------------------------------
    if kernel_source_file is not None:
        source_path = Path(kernel_source_file)
        out_file = (
            source_path.parent / f"{source_path.stem}_{kernel_name}_standalone.py"
        )
    else:
        out_file = output_dir / f"{kernel_name}_standalone.py"

    out_file.write_text(content)
    log.info("Standalone file: %s", out_file)
    return out_file
