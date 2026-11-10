"""Load a verified exported bundle after the caller checks its base runtime."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import site
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# Python 3.10 creates namespace loaders without exposing their class through
# importlib.machinery. Derive that type through the public module factory so
# import-time validation works on every supported Python version.
_NAMESPACE_LOADER = type(
    importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("_sdg_namespace", loader=None, is_package=True)
    ).__loader__
)

_PREFIXES = ("helion", "cutlass", "cutlass_dsl", "nvidia_cutlass_dsl")


def _extension_owns_child(name: str, module: object, vendor: Path) -> bool:
    """Embedded pybind modules inherit the identity of their loaded extension."""
    while "." in name:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None or getattr(parent, attribute, None) is not module:
            return False
        filename = getattr(parent, "__file__", None)
        if filename:
            return isinstance(
                parent.__loader__, importlib.machinery.ExtensionFileLoader
            ) and Path(filename).resolve().is_relative_to(vendor)
        name, module = parent_name, parent
    return False


def _verify_loaded_origins(vendor: Path, *, allow_namespace: bool = False) -> None:
    for name, module in tuple(sys.modules.items()):
        if module is None or not any(
            name == prefix or name.startswith(prefix + ".") for prefix in _PREFIXES
        ):
            continue
        if (
            allow_namespace
            and name == "nvidia_cutlass_dsl"
            and type(module.__loader__) is _NAMESPACE_LOADER
            and module.__spec__ is not None
            and module.__spec__.origin is None
            and getattr(module, "__file__", None) is None
        ):
            # Python's site startup may have imported this code-free namespace.
            # Its children still undergo the ordinary origin checks below.
            continue
        origins = []
        filename = getattr(module, "__file__", None)
        if filename:
            origins.append(filename)
        origins.extend(getattr(module, "__path__", ()))
        if not origins and _extension_owns_child(name, module, vendor):
            continue
        if not origins or any(
            not Path(origin).resolve().is_relative_to(vendor) for origin in origins
        ):
            raise RuntimeError(
                f"SDG bundle cannot use already loaded package {name!r} from "
                f"{origins!r}; start a fresh interpreter with this bundle's dependencies."
            )


def _verify_files(root: Path) -> None:
    manifest = json.loads((root / "provenance.json").read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Unsupported exported SDG provenance schema")
    actual = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlink in exported SDG bundle: {relative}")
        if path.is_file() and relative.as_posix() != "provenance.json":
            actual[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["files"]:
        raise RuntimeError("Exported SDG profile, source or dependency bytes changed")
    if actual["configs.json"] != manifest["profiles_sha256"]:
        raise RuntimeError("Exported SDG profile hash does not match provenance")


def load_bundle(root: Path, package: str) -> ModuleType:
    """Verify and load Helion/CuTe from a fixed exported bundle."""
    vendor = root / "_vendor"
    if "SDG_ATTENTION_PROFILES" in os.environ:
        raise RuntimeError(
            "Exported SDG bundles use their fixed configs.json. Unset "
            "SDG_ATTENTION_PROFILES; use the source entry for an external dataset."
        )
    _verify_files(root)
    _verify_loaded_origins(vendor, allow_namespace=True)
    # Process the actual trusted dependency .pth files, including CuTe's package
    # path setup. The bundle directory precedes all ambient site-package roots.
    sys.path.insert(0, str(vendor))
    _namespace_spec = importlib.machinery.PathFinder.find_spec(
        "nvidia_cutlass_dsl", [str(vendor)]
    )
    if (
        _namespace_spec is None
        or _namespace_spec.origin is not None
        or _namespace_spec.submodule_search_locations is None
    ):
        raise RuntimeError("The SDG bundle requires the CuTe namespace dependency")
    _namespace = importlib.util.module_from_spec(_namespace_spec)
    # Freeze this top-level namespace to the vendor. A dynamic NamespacePath would
    # otherwise merge ambient installations again when the .pth changes sys.path.
    _namespace.__path__ = list(_namespace_spec.submodule_search_locations)
    _namespace_spec.submodule_search_locations = _namespace.__path__
    sys.modules["nvidia_cutlass_dsl"] = _namespace
    _before_pth = tuple(sys.path)
    site.addsitedir(str(vendor))
    _added_paths = [path for path in sys.path if path not in _before_pth]
    sys.path[:] = [*_added_paths, str(vendor)] + [
        path for path in sys.path if path not in _added_paths and path != str(vendor)
    ]
    _verify_loaded_origins(vendor)

    # This compiler is identified by committed content, not an unrelated wheel's
    # version. The exporter deliberately omits the source dependency's Helion metadata.
    try:
        _helion_distribution = importlib.metadata.distribution("helion")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "An ambient Helion distribution is visible to the exported SDG bundle "
            f"at {_helion_distribution.locate_file('')}. Use an isolated interpreter; "
            "the bundled compiler is identified by provenance.json's Git tree."
        )
    _dsl_distribution = importlib.metadata.distribution("nvidia-cutlass-dsl")
    if (
        not Path(str(_dsl_distribution.locate_file("")))
        .resolve()
        .is_relative_to(vendor)
    ):
        raise RuntimeError(
            "CuTe distribution metadata does not belong to the SDG bundle"
        )
    for _package in ("cutlass", "cutlass.cutlass_dsl", "helion"):
        importlib.import_module(_package)
    _verify_loaded_origins(vendor)

    return importlib.import_module(".implementation", package)
