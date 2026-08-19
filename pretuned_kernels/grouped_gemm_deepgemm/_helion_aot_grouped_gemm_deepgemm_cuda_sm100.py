"""Reviewed B200 dispatch for ``grouped_gemm_deepgemm``.

Ordinary JIT evaluation selects one of the reviewed exact logical profiles,
while the user key separately specializes the physical B major, packed-A
source tile, and total packed-M work. An internal expected-M value of zero
represents a legacy call that omitted that metadata.

The kernel wrapper validates the actual argument device as NVIDIA B200 before
reviewed evaluation reaches this artifact.
"""

import hashlib as _hashlib
import importlib as _importlib
import importlib.util as _importlib_util
from pathlib import Path as _Path
import sys as _sys
import threading as _threading
from _thread import LockType as _LockType
from types import ModuleType as _ModuleType


_PROFILE_PATH = _Path(__file__).with_name("reviewed_profiles.py")
if not _PROFILE_PATH.is_file():
    raise ImportError(
        "the grouped-GEMM AOT artifact requires sibling reviewed_profiles.py"
    )
_PROFILE_PATH = _PROFILE_PATH.resolve()
_PROFILE_MODULE_NAME = (
    "_helion_grouped_gemm_reviewed_profiles_"
    + _hashlib.sha256(str(_PROFILE_PATH).encode()).hexdigest()[:16]
)
_PROFILE_STATE_MODULE_NAME = f"{_PROFILE_MODULE_NAME}_state"
_CANONICAL_PROFILE_MODULE_NAME = (
    "pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles"
)


def _profile_state_lock() -> _LockType:
    """Return one path-scoped lock shared by every artifact module alias."""

    state = _sys.modules.get(_PROFILE_STATE_MODULE_NAME)
    if state is None:
        state = _sys.modules.setdefault(
            _PROFILE_STATE_MODULE_NAME,
            _ModuleType(_PROFILE_STATE_MODULE_NAME),
        )
    lock = vars(state).get("profile_lock")
    if lock is None:
        lock = vars(state).setdefault("profile_lock", _threading.Lock())
    if not isinstance(lock, _LockType):
        raise ImportError("grouped-GEMM profile loader state is corrupted")
    return lock


def _load_reviewed_profiles() -> _ModuleType:
    canonical = _sys.modules.get(_CANONICAL_PROFILE_MODULE_NAME)
    if canonical is not None:
        canonical_path = getattr(canonical, "__file__", None)
        if canonical_path is not None and _Path(canonical_path).resolve() == _PROFILE_PATH:
            # import_module participates in Python's per-module lock, so a
            # concurrent canonical import cannot expose a partial module.
            return _importlib.import_module(_CANONICAL_PROFILE_MODULE_NAME)
    with _profile_state_lock():
        existing = _sys.modules.get(_PROFILE_MODULE_NAME)
        if existing is not None:
            return existing
        profile_spec = _importlib_util.spec_from_file_location(
            _PROFILE_MODULE_NAME,
            _PROFILE_PATH,
        )
        if profile_spec is None or profile_spec.loader is None:
            raise ImportError(
                f"unable to load grouped-GEMM profiles from {_PROFILE_PATH}"
            )
        reviewed = _importlib_util.module_from_spec(profile_spec)
        _sys.modules[_PROFILE_MODULE_NAME] = reviewed
        try:
            profile_spec.loader.exec_module(reviewed)
        except Exception:
            _sys.modules.pop(_PROFILE_MODULE_NAME, None)
            raise
        return reviewed


_REVIEWED = _load_reviewed_profiles()


def autotune_grouped_gemm_deepgemm(
    groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    b_major: str = "k",
    source_m_tile: int | None = None,
    packed_m: int | None = None,
) -> dict[str, object]:
    """Return the reviewed config for one logical grouped shape."""
    if any(type(value) is not int or value <= 0 for value in (groups, n, k)):
        raise ValueError("grouped-GEMM dispatch dimensions must be positive integers")
    if type(expected_m_per_group) is not int or expected_m_per_group < 0:
        raise ValueError("expected M/group must be a nonnegative integer key")
    # b_major and packed_m are validated AOT key features; reviewed profile
    # selection intentionally keys only on the logical shape and source tile.
    if b_major not in ("k", "n"):
        raise ValueError("grouped B major must be 'k' or 'n'")
    if packed_m is not None and (type(packed_m) is not int or packed_m <= 0):
        raise ValueError("packed M must be a positive integer key")
    config_name = _REVIEWED.reviewed_config_name(
        groups,
        expected_m_per_group or None,
        n,
        k,
        source_m_tile,
    )
    return _REVIEWED.reviewed_config_values(config_name)
