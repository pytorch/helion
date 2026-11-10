"""Record and replay external values read by the compiler.

The caller owns admission of operations and fingerprinting of supported values.
Replay only follows globals, builtins and ordinary module dictionaries.
"""

from __future__ import annotations

from collections.abc import Callable
import types
from typing import TypeAlias

Fingerprint: TypeAlias = Callable[[object], str | None]
_MISSING = object()
_VERSION = 1


def _value_fingerprint(value: object, fingerprint: Fingerprint) -> str | None:
    if type(value) is types.ModuleType:
        name = vars(value).get("__name__")
        return f"module:{name}" if type(name) is str else None
    key = fingerprint(value)
    return f"value:{key}" if type(key) is str else None


def _resolve(fn: types.FunctionType, path: list[str]) -> object:
    if type(fn.__globals__) is not dict:
        return _MISSING
    if path[0] == "builtin" and path[1] in fn.__globals__:
        return _MISSING
    namespace = fn.__globals__ if path[0] == "global" else fn.__builtins__
    if type(namespace) is not dict:
        return _MISSING
    value = namespace.get(path[1], _MISSING)
    for attribute in path[2:]:
        if type(value) is not types.ModuleType:
            return _MISSING
        value = vars(value).get(attribute, _MISSING)
    return value


def validate_dependencies(
    snapshot: object, fn: types.FunctionType, fingerprint: Fingerprint
) -> bool:
    """Check a saved certificate without executing attribute descriptors."""
    if type(snapshot) is not dict or type(fn) is not types.FunctionType:
        return False
    if type(snapshot.get("version")) is not int or snapshot["version"] != _VERSION:
        return False
    guards = snapshot.get("guards")
    if type(guards) is not list:
        return False
    for guard in guards:
        if type(guard) is not dict:
            return False
        path, expected = guard.get("path"), guard.get("fingerprint")
        if (
            type(path) is not list
            or len(path) < 2
            or any(type(part) is not str for part in path)
            or path[0] not in ("global", "builtin")
            or type(expected) is not str
        ):
            return False
        value = _resolve(fn, path)
        if value is _MISSING or _value_fingerprint(value, fingerprint) != expected:
            return False
    return True


class DependencyCapture:
    """A certificate for compiler reads whose operations the caller admits.

    A module may have several aliases. Attribute reads are recorded against
    every known path so replacing one alias cannot evade validation.
    """

    def __init__(self, fn: types.FunctionType, fingerprint: Fingerprint) -> None:
        self.fn = fn
        self.fingerprint = fingerprint
        self.reason: str | None = None
        self._guards: dict[tuple[str, ...], str] = {}
        self._modules: dict[int, tuple[types.ModuleType, list[tuple[str, ...]]]] = {}

    def reject(self, reason: str) -> None:
        """Keep the first reason this compilation cannot publish a certificate."""
        if self.reason is None:
            self.reason = reason

    def _record(self, path: tuple[str, ...], value: object) -> None:
        if self.reason is not None:
            return
        key = _value_fingerprint(value, self.fingerprint)
        if key is None:
            self.reject("an external value cannot be fingerprinted")
            return
        previous = self._guards.get(path)
        if previous is not None and previous != key:
            self.reject("an external value changed during compilation")
            return
        self._guards[path] = key
        if type(value) is types.ModuleType:
            _, paths = self._modules.setdefault(id(value), (value, []))
            if path not in paths:
                paths.append(path)

    def record_global(self, name: str, value: object) -> None:
        """Record the original value before the compiler converts it to a fake."""
        if (
            type(self.fn.__globals__) is not dict
            or type(self.fn.__builtins__) is not dict
        ):
            self.reject("global and builtin namespaces must be ordinary dictionaries")
            return
        if name in self.fn.__globals__:
            root, actual = "global", self.fn.__globals__[name]
        else:
            root, actual = "builtin", self.fn.__builtins__.get(name, _MISSING)
        if actual is not value:
            self.reject("a global read has no matching root in the kernel function")
            return
        self._record((root, name), value)

    def record_attribute(self, base: object, name: str, value: object) -> None:
        """Record a module attribute read, including reads through local aliases."""
        if type(base) is not types.ModuleType:
            self.reject("an attribute read is not from an ordinary module")
            return
        entry = self._modules.get(id(base))
        if entry is None or entry[0] is not base:
            self.reject("a module attribute has no recorded source")
            return
        if vars(base).get(name, _MISSING) is not value:
            self.reject("a module attribute requires dynamic resolution")
            return
        for path in tuple(entry[1]):
            self._record((*path, name), value)

    def snapshot(self) -> dict[str, object] | None:
        """Publish only if every captured value still matches after compilation."""
        if self.reason is not None:
            return None
        result: dict[str, object] = {
            "version": _VERSION,
            "guards": [
                {"path": list(path), "fingerprint": value}
                for path, value in sorted(self._guards.items())
            ],
        }
        if not validate_dependencies(result, self.fn, self.fingerprint):
            self.reject("an external value changed before certificate publication")
            return None
        return result
