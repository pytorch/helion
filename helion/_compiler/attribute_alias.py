"""``AttributeAlias``: an attribute name kept alive after its field moved.

Role: core utility.  A class declares ``old_name = AttributeAlias[T]("path.to.field")``
and reads and writes of ``instance.old_name`` go to ``instance.path.to.field``.
The backend-state regrouping uses it for the ``cute_*`` names core objects
(``CompileEnvironment``, ``GenerateAST``, ``DeviceFunction``) carried before
their state moved into the opaque ``backend_state`` slot.  They are kept for
one release; delete the alias line when the last reader of the old name is gone.
"""

from __future__ import annotations

import operator
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import overload

T = TypeVar("T")


class AttributeAlias(Generic[T]):
    """Data descriptor forwarding an attribute to ``path`` on the instance.

    ``path`` is a dotted attribute path relative to the instance; the last
    component is the field, the prefix names the object that holds it.
    """

    def __init__(self, path: str) -> None:
        prefix, _, self.name = path.rpartition(".")
        assert prefix, f"AttributeAlias needs a holder prefix: {path!r}"
        self.path = path
        self._holder = operator.attrgetter(prefix)

    @overload
    def __get__(self, obj: None, owner: type | None = None) -> AttributeAlias[T]: ...
    @overload
    def __get__(self, obj: object, owner: type | None = None) -> T: ...
    def __get__(self, obj: object | None, owner: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(self._holder(obj), self.name)

    def __set__(self, obj: object, value: T) -> None:
        setattr(self._holder(obj), self.name, value)

    def __repr__(self) -> str:
        return f"AttributeAlias({self.path!r})"
