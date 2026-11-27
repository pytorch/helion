"""A stand-in for a backend that holds the configs a kernel selects without
a search to some of its ``autotune_config_overrides``."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterator


@contextlib.contextmanager
def hold_block_sizes(surface: type, ineligible: Collection[int] = ()) -> Iterator[None]:
    """Stub the two hooks of ``surface``, a ``BackendSearchSurface`` class,
    as a backend that holds unsearched configs to the ``block_sizes``
    override (``overrides_held_without_search``) and compiles them with it
    (``overrides_without_search``), except on the bindings whose config spec
    ``ineligible`` names by ``id``: the hook decides by the binding. A
    config that has the override's ``block_sizes`` is left as it is, the
    hook's own result among them."""

    def held(self: object, overrides: dict[str, object]) -> dict[str, object]:
        return {k: v for k, v in overrides.items() if k == "block_sizes"}

    def applied(
        self: object,
        spec: object,
        config: dict[str, object],
        overrides: dict[str, object],
    ) -> dict[str, object] | None:
        if "block_sizes" not in overrides or id(spec) in ineligible:
            return None
        if config.get("block_sizes") == overrides["block_sizes"]:
            # Nothing to change: the hook's contract for its own result.
            return None
        return {**config, "block_sizes": overrides["block_sizes"]}

    with (
        patch.object(surface, "overrides_held_without_search", held),
        patch.object(surface, "overrides_without_search", applied),
    ):
        yield
