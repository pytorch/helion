from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.fake_tensor import _SymIntOutputStub

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.fake_tensor import _CacheKeyState
    from torch._subclasses.fake_tensor import _DispatchCacheEntryOutputInfo
    from torch._subclasses.fake_tensor import _DispatchCacheKey
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


def create_fake_tensor_mode(shape_env: ShapeEnv) -> FakeTensorMode:
    """Keep tile provenance when rebuilding cached tensor metadata."""
    mode = FakeTensorMode(shape_env=shape_env)
    # PyTorch's higher-order operators dispatch on the exact mode type, so
    # customize this instance instead of subclassing FakeTensorMode.
    # PyTorch 2.13 also passes the operator before the input arguments.
    rebuild = cast(
        "Callable[..., FakeTensor | None]", mode._get_output_tensor_from_cache_entry
    )
    # ``dataclasses.replace`` is type-checked against the installed torch's
    # ``_DispatchCacheEntryOutputInfo``; builds before pytorch/pytorch#192047
    # (PyTorch 2.13, the 2026-10-14 nightly) have no ``is_canonical_contiguous``
    # field, although the runtime path below is guarded by the ``getattr``.
    # Loosen the checker's view of the call, as for ``rebuild`` above.
    replace_entry = cast(
        "Callable[..., _DispatchCacheEntryOutputInfo]", dataclasses.replace
    )

    def get_output_tensor_from_cache_entry(
        state: _CacheKeyState,
        entry: _DispatchCacheEntryOutputInfo,
        key: _DispatchCacheKey,
        *args: object,
        **kwargs: object,
    ) -> FakeTensor | None:
        # pytorch/pytorch#192047 rebuilds contiguous outputs with torch.empty.
        # Unlike empty_strided, it unwraps constant-valued SymInts to integers,
        # losing the original expression that identifies a Helion tile axis.
        # Older torch versions have no contiguous-output cache fast path.
        if getattr(entry, "is_canonical_contiguous", False):
            assert entry.metadata is not None
            if any(isinstance(dim, _SymIntOutputStub) for dim in entry.metadata.shape):
                entry = replace_entry(entry, is_canonical_contiguous=False)
        return rebuild(state, entry, key, *args, **kwargs)

    mode._get_output_tensor_from_cache_entry = get_output_tensor_from_cache_entry
    return mode
