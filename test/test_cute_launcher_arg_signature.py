"""The launcher's per-argument facts are an on-disk cache key: pin them.

The compiled-wrapper schema tuples (``("tensor", str(dtype), ndim[, sizes,
strides][, alignment])``, ``("scalar", kind)``, ``("scalar_constexpr", kind,
cache_value, value)``), the launch-argument cache key and the last-launch
guard are derived from the same argument facts by three launch tiers.  Their
layouts are the key of the persisted CuTe compile cache
(``_cute_disk_cache_key`` hashes ``repr(schema_key)``) and are decoded
positionally by the generated wrapper, the standalone exporter and the
standalone launcher, so a refactor of the tiers must reproduce them byte for
byte.  ``test/data/cute_launcher_arg_signature.json`` holds the values the
launcher produced before the tiers shared one ``ArgSignature``; regenerate it
only for a deliberate key change:

    python test/test_cute_launcher_arg_signature.py --write
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest
import torch

from helion._testing.cute_launcher import mock_cute_launcher_runtime
from helion.runtime.cute import launcher

pytestmark = pytest.mark.cpu_only

GOLDEN = Path(__file__).parent / "data" / "cute_launcher_arg_signature.json"

_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint32,
    torch.uint64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
    torch.float4_e2m1fn_x2,
)


class _Arena:
    """One 64-byte-aligned CPU buffer; every fixture tensor is a view of it so
    pointers can be recorded relative to its base."""

    def __init__(self) -> None:
        raw = torch.empty(1 << 18, dtype=torch.uint8)
        skew = (-raw.data_ptr()) % 64
        self.base = raw[skew : skew + (1 << 17)]
        assert self.base.data_ptr() % 64 == 0

    def view(
        self,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        *,
        byte_offset: int,
        strides: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        itemsize = dtype.itemsize
        numel = 1
        for extent in shape:
            numel *= extent
        if strides is None:
            contiguous: list[int] = []
            stride = 1
            for extent in reversed(shape):
                contiguous.insert(0, stride)
                stride *= extent
            strides = tuple(contiguous)
        span = sum(
            (extent - 1) * stride for extent, stride in zip(shape, strides, strict=True)
        )
        length = (span + 1) * itemsize if numel else 0
        storage = self.base[byte_offset : byte_offset + max(length, itemsize)]
        if numel == 0:
            return storage.view(dtype)[:0].reshape(shape)
        # ``_cases`` filters ``offset % itemsize``: every view is element aligned.
        assert byte_offset % itemsize == 0, "fixture offsets are element aligned"
        return torch.as_strided(storage.view(dtype), shape, strides)

    def relative(self, pointer: int) -> int:
        # A zero-sized tensor's data pointer is 0 (recorded as is).
        return pointer - self.base.data_ptr() if pointer else 0


def _plain(value: object, arena: _Arena) -> object:
    """A JSON-stable form of a schema / key / guard component."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__class__": type(value).__name__,
            **{
                field.name: _plain(getattr(value, field.name), arena)
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return ["size", *value]
    if isinstance(value, tuple):
        return ["tuple", *(_plain(item, arena) for item in value)]
    if isinstance(value, list):
        return [_plain(item, arena) for item in value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    raise TypeError(f"unexpected fixture component {value!r}")


@dataclasses.dataclass(frozen=True)
class _Case:
    name: str
    kernel_attributes: dict[str, object]
    args: tuple[object, ...]
    grid: tuple[int, int, int]
    constexpr: tuple[bool, ...] = ()
    bake: bool | None = None


def _cases(arena: _Arena) -> list[_Case]:
    cases: list[_Case] = []
    offsets = (0, 1, 2, 4, 8, 16)
    for dtype in _DTYPES:
        for offset in offsets:
            if offset % dtype.itemsize:
                continue
            tensor = arena.view(dtype, (16,), byte_offset=offset)
            for bake in (False, True):
                cases.append(
                    _Case(
                        f"{dtype}-off{offset}-bake{bake}",
                        {},
                        (tensor,),
                        (1, 1, 1),
                        bake=bake,
                    )
                )
    # Ranks, strides, zero-sized and 0-d tensors; scalars by kind, value and
    # constexpr flag; -0.0 / NaN / large ints.
    strided = arena.view(
        torch.float16, (4, 8), byte_offset=128, strides=(16, 1)
    )  # a column slice of a (4, 16) row-major buffer
    cube = arena.view(torch.bfloat16, (2, 3, 4), byte_offset=512)
    scalar0d = arena.view(torch.float32, (1,), byte_offset=1024)[0]
    empty = arena.view(torch.float16, (0,), byte_offset=2048)
    empty2d = arena.view(torch.float32, (0, 4), byte_offset=2048)
    for bake in (False, True):
        cases.append(
            _Case(
                f"ranks-bake{bake}",
                {},
                (strided, cube, scalar0d, empty, empty2d),
                (3, 2, 1),
                bake=bake,
            )
        )
    scalar_args = (strided, 7, True, 1.5, -0.0, float("nan"), 2**40, False, 0.0)
    for flags in (
        (),
        (False, True, False, True, True, True, False, False, False),
        (False,) * 3 + (True,) * 6,
    ):
        cases.append(
            _Case(
                f"scalars-constexpr{sum(flags)}",
                {},
                scalar_args,
                (5, 1, 1),
                constexpr=flags,
            )
        )
    # Plans that bake / do not bake tensor shapes and the kernel-level opt-out.
    sized = arena.view(torch.float16, (256, 128), byte_offset=4096)
    cases.extend(
        [
            _Case(
                "kernel-disables-bake",
                {"_helion_cute_disable_bake_tensor_shapes": True},
                (sized, 3),
                (2, 1, 1),
            ),
            _Case(
                "tcgen05-full-tiles-bake",
                {
                    "_helion_cute_wrapper_plans": [
                        {
                            "kind": "tcgen05_ab_tma",
                            "m_size": 256,
                            "n_size": 128,
                            "k_total_size": 128,
                            "bm": 128,
                            "bn": 128,
                            "bk": 64,
                        }
                    ]
                },
                (sized, sized, 1.0),
                (2, 1, 1),
            ),
            _Case(
                "tcgen05-partial-tiles-no-bake",
                {
                    "_helion_cute_wrapper_plans": [
                        {
                            "kind": "tcgen05_ab_tma",
                            "m_size": 250,
                            "n_size": 128,
                            "k_total_size": 128,
                            "bm": 128,
                            "bn": 128,
                            "bk": 64,
                        }
                    ]
                },
                (sized, sized, 1.0),
                (2, 1, 1),
            ),
            _Case(
                "row-mma-bakes",
                {"_helion_cute_wrapper_plans": [{"kind": "helion_flash_row_mma"}]},
                (sized, cube),
                (4, 1, 1),
            ),
        ]
    )
    return cases


def _record(case: _Case, arena: _Arena) -> dict[str, object]:
    kernel = SimpleNamespace(**case.kernel_attributes)

    def make_ptr(
        dtype: object, address: int, space: object, *, assumed_align: int
    ) -> tuple[str, int, int]:
        return ("ptr", arena.relative(address), assumed_align)

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_cute_launcher_runtime(monkeypatch, make_ptr, constexpr=case.constexpr)
        built = (
            launcher._build_cute_schema_and_args(kernel, case.args, case.grid)
            if case.bake is None
            else launcher._build_cute_schema_and_args(
                kernel, case.args, case.grid, bake_tensor_shapes=case.bake
            )
        )
        key = launcher._cute_launch_arg_cache_key(kernel, case.args, case.grid)
        guard = launcher._cute_last_launch_arg_guard(kernel, case.args, case.grid)
        assert guard.matches(kernel, case.args, case.grid)
    key_plain = []
    for entry in key:
        if isinstance(entry, tuple) and entry and entry[0] == "tensor":
            entry = (*entry[:5], arena.relative(cast("int", entry[5])), *entry[6:])
        key_plain.append(_plain(entry, arena))
    guard_plain = _plain(guard, arena)
    for arg_guard in cast("dict[str, Any]", guard_plain)["arg_guards"][1:]:
        if arg_guard["__class__"] == "_CuteLastTensorArgGuard":
            arg_guard["data_ptr"] = arena.relative(arg_guard["data_ptr"])
    return {
        "schema": _plain(built.schema, arena),
        "launch_args": _plain(built.launch_args, arena),
        "owned_tensors": len(built.owned_tensors),
        "arg_cache_key": key_plain,
        "last_launch_guard": guard_plain,
    }


def snapshot() -> dict[str, object]:
    arena = _Arena()
    return {case.name: _record(case, arena) for case in _cases(arena)}


@pytest.mark.parametrize("name", [case.name for case in _cases(_Arena())])
def test_arg_signature_matches_golden(name: str) -> None:
    golden = json.loads(GOLDEN.read_text())
    assert name in golden, "regenerate the golden with --write"
    arena = _Arena()
    case = next(case for case in _cases(arena) if case.name == name)
    assert _record(case, arena) == golden[name]


def test_golden_covers_every_case() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == {case.name for case in _cases(_Arena())}


def main() -> None:
    if "--write" in sys.argv:
        out = (
            Path(sys.argv[sys.argv.index("--write") + 1])
            if len(sys.argv) > sys.argv.index("--write") + 1
            else GOLDEN
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {out}")
        return
    current = snapshot()
    golden = json.loads(GOLDEN.read_text())
    mismatched = [name for name in golden if golden[name] != current.get(name)]
    print(f"{len(golden)} cases; mismatched: {mismatched}")
    sys.exit(1 if mismatched or set(golden) != set(current) else 0)


if __name__ == "__main__":
    main()
