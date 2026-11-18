"""The on-disk compile-cache key names the device kernel."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
import helion.language as hl
from helion.runtime.cute import launcher
from helion.runtime.cute.compile_cache import CompiledCuteLauncher

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bump(x: torch.Tensor) -> torch.Tensor:
    """Ones added through atomics into a zero-filled accumulator. With one
    block of 256 threads the host's zero fill and the kernel proper launch
    with one argument of one schema and one block shape each."""
    (n,) = x.size()
    out = torch.zeros([n], dtype=torch.float32, device=x.device)
    for tile in hl.tile(n):
        hl.atomic_add(out, [tile], torch.ones_like(out[tile]))
    return out


ARGS = (torch.empty(4096, dtype=torch.float32),)
CONFIG = {"block_sizes": [256], "num_threads": [256], "cute_pdl": True}
MAIN = "_helion_bump"
FILL = "_helion_zero_fill_0_helion_bump"
SMS = 148


def _launchers(fn: Callable[..., object]) -> dict[str, tuple[object, ...]]:
    """What the launcher derives for every kernel ``fn`` launches, by the
    kernel's name: its on-disk key, its schema and block, and the compiled
    launcher it makes for the launch, which holds that key (not compiled).

    The key is the launcher's own: ``_get_compiled_cute_launcher`` computes
    it for the launcher it returns, from whatever inputs it has, on a device
    of ``SMS`` SMs."""
    calls = []

    def record(entry: object, grid: tuple[int, ...], *args: object, **options: object):
        calls.append((entry, grid, args, options))

    with cpu_render_target(fake_device_sm_count=SMS, forbid_native=False):
        fn(*ARGS, _launcher=record)
    found: dict[str, tuple[object, ...]] = {}
    for entry, grid, args, options in calls:
        block = tuple(int(extent) for extent in (*options["block"], 1, 1)[:3])
        grid3 = tuple(int(extent) for extent in (*grid, 1, 1)[:3])
        with patch.object(launcher, "_validate_cute_launcher_tensor", lambda t: None):
            built = launcher._build_cute_schema_and_args(entry, args, grid3)
        with patch.object(launcher, "_cute_num_sm_from_arch_args", lambda args: SMS):
            compiled = launcher._get_compiled_cute_launcher(
                entry,
                built.schema,
                block,
                compile_options=options.get("cute_compile_options"),
                arch_args=args,
            )
        assert isinstance(compiled, CompiledCuteLauncher)
        found[entry.__name__] = (compiled._cache_key, built.schema, block, compiled)
    return found


def test_the_kernels_of_one_module_have_keys_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's zero-fill kernel carries its module's source hash, and
    here it also agrees with the main kernel in schema, block, plans and
    options. The key names the kernel, so the two differ, in the process
    that generated the module and in a copy a worker job loads, and the
    main kernel finds no entry where the fill kernel stored its code."""
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(bump, ARGS)
        source = bound.to_code(
            bound.config_spec.normalized_config(helion.Config.from_dict(CONFIG))
        )
    module = PyCodeCache.load(source)
    bound.env.backend.annotate_compiled_module(module, source, bound.kernel.name)

    parent = _launchers(getattr(module, bound.kernel.name))
    assert list(parent) == [FILL, MAIN]
    main_key, main_schema, main_block, main = parent[MAIN]
    fill_key, fill_schema, fill_block, fill = parent[FILL]
    # Everything the key held besides the kernel agrees.
    assert (main_schema, main_block) == (fill_schema, fill_block)
    assert getattr(module, MAIN)._helion_cute_source_hash == (
        getattr(module, FILL)._helion_cute_source_hash
    )
    assert main_key is not None and fill_key is not None
    assert main_key != fill_key

    # A copy as a worker job loads it.
    copy = _load_compiled_fn(_serialize_compiled_fn(getattr(module, bound.kernel.name)))
    try:
        loaded = _launchers(copy)
    finally:
        _unload_compiled_fn(copy)
    assert loaded[MAIN][0] == main_key
    assert loaded[FILL][0] == fill_key

    # The fill kernel's entry on disk, and what each launcher finds there.
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path))
    _cache_dir, mlir, meta = fill._cache_file_paths()
    Path(mlir).write_bytes(b"the fill kernel's module")
    Path(meta).write_text(json.dumps({"function_name": "the fill kernel's entry"}))
    with (
        patch("cutlass.base_dsl.cache_helpers.load_ir", lambda *a, **k: (None, mlir)),
        patch.object(
            CompiledCuteLauncher,
            "_jit_cuda_dialect_function",
            lambda self, module, function_name, **options: function_name,
        ),
    ):
        assert fill._reload_from_disk() == "the fill kernel's entry"
        assert main._reload_from_disk() is None


def test_the_key_reads_the_kernels_name_without_a_default() -> None:
    """A kernel that carries a source hash and no name has no key: one made
    up for it would be the key of every such kernel of its module, the
    collision the name is in the key against. Reading it is an error."""

    def key(**attributes: object) -> str | None:
        kernel = SimpleNamespace(_helion_cute_source_hash="one module", **attributes)
        return launcher._cute_disk_cache_key(kernel, (), (256, 1, 1), (), None, None)

    with pytest.raises(AttributeError, match="__name__"):
        key()
    keys = [key(__name__=MAIN), key(__name__=FILL)]
    assert None not in keys
    assert keys[0] != keys[1]
