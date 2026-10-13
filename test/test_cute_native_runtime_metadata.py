from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from helion.runtime.config import Config
from helion.runtime.cute import launcher
from helion.runtime.cute import native_pool as pool
from helion.runtime.cute.native_pool_protocol import freeze
from helion.runtime.cute.native_pool_protocol import sha

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel


def cached_host(
    value: int, *, _launcher: Callable[..., object] | None = None
) -> object:
    assert _launcher is not None
    return _launcher(cached_host, (1,), value, block=(128, 1, 1))


@pytest.fixture
def populated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    kernel = cached_host
    monkeypatch.setattr(
        kernel, "_helion_cute_source_hash", "source-hash", raising=False
    )
    monkeypatch.setattr(kernel, "_helion_cute_param_constexpr", (False,), raising=False)
    # Restore attributes installed by the actual helpers at fixture teardown.
    for name in (
        "_helion_cute_compiled_launchers",
        "_helion_cute_launch_arg_cache",
        "_helion_cute_last_launch_cache",
        "_helion_cute_fastpath",
    ):
        monkeypatch.setattr(kernel, name, None, raising=False)
        monkeypatch.delattr(kernel, name)
    args = (7,)
    grid = (1, 1, 1)
    block = (128, 1, 1)
    launch = launcher._CuteLaunchArgCacheEntry(
        schema=(("scalar", "int"),),
        launch_args=(7, *grid),
        grouped_static_metadata=(),
        owned_tensors=(),
    )
    original = pool.kernel_attributes(kernel)
    monkeypatch.setattr(launcher, "_build_cute_schema_and_args", lambda *args: launch)
    monkeypatch.setattr(
        launcher, "_create_cute_wrapper", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(launcher, "_cute_disk_cache_key", lambda *args: "cache-key")
    monkeypatch.setattr(pool, "wrapper_ast", lambda *args: "wrapper-ast")
    assert launcher._build_cached_cute_schema_and_args(kernel, args, grid) is launch
    compiled = cast(
        "Any", launcher._get_compiled_cute_launcher(kernel, launch.schema, block)
    )
    # Mark the cached launcher ready; its invocation stays blocked in CPU tests.
    compiled._compiled = lambda *args: "ran"
    launcher._set_cute_last_launch_cache_entry(
        kernel, args, grid, block, None, launch, compiled
    )
    fast = launcher._CuteFastRelaunch(
        executor=object(),
        exe_args=[],
        tensor_guards=(),
        scalar_guards=(),
        constexpr_flags=(False,),
        tensor_slots=(),
        by_ref_writers=[],
        by_val_slots=[],
        arg_count=1,
        grid=grid,
        block=block,
        compile_options=None,
        device_index=0,
        last_raw=0,
        keepalive=(),
    )
    monkeypatch.setattr(launcher, "_cute_build_fast_relaunch", lambda *args: fast)
    launcher._cute_maybe_build_fastpath(
        kernel, args, grid, block, None, launch, compiled
    )
    attributes = vars(kernel)
    assert attributes["_helion_cute_launch_arg_cache"]
    assert attributes["_helion_cute_compiled_launchers"]
    assert isinstance(
        attributes["_helion_cute_last_launch_cache"],
        launcher._CuteLastLaunchCacheEntry,
    )
    assert attributes["_helion_cute_fastpath"] is fast
    # This is the old production boundary; it reproduces the post-use TypeError.
    with pytest.raises(TypeError, match="Unsupported compile metadata"):
        freeze(
            {
                key: value
                for key, value in attributes.items()
                if key.startswith("_helion_cute_")
                and key != "_helion_cute_compiled_launchers"
            }
        )
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    owner = torch.ones(2, 4)
    _, options, num_sm = launcher._cute_compiled_launcher_discriminator(
        launch.schema, block, None, args
    )
    request = pool.NativeRequest(
        {
            "source": str(source),
            "source_sha256": sha(source),
            "environment": freeze(launcher._cute_cache_relevant_env()),
            "cache_key": "cache-key",
            "options": options,
            "attributes": original,
            "schema": freeze(launch.schema),
            "wrapper_ast": "wrapper-ast",
            "block": block,
            "num_sm": num_sm,
        },
        compiled,
        (owner,),
        (owner.untyped_storage(),),
        (pool.owner_facts(owner),),
        (kernel, launch, args, cached_host, args, None),
    )
    return SimpleNamespace(
        kernel=kernel,
        original=original,
        request=request,
        source=source,
        owner=owner,
        args=args,
    )


def test_actual_launcher_caches_preserve_original_request(
    populated: SimpleNamespace,
) -> None:
    assert pool.kernel_attributes(populated.kernel) == populated.original
    populated.request.check()


def test_failed_fastpath_probe_marker_preserves_request(
    populated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(populated.kernel, "_helion_cute_fastpath")
    monkeypatch.setattr(launcher, "_cute_build_fast_relaunch", lambda *args: None)
    launcher._cute_maybe_build_fastpath(
        populated.kernel,
        populated.args,
        (1, 1, 1),
        (128, 1, 1),
        None,
        populated.request.retained[1],
        populated.request.compiled,
    )
    assert vars(populated.kernel)["_helion_cute_fastpath"] is False
    populated.request.check()


def test_required_recapture_passes_populated_metadata_boundary(
    populated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    bound = cast(
        "BoundKernel",
        SimpleNamespace(
            _normalized_config_copy=lambda config: config,
            _compile_cache={config: cached_host},
            get_cached_path=lambda config: str(Path(__file__).resolve()),
        ),
    )

    class CapturedMetadata(Exception):
        pass

    def boundary(*args: object) -> str:
        # The CPU test stops after capture's real attrs/schema/key path, before
        # its CUDA-owner admission. No owner or device predicate is replaced.
        raise CapturedMetadata

    monkeypatch.setattr(pool, "wrapper_ast", boundary)
    assert pool.host_supported(cached_host)
    with pytest.raises(CapturedMetadata):
        pool.capture(bound, config, cached_host, populated.args, include_cached=True)


@pytest.mark.parametrize(
    "name,value",
    [
        ("_helion_cute_source_hash", "different"),
        ("_helion_cute_param_constexpr", (True,)),
        ("_helion_cute_pointer_alignment", 4),
        ("_helion_cute_native_matmul_metadata", True),
        ("_helion_cute_fastpath_compile_policy", "different"),
    ],
)
def test_compile_metadata_still_invalidates_request(
    populated: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    monkeypatch.setattr(populated.kernel, name, value, raising=False)
    with pytest.raises(AssertionError):
        populated.request.check()


def test_unknown_runtime_like_attribute_still_rejects(
    populated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        populated.kernel, "_helion_cute_fastpath_extra", object(), raising=False
    )
    with pytest.raises(TypeError, match="Unsupported compile metadata"):
        populated.request.check()


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "owner",
        "key",
        "options",
        "schema",
        "wrapper",
        "environment",
        "disk_key",
    ],
)
def test_original_authority_checks_survive_runtime_cache_population(
    populated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    request = populated.request
    request.check()
    if field == "source":
        populated.source.write_text("changed\n")
    elif field == "owner":
        populated.owner.add_(1)
    elif field == "key":
        request.compiled._cache_key = "changed"
    elif field == "options":
        request.compiled._compile_options = "changed"
    elif field == "schema":
        old = request.retained
        request.retained = (
            old[0],
            replace(old[1], schema=(("scalar", "float"),)),
            *old[2:],
        )
    elif field == "wrapper":
        monkeypatch.setattr(pool, "wrapper_ast", lambda *args: "changed")
    elif field == "environment":
        monkeypatch.setenv("CUTE_DSL_TEST_RECAPTURE_FLAG", "changed")
    else:
        monkeypatch.setattr(launcher, "_cute_disk_cache_key", lambda *args: "changed")
    with pytest.raises(AssertionError):
        request.check()
