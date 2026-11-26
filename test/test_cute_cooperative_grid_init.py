"""Compile the private argument and its publishing rendezvous together."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import os
import sys
from threading import Event
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import cooperative_grid_init
from helion._compiler.cute import cooperative_grid_ptx
from helion._compiler.cute.autotune.config_keys import GRID_INIT_KEY
from helion._compiler.cute.cooperative_grid_ptx import _leaf_body
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_dependent_full_votes import dependent_full_votes
from helion._testing.cute_ordered_mma import duplicate_mma_loads
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _phases(x: torch.Tensor) -> torch.Tensor:
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    c = torch.empty_like(x)
    for t in hl.tile(x.numel()):
        a[t] = x[t] + 1
    hl.barrier()
    for t in hl.tile(x.numel()):
        b[t] = a[t] * 3
    hl.barrier()
    for t in hl.tile(x.numel()):
        c[t] = b[t] - 2
    return c


@helion.kernel(backend="cute", static_shapes=True)
def _one_join(x: torch.Tensor) -> torch.Tensor:
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for t in hl.tile(x.numel()):
        a[t] = x[t] + 1
    hl.barrier()
    for t in hl.tile(x.numel()):
        b[t] = a[t] * 3
    return b


def test_grid_assembly_has_no_external_operands() -> None:
    signature = ".visible .func helion_grid_sync()\n{"
    lines = ".reg .pred %p<2>;\n$L0:\n@%p1 bra $L0;\nbarrier.sync 0;\nret;\n}"
    asm = _leaf_body(signature + "\n" + lines)
    assert asm and "$L0" not in asm and asm.count("${:uid}") == 2
    for invalid in (
        "call callee;",
        "ld.global.u32 %r1, [some_global];",
        "ld.global.u32 %p1, [%word];",
        "@%external bra $L0;",
        "mov.u32 %p3, %tid.x;",
        "mov.u32 %p01, %tid.x;",
        "mov.u32 %p1, %envreg00;",
        "ret;",
        "bra missing;",
    ):
        assert _leaf_body(signature + "\n" + invalid + "\n" + lines) is None


def test_grid_seed_uses_implicit_barrier_pid_without_compiling() -> None:
    # Use the actual multi-phase seed pipeline, independently of the host
    # toolkit installation. Neither building the search nor rendering a
    # software counter may compile the CUDA cooperative-groups header.
    with (
        patch.object(
            cooperative_grid_init, "grid_toolkit_installed", return_value=True
        ),
        patch.object(cooperative_grid_init, "grid_sync_rejected", return_value=False),
        patch.object(
            cooperative_grid_ptx,
            "grid_sync_asm",
            side_effect=AssertionError("unexpected NVRTC compile"),
        ),
        cpu_render_target(fake_device_sm_count=148),
    ):
        bound = cpu_bind(_phases, (torch.empty(513),))
        gen = bound.config_spec.create_config_generation()
        pairs = [c for _, c in gen.seed_flat_config_pairs()]
        (selected,) = [c for c in pairs if c.get(GRID_INIT_KEY)]
        (control,) = [
            c
            for c in pairs
            if not c.get(GRID_INIT_KEY)
            and {k: v for k, v in c.config.items() if k != GRID_INIT_KEY}
            == {k: v for k, v in selected.config.items() if k != GRID_INIT_KEY}
        ]
        assert cooperative_grid_init.grid_init_schedule(
            bound.config_spec,
            {k: v for k, v in control.config.items() if k != "pid_type"},
        )
        assert not cooperative_grid_init.grid_init_schedule(
            bound.config_spec, dict(control.config, pid_type="flat")
        )
        assert "_cute_toolkit_grid_sync" not in bound.to_code(control)


def test_two_concurrent_first_requests_share_one_compilation() -> None:
    toolkit = cooperative_grid_ptx._Toolkit(cooperative_grid_ptx, (), frozenset({100}))
    started = Event()
    second_lookup = Event()
    release = Event()

    def compiler(owner, arch):
        if started.is_set():
            return "duplicate"
        started.set()
        assert release.wait(10)
        return "first"

    def lookup():
        if started.is_set():
            second_lookup.set()
        return toolkit

    with (
        patch.object(cooperative_grid_ptx, "_toolkit", side_effect=lookup),
        patch.object(
            cooperative_grid_ptx, "_compile_grid_sync", side_effect=compiler
        ) as compile_leaf,
        ThreadPoolExecutor(max_workers=2) as threads,
    ):
        try:
            first = threads.submit(cooperative_grid_ptx.grid_sync_asm, (10, 0))
            assert started.wait(10)
            other = threads.submit(cooperative_grid_ptx.grid_sync_asm, (10, 0))
            assert second_lookup.wait(10)
            assert not wait([other], timeout=0.1).done
        finally:
            release.set()
        assert first.result(timeout=10) == other.result(timeout=10) == "first"
        compile_leaf.assert_called_once_with(toolkit, 100)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="render-pool fork is POSIX-only")
def test_forked_renderer_owns_grid_compiler_locks() -> None:
    # Nonblocking acquisition exposes an inherited locked mutex without ever
    # hanging the subprocess or confusing a pytest timeout with a refusal.
    with cooperative_grid_ptx._discovery_lock, cooperative_grid_ptx._leaf_lock:
        pid = os.fork()
        if pid == 0:
            usable = cooperative_grid_ptx._discovery_lock.acquire(
                blocking=False
            ) and cooperative_grid_ptx._leaf_lock.acquire(blocking=False)
            os._exit(0 if usable else 1)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_standalone_checks_normalized_cooperative_geometry() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            duplicate_mma_loads,
            (
                torch.empty(128, 64, dtype=torch.bfloat16),
                torch.empty(64, 64, dtype=torch.bfloat16),
            ),
        )
        spec = bound.config_spec
        cfg = helion.Config(
            block_sizes=[128, 64, 64, 64],
            cute_mma_dag="warp",
            pid_type="persistent_blocked",
        )
        assert spec.normalized_config(cfg).pid_type == "flat"
        requested = helion.Config.from_dict(cfg.config | {GRID_INIT_KEY: True})
        assert spec.backend.standalone_config(
            spec, requested, stored=True
        ) == spec.normalized_config(cfg)
        with pytest.raises(
            exc.InvalidConfig, match="normalized cooperative barrier schedule"
        ):
            spec.backend.standalone_config(spec, requested)
        with pytest.raises(
            exc.InvalidConfig, match="normalized cooperative barrier schedule"
        ):
            bound.to_code(
                requested, options=helion.OutputCodeOptions(allow_helion_deps=False)
            )


def test_rejected_leaf_retires_only_its_architecture_without_recompiling() -> None:
    toolkit = cooperative_grid_ptx._Toolkit(
        cooperative_grid_ptx, (), frozenset({90, 100})
    )
    with (
        patch.object(cooperative_grid_ptx, "_toolkit", return_value=toolkit),
        patch.object(
            cooperative_grid_ptx, "_compile_grid_sync", return_value=None
        ) as compile_leaf,
        cpu_render_target(fake_device_sm_count=148),
    ):
        bound = cpu_bind(_phases, (torch.empty(513),))
        spec = bound.config_spec

        def enabled() -> list[helion.Config]:
            return [
                c
                for _, c in spec.create_config_generation().seed_flat_config_pairs()
                if c.get(GRID_INIT_KEY)
            ]

        (selected,) = enabled()
        hash_before = spec.cache_fingerprint_hash()
        raw_before = [c.config for c in spec.compiler_seed_configs]
        before = spec.create_config_generation()
        original = before.strict_config_pair(selected)[0]
        portable = spec.backend.standalone_config(spec, selected)
        assert not portable.get(GRID_INIT_KEY)
        assert portable.block_sizes == selected.block_sizes
        options = helion.OutputCodeOptions(allow_helion_deps=False)
        # This CPU check certifies the operand entering the public standalone
        # exporter. Its actual launch-schema capture needs a CUDA device.
        with patch.object(
            spec.backend,
            "build_standalone_code",
            side_effect=lambda bound, imports, body: body,
        ):
            assert bound.to_code(selected, options=options) == bound.to_code(
                portable, options=options
            )
        compile_leaf.assert_not_called()
        with pytest.raises(
            exc.InvalidConfig, match="in-kernel grid initialization needs"
        ):
            bound.to_code(selected)
        assert cooperative_grid_ptx.grid_sync_rejected((10, 0))
        assert not cooperative_grid_ptx.grid_sync_rejected((9, 0))
        # Presence in the persistent schema does not change with process
        # history. The actual search drops the known rejected treatment.
        assert enabled() == [selected]
        assert not spec.backend.autotune_config_is_viable(spec, selected)
        fresh = helion.kernel(backend="cute", static_shapes=True)(_phases.fn)
        fresh_spec = cpu_bind(fresh, (torch.empty(513),)).config_spec
        assert [c.config for c in fresh_spec.compiler_seed_configs] == raw_before
        assert fresh_spec.cache_fingerprint_hash() == hash_before
        repaired = spec.backend.aot_fallback_config(spec, selected)
        assert repaired == portable == spec.backend.standalone_config(spec, selected)
        expected = dict(selected.config)
        expected.pop(GRID_INIT_KEY)
        spec.normalize(expected)
        assert repaired.config == expected
        assert not spec.backend.can_reuse_autotune_result(spec, selected)
        assert spec.backend.can_reuse_autotune_result(spec, helion.Config())
        # The old immutable generation still round-trips after rejection.
        assert before.unflatten(original) is not None
        assert before.unflatten(before.random_flat()) is not None
        with pytest.raises(
            exc.InvalidConfig, match="in-kernel grid initialization needs"
        ):
            bound.to_code(selected)
        assert cooperative_grid_ptx.grid_sync_asm((10, 0)) is None
        compile_leaf.assert_called_once_with(toolkit, 100)


def test_mismatched_headers_do_not_hide_an_installed_compatible_toolkit(
    tmp_path: Path,
) -> None:
    roots = []
    for index, version in enumerate((12080, 13000, 13000)):
        root = tmp_path / str(index)
        inc = root / "include"
        inc.mkdir(parents=True)
        (inc / "cooperative_groups.h").touch()
        (inc / "cuda_runtime_api.h").write_text(f"#define CUDART_VERSION   {version}\n")
        if index != 1:
            for rel in (
                "cooperative_groups/details/info.h",
                "cooperative_groups/details/sync.h",
                "nv/target",
            ):
                dep = inc / rel
                dep.parent.mkdir(parents=True, exist_ok=True)
                dep.touch()
        roots.append(root)
    with patch.object(
        cooperative_grid_ptx, "_toolkit_roots", return_value=tuple(roots)
    ):
        assert cooperative_grid_ptx._include_options((13, 0)) == (
            (roots[2] / "include",),
        )
        assert cooperative_grid_ptx._include_options((12, 5)) == ()
        original = type(tmp_path).is_file

        def permission_error(path):
            if roots[0] in path.parents:
                raise PermissionError("unreadable toolkit")
            return original(path)

        with patch.object(type(tmp_path), "is_file", permission_error):
            assert cooperative_grid_ptx._include_options((13, 0)) == (
                (roots[2] / "include",),
            )


def test_unavailable_toolkit_preserves_default_and_rejects_explicit_grid() -> None:
    with (
        patch.object(
            cooperative_grid_init, "grid_toolkit_installed", return_value=False
        ),
        patch.object(
            cooperative_grid_ptx,
            "grid_sync_asm",
            side_effect=AssertionError("unexpected NVRTC compile"),
        ),
        cpu_render_target(fake_device_sm_count=148),
    ):
        bound = cpu_bind(_phases, (torch.empty(513),))
        spec = bound.config_spec
        assert not any(
            c.get(GRID_INIT_KEY)
            for _, c in spec.create_config_generation().seed_flat_config_pairs()
        )
        with pytest.raises(exc.InvalidConfig, match="CUDA grid-group support"):
            bound.to_code(
                helion.Config.from_dict(
                    dict(spec.default_config()) | {GRID_INIT_KEY: True}
                )
            )


def test_private_allocation_and_first_grid_join_are_paired() -> None:
    # This test inspects source only; compiling the public toolkit operation
    # is deliberately independent of whether this CPU CI worker has CUDA.
    with (
        patch.object(
            cooperative_grid_init, "grid_toolkit_installed", return_value=True
        ),
        patch.object(cooperative_grid_init, "grid_sync_rejected", return_value=False),
        patch.object(
            cooperative_grid_ptx, "grid_sync_asm", return_value="{ barrier.sync 0; }"
        ),
        cpu_render_target(fake_device_sm_count=148),
    ):
        # Explicit configs must still load when automatic seeds are disabled.
        op = helion.kernel(
            backend="cute", static_shapes=True, disable_autotuner_heuristics=True
        )(_phases.fn)
        bound = cpu_bind(op, (torch.empty(513),))
        spec = bound.config_spec
        raw = dict(spec.default_config()) | {"pid_type": "persistent_blocked"}
        if spec.cross_loop_pipeline is not None:
            raw["cross_loop_pipeline"] = "barrier"
        for use in (False, True):
            cfg = helion.Config.from_dict(raw | {GRID_INIT_KEY: use})
            source = bound.to_code(cfg)
            tree = ast.parse(source)
            dev, host = [fn for fn in tree.body if isinstance(fn, ast.FunctionDef)]
            syncs = [
                n.func.id
                for n in ast.walk(dev)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in {"_cute_toolkit_grid_sync", "_cute_grid_barrier"}
            ]
            assert syncs == (
                ["_cute_toolkit_grid_sync", "_cute_grid_barrier"]
                if use
                else ["_cute_grid_barrier"] * 2
            )
            assert "_helion_cute_cooperative = True" in source
            private = [
                n
                for n in ast.walk(host)
                if isinstance(n, ast.Call)
                and any(
                    k.arg == "dtype" and ast.unparse(k.value) == "torch.uint32"
                    for k in n.keywords
                )
            ]
            assert len(private) == 1 and ast.unparse(private[0].func) == (
                "torch.empty" if use else "torch.zeros"
            )
            assert ("x_grid_sem[0] = cutlass.Uint32(0)" in source) == use


def test_one_local_join_still_marks_cooperative_and_split_phases_decline() -> None:
    with (
        patch.object(
            cooperative_grid_init, "grid_toolkit_installed", return_value=True
        ),
        patch.object(cooperative_grid_init, "grid_sync_rejected", return_value=False),
        patch.object(
            cooperative_grid_ptx, "grid_sync_asm", return_value="{ barrier.sync 0; }"
        ),
        cpu_render_target(fake_device_sm_count=148),
    ):
        bound = cpu_bind(_one_join, (torch.empty(513),))
        cfg = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"pid_type": "persistent_blocked", GRID_INIT_KEY: True}
        )
        incompatible = helion.Config.from_dict(
            dict(cfg) | {"block_sizes": [*cfg.block_sizes, 16]}
        )
        with patch.object(
            cooperative_grid_ptx, "grid_sync_asm", return_value="available"
        ):
            assert bound.config_spec.backend.can_reuse_autotune_result(
                bound.config_spec, cfg
            )
            assert not bound.config_spec.backend.can_reuse_autotune_result(
                bound.config_spec, incompatible
            )
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.backend.aot_fallback_config(
                    bound.config_spec, incompatible
                )
        code = bound.to_code(cfg)
        dev = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
        names = [ast.unparse(n.func) for n in ast.walk(dev) if isinstance(n, ast.Call)]
        assert names.count("_cute_toolkit_grid_sync") == 1
        assert "_cute_grid_barrier" not in names
        assert "_helion_cute_cooperative = True" in code
        options = helion.OutputCodeOptions(allow_helion_deps=False)
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(incompatible, options=options)
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(
                helion.Config.from_dict(dict(cfg) | {"unrecognized_control": 1}),
                options=options,
            )

        # Unversioned bundles cannot embed a portable child program.
        with pytest.raises(exc.InvalidConfig, match="static_shapes=True"):
            bound.config_spec.backend.validate_aot_bundle_config(cfg)
        bound.config_spec.backend.validate_aot_bundle_config(helion.Config())

        # The hardware surface must permit a resident launch, not merely
        # recognize an explicit persistent value from a different kernel.
        original_pids = bound.config_spec.allowed_pid_types
        with patch.object(bound.config_spec, "allowed_pid_types", ("flat",)):
            assert not cooperative_grid_init.grid_init_surface(bound.config_spec)
            assert not bound.config_spec.backend.can_reuse_autotune_result(
                bound.config_spec, cfg
            )
        assert bound.config_spec.allowed_pid_types == original_pids
        split = cpu_bind(dependent_full_votes, (torch.empty(3, 32, 32),))
        stale = helion.Config(
            pid_type="persistent_blocked", cute_in_kernel_grid_init=True
        )
        actual = split.config_spec
        with pytest.raises(exc.InvalidConfig, match="cooperative barrier schedule"):
            actual.backend.standalone_config(actual, stale)
        # Model CUDA's register cap as well as its real cooperative PID
        # policy when reading a slot trained on a resident binding.
        resident = {
            "cross_loop_pipeline": "barrier",
            "cute_pdl": True,
            "maxnreg": 64,
            "num_sm_multiplier": 1,
        }
        with (
            patch("helion.autotuner.config_spec.supports_maxnreg", return_value=True),
            patch("helion.autotuner.config_spec._regs_per_block", return_value=65536),
        ):
            unchanged = stale.config | resident
            for config in (stale.config, unchanged):
                saved = helion.Config.from_dict(config)
                recovered = actual.backend.aot_fallback_config(actual, saved)
                assert recovered.pid_type == "flat" and not recovered.get("cute_pdl")
                assert (
                    "num_sm_multiplier" not in recovered and "maxnreg" not in recovered
                )
                assert len(stage_sources(split.to_code(recovered))) == 2
            for invalid in (
                {"cross_loop_pipeline": "static"},
                {"cross_loop_schedule": "static_pipeline"},
                {"maxnreg": -1},
                {"num_sm_multiplier": -1},
            ):
                with pytest.raises(exc.InvalidConfig):
                    actual.backend.aot_fallback_config(
                        actual, helion.Config.from_dict(unchanged | invalid)
                    )

        spec = split.config_spec
        assert spec.cute_dependent_phase_launches
        assert not cooperative_grid_init.grid_init_surface(spec)
        assert not spec.backend.can_reuse_autotune_result(spec, cfg)
        with pytest.raises(exc.InvalidConfig):
            split.to_code(
                helion.Config.from_dict(
                    dict(spec.default_config()) | {GRID_INIT_KEY: True}
                )
            )


@pytest.mark.cuda
@skipIfNotCUDA()
def test_separate_grid_counters_reset_after_poisoned_replay(tmp_path: Path) -> None:
    capability = torch.cuda.get_device_capability()
    if not cooperative_grid_ptx.grid_toolkit_installed(capability):
        pytest.skip(
            "installed CUDA toolkit does not support the cooperative grid operation"
        )
    if cooperative_grid_ptx.grid_sync_asm(capability) is None:
        # Future toolkit implementations may not provide the supported leaf.
        # On the actual SM100/toolkit pair we exercise, rejection is a bug.
        toolkit = cooperative_grid_ptx._toolkit()
        assert toolkit is not None
        if capability == (10, 0) and toolkit.nvrtc.nvrtcVersion()[1:] in {
            (13, 0),
            (13, 2),
        }:
            pytest.fail(cooperative_grid_ptx.grid_sync_requirement(capability))
        pytest.skip(cooperative_grid_ptx.grid_sync_requirement(capability))
    values = [torch.arange(513, device=DEVICE, dtype=torch.float32) for _ in range(2)]
    bound = _phases.bind((values[0],))
    spec = bound.config_spec
    raw = dict(spec.default_config()) | {
        "pid_type": "persistent_blocked",
        GRID_INIT_KEY: True,
    }
    if spec.cross_loop_pipeline is not None:
        raw["cross_loop_pipeline"] = "barrier"
    portable = bound.to_code(
        helion.Config.from_dict(raw),
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    assert "_cute_toolkit_grid_sync" not in portable
    assert "torch.zeros" in portable
    module_path = tmp_path / "portable_grid_program.py"
    module_path.write_text(portable)
    software = load_generated_module(module_path)._phases
    torch.testing.assert_close(
        software(values[0]), (values[0] + 1) * 3 - 2, rtol=0, atol=0
    )
    call = bound.compile_config(helion.Config.from_dict(raw))
    call(values[0])
    states = []
    allocate = torch.empty

    def remember(*args, **kwargs):
        value = allocate(*args, **kwargs)
        if value.dtype is torch.uint32 and value.shape == (1,):
            states.append(value)
        return value

    stream = torch.cuda.Stream()
    graphs = []
    for value in values:
        graph = torch.cuda.CUDAGraph()
        with (
            patch.object(torch, "empty", remember),
            torch.cuda.graph(graph, stream=stream),
        ):
            output = call(value)
        graphs.append((graph, torch.cuda.Stream(), output))
    assert len(states) == 2 and states[0].data_ptr() != states[1].data_ptr()
    for repeat in (1, 3):
        for i, (state, value) in enumerate(zip(states, values, strict=True)):
            state.fill_(0xDEAD0000 + repeat + i)
            value.add_(i + repeat)
        for gr, replay_stream, _ in graphs:
            replay_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(replay_stream):
                gr.replay()
        torch.cuda.synchronize()
        for (_, _, output), value, state in zip(graphs, values, states, strict=True):
            torch.testing.assert_close(output, (value + 1) * 3 - 2, rtol=0, atol=0)
            assert state.item() == 1 << 31
