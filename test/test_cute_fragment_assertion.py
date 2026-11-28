from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._compiler import per_thread_planning
from helion._compiler.cute.pdl_ops import config_conflicts_with_required_owner
from helion._compiler.cute.split_k_cluster import ClusterKFacts
from helion._testing import DEVICE
from helion._testing import import_example
from helion._testing import skipIfNotCUDA
from helion._testing.cute_aux import as_sm100
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import kernel_body
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
from helion._testing.cute_output_addresses import generated_output_address_coverage
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl
from helion.runtime.ref_mode import RefMode

pytestmark = pytest.mark.cpu_only

_MESSAGE = "bad group total: {limit} at 100% ±1"
_NEGATIVE_TOTAL = "negative group total"


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _checked_prefix(
    counts: torch.Tensor,
    check: hl.constexpr,
    message: hl.constexpr = _MESSAGE,
    check_nonnegative: hl.constexpr = False,
):
    edges = torch.empty(5, dtype=torch.int64, device=counts.device)
    ok = torch.empty(1, dtype=torch.bool, device=counts.device)
    for group in hl.grid(2):
        cols = hl.arange(4)
        values = counts[:, group * 4 + cols].sum(dim=0)
        if group == 1:
            edges[0] = 0
            edges[cols + 1] = torch.cumsum(values, dim=0)
            total = values.sum(dim=0)
            valid = total <= 128
            ok[0] = valid
            if check:
                torch._assert_async(valid, message)
                if check_nonnegative:
                    torch._assert_async(total >= 0, _NEGATIVE_TOTAL)
    return edges, ok


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _check_before_reuse(flags: torch.Tensor, later: torch.Tensor):
    out = torch.empty(2, dtype=torch.int64, device=flags.device)
    for group in hl.grid(2):
        torch._assert_async(torch.select(flags[group, :], 0, 63))
        values = later[group, :]
        out[group] = torch.select(values, 0, 12) + values.sum(dim=0)
    return out


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _check_host_scalar(flag: torch.Tensor, rows: torch.Tensor):
    out = torch.empty_like(rows)
    for row in hl.grid(2):
        torch._assert_async(flag)
        out[row, :] = torch.cumsum(rows[row, :], dim=0)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _check_captured_scalar(
    rows: torch.Tensor, flag: torch.Tensor, out: torch.Tensor, computed: hl.constexpr
):
    for row in hl.grid(3):
        if computed:
            value = rows[row, :].sum() > 0
        else:
            value = flag
        if row == 1:
            torch._assert_async(value)
        out[row] = rows[row, 0] > 0
    return out


def test_assertions_in_binary_cross_entropy_decomposition():
    @helion.kernel(backend="cute", static_shapes=True)
    def loss(x: torch.Tensor, target: torch.Tensor):
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = torch.nn.functional.binary_cross_entropy(
                x[tile], target[tile], reduction="none"
            )
        return out

    bound = cpu_bind(loss, (torch.ones(16), torch.zeros(16)))
    code = bound.to_code(helion.Config(block_sizes=[16]), output_origin_lines=False)
    assert code.count("call __assertfail") == 2


def _source(kernel, *args):
    with mock_cuda_unavailable():
        return kernel_body(cpu_bind(kernel, args).to_code(kernel.configs[0]))


def _guarded_assertion(source):
    # kernel_body strips source comments. Identify the thread coordinate
    # rather than relying on a compiler-generated temporary name.
    (lane,) = re.findall(
        r"^ *(\w+) = .*cute\.arch\.thread_idx\(\)\[0\]", source, re.MULTILINE
    )
    assert source.count("call __assertfail,") == 1
    guard = re.search(
        rf"^( *)if {lane} == 0:\n\1    [^\n]*call __assertfail,[^\n]*",
        source,
        re.MULTILINE,
    )
    assert guard is not None
    # A CTA-wide barrier must be outside the lane-zero body.
    sync = f"\n{guard[1]}cute.arch.sync_threads()\n"
    return source[: guard.start()], source[guard.end() :], sync


def test_computed_assertion_observes_prior_stores_and_precedes_scratch_reuse():
    counts = torch.ones((32, 8), dtype=torch.int32)
    assert "inline_asm_elementwise(" not in _source(_checked_prefix, counts, False)
    before, _, sync = _guarded_assertion(_source(_checked_prefix, counts, True))
    assert sync in before[before.index(".store(", before.index("ok.iterator")) :]

    flags = torch.ones((2, 128), dtype=torch.bool)
    _, after, sync = _guarded_assertion(
        _source(_check_before_reuse, flags, flags.clone())
    )
    assert sync in after[: after.index("later.iterator")]


def test_assertion_rejects_bare_host_scalar():
    with pytest.raises(
        exc.BackendUnsupported, match="requires a computed scalar Boolean"
    ):
        _source(_check_host_scalar, torch.tensor(True), torch.ones(2, 4))


def test_captured_assertion_checks_origin_before_selecting_a_schedule():
    args = torch.ones(3, 32), torch.tensor(True), torch.empty(3, dtype=torch.bool)
    with mock_cuda_unavailable():
        cpu_bind(_check_captured_scalar, (*args, True))
    with (
        mock_cuda_unavailable(),
        pytest.raises(
            exc.BackendUnsupported, match="requires a computed scalar Boolean"
        ),
    ):
        cpu_bind(_check_captured_scalar, (*args, False))


def test_assertion_rejects_independent_roots_before_tuning():
    @helion.kernel(backend="cute", static_shapes=True)
    def two_roots(x: torch.Tensor):
        out = torch.empty_like(x)
        for i in hl.grid(x.size(0)):
            value = x[i]
            torch._assert_async(value > 0)
            out[i] = value
        for j in hl.grid(x.size(0)):
            out[j] = out[j] + 1
        return out

    with (
        mock_cuda_unavailable(),
        pytest.raises(
            exc.BackendUnsupported, match="single complete CTA fragment root"
        ),
    ):
        cpu_bind(two_roots, (torch.ones(16),))


def test_assertion_message_limit_counts_utf8_bytes():
    counts = torch.ones((32, 8), dtype=torch.int32)
    message = "±" * 127  # 254 bytes is the largest permitted CUDA message.
    assert "call __assertfail," in _source(_checked_prefix, counts, True, message)
    with pytest.raises(exc.InvalidAPIUsage, match="must be shorter than 255 bytes"):
        _source(_checked_prefix, counts, True, message + "!")


def test_non_boolean_fragment_never_drops_the_assertion():
    flags = torch.ones((2, 128), dtype=torch.int32)
    with (
        mock_cuda_unavailable(),
        pytest.raises(
            exc.BackendUnsupported, match="requires a computed scalar Boolean"
        ),
    ):
        cpu_bind(_check_before_reuse, (flags, flags.clone())).to_code(
            _check_before_reuse.configs[0]
        )


def _run_valid_prefix() -> torch.Tensor:
    counts = torch.ones((32, 8), dtype=torch.int32, device=DEVICE)
    edges, ok = _checked_prefix(counts, True, check_nonnegative=True)
    assert edges.tolist() == [0, 32, 64, 96, 128] and ok.tolist() == [True]
    return counts


@pytest.mark.cuda
@skipIfNotCUDA()
def test_device_assertions_accept_valid_totals() -> None:
    _run_valid_prefix()


@pytest.mark.cuda
@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("site", "message"), [("capacity", _MESSAGE), ("negative", _NEGATIVE_TOTAL)]
)
def test_group_total_succeeds_then_reports_cuda_assertion_in_isolation(site, message):
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check-assertion", site],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "observed CUDA assertion 710" in child.stdout
    assert any(
        line.endswith(f"Assertion `{message}` failed.")
        for line in child.stderr.splitlines()
    ), child.stderr


@helion.kernel(backend="cute", static_shapes=True)
def _checked_sum(x: torch.Tensor, limit: int):
    out = torch.empty(x.size(0), dtype=x.dtype, device=x.device)
    for group in hl.grid(x.size(0)):
        counts = x[group, :]
        total = counts.sum()
        torch._assert_async(total <= limit, "group capacity")
        out[group] = total
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _normalized_rows(x: torch.Tensor, check: hl.constexpr):
    out = torch.empty_like(x)
    rows, columns = x.shape
    block = hl.register_block_size(columns)
    for row in hl.tile(rows):
        total = hl.zeros([row], dtype=torch.float32)
        for col in hl.tile(columns, block_size=block):
            total += x[row, col].sum(dim=1)
        if check:
            torch._assert_async(total.sum() > 0)
        for col in hl.tile(columns, block_size=block):
            out[row, col] = x[row, col] - total[:, None]
    return out


def test_scalar_assertion_owns_simple_reduction() -> None:
    with as_sm100():
        bound = cpu_bind(_checked_sum, (torch.ones(2, 32, dtype=torch.int64), 64))
        code = bound.to_code(helion.Config(block_sizes=[]), output_origin_lines=False)
        assert code.count("call __assertfail") == 1


def test_assertion_with_insufficient_shared_storage_rejects_the_config() -> None:
    with as_sm100():
        bound = cpu_bind(_checked_sum, (torch.ones(2, 32, dtype=torch.int64), 64))
        with (
            patch(
                "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
                return_value=1,
            ),
            pytest.raises(
                exc.InvalidConfig, match="computed fragments need .*shared bytes"
            ),
        ):
            bound.to_code(helion.Config(block_sizes=[]))


@pytest.mark.parametrize("checked", [False, True])
def test_assertions_screen_candidates_and_promotion(checked: bool) -> None:
    with as_sm100():
        bound = cpu_bind(_normalized_rows, (torch.ones(4, 128), checked))
        assert bound.host_function is not None
        spec, backend = bound.config_spec, bound.env.backend
        with bound.env:
            choice = spec._flat_fields()["cute_reduction_sequence"]
            assert isinstance(choice, EnumFragment)
            assert set(choice._active_choices()) == (
                {"scalar", "reload", "resident"}
                | (set() if checked else {"bounded", "bounded_layout"})
            )
            normal, native = (
                bound._normalized_config_copy(
                    helion.Config(block_sizes=[128, 1], cute_reduction_sequence=m)
                )
                for m in ("scalar", "bounded")
            )
            assert native.get("cute_reduction_sequence") == "bounded"
            assert backend.autotune_config_is_viable(spec, normal)
            assert backend.autotune_config_is_viable(spec, native) == (not checked)
            for default in (normal, native):
                spec.compiler_default_config = default
                seeds = backend.postprocess_seed_configs(
                    bound.env, bound.host_function.device_ir, [normal, native]
                )
                assert normal in seeds and (native in seeds) == (not checked)
                assert spec.compiler_default_config is (
                    None if checked and default is native else default
                )
        if checked:
            assert (
                bound.to_code(normal, output_origin_lines=False).count(
                    "call __assertfail"
                )
                == 1
            )
            with pytest.raises(
                exc.InvalidConfig, match="whole-kernel generator bounded_cache"
            ):
                bound.to_code(native, output_origin_lines=False)


def test_checked_affine_scan_search_keeps_only_the_ordinary_owner() -> None:
    @helion.kernel(backend="cute", static_shapes=True, fast_math=True)
    def affine(x: torch.Tensor, y: torch.Tensor, checked: hl.constexpr):
        for row in hl.tile(x.size(0)):
            values = x[row, :]
            if checked:
                torch._assert_async(x[row.begin, 0] > 0)
            total = values.sum(1)[:, None]
            # Two stores to each BF16 plane expose the direct scan search.
            x[row, :] = values + total
            y[row, :] = values
            x[row, :] = values - total
            y[row, :] = values + 1
        return y

    key = "cute_affine_scan_schedule"
    choices = ("ordinary", "direct_m16n8_v1", "direct_m16n16_v1")
    with as_sm100():
        x = torch.ones((128, 128), dtype=torch.bfloat16)
        for checked in (False, True):
            bound = cpu_bind(affine, (x, torch.empty_like(x), checked))
            spec = bound.config_spec
            with bound.env:
                field = spec._flat_fields()[key]
                assert isinstance(field, EnumFragment)
                assert field._active_choices() == (
                    ("ordinary",) if checked else choices
                )
                default = spec.default_config()
                assert default[key] == "ordinary"
            if checked:
                assert "call __assertfail" in bound.to_code(default)
                for direct in choices[1:]:
                    fixed = helion.Config(**{**default, key: direct})
                    with pytest.raises(exc.InvalidConfig, match=f"{key}={direct}"):
                        bound.to_code(fixed)


@helion.kernel(backend="cute", static_shapes=True)
def _one_row_tiles(x: torch.Tensor, out: torch.Tensor, check: hl.constexpr):
    for row, col in hl.tile(x.shape, block_size=[1, None]):
        if check:
            torch._assert_async(x[row.begin, col.begin] > 0)
        out[row, col] = x[row, col] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _checked_tile_copy(
    x, out, begin: hl.constexpr, width: hl.constexpr, trim: hl.constexpr = 0
):
    for tile in hl.tile(begin, x.size(0) - trim, block_size=width):
        torch._assert_async(x[tile.begin] > 0)
        out[tile] = x[tile] + 1
    return out


@pytest.mark.parametrize(("begin", "width"), [(0, 4), (3, 4), (3, 1)])
def test_checked_fixed_tile_matches_configured_origin(begin: int, width: int) -> None:
    with as_sm100():
        x = torch.ones(16)
        args = (x, torch.empty_like(x), begin)
        fixed = cpu_bind(_checked_tile_copy, (*args, width)).to_code(
            helion.Config(block_sizes=[]), output_origin_lines=False
        )
        tuned = cpu_bind(_checked_tile_copy, (*args, None)).to_code(
            helion.Config(block_sizes=[width]), output_origin_lines=False
        )
    # Same launch, predicate load and output address for the two spellings.
    assert fixed == tuned
    assert fixed.count("call __assertfail") == 1
    launch = re.search(
        r"_launcher\(\w+, \((\d+),\), x, out, block=\((\d+), 1, 1\)\)", fixed
    )
    assert launch is not None
    blocks, threads = map(int, launch.groups())
    assert generated_output_address_coverage(
        fixed, "out", blocks, threads
    ) == dict.fromkeys(range(begin, x.numel()), 1)


def test_fixed_tile_packets_respect_both_grid_bounds() -> None:
    begin, trim = 3, 0
    with as_sm100():
        x = torch.ones(16)
        args = (x, torch.empty_like(x), begin)

        def render(width: int | None) -> str:
            return cpu_bind(_checked_tile_copy, (*args, width, trim)).to_code(
                helion.Config(
                    block_sizes=[4] if width is None else [],
                    cute_fragment_vector_io=True,
                ),
                output_origin_lines=False,
            )

        fixed = render(4)
        assert render(None) == fixed
    copies = [
        line
        for line in fixed.splitlines()
        if "cute.copy(" in line and ".align(16)" in line
    ]
    assert fixed.count("call __assertfail") == 1
    assert not copies
    # The scalar check at tile.begin and the masked input element walk
    # each load from x; the tail must also guard the output element walk.
    assert (
        sum("x.iterator" in line and ".load()" in line for line in fixed.splitlines())
        == 2
    )
    launch = re.search(
        r"_launcher\(\w+, \((\d+),\), x, out, block=\((\d+), 1, 1\)\)", fixed
    )
    assert launch is not None
    blocks, threads = map(int, launch.groups())
    assert generated_output_address_coverage(
        fixed, "out", blocks, threads
    ) == dict.fromkeys(range(begin, x.numel() - trim), 1)


def test_checked_scalar_grid_stride_is_not_a_tile_width() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def copy(x, out):
        for i in hl.grid(2, x.size(0), 4):
            torch._assert_async(x[i] > 0)
            out[i] = x[i]
        return out

    # Each CTA owns one scalar; the third operand advances its coordinate.
    with as_sm100():
        x = torch.arange(1, 17, dtype=torch.float32)
        bound = cpu_bind(copy, (x, torch.empty_like(x)))
        source = bound.to_code(helion.Config(block_sizes=[]), output_origin_lines=False)
    launch = re.search(
        r"_launcher\(\w+, \((\d+),\), x, out, block=\((\d+), 1, 1\)\)", source
    )
    assert launch is not None and source.count("call __assertfail") == 1
    blocks, threads = map(int, launch.groups())
    writes: list[tuple[int, float]] = []

    class OutputPointer:
        def __add__(self, offset: int) -> SimpleNamespace:
            return SimpleNamespace(store=lambda value: writes.append((offset, value)))

    predicates: list[bool] = []
    for lane in range(threads):
        simulate_independent_fragment(
            source,
            {"x": x},
            {"out": SimpleNamespace(iterator=OutputPointer())},
            blocks,
            thread=(lane, 0, 0),
            per_thread_stores=True,
            inline_asm=lambda args, **kwargs: predicates.append(bool(args[0])),
        )
    assert predicates == [True] * blocks
    assert sorted(writes) == [(i, x[i].item()) for i in range(2, 16, 4)]


def test_packed_grid_tiles_cannot_skip_checked_rows() -> None:
    x = torch.ones(8, 32)
    out = torch.empty_like(x)
    with as_sm100():
        plain, checked = (
            cpu_bind(_one_row_tiles, (x, out, check)) for check in (False, True)
        )
        pack = plain._normalized_config_copy(
            helion.Config(block_sizes=[16], cute_tiles_per_block=2)
        )
        single = checked._normalized_config_copy(
            helion.Config(block_sizes=[16], cute_tiles_per_block=1)
        )
        # Each CTA owns one of 8 rows × 2 column tiles. In particular, lane
        # zero must check both the even and the odd rows.
        code = checked.to_code(single, output_origin_lines=False)
        assert "num_blocks_0 = 8" in code
        assert (
            "pid_0 = cutlass.Int32(cute.arch.block_idx()[0]) % num_blocks_0\n" in code
        )
        assert "pid_1 = cutlass.Int32(cute.arch.block_idx()[0]) // num_blocks_0" in code
        assert "tile_offset_1 = pid_1 * _BLOCK_SIZE_1" in code
        assert "_launcher(_helion__one_row_tiles, (16,)," in code
        assert code.count("call __assertfail") == 1 and ".store(" in code
        # The unasserted kernel still packs two distinct logical rows per CTA.
        ordinary = plain.to_code(pack, output_origin_lines=False)
        assert (
            "% num_blocks_0 * 2 + cutlass.Int32(cute.arch.thread_idx()[0])" in ordinary
        )
        assert "block=(2, 16, 1)" in ordinary and "call __assertfail" not in ordinary

        backend, spec = checked.env.backend, checked.config_spec
        assert checked.host_function is not None
        ir = checked.host_function.device_ir
        with checked.env:
            fields = spec._flat_fields()["cute_tiles_per_block"]
            assert isinstance(fields, EnumFragment)
            assert tuple(fields._active_choices()) == (1,)
            assert spec.default_config().get("cute_tiles_per_block", 1) == 1
            assert all(
                c.get("cute_tiles_per_block", 1) == 1
                for c in spec.compiler_seed_configs
            )
            assert backend.autotune_config_is_viable(spec, single)
            assert not backend.autotune_config_is_viable(spec, pack)
            spec.compiler_default_config = pack
            seeds = backend.postprocess_seed_configs(checked.env, ir, [single, pack])
            assert single in seeds and pack not in seeds
            assert spec.compiler_default_config is None
        with pytest.raises(exc.InvalidConfig, match="cute_tiles_per_block=2"):
            spec.normalized_config(pack)
        repaired = helion.Config(block_sizes=[16], cute_tiles_per_block=2)
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired.get("cute_tiles_per_block") == 1
        with pytest.raises(exc.InvalidConfig, match="cute_tiles_per_block=2"):
            checked.to_code(pack)
        with pytest.raises(exc.InvalidConfig, match="cute_tiles_per_block=2"):
            backend.validate_codegen_config(checked.env, ir, pack)
        assert plain.env.backend.autotune_config_is_viable(plain.config_spec, pack)


@helion.kernel(backend="cute", static_shapes=True)
def _checked_matmul(lhs: torch.Tensor, rhs: torch.Tensor, check: hl.constexpr = True):
    m, k = lhs.shape
    n = rhs.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=lhs.device)
    for i, j in hl.tile([m, n]):
        if check:
            torch._assert_async(lhs[i.begin, 0] > 0)
        acc = hl.zeros((i, j), dtype=torch.float32)
        for z in hl.tile(k):
            acc = hl.dot(lhs[i, z], rhs[z, j], acc)
        out[i, j] = acc
    return out


def test_checked_search_does_not_offer_rejected_generator_families(monkeypatch):
    args = tuple(torch.empty(128, 128, dtype=torch.float32) for _ in range(2))
    keys = (
        "cute_materialized_schedule",
        "cute_materialized_operand_schedule",
        "cute_mma_f32_conversion",
        "cute_grouped_rna_warps",
        "cute_split_k_schedule",
    )
    with as_sm100():
        for checked in (False, True):
            bound = cpu_bind(_checked_matmul, (*args, checked))
            spec = bound.config_spec
            # Keep the real assertion and matmul facts. Model independently
            # discovered optional generators so every family is exercised.
            monkeypatch.setattr(spec, "cute_materialized_schedule_search_enabled", True)
            monkeypatch.setattr(
                spec, "cute_materialized_operand_schedule_search_enabled", True
            )
            monkeypatch.setattr(spec, "cute_grouped_rna_k_choices", (64,))
            monkeypatch.setattr(spec, "cute_grouped_warp_tf32_available", True)
            monkeypatch.setattr(
                spec,
                "cute_split_k_cluster_facts",
                ClusterKFacts(32, 64, 4096, (0, 1, 2), 512, 232448),
            )
            with bound.env:
                fields = spec._flat_fields()
                for key in keys:
                    field = fields[key]
                    assert isinstance(field, EnumFragment)
                    assert len(field.choices) > 1
                    allowed = tuple(
                        value
                        for value in field.choices
                        if not config_conflicts_with_required_owner(
                            spec, helion.Config(**{key: value})
                        )
                    )
                    assert field._active_choices() == allowed
                    assert len(allowed) == (1 if checked else len(field.choices))


def test_checked_matmul_skips_tcgen05_tiles_and_seeds() -> None:
    args = tuple(torch.empty(1024, 1024, dtype=torch.bfloat16) for _ in range(2))
    with as_sm100():
        plain, checked = (
            cpu_bind(_checked_matmul, (*args, flag)) for flag in (False, True)
        )
        assert plain.config_spec.tcgen05.search_enabled
        assert any(
            "tcgen05_ab_stages" in c.config
            for c in plain.config_spec.autotune_seed_configs()
        )
        spec = checked.config_spec
        assert not spec.tcgen05.search_enabled
        errors: list[str] = []
        seeds = spec.create_config_generation().seed_flat_config_pairs(errors.append)
        assert not errors
        assert not spec.autotune_seed_configs()
        assert not any("tcgen05_ab_stages" in seed.config for _, seed in seeds)
        default = spec.default_config()
        assert (
            checked.to_code(default, output_origin_lines=False).count(
                "call __assertfail"
            )
            == 1
        )


def test_asserted_matmul_preserves_scalar_owner_and_rejects_native_modes() -> None:
    with as_sm100():
        bound = cpu_bind(
            _checked_matmul,
            tuple(torch.ones(64, 64, dtype=torch.float16) for _ in range(2)),
        )
        scalar = helion.Config(block_sizes=[16, 16, 16])
        assert (
            bound.to_code(scalar, output_origin_lines=False).count("call __assertfail")
            == 1
        )
        for key in ("cute_collective_mma", "cute_register_chain"):
            native = bound._normalized_config_copy(
                helion.Config(block_sizes=[16, 16, 16], **{key: True})
            )
            assert native.get(key) is True
            assert not bound.env.backend.autotune_config_is_viable(
                bound.config_spec, native
            )
            with pytest.raises(exc.InvalidConfig, match=key):
                bound.to_code(native, output_origin_lines=False)


def _insert_assertion(fn, directory: Path, before: str, predicate: str):
    # Helion rereads the function's source while binding it.
    source = inspect.getsource(fn)
    source = source[source.index(f"def {fn.__name__}(") :]
    assert source.count(before) == 1
    source = source.replace(
        before, f"        torch._assert_async({predicate})\n{before}"
    )
    path = directory / f"checked_{fn.__name__}.py"
    path.write_text(source)
    namespace = dict(fn.__globals__)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[fn.__name__]


def test_asserted_rms_backward_filters_real_resident_seeds(tmp_path: Path) -> None:
    original = import_example("rms_norm").rms_norm_bwd.fn
    checked = _insert_assertion(
        original,
        tmp_path,
        "        grad_w_m = weight.new_zeros",
        "x[mb_cta.begin, 0] > 0",
    )
    x = torch.ones(64, 1024, dtype=torch.float16)
    args = (x.clone(), x, torch.ones(1024, dtype=x.dtype), torch.ones(64, 1))

    def bind(fn):
        return cpu_bind(
            helion.kernel(
                fn,
                backend="cute",
                static_shapes=True,
                ignore_warnings=[exc.TensorOperationInWrapper],
            ),
            args,
        )

    with as_sm100():
        baseline = bind(original)
        bound = bind(checked)

        def schedule(cfg):
            return cfg.get("cute_reduction_schedule", "scalar")

        originals = baseline.config_spec.compiler_seed_configs
        assert {schedule(seed) for seed in originals} == {
            "scalar",
            "resident",
            "pipelined",
        }
        retained = bound.config_spec.compiler_seed_configs
        assert retained and all(schedule(seed) == "scalar" for seed in retained)
        for mode in ("resident", "pipelined"):
            native = bound._normalized_config_copy(
                next(seed for seed in originals if schedule(seed) == mode)
            )
            assert not bound.env.backend.autotune_config_is_viable(
                bound.config_spec, native
            )
            with pytest.raises(exc.InvalidConfig, match=mode):
                bound.to_code(native, output_origin_lines=False)
        assert (
            bound.to_code(
                bound.config_spec.default_config(), output_origin_lines=False
            ).count("call __assertfail")
            == 1
        )


def test_checked_flash_backward_is_rejected_before_search(tmp_path: Path) -> None:
    # Add an actual scalar device check to the reference backward program.
    # Keep the same dot, reduction, and store expressions that select flash.
    original = import_example("attention").attention_backward.fn
    checked = _insert_assertion(
        original, tmp_path, "        k_j = k[tile_n, :]", "lse[tile_n.begin] > 0"
    )
    fn = helion.kernel(checked, backend="cute", static_shapes=True)
    q, k, v, out, out_grad = (
        torch.empty(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(5)
    )
    lse, delta = (torch.empty(1, 2, 256) for _ in range(2))
    args = (q, k, v, out, lse, out_grad, delta, 0.125)
    with as_sm100():
        unchecked = helion.kernel(original, backend="cute", static_shapes=True)
        spec = unchecked._bind_isolated(args).config_spec
        assert spec.default_config().get("cute_flash_bwd_family") == "fa4"
        assert any(
            seed.get("cute_flash_bwd_family") == "row_mma"
            for seed in spec.autotune_seed_configs()
        )
        with pytest.raises(exc.BackendUnsupported, match=r"_assert_async.*global dQ"):
            fn._bind_isolated(args)


@pytest.mark.parametrize("access", ["load", "store", "shifted-load", "shifted-store"])
def test_checked_scalar_indices_wrap_literals_and_reject_unproved_displacements(
    access: str,
) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def checked(x, out, access: hl.constexpr):
        for row in hl.grid(x.size(0)):
            source = (
                row - 1 if access == "shifted-load" else -1 if access == "load" else row
            )
            dest = (
                row - 1
                if access == "shifted-store"
                else -1
                if access == "store"
                else row
            )
            torch._assert_async(x[source, 0] > 0)
            out[dest, 0] = x[row, 0] + x[row, 1]

    x = torch.ones(4, 2).T
    args = (x, torch.empty_like(x))
    with mock_cuda_unavailable():
        bound = cpu_bind(checked, (*args, access))
        for staged in (False, True):
            config = helion.Config(block_sizes=[], cute_fragment_slab_loads=staged)
            if access.startswith("shifted-"):
                with pytest.raises(exc.BackendUnsupported, match="scalar memory index"):
                    bound.to_code(config)
            else:
                assert "call __assertfail" in bound.to_code(config)


def test_checked_stacktensor_memory_requires_another_owner() -> None:
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[4])
    )
    def stack_copy(x: torch.Tensor, ptr: torch.Tensor, check: hl.constexpr):
        out = torch.empty(
            (hl.specialize(ptr.size(0)), x.size(0)), dtype=x.dtype, device=x.device
        )
        for tile in hl.tile(x.size(0)):
            if check:
                torch._assert_async(x[tile.begin] > 0)
            tensors = hl.stacktensor_like(x, ptr[:])
            out[:, tile] = tensors[tile]
        return out

    x, ptr = torch.ones(4), torch.ones(2, dtype=torch.uint64)
    assert "ptr.iterator" in _source(stack_copy, x, ptr, False)
    with pytest.raises(exc.BackendUnsupported, match="StackTensor memory"):
        _source(stack_copy, x, ptr, True)


@pytest.mark.parametrize(
    "in_body", [True, False], ids=["inside-loop", "uses-loop-result"]
)
def test_checked_loop_cannot_skip_a_tensor_bounded_range(in_body):
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int64)
    def check(bounds, flags, in_body: hl.constexpr):
        for row in hl.grid(flags.size(0)):
            start = bounds[row, 0]
            stop = bounds[row, 1]
            if in_body:
                for _j in hl.grid(start, stop):
                    torch._assert_async(flags[row])
            else:
                count = flags[row].to(torch.int32)
                for _j in hl.grid(start, stop):
                    count = count + 1
                torch._assert_async(count == 2)

    bounds = torch.tensor([[2**31 - 1, 2**31 + 1]], dtype=torch.int64)
    with as_sm100(), pytest.raises(exc.BackendUnsupported, match="Int32 loop range"):
        cpu_bind(check, (bounds, torch.tensor([False]), in_body)).to_code(
            helion.Config(block_sizes=[])
        )


def test_checked_loop_proves_literals_runtime_bounds_and_final_increment():
    @helion.kernel(backend="cute", static_shapes=True, index_dtype=torch.int64)
    def check(flags, start: hl.constexpr, stop: hl.constexpr, step: hl.constexpr):
        for row in hl.grid(flags.size(0)):
            for _j in hl.grid(start, stop, step):
                torch._assert_async(flags[row])

    @helion.kernel(backend="cute", static_shapes=True)
    def from_host(flags, start: int):
        for row in hl.grid(flags.size(0)):
            for _j in hl.grid(start, 7):
                torch._assert_async(flags[row])

    flags = torch.tensor([False])
    cfg = helion.Config(block_sizes=[])
    with as_sm100():
        code = cpu_bind(check, (flags, 3, 7, 2)).to_code(cfg, output_origin_lines=False)
        loops = [
            loop
            for loop in ast.walk(ast.parse(kernel_body(code)))
            if isinstance(loop, ast.For)
            and ast.unparse(loop.iter) == "range(cutlass.Int32(3), cutlass.Int32(7), 2)"
        ]
        assert len(loops) == 1 and "call __assertfail" in ast.unparse(loops[0])
        # A signed scalar induction is valid when it is not used as a tile slice.
        assert "call __assertfail" in cpu_bind(check, (flags, -3, -1, 1)).to_code(cfg)
        for bounds in ((2**32, 2**32 + 1, 1), (2**31 - 2, 2**31 - 1, 4), (3, 7, 2**32)):
            with pytest.raises(exc.BackendUnsupported, match="Int32 loop range"):
                cpu_bind(check, (flags, *bounds)).to_code(cfg)
        with pytest.raises(exc.BackendUnsupported, match="Int32 loop range"):
            cpu_bind(from_host, (flags, 2**32)).to_code(cfg)


def test_checked_xyz_grid_enforces_physical_axis_limits():
    @helion.kernel(backend="cute", static_shapes=True)
    def check(flags, x: hl.constexpr, y: hl.constexpr, z: hl.constexpr):
        for _i, _j, _k in hl.grid([x, y, z]):
            torch._assert_async(flags[0])

    inputs = torch.tensor([True])
    xyz = helion.Config(block_sizes=[], pid_type="xyz")
    with as_sm100():

        def render(shape, config=xyz):
            return cpu_bind(check, (inputs, *shape)).to_code(config)

        # X may exceed 65535, and the upper limit is inclusive on Y and Z.
        assert "call __assertfail" in render((65536, 2, 1))
        assert "call __assertfail" in render((1, 65535, 2))
        for shape in ((2, 65536, 1), (2, 1, 65536)):
            with pytest.raises(exc.InvalidConfig, match="CUDA grid"):
                render(shape)
        assert "call __assertfail" in render(
            (2, 65536, 1), helion.Config(block_sizes=[], pid_type="flat")
        )
        reordered = helion.Config(
            block_sizes=[], pid_type="xyz", loop_orders=[[1, 0, 2]], l2_groupings=[4]
        )
        assert "call __assertfail" in render((2, 65536, 1), reordered)
        with pytest.raises(exc.InvalidConfig, match="CUDA grid"):
            render((65536, 2, 1), reordered)


def test_unchecked_xyz_grid_rejects_an_oversized_physical_dimension():
    @helion.kernel(backend="cute", static_shapes=True)
    def copy(x, y, rows: hl.constexpr, cols: hl.constexpr):
        for _i, _j in hl.grid([rows, cols]):
            y[0] = x[0]

    with as_sm100():
        args = (torch.ones(1), torch.empty(1))
        cfg = helion.Config(block_sizes=[], pid_type="xyz")
        assert cpu_bind(copy, (*args, 65536, 2)).to_code(cfg)
        with pytest.raises(exc.InvalidConfig, match="CUDA grid"):
            cpu_bind(copy, (*args, 2, 65536)).to_code(cfg)


def test_checked_grouped_grid_bounds_the_decoder_product():
    @helion.kernel(backend="cute", static_shapes=True)
    def check(flags, x: hl.constexpr, y: hl.constexpr):
        for _i, _j, _k in hl.grid([x, y, 1]):
            torch._assert_async(flags[0])

    flags = torch.tensor([True])
    with as_sm100():

        def render(x, y, pid_type):
            config = helion.Config(block_sizes=[], pid_type=pid_type, l2_groupings=[4])
            return cpu_bind(check, (flags, x, y)).to_code(config)

        # The decoder groups M rows and multiplies only by the N extent.
        assert "call __assertfail" in render(2**28, 2, "xyz")
        with pytest.raises(exc.InvalidConfig, match="grid count"):
            render(2, 2**29, "flat")


@pytest.mark.parametrize("pid_type", ["flat", "persistent_interleaved"])
def test_checked_grid_proves_product_before_pid_decoding(pid_type):
    @helion.kernel(backend="cute", static_shapes=True)
    def check(flags, n: hl.constexpr):
        for _row, _col in hl.grid([n, n]):
            torch._assert_async(flags[0])

    config = helion.Config(block_sizes=[], pid_type=pid_type)
    with as_sm100():
        with pytest.raises(exc.InvalidConfig, match="grid count"):
            cpu_bind(check, (torch.tensor([False]), 2**16)).to_code(config)
        if pid_type == "persistent_interleaved":
            bound = cpu_bind(check, (torch.tensor([True]), 2))
            with (
                patch.object(bound.env.config_spec, "num_sm", 1),
                pytest.raises(exc.InvalidConfig, match="known multi-SM target"),
            ):
                bound.to_code(config)


def test_checked_negative_tile_slice_requires_nonnegative_origin():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def check(x, out, start: hl.constexpr):
        for row in hl.grid(1):
            for tile in hl.tile(start, start + 4, block_size=4):
                total = x[tile].sum()
                torch._assert_async(total > 0)
                out[row] = total
        return out

    x = torch.ones(16)
    out = torch.zeros(1)
    eager = helion.kernel(
        check.fn,
        backend="cute",
        static_shapes=True,
        ref_mode="eager",
        config=check.configs[0],
    )
    assert eager(x, out, -8).item() == x[-8:-4].sum().item() == 4
    assert "call __assertfail" in _source(check, x, out, 4)
    with pytest.raises(
        exc.BackendUnsupported, match="tile origin not proved nonnegative"
    ):
        _source(check, x, out, -8)


def test_checked_sparse_grid_counts_its_live_coordinates() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def sparse(flags, out):
        for _, _ in hl.grid([0, 0], [50000, 50000], [50000, 50000]):
            torch._assert_async(flags[0])
            out[0] = flags[0]
        return out

    with as_sm100():
        flags = torch.ones(1, dtype=torch.bool)
        bound = cpu_bind(sparse, (flags, torch.empty_like(flags)))
        source = bound.to_code(helion.Config(block_sizes=[]), output_origin_lines=False)
    launch = re.search(
        r"_launcher\(\w+, \((\d+),\), flags, out, block=\((\d+), 1, 1\)\)", source
    )
    assert launch is not None and source.count("call __assertfail") == 1
    blocks, threads = map(int, launch.groups())
    assert blocks == 1
    assert generated_output_address_coverage(source, "out", blocks, threads) == {0: 1}


def test_checked_scalar_root_stride_is_not_a_thread_planning_width(monkeypatch):
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def check(flags, step: hl.constexpr):
        for i, j in hl.grid([0, 0], [2, 2], [step, 1]):
            torch._assert_async(flags[i, j])

    flags = torch.ones((2, 2), dtype=torch.bool)
    assert "call __assertfail" in _source(check, flags, 1)
    assert "call __assertfail" in _source(check, flags, 2**30)
    with pytest.raises(exc.BackendUnsupported, match="Int32 loop range"):
        _source(check, flags, 2**31 - 1)

    # Scalar steps must never become thread-block widths.
    original = per_thread_planning._largest_divisor_at_most

    def bounded(size, limit):
        assert size < 2**31
        return original(size, limit)

    monkeypatch.setattr(per_thread_planning, "_largest_divisor_at_most", bounded)
    with pytest.raises(exc.BackendUnsupported, match="Int32 loop range"):
        _source(check, flags, 2**32)


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _checked_indirect_memory(x, ids, mask, dest, seen, access: hl.constexpr):
    for row in hl.grid(ids.size(0)):
        indices = ids[row, :]
        if access == "unmasked":
            values = x[indices]
        elif access == "cross-edge":
            values = x[hl.arange(2) + 2]
        elif access == "bounded":
            values = x[indices.clamp(0, 2)]
        else:
            values = hl.load(x, [indices], extra_mask=mask[row, :])
        seen[row, :] = values
        if access == "masked":
            hl.store(dest, [indices], values, extra_mask=mask[row, :])
        else:
            dest[indices] = values
        torch._assert_async(torch.all(values >= 0))


@pytest.mark.parametrize("access", ["unmasked", "cross-edge", "bounded", "masked"])
def test_checked_indirect_memory_requires_bounds_only_for_unmasked_reads(access):
    x = torch.tensor([5, 6, 7])
    indices = torch.tensor([[-1, 0], [3, 1]])
    mask = torch.tensor([[True, False], [True, False]])
    dest, seen = torch.full_like(x, -8), torch.full_like(indices, -8)
    args = (x, indices, mask, dest, seen, access)
    with as_sm100():
        with patch.object(_checked_indirect_memory.settings, "ref_mode", RefMode.EAGER):
            cpu_bind(_checked_indirect_memory, args).run_ref(*args)
        if access == "masked":
            assert seen.tolist() == [[0, 0], [0, 0]] and dest.tolist() == [-8, -8, -8]
        elif access in ("unmasked", "bounded"):
            assert seen.tolist() == [[5, 5], [7, 6]] and dest.tolist() == [5, 6, -8]
        else:
            assert seen.tolist() == [[7, 7], [7, 7]]
        bound = cpu_bind(_checked_indirect_memory, args)
        config = helion.Config(block_sizes=[])
        if access in ("unmasked", "cross-edge"):
            with pytest.raises(exc.BackendUnsupported, match="unmasked tensor index"):
                bound.to_code(config)
        else:
            assert "call __assertfail" in bound.to_code(config)


@pytest.mark.parametrize(
    ("index_dtype", "scale", "narrow_tensor"),
    [
        (None, 2**30, False),
        (torch.int64, 2**30, False),
        (None, 7, False),
        (None, 2**30, True),
    ],
    ids=["default-wide-value", "wide-addresses", "fits-int32", "tensor-int32"],
)
def test_checked_loop_compares_against_wide_induction_product(
    index_dtype, scale, narrow_tensor
):
    @helion.kernel(
        backend="cute",
        static_shapes=True,
        index_dtype=index_dtype,
        config=helion.Config(block_sizes=[]),
    )
    def check(x, scale: hl.constexpr, narrow: hl.constexpr):
        for row in hl.grid(x.size(0)):
            for i in hl.grid(2, 4):
                if narrow:
                    # A tensor multiply wraps before it reaches the comparison.
                    value = (
                        torch.full((), i, dtype=torch.int32, device=x.device) * scale
                    )
                else:
                    value = i * scale
                torch._assert_async(x[row] < value, "index product")

    if index_dtype is None and scale == 2**30 and not narrow_tensor:
        # ATen compares to the wrapped scalar (-2**31); promoting the
        # Int32 tensor along with the Python integer would pass incorrectly.
        narrow_x = torch.tensor([0], dtype=torch.int32)
        with (
            patch.object(check.settings, "ref_mode", RefMode.EAGER),
            pytest.raises(RuntimeError, match="index product"),
        ):
            check(narrow_x, scale, False)
        with pytest.raises(exc.BackendUnsupported, match="narrow integer tensor"):
            _source(check, narrow_x, scale, False)

    low = -(2**31) if narrow_tensor else 2 * scale
    x = torch.tensor([low - 1], dtype=torch.int64)
    with patch.object(check.settings, "ref_mode", RefMode.EAGER):
        check(x, scale, narrow_tensor)
        with pytest.raises(RuntimeError, match="index product"):
            check(x + 1, scale, narrow_tensor)
    before, marker, _ = _source(check, x, scale, narrow_tensor).partition(
        "call __assertfail"
    )
    assert marker
    (index,) = re.findall(
        r"for (\w+) in range\(cutlass\.Int32\(2\), cutlass\.Int32\(4\), 1\)", before
    )
    if narrow_tensor:
        assert f"cutlass.Int32({index})" in before
        assert f"cutlass.Int64({index})" not in before
    elif scale == 7:
        assert f"7 * {index}" in before
    else:
        assert f"1073741824 * cutlass.Int64({index})" in before


def test_checked_iota_widens_before_computing_its_value():
    @helion.kernel(
        backend="cute",
        static_shapes=True,
        index_dtype=torch.int64,
        config=helion.Config(block_sizes=[]),
    )
    def check(x, start: hl.constexpr, stop: hl.constexpr, step: hl.constexpr):
        for row in hl.grid(1):
            v = torch.arange(start, stop, step, dtype=torch.int64, device=x.device)
            torch._assert_async(v.min() == x[row], "iota minimum")

    for start, stop, step in ((2**31 - 1, 2**31 + 2, 1), (-(2**31), 1, 2**30)):
        x = torch.tensor([start], dtype=torch.int64)
        with patch.object(check.settings, "ref_mode", RefMode.EAGER):
            check(x, start, stop, step)
            with pytest.raises(RuntimeError, match="iota minimum"):
                check(x + 1, start, stop, step)
        source = _source(check, x, start, stop, step)
        assert "call __assertfail" in source
        # An Int64 cast around the completed sum cannot undo Int32 wrap.
        assert re.search(r"cutlass\.Int64\(fragment_coordinate\w*\) \*", source)


def test_checked_unmasked_tile_must_fit_the_physical_slice():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def checked(x):
        for tile in hl.tile(4, block_size=4):
            torch._assert_async(torch.all(x[tile] > 0))

    eager = helion.kernel(
        checked.fn,
        backend="cute",
        static_shapes=True,
        ref_mode="eager",
        config=checked.configs[0],
    )
    for size in (1, 3, 4):
        x = torch.ones(size)
        eager(x)
        if size == 4:
            assert "call __assertfail" in _source(checked, x)
        else:
            with pytest.raises(exc.BackendUnsupported, match="unmasked tile"):
                _source(checked, x)


def test_checked_literal_scalar_reads_require_an_in_bounds_row():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def check(x, index: hl.constexpr):
        for _ in hl.grid(1):
            torch._assert_async(x[index, 0] == 0)

    x = torch.zeros(2, 1)
    for index in (-3, 2):
        with pytest.raises(exc.BackendUnsupported, match="scalar memory index"):
            _source(check, x, index)
    assert "call __assertfail" in _source(check, x, -2)


def test_checked_grid_scalar_reads_use_the_physical_dimension():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def direct(x, stop: hl.constexpr):
        for i in hl.grid(stop):
            torch._assert_async(x[i] == 0)

    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def nested(x, stop: hl.constexpr):
        for tile in hl.tile(stop, block_size=4):
            for i in hl.grid(tile.begin, tile.end):
                torch._assert_async(x[i] == 0)

    x = torch.zeros(16)
    for fn in (direct, nested):
        eager = helion.kernel(
            fn.fn,
            backend="cute",
            static_shapes=True,
            ref_mode="eager",
            config=fn.configs[0],
        )
        eager(x, 16)
        with pytest.raises(IndexError):
            eager(x, 17)
        with pytest.raises(exc.BackendUnsupported, match="scalar memory index"):
            _source(fn, x, 17)
        assert "call __assertfail" in _source(fn, x, 16)


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _checked_python_grid_value(
    x,
    out,
    shift: hl.constexpr,
    scale: hl.constexpr,
    save: hl.constexpr,
    bias: hl.constexpr = 0,
):
    for row in hl.grid(x.size(0)):
        for i in hl.grid(2, 4):
            value = (i + shift) * scale + bias
            if save:
                out[row, i] = value
            torch._assert_async(x[row] < value, "grid value")


@pytest.mark.parametrize("shift,scale", [(1, 1), (-1, 1), (0, 2**30)])
def test_checked_python_grid_values_reach_stores_and_predicates(shift, scale):
    # The offsets fit in an Int32 tensor; the product must widen for both uses.
    x = torch.tensor([0], dtype=torch.int32 if scale == 1 else torch.int64)
    out = torch.full((1, 4), -17, dtype=torch.int64)
    args = (x, out, shift, scale, True)
    with patch.object(_checked_python_grid_value.settings, "ref_mode", RefMode.EAGER):
        _checked_python_grid_value(*args)
    assert out.tolist() == [[-17, -17, (2 + shift) * scale, (3 + shift) * scale]]
    src = _source(_checked_python_grid_value, *args)
    assert "call __assertfail" in src
    stored = next(
        line.split(".store(", 1)[1] for line in src.splitlines() if ".store(" in line
    )
    widened = re.search(r"cutlass\.Int64\(fragment_tile\w*\)", stored)
    assert bool(widened) == (scale != 1)
    for x0 in (0, (2 + shift) * scale):
        actual = torch.full_like(out, -17)
        predicates = []
        simulate_independent_fragment(
            src,
            {"x": torch.full_like(x, x0)},
            {"out": actual},
            1,
            int32=np.int32,
            int64=np.int64,
            inline_asm=lambda args, observed=predicates, **kwargs: observed.append(
                bool(args[0])
            ),
        )
        assert actual.tolist() == out.tolist()
        assert predicates == [x0 < (i + shift) * scale for i in (2, 3)]


@pytest.mark.parametrize(
    "dtype,scale,bias",
    [
        (torch.int16, 16384, 0),
        (torch.float64, 2**62, 0),
        # The result fits Int64, but its multiply does not.
        (torch.float64, 2**62, -(2**63)),
    ],
)
def test_checked_python_grid_values_require_representable_arithmetic(
    dtype, scale, bias
):
    x = torch.tensor([-1 if bias else 0], dtype=dtype)
    out = torch.full((1, 4), -17, dtype=torch.int64)
    args = (x, out, 0, scale, False, bias)
    with patch.object(_checked_python_grid_value.settings, "ref_mode", RefMode.EAGER):
        if dtype == torch.int16:
            with pytest.raises(RuntimeError, match="grid value"):
                _checked_python_grid_value(*args)
        else:
            _checked_python_grid_value(*args)
    with pytest.raises(exc.BackendUnsupported, match="checked symbolic integer"):
        _source(_checked_python_grid_value, *args)


@helion.kernel(
    backend="cute",
    static_shapes=True,
    config=helion.Config(block_sizes=[], pid_type="flat"),
)
def _checked_python_grid_branch(flags, out, threshold: hl.constexpr):
    for i, j in hl.grid([46340, 46340], [46342, 46342]):
        if i * j > threshold:
            out[0] = 1
            torch._assert_async(flags[0], "branch taken")


def test_checked_symbolic_branch_takes_the_eager_path_on_product_overflow():
    flags = torch.tensor([True])
    expected = torch.tensor([-1])
    threshold = 2**31 - 1
    with patch.object(_checked_python_grid_branch.settings, "ref_mode", RefMode.EAGER):
        _checked_python_grid_branch(flags, expected, threshold)
        with pytest.raises(RuntimeError, match="branch taken"):
            _checked_python_grid_branch(~flags, expected, threshold)
    assert expected.tolist() == [1]
    src = _source(_checked_python_grid_branch, flags, expected, threshold)
    for flag in (True, False):
        predicates = []
        out = torch.tensor([-1])
        simulate_independent_fragment(
            src,
            {"flags": torch.tensor([flag])},
            {"out": out},
            4,
            int32=np.int32,
            int64=np.int64,
            inline_asm=lambda args, observed=predicates, **kwargs: observed.append(
                bool(args[0])
            ),
        )
        assert out.tolist() == [1]
        assert predicates == [flag]  # Only (46341, 46341) takes the branch.


def test_checked_nested_tile_rejects_a_reused_block_id():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[4])
    )
    def check(flags):
        width = hl.register_block_size(4)
        for outer in hl.tile(4, block_size=width):
            end = outer.end
            for inner in hl.tile(2, block_size=width):
                if end > 2:
                    torch._assert_async(flags[inner.begin], "outer end")

    flags = torch.zeros(4, dtype=torch.bool)
    with (
        patch.object(check.settings, "ref_mode", RefMode.EAGER),
        pytest.raises(RuntimeError, match="outer end"),
    ):
        check(flags)
    with pytest.raises(exc.BackendUnsupported, match="reuses an active tile block"):
        _source(check, flags)


def test_checked_implicit_integer_branch_preserves_truth_at_int32_wrap():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def check(flags):
        for _row in hl.grid(1):
            for i in hl.grid(3, 5):
                if i * 2**30:
                    torch._assert_async(flags[i], "truth")

    flags = torch.tensor([True, True, True, True, False])
    with (
        patch.object(check.settings, "ref_mode", RefMode.EAGER),
        pytest.raises(RuntimeError, match="truth"),
    ):
        check(flags)
    observed = []
    simulate_independent_fragment(
        _source(check, flags),
        {"flags": flags},
        {},
        1,
        int32=np.int32,
        int64=np.int64,
        inline_asm=lambda args, **kwargs: observed.append(bool(args[0])),
    )
    assert observed == [True, False]


def test_checked_float64_comparison_declines_a_runtime_float():
    @helion.kernel(
        backend="cute", static_shapes=False, config=helion.Config(block_sizes=[])
    )
    def check(x, threshold):
        for i in hl.grid(1):
            torch._assert_async(x[i] >= threshold, "rounded threshold")

    x = torch.tensor([1.0], dtype=torch.float64)
    threshold = 1.0 + 2**-40  # Rounds to 1.0 in the Float32 launcher ABI.
    with (
        patch.object(check.settings, "ref_mode", RefMode.EAGER),
        pytest.raises(RuntimeError, match="rounded threshold"),
    ):
        check(x, threshold)
    with pytest.raises(exc.BackendUnsupported, match="symbolic float.*precision"):
        _source(check, x, threshold)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_checked_grid_value_stores_to_fp8():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def fill(flags, out):
        for i in hl.grid(16):
            torch._assert_async(flags[i], "valid")
            out[i] = i

    flags = torch.ones(16, dtype=torch.bool, device=DEVICE)
    out = torch.empty(16, dtype=torch.float8_e4m3fn, device=DEVICE)
    fill(flags, out)
    torch.testing.assert_close(out.float().cpu(), torch.arange(16, dtype=torch.float32))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_unchecked_fragment_grid_value_stores_to_fp8():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def fill(x, prefix, out):
        for i in hl.grid(out.size(0)):
            prefix[i, :] = torch.cumsum(x[i, :], dim=0)
            out[i] = i

    x = torch.ones((16, 4), dtype=torch.int32, device=DEVICE)
    prefix = torch.empty_like(x)
    out = torch.empty(16, dtype=torch.float8_e4m3fn, device=DEVICE)
    fill(x, prefix, out)
    torch.testing.assert_close(prefix, x.cumsum(1, dtype=x.dtype))
    torch.testing.assert_close(out.float().cpu(), torch.arange(16, dtype=torch.float32))


def test_checked_python_grid_store_requires_representable_integer_destination():
    x = torch.zeros(1, dtype=torch.int64)
    out = torch.full((1, 4), -17, dtype=torch.int16)
    unsafe = (x, out, 0, 16384, True)
    safe = (x, out, 0, 8192, True)
    with patch.object(_checked_python_grid_value.settings, "ref_mode", RefMode.EAGER):
        with pytest.raises(RuntimeError, match="overflow"):
            _checked_python_grid_value(*unsafe)
        assert out.tolist() == [[-17, -17, -17, -17]]
        _checked_python_grid_value(*safe)
    assert out.tolist() == [[-17, -17, 16384, 24576]]
    with pytest.raises(exc.BackendUnsupported, match="torch.int16"):
        _source(_checked_python_grid_value, *unsafe)
    actual = torch.full_like(out, -17)
    simulate_independent_fragment(
        _source(_checked_python_grid_value, *safe),
        {"x": x},
        {"out": actual},
        1,
        int32=np.int32,
        int64=np.int64,
        inline_asm=lambda *args, **kwargs: None,
    )
    assert actual.tolist() == out.tolist()


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def _checked_scalar_destination(flags, out, scale: hl.constexpr, form: hl.constexpr):
    for row in hl.grid(1):
        torch._assert_async(flags[row], "valid")
        for i in hl.grid(2, 4):
            if form == "literal":
                out[row] = 32768
            elif form == "constexpr":
                out[row] = scale
            elif form == "index":
                out[row] = i * scale
            else:
                for tile in hl.tile(4, block_size=4):
                    if form == "masked":
                        hl.store(out, [tile], i * scale, extra_mask=flags[tile])
                    elif form == "constant_fill":
                        out[tile] = hl.full([tile], scale, dtype=out.dtype)
                    else:
                        out[tile] = hl.full([tile], i * scale, dtype=out.dtype)


@pytest.mark.parametrize(
    ("form", "dtype", "scale"),
    [
        ("literal", torch.int16, 0),
        ("constexpr", torch.int16, -32769),
        ("masked", torch.float16, 32768),
        ("fill", torch.float16, 32768),
    ],
)
def test_checked_scalar_destination_preserves_eager_overflow(form, dtype, scale):
    out = torch.full((4,), -17, dtype=dtype)
    args = (torch.ones(4, dtype=torch.bool), out, scale, form)
    with (
        patch.object(_checked_scalar_destination.settings, "ref_mode", RefMode.EAGER),
        pytest.raises(RuntimeError, match="overflow"),
    ):
        _checked_scalar_destination(*args)
    assert out.tolist() == [-17] * 4
    with pytest.raises(exc.BackendUnsupported, match=str(dtype)):
        _source(_checked_scalar_destination, *args)


@pytest.mark.parametrize(
    ("form", "dtype", "scale", "expected"),
    [
        ("constexpr", torch.int16, -32768, [-32768, -17, -17, -17]),
        ("constexpr", torch.int16, 32767.5, [32767, -17, -17, -17]),
        ("constexpr", torch.float16, 65504, [65504, -17, -17, -17]),
        ("fill", torch.float16, 21834, [65504] * 4),
    ],
)
def test_checked_scalar_destination_keeps_valid_conversions(
    form, dtype, scale, expected
):
    flags = torch.ones(4, dtype=torch.bool)
    out = torch.full((4,), -17, dtype=dtype)
    args = (flags, out, scale, form)
    with patch.object(_checked_scalar_destination.settings, "ref_mode", RefMode.EAGER):
        _checked_scalar_destination(*args)
    assert out.tolist() == expected
    actual = torch.full_like(out, -17)
    with np.errstate(over="ignore"):
        simulate_independent_fragment(
            _source(_checked_scalar_destination, *args),
            {"flags": flags},
            {"out": actual},
            1,
            inline_asm=lambda *args, **kwargs: None,
        )
    assert actual.tolist() == expected


def test_checked_half_index_checks_cuda_assignment_before_rounding():
    # CUDA rejects 65,536 in scalar indexed assignment; the one-element
    # CPU fast path can round it to infinity. This is the CUDA backend.
    args = (
        torch.ones(4, dtype=torch.bool),
        torch.empty(4, dtype=torch.float16),
        32768,
        "index",
    )
    with pytest.raises(exc.BackendUnsupported, match="torch.float16"):
        _source(_checked_scalar_destination, *args)


@pytest.mark.parametrize("fill", [False, True])
def test_checked_runtime_float_assignment_cannot_use_the_binding_hint(fill):
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def assign(flags, out, value, fill: hl.constexpr):
        for row in hl.grid(1):
            torch._assert_async(flags[row], "valid")
            if fill:
                for tile in hl.tile(4, block_size=4):
                    out[tile] = hl.full([tile], value, dtype=out.dtype)
            else:
                out[row] = value

    flags = torch.ones(1, dtype=torch.bool)
    out = torch.zeros(4, dtype=torch.float16)
    # Binding with 1.25 must not admit the same kernel for a later 65536.0.
    with pytest.raises(exc.BackendUnsupported, match="symbolic float.*torch.float16"):
        _source(assign, flags, out, 1.25, fill)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_checked_fp8_fills_use_the_same_finite_range(dtype):
    flags = torch.ones(4, dtype=torch.bool)
    out = torch.empty(4, dtype=dtype)
    limit = int(torch.finfo(dtype).max)
    # The symbolic fill visits 2 * scale and 3 * scale.
    for form, safe, overflow in (
        ("constant_fill", limit, limit + 1),
        ("fill", limit // 3, limit // 2 + 1),
    ):
        assert ".store(" in _source(_checked_scalar_destination, flags, out, safe, form)
        with pytest.raises(exc.BackendUnsupported, match=str(dtype)):
            _source(_checked_scalar_destination, flags, out, overflow, form)
    if dtype == torch.float8_e5m2:
        assert ".store(" in _source(
            _checked_scalar_destination, flags, out, float("inf"), "constant_fill"
        )
    else:
        with pytest.raises(exc.BackendUnsupported, match=str(dtype)):
            _source(
                _checked_scalar_destination, flags, out, float("inf"), "constant_fill"
            )


def test_checked_unsigned_scalar_assignment_accepts_wrapping_integer_range():
    flags = torch.ones(4, dtype=torch.bool)
    out = torch.zeros(4, dtype=torch.uint8)
    args = (flags, out, -85, "index")
    with patch.object(_checked_scalar_destination.settings, "ref_mode", RefMode.EAGER):
        _checked_scalar_destination(*args)
    assert out.tolist() == [1, 0, 0, 0]  # -255 wraps to 1.
    assert ".store(" in _source(_checked_scalar_destination, *args)
    with pytest.raises(exc.BackendUnsupported, match="torch.uint8"):
        _source(_checked_scalar_destination, flags, out, -86, "index")


def test_checked_half_factory_distinguishes_scalar_assignment_from_tensor_conversion():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def factory(x, out, mode: hl.constexpr):
        for row in hl.grid(1):
            torch._assert_async(x[row] > 0, "valid")
            if mode == "scalar_tensor":
                out[row] = torch.scalar_tensor(
                    65536, dtype=out.dtype, device=out.device
                )
            elif mode == "tensor":
                out[row] = torch.tensor(65536, dtype=out.dtype, device=out.device)
            else:
                out[row] = x[row].to(out.dtype)

    x = torch.tensor([65536], dtype=torch.int64)
    out = torch.zeros(1, dtype=torch.float16)
    # CUDA scalar_tensor performs a checked fill, unlike constructing from a
    # Python list/scalar or converting an existing tensor.
    with pytest.raises(exc.BackendUnsupported, match="torch.float16"):
        _source(factory, x, out, "scalar_tensor")
    for mode in ("tensor", "to"):
        with mock_cuda_unavailable():
            source = cpu_bind(factory, (x, out, mode)).to_code(factory.configs[0])
        with np.errstate(over="ignore"):
            simulate_independent_fragment(
                source,
                {"x": x},
                {"out": out},
                1,
                inline_asm=lambda *args, **kwargs: None,
            )
        assert torch.isposinf(out[0])
        out.zero_()


if __name__ == "__main__" and sys.argv[1:2] == ["--check-assertion"]:
    from cuda.bindings import driver

    counts = _run_valid_prefix()
    # The total starts at 128. Each input violates only its own assertion.
    counts[7, 5] = {"capacity": 2, "negative": -128}[sys.argv[2]]
    # With launch blocking enabled PyTorch can report the fault here instead
    # of at the synchronization. Require the driver's sticky ASSERT either way.
    try:
        _checked_prefix(counts, True, check_nonnegative=True)
    except RuntimeError as error:
        if "device-side assert" not in str(error):
            raise
    (status,) = driver.cuCtxSynchronize()
    assert status == driver.CUresult.CUDA_ERROR_ASSERT, status
    print("observed CUDA assertion 710", flush=True)
