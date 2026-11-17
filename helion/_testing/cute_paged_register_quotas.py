"""Fixtures and checks of the paired schedule's register-quota tests.

The CPU side compiles the decode-step paged attention kernel, and the
partition fixture's kernel on a persistent work partition, offline
(``cute_offline.compile_offline``), through the launcher's own compile
function and the CuTe DSL's assembler, and reads the PTX the assembler was
given and the SASS it produced: no CUDA context is created. The GPU side
replays a split against the reference and compares it with the default
split bit for bit, under one unit per CTA and under the persistent
partitions.
"""

from __future__ import annotations

import functools
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING
from typing import cast

import torch

from . import cpu_render_target
from .cute_codegen import cpu_bind
from .cute_offline import PARTITION_UNITS
from .cute_offline import OfflineKernel
from .cute_offline import compile_offline
from .cute_offline import partition_bound
from .cute_optional_lse import bind_paged_attention_length_replay
from .cute_optional_lse import check_optional_lse_replay
from .cute_optional_lse import initialize_paged_attention_inputs
from .cute_optional_lse import optional_lse_fixture
from .cute_paged_attention_handoff import paged_decode_args
from .cute_paged_attention_handoff import paged_decode_attention
from .cute_paged_base2 import base2_attention_fixture
from .cute_paged_partition import PAGES
from .cute_paged_partition import partition_attention_args
from .cute_paged_partition import partition_attention_kernel
from .cute_paged_partition import partition_config
from .cute_paged_partition import randomize_partition_inputs
from .cute_paged_partition import set_request_pages
from .cute_paged_v_lag import paired_v_lag_config
from .cute_sass import cute_launcher_sass
from .poisoned_outputs import poisoned_launch
import helion
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_CHOICES,
)
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_DEFAULT,
)
from helion._compiler.cute.paged_attention_registers import MMA_DAG_REGISTER_QUOTAS_KEY
from helion._compiler.cute.paged_attention_registers import parse_register_quotas
from helion.runtime.cute import sass_check

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

_SASS_SETMAXREG = re.compile(
    r"^USETMAXREG\.(TRY_ALLOC|DEALLOC)\.CTAPOOL .*?(0x[0-9a-f]+)$"
)


def sass_quotas(sass: str) -> list[tuple[str, int]]:
    """``USETMAXREG`` actions and counts of a listing, in program order."""
    found = []
    for text in sass_check.instructions(sass):
        match = _SASS_SETMAXREG.match(text)
        if match is not None:
            found.append((match.group(1), int(match.group(2), 16)))
    return found


@functools.cache
def decode_bound(query_len: int) -> tuple[BoundKernel[object], tuple[object, ...]]:
    """The decode-step kernel bound on CPU: four query tokens are the draft
    step (32 query columns), five the verify step (40 columns, with LSE)."""
    args = list(paged_decode_args(4, query_len, "cpu"))
    if query_len == 5:
        args[8] = torch.empty((4 * query_len * 16,), dtype=torch.float32)
        args[11] = True
    inputs = tuple(args)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", paged_decode_attention), inputs)
    return bound, inputs


def decode_config(bound: BoundKernel[object], **overrides: object) -> helion.Config:
    """The kernel's own default (the paired schedule's leading seed) with
    ``overrides``."""
    assert bound.host_function is not None
    with (
        cpu_render_target(fake_device_sm_count=148),
        bound.env,
        bound.host_function,
    ):
        default = bound.config_spec.default_config()
    return helion.Config.from_dict({**default.config, **overrides})


@functools.cache
def decode_kernel(query_len: int, quotas: str) -> OfflineKernel:
    """The decode-step kernel of ``decode_bound(query_len)`` with the split
    ``quotas``, compiled offline once per process."""
    bound, args = decode_bound(query_len)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(
            decode_config(bound, **{MMA_DAG_REGISTER_QUOTAS_KEY: quotas})
        )
    with tempfile.TemporaryDirectory(prefix="helion-cute-quotas-") as dump_dir:
        return compile_offline(bound, args, source, Path(dump_dir))


# The GPU tests run the partition fixture's ``PARTITION_UNITS`` units on a
# capacity of eight CTAs, so a CTA runs up to four units and the balanced
# partition cuts a unit at the boundaries between CTAs: up to seven units
# run in two parts.
PARTITION_CAPACITY = 8


@functools.cache
def partition_kernel(partition: str, quotas: str) -> OfflineKernel:
    """The fixture's kernel on the persistent ``partition`` with the split
    ``quotas``, compiled offline once per process."""
    bound, args = partition_bound()
    config = partition_config(
        partition, cute_mma_dag_v_lag=2, cute_mma_dag_register_quotas=quotas
    )
    with cpu_render_target(fake_device_sm_count=148, num_sm=148):
        source = bound.to_code(config)
    with tempfile.TemporaryDirectory(prefix="helion-cute-quotas-") as dump_dir:
        return compile_offline(bound, args, source, Path(dump_dir))


def one_unit_quota_order(counts: tuple[int, ...]) -> list[tuple[str, int]]:
    """The ``setmaxnreg`` sites of a one-unit CTA in program order, the
    roles' order in the kernel: the compute groups' raise, then the K
    conversion, V conversion and control warps' counts."""
    return [("inc", counts[0]), *(("dec", count) for count in counts[1:])]


def persistent_quota_order(counts: tuple[int, ...]) -> list[tuple[str, int]]:
    """The ``setmaxnreg`` sites of a persistent CTA in program order: its
    four control warps take their quota behind the scan, ahead of the
    roles' dispatch; then the compute groups' raise and the conversion
    warps' count (one site where K and V conversion have one quota, a site
    each otherwise)."""
    compute, key, value, control = counts
    conversion = [("dec", key)] if key == value else [("dec", key), ("dec", value)]
    return [("dec", control), ("inc", compute), *conversion]


def sass_quota_order(order: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """``order``, sites as PTX names them (``OfflineKernel.ptx_quotas``), as
    ``sass_quotas`` reads them from the listing of the same kernel."""
    return [
        ("TRY_ALLOC" if kind == "inc" else "DEALLOC", count) for kind, count in order
    ]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(
        torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
    )


def check_register_quotas_replay(
    quotas: str,
    form: str,
    query_len: int,
    base2: bool,
    fast_math: bool,
    device: torch.device | str,
) -> None:
    """A split gives the default split's outputs bit for bit, issues its own
    ``setmaxnreg`` counts in the launched kernel, and passes the reference
    and replay checks of the paired schedule."""
    if base2:
        kernel, args = base2_attention_fixture(form, query_len, True, device)
        args = (*args, False)
    else:
        kernel, args = optional_lse_fixture(form, True, device, query_len=query_len)
        args = (*args, False, False)
    bound, args = bind_paged_attention_length_replay(
        kernel, args, form, fast_math=fast_math, reverse_pages=False
    )
    initialize_paged_attention_inputs(args)
    results = []
    for split in (MMA_DAG_REGISTER_QUOTAS_DEFAULT, quotas):
        config = paired_v_lag_config(2)
        config.config["block_sizes"] = [helion.next_power_of_2(query_len * 8)]
        config.config[MMA_DAG_REGISTER_QUOTAS_KEY] = split
        source = bound.to_code(config)
        assert "paged_attention_paired_body(" in source
        counts = parse_register_quotas(split)
        assert counts is not None
        assert (f"register_quotas={counts!r}" in source) == (
            split != MMA_DAG_REGISTER_QUOTAS_DEFAULT
        )
        compiled = bound.compile_config(config)
        output, lse = cast("tuple[torch.Tensor, torch.Tensor]", compiled(*args))
        torch.cuda.synchronize()
        results.append((_bits(output).clone(), _bits(lse).clone()))
        # The launched kernel carries the split at every role's site.
        assert sass_quotas(cute_launcher_sass(compiled)) == sass_quota_order(
            one_unit_quota_order(counts)
        )
    for default, changed in zip(*results, strict=True):
        assert torch.equal(default, changed)
    check_optional_lse_replay(
        bound,
        config,
        args,
        form,
        True,
        reference_scores=base2,
        base2_scores=base2,
    )


def check_partition_register_quotas(partition: str, device: torch.device | str) -> None:
    """On the persistent ``partition`` every other admitted split gives the
    default split's outputs bit for bit, every element stored by its own
    launch, and the launched kernel issues the split's counts at every
    role's site.

    The fixture runs on ``PARTITION_CAPACITY`` CTAs of ``device``, bound once,
    and compiles the partition's kernel once per split. The launches
    complete: the splits that use the whole pool leave the compute groups'
    raise nothing to spare, so a role of the persistent CTA that kept more
    registers than its quota would hang here.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    args = partition_attention_args(units=PARTITION_UNITS, device=device)
    randomize_partition_inputs(args, 0)
    bound = partition_attention_kernel(reserved_sms=sms - PARTITION_CAPACITY).bind(args)
    kernels = {}
    for quotas in MMA_DAG_REGISTER_QUOTAS_CHOICES:
        config = partition_config(
            partition, cute_mma_dag_v_lag=2, cute_mma_dag_register_quotas=quotas
        )
        source = bound.to_code(config)
        assert f"partition={partition!r}" in source
        counts = parse_register_quotas(quotas)
        assert (f"register_quotas={counts!r}" in source) == (
            quotas != MMA_DAG_REGISTER_QUOTAS_DEFAULT
        )
        kernels[quotas] = bound.compile_config(config)
    default = kernels.pop(MMA_DAG_REGISTER_QUOTAS_DEFAULT)
    requests = cast("torch.Tensor", args[6]).numel()
    generator = torch.Generator().manual_seed(requests)
    patterns = {
        "uniform": [PAGES] * requests,
        "ragged": torch.randint(
            0, PAGES + 1, (requests,), generator=generator
        ).tolist(),
        # Units without pages between units with pages.
        "sparse": [
            0 if index % 3 == 0 else 1 + index * 5 % PAGES for index in range(requests)
        ],
    }
    for name, pages in patterns.items():
        set_request_pages(args, pages)
        expected = poisoned_launch(lambda: default(*args))
        assert torch.isfinite(expected[1].view(torch.float32)).any(), name
        for quotas, changed in kernels.items():
            actual = poisoned_launch(lambda changed=changed: changed(*args))
            for value, reference in zip(actual, expected, strict=True):
                assert torch.equal(value, reference), (partition, quotas, name)
    for quotas, changed in kernels.items():
        counts = parse_register_quotas(quotas)
        assert counts is not None
        assert sass_quotas(cute_launcher_sass(changed)) == sass_quota_order(
            persistent_quota_order(counts)
        )
