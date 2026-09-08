from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

from benchmarks.cute import kda_prefill_staged
from benchmarks.cute.kda_prefill_kernels import KDA_PREPARE_CONFIG
from benchmarks.cute.kda_prefill_kernels import KDA_RECURRENCE_CONFIG
from benchmarks.cute.kda_prefill_kernels import kda_chunk_prepare
from benchmarks.cute.kda_prefill_kernels import kda_chunk_recurrence
from benchmarks.cute.kda_prefill_staged import CuteOriginAuxDag
from benchmarks.cute.kda_prefill_staged import KdaStagedResources
from benchmarks.cute.kda_prefill_staged import allocate_kda_factor_workspace
from benchmarks.cute.kda_prefill_staged import create_staged_kda_resources
from benchmarks.cute.kda_prefill_staged import kda_group_host_metadata
from benchmarks.cute.kda_prefill_staged import staged_longest_partition
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import cute_cuda_graph


@pytest.mark.parametrize(
    ("offsets", "longest", "rest"),
    (
        ((0, 3), 0, ()),
        ((0, 4, 7, 12), 2, ((0, 2),)),
        ((11, 14, 20, 22), 1, ((0, 1), (2, 3))),
        ((0, 5, 10), 0, ((1, 2),)),
    ),
)
def test_staged_longest_partition(
    offsets: tuple[int, ...],
    longest: int,
    rest: tuple[tuple[int, int], ...],
) -> None:
    plan = staged_longest_partition(offsets)

    assert plan.longest_sequence == longest
    assert plan.longest_range == (longest, longest + 1)
    assert plan.rest_ranges == rest


@pytest.mark.parametrize("offsets", ((), (0,), (0, 2, 1), (0, 5, 5), (0, 0, 0)))
def test_staged_longest_partition_rejects_invalid_offsets(
    offsets: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        staged_longest_partition(offsets)


def test_staged_longest_partition_rejects_non_integer_offsets() -> None:
    with pytest.raises(TypeError, match="must contain integers"):
        staged_longest_partition((0, 1.5))  # type: ignore[arg-type]


def test_cute_origin_aux_dag_launch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[str] = []

    class FakeStream:
        device = torch.device("cuda", 0)

        def wait_event(self, event: Any) -> None:
            operations.append(f"aux_wait:{event.name}")

    class FakeOriginStream:
        def wait_event(self, event: Any) -> None:
            operations.append(f"origin_wait:{event.name}")

    class FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream: Any = None) -> None:
            operations.append(
                f"record:{self.name}:{'current' if stream is None else 'origin'}"
            )

    auxiliary_stream = FakeStream()
    origin_stream = FakeOriginStream()

    @contextmanager
    def use_stream(stream: Any):
        assert stream is auxiliary_stream
        operations.append("enter_aux")
        try:
            yield
        finally:
            operations.append("exit_aux")

    @contextmanager
    def use_device(device: Any):
        assert device == auxiliary_stream.device
        operations.append("enter_device")
        try:
            yield
        finally:
            operations.append("exit_device")

    monkeypatch.setattr(torch.cuda, "device", use_device)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: origin_stream)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    dag = CuteOriginAuxDag(
        auxiliary_stream=auxiliary_stream,  # type: ignore[arg-type]
        fork_event=FakeEvent("fork"),  # type: ignore[arg-type]
        done_event=FakeEvent("done"),  # type: ignore[arg-type]
    )

    result = dag.launch(
        lambda: operations.append("prefix"),
        lambda: operations.append("origin") or "result",
        lambda: operations.append("auxiliary"),
    )

    assert result == "result"
    assert operations == [
        "enter_device",
        "prefix",
        "record:fork:origin",
        "aux_wait:fork",
        "origin",
        "enter_aux",
        "auxiliary",
        "record:done:current",
        "exit_aux",
        "origin_wait:done",
        "exit_device",
    ]


def test_packed_mixed_staged_group_metadata() -> None:
    offsets = (0, 1300, 1847, 3895, 4858, 5129, 8192)
    partition = staged_longest_partition(offsets)

    assert partition.longest_sequence == 5
    assert partition.longest_range == (5, 6)
    assert partition.rest_ranges == ((0, 5),)

    critical = kda_group_host_metadata(offsets, partition.longest_range)
    rest = kda_group_host_metadata(offsets, partition.rest_ranges[0])
    assert critical.cu_seqlens == (5129, 8192)
    assert critical.cu_chunks == (0, 192)
    assert critical.chunk_to_seq == (0,) * 192
    assert rest.cu_seqlens == (0, 1300, 1847, 3895, 4858, 5129)
    assert rest.cu_chunks == (0, 82, 117, 245, 306, 323)
    assert tuple(rest.chunk_to_seq.count(index) for index in range(5)) == (
        82,
        35,
        128,
        61,
        17,
    )


def test_group_metadata_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="invalid sequence range"):
        kda_group_host_metadata((0, 8, 16), (1, 3))


def test_factor_workspace_is_aligned_and_packed() -> None:
    workspace = allocate_kda_factor_workspace(2, 3, "cpu")

    assert workspace.storage.data_ptr() % 256 == 0
    assert workspace.kd.shape == (1, 2, 48, 128)
    assert workspace.qd.shape == workspace.kd.shape
    assert workspace.ak.shape == workspace.kd.shape
    assert workspace.aq.shape == (1, 2, 3, 256)
    assert workspace.g_total.shape == (1, 2, 3, 128)
    storage = workspace.storage.untyped_storage()._cdata
    assert {
        workspace.kd.untyped_storage()._cdata,
        workspace.qd.untyped_storage()._cdata,
        workspace.ak.untyped_storage()._cdata,
        workspace.aq.untyped_storage()._cdata,
        workspace.g_total.untyped_storage()._cdata,
    } == {storage}


def test_interior_longest_rejects_origin_aux_and_launches_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offsets = (0, 4, 12, 14, 17)
    partition = staged_longest_partition(offsets)
    groups = tuple(
        kda_prefill_staged.KdaStagedGroup(
            kda_prefill_staged.materialize_kda_group_metadata(
                kda_group_host_metadata(offsets, sequence_range), "cpu"
            ),
            allocate_kda_factor_workspace(1, 2, "cpu"),
        )
        for sequence_range in (partition.longest_range, *partition.rest_ranges)
    )
    operations: list[tuple[str, tuple[int, int]]] = []
    metadata_ranges = {
        id(group.metadata.cu_seqlens): group.metadata.host.sequence_range
        for group in groups
    }

    def fake_prepare(*args: Any) -> None:
        operations.append(("prepare", metadata_ranges[id(args[6])]))

    def fake_recurrence(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        operations.append(("recurrence", metadata_ranges[id(args[8])]))
        return args[6], args[7]

    class FakeDag:
        def launch(self, prefix: Any, origin: Any, auxiliary: Any) -> object:
            prefix()
            result = origin()
            auxiliary()
            return result

    monkeypatch.setattr(kda_prefill_staged, "kda_chunk_prepare", fake_prepare)
    monkeypatch.setattr(kda_prefill_staged, "kda_chunk_recurrence", fake_recurrence)
    resources = KdaStagedResources(
        critical=groups[0],
        auxiliary=groups[1:],
        dag=FakeDag(),  # type: ignore[arg-type]
    )
    q = torch.empty(1, offsets[-1], 1, 128)
    values = torch.empty_like(q)
    output = torch.empty_like(q)
    state = torch.empty(len(offsets) - 1, 1, 128, 128)

    with pytest.raises(ValueError, match="longest sequence is interior"):
        kda_prefill_staged.launch_staged_kda_prefill(
            resources,
            q=q,
            k=q,
            gate=q,
            beta_logits=torch.empty(1, offsets[-1], 1),
            a_log=torch.empty(1),
            dt_bias=torch.empty(1, 128),
            values=values,
            output=output,
            state=state,
            scale=128**-0.5,
            gate_scale_log2=-5.0,
        )
    assert operations == []

    actual_output, actual_state = kda_prefill_staged.launch_staged_kda_prefill(
        resources,
        q=q,
        k=q,
        gate=q,
        beta_logits=torch.empty(1, offsets[-1], 1),
        a_log=torch.empty(1),
        dt_bias=torch.empty(1, 128),
        values=values,
        output=output,
        state=state,
        scale=128**-0.5,
        gate_scale_log2=-5.0,
        topology="serial",
    )

    assert actual_output is output
    assert actual_state is state
    assert operations == [
        ("prepare", (1, 2)),
        ("recurrence", (1, 2)),
        ("prepare", (0, 1)),
        ("recurrence", (0, 1)),
        ("prepare", (2, 4)),
        ("recurrence", (2, 4)),
    ]

    operations.clear()

    def injected_prepare(*args: Any) -> None:
        operations.append(("injected_prepare", metadata_ranges[id(args[6])]))

    def injected_recurrence(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        operations.append(("injected_recurrence", metadata_ranges[id(args[8])]))
        return args[6], args[7]

    kda_prefill_staged.launch_staged_kda_prefill(
        resources,
        q=q,
        k=q,
        gate=q,
        beta_logits=torch.empty(1, offsets[-1], 1),
        a_log=torch.empty(1),
        dt_bias=torch.empty(1, 128),
        values=values,
        output=output,
        state=state,
        scale=128**-0.5,
        gate_scale_log2=-5.0,
        topology="serial",
        prepare_kernels=(injected_prepare,) * len(groups),
        recurrence_kernels=(injected_recurrence,) * len(groups),
    )
    assert operations == [
        ("injected_prepare", (1, 2)),
        ("injected_recurrence", (1, 2)),
        ("injected_prepare", (0, 1)),
        ("injected_recurrence", (0, 1)),
        ("injected_prepare", (2, 4)),
        ("injected_recurrence", (2, 4)),
    ]

    operations.clear()
    one_aux_resources = KdaStagedResources(
        critical=groups[0],
        auxiliary=(groups[1],),
        dag=FakeDag(),  # type: ignore[arg-type]
    )
    kda_prefill_staged.launch_staged_kda_prefill(
        one_aux_resources,
        q=q,
        k=q,
        gate=q,
        beta_logits=torch.empty(1, offsets[-1], 1),
        a_log=torch.empty(1),
        dt_bias=torch.empty(1, 128),
        values=values,
        output=output,
        state=state,
        scale=128**-0.5,
        gate_scale_log2=-5.0,
        topology="origin_aux",
        prepare_kernels=(injected_prepare,) * 2,
        recurrence_kernels=(injected_recurrence,) * 2,
    )
    assert operations == [
        ("injected_prepare", (1, 2)),
        ("injected_recurrence", (1, 2)),
        ("injected_prepare", (0, 1)),
        ("injected_recurrence", (0, 1)),
    ]

    with pytest.raises(ValueError, match="unsupported staged KDA topology"):
        kda_prefill_staged.launch_staged_kda_prefill(
            resources,
            q=q,
            k=q,
            gate=q,
            beta_logits=torch.empty(1, offsets[-1], 1),
            a_log=torch.empty(1),
            dt_bias=torch.empty(1, 128),
            values=values,
            output=output,
            state=state,
            scale=128**-0.5,
            gate_scale_log2=-5.0,
            topology="invalid",  # type: ignore[arg-type]
        )

    operations.clear()

    class FailingDag:
        def launch(self, prefix: Any, origin: Any, auxiliary: Any) -> object:
            raise AssertionError("single-group launch must not enter the staged DAG")

    fixed_resources = KdaStagedResources(
        critical=groups[0],
        auxiliary=(),
        dag=FailingDag(),  # type: ignore[arg-type]
    )
    kda_prefill_staged.launch_staged_kda_prefill(
        fixed_resources,
        q=q,
        k=q,
        gate=q,
        beta_logits=torch.empty(1, offsets[-1], 1),
        a_log=torch.empty(1),
        dt_bias=torch.empty(1, 128),
        values=values,
        output=output,
        state=state,
        scale=128**-0.5,
        gate_scale_log2=-5.0,
        prepare_kernels=(injected_prepare,),
        recurrence_kernels=(injected_recurrence,),
    )
    assert operations == [
        ("injected_prepare", (1, 2)),
        ("injected_recurrence", (1, 2)),
    ]


def _require_sm100_runtime() -> torch.device:
    from helion._compat import requires_cuda_version

    if not torch.cuda.is_available() or not requires_cuda_version("13"):
        pytest.skip("staged KDA runtime test requires CUDA >= 13")
    pytest.importorskip("cutlass.cute")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 10:
        pytest.skip("staged KDA runtime test requires SM100")
    return device


@torch.inference_mode()
def _safe_gate_token_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    lower_bound: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_f32 = q.float()
    k_f32 = k.float()
    q_f32 *= torch.rsqrt(q_f32.square().sum(-1, keepdim=True).clamp_min(1.0e-24))
    k_f32 *= torch.rsqrt(k_f32.square().sum(-1, keepdim=True).clamp_min(1.0e-24))
    q_f32 = q_f32.to(torch.bfloat16).float()
    k_f32 = k_f32.to(torch.bfloat16).float()
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(
            torch.exp(a_log)[None, None, :, None]
            * (gate.float() + dt_bias[None, None, :, :])
        )
    )
    beta = torch.sigmoid(beta_logits.float()).to(torch.bfloat16).float()
    state = initial_state.float()
    output = torch.empty_like(values, dtype=torch.float32)

    for token in range(q.size(1)):
        state *= decay[:, token].unsqueeze(-2)
        prediction = torch.einsum("bhvk,bhk->bhv", state, k_f32[:, token])
        residual = (values[:, token].float() - prediction) * beta[:, token, :, None]
        state += residual.unsqueeze(-1) * k_f32[:, token].unsqueeze(-2)
        output[:, token] = torch.einsum("bhvk,bhk->bhv", state, q_f32[:, token]) * scale
        if (token + 1) % 16 == 0:
            state = state.to(torch.bfloat16).float()

    return output, state.to(torch.bfloat16)


@pytest.mark.parametrize(
    ("dv_partitions", "expected_kind"),
    (
        (2, "chunk_recurrence_sm100"),
        (4, "chunk_recurrence_warp_dv4"),
    ),
)
@skipUnlessBackends(["cute"])
def test_fixed_h12_t512_staged_prefill_numerical_cuda(
    dv_partitions: int,
    expected_kind: str,
) -> None:
    device = _require_sm100_runtime()
    generator = torch.Generator(device=device).manual_seed(20260907)
    tokens = 512
    heads = 12
    width = 128
    shape = (1, tokens, heads, width)

    def random_bf16(tensor_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            tensor_shape,
            generator=generator,
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16)

    q = random_bf16(shape)
    k = random_bf16(shape)
    q[:, 0].fill_(1.0e-5)
    k[:, 0].fill_(1.0e-5)
    gate = random_bf16(shape)
    beta_logits = random_bf16(shape[:-1])
    values = random_bf16(shape)
    a_log = torch.rand(heads, generator=generator, dtype=torch.float32, device=device)
    dt_bias = torch.rand(
        (heads, width), generator=generator, dtype=torch.float32, device=device
    )
    initial_state = (
        torch.randn(
            (1, heads, width, width),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    state = initial_state.clone()
    output = torch.empty_like(values)
    scale = width**-0.5
    lower_bound = -5.0
    gate_scale_log2 = lower_bound * math.log2(math.e)

    expected_output, expected_state = _safe_gate_token_reference(
        q,
        k,
        gate,
        beta_logits,
        a_log,
        dt_bias,
        values,
        initial_state,
        lower_bound=lower_bound,
        scale=scale,
    )
    resources = create_staged_kda_resources((0, tokens), heads=heads, device=device)
    group = resources.critical
    metadata = group.metadata
    workspace = group.workspace
    prepare_args = (
        q,
        k,
        gate,
        beta_logits,
        a_log,
        dt_bias,
        metadata.cu_seqlens,
        metadata.cu_chunks,
        metadata.chunk_to_seq,
        workspace.kd,
        workspace.qd,
        workspace.ak,
        workspace.aq,
        workspace.g_total,
        None,
        gate_scale_log2,
    )
    recurrence_args = (
        workspace.kd,
        workspace.qd,
        workspace.ak,
        workspace.aq,
        workspace.g_total,
        values,
        output,
        state,
        metadata.cu_seqlens,
        metadata.cu_chunks,
        scale,
    )
    prepare = kda_chunk_prepare.bind(prepare_args).compile_config(
        helion.Config.from_dict(
            {
                **KDA_PREPARE_CONFIG.config,
                "cute_chunk_prepare_schedule": "split_alias_cpc2",
            }
        ),
        allow_print=False,
    )
    recurrence_config = {
        **KDA_RECURRENCE_CONFIG.config,
        "cute_chunk_recurrence_dv_partitions": dv_partitions,
    }
    if dv_partitions == 4:
        recurrence_config["cute_chunk_recurrence_register_cap"] = 72
    recurrence = kda_chunk_recurrence.bind(recurrence_args).compile_config(
        helion.Config.from_dict(recurrence_config),
        allow_print=False,
    )
    prepare_device = prepare.__globals__["_helion_kda_chunk_prepare"]
    recurrence_device = recurrence.__globals__["_helion_kda_chunk_recurrence"]
    assert prepare_device._helion_cute_wrapper_plans[0]["kind"] == "chunk_prepare_tma"
    assert recurrence_device._helion_cute_wrapper_plans[0]["kind"] == expected_kind

    actual_output, actual_state = kda_prefill_staged.launch_staged_kda_prefill(
        resources,
        q=q,
        k=k,
        gate=gate,
        beta_logits=beta_logits,
        a_log=a_log,
        dt_bias=dt_bias,
        values=values,
        output=output,
        state=state,
        scale=scale,
        gate_scale_log2=gate_scale_log2,
        prepare_kernels=(prepare,),
        recurrence_kernels=(recurrence,),
    )
    torch.cuda.synchronize(device)

    assert actual_output is output
    assert actual_state is state
    torch.testing.assert_close(
        actual_output.float(), expected_output, atol=2e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=2e-2, rtol=1e-2
    )

    state.copy_(initial_state)
    with cute_cuda_graph() as captured:
        kda_prefill_staged.launch_staged_kda_prefill(
            resources,
            q=q,
            k=k,
            gate=gate,
            beta_logits=beta_logits,
            a_log=a_log,
            dt_bias=dt_bias,
            values=values,
            output=output,
            state=state,
            scale=scale,
            gate_scale_log2=gate_scale_log2,
            prepare_kernels=(prepare,),
            recurrence_kernels=(recurrence,),
        )
    # The captured prepare launch must read and transform raw A_log on every
    # replay; a host-precomputed decay buffer would silently stay stale here.
    a_log.add_(0.75)
    mutated_output, mutated_state = _safe_gate_token_reference(
        q,
        k,
        gate,
        beta_logits,
        a_log,
        dt_bias,
        values,
        initial_state,
        lower_bound=lower_bound,
        scale=scale,
    )
    state.copy_(initial_state)
    output.fill_(float("nan"))
    captured.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output.float(), mutated_output, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(
        state.float(), mutated_state.float(), atol=2e-2, rtol=1e-2
    )
