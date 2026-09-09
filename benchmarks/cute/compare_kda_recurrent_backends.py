"""Reproduce the reportable recurrent KDA decode comparisons.

The two scoped rows are T1 unbounded-softplus (N=15, H=32) and T3 bounded
speculative decode (N=8, H=16). FlashInfer CuTe DSL and Helion CuTe each get a
private one-invocation CUDA graph. State restoration and cold-L2 flushing are
outside the timed interval, and samples alternate in balanced ABBA/BAAB order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from typing import Callable
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute.kda_benchmark_utils import DEFAULT_COOLDOWN_C  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import PairedGraphArm  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import assert_stable  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import canonical_config_json  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import capture_paired_cuda_graphs  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import file_sha256  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import gpu_info  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import measure_paired_cuda_graphs  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import parse_json_object  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import source_provenance  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import text_sha256  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import validate_pinned_source  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import (  # noqa: E402
    verify_helion_generated_wrapper,
)
import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

HEAD_DIM = 128
L2_EPSILON = 1e-6
LOG2_E = 1.4426950408889634
LOWER_BOUND = -5.0
IMPLEMENTATIONS = ("flashinfer-cute", "helion-cute")
DEFAULT_CASES = (
    "t1-unbounded-b15-h32",
    "t3-lower-bound-n8-h16",
)
PINNED_FLASHINFER_SHA = "f67bc2ed555c1ad6a764ad68f7aa9622178e9eae"


@dataclass(frozen=True)
class Case:
    name: str
    num_sequences: int
    num_tokens: int
    num_heads: int
    lower_bound: float | None
    beta_is_logit: bool

    @property
    def total_tokens(self) -> int:
        return self.num_sequences * self.num_tokens

    @property
    def gate_mode(self) -> str:
        return "unbounded-softplus" if self.lower_bound is None else "lower-bound"


CASES = {
    "t1-unbounded-b15-h32": Case(
        name="t1-unbounded-b15-h32",
        num_sequences=15,
        num_tokens=1,
        num_heads=32,
        lower_bound=None,
        beta_is_logit=True,
    ),
    "t3-lower-bound-n8-h16": Case(
        name="t3-lower-bound-n8-h16",
        num_sequences=8,
        num_tokens=3,
        num_heads=16,
        lower_bound=LOWER_BOUND,
        beta_is_logit=False,
    ),
}


@dataclass
class RecurrentInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    output: torch.Tensor

    def clone_mutable(self) -> RecurrentInputs:
        return RecurrentInputs(
            q=self.q,
            k=self.k,
            v=self.v,
            gate=self.gate,
            beta=self.beta,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            state=self.state.clone(),
            state_indices=self.state_indices,
            cu_seqlens=self.cu_seqlens,
            num_accepted_tokens=self.num_accepted_tokens,
            output=torch.empty_like(self.output),
        )

    def helion_args(self, case: Case) -> tuple[object, ...]:
        return (
            self.q,
            self.k,
            self.v,
            self.gate,
            self.beta,
            self.a_log,
            self.dt_bias,
            self.state,
            self.state_indices,
            self.num_accepted_tokens,
            self.output,
            HEAD_DIM**-0.5,
            0.0 if case.lower_bound is None else case.lower_bound,
            case.num_tokens,
            True,
            case.lower_bound is not None,
            case.beta_is_logit,
        )


def _helion_recurrent_kda_body(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    lower_bound: float,
    num_tokens: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_gate_in_kernel: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    beta_is_logit: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """One-launch fixed-T KDA recurrence over split input tensors."""

    num_sequences = hl.specialize(state_indices.size(0))
    num_heads = hl.specialize(q.size(1))
    num_value_heads = hl.specialize(v.size(1))
    value_heads_per_query_head = num_value_heads // num_heads
    head_dim = hl.specialize(q.size(2))
    value_dim = hl.specialize(v.size(2))
    hl.specialize(
        (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            raw_gate.stride(0),
            raw_gate.stride(1),
            raw_gate.stride(2),
            beta.stride(0),
            beta.stride(1),
            state_pool.stride(0),
            state_pool.stride(1),
            state_pool.stride(2),
            state_pool.stride(3),
            state_indices.stride(0),
            state_indices.stride(1),
            num_accepted_tokens.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
    )

    for tile_n, tile_h, tile_v in hl.tile(
        [num_sequences, num_value_heads, value_dim], block_size=[1, 1, None]
    ):
        i_n = tile_n.id
        i_hv = tile_h.id
        i_h = i_hv // value_heads_per_query_head
        k_offsets = hl.arange(head_dim)
        accepted_index = torch.clamp(
            num_accepted_tokens[i_n].long() - 1,
            min=0,
            max=num_tokens - 1,  # pyrefly: ignore[unsupported-operation]
        )
        initial_slot = state_indices[i_n, accepted_index].long()
        state = state_pool[initial_slot, i_hv, tile_v.index, k_offsets].float()

        for i_t in hl.static_range(num_tokens):  # pyrefly: ignore[bad-argument-type]
            token = i_n * num_tokens + i_t  # pyrefly: ignore[unsupported-operation]
            q_vec = q[token, i_h, k_offsets].float()
            k_vec = k[token, i_h, k_offsets].float()
            q_vec = q_vec * torch.rsqrt((q_vec * q_vec).sum() + L2_EPSILON)
            k_vec = k_vec * torch.rsqrt((k_vec * k_vec).sum() + L2_EPSILON)

            gate_input = raw_gate[token, i_hv, k_offsets].float()
            if use_gate_in_kernel:
                gate_input = gate_input + dt_bias[i_h * head_dim + k_offsets].float()
                decay_parameter = torch.exp2(a_log[i_h].float() * LOG2_E)
                if use_lower_bound:
                    log_decay = lower_bound * torch.sigmoid(
                        decay_parameter * gate_input
                    )
                else:
                    gate_exp = torch.exp2(gate_input * LOG2_E)
                    softplus = torch.where(
                        gate_input <= 20.0,
                        torch.log(1.0 + gate_exp),
                        gate_input,
                    )
                    log_decay = -decay_parameter * softplus
            else:
                log_decay = gate_input
            state = state * torch.exp2(log_decay * LOG2_E)[None, :]

            beta_value = beta[token, i_hv].float()
            if beta_is_logit:
                beta_value = torch.sigmoid(beta_value)
            value = v[token, i_hv, tile_v].float()
            residual = value - (state * k_vec[None, :]).sum(-1)
            state = state + (beta_value * residual)[:, None] * k_vec[None, :]

            projected = (state * q_vec[None, :]).sum(-1) * scale
            output[token, i_hv, tile_v] = projected.to(output.dtype)
            checkpoint_slot = state_indices[i_n, i_t].long()
            state_pool[checkpoint_slot, i_hv, tile_v.index, k_offsets] = state

    return output


def _source_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    harness_path = Path(__file__).resolve()
    helper_path = Path(measure_paired_cuda_graphs.__code__.co_filename).resolve()
    if helion.__file__ is None:
        raise RuntimeError("loaded Helion package has no source path")
    flashinfer_root = args.flashinfer_root.resolve()
    flashinfer_source = (
        flashinfer_root / "flashinfer" / "kda_kernels" / "recurrent_kda.py"
    )
    return {
        "benchmark": {"path": str(harness_path), "sha256": file_sha256(harness_path)},
        "paired_graph_helper": {
            "path": str(helper_path),
            "sha256": file_sha256(helper_path),
        },
        "helion": source_provenance(
            REPO_ROOT,
            {"package_init": Path(helion.__file__).resolve()},
            pathspecs=("helion",),
        ),
        "flashinfer": source_provenance(
            flashinfer_root,
            {"recurrent_kda": flashinfer_source},
            expected_commit=args.flashinfer_sha,
        ),
    }


def _case_semantics(case: Case, seed: int) -> dict[str, Any]:
    accepted_tokens = [
        index % case.num_tokens + 1 for index in range(case.num_sequences)
    ]
    return {
        "case": case.name,
        "seed": seed,
        "shape": {
            "num_sequences": case.num_sequences,
            "num_tokens": case.num_tokens,
            "num_query_heads": case.num_heads,
            "num_value_heads": case.num_heads,
            "head_dim": HEAD_DIM,
        },
        "dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "state_layout": "[checkpoint,value_head,value,key]",
        "state_load": "state_indices[n,clamp(num_accepted_tokens[n]-1,0,T-1)]",
        "state_store": "all-T-checkpoints-in-place",
        "state_index_pattern": (
            "odd-slots" if case.num_tokens == 1 else "contiguous-slots-starting-at-1"
        ),
        "num_accepted_tokens": accepted_tokens,
        "qk_l2norm": True,
        "l2_epsilon": L2_EPSILON,
        "scale": HEAD_DIM**-0.5,
        "gate_mode": case.gate_mode,
        "lower_bound": case.lower_bound,
        "beta_is_logit": case.beta_is_logit,
        "input_distribution": {
            "q_k_v_raw_gate": "normal-std-0.25-bfloat16",
            "beta_logits": "normal-std-1.0-bfloat16",
            "a_log": "uniform[-1.5,0)-float32",
            "dt_bias": "normal-std-0.1-float32",
            "state": "normal-std-0.02-bfloat16",
        },
    }


def _install_namespace(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _load_flashinfer_recurrent(root: Path) -> ModuleType:
    package_root = (root / "flashinfer").resolve()
    source = package_root / "kda_kernels" / "recurrent_kda.py"
    if not source.is_file():
        raise FileNotFoundError(f"not a FlashInfer source checkout: {root}")
    if importlib.util.find_spec("pynvml") is None:
        sys.modules["pynvml"] = ModuleType("pynvml")
    for name, relative in (
        ("flashinfer", ""),
        ("flashinfer.jit", "jit"),
        ("flashinfer.kda_kernels", "kda_kernels"),
        ("flashinfer.cute_dsl", "cute_dsl"),
    ):
        _install_namespace(name, package_root / relative)
    module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    if module.__file__ is None or Path(module.__file__).resolve() != source:
        raise RuntimeError(
            f"expected FlashInfer module {source}, got {module.__file__}"
        )
    return module


def _make_inputs(case: Case, seed: int) -> RecurrentInputs:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(shape: tuple[int, ...], std: float = 0.25) -> torch.Tensor:
        return torch.randn(
            shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(std)

    token_shape = (case.total_tokens, case.num_heads, HEAD_DIM)
    q = normal(token_shape)
    k = normal(token_shape)
    v = normal(token_shape)
    gate = normal(token_shape)
    beta_logits = normal((case.total_tokens, case.num_heads), std=1.0)
    beta = beta_logits if case.beta_is_logit else torch.sigmoid(beta_logits)
    a_log = torch.empty(case.num_heads, device=device, dtype=torch.float32)
    a_log.uniform_(-1.5, 0.0, generator=generator)
    dt_bias = torch.randn(
        case.num_heads * HEAD_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.1)

    if case.num_tokens == 1:
        state_slots = 2 * case.num_sequences + 1
        state_indices = (
            2 * torch.arange(case.num_sequences, device=device, dtype=torch.int32) + 1
        ).reshape(case.num_sequences, 1)
    else:
        state_slots = case.total_tokens + 6
        state_indices = torch.arange(
            1,
            case.total_tokens + 1,
            device=device,
            dtype=torch.int32,
        ).reshape(case.num_sequences, case.num_tokens)
    state = normal((state_slots, case.num_heads, HEAD_DIM, HEAD_DIM), std=0.02)
    cu_seqlens = torch.arange(
        0,
        case.total_tokens + 1,
        case.num_tokens,
        device=device,
        dtype=torch.int32,
    )
    num_accepted_tokens = (
        torch.arange(case.num_sequences, device=device, dtype=torch.int32)
        % case.num_tokens
        + 1
    )
    return RecurrentInputs(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        output=torch.empty_like(v),
    )


def _torch_reference(
    case: Case, inputs: RecurrentInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    q = inputs.q.float()
    k = inputs.k.float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_EPSILON)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_EPSILON)
    gate_input = inputs.gate.float() + inputs.dt_bias.reshape(
        1, case.num_heads, HEAD_DIM
    )
    decay_parameter = torch.exp(inputs.a_log).reshape(1, case.num_heads, 1)
    if case.lower_bound is None:
        log_decay = -decay_parameter * torch.nn.functional.softplus(gate_input)
    else:
        log_decay = case.lower_bound * torch.sigmoid(decay_parameter * gate_input)
    decay = torch.exp(log_decay)
    beta = inputs.beta.float()
    if case.beta_is_logit:
        beta = torch.sigmoid(beta)

    state_indices = inputs.state_indices.cpu().tolist()
    accepted_counts = inputs.num_accepted_tokens.cpu().tolist()
    for sequence, slots in enumerate(state_indices):
        accepted_index = min(max(accepted_counts[sequence] - 1, 0), len(slots) - 1)
        state = inputs.state[slots[accepted_index]].float()
        for local_token, slot in enumerate(slots):
            token = sequence * case.num_tokens + local_token
            state = state * decay[token, :, None, :]
            predicted = (state * k[token, :, None, :]).sum(-1)
            residual = (inputs.v[token].float() - predicted) * beta[token, :, None]
            state = state + residual[:, :, None] * k[token, :, None, :]
            projected = (state * q[token, :, None, :]).sum(-1)
            inputs.output[token] = (projected * HEAD_DIM**-0.5).to(inputs.output.dtype)
            inputs.state[slot] = state.to(inputs.state.dtype)
    return inputs.output, inputs.state


def _flashinfer_kwargs(case: Case, inputs: RecurrentInputs) -> dict[str, object]:
    outer_shape = (1, case.total_tokens)
    qkv_shape = (*outer_shape, case.num_heads, HEAD_DIM)
    beta_shape = (*outer_shape, case.num_heads)
    return {
        "q": inputs.q.view(qkv_shape),
        "k": inputs.k.view(qkv_shape),
        "v": inputs.v.view(qkv_shape),
        "g": inputs.gate.view(qkv_shape),
        "beta": inputs.beta.view(beta_shape),
        "A_log": inputs.a_log,
        "dt_bias": inputs.dt_bias,
        "scale": HEAD_DIM**-0.5,
        "initial_state": inputs.state,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": case.lower_bound,
        "cu_seqlens": inputs.cu_seqlens,
        "ssm_state_indices": (
            inputs.state_indices[:, 0] if case.num_tokens == 1 else inputs.state_indices
        ),
        "num_spec_tokens": None if case.num_tokens == 1 else case.num_tokens - 1,
        "num_accepted_tokens": (
            None if case.num_tokens == 1 else inputs.num_accepted_tokens
        ),
        "output": inputs.output.view(qkv_shape),
        "beta_is_logit": case.beta_is_logit,
    }


def _make_flashinfer_launch(
    module: ModuleType, case: Case, inputs: RecurrentInputs
) -> Callable[[], object]:
    return functools.partial(
        module.run_recurrent_kda,  # pyrefly: ignore[missing-attribute]
        **_flashinfer_kwargs(case, inputs),
        backend="cute-dsl",
    )


def _tracked_flashinfer_cute_call(
    module: ModuleType,
    case: Case,
    inputs: RecurrentInputs,
    launch: Callable[[], object],
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    one_warp_calls: list[dict[str, Any]] = []
    grouped_calls: list[list[int]] = []
    original_one_warp = module._get_compiled_kernel  # pyrefly: ignore[missing-attribute]
    original_grouped = module._get_grouped_compiled  # pyrefly: ignore[missing-attribute]

    def track_one_warp(*compile_args: object, **kwargs: object) -> object:
        one_warp_calls.append(
            {
                "head_dim": compile_args[0],
                "use_qk_l2norm": compile_args[1],
                "use_gate_in_kernel": compile_args[2],
                "has_dt_bias": compile_args[3],
                "use_lower_bound": compile_args[4],
                "use_cu_seqlens": compile_args[5],
                "has_initial_state_source": compile_args[6],
                "beta_is_logit": compile_args[7],
                "num_tokens": compile_args[8],
                "tile_rows": compile_args[9],
                "dot_reduction_schedule": compile_args[10],
                "zero_padded_output": compile_args[11],
                "has_num_accepted_tokens": compile_args[12],
            }
        )
        return original_one_warp(*compile_args, **kwargs)

    def track_grouped(*compile_args: object, **kwargs: object) -> object:
        grouped_key = compile_args[0]
        if not isinstance(grouped_key, tuple):
            raise AssertionError(f"unexpected grouped compile key: {grouped_key!r}")
        grouped_calls.append([int(value) for value in grouped_key])
        return original_grouped(*compile_args, **kwargs)

    module._get_compiled_kernel = track_one_warp  # pyrefly: ignore[missing-attribute]
    module._get_grouped_compiled = track_grouped  # pyrefly: ignore[missing-attribute]
    try:
        launch()
    finally:
        module._get_compiled_kernel = original_one_warp  # pyrefly: ignore[missing-attribute]
        module._get_grouped_compiled = original_grouped  # pyrefly: ignore[missing-attribute]

    if case.num_tokens == 1:
        expected_key = {
            "head_dim": HEAD_DIM,
            "use_qk_l2norm": 1,
            "use_gate_in_kernel": 1,
            "has_dt_bias": 1,
            "use_lower_bound": 0,
            "use_cu_seqlens": 1,
            "has_initial_state_source": 0,
            "beta_is_logit": 1,
            "num_tokens": 1,
            "tile_rows": 16,
            "dot_reduction_schedule": 1,
            "zero_padded_output": 1,
            "has_num_accepted_tokens": 0,
        }
        if one_warp_calls != [expected_key] or grouped_calls:
            raise AssertionError(
                f"{case.name} expected one packed tile16/dual-accumulator "
                f"FlashInfer CuTe launch, got one_warp={one_warp_calls}, "
                f"grouped={grouped_calls}"
            )
        schedule: dict[str, Any] = {
            "route": "one-warp",
            "compile_key": expected_key,
        }
    else:
        expected_values = [
            HEAD_DIM,
            case.num_tokens,
            case.num_heads,
            case.num_heads,
            2,
            1,
            int(case.beta_is_logit),
            1,
            0,
        ]
        if one_warp_calls or grouped_calls != [expected_values]:
            raise AssertionError(
                f"{case.name} expected one grouped-CTA FlashInfer CuTe launch "
                f"with key {expected_values}, got one_warp={one_warp_calls}, "
                f"grouped={grouped_calls}"
            )
        key_names = (
            "head_dim",
            "num_tokens",
            "num_heads",
            "num_value_heads",
            "gate_mode",
            "has_dt_bias",
            "beta_is_logit",
            "use_qk_l2norm",
            "has_initial_state_source",
        )
        schedule = {
            "route": "grouped-cta",
            "compile_key": dict(zip(key_names, expected_values, strict=True)),
            "compile_key_values": expected_values,
        }
    return (inputs.output, inputs.state), schedule


def _check_result(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    state_indices: torch.Tensor,
    pristine_state: torch.Tensor,
) -> dict[str, float | int | bool]:
    output, state = actual
    expected_output, expected_state = expected
    output_max_abs = float((output.float() - expected_output.float()).abs().max())
    active_slots = torch.unique(state_indices[state_indices >= 0].long())
    active_state = state.index_select(0, active_slots)
    expected_active_state = expected_state.index_select(0, active_slots)
    state_max_abs = float(
        (active_state.float() - expected_active_state.float()).abs().max()
    )
    torch.testing.assert_close(output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        active_state,
        expected_active_state,
        atol=1e-2,
        rtol=1e-2,
    )
    untouched_mask = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    untouched_mask[active_slots] = False
    untouched_equal = torch.equal(state[untouched_mask], pristine_state[untouched_mask])
    if not untouched_equal:
        raise AssertionError("implementation modified an untouched checkpoint slot")
    return {
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "active_state_max_abs": state_max_abs,
        "active_checkpoint_slots": active_slots.numel(),
        "untouched_checkpoint_slots": untouched_mask.sum().item(),
        "untouched_slots_bitwise_equal": untouched_equal,
        "atol": 1e-2,
        "rtol": 1e-2,
    }


def _effective_bytes(case: Case) -> int:
    state_elements = (
        (case.num_tokens + 1)
        * case.num_sequences
        * case.num_heads
        * HEAD_DIM
        * HEAD_DIM
    )
    token_elements = case.total_tokens * (
        case.num_heads * 2 * HEAD_DIM + case.num_heads * (3 * HEAD_DIM + 1)
    )
    parameter_bytes = (case.num_heads + case.num_heads * HEAD_DIM) * 4
    index_bytes = (
        case.num_sequences * case.num_tokens
        + case.num_sequences
        + case.num_sequences
        + 1
    ) * 4
    return 2 * (state_elements + token_elements) + parameter_bytes + index_bytes


def _make_helion_kernel(args: argparse.Namespace) -> helion.Kernel:
    return helion.kernel(
        _helion_recurrent_kda_body,
        static_shapes=False,
        autotune_effort="full",
        autotune_random_seed=args.autotune_seed,
        ignore_warnings=[helion.exc.ProcessGroupNameNotFound],
    )


def _helion_codegen_expectation(case: Case) -> tuple[str, tuple[str, ...]]:
    if case.num_tokens == 1:
        return (
            "split-single-token-rank1",
            (
                "split_t1_codegen_abi_version = 1",
                "split_t1_rank1_helper_abi_version = 4",
            ),
        )
    return (
        "fixed-token-rank1",
        (
            "fixed_rank1_codegen_abi_version = 3",
            "fixed_rank1_rank1_helper_abi_version = 4",
        ),
    )


def _load_helion_config_map(path: Path) -> dict[str, str]:
    raw = parse_json_object(path.read_text(encoding="utf-8"), "--helion-config-map")
    unknown = sorted(set(raw) - set(CASES))
    if unknown:
        raise ValueError(f"unknown cases in --helion-config-map: {unknown}")
    normalized: dict[str, str] = {}
    for case_name, config in raw.items():
        if isinstance(config, str):
            normalized[case_name] = canonical_config_json(
                config, f"--helion-config-map[{case_name!r}]"
            )
        elif isinstance(config, dict):
            normalized[case_name] = json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        else:
            raise ValueError(f"config for {case_name} must be a JSON object")
    return normalized


def _resolve_helion_config(
    args: argparse.Namespace, case_name: str
) -> tuple[str, str, str]:
    if args.helion_config_map is not None:
        configs = _load_helion_config_map(args.helion_config_map)
        if case_name not in configs:
            raise ValueError(f"--helion-config-map has no config for {case_name}")
        source_hash = file_sha256(args.helion_config_map)
        return (
            configs[case_name],
            f"{args.helion_config_map.resolve()}#{case_name}",
            source_hash,
        )
    if args.helion_config is None:
        raise ValueError("a pinned Helion config is required")
    return (
        canonical_config_json(args.helion_config, "--helion-config"),
        args.helion_config_source or "inline-cli",
        text_sha256(args.helion_config),
    )


def _paired_input_ownership(
    pristine: RecurrentInputs, arms: dict[str, RecurrentInputs]
) -> dict[str, Any]:
    immutable_fields = (
        "q",
        "k",
        "v",
        "gate",
        "beta",
        "a_log",
        "dt_bias",
        "state_indices",
        "cu_seqlens",
        "num_accepted_tokens",
    )
    shared = {
        field: all(
            getattr(inputs, field) is getattr(pristine, field)
            for inputs in arms.values()
        )
        for field in immutable_fields
    }
    if not all(shared.values()):
        raise RuntimeError(f"paired immutable inputs are not shared: {shared}")
    state_pointers = {
        "pristine": pristine.state.data_ptr(),
        **{name: inputs.state.data_ptr() for name, inputs in arms.items()},
    }
    output_pointers = {
        "pristine": pristine.output.data_ptr(),
        **{name: inputs.output.data_ptr() for name, inputs in arms.items()},
    }
    if len(set(state_pointers.values())) != len(state_pointers):
        raise RuntimeError(f"paired state buffers alias: {state_pointers}")
    if len(set(output_pointers.values())) != len(output_pointers):
        raise RuntimeError(f"paired output buffers alias: {output_pointers}")
    return {
        "shared_immutable_fields": shared,
        "private_state_buffers": True,
        "private_output_buffers": True,
        "state_pointers": state_pointers,
        "output_pointers": output_pointers,
    }


def _cross_provider_comparison(
    case: Case, baseline: RecurrentInputs, candidate: RecurrentInputs
) -> dict[str, float | int | bool]:
    active_slots = torch.unique(
        baseline.state_indices[baseline.state_indices >= 0].long()
    )
    baseline_state = baseline.state.index_select(0, active_slots)
    candidate_state = candidate.state.index_select(0, active_slots)
    return {
        "output_max_abs": float(
            (baseline.output.float() - candidate.output.float()).abs().max()
        ),
        "active_state_max_abs": float(
            (baseline_state.float() - candidate_state.float()).abs().max()
        ),
        "output_exact": bool(torch.equal(baseline.output, candidate.output)),
        "active_state_exact": bool(torch.equal(baseline_state, candidate_state)),
        "active_checkpoint_slots": active_slots.numel(),
        "num_tokens": case.num_tokens,
    }


def _run_paired_recurrent_graph(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("HELION_BACKEND") != "cute":
        raise RuntimeError("set HELION_BACKEND=cute for Helion graph timing")
    case = CASES[args.case]
    config_json, config_source, config_source_sha256 = _resolve_helion_config(
        args, case.name
    )
    source_before = _source_fingerprint(args)
    validate_pinned_source("FlashInfer", source_before["flashinfer"])

    pristine = _make_inputs(case, args.seed)
    expected = _torch_reference(case, pristine.clone_mutable())
    arm_inputs = {name: pristine.clone_mutable() for name in IMPLEMENTATIONS}
    ownership = _paired_input_ownership(pristine, arm_inputs)

    flashinfer_module = _load_flashinfer_recurrent(args.flashinfer_root.resolve())
    flashinfer_inputs = arm_inputs["flashinfer-cute"]
    flashinfer_launch = _make_flashinfer_launch(
        flashinfer_module, case, flashinfer_inputs
    )

    helion_inputs = arm_inputs["helion-cute"]
    helion_args = helion_inputs.helion_args(case)
    helion_bound = _make_helion_kernel(args).bind(helion_args)
    requested_config = helion.Config(
        **parse_json_object(config_json, "pinned Helion config")
    )
    config = helion_bound.config_spec.normalized_config(requested_config)
    helion_bound.set_config(config)
    resolved_config = canonical_config_json(config.to_json(), "resolved Helion config")

    def launch_helion(
        bound_kernel: Callable[..., object] = cast(
            "Callable[..., object]", helion_bound
        ),
        bound_args: tuple[object, ...] = helion_args,
    ) -> object:
        return bound_kernel(*bound_args)

    def reset_flashinfer() -> None:
        flashinfer_inputs.state.copy_(pristine.state)
        flashinfer_inputs.output.zero_()

    def reset_helion() -> None:
        helion_inputs.state.copy_(pristine.state)
        helion_inputs.output.zero_()

    arms = (
        PairedGraphArm("flashinfer-cute", flashinfer_launch, reset_flashinfer),
        PairedGraphArm("helion-cute", launch_helion, reset_helion),
    )

    eager_correctness: dict[str, Any] = {}
    reset_flashinfer()
    flashinfer_actual, schedule = _tracked_flashinfer_cute_call(
        flashinfer_module,
        case,
        flashinfer_inputs,
        flashinfer_launch,
    )
    torch.cuda.synchronize()
    eager_correctness["flashinfer-cute"] = _check_result(
        flashinfer_actual,
        expected,
        pristine.state_indices,
        pristine.state,
    )
    reset_helion()
    launch_helion()
    torch.cuda.synchronize()
    eager_correctness["helion-cute"] = _check_result(
        (helion_inputs.output, helion_inputs.state),
        expected,
        pristine.state_indices,
        pristine.state,
    )

    plan_kind, source_markers = _helion_codegen_expectation(case)
    generated_wrapper = verify_helion_generated_wrapper(
        helion_bound,
        config,
        expected_plan_kind=plan_kind,
        expected_source_markers=source_markers,
    )
    provider_metadata = {
        "flashinfer-cute": {
            "backend": "flashinfer-cute-dsl-recurrent",
            "schedule": schedule,
            "provenance": source_before["flashinfer"],
        },
        "helion-cute": {
            "backend": "helion-cute",
            "requested_config": json.loads(config_json),
            "config": json.loads(resolved_config),
            "config_sha256": text_sha256(resolved_config),
            "config_source": config_source,
            "config_source_sha256": config_source_sha256,
            "generated_wrapper": generated_wrapper,
        },
    }

    stream = torch.cuda.Stream()
    captured = capture_paired_cuda_graphs(arms, stream)
    graph_correctness: dict[str, Any] = {}
    for arm in arms:
        with torch.cuda.stream(stream):
            arm.reset()
            captured[arm.name].graph.replay()
        stream.synchronize()
        inputs = arm_inputs[arm.name]
        graph_correctness[arm.name] = _check_result(
            (inputs.output, inputs.state),
            expected,
            pristine.state_indices,
            pristine.state,
        )
    cross_provider_correctness = _cross_provider_comparison(
        case, flashinfer_inputs, helion_inputs
    )

    timing = measure_paired_cuda_graphs(
        captured,
        baseline="flashinfer-cute",
        candidate="helion-cute",
        measurement_cycles=args.graph_measurement_cycles,
        primer_cycles=args.graph_primer_cycles,
        cooldown_temp_c=args.cooldown_temp_c,
        cooldown_timeout_s=args.cooldown_timeout_s,
        stream=stream,
    )
    effective_bytes = _effective_bytes(case)
    for values in timing["implementations"].values():
        values["effective_gbps"] = effective_bytes / (values["median_ms"] * 1e6)

    source_after = _source_fingerprint(args)
    assert_stable(
        "source fingerprint during paired graph run", source_before, source_after
    )
    if _resolve_helion_config(args, case.name) != (
        config_json,
        config_source,
        config_source_sha256,
    ):
        raise RuntimeError("Helion config input changed during paired graph timing")
    final_config = getattr(helion_bound, "_config", None)
    if (
        final_config is None
        or canonical_config_json(final_config.to_json(), "final Helion config")
        != resolved_config
    ):
        raise RuntimeError("Helion config changed during paired graph timing")
    final_wrapper = verify_helion_generated_wrapper(
        helion_bound,
        final_config,
        expected_plan_kind=plan_kind,
        expected_source_markers=source_markers,
    )
    assert_stable(
        "Helion generated wrapper during paired graph run",
        generated_wrapper,
        final_wrapper,
    )

    return {
        "valid": True,
        "case": case.name,
        "implementations_order": list(IMPLEMENTATIONS),
        "semantic_metadata": _case_semantics(case, args.seed),
        "effective_bytes": effective_bytes,
        "timed_state_semantics": "pristine-reset-before-each-cuda-graph-replay",
        "correctness_state_semantics": "identical-pristine-clone-one-replay",
        "correctness": {
            "eager_vs_oracle": eager_correctness,
            "graph_replay_vs_oracle": graph_correctness,
            "cross_provider_after_one_graph": cross_provider_correctness,
        },
        "fairness_audit": {
            "same_process": True,
            "same_cuda_stream": True,
            "same_cuda_event_timer": True,
            "one_provider_invocation_per_graph": True,
            "one_graph_replay_per_timed_interval": True,
            "reset_outside_timing": True,
            "cold_l2_before_every_replay": True,
            "private_mutable_state_per_arm": True,
            "private_output_per_arm": True,
            "shared_immutable_inputs": True,
            "balanced_position_schedule": True,
            "outlier_policy": "none discarded",
        },
        "input_ownership": ownership,
        "graph_capture": {
            "graphs": len(captured),
            "provider_invocations_per_graph": 1,
            "graph_objects_kept_alive": all(
                isinstance(item.graph, torch.cuda.CUDAGraph)
                for item in captured.values()
            ),
        },
        "providers": provider_metadata,
        "source_fingerprint_before": source_before,
        "source_fingerprint_after": source_after,
        "environment": {
            "helion_backend": os.environ.get("HELION_BACKEND"),
            "helion_fast_math": os.environ.get("HELION_FAST_MATH"),
            "helion_skip_cache": os.environ.get("HELION_SKIP_CACHE"),
            "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "seed": args.seed,
        "argv": sys.argv,
        "gpu": gpu_info(),
        **timing,
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-cuda-graph", action="store_true")
    parser.add_argument("--impls", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--case", choices=DEFAULT_CASES, required=True)
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--flashinfer-sha", default=PINNED_FLASHINFER_SHA)
    parser.add_argument("--autotune-seed", type=int, default=1337)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--helion-config")
    config_group.add_argument(
        "--helion-config-map",
        "--helion-configs",
        dest="helion_config_map",
        type=Path,
    )
    parser.add_argument("--helion-config-source")
    parser.add_argument("--graph-primer-cycles", type=int, default=2)
    parser.add_argument("--graph-measurement-cycles", type=int, default=12)
    parser.add_argument("--cache-mode", choices=("cold",), default="cold")
    parser.add_argument("--cooldown-temp-c", type=float, default=DEFAULT_COOLDOWN_C)
    parser.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--jsonl", type=Path)
    return parser


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    implementations = tuple(value.strip() for value in args.impls.split(","))
    if implementations != IMPLEMENTATIONS:
        parser.error(f"--impls must be exactly {','.join(IMPLEMENTATIONS)}")
    if not args.paired_cuda_graph:
        parser.error("--paired-cuda-graph is required; legacy timing was removed")
    if args.graph_primer_cycles <= 0:
        parser.error("--graph-primer-cycles must be positive")
    if args.graph_measurement_cycles < 2 or args.graph_measurement_cycles % 2:
        parser.error("--graph-measurement-cycles must be an even integer >= 2")
    if args.cooldown_timeout_s <= 0:
        parser.error("--cooldown-timeout-s must be positive")
    if not math.isfinite(args.cooldown_temp_c):
        parser.error("--cooldown-temp-c must be finite")
    if args.helion_config_source is not None and args.helion_config is None:
        parser.error("--helion-config-source requires --helion-config")
    try:
        _resolve_helion_config(args, args.case)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    _validate_cli_args(parser, args)
    result = _run_paired_recurrent_graph(args)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("x", encoding="utf-8") as output_file:
            output_file.write(json.dumps(result) + "\n")
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
